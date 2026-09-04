/* Exact check of the Flash-Next FP4 expert GEMM (cuda/mmq/ds4_qwen_fp4.cu)
 * against a host reference: random NVFP4 experts, activations quantised by
 * the GPU and dequantised here, a hand-built expert plan, and the products
 * summed in double.  Run over the down projection's K (one whole-row step)
 * and the gate/up K (several pipelined steps).  Built by `make
 * cuda-regression`. */
#include "cuda/mmq/ds4_qwen_fp4.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_EXPERT = 5, N_USED = 3, ROWS = 45, TILE = 32, SLOTS = ROWS * N_USED };

static const float e2m1[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

static float ue4m3(uint8_t bits) {
    bits &= 0x7f;
    if (bits == 0x7f) return 0.0f;
    const int e = bits >> 3;
    const float m = (float)(bits & 7);
    return e == 0 ? ldexpf(m, -9) : ldexpf(1.0f + m * 0.125f, e - 7);
}

/* value k (0..63) of one 36-byte super-block: byte b of a 16-value
 * sub-block holds values b (low nibble) and b+8 (high) */
static float block_value(const uint8_t *blk, int k) {
    const int sub = k / 16, j = k % 16;
    const uint8_t byte = blk[4 + sub * 8 + (j % 8)];
    const uint8_t nib = j < 8 ? (byte & 15) : (byte >> 4);
    const float v = e2m1[nib & 7] * ((nib & 8) ? -1.0f : 1.0f);
    return v * ue4m3(blk[sub]);
}

static uint32_t lcg(uint32_t *s) { *s = *s * 1103515245u + 12345u; return *s >> 8; }

static void *dev_copy(const void *src, size_t bytes) {
    void *d = NULL;
    if (cudaMalloc(&d, bytes) != cudaSuccess) return NULL;
    if (src && cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    return d;
}

static int run(int K, int M) {
    const int n_super = K / 64;
    uint32_t seed = 7u + (uint32_t)K;
    /* random experts: valid UE4M3 scales (small exponents) and random nibbles */
    const size_t w_bytes = (size_t)N_EXPERT * M * n_super * 36;
    uint8_t *w = malloc(w_bytes);
    for (size_t i = 0; i < w_bytes; i++) {
        const size_t in_blk = i % 36;
        w[i] = in_blk < 4 ? (uint8_t)(0x28 + (lcg(&seed) % 16)) : (uint8_t)(lcg(&seed) & 0xff);
    }
    float scales[N_EXPERT];
    for (int e = 0; e < N_EXPERT; e++) scales[e] = 0.5f + 0.1f * e;
    float *x = malloc((size_t)ROWS * K * sizeof(float));
    for (size_t i = 0; i < (size_t)ROWS * K; i++) x[i] = ((float)(lcg(&seed) % 2001) - 1000.0f) / 250.0f;
    /* routing and the plan: counts, starts, tile starts, cursors */
    int32_t esel[SLOTS];
    for (int s = 0; s < SLOTS; s++) esel[s] = (int32_t)(lcg(&seed) % N_EXPERT);
    uint32_t plan[4][N_EXPERT + 1];
    memset(plan, 0, sizeof plan);
    for (int s = 0; s < SLOTS; s++) plan[0][esel[s]]++;
    for (int e = 0; e < N_EXPERT; e++) {
        plan[1][e + 1] = plan[1][e] + plan[0][e];
        plan[2][e + 1] = plan[2][e] + (plan[0][e] + TILE - 1) / TILE;
        plan[3][e] = plan[1][e];
    }
    int32_t order[SLOTS];
    for (int s = 0; s < SLOTS; s++) order[plan[3][esel[s]]++] = s;

    const size_t xq_bytes = (size_t)ROWS * n_super * 36;
    void *dw = dev_copy(w, w_bytes), *dsc = dev_copy(scales, sizeof scales);
    void *dx = dev_copy(x, (size_t)ROWS * K * sizeof(float)), *dxq = dev_copy(NULL, xq_bytes);
    void *dord = dev_copy(order, sizeof order), *dplan = dev_copy(plan, sizeof plan);
    void *dout = dev_copy(NULL, (size_t)SLOTS * M * sizeof(float));
    if (!dw || !dsc || !dx || !dxq || !dord || !dplan || !dout) return 1;
    if (ds4_qwen_fp4_quantize(dx, NULL, dxq, ROWS, K, 0) != 0 ||
        ds4_qwen_fp4_moe_gemm(dw, dsc, dxq, 0, dord, dplan, N_EXPERT, N_USED, K, M, ROWS, dout, 0) != 0 ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "fp4-test: launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    uint8_t *xq = malloc(xq_bytes);
    float *out = malloc((size_t)SLOTS * M * sizeof(float));
    if (cudaMemcpy(xq, dxq, xq_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(out, dout, (size_t)SLOTS * M * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;

    /* the quantiser must keep every value near its E2M1 grid point */
    double qerr = 0.0;
    for (int r = 0; r < ROWS; r++) {
        for (int k = 0; k < K; k++) {
            const float v = block_value(xq + ((size_t)r * n_super + k / 64) * 36, k % 64);
            qerr = fmax(qerr, fabs(v - x[(size_t)r * K + k]));
        }
    }
    double maxrel = 0.0;
    int bad = 0;
    for (int s = 0; s < SLOTS; s++) {
        const int e = esel[s], r = s / N_USED;
        for (int c = 0; c < M; c++) {
            double ref = 0.0;
            for (int k = 0; k < K; k++) {
                const float a = block_value(xq + ((size_t)r * n_super + k / 64) * 36, k % 64);
                const float b = block_value(w + (((size_t)e * M + c) * n_super + k / 64) * 36, k % 64);
                ref += (double)a * b;
            }
            ref *= scales[e];
            const double got = out[(size_t)s * M + c];
            const double rel = fabs(got - ref) / (fabs(ref) + 1e-3);
            if (rel > maxrel) maxrel = rel;
            if (rel > 1e-4 && bad++ < 5) {
                fprintf(stderr, "fp4-test: K %d slot %d (expert %d) col %d: got %.6g want %.6g\n", K, s, e, c, got, ref);
            }
        }
    }
    printf("fp4-test: K %d M %d: quantise max abs err %.3g (values up to 4), gemm max rel err %.3g, %d bad of %d\n",
           K, M, qerr, maxrel, bad, SLOTS * M);
    cudaFree(dw); cudaFree(dsc); cudaFree(dx); cudaFree(dxq); cudaFree(dord); cudaFree(dplan); cudaFree(dout);
    free(w); free(x); free(xq); free(out);
    return bad == 0 && qerr < 1.0 ? 0 : 1;
}

int main(void) {
    const int ok = run(640, 200) == 0 && run(2560, 136) == 0;
    printf("fp4-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
