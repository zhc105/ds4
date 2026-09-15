/* Timing of the Flash-Next FP4 expert GEMM (cuda/mmq/ds4_qwen_fp4.cu) at
 * the shapes a prefill chunk gives it: 512 experts, 10 per token, a chunk
 * of ROWS tokens, the gate/up projection (K 2560, M 640, per-token rows),
 * the down projection (K 640, M 2560, per-slot rows) and the fused gate+up
 * pass that quantises the down input in its epilogue.  Random routing,
 * random operands; the kernel's cost does not depend on the values.  A
 * device-to-device memcpy goes first as the card's achievable streaming
 * rate.  Built by `make tests/qwen_fp4_bench`. */
#include "cuda/mmq/ds4_qwen_fp4.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_EXPERT = 512, N_USED = 10, ROWS = 4096, TILE = DS4_QWEN_FP4_TILE_ROWS, SLOTS = ROWS * N_USED };

static uint32_t lcg(uint32_t *s) { *s = *s * 1103515245u + 12345u; return *s >> 8; }

static void *dev_copy(const void *src, size_t bytes) {
    void *d = NULL;
    if (cudaMalloc(&d, bytes) != cudaSuccess) return NULL;
    if (src && cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    return d;
}

static int run(const char *name, int K, int M, int x_per_slot, int gate_up) {
    const int n_super = K / 64;
    uint32_t seed = 11u + (uint32_t)K;
    const size_t w_bytes = (size_t)N_EXPERT * M * n_super * 36;
    uint8_t *w = malloc(w_bytes);
    for (size_t i = 0; i < w_bytes; i++) {
        const size_t in_blk = i % 36;
        w[i] = in_blk < 4 ? (uint8_t)(0x28 + (lcg(&seed) % 16)) : (uint8_t)(lcg(&seed) & 0xff);
    }
    float *scales = malloc(N_EXPERT * sizeof(float));
    for (int e = 0; e < N_EXPERT; e++) scales[e] = 0.5f + 0.001f * e;
    const size_t x_rows = x_per_slot ? SLOTS : ROWS;
    const size_t xq_bytes = x_rows * n_super * 36;
    uint8_t *xq = malloc(xq_bytes);
    for (size_t i = 0; i < xq_bytes; i++) {
        const size_t in_blk = i % 36;
        xq[i] = in_blk < 4 ? (uint8_t)(0x28 + (lcg(&seed) % 16)) : (uint8_t)(lcg(&seed) & 0xff);
    }
    int32_t *esel = malloc(SLOTS * sizeof(int32_t));
    for (int s = 0; s < SLOTS; s++) esel[s] = (int32_t)(lcg(&seed) % N_EXPERT);
    uint32_t (*plan)[N_EXPERT + 1] = calloc(4, sizeof(*plan));
    for (int s = 0; s < SLOTS; s++) plan[0][esel[s]]++;
    for (int e = 0; e < N_EXPERT; e++) {
        plan[1][e + 1] = plan[1][e] + plan[0][e];
        plan[2][e + 1] = plan[2][e] + (plan[0][e] + TILE - 1) / TILE;
        plan[3][e] = plan[1][e];
    }
    int32_t *order = malloc(SLOTS * sizeof(int32_t));
    for (int s = 0; s < SLOTS; s++) order[plan[3][esel[s]]++] = s;

    void *dw = dev_copy(w, w_bytes), *dsc = dev_copy(scales, N_EXPERT * sizeof(float));
    void *dxq = dev_copy(xq, xq_bytes), *dord = dev_copy(order, SLOTS * sizeof(int32_t));
    void *dplan = dev_copy(plan, 4 * sizeof(*plan));
    void *dout = dev_copy(NULL, gate_up ? (size_t)SLOTS * (M / 64) * 36 : (size_t)SLOTS * M * sizeof(uint16_t));
    void *dw2 = gate_up ? dev_copy(w, w_bytes) : NULL;   /* the up experts, their own rows in memory */
    if (!dw || !dsc || !dxq || !dord || !dplan || !dout || (gate_up && !dw2)) {
        fprintf(stderr, "fp4-bench: cudaMalloc failed\n");
        return 1;
    }
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const int reps = 20;
    for (int i = 0; i < 3 + reps; i++) {
        if (i == 3) cudaEventRecord(t0, 0);
        const int rc = gate_up
            ? ds4_qwen_fp4_moe_gate_up(dw, dsc, dw2, dsc, dxq, dord, dplan, N_EXPERT, N_USED, K, M, ROWS, dout, 0)
            : ds4_qwen_fp4_moe_gemm(dw, dsc, dxq, x_per_slot, dord, dplan, N_EXPERT, N_USED, K, M, ROWS, dout, 1, 0);
        if (rc != 0) {
            fprintf(stderr, "fp4-bench: launch failed\n");
            return 1;
        }
    }
    cudaEventRecord(t1, 0);
    if (cudaEventSynchronize(t1) != cudaSuccess) {
        fprintf(stderr, "fp4-bench: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= (float)reps;
    const double flop = 2.0 * SLOTS * (double)K * M * (gate_up ? 2.0 : 1.0);
    const double bytes = (double)w_bytes * (gate_up ? 2.0 : 1.0) + (double)xq_bytes + (gate_up ? (double)SLOTS * (M / 64) * 36.0 : (double)SLOTS * M * 2.0);
    printf("fp4-bench: %-5s K %4d M %4d: %.3f ms  %.1f TFLOP/s  %.0f GB/s over weights+activations+outputs (%.2f GB)\n",
           name, K, M, ms, flop / ms / 1e9, bytes / ms / 1e6, bytes / 1e9);
    cudaFree(dw); cudaFree(dsc); cudaFree(dxq); cudaFree(dord); cudaFree(dplan); cudaFree(dout); cudaFree(dw2);
    free(w); free(scales); free(xq); free(esel); free(plan); free(order);
    return 0;
}

static void memcpy_ref(void) {
    const size_t n = 472u << 20;
    void *a = dev_copy(NULL, n), *b = dev_copy(NULL, n);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice);
    cudaEventRecord(t0, 0);
    for (int i = 0; i < 10; i++) cudaMemcpy(b, a, n, cudaMemcpyDeviceToDevice);
    cudaEventRecord(t1, 0); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1); ms /= 10;
    printf("fp4-bench: d2d memcpy %.0f MB: %.3f ms  %.0f GB/s read+write\n", n / 1e6, ms, 2.0 * n / ms / 1e6);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("fp4-bench: %s: %d SMs, L2 %d MB, smem/SM %zu KB, smem/block opt-in %zu KB\n", prop.name,
           prop.multiProcessorCount, prop.l2CacheSize >> 20, prop.sharedMemPerMultiprocessor >> 10, prop.sharedMemPerBlockOptin >> 10);
    cudaFree(a); cudaFree(b);
}

int main(void) {
    memcpy_ref();
    return run("gate", 2560, 640, 0, 0) || run("down", 640, 2560, 1, 0) || run("gate+up", 2560, 640, 0, 1);
}
