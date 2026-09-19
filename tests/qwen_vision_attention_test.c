/* The Qwen vision tower's attention against a CPU reference in double:
 * softmax(q k^T / sqrt(72)) v per head over every patch, the tensors laid out
 * [patch][head][72] as the tower holds them.  Row counts cover one chunk of
 * queries, a count the softmax threads do not divide, and enough patches that
 * the scores are chunked (a 1920x1080 screenshot has 8160), where every 97th
 * query is checked.  The scores are peaked like a trained tower's so the softmax is
 * exercised away from uniform.  Built by `make cuda-regression`. */
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HEADS 16u
#define D 72u
#define DIM (HEADS * D)

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float unit(uint64_t *state) {
    *state = *state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((*state >> 40) & 0xffff) / 32768.0f - 1.0f;
}

/* The largest error of `got` against the reference, relative to the
 * reference's largest value. */
static double reference_error(const float *got, const float *q, const float *k, const float *v,
                              uint32_t rows, uint32_t every) {
    double *score = malloc((size_t)rows * sizeof(double));
    double worst = 0.0, scale = 0.0;
    for (uint32_t r = 0; r < rows; r += every) {
        for (uint32_t h = 0; h < HEADS; h++) {
            const float *qr = q + ((size_t)r * HEADS + h) * D;
            double top = -INFINITY, sum = 0.0;
            for (uint32_t j = 0; j < rows; j++) {
                const float *kj = k + ((size_t)j * HEADS + h) * D;
                double dot = 0.0;
                for (uint32_t i = 0; i < D; i++) dot += (double)qr[i] * kj[i];
                score[j] = dot / sqrt((double)D);
                if (score[j] > top) top = score[j];
            }
            for (uint32_t j = 0; j < rows; j++) sum += score[j] = exp(score[j] - top);
            for (uint32_t i = 0; i < D; i++) {
                double want = 0.0;
                for (uint32_t j = 0; j < rows; j++) want += score[j] * v[((size_t)j * HEADS + h) * D + i];
                want /= sum;
                const double err = fabs(want - got[((size_t)r * HEADS + h) * D + i]);
                if (err > worst) worst = err;
                if (fabs(want) > scale) scale = fabs(want);
            }
        }
    }
    free(score);
    return worst / scale;
}

static int run(uint32_t rows, uint32_t every) {
    const size_t n = (size_t)rows * DIM;
    float *q = malloc(n * sizeof(float)), *k = malloc(n * sizeof(float));
    float *v = malloc(n * sizeof(float)), *got = malloc(n * sizeof(float));
    uint64_t seed = 0x9e3779b97f4a7c15ull + rows;
    for (size_t i = 0; i < n; i++) {
        q[i] = 3.0f * unit(&seed);
        k[i] = unit(&seed);
        v[i] = unit(&seed);
    }
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(n * sizeof(float)), *tk = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *tv = ds4_gpu_tensor_alloc(n * sizeof(float)), *to = ds4_gpu_tensor_alloc(n * sizeof(float));
    int ok = tq && tk && tv && to &&
             ds4_gpu_tensor_write(tq, 0, q, n * sizeof(float)) &&
             ds4_gpu_tensor_write(tk, 0, k, n * sizeof(float)) &&
             ds4_gpu_tensor_write(tv, 0, v, n * sizeof(float));
    double ms = 0.0;
    for (int pass = 0; ok && pass < 2; pass++) {   /* the first pass pays cuBLAS's setup */
        const double t0 = now_sec();
        ok = ds4_gpu_qwen_vision_attention(to, tq, tk, tv, rows) && ds4_gpu_synchronize();
        ms = (now_sec() - t0) * 1000.0;
    }
    if (ok) ok = ds4_gpu_tensor_read(to, 0, got, n * sizeof(float)) != 0;
    if (!ok) {
        fprintf(stderr, "rows %u: the attention failed\n", rows);
    } else {
        const double err = reference_error(got, q, k, v, rows, every);
        printf("rows %5u: %8.1f ms, error %.2e of the largest value\n", rows, ms, err);
        if (!(err < 2e-5)) {   /* f32 rounding is 3e-6 here; TF32 products give 4e-4 */
            fprintf(stderr, "rows %u: differs from the reference\n", rows);
            ok = 0;
        }
    }
    ds4_gpu_tensor_free(to);
    ds4_gpu_tensor_free(tv);
    ds4_gpu_tensor_free(tk);
    ds4_gpu_tensor_free(tq);
    free(got);
    free(v);
    free(k);
    free(q);
    return ok;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    const int ok = run(4u, 1u) && run(320u, 1u) && run(1001u, 1u) && run(8160u, 97u);
    printf("qwen vision attention: %s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
