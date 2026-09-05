/* Exact check of the Flash-Next QSA block selection kernel against the CPU
 * rule in qwen_attention_cells: score every completed block by the sum over
 * indexer heads of relu(q . k), keep the budget best with ties to the older
 * block, emit the cells in position order, then the tail cells.  Queries and
 * block keys are drawn from multiples of 1/2 so every score is exact in f32
 * on both sides and ties are real.  Built by `make cuda-regression`. */
#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_HEAD = 4, D = 128, R = 4, BUDGET = 512, CTX = 16384, ROWS = 128 };
enum { MAX_SEL = BUDGET * R + R - 1 };

static float coarse(unsigned *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return (float)(int)((*seed >> 16) % 5u) * 0.5f - 1.0f;   /* -1, -0.5, 0, 0.5, 1 */
}

static uint32_t ref_select(int32_t *sel, const float *q, const float *bkey, float *score, uint32_t pos) {
    const uint32_t n_blocks = (pos + 1u) / R;
    if (n_blocks <= BUDGET) {
        for (uint32_t c = 0; c <= pos; c++) sel[c] = (int32_t)c;
        return pos + 1u;
    }
    for (uint32_t b = 0; b < n_blocks; b++) {
        float s = 0.0f;
        for (uint32_t h = 0; h < N_HEAD; h++) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < D; i++) dot += q[h * D + i] * bkey[(uint64_t)b * D + i];
            if (dot > 0.0f) s += dot;
        }
        score[b] = s;
    }
    uint32_t n_sel = 0;
    for (uint32_t k = 0; k < BUDGET; k++) {
        uint32_t best = 0;
        for (uint32_t b = 1; b < n_blocks; b++) if (score[b] > score[best]) best = b;
        score[best] = -1.0f;
        sel[n_sel++] = (int32_t)best;
    }
    for (uint32_t i = 1; i < n_sel; i++) {
        const int32_t v = sel[i];
        uint32_t j = i;
        while (j > 0 && sel[j - 1] > v) { sel[j] = sel[j - 1]; j--; }
        sel[j] = v;
    }
    for (uint32_t i = n_sel; i-- > 0;) {
        const int32_t b = sel[i];
        for (uint32_t j = 0; j < R; j++) sel[i * R + j] = b * R + (int32_t)j;
    }
    n_sel *= R;
    for (uint32_t c = n_blocks * R; c <= pos; c++) sel[n_sel++] = (int32_t)c;
    return n_sel;
}

static int run_case(uint32_t pos0, uint32_t n_tokens, unsigned seed,
                    ds4_gpu_tensor *sel_t, ds4_gpu_tensor *n_sel_t, ds4_gpu_tensor *keys_t,
                    ds4_gpu_tensor *q_t, ds4_gpu_tensor *bkey_t) {
    const uint64_t n_q = (uint64_t)n_tokens * N_HEAD * D;
    const uint64_t n_b = (uint64_t)(CTX / R) * D;
    float *q = malloc(n_q * sizeof(float));
    float *bkey = malloc(n_b * sizeof(float));
    float *score = malloc((CTX / R) * sizeof(float));
    int32_t *got = malloc((uint64_t)n_tokens * MAX_SEL * sizeof(int32_t));
    uint32_t *got_n = malloc(n_tokens * sizeof(uint32_t));
    int32_t *want = malloc(MAX_SEL * sizeof(int32_t));
    if (!q || !bkey || !score || !got || !got_n || !want) return 1;
    for (uint64_t i = 0; i < n_q; i++) q[i] = coarse(&seed);
    for (uint64_t i = 0; i < n_b; i++) bkey[i] = coarse(&seed);
    /* the block key cache is bf16; these values are exact in it */
    uint16_t *bkey_bf16 = malloc(n_b * sizeof(uint16_t));
    if (!bkey_bf16) return 1;
    for (uint64_t i = 0; i < n_b; i++) {
        uint32_t bits;
        memcpy(&bits, &bkey[i], sizeof bits);
        bkey_bf16[i] = (uint16_t)(bits >> 16);
    }
    int rc = 1;
    if (ds4_gpu_tensor_write(q_t, 0, q, n_q * sizeof(float)) &&
        ds4_gpu_tensor_write(bkey_t, 0, bkey_bf16, n_b * sizeof(uint16_t)) &&
        ds4_gpu_qwen4exp_qsa_select(sel_t, n_sel_t, keys_t, ROWS, q_t, bkey_t,
                                    N_HEAD, D, R, BUDGET, MAX_SEL, CTX, pos0, n_tokens) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(sel_t, 0, got, (uint64_t)n_tokens * MAX_SEL * sizeof(int32_t)) &&
        ds4_gpu_tensor_read(n_sel_t, 0, got_n, n_tokens * sizeof(uint32_t))) {
        rc = 0;
        /* the scored keys of the first token, straight from the scratch */
        {
            const uint32_t n_blocks = (pos0 + 1u) / R;
            uint64_t *keys = malloc((CTX / R) * sizeof(uint64_t));
            if (keys && n_blocks > BUDGET && ds4_gpu_tensor_read(keys_t, 0, keys, (CTX / R) * sizeof(uint64_t))) {
                uint32_t bad = 0;
                for (uint32_t b = 0; b < n_blocks; b++) {
                    float s = 0.0f;
                    for (uint32_t h = 0; h < N_HEAD; h++) {
                        float dot = 0.0f;
                        for (uint32_t i = 0; i < D; i++) dot += q[h * D + i] * bkey[(uint64_t)b * D + i];
                        if (dot > 0.0f) s += dot;
                    }
                    float g;
                    const uint32_t bits = (uint32_t)(keys[b] >> 32);
                    memcpy(&g, &bits, sizeof g);
                    if (g != s || (uint32_t)keys[b] != 0xFFFFFFFFu - b) {
                        if (bad++ < 3) fprintf(stderr, "  score block %u: gpu %g ref %g (low %u)\n", b, g, s, (uint32_t)keys[b]);
                    }
                }
                if (bad) fprintf(stderr, "qsa-select: pos %u: %u of %u block scores differ\n", pos0, bad, n_blocks);
            }
            free(keys);
        }
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            const uint32_t n = ref_select(want, q + (uint64_t)t * N_HEAD * D, bkey, score, pos0 + t);
            if (n != got_n[t] || memcmp(want, got + (uint64_t)t * MAX_SEL, n * sizeof(int32_t)) != 0) {
                fprintf(stderr, "qsa-select: pos %u: want %u cells, got %u\n", pos0 + t, n, got_n[t]);
                for (uint32_t i = 0; i < n && i < got_n[t]; i++) {
                    if (want[i] != got[(uint64_t)t * MAX_SEL + i]) {
                        fprintf(stderr, "  first difference at entry %u: want %d got %d\n",
                                i, want[i], got[(uint64_t)t * MAX_SEL + i]);
                        break;
                    }
                }
                rc = 1;
            }
        }
    }
    printf("qsa-select: pos0 %u, %u tokens: %s\n", pos0, n_tokens, rc ? "FAIL" : "ok");
    free(q); free(bkey); free(bkey_bf16); free(score); free(got); free(got_n); free(want);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((uint64_t)ROWS * 4u * MAX_SEL * sizeof(int32_t));
    ds4_gpu_tensor *n_sel = ds4_gpu_tensor_alloc(ROWS * 4u * sizeof(uint32_t));
    ds4_gpu_tensor *keys = ds4_gpu_tensor_alloc((uint64_t)ROWS * (CTX / R) * sizeof(uint64_t));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)ROWS * 4u * N_HEAD * D * sizeof(float));
    ds4_gpu_tensor *bkey = ds4_gpu_tensor_alloc((uint64_t)(CTX / R) * D * sizeof(float));
    if (!sel || !n_sel || !keys || !q || !bkey) return 1;
    int rc = 0;
    rc |= run_case(0, 64, 1u, sel, n_sel, keys, q, bkey);          /* everything within the budget */
    rc |= run_case(2000, 128, 2u, sel, n_sel, keys, q, bkey);      /* the chunk crosses the budget */
    rc |= run_case(2051, 1, 3u, sel, n_sel, keys, q, bkey);        /* first selecting token, decode */
    rc |= run_case(4093, 8, 4u, sel, n_sel, keys, q, bkey);        /* block boundaries inside the chunk */
    rc |= run_case(CTX - 300, 300, 5u, sel, n_sel, keys, q, bkey); /* more tokens than scratch rows, end of context */
    printf("qsa-select: %s\n", rc ? "FAIL" : "PASS");
    return rc;
}
