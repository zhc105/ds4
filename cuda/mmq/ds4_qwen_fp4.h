// SPDX-License-Identifier: MIT
// ds4_qwen_fp4.h - Qwen3.8-Flash-Next routed experts on the Blackwell FP4
// tensor cores.  Built with the vendored mma.cuh primitives (see VENDOR.md);
// the kernels themselves are ds4's own.
//
// The experts are NVFP4 (64-element super-blocks: 4 UE4M3 sub-block scales
// then 32 bytes of E2M1) and the activations are quantised into the same
// layout, so both operands feed mma.sync's block-scaled FP4 instruction
// directly and every weight byte is read exactly once per prefill chunk.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Quantise `rows` rows of K f32 values into NVFP4 super-blocks (K % 64 == 0):
 * per 16 values a UE4M3 scale of amax/6 and nearest E2M1 codes.  With `up`
 * the values are SiLU(x) * up (the expert activation), quantised without a
 * f32 round trip. */
int ds4_qwen_fp4_quantize(const float *x, const float *up, void *xq, int rows, int K, cudaStream_t stream);

/* Grouped expert GEMM: out[slot][col] = scales[e] * xq[row(slot)] . W[e][col]
 * for every (token, slot) pair, with `plan`/`order` the expert grouping made
 * by ds4_gpu_qwen4exp_expert_plan (32-slot tiles).  W is the stacked
 * [n_expert][M][K] NVFP4 tensor; xq has one NVFP4 row per slot (x_per_slot)
 * or per token (row = slot / n_used).  K % 128 == 0. */
int ds4_qwen_fp4_moe_gemm(
    const void     *W,
    const float    *scales,
    const void     *xq,
    int             x_per_slot,
    const int32_t  *order,
    const uint32_t *plan,
    int             n_expert,
    int             n_used,
    int             K,
    int             M,
    int             rows,
    float          *out,
    cudaStream_t    stream);

#ifdef __cplusplus
}
#endif
