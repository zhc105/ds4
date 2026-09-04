// SPDX-License-Identifier: MIT
// ds4_qwen_fp4.cu - see ds4_qwen_fp4.h.  Compiled with the vendored mmq
// include set for common.cuh (fp4/ue4m3 helpers) and mma.cuh (the tile
// types, ldmatrix loaders and the block-scaled FP4 mma wrapper).

#include "ds4_qwen_fp4.h"

#include "common.cuh"
#include "mma.cuh"

#include <cstdio>

using namespace ggml_cuda_mma;

namespace {

constexpr int TILE_ROWS = 32;                 /* slots per tile, matches the expert plan */
constexpr int TILE_COLS = 128;                /* output columns per block */
constexpr int LD        = 12;                 /* ints per shared row: 32 bytes of codes, 16-byte aligned */
constexpr int MAX_EXPERT = 512;

/* The plan arrays, in the order ds4_gpu_qwen4exp_expert_plan writes them. */
enum { PLAN_COUNT = 0, PLAN_START = 1, PLAN_TILE = 2 };

/* One thread per 16-value sub-block: a UE4M3 scale of amax/6 and nearest
 * E2M1 codes, byte b holding values b (low nibble) and b+8 (high). */
__global__ void quantize_kernel(const float *x, block_nvfp4 *xq, int rows, int K) {
    const int n_sub = K / QK_NVFP4_SUB;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * n_sub) return;
    const int row = (int)(i / n_sub);
    const int sub = (int)(i % n_sub);
    const float *src = x + (long long)row * K + sub * QK_NVFP4_SUB;
    float v[QK_NVFP4_SUB];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k++) {
        v[k] = src[k];
        amax = fmaxf(amax, fabsf(v[k]));
    }
    /* ggml's ue4m3 decoder returns half the E4M3 value (its E2M1 tables are
     * doubled), so the inverse scale for the true grid is 0.5 / that */
    const uint8_t code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
    const float half_scale = ggml_cuda_ue4m3_to_fp32(code);
    const float inv = half_scale > 0.0f ? 0.5f / half_scale : 0.0f;
    block_nvfp4 *blk = xq + (long long)row * (K / QK_NVFP4) + sub / (QK_NVFP4 / QK_NVFP4_SUB);
    const int s = sub % (QK_NVFP4 / QK_NVFP4_SUB);
    blk->d[s] = code;
    uint8_t *qs = blk->qs + s * (QK_NVFP4_SUB / 2);
#pragma unroll
    for (int b = 0; b < QK_NVFP4_SUB / 2; b++) {
        qs[b] = (uint8_t)(ggml_cuda_float_to_fp4_e2m1(v[b], inv) |
                          (ggml_cuda_float_to_fp4_e2m1(v[b + QK_NVFP4_SUB / 2], inv) << 4));
    }
}

/* Block (row tile, column tile), launched as 32 x 8 threads because the
 * mma.cuh tile helpers index lanes by threadIdx.x: the row tile maps through
 * the plan to one expert and up to 32 of its slots.  Each K step stages one
 * super-block per row of both operands in shared memory (codes for
 * ldmatrix, the packed scales beside them); warp w owns rows 16*(w/4).. and
 * columns 32*(w%4).., four m16n8k64 block-scaled MMAs per step. */
__global__ void moe_gemm_kernel(
        float *out, const block_nvfp4 *W, const float *scales, const block_nvfp4 *xq, uint32_t x_per_slot,
        const int32_t *order, const uint32_t *plan, uint32_t n_expert, uint32_t n_used, uint32_t K, uint32_t M) {
#ifdef BLACKWELL_MMA_AVAILABLE
    __shared__ __align__(16) int a_qs[TILE_ROWS][LD];
    __shared__ __align__(16) int b_qs[TILE_COLS][LD];
    __shared__ uint32_t a_sc[TILE_ROWS];
    __shared__ uint32_t b_sc[TILE_COLS];
    __shared__ int32_t slot_of[TILE_ROWS];
    __shared__ int32_t xrow_of[TILE_ROWS];
    __shared__ uint32_t tiles[MAX_EXPERT + 1];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32u + lane;
    const uint32_t *start = plan + PLAN_START * (n_expert + 1u);
    const uint32_t *tile_start = plan + PLAN_TILE * (n_expert + 1u);
    for (uint32_t i = tid; i <= n_expert; i += 256u) tiles[i] = tile_start[i];
    __syncthreads();
    const uint32_t bx = blockIdx.x;
    if (bx >= tiles[n_expert]) return;
    uint32_t lo = 0, hi = n_expert;              /* last e with tiles[e] <= bx */
    while (lo + 1u < hi) {
        const uint32_t mid = (lo + hi) / 2u;
        if (tiles[mid] <= bx) lo = mid; else hi = mid;
    }
    const uint32_t e = lo;
    const uint32_t first = start[e] + (bx - tiles[e]) * TILE_ROWS;
    const uint32_t end = start[e + 1u];
    if (tid < TILE_ROWS) {
        const uint32_t idx = first + tid;
        const int32_t slot = idx < end ? order[idx] : -1;
        slot_of[tid] = slot;
        xrow_of[tid] = slot < 0 ? -1 : (x_per_slot ? slot : slot / (int32_t)n_used);
    }
    __syncthreads();
    const uint32_t col0 = blockIdx.y * TILE_COLS;
    const uint32_t n_super = K / QK_NVFP4;
    const block_nvfp4 *We = W + (size_t)e * M * n_super;
    const uint32_t wr = (warp >> 2u) * 16u;
    const uint32_t wc = (warp & 3u) * 32u;
    const uint32_t tidx_a = lane / 4u + (lane % 2u) * 8u;   /* rows whose scales this lane supplies */
    const uint32_t tidx_b = lane / 4u;
    tile<16, 8, float> C[4];
    for (uint32_t b = 0; b < n_super; b++) {
        if (tid < TILE_ROWS) {
            const int32_t xr = xrow_of[tid];
            const uint32_t *src = xr < 0 ? NULL : (const uint32_t *)(xq + (size_t)xr * n_super + b);
            a_sc[tid] = src ? src[0] : 0u;
            for (int j = 0; j < 8; j++) a_qs[tid][j] = src ? (int)src[1 + j] : 0;
        } else if (tid < TILE_ROWS + TILE_COLS) {
            const uint32_t c = tid - TILE_ROWS;
            const uint32_t *src = col0 + c < M ? (const uint32_t *)(We + (size_t)(col0 + c) * n_super + b) : NULL;
            b_sc[c] = src ? src[0] : 0u;
            for (int j = 0; j < 8; j++) b_qs[c][j] = src ? (int)src[1 + j] : 0;
        }
        __syncthreads();
        tile<16, 8, int> A;
        load_ldmatrix(A, &a_qs[wr][0], LD);
        const uint32_t sa = a_sc[wr + tidx_a];
#pragma unroll
        for (int f = 0; f < 4; f++) {
            tile<8, 8, int> B;
            load_ldmatrix(B, &b_qs[wc + f * 8][0], LD);
            mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C[f], A, B, sa, b_sc[wc + f * 8 + tidx_b]);
        }
        __syncthreads();
    }
    const float gscale = scales[e];
#pragma unroll
    for (int f = 0; f < 4; f++) {
#pragma unroll
        for (int l = 0; l < tile<16, 8, float>::ne; l++) {
            const int32_t slot = slot_of[wr + tile<16, 8, float>::get_i(l)];
            const uint32_t col = col0 + wc + f * 8 + tile<16, 8, float>::get_j(l);
            if (slot >= 0 && col < M) out[(size_t)slot * M + col] = C[f].x[l] * gscale;
        }
    }
#else
    GGML_UNUSED_VARS(out, W, scales, xq, x_per_slot, order, plan, n_expert, n_used, K, M);
    NO_DEVICE_CODE;
#endif
}

} // namespace

extern "C" int ds4_qwen_fp4_quantize(const float *x, void *xq, int rows, int K, cudaStream_t stream) {
    if (!x || !xq || rows <= 0 || K <= 0 || K % QK_NVFP4 != 0) return -1;
    const long long n = (long long)rows * (K / QK_NVFP4_SUB);
    quantize_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(x, (block_nvfp4 *)xq, rows, K);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int ds4_qwen_fp4_moe_gemm(
        const void *W, const float *scales, const void *xq, int x_per_slot,
        const int32_t *order, const uint32_t *plan, int n_expert, int n_used,
        int K, int M, int rows, float *out, cudaStream_t stream) {
    if (!W || !scales || !xq || !order || !plan || !out || n_expert <= 0 || n_expert > MAX_EXPERT ||
        n_used <= 0 || K <= 0 || K % QK_NVFP4 != 0 || M <= 0 || rows <= 0) {
        return -1;
    }
    /* every expert adds at most one partial tile to the slots' full tiles */
    const unsigned max_tiles = (unsigned)(((long long)rows * n_used + TILE_ROWS - 1) / TILE_ROWS) + (unsigned)n_expert;
    const dim3 grid(max_tiles, (unsigned)((M + TILE_COLS - 1) / TILE_COLS), 1u);
    moe_gemm_kernel<<<grid, dim3(32, 8, 1), 0, stream>>>(
        out, (const block_nvfp4 *)W, scales, (const block_nvfp4 *)xq, (uint32_t)(x_per_slot != 0),
        order, plan, (uint32_t)n_expert, (uint32_t)n_used, (uint32_t)K, (uint32_t)M);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
