// SPDX-License-Identifier: MIT
// ds4_qwen_fp4.cu - see ds4_qwen_fp4.h.  Compiled with the vendored mmq
// include set for common.cuh (the fp4/ue4m3 helpers and block_nvfp4).

#include "ds4_qwen_fp4.h"

#include "common.cuh"

#include <cstdio>

namespace {

constexpr int TILE_ROWS = 32;                 /* slots per tile, matches the expert plan */
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

/* A row's 32 code bytes are staged as 8 words with the two 16-byte halves
 * swapped on rows 4-7 of every eight, so the eight row addresses of one
 * ldmatrix 8x8 matrix hit distinct bank groups without padding. */
__device__ __forceinline__ uint32_t swz(uint32_t row, uint32_t word) {
    return row * 8u + (((word >> 2u) ^ ((row >> 2u) & 1u)) << 2u) + (word & 3u);
}

__device__ __forceinline__ void ldsm_x4(uint32_t *r, const uint32_t *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldsm_x2(uint32_t *r, const uint32_t *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

/* m16n8k64 block-scaled E2M1 x E2M1 with UE4M3 scales, f32 accumulate:
 * the scale words hold the four sub-block scales of one row/column. */
__device__ __forceinline__ void mma_nvfp4(float c[4], const uint32_t a[4], const uint32_t b[2], uint32_t sa, uint32_t sb) {
#ifdef BLACKWELL_MMA_AVAILABLE
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
#else
    GGML_UNUSED_VARS(c, a, b, sa, sb);
    NO_DEVICE_CODE;
#endif
}

/* Block (row tile, column tile), 32 x 8 threads: the row tile maps through
 * the plan to one expert and up to 32 of its slots, the column tile covers
 * COLS outputs.  Each K step stages KS super-blocks per row of both
 * operands (codes for ldmatrix, the packed scales beside them) in one of
 * two shared buffers; the next step's 8-byte global loads are issued
 * before the current step's MMAs so their latency overlaps, and one
 * barrier per step separates the buffers.  Long segments keep more of
 * every fetched 32-byte sector, so K a multiple of 256 takes KS = 4
 * (144-byte rows, the shared-memory limit at 128 columns) and the K = 640
 * down projection KS = 2; a whole-row single-buffer variant and 64-column
 * tiles were both slower.  Warp w owns rows 16*(w/4).. and columns
 * (COLS/4)*(w%4).. . */
template <int KS, int COLS>
__global__ void __launch_bounds__(256, 2) moe_gemm_kernel(
        float *out, const block_nvfp4 *W, const float *scales, const block_nvfp4 *xq, uint32_t x_per_slot,
        const int32_t *order, const uint32_t *plan, uint32_t n_expert, uint32_t n_used, uint32_t K, uint32_t M) {
    constexpr int SEG_U2 = KS * 36 / 8;                       /* 8-byte words per row segment */
    constexpr int ITEMS = (TILE_ROWS + COLS) * SEG_U2;        /* words staged per step */
    constexpr int PER_THREAD = (ITEMS + 255) / 256;
    constexpr int NT = COLS / 32;                             /* n-tiles per warp */
    constexpr uint32_t NONE = 0xffffffffu;
    static_assert(KS * 36 % 8 == 0, "row segments must be whole 8-byte words");
    __shared__ __align__(16) uint32_t a_qs[2][KS][TILE_ROWS * 8];
    __shared__ __align__(16) uint32_t b_qs[2][KS][COLS * 8];
    __shared__ uint32_t a_sc[2][KS][TILE_ROWS];
    __shared__ uint32_t b_sc[2][KS][COLS];
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
    const uint32_t col0 = blockIdx.y * COLS;
    const uint32_t n_super = K / QK_NVFP4;
    const uint32_t n_steps = n_super / KS;
    const uint32_t row_u2 = n_super * 36u / 8u;                /* 8-byte words per operand row */
    const uint2 *xq2 = (const uint2 *)xq;
    const uint2 *w2 = (const uint2 *)(W + (size_t)e * M * n_super);

    /* this thread's share of a step's loads, as word offsets into the
     * activation rows (A items) or the expert (B items); NONE reads zero */
    uint32_t item_off[PER_THREAD];
#pragma unroll
    for (int p = 0; p < PER_THREAD; p++) {
        const uint32_t i = tid + 256u * p;
        const bool is_a = i < (uint32_t)(TILE_ROWS * SEG_U2);
        const uint32_t j = is_a ? i : i - TILE_ROWS * SEG_U2;
        const uint32_t r = j / SEG_U2, u = j % SEG_U2;
        item_off[p] = NONE;
        if (i < (uint32_t)ITEMS) {
            if (is_a) {
                const int32_t xr = xrow_of[r];
                if (xr >= 0) item_off[p] = (uint32_t)xr * row_u2 + u;
            } else if (col0 + r < M) {
                item_off[p] = (col0 + r) * row_u2 + u;
            }
        }
    }
    uint2 pre[PER_THREAD];
    auto fetch = [&](uint32_t step) {
#pragma unroll
        for (int p = 0; p < PER_THREAD; p++) {
            const bool is_a = tid + 256u * p < (uint32_t)(TILE_ROWS * SEG_U2);
            const uint2 *base = is_a ? xq2 : w2;
            pre[p] = item_off[p] != NONE ? base[item_off[p] + step * SEG_U2] : make_uint2(0u, 0u);
        }
    };
    auto stage = [&](uint32_t buf) {
#pragma unroll
        for (int p = 0; p < PER_THREAD; p++) {
            const uint32_t i = tid + 256u * p;
            if (i >= (uint32_t)ITEMS) continue;
            const bool is_a = i < (uint32_t)(TILE_ROWS * SEG_U2);
            const uint32_t j = is_a ? i : i - TILE_ROWS * SEG_U2;
            const uint32_t r = j / SEG_U2, u = j % SEG_U2;
            const uint32_t words[2] = { pre[p].x, pre[p].y };
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint32_t w = u * 2u + h;                    /* word within the segment */
                const uint32_t sb = w / 9u, k = w % 9u;
                if (is_a) {
                    if (k == 0u) a_sc[buf][sb][r] = words[h];
                    else a_qs[buf][sb][swz(r, k - 1u)] = words[h];
                } else {
                    if (k == 0u) b_sc[buf][sb][r] = words[h];
                    else b_qs[buf][sb][swz(r, k - 1u)] = words[h];
                }
            }
        }
    };

    const uint32_t wr = (warp >> 2u) * 16u;
    const uint32_t wc = (warp & 3u) * (COLS / 4u);
    const uint32_t tidx_a = lane / 4u + (lane % 2u) * 8u;   /* rows whose scales this lane supplies */
    const uint32_t tidx_b = lane / 4u;
    const uint32_t ar = wr + (lane & 7u) + ((lane >> 3u) & 1u) * 8u;   /* ldmatrix row addresses */
    const uint32_t ah = (lane >> 4u) * 4u;
    const uint32_t bh = ((lane >> 3u) & 1u) * 4u;
    float C[NT][4];
#pragma unroll
    for (int f = 0; f < NT; f++) C[f][0] = C[f][1] = C[f][2] = C[f][3] = 0.0f;

    fetch(0u);
    stage(0u);
    __syncthreads();
    for (uint32_t step = 0; step < n_steps; step++) {
        const uint32_t buf = step & 1u;
        if (step + 1u < n_steps) fetch(step + 1u);
#pragma unroll
        for (int sb = 0; sb < KS; sb++) {
            uint32_t A[4];
            ldsm_x4(A, &a_qs[buf][sb][swz(ar, ah)]);
            const uint32_t sa = a_sc[buf][sb][wr + tidx_a];
#pragma unroll
            for (int f = 0; f < NT; f++) {
                uint32_t B[2];
                const uint32_t n = wc + f * 8u + (lane & 7u);
                ldsm_x2(B, &b_qs[buf][sb][swz(n, bh)]);
                mma_nvfp4(C[f], A, B, sa, b_sc[buf][sb][wc + f * 8u + tidx_b]);
            }
        }
        if (step + 1u < n_steps) stage(buf ^ 1u);
        __syncthreads();
    }
    const float gscale = scales[e];
#pragma unroll
    for (int f = 0; f < NT; f++) {
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const int32_t slot = slot_of[wr + lane / 4u + (l >> 1) * 8u];
            const uint32_t col = col0 + wc + f * 8u + (lane % 4u) * 2u + (l & 1);
            if (slot >= 0 && col < M) out[(size_t)slot * M + col] = C[f][l] * gscale;
        }
    }
}

template <int KS, int COLS>
void launch_moe_gemm(
        float *out, const block_nvfp4 *W, const float *scales, const block_nvfp4 *xq, int x_per_slot,
        const int32_t *order, const uint32_t *plan, int n_expert, int n_used, int K, int M, int rows,
        cudaStream_t stream) {
    /* every expert adds at most one partial tile to the slots' full tiles */
    const unsigned max_tiles = (unsigned)(((long long)rows * n_used + TILE_ROWS - 1) / TILE_ROWS) + (unsigned)n_expert;
    const dim3 grid(max_tiles, (unsigned)((M + COLS - 1) / COLS), 1u);
    moe_gemm_kernel<KS, COLS><<<grid, dim3(32, 8, 1), 0, stream>>>(
        out, W, scales, xq, (uint32_t)(x_per_slot != 0), order, plan,
        (uint32_t)n_expert, (uint32_t)n_used, (uint32_t)K, (uint32_t)M);
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
        n_used <= 0 || K <= 0 || K % (2 * QK_NVFP4) != 0 || M <= 0 || rows <= 0) {
        return -1;
    }
    const block_nvfp4 *w = (const block_nvfp4 *)W;
    const block_nvfp4 *x = (const block_nvfp4 *)xq;
    if (K % (4 * QK_NVFP4) == 0) {
        launch_moe_gemm<4, 128>(out, w, scales, x, x_per_slot, order, plan, n_expert, n_used, K, M, rows, stream);
    } else {
        launch_moe_gemm<2, 128>(out, w, scales, x, x_per_slot, order, plan, n_expert, n_used, K, M, rows, stream);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
