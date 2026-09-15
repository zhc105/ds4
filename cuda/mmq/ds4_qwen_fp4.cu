// SPDX-License-Identifier: MIT
// ds4_qwen_fp4.cu - see ds4_qwen_fp4.h.  Compiled with the vendored mmq
// include set for common.cuh (the fp4/ue4m3 helpers and block_nvfp4).

#include "ds4_qwen_fp4.h"

#include "common.cuh"

#include <cstdio>
#include <cuda_bf16.h>

namespace {

constexpr int TILE_ROWS = DS4_QWEN_FP4_TILE_ROWS;   /* slots per tile, matches the expert plan */
constexpr int NTHREADS = 256;                        /* 8 warps: 2 row groups of 32 x 4 column quarters */
constexpr int MAX_EXPERT = 512;
constexpr int NSTAGE = 3;                            /* K steps in flight, cp.async; 4 to 6 measured no faster */

/* The plan arrays, in the order ds4_gpu_qwen4exp_expert_plan writes them. */
enum { PLAN_COUNT = 0, PLAN_START = 1, PLAN_TILE = 2 };

/* One thread per 16-value sub-block: a UE4M3 scale of amax/6 and nearest
 * E2M1 codes, byte b holding values b (low nibble) and b+8 (high).  With
 * `up` the value is SiLU(x) * up, as ds4's swiglu computes it. */
template <typename T>
__global__ void quantize_kernel(const T *x, const T *up, block_nvfp4 *xq, int rows, int K) {
    const int n_sub = K / QK_NVFP4_SUB;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * n_sub) return;
    const int row = (int)(i / n_sub);
    const int sub = (int)(i % n_sub);
    const long long off = (long long)row * K + sub * QK_NVFP4_SUB;
    float v[QK_NVFP4_SUB];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k++) {
        v[k] = (float)x[off + k];
        if (up) v[k] = v[k] / (1.0f + expf(-v[k])) * (float)up[off + k];
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

/* CB bytes (8 or 16) global -> shared, asynchronous; `bytes` 0 zero-fills. */
template <uint32_t CB>
__device__ __forceinline__ void cp_async(void *dst, const void *src, uint32_t bytes) {
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    if (CB == 16u) asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(bytes));
    else asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(d), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void prefetch_l2(const void *p) {
    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

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

/* One K step of an operand row in shared memory is its KS super-blocks as
 * they lie in memory (4 scale bytes then 32 code bytes each), so the copy
 * is CB-byte pieces of one contiguous segment.  The smem row stride keeps
 * the eight rows one fragment load touches on distinct banks: 36 words
 * for KS 4 (4 apart mod 32), 20 for KS 2 (its 18 padded). */
template <int KS>
struct moe_seg {
    static_assert(KS == 2 || KS == 4, "row stride chosen for KS 2 and 4");
    static constexpr uint32_t BYTES = KS * 36u;
    static constexpr uint32_t CB = BYTES % 16u == 0u ? 16u : 8u;
    static constexpr uint32_t STRIDE = KS == 2 ? 80u : BYTES;
};

template <int KS, int COLS>
__host__ __device__ constexpr int stage_bytes() { return (TILE_ROWS + COLS) * (int)moe_seg<KS>::STRIDE; }

/* Block (row tile, column tile), 32 x 8 threads: the row tile maps through
 * the plan to one expert and up to TILE_ROWS of its slots, the column tile
 * covers COLS outputs.  The FP4 MMA is nowhere near the limit here (a
 * chunk of a few thousand tokens gives every expert some tens of slots);
 * the kernel is bound by its operand loads.  Two things keep those near
 * the DRAM rate.  The tile's weight rows are prefetched into L2 whole at
 * the start: the K steps read each row in KS*36-byte pieces, and LPDDR
 * serves 20 such passes over 128 strided rows at half the rate it serves
 * the rows once in order (24 MB of L2 holds the tiles in flight).  Then
 * NSTAGE steps are in flight at once, copied global -> shared by cp.async
 * in 8- or 16-byte pieces of the rows' raw bytes, and the MMA fragments
 * are gathered from that layout with 32-bit shared loads: the two
 * operands share a 16-value scale block, so any fixed order of the values
 * inside it is exact.  Warp w owns rows 32*(w/4).. as two m16 fragments
 * and columns (COLS/4)*(w%4).. . */
template <int KS, int COLS>
__global__ void __launch_bounds__(NTHREADS, 2) moe_gemm_kernel(
        void *out, uint32_t out_bf16, const block_nvfp4 *W, const float *scales, const block_nvfp4 *xq, uint32_t x_per_slot,
        const int32_t *order, const uint32_t *plan, uint32_t n_expert, uint32_t n_used, uint32_t K, uint32_t M) {
    typedef moe_seg<KS> seg;
    constexpr uint32_t CPR = seg::BYTES / seg::CB;              /* copies per row segment */
    constexpr uint32_t ITEMS = (TILE_ROWS + COLS) * CPR;        /* copies per step */
    constexpr int PER_THREAD = (int)((ITEMS + NTHREADS - 1) / NTHREADS);
    constexpr uint32_t SW = seg::STRIDE / 4u;                   /* smem row stride in words */
    constexpr int NT = COLS / 32;                               /* n-tiles per warp */
    constexpr uint32_t NONE = 0xffffffffu;
    extern __shared__ __align__(16) uint32_t smem_raw[];
    constexpr uint32_t STAGE_WORDS = (uint32_t)stage_bytes<KS, COLS>() / 4u;
    __shared__ int32_t slot_of[TILE_ROWS];
    __shared__ int32_t xrow_of[TILE_ROWS];
    __shared__ uint32_t tiles[MAX_EXPERT + 1];
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = threadIdx.y;
    const uint32_t tid = warp * 32u + lane;
    const uint32_t *start = plan + PLAN_START * (n_expert + 1u);
    const uint32_t *tile_start = plan + PLAN_TILE * (n_expert + 1u);
    for (uint32_t i = tid; i <= n_expert; i += NTHREADS) tiles[i] = tile_start[i];
    __syncthreads();
    /* column tiles are the fast grid axis: a row tile's column tiles run
     * together, so its activation rows cross the DRAM once and its
     * expert's other row tile, a few blocks later, finds the weights in L2 */
    const uint32_t bx = blockIdx.y;
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
    const uint32_t col0 = blockIdx.x * COLS;
    const uint32_t n_super = K / QK_NVFP4;
    const uint32_t n_steps = n_super / KS;
    const uint32_t row_bytes = n_super * 36u;
    const char *xb = (const char *)xq;
    const char *wb = (const char *)(W + (size_t)e * M * n_super);

    /* the tile's weight rows into L2, whole */
    {
        const uint32_t lpr = (row_bytes + 127u) / 128u;
        for (uint32_t i = tid; i < COLS * lpr; i += NTHREADS) {
            const uint32_t r = col0 + i / lpr, l = (i % lpr) * 128u;
            if (r < M && l < row_bytes) prefetch_l2(wb + (size_t)r * row_bytes + l);
        }
    }

    /* this thread's share of a step's copies, as byte offsets into the
     * activation rows (A items) or the expert (B items); NONE copies zero */
    uint32_t src_off[PER_THREAD];
#pragma unroll
    for (int p = 0; p < PER_THREAD; p++) {
        const uint32_t i = tid + NTHREADS * p;
        const uint32_t r = i / CPR, c = i % CPR;
        src_off[p] = NONE;
        if (i < ITEMS) {
            if (r < TILE_ROWS) {
                const int32_t xr = xrow_of[r];
                if (xr >= 0) src_off[p] = (uint32_t)xr * row_bytes + c * seg::CB;
            } else if (col0 + r - TILE_ROWS < M) {
                src_off[p] = (col0 + r - TILE_ROWS) * row_bytes + c * seg::CB;
            }
        }
    }
    auto issue = [&](uint32_t step) {   /* the step's segments into stage step % NSTAGE */
        char *stage = (char *)(smem_raw + (step % NSTAGE) * STAGE_WORDS);
#pragma unroll
        for (int p = 0; p < PER_THREAD; p++) {
            const uint32_t i = tid + NTHREADS * p;
            if (i >= ITEMS) continue;
            const uint32_t r = i / CPR, c = i % CPR;
            const uint32_t off = src_off[p];
            const char *src = (r < TILE_ROWS ? xb : wb) + (off == NONE ? 0u : off + step * seg::BYTES);
            cp_async<seg::CB>(stage + r * seg::STRIDE + c * seg::CB, src, off == NONE ? 0u : seg::CB);
        }
        cp_async_commit();
    };

    /* fragment words: register i of an m16 A fragment is code word
     * (i/2)*4 + lane%4 of row lane/4 + (i%2)*8, of an n8 B fragment word
     * i*4 + lane%4 of row lane/4; the scale word of a super-block is its
     * first, the codes follow */
    const uint32_t wr = (warp >> 2u) * 32u;
    const uint32_t wc = (warp & 3u) * (COLS / 4u);
    const uint32_t frow = lane / 4u, fword = 1u + (lane & 3u);
    const uint32_t tidx_a = lane / 4u + (lane % 2u) * 8u;   /* rows whose scales this lane supplies */
    float C[2][NT][4];
#pragma unroll
    for (int rf = 0; rf < 2; rf++) {
#pragma unroll
        for (int f = 0; f < NT; f++) C[rf][f][0] = C[rf][f][1] = C[rf][f][2] = C[rf][f][3] = 0.0f;
    }

    /* NSTAGE - 1 steps ahead; empty groups keep the wait count uniform */
#pragma unroll
    for (uint32_t s = 0; s < NSTAGE - 1; s++) {
        if (s < n_steps) issue(s); else cp_async_commit();
    }
    for (uint32_t step = 0; step < n_steps; step++) {
        cp_async_wait<NSTAGE - 2>();   /* this thread's copies of the step landed */
        __syncthreads();               /* everyone's did, and step - 1's stage is free */
        if (step + NSTAGE - 1 < n_steps) issue(step + NSTAGE - 1); else cp_async_commit();
        const uint32_t *sa_rows = smem_raw + (step % NSTAGE) * STAGE_WORDS;
        const uint32_t *sb_rows = sa_rows + TILE_ROWS * SW;
#pragma unroll
        for (int sb = 0; sb < KS; sb++) {
            uint32_t A[2][4];
            uint32_t sa[2];
#pragma unroll
            for (int rf = 0; rf < 2; rf++) {
                const uint32_t *row = sa_rows + (wr + rf * 16u + frow) * SW + sb * 9u;
#pragma unroll
                for (int i = 0; i < 4; i++) A[rf][i] = row[(i & 1) * 8u * SW + (i >> 1) * 4u + fword];
                sa[rf] = sa_rows[(wr + rf * 16u + tidx_a) * SW + sb * 9u];
            }
#pragma unroll
            for (int f = 0; f < NT; f++) {
                const uint32_t *row = sb_rows + (wc + f * 8u + frow) * SW + sb * 9u;
                const uint32_t B[2] = { row[fword], row[4u + fword] };
                const uint32_t sbw = row[0];
#pragma unroll
                for (int rf = 0; rf < 2; rf++) mma_nvfp4(C[rf][f], A[rf], B, sa[rf], sbw);
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();   /* the stages are free for the output tile */
    const float gscale = scales[e];
    if (out_bf16 && M % 8u == 0u) {
        /* The fragments scatter 2-byte values over 16 rows per warp, which
         * the memory system pays for in whole sectors; the tile's outputs
         * are staged and each slot row leaves as contiguous 16-byte stores. */
        constexpr uint32_t LD = COLS + 8u;   /* bf16 row stride: 16-byte rows, skewed across banks */
        __nv_bfloat16 *cs = (__nv_bfloat16 *)smem_raw;
#pragma unroll
        for (int rf = 0; rf < 2; rf++) {
#pragma unroll
            for (int f = 0; f < NT; f++) {
#pragma unroll
                for (int l = 0; l < 4; l++) {
                    const uint32_t row = wr + rf * 16u + lane / 4u + (l >> 1) * 8u;
                    const uint32_t col = wc + f * 8u + (lane % 4u) * 2u + (l & 1);
                    cs[row * LD + col] = __float2bfloat16(C[rf][f][l] * gscale);
                }
            }
        }
        __syncthreads();
        constexpr uint32_t VEC = COLS / 8u;   /* uint4 of 8 bf16 per row */
        for (uint32_t i = tid; i < TILE_ROWS * VEC; i += NTHREADS) {
            const uint32_t row = i / VEC, col = col0 + (i % VEC) * 8u;
            const int32_t slot = slot_of[row];
            if (slot < 0 || col >= M) continue;
            *(uint4 *)((__nv_bfloat16 *)out + (size_t)slot * M + col) = *(const uint4 *)&cs[row * LD + (i % VEC) * 8u];
        }
        return;
    }
#pragma unroll
    for (int rf = 0; rf < 2; rf++) {
#pragma unroll
        for (int f = 0; f < NT; f++) {
#pragma unroll
            for (int l = 0; l < 4; l++) {
                const int32_t slot = slot_of[wr + rf * 16u + lane / 4u + (l >> 1) * 8u];
                const uint32_t col = col0 + wc + f * 8u + (lane % 4u) * 2u + (l & 1);
                if (slot < 0 || col >= M) continue;
                const float v = C[rf][f][l] * gscale;
                if (out_bf16) ((__nv_bfloat16 *)out)[(size_t)slot * M + col] = __float2bfloat16(v);
                else ((float *)out)[(size_t)slot * M + col] = v;
            }
        }
    }
}

template <int KS, int COLS>
int launch_moe_gemm(
        void *out, int out_bf16, const block_nvfp4 *W, const float *scales, const block_nvfp4 *xq, int x_per_slot,
        const int32_t *order, const uint32_t *plan, int n_expert, int n_used, int K, int M, int rows,
        cudaStream_t stream) {
    constexpr int STAGED_OUT = TILE_ROWS * (COLS + 8) * 2;   /* the bf16 output tile of the epilogue */
    constexpr int STAGES = NSTAGE * stage_bytes<KS, COLS>();
    constexpr int SMEM = STAGES > STAGED_OUT ? STAGES : STAGED_OUT;
    static bool attr_ok = false;   /* the stages may exceed the 48 KiB default: opt in once per instantiation */
    if (!attr_ok) {
        if (cudaFuncSetAttribute(moe_gemm_kernel<KS, COLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess ||
            cudaFuncSetAttribute(moe_gemm_kernel<KS, COLS>, cudaFuncAttributePreferredSharedMemoryCarveout, 100) != cudaSuccess) {
            return -3;
        }
        attr_ok = true;
    }
    /* every expert adds at most one partial tile to the slots' full tiles */
    const unsigned max_tiles = (unsigned)(((long long)rows * n_used + TILE_ROWS - 1) / TILE_ROWS) + (unsigned)n_expert;
    const dim3 grid((unsigned)((M + COLS - 1) / COLS), max_tiles, 1u);
    moe_gemm_kernel<KS, COLS><<<grid, dim3(32, NTHREADS / 32, 1), SMEM, stream>>>(
        out, (uint32_t)(out_bf16 != 0), W, scales, xq, (uint32_t)(x_per_slot != 0), order, plan,
        (uint32_t)n_expert, (uint32_t)n_used, (uint32_t)K, (uint32_t)M);
    return 0;
}

} // namespace

extern "C" int ds4_qwen_fp4_quantize(const void *x, const void *up, int in_bf16, void *xq, int rows, int K, cudaStream_t stream) {
    if (!x || !xq || rows <= 0 || K <= 0 || K % QK_NVFP4 != 0) return -1;
    const long long n = (long long)rows * (K / QK_NVFP4_SUB);
    const unsigned blocks = (unsigned)((n + 255) / 256);
    if (in_bf16) {
        quantize_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
            (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)up, (block_nvfp4 *)xq, rows, K);
    } else {
        quantize_kernel<float><<<blocks, 256, 0, stream>>>((const float *)x, (const float *)up, (block_nvfp4 *)xq, rows, K);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

extern "C" int ds4_qwen_fp4_moe_gemm(
        const void *W, const float *scales, const void *xq, int x_per_slot,
        const int32_t *order, const uint32_t *plan, int n_expert, int n_used,
        int K, int M, int rows, void *out, int out_bf16, cudaStream_t stream) {
    if (!W || !scales || !xq || !order || !plan || !out || n_expert <= 0 || n_expert > MAX_EXPERT ||
        n_used <= 0 || K <= 0 || K % (2 * QK_NVFP4) != 0 || M <= 0 || rows <= 0) {
        return -1;
    }
    const block_nvfp4 *w = (const block_nvfp4 *)W;
    const block_nvfp4 *x = (const block_nvfp4 *)xq;
    /* 4 super-blocks a step when the rows allow 16-byte copies (K % 256),
     * else 2 with 8-byte ones: the gate/up rows of 2560 take the first,
     * the down rows of 640 the second */
    const int rc = (K / QK_NVFP4) % 4 == 0
        ? launch_moe_gemm<4, 128>(out, out_bf16, w, scales, x, x_per_slot, order, plan, n_expert, n_used, K, M, rows, stream)
        : launch_moe_gemm<2, 128>(out, out_bf16, w, scales, x, x_per_slot, order, plan, n_expert, n_used, K, M, rows, stream);
    if (rc != 0) return rc;
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
