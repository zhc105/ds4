/* Qwen family CUDA kernels, written for Qwen3.5 and shared by Flash-Next:
 * NVFP4 matmul, Gated DeltaNet, gated GQA attention.
 *
 * Included from ds4_cuda.cu so the helpers there (cuda_resolve_weight_ptr,
 * cuda_decode_stream, warp_sum_f32, cuda_ok) are visible.  The ds4.c graph
 * calls the extern "C" wrappers below under DS4_QWEN_GPU; Metal and ROCm
 * builds never reference them.  Layouts mirror the CPU reference in ds4.c:
 *   mixed      [n_tok][2*k_dim + v_dim]   conv+SiLU output, q/k L2-normalised
 *   conv_state [n_conv-1][conv_dim]        most recent inputs, oldest first
 *   ssm_state  [n_v_head][hd][hd]          S[value j][key i]
 *   k/v cache  [ctx][n_head_kv * head_dim] f32, RoPE applied to K
 */

#define QWEN35_CUDA_GDN_DIM 128u
/* Attention key-range splitting for small batches (decode). */
#define QWEN35_ATTN_SPLIT_ROWS 8u
#define QWEN35_ATTN_SPLIT_MAX 64u

/* E2M1 nibble to float by building the IEEE bits directly: a lookup table in
 * constant memory would serialise on the divergent per-lane indices. */
__device__ __forceinline__ float qwen35_cuda_e2m1(uint32_t nib) {
    const uint32_t e = (nib >> 1u) & 3u;
    const uint32_t m = nib & 1u;
    uint32_t bits = e ? (((126u + e) << 23) | (m << 22)) : (m ? 0x3F000000u : 0u);
    bits |= (nib & 8u) << 28;
    return __uint_as_float(bits);
}

/* The same E2M1 nibble as bf16 bits: the value is exact in bf16, so the
 * f32 pattern above shifted right by 16. */
__device__ __forceinline__ uint32_t qwen35_e2m1_bf16_bits(uint32_t nib) {
    const uint32_t e = (nib >> 1u) & 3u;
    const uint32_t m = nib & 1u;
    const uint32_t bits = e ? (((126u + e) << 7) | (m << 6)) : (m ? 0x3F00u : 0u);
    return bits | ((nib & 8u) << 12);
}

__device__ __forceinline__ float qwen35_cuda_ue4m3(uint8_t bits) {
    bits &= 0x7fu;
    if (bits == 0x7fu) return 0.0f;
    const int e = bits >> 3;
    const float m = (float)(bits & 7u);
    return e == 0 ? ldexpf(m, -9) : ldexpf(1.0f + m * 0.125f, e - 7);
}

__device__ __forceinline__ float qwen35_cuda_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float qwen35_cuda_silu(float x) {
    return x * qwen35_cuda_sigmoid(x);
}

__device__ __forceinline__ float qwen35_cuda_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* Block-wide sum for blockDim.x <= 1024; every thread receives the total. */
__device__ __forceinline__ float qwen35_cuda_block_sum(float v, float *scratch32) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t n_warps = (blockDim.x + 31u) >> 5u;
    v = warp_sum_f32(v);
    __syncthreads();
    if (lane == 0u) scratch32[warp] = v;
    __syncthreads();
    float total = lane < n_warps ? scratch32[lane] : 0.0f;
    total = warp_sum_f32(total);
    return __shfl_sync(0xffffffffu, total, 0);
}

/* ---- NVFP4 matmul --------------------------------------------------------
 * Weight rows are 64-element super-blocks: 4 UE4M3 scales + 32 packed E2M1
 * bytes (element j low nibble, j+8 high nibble within each 16-block). */

/* Decode: one warp per output column.  Each lane owns one 16-element
 * sub-block, so a warp sweeps eight consecutive super-blocks (288 bytes) per
 * iteration and the loads coalesce into whole sectors; the scale bytes ride
 * in the same sectors. */
__device__ __forceinline__ float qwen35_nvfp4_warp_dot(
        const uint8_t *wrow, const float *xrow, uint32_t n_super, uint32_t lane) {
    const uint32_t sub = lane & 3u;          /* sub-block within the super-block */
    float sum = 0.0f;
    for (uint32_t b = lane >> 2u; b < n_super; b += 8u) {
        const uint8_t *blk = wrow + (uint64_t)b * 36u;
        const uint32_t *qs = (const uint32_t *)(blk + 4u + sub * 8u);
        const uint32_t lo = qs[0];
        const uint32_t hi = qs[1];
        const float4 *xs = (const float4 *)(xrow + (uint64_t)b * 64u + sub * 16u);
        const float4 x0 = xs[0], x1 = xs[1], x2 = xs[2], x3 = xs[3];
        const float xv[16] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w,
                               x2.x, x2.y, x2.z, x2.w, x3.x, x3.y, x3.z, x3.w };
        float acc = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            const uint32_t byte_lo = (lo >> (8u * j)) & 0xffu;
            const uint32_t byte_hi = (hi >> (8u * j)) & 0xffu;
            acc = fmaf(qwen35_cuda_e2m1(byte_lo & 15u), xv[j], acc);
            acc = fmaf(qwen35_cuda_e2m1(byte_lo >> 4u), xv[j + 8u], acc);
            acc = fmaf(qwen35_cuda_e2m1(byte_hi & 15u), xv[j + 4u], acc);
            acc = fmaf(qwen35_cuda_e2m1(byte_hi >> 4u), xv[j + 12u], acc);
        }
        sum = fmaf(qwen35_cuda_ue4m3(blk[sub]), acc, sum);
    }
    return warp_sum_f32(sum);
}

__global__ static void qwen35_nvfp4_matvec_kernel(
        float *out, const uint8_t *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, float scale) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col = blockIdx.x * 8u + warp;
    const uint32_t row = blockIdx.y;
    if (col >= out_dim || row >= n_tok) return;
    const uint32_t n_super = in_dim / 64u;
    const float sum = qwen35_nvfp4_warp_dot(w + (uint64_t)col * n_super * 36u,
                                            x + (uint64_t)row * in_dim, n_super, lane);
    if (lane == 0u) out[(uint64_t)row * out_dim + col] = sum * scale;
}

/* Weight storage types the Qwen graph feeds to the matmuls, as GGUF type ids
 * so ds4.c passes the tensor type straight through. */
enum { QWEN35_W_F32 = 0, QWEN35_W_Q8_0 = 8, QWEN35_W_BF16 = 30, QWEN35_W_NVFP4 = 40 };

/* Prefill GEMM tile: a 256-thread block accumulates a 64x64 output tile over
 * the full input width, weights of any storage type dequantised to shared
 * memory 64 input columns at a time, activations gathered through 64 row
 * pointers (NULL rows read as zero).  Thread (tx, ty) owns rows ty*4..+3
 * and columns tx*4..+3.  Plain f32 FMA on f32 activations: this is the
 * correctness path the CPU reference is compared against, not the
 * tensor-core path. */
__device__ static void qwen35_gemm_tile(
        float acc[4][4], float ws[64][65], float xs[64][65],
        const uint8_t *w, uint32_t wtype, const float *const *xr,
        uint32_t in_dim, uint32_t col0, uint32_t out_dim) {
    const uint32_t tid = threadIdx.x;
    const uint32_t tx = tid & 15u;
    const uint32_t ty = tid >> 4u;
    const uint32_t n_super = in_dim / 64u;
    for (uint32_t b = 0; b < n_super; b++) {
        {
            const uint32_t c = tid >> 2u;
            const uint32_t s = tid & 3u;
            const uint32_t col = col0 + c;
            if (col >= out_dim) {
                for (uint32_t j = 0; j < 16u; j++) ws[c][s * 16u + j] = 0.0f;
            } else if (wtype == QWEN35_W_NVFP4) {
                const uint8_t *blk = w + ((uint64_t)col * n_super + b) * 36u;
                const float d = qwen35_cuda_ue4m3(blk[s]);
                const uint8_t *qs = blk + 4u + s * 8u;
                for (uint32_t j = 0; j < 8u; j++) {
                    ws[c][s * 16u + j] = qwen35_cuda_e2m1(qs[j] & 15u) * d;
                    ws[c][s * 16u + j + 8u] = qwen35_cuda_e2m1(qs[j] >> 4u) * d;
                }
            } else if (wtype == QWEN35_W_BF16) {
                const uint16_t *src = (const uint16_t *)w + (uint64_t)col * in_dim + (uint64_t)b * 64u + s * 16u;
                for (uint32_t j = 0; j < 16u; j++) ws[c][s * 16u + j] = __uint_as_float((uint32_t)src[j] << 16);
            } else {
                const float *src = (const float *)w + (uint64_t)col * in_dim + (uint64_t)b * 64u + s * 16u;
                for (uint32_t j = 0; j < 16u; j++) ws[c][s * 16u + j] = src[j];
            }
        }
        {
            const uint32_t r = tid >> 2u;
            const uint32_t k0 = (tid & 3u) * 16u;
            const float *src = xr[r];
            for (uint32_t j = 0; j < 16u; j++) xs[r][k0 + j] = src ? src[(uint64_t)b * 64u + k0 + j] : 0.0f;
        }
        __syncthreads();
        for (uint32_t k = 0; k < 64u; k++) {
            float xv[4], wv[4];
            for (uint32_t i = 0; i < 4u; i++) {
                xv[i] = xs[ty * 4u + i][k];
                wv[i] = ws[tx * 4u + i][k];
            }
            for (uint32_t i = 0; i < 4u; i++) {
                for (uint32_t j = 0; j < 4u; j++) acc[i][j] = fmaf(xv[i], wv[j], acc[i][j]);
            }
        }
        __syncthreads();
    }
}

/* Tensor-core prefill tile for BF16 and NVFP4 weights: a 128x128 output
 * tile per 256-thread block on bf16 tensor cores with f32 accumulation,
 * 64 input columns (one NVFP4 super-block) per step.  The weights are exact
 * in bf16 (BF16 as stored; an E2M1 x UE4M3 product has five significant
 * bits).  The f32 activations are split into a bf16 high part and a bf16
 * remainder and multiplied twice, so the products carry about 16 bits of
 * the activation instead of 8 and the tile stays within f32 rounding of the
 * CPU reference; a single bf16 pass was measured to cost 3% argmax
 * agreement through the MoE routing.  Warp w owns rows 64*(w/4)..+63 and
 * columns 32*(w%4)..+31, eight accumulators.  Row r of the tile reads the
 * activation row xr[r] (NULL reads as zero) and writes output row orow[r]
 * (negative rows are dropped), which is how the dense GEMM and the grouped
 * expert GEMM share it.  Shared memory is dynamic: QWEN35_MMA_SMEM bytes. */
#define QWEN35_MMA_ROWS 32u                                 /* activation rows per tile */
#define QWEN35_MMA_RF (QWEN35_MMA_ROWS / 32u)               /* row fragments per warp */
#define QWEN35_MMA_A_PER_THREAD (QWEN35_MMA_ROWS / 16u)     /* float4 of activations per thread per step */
#define QWEN35_MMA_COLS 128u                                /* output columns per tile */
#define QWEN35_MMA_LD 72u                                   /* bf16 tile row stride, 144 bytes */
#define QWEN35_MMA_A_BYTES (QWEN35_MMA_ROWS * QWEN35_MMA_LD * 2u)
#define QWEN35_MMA_B_BYTES (QWEN35_MMA_COLS * QWEN35_MMA_LD * 2u)
#define QWEN35_MMA_STAGE_LD 20u                             /* per-warp 16x16 f32 staging, reuses the operand tiles */
#define QWEN35_MMA_SMEM (2u * QWEN35_MMA_A_BYTES + QWEN35_MMA_B_BYTES)

/* One thread's share of the operands, fetched from global memory ahead of
 * use so the loads overlap the tensor-core work: the activations one K step
 * ahead (QWEN35_MMA_A_PER_THREAD float4), BF16 weights one step ahead (four
 * uint4), and NVFP4 weights one group of four super-blocks ahead.  A
 * column's super-blocks are contiguous 36-byte records, so per group each of
 * the column's two threads reads its two records as one 72-byte run instead
 * of four scattered 16-byte pieces, and dequantises one record per step. */
#define QWEN35_MMA_GROUP 4u
typedef struct {
    float4 a[QWEN35_MMA_A_PER_THREAD];
    uint32_t q[18];                                          /* two raw NVFP4 super-blocks */
    uint4 wb[4];
} qwen35_mma_fetch;

__device__ __forceinline__ void qwen35_mma_fetch_step(
        qwen35_mma_fetch *f, const float *const *xr, const uint8_t *w, uint32_t wtype,
        uint32_t in_dim, uint32_t n_super, uint32_t col0, uint32_t out_dim, uint32_t b) {
    const uint32_t tid = threadIdx.x;
    for (uint32_t i = 0; i < QWEN35_MMA_A_PER_THREAD; i++) {
        const uint32_t idx = tid + i * 256u;
        const float *src = xr[idx >> 4u];
        f->a[i] = src ? *(const float4 *)(src + (uint64_t)b * 64u + (idx & 15u) * 4u)
                      : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const uint32_t col = col0 + (tid >> 1u);
    const uint32_t half = tid & 1u;
    if (wtype == QWEN35_W_NVFP4) {
        if (b % QWEN35_MMA_GROUP != 0u) return;
        const uint32_t first = b + half * 2u;
        if (col < out_dim && first < n_super) {
            const uint32_t *src = (const uint32_t *)(w + ((uint64_t)col * n_super + first) * 36u);
            const uint32_t n = first + 1u < n_super ? 18u : 9u;
            for (uint32_t j = 0; j < 18u; j++) f->q[j] = j < n ? src[j] : 0u;
        }
    } else if (wtype == QWEN35_W_Q8_0) {
        /* One 32-element q8_0 block per thread, dequantised straight into the
         * bf16 operand the store step copies, so the tensor cores see the same
         * bf16 path a BF16 weight would take while the bytes stay int8. */
        const uint32_t n_blocks = in_dim / 32u;
        const uint32_t blk_index = 2u * b + half;
        if (col >= out_dim || blk_index >= n_blocks) {
            for (uint32_t j = 0; j < 4u; j++) f->wb[j] = make_uint4(0u, 0u, 0u, 0u);
        } else {
            const uint8_t *blk = w + ((uint64_t)col * n_blocks + blk_index) * 34u;
            const float d = __half2float(*(const __half *)blk);
            const int8_t *qs = (const int8_t *)(blk + 2u);
            uint32_t v[16];
#pragma unroll
            for (uint32_t j = 0; j < 16u; j++) {
                const __nv_bfloat162 p = __floats2bfloat162_rn((float)qs[2u * j] * d,
                                                                (float)qs[2u * j + 1u] * d);
                v[j] = *(const uint32_t *)&p;
            }
            f->wb[0] = make_uint4(v[0], v[1], v[2], v[3]);
            f->wb[1] = make_uint4(v[4], v[5], v[6], v[7]);
            f->wb[2] = make_uint4(v[8], v[9], v[10], v[11]);
            f->wb[3] = make_uint4(v[12], v[13], v[14], v[15]);
        }
    } else if (col >= out_dim) {
        for (uint32_t j = 0; j < 4u; j++) f->wb[j] = make_uint4(0u, 0u, 0u, 0u);
    } else {
        const uint4 *src = (const uint4 *)((const uint16_t *)w + (uint64_t)col * in_dim + (uint64_t)b * 64u + half * 32u);
        for (uint32_t j = 0; j < 4u; j++) f->wb[j] = src[j];
    }
}

/* Convert the fetched operands of step b into the shared bf16 tiles:
 * activations as hi and lo halves, weights dequantised (NVFP4: the thread
 * holding this step's record writes the column's whole 64 values) or
 * copied (BF16). */
__device__ __forceinline__ void qwen35_mma_store_step(
        const qwen35_mma_fetch *f, __nv_bfloat16 (*as)[QWEN35_MMA_ROWS][QWEN35_MMA_LD],
        __nv_bfloat16 (*bs)[QWEN35_MMA_LD], uint32_t wtype, bool have_col, uint32_t b) {
    const uint32_t tid = threadIdx.x;
    for (uint32_t i = 0; i < QWEN35_MMA_A_PER_THREAD; i++) {
        const uint32_t idx = tid + i * 256u;
        const uint32_t r = idx >> 4u;
        const uint32_t k0 = (idx & 15u) * 4u;
        const float4 v = f->a[i];
        /* hi is the truncated f32 (an integer mask, no conversion), lo the
         * exact remainder rounded once: still 16 significant bits together */
        const uint32_t xb = __float_as_uint(v.x), yb = __float_as_uint(v.y);
        const uint32_t zb = __float_as_uint(v.z), wb = __float_as_uint(v.w);
        const uint32_t hi0 = __byte_perm(xb, yb, 0x7632), hi1 = __byte_perm(zb, wb, 0x7632);
        __nv_bfloat162 lo[2];
        lo[0] = __floats2bfloat162_rn(v.x - __uint_as_float(xb & 0xffff0000u), v.y - __uint_as_float(yb & 0xffff0000u));
        lo[1] = __floats2bfloat162_rn(v.z - __uint_as_float(zb & 0xffff0000u), v.w - __uint_as_float(wb & 0xffff0000u));
        *(uint2 *)&as[0][r][k0] = make_uint2(hi0, hi1);
        *(uint2 *)&as[1][r][k0] = *(const uint2 *)lo;
    }
    const uint32_t cidx = tid >> 1u;
    const uint32_t half = tid & 1u;
    if (wtype == QWEN35_W_NVFP4) {
        if (((b % QWEN35_MMA_GROUP) >> 1u) != half) return;   /* the other thread holds this record */
        const uint32_t *rec = f->q + (b & 1u) * 9u;
        const uint32_t scales = have_col ? rec[0] : 0u;
        for (uint32_t s = 0; s < 4u; s++) {
            /* the block scale as a bf16 pair (UE4M3 is exact in bf16), so the
             * E2M1 x scale products, which have five significant bits, come
             * out exact from a packed bf16 multiply: no float conversions */
            const float d = qwen35_cuda_ue4m3((uint8_t)(scales >> (8u * s)));
            const __nv_bfloat162 d2 = __floats2bfloat162_rn(d, d);
            const uint32_t lo = rec[1u + 2u * s], hi = rec[2u + 2u * s];
            uint32_t v[8];                                   /* 16 bf16 in element order */
            for (uint32_t j = 0; j < 4u; j += 2u) {
                const uint32_t lo0 = (lo >> (8u * j)) & 0xffu, lo1 = (lo >> (8u * j + 8u)) & 0xffu;
                const uint32_t hi0 = (hi >> (8u * j)) & 0xffu, hi1 = (hi >> (8u * j + 8u)) & 0xffu;
                /* element j from byte j's low nibble, j+8 from its high nibble
                 * (the lo word), j+4 and j+12 likewise from the hi word */
                v[j / 2u] = qwen35_e2m1_bf16_bits(lo0 & 15u) | (qwen35_e2m1_bf16_bits(lo1 & 15u) << 16);
                v[j / 2u + 4u] = qwen35_e2m1_bf16_bits(lo0 >> 4u) | (qwen35_e2m1_bf16_bits(lo1 >> 4u) << 16);
                v[j / 2u + 2u] = qwen35_e2m1_bf16_bits(hi0 & 15u) | (qwen35_e2m1_bf16_bits(hi1 & 15u) << 16);
                v[j / 2u + 6u] = qwen35_e2m1_bf16_bits(hi0 >> 4u) | (qwen35_e2m1_bf16_bits(hi1 >> 4u) << 16);
            }
            for (uint32_t i = 0; i < 8u; i++) {
                const __nv_bfloat162 p = __hmul2(*(const __nv_bfloat162 *)&v[i], d2);
                v[i] = *(const uint32_t *)&p;
            }
            uint4 *dst = (uint4 *)&bs[cidx][s * 16u];
            dst[0] = make_uint4(v[0], v[1], v[2], v[3]);
            dst[1] = make_uint4(v[4], v[5], v[6], v[7]);
        }
    } else {
        uint4 *dst = (uint4 *)&bs[cidx][half * 32u];
        for (uint32_t j = 0; j < 4u; j++) dst[j] = f->wb[j];
    }
}

/* Warp w owns rows (ROWS/2)*(w/4).. and columns 32*(w%4)..+31 of the
 * ROWS x 128 tile: two accumulators per row fragment. */
__device__ static void qwen35_mma_tile(
        float *out, uint32_t out_dim, float scale, const int32_t *orow, const float *const *xr,
        const uint8_t *w, uint32_t wtype, uint32_t in_dim, uint32_t col0, unsigned char *smem) {
    namespace wmma = nvcuda::wmma;
    __nv_bfloat16 (*as)[QWEN35_MMA_ROWS][QWEN35_MMA_LD] = (__nv_bfloat16 (*)[QWEN35_MMA_ROWS][QWEN35_MMA_LD])smem;
    __nv_bfloat16 (*bs)[QWEN35_MMA_LD] = (__nv_bfloat16 (*)[QWEN35_MMA_LD])(smem + 2u * QWEN35_MMA_A_BYTES);
    float (*stage)[16][QWEN35_MMA_STAGE_LD] = (float (*)[16][QWEN35_MMA_STAGE_LD])smem;   /* the operand tiles, once the K loop is done */
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t wr = (warp >> 2u) * (QWEN35_MMA_ROWS / 2u);   /* warp's first tile row */
    const uint32_t wc = (warp & 3u) * 32u;                  /* warp's first tile column */
    const uint32_t n_super = in_dim / 64u;
    const bool have_col = col0 + (tid >> 1u) < out_dim;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[QWEN35_MMA_RF][2];
    for (uint32_t i = 0; i < QWEN35_MMA_RF; i++) {
        wmma::fill_fragment(c[i][0], 0.0f);
        wmma::fill_fragment(c[i][1], 0.0f);
    }
    qwen35_mma_fetch f;
    qwen35_mma_fetch_step(&f, xr, w, wtype, in_dim, n_super, col0, out_dim, 0u);
    for (uint32_t b = 0; b < n_super; b++) {
        qwen35_mma_store_step(&f, as, bs, wtype, have_col, b);
        __syncthreads();
        if (b + 1u < n_super) qwen35_mma_fetch_step(&f, xr, w, wtype, in_dim, n_super, col0, out_dim, b + 1u);
        for (uint32_t kk = 0; kk < 64u; kk += 16u) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[2];
            wmma::load_matrix_sync(bf[0], &bs[wc][kk], QWEN35_MMA_LD);
            wmma::load_matrix_sync(bf[1], &bs[wc + 16u][kk], QWEN35_MMA_LD);
            for (uint32_t i = 0; i < QWEN35_MMA_RF; i++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_hi, a_lo;
                wmma::load_matrix_sync(a_hi, &as[0][wr + i * 16u][kk], QWEN35_MMA_LD);
                wmma::load_matrix_sync(a_lo, &as[1][wr + i * 16u][kk], QWEN35_MMA_LD);
                wmma::mma_sync(c[i][0], a_lo, bf[0], c[i][0]);
                wmma::mma_sync(c[i][0], a_hi, bf[0], c[i][0]);
                wmma::mma_sync(c[i][1], a_lo, bf[1], c[i][1]);
                wmma::mma_sync(c[i][1], a_hi, bf[1], c[i][1]);
            }
        }
        __syncthreads();
    }
    /* epilogue: each warp stages its fragments one at a time and scatters
     * them to the output rows */
    for (uint32_t i = 0; i < QWEN35_MMA_RF; i++) {
        for (uint32_t t = 0; t < 2u; t++) {
            wmma::store_matrix_sync(&stage[warp][0][0], c[i][t], QWEN35_MMA_STAGE_LD, wmma::mem_row_major);
            __syncwarp();
            for (uint32_t e = lane; e < 256u; e += 32u) {
                const uint32_t r = e >> 4u;
                const uint32_t cc = e & 15u;
                const int32_t row = orow[wr + i * 16u + r];
                const uint32_t col = col0 + wc + t * 16u + cc;
                if (row >= 0 && col < out_dim) out[(uint64_t)(uint32_t)row * out_dim + col] = stage[warp][r][cc] * scale;
            }
            __syncwarp();
        }
    }
}

/* Prefill GEMM on the f32 tile, kept for F32 weights (the router and the
 * shared-expert gate stay exact). */
__global__ static void qwen35_gemm_f32_kernel(
        float *out, const uint8_t *w, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, float scale) {
    __shared__ float ws[64][65];
    __shared__ float xs[64][65];
    __shared__ const float *xr[64];
    const uint32_t col0 = blockIdx.y * 64u;
    const uint32_t row0 = blockIdx.x * 64u;
    const uint32_t tid = threadIdx.x;
    const uint32_t tx = tid & 15u;
    const uint32_t ty = tid >> 4u;
    if (tid < 64u) xr[tid] = row0 + tid < n_tok ? x + (uint64_t)(row0 + tid) * in_dim : NULL;
    __syncthreads();
    float acc[4][4] = {{0.0f}};
    qwen35_gemm_tile(acc, ws, xs, w, QWEN35_W_F32, xr, in_dim, col0, out_dim);
    for (uint32_t i = 0; i < 4u; i++) {
        const uint32_t row = row0 + ty * 4u + i;
        if (row >= n_tok) continue;
        for (uint32_t j = 0; j < 4u; j++) {
            const uint32_t col = col0 + tx * 4u + j;
            if (col < out_dim) out[(uint64_t)row * out_dim + col] = acc[i][j] * scale;
        }
    }
}

/* Prefill GEMM on the tensor-core tile: block (row tile, column tile), row
 * tiles fastest so blocks sharing a weight tile run together. */
__global__ static void qwen35_gemm_mma_kernel(
        float *out, const uint8_t *w, uint32_t wtype, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, float scale) {
    extern __shared__ __align__(32) unsigned char smem[];
    __shared__ const float *xr[QWEN35_MMA_ROWS];
    __shared__ int32_t orow[QWEN35_MMA_ROWS];
    const uint32_t row0 = blockIdx.x * QWEN35_MMA_ROWS;
    const uint32_t col0 = blockIdx.y * QWEN35_MMA_COLS;
    const uint32_t tid = threadIdx.x;
    if (tid < QWEN35_MMA_ROWS) {
        const bool valid = row0 + tid < n_tok;
        xr[tid] = valid ? x + (uint64_t)(row0 + tid) * in_dim : NULL;
        orow[tid] = valid ? (int32_t)(row0 + tid) : -1;
    }
    __syncthreads();
    qwen35_mma_tile(out, out_dim, scale, orow, xr, w, wtype, in_dim, col0, smem);
}

/* Decode-sized BF16 matvec: a warp per output column walks the weight row
 * once and dots it with every activation row (up to 8, a speculative
 * batch), so the bandwidth-bound weight read is not repeated per token. */
template <int N>
__global__ static void qwen35_matvec_bf16_rows_kernel(
        float *out, const uint16_t *weights, const float *x, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col = blockIdx.x * 8u + warp;
    if (col >= out_dim) return;
    float sum[N];
#pragma unroll
    for (int r = 0; r < N; r++) sum[r] = 0.0f;
    /* four weights (8 bytes) and four activations (16 bytes) per lane per step */
    const uint2 *wrow = (const uint2 *)(weights + (uint64_t)col * in_dim);
    for (uint32_t i = lane; i < in_dim / 4u; i += 32u) {
        const uint2 wq = wrow[i];
        const float w0 = __uint_as_float(wq.x << 16), w1 = __uint_as_float(wq.x & 0xffff0000u);
        const float w2 = __uint_as_float(wq.y << 16), w3 = __uint_as_float(wq.y & 0xffff0000u);
#pragma unroll
        for (int r = 0; r < N; r++) {
            const float4 xv = *(const float4 *)(x + (uint64_t)r * in_dim + i * 4u);
            sum[r] = fmaf(w0, xv.x, sum[r]);
            sum[r] = fmaf(w1, xv.y, sum[r]);
            sum[r] = fmaf(w2, xv.z, sum[r]);
            sum[r] = fmaf(w3, xv.w, sum[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < N; r++) {
        const float total = warp_sum_f32(sum[r]);
        if (lane == 0u) out[(uint64_t)r * out_dim + col] = total;
    }
}

static void qwen35_matvec_bf16_rows(
        float *out, const uint16_t *w, const float *x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
        cudaStream_t stream) {
    const dim3 grid((out_dim + 7u) / 8u, 1u, 1u);
    switch (n_tok) {
    case 1: qwen35_matvec_bf16_rows_kernel<1><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 2: qwen35_matvec_bf16_rows_kernel<2><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 3: qwen35_matvec_bf16_rows_kernel<3><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 4: qwen35_matvec_bf16_rows_kernel<4><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 5: qwen35_matvec_bf16_rows_kernel<5><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 6: qwen35_matvec_bf16_rows_kernel<6><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 7: qwen35_matvec_bf16_rows_kernel<7><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    default: qwen35_matvec_bf16_rows_kernel<8><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    }
}

/* Round f32 activations to bf16 (the operands of the bypass GEMMs). */
__global__ static void qwen35_to_bf16_kernel(__nv_bfloat16 *dst, const float *x, uint64_t n) {
    const uint64_t i = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i + 4u <= n) {
        const float4 v = *(const float4 *)(x + i);
        *(__nv_bfloat162 *)(dst + i) = __floats2bfloat162_rn(v.x, v.y);
        *(__nv_bfloat162 *)(dst + i + 2u) = __floats2bfloat162_rn(v.z, v.w);
    } else {
        for (uint64_t j = i; j < n; j++) dst[j] = __float2bfloat16(x[j]);
    }
}

/* Launch helper: four elements per thread. */
static void qwen35_to_bf16(__nv_bfloat16 *dst, const float *x, uint64_t n, cudaStream_t stream) {
    qwen35_to_bf16_kernel<<<(unsigned)((n + 1023u) / 1024u), 256, 0, stream>>>(dst, x, n);
}

/* Decode-sized q8_0 matvec, the int8 twin of the BF16 rows kernel above: a
 * warp per output column walks the weight row once for every activation row.
 * Each lane takes four consecutive int8 of one 32-element block, so the
 * block's f16 scale stays a single scalar for the step and eight lanes cover
 * a block. */
template <int N>
__global__ static void qwen35_matvec_q8_0_rows_kernel(
        float *out, const uint8_t *w, const float *x, uint32_t in_dim, uint32_t out_dim) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col = blockIdx.x * 8u + warp;
    if (col >= out_dim) return;
    float sum[N];
#pragma unroll
    for (int r = 0; r < N; r++) sum[r] = 0.0f;
    const uint32_t n_blocks = in_dim / 32u;
    const uint8_t *wrow = w + (uint64_t)col * n_blocks * 34u;
    const uint32_t off = (lane & 7u) * 4u;
    for (uint32_t b = lane >> 3u; b < n_blocks; b += 4u) {
        const uint8_t *blk = wrow + (uint64_t)b * 34u;
        const float d = __half2float(*(const __half *)blk);
        const int8_t *qs = (const int8_t *)(blk + 2u);
        const float w0 = (float)qs[off + 0u] * d;
        const float w1 = (float)qs[off + 1u] * d;
        const float w2 = (float)qs[off + 2u] * d;
        const float w3 = (float)qs[off + 3u] * d;
        const uint32_t k = b * 32u + off;
#pragma unroll
        for (int r = 0; r < N; r++) {
            const float4 xv = *(const float4 *)(x + (uint64_t)r * in_dim + k);
            sum[r] = fmaf(w0, xv.x, sum[r]);
            sum[r] = fmaf(w1, xv.y, sum[r]);
            sum[r] = fmaf(w2, xv.z, sum[r]);
            sum[r] = fmaf(w3, xv.w, sum[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < N; r++) {
        const float total = warp_sum_f32(sum[r]);
        if (lane == 0u) out[(uint64_t)r * out_dim + col] = total;
    }
}

static void qwen35_matvec_q8_0_rows(
        float *out, const uint8_t *w, const float *x, uint32_t in_dim, uint32_t out_dim, uint32_t n_tok,
        cudaStream_t stream) {
    const dim3 grid((out_dim + 7u) / 8u, 1u, 1u);
    switch (n_tok) {
    case 1: qwen35_matvec_q8_0_rows_kernel<1><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 2: qwen35_matvec_q8_0_rows_kernel<2><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 3: qwen35_matvec_q8_0_rows_kernel<3><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 4: qwen35_matvec_q8_0_rows_kernel<4><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 5: qwen35_matvec_q8_0_rows_kernel<5><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 6: qwen35_matvec_q8_0_rows_kernel<6><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    case 7: qwen35_matvec_q8_0_rows_kernel<7><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    default: qwen35_matvec_q8_0_rows_kernel<8><<<grid, 256, 0, stream>>>(out, w, x, in_dim, out_dim); break;
    }
}

/* The inverse of the above: the generic quantized matmul takes f32
 * activations, and Flash-Next prefill keeps the token-mixer output in bf16. */
__global__ static void qwen35_bf16_to_f32_kernel(float *dst, const __nv_bfloat16 *x, uint64_t n) {
    const uint64_t i = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (i + 4u <= n) {
        const __nv_bfloat162 lo = *(__nv_bfloat162 *)(x + i);
        const __nv_bfloat162 hi = *(__nv_bfloat162 *)(x + i + 2u);
        *(float4 *)(dst + i) = make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y),
                                           __bfloat162float(hi.x), __bfloat162float(hi.y));
    } else {
        for (uint64_t j = i; j < n; j++) dst[j] = __bfloat162float(x[j]);
    }
}

static void qwen35_bf16_to_f32(float *dst, const __nv_bfloat16 *x, uint64_t n, cudaStream_t stream) {
    qwen35_bf16_to_f32_kernel<<<(unsigned)((n + 1023u) / 1024u), 256, 0, stream>>>(dst, x, n);
}

/* BF16-weight prefill GEMM through cuBLAS on bf16 activations with f32
 * accumulation, scaled by alpha: the checkpoint's own recipe for these
 * projections (vLLM and SGLang run them the same way), which measured
 * indistinguishable from an exact hi/lo split in perplexity and vLLM
 * agreement at twice the speed.  cuBLAS is deterministic here (no atomics)
 * and pipelines the tiles far better than the hand-written kernel, which
 * stays for NVFP4 weights. */
static int qwen35_cublas_bf16(
        float *out, const uint16_t *w, const float *x, const __nv_bfloat16 *x_bf16,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_rows, float scale, int tier, cudaStream_t stream) {
    const uint64_t n = (uint64_t)n_rows * in_dim;
    const __nv_bfloat16 *a = x_bf16;
    if (!a) {
        __nv_bfloat16 *buf = (__nv_bfloat16 *)cuda_tmp_alloc_on(tier, n * sizeof(__nv_bfloat16), "Qwen bf16 activations");
        if (!buf) return 0;
        qwen35_to_bf16(buf, x, n, stream);
        if (!cuda_ok(cudaGetLastError(), "Qwen bf16 convert launch")) return 0;
        a = buf;
    }
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(tier), CUBLAS_OP_T, CUBLAS_OP_N,
                                     (int)out_dim, (int)n_rows, (int)in_dim,
                                     &scale, w, CUDA_R_16BF, (int)in_dim, a, CUDA_R_16BF, (int)in_dim,
                                     &beta, out, CUDA_R_32F, (int)out_dim, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "Qwen BF16 GEMM");
}

/* Round an activation buffer to bf16 once for the matmuls that share it. */
extern "C" int ds4_gpu_qwen35_bf16(ds4_gpu_tensor *dst, const ds4_gpu_tensor *x, uint64_t n) {
    if (n == 0u || !dst || !x || dst->bytes < n * sizeof(__nv_bfloat16) || x->bytes < n * sizeof(float)) return 0;
    qwen35_to_bf16((__nv_bfloat16 *)dst->ptr, (const float *)x->ptr, n, cuda_decode_stream());
    return cuda_ok(cudaGetLastError(), "Qwen bf16 convert launch");
}

/* cuBLAS loads its GEMM kernels on first use, a few hundred milliseconds
 * per kernel family that would otherwise land in the first prompt's
 * prefill: run one small GEMM of each kind the graph uses (bf16 in with
 * f32 out, bf16 in and out, f32 pedantic) over scratch at graph creation. */
extern "C" int ds4_gpu_qwen35_warm(ds4_gpu_tensor *f32, ds4_gpu_tensor *bf16, ds4_gpu_tensor *out) {
    const uint32_t d = 64u;
    if (!g_cublas_ready) return 1;
    if (!f32 || !bf16 || !out || f32->bytes < (uint64_t)d * d * sizeof(float) ||
        bf16->bytes < (uint64_t)d * d * sizeof(__nv_bfloat16) || out->bytes < (uint64_t)d * d * sizeof(float)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(out);
    cublasHandle_t handle = cuda_cublas_for_tier(tier);
    const float one = 1.0f, zero = 0.0f;
    cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)d, (int)d, (int)d,
                                     &one, bf16->ptr, CUDA_R_16BF, (int)d, bf16->ptr, CUDA_R_16BF, (int)d,
                                     &zero, out->ptr, CUDA_R_32F, (int)d, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)d, (int)d, (int)d,
                          &one, bf16->ptr, CUDA_R_16BF, (int)d, bf16->ptr, CUDA_R_16BF, (int)d,
                          &zero, out->ptr, CUDA_R_16BF, (int)d, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)d, (int)d, (int)d,
                          &one, f32->ptr, CUDA_R_32F, (int)d, f32->ptr, CUDA_R_32F, (int)d,
                          &zero, out->ptr, CUDA_R_32F, (int)d, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
    }
    return cublas_ok(st, "Qwen cuBLAS warm-up") && cuda_ok(cudaStreamSynchronize(cuda_decode_stream()), "Qwen warm-up sync");
}

/* Opt a tensor-core kernel into its dynamic shared memory once. */
static bool qwen35_mma_smem_ready(const void *kernel, bool *ready) {
    if (*ready) return true;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)QWEN35_MMA_SMEM) != cudaSuccess) {
        fprintf(stderr, "ds4: Qwen tensor-core GEMM needs %u bytes of shared memory\n", (unsigned)QWEN35_MMA_SMEM);
        return false;
    }
    *ready = true;
    return true;
}

__global__ static void qwen35_scale_kernel(float *x, uint64_t n, float scale) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

static uint64_t qwen35_weight_bytes(uint32_t wtype, uint32_t in_dim, uint32_t out_dim) {
    switch (wtype) {
    case QWEN35_W_NVFP4: return in_dim % 64u == 0u ? (uint64_t)out_dim * (in_dim / 64u) * 36u : 0u;
    case QWEN35_W_BF16:  return (uint64_t)out_dim * in_dim * 2u;
    case QWEN35_W_F32:   return (uint64_t)out_dim * in_dim * 4u;
    /* GGML q8_0: 32 elements per block, an f16 scale and 32 int8. */
    case QWEN35_W_Q8_0:  return in_dim % 32u == 0u ? (uint64_t)out_dim * (in_dim / 32u) * 34u : 0u;
    default:             return 0u;
    }
}

/* f32 activations for a q8_0 prefill whose caller only produced the bf16
 * copy.  Process-lifetime: at most a prefill chunk of the widest projection
 * (~100 MiB), and only prefill-sized calls reach it (the decode islands
 * capture at n_tok <= 8). */
static ds4_gpu_tensor *qwen35_f32_activation_scratch(uint64_t elems) {
    static ds4_gpu_tensor *buf = NULL;
    static uint64_t cap = 0;
    if (buf && cap >= elems) return buf;
    if (buf) ds4_gpu_tensor_free(buf);
    buf = ds4_gpu_tensor_alloc(elems * sizeof(float));
    cap = buf ? elems : 0;
    return buf;
}

/* ---- Load-time q8 -----------------------------------------------------
 * Bypass projections the engine quantises as it loads them (ds4.c decides
 * which): int8 quants [out][in] contiguous and the fp16 block scales
 * [out][in/32] apart, GGML q8_0's numbers (a block of 32, d = amax/127,
 * q = round(x/d)) in a layout the decode kernel can stream with 16-byte
 * loads.  GGML q8_0 interleaves the 2-byte scale with every 32 quants, so
 * a row is only 2-byte aligned and its kernel loads bytes: 200 GB/s where
 * this layout reaches the 245 GB/s of a plain read.  A q8_0 tensor in the
 * file is re-laid identically, a bf16 one quantised; the source span is
 * then not cached on the device (ds4_gpu_model_range_replaced). */

struct qwen35_q8_pack {
    const void *map;
    uint64_t    offset;
    uint32_t    in_dim;
    uint32_t    out_dim;
    int8_t     *quants;
    __half     *scales;
};

static std::vector<qwen35_q8_pack> g_qwen35_q8_packs;

static const qwen35_q8_pack *qwen35_q8_pack_find(const void *map, uint64_t offset) {
    for (const qwen35_q8_pack &p : g_qwen35_q8_packs) {
        if (p.map == map && p.offset == offset) return &p;
    }
    return NULL;
}

static int qwen35_q8_pack_replaces(const void *map, uint64_t offset) {
    return qwen35_q8_pack_find(map, offset) != NULL;
}

/* One thread per block of 32: quantise a bf16 row segment (the converter's
 * q8_0_quantize: half-away-from-zero rounding, d = amax / 127 in f32). */
__global__ static void qwen35_q8_pack_bf16_kernel(
        int8_t *q, __half *s, const uint16_t *src, uint32_t in_dim, uint64_t n_blocks) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint2 *w = (const uint2 *)(src + b * 32u);
    float v[32];
    for (uint32_t i = 0; i < 8u; i++) {
        const uint2 p = w[i];
        v[i * 4u] = __uint_as_float(p.x << 16); v[i * 4u + 1u] = __uint_as_float(p.x & 0xffff0000u);
        v[i * 4u + 2u] = __uint_as_float(p.y << 16); v[i * 4u + 3u] = __uint_as_float(p.y & 0xffff0000u);
    }
    float amax = 0.0f;
    for (uint32_t i = 0; i < 32u; i++) amax = fmaxf(amax, fabsf(v[i]));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    s[b] = __float2half(d);
    uint32_t packed[8];
    for (uint32_t i = 0; i < 8u; i++) {
        uint32_t word = 0;
        for (uint32_t j = 0; j < 4u; j++) {
            const float x = v[i * 4u + j] * id;
            const float r = truncf(x + copysignf(0.5f, x));
            const int qi = (int)fminf(fmaxf(r, -127.0f), 127.0f);
            word |= ((uint32_t)qi & 0xffu) << (8u * j);
        }
        packed[i] = word;
    }
    ((uint4 *)q)[b * 2u] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    ((uint4 *)q)[b * 2u + 1u] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
    (void)in_dim;
}

/* One thread per block: split a GGML q8_0 block (2-byte scale, 32 quants). */
__global__ static void qwen35_q8_pack_q8_0_kernel(
        int8_t *q, __half *s, const uint8_t *src, uint64_t n_blocks) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t *blk = src + b * 34u;
    uint16_t bits;
    memcpy(&bits, blk, 2);
    s[b] = __ushort_as_half(bits);
    for (uint32_t i = 0; i < 32u; i++) q[b * 32u + i] = (int8_t)blk[2u + i];
}

extern "C" int ds4_gpu_qwen35_q8_pack(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint32_t    wtype,
        uint32_t    in_dim,
        uint32_t    out_dim,
        const char *name) {
    const uint64_t bytes = qwen35_weight_bytes(wtype, in_dim, out_dim);
    if (!model_map || bytes == 0u || in_dim % 32u != 0u || offset > model_size || bytes > model_size - offset ||
        (wtype != QWEN35_W_BF16 && wtype != QWEN35_W_Q8_0)) {
        return 0;
    }
    if (qwen35_q8_pack_find(model_map, offset)) return 1;
    const uint64_t n_blocks = (uint64_t)out_dim * (in_dim / 32u);
    qwen35_q8_pack p = { model_map, offset, in_dim, out_dim, NULL, NULL };
    if (cudaMalloc(&p.quants, n_blocks * 32u) != cudaSuccess ||
        cudaMalloc(&p.scales, n_blocks * sizeof(__half)) != cudaSuccess) {
        fprintf(stderr, "ds4: failed to allocate the packed q8 %s\n", name ? name : "weight");
        if (p.quants) (void)cudaFree(p.quants);
        (void)cudaGetLastError();
        return 0;
    }
    /* the source comes through host memory in row chunks: it may live in
     * nothing but the file mapping, since its span is never cached */
    const uint64_t row_bytes = bytes / out_dim;
    const uint64_t chunk_rows = (256ull << 20) / row_bytes > 0u ? (256ull << 20) / row_bytes : 1u;
    void *stage = NULL;
    if (cudaMalloc(&stage, chunk_rows * row_bytes) != cudaSuccess) {
        (void)cudaFree(p.quants); (void)cudaFree(p.scales); (void)cudaGetLastError();
        return 0;
    }
    const char *src = (const char *)model_map + offset;
    bool ok = true;
    for (uint64_t r0 = 0; ok && r0 < out_dim; r0 += chunk_rows) {
        const uint64_t rows = out_dim - r0 < chunk_rows ? out_dim - r0 : chunk_rows;
        const uint64_t blocks = rows * (in_dim / 32u);
        const uint64_t b0 = r0 * (in_dim / 32u);
        ok = cudaMemcpy(stage, src + r0 * row_bytes, rows * row_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!ok) break;
        const unsigned grid = (unsigned)((blocks + 255u) / 256u);
        if (wtype == QWEN35_W_BF16) {
            qwen35_q8_pack_bf16_kernel<<<grid, 256>>>(p.quants + b0 * 32u, p.scales + b0, (const uint16_t *)stage, in_dim, blocks);
        } else {
            qwen35_q8_pack_q8_0_kernel<<<grid, 256>>>(p.quants + b0 * 32u, p.scales + b0, (const uint8_t *)stage, blocks);
        }
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) ok = cudaDeviceSynchronize() == cudaSuccess;
    (void)cudaFree(stage);
    if (!ok) {
        fprintf(stderr, "ds4: packing q8 %s failed: %s\n", name ? name : "weight", cudaGetErrorString(cudaGetLastError()));
        (void)cudaFree(p.quants); (void)cudaFree(p.scales);
        return 0;
    }
    g_qwen35_q8_packs.push_back(p);
    return 1;
}

__device__ __forceinline__ void qwen35_unpack8(uint32_t a, uint32_t b, float *f) {
    f[0] = (float)(int8_t)(a & 0xffu); f[1] = (float)(int8_t)((a >> 8) & 0xffu);
    f[2] = (float)(int8_t)((a >> 16) & 0xffu); f[3] = (float)(int8_t)(a >> 24);
    f[4] = (float)(int8_t)(b & 0xffu); f[5] = (float)(int8_t)((b >> 8) & 0xffu);
    f[6] = (float)(int8_t)((b >> 16) & 0xffu); f[7] = (float)(int8_t)(b >> 24);
}

/* Decode matvec over the packed layout: two warps per output column split
 * the row, a lane takes 16 quants (one 16-byte load, half a block) per
 * step, so a warp step streams 512 weights, and the activation rows of a
 * speculative batch are dotted from the same weights.  Measured at the
 * device's plain-read bandwidth on every bypass shape. */
template <int N>
__global__ static void qwen35_matvec_q8p_rows_kernel(
        float *out, const int8_t *q, const __half *s, const float *x, uint32_t in_dim, uint32_t out_dim) {
    __shared__ float part[8][N];
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col = blockIdx.x * 4u + (warp >> 1u);
    const uint32_t kw = warp & 1u;
    if (col >= out_dim) return;
    const uint4 *qrow = (const uint4 *)(q + (uint64_t)col * in_dim);
    const __half *srow = s + (uint64_t)col * (in_dim / 32u);
    const uint32_t steps = in_dim / 16u;
    float sum[N];
#pragma unroll
    for (int r = 0; r < N; r++) sum[r] = 0.0f;
    for (uint32_t i = kw * 32u + lane; i < steps; i += 64u) {
        const uint4 wq = qrow[i];
        const float d = __half2float(srow[i >> 1]);
        float wf[16];
        qwen35_unpack8(wq.x, wq.y, wf);
        qwen35_unpack8(wq.z, wq.w, wf + 8);
#pragma unroll
        for (int r = 0; r < N; r++) {
            const float4 *xs = (const float4 *)(x + (uint64_t)r * in_dim + i * 16u);
            const float4 x0 = xs[0], x1 = xs[1], x2 = xs[2], x3 = xs[3];
            float acc = 0.0f;
            acc = fmaf(wf[0], x0.x, acc); acc = fmaf(wf[1], x0.y, acc); acc = fmaf(wf[2], x0.z, acc); acc = fmaf(wf[3], x0.w, acc);
            acc = fmaf(wf[4], x1.x, acc); acc = fmaf(wf[5], x1.y, acc); acc = fmaf(wf[6], x1.z, acc); acc = fmaf(wf[7], x1.w, acc);
            acc = fmaf(wf[8], x2.x, acc); acc = fmaf(wf[9], x2.y, acc); acc = fmaf(wf[10], x2.z, acc); acc = fmaf(wf[11], x2.w, acc);
            acc = fmaf(wf[12], x3.x, acc); acc = fmaf(wf[13], x3.y, acc); acc = fmaf(wf[14], x3.z, acc); acc = fmaf(wf[15], x3.w, acc);
            sum[r] = fmaf(d, acc, sum[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < N; r++) {
        const float total = warp_sum_f32(sum[r]);
        if (lane == 0u) part[warp][r] = total;
    }
    __syncthreads();
    if (kw == 0u && lane == 0u) {
#pragma unroll
        for (int r = 0; r < N; r++) out[(uint64_t)r * out_dim + col] = part[warp][r] + part[warp + 1u][r];
    }
}

static void qwen35_matvec_q8p_rows(
        float *out, const qwen35_q8_pack *p, const float *x, uint32_t n_tok, cudaStream_t stream) {
    const dim3 grid((p->out_dim + 3u) / 4u, 1u, 1u);
    const int8_t *q = p->quants;
    const __half *s = p->scales;
    switch (n_tok) {
    case 1: qwen35_matvec_q8p_rows_kernel<1><<<grid, 256, 0, stream>>>(out, q, s, x, p->in_dim, p->out_dim); break;
    case 2: qwen35_matvec_q8p_rows_kernel<2><<<grid, 256, 0, stream>>>(out, q, s, x, p->in_dim, p->out_dim); break;
    case 3: qwen35_matvec_q8p_rows_kernel<3><<<grid, 256, 0, stream>>>(out, q, s, x, p->in_dim, p->out_dim); break;
    case 4: qwen35_matvec_q8p_rows_kernel<4><<<grid, 256, 0, stream>>>(out, q, s, x, p->in_dim, p->out_dim); break;
    case 5: qwen35_matvec_q8p_rows_kernel<5><<<grid, 256, 0, stream>>>(out, q, s, x, p->in_dim, p->out_dim); break;
    case 6: qwen35_matvec_q8p_rows_kernel<6><<<grid, 256, 0, stream>>>(out, q, s, x, p->in_dim, p->out_dim); break;
    case 7: qwen35_matvec_q8p_rows_kernel<7><<<grid, 256, 0, stream>>>(out, q, s, x, p->in_dim, p->out_dim); break;
    default: qwen35_matvec_q8p_rows_kernel<8><<<grid, 256, 0, stream>>>(out, q, s, x, p->in_dim, p->out_dim); break;
    }
}

/* Prefill dequantises a slice of rows back to bf16 for the tensor-core
 * GEMM: the same bf16 x bf16 recipe as an unquantised bypass weight, with
 * the weights carrying q8_0's rounding. */
__global__ static void qwen35_q8p_to_bf16_kernel(
        __nv_bfloat16 *dst, const int8_t *q, const __half *s, uint64_t n_blocks) {
    const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float d = __half2float(s[b]);
    const uint4 *src = (const uint4 *)(q + b * 32u);
    for (uint32_t h = 0; h < 2u; h++) {
        const uint4 w = src[h];
        float f[16];
        qwen35_unpack8(w.x, w.y, f);
        qwen35_unpack8(w.z, w.w, f + 8);
        __nv_bfloat162 *o = (__nv_bfloat162 *)(dst + b * 32u + h * 16u);
        for (uint32_t i = 0; i < 8u; i++) o[i] = __floats2bfloat162_rn(f[2u * i] * d, f[2u * i + 1u] * d);
    }
}

static int qwen35_matmul_q8p(
        float *out, const qwen35_q8_pack *p, const float *x, const __nv_bfloat16 *x_bf16,
        uint32_t n_tok, int tier, cudaStream_t stream) {
    if (n_tok <= 8u) {
        qwen35_matvec_q8p_rows(out, p, x, n_tok, stream);
        return cuda_ok(cudaGetLastError(), "Qwen packed q8 matvec launch");
    }
    if (!g_cublas_ready) return 0;
    const __nv_bfloat16 *a = x_bf16;
    if (!a) {
        const uint64_t n = (uint64_t)n_tok * p->in_dim;
        __nv_bfloat16 *buf = (__nv_bfloat16 *)cuda_tmp_alloc_on(tier, n * sizeof(__nv_bfloat16), "Qwen bf16 activations");
        if (!buf) return 0;
        qwen35_to_bf16(buf, x, n, stream);
        if (!cuda_ok(cudaGetLastError(), "Qwen bf16 convert launch")) return 0;
        a = buf;
    }
    /* row slices of at most 64 MiB of bf16 weights through one scratch */
    static __nv_bfloat16 *scratch = NULL;
    static uint64_t scratch_elems = 0;
    const uint32_t slice_rows = p->in_dim ? (uint32_t)((32ull << 20) / p->in_dim) : 0u;
    const uint64_t need = (uint64_t)(slice_rows < p->out_dim ? slice_rows : p->out_dim) * p->in_dim;
    if (slice_rows == 0u) return 0;
    if (scratch_elems < need) {
        if (scratch) (void)cudaFree(scratch);
        scratch = NULL;
        scratch_elems = 0;
        if (cudaMalloc(&scratch, need * sizeof(__nv_bfloat16)) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
        scratch_elems = need;
    }
    const float alpha = 1.0f, beta = 0.0f;
    for (uint32_t c0 = 0; c0 < p->out_dim; c0 += slice_rows) {
        const uint32_t rows = p->out_dim - c0 < slice_rows ? p->out_dim - c0 : slice_rows;
        const uint64_t b0 = (uint64_t)c0 * (p->in_dim / 32u);
        const uint64_t blocks = (uint64_t)rows * (p->in_dim / 32u);
        qwen35_q8p_to_bf16_kernel<<<(unsigned)((blocks + 255u) / 256u), 256, 0, stream>>>(
            scratch, p->quants + b0 * 32u, p->scales + b0, blocks);
        if (!cuda_ok(cudaGetLastError(), "Qwen packed q8 dequant launch")) return 0;
        const cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(tier), CUBLAS_OP_T, CUBLAS_OP_N,
                                               (int)rows, (int)n_tok, (int)p->in_dim,
                                               &alpha, scratch, CUDA_R_16BF, (int)p->in_dim, a, CUDA_R_16BF, (int)p->in_dim,
                                               &beta, out + c0, CUDA_R_32F, (int)p->out_dim,
                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "Qwen packed q8 GEMM")) return 0;
    }
    return 1;
}

/* out[n_tok][out_dim] = x[n_tok][in_dim] W^T times the global scale, for a
 * weight of any storage type the loader accepts.  Decode-sized batches use
 * one warp or block per output column on f32 activations; larger ones the
 * tensor cores: BF16 weights through cuBLAS on bf16 activations (x_bf16
 * when the caller rounded x already, else rounded here), NVFP4 and F32
 * weights on kernels exact to f32 rounding. */
extern "C" int ds4_gpu_qwen35_matmul(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              wtype,
        float                 scale,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *x_bf16,
        uint32_t              n_tok) {
    const uint64_t weight_bytes = qwen35_weight_bytes(wtype, in_dim, out_dim);
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim % 64u != 0u || weight_bytes == 0u || weight_offset > model_size ||
        weight_bytes > model_size - weight_offset ||
        x->bytes < (uint64_t)n_tok * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tok * out_dim * sizeof(float) ||
        (x_bf16 && x_bf16->bytes < (uint64_t)n_tok * in_dim * sizeof(__nv_bfloat16))) {
        return 0;
    }
    /* a weight the loader packed to q8 lives only in its packed buffers */
    if (const qwen35_q8_pack *pack = qwen35_q8_pack_find(model_map, weight_offset)) {
        if (pack->in_dim != in_dim || pack->out_dim != out_dim || scale != 1.0f) {
            fprintf(stderr, "ds4: packed q8 weight at offset %llu does not match its use (%u x %u, scale %g)\n",
                    (unsigned long long)weight_offset, in_dim, out_dim, (double)scale);
            return 0;
        }
        return qwen35_matmul_q8p((float *)out->ptr, pack, (const float *)x->ptr,
                                 x_bf16 ? (const __nv_bfloat16 *)x_bf16->ptr : NULL, n_tok,
                                 ds4_tensor_device_idx(out), cuda_decode_stream());
    }
    const char *w = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes,
                                            ds4_tensor_device_idx(out), "Qwen weight");
    if (!w) return 0;
    cudaStream_t stream = cuda_decode_stream();
    float *o = (float *)out->ptr;
    const float *xp = (const float *)x->ptr;
    /* q8_0 weights (the int8 lm_head and GDN projections) get their own
     * kernels rather than the generic quantized matmul: that one allocates
     * from the mmq pool, which CUDA forbids inside the captured decode
     * graphs.  It also has no scale parameter, so a q8_0 weight must not
     * carry one (the converter writes none). */
    if (wtype == QWEN35_W_Q8_0) {
        if (scale != 1.0f) {
            fprintf(stderr, "ds4: q8_0 Qwen weight at offset %llu carries scale %g\n",
                    (unsigned long long)weight_offset, (double)scale);
            return 0;
        }
        if (in_dim % 4u != 0u) {
            fprintf(stderr, "ds4: q8_0 Qwen weight in_dim %u is not a multiple of 4\n", in_dim);
            return 0;
        }
        if (n_tok <= 8u) {
            /* Decode reads the weight straight from the q8_0 blocks: the
             * generic matmul below cannot, because it allocates from the mmq
             * pool and these rows are replayed inside captured graphs. */
            qwen35_matvec_q8_0_rows(o, (const uint8_t *)w, xp, in_dim, out_dim, n_tok, stream);
            return cuda_ok(cudaGetLastError(), "Qwen q8_0 matvec launch");
        }
        /* Prefill goes through the generic quantized matmul, which reads the
         * q8_0 weight exactly once.  It takes f32 activations: bring back the
         * bf16 copy when that is all the caller produced (Flash-Next keeps the
         * token-mixer output in bf16). */
        const ds4_gpu_tensor *act = x;
        if (x_bf16) {
            ds4_gpu_tensor *f32 = qwen35_f32_activation_scratch((uint64_t)n_tok * in_dim);
            if (!f32) {
                fprintf(stderr, "ds4: Qwen q8_0 activation scratch alloc failed (%u x %u)\n",
                        n_tok, in_dim);
                return 0;
            }
            qwen35_bf16_to_f32((float *)f32->ptr, (const __nv_bfloat16 *)x_bf16->ptr,
                               (uint64_t)n_tok * in_dim, stream);
            if (!cuda_ok(cudaGetLastError(), "Qwen bf16 activation convert")) return 0;
            act = f32;
        }
        const int mmq = ds4_gpu_matmul_quant_tensor(out, model_map, model_size, weight_offset, wtype,
                                                    in_dim, out_dim, act, n_tok);
        if (!mmq) {
            fprintf(stderr, "ds4: Qwen q8_0 prefill matmul failed in=%u out=%u n_tok=%u offset=%llu\n",
                    in_dim, out_dim, n_tok, (unsigned long long)weight_offset);
        }
        return mmq;
    }
    if (n_tok > 8u) {
        if (wtype == QWEN35_W_F32) {
            /* exact f32: the handle allows TF32, which the pedantic compute
             * type overrides; the router's top-k depends on it */
            if (g_cublas_ready) {
                const float beta = 0.0f;
                cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(ds4_tensor_device_idx(out)),
                                                 CUBLAS_OP_T, CUBLAS_OP_N, (int)out_dim, (int)n_tok, (int)in_dim,
                                                 &scale, w, CUDA_R_32F, (int)in_dim, xp, CUDA_R_32F, (int)in_dim,
                                                 &beta, o, CUDA_R_32F, (int)out_dim,
                                                 CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
                return cublas_ok(st, "Qwen f32 GEMM");
            }
            const dim3 grid((n_tok + 63u) / 64u, (out_dim + 63u) / 64u, 1u);
            qwen35_gemm_f32_kernel<<<grid, 256, 0, stream>>>(o, (const uint8_t *)w, xp, in_dim, out_dim, n_tok, scale);
            return cuda_ok(cudaGetLastError(), "Qwen f32 GEMM launch");
        }
        if (wtype == QWEN35_W_BF16 && g_cublas_ready) {
            return qwen35_cublas_bf16(o, (const uint16_t *)w, xp, x_bf16 ? (const __nv_bfloat16 *)x_bf16->ptr : NULL,
                                      in_dim, out_dim, n_tok, scale, ds4_tensor_device_idx(out), stream);
        }
        static bool ready = false;
        if (!qwen35_mma_smem_ready((const void *)qwen35_gemm_mma_kernel, &ready)) return 0;
        const dim3 grid((n_tok + QWEN35_MMA_ROWS - 1u) / QWEN35_MMA_ROWS, (out_dim + QWEN35_MMA_COLS - 1u) / QWEN35_MMA_COLS, 1u);
        qwen35_gemm_mma_kernel<<<grid, 256, QWEN35_MMA_SMEM, stream>>>(o, (const uint8_t *)w, wtype, xp,
                                                                       in_dim, out_dim, n_tok, scale);
        return cuda_ok(cudaGetLastError(), "Qwen tensor-core GEMM launch");
    }
    if (wtype == QWEN35_W_NVFP4) {
        const dim3 grid((out_dim + 7u) / 8u, n_tok, 1u);
        qwen35_nvfp4_matvec_kernel<<<grid, 256, 0, stream>>>(o, (const uint8_t *)w, xp,
                                                             in_dim, out_dim, n_tok, scale);
        return cuda_ok(cudaGetLastError(), "Qwen NVFP4 matvec launch");
    }
    /* The BF16 matvec (9 GiB of bypass weights per Flash-Next decode step)
     * runs at the memory bandwidth; every weight is read once for all the
     * rows of a speculative batch. */
    if (wtype == QWEN35_W_BF16 && in_dim % 4u == 0u) {
        qwen35_matvec_bf16_rows(o, (const uint16_t *)w, xp, in_dim, out_dim, n_tok, stream);
    } else if (wtype == QWEN35_W_BF16) {
        const dim3 grid((out_dim + 7u) / 8u, n_tok, 1u);
        glm53_matvec_bf16_f32_kernel<<<grid, 256, 0, stream>>>(o, (const uint16_t *)w, xp, in_dim, out_dim);
    } else {
        matmul_f32_kernel<<<dim3(out_dim, n_tok, 1u), 256, 0, stream>>>(o, (const float *)w, xp,
                                                                        in_dim, out_dim, n_tok);
    }
    if (scale != 1.0f) {
        const uint64_t n = (uint64_t)n_tok * out_dim;
        qwen35_scale_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(o, n, scale);
    }
    return cuda_ok(cudaGetLastError(), "Qwen matvec launch");
}

/* ---- Row table ------------------------------------------------------------
 * Every kernel below that reads or writes per-session state (the GDN
 * histories and recurrent state, the K/V and block-key caches, the PLE
 * window) takes a ds4_qwen_batch_slot row table instead of the state
 * buffers themselves, so the captured decode islands serve every session
 * sharing the scratch (ds4_gpu_mgpu.h).  A pass is sequential (one
 * session, its n tokens in order, row 0) or batched (n sessions, one token
 * each, row t); `tl` below is a token's index within its own session's
 * pass, so a batched token sees no earlier tokens and only its history. */

static const ds4_qwen_batch_slot *qwen35_slots(
        const ds4_gpu_tensor *table, uint32_t first, uint32_t n_tokens, uint32_t batched) {
    const uint64_t rows = batched ? n_tokens : 1u;
    if (!table || n_tokens == 0u || (batched && n_tokens > DS4_QWEN_BATCH_ROWS) ||
        table->bytes < ((uint64_t)first + rows) * sizeof(ds4_qwen_batch_slot)) {
        return NULL;
    }
    return (const ds4_qwen_batch_slot *)table->ptr + first;
}

/* Slide a history (row.p0, n_hist rows of dim) forward over the pass's rows:
 * the one history of a sequential pass, or each row's own by its single
 * token.  Each thread owns one channel and walks rows in increasing order,
 * so the in-place shift is hazard free. */
__global__ static void qwen35_history_slide_kernel(
        const ds4_qwen_batch_slot *slots, const float *rows, uint32_t dim, uint32_t n_hist,
        uint32_t n_tokens, uint32_t batched) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (c >= dim) return;
    float *hist = (float *)slots[row].p0;
    const float *src_rows = rows + (uint64_t)(batched ? row : 0u) * dim;
    const uint32_t n = batched ? 1u : n_tokens;
    for (uint32_t w = 0; w < n_hist; w++) {
        const int32_t src = (int32_t)n - (int32_t)n_hist + (int32_t)w;
        hist[(uint64_t)w * dim + c] = src >= 0
            ? src_rows[(uint64_t)src * dim + c]
            : hist[(uint64_t)(uint32_t)(src + (int32_t)n_hist) * dim + c];
    }
}

static void qwen35_history_slide(
        const ds4_qwen_batch_slot *slots, const float *rows, uint32_t dim, uint32_t n_hist,
        uint32_t n_tokens, uint32_t batched, cudaStream_t stream) {
    qwen35_history_slide_kernel<<<dim3((dim + 255u) / 256u, batched ? n_tokens : 1u, 1u), 256, 0, stream>>>(
        slots, rows, dim, n_hist, n_tokens, batched);
}

/* The history a sequential pass leaves after each of its tokens (for
 * speculative verification, whose rejected tail must be undone): snap[t] is
 * the last n_hist rows of the history followed by rows 0..t, the state
 * qwen35_history_slide would leave after t + 1 tokens. */
__global__ static void qwen35_history_snapshots_kernel(
        float *snap, const ds4_qwen_batch_slot *slots, const float *rows, uint32_t dim, uint32_t n_hist, uint32_t n_tokens) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (c >= dim || t >= n_tokens) return;
    const float *hist = (const float *)slots[0].p0;
    float *dst = snap + (uint64_t)t * n_hist * dim;
    for (uint32_t w = 0; w < n_hist; w++) {
        const int32_t src = (int32_t)t + 1 - (int32_t)n_hist + (int32_t)w;
        dst[(uint64_t)w * dim + c] = src >= 0
            ? rows[(uint64_t)src * dim + c]
            : hist[(uint64_t)(uint32_t)(src + (int32_t)n_hist) * dim + c];
    }
}

static int qwen35_history_snapshots(
        const ds4_gpu_tensor *snap, const ds4_qwen_batch_slot *slots, const float *rows, uint32_t dim, uint32_t n_hist,
        uint32_t n_tokens, cudaStream_t stream) {
    if (!snap) return 1;
    if (snap->bytes < (uint64_t)n_tokens * n_hist * dim * sizeof(float)) return 0;
    qwen35_history_snapshots_kernel<<<dim3((dim + 255u) / 256u, n_tokens, 1u), 256, 0, stream>>>(
        (float *)snap->ptr, slots, rows, dim, n_hist, n_tokens);
    return 1;
}

/* ---- Gated DeltaNet ------------------------------------------------------ */

/* Causal conv + SiLU over the qkv projection, then L2-normalise the q and k
 * heads.  One block per (token, head slot); taps older than the pass read
 * the conv history.  Slots: n_k q heads, n_k k heads, n_v v heads. */
__global__ static void qwen35_gdn_conv_kernel(
        float *mixed, const float *qkv, const ds4_qwen_batch_slot *slots, const float *conv_w,
        uint32_t n_k, uint32_t n_v, uint32_t n_conv, uint32_t n_tokens, uint32_t batched, float eps) {
    __shared__ float scratch[32];
    const uint32_t t = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t hd = QWEN35_CUDA_GDN_DIM;
    const uint32_t conv_dim = (2u * n_k + n_v) * hd;
    const uint32_t c = slot * hd + tid;
    if (t >= n_tokens || c >= conv_dim) return;
    const float *hist = (const float *)slots[batched ? t : 0u].p0;
    const uint32_t tl = batched ? 0u : t;
    const float *taps = conv_w + (uint64_t)c * n_conv;
    float acc = taps[n_conv - 1u] * qkv[(uint64_t)t * conv_dim + c];
    for (uint32_t w = 0; w + 1u < n_conv; w++) {
        const int32_t src = (int32_t)tl + (int32_t)w - (int32_t)(n_conv - 1u);
        const float v = src >= 0
            ? qkv[(uint64_t)(t - tl + (uint32_t)src) * conv_dim + c]
            : hist[(uint64_t)(uint32_t)(src + (int32_t)(n_conv - 1u)) * conv_dim + c];
        acc = fmaf(taps[w], v, acc);
    }
    float val = qwen35_cuda_silu(acc);
    if (slot < 2u * n_k) {
        const float total = qwen35_cuda_block_sum(val * val, scratch);
        val *= rsqrtf(total + eps);
    }
    mixed[(uint64_t)t * conv_dim + c] = val;
}

/* Recurrent delta rule.  Block per (v head, 4 value rows, row of a batched
 * pass); each warp owns one value row of S with four key columns per lane,
 * and walks the tokens of its row's pass. */
/* Warp-sum eight values at once by halving the set at every shuffle
 * distance: after xor 16 each lane keeps four columns (its half), after
 * xor 8 two, then one, which the last two steps finish.  Nine shuffles
 * instead of forty; lane l ends with the total of column (l / 4) % 8. */
__device__ __forceinline__ float qwen35_gdn_reduce8(const float v[8], uint32_t lane) {
    float a[4], b[2];
    const bool hi16 = (lane & 16u) != 0u;
    for (uint32_t i = 0; i < 4u; i++) {
        const float send = hi16 ? v[i] : v[i + 4u];
        a[i] = (hi16 ? v[i + 4u] : v[i]) + __shfl_xor_sync(0xffffffffu, send, 16u);
    }
    const bool hi8 = (lane & 8u) != 0u;
    for (uint32_t i = 0; i < 2u; i++) {
        const float send = hi8 ? a[i] : a[i + 2u];
        b[i] = (hi8 ? a[i + 2u] : a[i]) + __shfl_xor_sync(0xffffffffu, send, 8u);
    }
    const bool hi4 = (lane & 4u) != 0u;
    float s = (hi4 ? b[1] : b[0]) + __shfl_xor_sync(0xffffffffu, hi4 ? b[0] : b[1], 4u);
    s += __shfl_xor_sync(0xffffffffu, s, 2u);
    s += __shfl_xor_sync(0xffffffffu, s, 1u);
    return s;
}

__global__ static void qwen35_gdn_recurrence_kernel(
        float *o, const ds4_qwen_batch_slot *slots, float *snap, const float *mixed, const float *alpha, const float *beta,
        const float *a_neg, const float *dt_bias,
        uint32_t n_k, uint32_t n_v, uint32_t n_tokens, uint32_t batched, float q_scale) {
    const uint32_t hd = QWEN35_CUDA_GDN_DIM;
    const uint32_t hv = blockIdx.x;
    const uint32_t v0 = blockIdx.y * 32u + (threadIdx.x >> 5u) * 8u;   /* this warp's eight value columns */
    const uint32_t lane = threadIdx.x & 31u;
    if (hv >= n_v || v0 >= hd) return;
    const uint32_t hk = hv % n_k;
    const uint32_t conv_dim = (2u * n_k + n_v) * hd;
    /* a batched row walks its single token over its own state */
    const uint32_t row = blockIdx.z;
    float *state = (float *)slots[row].p1;
    const uint32_t tok0 = batched ? row : 0u;
    if (batched) n_tokens = 1u;
    mixed += (uint64_t)tok0 * conv_dim;
    alpha += (uint64_t)tok0 * n_v;
    beta += (uint64_t)tok0 * n_v;
    o += (uint64_t)tok0 * n_v * hd;
    const uint32_t k0 = lane * 4u;
    const float a_head = a_neg[hv];
    const float dt_head = dt_bias[hv];
    /* after qwen35_gdn_reduce8 this lane holds the total of column `mine` */
    const uint32_t mine = (lane >> 2u) & 7u;
    float4 h[8];
    for (uint32_t c = 0; c < 8u; c++) h[c] = *(const float4 *)(state + ((uint64_t)hv * hd + v0 + c) * hd + k0);
    /* The token chain is instruction-bound, so the next token's inputs are
     * fetched a step ahead and eight columns share every per-token cost. */
    float4 q_next = *(const float4 *)(mixed + hk * hd + k0);
    float4 k_next = *(const float4 *)(mixed + (n_k + hk) * hd + k0);
    float4 va_next = *(const float4 *)(mixed + (2u * n_k + hv) * hd + v0);
    float4 vb_next = *(const float4 *)(mixed + (2u * n_k + hv) * hd + v0 + 4u);
    float alpha_next = alpha[hv];
    float beta_next = beta[hv];
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float4 q4 = q_next, k4 = k_next, va = va_next, vb = vb_next;
        const float alpha_t = alpha_next, beta_t = beta_next;
        if (t + 1u < n_tokens) {
            const float *row = mixed + (uint64_t)(t + 1u) * conv_dim;
            q_next = *(const float4 *)(row + hk * hd + k0);
            k_next = *(const float4 *)(row + (n_k + hk) * hd + k0);
            va_next = *(const float4 *)(row + (2u * n_k + hv) * hd + v0);
            vb_next = *(const float4 *)(row + (2u * n_k + hv) * hd + v0 + 4u);
            alpha_next = alpha[(uint64_t)(t + 1u) * n_v + hv];
            beta_next = beta[(uint64_t)(t + 1u) * n_v + hv];
        }
        const float decay = expf(a_head * qwen35_cuda_softplus(alpha_t + dt_head));
        const float b = qwen35_cuda_sigmoid(beta_t);
        const float vval[8] = { va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w };
        float sk[8], res[8];
        for (uint32_t c = 0; c < 8u; c++) {
            h[c].x *= decay; h[c].y *= decay; h[c].z *= decay; h[c].w *= decay;
            sk[c] = dot4_f32(h[c], k4);
        }
        const float sk_mine = qwen35_gdn_reduce8(sk, lane);
        for (uint32_t c = 0; c < 8u; c++) {
            const float delta = (vval[c] - __shfl_sync(0xffffffffu, sk_mine, c * 4u)) * b;
            h[c].x = fmaf(k4.x, delta, h[c].x);
            h[c].y = fmaf(k4.y, delta, h[c].y);
            h[c].z = fmaf(k4.z, delta, h[c].z);
            h[c].w = fmaf(k4.w, delta, h[c].w);
            res[c] = dot4_f32(h[c], q4);
        }
        const float res_mine = qwen35_gdn_reduce8(res, lane);
        if ((lane & 3u) == 0u) o[(uint64_t)t * n_v * hd + hv * hd + v0 + mine] = res_mine * q_scale;
        if (snap) {   /* the state after this token, for speculative rollback */
            float4 *dst = (float4 *)(snap + (uint64_t)t * n_v * hd * hd);
            for (uint32_t c = 0; c < 8u; c++) dst[(((uint64_t)hv * hd + v0 + c) * hd + k0) / 4u] = h[c];
        }
    }
    for (uint32_t c = 0; c < 8u; c++) *(float4 *)(state + ((uint64_t)hv * hd + v0 + c) * hd + k0) = h[c];
}

/* RMSNormGated per head: normalise, scale, then multiply by the output gate,
 * SiLU(z) for Qwen3.5 and sigmoid(z) for Flash-Next. */
__global__ static void qwen35_gdn_norm_gate_kernel(
        float *o, __nv_bfloat16 *o_bf16, const float *z, const float *norm_w, uint32_t n_v, uint32_t n_tokens,
        uint32_t sigmoid_gate, float eps) {
    __shared__ float scratch[32];
    const uint32_t hd = QWEN35_CUDA_GDN_DIM;
    const uint32_t t = blockIdx.x;
    const uint32_t hv = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || hv >= n_v) return;
    const uint64_t idx = (uint64_t)t * n_v * hd + hv * hd + tid;
    const float raw = o[idx];
    const float total = qwen35_cuda_block_sum(raw * raw, scratch);
    const float scale = rsqrtf(total / (float)hd + eps);
    const float gate = sigmoid_gate ? qwen35_cuda_sigmoid(z[idx]) : qwen35_cuda_silu(z[idx]);
    const float v = raw * scale * norm_w[tid] * gate;
    if (o_bf16) o_bf16[idx] = __float2bfloat16(v);   /* the prefill out projection's operand */
    else o[idx] = v;
}

extern "C" int ds4_gpu_qwen35_gdn(
        ds4_gpu_tensor       *out,          /* [n_tok][v_dim] */
        ds4_gpu_tensor       *out_bf16,     /* optional: the output goes there in bf16 instead (prefill) */
        ds4_gpu_tensor       *mixed,        /* [n_tok][conv_dim] scratch */
        const ds4_gpu_tensor *slots,        /* row table: p0 the conv history [n_conv-1][conv_dim], p1 the state [n_v][hd][hd] */
        uint32_t              slot0,
        const ds4_gpu_tensor *qkv,          /* [n_tok][conv_dim] */
        const ds4_gpu_tensor *z,            /* [n_tok][v_dim] */
        const ds4_gpu_tensor *alpha,        /* [n_tok][n_v] */
        const ds4_gpu_tensor *beta,         /* [n_tok][n_v] */
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              conv_w_offset,
        uint64_t              a_offset,
        uint64_t              dt_bias_offset,
        uint64_t              norm_offset,
        uint32_t              n_k,
        uint32_t              n_v,
        uint32_t              n_conv,
        uint32_t              n_tokens,
        int                   batched,      /* one session per row rather than one session's n_tok tokens */
        int                   sigmoid_gate,
        float                 eps,
        const ds4_gpu_tensor *snap_ssm,     /* optional (sequential): the state after each token, [n_tok][n_v][hd][hd] */
        const ds4_gpu_tensor *snap_conv) {  /* optional (sequential): the conv history after each token */
    const uint32_t hd = QWEN35_CUDA_GDN_DIM;
    const ds4_qwen_batch_slot *rows = qwen35_slots(slots, slot0, n_tokens, batched != 0);
    if (!out || !mixed || !rows || !qkv || !z || !alpha || !beta ||
        !model_map || n_k == 0u || n_v == 0u || n_v % n_k != 0u || n_conv < 2u ||
        (batched && (snap_ssm || snap_conv))) {
        return 0;
    }
    const uint64_t conv_dim = (uint64_t)(2u * n_k + n_v) * hd;
    const uint64_t v_dim = (uint64_t)n_v * hd;
    if (qkv->bytes < n_tokens * conv_dim * sizeof(float) ||
        mixed->bytes < n_tokens * conv_dim * sizeof(float) ||
        z->bytes < n_tokens * v_dim * sizeof(float) ||
        out->bytes < n_tokens * v_dim * sizeof(float) ||
        alpha->bytes < (uint64_t)n_tokens * n_v * sizeof(float) ||
        beta->bytes < (uint64_t)n_tokens * n_v * sizeof(float)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(out);
    const float *conv_w = glm53_cuda_weight_f32(model_map, model_size, conv_w_offset,
                                                conv_dim * n_conv, tier, "GDN conv");
    const float *a_neg = glm53_cuda_weight_f32(model_map, model_size, a_offset,
                                               n_v, tier, "GDN A");
    const float *dt_bias = glm53_cuda_weight_f32(model_map, model_size, dt_bias_offset,
                                                 n_v, tier, "GDN dt bias");
    const float *norm_w = glm53_cuda_weight_f32(model_map, model_size, norm_offset,
                                                hd, tier, "GDN norm");
    if (!conv_w || !a_neg || !dt_bias || !norm_w) return 0;
    cudaStream_t stream = cuda_decode_stream();

    const uint32_t b = batched != 0;
    qwen35_gdn_conv_kernel<<<dim3(n_tokens, 2u * n_k + n_v, 1u), hd, 0, stream>>>(
        (float *)mixed->ptr, (const float *)qkv->ptr, rows,
        conv_w, n_k, n_v, n_conv, n_tokens, b, eps);
    if (!qwen35_history_snapshots(snap_conv, rows, (const float *)qkv->ptr,
                                  (uint32_t)conv_dim, n_conv - 1u, n_tokens, stream)) return 0;
    if (snap_ssm && snap_ssm->bytes < (uint64_t)n_tokens * v_dim * hd * sizeof(float)) return 0;
    qwen35_history_slide(rows, (const float *)qkv->ptr, (uint32_t)conv_dim, n_conv - 1u, n_tokens, b, stream);
    qwen35_gdn_recurrence_kernel<<<dim3(n_v, hd / 32u, b ? n_tokens : 1u), 128, 0, stream>>>(
        (float *)out->ptr, rows, snap_ssm ? (float *)snap_ssm->ptr : NULL, (const float *)mixed->ptr,
        (const float *)alpha->ptr, (const float *)beta->ptr, a_neg, dt_bias,
        n_k, n_v, n_tokens, b, rsqrtf((float)hd));
    if (out_bf16 && out_bf16->bytes < (uint64_t)n_tokens * n_v * hd * sizeof(__nv_bfloat16)) return 0;
    qwen35_gdn_norm_gate_kernel<<<dim3(n_tokens, n_v, 1u), hd, 0, stream>>>(
        (float *)out->ptr, out_bf16 ? (__nv_bfloat16 *)out_bf16->ptr : NULL, (const float *)z->ptr, norm_w,
        n_v, n_tokens, sigmoid_gate != 0, eps);
    return cuda_ok(cudaGetLastError(), "Qwen3.5 GDN launch");
}

/* ---- Gated GQA attention ------------------------------------------------- */

/* f32 into a bf16 high part and a bf16 remainder (both round-to-nearest):
 * the split operands of the tensor-core paths, hi + lo within 2^-16 of x. */
__device__ __forceinline__ void qwen35_split_f32(float x, __nv_bfloat16 *hi, __nv_bfloat16 *lo) {
    const __nv_bfloat16 h = __float2bfloat16(x);
    *hi = h;
    *lo = __float2bfloat16(x - __bfloat162float(h));
}

/* Per (token, head slot): RMS-normalise and RoPE the query heads in place
 * (the q buffer holds [query | gate] per head), do the same for the key heads
 * and store them in the cache, copy the value heads into the cache.  The
 * caches are bf16 (the CPU reference rounds its rows the same way).  With
 * qh/ql (prefill) the finished queries are also written split into bf16 for
 * the tensor-core attention. */
/* The paged caches (ds4_gpu_mgpu.h): the page of a position, then the
 * layer's K or V row at `off` (the slot's p0 or p1) within it, or its
 * block key.  The two places the layout is spelled out. */
__device__ __forceinline__ __nv_bfloat16 *qwen35_page(const ds4_qwen_batch_slot &row, uint32_t page) {
    return (__nv_bfloat16 *)((const uint64_t *)row.pages)[page];
}

__device__ __forceinline__ __nv_bfloat16 *qwen35_kv_ptr(const ds4_qwen_batch_slot &row, uint64_t off, uint32_t pos, uint64_t kv_stride) {
    return qwen35_page(row, pos / DS4_QWEN_PAGE_POSITIONS) + off + (uint64_t)(pos % DS4_QWEN_PAGE_POSITIONS) * kv_stride;
}

__device__ __forceinline__ __nv_bfloat16 *qwen35_bkey_ptr(const ds4_qwen_batch_slot &row, uint32_t block, uint32_t r, uint32_t d) {
    const uint32_t per_page = DS4_QWEN_PAGE_POSITIONS / r;
    return qwen35_page(row, block / per_page) + row.p1 + (uint64_t)(block % per_page) * d;
}

__global__ static void qwen35_attn_prepare_kernel(
        float *qg, const ds4_qwen_batch_slot *slots, const float *k, const float *v,
        const float *q_norm, const float *k_norm, __nv_bfloat16 *qh, __nv_bfloat16 *ql,
        uint32_t n_head, uint32_t n_kv, uint32_t hd, uint32_t n_rot,
        uint32_t n_tokens, uint32_t batched, float freq_base, float eps) {
    __shared__ float scratch[32];
    const uint32_t t = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= hd) return;
    const ds4_qwen_batch_slot row = slots[batched ? t : 0u];
    const uint32_t pos = batched ? row.pos : row.pos + t;
    float *head;
    const float *norm_w;
    __nv_bfloat16 *dst = NULL;
    if (slot < n_head) {
        head = qg + ((uint64_t)t * n_head + slot) * 2u * hd;
        norm_w = q_norm;
    } else if (slot < n_head + n_kv) {
        const uint32_t h = slot - n_head;
        head = (float *)k + ((uint64_t)t * n_kv + h) * hd;
        norm_w = k_norm;
        dst = qwen35_kv_ptr(row, row.p0, pos, (uint64_t)n_kv * hd) + (uint64_t)h * hd;
    } else {
        const uint32_t h = slot - n_head - n_kv;
        qwen35_kv_ptr(row, row.p1, pos, (uint64_t)n_kv * hd)[(uint64_t)h * hd + tid] =
            __float2bfloat16(v[((uint64_t)t * n_kv + h) * hd + tid]);
        return;
    }
    const float x = head[tid];
    const float total = qwen35_cuda_block_sum(x * x, scratch);
    head[tid] = x * rsqrtf(total / (float)hd + eps) * norm_w[tid];
    __syncthreads();
    const uint32_t half = n_rot / 2u;
    if (tid < half) {
        const float theta = (float)pos *
            powf(freq_base, -(float)(2u * tid) / (float)n_rot);
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = head[tid];
        const float x1 = head[tid + half];
        head[tid] = x0 * c - x1 * s;
        head[tid + half] = x0 * s + x1 * c;
    }
    if (dst || qh) __syncthreads();
    if (dst) dst[tid] = __float2bfloat16(head[tid]);
    if (qh && slot < n_head) {
        const uint64_t i = ((uint64_t)t * n_head + slot) * hd + tid;
        qwen35_split_f32(head[tid], qh + i, ql + i);
    }
}

/* Per (token, head, key split): scores against a key range, partial softmax
 * (max, sum) and unnormalised weighted values.  The keys are the causal
 * prefix, or with `sel` the token's own list of n_sel cells (QSA), gathered
 * from the cache by index.  With one split the block finalises directly,
 * including the sigmoid output gate; otherwise the partials go to `part`
 * and qwen35_attention_merge_kernel combines them.  Decode uses splits
 * because n_tokens * n_head blocks alone cannot fill the GPU while scanning
 * a long cache. */
/* Key splits of a decode-sized pass over n_keys keys (host and device agree
 * on this, the host for the grid, the device per row). */
__host__ __device__ __forceinline__ uint32_t qwen35_attn_splits(uint32_t n_keys) {
    uint32_t n = (n_keys + 127u) / 128u;
    if (n > QWEN35_ATTN_SPLIT_MAX) n = QWEN35_ATTN_SPLIT_MAX;
    return n == 0u ? 1u : n;
}

/* A batched row's keys under QSA: a row that sees no more completed blocks
 * than the budget (at most dense_keys keys) attends to its causal prefix
 * exactly as it would serially, whatever the pass's longest row decided;
 * the others take their selected cells.  A sequential pass follows the
 * host's one decision. */
__device__ __forceinline__ const int32_t *qwen35_attn_row_cells(
        const int32_t *sel, uint32_t max_sel, uint32_t dense_keys, uint32_t t, uint32_t pos, uint32_t batched) {
    if (!sel || (batched && pos + 1u <= dense_keys)) return NULL;
    return sel + (uint64_t)t * max_sel;
}

/* A row's split plan: a sequential pass plans once over its whole key range
 * (pos_end keys) for every token, a batched row over its own; max_splits ==
 * 1 is the tensor-core-less prefill, unsplit. */
__device__ __forceinline__ uint32_t qwen35_attn_row_splits(
        uint32_t max_splits, const int32_t *cells, uint32_t max_sel, uint32_t pos, uint32_t pos_end, uint32_t batched) {
    if (max_splits == 1u) return 1u;
    return qwen35_attn_splits(cells ? max_sel : (batched ? pos + 1u : pos_end));
}

__global__ static void qwen35_attention_kernel(
        float *att, float *part, const float *qg, const ds4_qwen_batch_slot *slots,
        const int32_t *sel, const uint32_t *n_sel, uint32_t max_sel, uint32_t dense_keys,
        uint32_t n_head, uint32_t n_kv, uint32_t hd, uint32_t pos_end, uint32_t n_tokens, uint32_t batched,
        uint32_t max_splits) {
    extern __shared__ float scores[];
    __shared__ float scratch[32];
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t split = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t n_warps = blockDim.x >> 5u;
    if (t >= n_tokens || h >= n_head) return;
    const ds4_qwen_batch_slot row = slots[batched ? t : 0u];
    const uint32_t pos = batched ? row.pos : row.pos + t;
    const int32_t *cells = qwen35_attn_row_cells(sel, max_sel, dense_keys, t, pos, batched);
    const uint32_t n_splits = qwen35_attn_row_splits(max_splits, cells, max_sel, pos, pos_end, batched);
    if (split >= n_splits) return;
    const float *q = qg + ((uint64_t)t * n_head + h) * 2u * hd;
    const float *gate = q + hd;
    const uint32_t kvh = h / (n_head / n_kv);
    const uint32_t n_keys = cells ? n_sel[t] : pos + 1u;
    const uint32_t chunk = (n_keys + n_splits - 1u) / n_splits;
    const uint32_t key0 = split * chunk;
    const uint32_t key1 = key0 + chunk < n_keys ? key0 + chunk : n_keys;
    const uint32_t n_local = key1 > key0 ? key1 - key0 : 0u;
    const uint64_t kv_stride = (uint64_t)n_kv * hd;
    const float scale = rsqrtf((float)hd);
    const uint32_t per = hd / 32u;
    float qv[8];
    for (uint32_t i = 0; i < per; i++) qv[i] = q[lane * per + i];
    for (uint32_t key = warp; key < n_local; key += n_warps) {
        const uint32_t cell = cells ? (uint32_t)cells[key0 + key] : key0 + key;
        const __nv_bfloat162 *kr = (const __nv_bfloat162 *)(qwen35_kv_ptr(row, row.p0, cell, kv_stride) + kvh * hd + lane * per);
        float dot = 0.0f;
        for (uint32_t i = 0; i < per / 2u; i++) {
            const float2 kk = __bfloat1622float2(kr[i]);
            dot = fmaf(qv[2u * i], kk.x, dot);
            dot = fmaf(qv[2u * i + 1u], kk.y, dot);
        }
        dot = warp_sum_f32(dot);
        if (lane == 0u) scores[key] = dot * scale;
    }
    __syncthreads();
    float local_max = -INFINITY;
    for (uint32_t key = tid; key < n_local; key += blockDim.x) local_max = fmaxf(local_max, scores[key]);
    local_max = warp_max_f32(local_max);
    if (lane == 0u) scratch[warp] = local_max;
    __syncthreads();
    float max = lane < n_warps ? scratch[lane] : -INFINITY;
    max = __shfl_sync(0xffffffffu, warp_max_f32(max), 0);
    __syncthreads();
    float local_sum = 0.0f;
    for (uint32_t key = tid; key < n_local; key += blockDim.x) {
        const float p = expf(scores[key] - max);
        scores[key] = p;
        local_sum += p;
    }
    const float sum = qwen35_cuda_block_sum(local_sum, scratch);
    __syncthreads();
    float acc = 0.0f;
    const uint64_t vcol = row.p1 + kvh * hd + tid;   /* this thread's V column within a page row */
    for (uint32_t key = 0; key < n_local; key++) {
        const uint32_t cell = cells ? (uint32_t)cells[key0 + key] : key0 + key;
        acc = fmaf(scores[key], __bfloat162float(qwen35_kv_ptr(row, vcol, cell, kv_stride)[0]), acc);
    }
    if (n_splits == 1u) {
        att[((uint64_t)t * n_head + h) * hd + tid] = acc / sum * qwen35_cuda_sigmoid(gate[tid]);
        return;
    }
    float *dst = part + (((uint64_t)t * n_head + h) * max_splits + split) * (hd + 2u);
    dst[tid] = acc;
    if (tid == 0u) {
        dst[hd] = max;
        dst[hd + 1u] = sum;
    }
}

__global__ static void qwen35_attention_merge_kernel(
        float *att, const float *part, const float *qg, const ds4_qwen_batch_slot *slots,
        const int32_t *sel, uint32_t max_sel, uint32_t dense_keys,
        uint32_t n_head, uint32_t hd, uint32_t pos_end, uint32_t n_tokens, uint32_t batched, uint32_t max_splits) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || h >= n_head) return;
    const ds4_qwen_batch_slot row = slots[batched ? t : 0u];
    const uint32_t pos = batched ? row.pos : row.pos + t;
    const int32_t *cells = qwen35_attn_row_cells(sel, max_sel, dense_keys, t, pos, batched);
    const uint32_t n_splits = qwen35_attn_row_splits(max_splits, cells, max_sel, pos, pos_end, batched);
    if (n_splits == 1u) return;   /* the attention kernel finalised this row itself */
    const float *base = part + ((uint64_t)t * n_head + h) * max_splits * (hd + 2u);
    float max = -INFINITY;
    for (uint32_t s = 0; s < n_splits; s++) max = fmaxf(max, base[s * (hd + 2u) + hd]);
    float sum = 0.0f;
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_splits; s++) {
        const float *p = base + s * (hd + 2u);
        const float w = expf(p[hd] - max);   /* empty splits carry -inf and vanish */
        sum = fmaf(p[hd + 1u], w, sum);
        acc = fmaf(p[tid], w, acc);
    }
    const float *gate = qg + ((uint64_t)t * n_head + h) * 2u * hd + hd;
    att[((uint64_t)t * n_head + h) * hd + tid] = acc / sum * qwen35_cuda_sigmoid(gate[tid]);
}

/* ---- Tensor-core prefill attention ---------------------------------------
 * Block per (token, KV head), eight warps.  The rows are the group's query
 * heads (up to 16) and the keys come in 16-cell tiles gathered from the bf16
 * caches, so the causal prefix and a QSA cell list cost the same.  Warp w
 * scores keys 8*(w%2).. over dims 64*(w/2)..
 * with its query fragments held in registers; one thread per (row, key)
 * sums the four partials and runs the online softmax; warp w then owns
 * output dims 32*w.. .  The mma.sync fragment layouts are the m16n8k16
 * ones from the PTX ISA (rows lane/4 and lane/4+8, column pairs lane%4). */
#define QWEN35_TC_KEYS 16u
#define QWEN35_TC_LD 264u    /* bf16 per staged K/V row: 256 + 8 so ldmatrix rows hit distinct banks */
#define QWEN35_TC_LDP 24u    /* bf16 per staged P row */

__device__ __forceinline__ void qwen35_mma_bf16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void qwen35_ldsm_x2(uint32_t *r, const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void qwen35_ldsm_x2_trans(uint32_t *r, const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void qwen35_ldsm_x4(uint32_t *r, const void *p) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

/* SPLIT: the queries and probabilities are bf16 hi/lo pairs and every
 * product takes two MMAs, within f32 rounding of the reference, which
 * rounds its K/V rows to bf16 as the cache does (the 2B's guard path).
 * Otherwise they are plain bf16 with f32 accumulation, the checkpoint's own
 * attention precision.  The kernel is bound by the L2 bandwidth of the
 * per-token K/V gather. */
template <bool SPLIT>
__global__ static void __launch_bounds__(256, 2) qwen35_attention_tc_kernel(
        float *att, __nv_bfloat16 *att_bf16, const float *qg, const __nv_bfloat16 *qh, const __nv_bfloat16 *ql,
        const ds4_qwen_batch_slot *slots,
        const int32_t *sel, const uint32_t *n_sel, uint32_t max_sel,
        uint32_t n_head, uint32_t n_kv, uint32_t n_tokens) {
    constexpr uint32_t hd = 256u;
    constexpr uint32_t NP = SPLIT ? 2u : 1u;                                        /* query and probability parts */
    __shared__ __align__(16) __nv_bfloat16 ks[QWEN35_TC_KEYS][QWEN35_TC_LD];      /* [key][dim] */
    __shared__ __align__(16) __nv_bfloat16 vs[QWEN35_TC_KEYS][QWEN35_TC_LD];
    __shared__ __align__(16) __nv_bfloat16 ps[NP][16][QWEN35_TC_LDP];             /* [hi, lo] probabilities [row][key] */
    __shared__ float sp[4][16][QWEN35_TC_KEYS];                                    /* score partials per dim quarter */
    __shared__ float alpha_s[16], lsum_s[16];
    const uint32_t t = blockIdx.x;
    const uint32_t kvh = blockIdx.y;
    const uint32_t group = n_head / n_kv;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const ds4_qwen_batch_slot row = slots[0];   /* a prefill chunk: one session, sequential */
    const uint32_t n_cells = sel ? n_sel[t] : row.pos + t + 1u;
    const int32_t *cells = sel ? sel + (uint64_t)t * max_sel : NULL;
    const uint64_t kv_stride = (uint64_t)n_kv * hd;
    const float scale = rsqrtf((float)hd);
    const uint32_t head0 = kvh * group;

    /* score work of this warp: keys sk.., dims sd..; the query fragments
     * (rows lane/4 and +8, column pairs lane%4, four 16-dim steps) */
    const uint32_t sk = (warp & 1u) * 8u;
    const uint32_t sd = (warp >> 1u) * 64u;
    uint32_t qa[2][4][4];
    for (uint32_t step = 0; step < 4u; step++) {
        for (uint32_t reg = 0; reg < 4u; reg++) {
            const uint32_t r = (lane >> 2u) + (reg & 1u) * 8u;
            const uint32_t d = sd + step * 16u + (reg >> 1u) * 8u + (lane & 3u) * 2u;
            const uint64_t i = ((uint64_t)t * n_head + head0 + r) * hd + d;
            qa[0][step][reg] = r < group ? *(const uint32_t *)(qh + i) : 0u;
            qa[1][step][reg] = SPLIT && r < group ? *(const uint32_t *)(ql + i) : 0u;
        }
    }
    float o[4][4];
    for (uint32_t j = 0; j < 4u; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    /* softmax state of row tid/16, identical in its 16 threads */
    const uint32_t srow = tid >> 4u;
    const uint32_t skey = tid & 15u;
    float m_row = -INFINITY;
    float l_row = 0.0f;

    for (uint32_t c0 = 0; c0 < n_cells; c0 += QWEN35_TC_KEYS) {
        /* gather the tile: 16 threads per key, 32 bytes of each row each;
         * masked keys are zero so their zero probability meets no garbage */
        {
            const uint32_t key = tid >> 4u;
            const uint32_t e = (tid & 15u) * 16u;
            const uint32_t c = c0 + key;
            const bool valid = c < n_cells;
            const uint32_t cell = valid ? (cells ? (uint32_t)cells[c] : c) : 0u;
            const __nv_bfloat16 *src[2] = { qwen35_kv_ptr(row, row.p0, cell, kv_stride) + kvh * hd + e,
                                            qwen35_kv_ptr(row, row.p1, cell, kv_stride) + kvh * hd + e };
            __nv_bfloat16 *dst[2] = { &ks[key][e], &vs[key][e] };
            const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
            for (uint32_t a = 0; a < 2u; a++) {
                const uint4 x0 = valid ? ((const uint4 *)src[a])[0] : zero;
                const uint4 x1 = valid ? ((const uint4 *)src[a])[1] : zero;
                ((uint4 *)dst[a])[0] = x0;
                ((uint4 *)dst[a])[1] = x1;
            }
        }
        __syncthreads();
        /* scores: this warp's 16 rows x 8 keys over its 64 dims */
        {
            float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            const uint32_t key = sk + (lane & 7u);
            for (uint32_t step = 0; step < 4u; step++) {
                const uint32_t d = sd + step * 16u + ((lane >> 3u) & 1u) * 8u;
                uint32_t bh[2];
                qwen35_ldsm_x2(bh, &ks[key][d]);
                qwen35_mma_bf16(s, qa[0][step], bh[0], bh[1]);
                if (SPLIT) qwen35_mma_bf16(s, qa[1][step], bh[0], bh[1]);
            }
            const uint32_t r = lane >> 2u;
            const uint32_t kc = sk + (lane & 3u) * 2u;
            sp[warp >> 1u][r][kc] = s[0];
            sp[warp >> 1u][r][kc + 1u] = s[1];
            sp[warp >> 1u][r + 8u][kc] = s[2];
            sp[warp >> 1u][r + 8u][kc + 1u] = s[3];
        }
        __syncthreads();
        /* online softmax, one thread per (row, key); the row's 16 threads
         * form a half-warp, so the max and sum are shuffle reductions */
        {
            const bool valid = srow < group && c0 + skey < n_cells;
            float v = -INFINITY;
            if (valid) v = (sp[0][srow][skey] + sp[1][srow][skey] + sp[2][srow][skey] + sp[3][srow][skey]) * scale;
            float mx = v;
            for (uint32_t off = 8u; off > 0u; off >>= 1u) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float m_new = fmaxf(m_row, mx);
            const float a = m_new == -INFINITY ? 1.0f : expf(m_row - m_new);
            const float p = valid ? expf(v - m_new) : 0.0f;
            float sum = p;
            for (uint32_t off = 8u; off > 0u; off >>= 1u) sum += __shfl_xor_sync(0xffffffffu, sum, off);
            l_row = l_row * a + sum;
            m_row = m_new;
            if (SPLIT) qwen35_split_f32(p, &ps[0][srow][skey], &ps[NP - 1u][srow][skey]);
            else ps[0][srow][skey] = __float2bfloat16(p);
            if (skey == 0u) alpha_s[srow] = a;
        }
        __syncthreads();
        /* output: rescale the rows, then P (16 x 16 keys) times the tile's
         * values over this warp's 32 dims */
        {
            const float a0 = alpha_s[lane >> 2u];
            const float a1 = alpha_s[(lane >> 2u) + 8u];
            const uint32_t pr = (lane & 7u) + ((lane >> 3u) & 1u) * 8u;
            const uint32_t pc = (lane >> 4u) * 8u;
            uint32_t ph[4], pl[4] = { 0u, 0u, 0u, 0u };
            qwen35_ldsm_x4(ph, &ps[0][pr][pc]);
            if (SPLIT) qwen35_ldsm_x4(pl, &ps[NP - 1u][pr][pc]);
            const uint32_t key = (lane & 7u) + ((lane >> 3u) & 1u) * 8u;
            for (uint32_t j = 0; j < 4u; j++) {
                const uint32_t d = warp * 32u + j * 8u;
                uint32_t bh[2];
                qwen35_ldsm_x2_trans(bh, &vs[key][d]);
                o[j][0] *= a0;
                o[j][1] *= a0;
                o[j][2] *= a1;
                o[j][3] *= a1;
                qwen35_mma_bf16(o[j], ph, bh[0], bh[1]);
                if (SPLIT) qwen35_mma_bf16(o[j], pl, bh[0], bh[1]);
            }
        }
        __syncthreads();   /* the next gather overwrites the tiles */
    }
    if (skey == 0u) lsum_s[srow] = l_row;
    __syncthreads();
    for (uint32_t half = 0; half < 2u; half++) {
        const uint32_t r = (lane >> 2u) + half * 8u;
        if (r >= group) continue;
        const float inv = 1.0f / lsum_s[r];
        const uint64_t head = (uint64_t)t * n_head + head0 + r;
        const float *gate = qg + head * 2u * hd + hd;
        for (uint32_t j = 0; j < 4u; j++) {
            const uint32_t d = warp * 32u + j * 8u + (lane & 3u) * 2u;
            const float v0 = o[j][half * 2u] * inv * qwen35_cuda_sigmoid(gate[d]);
            const float v1 = o[j][half * 2u + 1u] * inv * qwen35_cuda_sigmoid(gate[d + 1u]);
            if (att_bf16) {   /* the out projection's operand */
                *(__nv_bfloat162 *)(att_bf16 + head * hd + d) = __floats2bfloat162_rn(v0, v1);
            } else {
                att[head * hd + d] = v0;
                att[head * hd + d + 1u] = v1;
            }
        }
    }
}

extern "C" int ds4_gpu_qwen35_attention(
        ds4_gpu_tensor       *att,          /* [n_tok][n_head * hd] */
        ds4_gpu_tensor       *att_bf16,     /* optional: a prefill chunk's output goes there in bf16 instead */
        int                   exact,        /* prefill queries and probabilities as hi/lo pairs (f32-exact) or plain bf16 */
        ds4_gpu_tensor       *part,         /* split partials, see QWEN35_ATTN_SPLIT_MAX */
        ds4_gpu_tensor       *split,        /* prefill: bf16 hi/lo copies of the queries, see below */
        ds4_gpu_tensor       *qg,           /* [n_tok][n_head * 2 * hd], modified in place */
        const ds4_gpu_tensor *slots,        /* row table: p0 the bf16 K cache [ctx][n_kv * hd], p1 the V cache */
        uint32_t              slot0,
        const ds4_gpu_tensor *k,            /* [n_tok][n_kv * hd] */
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *sel,          /* QSA: int32 [n_tok][max_sel] cells per token, or NULL for the causal prefix */
        const ds4_gpu_tensor *n_sel,        /* QSA: uint32 [n_tok] */
        uint32_t              max_sel,
        uint32_t              dense_keys,   /* QSA, batched: rows with at most this many keys attend to their prefix instead */
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_norm_offset,
        uint64_t              k_norm_offset,
        uint32_t              n_head,
        uint32_t              n_kv,
        uint32_t              hd,
        uint32_t              n_rot,
        uint32_t              ctx,
        uint32_t              pos_end,      /* one past the highest position of the pass */
        uint32_t              n_tokens,
        int                   batched,
        float                 freq_base,
        float                 eps) {
    const uint32_t b = batched != 0;
    const ds4_qwen_batch_slot *rows = qwen35_slots(slots, slot0, n_tokens, b);
    if (!att || !part || !qg || !rows || !k || !v || !model_map ||
        n_head == 0u || n_kv == 0u || n_head % n_kv != 0u || hd == 0u || hd > 256u ||
        hd % 32u != 0u || n_rot == 0u || n_rot % 2u != 0u || n_rot > hd ||
        pos_end == 0u || pos_end > ctx || (!b && pos_end < n_tokens)) {
        return 0;
    }
    const uint64_t kv_dim = (uint64_t)n_kv * hd;
    if (qg->bytes < (uint64_t)n_tokens * n_head * 2u * hd * sizeof(float) ||
        att->bytes < (uint64_t)n_tokens * n_head * hd * sizeof(float) ||
        k->bytes < n_tokens * kv_dim * sizeof(float) ||
        v->bytes < n_tokens * kv_dim * sizeof(float)) {
        return 0;
    }
    if (sel && (!n_sel || max_sel == 0u ||
                sel->bytes < (uint64_t)n_tokens * max_sel * sizeof(int32_t) ||
                n_sel->bytes < (uint64_t)n_tokens * sizeof(uint32_t))) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(att);
    const float *q_norm = glm53_cuda_weight_f32(model_map, model_size, q_norm_offset, hd, tier, "attn q norm");
    const float *k_norm = glm53_cuda_weight_f32(model_map, model_size, k_norm_offset, hd, tier, "attn k norm");
    if (!q_norm || !k_norm) return 0;
    cudaStream_t stream = cuda_decode_stream();
    /* Prefill chunks take the tensor-core kernel over the caches and
     * split-bf16 copies of the prepared queries: split = [Q hi | Q lo]. */
    const bool tc = n_tokens > QWEN35_ATTN_SPLIT_ROWS && hd == 256u && n_head / n_kv <= 16u;
    const uint64_t q_elems = (uint64_t)n_tokens * n_head * hd;
    __nv_bfloat16 *qh = NULL, *ql = NULL;
    if (tc) {
        if (!split || split->bytes < 2u * q_elems * sizeof(__nv_bfloat16)) return 0;
        qh = (__nv_bfloat16 *)split->ptr;
        ql = qh + q_elems;
    }
    qwen35_attn_prepare_kernel<<<dim3(n_tokens, n_head + 2u * n_kv, 1u), hd, 0, stream>>>(
        (float *)qg->ptr, rows, (const float *)k->ptr, (const float *)v->ptr, q_norm, k_norm, qh, ql,
        n_head, n_kv, hd, n_rot, n_tokens, b, freq_base, eps);
    if (tc) {
        if (att_bf16 && att_bf16->bytes < (uint64_t)n_tokens * n_head * hd * sizeof(__nv_bfloat16)) return 0;
        const dim3 grid(n_tokens, n_kv, 1u);
        float *out = (float *)att->ptr;
        __nv_bfloat16 *out_bf16 = att_bf16 ? (__nv_bfloat16 *)att_bf16->ptr : NULL;
        const int32_t *cells = sel ? (const int32_t *)sel->ptr : NULL;
        const uint32_t *n_cells = sel ? (const uint32_t *)n_sel->ptr : NULL;
        if (exact) {
            qwen35_attention_tc_kernel<true><<<grid, 256, 0, stream>>>(
                out, out_bf16, (const float *)qg->ptr, qh, ql, rows, cells, n_cells, max_sel,
                n_head, n_kv, n_tokens);
        } else {
            qwen35_attention_tc_kernel<false><<<grid, 256, 0, stream>>>(
                out, out_bf16, (const float *)qg->ptr, qh, ql, rows, cells, n_cells, max_sel,
                n_head, n_kv, n_tokens);
        }
        return cuda_ok(cudaGetLastError(), "Qwen tensor-core attention launch");
    }
    /* Decode-sized passes split the key range so enough blocks are in
     * flight; the grid covers the longest row's plan and each row follows
     * its own (qwen35_attn_row_splits).  The scores of a row's split fit
     * its chunk: a row shorter than the longest plans fewer splits, so its
     * chunk is at most 128 keys unless the split count saturates, when it
     * is at most the longest row's. */
    const uint32_t n_keys = sel ? max_sel : pos_end;
    const uint32_t max_splits = n_tokens <= QWEN35_ATTN_SPLIT_ROWS ? qwen35_attn_splits(n_keys) : 1u;
    if (max_splits > 1u &&
        part->bytes < (uint64_t)n_tokens * n_head * max_splits * (hd + 2u) * sizeof(float)) {
        return 0;
    }
    uint32_t chunk = (n_keys + max_splits - 1u) / max_splits;
    if (b && chunk < 128u) chunk = 128u;
    const size_t smem = (size_t)chunk * sizeof(float);
    static size_t smem_limit = 0;
    if (smem > smem_limit) {
        if (cudaFuncSetAttribute(qwen35_attention_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem) != cudaSuccess) {
            fprintf(stderr, "ds4: Qwen3.5 attention needs %zu bytes of shared memory for %u keys\n",
                    smem, n_keys);
            return 0;
        }
        smem_limit = smem;
    }
    qwen35_attention_kernel<<<dim3(n_tokens, n_head, max_splits), hd, smem, stream>>>(
        (float *)att->ptr, (float *)part->ptr, (const float *)qg->ptr, rows,
        sel ? (const int32_t *)sel->ptr : NULL, sel ? (const uint32_t *)n_sel->ptr : NULL, max_sel, dense_keys,
        n_head, n_kv, hd, pos_end, n_tokens, b, max_splits);
    if (max_splits > 1u) {
        qwen35_attention_merge_kernel<<<dim3(n_tokens, n_head, 1u), hd, 0, stream>>>(
            (float *)att->ptr, (const float *)part->ptr, (const float *)qg->ptr, rows,
            sel ? (const int32_t *)sel->ptr : NULL, max_sel, dense_keys,
            n_head, hd, pos_end, n_tokens, b, max_splits);
    }
    return cuda_ok(cudaGetLastError(), "Qwen3.5 attention launch");
}

/* ---- Flash-Next ----------------------------------------------------------
 * Hyper-connection residual streams, routed NVFP4 experts and the PLE n-gram
 * injection, mirroring qwen_hc_mix / qwen_moe_forward / qwen_ple_forward in
 * ds4.c.  Layouts:
 *   x, xn, gate  [n_tok][n_hc][n_embd]   streams, normalised streams, stream gate
 *   inject       [n_tok][n_hc]
 *   sel / selw   [n_tok][n_used]         routed expert ids (int32) and weights
 *   eg, eu, ed   [n_tok * n_used][...]   one row per (token, expert slot)
 */

#define QWEN4EXP_MAX_EXPERT 512u

static bool qwen4exp_elems_fit(const ds4_gpu_tensor *t, uint64_t elems) {
    return t && t->bytes >= elems * sizeof(float);
}

/* The hyper-connection residual streams x live in bf16, as the checkpoint
 * runs them: every kernel that touches them accumulates in f32 and rounds
 * on the store.  Sub-layer outputs (y), h and the decode-side xn stay f32. */
static bool qwen4exp_bf16_fit(const ds4_gpu_tensor *t, uint64_t elems) {
    return t && t->bytes >= elems * sizeof(__nv_bfloat16);
}

/* RMSNorm of every stream with its own slice of the [n_hc * n] gamma: block
 * per (token, stream).  With out_bf16 the result is also written rounded
 * to bf16, the operand of the prefill GEMMs that read it. */
__global__ static void qwen4exp_stream_norm_kernel(
        float *out, __nv_bfloat16 *out_bf16, const __nv_bfloat16 *x, const float *gamma,
        uint32_t n, uint32_t n_hc, uint32_t rows, float eps) {
    __shared__ float scratch[32];
    const uint32_t row = blockIdx.x;
    if (row >= rows * n_hc) return;
    const float *g = gamma + (uint64_t)(row % n_hc) * n;
    const __nv_bfloat16 *xr = x + (uint64_t)row * n;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        ss = fmaf(v, v, ss);
    }
    ss = qwen35_cuda_block_sum(ss, scratch);
    const float scale = rsqrtf(ss / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]) * scale * g[i];
        out_bf16[(uint64_t)row * n + i] = __float2bfloat16(v);
        if (out) out[(uint64_t)row * n + i] = v;
    }
}

extern "C" int ds4_gpu_qwen4exp_stream_norm(
        ds4_gpu_tensor       *out,          /* optional f32 copy, for the decode matvecs */
        ds4_gpu_tensor       *out_bf16,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gamma_offset,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows,
        float                 eps) {
    const uint64_t elems = (uint64_t)rows * n_hc * n_embd;
    if (!model_map || n_embd == 0u || n_hc == 0u || rows == 0u ||
        !qwen4exp_bf16_fit(out_bf16, elems) || !qwen4exp_bf16_fit(x, elems) ||
        (out && !qwen4exp_elems_fit(out, elems))) {
        return 0;
    }
    const float *gamma = glm53_cuda_weight_f32(model_map, model_size, gamma_offset,
                                               (uint64_t)n_hc * n_embd, ds4_tensor_device_idx(out_bf16), "hc norm");
    if (!gamma) return 0;
    qwen4exp_stream_norm_kernel<<<rows * n_hc, 256, 0, cuda_decode_stream()>>>(
        out ? (float *)out->ptr : NULL, (__nv_bfloat16 *)out_bf16->ptr, (const __nv_bfloat16 *)x->ptr, gamma,
        n_embd, n_hc, rows, eps);
    return cuda_ok(cudaGetLastError(), "Flash-Next stream norm launch");
}

/* The low-rank hc gate: lo = SiLU(lo / n_hc). */
__global__ static void qwen4exp_hc_low_kernel(float *lo, uint64_t n, float n_hc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lo[i] = qwen35_cuda_silu(lo[i] / n_hc);
}

extern "C" int ds4_gpu_qwen4exp_hc_low(ds4_gpu_tensor *lo, uint32_t n_low, uint32_t n_hc, uint32_t rows) {
    const uint64_t n = (uint64_t)rows * n_low;
    if (n == 0u || n_hc == 0u || !qwen4exp_elems_fit(lo, n)) return 0;
    qwen4exp_hc_low_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)lo->ptr, n, (float)n_hc);
    return cuda_ok(cudaGetLastError(), "Flash-Next hc gate launch");
}

/* The prefill hc gate GEMM gate = lo W_up^T written in bf16: its [rows][n_hc *
 * n_embd] f32 output was the projection's whole cost (100 MB per call), and
 * the gate only feeds a sigmoid in qwen4exp_hc_mix, which reads it back in
 * bf16.  The checkpoint's recipe keeps this gate in bf16 as well. */
extern "C" int ds4_gpu_qwen4exp_hc_gate(
        ds4_gpu_tensor       *gate,          /* bf16 [rows][n_hc * n_embd] */
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset, /* BF16 [n_hc * n_embd][n_low] */
        uint32_t              wtype,
        float                 scale,
        const ds4_gpu_tensor *lo,            /* f32 [rows][n_low] */
        uint32_t              n_low,
        uint32_t              out_dim,
        uint32_t              rows) {
    const uint64_t weight_bytes = qwen35_weight_bytes(wtype, n_low, out_dim);
    if (!gate || !lo || !model_map || wtype != QWEN35_W_BF16 || !g_cublas_ready || rows == 0u || n_low == 0u ||
        out_dim == 0u || weight_bytes == 0u || weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        !qwen4exp_elems_fit(lo, (uint64_t)rows * n_low) ||
        gate->bytes < (uint64_t)rows * out_dim * sizeof(__nv_bfloat16)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(gate);
    cudaStream_t stream = cuda_decode_stream();
    const char *w = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, tier, "hc up");
    const uint64_t n = (uint64_t)rows * n_low;
    __nv_bfloat16 *a = (__nv_bfloat16 *)cuda_tmp_alloc_on(tier, n * sizeof(__nv_bfloat16), "hc gate activations");
    if (!w || !a) return 0;
    qwen35_to_bf16(a, (const float *)lo->ptr, n, stream);
    if (!cuda_ok(cudaGetLastError(), "hc gate convert launch")) return 0;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(tier), CUBLAS_OP_T, CUBLAS_OP_N,
                                     (int)out_dim, (int)rows, (int)n_low,
                                     &scale, w, CUDA_R_16BF, (int)n_low, a, CUDA_R_16BF, (int)n_low,
                                     &beta, gate->ptr, CUDA_R_16BF, (int)out_dim, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "hc gate GEMM");
}

/* mixed = mean over streams of xn * sigmoid(gate); the gate is bf16 after
 * the prefill GEMM above and f32 after a decode matvec. */
__global__ static void qwen4exp_hc_mix_kernel(
        float *mixed, const __nv_bfloat16 *xn, const void *gate, uint32_t gate_bf16, uint32_t n, uint32_t n_hc, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n) return;
    const uint64_t t = i / n;
    const uint64_t j = i - t * n;
    float acc = 0.0f;
    for (uint32_t c = 0; c < n_hc; c++) {
        const uint64_t idx = (t * n_hc + c) * n + j;
        const float g = gate_bf16 ? __bfloat162float(((const __nv_bfloat16 *)gate)[idx]) : ((const float *)gate)[idx];
        acc = fmaf(__bfloat162float(xn[idx]), qwen35_cuda_sigmoid(g), acc);
    }
    mixed[i] = acc / (float)n_hc;
}

extern "C" int ds4_gpu_qwen4exp_hc_mix(
        ds4_gpu_tensor       *mixed,
        const ds4_gpu_tensor *xn,            /* bf16 */
        const ds4_gpu_tensor *gate,
        int                   gate_bf16,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * n_embd;
    if (n == 0u || n_hc == 0u || !qwen4exp_elems_fit(mixed, n) || !qwen4exp_bf16_fit(xn, n * n_hc) ||
        !gate || gate->bytes < n * n_hc * (gate_bf16 ? sizeof(__nv_bfloat16) : sizeof(float))) {
        return 0;
    }
    qwen4exp_hc_mix_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)mixed->ptr, (const __nv_bfloat16 *)xn->ptr, gate->ptr, gate_bf16 != 0, n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "Flash-Next hc mix launch");
}

/* Every stream receives the sub-layer output weighted by 2*sigmoid(inject/n_hc). */
__global__ static void qwen4exp_hc_combine_kernel(
        __nv_bfloat16 *x, const float *y, const float *inject, uint32_t n, uint32_t n_hc, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n_hc * n) return;
    const uint64_t row = i / n;                /* token * n_hc + stream */
    const uint64_t j = i - row * n;
    const uint64_t t = row / n_hc;
    const float w = 2.0f * qwen35_cuda_sigmoid(inject[row] / (float)n_hc);
    x[i] = __float2bfloat16(fmaf(y[t * n + j], w, __bfloat162float(x[i])));
}

extern "C" int ds4_gpu_qwen4exp_hc_combine(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *y,
        const ds4_gpu_tensor *inject,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * n_hc * n_embd;
    if (n == 0u || !qwen4exp_bf16_fit(x, n) || !qwen4exp_elems_fit(y, (uint64_t)rows * n_embd) ||
        !qwen4exp_elems_fit(inject, (uint64_t)rows * n_hc)) {
        return 0;
    }
    qwen4exp_hc_combine_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (__nv_bfloat16 *)x->ptr, (const float *)y->ptr, (const float *)inject->ptr, n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "Flash-Next hc combine launch");
}

/* hc combine fused with the next sub-layer's stream norm: block per
 * (token, stream) updates its residual row and normalises the new values
 * while they are still in registers, so the row is read once instead of
 * twice.  Rows up to 16 * blockDim values. */
__global__ static void qwen4exp_hc_combine_norm_kernel(
        __nv_bfloat16 *x, float *xn, __nv_bfloat16 *xn_bf16, const float *y, const float *inject, const float *gamma,
        uint32_t n, uint32_t n_hc, uint32_t rows, float eps) {
    __shared__ float scratch[32];
    const uint32_t row = blockIdx.x;                 /* token * n_hc + stream */
    if (row >= rows * n_hc) return;
    const uint32_t t = row / n_hc;
    const float w = 2.0f * qwen35_cuda_sigmoid(inject[row] / (float)n_hc);
    const float *g = gamma + (uint64_t)(row % n_hc) * n;
    const float *yr = y + (uint64_t)t * n;
    __nv_bfloat16 *xr = x + (uint64_t)row * n;
    float v[16];
    float ss = 0.0f;
    for (uint32_t j = 0; j < 16u; j++) {
        const uint32_t i = threadIdx.x + j * blockDim.x;
        v[j] = 0.0f;
        if (i < n) {
            /* the residual is rounded to bf16 on the store and the norm
             * sees the rounded value, as the reference does */
            const __nv_bfloat16 r = __float2bfloat16(fmaf(yr[i], w, __bfloat162float(xr[i])));
            xr[i] = r;
            v[j] = __bfloat162float(r);
        }
        ss = fmaf(v[j], v[j], ss);
    }
    ss = qwen35_cuda_block_sum(ss, scratch);
    const float scale = rsqrtf(ss / (float)n + eps);
    for (uint32_t j = 0; j < 16u; j++) {
        const uint32_t i = threadIdx.x + j * blockDim.x;
        if (i >= n) break;
        const float o = v[j] * scale * g[i];
        xn_bf16[(uint64_t)row * n + i] = __float2bfloat16(o);
        if (xn) xn[(uint64_t)row * n + i] = o;
    }
}

extern "C" int ds4_gpu_qwen4exp_hc_combine_norm(
        ds4_gpu_tensor       *x,
        ds4_gpu_tensor       *xn,           /* optional f32 copy, for the decode matvecs */
        ds4_gpu_tensor       *xn_bf16,
        const ds4_gpu_tensor *y,
        const ds4_gpu_tensor *inject,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gamma_offset,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows,
        float                 eps) {
    const uint64_t elems = (uint64_t)rows * n_hc * n_embd;
    if (!model_map || n_embd == 0u || n_embd > 16u * 256u || n_hc == 0u || rows == 0u ||
        !qwen4exp_bf16_fit(x, elems) || !qwen4exp_bf16_fit(xn_bf16, elems) || (xn && !qwen4exp_elems_fit(xn, elems)) ||
        !qwen4exp_elems_fit(y, (uint64_t)rows * n_embd) || !qwen4exp_elems_fit(inject, (uint64_t)rows * n_hc)) {
        return 0;
    }
    const float *gamma = glm53_cuda_weight_f32(model_map, model_size, gamma_offset,
                                               (uint64_t)n_hc * n_embd, ds4_tensor_device_idx(x), "hc norm");
    if (!gamma) return 0;
    qwen4exp_hc_combine_norm_kernel<<<rows * n_hc, 256, 0, cuda_decode_stream()>>>(
        (__nv_bfloat16 *)x->ptr, xn ? (float *)xn->ptr : NULL, (__nv_bfloat16 *)xn_bf16->ptr,
        (const float *)y->ptr, (const float *)inject->ptr, gamma, n_embd, n_hc, rows, eps);
    return cuda_ok(cudaGetLastError(), "Flash-Next hc combine+norm launch");
}

/* MTP drafter input.  The main model's residual streams (bf16 [rows][dim])
 * take one RMSNorm over the whole widened row (gamma [dim]) into f32, the
 * operand of the shared per-stream projection; the projected streams then
 * receive the projected token embedding and become the drafter's residual. */
__global__ static void qwen4exp_mtp_norm_kernel(
        float *out, const __nv_bfloat16 *x, const float *gamma, uint32_t dim, uint32_t rows, float eps) {
    __shared__ float scratch[32];
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *xr = x + (uint64_t)row * dim;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        ss = fmaf(v, v, ss);
    }
    ss = qwen35_cuda_block_sum(ss, scratch);
    const float scale = rsqrtf(ss / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        out[(uint64_t)row * dim + i] = __bfloat162float(xr[i]) * scale * gamma[i];
    }
}

extern "C" int ds4_gpu_qwen4exp_mtp_norm(
        ds4_gpu_tensor       *out,          /* f32 [rows][dim] */
        const ds4_gpu_tensor *x,            /* bf16 [rows][dim] */
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gamma_offset,
        uint32_t              dim,
        uint32_t              rows,
        float                 eps) {
    const uint64_t n = (uint64_t)rows * dim;
    if (!model_map || n == 0u || !qwen4exp_elems_fit(out, n) || !qwen4exp_bf16_fit(x, n)) return 0;
    const float *gamma = glm53_cuda_weight_f32(model_map, model_size, gamma_offset, dim, ds4_tensor_device_idx(out), "mtp norm");
    if (!gamma) return 0;
    qwen4exp_mtp_norm_kernel<<<rows, 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const __nv_bfloat16 *)x->ptr, gamma, dim, rows, eps);
    return cuda_ok(cudaGetLastError(), "Flash-Next MTP norm launch");
}

__global__ static void qwen4exp_mtp_fuse_kernel(
        __nv_bfloat16 *x, const float *hidden, const float *emb, uint32_t n, uint32_t n_hc, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n_hc * n) return;
    const uint64_t t = i / ((uint64_t)n_hc * n);
    x[i] = __float2bfloat16(hidden[i] + emb[t * n + i % n]);
}

extern "C" int ds4_gpu_qwen4exp_mtp_fuse(
        ds4_gpu_tensor       *x,            /* bf16 [rows][n_hc][n_embd] out */
        const ds4_gpu_tensor *hidden,       /* f32 [rows * n_hc][n_embd], the projected streams */
        const ds4_gpu_tensor *emb,          /* f32 [rows][n_embd], the projected embedding */
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * n_hc * n_embd;
    if (n == 0u || !qwen4exp_bf16_fit(x, n) || !qwen4exp_elems_fit(hidden, n) ||
        !qwen4exp_elems_fit(emb, (uint64_t)rows * n_embd)) {
        return 0;
    }
    qwen4exp_mtp_fuse_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (__nv_bfloat16 *)x->ptr, (const float *)hidden->ptr, (const float *)emb->ptr, n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "Flash-Next MTP fuse launch");
}

/* The wide residual starts as n_hc copies of the embedding. */
__global__ static void qwen4exp_replicate_kernel(
        __nv_bfloat16 *x, const float *h, uint32_t n, uint32_t n_hc, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n_hc * n) return;
    const uint64_t t = i / ((uint64_t)n_hc * n);
    x[i] = __float2bfloat16(h[t * n + i % n]);
}

extern "C" int ds4_gpu_qwen4exp_replicate(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *h,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * n_hc * n_embd;
    if (n == 0u || !qwen4exp_bf16_fit(x, n) || !qwen4exp_elems_fit(h, (uint64_t)rows * n_embd)) return 0;
    qwen4exp_replicate_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (__nv_bfloat16 *)x->ptr, (const float *)h->ptr, n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "Flash-Next replicate launch");
}

/* Softmax router and top-k: one block per token, one thread per expert.
 * Repeated block argmax with ties to the lower expert id reproduces the CPU
 * scan exactly, and the kept probabilities are renormalised to sum to one. */
__global__ static void qwen4exp_router_kernel(
        int32_t *sel, float *selw, const float *logits, uint32_t n_expert, uint32_t n_used) {
    __shared__ float p[QWEN4EXP_MAX_EXPERT];
    __shared__ float rv[QWEN4EXP_MAX_EXPERT];
    __shared__ int32_t ri[QWEN4EXP_MAX_EXPERT];
    const uint32_t t = blockIdx.x;
    const uint32_t e = threadIdx.x;
    const float v = e < n_expert ? logits[(uint64_t)t * n_expert + e] : -INFINITY;
    rv[e] = v;
    __syncthreads();
    for (uint32_t s = QWEN4EXP_MAX_EXPERT / 2u; s > 0u; s >>= 1u) {
        if (e < s) rv[e] = fmaxf(rv[e], rv[e + s]);
        __syncthreads();
    }
    const float max = rv[0];
    __syncthreads();
    p[e] = e < n_expert ? expf(v - max) : -1.0f;
    float sum = 0.0f;
    for (uint32_t k = 0; k < n_used; k++) {
        __syncthreads();
        rv[e] = p[e];
        ri[e] = (int32_t)e;
        __syncthreads();
        for (uint32_t s = QWEN4EXP_MAX_EXPERT / 2u; s > 0u; s >>= 1u) {
            if (e < s && (rv[e + s] > rv[e] || (rv[e + s] == rv[e] && ri[e + s] < ri[e]))) {
                rv[e] = rv[e + s];
                ri[e] = ri[e + s];
            }
            __syncthreads();
        }
        const int32_t best = ri[0];
        const float bestv = rv[0];
        if (e == 0u) {
            sel[(uint64_t)t * n_used + k] = best;
            selw[(uint64_t)t * n_used + k] = bestv;
        }
        if ((int32_t)e == best) p[e] = -1.0f;
        sum += bestv;
    }
    __syncthreads();
    if (e < n_used) selw[(uint64_t)t * n_used + e] /= sum;
}

extern "C" int ds4_gpu_qwen4exp_router(
        ds4_gpu_tensor       *sel,
        ds4_gpu_tensor       *selw,
        const ds4_gpu_tensor *logits,
        uint32_t              n_expert,
        uint32_t              n_used,
        uint32_t              rows) {
    if (rows == 0u || n_expert == 0u || n_expert > QWEN4EXP_MAX_EXPERT || n_used == 0u || n_used > n_expert ||
        !qwen4exp_elems_fit(logits, (uint64_t)rows * n_expert) ||
        !sel || sel->bytes < (uint64_t)rows * n_used * sizeof(int32_t) ||
        !qwen4exp_elems_fit(selw, (uint64_t)rows * n_used)) {
        return 0;
    }
    qwen4exp_router_kernel<<<rows, QWEN4EXP_MAX_EXPERT, 0, cuda_decode_stream()>>>(
        (int32_t *)sel->ptr, (float *)selw->ptr, (const float *)logits->ptr, n_expert, n_used);
    return cuda_ok(cudaGetLastError(), "Flash-Next router launch");
}

/* One warp per (slot, output column) over the slot's expert: the stacked
 * [n_expert][out][in] NVFP4 tensor is indexed by sel and scaled per expert.
 * The input row is the slot's own row (x_per_slot) or its token's row. */
__global__ static void qwen4exp_expert_matvec_kernel(
        float *out, const uint8_t *w, uint64_t expert_bytes, const float *scales, const int32_t *sel,
        const float *x, uint32_t x_per_slot, uint32_t in_dim, uint32_t out_dim, uint32_t n_used) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col = blockIdx.x * 8u + warp;
    const uint32_t slot = blockIdx.y;
    if (col >= out_dim) return;
    const int32_t e = sel[slot];
    const uint32_t n_super = in_dim / 64u;
    const uint8_t *wrow = w + (uint64_t)e * expert_bytes + (uint64_t)col * n_super * 36u;
    const float *xrow = x + (uint64_t)(x_per_slot ? slot : slot / n_used) * in_dim;
    const float sum = qwen35_nvfp4_warp_dot(wrow, xrow, n_super, lane);
    if (lane == 0u) out[(uint64_t)slot * out_dim + col] = sum * scales[e];
}

extern "C" int ds4_gpu_qwen4exp_expert_matvec(
        ds4_gpu_tensor       *out,          /* [rows * n_used][out_dim] */
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset, /* [n_expert][out_dim][in_dim] NVFP4 */
        uint64_t              scales_offset, /* [n_expert] f32 */
        const ds4_gpu_tensor *sel,
        const ds4_gpu_tensor *x,
        int                   x_per_slot,
        uint32_t              n_expert,
        uint32_t              n_used,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              rows) {
    const uint64_t expert_bytes = qwen35_weight_bytes(QWEN35_W_NVFP4, in_dim, out_dim);
    const uint64_t slots = (uint64_t)rows * n_used;
    if (!model_map || rows == 0u || n_used == 0u || n_expert == 0u || expert_bytes == 0u ||
        weight_offset > model_size || expert_bytes * n_expert > model_size - weight_offset ||
        !qwen4exp_elems_fit(out, slots * out_dim) ||
        !qwen4exp_elems_fit(x, (x_per_slot ? slots : rows) * in_dim) ||
        !sel || sel->bytes < slots * sizeof(int32_t)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(out);
    const char *w = cuda_resolve_weight_ptr(model_map, weight_offset, expert_bytes * n_expert, tier,
                                            "Flash-Next experts");
    const float *scales = glm53_cuda_weight_f32(model_map, model_size, scales_offset, n_expert, tier,
                                                "Flash-Next expert scales");
    if (!w || !scales) return 0;
    const dim3 grid((out_dim + 7u) / 8u, (unsigned)slots, 1u);
    qwen4exp_expert_matvec_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const uint8_t *)w, expert_bytes, scales, (const int32_t *)sel->ptr,
        (const float *)x->ptr, x_per_slot != 0, in_dim, out_dim, n_used);
    return cuda_ok(cudaGetLastError(), "Flash-Next expert matvec launch");
}

/* Prefill grouping: the (token, slot) pairs sorted by expert so each expert's
 * weights are read once per chunk.  plan holds four [n_expert + 1] uint32
 * arrays: counts, slot starts, tile starts (QWEN35_MMA_ROWS slots per
 * tile), and the scatter cursors.
 * Slots inside an expert land in atomic order, which is harmless: every
 * output row is computed independently and written back to its own slot. */
enum { QWEN4EXP_PLAN_COUNT = 0, QWEN4EXP_PLAN_START = 1, QWEN4EXP_PLAN_TILE = 2, QWEN4EXP_PLAN_CURSOR = 3 };

__global__ static void qwen4exp_expert_count_kernel(uint32_t *count, const int32_t *esel, uint32_t slots) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < slots) atomicAdd(&count[esel[i]], 1u);
}

/* One block of n_expert threads: exclusive scans of the slot counts and of
 * the per-expert tile counts, plus the cursors the scatter consumes. */
__global__ static void qwen4exp_expert_plan_kernel(uint32_t *plan, uint32_t n_expert) {
    __shared__ uint32_t scan[QWEN4EXP_MAX_EXPERT];
    const uint32_t e = threadIdx.x;
    uint32_t *count = plan + QWEN4EXP_PLAN_COUNT * (n_expert + 1u);
    uint32_t *start = plan + QWEN4EXP_PLAN_START * (n_expert + 1u);
    uint32_t *tile = plan + QWEN4EXP_PLAN_TILE * (n_expert + 1u);
    uint32_t *cursor = plan + QWEN4EXP_PLAN_CURSOR * (n_expert + 1u);
    for (uint32_t pass = 0; pass < 2u; pass++) {
        const uint32_t v = e < n_expert ? (pass == 0u ? count[e] : (count[e] + QWEN35_MMA_ROWS - 1u) / QWEN35_MMA_ROWS) : 0u;
        scan[e] = v;
        __syncthreads();
        for (uint32_t off = 1; off < n_expert; off <<= 1) {
            const uint32_t add = e >= off ? scan[e - off] : 0u;
            __syncthreads();
            scan[e] += add;
            __syncthreads();
        }
        uint32_t *dst = pass == 0u ? start : tile;
        if (e < n_expert) dst[e] = scan[e] - v;
        if (e == n_expert - 1u) dst[n_expert] = scan[e];
        if (pass == 0u && e < n_expert) cursor[e] = scan[e] - v;
        __syncthreads();
    }
}

__global__ static void qwen4exp_expert_scatter_kernel(int32_t *order, uint32_t *cursor, const int32_t *esel, uint32_t slots) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < slots) order[atomicAdd(&cursor[esel[i]], 1u)] = (int32_t)i;
}

extern "C" int ds4_gpu_qwen4exp_expert_plan(
        ds4_gpu_tensor       *plan,         /* uint32 [4][n_expert + 1] */
        ds4_gpu_tensor       *order,        /* int32 [slots] slot ids grouped by expert */
        const ds4_gpu_tensor *esel,         /* int32 [slots] */
        uint32_t              n_expert,
        uint32_t              slots) {
    if (n_expert == 0u || n_expert > QWEN4EXP_MAX_EXPERT || slots == 0u ||
        !plan || plan->bytes < 4ull * (n_expert + 1u) * sizeof(uint32_t) ||
        !order || order->bytes < (uint64_t)slots * sizeof(int32_t) ||
        !esel || esel->bytes < (uint64_t)slots * sizeof(int32_t)) {
        return 0;
    }
    cudaStream_t stream = cuda_decode_stream();
    uint32_t *p = (uint32_t *)plan->ptr;
    if (!cuda_ok(cudaMemsetAsync(p, 0, (n_expert + 1u) * sizeof(uint32_t), stream), "Flash-Next expert count clear")) return 0;
    qwen4exp_expert_count_kernel<<<(slots + 255u) / 256u, 256, 0, stream>>>(p, (const int32_t *)esel->ptr, slots);
    qwen4exp_expert_plan_kernel<<<1, QWEN4EXP_MAX_EXPERT, 0, stream>>>(p, n_expert);
    qwen4exp_expert_scatter_kernel<<<(slots + 255u) / 256u, 256, 0, stream>>>(
        (int32_t *)order->ptr, p + QWEN4EXP_PLAN_CURSOR * (n_expert + 1u), (const int32_t *)esel->ptr, slots);
    return cuda_ok(cudaGetLastError(), "Flash-Next expert plan launch");
}

/* Prefill expert projections on the Blackwell FP4 tensor cores
 * (cuda/mmq/ds4_qwen_fp4.cu): the activations are quantised once per layer
 * into the weights' own NVFP4 layout and both operands feed the block-scaled
 * FP4 MMA, so every weight byte is read once per chunk and nothing is
 * dequantised.  This is the checkpoint's activation recipe, the one vLLM
 * and SGLang run. */
extern "C" int ds4_gpu_qwen4exp_quantize_fp4(
        ds4_gpu_tensor *xq, const ds4_gpu_tensor *x, const ds4_gpu_tensor *up, int in_bf16, uint32_t rows, uint32_t k) {
    const uint64_t in_bytes = (uint64_t)rows * k * (in_bf16 ? sizeof(__nv_bfloat16) : sizeof(float));
    if (!xq || !x || rows == 0u || k == 0u || k % 64u != 0u ||
        x->bytes < in_bytes || xq->bytes < (uint64_t)rows * (k / 64u) * 36u || (up && up->bytes < in_bytes)) {
        return 0;
    }
    return ds4_qwen_fp4_quantize(x->ptr, up ? up->ptr : NULL, in_bf16, xq->ptr,
                                 (int)rows, (int)k, cuda_decode_stream()) == 0 &&
           cuda_ok(cudaGetLastError(), "Flash-Next FP4 activation quantize launch");
}

extern "C" int ds4_gpu_qwen4exp_expert_fp4(
        ds4_gpu_tensor       *out,          /* [rows * n_used][out_dim] */
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset, /* [n_expert][out_dim][in_dim] NVFP4 */
        uint64_t              scales_offset, /* [n_expert] f32 */
        const ds4_gpu_tensor *xq,            /* NVFP4 rows: per slot (x_per_slot) or per token */
        int                   x_per_slot,
        const ds4_gpu_tensor *order,
        const ds4_gpu_tensor *plan,
        uint32_t              n_expert,
        uint32_t              n_used,
        uint32_t              in_dim,
        uint32_t              out_dim,
        uint32_t              rows,
        int                   out_bf16) {    /* outputs in bf16, the checkpoint's recipe for them */
    const uint64_t expert_bytes = qwen35_weight_bytes(QWEN35_W_NVFP4, in_dim, out_dim);
    const uint64_t slots = (uint64_t)rows * n_used;
    if (!model_map || rows == 0u || n_used == 0u || n_expert == 0u || expert_bytes == 0u ||
        weight_offset > model_size || expert_bytes * n_expert > model_size - weight_offset ||
        !qwen4exp_elems_fit(out, slots * out_dim) ||
        !xq || xq->bytes < (x_per_slot ? slots : rows) * (in_dim / 64u) * 36u ||
        !order || order->bytes < slots * sizeof(int32_t) ||
        !plan || plan->bytes < 4ull * (n_expert + 1u) * sizeof(uint32_t)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(out);
    const char *w = cuda_resolve_weight_ptr(model_map, weight_offset, expert_bytes * n_expert, tier, "Flash-Next experts");
    const float *scales = glm53_cuda_weight_f32(model_map, model_size, scales_offset, n_expert, tier, "Flash-Next expert scales");
    if (!w || !scales) return 0;
    return ds4_qwen_fp4_moe_gemm(w, scales, xq->ptr, x_per_slot, (const int32_t *)order->ptr, (const uint32_t *)plan->ptr,
                                 (int)n_expert, (int)n_used, (int)in_dim, (int)out_dim, (int)rows,
                                 out->ptr, out_bf16, cuda_decode_stream()) == 0 &&
           cuda_ok(cudaGetLastError(), "Flash-Next FP4 expert GEMM launch");
}

/* y = sum over slots of selw * ed, plus the shared expert already in y
 * scaled by its sigmoid gate. */
__global__ static void qwen4exp_moe_combine_kernel(
        float *y, const void *ed, uint32_t ed_bf16, const float *selw, const float *sg, uint32_t n, uint32_t n_used, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n) return;
    const uint64_t t = i / n;
    const uint64_t j = i - t * n;
    float acc = 0.0f;
    for (uint32_t k = 0; k < n_used; k++) {
        const uint64_t slot = t * n_used + k;
        const float e = ed_bf16 ? __bfloat162float(((const __nv_bfloat16 *)ed)[slot * n + j]) : ((const float *)ed)[slot * n + j];
        acc = fmaf(selw[slot], e, acc);
    }
    y[i] = fmaf(qwen35_cuda_sigmoid(sg[t]), y[i], acc);
}

extern "C" int ds4_gpu_qwen4exp_moe_combine(
        ds4_gpu_tensor       *y,
        const ds4_gpu_tensor *ed,
        int                   ed_bf16,
        const ds4_gpu_tensor *selw,
        const ds4_gpu_tensor *sg,
        uint32_t              n_embd,
        uint32_t              n_used,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * n_embd;
    if (n == 0u || n_used == 0u || !qwen4exp_elems_fit(y, n) || !ed ||
        ed->bytes < n * n_used * (ed_bf16 ? sizeof(__nv_bfloat16) : sizeof(float)) ||
        !qwen4exp_elems_fit(selw, (uint64_t)rows * n_used) || !qwen4exp_elems_fit(sg, rows)) {
        return 0;
    }
    qwen4exp_moe_combine_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)y->ptr, ed->ptr, ed_bf16 != 0, (const float *)selw->ptr, (const float *)sg->ptr,
        n_embd, n_used, rows);
    return cuda_ok(cudaGetLastError(), "Flash-Next MoE combine launch");
}

/* PLE gate, block per (token, stream): the normalised key against the
 * normalised stream gives a signed-sqrt sigmoid gate on the value; the gated
 * value is kept for the residual add and its stream-normalised form is the
 * conv input. */
__global__ static void qwen4exp_ple_gate_kernel(
        float *gated, float *pnorm, const float *pkey, const __nv_bfloat16 *x, const float *pval,
        const float *w_key, const float *w_query, const float *w_conv,
        uint32_t n, uint32_t n_hc, uint32_t rows, float eps) {
    __shared__ float scratch[32];
    const uint32_t row = blockIdx.x;
    if (row >= rows * n_hc) return;
    const uint32_t t = row / n_hc;
    const uint64_t off = (uint64_t)(row % n_hc) * n;
    const float *k = pkey + (uint64_t)row * n;
    const __nv_bfloat16 *q = x + (uint64_t)row * n;
    const float *v = pval + (uint64_t)t * n;
    float ssk = 0.0f, ssq = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float qi = __bfloat162float(q[i]);
        ssk = fmaf(k[i], k[i], ssk);
        ssq = fmaf(qi, qi, ssq);
    }
    ssk = qwen35_cuda_block_sum(ssk, scratch);
    ssq = qwen35_cuda_block_sum(ssq, scratch);
    const float sk = rsqrtf(ssk / (float)n + eps);
    const float sq = rsqrtf(ssq / (float)n + eps);
    float dot = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        dot = fmaf(k[i] * sk * w_key[off + i], __bfloat162float(q[i]) * sq * w_query[off + i], dot);
    }
    const float s = qwen35_cuda_block_sum(dot, scratch) / sqrtf((float)n);
    const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
    const float gate = qwen35_cuda_sigmoid(s > 0.0f ? mag : s < 0.0f ? -mag : 0.0f);
    float *g = gated + (uint64_t)row * n;
    float ssg = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float gv = v[i] * gate;
        g[i] = gv;
        ssg = fmaf(gv, gv, ssg);
    }
    const float sg = rsqrtf(qwen35_cuda_block_sum(ssg, scratch) / (float)n + eps);
    float *o = pnorm + (uint64_t)row * n;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) o[i] = g[i] * sg * w_conv[off + i];
}

extern "C" int ds4_gpu_qwen4exp_ple_gate(
        ds4_gpu_tensor       *gated,        /* [rows][n_hc][n_embd] */
        ds4_gpu_tensor       *pnorm,        /* [rows][n_hc][n_embd] */
        const ds4_gpu_tensor *pkey,         /* [rows][n_hc][n_embd] */
        const ds4_gpu_tensor *x,            /* [rows][n_hc][n_embd] */
        const ds4_gpu_tensor *pval,         /* [rows][n_embd] */
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              key_norm_offset,
        uint64_t              query_norm_offset,
        uint64_t              conv_norm_offset,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows,
        float                 eps) {
    const uint64_t wide = (uint64_t)rows * n_hc * n_embd;
    if (!model_map || wide == 0u || !qwen4exp_elems_fit(gated, wide) || !qwen4exp_elems_fit(pnorm, wide) ||
        !qwen4exp_elems_fit(pkey, wide) || !qwen4exp_bf16_fit(x, wide) ||
        !qwen4exp_elems_fit(pval, (uint64_t)rows * n_embd)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(gated);
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    const float *w_key = glm53_cuda_weight_f32(model_map, model_size, key_norm_offset, hc_dim, tier, "PLE key norm");
    const float *w_query = glm53_cuda_weight_f32(model_map, model_size, query_norm_offset, hc_dim, tier, "PLE query norm");
    const float *w_conv = glm53_cuda_weight_f32(model_map, model_size, conv_norm_offset, hc_dim, tier, "PLE conv norm");
    if (!w_key || !w_query || !w_conv) return 0;
    qwen4exp_ple_gate_kernel<<<rows * n_hc, 256, 0, cuda_decode_stream()>>>(
        (float *)gated->ptr, (float *)pnorm->ptr, (const float *)pkey->ptr, (const __nv_bfloat16 *)x->ptr,
        (const float *)pval->ptr, w_key, w_query, w_conv, n_embd, n_hc, rows, eps);
    return cuda_ok(cudaGetLastError(), "Flash-Next PLE gate launch");
}

/* Dilated depthwise causal conv over the normalised gated value, taps
 * oldest..newest with the last tap on the current token and tap k reading
 * (kernel-1-k)*dilation tokens back, from the batch or the history (oldest
 * first).  x += gated + SiLU(conv). */
__global__ static void qwen4exp_ple_conv_kernel(
        __nv_bfloat16 *x, const float *gated, const float *pnorm, const ds4_qwen_batch_slot *slots, const float *taps,
        uint32_t hc_dim, uint32_t kern, uint32_t dil, uint32_t rows, uint32_t batched) {
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (ch >= hc_dim || t >= rows) return;
    const float *hist = (const float *)slots[batched ? t : 0u].p0;
    const uint32_t tl = batched ? 0u : t;
    const float *tp = taps + (uint64_t)ch * kern;
    const int32_t hist_rows = (int32_t)((kern - 1u) * dil);
    const uint64_t cur = (uint64_t)t * hc_dim + ch;
    float acc = tp[kern - 1u] * pnorm[cur];
    for (uint32_t k = 0; k + 1u < kern; k++) {
        const int32_t src = (int32_t)tl - (int32_t)((kern - 1u - k) * dil);
        const float v = src >= 0
            ? pnorm[(uint64_t)(t - tl + (uint32_t)src) * hc_dim + ch]
            : hist[(uint64_t)(hist_rows + src) * hc_dim + ch];
        acc = fmaf(tp[k], v, acc);
    }
    x[cur] = __float2bfloat16(__bfloat162float(x[cur]) + gated[cur] + qwen35_cuda_silu(acc));
}

extern "C" int ds4_gpu_qwen4exp_ple_conv(
        ds4_gpu_tensor       *x,            /* [rows][hc_dim] */
        const ds4_gpu_tensor *gated,        /* [rows][hc_dim] */
        const ds4_gpu_tensor *pnorm,        /* [rows][hc_dim] */
        const ds4_gpu_tensor *slots,        /* row table: p0 the conv history [(kernel-1)*dilation][hc_dim], slid forward afterwards */
        uint32_t              slot0,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              taps_offset,  /* [hc_dim][kernel] f32 */
        uint32_t              hc_dim,
        uint32_t              kern,
        uint32_t              dil,
        uint32_t              rows,
        int                   batched,
        const ds4_gpu_tensor *snap_hist) {  /* optional (sequential): the conv history after each token */
    const uint64_t n = (uint64_t)rows * hc_dim;
    const uint32_t hist_rows = (kern - 1u) * dil;
    const uint32_t b = batched != 0;
    const ds4_qwen_batch_slot *table = qwen35_slots(slots, slot0, rows, b);
    if (!model_map || n == 0u || kern < 2u || dil == 0u || !table || (b && snap_hist) || !qwen4exp_bf16_fit(x, n) ||
        !qwen4exp_elems_fit(gated, n) || !qwen4exp_elems_fit(pnorm, n)) {
        return 0;
    }
    const float *taps = glm53_cuda_weight_f32(model_map, model_size, taps_offset, (uint64_t)hc_dim * kern,
                                              ds4_tensor_device_idx(x), "PLE conv");
    if (!taps) return 0;
    cudaStream_t stream = cuda_decode_stream();
    qwen4exp_ple_conv_kernel<<<dim3((hc_dim + 255u) / 256u, rows, 1u), 256, 0, stream>>>(
        (__nv_bfloat16 *)x->ptr, (const float *)gated->ptr, (const float *)pnorm->ptr, table,
        taps, hc_dim, kern, dil, rows, b);
    if (!qwen35_history_snapshots(snap_hist, table, (const float *)pnorm->ptr, hc_dim, hist_rows, rows, stream)) return 0;
    qwen35_history_slide(table, (const float *)pnorm->ptr, hc_dim, hist_rows, rows, b, stream);
    return cuda_ok(cudaGetLastError(), "Flash-Next PLE conv launch");
}

/* ---- QSA block indexer ----------------------------------------------------
 * Mirrors qwen_attention_cells: each completed block of `r` cells has one
 * key (the mean of the block's raw indexer keys, normalised, rotated at the
 * block's first position); a token scores every completed block it can see
 * by the sum over indexer heads of relu(q_h . k_b) and attends to the
 * `budget` best blocks plus the incomplete tail block.
 *   bkey   [ctx / r][d]        block key cache
 *   hist   [r - 1][d]          raw keys before the chunk, oldest first
 *   sel    [n_tok][max_sel]    selected cells in position order, n_sel each
 */

#define QWEN4EXP_INDEXER_MAX_DIM 128u
#define QWEN4EXP_INDEXER_MAX_Q 1024u
#define QWEN4EXP_SELECT_THREADS 1024u

/* NeoX rotation of the first n_rot dims of a head held in shared memory. */
__device__ __forceinline__ void qwen4exp_rope_shared(float *head, uint32_t n_rot, uint32_t pos, float freq_base) {
    const uint32_t half = n_rot / 2u;
    if (threadIdx.x < half) {
        const uint32_t i = threadIdx.x;
        const float theta = (float)pos * powf(freq_base, -(float)(2u * i) / (float)n_rot);
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = head[i];
        const float x1 = head[i + half];
        head[i] = x0 * c - x1 * s;
        head[i + half] = x0 * s + x1 * c;
    }
}

/* Block per token; only the token that completes a block writes its key,
 * summing the block's raw keys in position order like the CPU ring.  The
 * cache holds the keys in bf16, the scoring GEMM's operand, so a long
 * context is never re-converted per token. */
__global__ static void qwen4exp_block_key_kernel(
        const ds4_qwen_batch_slot *slots, const float *raw, const float *k_norm,
        uint32_t d, uint32_t r, uint32_t n_rot, uint32_t n_tokens, uint32_t batched, float freq_base, float eps) {
    __shared__ float scratch[32];
    __shared__ float kb[QWEN4EXP_INDEXER_MAX_DIM];
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const ds4_qwen_batch_slot row = slots[batched ? t : 0u];
    const uint32_t pos = batched ? row.pos : row.pos + t;
    if (pos % r != r - 1u) return;
    const float *hist = (const float *)row.p0;
    const uint32_t tl = batched ? 0u : t;
    float acc = 0.0f;
    for (uint32_t j = 0; j < r; j++) {
        const int32_t src = (int32_t)tl - (int32_t)(r - 1u) + (int32_t)j;
        acc += src >= 0 ? raw[(uint64_t)(t - tl + (uint32_t)src) * d + tid] : hist[(uint64_t)(int32_t)(r - 1u + src) * d + tid];
    }
    const float v = acc / (float)r;
    const float total = qwen35_cuda_block_sum(v * v, scratch);
    kb[tid] = v * rsqrtf(total / (float)d + eps) * k_norm[tid];
    __syncthreads();
    qwen4exp_rope_shared(kb, n_rot, pos + 1u - r, freq_base);
    __syncthreads();
    qwen35_bkey_ptr(row, pos / r, r, d)[tid] = __float2bfloat16(kb[tid]);
}

extern "C" int ds4_gpu_qwen4exp_block_keys(
        const ds4_gpu_tensor *slots,        /* row table: p0 the raw-key history [r-1][d] (slid forward afterwards), p1 the bf16 block keys [ctx / r][d] */
        uint32_t              slot0,
        const ds4_gpu_tensor *raw,          /* [n_tok][d] raw indexer keys of the chunk */
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              k_norm_offset,
        uint32_t              d,
        uint32_t              r,
        uint32_t              n_rot,
        uint32_t              ctx,
        uint32_t              pos_end,
        uint32_t              n_tokens,
        int                   batched,
        float                 freq_base,
        float                 eps,
        const ds4_gpu_tensor *snap_hist) {  /* optional (sequential): the raw-key history after each token */
    const uint32_t b = batched != 0;
    const ds4_qwen_batch_slot *rows = qwen35_slots(slots, slot0, n_tokens, b);
    if (!model_map || !rows || d == 0u || d > QWEN4EXP_INDEXER_MAX_DIM || d % 32u != 0u || r < 2u || n_rot == 0u ||
        n_rot % 2u != 0u || n_rot > d || pos_end == 0u || pos_end > ctx || (!b && pos_end < n_tokens) ||
        (b && snap_hist) || !qwen4exp_elems_fit(raw, (uint64_t)n_tokens * d)) {
        return 0;
    }
    const float *k_norm = glm53_cuda_weight_f32(model_map, model_size, k_norm_offset, d,
                                                ds4_tensor_device_idx(raw), "indexer k norm");
    if (!k_norm) return 0;
    cudaStream_t stream = cuda_decode_stream();
    qwen4exp_block_key_kernel<<<n_tokens, d, 0, stream>>>(
        rows, (const float *)raw->ptr, k_norm, d, r, n_rot, n_tokens, b, freq_base, eps);
    if (!qwen35_history_snapshots(snap_hist, rows, (const float *)raw->ptr, d, r - 1u, n_tokens, stream)) return 0;
    qwen35_history_slide(rows, (const float *)raw->ptr, d, r - 1u, n_tokens, b, stream);
    return cuda_ok(cudaGetLastError(), "Flash-Next block key launch");
}

/* Per (token, indexer head): RMS-normalise and rotate the query in place. */
__global__ static void qwen4exp_indexer_query_kernel(
        float *q, const ds4_qwen_batch_slot *slots, const float *q_norm, uint32_t n_head, uint32_t d, uint32_t n_rot,
        uint32_t n_tokens, uint32_t batched, float freq_base, float eps) {
    __shared__ float scratch[32];
    __shared__ float qh[QWEN4EXP_INDEXER_MAX_DIM];
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || h >= n_head) return;
    const uint32_t pos = batched ? slots[t].pos : slots[0].pos + t;
    float *head = q + ((uint64_t)t * n_head + h) * d;
    const float x = head[tid];
    const float total = qwen35_cuda_block_sum(x * x, scratch);
    qh[tid] = x * rsqrtf(total / (float)d + eps) * q_norm[tid];
    __syncthreads();
    qwen4exp_rope_shared(qh, n_rot, pos, freq_base);
    __syncthreads();
    head[tid] = qh[tid];
}

extern "C" int ds4_gpu_qwen4exp_indexer_query(
        ds4_gpu_tensor       *q,            /* [n_tok][n_head][d], modified in place */
        const ds4_gpu_tensor *slots,        /* row table, for the positions */
        uint32_t              slot0,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_norm_offset,
        uint32_t              n_head,
        uint32_t              d,
        uint32_t              n_rot,
        uint32_t              n_tokens,
        int                   batched,
        float                 freq_base,
        float                 eps) {
    const uint32_t b = batched != 0;
    const ds4_qwen_batch_slot *rows = qwen35_slots(slots, slot0, n_tokens, b);
    if (!model_map || !rows || n_head == 0u || d == 0u || d > QWEN4EXP_INDEXER_MAX_DIM || d % 32u != 0u ||
        n_rot == 0u || n_rot % 2u != 0u || n_rot > d ||
        !qwen4exp_elems_fit(q, (uint64_t)n_tokens * n_head * d)) {
        return 0;
    }
    const float *q_norm = glm53_cuda_weight_f32(model_map, model_size, q_norm_offset, d,
                                                ds4_tensor_device_idx(q), "indexer q norm");
    if (!q_norm) return 0;
    qwen4exp_indexer_query_kernel<<<dim3(n_tokens, n_head, 1u), d, 0, cuda_decode_stream()>>>(
        (float *)q->ptr, rows, q_norm, n_head, d, n_rot, n_tokens, b, freq_base, eps);
    return cuda_ok(cudaGetLastError(), "Flash-Next indexer query launch");
}

/* Block scores on the tensor cores: a 32-token x 128-block tile per block,
 * one head at a time (its 128 dims staged in shared memory for ldmatrix,
 * the query fragments read from global), ReLU applied per head to the f32
 * accumulator and summed.  Each score becomes the token's 64-bit key for
 * the select below (score bits above the negated block index, so keys are
 * unique and order by score first, older block first), written to that
 * token's row of the key scratch.  bf16 operands, the checkpoint's own
 * indexer precision, and 200x the scalar loop's rate at 93K context. */
#define QWEN4EXP_SCORE_ROWS 32u
#define QWEN4EXP_SCORE_COLS 128u
#define QWEN4EXP_SCORE_LD (QWEN4EXP_INDEXER_MAX_DIM + 8u)   /* bf16 per staged key row */

__global__ static void __launch_bounds__(256, 2) qwen4exp_indexer_score_kernel(
        uint64_t *keys, uint64_t keys_stride, const __nv_bfloat16 *q, const ds4_qwen_batch_slot *slots,
        uint32_t n_head, uint32_t d, uint32_t r, uint32_t n_tokens, uint32_t n_blocks, uint32_t batched) {
    __shared__ __align__(16) __nv_bfloat16 bs[QWEN4EXP_SCORE_COLS][QWEN4EXP_SCORE_LD];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t t0 = blockIdx.y * QWEN4EXP_SCORE_ROWS;
    const uint32_t b0 = blockIdx.x * QWEN4EXP_SCORE_COLS;
    /* a batched row scores its one token against its own block keys, in
     * its own tile (the grid's z); the tile columns past its blocks are idle */
    const uint32_t pass_row = blockIdx.z;
    const ds4_qwen_batch_slot slot = slots[pass_row];
    if (batched) {
        n_blocks = (slot.pos + 1u) / r;
        n_tokens = 1u;
        q += (uint64_t)pass_row * n_head * d;
        keys += (uint64_t)pass_row * keys_stride;
    }
    if (b0 >= n_blocks) return;
    const uint32_t wr = (warp >> 2u) * 16u;
    const uint32_t wc = (warp & 3u) * 32u;
    const uint32_t qd = n_head * d;                      /* query row length */
    float sum[4][4];
    for (uint32_t f = 0; f < 4u; f++) sum[f][0] = sum[f][1] = sum[f][2] = sum[f][3] = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        /* stage this head's 128 key rows: 2 threads per row, 8 x 16 bytes each */
        {
            const uint32_t row = tid >> 1u;
            const uint32_t half = (tid & 1u) * (d / 2u);
            const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
            const bool valid = b0 + row < n_blocks;
            const uint4 *src = (const uint4 *)(qwen35_bkey_ptr(slot, valid ? b0 + row : 0u, r, d) + half);
            uint4 *dst = (uint4 *)&bs[row][half];
            for (uint32_t i = 0; i < d / 16u; i++) dst[i] = valid ? src[i] : zero;
        }
        __syncthreads();
        float acc[4][4];
        for (uint32_t f = 0; f < 4u; f++) acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.0f;
        for (uint32_t ks = 0; ks < d / 16u; ks++) {
            uint32_t a[4];
            for (uint32_t reg = 0; reg < 4u; reg++) {
                const uint32_t row = t0 + wr + (lane >> 2u) + (reg & 1u) * 8u;
                const uint32_t col = h * d + ks * 16u + (reg >> 1u) * 8u + (lane & 3u) * 2u;
                a[reg] = row < n_tokens ? *(const uint32_t *)(q + (uint64_t)row * qd + col) : 0u;
            }
            const uint32_t kcol = ks * 16u + ((lane >> 3u) & 1u) * 8u;
            for (uint32_t f = 0; f < 4u; f++) {
                uint32_t b[2];
                qwen35_ldsm_x2(b, &bs[wc + f * 8u + (lane & 7u)][kcol]);
                qwen35_mma_bf16(acc[f], a, b[0], b[1]);
            }
        }
        for (uint32_t f = 0; f < 4u; f++) {
            for (uint32_t l = 0; l < 4u; l++) sum[f][l] += fmaxf(acc[f][l], 0.0f);
        }
        __syncthreads();   /* the next head overwrites the stage */
    }
    for (uint32_t f = 0; f < 4u; f++) {
        for (uint32_t l = 0; l < 4u; l++) {
            const uint32_t row = t0 + wr + (lane >> 2u) + (l >> 1u) * 8u;
            const uint32_t col = b0 + wc + f * 8u + (lane & 3u) * 2u + (l & 1u);
            if (row < n_tokens && col < n_blocks) {
                keys[row * keys_stride + col] = ((uint64_t)__float_as_uint(sum[f][l]) << 32) | (uint64_t)(0xFFFFFFFFu - col);
            }
        }
    }
}

/* Block selection, one 1024-thread block per token over its row of scored
 * keys.  Tokens that see no more than `budget` completed blocks attend to
 * every cell.  Otherwise an 8-pass radix select finds the budget-th largest
 * key exactly, and the blocks at or above it are emitted in position order
 * and expanded to cells, then the tail cells.  Deterministic, and the same
 * choice as the CPU's repeated argmax with ties to the older block. */
__global__ static void qwen4exp_qsa_select_kernel(
        int32_t *sel, uint32_t *n_sel, const uint64_t *keys_scratch, uint64_t keys_stride, const ds4_qwen_batch_slot *slots,
        uint32_t r, uint32_t budget, uint32_t max_sel, uint32_t pos0, uint32_t n_tokens, uint32_t batched) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t scan[QWEN4EXP_SELECT_THREADS];
    __shared__ uint32_t s_bin, s_k;
    const uint32_t tid = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t t = blockIdx.x;
    if (t >= n_tokens) return;
    const uint64_t *keys = keys_scratch + (uint64_t)t * keys_stride;
    {
        const uint32_t pos = batched ? slots[t].pos : pos0 + t;
        const uint32_t n_blocks = (pos + 1u) / r;
        int32_t *out = sel + (uint64_t)t * max_sel;
        if (n_blocks <= budget) {
            for (uint32_t c = tid; c <= pos; c += nthreads) out[c] = (int32_t)c;
            if (tid == 0u) n_sel[t] = pos + 1u;
            return;
        }
        uint64_t prefix = 0, mask = 0;
        uint32_t k = budget;
        for (int pass = 7; pass >= 0; pass--) {
            if (tid < 256u) hist[tid] = 0u;
            __syncthreads();
            for (uint32_t b = tid; b < n_blocks; b += nthreads) {
                const uint64_t key = keys[b];
                if ((key & mask) == prefix) atomicAdd(&hist[(uint32_t)(key >> (8 * pass)) & 0xffu], 1u);
            }
            __syncthreads();
            if (tid == 0u) {
                uint32_t cum = 0;
                int bin = 255;
                for (; bin > 0; bin--) {
                    if (cum + hist[bin] >= k) break;
                    cum += hist[bin];
                }
                s_bin = (uint32_t)bin;
                s_k = k - cum;
            }
            __syncthreads();
            prefix |= (uint64_t)s_bin << (8 * pass);
            mask |= 0xffull << (8 * pass);
            k = s_k;
            __syncthreads();
        }
        const uint64_t threshold = prefix;
        uint32_t base = 0;
        for (uint32_t tile = 0; tile < n_blocks; tile += nthreads) {
            const uint32_t b = tile + tid;
            const uint32_t flag = b < n_blocks && keys[b] >= threshold ? 1u : 0u;
            __syncthreads();
            scan[tid] = flag;
            __syncthreads();
            for (uint32_t off = 1; off < nthreads; off <<= 1) {
                const uint32_t v = tid >= off ? scan[tid - off] : 0u;
                __syncthreads();
                scan[tid] += v;
                __syncthreads();
            }
            if (flag) {
                const uint32_t o = base + scan[tid] - 1u;
                for (uint32_t j = 0; j < r; j++) out[o * r + j] = (int32_t)(b * r + j);
            }
            base += scan[nthreads - 1u];
        }
        const uint32_t tail0 = n_blocks * r;
        for (uint32_t c = tail0 + tid; c <= pos; c += nthreads) out[budget * r + (c - tail0)] = (int32_t)c;
        if (tid == 0u) n_sel[t] = budget * r + (pos + 1u - tail0);
    }
}

extern "C" int ds4_gpu_qwen4exp_qsa_select(
        ds4_gpu_tensor       *sel,          /* int32 [n_tok][max_sel] */
        ds4_gpu_tensor       *n_sel,        /* uint32 [n_tok] */
        ds4_gpu_tensor       *keys,         /* uint64 [keys_rows][ctx / r] scratch, one row per concurrent token */
        uint32_t              keys_rows,
        const ds4_gpu_tensor *q,            /* [n_tok][n_head][d] normalised, rotated */
        const ds4_gpu_tensor *slots,        /* row table: p1 the bf16 block keys [ctx / r][d] */
        uint32_t              slot0,
        uint32_t              n_head,
        uint32_t              d,
        uint32_t              r,
        uint32_t              budget,
        uint32_t              max_sel,
        uint32_t              ctx,
        uint32_t              pos_end,
        uint32_t              n_tokens,
        int                   batched) {
    const uint64_t max_blocks = ctx / r;
    const uint32_t b = batched != 0;
    const ds4_qwen_batch_slot *rows = qwen35_slots(slots, slot0, n_tokens, b);
    if (!rows || n_head == 0u || d == 0u || n_head * d > QWEN4EXP_INDEXER_MAX_Q || r < 2u || budget == 0u ||
        max_sel < budget * r + r - 1u || pos_end == 0u || pos_end > ctx || (!b && pos_end < n_tokens) ||
        keys_rows == 0u || (b && n_tokens > keys_rows) ||
        !sel || sel->bytes < (uint64_t)n_tokens * max_sel * sizeof(int32_t) ||
        !n_sel || n_sel->bytes < (uint64_t)n_tokens * sizeof(uint32_t) ||
        !keys || keys->bytes < (uint64_t)keys_rows * max_blocks * sizeof(uint64_t) ||
        !qwen4exp_elems_fit(q, (uint64_t)n_tokens * n_head * d)) {
        return 0;
    }
    if (d % 16u != 0u || d > QWEN4EXP_INDEXER_MAX_DIM) return 0;
    cudaStream_t stream = cuda_decode_stream();
    const int tier = ds4_tensor_device_idx(sel);
    /* the scores' operands: the chunk's queries in bf16, the block keys as cached */
    const uint64_t qn = (uint64_t)n_tokens * n_head * d;
    __nv_bfloat16 *qb = (__nv_bfloat16 *)cuda_tmp_alloc_on(tier, qn * sizeof(__nv_bfloat16), "indexer queries");
    if (!qb) return 0;
    qwen35_to_bf16(qb, (const float *)q->ptr, qn, stream);
    const uint32_t pos0 = b ? 0u : pos_end - n_tokens;
    /* tokens go in groups of keys_rows, each scoring into its own scratch
     * row; a batched pass is one group of one-token tiles */
    for (uint32_t g0 = 0; g0 < n_tokens; g0 += keys_rows) {
        const uint32_t ng = n_tokens - g0 < keys_rows ? n_tokens - g0 : keys_rows;
        const uint32_t nb = (b ? pos_end : pos0 + g0 + ng) / r;   /* blocks the group's furthest token sees */
        if (nb) {
            const dim3 grid((nb + QWEN4EXP_SCORE_COLS - 1u) / QWEN4EXP_SCORE_COLS,
                            b ? 1u : (ng + QWEN4EXP_SCORE_ROWS - 1u) / QWEN4EXP_SCORE_ROWS,
                            b ? ng : 1u);
            qwen4exp_indexer_score_kernel<<<grid, 256, 0, stream>>>(
                (uint64_t *)keys->ptr, max_blocks, qb + (uint64_t)g0 * n_head * d, rows, n_head, d, r, ng, nb, b);
        }
        qwen4exp_qsa_select_kernel<<<ng, QWEN4EXP_SELECT_THREADS, 0, stream>>>(
            (int32_t *)sel->ptr + (uint64_t)g0 * max_sel, (uint32_t *)n_sel->ptr + g0, (const uint64_t *)keys->ptr, max_blocks,
            rows, r, budget, max_sel, pos0 + g0, ng, b);
    }
    return cuda_ok(cudaGetLastError(), "Flash-Next QSA select launch");
}
