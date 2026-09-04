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
enum { QWEN35_W_F32 = 0, QWEN35_W_BF16 = 30, QWEN35_W_NVFP4 = 40 };

/* Prefill: 64x64 output tile per 256-thread block, weights of any storage
 * type dequantised to shared memory 64 input columns at a time.  Plain f32
 * FMA on f32 activations: this is the correctness path the CPU reference is
 * compared against, not the tensor-core path. */
__global__ static void qwen35_gemm_kernel(
        float *out, const uint8_t *w, uint32_t wtype, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, float scale) {
    __shared__ float ws[64][65];
    __shared__ float xs[64][65];
    const uint32_t col0 = blockIdx.x * 64u;
    const uint32_t row0 = blockIdx.y * 64u;
    const uint32_t tid = threadIdx.x;
    const uint32_t tx = tid & 15u;
    const uint32_t ty = tid >> 4u;
    const uint32_t n_super = in_dim / 64u;
    float acc[4][4] = {{0.0f}};
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
            const uint32_t row = row0 + r;
            const float *src = x + (uint64_t)row * in_dim + (uint64_t)b * 64u + k0;
            for (uint32_t j = 0; j < 16u; j++) xs[r][k0 + j] = row < n_tok ? src[j] : 0.0f;
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
    for (uint32_t i = 0; i < 4u; i++) {
        const uint32_t row = row0 + ty * 4u + i;
        if (row >= n_tok) continue;
        for (uint32_t j = 0; j < 4u; j++) {
            const uint32_t col = col0 + tx * 4u + j;
            if (col < out_dim) out[(uint64_t)row * out_dim + col] = acc[i][j] * scale;
        }
    }
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
    default:             return 0u;
    }
}

/* out[n_tok][out_dim] = x[n_tok][in_dim] W^T times the global scale, for a
 * weight of any storage type the loader accepts.  Decode-sized batches use
 * one warp or block per output column; larger ones the shared-memory GEMM.
 * Both keep the activations in f32, which is what makes the graph match the
 * CPU reference to rounding. */
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
        uint32_t              n_tok) {
    const uint64_t weight_bytes = qwen35_weight_bytes(wtype, in_dim, out_dim);
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u || n_tok == 0u ||
        in_dim % 64u != 0u || weight_bytes == 0u || weight_offset > model_size ||
        weight_bytes > model_size - weight_offset ||
        x->bytes < (uint64_t)n_tok * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tok * out_dim * sizeof(float)) {
        return 0;
    }
    const char *w = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes,
                                            ds4_tensor_device_idx(out), "Qwen weight");
    if (!w) return 0;
    cudaStream_t stream = cuda_decode_stream();
    float *o = (float *)out->ptr;
    const float *xp = (const float *)x->ptr;
    if (n_tok > 8u) {
        const dim3 grid((out_dim + 63u) / 64u, (n_tok + 63u) / 64u, 1u);
        qwen35_gemm_kernel<<<grid, 256, 0, stream>>>(o, (const uint8_t *)w, wtype, xp,
                                                     in_dim, out_dim, n_tok, scale);
        return cuda_ok(cudaGetLastError(), "Qwen GEMM launch");
    }
    if (wtype == QWEN35_W_NVFP4) {
        const dim3 grid((out_dim + 7u) / 8u, n_tok, 1u);
        qwen35_nvfp4_matvec_kernel<<<grid, 256, 0, stream>>>(o, (const uint8_t *)w, xp,
                                                             in_dim, out_dim, n_tok, scale);
        return cuda_ok(cudaGetLastError(), "Qwen NVFP4 matvec launch");
    }
    if (wtype == QWEN35_W_BF16) {
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

/* ---- Gated DeltaNet ------------------------------------------------------ */

/* Causal conv + SiLU over the qkv projection, then L2-normalise the q and k
 * heads.  One block per (token, head slot); taps older than the batch read
 * the conv history.  Slots: n_k q heads, n_k k heads, n_v v heads. */
__global__ static void qwen35_gdn_conv_kernel(
        float *mixed, const float *qkv, const float *hist, const float *conv_w,
        uint32_t n_k, uint32_t n_v, uint32_t n_conv, uint32_t n_tokens, float eps) {
    __shared__ float scratch[32];
    const uint32_t t = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t hd = QWEN35_CUDA_GDN_DIM;
    const uint32_t conv_dim = (2u * n_k + n_v) * hd;
    const uint32_t c = slot * hd + tid;
    if (t >= n_tokens || c >= conv_dim) return;
    const float *taps = conv_w + (uint64_t)c * n_conv;
    float acc = taps[n_conv - 1u] * qkv[(uint64_t)t * conv_dim + c];
    for (uint32_t w = 0; w + 1u < n_conv; w++) {
        const int32_t src = (int32_t)t + (int32_t)w - (int32_t)(n_conv - 1u);
        const float v = src >= 0
            ? qkv[(uint64_t)src * conv_dim + c]
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

/* Slide the conv history forward by n_tokens.  Each thread owns one channel
 * and walks rows in increasing order, so the in-place shift is hazard free. */
__global__ static void qwen35_gdn_conv_state_kernel(
        float *hist, const float *qkv, uint32_t conv_dim, uint32_t n_hist, uint32_t n_tokens) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    for (uint32_t w = 0; w < n_hist; w++) {
        const int32_t src = (int32_t)n_tokens - (int32_t)n_hist + (int32_t)w;
        hist[(uint64_t)w * conv_dim + c] = src >= 0
            ? qkv[(uint64_t)src * conv_dim + c]
            : hist[(uint64_t)(uint32_t)(src + (int32_t)n_hist) * conv_dim + c];
    }
}

/* Recurrent delta rule.  Block per (v head, 4 value rows); each warp owns one
 * value row of S with four key columns per lane, and walks the tokens. */
__global__ static void qwen35_gdn_recurrence_kernel(
        float *o, float *state, const float *mixed, const float *alpha, const float *beta,
        const float *a_neg, const float *dt_bias,
        uint32_t n_k, uint32_t n_v, uint32_t n_tokens, float q_scale) {
    const uint32_t hd = QWEN35_CUDA_GDN_DIM;
    const uint32_t hv = blockIdx.x;
    const uint32_t value = blockIdx.y * 4u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (hv >= n_v || value >= hd) return;
    const uint32_t hk = hv % n_k;
    const uint32_t conv_dim = (2u * n_k + n_v) * hd;
    const uint32_t k0 = lane * 4u;
    float4 *sp = (float4 *)(state + ((uint64_t)hv * hd + value) * hd + k0);
    float4 h = *sp;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *row = mixed + (uint64_t)t * conv_dim;
        const float4 q4 = *(const float4 *)(row + hk * hd + k0);
        const float4 k4 = *(const float4 *)(row + (n_k + hk) * hd + k0);
        const float vval = row[(2u * n_k + hv) * hd + value];
        const float decay = expf(a_neg[hv] *
            qwen35_cuda_softplus(alpha[(uint64_t)t * n_v + hv] + dt_bias[hv]));
        const float b = qwen35_cuda_sigmoid(beta[(uint64_t)t * n_v + hv]);
        h.x *= decay; h.y *= decay; h.z *= decay; h.w *= decay;
        const float sk = __shfl_sync(0xffffffffu, warp_sum_f32(dot4_f32(h, k4)), 0);
        const float delta = (vval - sk) * b;
        h.x = fmaf(k4.x, delta, h.x);
        h.y = fmaf(k4.y, delta, h.y);
        h.z = fmaf(k4.z, delta, h.z);
        h.w = fmaf(k4.w, delta, h.w);
        const float res = __shfl_sync(0xffffffffu, warp_sum_f32(dot4_f32(h, q4)), 0);
        if (lane == 0u) o[(uint64_t)t * n_v * hd + hv * hd + value] = res * q_scale;
    }
    *sp = h;
}

/* RMSNormGated per head: normalise, scale, then multiply by the output gate,
 * SiLU(z) for Qwen3.5 and sigmoid(z) for Flash-Next. */
__global__ static void qwen35_gdn_norm_gate_kernel(
        float *o, const float *z, const float *norm_w, uint32_t n_v, uint32_t n_tokens,
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
    o[idx] = raw * scale * norm_w[tid] * gate;
}

extern "C" int ds4_gpu_qwen35_gdn(
        ds4_gpu_tensor       *out,          /* [n_tok][v_dim] */
        ds4_gpu_tensor       *mixed,        /* [n_tok][conv_dim] scratch */
        ds4_gpu_tensor       *conv_state,   /* [n_conv-1][conv_dim] */
        ds4_gpu_tensor       *ssm_state,    /* [n_v][hd][hd] */
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
        int                   sigmoid_gate,
        float                 eps) {
    const uint32_t hd = QWEN35_CUDA_GDN_DIM;
    if (!out || !mixed || !conv_state || !ssm_state || !qkv || !z || !alpha || !beta ||
        !model_map || n_k == 0u || n_v == 0u || n_v % n_k != 0u || n_conv < 2u ||
        n_tokens == 0u) {
        return 0;
    }
    const uint64_t conv_dim = (uint64_t)(2u * n_k + n_v) * hd;
    const uint64_t v_dim = (uint64_t)n_v * hd;
    if (qkv->bytes < n_tokens * conv_dim * sizeof(float) ||
        mixed->bytes < n_tokens * conv_dim * sizeof(float) ||
        z->bytes < n_tokens * v_dim * sizeof(float) ||
        out->bytes < n_tokens * v_dim * sizeof(float) ||
        alpha->bytes < (uint64_t)n_tokens * n_v * sizeof(float) ||
        beta->bytes < (uint64_t)n_tokens * n_v * sizeof(float) ||
        conv_state->bytes < (uint64_t)(n_conv - 1u) * conv_dim * sizeof(float) ||
        ssm_state->bytes < v_dim * hd * sizeof(float)) {
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

    qwen35_gdn_conv_kernel<<<dim3(n_tokens, 2u * n_k + n_v, 1u), hd, 0, stream>>>(
        (float *)mixed->ptr, (const float *)qkv->ptr, (const float *)conv_state->ptr,
        conv_w, n_k, n_v, n_conv, n_tokens, eps);
    qwen35_gdn_conv_state_kernel<<<(unsigned)((conv_dim + 255u) / 256u), 256, 0, stream>>>(
        (float *)conv_state->ptr, (const float *)qkv->ptr, (uint32_t)conv_dim,
        n_conv - 1u, n_tokens);
    qwen35_gdn_recurrence_kernel<<<dim3(n_v, hd / 4u, 1u), 128, 0, stream>>>(
        (float *)out->ptr, (float *)ssm_state->ptr, (const float *)mixed->ptr,
        (const float *)alpha->ptr, (const float *)beta->ptr, a_neg, dt_bias,
        n_k, n_v, n_tokens, rsqrtf((float)hd));
    qwen35_gdn_norm_gate_kernel<<<dim3(n_tokens, n_v, 1u), hd, 0, stream>>>(
        (float *)out->ptr, (const float *)z->ptr, norm_w, n_v, n_tokens, sigmoid_gate != 0, eps);
    return cuda_ok(cudaGetLastError(), "Qwen3.5 GDN launch");
}

/* ---- Gated GQA attention ------------------------------------------------- */

/* Per (token, head slot): RMS-normalise and RoPE the query heads in place
 * (the q buffer holds [query | gate] per head), do the same for the key heads
 * and store them in the cache, copy the value heads into the cache. */
__global__ static void qwen35_attn_prepare_kernel(
        float *qg, float *k_cache, float *v_cache, const float *k, const float *v,
        const float *q_norm, const float *k_norm,
        uint32_t n_head, uint32_t n_kv, uint32_t hd, uint32_t n_rot,
        uint32_t pos0, uint32_t n_tokens, float freq_base, float eps) {
    __shared__ float scratch[32];
    const uint32_t t = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= hd) return;
    float *head;
    const float *norm_w;
    float *dst = NULL;
    if (slot < n_head) {
        head = qg + ((uint64_t)t * n_head + slot) * 2u * hd;
        norm_w = q_norm;
    } else if (slot < n_head + n_kv) {
        const uint32_t h = slot - n_head;
        head = (float *)k + ((uint64_t)t * n_kv + h) * hd;
        norm_w = k_norm;
        dst = k_cache + ((uint64_t)(pos0 + t) * n_kv + h) * hd;
    } else {
        const uint32_t h = slot - n_head - n_kv;
        v_cache[((uint64_t)(pos0 + t) * n_kv + h) * hd + tid] =
            v[((uint64_t)t * n_kv + h) * hd + tid];
        return;
    }
    const float x = head[tid];
    const float total = qwen35_cuda_block_sum(x * x, scratch);
    head[tid] = x * rsqrtf(total / (float)hd + eps) * norm_w[tid];
    __syncthreads();
    const uint32_t half = n_rot / 2u;
    if (tid < half) {
        const float theta = (float)(pos0 + t) *
            powf(freq_base, -(float)(2u * tid) / (float)n_rot);
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x0 = head[tid];
        const float x1 = head[tid + half];
        head[tid] = x0 * c - x1 * s;
        head[tid + half] = x0 * s + x1 * c;
    }
    if (dst) {
        __syncthreads();
        dst[tid] = head[tid];
    }
}

/* Per (token, head, key split): scores against a contiguous key range,
 * partial softmax (max, sum) and unnormalised weighted values.  With one
 * split the block finalises directly, including the sigmoid output gate;
 * otherwise the partials go to `part` and qwen35_attention_merge_kernel
 * combines them.  Decode uses splits because n_tokens * n_head blocks alone
 * cannot fill the GPU while scanning a long cache. */
__global__ static void qwen35_attention_kernel(
        float *att, float *part, const float *qg, const float *k_cache, const float *v_cache,
        uint32_t n_head, uint32_t n_kv, uint32_t hd, uint32_t pos0, uint32_t n_tokens,
        uint32_t n_splits) {
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
    const float *q = qg + ((uint64_t)t * n_head + h) * 2u * hd;
    const float *gate = q + hd;
    const uint32_t kvh = h / (n_head / n_kv);
    const uint32_t n_keys = pos0 + t + 1u;
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
        const float *kr = k_cache + (key0 + key) * kv_stride + kvh * hd + lane * per;
        float dot = 0.0f;
        for (uint32_t i = 0; i < per; i++) dot = fmaf(qv[i], kr[i], dot);
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
    const float *vc = v_cache + key0 * kv_stride + kvh * hd + tid;
    for (uint32_t key = 0; key < n_local; key++) acc = fmaf(scores[key], vc[key * kv_stride], acc);
    if (n_splits == 1u) {
        att[((uint64_t)t * n_head + h) * hd + tid] = acc / sum * qwen35_cuda_sigmoid(gate[tid]);
        return;
    }
    float *dst = part + (((uint64_t)t * n_head + h) * n_splits + split) * (hd + 2u);
    dst[tid] = acc;
    if (tid == 0u) {
        dst[hd] = max;
        dst[hd + 1u] = sum;
    }
}

__global__ static void qwen35_attention_merge_kernel(
        float *att, const float *part, const float *qg,
        uint32_t n_head, uint32_t hd, uint32_t n_tokens, uint32_t n_splits) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || h >= n_head) return;
    const float *base = part + ((uint64_t)t * n_head + h) * n_splits * (hd + 2u);
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

extern "C" int ds4_gpu_qwen35_attention(
        ds4_gpu_tensor       *att,          /* [n_tok][n_head * hd] */
        ds4_gpu_tensor       *part,         /* split partials, see QWEN35_ATTN_SPLIT_MAX */
        ds4_gpu_tensor       *qg,           /* [n_tok][n_head * 2 * hd], modified in place */
        ds4_gpu_tensor       *k_cache,      /* [ctx][n_kv * hd] */
        ds4_gpu_tensor       *v_cache,
        const ds4_gpu_tensor *k,            /* [n_tok][n_kv * hd] */
        const ds4_gpu_tensor *v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_norm_offset,
        uint64_t              k_norm_offset,
        uint32_t              n_head,
        uint32_t              n_kv,
        uint32_t              hd,
        uint32_t              n_rot,
        uint32_t              ctx,
        uint32_t              pos0,
        uint32_t              n_tokens,
        float                 freq_base,
        float                 eps) {
    if (!att || !part || !qg || !k_cache || !v_cache || !k || !v || !model_map ||
        n_head == 0u || n_kv == 0u || n_head % n_kv != 0u || hd == 0u || hd > 256u ||
        hd % 32u != 0u || n_rot == 0u || n_rot % 2u != 0u || n_rot > hd ||
        n_tokens == 0u || pos0 + n_tokens > ctx) {
        return 0;
    }
    const uint64_t kv_dim = (uint64_t)n_kv * hd;
    if (qg->bytes < (uint64_t)n_tokens * n_head * 2u * hd * sizeof(float) ||
        att->bytes < (uint64_t)n_tokens * n_head * hd * sizeof(float) ||
        k->bytes < n_tokens * kv_dim * sizeof(float) ||
        v->bytes < n_tokens * kv_dim * sizeof(float) ||
        k_cache->bytes < (uint64_t)ctx * kv_dim * sizeof(float) ||
        v_cache->bytes < (uint64_t)ctx * kv_dim * sizeof(float)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(att);
    const float *q_norm = glm53_cuda_weight_f32(model_map, model_size, q_norm_offset, hd, tier, "attn q norm");
    const float *k_norm = glm53_cuda_weight_f32(model_map, model_size, k_norm_offset, hd, tier, "attn k norm");
    if (!q_norm || !k_norm) return 0;
    cudaStream_t stream = cuda_decode_stream();
    qwen35_attn_prepare_kernel<<<dim3(n_tokens, n_head + 2u * n_kv, 1u), hd, 0, stream>>>(
        (float *)qg->ptr, (float *)k_cache->ptr, (float *)v_cache->ptr,
        (const float *)k->ptr, (const float *)v->ptr, q_norm, k_norm,
        n_head, n_kv, hd, n_rot, pos0, n_tokens, freq_base, eps);
    /* Decode-sized batches split the key range so enough blocks are in flight. */
    const uint32_t n_keys = pos0 + n_tokens;
    uint32_t n_splits = 1u;
    if (n_tokens <= QWEN35_ATTN_SPLIT_ROWS) {
        n_splits = (n_keys + 127u) / 128u;
        if (n_splits > QWEN35_ATTN_SPLIT_MAX) n_splits = QWEN35_ATTN_SPLIT_MAX;
        if (n_splits == 0u) n_splits = 1u;
    }
    if (n_splits > 1u &&
        part->bytes < (uint64_t)n_tokens * n_head * n_splits * (hd + 2u) * sizeof(float)) {
        return 0;
    }
    const size_t smem = (size_t)((n_keys + n_splits - 1u) / n_splits) * sizeof(float);
    static size_t smem_limit = 0;
    if (smem > smem_limit) {
        if (cudaFuncSetAttribute(qwen35_attention_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem) != cudaSuccess) {
            fprintf(stderr, "ds4: Qwen3.5 attention needs %zu bytes of shared memory for %u keys\n",
                    smem, pos0 + n_tokens);
            return 0;
        }
        smem_limit = smem;
    }
    qwen35_attention_kernel<<<dim3(n_tokens, n_head, n_splits), hd, smem, stream>>>(
        (float *)att->ptr, (float *)part->ptr, (const float *)qg->ptr, (const float *)k_cache->ptr,
        (const float *)v_cache->ptr, n_head, n_kv, hd, pos0, n_tokens, n_splits);
    if (n_splits > 1u) {
        qwen35_attention_merge_kernel<<<dim3(n_tokens, n_head, 1u), hd, 0, stream>>>(
            (float *)att->ptr, (const float *)part->ptr, (const float *)qg->ptr,
            n_head, hd, n_tokens, n_splits);
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

/* RMSNorm of every stream with its own slice of the [n_hc * n] gamma: block
 * per (token, stream). */
__global__ static void qwen4exp_stream_norm_kernel(
        float *out, const float *x, const float *gamma, uint32_t n, uint32_t n_hc, uint32_t rows, float eps) {
    __shared__ float scratch[32];
    const uint32_t row = blockIdx.x;
    if (row >= rows * n_hc) return;
    const float *g = gamma + (uint64_t)(row % n_hc) * n;
    const float *xr = x + (uint64_t)row * n;
    float *o = out + (uint64_t)row * n;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
    ss = qwen35_cuda_block_sum(ss, scratch);
    const float scale = rsqrtf(ss / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i] * scale * g[i];
}

extern "C" int ds4_gpu_qwen4exp_stream_norm(
        ds4_gpu_tensor       *out,
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
        !qwen4exp_elems_fit(out, elems) || !qwen4exp_elems_fit(x, elems)) {
        return 0;
    }
    const float *gamma = glm53_cuda_weight_f32(model_map, model_size, gamma_offset,
                                               (uint64_t)n_hc * n_embd, ds4_tensor_device_idx(out), "hc norm");
    if (!gamma) return 0;
    qwen4exp_stream_norm_kernel<<<rows * n_hc, 256, 0, cuda_decode_stream()>>>(
        (float *)out->ptr, (const float *)x->ptr, gamma, n_embd, n_hc, rows, eps);
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

/* mixed = mean over streams of xn * sigmoid(gate). */
__global__ static void qwen4exp_hc_mix_kernel(
        float *mixed, const float *xn, const float *gate, uint32_t n, uint32_t n_hc, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n) return;
    const uint64_t t = i / n;
    const uint64_t j = i - t * n;
    float acc = 0.0f;
    for (uint32_t c = 0; c < n_hc; c++) {
        const uint64_t idx = (t * n_hc + c) * n + j;
        acc = fmaf(xn[idx], qwen35_cuda_sigmoid(gate[idx]), acc);
    }
    mixed[i] = acc / (float)n_hc;
}

extern "C" int ds4_gpu_qwen4exp_hc_mix(
        ds4_gpu_tensor       *mixed,
        const ds4_gpu_tensor *xn,
        const ds4_gpu_tensor *gate,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * n_embd;
    if (n == 0u || n_hc == 0u || !qwen4exp_elems_fit(mixed, n) ||
        !qwen4exp_elems_fit(xn, n * n_hc) || !qwen4exp_elems_fit(gate, n * n_hc)) {
        return 0;
    }
    qwen4exp_hc_mix_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)mixed->ptr, (const float *)xn->ptr, (const float *)gate->ptr, n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "Flash-Next hc mix launch");
}

/* Every stream receives the sub-layer output weighted by 2*sigmoid(inject/n_hc). */
__global__ static void qwen4exp_hc_combine_kernel(
        float *x, const float *y, const float *inject, uint32_t n, uint32_t n_hc, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n_hc * n) return;
    const uint64_t row = i / n;                /* token * n_hc + stream */
    const uint64_t j = i - row * n;
    const uint64_t t = row / n_hc;
    const float w = 2.0f * qwen35_cuda_sigmoid(inject[row] / (float)n_hc);
    x[i] = fmaf(y[t * n + j], w, x[i]);
}

extern "C" int ds4_gpu_qwen4exp_hc_combine(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *y,
        const ds4_gpu_tensor *inject,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * n_hc * n_embd;
    if (n == 0u || !qwen4exp_elems_fit(x, n) || !qwen4exp_elems_fit(y, (uint64_t)rows * n_embd) ||
        !qwen4exp_elems_fit(inject, (uint64_t)rows * n_hc)) {
        return 0;
    }
    qwen4exp_hc_combine_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)x->ptr, (const float *)y->ptr, (const float *)inject->ptr, n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "Flash-Next hc combine launch");
}

/* The wide residual starts as n_hc copies of the embedding. */
__global__ static void qwen4exp_replicate_kernel(
        float *x, const float *h, uint32_t n, uint32_t n_hc, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n_hc * n) return;
    const uint64_t t = i / ((uint64_t)n_hc * n);
    x[i] = h[t * n + i % n];
}

extern "C" int ds4_gpu_qwen4exp_replicate(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *h,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * n_hc * n_embd;
    if (n == 0u || !qwen4exp_elems_fit(x, n) || !qwen4exp_elems_fit(h, (uint64_t)rows * n_embd)) return 0;
    qwen4exp_replicate_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)x->ptr, (const float *)h->ptr, n_embd, n_hc, rows);
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

/* y = sum over slots of selw * ed, plus the shared expert already in y
 * scaled by its sigmoid gate. */
__global__ static void qwen4exp_moe_combine_kernel(
        float *y, const float *ed, const float *selw, const float *sg, uint32_t n, uint32_t n_used, uint32_t rows) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n) return;
    const uint64_t t = i / n;
    const uint64_t j = i - t * n;
    float acc = 0.0f;
    for (uint32_t k = 0; k < n_used; k++) {
        const uint64_t slot = t * n_used + k;
        acc = fmaf(selw[slot], ed[slot * n + j], acc);
    }
    y[i] = fmaf(qwen35_cuda_sigmoid(sg[t]), y[i], acc);
}

extern "C" int ds4_gpu_qwen4exp_moe_combine(
        ds4_gpu_tensor       *y,
        const ds4_gpu_tensor *ed,
        const ds4_gpu_tensor *selw,
        const ds4_gpu_tensor *sg,
        uint32_t              n_embd,
        uint32_t              n_used,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * n_embd;
    if (n == 0u || n_used == 0u || !qwen4exp_elems_fit(y, n) || !qwen4exp_elems_fit(ed, n * n_used) ||
        !qwen4exp_elems_fit(selw, (uint64_t)rows * n_used) || !qwen4exp_elems_fit(sg, rows)) {
        return 0;
    }
    qwen4exp_moe_combine_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
        (float *)y->ptr, (const float *)ed->ptr, (const float *)selw->ptr, (const float *)sg->ptr,
        n_embd, n_used, rows);
    return cuda_ok(cudaGetLastError(), "Flash-Next MoE combine launch");
}

/* PLE gate, block per (token, stream): the normalised key against the
 * normalised stream gives a signed-sqrt sigmoid gate on the value; the gated
 * value is kept for the residual add and its stream-normalised form is the
 * conv input. */
__global__ static void qwen4exp_ple_gate_kernel(
        float *gated, float *pnorm, const float *pkey, const float *x, const float *pval,
        const float *w_key, const float *w_query, const float *w_conv,
        uint32_t n, uint32_t n_hc, uint32_t rows, float eps) {
    __shared__ float scratch[32];
    const uint32_t row = blockIdx.x;
    if (row >= rows * n_hc) return;
    const uint32_t t = row / n_hc;
    const uint64_t off = (uint64_t)(row % n_hc) * n;
    const float *k = pkey + (uint64_t)row * n;
    const float *q = x + (uint64_t)row * n;
    const float *v = pval + (uint64_t)t * n;
    float ssk = 0.0f, ssq = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        ssk = fmaf(k[i], k[i], ssk);
        ssq = fmaf(q[i], q[i], ssq);
    }
    ssk = qwen35_cuda_block_sum(ssk, scratch);
    ssq = qwen35_cuda_block_sum(ssq, scratch);
    const float sk = rsqrtf(ssk / (float)n + eps);
    const float sq = rsqrtf(ssq / (float)n + eps);
    float dot = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        dot = fmaf(k[i] * sk * w_key[off + i], q[i] * sq * w_query[off + i], dot);
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
        !qwen4exp_elems_fit(pkey, wide) || !qwen4exp_elems_fit(x, wide) ||
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
        (float *)gated->ptr, (float *)pnorm->ptr, (const float *)pkey->ptr, (const float *)x->ptr,
        (const float *)pval->ptr, w_key, w_query, w_conv, n_embd, n_hc, rows, eps);
    return cuda_ok(cudaGetLastError(), "Flash-Next PLE gate launch");
}

/* Dilated depthwise causal conv over the normalised gated value, taps
 * oldest..newest with the last tap on the current token and tap k reading
 * (kernel-1-k)*dilation tokens back, from the batch or the history (oldest
 * first).  x += gated + SiLU(conv). */
__global__ static void qwen4exp_ple_conv_kernel(
        float *x, const float *gated, const float *pnorm, const float *hist, const float *taps,
        uint32_t hc_dim, uint32_t kern, uint32_t dil, uint32_t rows) {
    const uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (ch >= hc_dim || t >= rows) return;
    const float *tp = taps + (uint64_t)ch * kern;
    const int32_t hist_rows = (int32_t)((kern - 1u) * dil);
    const uint64_t cur = (uint64_t)t * hc_dim + ch;
    float acc = tp[kern - 1u] * pnorm[cur];
    for (uint32_t k = 0; k + 1u < kern; k++) {
        const int32_t src = (int32_t)t - (int32_t)((kern - 1u - k) * dil);
        const float v = src >= 0
            ? pnorm[(uint64_t)src * hc_dim + ch]
            : hist[(uint64_t)(hist_rows + src) * hc_dim + ch];
        acc = fmaf(tp[k], v, acc);
    }
    x[cur] += gated[cur] + qwen35_cuda_silu(acc);
}

extern "C" int ds4_gpu_qwen4exp_ple_conv(
        ds4_gpu_tensor       *x,            /* [rows][hc_dim] */
        const ds4_gpu_tensor *gated,        /* [rows][hc_dim] */
        const ds4_gpu_tensor *pnorm,        /* [rows][hc_dim] */
        ds4_gpu_tensor       *hist,         /* [(kernel-1)*dilation][hc_dim], slid forward afterwards */
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              taps_offset,  /* [hc_dim][kernel] f32 */
        uint32_t              hc_dim,
        uint32_t              kern,
        uint32_t              dil,
        uint32_t              rows) {
    const uint64_t n = (uint64_t)rows * hc_dim;
    const uint32_t hist_rows = (kern - 1u) * dil;
    if (!model_map || n == 0u || kern < 2u || dil == 0u || !qwen4exp_elems_fit(x, n) ||
        !qwen4exp_elems_fit(gated, n) || !qwen4exp_elems_fit(pnorm, n) ||
        !qwen4exp_elems_fit(hist, (uint64_t)hist_rows * hc_dim)) {
        return 0;
    }
    const float *taps = glm53_cuda_weight_f32(model_map, model_size, taps_offset, (uint64_t)hc_dim * kern,
                                              ds4_tensor_device_idx(x), "PLE conv");
    if (!taps) return 0;
    cudaStream_t stream = cuda_decode_stream();
    qwen4exp_ple_conv_kernel<<<dim3((hc_dim + 255u) / 256u, rows, 1u), 256, 0, stream>>>(
        (float *)x->ptr, (const float *)gated->ptr, (const float *)pnorm->ptr, (const float *)hist->ptr,
        taps, hc_dim, kern, dil, rows);
    qwen35_gdn_conv_state_kernel<<<(hc_dim + 255u) / 256u, 256, 0, stream>>>(
        (float *)hist->ptr, (const float *)pnorm->ptr, hc_dim, hist_rows, rows);
    return cuda_ok(cudaGetLastError(), "Flash-Next PLE conv launch");
}
