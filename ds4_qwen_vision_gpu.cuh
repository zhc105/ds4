/* Qwen3.8-Flash-Next vision tower: the Qwen3-VL SigLIP ViT the checkpoint
 * ships (27 blocks of 1152, 16 heads of 72, LayerNorm with bias, 2-D RoPE
 * over the whole head, GELU(tanh) MLP, a learned 48x48 position grid
 * resampled to the image, a 2x2 merger to the 2560-wide language
 * embedding).  Rows are patches in 2x2 block order, which the preprocessor,
 * the RoPE positions and the merger all share.  Activations stay f32 over
 * the checkpoint's bf16 weights.  Included by ds4_cuda.cu after the GLM
 * vision file, whose bf16 loads, bias kernel and weight lookup it reuses. */

#ifndef DS4_QWEN_VISION_TYPES_DEFINED
#define DS4_QWEN_VISION_TYPES_DEFINED
#define DS4_QWEN_VISION_LAYERS 27u
typedef struct {
    uint64_t norm1_weight;
    uint64_t norm1_bias;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2_weight;
    uint64_t norm2_bias;
    uint64_t fc1_weight;
    uint64_t fc1_bias;
    uint64_t fc2_weight;
    uint64_t fc2_bias;
} ds4_qwen_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t pos_embed;
    uint64_t merger_norm_weight;
    uint64_t merger_norm_bias;
    uint64_t merger_fc1_weight;
    uint64_t merger_fc1_bias;
    uint64_t merger_fc2_weight;
    uint64_t merger_fc2_bias;
    ds4_qwen_vision_layer_weights layer[DS4_QWEN_VISION_LAYERS];
} ds4_qwen_vision_weights;
#endif

#define DS4_QWEN_VISION_STREAM cuda_decode_stream()
#define QWEN_VISION_DIM 1152u
#define QWEN_VISION_HEADS 16u
#define QWEN_VISION_HEAD_DIM 72u
#define QWEN_VISION_HALF 36u
#define QWEN_VISION_QKV (3u * QWEN_VISION_DIM)
#define QWEN_VISION_FF 4304u
#define QWEN_VISION_PATCH 1536u   /* 3 channels x 2 frames x 16 x 16 */
#define QWEN_VISION_GRID 48u
#define QWEN_VISION_MERGED (4u * QWEN_VISION_DIM)
#define QWEN_VISION_OUT 2560u

__global__ static void qwen_vision_layernorm_kernel(
        float          *out,
        const float    *x,
        const uint16_t *weight,
        const uint16_t *bias,
        uint32_t        width,
        float           eps) {
    __shared__ float partial[256];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * width;
    float *yr = out + (uint64_t)row * width;
    float sum = 0.0f;
    for (uint32_t d = tid; d < width; d += blockDim.x) sum += xr[d];
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float mean = partial[0] / (float)width;
    float var = 0.0f;
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        const float centered = xr[d] - mean;
        var = fmaf(centered, centered, var);
    }
    __syncthreads();
    partial[tid] = var;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / (float)width + eps);
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        yr[d] = (xr[d] - mean) * inv * glm53_vision_bf16(weight + d) +
                glm53_vision_bf16(bias + d);
    }
}

/* Grid position of patch row `row` in 2x2 block order. */
__device__ __forceinline__ static void qwen_vision_patch_pos(
        uint32_t row, uint32_t grid_w, uint32_t *h, uint32_t *w) {
    const uint32_t block = row / 4u;
    const uint32_t within = row & 3u;
    const uint32_t blocks_w = grid_w / 2u;
    *h = (block / blocks_w) * 2u + within / 2u;
    *w = (block % blocks_w) * 2u + within % 2u;
}

/* The learned 48x48 grid resampled to the image grid (bilinear, corners
 * aligned) and added to the patch embedding. */
__global__ static void qwen_vision_pos_embed_kernel(
        float          *x,
        const uint16_t *table,
        uint32_t        rows,
        uint32_t        grid_h,
        uint32_t        grid_w) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * QWEN_VISION_DIM) return;
    const uint32_t row = (uint32_t)(i / QWEN_VISION_DIM);
    const uint32_t d = (uint32_t)(i % QWEN_VISION_DIM);
    uint32_t h, w;
    qwen_vision_patch_pos(row, grid_w, &h, &w);
    const float side = (float)(QWEN_VISION_GRID - 1u);
    const float sh = grid_h > 1u ? (float)h * side / (float)(grid_h - 1u) : 0.0f;
    const float sw = grid_w > 1u ? (float)w * side / (float)(grid_w - 1u) : 0.0f;
    const uint32_t h0 = (uint32_t)floorf(sh), w0 = (uint32_t)floorf(sw);
    const uint32_t h1 = min(h0 + 1u, QWEN_VISION_GRID - 1u);
    const uint32_t w1 = min(w0 + 1u, QWEN_VISION_GRID - 1u);
    const float fh = sh - (float)h0, fw = sw - (float)w0;
#define QWEN_POS(hh, ww) glm53_vision_bf16(table + ((uint64_t)(hh) * QWEN_VISION_GRID + (ww)) * QWEN_VISION_DIM + d)
    x[i] += (1.0f - fh) * ((1.0f - fw) * QWEN_POS(h0, w0) + fw * QWEN_POS(h0, w1)) +
            fh * ((1.0f - fw) * QWEN_POS(h1, w0) + fw * QWEN_POS(h1, w1));
#undef QWEN_POS
}

/* QKV bias and the vision RoPE: pairs (i, i + 36) of a head rotate by the
 * patch row for i < 18 and by its column for 18 <= i < 36, with the
 * frequency of i mod 18. */
__global__ static void qwen_vision_qkv_rope_kernel(
        float          *q,
        float          *k,
        float          *v,
        const float    *qkv,
        const uint16_t *bias,
        uint32_t        rows,
        uint32_t        grid_w) {
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (row >= rows || lane >= QWEN_VISION_HALF) return;
    const uint64_t in = (uint64_t)row * QWEN_VISION_QKV + (uint64_t)head * QWEN_VISION_HEAD_DIM;
    const uint64_t out = (uint64_t)row * QWEN_VISION_DIM + (uint64_t)head * QWEN_VISION_HEAD_DIM;
    const uint32_t b = head * QWEN_VISION_HEAD_DIM + lane;
    const float q0 = qkv[in + lane] + glm53_vision_bf16(bias + b);
    const float q1 = qkv[in + lane + QWEN_VISION_HALF] + glm53_vision_bf16(bias + b + QWEN_VISION_HALF);
    const float k0 = qkv[in + QWEN_VISION_DIM + lane] + glm53_vision_bf16(bias + QWEN_VISION_DIM + b);
    const float k1 = qkv[in + QWEN_VISION_DIM + lane + QWEN_VISION_HALF] +
                     glm53_vision_bf16(bias + QWEN_VISION_DIM + b + QWEN_VISION_HALF);
    uint32_t h, w;
    qwen_vision_patch_pos(row, grid_w, &h, &w);
    const uint32_t pos = lane < 18u ? h : w;
    const float inv_freq = powf(10000.0f, -(float)(lane % 18u) / 18.0f);
    const float angle = (float)pos * inv_freq;
    const float cs = cosf(angle);
    const float sn = sinf(angle);
    q[out + lane] = q0 * cs - q1 * sn;
    q[out + lane + QWEN_VISION_HALF] = q1 * cs + q0 * sn;
    k[out + lane] = k0 * cs - k1 * sn;
    k[out + lane + QWEN_VISION_HALF] = k1 * cs + k0 * sn;
    v[out + lane] = qkv[in + 2u * QWEN_VISION_DIM + lane] +
                    glm53_vision_bf16(bias + 2u * QWEN_VISION_DIM + b);
    v[out + lane + QWEN_VISION_HALF] = qkv[in + 2u * QWEN_VISION_DIM + lane + QWEN_VISION_HALF] +
                    glm53_vision_bf16(bias + 2u * QWEN_VISION_DIM + b + QWEN_VISION_HALF);
}

/* One block per row of attention scores (a query of one head against every
 * key): softmax in place, the threads striding over the keys. */
__global__ static void qwen_vision_softmax_kernel(float *scores, uint32_t keys) {
    __shared__ float partial[256];
    const uint32_t tid = threadIdx.x;
    float *s = scores + ((uint64_t)blockIdx.y * gridDim.x + blockIdx.x) * keys;
    float top = -INFINITY;
    for (uint32_t i = tid; i < keys; i += blockDim.x) top = fmaxf(top, s[i]);
    partial[tid] = top;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    top = partial[0];
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t i = tid; i < keys; i += blockDim.x) {
        s[i] = expf(s[i] - top);
        sum += s[i];
    }
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = 1.0f / partial[0];
    for (uint32_t i = tid; i < keys; i += blockDim.x) s[i] *= inv;
}

static int qwen_vision_launch_ok(const char *label) {
    return cuda_ok(cudaGetLastError(), label);
}

/* Full bidirectional attention over the patches.  A screenshot is 8000 of
 * them and every one attends to every other in 27 blocks, so the products
 * are cuBLAS's: for a chunk of queries, the scores of every head against all
 * keys, their softmax, and the weighted sum of the values.  q, k, v and out
 * are [patch][head][72]; read with a leading dimension of a whole patch row
 * and a stride of one head, each head is a matrix of its own as it lies, so
 * nothing is repacked.  `scores` holds a chunk: [head][query][key].  The
 * products are pedantic f32 whatever the handle's math mode: TF32's ten
 * mantissa bits on a logit go through the exponential. */
#define QWEN_VISION_SCORE_BYTES ((uint64_t)512 << 20)

static uint32_t qwen_vision_attention_chunk(uint32_t rows) {
    const uint64_t fit = QWEN_VISION_SCORE_BYTES / ((uint64_t)QWEN_VISION_HEADS * rows * sizeof(float));
    return fit >= rows ? rows : fit > 0u ? (uint32_t)fit : 1u;
}

static int qwen_vision_attention(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        ds4_gpu_tensor       *scores,
        uint32_t              rows) {
    const uint32_t chunk = qwen_vision_attention_chunk(rows);
    const float scale = 1.0f / sqrtf((float)QWEN_VISION_HEAD_DIM), one = 1.0f, zero = 0.0f;
    const cublasHandle_t handle = cuda_cublas_for_tier(ds4_tensor_device_idx(out));
    for (uint32_t at = 0; at < rows; at += chunk) {
        const uint32_t n = rows - at < chunk ? rows - at : chunk;
        cublasStatus_t st = cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)rows, (int)n, (int)QWEN_VISION_HEAD_DIM, &scale,
                k->ptr, CUDA_R_32F, (int)QWEN_VISION_DIM, (long long)QWEN_VISION_HEAD_DIM,
                (const float *)q->ptr + (uint64_t)at * QWEN_VISION_DIM, CUDA_R_32F, (int)QWEN_VISION_DIM,
                (long long)QWEN_VISION_HEAD_DIM, &zero,
                scores->ptr, CUDA_R_32F, (int)rows, (long long)rows * n, (int)QWEN_VISION_HEADS,
                CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "Qwen vision attention scores")) return 0;
        qwen_vision_softmax_kernel<<<dim3(n, QWEN_VISION_HEADS, 1u), 256u, 0,
            DS4_QWEN_VISION_STREAM>>>((float *)scores->ptr, rows);
        if (!qwen_vision_launch_ok("Qwen vision attention softmax")) return 0;
        st = cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)QWEN_VISION_HEAD_DIM, (int)n, (int)rows, &one,
                v->ptr, CUDA_R_32F, (int)QWEN_VISION_DIM, (long long)QWEN_VISION_HEAD_DIM,
                scores->ptr, CUDA_R_32F, (int)rows, (long long)rows * n, &zero,
                (float *)out->ptr + (uint64_t)at * QWEN_VISION_DIM, CUDA_R_32F, (int)QWEN_VISION_DIM,
                (long long)QWEN_VISION_HEAD_DIM, (int)QWEN_VISION_HEADS,
                CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "Qwen vision attention values")) return 0;
    }
    return 1;
}

/* Bias plus GELU in place: the blocks use the tanh form, the merger the
 * exact one. */
__global__ static void qwen_vision_gelu_bias_kernel(
        float          *x,
        const uint16_t *bias,
        uint64_t        count,
        uint32_t        width,
        bool            tanh_form) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = x[i] + glm53_vision_bf16(bias + i % width);
    if (tanh_form) {
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    } else {
        x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    }
}

static int qwen_vision_layernorm(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t rows,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t bias_offset, const char *label) {
    const uint16_t *weight = glm53_vision_weight(model_map, model_size, weight_offset, QWEN_VISION_DIM, label);
    const uint16_t *bias = glm53_vision_weight(model_map, model_size, bias_offset, QWEN_VISION_DIM, label);
    if (!weight || !bias) return 0;
    qwen_vision_layernorm_kernel<<<rows, 256u, 0, DS4_QWEN_VISION_STREAM>>>(
        (float *)out->ptr, (const float *)x->ptr, weight, bias, QWEN_VISION_DIM, 1.0e-6f);
    return qwen_vision_launch_ok(label);
}

/* out = x + bias (+ residual). */
static int qwen_vision_bias(
        ds4_gpu_tensor *x, const ds4_gpu_tensor *residual, uint64_t count, uint32_t width,
        const void *model_map, uint64_t model_size, uint64_t bias_offset, const char *label) {
    const uint16_t *bias = glm53_vision_weight(model_map, model_size, bias_offset, width, label);
    if (!bias) return 0;
    glm53_vision_bias_kernel<<<(unsigned)((count + 255u) / 256u), 256u, 0, DS4_QWEN_VISION_STREAM>>>(
        (float *)x->ptr, bias, residual ? (const float *)residual->ptr : NULL, count, width);
    return qwen_vision_launch_ok(label);
}

static int qwen_vision_gelu(
        ds4_gpu_tensor *x, uint64_t count, uint32_t width, bool tanh_form,
        const void *model_map, uint64_t model_size, uint64_t bias_offset, const char *label) {
    const uint16_t *bias = glm53_vision_weight(model_map, model_size, bias_offset, width, label);
    if (!bias) return 0;
    qwen_vision_gelu_bias_kernel<<<(unsigned)((count + 255u) / 256u), 256u, 0, DS4_QWEN_VISION_STREAM>>>(
        (float *)x->ptr, bias, count, width, tanh_form);
    return qwen_vision_launch_ok(label);
}

extern "C" int ds4_gpu_qwen_vision_attention(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        uint32_t              rows) {
    const uint64_t bytes = (uint64_t)rows * QWEN_VISION_DIM * sizeof(float);
    if (!out || !q || !k || !v || rows == 0u || !g_cublas_ready ||
        out->bytes < bytes || q->bytes < bytes || k->bytes < bytes || v->bytes < bytes) return 0;
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(
            (uint64_t)QWEN_VISION_HEADS * qwen_vision_attention_chunk(rows) * rows * sizeof(float));
    const int ok = scores && qwen_vision_attention(out, q, k, v, scores, rows);
    ds4_gpu_tensor_free(scores);
    return ok;
}

extern "C" int ds4_gpu_qwen_vision_encode(
        float                         *out,
        const float                   *patches,
        uint32_t                       grid_h,
        uint32_t                       grid_w,
        const void                    *model_map,
        uint64_t                       model_size,
        const ds4_qwen_vision_weights *weights) {
    if (!out || !patches || !model_map || !weights || grid_h == 0u || grid_w == 0u ||
        (grid_h & 1u) != 0u || (grid_w & 1u) != 0u || grid_h > UINT32_MAX / grid_w) return 0;
    const uint32_t rows = grid_h * grid_w;
    const uint32_t merged = rows / 4u;
    const uint64_t row_dim = (uint64_t)rows * QWEN_VISION_DIM;
    const uint64_t row_ff = (uint64_t)rows * QWEN_VISION_FF;
    const uint64_t merged_out = (uint64_t)merged * QWEN_VISION_OUT;
    if (row_ff > SIZE_MAX / sizeof(float)) return 0;

    ds4_gpu_tensor *patch = NULL, *a = NULL, *b = NULL, *qkv = NULL;
    ds4_gpu_tensor *q = NULL, *k = NULL, *v = NULL, *attn = NULL, *scores = NULL, *mid = NULL, *proj = NULL;
    ds4_gpu_tensor *cur = NULL, *tmp = NULL;
    int ok = 0;
#define VISION_ALLOC(name_, count_) do { \
        name_ = ds4_gpu_tensor_alloc((count_) * sizeof(float)); \
        if (!(name_)) goto cleanup; \
    } while (0)
    VISION_ALLOC(patch, (uint64_t)rows * QWEN_VISION_PATCH);
    VISION_ALLOC(a, row_dim);
    VISION_ALLOC(b, row_dim);
    VISION_ALLOC(qkv, (uint64_t)rows * QWEN_VISION_QKV);
    VISION_ALLOC(q, row_dim);
    VISION_ALLOC(k, row_dim);
    VISION_ALLOC(v, row_dim);
    VISION_ALLOC(attn, row_dim);
    VISION_ALLOC(scores, (uint64_t)QWEN_VISION_HEADS * qwen_vision_attention_chunk(rows) * rows);
    VISION_ALLOC(mid, row_ff);   /* the block MLP, then the merger's 4608-wide rows */
    VISION_ALLOC(proj, merged_out);
#undef VISION_ALLOC
    if (!ds4_gpu_tensor_write(patch, 0, patches, (uint64_t)rows * QWEN_VISION_PATCH * sizeof(float)) ||
        !ds4_gpu_begin_commands()) goto cleanup;

    ok = ds4_gpu_glm53_matmul_bf16(a, model_map, model_size, weights->patch_weight,
                                   QWEN_VISION_PATCH, QWEN_VISION_DIM, patch, rows) &&
         qwen_vision_bias(a, NULL, row_dim, QWEN_VISION_DIM, model_map, model_size,
                          weights->patch_bias, "Qwen vision patch bias");
    if (ok) {
        const uint16_t *table = glm53_vision_weight(model_map, model_size, weights->pos_embed,
                (uint64_t)QWEN_VISION_GRID * QWEN_VISION_GRID * QWEN_VISION_DIM, "Qwen vision position grid");
        ok = table != NULL;
        if (ok) {
            qwen_vision_pos_embed_kernel<<<(unsigned)((row_dim + 255u) / 256u), 256u, 0,
                DS4_QWEN_VISION_STREAM>>>((float *)a->ptr, table, rows, grid_h, grid_w);
            ok = qwen_vision_launch_ok("Qwen vision position embedding");
        }
    }

    cur = a;
    tmp = b;
    for (uint32_t il = 0; ok && il < DS4_QWEN_VISION_LAYERS; il++) {
        const ds4_qwen_vision_layer_weights *w = &weights->layer[il];
        ok = qwen_vision_layernorm(tmp, cur, rows, model_map, model_size,
                                   w->norm1_weight, w->norm1_bias, "Qwen vision norm1") &&
             ds4_gpu_glm53_matmul_bf16(qkv, model_map, model_size, w->qkv_weight,
                                       QWEN_VISION_DIM, QWEN_VISION_QKV, tmp, rows);
        if (ok) {
            const uint16_t *bias = glm53_vision_weight(model_map, model_size, w->qkv_bias,
                                                       QWEN_VISION_QKV, "Qwen vision QKV bias");
            ok = bias != NULL;
            if (ok) {
                qwen_vision_qkv_rope_kernel<<<dim3(rows, QWEN_VISION_HEADS, 1u), QWEN_VISION_HALF, 0,
                    DS4_QWEN_VISION_STREAM>>>((float *)q->ptr, (float *)k->ptr, (float *)v->ptr,
                                              (const float *)qkv->ptr, bias, rows, grid_w);
                ok = qwen_vision_launch_ok("Qwen vision QKV");
            }
        }
        if (ok) ok = qwen_vision_attention(attn, q, k, v, scores, rows);
        if (ok) {
            ok = ds4_gpu_glm53_matmul_bf16(tmp, model_map, model_size, w->attn_proj_weight,
                                           QWEN_VISION_DIM, QWEN_VISION_DIM, attn, rows) &&
                 qwen_vision_bias(tmp, cur, row_dim, QWEN_VISION_DIM, model_map, model_size,
                                  w->attn_proj_bias, "Qwen vision attention residual");
        }
        ds4_gpu_tensor *swap = cur; cur = tmp; tmp = swap;
        if (ok) {
            ok = qwen_vision_layernorm(tmp, cur, rows, model_map, model_size,
                                       w->norm2_weight, w->norm2_bias, "Qwen vision norm2") &&
                 ds4_gpu_glm53_matmul_bf16(mid, model_map, model_size, w->fc1_weight,
                                           QWEN_VISION_DIM, QWEN_VISION_FF, tmp, rows) &&
                 qwen_vision_gelu(mid, row_ff, QWEN_VISION_FF, true, model_map, model_size,
                                  w->fc1_bias, "Qwen vision MLP GELU") &&
                 ds4_gpu_glm53_matmul_bf16(tmp, model_map, model_size, w->fc2_weight,
                                           QWEN_VISION_FF, QWEN_VISION_DIM, mid, rows) &&
                 qwen_vision_bias(tmp, cur, row_dim, QWEN_VISION_DIM, model_map, model_size,
                                  w->fc2_bias, "Qwen vision MLP residual");
        }
        swap = cur; cur = tmp; tmp = swap;
    }

    /* Merger: per-patch LayerNorm, then the four patches of a block read as
     * one 4608-wide row (the row-major layout already is that view). */
    if (ok) {
        ok = qwen_vision_layernorm(tmp, cur, rows, model_map, model_size,
                                   weights->merger_norm_weight, weights->merger_norm_bias,
                                   "Qwen vision merger norm") &&
             ds4_gpu_glm53_matmul_bf16(mid, model_map, model_size, weights->merger_fc1_weight,
                                       QWEN_VISION_MERGED, QWEN_VISION_MERGED, tmp, merged) &&
             qwen_vision_gelu(mid, (uint64_t)merged * QWEN_VISION_MERGED, QWEN_VISION_MERGED, false,
                              model_map, model_size, weights->merger_fc1_bias, "Qwen vision merger GELU") &&
             ds4_gpu_glm53_matmul_bf16(proj, model_map, model_size, weights->merger_fc2_weight,
                                       QWEN_VISION_MERGED, QWEN_VISION_OUT, mid, merged) &&
             qwen_vision_bias(proj, NULL, merged_out, QWEN_VISION_OUT, model_map, model_size,
                              weights->merger_fc2_bias, "Qwen vision merger bias");
    }
    if (ds4_gpu_end_commands() == 0) ok = 0;
    if (ok) ok = ds4_gpu_tensor_read(proj, 0, out, merged_out * sizeof(float));

cleanup:
    ds4_gpu_tensor_free(proj);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(scores);
    ds4_gpu_tensor_free(attn);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(qkv);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(patch);
    return ok;
}

template <typename T>
__device__ __forceinline__ static T qwen_vision_store(float v);
template <>
__device__ __forceinline__ float qwen_vision_store<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 qwen_vision_store<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T>
__global__ static void qwen4exp_scatter_image_kernel(
        T           *x,
        const float *image,
        uint32_t     dst_row,
        uint32_t     image_row,
        uint32_t     rows,
        uint32_t     total_rows,
        uint32_t     width,
        uint32_t     n_hc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)rows * n_hc * width) return;
    const uint32_t d = (uint32_t)(i % width);
    const uint64_t linear_row = i / width;
    const uint32_t delta = (uint32_t)(linear_row / n_hc);
    const uint32_t stream = (uint32_t)(linear_row % n_hc);
    if (dst_row + delta >= total_rows) return;
    x[((uint64_t)(dst_row + delta) * n_hc + stream) * width + d] =
        qwen_vision_store<T>(image[(uint64_t)(image_row + delta) * width + d]);
}

extern "C" int ds4_gpu_qwen4exp_scatter_image(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *image,
        uint32_t              dst_row,
        uint32_t              image_row,
        uint32_t              rows,
        uint32_t              total_rows,
        uint32_t              n_embd,
        uint32_t              n_hc) {
    if (!x || !image || rows == 0u || n_embd == 0u || n_hc == 0u ||
        dst_row > total_rows || rows > total_rows - dst_row) return 0;
    const uint64_t count = (uint64_t)rows * n_hc * n_embd;
    const unsigned blocks = (unsigned)((count + 255u) / 256u);
    if (n_hc > 1u) {
        qwen4exp_scatter_image_kernel<__nv_bfloat16><<<blocks, 256u, 0, DS4_QWEN_VISION_STREAM>>>(
            (__nv_bfloat16 *)x->ptr, (const float *)image->ptr,
            dst_row, image_row, rows, total_rows, n_embd, n_hc);
    } else {
        qwen4exp_scatter_image_kernel<float><<<blocks, 256u, 0, DS4_QWEN_VISION_STREAM>>>(
            (float *)x->ptr, (const float *)image->ptr,
            dst_row, image_row, rows, total_rows, n_embd, n_hc);
    }
    return qwen_vision_launch_ok("Qwen vision scatter");
}

#undef DS4_QWEN_VISION_STREAM
