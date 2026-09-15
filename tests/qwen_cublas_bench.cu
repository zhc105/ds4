/* cuBLAS bf16 GEMM rate at the dense projection shapes of a Flash-Next
 * prefill chunk (out[M][N] = x[M][K] W[N][K]^T as ds4 calls it, f32 and
 * bf16 outputs), so the engine's GEMM time can be read against the card's
 * ceiling.  Built by `make tests/qwen_cublas_bench`. */
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static void *dmalloc(size_t n) {
    void *p = NULL;
    if (cudaMalloc(&p, n) != cudaSuccess) { fprintf(stderr, "cudaMalloc %zu failed\n", n); exit(1); }
    cudaMemset(p, 0, n);
    return p;
}

static void bench(cublasHandle_t h, const char *name, int M, int N, int K, cudaDataType out_type) {
    void *w = dmalloc((size_t)N * K * 2), *x = dmalloc((size_t)M * K * 2);
    void *o = dmalloc((size_t)M * N * (out_type == CUDA_R_32F ? 4 : 2));
    const float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const int reps = 10;
    for (int i = 0; i < 2 + reps; i++) {
        if (i == 2) cudaEventRecord(t0, 0);
        cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K,
                                         &beta, o, out_type, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s: cublas status %d\n", name, (int)st); exit(1); }
    }
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= reps;
    const double flop = 2.0 * M * (double)N * K;
    const double bytes = (double)N * K * 2 + (double)M * K * 2 + (double)M * N * (out_type == CUDA_R_32F ? 4 : 2);
    printf("cublas-bench: %-10s M %5d N %5d K %5d out %s: %7.3f ms  %6.1f TFLOP/s  %4.0f GB/s\n", name, M, N, K,
           out_type == CUDA_R_32F ? "f32 " : "bf16", ms, flop / ms / 1e9, bytes / ms / 1e6);
    cudaFree(w); cudaFree(x); cudaFree(o);
}

int main(int argc, char **argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 8192;
    cublasHandle_t h;
    cublasCreate(&h);
    struct { const char *name; int N, K; } shapes[] = {
        {"gdn_qkv", 10240, 2560}, {"gdn_z", 6144, 2560}, {"gdn_out", 2560, 6144},
        {"attn_q", 12288, 2560}, {"attn_kv", 1024, 2560}, {"attn_o", 2560, 6144},
        {"shexp_gu", 640, 2560}, {"shexp_down", 2560, 640}, {"router", 512, 2560},
        {"hc_low", 320, 10240}, {"hc_up", 10240, 320}, {"big_sq", 8192, 8192},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        bench(h, shapes[i].name, M, shapes[i].N, shapes[i].K, CUDA_R_32F);
        bench(h, shapes[i].name, M, shapes[i].N, shapes[i].K, CUDA_R_16BF);
    }
    cublasDestroy(h);
    return 0;
}
