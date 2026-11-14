#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    size_t bytes = (size_t)n * n * sizeof(float);
    std::vector<float> c((size_t)n * n);

    static cublasHandle_t handle = nullptr;
    static float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    static size_t cap = 0;

    if (!handle) cublasCreate(&handle);
    if (bytes > cap) {
        if (d_a) { cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); }
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cap = bytes;
    }

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, 
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}