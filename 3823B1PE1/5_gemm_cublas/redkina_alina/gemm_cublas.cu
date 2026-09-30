#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);

    static cublasHandle_t handle = nullptr;
    static float* d = nullptr;
    static int cap = 0;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    if (n > cap) {
        cudaFree(d);
        cudaMalloc(&d, 3 * bytes);
        cap = n;
    }

    float* d_a = d;
    float* d_b = d + count;
    float* d_c = d_b + count;

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (n > 0) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_b, n, d_a, n, &beta, d_c, n);
    }

    std::vector<float> c(count);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    return c;
}
