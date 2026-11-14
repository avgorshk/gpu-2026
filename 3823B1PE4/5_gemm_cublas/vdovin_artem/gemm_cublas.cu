#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

void Check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void Check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS error " + std::to_string(status));
    }
}

cublasHandle_t g_handle = nullptr;
float* g_device = nullptr;
std::size_t g_capacity = 0;

cublasHandle_t GetHandle() {
    if (g_handle == nullptr) {
        Check(cublasCreate(&g_handle), "cublasCreate");
    }
    return g_handle;
}

float* GetDeviceBuffer(std::size_t size) {
    if (size > g_capacity) {
        if (g_device != nullptr) {
            cudaFree(g_device);
            g_device = nullptr;
        }
        Check(cudaMalloc(&g_device, size * sizeof(float)), "cudaMalloc");
        g_capacity = size;
    }
    return g_device;
}

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const std::size_t count = static_cast<std::size_t>(n) * n;
    std::vector<float> c(count);
    if (n <= 0) {
        return c;
    }

    const std::size_t bytes = count * sizeof(float);
    cublasHandle_t handle = GetHandle();
    float* d_a = GetDeviceBuffer(3 * count);
    float* d_b = d_a + count;
    float* d_c = d_b + count;

    Check(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
    Check(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    Check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      n, n, n,
                      &alpha,
                      d_b, n,
                      d_a, n,
                      &beta,
                      d_c, n),
          "cublasSgemm");

    Check(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");
    return c;
}
