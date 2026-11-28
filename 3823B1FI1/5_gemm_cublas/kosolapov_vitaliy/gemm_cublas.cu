#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace {

    struct CublasContext {
        cublasHandle_t handle = nullptr;
        float* memory = nullptr;
        std::size_t capacity = 0;

        CublasContext() {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
                handle = nullptr;
            }
        }

        ~CublasContext() {
            if (memory != nullptr) {
                cudaFree(memory);
            }

            if (handle != nullptr) {
                cublasDestroy(handle);
            }
        }

        bool Reserve(std::size_t bytes) {
            if (bytes <= capacity) {
                return true;
            }

            float* new_memory = nullptr;

            if (cudaMalloc(
                reinterpret_cast<void**>(&new_memory),
                bytes
            ) != cudaSuccess) {
                return false;
            }

            if (memory != nullptr) {
                cudaFree(memory);
            }

            memory = new_memory;
            capacity = bytes;

            return true;
        }
    };

    CublasContext& GetContext() {
        static CublasContext context;
        return context;
    }

}

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) {
        return {};
    }
    const std::size_t dim = static_cast<std::size_t>(n);
    const std::size_t elements = dim * dim;
    const std::size_t bytes = elements * sizeof(float);
    const std::size_t total_bytes = 3 * bytes;

    CublasContext& ctx = GetContext();

    if (ctx.handle == nullptr) {
        return {};
    }

    if (!ctx.Reserve(total_bytes)) {
        return {};
    }

    float* device_a = ctx.memory;
    float* device_b = device_a + elements;
    float* device_c = device_b + elements;

    cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice);

    std::vector<float> result(elements);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const cublasStatus_t status = cublasSgemm(
        ctx.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &alpha,
        device_b, n,
        device_a, n,
        &beta,
        device_c, n
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        return {};
    }
    if (cudaMemcpy(result.data(), device_c, bytes,
        cudaMemcpyDeviceToHost) != cudaSuccess) {
        return {};
    }

    return result;
}
