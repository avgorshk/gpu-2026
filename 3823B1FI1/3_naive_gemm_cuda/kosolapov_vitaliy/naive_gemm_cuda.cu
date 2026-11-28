#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

    class DeviceBuffer {
    public:
        explicit DeviceBuffer(std::size_t bytes) {
            cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes);            
        }

        ~DeviceBuffer() {
            if (ptr_ != nullptr) {
                (void)cudaFree(ptr_);
            }
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        float* get() const noexcept {
            return ptr_;
        }

    private:
        float* ptr_ = nullptr;
    };
    __global__ void GemmNaiveScalar(const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        int n) {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row >= n || col >= n) {
            return;
        }

        const std::size_t row_offset = static_cast<std::size_t>(row) * n;
        float sum = 0.0f;

#pragma unroll 4
        for (int k = 0; k < n; ++k) {
            const float a_value = A[row_offset + k];
            const float b_value = B[static_cast<std::size_t>(k) * n + col];
            sum = fmaf(a_value, b_value, sum);
        }

        C[row_offset + col] = sum;
    }
    __global__ void GemmNaiveFloat4(const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        int n) {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

        const std::size_t a_row_offset = static_cast<std::size_t>(row) * n;
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll 4
        for (int k = 0; k < n; ++k) {
            const float a_value = A[a_row_offset + k];
            const std::size_t b_offset = static_cast<std::size_t>(k) * n + col;
            const float4 b_value =
                *reinterpret_cast<const float4*>(B + b_offset);

            sum.x = fmaf(a_value, b_value.x, sum.x);
            sum.y = fmaf(a_value, b_value.y, sum.y);
            sum.z = fmaf(a_value, b_value.z, sum.z);
            sum.w = fmaf(a_value, b_value.w, sum.w);
        }

        const std::size_t c_offset = a_row_offset + col;
        *reinterpret_cast<float4*>(C + c_offset) = sum;
    }

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t ns = static_cast<std::size_t>(n);
    const std::size_t elements = ns * ns;
    const std::size_t matrix_bytes = elements * sizeof(float);
    std::vector<float> result(elements);
    DeviceBuffer device_memory(3 * matrix_bytes);
    float* d_a = device_memory.get();
    float* d_b = d_a + elements;
    float* d_c = d_b + elements;
    cudaMemcpy(d_a, a.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), matrix_bytes, cudaMemcpyHostToDevice);
    const dim3 block(32, 8);
    if (n >= 128) {
        const dim3 grid(n / 128, (n + 7) / 8);
        GemmNaiveFloat4 <<<grid, block>>> (d_a, d_b, d_c, n);
    }
    else {
        const dim3 grid((n + 31) / 32, (n + 7) / 8);
        GemmNaiveScalar <<<grid, block>>> (d_a, d_b, d_c, n);
    }
    cudaMemcpy(result.data(), d_c, matrix_bytes, cudaMemcpyDeviceToHost);
    return result;
}
