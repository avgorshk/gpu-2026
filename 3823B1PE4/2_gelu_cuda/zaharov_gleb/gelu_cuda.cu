#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct DeviceDeleter {
    void operator()(float* pointer) const noexcept {
        cudaFree(pointer);
    }
};

__global__ void GeluKernel(float* data, std::size_t size) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    constexpr float twice_sqrt_two_over_pi = 1.5957691216057308f;

    for (std::size_t i = start; i < size; i += stride) {
        const float x = data[i];
        const float argument = twice_sqrt_two_over_pi * (x + 0.044715f * x * x * x);
        // 0.5 * (1 + tanh(t)) = 1 / (1 + exp(-2 * t)).
        data[i] = x / (1.0f + expf(-argument));
    }
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const std::size_t size = input.size();
    const std::size_t bytes = size * sizeof(float);
    float* pointer = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes), "cudaMalloc");
    const std::unique_ptr<float, DeviceDeleter> device_data(pointer);

    CheckCuda(cudaMemcpy(device_data.get(), input.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host to device");

    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>((size - 1) / threads + 1, 65535));
    GeluKernel<<<blocks, threads>>>(device_data.get(), size);
    CheckCuda(cudaGetLastError(), "GeluKernel launch");

    std::vector<float> result(size);
    CheckCuda(cudaMemcpy(result.data(), device_data.get(), bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy device to host");
    return result;
}
