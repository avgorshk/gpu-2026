#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

__global__ void GeluKernel(const float* input, float* output, std::size_t size) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    const float x = input[index];
    const float x3 = x * x * x;
    output[index] = 0.5f * x * (1.0f + tanhf(kSqrtTwoOverPi * (x + 0.044715f * x3)));
}

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    if (input.empty()) {
        return output;
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    const std::size_t bytes = input.size() * sizeof(float);
    CheckCuda(cudaMalloc(&device_input, bytes), "cudaMalloc(input)");
    try {
        CheckCuda(cudaMalloc(&device_output, bytes), "cudaMalloc(output)");
        CheckCuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(input)");

        constexpr int block_size = 256;
        const int grid_size = static_cast<int>((input.size() + block_size - 1) / block_size);
        GeluKernel<<<grid_size, block_size>>>(device_input, device_output, input.size());
        CheckCuda(cudaGetLastError(), "GeluKernel launch");
        CheckCuda(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
    } catch (...) {
        cudaFree(device_output);
        cudaFree(device_input);
        throw;
    }
    CheckCuda(cudaFree(device_output), "cudaFree(output)");
    CheckCuda(cudaFree(device_input), "cudaFree(input)");
    return output;
}
