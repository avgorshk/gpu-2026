#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace {

__global__ void GeluKernel(const float* input, float* output, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    const float x = input[i];
    const float x3 = x * x * x;

    constexpr float kTwoOverPi = 0.63662f;
    constexpr float kCoefficient = 0.044715f;

    const float z = kTwoOverPi * (x + kCoefficient * x3);

    const float exp_value = expf(-2.0f * z);
    const float tanh_value =
        (1.0f - exp_value) / (1.0f + exp_value);

    output[i] = 0.5f * x * (1.0f + tanh_value);
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    if (input.empty()) {
        return output;
    }

    float* device_input = nullptr;
    float* device_output = nullptr;

    const size_t size = input.size() * sizeof(float);

    cudaMalloc(&device_input, size);
    cudaMalloc(&device_output, size);

    cudaMemcpy(
        device_input,
        input.data(),
        size,
        cudaMemcpyHostToDevice
    );

    constexpr int kBlockSize = 256;
    const int block_count =
        (static_cast<int>(input.size()) + kBlockSize - 1) / kBlockSize;

    GeluKernel<<<block_count, kBlockSize>>>(
        device_input,
        device_output,
        static_cast<int>(input.size())
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        output.data(),
        device_output,
        size,
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_input);
    cudaFree(device_output);

    return output;
}
