#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoef = 0.044715f;

__device__ __forceinline__ float FastTanh(float x)
{
    return 1.0f - 2.0f / (expf(2.0f * x) + 1.0f);
}

__global__ void GeluKernel(const float* input, float* output, std::size_t size)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    const float x = input[i];
    const float x2 = x * x;
    const float x3 = x2 * x;

    const float z =
        kSqrt2OverPi * (x + kGeluCoef * x3);

    output[i] = 0.5f * x * (1.0f + FastTanh(z));
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    std::vector<float> output(input.size());

    if (input.empty())
        return output;

    const std::size_t size = input.size();
    const std::size_t bytes = size * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);

    constexpr int blockSize = 256;
    const int gridSize =
        static_cast<int>((size + blockSize - 1) / blockSize);

    GeluKernel<<<gridSize, blockSize>>>(d_input, d_output, size);

    cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}