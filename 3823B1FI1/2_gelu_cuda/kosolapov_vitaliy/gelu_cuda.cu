#include "gelu_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

void CheckCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") +
            cudaGetErrorString(err)
        );
    }
}
__global__ void GeluKernel(const float* input, float* output, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    float x = input[idx];
    constexpr float c = 0.7978845608028654f;
    float x3 = x * x * x;
    float z = c * (x + 0.044715f * x3);
    float q = expf(-2.0f * fabsf(z));
    float sign = (z >= 0.0f) ? 1.0f : -1.0f;
    float tanh_z = sign * (1.0f - q) / (1.0f + q);
    output[idx] = 0.5f * x * (1.0f + tanh_z);
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0)
        return {};
    constexpr unsigned int threads = 256;
    const size_t blocks_count = n / threads + (n % threads != 0);
    const unsigned int blocks = static_cast<unsigned int>(blocks_count);
    std::vector<float> output(n);
    float* d_buffer = nullptr;
    const size_t bytes = n * sizeof(float);
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_buffer), 2 * bytes));
    float* d_input = d_buffer;
    float* d_output = d_buffer + n;
    CheckCuda(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
    GeluKernel <<<blocks, threads>>>(d_input, d_output, n);
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    CheckCuda(cudaFree(d_buffer));

    return output;
}