#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cstring>

__global__ void GeluKernel(const float* input, float* output, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float value = input[index];
    const float value_cubed = value * value * value;
    const float tanh_argument = 0.7978845608028654f * (value + 0.044715f * value_cubed);
    output[index] = value / (1.0f + expf(-2.0f * tanh_argument));
}

struct GpuBuffers {
    float* device_input = nullptr;
    float* device_output = nullptr;
    float* pinned_input = nullptr;
    cudaStream_t stream = nullptr;
    int capacity = 0;

    void Ensure(int count, std::size_t bytes) {
        if (stream == nullptr) cudaStreamCreate(&stream);
        if (count <= capacity) return;

        cudaFree(device_input);
        cudaFree(device_output);
        cudaFreeHost(pinned_input);
        cudaMalloc(&device_input, bytes);
        cudaMalloc(&device_output, bytes);
        cudaMallocHost(&pinned_input, bytes);
        capacity = count;
    }
};

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int count = static_cast<int>(input.size());
    if (count == 0) return {};

    static GpuBuffers buffers;
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
    buffers.Ensure(count, bytes);

    std::memcpy(buffers.pinned_input, input.data(), bytes);
    cudaMemcpyAsync(buffers.device_input, buffers.pinned_input, bytes, cudaMemcpyHostToDevice, buffers.stream);
    const int threads = 256;
    GeluKernel<<<(count + threads - 1) / threads, threads, 0, buffers.stream>>>(
        buffers.device_input, buffers.device_output, count);

    std::vector<float> result(count);
    cudaMemcpyAsync(result.data(), buffers.device_output, bytes, cudaMemcpyDeviceToHost, buffers.stream);
    cudaStreamSynchronize(buffers.stream);
    return result;
}
