#include <vector>
#include <cstdio>

#include <cuda_runtime.h>
#include "gelu_cuda.h"

namespace {
constexpr float LOG2E = 1.44269504089f;         // log2(e)
constexpr float A_LOG2E = 1.59576912f * LOG2E;  // 2 * sqrt(2/pi) * log2(e)
constexpr float B_LOG2E = 0.0713548f  * LOG2E;  // 2 * sqrt(2/pi) * 0.044715 * log2(e)

constexpr int THREADS_PER_BLOCK = 256;

struct DeviceState {
    float*       in       = nullptr;
    float*       out      = nullptr;
    std::size_t  capacity = 0;
    cudaStream_t stream   = nullptr;
};

DeviceState g_state;

bool ensure_device_buffers(std::size_t n) {
    if (n <= g_state.capacity) return true;

    if (g_state.in)  { cudaFree(g_state.in);  g_state.in  = nullptr; }
    if (g_state.out) { cudaFree(g_state.out); g_state.out = nullptr; }

    if (cudaMalloc(&g_state.in, n * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&g_state.out, n * sizeof(float)) != cudaSuccess) {
        cudaFree(g_state.in);
        g_state.in = nullptr;
        return false;
    }
    g_state.capacity = n;
    return true;
}

bool ensure_stream() {
    if (!g_state.stream && cudaStreamCreate(&g_state.stream) != cudaSuccess)
        return false;
    return true;
}
} // namespace

__global__ void gelu_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            const std::size_t n) {
    const std::size_t idx = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float x = in[idx];
    const float t = x * (A_LOG2E + B_LOG2E * x * x);
    out[idx] = x / (1.0f + exp2f(-t));
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const std::size_t n = input.size();
    if (n == 0) return {};

    if (!ensure_device_buffers(n) || !ensure_stream())
        return {};

    // Async H2D
    cudaMemcpyAsync(g_state.in,
                    input.data(),
                    n * sizeof(float),
                    cudaMemcpyHostToDevice,
                    g_state.stream);

    // Host-allocation
    std::vector<float> output(n);

    // Start kernel
    const std::size_t blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gelu_kernel<<<static_cast<unsigned>(blocks),
                  THREADS_PER_BLOCK, 0, g_state.stream>>>(
        g_state.in, g_state.out, n);

    // Async D2H
    cudaMemcpyAsync(output.data(),
                    g_state.out,
                    n * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    g_state.stream);

    // Synchronization
    cudaStreamSynchronize(g_state.stream);

    return output;
}
