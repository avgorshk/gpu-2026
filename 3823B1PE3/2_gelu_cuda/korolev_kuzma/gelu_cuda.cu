#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Workspace {
    float* data = nullptr;
    std::size_t capacity = 0;
    int device = -1;

    ~Workspace() { release(); }

    void release() noexcept {
        if (!data) return;
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess) return;
        if (current != device && cudaSetDevice(device) != cudaSuccess) return;
        cudaFree(data);
        if (current != device) cudaSetDevice(current);
        data = nullptr;
        capacity = 0;
    }

    void reserve(std::size_t count) {
        int current = 0;
        check(cudaGetDevice(&current), "cudaGetDevice");
        if (current != device) {
            release();
            device = current;
        }
        if (count > capacity) {
            release();
            check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)), "cudaMalloc");
            capacity = count;
        }
    }
};

__global__ void gelu(float* data, std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const float x = data[i];
        const float t = 1.5957691216057308f * x * (1.0f + 0.044715f * x * x);
        const float e = expf(-fabsf(t));
        data[i] = x * (x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
    }
}
}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) return {};
    // Separate workspaces allow concurrent calls from different host threads.
    static thread_local Workspace workspace;
    workspace.reserve(input.size());
    const std::size_t bytes = input.size() * sizeof(float);
    check(cudaMemcpy(workspace.data, input.data(), bytes, cudaMemcpyHostToDevice), "GELU input copy");
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((input.size() + threads - 1) / threads, 65535));
    gelu<<<blocks, threads>>>(workspace.data, input.size());
    check(cudaGetLastError(), "GELU kernel launch");
    // Host allocation can overlap the kernel; the blocking copy waits for it.
    std::vector<float> output(input.size());
    check(cudaMemcpy(output.data(), workspace.data, bytes, cudaMemcpyDeviceToHost), "GELU output copy");
    return output;
}
