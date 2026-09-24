#include "gelu_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoef = 0.044715f;
constexpr int kThreads = 256;

__device__ __forceinline__ float Gelu(float x) {
    const float z = kSqrt2OverPi * (x + kGeluCoef * x * x * x);
    const float t = 1.0f - 2.0f / (expf(2.0f * z) + 1.0f);
    return 0.5f * x * (1.0f + t);
}

__global__ void GeluKernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = Gelu(in[i]);
    }
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static int cap = 0;
    if (n > cap) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cap = n;
    }

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);
    if (n > 0) {
        GeluKernel<<<(n + kThreads - 1) / kThreads, kThreads>>>(d_in, d_out, n);
    }

    std::vector<float> output(static_cast<size_t>(n));
    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    return output;
}
