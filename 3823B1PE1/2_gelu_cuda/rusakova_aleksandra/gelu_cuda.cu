#include "gelu_cuda.h"

#include <cuda_runtime.h>

namespace {

// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// 0.5 * (1 + tanh(z)) = 1 / (1 + exp(-2z))
// GELU(x) = x / (1 + exp(-2 * sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
constexpr float kTwoSqrt2OverPi = 1.5957691216057308f;
constexpr float kGeluCoeff = 0.044715f;

__device__ __forceinline__ float gelu_exp(float x) {
    const float x2 = x * x;
    return x / (1.0f + __expf(-kTwoSqrt2OverPi * x * (1.0f + kGeluCoeff * x2)));
}

__global__ void gelu_kernel_vec4(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = in[i];
        v.x = gelu_exp(v.x);
        v.y = gelu_exp(v.y);
        v.z = gelu_exp(v.z);
        v.w = gelu_exp(v.w);
        out[i] = v;
    }
}

__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = gelu_exp(in[i]);
    }
}

struct DeviceWorkspace {
    float* in = nullptr;
    float* out = nullptr;
    size_t cap = 0;
    cudaStream_t stream = nullptr;

    void ensure(size_t n) {
        if (stream == nullptr) {
            cudaStreamCreate(&stream);
        }
        if (n <= cap) {
            return;
        }
        cudaFree(in);
        cudaFree(out);
        cudaMalloc(&in, n * sizeof(float));
        cudaMalloc(&out, n * sizeof(float));
        cap = n;
    }
};

DeviceWorkspace& workspace() {
    static DeviceWorkspace mem;
    return mem;
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }

    auto& d = workspace();
    d.ensure(static_cast<size_t>(n));

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    cudaMemcpyAsync(d.in, input.data(), bytes, cudaMemcpyHostToDevice, d.stream);

    constexpr int kBlock = 256;
    const int n4 = n >> 2;
    if (n4 > 0) {
        const int grid = (n4 + kBlock - 1) / kBlock;
        gelu_kernel_vec4<<<grid, kBlock, 0, d.stream>>>(
            reinterpret_cast<const float4*>(d.in),
            reinterpret_cast<float4*>(d.out),
            n4);
    }

    const int rem = n & 3;
    if (rem > 0) {
        const int offset = n4 << 2;
        gelu_kernel<<<1, rem, 0, d.stream>>>(d.in + offset, d.out + offset, rem);
    }

    // Host allocation overlaps with H2D + kernel on the stream.
    std::vector<float> output(static_cast<size_t>(n));

    cudaMemcpyAsync(output.data(), d.out, bytes, cudaMemcpyDeviceToHost, d.stream);
    cudaStreamSynchronize(d.stream);
    return output;
}
