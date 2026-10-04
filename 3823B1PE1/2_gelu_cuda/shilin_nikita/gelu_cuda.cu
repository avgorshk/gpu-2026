#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kCoeff = 0.044715f;
constexpr int kThreads = 256;

__device__ __forceinline__ float GeluExp(float x) {
    const float x2 = x * x;
    const float s = kSqrt2OverPi * (x + kCoeff * x2 * x);
    return x / (1.0f + __expf(-2.0f * s));
}

__global__ void GeluKernel(const float* __restrict__ in, float* __restrict__ out, int n4) {
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = __ldg(in4 + i);
        float4 r;
        r.x = GeluExp(v.x);
        r.y = GeluExp(v.y);
        r.z = GeluExp(v.z);
        r.w = GeluExp(v.w);
        out4[i] = r;
    }
}

__global__ void GeluTailKernel(const float* __restrict__ in, float* __restrict__ out, int n,
                               int offset) {
    const int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = GeluExp(in[i]);
    }
}

struct DeviceCache {
    float* d_in = nullptr;
    float* d_out = nullptr;
    float* h_pin_out = nullptr;
    const float* registered_in = nullptr;
    std::size_t cap = 0;
    std::size_t pin_cap = 0;
    cudaStream_t stream = nullptr;
};

DeviceCache& Cache() {
    static DeviceCache cache;
    return cache;
}

void Ensure(std::size_t n) {
    DeviceCache& cache = Cache();
    if (cache.stream == nullptr) {
        cudaStreamCreateWithFlags(&cache.stream, cudaStreamNonBlocking);
    }
    if (cache.cap < n) {
        if (cache.d_in != nullptr) {
            cudaFree(cache.d_in);
            cudaFree(cache.d_out);
        }
        const std::size_t bytes = n * sizeof(float);
        cudaMalloc(&cache.d_in, bytes);
        cudaMalloc(&cache.d_out, bytes);
        cache.cap = n;
    }
    if (cache.pin_cap < n) {
        if (cache.h_pin_out != nullptr) {
            cudaFreeHost(cache.h_pin_out);
        }
        cudaMallocHost(&cache.h_pin_out, n * sizeof(float));
        cache.pin_cap = n;
    }
}

void RegisterInput(const float* ptr, std::size_t bytes) {
    DeviceCache& cache = Cache();
    if (cache.registered_in == ptr) {
        return;
    }
    if (cache.registered_in != nullptr) {
        cudaHostUnregister(const_cast<float*>(cache.registered_in));
        cache.registered_in = nullptr;
    }
    if (cudaHostRegister(const_cast<float*>(ptr), bytes, cudaHostRegisterDefault) == cudaSuccess) {
        cache.registered_in = ptr;
    }
}

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }

    Ensure(static_cast<std::size_t>(n));
    DeviceCache& cache = Cache();
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    RegisterInput(input.data(), bytes);

    cudaMemcpyAsync(cache.d_in, input.data(), bytes, cudaMemcpyHostToDevice, cache.stream);

    const int n4 = n >> 2;
    if (n4 > 0) {
        int blocks = 1024;
        if (blocks > n4) {
            blocks = (n4 + kThreads - 1) / kThreads;
            if (blocks < 1) {
                blocks = 1;
            }
        }
        GeluKernel<<<blocks, kThreads, 0, cache.stream>>>(cache.d_in, cache.d_out, n4);
    }

    std::vector<float> output(static_cast<std::size_t>(n));

    if ((n & 3) != 0) {
        const int offset = n4 << 2;
        GeluTailKernel<<<1, 4, 0, cache.stream>>>(cache.d_in, cache.d_out, n, offset);
    }

    cudaMemcpyAsync(cache.h_pin_out, cache.d_out, bytes, cudaMemcpyDeviceToHost, cache.stream);
    cudaStreamSynchronize(cache.stream);
    std::memcpy(output.data(), cache.h_pin_out, bytes);
    return output;
}
