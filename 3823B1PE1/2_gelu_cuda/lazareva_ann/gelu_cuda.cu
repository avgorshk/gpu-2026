#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                   
    do {                                                                   
        cudaError_t err__ = (call);                                       
        if (err__ != cudaSuccess) {                                        
            throw std::runtime_error(std::string("CUDA error: ") +         
                                      cudaGetErrorString(err__));          
        }                                                                  
    } while (0)

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kCoeff       = 0.044715f;
constexpr int   kBlockSize   = 256;

__device__ __forceinline__ float GeluScalar(float x) {
    float z = kSqrt2OverPi * (x + kCoeff * x * x * x);
    float e = __expf(-2.0f * z);   // быстрый intrinsic exp
    return x / (1.0f + e);
}

__global__ void GeluKernelVec4(const float4* __restrict__ in,
                                float4* __restrict__ out,
                                size_t n4) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 x = in[idx];
        float4 y;
        y.x = GeluScalar(x.x);
        y.y = GeluScalar(x.y);
        y.z = GeluScalar(x.z);
        y.w = GeluScalar(x.w);
        out[idx] = y;
    }
}

__global__ void GeluKernelTail(const float* __restrict__ in,
                                float* __restrict__ out,
                                size_t offset,
                                size_t n) {
    size_t idx = offset + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = GeluScalar(in[idx]);
    }
}

struct CudaResources {
    float*        d_in      = nullptr;
    float*        d_out     = nullptr;
    size_t        capacity  = 0;
    cudaStream_t  stream    = nullptr;
    bool          ready     = false;

    void EnsureCapacity(size_t n) {
        if (!ready) {
            CUDA_CHECK(cudaStreamCreate(&stream));
            ready = true;
        }
        if (n > capacity) {
            if (d_in)  CUDA_CHECK(cudaFree(d_in));
            if (d_out) CUDA_CHECK(cudaFree(d_out));
            CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
            capacity = n;
        }
    }

    ~CudaResources() {
        if (d_in)  cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        if (ready) cudaStreamDestroy(stream);
    }
};

CudaResources g_res;

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    g_res.EnsureCapacity(n);
    cudaStream_t stream = g_res.stream;

    float* h_in  = const_cast<float*>(input.data());
    float* h_out = output.data();

    cudaError_t regInStatus  = cudaHostRegister(h_in,  n * sizeof(float),
                                                 cudaHostRegisterDefault);
    cudaError_t regOutStatus = cudaHostRegister(h_out, n * sizeof(float),
                                                 cudaHostRegisterDefault);

    CUDA_CHECK(cudaMemcpyAsync(g_res.d_in, h_in, n * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    const size_t n4         = n / 4;
    const size_t tail_start = n4 * 4;

    if (n4 > 0) {
        int grid = static_cast<int>((n4 + kBlockSize - 1) / kBlockSize);
        GeluKernelVec4<<<grid, kBlockSize, 0, stream>>>(
            reinterpret_cast<const float4*>(g_res.d_in),
            reinterpret_cast<float4*>(g_res.d_out),
            n4);
    }

    if (tail_start < n) {
        size_t tail_n = n - tail_start;
        int grid = static_cast<int>((tail_n + kBlockSize - 1) / kBlockSize);
        GeluKernelTail<<<grid, kBlockSize, 0, stream>>>(
            g_res.d_in, g_res.d_out, tail_start, n);
    }

    CUDA_CHECK(cudaMemcpyAsync(h_out, g_res.d_out, n * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (regInStatus  == cudaSuccess) cudaHostUnregister(h_in);
    if (regOutStatus == cudaSuccess) cudaHostUnregister(h_out);

    return output;
}