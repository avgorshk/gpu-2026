#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 32;

__global__ void block_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    __shared__ float as[kTile][kTile + 1];
    __shared__ float bs[kTile][kTile + 1];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float sum = 0.f;

    const int tiles = n / kTile;
    for (int t = 0; t < tiles; ++t) {
        as[threadIdx.y][threadIdx.x] =
            a[static_cast<size_t>(row) * n + t * kTile + threadIdx.x];
        bs[threadIdx.y][threadIdx.x] =
            b[static_cast<size_t>(t * kTile + threadIdx.y) * n + col];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum = fmaf(as[threadIdx.y][k], bs[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    c[static_cast<size_t>(row) * n + col] = sum;
}

struct DeviceWorkspace {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t cap = 0;
    cudaStream_t stream = nullptr;

    void ensure(size_t n) {
        if (stream == nullptr) {
            cudaStreamCreate(&stream);
        }
        const size_t need = n * n;
        if (need <= cap) {
            return;
        }
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cudaMalloc(&a, need * sizeof(float));
        cudaMalloc(&b, need * sizeof(float));
        cudaMalloc(&c, need * sizeof(float));
        cap = need;
    }
};

DeviceWorkspace& workspace() {
    static DeviceWorkspace mem;
    return mem;
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    auto& d = workspace();
    d.ensure(static_cast<size_t>(n));

    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    cudaMemcpyAsync(d.a, a.data(), bytes, cudaMemcpyHostToDevice, d.stream);
    cudaMemcpyAsync(d.b, b.data(), bytes, cudaMemcpyHostToDevice, d.stream);

    const dim3 block(kTile, kTile);
    const dim3 grid(n / kTile, n / kTile);
    block_gemm_kernel<<<grid, block, 0, d.stream>>>(d.a, d.b, d.c, n);

    std::vector<float> output(static_cast<size_t>(n) * static_cast<size_t>(n));
    cudaMemcpyAsync(output.data(), d.c, bytes, cudaMemcpyDeviceToHost, d.stream);
    cudaStreamSynchronize(d.stream);
    return output;
}
