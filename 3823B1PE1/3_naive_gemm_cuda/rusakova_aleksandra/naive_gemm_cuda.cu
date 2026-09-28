#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kBlockY = 8;
constexpr int kBlockX = 32;
constexpr int kColsPerThread = 8;

__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    const int row = blockIdx.y * kBlockY + threadIdx.y;
    const int col = (blockIdx.x * kBlockX + threadIdx.x) * kColsPerThread;
    if (row >= n || col >= n) {
        return;
    }

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;
    const float* a_row = a + static_cast<size_t>(row) * n;

    for (int k = 0; k < n; ++k) {
        const float av = a_row[k];
        const float* b_ptr = b + static_cast<size_t>(k) * n + col;
        const float4 b0 = *reinterpret_cast<const float4*>(b_ptr);
        const float4 b1 = *reinterpret_cast<const float4*>(b_ptr + 4);
        acc0 = fmaf(av, b0.x, acc0);
        acc1 = fmaf(av, b0.y, acc1);
        acc2 = fmaf(av, b0.z, acc2);
        acc3 = fmaf(av, b0.w, acc3);
        acc4 = fmaf(av, b1.x, acc4);
        acc5 = fmaf(av, b1.y, acc5);
        acc6 = fmaf(av, b1.z, acc6);
        acc7 = fmaf(av, b1.w, acc7);
    }

    float* c_ptr = c + static_cast<size_t>(row) * n + col;
    *reinterpret_cast<float4*>(c_ptr) = make_float4(acc0, acc1, acc2, acc3);
    *reinterpret_cast<float4*>(c_ptr + 4) = make_float4(acc4, acc5, acc6, acc7);
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
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

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((n / kColsPerThread + kBlockX - 1) / kBlockX,
                    (n + kBlockY - 1) / kBlockY);
    naive_gemm_kernel<<<grid, block, 0, d.stream>>>(d.a, d.b, d.c, n);

    std::vector<float> output(static_cast<size_t>(n) * static_cast<size_t>(n));
    cudaMemcpyAsync(output.data(), d.c, bytes, cudaMemcpyDeviceToHost, d.stream);
    cudaStreamSynchronize(d.stream);
    return output;
}
