#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstddef>
#include <cassert>

#include <cuda_runtime.h>
#include "naive_gemm_cuda.h"


#define CHECK_ERROR(X)                                              \
    do {                                                            \
        cudaError_t err_ = (X);                                     \
        if (err_ != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: '%s' -> %s\n",    \
                    __FILE__, __LINE__, #X,                         \
                    cudaGetErrorString(err_));                      \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

constexpr int kTile = 4;
constexpr int kBlockDimX = kBlockX * kTile; // 64 columns per block
constexpr int kBlockDimY = kBlockY * kTile; // 64 rows per block

__device__ __forceinline__ const float4& vec4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

__device__ __forceinline__ float4& vec4(float* p) {
    return *reinterpret_cast<float4*>(p);
}

__device__ __forceinline__ void fma_row(float4& acc, const float4& a,
                                        const float4& b0, const float4& b1,
                                        const float4& b2, const float4& b3) {
    acc.x += a.x * b0.x + a.y * b1.x + a.z * b2.x + a.w * b3.x;
    acc.y += a.x * b0.y + a.y * b1.y + a.z * b2.y + a.w * b3.y;
    acc.z += a.x * b0.z + a.y * b1.z + a.z * b2.z + a.w * b3.z;
    acc.w += a.x * b0.w + a.y * b1.w + a.z * b2.w + a.w * b3.w;
}

__global__ void naive_gemm_reg4x4_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         int n) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int row_start = by * kBlockDimY + ty * kTile;
    const int col_start = bx * kBlockDimX + tx * kTile;

    if (row_start >= n || col_start >= n) return;

    float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 acc1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 acc2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 acc3 = make_float4(0.f, 0.f, 0.f, 0.f);

    const float* a_row0 = a + static_cast<std::size_t>(row_start + 0) * n;
    const float* a_row1 = a + static_cast<std::size_t>(row_start + 1) * n;
    const float* a_row2 = a + static_cast<std::size_t>(row_start + 2) * n;
    const float* a_row3 = a + static_cast<std::size_t>(row_start + 3) * n;

    for (int k = 0; k < n; k += kTile) {
        const float4 a0 = vec4(a_row0 + k);
        const float4 a1 = vec4(a_row1 + k);
        const float4 a2 = vec4(a_row2 + k);
        const float4 a3 = vec4(a_row3 + k);

        const float4 b0 = vec4(b + static_cast<std::size_t>(k + 0) * n + col_start);
        const float4 b1 = vec4(b + static_cast<std::size_t>(k + 1) * n + col_start);
        const float4 b2 = vec4(b + static_cast<std::size_t>(k + 2) * n + col_start);
        const float4 b3 = vec4(b + static_cast<std::size_t>(k + 3) * n + col_start);

        fma_row(acc0, a0, b0, b1, b2, b3);
        fma_row(acc1, a1, b0, b1, b2, b3);
        fma_row(acc2, a2, b0, b1, b2, b3);
        fma_row(acc3, a3, b0, b1, b2, b3);
    }

    float* c_row0 = c + static_cast<std::size_t>(row_start + 0) * n + col_start;
    float* c_row1 = c + static_cast<std::size_t>(row_start + 1) * n + col_start;
    float* c_row2 = c + static_cast<std::size_t>(row_start + 2) * n + col_start;
    float* c_row3 = c + static_cast<std::size_t>(row_start + 3) * n + col_start;

    vec4(c_row0) = acc0;
    vec4(c_row1) = acc1;
    vec4(c_row2) = acc2;
    vec4(c_row3) = acc3;
}

// for n < 4
__global__ void naive_gemm_scalar_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    assert((n % kTile == 0 || n < 4) && "n must be a multiple of 4 or less than 4");

    const std::size_t elem_count = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elem_count * sizeof(float);

    std::vector<float> c(elem_count);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CHECK_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_ERROR(cudaMalloc(&d_c, bytes));

    CHECK_ERROR(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    if (n >= 4) {
        const dim3 block(kBlockX, kBlockY);
        const dim3 grid((n + kBlockDimX - 1) / kBlockDimX,
                        (n + kBlockDimY - 1) / kBlockDimY);
        naive_gemm_reg4x4_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    } else {
        const int block_size = 16;
        const dim3 block(block_size, block_size);
        const dim3 grid((n + block_size - 1) / block_size,
                        (n + block_size - 1) / block_size);
        naive_gemm_scalar_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    
    CHECK_ERROR(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_a));
    CHECK_ERROR(cudaFree(d_b));
    CHECK_ERROR(cudaFree(d_c));

    return c;
}