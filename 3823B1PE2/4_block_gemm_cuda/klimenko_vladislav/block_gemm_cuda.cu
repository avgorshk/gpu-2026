#include "block_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>
#include <vector>

namespace
{

    constexpr int kTile = 32;
    constexpr int kBlockX = 32;
    constexpr int kBlockY = 8;
    constexpr int kRows = 4;

    __global__ void __launch_bounds__(kBlockX *kBlockY, 4)
        block_gemm_kernel(const float *__restrict__ a,
                          const float *__restrict__ b,
                          float *__restrict__ c,
                          int n)
    {
        __shared__ float as[kTile][kTile];
        __shared__ float bs[kTile][kTile];

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int col = blockIdx.x * kTile + tx;
        const int row = blockIdx.y * kTile + ty * kRows;

        float acc[kRows] = {0.f, 0.f, 0.f, 0.f};

        const int tiles = n / kTile;
        for (int t = 0; t < tiles; ++t)
        {
#pragma unroll
            for (int i = 0; i < kRows; ++i)
            {
                as[ty * kRows + i][tx] = a[static_cast<size_t>(row + i) * n + t * kTile + tx];
                bs[ty * kRows + i][tx] = b[static_cast<size_t>(t * kTile + ty * kRows + i) * n + col];
            }
            __syncthreads();

#pragma unroll
            for (int k = 0; k < kTile; ++k)
            {
                const float bv = bs[k][tx];
#pragma unroll
                for (int i = 0; i < kRows; ++i)
                {
                    acc[i] = fmaf(as[ty * kRows + i][k], bv, acc[i]);
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < kRows; ++i)
        {
            c[static_cast<size_t>(row + i) * n + col] = acc[i];
        }
    }

    __global__ void gemm_small(const float *__restrict__ a,
                               const float *__restrict__ b,
                               float *__restrict__ c,
                               int n)
    {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (row >= n || col >= n)
            return;
        float sum = 0.f;
        const float *a_row = a + static_cast<size_t>(row) * n;
        for (int k = 0; k < n; ++k)
        {
            sum = fmaf(a_row[k], b[static_cast<size_t>(k) * n + col], sum);
        }
        c[static_cast<size_t>(row) * n + col] = sum;
    }

    struct Workspace
    {
        float *h_a = nullptr;
        float *h_b = nullptr;
        float *h_c = nullptr;
        float *d_a = nullptr;
        float *d_b = nullptr;
        float *d_c = nullptr;
        size_t cap = 0;
        cudaStream_t stream = nullptr;

        void ensure(size_t bytes)
        {
            if (stream == nullptr)
                cudaStreamCreate(&stream);
            if (bytes <= cap)
                return;
            if (h_a)
            {
                cudaFreeHost(h_a);
                cudaFreeHost(h_b);
                cudaFreeHost(h_c);
            }
            if (d_a)
            {
                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_c);
            }
            cudaMallocHost(&h_a, bytes);
            cudaMallocHost(&h_b, bytes);
            cudaMallocHost(&h_c, bytes);
            cudaMalloc(&d_a, bytes);
            cudaMalloc(&d_b, bytes);
            cudaMalloc(&d_c, bytes);
            cap = bytes;
        }
    };

    Workspace &workspace()
    {
        static Workspace ws;
        return ws;
    }

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b,
                                 int n)
{
    if (n <= 0)
        return {};

    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);

    auto &ws = workspace();
    ws.ensure(bytes);

    std::memcpy(ws.h_a, a.data(), bytes);
    std::memcpy(ws.h_b, b.data(), bytes);

    cudaMemcpyAsync(ws.d_a, ws.h_a, bytes, cudaMemcpyHostToDevice, ws.stream);
    cudaMemcpyAsync(ws.d_b, ws.h_b, bytes, cudaMemcpyHostToDevice, ws.stream);

    if (n >= kTile && n % kTile == 0)
    {
        const dim3 block(kBlockX, kBlockY);
        const dim3 grid(n / kTile, n / kTile);
        block_gemm_kernel<<<grid, block, 0, ws.stream>>>(ws.d_a, ws.d_b, ws.d_c, n);
    }
    else
    {
        const dim3 block(16, 16);
        const dim3 grid((n + 15) / 16, (n + 15) / 16);
        gemm_small<<<grid, block, 0, ws.stream>>>(ws.d_a, ws.d_b, ws.d_c, n);
    }

    cudaMemcpyAsync(ws.h_c, ws.d_c, bytes, cudaMemcpyDeviceToHost, ws.stream);
    cudaStreamSynchronize(ws.stream);

    return std::vector<float>(ws.h_c, ws.h_c + elements);
}