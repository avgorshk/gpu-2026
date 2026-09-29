#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstring>
#include <vector>

namespace
{

    constexpr int kBlockX = 32;
    constexpr int kBlockY = 8;
    constexpr int kColsPerThread = 8;

    __global__ void gemm_vec8(const float *__restrict__ a,
                              const float *__restrict__ b,
                              float *__restrict__ c,
                              int n)
    {
        const int row = blockIdx.y * kBlockY + threadIdx.y;
        const int col =
            (blockIdx.x * kBlockX + threadIdx.x) * kColsPerThread;

        if (row >= n || col + kColsPerThread > n)
            return;

        float acc[8] = {0.f};

        const float *a_row =
            a + static_cast<size_t>(row) * n;

#pragma unroll 4
        for (int k = 0; k < n; ++k)
        {
            const float av = a_row[k];

            const float *b_ptr =
                b + static_cast<size_t>(k) * n + col;

            const float4 b0 =
                *reinterpret_cast<const float4 *>(b_ptr);

            const float4 b1 =
                *reinterpret_cast<const float4 *>(b_ptr + 4);

            acc[0] = fmaf(av, b0.x, acc[0]);
            acc[1] = fmaf(av, b0.y, acc[1]);
            acc[2] = fmaf(av, b0.z, acc[2]);
            acc[3] = fmaf(av, b0.w, acc[3]);
            acc[4] = fmaf(av, b1.x, acc[4]);
            acc[5] = fmaf(av, b1.y, acc[5]);
            acc[6] = fmaf(av, b1.z, acc[6]);
            acc[7] = fmaf(av, b1.w, acc[7]);
        }

        float *c_ptr =
            c + static_cast<size_t>(row) * n + col;

        *reinterpret_cast<float4 *>(c_ptr) =
            make_float4(acc[0], acc[1], acc[2], acc[3]);

        *reinterpret_cast<float4 *>(c_ptr + 4) =
            make_float4(acc[4], acc[5], acc[6], acc[7]);
    }

    __global__ void gemm_scalar(const float *__restrict__ a,
                                const float *__restrict__ b,
                                float *__restrict__ c,
                                int n)
    {
        const int col =
            blockIdx.x * blockDim.x + threadIdx.x;
        const int row =
            blockIdx.y * blockDim.y + threadIdx.y;

        if (row >= n || col >= n)
            return;

        float sum = 0.f;

        const float *a_row =
            a + static_cast<size_t>(row) * n;

#pragma unroll 4
        for (int k = 0; k < n; ++k)
        {
            sum = fmaf(
                a_row[k],
                b[static_cast<size_t>(k) * n + col],
                sum);
        }

        c[static_cast<size_t>(row) * n + col] = sum;
    }

} // namespace

std::vector<float> NaiveGemmCUDA(
    const std::vector<float> &a,
    const std::vector<float> &b,
    int n)
{
    if (n <= 0)
        return {};

    const size_t elements =
        static_cast<size_t>(n) * n;

    const size_t bytes =
        elements * sizeof(float);

    static float *h_a = nullptr;
    static float *h_b = nullptr;
    static float *h_c = nullptr;

    static float *d_a = nullptr;
    static float *d_b = nullptr;
    static float *d_c = nullptr;

    static size_t cap = 0;
    static cudaStream_t stream = nullptr;

    if (stream == nullptr)
        cudaStreamCreate(&stream);

    if (bytes > cap)
    {
        if (h_a)
        {
            cudaFreeHost(h_a);
            cudaFreeHost(h_b);
            cudaFreeHost(h_c);
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        cudaMallocHost(&h_a, bytes);
        cudaMallocHost(&h_b, bytes);
        cudaMallocHost(&h_c, bytes);

        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        cap = bytes;
    }

    std::memcpy(h_a, a.data(), bytes);
    std::memcpy(h_b, b.data(), bytes);

    cudaMemcpyAsync(
        d_a, h_a, bytes,
        cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(
        d_b, h_b, bytes,
        cudaMemcpyHostToDevice, stream);

    const dim3 block(kBlockX, kBlockY);

    if (n % kColsPerThread == 0 &&
        n >= kColsPerThread)
    {

        const dim3 grid(
            (n / kColsPerThread + kBlockX - 1) / kBlockX,
            (n + kBlockY - 1) / kBlockY);

        gemm_vec8<<<grid, block, 0, stream>>>(
            d_a, d_b, d_c, n);
    }
    else
    {

        const dim3 grid(
            (n + kBlockX - 1) / kBlockX,
            (n + kBlockY - 1) / kBlockY);

        gemm_scalar<<<grid, block, 0, stream>>>(
            d_a, d_b, d_c, n);
    }

    cudaMemcpyAsync(
        h_c, d_c, bytes,
        cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    return std::vector<float>(
        h_c, h_c + elements);
}