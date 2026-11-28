#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace {

    constexpr int TILE_SIZE = 16;
    __global__ void BlockGemmKernel(const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ c,
        int n) {
        __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
        __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int row = blockIdx.y * TILE_SIZE + ty;
        const int col = blockIdx.x * TILE_SIZE + tx;

        float sum = 0.0f;

        for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
            const int a_col = tile_start + tx;
            const int b_row = tile_start + ty;

            tile_a[ty][tx] = (row < n && a_col < n)
                ? a[static_cast<std::size_t>(row) * n + a_col]
                : 0.0f;

            tile_b[ty][tx] = (b_row < n && col < n)
                ? b[static_cast<std::size_t>(b_row) * n + col]
                : 0.0f;

            __syncthreads();

#pragma unroll 4
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum = fmaf(tile_a[ty][k], tile_b[k][tx], sum);
            }
            __syncthreads();
        }

        if (row < n && col < n) {
            c[static_cast<std::size_t>(row) * n + col] = sum;
        }
    }

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t dim = static_cast<std::size_t>(n);
    const std::size_t elements = dim * dim;
    const std::size_t bytes = elements * sizeof(float);
    std::vector<float> result(elements, 0.0f);
    float* device_memory = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device_memory), 3 * bytes);

    float* device_a = device_memory;
    float* device_b = device_a + elements;
    float* device_c = device_b + elements;
    cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice);
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
        (n + TILE_SIZE - 1) / TILE_SIZE);
    BlockGemmKernel << <grid, block >> > (device_a, device_b, device_c, n);
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(device_memory);
        return {};
    }
    cudaMemcpy(result.data(), device_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(device_memory);
    return result;
}