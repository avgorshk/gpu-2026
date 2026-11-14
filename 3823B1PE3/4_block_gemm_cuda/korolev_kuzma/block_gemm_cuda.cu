#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void CheckCUDA(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

std::size_t MatrixElements(const std::vector<float>& a,
                           const std::vector<float>& b, int n) {
    if (n < 0) {
        throw std::invalid_argument("matrix dimension must be nonnegative");
    }
    const auto side = static_cast<std::size_t>(n);
    if (side != 0 && side > std::numeric_limits<std::size_t>::max() / side) {
        throw std::length_error("matrix dimension overflows size_t");
    }
    const auto count = side * side;
    if (a.size() != count || b.size() != count) {
        throw std::invalid_argument("both matrices must contain n*n elements");
    }
    if (count > std::numeric_limits<std::size_t>::max() / (3 * sizeof(float))) {
        throw std::length_error("matrix allocation size overflows size_t");
    }
    return count;
}

// One workspace per host thread; allocation is reused across timed calls.
struct Workspace {
    int device = -1;
    cudaStream_t stream = nullptr;
    float* storage = nullptr;
    std::size_t capacity = 0;

    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    ~Workspace() { Release(); }

    void Release() noexcept {
        int active_device = -1;
        cudaGetDevice(&active_device);
        if (device >= 0 && active_device != device) {
            cudaSetDevice(device);
        }
        if (stream) {
            cudaStreamSynchronize(stream);
        }

        if (storage) {
            cudaFree(storage);
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
        if (active_device >= 0 && active_device != device) {
            cudaSetDevice(active_device);
        }
        device = -1;
        stream = nullptr;
        storage = nullptr;
        capacity = 0;
    }

    void Reserve(std::size_t count) {
        int active_device = -1;
        CheckCUDA(cudaGetDevice(&active_device), "cudaGetDevice");
        if (active_device != device) {
            Release();
            device = active_device;
        }
        if (!stream) {
            CheckCUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                      "cudaStreamCreateWithFlags");
        }

        if (count > capacity) {
            float* replacement = nullptr;
            CheckCUDA(cudaMalloc(reinterpret_cast<void**>(&replacement),
                                 3 * count * sizeof(float)), "cudaMalloc");
            if (storage) {
                const auto status = cudaFree(storage);
                if (status != cudaSuccess) {
                    cudaFree(replacement);
                    CheckCUDA(status, "cudaFree");
                }
            }
            storage = replacement;
            capacity = count;
        }
    }

    float* A() const { return storage; }
    float* B() const { return storage + capacity; }
    float* C() const { return storage + 2 * capacity; }
};

constexpr int kTile = 32;

// One CUDA block computes one C tile; every thread owns one C element.
__global__ void BlockGemmKernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c, int n) {
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile];
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * kTile + tx;
    const int row = static_cast<int>(blockIdx.y) * kTile + ty;
    float sum = 0.0f;
    for (int base = 0; base < n; base += kTile) {
        tile_a[ty][tx] = (row < n && base + tx < n)
            ? a[static_cast<std::size_t>(row) * n + base + tx] : 0.0f;
        tile_b[ty][tx] = (base + ty < n && col < n)
            ? b[static_cast<std::size_t>(base + ty) * n + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum = fmaf(tile_a[ty][k], tile_b[k][tx], sum);
        }
        // All readers finish before the next iteration overwrites the tiles.
        __syncthreads();
    }
    if (row < n && col < n) {
        c[static_cast<std::size_t>(row) * n + col] = sum;
    }
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b, int n) {
    const auto count = MatrixElements(a, b, n);
    if (count == 0) {
        return {};
    }
    thread_local Workspace workspace;
    workspace.Reserve(count);
    const auto bytes = count * sizeof(float);
    std::vector<float> result;
    try {
        CheckCUDA(cudaMemcpyAsync(workspace.A(), a.data(), bytes,
                                  cudaMemcpyHostToDevice, workspace.stream),
                  "copy matrix A to device");
        CheckCUDA(cudaMemcpyAsync(workspace.B(), b.data(), bytes,
                                  cudaMemcpyHostToDevice, workspace.stream),
                  "copy matrix B to device");
        const dim3 block(kTile, kTile);
        const auto tiles = (static_cast<unsigned int>(n) + kTile - 1) / kTile;
        BlockGemmKernel<<<dim3(tiles, tiles), block, 0, workspace.stream>>>(
            workspace.A(), workspace.B(), workspace.C(), n);
        CheckCUDA(cudaGetLastError(), "launch block GEMM");
        // Allocate/initialize host output while the kernel is executing.
        result.resize(count);
        CheckCUDA(cudaMemcpyAsync(result.data(), workspace.C(), bytes,
                                  cudaMemcpyDeviceToHost, workspace.stream),
                  "copy matrix C to host");
        CheckCUDA(cudaStreamSynchronize(workspace.stream), "complete GEMM");
        return result;
    } catch (...) {
        // Input/output host memory may not outlive outstanding transfers.
        cudaStreamSynchronize(workspace.stream);
        throw;
    }
}
