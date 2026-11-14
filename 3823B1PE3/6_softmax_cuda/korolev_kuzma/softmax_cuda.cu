#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void Check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

// One allocation is reused by subsequent calls on this host thread.
class Workspace {
public:
    ~Workspace() { Release(); }

    void Reserve(std::size_t count) {
        int current_device = 0;
        Check(cudaGetDevice(&current_device), "cudaGetDevice");
        if (device_ == current_device && capacity_ >= count) return;
        if (count > std::numeric_limits<std::size_t>::max() / (2 * sizeof(float))) {
            throw std::length_error("SoftmaxCUDA input is too large");
        }
        Release();
        device_ = current_device;
        Check(cudaMalloc(reinterpret_cast<void**>(&data_), 2 * count * sizeof(float)),
              "cudaMalloc");
        capacity_ = count;
    }

    float* Input() const { return data_; }
    float* Output() const { return data_ + capacity_; }

private:
    void Release() noexcept {
        if (!data_) return;
        int previous = device_;
        if (cudaGetDevice(&previous) == cudaSuccess) {
            if (previous != device_) cudaSetDevice(device_);
            cudaFree(data_);
            if (previous != device_) cudaSetDevice(previous);
        }
        data_ = nullptr;
        capacity_ = 0;
    }

    float* data_ = nullptr;
    std::size_t capacity_ = 0;
    int device_ = -1;
};

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ float WarpMax(float value) {
    for (int offset = 16; offset; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ float WarpSum(float value) {
    for (int offset = 16; offset; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void SoftmaxRows(const float* __restrict__ input,
                            float* __restrict__ output,
                            std::size_t columns, int rows) {
    __shared__ float partial[kWarps];
    __shared__ float row_max;
    __shared__ float inverse_sum;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    for (std::size_t row = blockIdx.x; row < static_cast<std::size_t>(rows);
         row += gridDim.x) {
        const std::size_t base = row * columns;
        float maximum = -CUDART_INF_F;
        for (std::size_t col = threadIdx.x; col < columns; col += blockDim.x) {
            maximum = fmaxf(maximum, input[base + col]);
        }
        maximum = WarpMax(maximum);
        if (lane == 0) partial[warp] = maximum;
        __syncthreads();
        if (warp == 0) {
            maximum = WarpMax(lane < kWarps ? partial[lane] : -CUDART_INF_F);
            if (lane == 0) row_max = maximum;
        }
        __syncthreads();

        // Subtracting the row maximum prevents exponential overflow.
        float sum = 0.0f;
        for (std::size_t col = threadIdx.x; col < columns; col += blockDim.x) {
            sum += expf(input[base + col] - row_max);
        }
        sum = WarpSum(sum);
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = WarpSum(lane < kWarps ? partial[lane] : 0.0f);
            if (lane == 0) inverse_sum = 1.0f / sum;
        }
        __syncthreads();
        for (std::size_t col = threadIdx.x; col < columns; col += blockDim.x) {
            output[base + col] = expf(input[base + col] - row_max) * inverse_sum;
        }
        // Protect shared row statistics before this block processes another row.
        __syncthreads();
    }
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.size() % static_cast<std::size_t>(row_count) != 0) {
        throw std::invalid_argument("SoftmaxCUDA requires a positive row_count dividing input.size()");
    }
    if (input.empty()) return {};

    thread_local Workspace workspace;
    workspace.Reserve(input.size());
    const std::size_t bytes = input.size() * sizeof(float);
    Check(cudaMemcpy(workspace.Input(), input.data(), bytes, cudaMemcpyHostToDevice),
          "cudaMemcpy input");
    SoftmaxRows<<<std::min(row_count, 65535), kThreads>>>(
        workspace.Input(), workspace.Output(),
        input.size() / static_cast<std::size_t>(row_count), row_count);
    Check(cudaGetLastError(), "SoftmaxRows launch");

    // Host allocation can run while the kernel executes.
    std::vector<float> output(input.size());
    Check(cudaMemcpy(output.data(), workspace.Output(), bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy output");
    return output;
}
