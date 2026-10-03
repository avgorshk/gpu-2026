#include "softmax_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxCols = 16384;

__device__ float WarpReduce(float v, bool is_max) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffff, v, offset);
        v = is_max ? fmaxf(v, other) : (v + other);
    }
    return v;
}

__device__ float BlockReduce(float v, bool is_max) {
    __shared__ float buf[32];
    v = WarpReduce(v, is_max);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        buf[wid] = v;
    }
    __syncthreads();

    const int nwarps = blockDim.x >> 5;
    float w = (lane < nwarps) ? buf[lane] : (is_max ? -INFINITY : 0.0f);
    if (wid == 0) {
        w = WarpReduce(w, is_max);
    }
    if (threadIdx.x == 0) {
        buf[0] = w;
    }
    __syncthreads();
    return buf[0];
}

__global__ void SoftmaxCached(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float row[];
    const float* src = in + static_cast<size_t>(blockIdx.x) * n;
    float* dst = out + static_cast<size_t>(blockIdx.x) * n;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row[i] = src[i];
    }
    __syncthreads();

    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        m = fmaxf(m, row[i]);
    }
    m = BlockReduce(m, true);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf(row[i] - m);
        row[i] = e;
        sum += e;
    }
    sum = BlockReduce(sum, false);
    const float inv = 1.0f / sum;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = row[i] * inv;
    }
}

__global__ void SoftmaxPlain(const float* __restrict__ in, float* __restrict__ out, int n) {
    const float* src = in + static_cast<size_t>(blockIdx.x) * n;
    float* dst = out + static_cast<size_t>(blockIdx.x) * n;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        m = fmaxf(m, src[i]);
    }
    m = BlockReduce(m, true);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += expf(src[i] - m);
    }
    sum = BlockReduce(sum, false);
    const float inv = 1.0f / sum;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = expf(src[i] - m) * inv;
    }
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.empty()) {
        return {};
    }

    const int row_size = static_cast<int>(input.size() / static_cast<size_t>(row_count));
    const size_t bytes = input.size() * sizeof(float);

    static float* d_in = nullptr;
    static float* d_out = nullptr;
    static size_t cap = 0;
    static bool tuned = false;
    if (!tuned) {
        cudaFuncSetAttribute(SoftmaxCached, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxCols * sizeof(float));
        tuned = true;
    }
    if (bytes > cap) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cap = bytes;
    }

    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);
    if (row_size <= kMaxCols) {
        SoftmaxCached<<<row_count, kThreads, row_size * sizeof(float)>>>(d_in, d_out, row_size);
    } else {
        SoftmaxPlain<<<row_count, kThreads>>>(d_in, d_out, row_size);
    }

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    return output;
}
