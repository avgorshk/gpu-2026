#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace
{
  constexpr float kGeluScale = 1.5957691216057308f;
  constexpr float kGeluCoef = 0.044715f;

  __global__ void GeluKernel(const float *__restrict__ in,
                             float *__restrict__ out,
                             std::size_t size)
  {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size)
      return;

    const float x = in[i];
    const float arg = kGeluScale * (x + kGeluCoef * x * x * x);
    out[i] = x / (1.0f + __expf(-arg));
  }

  float *g_d_in = nullptr;
  float *g_d_out = nullptr;
  size_t g_cap = 0;

  void ensureBuffers(size_t bytes)
  {
    if (bytes <= g_cap)
      return;
    if (g_d_in)
      cudaFree(g_d_in);
    if (g_d_out)
      cudaFree(g_d_out);
    cudaMalloc(&g_d_in, bytes);
    cudaMalloc(&g_d_out, bytes);
    g_cap = bytes;
  }

} // namespace

std::vector<float> GeluCUDA(const std::vector<float> &input)
{
  const std::size_t size = input.size();
  std::vector<float> output(size);
  if (size == 0)
    return output;

  const std::size_t bytes = size * sizeof(float);
  ensureBuffers(bytes);

  cudaMemcpyAsync(g_d_in, input.data(), bytes,
                  cudaMemcpyHostToDevice, 0);

  constexpr int blockSize = 256;
  const int gridSize = static_cast<int>((size + blockSize - 1) / blockSize);

  GeluKernel<<<gridSize, blockSize>>>(g_d_in, g_d_out, size);

  cudaMemcpyAsync(output.data(), g_d_out, bytes,
                  cudaMemcpyDeviceToHost, 0);

  cudaDeviceSynchronize();

  return output;
}
