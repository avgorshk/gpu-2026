#include "naive_gemm_cuda.h"
#include <future>
#define R 4
__global__ void MatrixMultCUDA(float *res, float *a, float *b, int n) {

  int j0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int i0 = (blockIdx.y * blockDim.y + threadIdx.y) * R;
  if (i0 >= n || j0 >= n)
    return;
  float4 sum[R];
#pragma unroll
  for (int t = 0; t < R; ++t)
    sum[t] = make_float4(0, 0, 0, 0);

  for (int k = 0; k < n; k += 4) {
    float4 b0 = *reinterpret_cast<const float4 *>(&b[k * n + j0]);
    float4 b1 = *reinterpret_cast<const float4 *>(&b[(k + 1) * n + j0]);
    float4 b2 = *reinterpret_cast<const float4 *>(&b[(k + 2) * n + j0]);
    float4 b3 = *reinterpret_cast<const float4 *>(&b[(k + 3) * n + j0]);

#pragma unroll
    for (int t = 0; t < R; ++t) {
      float4 av = *reinterpret_cast<const float4 *>(&a[(i0 + t) * n + k]);
      sum[t].x += av.x * b0.x + av.y * b1.x + av.z * b2.x + av.w * b3.x; //  j0
      sum[t].y += av.x * b0.y + av.y * b1.y + av.z * b2.y + av.w * b3.y; // j0+1
      sum[t].z += av.x * b0.z + av.y * b1.z + av.z * b2.z + av.w * b3.z; // j0+2
      sum[t].w += av.x * b0.w + av.y * b1.w + av.z * b2.w + av.w * b3.w; // j0+3
    }
  }
#pragma unroll
  for (int t = 0; t < R; ++t)
    *reinterpret_cast<float4 *>(&res[(i0 + t) * n + j0]) = sum[t];
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int n) {
  size_t s = n * n;

  size_t bs = s * sizeof(float);
  auto fut =
      std::async(std::launch::async, [s] { return std::vector<float>(s); });

  static size_t allocated = 0;
  static float *d_a = nullptr;
  static float *d_b = nullptr;
  static float *d_res = nullptr;
  if (bs > allocated) {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    cudaMalloc(&d_a, bs);
    cudaMalloc(&d_b, bs);
    cudaMalloc(&d_res, bs);
    allocated = bs;
  }
  cudaMemcpy(d_a, a.data(), bs, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bs, cudaMemcpyHostToDevice);

  int p1 = 32, p2 = 8;
  dim3 ps(p1, p2);
  dim3 chunks((n / 4 + p1 - 1) / p1, (n / R + p2 - 1) / p2);
  MatrixMultCUDA<<<chunks, ps>>>(d_res, d_a, d_b, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
  std::vector<float> res = fut.get();
  cudaMemcpy(res.data(), d_res, bs, cudaMemcpyDeviceToHost);

  return res;
}
