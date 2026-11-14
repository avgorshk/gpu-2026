#ifndef GPU2026_BLOCK_GEMM_CUDA_H
#define GPU2026_BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                               const std::vector<float>& b, int n);

#endif
