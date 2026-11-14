#ifndef KOROLEV_KUZMA_GELU_OCL_H
#define KOROLEV_KUZMA_GELU_OCL_H

#include <vector>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform);

#endif
