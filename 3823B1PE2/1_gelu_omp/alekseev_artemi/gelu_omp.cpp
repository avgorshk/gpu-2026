#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float> &input) {
  const size_t n = input.size();
  std::vector<float> output(n);

  const float sqrt_2_over_pi = std::sqrt(2.0f / static_cast<float>(M_PI));
  const float coef = 0.044715f;

#pragma omp parallel for
  for (long long i = 0; i < static_cast<long long>(n); ++i) {
    const float x = input[i];
    const float z = sqrt_2_over_pi * (x + coef * x * x * x);
    const float e = std::exp(2.0f * z);
    output[i] = 0.5f * x * (1.0f + (e - 1.0f) / (e + 1.0f));
  }

  return output;
}
