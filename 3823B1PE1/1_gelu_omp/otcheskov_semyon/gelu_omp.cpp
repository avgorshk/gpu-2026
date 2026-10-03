#include <cstdio>
#include <cmath>
#include <vector>

#include <omp.h>
#include "gelu_omp.h"

namespace {
constexpr float LOG2E = 1.44269504089f;         // log2(e)
constexpr float A_LOG2E = 1.59576912f * LOG2E;  // 2 * sqrt(2/pi) * log2(e)
constexpr float B_LOG2E = 0.0713548f  * LOG2E;  // 2 * sqrt(2/pi) * 0.044715 * log2(e)
} // namespace

// exp(y)  = exp2(y * log2(e))
// GELU(x) = x / (1 + exp2(-x * (A_LOG2E + B_LOG2E * x^2)))
std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);

    if (n == 0) return output;

    const float* __restrict__ in = input.data();
    float* __restrict__  out = output.data();

    std::size_t i = 0;

    #pragma omp parallel for simd aligned(in, out: 16)
    for (i = 0; i < n; ++i) {
        float x = in[i];
        float t = x * (A_LOG2E + B_LOG2E * x * x);
        out[i] = x / (1.0f + exp2f(-t));
    }

    return output;
}