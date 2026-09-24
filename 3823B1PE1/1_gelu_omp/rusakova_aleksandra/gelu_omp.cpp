#include "gelu_omp.h"

#include <cmath>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O3,unroll-loops,fast-math,tree-vectorize")
#endif

namespace {

// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// tanh(z) = 1 - 2 / (exp(2z) + 1)  =>  0.5 * (1 + tanh(z)) = 1 / (1 + exp(-2z))
// so GELU(x) = x / (1 + exp(-2 * sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
constexpr float kTwoSqrt2OverPi = 1.5957691216057308f;  // 2 * sqrt(2 / pi)
constexpr float kGeluCoeff = 0.044715f;

}  // namespace

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<float> output(static_cast<std::size_t>(n));

    const float* __restrict__ in = input.data();
    float* __restrict__ out = output.data();

    const int n8 = n & ~7;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n8; i += 8) {
#pragma omp simd
        for (int j = 0; j < 8; ++j) {
            const float x = in[i + j];
            const float x2 = x * x;
            out[i + j] = x / (1.0f + expf(-kTwoSqrt2OverPi * x * (1.0f + kGeluCoeff * x2)));
        }
    }

    for (int i = n8; i < n; ++i) {
        const float x = in[i];
        const float x2 = x * x;
        out[i] = x / (1.0f + expf(-kTwoSqrt2OverPi * x * (1.0f + kGeluCoeff * x2)));
    }

    return output;
}
