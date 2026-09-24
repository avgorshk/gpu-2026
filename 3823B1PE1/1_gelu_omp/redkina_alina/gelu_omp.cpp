#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoef = 0.044715f;

inline float FastTanh(float z) {
    return 1.0f - 2.0f / (std::exp(2.0f * z) + 1.0f);
}

}  // namespace

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);

    const float* __restrict in = input.data();
    float* __restrict out = output.data();

#pragma omp parallel for simd schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const float x = in[i];
        out[i] = 0.5f * x * (1.0f + FastTanh(kSqrt2OverPi * (x + kGeluCoef * x * x * x)));
    }

    return output;
}
