#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const auto count = static_cast<std::ptrdiff_t>(input.size());
    // (1 + tanh(t))/2 = sigmoid(2*t). The exponent is nonpositive,
    // avoiding overflow and cancellation in the negative tail.
    #pragma omp parallel for simd schedule(static) if(count >= 16384)
    for (std::ptrdiff_t i = 0; i < count; ++i) {
        const float x = input[i];
        const float t = 1.5957691216057308f * x * (1.0f + 0.044715f * x * x);
        const float e = std::exp(-std::abs(t));
        output[i] = x * (x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
    }
    return output;
}
