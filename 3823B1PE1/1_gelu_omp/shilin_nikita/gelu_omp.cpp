#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    if (n == 0) {
        return {};
    }

    std::vector<float> output(n);
    const float* __restrict__ in = input.data();
    float* __restrict__ out = output.data();

    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;

    const std::ptrdiff_t nn = static_cast<std::ptrdiff_t>(n);
    const std::ptrdiff_t n8 = nn & ~std::ptrdiff_t{7};

#pragma omp parallel
    {
#pragma omp for simd schedule(static) nowait
        for (std::ptrdiff_t i = 0; i < n8; ++i) {
            const float x = in[i];
            const float x2 = x * x;
            const float s = kSqrt2OverPi * (x + kCoeff * x2 * x);
            out[i] = x / (1.0f + expf(-2.0f * s));
        }
#pragma omp single
        {
            for (std::ptrdiff_t i = n8; i < nn; ++i) {
                const float x = in[i];
                const float x2 = x * x;
                const float s = kSqrt2OverPi * (x + kCoeff * x2 * x);
                out[i] = x / (1.0f + expf(-2.0f * s));
            }
        }
    }

    return output;
}
