#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <omp.h>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoef = 0.044715f;

inline float FastTanh(float x)
{
    return 1.0f - 2.0f / (std::exp(2.0f * x) + 1.0f);
}

} // namespace

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    std::vector<float> output(input.size());

    #pragma omp parallel for simd schedule(static) default(none) \
        shared(input, output)
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        const float x = input[i];
        const float x2 = x * x;
        const float x3 = x2 * x;

        const float z = kSqrt2OverPi * (x + kGeluCoef * x3);

        output[i] = 0.5f * x * (1.0f + FastTanh(z));
    }

    return output;
}