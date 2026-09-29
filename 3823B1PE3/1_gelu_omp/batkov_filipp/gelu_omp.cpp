#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    constexpr float SqrtTwoOverPi = 0.7978845608028654f;
    constexpr float CubicCoefficient = 0.044715f;

    std::vector<float> result(input.size());

    const std::ptrdiff_t size =
        static_cast<std::ptrdiff_t>(input.size());

    #pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        const float x = input[i];
        const float xCubed = x * x * x;

        const float exponentArgument =
            SqrtTwoOverPi * (x + CubicCoefficient * xCubed);

        result[i] =
            x / (1.0f + std::exp(-2.0f * exponentArgument));
    }

    return result;
}
