#include "gelu_omp.h"

#include <cmath>

static float Gelu(float value) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float value_cubed_coefficient = 0.044715f;

    const float value_cubed = value * value * value;
    const float tanh_argument = sqrt_2_over_pi * (value + value_cubed_coefficient * value_cubed);
    
    return value / (1.0f + std::exp(-2.0f * tanh_argument));
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());

#pragma omp parallel for simd schedule(static)
    for (std::size_t i = 0; i < input.size(); ++i) {
        result[i] = Gelu(input[i]);
    }
    return result;
}
