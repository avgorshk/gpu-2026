#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    constexpr float kTwoOverPi = 0.63662f;
    constexpr float kCoefficient = 0.044715f;

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
        const float x = input[i];
        const float x3 = x * x * x;

        const float z = kTwoOverPi * (x + kCoefficient * x3);

        const float exp_value = std::exp(-2.0f * z);
        const float tanh_value =
            (1.0f - exp_value) / (1.0f + exp_value);

        output[i] = 0.5f * x * (1.0f + tanh_value);
    }

    return output;
}
