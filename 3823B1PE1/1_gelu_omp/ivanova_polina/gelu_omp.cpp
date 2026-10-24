#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    const size_t n_aligned = n & ~3ULL; // Кратно 4 для разворота

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_aligned; i += 4) {
        // Загрузка 4 элементов
        float x0 = input[i];
        float x1 = input[i + 1];
        float x2 = input[i + 2];
        float x3 = input[i + 3];

        float x0_3 = x0 * x0 * x0;
        float x1_3 = x1 * x1 * x1;
        float x2_3 = x2 * x2 * x2;
        float x3_3 = x3 * x3 * x3;

        float in0 = sqrt_2_over_pi * (x0 + coeff * x0_3);
        float in1 = sqrt_2_over_pi * (x1 + coeff * x1_3);
        float in2 = sqrt_2_over_pi * (x2 + coeff * x2_3);
        float in3 = sqrt_2_over_pi * (x3 + coeff * x3_3);

        float e0 = std::exp(2.0f * in0);
        float e1 = std::exp(2.0f * in1);
        float e2 = std::exp(2.0f * in2);
        float e3 = std::exp(2.0f * in3);

        float t0 = (e0 - 1.0f) / (e0 + 1.0f);
        float t1 = (e1 - 1.0f) / (e1 + 1.0f);
        float t2 = (e2 - 1.0f) / (e2 + 1.0f);
        float t3 = (e3 - 1.0f) / (e3 + 1.0f);

        output[i]     = 0.5f * x0 * (1.0f + t0);
        output[i + 1] = 0.5f * x1 * (1.0f + t1);
        output[i + 2] = 0.5f * x2 * (1.0f + t2);
        output[i + 3] = 0.5f * x3 * (1.0f + t3);
    }

    // Обработка оставшихся элементов (если n не делится на 4)
    #pragma omp parallel for schedule(static)
    for (size_t i = n_aligned; i < n; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        float e = std::exp(2.0f * inner);
        float t = (e - 1.0f) / (e + 1.0f);
        output[i] = 0.5f * x * (1.0f + t);
    }

    return output;
}