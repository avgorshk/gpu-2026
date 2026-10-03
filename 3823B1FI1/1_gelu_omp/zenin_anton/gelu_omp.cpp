#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<float> output(n);

    const float c1 = 1.5957691216057308f;
    const float c2 = 0.0713548162726009f;

    const float* in = input.data();
    float* out = output.data();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const float x = in[i];
        const float u = x * (c1 + c2 * x * x);
        out[i] = x / (1.0f + std::exp(-u));
    }

    return output;
    
}