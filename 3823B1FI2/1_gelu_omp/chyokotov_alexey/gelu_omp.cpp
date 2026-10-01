#include "gelu_omp.h"
#include <cmath>
#include <algorithm>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    const long long sz = static_cast<long long>(input.size());

    if (sz == 0) return output;

    const float* in_data = input.data();
    float* out_data = output.data();

    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    constexpr float COEFF = 0.044715f;

    #pragma omp parallel for simd
    for (long long i = 0; i < sz; ++i) {
        float x = in_data[i];
        
        float arg = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        
        float exp_val = expf(-2.0f * arg);
        float tanh_approx = (1.0f - exp_val) / (1.0f + exp_val);
        
        out_data[i] = 0.5f * x * (1.0f + tanh_approx);
    }

    return output;
}