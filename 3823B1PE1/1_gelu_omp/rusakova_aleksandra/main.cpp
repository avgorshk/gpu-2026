#include "gelu_omp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

float GeluRef(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + kCoeff * x * x * x)));
}

}  // namespace

int main() {
    constexpr int kCheckSize = 1024;
    std::vector<float> check(kCheckSize);
    for (int i = 0; i < kCheckSize; ++i) {
        check[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(kCheckSize - 1);
    }

    const std::vector<float> check_out = GeluOMP(check);
    float max_abs_err = 0.0f;
    for (int i = 0; i < kCheckSize; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(check_out[i] - GeluRef(check[i])));
    }
    std::cout << "max abs error vs tanh: " << max_abs_err << '\n';

    constexpr int size = 134217728;
    std::vector<float> input(size);
    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(i % 1000) * 0.01f - 5.0f;
    }

    // Warming-up
    GeluOMP(input);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> result = GeluOMP(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
        std::cout << "run " << i << ": " << duration.count() << " s, checksum=" << result[0] << '\n';
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    std::cout << "min time: " << time << " s\n";
    return max_abs_err > 1e-5f ? 1 : 0;
}
