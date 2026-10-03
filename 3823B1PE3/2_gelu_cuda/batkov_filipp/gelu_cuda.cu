#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

constexpr int ThreadsPerBlock = 256;

constexpr float SqrtTwoOverPi = 0.7978845608028654f;
constexpr float CubicCoefficient = 0.044715f;

/**
 * Проверяет результат выполнения CUDA-функции.
 *
 * @param error Код ошибки CUDA.
 * @param operation Название выполняемой операции.
 *
 * @throws std::runtime_error Если CUDA вернула ошибку.
 */
void CheckCudaError(
    const cudaError_t error,
    const char* operation
) {
    if (error == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string(operation) +
        ": " +
        cudaGetErrorString(error)
    );
}

/**
 * Вычисляет GELU для элементов массива на GPU.
 *
 * @param input Входной массив в памяти GPU.
 * @param output Выходной массив в памяти GPU.
 * @param size Количество элементов.
 */
__global__ void GeluKernel(
    const float* input,
    float* output,
    const std::size_t size
) {
    const std::size_t index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    const float x = input[index];

    const float xCubed = x * x * x;

    const float exponentArgument =
        SqrtTwoOverPi *
        (x + CubicCoefficient * xCubed);

    output[index] =
        x /
        (1.0f + expf(-2.0f * exponentArgument));
}

} // namespace

std::vector<float> GeluCUDA(
    const std::vector<float>& input
) {
    std::vector<float> result(input.size());

    if (input.empty()) {
        return result;
    }

    const std::size_t dataSize =
        input.size() * sizeof(float);

    float* deviceInput = nullptr;
    float* deviceOutput = nullptr;

    CheckCudaError(
        cudaMalloc(
            reinterpret_cast<void**>(&deviceInput),
            dataSize
        ),
        "Failed to allocate device input memory"
    );

    try {
        CheckCudaError(
            cudaMalloc(
                reinterpret_cast<void**>(&deviceOutput),
                dataSize
            ),
            "Failed to allocate device output memory"
        );

        CheckCudaError(
            cudaMemcpy(
                deviceInput,
                input.data(),
                dataSize,
                cudaMemcpyHostToDevice
            ),
            "Failed to copy input to GPU"
        );

        const int blocksCount =
            static_cast<int>(
                (input.size() + ThreadsPerBlock - 1) /
                ThreadsPerBlock
            );

        GeluKernel<<<blocksCount, ThreadsPerBlock>>>(
            deviceInput,
            deviceOutput,
            input.size()
        );

        CheckCudaError(
            cudaGetLastError(),
            "Failed to launch GELU kernel"
        );

        CheckCudaError(
            cudaMemcpy(
                result.data(),
                deviceOutput,
                dataSize,
                cudaMemcpyDeviceToHost
            ),
            "Failed to copy result from GPU"
        );
    }
    catch (...) {
        cudaFree(deviceOutput);
        cudaFree(deviceInput);

        throw;
    }

    CheckCudaError(
        cudaFree(deviceOutput),
        "Failed to free device output memory"
    );

    CheckCudaError(
        cudaFree(deviceInput),
        "Failed to free device input memory"
    );

    return result;
}