#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kLocalSize = 256;
constexpr std::size_t kVecWidth = 4;

const char kKernelSource[] = R"CLC(
__kernel void gelu(__global float* data, const int n) {
    const int lane = get_global_id(0);
    const int base = lane * 4;
    const float kScale = 1.5957691216057308f;
    const float kCubic = kScale * 0.044715f;
    if (base + 4 <= n) {
        const float4 x = vload4(lane, data);
        const float4 scaled = x * (kScale + kCubic * x * x);
        vstore4(x / (1.0f + exp(-scaled)), lane, data);
        return;
    }
    for (int i = base; i < n; ++i) {
        const float x = data[i];
        const float scaled = x * (kScale + kCubic * x * x);
        data[i] = x / (1.0f + exp(-scaled));
    }
}
)CLC";

void throw_if_cl_error(cl_int status) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error("OpenCL error");
    }
}

struct OpenClRuntime {
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;
    std::size_t capacity = 0;

    OpenClRuntime() = default;

    ~OpenClRuntime() {
        release();
    }

    OpenClRuntime(const OpenClRuntime&) = delete;
    OpenClRuntime& operator=(const OpenClRuntime&) = delete;
    OpenClRuntime(OpenClRuntime&&) = delete;
    OpenClRuntime& operator=(OpenClRuntime&&) = delete;

    void release() {
        if (buffer != nullptr) {
            clReleaseMemObject(buffer);
            buffer = nullptr;
        }
        capacity = 0;
        if (kernel != nullptr) {
            clReleaseKernel(kernel);
            kernel = nullptr;
        }
        if (program != nullptr) {
            clReleaseProgram(program);
            program = nullptr;
        }
        if (queue != nullptr) {
            clReleaseCommandQueue(queue);
            queue = nullptr;
        }
        if (context != nullptr) {
            clReleaseContext(context);
            context = nullptr;
        }
        platform_index = -1;
    }

    void ensure_device(int platform) {
        if (platform_index == platform && context != nullptr) {
            return;
        }
        release();

        cl_uint platform_count = 0;
        throw_if_cl_error(clGetPlatformIDs(0, nullptr, &platform_count));
        if (platform < 0 || static_cast<cl_uint>(platform) >= platform_count) {
            throw std::runtime_error("OpenCL platform index out of range");
        }

        std::vector<cl_platform_id> platforms(platform_count);
        throw_if_cl_error(clGetPlatformIDs(platform_count, platforms.data(), nullptr));

        cl_device_id device = nullptr;
        throw_if_cl_error(
            clGetDeviceIDs(platforms[static_cast<std::size_t>(platform)], CL_DEVICE_TYPE_GPU, 1,
                           &device, nullptr));

        cl_int status = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        throw_if_cl_error(status);

        queue = clCreateCommandQueue(context, device, 0, &status);
        throw_if_cl_error(status);

        const char* source = kKernelSource;
        program = clCreateProgramWithSource(context, 1, &source, nullptr, &status);
        throw_if_cl_error(status);
        throw_if_cl_error(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));

        kernel = clCreateKernel(program, "gelu", &status);
        throw_if_cl_error(status);
        platform_index = platform;
    }

    void grow(std::size_t n) {
        if (n <= capacity) {
            return;
        }
        if (buffer != nullptr) {
            throw_if_cl_error(clReleaseMemObject(buffer));
            buffer = nullptr;
            capacity = 0;
        }
        cl_int status = CL_SUCCESS;
        buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), nullptr, &status);
        throw_if_cl_error(status);
        capacity = n;
    }
};

OpenClRuntime& opencl_runtime() {
    static OpenClRuntime runtime;
    return runtime;
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    const std::size_t n = input.size();
    const std::size_t bytes = n * sizeof(float);
    OpenClRuntime& runtime = opencl_runtime();
    runtime.ensure_device(platform);
    runtime.grow(n);

    throw_if_cl_error(clEnqueueWriteBuffer(runtime.queue, runtime.buffer, CL_FALSE, 0, bytes,
                                           input.data(), 0, nullptr, nullptr));

    const int count = static_cast<int>(n);
    throw_if_cl_error(clSetKernelArg(runtime.kernel, 0, sizeof(cl_mem), &runtime.buffer));
    throw_if_cl_error(clSetKernelArg(runtime.kernel, 1, sizeof(int), &count));

    const std::size_t groups = (n + kVecWidth - 1) / kVecWidth;
    const std::size_t global = ((groups + kLocalSize - 1) / kLocalSize) * kLocalSize;
    throw_if_cl_error(clEnqueueNDRangeKernel(runtime.queue, runtime.kernel, 1, nullptr, &global,
                                             &kLocalSize, 0, nullptr, nullptr));

    std::vector<float> output(n);

    throw_if_cl_error(clEnqueueReadBuffer(runtime.queue, runtime.buffer, CL_TRUE, 0, bytes,
                                          output.data(), 0, nullptr, nullptr));
    return output;
}
