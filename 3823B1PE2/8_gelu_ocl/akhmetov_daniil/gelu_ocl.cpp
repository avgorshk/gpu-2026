#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <cstddef>

static const char* kGeluKernelSource = R"(
__kernel void gelu_kernel(__global const float* restrict input,
                          __global float* restrict output,
                          const unsigned long n) {
    size_t idx = get_global_id(0);
    if (idx < n) {
        float x = input[idx];
        float x2 = x * x;
        float exponent = x * fma(x2, -0.07135481f, -1.59576912f);
        output[idx] = x / (1.0f + native_exp(exponent));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t size = input.size();
    if (size == 0) {
        return {};
    }
    size_t bytes = size * sizeof(float);

    static bool initialized = false;
    static int cached_platform = -1;
    static cl_context context = nullptr;
    static cl_command_queue queue = nullptr;
    static cl_program program = nullptr;
    static cl_kernel kernel = nullptr;
    static cl_mem d_in = nullptr;
    static cl_mem d_out = nullptr;
    static size_t cached_bytes = 0;

    if (!initialized || cached_platform != platform) {
        if (initialized) {
            if (d_in) clReleaseMemObject(d_in);
            if (d_out) clReleaseMemObject(d_out);
            if (kernel) clReleaseKernel(kernel);
            if (program) clReleaseProgram(program);
            if (queue) clReleaseCommandQueue(queue);
            if (context) clReleaseContext(context);
            d_in = nullptr;
            d_out = nullptr;
            cached_bytes = 0;
        }

        cl_uint num_platforms = 0;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        cl_platform_id plat = platforms[platform];

        cl_device_id device = nullptr;
        clGetDeviceIDs(plat, CL_DEVICE_GPU, 1, &device, nullptr);

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);

        program = clCreateProgramWithSource(context, 1, &kGeluKernelSource, nullptr, nullptr);
        clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu_kernel", nullptr);

        cached_platform = platform;
        initialized = true;
    }

    if (cached_bytes < bytes) {
        if (d_in) clReleaseMemObject(d_in);
        if (d_out) clReleaseMemObject(d_out);
        d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
        cached_bytes = bytes;
    }

    clEnqueueWriteBuffer(queue, d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    cl_ulong n_arg = static_cast<cl_ulong>(size);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    clSetKernelArg(kernel, 2, sizeof(cl_ulong), &n_arg);

    const size_t local_size = 256;
    const size_t global_size = ((size + local_size - 1) / local_size) * local_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);

    clFlush(queue);

    std::vector<float> output(size);

    clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    return output;
}