#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

const char* KERNEL_SRC = R"CLC(
__kernel void gelu_vector(__global float4* __restrict data, const int n4) {
    int gid = get_global_id(0);
    if (gid < n4) {
        float4 x = data[gid];
        float4 exp_arg = -x * (1.5957691216f + 0.0713548163f * x * x);
        data[gid] = x / (1.0f + native_exp(exp_arg));
    }
}
)CLC";

struct OpenCLContext {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;
    size_t capacity_bytes = 0;
    size_t local_work_size = 256;

    ~OpenCLContext() {
        if (buffer) clReleaseMemObject(buffer);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    void Init(int platform_idx) {
        cl_uint num_platforms = 0;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        cl_platform_id plat = platforms[platform_idx];

        cl_device_id device = nullptr;
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, device, 0, &err);

        program = clCreateProgramWithSource(context, 1, &KERNEL_SRC, nullptr, &err);
        clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);

        kernel = clCreateKernel(program, "gelu_vector", &err);

        size_t wg_size = 0;
        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg_size), &wg_size, nullptr);
        if (wg_size > 0 && wg_size < local_work_size) {
            local_work_size = wg_size;
        }
    }

    void EnsureCapacity(size_t bytes) {
        if (bytes <= capacity_bytes) {
            return;
        }
        if (buffer) {
            clReleaseMemObject(buffer);
            buffer = nullptr;
            capacity_bytes = 0;
        }
        cl_int err = CL_SUCCESS;
        buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        capacity_bytes = bytes;
    }
};

std::mutex g_ocl_lock;
std::unordered_map<int, std::unique_ptr<OpenCLContext>> g_contexts;

OpenCLContext* GetContext(int platform) {
    auto it = g_contexts.find(platform);
    if (it != g_contexts.end()) {
        return it->second.get();
    }
    auto ctx = std::make_unique<OpenCLContext>();
    ctx->Init(platform);
    OpenCLContext* raw = ctx.get();
    g_contexts[platform] = std::move(ctx);
    return raw;
}

inline float GeluScalarCpu(float x) {
    float arg = -x * (1.5957691216f + 0.0713548163f * x * x);
    return x / (1.0f + std::exp(arg));
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    std::lock_guard<std::mutex> lock(g_ocl_lock);
    OpenCLContext* ctx = GetContext(platform);

    const size_t n4 = n / 4;
    const size_t main_n = n4 * 4;
    const size_t bytes = main_n * sizeof(float);

    std::vector<float> output;

    if (main_n > 0) {
        ctx->EnsureCapacity(bytes);

        clEnqueueWriteBuffer(ctx->queue, ctx->buffer, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

        int n4_int = static_cast<int>(n4);
        clSetKernelArg(ctx->kernel, 0, sizeof(cl_mem), &ctx->buffer);
        clSetKernelArg(ctx->kernel, 1, sizeof(int), &n4_int);

        const size_t global = ((n4 + ctx->local_work_size - 1) / ctx->local_work_size) * ctx->local_work_size;
        clEnqueueNDRangeKernel(ctx->queue, ctx->kernel, 1, nullptr, &global, &ctx->local_work_size, 0, nullptr, nullptr);
        clFlush(ctx->queue);

        output.resize(n);

        clEnqueueReadBuffer(ctx->queue, ctx->buffer, CL_FALSE, 0, bytes, output.data(), 0, nullptr, nullptr);
    } else {
        output.resize(n);
    }

    for (size_t i = main_n; i < n; ++i) {
        output[i] = GeluScalarCpu(input[i]);
    }

    if (main_n > 0) {
        clFinish(ctx->queue);
    }

    return output;
}