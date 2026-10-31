#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

namespace {

const char* kKernelSource = R"CLC(
__kernel void gelu_kernel(__global float* data, const uint n) {
    const float k1 = -1.5957691216057308f;
    const float k2 = -0.0713548162726009f;

    uint gid = get_global_id(0);
    uint n4 = n >> 2;

    if (gid < n4) {
        float4 v = vload4(gid, data);
        float4 z = v * (k1 + k2 * v * v);
        float4 e = native_exp(z);
        float4 r = v / (1.0f + e);
        vstore4(r, gid, data);
    } else {
        uint tail_idx = gid - n4;
        uint rem = n & 3u;
        if (tail_idx < rem) {
            uint idx = n4 * 4u + tail_idx;
            float x = data[idx];
            float z = x * (k1 + k2 * x * x);
            float e = native_exp(z);
            data[idx] = x / (1.0f + e);
        }
    }
}
)CLC";

constexpr size_t kLocalSize = 256;

struct OclContext {
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_mem d_data = nullptr;
    size_t capacity = 0;

    int cached_platform_index = -1;
    bool ready = false;
};

OclContext& Ctx() {
    static OclContext ctx;
    return ctx;
}

void InitOpenCL(OclContext& ctx, int platform_index) {
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    if (num_platforms == 0) {
        return;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    if (platform_index < 0 || static_cast<cl_uint>(platform_index) >= num_platforms) {
        platform_index = 0;
    }
    ctx.platform_id = platforms[platform_index];

    cl_int err = clGetDeviceIDs(ctx.platform_id, CL_DEVICE_TYPE_GPU, 1, &ctx.device_id, nullptr);
    if (err != CL_SUCCESS) {
        return;
    }

    ctx.context = clCreateContext(nullptr, 1, &ctx.device_id, nullptr, nullptr, &err);

#ifdef CL_VERSION_2_0
    ctx.queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device_id, nullptr, &err);
#else
    ctx.queue = clCreateCommandQueue(ctx.context, ctx.device_id, 0, &err);
#endif

    const char* src = kKernelSource;
    const size_t src_len = std::strlen(kKernelSource);
    ctx.program = clCreateProgramWithSource(ctx.context, 1, &src, &src_len, &err);
    clBuildProgram(ctx.program, 1, &ctx.device_id, "-cl-fast-relaxed-math", nullptr, nullptr);

    ctx.kernel = clCreateKernel(ctx.program, "gelu_kernel", &err);

    ctx.cached_platform_index = platform_index;
    ctx.ready = true;
}

void EnsureCapacity(OclContext& ctx, size_t n) {
    if (n > ctx.capacity) {
        if (ctx.d_data != nullptr) {
            clReleaseMemObject(ctx.d_data);
        }
        cl_int err = CL_SUCCESS;
        ctx.d_data = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, n * sizeof(float), nullptr, &err);
        ctx.capacity = n;
    }
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    OclContext& ctx = Ctx();
    if (!ctx.ready || ctx.cached_platform_index != platform) {
        InitOpenCL(ctx, platform);
    }
    if (!ctx.ready) {
        return {};
    }

    EnsureCapacity(ctx, n);

    const size_t bytes = n * sizeof(float);

    std::vector<float> output;
    std::thread allocator([&output, n] { output.resize(n); });

    clEnqueueWriteBuffer(ctx.queue, ctx.d_data, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    const size_t n4 = n / 4;
    const size_t rem = n % 4;
    const size_t total_threads = n4 + rem;
    const size_t global_size = ((total_threads + kLocalSize - 1) / kLocalSize) * kLocalSize;

    cl_uint n_arg = static_cast<cl_uint>(n);
    clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.d_data);
    clSetKernelArg(ctx.kernel, 1, sizeof(cl_uint), &n_arg);

    clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr, &global_size, &kLocalSize, 0, nullptr,
                           nullptr);

    allocator.join();

    clEnqueueReadBuffer(ctx.queue, ctx.d_data, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);

    return output;
}