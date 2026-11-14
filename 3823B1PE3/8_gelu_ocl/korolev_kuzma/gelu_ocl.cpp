#include "gelu_ocl.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr char kGeluSource[] = R"CLC(
__kernel void gelu(__global float* values, const ulong count) {
    const size_t i = get_global_id(0);
    if (i < count) {
        const float x = values[i];
        const float t = 1.5957691216057308f * x *
                        (1.0f + 0.044715f * x * x);
        // GELU(x) = x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715*x^3)).
        // The exponent is nonpositive, avoiding overflow in the tails.
        const float e = exp(-fabs(t));
        values[i] = x * (x >= 0.0f ? 1.0f / (1.0f + e)
                                   : e / (1.0f + e));
    }
}
)CLC";

void Check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed (OpenCL error " +
                                 std::to_string(status) + ")");
    }
}

struct OpenCLState {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buffer = nullptr;
    std::size_t capacity = 0;
    std::size_t local_size = 1;

    OpenCLState() = default;
    OpenCLState(const OpenCLState&) = delete;
    OpenCLState& operator=(const OpenCLState&) = delete;

    ~OpenCLState() {
        if (buffer) clReleaseMemObject(buffer);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    void Initialize(int platform_index) {
        cl_uint platform_count = 0;
        Check(clGetPlatformIDs(0, nullptr, &platform_count),
              "clGetPlatformIDs");
        if (static_cast<unsigned int>(platform_index) >= platform_count) {
            throw std::out_of_range("OpenCL platform index is out of range");
        }
        std::vector<cl_platform_id> platforms(platform_count);
        Check(clGetPlatformIDs(platform_count, platforms.data(), nullptr),
              "clGetPlatformIDs");

        // Device zero among GPU devices on exactly the requested platform.
        cl_device_id device = nullptr;
        Check(clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 1,
                             &device, nullptr),
              "clGetDeviceIDs(CL_DEVICE_TYPE_GPU)");
        cl_int status = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        Check(status, "clCreateContext");
        queue = clCreateCommandQueue(context, device, 0, &status);
        Check(status, "clCreateCommandQueue");
        const char* source = kGeluSource;
        program = clCreateProgramWithSource(context, 1, &source, nullptr,
                                            &status);
        Check(status, "clCreateProgramWithSource");
        status = clBuildProgram(program, 1, &device, "-cl-std=CL1.2",
                                nullptr, nullptr);
        if (status != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                  nullptr, &log_size);
            std::string log(log_size, '\0');
            if (log_size) {
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                      log_size, &log[0], nullptr);
                if (log.back() == '\0') log.pop_back();
            }
            throw std::runtime_error("clBuildProgram failed (OpenCL error " +
                                     std::to_string(status) + "): " + log);
        }
        kernel = clCreateKernel(program, "gelu", &status);
        Check(status, "clCreateKernel");

        std::size_t kernel_limit = 0;
        Check(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                       sizeof(kernel_limit), &kernel_limit,
                                       nullptr),
              "clGetKernelWorkGroupInfo");
        cl_uint dimensions = 0;
        Check(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                               sizeof(dimensions), &dimensions, nullptr),
              "clGetDeviceInfo(dimensions)");
        std::vector<std::size_t> item_limits(dimensions);
        Check(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                               item_limits.size() * sizeof(std::size_t),
                               item_limits.data(), nullptr),
              "clGetDeviceInfo(work item sizes)");
        local_size = std::min({std::size_t{256}, kernel_limit, item_limits[0]});
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) return;
        cl_int status = CL_SUCCESS;
        cl_mem replacement = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             count * sizeof(float), nullptr,
                                             &status);
        Check(status, "clCreateBuffer");
        if (buffer) clReleaseMemObject(buffer);
        buffer = replacement;
        capacity = count;
    }
};

OpenCLState& GetState(int platform) {
    // Independent queues, arguments and buffers make simultaneous calls safe.
    thread_local std::map<int, std::unique_ptr<OpenCLState>> states;
    auto found = states.find(platform);
    if (found == states.end()) {
        auto state = std::make_unique<OpenCLState>();
        // Ownership exists before initialization, so partial failures clean up.
        state->Initialize(platform);
        found = states.emplace(platform, std::move(state)).first;
    }
    return *found->second;
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (platform < 0) {
        throw std::invalid_argument("OpenCL platform index must be nonnegative");
    }
    if (input.empty()) return {};

    auto& state = GetState(platform);
    state.Reserve(input.size());
    const cl_ulong count = static_cast<cl_ulong>(input.size());
    const std::size_t bytes = input.size() * sizeof(float);
    const std::size_t global_size =
        ((input.size() + state.local_size - 1) / state.local_size) *
        state.local_size;

    try {
        Check(clEnqueueWriteBuffer(state.queue, state.buffer, CL_FALSE, 0,
                                    bytes, input.data(), 0, nullptr, nullptr),
              "clEnqueueWriteBuffer");
        Check(clSetKernelArg(state.kernel, 0, sizeof(state.buffer),
                             &state.buffer),
              "clSetKernelArg(buffer)");
        Check(clSetKernelArg(state.kernel, 1, sizeof(count), &count),
              "clSetKernelArg(count)");
        Check(clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr,
                                     &global_size, &state.local_size, 0,
                                     nullptr, nullptr),
              "clEnqueueNDRangeKernel");
        Check(clFlush(state.queue), "clFlush");

        // Allocate the result while the GPU processes the in-place device data.
        std::vector<float> output(input.size());
        Check(clEnqueueReadBuffer(state.queue, state.buffer, CL_TRUE, 0,
                                   bytes, output.data(), 0, nullptr, nullptr),
              "clEnqueueReadBuffer");
        return output;
    } catch (...) {
        // A nonblocking write may still refer to the caller's input buffer.
        clFinish(state.queue);
        throw;
    }
}
