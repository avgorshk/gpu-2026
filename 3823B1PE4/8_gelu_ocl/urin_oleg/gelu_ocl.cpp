#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kGeluKernel[] = R"CLC(
__kernel void gelu(__global const float* input,
                   __global float* output,
                   const ulong count) {
    const ulong i = (ulong)get_global_id(0);
    if (i < count) {
        const float x = input[i];
        const float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        output[i] = x / (1.0f + exp(-2.0f * z));
    }
}
)CLC";

void Check(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (OpenCL error " +
                                 std::to_string(error) + ")");
    }
}

const std::vector<cl_platform_id>& Platforms() {
    static const std::vector<cl_platform_id> platforms = [] {
        cl_uint count = 0;
        Check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs");
        std::vector<cl_platform_id> ids(count);
        Check(clGetPlatformIDs(count, ids.data(), nullptr), "clGetPlatformIDs");
        return ids;
    }();
    return platforms;
}

class OpenCLState {
public:
    explicit OpenCLState(cl_platform_id platform) {
        try {
            Initialize(platform);
        } catch (...) {
            Release();
            throw;
        }
    }

    ~OpenCLState() { Release(); }

    OpenCLState(const OpenCLState&) = delete;
    OpenCLState& operator=(const OpenCLState&) = delete;

    std::vector<float> Run(const std::vector<float>& values) {
        const size_t count = values.size();
        if (count > std::numeric_limits<size_t>::max() / sizeof(float) ||
            count > std::numeric_limits<cl_ulong>::max() ||
            count > std::numeric_limits<size_t>::max() - (local_size_ - 1)) {
            throw std::length_error("GELU input is too large");
        }
        const size_t bytes = count * sizeof(float);
        Reserve(bytes);

        Check(clEnqueueWriteBuffer(queue_, input_, CL_TRUE, 0, bytes, values.data(),
                                   0, nullptr, nullptr),
              "clEnqueueWriteBuffer");

        const cl_ulong element_count = static_cast<cl_ulong>(count);
        Check(clSetKernelArg(kernel_, 0, sizeof(input_), &input_), "clSetKernelArg(input)");
        Check(clSetKernelArg(kernel_, 1, sizeof(output_), &output_), "clSetKernelArg(output)");
        Check(clSetKernelArg(kernel_, 2, sizeof(element_count), &element_count),
              "clSetKernelArg(count)");

        const size_t global_size = ((count + local_size_ - 1) / local_size_) * local_size_;
        Check(clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &global_size,
                                     &local_size_, 0, nullptr, nullptr),
              "clEnqueueNDRangeKernel");
        Check(clFlush(queue_), "clFlush");

        std::vector<float> result(count);
        Check(clEnqueueReadBuffer(queue_, output_, CL_TRUE, 0, bytes, result.data(),
                                  0, nullptr, nullptr),
              "clEnqueueReadBuffer");
        return result;
    }

private:
    void Initialize(cl_platform_id platform) {
        Check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr),
              "clGetDeviceIDs(CL_DEVICE_TYPE_GPU)");

        const cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
        cl_int error = CL_SUCCESS;
        context_ = clCreateContext(properties, 1, &device_, nullptr, nullptr, &error);
        Check(error, "clCreateContext");

        queue_ = clCreateCommandQueue(context_, device_, 0, &error);
        Check(error, "clCreateCommandQueue");

        const char* source = kGeluKernel;
        program_ = clCreateProgramWithSource(context_, 1, &source, nullptr, &error);
        Check(error, "clCreateProgramWithSource");

        error = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                  &log_size);
            std::string log(log_size, '\0');
            if (log_size != 0) {
                clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size,
                                      &log[0], nullptr);
            }
            throw std::runtime_error("clBuildProgram failed (OpenCL error " +
                                     std::to_string(error) + "): " + log);
        }

        kernel_ = clCreateKernel(program_, "gelu", &error);
        Check(error, "clCreateKernel");

        size_t max_group_size = 0;
        Check(clGetKernelWorkGroupInfo(kernel_, device_, CL_KERNEL_WORK_GROUP_SIZE,
                                       sizeof(max_group_size), &max_group_size, nullptr),
              "clGetKernelWorkGroupInfo");
        if (max_group_size == 0) {
            throw std::runtime_error("OpenCL kernel has no supported work-group size");
        }
        local_size_ = std::min<size_t>(256, max_group_size);
    }

    void Release() {
        if (output_) clReleaseMemObject(output_);
        if (input_) clReleaseMemObject(input_);
        if (kernel_) clReleaseKernel(kernel_);
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }
    void Reserve(size_t bytes) {
        if (bytes <= capacity_) return;

        cl_int error = CL_SUCCESS;
        cl_mem new_input = clCreateBuffer(context_, CL_MEM_READ_ONLY, bytes, nullptr, &error);
        Check(error, "clCreateBuffer(input)");

        cl_mem new_output = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bytes, nullptr, &error);
        if (error != CL_SUCCESS) {
            clReleaseMemObject(new_input);
            Check(error, "clCreateBuffer(output)");
        }

        if (input_) clReleaseMemObject(input_);
        if (output_) clReleaseMemObject(output_);
        input_ = new_input;
        output_ = new_output;
        capacity_ = bytes;
    }

    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_mem input_ = nullptr;
    cl_mem output_ = nullptr;
    size_t capacity_ = 0;
    size_t local_size_ = 1;
};

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};
    const auto& platforms = Platforms();
    if (platform < 0 || static_cast<size_t>(platform) >= platforms.size()) {
        throw std::out_of_range("Invalid OpenCL platform index");
    }

    // The queue and buffers are shared by calls on the same platform.
    static std::mutex mutex;
    static std::map<int, std::unique_ptr<OpenCLState>> states;
    std::lock_guard<std::mutex> lock(mutex);
    auto& state = states[platform];
    if (!state) state = std::make_unique<OpenCLState>(platforms[platform]);
    return state->Run(input);
}
