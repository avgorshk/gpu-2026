#include "gelu_ocl.h"
#include <CL/cl.h>
#include <string>

static const char* KERNEL_SRC = R"(
__kernel void gelu_kernel_vec(__global const float4* restrict input,
                              __global float4* restrict output,
                              const int n4) {
    int i = get_global_id(0);
    if (i >= n4) return;

    float4 x = input[i];
    float4 y;

    const float c = 0.7978845608028654f;
    const float k = 0.044715f;

    float x0 = x.x, x1 = x.y, x2 = x.z, x3 = x.w;

    float z0 = c * (x0 + k * x0 * x0 * x0);
    float z1 = c * (x1 + k * x1 * x1 * x1);
    float z2 = c * (x2 + k * x2 * x2 * x2);
    float z3 = c * (x3 + k * x3 * x3 * x3);

    y.x = 0.5f * x0 * (1.0f - 2.0f / (exp(2.0f * z0) + 1.0f));
    y.y = 0.5f * x1 * (1.0f - 2.0f / (exp(2.0f * z1) + 1.0f));
    y.z = 0.5f * x2 * (1.0f - 2.0f / (exp(2.0f * z2) + 1.0f));
    y.w = 0.5f * x3 * (1.0f - 2.0f / (exp(2.0f * z3) + 1.0f));

    output[i] = y;
}

__kernel void gelu_kernel_scalar(__global const float* restrict input,
                                 __global float* restrict output,
                                 const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = input[i];
    const float c = 0.7978845608028654f;
    const float k = 0.044715f;
    float z = c * (x + k * x * x * x);
    output[i] = 0.5f * x * (1.0f - 2.0f / (exp(2.0f * z) + 1.0f));
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    cl_platform_id platforms[16];
    clGetPlatformIDs(16, platforms, NULL);
    cl_platform_id plat = platforms[platform];

    cl_device_id device;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &KERNEL_SRC, NULL, NULL);
    clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);

    cl_kernel kernel_vec = clCreateKernel(program, "gelu_kernel_vec", NULL);
    cl_kernel kernel_scl = clCreateKernel(program, "gelu_kernel_scalar", NULL);

    size_t n = input.size();
    size_t nbytes = n * sizeof(float);
    bool use_vec = (n % 4 == 0);
    size_t n4 = n / 4;

    cl_mem in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nbytes, (void*)input.data(), NULL);
    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, NULL);

    std::vector<float> output(n);

    if (use_vec) {
        clSetKernelArg(kernel_vec, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel_vec, 1, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel_vec, 2, sizeof(int), &n4);

        size_t local_size = 256;
        size_t global_size = ((n4 + local_size - 1) / local_size) * local_size;

        clEnqueueNDRangeKernel(queue, kernel_vec, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    } else {
        int n_int = static_cast<int>(n);
        clSetKernelArg(kernel_scl, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel_scl, 1, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel_scl, 2, sizeof(int), &n_int);

        size_t local_size = 256;
        size_t global_size = ((n + local_size - 1) / local_size) * local_size;

        clEnqueueNDRangeKernel(queue, kernel_scl, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    }

    clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, nbytes, output.data(), 0, NULL, NULL);

    clReleaseMemObject(in_buf);
    clReleaseMemObject(out_buf);
    clReleaseKernel(kernel_vec);
    clReleaseKernel(kernel_scl);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}