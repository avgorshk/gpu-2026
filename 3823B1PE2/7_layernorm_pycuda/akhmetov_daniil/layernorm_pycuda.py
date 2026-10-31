import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL_SOURCE = """
#include <cuda_runtime.h>

template <int BLOCK_THREADS, int VEC_ITEMS>
__global__ void layernorm_reg_kernel(const float* __restrict__ input,
                                     const float* __restrict__ gamma,
                                     const float* __restrict__ beta,
                                     float* __restrict__ output,
                                     float eps) {
    constexpr int ROW_SIZE = BLOCK_THREADS * VEC_ITEMS * 4;
    constexpr float INV_N = 1.0f / static_cast<float>(ROW_SIZE);

    const size_t row = blockIdx.x;
    const int tid = threadIdx.x;

    const float4* row_in = reinterpret_cast<const float4*>(input + row * ROW_SIZE);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    const float4* beta_vec = reinterpret_cast<const float4*>(beta);
    float4* row_out = reinterpret_cast<float4*>(output + row * ROW_SIZE);

    float4 reg[VEC_ITEMS];
    float local_sum = 0.0f;

    #pragma unroll
    for (int k = 0; k < VEC_ITEMS; ++k) {
        reg[k] = row_in[k * BLOCK_THREADS + tid];
        local_sum += (reg[k].x + reg[k].y) + (reg[k].z + reg[k].w);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    __shared__ float s_val[32];
    __shared__ float s_broadcast;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    if (lane == 0) {
        s_val[wid] = local_sum;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (BLOCK_THREADS / 32)) ? s_val[lane] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_THREADS / 64); offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            s_broadcast = val * INV_N;
        }
    }
    __syncthreads();
    const float mean = s_broadcast;

    float local_var = 0.0f;
    #pragma unroll
    for (int k = 0; k < VEC_ITEMS; ++k) {
        reg[k].x -= mean;
        reg[k].y -= mean;
        reg[k].z -= mean;
        reg[k].w -= mean;
        local_var += (reg[k].x * reg[k].x + reg[k].y * reg[k].y) +
                     (reg[k].z * reg[k].z + reg[k].w * reg[k].w);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_var += __shfl_down_sync(0xffffffff, local_var, offset);
    }

    if (lane == 0) {
        s_val[wid] = local_var;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (BLOCK_THREADS / 32)) ? s_val[lane] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_THREADS / 64); offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            // Быстрый аппаратный обратный квадратный корень
            s_broadcast = rsqrtf(val * INV_N + eps);
        }
    }
    __syncthreads();
    const float inv_std = s_broadcast;

    #pragma unroll
    for (int k = 0; k < VEC_ITEMS; ++k) {
        int idx = k * BLOCK_THREADS + tid;
        float4 g = __ldg(&gamma_vec[idx]);
        float4 b = __ldg(&beta_vec[idx]);

        reg[k].x = fmaf(reg[k].x * inv_std, g.x, b.x);
        reg[k].y = fmaf(reg[k].y * inv_std, g.y, b.y);
        reg[k].z = fmaf(reg[k].z * inv_std, g.z, b.z);
        reg[k].w = fmaf(reg[k].w * inv_std, g.w, b.w);

        row_out[idx] = reg[k];
    }
}

template <int BLOCK_THREADS>
__global__ void layernorm_general_kernel(const float* __restrict__ input,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ output,
                                         int row_size,
                                         float eps) {
    const size_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_in = input + row * static_cast<size_t>(row_size);
    float* row_out = output + row * static_cast<size_t>(row_size);

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_THREADS) {
        local_sum += row_in[i];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    __shared__ float s_val[32];
    __shared__ float s_broadcast;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    if (lane == 0) {
        s_val[wid] = local_sum;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (BLOCK_THREADS / 32)) ? s_val[lane] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_THREADS / 64); offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            s_broadcast = val / static_cast<float>(row_size);
        }
    }
    __syncthreads();
    const float mean = s_broadcast;

    float local_var = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_THREADS) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_var += __shfl_down_sync(0xffffffff, local_var, offset);
    }

    if (lane == 0) {
        s_val[wid] = local_var;
    }
    __syncthreads();

    if (wid == 0) {
        float val = (lane < (BLOCK_THREADS / 32)) ? s_val[lane] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_THREADS / 64); offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            s_broadcast = rsqrtf(val / static_cast<float>(row_size) + eps);
        }
    }
    __syncthreads();
    const float inv_std = s_broadcast;

    for (int i = tid; i < row_size; i += BLOCK_THREADS) {
        float norm = (row_in[i] - mean) * inv_std;
        row_out[i] = fmaf(norm, __ldg(&gamma[i]), __ldg(&beta[i]));
    }
}

extern "C" {
__global__ void ln_16384(const float* i, const float* g, const float* b, float* o, float eps) {
    layernorm_reg_kernel<512, 8>(i, g, b, o, eps);
}
__global__ void ln_8192(const float* i, const float* g, const float* b, float* o, float eps) {
    layernorm_reg_kernel<256, 8>(i, g, b, o, eps);
}
__global__ void ln_4096(const float* i, const float* g, const float* b, float* o, float eps) {
    layernorm_reg_kernel<256, 4>(i, g, b, o, eps);
}
__global__ void ln_2048(const float* i, const float* g, const float* b, float* o, float eps) {
    layernorm_reg_kernel<256, 2>(i, g, b, o, eps);
}
__global__ void ln_1024(const float* i, const float* g, const float* b, float* o, float eps) {
    layernorm_reg_kernel<256, 1>(i, g, b, o, eps);
}
__global__ void ln_general(const float* i, const float* g, const float* b, float* o, int row_size, float eps) {
    layernorm_general_kernel<256>(i, g, b, o, row_size, eps);
}
}
"""

_module = SourceModule(_KERNEL_SOURCE, options=["-O3", "--use_fast_math"], no_extern_c=True)
_kernels = {
    16384: (_module.get_function("ln_16384"), 512),
    8192: (_module.get_function("ln_8192"), 256),
    4096: (_module.get_function("ln_4096"), 256),
    2048: (_module.get_function("ln_2048"), 256),
    1024: (_module.get_function("ln_1024"), 256),
}
_general_kernel = _module.get_function("ln_general")
_stream = cuda.Stream()

_cache = {
    "in_bytes": 0,
    "param_bytes": 0,
    "d_in": None,
    "d_out": None,
    "d_gamma": None,
    "d_beta": None,
}


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.
    """
    x = np.ascontiguousarray(input, dtype=np.float32)
    g = np.ascontiguousarray(gamma, dtype=np.float32)
    b = np.ascontiguousarray(beta, dtype=np.float32)

    total_elems = x.size
    if total_elems == 0:
        return np.empty_like(x)

    row_size = int(row_size)
    row_count = total_elems // row_size
    in_bytes = x.nbytes
    param_bytes = g.nbytes

    if _cache["in_bytes"] < in_bytes:
        if _cache["d_in"] is not None:
            _cache["d_in"].free()
            _cache["d_out"].free()
        _cache["d_in"] = cuda.mem_alloc(in_bytes)
        _cache["d_out"] = cuda.mem_alloc(in_bytes)
        _cache["in_bytes"] = in_bytes

    if _cache["param_bytes"] < param_bytes:
        if _cache["d_gamma"] is not None:
            _cache["d_gamma"].free()
            _cache["d_beta"].free()
        _cache["d_gamma"] = cuda.mem_alloc(param_bytes)
        _cache["d_beta"] = cuda.mem_alloc(param_bytes)
        _cache["param_bytes"] = param_bytes

    d_in = _cache["d_in"]
    d_out = _cache["d_out"]
    d_gamma = _cache["d_gamma"]
    d_beta = _cache["d_beta"]

    cuda.memcpy_htod_async(d_in, x, _stream)
    cuda.memcpy_htod_async(d_gamma, g, _stream)
    cuda.memcpy_htod_async(d_beta, b, _stream)

    eps_f32 = np.float32(eps)
    if row_size in _kernels:
        func, threads = _kernels[row_size]
        func(
            d_in,
            d_gamma,
            d_beta,
            d_out,
            eps_f32,
            block=(threads, 1, 1),
            grid=(row_count, 1, 1),
            stream=_stream,
        )
    else:
        _general_kernel(
            d_in,
            d_gamma,
            d_beta,
            d_out,
            np.int32(row_size),
            eps_f32,
            block=(256, 1, 1),
            grid=(row_count, 1, 1),
            stream=_stream,
        )

    output = np.empty(x.shape, dtype=np.float32)

    cuda.memcpy_dtoh_async(output, d_out, _stream)
    _stream.synchronize()

    return output