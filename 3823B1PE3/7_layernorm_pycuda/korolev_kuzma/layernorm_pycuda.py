"""Row-wise float32 LayerNorm implemented with a compiled PyCUDA kernel."""

import atexit
import operator
import threading

import numpy as np


_SOURCE = r"""
#include <math.h>

__device__ float warp_sum(float value) {
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

extern "C" __global__ void layernorm_rows(
    float *input_output, const float *gamma, const float *beta,
    unsigned long long columns, unsigned long long rows, float eps) {
    __shared__ float partial[8];
    __shared__ float mean_offset;
    __shared__ float inverse_std;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    for (unsigned long long row = blockIdx.x; row < rows; row += gridDim.x) {
        const unsigned long long base = row * columns;
        // Center first so large common offsets do not spoil the mean.
        const float origin = input_output[base];
        float sum = 0.0f;
        for (unsigned long long col = threadIdx.x; col < columns; col += blockDim.x)
            sum += input_output[base + col] - origin;
        sum = warp_sum(sum);
        if (lane == 0) partial[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = warp_sum(lane < 8 ? partial[lane] : 0.0f);
            if (lane == 0) mean_offset = sum / (float)columns;
        }
        __syncthreads();

        // A second pass avoids cancellation in E[x*x] - E[x]*E[x].
        float squared = 0.0f;
        for (unsigned long long col = threadIdx.x; col < columns; col += blockDim.x) {
            const float centered = (input_output[base + col] - origin) - mean_offset;
            squared = fmaf(centered, centered, squared);
        }
        squared = warp_sum(squared);
        if (lane == 0) partial[warp] = squared;
        __syncthreads();
        if (warp == 0) {
            squared = warp_sum(lane < 8 ? partial[lane] : 0.0f);
            if (lane == 0) inverse_std = rsqrtf(squared / (float)columns + eps);
        }
        __syncthreads();
        for (unsigned long long col = threadIdx.x; col < columns; col += blockDim.x) {
            const float normalized =
                ((input_output[base + col] - origin) - mean_offset) * inverse_std;
            input_output[base + col] = fmaf(gamma[col], normalized, beta[col]);
        }
        __syncthreads();
    }
}
"""


_lock = threading.RLock()
_cuda = None
_fallback_context = None
_states = {}


class _State:
    """A module and grow-only buffers belonging to exactly one CUDA context."""

    def __init__(self):
        from pycuda.compiler import SourceModule

        self.module = SourceModule(_SOURCE, no_extern_c=True)
        self.kernel = self.module.get_function("layernorm_rows")
        self.data = self.gamma = self.beta = None
        self.data_bytes = self.parameter_bytes = 0

    def reserve(self, data_bytes, parameter_bytes):
        if data_bytes > self.data_bytes:
            if self.data is not None:
                self.data.free()
            self.data = None
            self.data_bytes = 0
            self.data = _cuda.mem_alloc(data_bytes)
            self.data_bytes = data_bytes
        if parameter_bytes > self.parameter_bytes:
            if self.gamma is not None:
                self.gamma.free()
            if self.beta is not None:
                self.beta.free()
            self.gamma = self.beta = None
            self.parameter_bytes = 0
            self.gamma = _cuda.mem_alloc(parameter_bytes)
            try:
                self.beta = _cuda.mem_alloc(parameter_bytes)
            except Exception:
                self.gamma.free()
                self.gamma = None
                raise
            self.parameter_bytes = parameter_bytes


def clear_layernorm_cache():
    """Release cached modules/buffers; never detach a caller-owned context."""
    with _lock:
        _states.clear()


def _shutdown():
    global _fallback_context
    clear_layernorm_cache()
    if _fallback_context is not None:
        try:
            _fallback_context.detach()
        except Exception:
            pass  # The driver may already have shut down during interpreter exit.
        _fallback_context = None


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """Normalize each row and apply gamma/beta; return a flat float32 ndarray.

    Inputs must be one-dimensional; gamma and beta have row_size entries.
    An existing PyCUDA context is used unchanged. Otherwise device 0's
    primary context is retained and pushed only for the duration of the call.
    CUDA compilation and device allocations are reused between calls.
    """
    global _cuda, _fallback_context

    if isinstance(row_size, (bool, np.bool_)):
        raise ValueError("row_size must be a positive integer")
    try:
        row_size = operator.index(row_size)
    except TypeError as error:
        raise ValueError("row_size must be a positive integer") from error
    if row_size <= 0:
        raise ValueError("row_size must be a positive integer")
    eps = np.float32(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive in float32")

    data = np.asarray(input, dtype=np.float32)
    scale = np.asarray(gamma, dtype=np.float32)
    shift = np.asarray(beta, dtype=np.float32)
    if data.ndim != 1 or scale.ndim != 1 or shift.ndim != 1:
        raise ValueError("input, gamma and beta must be one-dimensional")
    if data.size % row_size != 0:
        raise ValueError("input length must be divisible by row_size")
    if scale.size != row_size or shift.size != row_size:
        raise ValueError("gamma and beta must each contain row_size elements")
    if data.size == 0:
        return np.empty(0, dtype=np.float32)
    data = np.ascontiguousarray(data)
    scale = np.ascontiguousarray(scale)
    shift = np.ascontiguousarray(shift)
    rows = data.size // row_size

    # Serialize access to shared buffers, including callers using one context
    # from several host threads. This also keeps the context cache consistent.
    with _lock:
        if _cuda is None:
            import pycuda.driver as cuda

            cuda.init()
            _cuda = cuda
            atexit.register(_shutdown)
        context = _cuda.Context.get_current()
        pushed = context is None
        if pushed:
            if _fallback_context is None:
                _fallback_context = _cuda.Device(0).retain_primary_context()
            context = _fallback_context
            context.push()
        try:
            state = _states.get(context)
            if state is None:
                state = _State()
                _states[context] = state
            state.reserve(data.nbytes, scale.nbytes)
            _cuda.memcpy_htod(state.data, data)
            _cuda.memcpy_htod(state.gamma, scale)
            _cuda.memcpy_htod(state.beta, shift)
            state.kernel(
                state.data, state.gamma, state.beta,
                np.uint64(row_size), np.uint64(rows), eps,
                block=(256, 1, 1), grid=(min(rows, 65535), 1, 1),
            )
            output = np.empty_like(data)
            # The blocking copy waits for the kernel and reports CUDA failures.
            _cuda.memcpy_dtoh(output, state.data)
            return output
        finally:
            if pushed:
                _cuda.Context.pop()
