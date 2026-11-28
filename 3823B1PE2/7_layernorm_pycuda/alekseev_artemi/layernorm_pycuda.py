import numpy as np

KERNEL = r"""
extern "C" __global__
void layernorm_kernel(
    const float* x,
    const float* gamma,
    const float* beta,
    float* out,
    int n,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int count = blockDim.x;
    int n4 = n / 4;

    const float4* x4 =
        reinterpret_cast<const float4*>(x + (size_t)row * n);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);
    float4* y4 =
        reinterpret_cast<float4*>(out + (size_t)row * n);

    float sum = 0.0f;
    float sumsq = 0.0f;

    for (int i = tid; i < n4; i += count) {
        float4 v = x4[i];

        sum += v.x + v.y + v.z + v.w;
        sumsq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    __shared__ float sums[256];
    __shared__ float squares[256];

    sums[tid] = sum;
    squares[tid] = sumsq;
    __syncthreads();

    for (int s = count / 2; s > 0; s /= 2) {
        if (tid < s) {
            sums[tid] += sums[tid + s];
            squares[tid] += squares[tid + s];
        }
        __syncthreads();
    }

    float mean = sums[0] / n;
    float var = fmaxf(squares[0] / n - mean * mean, 0.0f);
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < n4; i += count) {
        float4 v = x4[i];
        float4 g = g4[i];
        float4 b = b4[i];
        float4 y;

        y.x = g.x * (v.x - mean) * inv_std + b.x;
        y.y = g.y * (v.y - mean) * inv_std + b.y;
        y.z = g.z * (v.z - mean) * inv_std + b.z;
        y.w = g.w * (v.w - mean) * inv_std + b.w;

        y4[i] = y;
    }
}
"""

_module = None
_func = None

_d_x = None
_d_y = None
_d_g = None
_d_b = None

_cap = 0
_gcap = 0


def _get_func():
    global _module, _func

    if _func is None:
        import pycuda.autoinit
        from pycuda.compiler import SourceModule

        _module = SourceModule(KERNEL, options=["-O3"])
        _func = _module.get_function("layernorm_kernel")

    return _func


def _get_buffers(size, n):
    global _d_x, _d_y, _d_g, _d_b, _cap, _gcap

    import pycuda.driver as cuda

    if size > _cap:
        if _d_x is not None:
            _d_x.free()
            _d_y.free()

        _d_x = cuda.mem_alloc(size * 4)
        _d_y = cuda.mem_alloc(size * 4)
        _cap = size

    if n > _gcap:
        if _d_g is not None:
            _d_g.free()
            _d_b.free()

        _d_g = cuda.mem_alloc(n * 4)
        _d_b = cuda.mem_alloc(n * 4)
        _gcap = n

    return _d_x, _d_y, _d_g, _d_b


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    import pycuda.driver as cuda

    x = np.asarray(input, dtype=np.float32).reshape(-1)
    g = np.asarray(gamma, dtype=np.float32).reshape(-1)
    b = np.asarray(beta, dtype=np.float32).reshape(-1)

    n = int(row_size)

    if x.size == 0:
        return np.empty(0, dtype=np.float32)

    x = np.ascontiguousarray(x)
    g = np.ascontiguousarray(g)
    b = np.ascontiguousarray(b)

    rows = x.size // n
    result = np.empty_like(x)

    func = _get_func()
    d_x, d_y, d_g, d_b = _get_buffers(x.size, n)

    cuda.memcpy_htod(d_x, x)
    cuda.memcpy_htod(d_g, g)
    cuda.memcpy_htod(d_b, b)

    func(
        d_x, d_g, d_b, d_y,
        np.int32(n),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(rows, 1, 1)
    )

    cuda.memcpy_dtoh(result, d_y)

    return result
