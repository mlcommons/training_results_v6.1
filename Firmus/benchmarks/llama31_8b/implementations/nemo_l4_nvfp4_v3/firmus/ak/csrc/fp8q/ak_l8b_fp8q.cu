// ak_l8b_fp8q.cu (Llama 3.1 8B kernel campaign, 2026-10-07): FP8 per-tensor current-scaling quantization of a bf16
// tensor with TE 2.20's arithmetic (Float8CurrentScalingQuantizer, row-wise, no amax reduction), byte for byte as
// ak_l8b_fp8cs (fp8cs overlay):
//   amax      = max |x| over the tensor, fmaxf (NaN ignored)
//   scale     = compute_scale_from_amax(amax, fp8_max, force_pow_2_scales, epsilon, FLT_MAX)
//   data      = fp8(x * scale), round to nearest, saturating
//   scale_inv = __frcp_rn(scale)
// What changes is the traffic:
//   * amax_cast: two launches.  The amax pass writes one partial maximum per block (no atomics, no zeroing kernel);
//     the cast's blocks each reduce those partials and compute the scale themselves (the same arithmetic in every
//     block), block 0 stores scale_inv -- TE's zero-amax and compute-scale kernels are gone.
//   * both passes are persistent and grid-strided, U 16-byte vectors per thread per step, and either pass can walk the
//     tensor back to front (rev): the producer (a GEMM, the RoPE kernel) leaves the tensor's tail in L2, a back-to-front
//     amax pass reads it there and leaves the head, which a front-to-back cast then reads there.  The order changes no
//     value (a max, and an element-wise cast).
//   * cast_from_ws: the cast alone, scale from ws[0] (the q|k|v path: the scale comes from the RoPE kernel's partials).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cstdint>

namespace {

union V8 {
  uint4 u;
  __nv_bfloat16 h[8];
};

// TE recipe_common.cuh compute_scale_from_amax (as ak_l8b_fp8cs)
__device__ __forceinline__ float compute_scale_from_amax(float amax, float max_fp8, bool force_pow_2_scales,
                                                         float epsilon, float value_for_inf) {
  if (amax < epsilon) amax = epsilon;
  float scale = 1.f;
  if (isinf(amax) || amax == 0.f || isnan(amax)) return scale;
  scale = __fdiv_rn(max_fp8, amax);
  if (isinf(scale)) scale = value_for_inf;
  if (force_pow_2_scales) {
    uint32_t scale_bits = __float_as_uint(scale);
    scale_bits &= 0xFF800000u;
    scale = __uint_as_float(scale_bits);
  }
  return scale;
}

template <int kThreads>
__device__ __forceinline__ float block_max(float m) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  __shared__ float red[kThreads / 32];
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  float r = 0.f;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w) r = fmaxf(r, red[w]);
  return r;  // every thread holds the block maximum
}

__device__ __forceinline__ float vmax(const V8& a, float m) {
#pragma unroll
  for (int e = 0; e < 8; ++e) m = fmaxf(m, fabsf(__bfloat162float(a.h[e])));
  return m;
}

// The tensor as steps of kThreads * U vectors; block b takes steps b, b + G, ...; rev maps step s to n_steps - 1 - s.
constexpr int kAmaxThreads = 512;
template <int U, bool kRev>
__global__ void __launch_bounds__(kAmaxThreads) amax_part_kernel(const uint4* __restrict__ x, int64_t n_vec,
                                                                 float* __restrict__ part) {
  constexpr int64_t kStep = (int64_t)kAmaxThreads * U;
  const int64_t n_steps = (n_vec + kStep - 1) / kStep;
  float m = 0.f;
  for (int64_t s = blockIdx.x; s < n_steps; s += gridDim.x) {
    const int64_t st = kRev ? (n_steps - 1 - s) : s;
    const int64_t base = st * kStep + threadIdx.x;
    V8 a[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t v = base + (int64_t)u * kAmaxThreads;
      if (v < n_vec) a[u].u = x[v]; else a[u].u = make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) m = vmax(a[u], m);
  }
  const float r = block_max<kAmaxThreads>(m);
  if (threadIdx.x == 0) part[blockIdx.x] = r;
}

template <bool kE5M2>
__device__ __forceinline__ uint2 cast8(const V8& a, float scale) {
  constexpr __nv_fp8_interpretation_t kFmt = kE5M2 ? __NV_E5M2 : __NV_E4M3;
  uint32_t w[2];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    uint32_t word = 0;
#pragma unroll
    for (int e = 0; e < 4; e += 2) {
      const float f0 = __bfloat162float(a.h[4 * p + e]) * scale;
      const float f1 = __bfloat162float(a.h[4 * p + e + 1]) * scale;
      const __nv_fp8x2_storage_t pr = __nv_cvt_float2_to_fp8x2(make_float2(f0, f1), __NV_SATFINITE, kFmt);
      word |= (uint32_t)pr << (8 * e);
    }
    w[p] = word;
  }
  return make_uint2(w[0], w[1]);
}

// kFromPart: reduce part[0..n_part) and compute the scale in every block (block 0 stores scale_inv and ws);
// else the scale is ws[0].
constexpr int kCastThreads = 256;
template <bool kE5M2, int U, bool kRev, bool kFromPart>
__global__ void __launch_bounds__(kCastThreads) cast_kernel(const uint4* __restrict__ x, int64_t n_vec,
                                                            const float* __restrict__ part, int n_part,
                                                            float* __restrict__ ws, float* __restrict__ scale_inv,
                                                            float max_fp8, int pow2, float eps,
                                                            uint2* __restrict__ out) {
  float scale;
  if (kFromPart) {
    float m = 0.f;
    for (int i = threadIdx.x; i < n_part; i += kCastThreads) m = fmaxf(m, part[i]);
    const float amax = block_max<kCastThreads>(m);
    scale = compute_scale_from_amax(amax, max_fp8, pow2 != 0, eps, FLT_MAX);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      const float si = __frcp_rn(scale);
      scale_inv[0] = si;
      ws[0] = scale; ws[1] = si; ws[2] = amax;
    }
  } else {
    scale = ws[0];
  }
  constexpr int64_t kStep = (int64_t)kCastThreads * U;
  const int64_t n_steps = (n_vec + kStep - 1) / kStep;
  for (int64_t s = blockIdx.x; s < n_steps; s += gridDim.x) {
    const int64_t st = kRev ? (n_steps - 1 - s) : s;
    const int64_t base = st * kStep + threadIdx.x;
    V8 a[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t v = base + (int64_t)u * kCastThreads;
      if (v < n_vec) a[u].u = x[v];
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int64_t v = base + (int64_t)u * kCastThreads;
      if (v < n_vec) out[v] = cast8<kE5M2>(a[u], scale);
    }
  }
}

// one CTA: amax = fmaxf over n partials (the q|k|v path: the RoPE kernel's per-position maxima), the scale and
// scale_inv into ws [3] (scale, scale_inv, amax) and scale_inv -- ak_l8b_fp8cs's scale_kernel
constexpr int kRedThreads = 1024;
__global__ void __launch_bounds__(kRedThreads) scale_kernel(const float* __restrict__ part, int64_t n,
                                                            float* __restrict__ ws, float* __restrict__ scale_inv,
                                                            float max_fp8, int pow2, float eps) {
  float m = 0.f;
  for (int64_t i = threadIdx.x; i < n; i += kRedThreads) m = fmaxf(m, part[i]);
  const float r = block_max<kRedThreads>(m);
  if (threadIdx.x == 0) {
    const float scale = compute_scale_from_amax(r, max_fp8, pow2 != 0, eps, FLT_MAX);
    ws[0] = scale;
    ws[1] = __frcp_rn(scale);
    ws[2] = r;
    scale_inv[0] = ws[1];
  }
}

void check_io(const at::Tensor& x, const at::Tensor& data, const at::Tensor& scale_inv) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "x: contiguous CUDA bf16");
  TORCH_CHECK(x.numel() % 8 == 0 && x.numel() > 0, "numel % 8 required");
  TORCH_CHECK(data.is_cuda() && data.scalar_type() == at::kByte && data.is_contiguous() && data.numel() == x.numel(),
              "data: contiguous uint8, numel as x");
  TORCH_CHECK(scale_inv.is_cuda() && scale_inv.scalar_type() == at::kFloat && scale_inv.numel() >= 1, "scale_inv");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(data.data_ptr()) % 8 == 0,
              "alignment");
}

template <bool kE5M2, int U, bool kRev, bool kFromPart>
void launch_cast_t(const at::Tensor& x, const float* part, int n_part, at::Tensor& ws, at::Tensor& scale_inv,
                   float fp8_max, bool pow2, float eps, at::Tensor& data, int grid, cudaStream_t stream) {
  cast_kernel<kE5M2, U, kRev, kFromPart><<<grid, kCastThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(x.data_ptr()), x.numel() / 8, part, n_part, ws.data_ptr<float>(),
      scale_inv.data_ptr<float>(), fp8_max, pow2 ? 1 : 0, eps, reinterpret_cast<uint2*>(data.data_ptr()));
}

template <bool kFromPart>
void launch_cast(const at::Tensor& x, const float* part, int n_part, at::Tensor& ws, at::Tensor& scale_inv,
                 float fp8_max, bool e5m2, bool pow2, float eps, at::Tensor& data, int u, bool rev, int grid,
                 cudaStream_t stream) {
#define AK_CAST(E, UU, R) launch_cast_t<E, UU, R, kFromPart>(x, part, n_part, ws, scale_inv, fp8_max, pow2, eps, data, grid, stream)
  if (e5m2) {
    if (u == 4) { if (rev) AK_CAST(true, 4, true); else AK_CAST(true, 4, false); }
    else if (u == 2) { if (rev) AK_CAST(true, 2, true); else AK_CAST(true, 2, false); }
    else { if (rev) AK_CAST(true, 1, true); else AK_CAST(true, 1, false); }
  } else {
    if (u == 4) { if (rev) AK_CAST(false, 4, true); else AK_CAST(false, 4, false); }
    else if (u == 2) { if (rev) AK_CAST(false, 2, true); else AK_CAST(false, 2, false); }
    else { if (rev) AK_CAST(false, 1, true); else AK_CAST(false, 1, false); }
  }
#undef AK_CAST
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

}  // namespace

// amax_cast(x, data u8, scale_inv f32 [1], fp8_max, e5m2, pow2, eps, u_amax, rev_amax, amax_blocks_per_sm, u_cast,
//           rev_cast, cast_blocks_per_sm) -> ws [3] (scale, scale_inv, amax)
at::Tensor amax_cast(const at::Tensor& x, at::Tensor data, at::Tensor scale_inv, double fp8_max, bool e5m2, bool pow2,
                     double eps, int64_t u_amax, bool rev_amax, int64_t amax_bps, int64_t u_cast, bool rev_cast,
                     int64_t cast_bps) {
  check_io(x, data, scale_inv);
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n_vec = x.numel() / 8;
  const int64_t steps_a = (n_vec + (int64_t)kAmaxThreads * u_amax - 1) / ((int64_t)kAmaxThreads * u_amax);
  const int grid_a = (int)std::max<int64_t>(1, std::min<int64_t>(steps_a, (int64_t)sms() * amax_bps));
  auto part = at::empty({grid_a}, x.options().dtype(at::kFloat));
  const uint4* xv = reinterpret_cast<const uint4*>(x.data_ptr());
#define AK_AMAX(UU, R) amax_part_kernel<UU, R><<<grid_a, kAmaxThreads, 0, stream>>>(xv, n_vec, part.data_ptr<float>())
  if (u_amax == 4) { if (rev_amax) AK_AMAX(4, true); else AK_AMAX(4, false); }
  else if (u_amax == 2) { if (rev_amax) AK_AMAX(2, true); else AK_AMAX(2, false); }
  else { if (rev_amax) AK_AMAX(1, true); else AK_AMAX(1, false); }
#undef AK_AMAX
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto ws = at::empty({3}, x.options().dtype(at::kFloat));
  const int64_t steps_c = (n_vec + (int64_t)kCastThreads * u_cast - 1) / ((int64_t)kCastThreads * u_cast);
  const int grid_c = (int)std::max<int64_t>(1, std::min<int64_t>(steps_c, (int64_t)sms() * cast_bps));
  launch_cast<true>(x, part.data_ptr<float>(), grid_a, ws, scale_inv, (float)fp8_max, e5m2, pow2, (float)eps, data,
                    (int)u_cast, rev_cast, grid_c, stream);
  return ws;
}

// amax_partials(x, part f32, u, rev, blocks_per_sm) -> the number of partial maxima written (the amax pass alone)
int64_t amax_partials(const at::Tensor& x, at::Tensor part, int64_t u_amax, bool rev_amax, int64_t amax_bps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.numel() % 8 == 0, "x");
  TORCH_CHECK(part.is_cuda() && part.scalar_type() == at::kFloat && part.is_contiguous(), "part");
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n_vec = x.numel() / 8;
  const int64_t steps_a = (n_vec + (int64_t)kAmaxThreads * u_amax - 1) / ((int64_t)kAmaxThreads * u_amax);
  const int grid_a = (int)std::max<int64_t>(1, std::min<int64_t>(steps_a, (int64_t)sms() * amax_bps));
  TORCH_CHECK(part.numel() >= grid_a, "part too small");
  const uint4* xv = reinterpret_cast<const uint4*>(x.data_ptr());
#define AK_AMAX(UU, R) amax_part_kernel<UU, R><<<grid_a, kAmaxThreads, 0, stream>>>(xv, n_vec, part.data_ptr<float>())
  if (u_amax == 4) { if (rev_amax) AK_AMAX(4, true); else AK_AMAX(4, false); }
  else if (u_amax == 2) { if (rev_amax) AK_AMAX(2, true); else AK_AMAX(2, false); }
  else { if (rev_amax) AK_AMAX(1, true); else AK_AMAX(1, false); }
#undef AK_AMAX
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grid_a;
}

// cast_from_ws(x, ws f32 [3] (ws[0] the scale), data u8, e5m2, u, rev, blocks_per_sm): the cast alone
void cast_from_ws(const at::Tensor& x, at::Tensor ws, at::Tensor data, bool e5m2, int64_t u, bool rev, int64_t bps) {
  TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == at::kFloat && ws.numel() >= 1, "ws");
  check_io(x, data, ws);
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n_vec = x.numel() / 8;
  const int64_t steps_c = (n_vec + (int64_t)kCastThreads * u - 1) / ((int64_t)kCastThreads * u);
  const int grid_c = (int)std::max<int64_t>(1, std::min<int64_t>(steps_c, (int64_t)sms() * bps));
  launch_cast<false>(x, nullptr, 0, ws, ws, 0.f, e5m2, false, 0.f, data, (int)u, rev, grid_c, stream);
}

// scale_from_partials(part f32 [P], ws f32 [3], scale_inv f32 [1], fp8_max, pow2, eps)
void scale_from_partials(const at::Tensor& part, at::Tensor ws, at::Tensor scale_inv, double fp8_max, bool pow2,
                         double eps) {
  TORCH_CHECK(part.is_cuda() && part.scalar_type() == at::kFloat && part.is_contiguous(), "part: contiguous f32");
  TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == at::kFloat && ws.numel() >= 3, "ws: f32 [3]");
  TORCH_CHECK(scale_inv.is_cuda() && scale_inv.scalar_type() == at::kFloat && scale_inv.numel() >= 1, "scale_inv");
  const at::cuda::CUDAGuard guard(part.device());
  scale_kernel<<<1, kRedThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      part.data_ptr<float>(), part.numel(), ws.data_ptr<float>(), scale_inv.data_ptr<float>(), (float)fp8_max,
      pow2 ? 1 : 0, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scale_from_partials", &scale_from_partials, "amax of partial maxima -> scale, scale_inv (one CTA)",
        py::arg("part"), py::arg("ws"), py::arg("scale_inv"), py::arg("fp8_max"), py::arg("pow2"), py::arg("eps"));
  m.def("amax_cast", &amax_cast, "FP8 current-scaling quantize: partial-amax pass + cast reducing the partials",
        py::arg("x"), py::arg("data"), py::arg("scale_inv"), py::arg("fp8_max"), py::arg("e5m2"), py::arg("pow2"),
        py::arg("eps"), py::arg("u_amax") = 4, py::arg("rev_amax") = false, py::arg("amax_bps") = 4,
        py::arg("u_cast") = 2, py::arg("rev_cast") = false, py::arg("cast_bps") = 8);
  m.def("amax_partials", &amax_partials, "the partial-amax pass alone", py::arg("x"), py::arg("part"),
        py::arg("u_amax") = 4, py::arg("rev_amax") = false, py::arg("amax_bps") = 4);
  m.def("cast_from_ws", &cast_from_ws, "FP8 cast with the scale in ws[0]", py::arg("x"), py::arg("ws"),
        py::arg("data"), py::arg("e5m2"), py::arg("u") = 2, py::arg("rev") = false, py::arg("bps") = 8);
}
