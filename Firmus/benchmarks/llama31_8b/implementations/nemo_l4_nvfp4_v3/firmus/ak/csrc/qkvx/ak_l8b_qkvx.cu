// ak_l8b_qkvx.cu (Llama 3.1 8B kernel campaign, 2026-10-07): the X12 fused QKV RoPE's passes in CUDA, each byte-exact
// against the Triton kernels of the x12 overlays (fused_qkv_rope.py, fused_qkv_rope_fp8.py) and TE:
//  * rope_bwd_amax: the RoPE backward (_qkv_rope_bwd_kernel's arithmetic: y_lo = fma(g_lo, cos_lo, g_hi * sin_hi),
//    y_hi = fma(g_hi, cos_hi, (-g_lo) * sin_lo), fp32, round to nearest bf16; value heads copied) writing dqkv
//    [s, b, h, (nq + 2) * 128], and in the same pass the two amaxes TE's NVFP4 gradient quantizer (RHT, post-RHT amax)
//    takes of it: the pre-RHT amax of the data and the post-RHT amax of its columnwise copy (16-token blocks), with the
//    epilogue of ak_l8b_rht_epi.cuh (TE's own amax instructions, as ak_l8b_nvfp4b.rht_amax).  Tiles of 32 tokens x 256
//    features (two heads), the outputs staged in shared memory for the RHT.
//  * rope_fwd_partials: QKV direct's pass 1 -- per (s, b) position, max |bf16(RoPE(x))| over its rotated q, k heads and
//    its v heads (the partial maxima of _qkv_rope_fwd_amax_kernel / _qkv_rope_amax_only_kernel), no stores but the
//    partials.  One CTA per s; each thread keeps its 8 features' angles in registers for all heads of the position(s).
//  * rope_fwd_fp8: QKV direct's pass 2 -- fp8(float(bf16(RoPE(x))) * scale), round to nearest even, saturating (TE's
//    cast_fp8_1D of the bf16 buffer), at each element's place in the packed [q | k | v] FP8 buffer.
// The forward arithmetic: y_lo = fma(x_lo, cos_lo, (-x_hi) * sin_lo), y_hi = fma(x_hi, cos_hi, x_lo * sin_hi).  The angle
// table is the Triton table kernel's (fused_qkv_rope._rope_table_kernel: [s, 4 * 128] fp32 -- cos and sin of the q
// angles, then of the k angles when they differ).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "ak_l8b_rht_epi.cuh"

namespace {

constexpr int kD = 128, kHalf = 64;

union V8 {
  uint4 u;
  uint32_t w[4];
  __nv_bfloat16 h[8];
};

__device__ __forceinline__ void ld8f(float (&o)[8], const float* p) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(p));
  const float4 b = __ldg(reinterpret_cast<const float4*>(p) + 1);
  o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
  o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}

struct Ang {             // one thread's 8 features of one angle section: cos / sin of the low and the high half
  float c_lo[8], c_hi[8], t_lo[8], t_hi[8];
  __device__ __forceinline__ void load(const float* sec, int d8) {   // sec: cos at [0, 128), sin at [128, 256)
    ld8f(c_lo, sec + d8 * 8);
    ld8f(c_hi, sec + kHalf + d8 * 8);
    ld8f(t_lo, sec + kD + d8 * 8);
    ld8f(t_hi, sec + kD + kHalf + d8 * 8);
  }
};

__device__ __forceinline__ uint32_t bmax4(const V8& v) {
  return ak_rht::bmax_xa(ak_rht::bmax_xa(v.w[0], v.w[1]), ak_rht::bmax_xa(v.w[2], v.w[3]));
}

// ------------------------------------------------------------------------------------------------ the RoPE backward
// y_lo = fma(g_lo, cos_lo, g_hi * sin_hi); y_hi = fma(g_hi, cos_hi, (-g_lo) * sin_lo)
__device__ __forceinline__ void rope_bwd8(const V8& lo, const V8& hi, const Ang& a, V8& olo, V8& ohi) {
  float ylo[8], yhi[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float g0 = __bfloat162float(lo.h[e]), g1 = __bfloat162float(hi.h[e]);
    ylo[e] = __fmaf_rn(g0, a.c_lo[e], __fmul_rn(g1, a.t_hi[e]));
    yhi[e] = __fmaf_rn(g1, a.c_hi[e], __fmul_rn(-g0, a.t_lo[e]));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    olo.w[i] = ak_rht::cvt_bf16x2(ylo[2 * i], ylo[2 * i + 1]);
    ohi.w[i] = ak_rht::cvt_bf16x2(yhi[2 * i], yhi[2 * i + 1]);
  }
}

// y_lo = fma(x_lo, cos_lo, (-x_hi) * sin_lo); y_hi = fma(x_hi, cos_hi, x_lo * sin_hi), rounded to bf16 (packed)
__device__ __forceinline__ void rope_fwd8(const V8& lo, const V8& hi, const Ang& a, V8& olo, V8& ohi) {
  float ylo[8], yhi[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float x0 = __bfloat162float(lo.h[e]), x1 = __bfloat162float(hi.h[e]);
    ylo[e] = __fmaf_rn(x0, a.c_lo[e], __fmul_rn(-x1, a.t_lo[e]));
    yhi[e] = __fmaf_rn(x1, a.c_hi[e], __fmul_rn(x0, a.t_hi[e]));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    olo.w[i] = ak_rht::cvt_bf16x2(ylo[2 * i], ylo[2 * i + 1]);
    ohi.w[i] = ak_rht::cvt_bf16x2(yhi[2 * i], yhi[2 * i + 1]);
  }
}

constexpr int kTR = 32, kTC = 256, kLD = kTC + 8, kThr = 256;

// grid (heads per row / 2, rows / 32): 32 tokens x two heads (256 features) per CTA, 2 units of 8 + 8 features each
// per thread.  Heads per group nq + 2 even, so a pair never straddles a group.
template <bool kSame>
__global__ void __launch_bounds__(kThr)
    rope_bwd_amax_kernel(const __nv_bfloat16* __restrict__ dq, const __nv_bfloat16* __restrict__ dk,
                         const __nv_bfloat16* __restrict__ dv, const float* __restrict__ tab,
                         __nv_bfloat16* __restrict__ dqkv, float* __restrict__ amax, int b, int H, int NQ,
                         uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh[kTR * kLD];
  const int HH = H * (NQ + 2);                  // heads per token
  const int64_t ncol = (int64_t)HH * kD;
  const int row0 = blockIdx.y * kTR;
  uint32_t pre = 0u;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int u = threadIdx.x + i * kThr;
    const int d8 = u & 7, s2 = (u >> 3) & 1, rr = u >> 4;
    const int r = row0 + rr;
    const int hh = blockIdx.x * 2 + s2, g = hh / (NQ + 2), j = hh - g * (NQ + 2);
    const __nv_bfloat16* src = j < NQ ? dq + ((int64_t)r * (H * NQ) + g * NQ + j) * kD
                                      : (j == NQ ? dk : dv) + ((int64_t)r * H + g) * kD;
    V8 lo, hi, olo, ohi;
    lo.u = __ldcs(reinterpret_cast<const uint4*>(src + d8 * 8));
    hi.u = __ldcs(reinterpret_cast<const uint4*>(src + kHalf + d8 * 8));
    if (j <= NQ) {
      Ang a;
      a.load(tab + (int64_t)(r / b) * (4 * kD) + ((!kSame && j == NQ) ? 2 * kD : 0), d8);
      rope_bwd8(lo, hi, a, olo, ohi);
    } else {
      olo = lo;
      ohi = hi;
    }
    __nv_bfloat16* dst = dqkv + (int64_t)r * ncol + (int64_t)hh * kD;
    *reinterpret_cast<uint4*>(dst + d8 * 8) = olo.u;
    *reinterpret_cast<uint4*>(dst + kHalf + d8 * 8) = ohi.u;
    *reinterpret_cast<uint4*>(sh + rr * kLD + s2 * kD + d8 * 8) = olo.u;
    *reinterpret_cast<uint4*>(sh + rr * kLD + s2 * kD + kHalf + d8 * 8) = ohi.u;
    pre = ak_rht::bmax_xa(pre, ak_rht::bmax_xa(bmax4(olo), bmax4(ohi)));
  }
  const ak_rht::RhtB B(sign_mask);
  __syncthreads();
  uint32_t post = 0u;
  constexpr int kSub = (kTR / 16) * (kTC / 16);   // 32 sub-tiles of 16 tokens x 16 features
#pragma unroll
  for (int sidx = threadIdx.x / 32; sidx < kSub; sidx += kThr / 32) {
    const int tb = sidx / (kTC / 16), fb = sidx % (kTC / 16);
    post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kLD>(sh + tb * 16 * kLD + fb * 16, B));
  }
  ak_rht::block_publish2<kThr / 32>(ak_rht::unpack_bmax(pre), ak_rht::unpack_bmax(post), amax, amax + 1);
}

// ------------------------------------------------------------------------------------------------ QKV direct passes
// One CTA per s (its b positions), 256 threads: d8 = feature octet (0..7), lane = head lane (0..31); head-rows
// hr = lane, lane + 32, ... over the b * HH head-rows of the s.  The thread's angles (q, and k when they differ) are
// loaded once.  kRev: the CTAs walk s from the end.
constexpr int kPThr = 256, kMaxB = 4;

template <bool kSame>
__device__ __forceinline__ void load_angles(Ang& aq, Ang& ak, const float* tab, int s, int d8) {
  const float* row = tab + (int64_t)s * (4 * kD);
  aq.load(row, d8);
  if (!kSame) ak.load(row + 2 * kD, d8);
}

template <bool kSame, bool kRev>
__global__ void __launch_bounds__(kPThr)
    rope_fwd_partials_kernel(const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ tab,
                             float* __restrict__ part, int S, int b, int H, int NQ) {
  const int s = kRev ? S - 1 - (int)blockIdx.x : (int)blockIdx.x;
  const int d8 = threadIdx.x & 7, lane = threadIdx.x >> 3;
  const int HH = H * (NQ + 2), n_hr = b * HH;
  Ang aq, ak;
  load_angles<kSame>(aq, ak, tab, s, d8);
  uint32_t acc[kMaxB];
#pragma unroll
  for (int p = 0; p < kMaxB; ++p) acc[p] = 0u;
  const __nv_bfloat16* base = qkv + (int64_t)s * b * HH * kD;
  for (int hr = lane; hr < n_hr; hr += kPThr / 8) {
    const int p = hr / HH, hh = hr - p * HH, j = hh % (NQ + 2);
    const __nv_bfloat16* src = base + (int64_t)hr * kD;
    V8 lo, hi;
    lo.u = __ldg(reinterpret_cast<const uint4*>(src + d8 * 8));
    hi.u = __ldg(reinterpret_cast<const uint4*>(src + kHalf + d8 * 8));
    uint32_t m;
    if (j <= NQ) {
      V8 olo, ohi;
      rope_fwd8(lo, hi, (kSame || j < NQ) ? aq : ak, olo, ohi);
      m = ak_rht::bmax_xa(bmax4(olo), bmax4(ohi));
    } else {
      m = ak_rht::bmax_xa(bmax4(lo), bmax4(hi));
    }
#pragma unroll
    for (int q = 0; q < kMaxB; ++q)
      if (q == p) acc[q] = ak_rht::bmax_xa(acc[q], m);
  }
  __shared__ uint32_t red[kMaxB][kPThr / 32];
  const int warp = threadIdx.x / 32, wl = threadIdx.x % 32;
#pragma unroll
  for (int p = 0; p < kMaxB; ++p) {
    uint32_t v = acc[p];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = ak_rht::bmax_xa(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (wl == 0) red[p][warp] = v;
  }
  __syncthreads();
  if (threadIdx.x < b) {
    uint32_t v = 0u;
#pragma unroll
    for (int w = 0; w < kPThr / 32; ++w) v = ak_rht::bmax_xa(v, red[threadIdx.x][w]);
    part[(int64_t)s * b + threadIdx.x] = ak_rht::unpack_bmax(v);
  }
}

__device__ __forceinline__ uint2 fp8x8(const V8& v, float scale) {   // fp8(float(bf16) * scale), E4M3, satfinite
  uint32_t w[2];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    uint32_t word = 0;
#pragma unroll
    for (int e = 0; e < 4; e += 2) {
      const float f0 = __bfloat162float(v.h[4 * p + e]) * scale;
      const float f1 = __bfloat162float(v.h[4 * p + e + 1]) * scale;
      const __nv_fp8x2_storage_t pr = __nv_cvt_float2_to_fp8x2(make_float2(f0, f1), __NV_SATFINITE, __NV_E4M3);
      word |= (uint32_t)pr << (8 * e);
    }
    w[p] = word;
  }
  return make_uint2(w[0], w[1]);
}

template <bool kSame, bool kRev>
__global__ void __launch_bounds__(kPThr)
    rope_fwd_fp8_kernel(const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ tab,
                        const float* __restrict__ ws, uint8_t* __restrict__ out, int S, int b, int H, int NQ) {
  const int s = kRev ? S - 1 - (int)blockIdx.x : (int)blockIdx.x;
  const int d8 = threadIdx.x & 7, lane = threadIdx.x >> 3;
  const int HH = H * (NQ + 2), n_hr = b * HH;
  const int64_t nqe = (int64_t)S * b * H * NQ * kD, nke = (int64_t)S * b * H * kD;
  const float scale = ws[0];
  Ang aq, ak;
  load_angles<kSame>(aq, ak, tab, s, d8);
  const __nv_bfloat16* base = qkv + (int64_t)s * b * HH * kD;
  for (int hr = lane; hr < n_hr; hr += kPThr / 8) {
    const int p = hr / HH, hh = hr - p * HH, g = hh / (NQ + 2), j = hh - g * (NQ + 2);
    const int64_t r = (int64_t)s * b + p;
    const __nv_bfloat16* src = base + (int64_t)hr * kD;
    V8 lo, hi;
    lo.u = __ldcs(reinterpret_cast<const uint4*>(src + d8 * 8));
    hi.u = __ldcs(reinterpret_cast<const uint4*>(src + kHalf + d8 * 8));
    uint8_t* dst;
    if (j < NQ) {
      dst = out + (r * (H * NQ) + g * NQ + j) * kD;
    } else if (j == NQ) {
      dst = out + nqe + (r * H + g) * kD;
    } else {
      dst = out + nqe + nke + (r * H + g) * kD;
    }
    if (j <= NQ) {
      V8 olo, ohi;
      rope_fwd8(lo, hi, (kSame || j < NQ) ? aq : ak, olo, ohi);
      lo = olo;
      hi = ohi;
    }
    *reinterpret_cast<uint2*>(dst + d8 * 8) = fp8x8(lo, scale);
    *reinterpret_cast<uint2*>(dst + kHalf + d8 * 8) = fp8x8(hi, scale);
  }
}

// ------------------------------------------------------------------------------------------------ version 2
// The same arithmetic, stores and amaxes with every load of a thread issued before its arithmetic (version 1 left one
// head in flight per thread: ~4 TB/s).
//  * the backward: tiles of 32 tokens x 4 heads (512 features), 256 threads; thread (token rr = t >> 3, octet
//    d8 = t & 7) does the 4 heads of its token -- 8 16-byte loads in flight, one angle slice;
//  * passes 1 and 2: one CTA of 128 threads per s; thread (lane = t >> 3, d8 = t & 7) takes head-rows lane + 16 u,
//    u < U (U = 6 for b 2 x 48 heads) -- 2 U 16-byte loads in flight; the rotation computed for every head-row and the
//    value heads' raw data selected (no divergence).
constexpr int kHB = 4, kTC2 = kHB * kD, kLD2 = kTC2 + 8;

template <bool kSame>
__global__ void __launch_bounds__(kThr)
    rope_bwd_amax_kernel_v2(const __nv_bfloat16* __restrict__ dq, const __nv_bfloat16* __restrict__ dk,
                            const __nv_bfloat16* __restrict__ dv, const float* __restrict__ tab,
                            __nv_bfloat16* __restrict__ dqkv, float* __restrict__ amax, int b, int H, int NQ,
                            uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh[kTR * kLD2];
  const int HH = H * (NQ + 2);
  const int64_t ncol = (int64_t)HH * kD;
  const int d8 = threadIdx.x & 7, rr = threadIdx.x >> 3;
  const int r = blockIdx.y * kTR + rr;
  const int hh0 = blockIdx.x * kHB;
  V8 lo[kHB], hi[kHB];
  int jj[kHB];
#pragma unroll
  for (int u = 0; u < kHB; ++u) {
    const int hh = hh0 + u, g = hh / (NQ + 2), j = hh - g * (NQ + 2);
    jj[u] = j;
    const __nv_bfloat16* src = j < NQ ? dq + ((int64_t)r * (H * NQ) + g * NQ + j) * kD
                                      : (j == NQ ? dk : dv) + ((int64_t)r * H + g) * kD;
    lo[u].u = __ldcs(reinterpret_cast<const uint4*>(src + d8 * 8));
    hi[u].u = __ldcs(reinterpret_cast<const uint4*>(src + kHalf + d8 * 8));
  }
  const float* trow = tab + (int64_t)(r / b) * (4 * kD);
  Ang aq, ak;
  aq.load(trow, d8);
  if (!kSame) ak.load(trow + 2 * kD, d8);
  uint32_t pre = 0u;
  __nv_bfloat16* drow = dqkv + (int64_t)r * ncol;
#pragma unroll
  for (int u = 0; u < kHB; ++u) {
    V8 olo, ohi;
    if (jj[u] <= NQ) {
      rope_bwd8(lo[u], hi[u], (kSame || jj[u] < NQ) ? aq : ak, olo, ohi);
    } else {
      olo = lo[u];
      ohi = hi[u];
    }
    __nv_bfloat16* dst = drow + (int64_t)(hh0 + u) * kD;
    *reinterpret_cast<uint4*>(dst + d8 * 8) = olo.u;
    *reinterpret_cast<uint4*>(dst + kHalf + d8 * 8) = ohi.u;
    *reinterpret_cast<uint4*>(sh + rr * kLD2 + u * kD + d8 * 8) = olo.u;
    *reinterpret_cast<uint4*>(sh + rr * kLD2 + u * kD + kHalf + d8 * 8) = ohi.u;
    pre = ak_rht::bmax_xa(pre, ak_rht::bmax_xa(bmax4(olo), bmax4(ohi)));
  }
  const ak_rht::RhtB B(sign_mask);
  __syncthreads();
  uint32_t post = 0u;
  constexpr int kSub = (kTR / 16) * (kTC2 / 16);   // 64 sub-tiles of 16 tokens x 16 features
#pragma unroll
  for (int sidx = threadIdx.x / 32; sidx < kSub; sidx += kThr / 32) {
    const int tb = sidx / (kTC2 / 16), fb = sidx % (kTC2 / 16);
    post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kLD2>(sh + tb * 16 * kLD2 + fb * 16, B));
  }
  ak_rht::block_publish2<kThr / 32>(ak_rht::unpack_bmax(pre), ak_rht::unpack_bmax(post), amax, amax + 1);
}

constexpr int kP2Thr = 128, kLanes = kP2Thr / 8;

// the head-rows of one s, loaded: lo / hi of head-row lane + 16 u (zero past the end)
template <int kU>
__device__ __forceinline__ void load_rows(V8 (&lo)[kU], V8 (&hi)[kU], const __nv_bfloat16* base, int lane, int d8,
                                          int n_hr, bool stream) {
#pragma unroll
  for (int u = 0; u < kU; ++u) {
    const int hr = lane + kLanes * u;
    if (hr < n_hr) {
      const uint4* p0 = reinterpret_cast<const uint4*>(base + (int64_t)hr * kD + d8 * 8);
      const uint4* p1 = reinterpret_cast<const uint4*>(base + (int64_t)hr * kD + kHalf + d8 * 8);
      lo[u].u = stream ? __ldcs(p0) : __ldg(p0);
      hi[u].u = stream ? __ldcs(p1) : __ldg(p1);
    } else {
      lo[u].u = make_uint4(0u, 0u, 0u, 0u);
      hi[u].u = lo[u].u;
    }
  }
}

template <bool kSame, bool kRev, int kU>
__global__ void __launch_bounds__(kP2Thr)
    rope_fwd_partials_kernel_v2(const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ tab,
                                float* __restrict__ part, int S, int b, int H, int NQ) {
  const int s = kRev ? S - 1 - (int)blockIdx.x : (int)blockIdx.x;
  const int d8 = threadIdx.x & 7, lane = threadIdx.x >> 3;
  const int HH = H * (NQ + 2), n_hr = b * HH;
  V8 lo[kU], hi[kU];
  load_rows<kU>(lo, hi, qkv + (int64_t)s * b * HH * kD, lane, d8, n_hr, false);
  Ang aq, ak;
  load_angles<kSame>(aq, ak, tab, s, d8);
  uint32_t acc[kMaxB];
#pragma unroll
  for (int p = 0; p < kMaxB; ++p) acc[p] = 0u;
#pragma unroll
  for (int u = 0; u < kU; ++u) {
    const int hr = lane + kLanes * u;
    const int p = hr / HH, j = (hr - p * HH) % (NQ + 2);
    V8 olo, ohi;
    rope_fwd8(lo[u], hi[u], (kSame || j < NQ) ? aq : ak, olo, ohi);
    const uint32_t m = j <= NQ ? ak_rht::bmax_xa(bmax4(olo), bmax4(ohi)) : ak_rht::bmax_xa(bmax4(lo[u]), bmax4(hi[u]));
    if (hr < n_hr) {
#pragma unroll
      for (int q = 0; q < kMaxB; ++q)
        if (q == p) acc[q] = ak_rht::bmax_xa(acc[q], m);
    }
  }
  __shared__ uint32_t red[kMaxB][kP2Thr / 32];
  const int warp = threadIdx.x / 32, wl = threadIdx.x % 32;
#pragma unroll
  for (int p = 0; p < kMaxB; ++p) {
    uint32_t v = acc[p];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = ak_rht::bmax_xa(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (wl == 0) red[p][warp] = v;
  }
  __syncthreads();
  if (threadIdx.x < b) {
    uint32_t v = 0u;
#pragma unroll
    for (int w = 0; w < kP2Thr / 32; ++w) v = ak_rht::bmax_xa(v, red[threadIdx.x][w]);
    part[(int64_t)s * b + threadIdx.x] = ak_rht::unpack_bmax(v);
  }
}

template <bool kSame, bool kRev, int kU>
__global__ void __launch_bounds__(kP2Thr)
    rope_fwd_fp8_kernel_v2(const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ tab,
                           const float* __restrict__ ws, uint8_t* __restrict__ out, int S, int b, int H, int NQ) {
  const int s = kRev ? S - 1 - (int)blockIdx.x : (int)blockIdx.x;
  const int d8 = threadIdx.x & 7, lane = threadIdx.x >> 3;
  const int HH = H * (NQ + 2), n_hr = b * HH;
  const int64_t nqe = (int64_t)S * b * H * NQ * kD, nke = (int64_t)S * b * H * kD;
  V8 lo[kU], hi[kU];
  load_rows<kU>(lo, hi, qkv + (int64_t)s * b * HH * kD, lane, d8, n_hr, true);
  Ang aq, ak;
  load_angles<kSame>(aq, ak, tab, s, d8);
  const float scale = ws[0];
#pragma unroll
  for (int u = 0; u < kU; ++u) {
    const int hr = lane + kLanes * u;
    if (hr >= n_hr) continue;
    const int p = hr / HH, hh = hr - p * HH, g = hh / (NQ + 2), j = hh - g * (NQ + 2);
    const int64_t r = (int64_t)s * b + p;
    uint8_t* dst = j < NQ ? out + (r * (H * NQ) + g * NQ + j) * kD
                          : out + nqe + (j == NQ ? 0 : nke) + (r * H + g) * kD;
    V8 olo, ohi;
    rope_fwd8(lo[u], hi[u], (kSame || j < NQ) ? aq : ak, olo, ohi);
    if (j > NQ) {
      olo = lo[u];
      ohi = hi[u];
    }
    *reinterpret_cast<uint2*>(dst + d8 * 8) = fp8x8(olo, scale);
    *reinterpret_cast<uint2*>(dst + kHalf + d8 * 8) = fp8x8(ohi, scale);
  }
}

// ------------------------------------------------------------------------------------------------ version 3
// Version 2 with the Llama shape compile-time (h 8 groups, nq 4, b 2: every index a constant expression) and the CTA
// shapes as template parameters, to find the bandwidth:
//  * the backward: kRows tokens x kHB heads per CTA, kRows * 8 threads (thread: one token, one octet, kHB heads; every
//    load first), __launch_bounds__(threads, kMinB);
//  * passes 1 / 2: one s per CTA, kT threads (kT / 8 lanes, U = b * heads / lanes head-rows each).
// Variants -- backward 30-39: (tokens, heads, min CTAs) = (16, 4, 4), (16, 8, 3), (16, 8, 4), (32, 4, 2), (16, 4, 6),
// (16, 6, 4), (16, 4, 8), (16, 3, 6), (16, 2, 8), (16, 4, 5); passes 30-38: (threads, min CTAs) = (128, 4), (128, 5),
// (256, 2), (256, 3), (64, 8), (128, 3), (192, 3), (96, 5), (192, 2).
template <bool kSame, int kH, int kNQ, int kB, int kRows, int kHB, int kMinB>
__global__ void __launch_bounds__(kRows * 8, kMinB)
    rope_bwd_amax_kernel_v3(const __nv_bfloat16* __restrict__ dq, const __nv_bfloat16* __restrict__ dk,
                            const __nv_bfloat16* __restrict__ dv, const float* __restrict__ tab,
                            __nv_bfloat16* __restrict__ dqkv, float* __restrict__ amax, uint32_t sign_mask) {
  constexpr int kHH = kH * (kNQ + 2), kTCv = kHB * kD, kLDv = kTCv + 8, kT = kRows * 8;
  constexpr int64_t kNcol = (int64_t)kHH * kD;
  __shared__ __align__(128) __nv_bfloat16 sh[kRows * kLDv];
  const int d8 = threadIdx.x & 7, rr = threadIdx.x >> 3;
  const int r = blockIdx.y * kRows + rr;
  const int hh0 = blockIdx.x * kHB;
  V8 lo[kHB], hi[kHB];
#pragma unroll
  for (int u = 0; u < kHB; ++u) {
    const int hh = hh0 + u, g = hh / (kNQ + 2), j = hh - g * (kNQ + 2);
    const __nv_bfloat16* src = j < kNQ ? dq + ((int64_t)r * (kH * kNQ) + g * kNQ + j) * kD
                                       : (j == kNQ ? dk : dv) + ((int64_t)r * kH + g) * kD;
    lo[u].u = __ldcs(reinterpret_cast<const uint4*>(src + d8 * 8));
    hi[u].u = __ldcs(reinterpret_cast<const uint4*>(src + kHalf + d8 * 8));
  }
  const float* trow = tab + (int64_t)(r / kB) * (4 * kD);
  Ang aq, ak;
  aq.load(trow, d8);
  if (!kSame) ak.load(trow + 2 * kD, d8);
  uint32_t pre = 0u;
  __nv_bfloat16* drow = dqkv + (int64_t)r * kNcol;
#pragma unroll
  for (int u = 0; u < kHB; ++u) {
    const int hh = hh0 + u, j = hh % (kNQ + 2);
    V8 olo, ohi;
    if (j <= kNQ) {
      rope_bwd8(lo[u], hi[u], (kSame || j < kNQ) ? aq : ak, olo, ohi);
    } else {
      olo = lo[u];
      ohi = hi[u];
    }
    __nv_bfloat16* dst = drow + (int64_t)hh * kD;
    *reinterpret_cast<uint4*>(dst + d8 * 8) = olo.u;
    *reinterpret_cast<uint4*>(dst + kHalf + d8 * 8) = ohi.u;
    *reinterpret_cast<uint4*>(sh + rr * kLDv + u * kD + d8 * 8) = olo.u;
    *reinterpret_cast<uint4*>(sh + rr * kLDv + u * kD + kHalf + d8 * 8) = ohi.u;
    pre = ak_rht::bmax_xa(pre, ak_rht::bmax_xa(bmax4(olo), bmax4(ohi)));
  }
  const ak_rht::RhtB B(sign_mask);
  __syncthreads();
  uint32_t post = 0u;
  constexpr int kSub = (kRows / 16) * (kTCv / 16);
#pragma unroll
  for (int sidx = threadIdx.x / 32; sidx < kSub; sidx += kT / 32) {
    const int tb = sidx / (kTCv / 16), fb = sidx % (kTCv / 16);
    post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kLDv>(sh + tb * 16 * kLDv + fb * 16, B));
  }
  ak_rht::block_publish2<kT / 32>(ak_rht::unpack_bmax(pre), ak_rht::unpack_bmax(post), amax, amax + 1);
}

template <bool kSame, bool kRev, int kH, int kNQ, int kB, int kT, int kMinB, bool kFp8>
__global__ void __launch_bounds__(kT, kMinB)
    rope_fwd_pass_kernel_v3(const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ tab,
                            float* __restrict__ part, const float* __restrict__ ws, uint8_t* __restrict__ out, int S) {
  constexpr int kHH = kH * (kNQ + 2), kNhr = kB * kHH, kL = kT / 8, kU = kNhr / kL;
  static_assert(kNhr % kL == 0, "head-rows per lane");
  const int s = kRev ? S - 1 - (int)blockIdx.x : (int)blockIdx.x;
  const int d8 = threadIdx.x & 7, lane = threadIdx.x >> 3;
  const __nv_bfloat16* base = qkv + (int64_t)s * kNhr * kD;
  V8 lo[kU], hi[kU];
#pragma unroll
  for (int u = 0; u < kU; ++u) {
    const uint4* p0 = reinterpret_cast<const uint4*>(base + (int64_t)(lane + kL * u) * kD + d8 * 8);
    lo[u].u = kFp8 ? __ldcs(p0) : __ldg(p0);
    hi[u].u = kFp8 ? __ldcs(p0 + kHalf / 8) : __ldg(p0 + kHalf / 8);
  }
  Ang aq, ak;
  load_angles<kSame>(aq, ak, tab, s, d8);
  if (kFp8) {
    const int64_t nqe = (int64_t)S * kB * kH * kNQ * kD, nke = (int64_t)S * kB * kH * kD;
    const float scale = ws[0];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int hr = lane + kL * u, p = hr / kHH, hh = hr - p * kHH, g = hh / (kNQ + 2), j = hh - g * (kNQ + 2);
      const int64_t r = (int64_t)s * kB + p;
      uint8_t* dst = j < kNQ ? out + (r * (kH * kNQ) + g * kNQ + j) * kD
                             : out + nqe + (j == kNQ ? 0 : nke) + (r * kH + g) * kD;
      V8 olo, ohi;
      rope_fwd8(lo[u], hi[u], (kSame || j < kNQ) ? aq : ak, olo, ohi);
      if (j > kNQ) {
        olo = lo[u];
        ohi = hi[u];
      }
      *reinterpret_cast<uint2*>(dst + d8 * 8) = fp8x8(olo, scale);
      *reinterpret_cast<uint2*>(dst + kHalf + d8 * 8) = fp8x8(ohi, scale);
    }
  } else {
    uint32_t acc[kB];
#pragma unroll
    for (int p = 0; p < kB; ++p) acc[p] = 0u;
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int hr = lane + kL * u, p = hr / kHH, j = (hr - p * kHH) % (kNQ + 2);
      V8 olo, ohi;
      rope_fwd8(lo[u], hi[u], (kSame || j < kNQ) ? aq : ak, olo, ohi);
      const uint32_t m = j <= kNQ ? ak_rht::bmax_xa(bmax4(olo), bmax4(ohi)) : ak_rht::bmax_xa(bmax4(lo[u]), bmax4(hi[u]));
#pragma unroll
      for (int q = 0; q < kB; ++q)
        if (q == p) acc[q] = ak_rht::bmax_xa(acc[q], m);
    }
    __shared__ uint32_t red[kB][kT / 32];
    const int warp = threadIdx.x / 32, wl = threadIdx.x % 32;
#pragma unroll
    for (int p = 0; p < kB; ++p) {
      uint32_t v = acc[p];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v = ak_rht::bmax_xa(v, __shfl_xor_sync(0xffffffffu, v, o));
      if (wl == 0) red[p][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < kB) {
      uint32_t v = 0u;
#pragma unroll
      for (int w = 0; w < kT / 32; ++w) v = ak_rht::bmax_xa(v, red[threadIdx.x][w]);
      part[(int64_t)s * kB + threadIdx.x] = ak_rht::unpack_bmax(v);
    }
  }
}

void check_bf16(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
              " must be a contiguous CUDA bf16 tensor");
}
void check_tab(const at::Tensor& tab, int64_t S) {
  TORCH_CHECK(tab.is_cuda() && tab.scalar_type() == at::kFloat && tab.is_contiguous() && tab.dim() == 2 &&
                  tab.size(0) >= S && tab.size(1) == 4 * kD,
              "tab: contiguous f32 [s, 512]");
}

}  // namespace

// rope_bwd_amax(dq [s, b, h * nq, 128], dk [s, b, h, 128], dv [s, b, h, 128], tab [s, 512] f32, same, sign_mask)
//   -> (dqkv [s, b, h, (nq + 2) * 128] bf16, amax [2] f32: pre-RHT, post-RHT).  (s * b) % 32 == 0, nq even.
std::vector<at::Tensor> rope_bwd_amax(const at::Tensor& dq, const at::Tensor& dk, const at::Tensor& dv,
                                      const at::Tensor& tab, bool same, int64_t sign_mask, int64_t version) {
  check_bf16(dq, "dq");
  check_bf16(dk, "dk");
  check_bf16(dv, "dv");
  TORCH_CHECK(dq.dim() == 4 && dk.dim() == 4 && dv.sizes() == dk.sizes() && dq.size(3) == kD && dk.size(3) == kD &&
                  dq.size(0) == dk.size(0) && dq.size(1) == dk.size(1) && dq.size(2) % dk.size(2) == 0,
              "dq [s, b, h * nq, 128], dk = dv [s, b, h, 128]");
  const int64_t S = dq.size(0), b = dq.size(1), H = dk.size(2), NQ = dq.size(2) / H, rows = S * b;
  TORCH_CHECK((NQ + 2) % 2 == 0 && rows % kTR == 0 && rows / kTR <= 65535 && b <= 65535, "rows % 32, nq even");
  check_tab(tab, S);
  const at::cuda::CUDAGuard guard(dq.device());
  auto dqkv = at::empty({S, b, H, (NQ + 2) * kD}, dq.options());
  auto amax = at::zeros({2}, dq.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* pq = reinterpret_cast<const __nv_bfloat16*>(dq.data_ptr());
  const auto* pk = reinterpret_cast<const __nv_bfloat16*>(dk.data_ptr());
  const auto* pv = reinterpret_cast<const __nv_bfloat16*>(dv.data_ptr());
  auto* po = reinterpret_cast<__nv_bfloat16*>(dqkv.data_ptr());
  if (version >= 30) {          // version 3, variant version - 30: (rows, heads, min CTAs) per the table below
    TORCH_CHECK(H == 8 && NQ == 4 && b == 2, "version 3: h 8, nq 4, b 2");
    const int v = (int)version - 30;
    auto* pt = tab.data_ptr<float>();
    auto* pa = amax.data_ptr<float>();
#define AK_B3(SM, R, HB, MB)                                                                            \
  rope_bwd_amax_kernel_v3<SM, 8, 4, 2, R, HB, MB><<<dim3((unsigned)(48 / HB), (unsigned)(rows / R)), R * 8, 0, stream>>>( \
      pq, pk, pv, pt, po, pa, (uint32_t)sign_mask)
#define AK_B3V(SM)                                                                                      \
  switch (v) {                                                                                          \
    case 0: AK_B3(SM, 16, 4, 4); break;                                                                 \
    case 1: AK_B3(SM, 16, 8, 3); break;                                                                 \
    case 2: AK_B3(SM, 16, 8, 4); break;                                                                 \
    case 3: AK_B3(SM, 32, 4, 2); break;                                                                 \
    case 4: AK_B3(SM, 16, 4, 6); break; \
    case 5: AK_B3(SM, 16, 6, 4); break; \
    case 6: AK_B3(SM, 16, 4, 8); break; \
    case 7: AK_B3(SM, 16, 3, 6); break; \
    case 8: AK_B3(SM, 16, 2, 8); break; \
    default: AK_B3(SM, 16, 4, 5); break; \
  }
    TORCH_CHECK(rows % 16 == 0 && (v != 3 || rows % 32 == 0) && v >= 0 && v <= 9, "rows / variant");
    if (same) { AK_B3V(true) } else { AK_B3V(false) }
#undef AK_B3V
#undef AK_B3
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dqkv, amax};
  }
  if (version == 2) {
    TORCH_CHECK((H * (NQ + 2)) % kHB == 0, "heads per token % 4 == 0 (version 2)");
    dim3 grid2((unsigned)(H * (NQ + 2) / kHB), (unsigned)(rows / kTR));
    if (same)
      rope_bwd_amax_kernel_v2<true><<<grid2, kThr, 0, stream>>>(pq, pk, pv, tab.data_ptr<float>(), po,
                                                                amax.data_ptr<float>(), (int)b, (int)H, (int)NQ,
                                                                (uint32_t)sign_mask);
    else
      rope_bwd_amax_kernel_v2<false><<<grid2, kThr, 0, stream>>>(pq, pk, pv, tab.data_ptr<float>(), po,
                                                                 amax.data_ptr<float>(), (int)b, (int)H, (int)NQ,
                                                                 (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dqkv, amax};
  }
  dim3 grid((unsigned)(H * (NQ + 2) / 2), (unsigned)(rows / kTR));
  if (same)
    rope_bwd_amax_kernel<true><<<grid, kThr, 0, stream>>>(pq, pk, pv, tab.data_ptr<float>(), po, amax.data_ptr<float>(),
                                                          (int)b, (int)H, (int)NQ, (uint32_t)sign_mask);
  else
    rope_bwd_amax_kernel<false><<<grid, kThr, 0, stream>>>(pq, pk, pv, tab.data_ptr<float>(), po,
                                                           amax.data_ptr<float>(), (int)b, (int)H, (int)NQ,
                                                           (uint32_t)sign_mask);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dqkv, amax};
}

// rope_fwd_partials(qkv [s, b, h, (nq + 2) * 128] bf16, tab, same, part f32 [s * b], rev): QKV direct's pass 1
int pick_u(int64_t n_hr) {          // version 2: head-rows per thread (16 lanes)
  for (int u : {3, 6, 12})
    if ((int64_t)kLanes * u >= n_hr) return u;
  return 0;
}

void rope_fwd_partials(const at::Tensor& qkv, const at::Tensor& tab, bool same, at::Tensor part, bool rev,
                       int64_t version) {
  check_bf16(qkv, "qkv");
  TORCH_CHECK(qkv.dim() == 4 && qkv.size(3) % kD == 0 && (qkv.size(3) / kD) >= 3, "qkv [s, b, h, (nq + 2) * 128]");
  const int64_t S = qkv.size(0), b = qkv.size(1), H = qkv.size(2), NQ = qkv.size(3) / kD - 2;
  TORCH_CHECK(b >= 1 && b <= kMaxB, "b <= 4");
  check_tab(tab, S);
  TORCH_CHECK(part.is_cuda() && part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() >= S * b,
              "part: f32 [s * b]");
  const at::cuda::CUDAGuard guard(qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* px = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  if (version >= 30) {          // version 3: (threads, min CTAs) variant version - 30
    TORCH_CHECK(H == 8 && NQ == 4 && b == 2, "version 3: h 8, nq 4, b 2");
    const int v = (int)version - 30;
    float* pp = part.data_ptr<float>();
    const float* pt = tab.data_ptr<float>();
#define AK_F3(SM, RV, T, MB) rope_fwd_pass_kernel_v3<SM, RV, 8, 4, 2, T, MB, false><<<(unsigned)S, T, 0, stream>>>( \
      px, pt, pp, nullptr, nullptr, (int)S)
#define AK_F3V(SM, RV) \
  switch (v) { case 0: AK_F3(SM, RV, 128, 4); break; case 1: AK_F3(SM, RV, 128, 5); break; \
               case 2: AK_F3(SM, RV, 256, 2); break; case 3: AK_F3(SM, RV, 256, 3); break; \
               case 4: AK_F3(SM, RV, 64, 8); break; case 5: AK_F3(SM, RV, 128, 3); break; \
               case 6: AK_F3(SM, RV, 192, 3); break; case 7: AK_F3(SM, RV, 96, 5); break; \
               default: AK_F3(SM, RV, 192, 2); break; }
    if (same) { if (rev) { AK_F3V(true, true) } else { AK_F3V(true, false) } }
    else { if (rev) { AK_F3V(false, true) } else { AK_F3V(false, false) } }
#undef AK_F3V
#undef AK_F3
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (version == 2) {
    const int u = pick_u(b * H * (NQ + 2));
    TORCH_CHECK(u > 0, "too many head-rows per s for version 2");
#define AK_P1V2(SM, RV, U) rope_fwd_partials_kernel_v2<SM, RV, U><<<(unsigned)S, kP2Thr, 0, stream>>>( \
      px, tab.data_ptr<float>(), part.data_ptr<float>(), (int)S, (int)b, (int)H, (int)NQ)
#define AK_P1V2U(SM, RV) \
  if (u == 3) AK_P1V2(SM, RV, 3); else if (u == 6) AK_P1V2(SM, RV, 6); else AK_P1V2(SM, RV, 12)
    if (same) { if (rev) { AK_P1V2U(true, true); } else { AK_P1V2U(true, false); } }
    else { if (rev) { AK_P1V2U(false, true); } else { AK_P1V2U(false, false); } }
#undef AK_P1V2U
#undef AK_P1V2
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#define AK_P1(SM, RV) rope_fwd_partials_kernel<SM, RV><<<(unsigned)S, kPThr, 0, stream>>>(px, tab.data_ptr<float>(), \
      part.data_ptr<float>(), (int)S, (int)b, (int)H, (int)NQ)
  if (same) { if (rev) AK_P1(true, true); else AK_P1(true, false); }
  else { if (rev) AK_P1(false, true); else AK_P1(false, false); }
#undef AK_P1
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// rope_fwd_fp8(qkv, tab, same, ws f32 [>= 1] (ws[0] the scale), out u8 [packed q | k | v], rev): QKV direct's pass 2
void rope_fwd_fp8(const at::Tensor& qkv, const at::Tensor& tab, bool same, const at::Tensor& ws, at::Tensor out,
                  bool rev, int64_t version) {
  check_bf16(qkv, "qkv");
  TORCH_CHECK(qkv.dim() == 4 && qkv.size(3) % kD == 0 && (qkv.size(3) / kD) >= 3, "qkv [s, b, h, (nq + 2) * 128]");
  const int64_t S = qkv.size(0), b = qkv.size(1), H = qkv.size(2), NQ = qkv.size(3) / kD - 2;
  check_tab(tab, S);
  TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == at::kFloat && ws.numel() >= 1, "ws");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kByte && out.is_contiguous() && out.numel() == qkv.numel() &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
              "out: contiguous u8, numel as qkv");
  const at::cuda::CUDAGuard guard(qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* px = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  if (version >= 30) {
    TORCH_CHECK(H == 8 && NQ == 4 && b == 2, "version 3: h 8, nq 4, b 2");
    const int v = (int)version - 30;
    const float* pt = tab.data_ptr<float>();
    const float* pw = ws.data_ptr<float>();
    uint8_t* po = out.data_ptr<uint8_t>();
#define AK_G3(SM, RV, T, MB) rope_fwd_pass_kernel_v3<SM, RV, 8, 4, 2, T, MB, true><<<(unsigned)S, T, 0, stream>>>( \
      px, pt, nullptr, pw, po, (int)S)
#define AK_G3V(SM, RV) \
  switch (v) { case 0: AK_G3(SM, RV, 128, 4); break; case 1: AK_G3(SM, RV, 128, 5); break; \
               case 2: AK_G3(SM, RV, 256, 2); break; case 3: AK_G3(SM, RV, 256, 3); break; \
               case 4: AK_G3(SM, RV, 64, 8); break; case 5: AK_G3(SM, RV, 128, 3); break; \
               case 6: AK_G3(SM, RV, 192, 3); break; case 7: AK_G3(SM, RV, 96, 5); break; \
               default: AK_G3(SM, RV, 192, 2); break; }
    if (same) { if (rev) { AK_G3V(true, true) } else { AK_G3V(true, false) } }
    else { if (rev) { AK_G3V(false, true) } else { AK_G3V(false, false) } }
#undef AK_G3V
#undef AK_G3
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (version == 2) {
    const int u = pick_u(b * H * (NQ + 2));
    TORCH_CHECK(u > 0, "too many head-rows per s for version 2");
#define AK_P2V2(SM, RV, U) rope_fwd_fp8_kernel_v2<SM, RV, U><<<(unsigned)S, kP2Thr, 0, stream>>>( \
      px, tab.data_ptr<float>(), ws.data_ptr<float>(), out.data_ptr<uint8_t>(), (int)S, (int)b, (int)H, (int)NQ)
#define AK_P2V2U(SM, RV) \
  if (u == 3) AK_P2V2(SM, RV, 3); else if (u == 6) AK_P2V2(SM, RV, 6); else AK_P2V2(SM, RV, 12)
    if (same) { if (rev) { AK_P2V2U(true, true); } else { AK_P2V2U(true, false); } }
    else { if (rev) { AK_P2V2U(false, true); } else { AK_P2V2U(false, false); } }
#undef AK_P2V2U
#undef AK_P2V2
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#define AK_P2(SM, RV) rope_fwd_fp8_kernel<SM, RV><<<(unsigned)S, kPThr, 0, stream>>>(px, tab.data_ptr<float>(), \
      ws.data_ptr<float>(), out.data_ptr<uint8_t>(), (int)S, (int)b, (int)H, (int)NQ)
  if (same) { if (rev) AK_P2(true, true); else AK_P2(true, false); }
  else { if (rev) AK_P2(false, true); else AK_P2(false, false); }
#undef AK_P2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_bwd_amax", &rope_bwd_amax, "X12 RoPE backward + the NVFP4 gradient quantizer's pre/post-RHT amaxes",
        py::arg("dq"), py::arg("dk"), py::arg("dv"), py::arg("tab"), py::arg("same"), py::arg("sign_mask"),
        py::arg("version") = 2);
  m.def("rope_fwd_partials", &rope_fwd_partials, "QKV direct pass 1: per-position max |bf16(RoPE(x))|", py::arg("qkv"),
        py::arg("tab"), py::arg("same"), py::arg("part"), py::arg("rev") = false, py::arg("version") = 2);
  m.def("rope_fwd_fp8", &rope_fwd_fp8, "QKV direct pass 2: the RoPE straight to the packed E4M3 buffer",
        py::arg("qkv"), py::arg("tab"), py::arg("same"), py::arg("ws"), py::arg("out"), py::arg("rev") = true,
        py::arg("version") = 2);
}
