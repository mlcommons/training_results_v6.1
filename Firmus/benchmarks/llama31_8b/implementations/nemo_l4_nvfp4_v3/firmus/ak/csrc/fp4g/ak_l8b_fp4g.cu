// ak_l8b_fp4g.cu (Llama 3.1 8B kernel campaign, 2026-10-07; dev g2): the kernels of an NVFP4 parameter gather.
//
// The recipe's weight quantizer (TE's NVFP4Quantizer: 2D 16x16 blocks, round to nearest, no RHT, GEMM-swizzled scales)
// turns a bf16 weight W [M, K] into: one E4M3 scale per 16x16 tile (from the tile's amax and the weight's S_enc), the
// FP4 row-wise data [M, K/2], the column-wise data [K, M/2] -- the same nibbles transposed, since in 2D mode a tile's
// row-wise and column-wise scales are the same -- and both scale tensors in cuBLAS's swizzled layout.  These kernels
// produce the same bytes from the distributed optimizer's shards (each rank holds a flat element range [lo, hi) of W
// with lo and hi multiples of 16, so a shard boundary never splits a 1x16 block or a byte):
//   split_amax  - for each tile-row of W that shard boundaries cut ("split"), this rank's partial tile amaxes (0 for the
//                 tiles it does not touch), to be max-all-reduced across the ranks that share the tile-row;
//   shard_quant - this rank's shard quantized: the tile amaxes of the tile-rows it holds whole computed here, those of
//                 the split tile-rows read from the all-reduced partials; the nibbles of its own elements written into
//                 the row-wise image (whole 4-byte words) and the E4M3 tile scales of every tile it touches into a
//                 per-weight tile-scale array [M/16, K/16] (to be max-all-reduced as bytes: the E4M3 codes of
//                 non-negative values order as their values, and the ranks that share a split tile write equal codes);
//   post_ag     - after the gather, per 256 x 128 block: the column-wise data (an 8x8 nibble transpose in registers,
//                 staged through shared memory for coalesced 16-byte stores) and both scale tensors written directly in
//                 the swizzled layout (swizzle::gemm_swizzled_scale_idx, cast/nvfp4/swizzle.cuh).
// The arithmetic is TE's quantize_transpose_nvfp4_2D_kernel's (common/cast/nvfp4/quantize_transpose_nvfp4.cuh and
// core_nvfp4.cuh, TE 2.20.0+131157b5): S_enc = 2688 / amax (FLT_MAX-clamped; 1 for amax 0 or S_enc 0); S_dec =
// float(1.0 / double(S_enc)); per tile S_dec_b = e4m3_satfinite(min(tile_amax * (S_enc * (1/6)), FLT_MAX)); encode =
// min(1 / (float(S_dec_b) * S_dec), FLT_MAX); per element float(bf16) * encode (mul.f32x2) and cvt.rn.satfinite.e2m1x2
// -- TE's own PTX (util/ptx_arch_spec.cuh mul_cvt_bf16_to_fp4_4x_with_rn).  The tile amax is the max of |x| over the
// tile's non-NaN elements, as TE's max.xorsign.abs.bf16x2 + fmaxf reduction.  Built without fast math, for sm_103a.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cfloat>
#include <cstdint>

namespace {

// ------------------------------------------------------------------------------------------------ TE's arithmetic
__device__ __forceinline__ float te_global_encode_scale(const float amax) {
  constexpr float fp8_max = 448.0f;
  constexpr float fp4_max = 6.0f;
  float s = fp8_max * fp4_max / amax;
  s = fminf(s, FLT_MAX);
  if (amax == 0.0f || s == 0.0f) return 1.0f;
  return s;
}

__device__ __forceinline__ __nv_fp8_e4m3 te_tile_scale(const float tile_amax, const float S_enc) {
  constexpr float fp4_max_inv = 1.0f / 6.0f;
  const float s = tile_amax * (S_enc * fp4_max_inv);
  return static_cast<__nv_fp8_e4m3>(fminf(s, FLT_MAX));
}

__device__ __forceinline__ float te_tile_encode(const __nv_fp8_e4m3 sc, const float S_dec) {
  return fminf(1.0f / (static_cast<float>(sc) * S_dec), FLT_MAX);
}

// TE's mul_cvt_bf16_to_fp4_4x_with_rn, verbatim: 4 bf16 (one 64-bit word) -> 16 bits, element 0 in the low nibble
__device__ __forceinline__ uint32_t te_cvt4(const uint64_t in_4x, const float2 scale) {
  uint32_t out_4x = 0;
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b16 v0_bf16; \n\t"
      ".reg.b16 v1_bf16; \n\t"
      ".reg.b16 v2_bf16; \n\t"
      ".reg.b16 v3_bf16; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      ".reg.b8 f0; \n\t"
      ".reg.b8 f1; \n\t"
      "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16} , %1; \n\t"
      "cvt.f32.bf16 v0, v0_bf16; \n\t"
      "cvt.f32.bf16 v1, v1_bf16; \n\t"
      "cvt.f32.bf16 v2, v2_bf16; \n\t"
      "cvt.f32.bf16 v3, v3_bf16; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %2; \n\t"
      "mul.f32x2 v23, v23, %2; \n\t"
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "mov.b32 %0, {f0, f1, f0, f1};\n\t"
      "}"
      : "=r"(out_4x)
      : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale)));
  return out_4x & 0xFFFFu;
}

// max |x| over 8 bf16 (NaN ignored: fmaxf returns the other operand), as float -- exact
__device__ __forceinline__ float absmax8(const uint4 v, float acc) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    acc = fmaxf(acc, __uint_as_float((w[k] << 16) & 0x7FFF0000u));
    acc = fmaxf(acc, __uint_as_float(w[k] & 0x7FFF0000u));
  }
  return acc;
}

__device__ __forceinline__ uint4 ld_nc16(const void *p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

// the segment whose CTA range holds `cta` (begin: ascending int32 CTA offsets, n entries)
__device__ __forceinline__ int find_seg(const int32_t *__restrict__ begin, const int n, const int32_t cta) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (__ldg(begin + mid) <= cta) lo = mid; else hi = mid - 1;
  }
  return lo;
}

// ------------------------------------------------------------------------------------------------ shard kernels
// Segment table (int64 per field): one entry per (weight, rank) shard, or per split tile-row for split_amax.
enum SegField { S_SRC = 0, S_DST, S_TSC, S_AMAX, S_SLOT_FIRST, S_SLOT_LAST, S_K, S_LO, S_HI, S_TI0, S_TI1, S_NF };
// S_SRC: bf16* -> element lo of W (this rank's shard);  S_DST: uint8* -> row-wise data of W (element 0);
// S_TSC: uint8* -> tile scales of W [M/16, K/16];  S_AMAX: float* -> W's amax;  S_SLOT_FIRST / S_SLOT_LAST: float* ->
// the all-reduced partial amaxes (K/16 floats) of tile-row S_TI0 / S_TI1 when split, else 0;  S_TI0 .. S_TI1: the
// tile-rows the CTAs cover (shard_quant: every tile-row the shard touches; split_amax: one split tile-row, S_TI0 ==
// S_TI1, its slot in S_SLOT_FIRST).

template <bool kAmaxOnly>
__global__ void __launch_bounds__(128) shard_kernel(const int64_t *__restrict__ tab, const int32_t *__restrict__ begin,
                                                    const int n_seg) {
  __shared__ float red[4][16];
  const int s = find_seg(begin, n_seg, static_cast<int32_t>(blockIdx.x));
  const int64_t *e = tab + static_cast<int64_t>(s) * S_NF;
  const int64_t K = e[S_K], lo = e[S_LO], hi = e[S_HI], ti0 = e[S_TI0], ti1 = e[S_TI1];
  const int64_t nb = K / 256;
  const int64_t local = static_cast<int64_t>(blockIdx.x) - __ldg(begin + s);
  const int64_t ti = ti0 + local / nb, cb = local % nb;
  const __nv_bfloat16 *src = reinterpret_cast<const __nv_bfloat16 *>(e[S_SRC]);
  const float *slot = nullptr;
  if (!kAmaxOnly) {
    if (ti == ti0 && e[S_SLOT_FIRST] != 0) slot = reinterpret_cast<const float *>(e[S_SLOT_FIRST]);
    else if (ti == ti1 && e[S_SLOT_LAST] != 0) slot = reinterpret_cast<const float *>(e[S_SLOT_LAST]);
  }
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t col = cb * 256 + lane * 8;
  uint4 v[4];
  bool own[4];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t f = (ti * 16 + w * 4 + i) * K + col;
    own[i] = f >= lo && f < hi;
    if (own[i]) v[i] = ld_nc16(src + (f - lo));
  }
  if (slot == nullptr) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (own[i]) amax = absmax8(v[i], amax);
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
    if ((lane & 1) == 0) red[w][lane >> 1] = amax;
    __syncthreads();
  }
  const int tj = lane >> 1;  // tile within this CTA's 256 columns
  float tile_amax;
  if (slot != nullptr) {
    tile_amax = __ldg(slot + cb * 16 + tj);
  } else {
    tile_amax = fmaxf(fmaxf(red[0][tj], red[1][tj]), fmaxf(red[2][tj], red[3][tj]));
  }
  if (kAmaxOnly) {
    if (w == 0 && (lane & 1) == 0) reinterpret_cast<float *>(e[S_SLOT_FIRST])[cb * 16 + tj] = tile_amax;
    return;
  }
  const float S_enc = te_global_encode_scale(*reinterpret_cast<const float *>(e[S_AMAX]));
  const float S_dec = static_cast<float>(1.0 / static_cast<double>(S_enc));
  const __nv_fp8_e4m3 sc = te_tile_scale(tile_amax, S_enc);
  const float enc = te_tile_encode(sc, S_dec);
  const float2 enc2 = make_float2(enc, enc);
  uint8_t *dst = reinterpret_cast<uint8_t *>(e[S_DST]);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (!own[i]) continue;
    const int64_t f = (ti * 16 + w * 4 + i) * K + col;
    const uint64_t a = (static_cast<uint64_t>(v[i].y) << 32) | v[i].x;
    const uint64_t b = (static_cast<uint64_t>(v[i].w) << 32) | v[i].z;
    const uint32_t out = te_cvt4(a, enc2) | (te_cvt4(b, enc2) << 16);
    *reinterpret_cast<uint32_t *>(dst + f / 2) = out;
  }
  if (w == 0 && (lane & 1) == 0) {
    reinterpret_cast<uint8_t *>(e[S_TSC])[ti * (K / 16) + cb * 16 + tj] = sc.__x;
  }
}


// ---- version 2 (shard_quant2 / split_amax2): S_dec = 1.0f / S_enc in float (bit-identical to TE's float(1.0 / double):
// IEEE division rounded twice through double is innocuous for float operands -- checked exhaustively by sdec_check),
// 32-bit CTA mapping, `cb_per_cta` 256-column blocks per CTA with the next block's loads in flight during the current
// block's reduction and cast.  CTAs per entry: n_tile_rows * ceil((K / 256) / cb_per_cta).
template <bool kAmaxOnly>
__global__ void __launch_bounds__(128) shard_kernel2(const int64_t *__restrict__ tab, const int32_t *__restrict__ begin,
                                                     const int n_seg, const int cb_per_cta, const int anchor_rule) {
  __shared__ float red[2][4][16];
  const int s = find_seg(begin, n_seg, static_cast<int32_t>(blockIdx.x));
  const int64_t *e = tab + static_cast<int64_t>(s) * S_NF;
  const int64_t K = e[S_K], lo = e[S_LO], hi = e[S_HI], ti0 = e[S_TI0], ti1 = e[S_TI1];
  const uint32_t nb = static_cast<uint32_t>(K / 256);
  const uint32_t nbc = (nb + cb_per_cta - 1) / cb_per_cta;
  const uint32_t local = blockIdx.x - static_cast<uint32_t>(__ldg(begin + s));
  const int64_t ti = ti0 + local / nbc;
  const uint32_t cb0 = (local % nbc) * cb_per_cta;
  const uint32_t n_my = min(static_cast<uint32_t>(cb_per_cta), nb - cb0);
  const __nv_bfloat16 *src = reinterpret_cast<const __nv_bfloat16 *>(e[S_SRC]);
  const float *slot = nullptr;
  if (!kAmaxOnly) {
    if (ti == ti0 && e[S_SLOT_FIRST] != 0) slot = reinterpret_cast<const float *>(e[S_SLOT_FIRST]);
    else if (ti == ti1 && e[S_SLOT_LAST] != 0) slot = reinterpret_cast<const float *>(e[S_SLOT_LAST]);
  }
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, tj = lane >> 1;
  const int64_t rowf = (ti * 16 + w * 4) * K + lane * 8;  // flat index of (row ti*16 + 4w, column 8 lane)
  float S_enc = 1.0f, S_dec = 1.0f;
  uint8_t *dst = nullptr;
  if (!kAmaxOnly) {
    S_enc = te_global_encode_scale(*reinterpret_cast<const float *>(e[S_AMAX]));
    S_dec = 1.0f / S_enc;
    dst = reinterpret_cast<uint8_t *>(e[S_DST]);
  }
  uint4 v[4], nx[4];  // the current block's vectors, the next block's (static indexing only: no local memory)
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t f = rowf + i * K + cb0 * 256;
    if (f >= lo && f < hi) v[i] = ld_nc16(src + (f - lo));
  }
  for (uint32_t k = 0; k < n_my; ++k) {
    const int cur = k & 1;
    const uint32_t cb = cb0 + k;
    if (k + 1 < n_my) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t f = rowf + i * K + (cb + 1) * 256;
        if (f >= lo && f < hi) nx[i] = ld_nc16(src + (f - lo));
      }
    }
    float tile_amax;
    if (slot != nullptr) {
      tile_amax = __ldg(slot + cb * 16 + tj);
    } else {
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t f = rowf + i * K + cb * 256;
        if (f >= lo && f < hi) amax = absmax8(v[i], amax);
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
      if ((lane & 1) == 0) red[cur][w][tj] = amax;
      __syncthreads();
      tile_amax = fmaxf(fmaxf(red[cur][0][tj], red[cur][1][tj]), fmaxf(red[cur][2][tj], red[cur][3][tj]));
    }
    if (kAmaxOnly) {
      if (w == 0 && (lane & 1) == 0) reinterpret_cast<float *>(e[S_SLOT_FIRST])[cb * 16 + tj] = tile_amax;
    } else {
      const __nv_fp8_e4m3 sc = te_tile_scale(tile_amax, S_enc);
      const float enc = te_tile_encode(sc, S_dec);
      const float2 enc2 = make_float2(enc, enc);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t f = rowf + i * K + cb * 256;
        if (!(f >= lo && f < hi)) continue;
        const uint64_t a = (static_cast<uint64_t>(v[i].y) << 32) | v[i].x;
        const uint64_t b = (static_cast<uint64_t>(v[i].w) << 32) | v[i].z;
        *reinterpret_cast<uint32_t *>(dst + f / 2) = te_cvt4(a, enc2) | (te_cvt4(b, enc2) << 16);
      }
      // the tile's scale: written by every rank that touches the tile (anchor_rule 0: the tile-scale buffers are then
      // max-all-reduced), or only by the rank that owns the tile's anchor element 256 t (anchor_rule 1: the per-bucket
      // tile-scale image, sharded as the elements, is all-gathered; a rank that shares a cut tile-row has its full
      // amaxes from the all-reduced partials either way)
      const int64_t t_idx = ti * (K / 16) + cb * 16 + tj;
      if (w == 0 && (lane & 1) == 0 && (!anchor_rule || (t_idx * 256 >= lo && t_idx * 256 < hi)))
        reinterpret_cast<uint8_t *>(e[S_TSC])[t_idx] = sc.__x;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = nx[i];
  }
}

// sdec_check: for every positive finite float x (bit patterns 1 .. 0x7F7FFFFF) and +inf, float(1.0 / double(x)) against
// 1.0f / x; counts the bit patterns that differ
__global__ void sdec_check_kernel(unsigned long long *bad) {
  unsigned long long n = 0;
  for (uint64_t u = 1 + blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x; u <= 0x7F800000ull;
       u += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    const float x = __uint_as_float(static_cast<uint32_t>(u));
    const float a = static_cast<float>(1.0 / static_cast<double>(x));
    const float b = 1.0f / x;
    n += (__float_as_uint(a) != __float_as_uint(b));
  }
  atomicAdd(bad, n);
}

// ------------------------------------------------------------------------------------------------ post-gather kernel
enum PostField { P_ROW = 0, P_TSC, P_COL, P_RSC, P_CSC, P_M, P_K, P_NF };

__device__ __forceinline__ void tr8x8(uint32_t (&a)[8]) {
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const uint32_t t = ((a[i] >> 4) ^ a[i + 1]) & 0x0F0F0F0Fu;
    a[i + 1] ^= t;
    a[i] ^= t << 4;
  }
#pragma unroll
  for (int i = 0; i < 8; i += 4) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const uint32_t t = ((a[i + k] >> 8) ^ a[i + k + 2]) & 0x00FF00FFu;
      a[i + k + 2] ^= t;
      a[i + k] ^= t << 8;
    }
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint32_t t = ((a[k] >> 16) ^ a[k + 4]) & 0x0000FFFFu;
    a[k + 4] ^= t;
    a[k] ^= t << 16;
  }
}

// one CTA = rows [R0, R0 + 256) x columns [C0, C0 + 128) of W (needs M % 256 == 0, K % 128 == 0)
__global__ void __launch_bounds__(256) post_ag_kernel(const int64_t *__restrict__ tab,
                                                      const int32_t *__restrict__ begin, const int n_w) {
  constexpr int kStride = 36;  // words per staged output row (32 + 4: conflict-free 16-byte reads)
  __shared__ __align__(16) uint32_t sout[128 * kStride];
  __shared__ __align__(16) uint8_t ssc[16][8];
  const int s = find_seg(begin, n_w, static_cast<int32_t>(blockIdx.x));
  const int64_t *e = tab + static_cast<int64_t>(s) * P_NF;
  const int64_t M = e[P_M], K = e[P_K];
  const int64_t nbk = K / 128;
  const int64_t local = static_cast<int64_t>(blockIdx.x) - __ldg(begin + s);
  const int64_t R0 = (local / nbk) * 256, C0 = (local % nbk) * 128;
  const uint8_t *row = reinterpret_cast<const uint8_t *>(e[P_ROW]);
  const uint8_t *tsc = reinterpret_cast<const uint8_t *>(e[P_TSC]);
  const int t = threadIdx.x;
  const int h = t & 15, g0 = t >> 4;
  // tile scales of the block: tile-rows R0/16 .. +16, tile-columns C0/16 .. +8 (8 bytes per tile-row)
  if (t < 32) {
    const int tr = t >> 1, half = t & 1;
    const uint32_t wv = __ldg(reinterpret_cast<const uint32_t *>(tsc + (R0 / 16 + tr) * (K / 16) + C0 / 16) + half);
    reinterpret_cast<uint32_t *>(&ssc[tr][0])[half] = wv;
  }
  uint32_t a[2][8];
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int g = g0 + 16 * q;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      a[q][i] = __ldg(reinterpret_cast<const uint32_t *>(row + (R0 + 8 * g + i) * (K / 2) + C0 / 2) + h);
    }
  }
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    tr8x8(a[q]);
    const int g = g0 + 16 * q;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = 8 * h + j;
      sout[c * kStride + (g ^ (2 * h))] = a[q][j];
    }
  }
  __syncthreads();
  // column-wise data: output row C0 + c holds W's rows R0 .. R0 + 255 of column C0 + c (128 bytes)
  {
    uint8_t *col = reinterpret_cast<uint8_t *>(e[P_COL]);
    const int c = t >> 1, half = t & 1;
    const int x = 2 * (c >> 3);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int gs = 16 * half + 4 * k;
      uint4 v = *reinterpret_cast<const uint4 *>(&sout[c * kStride + (gs ^ (x & ~3))]);
      if (x & 2) v = make_uint4(v.z, v.w, v.x, v.y);
      *reinterpret_cast<uint4 *>(col + (C0 + c) * (M / 2) + R0 / 2 + 4 * gs) = v;
    }
  }
  // row-wise scales, swizzled: rows R0 .. +256 (two 128-row base blocks) x scale columns C0/16 .. +8 (two groups of 4)
  {
    uint8_t *rsc = reinterpret_cast<uint8_t *>(e[P_RSC]);
    const int64_t cbg_cnt = (K / 16 + 3) / 4;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = t + 256 * u, bb = idx >> 7, p = idx & 127;
      const int rbb = bb >> 1, cg = bb & 1;
      const int r = 128 * rbb + 32 * (p & 3) + (p >> 2);
      const uint32_t wv = reinterpret_cast<const uint32_t *>(&ssc[r >> 4][0])[cg];
      const int64_t blk = ((R0 + 128 * rbb) / 128) * cbg_cnt + C0 / 64 + cg;
      reinterpret_cast<uint32_t *>(rsc + blk * 512)[p] = wv;
    }
  }
  // column-wise scales, swizzled: rows C0 .. +128 (one base block) x scale columns R0/16 .. +16 (four groups of 4)
  {
    uint8_t *csc = reinterpret_cast<uint8_t *>(e[P_CSC]);
    const int64_t cbg_cnt = (M / 16 + 3) / 4;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = t + 256 * u, cg = idx >> 7, p = idx & 127;
      const int c = 32 * (p & 3) + (p >> 2);
      const int tc = c >> 4;
      const uint32_t wv = static_cast<uint32_t>(ssc[4 * cg + 0][tc]) | (static_cast<uint32_t>(ssc[4 * cg + 1][tc]) << 8) |
                          (static_cast<uint32_t>(ssc[4 * cg + 2][tc]) << 16) |
                          (static_cast<uint32_t>(ssc[4 * cg + 3][tc]) << 24);
      const int64_t blk = (C0 / 128) * cbg_cnt + R0 / 64 + cg;
      reinterpret_cast<uint32_t *>(csc + blk * 512)[p] = wv;
    }
  }
}


// ---- version 2 (post_ag2): one CTA = rows [R0, R0 + 256) x columns [C0, C0 + 256) (needs M % 256 == 0, K % 256 == 0).
// Each thread holds 16 columns (two words) x 8 rows of two row-groups: 8-byte loads, 128 bytes per row per half warp;
// four 8x8 transposes; the 256 output rows (columns of W) staged in shared memory (36-word rows, the word of row-group g
// of output row c at g ^ (2 * (c >> 4)): conflict-free writes and 16-byte reads), then stored two threads per row
// (32 contiguous bytes per thread pair per instruction).
__global__ void __launch_bounds__(256) post_ag_kernel2(const int64_t *__restrict__ tab,
                                                       const int32_t *__restrict__ begin, const int n_w) {
  constexpr int kStride = 36;
  extern __shared__ __align__(16) uint32_t smem2[];
  uint32_t *sout = smem2;                                              // 256 x 36 words
  uint8_t *ssc = reinterpret_cast<uint8_t *>(smem2 + 256 * kStride);  // 16 x 16 tile scales
  const int s = find_seg(begin, n_w, static_cast<int32_t>(blockIdx.x));
  const int64_t *e = tab + static_cast<int64_t>(s) * P_NF;
  const int64_t M = e[P_M], K = e[P_K];
  const uint32_t nbk = static_cast<uint32_t>(K / 256);
  const uint32_t local = blockIdx.x - static_cast<uint32_t>(__ldg(begin + s));
  const int64_t R0 = static_cast<int64_t>(local / nbk) * 256, C0 = static_cast<int64_t>(local % nbk) * 256;
  const uint8_t *row = reinterpret_cast<const uint8_t *>(e[P_ROW]);
  const uint8_t *tsc = reinterpret_cast<const uint8_t *>(e[P_TSC]);
  const int t = threadIdx.x;
  const int h2 = t & 15, g0 = t >> 4;
  if (t < 64) {  // tile-rows R0/16 .. +16 x tile-columns C0/16 .. +16: 16 bytes per tile-row
    const int tr = t >> 2, qq = t & 3;
    const uint32_t wv = __ldg(reinterpret_cast<const uint32_t *>(tsc + (R0 / 16 + tr) * (K / 16) + C0 / 16) + qq);
    reinterpret_cast<uint32_t *>(ssc + 16 * tr)[qq] = wv;
  }
  uint32_t a[4][8];  // [row-group half q][word j2] -> a[2 q + j2]
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int g = g0 + 16 * q;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint2 d = __ldg(reinterpret_cast<const uint2 *>(row + (R0 + 8 * g + i) * (K / 2) + C0 / 2) + h2);
      a[2 * q][i] = d.x;
      a[2 * q + 1][i] = d.y;
    }
  }
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int g = g0 + 16 * q;
#pragma unroll
    for (int j2 = 0; j2 < 2; ++j2) {
      tr8x8(a[2 * q + j2]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int c = 16 * h2 + 8 * j2 + j;
        sout[c * kStride + (g ^ (2 * h2))] = a[2 * q + j2][j];
      }
    }
  }
  __syncthreads();
  {
    uint8_t *col = reinterpret_cast<uint8_t *>(e[P_COL]);
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const int c = 128 * pass + (t >> 1), half = t & 1;
      const int x = 2 * (c >> 4);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int gs = 16 * half + 4 * k;
        uint4 v = *reinterpret_cast<const uint4 *>(&sout[c * kStride + (gs ^ (x & ~3))]);
        if (x & 2) v = make_uint4(v.z, v.w, v.x, v.y);
        *reinterpret_cast<uint4 *>(col + (C0 + c) * (M / 2) + R0 / 2 + 4 * gs) = v;
      }
    }
  }
  {  // row-wise scales: 2 row base blocks x 4 column groups = 8 blocks of 128 words
    uint8_t *rsc = reinterpret_cast<uint8_t *>(e[P_RSC]);
    const int64_t cbg_cnt = (K / 16 + 3) / 4;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = t + 256 * u, bb = idx >> 7, p = idx & 127;
      const int rbb = bb >> 2, cg = bb & 3;
      const int r = 128 * rbb + 32 * (p & 3) + (p >> 2);
      const uint32_t wv = reinterpret_cast<const uint32_t *>(ssc + 16 * (r >> 4))[cg];
      const int64_t blk = ((R0 + 128 * rbb) / 128) * cbg_cnt + C0 / 64 + cg;
      reinterpret_cast<uint32_t *>(rsc + blk * 512)[p] = wv;
    }
  }
  {  // column-wise scales: 2 base blocks of 128 columns x 4 groups of 4 tile-rows
    uint8_t *csc = reinterpret_cast<uint8_t *>(e[P_CSC]);
    const int64_t cbg_cnt = (M / 16 + 3) / 4;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = t + 256 * u, bb = idx >> 7, p = idx & 127;
      const int cbb = bb >> 2, cg = bb & 3;
      const int c = 128 * cbb + 32 * (p & 3) + (p >> 2);
      const int tc = c >> 4;
      const uint32_t wv = static_cast<uint32_t>(ssc[16 * (4 * cg + 0) + tc]) |
                          (static_cast<uint32_t>(ssc[16 * (4 * cg + 1) + tc]) << 8) |
                          (static_cast<uint32_t>(ssc[16 * (4 * cg + 2) + tc]) << 16) |
                          (static_cast<uint32_t>(ssc[16 * (4 * cg + 3) + tc]) << 24);
      const int64_t blk = ((C0 + 128 * cbb) / 128) * cbg_cnt + R0 / 64 + cg;
      reinterpret_cast<uint32_t *>(csc + blk * 512)[p] = wv;
    }
  }
}


// ---- version 3 (post_ag3<RG>): version 1's thread layout (16 word-columns x 16 row-group slots, 4-byte loads) with RG
// row-groups per thread -- a CTA = rows [R0, R0 + 128 * RG) x columns [C0, C0 + 128) (M % (128 RG) == 0, K % 128 ==
// 0) -- and an optional 2D CTA order: `gb` row blocks per group, row blocks fastest, so consecutive CTAs write
// consecutive segments of the same column-wise rows while a group still covers whole input rows (gb 0: version 1's order,
// column blocks fastest).  Staged output rows hold 16 RG words (+4); each thread stores two separated 64-byte runs so
// the two threads of a row hit different banks.
template <int RG>
__global__ void __launch_bounds__(256) post_ag_kernel3(const int64_t *__restrict__ tab, const int32_t *__restrict__ begin,
                                                       const int n_w, const int gb) {
  constexpr int kWords = 16 * RG;        // words per staged output row (= W rows / 8)
  constexpr int kStride = kWords + 4;
  constexpr int kTR = 8 * RG;            // tile-rows per CTA
  extern __shared__ __align__(16) uint32_t smem3[];
  uint32_t *sout = smem3;                                             // 128 x kStride words
  uint8_t *ssc = reinterpret_cast<uint8_t *>(smem3 + 128 * kStride);  // kTR x 8 tile scales
  const int s = find_seg(begin, n_w, static_cast<int32_t>(blockIdx.x));
  const int64_t *e = tab + static_cast<int64_t>(s) * P_NF;
  const int64_t M = e[P_M], K = e[P_K];
  const uint32_t nbk = static_cast<uint32_t>(K / 128), nbm = static_cast<uint32_t>(M / (128 * RG));
  const uint32_t local = blockIdx.x - static_cast<uint32_t>(__ldg(begin + s));
  uint32_t bm, bk;
  if (gb <= 0) {
    bm = local / nbk;
    bk = local % nbk;
  } else {
    const uint32_t grp = local / (gb * nbk), idx = local % (gb * nbk);
    const uint32_t g_rows = min(static_cast<uint32_t>(gb), nbm - grp * gb);
    bm = grp * gb + idx % g_rows;
    bk = idx / g_rows;
  }
  const int64_t R0 = static_cast<int64_t>(bm) * 128 * RG, C0 = static_cast<int64_t>(bk) * 128;
  const uint8_t *row = reinterpret_cast<const uint8_t *>(e[P_ROW]);
  const uint8_t *tsc = reinterpret_cast<const uint8_t *>(e[P_TSC]);
  const int t = threadIdx.x;
  const int h = t & 15, g0 = t >> 4;
  if (t < 2 * kTR) {
    const int tr = t >> 1, half = t & 1;
    const uint32_t wv = __ldg(reinterpret_cast<const uint32_t *>(tsc + (R0 / 16 + tr) * (K / 16) + C0 / 16) + half);
    reinterpret_cast<uint32_t *>(ssc + 8 * tr)[half] = wv;
  }
  uint32_t a[RG][8];
#pragma unroll
  for (int q = 0; q < RG; ++q) {
    const int g = g0 + 16 * q;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      a[q][i] = __ldg(reinterpret_cast<const uint32_t *>(row + (R0 + 8 * g + i) * (K / 2) + C0 / 2) + h);
    }
  }
#pragma unroll
  for (int q = 0; q < RG; ++q) {
    tr8x8(a[q]);
    const int g = g0 + 16 * q;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = 8 * h + j;
      sout[c * kStride + (g ^ (2 * h))] = a[q][j];
    }
  }
  __syncthreads();
  {
    uint8_t *col = reinterpret_cast<uint8_t *>(e[P_COL]);
    const int c = t >> 1, half = t & 1;
    const int x = 2 * (c >> 3);
#pragma unroll
    for (int u = 0; u < RG / 2; ++u) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int gs = 32 * u + 16 * half + 4 * k;
        uint4 v = *reinterpret_cast<const uint4 *>(&sout[c * kStride + (gs ^ (x & ~3))]);
        if (x & 2) v = make_uint4(v.z, v.w, v.x, v.y);
        *reinterpret_cast<uint4 *>(col + (C0 + c) * (M / 2) + R0 / 2 + 4 * gs) = v;
      }
    }
  }
  {  // row-wise scales: RG row base blocks x 2 column groups
    uint8_t *rsc = reinterpret_cast<uint8_t *>(e[P_RSC]);
    const int64_t cbg_cnt = (K / 16 + 3) / 4;
#pragma unroll
    for (int u = 0; u < RG; ++u) {
      const int idx = t + 256 * u, bb = idx >> 7, p = idx & 127;
      const int rbb = bb >> 1, cg = bb & 1;
      const int r = 128 * rbb + 32 * (p & 3) + (p >> 2);
      const uint32_t wv = reinterpret_cast<const uint32_t *>(ssc + 8 * (r >> 4))[cg];
      const int64_t blk = ((R0 + 128 * rbb) / 128) * cbg_cnt + C0 / 64 + cg;
      reinterpret_cast<uint32_t *>(rsc + blk * 512)[p] = wv;
    }
  }
  {  // column-wise scales: one base block of 128 columns x 2 RG groups of 4 tile-rows
    uint8_t *csc = reinterpret_cast<uint8_t *>(e[P_CSC]);
    const int64_t cbg_cnt = (M / 16 + 3) / 4;
#pragma unroll
    for (int u = 0; u < RG; ++u) {
      const int idx = t + 256 * u, cg = idx >> 7, p = idx & 127;
      const int c = 32 * (p & 3) + (p >> 2);
      const int tc = c >> 4;
      const uint32_t wv = static_cast<uint32_t>(ssc[8 * (4 * cg + 0) + tc]) |
                          (static_cast<uint32_t>(ssc[8 * (4 * cg + 1) + tc]) << 8) |
                          (static_cast<uint32_t>(ssc[8 * (4 * cg + 2) + tc]) << 16) |
                          (static_cast<uint32_t>(ssc[8 * (4 * cg + 3) + tc]) << 24);
      const int64_t blk = (C0 / 128) * cbg_cnt + R0 / 64 + cg;
      reinterpret_cast<uint32_t *>(csc + blk * 512)[p] = wv;
    }
  }
}

void check_tab(const at::Tensor &tab, const at::Tensor &begin, int64_t nf) {
  TORCH_CHECK(tab.is_cuda() && tab.scalar_type() == at::kLong && tab.is_contiguous() && tab.dim() == 2 &&
                  tab.size(1) == nf, "table: contiguous CUDA int64 [n, ", nf, "]");
  TORCH_CHECK(begin.is_cuda() && begin.scalar_type() == at::kInt && begin.is_contiguous() &&
                  begin.numel() == tab.size(0), "begin: contiguous CUDA int32 [n]");
}

}  // namespace

// split_amax(tab, begin, n_cta): one entry per split tile-row (S_TI0 == S_TI1; partial amaxes into S_SLOT_FIRST)
void split_amax(const at::Tensor &tab, const at::Tensor &begin, int64_t n_cta) {
  check_tab(tab, begin, S_NF);
  if (n_cta == 0) return;
  const at::cuda::CUDAGuard guard(tab.device());
  shard_kernel<true><<<static_cast<unsigned>(n_cta), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      tab.data_ptr<int64_t>(), begin.data_ptr<int32_t>(), static_cast<int>(tab.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// shard_quant(tab, begin, n_cta): one entry per (weight, rank) shard
void shard_quant(const at::Tensor &tab, const at::Tensor &begin, int64_t n_cta) {
  check_tab(tab, begin, S_NF);
  if (n_cta == 0) return;
  const at::cuda::CUDAGuard guard(tab.device());
  shard_kernel<false><<<static_cast<unsigned>(n_cta), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      tab.data_ptr<int64_t>(), begin.data_ptr<int32_t>(), static_cast<int>(tab.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// post_ag(tab, begin, n_cta): one entry per weight
void post_ag(const at::Tensor &tab, const at::Tensor &begin, int64_t n_cta) {
  check_tab(tab, begin, P_NF);
  if (n_cta == 0) return;
  const at::cuda::CUDAGuard guard(tab.device());
  post_ag_kernel<<<static_cast<unsigned>(n_cta), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      tab.data_ptr<int64_t>(), begin.data_ptr<int32_t>(), static_cast<int>(tab.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void split_amax2(const at::Tensor &tab, const at::Tensor &begin, int64_t n_cta, int64_t cb_per_cta) {
  check_tab(tab, begin, S_NF);
  if (n_cta == 0) return;
  const at::cuda::CUDAGuard guard(tab.device());
  shard_kernel2<true><<<static_cast<unsigned>(n_cta), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      tab.data_ptr<int64_t>(), begin.data_ptr<int32_t>(), static_cast<int>(tab.size(0)), static_cast<int>(cb_per_cta), 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shard_quant2(const at::Tensor &tab, const at::Tensor &begin, int64_t n_cta, int64_t cb_per_cta, int64_t anchor_rule) {
  check_tab(tab, begin, S_NF);
  if (n_cta == 0) return;
  const at::cuda::CUDAGuard guard(tab.device());
  shard_kernel2<false><<<static_cast<unsigned>(n_cta), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      tab.data_ptr<int64_t>(), begin.data_ptr<int32_t>(), static_cast<int>(tab.size(0)), static_cast<int>(cb_per_cta),
      static_cast<int>(anchor_rule));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void post_ag2(const at::Tensor &tab, const at::Tensor &begin, int64_t n_cta) {
  check_tab(tab, begin, P_NF);
  if (n_cta == 0) return;
  const at::cuda::CUDAGuard guard(tab.device());
  const size_t smem = (256 * 36) * sizeof(uint32_t) + 256;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(post_ag_kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    attr = true;
  }
  post_ag_kernel2<<<static_cast<unsigned>(n_cta), 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      tab.data_ptr<int64_t>(), begin.data_ptr<int32_t>(), static_cast<int>(tab.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t sdec_check() {
  auto bad = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  sdec_check_kernel<<<4096, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<unsigned long long *>(bad.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return bad.item<int64_t>();
}


template <int RG>
void post_ag3_launch(const at::Tensor &tab, const at::Tensor &begin, int64_t n_cta, int64_t gb) {
  const size_t smem = (128 * (16 * RG + 4)) * sizeof(uint32_t) + 8 * 8 * RG;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(post_ag_kernel3<RG>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem)));
    attr = true;
  }
  post_ag_kernel3<RG><<<static_cast<unsigned>(n_cta), 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      tab.data_ptr<int64_t>(), begin.data_ptr<int32_t>(), static_cast<int>(tab.size(0)), static_cast<int>(gb));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// post_ag3(tab, begin, n_cta, rg (2 or 4), gb): CTAs per weight (M / (128 rg)) * (K / 128)
void post_ag3(const at::Tensor &tab, const at::Tensor &begin, int64_t n_cta, int64_t rg, int64_t gb) {
  check_tab(tab, begin, P_NF);
  if (n_cta == 0) return;
  const at::cuda::CUDAGuard guard(tab.device());
  if (rg == 2) post_ag3_launch<2>(tab, begin, n_cta, gb);
  else if (rg == 4) post_ag3_launch<4>(tab, begin, n_cta, gb);
  else TORCH_CHECK(false, "rg must be 2 or 4");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("split_amax", &split_amax, "partial tile amaxes of the split tile-rows of this rank's shards");
  m.def("shard_quant", &shard_quant, "NVFP4 2D quantize of this rank's shards (row-wise nibbles + tile scales)");
  m.def("post_ag", &post_ag, "after the gather: column-wise data and both scale tensors, GEMM-swizzled");
  m.def("split_amax2", &split_amax2, "version 2 of split_amax");
  m.def("shard_quant2", &shard_quant2, "version 2 of shard_quant (float S_dec, CB blocks per CTA, prefetch)",
        py::arg("tab"), py::arg("begin"), py::arg("n_cta"), py::arg("cb_per_cta"), py::arg("anchor_rule") = 0);
  m.def("post_ag2", &post_ag2, "version 2 of post_ag (256 x 256 blocks)");
  m.def("post_ag3", &post_ag3, "version 3 of post_ag (rg row-groups per thread, optional 2D CTA order)");
  m.def("sdec_check", &sdec_check, "float(1.0 / double(x)) v 1.0f / x over every positive float: differing patterns");
  m.attr("SEG_FIELDS") = static_cast<int>(S_NF);
  m.attr("POST_FIELDS") = static_cast<int>(P_NF);
}
