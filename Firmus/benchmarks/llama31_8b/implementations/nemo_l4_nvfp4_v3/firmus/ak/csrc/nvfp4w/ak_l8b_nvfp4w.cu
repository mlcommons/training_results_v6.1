// ak_l8b_nvfp4w.cu (Llama 3.1 8B kernel campaign, 2026-10-02): the NVFP4 2D (16x16 block) weight quantization of the
// recipe -- TE's amax_kernel + quantize_transpose_nvfp4_2D_kernel -- reproduced for bf16 weights whose dims are
// multiples of 128.  PROF-L0: per layer 113.5 us of weight amax and 189.3 us of 2D quantize (the fc1 weight's
// 58.2 + 102.7 us move 235 MB + 349 MB, i.e. about 4.0 and 3.4 TB/s).
//
// TE 2.20 arithmetic, reproduced (cast/nvfp4/quantize_transpose_nvfp4.cuh 856-1400, core_nvfp4.cuh,
// util/ptx_arch_spec.cuh, recipe/current_scaling.cu amax_kernel):
//   amax          = max |w| over the tensor (bf16 __hmax(__habs) in TE: NaN ignored, exact)
//   S_enc         = fminf(448 * 6 / amax, FLT_MAX); 1 if amax == 0 or S_enc == 0      (fp32 division)
//   S_dec         = (float)(1.0 / S_enc)                                              (DOUBLE division, as TE)
//   block amax    = max |w| over the 16 x 16 block
//   S_dec_b_fp8   = e4m3(fminf(block_amax * (S_enc * (1/6)), FLT_MAX))                (cvt rn satfinite)
//   inv           = fminf(1 / (float(S_dec_b_fp8) * S_dec), FLT_MAX)
//   q             = cvt.rn.satfinite.e2m1x2.f32(float(w) * inv)   (mul.f32 rn; element 2k in the low nibble)
// Both copies use the block's one scale, so the columnwise data holds the same codes transposed.  Scales are written
// per row of the compact matrix (all 16 rows of a block carry its value), compact or cuBLAS-GEMM-swizzled
// (swizzle::gemm_swizzled_scale_idx).  Built for sm_103a (the e2m1 cvt is arch-specific).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "ak_l8b_rht_epi.cuh"

namespace {

constexpr int kT = 128;            // tile 128 x 128
constexpr int kLd = kT + 8;        // smem row stride (elements), 16 B of padding
constexpr int kThreads = 256;

__device__ __forceinline__ void atomic_max_nonneg(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

union V8 {
  uint4 u;
  __nv_bfloat16 h[8];
};

// ---- amax (TE amax_kernel semantics: max |x|, NaN ignored) ------------------------------------------------------
__global__ void __launch_bounds__(512) amax_bf16_kernel(const __nv_bfloat16* __restrict__ x, int64_t n_vec,
                                                        float* __restrict__ amax) {
  float m = 0.f;
  for (int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; v < n_vec; v += (int64_t)gridDim.x * blockDim.x) {
    V8 a;
    a.u = __ldcs(reinterpret_cast<const uint4*>(x) + v);
#pragma unroll
    for (int e = 0; e < 8; ++e) m = fmaxf(m, fabsf(__bfloat162float(a.h[e])));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  __shared__ float red[16];
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float r = 0.f;
    for (int w = 0; w < (int)(blockDim.x / 32); ++w) r = fmaxf(r, red[w]);
    atomic_max_nonneg(amax, r);
  }
}

// ---- TE's scale arithmetic ------------------------------------------------------------------------------------
__device__ __forceinline__ float global_encode_scale(const float global_amax) {
  constexpr float fp8_max = 448.0f;
  constexpr float fp4_max = 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  global_encode_scale = fminf(global_encode_scale, 3.402823466e+38f);
  if (global_amax == 0.0f || global_encode_scale == 0.0f) return 1.0f;
  return global_encode_scale;
}

__device__ __forceinline__ __nv_fp8_e4m3 decoding_scale(const float block_amax, const float S_enc) {
  constexpr float fp4_max = 6.0f;
  constexpr float fp4_max_inv = 1.0f / fp4_max;
  const float S_dec_b = block_amax * (S_enc * fp4_max_inv);
  return static_cast<__nv_fp8_e4m3>(fminf(S_dec_b, 3.402823466e+38f));
}

// two fp32 -> one byte of two e2m1 (first argument in the HIGH nibble, as TE's cvt with {v1, v0} swapped operands)
__device__ __forceinline__ uint8_t cvt_e2m1x2(float hi, float lo) {
  uint16_t out;
  asm volatile(
      "{\n\t.reg .b8 f;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f, %1, %2;\n\t"
      "mov.b16 %0, {f, f};\n\t}"
      : "=h"(out)
      : "f"(hi), "f"(lo));
  return (uint8_t)out;
}

__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// swizzle::gemm_swizzled_scale_idx (cast/nvfp4/...): (row, col) of the compact scale matrix -> swizzled offset
__device__ __forceinline__ size_t gemm_swizzled_scale_idx(size_t row_idx, size_t col_idx, size_t col_length) {
  const size_t rb = row_idx / 128;
  const size_t rem = row_idx % 128;
  const size_t d4 = rem / 32;
  const size_t d3 = rem % 32;
  const size_t cbg = col_idx / 4;
  const size_t d5 = col_idx % 4;
  const size_t cbg_cnt = (col_length + 3) / 4;
  return ((rb * cbg_cnt + cbg) * 32 + d3) * 16 + d4 * 4 + d5;
}

// ---- 2D quantize: one 128 x 128 tile per CTA --------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads)
    quantize_2d_kernel(const __nv_bfloat16* __restrict__ w, int64_t N, int64_t K, const float* __restrict__ amax_ptr,
                       uint8_t* __restrict__ row_data, uint8_t* __restrict__ row_scales, int64_t row_scale_stride,
                       uint8_t* __restrict__ col_data, uint8_t* __restrict__ col_scales, int64_t col_scale_stride,
                       int do_row, int do_col, int swizzled) {
  __shared__ __align__(16) __nv_bfloat16 sh[kT * kLd];
  __shared__ float sh_inv[8][8];
  __shared__ uint8_t sh_sf[8][8];
  const int64_t r0 = (int64_t)blockIdx.y * kT, c0 = (int64_t)blockIdx.x * kT;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  // load the tile: 128 rows x 16 vectors
#pragma unroll
  for (int i = 0; i < (kT * kT / 8) / kThreads; ++i) {
    const int v = tid + i * kThreads;
    const int r = v / (kT / 8), c8 = (v % (kT / 8)) * 8;
    *reinterpret_cast<uint4*>(sh + r * kLd + c8) =
        __ldcs(reinterpret_cast<const uint4*>(w + (r0 + r) * K + c0 + c8));
  }
  const float S_enc = global_encode_scale(*amax_ptr);
  const float S_dec = 1.0 / S_enc;
  __syncthreads();
  // block amaxes: warp w handles block row w (8 blocks of 16 x 16); lane l covers rows 2(l/16).. of block (l%16)/2 ...
  // simple form: each lane takes 8 of the 256 elements of each block, then an xor-max over the warp
#pragma unroll
  for (int b = 0; b < 8; ++b) {
    const int br = warp, bc = b;
    float m = 0.f;
    // 256 elements = 16 rows x 16 cols; lane covers row (lane / 2), columns 8 * (lane % 2) .. + 8
    const int rr = br * 16 + lane / 2, cc = bc * 16 + 8 * (lane % 2);
    V8 a;
    a.u = *reinterpret_cast<const uint4*>(sh + rr * kLd + cc);
#pragma unroll
    for (int e = 0; e < 8; ++e) m = fmaxf(m, fabsf(__bfloat162float(a.h[e])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) {
      const __nv_fp8_e4m3 sf = decoding_scale(m, S_enc);
      const float inv = fminf(1.0f / (static_cast<float>(sf) * S_dec), 3.402823466e+38f);
      sh_inv[br][bc] = inv;
      sh_sf[br][bc] = sf.__x;
    }
  }
  __syncthreads();
  // scales: 128 rows x 8 blocks per copy
  for (int s = tid; s < kT * 8; s += kThreads) {
    const int r = s / 8, b = s % 8;
    if (do_row) {
      const size_t gr = r0 + r, gc = c0 / 16 + b;
      const size_t idx = swizzled ? gemm_swizzled_scale_idx(gr, gc, K / 16) : gr * row_scale_stride + gc;
      row_scales[idx] = sh_sf[r / 16][b];
    }
    if (do_col) {
      const size_t gr = c0 + r, gc = r0 / 16 + b;     // columnwise: row = W column, col = W row block
      const size_t idx = swizzled ? gemm_swizzled_scale_idx(gr, gc, N / 16) : gr * col_scale_stride + gc;
      col_scales[idx] = sh_sf[b][r / 16];
    }
  }
  // rowwise data: 128 rows x 8 segments of 16 elements -> 8 bytes each
  if (do_row) {
    for (int s = tid; s < kT * 8; s += kThreads) {
      const int r = s / 8, b = s % 8;
      const float inv = sh_inv[r / 16][b];
      uint8_t bytes[8];
      const __nv_bfloat16* src = sh + r * kLd + b * 16;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float x0 = mul_rn(__bfloat162float(src[2 * k]), inv);
        const float x1 = mul_rn(__bfloat162float(src[2 * k + 1]), inv);
        bytes[k] = cvt_e2m1x2(x1, x0);
      }
      *reinterpret_cast<uint2*>(row_data + (r0 + r) * (K / 2) + (c0 + b * 16) / 2) = *reinterpret_cast<uint2*>(bytes);
    }
  }
  // columnwise data: 128 W-columns x 8 segments of 16 W-rows
  if (do_col) {
    for (int s = tid; s < kT * 8; s += kThreads) {
      const int c = s / 8, b = s % 8;
      const float inv = sh_inv[b][c / 16];
      uint8_t bytes[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float x0 = mul_rn(__bfloat162float(sh[(b * 16 + 2 * k) * kLd + c]), inv);
        const float x1 = mul_rn(__bfloat162float(sh[(b * 16 + 2 * k + 1) * kLd + c]), inv);
        bytes[k] = cvt_e2m1x2(x1, x0);
      }
      *reinterpret_cast<uint2*>(col_data + (c0 + c) * (N / 2) + (r0 + b * 16) / 2) = *reinterpret_cast<uint2*>(bytes);
    }
  }
}


// ---- v2 (2026-10-03): the same arithmetic.  Threads 0-127 each own one tile ROW (16-byte vector reads of its 128
// values, 64 output bytes, the row's 8 scale bytes as two 4-byte words); threads 128-255 each own one tile COLUMN (the
// warp reads 32 consecutive columns of a row: no bank conflict; v1 read one column at a 16-row stride, 8-way
// conflicted).  The tile arrives by cp.async. ---------------------------------------------------------------------
__device__ __forceinline__ void cp_async16w(void* smem, const void* gmem) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

__global__ void __launch_bounds__(kThreads)
    quantize_2d_kernel_v2(const __nv_bfloat16* __restrict__ w, int64_t N, int64_t K, const float* __restrict__ amax_ptr,
                          uint8_t* __restrict__ row_data, uint8_t* __restrict__ row_scales, int64_t row_scale_stride,
                          uint8_t* __restrict__ col_data, uint8_t* __restrict__ col_scales, int64_t col_scale_stride,
                          int do_row, int do_col, int swizzled) {
  __shared__ __align__(16) __nv_bfloat16 sh[kT * kLd];
  __shared__ float sh_inv[8][8];
  __shared__ __align__(4) uint8_t sh_sf[8][8];
  const int64_t r0 = (int64_t)blockIdx.y * kT, c0 = (int64_t)blockIdx.x * kT;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
#pragma unroll
  for (int i = 0; i < (kT * kT / 8) / kThreads; ++i) {
    const int v = tid + i * kThreads;
    const int r = v / (kT / 8), c8 = (v % (kT / 8)) * 8;
    cp_async16w(sh + r * kLd + c8, w + (r0 + r) * K + c0 + c8);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  const float S_enc = global_encode_scale(*amax_ptr);
  const float S_dec = 1.0 / S_enc;
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
#pragma unroll
  for (int b = 0; b < 8; ++b) {
    const int br = warp, bc = b;
    float m = 0.f;
    const int rr = br * 16 + lane / 2, cc = bc * 16 + 8 * (lane % 2);
    V8 a;
    a.u = *reinterpret_cast<const uint4*>(sh + rr * kLd + cc);
#pragma unroll
    for (int e = 0; e < 8; ++e) m = fmaxf(m, fabsf(__bfloat162float(a.h[e])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) {
      const __nv_fp8_e4m3 sf = decoding_scale(m, S_enc);
      const float inv = fminf(1.0f / (static_cast<float>(sf) * S_dec), 3.402823466e+38f);
      sh_inv[br][bc] = inv;
      sh_sf[br][bc] = sf.__x;
    }
  }
  __syncthreads();
  if (tid < kT) {
    // ---- row r of the tile: 8 blocks of 16 -> 64 bytes; scales: the 8 blocks of row-block r / 16
    if (do_row) {
      const int r = tid;
      uint32_t outw[16];
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const float inv = sh_inv[r / 16][b];
        V8 a0, a1;
        a0.u = *reinterpret_cast<const uint4*>(sh + r * kLd + b * 16);
        a1.u = *reinterpret_cast<const uint4*>(sh + r * kLd + b * 16 + 8);
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          lo |= (uint32_t)cvt_e2m1x2(mul_rn(__bfloat162float(a0.h[2 * k + 1]), inv),
                                     mul_rn(__bfloat162float(a0.h[2 * k]), inv)) << (8 * k);
          hi |= (uint32_t)cvt_e2m1x2(mul_rn(__bfloat162float(a1.h[2 * k + 1]), inv),
                                     mul_rn(__bfloat162float(a1.h[2 * k]), inv)) << (8 * k);
        }
        outw[2 * b] = lo;
        outw[2 * b + 1] = hi;
      }
      uint4* dst = reinterpret_cast<uint4*>(row_data + (r0 + r) * (K / 2) + c0 / 2);
#pragma unroll
      for (int q = 0; q < 4; ++q) dst[q] = make_uint4(outw[4 * q], outw[4 * q + 1], outw[4 * q + 2], outw[4 * q + 3]);
      const size_t gr = r0 + r;
#pragma unroll
      for (int g4 = 0; g4 < 2; ++g4) {
        const size_t gc = c0 / 16 + 4 * g4;
        const size_t idx = swizzled ? gemm_swizzled_scale_idx(gr, gc, K / 16) : gr * row_scale_stride + gc;
        *reinterpret_cast<uint32_t*>(row_scales + idx) = *reinterpret_cast<const uint32_t*>(&sh_sf[r / 16][4 * g4]);
      }
    }
  } else {
    // ---- column c of the tile (row c0 + c of the columnwise copy): 8 blocks of 16 W-rows -> 64 bytes
    if (do_col) {
      const int c = tid - kT;
      uint32_t outw[16];
      uint8_t sfc[8];
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const float inv = sh_inv[b][c / 16];
        sfc[b] = sh_sf[b][c / 16];
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float x0 = mul_rn(__bfloat162float(sh[(b * 16 + 2 * k) * kLd + c]), inv);
          const float x1 = mul_rn(__bfloat162float(sh[(b * 16 + 2 * k + 1) * kLd + c]), inv);
          const float y0 = mul_rn(__bfloat162float(sh[(b * 16 + 8 + 2 * k) * kLd + c]), inv);
          const float y1 = mul_rn(__bfloat162float(sh[(b * 16 + 8 + 2 * k + 1) * kLd + c]), inv);
          lo |= (uint32_t)cvt_e2m1x2(x1, x0) << (8 * k);
          hi |= (uint32_t)cvt_e2m1x2(y1, y0) << (8 * k);
        }
        outw[2 * b] = lo;
        outw[2 * b + 1] = hi;
      }
      uint4* dst = reinterpret_cast<uint4*>(col_data + (c0 + c) * (N / 2) + r0 / 2);
#pragma unroll
      for (int q = 0; q < 4; ++q) dst[q] = make_uint4(outw[4 * q], outw[4 * q + 1], outw[4 * q + 2], outw[4 * q + 3]);
      const size_t gr = c0 + c;
#pragma unroll
      for (int g4 = 0; g4 < 2; ++g4) {
        const size_t gc = r0 / 16 + 4 * g4;
        const size_t idx = swizzled ? gemm_swizzled_scale_idx(gr, gc, N / 16) : gr * col_scale_stride + gc;
        const uint32_t word = (uint32_t)sfc[4 * g4] | ((uint32_t)sfc[4 * g4 + 1] << 8) |
                              ((uint32_t)sfc[4 * g4 + 2] << 16) | ((uint32_t)sfc[4 * g4 + 3] << 24);
        *reinterpret_cast<uint32_t*>(col_scales + idx) = word;
      }
    }
  }
}


// ---- v3 (2026-10-03): the same arithmetic, pipelined: persistent CTAs (4 per SM) over 64 x 128 tiles with a 3-stage
// cp.async ring (two tiles prefetched); per tile all 8 warps compute the 32 block scales (16 x 16), then warps 0-3 cast
// the row-wise copy (thread = row, 64 columns, 4 blocks) while warps 4-7 cast the column-wise copy (thread = column,
// 64 rows, 4 blocks). ---------------------------------------------------------------------------------------------
constexpr int kR3 = 64;                      // rows per tile
constexpr int kStagesW = 3;
constexpr size_t kSmemW3 = sizeof(__nv_bfloat16) * kStagesW * kR3 * kLd;

__global__ void __launch_bounds__(kThreads, 4)
    quantize_2d_kernel_v3(const __nv_bfloat16* __restrict__ w, int64_t N, int64_t K, const float* __restrict__ amax_ptr,
                          uint8_t* __restrict__ row_data, uint8_t* __restrict__ row_scales, int64_t row_scale_stride,
                          uint8_t* __restrict__ col_data, uint8_t* __restrict__ col_scales, int64_t col_scale_stride,
                          int do_row, int do_col, int swizzled) {
  extern __shared__ __align__(128) unsigned char smem_w[];
  __nv_bfloat16* const sh0 = reinterpret_cast<__nv_bfloat16*>(smem_w);   // [kStagesW][64][kLd]
  __shared__ float sh_inv[4][8];
  __shared__ __align__(4) uint8_t sh_sf[4][8];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int64_t ctiles = K / kT, n_tiles = ctiles * (N / kR3);
  const float S_enc = global_encode_scale(*amax_ptr);
  const float S_dec = 1.0 / S_enc;

  auto load_tile = [&](int64_t tile, __nv_bfloat16* dst) {
    if (tile < n_tiles) {
      const int64_t ct = tile % ctiles, rt = tile / ctiles;
      const __nv_bfloat16* src = w + (rt * kR3) * K + ct * kT;
#pragma unroll
      for (int i = 0; i < (kR3 * kT / 8) / kThreads; ++i) {
        const int v = tid + i * kThreads, r = v >> 4, c8 = (v & 15) * 8;
        cp_async16w(dst + r * kLd + c8, src + r * K + c8);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
#pragma unroll
  for (int j = 0; j < kStagesW - 1; ++j) load_tile(blockIdx.x + (int64_t)j * gridDim.x, sh0 + j * kR3 * kLd);

  int j = 0;
  for (int64_t tile = blockIdx.x; tile < n_tiles; tile += gridDim.x, ++j) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kStagesW - 2));
    __syncthreads();   // this tile landed; the previous tile (its stage, sh_inv, sh_sf) finished by everyone
    load_tile(tile + (int64_t)(kStagesW - 1) * gridDim.x, sh0 + ((j + kStagesW - 1) % kStagesW) * kR3 * kLd);
    const __nv_bfloat16* sh = sh0 + (j % kStagesW) * kR3 * kLd;
    const int64_t ct = tile % ctiles, rt = tile / ctiles;
    const int64_t r0 = rt * kR3, c0 = ct * kT;
    // ---- 32 block scales: warp w takes blocks (w / 2, 4 (w % 2) .. 4 (w % 2) + 3); lane: row lane / 2, 8 columns
#pragma unroll
    for (int bb = 0; bb < 4; ++bb) {
      const int br = warp / 2, bc = 4 * (warp % 2) + bb;
      const int rr = br * 16 + lane / 2, cc = bc * 16 + 8 * (lane % 2);
      V8 a;
      a.u = *reinterpret_cast<const uint4*>(sh + rr * kLd + cc);
      float m = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) m = fmaxf(m, fabsf(__bfloat162float(a.h[e])));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      if (lane == 0) {
        const __nv_fp8_e4m3 sf = decoding_scale(m, S_enc);
        sh_inv[br][bc] = fminf(1.0f / (static_cast<float>(sf) * S_dec), 3.402823466e+38f);
        sh_sf[br][bc] = sf.__x;
      }
    }
    __syncthreads();
    if (tid < 128) {
      if (do_row) {   // ---- row r of the tile, columns 64 hf .. 64 hf + 63
        const int r = tid >> 1, hf = tid & 1;
        uint32_t outw[8];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int bc = 4 * hf + b;
          const float inv = sh_inv[r / 16][bc];
          V8 a0, a1;
          a0.u = *reinterpret_cast<const uint4*>(sh + r * kLd + bc * 16);
          a1.u = *reinterpret_cast<const uint4*>(sh + r * kLd + bc * 16 + 8);
          uint32_t lo = 0, hi = 0;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            lo |= (uint32_t)cvt_e2m1x2(mul_rn(__bfloat162float(a0.h[2 * k + 1]), inv),
                                       mul_rn(__bfloat162float(a0.h[2 * k]), inv)) << (8 * k);
            hi |= (uint32_t)cvt_e2m1x2(mul_rn(__bfloat162float(a1.h[2 * k + 1]), inv),
                                       mul_rn(__bfloat162float(a1.h[2 * k]), inv)) << (8 * k);
          }
          outw[2 * b] = lo;
          outw[2 * b + 1] = hi;
        }
        uint4* dst = reinterpret_cast<uint4*>(row_data + (r0 + r) * (K / 2) + (c0 + 64 * hf) / 2);
        dst[0] = make_uint4(outw[0], outw[1], outw[2], outw[3]);
        dst[1] = make_uint4(outw[4], outw[5], outw[6], outw[7]);
        const size_t gr = r0 + r, gc = c0 / 16 + 4 * hf;
        const size_t idx = swizzled ? gemm_swizzled_scale_idx(gr, gc, K / 16) : gr * row_scale_stride + gc;
        *reinterpret_cast<uint32_t*>(row_scales + idx) = *reinterpret_cast<const uint32_t*>(&sh_sf[r / 16][4 * hf]);
      }
    } else if (do_col) {   // ---- column c of the tile (row c0 + c of the column-wise copy), the tile's 64 rows
      const int c = tid - 128;
      uint32_t outw[8];
      uint32_t sfw = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const float inv = sh_inv[b][c / 16];
        sfw |= (uint32_t)sh_sf[b][c / 16] << (8 * b);
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float x0 = mul_rn(__bfloat162float(sh[(b * 16 + 2 * k) * kLd + c]), inv);
          const float x1 = mul_rn(__bfloat162float(sh[(b * 16 + 2 * k + 1) * kLd + c]), inv);
          const float y0 = mul_rn(__bfloat162float(sh[(b * 16 + 8 + 2 * k) * kLd + c]), inv);
          const float y1 = mul_rn(__bfloat162float(sh[(b * 16 + 8 + 2 * k + 1) * kLd + c]), inv);
          lo |= (uint32_t)cvt_e2m1x2(x1, x0) << (8 * k);
          hi |= (uint32_t)cvt_e2m1x2(y1, y0) << (8 * k);
        }
        outw[2 * b] = lo;
        outw[2 * b + 1] = hi;
      }
      uint4* dst = reinterpret_cast<uint4*>(col_data + (c0 + c) * (N / 2) + r0 / 2);
      dst[0] = make_uint4(outw[0], outw[1], outw[2], outw[3]);
      dst[1] = make_uint4(outw[4], outw[5], outw[6], outw[7]);
      const size_t gr = c0 + c, gc = r0 / 16;
      const size_t idx = swizzled ? gemm_swizzled_scale_idx(gr, gc, N / 16) : gr * col_scale_stride + gc;
      *reinterpret_cast<uint32_t*>(col_scales + idx) = sfw;
    }
  }
}

// The weights' amaxes from the optimizer's shards (2026-10-05): one CTA per chunk of a bf16 shard (any alignment), the
// max |x| of the chunk into out[slot] by an integer atomic max.  TE's amax_kernel semantics (NaN ignored as fmaxf does;
// no flush to zero: max.f32 and a sign-bit mask, whatever the build flags).
__global__ void __launch_bounds__(256) multi_amax_bf16_kernel(const int64_t* __restrict__ cptr,
                                                              const int32_t* __restrict__ clen,
                                                              const int32_t* __restrict__ cslot, float* __restrict__ out) {
  const int c = blockIdx.x;
  const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(cptr[c]);
  const int n = clen[c];
  const int head = min(n, (int)(((16u - (unsigned)(reinterpret_cast<uintptr_t>(p) & 15u)) & 15u) / 2u));
  float m = 0.f;
  for (int i = threadIdx.x; i < head; i += blockDim.x)
    m = ak_rht::fmax_noftz(m, ak_rht::fabs_noftz(__bfloat162float(p[i])));
  const int nv = (n - head) / 8;
  const uint4* pv = reinterpret_cast<const uint4*>(p + head);
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    V8 a;
    a.u = __ldcs(pv + i);
#pragma unroll
    for (int e = 0; e < 8; ++e) m = ak_rht::fmax_noftz(m, ak_rht::fabs_noftz(__bfloat162float(a.h[e])));
  }
  for (int i = head + nv * 8 + threadIdx.x; i < n; i += blockDim.x)
    m = ak_rht::fmax_noftz(m, ak_rht::fabs_noftz(__bfloat162float(p[i])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = ak_rht::fmax_noftz(m, __shfl_xor_sync(0xffffffffu, m, o));
  __shared__ float red[8];
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float r = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) r = ak_rht::fmax_noftz(r, red[w]);
    atomicMax(reinterpret_cast<int*>(out + cslot[c]), __float_as_int(r));
  }
}

void check_bf16(const at::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), n, ": contiguous CUDA bf16");
}

}  // namespace

// amax_bf16(x, out[1] f32): out = max |x| (out zeroed here)
void amax_bf16(const at::Tensor& x, at::Tensor& out) {
  check_bf16(x, "x");
  TORCH_CHECK(x.numel() % 8 == 0, "numel % 8");
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr(), 0, sizeof(float), stream));
  const int64_t n_vec = x.numel() / 8;
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t blocks = std::min<int64_t>((n_vec + 511) / 512, (int64_t)sms * 8);
  amax_bf16_kernel<<<(unsigned)blocks, 512, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), n_vec,
                                                         out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// multi_amax_bf16(chunk_ptr int64 [C] (device addresses of bf16 data), chunk_len int32 [C], chunk_slot int32 [C],
// out f32 [S]): out[s] = max |x| over the chunks of slot s (out zeroed here)
void multi_amax_bf16(const at::Tensor& cptr, const at::Tensor& clen, const at::Tensor& cslot, at::Tensor& out) {
  TORCH_CHECK(cptr.is_cuda() && cptr.scalar_type() == at::kLong && cptr.is_contiguous(), "chunk_ptr: int64 CUDA");
  TORCH_CHECK(clen.is_cuda() && clen.scalar_type() == at::kInt && clen.is_contiguous() && clen.numel() == cptr.numel(),
              "chunk_len: int32 CUDA [C]");
  TORCH_CHECK(cslot.is_cuda() && cslot.scalar_type() == at::kInt && cslot.is_contiguous() &&
                  cslot.numel() == cptr.numel(), "chunk_slot: int32 CUDA [C]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous(), "out: f32 CUDA");
  const at::cuda::CUDAGuard guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr(), 0, sizeof(float) * out.numel(), stream));
  const int64_t C = cptr.numel();
  TORCH_CHECK(C <= INT32_MAX, "too many chunks");
  if (C == 0) return;
  multi_amax_bf16_kernel<<<(unsigned)C, 256, 0, stream>>>(cptr.data_ptr<int64_t>(), clen.data_ptr<int32_t>(),
                                                         cslot.data_ptr<int32_t>(), out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// quantize_2d(w [N, K] bf16, amax [1] f32, row_data, row_scales, col_data, col_scales, do_row, do_col, swizzled)
void quantize_2d(const at::Tensor& w, const at::Tensor& amax, at::Tensor& row_data, at::Tensor& row_scales,
                 at::Tensor& col_data, at::Tensor& col_scales, bool do_row, bool do_col, bool swizzled,
                 int64_t version) {
  check_bf16(w, "w");
  TORCH_CHECK(w.dim() == 2 && w.size(0) % kT == 0 && w.size(1) % kT == 0, "w: [N, K] with N, K % 128 == 0");
  const int64_t N = w.size(0), K = w.size(1);
  const at::cuda::CUDAGuard guard(w.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)(K / kT), (unsigned)(N / kT));
  if (version == 3) {
    static bool attr = false;
    static int sms = 0;
    if (!attr) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(quantize_2d_kernel_v3, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          (int)kSmemW3));
      sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
      attr = true;
    }
    const int64_t n_tiles = (K / kT) * (N / kR3);
    quantize_2d_kernel_v3<<<(unsigned)std::min<int64_t>(n_tiles, (int64_t)sms * 4), kThreads, kSmemW3, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), N, K, amax.data_ptr<float>(),
        do_row ? reinterpret_cast<uint8_t*>(row_data.data_ptr()) : nullptr,
        do_row ? reinterpret_cast<uint8_t*>(row_scales.data_ptr()) : nullptr, do_row ? row_scales.size(-1) : 0,
        do_col ? reinterpret_cast<uint8_t*>(col_data.data_ptr()) : nullptr,
        do_col ? reinterpret_cast<uint8_t*>(col_scales.data_ptr()) : nullptr, do_col ? col_scales.size(-1) : 0,
        do_row ? 1 : 0, do_col ? 1 : 0, swizzled ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  auto kern = version == 2 ? quantize_2d_kernel_v2 : quantize_2d_kernel;
  kern<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), N, K, amax.data_ptr<float>(),
      do_row ? reinterpret_cast<uint8_t*>(row_data.data_ptr()) : nullptr,
      do_row ? reinterpret_cast<uint8_t*>(row_scales.data_ptr()) : nullptr, do_row ? row_scales.size(-1) : 0,
      do_col ? reinterpret_cast<uint8_t*>(col_data.data_ptr()) : nullptr,
      do_col ? reinterpret_cast<uint8_t*>(col_scales.data_ptr()) : nullptr, do_col ? col_scales.size(-1) : 0,
      do_row ? 1 : 0, do_col ? 1 : 0, swizzled ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("amax_bf16", &amax_bf16, "max |x| over a bf16 tensor (TE amax_kernel semantics)");
  m.def("multi_amax_bf16", &multi_amax_bf16, "per-slot max |x| over chunks of bf16 shards (TE amax_kernel semantics)",
        py::arg("chunk_ptr"), py::arg("chunk_len"), py::arg("chunk_slot"), py::arg("out"));
  m.def("quantize_2d", &quantize_2d, "NVFP4 2D (16x16) quantization, TE arithmetic", py::arg("w"), py::arg("amax"),
        py::arg("row_data"), py::arg("row_scales"), py::arg("col_data"), py::arg("col_scales"), py::arg("do_row"),
        py::arg("do_col"), py::arg("swizzled"), py::arg("version") = 1);
}
