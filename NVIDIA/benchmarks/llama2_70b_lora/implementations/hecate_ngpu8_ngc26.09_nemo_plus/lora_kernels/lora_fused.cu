/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

// Fused LoRA branch kernels for the LoRA benchmark (rank 16, dropout before LoRA A).
//
// Forward. The kernel makes the dropout mask bits and computes:
//   h[t, k] = s * sum_j (mask * x)[t, j] * A[k, j]  (bf16 h).
// Backward (a). From dY:
//   dh[t, k] = scale * sum_j dY[t, j] * B[j, k];  dB += scale * dY^T h.
// Backward (b). From dh and the saved mask:
//   dx[t, j] = bf16(mask * s * sum_k dh[t, k] * A[k, j]);  dA += s * dh^T (mask * x).
//
// s is the dropout scale 1 / (1 - p). For an NVFP4 input, s also holds the tensor scale
// amax / (6 * 448). The staged NVFP4 tile is the exact product fp4 * e4m3 (it fits in bf16).
// The kernels apply s in the fp32 reduce of the partials, not to each element. The results
// differ from the TE path by these fp32 rounding points (bf16 noise). The dropout masks are
// equal to the TE masks (same Philox scheme) unless the fast RNG is on.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRank = 16;
constexpr int kChunk = 16;           // elements of one RNG chunk
constexpr int kCols = 512;           // columns of one block
constexpr int kChunksPerRow = kCols / kChunk;  // 32: one lane for each chunk of a block row
constexpr int kColsPad = kCols + 8;  // smem row stride (elements): conflict-free ldmatrix
constexpr int kSmallPad = kRank + 8; // smem row stride of the 16-wide tiles
constexpr int kMaskPad = kChunksPerRow + 2;  // smem row stride of the mask tile (odd word count)
constexpr int kSubTokens = 32;       // tokens of one tile (4 rows for each warp)
constexpr int kRows = kSubTokens / kWarps;
constexpr int kBlocksPerSm = 2;      // the grid holds this many blocks for each SM
constexpr int kPrefetchTiles = 2;    // L2 prefetch distance in tiles
constexpr float kNvfp4Factor = 1.0f / (6.0f * 448.0f);
// Modes of the LoRA input x.
constexpr int kInBf16 = 0;   // bf16 [T][N]
constexpr int kInNvfp4 = 1;  // NVFP4 [T][N / 2] bytes with E4M3 block scales and a tensor amax
constexpr int kInFp8 = 2;    // FP8 E4M3 [T][N] bytes with one scale_inv (the attention output O)

using bf16 = __nv_bfloat16;

// ------------------------------------------------------------------------------------------
// Random bits of one chunk. TE: Philox4x32-10 with the cuRAND constants (same mask bits as
// tex.dropout_fwd). Fast: two splitmix64 mixes of the chunk counter.
// ------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t bytewise_less_than(uint32_t a, uint32_t b) {
  uint32_t result = (a | 0x80808080) - (b & 0x7F7F7F7F);
  asm("lop3.b32 %0, %1, %2, %3, 0x4D;\n\t" : "=r"(result) : "r"(a), "r"(b), "r"(result));
  result &= 0x80808080;
  return result;
}

constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

__device__ __forceinline__ uint4 philox_round(const uint4 c, const uint32_t kx, const uint32_t ky) {
  const uint64_t p0 = static_cast<uint64_t>(kPhiloxM0) * c.x;
  const uint64_t p1 = static_cast<uint64_t>(kPhiloxM1) * c.z;
  return make_uint4(static_cast<uint32_t>(p1 >> 32) ^ c.y ^ kx, static_cast<uint32_t>(p1),
                    static_cast<uint32_t>(p0 >> 32) ^ c.w ^ ky, static_cast<uint32_t>(p0));
}

// The bits of curand_init(seed, chunk, offset) + curand4().
__device__ __forceinline__ uint4 philox_chunk(uint64_t seed, uint64_t chunk, uint64_t offset) {
  const uint64_t n = offset >> 2;
  uint4 c = make_uint4(static_cast<uint32_t>(n), static_cast<uint32_t>(n >> 32),
                       static_cast<uint32_t>(chunk), static_cast<uint32_t>(chunk >> 32));
  uint32_t kx = static_cast<uint32_t>(seed);
  uint32_t ky = static_cast<uint32_t>(seed >> 32);
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    c = philox_round(c, kx, ky);
    kx += kPhiloxW0;
    ky += kPhiloxW1;
  }
  return c;
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

template <bool kTeRng>
__device__ __forceinline__ uint4 chunk_random_bits(uint64_t seed, uint64_t offset, uint64_t chunk) {
  if constexpr (kTeRng) {
    return philox_chunk(seed, chunk, offset);
  } else {
    const uint64_t key = seed ^ (offset * 0xD1B54A32D192ED03ull);
    const uint64_t a = splitmix64(key + (chunk * 2) * 0x9E3779B97F4A7C15ull);
    const uint64_t b = splitmix64(key + (chunk * 2 + 1) * 0x9E3779B97F4A7C15ull);
    return make_uint4(static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32), static_cast<uint32_t>(b),
                      static_cast<uint32_t>(b >> 32));
  }
}

// TE dropout: one byte of random bits for each element; keep if byte >= p * 256.
__device__ __forceinline__ uint32_t mask_from_bits(const uint4 bits, uint32_t drop_prob) {
  uint32_t result = bytewise_less_than(bits.x, drop_prob);
  result = (result >> 2) | bytewise_less_than(bits.y, drop_prob);
  result = (result >> 2) | bytewise_less_than(bits.z, drop_prob);
  result = (result >> 2) | bytewise_less_than(bits.w, drop_prob);
  result |= result >> 17;
  result = ~result;
  return result & 0xFFFFu;
}

// ------------------------------------------------------------------------------------------
// Dropout mask as an and-mask on bf16 pairs, and the NVFP4 dequantize.
// ------------------------------------------------------------------------------------------

// Bits 0 and 1 of `bits` -> 0xFFFF for the low and the high bf16 of a pair.
__device__ __forceinline__ uint32_t pair_and_mask(uint32_t bits) {
  return (bits & 1u) * 0xFFFFu + (bits & 2u) * 0x7FFF8000u;
}

template <int kPairs>
__device__ __forceinline__ void apply_mask(uint32_t *pairs, uint32_t mask) {
#pragma unroll
  for (int i = 0; i < kPairs; ++i) {
    pairs[i] &= pair_and_mask(mask >> (2 * i));
  }
}

// Two FP4 (E2M1) values of one byte to f16x2 (hardware cvt, exact).
__device__ __forceinline__ __half2 fp4x2_to_half2(uint32_t byte) {
  uint32_t h2;
  asm("{\n\t.reg .b8 b;\n\tcvt.u8.u16 b, %1;\n\tcvt.rn.f16x2.e2m1x2 %0, b;\n\t}"
      : "=r"(h2)
      : "h"(static_cast<uint16_t>(byte)));
  return *reinterpret_cast<const __half2 *>(&h2);
}

__device__ __forceinline__ uint32_t swizzled_scale_idx(uint32_t i, uint32_t j, uint32_t num_tiles_x) {
  uint32_t idx = ((i / 128u) * num_tiles_x + j / 4u) * 512u;
  const uint32_t in_y = i % 128u;
  idx += (in_y % 32u) * 16u + (in_y / 32u) * 4u + j % 4u;
  return idx;
}

struct InputDesc {
  const uint8_t *data;    // NVFP4: T x N/2 bytes; bf16: T x N elements
  const uint8_t *scales;  // NVFP4 block scales (E4M3)
  const float *amax;      // NVFP4 tensor amax (used by the reduce)
  const float *scale_inv; // FP8 dequantize scale
  uint32_t scale_stride;  // plain layout
  uint32_t num_tiles_x;   // swizzled layout
  bool swizzled;
  bool nvfp4;
  bool fp8;
};

// One 16-element NVFP4 chunk (row y, chunk x) to 8 bf16 pairs of the exact product fp4 * e4m3.
__device__ __forceinline__ void dequant_chunk(const InputDesc &in, uint32_t y, uint32_t x, uint2 packed,
                                              uint32_t *pairs) {
  const uint32_t scale_idx = in.swizzled ? swizzled_scale_idx(y, x, in.num_tiles_x) : x + y * in.scale_stride;
  __nv_fp8_e4m3 scale_fp8;
  scale_fp8.__x = in.scales[scale_idx];
  const __half2 scale2 = __float2half2_rn(static_cast<float>(scale_fp8));
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const uint32_t word = k < 4 ? packed.x : packed.y;
    const __half2 product = __hmul2(fp4x2_to_half2(__byte_perm(word, 0, k % 4)), scale2);
    const __nv_bfloat162 pair = __float22bfloat162_rn(__half22float2(product));
    pairs[k] = *reinterpret_cast<const uint32_t *>(&pair);
  }
}

// ------------------------------------------------------------------------------------------
// Tensor core helpers: mma.m16n8k16 bf16, ldmatrix, stmatrix.
// ------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma_16816(float *d, const uint32_t *a, const uint32_t *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ldsm_x4(uint32_t *r, const void *p) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t *r, const void *p) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2(uint32_t *r, const void *p) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2_trans(uint32_t *r, const void *p) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

// Store four 8x8 b16 matrices (accumulator layout) to smem; lane l gives the row l % 8 of matrix l / 8.
__device__ __forceinline__ void stsm_x4(void *p, const uint32_t *r) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
               :
               : "r"(addr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3])
               : "memory");
}

// A fragment (16 m x 16 k) from a row-major [m][k] tile with row stride s (elements).
__device__ __forceinline__ void frag_a_rowmajor(uint32_t *a, const bf16 *tile, int s, int lane) {
  const int m = lane >> 3, r = lane & 7;
  ldsm_x4(a, tile + (r + 8 * (m & 1)) * s + 8 * (m >> 1));
}

// A fragment (16 m x 16 k) from a [k][m] tile (the storage holds the transpose).
__device__ __forceinline__ void frag_a_trans(uint32_t *a, const bf16 *tile, int s, int lane) {
  const int m = lane >> 3, r = lane & 7;
  ldsm_x4_trans(a, tile + (r + 8 * (m >> 1)) * s + 8 * (m & 1));
}

// B fragment of one n-tile (8 n x 16 k) from an [n][k] tile.
__device__ __forceinline__ void frag_b1_nk(uint32_t *b, const bf16 *tile, int s, int lane) {
  const int m = (lane >> 3) & 1, r = lane & 7;
  ldsm_x2(b, tile + r * s + 8 * m);
}

// B fragments of two n-tiles from a [k][n] tile.
__device__ __forceinline__ void frag_b2_kn(uint32_t *b, const bf16 *tile, int s, int lane) {
  const int m = lane >> 3, r = lane & 7;
  ldsm_x4_trans(b, tile + (r + 8 * (m & 1)) * s + 8 * (m >> 1));
}

// B fragment of one n-tile from a [k][n] tile.
__device__ __forceinline__ void frag_b1_kn(uint32_t *b, const bf16 *tile, int s, int lane) {
  const int m = (lane >> 3) & 1, r = lane & 7;
  ldsm_x2_trans(b, tile + (r + 8 * m) * s);
}

// Bring `bytes` bytes at a 16-byte aligned global address into L2.
__device__ __forceinline__ void prefetch_l2(const void *ptr, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n"
               :
               : "l"(__cvta_generic_to_global(ptr)), "r"(bytes)
               : "memory");
}

// ------------------------------------------------------------------------------------------
// Row staging. A warp stages one 512-column row. Lane i owns chunk i (16 elements).
// bf16 input: lane i loads the 16-byte vectors i and 32 + i (coalesced). The mask bits of a
// vector come from the lane that owns its chunk (a shuffle).
// NVFP4 input: lane i loads the 8 bytes of chunk i and dequantizes them.
// ------------------------------------------------------------------------------------------

template <int kMode>
struct RowData;

template <>
struct RowData<kInNvfp4> {
  uint2 packed;
};

template <>
struct RowData<kInBf16> {
  uint4 v0, v1;
};

template <>
struct RowData<kInFp8> {
  uint4 v;  // the 16 bytes of the chunk of this lane
};

template <int kMode>
__device__ __forceinline__ RowData<kMode> load_row(const InputDesc &in, uint32_t cols, uint32_t y,
                                                    uint32_t col0, int lane) {
  RowData<kMode> d;
  if constexpr (kMode == kInNvfp4) {
    d.packed = *reinterpret_cast<const uint2 *>(in.data + (static_cast<size_t>(y) * cols + col0) / 2 + lane * 8);
  } else if constexpr (kMode == kInFp8) {
    d.v = *reinterpret_cast<const uint4 *>(in.data + static_cast<size_t>(y) * cols + col0 + lane * kChunk);
  } else {
    const uint4 *src = reinterpret_cast<const uint4 *>(in.data + (static_cast<size_t>(y) * cols + col0) * 2);
    d.v0 = src[lane];
    d.v1 = src[kChunksPerRow + lane];
  }
  return d;
}

template <int kMode>
__device__ __forceinline__ void prefetch_rows(const InputDesc &in, uint32_t cols, uint32_t t0, uint32_t col0,
                                              int warp, int lane, RowData<kMode> *rows) {
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    rows[r] = load_row<kMode>(in, cols, t0 + warp * kRows + r, col0, lane);
  }
}

// L2 prefetch of the 32 rows of the tile at t0 (one warp, one row for each lane).
template <int kMode>
__device__ __forceinline__ void prefetch_tile_l2(const InputDesc &in, uint32_t cols, uint32_t t0, uint32_t col0,
                                                 int lane) {
  const size_t elements = static_cast<size_t>(t0 + lane) * cols + col0;
  const size_t row_bytes = kMode == kInNvfp4 ? kCols / 2 : (kMode == kInFp8 ? kCols : kCols * 2);
  const size_t offset = kMode == kInNvfp4 ? elements / 2 : (kMode == kInFp8 ? elements : elements * 2);
  prefetch_l2(in.data + offset, static_cast<uint32_t>(row_bytes));
}

// Write mask * x of one row to smem. mask16 holds the bits of the chunk of this lane.
template <int kMode>
__device__ __forceinline__ void stage_row(const InputDesc &in, uint32_t y, uint32_t col0, int lane,
                                          const RowData<kMode> &d, uint32_t mask16, bf16 *row_smem) {
  if constexpr (kMode == kInNvfp4) {
    uint32_t pairs[8];
    dequant_chunk(in, y, col0 / kChunk + lane, d.packed, pairs);
    apply_mask<8>(pairs, mask16);
    uint4 *dst = reinterpret_cast<uint4 *>(row_smem + lane * kChunk);
    dst[0] = make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
    dst[1] = make_uint4(pairs[4], pairs[5], pairs[6], pairs[7]);
  } else if constexpr (kMode == kInFp8) {
    // The TE dequantize math: bf16(float(e4m3) * scale_inv). Pair k holds elements 2k, 2k + 1.
    const float scale_inv = __ldg(in.scale_inv);
    const uint32_t words[4] = {d.v.x, d.v.y, d.v.z, d.v.w};
    uint32_t pairs[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const auto two = static_cast<__nv_fp8x2_storage_t>(words[k / 2] >> (16 * (k % 2)));
      const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(two, __NV_E4M3);
      const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&raw));
      const __nv_bfloat162 pair = __floats2bfloat162_rn(f.x * scale_inv, f.y * scale_inv);
      pairs[k] = *reinterpret_cast<const uint32_t *>(&pair);
    }
    apply_mask<8>(pairs, mask16);
    uint4 *dst = reinterpret_cast<uint4 *>(row_smem + lane * kChunk);
    dst[0] = make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
    dst[1] = make_uint4(pairs[4], pairs[5], pairs[6], pairs[7]);
  } else {
    // Vector v holds the elements 8v .. 8v + 7: chunk v / 2, mask byte v % 2.
    const uint32_t shift = 8 * (lane & 1);
    const uint32_t m0 = __shfl_sync(0xffffffffu, mask16, lane >> 1) >> shift;
    const uint32_t m1 = __shfl_sync(0xffffffffu, mask16, 16 + (lane >> 1)) >> shift;
    uint32_t pairs[4] = {d.v0.x, d.v0.y, d.v0.z, d.v0.w};
    apply_mask<4>(pairs, m0);
    *reinterpret_cast<uint4 *>(row_smem + lane * 8) = make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
    pairs[0] = d.v1.x; pairs[1] = d.v1.y; pairs[2] = d.v1.z; pairs[3] = d.v1.w;
    apply_mask<4>(pairs, m1);
    *reinterpret_cast<uint4 *>(row_smem + (kChunksPerRow + lane) * 8) =
        make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
  }
}

// Copy a [16][512] weight slice (row stride `stride` elements) to smem with the padded stride.
__device__ __forceinline__ void load_weight_slice(const bf16 *src, uint32_t stride, uint32_t col0, bf16 *dst) {
  for (int idx = threadIdx.x; idx < kRank * kCols / 8; idx += kThreads) {
    const int k = idx / (kCols / 8);
    const int v = idx % (kCols / 8);
    *reinterpret_cast<uint4 *>(dst + k * kColsPad + v * 8) =
        *reinterpret_cast<const uint4 *>(src + static_cast<size_t>(k) * stride + col0 + v * 8);
  }
}

// Copy the 16-wide rows t0 .. t0 + kSubTokens of a [T][16] tensor to smem (64 threads).
__device__ __forceinline__ void load_small_rows(const bf16 *src, uint32_t t0, bf16 *dst) {
  if (threadIdx.x < kSubTokens * 2) {
    const int row = threadIdx.x >> 1;
    const int half = threadIdx.x & 1;
    *reinterpret_cast<uint4 *>(dst + row * kSmallPad + half * 8) =
        *reinterpret_cast<const uint4 *>(src + (static_cast<size_t>(t0) + row) * kRank + half * 8);
  }
}

// Tiles of block group g of G: [g * S / G, (g + 1) * S / G) of the S tiles.
__device__ __forceinline__ int2 group_range(int g, int G, int S) {
  return make_int2((g * S) / G, ((g + 1) * S) / G);
}

// A 32 x 16 tile product in two K halves: the K-half-1 warps put their sums in cs, the
// K-half-0 warps add them. Call with all threads (it has a barrier).
__device__ __forceinline__ void combine_k_halves(float *acc, float *cs, int kh, int tt, int rt, int lane) {
  const int gq = lane >> 2, cq = lane & 3;
  float *tile = cs + (tt * 16 + gq) * kRank + rt * 8 + 2 * cq;
  if (kh == 1) {
    *reinterpret_cast<float2 *>(tile) = make_float2(acc[0], acc[1]);
    *reinterpret_cast<float2 *>(tile + 8 * kRank) = make_float2(acc[2], acc[3]);
  }
  __syncthreads();
  if (kh == 0) {
    const float2 lo = *reinterpret_cast<const float2 *>(tile);
    const float2 hi = *reinterpret_cast<const float2 *>(tile + 8 * kRank);
    acc[0] += lo.x; acc[1] += lo.y; acc[2] += hi.x; acc[3] += hi.y;
  }
}

// 32 x 16 tile (bf16 A operand, K = 512) times a 16-wide B operand, in two K halves
// (warp = token tile x rank tile x K half). Two accumulator chains for each warp.
template <bool kBNk>
__device__ __forceinline__ void tile_dot_k512(const bf16 *a_tile, const bf16 *b_tile, int warp, int lane,
                                              float *acc) {
  const int tt = warp & 1;
  const int rt = (warp >> 1) & 1;
  const int kh = warp >> 2;
  float acc1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < kCols / 32; k += 2) {
    const int ks = kh * (kCols / 32) + k;
    uint32_t a0[4], a1[4], b0[2], b1[2];
    frag_a_rowmajor(a0, a_tile + tt * 16 * kColsPad + ks * 16, kColsPad, lane);
    frag_a_rowmajor(a1, a_tile + tt * 16 * kColsPad + (ks + 1) * 16, kColsPad, lane);
    if constexpr (kBNk) {
      frag_b1_nk(b0, b_tile + rt * 8 * kColsPad + ks * 16, kColsPad, lane);
      frag_b1_nk(b1, b_tile + rt * 8 * kColsPad + (ks + 1) * 16, kColsPad, lane);
    } else {
      frag_b1_kn(b0, b_tile + ks * 16 * kSmallPad + rt * 8, kSmallPad, lane);
      frag_b1_kn(b1, b_tile + (ks + 1) * 16 * kSmallPad + rt * 8, kSmallPad, lane);
    }
    mma_16816(acc, a0, b0);
    mma_16816(acc1, a1, b1);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i] += acc1[i];
}

// Write the 32 x 16 fp32 tile of the K-half-0 warps to part[(t0 + t) * 16 + k].
__device__ __forceinline__ void store_small_tile(float *part, uint32_t t0, const float *acc, int warp, int lane) {
  const int tt = warp & 1;
  const int rt = (warp >> 1) & 1;
  const int gq = lane >> 2, cq = lane & 3;
  float *dst = part + (static_cast<size_t>(t0) + tt * 16 + gq) * kRank + rt * 8 + 2 * cq;
  *reinterpret_cast<float2 *>(dst) = make_float2(acc[0], acc[1]);
  *reinterpret_cast<float2 *>(dst + 8 * kRank) = make_float2(acc[2], acc[3]);
}

// acc[8][4] (16 ranks x 512 cols, 8 n-tiles for each warp) += A^T (16 x 32 tokens) * tile (32 x 512).
__device__ __forceinline__ void rank_outer_product(const bf16 *small_tile, const bf16 *tile, int warp, int lane,
                                                   float (*acc)[4]) {
#pragma unroll
  for (int ks = 0; ks < kSubTokens / 16; ++ks) {
    uint32_t a[4];
    frag_a_trans(a, small_tile + ks * 16 * kSmallPad, kSmallPad, lane);
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      uint32_t b[4];
      frag_b2_kn(b, tile + ks * 16 * kColsPad + (warp * 8 + np * 2) * 8, kColsPad, lane);
      mma_16816(acc[np * 2], a, b);
      mma_16816(acc[np * 2 + 1], a, b + 2);
    }
  }
}

// ------------------------------------------------------------------------------------------
// Forward: mask, dropout, h partials. grid (N / 512, G); each block loops over its tiles.
// ------------------------------------------------------------------------------------------

struct FwdArgs {
  InputDesc in;
  const bf16 *lora_a;    // [16][N]
  uint16_t *mask;        // T * N / 16 chunks
  float *partials;       // [N/512][T][16]
  at::PhiloxCudaState philox;
  uint32_t cols;         // N
  uint32_t tokens;       // T
  uint32_t drop_prob;    // bytewise
};

template <int kMode, bool kTeRng>
__global__ void __launch_bounds__(kThreads, kBlocksPerSm) lora_fwd_kernel(FwdArgs args) {
  __shared__ __align__(16) bf16 xs[kSubTokens * kColsPad];
  __shared__ __align__(16) bf16 as[kRank * kColsPad];
  __shared__ __align__(16) float cs[kSubTokens * kRank];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const uint32_t cb = blockIdx.x;
  const uint32_t col0 = cb * kCols;
  const uint32_t chunks_per_row = args.cols / kChunk;
  const int2 range = group_range(blockIdx.y, gridDim.y, args.tokens / kSubTokens);
  const auto seeds = at::cuda::philox::unpack(args.philox);
  const uint64_t seed = std::get<0>(seeds);
  const uint64_t offset = std::get<1>(seeds);

  if (warp == 0) {
    for (int s = range.x; s < range.x + kPrefetchTiles && s < range.y; ++s) {
      prefetch_tile_l2<kMode>(args.in, args.cols, s * kSubTokens, col0, lane);
    }
  }
  load_weight_slice(args.lora_a, args.cols, col0, as);
  RowData<kMode> rows[kRows];
  prefetch_rows<kMode>(args.in, args.cols, range.x * kSubTokens, col0, warp, lane, rows);

  for (int s = range.x; s < range.y; ++s) {
    const uint32_t t0 = s * kSubTokens;
    if (warp == 0 && s + kPrefetchTiles < range.y) {
      prefetch_tile_l2<kMode>(args.in, args.cols, t0 + kPrefetchTiles * kSubTokens, col0, lane);
    }
    __syncthreads();  // the previous tile no longer needs xs and cs
    uint32_t masks[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const uint64_t chunk = static_cast<uint64_t>(t0 + warp * kRows + r) * chunks_per_row + col0 / kChunk + lane;
      masks[r] = mask_from_bits(chunk_random_bits<kTeRng>(seed, offset, chunk), args.drop_prob);
      args.mask[chunk] = static_cast<uint16_t>(masks[r]);
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int row = warp * kRows + r;
      stage_row<kMode>(args.in, t0 + row, col0, lane, rows[r], masks[r], xs + row * kColsPad);
    }
    if (s + 1 < range.y) {
      prefetch_rows<kMode>(args.in, args.cols, t0 + kSubTokens, col0, warp, lane, rows);
    }
    __syncthreads();

    // h partial (32 tokens x 16 ranks) = (mask * x) tile (32 x 512) * A slice^T.
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    tile_dot_k512<true>(xs, as, warp, lane, acc);
    combine_k_halves(acc, cs, warp >> 2, warp & 1, (warp >> 1) & 1, lane);
    if (warp < 4) {
      store_small_tile(args.partials + static_cast<size_t>(cb) * args.tokens * kRank, t0, acc, warp, lane);
    }
  }
}

// ------------------------------------------------------------------------------------------
// Backward (a): dh and dB from dY. grid (M / 512, G); each block loops over its tiles.
// ------------------------------------------------------------------------------------------

struct DyArgs {
  const bf16 *dy;      // [T][M]
  const bf16 *lora_b;  // [M][16]
  const bf16 *h;       // [T][16]
  float *dh_part;      // [M/512][T][16]
  float *db_part;      // [G][M][16]
  uint32_t out_cols;   // M
  uint32_t tokens;     // T
};

// dY tile [32][512]: 2048 vectors, 8 for each thread.
__device__ __forceinline__ void load_dy_tile(const DyArgs &args, uint32_t t0, uint32_t col0, uint4 *pre) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = threadIdx.x + i * kThreads;
    const int row = idx / (kCols / 8);
    const int v = idx % (kCols / 8);
    pre[i] = *reinterpret_cast<const uint4 *>(args.dy + (static_cast<size_t>(t0) + row) * args.out_cols + col0 + v * 8);
  }
}

__device__ __forceinline__ void store_dy_tile(const uint4 *pre, bf16 *dys) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = threadIdx.x + i * kThreads;
    const int row = idx / (kCols / 8);
    const int v = idx % (kCols / 8);
    *reinterpret_cast<uint4 *>(dys + row * kColsPad + v * 8) = pre[i];
  }
}

__device__ __forceinline__ void prefetch_dy_l2(const DyArgs &args, uint32_t t0, uint32_t col0, int lane) {
  prefetch_l2(args.dy + (static_cast<size_t>(t0) + lane) * args.out_cols + col0, kCols * 2);
}

__global__ void __launch_bounds__(kThreads, kBlocksPerSm) lora_bwd_dy_kernel(DyArgs args) {
  __shared__ __align__(16) bf16 dys[kSubTokens * kColsPad];
  __shared__ __align__(16) bf16 bs[kCols * kSmallPad];      // [j][k]
  __shared__ __align__(16) bf16 hs[kSubTokens * kSmallPad]; // [t][k]
  __shared__ __align__(16) float cs[kSubTokens * kRank];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const uint32_t cb = blockIdx.x;
  const uint32_t col0 = cb * kCols;
  const uint32_t M = args.out_cols;
  const int2 range = group_range(blockIdx.y, gridDim.y, args.tokens / kSubTokens);
  const int gq = lane >> 2, cq = lane & 3;

  if (warp == 0) {
    for (int s = range.x; s < range.x + kPrefetchTiles && s < range.y; ++s) {
      prefetch_dy_l2(args, s * kSubTokens, col0, lane);
    }
  }
  // B slice [512][16]: 1024 16-byte vectors.
  for (int idx = tid; idx < kCols * 2; idx += kThreads) {
    const int j = idx >> 1;
    const int half = idx & 1;
    *reinterpret_cast<uint4 *>(bs + j * kSmallPad + half * 8) =
        *reinterpret_cast<const uint4 *>(args.lora_b + (static_cast<size_t>(col0) + j) * kRank + half * 8);
  }
  float acc_db[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc_db[n][i] = 0.f;
  }
  uint4 pre[8];
  load_dy_tile(args, range.x * kSubTokens, col0, pre);

  for (int s = range.x; s < range.y; ++s) {
    const uint32_t t0 = s * kSubTokens;
    if (warp == 0 && s + kPrefetchTiles < range.y) {
      prefetch_dy_l2(args, t0 + kPrefetchTiles * kSubTokens, col0, lane);
    }
    __syncthreads();  // the previous tile no longer needs dys, hs, cs
    store_dy_tile(pre, dys);
    load_small_rows(args.h, t0, hs);
    if (s + 1 < range.y) {
      load_dy_tile(args, t0 + kSubTokens, col0, pre);
    }
    __syncthreads();

    // dh partial (32 x 16) = dY tile (32 x 512) * B slice (512 x 16).
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    tile_dot_k512<false>(dys, bs, warp, lane, acc);
    combine_k_halves(acc, cs, warp >> 2, warp & 1, (warp >> 1) & 1, lane);
    if (warp < 4) {
      store_small_tile(args.dh_part + static_cast<size_t>(cb) * args.tokens * kRank, t0, acc, warp, lane);
    }
    // dB partial (16 ranks x 512 cols) += h^T (16 x 32 tokens) * dY tile (32 x 512).
    rank_outer_product(hs, dys, warp, lane, acc_db);
  }

  // dB partials [g][col][k].
  float *part = args.db_part + static_cast<size_t>(blockIdx.y) * M * kRank;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const uint32_t col = col0 + (warp * 8 + n) * 8 + 2 * cq;
    part[static_cast<size_t>(col) * kRank + gq] = acc_db[n][0];
    part[static_cast<size_t>(col + 1) * kRank + gq] = acc_db[n][1];
    part[static_cast<size_t>(col) * kRank + gq + 8] = acc_db[n][2];
    part[static_cast<size_t>(col + 1) * kRank + gq + 8] = acc_db[n][3];
  }
}

// ------------------------------------------------------------------------------------------
// Backward (b): dx and dA. grid (N / 512, G); each block loops over its tiles.
// ------------------------------------------------------------------------------------------

struct DxArgs {
  InputDesc in;
  const uint16_t *mask;  // T * N / 16 chunks
  const bf16 *dh;        // [T][16]
  const bf16 *lora_a;    // [16][N]
  bf16 *dx;              // [T][N]
  float *da_part;        // [G][16][N]
  uint32_t cols;         // N
  uint32_t tokens;       // T
  float scale;           // s: dropout scale (times the NVFP4 tensor scale, on the host)
};

template <int kMode>
struct DxPrefetch {
  RowData<kMode> rows[kRows];
  uint32_t masks[kRows];
};

template <int kMode>
__device__ __forceinline__ void prefetch_dx_rows(const DxArgs &args, uint32_t t0, uint32_t col0, int warp, int lane,
                                                 DxPrefetch<kMode> &pre) {
  const uint32_t chunks_per_row = args.cols / kChunk;
  prefetch_rows<kMode>(args.in, args.cols, t0, col0, warp, lane, pre.rows);
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint32_t y = t0 + warp * kRows + r;
    pre.masks[r] = args.mask[static_cast<uint64_t>(y) * chunks_per_row + col0 / kChunk + lane];
  }
}

// dx tile (32 x 512) = s * dh tile (32 x 16) * A slice (16 x 512), masked, bf16, into os.
// Warp = token tile (2) x 16 n-tiles. stmatrix writes the accumulator fragments.
__device__ __forceinline__ void dx_tile_to_smem(const bf16 *dhs, const bf16 *as, const uint16_t *ms, float scale,
                                                int warp, int lane, bf16 *os) {
  const int mt = warp & 1;
  const int nt0 = (warp >> 1) * 16;
  const int gq = lane >> 2, cq = lane & 3;
  uint32_t a[4];
  frag_a_rowmajor(a, dhs + mt * 16 * kSmallPad, kSmallPad, lane);
  // stmatrix: lane l writes row l % 8 of matrix l / 8; matrices = (rows 0-7, 8-15) x (n-tile 0, 1).
  const int st_row = mt * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
  const int st_ntile = lane >> 4;
#pragma unroll
  for (int np = 0; np < 8; ++np) {
    uint32_t b[4];
    frag_b2_kn(b, as + (nt0 + np * 2) * 8, kColsPad, lane);
    float acc0[4] = {0.f, 0.f, 0.f, 0.f};
    float acc1[4] = {0.f, 0.f, 0.f, 0.f};
    mma_16816(acc0, a, b);
    mma_16816(acc1, a, b + 2);
    // Mask bits of this lane: rows gq and gq + 8 of the token tile, chunk nt0 / 2 + np.
    const int chunk = (nt0 + np * 2) / 2;
    const uint32_t m_lo = ms[(mt * 16 + gq) * kMaskPad + chunk];
    const uint32_t m_hi = ms[(mt * 16 + gq + 8) * kMaskPad + chunk];
    uint32_t r[4];
    const float *accs[2] = {acc0, acc1};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int bit = half * 8 + 2 * cq;
      const __nv_bfloat162 lo = __floats2bfloat162_rn(accs[half][0] * scale, accs[half][1] * scale);
      const __nv_bfloat162 hi = __floats2bfloat162_rn(accs[half][2] * scale, accs[half][3] * scale);
      r[half * 2] = *reinterpret_cast<const uint32_t *>(&lo) & pair_and_mask(m_lo >> bit);
      r[half * 2 + 1] = *reinterpret_cast<const uint32_t *>(&hi) & pair_and_mask(m_hi >> bit);
    }
    stsm_x4(os + st_row * kColsPad + (nt0 + np * 2 + st_ntile) * 8, r);
  }
}

template <int kMode>
__global__ void __launch_bounds__(kThreads, kBlocksPerSm) lora_bwd_dx_kernel(DxArgs args) {
  __shared__ __align__(16) bf16 xs[kSubTokens * kColsPad];   // (mask * x) tile [t][j]
  __shared__ __align__(16) bf16 os[kSubTokens * kColsPad];   // dx tile [t][j]
  __shared__ __align__(16) bf16 as[kRank * kColsPad];        // A slice [k][j]
  __shared__ __align__(16) bf16 dhs[kSubTokens * kSmallPad]; // dh tile [t][k]
  __shared__ __align__(16) uint16_t ms[kSubTokens * kMaskPad];  // masks [t][chunk]
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const uint32_t cb = blockIdx.x;
  const uint32_t col0 = cb * kCols;
  const uint32_t N = args.cols;
  const int2 range = group_range(blockIdx.y, gridDim.y, args.tokens / kSubTokens);
  const int gq = lane >> 2, cq = lane & 3;

  if (warp == 0) {
    for (int s = range.x; s < range.x + kPrefetchTiles && s < range.y; ++s) {
      prefetch_tile_l2<kMode>(args.in, N, s * kSubTokens, col0, lane);
    }
  }
  load_weight_slice(args.lora_a, N, col0, as);
  float acc_da[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc_da[n][i] = 0.f;
  }
  DxPrefetch<kMode> pre;
  prefetch_dx_rows<kMode>(args, range.x * kSubTokens, col0, warp, lane, pre);

  for (int s = range.x; s < range.y; ++s) {
    const uint32_t t0 = s * kSubTokens;
    if (warp == 0 && s + kPrefetchTiles < range.y) {
      prefetch_tile_l2<kMode>(args.in, N, t0 + kPrefetchTiles * kSubTokens, col0, lane);
    }
    __syncthreads();  // the previous tile no longer needs xs, os, dhs, ms
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int row = warp * kRows + r;
      ms[row * kMaskPad + lane] = static_cast<uint16_t>(pre.masks[r]);
      stage_row<kMode>(args.in, t0 + row, col0, lane, pre.rows[r], pre.masks[r], xs + row * kColsPad);
    }
    load_small_rows(args.dh, t0, dhs);
    if (s + 1 < range.y) {
      prefetch_dx_rows<kMode>(args, t0 + kSubTokens, col0, warp, lane, pre);
    }
    __syncthreads();

    dx_tile_to_smem(dhs, as, ms, args.scale, warp, lane, os);
    // dA partial (16 ranks x 512 cols) += dh^T (16 x 32 tokens) * (mask * x) tile (32 x 512).
    rank_outer_product(dhs, xs, warp, lane, acc_da);
    __syncthreads();
    // dx tile out: 32 rows x 512 cols = 2048 vectors, 16 bytes for each lane.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = tid + i * kThreads;
      const int row = idx / (kCols / 8);
      const int v = idx % (kCols / 8);
      *reinterpret_cast<uint4 *>(args.dx + (static_cast<size_t>(t0) + row) * N + col0 + v * 8) =
          *reinterpret_cast<const uint4 *>(os + row * kColsPad + v * 8);
    }
  }

  // dA partials [g][k][col].
  float *part = args.da_part + static_cast<size_t>(blockIdx.y) * kRank * N;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const uint32_t col = col0 + (warp * 8 + n) * 8 + 2 * cq;
    *reinterpret_cast<float2 *>(part + static_cast<size_t>(gq) * N + col) = make_float2(acc_da[n][0], acc_da[n][1]);
    *reinterpret_cast<float2 *>(part + static_cast<size_t>(gq + 8) * N + col) = make_float2(acc_da[n][2], acc_da[n][3]);
  }
}

// ------------------------------------------------------------------------------------------
// Reduce: out[i] = bf16(base[i] + s * sum_p part[p * n + i]). Each thread makes 4 elements.
// With an amax pointer, s = scale * amax * kNvfp4Factor. Without it, s = scale.
// ------------------------------------------------------------------------------------------

struct ReduceArgs {
  const float *part;
  const bf16 *base;   // nullptr: no base
  const float *amax;  // nullptr: no tensor scale
  bf16 *out;
  size_t n;
  int num_parts;
  float scale;
};

__global__ void reduce_partials_kernel(ReduceArgs args) {
  const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (i >= args.n) {
    return;
  }
  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int p = 0; p < args.num_parts; ++p) {
    const float4 v = __ldcg(reinterpret_cast<const float4 *>(args.part + p * args.n + i));
    sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
  }
  const float s = args.amax == nullptr ? args.scale : args.scale * (*args.amax * kNvfp4Factor);
  float4 base = make_float4(0.f, 0.f, 0.f, 0.f);
  if (args.base != nullptr) {
    const uint2 raw = *reinterpret_cast<const uint2 *>(args.base + i);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.y));
    base = make_float4(lo.x, lo.y, hi.x, hi.y);
  }
  const __nv_bfloat162 lo = __floats2bfloat162_rn(base.x + s * sum.x, base.y + s * sum.y);
  const __nv_bfloat162 hi = __floats2bfloat162_rn(base.z + s * sum.z, base.w + s * sum.w);
  *reinterpret_cast<uint2 *>(args.out + i) =
      make_uint2(*reinterpret_cast<const uint32_t *>(&lo), *reinterpret_cast<const uint32_t *>(&hi));
}

void reduce_partials(const at::Tensor &part, const at::Tensor *base, const float *amax, at::Tensor &out,
                     int num_parts, float scale) {
  ReduceArgs args{};
  args.part = part.data_ptr<float>();
  args.base = base == nullptr ? nullptr : reinterpret_cast<const bf16 *>(base->data_ptr());
  args.amax = amax;
  args.out = reinterpret_cast<bf16 *>(out.data_ptr());
  args.n = static_cast<size_t>(out.numel());
  args.num_parts = num_parts;
  args.scale = scale;
  const unsigned blocks = static_cast<unsigned>((args.n / 4 + kThreads - 1) / kThreads);
  reduce_partials_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ------------------------------------------------------------------------------------------
// Host side
// ------------------------------------------------------------------------------------------

// Block groups along the tokens: kBlocksPerSm blocks for each SM, at most one for each tile.
int token_groups(int64_t col_blocks, int64_t tiles) {
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t groups = (static_cast<int64_t>(sms) * kBlocksPerSm) / col_blocks;
  return static_cast<int>(std::max<int64_t>(1, std::min(groups, tiles)));
}

uint32_t bytewise_drop_probability(double p) {
  TORCH_CHECK(p >= 0 && p < 1, "invalid dropout probability");
  uint32_t value = static_cast<uint32_t>(std::floor(p * 256));
  value |= value << 8;
  value |= value << 16;
  return value;
}

at::PhiloxCudaState next_philox_state() {
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  return gen->philox_cuda_state(4);  // as TE dropout_fwd
}

InputDesc make_input(const at::Tensor &x, const std::optional<at::Tensor> &scales,
                     const std::optional<at::Tensor> &amax, bool swizzled, int64_t tokens, int64_t cols) {
  InputDesc in{};
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be a contiguous CUDA tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
  in.data = reinterpret_cast<const uint8_t *>(x.data_ptr());
  in.nvfp4 = x.scalar_type() == at::kByte && x.numel() * 2 == tokens * cols;
  in.fp8 = x.scalar_type() == at::kByte && x.numel() == tokens * cols;
  if (in.fp8) {
    // FP8 E4M3 input: scales holds the float scale_inv, and there is no amax.
    TORCH_CHECK(scales.has_value() && scales->is_cuda() && scales->scalar_type() == at::kFloat &&
                    scales->numel() == 1 && !amax.has_value(),
                "FP8 input needs one float scale_inv in scales and no amax");
    in.scale_inv = scales->data_ptr<float>();
  } else if (in.nvfp4) {
    TORCH_CHECK(x.numel() * 2 == tokens * cols, "NVFP4 data size mismatch");
    TORCH_CHECK(scales.has_value() && amax.has_value(), "NVFP4 input needs scales and amax");
    TORCH_CHECK(scales->is_cuda() && scales->scalar_type() == at::kByte && scales->is_contiguous() && scales->dim() == 2,
                "scales must be a contiguous 2D uint8 CUDA tensor");
    TORCH_CHECK(amax->is_cuda() && amax->scalar_type() == at::kFloat && amax->numel() == 1, "amax must be one fp32 value");
    in.scales = scales->data_ptr<uint8_t>();
    in.amax = amax->data_ptr<float>();
    in.scale_stride = static_cast<uint32_t>(scales->size(1));
    in.num_tiles_x = static_cast<uint32_t>((cols / kChunk + 3) / 4);
    in.swizzled = swizzled;
  } else {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16, NVFP4 (uint8, T x N / 2) or FP8 (uint8, T x N)");
    TORCH_CHECK(x.numel() == tokens * cols, "bf16 input size mismatch");
  }
  return in;
}

void check_small(const at::Tensor &t, int64_t rows, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.dim() == 2 &&
                  t.size(0) == rows && t.size(1) == kRank,
              name, " must be a contiguous bf16 [", rows, ", 16] tensor");
}

void check_lora_a(const at::Tensor &t, int64_t cols, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.dim() == 2 &&
                  t.size(0) == kRank && t.size(1) == cols,
              name, " must be a contiguous bf16 [16, ", cols, "] tensor");
}

std::vector<at::Tensor> lora_forward(const at::Tensor &x, const std::optional<at::Tensor> &scales,
                                     const std::optional<at::Tensor> &amax, bool swizzled,
                                     int64_t tokens, int64_t cols, const at::Tensor &lora_a, double p,
                                     bool te_rng) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(cols % kCols == 0 && tokens % kSubTokens == 0, "shape: cols % 512 == 0 and tokens % 32 == 0");
  check_lora_a(lora_a, cols, "lora_a");
  const int64_t num_chunks = tokens * cols / kChunk;
  const int64_t col_blocks = cols / kCols;
  auto options = at::TensorOptions().device(x.device());
  at::Tensor mask = at::empty({num_chunks * 2}, options.dtype(at::kByte));
  at::Tensor h = at::empty({tokens, kRank}, options.dtype(at::kBFloat16));
  at::Tensor partials = at::empty({col_blocks * tokens * kRank}, options.dtype(at::kFloat));
  FwdArgs args{};
  args.in = make_input(x, scales, amax, swizzled, tokens, cols);
  args.lora_a = reinterpret_cast<const bf16 *>(lora_a.data_ptr());
  args.mask = reinterpret_cast<uint16_t *>(mask.data_ptr());
  args.partials = partials.data_ptr<float>();
  args.philox = next_philox_state();
  args.cols = static_cast<uint32_t>(cols);
  args.tokens = static_cast<uint32_t>(tokens);
  args.drop_prob = bytewise_drop_probability(p);
  const dim3 grid(static_cast<unsigned>(col_blocks), token_groups(col_blocks, tokens / kSubTokens));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (args.in.fp8) {
    TORCH_CHECK(te_rng, "FP8 input needs the TE RNG");
    lora_fwd_kernel<kInFp8, true><<<grid, kThreads, 0, stream>>>(args);
  } else if (args.in.nvfp4 && te_rng) {
    lora_fwd_kernel<kInNvfp4, true><<<grid, kThreads, 0, stream>>>(args);
  } else if (args.in.nvfp4) {
    lora_fwd_kernel<kInNvfp4, false><<<grid, kThreads, 0, stream>>>(args);
  } else if (te_rng) {
    lora_fwd_kernel<kInBf16, true><<<grid, kThreads, 0, stream>>>(args);
  } else {
    lora_fwd_kernel<kInBf16, false><<<grid, kThreads, 0, stream>>>(args);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const float drop_scale = static_cast<float>(1.0 / (1.0 - p));
  reduce_partials(partials, nullptr, args.in.amax, h, static_cast<int>(col_blocks), drop_scale);
  return {h, mask};
}

at::Tensor lora_backward_dy(const at::Tensor &dy, const at::Tensor &lora_b, const at::Tensor &h,
                            at::Tensor grad_b, bool accumulate, double scale) {
  c10::cuda::CUDAGuard guard(dy.device());
  TORCH_CHECK(dy.is_cuda() && dy.scalar_type() == at::kBFloat16 && dy.is_contiguous() && dy.dim() == 2,
              "dy must be a contiguous bf16 [T, M] tensor");
  const int64_t tokens = dy.size(0);
  const int64_t out_cols = dy.size(1);
  TORCH_CHECK(tokens % kSubTokens == 0 && out_cols % kCols == 0, "shape: T % 32 == 0 and M % 512 == 0");
  check_small(lora_b, out_cols, "lora_b");
  check_small(h, tokens, "h");
  check_small(grad_b, out_cols, "grad_b");
  auto options = at::TensorOptions().device(dy.device());
  const int64_t col_blocks = out_cols / kCols;
  const int64_t tok_blocks = token_groups(col_blocks, tokens / kSubTokens);
  at::Tensor dh = at::empty({tokens, kRank}, options.dtype(at::kBFloat16));
  at::Tensor dh_part = at::empty({col_blocks * tokens * kRank}, options.dtype(at::kFloat));
  at::Tensor db_part = at::empty({tok_blocks * out_cols * kRank}, options.dtype(at::kFloat));
  DyArgs args{};
  args.dy = reinterpret_cast<const bf16 *>(dy.data_ptr());
  args.lora_b = reinterpret_cast<const bf16 *>(lora_b.data_ptr());
  args.h = reinterpret_cast<const bf16 *>(h.data_ptr());
  args.dh_part = dh_part.data_ptr<float>();
  args.db_part = db_part.data_ptr<float>();
  args.out_cols = static_cast<uint32_t>(out_cols);
  args.tokens = static_cast<uint32_t>(tokens);
  const dim3 grid(static_cast<unsigned>(col_blocks), static_cast<unsigned>(tok_blocks));
  lora_bwd_dy_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  reduce_partials(dh_part, nullptr, nullptr, dh, static_cast<int>(col_blocks), static_cast<float>(scale));
  reduce_partials(db_part, accumulate ? &grad_b : nullptr, nullptr, grad_b, static_cast<int>(tok_blocks),
                  static_cast<float>(scale));
  return dh;
}

at::Tensor lora_backward_dx(const at::Tensor &x, const std::optional<at::Tensor> &scales,
                            const std::optional<at::Tensor> &amax, bool swizzled, int64_t tokens,
                            int64_t cols, const at::Tensor &mask, const at::Tensor &dh,
                            const at::Tensor &lora_a, at::Tensor grad_a, bool accumulate, double p) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(cols % kCols == 0 && tokens % kSubTokens == 0, "shape: cols % 512 == 0 and tokens % 32 == 0");
  TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == at::kByte && mask.numel() == tokens * cols / 8,
              "mask must hold one bit for each element");
  check_small(dh, tokens, "dh");
  check_lora_a(lora_a, cols, "lora_a");
  check_lora_a(grad_a, cols, "grad_a");
  auto options = at::TensorOptions().device(x.device());
  const int64_t col_blocks = cols / kCols;
  const int64_t tok_blocks = token_groups(col_blocks, tokens / kSubTokens);
  at::Tensor dx = at::empty({tokens, cols}, options.dtype(at::kBFloat16));
  at::Tensor da_part = at::empty({tok_blocks * kRank * cols}, options.dtype(at::kFloat));
  const float drop_scale = static_cast<float>(1.0 / (1.0 - p));
  DxArgs args{};
  args.in = make_input(x, scales, amax, swizzled, tokens, cols);
  args.mask = reinterpret_cast<const uint16_t *>(mask.data_ptr());
  args.dh = reinterpret_cast<const bf16 *>(dh.data_ptr());
  args.lora_a = reinterpret_cast<const bf16 *>(lora_a.data_ptr());
  args.dx = reinterpret_cast<bf16 *>(dx.data_ptr());
  args.da_part = da_part.data_ptr<float>();
  args.cols = static_cast<uint32_t>(cols);
  args.tokens = static_cast<uint32_t>(tokens);
  args.scale = drop_scale;
  const dim3 grid(static_cast<unsigned>(col_blocks), static_cast<unsigned>(tok_blocks));
  auto stream = at::cuda::getCurrentCUDAStream();
  if (args.in.fp8) {
    lora_bwd_dx_kernel<kInFp8><<<grid, kThreads, 0, stream>>>(args);
  } else if (args.in.nvfp4) {
    lora_bwd_dx_kernel<kInNvfp4><<<grid, kThreads, 0, stream>>>(args);
  } else {
    lora_bwd_dx_kernel<kInBf16><<<grid, kThreads, 0, stream>>>(args);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  reduce_partials(da_part, accumulate ? &grad_a : nullptr, args.in.amax, grad_a, static_cast<int>(tok_blocks),
                  drop_scale);
  return dx;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lora_forward", &lora_forward, "Dropout mask and h = s drop(x) A^T", py::arg("x"), py::arg("scales"),
        py::arg("amax"), py::arg("swizzled"), py::arg("tokens"), py::arg("cols"), py::arg("lora_a"), py::arg("p"),
        py::arg("te_rng") = true);
  m.def("lora_backward_dy", &lora_backward_dy, "dh = scale dY B and grad_b += scale dY^T h", py::arg("dy"),
        py::arg("lora_b"), py::arg("h"), py::arg("grad_b"), py::arg("accumulate"), py::arg("scale"));
  m.def("lora_backward_dx", &lora_backward_dx, "dx = drop_bwd(dh A) and grad_a += dh^T drop(x)", py::arg("x"),
        py::arg("scales"), py::arg("amax"), py::arg("swizzled"), py::arg("tokens"), py::arg("cols"),
        py::arg("mask"), py::arg("dh"), py::arg("lora_a"), py::arg("grad_a"), py::arg("accumulate"), py::arg("p"));
}
