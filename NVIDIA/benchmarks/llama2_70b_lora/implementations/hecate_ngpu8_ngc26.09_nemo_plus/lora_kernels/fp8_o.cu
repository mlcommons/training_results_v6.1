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

// FP8-input kernels for the proj branch of a LoRA layer: the attention output O stays FP8.
// Each kernel gives the same bytes as the TE path, which dequantizes O to bf16 first:
//   bf16 value = bf16(float(x) * scale_inv)            (TE dequantize, unary_kernel)
//   - fp8_amax_partials:   per-block max |value|, as the TE amax_kernel of the quantize.
//   - quantize_fp8_*:      NVFP4 quantize of the values (nvfp4q math of TE 5b28d7aa, 1D blocks of
//                          16, no RHT, GEMM swizzled scales, same stochastic rounding bits).
// The quantize helpers are copies of nvfp4q_math.cuh, with FP8 loads.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kUnitVecs = 32;              // One unit: 32 lanes x 8 elements = 256 elements.
constexpr int kPhiloxRounds = 10;          // TE default of NVTE_BUILD_NUM_PHILOX_ROUNDS.
constexpr int kRngEltsPerThread = 1024;    // TE quantizer.cpp: rng_elts_per_thread.
constexpr float kFp4Max = 6.0f;
constexpr float kFp4MaxInv = 1.0f / kFp4Max;
constexpr float kFp8Max = 448.0f;

// ---------------------------------------------------------------------------------------------
// Philox4x32 of the TE stochastic rounding: the same code as TE util/curanddx.hpp.
constexpr unsigned int kTePhiloxW0 = 0x9E3779B9U;
constexpr unsigned int kTePhiloxW1 = 0xBB67AE85U;
constexpr unsigned int kTePhiloxM0 = 0xD2511F53U;
constexpr unsigned int kTePhiloxM1 = 0xCD9E8D57U;

__device__ __forceinline__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                                  unsigned int *hip) {
  uint64_t product;
  asm("mul.wide.u32 %0, %1, %2;" : "=l"(product) : "r"(a), "r"(b));
  *hip = static_cast<unsigned int>(product >> 32);
  return static_cast<unsigned int>(product);
}

__device__ __forceinline__ uint4 te_philox_round(uint4 ctr, uint2 key) {
  unsigned int hi0;
  unsigned int hi1;
  unsigned int lo0 = mulhilo32(kTePhiloxM0, ctr.x, &hi0);
  unsigned int lo1 = mulhilo32(kTePhiloxM1, ctr.z, &hi1);
  uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
  return ret;
}

template <int ROUNDS>
__device__ __forceinline__ uint4 te_philox_rounds(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < ROUNDS - 1; i++) {
    c = te_philox_round(c, k);
    k.x += kTePhiloxW0;
    k.y += kTePhiloxW1;
  }
  return te_philox_round(c, k);
}

__device__ __forceinline__ void philox_incr(uint4 &ctr, uint64_t n) {
  unsigned int nlo = static_cast<unsigned int>(n);
  unsigned int nhi = static_cast<unsigned int>(n >> 32);
  ctr.x += nlo;
  if (ctr.x < nlo) nhi++;
  ctr.y += nhi;
  if (nhi <= ctr.y) return;
  if (++ctr.z) return;
  ++ctr.w;
}

__device__ __forceinline__ uint4 philox_generate(uint64_t seed, uint64_t subsequence,
                                                 uint64_t offset, unsigned int index) {
  uint4 ctr = make_uint4(0, 0, 0, 0);
  const uint2 key = make_uint2(static_cast<unsigned int>(seed),
                               static_cast<unsigned int>(seed >> 32));
  unsigned int nlo = static_cast<unsigned int>(subsequence);
  unsigned int nhi = static_cast<unsigned int>(subsequence >> 32);
  ctr.z += nlo;
  if (ctr.z < nlo) nhi++;
  ctr.w += nhi;
  philox_incr(ctr, offset);
  philox_incr(ctr, index);
  return te_philox_rounds<kPhiloxRounds>(ctr, key);
}

__device__ __forceinline__ void unpack_philox(const at::PhiloxCudaState &state, uint64_t &seed,
                                              uint64_t &offset) {
  if (state.captured_) {
    seed = static_cast<uint64_t>(*state.seed_.ptr);
    offset = static_cast<uint64_t>(*state.offset_.ptr +
                                   static_cast<int64_t>(state.offset_intragraph_));
  } else {
    seed = state.seed_.val;
    offset = state.offset_.val;
  }
}

struct RngPosition {
  uint32_t sequence;        // TE rng_sequence of the thread that quantizes the block.
  uint32_t index;           // Number of the generate4() call in that thread.
  uint32_t bank_group;      // 1 if that thread quantizes elements 8..15 first.
};

__device__ __forceinline__ RngPosition te_rng_position(uint32_t row, uint32_t ucol,
                                                       uint32_t block_in_unit,
                                                       uint32_t cta_cols) {
  const uint32_t cta_y = row >> 7;
  const uint32_t cta_x = ucol * 2 + (block_in_unit >> 3);
  const uint32_t r = row & 127;
  const uint32_t c = block_in_unit & 7;
  const uint32_t stage = (r >> 6) * 2 + (c >> 2);
  const uint32_t row_in_tile = r & 63;
  const uint32_t tid_y = row_in_tile & 31;
  const uint32_t thread = tid_y * 4 + (c & 3);
  RngPosition position;
  position.sequence = thread + (cta_x + cta_y * cta_cols) * 128;
  position.index = stage * 2 + (row_in_tile >> 5);
  position.bank_group = (tid_y & 7) >> 2;
  return position;
}

// ---------------------------------------------------------------------------------------------
// Scale math: the same as TE core_nvfp4.cuh and quantize_transpose_nvfp4_tuned_1D.cuh.
__device__ __forceinline__ float global_encode_scale(const float global_amax) {
  float scale = kFp8Max * kFp4Max / global_amax;
  scale = fminf(scale, FLT_MAX);
  if (global_amax == 0.0f || scale == 0.0f) {
    return 1.0f;
  }
  return scale;
}

__device__ __forceinline__ __nv_fp8_e4m3 block_decode_scale(const float block_amax,
                                                            const float s_enc_times_fp4_inv) {
  return static_cast<__nv_fp8_e4m3>(fminf(block_amax * s_enc_times_fp4_inv, FLT_MAX));
}

__device__ __forceinline__ float block_coefficient(const __nv_fp8_e4m3 s_dec_block,
                                                   const float s_dec) {
  return fminf(1.0f / (static_cast<float>(s_dec_block) * s_dec), FLT_MAX);
}

// TE ptx::abs_max_2x for bf16x2: max of |a| and |b|, sign = xor of the signs.
__device__ __forceinline__ uint32_t abs_max_2x(const uint32_t a, const uint32_t b) {
  uint32_t d;
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// TE get_amax_of_pair.
__device__ __forceinline__ float amax_of_pair(const uint32_t pair) {
  __nv_bfloat162 values;
  memcpy(&values, &pair, sizeof(pair));
  return __bfloat162float(__hmax(__habs(values.x), __habs(values.y)));
}

// TE ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float> (copied PTX).
__device__ __forceinline__ uint32_t mul_cvt_rn_8x(const uint64_t in03, const uint64_t in47,
                                                  const float scaling_coefficient) {
  uint32_t out_8x = 0;
  asm volatile(
      "{\n"
      ".reg.b64 scaling_coeff_2x; \n\t"
      "mov.b64 scaling_coeff_2x, {%3, %3}; \n\t"
      ".reg.b16 v0_bf16, v1_bf16, v2_bf16, v3_bf16, v4_bf16, v5_bf16, v6_bf16, v7_bf16; \n\t"
      "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
      "mov.b64 {v4_bf16, v5_bf16, v6_bf16, v7_bf16}, %2; \n\t"
      ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
      "cvt.f32.bf16 v0, v0_bf16; \n\t"
      "cvt.f32.bf16 v1, v1_bf16; \n\t"
      "cvt.f32.bf16 v2, v2_bf16; \n\t"
      "cvt.f32.bf16 v3, v3_bf16; \n\t"
      "cvt.f32.bf16 v4, v4_bf16; \n\t"
      "cvt.f32.bf16 v5, v5_bf16; \n\t"
      "cvt.f32.bf16 v6, v6_bf16; \n\t"
      "cvt.f32.bf16 v7, v7_bf16; \n\t"
      ".reg.b64 v01, v23, v45, v67; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mov.b64 v45, {v4, v5}; \n\t"
      "mov.b64 v67, {v6, v7}; \n\t"
      "mul.f32x2 v01, v01, scaling_coeff_2x; \n\t"
      "mul.f32x2 v23, v23, scaling_coeff_2x; \n\t"
      "mul.f32x2 v45, v45, scaling_coeff_2x; \n\t"
      "mul.f32x2 v67, v67, scaling_coeff_2x; \n\t"
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "mov.b64 {v5, v4}, v45; \n\t"
      "mov.b64 {v7, v6}, v67; \n\t"
      ".reg.b8 f0, f1, f2, f3; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f2, v4, v5;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f3, v6, v7;\n\t"
      "mov.b32 %0, {f0, f1, f2, f3};\n\t"
      "}"
      : "=r"(out_8x)
      : "l"(in03), "l"(in47), "f"(scaling_coefficient));
  return out_8x;
}

// TE ptx::mul_cvt_bf16_to_fp4_8x_stochastic_rounding<float> (copied PTX).
__device__ __forceinline__ uint32_t mul_cvt_sr_8x(const uint64_t in03, const uint64_t in47,
                                                  const float scaling_coefficient,
                                                  const uint32_t rbits03, const uint32_t rbits47) {
  uint32_t out_8x = 0;
  asm volatile(
      "{\n"
      ".reg.b64 scaling_coeff_2x; \n\t"
      "mov.b64 scaling_coeff_2x, {%3, %3}; \n\t"
      ".reg.b16 v0_bf16, v1_bf16, v2_bf16, v3_bf16, v4_bf16, v5_bf16, v6_bf16, v7_bf16; \n\t"
      "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
      "mov.b64 {v4_bf16, v5_bf16, v6_bf16, v7_bf16}, %2; \n\t"
      ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
      "cvt.f32.bf16 v0, v0_bf16; \n\t"
      "cvt.f32.bf16 v1, v1_bf16; \n\t"
      "cvt.f32.bf16 v2, v2_bf16; \n\t"
      "cvt.f32.bf16 v3, v3_bf16; \n\t"
      "cvt.f32.bf16 v4, v4_bf16; \n\t"
      "cvt.f32.bf16 v5, v5_bf16; \n\t"
      "cvt.f32.bf16 v6, v6_bf16; \n\t"
      "cvt.f32.bf16 v7, v7_bf16; \n\t"
      ".reg.b64 v01, v23, v45, v67; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mov.b64 v45, {v4, v5}; \n\t"
      "mov.b64 v67, {v6, v7}; \n\t"
      "mul.f32x2 v01, v01, scaling_coeff_2x; \n\t"
      "mul.f32x2 v23, v23, scaling_coeff_2x; \n\t"
      "mul.f32x2 v45, v45, scaling_coeff_2x; \n\t"
      "mul.f32x2 v67, v67, scaling_coeff_2x; \n\t"
      "mov.b64 {v0, v1}, v01; \n\t"
      "mov.b64 {v2, v3}, v23; \n\t"
      "mov.b64 {v4, v5}, v45; \n\t"
      "mov.b64 {v6, v7}, v67; \n\t"
      ".reg.b16 b03, b47; \n\t"
      "cvt.rs.satfinite.e2m1x4.f32 b03, {v3, v2, v1, v0}, %4; \n\t"
      "cvt.rs.satfinite.e2m1x4.f32 b47, {v7, v6, v5, v4}, %5; \n\t"
      "mov.b32 %0, {b03, b47};\n"
      "}"
      : "=r"(out_8x)
      : "l"(in03), "l"(in47), "f"(scaling_coefficient), "r"(rbits03), "r"(rbits47));
  return out_8x;
}

// TE mxfp8::swizzle::gemm_swizzled_scale_idx: tiles of 128 rows x 4 scales, 512 bytes each.
__device__ __forceinline__ uint32_t swizzled_scale_index(uint32_t row, uint32_t col,
                                                         uint32_t num_tiles_x) {
  const uint32_t tile = (row >> 7) * num_tiles_x + (col >> 2);
  const uint32_t row_in_tile = row & 127;
  return tile * 512 + (row_in_tile & 31) * 16 + (row_in_tile >> 5) * 4 + (col & 3);
}

// Loads that do not keep the line in L1. The kernels read each byte one time.
__device__ __forceinline__ uint4 load_stream16(const uint4 *ptr) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint2 load_stream8(const uint2 *ptr) {
  uint2 v;
  asm("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr));
  return v;
}

// Start an asynchronous copy of global memory into L2. The thread does not wait for it.
__device__ __forceinline__ void prefetch_l2(const void *ptr, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr), "r"(bytes) : "memory");
}

__device__ __forceinline__ uint32_t abs_max_vec(uint32_t acc, const uint4 v) {
  acc = abs_max_2x(acc, v.x);
  acc = abs_max_2x(acc, v.y);
  acc = abs_max_2x(acc, v.z);
  return abs_max_2x(acc, v.w);
}

// Max over the block. All threads must call it. All threads get the result.
template <int THREADS>
__device__ __forceinline__ float block_max(float value, float *shared) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < THREADS / 32 ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

// ---------------------------------------------------------------------------------------------
// FP8 dequantize with the TE math: bf16(float(x) * scale_inv). The FP8 to half conversion is
// exact for E4M3 and E5M2, and half to float is exact.
template <__nv_fp8_interpretation_t kFormat>
__device__ __forceinline__ uint4 dequantize_fp8x8(uint2 raw, float scale_inv) {
  uint4 out;
  uint32_t *out_w = reinterpret_cast<uint32_t *>(&out);
  const __nv_fp8x2_storage_t *pairs = reinterpret_cast<const __nv_fp8x2_storage_t *>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __half2 values(__nv_cvt_fp8x2_to_halfraw2(pairs[i], kFormat));
    const float2 f = __half22float2(values);
    const __nv_bfloat162 pair = __floats2bfloat162_rn(f.x * scale_inv, f.y * scale_inv);
    memcpy(&out_w[i], &pair, sizeof(pair));
  }
  return out;
}

template <__nv_fp8_interpretation_t kFormat>
__device__ __forceinline__ float dequantize_fp8_scalar(uint8_t byte, float scale_inv) {
  const __nv_fp8x2_storage_t pair = static_cast<__nv_fp8x2_storage_t>(byte);
  const __half2 values(__nv_cvt_fp8x2_to_halfraw2(pair, kFormat));
  const float2 f = __half22float2(values);
  return __bfloat162float(__float2bfloat16(f.x * scale_inv));
}

// ---------------------------------------------------------------------------------------------
// Amax pass over FP8 data. |value| grows with the FP8 magnitude bits (sign-magnitude format,
// a positive scale_inv, and a monotonic bf16 rounding). Thus the block max of the magnitude
// bytes gives the block amax of the bf16 values after one dequantize of the max byte. This is
// the value that the TE amax_kernel finds in the dequantized bf16 tensor.
constexpr int kAmaxThreads = 256;
constexpr int kAmaxUnroll = 4;

__device__ __forceinline__ uint32_t magnitude_max4(uint32_t acc, uint32_t word) {
  return __vmaxu4(acc, word & 0x7f7f7f7fu);
}

__device__ __forceinline__ uint32_t magnitude_max16(uint32_t acc, const uint4 v) {
  acc = magnitude_max4(acc, v.x);
  acc = magnitude_max4(acc, v.y);
  acc = magnitude_max4(acc, v.z);
  return magnitude_max4(acc, v.w);
}

__device__ __forceinline__ uint32_t max_byte(uint32_t word) {
  uint32_t m = word & 0xffu;
  m = max(m, (word >> 8) & 0xffu);
  m = max(m, (word >> 16) & 0xffu);
  return max(m, word >> 24);
}

template <__nv_fp8_interpretation_t kFormat>
__global__ void __launch_bounds__(kAmaxThreads)
    fp8_amax_partials_kernel(const uint4 *__restrict__ x, const int64_t n_vec16,
                             const float *__restrict__ scale_inv_ptr, float *__restrict__ partial) {
  __shared__ float shared[kAmaxThreads / 32];
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kAmaxThreads;
  int64_t i = static_cast<int64_t>(blockIdx.x) * kAmaxThreads + threadIdx.x;
  uint32_t acc = 0;
  for (; i + (kAmaxUnroll - 1) * stride < n_vec16; i += kAmaxUnroll * stride) {
    uint4 v[kAmaxUnroll];
#pragma unroll
    for (int u = 0; u < kAmaxUnroll; ++u) v[u] = load_stream16(x + i + u * stride);
#pragma unroll
    for (int u = 0; u < kAmaxUnroll; ++u) acc = magnitude_max16(acc, v[u]);
  }
  for (; i < n_vec16; i += stride) acc = magnitude_max16(acc, load_stream16(x + i));
  const float value = dequantize_fp8_scalar<kFormat>(static_cast<uint8_t>(max_byte(acc)),
                                                     *scale_inv_ptr);
  const float block_amax = block_max<kAmaxThreads>(value, shared);
  if (threadIdx.x == 0) partial[blockIdx.x] = block_amax;
}

// ---------------------------------------------------------------------------------------------
// Quantize pass: nvfp4q quantize_kernel with FP8 loads and the dequantize in registers.
constexpr int kUnroll = 4;  // Units in each warp step. It must be even (pairs of units).

struct QuantizeArgs {
  const uint2 *x;          // FP8 input, 8 elements per vector.
  const float *scale_inv;  // FP8 scale_inv (one float).
  uint32_t *data;          // FP4 output, 8 elements per word, row-major like TE.
  uint8_t *scales;         // E4M3 block scales in the GEMM swizzled layout.
  const float *partial;    // Amax partials. The global amax is their max.
  int n_partial;
  float *amax_out;         // Gets the global amax. Can be null.
  uint32_t rows;
  uint32_t cols;
  int prefetch;            // L2 prefetch distance in warp steps (0 = off).
  at::PhiloxCudaState philox;
  const float *alpha_weight_amax = nullptr;  // The amax of the weight of the next GEMM.
  float *alpha_out = nullptr;                // Gets the alpha of the next GEMM. Can be null.
};

// The alpha of the next NVFP4 GEMM, as TE nvfp4_compute_per_tensor_scale makes it with
// alpha_in = 1: (amax_weight * amax) / (6 * 6 * 448 * 448) in fp32. Then the GEMM needs no
// alpha kernel.
constexpr float kGemmAlphaFactor = 1.0f / (kFp4Max * kFp4Max * kFp8Max * kFp8Max);

__device__ __forceinline__ void write_gemm_alpha(const float *weight_amax, float *alpha,
                                                 const float amax) {
  if (alpha != nullptr) alpha[0] = (weight_amax[0] * amax) * kGemmAlphaFactor;
}

// The alpha pointers of the next GEMM: two optional float[1] CUDA tensors, both or none.
struct GemmAlpha {
  const float *weight_amax = nullptr;
  float *out = nullptr;
};

inline GemmAlpha gemm_alpha(const std::optional<at::Tensor> &weight_amax,
                            const std::optional<at::Tensor> &alpha) {
  TORCH_CHECK(weight_amax.has_value() == alpha.has_value(), "give both alpha tensors or none");
  GemmAlpha result;
  if (!alpha.has_value()) return result;
  TORCH_CHECK(weight_amax->is_cuda() && weight_amax->scalar_type() == at::kFloat &&
                  weight_amax->numel() >= 1 && alpha->is_cuda() &&
                  alpha->scalar_type() == at::kFloat && alpha->numel() >= 1,
              "alpha tensors: float CUDA tensors with one value");
  result.weight_amax = weight_amax->data_ptr<float>();
  result.out = alpha->data_ptr<float>();
  return result;
}

// Row and unit column of kUnroll units in reverse order: first_unit, first_unit - 1, ...
__device__ __forceinline__ void unit_positions(uint32_t first_unit, uint32_t units_per_row,
                                               uint32_t (&row)[kUnroll],
                                               uint32_t (&ucol)[kUnroll]) {
  row[0] = first_unit / units_per_row;
  ucol[0] = first_unit - row[0] * units_per_row;
#pragma unroll
  for (int k = 1; k < kUnroll; ++k) {
    const bool wrap = ucol[k - 1] == 0;
    row[k] = wrap ? row[k - 1] - 1 : row[k - 1];
    ucol[k] = wrap ? units_per_row - 1 : ucol[k - 1] - 1;
  }
}

// Random words for units k and k + 1. Lane half h makes the Philox call of its block in unit
// k + h. Then the two lanes swap the two words that the other lane needs.
__device__ __forceinline__ void pair_random_words(const uint32_t (&row)[2],
                                                  const uint32_t (&ucol)[2], uint32_t lane,
                                                  uint32_t cta_cols, uint64_t seed,
                                                  uint64_t offset, uint32_t (&rbits)[2][2]) {
  const uint32_t half = lane & 1;
  const uint32_t own_row = half ? row[1] : row[0];
  const uint32_t own_ucol = half ? ucol[1] : ucol[0];
  const RngPosition position = te_rng_position(own_row, own_ucol, lane >> 1, cta_cols);
  const uint4 r = philox_generate(seed, position.sequence, offset, position.index);
  const bool keep_xy = (half ^ position.bank_group) == 0;
  const uint32_t keep0 = keep_xy ? r.x : r.z;
  const uint32_t keep1 = keep_xy ? r.y : r.w;
  const uint32_t give0 = keep_xy ? r.z : r.x;
  const uint32_t give1 = keep_xy ? r.w : r.y;
  const uint32_t got0 = __shfl_xor_sync(0xffffffffu, give0, 1);
  const uint32_t got1 = __shfl_xor_sync(0xffffffffu, give1, 1);
  rbits[0][0] = half ? got0 : keep0;
  rbits[0][1] = half ? got1 : keep1;
  rbits[1][0] = half ? keep0 : got0;
  rbits[1][1] = half ? keep1 : got1;
}

struct ScaleConstants {
  float s_enc_times_fp4_inv;
  float s_dec;
  uint32_t num_tiles_x;
};

// Block scales of units k and k + 1. Lane half h makes the scale of its block in unit k + h and
// stores it. The pair swaps the coefficients, so each lane gets the coefficients of both units.
__device__ __forceinline__ void pair_coefficients(const uint4 (&v)[2], const uint32_t (&row)[2],
                                                  const uint32_t (&ucol)[2], uint32_t lane,
                                                  const ScaleConstants &constants,
                                                  uint8_t *__restrict__ scales,
                                                  float (&coefficient)[2]) {
  const uint32_t half = lane & 1;
  uint32_t acc0 = abs_max_vec(0u, v[0]);
  uint32_t acc1 = abs_max_vec(0u, v[1]);
  acc0 = abs_max_2x(acc0, __shfl_xor_sync(0xffffffffu, acc0, 1));
  acc1 = abs_max_2x(acc1, __shfl_xor_sync(0xffffffffu, acc1, 1));
  const __nv_fp8_e4m3 s_dec_block =
      block_decode_scale(amax_of_pair(half ? acc1 : acc0), constants.s_enc_times_fp4_inv);
  const float own = block_coefficient(s_dec_block, constants.s_dec);
  const float other = __shfl_xor_sync(0xffffffffu, own, 1);
  coefficient[0] = half ? other : own;
  coefficient[1] = half ? own : other;
  const uint32_t own_row = half ? row[1] : row[0];
  const uint32_t own_col = (half ? ucol[1] : ucol[0]) * 16 + (lane >> 1);
  scales[swizzled_scale_index(own_row, own_col, constants.num_tiles_x)] = s_dec_block.__x;
}

// FP4 words of one pair of units, then the stores.
template <bool SR>
__device__ __forceinline__ void pair_convert(const uint4 (&v)[2], const float (&coefficient)[2],
                                             const uint32_t (&rbits)[2][2],
                                             uint32_t *__restrict__ data, uint32_t unit0,
                                             uint32_t lane) {
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const uint64_t in03 = (static_cast<uint64_t>(v[j].y) << 32) | v[j].x;
    const uint64_t in47 = (static_cast<uint64_t>(v[j].w) << 32) | v[j].z;
    uint32_t out;
    if constexpr (SR) {
      out = mul_cvt_sr_8x(in03, in47, coefficient[j], rbits[j][0], rbits[j][1]);
    } else {
      out = mul_cvt_rn_8x(in03, in47, coefficient[j]);
    }
    data[(unit0 - j) * kUnitVecs + lane] = out;
  }
}

// Quantize kUnroll units in each warp step. A unit is 256 elements of one row. Lane l holds
// elements 8l..8l+7 of the unit, so lanes 2p and 2p+1 hold the 16-element block p.
// The units go from the end of the tensor to the start. The amax pass read the tensor from the
// start, thus the end is most likely still in L2.
template <bool SR, __nv_fp8_interpretation_t kFormat>
__global__ void __launch_bounds__(kThreads) quantize_fp8_kernel(const QuantizeArgs args) {
  __shared__ float shared[kWarps];
  float amax = 0.0f;
  for (int b = threadIdx.x; b < args.n_partial; b += kThreads) amax = fmaxf(amax, args.partial[b]);
  amax = block_max<kThreads>(amax, shared);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (args.amax_out != nullptr) args.amax_out[0] = amax;
    write_gemm_alpha(args.alpha_weight_amax, args.alpha_out, amax);
  }
  ScaleConstants constants;
  const float s_enc = global_encode_scale(amax);
  constants.s_enc_times_fp4_inv = s_enc * kFp4MaxInv;
  constants.s_dec = 1.0f / s_enc;
  constants.num_tiles_x = args.cols >> 6;
  const float scale_inv = *args.scale_inv;

  uint64_t seed = 0;
  uint64_t offset = 0;
  if constexpr (SR) unpack_philox(args.philox, seed, offset);

  const uint32_t lane = threadIdx.x & 31;
  const uint32_t units_per_row = args.cols >> 8;
  const uint32_t n_units = args.rows * units_per_row;
  const uint32_t cta_cols = args.cols >> 7;
  const uint32_t warp_first = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * kUnroll;
  const uint32_t step = gridDim.x * kWarps * kUnroll;
  const auto prefetch_step = [&](uint32_t ahead) {
    if (lane == 0 && ahead < n_units) {
      const uint32_t low_unit = n_units - ahead - kUnroll;
      prefetch_l2(args.x + low_unit * kUnitVecs, kUnroll * kUnitVecs * sizeof(uint2));
    }
  };
  for (int d = 0; d < args.prefetch; ++d) prefetch_step(warp_first + d * step);

  for (uint32_t first = warp_first; first < n_units; first += step) {
    if (args.prefetch > 0) prefetch_step(first + args.prefetch * step);
    const uint32_t first_unit = n_units - 1 - first;
    uint2 raw[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      raw[k] = load_stream8(args.x + (first_unit - k) * kUnitVecs + lane);
    }
    uint4 v[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) v[k] = dequantize_fp8x8<kFormat>(raw[k], scale_inv);
    uint32_t row[kUnroll];
    uint32_t ucol[kUnroll];
    unit_positions(first_unit, units_per_row, row, ucol);
#pragma unroll
    for (int k = 0; k < kUnroll; k += 2) {
      const uint4 pair_v[2] = {v[k], v[k + 1]};
      const uint32_t pair_row[2] = {row[k], row[k + 1]};
      const uint32_t pair_ucol[2] = {ucol[k], ucol[k + 1]};
      float coefficient[2];
      pair_coefficients(pair_v, pair_row, pair_ucol, lane, constants, args.scales, coefficient);
      uint32_t rbits[2][2] = {{0, 0}, {0, 0}};
      if constexpr (SR) {
        pair_random_words(pair_row, pair_ucol, lane, cta_cols, seed, offset, rbits);
      }
      pair_convert<SR>(pair_v, coefficient, rbits, args.data, first_unit - k, lane);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Host side.
int sm_count() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

void check_fp8_data(const at::Tensor &data, const at::Tensor &scale_inv) {
  TORCH_CHECK(data.is_cuda() && data.scalar_type() == at::kByte && data.is_contiguous(),
              "data: contiguous uint8 CUDA tensor");
  TORCH_CHECK(data.numel() % 16 == 0 && reinterpret_cast<uintptr_t>(data.data_ptr()) % 16 == 0,
              "data: numel must be a multiple of 16, 16-byte aligned");
  TORCH_CHECK(scale_inv.is_cuda() && scale_inv.scalar_type() == at::kFloat &&
                  scale_inv.numel() >= 1,
              "scale_inv: CUDA float tensor");
}

void check_quantize_shape(const at::Tensor &data) {
  TORCH_CHECK(data.dim() >= 2, "data must have 2 or more dims");
  TORCH_CHECK(data.numel() < (int64_t{1} << 31), "data must have less than 2^31 elements");
  const int64_t cols = data.size(-1);
  TORCH_CHECK(cols > 0 && cols % 256 == 0, "cols must be a positive multiple of 256");
  const int64_t rows = data.numel() / cols;
  TORCH_CHECK(rows > 0 && rows % 128 == 0, "rows must be a positive multiple of 128");
}

template <__nv_fp8_interpretation_t kFormat>
int amax_blocks_per_sm() {
  static int blocks = 0;
  if (blocks == 0) {
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks, fp8_amax_partials_kernel<kFormat>, kAmaxThreads, 0) == cudaSuccess);
  }
  return blocks;
}

// Grid of one wave: each block stays on its SM until the end of the kernel.
int amax_grid(int64_t n_vec16, int blocks_per_sm) {
  const int64_t needed = (n_vec16 + kAmaxThreads * kAmaxUnroll - 1) / (kAmaxThreads * kAmaxUnroll);
  const int64_t limit = static_cast<int64_t>(blocks_per_sm) * sm_count();
  return static_cast<int>(std::max<int64_t>(1, std::min(needed, limit)));
}

// Per-block amax of the dequantized values. Returns the partials (one float for each block).
at::Tensor fp8_amax_partials(const at::Tensor &data, const at::Tensor &scale_inv, bool e5m2) {
  check_fp8_data(data, scale_inv);
  c10::cuda::CUDAGuard guard(data.device());
  const int64_t n_vec16 = data.numel() / 16;
  const int blocks_per_sm = e5m2 ? amax_blocks_per_sm<__NV_E5M2>() : amax_blocks_per_sm<__NV_E4M3>();
  const int grid = amax_grid(n_vec16, blocks_per_sm);
  at::Tensor partial = at::empty({grid}, data.options().dtype(at::kFloat));
  const auto *x = reinterpret_cast<const uint4 *>(data.data_ptr());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (e5m2) {
    fp8_amax_partials_kernel<__NV_E5M2><<<grid, kAmaxThreads, 0, stream>>>(
        x, n_vec16, scale_inv.data_ptr<float>(), partial.data_ptr<float>());
  } else {
    fp8_amax_partials_kernel<__NV_E4M3><<<grid, kAmaxThreads, 0, stream>>>(
        x, n_vec16, scale_inv.data_ptr<float>(), partial.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return partial;
}

template <bool SR, __nv_fp8_interpretation_t kFormat>
void launch_quantize(const QuantizeArgs &args) {
  static int blocks_per_sm = 0;
  if (blocks_per_sm == 0) {
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, quantize_fp8_kernel<SR, kFormat>, kThreads, 0) == cudaSuccess);
  }
  const int64_t n_units = static_cast<int64_t>(args.rows) * (args.cols >> 8);
  const int64_t needed = (n_units + kWarps * kUnroll - 1) / (kWarps * kUnroll);
  const int64_t limit = static_cast<int64_t>(blocks_per_sm) * sm_count();
  const int grid = static_cast<int>(std::max<int64_t>(1, std::min(needed, limit)));
  quantize_fp8_kernel<SR, kFormat><<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::PhiloxCudaState next_philox_state() {
  // Same calls as TE init_philox_state(gen, rng_elts_per_thread = 1024).
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  return gen->philox_cuda_state(kRngEltsPerThread);
}

// NVFP4 quantize of the dequantized values, with the amax in partials. Returns
// (fp4 data, swizzled scales, amax).
std::vector<at::Tensor> quantize_fp8_from_partials(const at::Tensor &data,
                                                   const at::Tensor &scale_inv, bool e5m2,
                                                   const at::Tensor &partials,
                                                   bool stochastic_rounding, int64_t prefetch,
                                                   const std::optional<at::Tensor> &alpha_weight_amax,
                                                   const std::optional<at::Tensor> &alpha_out) {
  check_fp8_data(data, scale_inv);
  check_quantize_shape(data);
  TORCH_CHECK(partials.is_cuda() && partials.scalar_type() == at::kFloat &&
                  partials.is_contiguous() && partials.numel() >= 1 &&
                  partials.numel() < (int64_t{1} << 30),
              "partials must be a contiguous CUDA float tensor");
  c10::cuda::CUDAGuard guard(data.device());
  const int64_t cols = data.size(-1);
  const int64_t rows = data.numel() / cols;
  auto out_shape = data.sizes().vec();
  out_shape.back() = cols / 2;
  at::Tensor fp4 = at::empty(out_shape, data.options());
  at::Tensor scales = at::empty({rows, cols / 16}, data.options());
  at::Tensor amax = at::empty({1}, data.options().dtype(at::kFloat));
  QuantizeArgs args;
  args.x = reinterpret_cast<const uint2 *>(data.data_ptr());
  args.scale_inv = scale_inv.data_ptr<float>();
  args.data = reinterpret_cast<uint32_t *>(fp4.data_ptr());
  args.scales = scales.data_ptr<uint8_t>();
  args.partial = partials.data_ptr<float>();
  args.n_partial = static_cast<int>(partials.numel());
  args.amax_out = amax.data_ptr<float>();
  args.rows = static_cast<uint32_t>(rows);
  args.cols = static_cast<uint32_t>(cols);
  args.prefetch = static_cast<int>(prefetch);
  const GemmAlpha alpha = gemm_alpha(alpha_weight_amax, alpha_out);
  args.alpha_weight_amax = alpha.weight_amax;
  args.alpha_out = alpha.out;
  if (stochastic_rounding) {
    args.philox = next_philox_state();
    if (e5m2) {
      launch_quantize<true, __NV_E5M2>(args);
    } else {
      launch_quantize<true, __NV_E4M3>(args);
    }
  } else if (e5m2) {
    launch_quantize<false, __NV_E5M2>(args);
  } else {
    launch_quantize<false, __NV_E4M3>(args);
  }
  return {fp4, scales, amax};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_amax_partials", &fp8_amax_partials,
        "Per-block amax of the dequantized bf16 values of an FP8 tensor", py::arg("data"),
        py::arg("scale_inv"), py::arg("e5m2"));
  m.def("quantize_fp8_from_partials", &quantize_fp8_from_partials,
        "NVFP4 quantize of an FP8 tensor with given amax partials (swizzled scales)",
        py::arg("data"), py::arg("scale_inv"), py::arg("e5m2"), py::arg("partials"),
        py::arg("stochastic_rounding"), py::arg("prefetch") = 1,
        py::arg("alpha_weight_amax") = py::none(), py::arg("alpha_out") = py::none());
}
