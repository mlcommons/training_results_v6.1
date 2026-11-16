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

// Device math of the NVFP4 quantize (nvfp4q.cu and resnorm.cu use it).
#pragma once
// Fast NVFP4 quantize of bf16 activations: rowwise, 1D blocks of 16 elements, no RHT, no 2D.
// The math is the same as TE 5b28d7aa (built without fast math):
//   cast/nvfp4/specialized/quantize_transpose_nvfp4_tuned_1D.cuh, USE_FAST_MATH = false,
//   cast/nvfp4/core_nvfp4.cuh, util/ptx_arch_spec.cuh, util/curanddx.hpp,
//   recipe/current_scaling.cu (amax), pytorch/csrc/quantizer.cpp (Philox state).
// Changes from TE:
//   - The scales go out in the GEMM swizzled layout (mxfp8::swizzle::gemm_swizzled_scale_idx).
//   - No kernel needs a memset. The amax partials go to an empty buffer, and the quantize
//     kernel reduces them.
//   - The stochastic-rounding kernel unpacks the Philox state itself. It uses the same random
//     bits for each 16-element block as the TE kernel with its 128x128 CTA grid.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>
#include <mutex>
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
// Philox4x32: the same code as TE util/curanddx.hpp.
constexpr unsigned int kPhiloxW0 = 0x9E3779B9U;
constexpr unsigned int kPhiloxW1 = 0xBB67AE85U;
constexpr unsigned int kPhiloxM0 = 0xD2511F53U;
constexpr unsigned int kPhiloxM1 = 0xCD9E8D57U;

__device__ __forceinline__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                                  unsigned int *hip) {
  uint64_t product;
  asm("mul.wide.u32 %0, %1, %2;" : "=l"(product) : "r"(a), "r"(b));
  *hip = static_cast<unsigned int>(product >> 32);
  return static_cast<unsigned int>(product);
}

__device__ __forceinline__ uint4 philox_round(uint4 ctr, uint2 key) {
  unsigned int hi0;
  unsigned int hi1;
  unsigned int lo0 = mulhilo32(kPhiloxM0, ctr.x, &hi0);
  unsigned int lo1 = mulhilo32(kPhiloxM1, ctr.z, &hi1);
  uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
  return ret;
}

template <int ROUNDS>
__device__ __forceinline__ uint4 philox_rounds(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < ROUNDS - 1; i++) {
    c = philox_round(c, k);
    k.x += kPhiloxW0;
    k.y += kPhiloxW1;
  }
  return philox_round(c, k);
}

// TE philox4x32_native_state::philox_state_incr(size_t n).
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

// The result of the generate4() call number `index` after TE init(seed, subsequence, offset).
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
  return philox_rounds<kPhiloxRounds>(ctr, key);
}

// Same as at::cuda::philox::unpack and TE extract_seed_and_offset.
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

// Position of a 16-element block in the TE tuned 1D kernel. That kernel uses 128x128 CTA
// chunks, 128 threads, 4 stages of 64x64 tiles and 2 iterations of 32 rows.
// Each block uses one generate4() result.
struct RngPosition {
  uint32_t sequence;        // TE rng_sequence of the thread that quantizes the block.
  uint32_t index;           // Number of the generate4() call in that thread.
  uint32_t bank_group;      // 1 if that thread quantizes elements 8..15 first.
};

// The block is block_in_unit (0..15) of the 256-element unit ucol of the row.
// All values fit in 32 bits because the host limits numel to less than 2^31.
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

// quantization_and_transposition_SF::compute_decoding_scaling_factor, with the
// S_enc * fp4_max_inv product made one time.
__device__ __forceinline__ __nv_fp8_e4m3 block_decode_scale(const float block_amax,
                                                            const float s_enc_times_fp4_inv) {
  return static_cast<__nv_fp8_e4m3>(fminf(block_amax * s_enc_times_fp4_inv, FLT_MAX));
}

// compute_nvfp4_scaling_coefficient<float>, with S_dec = 1 / S_enc made one time.
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
      // Elements reordered to match the packing order (v1,v0)
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
      // Elements reordered to match e2m1x4 packing order (v3,v2,v1,v0)
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

// 16-byte load that does not keep the line in L1. The kernels read each byte one time.
__device__ __forceinline__ uint4 load_stream(const uint4 *ptr) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
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
// Amax pass.
constexpr int kAmaxUnroll = 4;

// L2 prefetch of one amax loop step of a warp. The step starts at vector warp_first.
// It has kAmaxUnroll warp parts of 512 bytes, and the parts are stride vectors apart.
__device__ __forceinline__ void prefetch_amax_step(const uint4 *x, int64_t warp_first,
                                                   int64_t stride, int64_t n_vec) {
#pragma unroll
  for (int u = 0; u < kAmaxUnroll; ++u) {
    const int64_t start = warp_first + u * stride;
    if (start + 32 <= n_vec) prefetch_l2(x + start, 32 * sizeof(uint4));
  }
}

// Max |x| of a bf16 tensor. The value is the same as in TE amax_kernel, because max is exact and
// the order does not matter. Each block writes its max to partial[blockIdx.x]. With FINISH, the
// last block also writes the total to out[0..n_out) and sets the counter back to 0.
template <int THREADS, bool FINISH>
__global__ void __launch_bounds__(THREADS)
    amax_kernel(const uint4 *__restrict__ x, const int64_t n_vec, float *__restrict__ partial,
                unsigned int *__restrict__ counter, float *__restrict__ out, const int n_out,
                const int prefetch) {
  __shared__ float shared[THREADS / 32];
  __shared__ bool is_last;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * THREADS;
  int64_t i = static_cast<int64_t>(blockIdx.x) * THREADS + threadIdx.x;
  const bool prefetch_lane = (threadIdx.x & 31) == 0;
  if (prefetch_lane) {
    for (int d = 0; d < prefetch; ++d) {
      prefetch_amax_step(x, i + d * kAmaxUnroll * stride, stride, n_vec);
    }
  }
  uint32_t acc = 0;
  for (; i + (kAmaxUnroll - 1) * stride < n_vec; i += kAmaxUnroll * stride) {
    if (prefetch > 0 && prefetch_lane) {
      prefetch_amax_step(x, i + prefetch * kAmaxUnroll * stride, stride, n_vec);
    }
    uint4 v[kAmaxUnroll];
#pragma unroll
    for (int u = 0; u < kAmaxUnroll; ++u) v[u] = load_stream(x + i + u * stride);
#pragma unroll
    for (int u = 0; u < kAmaxUnroll; ++u) acc = abs_max_vec(acc, v[u]);
  }
  for (; i < n_vec; i += stride) acc = abs_max_vec(acc, load_stream(x + i));
  const float block_amax = block_max<THREADS>(amax_of_pair(acc), shared);
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = block_amax;
    if constexpr (FINISH) {
      __threadfence();
      is_last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
  }
  if constexpr (FINISH) {
    __syncthreads();
    if (is_last) {
      __threadfence();
      float total = 0.0f;
      for (int b = threadIdx.x; b < static_cast<int>(gridDim.x); b += THREADS) {
        total = fmaxf(total, __ldcg(partial + b));
      }
      total = block_max<THREADS>(total, shared);
      if (threadIdx.x == 0) {
        for (int k = 0; k < n_out; ++k) out[k] = total;
        *counter = 0;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Quantize pass.
constexpr int kUnroll = 4;  // Units in each warp step. It must be even (pairs of units).

struct QuantizeArgs {
  const uint4 *x;          // bf16 input, 8 elements per vector.
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
  // Half g of a block uses words (x, y) if g ^ bank_group == 0, else (z, w).
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

}  // namespace
