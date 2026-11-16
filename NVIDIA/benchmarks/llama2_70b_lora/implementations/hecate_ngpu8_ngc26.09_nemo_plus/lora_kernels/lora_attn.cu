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

// Fused kernels for the LoRA attention block, with the amax producer kernels in the same module.
// Each kernel gives the same bytes as the TE path.
// Some kernels have faster variants with the same math. The variant
// argument of each function selects one. The default is the fastest tested variant.
// This file copies the per-element math of TE commit 5b28d7aa. The sources are:
// - RoPE: common/fused_rope/fused_rope.cu (through fast_rope.cu).
// - FP8 cast: common/cast/fp8/quantize_fp8.cuh (cast_fp8_1D_kernel).
// - FP8 dequantize: common/cast/fp8/dequantize_fp8.cuh and util/vectorized_pointwise.h.
// - NVFP4 dequantize: common/cast/nvfp4/dequantize_nvfp4.cuh.
// - Dropout: common/dropout/dropout.cu and pytorch/csrc/extensions/dropout.cpp.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <curand_kernel.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <type_traits>

namespace {

// ---------------------------------------------------------------------------------------------
// RoPE (same layout and math as fast_rope.cu)
// ---------------------------------------------------------------------------------------------

constexpr int kDim = 128;  // head dim and rotary dim
constexpr int kHalf = kDim / 2;
constexpr int kVec = 8;  // bf16 elements in 16 bytes
constexpr int kThreads = 128;
// Largest token row in dynamic shared memory without an opt-in attribute.
constexpr int kMaxRowBytes = 48 * 1024;

struct alignas(16) Bf16x8 {
  __nv_bfloat16 v[kVec];
};

struct RowOffsets {
  long long src;
  long long q;
  long long k;
  long long v;
};

__device__ __forceinline__ RowOffsets token_offsets(int s_id, int b_id, int s, int b, int h, int q_split,
                                    int k_split, int v_split, bool sbhd) {
  const long long token = sbhd ? (static_cast<long long>(s_id) * b + b_id)
                               : (static_cast<long long>(b_id) * s + s_id);
  const int total_d = q_split + k_split + v_split;
  return {token * h * total_d, token * h * q_split, token * h * k_split, token * h * v_split};
}

// Same call and same order as TE: sincosf(freq, &sin, &cos) for q and for k.
__device__ __forceinline__ void load_cos_sin(const float *q_freqs, const float *k_freqs, int s_id, float *smem) {
  for (int i = threadIdx.x; i < kDim; i += blockDim.x) {
    sincosf(q_freqs[s_id * kDim + i], &smem[kDim + i], &smem[i]);
    sincosf(k_freqs[s_id * kDim + i], &smem[3 * kDim + i], &smem[2 * kDim + i]);
  }
  __syncthreads();
}

// Read 8 table values with two 16-byte shared memory loads. The start must be 16-byte aligned.
struct Float8Values {
  float v[kVec];
};

__device__ __forceinline__ Float8Values load_table8(const float *ptr) {
  const float4 first = *reinterpret_cast<const float4 *>(ptr);
  const float4 second = *reinterpret_cast<const float4 *>(ptr + 4);
  return {{first.x, first.y, first.z, first.w, second.x, second.y, second.z, second.w}};
}

// Backward math of TE for the first half and the second half of a head.
__device__ __forceinline__ void rope_backward_pair(const Bf16x8 &src_lo, const Bf16x8 &src_hi, const float *cos_v,
                                   const float *sin_v, int j, Bf16x8 &out_lo, Bf16x8 &out_hi) {
  const Float8Values cos_lo = load_table8(cos_v + j);
  const Float8Values cos_hi = load_table8(cos_v + j + kHalf);
  const Float8Values sin_lo = load_table8(sin_v + j);
  const Float8Values sin_hi = load_table8(sin_v + j + kHalf);
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    {
      const float v_src = __bfloat162float(src_lo.v[e]);
      const float v_cos = cos_lo.v[e];
      const float v_src_rotate = __bfloat162float(src_hi.v[e]);
      const float v_sin = sin_hi.v[e];
      out_lo.v[e] = __float2bfloat16(v_src * v_cos + v_src_rotate * v_sin);
    }
    {
      const float v_src = __bfloat162float(src_hi.v[e]);
      const float v_cos = cos_hi.v[e];
      const float v_src_rotate = __bfloat162float(src_lo.v[e]);
      const float v_sin = -sin_lo.v[e];
      out_hi.v[e] = __float2bfloat16(v_src * v_cos + v_src_rotate * v_sin);
    }
  }
}

__device__ __forceinline__ Bf16x8 load_vec(const __nv_bfloat16 *ptr) { return *reinterpret_cast<const Bf16x8 *>(ptr); }

__device__ __forceinline__ void store_vec(__nv_bfloat16 *ptr, const Bf16x8 &value) {
  *reinterpret_cast<Bf16x8 *>(ptr) = value;
}

// Pair tasks in one group row: q pairs, then k pairs. One pair is 8 elements of each half.
struct GroupTasks {
  int q;
  int k;
};

__device__ __forceinline__ GroupTasks group_tasks(int q_split, int k_split) {
  const int pairs_per_head = kHalf / kVec;
  return {(q_split / kDim) * pairs_per_head, (k_split / kDim) * pairs_per_head};
}

// Forward math of TE for the first half (d_id < 64) and the second half (d_id >= 64) of a head.
// The results stay in float until a bf16x2 rounding. Each rounding is the same as in TE.
__device__ __forceinline__ void rope_forward_pair_x2(const Bf16x8 &src_lo, const Bf16x8 &src_hi,
                                     const float *cos_v, const float *sin_v, int j,
                                     __nv_bfloat162 *out_lo, __nv_bfloat162 *out_hi) {
  const Float8Values cos_lo = load_table8(cos_v + j);
  const Float8Values cos_hi = load_table8(cos_v + j + kHalf);
  const Float8Values sin_lo = load_table8(sin_v + j);
  const Float8Values sin_hi = load_table8(sin_v + j + kHalf);
#pragma unroll
  for (int e = 0; e < kVec; e += 2) {
    float lo[2];
    float hi[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      {
        const float v_cos = cos_lo.v[e + i];
        const float v_sin = sin_lo.v[e + i];
        const float v_src = __bfloat162float(src_lo.v[e + i]);
        const float v_src_rotate = -__bfloat162float(src_hi.v[e + i]);
        lo[i] = v_src * v_cos + v_src_rotate * v_sin;
      }
      {
        const float v_cos = cos_hi.v[e + i];
        const float v_sin = sin_hi.v[e + i];
        const float v_src = __bfloat162float(src_hi.v[e + i]);
        const float v_src_rotate = __bfloat162float(src_lo.v[e + i]);
        hi[i] = v_src * v_cos + v_src_rotate * v_sin;
      }
    }
    out_lo[e / 2] = __floats2bfloat162_rn(lo[0], lo[1]);
    out_hi[e / 2] = __floats2bfloat162_rn(hi[0], hi[1]);
  }
}

// TE cast_fp8_1D_kernel for two elements: amax = max(amax, |x|), out = fp8e4m3(float(x) * scale).
// The bf16 max is exact, thus it gives the same amax as the fp32 max of TE.
__device__ __forceinline__ uint32_t cast_fp8x2(__nv_bfloat162 x, float scale,
                                               __nv_bfloat162 &amax2) {
  amax2 = __hmax2(amax2, __habs2(x));
  const float2 f = __bfloat1622float2(x);
  return __nv_cvt_float2_to_fp8x2(make_float2(f.x * scale, f.y * scale), __NV_SATFINITE,
                                  __NV_E4M3);
}

__device__ __forceinline__ uint2 cast_fp8x8(const __nv_bfloat162 *x, float scale,
                                            __nv_bfloat162 &amax2) {
  uint2 out;
  out.x = cast_fp8x2(x[0], scale, amax2) | (cast_fp8x2(x[1], scale, amax2) << 16);
  out.y = cast_fp8x2(x[2], scale, amax2) | (cast_fp8x2(x[3], scale, amax2) << 16);
  return out;
}

// TE FP8 dequantize for 8 values: bf16(float(x) * scale_inv), two values at a time.
template <__nv_fp8_interpretation_t kFormat>
__device__ __forceinline__ Bf16x8 dequantize_fp8x8(uint2 raw, float scale_inv) {
  Bf16x8 out;
  __nv_bfloat162 *out2 = reinterpret_cast<__nv_bfloat162 *>(&out);
  const __nv_fp8x2_storage_t *pairs = reinterpret_cast<const __nv_fp8x2_storage_t *>(&raw);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    const __half2 values(__nv_cvt_fp8x2_to_halfraw2(pairs[i], kFormat));
    const float2 f = __half22float2(values);
    out2[i] = __floats2bfloat162_rn(f.x * scale_inv, f.y * scale_inv);
  }
  return out;
}

// Warp max and one atomic max on the int bits, as TE atomicMaxFloat.
// amax_seen is a value of amax_ptr from the kernel start. The stored value only grows, thus the
// atomic is not necessary when the warp max is not larger than amax_seen.
__device__ __forceinline__ void warp_amax_update(float amax, int amax_seen, float *amax_ptr) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
  }
  if (threadIdx.x % 32 == 0 && __float_as_int(amax) > amax_seen) {
    atomicMax(reinterpret_cast<int *>(amax_ptr), __float_as_int(amax));
  }
}

// Offsets of one q or k pair task (8 elements of each half of one head) inside one token:
// in the packed qkv row of the token, and in the q row or the k row of the token.
struct TokenPair {
  int packed;
  int split;
  int j;  // first of the 8 elements in the half
  bool is_q;
};

__device__ __forceinline__ TokenPair token_pair(int task, const GroupTasks &tasks, int total_d, int q_split,
                                int k_split) {
  const int qk_per_group = tasks.q + tasks.k;
  const int g = task / qk_per_group;
  const int t = task % qk_per_group;
  const bool is_q = t < tasks.q;
  const int pair = is_q ? t : t - tasks.q;
  const int j = (pair % (kHalf / kVec)) * kVec;
  const int p = (pair / (kHalf / kVec)) * kDim + j;
  return {g * total_d + (is_q ? 0 : q_split) + p, g * (is_q ? q_split : k_split) + p, j, is_q};
}

// The token row goes into shared memory with asynchronous bulk copies (TMA). One mbarrier
// tells when all bytes arrived. Thread 0 starts the copies. The other threads wait after a
// __syncthreads, thus they see the initialized barrier.
__device__ __forceinline__ uint32_t shared_address(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// The bulk copy and the mbarrier wait need sm_90 or later. The Python side uses these kernels
// only on such GPUs. Other targets of the build stop with a trap.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define LORA_ATTN_HAS_BULK_COPY 1
#else
#define LORA_ATTN_HAS_BULK_COPY 0
#endif

__device__ __forceinline__ void start_barrier(uint64_t *barrier, uint32_t bytes) {
#if LORA_ATTN_HAS_BULK_COPY
  const uint32_t address = shared_address(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(address) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(address),
               "r"(bytes)
               : "memory");
#else
  __trap();
#endif
}

__device__ __forceinline__ void copy_to_shared(void *dst, const void *src, uint32_t bytes,
                                               uint64_t *barrier) {
#if LORA_ATTN_HAS_BULK_COPY
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::
          "r"(shared_address(dst)),
      "l"(src), "r"(bytes), "r"(shared_address(barrier))
      : "memory");
#else
  __trap();
#endif
}

__device__ __forceinline__ void wait_barrier(uint64_t *barrier) {
#if LORA_ATTN_HAS_BULK_COPY
  const uint32_t address = shared_address(barrier);
  uint32_t done = 0;
  while (done == 0) {
    asm volatile(
        "{\n"
        ".reg .pred complete;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 complete, [%1], 0;\n"
        "selp.u32 %0, 1, 0, complete;\n"
        "}\n"
        : "=r"(done)
        : "r"(address)
        : "memory");
  }
#else
  __trap();
#endif
}

// All threads must be done with the barrier before thread 0 invalidates it.
__device__ __forceinline__ void end_barrier(uint64_t *barrier) {
  __syncthreads();
#if LORA_ATTN_HAS_BULK_COPY
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared_address(barrier)) : "memory");
  }
#endif
}

// RoPE and FP8 cast of one q or k pair task.
__device__ __forceinline__ void forward_pair_fp8(const TokenPair &task, const __nv_bfloat16 *row,
                                 const float *smem, float scale, uint8_t *dst,
                                 __nv_bfloat162 &amax2) {
  const Bf16x8 src_lo = load_vec(row + task.packed);
  const Bf16x8 src_hi = load_vec(row + task.packed + kHalf);
  const float *cos_v = task.is_q ? smem : smem + 2 * kDim;
  __nv_bfloat162 out_lo[kVec / 2];
  __nv_bfloat162 out_hi[kVec / 2];
  rope_forward_pair_x2(src_lo, src_hi, cos_v, cos_v + kDim, task.j, out_lo, out_hi);
  *reinterpret_cast<uint2 *>(dst) = cast_fp8x8(out_lo, scale, amax2);
  *reinterpret_cast<uint2 *>(dst + kHalf) = cast_fp8x8(out_hi, scale, amax2);
}

// FP8 cast of one v task: 16 elements, no rotation.
__device__ __forceinline__ void forward_v_fp8(const __nv_bfloat16 *src, float scale, uint8_t *dst,
                              __nv_bfloat162 &amax2) {
  const Bf16x8 first_in = load_vec(src);
  const Bf16x8 second_in = load_vec(src + kVec);
  const uint2 first =
      cast_fp8x8(reinterpret_cast<const __nv_bfloat162 *>(&first_in), scale, amax2);
  const uint2 second =
      cast_fp8x8(reinterpret_cast<const __nv_bfloat162 *>(&second_in), scale, amax2);
  *reinterpret_cast<uint4 *>(dst) = make_uint4(first.x, first.y, second.x, second.y);
}

// One block does one token. The bulk copy of the bf16 qkv row and the cos/sin table overlap.
__global__ void __launch_bounds__(kThreads)
    qkv_rope_forward_fp8_kernel(const __nv_bfloat16 *__restrict__ qkv,
                                const float *__restrict__ q_freqs,
                                const float *__restrict__ k_freqs, uint8_t *__restrict__ q_out,
                                uint8_t *__restrict__ k_out, uint8_t *__restrict__ v_out,
                                const float *__restrict__ scale_ptr, float *__restrict__ amax_ptr,
                                float *__restrict__ scale_inv_ptr, int s, int b, int h,
                                int q_split, int k_split, int v_split, bool sbhd) {
  extern __shared__ __align__(128) uint8_t token_row[];
  __shared__ __align__(16) float smem[4 * kDim];  // cos_q, sin_q, cos_k, sin_k
  __shared__ __align__(8) uint64_t row_ready;
  const RowOffsets off =
      token_offsets(blockIdx.x, blockIdx.y, s, b, h, q_split, k_split, v_split, sbhd);
  const int total_d = q_split + k_split + v_split;
  if (threadIdx.x == 0) {
    const uint32_t bytes = h * total_d * sizeof(__nv_bfloat16);
    start_barrier(&row_ready, bytes);
    copy_to_shared(token_row, qkv + off.src, bytes, &row_ready);
  }
  const float scale = *scale_ptr;
  const int amax_seen = __ldcg(reinterpret_cast<const int *>(amax_ptr));
  load_cos_sin(q_freqs, k_freqs, blockIdx.x, smem);
  wait_barrier(&row_ready);
  const __nv_bfloat16 *row = reinterpret_cast<const __nv_bfloat16 *>(token_row);
  const GroupTasks tasks = group_tasks(q_split, k_split);
  __nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
  const int v_per_group = v_split / (2 * kVec);
  for (int task = threadIdx.x; task < h * v_per_group; task += blockDim.x) {
    const int g = task / v_per_group;
    const int p = (task % v_per_group) * 2 * kVec;
    forward_v_fp8(row + g * total_d + q_split + k_split + p, scale,
                  v_out + off.v + g * v_split + p, amax2);
  }
  for (int task = threadIdx.x; task < h * (tasks.q + tasks.k); task += blockDim.x) {
    const TokenPair pair = token_pair(task, tasks, total_d, q_split, k_split);
    uint8_t *dst = (pair.is_q ? q_out + off.q : k_out + off.k) + pair.split;
    forward_pair_fp8(pair, row, smem, scale, dst, amax2);
  }
  warp_amax_update(fmaxf(__low2float(amax2), __high2float(amax2)), amax_seen, amax_ptr);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    *scale_inv_ptr = __frcp_rn(scale);
  }
  end_barrier(&row_ready);
}

// FP8 dequantize and RoPE backward of one q or k pair task.
template <__nv_fp8_interpretation_t kFormat>
__device__ __forceinline__ void backward_pair_fp8(const TokenPair &task, const uint8_t *grad, const float *smem,
                                  float scale_inv, __nv_bfloat16 *dst) {
  const Bf16x8 src_lo =
      dequantize_fp8x8<kFormat>(*reinterpret_cast<const uint2 *>(grad), scale_inv);
  const Bf16x8 src_hi =
      dequantize_fp8x8<kFormat>(*reinterpret_cast<const uint2 *>(grad + kHalf), scale_inv);
  const float *cos_v = task.is_q ? smem : smem + 2 * kDim;
  Bf16x8 out_lo, out_hi;
  rope_backward_pair(src_lo, src_hi, cos_v, cos_v + kDim, task.j, out_lo, out_hi);
  store_vec(dst, out_lo);
  store_vec(dst + kHalf, out_hi);
}

// One block does one token. The bulk copies of the FP8 dq, dk and dv rows and the cos/sin table
// overlap.
template <__nv_fp8_interpretation_t kFormat>
__global__ void __launch_bounds__(kThreads)
    qkv_rope_backward_fp8_kernel(const uint8_t *__restrict__ dq, const uint8_t *__restrict__ dk,
                                 const uint8_t *__restrict__ dv,
                                 const float *__restrict__ scale_inv_ptr,
                                 const float *__restrict__ q_freqs,
                                 const float *__restrict__ k_freqs,
                                 __nv_bfloat16 *__restrict__ dqkv, int s, int b, int h,
                                 int q_split, int k_split, int v_split, bool sbhd) {
  extern __shared__ __align__(128) uint8_t token_row[];  // dq row, dk row, dv row
  __shared__ __align__(16) float smem[4 * kDim];
  __shared__ __align__(8) uint64_t row_ready;
  const RowOffsets off =
      token_offsets(blockIdx.x, blockIdx.y, s, b, h, q_split, k_split, v_split, sbhd);
  const int total_d = q_split + k_split + v_split;
  const int q_bytes = h * q_split;
  const int k_bytes = h * k_split;
  if (threadIdx.x == 0) {
    const int v_bytes = h * v_split;
    start_barrier(&row_ready, q_bytes + k_bytes + v_bytes);
    copy_to_shared(token_row, dq + off.q, q_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes, dk + off.k, k_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes + k_bytes, dv + off.v, v_bytes, &row_ready);
  }
  const float scale_inv = *scale_inv_ptr;
  load_cos_sin(q_freqs, k_freqs, blockIdx.x, smem);
  wait_barrier(&row_ready);
  const GroupTasks tasks = group_tasks(q_split, k_split);
  const int v_per_group = v_split / (2 * kVec);
  for (int task = threadIdx.x; task < h * v_per_group; task += blockDim.x) {
    const int g = task / v_per_group;
    const int p = (task % v_per_group) * 2 * kVec;
    const uint4 packed =
        *reinterpret_cast<const uint4 *>(token_row + q_bytes + k_bytes + g * v_split + p);
    __nv_bfloat16 *dst = dqkv + off.src + static_cast<long long>(g) * total_d + q_split +
                         k_split + p;
    store_vec(dst, dequantize_fp8x8<kFormat>(make_uint2(packed.x, packed.y), scale_inv));
    store_vec(dst + kVec, dequantize_fp8x8<kFormat>(make_uint2(packed.z, packed.w), scale_inv));
  }
  for (int task = threadIdx.x; task < h * (tasks.q + tasks.k); task += blockDim.x) {
    const TokenPair pair = token_pair(task, tasks, total_d, q_split, k_split);
    const uint8_t *grad = token_row + (pair.is_q ? 0 : q_bytes) + pair.split;
    backward_pair_fp8<kFormat>(pair, grad, smem, scale_inv, dqkv + off.src + pair.packed);
  }
  end_barrier(&row_ready);
}

// ---------------------------------------------------------------------------------------------
// Tuned RoPE variants. Same per-element math as the kernels above.
// ---------------------------------------------------------------------------------------------

constexpr int kSlots = kHalf / kVec;  // 8 slots of 8 elements in each half of a head

// Table values of one slot: cos and sin of the 8 elements in each half of a head.
struct SlotTable {
  Float8Values cos_lo;
  Float8Values cos_hi;
  Float8Values sin_lo;
  Float8Values sin_hi;
};

__device__ __forceinline__ SlotTable load_slot_table(const float *cos_v, int j) {
  const float *sin_v = cos_v + kDim;
  return {load_table8(cos_v + j), load_table8(cos_v + j + kHalf), load_table8(sin_v + j),
          load_table8(sin_v + j + kHalf)};
}

// Same arithmetic as rope_forward_pair_x2, with the table in registers.
__device__ __forceinline__ void rope_forward_pair_reg(const Bf16x8 &src_lo, const Bf16x8 &src_hi,
                                                      const SlotTable &t, __nv_bfloat162 *out_lo,
                                                      __nv_bfloat162 *out_hi) {
#pragma unroll
  for (int e = 0; e < kVec; e += 2) {
    float lo[2];
    float hi[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      {
        const float v_cos = t.cos_lo.v[e + i];
        const float v_sin = t.sin_lo.v[e + i];
        const float v_src = __bfloat162float(src_lo.v[e + i]);
        const float v_src_rotate = -__bfloat162float(src_hi.v[e + i]);
        lo[i] = v_src * v_cos + v_src_rotate * v_sin;
      }
      {
        const float v_cos = t.cos_hi.v[e + i];
        const float v_sin = t.sin_hi.v[e + i];
        const float v_src = __bfloat162float(src_hi.v[e + i]);
        const float v_src_rotate = __bfloat162float(src_lo.v[e + i]);
        hi[i] = v_src * v_cos + v_src_rotate * v_sin;
      }
    }
    out_lo[e / 2] = __floats2bfloat162_rn(lo[0], lo[1]);
    out_hi[e / 2] = __floats2bfloat162_rn(hi[0], hi[1]);
  }
}

// Same arithmetic as rope_backward_pair, with the table in registers.
__device__ __forceinline__ void rope_backward_pair_reg(const Bf16x8 &src_lo, const Bf16x8 &src_hi,
                                                       const SlotTable &t, Bf16x8 &out_lo,
                                                       Bf16x8 &out_hi) {
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    {
      const float v_src = __bfloat162float(src_lo.v[e]);
      const float v_cos = t.cos_lo.v[e];
      const float v_src_rotate = __bfloat162float(src_hi.v[e]);
      const float v_sin = t.sin_hi.v[e];
      out_lo.v[e] = __float2bfloat16(v_src * v_cos + v_src_rotate * v_sin);
    }
    {
      const float v_src = __bfloat162float(src_hi.v[e]);
      const float v_cos = t.cos_hi.v[e];
      const float v_src_rotate = __bfloat162float(src_lo.v[e]);
      const float v_sin = -t.sin_lo.v[e];
      out_hi.v[e] = __float2bfloat16(v_src * v_cos + v_src_rotate * v_sin);
    }
  }
}

// One head of one token. Head index hs runs over the q heads (group major), the k heads and the
// v heads. src is the element offset in the packed row. dst is the element offset in the q, k or
// v row of the token.
struct HeadTask {
  int kind;  // 0 = q, 1 = k, 2 = v
  int src;
  int dst;
};

__device__ __forceinline__ HeadTask head_task(int hs, int h, int qh, int total_d, int q_split,
                                              int k_split) {
  if (hs < h * qh) {
    const int g = hs / qh;
    return {0, g * total_d + (hs - g * qh) * kDim, hs * kDim};
  }
  hs -= h * qh;
  if (hs < h) {
    return {1, hs * total_d + q_split, hs * kDim};
  }
  hs -= h;
  return {2, hs * total_d + q_split + k_split, hs * kDim};
}

__device__ __forceinline__ Bf16x8 load_vec_nc(const __nv_bfloat16 *ptr) {
  const uint4 raw = __ldg(reinterpret_cast<const uint4 *>(ptr));
  return *reinterpret_cast<const Bf16x8 *>(&raw);
}

// Thread layout of the tuned kernels: thread t has slot j = t % 8 and head row t / 8. In
// iteration i it does head hs = (i * kThreads + t) / 8. All threads of one warp have the same
// head kind in almost all iterations.
struct RopeShape {
  int s, b, h, q_split, k_split, v_split, qh;
  bool sbhd;
};

// v2: the row comes with a TMA bulk copy, as in the base kernel. The slot table is in registers.
__global__ void __launch_bounds__(kThreads)
    rope_fwd_fp8_v2_kernel(const __nv_bfloat16 *__restrict__ qkv, const float *__restrict__ q_freqs,
                           const float *__restrict__ k_freqs, uint8_t *__restrict__ q_out,
                           uint8_t *__restrict__ k_out, uint8_t *__restrict__ v_out,
                           const float *__restrict__ scale_ptr, float *__restrict__ amax_ptr,
                           float *__restrict__ scale_inv_ptr, RopeShape sh) {
  extern __shared__ __align__(128) uint8_t token_row[];
  __shared__ __align__(16) float smem[4 * kDim];
  __shared__ __align__(8) uint64_t row_ready;
  const RowOffsets off = token_offsets(blockIdx.x, blockIdx.y, sh.s, sh.b, sh.h, sh.q_split,
                                       sh.k_split, sh.v_split, sh.sbhd);
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  if (threadIdx.x == 0) {
    const uint32_t bytes = sh.h * total_d * sizeof(__nv_bfloat16);
    start_barrier(&row_ready, bytes);
    copy_to_shared(token_row, qkv + off.src, bytes, &row_ready);
  }
  const float scale = *scale_ptr;
  const int amax_seen = __ldcg(reinterpret_cast<const int *>(amax_ptr));
  load_cos_sin(q_freqs, k_freqs, blockIdx.x, smem);
  const int slot = (threadIdx.x % kSlots) * kVec;
  SlotTable table = load_slot_table(smem, slot);
  int table_kind = 0;
  wait_barrier(&row_ready);
  const __nv_bfloat16 *row = reinterpret_cast<const __nv_bfloat16 *>(token_row);
  __nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
  const int heads = sh.h * (sh.qh + 2);
  for (int hs = threadIdx.x / kSlots; hs < heads; hs += kThreads / kSlots) {
    const HeadTask task = head_task(hs, sh.h, sh.qh, total_d, sh.q_split, sh.k_split);
    if (task.kind == 2) {
      forward_v_fp8(row + task.src + 2 * slot, scale, v_out + off.v + task.dst + 2 * slot, amax2);
      continue;
    }
    if (task.kind != table_kind) {
      table = load_slot_table(smem + 2 * kDim * task.kind, slot);
      table_kind = task.kind;
    }
    const Bf16x8 src_lo = load_vec(row + task.src + slot);
    const Bf16x8 src_hi = load_vec(row + task.src + slot + kHalf);
    __nv_bfloat162 out_lo[kVec / 2];
    __nv_bfloat162 out_hi[kVec / 2];
    rope_forward_pair_reg(src_lo, src_hi, table, out_lo, out_hi);
    uint8_t *dst = (task.kind == 0 ? q_out + off.q : k_out + off.k) + task.dst + slot;
    *reinterpret_cast<uint2 *>(dst) = cast_fp8x8(out_lo, scale, amax2);
    *reinterpret_cast<uint2 *>(dst + kHalf) = cast_fp8x8(out_hi, scale, amax2);
  }
  warp_amax_update(fmaxf(__low2float(amax2), __high2float(amax2)), amax_seen, amax_ptr);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    *scale_inv_ptr = __frcp_rn(scale);
  }
  end_barrier(&row_ready);
}

// v5: direct global loads of all tasks first, then the table, then the math. kTasks is the
// number of iterations of one thread (host: ceil(8 * h * (qh + 2) / kThreads)).
template <int kTasks, int kMinBlocks>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    rope_fwd_fp8_v5_kernel(const __nv_bfloat16 *__restrict__ qkv, const float *__restrict__ q_freqs,
                           const float *__restrict__ k_freqs, uint8_t *__restrict__ q_out,
                           uint8_t *__restrict__ k_out, uint8_t *__restrict__ v_out,
                           const float *__restrict__ scale_ptr, float *__restrict__ amax_ptr,
                           float *__restrict__ scale_inv_ptr, RopeShape sh) {
  __shared__ __align__(16) float smem[4 * kDim];
  const RowOffsets off = token_offsets(blockIdx.x, blockIdx.y, sh.s, sh.b, sh.h, sh.q_split,
                                       sh.k_split, sh.v_split, sh.sbhd);
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  const int heads = sh.h * (sh.qh + 2);
  const int slot = (threadIdx.x % kSlots) * kVec;
  const __nv_bfloat16 *row = qkv + off.src;
  HeadTask task[kTasks];
  Bf16x8 lo[kTasks];
  Bf16x8 hi[kTasks];
#pragma unroll
  for (int i = 0; i < kTasks; ++i) {
    const int hs = (i * kThreads + threadIdx.x) / kSlots;
    task[i].kind = 3;
    if (hs < heads) {
      task[i] = head_task(hs, sh.h, sh.qh, total_d, sh.q_split, sh.k_split);
      const int base = task[i].src + (task[i].kind == 2 ? 2 * slot : slot);
      lo[i] = load_vec_nc(row + base);
      hi[i] = load_vec_nc(row + base + (task[i].kind == 2 ? kVec : kHalf));
    }
  }
  const float scale = *scale_ptr;
  const int amax_seen = __ldcg(reinterpret_cast<const int *>(amax_ptr));
  load_cos_sin(q_freqs, k_freqs, blockIdx.x, smem);
  SlotTable table = load_slot_table(smem, slot);
  int table_kind = 0;
  __nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
#pragma unroll
  for (int i = 0; i < kTasks; ++i) {
    if (task[i].kind == 3) {
      continue;
    }
    if (task[i].kind == 2) {
      const uint2 first = cast_fp8x8(reinterpret_cast<const __nv_bfloat162 *>(&lo[i]), scale, amax2);
      const uint2 second = cast_fp8x8(reinterpret_cast<const __nv_bfloat162 *>(&hi[i]), scale, amax2);
      *reinterpret_cast<uint4 *>(v_out + off.v + task[i].dst + 2 * slot) =
          make_uint4(first.x, first.y, second.x, second.y);
      continue;
    }
    if (task[i].kind != table_kind) {
      table = load_slot_table(smem + 2 * kDim * task[i].kind, slot);
      table_kind = task[i].kind;
    }
    __nv_bfloat162 out_lo[kVec / 2];
    __nv_bfloat162 out_hi[kVec / 2];
    rope_forward_pair_reg(lo[i], hi[i], table, out_lo, out_hi);
    uint8_t *dst = (task[i].kind == 0 ? q_out + off.q : k_out + off.k) + task[i].dst + slot;
    *reinterpret_cast<uint2 *>(dst) = cast_fp8x8(out_lo, scale, amax2);
    *reinterpret_cast<uint2 *>(dst + kHalf) = cast_fp8x8(out_hi, scale, amax2);
  }
  warp_amax_update(fmaxf(__low2float(amax2), __high2float(amax2)), amax_seen, amax_ptr);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    *scale_inv_ptr = __frcp_rn(scale);
  }
}

// Backward v2: TMA rows (as the base kernel), slot table in registers, uniform head mapping.
template <__nv_fp8_interpretation_t kFormat>
__global__ void __launch_bounds__(kThreads)
    rope_bwd_fp8_v2_kernel(const uint8_t *__restrict__ dq, const uint8_t *__restrict__ dk,
                           const uint8_t *__restrict__ dv, const float *__restrict__ scale_inv_ptr,
                           const float *__restrict__ q_freqs, const float *__restrict__ k_freqs,
                           __nv_bfloat16 *__restrict__ dqkv, RopeShape sh) {
  extern __shared__ __align__(128) uint8_t token_row[];  // dq row, dk row, dv row
  __shared__ __align__(16) float smem[4 * kDim];
  __shared__ __align__(8) uint64_t row_ready;
  const RowOffsets off = token_offsets(blockIdx.x, blockIdx.y, sh.s, sh.b, sh.h, sh.q_split,
                                       sh.k_split, sh.v_split, sh.sbhd);
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  const int q_bytes = sh.h * sh.q_split;
  const int k_bytes = sh.h * sh.k_split;
  if (threadIdx.x == 0) {
    const int v_bytes = sh.h * sh.v_split;
    start_barrier(&row_ready, q_bytes + k_bytes + v_bytes);
    copy_to_shared(token_row, dq + off.q, q_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes, dk + off.k, k_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes + k_bytes, dv + off.v, v_bytes, &row_ready);
  }
  const float scale_inv = *scale_inv_ptr;
  load_cos_sin(q_freqs, k_freqs, blockIdx.x, smem);
  const int slot = (threadIdx.x % kSlots) * kVec;
  SlotTable table = load_slot_table(smem, slot);
  int table_kind = 0;
  wait_barrier(&row_ready);
  const int heads = sh.h * (sh.qh + 2);
  for (int hs = threadIdx.x / kSlots; hs < heads; hs += kThreads / kSlots) {
    const HeadTask task = head_task(hs, sh.h, sh.qh, total_d, sh.q_split, sh.k_split);
    __nv_bfloat16 *dst = dqkv + off.src + task.src;
    if (task.kind == 2) {
      const uint4 packed =
          *reinterpret_cast<const uint4 *>(token_row + q_bytes + k_bytes + task.dst + 2 * slot);
      store_vec(dst + 2 * slot, dequantize_fp8x8<kFormat>(make_uint2(packed.x, packed.y), scale_inv));
      store_vec(dst + 2 * slot + kVec,
                dequantize_fp8x8<kFormat>(make_uint2(packed.z, packed.w), scale_inv));
      continue;
    }
    if (task.kind != table_kind) {
      table = load_slot_table(smem + 2 * kDim * task.kind, slot);
      table_kind = task.kind;
    }
    const uint8_t *grad = token_row + (task.kind == 0 ? 0 : q_bytes) + task.dst + slot;
    const Bf16x8 src_lo = dequantize_fp8x8<kFormat>(*reinterpret_cast<const uint2 *>(grad), scale_inv);
    const Bf16x8 src_hi =
        dequantize_fp8x8<kFormat>(*reinterpret_cast<const uint2 *>(grad + kHalf), scale_inv);
    Bf16x8 out_lo, out_hi;
    rope_backward_pair_reg(src_lo, src_hi, table, out_lo, out_hi);
    store_vec(dst + slot, out_lo);
    store_vec(dst + slot + kHalf, out_hi);
  }
  end_barrier(&row_ready);
}

__device__ __forceinline__ uint2 load_u2_nc(const uint8_t *ptr) {
  return __ldg(reinterpret_cast<const uint2 *>(ptr));
}

// Backward v5: direct global loads of all tasks first.
template <__nv_fp8_interpretation_t kFormat, int kTasks, int kMinBlocks>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    rope_bwd_fp8_v5_kernel(const uint8_t *__restrict__ dq, const uint8_t *__restrict__ dk,
                           const uint8_t *__restrict__ dv, const float *__restrict__ scale_inv_ptr,
                           const float *__restrict__ q_freqs, const float *__restrict__ k_freqs,
                           __nv_bfloat16 *__restrict__ dqkv, RopeShape sh) {
  __shared__ __align__(16) float smem[4 * kDim];
  const RowOffsets off = token_offsets(blockIdx.x, blockIdx.y, sh.s, sh.b, sh.h, sh.q_split,
                                       sh.k_split, sh.v_split, sh.sbhd);
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  const int heads = sh.h * (sh.qh + 2);
  const int slot = (threadIdx.x % kSlots) * kVec;
  HeadTask task[kTasks];
  uint2 lo[kTasks];
  uint2 hi[kTasks];
#pragma unroll
  for (int i = 0; i < kTasks; ++i) {
    const int hs = (i * kThreads + threadIdx.x) / kSlots;
    task[i].kind = 3;
    if (hs < heads) {
      task[i] = head_task(hs, sh.h, sh.qh, total_d, sh.q_split, sh.k_split);
      if (task[i].kind == 2) {
        const uint4 packed = __ldg(reinterpret_cast<const uint4 *>(dv + off.v + task[i].dst + 2 * slot));
        lo[i] = make_uint2(packed.x, packed.y);
        hi[i] = make_uint2(packed.z, packed.w);
      } else {
        const uint8_t *grad = (task[i].kind == 0 ? dq + off.q : dk + off.k) + task[i].dst + slot;
        lo[i] = load_u2_nc(grad);
        hi[i] = load_u2_nc(grad + kHalf);
      }
    }
  }
  const float scale_inv = *scale_inv_ptr;
  load_cos_sin(q_freqs, k_freqs, blockIdx.x, smem);
  SlotTable table = load_slot_table(smem, slot);
  int table_kind = 0;
#pragma unroll
  for (int i = 0; i < kTasks; ++i) {
    if (task[i].kind == 3) {
      continue;
    }
    __nv_bfloat16 *dst = dqkv + off.src + task[i].src;
    if (task[i].kind == 2) {
      store_vec(dst + 2 * slot, dequantize_fp8x8<kFormat>(lo[i], scale_inv));
      store_vec(dst + 2 * slot + kVec, dequantize_fp8x8<kFormat>(hi[i], scale_inv));
      continue;
    }
    if (task[i].kind != table_kind) {
      table = load_slot_table(smem + 2 * kDim * task[i].kind, slot);
      table_kind = task[i].kind;
    }
    const Bf16x8 src_lo = dequantize_fp8x8<kFormat>(lo[i], scale_inv);
    const Bf16x8 src_hi = dequantize_fp8x8<kFormat>(hi[i], scale_inv);
    Bf16x8 out_lo, out_hi;
    rope_backward_pair_reg(src_lo, src_hi, table, out_lo, out_hi);
    store_vec(dst + slot, out_lo);
    store_vec(dst + slot + kHalf, out_hi);
  }
}

// Pipelined persistent RoPE kernels (variants 3 and 4): each block loops over tokens with a
// ring of kStages row buffers and tables. The TMA copy and the table of the next token overlap
// with the math of the current token.

__device__ __forceinline__ void arm_barrier(uint64_t *barrier, uint32_t bytes) {
#if LORA_ATTN_HAS_BULK_COPY
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(shared_address(barrier)),
               "r"(bytes)
               : "memory");
#else
  __trap();
#endif
}

__device__ __forceinline__ void wait_barrier_parity(uint64_t *barrier, uint32_t parity) {
#if LORA_ATTN_HAS_BULK_COPY
  const uint32_t address = shared_address(barrier);
  uint32_t done = 0;
  while (done == 0) {
    asm volatile(
        "{\n"
        ".reg .pred complete;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 complete, [%1], %2;\n"
        "selp.u32 %0, 1, 0, complete;\n"
        "}\n"
        : "=r"(done)
        : "r"(address), "r"(parity)
        : "memory");
  }
#else
  __trap();
#endif
}

// Token t of a block in iteration i: t = blockIdx.x + i * gridDim.x. Returns (s_id, b_id).
__device__ __forceinline__ int2 token_position(int t, const RopeShape &sh) {
  return sh.sbhd ? make_int2(t / sh.b, t % sh.b) : make_int2(t % sh.s, t / sh.s);
}

// Cos and sin of one token into one table of the ring, from the freqs that this thread holds
// in registers (one q and one k angle for each thread). The caller syncs the threads.
static_assert(kThreads == kDim, "one table angle for each thread");

__device__ __forceinline__ void fill_table(float fq, float fk, float *table) {
  sincosf(fq, &table[kDim + threadIdx.x], &table[threadIdx.x]);
  sincosf(fk, &table[3 * kDim + threadIdx.x], &table[2 * kDim + threadIdx.x]);
}

constexpr int kTableFloats = 4 * kDim;

template <int kStages>
__global__ void __launch_bounds__(kThreads)
    rope_fwd_fp8_pipe_kernel(const __nv_bfloat16 *__restrict__ qkv, const float *__restrict__ q_freqs,
                             const float *__restrict__ k_freqs, uint8_t *__restrict__ q_out,
                             uint8_t *__restrict__ k_out, uint8_t *__restrict__ v_out,
                             const float *__restrict__ scale_ptr, float *__restrict__ amax_ptr,
                             float *__restrict__ scale_inv_ptr, RopeShape sh, int row_stride,
                             int tokens) {
  extern __shared__ __align__(128) uint8_t ring[];
  __shared__ __align__(8) uint64_t bars[kStages];
  float *tables = reinterpret_cast<float *>(ring + kStages * row_stride);
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  const uint32_t row_bytes = sh.h * total_d * sizeof(__nv_bfloat16);
  const int heads = sh.h * (sh.qh + 2);
  const int slot = (threadIdx.x % kSlots) * kVec;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(shared_address(&bars[s])) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const float scale = *scale_ptr;
  const int amax_seen = __ldcg(reinterpret_cast<const int *>(amax_ptr));
  __nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
  const int iters = (tokens - static_cast<int>(blockIdx.x) + static_cast<int>(gridDim.x) - 1) /
                    static_cast<int>(gridDim.x);
  float fq = 0.f;
  float fk = 0.f;
  auto prefetch = [&](int i) {
    const int t = blockIdx.x + i * gridDim.x;
    if (t < tokens) {
      const int s_id = token_position(t, sh).x;
      fq = __ldg(q_freqs + s_id * kDim + threadIdx.x);
      fk = __ldg(k_freqs + s_id * kDim + threadIdx.x);
    }
  };
  auto issue = [&](int i) {
    const int t = blockIdx.x + i * gridDim.x;
    if (t >= tokens) {
      return;
    }
    const int2 pos = token_position(t, sh);
    const int s = i % kStages;
    if (threadIdx.x == 0) {
      const RowOffsets off = token_offsets(pos.x, pos.y, sh.s, sh.b, sh.h, sh.q_split, sh.k_split,
                                           sh.v_split, sh.sbhd);
      arm_barrier(&bars[s], row_bytes);
      copy_to_shared(ring + s * row_stride, qkv + off.src, row_bytes, &bars[s]);
    }
    fill_table(fq, fk, tables + s * kTableFloats);
  };
  prefetch(0);
#pragma unroll
  for (int p = 0; p < kStages - 1; ++p) {
    issue(p);
    prefetch(p + 1);
  }
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    issue(i + kStages - 1);
    prefetch(i + kStages);
    const int s = i % kStages;
    wait_barrier_parity(&bars[s], (i / kStages) & 1);
    const int t = blockIdx.x + i * gridDim.x;
    const int2 pos = token_position(t, sh);
    const RowOffsets off = token_offsets(pos.x, pos.y, sh.s, sh.b, sh.h, sh.q_split, sh.k_split,
                                         sh.v_split, sh.sbhd);
    const float *table_base = tables + s * kTableFloats;
    SlotTable table = load_slot_table(table_base, slot);
    int table_kind = 0;
    const __nv_bfloat16 *row = reinterpret_cast<const __nv_bfloat16 *>(ring + s * row_stride);
    for (int hs = threadIdx.x / kSlots; hs < heads; hs += kThreads / kSlots) {
      const HeadTask task = head_task(hs, sh.h, sh.qh, total_d, sh.q_split, sh.k_split);
      if (task.kind == 2) {
        forward_v_fp8(row + task.src + 2 * slot, scale, v_out + off.v + task.dst + 2 * slot, amax2);
        continue;
      }
      if (task.kind != table_kind) {
        table = load_slot_table(table_base + 2 * kDim * task.kind, slot);
        table_kind = task.kind;
      }
      const Bf16x8 src_lo = load_vec(row + task.src + slot);
      const Bf16x8 src_hi = load_vec(row + task.src + slot + kHalf);
      __nv_bfloat162 out_lo[kVec / 2];
      __nv_bfloat162 out_hi[kVec / 2];
      rope_forward_pair_reg(src_lo, src_hi, table, out_lo, out_hi);
      uint8_t *dst = (task.kind == 0 ? q_out + off.q : k_out + off.k) + task.dst + slot;
      *reinterpret_cast<uint2 *>(dst) = cast_fp8x8(out_lo, scale, amax2);
      *reinterpret_cast<uint2 *>(dst + kHalf) = cast_fp8x8(out_hi, scale, amax2);
    }
    __syncthreads();
  }
  warp_amax_update(fmaxf(__low2float(amax2), __high2float(amax2)), amax_seen, amax_ptr);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *scale_inv_ptr = __frcp_rn(scale);
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared_address(&bars[s])) : "memory");
    }
  }
}

template <__nv_fp8_interpretation_t kFormat, int kStages>
__global__ void __launch_bounds__(kThreads)
    rope_bwd_fp8_pipe_kernel(const uint8_t *__restrict__ dq, const uint8_t *__restrict__ dk,
                             const uint8_t *__restrict__ dv, const float *__restrict__ scale_inv_ptr,
                             const float *__restrict__ q_freqs, const float *__restrict__ k_freqs,
                             __nv_bfloat16 *__restrict__ dqkv, RopeShape sh, int row_stride,
                             int tokens) {
  extern __shared__ __align__(128) uint8_t ring[];  // dq row, dk row, dv row of each stage
  __shared__ __align__(8) uint64_t bars[kStages];
  float *tables = reinterpret_cast<float *>(ring + kStages * row_stride);
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  const int q_bytes = sh.h * sh.q_split;
  const int k_bytes = sh.h * sh.k_split;
  const int v_bytes = sh.h * sh.v_split;
  const int heads = sh.h * (sh.qh + 2);
  const int slot = (threadIdx.x % kSlots) * kVec;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(shared_address(&bars[s])) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  const float scale_inv = *scale_inv_ptr;
  const int iters = (tokens - static_cast<int>(blockIdx.x) + static_cast<int>(gridDim.x) - 1) /
                    static_cast<int>(gridDim.x);
  float fq = 0.f;
  float fk = 0.f;
  auto prefetch = [&](int i) {
    const int t = blockIdx.x + i * gridDim.x;
    if (t < tokens) {
      const int s_id = token_position(t, sh).x;
      fq = __ldg(q_freqs + s_id * kDim + threadIdx.x);
      fk = __ldg(k_freqs + s_id * kDim + threadIdx.x);
    }
  };
  auto issue = [&](int i) {
    const int t = blockIdx.x + i * gridDim.x;
    if (t >= tokens) {
      return;
    }
    const int2 pos = token_position(t, sh);
    const int s = i % kStages;
    if (threadIdx.x == 0) {
      const RowOffsets off = token_offsets(pos.x, pos.y, sh.s, sh.b, sh.h, sh.q_split, sh.k_split,
                                           sh.v_split, sh.sbhd);
      uint8_t *buf = ring + s * row_stride;
      arm_barrier(&bars[s], q_bytes + k_bytes + v_bytes);
      copy_to_shared(buf, dq + off.q, q_bytes, &bars[s]);
      copy_to_shared(buf + q_bytes, dk + off.k, k_bytes, &bars[s]);
      copy_to_shared(buf + q_bytes + k_bytes, dv + off.v, v_bytes, &bars[s]);
    }
    fill_table(fq, fk, tables + s * kTableFloats);
  };
  prefetch(0);
#pragma unroll
  for (int p = 0; p < kStages - 1; ++p) {
    issue(p);
    prefetch(p + 1);
  }
  __syncthreads();
  for (int i = 0; i < iters; ++i) {
    issue(i + kStages - 1);
    prefetch(i + kStages);
    const int s = i % kStages;
    wait_barrier_parity(&bars[s], (i / kStages) & 1);
    const int t = blockIdx.x + i * gridDim.x;
    const int2 pos = token_position(t, sh);
    const RowOffsets off = token_offsets(pos.x, pos.y, sh.s, sh.b, sh.h, sh.q_split, sh.k_split,
                                         sh.v_split, sh.sbhd);
    const float *table_base = tables + s * kTableFloats;
    SlotTable table = load_slot_table(table_base, slot);
    int table_kind = 0;
    const uint8_t *buf = ring + s * row_stride;
    for (int hs = threadIdx.x / kSlots; hs < heads; hs += kThreads / kSlots) {
      const HeadTask task = head_task(hs, sh.h, sh.qh, total_d, sh.q_split, sh.k_split);
      __nv_bfloat16 *dst = dqkv + off.src + task.src;
      if (task.kind == 2) {
        const uint4 packed =
            *reinterpret_cast<const uint4 *>(buf + q_bytes + k_bytes + task.dst + 2 * slot);
        store_vec(dst + 2 * slot, dequantize_fp8x8<kFormat>(make_uint2(packed.x, packed.y), scale_inv));
        store_vec(dst + 2 * slot + kVec,
                  dequantize_fp8x8<kFormat>(make_uint2(packed.z, packed.w), scale_inv));
        continue;
      }
      if (task.kind != table_kind) {
        table = load_slot_table(table_base + 2 * kDim * task.kind, slot);
        table_kind = task.kind;
      }
      const uint8_t *grad = buf + (task.kind == 0 ? 0 : q_bytes) + task.dst + slot;
      const Bf16x8 src_lo = dequantize_fp8x8<kFormat>(*reinterpret_cast<const uint2 *>(grad), scale_inv);
      const Bf16x8 src_hi =
          dequantize_fp8x8<kFormat>(*reinterpret_cast<const uint2 *>(grad + kHalf), scale_inv);
      Bf16x8 out_lo, out_hi;
      rope_backward_pair_reg(src_lo, src_hi, table, out_lo, out_hi);
      store_vec(dst + slot, out_lo);
      store_vec(dst + slot + kHalf, out_hi);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(shared_address(&bars[s])) : "memory");
    }
  }
}

// ---------------------------------------------------------------------------------------------
// RoPE v6: TMA row, slot table in registers, f32x2 math (two lanes in each FMA, MUL and max),
// a power-of-two head split and one sincos table when q and k share the freqs. Each f32x2 lane
// rounds as the scalar instruction, thus the bytes stay the same as in the TE kernels.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ uint64_t pk2(const float lo, const float hi) {
  uint64_t v;
  asm("mov.b64 %0, {%1, %2};" : "=l"(v) : "f"(lo), "f"(hi));
  return v;
}

__device__ __forceinline__ float pk2_lo(const uint64_t v) {
  return __uint_as_float(static_cast<uint32_t>(v));
}

__device__ __forceinline__ float pk2_hi(const uint64_t v) {
  return __uint_as_float(static_cast<uint32_t>(v >> 32));
}

__device__ __forceinline__ uint64_t fma2x(const uint64_t a, const uint64_t b, const uint64_t c) {
  uint64_t d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
  return d;
}

__device__ __forceinline__ uint64_t mul2x(const uint64_t a, const uint64_t b) {
  uint64_t d;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}

// Two bf16 values of one word as two floats (exact).
__device__ __forceinline__ uint64_t bf16x2_to_f32x2(const uint32_t w) {
  return pk2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

// bf16x2 word of two floats (round to nearest), as __floats2bfloat162_rn.
__device__ __forceinline__ uint32_t f32x2_to_bf16x2(const uint64_t v) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(pk2_lo(v), pk2_hi(v));
  return *reinterpret_cast<const uint32_t *>(&pair);
}

// Slot table for v6: cos of both halves, -sin of the first half, sin of the second half.
// Pairs of elements e, e+1 are packed for the f32x2 math.
struct SlotTable2 {
  uint64_t cos_lo[kVec / 2];
  uint64_t cos_hi[kVec / 2];
  uint64_t neg_sin_lo[kVec / 2];
  uint64_t sin_hi[kVec / 2];
};

__device__ __forceinline__ SlotTable2 load_slot_table2(const float *cos_v, int j) {
  const SlotTable t = load_slot_table(cos_v, j);
  SlotTable2 out;
#pragma unroll
  for (int p = 0; p < kVec / 2; ++p) {
    out.cos_lo[p] = pk2(t.cos_lo.v[2 * p], t.cos_lo.v[2 * p + 1]);
    out.cos_hi[p] = pk2(t.cos_hi.v[2 * p], t.cos_hi.v[2 * p + 1]);
    out.neg_sin_lo[p] = pk2(-t.sin_lo.v[2 * p], -t.sin_lo.v[2 * p + 1]);
    out.sin_hi[p] = pk2(t.sin_hi.v[2 * p], t.sin_hi.v[2 * p + 1]);
  }
  return out;
}

// a * b + c * d with one FMUL and one FFMA, as nvcc contracts the TE expression. kFmaFirst
// selects which product is fused: fma(a, b, c * d) or fma(c, d, a * b).
template <bool kFmaFirst>
__device__ __forceinline__ uint64_t mul_add2(const uint64_t a, const uint64_t b, const uint64_t c,
                                             const uint64_t d) {
  if constexpr (kFmaFirst) {
    return fma2x(a, b, mul2x(c, d));
  } else {
    return fma2x(c, d, mul2x(a, b));
  }
}

// Forward: lo = src_lo * cos_lo + (-src_hi) * sin_lo = src_lo * cos_lo + src_hi * (-sin_lo),
//          hi = src_hi * cos_hi + src_lo * sin_hi. Output: bf16x2 words.
template <bool kFmaFirst>
__device__ __forceinline__ void rope_forward_x2(const Bf16x8 &src_lo, const Bf16x8 &src_hi,
                                                const SlotTable2 &t, uint32_t *out_lo,
                                                uint32_t *out_hi) {
  const uint32_t *lo_w = reinterpret_cast<const uint32_t *>(&src_lo);
  const uint32_t *hi_w = reinterpret_cast<const uint32_t *>(&src_hi);
#pragma unroll
  for (int p = 0; p < kVec / 2; ++p) {
    const uint64_t a = bf16x2_to_f32x2(lo_w[p]);
    const uint64_t b = bf16x2_to_f32x2(hi_w[p]);
    out_lo[p] = f32x2_to_bf16x2(mul_add2<kFmaFirst>(a, t.cos_lo[p], b, t.neg_sin_lo[p]));
    out_hi[p] = f32x2_to_bf16x2(mul_add2<kFmaFirst>(b, t.cos_hi[p], a, t.sin_hi[p]));
  }
}

// Backward: lo = a * cos_lo + b * sin_hi, hi = b * cos_hi + a * (-sin_lo).
template <bool kFmaFirst>
__device__ __forceinline__ void rope_backward_x2(const Bf16x8 &src_lo, const Bf16x8 &src_hi,
                                                 const SlotTable2 &t, uint32_t *out_lo,
                                                 uint32_t *out_hi) {
  const uint32_t *lo_w = reinterpret_cast<const uint32_t *>(&src_lo);
  const uint32_t *hi_w = reinterpret_cast<const uint32_t *>(&src_hi);
#pragma unroll
  for (int p = 0; p < kVec / 2; ++p) {
    const uint64_t a = bf16x2_to_f32x2(lo_w[p]);
    const uint64_t b = bf16x2_to_f32x2(hi_w[p]);
    out_lo[p] = f32x2_to_bf16x2(mul_add2<kFmaFirst>(a, t.cos_lo[p], b, t.sin_hi[p]));
    out_hi[p] = f32x2_to_bf16x2(mul_add2<kFmaFirst>(b, t.cos_hi[p], a, t.neg_sin_lo[p]));
  }
}

// TE cast of 8 bf16 values (4 words): amax on the bf16 values, out = fp8e4m3(float(x) * scale).
__device__ __forceinline__ uint2 cast_fp8x8_x2(const uint32_t *words, const uint64_t scale2,
                                               __nv_bfloat162 &amax2) {
  uint32_t out[4];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162 *>(&words[p]);
    amax2 = __hmax2(amax2, __habs2(x));
    const uint64_t o = mul2x(bf16x2_to_f32x2(words[p]), scale2);
    out[p] = __nv_cvt_float2_to_fp8x2(make_float2(pk2_lo(o), pk2_hi(o)), __NV_SATFINITE, __NV_E4M3);
  }
  return make_uint2(out[0] | (out[1] << 16), out[2] | (out[3] << 16));
}

// TE FP8 dequantize of 8 values: bf16(float(x) * scale_inv), two at a time.
template <__nv_fp8_interpretation_t kFormat>
__device__ __forceinline__ Bf16x8 dequantize_fp8x8_x2(uint2 raw, const uint64_t scale_inv2) {
  Bf16x8 out;
  uint32_t *out_w = reinterpret_cast<uint32_t *>(&out);
  const __nv_fp8x2_storage_t *pairs = reinterpret_cast<const __nv_fp8x2_storage_t *>(&raw);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) {
    const __half2 values(__nv_cvt_fp8x2_to_halfraw2(pairs[i], kFormat));
    const float2 f = __half22float2(values);
    out_w[i] = f32x2_to_bf16x2(mul2x(pk2(f.x, f.y), scale_inv2));
  }
  return out;
}

template <int kQh>
__device__ __forceinline__ HeadTask head_task_pow2(int hs, int h, int total_d, int q_split,
                                                   int k_split) {
  static_assert((kQh & (kQh - 1)) == 0, "kQh must be a power of two");
  if (hs < h * kQh) {
    const int g = hs / kQh;
    return {0, g * total_d + (hs - g * kQh) * kDim, hs * kDim};
  }
  hs -= h * kQh;
  if (hs < h) {
    return {1, hs * total_d + q_split, hs * kDim};
  }
  hs -= h;
  return {2, hs * total_d + q_split + k_split, hs * kDim};
}

// One sincos table (q) or two (q and k) into smem, then the threads sync.
template <bool kSameFreqs>
__device__ __forceinline__ void load_cos_sin_v6(const float *q_freqs, const float *k_freqs, int s_id,
                                                float *smem) {
  sincosf(q_freqs[s_id * kDim + threadIdx.x], &smem[kDim + threadIdx.x], &smem[threadIdx.x]);
  if constexpr (!kSameFreqs) {
    sincosf(k_freqs[s_id * kDim + threadIdx.x], &smem[3 * kDim + threadIdx.x],
            &smem[2 * kDim + threadIdx.x]);
  }
  __syncthreads();
}

template <int kQh, bool kSameFreqs, bool kFmaFirst>
__global__ void __launch_bounds__(kThreads)
    rope_fwd_fp8_v6_kernel(const __nv_bfloat16 *__restrict__ qkv, const float *__restrict__ q_freqs,
                           const float *__restrict__ k_freqs, uint8_t *__restrict__ q_out,
                           uint8_t *__restrict__ k_out, uint8_t *__restrict__ v_out,
                           const float *__restrict__ scale_ptr, float *__restrict__ amax_ptr,
                           float *__restrict__ scale_inv_ptr, RopeShape sh) {
  extern __shared__ __align__(128) uint8_t token_row[];
  __shared__ __align__(16) float smem[4 * kDim];
  __shared__ __align__(8) uint64_t row_ready;
  const RowOffsets off = token_offsets(blockIdx.x, blockIdx.y, sh.s, sh.b, sh.h, sh.q_split,
                                       sh.k_split, sh.v_split, sh.sbhd);
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  if (threadIdx.x == 0) {
    const uint32_t bytes = sh.h * total_d * sizeof(__nv_bfloat16);
    start_barrier(&row_ready, bytes);
    copy_to_shared(token_row, qkv + off.src, bytes, &row_ready);
  }
  const float scale = *scale_ptr;
  const int amax_seen = __ldcg(reinterpret_cast<const int *>(amax_ptr));
  load_cos_sin_v6<kSameFreqs>(q_freqs, k_freqs, blockIdx.x, smem);
  const int slot = (threadIdx.x % kSlots) * kVec;
  SlotTable2 table = load_slot_table2(smem, slot);
  int table_kind = 0;
  const uint64_t scale2 = pk2(scale, scale);
  wait_barrier(&row_ready);
  const __nv_bfloat16 *row = reinterpret_cast<const __nv_bfloat16 *>(token_row);
  __nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
  const int heads = sh.h * (kQh + 2);
  for (int hs = threadIdx.x / kSlots; hs < heads; hs += kThreads / kSlots) {
    const HeadTask task = head_task_pow2<kQh>(hs, sh.h, total_d, sh.q_split, sh.k_split);
    if (task.kind == 2) {
      const Bf16x8 first = load_vec(row + task.src + 2 * slot);
      const Bf16x8 second = load_vec(row + task.src + 2 * slot + kVec);
      const uint2 a = cast_fp8x8_x2(reinterpret_cast<const uint32_t *>(&first), scale2, amax2);
      const uint2 b = cast_fp8x8_x2(reinterpret_cast<const uint32_t *>(&second), scale2, amax2);
      *reinterpret_cast<uint4 *>(v_out + off.v + task.dst + 2 * slot) = make_uint4(a.x, a.y, b.x, b.y);
      continue;
    }
    if (!kSameFreqs && task.kind != table_kind) {
      table = load_slot_table2(smem + 2 * kDim * task.kind, slot);
      table_kind = task.kind;
    }
    const Bf16x8 src_lo = load_vec(row + task.src + slot);
    const Bf16x8 src_hi = load_vec(row + task.src + slot + kHalf);
    uint32_t out_lo[kVec / 2];
    uint32_t out_hi[kVec / 2];
    rope_forward_x2<kFmaFirst>(src_lo, src_hi, table, out_lo, out_hi);
    uint8_t *dst = (task.kind == 0 ? q_out + off.q : k_out + off.k) + task.dst + slot;
    *reinterpret_cast<uint2 *>(dst) = cast_fp8x8_x2(out_lo, scale2, amax2);
    *reinterpret_cast<uint2 *>(dst + kHalf) = cast_fp8x8_x2(out_hi, scale2, amax2);
  }
  warp_amax_update(fmaxf(__low2float(amax2), __high2float(amax2)), amax_seen, amax_ptr);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    *scale_inv_ptr = __frcp_rn(scale);
  }
  end_barrier(&row_ready);
}

template <__nv_fp8_interpretation_t kFormat, int kQh, bool kSameFreqs, bool kFmaFirst>
__global__ void __launch_bounds__(kThreads)
    rope_bwd_fp8_v6_kernel(const uint8_t *__restrict__ dq, const uint8_t *__restrict__ dk,
                           const uint8_t *__restrict__ dv, const float *__restrict__ scale_inv_ptr,
                           const float *__restrict__ q_freqs, const float *__restrict__ k_freqs,
                           __nv_bfloat16 *__restrict__ dqkv, RopeShape sh) {
  extern __shared__ __align__(128) uint8_t token_row[];  // dq row, dk row, dv row
  __shared__ __align__(16) float smem[4 * kDim];
  __shared__ __align__(8) uint64_t row_ready;
  const RowOffsets off = token_offsets(blockIdx.x, blockIdx.y, sh.s, sh.b, sh.h, sh.q_split,
                                       sh.k_split, sh.v_split, sh.sbhd);
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  const int q_bytes = sh.h * sh.q_split;
  const int k_bytes = sh.h * sh.k_split;
  if (threadIdx.x == 0) {
    const int v_bytes = sh.h * sh.v_split;
    start_barrier(&row_ready, q_bytes + k_bytes + v_bytes);
    copy_to_shared(token_row, dq + off.q, q_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes, dk + off.k, k_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes + k_bytes, dv + off.v, v_bytes, &row_ready);
  }
  const float scale_inv = *scale_inv_ptr;
  const uint64_t scale_inv2 = pk2(scale_inv, scale_inv);
  load_cos_sin_v6<kSameFreqs>(q_freqs, k_freqs, blockIdx.x, smem);
  const int slot = (threadIdx.x % kSlots) * kVec;
  SlotTable2 table = load_slot_table2(smem, slot);
  int table_kind = 0;
  wait_barrier(&row_ready);
  const int heads = sh.h * (kQh + 2);
  for (int hs = threadIdx.x / kSlots; hs < heads; hs += kThreads / kSlots) {
    const HeadTask task = head_task_pow2<kQh>(hs, sh.h, total_d, sh.q_split, sh.k_split);
    __nv_bfloat16 *dst = dqkv + off.src + task.src;
    if (task.kind == 2) {
      const uint4 packed =
          *reinterpret_cast<const uint4 *>(token_row + q_bytes + k_bytes + task.dst + 2 * slot);
      store_vec(dst + 2 * slot, dequantize_fp8x8_x2<kFormat>(make_uint2(packed.x, packed.y), scale_inv2));
      store_vec(dst + 2 * slot + kVec,
                dequantize_fp8x8_x2<kFormat>(make_uint2(packed.z, packed.w), scale_inv2));
      continue;
    }
    if (!kSameFreqs && task.kind != table_kind) {
      table = load_slot_table2(smem + 2 * kDim * task.kind, slot);
      table_kind = task.kind;
    }
    const uint8_t *grad = token_row + (task.kind == 0 ? 0 : q_bytes) + task.dst + slot;
    const Bf16x8 src_lo =
        dequantize_fp8x8_x2<kFormat>(*reinterpret_cast<const uint2 *>(grad), scale_inv2);
    const Bf16x8 src_hi =
        dequantize_fp8x8_x2<kFormat>(*reinterpret_cast<const uint2 *>(grad + kHalf), scale_inv2);
    uint32_t out_lo[kVec / 2];
    uint32_t out_hi[kVec / 2];
    rope_backward_x2<kFmaFirst>(src_lo, src_hi, table, out_lo, out_hi);
    *reinterpret_cast<uint4 *>(dst + slot) = make_uint4(out_lo[0], out_lo[1], out_lo[2], out_lo[3]);
    *reinterpret_cast<uint4 *>(dst + slot + kHalf) =
        make_uint4(out_hi[0], out_hi[1], out_hi[2], out_hi[3]);
  }
  end_barrier(&row_ready);
}

// RoPE v8: two tokens in one block of 256 threads (the v6 math). Each half block does one token.
// One TMA barrier covers both rows. Blocks of 256 threads fill the 32 warps of an SM with 4
// blocks, and 8 rows are in flight for each SM.
constexpr int kTokensPerBlock = 2;
constexpr int kThreads8 = kThreads * kTokensPerBlock;

template <int kQh, bool kSameFreqs>
__global__ void __launch_bounds__(kThreads8)
    rope_fwd_fp8_v8_kernel(const __nv_bfloat16 *__restrict__ qkv, const float *__restrict__ q_freqs,
                           const float *__restrict__ k_freqs, uint8_t *__restrict__ q_out,
                           uint8_t *__restrict__ k_out, uint8_t *__restrict__ v_out,
                           const float *__restrict__ scale_ptr, float *__restrict__ amax_ptr,
                           float *__restrict__ scale_inv_ptr, RopeShape sh, int row_stride,
                           int tokens) {
  extern __shared__ __align__(128) uint8_t rows[];
  __shared__ __align__(16) float smem[kTokensPerBlock][4 * kDim];
  __shared__ __align__(8) uint64_t rows_ready;
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  const uint32_t row_bytes = sh.h * total_d * sizeof(__nv_bfloat16);
  const int half = threadIdx.x / kThreads;  // token of this half block
  const int tid = threadIdx.x % kThreads;
  const int t = blockIdx.x * kTokensPerBlock + half;
  const bool valid = t < tokens;
  const int2 pos = token_position(valid ? t : 0, sh);
  const RowOffsets off = token_offsets(pos.x, pos.y, sh.s, sh.b, sh.h, sh.q_split, sh.k_split,
                                       sh.v_split, sh.sbhd);
  if (threadIdx.x == 0) {
    const int count = min(kTokensPerBlock, tokens - static_cast<int>(blockIdx.x) * kTokensPerBlock);
    start_barrier(&rows_ready, row_bytes * count);
    for (int i = 0; i < count; ++i) {
      const int2 p = token_position(blockIdx.x * kTokensPerBlock + i, sh);
      const RowOffsets o = token_offsets(p.x, p.y, sh.s, sh.b, sh.h, sh.q_split, sh.k_split,
                                         sh.v_split, sh.sbhd);
      copy_to_shared(rows + i * row_stride, qkv + o.src, row_bytes, &rows_ready);
    }
  }
  const float scale = *scale_ptr;
  const int amax_seen = __ldcg(reinterpret_cast<const int *>(amax_ptr));
  if (valid) {
    sincosf(q_freqs[pos.x * kDim + tid], &smem[half][kDim + tid], &smem[half][tid]);
    if constexpr (!kSameFreqs) {
      sincosf(k_freqs[pos.x * kDim + tid], &smem[half][3 * kDim + tid], &smem[half][2 * kDim + tid]);
    }
  }
  __syncthreads();
  const int slot = (tid % kSlots) * kVec;
  SlotTable2 table = load_slot_table2(smem[half], slot);
  int table_kind = 0;
  const uint64_t scale2 = pk2(scale, scale);
  wait_barrier(&rows_ready);
  __nv_bfloat162 amax2 = __floats2bfloat162_rn(0.0f, 0.0f);
  if (valid) {
    const __nv_bfloat16 *row = reinterpret_cast<const __nv_bfloat16 *>(rows + half * row_stride);
    const int heads = sh.h * (kQh + 2);
    for (int hs = tid / kSlots; hs < heads; hs += kThreads / kSlots) {
      const HeadTask task = head_task_pow2<kQh>(hs, sh.h, total_d, sh.q_split, sh.k_split);
      if (task.kind == 2) {
        const Bf16x8 first = load_vec(row + task.src + 2 * slot);
        const Bf16x8 second = load_vec(row + task.src + 2 * slot + kVec);
        const uint2 a = cast_fp8x8_x2(reinterpret_cast<const uint32_t *>(&first), scale2, amax2);
        const uint2 b = cast_fp8x8_x2(reinterpret_cast<const uint32_t *>(&second), scale2, amax2);
        *reinterpret_cast<uint4 *>(v_out + off.v + task.dst + 2 * slot) =
            make_uint4(a.x, a.y, b.x, b.y);
        continue;
      }
      if (!kSameFreqs && task.kind != table_kind) {
        table = load_slot_table2(smem[half] + 2 * kDim * task.kind, slot);
        table_kind = task.kind;
      }
      const Bf16x8 src_lo = load_vec(row + task.src + slot);
      const Bf16x8 src_hi = load_vec(row + task.src + slot + kHalf);
      uint32_t out_lo[kVec / 2];
      uint32_t out_hi[kVec / 2];
      rope_forward_x2<true>(src_lo, src_hi, table, out_lo, out_hi);
      uint8_t *dst = (task.kind == 0 ? q_out + off.q : k_out + off.k) + task.dst + slot;
      *reinterpret_cast<uint2 *>(dst) = cast_fp8x8_x2(out_lo, scale2, amax2);
      *reinterpret_cast<uint2 *>(dst + kHalf) = cast_fp8x8_x2(out_hi, scale2, amax2);
    }
  }
  warp_amax_update(fmaxf(__low2float(amax2), __high2float(amax2)), amax_seen, amax_ptr);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *scale_inv_ptr = __frcp_rn(scale);
  }
  end_barrier(&rows_ready);
}

template <__nv_fp8_interpretation_t kFormat, int kQh, bool kSameFreqs>
__global__ void __launch_bounds__(kThreads8)
    rope_bwd_fp8_v8_kernel(const uint8_t *__restrict__ dq, const uint8_t *__restrict__ dk,
                           const uint8_t *__restrict__ dv, const float *__restrict__ scale_inv_ptr,
                           const float *__restrict__ q_freqs, const float *__restrict__ k_freqs,
                           __nv_bfloat16 *__restrict__ dqkv, RopeShape sh, int row_stride,
                           int tokens) {
  extern __shared__ __align__(128) uint8_t rows[];  // dq row, dk row, dv row of each token
  __shared__ __align__(16) float smem[kTokensPerBlock][4 * kDim];
  __shared__ __align__(8) uint64_t rows_ready;
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  const int q_bytes = sh.h * sh.q_split;
  const int k_bytes = sh.h * sh.k_split;
  const int v_bytes = sh.h * sh.v_split;
  const int half = threadIdx.x / kThreads;
  const int tid = threadIdx.x % kThreads;
  const int t = blockIdx.x * kTokensPerBlock + half;
  const bool valid = t < tokens;
  const int2 pos = token_position(valid ? t : 0, sh);
  const RowOffsets off = token_offsets(pos.x, pos.y, sh.s, sh.b, sh.h, sh.q_split, sh.k_split,
                                       sh.v_split, sh.sbhd);
  if (threadIdx.x == 0) {
    const int count = min(kTokensPerBlock, tokens - static_cast<int>(blockIdx.x) * kTokensPerBlock);
    start_barrier(&rows_ready, (q_bytes + k_bytes + v_bytes) * count);
    for (int i = 0; i < count; ++i) {
      const int2 p = token_position(blockIdx.x * kTokensPerBlock + i, sh);
      const RowOffsets o = token_offsets(p.x, p.y, sh.s, sh.b, sh.h, sh.q_split, sh.k_split,
                                         sh.v_split, sh.sbhd);
      uint8_t *buf = rows + i * row_stride;
      copy_to_shared(buf, dq + o.q, q_bytes, &rows_ready);
      copy_to_shared(buf + q_bytes, dk + o.k, k_bytes, &rows_ready);
      copy_to_shared(buf + q_bytes + k_bytes, dv + o.v, v_bytes, &rows_ready);
    }
  }
  const float scale_inv = *scale_inv_ptr;
  const uint64_t scale_inv2 = pk2(scale_inv, scale_inv);
  if (valid) {
    sincosf(q_freqs[pos.x * kDim + tid], &smem[half][kDim + tid], &smem[half][tid]);
    if constexpr (!kSameFreqs) {
      sincosf(k_freqs[pos.x * kDim + tid], &smem[half][3 * kDim + tid], &smem[half][2 * kDim + tid]);
    }
  }
  __syncthreads();
  const int slot = (tid % kSlots) * kVec;
  SlotTable2 table = load_slot_table2(smem[half], slot);
  int table_kind = 0;
  wait_barrier(&rows_ready);
  if (valid) {
    const uint8_t *buf = rows + half * row_stride;
    const int heads = sh.h * (kQh + 2);
    for (int hs = tid / kSlots; hs < heads; hs += kThreads / kSlots) {
      const HeadTask task = head_task_pow2<kQh>(hs, sh.h, total_d, sh.q_split, sh.k_split);
      __nv_bfloat16 *dst = dqkv + off.src + task.src;
      if (task.kind == 2) {
        const uint4 packed =
            *reinterpret_cast<const uint4 *>(buf + q_bytes + k_bytes + task.dst + 2 * slot);
        store_vec(dst + 2 * slot,
                  dequantize_fp8x8_x2<kFormat>(make_uint2(packed.x, packed.y), scale_inv2));
        store_vec(dst + 2 * slot + kVec,
                  dequantize_fp8x8_x2<kFormat>(make_uint2(packed.z, packed.w), scale_inv2));
        continue;
      }
      if (!kSameFreqs && task.kind != table_kind) {
        table = load_slot_table2(smem[half] + 2 * kDim * task.kind, slot);
        table_kind = task.kind;
      }
      const uint8_t *grad = buf + (task.kind == 0 ? 0 : q_bytes) + task.dst + slot;
      const Bf16x8 src_lo =
          dequantize_fp8x8_x2<kFormat>(*reinterpret_cast<const uint2 *>(grad), scale_inv2);
      const Bf16x8 src_hi =
          dequantize_fp8x8_x2<kFormat>(*reinterpret_cast<const uint2 *>(grad + kHalf), scale_inv2);
      uint32_t out_lo[kVec / 2];
      uint32_t out_hi[kVec / 2];
      rope_backward_x2<true>(src_lo, src_hi, table, out_lo, out_hi);
      *reinterpret_cast<uint4 *>(dst + slot) = make_uint4(out_lo[0], out_lo[1], out_lo[2], out_lo[3]);
      *reinterpret_cast<uint4 *>(dst + slot + kHalf) =
          make_uint4(out_hi[0], out_hi[1], out_hi[2], out_hi[3]);
    }
  }
  end_barrier(&rows_ready);
}

// ---------------------------------------------------------------------------------------------
// Dropout (mask math copied from TE dropout.cu)
// ---------------------------------------------------------------------------------------------

constexpr int kDropThreads = 256;
constexpr int kChunk = 16;          // elements of one RNG chunk
constexpr int kChunksPerThread = 2;  // chunks of one thread (block-strided)

/* Byte-wise less-than comparison. The result is in the most-significant bit of each byte. */
__device__ __forceinline__ uint32_t bytewise_less_than(uint32_t a, uint32_t b) {
  uint32_t result = (a | 0x80808080) - (b & 0x7F7F7F7F);
  asm("lop3.b32 %0, %1, %2, %3, 0x4D;\n\t" : "=r"(result) : "r"(a), "r"(b), "r"(result));
  result &= 0x80808080;
  return result;
}

/* Dropout mask of 16 bits: 1 is keep, 0 is drop. Uses 4 values of the cuRAND Philox generator. */
__device__ __forceinline__ uint16_t make_16bit_mask(uint64_t chunk_idx, uint64_t rng_seed,
                                                    uint64_t rng_offset,
                                                    uint32_t bytewise_drop_prob) {
  curandStatePhilox4_32_10_t state;
  curand_init(rng_seed, chunk_idx, rng_offset, &state);
  const uint4 rand_bits = curand4(&state);
  uint32_t result = bytewise_less_than(rand_bits.x, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.y, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.z, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.w, bytewise_drop_prob);
  result |= result >> 17;
  result = ~result;
  return result;
}

struct alignas(16) Bf16x16 {
  uint4 lo;
  uint4 hi;
};

// TE dropout per element: val = float(x); val = 0 if dropped; val *= scale; out = bf16(val).
// Bit i of the mask is element i. Two elements at a time.
__device__ __forceinline__ void apply_mask_16(Bf16x16 &data, uint32_t mask, float scale) {
  __nv_bfloat162 *pairs = reinterpret_cast<__nv_bfloat162 *>(&data);
#pragma unroll
  for (int i = 0; i < kChunk / 2; ++i) {
    const float2 values = __bfloat1622float2(pairs[i]);
    const float lo = (mask & 0x1) ? values.x : 0.0f;
    const float hi = (mask & 0x2) ? values.y : 0.0f;
    pairs[i] = __floats2bfloat162_rn(lo * scale, hi * scale);
    mask >>= 2;
  }
}

__global__ void __launch_bounds__(kDropThreads)
    dropout_fwd_bf16_kernel(const Bf16x16 *__restrict__ input, Bf16x16 *__restrict__ output,
                            uint16_t *__restrict__ mask, at::PhiloxCudaState philox,
                            long long num_chunks, uint32_t bytewise_drop_prob, float scale) {
  const auto seeds = at::cuda::philox::unpack(philox);
  const long long first = static_cast<long long>(blockIdx.x) * kDropThreads * kChunksPerThread;
  Bf16x16 data[kChunksPerThread];
#pragma unroll
  for (int c = 0; c < kChunksPerThread; ++c) {
    const long long chunk = first + c * kDropThreads + threadIdx.x;
    if (chunk < num_chunks) {
      data[c] = input[chunk];
    }
  }
#pragma unroll
  for (int c = 0; c < kChunksPerThread; ++c) {
    const long long chunk = first + c * kDropThreads + threadIdx.x;
    if (chunk < num_chunks) {
      const uint16_t local_mask = make_16bit_mask(chunk, std::get<0>(seeds), std::get<1>(seeds),
                                                  bytewise_drop_prob);
      mask[chunk] = local_mask;
      apply_mask_16(data[c], local_mask, scale);
      output[chunk] = data[c];
    }
  }
}

// Float value of each FP4 E2M1 code, from the cuda_fp4.h conversion that TE also uses.
__device__ __forceinline__ void fill_fp4_table(float *table) {
  if (threadIdx.x < 16) {
    __nv_fp4_e2m1 code;
    code.__x = static_cast<__nv_fp4_storage_t>(threadIdx.x);
    table[threadIdx.x] = static_cast<float>(code);
  }
  __syncthreads();
}

// TE NVFP4 dequantize (dequantize_fp4_kernel) of one 16-element block: bf16(fp4 * final_scale).
// Byte k holds element 2k in the low 4 bits and element 2k + 1 in the high 4 bits.
__device__ __forceinline__ void dequantize_nvfp4_16(uint2 packed, float final_scale,
                                                    const float *table, Bf16x16 &out) {
  __nv_bfloat162 *out2 = reinterpret_cast<__nv_bfloat162 *>(&out);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const uint32_t word = k < 4 ? packed.x : packed.y;
    const uint32_t byte = (word >> (8 * (k % 4))) & 0xFF;
    out2[k] = __floats2bfloat162_rn(table[byte & 0xF] * final_scale,
                                    table[byte >> 4] * final_scale);
  }
}

// Index of a scale in the GEMM swizzled layout (TE mxfp8/swizzle.cuh gemm_swizzled_scale_idx).
template <typename IndexT>
__device__ __forceinline__ IndexT swizzled_scale_idx(IndexT i, IndexT j, IndexT num_tiles_x) {
  constexpr IndexT kTileX = 4;
  constexpr IndexT kTileY = 128;
  IndexT idx = ((i / kTileY) * num_tiles_x + j / kTileX) * (kTileX * kTileY);
  const IndexT in_y = i % kTileY;
  idx += (in_y % 32) * 16 + (in_y / 32) * 4 + j % kTileX;
  return idx;
}

// Shape of the NVFP4 data and of its scales, in chunks of 16 elements.
template <typename IndexT>
struct Nvfp4Layout {
  IndexT num_chunks;
  IndexT blocks_per_row;
  IndexT scale_stride;
  IndexT num_tiles_x;
  bool swizzled;
};

template <typename IndexT>
__global__ void __launch_bounds__(kDropThreads)
    dropout_fwd_nvfp4_kernel(const uint2 *__restrict__ data, const uint8_t *__restrict__ scales,
                             const float *__restrict__ amax_ptr, Bf16x16 *__restrict__ output,
                             uint16_t *__restrict__ mask, at::PhiloxCudaState philox,
                             Nvfp4Layout<IndexT> layout, uint32_t bytewise_drop_prob,
                             float scale) {
  constexpr float factor_inv = 1.0f / (6.0f * 448.0f);
  __shared__ float fp4_table[16];
  fill_fp4_table(fp4_table);
  const auto seeds = at::cuda::philox::unpack(philox);
  const float amax = *amax_ptr;
  const IndexT first = static_cast<IndexT>(blockIdx.x) * kDropThreads * kChunksPerThread;
  uint2 packed[kChunksPerThread];
  uint8_t block_scale[kChunksPerThread];
#pragma unroll
  for (int c = 0; c < kChunksPerThread; ++c) {
    const IndexT chunk = first + c * kDropThreads + threadIdx.x;
    if (chunk < layout.num_chunks) {
      const IndexT y = chunk / layout.blocks_per_row;
      const IndexT x = chunk - y * layout.blocks_per_row;
      const IndexT scale_idx = layout.swizzled ? swizzled_scale_idx(y, x, layout.num_tiles_x)
                                               : x + y * layout.scale_stride;
      packed[c] = data[chunk];
      block_scale[c] = scales[scale_idx];
    }
  }
#pragma unroll
  for (int c = 0; c < kChunksPerThread; ++c) {
    const IndexT chunk = first + c * kDropThreads + threadIdx.x;
    if (chunk < layout.num_chunks) {
      __nv_fp8_e4m3 scale_fp8;
      scale_fp8.__x = block_scale[c];
      const float final_scale = static_cast<float>(scale_fp8) * amax * factor_inv;
      Bf16x16 values;
      dequantize_nvfp4_16(packed[c], final_scale, fp4_table, values);
      const uint16_t local_mask = make_16bit_mask(chunk, std::get<0>(seeds), std::get<1>(seeds),
                                                  bytewise_drop_prob);
      mask[chunk] = local_mask;
      apply_mask_16(values, local_mask, scale);
      output[chunk] = values;
    }
  }
}

// TE apply_dropout_mask: same per-element math, 16 elements per chunk.
__global__ void __launch_bounds__(kDropThreads)
    dropout_bwd_bf16_kernel(const Bf16x16 *__restrict__ grad, const uint16_t *__restrict__ mask,
                            Bf16x16 *__restrict__ grad_input, long long num_chunks, float scale) {
  const long long first = static_cast<long long>(blockIdx.x) * kDropThreads * kChunksPerThread;
  Bf16x16 data[kChunksPerThread];
  uint16_t local_mask[kChunksPerThread];
#pragma unroll
  for (int c = 0; c < kChunksPerThread; ++c) {
    const long long chunk = first + c * kDropThreads + threadIdx.x;
    if (chunk < num_chunks) {
      data[c] = grad[chunk];
      local_mask[c] = mask[chunk];
    }
  }
#pragma unroll
  for (int c = 0; c < kChunksPerThread; ++c) {
    const long long chunk = first + c * kDropThreads + threadIdx.x;
    if (chunk < num_chunks) {
      apply_mask_16(data[c], local_mask[c], scale);
      grad_input[chunk] = data[c];
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Tuned dropout variants
// ---------------------------------------------------------------------------------------------

// Philox4x32-10 with the cuRAND constants. philox_chunk gives the bits of
// curand_init(seed, chunk, offset) + curand4(): the counter is {offset / 4, chunk}.
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

__device__ __forceinline__ uint4 philox_chunk(uint64_t seed, uint64_t chunk, uint64_t offset) {
  // curand_init: ctr = 0, skipahead_sequence(chunk) adds chunk to {z, w}, skipahead(offset)
  // adds offset / 4 to {x, y}. Both start from zero, thus no carry.
  const uint64_t n = offset >> 2;
  uint4 c = make_uint4(static_cast<uint32_t>(n), static_cast<uint32_t>(n >> 32),
                       static_cast<uint32_t>(chunk), static_cast<uint32_t>(chunk >> 32));
  uint32_t kx = static_cast<uint32_t>(seed);
  uint32_t ky = static_cast<uint32_t>(seed >> 32);
#pragma unroll
  for (int r = 0; r < 9; ++r) {
    c = philox_round(c, kx, ky);
    kx += kPhiloxW0;
    ky += kPhiloxW1;
  }
  return philox_round(c, kx, ky);
}

// Philox of kChunks consecutive chunks at once. The key schedule is shared, thus the key adds
// happen once for each round, and the kChunks chains interleave.
template <int kChunks>
__device__ __forceinline__ void philox_chunks(uint64_t seed, uint64_t first_chunk, uint64_t offset,
                                              uint4 *out) {
  const uint64_t n = offset >> 2;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const uint64_t chunk = first_chunk + c;
    out[c] = make_uint4(static_cast<uint32_t>(n), static_cast<uint32_t>(n >> 32),
                        static_cast<uint32_t>(chunk), static_cast<uint32_t>(chunk >> 32));
  }
  uint32_t kx = static_cast<uint32_t>(seed);
  uint32_t ky = static_cast<uint32_t>(seed >> 32);
#pragma unroll
  for (int r = 0; r < 10; ++r) {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      out[c] = philox_round(out[c], kx, ky);
    }
    kx += kPhiloxW0;
    ky += kPhiloxW1;
  }
}

__device__ __forceinline__ uint16_t mask_from_bits(const uint4 rand_bits, uint32_t bytewise_drop_prob) {
  uint32_t result = bytewise_less_than(rand_bits.x, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.y, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.z, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.w, bytewise_drop_prob);
  result |= result >> 17;
  result = ~result;
  return result;
}

template <int kChunks>
__device__ __forceinline__ void make_masks_batched(uint64_t first_chunk, uint64_t rng_seed,
                                                   uint64_t rng_offset, uint32_t bytewise_drop_prob,
                                                   uint16_t *masks) {
  uint4 bits[kChunks];
  philox_chunks<kChunks>(rng_seed, first_chunk, rng_offset, bits);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    masks[c] = mask_from_bits(bits[c], bytewise_drop_prob);
  }
}

__device__ __forceinline__ uint16_t make_16bit_mask_fast(uint64_t chunk_idx, uint64_t rng_seed,
                                                         uint64_t rng_offset,
                                                         uint32_t bytewise_drop_prob) {
  const uint4 rand_bits = philox_chunk(rng_seed, chunk_idx, rng_offset);
  uint32_t result = bytewise_less_than(rand_bits.x, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.y, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.z, bytewise_drop_prob);
  result = (result >> 2) | bytewise_less_than(rand_bits.w, bytewise_drop_prob);
  result |= result >> 17;
  result = ~result;
  return result;
}

// 256-bit global loads and stores (sm_100 or later).
__device__ __forceinline__ void load_256(const void *ptr, uint4 &a, uint4 &b) {
  asm("ld.global.nc.L1::no_allocate.L2::256B.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w), "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
      : "l"(ptr));
}

__device__ __forceinline__ void store_256(void *ptr, const uint4 &a, const uint4 &b) {
  asm volatile("st.global.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(ptr), "r"(a.x),
               "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w)
               : "memory");
}

// NVFP4 dequantize of 16 elements with the hardware conversion (cvt e2m1x2 -> f16x2). The
// half values are exact, thus bf16(float(half) * final_scale) equals the table version.
__device__ __forceinline__ void dequantize_nvfp4_16_cvt(uint2 packed, float final_scale,
                                                        Bf16x16 &out) {
  __nv_bfloat162 *out2 = reinterpret_cast<__nv_bfloat162 *>(&out);
  const __nv_fp4x2_storage_t *bytes = reinterpret_cast<const __nv_fp4x2_storage_t *>(&packed);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const __half2 values(__nv_cvt_fp4x2_to_halfraw2(bytes[k], __NV_E2M1));
    const float2 f = __half22float2(values);
    out2[k] = __floats2bfloat162_rn(f.x * final_scale, f.y * final_scale);
  }
}

// Store the 16-bit masks of kChunks consecutive chunks with one aligned store.
template <int kChunks>
__device__ __forceinline__ void store_masks(uint16_t *dst, const uint16_t *masks) {
  if constexpr (kChunks == 8) {
    *reinterpret_cast<uint4 *>(dst) =
        make_uint4(masks[0] | (static_cast<uint32_t>(masks[1]) << 16),
                   masks[2] | (static_cast<uint32_t>(masks[3]) << 16),
                   masks[4] | (static_cast<uint32_t>(masks[5]) << 16),
                   masks[6] | (static_cast<uint32_t>(masks[7]) << 16));
  } else if constexpr (kChunks == 4) {
    *reinterpret_cast<uint2 *>(dst) =
        make_uint2(masks[0] | (static_cast<uint32_t>(masks[1]) << 16),
                   masks[2] | (static_cast<uint32_t>(masks[3]) << 16));
  } else if constexpr (kChunks == 2) {
    *reinterpret_cast<uint32_t *>(dst) = masks[0] | (static_cast<uint32_t>(masks[1]) << 16);
  } else {
    dst[0] = masks[0];
  }
}

// Each thread does kChunks consecutive chunks (16 elements each). The host checks that the
// chunk count is a multiple of kChunks and that the chunks of one thread stay in one row and
// in one 4-chunk scale group (swizzled layout).
template <int kChunks, bool kCvt, typename IndexT, bool kBatch = false>
__global__ void __launch_bounds__(kDropThreads)
    dropout_fwd_nvfp4_v2_kernel(const uint2 *__restrict__ data, const uint8_t *__restrict__ scales,
                                const float *__restrict__ amax_ptr, Bf16x16 *__restrict__ output,
                                uint16_t *__restrict__ mask, at::PhiloxCudaState philox,
                                Nvfp4Layout<IndexT> layout, uint32_t bytewise_drop_prob,
                                float scale) {
  constexpr float factor_inv = 1.0f / (6.0f * 448.0f);
  __shared__ float fp4_table[16];
  if constexpr (!kCvt) {
    fill_fp4_table(fp4_table);
  }
  const IndexT first = (static_cast<IndexT>(blockIdx.x) * kDropThreads + threadIdx.x) * kChunks;
  if (first >= layout.num_chunks) {
    return;
  }
  uint2 packed[kChunks];
  if constexpr (kChunks >= 4) {
#pragma unroll
    for (int g = 0; g < kChunks / 4; ++g) {
      uint4 a, b;
      load_256(data + first + 4 * g, a, b);
      packed[4 * g] = make_uint2(a.x, a.y);
      packed[4 * g + 1] = make_uint2(a.z, a.w);
      packed[4 * g + 2] = make_uint2(b.x, b.y);
      packed[4 * g + 3] = make_uint2(b.z, b.w);
    }
  } else {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      packed[c] = __ldg(data + first + c);
    }
  }
  const IndexT y = first / layout.blocks_per_row;
  const IndexT x = first - y * layout.blocks_per_row;
  const IndexT scale_idx =
      layout.swizzled ? swizzled_scale_idx(y, x, layout.num_tiles_x) : x + y * layout.scale_stride;
  // Scale bytes of the chunks: 4 consecutive chunks share one 4-byte group in both layouts.
  uint32_t scale_bits[(kChunks + 3) / 4];
  if constexpr (kChunks >= 4) {
#pragma unroll
    for (int g = 0; g < kChunks / 4; ++g) {
      const IndexT idx = g == 0 ? scale_idx
                         : (layout.swizzled ? swizzled_scale_idx(y, x + 4 * g, layout.num_tiles_x)
                                            : scale_idx + 4 * g);
      scale_bits[g] = __ldg(reinterpret_cast<const uint32_t *>(scales + idx));
    }
  } else if constexpr (kChunks == 2) {
    scale_bits[0] = __ldg(reinterpret_cast<const uint16_t *>(scales + scale_idx));
  } else {
    scale_bits[0] = __ldg(scales + scale_idx);
  }
  const auto seeds = at::cuda::philox::unpack(philox);
  const float amax = *amax_ptr;
  uint16_t masks[kChunks];
  if constexpr (kBatch) {
    make_masks_batched<kChunks>(first, std::get<0>(seeds), std::get<1>(seeds), bytewise_drop_prob,
                                masks);
  } else {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      masks[c] = make_16bit_mask_fast(first + c, std::get<0>(seeds), std::get<1>(seeds),
                                      bytewise_drop_prob);
    }
  }
  store_masks<kChunks>(mask + first, masks);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    __nv_fp8_e4m3 scale_fp8;
    scale_fp8.__x = static_cast<uint8_t>(scale_bits[c / 4] >> (8 * (c % 4)));
    const float final_scale = static_cast<float>(scale_fp8) * amax * factor_inv;
    Bf16x16 values;
    if constexpr (kCvt) {
      dequantize_nvfp4_16_cvt(packed[c], final_scale, values);
    } else {
      dequantize_nvfp4_16(packed[c], final_scale, fp4_table, values);
    }
    apply_mask_16(values, masks[c], scale);
    if constexpr (kChunks >= 4) {
      store_256(output + first + c, values.lo, values.hi);
    } else {
      output[first + c] = values;
    }
  }
}

// bf16 dropout forward, kChunks consecutive chunks of one thread.
template <int kChunks, int kMinBlocks, bool kBatch = false>
__global__ void __launch_bounds__(kDropThreads, kMinBlocks)
    dropout_fwd_bf16_v2_kernel(const Bf16x16 *__restrict__ input, Bf16x16 *__restrict__ output,
                               uint16_t *__restrict__ mask, at::PhiloxCudaState philox,
                               long long num_chunks, uint32_t bytewise_drop_prob, float scale) {
  const long long first =
      (static_cast<long long>(blockIdx.x) * kDropThreads + threadIdx.x) * kChunks;
  if (first >= num_chunks) {
    return;
  }
  Bf16x16 data[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    load_256(input + first + c, data[c].lo, data[c].hi);
  }
  const auto seeds = at::cuda::philox::unpack(philox);
  uint16_t masks[kChunks];
  if constexpr (kBatch) {
    make_masks_batched<kChunks>(first, std::get<0>(seeds), std::get<1>(seeds), bytewise_drop_prob,
                                masks);
  } else {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      masks[c] = make_16bit_mask_fast(first + c, std::get<0>(seeds), std::get<1>(seeds),
                                      bytewise_drop_prob);
    }
  }
  store_masks<kChunks>(mask + first, masks);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    apply_mask_16(data[c], masks[c], scale);
    store_256(output + first + c, data[c].lo, data[c].hi);
  }
}

// Dropout backward (mask), kChunks consecutive chunks of one thread.
template <int kChunks>
__global__ void __launch_bounds__(kDropThreads)
    dropout_bwd_bf16_v2_kernel(const Bf16x16 *__restrict__ grad, const uint16_t *__restrict__ mask,
                               Bf16x16 *__restrict__ grad_input, long long num_chunks,
                               float scale) {
  const long long first =
      (static_cast<long long>(blockIdx.x) * kDropThreads + threadIdx.x) * kChunks;
  if (first >= num_chunks) {
    return;
  }
  Bf16x16 data[kChunks];
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    load_256(grad + first + c, data[c].lo, data[c].hi);
  }
  uint32_t masks[kChunks];
  if constexpr (kChunks == 8) {
    const uint4 m = __ldg(reinterpret_cast<const uint4 *>(mask + first));
    const uint32_t words[4] = {m.x, m.y, m.z, m.w};
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      masks[2 * c] = words[c] & 0xFFFF;
      masks[2 * c + 1] = words[c] >> 16;
    }
  } else if constexpr (kChunks == 4) {
    const uint2 m = __ldg(reinterpret_cast<const uint2 *>(mask + first));
    masks[0] = m.x & 0xFFFF;
    masks[1] = m.x >> 16;
    masks[2] = m.y & 0xFFFF;
    masks[3] = m.y >> 16;
  } else {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      masks[c] = __ldg(mask + first + c);
    }
  }
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    apply_mask_16(data[c], masks[c], scale);
    store_256(grad_input + first + c, data[c].lo, data[c].hi);
  }
}

// ---------------------------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------------------------

// Default variants of the Python-visible functions (the fastest measured variants).
constexpr int64_t kBestRopeFwd = 6;
constexpr int64_t kBestRopeBwd = 6;
constexpr int64_t kBestDropoutNvfp4 = 44;
constexpr int64_t kBestDropoutBf16 = 22;
constexpr int64_t kBestDropoutBwd = 22;

void check_rope_inputs(const at::Tensor &freqs, const std::vector<int64_t> &split) {
  TORCH_CHECK(freqs.scalar_type() == at::kFloat && freqs.is_contiguous(), "freqs: fp32 contiguous");
  TORCH_CHECK(freqs.dim() == 4 && freqs.size(1) == 1 && freqs.size(2) == 1 &&
                  freqs.size(3) == kDim,
              "freqs: [s, 1, 1, 128]");
  TORCH_CHECK(split.size() == 3 && split[1] == kDim && split[2] == kDim, "k and v split: 128");
  TORCH_CHECK(split[0] % kDim == 0, "q split must be a multiple of 128");
}

void check_byte_tensor(const at::Tensor &t, at::IntArrayRef size, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kByte && t.is_contiguous(), name,
              ": contiguous uint8 CUDA tensor");
  TORCH_CHECK(t.sizes() == size, name, ": wrong shape");
}

void check_float_scalar(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.numel() == 1, name,
              ": one fp32 CUDA value");
}

// Iterations of one thread in the v5 kernels: 8 slots for each head.
int rope_tasks_per_thread(const RopeShape &sh) {
  const int thread_tasks = kSlots * sh.h * (sh.qh + 2);
  return (thread_tasks + kThreads - 1) / kThreads;
}

// Launch shape of the pipelined kernels: a persistent grid and the dynamic shared memory of
// kStages rows (128-byte stride) and kStages tables.
struct PipeLaunch {
  int grid;
  int smem;
  int row_stride;
};

PipeLaunch pipe_launch(int stages, int row_bytes, int tokens, const void *kernel) {
  const int row_stride = (row_bytes + 127) / 128 * 128;
  const int smem = stages * (row_stride + kTableFloats * static_cast<int>(sizeof(float)));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, smem));
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = std::max(1, std::min(tokens, sms * std::max(per_sm, 1)));
  return {grid, smem, row_stride};
}

// v5 variants: 5, 6, 7 give __launch_bounds__ minimum blocks 4, 6, 8.
template <int kTasks>
void launch_rope_fwd_v5(int64_t variant, const dim3 grid, cudaStream_t stream,
                        const __nv_bfloat16 *qkv, const float *q_freqs, const float *k_freqs,
                        uint8_t *q_out, uint8_t *k_out, uint8_t *v_out, const float *scale,
                        float *amax, float *scale_inv, const RopeShape &sh) {
  if (variant == 6) {
    rope_fwd_fp8_v5_kernel<kTasks, 6><<<grid, kThreads, 0, stream>>>(
        qkv, q_freqs, k_freqs, q_out, k_out, v_out, scale, amax, scale_inv, sh);
  } else if (variant == 7) {
    rope_fwd_fp8_v5_kernel<kTasks, 8><<<grid, kThreads, 0, stream>>>(
        qkv, q_freqs, k_freqs, q_out, k_out, v_out, scale, amax, scale_inv, sh);
  } else {
    rope_fwd_fp8_v5_kernel<kTasks, 4><<<grid, kThreads, 0, stream>>>(
        qkv, q_freqs, k_freqs, q_out, k_out, v_out, scale, amax, scale_inv, sh);
  }
}

// Fused RoPE and FP8 cast. Writes the E4M3 bytes of q, k, v into the given uint8 tensors,
// does an atomic max into amax and writes scale_inv = 1 / scale.
void rope_forward_fp8(const at::Tensor &qkv, const at::Tensor &q_freqs, const at::Tensor &k_freqs,
                      std::vector<int64_t> split, bool sbhd, const at::Tensor &scale,
                      const at::Tensor &amax, const at::Tensor &scale_inv, const at::Tensor &q_out,
                      const at::Tensor &k_out, const at::Tensor &v_out, int64_t variant) {
  check_rope_inputs(q_freqs, split);
  check_rope_inputs(k_freqs, split);
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.is_contiguous() && qkv.dim() == 4,
              "qkv: 4D bf16 contiguous");
  TORCH_CHECK(qkv.size(3) == split[0] + split[1] + split[2], "qkv: last dim must match split");
  const int s = sbhd ? qkv.size(0) : qkv.size(1);
  const int b = sbhd ? qkv.size(1) : qkv.size(0);
  const int h = qkv.size(2);
  TORCH_CHECK(q_freqs.size(0) >= s && k_freqs.size(0) >= s, "freqs: too few positions");
  auto q_size = qkv.sizes().vec();
  q_size[2] = q_size[2] * split[0] / split[1];
  q_size[3] = split[1];
  auto kv_size = qkv.sizes().vec();
  kv_size[3] = split[1];
  check_byte_tensor(q_out, q_size, "q_out");
  check_byte_tensor(k_out, kv_size, "k_out");
  check_byte_tensor(v_out, kv_size, "v_out");
  check_float_scalar(scale, "scale");
  check_float_scalar(amax, "amax");
  check_float_scalar(scale_inv, "scale_inv");
  const int row_bytes = h * qkv.size(3) * static_cast<int>(sizeof(__nv_bfloat16));
  TORCH_CHECK(row_bytes <= kMaxRowBytes, "qkv: token row larger than the shared memory buffer");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0, "qkv: 16-byte aligned");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const RopeShape sh{s, b, h, static_cast<int>(split[0]), static_cast<int>(split[1]),
                     static_cast<int>(split[2]), static_cast<int>(split[0] / kDim), sbhd};
  const auto *qkv_ptr = reinterpret_cast<const __nv_bfloat16 *>(qkv.data_ptr());
  const dim3 grid(s, b);
  const int tasks = rope_tasks_per_thread(sh);
  if (variant == 1) {
    qkv_rope_forward_fp8_kernel<<<grid, kThreads, row_bytes, stream>>>(
        qkv_ptr, q_freqs.data_ptr<float>(), k_freqs.data_ptr<float>(), q_out.data_ptr<uint8_t>(),
        k_out.data_ptr<uint8_t>(), v_out.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        amax.data_ptr<float>(), scale_inv.data_ptr<float>(), s, b, h, split[0], split[1],
        split[2], sbhd);
  } else if (variant == 8 && sh.qh == 8) {
    const bool same = q_freqs.data_ptr() == k_freqs.data_ptr();
    const int row_stride = (row_bytes + 127) / 128 * 128;
    const int smem = kTokensPerBlock * row_stride;
    const int tokens = s * b;
    const dim3 grid8((tokens + kTokensPerBlock - 1) / kTokensPerBlock);
    auto launch = [&](auto same_tag) {
      auto kernel = rope_fwd_fp8_v8_kernel<8, decltype(same_tag)::value>;
      C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      kernel<<<grid8, kThreads8, smem, stream>>>(
          qkv_ptr, q_freqs.data_ptr<float>(), k_freqs.data_ptr<float>(), q_out.data_ptr<uint8_t>(),
          k_out.data_ptr<uint8_t>(), v_out.data_ptr<uint8_t>(), scale.data_ptr<float>(),
          amax.data_ptr<float>(), scale_inv.data_ptr<float>(), sh, row_stride, tokens);
    };
    if (same) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
  } else if ((variant == 6 || variant == 7) && sh.qh == 8) {
    const bool same = q_freqs.data_ptr() == k_freqs.data_ptr();
    auto launch = [&](auto same_tag, auto fma_tag) {
      rope_fwd_fp8_v6_kernel<8, decltype(same_tag)::value, decltype(fma_tag)::value>
          <<<grid, kThreads, row_bytes, stream>>>(
              qkv_ptr, q_freqs.data_ptr<float>(), k_freqs.data_ptr<float>(),
              q_out.data_ptr<uint8_t>(), k_out.data_ptr<uint8_t>(), v_out.data_ptr<uint8_t>(),
              scale.data_ptr<float>(), amax.data_ptr<float>(), scale_inv.data_ptr<float>(), sh);
    };
    if (same && variant == 6) {
      launch(std::true_type{}, std::true_type{});
    } else if (same) {
      launch(std::true_type{}, std::false_type{});
    } else if (variant == 6) {
      launch(std::false_type{}, std::true_type{});
    } else {
      launch(std::false_type{}, std::false_type{});
    }
  } else if (variant == 3 || variant == 4) {
    const PipeLaunch pl = pipe_launch(variant == 3 ? 2 : 3, row_bytes, s * b,
                                      variant == 3 ? (const void *)rope_fwd_fp8_pipe_kernel<2>
                                                   : (const void *)rope_fwd_fp8_pipe_kernel<3>);
    if (variant == 3) {
      rope_fwd_fp8_pipe_kernel<2><<<pl.grid, kThreads, pl.smem, stream>>>(
          qkv_ptr, q_freqs.data_ptr<float>(), k_freqs.data_ptr<float>(), q_out.data_ptr<uint8_t>(),
          k_out.data_ptr<uint8_t>(), v_out.data_ptr<uint8_t>(), scale.data_ptr<float>(),
          amax.data_ptr<float>(), scale_inv.data_ptr<float>(), sh, pl.row_stride, s * b);
    } else {
      rope_fwd_fp8_pipe_kernel<3><<<pl.grid, kThreads, pl.smem, stream>>>(
          qkv_ptr, q_freqs.data_ptr<float>(), k_freqs.data_ptr<float>(), q_out.data_ptr<uint8_t>(),
          k_out.data_ptr<uint8_t>(), v_out.data_ptr<uint8_t>(), scale.data_ptr<float>(),
          amax.data_ptr<float>(), scale_inv.data_ptr<float>(), sh, pl.row_stride, s * b);
    }
  } else if (variant == 2 || tasks > 8 || tasks < 4) {
    rope_fwd_fp8_v2_kernel<<<grid, kThreads, row_bytes, stream>>>(
        qkv_ptr, q_freqs.data_ptr<float>(), k_freqs.data_ptr<float>(), q_out.data_ptr<uint8_t>(),
        k_out.data_ptr<uint8_t>(), v_out.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        amax.data_ptr<float>(), scale_inv.data_ptr<float>(), sh);
  } else {
    auto launch = [&](auto tag) {
      launch_rope_fwd_v5<decltype(tag)::value>(
          variant, grid, stream, qkv_ptr, q_freqs.data_ptr<float>(), k_freqs.data_ptr<float>(),
          q_out.data_ptr<uint8_t>(), k_out.data_ptr<uint8_t>(), v_out.data_ptr<uint8_t>(),
          scale.data_ptr<float>(), amax.data_ptr<float>(), scale_inv.data_ptr<float>(), sh);
    };
    switch (tasks) {
      case 4: launch(std::integral_constant<int, 4>{}); break;
      case 5: launch(std::integral_constant<int, 5>{}); break;
      case 6: launch(std::integral_constant<int, 6>{}); break;
      case 7: launch(std::integral_constant<int, 7>{}); break;
      default: launch(std::integral_constant<int, 8>{}); break;
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <__nv_fp8_interpretation_t kFormat, int kTasks>
void launch_rope_bwd_v5(int64_t variant, const dim3 grid, cudaStream_t stream, const uint8_t *dq,
                        const uint8_t *dk, const uint8_t *dv, const float *scale_inv,
                        const float *q_freqs, const float *k_freqs, __nv_bfloat16 *out,
                        const RopeShape &sh) {
  if (variant == 6) {
    rope_bwd_fp8_v5_kernel<kFormat, kTasks, 6><<<grid, kThreads, 0, stream>>>(
        dq, dk, dv, scale_inv, q_freqs, k_freqs, out, sh);
  } else if (variant == 7) {
    rope_bwd_fp8_v5_kernel<kFormat, kTasks, 8><<<grid, kThreads, 0, stream>>>(
        dq, dk, dv, scale_inv, q_freqs, k_freqs, out, sh);
  } else {
    rope_bwd_fp8_v5_kernel<kFormat, kTasks, 4><<<grid, kThreads, 0, stream>>>(
        dq, dk, dv, scale_inv, q_freqs, k_freqs, out, sh);
  }
}

template <__nv_fp8_interpretation_t kFormat>
void launch_rope_bwd(int64_t variant, const dim3 grid, int row_bytes, cudaStream_t stream,
                     const uint8_t *dq, const uint8_t *dk, const uint8_t *dv,
                     const float *scale_inv, const float *q_freqs, const float *k_freqs,
                     __nv_bfloat16 *out, const RopeShape &sh) {
  const int tasks = rope_tasks_per_thread(sh);
  if (variant == 1) {
    qkv_rope_backward_fp8_kernel<kFormat><<<grid, kThreads, row_bytes, stream>>>(
        dq, dk, dv, scale_inv, q_freqs, k_freqs, out, sh.s, sh.b, sh.h, sh.q_split, sh.k_split,
        sh.v_split, sh.sbhd);
  } else if (variant == 8 && sh.qh == 8) {
    const bool same = q_freqs == k_freqs;
    const int row_stride = (row_bytes + 127) / 128 * 128;
    const int smem = kTokensPerBlock * row_stride;
    const int tokens = sh.s * sh.b;
    const dim3 grid8((tokens + kTokensPerBlock - 1) / kTokensPerBlock);
    auto launch = [&](auto same_tag) {
      auto kernel = rope_bwd_fp8_v8_kernel<kFormat, 8, decltype(same_tag)::value>;
      C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
      kernel<<<grid8, kThreads8, smem, stream>>>(dq, dk, dv, scale_inv, q_freqs, k_freqs, out, sh,
                                                 row_stride, tokens);
    };
    if (same) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
  } else if ((variant == 6 || variant == 7) && sh.qh == 8) {
    const bool same = q_freqs == k_freqs;
    auto launch = [&](auto same_tag, auto fma_tag) {
      rope_bwd_fp8_v6_kernel<kFormat, 8, decltype(same_tag)::value, decltype(fma_tag)::value>
          <<<grid, kThreads, row_bytes, stream>>>(dq, dk, dv, scale_inv, q_freqs, k_freqs, out, sh);
    };
    if (same && variant == 6) {
      launch(std::true_type{}, std::true_type{});
    } else if (same) {
      launch(std::true_type{}, std::false_type{});
    } else if (variant == 6) {
      launch(std::false_type{}, std::true_type{});
    } else {
      launch(std::false_type{}, std::false_type{});
    }
  } else if (variant == 3 || variant == 4) {
    const PipeLaunch pl =
        pipe_launch(variant == 3 ? 2 : 3, row_bytes, sh.s * sh.b,
                    variant == 3 ? (const void *)rope_bwd_fp8_pipe_kernel<kFormat, 2>
                                 : (const void *)rope_bwd_fp8_pipe_kernel<kFormat, 3>);
    if (variant == 3) {
      rope_bwd_fp8_pipe_kernel<kFormat, 2><<<pl.grid, kThreads, pl.smem, stream>>>(
          dq, dk, dv, scale_inv, q_freqs, k_freqs, out, sh, pl.row_stride, sh.s * sh.b);
    } else {
      rope_bwd_fp8_pipe_kernel<kFormat, 3><<<pl.grid, kThreads, pl.smem, stream>>>(
          dq, dk, dv, scale_inv, q_freqs, k_freqs, out, sh, pl.row_stride, sh.s * sh.b);
    }
  } else if (variant == 2 || tasks > 8 || tasks < 4) {
    rope_bwd_fp8_v2_kernel<kFormat><<<grid, kThreads, row_bytes, stream>>>(
        dq, dk, dv, scale_inv, q_freqs, k_freqs, out, sh);
  } else {
    auto launch = [&](auto tag) {
      launch_rope_bwd_v5<kFormat, decltype(tag)::value>(variant, grid, stream, dq, dk, dv,
                                                         scale_inv, q_freqs, k_freqs, out, sh);
    };
    switch (tasks) {
      case 4: launch(std::integral_constant<int, 4>{}); break;
      case 5: launch(std::integral_constant<int, 5>{}); break;
      case 6: launch(std::integral_constant<int, 6>{}); break;
      case 7: launch(std::integral_constant<int, 7>{}); break;
      default: launch(std::integral_constant<int, 8>{}); break;
    }
  }
}

// Fused FP8 dequantize (one scale_inv for dq, dk and dv) and RoPE backward.
at::Tensor rope_backward_fp8(const at::Tensor &dq, const at::Tensor &dk, const at::Tensor &dv,
                             const at::Tensor &scale_inv, const at::Tensor &q_freqs,
                             const at::Tensor &k_freqs, std::vector<int64_t> split, bool sbhd,
                             bool e5m2, int64_t variant) {
  check_rope_inputs(q_freqs, split);
  check_rope_inputs(k_freqs, split);
  check_float_scalar(scale_inv, "scale_inv");
  TORCH_CHECK(dq.dim() == 4 && dk.dim() == 4 && dv.dim() == 4, "grads: 4D");
  check_byte_tensor(dq, dq.sizes(), "dq");
  check_byte_tensor(dk, dk.sizes(), "dk");
  check_byte_tensor(dv, dv.sizes(), "dv");
  TORCH_CHECK(dq.size(3) == kDim && dk.size(3) == kDim && dv.size(3) == kDim, "grads: d = 128");
  for (const auto &t : {dk, dv}) {
    TORCH_CHECK(t.size(0) == dq.size(0) && t.size(1) == dq.size(1), "grads: s and b must match");
  }
  const int s = sbhd ? dq.size(0) : dq.size(1);
  const int b = sbhd ? dq.size(1) : dq.size(0);
  TORCH_CHECK(q_freqs.size(0) >= s && k_freqs.size(0) >= s, "freqs: too few positions");
  const int64_t total_d = split[0] + split[1] + split[2];
  const int64_t total_hd = (dq.size(2) + dk.size(2) + dv.size(2)) * dq.size(3);
  TORCH_CHECK(total_hd % total_d == 0, "grads: head counts do not match split");
  auto size = dq.sizes().vec();
  size[2] = total_hd / total_d;
  size[3] = total_d;
  TORCH_CHECK(dq.size(2) == size[2] * split[0] / kDim && dk.size(2) == size[2] &&
                  dv.size(2) == size[2],
              "grads: head counts do not match split");
  auto dqkv = at::empty(size, dq.options().dtype(at::kBFloat16));
  const dim3 grid(s, b);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto *dq_ptr = dq.data_ptr<uint8_t>();
  const auto *dk_ptr = dk.data_ptr<uint8_t>();
  const auto *dv_ptr = dv.data_ptr<uint8_t>();
  auto *out_ptr = reinterpret_cast<__nv_bfloat16 *>(dqkv.data_ptr());
  const int row_bytes = static_cast<int>((dq.size(2) + dk.size(2) + dv.size(2)) * dq.size(3));
  TORCH_CHECK(row_bytes <= kMaxRowBytes, "grads: token row larger than the shared memory buffer");
  for (const auto &t : {dq, dk, dv}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "grads: 16-byte aligned");
  }
  const RopeShape sh{s, b, static_cast<int>(size[2]), static_cast<int>(split[0]),
                     static_cast<int>(split[1]), static_cast<int>(split[2]),
                     static_cast<int>(split[0] / kDim), sbhd};
  if (e5m2) {
    launch_rope_bwd<__NV_E5M2>(variant, grid, row_bytes, stream, dq_ptr, dk_ptr, dv_ptr,
                               scale_inv.data_ptr<float>(), q_freqs.data_ptr<float>(),
                               k_freqs.data_ptr<float>(), out_ptr, sh);
  } else {
    launch_rope_bwd<__NV_E4M3>(variant, grid, row_bytes, stream, dq_ptr, dk_ptr, dv_ptr,
                               scale_inv.data_ptr<float>(), q_freqs.data_ptr<float>(),
                               k_freqs.data_ptr<float>(), out_ptr, sh);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dqkv;
}

// Same generator call as TE dropout_fwd (pytorch/csrc/extensions/dropout.cpp).
at::PhiloxCudaState next_dropout_philox_state() {
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  constexpr int64_t rng_elts_per_thread = 4;
  return gen->philox_cuda_state(rng_elts_per_thread);
}

// Same host math as TE dropout_fwd (common/dropout/dropout.cu).
uint32_t bytewise_drop_probability(float dropout_probability) {
  TORCH_CHECK(dropout_probability >= 0 && dropout_probability < 1, "Invalid dropout probability");
  uint32_t value = static_cast<uint32_t>(std::floor(dropout_probability * 256));
  value |= value << 8;
  value |= value << 16;
  return value;
}

float dropout_scale(float dropout_probability) { return 1 / (1 - dropout_probability); }

unsigned int drop_blocks(long long num_chunks) {
  const long long per_block = static_cast<long long>(kDropThreads) * kChunksPerThread;
  return static_cast<unsigned int>((num_chunks + per_block - 1) / per_block);
}

// Variant codes of the dropout kernels: 1 = base kernel; 2x = kChunks x with the table
// dequantize; 3x = kChunks x with the hardware fp4 conversion (NVFP4 only).
int variant_chunks(int64_t variant) {
  const int chunks = static_cast<int>(variant % 10);
  TORCH_CHECK(chunks == 1 || chunks == 2 || chunks == 4 || chunks == 5 || chunks == 8,
              "variant: chunks must be 1, 2, 4, 5 (4 chunks, 4 blocks) or 8");
  return chunks == 5 ? 4 : chunks;
}

unsigned int drop_blocks_v2(long long num_chunks, int chunks) {
  const long long per_block = static_cast<long long>(kDropThreads) * chunks;
  return static_cast<unsigned int>((num_chunks + per_block - 1) / per_block);
}

// Dropout forward of a bf16 tensor. Returns [output, mask] like tex.dropout_fwd.
std::vector<at::Tensor> dropout_fwd_bf16(const at::Tensor &input, float dropout_probability,
                                         int64_t variant) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.is_contiguous(),
              "input: contiguous bf16 CUDA tensor");
  TORCH_CHECK(input.numel() % kChunk == 0, "input: numel must be a multiple of 16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0, "input: 16-byte aligned");
  const uint32_t drop_prob = bytewise_drop_probability(dropout_probability);
  const float scale = dropout_scale(dropout_probability);
  auto output = at::empty_like(input);
  auto mask = at::empty({input.numel() / 8}, input.options().dtype(at::kByte));
  const at::PhiloxCudaState philox = next_dropout_philox_state();
  const long long num_chunks = input.numel() / kChunk;
  if (num_chunks == 0) {
    return {output, mask};
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto *in_ptr = reinterpret_cast<const Bf16x16 *>(input.data_ptr());
  auto *out_ptr = reinterpret_cast<Bf16x16 *>(output.data_ptr());
  auto *mask_ptr = reinterpret_cast<uint16_t *>(mask.data_ptr());
  const bool aligned = reinterpret_cast<uintptr_t>(input.data_ptr()) % 32 == 0 &&
                       reinterpret_cast<uintptr_t>(output.data_ptr()) % 32 == 0;
  const int chunks = variant == 1 ? 1 : variant_chunks(variant);
  if (variant == 1 || !aligned || num_chunks % chunks != 0) {
    dropout_fwd_bf16_kernel<<<drop_blocks(num_chunks), kDropThreads, 0, stream>>>(
        in_ptr, out_ptr, mask_ptr, philox, num_chunks, drop_prob, scale);
  } else if (variant == 42) {
    dropout_fwd_bf16_v2_kernel<2, 1, true><<<drop_blocks_v2(num_chunks, 2), kDropThreads, 0, stream>>>(
        in_ptr, out_ptr, mask_ptr, philox, num_chunks, drop_prob, scale);
  } else if (variant == 44) {
    dropout_fwd_bf16_v2_kernel<4, 1, true><<<drop_blocks_v2(num_chunks, 4), kDropThreads, 0, stream>>>(
        in_ptr, out_ptr, mask_ptr, philox, num_chunks, drop_prob, scale);
  } else if (variant % 10 == 5) {
    dropout_fwd_bf16_v2_kernel<4, 4><<<drop_blocks_v2(num_chunks, 4), kDropThreads, 0, stream>>>(
        in_ptr, out_ptr, mask_ptr, philox, num_chunks, drop_prob, scale);
  } else if (chunks == 8) {
    dropout_fwd_bf16_v2_kernel<8, 1><<<drop_blocks_v2(num_chunks, 8), kDropThreads, 0, stream>>>(
        in_ptr, out_ptr, mask_ptr, philox, num_chunks, drop_prob, scale);
  } else if (chunks == 4) {
    dropout_fwd_bf16_v2_kernel<4, 1><<<drop_blocks_v2(num_chunks, 4), kDropThreads, 0, stream>>>(
        in_ptr, out_ptr, mask_ptr, philox, num_chunks, drop_prob, scale);
  } else if (chunks == 2) {
    dropout_fwd_bf16_v2_kernel<2, 1><<<drop_blocks_v2(num_chunks, 2), kDropThreads, 0, stream>>>(
        in_ptr, out_ptr, mask_ptr, philox, num_chunks, drop_prob, scale);
  } else {
    dropout_fwd_bf16_v2_kernel<1, 1><<<drop_blocks_v2(num_chunks, 1), kDropThreads, 0, stream>>>(
        in_ptr, out_ptr, mask_ptr, philox, num_chunks, drop_prob, scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, mask};
}

template <int kChunks, bool kCvt, bool kBatch = false>
void launch_dropout_nvfp4_v2(const uint2 *data, const uint8_t *scales, const float *amax,
                             Bf16x16 *out, uint16_t *mask, at::PhiloxCudaState philox,
                             const Nvfp4Layout<uint32_t> &layout, uint32_t drop_prob, float scale,
                             cudaStream_t stream) {
  dropout_fwd_nvfp4_v2_kernel<kChunks, kCvt, uint32_t, kBatch>
      <<<drop_blocks_v2(layout.num_chunks, kChunks), kDropThreads, 0, stream>>>(
          data, scales, amax, out, mask, philox, layout, drop_prob, scale);
}

// NVFP4 dequantize (rowwise data, one tensor amax) and dropout forward in one kernel.
// The output is bf16 with the given shape. Returns [output, mask] like tex.dropout_fwd.
std::vector<at::Tensor> dropout_fwd_nvfp4(const at::Tensor &data, const at::Tensor &scales,
                                          const at::Tensor &amax, bool swizzled,
                                          float dropout_probability, std::vector<int64_t> shape,
                                          int64_t variant) {
  TORCH_CHECK(data.is_cuda() && data.scalar_type() == at::kByte && data.is_contiguous(),
              "data: contiguous uint8 CUDA tensor");
  TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == at::kByte && scales.is_contiguous() &&
                  scales.dim() == 2,
              "scales: 2D contiguous uint8 CUDA tensor");
  check_float_scalar(amax, "amax");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(data.data_ptr()) % 16 == 0, "data: 16-byte aligned");
  const long long cols = data.size(-1) * 2;
  TORCH_CHECK(cols % kChunk == 0, "cols must be a multiple of 16");
  const long long numel = data.numel() * 2;
  long long shape_numel = 1;
  for (const int64_t dim : shape) {
    shape_numel *= dim;
  }
  TORCH_CHECK(shape_numel == numel && !shape.empty() && shape.back() == cols, "shape mismatch");
  const long long rows = numel / cols;
  const long long blocks_per_row = cols / kChunk;
  const long long num_tiles_x = (blocks_per_row + 3) / 4;
  if (swizzled) {
    TORCH_CHECK(scales.numel() >= ((rows + 127) / 128) * 128 * num_tiles_x * 4,
                "scales: too small for the swizzled layout");
  } else {
    TORCH_CHECK(scales.size(0) >= rows && scales.size(1) >= blocks_per_row, "scales: too small");
  }
  const uint32_t drop_prob = bytewise_drop_probability(dropout_probability);
  const float scale = dropout_scale(dropout_probability);
  auto output = at::empty(shape, data.options().dtype(at::kBFloat16));
  auto mask = at::empty({numel / 8}, data.options());
  const at::PhiloxCudaState philox = next_dropout_philox_state();
  const long long num_chunks = numel / kChunk;
  if (num_chunks == 0) {
    return {output, mask};
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto *data_ptr = reinterpret_cast<const uint2 *>(data.data_ptr());
  auto *out_ptr = reinterpret_cast<Bf16x16 *>(output.data_ptr());
  auto *mask_ptr = reinterpret_cast<uint16_t *>(mask.data_ptr());
  const long long scale_numel = scales.numel();
  const bool small = num_chunks < (1LL << 31) && scale_numel < (1LL << 31);
  const int chunks = variant == 1 ? 1 : variant_chunks(variant);
  const bool rows_ok = blocks_per_row % chunks == 0 && (swizzled || scales.size(1) % chunks == 0);
  const bool aligned = reinterpret_cast<uintptr_t>(data.data_ptr()) % 32 == 0 &&
                       reinterpret_cast<uintptr_t>(output.data_ptr()) % 32 == 0;
  if (variant == 1 || !small || !rows_ok || !aligned) {
    if (small) {
      const Nvfp4Layout<uint32_t> layout{static_cast<uint32_t>(num_chunks),
                                         static_cast<uint32_t>(blocks_per_row),
                                         static_cast<uint32_t>(scales.size(1)),
                                         static_cast<uint32_t>(num_tiles_x), swizzled};
      dropout_fwd_nvfp4_kernel<uint32_t><<<drop_blocks(num_chunks), kDropThreads, 0, stream>>>(
          data_ptr, scales.data_ptr<uint8_t>(), amax.data_ptr<float>(), out_ptr, mask_ptr, philox,
          layout, drop_prob, scale);
    } else {
      const Nvfp4Layout<uint64_t> layout{static_cast<uint64_t>(num_chunks),
                                         static_cast<uint64_t>(blocks_per_row),
                                         static_cast<uint64_t>(scales.size(1)),
                                         static_cast<uint64_t>(num_tiles_x), swizzled};
      dropout_fwd_nvfp4_kernel<uint64_t><<<drop_blocks(num_chunks), kDropThreads, 0, stream>>>(
          data_ptr, scales.data_ptr<uint8_t>(), amax.data_ptr<float>(), out_ptr, mask_ptr, philox,
          layout, drop_prob, scale);
    }
  } else {
    const Nvfp4Layout<uint32_t> layout{static_cast<uint32_t>(num_chunks),
                                       static_cast<uint32_t>(blocks_per_row),
                                       static_cast<uint32_t>(scales.size(1)),
                                       static_cast<uint32_t>(num_tiles_x), swizzled};
    const bool cvt = variant >= 30;
    const auto *scales_ptr = scales.data_ptr<uint8_t>();
    const auto *amax_ptr = amax.data_ptr<float>();
    if (variant == 44) {
      launch_dropout_nvfp4_v2<4, true, true>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr,
                                             philox, layout, drop_prob, scale, stream);
    } else if (variant == 48) {
      launch_dropout_nvfp4_v2<8, true, true>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr,
                                             philox, layout, drop_prob, scale, stream);
    } else if (variant == 42) {
      launch_dropout_nvfp4_v2<2, true, true>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr,
                                             philox, layout, drop_prob, scale, stream);
    } else if (chunks == 8 && cvt) {
      launch_dropout_nvfp4_v2<8, true>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr, philox,
                                       layout, drop_prob, scale, stream);
    } else if (chunks == 8) {
      launch_dropout_nvfp4_v2<8, false>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr, philox,
                                        layout, drop_prob, scale, stream);
    } else if (chunks == 4 && cvt) {
      launch_dropout_nvfp4_v2<4, true>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr, philox,
                                       layout, drop_prob, scale, stream);
    } else if (chunks == 4) {
      launch_dropout_nvfp4_v2<4, false>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr, philox,
                                        layout, drop_prob, scale, stream);
    } else if (chunks == 2 && cvt) {
      launch_dropout_nvfp4_v2<2, true>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr, philox,
                                       layout, drop_prob, scale, stream);
    } else if (chunks == 2) {
      launch_dropout_nvfp4_v2<2, false>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr, philox,
                                        layout, drop_prob, scale, stream);
    } else if (cvt) {
      launch_dropout_nvfp4_v2<1, true>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr, philox,
                                       layout, drop_prob, scale, stream);
    } else {
      launch_dropout_nvfp4_v2<1, false>(data_ptr, scales_ptr, amax_ptr, out_ptr, mask_ptr, philox,
                                        layout, drop_prob, scale, stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, mask};
}

// Dropout backward of a bf16 tensor, like tex.dropout_bwd.
at::Tensor dropout_bwd_bf16(const at::Tensor &grad, const at::Tensor &mask,
                            float dropout_probability, int64_t variant) {
  TORCH_CHECK(grad.is_cuda() && grad.scalar_type() == at::kBFloat16 && grad.is_contiguous(),
              "grad: contiguous bf16 CUDA tensor");
  TORCH_CHECK(grad.numel() % kChunk == 0, "grad: numel must be a multiple of 16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0, "grad: 16-byte aligned");
  TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == at::kByte && mask.is_contiguous() &&
                  mask.numel() * 8 == grad.numel(),
              "mask: contiguous uint8 with numel / 8 entries");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(mask.data_ptr()) % 2 == 0, "mask: 2-byte aligned");
  TORCH_CHECK(dropout_probability >= 0 && dropout_probability < 1, "Invalid dropout probability");
  const float scale = dropout_scale(dropout_probability);
  auto grad_input = at::empty_like(grad);
  const long long num_chunks = grad.numel() / kChunk;
  if (num_chunks == 0) {
    return grad_input;
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto *grad_ptr = reinterpret_cast<const Bf16x16 *>(grad.data_ptr());
  const auto *mask_ptr = reinterpret_cast<const uint16_t *>(mask.data_ptr());
  auto *out_ptr = reinterpret_cast<Bf16x16 *>(grad_input.data_ptr());
  const bool aligned = reinterpret_cast<uintptr_t>(grad.data_ptr()) % 32 == 0 &&
                       reinterpret_cast<uintptr_t>(grad_input.data_ptr()) % 32 == 0 &&
                       reinterpret_cast<uintptr_t>(mask.data_ptr()) % 8 == 0;
  const int chunks = variant == 1 ? 1 : variant_chunks(variant);
  if (variant == 1 || !aligned || num_chunks % chunks != 0) {
    dropout_bwd_bf16_kernel<<<drop_blocks(num_chunks), kDropThreads, 0, stream>>>(
        grad_ptr, mask_ptr, out_ptr, num_chunks, scale);
  } else if (chunks == 8) {
    dropout_bwd_bf16_v2_kernel<8><<<drop_blocks_v2(num_chunks, 8), kDropThreads, 0, stream>>>(
        grad_ptr, mask_ptr, out_ptr, num_chunks, scale);
  } else if (chunks == 4) {
    dropout_bwd_bf16_v2_kernel<4><<<drop_blocks_v2(num_chunks, 4), kDropThreads, 0, stream>>>(
        grad_ptr, mask_ptr, out_ptr, num_chunks, scale);
  } else if (chunks == 2) {
    dropout_bwd_bf16_v2_kernel<2><<<drop_blocks_v2(num_chunks, 2), kDropThreads, 0, stream>>>(
        grad_ptr, mask_ptr, out_ptr, num_chunks, scale);
  } else {
    dropout_bwd_bf16_v2_kernel<1><<<drop_blocks_v2(num_chunks, 1), kDropThreads, 0, stream>>>(
        grad_ptr, mask_ptr, out_ptr, num_chunks, scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_input;
}

// ---------------------------------------------------------------------------------------------
// Amax producers (amax_prod): the same output bytes as the kernels above, plus the per-block
// maxima of |output|. The nvfp4q quantize reduces the partials, so no separate amax pass runs.
// ---------------------------------------------------------------------------------------------

constexpr int kProdThreads = 256;
constexpr int kProdVecs = 4;  // 8-element vectors of one thread, one block width apart.

// Loads that do not keep the line in L1. The kernels read each byte one time.
__device__ __forceinline__ uint4 load_nc(const uint4 *ptr) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint2 load_nc8(const uint2 *ptr) {
  uint2 v;
  asm("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr));
  return v;
}

// bf16 max of |x| over 8 elements. The max is exact in bf16, and bf16 to float is exact, thus the
// result is the TE amax (max of the float values).
__device__ __forceinline__ __nv_bfloat162 acc_abs_max8(__nv_bfloat162 acc, const Bf16x8 &v) {
  const __nv_bfloat162 *pairs = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
  for (int i = 0; i < kVec / 2; ++i) acc = __hmax2(acc, __habs2(pairs[i]));
  return acc;
}

__device__ __forceinline__ float pair_max_to_float(__nv_bfloat162 acc) {
  return fmaxf(__low2float(acc), __high2float(acc));
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

// Max over the block into partial[blockIdx.x]. All threads must call it. One barrier.
template <int THREADS>
__device__ __forceinline__ void write_block_max(float value, float *shared, float *partial) {
  value = warp_max(value);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < THREADS / 32 ? shared[lane] : 0.0f;
    value = warp_max(value);
    if (lane == 0) partial[blockIdx.x] = value;
  }
}

// FP8 dequantize of 8 elements (TE math, see dequantize_fp8x8).
template <__nv_fp8_interpretation_t kFormat>
__device__ __forceinline__ void dequantize_vec8(uint2 raw, float scale_inv, Bf16x8 *dst,
                                                __nv_bfloat162 &acc) {
  const Bf16x8 out = dequantize_fp8x8<kFormat>(raw, scale_inv);
  *dst = out;
  acc = acc_abs_max8(acc, out);
}

// FP8 dequantize with partials. Dense grid, as the TE unary kernel: the block scheduler keeps
// the active blocks in one memory region. Each thread handles kProdVecs vectors of 8 elements,
// one block width apart, so each load and store instruction of a warp is contiguous.
template <__nv_fp8_interpretation_t kFormat, bool NC>
__global__ void __launch_bounds__(kProdThreads)
    fp8_dequantize_partials_kernel(const uint2 *__restrict__ data,
                                   const float *__restrict__ scale_inv_ptr,
                                   Bf16x8 *__restrict__ out, long long n_vec8,
                                   float *__restrict__ partial) {
  __shared__ float shared[kProdThreads / 32];
  const long long first = blockIdx.x * (static_cast<long long>(kProdVecs) * kProdThreads) + threadIdx.x;
  __nv_bfloat162 acc = __floats2bfloat162_rn(0.0f, 0.0f);
  const float scale_inv = *scale_inv_ptr;
  if (first + (kProdVecs - 1) * kProdThreads < n_vec8) {
    uint2 raw[kProdVecs];
#pragma unroll
    for (int u = 0; u < kProdVecs; ++u) {
      raw[u] = NC ? load_nc8(data + first + u * kProdThreads) : data[first + u * kProdThreads];
    }
#pragma unroll
    for (int u = 0; u < kProdVecs; ++u) {
      dequantize_vec8<kFormat>(raw[u], scale_inv, out + first + u * kProdThreads, acc);
    }
  } else {
    for (long long i = first; i < n_vec8; i += kProdThreads) {
      dequantize_vec8<kFormat>(load_nc8(data + i), scale_inv, out + i, acc);
    }
  }
  write_block_max<kProdThreads>(pair_max_to_float(acc), shared, partial);
}

// out = bf16(float(a) + float(b)), the PyTorch bf16 add.
__device__ __forceinline__ void add_vec8(uint4 a_raw, uint4 b_raw, Bf16x8 *dst,
                                         __nv_bfloat162 &acc) {
  const Bf16x8 &x = reinterpret_cast<const Bf16x8 &>(a_raw);
  const Bf16x8 &y = reinterpret_cast<const Bf16x8 &>(b_raw);
  Bf16x8 z;
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    z.v[e] = __float2bfloat16(__bfloat162float(x.v[e]) + __bfloat162float(y.v[e]));
  }
  *dst = z;
  acc = acc_abs_max8(acc, z);
}

// bf16 add with partials. Dense grid. Each thread handles kProdVecs vectors of 8 elements, one
// block width apart, so each load and store instruction of a warp is contiguous. Plain loads:
// with two input streams, the L1::no_allocate loads cost 4 us more.
__global__ void __launch_bounds__(kProdThreads)
    add_partials_kernel(const uint4 *__restrict__ a, const uint4 *__restrict__ b,
                        Bf16x8 *__restrict__ out, long long n_vec, float *__restrict__ partial) {
  __shared__ float shared[kProdThreads / 32];
  const long long first = blockIdx.x * (static_cast<long long>(kProdVecs) * kProdThreads) + threadIdx.x;
  __nv_bfloat162 acc = __floats2bfloat162_rn(0.0f, 0.0f);
  if (first + (kProdVecs - 1) * kProdThreads < n_vec) {
    uint4 a_raw[kProdVecs];
    uint4 b_raw[kProdVecs];
#pragma unroll
    for (int u = 0; u < kProdVecs; ++u) {
      a_raw[u] = a[first + u * kProdThreads];
      b_raw[u] = b[first + u * kProdThreads];
    }
#pragma unroll
    for (int u = 0; u < kProdVecs; ++u) add_vec8(a_raw[u], b_raw[u], out + first + u * kProdThreads, acc);
  } else {
    for (long long i = first; i < n_vec; i += kProdThreads) add_vec8(a[i], b[i], out + i, acc);
  }
  write_block_max<kProdThreads>(pair_max_to_float(acc), shared, partial);
}

// Tuning variants of the add kernel: block size, vectors of one thread, and plain loads.
template <int THREADS, int VECS, bool NC>
__global__ void __launch_bounds__(THREADS)
    add_partials_variant_kernel(const uint4 *__restrict__ a, const uint4 *__restrict__ b,
                                Bf16x8 *__restrict__ out, long long n_vec,
                                float *__restrict__ partial) {
  __shared__ float shared[THREADS / 32];
  const long long first = blockIdx.x * (static_cast<long long>(VECS) * THREADS) + threadIdx.x;
  __nv_bfloat162 acc = __floats2bfloat162_rn(0.0f, 0.0f);
  if (first + (VECS - 1) * THREADS < n_vec) {
    uint4 a_raw[VECS];
    uint4 b_raw[VECS];
#pragma unroll
    for (int u = 0; u < VECS; ++u) {
      a_raw[u] = NC ? load_nc(a + first + u * THREADS) : a[first + u * THREADS];
      b_raw[u] = NC ? load_nc(b + first + u * THREADS) : b[first + u * THREADS];
    }
#pragma unroll
    for (int u = 0; u < VECS; ++u) add_vec8(a_raw[u], b_raw[u], out + first + u * THREADS, acc);
  } else {
    for (long long i = first; i < n_vec; i += THREADS) {
      add_vec8(load_nc(a + i), load_nc(b + i), out + i, acc);
    }
  }
  write_block_max<THREADS>(pair_max_to_float(acc), shared, partial);
}

template <int THREADS, int VECS, bool NC>
std::vector<at::Tensor> launch_add_variant(const at::Tensor &a, const at::Tensor &b) {
  auto out = at::empty_like(a);
  const long long n_vec = a.numel() / kVec;
  const long long per_block = static_cast<long long>(VECS) * THREADS;
  const int grid = static_cast<int>(std::max<long long>(1, (n_vec + per_block - 1) / per_block));
  auto partials = at::empty({grid}, a.options().dtype(at::kFloat));
  add_partials_variant_kernel<THREADS, VECS, NC><<<grid, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4 *>(a.data_ptr()), reinterpret_cast<const uint4 *>(b.data_ptr()),
      reinterpret_cast<Bf16x8 *>(out.data_ptr()), n_vec, partials.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, partials};
}

std::vector<at::Tensor> add_partials_variant(const at::Tensor &a, const at::Tensor &b, int64_t variant) {
  c10::cuda::CUDAGuard guard(a.device());
  switch (variant) {
    case 0: return launch_add_variant<256, 4, true>(a, b);
    case 1: return launch_add_variant<256, 4, false>(a, b);
    case 2: return launch_add_variant<256, 8, true>(a, b);
    case 3: return launch_add_variant<256, 8, false>(a, b);
    case 4: return launch_add_variant<128, 8, true>(a, b);
    case 5: return launch_add_variant<128, 8, false>(a, b);
    case 6: return launch_add_variant<512, 4, true>(a, b);
    case 7: return launch_add_variant<512, 2, false>(a, b);
    case 8: return launch_add_variant<128, 4, false>(a, b);
    case 9: return launch_add_variant<64, 8, false>(a, b);
    default: TORCH_CHECK(false, "bad variant");
  }
}

// backward_pair_fp8 with the max of the two output vectors.
template <__nv_fp8_interpretation_t kFormat>
__device__ __forceinline__ void backward_pair_fp8_acc(const TokenPair &task, const uint8_t *grad,
                                                      const float *smem, float scale_inv,
                                                      __nv_bfloat16 *dst, __nv_bfloat162 &acc) {
  const Bf16x8 src_lo =
      dequantize_fp8x8<kFormat>(*reinterpret_cast<const uint2 *>(grad), scale_inv);
  const Bf16x8 src_hi =
      dequantize_fp8x8<kFormat>(*reinterpret_cast<const uint2 *>(grad + kHalf), scale_inv);
  const float *cos_v = task.is_q ? smem : smem + 2 * kDim;
  Bf16x8 out_lo, out_hi;
  rope_backward_pair(src_lo, src_hi, cos_v, cos_v + kDim, task.j, out_lo, out_hi);
  store_vec(dst, out_lo);
  store_vec(dst + kHalf, out_hi);
  acc = acc_abs_max8(acc, out_lo);
  acc = acc_abs_max8(acc, out_hi);
}

// qkv_rope_backward_fp8_kernel with partials: one partial for each token block. The block
// reduction uses the __syncthreads of end_barrier, so it adds no extra barrier.
template <__nv_fp8_interpretation_t kFormat>
__global__ void __launch_bounds__(kThreads) qkv_rope_backward_fp8_partials_kernel(
    const uint8_t *__restrict__ dq, const uint8_t *__restrict__ dk, const uint8_t *__restrict__ dv,
    const float *__restrict__ scale_inv_ptr, const float *__restrict__ q_freqs,
    const float *__restrict__ k_freqs, __nv_bfloat16 *__restrict__ dqkv, int s, int b, int h,
    int q_split, int k_split, int v_split, bool sbhd, float *__restrict__ partial) {
  extern __shared__ __align__(128) uint8_t token_row[];  // dq row, dk row, dv row
  __shared__ __align__(16) float smem[4 * kDim];
  __shared__ float warp_amax[kThreads / 32];
  __shared__ __align__(8) uint64_t row_ready;
  const RowOffsets off =
      token_offsets(blockIdx.x, blockIdx.y, s, b, h, q_split, k_split, v_split, sbhd);
  const int total_d = q_split + k_split + v_split;
  const int q_bytes = h * q_split;
  const int k_bytes = h * k_split;
  if (threadIdx.x == 0) {
    const int v_bytes = h * v_split;
    start_barrier(&row_ready, q_bytes + k_bytes + v_bytes);
    copy_to_shared(token_row, dq + off.q, q_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes, dk + off.k, k_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes + k_bytes, dv + off.v, v_bytes, &row_ready);
  }
  const float scale_inv = *scale_inv_ptr;
  load_cos_sin(q_freqs, k_freqs, blockIdx.x, smem);
  wait_barrier(&row_ready);
  const GroupTasks tasks = group_tasks(q_split, k_split);
  const int v_per_group = v_split / (2 * kVec);
  __nv_bfloat162 acc = __floats2bfloat162_rn(0.0f, 0.0f);
  for (int task = threadIdx.x; task < h * v_per_group; task += blockDim.x) {
    const int g = task / v_per_group;
    const int p = (task % v_per_group) * 2 * kVec;
    const uint4 packed =
        *reinterpret_cast<const uint4 *>(token_row + q_bytes + k_bytes + g * v_split + p);
    __nv_bfloat16 *dst = dqkv + off.src + static_cast<long long>(g) * total_d + q_split +
                         k_split + p;
    const Bf16x8 lo = dequantize_fp8x8<kFormat>(make_uint2(packed.x, packed.y), scale_inv);
    const Bf16x8 hi = dequantize_fp8x8<kFormat>(make_uint2(packed.z, packed.w), scale_inv);
    store_vec(dst, lo);
    store_vec(dst + kVec, hi);
    acc = acc_abs_max8(acc, lo);
    acc = acc_abs_max8(acc, hi);
  }
  for (int task = threadIdx.x; task < h * (tasks.q + tasks.k); task += blockDim.x) {
    const TokenPair pair = token_pair(task, tasks, total_d, q_split, k_split);
    const uint8_t *grad = token_row + (pair.is_q ? 0 : q_bytes) + pair.split;
    backward_pair_fp8_acc<kFormat>(pair, grad, smem, scale_inv, dqkv + off.src + pair.packed, acc);
  }
  const float lane_max = warp_max(pair_max_to_float(acc));
  if ((threadIdx.x & 31) == 0) warp_amax[threadIdx.x >> 5] = lane_max;
  end_barrier(&row_ready);
  if (threadIdx.x == 0) {
    float block_amax = warp_amax[0];
#pragma unroll
    for (int w = 1; w < kThreads / 32; ++w) block_amax = fmaxf(block_amax, warp_amax[w]);
    partial[blockIdx.y * gridDim.x + blockIdx.x] = block_amax;
  }
}

// rope_bwd_fp8_v6_kernel with partials: one partial for each token (block). The output bytes
// are those of the tuned kernel. The block reduction uses the __syncthreads of end_barrier, so
// it adds no extra barrier.
template <__nv_fp8_interpretation_t kFormat, int kQh, bool kSameFreqs, bool kFmaFirst>
__global__ void __launch_bounds__(kThreads)
    rope_bwd_fp8_v6_partials_kernel(const uint8_t *__restrict__ dq, const uint8_t *__restrict__ dk,
                                    const uint8_t *__restrict__ dv,
                                    const float *__restrict__ scale_inv_ptr,
                                    const float *__restrict__ q_freqs,
                                    const float *__restrict__ k_freqs,
                                    __nv_bfloat16 *__restrict__ dqkv, RopeShape sh,
                                    float *__restrict__ partial) {
  extern __shared__ __align__(128) uint8_t token_row[];  // dq row, dk row, dv row
  __shared__ __align__(16) float smem[4 * kDim];
  __shared__ float warp_amax[kThreads / 32];
  __shared__ __align__(8) uint64_t row_ready;
  const RowOffsets off = token_offsets(blockIdx.x, blockIdx.y, sh.s, sh.b, sh.h, sh.q_split,
                                       sh.k_split, sh.v_split, sh.sbhd);
  const int total_d = sh.q_split + sh.k_split + sh.v_split;
  const int q_bytes = sh.h * sh.q_split;
  const int k_bytes = sh.h * sh.k_split;
  if (threadIdx.x == 0) {
    const int v_bytes = sh.h * sh.v_split;
    start_barrier(&row_ready, q_bytes + k_bytes + v_bytes);
    copy_to_shared(token_row, dq + off.q, q_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes, dk + off.k, k_bytes, &row_ready);
    copy_to_shared(token_row + q_bytes + k_bytes, dv + off.v, v_bytes, &row_ready);
  }
  const float scale_inv = *scale_inv_ptr;
  const uint64_t scale_inv2 = pk2(scale_inv, scale_inv);
  load_cos_sin_v6<kSameFreqs>(q_freqs, k_freqs, blockIdx.x, smem);
  const int slot = (threadIdx.x % kSlots) * kVec;
  SlotTable2 table = load_slot_table2(smem, slot);
  int table_kind = 0;
  wait_barrier(&row_ready);
  __nv_bfloat162 acc = __floats2bfloat162_rn(0.0f, 0.0f);
  const int heads = sh.h * (kQh + 2);
  for (int hs = threadIdx.x / kSlots; hs < heads; hs += kThreads / kSlots) {
    const HeadTask task = head_task_pow2<kQh>(hs, sh.h, total_d, sh.q_split, sh.k_split);
    __nv_bfloat16 *dst = dqkv + off.src + task.src;
    if (task.kind == 2) {
      const uint4 packed =
          *reinterpret_cast<const uint4 *>(token_row + q_bytes + k_bytes + task.dst + 2 * slot);
      const Bf16x8 lo = dequantize_fp8x8_x2<kFormat>(make_uint2(packed.x, packed.y), scale_inv2);
      const Bf16x8 hi = dequantize_fp8x8_x2<kFormat>(make_uint2(packed.z, packed.w), scale_inv2);
      store_vec(dst + 2 * slot, lo);
      store_vec(dst + 2 * slot + kVec, hi);
      acc = acc_abs_max8(acc, lo);
      acc = acc_abs_max8(acc, hi);
      continue;
    }
    if (!kSameFreqs && task.kind != table_kind) {
      table = load_slot_table2(smem + 2 * kDim * task.kind, slot);
      table_kind = task.kind;
    }
    const uint8_t *grad = token_row + (task.kind == 0 ? 0 : q_bytes) + task.dst + slot;
    const Bf16x8 src_lo =
        dequantize_fp8x8_x2<kFormat>(*reinterpret_cast<const uint2 *>(grad), scale_inv2);
    const Bf16x8 src_hi =
        dequantize_fp8x8_x2<kFormat>(*reinterpret_cast<const uint2 *>(grad + kHalf), scale_inv2);
    uint32_t out_lo[kVec / 2];
    uint32_t out_hi[kVec / 2];
    rope_backward_x2<kFmaFirst>(src_lo, src_hi, table, out_lo, out_hi);
    const uint4 lo4 = make_uint4(out_lo[0], out_lo[1], out_lo[2], out_lo[3]);
    const uint4 hi4 = make_uint4(out_hi[0], out_hi[1], out_hi[2], out_hi[3]);
    *reinterpret_cast<uint4 *>(dst + slot) = lo4;
    *reinterpret_cast<uint4 *>(dst + slot + kHalf) = hi4;
    acc = acc_abs_max8(acc, reinterpret_cast<const Bf16x8 &>(lo4));
    acc = acc_abs_max8(acc, reinterpret_cast<const Bf16x8 &>(hi4));
  }
  const float lane_max = warp_max(pair_max_to_float(acc));
  if ((threadIdx.x & 31) == 0) warp_amax[threadIdx.x >> 5] = lane_max;
  end_barrier(&row_ready);
  if (threadIdx.x == 0) {
    float block_amax = warp_amax[0];
#pragma unroll
    for (int w = 1; w < kThreads / 32; ++w) block_amax = fmaxf(block_amax, warp_amax[w]);
    partial[blockIdx.y * gridDim.x + blockIdx.x] = block_amax;
  }
}

// Dense grid: the number of blocks is the number of partials.
int dense_grid(long long work_items) {
  TORCH_CHECK(work_items < (int64_t{1} << 31) * kProdThreads, "tensor too large");
  return static_cast<int>(std::max<long long>(1, (work_items + kProdThreads - 1) / kProdThreads));
}

// FP8 (tensor scale) dequantize to bf16 as TE, plus partials. Returns (out, partials).
std::vector<at::Tensor> fp8_dequantize_partials_variant(const at::Tensor &data,
                                                        const at::Tensor &scale_inv, bool e5m2,
                                                        bool nc) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() && data.scalar_type() == at::kByte,
              "data: contiguous uint8 CUDA tensor");
  TORCH_CHECK(data.numel() % 16 == 0 && reinterpret_cast<uintptr_t>(data.data_ptr()) % 16 == 0,
              "data: numel must be a multiple of 16, 16-byte aligned");
  check_float_scalar(scale_inv, "scale_inv");
  c10::cuda::CUDAGuard guard(data.device());
  auto out = at::empty(data.sizes(), data.options().dtype(at::kBFloat16));
  const long long n_vec8 = data.numel() / 8;
  const int grid = dense_grid((n_vec8 + kProdVecs - 1) / kProdVecs);
  auto partials = at::empty({grid}, data.options().dtype(at::kFloat));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto *in = reinterpret_cast<const uint2 *>(data.data_ptr());
  auto *dst = reinterpret_cast<Bf16x8 *>(out.data_ptr());
  auto *out_ptr = partials.data_ptr<float>();
  const auto *scale_ptr = scale_inv.data_ptr<float>();
  if (e5m2 && nc) {
    fp8_dequantize_partials_kernel<__NV_E5M2, true><<<grid, kProdThreads, 0, stream>>>(
        in, scale_ptr, dst, n_vec8, out_ptr);
  } else if (e5m2) {
    fp8_dequantize_partials_kernel<__NV_E5M2, false><<<grid, kProdThreads, 0, stream>>>(
        in, scale_ptr, dst, n_vec8, out_ptr);
  } else if (nc) {
    fp8_dequantize_partials_kernel<__NV_E4M3, true><<<grid, kProdThreads, 0, stream>>>(
        in, scale_ptr, dst, n_vec8, out_ptr);
  } else {
    fp8_dequantize_partials_kernel<__NV_E4M3, false><<<grid, kProdThreads, 0, stream>>>(
        in, scale_ptr, dst, n_vec8, out_ptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, partials};
}

// The default: loads that do not keep the line in L1.
std::vector<at::Tensor> fp8_dequantize_partials(const at::Tensor &data, const at::Tensor &scale_inv,
                                                bool e5m2) {
  return fp8_dequantize_partials_variant(data, scale_inv, e5m2, true);
}

// bf16 a + b as PyTorch, plus partials. Returns (out, partials).
std::vector<at::Tensor> add_partials(const at::Tensor &a, const at::Tensor &b) {
  for (const auto &t : {a, b}) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(),
                "a, b: contiguous bf16 CUDA tensors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "a, b: 16-byte aligned");
  }
  TORCH_CHECK(a.sizes() == b.sizes() && a.numel() % kVec == 0, "a, b: same shape, numel % 8 == 0");
  c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty_like(a);
  const long long n_vec = a.numel() / kVec;
  const int grid = dense_grid((n_vec + kProdVecs - 1) / kProdVecs);
  auto partials = at::empty({grid}, a.options().dtype(at::kFloat));
  add_partials_kernel<<<grid, kProdThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4 *>(a.data_ptr()), reinterpret_cast<const uint4 *>(b.data_ptr()),
      reinterpret_cast<Bf16x8 *>(out.data_ptr()), n_vec, partials.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, partials};
}

// rope_backward_fp8 with partials (one for each token). Returns (dqkv, partials).
// Variant 6 (the default) is the tuned v6 kernel with partials; it needs 8 q heads for each k
// head. Other shapes and variant 1 run the base kernel with partials.
std::vector<at::Tensor> rope_backward_fp8_partials(const at::Tensor &dq, const at::Tensor &dk,
                                                   const at::Tensor &dv,
                                                   const at::Tensor &scale_inv,
                                                   const at::Tensor &q_freqs,
                                                   const at::Tensor &k_freqs,
                                                   std::vector<int64_t> split, bool sbhd,
                                                   bool e5m2, int64_t variant) {
  check_rope_inputs(q_freqs, split);
  check_rope_inputs(k_freqs, split);
  check_float_scalar(scale_inv, "scale_inv");
  TORCH_CHECK(dq.dim() == 4 && dk.dim() == 4 && dv.dim() == 4, "grads: 4D");
  check_byte_tensor(dq, dq.sizes(), "dq");
  check_byte_tensor(dk, dk.sizes(), "dk");
  check_byte_tensor(dv, dv.sizes(), "dv");
  TORCH_CHECK(dq.size(3) == kDim && dk.size(3) == kDim && dv.size(3) == kDim, "grads: d = 128");
  for (const auto &t : {dk, dv}) {
    TORCH_CHECK(t.size(0) == dq.size(0) && t.size(1) == dq.size(1), "grads: s and b must match");
  }
  const int s = sbhd ? dq.size(0) : dq.size(1);
  const int b = sbhd ? dq.size(1) : dq.size(0);
  TORCH_CHECK(q_freqs.size(0) >= s && k_freqs.size(0) >= s, "freqs: too few positions");
  const int64_t total_d = split[0] + split[1] + split[2];
  const int64_t total_hd = (dq.size(2) + dk.size(2) + dv.size(2)) * dq.size(3);
  TORCH_CHECK(total_hd % total_d == 0, "grads: head counts do not match split");
  auto size = dq.sizes().vec();
  size[2] = total_hd / total_d;
  size[3] = total_d;
  TORCH_CHECK(dq.size(2) == size[2] * split[0] / kDim && dk.size(2) == size[2] &&
                  dv.size(2) == size[2],
              "grads: head counts do not match split");
  c10::cuda::CUDAGuard guard(dq.device());
  auto dqkv = at::empty(size, dq.options().dtype(at::kBFloat16));
  auto partials = at::empty({static_cast<int64_t>(s) * b}, dq.options().dtype(at::kFloat));
  const dim3 grid(s, b);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto *dq_ptr = dq.data_ptr<uint8_t>();
  const auto *dk_ptr = dk.data_ptr<uint8_t>();
  const auto *dv_ptr = dv.data_ptr<uint8_t>();
  auto *out_ptr = reinterpret_cast<__nv_bfloat16 *>(dqkv.data_ptr());
  auto *part_ptr = partials.data_ptr<float>();
  const int row_bytes = static_cast<int>((dq.size(2) + dk.size(2) + dv.size(2)) * dq.size(3));
  TORCH_CHECK(row_bytes <= kMaxRowBytes, "grads: token row larger than the shared memory buffer");
  for (const auto &t : {dq, dk, dv}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "grads: 16-byte aligned");
  }
  const RopeShape sh{s, b, static_cast<int>(size[2]), static_cast<int>(split[0]),
                     static_cast<int>(split[1]), static_cast<int>(split[2]),
                     static_cast<int>(split[0] / kDim), sbhd};
  const bool same = q_freqs.data_ptr() == k_freqs.data_ptr();
  const bool tuned = (variant == 6 || variant == 7) && sh.qh == 8;
  auto launch_v6 = [&](auto format_tag, auto same_tag, auto fma_tag) {
    rope_bwd_fp8_v6_partials_kernel<decltype(format_tag)::value, 8, decltype(same_tag)::value,
                                    decltype(fma_tag)::value>
        <<<grid, kThreads, row_bytes, stream>>>(dq_ptr, dk_ptr, dv_ptr,
                                                scale_inv.data_ptr<float>(),
                                                q_freqs.data_ptr<float>(),
                                                k_freqs.data_ptr<float>(), out_ptr, sh, part_ptr);
  };
  auto launch_format = [&](auto format_tag) {
    if (!tuned) {
      qkv_rope_backward_fp8_partials_kernel<decltype(format_tag)::value>
          <<<grid, kThreads, row_bytes, stream>>>(
              dq_ptr, dk_ptr, dv_ptr, scale_inv.data_ptr<float>(), q_freqs.data_ptr<float>(),
              k_freqs.data_ptr<float>(), out_ptr, s, b, size[2], split[0], split[1], split[2],
              sbhd, part_ptr);
    } else if (same && variant == 6) {
      launch_v6(format_tag, std::true_type{}, std::true_type{});
    } else if (same) {
      launch_v6(format_tag, std::true_type{}, std::false_type{});
    } else if (variant == 6) {
      launch_v6(format_tag, std::false_type{}, std::true_type{});
    } else {
      launch_v6(format_tag, std::false_type{}, std::false_type{});
    }
  };
  if (e5m2) {
    launch_format(std::integral_constant<__nv_fp8_interpretation_t, __NV_E5M2>{});
  } else {
    launch_format(std::integral_constant<__nv_fp8_interpretation_t, __NV_E4M3>{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dqkv, partials};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_forward_fp8", &rope_forward_fp8, "Fused QKV RoPE forward with FP8 E4M3 output",
        py::arg("qkv"), py::arg("q_freqs"), py::arg("k_freqs"), py::arg("split"), py::arg("sbhd"),
        py::arg("scale"), py::arg("amax"), py::arg("scale_inv"), py::arg("q_out"), py::arg("k_out"),
        py::arg("v_out"), py::arg("variant") = kBestRopeFwd);
  m.def("rope_backward_fp8", &rope_backward_fp8, "Fused FP8 dequantize and QKV RoPE backward",
        py::arg("dq"), py::arg("dk"), py::arg("dv"), py::arg("scale_inv"), py::arg("q_freqs"),
        py::arg("k_freqs"), py::arg("split"), py::arg("sbhd"), py::arg("e5m2"),
        py::arg("variant") = kBestRopeBwd);
  m.def("dropout_fwd_bf16", &dropout_fwd_bf16, "Dropout forward, same bits as TE",
        py::arg("input"), py::arg("dropout_probability"), py::arg("variant") = kBestDropoutBf16);
  m.def("dropout_fwd_nvfp4", &dropout_fwd_nvfp4, "NVFP4 dequantize and dropout forward",
        py::arg("data"), py::arg("scales"), py::arg("amax"), py::arg("swizzled"),
        py::arg("dropout_probability"), py::arg("shape"), py::arg("variant") = kBestDropoutNvfp4);
  m.def("dropout_bwd_bf16", &dropout_bwd_bf16, "Dropout backward, same bits as TE",
        py::arg("grad"), py::arg("mask"), py::arg("dropout_probability"),
        py::arg("variant") = kBestDropoutBwd);
  m.def("fp8_dequantize_partials", &fp8_dequantize_partials,
        "FP8 dequantize to bf16 as TE, with amax partials");
  m.def("add_partials", &add_partials, "bf16 a + b as PyTorch, with amax partials");
  m.def("fp8_dequantize_partials_variant", &fp8_dequantize_partials_variant,
        "Tuning variant of fp8_dequantize_partials", py::arg("data"), py::arg("scale_inv"),
        py::arg("e5m2"), py::arg("nc"));
  m.def("add_partials_variant", &add_partials_variant, "Tuning variants of add_partials",
        py::arg("a"), py::arg("b"), py::arg("variant"));
  m.def("rope_backward_fp8_partials", &rope_backward_fp8_partials,
        "Fused FP8 dequantize and QKV RoPE backward, with amax partials", py::arg("dq"),
        py::arg("dk"), py::arg("dv"), py::arg("scale_inv"), py::arg("q_freqs"),
        py::arg("k_freqs"), py::arg("split"), py::arg("sbhd"), py::arg("e5m2"),
        py::arg("variant") = kBestRopeBwd);
}
