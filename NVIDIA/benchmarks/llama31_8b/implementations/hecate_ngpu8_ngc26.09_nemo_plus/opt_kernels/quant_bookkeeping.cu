// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Portions adapted from NVIDIA Transformer Engine (Apache-2.0, Copyright (c) 2022-2026 NVIDIA CORPORATION
// & AFFILIATES): the mbarrier / TMA helpers follow common/util/ptx.cuh, warp_reduce_max follows
// common/utils.cuh, the 16x16 Hadamard transform fragments and amax tile code follow
// common/hadamard_transform/hadamard_transform_utils.cuh and hadamard_transform.cu, and
// write_rng_state follows common/fused_attn/utils.cu.

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "quant_bookkeeping.h"

namespace pqb {

namespace {

constexpr int kThreadsPerWarp = 32;
constexpr float k16x16HadamardScale = 0.25f;

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, const uint32_t count) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(p), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_invalid(uint64_t* mbar) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(p) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(p) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, const uint32_t tx) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(p), "r"(tx) : "memory");
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint32_t mbar_ptr, const uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(done)
      : "r"(mbar_ptr), "r"(parity)
      : "memory");
  return static_cast<bool>(done);
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, const uint32_t parity) {
  uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  while (!mbarrier_try_wait_parity(p, parity)) {
  }
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;");
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
    void* dst_shmem, const void* tensor_map_ptr, const uint32_t offset_x, const uint32_t offset_y,
    uint64_t* mbar) {
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst_shmem));
  uint32_t m = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(d),
      "l"(tensor_map_ptr), "r"(offset_x), "r"(offset_y), "r"(m)
      : "memory");
}

template <int kNumBarriers, int kThreads>
__device__ __forceinline__ void initialize_barriers(uint64_t* mbar, const bool is_master) {
  if (is_master) {
#pragma unroll
    for (int i = 0; i < kNumBarriers; ++i) mbarrier_init(&mbar[i], kThreads);
    fence_proxy_async_shared_cta();
  }
  __syncthreads();
}

template <int kNumBarriers>
__device__ __forceinline__ void destroy_barriers(uint64_t* mbar, const bool is_master) {
  if (is_master) {
#pragma unroll
    for (int i = 0; i < kNumBarriers; ++i) mbarrier_invalid(&mbar[i]);
  }
}

__device__ __forceinline__ void copy_2d_to_shared(void* dst, const void* tensor_map,
                                                  const uint32_t chunk_x, const uint32_t chunk_y,
                                                  const uint32_t num_bytes, uint64_t* barrier,
                                                  const bool is_master) {
  if (is_master) {
    cp_async_bulk_tensor_2d_global_to_shared(dst, tensor_map, chunk_x, chunk_y, barrier);
    mbarrier_arrive_expect_tx(barrier, num_bytes);
  } else {
    mbarrier_arrive(barrier);
  }
}

template <int kNumElems>
__device__ __forceinline__ float warp_reduce_max(const float m) {
  float tmp = m;
#pragma unroll
  for (int delta = kNumElems / 2; delta > 0; delta /= 2) {
    const float other = __shfl_down_sync(0xFFFFFFFF, tmp, delta);
    tmp = fmaxf(tmp, other);
  }
  return tmp;
}

template <bool kTranspose>
__device__ __forceinline__ void ldmatrix_x4_m8n8_shared_b16(uint32_t& a0, uint32_t& a1,
                                                            uint32_t& a2, uint32_t& a3,
                                                            void* addr) {
  auto s = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  if constexpr (kTranspose) {
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(s));
  } else {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(s));
  }
}

__device__ __forceinline__ void unpack_max_of_packed_bf16(uint32_t& packed_bf16, float& dst) {
  __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&packed_bf16);
  float f_a = __bfloat162float(v.x);
  float f_b = __bfloat162float(v.y);
  asm volatile("max.xorsign.abs.f32 %0, %1, %2;\n\t" : "=f"(dst) : "f"(f_a), "f"(f_b));
  dst = fabsf(dst);
}

__device__ __forceinline__ void mma_m16_n16_k16_amax(uint32_t a0, uint32_t a1, uint32_t a2,
                                                     uint32_t a3, uint32_t b0, uint32_t b1,
                                                     uint32_t b2, uint32_t b3,
                                                     uint32_t& amax_result) {
  uint32_t zero = 0;
  uint32_t t0, t1, t2, t3, t4, t5, t6, t7;
  uint32_t c0, c1, c2, c3;
  asm volatile(
      "wmma.mma.sync.aligned.row.row.m16n16k16.f32.bf16.bf16.f32 \n"
      "{%0, %1, %2, %3, %4, %5, %6, %7}, \n"
      "{%8, %9, %10, %11}, \n"
      "{%12, %13, %14, %15}, \n"
      "{%16, %17, %18, %19, %20, %21, %22, %23};\n\t"
      : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3), "=r"(t4), "=r"(t5), "=r"(t6), "=r"(t7)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(zero),
        "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(zero));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c0) : "r"(t1), "r"(t0));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c1) : "r"(t3), "r"(t2));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c2) : "r"(t5), "r"(t4));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c3) : "r"(t7), "r"(t6));
  uint32_t max_even, max_odd;
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t" : "=r"(max_even) : "r"(c0), "r"(c2));
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t" : "=r"(max_odd) : "r"(c1), "r"(c3));
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
               : "=r"(amax_result)
               : "r"(max_even), "r"(max_odd));
}

__device__ __forceinline__ void get_hadamard_fragment_t(uint32_t* had_frag_t,
                                                        uint16_t random_sign_mask_t) {
  const int32_t tid = threadIdx.x % 32;
  float temp_t[2];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const uint32_t r = i * 8 + tid / 4;
#pragma unroll
    for (int j = 0; j < 2; j++) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const uint32_t c = j * 8 + k + tid % 4 * 2;
        const int32_t base_sign = __popc(r & c);
        const int32_t sign_t = ((random_sign_mask_t >> c) ^ base_sign);
        temp_t[k] = copysignf(k16x16HadamardScale, __int_as_float(sign_t << 31));
      }
      asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t"
                   : "=r"(had_frag_t[i * 2 + j])
                   : "f"(temp_t[1]), "f"(temp_t[0]));
    }
  }
}

__device__ __forceinline__ uint32_t swizzle_128B_atom_32B(uint32_t row, uint32_t col) {
  const uint32_t xor_factor = (row * 2) % 8;
  return row * 8 + (col ^ xor_factor);
}

template <bool kPreRht, bool kTransposed>
__device__ __forceinline__ void compute_tile(const uint32_t b_frag_t[4], __nv_bfloat16* in_sh_ptr,
                                             int swizzle_idx, uint32_t& pre_rht_amax_reg,
                                             uint32_t& amax_t_reg) {
  uint32_t a[4];
  if constexpr (kTransposed) {
    uint32_t tmp;
    ldmatrix_x4_m8n8_shared_b16<true>(a[0], a[1], a[2], a[3],
                                      reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
    mma_m16_n16_k16_amax(a[0], a[2], a[1], a[3], b_frag_t[0], b_frag_t[1], b_frag_t[2],
                         b_frag_t[3], tmp);
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(amax_t_reg)
                 : "r"(amax_t_reg), "r"(tmp));
  }
  if constexpr (kPreRht) {
    if constexpr (!kTransposed) {
      ldmatrix_x4_m8n8_shared_b16<false>(a[0], a[1], a[2], a[3],
                                         reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
    }
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t" : "=r"(a[0]) : "r"(a[0]), "r"(a[1]));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t" : "=r"(a[2]) : "r"(a[2]), "r"(a[3]));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t" : "=r"(a[0]) : "r"(a[0]), "r"(a[2]));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(pre_rht_amax_reg)
                 : "r"(a[0]), "r"(pre_rht_amax_reg));
  }
}

__device__ __forceinline__ void write_rng_state(int64_t* rng_state, const PhiloxArgs& ph) {
  if (ph.captured) {
    rng_state[0] = *ph.seed_ptr;
    rng_state[1] = static_cast<int64_t>(*ph.offset_ptr + static_cast<int64_t>(ph.offset_intragraph));
  } else {
    rng_state[0] = static_cast<int64_t>(ph.seed);
    rng_state[1] = static_cast<int64_t>(ph.offset);
  }
}

__device__ __forceinline__ void publish_amax(float a, float b, float* out_a, float* out_b,
                                             float* out_b2, uint32_t* ws, unsigned int num_ctas) {
  atomicMax(reinterpret_cast<int*>(&ws[0]), __float_as_int(a));
  atomicMax(reinterpret_cast<int*>(&ws[1]), __float_as_int(b));
  __threadfence();
  const unsigned int ticket = atomicAdd(&ws[2], 1u);
  if (ticket == num_ctas - 1) {
    __threadfence();
    const uint32_t ra = atomicExch(&ws[0], 0u);
    const uint32_t rb = atomicExch(&ws[1], 0u);
    atomicExch(&ws[2], 0u);
    if (out_a != nullptr) *out_a = __uint_as_float(ra);
    if (out_b != nullptr) *out_b = __uint_as_float(rb);
    if (out_b2 != nullptr) *out_b2 = __uint_as_float(rb);
  }
}

constexpr int kHad = 16;
constexpr int kBuffY = 64, kBuffX = 64;
constexpr int kThreadsX = 128;
constexpr int kNumWarps = kThreadsX / kThreadsPerWarp;

constexpr int kNumBuf = 3, kCtasPerSm = 4;

template <bool kPreRht, bool kTransposed>
__global__ void __launch_bounds__(kThreadsX)
    rht_amax_kernel(const __grid_constant__ CUtensorMap tensor_map_input, float* row_amax,
                    float* col_amax, uint16_t random_sign_mask_t, uint32_t* ws,
                    int64_t* rng_state, const PhiloxArgs ph, int tiles_x, int num_tiles) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const bool is_master = threadIdx.x == 0;
  if (rng_state != nullptr && blockIdx.x == 0 && is_master) {
    write_rng_state(rng_state, ph);
  }

  extern __shared__ __align__(128) char dynamic_shmem[];
  uint8_t* dshmem =
      reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(dynamic_shmem) + 127) & ~127ULL);
  constexpr int buff_bytes = kBuffX * kBuffY * sizeof(__nv_bfloat16);
  __nv_bfloat16* bufs = reinterpret_cast<__nv_bfloat16*>(dshmem);
  dshmem += kNumBuf * buff_bytes;
  uint64_t* mbar = reinterpret_cast<uint64_t*>(dshmem);
  dshmem += sizeof(uint64_t) * kNumBuf;
  float* stage_t = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;
  float* stage_pre = reinterpret_cast<float*>(dshmem);

  const int G = static_cast<int>(gridDim.x);
  const int my_tiles = (num_tiles - static_cast<int>(blockIdx.x) + G - 1) / G;
  auto issue = [&](int i) {
    const int tile = static_cast<int>(blockIdx.x) + i * G;
    copy_2d_to_shared(bufs + (i % kNumBuf) * (kBuffX * kBuffY), &tensor_map_input,
                      static_cast<uint32_t>((tile % tiles_x) * kBuffX),
                      static_cast<uint32_t>((tile / tiles_x) * kBuffY), buff_bytes,
                      &mbar[i % kNumBuf], is_master);
  };

  initialize_barriers<kNumBuf, kThreadsX>(mbar, is_master);
#pragma unroll
  for (int i = 0; i < kNumBuf - 1; ++i) {
    if (i < my_tiles) issue(i);
  }

  uint32_t had_frag_t[4] = {0, 0, 0, 0};
  if constexpr (kTransposed) get_hadamard_fragment_t(had_frag_t, random_sign_mask_t);

  float zero = 0.0f;
  uint32_t pre_reg = __float_as_uint(zero);
  uint32_t t_reg = __float_as_uint(zero);

  const int warp_id = threadIdx.x / kThreadsPerWarp;
  const int lane = threadIdx.x % kThreadsPerWarp;
  const int swizzle_idx = swizzle_128B_atom_32B(lane % kHad, lane / kHad + warp_id * 2);

  for (int i = 0; i < my_tiles; ++i) {

    if (i + kNumBuf - 1 < my_tiles) issue(i + kNumBuf - 1);
    mbarrier_wait_parity(&mbar[i % kNumBuf], static_cast<uint32_t>((i / kNumBuf) & 1));
    __nv_bfloat16* in_sh_ptr = bufs + (i % kNumBuf) * (kBuffX * kBuffY);

#pragma unroll
    for (int cy = 0; cy < kBuffY / kHad; cy++) {
      compute_tile<kPreRht, kTransposed>(had_frag_t, in_sh_ptr + cy * kHad * kBuffX, swizzle_idx,
                                         pre_reg, t_reg);
    }
    __syncthreads();
    fence_proxy_async_shared_cta();
  }

  float pre_amax = 0.0f, t_amax = 0.0f;
  if constexpr (kPreRht) unpack_max_of_packed_bf16(pre_reg, pre_amax);
  if constexpr (kTransposed) unpack_max_of_packed_bf16(t_reg, t_amax);

  const float wpre = kPreRht ? warp_reduce_max<kThreadsPerWarp>(pre_amax) : 0.0f;
  const float wt = kTransposed ? warp_reduce_max<kThreadsPerWarp>(t_amax) : 0.0f;
  if (lane == 0) {
    stage_pre[warp_id] = wpre;
    stage_t[warp_id] = wt;
  }
  __syncthreads();
  if (warp_id == 0) {
    float p = lane < kNumWarps ? stage_pre[lane] : 0.0f;
    float t = lane < kNumWarps ? stage_t[lane] : 0.0f;
    p = warp_reduce_max<4>(p);
    t = warp_reduce_max<4>(t);
    if (lane == 0) {
      publish_amax(p, t, kPreRht ? row_amax : nullptr, kTransposed ? col_amax : nullptr, nullptr,
                   ws, gridDim.x);
    }
  }
  destroy_barriers<kNumBuf>(mbar, is_master);
#endif
}

constexpr int kAmaxThreads = 512;

__global__ void __launch_bounds__(kAmaxThreads)
    amax_kernel(const uint4* __restrict__ x, int64_t num_vec, float* out_a, float* out_b,
                uint32_t* ws, const float* noop) {
  if (noop != nullptr && noop[0] == 1.0f) return;
  const __nv_bfloat162 zero2 = __floats2bfloat162_rn(0.0f, 0.0f);
  __nv_bfloat162 m0 = zero2, m1 = zero2;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < num_vec;
       i += stride) {
    uint4 v = __ldg(x + i);
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
    m0 = __hmax2(__habs2(p[0]), m0);
    m1 = __hmax2(__habs2(p[1]), m1);
    m0 = __hmax2(__habs2(p[2]), m0);
    m1 = __hmax2(__habs2(p[3]), m1);
  }
  m0 = __hmax2(m0, m1);
  float m = fmaxf(__bfloat162float(m0.x), __bfloat162float(m0.y));
  __shared__ float staging[kAmaxThreads / kThreadsPerWarp];
  const int warp_id = threadIdx.x / kThreadsPerWarp;
  const int lane = threadIdx.x % kThreadsPerWarp;
  m = warp_reduce_max<kThreadsPerWarp>(m);
  if (lane == 0) staging[warp_id] = m;
  __syncthreads();
  if (warp_id == 0) {
    float r = lane < kAmaxThreads / kThreadsPerWarp ? staging[lane] : 0.0f;
    r = warp_reduce_max<kAmaxThreads / kThreadsPerWarp>(r);
    if (lane == 0) publish_amax(0.0f, r, nullptr, out_a, out_b, ws, gridDim.x);
  }
}

PFN_cuTensorMapEncodeTiled_v12000 get_encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = []() {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &q);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

int sm_count() {
  static int n = []() {
    int dev = 0, c = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, dev);
    return c;
  }();
  return n;
}

template <bool kPreRht, bool kTransposed>
cudaError_t launch_rht_amax_t(const CUtensorMap& tmap, float* row_amax, float* col_amax,
                              uint16_t sign_mask_t, uint32_t* ws, int64_t* rng_state,
                              const PhiloxArgs& ph, int tiles_x, int num_tiles,
                              cudaStream_t stream) {
  const size_t shmem = kNumBuf * kBuffX * kBuffY * sizeof(__nv_bfloat16) +
                       sizeof(uint64_t) * kNumBuf + 2 * kNumWarps * sizeof(float) + 128;
  int ctas = kCtasPerSm * sm_count();
  if (ctas > num_tiles) ctas = num_tiles;
  rht_amax_kernel<kPreRht, kTransposed><<<ctas, kThreadsX, shmem, stream>>>(
      tmap, row_amax, col_amax, sign_mask_t, ws, rng_state, ph, tiles_x, num_tiles);
  return cudaGetLastError();
}

}

cudaError_t launch_rht_amax(const void* x, int64_t M, int64_t N, float* row_amax, float* col_amax,
                            uint16_t sign_mask_t, uint32_t* ws, int64_t* rng_state,
                            const PhiloxArgs& ph, cudaStream_t stream) {
  if (M % kHad != 0 || N % 8 != 0 || (reinterpret_cast<uintptr_t>(x) & 15) != 0) {
    return cudaErrorInvalidValue;
  }
  if (row_amax == nullptr && col_amax == nullptr) return cudaErrorInvalidValue;
  auto encode = get_encode_fn();
  if (encode == nullptr) return cudaErrorInvalidDeviceFunction;
  alignas(64) CUtensorMap tmap{};
  cuuint64_t size[2] = {static_cast<cuuint64_t>(N), static_cast<cuuint64_t>(M)};
  cuuint64_t stride[1] = {static_cast<cuuint64_t>(N) * sizeof(__nv_bfloat16)};
  cuuint32_t box[2] = {kBuffX, kBuffY};
  cuuint32_t elem_stride[2] = {1, 1};
  CUresult r = encode(&tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(x), size,
                      stride, box, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                      CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) return cudaErrorInvalidValue;
  const int tiles_x = static_cast<int>((N + kBuffX - 1) / kBuffX);
  const int tiles_y = static_cast<int>((M + kBuffY - 1) / kBuffY);
  const int num_tiles = tiles_x * tiles_y;
  if (row_amax != nullptr && col_amax != nullptr) {
    return launch_rht_amax_t<true, true>(tmap, row_amax, col_amax, sign_mask_t, ws, rng_state, ph,
                                         tiles_x, num_tiles, stream);
  } else if (row_amax != nullptr) {
    return launch_rht_amax_t<true, false>(tmap, row_amax, col_amax, sign_mask_t, ws, rng_state, ph,
                                          tiles_x, num_tiles, stream);
  }
  return launch_rht_amax_t<false, true>(tmap, row_amax, col_amax, sign_mask_t, ws, rng_state, ph,
                                        tiles_x, num_tiles, stream);
}

cudaError_t launch_amax(const void* x, int64_t n, float* out_a, float* out_b, uint32_t* ws,
                        const float* noop, cudaStream_t stream) {
  if (n <= 0 || n % 8 != 0 || (reinterpret_cast<uintptr_t>(x) & 15) != 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t num_vec = n / 8;
  int64_t blocks = (num_vec + kAmaxThreads - 1) / kAmaxThreads;
  const int64_t cap = static_cast<int64_t>(sm_count()) * 4;
  if (blocks > cap) blocks = cap;
  amax_kernel<<<static_cast<unsigned>(blocks), kAmaxThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(x), num_vec, out_a, out_b, ws, noop);
  return cudaGetLastError();
}

}
