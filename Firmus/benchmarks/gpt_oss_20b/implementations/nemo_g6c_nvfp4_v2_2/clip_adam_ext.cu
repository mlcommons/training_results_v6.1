// Copyright (c) 2026, Firmus Technologies. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Classification: Internal
// FP32 Adam step that applies the gradient-clip coefficient while loading the
// gradient. The math matches transformer_engine::multi_tensor_adam::AdamFunctor
// for fp32 g/p/m/v, preceded by multi_tensor_scale_tensor on the gradients. FP32 or BF16
// gradients; BF16 products are rounded to BF16 as the in-place reference scale stores them.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>

namespace {

constexpr int kILP = 4;
constexpr int kBlock = 512;
constexpr int64_t kChunk = 65536;

typedef enum { ADAM_MODE_0 = 0, ADAM_MODE_1 = 1 } adamMode_t;

// Scaled gradient as the reference sees it: multi_tensor_scale_tensor writes g * s back in the
// gradient dtype, and the Adam op reads that stored value.
__device__ __forceinline__ float scaled_grad(float g, float s) { return __fmul_rn(g, s); }
__device__ __forceinline__ float scaled_grad(__nv_bfloat16 g, float s) {
  return __bfloat162float(__float2bfloat16_rn(__fmul_rn(__bfloat162float(g), s)));
}

// Upper bound on the grid; 0 launches one CTA per chunk. A capped grid loops over the chunks,
// which lets the step share the GPU with work on other streams.
int g_max_ctas = 0;

template <typename GT>
__device__ __forceinline__ void clip_adam_chunk(
    const int b, const int64_t *__restrict__ ptrs, const int64_t *__restrict__ sizes,
    const int *__restrict__ block_tensor, const int *__restrict__ block_chunk,
    const float *__restrict__ scale, const float beta1, const float beta2,
    const float beta1_correction, const float beta2_correction, const float epsilon,
    const float lr, const int mode, const float decay) {
  using MATH_T = float;
  const int t = block_tensor[b];
  const int64_t chunk_idx = block_chunk[b];
  int64_t n = sizes[t];
  const GT *g = reinterpret_cast<const GT *>(ptrs[4 * t + 0]) + chunk_idx * kChunk;
  float *p = reinterpret_cast<float *>(ptrs[4 * t + 1]) + chunk_idx * kChunk;
  float *m = reinterpret_cast<float *>(ptrs[4 * t + 2]) + chunk_idx * kChunk;
  float *v = reinterpret_cast<float *>(ptrs[4 * t + 3]) + chunk_idx * kChunk;
  n -= chunk_idx * kChunk;
  const float s = *scale;

  for (int64_t i_start = 0; i_start < n && i_start < kChunk; i_start += blockDim.x * kILP) {
    MATH_T r_g[kILP];
    MATH_T r_p[kILP];
    MATH_T r_m[kILP];
    MATH_T r_v[kILP];
#pragma unroll
    for (int ii = 0; ii < kILP; ii++) {
      int64_t i = i_start + threadIdx.x + ii * blockDim.x;
      if (i < n && i < kChunk) {
        // __fmul_rn keeps the product rounded on its own, as when it is stored to memory.
        r_g[ii] = scaled_grad(g[i], s);
        r_p[ii] = p[i];
        r_m[ii] = m[i];
        r_v[ii] = v[i];
      } else {
        r_g[ii] = MATH_T(0);
        r_p[ii] = MATH_T(0);
        r_m[ii] = MATH_T(0);
        r_v[ii] = MATH_T(0);
      }
    }
#pragma unroll
    for (int ii = 0; ii < kILP; ii++) {
      if (mode == ADAM_MODE_0) {  // L2
        r_g[ii] = r_g[ii] + (decay * r_p[ii]);
        r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
        r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
        MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
        MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
        MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
        MATH_T update = next_m_unbiased / denom;
        r_p[ii] = r_p[ii] - (lr * update);
      } else {  // weight decay
        r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
        r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
        MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
        MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
        MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
        MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
        r_p[ii] = r_p[ii] - (lr * update);
      }
    }
#pragma unroll
    for (int ii = 0; ii < kILP; ii++) {
      int64_t i = i_start + threadIdx.x + ii * blockDim.x;
      if (i < n && i < kChunk) {
        p[i] = r_p[ii];
        m[i] = r_m[ii];
        v[i] = r_v[ii];
      }
    }
  }
}

template <typename GT>
__global__ void __launch_bounds__(kBlock)
    clip_adam_kernel(const int nblocks, const int64_t *__restrict__ ptrs,
                     const int64_t *__restrict__ sizes, const int *__restrict__ block_tensor,
                     const int *__restrict__ block_chunk, const float *__restrict__ scale,
                     const float beta1, const float beta2, const float beta1_correction,
                     const float beta2_correction, const float epsilon, const float lr,
                     const int mode, const float decay) {
  for (int b = blockIdx.x; b < nblocks; b += gridDim.x) {
    clip_adam_chunk<GT>(b, ptrs, sizes, block_tensor, block_chunk, scale, beta1, beta2,
                        beta1_correction, beta2_correction, epsilon, lr, mode, decay);
  }
}

// Same update for BF16 params with int16 remainders holding the low half of the FP32
// master (TE AdamFunctorMasterParamRemainder). ptrs rows are g, p, m, v, p_remainder.
// Each thread handles kVec contiguous elements with 16-byte loads and stores; the per-element
// math is the same as TE's, so only the memory access pattern differs.
constexpr int kVec = 8;

// The update is spelled out with intrinsics in the exact FMA shape of TE's fast-math build of
// AdamFunctorMasterParamRemainder (read from its SASS). Left to contraction, the unrolled vector
// loop fuses the m and v updates the other way round and drifts by an ulp.
__device__ __forceinline__ void adam_rem_elem(float g, int16_t &p, int16_t &p_rem, float &m,
                                              float &v, const float beta1, const float beta2,
                                              const float beta1_correction,
                                              const float beta2_correction, const float epsilon,
                                              const float lr, const int mode,
                                              const float decay) {
  union fp32_or_int162 {
    float fp32;
    int16_t int16[2];
  };
  fp32_or_int162 master;
  int16_t lp = p;
  if (p_rem < 0) lp--;  // Undo rounding
  master.int16[1] = lp;
  master.int16[0] = p_rem;
  float r_p = master.fp32;
  if (mode == ADAM_MODE_0) g = __fmaf_rn(decay, r_p, g);  // L2
  m = __fmaf_rn(beta1, m, __fmul_rn(1 - beta1, g));
  v = __fmaf_rn(beta2, v, __fmul_rn(__fmul_rn(1 - beta2, g), g));
  const float next_m_unbiased = __fdividef(m, beta1_correction);
  const float next_v_unbiased = __fdividef(v, beta2_correction);
  const float denom = __fadd_rn(sqrtf(next_v_unbiased), epsilon);
  float update = __fdividef(next_m_unbiased, denom);
  if (mode != ADAM_MODE_0) update = __fmaf_rn(decay, r_p, update);  // weight decay
  r_p = __fmaf_rn(-update, lr, r_p);
  master.fp32 = r_p;
  lp = master.int16[1];
  p_rem = master.int16[0];
  if (p_rem < 0) lp++;  // Round up
  p = lp;
}

template <typename T>
__device__ __forceinline__ void load_vec(const T *src, T (&dst)[kVec]) {
  constexpr int n = sizeof(T) * kVec / 16;
  const uint4 *s = reinterpret_cast<const uint4 *>(src);
  uint4 *d = reinterpret_cast<uint4 *>(dst);
#pragma unroll
  for (int k = 0; k < n; k++) d[k] = s[k];
}

template <typename T>
__device__ __forceinline__ void store_vec(T *dst, const T (&src)[kVec]) {
  constexpr int n = sizeof(T) * kVec / 16;
  uint4 *d = reinterpret_cast<uint4 *>(dst);
  const uint4 *s = reinterpret_cast<const uint4 *>(src);
#pragma unroll
  for (int k = 0; k < n; k++) d[k] = s[k];
}

__device__ __forceinline__ float bf16_bits_abs(const int16_t x) {
  return fabsf(__int_as_float(static_cast<int>(static_cast<uint16_t>(x)) << 16));
}

// Max over the block, valid in thread 0.
__device__ __forceinline__ float block_max(float v) {
  __shared__ float warp_max[kBlock / 32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  if (lane == 0) warp_max[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < kBlock / 32 ? warp_max[lane] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  }
  __syncthreads();
  return v;
}

// tensor_slot[t] >= 0 folds max |updated bf16 param| of tensor t into amax[tensor_slot[t]]
// (non-negative floats order like their bit patterns). nullptr skips it.
template <typename GT>
__device__ __forceinline__ void clip_adam_rem_chunk(
    const int b, const int64_t *__restrict__ ptrs, const int64_t *__restrict__ sizes,
    const int *__restrict__ block_tensor, const int *__restrict__ block_chunk,
    const float *__restrict__ scale, const float beta1, const float beta2,
    const float beta1_correction, const float beta2_correction, const float epsilon,
    const float lr, const int mode, const float decay, const int *__restrict__ tensor_slot,
    float *__restrict__ amax) {
  const int t = block_tensor[b];
  const int64_t chunk_idx = block_chunk[b];
  const GT *g = reinterpret_cast<const GT *>(ptrs[5 * t + 0]) + chunk_idx * kChunk;
  int16_t *p = reinterpret_cast<int16_t *>(ptrs[5 * t + 1]) + chunk_idx * kChunk;
  float *m = reinterpret_cast<float *>(ptrs[5 * t + 2]) + chunk_idx * kChunk;
  float *v = reinterpret_cast<float *>(ptrs[5 * t + 3]) + chunk_idx * kChunk;
  int16_t *p_remainder = reinterpret_cast<int16_t *>(ptrs[5 * t + 4]) + chunk_idx * kChunk;
  const int64_t left = sizes[t] - chunk_idx * kChunk;
  const int64_t n = left < kChunk ? left : kChunk;
  const float s = *scale;
  const bool aligned = ((reinterpret_cast<uintptr_t>(g) | reinterpret_cast<uintptr_t>(p) |
                         reinterpret_cast<uintptr_t>(m) | reinterpret_cast<uintptr_t>(v) |
                         reinterpret_cast<uintptr_t>(p_remainder)) &
                        15) == 0;
  const int64_t n_vec = aligned ? n / kVec * kVec : 0;
  float amx = 0.f;

  for (int64_t i0 = int64_t(threadIdx.x) * kVec; i0 < n_vec; i0 += int64_t(blockDim.x) * kVec) {
    alignas(16) GT r_g[kVec];
    alignas(16) int16_t r_p[kVec];
    alignas(16) int16_t r_rem[kVec];
    alignas(16) float r_m[kVec];
    alignas(16) float r_v[kVec];
    load_vec(g + i0, r_g);
    load_vec(p + i0, r_p);
    load_vec(p_remainder + i0, r_rem);
    load_vec(m + i0, r_m);
    load_vec(v + i0, r_v);
#pragma unroll
    for (int k = 0; k < kVec; k++) {
      adam_rem_elem(scaled_grad(r_g[k], s), r_p[k], r_rem[k], r_m[k], r_v[k], beta1, beta2,
                    beta1_correction, beta2_correction, epsilon, lr, mode, decay);
      amx = fmaxf(amx, bf16_bits_abs(r_p[k]));
    }
    store_vec(p_remainder + i0, r_rem);
    store_vec(p + i0, r_p);
    store_vec(m + i0, r_m);
    store_vec(v + i0, r_v);
  }
  for (int64_t i = n_vec + threadIdx.x; i < n; i += blockDim.x) {
    int16_t lp = p[i], lrem = p_remainder[i];
    float lm = m[i], lv = v[i];
    adam_rem_elem(scaled_grad(g[i], s), lp, lrem, lm, lv, beta1, beta2, beta1_correction,
                  beta2_correction, epsilon, lr, mode, decay);
    amx = fmaxf(amx, bf16_bits_abs(lp));
    p_remainder[i] = lrem;
    p[i] = lp;
    m[i] = lm;
    v[i] = lv;
  }
  if (tensor_slot != nullptr && tensor_slot[t] >= 0) {
    amx = block_max(amx);
    if (threadIdx.x == 0) atomicMax(reinterpret_cast<int *>(amax + tensor_slot[t]), __float_as_int(amx));
  }
}

template <typename GT>
__global__ void __launch_bounds__(kBlock)
    clip_adam_rem_kernel(const int nblocks, const int64_t *__restrict__ ptrs,
                         const int64_t *__restrict__ sizes, const int *__restrict__ block_tensor,
                         const int *__restrict__ block_chunk, const float *__restrict__ scale,
                         const float beta1, const float beta2, const float beta1_correction,
                         const float beta2_correction, const float epsilon, const float lr,
                         const int mode, const float decay, const int *__restrict__ tensor_slot,
                         float *__restrict__ amax) {
  for (int b = blockIdx.x; b < nblocks; b += gridDim.x) {
    clip_adam_rem_chunk<GT>(b, ptrs, sizes, block_tensor, block_chunk, scale, beta1, beta2,
                            beta1_correction, beta2_correction, epsilon, lr, mode, decay,
                            tensor_slot, amax);
  }
}

void launch(bool remainder, bool bf16_grad, at::Tensor ptrs, at::Tensor sizes,
            at::Tensor block_tensor, at::Tensor block_chunk, at::Tensor scale, const float lr,
            const float beta1, const float beta2, const float epsilon, const int step,
            const int mode, const int bias_correction, const float weight_decay,
            const at::Tensor *tensor_slot = nullptr, const at::Tensor *amax = nullptr) {
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }
  TORCH_CHECK(ptrs.is_cuda() && ptrs.scalar_type() == at::kLong && ptrs.is_contiguous());
  TORCH_CHECK(sizes.is_cuda() && sizes.scalar_type() == at::kLong);
  TORCH_CHECK(block_tensor.is_cuda() && block_tensor.scalar_type() == at::kInt);
  TORCH_CHECK(block_chunk.is_cuda() && block_chunk.scalar_type() == at::kInt);
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::kFloat && scale.numel() == 1);
  const int64_t nblocks = block_tensor.numel();
  if (nblocks == 0) return;
  TORCH_CHECK(nblocks < (int64_t(1) << 31));
  const int64_t grid = g_max_ctas > 0 && g_max_ctas < nblocks ? g_max_ctas : nblocks;
  const int *slot_ptr = nullptr;
  float *amax_ptr = nullptr;
  if (tensor_slot != nullptr) {
    TORCH_CHECK(remainder, "the amax fold is only in the remainder kernel");
    TORCH_CHECK(tensor_slot->is_cuda() && tensor_slot->scalar_type() == at::kInt &&
                tensor_slot->numel() == sizes.numel());
    TORCH_CHECK(amax->is_cuda() && amax->scalar_type() == at::kFloat && amax->is_contiguous());
    slot_ptr = tensor_slot->data_ptr<int>();
    amax_ptr = amax->data_ptr<float>();
  }
  auto stream = at::cuda::getCurrentCUDAStream();
#define CLIP_ADAM_ARGS                                                                      \
  static_cast<int>(nblocks), ptrs.data_ptr<int64_t>(), sizes.data_ptr<int64_t>(),            \
      block_tensor.data_ptr<int>(), block_chunk.data_ptr<int>(), scale.data_ptr<float>(),    \
      beta1, beta2, bias_correction1, bias_correction2, epsilon, lr, mode, weight_decay
  if (remainder && bf16_grad) {
    clip_adam_rem_kernel<__nv_bfloat16>
        <<<grid, kBlock, 0, stream>>>(CLIP_ADAM_ARGS, slot_ptr, amax_ptr);
  } else if (remainder) {
    clip_adam_rem_kernel<float><<<grid, kBlock, 0, stream>>>(CLIP_ADAM_ARGS, slot_ptr, amax_ptr);
  } else if (bf16_grad) {
    clip_adam_kernel<__nv_bfloat16><<<grid, kBlock, 0, stream>>>(CLIP_ADAM_ARGS);
  } else {
    clip_adam_kernel<float><<<grid, kBlock, 0, stream>>>(CLIP_ADAM_ARGS);
  }
#undef CLIP_ADAM_ARGS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define CLIP_ADAM_ENTRY(NAME, REM, BF16G)                                                    \
  void NAME(at::Tensor ptrs, at::Tensor sizes, at::Tensor block_tensor, at::Tensor block_chunk, \
            at::Tensor scale, const float lr, const float beta1, const float beta2,            \
            const float epsilon, const int step, const int mode, const int bias_correction,    \
            const float weight_decay) {                                                        \
    launch(REM, BF16G, ptrs, sizes, block_tensor, block_chunk, scale, lr, beta1, beta2,        \
           epsilon, step, mode, bias_correction, weight_decay);                                \
  }
CLIP_ADAM_ENTRY(clip_adam, false, false)
CLIP_ADAM_ENTRY(clip_adam_bf16g, false, true)
CLIP_ADAM_ENTRY(clip_adam_rem, true, false)
CLIP_ADAM_ENTRY(clip_adam_rem_bf16g, true, true)
#undef CLIP_ADAM_ENTRY

#define CLIP_ADAM_AMAX_ENTRY(NAME, BF16G)                                                    \
  void NAME(at::Tensor ptrs, at::Tensor sizes, at::Tensor block_tensor, at::Tensor block_chunk, \
            at::Tensor scale, const float lr, const float beta1, const float beta2,            \
            const float epsilon, const int step, const int mode, const int bias_correction,    \
            const float weight_decay, at::Tensor tensor_slot, at::Tensor amax) {               \
    launch(true, BF16G, ptrs, sizes, block_tensor, block_chunk, scale, lr, beta1, beta2,       \
           epsilon, step, mode, bias_correction, weight_decay, &tensor_slot, &amax);           \
  }
CLIP_ADAM_AMAX_ENTRY(clip_adam_rem_amax, false)
CLIP_ADAM_AMAX_ENTRY(clip_adam_rem_bf16g_amax, true)
#undef CLIP_ADAM_AMAX_ENTRY

void set_max_ctas(const int n) { g_max_ctas = n > 0 ? n : 0; }

// Global L2 norm equal to multi_tensor_l2norm (per_tensor=False). Each chunk's block sum
// depends only on its data, so all chunks go in one launch; l2norm_assemble then adds the
// sums per block slot in the reference's launch order and runs its cleanup reduction.
constexpr int kL2Block = 512;
constexpr int kL2ILP = 4;
constexpr int kL2Slots = 320;

__device__ __forceinline__ float to_float(const float x) { return x; }
__device__ __forceinline__ float to_float(const __nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float reduce_block_into_lanes(float *x, float val) {
  const int tid = threadIdx.x;
  x[tid] = val;
  __syncthreads();
#pragma unroll
  for (int i = kL2Block >> 1; i >= 64; i >>= 1) {
    if (tid < i) x[tid] = x[tid] + x[tid + i];
    __syncthreads();
  }
  float final = 0.f;
  if (tid < 32) {
    final = x[tid] + x[tid + 32];
#pragma unroll
    for (int i = 16; i >= 1; i >>= 1) final = final + __shfl_down_sync(0xffffffff, final, i);
  }
  return final;
}

template <typename T>
__global__ void __launch_bounds__(kL2Block)
    l2norm_chunks_kernel(const int64_t *__restrict__ ptrs, const int64_t *__restrict__ sizes,
                         const int *__restrict__ block_tensor, const int *__restrict__ block_chunk,
                         const int *__restrict__ sel, const int chunk_size,
                         float *__restrict__ sums, int *noop) {
  const int b = sel ? sel[blockIdx.x] : blockIdx.x;
  const int t = block_tensor[b];
  const int chunk_idx = block_chunk[b];
  int n = static_cast<int>(sizes[t]);
  const T *x = reinterpret_cast<const T *>(ptrs[t]) + chunk_idx * chunk_size;
  n -= chunk_idx * chunk_size;
  __shared__ float s_vals[kL2Block];
  float vals[kL2ILP];
#pragma unroll
  for (int i = 0; i < kL2ILP; i++) vals[i] = 0.f;
  if (n % kL2ILP == 0 && chunk_size % kL2ILP == 0 &&
      reinterpret_cast<uintptr_t>(x) % (kL2ILP * sizeof(T)) == 0) {
    typedef typename std::aligned_storage<kL2ILP * sizeof(T), kL2ILP * alignof(T)>::type LT;
    for (int i_start = threadIdx.x; i_start * kL2ILP < n && i_start * kL2ILP < chunk_size;
         i_start += blockDim.x) {
      alignas(LT) T r_x[kL2ILP];
      *reinterpret_cast<LT *>(r_x) = reinterpret_cast<const LT *>(x)[i_start];
#pragma unroll
      for (int ii = 0; ii < kL2ILP; ii++) {
        const float next = to_float(r_x[ii]);
        vals[ii] += next * next;
      }
    }
  } else {
    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * kL2ILP) {
#pragma unroll
      for (int ii = 0; ii < kL2ILP; ii++) {
        const int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          const float next = to_float(x[i]);
          vals[ii] += next * next;
        }
      }
    }
  }
  float val = 0.f;
  for (int i = 0; i < kL2ILP; i++) val += vals[i];
  const float final = reduce_block_into_lanes(s_vals, val);
  if (threadIdx.x == 0) {
    if (!isfinite(final)) *noop = 1;
    sums[b] = final;
  }
}

// slot_chunk[l * kL2Slots + b] is the chunk block b of reference launch l handled, or -1.
__global__ void __launch_bounds__(kL2Block)
    l2norm_assemble_kernel(const float *__restrict__ sums, const int *__restrict__ slot_chunk,
                           const int nlaunch, float *__restrict__ ret, int *pre_noop,
                           int *noop) {
  __shared__ float s_vals[kL2Block];
  if (pre_noop && threadIdx.x == 0) {
    if (*pre_noop) *noop = 1;
    *pre_noop = 0;
  }
  float acc = 0.f;
  if (threadIdx.x < kL2Slots) {
    for (int l = 0; l < nlaunch; l++) {
      const int c = slot_chunk[l * kL2Slots + threadIdx.x];
      if (c >= 0) acc += sums[c];
    }
  }
  const float final = reduce_block_into_lanes(s_vals, acc);
  if (threadIdx.x == 0) *ret = sqrt(final);
}

void launch_l2_chunks(at::Tensor &ptrs, at::Tensor &sizes, at::Tensor &block_tensor,
                      at::Tensor &block_chunk, const int *sel, const int64_t nblocks,
                      const int chunk_size, const bool bf16, at::Tensor &sums, at::Tensor &noop) {
  auto stream = at::cuda::getCurrentCUDAStream();
#define L2NORM_LAUNCH(T)                                                                     \
  l2norm_chunks_kernel<T><<<nblocks, kL2Block, 0, stream>>>(                                 \
      ptrs.data_ptr<int64_t>(), sizes.data_ptr<int64_t>(), block_tensor.data_ptr<int>(),     \
      block_chunk.data_ptr<int>(), sel, chunk_size, sums.data_ptr<float>(),                  \
      noop.data_ptr<int>())
  if (bf16) {
    L2NORM_LAUNCH(__nv_bfloat16);
  } else {
    L2NORM_LAUNCH(float);
  }
#undef L2NORM_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_l2_assemble(at::Tensor &sums, at::Tensor &slot_chunk, at::Tensor &ret, int *pre_noop,
                        at::Tensor &noop) {
  l2norm_assemble_kernel<<<1, kL2Block, 0, at::cuda::getCurrentCUDAStream()>>>(
      sums.data_ptr<float>(), slot_chunk.data_ptr<int>(),
      static_cast<int>(slot_chunk.numel() / kL2Slots), ret.data_ptr<float>(), pre_noop,
      noop.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void l2norm(at::Tensor ptrs, at::Tensor sizes, at::Tensor block_tensor, at::Tensor block_chunk,
            at::Tensor slot_chunk, const int chunk_size, const bool bf16, at::Tensor sums,
            at::Tensor noop, at::Tensor ret) {
  const int64_t nblocks = block_tensor.numel();
  TORCH_CHECK(nblocks > 0 && nblocks < (int64_t(1) << 31) && sums.numel() >= nblocks);
  TORCH_CHECK(slot_chunk.numel() % kL2Slots == 0 && ret.scalar_type() == at::kFloat);
  TORCH_CHECK(noop.scalar_type() == at::kInt && sums.scalar_type() == at::kFloat);
  launch_l2_chunks(ptrs, sizes, block_tensor, block_chunk, nullptr, nblocks, chunk_size, bf16,
                   sums, noop);
  launch_l2_assemble(sums, slot_chunk, ret, nullptr, noop);
}

// l2norm in parts: l2norm_part computes the chunk sums of the blocks listed in `sel`
// (into their own slots of `sums`, flagging non-finite sums in `noop`), l2norm_finish
// assembles all of them and moves pre_noop's flag into noop.
void l2norm_part(at::Tensor ptrs, at::Tensor sizes, at::Tensor block_tensor,
                 at::Tensor block_chunk, at::Tensor sel, const int chunk_size, const bool bf16,
                 at::Tensor sums, at::Tensor noop) {
  const int64_t nsel = sel.numel();
  TORCH_CHECK(nsel > 0 && nsel <= block_tensor.numel() && sel.scalar_type() == at::kInt);
  TORCH_CHECK(noop.scalar_type() == at::kInt && sums.scalar_type() == at::kFloat);
  TORCH_CHECK(sums.numel() >= block_tensor.numel());
  launch_l2_chunks(ptrs, sizes, block_tensor, block_chunk, sel.data_ptr<int>(), nsel,
                   chunk_size, bf16, sums, noop);
}

void l2norm_finish(at::Tensor sums, at::Tensor slot_chunk, at::Tensor pre_noop, at::Tensor noop,
                   at::Tensor ret) {
  TORCH_CHECK(slot_chunk.numel() % kL2Slots == 0 && ret.scalar_type() == at::kFloat);
  TORCH_CHECK(noop.scalar_type() == at::kInt && pre_noop.scalar_type() == at::kInt);
  launch_l2_assemble(sums, slot_chunk, ret, pre_noop.data_ptr<int>(), noop);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("clip_adam", &clip_adam, "FP32 Adam with fused gradient clip scale");
  m.def("clip_adam_bf16g", &clip_adam_bf16g, "clip_adam with BF16 gradients");
  m.def("clip_adam_rem", &clip_adam_rem,
        "BF16 param + int16 remainder Adam with fused gradient clip scale");
  m.def("clip_adam_rem_bf16g", &clip_adam_rem_bf16g, "clip_adam_rem with BF16 gradients");
  m.def("clip_adam_rem_amax", &clip_adam_rem_amax,
        "clip_adam_rem that also max-reduces |updated param| into per-tensor amax slots");
  m.def("clip_adam_rem_bf16g_amax", &clip_adam_rem_bf16g_amax,
        "clip_adam_rem_amax with BF16 gradients");
  m.def("set_max_ctas", &set_max_ctas, "Cap the grid of later launches (0: one CTA per chunk)");
  m.def("l2norm", &l2norm, "multi_tensor_l2norm (per_tensor=False) in two launches");
  m.def("l2norm_part", &l2norm_part, "Chunk sums of the selected l2norm blocks");
  m.def("l2norm_finish", &l2norm_finish, "Assemble the l2norm chunk sums");
  m.attr("CHUNK") = kChunk;
}
