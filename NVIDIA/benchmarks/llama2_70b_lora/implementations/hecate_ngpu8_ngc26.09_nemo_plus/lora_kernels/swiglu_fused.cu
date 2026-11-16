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

// SwiGLU forward and backward for bf16 with the math of TE (commit 5b28d7aa), plus the amax.
// - Forward: cast/fp8/gated_fp8.cuh, IS_BWD == false, ActOP == silu (util/math.h).
// - Backward: the same file, IS_BWD == true, the silu/dsilu branch with sigmoidf.
// The amax is max|value| over the fp32 values before the cast to bf16, as in TE.
// TE reads scale at run time (1 for bf16 output), so scale is a kernel argument here too.
// The amax needs no memset. Each kernel uses a device scratch value and a block counter.
// The last block writes the amax and sets both values back to zero.
// Thus kernels that share one scratch slot must not run at the same time.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include "swiglu_math.cuh"

#include <map>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

using namespace swiglu_math;  // The shared math (swiglu_math.cuh).

// Backward of one element, as the TE silu/dsilu branch.
template <int kMath>
__device__ __forceinline__ void backward_value(const float act_elt, const float gate_elt,
                                               const float grad_elt, float &after_dact,
                                               float &after_dgate) {
  if constexpr (kMath == kExact) {
    const float dgate_elt = 1.0f;
    const float x = act_elt;
    const float s = te_sigmoidf(x);
    const float act_x = x * s;
    const float dact_x = x * s * (1 - s) + s;
    after_dact = dact_x * grad_elt * gate_elt;
    after_dgate = dgate_elt * act_x * grad_elt;
  } else {
    after_dact = act_elt * grad_elt;
    after_dgate = gate_elt * grad_elt;
  }
}

template <int kMath, int kWords>
__device__ __forceinline__ void backward_vec(const Vec<kWords> &grad, const Vec<kWords> &act,
                                             const Vec<kWords> &gate, const float scale,
                                             Vec<kWords> &out_act, Vec<kWords> &out_gate,
                                             float &local) {
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    float da0, dg0, da1, dg1;
    backward_value<kMath>(bf16_lo(act.w[w]), bf16_lo(gate.w[w]), bf16_lo(grad.w[w]), da0, dg0);
    backward_value<kMath>(bf16_hi(act.w[w]), bf16_hi(gate.w[w]), bf16_hi(grad.w[w]), da1, dg1);
    out_act.w[w] = pack_bf16(scale * da0, scale * da1);
    out_gate.w[w] = pack_bf16(scale * dg0, scale * dg1);
    local = fmaxf(local, fabsf(da0));
    local = fmaxf(local, fabsf(dg0));
    local = fmaxf(local, fabsf(da1));
    local = fmaxf(local, fabsf(dg1));
  }
}

// The kExact backward with f32x2 math. __expf(-x) is ex2(-x * l) with the float l below.
// Two cases need the scalar kExact code for the full vector:
// - One ex2 input is below -126. The scalar code then uses the ex2 range step.
// - One d is not in [1, 2^126). The scalar code then uses the slow path of __frcp_rn.
template <int kWords>
__device__ __forceinline__ void backward_vec_packed(const Vec<kWords> &grad,
                                                    const Vec<kWords> &act,
                                                    const Vec<kWords> &gate, const float scale,
                                                    Vec<kWords> &out_act, Vec<kWords> &out_gate,
                                                    float &local) {
  const float l = -1.4426950216293334961f;
  uint64_t d[kWords];
  float r_min = 0.f;
  float d_max = 0.f;
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const uint64_t r = mul2(pair_of(act.w[w]), pk(l, l));
    r_min = fminf(r_min, fminf(pk_lo(r), pk_hi(r)));
    d[w] = add2(ex2_pair(r), pk(1.f, 1.f));
    d_max = fmaxf(d_max, fmaxf(pk_lo(d[w]), pk_hi(d[w])));
  }
  if (!(r_min >= -126.f) || !(d_max < kTwoPow126)) {
    backward_vec<kExact>(grad, act, gate, scale, out_act, out_gate, local);
    return;
  }
  const uint64_t scale2 = pk(scale, scale);
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const uint64_t s = reciprocal_pair(d[w]);
    const uint64_t act_x = mul2(pair_of(act.w[w]), s);
    const uint64_t one_minus_s = fma2(s, pk(-1.f, -1.f), pk(1.f, 1.f));
    const uint64_t dact_x = fma2(act_x, one_minus_s, s);
    const uint64_t g = pair_of(grad.w[w]);
    const uint64_t after_dact = mul2(mul2(dact_x, g), pair_of(gate.w[w]));
    const uint64_t after_dgate = mul2(act_x, g);
    const uint64_t oa = mul2(scale2, after_dact);
    const uint64_t og = mul2(scale2, after_dgate);
    out_act.w[w] = pack_bf16(pk_lo(oa), pk_hi(oa));
    out_gate.w[w] = pack_bf16(pk_lo(og), pk_hi(og));
    local = fmaxf(local, fmaxf(fabsf(pk_lo(after_dact)), fabsf(pk_hi(after_dact))));
    local = fmaxf(local, fmaxf(fabsf(pk_lo(after_dgate)), fabsf(pk_hi(after_dgate))));
  }
}

// ---- FP8 output: the FP8 delayed-scaling path of TE cast_fp8_gated_kernel ----
// TE: out = OType(scale * after_act), amax = max(amax, |after_act|). The conversion is
// cvt.rn.satfinite to E4M3 (forward) or E5M2 (backward).

template <__nv_fp8_interpretation_t kFormat>
__device__ __forceinline__ uint32_t cvt_fp8x2(const float lo, const float hi) {
  return __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, kFormat);
}

// Put the 16-bit pair of word w into the output word w / 2.
__device__ __forceinline__ void put_pair(uint32_t *out, const int w, const uint32_t pair) {
  if (w % 2 == 0) {
    out[w / 2] = pair;
  } else {
    out[w / 2] |= pair << 16;
  }
}

template <__nv_fp8_interpretation_t kFormat, int kWords>
__device__ __forceinline__ Vec<kWords / 2> forward_vec_fp8_exact(const Vec<kWords> &act,
                                                                 const Vec<kWords> &gate,
                                                                 const float scale,
                                                                 float &local) {
  Vec<kWords / 2> out;
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const float v0 = forward_value<kExact>(bf16_lo(act.w[w]), bf16_lo(gate.w[w]));
    const float v1 = forward_value<kExact>(bf16_hi(act.w[w]), bf16_hi(gate.w[w]));
    put_pair(out.w, w, cvt_fp8x2<kFormat>(scale * v0, scale * v1));
    local = fmaxf(local, fabsf(v0));
    local = fmaxf(local, fabsf(v1));
  }
  return out;
}

template <__nv_fp8_interpretation_t kFormat, int kWords>
__device__ __forceinline__ Vec<kWords / 2> forward_vec_fp8_packed(const Vec<kWords> &act,
                                                                  const Vec<kWords> &gate,
                                                                  const float scale,
                                                                  float &local) {
  uint64_t d[kWords];
  float d_max = 0.f;
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    d[w] = one_plus_expf_neg_pair(pair_of(act.w[w]));
    d_max = fmaxf(d_max, fmaxf(pk_lo(d[w]), pk_hi(d[w])));
  }
  if (!(d_max < kTwoPow126)) {
    return forward_vec_fp8_exact<kFormat>(act, gate, scale, local);
  }
  Vec<kWords / 2> out;
  const uint64_t scale2 = pk(scale, scale);
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const uint64_t silu = mul2(pair_of(act.w[w]), reciprocal_pair(d[w]));
    const uint64_t v = mul2(silu, pair_of(gate.w[w]));
    const uint64_t o = mul2(scale2, v);
    put_pair(out.w, w, cvt_fp8x2<kFormat>(pk_lo(o), pk_hi(o)));
    local = fmaxf(local, fmaxf(fabsf(pk_lo(v)), fabsf(pk_hi(v))));
  }
  return out;
}

template <__nv_fp8_interpretation_t kFormat, int kWords>
__device__ __forceinline__ void backward_vec_fp8_exact(const Vec<kWords> &grad,
                                                       const Vec<kWords> &act,
                                                       const Vec<kWords> &gate, const float scale,
                                                       Vec<kWords / 2> &out_act,
                                                       Vec<kWords / 2> &out_gate, float &local) {
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    float da0, dg0, da1, dg1;
    backward_value<kExact>(bf16_lo(act.w[w]), bf16_lo(gate.w[w]), bf16_lo(grad.w[w]), da0, dg0);
    backward_value<kExact>(bf16_hi(act.w[w]), bf16_hi(gate.w[w]), bf16_hi(grad.w[w]), da1, dg1);
    put_pair(out_act.w, w, cvt_fp8x2<kFormat>(scale * da0, scale * da1));
    put_pair(out_gate.w, w, cvt_fp8x2<kFormat>(scale * dg0, scale * dg1));
    local = fmaxf(local, fabsf(da0));
    local = fmaxf(local, fabsf(dg0));
    local = fmaxf(local, fabsf(da1));
    local = fmaxf(local, fabsf(dg1));
  }
}

template <__nv_fp8_interpretation_t kFormat, int kWords>
__device__ __forceinline__ void backward_vec_fp8_packed(const Vec<kWords> &grad,
                                                        const Vec<kWords> &act,
                                                        const Vec<kWords> &gate, const float scale,
                                                        Vec<kWords / 2> &out_act,
                                                        Vec<kWords / 2> &out_gate, float &local) {
  const float l = -1.4426950216293334961f;
  uint64_t d[kWords];
  float r_min = 0.f;
  float d_max = 0.f;
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const uint64_t r = mul2(pair_of(act.w[w]), pk(l, l));
    r_min = fminf(r_min, fminf(pk_lo(r), pk_hi(r)));
    d[w] = add2(ex2_pair(r), pk(1.f, 1.f));
    d_max = fmaxf(d_max, fmaxf(pk_lo(d[w]), pk_hi(d[w])));
  }
  if (!(r_min >= -126.f) || !(d_max < kTwoPow126)) {
    backward_vec_fp8_exact<kFormat>(grad, act, gate, scale, out_act, out_gate, local);
    return;
  }
  const uint64_t scale2 = pk(scale, scale);
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const uint64_t s = reciprocal_pair(d[w]);
    const uint64_t act_x = mul2(pair_of(act.w[w]), s);
    const uint64_t one_minus_s = fma2(s, pk(-1.f, -1.f), pk(1.f, 1.f));
    const uint64_t dact_x = fma2(act_x, one_minus_s, s);
    const uint64_t g = pair_of(grad.w[w]);
    const uint64_t after_dact = mul2(mul2(dact_x, g), pair_of(gate.w[w]));
    const uint64_t after_dgate = mul2(act_x, g);
    const uint64_t oa = mul2(scale2, after_dact);
    const uint64_t og = mul2(scale2, after_dgate);
    put_pair(out_act.w, w, cvt_fp8x2<kFormat>(pk_lo(oa), pk_hi(oa)));
    put_pair(out_gate.w, w, cvt_fp8x2<kFormat>(pk_lo(og), pk_hi(og)));
    local = fmaxf(local, fmaxf(fabsf(pk_lo(after_dact)), fabsf(pk_hi(after_dact))));
    local = fmaxf(local, fmaxf(fabsf(pk_lo(after_dgate)), fabsf(pk_hi(after_dgate))));
  }
}

// TE semantics for the FP8 path: each block does an atomic max into the quantizer amax, and
// block 0 writes scale_inv = 1 / scale. The local max is never negative, thus the int order of
// its bits is the float order.
__device__ __forceinline__ void finish_amax_atomic(float local, float *amax_ptr) {
  __shared__ float warp_max[kWarps];
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, offset));
  }
  if (threadIdx.x % 32 == 0) {
    warp_max[threadIdx.x / 32] = local;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_max = warp_max[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
      block_max = fmaxf(block_max, warp_max[w]);
    }
    atomicMax(reinterpret_cast<int *>(amax_ptr), __float_as_int(block_max));
  }
}

// Find the block max. Then find the global max with the scratch value and the block counter.
// The local max is never negative. Thus the int order of its bits is the float order.
__device__ __forceinline__ void finish_amax(float local, float *amax_out, int *slot) {
  int *scratch = slot;
  unsigned int *counter = reinterpret_cast<unsigned int *>(slot + 1);
  __shared__ float warp_max[kWarps];
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, offset));
  }
  if (threadIdx.x % 32 == 0) {
    warp_max[threadIdx.x / 32] = local;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float block_max = warp_max[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
      block_max = fmaxf(block_max, warp_max[w]);
    }
    atomicMax(scratch, __float_as_int(block_max));
    __threadfence();
    const unsigned int ticket = atomicAdd(counter, 1u);
    if (ticket == gridDim.x - 1) {
      __threadfence();
      *amax_out = __int_as_float(atomicExch(scratch, 0));
      atomicExch(counter, 0u);
    }
  }
}

// Compile-time settings of one kernel.
template <int kMath_, int kLoad_, int kStore_, int kWords_, int kUnroll_, int kMinBlocks_>
struct Config {
  static constexpr int kMath = kMath_;
  static constexpr int kLoad = kLoad_;
  static constexpr int kStore = kStore_;
  static constexpr int kWords = kWords_;
  static constexpr int kUnroll = kUnroll_;
  static constexpr int kMinBlocks = kMinBlocks_;  // Minimum blocks per SM, to limit registers.
};

// x: [rows, 2 * vpr] vectors (act, then gate). y: [rows, vpr].
// Each block takes whole rows. Each thread keeps kUnroll vector pairs in flight.
template <typename C>
__global__ void __launch_bounds__(kThreads, C::kMinBlocks)
    swiglu_forward_kernel(const Vec<C::kWords> *__restrict__ x, Vec<C::kWords> *__restrict__ y,
                          const int rows, const int vpr, const float scale,
                          float *__restrict__ amax_out, int *__restrict__ slot) {
  using V = Vec<C::kWords>;
  float local = 0.f;
  for (int r = blockIdx.x; r < rows; r += gridDim.x) {
    const V *act_row = x + static_cast<size_t>(r) * 2 * vpr;
    const V *gate_row = act_row + vpr;
    V *out_row = y + static_cast<size_t>(r) * vpr;
    for (int c = threadIdx.x; c < vpr; c += C::kUnroll * kThreads) {
      V act[C::kUnroll];
      V gate[C::kUnroll];
#pragma unroll
      for (int u = 0; u < C::kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          act[u] = load_vec<C::kLoad>(act_row + cu);
          gate[u] = load_vec<C::kLoad>(gate_row + cu);
        }
      }
#pragma unroll
      for (int u = 0; u < C::kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          if constexpr (C::kMath == kPacked) {
            store_vec<C::kStore>(out_row + cu,
                                 forward_vec_packed(act[u], gate[u], scale, local));
          } else {
            store_vec<C::kStore>(out_row + cu,
                                 forward_vec<C::kMath>(act[u], gate[u], scale, local));
          }
        }
      }
    }
  }
  finish_amax(local, amax_out, slot);
}

// grad: [rows, vpr]. x and dx: [rows, 2 * vpr] (act part, then gate part).
template <typename C>
__global__ void __launch_bounds__(kThreads, C::kMinBlocks)
    swiglu_backward_kernel(const Vec<C::kWords> *__restrict__ grad,
                           const Vec<C::kWords> *__restrict__ x, Vec<C::kWords> *__restrict__ dx,
                           const int rows, const int vpr, const float scale,
                           float *__restrict__ amax_out, int *__restrict__ slot) {
  using V = Vec<C::kWords>;
  float local = 0.f;
  for (int r = blockIdx.x; r < rows; r += gridDim.x) {
    const V *grad_row = grad + static_cast<size_t>(r) * vpr;
    const V *act_row = x + static_cast<size_t>(r) * 2 * vpr;
    const V *gate_row = act_row + vpr;
    V *out_act_row = dx + static_cast<size_t>(r) * 2 * vpr;
    V *out_gate_row = out_act_row + vpr;
    for (int c = threadIdx.x; c < vpr; c += C::kUnroll * kThreads) {
      V g[C::kUnroll];
      V act[C::kUnroll];
      V gate[C::kUnroll];
#pragma unroll
      for (int u = 0; u < C::kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          g[u] = load_vec<C::kLoad>(grad_row + cu);
          act[u] = load_vec<C::kLoad>(act_row + cu);
          gate[u] = load_vec<C::kLoad>(gate_row + cu);
        }
      }
#pragma unroll
      for (int u = 0; u < C::kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          V out_act, out_gate;
          if constexpr (C::kMath == kPacked) {
            backward_vec_packed(g[u], act[u], gate[u], scale, out_act, out_gate, local);
          } else {
            backward_vec<C::kMath>(g[u], act[u], gate[u], scale, out_act, out_gate, local);
          }
          store_vec<C::kStore>(out_act_row + cu, out_act);
          store_vec<C::kStore>(out_gate_row + cu, out_gate);
        }
      }
    }
  }
  finish_amax(local, amax_out, slot);
}

// FP8 forward: x [rows, 2 * vpr] bf16 vectors, y [rows, vpr] FP8 vectors of half the words.
template <typename C, __nv_fp8_interpretation_t kFormat>
__global__ void __launch_bounds__(kThreads, C::kMinBlocks)
    swiglu_forward_fp8_kernel(const Vec<C::kWords> *__restrict__ x,
                              Vec<C::kWords / 2> *__restrict__ y, const int rows, const int vpr,
                              const float *__restrict__ scale_ptr, float *__restrict__ amax_ptr,
                              float *__restrict__ scale_inv_ptr) {
  using V = Vec<C::kWords>;
  using O = Vec<C::kWords / 2>;
  const float scale = *scale_ptr;
  float local = 0.f;
  for (int r = blockIdx.x; r < rows; r += gridDim.x) {
    const V *act_row = x + static_cast<size_t>(r) * 2 * vpr;
    const V *gate_row = act_row + vpr;
    O *out_row = y + static_cast<size_t>(r) * vpr;
    for (int c = threadIdx.x; c < vpr; c += C::kUnroll * kThreads) {
      V act[C::kUnroll];
      V gate[C::kUnroll];
#pragma unroll
      for (int u = 0; u < C::kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          act[u] = load_vec<C::kLoad>(act_row + cu);
          gate[u] = load_vec<C::kLoad>(gate_row + cu);
        }
      }
#pragma unroll
      for (int u = 0; u < C::kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          if constexpr (C::kMath == kPacked) {
            store_vec<C::kStore>(out_row + cu,
                                 forward_vec_fp8_packed<kFormat>(act[u], gate[u], scale, local));
          } else {
            store_vec<C::kStore>(out_row + cu,
                                 forward_vec_fp8_exact<kFormat>(act[u], gate[u], scale, local));
          }
        }
      }
    }
  }
  finish_amax_atomic(local, amax_ptr);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *scale_inv_ptr = __frcp_rn(scale);
  }
}

// FP8 backward: grad [rows, vpr] bf16, x [rows, 2 * vpr] bf16, dx [rows, 2 * vpr] FP8.
template <typename C, __nv_fp8_interpretation_t kFormat>
__global__ void __launch_bounds__(kThreads, C::kMinBlocks)
    swiglu_backward_fp8_kernel(const Vec<C::kWords> *__restrict__ grad,
                               const Vec<C::kWords> *__restrict__ x,
                               Vec<C::kWords / 2> *__restrict__ dx, const int rows, const int vpr,
                               const float *__restrict__ scale_ptr, float *__restrict__ amax_ptr,
                               float *__restrict__ scale_inv_ptr) {
  using V = Vec<C::kWords>;
  using O = Vec<C::kWords / 2>;
  const float scale = *scale_ptr;
  float local = 0.f;
  for (int r = blockIdx.x; r < rows; r += gridDim.x) {
    const V *grad_row = grad + static_cast<size_t>(r) * vpr;
    const V *act_row = x + static_cast<size_t>(r) * 2 * vpr;
    const V *gate_row = act_row + vpr;
    O *out_act_row = dx + static_cast<size_t>(r) * 2 * vpr;
    O *out_gate_row = out_act_row + vpr;
    for (int c = threadIdx.x; c < vpr; c += C::kUnroll * kThreads) {
      V g[C::kUnroll];
      V act[C::kUnroll];
      V gate[C::kUnroll];
#pragma unroll
      for (int u = 0; u < C::kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          g[u] = load_vec<C::kLoad>(grad_row + cu);
          act[u] = load_vec<C::kLoad>(act_row + cu);
          gate[u] = load_vec<C::kLoad>(gate_row + cu);
        }
      }
#pragma unroll
      for (int u = 0; u < C::kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          O out_act, out_gate;
          if constexpr (C::kMath == kPacked) {
            backward_vec_fp8_packed<kFormat>(g[u], act[u], gate[u], scale, out_act, out_gate,
                                             local);
          } else {
            backward_vec_fp8_exact<kFormat>(g[u], act[u], gate[u], scale, out_act, out_gate,
                                            local);
          }
          store_vec<C::kStore>(out_act_row + cu, out_act);
          store_vec<C::kStore>(out_gate_row + cu, out_gate);
        }
      }
    }
  }
  finish_amax_atomic(local, amax_ptr);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *scale_inv_ptr = __frcp_rn(scale);
  }
}

// One scratch slot (scratch value, block counter) for each call site and device.
constexpr int kSlots = 4;
std::map<int, at::Tensor> g_workspace;

int *workspace_slot(const at::Device &device, const int slot) {
  TORCH_CHECK(slot >= 0 && slot < kSlots, "slot out of range");
  const int index = device.index();
  auto it = g_workspace.find(index);
  if (it == g_workspace.end()) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status));
    TORCH_CHECK(status == cudaStreamCaptureStatusNone,
                "call init_workspace() before CUDA graph capture");
    auto ws = at::zeros({2 * kSlots}, at::TensorOptions().dtype(at::kInt).device(device));
    it = g_workspace.emplace(index, ws).first;
  }
  return it->second.data_ptr<int>() + 2 * slot;
}

void init_workspace() {
  const auto device = at::Device(at::kCUDA, c10::cuda::current_device());
  workspace_slot(device, 0);
}

void check_bf16(const at::Tensor &t, const char *name, const int align) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
              ": contiguous bf16 CUDA tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % align == 0, name, ": ", align,
              "-byte aligned");
}

// The grid has as many blocks as can be resident at the same time, but not more than rows.
template <typename Kernel>
int grid_for(Kernel kernel, const int rows) {
  static std::map<std::pair<int, const void *>, int> blocks_per_sm;
  const auto key = std::make_pair(static_cast<int>(c10::cuda::current_device()),
                                  reinterpret_cast<const void *>(kernel));
  auto it = blocks_per_sm.find(key);
  if (it == blocks_per_sm.end()) {
    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0));
    it = blocks_per_sm.emplace(key, std::max(per_sm, 1)).first;
  }
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return std::max(1, std::min(rows, sms * it->second));
}

struct Shape {
  int rows;
  int vpr;  // Vectors in one row of the output (forward) or of grad (backward).
};

template <typename C>
void launch_forward(const at::Tensor &x, at::Tensor &y, at::Tensor &amax, int *slot,
                    const Shape &s) {
  using V = Vec<C::kWords>;
  auto kernel = swiglu_forward_kernel<C>;
  kernel<<<grid_for(kernel, s.rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const V *>(x.data_ptr()), reinterpret_cast<V *>(y.data_ptr()), s.rows,
      s.vpr, 1.f, amax.data_ptr<float>(), slot);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename C>
void launch_backward(const at::Tensor &grad, const at::Tensor &x, at::Tensor &dx,
                     at::Tensor &amax, int *slot, const Shape &s) {
  using V = Vec<C::kWords>;
  auto kernel = swiglu_backward_kernel<C>;
  kernel<<<grid_for(kernel, s.rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const V *>(grad.data_ptr()), reinterpret_cast<const V *>(x.data_ptr()),
      reinterpret_cast<V *>(dx.data_ptr()), s.rows, s.vpr, 1.f, amax.data_ptr<float>(), slot);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The variant table. The same id selects the same settings in forward and backward.
// Variants with 8 words need cols % 16 == 0.
#define SWIGLU_VARIANTS(X)                                  \
  X(0, kExact, kLoadPlain, kStorePlain, 4, 2, 1)            \
  X(1, kExact, kLoadStream, kStoreStream, 4, 1, 1)          \
  X(2, kExact, kLoadStream, kStoreStream, 4, 2, 1)          \
  X(3, kExact, kLoadStream, kStoreStream, 4, 4, 1)          \
  X(4, kExact, kLoadStream, kStorePlain, 4, 4, 1)           \
  X(5, kExact, kLoadPlain, kStorePlain, 4, 4, 1)            \
  X(6, kExact, kLoadStream, kStoreStream, 8, 2, 1)          \
  X(7, kExact, kLoadStream, kStoreStream, 4, 8, 1)          \
  X(8, kExact, kLoadStream, kStorePlain, 8, 2, 1)           \
  X(10, kProbe, kLoadPlain, kStorePlain, 4, 2, 1)           \
  X(11, kProbe, kLoadStream, kStoreStream, 4, 2, 1)         \
  X(12, kProbe, kLoadStream, kStoreStream, 4, 4, 1)         \
  X(13, kProbe, kLoadStream, kStorePlain, 4, 4, 1)          \
  X(14, kProbe, kLoadPlain, kStorePlain, 4, 4, 1)           \
  X(15, kProbe, kLoadStream, kStoreStream, 8, 2, 1)         \
  X(16, kProbe, kLoadStream, kStoreStream, 4, 8, 1)         \
  X(17, kProbe, kLoadStream, kStoreStream, 8, 4, 1)         \
  X(18, kProbe, kLoadStream, kStorePlain, 8, 2, 1)          \
  X(20, kPacked, kLoadStream, kStoreStream, 4, 2, 1)        \
  X(21, kPacked, kLoadStream, kStoreStream, 4, 4, 1)        \
  X(22, kPacked, kLoadStream, kStoreStream, 8, 2, 1)        \
  X(23, kPacked, kLoadStream, kStorePlain, 8, 2, 1)         \
  X(24, kPacked, kLoadStream, kStoreStream, 8, 4, 1)        \
  X(25, kPacked, kLoadPlain, kStorePlain, 4, 4, 1)          \
  X(26, kPacked, kLoadStream, kStoreStream, 4, 1, 1)        \
  X(27, kPacked, kLoadStream, kStoreStream, 8, 4, 2)        \
  X(28, kPacked, kLoadStream, kStoreStream, 8, 3, 1)        \
  X(29, kPacked, kLoadStream, kStorePlain, 8, 4, 1)         \
  X(30, kPacked, kLoadPlain, kStoreStream, 8, 4, 1)         \
  X(31, kPacked, kLoadStream, kStoreStream, 8, 3, 2)        \
  X(32, kPacked, kLoadStream, kStorePlain, 8, 4, 2)         \
  X(33, kPacked, kLoadPlain, kStoreStream, 8, 4, 2)         \
  X(34, kPacked, kLoadStream, kStoreStream, 8, 5, 1)        \
  X(40, kPacked, kLoadStream, kStoreNone, 8, 4, 1)          \
  X(41, kPacked, kLoadStream, kStoreNone, 8, 2, 1)          \
  X(42, kPacked, kLoadStream, kStoreNone, 4, 4, 1)          \
  X(43, kPacked, kLoadStream, kStoreNone, 8, 4, 2)

int words_of(const int64_t variant) {
#define WORDS_CASE(id, math, load, store, words, unroll, minb) \
  case id:                                               \
    return words;
  switch (variant) {
    SWIGLU_VARIANTS(WORDS_CASE)
    default:
      TORCH_CHECK(false, "unknown variant ", variant);
  }
#undef WORDS_CASE
  return 0;
}

std::vector<at::Tensor> forward(const at::Tensor &x, const int64_t variant) {
  const int words = words_of(variant);
  check_bf16(x, "x", 4 * words);
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t cols = x.size(-1) / 2;
  TORCH_CHECK(x.size(-1) % 2 == 0 && cols % (2 * words) == 0,
              "x: last dim / 2 must be a multiple of ", 2 * words);
  const int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(rows < (1LL << 31) && cols < (1LL << 30), "x: too large");
  auto out_size = x.sizes().vec();
  out_size.back() = cols;
  auto y = at::empty(out_size, x.options());
  auto amax = at::empty({1}, x.options().dtype(at::kFloat));
  if (rows == 0) {
    amax.zero_();
    return {y, amax};
  }
  int *slot = workspace_slot(x.device(), 0);
  const Shape s{static_cast<int>(rows), static_cast<int>(cols / (2 * words))};
#define FORWARD_CASE(id, math, load, store, words, unroll, minb)                    \
  case id:                                                                    \
    launch_forward<Config<math, load, store, words, unroll, minb>>(x, y, amax, slot, s); \
    break;
  switch (variant) {
    SWIGLU_VARIANTS(FORWARD_CASE)
    default:
      TORCH_CHECK(false, "unknown variant ", variant);
  }
#undef FORWARD_CASE
  return {y, amax};
}

std::vector<at::Tensor> backward(const at::Tensor &grad, const at::Tensor &x,
                                 const int64_t variant) {
  const int words = words_of(variant);
  check_bf16(grad, "grad", 4 * words);
  check_bf16(x, "x", 4 * words);
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t cols = x.size(-1) / 2;
  TORCH_CHECK(x.size(-1) % 2 == 0 && cols % (2 * words) == 0,
              "x: last dim / 2 must be a multiple of ", 2 * words);
  TORCH_CHECK(grad.size(-1) == cols && grad.numel() * 2 == x.numel(),
              "grad: shape must be x shape with half the last dim");
  const int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(rows < (1LL << 31) && cols < (1LL << 30), "x: too large");
  auto dx = at::empty_like(x);
  auto amax = at::empty({1}, x.options().dtype(at::kFloat));
  if (rows == 0) {
    amax.zero_();
    return {dx, amax};
  }
  int *slot = workspace_slot(x.device(), 1);
  const Shape s{static_cast<int>(rows), static_cast<int>(cols / (2 * words))};
#define BACKWARD_CASE(id, math, load, store, words, unroll, minb)                              \
  case id:                                                                               \
    launch_backward<Config<math, load, store, words, unroll, minb>>(grad, x, dx, amax, slot, s); \
    break;
  switch (variant) {
    SWIGLU_VARIANTS(BACKWARD_CASE)
    default:
      TORCH_CHECK(false, "unknown variant ", variant);
  }
#undef BACKWARD_CASE
  return {dx, amax};
}

// The FP8 variants use the 8-word configs of the table (variant ids with words == 8).
void check_fp8_out(const at::Tensor &t, const at::IntArrayRef size, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kByte && t.is_contiguous() && t.sizes() == size,
              name, ": contiguous uint8 CUDA tensor of the output shape");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
}

void check_scalar(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.numel() == 1, name,
              ": one fp32 CUDA value");
}

template <typename C, __nv_fp8_interpretation_t kFormat>
void launch_forward_fp8(const at::Tensor &x, const at::Tensor &y, const at::Tensor &scale,
                        const at::Tensor &amax, const at::Tensor &scale_inv, const Shape &s) {
  using V = Vec<C::kWords>;
  using O = Vec<C::kWords / 2>;
  auto kernel = swiglu_forward_fp8_kernel<C, kFormat>;
  kernel<<<grid_for(kernel, s.rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const V *>(x.data_ptr()), reinterpret_cast<O *>(y.data_ptr()), s.rows,
      s.vpr, scale.data_ptr<float>(), amax.data_ptr<float>(), scale_inv.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename C, __nv_fp8_interpretation_t kFormat>
void launch_backward_fp8(const at::Tensor &grad, const at::Tensor &x, const at::Tensor &dx,
                         const at::Tensor &scale, const at::Tensor &amax,
                         const at::Tensor &scale_inv, const Shape &s) {
  using V = Vec<C::kWords>;
  using O = Vec<C::kWords / 2>;
  auto kernel = swiglu_backward_fp8_kernel<C, kFormat>;
  kernel<<<grid_for(kernel, s.rows), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const V *>(grad.data_ptr()), reinterpret_cast<const V *>(x.data_ptr()),
      reinterpret_cast<O *>(dx.data_ptr()), s.rows, s.vpr, scale.data_ptr<float>(),
      amax.data_ptr<float>(), scale_inv.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// SwiGLU forward with FP8 output into y (uint8, shape of x with half the last dim).
// Atomic max into amax, and scale_inv = 1 / scale, as the TE delayed-scaling path.
void forward_fp8(const at::Tensor &x, const at::Tensor &scale, const at::Tensor &amax,
                 const at::Tensor &scale_inv, const at::Tensor &y, const bool e5m2,
                 const int64_t variant) {
  const int words = words_of(variant);
  TORCH_CHECK(words == 8, "FP8 variants need an 8-word config");
  check_bf16(x, "x", 4 * words);
  check_scalar(scale, "scale");
  check_scalar(amax, "amax");
  check_scalar(scale_inv, "scale_inv");
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t cols = x.size(-1) / 2;
  TORCH_CHECK(x.size(-1) % 2 == 0 && cols % (2 * words) == 0,
              "x: last dim / 2 must be a multiple of ", 2 * words);
  const int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(rows < (1LL << 31) && cols < (1LL << 30), "x: too large");
  auto out_size = x.sizes().vec();
  out_size.back() = cols;
  check_fp8_out(y, out_size, "y");
  if (rows == 0) {
    return;
  }
  const Shape s{static_cast<int>(rows), static_cast<int>(cols / (2 * words))};
#define FORWARD_FP8_CASE(id, math, load, store, words_, unroll, minb)                          \
  case id:                                                                                    \
    if constexpr (words_ == 8) {                                                              \
      if (e5m2) {                                                                             \
        launch_forward_fp8<Config<math, load, store, words_, unroll, minb>, __NV_E5M2>(      \
            x, y, scale, amax, scale_inv, s);                                                 \
      } else {                                                                                \
        launch_forward_fp8<Config<math, load, store, words_, unroll, minb>, __NV_E4M3>(      \
            x, y, scale, amax, scale_inv, s);                                                 \
      }                                                                                       \
    }                                                                                         \
    break;
  switch (variant) {
    SWIGLU_VARIANTS(FORWARD_FP8_CASE)
    default:
      TORCH_CHECK(false, "unknown variant ", variant);
  }
#undef FORWARD_FP8_CASE
}

// SwiGLU backward with FP8 output into dx (uint8, shape of x).
void backward_fp8(const at::Tensor &grad, const at::Tensor &x, const at::Tensor &scale,
                  const at::Tensor &amax, const at::Tensor &scale_inv, const at::Tensor &dx,
                  const bool e5m2, const int64_t variant) {
  const int words = words_of(variant);
  TORCH_CHECK(words == 8, "FP8 variants need an 8-word config");
  check_bf16(grad, "grad", 4 * words);
  check_bf16(x, "x", 4 * words);
  check_scalar(scale, "scale");
  check_scalar(amax, "amax");
  check_scalar(scale_inv, "scale_inv");
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t cols = x.size(-1) / 2;
  TORCH_CHECK(x.size(-1) % 2 == 0 && cols % (2 * words) == 0,
              "x: last dim / 2 must be a multiple of ", 2 * words);
  TORCH_CHECK(grad.size(-1) == cols && grad.numel() * 2 == x.numel(),
              "grad: shape must be x shape with half the last dim");
  const int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(rows < (1LL << 31) && cols < (1LL << 30), "x: too large");
  check_fp8_out(dx, x.sizes(), "dx");
  if (rows == 0) {
    return;
  }
  const Shape s{static_cast<int>(rows), static_cast<int>(cols / (2 * words))};
#define BACKWARD_FP8_CASE(id, math, load, store, words_, unroll, minb)                         \
  case id:                                                                                    \
    if constexpr (words_ == 8) {                                                              \
      if (e5m2) {                                                                             \
        launch_backward_fp8<Config<math, load, store, words_, unroll, minb>, __NV_E5M2>(     \
            grad, x, dx, scale, amax, scale_inv, s);                                          \
      } else {                                                                                \
        launch_backward_fp8<Config<math, load, store, words_, unroll, minb>, __NV_E4M3>(     \
            grad, x, dx, scale, amax, scale_inv, s);                                          \
      }                                                                                       \
    }                                                                                         \
    break;
  switch (variant) {
    SWIGLU_VARIANTS(BACKWARD_FP8_CASE)
    default:
      TORCH_CHECK(false, "unknown variant ", variant);
  }
#undef BACKWARD_FP8_CASE
}

// Registers and blocks per SM of a kernel variant: [fwd regs, fwd blocks, bwd regs, bwd blocks].
std::vector<int64_t> kernel_info(const int64_t variant) {
  std::vector<int64_t> info;
#define INFO_CASE(id, math, load, store, words, unroll, minb)                                    \
  case id: {                                                                               \
    using C = Config<math, load, store, words, unroll, minb>;                                    \
    cudaFuncAttributes attr;                                                               \
    int blocks = 0;                                                                        \
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, swiglu_forward_kernel<C>));                \
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                          \
        &blocks, swiglu_forward_kernel<C>, kThreads, 0));                                  \
    info.push_back(attr.numRegs);                                                          \
    info.push_back(blocks);                                                                \
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, swiglu_backward_kernel<C>));               \
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                          \
        &blocks, swiglu_backward_kernel<C>, kThreads, 0));                                 \
    info.push_back(attr.numRegs);                                                          \
    info.push_back(blocks);                                                                \
  } break;
  switch (variant) {
    SWIGLU_VARIANTS(INFO_CASE)
    default:
      TORCH_CHECK(false, "unknown variant ", variant);
  }
#undef INFO_CASE
  return info;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_workspace", &init_workspace, "Make the amax scratch of the current device");
  m.def("forward", &forward, "SwiGLU forward: (bf16 y, fp32 amax)");
  m.def("backward", &backward, "SwiGLU backward: (bf16 dx, fp32 amax)");
  m.def("kernel_info", &kernel_info, "Registers and blocks per SM of a variant");
  m.def("forward_fp8", &forward_fp8, "SwiGLU forward with FP8 output (TE delayed scaling)");
  m.def("backward_fp8", &backward_fp8, "SwiGLU backward with FP8 output (TE delayed scaling)");
}
