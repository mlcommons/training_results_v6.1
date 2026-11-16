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

// The TE SwiGLU math and vector helpers of swiglu_fused.cu.
#pragma once
#include <cuda_bf16.h>

namespace swiglu_math {

// Same as TE util/math.h: sigmoid<float, float> and silu<float, float>.
__device__ __forceinline__ float te_sigmoid(const float cval) {
  return 1.f / (1.f + expf(-cval));
}

__device__ __forceinline__ float te_silu(const float cval) { return cval * te_sigmoid(cval); }

// Same as TE util/math.h: sigmoidf.
__device__ __forceinline__ float te_sigmoidf(const float x) {
  return __frcp_rn(1.0f + __expf(-x));
}

// A vector of kWords 32-bit words. Each word holds 2 bf16 values.
template <int kWords>
struct alignas(4 * kWords) Vec {
  uint32_t w[kWords];
};

// Load modes.
constexpr int kLoadPlain = 0;
constexpr int kLoadStream = 1;  // Non-coherent, no L1 allocation, 256-byte L2 prefetch.
// Store modes.
constexpr int kStorePlain = 0;
constexpr int kStoreStream = 1;  // Evict-first (.cs).
constexpr int kStoreNone = 2;  // No store. These variants give only the amax.

template <int kLoad, int kWords>
__device__ __forceinline__ Vec<kWords> load_vec(const Vec<kWords> *ptr) {
  Vec<kWords> v;
  if constexpr (kLoad == kLoadPlain) {
    v = *ptr;
  } else if constexpr (kWords == 4) {
    asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.w[0]), "=r"(v.w[1]), "=r"(v.w[2]), "=r"(v.w[3])
        : "l"(ptr));
  } else {
    static_assert(kWords == 8, "kWords must be 4 or 8");
    asm("ld.global.nc.L1::no_allocate.L2::256B.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(v.w[0]), "=r"(v.w[1]), "=r"(v.w[2]), "=r"(v.w[3]), "=r"(v.w[4]),
          "=r"(v.w[5]), "=r"(v.w[6]), "=r"(v.w[7])
        : "l"(ptr));
  }
  return v;
}

template <int kStore, int kWords>
__device__ __forceinline__ void store_vec(Vec<kWords> *ptr, const Vec<kWords> &v) {
  if constexpr (kStore == kStoreNone) {
    return;
  } else if constexpr (kStore == kStorePlain) {
    *ptr = v;
  } else if constexpr (kWords == 4) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(v.w[0]),
                 "r"(v.w[1]), "r"(v.w[2]), "r"(v.w[3])
                 : "memory");
  } else {
    static_assert(kWords == 8, "kWords must be 4 or 8");
    asm volatile("st.global.cs.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(ptr),
                 "r"(v.w[0]), "r"(v.w[1]), "r"(v.w[2]), "r"(v.w[3]), "r"(v.w[4]), "r"(v.w[5]),
                 "r"(v.w[6]), "r"(v.w[7])
                 : "memory");
  }
}

__device__ __forceinline__ float bf16_lo(const uint32_t word) {
  return __uint_as_float(word << 16);
}

__device__ __forceinline__ float bf16_hi(const uint32_t word) {
  return __uint_as_float(word & 0xffff0000u);
}

__device__ __forceinline__ uint32_t pack_bf16(const float lo, const float hi) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t *>(&pair);
}

// Math modes.
constexpr int kExact = 0;
constexpr int kProbe = 1;  // Trivial math with the same memory traffic, to find the memory floor.
constexpr int kPacked = 2;  // The kExact sequence with two lanes in each f32x2 operation.

// Packed f32x2 helpers. Each lane rounds as the scalar instruction with the same name.
__device__ __forceinline__ uint64_t pk(const float lo, const float hi) {
  uint64_t v;
  asm("mov.b64 %0, {%1, %2};" : "=l"(v) : "f"(lo), "f"(hi));
  return v;
}

__device__ __forceinline__ float pk_lo(const uint64_t v) {
  return __uint_as_float(static_cast<uint32_t>(v));
}

__device__ __forceinline__ float pk_hi(const uint64_t v) {
  return __uint_as_float(static_cast<uint32_t>(v >> 32));
}

__device__ __forceinline__ uint64_t fma2(const uint64_t a, const uint64_t b, const uint64_t c) {
  uint64_t d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
  return d;
}

// PTX has no .sat for fma.f32x2. Thus each lane uses the scalar FFMA.SAT.
__device__ __forceinline__ float fma_sat(const float a, const float b, const float c) {
  float d;
  asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

__device__ __forceinline__ uint64_t fma2_sat(const uint64_t a, const uint64_t b,
                                             const uint64_t c) {
  return pk(fma_sat(pk_lo(a), pk_lo(b), pk_lo(c)), fma_sat(pk_hi(a), pk_hi(b), pk_hi(c)));
}

__device__ __forceinline__ uint64_t fma2_rm(const uint64_t a, const uint64_t b, const uint64_t c) {
  uint64_t d;
  asm("fma.rm.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
  return d;
}

__device__ __forceinline__ uint64_t mul2(const uint64_t a, const uint64_t b) {
  uint64_t d;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}

__device__ __forceinline__ uint64_t add2(const uint64_t a, const uint64_t b) {
  uint64_t d;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}

// MUFU.EX2 and MUFU.RCP with no range handling. The callers use them only in a safe range.
__device__ __forceinline__ float ex2_mufu(const float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float rcp_mufu(const float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint64_t ex2_pair(const uint64_t r) {
  return pk(ex2_mufu(pk_lo(r)), ex2_mufu(pk_hi(r)));
}

__device__ __forceinline__ uint64_t pair_of(const uint32_t word) {
  return pk(bf16_lo(word), bf16_hi(word));
}

// This is the fast path of the IEEE division 1.f / d in the SASS of nvcc 13.4.
// It applies for 1 <= d < 2^126: rr = MUFU.RCP(d), e = fma(d, rr, -1), s = fma(rr, -e, rr).
// The FTZ of -e in the SASS has no effect, because |rr * e| is far below half an ulp of rr.
__device__ __forceinline__ uint64_t reciprocal_pair(const uint64_t d) {
  const uint64_t minus_one = pk(-1.f, -1.f);
  const uint64_t rr = pk(rcp_mufu(pk_lo(d)), rcp_mufu(pk_hi(d)));
  const uint64_t e = fma2(d, rr, minus_one);
  return fma2(rr, mul2(e, minus_one), rr);
}

// 1.f + expf(-x) as in the SASS of nvcc 13.4 (expf with no fast math). The last FMUL of
// expf and the add of 1 are one FFMA, as nvcc contracts them.
__device__ __forceinline__ uint64_t one_plus_expf_neg_pair(const uint64_t x) {
  const float c1 = 0.0057249800302088260651f;
  const uint64_t t = fma2_sat(x, pk(-c1, -c1), pk(0.5f, 0.5f));
  const uint64_t j = fma2_rm(t, pk(252.f, 252.f), pk(12582913.f, 12582913.f));
  const uint64_t minus_jf = fma2(j, pk(-1.f, -1.f), pk(12583039.f, 12583039.f));
  const float l1 = -1.4426950216293334961f;
  const float l2 = -1.925963033500011079e-08f;
  uint64_t r = fma2(x, pk(l1, l1), minus_jf);
  r = fma2(x, pk(l2, l2), r);
  const uint64_t scale = pk(__uint_as_float(__float_as_uint(pk_lo(j)) << 23),
                            __uint_as_float(__float_as_uint(pk_hi(j)) << 23));
  return fma2(scale, ex2_pair(r), pk(1.f, 1.f));
}

constexpr float kTwoPow126 = 8.507059173023462e+37f;

template <int kMath>
__device__ __forceinline__ float forward_value(const float act, const float gate) {
  if constexpr (kMath == kExact) {
    return te_silu(act) * gate;
  } else {
    return act * gate;
  }
}

// TE: out = OType(scale * after_act), amax = max(amax, |after_act|).
template <int kMath, int kWords>
__device__ __forceinline__ Vec<kWords> forward_vec(const Vec<kWords> &act,
                                                   const Vec<kWords> &gate, const float scale,
                                                   float &local) {
  Vec<kWords> out;
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const float v0 = forward_value<kMath>(bf16_lo(act.w[w]), bf16_lo(gate.w[w]));
    const float v1 = forward_value<kMath>(bf16_hi(act.w[w]), bf16_hi(gate.w[w]));
    out.w[w] = pack_bf16(scale * v0, scale * v1);
    local = fmaxf(local, fabsf(v0));
    local = fmaxf(local, fabsf(v1));
  }
  return out;
}

// The kExact forward with f32x2 math. If one d is not in [1, 2^126), the slow path of the
// division is necessary. Then the scalar kExact code does the full vector.
template <int kWords>
__device__ __forceinline__ Vec<kWords> forward_vec_packed(const Vec<kWords> &act,
                                                          const Vec<kWords> &gate,
                                                          const float scale, float &local) {
  uint64_t d[kWords];
  float d_max = 0.f;
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    d[w] = one_plus_expf_neg_pair(pair_of(act.w[w]));
    d_max = fmaxf(d_max, fmaxf(pk_lo(d[w]), pk_hi(d[w])));
  }
  if (!(d_max < kTwoPow126)) {
    return forward_vec<kExact>(act, gate, scale, local);
  }
  Vec<kWords> out;
  const uint64_t scale2 = pk(scale, scale);
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const uint64_t silu = mul2(pair_of(act.w[w]), reciprocal_pair(d[w]));
    const uint64_t v = mul2(silu, pair_of(gate.w[w]));
    const uint64_t o = mul2(scale2, v);
    out.w[w] = pack_bf16(pk_lo(o), pk_hi(o));
    local = fmaxf(local, fmaxf(fabsf(pk_lo(v)), fabsf(pk_hi(v))));
  }
  return out;
}

}  // namespace swiglu_math
