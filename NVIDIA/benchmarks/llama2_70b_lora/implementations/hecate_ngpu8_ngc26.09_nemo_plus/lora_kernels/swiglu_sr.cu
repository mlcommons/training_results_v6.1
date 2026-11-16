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

// Two-pass SwiGLU backward with the NVFP4 quantize of the gradient (TE 5b28d7aa math).
// TE runs the dSwiGLU kernel (bf16 dx and amax), then quantizes dx. This code does not write
// the bf16 dx:
//   Pass 1: the dSwiGLU values, and only max |value| (fp32, before the cast to bf16, as TE).
//           Each block writes its max to an empty buffer, so no memset is necessary.
//   Pass 2: the dSwiGLU values again, the cast to bf16, and the NVFP4 quantize with the global
//           amax. The kernel writes the FP4 data and the scales in the GEMM swizzled layout.
// Pass 2 goes from the end of the tensor to the start. Thus it reads from L2 the data that
// pass 1 read last.
// Sources of the copied device code:
//   - dSwiGLU math (f32x2 copy of the nvcc 13.4 SASS of the TE kernel): swiglu_fused.cu.
//   - Scale math, Philox mapping, FP4 conversion and swizzled scale index: nvfp4q_math.cuh.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPhiloxRounds = 10;        // TE default of NVTE_BUILD_NUM_PHILOX_ROUNDS.
constexpr int kRngEltsPerThread = 1024;  // TE quantizer.cpp: rng_elts_per_thread.
constexpr float kFp4Max = 6.0f;
constexpr float kFp4MaxInv = 1.0f / kFp4Max;
constexpr float kFp8Max = 448.0f;

// =============================================================================================
// dSwiGLU math. Copied from swiglu_fused.cu.

// Same as TE util/math.h: sigmoidf.
__device__ __forceinline__ float te_sigmoidf(const float x) {
  return __frcp_rn(1.0f + __expf(-x));
}

// A vector of kWords 32-bit words. Each word holds 2 bf16 values.
template <int kWords>
struct alignas(4 * kWords) Vec {
  uint32_t w[kWords];
};

// Non-coherent 256-bit load with no L1 allocation and a 256-byte L2 prefetch.
__device__ __forceinline__ Vec<8> load_vec8(const Vec<8> *ptr) {
  Vec<8> v;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(v.w[0]), "=r"(v.w[1]), "=r"(v.w[2]), "=r"(v.w[3]), "=r"(v.w[4]), "=r"(v.w[5]),
        "=r"(v.w[6]), "=r"(v.w[7])
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ Vec<4> load_vec4(const Vec<4> *ptr) {
  Vec<4> v;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.w[0]), "=r"(v.w[1]), "=r"(v.w[2]), "=r"(v.w[3])
      : "l"(ptr));
  return v;
}

template <int kWords>
__device__ __forceinline__ Vec<kWords> load_vec(const Vec<kWords> *ptr) {
  if constexpr (kWords == 8) {
    return load_vec8(ptr);
  } else {
    static_assert(kWords == 4, "kWords must be 4 or 8");
    return load_vec4(ptr);
  }
}

// The low bf16 of the word as fp32: the word shifted left by 16 bits. PRMT (ALU pipe) makes it,
// because nvcc makes the shift with IMAD (FMA pipe), and the FMA pipe is the busiest pipe.
__device__ __forceinline__ float bf16_lo(const uint32_t word) {
  return __uint_as_float(__byte_perm(word, 0u, 0x1044u));
}

__device__ __forceinline__ float bf16_hi(const uint32_t word) {
  return __uint_as_float(word & 0xffff0000u);
}

__device__ __forceinline__ uint32_t pack_bf16(const float lo, const float hi) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  uint32_t word;
  memcpy(&word, &pair, sizeof(word));
  return word;
}

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

// One element, as the TE silu/dsilu branch of cast_fp8_gated_kernel<IS_BWD = true>.
__device__ __forceinline__ void backward_value(const float act_elt, const float gate_elt,
                                               const float grad_elt, float &after_dact,
                                               float &after_dgate) {
  const float dgate_elt = 1.0f;
  const float x = act_elt;
  const float s = te_sigmoidf(x);
  const float act_x = x * s;
  const float dact_x = x * s * (1 - s) + s;
  after_dact = dact_x * grad_elt * gate_elt;
  after_dgate = dgate_elt * act_x * grad_elt;
}

// TE: out = OType(scale * value), amax = max(amax, |value|), with scale = 1 for bf16.
template <int kWords>
__device__ __forceinline__ void backward_vec_scalar(const Vec<kWords> &grad,
                                                    const Vec<kWords> &act,
                                                    const Vec<kWords> &gate, const float scale,
                                                    Vec<kWords> &out_act, Vec<kWords> &out_gate,
                                                    float &local) {
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    float da0, dg0, da1, dg1;
    backward_value(bf16_lo(act.w[w]), bf16_lo(gate.w[w]), bf16_lo(grad.w[w]), da0, dg0);
    backward_value(bf16_hi(act.w[w]), bf16_hi(gate.w[w]), bf16_hi(grad.w[w]), da1, dg1);
    out_act.w[w] = pack_bf16(scale * da0, scale * da1);
    out_gate.w[w] = pack_bf16(scale * dg0, scale * dg1);
    local = fmaxf(local, fabsf(da0));
    local = fmaxf(local, fabsf(dg0));
    local = fmaxf(local, fabsf(da1));
    local = fmaxf(local, fabsf(dg1));
  }
}

// The same math with f32x2 operations. __expf(-x) is ex2(-x * l) with the float l below.
// The f32x2 code gives the scalar result only if two conditions are true:
// - Each ex2 input r is at or above -126. Then the scalar code has no ex2 range step.
// - Each d = ex2(r) + 1 is below 2^126. Then __frcp_rn has no slow path.
// Both conditions are true if |r| <= 125. Else the scalar code does the full vector.
// With kScale false, the code does not multiply by scale.
// TE uses scale = 1 for the bf16 dx of its fused NVFP4 path.
// The product 1 * v has the bits of v, because NaNs from arithmetic are canonical.
template <int kWords, bool kScale = true>
__device__ __forceinline__ void backward_vec(const Vec<kWords> &grad, const Vec<kWords> &act,
                                             const Vec<kWords> &gate, const float scale,
                                             Vec<kWords> &out_act, Vec<kWords> &out_gate,
                                             float &local) {
  const float l = -1.4426950216293334961f;
  uint64_t d[kWords];
  float r_abs_max = 0.f;
#pragma unroll
  for (int w = 0; w < kWords; ++w) {
    const uint64_t r = mul2(pair_of(act.w[w]), pk(l, l));
    r_abs_max = fmaxf(r_abs_max, fmaxf(fabsf(pk_lo(r)), fabsf(pk_hi(r))));
    d[w] = add2(ex2_pair(r), pk(1.f, 1.f));
  }
  if (!(r_abs_max <= 125.f)) {
    backward_vec_scalar(grad, act, gate, scale, out_act, out_gate, local);
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
    const uint64_t oa = kScale ? mul2(scale2, after_dact) : after_dact;
    const uint64_t og = kScale ? mul2(scale2, after_dgate) : after_dgate;
    out_act.w[w] = pack_bf16(pk_lo(oa), pk_hi(oa));
    out_gate.w[w] = pack_bf16(pk_lo(og), pk_hi(og));
    local = fmaxf(local, fmaxf(fabsf(pk_lo(after_dact)), fabsf(pk_hi(after_dact))));
    local = fmaxf(local, fmaxf(fabsf(pk_lo(after_dgate)), fabsf(pk_hi(after_dgate))));
  }
}

// =============================================================================================
// NVFP4 quantize math. Copied from nvfp4q_math.cuh.

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
  uint32_t sequence;    // TE rng_sequence of the thread that quantizes the block.
  uint32_t index;       // Number of the generate4() call in that thread.
  uint32_t bank_group;  // 1 if that thread quantizes elements 8..15 first.
};

// The block is block_in_unit (0..15) of the 256-element unit ucol of the row.
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

// The same conversions with the fp32 values of the 8 bf16 inputs as 4 pairs. The TE PTX makes
// these fp32 values with cvt.f32.bf16, which is exact. bf16_lo and bf16_hi give the same bits.
__device__ __forceinline__ uint32_t mul_cvt_rn_8x_pairs(uint64_t v01, uint64_t v23,
                                                        uint64_t v45, uint64_t v67,
                                                        const float scaling_coefficient) {
  uint32_t out_8x = 0;
  asm volatile(
      "{\n"
      ".reg.b64 scaling_coeff_2x; \n\t"
      "mov.b64 scaling_coeff_2x, {%5, %5}; \n\t"
      ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
      "mul.f32x2 %1, %1, scaling_coeff_2x; \n\t"
      "mul.f32x2 %2, %2, scaling_coeff_2x; \n\t"
      "mul.f32x2 %3, %3, scaling_coeff_2x; \n\t"
      "mul.f32x2 %4, %4, scaling_coeff_2x; \n\t"
      "mov.b64 {v1, v0}, %1; \n\t"
      "mov.b64 {v3, v2}, %2; \n\t"
      "mov.b64 {v5, v4}, %3; \n\t"
      "mov.b64 {v7, v6}, %4; \n\t"
      ".reg.b8 f0, f1, f2, f3; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f2, v4, v5;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f3, v6, v7;\n\t"
      "mov.b32 %0, {f0, f1, f2, f3};\n\t"
      "}"
      : "=r"(out_8x), "+l"(v01), "+l"(v23), "+l"(v45), "+l"(v67)
      : "f"(scaling_coefficient));
  return out_8x;
}

__device__ __forceinline__ uint32_t mul_cvt_sr_8x_pairs(uint64_t v01, uint64_t v23,
                                                        uint64_t v45, uint64_t v67,
                                                        const float scaling_coefficient,
                                                        const uint32_t rbits03,
                                                        const uint32_t rbits47) {
  uint32_t out_8x = 0;
  asm volatile(
      "{\n"
      ".reg.b64 scaling_coeff_2x; \n\t"
      "mov.b64 scaling_coeff_2x, {%5, %5}; \n\t"
      ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
      "mul.f32x2 %1, %1, scaling_coeff_2x; \n\t"
      "mul.f32x2 %2, %2, scaling_coeff_2x; \n\t"
      "mul.f32x2 %3, %3, scaling_coeff_2x; \n\t"
      "mul.f32x2 %4, %4, scaling_coeff_2x; \n\t"
      "mov.b64 {v0, v1}, %1; \n\t"
      "mov.b64 {v2, v3}, %2; \n\t"
      "mov.b64 {v4, v5}, %3; \n\t"
      "mov.b64 {v6, v7}, %4; \n\t"
      ".reg.b16 b03, b47; \n\t"
      "cvt.rs.satfinite.e2m1x4.f32 b03, {v3, v2, v1, v0}, %6; \n\t"
      "cvt.rs.satfinite.e2m1x4.f32 b47, {v7, v6, v5, v4}, %7; \n\t"
      "mov.b32 %0, {b03, b47};\n"
      "}"
      : "=r"(out_8x), "+l"(v01), "+l"(v23), "+l"(v45), "+l"(v67)
      : "f"(scaling_coefficient), "r"(rbits03), "r"(rbits47));
  return out_8x;
}

// TE mxfp8::swizzle::gemm_swizzled_scale_idx: tiles of 128 rows x 4 scales, 512 bytes each.
__device__ __forceinline__ uint32_t swizzled_scale_index(uint32_t row, uint32_t col,
                                                         uint32_t num_tiles_x) {
  const uint32_t tile = (row >> 7) * num_tiles_x + (col >> 2);
  const uint32_t row_in_tile = row & 127;
  return tile * 512 + (row_in_tile & 31) * 16 + (row_in_tile >> 5) * 4 + (col & 3);
}

// Max over the block. All threads must call it. All threads get the result.
__device__ __forceinline__ float block_max(float value, float *shared) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

// =============================================================================================
// Pass 1: max |dSwiGLU value| over the fp32 values. Block b writes its max to partial[b].
// grad: [rows, vpr] vectors. x: [rows, 2 * vpr] vectors (act part, then gate part).
template <int kWords, int kUnroll, int kMinBlocks>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    amax_pass_kernel(const Vec<kWords> *__restrict__ grad, const Vec<kWords> *__restrict__ x,
                     const int rows, const int vpr, float *__restrict__ partial) {
  using V = Vec<kWords>;
  __shared__ float shared[kWarps];
  float local = 0.f;
  for (int r = blockIdx.x; r < rows; r += gridDim.x) {
    const V *grad_row = grad + static_cast<size_t>(r) * vpr;
    const V *act_row = x + static_cast<size_t>(r) * 2 * vpr;
    const V *gate_row = act_row + vpr;
    for (int c = threadIdx.x; c < vpr; c += kUnroll * kThreads) {
      V g[kUnroll];
      V act[kUnroll];
      V gate[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          g[u] = load_vec<kWords>(grad_row + cu);
          act[u] = load_vec<kWords>(act_row + cu);
          gate[u] = load_vec<kWords>(gate_row + cu);
        }
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int cu = c + u * kThreads;
        if (cu < vpr) {
          V out_act, out_gate;
          backward_vec(g[u], act[u], gate[u], 1.f, out_act, out_gate, local);
        }
      }
    }
  }
  const float block_amax = block_max(local, shared);
  if (threadIdx.x == 0) partial[blockIdx.x] = block_amax;
}

// =============================================================================================
// Pass 2: dSwiGLU again, then the NVFP4 quantize.
struct QuantizeConstants {
  float s_enc_times_fp4_inv;
  float s_dec;
  uint32_t num_tiles_x;  // Swizzled scale tiles in one row of scales.
  uint32_t cta_cols;     // TE quantize CTAs in one row (128 columns each).
  uint32_t out_cols;     // Columns of dx: 2 * cols.
};

struct QuantizeArgs {
  const Vec<8> *grad;    // [rows, cols] bf16, 16 elements per vector.
  const Vec<8> *x;       // [rows, 2 * cols] bf16 (act part, then gate part).
  uint2 *data;           // FP4 output of dx, 16 elements per uint2, row-major like TE.
  uint8_t *scales;       // E4M3 block scales in the GEMM swizzled layout.
  const float *partial;  // Pass 1 block maxima. The global amax is their max.
  int n_partial;
  float *amax_out;       // Gets the global amax.
  uint32_t rows;
  uint32_t cols;         // Columns of grad (half of the columns of x and dx).
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

// Quantize one block of 16 dx values at (row, col) of dx. The 8 words hold bf16 pairs.
// The RNG position p of the block applies only with SR.
// With a coefficient table (coefficients != nullptr), the coefficient of each E4M3 scale comes
// from the table. The table has the block_coefficient value of all 256 E4M3 bit patterns.
template <bool SR>
__device__ __forceinline__ void quantize_block_at(const Vec<8> &v, const uint32_t row,
                                                  const uint32_t col, const RngPosition &p,
                                                  const QuantizeConstants &k, const uint64_t seed,
                                                  const uint64_t offset, uint2 *__restrict__ data,
                                                  uint8_t *__restrict__ scales,
                                                  const float *coefficients = nullptr) {
  uint32_t acc = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) acc = abs_max_2x(acc, v.w[w]);
  const __nv_fp8_e4m3 s_dec_block = block_decode_scale(amax_of_pair(acc), k.s_enc_times_fp4_inv);
  const float coefficient = coefficients != nullptr ? coefficients[s_dec_block.__x]
                                                    : block_coefficient(s_dec_block, k.s_dec);
  scales[swizzled_scale_index(row, col >> 4, k.num_tiles_x)] = s_dec_block.__x;
  uint64_t f[8];  // fp32 pairs of the 16 bf16 values.
#pragma unroll
  for (int w = 0; w < 8; ++w) f[w] = pair_of(v.w[w]);
  uint2 out;
  if constexpr (SR) {
    const uint4 r = philox_generate(seed, p.sequence, offset, p.index);
    // Half h of the block uses the words (x, y) if h ^ bank_group == 0, else (z, w).
    const bool first_xy = p.bank_group == 0;
    out.x = mul_cvt_sr_8x_pairs(f[0], f[1], f[2], f[3], coefficient, first_xy ? r.x : r.z,
                                first_xy ? r.y : r.w);
    out.y = mul_cvt_sr_8x_pairs(f[4], f[5], f[6], f[7], coefficient, first_xy ? r.z : r.x,
                                first_xy ? r.w : r.y);
  } else {
    out.x = mul_cvt_rn_8x_pairs(f[0], f[1], f[2], f[3], coefficient);
    out.y = mul_cvt_rn_8x_pairs(f[4], f[5], f[6], f[7], coefficient);
  }
  data[(row * k.out_cols + col) >> 4] = out;
}

template <bool SR>
__device__ __forceinline__ void quantize_block(const Vec<8> &v, const uint32_t row,
                                               const uint32_t col, const QuantizeConstants &k,
                                               const uint64_t seed, const uint64_t offset,
                                               uint2 *__restrict__ data,
                                               uint8_t *__restrict__ scales) {
  RngPosition p{};
  if constexpr (SR) p = te_rng_position(row, col >> 8, (col >> 4) & 15, k.cta_cols);
  quantize_block_at<SR>(v, row, col, p, k, seed, offset, data, scales);
}

// A segment is 512 columns of the act part of one row: lane l takes the 16 columns 16l..16l+15.
// For each column, the lane makes the dact block and the dgate block of dx, then quantizes
// both. The warps take kSegs segments in each step, from the end of the tensor to the start.
template <bool SR, int kSegs, int kMinBlocks>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    quantize_pass_kernel(const QuantizeArgs args) {
  __shared__ float shared[kWarps];
  float amax = 0.0f;
  for (int b = threadIdx.x; b < args.n_partial; b += kThreads) amax = fmaxf(amax, args.partial[b]);
  amax = block_max(amax, shared);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    args.amax_out[0] = amax;
    write_gemm_alpha(args.alpha_weight_amax, args.alpha_out, amax);
  }
  QuantizeConstants k;
  const float s_enc = global_encode_scale(amax);
  k.s_enc_times_fp4_inv = s_enc * kFp4MaxInv;
  k.s_dec = 1.0f / s_enc;
  k.out_cols = 2 * args.cols;
  k.num_tiles_x = k.out_cols >> 6;
  k.cta_cols = k.out_cols >> 7;
  uint64_t seed = 0;
  uint64_t offset = 0;
  if constexpr (SR) unpack_philox(args.philox, seed, offset);

  const uint32_t lane = threadIdx.x & 31;
  const uint32_t vph = args.cols >> 4;            // Vectors in the act part of one row.
  const uint32_t segs_per_row = (vph + 31) >> 5;  // 32 vectors (512 columns) per segment.
  const uint32_t n_segs = args.rows * segs_per_row;
  const uint32_t warp = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const uint32_t n_warps = gridDim.x * kWarps;
  for (uint32_t first = warp * kSegs; first < n_segs; first += n_warps * kSegs) {
    Vec<8> g[kSegs];
    Vec<8> act[kSegs];
    Vec<8> gate[kSegs];
    uint32_t row[kSegs];
    uint32_t vec[kSegs];
    bool valid[kSegs];
#pragma unroll
    for (int s = 0; s < kSegs; ++s) {
      const uint32_t index = first + s;
      const uint32_t seg = n_segs - 1 - index;
      row[s] = seg / segs_per_row;
      vec[s] = (seg - row[s] * segs_per_row) * 32 + lane;
      valid[s] = index < n_segs && vec[s] < vph;
      if (valid[s]) {
        g[s] = load_vec8(args.grad + row[s] * vph + vec[s]);
        act[s] = load_vec8(args.x + row[s] * 2 * vph + vec[s]);
        gate[s] = load_vec8(args.x + row[s] * 2 * vph + vph + vec[s]);
      }
    }
#pragma unroll
    for (int s = 0; s < kSegs; ++s) {
      if (valid[s]) {
        Vec<8> out_act;
        Vec<8> out_gate;
        float unused = 0.f;
        backward_vec(g[s], act[s], gate[s], 1.f, out_act, out_gate, unused);
        const uint32_t col = vec[s] * 16;
        quantize_block<SR>(out_act, row[s], col, k, seed, offset, args.data, args.scales);
        quantize_block<SR>(out_gate, row[s], args.cols + col, k, seed, offset, args.data,
                           args.scales);
      }
    }
  }
}

// =============================================================================================
// Pass 2 with a TMA pipeline. Each warp has kStages shared-memory stages. Lane 0 starts bulk
// copies (cp.async.bulk) of the next segments, and an mbarrier tells when a stage is full.
// Thus the warp computes one segment while the next segments come from memory, and the data in
// flight does not use registers. The row and the segment column of the next segment come from
// fixed steps, so the loop has no integer division.
constexpr uint32_t kSegVecs = 32;                // Vectors (16 elements) in one segment.
constexpr uint32_t kSegBytes = kSegVecs * 32;    // Bytes of one input of one segment.
constexpr uint32_t kStageBytes = 3 * kSegBytes;  // grad, act and gate parts.

__device__ __forceinline__ uint32_t smem_address(const void *pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrier_init(uint64_t *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_address(bar)), "r"(count)
               : "memory");
}

// Order the generic-proxy accesses to shared memory before the next async-proxy accesses.
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t *bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_address(bar)),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void bulk_load(void *dst, const void *src, uint32_t bytes,
                                          uint64_t *bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      ::"r"(smem_address(dst)), "l"(src), "r"(bytes), "r"(smem_address(bar))
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t *bar, uint32_t parity) {
  const uint32_t address = smem_address(bar);
  uint32_t done = 0;
  while (done == 0) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(done)
        : "r"(address), "r"(parity)
        : "memory");
  }
}

// Row and segment column of a segment. The segments go from the end to the start: the
// segment number is n_segs - 1 - index. A negative row shows the end of the work.
struct SegmentCursor {
  int32_t row;
  int32_t column;
};

struct CursorStep {
  int32_t rows;          // index step / segments per row
  int32_t columns;       // index step % segments per row
  int32_t segs_per_row;
};

__device__ __forceinline__ SegmentCursor first_segment(uint32_t index, uint32_t n_segs,
                                                       uint32_t segs_per_row) {
  SegmentCursor cursor{-1, 0};
  if (index < n_segs) {
    const uint32_t seg = n_segs - 1 - index;
    cursor.row = static_cast<int32_t>(seg / segs_per_row);
    cursor.column = static_cast<int32_t>(seg - static_cast<uint32_t>(cursor.row) * segs_per_row);
  }
  return cursor;
}

__device__ __forceinline__ void next_segment(SegmentCursor &cursor, const CursorStep &step) {
  cursor.row -= step.rows;
  cursor.column -= step.columns;
  if (cursor.column < 0) {
    cursor.column += step.segs_per_row;
    cursor.row -= 1;
  }
}

// Lane 0 starts the three bulk copies of one segment into one stage.
__device__ __forceinline__ void start_segment(const QuantizeArgs &args, const SegmentCursor &cursor,
                                              uint32_t vph, uint8_t *stage, uint64_t *bar) {
  if (cursor.row < 0) return;
  const uint32_t row = static_cast<uint32_t>(cursor.row);
  const uint32_t vec0 = static_cast<uint32_t>(cursor.column) * kSegVecs;
  const uint32_t bytes = min(kSegVecs, vph - vec0) * 32;
  mbarrier_expect_tx(bar, 3 * bytes);
  bulk_load(stage, args.grad + row * vph + vec0, bytes, bar);
  bulk_load(stage + kSegBytes, args.x + row * 2 * vph + vec0, bytes, bar);
  bulk_load(stage + 2 * kSegBytes, args.x + row * 2 * vph + vph + vec0, bytes, bar);
}

template <bool SR, int kStages, int kMinBlocks>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    quantize_pass_tma_kernel(const QuantizeArgs args, const CursorStep step) {
  extern __shared__ __align__(128) uint8_t stage_memory[];
  __shared__ uint64_t bars[kWarps][kStages];
  __shared__ float shared[kWarps];
  __shared__ float coefficients[256];
  float amax = 0.0f;
  for (int b = threadIdx.x; b < args.n_partial; b += kThreads) amax = fmaxf(amax, args.partial[b]);
  amax = block_max(amax, shared);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    args.amax_out[0] = amax;
    write_gemm_alpha(args.alpha_weight_amax, args.alpha_out, amax);
  }
  QuantizeConstants k;
  const float s_enc = global_encode_scale(amax);
  k.s_enc_times_fp4_inv = s_enc * kFp4MaxInv;
  k.s_dec = 1.0f / s_enc;
  k.out_cols = 2 * args.cols;
  k.num_tiles_x = k.out_cols >> 6;
  k.cta_cols = k.out_cols >> 7;
  static_assert(kThreads == 256, "one coefficient for each thread");
  __nv_fp8_e4m3 table_scale;
  table_scale.__x = static_cast<__nv_fp8_storage_t>(threadIdx.x);
  coefficients[threadIdx.x] = block_coefficient(table_scale, k.s_dec);
  __syncthreads();
  uint64_t seed = 0;
  uint64_t offset = 0;
  if constexpr (SR) unpack_philox(args.philox, seed, offset);

  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp_in_block = threadIdx.x >> 5;
  const uint32_t vph = args.cols >> 4;
  const uint32_t segs_per_row = static_cast<uint32_t>(step.segs_per_row);
  const uint32_t n_segs = args.rows * segs_per_row;
  uint8_t *stages = stage_memory + warp_in_block * (kStages * kStageBytes);
  uint64_t *warp_bars = bars[warp_in_block];
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) mbarrier_init(&warp_bars[s], 1);
    fence_proxy_async();
  }
  __syncwarp();

  const uint32_t warp = blockIdx.x * kWarps + warp_in_block;
  SegmentCursor ahead = first_segment(warp, n_segs, segs_per_row);
  SegmentCursor current = ahead;
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      start_segment(args, ahead, vph, stages + s * kStageBytes, &warp_bars[s]);
      next_segment(ahead, step);
    }
  } else {
#pragma unroll
    for (int s = 0; s < kStages; ++s) next_segment(ahead, step);
  }
  for (uint32_t n = 0; current.row >= 0; ++n) {
    const uint32_t s = n % kStages;
    mbarrier_wait(&warp_bars[s], (n / kStages) & 1);
    const uint32_t row = static_cast<uint32_t>(current.row);
    const uint32_t vec = static_cast<uint32_t>(current.column) * kSegVecs + lane;
    if (vec < vph) {
      const Vec<8> *stage = reinterpret_cast<const Vec<8> *>(stages + s * kStageBytes);
      const Vec<8> g = stage[lane];
      const Vec<8> act = stage[kSegVecs + lane];
      const Vec<8> gate = stage[2 * kSegVecs + lane];
      Vec<8> out_act;
      Vec<8> out_gate;
      float unused = 0.f;
      backward_vec<8, false>(g, act, gate, 1.f, out_act, out_gate, unused);
      const uint32_t col = vec * 16;
      RngPosition p{};
      if constexpr (SR) p = te_rng_position(row, col >> 8, (col >> 4) & 15, k.cta_cols);
      quantize_block_at<SR>(out_act, row, col, p, k, seed, offset, args.data, args.scales,
                            coefficients);
      // The gate block is cols columns to the right. It has the same index and bank group.
      // Its TE CTA is cols / 128 CTAs to the right, thus its sequence is larger by cols.
      p.sequence += args.cols;
      quantize_block_at<SR>(out_gate, row, args.cols + col, p, k, seed, offset, args.data,
                            args.scales, coefficients);
    }
    __syncwarp();
    if (lane == 0) {
      fence_proxy_async();
      start_segment(args, ahead, vph, stages + s * kStageBytes, &warp_bars[s]);
    }
    next_segment(ahead, step);
    next_segment(current, step);
  }
}

// =============================================================================================
// Host side.

int sm_count() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

// Blocks per SM of a kernel, cached for each device and kernel.
template <typename Kernel>
int blocks_per_sm(Kernel kernel) {
  static std::map<std::pair<int, const void *>, int> cache;
  const auto key = std::make_pair(static_cast<int>(c10::cuda::current_device()),
                                  reinterpret_cast<const void *>(kernel));
  auto it = cache.find(key);
  if (it == cache.end()) {
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kThreads, 0));
    it = cache.emplace(key, std::max(blocks, 1)).first;
  }
  return it->second;
}

struct PassOneConfig {
  int words;
  int unroll;
  int min_blocks;
};

// Pass 1 variants: (words, unroll, min blocks per SM).
constexpr PassOneConfig kPassOne[] = {{8, 2, 1}, {4, 4, 1}, {8, 3, 2}, {8, 4, 2}};
constexpr int kNumPassOne = sizeof(kPassOne) / sizeof(kPassOne[0]);

template <int kWords, int kUnroll, int kMinBlocks>
int launch_amax_pass(const at::Tensor &grad, const at::Tensor &x, int rows, int cols,
                     at::Tensor &partial) {
  using V = Vec<kWords>;
  auto kernel = amax_pass_kernel<kWords, kUnroll, kMinBlocks>;
  const int grid = std::max(1, std::min(rows, sm_count() * blocks_per_sm(kernel)));
  partial = at::empty({grid}, x.options().dtype(at::kFloat));
  const int vpr = cols / (2 * kWords);
  kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const V *>(grad.data_ptr()), reinterpret_cast<const V *>(x.data_ptr()),
      rows, vpr, partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grid;
}

int amax_pass(const at::Tensor &grad, const at::Tensor &x, int rows, int cols, int variant,
              at::Tensor &partial) {
  switch (variant) {
    case 0: return launch_amax_pass<8, 2, 1>(grad, x, rows, cols, partial);
    case 1: return launch_amax_pass<4, 4, 1>(grad, x, rows, cols, partial);
    case 2: return launch_amax_pass<8, 3, 2>(grad, x, rows, cols, partial);
    case 3: return launch_amax_pass<8, 4, 2>(grad, x, rows, cols, partial);
    default: TORCH_CHECK(false, "unknown pass 1 variant ", variant);
  }
  return 0;
}

template <bool SR, int kSegs, int kMinBlocks>
void launch_quantize_pass(const QuantizeArgs &args) {
  auto kernel = quantize_pass_kernel<SR, kSegs, kMinBlocks>;
  const int64_t segs = static_cast<int64_t>(args.rows) * (((args.cols >> 4) + 31) >> 5);
  const int64_t needed = (segs + kWarps * kSegs - 1) / (kWarps * kSegs);
  const int64_t limit = static_cast<int64_t>(sm_count()) * blocks_per_sm(kernel);
  const int grid = static_cast<int>(std::max<int64_t>(1, std::min(needed, limit)));
  kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool SR, int kStages, int kMinBlocks>
int launch_quantize_tma(const QuantizeArgs &args) {
  auto kernel = quantize_pass_tma_kernel<SR, kStages, kMinBlocks>;
  const int smem = kStages * static_cast<int>(kStageBytes) * kWarps;
  static std::map<int, int> blocks_for_device;
  const int device = static_cast<int>(c10::cuda::current_device());
  auto it = blocks_for_device.find(device);
  if (it == blocks_for_device.end()) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kThreads, smem));
    it = blocks_for_device.emplace(device, std::max(blocks, 1)).first;
  }
  const int64_t segs_per_row = ((args.cols >> 4) + kSegVecs - 1) / kSegVecs;
  const int64_t segs = static_cast<int64_t>(args.rows) * segs_per_row;
  const int64_t needed = (segs + kWarps - 1) / kWarps;
  const int64_t limit = static_cast<int64_t>(sm_count()) * it->second;
  const int grid = static_cast<int>(std::max<int64_t>(1, std::min(needed, limit)));
  const int64_t n_warps = static_cast<int64_t>(grid) * kWarps;
  CursorStep step;
  step.rows = static_cast<int32_t>(n_warps / segs_per_row);
  step.columns = static_cast<int32_t>(n_warps % segs_per_row);
  step.segs_per_row = static_cast<int32_t>(segs_per_row);
  kernel<<<grid, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(args, step);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grid;
}

// Pass 2 variants. 0-3: direct loads (segments per warp step, min blocks per SM).
// 4-8: TMA pipeline (stages, min blocks per SM): (2, 2), (3, 2), (4, 2), (2, 3), (3, 3).
template <bool SR>
int quantize_pass(const QuantizeArgs &args, int variant) {
  switch (variant) {
    case 0: launch_quantize_pass<SR, 1, 1>(args); return 0;
    case 1: launch_quantize_pass<SR, 2, 1>(args); return 0;
    case 2: launch_quantize_pass<SR, 1, 2>(args); return 0;
    case 3: launch_quantize_pass<SR, 2, 2>(args); return 0;
    case 4: return launch_quantize_tma<SR, 2, 2>(args);
    case 5: return launch_quantize_tma<SR, 3, 2>(args);
    case 6: return launch_quantize_tma<SR, 4, 2>(args);
    case 7: return launch_quantize_tma<SR, 2, 3>(args);
    case 8: return launch_quantize_tma<SR, 3, 3>(args);
    default: TORCH_CHECK(false, "unknown pass 2 variant ", variant);
  }
  return 0;
}

at::PhiloxCudaState next_philox_state() {
  // Same calls as TE init_philox_state(gen, rng_elts_per_thread = 1024).
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  return gen->philox_cuda_state(kRngEltsPerThread);
}

void check_bf16(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
              " must be a contiguous CUDA bf16 tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 32 == 0, name,
              " must be 32-byte aligned");
}

// The shape rules of the kernels. They return (rows, cols), with cols = grad columns.
std::pair<int, int> check_shapes(const at::Tensor &grad, const at::Tensor &x) {
  check_bf16(grad, "grad");
  check_bf16(x, "x");
  TORCH_CHECK(x.dim() >= 2 && x.size(-1) % 2 == 0, "x must have an even last dim");
  const int64_t cols = x.size(-1) / 2;
  TORCH_CHECK(cols > 0 && cols % 256 == 0, "x last dim must be a multiple of 512");
  TORCH_CHECK(grad.size(-1) == cols && grad.numel() * 2 == x.numel(),
              "grad must have the shape of x with half the last dim");
  TORCH_CHECK(x.numel() < (int64_t{1} << 31), "x must have less than 2^31 elements");
  const int64_t rows = x.numel() / x.size(-1);
  TORCH_CHECK(rows > 0 && rows % 128 == 0, "rows must be a positive multiple of 128");
  return {static_cast<int>(rows), static_cast<int>(cols)};
}

// Two-pass dSwiGLU + NVFP4 quantize of dx. Returns (data, swizzled scales, amax).
std::vector<at::Tensor> dswiglu_nvfp4(const at::Tensor &grad, const at::Tensor &x,
                                      bool stochastic_rounding, int64_t pass1_variant,
                                      int64_t pass2_variant,
    const std::optional<at::Tensor> &alpha_weight_amax,
    const std::optional<at::Tensor> &alpha_out) {
  const auto [rows, cols] = check_shapes(grad, x);
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor partial;
  const int n_partial = amax_pass(grad, x, rows, cols, static_cast<int>(pass1_variant), partial);
  auto data_shape = x.sizes().vec();
  data_shape.back() = cols;  // 2 * cols FP4 values = cols bytes in each row.
  at::Tensor data = at::empty(data_shape, x.options().dtype(at::kByte));
  at::Tensor scales = at::empty({rows, 2 * cols / 16}, x.options().dtype(at::kByte));
  at::Tensor amax = at::empty({1}, x.options().dtype(at::kFloat));
  QuantizeArgs args;
  args.grad = reinterpret_cast<const Vec<8> *>(grad.data_ptr());
  args.x = reinterpret_cast<const Vec<8> *>(x.data_ptr());
  args.data = reinterpret_cast<uint2 *>(data.data_ptr());
  args.scales = scales.data_ptr<uint8_t>();
  args.partial = partial.data_ptr<float>();
  args.n_partial = n_partial;
  args.amax_out = amax.data_ptr<float>();
  args.rows = static_cast<uint32_t>(rows);
  args.cols = static_cast<uint32_t>(cols);
  const GemmAlpha alpha = gemm_alpha(alpha_weight_amax, alpha_out);
  args.alpha_weight_amax = alpha.weight_amax;
  args.alpha_out = alpha.out;
  if (stochastic_rounding) {
    args.philox = next_philox_state();
    quantize_pass<true>(args, static_cast<int>(pass2_variant));
  } else {
    quantize_pass<false>(args, static_cast<int>(pass2_variant));
  }
  return {data, scales, amax};
}

// Only pass 1, for the timing of the passes. Returns the block maxima.
at::Tensor amax_partials(const at::Tensor &grad, const at::Tensor &x, int64_t pass1_variant) {
  const auto [rows, cols] = check_shapes(grad, x);
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor partial;
  amax_pass(grad, x, rows, cols, static_cast<int>(pass1_variant), partial);
  return partial;
}

// Registers and blocks per SM of pass 1 variants 0-3 and pass 2 SR variants 0-8, two numbers
// each. The blocks per SM of variants 4-8 are 0 here (they depend on the shared memory).
std::vector<int64_t> kernel_info() {
  std::vector<int64_t> info;
  const auto add = [&info](auto kernel) {
    cudaFuncAttributes attr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    info.push_back(attr.numRegs);
    info.push_back(blocks_per_sm(kernel));
  };
  add(amax_pass_kernel<8, 2, 1>);
  add(amax_pass_kernel<4, 4, 1>);
  add(amax_pass_kernel<8, 3, 2>);
  add(amax_pass_kernel<8, 4, 2>);
  add(quantize_pass_kernel<true, 1, 1>);
  add(quantize_pass_kernel<true, 2, 1>);
  add(quantize_pass_kernel<true, 1, 2>);
  add(quantize_pass_kernel<true, 2, 2>);
  for (const auto kernel :
       {quantize_pass_tma_kernel<true, 2, 2>, quantize_pass_tma_kernel<true, 3, 2>,
        quantize_pass_tma_kernel<true, 4, 2>, quantize_pass_tma_kernel<true, 2, 3>,
        quantize_pass_tma_kernel<true, 3, 3>}) {
    cudaFuncAttributes attr;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    info.push_back(attr.numRegs);
    info.push_back(0);
  }
  return info;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dswiglu_nvfp4", &dswiglu_nvfp4, "Two-pass dSwiGLU + NVFP4 quantize: (data, scales, amax)",
        py::arg("grad"), py::arg("x"), py::arg("stochastic_rounding"),
        py::arg("pass1_variant") = 2, py::arg("pass2_variant") = 5,
        py::arg("alpha_weight_amax") = py::none(), py::arg("alpha_out") = py::none());
  m.def("amax_partials", &amax_partials, "Pass 1 only (timing)", py::arg("grad"), py::arg("x"),
        py::arg("pass1_variant") = 2);
  m.def("kernel_info", &kernel_info, "Registers and blocks per SM of the kernels");
}
