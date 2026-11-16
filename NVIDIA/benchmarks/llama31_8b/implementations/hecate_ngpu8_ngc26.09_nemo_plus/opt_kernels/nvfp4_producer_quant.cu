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
// The NVFP4 global-scale / block-scale arithmetic, the e2m1 conversion order and the Philox
// draw schedule mirror NVIDIA Transformer Engine (Apache-2.0, Copyright (c) 2022-2026 NVIDIA
// CORPORATION & AFFILIATES): common/cast/nvfp4/core_nvfp4.cuh,
// common/cast/nvfp4/row_cast_col_hadamard_transform_cast_fusion.cu, common/util/ptx.cuh and
// common/util/curanddx.hpp. The kernels themselves are written from scratch.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#include "nvfp4_producer_quant.h"

namespace pbq {

constexpr int TN = 64;
constexpr int NTHR = 256;

__device__ __forceinline__ int soff(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

struct Params {
  int M, N, F;
  const __nv_bfloat16* in0;
  const __nv_bfloat16* in1;
  const __nv_bfloat16* gamma;
  float eps;
  float* partials;
  int P;
  const __nv_bfloat16* rht;
  uint8_t* row_data;
  uint8_t* row_scale;
  float* row_amax;
  uint8_t* col_data;
  uint8_t* col_scale;
  float* col_amax;
  int do_row, do_col, swizzled, sr;
  PhiloxArgs ph;
  int* sched;
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ float lo_f(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi_f(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  f[0] = lo_f(u.x); f[1] = hi_f(u.x);
  f[2] = lo_f(u.y); f[3] = hi_f(u.y);
  f[4] = lo_f(u.z); f[5] = hi_f(u.z);
  f[6] = lo_f(u.w); f[7] = hi_f(u.w);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ uint4 pack8(const float* f) {
  return make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]), pack_bf16x2(f[4], f[5]),
                    pack_bf16x2(f[6], f[7]));
}

__device__ __forceinline__ uint32_t amax2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float amax_of(uint32_t packed) {
  return fmaxf(fabsf(lo_f(packed)), fabsf(hi_f(packed)));
}

constexpr int kPhiloxRounds = 3;

struct Rng {
  uint2 key;
  uint64_t off;
  int kmax, tiles_m;
  int k;
  uint64_t lin_base;
  __device__ __forceinline__ void init(const PhiloxArgs& a, int M, int N) {
    uint64_t seed;
    if (a.captured) {
      seed = static_cast<uint64_t>(*a.seed_ptr);
      off = static_cast<uint64_t>(*a.offset_ptr) + a.offset_intragraph;
    } else {
      seed = a.seed;
      off = a.offset;
    }
    key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
    kmax = (min(M, 1024) + 127) / 128;
    tiles_m = N / 128;
    k = 0;
    lin_base = 0;
  }
  __device__ __forceinline__ void tile(int token_tile) {
    k = token_tile % kmax;
    lin_base = static_cast<uint64_t>(token_tile / kmax) * tiles_m;
  }
  __device__ __forceinline__ uint4 philox(uint64_t ctr, uint64_t seq) const {
    uint4 c = make_uint4(static_cast<uint32_t>(ctr), static_cast<uint32_t>(ctr >> 32),
                         static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32));
    uint2 kk = key;
#pragma unroll
    for (int i = 0; i < kPhiloxRounds; ++i) {
      const uint32_t hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
      const uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
      c = make_uint4(hi1 ^ c.y ^ kk.x, lo1, hi0 ^ c.w ^ kk.y, lo0);
      kk.x += 0x9E3779B9u;
      kk.y += 0xBB67AE85u;
    }
    return c;
  }

  __device__ __forceinline__ uint4 row_bits(int t, int n) const {
    const uint64_t lin = lin_base + (n >> 7);
    const uint64_t seq = 256 + 2 * t + ((n >> 6) & 1) + static_cast<uint64_t>(k + 1) * 512 +
                         lin * kmax * 512;
    return philox(off + ((n >> 4) & 3), seq);
  }

  __device__ __forceinline__ uint4 col_bits(int n, int j) const {
    const uint64_t lin = lin_base + (n >> 7);
    const uint64_t seq = 128 + (n & 127) + static_cast<uint64_t>(k) * 512 + lin * kmax * 512;
    return philox(off + j, seq);
  }
};

struct GScale {
  float dec, mult;
};
__device__ __forceinline__ GScale make_gscale(float amax) {
  constexpr float fp4_max = 6.0f;
  constexpr float fp8_max = 448.0f;
  const float fp4_max_inv = 1.0f / fp4_max;
  const float S = amax > 0.0f ? fminf((fp8_max * fp4_max) / amax, FLT_MAX) : 1.0f;
  GScale g;
  g.dec = 1.0f / S;
  g.mult = S * fp4_max_inv;
  return g;
}

__device__ __forceinline__ float lut_entry(int code, float dec) {
  __nv_fp8_e4m3 sf;
  sf.__x = static_cast<__nv_fp8_storage_t>(code);
  const float q = static_cast<float>(sf) * dec;
  return fminf(1.0f / q, FLT_MAX);
}

__device__ __forceinline__ uint32_t e4m3_code(float pv) {
  return static_cast<uint32_t>(__nv_cvt_float_to_fp8(pv, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint32_t cvt8_rn(const float* v) {
  uint32_t out;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(out)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return out;
}

__device__ __forceinline__ uint32_t cvt4_rn(float v0, float v1, float v2, float v3) {
  uint32_t out;
  asm volatile(
      "{\n"
      ".reg .b8 b0, b1;\n"
      ".reg .b16 h;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "mov.b16 h, {b0, b1};\n"
      "cvt.u32.u16 %0, h;\n"
      "}\n"
      : "=r"(out)
      : "f"(v0), "f"(v1), "f"(v2), "f"(v3));
  return out;
}

__device__ __forceinline__ uint32_t cvt4_rs(float v0, float v1, float v2, float v3, uint32_t rb) {
  uint16_t out;
  asm volatile("cvt.rs.satfinite.e2m1x4.f32 %0, {%4, %3, %2, %1}, %5;\n"
               : "=h"(out)
               : "f"(v0), "f"(v1), "f"(v2), "f"(v3), "r"(rb));
  return out;
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t& a0, uint32_t& a1, uint32_t& a2,
                                              uint32_t& a3, const void* p) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(addr));
}

__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}

struct RhtFrag {
  uint32_t b0[2], b1[2];
  __device__ __forceinline__ void load(const __nv_bfloat16* rht, int lane) {
    const int q = lane & 3, r = lane >> 2;
    const uint16_t* h = reinterpret_cast<const uint16_t*>(rht);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int j = 8 * t + r;
      b0[t] = static_cast<uint32_t>(h[(2 * q) * 16 + j]) |
              (static_cast<uint32_t>(h[(2 * q + 1) * 16 + j]) << 16);
      b1[t] = static_cast<uint32_t>(h[(2 * q + 8) * 16 + j]) |
              (static_cast<uint32_t>(h[(2 * q + 9) * 16 + j]) << 16);
    }
  }
};

__device__ __forceinline__ void rht_mma(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        const RhtFrag& B, uint32_t* p) {
  float d0[4], d1[4];
  mma16816(d0, a0, a1, a2, a3, B.b0[0], B.b1[0]);
  mma16816(d1, a0, a1, a2, a3, B.b0[1], B.b1[1]);

  p[0] = pack_bf16x2(d0[0], d0[1]);
  p[1] = pack_bf16x2(d1[0], d1[1]);
  p[2] = pack_bf16x2(d0[2], d0[3]);
  p[3] = pack_bf16x2(d1[2], d1[3]);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ void block_reduce2_store(float a, float b, float* red, float* outa,
                                                    float* outb) {
  a = warp_max(a);
  b = warp_max(b);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
  if (lane == 0) {
    red[warp] = a;
    red[32 + warp] = b;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float x = 0.f, y = 0.f;
    for (int w = 0; w < nw; ++w) {
      x = fmaxf(x, red[w]);
      y = fmaxf(y, red[32 + w]);
    }
    *outa = x;
    *outb = y;
  }
}

__device__ __forceinline__ float amax16(const uint4& a, const uint4& b) {
  return amax_of(amax2(amax2(amax2(a.x, a.y), amax2(a.z, a.w)), amax2(amax2(b.x, b.y), amax2(b.z, b.w))));
}

__device__ __forceinline__ void scale2(float& a, float& b, float s) {
  asm("{\n"
      ".reg .b64 x, y, z;\n"
      "mov.b64 x, {%2, %3};\n"
      "mov.b64 y, {%4, %4};\n"
      "mul.rn.f32x2 z, x, y;\n"
      "mov.b64 {%0, %1}, z;\n"
      "}\n"
      : "=f"(a), "=f"(b)
      : "f"(a), "f"(b), "f"(s));
}

__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
  float2 d;
  asm("{\n.reg .b64 x, y, z;\nmov.b64 x, {%2, %3};\nmov.b64 y, {%4, %5};\n"
      "mul.rn.f32x2 z, x, y;\nmov.b64 {%0, %1}, z;\n}\n"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}
__device__ __forceinline__ float2 add2(float2 a, float2 b) {
  float2 d;
  asm("{\n.reg .b64 x, y, z;\nmov.b64 x, {%2, %3};\nmov.b64 y, {%4, %5};\n"
      "add.rn.f32x2 z, x, y;\nmov.b64 {%0, %1}, z;\n}\n"
      : "=f"(d.x), "=f"(d.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("{\n.reg .b64 x, y, z, w;\nmov.b64 x, {%2, %3};\nmov.b64 y, {%4, %5};\n"
      "mov.b64 z, {%6, %7};\nfma.rn.f32x2 w, x, y, z;\nmov.b64 {%0, %1}, w;\n}\n"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
}
__device__ __forceinline__ float rcp_approx(float d) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d));
  return r;
}

template <bool SR>
__device__ __forceinline__ uint2 encode16(const uint4& a, const uint4& b, float enc,
                                          const uint4& rb) {
  float f[16];
  unpack8(a, f);
  unpack8(b, f + 8);
#pragma unroll
  for (int e = 0; e < 16; e += 2) scale2(f[e], f[e + 1], enc);
  uint2 out;
  if constexpr (SR) {
    out.x = cvt4_rs(f[0], f[1], f[2], f[3], rb.x) | (cvt4_rs(f[4], f[5], f[6], f[7], rb.y) << 16);
    out.y = cvt4_rs(f[8], f[9], f[10], f[11], rb.z) |
            (cvt4_rs(f[12], f[13], f[14], f[15], rb.w) << 16);
  } else {
    out.x = cvt8_rn(f);
    out.y = cvt8_rn(f + 8);
  }
  return out;
}

__device__ __forceinline__ float silu_fwd(float x) {
  return x * __fdividef(1.f, 1.f + __expf(-x));
}

struct ThreadMap {
  int ld_s;
  int ld_row;
  int ld_col;
  int cp_s0;
  int cp_s1;
  int cp_row;
  int cp_blk;
};

__device__ __forceinline__ ThreadMap make_map() {
  ThreadMap m;
  const int t = threadIdx.x;
  m.ld_row = t >> 3;
  m.ld_col = (t & 7) * 8;
  m.ld_s = soff(m.ld_row, t & 7);
  m.cp_row = t >> 2;
  m.cp_blk = t & 3;
  m.cp_s0 = soff(m.cp_row, 2 * m.cp_blk);
  m.cp_s1 = soff(m.cp_row, 2 * m.cp_blk + 1);
  return m;
}

template <int PROD>
struct Prod;

template <>
struct Prod<kIdent> {
  static constexpr int kTM = 128, kSlots = 1, kHalves = 1, kStages = 2, kExtraT = 1, kMinBlocks = 4;
  static constexpr int kSB = kTM * TN * 2, kBlk = kTM / 64;
  __device__ static int tiles_n(const Params& p) { return p.N / TN; }
  __device__ static void issue(const Params& p, const ThreadMap& tm, int m0, int tn, uint8_t* st) {
    const __nv_bfloat16* src =
        p.in0 + static_cast<size_t>(m0 + tm.ld_row) * p.N + tn * TN + tm.ld_col;
    const size_t step = static_cast<size_t>(32) * p.N;
#pragma unroll
    for (int j = 0; j < kTM / 32; ++j) cp_async16(st + tm.ld_s + 4096 * j, src + j * step);
  }
  __device__ static void compute(const Params&, const ThreadMap& tm, int, int, uint8_t* st,
                                 uint4 (*u)[kBlk][2]) {
#pragma unroll
    for (int i = 0; i < kBlk; ++i) {
      u[0][i][0] = *reinterpret_cast<const uint4*>(st + tm.cp_s0 + 8192 * i);
      u[0][i][1] = *reinterpret_cast<const uint4*>(st + tm.cp_s1 + 8192 * i);
    }
  }
  __device__ static uint8_t* tbuf(uint8_t*, uint8_t* extra, int) { return extra; }
};

template <>
struct Prod<kSwiGLU> {
  static constexpr int kTM = 128, kSlots = 2, kHalves = 1, kStages = 3, kExtraT = 0, kMinBlocks = 2;
  static constexpr int kSB = kTM * TN * 2, kBlk = kTM / 64;
  __device__ static int tiles_n(const Params& p) { return p.N / TN; }
  __device__ static void issue(const Params& p, const ThreadMap& tm, int m0, int tn, uint8_t* st) {
    const size_t ld = 2 * static_cast<size_t>(p.N);
    const __nv_bfloat16* src = p.in0 + (m0 + tm.ld_row) * ld + tn * TN + tm.ld_col;
#pragma unroll
    for (int j = 0; j < kTM / 32; ++j) {
      cp_async16(st + tm.ld_s + 4096 * j, src + j * 32 * ld);
      cp_async16(st + kSB + tm.ld_s + 4096 * j, src + j * 32 * ld + p.N);
    }
  }
  __device__ static void compute(const Params&, const ThreadMap& tm, int, int, uint8_t* st,
                                 uint4 (*u)[kBlk][2]) {
#pragma unroll
    for (int i = 0; i < kBlk; ++i) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint8_t* sp = st + (h ? tm.cp_s1 : tm.cp_s0) + 8192 * i;
        float a[8], bb[8];
        unpack8(*reinterpret_cast<const uint4*>(sp), a);
        unpack8(*reinterpret_cast<const uint4*>(sp + kSB), bb);
#pragma unroll
        for (int e = 0; e < 8; ++e) a[e] = silu_fwd(a[e]) * bb[e];
        u[0][i][h] = pack8(a);
        *reinterpret_cast<uint4*>(sp) = u[0][i][h];
      }
    }
  }

  __device__ static uint8_t* tbuf(uint8_t* st, uint8_t*, int) { return st + kSB; }
};

template <>
struct Prod<kDSwiGLU> {
  static constexpr int kTM = 128, kSlots = 3, kHalves = 2, kStages = 2, kExtraT = 0, kMinBlocks = 2;
  static constexpr int kSB = kTM * TN * 2, kBlk = kTM / 64;
  __device__ static int tiles_n(const Params& p) { return p.F / TN; }
  __device__ static void issue(const Params& p, const ThreadMap& tm, int m0, int tn, uint8_t* st) {
    const size_t F = p.F;
    const __nv_bfloat16* xa = p.in1 + (m0 + tm.ld_row) * (2 * F) + tn * TN + tm.ld_col;
    const __nv_bfloat16* xg = p.in0 + (m0 + tm.ld_row) * F + tn * TN + tm.ld_col;
#pragma unroll
    for (int j = 0; j < kTM / 32; ++j) {
      cp_async16(st + tm.ld_s + 4096 * j, xa + j * 32 * (2 * F));
      cp_async16(st + kSB + tm.ld_s + 4096 * j, xa + j * 32 * (2 * F) + F);
      cp_async16(st + 2 * kSB + tm.ld_s + 4096 * j, xg + j * 32 * F);
    }
  }
  __device__ static void compute(const Params&, const ThreadMap& tm, int, int, uint8_t* st,
                                 uint4 (*u)[kBlk][2]) {
#pragma unroll
    for (int i = 0; i < kBlk; ++i) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint8_t* sp = st + (h ? tm.cp_s1 : tm.cp_s0) + 8192 * i;
        float a[8], bb[8], g[8], da[8], dg[8];
        unpack8(*reinterpret_cast<const uint4*>(sp), a);
        unpack8(*reinterpret_cast<const uint4*>(sp + kSB), bb);
        unpack8(*reinterpret_cast<const uint4*>(sp + 2 * kSB), g);

        float d[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) d[e] = 1.0f + __expf(-a[e]);
        float dmax = d[0];
#pragma unroll
        for (int e = 1; e < 8; ++e) dmax = fmaxf(dmax, d[e]);
        float sv[8];
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
          const float2 dd = make_float2(d[e], d[e + 1]);
          const float2 r = make_float2(rcp_approx(dd.x), rcp_approx(dd.y));
          const float2 er = fma2(make_float2(-dd.x, -dd.y), r, make_float2(1.0f, 1.0f));
          const float2 y = fma2(r, er, r);
          sv[e] = y.x;
          sv[e + 1] = y.y;
        }
        if (__builtin_expect(dmax >= 0x1p126f, 0)) {
#pragma unroll
          for (int e = 0; e < 8; ++e) sv[e] = __frcp_rn(d[e]);
        }
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
          const float2 x = make_float2(a[e], a[e + 1]);
          const float2 sp2 = make_float2(sv[e], sv[e + 1]);
          const float2 t = mul2(x, sp2);
          const float2 one_m_s = add2(make_float2(1.0f, 1.0f), make_float2(-sp2.x, -sp2.y));
          const float2 dact = fma2(t, one_m_s, sp2);
          const float2 gg = make_float2(g[e], g[e + 1]);
          const float2 dd = mul2(mul2(dact, gg), make_float2(bb[e], bb[e + 1]));
          const float2 dgv = mul2(t, gg);
          da[e] = dd.x;
          da[e + 1] = dd.y;
          dg[e] = dgv.x;
          dg[e + 1] = dgv.y;
        }
        u[0][i][h] = pack8(da);
        u[1][i][h] = pack8(dg);
        *reinterpret_cast<uint4*>(sp) = u[0][i][h];
        *reinterpret_cast<uint4*>(sp + kSB) = u[1][i][h];
      }
    }
  }

  __device__ static uint8_t* tbuf(uint8_t* st, uint8_t*, int h) {
    return h == 0 ? st + 2 * kSB : st;
  }
};

struct Aux {
  float lut_row[256];
  float lut_col[256];
  float red[64];
};

template <int PROD>
constexpr int stage_bytes() {
  return Prod<PROD>::kSlots * Prod<PROD>::kSB;
}
template <int PROD>
constexpr size_t smem_bytes() {
  return static_cast<size_t>(Prod<PROD>::kStages) * stage_bytes<PROD>() +
         Prod<PROD>::kExtraT * Prod<PROD>::kSB + sizeof(Aux);
}

template <int PROD, bool CAST, bool SR, bool SWZ>
__global__ void __launch_bounds__(NTHR, Prod<PROD>::kMinBlocks) pq_kernel(Params p) {
  using Pr = Prod<PROD>;
  constexpr int NS = Pr::kStages;
  constexpr int SB = stage_bytes<PROD>();
  constexpr int TM = Pr::kTM, kSB = Pr::kSB, BLK = Pr::kBlk, TROW = 2 * TM;
  extern __shared__ __align__(128) uint8_t dsm[];
  uint8_t* stages = dsm;
  uint8_t* extra = dsm + NS * SB;
  Aux& ax = *reinterpret_cast<Aux*>(extra + Pr::kExtraT * kSB);

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const ThreadMap tmap = make_map();
  const int tn_count = Pr::tiles_n(p);
  const int ntiles = (p.M / TM) * tn_count;

  constexpr int R = NS + 1;
  __shared__ int s_ring[R];
  int* claim_ctr = p.sched + (CAST ? 2 : 0);
  int* done_ctr = claim_ctr + 1;
  auto tile_index = [&](int c) -> int { return CAST ? ntiles - 1 - c : c; };
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s2 = 0; s2 < NS; ++s2) s_ring[s2] = atomicAdd(claim_ctr, 1);
  }
  __syncthreads();
#pragma unroll
  for (int s2 = 0; s2 < NS - 1; ++s2) {
    const int c = s_ring[s2];
    if (c < ntiles) {
      const int t = tile_index(c), tm = t / tn_count, tn = t - tm * tn_count;
      Pr::issue(p, tmap, tm * TM, tn, stages + s2 * SB);
    }
    cp_async_commit();
  }

  RhtFrag B;
  B.load(p.rht, lane);

  const int fi = warp & 3, g0 = warp >> 2;
  const int mi = lane >> 3, rr = lane & 7;
  const int rht_s = soff(g0 * 16 + (mi >> 1) * 8 + rr, fi * 2 + (mi & 1));
  const int q = lane & 3, r = lane >> 2;
  const int f0 = fi * 16 + r;

  const int fq = (lane & 7) + 8 * warp, gq = lane >> 3;
  const int tq0 = fq * TROW + (((2 * gq) ^ (lane & 7)) << 4);
  const int tq1 = fq * TROW + (((2 * gq + 1) ^ (lane & 7)) << 4);

  float mult_row = 0.f, mult_col = 0.f;
  Rng rng;
  if constexpr (CAST) {
    float ra = 0.f, ca = 0.f;
    for (int i = threadIdx.x; i < p.P; i += NTHR) {
      ra = fmaxf(ra, __ldcg(p.partials + i));
      ca = fmaxf(ca, __ldcg(p.partials + p.P + i));
    }
    ra = warp_max(ra);
    ca = warp_max(ca);
    if (lane == 0) {
      ax.red[warp] = ra;
      ax.red[32 + warp] = ca;
    }
    __syncthreads();
    float row_amax = 0.f, col_amax = 0.f;
#pragma unroll
    for (int w = 0; w < NTHR / 32; ++w) {
      row_amax = fmaxf(row_amax, ax.red[w]);
      col_amax = fmaxf(col_amax, ax.red[32 + w]);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      if (p.do_row) *p.row_amax = row_amax;
      if (p.do_col) *p.col_amax = col_amax;
    }
    const GScale grow = make_gscale(row_amax), gcol = make_gscale(col_amax);
    mult_row = grow.mult;
    mult_col = gcol.mult;
    ax.lut_row[threadIdx.x] = lut_entry(threadIdx.x, grow.dec);
    ax.lut_col[threadIdx.x] = lut_entry(threadIdx.x, gcol.dec);
    if constexpr (SR) rng.init(p.ph, p.M, p.N);
  }

  float rmax = 0.f, cmax = 0.f;
  const int Nh = p.N / 2, Mh = p.M / 2;
  const int rC4 = ((p.N / 16) + 3) / 4, cC4 = ((p.M / 16) + 3) / 4;

  const int rsc_in = ((tmap.cp_row & 31) << 4) + ((tmap.cp_row >> 5) << 2) + tmap.cp_blk;
  const int csc_in = ((fq & 31) << 4) + ((fq >> 5) << 2) + gq;

  int pending = 0;
  for (int k = 0;; ++k) {
    cp_async_wait<NS - 2>();
    __syncthreads();
    const int cur = s_ring[k % R];
    if (cur >= ntiles) break;
    if (threadIdx.x == 0) pending = atomicAdd(claim_ctr, 1);
    {
      const int c = s_ring[(k + NS - 1) % R];
      if (c < ntiles) {
        const int t2 = tile_index(c), tm2 = t2 / tn_count, tn2 = t2 - tm2 * tn_count;
        Pr::issue(p, tmap, tm2 * TM, tn2, stages + ((k + NS - 1) % NS) * SB);
      }
      cp_async_commit();
    }
    const int t = tile_index(cur), tmi = t / tn_count, tn = t - tmi * tn_count;
    const int m0 = tmi * TM;
    if constexpr (SR) rng.tile(tmi);
    uint8_t* st = stages + (k % NS) * SB;

    uint4 u[2][BLK][2];
    Pr::compute(p, tmap, m0, tn, st, u);

#pragma unroll
    for (int h = 0; h < Pr::kHalves; ++h) {
      if constexpr (!CAST) {
#pragma unroll
        for (int i = 0; i < BLK; ++i) rmax = fmaxf(rmax, amax16(u[h][i][0], u[h][i][1]));
      } else if (p.do_row) {
        const int n0 = (PROD == kDSwiGLU) ? (h * p.F + tn * TN) : tn * TN;
        uint8_t* rd = p.row_data + static_cast<size_t>(m0 + tmap.cp_row) * Nh + n0 / 2 +
                      tmap.cp_blk * 8;
        uint8_t* rs;
        if constexpr (SWZ) {
          rs = p.row_scale + (static_cast<size_t>(m0 >> 7) * rC4 + (n0 >> 6)) * 512 + rsc_in +
               (((m0 >> 6) & 1) << 3);
        } else {
          rs = p.row_scale + static_cast<size_t>(m0 + tmap.cp_row) * (rC4 * 4) + n0 / 16 +
               tmap.cp_blk;
        }
#pragma unroll
        for (int i = 0; i < BLK; ++i) {
          const float bm = amax16(u[h][i][0], u[h][i][1]);
          const uint32_t code = e4m3_code(bm * mult_row);
          const float enc = ax.lut_row[code];
          uint4 rb = make_uint4(0, 0, 0, 0);
          if constexpr (SR) rb = rng.row_bits(tmap.cp_row + 64 * i, n0 + 16 * tmap.cp_blk);
          *reinterpret_cast<uint2*>(rd + static_cast<size_t>(64 * i) * Nh) =
              encode16<SR>(u[h][i][0], u[h][i][1], enc, rb);
          if constexpr (SWZ) {
            rs[8 * i] = static_cast<uint8_t>(code);
          } else {
            rs[static_cast<size_t>(64 * i) * (rC4 * 4)] = static_cast<uint8_t>(code);
          }
        }
      }
    }
    if constexpr (PROD != kIdent) __syncthreads();

    if (p.do_col) {
#pragma unroll
      for (int h = 0; h < Pr::kHalves; ++h) {
        const uint8_t* slot = st + h * kSB;
        if constexpr (!CAST) {
#pragma unroll
          for (int kq = 0; kq < TM / 32; ++kq) {
            uint32_t a0, a1, a2, a3, pk[4];
            ldsm_x4_trans(a0, a1, a2, a3, slot + rht_s + 4096 * kq);
            rht_mma(a0, a1, a2, a3, B, pk);
            cmax = fmaxf(cmax, fmaxf(amax_of(amax2(pk[0], pk[1])), amax_of(amax2(pk[2], pk[3]))));
          }
        } else {

          uint8_t* T = Pr::tbuf(st, extra, h);
#pragma unroll
          for (int kq = 0; kq < TM / 32; ++kq) {
            uint32_t a0, a1, a2, a3, pk[4];
            ldsm_x4_trans(a0, a1, a2, a3, slot + rht_s + 4096 * kq);
            rht_mma(a0, a1, a2, a3, B, pk);
            const int c0 = 2 * (g0 + 2 * kq);
            uint8_t* w0 = T + f0 * TROW + 4 * q;
            *reinterpret_cast<uint32_t*>(w0 + ((c0 ^ r) << 4)) = pk[0];
            *reinterpret_cast<uint32_t*>(w0 + (((c0 + 1) ^ r) << 4)) = pk[1];
            *reinterpret_cast<uint32_t*>(w0 + 8 * TROW + ((c0 ^ r) << 4)) = pk[2];
            *reinterpret_cast<uint32_t*>(w0 + 8 * TROW + (((c0 + 1) ^ r) << 4)) = pk[3];
          }
          __syncthreads();
          const int n0 = (PROD == kDSwiGLU) ? (h * p.F + tn * TN) : tn * TN;
          uint8_t* cd = p.col_data + static_cast<size_t>(n0 + fq) * Mh + m0 / 2 + gq * 8;
          uint8_t* cs;
          if constexpr (SWZ) {
            cs = p.col_scale + (static_cast<size_t>(n0 >> 7) * cC4 + (m0 >> 6)) * 512 + csc_in +
                 (((n0 >> 6) & 1) << 3);
          } else {
            cs = p.col_scale + static_cast<size_t>(n0 + fq) * (cC4 * 4) + m0 / 16 + gq;
          }
#pragma unroll
          for (int i = 0; i < BLK; ++i) {
            const uint4 c0v = *reinterpret_cast<const uint4*>(T + tq0 + 128 * i);
            const uint4 c1v = *reinterpret_cast<const uint4*>(T + tq1 + 128 * i);
            const float bm = amax16(c0v, c1v);
            const uint32_t code = e4m3_code(bm * mult_col);
            const float enc = ax.lut_col[code];
            uint4 rb = make_uint4(0, 0, 0, 0);
            if constexpr (SR) rb = rng.col_bits(n0 + fq, gq + 4 * i);
            *reinterpret_cast<uint2*>(cd + 32 * i) = encode16<SR>(c0v, c1v, enc, rb);
            if constexpr (SWZ) {
              cs[512 * i] = static_cast<uint8_t>(code);
            } else {
              cs[4 * i] = static_cast<uint8_t>(code);
            }
          }
        }
      }
    }
    if (threadIdx.x == 0) s_ring[(k + NS) % R] = pending;
  }
  cp_async_wait<0>();
  if constexpr (!CAST) {
    block_reduce2_store(rmax, cmax, ax.red, p.partials + blockIdx.x, p.partials + p.P + blockIdx.x);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(done_ctr, 1) == static_cast<int>(gridDim.x) - 1) {
      atomicExch(claim_ctr, 0);
      atomicExch(done_ctr, 0);
    }
  }
}

constexpr int RQ_THR = 512;
constexpr int RQ_H = 4096;
constexpr int RQ_CH = 256;
constexpr int RQ_NCH = RQ_H / RQ_CH;
constexpr int RQ_CSTRIDE = RQ_CH * 2 + 16;
constexpr int RQ_CBYTES = 64 * RQ_CSTRIDE;

constexpr size_t RQ_SMEM = 2 * RQ_CBYTES + RQ_H * 4 + 2 * 256 * sizeof(float);

struct RqParams {
  int M, nbig;
  const __nv_bfloat16* x;
  const __nv_bfloat16* gamma;
  float* rstd;
  float eps;
  const __nv_bfloat16* rht;
  float* stf;
  int* sti;
  float factor;
  uint8_t* row_data;
  uint8_t* row_scale;
  float* row_amax;
  uint8_t* col_data;
  uint8_t* col_scale;
  float* col_amax;
};

__device__ __forceinline__ int rq_unit_m0(int u, int nbig, int M, int& R) {
  if (u < nbig) {
    R = 4;
    return 64 * u;
  }
  const int m0 = 64 * nbig + 16 * (u - nbig);
  R = m0 < M ? 1 : 0;
  return m0;
}

__device__ __forceinline__ float2 unpack2(uint32_t u) { return make_float2(lo_f(u), hi_f(u)); }

__device__ __forceinline__ float2 sumsq8(const uint4& v, float2 acc) {
  const float2 x0 = unpack2(v.x), x1 = unpack2(v.y), x2 = unpack2(v.z), x3 = unpack2(v.w);
  acc = fma2(x0, x0, acc);
  acc = fma2(x1, x1, acc);
  acc = fma2(x2, x2, acc);
  return fma2(x3, x3, acc);
}

struct RhtFragP {
  uint32_t b0[2], b1[2];
  __device__ __forceinline__ void load(const __nv_bfloat16* rht, int lane) {
    const int q = lane & 3, r = lane >> 2;
    const uint16_t* h = reinterpret_cast<const uint16_t*>(rht);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
      const int j = 4 * (r >> 1) + (r & 1) + 2 * t;
      b0[t] = static_cast<uint32_t>(h[(2 * q) * 16 + j]) |
              (static_cast<uint32_t>(h[(2 * q + 1) * 16 + j]) << 16);
      b1[t] = static_cast<uint32_t>(h[(2 * q + 8) * 16 + j]) |
              (static_cast<uint32_t>(h[(2 * q + 9) * 16 + j]) << 16);
    }
  }
};

__device__ __forceinline__ uint2 quad_t16(uint32_t lo, uint32_t hi, int q) {
  const uint32_t t = __shfl_xor_sync(0xffffffffu, (q & 2) ? lo : hi, 2);
  const uint32_t X = (q & 2) ? t : lo, Y = (q & 2) ? hi : t;
  const uint32_t S = __byte_perm(X, Y, (q & 1) ? 0x5410 : 0x7632);
  const uint32_t Rv = __shfl_xor_sync(0xffffffffu, S, 1);
  uint2 o;
  if (q & 1) {
    o.x = __byte_perm(Rv, X, 0x7610);
    o.y = __byte_perm(Rv, Y, 0x7632);
  } else {
    o.x = __byte_perm(X, Rv, 0x5410);
    o.y = __byte_perm(Y, Rv, 0x7610);
  }
  return o;
}

struct RqCtx {
  int lane, warp, q, r, ld_off, Mh, rC4, cC4;
  float mult_row, mult_col;
  const float4* gsm4;
  const float* lut_row;
  const float* lut_col;
  uint8_t* cbuf;
  RhtFragP B;
};

template <int RN>
__device__ __forceinline__ void rq_stats_load(const uint4* const (&sp)[4], int c, uint4 (&sx)[4]) {
#pragma unroll
  for (int rr = 0; rr < RN; ++rr) sx[rr] = __ldcg(sp[rr] + 32 * c);
}

template <int RN>
__device__ __forceinline__ void rq_stats_finish(float eps, float2 (&acc)[4], float (&rs)[4]) {
#pragma unroll
  for (int rr = 0; rr < RN; ++rr) {
    float t = acc[rr].x + acc[rr].y;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    constexpr float rn = 1.f / float(RQ_H);
    rs[rr] = rsqrtf(t * rn + eps);
  }
#pragma unroll
  for (int rr = 0; rr < 4; ++rr) acc[rr] = make_float2(0.f, 0.f);
}

__device__ __forceinline__ float amax4f(float a, float b, float c, float d) {
  return fmaxf(fmaxf(fabsf(a), fabsf(b)), fmaxf(fabsf(c), fabsf(d)));
}

__device__ __forceinline__ uint32_t enc4(float v0, float v1, float v2, float v3, float e) {
  scale2(v0, v1, e);
  scale2(v2, v3, e);
  return cvt4_rn(v0, v1, v2, v3);
}

template <int R, int RN, bool CAST, bool SWZ>
__device__ __forceinline__ void rq_unit(const RqParams& p, const RqCtx& cx, int m0,
                                        const float (&rs)[4], int m0n, float2 (&acc)[4], int& kc,
                                        uint32_t& rmax2, float& cmax) {
  const int lane = cx.lane, warp = cx.warp, q = cx.q, r = cx.r;
  if (CAST && lane == 0) {
#pragma unroll
    for (int rr = 0; rr < R; ++rr) p.rstd[m0 + warp + 16 * rr] = rs[rr];
  }

  const uint4* xp[R];
  uint8_t* rdp[R];
  uint8_t* rsp[R];
#pragma unroll
  for (int rr = 0; rr < R; ++rr) {
    const int m = m0 + warp + 16 * rr;
    xp[rr] = reinterpret_cast<const uint4*>(p.x) + static_cast<size_t>(m) * (RQ_H / 8) + lane;
    rdp[rr] = p.row_data + static_cast<size_t>(m) * (RQ_H / 2) + lane * 4;
    const int bl = lane >> 1;
    size_t so;
    if constexpr (SWZ) {
      so = (static_cast<size_t>(m >> 7) * cx.rC4 + (bl >> 2)) * 512 + (m & 31) * 16 +
           ((m >> 5) & 3) * 4 + (bl & 3);
    } else {
      so = static_cast<size_t>(m) * (cx.rC4 * 4) + bl;
    }
    rsp[rr] = p.row_scale + so;
  }
  const uint4* sp[4];
#pragma unroll
  for (int rr = 0; rr < 4; ++rr)
    sp[rr] = reinterpret_cast<const uint4*>(p.x) +
             static_cast<size_t>(m0n + warp + 16 * (rr < RN ? rr : 0)) * (RQ_H / 8) + lane;

  const int gi0 = m0 >> 4;
  uint8_t* cdp = p.col_data + static_cast<size_t>(warp * 16 + r) * cx.Mh + (m0 >> 1) +
                 (R == 4 ? 8 * q : 2 * q);
  const size_t cd_step = static_cast<size_t>(RQ_CH) * cx.Mh;
  uint8_t* csp;
  size_t cs_step;
  {
    const int fl = warp * 16 + r + 8 * (q & 1);
    if constexpr (SWZ) {
      csp = p.col_scale + (static_cast<size_t>(fl >> 7) * cx.cC4 + (gi0 >> 2)) * 512 +
            (fl & 31) * 16 + ((fl >> 5) & 3) * 4 + (R == 4 ? 0 : (gi0 & 3));
      cs_step = static_cast<size_t>(2) * cx.cC4 * 512;
    } else {
      csp = p.col_scale + static_cast<size_t>(fl) * (cx.cC4 * 4) + gi0;
      cs_step = static_cast<size_t>(RQ_CH) * (cx.cC4 * 4);
    }
  }
  uint4 xn[R];
#pragma unroll
  for (int rr = 0; rr < R; ++rr) xn[rr] = __ldcs(xp[rr]);

#pragma unroll 1
  for (int c = 0; c < RQ_NCH; ++c, ++kc) {
    uint4 sx[4];
    rq_stats_load<RN>(sp, c, sx);
    uint8_t* cb = cx.cbuf + (kc & 1) * RQ_CBYTES;
    const float4 g0 = cx.gsm4[c * 64 + lane], g1 = cx.gsm4[c * 64 + 32 + lane];

#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
      const float2 rs2 = make_float2(rs[rr], rs[rr]);
      const uint4 xv = xn[rr];
      float2 y0 = mul2(make_float2(g0.x, g0.y), mul2(rs2, unpack2(xv.x)));
      float2 y1 = mul2(make_float2(g0.z, g0.w), mul2(rs2, unpack2(xv.y)));
      float2 y2 = mul2(make_float2(g1.x, g1.y), mul2(rs2, unpack2(xv.z)));
      float2 y3 = mul2(make_float2(g1.z, g1.w), mul2(rs2, unpack2(xv.w)));
      const uint4 yv = make_uint4(pack_bf16x2(y0.x, y0.y), pack_bf16x2(y1.x, y1.y),
                                  pack_bf16x2(y2.x, y2.y), pack_bf16x2(y3.x, y3.y));
      *reinterpret_cast<uint4*>(cb + (warp + 16 * rr) * RQ_CSTRIDE + lane * 16) = yv;
      const uint32_t a2 = amax2(amax2(yv.x, yv.y), amax2(yv.z, yv.w));
      rmax2 = amax2(rmax2, a2);
      if constexpr (CAST) {

        const float a8 = amax_of(a2);
        const float bm = fmaxf(a8, __shfl_xor_sync(0xffffffffu, a8, 1));
        const uint32_t code = e4m3_code(bm * cx.mult_row);
        const float2 e2 = make_float2(cx.lut_row[code], cx.lut_row[code]);
        y0 = mul2(y0, e2);
        y1 = mul2(y1, e2);
        y2 = mul2(y2, e2);
        y3 = mul2(y3, e2);
        const float f[8] = {y0.x, y0.y, y1.x, y1.y, y2.x, y2.y, y3.x, y3.y};
        *reinterpret_cast<uint32_t*>(rdp[rr] + c * 128) = cvt8_rn(f);
        if ((lane & 1) == 0) rsp[rr][SWZ ? c * 2048 : c * 16] = static_cast<uint8_t>(code);
      }
    }

    if (c + 1 < RQ_NCH) {
#pragma unroll
      for (int rr = 0; rr < R; ++rr) xn[rr] = __ldcs(xp[rr] + 32 * (c + 1));
    }
#pragma unroll
    for (int rr = 0; rr < RN; ++rr) acc[rr] = sumsq8(sx[rr], acc[rr]);
    __syncthreads();

    uint32_t pA[2] = {0u, 0u}, pB[2] = {0u, 0u}, cw = 0u;
#pragma unroll
    for (int gt = 0; gt < R; ++gt) {
      uint32_t a0, a1, a2, a3;
      ldsm_x4_trans(a0, a1, a2, a3, cb + gt * 16 * RQ_CSTRIDE + cx.ld_off + warp * 32);
      float d0[4], d1[4];
      mma16816(d0, a0, a1, a2, a3, cx.B.b0[0], cx.B.b1[0]);
      mma16816(d1, a0, a1, a2, a3, cx.B.b0[1], cx.B.b1[1]);

      float aA = amax4f(d0[0], d0[1], d1[0], d1[1]);
      float aB = amax4f(d0[2], d0[3], d1[2], d1[3]);
      aA = fmaxf(aA, __shfl_xor_sync(0xffffffffu, aA, 1));
      aB = fmaxf(aB, __shfl_xor_sync(0xffffffffu, aB, 1));
      aA = fmaxf(aA, __shfl_xor_sync(0xffffffffu, aA, 2));
      aB = fmaxf(aB, __shfl_xor_sync(0xffffffffu, aB, 2));
      cmax = fmaxf(cmax, fmaxf(aA, aB));
      if constexpr (CAST) {
        const uint32_t cA = e4m3_code(aA * cx.mult_col), cB = e4m3_code(aB * cx.mult_col);
        const uint32_t wA = enc4(d0[0], d0[1], d1[0], d1[1], cx.lut_col[cA]);
        const uint32_t wB = enc4(d0[2], d0[3], d1[2], d1[3], cx.lut_col[cB]);
        const uint32_t cq = (q & 1) ? cB : cA;
        if constexpr (R == 4) {
          pA[gt >> 1] |= wA << (16 * (gt & 1));
          pB[gt >> 1] |= wB << (16 * (gt & 1));
          cw |= cq << (8 * gt);
        } else {
          uint8_t* cd = cdp + c * cd_step;
          *reinterpret_cast<uint16_t*>(cd) = static_cast<uint16_t>(wA);
          *reinterpret_cast<uint16_t*>(cd + static_cast<size_t>(8) * cx.Mh) =
              static_cast<uint16_t>(wB);
          if (q < 2) csp[c * cs_step] = static_cast<uint8_t>(cq);
        }
      }
    }
    if constexpr (CAST && R == 4) {

      const uint2 oA = quad_t16(pA[0], pA[1], q), oB = quad_t16(pB[0], pB[1], q);
      uint8_t* cd = cdp + c * cd_step;
      *reinterpret_cast<uint2*>(cd) = oA;
      *reinterpret_cast<uint2*>(cd + static_cast<size_t>(8) * cx.Mh) = oB;
      if (q < 2) *reinterpret_cast<uint32_t*>(csp + c * cs_step) = cw;
    }
  }
}

template <int R, bool CAST, bool SWZ>
__device__ __forceinline__ void rq_unit_any(const RqParams& p, const RqCtx& cx, int m0,
                                            const float (&rs)[4], int m0n, int Rn,
                                            float2 (&acc)[4], int& kc, uint32_t& rmax2,
                                            float& cmax) {
  if (Rn == 4) {
    rq_unit<R, 4, CAST, SWZ>(p, cx, m0, rs, m0n, acc, kc, rmax2, cmax);
  } else if (Rn == 1) {
    rq_unit<R, 1, CAST, SWZ>(p, cx, m0, rs, m0n, acc, kc, rmax2, cmax);
  } else {
    rq_unit<R, 0, CAST, SWZ>(p, cx, m0, rs, m0n, acc, kc, rmax2, cmax);
  }
}

template <bool CAST, bool SWZ>
__global__ void __launch_bounds__(RQ_THR, 1) rq_kernel(RqParams p) {
  extern __shared__ __align__(128) uint8_t dsm[];
  __shared__ int s_claim[2], s_init[2];
  __shared__ uint32_t s_red[2][RQ_THR / 32];

  RqCtx cx;
  cx.lane = threadIdx.x & 31;
  cx.warp = threadIdx.x >> 5;
  cx.q = cx.lane & 3;
  cx.r = cx.lane >> 2;
  {
    const int mi = cx.lane >> 3, r8 = cx.lane & 7;
    cx.ld_off = ((mi >> 1) * 8 + r8) * RQ_CSTRIDE + (mi & 1) * 16;
  }
  cx.cbuf = dsm;
  cx.gsm4 = reinterpret_cast<const float4*>(dsm + 2 * RQ_CBYTES);
  float* lut_row = reinterpret_cast<float*>(dsm + 2 * RQ_CBYTES + RQ_H * 4);
  float* lut_col = lut_row + 256;
  cx.lut_row = lut_row;
  cx.lut_col = lut_col;
  const int M = p.M;
  cx.Mh = M / 2;
  cx.rC4 = (RQ_H / 16 + 3) / 4;
  cx.cC4 = ((M / 16) + 3) / 4;
  int* done_ctr = p.sti;
  int* claim_ctr = p.sti + 1;

  if (threadIdx.x == 0) s_init[0] = atomicAdd(claim_ctr, 1);
  {

    const int vv = threadIdx.x;
    float g[8];
    unpack8(__ldg(reinterpret_cast<const uint4*>(p.gamma) + vv), g);
    float4* dst = reinterpret_cast<float4*>(dsm + 2 * RQ_CBYTES) + (vv >> 5) * 64 + (vv & 31);
    dst[0] = make_float4(g[0], g[1], g[2], g[3]);
    dst[32] = make_float4(g[4], g[5], g[6], g[7]);
  }
  cx.mult_row = 0.f;
  cx.mult_col = 0.f;
  if constexpr (CAST) {
    const float ae_r = __ldcg(p.stf + 0) * p.factor;
    const float ae_c = __ldcg(p.stf + 1) * p.factor;
    const GScale grow = make_gscale(ae_r), gcol = make_gscale(ae_c);
    cx.mult_row = grow.mult;
    cx.mult_col = gcol.mult;
    if (threadIdx.x < 256) {
      lut_row[threadIdx.x] = lut_entry(threadIdx.x, grow.dec);
    } else {
      lut_col[threadIdx.x - 256] = lut_entry(threadIdx.x - 256, gcol.dec);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      *p.row_amax = ae_r;
      *p.col_amax = ae_c;
    }
  }
  cx.B.load(p.rht, cx.lane);
  __syncthreads();

  uint32_t rmax2 = 0u;
  float cmax = 0.f;
  int R, Rn;
  int m0 = rq_unit_m0(s_init[0], p.nbig, M, R);
  float rs[4] = {0.f, 0.f, 0.f, 0.f};
  float2 acc[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i] = make_float2(0.f, 0.f);
  if (R > 0) {

    const uint4* sp[4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr)
      sp[rr] = reinterpret_cast<const uint4*>(p.x) +
               static_cast<size_t>(m0 + cx.warp + 16 * (rr < R ? rr : 0)) * (RQ_H / 8) + cx.lane;
    if (R == 4) {
#pragma unroll 2
      for (int c = 0; c < RQ_NCH; ++c) {
        uint4 sx[4];
        rq_stats_load<4>(sp, c, sx);
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) acc[rr] = sumsq8(sx[rr], acc[rr]);
      }
      rq_stats_finish<4>(p.eps, acc, rs);
    } else {
#pragma unroll 4
      for (int c = 0; c < RQ_NCH; ++c) {
        uint4 sx[4];
        rq_stats_load<1>(sp, c, sx);
        acc[0] = sumsq8(sx[0], acc[0]);
      }
      rq_stats_finish<1>(p.eps, acc, rs);
    }
  }

  if (threadIdx.x == 0) s_init[1] = atomicAdd(claim_ctr, 1);
  __syncthreads();
  int nxt = s_init[1];
  int kc = 0, slot = 0;
  while (R > 0) {
    const int m0n = rq_unit_m0(nxt, p.nbig, M, Rn);
    if (threadIdx.x == 0) s_claim[slot] = atomicAdd(claim_ctr, 1);
    if (R == 4) {
      rq_unit_any<4, CAST, SWZ>(p, cx, m0, rs, m0n, Rn, acc, kc, rmax2, cmax);
    } else {
      rq_unit_any<1, CAST, SWZ>(p, cx, m0, rs, m0n, Rn, acc, kc, rmax2, cmax);
    }
    const int nn = s_claim[slot];
    slot ^= 1;
    if (Rn == 4) {
      rq_stats_finish<4>(p.eps, acc, rs);
    } else if (Rn == 1) {
      rq_stats_finish<1>(p.eps, acc, rs);
    }
    m0 = m0n;
    R = Rn;
    nxt = nn;
  }

  float rmax = amax_of(rmax2);
  rmax = warp_max(rmax);
  cmax = warp_max(cmax);
  if (cx.lane == 0) {
    s_red[0][cx.warp] = __float_as_uint(rmax);
    s_red[1][cx.warp] = __float_as_uint(cmax);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = 0.f, b = 0.f;
#pragma unroll
    for (int w = 0; w < RQ_THR / 32; ++w) {
      a = fmaxf(a, __uint_as_float(s_red[0][w]));
      b = fmaxf(b, __uint_as_float(s_red[1][w]));
    }
    unsigned int* nx = reinterpret_cast<unsigned int*>(p.stf + 2);
    atomicMax(nx, __float_as_uint(a));
    atomicMax(nx + 1, __float_as_uint(b));
    __threadfence();
    if (atomicAdd(done_ctr, 1) == static_cast<int>(gridDim.x) - 1) {
      __threadfence();
      const unsigned int nr = atomicExch(nx, 0u), nc = atomicExch(nx + 1, 0u);
      p.stf[0] = __uint_as_float(nr);
      p.stf[1] = __uint_as_float(nc);
      atomicExch(claim_ctr, 0);
      atomicExch(done_ctr, 0);
    }
  }
}

static int num_sms() {
  static int n = 0;
  if (n == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}

template <typename K>
static int occupancy(K kernel, int threads, size_t smem) {
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  int n = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, smem);
  return n < 1 ? 1 : n;
}

template <int PROD>
static int ctas_per_sm() {
  static int n = 0;
  if (n == 0) {
    const size_t sm = smem_bytes<PROD>();
    int v[5] = {occupancy(pq_kernel<PROD, false, false, true>, NTHR, sm),
                occupancy(pq_kernel<PROD, true, false, true>, NTHR, sm),
                occupancy(pq_kernel<PROD, true, true, true>, NTHR, sm),
                occupancy(pq_kernel<PROD, true, false, false>, NTHR, sm),
                occupancy(pq_kernel<PROD, true, true, false>, NTHR, sm)};
    n = v[0];
    for (int i = 1; i < 5; ++i) n = v[i] < n ? v[i] : n;
  }
  return n;
}

static Params make_params(const QuantDesc& d) {
  Params p;
  p.M = d.M;
  p.N = d.N;
  p.F = (d.producer == kDSwiGLU) ? d.N / 2 : (d.producer == kSwiGLU ? d.N : 0);
  p.in0 = reinterpret_cast<const __nv_bfloat16*>(d.in0);
  p.in1 = reinterpret_cast<const __nv_bfloat16*>(d.in1);
  p.gamma = reinterpret_cast<const __nv_bfloat16*>(d.gamma);
  p.eps = d.eps;
  p.partials = d.partials;
  p.P = d.num_partials;
  p.rht = reinterpret_cast<const __nv_bfloat16*>(d.rht);
  p.row_data = d.row_data;
  p.row_scale = d.row_scale;
  p.row_amax = d.row_amax;
  p.col_data = d.col_data;
  p.col_scale = d.col_scale;
  p.col_amax = d.col_amax;
  p.do_row = d.do_row;
  p.do_col = d.do_col;
  p.swizzled = d.swizzled;
  p.sr = d.stochastic;
  p.ph = d.philox;
  p.sched = d.sched;
  return p;
}

static long tiles_of(int producer, int M, int N) {
  const int tn = (producer == kDSwiGLU) ? (N / 2) / TN : N / TN;
  int tm = Prod<kIdent>::kTM;
  if (producer == kSwiGLU) tm = Prod<kSwiGLU>::kTM;
  if (producer == kDSwiGLU) tm = Prod<kDSwiGLU>::kTM;
  return static_cast<long>(M / tm) * tn;
}

static int grid_for(int producer, int M, int N) {
  long cap;
  switch (producer) {
    case kIdent: cap = static_cast<long>(num_sms()) * ctas_per_sm<kIdent>(); break;
    case kSwiGLU: cap = static_cast<long>(num_sms()) * ctas_per_sm<kSwiGLU>(); break;
    default: cap = static_cast<long>(num_sms()) * ctas_per_sm<kDSwiGLU>(); break;
  }
  const long work = tiles_of(producer, M, N);
  return static_cast<int>(work < cap ? work : cap);
}

int partials_needed(int producer, int M, int N) {
  if (producer == kRMSNorm) return 0;
  return grid_for(producer, M, N);
}

template <int PROD>
static void launch_stats_t(const Params& p, cudaStream_t stream) {
  ctas_per_sm<PROD>();
  pq_kernel<PROD, false, false, true><<<p.P, NTHR, smem_bytes<PROD>(), stream>>>(p);
}

template <int PROD>
static void launch_cast_t(const Params& p, int grid, cudaStream_t stream) {
  ctas_per_sm<PROD>();
  const size_t sm = smem_bytes<PROD>();
  if (p.sr) {
    if (p.swizzled) pq_kernel<PROD, true, true, true><<<grid, NTHR, sm, stream>>>(p);
    else pq_kernel<PROD, true, true, false><<<grid, NTHR, sm, stream>>>(p);
  } else {
    if (p.swizzled) pq_kernel<PROD, true, false, true><<<grid, NTHR, sm, stream>>>(p);
    else pq_kernel<PROD, true, false, false><<<grid, NTHR, sm, stream>>>(p);
  }
}

cudaError_t launch_stats(const QuantDesc& d, cudaStream_t stream) {
  Params p = make_params(d);
  switch (d.producer) {
    case kIdent: launch_stats_t<kIdent>(p, stream); break;
    case kSwiGLU: launch_stats_t<kSwiGLU>(p, stream); break;
    case kDSwiGLU: launch_stats_t<kDSwiGLU>(p, stream); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t launch_cast(const QuantDesc& d, cudaStream_t stream) {
  Params p = make_params(d);
  const int grid = grid_for(d.producer, d.M, d.N);
  switch (d.producer) {
    case kIdent: launch_cast_t<kIdent>(p, grid, stream); break;
    case kSwiGLU: launch_cast_t<kSwiGLU>(p, grid, stream); break;
    case kDSwiGLU: launch_cast_t<kDSwiGLU>(p, grid, stream); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

template <bool CAST, bool SWZ>
static int rq_grid(int ngroups) {
  static int cap = 0;
  if (cap == 0) {
    cap = num_sms() * occupancy(rq_kernel<CAST, SWZ>, RQ_THR, RQ_SMEM);
  }
  return ngroups < cap ? ngroups : cap;
}

template <bool CAST, bool SWZ>
static void launch_rq_t(RqParams p, cudaStream_t stream) {
  const int grid = rq_grid<CAST, SWZ>(p.M / 16);

  const int big = p.M / 64;
  p.nbig = grid < big ? grid : big;
  rq_kernel<CAST, SWZ><<<grid, RQ_THR, RQ_SMEM, stream>>>(p);
}

cudaError_t launch_rmsnorm_quant(const RmsQuantDesc& d, cudaStream_t stream) {
  if (d.M <= 0 || d.M % 16 != 0 || d.N != RQ_H) return cudaErrorInvalidValue;
  if (d.cast && !(d.do_row && d.do_col)) return cudaErrorInvalidValue;
  RqParams p;
  p.M = d.M;
  p.x = reinterpret_cast<const __nv_bfloat16*>(d.x);
  p.gamma = reinterpret_cast<const __nv_bfloat16*>(d.gamma);
  p.rstd = d.rstd;
  p.eps = d.eps;
  p.rht = reinterpret_cast<const __nv_bfloat16*>(d.rht);
  p.stf = d.state_f32;
  p.sti = d.state_i32;
  p.factor = d.factor;
  p.row_data = d.row_data;
  p.row_scale = d.row_scale;
  p.row_amax = d.row_amax;
  p.col_data = d.col_data;
  p.col_scale = d.col_scale;
  p.col_amax = d.col_amax;
  p.nbig = 0;
  if (!d.cast) {
    p.rstd = nullptr;
    launch_rq_t<false, true>(p, stream);
  } else if (d.swizzled) {
    launch_rq_t<true, true>(p, stream);
  } else {
    launch_rq_t<true, false>(p, stream);
  }
  return cudaGetLastError();
}

}
