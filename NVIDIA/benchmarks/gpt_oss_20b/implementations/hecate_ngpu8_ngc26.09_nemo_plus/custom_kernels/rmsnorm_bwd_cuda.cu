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

// Single-sweep fused RMSNorm backward for gpt-oss-20b (hidden 2880), CUDA C++.
//
// Replaces the Triton pair (_rmsnorm_bwd_fused_kernel: two sweeps over N, the second re-reading dy and x from
// DRAM; _colsum_kernel over [M/32, N|2N] fp32 partials) with ONE kernel that reads dy and x exactly once:
//
//   * one CTA per ROWS rows (ROWS = 8 or 4), a thread-block CLUSTER of 32/ROWS CTAs per 32-row MXFP8 column block;
//     the CTA's ROWS x N bf16 slab of dy and x is loaded by TMA bulk copies (cp.async.bulk, one mbarrier) and stays
//     resident in shared memory for the whole kernel (92 KB for ROWS = 8 -> 2 CTAs per SM);
//   * sweep 1 (shared memory only): per-row sum_j dy*g*x -> k_r = sum * rstd^3 / N;
//   * sweep 2 (shared memory + the streaming dres read): dx = rstd*dy*g - x*k (+ dres) -> bf16 -> global out and, in
//     place of dy, back into shared memory; per-column dgamma (sum dy*x*rstd), dbias (sum of the bf16-rounded output)
//     and the columnwise |out| amax over the CTA's rows, all in registers (thread t owns columns 8t..8t+7);
//   * the cluster reduces the per-CTA column partials through DSMEM to one [N|2N] fp32 row per 32-row block and
//     computes the 32-row columnwise MXFP8 amax the same way; sweep 3 re-reads the bf16 output from shared memory and
//     emits the MXFP8 rowwise (1x32) and columnwise (32x1) data + e8m0 scales in TE's GEMM-swizzled layout;
//   * the dgamma/dbias column reduction is finished IN-KERNEL (no colsum launch): per-CTA last-arrives tickets over the
//     [M/32, N|2N] partials in a 3-level tree (16 row blocks -> level-1 group partial, 8 groups -> level-2 partial, the
//     last CTA -> bf16), every level a fixed-order sum (deterministic) by the last-arriving CTA alone, written straight
//     into the targets -- the parameters' main_grad slots when the hook resolves them (grad_added_to_main_grad
//     protocol), with an optional accumulate.  The tickets re-arm themselves, so a CUDA-graph replay needs no memset.
//     The dres rows are loaded (all ROWS 16-B loads per thread in flight) at CTA start, overlapping the slab's TMA load.
//
// MXFP8 semantics are TE's (verified bitwise against TE's MXFP8Quantizer): e8m0 = biased exponent of
// the power of two >= amax / fp8_max (round up, satfinite), data = cvt.rn.satfinite of (x * 2^(127 - e8m0)) computed on
// the bf16-ROUNDED gradient; padding blocks / columns of both scale matrices are written as 0.
//
// Sample inputs (the workload shape, per rank): M = 24576, N = 2880, bf16; norm1 (+dres) and the final norm (plain) run
// without MXFP8 / dbias, norm2 with dres + MXFP8 row/col + dbias.
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "rmsnorm_bwd_cuda.h"

namespace cg = cooperative_groups;

namespace pbn {
namespace {

// ------------------------------------------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

// ------------------------------------------------------------------------------------------------- numeric helpers
__device__ __forceinline__ float bf2f(uint32_t packed, int hi) {
  return __uint_as_float(hi ? (packed & 0xFFFF0000u) : (packed << 16));
}
__device__ __forceinline__ uint16_t f2bf(float f) {
  __nv_bfloat16 b = __float2bfloat16_rn(f);
  return *reinterpret_cast<uint16_t*>(&b);
}
__device__ __forceinline__ float bf16bits2f(uint16_t h) { return __uint_as_float(((uint32_t)h) << 16); }

// TE float_to_e8m0(amax * (1/fp8_max)): biased exponent of the power of two >= v (round up), satfinite; 0 for v == 0.
__device__ __forceinline__ int e8m0_from_amax(float amax, float fp8_max_rcp) {
  const float v = amax * fp8_max_rcp;
  const uint32_t bits = __float_as_uint(v);
  const int exp = (int)((bits >> 23) & 0xFFu);
  const uint32_t mant = bits & 0x7FFFFFu;
  const bool inc = (mant > 0u) && (exp != 0xFE) && ((exp != 0) || (mant > 0x400000u));
  const int e = exp + (inc ? 1 : 0);
  return (v == 0.0f) ? 0 : e;
}
// TE exp2f_rcp(e8m0): 1 for e == 0, else 2^(127 - e)
__device__ __forceinline__ float scale_rcp(int e) {
  float r = __uint_as_float(((uint32_t)(254 - e)) << 23);
  if (e == 0) r = 1.0f;
  if (e >= 254) r = 5.877471754111438e-39f;
  return r;
}
// TE gemm_swizzled_scale_idx(i, j, num_tiles_X): flat index in the cuBLAS 128x4 block-scaling tile layout
__device__ __forceinline__ uint32_t swz(int i, int j, int ntx) {
  const int iy = i & 127;
  return ((uint32_t)((i >> 7) * ntx + (j >> 2))) * 512u + (uint32_t)((iy & 31) * 16 + (iy >> 5) * 4 + (j & 3));
}
template <bool E5M2>
__device__ __forceinline__ uint16_t cvt_fp8x2(float lo, float hi) {
  return (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, E5M2 ? __NV_E5M2 : __NV_E4M3);
}

__device__ __forceinline__ float4 ld_cg_f4(const float* p) { return __ldcg(reinterpret_cast<const float4*>(p)); }

// distributed shared memory, push model: explicit cluster-window addressing (mapa + st.shared::cluster)
__device__ __forceinline__ uint32_t mapa_u32(uint32_t smem_addr, int rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_dsmem_f4(uint32_t addr, float4 v) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void st_dsmem_u4(uint32_t addr, uint4 v) {
  asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* gptr, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(gptr), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);  // cvt.rn.bf16x2.f32: .x = lo (low 16 bits), .y = hi
  return *reinterpret_cast<uint32_t*>(&h);
}

// ------------------------------------------------------------------------------------------------- shared memory map
// MX    : [dy slab ROWS*N bf16 -- after sweep 2 the bf16 output][x slab ROWS*N bf16 -- after sweep 2 the cluster inboxes:
//          colamax inbox [CL][N] bf16 | dgamma/dbias slice inbox [CL][2][N/CL] fp32][misc]
// non-MX: [dy slab -- then the bf16 output][x slab][dgamma slice inbox [CL][N/CL] fp32][misc]
// The inboxes are filled by the PEERS (st.shared::cluster) before the one full cluster barrier; afterwards every CTA
// reads only its own shared memory and may exit freely.  For N = 2880 and ROWS = 8 the MX inboxes fill the x slab
// exactly (46080 B); the non-MX inbox is dedicated (11520 B), so it needs no rendezvous before the pushes.
__host__ __device__ constexpr int inbox_bytes(int rows, int N, bool mx) {
  const int cl = 32 / rows;
  return (mx ? cl * N * 2 : 0) + cl * (mx ? 2 : 1) * (N / cl) * 4;
}
__host__ __device__ constexpr int x_region_bytes(int rows, int N, bool mx) {
  const int ib = inbox_bytes(rows, N, mx);
  return mx ? ((rows * N * 2 > ib) ? rows * N * 2 : ib) : rows * N * 2 + ib;
}
__host__ __device__ constexpr int misc_bytes(int rows) {
  return 16 /*mbarrier*/ + 12 * rows * 4 /*warp row sums*/ + rows * 4 /*k*/ + rows * 4 /*rstd*/ + 16 /*flags*/;
}
__host__ __device__ constexpr size_t smem_bytes_c(int rows, int N, bool mx) {
  return (size_t)rows * N * 2 + (size_t)x_region_bytes(rows, N, mx) + (size_t)misc_bytes(rows);
}

// one CTA alone sums `nrows` fp32 rows of stride PS (fixed order, deterministic) into dst (fp32) -- the ticket tree
__device__ __forceinline__ void reduce_rows_f32(const float* base, int nrows, int PS, float* dst) {
  for (int i = threadIdx.x; i < (PS >> 2); i += kThreads) {
    const int c = i * 4;
    float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int j = 0;
    for (; j + 4 <= nrows; j += 4) {  // 4 independent loads in flight per thread
      const float4 v0 = ld_cg_f4(base + (size_t)(j + 0) * PS + c);
      const float4 v1 = ld_cg_f4(base + (size_t)(j + 1) * PS + c);
      const float4 v2 = ld_cg_f4(base + (size_t)(j + 2) * PS + c);
      const float4 v3 = ld_cg_f4(base + (size_t)(j + 3) * PS + c);
      s.x += v0.x; s.y += v0.y; s.z += v0.z; s.w += v0.w;
      s.x += v1.x; s.y += v1.y; s.z += v1.z; s.w += v1.w;
      s.x += v2.x; s.y += v2.y; s.z += v2.z; s.w += v2.w;
      s.x += v3.x; s.y += v3.y; s.z += v3.z; s.w += v3.w;
    }
    for (; j < nrows; ++j) {
      const float4 v = ld_cg_f4(base + (size_t)j * PS + c);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    *reinterpret_cast<float4*>(dst + c) = s;
  }
}

// ticket, blocking form (levels 2 and 3): this CTA's stores are fenced; true (CTA-uniform) when it was the last arrival
__device__ __forceinline__ bool ticket_last(int* counter, int target, int* s_flag) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const int old = atomicAdd(counter, 1);
    const int last = (old == target - 1) ? 1 : 0;
    if (last) *counter = 0;  // re-arm: every arrival is in (the kernel boundary publishes it to the next launch)
    __threadfence();
    *s_flag = last;
  }
  __syncthreads();
  return *s_flag != 0;
}

// ------------------------------------------------------------------------------------------------- the kernel
template <int ROWS, bool HAS_RES, bool MX, bool E5M2>
__global__ void __launch_bounds__(kThreads, (ROWS == 4 ? 3 : 2)) rmsnorm_bwd_single_sweep_kernel(const Params p) {
  constexpr int CL = 32 / ROWS;  // cluster size = CTAs per 32-row block
  constexpr int NWARP = kThreads / 32;
  constexpr int NARR = MX ? 2 : 1;  // dgamma (| dbias) slices in the inbox
  extern __shared__ __align__(128) uint8_t smem[];

  const int N = p.N;
  const int NV = N >> 3;  // 8-column groups
  const int cslice = N / CL;  // this CTA's column slice of the partial row
  uint16_t* s_dy = reinterpret_cast<uint16_t*>(smem);
  uint16_t* s_x = reinterpret_cast<uint16_t*>(smem + (size_t)ROWS * N * 2);
  // inboxes: MX aliases the x slab (colamax [CL][N] bf16, then the dgamma/dbias slices); non-MX sits after the x slab
  uint16_t* s_cm_in = reinterpret_cast<uint16_t*>(smem + (size_t)ROWS * N * 2);                 // [CL][N] bf16 (MX)
  float* s_dg_in = reinterpret_cast<float*>(smem + (size_t)ROWS * N * 2 + (MX ? (size_t)CL * N * 2 : (size_t)ROWS * N * 2));
  uint8_t* misc = smem + (size_t)ROWS * N * 2 + x_region_bytes(ROWS, N, MX);
  uint64_t* mbar = reinterpret_cast<uint64_t*>(misc);
  float* s_red = reinterpret_cast<float*>(misc + 16);
  float* s_k = s_red + NWARP * ROWS;
  float* s_rstd = s_k + ROWS;
  int* s_flag = reinterpret_cast<int*>(s_rstd + ROWS);

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = (int)cluster.block_rank();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int blk = blockIdx.x;
  const int blk32 = blk / CL;
  const int row0 = blk * ROWS;
  const bool owner = tid < NV;  // this thread owns columns 8*tid .. 8*tid+7
  const int c0 = tid * 8;
  const int PS = p.pstride;

  // ---------------------------------------------------------------- phase 0: TMA bulk loads of the dy / x slab, L2 prefetch of dres
  if (tid == 0) {
    mbar_init(mbar, 1);
    fence_barrier_init();
  }
  __syncthreads();
  // DSMEM may be touched only once every CTA of the cluster has started: the first half of a split cluster barrier
  // (non-MX: now, the inbox is dedicated; MX: after sweep 2, when the x slab the inbox aliases is dead)
  if (!MX) asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  if (tid == 0 && !(p.dbg & 4)) {
    mbar_expect_tx(mbar, (uint32_t)(2 * ROWS * N * 2));
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      bulk_g2s(s_dy + r * N, p.dy + (size_t)(row0 + r) * N, (uint32_t)(N * 2), mbar);
      bulk_g2s(s_x + r * N, p.x + (size_t)(row0 + r) * N, (uint32_t)(N * 2), mbar);
    }
  }
  // the residual-gradient rows: all ROWS 16-B loads in flight at once, overlapping the slab's TMA load and sweep 1
  uint4 dres_v[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    dres_v[r] = make_uint4(0u, 0u, 0u, 0u);
    if (HAS_RES && owner) dres_v[r] = __ldg(reinterpret_cast<const uint4*>(p.dres + (size_t)(row0 + r) * N + c0));
  }
  if (tid < ROWS) s_rstd[tid] = p.rstd[row0 + tid];
  float g[8];
  {
    uint4 gv = make_uint4(0u, 0u, 0u, 0u);
    if (owner) gv = __ldg(reinterpret_cast<const uint4*>(p.gamma + c0));
    const uint32_t gw[4] = {gv.x, gv.y, gv.z, gv.w};
    const float add = p.zero_centered ? 1.0f : 0.0f;
#pragma unroll
    for (int c = 0; c < 8; ++c) g[c] = bf2f(gw[c >> 1], c & 1) + add;
  }
  __syncthreads();  // s_rstd
  if (!(p.dbg & 4)) mbar_wait(mbar, 0);

  // ---------------------------------------------------------------- sweep 1: per-row sum dy * g * x (shared memory)
  {
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    if (owner) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint4 a = *reinterpret_cast<const uint4*>(s_dy + r * N + c0);
        const uint4 b = *reinterpret_cast<const uint4*>(s_x + r * N + c0);
        const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
        const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c) sum += bf2f(aw[c >> 1], c & 1) * g[c] * bf2f(bw[c >> 1], c & 1);
        acc[r] = sum;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) s_red[warp * ROWS + r] = acc[r];
    }
    __syncthreads();
    if (tid < ROWS) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < NWARP; ++w) sum += s_red[w * ROWS + tid];
      const float rs = s_rstd[tid];
      s_k[tid] = sum * (rs * rs * rs) / (float)N;
    }
    __syncthreads();
  }

  // ---------------------------------------------------------------- sweep 2: dx (+dres) -> bf16, in place of dy in shared
  // memory; per-column dgamma over the CTA's rows in registers, pushed to the owning peer's inbox
  float dg_keep[8];  // MX: the dgamma partial travels with the sweep-3a pushes (after the split barrier's wait)
  {
    float dg[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) dg[c] = 0.0f;
    if (owner) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint4 a = *reinterpret_cast<const uint4*>(s_dy + r * N + c0);
        const uint4 b = *reinterpret_cast<const uint4*>(s_x + r * N + c0);
        const uint4 d = dres_v[r];
        const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
        const uint32_t bw[4] = {b.x, b.y, b.z, b.w};
        const uint32_t dw[4] = {d.x, d.y, d.z, d.w};
        const float rs = s_rstd[r];
        const float k = s_k[r];
        uint32_t ow[4];
#pragma unroll
        for (int c = 0; c < 8; c += 2) {
          float v2[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float dyf = bf2f(aw[c >> 1], e);
            const float xf = bf2f(bw[c >> 1], e);
            float v = rs * (dyf * g[c + e]) - xf * k;
            if (HAS_RES) v += bf2f(dw[c >> 1], e);
            v2[e] = v;
            dg[c + e] += dyf * xf * rs;
          }
          ow[c >> 1] = pack_bf16x2(v2[0], v2[1]);
        }
        *reinterpret_cast<uint4*>(s_dy + r * N + c0) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
      }
    }
    __syncthreads();  // every read of the x slab is done
    if (MX) asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");  // my x slab is dead: peers may push into it
    if (!MX) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // every peer has started (arrived at its start)
      if (owner) {
        const int q = c0 / cslice;  // the peer whose column slice holds my 8 columns
        const uint32_t dst = mapa_u32(smem_u32(s_dg_in + (rank * NARR + 0) * cslice + (c0 - q * cslice)), q);
        st_dsmem_f4(dst, make_float4(dg[0], dg[1], dg[2], dg[3]));
        st_dsmem_f4(dst + 16u, make_float4(dg[4], dg[5], dg[6], dg[7]));
      }
    } else {
      dg_keep[0] = dg[0]; dg_keep[1] = dg[1]; dg_keep[2] = dg[2]; dg_keep[3] = dg[3];
      dg_keep[4] = dg[4]; dg_keep[5] = dg[5]; dg_keep[6] = dg[6]; dg_keep[7] = dg[7];
    }
  }

  // ---------------------------------------------------------------- sweep 3a (MX): per-column dbias and |out| amax over the
  // CTA's rows from the bf16 output in shared memory (local work that hides the split barrier), then the pushes:
  // amax (bf16-exact) to every peer, dbias / dgamma slices to the owning peer
  if (MX) {
    float db[8], cm[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      db[c] = 0.0f;
      cm[c] = 0.0f;
    }
    if (owner) {
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint4 a = *reinterpret_cast<const uint4*>(s_dy + r * N + c0);
        const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const float of = bf2f(aw[c >> 1], c & 1);
          db[c] += of;
          cm[c] = fmaxf(cm[c], fabsf(of));
        }
      }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // every peer's x slab is dead: push
    if (owner) {
      uint32_t cmw[4];
#pragma unroll
      for (int c = 0; c < 8; c += 2) cmw[c >> 1] = (__float_as_uint(cm[c]) >> 16) | (__float_as_uint(cm[c + 1]) & 0xFFFF0000u);  // exact: |bf16|
      const uint4 cmv = make_uint4(cmw[0], cmw[1], cmw[2], cmw[3]);
      const uint32_t cm_local = smem_u32(s_cm_in + rank * N + c0);
#pragma unroll
      for (int q = 0; q < CL; ++q) st_dsmem_u4(mapa_u32(cm_local, q), cmv);
      const int q = c0 / cslice;
      const uint32_t dst = mapa_u32(smem_u32(s_dg_in + (rank * NARR + 0) * cslice + (c0 - q * cslice)), q);
      st_dsmem_f4(dst, make_float4(dg_keep[0], dg_keep[1], dg_keep[2], dg_keep[3]));
      st_dsmem_f4(dst + 16u, make_float4(dg_keep[4], dg_keep[5], dg_keep[6], dg_keep[7]));
      const uint32_t dstb = mapa_u32(smem_u32(s_dg_in + (rank * NARR + 1) * cslice + (c0 - q * cslice)), q);
      st_dsmem_f4(dstb, make_float4(db[0], db[1], db[2], db[3]));
      st_dsmem_f4(dstb + 16u, make_float4(db[4], db[5], db[6], db[7]));
    }
  }
  cluster.sync();  // THE cluster barrier: every push has landed; from here on only local shared memory is read

  // ---------------------------------------------------------------- partial row of this 32-row block (my column slice)
  if (!(p.dbg & 1)) {
    float* part_row = p.part + (size_t)blk32 * PS;
    for (int i = tid; i < (NARR * cslice) >> 2; i += kThreads) {
      const int arr = (i * 4) / cslice;        // 0 = dgamma, 1 = dbias
      const int cc = (i * 4) - arr * cslice;   // column inside the slice
      float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
      for (int q = 0; q < CL; ++q) {
        const float4 v = *reinterpret_cast<const float4*>(s_dg_in + (q * NARR + arr) * cslice + cc);
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
      }
      *reinterpret_cast<float4*>(part_row + arr * N + rank * cslice + cc) = sum;
    }
  }
  // the level-1 ticket, issued now (thread 0 alone, async): its fence and the atomic's latency overlap sweep 3b below
  __syncthreads();
  const int g1 = blk32 / p.group;
  const int gsize1 = min(p.group, p.nb32 - g1 * p.group);
  if (tid == 0) {
    int last = 0;
    if (!(p.dbg & 1)) {
      __threadfence();
      const int old = atomicAdd(p.tickets + 1 + g1, 1);
      last = (old == gsize1 * CL - 1) ? 1 : 0;
      if (last) p.tickets[1 + g1] = 0;
    }
    s_flag[0] = last;
  }

  // ---------------------------------------------------------------- sweep 3b: the bf16 output to global (+ MXFP8 rowwise /
  // columnwise quantize of it from shared memory)
  float rcp_c[8];
  if (MX && owner) {
    int ec[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      uint32_t mb = 0u;
#pragma unroll
      for (int q = 0; q < CL; ++q) mb = max(mb, (uint32_t)s_cm_in[q * N + c0 + c]);  // |bf16| compares as its bits
      ec[c] = e8m0_from_amax(__uint_as_float(mb << 16), p.fp8_max_rcp);
      rcp_c[c] = scale_rcp(ec[c]);
    }
    if (c0 / cslice == rank) {  // this rank writes the column scales of its slice (one writer per column)
#pragma unroll
      for (int c = 0; c < 8; ++c) p.cs[swz(c0 + c, blk32, p.ntx_c)] = (uint8_t)ec[c];
    }
  }
  if (MX && rank == 0) {  // TE's zero padding of the columnwise scale matrix (columns N .. npad_c)
    for (int i = tid; i < p.npad_c - N; i += kThreads) p.cs[swz(N + i, blk32, p.ntx_c)] = 0;
  }
  if (!(p.dbg & 2)) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int row = row0 + r;
      uint4 a = make_uint4(0u, 0u, 0u, 0u);
      if (owner) a = *reinterpret_cast<const uint4*>(s_dy + r * N + c0);
      if (owner && !(p.dbg & 8)) *reinterpret_cast<uint4*>(p.out + (size_t)row * N + c0) = a;
      if (MX) {
        const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
        float of[8];
        float am = 0.0f;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          of[c] = bf2f(aw[c >> 1], c & 1);
          am = fmaxf(am, fabsf(of[c]));
        }
        // the 32-column rowwise block = 4 consecutive lanes
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        const int er = e8m0_from_amax(am, p.fp8_max_rcp);
        const float rr = scale_rcp(er);
        if (owner) {
          uint16_t qr[4], qc[4];
#pragma unroll
          for (int c = 0; c < 8; c += 2) {
            qr[c >> 1] = cvt_fp8x2<E5M2>(of[c] * rr, of[c + 1] * rr);
            qc[c >> 1] = cvt_fp8x2<E5M2>(of[c] * rcp_c[c], of[c + 1] * rcp_c[c + 1]);
          }
          const uint2 vr = make_uint2((uint32_t)qr[0] | ((uint32_t)qr[1] << 16), (uint32_t)qr[2] | ((uint32_t)qr[3] << 16));
          const uint2 vc = make_uint2((uint32_t)qc[0] | ((uint32_t)qc[1] << 16), (uint32_t)qc[2] | ((uint32_t)qc[3] << 16));
          *reinterpret_cast<uint2*>(p.rd + (size_t)row * N + c0) = vr;
          *reinterpret_cast<uint2*>(p.cd + (size_t)row * N + c0) = vc;
          if ((tid & 3) == 0) p.rs[swz(row, tid >> 2, p.ntx_r)] = (uint8_t)er;
        } else {
          const int j = tid - NV;  // TE's zero padding of the rowwise scale matrix (blocks N/32 .. nblk_pad)
          const int npad = p.nblk_pad - (N >> 5);
          if (j < npad) p.rs[swz(row, (N >> 5) + j, p.ntx_r)] = 0;
        }
      }
    }
  }

  // ---------------------------------------------------------------- in-kernel dgamma / dbias reduction: a 3-level tree of
  // fixed-order sums (deterministic), each level finished by the last-arriving CTA alone
  __syncthreads();
  if (s_flag[0] == 0) return;
  if (tid == 0) __threadfence();  // acquire side of the level-1 ticket (observed by thread 0 above)
  __syncthreads();
  reduce_rows_f32(p.part + (size_t)g1 * p.group * PS, gsize1, PS, p.part2 + (size_t)g1 * PS);
  const int g2 = g1 / kFan2;  // level-2 group of level-1 groups
  const int gsize2 = min(kFan2, p.ngroups - g2 * kFan2);
  if (!ticket_last(p.tickets + 1 + kMaxGroups + g2, gsize2, s_flag)) return;
  reduce_rows_f32(p.part2 + (size_t)g2 * kFan2 * PS, gsize2, PS, p.part3 + (size_t)g2 * PS);
  const int ngroups2 = (p.ngroups + kFan2 - 1) / kFan2;
  if (!ticket_last(p.tickets, ngroups2, s_flag)) return;
  // final: sum the level-2 partials (fixed order) and write bf16 into the targets
  for (int i = tid; i < (PS >> 2); i += kThreads) {
    const int c = i * 4;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int gg = 0; gg < ngroups2; ++gg) {
      const float4 v = ld_cg_f4(p.part3 + (size_t)gg * PS + c);
      sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    const float sv[4] = {sum.x, sum.y, sum.z, sum.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int cc = c + e;
      uint16_t* dst = (cc < N) ? (p.dgamma_out + cc) : (p.dbias_out + (cc - N));
      float v = sv[e];
      if (p.accumulate) v += bf16bits2f(*dst);
      *dst = f2bf(v);
    }
  }
}

// ------------------------------------------------------------------------------------------------- host side
template <int ROWS, bool HAS_RES, bool MX, bool E5M2>
cudaError_t launch_t(const Params& p, cudaStream_t stream) {
  auto* kfn = rmsnorm_bwd_single_sweep_kernel<ROWS, HAS_RES, MX, E5M2>;
  const size_t smem = smem_bytes_c(ROWS, p.N, MX);
  static bool attr_set = false;  // one instantiation per (ROWS, HAS_RES, MX, E5M2): set the opt-in once
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
    attr_set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(p.M / ROWS), 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 32 / ROWS;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kfn, p);
}

template <int ROWS, bool HAS_RES, bool MX, bool E5M2>
cudaError_t attrs_t(int N, int* num_regs, int* blocks_per_sm, int* dyn_smem, int* local_bytes, int* max_clusters) {
  auto* kfn = rmsnorm_bwd_single_sweep_kernel<ROWS, HAS_RES, MX, E5M2>;
  const size_t smem = smem_bytes_c(ROWS, N, MX);
  cudaError_t e = cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (e != cudaSuccess) return e;
  cudaFuncAttributes fa;
  e = cudaFuncGetAttributes(&fa, kfn);
  if (e != cudaSuccess) return e;
  *num_regs = fa.numRegs;
  *local_bytes = (int)fa.localSizeBytes;
  *dyn_smem = (int)smem;
  int nb = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kfn, kThreads, smem);
  if (e != cudaSuccess) return e;
  *blocks_per_sm = nb;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(32 / ROWS, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 32 / ROWS;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int nc = 0;
  e = cudaOccupancyMaxActiveClusters(&nc, kfn, &cfg);
  if (e != cudaSuccess) return e;
  *max_clusters = nc;
  return cudaSuccess;
}

template <int ROWS, bool HAS_RES>
cudaError_t launch_mx(const Params& p, bool mx, bool e5m2, cudaStream_t s) {
  if (!mx) return launch_t<ROWS, HAS_RES, false, false>(p, s);
  return e5m2 ? launch_t<ROWS, HAS_RES, true, true>(p, s) : launch_t<ROWS, HAS_RES, true, false>(p, s);
}
template <int ROWS>
cudaError_t launch_rows(const Params& p, bool has_res, bool mx, bool e5m2, cudaStream_t s) {
  return has_res ? launch_mx<ROWS, true>(p, mx, e5m2, s) : launch_mx<ROWS, false>(p, mx, e5m2, s);
}
template <int ROWS, bool HAS_RES>
cudaError_t attrs_mx(bool mx, bool e5m2, int N, int* a, int* b, int* c, int* d, int* e) {
  if (!mx) return attrs_t<ROWS, HAS_RES, false, false>(N, a, b, c, d, e);
  return e5m2 ? attrs_t<ROWS, HAS_RES, true, true>(N, a, b, c, d, e) : attrs_t<ROWS, HAS_RES, true, false>(N, a, b, c, d, e);
}
template <int ROWS>
cudaError_t attrs_rows(bool has_res, bool mx, bool e5m2, int N, int* a, int* b, int* c, int* d, int* e) {
  return has_res ? attrs_mx<ROWS, true>(mx, e5m2, N, a, b, c, d, e) : attrs_mx<ROWS, false>(mx, e5m2, N, a, b, c, d, e);
}

}  // namespace

size_t smem_bytes(int rows, int N, bool mx) { return rows == 4 ? smem_bytes_c(4, N, mx) : smem_bytes_c(8, N, mx); }

cudaError_t launch(const Params& p, int rows, bool has_res, bool mxfp8, bool e5m2, cudaStream_t stream) {
  if (rows == 8) return launch_rows<8>(p, has_res, mxfp8, e5m2, stream);
  if (rows == 4) return launch_rows<4>(p, has_res, mxfp8, e5m2, stream);
  return cudaErrorInvalidValue;
}

cudaError_t attrs(int rows, bool has_res, bool mxfp8, bool e5m2, int N, int* num_regs, int* blocks_per_sm, int* dyn_smem,
                  int* local_bytes, int* max_clusters) {
  if (rows == 8) return attrs_rows<8>(has_res, mxfp8, e5m2, N, num_regs, blocks_per_sm, dyn_smem, local_bytes, max_clusters);
  if (rows == 4) return attrs_rows<4>(has_res, mxfp8, e5m2, N, num_regs, blocks_per_sm, dyn_smem, local_bytes, max_clusters);
  return cudaErrorInvalidValue;
}

}  // namespace pbn
