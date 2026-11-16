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

// MoE token gather fused with the dual-orientation MXFP8 quantization (HybridEP dense layout), CUDA C++.
// Rewrites the Triton kernels of moe_qperm.py (_qperm_gather_quant_kernel, the forward, and
// _qperm_gather_quant_scaled_kernel, the combine backward) with the same bytes out:
//
//   rowwise_data    [P, H] e4m3   blocks of 32 along H,  scales GEMM-swizzled over one global (P/128, H/128) tile grid
//   columnwise_data [P, H] e4m3   blocks of 32 along P,  scales in TE's grouped storage (one swizzled block per expert)
//   scale bytes e = TE float_to_e8m0(amax * (1/448)); scale_inv = 2^(127 - e); q = cvt.rn.satfinite.e4m3 (x * scale_inv)
//   backward (SCALED): x = bf16(prob[p] * dres[row_to_token[p]]), dprob[p] = dres[tok] . z[p] (fp32, fixed order via a
//   [KT, P] partial buffer + a reduce kernel), optional bf16 x rows, optional per-expert fp32 column sum of x (fc2 dbias)
//
// Design: one CTA = 4 warps = 128 threads, tile = 32 rows x 128 columns (one 32-row
// columnwise block x one 128-column scale tile), the bf16 tile staged in shared memory by asynchronous copies --
// LOAD_MODE 1: 16-B-granular TMA 1-D bulk copies (cp.async.bulk, one per gathered row segment, mbarrier transaction
// count), LOAD_MODE 0: per-thread 16-B cp.async (zero-fill for padding rows) -- NSTAGE-deep ring (2/3/4), persistent CTAs
// (grid = resident capacity) over 4-tile units (128 rows x one scale-tile column).
//
// Unit scheduling: the units are handed out DYNAMICALLY -- thread 0
// of a CTA atomicAdd's a per-launch global counter (Params::sched) to grab the next slot of `grab` consecutive units
// (grab 2 = the (kt, half) pair of one 256-row chunk, the column-sum flush cadence of the old contiguous partition) and
// broadcasts it through shared memory; a slot is grabbed exactly when the NSTAGE ring is about to prefetch its first
// tile, so a CTA never holds more than one unstarted slot.  A CTA's slots still arrive in increasing unit order, so the
// expert walk and the column-sum flush logic are unchanged.  The last CTA to finish resets the counter (graph-replay
// safe, no memset per launch).  The old static partition (u_begin/u_end from blockIdx over a grid sized for an empty
// GPU) ran a CTA's whole fixed share as a second wave whenever another kernel (the NCCL reduce-scatter, a WGRAD-stream
// GEMM, the optimizer's copies) held SMs: in situ the backward ran at 1.47-1.66x its isolated time, the forward at
// 1.15-1.36x.  sched == nullptr keeps the static partition (A/B reference).
// Thread (c = tid % 16, rg = tid / 16) owns the 16-B column chunk c (8 columns) of rows 4rg..4rg+3: the rowwise amax
// of a 32-column block is 4 lanes (xor 1, 2 shuffles), the columnwise amax of the 32-row slab is the thread's 4 rows,
// a xor-16 shuffle (8 rows per warp) and one 1-KB shared-memory exchange across the 4 warps; one thread per column
// then derives that column's e8m0 and scale_inv once (shared with the slab).  Scale bytes are staged per 4-tile unit
// (128 rows = one swizzled 512-B tile) and written as 16-B stores.  Padding rows (row_to_token < 0) are zero-filled
// in shared memory, so their data/scales come out 0x00 exactly like TE's quantization of HybridEP's zero rows.
//
// Sample inputs (workload shape, per rank): T = 24576, H = 2880, E = 32, top-4, PAD = 256, P = 118016,
// n_used = sum(round_up(tokens_per_expert, 256)) ~ 102400.  No torch dependency in this file.
#include <cuda_runtime.h>
#include <stdint.h>

#include "moe_qperm_cuda.h"

namespace pbq {

constexpr int ROWS = 32;              // rows per tile = one columnwise block
constexpr int BK = 128;               // columns per tile = one scale tile column
constexpr int NCH = BK / 8;           // 16-B chunks per row
constexpr int NT = 128;               // threads per CTA: NCH chunks x 8 row groups of 4 rows
constexpr int ROW_BYTES = BK * 2;     // 256
constexpr int TILE_BYTES = ROWS * ROW_BYTES;   // 8192
constexpr int MAX_E = 256;
constexpr int XCHG_BYTES = 4 * BK * 2;         // per-warp column amax partials, bf16
constexpr int SINV_BYTES = BK * 4;             // columnwise scale_inv per column, fp32
constexpr int STAGE_SCALE_BYTES = 512;         // one swizzled scale tile
constexpr int CSUM_BYTES = 8 * BK * 4;         // column-sum partials per row group, fp32

struct SmemLayout {
  int data, zdata, bars, xchg, sinv, rws, cws, pc, starts, csum, misc, total;
};

__host__ __device__ constexpr SmemLayout smem_layout(bool scaled, int nstage, bool colsum) {
  SmemLayout L{};
  int off = 0;
  L.data = off;   off += nstage * TILE_BYTES;
  L.zdata = off;  off += scaled ? nstage * TILE_BYTES : 0;
  L.bars = off;   off += nstage * 8;
  off = (off + 127) & ~127;
  L.xchg = off;   off += XCHG_BYTES;
  L.sinv = off;   off += SINV_BYTES;
  L.rws = off;    off += STAGE_SCALE_BYTES;
  L.cws = off;    off += STAGE_SCALE_BYTES;
  L.pc = off;     off += MAX_E * 4;
  L.starts = off; off += MAX_E * 4;
  L.csum = off;   off += (scaled && colsum) ? CSUM_BYTES : 0;
  L.misc = off;   off += 128;
  L.total = off;
  return L;
}

// ------------------------------------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT;\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void tma_bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}

__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ uint4 lds128(const uint8_t* smem, int off) {
  return *reinterpret_cast<const uint4*>(smem + off);
}

__device__ __forceinline__ void sts128(uint8_t* smem, int off, uint4 v) {
  *reinterpret_cast<uint4*>(smem + off) = v;
}

__device__ __forceinline__ uint32_t bmax2(uint32_t a, uint32_t b) {   // packed bf16x2 max (abs values: monotonic)
  uint32_t d;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

__device__ __forceinline__ uint32_t pack_bf16x2_rn(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ uint32_t cvt_e4m3x2(float lo, float hi) {   // byte 0 = lo, byte 1 = hi
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return static_cast<uint32_t>(r);
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// TE float_to_e8m0(v): biased exponent of v rounded UP to the next power of two (0 -> 0, inf -> 0xFE, nan -> 0xFF)
__device__ __forceinline__ uint32_t e8m0_int(float v) {
  uint32_t b = __float_as_uint(v);
  uint32_t exp = b >> 23;
  uint32_t man = b & 0x7FFFFFu;
  uint32_t e = exp + ((man != 0u) & (exp != 254u));
  if (exp == 255u) e = man ? 255u : 254u;
  return e;
}

// two e8m0 bytes: byte 0 = lo, byte 1 = hi (integer path everywhere: cvt.rp.satfinite.ue8m0x2 is an architecture-
// specific sm_100a/sm_10xf feature that ptxas rejects for the generic sm_1xx targets this extension is built for)
__device__ __forceinline__ uint32_t e8m0_pair(float lo, float hi) {
  return e8m0_int(lo) | (e8m0_int(hi) << 8);
}

// TE exp2f_rcp(e) = 2^(127 - e), 1.0 for e == 0 (denormal for e >= 254)
__device__ __forceinline__ float sinv_from_e(uint32_t e) {
  uint32_t sb = (254u - e) << 23;
  if (e == 0u) sb = 0x3F800000u;
  if (e >= 254u) sb = 0x00400000u >> (e - 254u);
  return __uint_as_float(sb);
}

__device__ __forceinline__ float warp_sum16(float v) {   // sum over the 16 lanes of a half-warp
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  return v;
}

// 4 fp32 -> 4 e4m3 bytes (element i in byte i)
__device__ __forceinline__ uint32_t quant4(float x0, float x1, float x2, float x3, float s0, float s1, float s2, float s3) {
  uint32_t lo = cvt_e4m3x2(x0 * s0, x1 * s1);
  uint32_t hi = cvt_e4m3x2(x2 * s2, x3 * s3);
  return prmt(lo, hi, 0x5410u);
}

struct Unit {
  int p0;      // first row of the unit's first tile
  int kt;      // scale tile column
};

// unit order: 256-row chunks, kt-minor (the two 128-row halves of a chunk at the same kt are consecutive), so a
// CTA's consecutive units at one (kt, expert) are as long as possible (column-sum flushes) while neighbouring CTAs
// read the same token rows' other column tiles at about the same time (DRAM locality of the 4x gathered rows)
__device__ __forceinline__ Unit decode_unit(int u, int KT, int nfull) {
  Unit r;
  const int per_chunk = 2 * KT;
  if (u < nfull * per_chunk) {
    const int chunk = u / per_chunk;
    const int rem = u - chunk * per_chunk;
    r.kt = rem >> 1;
    r.p0 = chunk * 256 + (rem & 1) * 128;
  } else {
    const int rem = u - nfull * per_chunk;
    r.kt = rem;
    r.p0 = nfull * 256;
  }
  return r;
}

// dynamic scheduler: every CTA (work or not) counts itself out; the last one re-arms the counter for the next launch.
// Each CTA's grabs of sched[0] happen-before its own increment of sched[1] (same thread, fenced), so when the
// increment returns gridDim.x - 1 no CTA will touch sched[0] again in this launch.
__device__ __forceinline__ void sched_exit(int* sched) {
  if (sched == nullptr) return;
  if (threadIdx.x == 0) {
    __threadfence();
    const int d = atomicAdd(sched + 1, 1);
    if (d == (int)gridDim.x - 1) {
      atomicExch(sched, 0);
      atomicExch(sched + 1, 0);
      __threadfence();
    }
  }
}

// ------------------------------------------------------------------------------------------------ the kernel
// minimum resident CTAs per SM = the residency the shared-memory ring admits (fwd 7 at NSTAGE 3, bwd 5 at NSTAGE 2),
// so the register allocation (fwd <= 72, bwd <= 102) never caps it below that; no spills in either toolchain tried
template <bool SCALED, bool WRITE_BF16, bool COLSUM, int LOAD_MODE, int NSTAGE>
__global__ void __launch_bounds__(NT, (SCALED ? 5 : 7)) qperm_gather_quant_kernel(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr SmemLayout L = smem_layout(SCALED, NSTAGE, COLSUM);
  int32_t* pc = reinterpret_cast<int32_t*>(smem + L.pc);
  int32_t* starts = reinterpret_cast<int32_t*>(smem + L.starts);
  int32_t* misc = reinterpret_cast<int32_t*>(smem + L.misc);
  uint16_t* xchg = reinterpret_cast<uint16_t*>(smem + L.xchg);
  float* sinv_sm = reinterpret_cast<float*>(smem + L.sinv);
  uint8_t* rws_stage = smem + L.rws;
  uint8_t* cws_stage = smem + L.cws;
  float* csum_x = reinterpret_cast<float*>(smem + L.csum);
  // misc[0] rows in use; misc[1] static u_begin; misc[8..11] ring of grabbed slot indices; misc[12] the CTA's tile count
  volatile int32_t* slot_v = misc + 8;
  volatile int32_t* end_v = misc + 12;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int c = tid & (NCH - 1);     // 16-B chunk (8 columns) within the tile row
  const int rg = tid >> 4;           // row group: rows 4rg .. 4rg+3
  const int H = p.H, T = p.T, P = p.P, E = p.E, KT = p.KT;
  const float rcp = p.rcp;
  const bool dyn = (p.sched != nullptr);
  const int grab = dyn ? p.grab : 1;             // units per slot (host-validated 1 or 2)
  const int tps_shift = 2 + (grab >> 1);         // tiles per slot = 4 * grab = 1 << tps_shift
  constexpr int END_UNKNOWN = 0x7FFFFFFF;

  // ---- padded expert counts, exclusive prefix, rows in use
  for (int e = tid; e < E; e += NT) {
    int cnt = p.tpe[e];
    pc[e] = ((cnt + (p.PAD - 1)) / p.PAD) * p.PAD;
  }
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) mbar_init(smem_u32(smem + L.bars + 8 * s), LOAD_MODE == 1 ? 1u : (uint32_t)NT);
    fence_barrier_init();
  }
  __syncthreads();
  if (tid == 0) {
    int s = 0;
    for (int e = 0; e < E; ++e) { starts[e] = s; s += pc[e]; }
    misc[0] = s;
  }
  __syncthreads();
  const int n_used = misc[0];
  const int q128 = n_used / 128;                 // 128-row units in use (n_used % 128 == 0 by construction)
  const int nfull = q128 / 2;
  const int U = q128 * KT;                       // units
  const int W = (U + grab - 1) / grab;           // slots

  // thread 0: grab slot k (= the next `grab` units) and publish it; the first grab past the end (or the short last
  // slot) fixes the CTA's tile count.  Called before barrier B1 of a tile; read after barrier B3 of the same tile.
  auto grab_slot = [&](int k) {
    const int w = atomicAdd(p.sched, 1);
    slot_v[k & 3] = w;
    if (end_v[0] == END_UNKNOWN) {
      if (w >= W) end_v[0] = k << tps_shift;
      else if (grab * w + grab > U) end_v[0] = (k << tps_shift) + 4 * (U - grab * w);
    }
  };
  if (tid == 0) {
    if (dyn) {
      end_v[0] = END_UNKNOWN;
      grab_slot(0);
    } else {
      const int G = gridDim.x;
      const int u_begin = (int)(((long long)blockIdx.x * U) / G);
      const int u_end = (int)(((long long)(blockIdx.x + 1) * U) / G);
      misc[1] = u_begin;
      end_v[0] = (u_end - u_begin) * 4;
    }
  }
  __syncthreads();
  // the unit of the CTA's tile `it` (dynamic: slot ring; static: contiguous range)
  auto unit_of = [&](int it) -> int {
    if (dyn) return grab * slot_v[(it >> tps_shift) & 3] + ((it >> 2) & (grab - 1));
    return misc[1] + (it >> 2);
  };
  if (end_v[0] <= 0) { sched_exit(p.sched); return; }

  const uint32_t bar0 = smem_u32(smem + L.bars);
  const uint32_t data0 = smem_u32(smem + L.data);
  const uint32_t zdata0 = smem_u32(smem + L.zdata);

  // ---- producer: stage the 32 gathered row segments (and the 32 z row segments) of tile `it` into `stage`
  auto issue = [&](int it, int stage) {
    const Unit un = decode_unit(unit_of(it), KT, nfull);
    const int p0 = un.p0 + (it & 3) * ROWS;
    const int col0 = un.kt * BK;
    const int ncols = min(BK, H - col0);
    const uint32_t row_bytes = (uint32_t)ncols * 2u;
    const uint32_t bar = bar0 + 8 * stage;
    const uint32_t dst = data0 + stage * TILE_BYTES;
    const uint32_t zdst = zdata0 + stage * TILE_BYTES;
    if (LOAD_MODE == 1) {
      if (warp == 0) {
        const int r = lane;
        const int tok = p.r2t[p0 + r];
        const bool valid = (tok >= 0) && (tok < T);
        uint32_t bytes = valid ? row_bytes : 0u;
        if (SCALED) bytes *= 2u;
        // phase transaction count = the bytes the bulk copies will land
        uint32_t total = bytes;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
        if (!valid) {   // padding row: zeros, like TE's quantization of HybridEP's zeroed padding rows
          uint8_t* rowp = smem + L.data + stage * TILE_BYTES + r * ROW_BYTES;
          const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
          for (int k = 0; k < ROW_BYTES / 16; ++k) sts128(rowp, 16 * k, z4);
          if (SCALED) {
            uint8_t* zrowp = smem + L.zdata + stage * TILE_BYTES + r * ROW_BYTES;
#pragma unroll
            for (int k = 0; k < ROW_BYTES / 16; ++k) sts128(zrowp, 16 * k, z4);
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive_expect_tx(bar, total);
        __syncwarp();
        if (valid) {
          const uint16_t* srow = p.src + (size_t)tok * (size_t)H + col0;
          tma_bulk_g2s(dst + r * ROW_BYTES, srow, row_bytes, bar);
          if (SCALED) {
            const uint16_t* zrow = p.z + (size_t)(p0 + r) * (size_t)H + col0;
            tma_bulk_g2s(zdst + r * ROW_BYTES, zrow, row_bytes, bar);
          }
        }
      }
    } else {
      const bool chunk_ok = (c * 8) < ncols;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = 4 * rg + i;
        const int tok = p.r2t[p0 + r];
        const bool valid = (tok >= 0) && (tok < T) && chunk_ok;
        const uint16_t* srow = p.src + (size_t)(valid ? tok : 0) * (size_t)H + col0 + 8 * c;
        cp_async_16(dst + r * ROW_BYTES + 16 * c, srow, valid ? 16u : 0u);
        if (SCALED) {
          const uint16_t* zrow = p.z + (size_t)(p0 + r) * (size_t)H + col0 + 8 * c;
          cp_async_16(zdst + r * ROW_BYTES + 16 * c, zrow, valid ? 16u : 0u);
        }
      }
      cp_async_mbar_arrive_noinc(bar);
    }
  };

  // the prologue stays inside slot 0 (NSTAGE <= 4 <= tiles per slot)
#pragma unroll 1
  for (int s = 0; s < NSTAGE; ++s)
    if (s < end_v[0]) issue(s, s);

  // ---- per-CTA state across tiles
  int g = 0;                      // expert group of the current unit (units are in increasing row order)
  int cur_kt = -1, cur_g = -1;    // (kt, group) the column-sum accumulator belongs to
  float csum[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) csum[k] = 0.f;

  auto flush_colsum = [&](int kt_f, int g_f) {
    // the thread's 8-column partial (its 4 rows per tile, accumulated over the tiles) -> 8 row groups -> one atomic
    // per column into the expert's row of dbias
    *reinterpret_cast<float4*>(csum_x + rg * BK + 8 * c) = make_float4(csum[0], csum[1], csum[2], csum[3]);
    *reinterpret_cast<float4*>(csum_x + rg * BK + 8 * c + 4) = make_float4(csum[4], csum[5], csum[6], csum[7]);
    __syncthreads();
    const int col = tid;
    float s = 0.f;
#pragma unroll
    for (int r = 0; r < 8; ++r) s += csum_x[r * BK + col];
    const int gcol = kt_f * BK + col;
    if (gcol < H && g_f >= 0) atomicAdd(p.dbias + (size_t)g_f * H + gcol, s);
#pragma unroll
    for (int k = 0; k < 8; ++k) csum[k] = 0.f;
  };

#pragma unroll 1
  for (int it = 0; it < end_v[0]; ++it) {
    const int stage = it % NSTAGE;
    const uint32_t parity = (uint32_t)((it / NSTAGE) & 1);
    const Unit un = decode_unit(unit_of(it), KT, nfull);
    const int m = it & 3;
    const int p0 = un.p0 + m * ROWS;
    const int kt = un.kt;
    const int col0 = kt * BK;
    const int ncols = min(BK, H - col0);
    const bool cvalid = (c * 8) < ncols;
    // the next slot is grabbed at the tile whose refill (after B3 below) issues that slot's first tile
    if (dyn && tid == 0 && ((it + NSTAGE) & ((1 << tps_shift) - 1)) == 0 && end_v[0] == END_UNKNOWN)
      grab_slot((it + NSTAGE) >> tps_shift);
    if (m == 0) {
      // units arrive in increasing order, but the two 128-row halves of a 256-row chunk alternate (+0, +128, +0, ...):
      // with 128-row expert padding a chunk can straddle two groups, so the walk steps back as well as forward
      while (g > 0 && p0 < starts[g]) --g;
      while (p0 >= starts[g] + pc[g]) ++g;     // skips empty groups
      if (SCALED && COLSUM) {
        if ((kt != cur_kt || g != cur_g) && cur_kt >= 0) flush_colsum(cur_kt, cur_g);
        cur_kt = kt; cur_g = g;
      }
    }
    float pp[4];
    if (SCALED) {
#pragma unroll
      for (int i = 0; i < 4; ++i) pp[i] = p.pprob[p0 + 4 * rg + i];
    }

    mbar_wait(bar0 + 8 * stage, parity);

    // ---- the thread's 4 x 8 bf16 (and 4 x 8 z values)
    const uint8_t* dbase = smem + L.data + stage * TILE_BYTES + 16 * c;
    uint32_t x[4][4];
    float dot[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint4 d = make_uint4(0u, 0u, 0u, 0u);
      if (cvalid) d = lds128(dbase, (4 * rg + i) * ROW_BYTES);
      if (SCALED) {
        uint4 zt = make_uint4(0u, 0u, 0u, 0u);
        if (cvalid) zt = lds128(smem + L.zdata + stage * TILE_BYTES + 16 * c, (4 * rg + i) * ROW_BYTES);
        const uint32_t dw[4] = {d.x, d.y, d.z, d.w};
        const uint32_t zw[4] = {zt.x, zt.y, zt.z, zt.w};
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float vlo = bf_lo(dw[k]), vhi = bf_hi(dw[k]);
          acc = fmaf(vlo, bf_lo(zw[k]), acc);
          acc = fmaf(vhi, bf_hi(zw[k]), acc);
          x[i][k] = pack_bf16x2_rn(vlo * pp[i], vhi * pp[i]);   // dz = bf16(p * dres), autograd's rounding
        }
        dot[i] = acc;
      } else {
        x[i][0] = d.x; x[i][1] = d.y; x[i][2] = d.z; x[i][3] = d.w;
      }
    }

    // ---- amax: rowwise (32 columns = 4 lanes) and columnwise partial (the thread's 4 rows)
    uint32_t cmax[4];
    uint32_t rpack[2];
    {
      uint32_t rm[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t a0 = x[i][0] & 0x7FFF7FFFu, a1 = x[i][1] & 0x7FFF7FFFu;
        const uint32_t a2 = x[i][2] & 0x7FFF7FFFu, a3 = x[i][3] & 0x7FFF7FFFu;
        if (i == 0) { cmax[0] = a0; cmax[1] = a1; cmax[2] = a2; cmax[3] = a3; }
        else { cmax[0] = bmax2(cmax[0], a0); cmax[1] = bmax2(cmax[1], a1); cmax[2] = bmax2(cmax[2], a2); cmax[3] = bmax2(cmax[3], a3); }
        uint32_t mm = bmax2(bmax2(a0, a1), bmax2(a2, a3));
        mm = bmax2(mm, mm >> 16);
        rm[i] = mm & 0xFFFFu;
      }
      rpack[0] = rm[0] | (rm[1] << 16);
      rpack[1] = rm[2] | (rm[3] << 16);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        rpack[j] = bmax2(rpack[j], __shfl_xor_sync(0xffffffffu, rpack[j], 1));
        rpack[j] = bmax2(rpack[j], __shfl_xor_sync(0xffffffffu, rpack[j], 2));
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) cmax[k] = bmax2(cmax[k], __shfl_xor_sync(0xffffffffu, cmax[k], 16));
    }
    if (lane < 16) sts128(smem + L.xchg, warp * (BK * 2) + 16 * c, make_uint4(cmax[0], cmax[1], cmax[2], cmax[3]));

    // ---- rowwise scales (redundant across the block's 4 lanes)
    float sinv_r[4];
    uint32_t e_r[4];
    {
      const float am0 = __uint_as_float(rpack[0] << 16), am1 = __uint_as_float(rpack[0] & 0xFFFF0000u);
      const float am2 = __uint_as_float(rpack[1] << 16), am3 = __uint_as_float(rpack[1] & 0xFFFF0000u);
      const uint32_t e01 = e8m0_pair(am0 * rcp, am1 * rcp);
      const uint32_t e23 = e8m0_pair(am2 * rcp, am3 * rcp);
      e_r[0] = e01 & 0xFFu; e_r[1] = (e01 >> 8) & 0xFFu; e_r[2] = e23 & 0xFFu; e_r[3] = (e23 >> 8) & 0xFFu;
#pragma unroll
      for (int i = 0; i < 4; ++i) sinv_r[i] = sinv_from_e(e_r[i]);
    }
    __syncthreads();   // B1: column partials visible

    // ---- columnwise scale of column 8c + rg: 4 warp partials -> amax -> e8m0, scale_inv (shared through smem)
    {
      const int col = 8 * c + rg;
      uint32_t a = xchg[0 * BK + col];
      a = bmax2(a, xchg[1 * BK + col]);
      a = bmax2(a, xchg[2 * BK + col]);
      a = bmax2(a, xchg[3 * BK + col]);
      const float am = __uint_as_float((a & 0xFFFFu) << 16);
      const uint32_t e = e8m0_pair(am * rcp, 0.f) & 0xFFu;
      sinv_sm[col] = sinv_from_e(e);
      // TE grouped storage, swizzled tile: byte (col % 32) * 16 + ((col / 32) % 4) * 4 + (mb_local % 4); mb_local % 4 == m
      cws_stage[(col & 31) * 16 + ((col >> 5) & 3) * 4 + m] = (col0 + col < H) ? (uint8_t)e : (uint8_t)0;
    }
    if ((c & 3) == 0) {
      const int j = c >> 2;                       // 32-column block within the tile
      const bool kb_ok = ((kt * 4 + j) * 32) < H;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        rws_stage[(4 * rg + i) * 16 + m * 4 + j] = kb_ok ? (uint8_t)e_r[i] : (uint8_t)0;
    }
    __syncthreads();   // B2: columnwise scale_inv visible

    // ---- quantize + store
    float sc[8];
    {
      const float4 s0 = *reinterpret_cast<const float4*>(sinv_sm + 8 * c);
      const float4 s1 = *reinterpret_cast<const float4*>(sinv_sm + 8 * c + 4);
      sc[0] = s0.x; sc[1] = s0.y; sc[2] = s0.z; sc[3] = s0.w; sc[4] = s1.x; sc[5] = s1.y; sc[6] = s1.z; sc[7] = s1.w;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const size_t row = (size_t)(p0 + 4 * rg + i);
      const size_t off = row * (size_t)H + (size_t)(col0 + 8 * c);
      float f[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) { f[2 * k] = bf_lo(x[i][k]); f[2 * k + 1] = bf_hi(x[i][k]); }
      if (SCALED && COLSUM) {
#pragma unroll
        for (int k = 0; k < 8; ++k) csum[k] += f[k];
      }
      if (cvalid) {
        const float sr = sinv_r[i];
        uint2 qr, qc;
        qr.x = quant4(f[0], f[1], f[2], f[3], sr, sr, sr, sr);
        qr.y = quant4(f[4], f[5], f[6], f[7], sr, sr, sr, sr);
        qc.x = quant4(f[0], f[1], f[2], f[3], sc[0], sc[1], sc[2], sc[3]);
        qc.y = quant4(f[4], f[5], f[6], f[7], sc[4], sc[5], sc[6], sc[7]);
        *reinterpret_cast<uint2*>(p.rw + off) = qr;
        *reinterpret_cast<uint2*>(p.cw + off) = qc;
        if (WRITE_BF16) *reinterpret_cast<uint4*>(p.out + off) = make_uint4(x[i][0], x[i][1], x[i][2], x[i][3]);
      }
    }
    if (SCALED) {
#pragma unroll
      for (int i = 0; i < 4; ++i) dot[i] = warp_sum16(dot[i]);
      if (c == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) p.dprob_part[(size_t)kt * P + p0 + 4 * rg + i] = dot[i];
      }
    }
    __syncthreads();   // B3: stage free for the refill, staging buffers complete for this tile, slot ring / end published

    if (m == 3) {
      // the unit's 128 rows = one swizzled 512-B scale tile in each orientation: 16-B stores
      if (tid < 32) {
        const uint4 v = lds128(rws_stage, 16 * tid);
        const size_t tile = (size_t)(un.p0 / 128) * KT + kt;
        *reinterpret_cast<uint4*>(p.rws + tile * 512 + 16 * tid) = v;
      } else if (tid < 64) {
        const int t = tid - 32;
        const uint4 v = lds128(cws_stage, 16 * t);
        const int start_g = starts[g];
        const int m_g = pc[g];
        const int mb4 = (un.p0 - start_g) / 128;                  // mb_local / 4
        const size_t base_g = (size_t)(start_g / 32) * (size_t)p.K_PAD;
        const size_t tile = (size_t)kt * (size_t)(m_g / 128) + mb4;
        *reinterpret_cast<uint4*>(p.cws + base_g + tile * 512 + 16 * t) = v;
      }
    }
    if (it + NSTAGE < end_v[0]) issue(it + NSTAGE, stage);
  }
  if (SCALED && COLSUM) {
    if (cur_kt >= 0) flush_colsum(cur_kt, cur_g);
  }
  sched_exit(p.sched);
}

// dprob[p] = sum_kt part[kt, p] for the rows in use, 0 beyond them (fixed summation order)
__global__ void dprob_reduce_kernel(const float* __restrict__ part, float* __restrict__ dprob,
                                    const int32_t* __restrict__ tpe, int E, int PAD, int KT, int P) {
  __shared__ int n_used_sm;
  if (threadIdx.x == 0) {
    int s = 0;
    for (int e = 0; e < E; ++e) s += ((tpe[e] + (PAD - 1)) / PAD) * PAD;
    n_used_sm = s;
  }
  __syncthreads();
  const int n_used = n_used_sm;
  const int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pidx >= P) return;
  float s = 0.f;
  if (pidx < n_used) {
#pragma unroll 1
    for (int kt = 0; kt < KT; ++kt) s += part[(size_t)kt * P + pidx];
  }
  dprob[pidx] = s;
}

// ------------------------------------------------------------------------------------------------ host side
// attrs != nullptr: an attribute query instead of a launch -- attrs[0..5] = registers per thread, resident CTAs per SM
// (occupancy at the variant's dynamic shared memory), dynamic smem bytes, static smem bytes, local (stack) bytes per
// thread, max threads per CTA.  Every rank logs these for the variants in use, so the <= 64 regs / 8 CTAs x 4 warps
// resident design point is a measured fact of the target build, not of a local sm_90 compile.
template <bool SCALED, bool WRITE_BF16, bool COLSUM, int LOAD_MODE, int NSTAGE>
static cudaError_t launch_variant(const Params& p, int num_sms, cudaStream_t stream, int* grid_used, int* attrs) {
  auto kern = qperm_gather_quant_kernel<SCALED, WRITE_BF16, COLSUM, LOAD_MODE, NSTAGE>;
  constexpr SmemLayout L = smem_layout(SCALED, NSTAGE, COLSUM);
  static int blocks_per_sm = -1;     // per instantiation
  if (blocks_per_sm < 0) {
    cudaError_t err = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L.total);
    if (err != cudaSuccess) return err;
    int nb = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, NT, L.total);
    if (err != cudaSuccess) return err;
    blocks_per_sm = nb < 1 ? 1 : nb;
  }
  if (attrs != nullptr) {
    cudaFuncAttributes fa{};
    cudaError_t err = cudaFuncGetAttributes(&fa, kern);
    if (err != cudaSuccess) return err;
    attrs[0] = fa.numRegs;
    attrs[1] = blocks_per_sm;
    attrs[2] = L.total;
    attrs[3] = (int)fa.sharedSizeBytes;
    attrs[4] = (int)fa.localSizeBytes;
    attrs[5] = fa.maxThreadsPerBlock;
    return cudaSuccess;
  }
  const int grab = (p.sched != nullptr) ? p.grab : 1;
  long long slots = ((long long)(p.P / 128) * p.KT + grab - 1) / grab;   // an upper bound (P >= the rows in use)
  long long grid = (long long)num_sms * blocks_per_sm;
  if (grid > slots) grid = slots;
  if (grid < 1) grid = 1;
  if (grid_used) *grid_used = (int)grid;
  kern<<<(unsigned)grid, NT, L.total, stream>>>(p);
  return cudaGetLastError();
}

template <bool SCALED, bool WRITE_BF16, bool COLSUM, int LOAD_MODE>
static cudaError_t launch_ns(const Params& p, int nstage, int num_sms, cudaStream_t stream, int* grid_used, int* attrs) {
  switch (nstage) {
    case 2: return launch_variant<SCALED, WRITE_BF16, COLSUM, LOAD_MODE, 2>(p, num_sms, stream, grid_used, attrs);
    case 3: return launch_variant<SCALED, WRITE_BF16, COLSUM, LOAD_MODE, 3>(p, num_sms, stream, grid_used, attrs);
    case 4: return launch_variant<SCALED, WRITE_BF16, COLSUM, LOAD_MODE, 4>(p, num_sms, stream, grid_used, attrs);
    default: return cudaErrorInvalidValue;
  }
}

template <bool SCALED, bool WRITE_BF16, bool COLSUM>
static cudaError_t launch_lm(const Params& p, int load_mode, int nstage, int num_sms, cudaStream_t stream, int* grid_used,
                             int* attrs) {
  if (load_mode == 1) return launch_ns<SCALED, WRITE_BF16, COLSUM, 1>(p, nstage, num_sms, stream, grid_used, attrs);
  if (load_mode == 0) return launch_ns<SCALED, WRITE_BF16, COLSUM, 0>(p, nstage, num_sms, stream, grid_used, attrs);
  return cudaErrorInvalidValue;
}

static cudaError_t dispatch(const Params& p, int scaled, int write_bf16, int colsum, int load_mode, int nstage, int num_sms,
                            cudaStream_t stream, int* grid_used, int* attrs) {
  if (!scaled) {
    if (write_bf16) return launch_lm<false, true, false>(p, load_mode, nstage, num_sms, stream, grid_used, attrs);
    return launch_lm<false, false, false>(p, load_mode, nstage, num_sms, stream, grid_used, attrs);
  }
  if (write_bf16) {
    if (colsum) return launch_lm<true, true, true>(p, load_mode, nstage, num_sms, stream, grid_used, attrs);
    return launch_lm<true, true, false>(p, load_mode, nstage, num_sms, stream, grid_used, attrs);
  }
  if (colsum) return launch_lm<true, false, true>(p, load_mode, nstage, num_sms, stream, grid_used, attrs);
  return launch_lm<true, false, false>(p, load_mode, nstage, num_sms, stream, grid_used, attrs);
}

}  // namespace pbq

extern "C" int pbq_smem_bytes(int scaled, int nstage, int colsum) {
  return pbq::smem_layout(scaled != 0, nstage, colsum != 0).total;   // the kernels' own layout (nstage validated by the launcher)
}

extern "C" int pbq_max_experts() { return pbq::MAX_E; }

extern "C" cudaError_t pbq_gather_quant(const pbq::Params* p, int scaled, int write_bf16, int colsum, int load_mode,
                                        int nstage, int num_sms, cudaStream_t stream, int* grid_used) {
  if (p->E > pbq::MAX_E || p->E < 1) return cudaErrorInvalidValue;
  if (p->sched != nullptr && p->grab != 1 && p->grab != 2) return cudaErrorInvalidValue;
  return pbq::dispatch(*p, scaled, write_bf16, colsum, load_mode, nstage, num_sms, stream, grid_used, nullptr);
}

extern "C" cudaError_t pbq_kernel_attrs(int scaled, int write_bf16, int colsum, int load_mode, int nstage, int attrs[6]) {
  pbq::Params none{};
  return pbq::dispatch(none, scaled, write_bf16, colsum, load_mode, nstage, 1, nullptr, nullptr, attrs);
}

extern "C" cudaError_t pbq_dprob_reduce(const float* part, float* dprob, const int32_t* tpe, int E, int PAD, int KT,
                                        int P, cudaStream_t stream) {
  if (P <= 0) return cudaSuccess;
  const int threads = 256;
  const int blocks = (P + threads - 1) / threads;
  pbq::dprob_reduce_kernel<<<blocks, threads, 0, stream>>>(part, dprob, tpe, E, PAD, KT, P);
  return cudaGetLastError();
}
