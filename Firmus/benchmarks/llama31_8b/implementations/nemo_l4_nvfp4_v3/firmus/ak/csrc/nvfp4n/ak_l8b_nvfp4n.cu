// ak_l8b_nvfp4n.cu (Llama 3.1 8B kernel campaign, 2026-10-02): RMSNorm forward (hidden 4096, bf16) that ALSO emits the
// two amaxes TransformerEngine's NVFP4 quantizer needs, so the quantize that follows skips its amax pass.
//
// Stock TE 2.20 (PROF-L0): rmsnorm_fwd_tuned_kernel writes the BF16 output (and rsigma), then TE's NVFP4 quantize reads
// it twice (HadamardAmaxTmaKernel for the pre-RHT and post-RHT amaxes; row_col_rht_gemm_device for the cast).
//
// The per-row arithmetic is TE's own, reproduced for the one configuration the recipe runs -- the tuned launcher
// registered for (4096, bf16, bf16, bf16, fp32) with CTAS_PER_ROW 1, WARPS_M 1, WARPS_N 4, 16-byte loads
// (rmsnorm_fwd_cuda_kernel.cu:187): 128 threads per row, thread t owning vectors t, t+128, t+256, t+384 of 8 elements;
// Stats<float, 1, 1, 4> (utils.cuh): per-thread sequential sum, xor-butterfly warp all-reduce (1, 2, 4, 8, 16), mean,
// sum of squared differences, all-reduce, then Chan's merge across the 4 warps (warp_chan_upd_dynamic); rs =
// rsqrtf(rn * m2 + mu * mu + eps); out = bf16(gamma * (rs * x)).  Compiled WITHOUT --use_fast_math (TE's normalization
// sources are not in its fast-math list).  Here a CTA of 512 threads runs four such 128-thread row groups over 16
// consecutive rows, keeping the 16 output rows in shared memory for the columnwise RHT amax (16-token blocks).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>

#include "ak_l8b_rht_epi.cuh"

using namespace nvcuda;

namespace {

constexpr int kCols = 4096;
constexpr int kRowsPerCta = 16;
constexpr int kGroupThreads = 128;           // TE: WARPS_N 4 x 32
constexpr int kGroups = 4;                   // row groups per CTA
constexpr int kThreads = kGroups * kGroupThreads;
constexpr int kLdgs = 4;                     // TE: LDGS = VEC_COLS / VEC_COLS_PER_LDG = 512 / 128
constexpr int kElts = 8;                     // TE: NUM_ELTS = 16 B / 2 B
constexpr int kLd = kCols + 8;               // smem row stride (elements): 16 B of padding against bank conflicts

__device__ __forceinline__ void atomic_max_nonneg(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

union V8 {
  uint4 u;
  __nv_bfloat16 h[8];
};

// TE utils.cuh Reducer<T, 1, WARPS_M, 1>::allreduce_ with Sum<float>
__device__ __forceinline__ float warp_allreduce_sum(float data) {
#pragma unroll
  for (int it = 1; it < 32; it *= 2) {
    data = data + __shfl_xor_sync(static_cast<uint32_t>(-1), data, it);
  }
  return data;
}

// TE utils.cuh warp_chan_upd_dynamic<float>(m, m2, n, num_active = 4)
__device__ __forceinline__ void warp_chan_upd_4(float& m_a, float& m2_a, float& n_a) {
  const int num_active = 4;
  int highest_bit_set = (8 * sizeof(num_active)) - __clz(num_active - 1);
#pragma unroll
  for (int step = (1 << (highest_bit_set - 1)); step > 0; step /= 2) {
    float n_b = __shfl_down_sync(static_cast<uint32_t>(-1), n_a, step);
    float m_b = __shfl_down_sync(static_cast<uint32_t>(-1), m_a, step);
    float m2_b = __shfl_down_sync(static_cast<uint32_t>(-1), m2_a, step);
    const float n_ab = n_a + n_b;
    const float rn_ab = 1.f / n_ab;
    const float delta = m_a - m_b;
    const float m2_ab = m2_a + m2_b + delta * delta * n_a * n_b * rn_ab;
    const float m_ab = (n_a * m_a + n_b * m_b) * rn_ab;
    n_a = n_ab;
    m_a = m_ab;
    m2_a = m2_ab;
  }
  m_a = __shfl_sync(static_cast<uint32_t>(-1), m_a, 0);
  m2_a = __shfl_sync(static_cast<uint32_t>(-1), m2_a, 0);
}

__global__ void __launch_bounds__(kThreads)
    rmsnorm_fwd_amax_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma,
                            __nv_bfloat16* __restrict__ z, float* __restrict__ rs_out, float* __restrict__ amax_pre,
                            float* __restrict__ amax_post, int64_t rows, float epsilon, uint32_t sign_mask) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* sh_z = reinterpret_cast<__nv_bfloat16*>(smem_raw);                 // [16][kLd]
  __nv_bfloat16* sh_h = sh_z + kRowsPerCta * kLd;                                   // [16][16]
  float2* sh_stats = reinterpret_cast<float2*>(sh_h + 256);                          // [kGroups][4]
  float* sh_red = reinterpret_cast<float*>(sh_stats + kGroups * 4);                  // [2][16 warps]

  const int tid = threadIdx.x;
  const int group = tid / kGroupThreads;         // row group 0..3
  const int c = tid % kGroupThreads;             // TE: c (vector column index of this thread)
  const int warp_n = c / 32;                     // TE: warp_n within the row group
  const int lane = tid % 32;
  const int64_t row0 = (int64_t)blockIdx.x * kRowsPerCta;

  for (int t = tid; t < 256; t += kThreads) {
    const int k = t / 16, n = t % 16;
    float v = (__popc(k & n) & 1) ? -0.25f : 0.25f;
    if ((sign_mask >> k) & 1u) v = -v;
    sh_h[t] = __float2bfloat16_rn(v);
  }

  V8 g[kLdgs];
#pragma unroll
  for (int it = 0; it < kLdgs; it++) {
    g[it].u = *reinterpret_cast<const uint4*>(gamma + (c + it * kGroupThreads) * kElts);
  }
  constexpr float rn = 1.f / float(kCols);
  float a_pre = 0.0f;

  for (int rr = group; rr < kRowsPerCta; rr += kGroups) {
    const int64_t row = row0 + rr;
    float xf[kLdgs * kElts];
#pragma unroll
    for (int it = 0; it < kLdgs; it++) {
      V8 xv;
      xv.u = *reinterpret_cast<const uint4*>(x + row * kCols + (c + it * kGroupThreads) * kElts);
#pragma unroll
      for (int jt = 0; jt < kElts; jt++) xf[it * kElts + jt] = __bfloat162float(xv.h[jt]);
    }
    // ---- Stats<float, 1, 1, 4>::compute (warp part: Stats<float, 1, 1, 1>::compute with warp_rn) ----
    constexpr float warp_rn = 1.f / float(kLdgs * kElts * 32);
    float m = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs * kElts; it++) m += xf[it];
    m = warp_allreduce_sum(m) * warp_rn;
    float m2 = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs * kElts; it++) {
      float diff = (xf[it] - m);
      m2 += diff * diff;
    }
    m2 = warp_allreduce_sum(m2);
    float2* smem = sh_stats + group * 4;
    if (lane == 0) smem[warp_n] = make_float2(m, m2);
    __syncthreads();
    float n = 0.f, mm = 0.f, mm2 = 0.f;
    if (lane < 4) {
      float2 result = smem[lane];
      n = float(kLdgs * kElts * 32);
      mm = result.x;
      mm2 = result.y;
    }
    warp_chan_upd_4(mm, mm2, n);
    __syncthreads();
    // ---- rmsnorm_fwd_tuned_kernel ----
    const float mu = mm;
    const float rs = rsqrtf(rn * mm2 + mu * mu + epsilon);
    if (warp_n == 0 && lane == 0) rs_out[row] = rs;
#pragma unroll
    for (int it = 0; it < kLdgs; it++) {
      V8 o;
#pragma unroll
      for (int jt = 0; jt < kElts; jt++) {
        float y_ij = rs * (xf[it * kElts + jt]);
        float g_ij = __bfloat162float(g[it].h[jt]);
        float temp_output = g_ij * y_ij;
        o.h[jt] = __float2bfloat16(temp_output);
        a_pre = fmaxf(a_pre, fabsf(__bfloat162float(o.h[jt])));
      }
      const int col = (c + it * kGroupThreads) * kElts;
      *reinterpret_cast<uint4*>(z + row * kCols + col) = o.u;
      *reinterpret_cast<uint4*>(sh_z + rr * kLd + col) = o.u;
    }
  }
  __syncthreads();
  // ---- columnwise RHT amax over the 16 x 4096 tile: 256 sub-tiles of 16 tokens x 16 features ----
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a_post = 0.0f;
  const int warp = tid / 32;
  for (int fb = warp; fb < kCols / 16; fb += kThreads / 32) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> afrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
    wmma::load_matrix_sync(afrag, sh_z + fb * 16, kLd);
    wmma::fill_fragment(cfrag, 0.0f);
    wmma::mma_sync(cfrag, afrag, bfrag, cfrag);
#pragma unroll
    for (int i = 0; i < cfrag.num_elements; ++i) {
      a_post = fmaxf(a_post, fabsf(__bfloat162float(__float2bfloat16_rn(cfrag.x[i]))));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    a_pre = fmaxf(a_pre, __shfl_xor_sync(0xffffffffu, a_pre, o));
    a_post = fmaxf(a_post, __shfl_xor_sync(0xffffffffu, a_post, o));
  }
  if (lane == 0) {
    sh_red[warp] = a_pre;
    sh_red[16 + warp] = a_post;
  }
  __syncthreads();
  if (tid == 0) {
    float p = 0.f, q = 0.f;
    for (int w = 0; w < kThreads / 32; ++w) {
      p = fmaxf(p, sh_red[w]);
      q = fmaxf(q, sh_red[16 + w]);
    }
    atomic_max_nonneg(amax_pre, p);
    atomic_max_nonneg(amax_post, q);
  }
}


// ---- v2 (2026-10-02 22:20): one WARP per row.  Lane l plays TE's four threads t = l + 32 w (w = 0..3, TE's four warps of
// the row group), so each of TE's warp-level butterflies is this warp's butterfly over lanes, and TE's final Chan merge
// over the four warps' stats (lanes 0..3, shuffle-down steps 2 then 1: (w0,w2), (w1,w3), then the two) is done in
// registers in the same order.  No block-wide barrier per row; all 16 loads of a lane are issued up front. ----------
struct ChanStat {
  float n, m, m2;
};
__device__ __forceinline__ ChanStat chan_merge(ChanStat a, ChanStat b) {
  // TE utils.cuh warp_chan_upd_dynamic, one step, a = this lane, b = the lane `step` above
  const float n_ab = a.n + b.n;
  const float rn_ab = 1.f / n_ab;
  const float delta = a.m - b.m;
  const float m2_ab = a.m2 + b.m2 + delta * delta * a.n * b.n * rn_ab;
  const float m_ab = (a.n * a.m + b.n * b.m) * rn_ab;
  return {n_ab, m_ab, m2_ab};
}

__global__ void __launch_bounds__(kThreads)
    rmsnorm_fwd_amax_kernel_v2(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma,
                               __nv_bfloat16* __restrict__ z, float* __restrict__ rs_out, float* __restrict__ amax_pre,
                               float* __restrict__ amax_post, int64_t rows, float epsilon, uint32_t sign_mask) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* sh_z = reinterpret_cast<__nv_bfloat16*>(smem_raw);                 // [16][kLd]
  __nv_bfloat16* sh_h = sh_z + kRowsPerCta * kLd;                                   // [16][16]
  __nv_bfloat16* sh_g = sh_h + 256;                                                 // [4096]
  float* sh_red = reinterpret_cast<float*>(sh_g + kCols);                            // [2][16 warps]
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int64_t row = (int64_t)blockIdx.x * kRowsPerCta + warp;
  // all of this lane's loads first: TE thread t = lane + 32 w owns vectors t + 128 it (it = 0..3)
  V8 xv[4][kLdgs];
#pragma unroll
  for (int w = 0; w < 4; ++w)
#pragma unroll
    for (int it = 0; it < kLdgs; ++it)
      xv[w][it].u = __ldcs(reinterpret_cast<const uint4*>(x + row * kCols + (lane + 32 * w + it * kGroupThreads) * kElts));
  for (int t = tid; t < 256; t += kThreads) {
    const int k = t / 16, n = t % 16;
    float v = (__popc(k & n) & 1) ? -0.25f : 0.25f;
    if ((sign_mask >> k) & 1u) v = -v;
    sh_h[t] = __float2bfloat16_rn(v);
  }
  for (int v = tid; v < kCols / 8; v += kThreads) {
    reinterpret_cast<uint4*>(sh_g)[v] = reinterpret_cast<const uint4*>(gamma)[v];
  }
  // TE Stats<float, 1, 1, 1>::compute for each of the four emulated warps
  constexpr float warp_rn = 1.f / float(kLdgs * kElts * 32);
  ChanStat st[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    float m = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs; ++it)
#pragma unroll
      for (int jt = 0; jt < kElts; ++jt) m += __bfloat162float(xv[w][it].h[jt]);
    m = warp_allreduce_sum(m) * warp_rn;
    float m2 = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs; ++it)
#pragma unroll
      for (int jt = 0; jt < kElts; ++jt) {
        float diff = (__bfloat162float(xv[w][it].h[jt]) - m);
        m2 += diff * diff;
      }
    m2 = warp_allreduce_sum(m2);
    st[w] = {float(kLdgs * kElts * 32), m, m2};
  }
  // TE's Chan merge over the four warps (warp_chan_upd_dynamic with num_active 4): step 2 then step 1, lane 0's result
  const ChanStat s02 = chan_merge(st[0], st[2]);
  const ChanStat s13 = chan_merge(st[1], st[3]);
  const ChanStat sall = chan_merge(s02, s13);
  constexpr float rn = 1.f / float(kCols);
  const float mu = sall.m;
  const float rs = rsqrtf(rn * sall.m2 + mu * mu + epsilon);
  if (lane == 0) rs_out[row] = rs;
  __syncthreads();   // gamma in shared memory
  float a_pre = 0.0f;
#pragma unroll
  for (int w = 0; w < 4; ++w)
#pragma unroll
    for (int it = 0; it < kLdgs; ++it) {
      const int col = (lane + 32 * w + it * kGroupThreads) * kElts;
      V8 gv, o;
      gv.u = *reinterpret_cast<const uint4*>(sh_g + col);
#pragma unroll
      for (int jt = 0; jt < kElts; ++jt) {
        float y_ij = rs * (__bfloat162float(xv[w][it].h[jt]));
        float g_ij = __bfloat162float(gv.h[jt]);
        float temp_output = g_ij * y_ij;
        o.h[jt] = __float2bfloat16(temp_output);
        a_pre = fmaxf(a_pre, fabsf(__bfloat162float(o.h[jt])));
      }
      *reinterpret_cast<uint4*>(z + row * kCols + col) = o.u;
      *reinterpret_cast<uint4*>(sh_z + warp * kLd + col) = o.u;
    }
  __syncthreads();
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a_post = 0.0f;
  for (int fb = warp; fb < kCols / 16; fb += kThreads / 32) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> afrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
    wmma::load_matrix_sync(afrag, sh_z + fb * 16, kLd);
    wmma::fill_fragment(cfrag, 0.0f);
    wmma::mma_sync(cfrag, afrag, bfrag, cfrag);
#pragma unroll
    for (int i = 0; i < cfrag.num_elements; ++i) {
      a_post = fmaxf(a_post, fabsf(__bfloat162float(__float2bfloat16_rn(cfrag.x[i]))));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    a_pre = fmaxf(a_pre, __shfl_xor_sync(0xffffffffu, a_pre, o));
    a_post = fmaxf(a_post, __shfl_xor_sync(0xffffffffu, a_post, o));
  }
  if (lane == 0) {
    sh_red[warp] = a_pre;
    sh_red[16 + warp] = a_post;
  }
  __syncthreads();
  if (tid == 0) {
    float p = 0.f, q = 0.f;
    for (int w = 0; w < kThreads / 32; ++w) {
      p = fmaxf(p, sh_red[w]);
      q = fmaxf(q, sh_red[16 + w]);
    }
    atomic_max_nonneg(amax_pre, p);
    atomic_max_nonneg(amax_post, q);
  }
}


// ---- v3 (2026-10-02 23:00): two kernels, so neither needs a 16 x 4096 tile.  (1) stats: one warp per row, TE's Welford
// order exactly as v2, writing only rsigma (read-only otherwise); (2) elementwise: out = bf16(gamma * (rs * x)) on
// 32-row x 256-column tiles with the 16-token RHT amax epilogue (the SwiGLU v1 structure). ----------------------------
constexpr int kStatsWarps = 8;
__global__ void __launch_bounds__(kStatsWarps * 32)
    rmsnorm_stats_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ rs_out, int64_t rows, float epsilon) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t row = (int64_t)blockIdx.x * kStatsWarps + warp;
  if (row >= rows) return;
  V8 xv[4][kLdgs];
#pragma unroll
  for (int w = 0; w < 4; ++w)
#pragma unroll
    for (int it = 0; it < kLdgs; ++it)
      xv[w][it].u = __ldcs(reinterpret_cast<const uint4*>(x + row * kCols + (lane + 32 * w + it * kGroupThreads) * kElts));
  constexpr float warp_rn = 1.f / float(kLdgs * kElts * 32);
  ChanStat st[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    float m = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs; ++it)
#pragma unroll
      for (int jt = 0; jt < kElts; ++jt) m += __bfloat162float(xv[w][it].h[jt]);
    m = warp_allreduce_sum(m) * warp_rn;
    float m2 = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs; ++it)
#pragma unroll
      for (int jt = 0; jt < kElts; ++jt) {
        float diff = (__bfloat162float(xv[w][it].h[jt]) - m);
        m2 += diff * diff;
      }
    m2 = warp_allreduce_sum(m2);
    st[w] = {float(kLdgs * kElts * 32), m, m2};
  }
  const ChanStat s02 = chan_merge(st[0], st[2]);
  const ChanStat s13 = chan_merge(st[1], st[3]);
  const ChanStat sall = chan_merge(s02, s13);
  constexpr float rn = 1.f / float(kCols);
  const float mu = sall.m;
  const float rs = rsqrtf(rn * sall.m2 + mu * mu + epsilon);
  if (lane == 0) rs_out[row] = rs;
}

// ---- add + stats (2026-10-03, for the residual add that precedes a norm): h = bf16(float(r) + float(o)) -- the bf16 add
// of torch / the compiled bias-dropout-add -- written out, and TE's rsigma of h exactly as rmsnorm_stats_kernel computes
// it from the same bf16 values.  The norm that consumes h then runs only the apply pass.
__global__ void __launch_bounds__(kStatsWarps * 32)
    rmsnorm_add_stats_kernel(const __nv_bfloat16* __restrict__ r, const __nv_bfloat16* __restrict__ o,
                             __nv_bfloat16* __restrict__ h, float* __restrict__ rs_out, int64_t rows, float epsilon) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t row = (int64_t)blockIdx.x * kStatsWarps + warp;
  if (row >= rows) return;
  V8 xv[4][kLdgs];
#pragma unroll
  for (int w = 0; w < 4; ++w)
#pragma unroll
    for (int it = 0; it < kLdgs; ++it) {
      const int64_t off = row * kCols + (lane + 32 * w + it * kGroupThreads) * kElts;
      V8 a, b;
      a.u = __ldcs(reinterpret_cast<const uint4*>(r + off));
      b.u = __ldcs(reinterpret_cast<const uint4*>(o + off));
#pragma unroll
      for (int jt = 0; jt < kElts; ++jt)
        xv[w][it].h[jt] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(a.h[jt]), __bfloat162float(b.h[jt])));
      *reinterpret_cast<uint4*>(h + off) = xv[w][it].u;
    }
  constexpr float warp_rn = 1.f / float(kLdgs * kElts * 32);
  ChanStat st[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    float m = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs; ++it)
#pragma unroll
      for (int jt = 0; jt < kElts; ++jt) m += __bfloat162float(xv[w][it].h[jt]);
    m = warp_allreduce_sum(m) * warp_rn;
    float m2 = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs; ++it)
#pragma unroll
      for (int jt = 0; jt < kElts; ++jt) {
        float diff = (__bfloat162float(xv[w][it].h[jt]) - m);
        m2 += diff * diff;
      }
    m2 = warp_allreduce_sum(m2);
    st[w] = {float(kLdgs * kElts * 32), m, m2};
  }
  const ChanStat s02 = chan_merge(st[0], st[2]);
  const ChanStat s13 = chan_merge(st[1], st[3]);
  const ChanStat sall = chan_merge(s02, s13);
  constexpr float rn = 1.f / float(kCols);
  const float mu = sall.m;
  const float rs = rsqrtf(rn * sall.m2 + mu * mu + epsilon);
  if (lane == 0) rs_out[row] = rs;
}

// ---- add + stats v2 (2026-10-06): rmsnorm_add_stats_kernel's arithmetic in the same order, software-pipelined.  v1
// holds a whole row per warp (all 32 loads in flight, then the adds, stores and sums), so few warps fit on an SM and
// their compute phases leave the memory pipe idle: 72.3 us against torch's plain add of the same bytes, 56.9 us.  Here a
// warp walks R consecutive rows in units of one TE warp group (w = 0..3: vectors lane + 32 w + 128 it, it = 0..3, as
// v1's xv[w]): it waits for the current unit's 8 loads, forms h = bf16(r + o), issues the NEXT unit's loads (the next
// group, or the next row's group 0), then stores h and runs the group's sums while those loads are in flight.  Per group
// the same sequential sums, xor all-reduces and, per row, the same Chan merges as v1, so h and rsigma are v1's bit for
// bit.  128 threads per CTA; MINB CTAs per SM requested from the compiler (registers). ---------------------------------
constexpr int kAS2Warps = 4;
template <int R, int MINB>
__global__ void __launch_bounds__(kAS2Warps * 32, MINB)
    rmsnorm_add_stats_kernel_v2(const __nv_bfloat16* __restrict__ r, const __nv_bfloat16* __restrict__ o,
                                __nv_bfloat16* __restrict__ h, float* __restrict__ rs_out, int64_t rows, float epsilon) {
  const int lane = threadIdx.x % 32;
  const int64_t row0 = ((int64_t)blockIdx.x * kAS2Warps + threadIdx.x / 32) * R;
  if (row0 >= rows) return;
  const int nrows = (int)(rows - row0 < R ? rows - row0 : R);
  uint4 na[kLdgs], nb[kLdgs];
#pragma unroll
  for (int it = 0; it < kLdgs; ++it) {
    const int64_t off = row0 * kCols + (lane + it * kGroupThreads) * kElts;
    na[it] = __ldcs(reinterpret_cast<const uint4*>(r + off));
    nb[it] = __ldcs(reinterpret_cast<const uint4*>(o + off));
  }
  constexpr float warp_rn = 1.f / float(kLdgs * kElts * 32);
#pragma unroll 1
  for (int i = 0; i < nrows; ++i) {
    const int64_t row = row0 + i;
    ChanStat st[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      V8 xv[kLdgs];
#pragma unroll
      for (int it = 0; it < kLdgs; ++it) {
        V8 a, b;
        a.u = na[it];
        b.u = nb[it];
#pragma unroll
        for (int jt = 0; jt < kElts; ++jt)
          xv[it].h[jt] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(a.h[jt]), __bfloat162float(b.h[jt])));
      }
      // the next unit's loads: group w + 1 of this row, or group 0 of the next row
      const bool more = w < 3 || i + 1 < nrows;
      if (more) {
        const int64_t nrow = w < 3 ? row : row + 1;
        const int nw = w < 3 ? w + 1 : 0;
#pragma unroll
        for (int it = 0; it < kLdgs; ++it) {
          const int64_t off = nrow * kCols + (lane + 32 * nw + it * kGroupThreads) * kElts;
          na[it] = __ldcs(reinterpret_cast<const uint4*>(r + off));
          nb[it] = __ldcs(reinterpret_cast<const uint4*>(o + off));
        }
      }
#pragma unroll
      for (int it = 0; it < kLdgs; ++it)
        *reinterpret_cast<uint4*>(h + row * kCols + (lane + 32 * w + it * kGroupThreads) * kElts) = xv[it].u;
      float m = 0.f;
#pragma unroll
      for (int it = 0; it < kLdgs; ++it)
#pragma unroll
        for (int jt = 0; jt < kElts; ++jt) m += __bfloat162float(xv[it].h[jt]);
      m = warp_allreduce_sum(m) * warp_rn;
      float m2 = 0.f;
#pragma unroll
      for (int it = 0; it < kLdgs; ++it)
#pragma unroll
        for (int jt = 0; jt < kElts; ++jt) {
          float diff = (__bfloat162float(xv[it].h[jt]) - m);
          m2 += diff * diff;
        }
      m2 = warp_allreduce_sum(m2);
      st[w] = {float(kLdgs * kElts * 32), m, m2};
    }
    const ChanStat s02 = chan_merge(st[0], st[2]);
    const ChanStat s13 = chan_merge(st[1], st[3]);
    const ChanStat sall = chan_merge(s02, s13);
    constexpr float rn = 1.f / float(kCols);
    const float mu = sall.m;
    const float rs = rsqrtf(rn * sall.m2 + mu * mu + epsilon);
    if (lane == 0) rs_out[row] = rs;
  }
}

constexpr int kTR3 = 32, kTC3 = 256, kThreads3 = 256;
// LD: the staging tile's row stride, kTC3 (v3) or kTC3 + 8 (v4: rows 528 B apart, no bank conflict in the wmma loads)
template <int LD>
__global__ void __launch_bounds__(kThreads3)
    rmsnorm_apply_amax_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma,
                              const float* __restrict__ rs_in, __nv_bfloat16* __restrict__ z,
                              float* __restrict__ amax_pre, float* __restrict__ amax_post, uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh_o[kTR3 * LD];
  __shared__ __align__(128) __nv_bfloat16 sh_h[256];
  __shared__ float red[2][kThreads3 / 32];
  const int64_t row0 = (int64_t)blockIdx.y * kTR3, col0 = (int64_t)blockIdx.x * kTC3;
  for (int t = threadIdx.x; t < 256; t += kThreads3) {
    const int k = t / 16, n = t % 16;
    float v = (__popc(k & n) & 1) ? -0.25f : 0.25f;
    if ((sign_mask >> k) & 1u) v = -v;
    sh_h[t] = __float2bfloat16_rn(v);
  }
  float a_pre = 0.0f;
  constexpr int kVecPerRow = kTC3 / 8;
#pragma unroll
  for (int i = 0; i < kTR3 * kVecPerRow / kThreads3; ++i) {
    const int v = threadIdx.x + i * kThreads3;
    const int r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    V8 xv, gv, o;
    xv.u = __ldcs(reinterpret_cast<const uint4*>(x + gi * kCols + gj));
    gv.u = *reinterpret_cast<const uint4*>(gamma + gj);
    const float rs = rs_in[gi];
#pragma unroll
    for (int jt = 0; jt < 8; ++jt) {
      float y_ij = rs * (__bfloat162float(xv.h[jt]));
      float g_ij = __bfloat162float(gv.h[jt]);
      float temp_output = g_ij * y_ij;
      o.h[jt] = __float2bfloat16(temp_output);
      a_pre = fmaxf(a_pre, fabsf(__bfloat162float(o.h[jt])));
    }
    *reinterpret_cast<uint4*>(z + gi * kCols + gj) = o.u;
    *reinterpret_cast<uint4*>(sh_o + r * LD + c8) = o.u;
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a_post = 0.0f;
  constexpr int kSub = (kTR3 / 16) * (kTC3 / 16);
  for (int t = threadIdx.x / 32; t < kSub; t += kThreads3 / 32) {
    const int tb = t / (kTC3 / 16), fb = t % (kTC3 / 16);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> afrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
    wmma::load_matrix_sync(afrag, sh_o + tb * 16 * LD + fb * 16, LD);
    wmma::fill_fragment(cfrag, 0.0f);
    wmma::mma_sync(cfrag, afrag, bfrag, cfrag);
#pragma unroll
    for (int i = 0; i < cfrag.num_elements; ++i) {
      a_post = fmaxf(a_post, fabsf(__bfloat162float(__float2bfloat16_rn(cfrag.x[i]))));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    a_pre = fmaxf(a_pre, __shfl_xor_sync(0xffffffffu, a_pre, o));
    a_post = fmaxf(a_post, __shfl_xor_sync(0xffffffffu, a_post, o));
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) {
    red[0][warp] = a_pre;
    red[1][warp] = a_post;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float p = 0.f, q = 0.f;
    for (int w = 0; w < kThreads3 / 32; ++w) {
      p = fmaxf(p, red[0][w]);
      q = fmaxf(q, red[1][w]);
    }
    atomic_max_nonneg(amax_pre, p);
    atomic_max_nonneg(amax_post, q);
  }
}

// v5 (2026-10-05): v4's apply pass (the same tiles, staging and per-element arithmetic; the two outputs of adjacent
// features rounded together by cvt.rn.bf16x2.f32, the same round-to-nearest-even) with the epilogue of
// ak_l8b_rht_epi.cuh: the pre-RHT amax by TE's max.xorsign.abs.bf16x2 on the packed output, the post-RHT amax by TE's
// instruction sequence (ldmatrix.trans + mma.sync with the RHT in registers, cvt.rn.bf16x2.f32, max.xorsign.abs.bf16x2).
__global__ void __launch_bounds__(kThreads3)
    rmsnorm_apply_amax_kernel_v5(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma,
                                 const float* __restrict__ rs_in, __nv_bfloat16* __restrict__ z,
                                 float* __restrict__ amax_pre, float* __restrict__ amax_post, uint32_t sign_mask) {
  constexpr int LD = kTC3 + 8;
  __shared__ __align__(128) __nv_bfloat16 sh_o[kTR3 * LD];
  const int64_t row0 = (int64_t)blockIdx.y * kTR3, col0 = (int64_t)blockIdx.x * kTC3;
  uint32_t pre0 = 0u, pre1 = 0u;                       // packed bf16 abs-max accumulators (+0, as TE starts)
  constexpr int kVecPerRow = kTC3 / 8;
#pragma unroll
  for (int i = 0; i < kTR3 * kVecPerRow / kThreads3; ++i) {
    const int v = threadIdx.x + i * kThreads3;
    const int r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    V8 xv, gv;
    xv.u = __ldcs(reinterpret_cast<const uint4*>(x + gi * kCols + gj));
    gv.u = *reinterpret_cast<const uint4*>(gamma + gj);
    const float rs = rs_in[gi];
    uint32_t o[4];
#pragma unroll
    for (int j2 = 0; j2 < 4; ++j2) {
      float f[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int jt = 2 * j2 + h;
        float y_ij = rs * (__bfloat162float(xv.h[jt]));
        float g_ij = __bfloat162float(gv.h[jt]);
        float temp_output = g_ij * y_ij;
        f[h] = temp_output;
      }
      o[j2] = ak_rht::cvt_bf16x2(f[0], f[1]);
      if (j2 & 1) pre1 = ak_rht::bmax_xa(pre1, o[j2]); else pre0 = ak_rht::bmax_xa(pre0, o[j2]);
    }
    const uint4 uo = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(z + gi * kCols + gj) = uo;
    *reinterpret_cast<uint4*>(sh_o + r * LD + c8) = uo;
  }
  const ak_rht::RhtB B(sign_mask);
  __syncthreads();
  uint32_t post = 0u;
  constexpr int kSub = (kTR3 / 16) * (kTC3 / 16);
#pragma unroll
  for (int t = threadIdx.x / 32; t < kSub; t += kThreads3 / 32) {
    const int tb = t / (kTC3 / 16), fb = t % (kTC3 / 16);
    post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<LD>(sh_o + tb * 16 * LD + fb * 16, B));
  }
  ak_rht::block_publish2<kThreads3 / 32>(ak_rht::fmax_noftz(ak_rht::unpack_bmax(pre0), ak_rht::unpack_bmax(pre1)),
                                         ak_rht::unpack_bmax(post), amax_pre, amax_post);
}

// v6 (2026-10-05): the apply pass in warp-owned 16-row x 64-feature blocks (the layout of ak_l8b_nvfp4p's dSwiGLU v7):
// lane l of a warp takes row 4 it + l / 8 of iteration it, features 8 (l % 8) .. + 7 (each row segment one 128-byte
// line), the next vector's loads issued before the current one is computed, the block staged in the warp's own
// shared-memory region and its post-RHT amax taken right away (no CTA barrier); the loop unrolled, 5 CTAs per SM.
// Per element v4's arithmetic; the epilogue of ak_l8b_rht_epi.cuh.
constexpr int kAB_R = 16, kAB_C = 64, kAB_LD = kAB_C + 8;
__global__ void __launch_bounds__(256, 5)
    rmsnorm_apply_amax_kernel_v6(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma,
                                 const float* __restrict__ rs_in, __nv_bfloat16* __restrict__ z,
                                 float* __restrict__ amax_pre, float* __restrict__ amax_post, int64_t rows,
                                 uint32_t sign_mask) {
  constexpr int ITERS = kAB_R * kAB_C / 8 / 32;      // 4
  __shared__ __align__(128) __nv_bfloat16 sh[8][kAB_R * kAB_LD];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __nv_bfloat16* s0 = sh[warp];
  constexpr int64_t ncb = kCols / kAB_C;
  const int64_t nblocks = (rows / kAB_R) * ncb;
  const int64_t b = (int64_t)blockIdx.x * 8 + warp;
  const int rr = lane >> 3, c8 = (lane & 7) * 8;
  uint32_t pre0 = 0u, pre1 = 0u, post = 0u;
  const ak_rht::RhtB B(sign_mask);
  if (b < nblocks) {
    const int64_t rowb = (b / ncb) * kAB_R, colb = (b % ncb) * kAB_C + c8;
    V8 gv;
    gv.u = *reinterpret_cast<const uint4*>(gamma + colb);
    V8 nx;
    float nrs;
    nx.u = __ldcs(reinterpret_cast<const uint4*>(x + (rowb + rr) * kCols + colb));
    nrs = rs_in[rowb + rr];
#pragma unroll 4
    for (int it = 0; it < ITERS; ++it) {
      const int r = it * 4 + rr;
      const int64_t gi = rowb + r;
      const V8 xv = nx;
      const float rs = nrs;
      if (it + 1 < ITERS) {
        nx.u = __ldcs(reinterpret_cast<const uint4*>(x + (gi + 4) * kCols + colb));
        nrs = rs_in[gi + 4];
      }
      uint32_t o[4];
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        float f[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int jt = 2 * j2 + h;
          float y_ij = rs * (__bfloat162float(xv.h[jt]));
          float g_ij = __bfloat162float(gv.h[jt]);
          float temp_output = g_ij * y_ij;
          f[h] = temp_output;
        }
        o[j2] = ak_rht::cvt_bf16x2(f[0], f[1]);
        if (j2 & 1) pre1 = ak_rht::bmax_xa(pre1, o[j2]); else pre0 = ak_rht::bmax_xa(pre0, o[j2]);
      }
      const uint4 uo = make_uint4(o[0], o[1], o[2], o[3]);
      *reinterpret_cast<uint4*>(z + gi * kCols + colb) = uo;
      *reinterpret_cast<uint4*>(s0 + r * kAB_LD + c8) = uo;
    }
    __syncwarp();
#pragma unroll
    for (int fb = 0; fb < kAB_C / 16; ++fb) post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kAB_LD>(s0 + fb * 16, B));
  }
  ak_rht::block_publish2<8>(ak_rht::fmax_noftz(ak_rht::unpack_bmax(pre0), ak_rht::unpack_bmax(pre1)),
                            ak_rht::unpack_bmax(post), amax_pre, amax_post);
}

}  // namespace

// rmsnorm_fwd_amax(x [.., 4096] bf16, gamma [4096] bf16, eps, sign_mask) -> (z bf16, rsigma f32 [rows], amax_pre, amax_post)
std::vector<at::Tensor> rmsnorm_fwd_amax(const at::Tensor& x, const at::Tensor& gamma, double eps, int64_t sign_mask,
                                         int64_t version) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "x: contiguous CUDA bf16");
  TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous() &&
                  gamma.numel() == kCols, "gamma: contiguous CUDA bf16 [4096]");
  TORCH_CHECK(x.size(-1) == kCols, "hidden must be 4096");
  const int64_t rows = x.numel() / kCols;
  TORCH_CHECK(rows % 32 == 0, "rows % 32 required");
  const at::cuda::CUDAGuard guard(x.device());
  auto z = at::empty_like(x);
  auto rs = at::empty({rows}, x.options().dtype(at::kFloat));
  auto amax = at::zeros({2}, x.options().dtype(at::kFloat));
  const size_t smem1 = sizeof(__nv_bfloat16) * (kRowsPerCta * kLd + 256) + sizeof(float2) * kGroups * 4 +
                       sizeof(float) * 32;
  const size_t smem2 = sizeof(__nv_bfloat16) * (kRowsPerCta * kLd + 256 + kCols) + sizeof(float) * 32;
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(rmsnorm_fwd_amax_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem1));
    C10_CUDA_CHECK(cudaFuncSetAttribute(rmsnorm_fwd_amax_kernel_v2, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)smem2));
    attr_set = true;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (version == 3 || version == 4 || version == 5 || version == 6) {
    rmsnorm_stats_kernel<<<(unsigned)((rows + kStatsWarps - 1) / kStatsWarps), kStatsWarps * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), rs.data_ptr<float>(), rows, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (version == 6) {
      const int64_t nblocks = (rows / kAB_R) * (kCols / kAB_C);
      rmsnorm_apply_amax_kernel_v6<<<(unsigned)((nblocks + 7) / 8), 256, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
          rs.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(z.data_ptr()), amax.data_ptr<float>(),
          amax.data_ptr<float>() + 1, rows, (uint32_t)sign_mask);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return {z, rs, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
    }
    dim3 grid3((unsigned)(kCols / kTC3), (unsigned)(rows / kTR3));
    auto apply = version == 5   ? rmsnorm_apply_amax_kernel_v5
                 : version == 4 ? rmsnorm_apply_amax_kernel<kTC3 + 8>
                                : rmsnorm_apply_amax_kernel<kTC3>;
    apply<<<grid3, kThreads3, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        rs.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(z.data_ptr()), amax.data_ptr<float>(),
        amax.data_ptr<float>() + 1, (uint32_t)sign_mask);
  } else if (version == 2) {
    rmsnorm_fwd_amax_kernel_v2<<<(unsigned)(rows / kRowsPerCta), kThreads, smem2, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(z.data_ptr()), rs.data_ptr<float>(), amax.data_ptr<float>(),
        amax.data_ptr<float>() + 1, rows, (float)eps, (uint32_t)sign_mask);
  } else {
    rmsnorm_fwd_amax_kernel<<<(unsigned)(rows / kRowsPerCta), kThreads, smem1, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(z.data_ptr()), rs.data_ptr<float>(), amax.data_ptr<float>(),
        amax.data_ptr<float>() + 1, rows, (float)eps, (uint32_t)sign_mask);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {z, rs, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
}

template <int R, int MINB>
void launch_add_stats_v2(const at::Tensor& r, const at::Tensor& o, at::Tensor& h, at::Tensor& rs, int64_t rows,
                         double eps) {
  const int64_t per_cta = (int64_t)kAS2Warps * R;
  rmsnorm_add_stats_kernel_v2<R, MINB><<<(unsigned)((rows + per_cta - 1) / per_cta), kAS2Warps * 32, 0,
                                         at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(r.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr()), rs.data_ptr<float>(), rows, (float)eps);
}

// rmsnorm_add_stats(r, o [.., 4096] bf16, eps, version) -> (h = bf16(r + o), rsigma f32 [rows] of h).  version 1: one
// row per warp (v1); 2..7: the pipelined kernel v2 at (rows per warp R, CTAs per SM requested): 2 (2, 6), 3 (4, 6),
// 4 (2, 8), 5 (4, 8), 6 (1, 8), 7 (8, 6) -- the bench round picks one.
std::vector<at::Tensor> rmsnorm_add_stats(const at::Tensor& r, const at::Tensor& o, double eps, int64_t version) {
  TORCH_CHECK(r.is_cuda() && r.scalar_type() == at::kBFloat16 && r.is_contiguous(), "r: contiguous CUDA bf16");
  TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kBFloat16 && o.is_contiguous() && o.sizes() == r.sizes(),
              "o: r's shape, contiguous CUDA bf16");
  TORCH_CHECK(r.size(-1) == kCols, "hidden must be 4096");
  TORCH_CHECK(version >= 1 && version <= 7, "rmsnorm_add_stats: version 1..7");
  const int64_t rows = r.numel() / kCols;
  const at::cuda::CUDAGuard guard(r.device());
  auto h = at::empty_like(r);
  auto rs = at::empty({rows}, r.options().dtype(at::kFloat));
  switch (version) {
    case 2: launch_add_stats_v2<2, 6>(r, o, h, rs, rows, eps); break;
    case 3: launch_add_stats_v2<4, 6>(r, o, h, rs, rows, eps); break;
    case 4: launch_add_stats_v2<2, 8>(r, o, h, rs, rows, eps); break;
    case 5: launch_add_stats_v2<4, 8>(r, o, h, rs, rows, eps); break;
    case 6: launch_add_stats_v2<1, 8>(r, o, h, rs, rows, eps); break;
    case 7: launch_add_stats_v2<8, 6>(r, o, h, rs, rows, eps); break;
    default:
      rmsnorm_add_stats_kernel<<<(unsigned)((rows + kStatsWarps - 1) / kStatsWarps), kStatsWarps * 32, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16*>(r.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(h.data_ptr()), rs.data_ptr<float>(), rows, (float)eps);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {h, rs};
}

// rmsnorm_apply_amax(x, gamma, rsigma, sign_mask, version) -> (z, amax_pre, amax_post): the apply pass (v4, or v5 with
// the epilogue of ak_l8b_rht_epi.cuh) with a given rsigma
std::vector<at::Tensor> rmsnorm_apply_amax(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& rs,
                                           int64_t sign_mask, int64_t version) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "x: contiguous CUDA bf16");
  TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous() &&
                  gamma.numel() == kCols, "gamma: contiguous CUDA bf16 [4096]");
  TORCH_CHECK(x.size(-1) == kCols, "hidden must be 4096");
  const int64_t rows = x.numel() / kCols;
  TORCH_CHECK(rows % 32 == 0, "rows % 32 required");
  TORCH_CHECK(rs.is_cuda() && rs.scalar_type() == at::kFloat && rs.is_contiguous() && rs.numel() == rows, "rs: f32 [rows]");
  const at::cuda::CUDAGuard guard(x.device());
  auto z = at::empty_like(x);
  auto amax = at::zeros({2}, x.options().dtype(at::kFloat));
  dim3 grid3((unsigned)(kCols / kTC3), (unsigned)(rows / kTR3));
  if (version == 6) {
    const int64_t nblocks = (rows / kAB_R) * (kCols / kAB_C);
    rmsnorm_apply_amax_kernel_v6<<<(unsigned)((nblocks + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        rs.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(z.data_ptr()), amax.data_ptr<float>(),
        amax.data_ptr<float>() + 1, rows, (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {z, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  auto apply = version == 5 ? rmsnorm_apply_amax_kernel_v5 : rmsnorm_apply_amax_kernel<kTC3 + 8>;
  apply<<<grid3, kThreads3, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
      rs.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(z.data_ptr()), amax.data_ptr<float>(),
      amax.data_ptr<float>() + 1, (uint32_t)sign_mask);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {z, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_add_stats", &rmsnorm_add_stats, "h = bf16(r + o) and TE's RMSNorm rsigma of h (hidden 4096)",
        py::arg("r"), py::arg("o"), py::arg("eps"), py::arg("version") = 1);
  m.def("rmsnorm_apply_amax", &rmsnorm_apply_amax, "RMSNorm apply pass (v4 / v5) with a given rsigma + pre/post-RHT amax",
        py::arg("x"), py::arg("gamma"), py::arg("rs"), py::arg("sign_mask"), py::arg("version") = 4);
  m.def("rmsnorm_fwd_amax", &rmsnorm_fwd_amax, "RMSNorm forward (TE arithmetic, hidden 4096) + pre/post-RHT amax",
        py::arg("x"), py::arg("gamma"), py::arg("eps"), py::arg("sign_mask"), py::arg("version") = 2);
}
