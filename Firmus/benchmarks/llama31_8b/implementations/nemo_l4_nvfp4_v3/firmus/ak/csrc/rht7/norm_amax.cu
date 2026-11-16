// norm_amax.cu (Llama 3.1 8B kernel campaign, 2026-10-07, NVFP4P version 14): the first pass of the RMSNorm forward
// whose apply is folded into the RHT cast.  Version 13g's norm forward is: TE's rsigma (from the residual add, or
// rmsnorm_stats) -> rmsnorm_apply_amax_kernel_v6 (z = bf16(gamma * (rs * x)) written out, with its pre/post-RHT
// amaxes) -> the RHT cast reading z.  Here the apply pass computes the same z in registers and only its two amaxes
// (rmsnorm_amax); the cast forms z again from x, rsigma and gamma (rht_cast_norm, norm_variant.cuh), so z never goes
// through memory.
//
// rmsnorm_amax: ak_l8b_nvfp4n's rmsnorm_apply_amax_kernel_v6 without its store of z -- the same warp-owned 16-row x
// 64-feature blocks, the same per-element arithmetic (y = rs * x, t = gamma * y, both IEEE fp32 products, then
// cvt.rn.bf16x2.f32), the pre-RHT amax by max.xorsign.abs.bf16x2 over the packed z and the post-RHT amax of each
// 16 x 16 sub-tile by ldmatrix.trans + mma.sync with TE's RHT matrix (ak_l8b_rht_epi.cuh).  Variants change only the
// order of the blocks (REV: the last rows first, so the rows the residual add wrote last are read while still in L2,
// and the cast that follows finds the first rows there) and how many loads a lane has in flight (ALL: the block's four
// rows at once).  rmsnorm_stats: ak_l8b_nvfp4n's rmsnorm_stats_kernel (TE's rsigma, one warp per row), for the norm
// whose input is not a residual sum (layer 0's).  Compiled WITHOUT fast math, as ak_l8b_nvfp4n.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include "ak_l8b_rht_epi.cuh"

namespace ak_nrm7 {

constexpr int kCols = 4096;
constexpr int kGroupThreads = 128;           // TE: WARPS_N 4 x 32
constexpr int kLdgs = 4;                     // TE: LDGS = VEC_COLS / VEC_COLS_PER_LDG = 512 / 128
constexpr int kElts = 8;                     // TE: NUM_ELTS = 16 B / 2 B

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

// ---- ak_l8b_nvfp4n's rmsnorm_stats_kernel, verbatim (TE's Welford order; writes only rsigma) ----
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

// ---- the amax pass: rmsnorm_apply_amax_kernel_v6 without the store of z ----
constexpr int kAB_R = 16, kAB_C = 64, kAB_LD = kAB_C + 8;

// z of one 8-feature vector of row `rs`: v6's per-element arithmetic, packed as v6 packs it
__device__ __forceinline__ void apply8(const V8& xv, const float (&g)[8], float rs, uint32_t (&o)[4]) {
#pragma unroll
  for (int j2 = 0; j2 < 4; ++j2) {
    float f[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int jt = 2 * j2 + h;
      float y_ij = rs * (__bfloat162float(xv.h[jt]));
      float temp_output = g[jt] * y_ij;
      f[h] = temp_output;
    }
    o[j2] = ak_rht::cvt_bf16x2(f[0], f[1]);
  }
}

// as apply8 with the two products of each feature pair as packed mul.rn.f32x2 (the same IEEE products)
__device__ __forceinline__ void apply8_x2(const V8& xv, const float (&g)[8], float rs, uint32_t (&o)[4]) {
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv.u);
#pragma unroll
  for (int j2 = 0; j2 < 4; ++j2) {
    const float x0 = __uint_as_float(xw[j2] << 16), x1 = __uint_as_float(xw[j2] & 0xffff0000u);
    float t0, t1;
    asm("{\n\t.reg .b64 ta, tb, td;\n\t"
        "mov.b64 ta, {%2, %3};\n\t"
        "mov.b64 tb, {%4, %4};\n\t"
        "mul.rn.f32x2 td, ta, tb;\n\t"
        "mov.b64 tb, {%5, %6};\n\t"
        "mul.rn.f32x2 td, tb, td;\n\t"
        "mov.b64 {%0, %1}, td;\n\t}"
        : "=f"(t0), "=f"(t1) : "f"(x0), "f"(x1), "f"(rs), "f"(g[2 * j2]), "f"(g[2 * j2 + 1]));
    o[j2] = ak_rht::cvt_bf16x2(t0, t1);
  }
}

template <bool REV, bool ALL, int MINB>
__global__ void __launch_bounds__(256, MINB)
    rmsnorm_amax_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma,
                        const float* __restrict__ rs_in, float* __restrict__ amax_pre, float* __restrict__ amax_post,
                        int64_t rows, uint32_t sign_mask) {
  constexpr int ITERS = kAB_R * kAB_C / 8 / 32;      // 4
  __shared__ __align__(128) __nv_bfloat16 sh[8][kAB_R * kAB_LD];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __nv_bfloat16* s0 = sh[warp];
  constexpr int64_t ncb = kCols / kAB_C;
  const int64_t nblocks = (rows / kAB_R) * ncb;
  const int64_t bf = (int64_t)blockIdx.x * 8 + warp;
  const int rr = lane >> 3, c8 = (lane & 7) * 8;
  uint32_t pre0 = 0u, pre1 = 0u, post = 0u;
  const ak_rht::RhtB B(sign_mask);
  if (bf < nblocks) {
    const int64_t b = REV ? nblocks - 1 - bf : bf;
    const int64_t rowb = (b / ncb) * kAB_R, colb = (b % ncb) * kAB_C + c8;
    V8 gv;
    gv.u = *reinterpret_cast<const uint4*>(gamma + colb);
    float g[8];
#pragma unroll
    for (int jt = 0; jt < 8; ++jt) g[jt] = __bfloat162float(gv.h[jt]);
    if constexpr (ALL) {
      V8 xv[ITERS];
      float rsv[ITERS];
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        xv[it].u = *reinterpret_cast<const uint4*>(x + (rowb + it * 4 + rr) * kCols + colb);
        rsv[it] = rs_in[rowb + it * 4 + rr];
      }
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const int r = it * 4 + rr;
        uint32_t o[4];
        apply8(xv[it], g, rsv[it], o);
#pragma unroll
        for (int j2 = 0; j2 < 4; ++j2) {
          if (j2 & 1) pre1 = ak_rht::bmax_xa(pre1, o[j2]); else pre0 = ak_rht::bmax_xa(pre0, o[j2]);
        }
        *reinterpret_cast<uint4*>(s0 + r * kAB_LD + c8) = make_uint4(o[0], o[1], o[2], o[3]);
      }
    } else {
      V8 nx;
      float nrs;
      nx.u = *reinterpret_cast<const uint4*>(x + (rowb + rr) * kCols + colb);
      nrs = rs_in[rowb + rr];
#pragma unroll 4
      for (int it = 0; it < ITERS; ++it) {
        const int r = it * 4 + rr;
        const int64_t gi = rowb + r;
        const V8 xv = nx;
        const float rs = nrs;
        if (it + 1 < ITERS) {
          nx.u = *reinterpret_cast<const uint4*>(x + (gi + 4) * kCols + colb);
          nrs = rs_in[gi + 4];
        }
        uint32_t o[4];
        apply8(xv, g, rs, o);
#pragma unroll
        for (int j2 = 0; j2 < 4; ++j2) {
          if (j2 & 1) pre1 = ak_rht::bmax_xa(pre1, o[j2]); else pre0 = ak_rht::bmax_xa(pre0, o[j2]);
        }
        *reinterpret_cast<uint4*>(s0 + r * kAB_LD + c8) = make_uint4(o[0], o[1], o[2], o[3]);
      }
    }
    __syncwarp();
#pragma unroll
    for (int fb = 0; fb < kAB_C / 16; ++fb) post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kAB_LD>(s0 + fb * 16, B));
  }
  ak_rht::block_publish2<8>(ak_rht::fmax_noftz(ak_rht::unpack_bmax(pre0), ak_rht::unpack_bmax(pre1)),
                            ak_rht::unpack_bmax(post), amax_pre, amax_post);
}

// persistent versions: gridDim.x = SMs x MINB CTAs; each warp walks the blocks with a stride of every warp in the grid,
// the next block's four rows (and its gamma vector) loaded while the current one is computed; one publish per CTA.
// ORDER: 0 forward, 1 every block reversed, 2 the 16-row blocks reversed (columns forward within each); X2: the two
// products of each feature pair as packed mul.rn.f32x2; STORE: z written too (apply v6's output, for the cast that
// reads it).
template <int ORDER, int MINB, bool X2, bool STORE>
__global__ void __launch_bounds__(256, MINB)
    rmsnorm_norm_kernel_p(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma,
                          const float* __restrict__ rs_in, __nv_bfloat16* __restrict__ z, float* __restrict__ amax_pre,
                          float* __restrict__ amax_post, int64_t rows, uint32_t sign_mask) {
  constexpr int ITERS = kAB_R * kAB_C / 8 / 32;      // 4
  __shared__ __align__(128) __nv_bfloat16 sh[8][kAB_R * kAB_LD];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __nv_bfloat16* s0 = sh[warp];
  constexpr int64_t ncb = kCols / kAB_C;
  const int64_t nrb = rows / kAB_R;
  const int64_t nblocks = nrb * ncb;
  const int64_t stride = (int64_t)gridDim.x * 8;
  const int rr = lane >> 3, c8 = (lane & 7) * 8;
  uint32_t pre0 = 0u, pre1 = 0u, post = 0u;
  const ak_rht::RhtB B(sign_mask);
  V8 cx[ITERS], nx[ITERS];
  float crs[ITERS], nrs[ITERS];
  V8 cg, ng;
  int64_t cpos = 0, npos = 0;                       // element offset of lane's (row rr, column) in the block
  auto load = [&](int64_t bf, V8 (&xv)[ITERS], float (&r4)[ITERS], V8& gv, int64_t& pos) {
    int64_t b;
    if constexpr (ORDER == 1) b = nblocks - 1 - bf;
    else if constexpr (ORDER == 2) b = (nrb - 1 - bf / ncb) * ncb + bf % ncb;
    else b = bf;
    const int64_t rowb = (b / ncb) * kAB_R, colb = (b % ncb) * kAB_C + c8;
    pos = (rowb + rr) * kCols + colb;
    gv.u = *reinterpret_cast<const uint4*>(gamma + colb);
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      xv[it].u = *reinterpret_cast<const uint4*>(x + pos + (int64_t)it * 4 * kCols);
      r4[it] = rs_in[rowb + it * 4 + rr];
    }
  };
  int64_t bf = (int64_t)blockIdx.x * 8 + warp;
  if (bf < nblocks) load(bf, cx, crs, cg, cpos);
  for (; bf < nblocks; bf += stride) {
    const bool more = bf + stride < nblocks;
    if (more) load(bf + stride, nx, nrs, ng, npos);
    float g[8];
#pragma unroll
    for (int jt = 0; jt < 8; ++jt) g[jt] = __bfloat162float(cg.h[jt]);
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int r = it * 4 + rr;
      uint32_t o[4];
      if constexpr (X2) apply8_x2(cx[it], g, crs[it], o); else apply8(cx[it], g, crs[it], o);
#pragma unroll
      for (int j2 = 0; j2 < 4; ++j2) {
        if (j2 & 1) pre1 = ak_rht::bmax_xa(pre1, o[j2]); else pre0 = ak_rht::bmax_xa(pre0, o[j2]);
      }
      const uint4 uo = make_uint4(o[0], o[1], o[2], o[3]);
      if constexpr (STORE) *reinterpret_cast<uint4*>(z + cpos + (int64_t)it * 4 * kCols) = uo;
      *reinterpret_cast<uint4*>(s0 + r * kAB_LD + c8) = uo;
    }
    __syncwarp();
#pragma unroll
    for (int fb = 0; fb < kAB_C / 16; ++fb) post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kAB_LD>(s0 + fb * 16, B));
    __syncwarp();
    if (more) {
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        cx[it] = nx[it];
        crs[it] = nrs[it];
      }
      cg = ng;
      cpos = npos;
    }
  }
  ak_rht::block_publish2<8>(ak_rht::fmax_noftz(ak_rht::unpack_bmax(pre0), ak_rht::unpack_bmax(pre1)),
                            ak_rht::unpack_bmax(post), amax_pre, amax_post);
}

// OVER: CTAs per resident slot (SMs x MINB); above 1 the work is split finer, so CTAs that cannot start at once (SMs
// held by a concurrent kernel, e.g. a collective) leave a shorter tail
template <int ORDER, int MINB, bool X2, bool STORE, int OVER = 1>
void launch_norm_p(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& rs, __nv_bfloat16* z, at::Tensor& amax,
                   int64_t rows, uint32_t mask) {
  const int64_t nblocks = (rows / kAB_R) * (kCols / kAB_C);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t ctas = std::min<int64_t>((int64_t)sms * MINB * OVER, (nblocks + 7) / 8);
  rmsnorm_norm_kernel_p<ORDER, MINB, X2, STORE><<<(unsigned)ctas, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
      rs.data_ptr<float>(), z, amax.data_ptr<float>(), amax.data_ptr<float>() + 1, rows, mask);
}

template <bool REV, bool ALL, int MINB>
void launch_amax(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& rs, at::Tensor& amax, int64_t rows,
                 uint32_t mask) {
  const int64_t nblocks = (rows / kAB_R) * (kCols / kAB_C);
  rmsnorm_amax_kernel<REV, ALL, MINB><<<(unsigned)((nblocks + 7) / 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
      rs.data_ptr<float>(), amax.data_ptr<float>(), amax.data_ptr<float>() + 1, rows, mask);
}

}  // namespace ak_nrm7

// rmsnorm_amax(x [.., 4096] bf16, gamma [4096] bf16, rsigma f32 [rows], sign_mask, version) -> (amax_pre, amax_post):
// the two amaxes of z = bf16(gamma * (rs * x)) without writing z.  version: 1 forward / prefetch one row (v6's loop),
// 2 reversed / prefetch one, 3 forward / all four rows, 4 reversed / all four, 5 reversed / all four at 8 CTAs per SM;
// persistent (one CTA per SM x MINB, each warp prefetching its next block): 6 reversed MINB 3, 7 forward MINB 3,
// 8 reversed MINB 2, 9 reversed MINB 4; 10, 11: 6 and 9 with the products as packed mul.rn.f32x2; 12 the 16-row blocks
// reversed (columns forward) and 13 forward, both MINB 3 and packed.
std::vector<at::Tensor> rmsnorm_amax(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& rs, int64_t sign_mask,
                                     int64_t version) {
  using namespace ak_nrm7;
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "x: contiguous CUDA bf16");
  TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous() &&
                  gamma.numel() == kCols, "gamma: contiguous CUDA bf16 [4096]");
  TORCH_CHECK(x.size(-1) == kCols, "hidden must be 4096");
  const int64_t rows = x.numel() / kCols;
  TORCH_CHECK(rows % 32 == 0, "rows % 32 required");
  TORCH_CHECK(rs.is_cuda() && rs.scalar_type() == at::kFloat && rs.is_contiguous() && rs.numel() == rows, "rs: f32 [rows]");
  const at::cuda::CUDAGuard guard(x.device());
  auto amax = at::zeros({2}, x.options().dtype(at::kFloat));
  const uint32_t mask = (uint32_t)sign_mask;
  switch (version) {
    case 1: launch_amax<false, false, 5>(x, gamma, rs, amax, rows, mask); break;
    case 2: launch_amax<true, false, 5>(x, gamma, rs, amax, rows, mask); break;
    case 3: launch_amax<false, true, 5>(x, gamma, rs, amax, rows, mask); break;
    case 4: launch_amax<true, true, 5>(x, gamma, rs, amax, rows, mask); break;
    case 5: launch_amax<true, true, 8>(x, gamma, rs, amax, rows, mask); break;
    case 6: launch_norm_p<1, 3, false, false>(x, gamma, rs, nullptr, amax, rows, mask); break;
    case 7: launch_norm_p<0, 3, false, false>(x, gamma, rs, nullptr, amax, rows, mask); break;
    case 8: launch_norm_p<1, 2, false, false>(x, gamma, rs, nullptr, amax, rows, mask); break;
    case 9: launch_norm_p<1, 4, false, false>(x, gamma, rs, nullptr, amax, rows, mask); break;
    case 10: launch_norm_p<1, 3, true, false>(x, gamma, rs, nullptr, amax, rows, mask); break;
    case 11: launch_norm_p<1, 4, true, false>(x, gamma, rs, nullptr, amax, rows, mask); break;
    case 12: launch_norm_p<2, 3, true, false>(x, gamma, rs, nullptr, amax, rows, mask); break;
    case 13: launch_norm_p<0, 3, true, false>(x, gamma, rs, nullptr, amax, rows, mask); break;
    default: TORCH_CHECK(false, "rmsnorm_amax: version 1..13");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
}

// rmsnorm_apply_amax7(x, gamma, rsigma, sign_mask, version) -> (z, amax_pre, amax_post): ak_l8b_nvfp4n's
// rmsnorm_apply_amax (v6: z = bf16(gamma * (rs * x)) written, and its two amaxes) in the persistent schedule above:
// version 1 forward, 2 every block reversed, 3 the 16-row blocks reversed; all MINB 3 with packed products; 4 as 1, 5 as
// 3 with scalar products; 6, 7, 8: 5 with 2, 4, 8 CTAs per resident slot.
std::vector<at::Tensor> rmsnorm_apply_amax7(const at::Tensor& x, const at::Tensor& gamma, const at::Tensor& rs,
                                            int64_t sign_mask, int64_t version) {
  using namespace ak_nrm7;
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
  const uint32_t mask = (uint32_t)sign_mask;
  auto zp = reinterpret_cast<__nv_bfloat16*>(z.data_ptr());
  switch (version) {
    case 1: launch_norm_p<0, 3, true, true>(x, gamma, rs, zp, amax, rows, mask); break;
    case 2: launch_norm_p<1, 3, true, true>(x, gamma, rs, zp, amax, rows, mask); break;
    case 3: launch_norm_p<2, 3, true, true>(x, gamma, rs, zp, amax, rows, mask); break;
    case 4: launch_norm_p<0, 3, false, true>(x, gamma, rs, zp, amax, rows, mask); break;
    case 5: launch_norm_p<2, 3, false, true>(x, gamma, rs, zp, amax, rows, mask); break;
    case 6: launch_norm_p<2, 3, false, true, 2>(x, gamma, rs, zp, amax, rows, mask); break;
    case 7: launch_norm_p<2, 3, false, true, 4>(x, gamma, rs, zp, amax, rows, mask); break;
    case 8: launch_norm_p<2, 3, false, true, 8>(x, gamma, rs, zp, amax, rows, mask); break;
    default: TORCH_CHECK(false, "rmsnorm_apply_amax7: version 1..8");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {z, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
}

namespace ak_nrm7 {
__global__ void __launch_bounds__(1024, 1) sm_hog_kernel(int64_t ns, int* sink) {
  // a stand-in for a concurrent collective: holds an SM (1024 threads x 64 registers) for ns nanoseconds
  uint64_t t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  float acc[48];
#pragma unroll
  for (int i = 0; i < 48; ++i) acc[i] = (float)(threadIdx.x + i);
  uint64_t t = t0;
  while ((int64_t)(t - t0) < ns) {
#pragma unroll
    for (int i = 0; i < 48; ++i) acc[i] = acc[i] * 1.0000001f + 1e-7f;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 48; ++i) s += acc[i];
  if (s == 12345.678f) sink[0] = 1;
}
}  // namespace ak_nrm7

// sm_hog(ctas, ns): the test's stand-in for a concurrent collective on the current stream
void sm_hog(int64_t ctas, int64_t ns, at::Tensor sink) {
  ak_nrm7::sm_hog_kernel<<<(unsigned)ctas, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(ns, sink.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// rmsnorm_stats(x [.., 4096] bf16, eps) -> rsigma f32 [rows] (TE's, as ak_l8b_nvfp4n.rmsnorm_fwd_amax computes it)
at::Tensor rmsnorm_stats(const at::Tensor& x, double eps) {
  using namespace ak_nrm7;
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "x: contiguous CUDA bf16");
  TORCH_CHECK(x.size(-1) == kCols, "hidden must be 4096");
  const int64_t rows = x.numel() / kCols;
  const at::cuda::CUDAGuard guard(x.device());
  auto rs = at::empty({rows}, x.options().dtype(at::kFloat));
  rmsnorm_stats_kernel<<<(unsigned)((rows + kStatsWarps - 1) / kStatsWarps), kStatsWarps * 32, 0,
                         at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                                             rs.data_ptr<float>(), rows, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rs;
}
