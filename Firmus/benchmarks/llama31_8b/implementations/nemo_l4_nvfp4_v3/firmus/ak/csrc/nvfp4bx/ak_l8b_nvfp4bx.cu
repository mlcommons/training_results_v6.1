// ak_l8b_nvfp4b.cu (Llama 3.1 8B kernel campaign, 2026-10-03): RMSNorm backward (hidden 4096, bf16) fused with the
// residual-gradient add that follows it, and the two amaxes TransformerEngine's NVFP4 quantizer needs for the sum.
//
// Stock (PROF-L0 / PROF-L2f), per norm backward of a layer (64 per step):
//   rmsnorm_bwd_tuned_kernel (dx, dgamma partials)                     83 us   reads x, dz; writes dx
//   rmsnorm_bwd_finalize_tuned_kernel (dgamma)                          3.5 us
//   autograd's residual add  g = dx + g_res  (vectorized CUDAFunctor_add) 58 us   reads dx, g_res; writes g
//   HadamardAmaxTmaKernel on g (the next linear's gradient quantize)    27 us   reads g
// Here one kernel computes dx exactly as TE does, rounds it to bf16 as TE's store does, adds the residual gradient as
// autograd's add does (bf16(float(dx_bf16) + float(g_res))), writes only the sum, and takes its pre-RHT amax; the
// finalize is TE's; a second small kernel takes the post-RHT amax of the sum.
//
// E0 by construction:
//  * TE's launcher for (4096, bf16, bf16, bf16, fp32) is the tuned one with CTAS_PER_ROW 1, WARPS_M 1, WARPS_N 4,
//    16-byte loads (rmsnorm_bwd_semi_cuda_kernel.cu:233): ctas_per_col = (SMs - sm_margin) * ctas_per_sm CTAs of 128
//    threads, CTA b taking rows b, b + ctas_per_col, ... in order; thread t owning the 8-element vectors t, t+128,
//    t+256, t+384 of a row.  The dgamma partial of CTA b is the per-element FMA chain over its rows in that order, so
//    this kernel keeps the same grid (ctas_per_col is an argument: TE's NVRTC build of the kernel has 167 registers,
//    3 CTAs per SM; PROF-L0 grid 408 = (152 - 16) * 3) and the same row order.
//  * Per element (rmsnorm_bwd_kernels.cuh:118-162, NVRTC with the default --fmad=true): y = rs * x; dy = (gamma +
//    shift) * dz; mdyy += dy * y (FMA); dzy += dz * y (FMA); the row sum is Reducer<float2, 1, 1, 4>::allreduce
//    (shfl_down 16..1 to lane 0, then 0 + s0 + s1 + s2 + s3); mdyy *= 1/4096; dx = rs * (dy - mdyy * y) with the
//    product mdyy * y fused into the subtraction (the multiply with fewer uses, as LLVM's FMA combine prefers).  The
//    multiplies that TE's compiler leaves unfused are written __fmul_rn so this build cannot fuse them differently.
//  * The finalize is rmsnorm_bwd_finalize_tuned_kernel for Kernel_traits_finalize<4096, ..., 1024 threads, 4 B>:
//    128 CTAs, warp w summing partial rows w, w+32, ... per column, an xor-transposed shared tile, the xor-butterfly
//    allreduce across the 32 warp sums, __float22bfloat162_rn.
//  * The post-RHT amax is max |bf16(T^T-tile x RHT)| over 16-token x 16-feature tiles with the bf16 m16n16k16 MMA TE
//    uses (as ak_l8b_nvfp4p / ak_l8b_nvfp4n); the pre-RHT amax is max |bf16 sum| (NaN ignored, as TE's).
// 2026-10-07 dev bw19: rmsnorm_bwd_add version 6 (= 3 with the post-RHT pass in reversed row order, plain loads) and
// rht_amax version 2, the default (reversed rows, plain loads): see rht_amax_post_kernel_v3, rht_amax_kernel_r.
// 2026-10-08 dev ci27 (module ak_l8b_nvfp4bx): rmsnorm_bwd_add version 7 -- rmsnorm_bwd_add_v4 (v2 with fewer
// instructions per element, the same bytes), then version 6's post-RHT amax pass.  See the version 7 section.
// 2026-10-08 dev c18: rmsnorm_bwd_add takes an optional main_grad (the norm weight's bf16 [4096] gradient buffer): the
// finalize then also writes main_grad = bf16(float(main_grad) + float(dgamma_bf16)) -- what the DDP hook's
// main_grad.add_(param.grad) computes (torch's CUDAFunctor_add<BFloat16>: opmath float, alpha 1, one round to nearest
// even) -- so the hook's 65 small adds per step can be skipped (rmsnorm_bwd_finalize_mg; the plain finalize unchanged).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>

#include "ak_l8b_rht_epi.cuh"

using namespace nvcuda;

namespace {

constexpr int kCols = 4096;
constexpr int kThreads = 128;                 // TE: WARPS_M 1 x WARPS_N 4 x 32
constexpr int kLdgs = 4;                      // TE: LDGS = VEC_COLS (512) / VEC_COLS_PER_LDG (128)
constexpr int kElts = 8;                      // TE: NUM_ELTS = 16 B / 2 B
constexpr int kVecCols = kCols / kElts;       // 512
constexpr int kRowBytes = kCols * 2;          // 8192

union V8 {
  uint4 u;
  __nv_bfloat16 h[8];
};

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ void atomic_max_nonneg(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

__device__ __forceinline__ void block_max_publish(float v, float* dst) {
  __shared__ float red_max[32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
  if (lane == 0) red_max[warp] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = 0.f;
    for (int w = 0; w < nw; ++w) m = fmaxf(m, red_max[w]);
    atomic_max_nonneg(dst, m);
  }
}

// ---- the per-row arithmetic, TE's (see the header) ------------------------------------------------------------------
struct RowMath {
  // pass 1 over one 8-element vector: the mdyy partial and the dgamma partials
  static __device__ __forceinline__ void pass1(const V8& xv, const V8& dv, const V8& gv, float rs_r, float shift,
                                               float& mdyy_local, float* dzy) {
#pragma unroll
    for (int jt = 0; jt < kElts; jt++) {
      const float x_tmp = bf(xv.h[jt]);
      const float y_tmp = __fmul_rn(rs_r, x_tmp);
      float dy_tmp = __fadd_rn(bf(gv.h[jt]), shift);
      dy_tmp = __fmul_rn(dy_tmp, bf(dv.h[jt]));
      const float dz_tmp = bf(dv.h[jt]);
      mdyy_local = __fmaf_rn(dy_tmp, y_tmp, mdyy_local);
      dzy[jt] = __fmaf_rn(dz_tmp, y_tmp, dzy[jt]);
    }
  }
  // pass 2 over one vector: dx (bf16, TE's store), then + the residual gradient (autograd's add); amax of the result
  template <bool kAdd>
  static __device__ __forceinline__ V8 pass2(const V8& xv, const V8& dv, const V8& gv, const V8& av, float rs_r,
                                             float shift, float mdyy, float& a_pre) {
    V8 o;
#pragma unroll
    for (int jt = 0; jt < kElts; jt++) {
      const float y_tmp = __fmul_rn(rs_r, bf(xv.h[jt]));
      float dy_tmp = __fadd_rn(bf(gv.h[jt]), shift);
      dy_tmp = __fmul_rn(dy_tmp, bf(dv.h[jt]));
      const float dx_tmp = __fmul_rn(rs_r, __fmaf_rn(-mdyy, y_tmp, dy_tmp));
      const __nv_bfloat16 dxb = __float2bfloat16_rn(dx_tmp);
      o.h[jt] = kAdd ? __float2bfloat16_rn(__fadd_rn(bf(dxb), bf(av.h[jt]))) : dxb;
      a_pre = fmaxf(a_pre, fabsf(bf(o.h[jt])));
    }
    return o;
  }
  // Reducer<float2, 1, 1, 4>::allreduce on {0, v}: Base::reduce (shfl_down 16..1, lane 0 holds), smem slot per warp,
  // one __syncthreads, every thread sums 0 + s0 + s1 + s2 + s3.  Two alternating slots, as TE's use0_.
  static __device__ __forceinline__ float row_allreduce(float v, float* slot, int warp, int lane) {
#pragma unroll
    for (int it = 16; it > 0; it /= 2) v = v + __shfl_down_sync(0xffffffffu, v, it);
    if (lane == 0) slot[warp] = v;
    __syncthreads();
    float out = 0.f;
#pragma unroll
    for (int it = 0; it < 4; it++) out = out + slot[it];
    return out;
  }
};

__device__ __forceinline__ void store_dgamma_part(float* dgamma_part, int r, int c, const float (&dzy)[kLdgs * kElts]) {
  // TE (WARPS_M == 1): idx = r * VEC_COLS + c; Cvec store_to(dgamma_part, idx); idx += VEC_COLS_PER_LDG
#pragma unroll
  for (int it = 0; it < kLdgs; it++) {
    float4* p = reinterpret_cast<float4*>(dgamma_part + ((int64_t)r * kVecCols + c + it * 128) * kElts);
    p[0] = make_float4(dzy[it * 8 + 0], dzy[it * 8 + 1], dzy[it * 8 + 2], dzy[it * 8 + 3]);
    p[1] = make_float4(dzy[it * 8 + 4], dzy[it * 8 + 5], dzy[it * 8 + 6], dzy[it * 8 + 7]);
  }
}

// ---- v1: TE's loop as written (loads at the top of each row), plus the add and the pre-RHT amax -------------------
template <bool kAdd>
__global__ void __launch_bounds__(kThreads, 3)
    rmsnorm_bwd_add_v1(const __nv_bfloat16* __restrict__ dz, const __nv_bfloat16* __restrict__ x,
                       const float* __restrict__ rs, const __nv_bfloat16* __restrict__ gamma,
                       const __nv_bfloat16* __restrict__ add, __nv_bfloat16* __restrict__ out,
                       float* __restrict__ dgamma_part, float* __restrict__ amax_pre, int rows, int ctas_per_col,
                       float shift) {
  __shared__ float slots[2][4];
  const int tidx = threadIdx.x, lane = tidx % 32, warp = tidx / 32;
  const int r = blockIdx.x;                    // bidm * ROWS_PER_CTA + warp_m
  const int c = warp * 32 + lane;              // bidn * THREADS_PER_ROW + warp_n * 32 + lane
  float dzy[kLdgs * kElts];
#pragma unroll
  for (int i = 0; i < kLdgs * kElts; i++) dzy[i] = 0.f;
  V8 g[kLdgs];
#pragma unroll
  for (int it = 0; it < kLdgs; it++) g[it].u = *reinterpret_cast<const uint4*>(gamma + (c + it * 128) * kElts);
  constexpr float rn = 1.f / static_cast<float>(kCols);
  float a_pre = 0.f;
  int use0 = 1;
#pragma unroll 1
  for (int row = r; row < rows; row += ctas_per_col) {
    const float rs_r = rs[row];
    V8 xv[kLdgs], dv[kLdgs], av[kLdgs];
    const int64_t base = (int64_t)row * kVecCols + c;
#pragma unroll
    for (int it = 0; it < kLdgs; it++) {
      dv[it].u = *reinterpret_cast<const uint4*>(dz + (base + it * 128) * kElts);
      xv[it].u = *reinterpret_cast<const uint4*>(x + (base + it * 128) * kElts);
    }
    float mdyy_local = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs; it++) RowMath::pass1(xv[it], dv[it], g[it], rs_r, shift, mdyy_local, dzy + it * kElts);
    if (kAdd) {
#pragma unroll
      for (int it = 0; it < kLdgs; it++) av[it].u = *reinterpret_cast<const uint4*>(add + (base + it * 128) * kElts);
    }
    float* slot = use0 ? slots[0] : slots[1];
    use0 = !use0;
    const float mdyy = __fmul_rn(RowMath::row_allreduce(mdyy_local, slot, warp, lane), rn);
#pragma unroll
    for (int it = 0; it < kLdgs; it++) {
      const V8 o = RowMath::pass2<kAdd>(xv[it], dv[it], g[it], av[it], rs_r, shift, mdyy, a_pre);
      *reinterpret_cast<uint4*>(out + (base + it * 128) * kElts) = o.u;
    }
  }
  store_dgamma_part(dgamma_part, r, c, dzy);
  block_max_publish(a_pre, amax_pre);
}

// ---- v2: the same rows and arithmetic, the inputs of the next rows already in shared memory (bulk async copies
// with an mbarrier per stage, two stages) ----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "AK_WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra AK_WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

template <bool kAdd>
constexpr int stage_rows() { return kAdd ? 3 : 2; }   // x, dz (, add) rows per stage

template <bool kAdd>
__global__ void __launch_bounds__(kThreads, 3)
    rmsnorm_bwd_add_v2(const __nv_bfloat16* __restrict__ dz, const __nv_bfloat16* __restrict__ x,
                       const float* __restrict__ rs, const __nv_bfloat16* __restrict__ gamma,
                       const __nv_bfloat16* __restrict__ add, __nv_bfloat16* __restrict__ out,
                       float* __restrict__ dgamma_part, float* __restrict__ amax_pre, int rows, int ctas_per_col,
                       float shift) {
  constexpr int kNT = stage_rows<kAdd>();
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(smem_raw);       // [2][kNT][kCols]: x, dz, add
  __shared__ __align__(8) uint64_t full[2];
  __shared__ float slots[2][4];
  const int tidx = threadIdx.x, lane = tidx % 32, warp = tidx / 32;
  const int r = blockIdx.x;
  const int c = warp * 32 + lane;
  const int nrows = r < rows ? (rows - 1 - r) / ctas_per_col + 1 : 0;
  if (tidx == 0) {
    mbar_init(&full[0], 1);
    mbar_init(&full[1], 1);
    mbar_fence_init();
  }
  __syncthreads();
  auto issue = [&](int k, int s) {
    const int64_t row = (int64_t)r + (int64_t)k * ctas_per_col;
    __nv_bfloat16* d = st + (int64_t)s * kNT * kCols;
    mbar_expect_tx(&full[s], kNT * kRowBytes);
    bulk_g2s(d, x + row * kCols, kRowBytes, &full[s]);
    bulk_g2s(d + kCols, dz + row * kCols, kRowBytes, &full[s]);
    if (kAdd) bulk_g2s(d + 2 * kCols, add + row * kCols, kRowBytes, &full[s]);
  };
  if (tidx == 0) {
    if (nrows > 0) issue(0, 0);
    if (nrows > 1) issue(1, 1);
  }
  float dzy[kLdgs * kElts];
#pragma unroll
  for (int i = 0; i < kLdgs * kElts; i++) dzy[i] = 0.f;
  V8 g[kLdgs];
#pragma unroll
  for (int it = 0; it < kLdgs; it++) g[it].u = *reinterpret_cast<const uint4*>(gamma + (c + it * 128) * kElts);
  constexpr float rn = 1.f / static_cast<float>(kCols);
  float a_pre = 0.f;
  int use0 = 1;
#pragma unroll 1
  for (int k = 0; k < nrows; ++k) {
    const int s = k & 1;
    const int64_t row = (int64_t)r + (int64_t)k * ctas_per_col;
    const float rs_r = rs[row];
    mbar_wait(&full[s], (k >> 1) & 1);
    const __nv_bfloat16* sx = st + (int64_t)s * kNT * kCols;
    const __nv_bfloat16* sd = sx + kCols;
    const __nv_bfloat16* sa = sx + 2 * kCols;
    float mdyy_local = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs; it++) {
      V8 xv, dv;
      xv.u = *reinterpret_cast<const uint4*>(sx + (c + it * 128) * kElts);
      dv.u = *reinterpret_cast<const uint4*>(sd + (c + it * 128) * kElts);
      RowMath::pass1(xv, dv, g[it], rs_r, shift, mdyy_local, dzy + it * kElts);
    }
    float* slot = use0 ? slots[0] : slots[1];
    use0 = !use0;
    const float mdyy = __fmul_rn(RowMath::row_allreduce(mdyy_local, slot, warp, lane), rn);
    const int64_t base = row * kVecCols + c;
#pragma unroll
    for (int it = 0; it < kLdgs; it++) {
      V8 xv, dv, av;
      xv.u = *reinterpret_cast<const uint4*>(sx + (c + it * 128) * kElts);
      dv.u = *reinterpret_cast<const uint4*>(sd + (c + it * 128) * kElts);
      if (kAdd) av.u = *reinterpret_cast<const uint4*>(sa + (c + it * 128) * kElts);
      const V8 o = RowMath::pass2<kAdd>(xv, dv, g[it], av, rs_r, shift, mdyy, a_pre);
      *reinterpret_cast<uint4*>(out + (base + it * 128) * kElts) = o.u;
    }
    __syncthreads();                                   // stage s fully read: refill it with row k + 2
    if (tidx == 0 && k + 2 < nrows) issue(k + 2, s);
  }
  store_dgamma_part(dgamma_part, r, c, dzy);
  block_max_publish(a_pre, amax_pre);
}

template <bool kAdd>
constexpr size_t smem_v2() { return sizeof(__nv_bfloat16) * 2 * stage_rows<kAdd>() * kCols; }

// ---- TE's rmsnorm_bwd_finalize_tuned_kernel, Kernel_traits_finalize<4096, bf16, bf16, bf16, fp32, uint32_t, 1024, 4>
__global__ void __launch_bounds__(1024)
    rmsnorm_bwd_finalize(const float* __restrict__ dgamma_part, __nv_bfloat16* __restrict__ dgamma,
                         int ctas_per_col) {
  __shared__ __align__(16) char smem_[2 * 4096 + 2 * 128];   // SMEM_BYTES_PER_CTA
  const uint32_t bidn = blockIdx.x, tidx = threadIdx.x, warp = tidx / 32, lane = tidx % 32;
  const uint32_t c = bidn * 32 + lane;
  const uint32_t c_out = bidn * 32 / 2 + lane;
  constexpr uint32_t COL_STRIDE = 128 * 32;                   // CTAS * THREADS_PER_WARP
  for (uint32_t col = c, col_out = c_out; col < (uint32_t)kCols; col += COL_STRIDE, col_out += COL_STRIDE / 2) {
    float dgamma_local = 0.f;
    for (uint32_t row = warp; row < (uint32_t)ctas_per_col; row += 32) {
      dgamma_local += dgamma_part[row * (uint32_t)kCols + col];
    }
    float* smem_gamma = reinterpret_cast<float*>(smem_);
    const int write_row = warp;
    const int write_col = lane ^ write_row;
    smem_gamma[write_row * 32 + write_col] = dgamma_local;
    __syncthreads();
    float* smem_gamma_out = reinterpret_cast<float*>(&smem_[2 * 4096]);
    for (int w = warp; w < 32; w += 32) {
      const int read_row = lane;
      const int read_col = w ^ read_row;
      float g_i = 0.f;
      if (read_row < 32) g_i = smem_gamma[read_row * 32 + read_col];
#pragma unroll
      for (int it = 1; it < 32; it *= 2) g_i = g_i + __shfl_xor_sync(0xffffffffu, g_i, it);
      if (lane == 0) smem_gamma_out[w] = g_i;
    }
    __syncthreads();
    if (warp == 32 - 1 && lane < 32 / 2) {
      const float2 v = reinterpret_cast<const float2*>(smem_gamma_out)[lane];
      reinterpret_cast<__nv_bfloat162*>(dgamma)[col_out] = __float22bfloat162_rn(v);
    }
  }
}

// ---- the finalize with the norm weight's gradient accumulation: rmsnorm_bwd_finalize, then main_grad += dgamma as
// torch's bf16 add (float(main_grad) + float(dgamma_bf16), one round to nearest even), each column pair once
__global__ void __launch_bounds__(1024)
    rmsnorm_bwd_finalize_mg(const float* __restrict__ dgamma_part, __nv_bfloat16* __restrict__ dgamma,
                            __nv_bfloat16* __restrict__ main_grad, int ctas_per_col) {
  __shared__ __align__(16) char smem_[2 * 4096 + 2 * 128];   // SMEM_BYTES_PER_CTA
  const uint32_t bidn = blockIdx.x, tidx = threadIdx.x, warp = tidx / 32, lane = tidx % 32;
  const uint32_t c = bidn * 32 + lane;
  const uint32_t c_out = bidn * 32 / 2 + lane;
  constexpr uint32_t COL_STRIDE = 128 * 32;                   // CTAS * THREADS_PER_WARP
  for (uint32_t col = c, col_out = c_out; col < (uint32_t)kCols; col += COL_STRIDE, col_out += COL_STRIDE / 2) {
    float dgamma_local = 0.f;
    for (uint32_t row = warp; row < (uint32_t)ctas_per_col; row += 32) {
      dgamma_local += dgamma_part[row * (uint32_t)kCols + col];
    }
    float* smem_gamma = reinterpret_cast<float*>(smem_);
    const int write_row = warp;
    const int write_col = lane ^ write_row;
    smem_gamma[write_row * 32 + write_col] = dgamma_local;
    __syncthreads();
    float* smem_gamma_out = reinterpret_cast<float*>(&smem_[2 * 4096]);
    for (int w = warp; w < 32; w += 32) {
      const int read_row = lane;
      const int read_col = w ^ read_row;
      float g_i = 0.f;
      if (read_row < 32) g_i = smem_gamma[read_row * 32 + read_col];
#pragma unroll
      for (int it = 1; it < 32; it *= 2) g_i = g_i + __shfl_xor_sync(0xffffffffu, g_i, it);
      if (lane == 0) smem_gamma_out[w] = g_i;
    }
    __syncthreads();
    if (warp == 32 - 1 && lane < 32 / 2) {
      const float2 v = reinterpret_cast<const float2*>(smem_gamma_out)[lane];
      const __nv_bfloat162 d2 = __float22bfloat162_rn(v);
      reinterpret_cast<__nv_bfloat162*>(dgamma)[col_out] = d2;
      __nv_bfloat162* m2p = reinterpret_cast<__nv_bfloat162*>(main_grad) + col_out;
      const float2 mf = __bfloat1622float2(*m2p);
      const float2 df = __bfloat1622float2(d2);
      *m2p = __floats2bfloat162_rn(__fadd_rn(mf.x, df.x), __fadd_rn(mf.y, df.y));
    }
  }
}

// ---- post-RHT amax of a [rows, 4096] bf16 tensor: 16-token RHT blocks along the rows ---------------------------------
constexpr int kTRb = 32, kTCb = 256, kLDb = kTCb + 8, kThreadsB = 256;

__device__ __forceinline__ void build_rht(__nv_bfloat16* sh_h, uint32_t sign_mask) {
  for (int t = threadIdx.x; t < 256; t += blockDim.x) {
    const int k = t / 16, n = t % 16;
    float v = (__popc(k & n) & 1) ? -0.25f : 0.25f;
    if ((sign_mask >> k) & 1u) v = -v;
    sh_h[t] = __float2bfloat16_rn(v);
  }
}

__device__ __forceinline__ float rht_tile_amax(const __nv_bfloat16* tile, int ld,
                                               const wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                                                                    wmma::row_major>& bfrag) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> afrag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
  wmma::load_matrix_sync(afrag, tile, ld);
  wmma::fill_fragment(cfrag, 0.0f);
  wmma::mma_sync(cfrag, afrag, bfrag, cfrag);
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < cfrag.num_elements; ++i) m = fmaxf(m, fabsf(__bfloat162float(__float2bfloat16_rn(cfrag.x[i]))));
  return m;
}

__global__ void __launch_bounds__(kThreadsB)
    rht_amax_post_kernel(const __nv_bfloat16* __restrict__ t, float* __restrict__ amax_post, uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh[kTRb * kLDb];
  __shared__ __align__(128) __nv_bfloat16 sh_h[256];
  const int64_t row0 = (int64_t)blockIdx.y * kTRb, col0 = (int64_t)blockIdx.x * kTCb;
  constexpr int kVec = kTRb * kTCb / 8 / kThreadsB;   // 4 vectors per thread
  V8 v[kVec];
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    v[i].u = __ldcs(reinterpret_cast<const uint4*>(t + (row0 + rr) * kCols + col0 + c8));
  }
  build_rht(sh_h, sign_mask);
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    *reinterpret_cast<uint4*>(sh + rr * kLDb + c8) = v[i].u;
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a = 0.f;
  constexpr int kSub = (kTRb / 16) * (kTCb / 16);     // 32 sub-tiles
  for (int s = threadIdx.x / 32; s < kSub; s += kThreadsB / 32) {
    const int tb = s / (kTCb / 16), fb = s % (kTCb / 16);
    a = fmaxf(a, rht_tile_amax(sh + tb * 16 * kLDb + fb * 16, kLDb, bfrag));
  }
  block_max_publish(a, amax_post);
}

// v2 (2026-10-05): the same tile and staging with the epilogue of ak_l8b_rht_epi.cuh (TE's own amax instructions:
// ldmatrix.trans + mma.sync with the RHT in registers, cvt.rn.bf16x2.f32, max.xorsign.abs.bf16x2)
__global__ void __launch_bounds__(kThreadsB)
    rht_amax_post_kernel_v2(const __nv_bfloat16* __restrict__ t, float* __restrict__ amax_post, uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh[kTRb * kLDb];
  const int64_t row0 = (int64_t)blockIdx.y * kTRb, col0 = (int64_t)blockIdx.x * kTCb;
  constexpr int kVec = kTRb * kTCb / 8 / kThreadsB;   // 4 vectors per thread
  V8 v[kVec];
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    v[i].u = __ldcs(reinterpret_cast<const uint4*>(t + (row0 + rr) * kCols + col0 + c8));
  }
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    *reinterpret_cast<uint4*>(sh + rr * kLDb + c8) = v[i].u;
  }
  const ak_rht::RhtB B(sign_mask);
  __syncthreads();
  uint32_t post = 0u;
  constexpr int kSub = (kTRb / 16) * (kTCb / 16);     // 32 sub-tiles
#pragma unroll
  for (int s = threadIdx.x / 32; s < kSub; s += kThreadsB / 32) {
    const int tb = s / (kTCb / 16), fb = s % (kTCb / 16);
    post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kLDb>(sh + tb * 16 * kLDb + fb * 16, B));
  }
  ak_rht::block_publish2<kThreadsB / 32>(0.f, ak_rht::unpack_bmax(post), nullptr, amax_post);
}

// Both amaxes of any [rows, N] bf16 tensor (rows % 32, N % 256), TE's HadamardAmaxTmaKernel contract with its own amax
// instructions (2026-10-05): the pre-RHT amax by max.xorsign.abs.bf16x2 over the data, the post-RHT amax of the
// columnwise copy (16-token blocks) by ak_l8b_rht_epi.cuh.  For the quantizes whose producer is not ours (the attention
// output feeding linear_proj, the QKV gradient from the RoPE backward): our amaxes, then TE's cast.
__global__ void __launch_bounds__(kThreadsB)
    rht_amax_kernel(const __nv_bfloat16* __restrict__ t, float* __restrict__ amax, int64_t N, uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh[kTRb * kLDb];
  const int64_t row0 = (int64_t)blockIdx.y * kTRb, col0 = (int64_t)blockIdx.x * kTCb;
  constexpr int kVec = kTRb * kTCb / 8 / kThreadsB;   // 4 vectors per thread
  V8 v[kVec];
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    v[i].u = __ldcs(reinterpret_cast<const uint4*>(t + (row0 + rr) * N + col0 + c8));
  }
  uint32_t pre0 = 0u, pre1 = 0u;
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    *reinterpret_cast<uint4*>(sh + rr * kLDb + c8) = v[i].u;
    pre0 = ak_rht::bmax_xa(pre0, ak_rht::bmax_xa(v[i].u.x, v[i].u.y));
    pre1 = ak_rht::bmax_xa(pre1, ak_rht::bmax_xa(v[i].u.z, v[i].u.w));
  }
  const ak_rht::RhtB B(sign_mask);
  __syncthreads();
  uint32_t post = 0u;
  constexpr int kSub = (kTRb / 16) * (kTCb / 16);     // 32 sub-tiles
#pragma unroll
  for (int s = threadIdx.x / 32; s < kSub; s += kThreadsB / 32) {
    const int tb = s / (kTCb / 16), fb = s % (kTCb / 16);
    post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kLDb>(sh + tb * 16 * kLDb + fb * 16, B));
  }
  ak_rht::block_publish2<kThreadsB / 32>(ak_rht::fmax_noftz(ak_rht::unpack_bmax(pre0), ak_rht::unpack_bmax(pre1)),
                                         ak_rht::unpack_bmax(post), amax, amax + 1);
}

// v3 (2026-10-07, dev bw19): rht_amax_post_kernel_v2 with its 32-row blocks in reverse order and plain loads (not
// __ldcs): it runs right after the norm backward that wrote the tensor, so the rows written last are read first while
// they are in L2, and the rows it reads last -- the first rows -- stay in L2 for the cast that reads from the top (job
// 20934: 133.3 -> 128.1 us for norm backward + post pass + a forward reader; the amax bit-identical: maxima of the same
// sub-tiles in another order)
__global__ void __launch_bounds__(kThreadsB)
    rht_amax_post_kernel_v3(const __nv_bfloat16* __restrict__ t, float* __restrict__ amax_post, uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh[kTRb * kLDb];
  const int64_t row0 = (int64_t)(gridDim.y - 1 - blockIdx.y) * kTRb, col0 = (int64_t)blockIdx.x * kTCb;
  constexpr int kVec = kTRb * kTCb / 8 / kThreadsB;   // 4 vectors per thread
  V8 v[kVec];
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    v[i].u = *reinterpret_cast<const uint4*>(t + (row0 + rr) * kCols + col0 + c8);
  }
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    *reinterpret_cast<uint4*>(sh + rr * kLDb + c8) = v[i].u;
  }
  const ak_rht::RhtB B(sign_mask);
  __syncthreads();
  uint32_t post = 0u;
  constexpr int kSub = (kTRb / 16) * (kTCb / 16);     // 32 sub-tiles
#pragma unroll
  for (int s = threadIdx.x / 32; s < kSub; s += kThreadsB / 32) {
    const int tb = s / (kTCb / 16), fb = s % (kTCb / 16);
    post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kLDb>(sh + tb * 16 * kLDb + fb * 16, B));
  }
  ak_rht::block_publish2<kThreadsB / 32>(0.f, ak_rht::unpack_bmax(post), nullptr, amax_post);
}

// rht_amax_kernel with reversed 32-row blocks and plain loads (the same reasons: its tensor -- the attention output -- was
// just written by its producer and is read next by the cast; job 20934: 86.3 -> 80.2 us for producer + amax pass + a
// forward reader; both amaxes bit-identical)
__global__ void __launch_bounds__(kThreadsB)
    rht_amax_kernel_r(const __nv_bfloat16* __restrict__ t, float* __restrict__ amax, int64_t N, uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh[kTRb * kLDb];
  const int64_t row0 = (int64_t)(gridDim.y - 1 - blockIdx.y) * kTRb, col0 = (int64_t)blockIdx.x * kTCb;
  constexpr int kVec = kTRb * kTCb / 8 / kThreadsB;   // 4 vectors per thread
  V8 v[kVec];
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    v[i].u = *reinterpret_cast<const uint4*>(t + (row0 + rr) * N + col0 + c8);
  }
  uint32_t pre0 = 0u, pre1 = 0u;
#pragma unroll
  for (int i = 0; i < kVec; i++) {
    const int e = threadIdx.x + i * kThreadsB, rr = e / (kTCb / 8), c8 = (e % (kTCb / 8)) * 8;
    *reinterpret_cast<uint4*>(sh + rr * kLDb + c8) = v[i].u;
    pre0 = ak_rht::bmax_xa(pre0, ak_rht::bmax_xa(v[i].u.x, v[i].u.y));
    pre1 = ak_rht::bmax_xa(pre1, ak_rht::bmax_xa(v[i].u.z, v[i].u.w));
  }
  const ak_rht::RhtB B(sign_mask);
  __syncthreads();
  uint32_t post = 0u;
  constexpr int kSub = (kTRb / 16) * (kTCb / 16);     // 32 sub-tiles
#pragma unroll
  for (int s = threadIdx.x / 32; s < kSub; s += kThreadsB / 32) {
    const int tb = s / (kTCb / 16), fb = s % (kTCb / 16);
    post = ak_rht::bmax_xa(post, ak_rht::rht_subtile_bmax<kLDb>(sh + tb * 16 * kLDb + fb * 16, B));
  }
  ak_rht::block_publish2<kThreadsB / 32>(ak_rht::fmax_noftz(ak_rht::unpack_bmax(pre0), ak_rht::unpack_bmax(pre1)),
                                         ak_rht::unpack_bmax(post), amax, amax + 1);
}

// ---- version 7 (2026-10-08, dev ci27): rmsnorm_bwd_add_v4 = v2's grid, rows, staging and arithmetic with fewer
// instructions per element (v2: 907 per 32-element row iteration per thread, 64 single converts, 361 bf16 unpack /
// pack operations):
//  * (bf16(gamma) + shift) once per CTA into shared memory, laid out [8][kThreads] float4 so a warp's reads are
//    conflict-free, read back in both passes instead of an unpack + add per element and row (the same __fadd_rn of the
//    same operands, so the same float);
//  * dx's bf16 and the sum's bf16 by the packed round-to-nearest convert (cvt.rn.bf16x2.f32: each half rounded as the
//    single convert rounds it), which also forms the stored words directly;
//  * the pre-RHT amax by TE's packed max.xorsign.abs.bf16x2 on the stored words (no flush to zero, NaN ignored as
//    fmaxf ignores it), the two halves widened exactly at the end: the maximum of the same values.
// The row loop, the mdyy allreduce, the dgamma FMA chains and their partial store, the block publish and the row ->
// CTA map are v2's.
template <bool kAdd>
constexpr size_t smem_v4() { return smem_v2<kAdd>() + sizeof(float) * kThreads * kLdgs * kElts; }

template <bool kAdd>
__global__ void __launch_bounds__(kThreads, 3)
    rmsnorm_bwd_add_v4(const __nv_bfloat16* __restrict__ dz, const __nv_bfloat16* __restrict__ x,
                       const float* __restrict__ rs, const __nv_bfloat16* __restrict__ gamma,
                       const __nv_bfloat16* __restrict__ add, __nv_bfloat16* __restrict__ out,
                       float* __restrict__ dgamma_part, float* __restrict__ amax_pre, int rows, int ctas_per_col,
                       float shift) {
  constexpr int kNT = stage_rows<kAdd>();
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(smem_raw);       // [2][kNT][kCols]: x, dz, add
  float4* gs4 = reinterpret_cast<float4*>(smem_raw + smem_v2<kAdd>());  // [kLdgs * 2][kThreads]: gamma + shift
  __shared__ __align__(8) uint64_t full[2];
  __shared__ float slots[2][4];
  const int tidx = threadIdx.x, lane = tidx % 32, warp = tidx / 32;
  const int r = blockIdx.x;
  const int c = warp * 32 + lane;
  const int nrows = r < rows ? (rows - 1 - r) / ctas_per_col + 1 : 0;
  if (tidx == 0) {
    mbar_init(&full[0], 1);
    mbar_init(&full[1], 1);
    mbar_fence_init();
  }
  __syncthreads();
  auto issue = [&](int k, int s) {
    const int64_t row = (int64_t)r + (int64_t)k * ctas_per_col;
    __nv_bfloat16* d = st + (int64_t)s * kNT * kCols;
    mbar_expect_tx(&full[s], kNT * kRowBytes);
    bulk_g2s(d, x + row * kCols, kRowBytes, &full[s]);
    bulk_g2s(d + kCols, dz + row * kCols, kRowBytes, &full[s]);
    if (kAdd) bulk_g2s(d + 2 * kCols, add + row * kCols, kRowBytes, &full[s]);
  };
  if (tidx == 0) {
    if (nrows > 0) issue(0, 0);
    if (nrows > 1) issue(1, 1);
  }
  float dzy[kLdgs * kElts];
#pragma unroll
  for (int i = 0; i < kLdgs * kElts; i++) dzy[i] = 0.f;
#pragma unroll
  for (int it = 0; it < kLdgs; it++) {
    V8 gv;
    gv.u = *reinterpret_cast<const uint4*>(gamma + (c + it * 128) * kElts);
    float gsv[kElts];
#pragma unroll
    for (int jt = 0; jt < kElts; jt++) gsv[jt] = __fadd_rn(bf(gv.h[jt]), shift);
    gs4[(it * 2 + 0) * kThreads + tidx] = make_float4(gsv[0], gsv[1], gsv[2], gsv[3]);
    gs4[(it * 2 + 1) * kThreads + tidx] = make_float4(gsv[4], gsv[5], gsv[6], gsv[7]);
  }
  constexpr float rn = 1.f / static_cast<float>(kCols);
  uint32_t a2 = 0u;                                    // packed |max| of the stored bf16 words
  int use0 = 1;
#pragma unroll 1
  for (int k = 0; k < nrows; ++k) {
    const int s = k & 1;
    const int64_t row = (int64_t)r + (int64_t)k * ctas_per_col;
    const float rs_r = rs[row];
    mbar_wait(&full[s], (k >> 1) & 1);
    const __nv_bfloat16* sx = st + (int64_t)s * kNT * kCols;
    const __nv_bfloat16* sd = sx + kCols;
    const __nv_bfloat16* sa = sx + 2 * kCols;
    float mdyy_local = 0.f;
#pragma unroll
    for (int it = 0; it < kLdgs; it++) {
      V8 xv, dv;
      xv.u = *reinterpret_cast<const uint4*>(sx + (c + it * 128) * kElts);
      dv.u = *reinterpret_cast<const uint4*>(sd + (c + it * 128) * kElts);
      const float4 g0 = gs4[(it * 2 + 0) * kThreads + tidx], g1 = gs4[(it * 2 + 1) * kThreads + tidx];
      const float gsv[kElts] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
      for (int jt = 0; jt < kElts; jt++) {
        const float x_tmp = bf(xv.h[jt]);
        const float y_tmp = __fmul_rn(rs_r, x_tmp);
        const float dy_tmp = __fmul_rn(gsv[jt], bf(dv.h[jt]));
        const float dz_tmp = bf(dv.h[jt]);
        mdyy_local = __fmaf_rn(dy_tmp, y_tmp, mdyy_local);
        dzy[it * kElts + jt] = __fmaf_rn(dz_tmp, y_tmp, dzy[it * kElts + jt]);
      }
    }
    float* slot = use0 ? slots[0] : slots[1];
    use0 = !use0;
    const float mdyy = __fmul_rn(RowMath::row_allreduce(mdyy_local, slot, warp, lane), rn);
    const int64_t base = row * kVecCols + c;
#pragma unroll
    for (int it = 0; it < kLdgs; it++) {
      V8 xv, dv, av;
      xv.u = *reinterpret_cast<const uint4*>(sx + (c + it * 128) * kElts);
      dv.u = *reinterpret_cast<const uint4*>(sd + (c + it * 128) * kElts);
      if (kAdd) av.u = *reinterpret_cast<const uint4*>(sa + (c + it * 128) * kElts);
      const float4 g0 = gs4[(it * 2 + 0) * kThreads + tidx], g1 = gs4[(it * 2 + 1) * kThreads + tidx];
      const float gsv[kElts] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
      uint32_t w[4];
#pragma unroll
      for (int p = 0; p < 4; p++) {
        float dxv[2];
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int jt = 2 * p + e;
          const float y_tmp = __fmul_rn(rs_r, bf(xv.h[jt]));
          const float dy_tmp = __fmul_rn(gsv[jt], bf(dv.h[jt]));
          dxv[e] = __fmul_rn(rs_r, __fmaf_rn(-mdyy, y_tmp, dy_tmp));
        }
        const uint32_t dxp = ak_rht::cvt_bf16x2(dxv[0], dxv[1]);
        if (kAdd) {
          const float s0 = __fadd_rn(__uint_as_float(dxp << 16), bf(av.h[2 * p]));
          const float s1 = __fadd_rn(__uint_as_float(dxp & 0xffff0000u), bf(av.h[2 * p + 1]));
          w[p] = ak_rht::cvt_bf16x2(s0, s1);
        } else {
          w[p] = dxp;
        }
        a2 = ak_rht::bmax_xa(a2, w[p]);
      }
      *reinterpret_cast<uint4*>(out + (base + it * 128) * kElts) = make_uint4(w[0], w[1], w[2], w[3]);
    }
    __syncthreads();                                   // stage s fully read: refill it with row k + 2
    if (tidx == 0 && k + 2 < nrows) issue(k + 2, s);
  }
  store_dgamma_part(dgamma_part, r, c, dzy);
  block_max_publish(ak_rht::unpack_bmax(a2), amax_pre);
}

void check_bf16(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
              " must be a contiguous CUDA bf16 tensor");
}

}  // namespace

// rmsnorm_bwd_add(dz, x, rsigma, gamma, add, ctas_per_col, zero_centered_gamma, sign_mask, post_amax, version)
//   -> (out [.., 4096] bf16, dgamma [4096] bf16, amax [2] f32: pre-RHT, post-RHT (0 unless post_amax))
// add: the residual gradient to add (bf16, dz's shape) or an empty tensor (out = TE's dx)
std::vector<at::Tensor> rmsnorm_bwd_add(const at::Tensor& dz, const at::Tensor& x, const at::Tensor& rsigma,
                                        const at::Tensor& gamma, const at::Tensor& add, int64_t ctas_per_col,
                                        bool zero_centered_gamma, int64_t sign_mask, bool post_amax, int64_t version,
                                        const c10::optional<at::Tensor>& main_grad) {
  check_bf16(dz, "dz");
  check_bf16(x, "x");
  check_bf16(gamma, "gamma");
  TORCH_CHECK(rsigma.is_cuda() && rsigma.scalar_type() == at::kFloat && rsigma.is_contiguous(), "rsigma: f32 contiguous");
  TORCH_CHECK(x.size(-1) == kCols && dz.sizes() == x.sizes() && gamma.numel() == kCols, "hidden 4096 only");
  const int64_t rows = x.numel() / kCols;
  TORCH_CHECK(rsigma.numel() == rows, "rsigma: one per row");
  TORCH_CHECK(ctas_per_col >= 1 && ctas_per_col <= 65535 && rows <= INT32_MAX, "ctas_per_col in [1, 65535]");
  const bool has_add = add.numel() > 0;
  if (has_add) {
    check_bf16(add, "add");
    TORCH_CHECK(add.numel() == x.numel(), "add: dz's size");
  }
  TORCH_CHECK(!post_amax || rows % kTRb == 0, "post-RHT amax: rows % 32");
  const bool has_mg = main_grad.has_value() && main_grad->defined() && main_grad->numel() > 0;
  if (has_mg) {
    check_bf16(*main_grad, "main_grad");
    TORCH_CHECK(main_grad->numel() == kCols && main_grad->device() == x.device(), "main_grad: bf16 [4096] on x's device");
  }
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = at::empty_like(x);
  auto dgamma = at::empty_like(gamma);
  auto amax = at::zeros({2}, x.options().dtype(at::kFloat));
  auto part = at::empty({ctas_per_col, kCols}, x.options().dtype(at::kFloat));
  const float shift = zero_centered_gamma ? 1.0f : 0.f;
  const auto* pdz = reinterpret_cast<const __nv_bfloat16*>(dz.data_ptr());
  const auto* px = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* pg = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr());
  const auto* pa = has_add ? reinterpret_cast<const __nv_bfloat16*>(add.data_ptr()) : nullptr;
  auto* po = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const int nr = (int)rows, cpc = (int)ctas_per_col;
  if (version == 7) {
    static bool attr7[2] = {false, false};
    if (has_add) {
      if (!attr7[1]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(rmsnorm_bwd_add_v4<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)smem_v4<true>()));
        attr7[1] = true;
      }
      rmsnorm_bwd_add_v4<true><<<cpc, kThreads, smem_v4<true>(), stream>>>(
          pdz, px, rsigma.data_ptr<float>(), pg, pa, po, part.data_ptr<float>(), amax.data_ptr<float>(), nr, cpc, shift);
    } else {
      if (!attr7[0]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(rmsnorm_bwd_add_v4<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)smem_v4<false>()));
        attr7[0] = true;
      }
      rmsnorm_bwd_add_v4<false><<<cpc, kThreads, smem_v4<false>(), stream>>>(
          pdz, px, rsigma.data_ptr<float>(), pg, pa, po, part.data_ptr<float>(), amax.data_ptr<float>(), nr, cpc, shift);
    }
  } else if (version == 2 || version == 3 || version == 6) {
    static bool attr[2] = {false, false};
    if (has_add) {
      if (!attr[1]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(rmsnorm_bwd_add_v2<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)smem_v2<true>()));
        attr[1] = true;
      }
      rmsnorm_bwd_add_v2<true><<<cpc, kThreads, smem_v2<true>(), stream>>>(
          pdz, px, rsigma.data_ptr<float>(), pg, pa, po, part.data_ptr<float>(), amax.data_ptr<float>(), nr, cpc, shift);
    } else {
      if (!attr[0]) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(rmsnorm_bwd_add_v2<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)smem_v2<false>()));
        attr[0] = true;
      }
      rmsnorm_bwd_add_v2<false><<<cpc, kThreads, smem_v2<false>(), stream>>>(
          pdz, px, rsigma.data_ptr<float>(), pg, pa, po, part.data_ptr<float>(), amax.data_ptr<float>(), nr, cpc, shift);
    }
  } else {
    if (has_add)
      rmsnorm_bwd_add_v1<true><<<cpc, kThreads, 0, stream>>>(pdz, px, rsigma.data_ptr<float>(), pg, pa, po,
                                                            part.data_ptr<float>(), amax.data_ptr<float>(), nr, cpc,
                                                            shift);
    else
      rmsnorm_bwd_add_v1<false><<<cpc, kThreads, 0, stream>>>(pdz, px, rsigma.data_ptr<float>(), pg, pa, po,
                                                             part.data_ptr<float>(), amax.data_ptr<float>(), nr, cpc,
                                                             shift);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (has_mg)
    rmsnorm_bwd_finalize_mg<<<128, 1024, 0, stream>>>(part.data_ptr<float>(),
                                                      reinterpret_cast<__nv_bfloat16*>(dgamma.data_ptr()),
                                                      reinterpret_cast<__nv_bfloat16*>(main_grad->data_ptr()), cpc);
  else
    rmsnorm_bwd_finalize<<<128, 1024, 0, stream>>>(part.data_ptr<float>(),
                                                   reinterpret_cast<__nv_bfloat16*>(dgamma.data_ptr()), cpc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (post_amax) {
    dim3 grid(kCols / kTCb, (unsigned)(rows / kTRb));
    if (version == 6 || version == 7)
      rht_amax_post_kernel_v3<<<grid, kThreadsB, 0, stream>>>(po, amax.data_ptr<float>() + 1, (uint32_t)sign_mask);
    else if (version == 3)
      rht_amax_post_kernel_v2<<<grid, kThreadsB, 0, stream>>>(po, amax.data_ptr<float>() + 1, (uint32_t)sign_mask);
    else
      rht_amax_post_kernel<<<grid, kThreadsB, 0, stream>>>(po, amax.data_ptr<float>() + 1, (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out, dgamma, amax};
}

// rht_amax_post(t [rows, 4096] bf16, sign_mask, version) -> amax [1] f32 (the post-RHT amax alone; for tests)
at::Tensor rht_amax_post(const at::Tensor& t, int64_t sign_mask, int64_t version) {
  check_bf16(t, "t");
  TORCH_CHECK(t.size(-1) == kCols && (t.numel() / kCols) % kTRb == 0, "[rows % 32, 4096]");
  const at::cuda::CUDAGuard guard(t.device());
  auto amax = at::zeros({1}, t.options().dtype(at::kFloat));
  dim3 grid(kCols / kTCb, (unsigned)(t.numel() / kCols / kTRb));
  if (version == 2)
    rht_amax_post_kernel_v2<<<grid, kThreadsB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()), amax.data_ptr<float>(), (uint32_t)sign_mask);
  else
    rht_amax_post_kernel<<<grid, kThreadsB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()), amax.data_ptr<float>(), (uint32_t)sign_mask);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return amax;
}

// rht_amax(t [.., N] bf16, sign_mask) -> amax [2] f32: pre-RHT, post-RHT (rows % 32, N % 256)
// (dev bw19, this build: version 2 -- the default -- is rht_amax_kernel_r, the reversed rows with plain loads; version 1
// the forward kernel with __ldcs as before; the same amaxes)
at::Tensor rht_amax(const at::Tensor& t, int64_t sign_mask, int64_t version) {
  check_bf16(t, "t");
  const int64_t N = t.size(-1), rows = t.numel() / N;
  TORCH_CHECK(t.dim() >= 2 && N % kTCb == 0 && rows % kTRb == 0 && rows / kTRb <= 65535, "[rows % 32, N % 256]");
  const at::cuda::CUDAGuard guard(t.device());
  auto amax = at::zeros({2}, t.options().dtype(at::kFloat));
  dim3 grid((unsigned)(N / kTCb), (unsigned)(rows / kTRb));
  if (version == 1)
    rht_amax_kernel<<<grid, kThreadsB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()), amax.data_ptr<float>(), N, (uint32_t)sign_mask);
  else
    rht_amax_kernel_r<<<grid, kThreadsB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()), amax.data_ptr<float>(), N, (uint32_t)sign_mask);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return amax;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rht_amax", &rht_amax, "pre- and post-RHT amax (16-token blocks) of a [rows, N] bf16 tensor", py::arg("t"),
        py::arg("sign_mask"), py::arg("version") = 2);
  m.def("rmsnorm_bwd_add", &rmsnorm_bwd_add, "TE's RMSNorm backward (hidden 4096, bf16) + residual add + amaxes",
        py::arg("dz"), py::arg("x"), py::arg("rsigma"), py::arg("gamma"), py::arg("add"), py::arg("ctas_per_col"),
        py::arg("zero_centered_gamma"), py::arg("sign_mask"), py::arg("post_amax"), py::arg("version") = 2,
        py::arg("main_grad") = py::none());
  m.def("rht_amax_post", &rht_amax_post, "post-RHT amax (16-token blocks) of a [rows, 4096] bf16 tensor", py::arg("t"),
        py::arg("sign_mask"), py::arg("version") = 1);
}
