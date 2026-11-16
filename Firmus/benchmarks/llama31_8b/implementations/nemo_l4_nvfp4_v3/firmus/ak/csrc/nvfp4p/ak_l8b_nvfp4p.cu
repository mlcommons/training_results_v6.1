// ak_l8b_nvfp4p.cu (Llama 3.1 8B kernel campaign, 2026-10-02; 2026-10-07 dev bw19: dSwiGLU v8 = v7 with the block's
// loads up front, versions 9 / 10 = the reversed block orders, see dswiglu_amax_kernel_v8): SwiGLU forward / backward that ALSO emit the two amaxes
// TransformerEngine's NVFP4 quantizer needs, so the quantize that follows can skip its own amax pass.
//
// Stock TE 2.20 (PROF-L0): `cast_fp8_gated_kernel` writes the BF16 activation (or its gradient); TE's NVFP4 quantize then
// reads it twice -- `HadamardAmaxTmaKernel` (the pre-RHT amax of the tensor, and the amax of the columnwise random
// Hadamard transform of it) and `row_col_rht_gemm_device` (the cast).  Here the producer computes both amaxes from the
// values it has just written, and tex.nvfp4_quantize_with_amax does the cast with TE's own kernel.
//
// E0 by construction, each piece checked against TE by the unit test and by the overlay's first-call check:
//  * the activation arithmetic is TE's own text (transformer_engine/common/util/math.h and cast/fp8/gated_fp8.cuh at
//    131157b5), compiled with --use_fast_math as TE's activation kernels are in this image
//    (NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1);
//  * the pre-RHT amax is max |bf16 output| (exact; NaN ignored as TE's max.xorsign.abs does);
//  * the post-RHT amax is max |bf16(T^T-tile x RHT)| with TE's transform -- T = diag(sign) H16 * 0.25 applied to 16
//    consecutive TOKENS of one feature (the columnwise copy), computed with the same bf16 m16n16k16 tensor-core MMA and
//    zero accumulator TE uses, then rounded to bf16 (cvt.rn) before the abs-max.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <mma.h>

#include "ak_l8b_rht_epi.cuh"

using namespace nvcuda;

namespace {

// (torch's extension build defines __CUDA_NO_BFLOAT16_CONVERSIONS__: the bf16 <-> float conversions TE writes as casts are
// spelled as the intrinsics they compile to -- __bfloat162float (exact) and __float2bfloat16 (round to nearest even).)
// ---- TE's math, verbatim (util/math.h) ----------------------------------------------------------------------------
__device__ inline float te_sigmoid(const float val) {
  const float cval = val;
  return 1.f / (1.f + expf(-cval));
}
__device__ inline float te_sigmoidf(const float x) { return __frcp_rn(1.0f + __expf(-x)); }
__device__ inline float te_silu(const float val) {
  const float cval = val;
  return cval * te_sigmoid(cval);
}

__device__ __forceinline__ void atomic_max_nonneg(float* addr, float v) {
  // v >= 0 (or +inf): the int order of non-negative floats is their numeric order (TE's atomicMaxFloat contract)
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

__device__ __forceinline__ float bf16_abs_f(__nv_bfloat16 v) { return fabsf(__bfloat162float(v)); }

constexpr int kTR = 32;     // tokens (rows) per CTA tile: two 16-token RHT blocks
constexpr int kTC = 256;    // features (columns) per CTA tile: sixteen 16-feature RHT blocks
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// The 16x16 RHT matrix T[k][n] = sign[k] * H16[k][n] * 0.25 (bf16, exact), sign[k] = -1 where bit k of the mask is set
// (TE's reference: diag(get_wgrad_sign_vector()) @ H16 / sqrt(16); mask 0xD7E8 <-> that vector).
__device__ __forceinline__ void build_rht(__nv_bfloat16* sh_h, uint32_t sign_mask) {
  for (int t = threadIdx.x; t < 256; t += blockDim.x) {
    const int k = t / 16, n = t % 16;
    float v = (__popc(k & n) & 1) ? -0.25f : 0.25f;
    if ((sign_mask >> k) & 1u) v = -v;
    sh_h[t] = __float2bfloat16_rn(v);
  }
}

// Post-RHT amax of one 16-token x 16-feature sub-tile held row-major [token][feature] in shared memory (leading dim ld):
// A = tile^T (features x tokens, loaded col-major), B = T, D = A B + 0 (fp32), each element rounded to bf16, abs-max.
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
  for (int i = 0; i < cfrag.num_elements; ++i) {
    m = fmaxf(m, fabsf(__bfloat162float(__float2bfloat16_rn(cfrag.x[i]))));
  }
  return m;
}

__device__ __forceinline__ void block_reduce_and_publish(float a_pre, float a_post, float* amax_pre,
                                                         float* amax_post) {
  __shared__ float red[2][kWarps];
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
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      p = fmaxf(p, red[0][w]);
      q = fmaxf(q, red[1][w]);
    }
    atomic_max_nonneg(amax_pre, p);
    atomic_max_nonneg(amax_post, q);
  }
}

union V8 {
  uint4 u;
  __nv_bfloat16 h[8];
};

// ---- backward: d(silu(act) * gate) -> [dact | dgate] -------------------------------------------------------------
// LD = the staging tiles' row stride: kTC (v1) or kTC + 8 (v4, 2026-10-02 23:35: rows 528 B apart, so the 16 rows of a
// wmma tile load fall in different shared-memory banks; at 512 B every row of a load hit the same banks, 8-way)
template <int LD>
__global__ void __launch_bounds__(kThreads)
    dswiglu_amax_kernel_t(const __nv_bfloat16* __restrict__ grad,  // [M, N]
                          const __nv_bfloat16* __restrict__ x,     // [M, 2N]: act | gate
                          __nv_bfloat16* __restrict__ out,         // [M, 2N]: dact | dgate
                          float* __restrict__ amax_pre, float* __restrict__ amax_post, int64_t M, int64_t N,
                          uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh_act[kTR * LD];
  __shared__ __align__(128) __nv_bfloat16 sh_gate[kTR * LD];
  __shared__ __align__(128) __nv_bfloat16 sh_h[256];
  const int64_t row0 = (int64_t)blockIdx.y * kTR, col0 = (int64_t)blockIdx.x * kTC;
  build_rht(sh_h, sign_mask);
  float a_pre = 0.0f;
  constexpr int kVecPerRow = kTC / 8;
  for (int v = threadIdx.x; v < kTR * kVecPerRow; v += kThreads) {
    const int r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    V8 g, a, b, oa, og;
    g.u = *reinterpret_cast<const uint4*>(grad + gi * N + gj);
    a.u = *reinterpret_cast<const uint4*>(x + gi * 2 * N + gj);
    b.u = *reinterpret_cast<const uint4*>(x + gi * 2 * N + N + gj);
    const float scale = 1;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      // cast/fp8/gated_fp8.cuh, IS_BWD branch, ParamOP Empty, ActOP silu / DActOP dsilu (the sigmoidf special case)
      const float act_elt = __bfloat162float(a.h[e]);
      const float gate_elt = __bfloat162float(b.h[e]);
      const float dgate_elt = 1.0f;
      const float grad_elt = __bfloat162float(g.h[e]);
      const float xv = act_elt;
      float act_x;
      float dact_x;
      {
        const float s = te_sigmoidf(xv);
        act_x = xv * s;
        dact_x = xv * s * (1 - s) + s;
      }
      const float after_dact = dact_x * grad_elt * gate_elt;
      const float after_dgate = dgate_elt * act_x * grad_elt;
      oa.h[e] = __float2bfloat16(scale * after_dact);
      og.h[e] = __float2bfloat16(scale * after_dgate);
      a_pre = fmaxf(a_pre, bf16_abs_f(oa.h[e]));
      a_pre = fmaxf(a_pre, bf16_abs_f(og.h[e]));
    }
    *reinterpret_cast<uint4*>(out + gi * 2 * N + gj) = oa.u;
    *reinterpret_cast<uint4*>(out + gi * 2 * N + N + gj) = og.u;
    *reinterpret_cast<uint4*>(sh_act + r * LD + c8) = oa.u;
    *reinterpret_cast<uint4*>(sh_gate + r * LD + c8) = og.u;
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a_post = 0.0f;
  constexpr int kSub = (kTR / 16) * (kTC / 16);  // sub-tiles per tensor
  for (int t = threadIdx.x / 32; t < 2 * kSub; t += kWarps) {
    const int which = t / kSub, rem = t % kSub, tb = rem / (kTC / 16), fb = rem % (kTC / 16);
    const __nv_bfloat16* base = (which ? sh_gate : sh_act) + tb * 16 * LD + fb * 16;
    a_post = fmaxf(a_post, rht_tile_amax(base, LD, bfrag));
  }
  block_reduce_and_publish(a_pre, a_post, amax_pre, amax_post);
}

// ---- forward: silu(act) * gate ---------------------------------------------------------------------------------
template <int LD>
__global__ void __launch_bounds__(kThreads)
    swiglu_amax_kernel_t(const __nv_bfloat16* __restrict__ x,  // [M, 2N]: act | gate
                         __nv_bfloat16* __restrict__ out,      // [M, N]
                         float* __restrict__ amax_pre, float* __restrict__ amax_post, int64_t M, int64_t N,
                         uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh_out[kTR * LD];
  __shared__ __align__(128) __nv_bfloat16 sh_h[256];
  const int64_t row0 = (int64_t)blockIdx.y * kTR, col0 = (int64_t)blockIdx.x * kTC;
  build_rht(sh_h, sign_mask);
  float a_pre = 0.0f;
  constexpr int kVecPerRow = kTC / 8;
  for (int v = threadIdx.x; v < kTR * kVecPerRow; v += kThreads) {
    const int r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    V8 a, b, o;
    a.u = *reinterpret_cast<const uint4*>(x + gi * 2 * N + gj);
    b.u = *reinterpret_cast<const uint4*>(x + gi * 2 * N + N + gj);
    const float scale = 1;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      // cast/fp8/gated_fp8.cuh, forward branch: after_act = ActOP(act_elt, p) * gate_elt, ActOP = silu<fp32, fp32>
      const float act_elt = __bfloat162float(a.h[e]);
      const float gate_elt = __bfloat162float(b.h[e]);
      const float after_act = te_silu(act_elt) * gate_elt;
      o.h[e] = __float2bfloat16(scale * after_act);
      a_pre = fmaxf(a_pre, bf16_abs_f(o.h[e]));
    }
    *reinterpret_cast<uint4*>(out + gi * N + gj) = o.u;
    *reinterpret_cast<uint4*>(sh_out + r * LD + c8) = o.u;
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a_post = 0.0f;
  constexpr int kSub = (kTR / 16) * (kTC / 16);
  for (int t = threadIdx.x / 32; t < kSub; t += kWarps) {
    const int tb = t / (kTC / 16), fb = t % (kTC / 16);
    a_post = fmaxf(a_post, rht_tile_amax(sh_out + tb * 16 * LD + fb * 16, LD, bfrag));
  }
  block_reduce_and_publish(a_pre, a_post, amax_pre, amax_post);
}

// ---- v2 (2026-10-02 22:15): the same arithmetic and the same amax epilogue; every load of a thread is issued before
// any of them is used (four 16-byte vectors per input per thread in flight instead of one) ----------------------------
constexpr int kIters = kTR * (kTC / 8) / kThreads;   // 4

__global__ void __launch_bounds__(kThreads)
    dswiglu_amax_kernel_v2(const __nv_bfloat16* __restrict__ grad, const __nv_bfloat16* __restrict__ x,
                           __nv_bfloat16* __restrict__ out, float* __restrict__ amax_pre,
                           float* __restrict__ amax_post, int64_t M, int64_t N, uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh_act[kTR * kTC];
  __shared__ __align__(128) __nv_bfloat16 sh_gate[kTR * kTC];
  __shared__ __align__(128) __nv_bfloat16 sh_h[256];
  const int64_t row0 = (int64_t)blockIdx.y * kTR, col0 = (int64_t)blockIdx.x * kTC;
  constexpr int kVecPerRow = kTC / 8;
  V8 g[kIters], a[kIters], b[kIters];
#pragma unroll
  for (int i = 0; i < kIters; ++i) {
    const int v = threadIdx.x + i * kThreads;
    const int r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    g[i].u = __ldcs(reinterpret_cast<const uint4*>(grad + gi * N + gj));
    a[i].u = __ldcs(reinterpret_cast<const uint4*>(x + gi * 2 * N + gj));
    b[i].u = __ldcs(reinterpret_cast<const uint4*>(x + gi * 2 * N + N + gj));
  }
  build_rht(sh_h, sign_mask);
  float a_pre = 0.0f;
#pragma unroll
  for (int i = 0; i < kIters; ++i) {
    const int v = threadIdx.x + i * kThreads;
    const int r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    V8 oa, og;
    const float scale = 1;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float act_elt = __bfloat162float(a[i].h[e]);
      const float gate_elt = __bfloat162float(b[i].h[e]);
      const float dgate_elt = 1.0f;
      const float grad_elt = __bfloat162float(g[i].h[e]);
      const float xv = act_elt;
      float act_x;
      float dact_x;
      {
        const float s = te_sigmoidf(xv);
        act_x = xv * s;
        dact_x = xv * s * (1 - s) + s;
      }
      const float after_dact = dact_x * grad_elt * gate_elt;
      const float after_dgate = dgate_elt * act_x * grad_elt;
      oa.h[e] = __float2bfloat16(scale * after_dact);
      og.h[e] = __float2bfloat16(scale * after_dgate);
      a_pre = fmaxf(a_pre, bf16_abs_f(oa.h[e]));
      a_pre = fmaxf(a_pre, bf16_abs_f(og.h[e]));
    }
    *reinterpret_cast<uint4*>(out + gi * 2 * N + gj) = oa.u;
    *reinterpret_cast<uint4*>(out + gi * 2 * N + N + gj) = og.u;
    *reinterpret_cast<uint4*>(sh_act + r * kTC + c8) = oa.u;
    *reinterpret_cast<uint4*>(sh_gate + r * kTC + c8) = og.u;
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a_post = 0.0f;
  constexpr int kSub = (kTR / 16) * (kTC / 16);
  for (int t = threadIdx.x / 32; t < 2 * kSub; t += kWarps) {
    const int which = t / kSub, rem = t % kSub, tb = rem / (kTC / 16), fb = rem % (kTC / 16);
    const __nv_bfloat16* base = (which ? sh_gate : sh_act) + tb * 16 * kTC + fb * 16;
    a_post = fmaxf(a_post, rht_tile_amax(base, kTC, bfrag));
  }
  block_reduce_and_publish(a_pre, a_post, amax_pre, amax_post);
}

__global__ void __launch_bounds__(kThreads)
    swiglu_amax_kernel_v2(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
                          float* __restrict__ amax_pre, float* __restrict__ amax_post, int64_t M, int64_t N,
                          uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh_out[kTR * kTC];
  __shared__ __align__(128) __nv_bfloat16 sh_h[256];
  const int64_t row0 = (int64_t)blockIdx.y * kTR, col0 = (int64_t)blockIdx.x * kTC;
  constexpr int kVecPerRow = kTC / 8;
  V8 a[kIters], b[kIters];
#pragma unroll
  for (int i = 0; i < kIters; ++i) {
    const int v = threadIdx.x + i * kThreads;
    const int r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    a[i].u = __ldcs(reinterpret_cast<const uint4*>(x + gi * 2 * N + gj));
    b[i].u = __ldcs(reinterpret_cast<const uint4*>(x + gi * 2 * N + N + gj));
  }
  build_rht(sh_h, sign_mask);
  float a_pre = 0.0f;
#pragma unroll
  for (int i = 0; i < kIters; ++i) {
    const int v = threadIdx.x + i * kThreads;
    const int r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    V8 o;
    const float scale = 1;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float act_elt = __bfloat162float(a[i].h[e]);
      const float gate_elt = __bfloat162float(b[i].h[e]);
      const float after_act = te_silu(act_elt) * gate_elt;
      o.h[e] = __float2bfloat16(scale * after_act);
      a_pre = fmaxf(a_pre, bf16_abs_f(o.h[e]));
    }
    *reinterpret_cast<uint4*>(out + gi * N + gj) = o.u;
    *reinterpret_cast<uint4*>(sh_out + r * kTC + c8) = o.u;
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a_post = 0.0f;
  constexpr int kSub = (kTR / 16) * (kTC / 16);
  for (int t = threadIdx.x / 32; t < kSub; t += kWarps) {
    const int tb = t / (kTC / 16), fb = t % (kTC / 16);
    a_post = fmaxf(a_post, rht_tile_amax(sh_out + tb * 16 * kTC + fb * 16, kTC, bfrag));
  }
  block_reduce_and_publish(a_pre, a_post, amax_pre, amax_post);
}

// ---- v3 (2026-10-02 22:35): persistent CTAs with a cp.async double buffer (the next tile's inputs land in shared
// memory while the current tile is computed and its RHT amax taken).  Tile 32 rows x 128 columns; the same per-element
// arithmetic and the same amax epilogue as v1. ------------------------------------------------------------------------
constexpr int kTR3 = 32, kTC3 = 128;
constexpr int kTile3 = kTR3 * kTC3;                    // elements per input tile
constexpr int kChunks3 = kTile3 * 2 / 16;              // 16-byte chunks per input tile (512)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <bool BWD>
__device__ __forceinline__ void load_tile3(__nv_bfloat16* st, const __nv_bfloat16* __restrict__ grad,
                                          const __nv_bfloat16* __restrict__ x, int64_t row0, int64_t col0, int64_t N) {
  // st: [grad | act | gate] tiles of kTile3 elements each (grad unused in the forward)
#pragma unroll
  for (int i = 0; i < kChunks3 / kThreads; ++i) {
    const int c = threadIdx.x + i * kThreads;
    const int r = c / (kTC3 / 8), c8 = (c % (kTC3 / 8)) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    if (BWD) cp_async16(st + r * kTC3 + c8, grad + gi * N + gj);
    cp_async16(st + kTile3 + r * kTC3 + c8, x + gi * 2 * N + gj);
    cp_async16(st + 2 * kTile3 + r * kTC3 + c8, x + gi * 2 * N + N + gj);
  }
}

template <bool BWD>
__global__ void __launch_bounds__(kThreads)
    swiglu_family_amax_kernel_v3(const __nv_bfloat16* __restrict__ grad, const __nv_bfloat16* __restrict__ x,
                                 __nv_bfloat16* __restrict__ out, float* __restrict__ amax_pre,
                                 float* __restrict__ amax_post, int64_t M, int64_t N, uint32_t sign_mask) {
  extern __shared__ __align__(128) unsigned char smem3[];
  __nv_bfloat16* stage[2] = {reinterpret_cast<__nv_bfloat16*>(smem3),
                             reinterpret_cast<__nv_bfloat16*>(smem3) + 3 * kTile3};
  __nv_bfloat16* sh_o0 = reinterpret_cast<__nv_bfloat16*>(smem3) + 6 * kTile3;   // dact (or y)
  __nv_bfloat16* sh_o1 = sh_o0 + kTile3;                                           // dgate
  __nv_bfloat16* sh_h = sh_o1 + kTile3;
  build_rht(sh_h, sign_mask);
  const int64_t tiles_x = N / kTC3, tiles = tiles_x * (M / kTR3);
  int64_t tile = blockIdx.x;
  if (tile < tiles) load_tile3<BWD>(stage[0], grad, x, (tile / tiles_x) * kTR3, (tile % tiles_x) * kTC3, N);
  cp_async_commit();
  __syncthreads();
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a_pre = 0.0f, a_post = 0.0f;
  for (int it = 0; tile < tiles; ++it, tile += gridDim.x) {
    const int64_t next = tile + gridDim.x;
    if (next < tiles) load_tile3<BWD>(stage[(it + 1) & 1], grad, x, (next / tiles_x) * kTR3, (next % tiles_x) * kTC3, N);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const __nv_bfloat16* st = stage[it & 1];
    const int64_t row0 = (tile / tiles_x) * kTR3, col0 = (tile % tiles_x) * kTC3;
#pragma unroll
    for (int i = 0; i < kChunks3 / kThreads; ++i) {
      const int c = threadIdx.x + i * kThreads;
      const int r = c / (kTC3 / 8), c8 = (c % (kTC3 / 8)) * 8;
      const int64_t gi = row0 + r, gj = col0 + c8;
      V8 a, b, o0, o1;
      a.u = *reinterpret_cast<const uint4*>(st + kTile3 + r * kTC3 + c8);
      b.u = *reinterpret_cast<const uint4*>(st + 2 * kTile3 + r * kTC3 + c8);
      const float scale = 1;
      if (BWD) {
        V8 g;
        g.u = *reinterpret_cast<const uint4*>(st + r * kTC3 + c8);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float act_elt = __bfloat162float(a.h[e]);
          const float gate_elt = __bfloat162float(b.h[e]);
          const float dgate_elt = 1.0f;
          const float grad_elt = __bfloat162float(g.h[e]);
          const float xv = act_elt;
          float act_x;
          float dact_x;
          {
            const float s = te_sigmoidf(xv);
            act_x = xv * s;
            dact_x = xv * s * (1 - s) + s;
          }
          const float after_dact = dact_x * grad_elt * gate_elt;
          const float after_dgate = dgate_elt * act_x * grad_elt;
          o0.h[e] = __float2bfloat16(scale * after_dact);
          o1.h[e] = __float2bfloat16(scale * after_dgate);
          a_pre = fmaxf(a_pre, bf16_abs_f(o0.h[e]));
          a_pre = fmaxf(a_pre, bf16_abs_f(o1.h[e]));
        }
        *reinterpret_cast<uint4*>(out + gi * 2 * N + gj) = o0.u;
        *reinterpret_cast<uint4*>(out + gi * 2 * N + N + gj) = o1.u;
        *reinterpret_cast<uint4*>(sh_o1 + r * kTC3 + c8) = o1.u;
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float act_elt = __bfloat162float(a.h[e]);
          const float gate_elt = __bfloat162float(b.h[e]);
          const float after_act = te_silu(act_elt) * gate_elt;
          o0.h[e] = __float2bfloat16(scale * after_act);
          a_pre = fmaxf(a_pre, bf16_abs_f(o0.h[e]));
        }
        *reinterpret_cast<uint4*>(out + gi * N + gj) = o0.u;
      }
      *reinterpret_cast<uint4*>(sh_o0 + r * kTC3 + c8) = o0.u;
    }
    __syncthreads();
    constexpr int kSub3 = (kTR3 / 16) * (kTC3 / 16);    // 16 sub-tiles per output tensor
    for (int t = threadIdx.x / 32; t < (BWD ? 2 : 1) * kSub3; t += kWarps) {
      const int which = t / kSub3, rem = t % kSub3, tb = rem / (kTC3 / 16), fb = rem % (kTC3 / 16);
      a_post = fmaxf(a_post, rht_tile_amax((which ? sh_o1 : sh_o0) + tb * 16 * kTC3 + fb * 16, kTC3, bfrag));
    }
    __syncthreads();
  }
  cp_async_wait<0>();
  block_reduce_and_publish(a_pre, a_post, amax_pre, amax_post);
}

constexpr size_t kSmem3 = sizeof(__nv_bfloat16) * (8 * kTile3 + 256);

// ---- v5 (2026-10-03): persistent CTAs (one per SM) fed by 2D bulk tensor copies (TMA) of the 32 x 256 input tiles into a
// ring of STAGES shared-memory stages, one mbarrier per stage; the next tiles' copies are in flight while a tile is
// computed and while its post-RHT amax is taken.  Per element the arithmetic is v4's (TE's), and the amax epilogue is
// v4's on the same padded staging (rows 528 B apart).  Tiles are visited column-tile fastest, as v4's grid.
constexpr int kTR5 = 32, kTC5 = 256, kLD5 = kTC5 + 8;
constexpr int kBox5 = kTR5 * kTC5 * 2;               // bytes per 32 x 256 bf16 box (16 KB)

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
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "AK_WAIT5_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra AK_WAIT5_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(smem_u32(bar))
      : "memory");
}

template <bool BWD, int STAGES>
constexpr size_t smem5_bytes() {
  return 1024 /* alignment slack */ + (size_t)STAGES * (BWD ? 3 : 2) * kBox5 +
         sizeof(__nv_bfloat16) * ((BWD ? 2 : 1) * kTR5 * kLD5 + 256);
}

template <bool BWD, int STAGES>
__global__ void __launch_bounds__(kThreads, 1)
    swiglu_family_amax_kernel_v5(const __grid_constant__ CUtensorMap tm_x, const __grid_constant__ CUtensorMap tm_g,
                                 __nv_bfloat16* __restrict__ out, float* __restrict__ amax_pre,
                                 float* __restrict__ amax_post, int64_t M, int64_t N, uint32_t sign_mask) {
  extern __shared__ unsigned char smem5_raw[];
  constexpr int kIn = BWD ? 3 : 2;                    // boxes per stage: [grad |] act | gate
  constexpr int kStageBytes = kIn * kBox5;
  unsigned char* smem5 = smem5_raw + ((1024 - (smem_u32(smem5_raw) & 1023)) & 1023);   // 1024-B aligned
  __nv_bfloat16* sh_o0 = reinterpret_cast<__nv_bfloat16*>(smem5 + STAGES * kStageBytes);   // dact (or y)
  __nv_bfloat16* sh_o1 = sh_o0 + kTR5 * kLD5;                                               // dgate (BWD)
  __nv_bfloat16* sh_h = sh_o0 + (BWD ? 2 : 1) * kTR5 * kLD5;
  __shared__ __align__(8) uint64_t full[STAGES];
  const int64_t tiles_x = N / kTC5, tiles = tiles_x * (M / kTR5);
  const int64_t mine = tiles > (int64_t)blockIdx.x ? (tiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) mbar_init(&full[s], 1);
    mbar_fence_init();
  }
  build_rht(sh_h, sign_mask);
  __syncthreads();
  auto issue = [&](int64_t j, int s) {
    const int64_t t = (int64_t)blockIdx.x + j * gridDim.x;
    const int row0 = (int)((t / tiles_x) * kTR5), col0 = (int)((t % tiles_x) * kTC5);
    unsigned char* st = smem5 + s * kStageBytes;
    mbar_expect_tx(&full[s], kStageBytes);
    if (BWD) tma_load_2d(st, &tm_g, col0, row0, &full[s]);
    tma_load_2d(st + (kIn - 2) * kBox5, &tm_x, col0, row0, &full[s]);
    tma_load_2d(st + (kIn - 1) * kBox5, &tm_x, (int)N + col0, row0, &full[s]);
  };
  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES && s < mine; ++s) issue(s, s);
  }
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bfrag;
  wmma::load_matrix_sync(bfrag, sh_h, 16);
  float a_pre = 0.0f, a_post = 0.0f;
  constexpr int kVecPerRow = kTC5 / 8;
  constexpr int kPer = kTR5 * kVecPerRow / kThreads;   // 4 vectors per thread per tile
  for (int64_t j = 0; j < mine; ++j) {
    const int s = (int)(j % STAGES);
    mbar_wait(&full[s], (uint32_t)((j / STAGES) & 1));
    const int64_t t = (int64_t)blockIdx.x + j * gridDim.x;
    const int64_t row0 = (t / tiles_x) * kTR5, col0 = (t % tiles_x) * kTC5;
    const unsigned char* st = smem5 + s * kStageBytes;
    const __nv_bfloat16* sg = reinterpret_cast<const __nv_bfloat16*>(st);
    const __nv_bfloat16* sa = reinterpret_cast<const __nv_bfloat16*>(st + (kIn - 2) * kBox5);
    const __nv_bfloat16* sb = reinterpret_cast<const __nv_bfloat16*>(st + (kIn - 1) * kBox5);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int v = threadIdx.x + i * kThreads, r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
      const int64_t gi = row0 + r, gj = col0 + c8;
      V8 a, b, o0, o1;
      a.u = *reinterpret_cast<const uint4*>(sa + r * kTC5 + c8);
      b.u = *reinterpret_cast<const uint4*>(sb + r * kTC5 + c8);
      const float scale = 1;
      if (BWD) {
        V8 g;
        g.u = *reinterpret_cast<const uint4*>(sg + r * kTC5 + c8);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float act_elt = __bfloat162float(a.h[e]);
          const float gate_elt = __bfloat162float(b.h[e]);
          const float dgate_elt = 1.0f;
          const float grad_elt = __bfloat162float(g.h[e]);
          const float xv = act_elt;
          float act_x;
          float dact_x;
          {
            const float s_ = te_sigmoidf(xv);
            act_x = xv * s_;
            dact_x = xv * s_ * (1 - s_) + s_;
          }
          const float after_dact = dact_x * grad_elt * gate_elt;
          const float after_dgate = dgate_elt * act_x * grad_elt;
          o0.h[e] = __float2bfloat16(scale * after_dact);
          o1.h[e] = __float2bfloat16(scale * after_dgate);
          a_pre = fmaxf(a_pre, bf16_abs_f(o0.h[e]));
          a_pre = fmaxf(a_pre, bf16_abs_f(o1.h[e]));
        }
        *reinterpret_cast<uint4*>(out + gi * 2 * N + gj) = o0.u;
        *reinterpret_cast<uint4*>(out + gi * 2 * N + N + gj) = o1.u;
        *reinterpret_cast<uint4*>(sh_o1 + r * kLD5 + c8) = o1.u;
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float act_elt = __bfloat162float(a.h[e]);
          const float gate_elt = __bfloat162float(b.h[e]);
          const float after_act = te_silu(act_elt) * gate_elt;
          o0.h[e] = __float2bfloat16(scale * after_act);
          a_pre = fmaxf(a_pre, bf16_abs_f(o0.h[e]));
        }
        *reinterpret_cast<uint4*>(out + gi * N + gj) = o0.u;
      }
      *reinterpret_cast<uint4*>(sh_o0 + r * kLD5 + c8) = o0.u;
    }
    __syncthreads();                                   // stage s read by every thread; the staging is complete
    if (threadIdx.x == 0 && j + STAGES < mine) issue(j + STAGES, s);
    constexpr int kSub5 = (kTR5 / 16) * (kTC5 / 16);    // 32 sub-tiles per output tensor
    for (int tt = threadIdx.x / 32; tt < (BWD ? 2 : 1) * kSub5; tt += kWarps) {
      const int which = tt / kSub5, rem = tt % kSub5, tb = rem / (kTC5 / 16), fb = rem % (kTC5 / 16);
      a_post = fmaxf(a_post, rht_tile_amax((which ? sh_o1 : sh_o0) + tb * 16 * kLD5 + fb * 16, kLD5, bfrag));
    }
    __syncthreads();                                   // the staging may be overwritten by the next tile
  }
  block_reduce_and_publish(a_pre, a_post, amax_pre, amax_post);
}

constexpr int kStagesBwd5 = 3, kStagesFwd5 = 4;

// ---- v6 (2026-10-05): v4's tiles (32 x 256, padded staging rows) and per-element arithmetic, with a cheaper epilogue
// built from TE's own amax instructions (HadamardAmaxTmaKernel, hadamard_transform.cu / hadamard_transform_utils.cuh):
//  * the outputs of two adjacent features are rounded together (cvt.rn.bf16x2.f32: the same round-to-nearest-even per
//    element as __float2bfloat16);
//  * the pre-RHT amax is kept in packed bf16 with max.xorsign.abs.bf16x2, TE's instruction for it (no flush to zero);
//  * the post-RHT amax of each 16-token x 16-feature sub-tile: ldmatrix.trans loads tile^T as the A fragments, the
//    constant B = T fragments are built in registers, mma.sync m16n8k16 (bf16 in, fp32 accumulate from zero: the HMMA
//    that TE's wmma m16n16k16 issues, the same k = token mapping as v4), then TE's cvt.rn.bf16x2.f32 and
//    max.xorsign.abs.bf16x2;
//  * the reductions after that in fp32 with max.f32 and a sign-bit mask (no flush to zero, as TE's ReduceMax, which is
//    not built with fast math; fmaxf / fabsf here would flush a subnormal amax).
// The ablation of 2026-10-03 (job 20029) put v4's epilogue at ~85 us of 426 us per dSwiGLU call.
// The pieces are in ak_l8b_rht_epi.cuh (shared with ak_l8b_nvfp4n's apply pass and ak_l8b_nvfp4b's post-RHT amax).
using namespace ak_rht;

constexpr int kLD6 = kTC + 8;

template <bool BWD, bool REV = false>
__global__ void __launch_bounds__(kThreads)
    swiglu_family_amax_kernel_v6(const __nv_bfloat16* __restrict__ grad,  // [M, N] (backward only)
                                 const __nv_bfloat16* __restrict__ x,     // [M, 2N]: act | gate
                                 __nv_bfloat16* __restrict__ out,         // [M, 2N] dact | dgate, or [M, N]
                                 float* __restrict__ amax_pre, float* __restrict__ amax_post, int64_t M, int64_t N,
                                 uint32_t sign_mask) {
  __shared__ __align__(128) __nv_bfloat16 sh_o0[kTR * kLD6];
  __shared__ __align__(128) __nv_bfloat16 sh_o1[BWD ? kTR * kLD6 : 8];
  const int64_t row0 = (int64_t)(REV ? gridDim.y - 1 - blockIdx.y : blockIdx.y) * kTR, col0 = (int64_t)blockIdx.x * kTC;
  constexpr int kVecPerRow = kTC / 8;
  uint32_t pre0 = 0u, pre1 = 0u;                       // packed bf16 abs-max accumulators (+0, as TE starts)
  for (int v = threadIdx.x; v < kTR * kVecPerRow; v += kThreads) {
    const int r = v / kVecPerRow, c8 = (v % kVecPerRow) * 8;
    const int64_t gi = row0 + r, gj = col0 + c8;
    V8 a, b;
    a.u = *reinterpret_cast<const uint4*>(x + gi * 2 * N + gj);
    b.u = *reinterpret_cast<const uint4*>(x + gi * 2 * N + N + gj);
    const float scale = 1;
    if (BWD) {
      V8 g;
      g.u = *reinterpret_cast<const uint4*>(grad + gi * N + gj);
      uint32_t oa[4], og[4];
#pragma unroll
      for (int e2 = 0; e2 < 4; ++e2) {
        float fa[2], fg[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int e = 2 * e2 + h;
          // cast/fp8/gated_fp8.cuh, IS_BWD branch, ParamOP Empty, ActOP silu / DActOP dsilu (as v1-v4)
          const float act_elt = __bfloat162float(a.h[e]);
          const float gate_elt = __bfloat162float(b.h[e]);
          const float dgate_elt = 1.0f;
          const float grad_elt = __bfloat162float(g.h[e]);
          const float xv = act_elt;
          float act_x;
          float dact_x;
          {
            const float s = te_sigmoidf(xv);
            act_x = xv * s;
            dact_x = xv * s * (1 - s) + s;
          }
          const float after_dact = dact_x * grad_elt * gate_elt;
          const float after_dgate = dgate_elt * act_x * grad_elt;
          fa[h] = scale * after_dact;
          fg[h] = scale * after_dgate;
        }
        oa[e2] = cvt_bf16x2(fa[0], fa[1]);
        og[e2] = cvt_bf16x2(fg[0], fg[1]);
        pre0 = bmax_xa(pre0, oa[e2]);
        pre1 = bmax_xa(pre1, og[e2]);
      }
      const uint4 ua = make_uint4(oa[0], oa[1], oa[2], oa[3]), ug = make_uint4(og[0], og[1], og[2], og[3]);
      *reinterpret_cast<uint4*>(out + gi * 2 * N + gj) = ua;
      *reinterpret_cast<uint4*>(out + gi * 2 * N + N + gj) = ug;
      *reinterpret_cast<uint4*>(sh_o0 + r * kLD6 + c8) = ua;
      *reinterpret_cast<uint4*>(sh_o1 + r * kLD6 + c8) = ug;
    } else {
      uint32_t o[4];
#pragma unroll
      for (int e2 = 0; e2 < 4; ++e2) {
        float f[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int e = 2 * e2 + h;
          // cast/fp8/gated_fp8.cuh, forward branch: after_act = ActOP(act_elt, p) * gate_elt, ActOP = silu<fp32, fp32>
          const float act_elt = __bfloat162float(a.h[e]);
          const float gate_elt = __bfloat162float(b.h[e]);
          const float after_act = te_silu(act_elt) * gate_elt;
          f[h] = scale * after_act;
        }
        o[e2] = cvt_bf16x2(f[0], f[1]);
        if (e2 & 1) pre1 = bmax_xa(pre1, o[e2]); else pre0 = bmax_xa(pre0, o[e2]);
      }
      const uint4 uo = make_uint4(o[0], o[1], o[2], o[3]);
      *reinterpret_cast<uint4*>(out + gi * N + gj) = uo;
      *reinterpret_cast<uint4*>(sh_o0 + r * kLD6 + c8) = uo;
    }
  }
  const RhtB B(sign_mask);
  __syncthreads();
  uint32_t post = 0u;
  constexpr int kSub = (kTR / 16) * (kTC / 16);       // 32 sub-tiles per output tensor
#pragma unroll 4
  for (int t = threadIdx.x / 32; t < (BWD ? 2 : 1) * kSub; t += kWarps) {
    const int which = t / kSub, rem = t % kSub, tb = rem / (kTC / 16), fb = rem % (kTC / 16);
    post = bmax_xa(post, rht_subtile_bmax<kLD6>((which ? sh_o1 : sh_o0) + tb * 16 * kLD6 + fb * 16, B));
  }
  block_publish2<kWarps>(fmax_noftz(unpack_bmax(pre0), unpack_bmax(pre1)), unpack_bmax(post), amax_pre, amax_post);
}

// ---- v7 (2026-10-05): dSwiGLU in warp-owned 16-token x 64-feature blocks (bench_dswiglu2.cu variant 16, job 20416:
// 360.9 against v6's 397.3 us, outputs and amaxes byte-identical to v6 on Llama and edge data).  Each warp computes its
// block with v6's per-element arithmetic (lane l: token 4 it + l / 8 of iteration it, features 8 (l % 8) .. + 7; each
// row segment one 128-byte line), the loads of its next vector issued before the current one is computed, stages the
// block in its own shared-memory region and takes its post-RHT amax right away: no CTA-wide barrier before the
// epilogue.  The loop is unrolled and the kernel bounded to 5 CTAs per SM: occupancy decided the bench.  SwiGLU stays on
// v6 (at the bandwidth floor).
struct In3 {
  V8 g, a, b;
};
__device__ __forceinline__ void load3(In3& in, const __nv_bfloat16* __restrict__ grad, const __nv_bfloat16* __restrict__ x,
                                      int64_t gi, int64_t gj, int64_t N) {
  in.g.u = *reinterpret_cast<const uint4*>(grad + gi * N + gj);
  in.a.u = *reinterpret_cast<const uint4*>(x + gi * 2 * N + gj);
  in.b.u = *reinterpret_cast<const uint4*>(x + gi * 2 * N + N + gj);
}
// v6's per-element arithmetic (TE's cast/fp8/gated_fp8.cuh IS_BWD branch, silu / dsilu) and packed outputs
__device__ __forceinline__ void dsw_math(const In3& in, uint32_t (&oa)[4], uint32_t (&og)[4], uint32_t& pre0,
                                         uint32_t& pre1) {
  const float scale = 1;
#pragma unroll
  for (int e2 = 0; e2 < 4; ++e2) {
    float fa[2], fg[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int e = 2 * e2 + h;
      const float act_elt = __bfloat162float(in.a.h[e]);
      const float gate_elt = __bfloat162float(in.b.h[e]);
      const float dgate_elt = 1.0f;
      const float grad_elt = __bfloat162float(in.g.h[e]);
      const float xv = act_elt;
      float act_x;
      float dact_x;
      {
        const float s = te_sigmoidf(xv);
        act_x = xv * s;
        dact_x = xv * s * (1 - s) + s;
      }
      const float after_dact = dact_x * grad_elt * gate_elt;
      const float after_dgate = dgate_elt * act_x * grad_elt;
      fa[h] = scale * after_dact;
      fg[h] = scale * after_dgate;
    }
    oa[e2] = cvt_bf16x2(fa[0], fa[1]);
    og[e2] = cvt_bf16x2(fg[0], fg[1]);
    pre0 = bmax_xa(pre0, oa[e2]);
    pre1 = bmax_xa(pre1, og[e2]);
  }
}
constexpr int kWB_R = 16, kWB_C = 64, kWB_LD = kWB_C + 8;   // 144-byte rows: the 8 rows of an ldmatrix in distinct banks
__global__ void __launch_bounds__(256, 5)
    dswiglu_amax_kernel_v7(const __nv_bfloat16* __restrict__ grad, const __nv_bfloat16* __restrict__ x,
                           __nv_bfloat16* __restrict__ out, float* __restrict__ amax_pre, float* __restrict__ amax_post,
                           int64_t M, int64_t N, uint32_t sign_mask) {
  constexpr int ITERS = kWB_R * kWB_C / 8 / 32;      // 4
  __shared__ __align__(128) __nv_bfloat16 sh[8][2][kWB_R * kWB_LD];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __nv_bfloat16* s0 = sh[warp][0];
  __nv_bfloat16* s1 = sh[warp][1];
  const int64_t ncb = N / kWB_C, nblocks = (M / kWB_R) * ncb;
  const int64_t gw = (int64_t)blockIdx.x * 8 + warp;
  const int rr = lane >> 3, c8 = (lane & 7) * 8;
  uint32_t pre0 = 0u, pre1 = 0u, post = 0u;
  const RhtB B(sign_mask);
  In3 nxt;
  if (gw < nblocks) load3(nxt, grad, x, (gw / ncb) * kWB_R + rr, (gw % ncb) * kWB_C + c8, N);
  for (int64_t b = gw; b < nblocks; b += nblocks) {   // one block per warp
    const int64_t rowb = (b / ncb) * kWB_R, colb = (b % ncb) * kWB_C + c8;
#pragma unroll 4
    for (int it = 0; it < ITERS; ++it) {
      const int r = it * 4 + rr;
      const int64_t gi = rowb + r;
      In3 cur;
      cur = nxt;
      if (it + 1 < ITERS) load3(nxt, grad, x, gi + 4, colb, N);
      uint32_t oa[4], og[4];
      dsw_math(cur, oa, og, pre0, pre1);
      const uint4 ua = make_uint4(oa[0], oa[1], oa[2], oa[3]), ug = make_uint4(og[0], og[1], og[2], og[3]);
      *reinterpret_cast<uint4*>(out + gi * 2 * N + colb) = ua;
      *reinterpret_cast<uint4*>(out + gi * 2 * N + N + colb) = ug;
      *reinterpret_cast<uint4*>(s0 + r * kWB_LD + c8) = ua;
      *reinterpret_cast<uint4*>(s1 + r * kWB_LD + c8) = ug;
    }
    __syncwarp();
#pragma unroll
    for (int fb = 0; fb < kWB_C / 16; ++fb) {
      post = bmax_xa(post, rht_subtile_bmax<kWB_LD>(s0 + fb * 16, B));
      post = bmax_xa(post, rht_subtile_bmax<kWB_LD>(s1 + fb * 16, B));
    }
    __syncwarp();
  }
  block_publish2<8>(fmax_noftz(unpack_bmax(pre0), unpack_bmax(pre1)), unpack_bmax(post), amax_pre, amax_post);
}

// ---- v8 (2026-10-07, dev bw19): v7's warp-owned 16-token x 64-feature blocks, per-element arithmetic, staging and
// epilogue, with the block's four iterations' loads issued up front (12 vectors per lane, 3 CTAs per SM; job 20919:
// 354.9 against v7's 362.7 us idle, 369.1 against 381.7 beside a 32-CTA copy; outputs and amaxes byte-identical to v7 on
// Llama, edge, large and odd shapes, eager and in a graph); REV: the blocks in reverse order (the last tokens first: the
// rows a producer wrote last are read while in L2, and the rows the next cast reads first are written last).
template <bool REV>
__global__ void __launch_bounds__(256, 3)
    dswiglu_amax_kernel_v8(const __nv_bfloat16* __restrict__ grad, const __nv_bfloat16* __restrict__ x,
                           __nv_bfloat16* __restrict__ out, float* __restrict__ amax_pre, float* __restrict__ amax_post,
                           int64_t M, int64_t N, uint32_t sign_mask) {
  constexpr int ITERS = kWB_R * kWB_C / 8 / 32;      // 4
  __shared__ __align__(128) __nv_bfloat16 sh[8][2][kWB_R * kWB_LD];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __nv_bfloat16* s0 = sh[warp][0];
  __nv_bfloat16* s1 = sh[warp][1];
  const int64_t ncb = N / kWB_C, nblocks = (M / kWB_R) * ncb;
  const int64_t gw = (int64_t)blockIdx.x * 8 + warp;
  const int rr = lane >> 3, c8 = (lane & 7) * 8;
  uint32_t pre0 = 0u, pre1 = 0u, post = 0u;
  const RhtB B(sign_mask);
  if (gw < nblocks) {
    const int64_t b = REV ? nblocks - 1 - gw : gw;
    const int64_t rowb = (b / ncb) * kWB_R, colb = (b % ncb) * kWB_C + c8;
    In3 q[ITERS];
#pragma unroll
    for (int it = 0; it < ITERS; ++it) load3(q[it], grad, x, rowb + it * 4 + rr, colb, N);
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int r = it * 4 + rr;
      const int64_t gi = rowb + r;
      uint32_t oa[4], og[4];
      dsw_math(q[it], oa, og, pre0, pre1);
      const uint4 ua = make_uint4(oa[0], oa[1], oa[2], oa[3]), ug = make_uint4(og[0], og[1], og[2], og[3]);
      *reinterpret_cast<uint4*>(out + gi * 2 * N + colb) = ua;
      *reinterpret_cast<uint4*>(out + gi * 2 * N + N + colb) = ug;
      *reinterpret_cast<uint4*>(s0 + r * kWB_LD + c8) = ua;
      *reinterpret_cast<uint4*>(s1 + r * kWB_LD + c8) = ug;
    }
    __syncwarp();
#pragma unroll
    for (int fb = 0; fb < kWB_C / 16; ++fb) {
      post = bmax_xa(post, rht_subtile_bmax<kWB_LD>(s0 + fb * 16, B));
      post = bmax_xa(post, rht_subtile_bmax<kWB_LD>(s1 + fb * 16, B));
    }
  }
  block_publish2<8>(fmax_noftz(unpack_bmax(pre0), unpack_bmax(pre1)), unpack_bmax(post), amax_pre, amax_post);
}

PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q));
    TORCH_CHECK(q == cudaDriverEntryPointSuccess && p != nullptr, "cuTensorMapEncodeTiled not available");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// a row-major [rows, cols] bf16 tensor, boxes of kTR5 rows x kTC5 columns, no swizzle (the inputs are read row-wise)
CUtensorMap tensor_map_2d(const void* base, int64_t rows, int64_t cols) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {(cuuint32_t)kTC5, (cuuint32_t)kTR5};
  cuuint32_t estr[2] = {1, 1};
  const CUresult r = tensor_map_encoder()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides,
                                          box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

void check_inputs(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
              " must be a contiguous CUDA bf16 tensor");
}

}  // namespace

// dswiglu_amax(grad [.., N], x [.., 2N], sign_mask) -> (out [.., 2N] bf16, amax_pre [1] f32, amax_post [1] f32)
std::vector<at::Tensor> dswiglu_amax(const at::Tensor& grad, const at::Tensor& x, int64_t sign_mask, int64_t version) {
  check_inputs(grad, "grad");
  check_inputs(x, "x");
  const int64_t N = grad.size(-1);
  const int64_t M = grad.numel() / N;
  TORCH_CHECK(x.size(-1) == 2 * N && x.numel() == 2 * grad.numel(), "x must be [.., 2N] with grad [.., N]");
  TORCH_CHECK(M % kTR == 0 && N % kTC == 0, "dswiglu_amax: rows % 32 and N % 256 required");
  const at::cuda::CUDAGuard guard(grad.device());
  auto out = at::empty_like(x);
  auto amax = at::zeros({2}, grad.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)(N / kTC), (unsigned)(M / kTR));
  if (version == 5) {
    static bool attr5 = false;
    constexpr size_t smem = smem5_bytes<true, kStagesBwd5>();
    if (!attr5) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(swiglu_family_amax_kernel_v5<true, kStagesBwd5>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      attr5 = true;
    }
    const CUtensorMap tx = tensor_map_2d(x.data_ptr(), M, 2 * N);
    const CUtensorMap tg = tensor_map_2d(grad.data_ptr(), M, N);
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    swiglu_family_amax_kernel_v5<true, kStagesBwd5><<<(unsigned)sms, kThreads, smem, stream>>>(
        tx, tg, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M,
        N, (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  if (version == 3) {
    static bool attr = false;
    if (!attr) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(swiglu_family_amax_kernel_v3<true>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem3));
      attr = true;
    }
    int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    swiglu_family_amax_kernel_v3<true><<<(unsigned)(sms * 3), kThreads, kSmem3, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N,
        (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  if (version == 8 || version == 9 || version == 10) {
    const int64_t nblocks = (M / kWB_R) * (N / kWB_C);
    const unsigned g = (unsigned)((nblocks + 7) / 8);
    if (version == 8)
      dswiglu_amax_kernel_v8<false><<<g, 256, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N,
          (uint32_t)sign_mask);
    else
      dswiglu_amax_kernel_v8<true><<<g, 256, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N,
          (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  if (version == 7) {
    const int64_t nblocks = (M / kWB_R) * (N / kWB_C);
    dswiglu_amax_kernel_v7<<<(unsigned)((nblocks + 7) / 8), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N,
        (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  if (version == 6) {
    swiglu_family_amax_kernel_v6<true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N,
        (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  auto kern = version == 2 ? dswiglu_amax_kernel_v2
              : version == 4 ? dswiglu_amax_kernel_t<kTC + 8> : dswiglu_amax_kernel_t<kTC>;
  kern<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N,
      (uint32_t)sign_mask);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
}

// swiglu_amax(x [.., 2N], sign_mask) -> (out [.., N] bf16, amax_pre [1] f32, amax_post [1] f32)
std::vector<at::Tensor> swiglu_amax(const at::Tensor& x, int64_t sign_mask, int64_t version) {
  check_inputs(x, "x");
  const int64_t N = x.size(-1) / 2;
  const int64_t M = x.numel() / (2 * N);
  TORCH_CHECK(M % kTR == 0 && N % kTC == 0, "swiglu_amax: rows % 32 and N % 256 required");
  const at::cuda::CUDAGuard guard(x.device());
  auto sizes = x.sizes().vec();
  sizes.back() = N;
  auto out = at::empty(sizes, x.options());
  auto amax = at::zeros({2}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)(N / kTC), (unsigned)(M / kTR));
  if (version == 5) {
    static bool attr5 = false;
    constexpr size_t smem = smem5_bytes<false, kStagesFwd5>();
    if (!attr5) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(swiglu_family_amax_kernel_v5<false, kStagesFwd5>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      attr5 = true;
    }
    const CUtensorMap tx = tensor_map_2d(x.data_ptr(), M, 2 * N);
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    swiglu_family_amax_kernel_v5<false, kStagesFwd5><<<(unsigned)sms, kThreads, smem, stream>>>(
        tx, tx, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M,
        N, (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  if (version == 3) {
    static bool attr = false;
    if (!attr) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(swiglu_family_amax_kernel_v3<false>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmem3));
      attr = true;
    }
    int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    swiglu_family_amax_kernel_v3<false><<<(unsigned)(sms * 3), kThreads, kSmem3, stream>>>(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N, (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  if (version == 9) {                             // v9: SwiGLU v6 with the row blocks in reverse order
    swiglu_family_amax_kernel_v6<false, true><<<grid, kThreads, 0, stream>>>(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N, (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  if (version == 6 || version == 7 || version == 8 || version == 10) {   // v7, v8, v10 are dSwiGLU changes
    swiglu_family_amax_kernel_v6<false><<<grid, kThreads, 0, stream>>>(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N, (uint32_t)sign_mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
  }
  auto kern = version == 2 ? swiglu_amax_kernel_v2
              : version == 4 ? swiglu_amax_kernel_t<kTC + 8> : swiglu_amax_kernel_t<kTC>;
  kern<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      amax.data_ptr<float>(), amax.data_ptr<float>() + 1, M, N, (uint32_t)sign_mask);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, amax.narrow(0, 0, 1), amax.narrow(0, 1, 1)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dswiglu_amax", &dswiglu_amax, "dSwiGLU (TE arithmetic) + pre/post-RHT amax", py::arg("grad"), py::arg("x"),
        py::arg("sign_mask"), py::arg("version") = 2);
  m.def("swiglu_amax", &swiglu_amax, "SwiGLU (TE arithmetic) + pre/post-RHT amax", py::arg("x"), py::arg("sign_mask"),
        py::arg("version") = 2);
}
