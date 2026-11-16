// ak_l8b_rht_epi.cuh (Llama 3.1 8B kernel campaign, 2026-10-05): the NVFP4 amax epilogue shared by the producer kernels
// (ak_l8b_nvfp4p SwiGLU / dSwiGLU, ak_l8b_nvfp4n RMSNorm apply, ak_l8b_nvfp4b post-RHT amax), built from TE's own amax
// instructions in HadamardAmaxTmaKernel (common/hadamard_transform/hadamard_transform.cu, hadamard_transform_utils.cuh at
// 131157b5):
//  * the pre-RHT amax: max.xorsign.abs.bf16x2 over the packed bf16 output (no flush to zero);
//  * the post-RHT amax of each 16-token x 16-feature sub-tile of the columnwise copy: A = tile^T by ldmatrix.trans from
//    the staging tile, the constant B = T = diag(sign) H16 * 0.25 fragments in registers, mma.sync m16n8k16 (bf16 in,
//    fp32 accumulate from zero: the HMMA that TE's wmma m16n16k16 issues, k = token as TE), then TE's
//    cvt.rn.bf16x2.f32 and max.xorsign.abs.bf16x2;
//  * the reductions after that in fp32 with max.f32 and a sign-bit mask, as TE's ReduceMax (not built with fast math):
//    no flush to zero whatever the including file's flags.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace ak_rht {

__device__ __forceinline__ uint32_t bmax_xa(uint32_t a, uint32_t b) {      // TE's packed bf16 abs-max
  uint32_t d;
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t cvt_bf16x2(float lo, float hi) {     // {lo -> bits 0-15, hi -> bits 16-31}
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ float fmax_noftz(float a, float b) {
  float d;
  asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float fabs_noftz(float a) { return __int_as_float(__float_as_int(a) & 0x7fffffff); }
// |max| of the two halves of a packed bf16 abs-max (exact widening), as TE's unpack_max_of_packed_bf16
__device__ __forceinline__ float unpack_bmax(uint32_t p) {
  return fmax_noftz(fabs_noftz(__uint_as_float(p << 16)), fabs_noftz(__uint_as_float(p & 0xffff0000u)));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a)
               : "memory");
}
__device__ __forceinline__ void mma_16816_zero(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
}
// two bf16 entries T[k][n], T[k + 1][n] of the RHT matrix (sign[k] * H16[k][n] * 0.25, sign[k] = -1 where bit k of the
// mask is set), packed low / high
__device__ __forceinline__ uint32_t rht_pair(int k, int n, uint32_t sign_mask) {
  const uint32_t s0 = (uint32_t)(__popc(k & n) & 1) ^ ((sign_mask >> k) & 1u);
  const uint32_t s1 = (uint32_t)(__popc((k + 1) & n) & 1) ^ ((sign_mask >> (k + 1)) & 1u);
  return (s0 ? 0xBE80u : 0x3E80u) | ((s1 ? 0xBE80u : 0x3E80u) << 16);
}
// The B fragments of mma m16n8k16 (.col) for the two 8-column halves of T: lane = 4 g + q holds rows 2q, 2q + 1 (b0) and
// 2q + 8, 2q + 9 (b1) of column n = 8 nb + g.
struct RhtB {
  uint32_t b[2][2];
  __device__ __forceinline__ explicit RhtB(uint32_t sign_mask) {
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
      b[nb][0] = rht_pair(2 * q, 8 * nb + g, sign_mask);
      b[nb][1] = rht_pair(2 * q + 8, 8 * nb + g, sign_mask);
    }
  }
};
// The packed bf16 abs-max of bf16(tile^T x T) for one 16 x 16 sub-tile [token][feature] at `tile` (row stride LD
// elements, rows 16-byte aligned): A[m = feature][k = token] by ldmatrix.trans (lane l addresses row l % 8 of 8 x 8 block
// l / 8; blocks: tokens 0-7 / features 0-7, tokens 0-7 / features 8-15, tokens 8-15 / features 0-7, tokens 8-15 /
// features 8-15 = a0..a3).  Called by all 32 lanes of a warp.
template <int LD>
__device__ __forceinline__ uint32_t rht_subtile_bmax(const __nv_bfloat16* tile, const RhtB& B) {
  const int lane = threadIdx.x & 31, blk = lane >> 3;
  uint32_t a0, a1, a2, a3;
  ldsm_x4_trans(a0, a1, a2, a3, tile + ((blk >> 1) * 8 + (lane & 7)) * LD + (blk & 1) * 8);
  float d0[4], d1[4];
  mma_16816_zero(d0, a0, a1, a2, a3, B.b[0][0], B.b[0][1]);
  mma_16816_zero(d1, a0, a1, a2, a3, B.b[1][0], B.b[1][1]);
  const uint32_t c0 = cvt_bf16x2(d0[0], d0[1]), c1 = cvt_bf16x2(d0[2], d0[3]);
  const uint32_t c2 = cvt_bf16x2(d1[0], d1[1]), c3 = cvt_bf16x2(d1[2], d1[3]);
  return bmax_xa(bmax_xa(c0, c2), bmax_xa(c1, c3));
}
// The block's two amaxes (non-negative) to global with an integer atomic max (TE's atomicMaxFloat contract); WARPS warps.
template <int WARPS>
__device__ __forceinline__ void block_publish2(float a_pre, float a_post, float* amax_pre, float* amax_post) {
  __shared__ float red_ak[2][WARPS];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    a_pre = fmax_noftz(a_pre, __shfl_xor_sync(0xffffffffu, a_pre, o));
    a_post = fmax_noftz(a_post, __shfl_xor_sync(0xffffffffu, a_post, o));
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) {
    red_ak[0][warp] = a_pre;
    red_ak[1][warp] = a_post;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float p = 0.f, q = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      p = fmax_noftz(p, red_ak[0][w]);
      q = fmax_noftz(q, red_ak[1][w]);
    }
    if (amax_pre != nullptr) atomicMax(reinterpret_cast<int*>(amax_pre), __float_as_int(p));
    if (amax_post != nullptr) atomicMax(reinterpret_cast<int*>(amax_post), __float_as_int(q));
  }
}

}  // namespace ak_rht
