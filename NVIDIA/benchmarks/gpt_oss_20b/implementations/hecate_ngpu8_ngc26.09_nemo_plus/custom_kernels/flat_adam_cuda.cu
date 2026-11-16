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

// Flat per-bucket capturable master Adam kernel: one launch per distributed-optimizer bucket shard instead of TE's
// AdamCapturableMasterFunctor multi_tensor_apply launches (150 per step, ~20.6 ms at 4.9 TB/s).
//
// Per 16-B vector of 8 elements: one 16-B grad load, three 32-B fp32 state loads (master, m, v), one 16-B param store,
// three 32-B state stores = 28 B/element; the loads are cache-streaming (read once).  The element math is TE's
// AdamCapturableMasterFunctor (multi_tensor/adam.cu) verbatim -- same expressions, same float types, pow(float, int)
// bias correction from the device step scalar, g * (*inv_scale) on load -- and the .cu is compiled with the same
// --use_fast_math TE's CMake applies to adam.cu, so the update is bit-identical to the TE kernel (verified bitwise).
// The parameter group of a vector comes from a binary search over the sorted segment table in shared memory
// (seg_start[]); a vector straddling a segment boundary (never with Megatron's 64-element param alignment, but
// handled) takes the scalar path; padding gaps (group -1) are skipped.
// Sample inputs: one bucket shard numel = 96.6M (bucket 1.44 GiB bf16 / DP 8), 2 groups, ~40 segments.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "flat_adam_cuda.h"

namespace pfa {
namespace {

__device__ __forceinline__ void adam_elem(float& r_g, float& r_p, float& r_m, float& r_v, const float beta1,
                                          const float beta2, const float beta1_correction,
                                          const float beta2_correction, const float epsilon, const float lr,
                                          const float decay, const int mode) {
  // transformer_engine/common/multi_tensor/adam.cu, AdamCapturableMasterFunctor, one ILP element
  if (mode == 0) {  // L2
    r_g = r_g + (decay * r_p);
    r_m = beta1 * r_m + (1 - beta1) * r_g;
    r_v = beta2 * r_v + (1 - beta2) * r_g * r_g;
    float next_m_unbiased = r_m / beta1_correction;
    float next_v_unbiased = r_v / beta2_correction;
    float denom = sqrtf(next_v_unbiased) + epsilon;
    float update = next_m_unbiased / denom;
    r_p = r_p - (lr * update);
  } else {  // weight decay
    r_m = beta1 * r_m + (1 - beta1) * r_g;
    r_v = beta2 * r_v + (1 - beta2) * r_g * r_g;
    float next_m_unbiased = r_m / beta1_correction;
    float next_v_unbiased = r_v / beta2_correction;
    float denom = sqrtf(next_v_unbiased) + epsilon;
    float update = (next_m_unbiased / denom) + (decay * r_p);
    r_p = r_p - (lr * update);
  }
}

template <int THREADS>
__global__ void __launch_bounds__(THREADS) flat_adam_kernel(const Args a, const Groups gr) {
  if (*a.noop == 1) return;

  __shared__ long long s_start[kMaxSeg + 1];
  __shared__ int s_group[kMaxSeg];
  __shared__ float s_lr[kMaxGroups], s_bc1[kMaxGroups], s_bc2[kMaxGroups];
  __shared__ float s_b1[kMaxGroups], s_b2[kMaxGroups], s_eps[kMaxGroups], s_wd[kMaxGroups];
  __shared__ int s_mode[kMaxGroups];

  for (int i = threadIdx.x; i <= a.nseg; i += THREADS) s_start[i] = a.seg_start[i];
  for (int i = threadIdx.x; i < a.nseg; i += THREADS) s_group[i] = a.seg_group[i];
  if (threadIdx.x < gr.n) {
    const Group G = gr.g[threadIdx.x];
    float beta1_correction = 1.0f, beta2_correction = 1.0f;
    if (G.bias_correction == 1) {
      beta1_correction = 1 - pow(G.beta1, *G.step);
      beta2_correction = 1 - pow(G.beta2, *G.step);
    }
    s_lr[threadIdx.x] = *G.lr;
    s_bc1[threadIdx.x] = beta1_correction;
    s_bc2[threadIdx.x] = beta2_correction;
    s_b1[threadIdx.x] = G.beta1;
    s_b2[threadIdx.x] = G.beta2;
    s_eps[threadIdx.x] = G.eps;
    s_wd[threadIdx.x] = G.decay;
    s_mode[threadIdx.x] = G.mode;
  }
  __syncthreads();

  const float inv_scale = *a.inv_scale;
  const long long nvec = a.numel / kVec;
  __nv_bfloat16* gp_all = reinterpret_cast<__nv_bfloat16*>(a.g);
  __nv_bfloat16* pp_all = reinterpret_cast<__nv_bfloat16*>(a.p);

  for (long long vi = static_cast<long long>(blockIdx.x) * THREADS + threadIdx.x; vi < nvec;
       vi += static_cast<long long>(gridDim.x) * THREADS) {
    const long long e0 = vi * kVec;
    // largest s with s_start[s] <= e0
    int lo = 0, hi = a.nseg;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (s_start[mid] <= e0) lo = mid; else hi = mid;
    }
    const int gi = s_group[lo];
    const long long seg_end = s_start[lo + 1];

    if (e0 + kVec <= seg_end) {
      if (gi < 0) continue;  // padding gap
      const float beta1 = s_b1[gi], beta2 = s_b2[gi], bc1 = s_bc1[gi], bc2 = s_bc2[gi];
      const float epsilon = s_eps[gi], lr = s_lr[gi], decay = s_wd[gi];
      const int mode = s_mode[gi];

      const uint4 gv = __ldcs(reinterpret_cast<const uint4*>(a.g + e0));
      const float4 ma = __ldcs(reinterpret_cast<const float4*>(a.master + e0));
      const float4 mb = __ldcs(reinterpret_cast<const float4*>(a.master + e0 + 4));
      const float4 m0 = __ldcs(reinterpret_cast<const float4*>(a.m + e0));
      const float4 m1 = __ldcs(reinterpret_cast<const float4*>(a.m + e0 + 4));
      const float4 v0 = __ldcs(reinterpret_cast<const float4*>(a.v + e0));
      const float4 v1 = __ldcs(reinterpret_cast<const float4*>(a.v + e0 + 4));

      float r_g[kVec], r_p[kVec], r_m[kVec], r_v[kVec];
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 f = __bfloat1622float2(g2[k]);
        r_g[2 * k] = f.x * inv_scale;
        r_g[2 * k + 1] = f.y * inv_scale;
      }
      r_p[0] = ma.x; r_p[1] = ma.y; r_p[2] = ma.z; r_p[3] = ma.w; r_p[4] = mb.x; r_p[5] = mb.y; r_p[6] = mb.z; r_p[7] = mb.w;
      r_m[0] = m0.x; r_m[1] = m0.y; r_m[2] = m0.z; r_m[3] = m0.w; r_m[4] = m1.x; r_m[5] = m1.y; r_m[6] = m1.z; r_m[7] = m1.w;
      r_v[0] = v0.x; r_v[1] = v0.y; r_v[2] = v0.z; r_v[3] = v0.w; r_v[4] = v1.x; r_v[5] = v1.y; r_v[6] = v1.z; r_v[7] = v1.w;

      if (a.write_g) {  // TE writes the scaled grad back (before the L2 decay term is added)
        uint4 gs;
        __nv_bfloat162* s2 = reinterpret_cast<__nv_bfloat162*>(&gs);
#pragma unroll
        for (int k = 0; k < 4; ++k) s2[k] = __floats2bfloat162_rn(r_g[2 * k], r_g[2 * k + 1]);
        __stcs(reinterpret_cast<uint4*>(a.g + e0), gs);
      }

#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        adam_elem(r_g[k], r_p[k], r_m[k], r_v[k], beta1, beta2, bc1, bc2, epsilon, lr, decay, mode);
      }

      uint4 pv;
      __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&pv);
#pragma unroll
      for (int k = 0; k < 4; ++k) p2[k] = __floats2bfloat162_rn(r_p[2 * k], r_p[2 * k + 1]);
      __stcs(reinterpret_cast<uint4*>(a.p + e0), pv);
      __stcs(reinterpret_cast<float4*>(a.master + e0), make_float4(r_p[0], r_p[1], r_p[2], r_p[3]));
      __stcs(reinterpret_cast<float4*>(a.master + e0 + 4), make_float4(r_p[4], r_p[5], r_p[6], r_p[7]));
      __stcs(reinterpret_cast<float4*>(a.m + e0), make_float4(r_m[0], r_m[1], r_m[2], r_m[3]));
      __stcs(reinterpret_cast<float4*>(a.m + e0 + 4), make_float4(r_m[4], r_m[5], r_m[6], r_m[7]));
      __stcs(reinterpret_cast<float4*>(a.v + e0), make_float4(r_v[0], r_v[1], r_v[2], r_v[3]));
      __stcs(reinterpret_cast<float4*>(a.v + e0 + 4), make_float4(r_v[4], r_v[5], r_v[6], r_v[7]));
    } else {
      // the vector straddles a segment boundary: scalar path, per-element group
      int s = lo;
      for (int j = 0; j < kVec; ++j) {
        const long long e = e0 + j;
        while (s + 1 < a.nseg && s_start[s + 1] <= e) ++s;
        const int gj = s_group[s];
        if (gj < 0) continue;
        float r_g = __bfloat162float(gp_all[e]) * inv_scale;
        if (a.write_g) gp_all[e] = __float2bfloat16(r_g);
        float r_p = a.master[e];
        float r_m = a.m[e];
        float r_v = a.v[e];
        adam_elem(r_g, r_p, r_m, r_v, s_b1[gj], s_b2[gj], s_bc1[gj], s_bc2[gj], s_eps[gj], s_lr[gj], s_wd[gj], s_mode[gj]);
        pp_all[e] = __float2bfloat16(r_p);
        a.master[e] = r_p;
        a.m[e] = r_m;
        a.v[e] = r_v;
      }
    }
  }
}

}  // namespace

cudaError_t launch(const Args& a, const Groups& gr, int grid, int threads, cudaStream_t stream) {
  if (grid <= 0) return cudaErrorInvalidValue;
  switch (threads) {
    case 128: flat_adam_kernel<128><<<grid, 128, 0, stream>>>(a, gr); break;
    case 256: flat_adam_kernel<256><<<grid, 256, 0, stream>>>(a, gr); break;
    case 512: flat_adam_kernel<512><<<grid, 512, 0, stream>>>(a, gr); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t attrs(int threads, int* num_regs, int* blocks_per_sm, int* local_bytes, int* static_smem) {
  const void* fn = nullptr;
  switch (threads) {
    case 128: fn = reinterpret_cast<const void*>(&flat_adam_kernel<128>); break;
    case 256: fn = reinterpret_cast<const void*>(&flat_adam_kernel<256>); break;
    case 512: fn = reinterpret_cast<const void*>(&flat_adam_kernel<512>); break;
    default: return cudaErrorInvalidValue;
  }
  cudaFuncAttributes fa;
  cudaError_t err = cudaFuncGetAttributes(&fa, fn);
  if (err != cudaSuccess) return err;
  *num_regs = fa.numRegs;
  *local_bytes = static_cast<int>(fa.localSizeBytes);
  *static_smem = static_cast<int>(fa.sharedSizeBytes);
  int blocks = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, threads, 0);
  if (err != cudaSuccess) return err;
  *blocks_per_sm = blocks;
  return cudaSuccess;
}

}  // namespace pfa
