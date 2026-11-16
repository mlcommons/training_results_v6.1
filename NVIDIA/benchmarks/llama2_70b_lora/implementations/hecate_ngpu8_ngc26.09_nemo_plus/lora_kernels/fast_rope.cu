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

// Fused QKV RoPE (forward and backward) for bf16, non-interleaved, d == d2 == 128.
// The per-element math is a copy of TE fused_rope.cu (commit 5b28d7aa).
// Only the memory access changes: each thread moves 8 elements with 16-byte loads and stores.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

constexpr int kDim = 128;      // head dim and rotary dim
constexpr int kHalf = kDim / 2;
constexpr int kVec = 8;        // bf16 elements in 16 bytes
constexpr int kThreads = 128;

struct alignas(16) Bf16x8 {
  __nv_bfloat16 v[kVec];
};

struct RowOffsets {
  long long src;
  long long q;
  long long k;
  long long v;
};

__device__ RowOffsets token_offsets(int s_id, int b_id, int s, int b, int h, int q_split,
                                    int k_split, int v_split, bool sbhd) {
  const long long token = sbhd ? (static_cast<long long>(s_id) * b + b_id)
                               : (static_cast<long long>(b_id) * s + s_id);
  const int total_d = q_split + k_split + v_split;
  return {token * h * total_d, token * h * q_split, token * h * k_split, token * h * v_split};
}

// Same call and same order as TE: sincosf(freq, &sin, &cos) for q and for k.
__device__ void load_cos_sin(const float *q_freqs, const float *k_freqs, int s_id, float *smem) {
  for (int i = threadIdx.x; i < kDim; i += blockDim.x) {
    sincosf(q_freqs[s_id * kDim + i], &smem[kDim + i], &smem[i]);
    sincosf(k_freqs[s_id * kDim + i], &smem[3 * kDim + i], &smem[2 * kDim + i]);
  }
  __syncthreads();
}

// Forward math of TE for the first half (d_id < 64) and the second half (d_id >= 64) of a head.
// src_lo holds elements j..j+7 and src_hi holds elements j+64..j+71 of the same head.
__device__ void rope_forward_pair(const Bf16x8 &src_lo, const Bf16x8 &src_hi, const float *cos_v,
                                  const float *sin_v, int j, Bf16x8 &out_lo, Bf16x8 &out_hi) {
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    {
      const int d_id = j + e;
      const float v_cos = cos_v[d_id];
      const float v_sin = sin_v[d_id];
      const float v_src = __bfloat162float(src_lo.v[e]);
      const float v_src_rotate = -__bfloat162float(src_hi.v[e]);
      out_lo.v[e] = __float2bfloat16(v_src * v_cos + v_src_rotate * v_sin);
    }
    {
      const int d_id = j + kHalf + e;
      const float v_cos = cos_v[d_id];
      const float v_sin = sin_v[d_id];
      const float v_src = __bfloat162float(src_hi.v[e]);
      const float v_src_rotate = __bfloat162float(src_lo.v[e]);
      out_hi.v[e] = __float2bfloat16(v_src * v_cos + v_src_rotate * v_sin);
    }
  }
}

// Backward math of TE for the two halves of a head, as in rope_forward_pair.
__device__ void rope_backward_pair(const Bf16x8 &src_lo, const Bf16x8 &src_hi, const float *cos_v,
                                   const float *sin_v, int j, Bf16x8 &out_lo, Bf16x8 &out_hi) {
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    {
      const int d_id = j + e;
      const float v_src = __bfloat162float(src_lo.v[e]);
      const float v_cos = cos_v[d_id];
      const float v_src_rotate = __bfloat162float(src_hi.v[e]);
      const float v_sin = sin_v[d_id + kHalf];
      out_lo.v[e] = __float2bfloat16(v_src * v_cos + v_src_rotate * v_sin);
    }
    {
      const int d_id = j + kHalf + e;
      const float v_src = __bfloat162float(src_hi.v[e]);
      const float v_cos = cos_v[d_id];
      const float v_src_rotate = __bfloat162float(src_lo.v[e]);
      const float v_sin = -sin_v[d_id + kHalf - kDim];
      out_hi.v[e] = __float2bfloat16(v_src * v_cos + v_src_rotate * v_sin);
    }
  }
}

__device__ Bf16x8 load_vec(const __nv_bfloat16 *ptr) { return *reinterpret_cast<const Bf16x8 *>(ptr); }

__device__ void store_vec(__nv_bfloat16 *ptr, const Bf16x8 &value) {
  *reinterpret_cast<Bf16x8 *>(ptr) = value;
}

// Tasks in one group row: q pairs, then k pairs, then v copies of two vectors.
struct GroupTasks {
  int q;
  int k;
  int total;
};

__device__ GroupTasks group_tasks(int q_split, int k_split, int v_split) {
  const int pairs_per_head = kHalf / kVec;
  const int q = (q_split / kDim) * pairs_per_head;
  const int k = (k_split / kDim) * pairs_per_head;
  return {q, k, q + k + v_split / (2 * kVec)};
}

__global__ void __launch_bounds__(kThreads)
    qkv_rope_forward_kernel(const __nv_bfloat16 *__restrict__ qkv, const float *__restrict__ q_freqs,
                            const float *__restrict__ k_freqs, __nv_bfloat16 *__restrict__ q_out,
                            __nv_bfloat16 *__restrict__ k_out, __nv_bfloat16 *__restrict__ v_out,
                            int s, int b, int h, int q_split, int k_split, int v_split, bool sbhd) {
  __shared__ float smem[4 * kDim];  // cos_q, sin_q, cos_k, sin_k
  load_cos_sin(q_freqs, k_freqs, blockIdx.x, smem);
  const RowOffsets off =
      token_offsets(blockIdx.x, blockIdx.y, s, b, h, q_split, k_split, v_split, sbhd);
  const int total_d = q_split + k_split + v_split;
  const GroupTasks tasks = group_tasks(q_split, k_split, v_split);
  for (int task = threadIdx.x; task < h * tasks.total; task += blockDim.x) {
    const int g = task / tasks.total;
    const int t = task % tasks.total;
    const __nv_bfloat16 *row = qkv + off.src + static_cast<long long>(g) * total_d;
    if (t >= tasks.q + tasks.k) {
      const int p = (t - tasks.q - tasks.k) * 2 * kVec;
      const __nv_bfloat16 *src = row + q_split + k_split + p;
      __nv_bfloat16 *dst = v_out + off.v + g * v_split + p;
      const Bf16x8 first = load_vec(src);
      const Bf16x8 second = load_vec(src + kVec);
      store_vec(dst, first);
      store_vec(dst + kVec, second);
      continue;
    }
    const bool is_q = t < tasks.q;
    const int pair = is_q ? t : t - tasks.q;
    const int j = (pair % (kHalf / kVec)) * kVec;
    const int p = (pair / (kHalf / kVec)) * kDim + j;
    const __nv_bfloat16 *part = row + (is_q ? 0 : q_split);
    const Bf16x8 src_lo = load_vec(part + p);
    const Bf16x8 src_hi = load_vec(part + p + kHalf);
    const float *cos_v = is_q ? smem : smem + 2 * kDim;
    Bf16x8 out_lo, out_hi;
    rope_forward_pair(src_lo, src_hi, cos_v, cos_v + kDim, j, out_lo, out_hi);
    __nv_bfloat16 *dst = is_q ? q_out + off.q + g * q_split : k_out + off.k + g * k_split;
    store_vec(dst + p, out_lo);
    store_vec(dst + p + kHalf, out_hi);
  }
}

__global__ void __launch_bounds__(kThreads)
    qkv_rope_backward_kernel(const __nv_bfloat16 *__restrict__ dq,
                             const __nv_bfloat16 *__restrict__ dk,
                             const __nv_bfloat16 *__restrict__ dv, const float *__restrict__ q_freqs,
                             const float *__restrict__ k_freqs, __nv_bfloat16 *__restrict__ dqkv,
                             int s, int b, int h, int q_split, int k_split, int v_split, bool sbhd) {
  __shared__ float smem[4 * kDim];
  load_cos_sin(q_freqs, k_freqs, blockIdx.x, smem);
  const RowOffsets off =
      token_offsets(blockIdx.x, blockIdx.y, s, b, h, q_split, k_split, v_split, sbhd);
  const int total_d = q_split + k_split + v_split;
  const GroupTasks tasks = group_tasks(q_split, k_split, v_split);
  for (int task = threadIdx.x; task < h * tasks.total; task += blockDim.x) {
    const int g = task / tasks.total;
    const int t = task % tasks.total;
    __nv_bfloat16 *row = dqkv + off.src + static_cast<long long>(g) * total_d;
    if (t >= tasks.q + tasks.k) {
      const int p = (t - tasks.q - tasks.k) * 2 * kVec;
      const __nv_bfloat16 *src = dv + off.v + g * v_split + p;
      const Bf16x8 first = load_vec(src);
      const Bf16x8 second = load_vec(src + kVec);
      store_vec(row + q_split + k_split + p, first);
      store_vec(row + q_split + k_split + p + kVec, second);
      continue;
    }
    const bool is_q = t < tasks.q;
    const int pair = is_q ? t : t - tasks.q;
    const int j = (pair % (kHalf / kVec)) * kVec;
    const int p = (pair / (kHalf / kVec)) * kDim + j;
    const __nv_bfloat16 *grad = is_q ? dq + off.q + g * q_split : dk + off.k + g * k_split;
    const Bf16x8 src_lo = load_vec(grad + p);
    const Bf16x8 src_hi = load_vec(grad + p + kHalf);
    const float *cos_v = is_q ? smem : smem + 2 * kDim;
    Bf16x8 out_lo, out_hi;
    rope_backward_pair(src_lo, src_hi, cos_v, cos_v + kDim, j, out_lo, out_hi);
    __nv_bfloat16 *dst = row + (is_q ? 0 : q_split);
    store_vec(dst + p, out_lo);
    store_vec(dst + p + kHalf, out_hi);
  }
}

void check_inputs(const at::Tensor &freqs, const std::vector<int64_t> &split) {
  TORCH_CHECK(freqs.scalar_type() == at::kFloat && freqs.is_contiguous(), "freqs: fp32 contiguous");
  TORCH_CHECK(freqs.size(3) == kDim, "rotary dim must be 128");
  TORCH_CHECK(split.size() == 3 && split[1] == kDim && split[2] == kDim, "k and v split must be 128");
  TORCH_CHECK(split[0] % kDim == 0, "q split must be a multiple of 128");
}

std::vector<at::Tensor> qkv_rope_forward(const at::Tensor &qkv, const at::Tensor &q_freqs,
                                         const at::Tensor &k_freqs, std::vector<int64_t> split,
                                         bool sbhd) {
  check_inputs(q_freqs, split);
  check_inputs(k_freqs, split);
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.is_contiguous() && qkv.dim() == 4,
              "qkv: 4D bf16 contiguous");
  const int s = sbhd ? qkv.size(0) : qkv.size(1);
  const int b = sbhd ? qkv.size(1) : qkv.size(0);
  const int h = qkv.size(2);
  auto q_size = qkv.sizes().vec();
  q_size[2] = q_size[2] * split[0] / split[1];
  q_size[3] = split[1];
  auto kv_size = qkv.sizes().vec();
  kv_size[3] = split[1];
  auto q_out = at::empty(q_size, qkv.options());
  auto k_out = at::empty(kv_size, qkv.options());
  auto v_out = at::empty(kv_size, qkv.options());
  qkv_rope_forward_kernel<<<dim3(s, b), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(qkv.data_ptr()), q_freqs.data_ptr<float>(),
      k_freqs.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(q_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(k_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(v_out.data_ptr()), s, b, h, split[0], split[1], split[2],
      sbhd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_out, v_out};
}

at::Tensor qkv_rope_backward(const at::Tensor &dq, const at::Tensor &dk, const at::Tensor &dv,
                             const at::Tensor &q_freqs, const at::Tensor &k_freqs,
                             std::vector<int64_t> split, bool sbhd) {
  check_inputs(q_freqs, split);
  check_inputs(k_freqs, split);
  for (const auto &t : {dq, dk, dv}) {
    TORCH_CHECK(t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.dim() == 4,
                "grads: 4D bf16 contiguous");
  }
  const int s = sbhd ? dq.size(0) : dq.size(1);
  const int b = sbhd ? dq.size(1) : dq.size(0);
  const int64_t total_d = split[0] + split[1] + split[2];
  const int64_t total_hd = (dq.size(2) + dk.size(2) + dv.size(2)) * dq.size(3);
  auto size = dq.sizes().vec();
  size[2] = total_hd / total_d;
  size[3] = total_d;
  auto dqkv = at::empty(size, dq.options());
  qkv_rope_backward_kernel<<<dim3(s, b), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(dq.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(dk.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(dv.data_ptr()), q_freqs.data_ptr<float>(),
      k_freqs.data_ptr<float>(), reinterpret_cast<__nv_bfloat16 *>(dqkv.data_ptr()), s, b,
      size[2], split[0], split[1], split[2], sbhd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dqkv;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &qkv_rope_forward, "Fused QKV RoPE forward (bf16, d=128)");
  m.def("backward", &qkv_rope_backward, "Fused QKV RoPE backward (bf16, d=128)");
}
