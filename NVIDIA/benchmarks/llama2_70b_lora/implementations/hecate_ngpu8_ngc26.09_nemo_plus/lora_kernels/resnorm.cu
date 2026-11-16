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

// Residual add + RMSNorm + NVFP4 quantize across the sublayer boundary, and its backward.
//
// MCore computes h = residual + x. The RMSNorm of the next sublayer reads h: the MLP of the same
// layer, or the qkv LoRA linear of the next layer. TE runs four kernels: torch add, cuDNN
// RMSNorm, amax pass, NVFP4 quantize. Here:
//   - forward_stats: h = bf16(residual + x), rstd per row, and amax partials of y = norm(h).
//   - quantize_norm: y from h, rstd, and the norm weight, then the NVFP4 quantize with the
//     nvfp4q math (TE 5b28d7aa) and the amax from the partials. For the qkv LoRA linear, it also
//     writes y in bf16: the LoRA branch reads the unquantized norm output (Bridge #6342).
//   - backward: dh = RMSNorm backward of the grads of y (one or two), plus the residual grad,
//     with amax partials for the next NVFP4 quantize. LoRA does not train the norm weight, so
//     the kernel makes no dgamma.
// forward_stats and the backward with one grad of y use a TMA row pipeline; the backward with two
// grads uses a register kernel with 1024 threads (the fastest choices on Rubin). The rstd sum
// order differs from cuDNN, and the two grads of y add in fp32 (TE rounds their sum to bf16), so
// the bits can differ from the TE path.
#include "nvfp4q_math.cuh"

#include <map>

namespace {

constexpr int kNormCols = 8192;  // The kernels support rows of 8192 bf16 values.
// A block of THREADS threads works on one row at a time. Thread t holds the vectors
// (8 bf16 each) t + k * THREADS, k < VPT, with VPT = kNormCols / 8 / THREADS.
template <int THREADS>
struct NormShape {
  static constexpr int kVpt = kNormCols / 8 / THREADS;
  static_assert(kVpt * 8 * THREADS == kNormCols, "THREADS must divide 1024");
};

// Sum over the block. All threads must call it. All threads get the result.
template <int THREADS>
__device__ __forceinline__ float block_sum(float value, float *shared) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < THREADS / 32 ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
  const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(words[i] << 16);
    f[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ uint32_t pack2(const float lo, const float hi) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t *>(&pair);
}

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  return make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7]));
}

__device__ __forceinline__ float bf16_round(const float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

// The norm weight of 8 columns as floats (+1 for a zero-centered gamma).
__device__ __forceinline__ void load_weight(const uint4 *w, const uint32_t vec, const bool zc,
                                            float (&g)[8]) {
  unpack8(__ldg(w + vec), g);
  if (zc) {
#pragma unroll
    for (int i = 0; i < 8; ++i) g[i] += 1.0f;
  }
}

// y = bf16(g * (h * rstd)), the TE rmsnorm forward order.
__device__ __forceinline__ float norm_value(const float h, const float rs, const float g) {
  return bf16_round(g * (h * rs));
}

// The warps of the block share the L2 prefetch of one row (16 KB) of a bf16 tensor.
template <int THREADS>
__device__ __forceinline__ void prefetch_row(const uint4 *base, const uint32_t row,
                                             const uint32_t rows) {
  constexpr uint32_t kWarpsInBlock = THREADS / 32;
  constexpr uint32_t kRowBytes = kNormCols * 2;
  constexpr uint32_t kPart = kRowBytes / kWarpsInBlock;
  if (row < rows && (threadIdx.x & 31) == 0) {
    const char *start = reinterpret_cast<const char *>(base) + static_cast<size_t>(row) * kRowBytes;
    prefetch_l2(start + (threadIdx.x >> 5) * kPart, kPart);
  }
}

struct StatsArgs {
  const uint4 *residual;
  const uint4 *x;
  const uint4 *w;
  uint4 *h;
  float *rstd;
  float *partial;
  uint32_t rows;
  float eps;
  bool zero_centered;
};

struct NormQuantizeArgs {
  QuantizeArgs q;  // q.x holds h.
  const float *rstd;
  const uint4 *w;
  uint4 *y;  // If not null: gets y = norm(h) in bf16 (the input of the LoRA branch).
  bool zero_centered;
};

// Two fp32 values in one 64-bit register, for the sm_100+ f32x2 instructions.
__device__ __forceinline__ uint64_t pair64(const float lo, const float hi) {
  uint64_t r;
  asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(lo), "f"(hi));
  return r;
}

// Two IEEE round-to-nearest products in one instruction (the same bits as two FMUL).
__device__ __forceinline__ uint64_t mul2(const uint64_t a, const uint64_t b) {
  uint64_t r;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}

// bf16x2 of a pair (round to nearest), low half from the low float.
__device__ __forceinline__ uint32_t pack_pair(const uint64_t p) {
  float lo;
  float hi;
  asm("mov.b64 {%0, %1}, %2;" : "=f"(lo), "=f"(hi) : "l"(p));
  return pack2(lo, hi);
}

// y of one unit vector: lane l of unit (row, ucol) holds columns ucol * 256 + 8l .. + 7.
// y = bf16(g * (h * rstd)) with packed fp32x2 products. ZC: zero-centered gamma.
template <bool ZC>
__device__ __forceinline__ uint4 norm_vector(const uint4 hv, const NormQuantizeArgs &a,
                                             const uint32_t row, const uint32_t ucol,
                                             const uint32_t lane) {
  const uint4 wv = __ldg(a.w + ucol * kUnitVecs + lane);
  const float rs = __ldg(a.rstd + row);
  const uint64_t rs2 = pair64(rs, rs);
  const uint32_t hw[4] = {hv.x, hv.y, hv.z, hv.w};
  const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
  uint32_t out[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float g0 = __uint_as_float(ww[i] << 16);
    float g1 = __uint_as_float(ww[i] & 0xffff0000u);
    if constexpr (ZC) {
      g0 += 1.0f;
      g1 += 1.0f;
    }
    const uint64_t h2 = pair64(__uint_as_float(hw[i] << 16), __uint_as_float(hw[i] & 0xffff0000u));
    out[i] = pack_pair(mul2(pair64(g0, g1), mul2(h2, rs2)));
  }
  return make_uint4(out[0], out[1], out[2], out[3]);
}

// The nvfp4q RN quantize kernel with the norm applied to each loaded vector.
template <bool ZC>
__global__ void __launch_bounds__(kThreads) quantize_norm_kernel(const NormQuantizeArgs a) {
  __shared__ float shared[kWarps];
  const QuantizeArgs &args = a.q;
  float amax = 0.0f;
  for (int b = threadIdx.x; b < args.n_partial; b += kThreads) amax = fmaxf(amax, args.partial[b]);
  amax = block_max<kThreads>(amax, shared);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (args.amax_out != nullptr) args.amax_out[0] = amax;
    write_gemm_alpha(args.alpha_weight_amax, args.alpha_out, amax);
  }
  ScaleConstants constants;
  const float s_enc = global_encode_scale(amax);
  constants.s_enc_times_fp4_inv = s_enc * kFp4MaxInv;
  constants.s_dec = 1.0f / s_enc;
  constants.num_tiles_x = args.cols >> 6;
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t units_per_row = args.cols >> 8;
  const uint32_t n_units = args.rows * units_per_row;
  const uint32_t warp_first = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * kUnroll;
  const uint32_t step = gridDim.x * kWarps * kUnroll;
  const auto prefetch_step = [&](uint32_t ahead) {
    if (lane == 0 && ahead < n_units) {
      const uint32_t low_unit = n_units - ahead - kUnroll;
      prefetch_l2(args.x + low_unit * kUnitVecs, kUnroll * kUnitVecs * sizeof(uint4));
    }
  };
  for (int d = 0; d < args.prefetch; ++d) prefetch_step(warp_first + d * step);
  for (uint32_t first = warp_first; first < n_units; first += step) {
    if (args.prefetch > 0) prefetch_step(first + args.prefetch * step);
    const uint32_t first_unit = n_units - 1 - first;
    uint32_t row[kUnroll];
    uint32_t ucol[kUnroll];
    unit_positions(first_unit, units_per_row, row, ucol);
    uint4 v[kUnroll];
#pragma unroll
    for (int k = 0; k < kUnroll; ++k) {
      v[k] = norm_vector<ZC>(load_stream(args.x + (first_unit - k) * kUnitVecs + lane), a,
                             row[k], ucol[k], lane);
      if (a.y != nullptr) a.y[(first_unit - k) * kUnitVecs + lane] = v[k];
    }
#pragma unroll
    for (int k = 0; k < kUnroll; k += 2) {
      const uint4 pair_v[2] = {v[k], v[k + 1]};
      const uint32_t pair_row[2] = {row[k], row[k + 1]};
      const uint32_t pair_ucol[2] = {ucol[k], ucol[k + 1]};
      float coefficient[2];
      pair_coefficients(pair_v, pair_row, pair_ucol, lane, constants, args.scales, coefficient);
      const uint32_t rbits[2][2] = {{0, 0}, {0, 0}};
      pair_convert<false>(pair_v, coefficient, rbits, args.data, first_unit - k, lane);
    }
  }
}

struct BackwardArgs {
  const uint4 *grad_h;  // Grad of h from the residual stream. Can be null.
  const uint4 *dy;      // Grad of y. Can be null.
  const uint4 *dy2;     // Second grad of y (the LoRA branch). Can be null.
  const uint4 *h;
  const float *rstd;
  const uint4 *w;
  uint4 *out;
  float *partial;
  uint32_t rows;
  bool zero_centered;
  int prefetch;  // L2 prefetch distance in rows of this block (0 = off).
};

// out = rstd * (g - xh * mean(g * xh)) + grad_h, with g = w * (dy + dy2) and xh = h * rstd.
template <int THREADS>
__global__ void __launch_bounds__(THREADS) backward_kernel(const BackwardArgs a) {
  constexpr int kVpt = NormShape<THREADS>::kVpt;
  __shared__ float shared[THREADS / 32];
  constexpr uint32_t kVpr = kNormCols / 8;
  float w[kVpt][8];
#pragma unroll
  for (int k = 0; k < kVpt; ++k) load_weight(a.w, threadIdx.x + k * THREADS, a.zero_centered, w[k]);
  float amax = 0.0f;
  for (int d = 1; d < a.prefetch; ++d) {
    const uint32_t ahead = blockIdx.x + d * gridDim.x;
    prefetch_row<THREADS>(a.h, ahead, a.rows);
    if (a.dy != nullptr) prefetch_row<THREADS>(a.dy, ahead, a.rows);
    if (a.dy2 != nullptr) prefetch_row<THREADS>(a.dy2, ahead, a.rows);
    if (a.grad_h != nullptr) prefetch_row<THREADS>(a.grad_h, ahead, a.rows);
  }
  for (uint32_t row = blockIdx.x; row < a.rows; row += gridDim.x) {
    if (a.prefetch > 0) {
      const uint32_t ahead = row + a.prefetch * gridDim.x;
      prefetch_row<THREADS>(a.h, ahead, a.rows);
      if (a.dy != nullptr) prefetch_row<THREADS>(a.dy, ahead, a.rows);
      if (a.dy2 != nullptr) prefetch_row<THREADS>(a.dy2, ahead, a.rows);
      if (a.grad_h != nullptr) prefetch_row<THREADS>(a.grad_h, ahead, a.rows);
    }
    const float rs = __ldg(a.rstd + row);
    float gw[kVpt][8];
    float xh[kVpt][8];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < kVpt; ++k) {
      const size_t idx = static_cast<size_t>(row) * kVpr + threadIdx.x + k * THREADS;
      float hv[8];
      unpack8(load_stream(a.h + idx), hv);
      float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (a.dy != nullptr) unpack8(load_stream(a.dy + idx), d);
      if (a.dy2 != nullptr) {
        float d2[8];
        unpack8(load_stream(a.dy2 + idx), d2);
#pragma unroll
        for (int i = 0; i < 8; ++i) d[i] += d2[i];
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        gw[k][i] = w[k][i] * d[i];
        xh[k][i] = hv[i] * rs;
        dot += gw[k][i] * xh[k][i];
      }
    }
    const float mean = block_sum<THREADS>(dot, shared) / static_cast<float>(kNormCols);
#pragma unroll
    for (int k = 0; k < kVpt; ++k) {
      const size_t idx = static_cast<size_t>(row) * kVpr + threadIdx.x + k * THREADS;
      float res[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (a.grad_h != nullptr) unpack8(load_stream(a.grad_h + idx), res);
      float o[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o[i] = rs * (gw[k][i] - xh[k][i] * mean) + res[i];
        amax = fmaxf(amax, fabsf(bf16_round(o[i])));
      }
      a.out[idx] = pack8(o);
    }
  }
  __syncthreads();  // All threads must read the last row sum before block_max reuses shared.
  amax = block_max<THREADS>(amax, shared);
  if (threadIdx.x == 0) a.partial[blockIdx.x] = amax;
}

// ---------------------------------------------------------------------------------------------
// TMA row pipeline. Thread 0 starts bulk copies (cp.async.bulk) of the input rows of the next
// STAGES rows of the block into shared memory, and an mbarrier tells when a stage is full. Thus
// the loads stay in flight during the row sums, and the data in flight uses no registers. After
// the row sum, the threads read their vectors from shared memory again.
constexpr uint32_t kRowBytes = kNormCols * 2;  // One bf16 row: 16 KB.

__device__ __forceinline__ uint32_t smem_address(const void *pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrier_init(uint64_t *bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_address(bar)) : "memory");
}

// Order the generic-proxy accesses to shared memory before the next async-proxy accesses.
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t *bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_address(bar)),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void bulk_load(void *dst, const void *src, uint32_t bytes,
                                          uint64_t *bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      ::"r"(smem_address(dst)), "l"(src), "r"(bytes), "r"(smem_address(bar))
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t *bar, uint32_t parity) {
  const uint32_t address = smem_address(bar);
  uint32_t done = 0;
  while (done == 0) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(done)
        : "r"(address), "r"(parity)
        : "memory");
  }
}

// Start the copies of the rows of `row` of up to four tensors (null tensors are not copied).
// The rows go one after the other into the stage.
__device__ __forceinline__ void start_rows(const uint4 *const (&inputs)[4], uint32_t n_inputs,
                                           uint32_t row, uint32_t rows, uint8_t *stage,
                                           uint64_t *bar) {
  if (row >= rows) return;
  mbarrier_expect_tx(bar, n_inputs * kRowBytes);
  const size_t offset = static_cast<size_t>(row) * (kNormCols / 8);
  for (uint32_t i = 0; i < n_inputs; ++i) {
    bulk_load(stage + i * kRowBytes, inputs[i] + offset, kRowBytes, bar);
  }
}

// forward_stats with the TMA row pipeline. Dynamic shared memory: STAGES x 2 rows.
template <int THREADS, int STAGES>
__global__ void __launch_bounds__(THREADS) forward_stats_tma_kernel(const StatsArgs a) {
  constexpr int kVpt = NormShape<THREADS>::kVpt;
  constexpr uint32_t kVpr = kNormCols / 8;
  constexpr uint32_t kStageBytes = 2 * kRowBytes;
  extern __shared__ __align__(128) uint8_t stage_memory[];
  __shared__ uint64_t bars[STAGES];
  __shared__ float shared[THREADS / 32];
  const uint4 *const inputs[4] = {a.residual, a.x, nullptr, nullptr};
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) mbarrier_init(&bars[s]);
    fence_proxy_async();
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      start_rows(inputs, 2, blockIdx.x + s * gridDim.x, a.rows,
                 stage_memory + s * kStageBytes, &bars[s]);
    }
  }
  float g[kVpt][8];
#pragma unroll
  for (int k = 0; k < kVpt; ++k) load_weight(a.w, threadIdx.x + k * THREADS, a.zero_centered, g[k]);
  __syncthreads();
  float amax = 0.0f;
  uint32_t n = 0;
  for (uint32_t row = blockIdx.x; row < a.rows; row += gridDim.x, ++n) {
    const uint32_t s = n % STAGES;
    mbarrier_wait(&bars[s], (n / STAGES) & 1);
    const uint4 *res_row = reinterpret_cast<const uint4 *>(stage_memory + s * kStageBytes);
    const uint4 *x_row = res_row + kVpr;
    float sumsq = 0.0f;
#pragma unroll
    for (int k = 0; k < kVpt; ++k) {
      const uint32_t vec = threadIdx.x + k * THREADS;
      float rv[8];
      float xv[8];
      float hv[8];
      unpack8(res_row[vec], rv);
      unpack8(x_row[vec], xv);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        hv[i] = bf16_round(rv[i] + xv[i]);
        sumsq += hv[i] * hv[i];
      }
      a.h[static_cast<size_t>(row) * kVpr + vec] = pack8(hv);
    }
    sumsq = block_sum<THREADS>(sumsq, shared);
    const float rs = rsqrtf(sumsq / static_cast<float>(kNormCols) + a.eps);
    if (threadIdx.x == 0) a.rstd[row] = rs;
#pragma unroll
    for (int k = 0; k < kVpt; ++k) {
      const uint32_t vec = threadIdx.x + k * THREADS;
      float rv[8];
      float xv[8];
      unpack8(res_row[vec], rv);
      unpack8(x_row[vec], xv);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        amax = fmaxf(amax, fabsf(norm_value(bf16_round(rv[i] + xv[i]), rs, g[k][i])));
      }
    }
    __syncthreads();  // All threads are done with stage s.
    if (threadIdx.x == 0) {
      fence_proxy_async();
      start_rows(inputs, 2, row + STAGES * gridDim.x, a.rows, stage_memory + s * kStageBytes,
                 &bars[s]);
    }
  }
  amax = block_max<THREADS>(amax, shared);
  if (threadIdx.x == 0) a.partial[blockIdx.x] = amax;
}

// backward with the TMA row pipeline. The stage holds the rows of h, dy, dy2 and grad_h (the
// null ones are left out). Dynamic shared memory: STAGES x (number of inputs) rows.
template <int THREADS, int STAGES>
__global__ void __launch_bounds__(THREADS) backward_tma_kernel(const BackwardArgs a) {
  constexpr int kVpt = NormShape<THREADS>::kVpt;
  constexpr uint32_t kVpr = kNormCols / 8;
  extern __shared__ __align__(128) uint8_t stage_memory[];
  __shared__ uint64_t bars[STAGES];
  __shared__ float shared[THREADS / 32];
  // Input order in the stage: h, then dy, dy2, grad_h if not null.
  uint32_t n_inputs = 0;
  const uint4 *inputs[4] = {nullptr, nullptr, nullptr, nullptr};
  inputs[n_inputs++] = a.h;
  const uint32_t dy_slot = n_inputs;
  if (a.dy != nullptr) inputs[n_inputs++] = a.dy;
  const uint32_t dy2_slot = n_inputs;
  if (a.dy2 != nullptr) inputs[n_inputs++] = a.dy2;
  const uint32_t grad_h_slot = n_inputs;
  if (a.grad_h != nullptr) inputs[n_inputs++] = a.grad_h;
  const uint32_t stage_bytes = n_inputs * kRowBytes;
  const uint4 *const fixed_inputs[4] = {inputs[0], inputs[1], inputs[2], inputs[3]};
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) mbarrier_init(&bars[s]);
    fence_proxy_async();
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      start_rows(fixed_inputs, n_inputs, blockIdx.x + s * gridDim.x, a.rows,
                 stage_memory + s * stage_bytes, &bars[s]);
    }
  }
  float w[kVpt][8];
#pragma unroll
  for (int k = 0; k < kVpt; ++k) load_weight(a.w, threadIdx.x + k * THREADS, a.zero_centered, w[k]);
  __syncthreads();
  float amax = 0.0f;
  uint32_t n = 0;
  for (uint32_t row = blockIdx.x; row < a.rows; row += gridDim.x, ++n) {
    const uint32_t s = n % STAGES;
    const float rs = __ldg(a.rstd + row);
    mbarrier_wait(&bars[s], (n / STAGES) & 1);
    const uint4 *stage = reinterpret_cast<const uint4 *>(stage_memory + s * stage_bytes);
    const uint4 *h_row = stage;
    const uint4 *dy_row = stage + dy_slot * kVpr;
    const uint4 *dy2_row = stage + dy2_slot * kVpr;
    const uint4 *gh_row = stage + grad_h_slot * kVpr;
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < kVpt; ++k) {
      const uint32_t vec = threadIdx.x + k * THREADS;
      float hv[8];
      float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      unpack8(h_row[vec], hv);
      if (a.dy != nullptr) unpack8(dy_row[vec], d);
      if (a.dy2 != nullptr) {
        float d2[8];
        unpack8(dy2_row[vec], d2);
#pragma unroll
        for (int i = 0; i < 8; ++i) d[i] += d2[i];
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) dot += (w[k][i] * d[i]) * (hv[i] * rs);
    }
    const float mean = block_sum<THREADS>(dot, shared) / static_cast<float>(kNormCols);
#pragma unroll
    for (int k = 0; k < kVpt; ++k) {
      const uint32_t vec = threadIdx.x + k * THREADS;
      float hv[8];
      float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float res[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      unpack8(h_row[vec], hv);
      if (a.dy != nullptr) unpack8(dy_row[vec], d);
      if (a.dy2 != nullptr) {
        float d2[8];
        unpack8(dy2_row[vec], d2);
#pragma unroll
        for (int i = 0; i < 8; ++i) d[i] += d2[i];
      }
      if (a.grad_h != nullptr) unpack8(gh_row[vec], res);
      float o[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o[i] = rs * ((w[k][i] * d[i]) - (hv[i] * rs) * mean) + res[i];
        amax = fmaxf(amax, fabsf(bf16_round(o[i])));
      }
      a.out[static_cast<size_t>(row) * kVpr + vec] = pack8(o);
    }
    __syncthreads();  // All threads are done with stage s.
    if (threadIdx.x == 0) {
      fence_proxy_async();
      start_rows(fixed_inputs, n_inputs, row + STAGES * gridDim.x, a.rows,
                 stage_memory + s * stage_bytes, &bars[s]);
    }
  }
  amax = block_max<THREADS>(amax, shared);
  if (threadIdx.x == 0) a.partial[blockIdx.x] = amax;
}

// ---------------------------------------------------------------------------------------------
// Host side.

int norm_grid(const void *kernel, const int threads, const int64_t rows, const int smem) {
  static std::map<std::pair<const void *, int>, int> cache;
  static std::map<const void *, int> smem_limit;  // The largest size set for each kernel.
  const auto key = std::make_pair(kernel, smem);
  auto it = cache.find(key);
  if (it == cache.end()) {
    if (smem > smem_limit[kernel]) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem));
      smem_limit[kernel] = smem;
    }
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, smem));
    TORCH_CHECK(blocks > 0, "resnorm: the kernel does not fit on an SM");
    it = cache.emplace(key, blocks).first;
  }
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(rows, sms * it->second)));
}

void check_rows(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
              ": contiguous bf16 CUDA tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, ": 16-byte aligned");
  TORCH_CHECK(t.size(-1) == kNormCols, name, ": last dim must be ", kNormCols);
  TORCH_CHECK(t.numel() / kNormCols % 128 == 0, name, ": rows must be a multiple of 128");
}

void check_weight(const at::Tensor &w) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() &&
                  w.numel() == kNormCols && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "w: contiguous bf16 CUDA tensor with ", kNormCols, " elements");
}

// Launch a kernel on a persistent grid with smem bytes of dynamic shared memory. Args must hold
// a partial pointer field; the function makes the partial tensor and sets the field. Returns it.
template <typename Args>
at::Tensor launch_norm_kernel(const void *kernel, int threads, int smem, int64_t rows,
                              const at::TensorOptions &options, Args &a) {
  const int grid = norm_grid(kernel, threads, rows, smem);
  at::Tensor partial = at::empty({grid}, options.dtype(at::kFloat));
  a.partial = partial.data_ptr<float>();
  void *params[] = {&a};
  C10_CUDA_CHECK(cudaLaunchKernel(kernel, dim3(grid), dim3(threads), params, smem,
                                  at::cuda::getCurrentCUDAStream()));
  return partial;
}

// Returns (h, rstd, partial).
std::vector<at::Tensor> forward_stats(const at::Tensor &residual, const at::Tensor &x,
                                      const at::Tensor &w, double eps, bool zero_centered) {
  check_rows(residual, "residual");
  check_rows(x, "x");
  check_weight(w);
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual and x: same shape");
  c10::cuda::CUDAGuard guard(x.device());
  const int64_t rows = x.numel() / kNormCols;
  at::Tensor h = at::empty_like(x);
  at::Tensor rstd = at::empty({rows}, x.options().dtype(at::kFloat));
  StatsArgs a;
  a.residual = reinterpret_cast<const uint4 *>(residual.data_ptr());
  a.x = reinterpret_cast<const uint4 *>(x.data_ptr());
  a.w = reinterpret_cast<const uint4 *>(w.data_ptr());
  a.h = reinterpret_cast<uint4 *>(h.data_ptr());
  a.rstd = rstd.data_ptr<float>();
  a.rows = static_cast<uint32_t>(rows);
  a.eps = static_cast<float>(eps);
  a.zero_centered = zero_centered;
  constexpr int kStages = 2;
  const int smem = kStages * 2 * static_cast<int>(kRowBytes);
  at::Tensor partial = launch_norm_kernel(
      reinterpret_cast<const void *>(forward_stats_tma_kernel<512, kStages>), 512, smem, rows,
      x.options(), a);
  return {h, rstd, partial};
}

// Returns (data, swizzled scales, amax) of NVFP4(norm(h)), and y = norm(h) in bf16 if with_y.
std::vector<at::Tensor> quantize_norm(const at::Tensor &h, const at::Tensor &rstd,
                                      const at::Tensor &w, const at::Tensor &partial,
                                      bool zero_centered,
    const std::optional<at::Tensor> &alpha_weight_amax,
    const std::optional<at::Tensor> &alpha_out, bool with_y) {
  check_rows(h, "h");
  check_weight(w);
  c10::cuda::CUDAGuard guard(h.device());
  const int64_t rows = h.numel() / kNormCols;
  TORCH_CHECK(rstd.numel() == rows && rstd.scalar_type() == at::kFloat, "rstd: float [rows]");
  auto data_shape = h.sizes().vec();
  data_shape.back() = kNormCols / 2;
  at::Tensor data = at::empty(data_shape, h.options().dtype(at::kByte));
  at::Tensor scales = at::empty({rows, kNormCols / 16}, h.options().dtype(at::kByte));
  at::Tensor amax = at::empty({1}, h.options().dtype(at::kFloat));
  NormQuantizeArgs a;
  a.q.x = reinterpret_cast<const uint4 *>(h.data_ptr());
  a.q.data = reinterpret_cast<uint32_t *>(data.data_ptr());
  a.q.scales = scales.data_ptr<uint8_t>();
  a.q.partial = partial.data_ptr<float>();
  a.q.n_partial = static_cast<int>(partial.numel());
  a.q.amax_out = amax.data_ptr<float>();
  a.q.rows = static_cast<uint32_t>(rows);
  a.q.cols = kNormCols;
  a.q.prefetch = 1;
  const GemmAlpha alpha = gemm_alpha(alpha_weight_amax, alpha_out);
  a.q.alpha_weight_amax = alpha.weight_amax;
  a.q.alpha_out = alpha.out;
  a.rstd = rstd.data_ptr<float>();
  a.w = reinterpret_cast<const uint4 *>(w.data_ptr());
  at::Tensor y = with_y ? at::empty_like(h) : at::Tensor();
  a.y = with_y ? reinterpret_cast<uint4 *>(y.data_ptr()) : nullptr;
  a.zero_centered = zero_centered;
  const void *kernel = zero_centered ? reinterpret_cast<const void *>(quantize_norm_kernel<true>)
                                      : reinterpret_cast<const void *>(quantize_norm_kernel<false>);
  static std::map<const void *, int> occupancy;
  if (occupancy.find(kernel) == occupancy.end()) {
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kThreads, 0));
    occupancy[kernel] = blocks;
  }
  const int blocks_per_sm = occupancy[kernel];
  const int64_t n_units = rows * (kNormCols >> 8);
  const int64_t needed = (n_units + kWarps * kUnroll - 1) / (kWarps * kUnroll);
  const int64_t limit = static_cast<int64_t>(std::max(blocks_per_sm, 1)) *
                        at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = static_cast<int>(std::max<int64_t>(1, std::min(needed, limit)));
  void *params[] = {&a};
  C10_CUDA_CHECK(cudaLaunchKernel(kernel, dim3(grid), dim3(kThreads), params, 0,
                                  at::cuda::getCurrentCUDAStream()));
  if (with_y) return {data, scales, amax, y};
  return {data, scales, amax};
}

const uint4 *optional_rows(const c10::optional<at::Tensor> &t, const char *name) {
  if (!t.has_value()) return nullptr;
  check_rows(*t, name);
  return reinterpret_cast<const uint4 *>(t->data_ptr());
}

// Returns (out, partial). One grad of y: the TMA kernel; two grads: the register kernel with
// 1024 threads and an L2 prefetch of the next row.
std::vector<at::Tensor> backward(const c10::optional<at::Tensor> &grad_h,
                                 const c10::optional<at::Tensor> &dy,
                                 const c10::optional<at::Tensor> &dy2, const at::Tensor &h,
                                 const at::Tensor &rstd, const at::Tensor &w,
                                 bool zero_centered) {
  check_rows(h, "h");
  check_weight(w);
  c10::cuda::CUDAGuard guard(h.device());
  const int64_t rows = h.numel() / kNormCols;
  TORCH_CHECK(rstd.numel() == rows && rstd.scalar_type() == at::kFloat, "rstd: float [rows]");
  at::Tensor out = at::empty_like(h);
  BackwardArgs a;
  a.grad_h = optional_rows(grad_h, "grad_h");
  a.dy = optional_rows(dy, "dy");
  a.dy2 = optional_rows(dy2, "dy2");
  a.h = reinterpret_cast<const uint4 *>(h.data_ptr());
  a.rstd = rstd.data_ptr<float>();
  a.w = reinterpret_cast<const uint4 *>(w.data_ptr());
  a.out = reinterpret_cast<uint4 *>(out.data_ptr());
  a.rows = static_cast<uint32_t>(rows);
  a.zero_centered = zero_centered;
  a.prefetch = 1;
  at::Tensor partial;
  if (a.dy2 == nullptr) {
    constexpr int kStages = 2;
    const int n_inputs = 1 + (a.dy != nullptr) + (a.grad_h != nullptr);
    const int smem = kStages * n_inputs * static_cast<int>(kRowBytes);
    partial = launch_norm_kernel(reinterpret_cast<const void *>(backward_tma_kernel<512, kStages>),
                                 512, smem, rows, h.options(), a);
  } else {
    partial = launch_norm_kernel(reinterpret_cast<const void *>(backward_kernel<1024>), 1024, 0,
                                 rows, h.options(), a);
  }
  return {out, partial};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_stats", &forward_stats, "h = residual + x, rstd, and amax partials of norm(h)",
        py::arg("residual"), py::arg("x"), py::arg("w"), py::arg("eps"),
        py::arg("zero_centered"));
  m.def("quantize_norm", &quantize_norm, "NVFP4 quantize of norm(h) with the amax partials",
        py::arg("h"), py::arg("rstd"), py::arg("w"), py::arg("partial"),
        py::arg("zero_centered"),
        py::arg("alpha_weight_amax") = py::none(), py::arg("alpha_out") = py::none(),
        py::arg("with_y") = false);
  m.def("backward", &backward, "RMSNorm backward + residual grad, with amax partials",
        py::arg("grad_h"), py::arg("dy"), py::arg("dy2"), py::arg("h"), py::arg("rstd"),
        py::arg("w"), py::arg("zero_centered"));
}
