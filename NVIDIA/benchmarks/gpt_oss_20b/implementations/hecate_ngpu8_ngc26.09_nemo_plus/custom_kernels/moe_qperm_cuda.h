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

// MoE gather + MXFP8 quantize CUDA kernels -- host/device shared parameter block (no torch dependency).
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace pbq {
struct Params {
  const uint16_t* src;      // [T, H] bf16 rows being permuted (norm output / combined-output gradient)
  const int32_t* r2t;       // [P] source token of permuted row p, -1 = padding
  const int32_t* tpe;       // [E] raw tokens per expert
  const float* pprob;       // [P] routing probability of row p (SCALED)
  const uint16_t* z;        // [P, H] bf16 unscaled expert output (SCALED)
  uint8_t* rw;              // [P * H] e4m3 rowwise
  uint8_t* rws;             // swizzled e8m0, one global tile grid
  uint8_t* cw;              // [P * H] e4m3 columnwise
  uint8_t* cws;             // e8m0, TE grouped storage (one swizzled block per expert)
  uint16_t* out;            // [P, H] bf16 rows (WRITE_BF16)
  float* dprob_part;        // [KT, P] fp32 partial dot products (SCALED)
  float* dbias;             // [E, H] fp32 column sums (COLSUM)
  int* sched;               // dynamic unit scheduler: 2 ints {next slot, finished CTAs}, zero before the launch, reset to
                            // zero by the last CTA (one launch at a time per buffer); nullptr = static contiguous partition
  int grab;                 // units per scheduler grab: 1, or 2 = the (kt, half) pair of one 256-row chunk
  int T, H, P, E, PAD, KT, K_PAD;
  float rcp;                // 1 / 448
};
}  // namespace pbq

extern "C" int pbq_smem_bytes(int scaled, int nstage, int colsum);
extern "C" int pbq_max_experts();
extern "C" cudaError_t pbq_gather_quant(const pbq::Params* p, int scaled, int write_bf16, int colsum, int load_mode,
                                        int nstage, int num_sms, cudaStream_t stream, int* grid_used);
// attribute query of one kernel variant (no launch): attrs = {regs/thread, resident CTAs/SM at its dynamic smem,
// dynamic smem bytes, static smem bytes, local bytes/thread, max threads/CTA}
extern "C" cudaError_t pbq_kernel_attrs(int scaled, int write_bf16, int colsum, int load_mode, int nstage, int attrs[6]);
extern "C" cudaError_t pbq_dprob_reduce(const float* part, float* dprob, const int32_t* tpe, int E, int PAD, int KT,
                                        int P, cudaStream_t stream);
