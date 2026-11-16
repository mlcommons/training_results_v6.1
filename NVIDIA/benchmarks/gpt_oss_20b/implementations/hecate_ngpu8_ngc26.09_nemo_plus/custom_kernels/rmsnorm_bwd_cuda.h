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

// Single-sweep fused RMSNorm backward (CUDA C++), C interface shared by the kernel file and the binding.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace pbn {

constexpr int kThreads = 384;        // 12 warps; one thread per 8-column group (N <= 3072)
constexpr int kMaxN = kThreads * 8;  // 3072
constexpr int kMaxGroups = 64;       // level-1 groups (16 row blocks each at M = 24576)
constexpr int kFan2 = 8;             // level-1 groups per level-2 group
constexpr int kMaxGroups2 = (kMaxGroups + kFan2 - 1) / kFan2;
constexpr int kTicketSlot = 1 + kMaxGroups + kMaxGroups2;  // int32 per slot: [final, level-1 tickets, level-2 tickets]

struct Params {
  const uint16_t* dy;      // [M, N] bf16
  const uint16_t* x;       // [M, N] bf16
  const float* rstd;       // [M]
  const uint16_t* gamma;   // [N] bf16
  const uint16_t* dres;    // [M, N] bf16 or nullptr
  uint16_t* out;           // [M, N] bf16 = dx (+ dres)
  float* part;             // [M/32, pstride] fp32 partials (dgamma | dbias)
  float* part2;            // [ngroups, pstride] fp32 level-1 group partials
  float* part3;            // [ceil(ngroups/8), pstride] fp32 level-2 partials
  int* tickets;            // [kTicketSlot] int32, zero at rest (re-armed by the kernel)
  uint16_t* dgamma_out;    // [N] bf16 (a main_grad slot or a fresh tensor)
  uint16_t* dbias_out;     // [N] bf16 or nullptr
  uint8_t* rd;             // MXFP8 rowwise data [M, N]
  uint8_t* rs;             // rowwise scales, GEMM-swizzled [round_up(M,128), nblk_pad]
  uint8_t* cd;             // columnwise data [M, N]
  uint8_t* cs;             // columnwise scales, GEMM-swizzled [round_up(M/32,4), npad_c]
  int M, N, pstride, nb32, group, ngroups, ntx_r, ntx_c, nblk_pad, npad_c;
  float fp8_max_rcp;
  int zero_centered;
  int accumulate;
  int dbg;                 // development knob (0 in production): bit0 skip reduce+tickets, bit1 skip sweep 3, bit2 skip the
                           // TMA load, bit3 skip the out stores, bit4 no cluster exchange (own arrays)
};

// dynamic shared memory of one CTA
size_t smem_bytes(int rows, int N, bool mxfp8);
// launch (rows in {4, 8}); returns a cudaError_t
cudaError_t launch(const Params& p, int rows, bool has_res, bool mxfp8, bool e5m2, cudaStream_t stream);
// cudaFuncGetAttributes + occupancy of one instantiation (num_regs, blocks_per_sm, dyn_smem, local_bytes, max_clusters)
cudaError_t attrs(int rows, bool has_res, bool mxfp8, bool e5m2, int N, int* num_regs, int* blocks_per_sm,
                  int* dyn_smem, int* local_bytes, int* max_clusters);

}  // namespace pbn
