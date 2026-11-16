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

// Flat per-bucket capturable master Adam (CUDA C++).
// One launch per distributed-optimizer bucket shard: bf16 reduced-grad shard (read, scaled by the device clip
// coefficient), bf16 model-param shard (written), fp32 master / exp_avg / exp_avg_sq shards (read + written) as FLAT
// pointers indexed by the same element offset, 16-B vectorised, grid-stride; the parameter-group of every element
// (lr / step device scalars, betas / eps / weight decay / bias correction / mode baked) comes from a sorted segment
// table (gaps = bucket padding, group -1, never touched).  Pure CUDA: no PyTorch dependency in this header or the .cu.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace pfa {

constexpr int kMaxGroups = 8;     // optimizer param groups per launch (the workload has 2: decay / no-decay)
constexpr int kMaxSeg = 1024;     // (param-piece run) segments per bucket shard held in shared memory
constexpr int kVec = 8;           // elements per 16-B bf16 vector (= two float4 of fp32 state)

struct Group {
  const float* lr;        // device scalar (filled before every graph replay)
  const int* step;        // device scalar
  float beta1, beta2, eps, decay;
  int bias_correction, mode;   // mode 0 = L2 (ADAM_MODE_0), 1 = decoupled weight decay (AdamW)
};

struct Groups {
  Group g[kMaxGroups];
  int n;
};

struct Args {
  uint16_t* g;            // bf16 grad shard [numel]  (read; written back scaled only when write_g)
  uint16_t* p;            // bf16 model-param shard [numel] (written)
  float* master;          // fp32 master shard [numel]
  float* m;               // fp32 exp_avg shard [numel]
  float* v;               // fp32 exp_avg_sq shard [numel]
  long long numel;        // multiple of kVec
  const int* noop;        // device flag: 1 -> the launch does nothing (TE's noop_gmem semantics)
  const float* inv_scale; // device scalar: the clip coefficient applied to every grad on load
  const long long* seg_start;  // [nseg + 1] sorted element offsets; seg_start[nseg] == numel
  const int* seg_group;        // [nseg] group index or -1 (padding gap: skipped)
  int nseg;
  int write_g;            // 1: also store the scaled grad back (TE AdamCapturableMasterFunctor does); 0: 28 B/element
};

// grid-stride launch; threads in {128, 256, 512}
cudaError_t launch(const Args& a, const Groups& gr, int grid, int threads, cudaStream_t stream);
// registers / occupancy of the kernel variant
cudaError_t attrs(int threads, int* num_regs, int* blocks_per_sm, int* local_bytes, int* static_smem);

}  // namespace pfa
