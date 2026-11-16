// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace pqb {

struct PhiloxArgs {
  uint64_t seed;
  uint64_t offset;
  const int64_t* seed_ptr;
  const int64_t* offset_ptr;
  uint32_t offset_intragraph;
  int captured;
};

cudaError_t launch_rht_amax(const void* x, int64_t M, int64_t N, float* row_amax, float* col_amax,
                            uint16_t sign_mask_t, uint32_t* ws, int64_t* rng_state,
                            const PhiloxArgs& ph, cudaStream_t stream);

cudaError_t launch_amax(const void* x, int64_t n, float* out_a, float* out_b, uint32_t* ws,
                        const float* noop, cudaStream_t stream);

}
