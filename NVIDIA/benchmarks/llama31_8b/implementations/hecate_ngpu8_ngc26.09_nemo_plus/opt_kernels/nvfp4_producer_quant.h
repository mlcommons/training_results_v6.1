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

namespace pbq {

enum ProducerKind { kIdent = 0, kRMSNorm = 1, kSwiGLU = 2, kDSwiGLU = 3 };

struct PhiloxArgs {
  uint64_t seed;
  uint64_t offset;
  const int64_t* seed_ptr;
  const int64_t* offset_ptr;
  uint32_t offset_intragraph;
  int captured;
};

struct QuantDesc {
  int producer;
  int M, N;
  const void* in0;
  const void* in1;
  const void* gamma;
  float eps;
  float* partials;
  int num_partials;
  const void* rht;
  uint8_t* row_data;
  uint8_t* row_scale;
  float* row_amax;
  uint8_t* col_data;
  uint8_t* col_scale;
  float* col_amax;
  int do_row, do_col, swizzled, stochastic;
  PhiloxArgs philox;
  int* sched;
};

struct RmsQuantDesc {
  int M, N;
  const void* x;
  const void* gamma;
  float* rstd;
  float eps;
  const void* rht;
  float* state_f32;
  int* state_i32;
  float factor;
  int cast, do_row, do_col, swizzled;
  uint8_t* row_data;
  uint8_t* row_scale;
  float* row_amax;
  uint8_t* col_data;
  uint8_t* col_scale;
  float* col_amax;
};

int partials_needed(int producer, int M, int N);
cudaError_t launch_rmsnorm_quant(const RmsQuantDesc& d, cudaStream_t stream);
cudaError_t launch_stats(const QuantDesc& d, cudaStream_t stream);
cudaError_t launch_cast(const QuantDesc& d, cudaStream_t stream);

}
