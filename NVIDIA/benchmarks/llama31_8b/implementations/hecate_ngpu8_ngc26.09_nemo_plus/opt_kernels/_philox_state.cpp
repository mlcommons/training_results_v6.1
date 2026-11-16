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

#include <torch/extension.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cstdint>
#include <mutex>
#include <vector>

std::vector<int64_t> philox_cuda_state(int64_t increment) {
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  at::PhiloxCudaState s = gen->philox_cuda_state(static_cast<uint64_t>(increment));
  if (s.captured_) {
    return {1, static_cast<int64_t>(reinterpret_cast<intptr_t>(s.seed_.ptr)),
            static_cast<int64_t>(reinterpret_cast<intptr_t>(s.offset_.ptr)),
            static_cast<int64_t>(s.offset_intragraph_)};
  }
  return {0, static_cast<int64_t>(s.seed_.val), static_cast<int64_t>(s.offset_.val), 0};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("philox_cuda_state", &philox_cuda_state,
        "Default CUDA generator Philox (seed, offset) advanced by `increment` (TE semantics)");
}
