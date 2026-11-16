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

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/hadamard_transform.h>
#include <transformer_engine/transformer_engine.h>

#include <mutex>
#include <optional>
#include <vector>

#include "quant_bookkeeping.h"

namespace {

using transformer_engine::DType;
using transformer_engine::QuantizationConfigWrapper;
using transformer_engine::TensorWrapper;

std::vector<size_t> shape_of(const at::Tensor& t) {
  return std::vector<size_t>(t.sizes().begin(), t.sizes().end());
}

void check_u8(const std::optional<at::Tensor>& t, const char* name) {
  TORCH_CHECK(t.has_value() && t->is_cuda() && t->scalar_type() == at::kByte &&
                  t->is_contiguous(),
              name, " must be a contiguous CUDA uint8 tensor");
}

void check_f32_scalar(const std::optional<at::Tensor>& t, const char* name) {
  TORCH_CHECK(t.has_value() && t->is_cuda() && t->scalar_type() == at::kFloat && t->numel() == 1,
              name, " must be a CUDA float32 scalar tensor");
}

TensorWrapper make_nvfp4_out(const at::Tensor& x, int64_t M, int64_t N,
                             const std::optional<at::Tensor>& row_data,
                             const std::optional<at::Tensor>& row_scale,
                             const std::optional<at::Tensor>& row_amax,
                             const std::optional<at::Tensor>& col_data,
                             const std::optional<at::Tensor>& col_scale,
                             const std::optional<at::Tensor>& col_amax, bool swizzled) {
  TensorWrapper out(NVTE_NVFP4_1D_SCALING);
  if (row_data.has_value()) {
    check_u8(row_data, "row_data");
    check_u8(row_scale, "row_scale");
    check_f32_scalar(row_amax, "row_amax");
    TORCH_CHECK(row_data->numel() * 2 == M * N, "row_data must hold M * N / 2 bytes");
    out.set_rowwise_data(row_data->data_ptr(), DType::kFloat4E2M1, shape_of(x));
    out.set_rowwise_scale_inv(row_scale->data_ptr(), DType::kFloat8E4M3, shape_of(*row_scale));
    out.set_amax(row_amax->data_ptr(), DType::kFloat32, shape_of(*row_amax));
  }
  if (col_data.has_value()) {
    check_u8(col_data, "col_data");
    check_u8(col_scale, "col_scale");
    check_f32_scalar(col_amax, "col_amax");
    TORCH_CHECK(col_data->numel() * 2 == M * N, "col_data must hold M * N / 2 bytes");
    out.set_columnwise_data(col_data->data_ptr(), DType::kFloat4E2M1,
                            std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(M)});
    out.set_columnwise_scale_inv(col_scale->data_ptr(), DType::kFloat8E4M3,
                                 shape_of(*col_scale));
    out.set_columnwise_amax(col_amax->data_ptr(), DType::kFloat32, shape_of(*col_amax));
  }
  out.set_with_gemm_swizzled_scales(swizzled);
  out.set_row_scaled_nvfp4(false);
  out.set_nvfp4_e4m3_max(448);
  return out;
}

void check_input(const at::Tensor& x, const at::Tensor& ws, int64_t& M, int64_t& N) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() >= 2,
              "x must be a contiguous CUDA bfloat16 tensor with at least 2 dims");
  TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == at::kInt && ws.numel() >= 4,
              "ws must be an int32 CUDA tensor of 4 counters");
  N = x.size(-1);
  M = x.numel() / N;
}

void rht_quantize(const at::Tensor& x, const at::Tensor& rht, int64_t sign_mask_t,
                  const std::optional<at::Tensor>& row_data,
                  const std::optional<at::Tensor>& row_scale,
                  const std::optional<at::Tensor>& row_amax,
                  const std::optional<at::Tensor>& col_data,
                  const std::optional<at::Tensor>& col_scale,
                  const std::optional<at::Tensor>& col_amax, bool swizzled, bool stochastic,
                  const std::optional<at::Tensor>& rng_state, const at::Tensor& ws) {
  const c10::cuda::CUDAGuard guard(x.device());
  int64_t M, N;
  check_input(x, ws, M, N);
  TORCH_CHECK(rht.is_cuda() && rht.scalar_type() == at::kBFloat16 && rht.numel() == 256 &&
                  rht.is_contiguous(),
              "rht must be a contiguous 16x16 bfloat16 CUDA tensor");
  TORCH_CHECK(row_data.has_value() || col_data.has_value(), "nothing to produce");
  TensorWrapper out = make_nvfp4_out(x, M, N, row_data, row_scale, row_amax, col_data, col_scale,
                                     col_amax, swizzled);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  pqb::PhiloxArgs ph{};
  int64_t* rng_ptr = nullptr;
  if (stochastic) {
    TORCH_CHECK(rng_state.has_value() && rng_state->is_cuda() &&
                    rng_state->scalar_type() == at::kLong && rng_state->numel() == 2,
                "stochastic rounding needs an int64[2] CUDA rng_state");
    rng_ptr = rng_state->data_ptr<int64_t>();

    auto gen = at::cuda::detail::getDefaultCUDAGenerator(x.device().index());
    auto* impl = at::check_generator<at::CUDAGeneratorImpl>(gen);
    at::PhiloxCudaState st;
    {
      std::lock_guard<std::mutex> lock(gen.mutex());
      st = impl->philox_cuda_state(1024);
    }
    ph.captured = st.captured_ ? 1 : 0;
    ph.offset_intragraph = st.offset_intragraph_;
    if (st.captured_) {
      ph.seed_ptr = st.seed_.ptr;
      ph.offset_ptr = st.offset_.ptr;
    } else {
      ph.seed = st.seed_.val;
      ph.offset = st.offset_.val;
    }
  }

  cudaError_t err = pqb::launch_rht_amax(
      x.data_ptr(), M, N, row_data.has_value() ? row_amax->data_ptr<float>() : nullptr,
      col_data.has_value() ? col_amax->data_ptr<float>() : nullptr,
      static_cast<uint16_t>(sign_mask_t), reinterpret_cast<uint32_t*>(ws.data_ptr<int>()),
      rng_ptr, ph, stream);
  TORCH_CHECK(err == cudaSuccess, "rht amax launch failed: ", cudaGetErrorString(err));

  TensorWrapper in(x.data_ptr(), shape_of(x), DType::kBFloat16);
  TensorWrapper had(rht.data_ptr(), std::vector<size_t>{16, 16}, DType::kBFloat16);
  QuantizationConfigWrapper cfg;
  cfg.set_nvfp4_2d_quantization(false);
  cfg.set_stochastic_rounding(stochastic);
  cfg.set_nvfp4_4over6_mode(kNVTENVFP44Over6Disabled);
  TensorWrapper te_rng;
  if (stochastic) {
    te_rng = TensorWrapper(rng_ptr, std::vector<size_t>{2}, DType::kInt64);
    cfg.set_rng_state(te_rng.data());
  }
  nvte_quantize_with_hadamard_transform(in.data(), out.data(), had.data(), cfg, stream);
}

void quantize_2d(const at::Tensor& x, const std::optional<at::Tensor>& row_data,
                 const std::optional<at::Tensor>& row_scale,
                 const std::optional<at::Tensor>& row_amax,
                 const std::optional<at::Tensor>& col_data,
                 const std::optional<at::Tensor>& col_scale,
                 const std::optional<at::Tensor>& col_amax, bool swizzled,
                 const std::optional<at::Tensor>& noop, const at::Tensor& ws) {
  const c10::cuda::CUDAGuard guard(x.device());
  int64_t M, N;
  check_input(x, ws, M, N);
  TORCH_CHECK(row_data.has_value() || col_data.has_value(), "nothing to produce");
  TensorWrapper out = make_nvfp4_out(x, M, N, row_data, row_scale, row_amax, col_data, col_scale,
                                     col_amax, swizzled);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaError_t err = pqb::launch_amax(
      x.data_ptr(), x.numel(), row_data.has_value() ? row_amax->data_ptr<float>() : nullptr,
      col_data.has_value() ? col_amax->data_ptr<float>() : nullptr,
      reinterpret_cast<uint32_t*>(ws.data_ptr<int>()),
      noop.has_value() ? noop->data_ptr<float>() : nullptr, stream);
  TORCH_CHECK(err == cudaSuccess, "amax launch failed: ", cudaGetErrorString(err));

  TensorWrapper in(x.data_ptr(), shape_of(x), DType::kBFloat16);
  QuantizationConfigWrapper cfg;
  cfg.set_nvfp4_2d_quantization(true);
  cfg.set_stochastic_rounding(false);
  cfg.set_nvfp4_4over6_mode(kNVTENVFP44Over6Disabled);
  TensorWrapper te_noop;
  if (noop.has_value()) {
    TORCH_CHECK(noop->is_cuda() && noop->scalar_type() == at::kFloat && noop->numel() == 1,
                "noop must be a CUDA float32 scalar tensor");
    te_noop = TensorWrapper(noop->data_ptr(), std::vector<size_t>{1}, DType::kFloat32);
    cfg.set_noop_tensor(te_noop.data());
  }
  nvte_quantize_v2(in.data(), out.data(), cfg, stream);
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rht_quantize", &rht_quantize);
  m.def("quantize_2d", &quantize_2d);
}
