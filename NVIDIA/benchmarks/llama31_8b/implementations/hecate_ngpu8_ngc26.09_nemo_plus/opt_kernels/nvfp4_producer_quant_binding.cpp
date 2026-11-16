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

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <mutex>
#include <optional>

#include "nvfp4_producer_quant.h"

namespace {

const void* ptr_or_null(const std::optional<at::Tensor>& t) {
  return (t.has_value() && t->defined()) ? t->data_ptr() : nullptr;
}

void check_bf16(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name,
              " must be a contiguous CUDA bfloat16 tensor");
}

void check_shapes(int64_t producer, int64_t M, int64_t N) {
  TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128");
  if (producer == pbq::kDSwiGLU) {
    TORCH_CHECK((N / 2) % 128 == 0, "DSwiGLU half width must be a multiple of 128");
  } else {
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128");
  }
  if (producer == pbq::kRMSNorm) TORCH_CHECK(N == 4096, "RMSNorm producer implements hidden 4096");
}

pbq::QuantDesc base_desc(int64_t producer, int64_t M, int64_t N, const at::Tensor& in0,
                         const std::optional<at::Tensor>& in1,
                         const std::optional<at::Tensor>& gamma,
                         double eps,
                         const at::Tensor& partials, const at::Tensor& rht,
                         const at::Tensor& sched) {
  check_shapes(producer, M, N);
  check_bf16(in0, "in0");
  check_bf16(rht, "rht");
  TORCH_CHECK(rht.numel() == 256, "rht must be 16x16");
  TORCH_CHECK(partials.is_cuda() && partials.scalar_type() == at::kFloat, "partials must be fp32");
  TORCH_CHECK(sched.is_cuda() && sched.scalar_type() == at::kInt && sched.numel() >= 4,
              "sched must be an int32 CUDA tensor of 4 counters");
  if (producer == pbq::kDSwiGLU) {
    TORCH_CHECK(in1.has_value(), "DSwiGLU needs the activation input");
    check_bf16(*in1, "in1");
    TORCH_CHECK(in0.numel() == M * (N / 2) && in1->numel() == M * N, "DSwiGLU shape mismatch");
  } else if (producer == pbq::kSwiGLU) {
    TORCH_CHECK(in0.numel() == M * 2 * N, "SwiGLU input must be [M, 2N]");
  } else {
    TORCH_CHECK(in0.numel() == M * N, "input must be [M, N]");
  }
  if (producer == pbq::kRMSNorm) {
    TORCH_CHECK(gamma.has_value(), "RMSNorm needs gamma");
    check_bf16(*gamma, "gamma");
  }
  pbq::QuantDesc d{};
  d.producer = static_cast<int>(producer);
  d.M = static_cast<int>(M);
  d.N = static_cast<int>(N);
  d.in0 = in0.data_ptr();
  d.in1 = ptr_or_null(in1);
  d.gamma = ptr_or_null(gamma);
  d.eps = static_cast<float>(eps);
  d.partials = partials.data_ptr<float>();
  d.num_partials = static_cast<int>(partials.numel() / 2);
  d.rht = rht.data_ptr();
  d.sched = sched.data_ptr<int>();
  return d;
}

int64_t partials_needed(int64_t producer, int64_t M, int64_t N) {
  return pbq::partials_needed(static_cast<int>(producer), static_cast<int>(M),
                              static_cast<int>(N));
}

void stats(int64_t producer, int64_t M, int64_t N, const at::Tensor& in0,
           const std::optional<at::Tensor>& in1, const std::optional<at::Tensor>& gamma,
           double eps, const at::Tensor& partials,
           const at::Tensor& rht, const at::Tensor& sched, bool do_col) {
  const c10::cuda::CUDAGuard guard(in0.device());
  pbq::QuantDesc d = base_desc(producer, M, N, in0, in1, gamma, eps, partials, rht, sched);
  TORCH_CHECK(d.num_partials == pbq::partials_needed(d.producer, d.M, d.N),
              "partials must hold 2 * partials_needed() floats");
  d.do_row = 1;
  d.do_col = do_col ? 1 : 0;
  const cudaError_t err = pbq::launch_stats(d, c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "nvfp4 stats launch failed: ", cudaGetErrorString(err));
}

void cast(int64_t producer, int64_t M, int64_t N, const at::Tensor& in0,
          const std::optional<at::Tensor>& in1, const std::optional<at::Tensor>& gamma,
          const at::Tensor& partials,
          const at::Tensor& rht, const at::Tensor& sched, const std::optional<at::Tensor>& row_data,
          const std::optional<at::Tensor>& row_scale, const std::optional<at::Tensor>& row_amax,
          const std::optional<at::Tensor>& col_data, const std::optional<at::Tensor>& col_scale,
          const std::optional<at::Tensor>& col_amax, bool swizzled, bool stochastic) {
  const c10::cuda::CUDAGuard guard(in0.device());
  pbq::QuantDesc d = base_desc(producer, M, N, in0, in1, gamma, 0.0, partials, rht, sched);
  d.do_row = row_data.has_value() ? 1 : 0;
  d.do_col = col_data.has_value() ? 1 : 0;
  TORCH_CHECK(d.do_row || d.do_col, "nothing to produce");
  if (d.do_row) {
    TORCH_CHECK(row_scale.has_value() && row_amax.has_value(), "row outputs incomplete");
    TORCH_CHECK(row_data->numel() == M * N / 2, "row_data must be [M, N/2] uint8");
    d.row_data = static_cast<uint8_t*>(row_data->data_ptr());
    d.row_scale = static_cast<uint8_t*>(row_scale->data_ptr());
    d.row_amax = row_amax->data_ptr<float>();
  }
  if (d.do_col) {
    TORCH_CHECK(col_scale.has_value() && col_amax.has_value(), "column outputs incomplete");
    TORCH_CHECK(col_data->numel() == M * N / 2, "col_data must be [N, M/2] uint8");
    d.col_data = static_cast<uint8_t*>(col_data->data_ptr());
    d.col_scale = static_cast<uint8_t*>(col_scale->data_ptr());
    d.col_amax = col_amax->data_ptr<float>();
  }
  d.swizzled = swizzled ? 1 : 0;
  d.stochastic = stochastic ? 1 : 0;
  if (stochastic) {
    auto gen = at::cuda::detail::getDefaultCUDAGenerator(in0.device().index());
    auto* impl = at::check_generator<at::CUDAGeneratorImpl>(gen);
    at::PhiloxCudaState st;
    {

      std::lock_guard<std::mutex> lock(gen.mutex());
      st = impl->philox_cuda_state(1024);
    }
    d.philox.captured = st.captured_ ? 1 : 0;
    d.philox.offset_intragraph = st.offset_intragraph_;
    if (st.captured_) {
      d.philox.seed_ptr = st.seed_.ptr;
      d.philox.offset_ptr = st.offset_.ptr;
    } else {
      d.philox.seed = st.seed_.val;
      d.philox.offset = st.offset_.val;
    }
  }
  const cudaError_t err = pbq::launch_cast(d, c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "nvfp4 cast launch failed: ", cudaGetErrorString(err));
}

void rmsnorm_quant(const at::Tensor& x, const at::Tensor& gamma, double eps,
                   const std::optional<at::Tensor>& rstd, const at::Tensor& rht,
                   const at::Tensor& state_f32, const at::Tensor& state_i32, double factor,
                   bool cast, const std::optional<at::Tensor>& row_data,
                   const std::optional<at::Tensor>& row_scale,
                   const std::optional<at::Tensor>& row_amax,
                   const std::optional<at::Tensor>& col_data,
                   const std::optional<at::Tensor>& col_scale,
                   const std::optional<at::Tensor>& col_amax, bool swizzled) {
  const c10::cuda::CUDAGuard guard(x.device());
  check_bf16(x, "x");
  check_bf16(gamma, "gamma");
  check_bf16(rht, "rht");
  const int64_t N = x.size(-1);
  const int64_t M = x.numel() / N;
  TORCH_CHECK(N == 4096 && gamma.numel() == 4096, "RMSNorm producer implements hidden 4096");
  TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128");
  TORCH_CHECK(rht.numel() == 256, "rht must be 16x16");
  TORCH_CHECK(state_f32.is_cuda() && state_f32.scalar_type() == at::kFloat &&
                  state_f32.numel() >= 4 && state_f32.is_contiguous(),
              "state_f32 must be fp32 [4]");
  TORCH_CHECK(state_i32.is_cuda() && state_i32.scalar_type() == at::kInt &&
                  state_i32.numel() >= 2 && state_i32.is_contiguous(),
              "state_i32 must be int32 [2]");
  pbq::RmsQuantDesc d{};
  d.M = static_cast<int>(M);
  d.N = static_cast<int>(N);
  d.x = x.data_ptr();
  d.gamma = gamma.data_ptr();
  d.eps = static_cast<float>(eps);
  d.rht = rht.data_ptr();
  d.state_f32 = state_f32.data_ptr<float>();
  d.state_i32 = state_i32.data_ptr<int>();
  d.factor = static_cast<float>(factor);
  d.cast = cast ? 1 : 0;
  d.swizzled = swizzled ? 1 : 0;
  if (cast) {
    TORCH_CHECK(rstd.has_value() && rstd->scalar_type() == at::kFloat && rstd->numel() == M,
                "rstd must be fp32 [M]");
    d.rstd = rstd->data_ptr<float>();
    d.do_row = row_data.has_value() ? 1 : 0;
    d.do_col = col_data.has_value() ? 1 : 0;
    TORCH_CHECK(d.do_row || d.do_col, "nothing to produce");
    if (d.do_row) {
      TORCH_CHECK(row_scale.has_value() && row_amax.has_value(), "row outputs incomplete");
      TORCH_CHECK(row_data->numel() == M * N / 2, "row_data must be [M, N/2] uint8");
      d.row_data = static_cast<uint8_t*>(row_data->data_ptr());
      d.row_scale = static_cast<uint8_t*>(row_scale->data_ptr());
      d.row_amax = row_amax->data_ptr<float>();
    }
    if (d.do_col) {
      TORCH_CHECK(col_scale.has_value() && col_amax.has_value(), "column outputs incomplete");
      TORCH_CHECK(col_data->numel() == M * N / 2, "col_data must be [N, M/2] uint8");
      d.col_data = static_cast<uint8_t*>(col_data->data_ptr());
      d.col_scale = static_cast<uint8_t*>(col_scale->data_ptr());
      d.col_amax = col_amax->data_ptr<float>();
    }
  }
  const cudaError_t err =
      pbq::launch_rmsnorm_quant(d, c10::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "rmsnorm_quant launch failed: ", cudaGetErrorString(err));
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partials_needed", &partials_needed);
  m.def("stats", &stats);
  m.def("cast", &cast);
  m.def("rmsnorm_quant", &rmsnorm_quant);
}
