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

// pybind11 binding of the flat per-bucket capturable master Adam (flat_adam_cuda.cu).
// Tensor plumbing only: dtype / contiguity / alignment checks, raw pointers, the current CUDA stream.  Every buffer
// (the shards, the device scalars, the segment table) is owned by the Python caller; this file issues no torch operator.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "flat_adam_cuda.h"

namespace {

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("flat_adam_cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument("flat_adam_cuda: " + msg);
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// flat_adam(g, p, master, m, v, noop, inv_scale, seg_start, seg_group, lrs, steps, beta1, beta2, eps, decay,
//           bias_correction, mode, grid, threads, write_g)
void flat_adam(at::Tensor g, at::Tensor p, at::Tensor master, at::Tensor m, at::Tensor v, const at::Tensor& noop,
               const at::Tensor& inv_scale, const at::Tensor& seg_start, const at::Tensor& seg_group,
               const std::vector<at::Tensor>& lrs, const std::vector<at::Tensor>& steps,
               const std::vector<double>& beta1, const std::vector<double>& beta2, const std::vector<double>& eps,
               const std::vector<double>& decay, const std::vector<int64_t>& bias_correction,
               const std::vector<int64_t>& mode, int64_t grid, int64_t threads, bool write_g) {
  require(g.is_cuda() && g.dim() == 1 && g.scalar_type() == at::kBFloat16 && g.is_contiguous(), "g must be a contiguous CUDA bf16 [n]");
  const int64_t n = g.numel();
  require(n > 0 && n % pfa::kVec == 0, "numel must be a positive multiple of 8");
  require(p.is_cuda() && p.dim() == 1 && p.scalar_type() == at::kBFloat16 && p.is_contiguous() && p.numel() == n, "p must be bf16 [n], contiguous");
  for (const auto* t : {&master, &m, &v}) {
    require(t->is_cuda() && t->dim() == 1 && t->scalar_type() == at::kFloat && t->is_contiguous() && t->numel() == n, "master / m / v must be fp32 [n], contiguous");
  }
  require(noop.is_cuda() && noop.scalar_type() == at::kInt && noop.numel() == 1, "noop must be int32 [1]");
  require(inv_scale.is_cuda() && inv_scale.scalar_type() == at::kFloat && inv_scale.numel() == 1, "inv_scale must be fp32 [1]");
  require(seg_start.is_cuda() && seg_start.scalar_type() == at::kLong && seg_start.is_contiguous(), "seg_start must be int64 contiguous");
  require(seg_group.is_cuda() && seg_group.scalar_type() == at::kInt && seg_group.is_contiguous(), "seg_group must be int32 contiguous");
  const int64_t nseg = seg_group.numel();
  require(nseg >= 1 && nseg <= pfa::kMaxSeg && seg_start.numel() == nseg + 1, "segment table: 1 <= nseg <= kMaxSeg and seg_start has nseg + 1 entries");
  const size_t ng = lrs.size();
  require(ng >= 1 && ng <= static_cast<size_t>(pfa::kMaxGroups), "1 <= groups <= kMaxGroups");
  require(steps.size() == ng && beta1.size() == ng && beta2.size() == ng && eps.size() == ng && decay.size() == ng
          && bias_correction.size() == ng && mode.size() == ng, "every per-group list must have the same length");
  require(aligned16(g.data_ptr()) && aligned16(p.data_ptr()) && aligned16(master.data_ptr()) && aligned16(m.data_ptr()) && aligned16(v.data_ptr()),
          "g / p / master / m / v must be 16-B aligned");
  require(threads == 128 || threads == 256 || threads == 512, "threads must be 128, 256 or 512");
  require(grid >= 1 && grid <= 0x7fffffff, "grid must be >= 1");

  pfa::Args a;
  a.g = reinterpret_cast<uint16_t*>(g.data_ptr());
  a.p = reinterpret_cast<uint16_t*>(p.data_ptr());
  a.master = master.data_ptr<float>();
  a.m = m.data_ptr<float>();
  a.v = v.data_ptr<float>();
  a.numel = n;
  a.noop = noop.data_ptr<int32_t>();
  a.inv_scale = inv_scale.data_ptr<float>();
  a.seg_start = reinterpret_cast<const long long*>(seg_start.data_ptr<int64_t>());
  a.seg_group = seg_group.data_ptr<int32_t>();
  a.nseg = static_cast<int>(nseg);
  a.write_g = write_g ? 1 : 0;

  pfa::Groups gr;
  gr.n = static_cast<int>(ng);
  for (size_t i = 0; i < ng; ++i) {
    require(lrs[i].is_cuda() && lrs[i].scalar_type() == at::kFloat && lrs[i].numel() == 1, "lr[i] must be fp32 [1] on the device");
    require(steps[i].is_cuda() && steps[i].scalar_type() == at::kInt && steps[i].numel() == 1, "step[i] must be int32 [1] on the device");
    require(mode[i] == 0 || mode[i] == 1, "mode must be 0 (L2) or 1 (AdamW)");
    gr.g[i].lr = lrs[i].data_ptr<float>();
    gr.g[i].step = steps[i].data_ptr<int32_t>();
    gr.g[i].beta1 = static_cast<float>(beta1[i]);
    gr.g[i].beta2 = static_cast<float>(beta2[i]);
    gr.g[i].eps = static_cast<float>(eps[i]);
    gr.g[i].decay = static_cast<float>(decay[i]);
    gr.g[i].bias_correction = static_cast<int>(bias_correction[i]);
    gr.g[i].mode = static_cast<int>(mode[i]);
  }
  for (size_t i = ng; i < static_cast<size_t>(pfa::kMaxGroups); ++i) {
    gr.g[i] = pfa::Group{nullptr, nullptr, 0.f, 0.f, 0.f, 0.f, 0, 0};
  }

  const c10::cuda::CUDAGuard guard(g.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(g.get_device()).stream();
  check_cuda(pfa::launch(a, gr, static_cast<int>(grid), static_cast<int>(threads), stream), "launch");
}

py::dict device_info() {
  int dev = 0;
  check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
  int sms = 0, major = 0, minor = 0;
  check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SMs");
  check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc major");
  check_cuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev), "cc minor");
  py::dict d;
  d["sms"] = sms;
  d["cc_major"] = major;
  d["cc_minor"] = minor;
  d["max_groups"] = pfa::kMaxGroups;
  d["max_seg"] = pfa::kMaxSeg;
  d["vec"] = pfa::kVec;
  return d;
}

py::dict kernel_attrs(int64_t threads) {
  int num_regs = 0, blocks = 0, local = 0, smem = 0;
  check_cuda(pfa::attrs(static_cast<int>(threads), &num_regs, &blocks, &local, &smem), "attrs");
  py::dict d;
  d["num_regs"] = num_regs;
  d["blocks_per_sm"] = blocks;
  d["local_bytes"] = local;
  d["static_smem"] = smem;
  d["threads"] = threads;
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "flat per-bucket capturable master Adam (CUDA C++, 16-B vectorised, device-scalar lr/step/clip)";
  m.def("flat_adam", &flat_adam, "one Adam launch over a bucket shard (bf16 g/p, fp32 master/m/v, segment table of param groups)");
  m.def("device_info", &device_info, "SM count, compute capability, table limits");
  m.def("kernel_attrs", &kernel_attrs, "registers / occupancy / shared memory of one thread-count variant");
}
