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

// pybind11 binding of the single-sweep fused RMSNorm backward (rmsnorm_bwd_cuda.cu).
// Tensor plumbing only: shape/dtype/alignment checks, raw pointers, the current CUDA stream.  Every buffer is allocated
// by the Python caller (rmsnorm_bwd_cuda.py); this file issues no torch operator.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "rmsnorm_bwd_cuda.h"

namespace {

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("rmsnorm_bwd_cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument("rmsnorm_bwd_cuda: " + msg);
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// rmsnorm_bwd_fused(dy, x, rstd, gamma, dres?, out, part, part2, tickets, dgamma_out, dbias_out?, rd, rs, cd, cs,
//                   zero_centered, fp8_code, accumulate, rows, group, ngroups, ntx_r, ntx_c, nblk_pad, npad_c)
void rmsnorm_bwd_fused(const at::Tensor& dy, const at::Tensor& x, const at::Tensor& rstd, const at::Tensor& gamma,
                       const c10::optional<at::Tensor>& dres, at::Tensor out, at::Tensor part, at::Tensor part2,
                       at::Tensor part3, at::Tensor tickets, at::Tensor dgamma_out, const c10::optional<at::Tensor>& dbias_out,
                       at::Tensor rd, at::Tensor rs, at::Tensor cd, at::Tensor cs, bool zero_centered, int64_t fp8_code,
                       bool accumulate, int64_t rows, int64_t group, int64_t ngroups, int64_t ntx_r, int64_t ntx_c,
                       int64_t nblk_pad, int64_t npad_c, int64_t dbg) {
  require(dy.is_cuda() && dy.dim() == 2 && dy.scalar_type() == at::kBFloat16 && dy.is_contiguous(), "dy must be a contiguous CUDA bf16 [M, N]");
  const int64_t M = dy.size(0), N = dy.size(1);
  require(x.sizes() == dy.sizes() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "x must be bf16 [M, N], contiguous");
  require(rstd.scalar_type() == at::kFloat && rstd.is_contiguous() && rstd.numel() == M, "rstd must be fp32 [M]");
  require(gamma.scalar_type() == at::kBFloat16 && gamma.is_contiguous() && gamma.numel() == N, "gamma must be bf16 [N]");
  require(N % 32 == 0 && N > 0 && N <= pbn::kMaxN, "N must be a multiple of 32 and <= 3072");
  require(M % 32 == 0 && M > 0, "M must be a multiple of 32");
  require(rows == 4 || rows == 8, "rows must be 4 or 8");
  require(N % (32 / rows * 4) == 0, "N must be a multiple of 4 * cluster size");
  const bool has_res = dres.has_value();
  if (has_res) {
    require(dres->sizes() == dy.sizes() && dres->scalar_type() == at::kBFloat16 && dres->is_contiguous(), "dres must be bf16 [M, N], contiguous");
    require(aligned16(dres->data_ptr()), "dres must be 16-B aligned");
  }
  require(out.sizes() == dy.sizes() && out.scalar_type() == at::kBFloat16 && out.is_contiguous(), "out must be bf16 [M, N]");
  const bool mx = rd.numel() > 0;
  const bool has_dbias = dbias_out.has_value();
  const int64_t pstride = has_dbias ? 2 * N : N;
  const int64_t nb32 = M / 32;
  require(part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() == nb32 * pstride, "part must be fp32 [M/32, pstride]");
  require(group >= 1 && ngroups >= 1 && ngroups <= pbn::kMaxGroups && (nb32 + group - 1) / group == ngroups, "group / ngroups inconsistent with M/32");
  require(part2.scalar_type() == at::kFloat && part2.is_contiguous() && part2.numel() == ngroups * pstride, "part2 must be fp32 [ngroups, pstride]");
  const int64_t ngroups2 = (ngroups + pbn::kFan2 - 1) / pbn::kFan2;
  require(part3.scalar_type() == at::kFloat && part3.is_contiguous() && part3.numel() == ngroups2 * pstride, "part3 must be fp32 [ceil(ngroups/8), pstride]");
  require(tickets.scalar_type() == at::kInt && tickets.is_contiguous() && tickets.numel() == pbn::kTicketSlot, "tickets must be int32 [kTicketSlot]");
  require(dgamma_out.scalar_type() == at::kBFloat16 && dgamma_out.is_contiguous() && dgamma_out.numel() == N, "dgamma_out must be bf16 [N]");
  if (has_dbias) {
    require(dbias_out->scalar_type() == at::kBFloat16 && dbias_out->is_contiguous() && dbias_out->numel() == N, "dbias_out must be bf16 [N]");
  }
  if (mx) {
    require(fp8_code == 7 || fp8_code == 8, "fp8_code must be 7 (e4m3) or 8 (e5m2)");
    require(rd.scalar_type() == at::kByte && rd.is_contiguous() && rd.numel() == M * N, "rowwise data must be uint8 [M*N]");
    require(cd.scalar_type() == at::kByte && cd.is_contiguous() && cd.numel() == M * N, "columnwise data must be uint8 [M*N]");
    require(rs.scalar_type() == at::kByte && rs.is_contiguous() && rs.numel() >= ((M + 127) / 128) * 128 * nblk_pad, "rowwise scales too small");
    require(cs.scalar_type() == at::kByte && cs.is_contiguous() && cs.numel() >= ((nb32 + 3) / 4) * 4 * npad_c, "columnwise scales too small");
    require(nblk_pad >= N / 32 && nblk_pad % 4 == 0 && ntx_r == nblk_pad / 4, "nblk_pad / ntx_r inconsistent");
    require(npad_c >= N && npad_c % 128 == 0 && ntx_c == ((nb32 + 3) / 4), "npad_c / ntx_c inconsistent");
    require(aligned16(rd.data_ptr()) && aligned16(cd.data_ptr()), "fp8 data buffers must be 16-B aligned");
  }
  require(aligned16(dy.data_ptr()) && aligned16(x.data_ptr()) && aligned16(gamma.data_ptr()) && aligned16(out.data_ptr())
          && aligned16(part.data_ptr()) && aligned16(part2.data_ptr()) && aligned16(part3.data_ptr()), "dy / x / gamma / out / partials must be 16-B aligned");

  pbn::Params p;
  p.dy = reinterpret_cast<const uint16_t*>(dy.data_ptr());
  p.x = reinterpret_cast<const uint16_t*>(x.data_ptr());
  p.rstd = rstd.data_ptr<float>();
  p.gamma = reinterpret_cast<const uint16_t*>(gamma.data_ptr());
  p.dres = has_res ? reinterpret_cast<const uint16_t*>(dres->data_ptr()) : nullptr;
  p.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  p.part = part.data_ptr<float>();
  p.part2 = part2.data_ptr<float>();
  p.part3 = part3.data_ptr<float>();
  p.tickets = tickets.data_ptr<int32_t>();
  p.dgamma_out = reinterpret_cast<uint16_t*>(dgamma_out.data_ptr());
  p.dbias_out = has_dbias ? reinterpret_cast<uint16_t*>(dbias_out->data_ptr()) : nullptr;
  p.rd = mx ? rd.data_ptr<uint8_t>() : nullptr;
  p.rs = mx ? rs.data_ptr<uint8_t>() : nullptr;
  p.cd = mx ? cd.data_ptr<uint8_t>() : nullptr;
  p.cs = mx ? cs.data_ptr<uint8_t>() : nullptr;
  p.M = (int)M;
  p.N = (int)N;
  p.pstride = (int)pstride;
  p.nb32 = (int)nb32;
  p.group = (int)group;
  p.ngroups = (int)ngroups;
  p.ntx_r = (int)ntx_r;
  p.ntx_c = (int)ntx_c;
  p.nblk_pad = (int)nblk_pad;
  p.npad_c = (int)npad_c;
  p.fp8_max_rcp = (fp8_code == 8) ? (1.0f / 57344.0f) : (1.0f / 448.0f);
  p.zero_centered = zero_centered ? 1 : 0;
  p.accumulate = accumulate ? 1 : 0;
  p.dbg = (int)dbg;
  require(mx == has_dbias, "the MXFP8 variant carries the dbias reduction (and only it)");

  const c10::cuda::CUDAGuard guard(dy.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dy.get_device()).stream();
  check_cuda(pbn::launch(p, (int)rows, has_res, mx, fp8_code == 8, stream), "launch");
}

py::dict device_info() {
  int dev = 0;
  check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
  int sms = 0, major = 0, minor = 0, smem_optin = 0;
  check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "SMs");
  check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc major");
  check_cuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev), "cc minor");
  check_cuda(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev), "smem optin");
  py::dict d;
  d["sms"] = sms;
  d["cc_major"] = major;
  d["cc_minor"] = minor;
  d["smem_optin"] = smem_optin;
  d["smem_rows8_n2880"] = (int64_t)pbn::smem_bytes(8, 2880, false);
  d["smem_rows8_n2880_mx"] = (int64_t)pbn::smem_bytes(8, 2880, true);
  d["smem_rows4_n2880"] = (int64_t)pbn::smem_bytes(4, 2880, false);
  d["smem_rows4_n2880_mx"] = (int64_t)pbn::smem_bytes(4, 2880, true);
  return d;
}

int64_t smem_bytes(int64_t rows, int64_t N, bool mxfp8) { return (int64_t)pbn::smem_bytes((int)rows, (int)N, mxfp8); }

py::dict kernel_attrs(int64_t rows, bool has_res, bool mxfp8, bool e5m2, int64_t N) {
  int num_regs = 0, blocks = 0, dyn = 0, local = 0, clusters = 0;
  check_cuda(pbn::attrs((int)rows, has_res, mxfp8, e5m2, (int)N, &num_regs, &blocks, &dyn, &local, &clusters), "attrs");
  py::dict d;
  d["num_regs"] = num_regs;
  d["blocks_per_sm"] = blocks;
  d["dyn_smem"] = dyn;
  d["local_bytes"] = local;
  d["max_clusters"] = clusters;
  d["threads"] = pbn::kThreads;
  d["cluster"] = (int)(32 / rows);
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "single-sweep fused RMSNorm backward (CUDA C++, TMA bulk slab, cluster DSMEM reduce, in-kernel dgamma/dbias)";
  m.def("rmsnorm_bwd_fused", &rmsnorm_bwd_fused, "fused RMSNorm backward, one sweep, dgamma/dbias finished in-kernel");
  m.def("device_info", &device_info, "SM count, compute capability, shared-memory opt-in");
  m.def("smem_bytes", &smem_bytes, "dynamic shared memory of one CTA for (rows, N, mxfp8)");
  m.def("kernel_attrs", &kernel_attrs, "registers / occupancy / shared memory of one kernel variant");
}
