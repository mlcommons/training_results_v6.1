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

// pybind11 binding of the MoE gather + MXFP8 quantize CUDA kernels (moe_qperm_cuda.cu).
// Tensor plumbing only: shape/dtype/alignment checks, raw pointers, the current CUDA stream.  Every output buffer is
// allocated by the Python caller (moe_qperm_cuda.py), so this file issues no torch operator.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "moe_qperm_cuda.h"

namespace {

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("moe_qperm_cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument("moe_qperm_cuda: " + msg);
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

int num_sms_for(const at::Tensor& t) {
  static int cached = -1;
  if (cached < 0) {
    int dev = t.get_device();
    int n = 0;
    check_cuda(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute(SMs)");
    cached = n;
  }
  return cached;
}

void fill_common(pbq::Params& p, const at::Tensor& src, const at::Tensor& r2t, const at::Tensor& tpe,
                 const at::Tensor& rw, const at::Tensor& rws, const at::Tensor& cw, const at::Tensor& cws,
                 const at::Tensor& out, int64_t P, int64_t pad, bool write_bf16) {
  require(src.dim() == 2 && src.scalar_type() == at::kBFloat16 && src.is_contiguous() && src.is_cuda(), "src must be a contiguous CUDA bf16 [T, H]");
  const int64_t T = src.size(0), H = src.size(1);
  require(H % 32 == 0 && H > 0, "H must be a multiple of 32");
  require(P % 128 == 0 && P > 0, "P must be a multiple of 128");
  require(pad % 128 == 0 && pad > 0, "pad_multiple must be a multiple of 128 (a 128-row unit lies in one expert)");
  require(r2t.scalar_type() == at::kInt && r2t.is_contiguous() && r2t.numel() == P, "row_to_token must be int32 [P]");
  require(tpe.scalar_type() == at::kInt && tpe.is_contiguous() && tpe.numel() >= 1 && tpe.numel() <= pbq_max_experts(), "tokens_per_expert must be int32 [E], E <= 256");
  require(rw.scalar_type() == at::kByte && rw.is_contiguous() && rw.numel() == P * H, "rowwise data must be uint8 [P*H]");
  require(cw.scalar_type() == at::kByte && cw.is_contiguous() && cw.numel() == P * H, "columnwise data must be uint8 [P*H]");
  const int64_t KT = (H + 127) / 128;
  const int64_t n_scale = (P / 128) * KT * 512;
  require(rws.scalar_type() == at::kByte && rws.is_contiguous() && rws.numel() == n_scale, "rowwise scales must be uint8 [(P/128)*KT*512]");
  require(cws.scalar_type() == at::kByte && cws.is_contiguous() && cws.numel() == n_scale, "columnwise scales must be uint8 [(P/128)*KT*512]");
  if (write_bf16) {
    require(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.dim() == 2 && out.size(0) == P && out.size(1) == H, "out must be bf16 [P, H]");
    require(aligned16(out.data_ptr()), "out must be 16-B aligned");
  }
  require(aligned16(src.data_ptr()) && aligned16(rws.data_ptr()) && aligned16(cws.data_ptr()) && aligned16(rw.data_ptr()) && aligned16(cw.data_ptr()),
          "src / data / scale buffers must be 16-B aligned");
  p.src = reinterpret_cast<const uint16_t*>(src.data_ptr());
  p.r2t = r2t.data_ptr<int32_t>();
  p.tpe = tpe.data_ptr<int32_t>();
  p.pprob = nullptr;
  p.z = nullptr;
  p.rw = rw.data_ptr<uint8_t>();
  p.rws = rws.data_ptr<uint8_t>();
  p.cw = cw.data_ptr<uint8_t>();
  p.cws = cws.data_ptr<uint8_t>();
  p.out = write_bf16 ? reinterpret_cast<uint16_t*>(out.data_ptr()) : nullptr;
  p.dprob_part = nullptr;
  p.dbias = nullptr;
  p.sched = nullptr;
  p.grab = 1;
  p.T = (int)T;
  p.H = (int)H;
  p.P = (int)P;
  p.E = (int)tpe.numel();
  p.PAD = (int)pad;
  p.KT = (int)KT;
  p.K_PAD = (int)(KT * 128);
  p.rcp = 1.0f / 448.0f;
}

// dynamic unit scheduler buffer: sched.numel() == 0 -> the static contiguous partition; else 2 (or more) int32 on the
// source's device, zero at the launch (the kernel's last CTA re-arms it), one in-flight launch per buffer
void fill_sched(pbq::Params& p, const at::Tensor& src, const at::Tensor& sched, int64_t grab) {
  if (sched.numel() == 0) return;
  require(sched.scalar_type() == at::kInt && sched.is_contiguous() && sched.is_cuda() && sched.numel() >= 2 &&
          sched.get_device() == src.get_device(), "sched must be a contiguous int32 CUDA tensor of >= 2 elements on src's device");
  require(grab == 1 || grab == 2, "grab (units per scheduler grab) must be 1 or 2");
  p.sched = sched.data_ptr<int32_t>();
  p.grab = (int)grab;
}

}  // namespace

// forward (and the plain backward): gather + rowwise/columnwise MXFP8 (+ bf16 rows); returns the grid size used
int64_t gather_quant(at::Tensor src, at::Tensor r2t, at::Tensor tpe, at::Tensor rw, at::Tensor rws, at::Tensor cw,
                     at::Tensor cws, at::Tensor out, int64_t P, int64_t pad, bool write_bf16, int64_t load_mode,
                     int64_t nstage, at::Tensor sched, int64_t grab) {
  pbq::Params p{};
  fill_common(p, src, r2t, tpe, rw, rws, cw, cws, out, P, pad, write_bf16);
  fill_sched(p, src, sched, grab);
  const c10::cuda::CUDAGuard guard(src.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  int grid = 0;
  check_cuda(pbq_gather_quant(&p, 0, write_bf16 ? 1 : 0, 0, (int)load_mode, (int)nstage, num_sms_for(src), stream, &grid),
             "gather_quant launch");
  return grid;
}

// combine backward: dz = bf16(prob * dres[tok]) quantized both ways, dprob = dres[tok] . z (partials [KT, P] -> dprob
// [P] by the reduce kernel), optional bf16 dz rows, optional fp32 per-expert column sums of dz (fc2 dbias)
int64_t gather_quant_scaled(at::Tensor src, at::Tensor r2t, at::Tensor pprob, at::Tensor z, at::Tensor tpe,
                            at::Tensor rw, at::Tensor rws, at::Tensor cw, at::Tensor cws, at::Tensor out,
                            at::Tensor dprob_part, at::Tensor dprob, at::Tensor dbias, int64_t P, int64_t pad,
                            bool write_bf16, bool colsum, int64_t load_mode, int64_t nstage, at::Tensor sched,
                            int64_t grab) {
  pbq::Params p{};
  fill_common(p, src, r2t, tpe, rw, rws, cw, cws, out, P, pad, write_bf16);
  fill_sched(p, src, sched, grab);
  const int64_t H = src.size(1);
  require(pprob.scalar_type() == at::kFloat && pprob.is_contiguous() && pprob.numel() == P, "permuted_probs must be fp32 [P]");
  require(z.scalar_type() == at::kBFloat16 && z.is_contiguous() && z.dim() == 2 && z.size(0) == P && z.size(1) == H && aligned16(z.data_ptr()), "z must be a contiguous 16-B aligned bf16 [P, H]");
  require(dprob_part.scalar_type() == at::kFloat && dprob_part.is_contiguous() && dprob_part.numel() == (int64_t)p.KT * P, "dprob_part must be fp32 [KT*P]");
  require(dprob.scalar_type() == at::kFloat && dprob.is_contiguous() && dprob.numel() == P, "dprob must be fp32 [P]");
  if (colsum) {
    require(dbias.scalar_type() == at::kFloat && dbias.is_contiguous() && dbias.dim() == 2 && dbias.size(0) == p.E && dbias.size(1) == H, "dbias must be fp32 [E, H]");
    p.dbias = dbias.data_ptr<float>();
  }
  p.pprob = pprob.data_ptr<float>();
  p.z = reinterpret_cast<const uint16_t*>(z.data_ptr());
  p.dprob_part = dprob_part.data_ptr<float>();
  const c10::cuda::CUDAGuard guard(src.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  int grid = 0;
  check_cuda(pbq_gather_quant(&p, 1, write_bf16 ? 1 : 0, colsum ? 1 : 0, (int)load_mode, (int)nstage, num_sms_for(src),
                              stream, &grid),
             "gather_quant_scaled launch");
  check_cuda(pbq_dprob_reduce(p.dprob_part, dprob.data_ptr<float>(), p.tpe, p.E, p.PAD, p.KT, p.P, stream),
             "dprob_reduce launch");
  return grid;
}

py::dict device_info() {
  int dev = 0;
  check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
  int sms = 0, major = 0, minor = 0, smem_optin = 0;
  check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "attr SMs");
  check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "attr cc major");
  check_cuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev), "attr cc minor");
  check_cuda(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev), "attr smem optin");
  py::dict d;
  d["sms"] = sms;
  d["cc_major"] = major;
  d["cc_minor"] = minor;
  d["smem_optin"] = smem_optin;
  d["smem_fwd_nstage3"] = pbq_smem_bytes(0, 3, 0);
  d["smem_bwd_nstage2"] = pbq_smem_bytes(1, 2, 1);
  d["max_experts"] = pbq_max_experts();
  return d;
}

int64_t smem_bytes(bool scaled, int64_t nstage, bool colsum) {
  return pbq_smem_bytes(scaled ? 1 : 0, (int)nstage, colsum ? 1 : 0);
}

// registers / occupancy / shared memory of one kernel variant on the current device (cudaFuncGetAttributes)
py::dict kernel_attrs(bool scaled, bool write_bf16, bool colsum, int64_t load_mode, int64_t nstage) {
  int a[6] = {0, 0, 0, 0, 0, 0};
  check_cuda(pbq_kernel_attrs(scaled ? 1 : 0, write_bf16 ? 1 : 0, colsum ? 1 : 0, (int)load_mode, (int)nstage, a),
             "kernel_attrs");
  py::dict d;
  d["num_regs"] = a[0];
  d["blocks_per_sm"] = a[1];
  d["dyn_smem"] = a[2];
  d["static_smem"] = a[3];
  d["local_bytes"] = a[4];
  d["max_threads"] = a[5];
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "MoE gather + MXFP8 quantize (CUDA C++, TMA/cp.async staged, warp-shuffle amax, dynamic unit scheduler)";
  m.def("gather_quant", &gather_quant, "gather + MXFP8 rowwise/columnwise quantize");
  m.def("gather_quant_scaled", &gather_quant_scaled, "combine backward: probability-scaled gather + quantize + dprob (+ dz rows, + fc2 dbias column sums)");
  m.def("device_info", &device_info, "device attributes and shared-memory footprints");
  m.def("smem_bytes", &smem_bytes, "dynamic shared memory of a kernel variant");
  m.def("kernel_attrs", &kernel_attrs, "registers, resident CTAs per SM, shared/local memory of a kernel variant");
}
