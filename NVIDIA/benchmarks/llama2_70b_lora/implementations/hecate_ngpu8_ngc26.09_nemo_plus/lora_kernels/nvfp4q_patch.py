# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fast NVFP4 quantize of plain bf16 activations, with the same result as TE.

apply() replaces NVFP4Quantizer.quantize_impl and the rmsnorm_fwd function of te.ops RMSNorm.
Plain cases go to the nvfp4q kernels (amax and quantize, no memset). The output keeps the
scales in the GEMM swizzled layout, so the GEMM does not run a swizzle kernel.
All other cases go to TE.

Do not apply this module together with fast_quant.py (v1): both replace quantize_impl.
"""

import os

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine.pytorch.ops.basic.rmsnorm as te_ops_rmsnorm
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage

import gemm_alpha
import nvfp4q_ext

_AMAX_SLOTS = 16
_state = {
    "quantize_impl": None,
    "rmsnorm_fwd": None,
    "amax_pool": None,
    "amax_slots": {},
}


def _env_flag(name: str) -> bool | None:
    """Return the TE value of a bool env variable, or None if TE does not accept the value."""
    value = os.environ.get(name, "")
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    return None


def _plain_quantizer(quantizer: NVFP4Quantizer) -> bool:
    """Return True for a 1D, rowwise-only quantizer with no RHT, 4over6 or amax reduction."""
    special = quantizer.with_rht or quantizer.with_amax_reduction or quantizer.row_scaled_nvfp4
    special = special or quantizer.nvfp4_use_4over6 or quantizer.with_2d_quantization
    usage = quantizer.rowwise_usage and not quantizer.columnwise_usage
    return usage and not special and quantizer.dtype == tex.DType.kFloat4E2M1


def _plain_tensor(tensor: torch.Tensor) -> bool:
    """Return True for a contiguous CUDA bf16 tensor that the kernels support."""
    if type(tensor) is not torch.Tensor or not tensor.is_cuda or tensor.dtype != torch.bfloat16:
        return False
    if not tensor.is_contiguous() or tensor.dim() < 2 or tensor.data_ptr() % 16 != 0:
        return False
    cols = tensor.shape[-1]
    if cols == 0 or cols % 256 != 0 or tensor.numel() >= 2**31:
        return False
    rows = tensor.numel() // cols
    return rows > 0 and rows % 128 == 0


def _make_output(quantizer: NVFP4Quantizer, tensor: torch.Tensor, parts: list) -> object:
    """Wrap (data, scales, amax) in the same Python class that TE makes."""
    data, scales, amax = parts
    kwargs = {
        "rowwise_data": data,
        "rowwise_scale_inv": scales,
        "columnwise_data": None,
        "columnwise_scale_inv": None,
        "amax_rowwise": amax,
        "amax_columnwise": None,
        "fp4_dtype": quantizer.dtype,
        "quantizer": quantizer,
        "with_gemm_swizzled_scales": True,
        "row_scaled_nvfp4": False,
        "nvfp4_use_4over6": False,
        "nvfp4_e4m3_max": 448,
    }
    if quantizer.internal:
        return NVFP4TensorStorage(fake_dtype=tensor.dtype, **kwargs)
    return NVFP4Tensor(shape=tensor.shape, dtype=tensor.dtype, device=tensor.device, **kwargs)


def fast_quantize(quantizer: NVFP4Quantizer, tensor: torch.Tensor) -> object:
    """Quantize a plain tensor with the nvfp4q kernels."""
    module = nvfp4q_ext.get_extension()
    weight_amax, alpha = gemm_alpha.kernel_tensors(quantizer)
    parts = module.quantize(tensor, bool(quantizer.stochastic_rounding),
                            alpha_weight_amax=weight_amax, alpha_out=alpha)
    gemm_alpha.record(parts[2], quantizer, alpha is not None)
    return _make_output(quantizer, tensor, parts)


def fast_quantize_impl(self: NVFP4Quantizer, tensor: torch.Tensor) -> object:
    """Replacement of NVFP4Quantizer.quantize_impl."""
    if _plain_quantizer(self) and _plain_tensor(tensor):
        return fast_quantize(self, tensor)
    return _state["quantize_impl"](self, tensor)


def fast_rmsnorm_fwd(*args: object) -> tuple:
    """Replacement of rmsnorm_fwd in te.ops RMSNorm, with the signature of tex.rmsnorm_fwd.

    With NVTE_NORM_FWD_USE_CUDNN=1, TE makes the norm in bf16 and then quantizes it with
    compute_amax. Here the same norm call makes the bf16 output, and nvfp4q quantizes it.
    """
    x, weight, eps, out, quantizer, out_dtype, sm_margin, zero_centered_gamma = args
    plain = isinstance(quantizer, NVFP4Quantizer) and _plain_quantizer(quantizer)
    if out is not None or not plain or out_dtype != tex.DType.kBFloat16 or not _plain_tensor(x):
        return _state["rmsnorm_fwd"](*args)
    norm_args = (x, weight, eps, None, None, out_dtype, sm_margin, zero_centered_gamma)
    y, _, rstdevs = _state["rmsnorm_fwd"](*norm_args)
    if not _plain_tensor(y):
        return _state["quantize_impl"](quantizer, y), None, rstdevs
    return fast_quantize(quantizer, y), None, rstdevs


def _amax_pool_slot(device: torch.device) -> tuple:
    """Return the (partial, counter) buffers of the current stream."""
    if _state["amax_pool"] is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Call amax_into one time before CUDA graph capture.")
        size = nvfp4q_ext.get_extension().amax_partial_size()
        partial = torch.empty(_AMAX_SLOTS, size, dtype=torch.float32, device=device)
        counter = torch.zeros(_AMAX_SLOTS, dtype=torch.int32, device=device)
        _state["amax_pool"] = (partial, counter)
    partial, counter = _state["amax_pool"]
    key = torch.cuda.current_stream(device).cuda_stream
    slot = _state["amax_slots"].setdefault(key, len(_state["amax_slots"]))
    if slot >= _AMAX_SLOTS:
        raise RuntimeError("Too many streams for the amax pool.")
    return partial[slot], counter[slot : slot + 1]


def amax_into(tensor: torch.Tensor, out: torch.Tensor) -> None:
    """Write max |tensor| into each element of out (1 or 2 floats), with no memset."""
    partial, counter = _amax_pool_slot(tensor.device)
    nvfp4q_ext.get_extension().amax_into(tensor, out, partial, counter)


def apply() -> None:
    """Replace the TE quantize_impl, and rmsnorm_fwd of te.ops RMSNorm if TE uses cuDNN norm."""
    if _state["quantize_impl"] is not None:
        return
    if _env_flag("NVTE_USE_FAST_MATH") is not False:
        return
    _state["quantize_impl"] = NVFP4Quantizer.quantize_impl
    NVFP4Quantizer.quantize_impl = fast_quantize_impl
    if _env_flag("NVTE_NORM_FWD_USE_CUDNN"):
        _state["rmsnorm_fwd"] = te_ops_rmsnorm.rmsnorm_fwd
        te_ops_rmsnorm.rmsnorm_fwd = fast_rmsnorm_fwd
