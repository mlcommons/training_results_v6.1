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

"""NVFP4 quantize that takes the amax partials of a producer kernel from the side channel.

apply() replaces NVFP4Quantizer.quantize_impl. For a plain quantizer and a plain tensor with
partials in the channel, one kernel quantizes from the partials (no amax pass). All other calls
go to the quantize_impl that was in place before (nvfp4q fast path, or TE).
"""

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (
    NVFP4TensorStorage,
)

import amax_channel
import gemm_alpha
import nvfp4q_ap_ext

_state = {"previous": None}


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


def quantize_from_partials(quantizer: NVFP4Quantizer, tensor: torch.Tensor,
                           partials: torch.Tensor) -> object:
    """Quantize a plain tensor with known amax partials."""
    module = nvfp4q_ap_ext.get_extension()
    weight_amax, alpha = gemm_alpha.kernel_tensors(quantizer)
    parts = module.quantize_from_partials(tensor, partials, bool(quantizer.stochastic_rounding),
                                          alpha_weight_amax=weight_amax, alpha_out=alpha)
    gemm_alpha.record(parts[2], quantizer, alpha is not None)
    return _make_output(quantizer, tensor, parts)


def fast_quantize_impl(self: NVFP4Quantizer, tensor: torch.Tensor) -> object:
    """Replacement of NVFP4Quantizer.quantize_impl."""
    if _plain_quantizer(self) and _plain_tensor(tensor):
        partials = amax_channel.take(tensor)
        if partials is not None:
            return quantize_from_partials(self, tensor, partials)
    return _state["previous"](self, tensor)


def apply() -> None:
    """Install the channel-aware quantize_impl in front of the current one."""
    if _state["previous"] is not None:
        return
    nvfp4q_ap_ext.get_extension()
    _state["previous"] = NVFP4Quantizer.quantize_impl
    NVFP4Quantizer.quantize_impl = fast_quantize_impl
