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

"""Two-pass SwiGLU backward with the NVFP4 quantize in the te.ops SwiGLU op (same bits as TE).

apply() wraps the SwiGLU._tex_swiglu_backward function that is in place at the call.
Apply it after the other SwiGLU patches, for example after lora_fast_patches.apply().
Then the cases that this path does not support go to the other patches.

For a plain NVFP4 quantizer and supported shapes, the new backward runs two kernels:
1. Pass 1 computes only the amax of dx.
2. Pass 2 computes dx again and quantizes it directly to FP4 data and swizzled scales.

The bf16 dx tensor does not exist.
make_output(quantizer, like, [data, scales, amax]) wraps the parts in the TE tensor class.
The like tensor has the shape, dtype and device of dx.
"""

import os

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine_torch as tex
from transformer_engine.pytorch.ops.basic.swiglu import SwiGLU
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage

import gemm_alpha
import swiglu_sr_ext

ENV_FLAG = "LORA_SWIGLU_SR_TWO_PASS"
_PASS1_VARIANT = int(os.environ.get("SWIGLU_SR_PASS1_VARIANT", "2"))
_PASS2_VARIANT = int(os.environ.get("SWIGLU_SR_PASS2_VARIANT", "5"))
_state = {"backward": None, "make_output": None, "calls": 0}


def make_output(quantizer: NVFP4Quantizer, like: torch.Tensor, parts: list) -> object:
    """Wrap (data, swizzled scales, amax) in the class that TE makes (as nvfp4q_patch.py does)."""
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
        return NVFP4TensorStorage(fake_dtype=like.dtype, **kwargs)
    return NVFP4Tensor(shape=like.shape, dtype=like.dtype, device=like.device, **kwargs)


def _plain_quantizer(quantizer: object) -> bool:
    """Return True for a 1D, rowwise-only NVFP4 quantizer with no special mode."""
    if type(quantizer) is not NVFP4Quantizer:
        return False
    special = quantizer.with_rht or quantizer.with_amax_reduction or quantizer.row_scaled_nvfp4
    special = special or quantizer.nvfp4_use_4over6 or quantizer.with_2d_quantization
    special = special or quantizer.nvfp4_e4m3_max != 448
    usage = quantizer.rowwise_usage and not quantizer.columnwise_usage
    return usage and not special and quantizer.dtype == tex.DType.kFloat4E2M1


def _plain_bf16(tensor: torch.Tensor) -> bool:
    """Return True for a contiguous CUDA bf16 tensor with 32-byte alignment."""
    if type(tensor) is not torch.Tensor or not tensor.is_cuda or tensor.dtype != torch.bfloat16:
        return False
    return tensor.is_contiguous() and tensor.dim() >= 2 and tensor.data_ptr() % 32 == 0


def _supported(grad_output: torch.Tensor, input_: torch.Tensor) -> bool:
    """Return True if the kernels support the shapes (see check_shapes in swiglu_sr.cu)."""
    if not (_plain_bf16(grad_output) and _plain_bf16(input_)):
        return False
    width = input_.shape[-1]
    if width == 0 or width % 512 != 0 or input_.numel() >= 2**31:
        return False
    if grad_output.shape[:-1] != input_.shape[:-1] or grad_output.shape[-1] * 2 != width:
        return False
    rows = input_.numel() // width
    return rows > 0 and rows % 128 == 0


def _fast_math_off() -> bool:
    """TE fast math changes the quantize math. The kernels support only the default."""
    return os.environ.get("NVTE_USE_FAST_MATH", "") in ("", "0")


def two_pass_backward(self: SwiGLU, grad_output: torch.Tensor, input_: torch.Tensor,
                      quantizer: object) -> object:
    """Replacement of SwiGLU._tex_swiglu_backward."""
    usable = type(self) is SwiGLU and _plain_quantizer(quantizer) and _fast_math_off()
    if not (usable and _supported(grad_output, input_)):
        return _state["backward"](self, grad_output, input_, quantizer)
    module = swiglu_sr_ext.get_extension()
    stochastic = bool(quantizer.stochastic_rounding)
    weight_amax, alpha = gemm_alpha.kernel_tensors(quantizer)
    parts = module.dswiglu_nvfp4(grad_output, input_, stochastic, _PASS1_VARIANT, _PASS2_VARIANT,
                                 alpha_weight_amax=weight_amax, alpha_out=alpha)
    _state["calls"] += 1
    gemm_alpha.record(parts[2], quantizer, alpha is not None)
    return _state["make_output"](quantizer, input_, parts)


def apply(make_output_fn: object = make_output, force: bool = False) -> bool:
    """Install the two-pass backward if LORA_SWIGLU_SR_TWO_PASS=1 (or force). Return True if on."""
    if not force and os.environ.get(ENV_FLAG, "0") != "1":
        return False
    if _state["backward"] is None:
        swiglu_sr_ext.get_extension()
        _state["make_output"] = make_output_fn
        _state["backward"] = SwiGLU._tex_swiglu_backward
        SwiGLU._tex_swiglu_backward = two_pass_backward
    return True
