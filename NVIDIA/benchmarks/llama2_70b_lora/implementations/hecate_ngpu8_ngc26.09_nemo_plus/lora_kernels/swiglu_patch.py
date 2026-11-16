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

"""Use the fast SwiGLU kernels in the te.ops SwiGLU op. The result has the same bits as TE.

apply(quantize_fn) replaces SwiGLU._tex_swiglu_forward and SwiGLU._tex_swiglu_backward.
For a plain rowwise NVFP4 quantizer (or no quantizer) and a plain bf16 input, the forward
runs swiglu_fused.forward and the backward runs swiglu_fused.backward. Then
quantize_fn(tensor, amax, quantizer) makes the NVFP4 tensor. The default quantize_fn is the
TE quantize with a known amax, which is the second step of the TE fused path. For a rowwise
delayed-scaling FP8 quantizer (the FP8 healing phase), the kernels write FP8 directly
(swiglu_fused.forward_fp8 and backward_fp8). Other cases go to TE. Call apply() before CUDA
graph capture.
"""

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine_torch as tex
from transformer_engine.pytorch.ops.basic.swiglu import SwiGLU
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

import swiglu_fused

_state = {"forward": None, "backward": None, "quantize": None}


def te_quantize(tensor: torch.Tensor, amax: torch.Tensor, quantizer: NVFP4Quantizer) -> object:
    """Quantize to NVFP4 with a known amax, as the TE fused activation path does."""
    return tex.nvfp4_quantize_with_amax(tensor, quantizer, amax, amax)


def _plain_quantizer(quantizer: object) -> bool:
    """Return True for a rowwise-only NVFP4 quantizer with no special mode."""
    if type(quantizer) is not NVFP4Quantizer:
        return False
    special = (quantizer.with_rht or quantizer.with_amax_reduction
               or quantizer.row_scaled_nvfp4 or quantizer.nvfp4_use_4over6
               or quantizer.with_2d_quantization)
    return not special and quantizer.rowwise_usage and not quantizer.columnwise_usage


def _plain_tensor(tensor: torch.Tensor) -> bool:
    """Return True for a contiguous bf16 CUDA tensor that the 256-bit kernels can use."""
    if type(tensor) is not torch.Tensor or not tensor.is_cuda:
        return False
    if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or tensor.dim() < 2:
        return False
    return tensor.shape[-1] % 32 == 0 and tensor.data_ptr() % 32 == 0


def _plain_fp8_quantizer(quantizer: object) -> bool:
    """Return True for a rowwise-only delayed-scaling FP8 quantizer (the FP8 healing phase)."""
    if type(quantizer) is not Float8Quantizer:
        return False
    return quantizer.rowwise_usage and not quantizer.columnwise_usage


def _usable(op: SwiGLU, quantizer: object) -> bool:
    """Return True if the op is a plain SwiGLU and the fast path can use the quantizer."""
    if type(op) is not SwiGLU:
        return False
    return quantizer is None or _plain_quantizer(quantizer) or _plain_fp8_quantizer(quantizer)


def fast_forward(self: SwiGLU, input_: torch.Tensor, quantizer: object) -> object:
    """Replacement of SwiGLU._tex_swiglu_forward."""
    if not (_usable(self, quantizer) and _plain_tensor(input_)):
        return _state["forward"](self, input_, quantizer)
    if _plain_fp8_quantizer(quantizer):
        return swiglu_fused.forward_fp8(input_, quantizer)
    y, amax = swiglu_fused.forward(input_)
    if quantizer is None:
        return y
    return _state["quantize"](y, amax, quantizer)


def fast_backward(self: SwiGLU, grad_output: torch.Tensor, input_: torch.Tensor,
                  quantizer: object) -> object:
    """Replacement of SwiGLU._tex_swiglu_backward."""
    plain = _plain_tensor(input_) and _plain_tensor(grad_output)
    if not (_usable(self, quantizer) and plain):
        return _state["backward"](self, grad_output, input_, quantizer)
    if _plain_fp8_quantizer(quantizer):
        return swiglu_fused.backward_fp8(grad_output, input_, quantizer)
    dx, amax = swiglu_fused.backward(grad_output, input_)
    if quantizer is None:
        return dx
    return _state["quantize"](dx, amax, quantizer)


def apply(quantize_fn: object = te_quantize) -> None:
    """Replace the TE SwiGLU kernel calls. Do this before CUDA graph capture."""
    if _state["forward"] is not None:
        return
    swiglu_fused.init()
    _state["quantize"] = quantize_fn
    _state["forward"] = SwiGLU._tex_swiglu_forward
    _state["backward"] = SwiGLU._tex_swiglu_backward
    SwiGLU._tex_swiglu_forward = fast_forward
    SwiGLU._tex_swiglu_backward = fast_backward
