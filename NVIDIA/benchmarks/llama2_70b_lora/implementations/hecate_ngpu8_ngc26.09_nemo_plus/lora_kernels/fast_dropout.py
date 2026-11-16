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

"""Faster LoRA dropout with the same output, mask and RNG use as TE.

apply() makes three changes:
- tex.dropout_fwd: a faster kernel for plain bf16 input (the proj LoRA dropout).
- tex.dropout_bwd: a faster mask kernel for bf16 grads (both LoRA dropouts).
- te.ops Dropout forward: NVFP4 input goes to one kernel that does the NVFP4 dequantize
  and the dropout (the QKV LoRA dropout). TE does a dequantize kernel and a dropout kernel.
Other inputs go to the TE functions.
"""

import math

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpu_offload import (
    is_cpu_offload_enabled,
    mark_activation_offload,
)
from transformer_engine.pytorch.ops._common import maybe_autocast_dtype
from transformer_engine.pytorch.ops.basic.dropout import Dropout
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (
    NVFP4TensorStorage,
)

import lora_attn_ext

_state = {"op_forward": None, "dropout_fwd": None, "dropout_bwd": None}


def _plain_bf16(tensor: object) -> bool:
    """Return True for a contiguous, 16-byte aligned bf16 CUDA tensor (not a subclass)."""
    if type(tensor) is not torch.Tensor or not tensor.is_cuda:
        return False
    if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
        return False
    return tensor.numel() % 16 == 0 and tensor.data_ptr() % 16 == 0


def _plain_nvfp4_data(data: torch.Tensor | None) -> bool:
    """Return True for contiguous, 16-byte aligned rowwise NVFP4 data on the GPU."""
    if data is None or not data.is_cuda or not data.is_contiguous():
        return False
    return data.dim() >= 2 and data.numel() > 0 and data.data_ptr() % 16 == 0


def _plain_nvfp4(tensor: object) -> bool:
    """Return True for NVFP4 data that dequantize_fp4_kernel reads with one tensor amax."""
    if not isinstance(tensor, NVFP4TensorStorage):
        return False
    if tensor._row_scaled_nvfp4 or tensor._nvfp4_use_4over6:
        return False
    amax = tensor._amax_rowwise
    if amax is None or amax.numel() != 1 or amax.dtype != torch.float32:
        return False
    scales = tensor._rowwise_scale_inv
    if scales is None or scales.dim() != 2 or not scales.is_contiguous():
        return False
    return scales.dtype == torch.uint8 and _plain_nvfp4_data(tensor._rowwise_data)


def _plain_mask(mask: torch.Tensor, grad_output: torch.Tensor) -> bool:
    """Return True for a contiguous, 2-byte aligned uint8 mask with one bit per element."""
    if not mask.is_cuda or mask.dtype != torch.uint8 or not mask.is_contiguous():
        return False
    return mask.data_ptr() % 2 == 0 and mask.numel() * 8 == grad_output.numel()


def fast_dropout_fwd(
    input: torch.Tensor,
    dropout_probability: float,
    out: torch.Tensor | None = None,
) -> list:
    """Same signature and result as tex.dropout_fwd."""
    if out is None and _plain_bf16(input):
        return lora_attn_ext.get_module().dropout_fwd_bf16(input, dropout_probability)
    return _state["dropout_fwd"](input, dropout_probability, out)


def fast_dropout_bwd(
    grad_output: torch.Tensor,
    mask: torch.Tensor,
    dropout_probability: float,
    grad_input: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same signature and result as tex.dropout_bwd."""
    if grad_input is None and _plain_bf16(grad_output) and _plain_mask(mask, grad_output):
        module = lora_attn_ext.get_module()
        return module.dropout_bwd_bf16(grad_output, mask, dropout_probability)
    return _state["dropout_bwd"](grad_output, mask, dropout_probability, grad_input)


def _nvfp4_dropout(op: Dropout, input_: NVFP4TensorStorage) -> tuple:
    """Dequantize the NVFP4 input and apply dropout in one kernel."""
    module = lora_attn_ext.get_module()
    return module.dropout_fwd_nvfp4(
        input_._rowwise_data,
        input_._rowwise_scale_inv,
        input_._amax_rowwise,
        bool(input_._with_gemm_swizzled_scales),
        op.dropout_probability,
        list(input_.size()),
    )


def dropout_op_forward(
    self: Dropout,
    ctx: object,
    input_: torch.Tensor,
    prev_op_grad_output_quantizer: object,
    next_op_input_quantizer: object,
) -> torch.Tensor:
    """Replacement of te.ops Dropout.op_forward with the same saved state."""
    args = (ctx, input_, prev_op_grad_output_quantizer, next_op_input_quantizer)
    if not self.training or not _plain_nvfp4(input_):
        return _state["op_forward"](self, *args)
    dtype = maybe_autocast_dtype(default_dtype=input_.dtype)
    if dtype != torch.bfloat16 or math.prod(input_.size()) % 16 != 0:
        return _state["op_forward"](self, *args)
    out, mask = _nvfp4_dropout(self, input_)
    if ctx.requires_grad:
        if is_cpu_offload_enabled():
            mark_activation_offload(mask)
        ctx.save_for_backward(mask)
        ctx.impl = "fused"
        ctx.dropout_probability = self.dropout_probability
        ctx.dtype = dtype
    return out


def apply() -> None:
    """Install the faster dropout functions."""
    if _state["op_forward"] is not None:
        return
    lora_attn_ext.get_module()
    _state["op_forward"] = Dropout.op_forward
    _state["dropout_fwd"] = tex.dropout_fwd
    _state["dropout_bwd"] = tex.dropout_bwd
    Dropout.op_forward = dropout_op_forward
    tex.dropout_fwd = fast_dropout_fwd
    tex.dropout_bwd = fast_dropout_bwd
