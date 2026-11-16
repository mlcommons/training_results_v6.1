# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Make the LoRA dropout drop the values with the exact probability p, as nn.Dropout does.

The TE fused dropout of the te.ops Dropout op compares random bytes with floor(p * 256). For
p = 0.1, it drops 25/256 = 9.77% of the values. But it scales the other values by 1 / (1 - p).
apply() gives the op the torch dropout kernel, which compares random floats with p.
"""

import torch
from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from transformer_engine.pytorch.ops._common import maybe_autocast_dtype, maybe_dequantize
from transformer_engine.pytorch.ops.basic.dropout import Dropout

_original = {}


def _forward(
    self: Dropout,
    ctx: object,
    input_: torch.Tensor,
    prev_op_grad_output_quantizer: object,
    next_op_input_quantizer: object,
) -> torch.Tensor:
    """Replacement of Dropout.op_forward. Eval mode and p = 0 keep the TE path."""
    if not self.training or self.dropout_probability == 0:
        return _original["forward"](
            self, ctx, input_, prev_op_grad_output_quantizer, next_op_input_quantizer
        )
    dtype = maybe_autocast_dtype(default_dtype=input_.dtype)
    x = maybe_dequantize(input_, dtype=dtype)
    out, mask = torch.ops.aten.native_dropout(x, self.dropout_probability, True)
    if ctx.requires_grad:
        if is_cpu_offload_enabled():
            mark_activation_offload(mask)
        ctx.save_for_backward(mask)
        ctx.impl = "torch"
        ctx.dropout_probability = self.dropout_probability
        ctx.dtype = dtype
    return out


def _backward(self: Dropout, ctx: object, grad_output: torch.Tensor) -> tuple:
    """Replacement of Dropout.op_backward."""
    if getattr(ctx, "impl", None) != "torch":
        return _original["backward"](self, ctx, grad_output)
    (mask,) = ctx.saved_tensors
    dy = maybe_dequantize(grad_output, dtype=ctx.dtype)
    scale = 1.0 / (1.0 - ctx.dropout_probability)
    return torch.ops.aten.native_dropout_backward(dy, mask, scale), ()


def apply() -> None:
    """Patch the te.ops Dropout op one time."""
    if _original:
        return
    _original["forward"] = Dropout.op_forward
    _original["backward"] = Dropout.op_backward
    Dropout.op_forward = _forward
    Dropout.op_backward = _backward
