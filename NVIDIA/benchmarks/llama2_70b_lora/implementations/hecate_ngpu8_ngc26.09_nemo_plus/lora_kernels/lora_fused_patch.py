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

"""Fused LoRA branch for Bridge's TEFusedLoRALinear (switch LORA_PATCH_FUSED_LORA).

The te.ops branch runs: Dropout -> LoRA A -> LoRA B (+ scale, added into the base output).
This patch keeps the LoRA B GEMM as the cuBLAS GEMM with alpha = scale and its add into the base
output. The other ops run in three fused kernels (lora_fused.cu). The kernels apply the dropout
scale and the NVFP4 tensor scale in the fp32 reduce, so the results differ from TE by bf16
rounding noise. The dropout masks come from the same Philox scheme as the TE dropout (same
seed, offset, and chunk index) unless LORA_FUSED_FAST_RNG=1.
"""

import os

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.ops._common import get_dummy_wgrads_for_params
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (
    NVFP4TensorStorage,
)

import lora_fused_ext

RANK = 16
BLOCK = 512
# LORA_FUSED_FAST_RNG=1: splitmix64 mask bits (cheaper); default: the TE Philox mask bits.
TE_RNG = os.environ.get("LORA_FUSED_FAST_RNG", "0") != "1"
_state = {"forward": None}


def _nvfp4_parts(tensor: NVFP4TensorStorage) -> tuple | None:
    data, scales, amax = tensor._rowwise_data, tensor._rowwise_scale_inv, tensor._amax_rowwise
    if data is None or scales is None or amax is None or amax.numel() != 1:
        return None
    if getattr(tensor, "_row_scaled_nvfp4", False) or getattr(tensor, "_nvfp4_use_4over6", False):
        return None
    cols = tensor.shape[-1]
    tokens = tensor.numel() // cols
    swizzled = bool(getattr(tensor, "_with_gemm_swizzled_scales", False))
    return data.contiguous().view(tokens, cols // 2), scales.contiguous(), amax, swizzled, tokens, cols


def _nvfp4_phase() -> bool:
    """Return True in the NVFP4 training phase (an NVFP4 autocast recipe)."""
    if not FP8GlobalStateManager.is_fp8_enabled():
        return False
    return isinstance(FP8GlobalStateManager.get_fp8_recipe(), NVFP4BlockScaling)


def _fp8_parts(tensor: Float8TensorStorage) -> tuple | None:
    """FP8 E4M3 input in the NVFP4 phase: the attention output O of LORA_PATCH_FP8_O.

    The scales slot holds the float scale_inv. The kernels dequantize with the TE math.
    """
    data, scale_inv = tensor._data, tensor._scale_inv
    if data is None or scale_inv is None or scale_inv.numel() != 1:
        return None
    if tensor._fp8_dtype != tex.DType.kFloat8E4M3 or not _nvfp4_phase():
        return None
    cols = data.shape[-1]
    tokens = data.numel() // cols
    return data.contiguous().view(tokens, cols), scale_inv, None, False, tokens, cols


def input_parts(tensor: torch.Tensor) -> tuple | None:
    """Return (data, scales, amax, swizzled, tokens, cols) of a supported branch input, or None.

    NVFP4 input: rowwise data with one tensor amax (plain or GEMM-swizzled scales).
    FP8 input: the E4M3 attention output in the NVFP4 phase (scales holds scale_inv).
    bf16 input: a contiguous tensor. Other inputs (Float8Tensor in the FP8 phase) return None.
    """
    if isinstance(tensor, Float8TensorStorage):
        return _fp8_parts(tensor)
    if isinstance(tensor, NVFP4TensorStorage):
        return _nvfp4_parts(tensor)
    if type(tensor) is torch.Tensor and tensor.dtype == torch.bfloat16 and tensor.is_cuda:
        cols = tensor.shape[-1]
        tokens = tensor.numel() // cols
        return tensor.contiguous().view(tokens, cols), None, None, False, tokens, cols
    return None


def shapes_ok(tokens: int, cols: int, out_cols: int) -> bool:
    return tokens % BLOCK == 0 and cols % BLOCK == 0 and out_cols % BLOCK == 0


class FusedLoRABranch(torch.autograd.Function):
    """out = base + scale * LoRA_B(LoRA_A(dropout(x))), with the LoRA grads into main_grad."""

    @staticmethod
    def forward(ctx: object, linear_input: torch.Tensor, linear_output: torch.Tensor,
                lora_a: torch.Tensor, lora_b: torch.Tensor, options: tuple) -> torch.Tensor:
        # options = (scale, dropout probability, accumulate into main_grad)
        scale, p, into_main_grad = options
        data, scales, amax, swizzled, tokens, cols = input_parts(linear_input)
        ext = lora_fused_ext.get_extension()
        h, mask = ext.lora_forward(data, scales, amax, swizzled, tokens, cols, lora_a.detach(), p, TE_RNG)
        # LoRA B GEMM with alpha = scale, added into the base output (as the te.ops fuser does).
        out = linear_output.detach()
        general_gemm(lora_b.detach(), h, out_dtype=out.dtype, alpha=scale, accumulate=True,
                     out=out.view(tokens, -1), layout="TN")
        ctx.save_for_backward(data, scales, amax, mask, h, lora_a, lora_b)
        ctx.meta = (swizzled, tokens, cols, scale, p, into_main_grad, tuple(linear_input.shape))
        return out

    @staticmethod
    def backward(ctx: object, grad_out: torch.Tensor) -> tuple:
        data, scales, amax, mask, h, lora_a, lora_b = ctx.saved_tensors
        swizzled, tokens, cols, scale, p, into_main_grad, in_shape = ctx.meta
        ext = lora_fused_ext.get_extension()
        dy = grad_out.contiguous().view(tokens, -1)
        if into_main_grad:
            grad_b, grad_a = lora_b.main_grad, lora_a.main_grad
        else:
            grad_b, grad_a = torch.empty_like(lora_b), torch.empty_like(lora_a)
        dh = ext.lora_backward_dy(dy, lora_b.detach(), h, grad_b, into_main_grad, scale)
        dx = ext.lora_backward_dx(data, scales, amax, swizzled, tokens, cols, mask, dh, lora_a.detach(),
                                  grad_a, into_main_grad, p)
        if into_main_grad:
            wgrads = get_dummy_wgrads_for_params([lora_a, lora_b])
            grad_a, grad_b = wgrads[0], wgrads[1]
        return dx.view(in_shape), grad_out, grad_a, grad_b, None


def branch_supported(module: object, parts: tuple | None) -> bool:
    """Return True if the fused branch applies to this LoRA module and input."""
    if parts is None:
        return False
    adapter = module.adapter
    if getattr(adapter, "dropout_position", "pre") != "pre" or getattr(adapter.dropout, "p", 0.0) <= 0:
        return False
    lora_a, lora_b = adapter.linear_in.weight, adapter.linear_out.weight
    if lora_a.dtype != torch.bfloat16 or lora_b.dtype != torch.bfloat16 or lora_a.shape[0] != RANK:
        return False
    _, _, _, _, tokens, cols = parts
    return shapes_ok(tokens, cols, lora_b.shape[0]) and lora_a.shape[1] == cols


def _uses_main_grad(*params: torch.Tensor) -> bool:
    return all(getattr(param, "main_grad", None) is not None for param in params)


def _dropout_training(lora_branch: object) -> bool:
    """Return True if the te.ops Dropout of the branch is in training mode (then TE drops)."""
    return any(isinstance(op, te.ops.Dropout) and op.training for op in lora_branch.modules())


def fused_forward(self: object, x: torch.Tensor) -> tuple:
    """Replacement of TEFusedLoRALinear.forward."""
    if not self._adapter_enabled:
        return _state["forward"](self, x)
    if self._fused_branches is None:
        self._fused_branches = self._make_fused_branches()
    main_branch, lora_branch = self._fused_branches
    linear_output, linear_input = main_branch(x)
    parts = input_parts(linear_input)
    # The kernels always apply the dropout. In eval mode, the te.ops branch runs with no dropout.
    if not branch_supported(self, parts) or not _dropout_training(lora_branch):
        with te.fp8_autocast(enabled=False):
            out = lora_branch(linear_input, linear_output)
        return out, None
    adapter = self.adapter
    lora_a, lora_b = adapter.linear_in.weight, adapter.linear_out.weight
    options = (adapter.alpha / adapter.dim, adapter.dropout.p, _uses_main_grad(lora_a, lora_b))
    out = FusedLoRABranch.apply(linear_input, linear_output, lora_a, lora_b, options)
    # The te.ops branch carries the forward hooks of the adapter submodules.
    for hook in lora_branch._forward_hooks.values():
        hook(lora_branch, None, None)
    return out, None


def apply() -> None:
    """Replace TEFusedLoRALinear.forward when LORA_PATCH_FUSED_LORA=1."""
    if _state["forward"] is not None or os.environ.get("LORA_PATCH_FUSED_LORA", "0") != "1":
        return
    from megatron.bridge.peft.lora_layers import TEFusedLoRALinear

    lora_fused_ext.get_extension()
    _state["forward"] = TEFusedLoRALinear.forward
    TEFusedLoRALinear.forward = fused_forward
