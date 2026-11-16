# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""BF16 fallback implementations for fp8_ops.

Same signatures as fp8_ops.py, pure BF16 ATen ops. cached_fp8_x is the 2D BF16
input itself; fp8_act from fp8_gelu_fwd is the BF16 GELU output.
"""

from __future__ import annotations

import torch
from torch import Tensor


# Linear forward

def _addmm(x_2d: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    wT = torch.ops.aten.t.default(weight)
    if bias is not None:
        return torch.ops.aten.addmm.default(bias, x_2d, wT)
    return torch.ops.aten.mm.default(x_2d, wT)


def _reshape_in(x: Tensor) -> tuple[Tensor, list]:
    shape = list(x.shape)
    if len(shape) == 3:
        return torch.ops.aten.reshape.default(x, [-1, shape[-1]]), shape
    return x, shape


def _reshape_out(out_2d: Tensor, shape: list, out_features: int) -> Tensor:
    if len(shape) == 3:
        return torch.ops.aten.reshape.default(
            out_2d, [shape[0], shape[1], out_features],
        )
    return out_2d


def fp8_linear_fwd(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    x_2d, shape = _reshape_in(x)
    out_2d = _addmm(x_2d, weight, bias)
    return _reshape_out(out_2d, shape, weight.shape[0])


def fp8_linear_fwd_save(
    x: Tensor, weight: Tensor, bias: Tensor,
) -> tuple[Tensor, Tensor]:
    x_2d, shape = _reshape_in(x)
    out_2d = _addmm(x_2d, weight, bias)
    out = _reshape_out(out_2d, shape, weight.shape[0])
    # the 2D BF16 input stands in for cached_fp8_x
    return out, x_2d


def fp8_linear_fwd_gelu(
    x: Tensor, weight: Tensor, bias: Tensor,
) -> tuple[Tensor, Tensor]:
    x_2d, shape = _reshape_in(x)
    gelu_in_2d = _addmm(x_2d, weight, bias)
    gelu_out_2d = torch.ops.aten.gelu.default(gelu_in_2d, approximate="tanh")
    out_features = weight.shape[0]
    if len(shape) == 3:
        return (
            _reshape_out(gelu_out_2d, shape, out_features),
            _reshape_out(gelu_in_2d, shape, out_features),
        )
    return gelu_out_2d, gelu_in_2d


def fp8_linear_fwd_preq(
    fp8_act: Tensor, weight: Tensor, bias: Tensor | None,
    act_shape: list,
) -> Tensor:
    # fp8_act is the 2D BF16 activation here
    out_2d = _addmm(fp8_act, weight, bias)
    if len(act_shape) == 3:
        return _reshape_out(out_2d, act_shape, weight.shape[0])
    return out_2d


# Linear backward

def _bwd_core(
    grad_2d: Tensor, x_2d: Tensor, weight: Tensor,
    wgrad_out: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    grad_x_2d = torch.ops.aten.mm.default(grad_2d, weight)
    grad_T = torch.ops.aten.t.default(grad_2d)
    if wgrad_out is not None:
        torch.ops.aten.mm.out(grad_T, x_2d, out=wgrad_out)
        grad_w = wgrad_out
    else:
        grad_w = torch.ops.aten.mm.default(grad_T, x_2d)
    grad_b_2d = torch.ops.aten.sum.dim_IntList(grad_2d, [0], True)
    grad_b = torch.ops.aten.reshape.default(grad_b_2d, [weight.shape[0]])
    return grad_x_2d, grad_w, grad_b


def fp8_linear_bwd(
    grad: Tensor, x: Tensor | None, weight: Tensor,
    wgrad_out: Tensor | None = None,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    shape = x_shape if x_shape is not None else list(x.shape)
    if cached_fp8_x is not None:
        x_2d = cached_fp8_x  # already 2D BF16
    else:
        x_2d = (
            torch.ops.aten.reshape.default(x, [-1, shape[-1]])
            if len(shape) == 3 else x
        )
    grad_2d = (
        torch.ops.aten.reshape.default(grad, [-1, weight.shape[0]])
        if len(shape) == 3 else grad
    )
    grad_x_2d, grad_w, grad_b = _bwd_core(
        grad_2d, x_2d, weight, wgrad_out,
    )
    if len(shape) == 3:
        grad_x = torch.ops.aten.reshape.default(grad_x_2d, shape)
    else:
        grad_x = grad_x_2d
    return grad_x, grad_w, grad_b


def fp8_linear_bwd_dgrad(
    grad: Tensor, x: Tensor | None, weight: Tensor,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, dict]:
    shape = x_shape if x_shape is not None else list(x.shape)
    if cached_fp8_x is not None:
        x_2d = cached_fp8_x
    else:
        x_2d = (
            torch.ops.aten.reshape.default(x, [-1, shape[-1]])
            if len(shape) == 3 else x
        )
    grad_2d = (
        torch.ops.aten.reshape.default(grad, [-1, weight.shape[0]])
        if len(shape) == 3 else grad
    )
    grad_x_2d = torch.ops.aten.mm.default(grad_2d, weight)
    if len(shape) == 3:
        grad_x = torch.ops.aten.reshape.default(grad_x_2d, shape)
    else:
        grad_x = grad_x_2d
    wctx = {"grad_2d": grad_2d, "x_2d": x_2d, "weight": weight}
    return grad_x, wctx


def fp8_linear_bwd_wgrad(
    ctx: dict, wgrad_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    grad_2d = ctx["grad_2d"]
    x_2d = ctx["x_2d"]
    weight = ctx["weight"]
    grad_T = torch.ops.aten.t.default(grad_2d)
    if wgrad_out is not None:
        torch.ops.aten.mm.out(grad_T, x_2d, out=wgrad_out)
        grad_w = wgrad_out
    else:
        grad_w = torch.ops.aten.mm.default(grad_T, x_2d)
    grad_b_2d = torch.ops.aten.sum.dim_IntList(grad_2d, [0], True)
    grad_b = torch.ops.aten.reshape.default(grad_b_2d, [weight.shape[0]])
    return grad_w, grad_b


def fp8_linear_bwd_dgelu(
    grad: Tensor, gelu_input: Tensor,
    x: Tensor | None, weight: Tensor,
    wgrad_out: Tensor | None = None,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    grad_pre = torch.ops.aten.gelu_backward.default(
        grad, gelu_input, approximate="tanh",
    )
    return fp8_linear_bwd(
        grad_pre, x, weight, wgrad_out,
        cached_fp8_x=cached_fp8_x, x_shape=x_shape,
    )


def fp8_linear_bwd_dgrad_dgelu(
    grad: Tensor, gelu_input: Tensor,
    x: Tensor | None, weight: Tensor,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, dict]:
    grad_pre = torch.ops.aten.gelu_backward.default(
        grad, gelu_input, approximate="tanh",
    )
    return fp8_linear_bwd_dgrad(
        grad_pre, x, weight,
        cached_fp8_x=cached_fp8_x, x_shape=x_shape,
    )


def fp8_linear_bwd_dgrad_preq_grad(
    fp8_grad: Tensor, gb: Tensor | None, weight: Tensor,
    x_shape: list,
    *,
    cached_fp8_x: object | None = None,
) -> tuple[Tensor, dict]:
    # fp8_grad is the BF16 grad_2d here
    grad_x_2d = torch.ops.aten.mm.default(fp8_grad, weight)
    if len(x_shape) == 3:
        grad_x = torch.ops.aten.reshape.default(grad_x_2d, x_shape)
    else:
        grad_x = grad_x_2d
    wctx = {
        "grad_2d": fp8_grad,
        "x_2d": cached_fp8_x,
        "weight": weight,
        "grad_b": gb,
    }
    return grad_x, wctx


def fp8_linear_bwd_preq_grad(
    fp8_grad: Tensor, weight: Tensor, x_shape: list,
) -> Tensor:
    grad_x_2d = torch.ops.aten.mm.default(fp8_grad, weight)
    if len(x_shape) == 3:
        return torch.ops.aten.reshape.default(grad_x_2d, x_shape)
    return grad_x_2d


def fp8_linear_bwd_preq_both(
    fp8_grad: Tensor, fp8_x: Tensor, weight: Tensor,
    x_shape: list,
    wgrad_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    grad_x_2d = torch.ops.aten.mm.default(fp8_grad, weight)
    grad_T = torch.ops.aten.t.default(fp8_grad)
    if wgrad_out is not None:
        torch.ops.aten.mm.out(grad_T, fp8_x, out=wgrad_out)
        grad_w = wgrad_out
    else:
        grad_w = torch.ops.aten.mm.default(grad_T, fp8_x)
    if len(x_shape) == 3:
        grad_x = torch.ops.aten.reshape.default(grad_x_2d, x_shape)
    else:
        grad_x = grad_x_2d
    return grad_x, grad_w


# Quantize/dequantize fallbacks

def release_fwd(cached: object) -> None:
    """No-op: the BF16 fallback quantizes nothing, so nothing is held.

    Args:
        cached: The forward feeder's return value. Ignored.
    """


def fp8_quantize_fwd(x: Tensor, *weights: Tensor) -> tuple:
    # weights are ignored; every seam implementation must take the same arguments
    shape = list(x.shape)
    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x
    return x_2d, shape


def fp8_quantize_bwd(x: Tensor) -> tuple:
    shape = list(x.shape)
    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x
    return x_2d, shape


def fp8_gelu_fwd(x: Tensor, *weights: Tensor) -> tuple:
    # returns (bf16_act_2d, original_shape); weights are ignored
    shape = list(x.shape)
    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x
    out_2d = torch.ops.aten.gelu.default(x_2d, approximate="tanh")
    return out_2d, shape


def fp8_bgrad_quantize(grad: Tensor, *weights: Tensor) -> tuple:
    # weights are ignored
    grad_2d = grad
    if grad.ndim == 3:
        grad_2d = torch.ops.aten.reshape.default(
            grad, [-1, grad.shape[-1]],
        )
    grad_b_2d = torch.ops.aten.sum.dim_IntList(grad_2d, [0], True)
    grad_b = torch.ops.aten.reshape.default(grad_b_2d, [grad.shape[-1]])
    return grad_b, grad_2d, None


def fp8_rope_join_bgrad_quantize(
    grad_q_rope: Tensor, grad_k_rope: Tensor, grad_v: Tensor, qkv: Tensor, H: int,
    q_rstd: Tensor, k_rstd: Tensor, qn_weight: Tensor, kn_weight: Tensor,
    grad_qn_weight: Tensor, grad_kn_weight: Tensor, freqs: Tensor, pos_offset: int,
    *weights: Tensor,
) -> tuple:
    grad_qkv = torch.ops.flux.rope_norm_join_ctm(
        grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd, qn_weight,
        kn_weight, grad_qn_weight, grad_kn_weight, freqs, pos_offset,
    )
    return fp8_bgrad_quantize(grad_qkv, *weights)


def fp8_gate_bwd_quantize(
    grad: Tensor, gate: Tensor, proj: Tensor, *weights: Tensor,
    bias_grad: Tensor | None = None,
) -> tuple:
    grad_proj, grad_gate = torch.ops.flux.fused_gate_bwd(grad, gate, proj)
    grad_b, grad_2d, shape = fp8_bgrad_quantize(grad_proj, *weights)
    if bias_grad is not None:
        torch.ops.aten.copy_.default(bias_grad, grad_b)
        grad_b = bias_grad
    return grad_b, grad_2d, shape, grad_gate


def fp8_qgelu_fwd(x: Tensor, *weights: Tensor) -> tuple:
    return fp8_gelu_fwd(x, *weights)


def fp8_dbias_dqgelu_bwd(grad: Tensor, fwd_input: Tensor) -> tuple:
    grad_pre = torch.ops.aten.gelu_backward.default(
        grad, fwd_input, approximate="tanh",
    )
    D = fwd_input.shape[-1]
    if grad_pre.ndim == 3:
        grad_pre_2d = torch.ops.aten.reshape.default(grad_pre, [-1, D])
    else:
        grad_pre_2d = grad_pre
    grad_b_2d = torch.ops.aten.sum.dim_IntList(grad_pre_2d, [0], True)
    grad_b = torch.ops.aten.reshape.default(grad_b_2d, [D])
    return grad_b, grad_pre_2d, None


def clear_fp8_weight_cache() -> None:
    pass
