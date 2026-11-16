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

"""MXFP8 linear operations wrapping TransformerEngine CUDA kernels.

Provides drop-in replacements for _linear_fwd / _linear_bwd that quantize
inputs to FP8 E4M3 with 32-element MXFP8 block scaling and execute GEMMs via
cuBLASLt.
"""

from __future__ import annotations

import os as _os

import torch
from torch import Tensor

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.tensor.mxfp8_tensor import (
    MXFP8Quantizer,
)
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm

from flat.weight_cache import WeightCache

# FLAT_BF16=1 routes every fp8_linear_* through plain BF16 aten ops (fp8_ops_bf16).
_FLAT_BF16 = _os.environ.get("FLAT_BF16", "0") == "1"

# Both quantizers emit rowwise + columnwise: in the all-MXFP8 path every half has
# a consumer (fprop TN reads rowwise; dgrad NN and wgrad NT read columnwise).
_fwd_quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)
_fwd_quantizer.optimize_for_gemm = True
_bwd_quantizer = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=True,
)
_bwd_quantizer.optimize_for_gemm = True

# weight tensor -> MXFP8Tensor, keyed on tensor identity.
# Cleared by clear_fp8_weight_cache() after the optimizer step.
_weight_fp8_cache = WeightCache()

def clear_fp8_weight_cache() -> None:
    """Clear cached FP8 weights. Call after optimizer step."""
    _weight_fp8_cache.clear()


def _quantize(t: Tensor, quantizer: MXFP8Quantizer):
    """Quantize a BF16/FP32 tensor to MXFP8; both dims must be divisible by 32."""
    if not t.is_contiguous():
        t = t.contiguous()
    return tex.quantize(t, quantizer)


def fp8_linear_fwd(
    x: Tensor, weight: Tensor, bias: Tensor,
) -> Tensor:
    """FP8 linear forward: out = x @ weight^T + bias.

    x: [*, in_features], weight: [out_features, in_features], bias: [out_features]; all BF16.
    """
    shape = list(x.shape)
    out_features = weight.shape[0]

    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x

    fp8_x = _quantize(x_2d, _fwd_quantizer)
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _fwd_quantizer)
        _weight_fp8_cache.put(weight, fp8_w)

    # general_gemm(A=W, B=x, layout='TN') computes x @ W^T
    out_2d, _, _, _ = general_gemm(
        fp8_w, fp8_x,
        out_dtype=torch.bfloat16,
        layout="TN",
        bias=bias,
    )

    if len(shape) == 3:
        return torch.ops.aten.reshape.default(
            out_2d, [shape[0], shape[1], out_features],
        )
    return out_2d


def fp8_linear_fwd_save(
    x: Tensor, weight: Tensor, bias: Tensor,
) -> tuple[Tensor, object]:
    """FP8 linear forward that also returns the quantized input for backward reuse.

    Returns (output, fp8_x): output [*, out_features] BF16, fp8_x an MXFP8Tensor.
    """
    shape = list(x.shape)
    out_features = weight.shape[0]

    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x

    fp8_x = _quantize(x_2d, _fwd_quantizer)
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _fwd_quantizer)
        _weight_fp8_cache.put(weight, fp8_w)

    out_2d, _, _, _ = general_gemm(
        fp8_w, fp8_x,
        out_dtype=torch.bfloat16,
        layout="TN",
        bias=bias,
    )

    if len(shape) == 3:
        return torch.ops.aten.reshape.default(
            out_2d, [shape[0], shape[1], out_features],
        ), fp8_x
    return out_2d, fp8_x


def fp8_linear_fwd_gelu(
    x: Tensor, weight: Tensor, bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """FP8 linear forward with GELU fused into the GEMM epilogue.

    Returns (gelu_output, gelu_input), both [*, out_features] BF16; gelu_input is pre-GELU.
    """
    shape = list(x.shape)
    out_features = weight.shape[0]

    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x

    fp8_x = _quantize(x_2d, _fwd_quantizer)
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _fwd_quantizer)
        _weight_fp8_cache.put(weight, fp8_w)

    # third output is the pre-GELU activation
    out_2d, _, gelu_in_2d, _ = general_gemm(
        fp8_w, fp8_x,
        out_dtype=torch.bfloat16,
        layout="TN",
        bias=bias,
        gelu=True,
    )

    if len(shape) == 3:
        out = torch.ops.aten.reshape.default(
            out_2d, [shape[0], shape[1], out_features],
        )
        gelu_in = torch.ops.aten.reshape.default(
            gelu_in_2d, [shape[0], shape[1], out_features],
        )
        return out, gelu_in
    return out_2d, gelu_in_2d


def fp8_linear_bwd(
    grad: Tensor, x: Tensor | None, weight: Tensor,
    wgrad_out: Tensor | None = None,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """FP8 linear backward: grad_x = grad @ W, grad_w = grad^T @ x, grad_b = sum(grad, 0)."""
    shape = x_shape if x_shape is not None else list(x.shape)
    if len(shape) == 3:
        grad_2d = torch.ops.aten.reshape.default(
            grad, [-1, weight.shape[0]],
        )
        if cached_fp8_x is None:
            x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        grad_2d = grad
        if cached_fp8_x is None:
            x_2d = x

    grad_b, fp8_grad = tex.bgrad_quantize(grad_2d, _bwd_quantizer)
    fp8_x = cached_fp8_x if cached_fp8_x is not None else _quantize(
        x_2d, _bwd_quantizer,
    )
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _bwd_quantizer)

    # general_gemm(A, B) layouts: TN -> B @ A^T, NN -> B @ A, NT -> B^T @ A
    # grad_x = grad @ W: NN with A=W, B=grad
    grad_x_2d, _, _, _ = general_gemm(
        fp8_w, fp8_grad,
        out_dtype=torch.bfloat16,
        layout="NN",
    )

    # grad_w = grad^T @ x: NT with A=x, B=grad
    grad_w, _, _, _ = general_gemm(
        fp8_x, fp8_grad,
        out_dtype=torch.bfloat16,
        layout="NT",
        out=wgrad_out,
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
    """FP8 linear backward, dgrad only; wgrad and grad_b deferred to fp8_linear_bwd_wgrad()."""
    shape = x_shape if x_shape is not None else list(x.shape)
    if len(shape) == 3:
        grad_2d = torch.ops.aten.reshape.default(
            grad, [-1, weight.shape[0]],
        )
        if cached_fp8_x is None:
            x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        grad_2d = grad
        if cached_fp8_x is None:
            x_2d = x

    grad_b, fp8_grad = tex.bgrad_quantize(grad_2d, _bwd_quantizer)
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _bwd_quantizer)

    grad_x_2d, _, _, _ = general_gemm(
        fp8_w, fp8_grad,
        out_dtype=torch.bfloat16,
        layout="NN",
    )

    if len(shape) == 3:
        grad_x = torch.ops.aten.reshape.default(grad_x_2d, shape)
    else:
        grad_x = grad_x_2d

    wgrad_ctx = {
        "fp8_grad": fp8_grad,
        "grad_b": grad_b,
        "weight_shape": weight.shape,
    }
    if cached_fp8_x is not None:
        wgrad_ctx["fp8_x"] = cached_fp8_x
    else:
        wgrad_ctx["x_2d"] = x_2d
    return grad_x, wgrad_ctx


def fp8_linear_bwd_wgrad(
    ctx: dict,
    wgrad_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """FP8 linear backward, wgrad only; returns (grad_w, grad_b), grad_b from the dgrad ctx."""
    fp8_x = ctx.get("fp8_x")
    if fp8_x is None:
        fp8_x = _quantize(ctx["x_2d"], _bwd_quantizer)

    grad_w, _, _, _ = general_gemm(
        fp8_x, ctx["fp8_grad"],
        out_dtype=torch.bfloat16,
        layout="NT",
        out=wgrad_out,
    )

    return grad_w, ctx["grad_b"]


def fp8_linear_bwd_dgelu(
    grad: Tensor, gelu_input: Tensor,
    x: Tensor, weight: Tensor,
    wgrad_out: Tensor | None = None,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """FP8 linear backward for a linear whose output feeds a GELU (fused dgelu + dbias)."""
    shape = x_shape if x_shape is not None else list(x.shape)
    if len(shape) == 3:
        grad_2d = torch.ops.aten.reshape.default(
            grad, [-1, weight.shape[0]],
        )
        gelu_in_2d = torch.ops.aten.reshape.default(
            gelu_input, [-1, weight.shape[0]],
        )
        if cached_fp8_x is None:
            x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        grad_2d = grad
        gelu_in_2d = gelu_input
        if cached_fp8_x is None:
            x_2d = x

    grad_b, fp8_dgelu = tex.dbias_dgelu(
        grad_2d, gelu_in_2d, _bwd_quantizer,
    )
    fp8_x = cached_fp8_x if cached_fp8_x is not None else _quantize(
        x_2d, _bwd_quantizer,
    )
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _bwd_quantizer)

    grad_x_2d, _, _, _ = general_gemm(
        fp8_w, fp8_dgelu,
        out_dtype=torch.bfloat16,
        layout="NN",
    )

    grad_w, _, _, _ = general_gemm(
        fp8_x, fp8_dgelu,
        out_dtype=torch.bfloat16,
        layout="NT",
        out=wgrad_out,
    )

    if len(shape) == 3:
        grad_x = torch.ops.aten.reshape.default(grad_x_2d, shape)
    else:
        grad_x = grad_x_2d

    return grad_x, grad_w, grad_b


def fp8_linear_bwd_dgrad_dgelu(
    grad: Tensor, gelu_input: Tensor,
    x: Tensor, weight: Tensor,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, dict]:
    """FP8 linear backward, dgrad only with fused dgelu; wgrad deferred to fp8_linear_bwd_wgrad."""
    shape = x_shape if x_shape is not None else list(x.shape)
    if len(shape) == 3:
        grad_2d = torch.ops.aten.reshape.default(
            grad, [-1, weight.shape[0]],
        )
        gelu_in_2d = torch.ops.aten.reshape.default(
            gelu_input, [-1, weight.shape[0]],
        )
        if cached_fp8_x is None:
            x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        grad_2d = grad
        gelu_in_2d = gelu_input
        if cached_fp8_x is None:
            x_2d = x

    grad_b, fp8_dgelu = tex.dbias_dgelu(
        grad_2d, gelu_in_2d, _bwd_quantizer,
    )
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _bwd_quantizer)

    grad_x_2d, _, _, _ = general_gemm(
        fp8_w, fp8_dgelu,
        out_dtype=torch.bfloat16,
        layout="NN",
    )

    if len(shape) == 3:
        grad_x = torch.ops.aten.reshape.default(grad_x_2d, shape)
    else:
        grad_x = grad_x_2d

    wgrad_ctx = {
        "fp8_grad": fp8_dgelu,
        "grad_b": grad_b,
        "weight_shape": weight.shape,
    }
    if cached_fp8_x is not None:
        wgrad_ctx["fp8_x"] = cached_fp8_x
    else:
        wgrad_ctx["x_2d"] = x_2d
    return grad_x, wgrad_ctx


# Shared-quantization variants: one cast for a tensor that feeds several GEMMs

def release_fwd(cached: object) -> None:
    """No-op: under MXFP8 a single cast serves both fprop and wgrad.

    The seam requires this name to exist here before ``nvfp4_ops`` can rebind
    it (``_MXFP8_IMPL`` is built from ``fp8_ops``' globals). There is nothing
    to release on this path -- one cast, two readers.

    Args:
        cached: The forward feeder's return value. Ignored.
    """


def fp8_quantize_fwd(x: Tensor, *weights: Tensor) -> tuple:
    """Quantize an activation to MXFP8 for reuse across forward GEMMs.

    Returns (fp8_x, orig_shape). weights are ignored; the seam signature requires them.
    """
    shape = list(x.shape)
    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x
    return _quantize(x_2d, _fwd_quantizer), shape


def fp8_quantize_bwd(x: Tensor) -> tuple:
    """Quantize a saved activation to MXFP8 for the backward GEMMs; returns (fp8_x, orig_shape)."""
    shape = list(x.shape)
    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x
    return _quantize(x_2d, _bwd_quantizer), shape


def fp8_gelu_fwd(x: Tensor, *weights: Tensor) -> tuple:
    """Fused tanh-GELU + MXFP8 quantize; returns (fp8_out, orig_shape). weights are ignored."""
    shape = list(x.shape)
    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x
    return tex.gelu(x_2d, _fwd_quantizer), shape


def fp8_bgrad_quantize(grad: Tensor, *weights: Tensor) -> tuple:
    """Fused bias gradient + MXFP8 quantize; returns (grad_b, fp8_grad, orig_shape)."""
    shape = list(grad.shape)
    if len(shape) == 3:
        grad_2d = torch.ops.aten.reshape.default(
            grad, [-1, shape[-1]],
        )
    else:
        grad_2d = grad
    grad_b, fp8_grad = tex.bgrad_quantize(grad_2d, _bwd_quantizer)
    return grad_b, fp8_grad, shape


def fp8_rope_join_bgrad_quantize(
    grad_q_rope: Tensor, grad_k_rope: Tensor, grad_v: Tensor, qkv: Tensor, H: int,
    q_rstd: Tensor, k_rstd: Tensor, qn_weight: Tensor, kn_weight: Tensor,
    grad_qn_weight: Tensor, grad_kn_weight: Tensor, freqs: Tensor, pos_offset: int,
    *weights: Tensor,
) -> tuple:
    """RoPE/QK-norm backward + QKV join, then fp8_bgrad_quantize on the result."""
    grad_qkv = torch.ops.flux.rope_norm_join_ctm(
        grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd, qn_weight,
        kn_weight, grad_qn_weight, grad_kn_weight, freqs, pos_offset,
    )
    return fp8_bgrad_quantize(grad_qkv, *weights)


def fp8_gate_bwd_quantize(
    grad: Tensor, gate: Tensor, proj: Tensor, *weights: Tensor,
    bias_grad: Tensor | None = None,
) -> tuple:
    """Gate backward, then fp8_bgrad_quantize on grad_proj.

    grad, proj: [B, S, D]; gate: [B, 1, D]. When bias_grad is given the bias
    gradient lands in that buffer and is returned as grad_b, so the caller has
    nothing left to accumulate.

    Returns (grad_b, fp8_grad_proj, orig_shape, grad_gate).
    """
    grad_proj, grad_gate = torch.ops.flux.fused_gate_bwd(grad, gate, proj)
    grad_b, fp8_grad, shape = fp8_bgrad_quantize(grad_proj, *weights)
    if bias_grad is not None:
        torch.ops.aten.copy_.default(bias_grad, grad_b)
        grad_b = bias_grad
    return grad_b, fp8_grad, shape, grad_gate


def fp8_qgelu_fwd(x: Tensor, *weights: Tensor) -> tuple:
    """Alias of fp8_gelu_fwd."""
    return fp8_gelu_fwd(x, *weights)


def fp8_dbias_dqgelu_bwd(grad: Tensor, fwd_input: Tensor) -> tuple:
    """Fused dgelu + bias gradient + MXFP8 quantize; returns (grad_b, fp8_dgelu, orig_shape)."""
    shape = list(grad.shape)
    if len(shape) == 3:
        grad_2d = torch.ops.aten.reshape.default(
            grad, [-1, shape[-1]],
        )
        fwd_2d = torch.ops.aten.reshape.default(
            fwd_input, [-1, shape[-1]],
        )
    else:
        grad_2d = grad
        fwd_2d = fwd_input
    grad_b, fp8_dgelu = tex.dbias_dgelu(
        grad_2d, fwd_2d, _bwd_quantizer,
    )
    return grad_b, fp8_dgelu, shape


def fp8_linear_fwd_preq(
    fp8_x, weight: Tensor, bias: Tensor, orig_shape: list,
) -> Tensor:
    """FP8 linear forward with a pre-quantized MXFP8 input; orig_shape restores the output dims."""
    out_features = weight.shape[0]
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _fwd_quantizer)
        _weight_fp8_cache.put(weight, fp8_w)

    out_2d, _, _, _ = general_gemm(
        fp8_w, fp8_x,
        out_dtype=torch.bfloat16,
        layout="TN",
        bias=bias,
    )

    if len(orig_shape) == 3:
        return torch.ops.aten.reshape.default(
            out_2d, [orig_shape[0], orig_shape[1], out_features],
        )
    return out_2d


def fp8_linear_bwd_dgrad_preq_grad(
    fp8_grad, grad_b, weight: Tensor,
    orig_shape: list,
    *,
    cached_fp8_x: object | None = None,
) -> tuple[Tensor, dict]:
    """FP8 linear backward, dgrad only, with a pre-quantized gradient; wgrad deferred."""
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _bwd_quantizer)

    grad_x_2d, _, _, _ = general_gemm(
        fp8_w, fp8_grad,
        out_dtype=torch.bfloat16,
        layout="NN",
    )

    if len(orig_shape) == 3:
        grad_x = torch.ops.aten.reshape.default(
            grad_x_2d, orig_shape,
        )
    else:
        grad_x = grad_x_2d

    wgrad_ctx = {
        "fp8_grad": fp8_grad,
        "grad_b": grad_b,
        "weight_shape": weight.shape,
    }
    if cached_fp8_x is not None:
        wgrad_ctx["fp8_x"] = cached_fp8_x
    return grad_x, wgrad_ctx


def fp8_linear_bwd_preq_grad(
    fp8_grad, x: Tensor, weight: Tensor,
    wgrad_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """FP8 linear backward with a pre-quantized gradient.

    Returns (grad_x, grad_w) in BF16; the bias gradient is the caller's.
    """
    shape = list(x.shape)
    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x

    fp8_x = _quantize(x_2d, _bwd_quantizer)
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _bwd_quantizer)

    grad_x_2d, _, _, _ = general_gemm(
        fp8_w, fp8_grad,
        out_dtype=torch.bfloat16,
        layout="NN",
    )
    grad_w, _, _, _ = general_gemm(
        fp8_x, fp8_grad,
        out_dtype=torch.bfloat16,
        layout="NT",
        out=wgrad_out,
    )

    if len(shape) == 3:
        grad_x = torch.ops.aten.reshape.default(grad_x_2d, shape)
    else:
        grad_x = grad_x_2d
    return grad_x, grad_w


def fp8_linear_bwd_preq_both(
    fp8_grad, fp8_x, weight: Tensor,
    orig_shape: list,
    wgrad_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """FP8 linear backward with both grad and x pre-quantized.

    Returns (grad_x, grad_w) in BF16; the bias gradient is the caller's.
    """
    fp8_w = _weight_fp8_cache.get(weight)
    if fp8_w is None:
        fp8_w = _quantize(weight, _bwd_quantizer)

    grad_x_2d, _, _, _ = general_gemm(
        fp8_w, fp8_grad,
        out_dtype=torch.bfloat16,
        layout="NN",
    )
    grad_w, _, _, _ = general_gemm(
        fp8_x, fp8_grad,
        out_dtype=torch.bfloat16,
        layout="NT",
        out=wgrad_out,
    )

    if len(orig_shape) == 3:
        grad_x = torch.ops.aten.reshape.default(
            grad_x_2d, orig_shape,
        )
    else:
        grad_x = grad_x_2d
    return grad_x, grad_w


# Recipe dispatch. flat.nvfp4_ops is imported only when NVFP4 is requested:
# constructing an NVFP4Quantizer touches the GPU at import time.
def _recipe_flag(name: str, default: bool) -> bool:
    value = _os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={value!r} is not a boolean")


NVFP4_ENABLED = _recipe_flag("FP4", True)

# This module's own MXFP8 implementations, snapshotted before the rebinding
# below; flat.nvfp4_ops uses them for demoted GEMM classes.
_MXFP8_IMPL: dict[str, object] = {}

if _FLAT_BF16:
    from flat import fp8_ops_bf16 as _bf16
    fp8_linear_fwd = _bf16.fp8_linear_fwd
    fp8_linear_fwd_save = _bf16.fp8_linear_fwd_save
    fp8_linear_fwd_gelu = _bf16.fp8_linear_fwd_gelu
    fp8_linear_fwd_preq = _bf16.fp8_linear_fwd_preq
    fp8_linear_bwd = _bf16.fp8_linear_bwd
    fp8_linear_bwd_dgrad = _bf16.fp8_linear_bwd_dgrad
    fp8_linear_bwd_wgrad = _bf16.fp8_linear_bwd_wgrad
    fp8_linear_bwd_dgelu = _bf16.fp8_linear_bwd_dgelu
    fp8_linear_bwd_dgrad_dgelu = _bf16.fp8_linear_bwd_dgrad_dgelu
    fp8_linear_bwd_dgrad_preq_grad = _bf16.fp8_linear_bwd_dgrad_preq_grad
    fp8_linear_bwd_preq_grad = _bf16.fp8_linear_bwd_preq_grad
    fp8_linear_bwd_preq_both = _bf16.fp8_linear_bwd_preq_both
    fp8_quantize_fwd = _bf16.fp8_quantize_fwd
    fp8_quantize_bwd = _bf16.fp8_quantize_bwd
    fp8_gelu_fwd = _bf16.fp8_gelu_fwd
    fp8_bgrad_quantize = _bf16.fp8_bgrad_quantize
    fp8_rope_join_bgrad_quantize = _bf16.fp8_rope_join_bgrad_quantize
    fp8_gate_bwd_quantize = _bf16.fp8_gate_bwd_quantize
    fp8_qgelu_fwd = _bf16.fp8_qgelu_fwd
    fp8_dbias_dqgelu_bwd = _bf16.fp8_dbias_dqgelu_bwd
    release_fwd = _bf16.release_fwd
    clear_fp8_weight_cache = _bf16.clear_fp8_weight_cache
elif NVFP4_ENABLED:
    from flat import nvfp4_ops as _nvfp4
    _MXFP8_IMPL = {name: globals()[name] for name in _nvfp4.SEAM}
    for _name in _nvfp4.SEAM:
        globals()[_name] = getattr(_nvfp4, _name)
    del _name
