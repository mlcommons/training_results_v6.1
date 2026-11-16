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

"""Flat ATen-level FluxModel for MLPerf Flux.1 transformer.

Every module has explicit forward() and backward(); children are called via
self.child.forward(x) / self.child.backward(grad).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from flat.fp8_ops import (
    fp8_linear_fwd,
    fp8_linear_fwd_save,
    fp8_linear_bwd_dgrad_preq_grad,
    fp8_linear_bwd_dgelu,
    fp8_linear_bwd_dgrad_dgelu,
    fp8_quantize_fwd,
    release_fwd,
    fp8_rope_join_bgrad_quantize,
    fp8_gate_bwd_quantize,
    fp8_dbias_dqgelu_bwd,
    fp8_gelu_fwd,
    fp8_linear_fwd_preq,
    fp8_linear_bwd_preq_both,
)
import flat.custom_ops  # noqa: F401  -- registers torch.ops.flux.*
from flat.custom_ops import amax_arena
from flat.custom_ops import aten_tail_ctm as _aten_tail

#: Whether the CUTLASS-DSL aten-tail kernels are live on this device.
_TAIL = _aten_tail.ENABLED

from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    tex,
    fused_attn_fwd, fused_attn_bwd, FusedAttnBackend,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from flat.mxfp8_attn_quant import (
    combine_and_quantize,
    mxfp8_quantize_fast_path,
)


# Standalone ATen helpers

def _linear_fwd(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """ATen linear forward, handles 2D or 3D input."""
    shape = list(x.shape)
    out_features = weight.shape[0]
    if len(shape) == 3:
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        x_2d = x
    wT = torch.ops.aten.t.default(weight)
    out_2d = torch.ops.aten.addmm.default(bias, x_2d, wT)
    if len(shape) == 3:
        return torch.ops.aten.reshape.default(out_2d, [shape[0], shape[1], out_features])
    return out_2d


def _linear_bwd(
    grad: Tensor, x: Tensor, weight: Tensor,
    wgrad_out: Tensor | None = None,
    bias: nn.Parameter | None = None,
):
    """ATen linear backward. Returns (grad_x, grad_w, grad_b). Handles 2D/3D.

    When wgrad_out is provided, grad_w is written into that buffer. When bias is
    provided the bias gradient lands in bias.grad and grad_b comes back None, so
    the caller has nothing left to accumulate.
    """
    shape = list(x.shape)
    if len(shape) == 3:
        grad_2d = torch.ops.aten.reshape.default(grad, [-1, weight.shape[0]])
        x_2d = torch.ops.aten.reshape.default(x, [-1, shape[-1]])
    else:
        grad_2d = grad
        x_2d = x
    grad_x_2d = torch.ops.aten.mm.default(grad_2d, weight)
    grad_T = torch.ops.aten.t.default(grad_2d)
    if wgrad_out is not None:
        torch.ops.aten.mm.out(grad_T, x_2d, out=wgrad_out)
        grad_w = wgrad_out
    else:
        grad_w = torch.ops.aten.mm.default(grad_T, x_2d)
    if bias is not None and _TAIL:
        torch.ops.flux.bias_colsum_ctm(grad_2d, bias.grad)
        grad_b = None
    else:
        grad_b = torch.ops.aten.sum.dim_IntList(grad_2d, [0], True)
        grad_b = torch.ops.aten.reshape.default(grad_b, [weight.shape[0]])
        if bias is not None:
            _accum_grad(bias, grad_b)
            grad_b = None
    if len(shape) == 3:
        grad_x = torch.ops.aten.reshape.default(grad_x_2d, shape)
    else:
        grad_x = grad_x_2d
    return grad_x, grad_w, grad_b


def _rms_norm_fwd(x: Tensor, weight: Tensor, normalized_shape: list, eps: float = 1e-6):
    """RMSNorm forward using TE's fused kernel. Returns (output, rsigma)."""
    D = normalized_shape[-1]
    orig_shape = x.shape
    x_2d = x.reshape(-1, D) if x.ndim > 2 else x
    ln_out = torch.empty_like(x_2d)
    result = tex.rmsnorm_fwd(
        x_2d, weight, eps, ln_out, None, tex.DType.kBFloat16, 0, False,
    )
    normed = result[0].reshape(orig_shape) if x.ndim > 2 else result[0]
    return normed, result[2]  # result[2] is rsigma


def _rms_norm_bwd(grad: Tensor, x: Tensor, normalized_shape: list,
                  rsigma: Tensor, weight: Tensor):
    """RMSNorm backward using TE's fused kernel. Returns (grad_x, grad_weight)."""
    D = normalized_shape[-1]
    orig_shape = x.shape
    grad_2d = grad.reshape(-1, D) if grad.ndim > 2 else grad
    x_2d = x.reshape(-1, D) if x.ndim > 2 else x
    result = tex.rmsnorm_bwd(grad_2d, x_2d, rsigma, weight, 0, False)
    dx = result[0].reshape(orig_shape) if x.ndim > 2 else result[0]
    return dx, result[1]


import os as _os


def _parse_attn_precision() -> str:
    """Resolve the attention-core format from the ``FP8_ATTENTION`` flag.

    Returns ``"bfloat16"`` (the default) or ``"fp8"``, meaning MXFP8.
    """
    value = _os.environ.get("FP8_ATTENTION", "").strip().lower()
    if not value or value in ("0", "false", "no", "off"):
        return "bfloat16"
    if value in ("1", "true", "yes", "on"):
        return "fp8"
    raise ValueError(f"FP8_ATTENTION={value!r} is not a boolean")


_ATTN_PRECISION = _parse_attn_precision()


def attn_precision() -> str:
    """The attention-core format this configuration selected."""
    return _ATTN_PRECISION

_FP8_ATTENTION = _ATTN_PRECISION == "fp8"
_FUSED_ATTN_BACKEND = FusedAttnBackend["F16_arbitrary_seqlen"]
_FP8_ATTN_BACKEND = FusedAttnBackend["FP8"]

# Cache for cu_seqlens tensors (keyed by (B, S, device))
_cu_seqlens_cache: dict[tuple, Tensor] = {}


def _get_cu_seqlens(B: int, S: int, device: torch.device) -> Tensor:
    """Get or create cu_seqlens for fixed-length attention."""
    key = (B, S, device)
    t = _cu_seqlens_cache.get(key)
    if t is None:
        t = torch.arange(
            0, (B + 1) * S, S, dtype=torch.int32, device=device,
        )
        _cu_seqlens_cache[key] = t
    return t


# MXFP8 attention quantizers (lazy-init)
_mxfp8_attn_q: dict[str, MXFP8Quantizer] | None = None


def _init_mxfp8_attn() -> dict[str, MXFP8Quantizer]:
    """Build the MXFP8 quantizers for the attention core.

    Returns quantizers keyed by role: ``qkv``, ``o``, ``do``, ``dqkv``.
    """
    global _mxfp8_attn_q
    if _mxfp8_attn_q is not None:
        return _mxfp8_attn_q
    quantizers: dict[str, MXFP8Quantizer] = {}
    for name, dtype, optimize_for_gemm in (
        ("qkv", tex.DType.kFloat8E4M3, True),
        ("o", tex.DType.kFloat8E4M3, False),
        ("do", tex.DType.kFloat8E5M2, True),
        ("dqkv", tex.DType.kFloat8E5M2, False),
    ):
        q = MXFP8Quantizer(fp8_dtype=dtype)
        q.set_usage(rowwise=True, columnwise=True)
        q.internal = False
        q.optimize_for_gemm = optimize_for_gemm
        quantizers[name] = q
    _mxfp8_attn_q = quantizers
    return quantizers


# Attention forward/backward

def _sdpa_fwd(q: Tensor, k: Tensor, v: Tensor):
    """Scaled dot-product attention forward (BSHD format).

    Q, K, V are [B, S, H, Dh]; the output has the same shape and is BF16 even
    when the attention core runs in MXFP8.
    """
    B, S, H, Dh = q.shape
    scale = Dh ** -0.5
    cu_seqlens = _get_cu_seqlens(B, S, q.device)

    if _FP8_ATTENTION:
        aq = _init_mxfp8_attn()
        q_mx, k_mx, v_mx, layout, scale_inv_format = combine_and_quantize(
            "bshd_bshd_bshd", q, k, v, aq["qkv"], used_in_backward=True,
        )
        attn_out, aux_ctx = fused_attn_fwd(
            is_training=True,
            max_seqlen_q=S,
            max_seqlen_kv=S,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            q=q_mx, k=k_mx, v=v_mx,
            fake_dtype=q.dtype,
            fused_attention_backend=_FP8_ATTN_BACKEND,
            qkv_layout=layout,
            o_format="bshd",
            qkv_scale_inv_format=scale_inv_format,
            attn_mask_type="no_mask",
            attn_scale=scale,
            s_quantizer=None,
            o_quantizer=aq["o"],
            cuda_graph=torch.cuda.is_current_stream_capturing(),
        )
        # Under MXFP8 the kernel emits BF16; only Q/K/V stay quantized.
        saved = (q_mx, k_mx, v_mx, attn_out, aux_ctx, cu_seqlens, S, scale,
                 layout, scale_inv_format)
        return attn_out, saved

    attn_out, aux_ctx = fused_attn_fwd(
        is_training=True,
        max_seqlen_q=S,
        max_seqlen_kv=S,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        q=q, k=k, v=v,
        fake_dtype=q.dtype,
        fused_attention_backend=_FUSED_ATTN_BACKEND,
        qkv_layout="bshd_bshd_bshd",
        o_format="bshd",
        attn_mask_type="no_mask",
        attn_scale=scale,
    )

    saved = (q, k, v, attn_out, aux_ctx, cu_seqlens, S, scale,
             "bshd_bshd_bshd", None)
    return attn_out, saved


def _sdpa_bwd(grad_output: Tensor, saved: tuple):
    """SDPA backward (BSHD format). Returns (grad_q, grad_k, grad_v)."""
    (q, k, v, attn_out, aux_ctx, cu_seqlens, S, scale, layout,
     scale_inv_format) = saved

    if _FP8_ATTENTION:
        aq = _init_mxfp8_attn()
        # q, k, v are the MXFP8Tensors saved by forward; only grad_output
        # (BF16) is quantized here.
        (d_o_mx,), do_scale_inv_format = mxfp8_quantize_fast_path(
            [(grad_output, aq["do"])], "bshd",
        )
        # The MXFP8 kernel also takes the BF16 output and incoming gradient.
        dq, dk, dv, *_ = fused_attn_bwd(
            max_seqlen_q=S,
            max_seqlen_kv=S,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            q=q, k=k, v=v,
            o=attn_out, d_o=d_o_mx,
            fake_dtype=torch.bfloat16,
            aux_ctx_tensors=list(aux_ctx) + [grad_output],
            fused_attention_backend=_FP8_ATTN_BACKEND,
            qkv_layout=layout,
            o_format="bshd",
            do_format="bshd",
            dqkv_layout="bshd_bshd_bshd",
            qkv_scale_inv_format=scale_inv_format,
            do_scale_inv_format=do_scale_inv_format,
            attn_mask_type="no_mask",
            attn_scale=scale,
            s_quantizer=None,
            dp_quantizer=None,
            dqkv_quantizer=aq["dqkv"],
            cuda_graph=torch.cuda.is_current_stream_capturing(),
        )
        # Under MXFP8 dq/dk/dv come back as plain BF16 tensors.
        return dq, dk, dv

    dq, dk, dv, *_ = fused_attn_bwd(
        max_seqlen_q=S,
        max_seqlen_kv=S,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        q=q, k=k, v=v, o=attn_out, d_o=grad_output,
        fake_dtype=q.dtype,
        aux_ctx_tensors=aux_ctx,
        fused_attention_backend=_FUSED_ATTN_BACKEND,
        qkv_layout="bshd_bshd_bshd",
        o_format="bshd",
        do_format="bshd",
        dqkv_layout="bshd_bshd_bshd",
        attn_mask_type="no_mask",
        attn_scale=scale,
    )
    return dq, dk, dv


# Gradient accumulation helpers

def _accum_grad(param: nn.Parameter, grad: Tensor) -> None:
    """Copy gradient into pre-allocated buffer view.

    Requires param.grad to be a view into the grad buffer (assign_grad_views).
    """
    torch.ops.aten.copy_.default(param.grad, grad)


def _copy_grad_f32(param: nn.Parameter, src_f32: Tensor) -> None:
    """Copy FP32 scratch buffer to BF16 grad in one kernel (implicit cast)."""
    torch.ops.aten.copy_.default(param.grad, src_f32)


# CUTLASS-DSL aten-tail seams. Each falls back to the ATen/TE form when the DSL
# kernels are not live on this device, so the two paths stay interchangeable.

def _bias_grad(param: nn.Parameter, grad_2d: Tensor) -> None:
    """Column-sum of grad_2d straight into param.grad."""
    if _TAIL:
        torch.ops.flux.bias_colsum_ctm(grad_2d, param.grad)
    else:
        gb = torch.ops.aten.sum.dim_IntList(grad_2d, [0], True)
        _accum_grad(param, torch.ops.aten.reshape.default(gb, [param.numel()]))


def _silu(x: Tensor) -> Tensor:
    return torch.ops.flux.silu_ctm(x) if _TAIL else tex.silu(x, None)


def _dsilu(grad: Tensor, x: Tensor) -> Tensor:
    return torch.ops.flux.dsilu_ctm(grad, x) if _TAIL else tex.dsilu(grad, x, None)


def _qknorm_fin(
    sq: Tensor, sk: Tensor, qn: nn.Parameter, kn: nn.Parameter,
) -> None:
    """Finalise both QK-norm weight gradients from their FP32 scratch buffers.

    The DSL kernel also zeroes the scratch for the next step, which is why the
    callers' explicit scratch zero_() is skipped when it is live.
    """
    if _TAIL:
        torch.ops.flux.qknorm_grad_fin_ctm(sq, sk, qn.grad, kn.grad)
    else:
        _copy_grad_f32(qn, sq)
        _copy_grad_f32(kn, sk)


def _bias_grad_view(param: nn.Parameter) -> Tensor | None:
    """The buffer a quantize seam should write the bias gradient into, or None."""
    return param.grad if _TAIL else None


def _defer_bias(param: nn.Parameter) -> nn.Parameter | None:
    """None when the bias gradient is already final, so the bucket skips it."""
    return None if _TAIL else param


def _accum_bias_grad(param: nn.Parameter, gb: Tensor) -> None:
    """Accumulate gb only when the seam did not already write param.grad."""
    if not _TAIL:
        _accum_grad(param, gb)


def _dsilu_accum_mod(
    acc: Tensor, gs0: Tensor, x0: Tensor,
    gs1: Tensor | None, x1: Tensor | None,
) -> None:
    """dsilu of one or two modulation branches, accumulated into acc in one kernel."""
    torch.ops.flux.dsilu_accum_ctm(acc, gs0, x0, gs1, x1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.in_layer = nn.Module()
        self.in_layer.weight = nn.Parameter(torch.empty(hidden_dim, in_dim))
        self.in_layer.bias = nn.Parameter(torch.empty(hidden_dim))
        self.out_layer = nn.Module()
        self.out_layer.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.out_layer.bias = nn.Parameter(torch.empty(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        wT_in = torch.ops.aten.t.default(self.in_layer.weight)
        addmm_out = torch.ops.aten.addmm.default(self.in_layer.bias, x, wT_in)
        silu_out = _silu(addmm_out)
        wT_out = torch.ops.aten.t.default(self.out_layer.weight)
        result = torch.ops.aten.addmm.default(self.out_layer.bias, silu_out, wT_out)
        self._saved_x = x
        self._saved_addmm_out = addmm_out
        self._saved_silu_out = silu_out
        return result

    def backward(self, grad_output: Tensor) -> Tensor:
        with torch.no_grad():
            x = self._saved_x
            addmm_out = self._saved_addmm_out
            silu_out = self._saved_silu_out

            # out_layer backward
            grad_silu = torch.ops.aten.mm.default(grad_output, self.out_layer.weight)
            grad_out_T = torch.ops.aten.t.default(grad_output)
            torch.ops.aten.mm.out(grad_out_T, silu_out, out=self.out_layer.weight.grad)
            _bias_grad(self.out_layer.bias, grad_output)

            # silu backward
            grad_addmm = _dsilu(grad_silu, addmm_out)

            # in_layer backward
            grad_x = torch.ops.aten.mm.default(grad_addmm, self.in_layer.weight)
            grad_in_T = torch.ops.aten.t.default(grad_addmm)
            torch.ops.aten.mm.out(grad_in_T, x, out=self.in_layer.weight.grad)
            _bias_grad(self.in_layer.bias, grad_addmm)

            del self._saved_x, self._saved_addmm_out, self._saved_silu_out
            return grad_x


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.dim = dim
        self.lin = nn.Module()
        self.lin.weight = nn.Parameter(torch.empty(self.multiplier * dim, dim))
        self.lin.bias = nn.Parameter(torch.empty(self.multiplier * dim))

    def forward(self, vec: Tensor) -> tuple:
        silu_out = _silu(vec)
        wT = torch.ops.aten.t.default(self.lin.weight)
        lin_out = torch.ops.aten.addmm.default(self.lin.bias, silu_out, wT)
        unsqueezed = torch.ops.aten.unsqueeze.default(lin_out, 1)
        chunks = torch.ops.aten.split.Tensor(unsqueezed, self.dim, -1)
        self._saved_vec = vec
        self._saved_silu_out = silu_out
        return chunks

    def backward_lin(self, grad_output: Tensor) -> tuple:
        """The lin half of the backward; returns (grad_silu, vec) for the dsilu tail.

        Split out so a caller holding two Modulations can feed both into a single
        fused dsilu-accumulate instead of one dsilu each.
        """
        with torch.no_grad():
            vec = self._saved_vec
            silu_out = self._saved_silu_out

            # squeeze [B, 1, M*D] -> [B, M*D]
            grad_lin = torch.ops.aten.reshape.default(
                grad_output, [-1, self.multiplier * self.dim]
            )
            # lin backward
            grad_silu = torch.ops.aten.mm.default(grad_lin, self.lin.weight)
            grad_lin_T = torch.ops.aten.t.default(grad_lin)
            torch.ops.aten.mm.out(grad_lin_T, silu_out, out=self.lin.weight.grad)
            _bias_grad(self.lin.bias, grad_lin)
            del self._saved_vec, self._saved_silu_out
            return grad_silu, vec

    def backward(self, grad_output: Tensor) -> Tensor:
        """grad_output: [B, 1, multiplier*dim] (concatenated chunk grads)."""
        grad_silu, vec = self.backward_lin(grad_output)
        with torch.no_grad():
            return _dsilu(grad_silu, vec)


class DoubleStreamBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int,
        mlp_ratio: float = 4.0, qkv_bias: bool = False,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.qkv_bias_flag = qkv_bias

        self.img_mod = Modulation(dim=hidden_size, double=True)
        self.txt_mod = Modulation(dim=hidden_size, double=True)

        self.img_attn = nn.Module()
        self.img_attn.qkv = nn.Module()
        self.img_attn.qkv.weight = nn.Parameter(torch.empty(hidden_size * 3, hidden_size))
        if qkv_bias:
            self.img_attn.qkv.bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.img_attn.norm = nn.Module()
        self.img_attn.norm.query_norm = nn.Module()
        self.img_attn.norm.query_norm.weight = nn.Parameter(torch.empty(self.head_dim))
        self.img_attn.norm.key_norm = nn.Module()
        self.img_attn.norm.key_norm.weight = nn.Parameter(torch.empty(self.head_dim))
        self.img_attn.proj = nn.Module()
        self.img_attn.proj.weight = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.img_attn.proj.bias = nn.Parameter(torch.empty(hidden_size))

        self.txt_attn = nn.Module()
        self.txt_attn.qkv = nn.Module()
        self.txt_attn.qkv.weight = nn.Parameter(torch.empty(hidden_size * 3, hidden_size))
        if qkv_bias:
            self.txt_attn.qkv.bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.txt_attn.norm = nn.Module()
        self.txt_attn.norm.query_norm = nn.Module()
        self.txt_attn.norm.query_norm.weight = nn.Parameter(torch.empty(self.head_dim))
        self.txt_attn.norm.key_norm = nn.Module()
        self.txt_attn.norm.key_norm.weight = nn.Parameter(torch.empty(self.head_dim))
        self.txt_attn.proj = nn.Module()
        self.txt_attn.proj.weight = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.txt_attn.proj.bias = nn.Parameter(torch.empty(hidden_size))

        # img_mlp: Linear -> GELU(tanh) -> Linear
        self.img_mlp = nn.Module()
        img_mlp_0 = nn.Module()
        img_mlp_0.weight = nn.Parameter(torch.empty(mlp_hidden_dim, hidden_size))
        img_mlp_0.bias = nn.Parameter(torch.empty(mlp_hidden_dim))
        self.img_mlp.register_module("0", img_mlp_0)
        img_mlp_2 = nn.Module()
        img_mlp_2.weight = nn.Parameter(torch.empty(hidden_size, mlp_hidden_dim))
        img_mlp_2.bias = nn.Parameter(torch.empty(hidden_size))
        self.img_mlp.register_module("2", img_mlp_2)
        object.__setattr__(self, "_img_mlp_0", self.img_mlp._modules["0"])
        object.__setattr__(self, "_img_mlp_2", self.img_mlp._modules["2"])

        self.txt_mlp = nn.Module()
        txt_mlp_0 = nn.Module()
        txt_mlp_0.weight = nn.Parameter(torch.empty(mlp_hidden_dim, hidden_size))
        txt_mlp_0.bias = nn.Parameter(torch.empty(mlp_hidden_dim))
        self.txt_mlp.register_module("0", txt_mlp_0)
        txt_mlp_2 = nn.Module()
        txt_mlp_2.weight = nn.Parameter(torch.empty(hidden_size, mlp_hidden_dim))
        txt_mlp_2.bias = nn.Parameter(torch.empty(hidden_size))
        self.txt_mlp.register_module("2", txt_mlp_2)
        object.__setattr__(self, "_txt_mlp_0", self.txt_mlp._modules["0"])
        object.__setattr__(self, "_txt_mlp_2", self.txt_mlp._modules["2"])

    def alloc_scratch_buffers(self) -> None:
        """Pre-allocate FP32 scratch buffers for norm weight gradients.

        Call after the model is on the GPU.
        """
        dev = self.img_attn.proj.weight.device
        Dh = self.head_dim
        self._scratch = torch.zeros(4, Dh, dtype=torch.float32, device=dev)
        self._scratch_img_qn = self._scratch[0]
        self._scratch_img_kn = self._scratch[1]
        self._scratch_txt_qn = self._scratch[2]
        self._scratch_txt_kn = self._scratch[3]

    def _get_qkv_bias(self, which: str) -> Tensor:
        """Return qkv bias or zeros if no bias."""
        attn = getattr(self, which + "_attn")
        bias = getattr(attn.qkv, "bias", None)
        if bias is not None:
            return bias
        return torch.ops.aten.zeros.default(
            [self.hidden_size * 3],
            dtype=self.img_attn.proj.weight.dtype,
            device=self.img_attn.proj.weight.device,
        )

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor,
    ) -> tuple:
        B = img.shape[0]
        L_img = img.shape[1]
        L_txt = txt.shape[1]
        D = self.hidden_size
        H = self.num_heads
        Dh = self.head_dim

        img_input = img
        txt_input = txt

        # Modulation
        img_mod_chunks = self.img_mod.forward(vec)
        txt_mod_chunks = self.txt_mod.forward(vec)
        img_shift1, img_scale1, img_gate1 = img_mod_chunks[0], img_mod_chunks[1], img_mod_chunks[2]
        img_shift2, img_scale2, img_gate2 = img_mod_chunks[3], img_mod_chunks[4], img_mod_chunks[5]
        txt_shift1, txt_scale1, txt_gate1 = txt_mod_chunks[0], txt_mod_chunks[1], txt_mod_chunks[2]
        txt_shift2, txt_scale2, txt_gate2 = txt_mod_chunks[3], txt_mod_chunks[4], txt_mod_chunks[5]

        # img_norm1 + modulate (fused)
        img_modulated, img_mean1, img_rstd1 = torch.ops.flux.fused_ln_adaln(
            img, img_scale1, img_shift1,
        )

        # img QKV
        img_qkv_bias = self._get_qkv_bias("img")
        img_qkv, fp8_img_mod = fp8_linear_fwd_save(img_modulated, self.img_attn.qkv.weight, img_qkv_bias)

        # txt_norm1 + modulate (fused)
        txt_modulated, txt_mean1, txt_rstd1 = torch.ops.flux.fused_ln_adaln(
            txt, txt_scale1, txt_shift1,
        )

        # txt QKV
        txt_qkv_bias = self._get_qkv_bias("txt")
        txt_qkv, fp8_txt_mod = fp8_linear_fwd_save(txt_modulated, self.txt_attn.qkv.weight, txt_qkv_bias)

        # Combined [txt, img] Q/K/V buffers, written in place below.
        L_total = L_txt + L_img
        q_rope = torch.ops.aten.empty.memory_format(
            [B, L_total, H, Dh], dtype=img_qkv.dtype,
            device=img_qkv.device, memory_format=torch.contiguous_format,
        )
        k_rope = torch.ops.aten.empty.memory_format(
            [B, L_total, H, Dh], dtype=img_qkv.dtype,
            device=img_qkv.device, memory_format=torch.contiguous_format,
        )
        v_attn = torch.ops.aten.empty.memory_format(
            [B, L_total, H, Dh], dtype=img_qkv.dtype,
            device=img_qkv.device, memory_format=torch.contiguous_format,
        )

        # img -> combined[:, L_txt:, :, :]
        img_q_rstd, img_k_rstd = (
            torch.ops.flux.qkv_split_norm_rope_inplace_ctm(
                img_qkv, H,
                self.img_attn.norm.query_norm.weight,
                self.img_attn.norm.key_norm.weight,
                pe, L_txt,  # img positions start at L_txt
                q_rope[:, L_txt:, :, :],
                k_rope[:, L_txt:, :, :],
                v_attn[:, L_txt:, :, :],
            )
        )

        # txt -> combined[:, :L_txt, :, :]
        txt_q_rstd, txt_k_rstd = (
            torch.ops.flux.qkv_split_norm_rope_inplace_ctm(
                txt_qkv, H,
                self.txt_attn.norm.query_norm.weight,
                self.txt_attn.norm.key_norm.weight,
                pe, 0,  # txt positions start at 0
                q_rope[:, :L_txt, :, :],
                k_rope[:, :L_txt, :, :],
                v_attn[:, :L_txt, :, :],
            )
        )
        attn_out, sdpa_saved = _sdpa_fwd(q_rope, k_rope, v_attn)
        attn_flat = torch.ops.aten.reshape.default(attn_out, [B, L_total, D])

        txt_attn_out, img_attn_out = torch.ops.flux.seq_split_ctm3(attn_flat, L_txt)

        # img proj + residual
        img_proj, fp8_img_attn_out = fp8_linear_fwd_save(img_attn_out, self.img_attn.proj.weight, self.img_attn.proj.bias)
        img = torch.ops.flux.fused_gate_residual_a2(img_input, img_gate1, img_proj)

        # img MLP branch
        img_after_attn = img  # save for backward
        img_mlp_in, img_mean2, img_rstd2 = torch.ops.flux.fused_ln_adaln(
            img, img_scale2, img_shift2,
        )
        img_mlp_1, fp8_img_mlp_in = fp8_linear_fwd_save(img_mlp_in, self._img_mlp_0.weight, self._img_mlp_0.bias)
        # GELU + quantize
        fp8_img_mlp_act, img_mlp_act_shape = fp8_gelu_fwd(
            img_mlp_1, self._img_mlp_2.weight,
        )
        img_mlp_out = fp8_linear_fwd_preq(fp8_img_mlp_act, self._img_mlp_2.weight, self._img_mlp_2.bias, img_mlp_act_shape)
        release_fwd(fp8_img_mlp_act)  # fc2 was its only forward reader
        img = torch.ops.flux.fused_gate_residual_a2(
            img_after_attn, img_gate2, img_mlp_out,
        )

        # txt proj + residual
        txt_proj, fp8_txt_attn_out = fp8_linear_fwd_save(txt_attn_out, self.txt_attn.proj.weight, self.txt_attn.proj.bias)
        txt = torch.ops.flux.fused_gate_residual_a2(txt_input, txt_gate1, txt_proj)

        # txt MLP branch
        txt_after_attn = txt
        txt_mlp_in, txt_mean2, txt_rstd2 = torch.ops.flux.fused_ln_adaln(
            txt, txt_scale2, txt_shift2,
        )
        txt_mlp_1, fp8_txt_mlp_in = fp8_linear_fwd_save(txt_mlp_in, self._txt_mlp_0.weight, self._txt_mlp_0.bias)
        # GELU + quantize
        fp8_txt_mlp_act, txt_mlp_act_shape = fp8_gelu_fwd(
            txt_mlp_1, self._txt_mlp_2.weight,
        )
        txt_mlp_out = fp8_linear_fwd_preq(fp8_txt_mlp_act, self._txt_mlp_2.weight, self._txt_mlp_2.bias, txt_mlp_act_shape)
        release_fwd(fp8_txt_mlp_act)  # fc2 was its only forward reader
        txt = torch.ops.flux.fused_gate_residual_a2(
            txt_after_attn, txt_gate2, txt_mlp_out,
        )

        # Save for backward
        self._saved_fwd = {
            "img_input": img_input, "txt_input": txt_input, "vec": vec, "pe": pe,
            "B": B, "L_img": L_img, "L_txt": L_txt,
            "img_mean1": img_mean1, "img_rstd1": img_rstd1,
            "img_scale1": img_scale1, "img_shift1": img_shift1, "img_gate1": img_gate1,
            "img_scale2": img_scale2, "img_shift2": img_shift2, "img_gate2": img_gate2,
            "txt_mean1": txt_mean1, "txt_rstd1": txt_rstd1,
            "txt_scale1": txt_scale1, "txt_shift1": txt_shift1, "txt_gate1": txt_gate1,
            "txt_scale2": txt_scale2, "txt_shift2": txt_shift2, "txt_gate2": txt_gate2,
            "img_proj": img_proj, "txt_proj": txt_proj,
            "img_after_attn": img_after_attn, "txt_after_attn": txt_after_attn,
            "img_mean2": img_mean2, "img_rstd2": img_rstd2,
            "img_mlp_in_shape": list(img_mlp_in.shape),
            "img_mlp_1": img_mlp_1,
            "img_mlp_out": img_mlp_out,
            "txt_mean2": txt_mean2, "txt_rstd2": txt_rstd2,
            "txt_mlp_in_shape": list(txt_mlp_in.shape),
            "txt_mlp_1": txt_mlp_1,
            "txt_mlp_out": txt_mlp_out,
            # pre-norm Q/K for backward are qkv[..., :D] / qkv[..., D:2D]
            "img_qkv": img_qkv, "txt_qkv": txt_qkv,
            "img_q_rstd": img_q_rstd, "img_k_rstd": img_k_rstd,
            "txt_q_rstd": txt_q_rstd, "txt_k_rstd": txt_k_rstd,
            "sdpa_saved": sdpa_saved,
            "fp8_img_mod": fp8_img_mod, "fp8_txt_mod": fp8_txt_mod,
            "img_mod_shape": list(img_modulated.shape),
            "txt_mod_shape": list(txt_modulated.shape),
            "fp8_img_attn_out": fp8_img_attn_out,
            "fp8_txt_attn_out": fp8_txt_attn_out,
            "img_attn_out_shape": list(img_attn_out.shape),
            "txt_attn_out_shape": list(txt_attn_out.shape),
            "fp8_img_mlp_in": fp8_img_mlp_in,
            "fp8_img_mlp_act": fp8_img_mlp_act,
            "img_mlp_act_shape": img_mlp_act_shape,
            "fp8_txt_mlp_in": fp8_txt_mlp_in,
            "fp8_txt_mlp_act": fp8_txt_mlp_act,
            "txt_mlp_act_shape": txt_mlp_act_shape,
        }
        return img, txt

    def backward(
        self,
        grad_img: Tensor,
        grad_txt: Tensor,
        wgrad_mgr: "BucketManager | None" = None,
        grad_vec_acc: Tensor | None = None,
    ) -> tuple:
        with torch.no_grad():
            s = self._saved_fwd
            B = s["B"]
            L_img = s["L_img"]
            L_txt = s["L_txt"]
            D = self.hidden_size
            H = self.num_heads
            Dh = self.head_dim

            # txt MLP branch backward
            grad_txt_gated_mlp = grad_txt
            gb, fp8_grad_txt_mlp_out, _, grad_txt_gate2_contrib = (
                fp8_gate_bwd_quantize(
                    grad_txt_gated_mlp, s["txt_gate2"], s["txt_mlp_out"],
                    self._txt_mlp_2.weight,
                    bias_grad=_bias_grad_view(self._txt_mlp_2.bias),
                )
            )

            if wgrad_mgr is not None:
                grad_txt_mlp_act, wctx = fp8_linear_bwd_dgrad_preq_grad(
                    fp8_grad_txt_mlp_out, gb,
                    self._txt_mlp_2.weight,
                    s["txt_mlp_act_shape"],
                    cached_fp8_x=s["fp8_txt_mlp_act"],
                )
                wgrad_mgr.defer_wgrad(
                    self._txt_mlp_2.weight,
                    _defer_bias(self._txt_mlp_2.bias), wctx,
                )
            else:
                grad_txt_mlp_act, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_txt_mlp_out, s["fp8_txt_mlp_act"],
                    self._txt_mlp_2.weight,
                    s["txt_mlp_act_shape"],
                    wgrad_out=self._txt_mlp_2.weight.grad,
                )
                _accum_bias_grad(self._txt_mlp_2.bias, gb)
            if wgrad_mgr is not None:
                grad_txt_mlp_in, wctx = fp8_linear_bwd_dgrad_dgelu(
                    grad_txt_mlp_act, s["txt_mlp_1"],
                    None, self._txt_mlp_0.weight,
                    cached_fp8_x=s["fp8_txt_mlp_in"],
                    x_shape=s["txt_mlp_in_shape"],
                )
                wgrad_mgr.defer_wgrad(
                    self._txt_mlp_0.weight,
                    self._txt_mlp_0.bias, wctx,
                )
            else:
                grad_txt_mlp_in, _, gb = fp8_linear_bwd_dgelu(
                    grad_txt_mlp_act, s["txt_mlp_1"],
                    None, self._txt_mlp_0.weight,
                    wgrad_out=self._txt_mlp_0.weight.grad,
                    cached_fp8_x=s["fp8_txt_mlp_in"],
                    x_shape=s["txt_mlp_in_shape"],
                )
                _accum_grad(self._txt_mlp_0.bias, gb)

            # txt_mlp_in = (1+scale2) * LN(txt_after_attn) + shift2
            grad_txt_after_attn, grad_txt_scale2 = torch.ops.flux.fused_ln_adaln_bwd(
                grad_txt_mlp_in, s["txt_after_attn"], s["txt_scale2"],
                s["txt_mean2"], s["txt_rstd2"], L_txt, grad_txt,
            )
            grad_txt_shift2 = grad_txt_mlp_in

            # txt proj + residual backward
            grad_txt_gated_proj = grad_txt_after_attn
            gb, fp8_grad_txt_proj, _, grad_txt_gate1_contrib = (
                fp8_gate_bwd_quantize(
                    grad_txt_gated_proj, s["txt_gate1"], s["txt_proj"],
                    self.txt_attn.proj.weight,
                    bias_grad=_bias_grad_view(self.txt_attn.proj.bias),
                )
            )

            if wgrad_mgr is not None:
                grad_txt_attn_out, wctx = fp8_linear_bwd_dgrad_preq_grad(
                    fp8_grad_txt_proj, gb,
                    self.txt_attn.proj.weight,
                    s["txt_attn_out_shape"],
                    cached_fp8_x=s["fp8_txt_attn_out"],
                )
                wgrad_mgr.defer_wgrad(
                    self.txt_attn.proj.weight,
                    _defer_bias(self.txt_attn.proj.bias), wctx,
                )
            else:
                grad_txt_attn_out, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_txt_proj, s["fp8_txt_attn_out"],
                    self.txt_attn.proj.weight,
                    s["txt_attn_out_shape"],
                    wgrad_out=self.txt_attn.proj.weight.grad,
                )
                _accum_bias_grad(self.txt_attn.proj.bias, gb)

            grad_txt_input = grad_txt_after_attn

            # img MLP branch backward
            grad_img_gated_mlp = grad_img
            gb, fp8_grad_img_mlp_out, _, grad_img_gate2_contrib = (
                fp8_gate_bwd_quantize(
                    grad_img_gated_mlp, s["img_gate2"], s["img_mlp_out"],
                    self._img_mlp_2.weight,
                    bias_grad=_bias_grad_view(self._img_mlp_2.bias),
                )
            )

            if wgrad_mgr is not None:
                grad_img_mlp_act, wctx = fp8_linear_bwd_dgrad_preq_grad(
                    fp8_grad_img_mlp_out, gb,
                    self._img_mlp_2.weight,
                    s["img_mlp_act_shape"],
                    cached_fp8_x=s["fp8_img_mlp_act"],
                )
                wgrad_mgr.defer_wgrad(
                    self._img_mlp_2.weight,
                    _defer_bias(self._img_mlp_2.bias), wctx,
                )
            else:
                grad_img_mlp_act, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_img_mlp_out, s["fp8_img_mlp_act"],
                    self._img_mlp_2.weight,
                    s["img_mlp_act_shape"],
                    wgrad_out=self._img_mlp_2.weight.grad,
                )
                _accum_bias_grad(self._img_mlp_2.bias, gb)
            if wgrad_mgr is not None:
                grad_img_mlp_in, wctx = fp8_linear_bwd_dgrad_dgelu(
                    grad_img_mlp_act, s["img_mlp_1"],
                    None, self._img_mlp_0.weight,
                    cached_fp8_x=s["fp8_img_mlp_in"],
                    x_shape=s["img_mlp_in_shape"],
                )
                wgrad_mgr.defer_wgrad(
                    self._img_mlp_0.weight,
                    self._img_mlp_0.bias, wctx,
                )
            else:
                grad_img_mlp_in, _, gb = fp8_linear_bwd_dgelu(
                    grad_img_mlp_act, s["img_mlp_1"],
                    None, self._img_mlp_0.weight,
                    wgrad_out=self._img_mlp_0.weight.grad,
                    cached_fp8_x=s["fp8_img_mlp_in"],
                    x_shape=s["img_mlp_in_shape"],
                )
                _accum_grad(self._img_mlp_0.bias, gb)

            grad_img_after_attn, grad_img_scale2 = torch.ops.flux.fused_ln_adaln_bwd(
                grad_img_mlp_in, s["img_after_attn"], s["img_scale2"],
                s["img_mean2"], s["img_rstd2"], L_img, grad_img,
            )
            grad_img_shift2 = grad_img_mlp_in

            # img proj + residual backward
            grad_img_gated_proj = grad_img_after_attn
            gb, fp8_grad_img_proj, _, grad_img_gate1_contrib = (
                fp8_gate_bwd_quantize(
                    grad_img_gated_proj, s["img_gate1"], s["img_proj"],
                    self.img_attn.proj.weight,
                    bias_grad=_bias_grad_view(self.img_attn.proj.bias),
                )
            )

            if wgrad_mgr is not None:
                grad_img_attn_out, wctx = fp8_linear_bwd_dgrad_preq_grad(
                    fp8_grad_img_proj, gb,
                    self.img_attn.proj.weight,
                    s["img_attn_out_shape"],
                    cached_fp8_x=s["fp8_img_attn_out"],
                )
                wgrad_mgr.defer_wgrad(
                    self.img_attn.proj.weight,
                    _defer_bias(self.img_attn.proj.bias), wctx,
                )
            else:
                grad_img_attn_out, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_img_proj, s["fp8_img_attn_out"],
                    self.img_attn.proj.weight,
                    s["img_attn_out_shape"],
                    wgrad_out=self.img_attn.proj.weight.grad,
                )
                _accum_bias_grad(self.img_attn.proj.bias, gb)

            grad_img_input = grad_img_after_attn

            # Attention backward (BSHD)
            L_total = L_txt + L_img
            grad_attn_flat = torch.ops.flux.seq_join_ctm3(
                grad_txt_attn_out, grad_img_attn_out,
            )
            # [B, L_total, D] -> [B, L_total, H, Dh] (contiguous, no copy)
            grad_attn = torch.ops.aten.reshape.default(
                grad_attn_flat, [B, L_total, H, Dh],
            )

            grad_q_rope, grad_k_rope, grad_v = _sdpa_bwd(grad_attn, s["sdpa_saved"])

            # [B, L_total, H, Dh] -> [B, L_total, D] (view)
            grad_q_rope_flat = torch.ops.aten.reshape.default(
                grad_q_rope, [B, L_total, D],
            )
            grad_k_rope_flat = torch.ops.aten.reshape.default(
                grad_k_rope, [B, L_total, D],
            )
            grad_v_flat = torch.ops.aten.reshape.default(
                grad_v, [B, L_total, D],
            )

            # img: fused RoPE bwd + QK norm bwd + QKV join
            if not _TAIL:
                self._scratch.zero_()
            _scratch_qn = self._scratch_img_qn
            _scratch_kn = self._scratch_img_kn
            gb, fp8_grad_img_qkv, _ = fp8_rope_join_bgrad_quantize(
                grad_q_rope_flat[:, L_txt:, :],
                grad_k_rope_flat[:, L_txt:, :],
                grad_v_flat[:, L_txt:, :],
                s["img_qkv"], H,
                s["img_q_rstd"], s["img_k_rstd"],
                self.img_attn.norm.query_norm.weight,
                self.img_attn.norm.key_norm.weight,
                _scratch_qn, _scratch_kn,
                s["pe"], L_txt,  # img positions start at L_txt
                self.img_attn.qkv.weight,
            )
            _qknorm_fin(
                _scratch_qn, _scratch_kn,
                self.img_attn.norm.query_norm.weight,
                self.img_attn.norm.key_norm.weight,
            )

            if wgrad_mgr is not None:
                grad_img_modulated, wctx = fp8_linear_bwd_dgrad_preq_grad(
                    fp8_grad_img_qkv, gb,
                    self.img_attn.qkv.weight,
                    s["img_mod_shape"],
                    cached_fp8_x=s["fp8_img_mod"],
                )
                wgrad_mgr.defer_wgrad(
                    self.img_attn.qkv.weight,
                    self.img_attn.qkv.bias
                    if hasattr(self.img_attn.qkv, "bias")
                    else None,
                    wctx,
                )
            else:
                grad_img_modulated, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_img_qkv, s["fp8_img_mod"],
                    self.img_attn.qkv.weight,
                    s["img_mod_shape"],
                    wgrad_out=self.img_attn.qkv.weight.grad,
                )
                if hasattr(self.img_attn.qkv, "bias"):
                    _accum_grad(self.img_attn.qkv.bias, gb)

            # img_modulated = (1+scale1) * LN(img_input) + shift1
            grad_img_input, grad_img_scale1 = torch.ops.flux.fused_ln_adaln_bwd(
                grad_img_modulated, s["img_input"], s["img_scale1"],
                s["img_mean1"], s["img_rstd1"], L_img, grad_img_input,
            )
            grad_img_shift1 = grad_img_modulated

            # txt: fused RoPE bwd + QK norm bwd + QKV join
            # Only the txt rows of _scratch need zeroing; the img rows are consumed.
            if not _TAIL:
                self._scratch_txt_qn.zero_()
                self._scratch_txt_kn.zero_()
            _scratch_qn = self._scratch_txt_qn
            _scratch_kn = self._scratch_txt_kn
            gb, fp8_grad_txt_qkv, _ = fp8_rope_join_bgrad_quantize(
                grad_q_rope_flat[:, :L_txt, :],
                grad_k_rope_flat[:, :L_txt, :],
                grad_v_flat[:, :L_txt, :],
                s["txt_qkv"], H,
                s["txt_q_rstd"], s["txt_k_rstd"],
                self.txt_attn.norm.query_norm.weight,
                self.txt_attn.norm.key_norm.weight,
                _scratch_qn, _scratch_kn,
                s["pe"], 0,  # txt positions start at 0
                self.txt_attn.qkv.weight,
            )
            _qknorm_fin(
                _scratch_qn, _scratch_kn,
                self.txt_attn.norm.query_norm.weight,
                self.txt_attn.norm.key_norm.weight,
            )

            if wgrad_mgr is not None:
                grad_txt_modulated, wctx = fp8_linear_bwd_dgrad_preq_grad(
                    fp8_grad_txt_qkv, gb,
                    self.txt_attn.qkv.weight,
                    s["txt_mod_shape"],
                    cached_fp8_x=s["fp8_txt_mod"],
                )
                wgrad_mgr.defer_wgrad(
                    self.txt_attn.qkv.weight,
                    self.txt_attn.qkv.bias
                    if hasattr(self.txt_attn.qkv, "bias")
                    else None,
                    wctx,
                )
            else:
                grad_txt_modulated, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_txt_qkv, s["fp8_txt_mod"],
                    self.txt_attn.qkv.weight,
                    s["txt_mod_shape"],
                    wgrad_out=self.txt_attn.qkv.weight.grad,
                )
                if hasattr(self.txt_attn.qkv, "bias"):
                    _accum_grad(self.txt_attn.qkv.bias, gb)

            grad_txt_input, grad_txt_scale1 = torch.ops.flux.fused_ln_adaln_bwd(
                grad_txt_modulated, s["txt_input"], s["txt_scale1"],
                s["txt_mean1"], s["txt_rstd1"], L_txt, grad_txt_input,
            )
            grad_txt_shift1 = grad_txt_modulated

            # Modulation backward
            img_mod_grad = torch.ops.flux.fused_sum_cat6_ctm(
                grad_img_shift1, grad_img_scale1,
                grad_img_gate1_contrib,
                grad_img_shift2, grad_img_scale2,
                grad_img_gate2_contrib,
            )
            txt_mod_grad = torch.ops.flux.fused_sum_cat6_ctm(
                grad_txt_shift1, grad_txt_scale1,
                grad_txt_gate1_contrib,
                grad_txt_shift2, grad_txt_scale2,
                grad_txt_gate2_contrib,
            )
            if grad_vec_acc is not None:
                # Both branches' dsilu and the accumulate in one kernel.
                gs_img, vec_img = self.img_mod.backward_lin(img_mod_grad)
                gs_txt, vec_txt = self.txt_mod.backward_lin(txt_mod_grad)
                _dsilu_accum_mod(grad_vec_acc, gs_img, vec_img, gs_txt, vec_txt)
                grad_vec = None
            else:
                grad_vec_img = self.img_mod.backward(img_mod_grad)
                grad_vec_txt = self.txt_mod.backward(txt_mod_grad)
                grad_vec = torch.ops.aten.add.Tensor(grad_vec_img, grad_vec_txt)

            del self._saved_fwd
            return grad_img_input, grad_txt_input, grad_vec


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.hidden_size = hidden_size

        self.linear1 = nn.Module()
        self.linear1.weight = nn.Parameter(
            torch.empty(hidden_size * 3 + self.mlp_hidden_dim, hidden_size),
        )
        self.linear1.bias = nn.Parameter(
            torch.empty(hidden_size * 3 + self.mlp_hidden_dim),
        )
        self.linear2 = nn.Module()
        self.linear2.weight = nn.Parameter(
            torch.empty(hidden_size, hidden_size + self.mlp_hidden_dim),
        )
        self.linear2.bias = nn.Parameter(torch.empty(hidden_size))

        self.norm = nn.Module()
        self.norm.query_norm = nn.Module()
        self.norm.query_norm.weight = nn.Parameter(torch.empty(self.head_dim))
        self.norm.key_norm = nn.Module()
        self.norm.key_norm.weight = nn.Parameter(torch.empty(self.head_dim))

        self.modulation = Modulation(dim=hidden_size, double=False)

    def alloc_scratch_buffers(self) -> None:
        """Pre-allocate FP32 scratch buffers for norm weight gradients."""
        dev = self.norm.query_norm.weight.device
        Dh = self.head_dim
        self._scratch = torch.zeros(2, Dh, dtype=torch.float32, device=dev)
        self._scratch_qn = self._scratch[0]
        self._scratch_kn = self._scratch[1]

    def split_linears(self) -> None:
        """Split combined linear1/linear2 into separate QKV+MLP projections.

        Call after load_state_dict.
        """
        D = self.hidden_size
        mlp_dim = self.mlp_hidden_dim

        # linear1 [3D+mlp, D]: row slices are contiguous views
        w1 = self.linear1.weight.data
        b1 = self.linear1.bias.data
        self.linear1_qkv = nn.Module()
        self.linear1_qkv.weight = nn.Parameter(w1[:3 * D].clone())
        self.linear1_qkv.bias = nn.Parameter(b1[:3 * D].clone())
        self.linear1_mlp = nn.Module()
        self.linear1_mlp.weight = nn.Parameter(w1[3 * D:].clone())
        self.linear1_mlp.bias = nn.Parameter(b1[3 * D:].clone())
        del self.linear1

        # linear2 [D, D+mlp]: col slices are non-contiguous
        w2 = self.linear2.weight.data
        self.linear2_attn = nn.Module()
        self.linear2_attn.weight = nn.Parameter(w2[:, :D].contiguous())
        self.linear2_mlp = nn.Module()
        self.linear2_mlp.weight = nn.Parameter(w2[:, D:].contiguous())
        # bias stays on linear2 (shared)
        bias_data = self.linear2.bias.data.clone()
        del self.linear2
        self.linear2 = nn.Module()
        self.linear2.bias = nn.Parameter(bias_data)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        B = x.shape[0]
        L = x.shape[1]
        D = self.hidden_size
        H = self.num_heads
        Dh = self.head_dim

        x_input = x

        # Modulation
        mod_chunks = self.modulation.forward(vec)
        mod_shift = mod_chunks[0]
        mod_scale = mod_chunks[1]
        mod_gate = mod_chunks[2]

        # pre_norm + modulate (fused)
        x_mod, x_mean, x_rstd = torch.ops.flux.fused_ln_adaln(
            x, mod_scale, mod_shift,
        )

        # Separate QKV and MLP projections; x_mod is cast once for both.
        fp8_x_mod, x_mod_shape = fp8_quantize_fwd(
            x_mod, self.linear1_qkv.weight, self.linear1_mlp.weight,
        )
        qkv = fp8_linear_fwd_preq(
            fp8_x_mod, self.linear1_qkv.weight,
            self.linear1_qkv.bias, x_mod_shape,
        )
        mlp = fp8_linear_fwd_preq(
            fp8_x_mod, self.linear1_mlp.weight,
            self.linear1_mlp.bias, x_mod_shape,
        )
        # AFTER the second reader: x_mod feeds qkv AND fc1 in the forward.
        release_fwd(fp8_x_mod)

        # QKV split + QK norm + RoPE (fused)
        q_rope, k_rope, v, q_rstd, k_rstd = (
            torch.ops.flux.qkv_split_norm_rope_ctm(
                qkv, H,
                self.norm.query_norm.weight,
                self.norm.key_norm.weight,
                pe, 0,  # positions start at 0
            )
        )
        attn_out, sdpa_saved = _sdpa_fwd(q_rope, k_rope, v)

        # BSHD [B, L, H, Dh] -> [B, L, D] (contiguous, no copy)
        attn_flat = torch.ops.aten.reshape.default(attn_out, [B, L, D])

        # GELU + quantize
        fp8_mlp_act, mlp_act_shape = fp8_gelu_fwd(
            mlp, self.linear2_mlp.weight,
        )

        # Separate attn + MLP projections
        out_attn, fp8_attn_flat = fp8_linear_fwd_save(
            attn_flat, self.linear2_attn.weight, self.linear2.bias,
        )
        out_mlp = fp8_linear_fwd_preq(
            fp8_mlp_act, self.linear2_mlp.weight, None, mlp_act_shape,
        )
        release_fwd(fp8_mlp_act)  # fc2 was its only forward reader
        result, output = torch.ops.flux.fused_gate_residual_add_a2(
            x_input, mod_gate, out_attn, out_mlp,
        )

        self._saved_fwd = {
            "x_input": x_input, "x_mean": x_mean, "x_rstd": x_rstd,
            "vec": vec, "pe": pe,
            "mod_shift": mod_shift, "mod_scale": mod_scale, "mod_gate": mod_gate,
            "mlp": mlp,
            # pre-norm Q/K for backward are qkv[..., :D] / qkv[..., D:2D]
            "qkv": qkv,
            "q_rstd": q_rstd, "k_rstd": k_rstd,
            "sdpa_saved": sdpa_saved,
            "output": output,
            "B": B, "L": L,
            "fp8_x_mod": fp8_x_mod,
            "x_mod_shape": list(x_mod.shape),
            "fp8_attn_flat": fp8_attn_flat,
            "attn_flat_shape": list(attn_flat.shape),
            "fp8_mlp_act": fp8_mlp_act,
            "mlp_act_shape": mlp_act_shape,
        }
        return result

    def backward(
        self,
        grad_output: Tensor,
        wgrad_mgr: "BucketManager | None" = None,
        grad_vec_acc: Tensor | None = None,
    ) -> tuple:
        with torch.no_grad():
            s = self._saved_fwd
            B = s["B"]
            L = s["L"]
            D = self.hidden_size
            H = self.num_heads
            Dh = self.head_dim

            # result = x_input + mod_gate * output
            # grad_gated is quantized once for both linear2 backward GEMMs.
            gb, fp8_grad_gated, _, grad_gate_contrib = fp8_gate_bwd_quantize(
                grad_output, s["mod_gate"], s["output"],
                self.linear2_attn.weight, self.linear2_mlp.weight,
                bias_grad=_bias_grad_view(self.linear2.bias),
            )
            grad_x = grad_output  # residual

            # linear2 backward
            if wgrad_mgr is not None:
                grad_attn_flat, wctx = fp8_linear_bwd_dgrad_preq_grad(
                    fp8_grad_gated, gb,
                    self.linear2_attn.weight,
                    s["attn_flat_shape"],
                    cached_fp8_x=s["fp8_attn_flat"],
                )
                wgrad_mgr.defer_wgrad(
                    self.linear2_attn.weight,
                    _defer_bias(self.linear2.bias), wctx,
                )
                grad_mlp_act, wctx2 = fp8_linear_bwd_dgrad_preq_grad(
                    fp8_grad_gated, None,
                    self.linear2_mlp.weight,
                    s["mlp_act_shape"],
                    cached_fp8_x=s["fp8_mlp_act"],
                )
                wgrad_mgr.defer_wgrad(
                    self.linear2_mlp.weight, None, wctx2,
                )
            else:
                _accum_bias_grad(self.linear2.bias, gb)
                grad_attn_flat, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_gated, s["fp8_attn_flat"],
                    self.linear2_attn.weight,
                    s["attn_flat_shape"],
                    wgrad_out=self.linear2_attn.weight.grad,
                )
                grad_mlp_act, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_gated, s["fp8_mlp_act"],
                    self.linear2_mlp.weight,
                    s["mlp_act_shape"],
                    wgrad_out=self.linear2_mlp.weight.grad,
                )

            # Attn reshape bwd (BSHD: [B, L, D] -> [B, L, H, Dh], no copy)
            grad_attn = torch.ops.aten.reshape.default(
                grad_attn_flat, [B, L, H, Dh],
            )

            grad_q_rope, grad_k_rope, grad_v = _sdpa_bwd(grad_attn, s["sdpa_saved"])

            # Fused RoPE bwd + QK norm bwd + QKV join
            if not _TAIL:
                self._scratch.zero_()
            _scratch_qn = self._scratch_qn
            _scratch_kn = self._scratch_kn
            gb_qkv, fp8_grad_qkv, _ = fp8_rope_join_bgrad_quantize(
                torch.ops.aten.reshape.default(grad_q_rope, [B, L, D]),
                torch.ops.aten.reshape.default(grad_k_rope, [B, L, D]),
                torch.ops.aten.reshape.default(grad_v, [B, L, D]),
                s["qkv"], H,
                s["q_rstd"], s["k_rstd"],
                self.norm.query_norm.weight,
                self.norm.key_norm.weight,
                _scratch_qn, _scratch_kn,
                s["pe"], 0,  # SSB positions start at 0
                self.linear1_qkv.weight,
            )
            _qknorm_fin(
                _scratch_qn, _scratch_kn,
                self.norm.query_norm.weight,
                self.norm.key_norm.weight,
            )

            if wgrad_mgr is not None:
                grad_x_mod, wctx = fp8_linear_bwd_dgrad_preq_grad(
                    fp8_grad_qkv, gb_qkv,
                    self.linear1_qkv.weight,
                    s["x_mod_shape"],
                    cached_fp8_x=s["fp8_x_mod"],
                )
                wgrad_mgr.defer_wgrad(
                    self.linear1_qkv.weight,
                    self.linear1_qkv.bias, wctx,
                )
                grad_x_mlp, wctx2 = fp8_linear_bwd_dgrad_dgelu(
                    grad_mlp_act, s["mlp"],
                    None, self.linear1_mlp.weight,
                    cached_fp8_x=s["fp8_x_mod"],
                    x_shape=s["x_mod_shape"],
                )
                wgrad_mgr.defer_wgrad(
                    self.linear1_mlp.weight,
                    self.linear1_mlp.bias, wctx2,
                )
                grad_x, grad_mod_scale, grad_mod_shift = (
                    torch.ops.flux.fused_ln_adaln_bwd_add(
                        grad_x_mod, grad_x_mlp,
                        s["x_input"], s["mod_scale"],
                        s["x_mean"], s["x_rstd"], L, grad_x,
                    )
                )
            else:
                fp8_x_mod_bwd = s["fp8_x_mod"]
                x_mod_shape = s["x_mod_shape"]
                grad_x_mod, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_qkv, fp8_x_mod_bwd,
                    self.linear1_qkv.weight,
                    x_mod_shape,
                    wgrad_out=self.linear1_qkv.weight.grad,
                )
                _accum_grad(self.linear1_qkv.bias, gb_qkv)
                gb_mlp, fp8_grad_mlp, _ = (
                    fp8_dbias_dqgelu_bwd(grad_mlp_act, s["mlp"])
                )
                grad_x_mlp, _ = fp8_linear_bwd_preq_both(
                    fp8_grad_mlp, fp8_x_mod_bwd,
                    self.linear1_mlp.weight,
                    x_mod_shape,
                    wgrad_out=self.linear1_mlp.weight.grad,
                )
                _accum_grad(self.linear1_mlp.bias, gb_mlp)
                grad_x, grad_mod_scale, grad_mod_shift = (
                    torch.ops.flux.fused_ln_adaln_bwd_add(
                        grad_x_mod, grad_x_mlp,
                        s["x_input"], s["mod_scale"],
                        s["x_mean"], s["x_rstd"], L, grad_x,
                    )
                )

            # Modulation backward
            mod_grad = torch.ops.flux.fused_sum_cat3_ctm(
                grad_mod_shift, grad_mod_scale, grad_gate_contrib,
            )
            if grad_vec_acc is not None:
                gs, vec = self.modulation.backward_lin(mod_grad)
                _dsilu_accum_mod(grad_vec_acc, gs, vec, None, None)
                grad_vec = None
            else:
                grad_vec = self.modulation.backward(mod_grad)

            del self._saved_fwd
            return grad_x, grad_vec


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.out_dim = patch_size * patch_size * out_channels

        self.adaLN_modulation = nn.Module()
        lin = nn.Module()
        lin.weight = nn.Parameter(
            torch.empty(2 * hidden_size, hidden_size),
        )
        lin.bias = nn.Parameter(torch.empty(2 * hidden_size))
        self.adaLN_modulation.register_module("1", lin)
        object.__setattr__(self, "_adaLN_mod_1", self.adaLN_modulation._modules["1"])

        self.linear = nn.Module()
        self.linear.weight = nn.Parameter(torch.empty(self.out_dim, hidden_size))
        self.linear.bias = nn.Parameter(torch.empty(self.out_dim))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        D = self.hidden_size

        silu_out = _silu(vec)
        wT = torch.ops.aten.t.default(self._adaLN_mod_1.weight)
        adaln_out = torch.ops.aten.addmm.default(self._adaLN_mod_1.bias, silu_out, wT)
        parts = torch.ops.aten.split.Tensor(adaln_out, D, 1)
        shift = parts[0]
        scale = parts[1]

        scale_unsq = torch.ops.aten.unsqueeze.default(scale, 1)
        shift_unsq = torch.ops.aten.unsqueeze.default(shift, 1)
        modulated, mean, rstd = torch.ops.flux.fused_ln_adaln(
            x, scale_unsq, shift_unsq,
        )

        out = _linear_fwd(modulated, self.linear.weight, self.linear.bias)

        self._saved_x = x
        self._saved_vec = vec
        self._saved_silu_out = silu_out
        self._saved_mean = mean
        self._saved_rstd = rstd
        self._saved_scale = scale
        self._saved_shift = shift
        self._saved_modulated = modulated

        return out

    def backward(self, grad_output: Tensor) -> tuple:
        with torch.no_grad():
            D = self.hidden_size
            x = self._saved_x
            vec = self._saved_vec
            silu_out = self._saved_silu_out
            mean = self._saved_mean
            rstd = self._saved_rstd
            scale = self._saved_scale
            modulated = self._saved_modulated

            # Linear backward
            grad_modulated, _, _ = _linear_bwd(
                grad_output, modulated, self.linear.weight,
                wgrad_out=self.linear.weight.grad,
                bias=self.linear.bias,
            )

            # Fused LN + AdaLN modulation backward
            scale_unsq = torch.ops.aten.unsqueeze.default(scale, 1)
            L = grad_modulated.shape[1]
            grad_x, grad_scale_unsq = torch.ops.flux.fused_ln_adaln_bwd(
                grad_modulated, x, scale_unsq, mean, rstd, L,
            )
            grad_shift_unsq = grad_modulated

            # adaLN linear backward
            if _TAIL:
                grad_adaln = torch.ops.aten.reshape.default(
                    torch.ops.flux.fused_sum_cat2_ctm(
                        grad_shift_unsq, grad_scale_unsq,
                    ),
                    [-1, 2 * D],
                )
            else:
                grad_scale = torch.ops.aten.sum.dim_IntList(
                    grad_scale_unsq, [1], False,
                )
                grad_shift = torch.ops.aten.sum.dim_IntList(
                    grad_shift_unsq, [1], False,
                )
                grad_adaln = torch.ops.aten.cat.default(
                    [grad_shift, grad_scale], 1,
                )
            grad_silu = torch.ops.aten.mm.default(grad_adaln, self._adaLN_mod_1.weight)
            grad_adaln_T = torch.ops.aten.t.default(grad_adaln)
            torch.ops.aten.mm.out(
                grad_adaln_T, silu_out,
                out=self._adaLN_mod_1.weight.grad,
            )
            _bias_grad(self._adaLN_mod_1.bias, grad_adaln)

            # SiLU backward
            grad_vec = _dsilu(grad_silu, vec)

            del self._saved_x, self._saved_vec, self._saved_silu_out
            del self._saved_mean, self._saved_rstd
            del self._saved_scale, self._saved_shift, self._saved_modulated

            return grad_x, grad_vec


class FluxModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        hidden_size: int = 3072,
        mlp_ratio: float = 4.0,
        num_heads: int = 24,
        depth: int = 19,
        depth_single_blocks: int = 38,
        axes_dim: tuple = (16, 56, 56),
        theta: int = 10_000,
        qkv_bias: bool = True,
        vec_in_dim: int = 768,
        context_in_dim: int = 4096,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.axes_dim = axes_dim
        self.theta = theta

        self.img_in = nn.Module()
        self.img_in.weight = nn.Parameter(torch.empty(hidden_size, in_channels))
        self.img_in.bias = nn.Parameter(torch.empty(hidden_size))
        self.txt_in = nn.Module()
        self.txt_in.weight = nn.Parameter(torch.empty(hidden_size, context_in_dim))
        self.txt_in.bias = nn.Parameter(torch.empty(hidden_size))

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        self.vector_in = MLPEmbedder(in_dim=vec_in_dim, hidden_dim=hidden_size)

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias)
             for _ in range(depth)],
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio)
             for _ in range(depth_single_blocks)],
        )
        self.final_layer = LastLayer(hidden_size, patch_size=1, out_channels=out_channels)

        # Cached RoPE frequencies, constant across iterations.
        self._cached_pe: Tensor | None = None

    def _timestep_embedding(self, t: Tensor, dim: int) -> Tensor:
        """Inline timestep_embedding using ATen ops."""
        max_period = 10000
        time_factor = 1000.0
        t_scaled = torch.ops.aten.mul.Scalar(t, time_factor)
        half = dim // 2
        arange_vals = torch.ops.aten.arange.default(
            half, dtype=torch.float32, device=t.device,
        )
        log_period = torch.ops.aten.mul.Scalar(
            arange_vals, -math.log(max_period) / half,
        )
        freqs = torch.ops.aten.exp.default(log_period)
        t_unsq = torch.ops.aten.unsqueeze.default(
            torch.ops.aten._to_copy.default(t_scaled, dtype=torch.float32), 1,
        )
        freqs_unsq = torch.ops.aten.unsqueeze.default(freqs, 0)
        args = torch.ops.aten.mul.Tensor(t_unsq, freqs_unsq)
        cos_args = torch.ops.aten.cos.default(args)
        sin_args = torch.ops.aten.sin.default(args)
        embedding = torch.ops.aten.cat.default([cos_args, sin_args], 1)
        if t.dtype != torch.float32 and t.is_floating_point():
            embedding = torch.ops.aten._to_copy.default(embedding, dtype=t.dtype)
        return embedding

    def _embed_nd(self, ids: Tensor) -> Tensor:
        """Compute raw RoPE frequencies for the TE fused kernel.

        ids: [B, S, n_axes] position IDs. Returns [S, 1, 1, D] float32 angles
        with interleaved duplicate pairs (tex.fused_rope_forward interleaved=True).
        """
        n_axes = ids.shape[-1]
        embs = []
        for i in range(n_axes):
            ax_dim = self.axes_dim[i]
            pos = torch.ops.aten.select.int(ids, -1, i)
            assert ax_dim % 2 == 0
            half_dim = ax_dim // 2
            scale = torch.ops.aten.arange.start_step(
                0, ax_dim, 2, dtype=torch.float64, device=pos.device,
            )
            scale = torch.ops.aten.div.Scalar(scale, ax_dim)
            theta_tensor = torch.ops.aten.scalar_tensor.default(
                float(self.theta), dtype=torch.float64, device=pos.device,
            )
            theta_pow = torch.ops.aten.pow.Tensor_Tensor(
                torch.ops.aten.expand.default(
                    theta_tensor, list(scale.shape),
                ),
                scale,
            )
            omega = torch.ops.aten.reciprocal.default(theta_pow)
            # out = pos * omega -> [B, S, half_dim]
            pos_f = torch.ops.aten._to_copy.default(
                pos, dtype=torch.float32,
            )
            pos_unsq = torch.ops.aten.unsqueeze.default(pos_f, -1)
            omega_f32 = torch.ops.aten._to_copy.default(
                omega, dtype=torch.float32,
            )
            omega_r = torch.ops.aten.reshape.default(
                omega_f32, [1, 1, half_dim],
            )
            out = torch.ops.aten.mul.Tensor(pos_unsq, omega_r)
            embs.append(out)
        # emb: [B, S, total_half_dim]
        emb = torch.ops.aten.cat.default(embs, -1)
        # Permute to [S, B, 1, total_half_dim]
        emb = torch.ops.aten.unsqueeze.default(emb, 2)
        emb = torch.ops.aten.permute.default(emb, [1, 0, 2, 3])
        # Interleave: duplicate each freq for paired rotation
        # [S, B, 1, half] -> stack [S, B, 1, half, 2] -> [S, B, 1, dim]
        emb = torch.ops.aten.stack.default([emb, emb], -1)
        S = emb.shape[0]
        dim = emb.shape[3] * 2
        emb = torch.ops.aten.reshape.default(emb, [S, -1, 1, dim])
        # Take first batch element (freqs are batch-invariant).
        # Must be contiguous: TE fused RoPE kernel reads raw memory.
        emb = torch.ops.aten.slice.Tensor(emb, 1, 0, 1)
        emb = torch.ops.aten.contiguous.default(emb)
        return emb

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        on_block_start: "Callable[[int], None] | None" = None,
    ) -> Tensor:
        amax_arena.reset()
        # Wait for stem params (bucket 0) before stem ops
        if on_block_start:
            on_block_start(0)

        img_input = img
        img = _linear_fwd(img, self.img_in.weight, self.img_in.bias)

        # timestep + vec embeddings
        if _TAIL and timesteps.dtype == torch.bfloat16:
            t_emb = torch.ops.flux.timestep_embed_ctm(timesteps, 256)
        else:
            t_emb = self._timestep_embedding(timesteps, 256)
        vec = self.time_in.forward(t_emb)
        vec_y = self.vector_in.forward(y)
        if _TAIL:
            vec = torch.ops.flux.add_ctm(vec, vec_y)
        else:
            vec = torch.ops.aten.add.Tensor(vec, vec_y)

        txt_input = txt
        txt = _linear_fwd(txt, self.txt_in.weight, self.txt_in.bias)

        # Positional embedding (cached; ids are constant across iterations)
        if self._cached_pe is None:
            ids = torch.ops.aten.cat.default([txt_ids, img_ids], 1)
            self._cached_pe = self._embed_nd(ids)
        pe = self._cached_pe

        for i, block in enumerate(self.double_blocks):
            if on_block_start:
                on_block_start(i + 1)
            img, txt = block.forward(img=img, txt=txt, vec=vec, pe=pe)

        # Concat for single stream
        txt_for_single = txt
        if _TAIL:
            img = torch.ops.flux.seq_join_ctm3(txt, img)
        else:
            img = torch.ops.aten.cat.default([txt, img], 1)

        for i, block in enumerate(self.single_blocks):
            if on_block_start:
                on_block_start(20 + i)
            img = block.forward(img, vec=vec, pe=pe)

        # Slice off txt portion
        txt_len = txt_for_single.shape[1]
        img = torch.ops.aten.slice.Tensor(img, 1, txt_len, img.shape[1])

        if on_block_start:
            on_block_start(58)
        img = self.final_layer.forward(img, vec)

        self._saved_fwd = {
            "img_input": img_input, "txt_input": txt_input,
            "t_emb": t_emb, "y": y, "vec": vec,
            "txt_len": txt_len, "pe": pe,
        }
        return img

    def backward(
        self,
        grad_output: Tensor,
        on_block_done: "Callable[[int], None] | None" = None,
        wgrad_mgr: "object | None" = None,
    ) -> None:
        """Backward pass through all blocks.

        on_block_done(bucket_id) fires after each block's backward; bucket ids:
        0 = stem, 1-19 = double_blocks, 20-57 = single_blocks, 58 = final_layer.
        """
        with torch.no_grad():
            s = self._saved_fwd
            vec = s["vec"]
            pe = s["pe"]
            txt_len = s["txt_len"]
            n_single = len(self.single_blocks)

            grad_img, grad_vec = self.final_layer.backward(grad_output)
            if on_block_done:
                on_block_done(58)  # final_layer bucket

            # Undo slice: pad zeros for txt portion
            B = grad_img.shape[0]
            D = self.hidden_size
            if _TAIL:
                grad_combined = torch.ops.flux.pad_join_ctm(grad_img, txt_len)
            else:
                grad_txt_pad = torch.ops.aten.zeros.default(
                    [B, txt_len, D], dtype=grad_img.dtype, device=grad_img.device,
                )
                grad_combined = torch.ops.aten.cat.default(
                    [grad_txt_pad, grad_img], 1,
                )
            # When the DSL tail is live every block accumulates its vec gradient
            # straight into grad_vec, so there is nothing to add per block.
            grad_vec_acc = grad_vec if _TAIL else None

            for i, block in enumerate(
                reversed(list(self.single_blocks))
            ):
                grad_combined, grad_vec_block = block.backward(
                    grad_combined, wgrad_mgr=wgrad_mgr,
                    grad_vec_acc=grad_vec_acc,
                )
                if grad_vec_acc is None:
                    grad_vec = torch.ops.aten.add.Tensor(grad_vec, grad_vec_block)
                if on_block_done:
                    # single_blocks in reverse: i=0 is last block
                    bucket_id = 20 + (n_single - 1 - i)
                    on_block_done(bucket_id)

            # Split combined -> txt, img; dim-1 slices are non-contiguous
            if _TAIL:
                grad_txt, grad_img = torch.ops.flux.seq_split_ctm3(
                    grad_combined, txt_len,
                )
            else:
                grad_txt = torch.ops.aten.slice.Tensor(
                    grad_combined, 1, 0, txt_len,
                ).contiguous()
                grad_img = torch.ops.aten.slice.Tensor(
                    grad_combined, 1, txt_len, grad_combined.shape[1],
                ).contiguous()

            n_double = len(self.double_blocks)
            for i, block in enumerate(
                reversed(list(self.double_blocks))
            ):
                grad_img, grad_txt, grad_vec_block = block.backward(
                    grad_img, grad_txt, wgrad_mgr=wgrad_mgr,
                    grad_vec_acc=grad_vec_acc,
                )
                if grad_vec_acc is None:
                    grad_vec = torch.ops.aten.add.Tensor(grad_vec, grad_vec_block)
                if on_block_done:
                    bucket_id = 1 + (n_double - 1 - i)
                    on_block_done(bucket_id)

            grad_txt_input, _, _ = _linear_bwd(
                grad_txt, s["txt_input"], self.txt_in.weight,
                wgrad_out=self.txt_in.weight.grad,
                bias=self.txt_in.bias,
            )

            grad_img_input, _, _ = _linear_bwd(
                grad_img, s["img_input"], self.img_in.weight,
                wgrad_out=self.img_in.weight.grad,
                bias=self.img_in.bias,
            )

            # vec backward: vec = time_in(t_emb) + vector_in(y)
            grad_t_emb = self.time_in.backward(grad_vec)
            grad_y = self.vector_in.backward(grad_vec)

            if on_block_done:
                on_block_done(0)  # stem bucket

            del self._saved_fwd


def build_model() -> FluxModel:
    """Build full-scale FluxModel (19 double + 38 single blocks)."""
    model = FluxModel(
        in_channels=64,
        out_channels=64,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        vec_in_dim=768,
        context_in_dim=4096,
    )
    return model


_BATCH = 2
_IMG_SEQ = 256
_TXT_SEQ = 256

example_inputs = (
    torch.randn(_BATCH, _IMG_SEQ, 64),
    torch.zeros(_BATCH, _IMG_SEQ, 3),
    torch.randn(_BATCH, _TXT_SEQ, 4096),
    torch.zeros(_BATCH, _TXT_SEQ, 3),
    torch.rand(_BATCH),
    torch.randn(_BATCH, 768),
)
