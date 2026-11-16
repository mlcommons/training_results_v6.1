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

"""Routes TransformerEngine's FP8 fused attention backward to `torch.ops.gpt_oss.fused_attn_bwd_fp8` for the gpt-oss-20b
Float8CurrentScaling SDPA (head_dim 64, GQA 64q/8kv, sbhd, causal / sliding-window-127, learnable softmax sink); every
other call stays on cuDNN (`transformer_engine.pytorch.cpp_extensions.fused_attn_bwd`).

Hook point: `transformer_engine.pytorch.attention.dot_product_attention.backends.fused_attn_bwd` is a module-global name
resolved at call time inside `_fused_attn_backward_impl`, so rebinding it on the `backends` module is enough.

Contract of the replaced call (TE commit 9c33764; global recipe MXFP8, DPA recipe Float8CurrentScaling, NVTE_DPA_FP8CS_O_in_F16=1):
q/k/v Float8Tensor e4m3 sbhd with per-tensor scale_inv; o bf16 sbhd; d_o Float8Tensor e4m3 (e5m2 under a HYBRID recipe);
aux_ctx = [softmax_stats fp32 [b,h,s,1], rng_state, softmax_offset fp32 [1,h,1,1]]; s_quantizer / dp_quantizer are delayed-scaling
Float8Quantizers (scale for P -> e4m3, scale for dS*attn_scale -> fp8; the kernel writes amax_dP into dp_quantizer);
dqkv_quantizer is a Float8CurrentScalingQuantizer, so dq/dk/dv are plain bf16. Returns (dq, dk, dv, dbias, d_softmax_offset fp32 [1,h,1,1]).

Gate: FUSED_ATTN_BWD=0 leaves the reference path (cuDNN); default on.
"""
import os
import sys

import torch

_INSTALLED = False
_COUNTS = {"routed": 0, "fallback": 0}


def _say(msg: str) -> None:
    # utils.py replaces builtins.print with a rank-0-only print; write through sys.__stdout__ instead
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"FUSED_ATTN_BWD rank={rank} {msg}\n")
    sys.__stdout__.flush()


def _is_float8(t) -> bool:
    return hasattr(t, "_fp8_dtype") and hasattr(t, "_scale_inv") and hasattr(t, "_data")


def _quantizer_dtype(quantizer, default=8) -> int:
    """TE DType code of a Float8Quantizer (its .dtype is a tex.DType enum; fall back to e5m2)."""
    try:
        return int(quantizer.dtype)
    except Exception:  # noqa
        try:
            return int(quantizer.dtype.value)
        except Exception:  # noqa
            return default


def _eligible(q, k, v, o, d_o, backend, qkv_layout, o_format, do_format, dqkv_layout, attn_bias_type,
              attn_mask_type, softmax_type, window_size, dropout, cu_seqlens_q_padded,
              cu_seqlens_kv_padded) -> bool:
    try:
        from transformer_engine.pytorch.attention.dot_product_attention.utils import FusedAttnBackend
        if FusedAttnBackend.cast(backend) != FusedAttnBackend["FP8"]:
            return False
    except Exception:  # noqa
        return False
    if not (_is_float8(q) and _is_float8(k) and _is_float8(v) and _is_float8(d_o)):
        return False
    # TE DType codes: 7 = kFloat8E4M3, 8 = kFloat8E5M2. q/k/v must be e4m3; dO is e4m3 or e5m2
    if any(int(t._fp8_dtype) != 7 for t in (q, k, v)) or int(d_o._fp8_dtype) not in (7, 8):
        return False
    if _is_float8(o) or o.dtype != torch.bfloat16:
        return False  # only the O-in-F16 variant is implemented
    if qkv_layout != "sbhd_sbhd_sbhd" or dqkv_layout != "sbhd_sbhd_sbhd":
        return False
    if o_format != "sbhd" or do_format != "sbhd":
        return False
    if attn_bias_type != "no_bias" or dropout != 0.0:
        return False
    if attn_mask_type != "causal" or tuple(window_size) not in ((-1, 0), (-1, -1), (127, 0)):
        return False
    if softmax_type not in ("vanilla", "learnable"):
        return False
    if cu_seqlens_q_padded is not None or cu_seqlens_kv_padded is not None:
        return False
    s, b, h, d = q.shape
    sk, bk, hg, dk = k.shape
    if d != 64 or dk != 64 or s != sk or b != bk or h % hg != 0:
        return False
    return True


def _import_kernel() -> None:
    """Register torch.ops.gpt_oss.fused_attn_bwd_fp8 (no-op when already registered)."""
    if hasattr(torch.ops, "gpt_oss") and hasattr(torch.ops.gpt_oss, "fused_attn_bwd_fp8"):
        return
    try:
        from . import fused_attn_bwd_fp8  # noqa: F401
    except ImportError:  # loaded by file path, no parent package
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fused_attn_bwd_fp8.py")
        spec = importlib.util.spec_from_file_location("fused_attn_bwd_fp8", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)


def install() -> bool:
    """Rebind backends.fused_attn_bwd. Returns True when the hook is active; FUSED_ATTN_BWD=0 leaves TE untouched."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.getenv("FUSED_ATTN_BWD", "1") != "1":
        _say("disabled by FUSED_ATTN_BWD=0 (control path: cuDNN fused_attn_bwd)")
        return False
    try:
        _import_kernel()
    except Exception as e:  # noqa
        _say(f"kernel import failed, hook NOT installed: {type(e).__name__}: {e}")
        return False
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    orig = backends.fused_attn_bwd

    def fused_attn_bwd(max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv, q, k, v, o, d_o,
                       fake_dtype, aux_ctx_tensors, fused_attention_backend,
                       cu_seqlens_q_padded=None, cu_seqlens_kv_padded=None, s_quantizer=None,
                       dp_quantizer=None, dqkv_quantizer=None, attn_scale=None, dropout=0.0,
                       fast_zero_fill=True, qkv_layout="sbh3d", o_format="sbhd", do_format="sbhd",
                       dqkv_layout="sbh3d", qkv_scale_inv_format=None, do_scale_inv_format=None,
                       attn_bias_type="no_bias", attn_mask_type="padding", softmax_type="vanilla",
                       window_size=(-1, -1), bottom_right_diagonal=None, deterministic=False,
                       cuda_graph=False):
        if not _eligible(q, k, v, o, d_o, fused_attention_backend, qkv_layout, o_format, do_format,
                         dqkv_layout, attn_bias_type, attn_mask_type, softmax_type, window_size, dropout,
                         cu_seqlens_q_padded, cu_seqlens_kv_padded):
            _COUNTS["fallback"] += 1
            if _COUNTS["fallback"] <= 2:
                _say(f"fallback to cuDNN: layout={qkv_layout} mask={attn_mask_type} window={tuple(window_size)} "
                     f"softmax={softmax_type} q={type(q).__name__} o={o.dtype} d_o={type(d_o).__name__}")
            return orig(max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv, q, k, v, o, d_o,
                        fake_dtype, aux_ctx_tensors, fused_attention_backend, cu_seqlens_q_padded,
                        cu_seqlens_kv_padded, s_quantizer, dp_quantizer, dqkv_quantizer, attn_scale,
                        dropout, fast_zero_fill, qkv_layout, o_format, do_format, dqkv_layout,
                        qkv_scale_inv_format, do_scale_inv_format, attn_bias_type, attn_mask_type,
                        softmax_type, window_size, bottom_right_diagonal, deterministic, cuda_graph)
        s, b, h, d = q.shape
        if attn_scale is None:
            attn_scale = 1.0 / (d ** 0.5)
        _COUNTS["routed"] += 1
        if _COUNTS["routed"] <= 2:
            _say(f"routed to gpt_oss::fused_attn_bwd_fp8: shape={tuple(q.shape)} kv={tuple(k.shape)} "
                 f"window={tuple(window_size)} softmax={softmax_type} dO_dtype={int(d_o._fp8_dtype)} "
                 f"dS_dtype={_quantizer_dtype(dp_quantizer)}")
        stats = aux_ctx_tensors[0]                    # fp32 [b, h, s, 1] (LSE, natural log)
        softmax_offset = aux_ctx_tensors[-1] if softmax_type != "vanilla" else None
        dq, dk, dv, d_softmax_offset, amax_dp = torch.ops.gpt_oss.fused_attn_bwd_fp8(
            q._data, k._data, v._data, o, d_o._data,
            q._scale_inv.reshape(1), k._scale_inv.reshape(1), v._scale_inv.reshape(1),
            d_o._scale_inv.reshape(1), s_quantizer.scale.reshape(1), dp_quantizer.scale.reshape(1),
            stats, softmax_offset, float(attn_scale), int(window_size[0]),
            int(d_o._fp8_dtype), _quantizer_dtype(dp_quantizer),
        )
        # the cuDNN path writes amax_dP into the dP quantizer's amax buffer (delayed-scaling update); match it
        if getattr(dp_quantizer, "amax", None) is not None:
            dp_quantizer.amax.view(-1)[0:1].copy_(amax_dp.view(-1)[0:1])
        return dq, dk, dv, torch.empty(0, device=q.device), d_softmax_offset

    backends.fused_attn_bwd = fused_attn_bwd
    _INSTALLED = True
    _say("installed: backends.fused_attn_bwd -> gpt_oss::fused_attn_bwd_fp8 for eligible FP8 current-scaling calls")
    return True
