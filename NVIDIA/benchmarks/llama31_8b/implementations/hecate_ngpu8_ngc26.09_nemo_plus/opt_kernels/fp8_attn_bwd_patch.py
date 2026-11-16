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

import os

import torch

_ENABLED = os.environ.get("OPT_FP8_ATTN_BWD", "1") == "1"
_announced = False
_notes = set()


def _note(key, msg):
    if key not in _notes:
        _notes.add(key)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[opt fp8_attn_bwd] {msg}", flush=True)


def _install():
    import transformer_engine.pytorch.attention.dot_product_attention.backends as be
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    E4M3, E5M2 = int(tex.DType.kFloat8E4M3), int(tex.DType.kFloat8E5M2)

    from . import fp8_attn_bwd_sm100 as k  # noqa: F401  (registers torch.ops.optimized.*)

    orig = be.fused_attn_bwd
    if getattr(orig, "_opt_wrapped", False):
        return

    def _fp8_view(t):
        dt = torch.float8_e4m3fn if int(t._fp8_dtype) == E4M3 else torch.float8_e5m2
        return t._data.view(dt)

    def fused_attn_bwd(*args, **kwargs):
        global _announced
        names = (
            "max_seqlen_q", "max_seqlen_kv", "cu_seqlens_q", "cu_seqlens_kv", "q", "k", "v",
            "o", "d_o", "fake_dtype", "aux_ctx_tensors", "fused_attention_backend",
            "cu_seqlens_q_padded", "cu_seqlens_kv_padded", "s_quantizer", "dp_quantizer",
            "dqkv_quantizer", "attn_scale", "dropout", "fast_zero_fill", "qkv_layout",
            "o_format", "do_format", "dqkv_layout", "qkv_scale_inv_format",
            "do_scale_inv_format", "attn_bias_type", "attn_mask_type", "softmax_type",
            "window_size", "bottom_right_diagonal", "deterministic", "cuda_graph",
        )
        a = dict(zip(names, args))
        a.update(kwargs)
        q, k_, v, o, do = a.get("q"), a.get("k"), a.get("v"), a.get("o"), a.get("d_o")
        sq, dpq = a.get("s_quantizer"), a.get("dp_quantizer")
        layout = a.get("qkv_layout", "sbh3d")
        fmt = layout.split("_")[0] if isinstance(layout, str) else ""
        mask = a.get("attn_mask_type", "padding")
        if a.get("deterministic", False):
            _note("deterministic", "deterministic=True requested: TE / cuDNN FP8 attention backward runs "
                  "instead (the CuTe kernel's dQ reduction is unordered)")
            return orig(*args, **kwargs)
        ok = (
            all(isinstance(t, Float8Tensor) for t in (q, k_, v, do))
            and isinstance(o, torch.Tensor) and not isinstance(o, Float8Tensor)
            and o.dtype == torch.bfloat16 and a.get("fake_dtype") == torch.bfloat16
            and layout in ("sbhd_sbhd_sbhd", "bshd_bshd_bshd")
            and a.get("o_format", fmt) == fmt and a.get("do_format", fmt) == fmt
            and a.get("dqkv_layout", layout) == layout
            and a.get("qkv_scale_inv_format") is None and a.get("do_scale_inv_format") is None
            and mask in ("causal", "no_mask")
            and not a.get("bottom_right_diagonal", False)
            and tuple(a.get("window_size", (-1, -1))) in ((-1, 0), (-1, -1))
            and a.get("dropout", 0.0) == 0.0
            and a.get("attn_bias_type", "no_bias") == "no_bias"
            and a.get("softmax_type", "vanilla") == "vanilla"
            and a.get("max_seqlen_q") == a.get("max_seqlen_kv")
            and sq is not None and dpq is not None
            and hasattr(sq, "scale") and hasattr(dpq, "scale") and hasattr(dpq, "amax")
            and all(int(t._fp8_dtype) == E4M3 for t in (q, k_, v))
            and int(do._fp8_dtype) == E5M2
            and all(t._data is not None for t in (q, k_, v, do))
        )
        if ok:
            perm = (1, 0, 2, 3) if fmt == "sbhd" else (0, 1, 2, 3)
            qv, kv, vv, dov = [_fp8_view(t).permute(*perm) for t in (q, k_, v, do)]
            ov = o.permute(*perm)
            stats = a["aux_ctx_tensors"][0]
            ok = (
                k.supported(qv, kv, vv)
                and qv.shape[1] == a.get("max_seqlen_q")
                and stats.dtype == torch.float32 and stats.dim() == 4
                and tuple(stats.shape) == (qv.shape[0], qv.shape[2], qv.shape[1], 1)
                and stats.is_contiguous()
            )
        if not ok:
            return orig(*args, **kwargs)
        attn_scale = a.get("attn_scale")
        if attn_scale is None:
            attn_scale = 1.0 / (qv.shape[-1] ** 0.5)
        rope = getattr(q, "_opt_rope", None)
        if rope is not None:
            from . import rope_fp8_qkv_patch as rp
            res = rp.attn_bwd_dqkv(
                rope, qv, kv, vv, ov, dov,
                stats.view(stats.shape[0], stats.shape[1], stats.shape[2]),
                q._scale_inv, k_._scale_inv, v._scale_inv, do._scale_inv,
                sq.scale, dpq.scale, dpq.amax.view(-1)[:1],
                float(attn_scale), mask == "causal", fmt == "sbhd",
            )
            if res is not None:
                return res
        dq, dk, dv = torch.ops.optimized.fp8_attn_bwd(
            qv, kv, vv, ov, dov, stats.view(stats.shape[0], stats.shape[1], stats.shape[2]),
            q._scale_inv, k_._scale_inv, v._scale_inv, do._scale_inv,
            sq.scale, dpq.scale, dpq.amax.view(-1)[:1],
            float(attn_scale), mask == "causal", fmt == "sbhd",
        )
        if not _announced:
            _announced = True
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[opt fp8_attn_bwd] CuTe DSL FP8 attention backward active "
                      f"(q {tuple(qv.shape)}, kv heads {kv.shape[2]}, {layout}, {mask})",
                      flush=True)
        dq, dk, dv = [t.permute(*perm) for t in (dq, dk, dv)]
        return [dq, dk, dv, None, None]

    fused_attn_bwd._opt_wrapped = True
    fused_attn_bwd._orig = orig
    be.fused_attn_bwd = fused_attn_bwd


if _ENABLED:
    _install()
