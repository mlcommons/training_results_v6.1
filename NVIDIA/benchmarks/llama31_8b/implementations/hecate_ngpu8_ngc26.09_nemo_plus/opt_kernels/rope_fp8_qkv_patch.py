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


_ENABLED = os.environ.get("OPT_ROPE_FP8_QKV", "1") == "1"
_announced = set()


def _announce(key, msg):
    if key not in _announced:
        _announced.add(key)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[opt rope_fp8_qkv] {msg}", flush=True)


class RopeCtx:

    __slots__ = ("mixed", "freqs", "cos", "sin", "split", "s", "b", "g", "hq", "d", "consumed",
                 "materialized", "dqkv", "__weakref__")

    def __init__(self, mixed, freqs, split):
        self.mixed = mixed
        self.freqs = freqs
        self.cos = self.sin = None
        self.split = list(split)
        self.s, self.b, self.g, _ = mixed.shape
        self.d = split[1]
        self.hq = split[0] // split[1]
        self.consumed = False
        self.materialized = False
        self.dqkv = None


_te = {}
_ZERO = {}


def _install():
    import transformer_engine_torch as tex
    import transformer_engine.pytorch.attention.rope as terope
    import transformer_engine.pytorch.attention.dot_product_attention.backends as be
    from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
        DotProductAttention,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8CurrentScalingQuantizer,
        Float8Tensor,
    )
    import megatron.core.transformer.attention as ma

    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    except ImportError:
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    from . import rope_fp8_qkv as K  # noqa: F401  (registers torch.ops.optimized.*)

    if getattr(ma.apply_fused_qkv_rotary_pos_emb, "_opt_wrapped", False):
        return
    E4M3, E5M2 = tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2
    FP8_TORCH = {int(E4M3): torch.float8_e4m3fn, int(E5M2): torch.float8_e5m2}
    sbhd = terope.QKVFormat["sbhd"]
    orig_rope = ma.apply_fused_qkv_rotary_pos_emb
    orig_cq = be.combine_and_quantize
    orig_bwd_impl = be._fused_attn_backward_impl
    orig_dpa_fwd = DotProductAttention.forward
    _te["orig_rope"] = orig_rope

    def _fp8_attention_active():
        if not FP8GlobalStateManager.is_fp8_enabled():
            return False
        return bool(getattr(FP8GlobalStateManager.get_fp8_recipe(), "fp8_dpa", False))

    class _DeferredRoPE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mixed, rc):
            s, b, g, _ = mixed.shape
            kw = dict(dtype=mixed.dtype, device=mixed.device)
            q = torch.empty(s, b, g * rc.hq, rc.d, **kw)
            k = torch.empty(s, b, g, rc.d, **kw)
            v = torch.empty(s, b, g, rc.d, **kw)
            ctx.rc = rc
            return q, k, v

        @staticmethod
        def backward(ctx, dq, dk, dv):
            rc = ctx.rc
            ctx.rc = None
            if rc.dqkv is not None:
                out, rc.dqkv = rc.dqkv, None
                return out, None
            out = tex.fused_qkv_rope_backward(
                dq.contiguous(), dk.contiguous(), dv.contiguous(), rc.freqs, rc.freqs, rc.split,
                sbhd, False, 1, 0,
            )
            return out, None

    def _materialize(q, k, v, rc):
        with torch.no_grad():
            qr, kr, vr = orig_rope(rc.mixed, rc.freqs, rc.freqs, rc.split)
            q.copy_(qr)
            k.copy_(kr)
            v.copy_(vr)
        rc.consumed = rc.materialized = True
        rc.mixed = None

    def apply_fused_qkv_rotary_pos_emb(qkv, q_freqs, k_freqs, qkv_split_arg_list,
                                       tensor_format="sbhd", start_positions=None,
                                       interleaved=False, cu_seqlens=None, cp_size=1, cp_rank=0):
        split = list(qkv_split_arg_list)
        ok = (
            _fp8_attention_active()
            and start_positions is None and tensor_format == "sbhd" and not interleaved
            and cu_seqlens is None and cp_size == 1 and qkv.dim() == 4 and qkv.is_cuda
            and qkv.dtype == torch.bfloat16 and qkv.is_contiguous()
            and len(split) == 3 and split[1] == split[2] == 128 and split[0] % split[1] == 0
            and qkv.shape[3] == sum(split)
            and (q_freqs is k_freqs or (q_freqs.data_ptr() == k_freqs.data_ptr()
                                        and q_freqs.shape == k_freqs.shape
                                        and q_freqs.stride() == k_freqs.stride()))
            and q_freqs.dim() == 4 and q_freqs.shape[1] == 1 and q_freqs.shape[2] == 1
            and q_freqs.shape[3] == split[1] and q_freqs.shape[0] >= qkv.shape[0]
            and q_freqs.is_cuda
        )
        if not ok:
            return orig_rope(qkv, q_freqs, k_freqs, qkv_split_arg_list, tensor_format=tensor_format,
                             start_positions=start_positions, interleaved=interleaved,
                             cu_seqlens=cu_seqlens, cp_size=cp_size, cp_rank=cp_rank)
        freqs = q_freqs if q_freqs.dtype == torch.float32 else q_freqs.float()
        if not freqs.is_contiguous():
            freqs = freqs.contiguous()
        rc = RopeCtx(qkv, freqs, split)
        q, k, v = _DeferredRoPE.apply(qkv, rc)
        for t in (q, k, v):
            t._opt_rope = rc
        return q, k, v

    apply_fused_qkv_rotary_pos_emb._opt_wrapped = True

    def combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer, used_in_forward=True,
                             used_in_backward=False, keep_same_data_and_scale_inv_format=False,
                             combined_qkv=None, combined_kv=None):
        rc = getattr(q, "_opt_rope", None)
        if rc is None or rc.consumed:
            return orig_cq(qkv_layout, q, k, v, qkv_quantizer, used_in_forward, used_in_backward,
                           keep_same_data_and_scale_inv_format, combined_qkv, combined_kv)
        ok = (
            type(qkv_quantizer) is Float8CurrentScalingQuantizer
            and not qkv_quantizer.with_amax_reduction
            and int(qkv_quantizer.dtype) == int(E4M3)
            and qkv_layout == "sbhd_sbhd_sbhd"
            and combined_qkv is None and combined_kv is None
            and getattr(k, "_opt_rope", None) is rc and getattr(v, "_opt_rope", None) is rc
            and q.dtype == rc.mixed.dtype
        )
        if ok:
            s, b, g, d, hq = rc.s, rc.b, rc.g, rc.d, rc.hq
            nq, nk = s * b * g * hq * d, s * b * g * d
            qkv_fp8 = qkv_quantizer.make_empty((nq + 2 * nk,), dtype=q.dtype, device=q.device)
            ok = (qkv_fp8._data is not None and qkv_fp8._transpose is None
                  and qkv_fp8._scale_inv.numel() == 1)
        if not ok:
            _materialize(q, k, v, rc)
            return orig_cq(qkv_layout, q, k, v, qkv_quantizer, used_in_forward, used_in_backward,
                           keep_same_data_and_scale_inv_format, combined_qkv, combined_kv)
        data = qkv_fp8._data
        rc.cos, rc.sin = K.trig_tables(rc.freqs)
        torch.ops.optimized.rope_qkv_fp8(
            rc.mixed, rc.cos, rc.sin, data.view(torch.float8_e4m3fn), qkv_fp8._scale_inv, hq,
            float(qkv_quantizer.amax_epsilon), bool(qkv_quantizer.force_pow_2_scales),
        )
        views = (data[:nq].view(q.shape), data[nq:nq + nk].view(k.shape),
                 data[nq + nk:].view(v.shape))
        q_fp8, k_fp8, v_fp8 = [Float8Tensor.make_like(qkv_fp8, data=x, dtype=q.dtype)
                               for x in views]
        q_fp8._opt_rope = rc
        rc.consumed = True
        rc.mixed = None
        _announce("fwd", f"fused RoPE + FP8 Q/K/V quantize active (qkv {(s, b, g, (hq + 2) * d)})")
        return q_fp8, k_fp8, v_fp8, qkv_layout, None

    def dpa_forward(self, query_layer, key_layer, value_layer, *args, **kwargs):
        rc = getattr(query_layer, "_opt_rope", None)
        if rc is not None and not rc.consumed and not _fp8_attention_active():
            _announce("materialize", "deferred RoPE placeholders materialized before a non-FP8 attention "
                      "consumer")
            _materialize(query_layer, key_layer, value_layer, rc)
        out = orig_dpa_fwd(self, query_layer, key_layer, value_layer, *args, **kwargs)
        if rc is not None and not rc.consumed:
            raise RuntimeError(
                "opt rope_fp8_qkv: deferred RoPE placeholders reached a DotProductAttention "
                "path that did not quantize them; run with OPT_ROPE_FP8_QKV=0")
        return out

    def _fused_attn_backward_impl(args):
        d_out = args.grad_output
        qz = args.dO_quantizer
        if (
            args.fp8 and not args.use_FAv2_bwd and isinstance(d_out, torch.Tensor)
            and not isinstance(d_out, be.QuantizedTensorStorage)
            and type(qz) is Float8CurrentScalingQuantizer and not qz.with_amax_reduction
            and int(qz.dtype) in FP8_TORCH and d_out.dtype == torch.bfloat16 and d_out.is_cuda
            and d_out.numel() > 0
        ):
            x = d_out.contiguous()
            o = qz.make_empty(x.shape, dtype=x.dtype, device=x.device)
            if o._data is not None and o._transpose is None and o._scale_inv.numel() == 1:
                torch.ops.optimized.fp8_cs_quantize(
                    x, o._data.view(FP8_TORCH[int(qz.dtype)]), o._scale_inv,
                    float(qz.amax_epsilon), bool(qz.force_pow_2_scales))
                args.grad_output = o
                _announce("do", "fused dO amax + FP8 cast active")
        return orig_bwd_impl(args)

    ma.apply_fused_qkv_rotary_pos_emb = apply_fused_qkv_rotary_pos_emb
    be.combine_and_quantize = combine_and_quantize
    be._fused_attn_backward_impl = _fused_attn_backward_impl
    DotProductAttention.forward = dpa_forward


def attn_bwd_dqkv(rc, qv, kv, vv, ov, dov, stats, q_si, k_si, v_si, do_si, s_scale, dp_scale,
                  dp_amax, attn_scale, causal, out_sbhd):
    if (not _ENABLED or rc is None or rc.materialized or rc.dqkv is not None or not out_sbhd
            or rc.cos is None):
        return None
    b, s, h, d = qv.shape
    g = kv.shape[2]
    if (s, b, g, h // g, d) != (rc.s, rc.b, rc.g, rc.hq, rc.d) or s % 128 != 0:
        return None
    from . import fp8_attn_bwd_sm100 as fab

    hq = rc.hq
    dqkv = torch.empty(s, b, g, (hq + 2) * d, dtype=torch.bfloat16, device=qv.device)
    dk_out = dqkv[..., hq * d:(hq + 1) * d].permute(1, 0, 2, 3)
    dv_out = dqkv[..., (hq + 1) * d:].permute(1, 0, 2, 3)

    def post(dq_acc, scales):
        torch.ops.optimized.dqkv_rope_post(dq_acc, scales, rc.cos, rc.sin, dqkv, b, s, h)

    fab._fp8_attn_bwd_impl(qv, kv, vv, ov, dov, stats, q_si, k_si, v_si, do_si, s_scale, dp_scale,
                           dp_amax, attn_scale, causal, out_sbhd, dk_out=dk_out, dv_out=dv_out,
                           post=post)
    rc.dqkv = dqkv
    _announce("bwd", "fused dQ conversion + inverse RoPE into dQKV active")
    z = _ZERO.get(qv.device)
    if z is None:
        z = _ZERO[qv.device] = torch.zeros((), dtype=torch.bfloat16, device=qv.device)
    return [z.expand(s, b, h, d), z.expand(s, b, g, d), z.expand(s, b, g, d), None, None]


if _ENABLED:
    _install()
