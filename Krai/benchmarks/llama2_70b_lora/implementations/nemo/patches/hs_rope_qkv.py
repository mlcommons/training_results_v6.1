# Copyright (c) 2026 Krai Ltd. All Rights Reserved.
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

"""Fix 4: fused qkv split + RoPE (forward) and fused RoPE backward + dq/dk/dv concat (backward), wired into fix #2's
attention mixin (hs_lora_epilogue.py, HS_ROPE_FUSED=1). Prototype and measurements: microbenchmark lane l2_rope,
campaign kfix_261010 (bit-exact against the apex chain at full shape incl. special values; fwd 0.47x, bwd 0.39x).

Replaces, per attention layer (Llama 2 70B, GQA 8 groups x (8 q + 1 k + 1 v) x 128, sbhd, b = 1):
  fwd: SplitAlongDim view -> q.reshape copy (`copy_`) -> apex fused_rope_forward(q), (k) -> TE get_qkv_layout's
       .contiguous() of v (and q, k, already contiguous)
  bwd: apex fused_rope_backward(dq), (dk) -> SplitAlongDim.backward torch.cat(dq, dk, dv) -> [s, 1, 10240]

Numerics mirror the ROCm apex kernel *as compiled* (gfx942 ISA of fused_rotary_positional_embedding.so, apex
1.8.0a0+rocm7.0.0.git3f26640c), which does the arithmetic in c10::BFloat16, not in fp32:
  fwd  c = bf16(cos f), s = bf16(sin f)    (sincosf, ocml, then RNE to bf16)
       d <  64: out = bf16( bf16(x[d] * c) + bf16(-x[d+64] * s) )
       d >= 64: out = bf16( bf16(x[d] * c) + bf16( x[d-64] * s) )
  bwd  c = bf16(cosf f[d]), s' = bf16(+sinf f[d+64]) for d < 64, bf16(-sinf f[d-64]) for d >= 64
       out = bf16( bf16(g[d] * c) + bf16(g[d +- 64] * s') )
The products of two bf16 values are exact in fp32, so each bf16() above is the only rounding; no FMA can form
because every product is rounded to bf16 before the add (kernels are still launched with enable_fp_fusion=False).
Negation is written `x * -1.0`, not `-x`: Triton lowers unary minus to `0 - x`, which turns +0 into +0 instead of -0
and broke bit-exactness on signed zeros (found by the special-value test).
Megatron's freqs are cat(f, f) (rotary_interleaved=False), so the table needs only the first 64 columns.

cos/sin source (TRIG constexpr): 0 = a bf16 [s, 64] table built once per freqs tensor (torch.cos / torch.sin, i.e.
ocml, rounded to bf16), 1 = in-kernel libdevice cos/sin (ocml) from the fp32 freqs.
"""
import weakref

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from hs_dswiglu_e5m2 import _e5m2_bits   # TE unary_kernel e5m2 semantics (fix 8c: optional e5m2 copy of d_mixed)

G, QH, D = 8, 8, 128           # query groups, q heads per group, head dim
ROW = G * (QH + 2) * D         # 10240
HALF = D // 2


@triton.jit
def _cs(cos_ptr, sin_ptr, f_ptr, t, TRIG: tl.constexpr):
    d = tl.arange(0, 64)
    if TRIG:
        f = tl.load(f_ptr + t[:, None] * 128 + d[None, :])
        c = libdevice.cos(f).to(tl.bfloat16).to(tl.float32)
        s = libdevice.sin(f).to(tl.bfloat16).to(tl.float32)
    else:
        c = tl.load(cos_ptr + t[:, None] * 64 + d[None, :]).to(tl.float32)
        s = tl.load(sin_ptr + t[:, None] * 64 + d[None, :]).to(tl.float32)
    return c[:, None, :], s[:, None, :]


@triton.jit
def _rne_bits(x):
    # fp32 -> RNE bf16 in the upper 16 bits, no NaN check: exact for every finite / inf value and for NaNs whose
    # low 16 bits are zero, which is every NaN here (operands are bf16 values, so a propagated NaN keeps a bf16
    # payload and a generated one is 0x7fc00000). NaN payloads may differ from c10's 0x7fc0 (still NaN).
    b = x.to(tl.uint32, bitcast=True)
    return b + 0x7FFF + ((b >> 16) & 1)


@triton.jit
def _r32(x, RB: tl.constexpr):
    if RB:
        return ((_rne_bits(x) >> 16) << 16).to(tl.float32, bitcast=True)
    else:
        return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rot(a, b, c, s, RB: tl.constexpr):
    # bf16( bf16(a*c) + bf16(b*s) ), every op as c10::BFloat16 arithmetic does it
    y = _r32(a * c, RB) + _r32(b * s, RB)
    if RB:
        return (_rne_bits(y) >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    else:
        return y.to(tl.bfloat16)


@triton.jit
def _rope_qkv_fwd_kernel(x_ptr, cos_ptr, sin_ptr, f_ptr, q_ptr, k_ptr, v_ptr,
                         TOK: tl.constexpr, GPP: tl.constexpr, TRIG: tl.constexpr,
                         RB: tl.constexpr = 0):
    pid_t = tl.program_id(0)
    pid_g = tl.program_id(1)
    t = pid_t * TOK + tl.arange(0, TOK)
    c, s = _cs(cos_ptr, sin_ptr, f_ptr, t, TRIG)
    d = tl.arange(0, 64)
    t3 = t[:, None, None]
    d3 = d[None, None, :]
    # q: GPP groups x 8 heads
    hq = tl.arange(0, GPP * 8)
    gq = pid_g * GPP + hq // 8
    src = x_ptr + t3 * 10240 + (gq * 1280 + (hq % 8) * 128)[None, :, None] + d3
    x1 = tl.load(src).to(tl.float32)
    x2 = tl.load(src + 64).to(tl.float32)
    dst = q_ptr + t3 * 8192 + ((pid_g * GPP * 8 + hq) * 128)[None, :, None] + d3
    tl.store(dst, _rot(x1, x2 * -1.0, c, s, RB))
    tl.store(dst + 64, _rot(x2, x1, c, s, RB))
    # k: GPP groups x 1 head
    gk = pid_g * GPP + tl.arange(0, GPP)
    src = x_ptr + t3 * 10240 + (gk * 1280 + 1024)[None, :, None] + d3
    x1 = tl.load(src).to(tl.float32)
    x2 = tl.load(src + 64).to(tl.float32)
    dst = k_ptr + t3 * 1024 + (gk * 128)[None, :, None] + d3
    tl.store(dst, _rot(x1, x2 * -1.0, c, s, RB))
    tl.store(dst + 64, _rot(x2, x1, c, s, RB))
    # v: copy
    e = tl.arange(0, 128)
    src = x_ptr + t3 * 10240 + (gk * 1280 + 1152)[None, :, None] + e[None, None, :]
    dst = v_ptr + t3 * 1024 + (gk * 128)[None, :, None] + e[None, None, :]
    tl.store(dst, tl.load(src))


@triton.jit
def _rope_qkv_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, f_ptr, out_ptr,
                         f8_ptr, scale_ptr, amax_ptr, sinv_ptr,
                         TOK: tl.constexpr, GPP: tl.constexpr, TRIG: tl.constexpr,
                         RB: tl.constexpr = 0, E5M2: tl.constexpr = 0):
    """E5M2 (fix 8c): also write the e5m2fnuz bytes of every stored BF16 value (TE unary_kernel semantics, scale from
    scale_ptr) to f8_ptr at the same offsets, max-accumulate |value| into amax_ptr (fp32 bits, atomic max) and
    1/scale into sinv_ptr. The BF16 output is computed and stored exactly as without it."""
    pid_t = tl.program_id(0)
    pid_g = tl.program_id(1)
    t = pid_t * TOK + tl.arange(0, TOK)
    c, s = _cs(cos_ptr, sin_ptr, f_ptr, t, TRIG)
    d = tl.arange(0, 64)
    t3 = t[:, None, None]
    d3 = d[None, None, :]
    if E5M2:
        scale = tl.load(scale_ptr)
        m = tl.zeros((TOK, ), dtype=tl.float32)
    hq = tl.arange(0, GPP * 8)
    gq = pid_g * GPP + hq // 8
    src = dq_ptr + t3 * 8192 + ((pid_g * GPP * 8 + hq) * 128)[None, :, None] + d3
    g1 = tl.load(src).to(tl.float32)
    g2 = tl.load(src + 64).to(tl.float32)
    off = t3 * 10240 + (gq * 1280 + (hq % 8) * 128)[None, :, None] + d3
    o1 = _rot(g1, g2, c, s, RB)
    o2 = _rot(g2, g1, c, s * -1.0, RB)
    tl.store(out_ptr + off, o1)
    tl.store(out_ptr + off + 64, o2)
    if E5M2:
        f1 = o1.to(tl.float32)
        f2 = o2.to(tl.float32)
        tl.store(f8_ptr + off, _e5m2_bits(f1, scale))
        tl.store(f8_ptr + off + 64, _e5m2_bits(f2, scale))
        m = tl.maximum(m, tl.max(tl.max(tl.maximum(tl.abs(f1), tl.abs(f2)), axis=2), axis=1))
    gk = pid_g * GPP + tl.arange(0, GPP)
    src = dk_ptr + t3 * 1024 + (gk * 128)[None, :, None] + d3
    g1 = tl.load(src).to(tl.float32)
    g2 = tl.load(src + 64).to(tl.float32)
    off = t3 * 10240 + (gk * 1280 + 1024)[None, :, None] + d3
    o1 = _rot(g1, g2, c, s, RB)
    o2 = _rot(g2, g1, c, s * -1.0, RB)
    tl.store(out_ptr + off, o1)
    tl.store(out_ptr + off + 64, o2)
    if E5M2:
        f1 = o1.to(tl.float32)
        f2 = o2.to(tl.float32)
        tl.store(f8_ptr + off, _e5m2_bits(f1, scale))
        tl.store(f8_ptr + off + 64, _e5m2_bits(f2, scale))
        m = tl.maximum(m, tl.max(tl.max(tl.maximum(tl.abs(f1), tl.abs(f2)), axis=2), axis=1))
    e = tl.arange(0, 128)
    src = dv_ptr + t3 * 1024 + (gk * 128)[None, :, None] + e[None, None, :]
    off = t3 * 10240 + (gk * 1280 + 1152)[None, :, None] + e[None, None, :]
    vv = tl.load(src)
    tl.store(out_ptr + off, vv)
    if E5M2:
        fv = vv.to(tl.float32)
        tl.store(f8_ptr + off, _e5m2_bits(fv, scale))
        m = tl.maximum(m, tl.max(tl.max(tl.abs(fv), axis=2), axis=1))
        # amax of |BF16 d_mixed|: maximum ignores NaN like fmaxf; all-NaN -> 0 as TE's `max = 0` init
        mx = tl.maximum(tl.max(m, axis=0), 0.0)
        mi = mx.to(tl.int32, bitcast=True)
        cur = tl.load(amax_ptr, volatile=True)
        tl.atomic_max(amax_ptr, mi, mask=mi > cur, sem="relaxed")
        tl.store(sinv_ptr, 1.0 / scale, mask=(pid_t == 0) & (pid_g == 0))


# ------------------------------------------------------------------------------------------------ host side

_TABLE = {}


def cos_sin_table(freqs):
    """bf16 cos / sin [s, 64] of Megatron's fp32 freqs [s, 1, 1, 128] (= cat(f, f)), cached on the freqs object
    itself (weakref + its version counter), so a recycled address can never serve a stale table. In training the
    RotaryEmbedding output is one tensor per step shared by every layer: one 18 us build per step."""
    ent = _TABLE.get("t")
    if ent is not None and ent[0]() is freqs and ent[1] == freqs._version:
        return ent[2]
    f = freqs.reshape(freqs.shape[0], -1)[:, :64].float()
    tab = (torch.cos(f).to(torch.bfloat16).contiguous(), torch.sin(f).to(torch.bfloat16).contiguous())
    _TABLE["t"] = (weakref.ref(freqs), freqs._version, tab)
    return tab


# lane sweet spot (TOK 1, GPP 8, 8 warps, cos/sin table) with Triton's own bf16 conversion (RB 0, +1.3% vs the bit
# trick) so no assumption on NaN payloads is needed
DEFAULT = dict(TOK=1, GPP=8, num_warps=8, TRIG=0, RB=0)


def rope_qkv_fwd(mixed, freqs, cfg=None, tab=None):
    """mixed [s, 1, 10240] bf16 contiguous -> (q [s,1,64,128], k [s,1,8,128], v [s,1,8,128]) contiguous."""
    cfg = cfg or DEFAULT
    S = mixed.shape[0]
    assert mixed.is_contiguous() and mixed.shape[-1] == ROW and mixed.numel() == S * ROW
    assert freqs.is_contiguous() and freqs.dtype == torch.float32
    assert S % cfg["TOK"] == 0
    q = torch.empty(S, 1, G * QH, D, device=mixed.device, dtype=mixed.dtype)
    k = torch.empty(S, 1, G, D, device=mixed.device, dtype=mixed.dtype)
    v = torch.empty(S, 1, G, D, device=mixed.device, dtype=mixed.dtype)
    if tab is None:
        tab = cos_sin_table(freqs) if not cfg["TRIG"] else (freqs, freqs)
    grid = (S // cfg["TOK"], G // cfg["GPP"])
    _rope_qkv_fwd_kernel[grid](mixed, tab[0], tab[1], freqs, q, k, v, TOK=cfg["TOK"], GPP=cfg["GPP"],
                               TRIG=cfg["TRIG"], RB=cfg.get("RB", 0), num_warps=cfg["num_warps"], enable_fp_fusion=False)
    return q, k, v


def rope_qkv_bwd(dq, dk, dv, freqs, cfg=None, tab=None, e5m2=None):
    """dq [s,1,64,128], dk, dv [s,1,8,128] contiguous -> d_mixed [s, 1, 10240]. e5m2 (fix 8c): optional
    (f8 uint8 [s,1,10240], scale fp32 [1], amax int32 view of the quantizer's fp32 amax, scale_inv fp32 [1]):
    the e5m2 bytes of d_mixed are written alongside (TE quantizer semantics), amax max-accumulated in place."""
    cfg = cfg or DEFAULT
    S = dq.shape[0]
    assert dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous()
    out = torch.empty(S, 1, ROW, device=dq.device, dtype=dq.dtype)
    if tab is None:
        tab = cos_sin_table(freqs) if not cfg["TRIG"] else (freqs, freqs)
    grid = (S // cfg["TOK"], G // cfg["GPP"])
    if e5m2 is None:
        f8 = scale = amax = sinv = None
    else:
        f8, scale, amax, sinv = e5m2
        assert f8.dtype == torch.uint8 and f8.shape == out.shape and f8.is_contiguous()
        assert scale.dtype == sinv.dtype == torch.float32 and amax.dtype == torch.int32
    _rope_qkv_bwd_kernel[grid](dq, dk, dv, tab[0], tab[1], freqs, out, f8, scale, amax, sinv,
                               TOK=cfg["TOK"], GPP=cfg["GPP"], TRIG=cfg["TRIG"], RB=cfg.get("RB", 0),
                               E5M2=0 if e5m2 is None else 1, num_warps=cfg["num_warps"], enable_fp_fusion=False)
    return out


class FusedQKVRoPE(torch.autograd.Function):
    """mixed_qkv [s,1,10240] -> rotated q, rotated k, v (contiguous sbhd, TE layout sbhd_sbhd_sbhd)."""

    @staticmethod
    def forward(ctx, mixed, freqs, cfg_fwd=None, cfg_bwd=None):
        q, k, v = rope_qkv_fwd(mixed.contiguous(), freqs, cfg_fwd)
        ctx.save_for_backward(freqs)
        ctx.cfg_bwd = cfg_bwd
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        (freqs,) = ctx.saved_tensors
        return rope_qkv_bwd(dq.contiguous(), dk.contiguous(), dv.contiguous(), freqs, ctx.cfg_bwd), None, None, None


STATS = {"fused": 0}


def fused_qkv_rope(mixed, freqs):
    """mixed_qkv [s, 1, 10240] -> (rotated q [s,1,64,128], rotated k [s,1,8,128], v [s,1,8,128]), contiguous."""
    STATS["fused"] += 1
    return FusedQKVRoPE.apply(mixed, freqs, DEFAULT, DEFAULT)
