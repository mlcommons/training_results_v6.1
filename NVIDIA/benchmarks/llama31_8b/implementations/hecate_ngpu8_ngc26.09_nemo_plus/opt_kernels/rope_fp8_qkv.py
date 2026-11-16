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
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    from triton.language.extra.cuda import libdevice

FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}

_FMA = int(os.environ.get("OPT_ROPE_FMA", "0"))


@triton.jit
def _rot(x, c, xr, s, FMA: tl.constexpr):
    if FMA == 0:
        y = tl.fma(x, c, xr * s)
    else:
        y = tl.fma(xr, s, x * c)
    return y


@triton.jit
def _ld(ptr, mask, EVEN: tl.constexpr):
    if EVEN:
        return tl.load(ptr)
    else:
        return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def _st(ptr, val, mask, EVEN: tl.constexpr):
    if EVEN:
        tl.store(ptr, val)
    else:
        tl.store(ptr, val, mask=mask)


@triton.jit
def _finalize(ws_ptr, scale_inv_ptr, n_prog, eps, FP8MAX: tl.constexpr, POW2: tl.constexpr):
    cnt = tl.atomic_add(ws_ptr + 1, 1.0)
    if cnt == n_prog - 1:
        amax = tl.atomic_xchg(ws_ptr, 0.0)
        tl.atomic_xchg(ws_ptr + 1, 0.0)
        bad = (amax == float("inf")) | (amax != amax)
        amax = tl.where(amax < eps, eps, amax)
        bad = bad | (amax == 0.0)
        scale = libdevice.div_rn(FP8MAX, amax)
        scale = tl.where(scale == float("inf"), 3.4028234663852886e38, scale)
        if POW2:
            scale = (scale.to(tl.uint32, bitcast=True) & 0xFF800000).to(tl.float32, bitcast=True)
        scale = tl.where(bad, 1.0, scale)
        tl.store(ws_ptr + 2, scale)
        tl.store(scale_inv_ptr, libdevice.rcp_rn(scale))


@triton.jit
def _trig_kernel(f_ptr, cos_ptr, sin_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    f = tl.load(f_ptr + offs, mask=m, other=0.0)
    tl.store(cos_ptr + offs, libdevice.cos(f), mask=m)
    tl.store(sin_ptr + offs, libdevice.sin(f), mask=m)


@triton.jit
def _rot_fwd(lo, hi, c_lo, s_lo, c_hi, s_hi, FMA: tl.constexpr):
    y_lo = _rot(lo, c_lo, -hi, s_lo, FMA).to(tl.bfloat16).to(tl.float32)
    y_hi = _rot(hi, c_hi, lo, s_hi, FMA).to(tl.bfloat16).to(tl.float32)
    return y_lo, y_hi


@triton.jit
def _rope_qkv_tile(
    x_ptr, cos_ptr, sin_ptr, out_ptr, sc, blk, n_tok,
    B: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr, CAST: tl.constexpr, FMA: tl.constexpr, EVEN: tl.constexpr,
):
    W: tl.constexpr = (HQ + 2) * D
    HD: tl.constexpr = D // 2
    t = blk * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < n_tok
    pos = t // B
    g = tl.arange(0, G)
    jq = tl.arange(0, HQ)
    dl = tl.arange(0, HD)
    d = tl.arange(0, D)
    tb = pos[:, None] * D + dl[None, :]
    m2 = tmask[:, None]
    c_lo = _ld(cos_ptr + tb, m2, EVEN)
    s_lo = _ld(sin_ptr + tb, m2, EVEN)
    c_hi = _ld(cos_ptr + tb + HD, m2, EVEN)
    s_hi = _ld(sin_ptr + tb + HD, m2, EVEN)
    row = t[:, None] * (G * W) + g[None, :] * W
    m3 = tmask[:, None, None]
    m4 = tmask[:, None, None, None]
    q_src = x_ptr + row[:, :, None, None] + jq[None, None, :, None] * D + dl[None, None, None, :]
    k_src = x_ptr + row[:, :, None] + HQ * D + dl[None, None, :]
    v_src = x_ptr + row[:, :, None] + (HQ + 1) * D + d[None, None, :]
    q_lo, q_hi = _rot_fwd(
        _ld(q_src, m4, EVEN).to(tl.float32),
        _ld(q_src + HD, m4, EVEN).to(tl.float32),
        c_lo[:, None, None, :], s_lo[:, None, None, :], c_hi[:, None, None, :],
        s_hi[:, None, None, :], FMA)
    k_lo, k_hi = _rot_fwd(
        _ld(k_src, m3, EVEN).to(tl.float32),
        _ld(k_src + HD, m3, EVEN).to(tl.float32),
        c_lo[:, None, :], s_lo[:, None, :], c_hi[:, None, :], s_hi[:, None, :], FMA)
    v = _ld(v_src, m3, EVEN).to(tl.float32)
    if CAST:
        nq = n_tok * (G * HQ * D)
        nk = n_tok * (G * D)
        q_dst = (out_ptr + t[:, None, None, None] * (G * HQ * D) + g[None, :, None, None] * (HQ * D)
                 + jq[None, None, :, None] * D + dl[None, None, None, :])
        k_dst = out_ptr + nq + t[:, None, None] * (G * D) + g[None, :, None] * D + dl[None, None, :]
        v_dst = out_ptr + nq + nk + t[:, None, None] * (G * D) + g[None, :, None] * D + d[None, None, :]
        ety = out_ptr.dtype.element_ty
        _st(q_dst, (q_lo * sc).to(ety), m4, EVEN)
        _st(q_dst + HD, (q_hi * sc).to(ety), m4, EVEN)
        _st(k_dst, (k_lo * sc).to(ety), m3, EVEN)
        _st(k_dst + HD, (k_hi * sc).to(ety), m3, EVEN)
        _st(v_dst, (v * sc).to(ety), m3, EVEN)
        return 0.0
    else:
        aq = tl.max(tl.max(tl.max(tl.max(tl.maximum(tl.abs(q_lo), tl.abs(q_hi)), 3), 2), 1), 0)
        ak = tl.max(tl.max(tl.max(tl.maximum(tl.abs(k_lo), tl.abs(k_hi)), 2), 1), 0)
        av = tl.max(tl.max(tl.max(tl.abs(v), 2), 1), 0)
        return tl.maximum(tl.maximum(aq, ak), av)


@triton.jit
def _rope_qkv_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr, ws_ptr, scale_inv_ptr, n_tok, eps,
    B: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr, CAST: tl.constexpr, FP8MAX: tl.constexpr, POW2: tl.constexpr,
    FMA: tl.constexpr, RAW: tl.constexpr = False, NS: tl.constexpr = 1,
    EVEN: tl.constexpr = False,
):
    n_prog = tl.num_programs(0)
    pid = tl.program_id(0)
    n_blk = tl.cdiv(n_tok, BLOCK_T)
    if CAST:
        if RAW:
            sc = 1.0
        else:
            sc = tl.load(ws_ptr + 2)
        _rope_qkv_tile(x_ptr, cos_ptr, sin_ptr, out_ptr, sc, n_prog - 1 - pid, n_tok,
                       B, G, HQ, D, BLOCK_T, True, FMA, EVEN)
    else:
        amax = 0.0
        for blk in tl.range(pid, n_blk, n_prog, num_stages=NS):
            a = _rope_qkv_tile(x_ptr, cos_ptr, sin_ptr, out_ptr, 1.0, blk, n_tok,
                               B, G, HQ, D, BLOCK_T, False, FMA, EVEN)
            amax = tl.maximum(amax, a)
        tl.atomic_max(ws_ptr, amax)
        _finalize(ws_ptr, scale_inv_ptr, n_prog, eps, FP8MAX, POW2)


@triton.jit
def _cs_kernel(x_ptr, out_ptr, ws_ptr, scale_inv_ptr, n, eps, BLOCK: tl.constexpr,
               CAST: tl.constexpr, FP8MAX: tl.constexpr, POW2: tl.constexpr):
    n_prog = tl.num_programs(0)
    pid = tl.program_id(0)
    if CAST:
        offs = (n_prog - 1 - pid).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        scale = tl.load(ws_ptr + 2)
        tl.store(out_ptr + offs, (x * scale).to(out_ptr.dtype.element_ty), mask=m)
    else:
        amax = tl.zeros([BLOCK], tl.float32)
        for blk in tl.range(pid, tl.cdiv(n, BLOCK), n_prog):
            offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(x))
        tl.atomic_max(ws_ptr, tl.max(amax, axis=0))
        _finalize(ws_ptr, scale_inv_ptr, n_prog, eps, FP8MAX, POW2)


@triton.jit
def _dqkv_rope_post_kernel(
    acc_ptr, scales_ptr, cos_ptr, sin_ptr, dqkv_ptr, S, n_q_progs, n_k_rows,
    B: tl.constexpr, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, FMA: tl.constexpr,
):
    HQ: tl.constexpr = H // G
    W: tl.constexpr = (HQ + 2) * D
    HD: tl.constexpr = D // 2
    TR: tl.constexpr = 64
    pid = tl.program_id(0)
    dl = tl.arange(0, HD)
    rr = tl.arange(0, TR)
    r = (pid - n_q_progs).to(tl.int64) * TR + rr
    rm = (r < n_k_rows)[:, None]
    if pid < n_q_progs:
        rank = pid % 2
        rest = pid // 2
        n_mb = S // 128
        mb = rest % n_mb
        bh = rest // n_mb
        h = bh % H
        b = bh // H
        base = acc_ptr + (bh.to(tl.int64) * S + mb * 128) * D + rank * (8 * 1024)
        k16 = tl.arange(0, 16)
        jj = tl.arange(0, 4)
        off = k16[:, None, None] * 512 + rr[None, :, None] * 4 + jj[None, None, :]
        sc = tl.load(scales_ptr + 4)
        lo = tl.reshape(tl.permute(tl.load(base + off), [1, 0, 2]), [TR, HD])
        hi = tl.reshape(tl.permute(tl.load(base + 256 + off), [1, 0, 2]), [TR, HD])
        lo = (lo * sc).to(tl.bfloat16).to(tl.float32)
        hi = (hi * sc).to(tl.bfloat16).to(tl.float32)
        s_pos = (mb * 128 + rank * 64 + rr).to(tl.int64)
        dst = dqkv_ptr + ((s_pos * B + b) * G + h // HQ)[:, None] * W + (h % HQ) * D
    else:
        s_pos = r // (B * G)
        dst = dqkv_ptr + r[:, None] * W + HQ * D
        lo = tl.load(dst + dl[None, :], mask=rm, other=0.0).to(tl.float32)
        hi = tl.load(dst + HD + dl[None, :], mask=rm, other=0.0).to(tl.float32)
    s_pos = tl.where(s_pos < S, s_pos, 0)
    tb = s_pos[:, None] * D + dl[None, :]
    o_lo = _rot(lo, tl.load(cos_ptr + tb), hi, tl.load(sin_ptr + tb + HD), FMA).to(tl.bfloat16)
    o_hi = _rot(hi, tl.load(cos_ptr + tb + HD), lo, -tl.load(sin_ptr + tb), FMA).to(tl.bfloat16)
    if pid < n_q_progs:
        tl.store(dst + dl[None, :], o_lo)
        tl.store(dst + HD + dl[None, :], o_hi)
    else:
        tl.store(dst + dl[None, :], o_lo, mask=rm)
        tl.store(dst + HD + dl[None, :], o_hi, mask=rm)


_WS = {}
_BLOCK_T = int(os.environ.get("OPT_ROPE_BLOCK_T", "1"))
_ROPE_WARPS = int(os.environ.get("OPT_ROPE_WARPS", "2"))
_CAST_WARPS = int(os.environ.get("OPT_CAST_WARPS", "4"))
_POST_WARPS = int(os.environ.get("OPT_POST_WARPS", "8"))
_CS_BLOCK = 8192
_AMAX_STAGES = int(os.environ.get("OPT_AMAX_STAGES", "3"))
_AMAX_CTAS = int(os.environ.get("OPT_AMAX_CTAS_PER_SM", "4"))
_SMS = {}


def _persistent(device, per_sm):
    if device not in _SMS:
        _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device] * per_sm


def _ws(device, slot):
    key = (device, slot)
    if key not in _WS:
        _WS[key] = torch.zeros(4, dtype=torch.float32, device=device)
    return _WS[key]


@torch.library.custom_op("optimized::rope_trig_table", mutates_args=("cos", "sin"),
                         device_types="cuda")
def rope_trig_table(freqs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    n = freqs.numel()
    _trig_kernel[(triton.cdiv(n, 1024),)](freqs, cos, sin, n, BLOCK=1024, num_warps=4)


@rope_trig_table.register_fake
def _(freqs, cos, sin):
    return None


_TRIG = {}


def trig_tables(freqs2d: torch.Tensor):
    ent = _TRIG.get(freqs2d.device)
    if ent is not None and ent[0] is freqs2d and ent[1] == freqs2d._version:
        return ent[2], ent[3]
    cos = torch.empty_like(freqs2d)
    sin = torch.empty_like(freqs2d)
    torch.ops.optimized.rope_trig_table(freqs2d, cos, sin)
    _TRIG[freqs2d.device] = (freqs2d, freqs2d._version, cos, sin)
    return cos, sin


@torch.library.custom_op("optimized::rope_qkv_fp8", mutates_args=("out", "scale_inv"),
                         device_types="cuda")
def rope_qkv_fp8(mixed_qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 out: torch.Tensor, scale_inv: torch.Tensor, n_q_per_group: int, eps: float,
                 pow2: bool) -> None:
    s, b, g, w = mixed_qkv.shape
    d = cos.shape[-1]
    hq = n_q_per_group
    n_tok = s * b
    ws = _ws(mixed_qkv.device, "qkv")
    n_blk = triton.cdiv(n_tok, _BLOCK_T)
    common = dict(B=b, G=g, HQ=hq, D=d, BLOCK_T=_BLOCK_T, FP8MAX=FP8_MAX[out.dtype], POW2=pow2,
                  FMA=_FMA, EVEN=n_tok % _BLOCK_T == 0)
    _rope_qkv_kernel[(min(n_blk, _persistent(mixed_qkv.device, _AMAX_CTAS)),)](
        mixed_qkv, cos, sin, out, ws, scale_inv, n_tok, eps, CAST=False, num_warps=_ROPE_WARPS,
        NS=_AMAX_STAGES, **common)
    _rope_qkv_kernel[(n_blk,)](mixed_qkv, cos, sin, out, ws, scale_inv, n_tok, eps, CAST=True,
                               num_warps=_CAST_WARPS, **common)


@rope_qkv_fp8.register_fake
def _(mixed_qkv, cos, sin, out, scale_inv, n_q_per_group, eps, pow2):
    return None


@torch.library.custom_op("optimized::fp8_cs_quantize", mutates_args=("out", "scale_inv"),
                         device_types="cuda")
def fp8_cs_quantize(x: torch.Tensor, out: torch.Tensor, scale_inv: torch.Tensor, eps: float,
                    pow2: bool) -> None:
    n = x.numel()
    ws = _ws(x.device, "cs")
    n_blk = triton.cdiv(n, _CS_BLOCK)
    common = dict(BLOCK=_CS_BLOCK, FP8MAX=FP8_MAX[out.dtype], POW2=pow2, num_warps=8)
    _cs_kernel[(min(n_blk, _persistent(x.device, _AMAX_CTAS)),)](
        x, out, ws, scale_inv, n, eps, CAST=False, **common)
    _cs_kernel[(n_blk,)](x, out, ws, scale_inv, n, eps, CAST=True, **common)


@fp8_cs_quantize.register_fake
def _(x, out, scale_inv, eps, pow2):
    return None


@torch.library.custom_op("optimized::dqkv_rope_post", mutates_args=("dqkv",), device_types="cuda")
def dqkv_rope_post(dq_acc: torch.Tensor, scales: torch.Tensor, cos: torch.Tensor,
                   sin: torch.Tensor, dqkv: torch.Tensor, b: int, s: int, h: int) -> None:
    _, _, g, w = dqkv.shape
    d = cos.shape[-1]
    assert s % 128 == 0 and d == 128
    n_q_progs = b * h * (s // 128) * 2
    n_k_rows = s * b * g
    grid = (n_q_progs + triton.cdiv(n_k_rows, 64),)
    _dqkv_rope_post_kernel[grid](dq_acc, scales, cos, sin, dqkv, s, n_q_progs, n_k_rows,
                                 B=b, H=h, G=g, D=d, FMA=_FMA, num_warps=_POST_WARPS)


@dqkv_rope_post.register_fake
def _(dq_acc, scales, cos, sin, dqkv, b, s, h):
    return None
