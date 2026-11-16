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

import argparse
import os
import sys
from typing import Tuple

import torch
import triton
import triton.language as tl

HD = 64
H2 = 32
_FWD_ROWS = int(os.environ.get("OPT_QKNORM_ROPE_FWD_ROWS", "32"))
_BWD_ROWS = int(os.environ.get("OPT_QKNORM_ROPE_BWD_ROWS", "32"))
_BWD_CTAS = int(os.environ.get("OPT_QKNORM_ROPE_BWD_CTAS", "8"))
_FWD_WARPS = int(os.environ.get("OPT_QKNORM_ROPE_FWD_WARPS", "4"))
_BWD_WARPS = int(os.environ.get("OPT_QKNORM_ROPE_BWD_WARPS", "4"))


@triton.jit(do_not_specialize=["S"])
def _qknorm_sincos_kernel(fq_ptr, fk_ptr, tab_ptr, S, BS: tl.constexpr):
    H2: tl.constexpr = 32
    pid = tl.program_id(0)
    sb = pid * BS + tl.arange(0, BS)
    m = (sb < S)[:, None]
    c = tl.arange(0, H2)[None, :]
    off = sb[:, None] * 64 + c
    aq = tl.load(fq_ptr + off, mask=m, other=0.0)
    ak = tl.load(fk_ptr + off, mask=m, other=0.0)
    to = sb[:, None] * H2 + c
    tl.store(tab_ptr + to, tl.cos(aq), mask=m)
    tl.store(tab_ptr + S * H2 + to, tl.sin(aq), mask=m)
    tl.store(tab_ptr + 2 * S * H2 + to, tl.cos(ak), mask=m)
    tl.store(tab_ptr + 3 * S * H2 + to, tl.sin(ak), mask=m)


@triton.jit(do_not_specialize=["B", "S", "NROWS"])
def _qknorm_rope_fwd_kernel(
    x_ptr,
    gq_ptr,
    gk_ptr,
    tab_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    B,
    S,
    NROWS,
    eps,
    NG: tl.constexpr,
    NQ: tl.constexpr,
    R: tl.constexpr,
    ZC: tl.constexpr,
):
    HD: tl.constexpr = 64
    H2: tl.constexpr = 32
    GR: tl.constexpr = NQ + 2
    pid = tl.program_id(0)
    row = pid * R + tl.arange(0, R)
    rm = row < NROWS
    t = row // (NG * GR)
    g = (row // GR) % NG
    rr = row % GR
    s_id = t // B
    is_q = rr < NQ
    is_k = rr == NQ
    is_v = rr == NQ + 1
    c2 = tl.max_contiguous(tl.multiple_of(tl.arange(0, H2), H2), H2)[None, :]
    mrow = rm[:, None]
    xrow = x_ptr + row.to(tl.int64) * HD
    xlo = tl.load(xrow[:, None] + c2, mask=mrow, other=0.0)
    xhi = tl.load(xrow[:, None] + H2 + c2, mask=mrow, other=0.0)
    flo = xlo.to(tl.float32)
    fhi = xhi.to(tl.float32)
    ss = tl.sum(flo * flo, axis=1) + tl.sum(fhi * fhi, axis=1)
    rs = tl.where(is_v, 1.0, tl.rsqrt(ss * (1.0 / HD) + eps))[:, None]
    gqlo = tl.load(gq_ptr + c2).to(tl.float32)
    gqhi = tl.load(gq_ptr + H2 + c2).to(tl.float32)
    gklo = tl.load(gk_ptr + c2).to(tl.float32)
    gkhi = tl.load(gk_ptr + H2 + c2).to(tl.float32)
    if ZC:
        gqlo += 1.0
        gqhi += 1.0
        gklo += 1.0
        gkhi += 1.0
    glo = tl.where(is_q[:, None], gqlo, tl.where(is_k[:, None], gklo, 1.0))
    ghi = tl.where(is_q[:, None], gqhi, tl.where(is_k[:, None], gkhi, 1.0))
    ylo = flo * rs * glo
    yhi = fhi * rs * ghi
    to = (tl.where(is_q, 0, 2 * S * H2) + s_id * H2)[:, None] + c2
    mrot = (rm & (is_q | is_k))[:, None]
    cs = tl.load(tab_ptr + to, mask=mrot, other=1.0)
    sn = tl.load(tab_ptr + S * H2 + to, mask=mrot, other=0.0)
    olo = (ylo * cs - yhi * sn).to(tl.bfloat16)
    ohi = (yhi * cs + ylo * sn).to(tl.bfloat16)
    qrow = q_ptr + ((t * NG + g) * NQ + rr).to(tl.int64) * HD
    krow = k_ptr + (t * NG + g).to(tl.int64) * HD
    vrow = v_ptr + (t * NG + g).to(tl.int64) * HD
    mq = (rm & is_q)[:, None]
    mk = (rm & is_k)[:, None]
    mv = (rm & is_v)[:, None]
    tl.store(qrow[:, None] + c2, olo, mask=mq)
    tl.store(qrow[:, None] + H2 + c2, ohi, mask=mq)
    tl.store(krow[:, None] + c2, olo, mask=mk)
    tl.store(krow[:, None] + H2 + c2, ohi, mask=mk)
    tl.store(vrow[:, None] + c2, xlo, mask=mv)
    tl.store(vrow[:, None] + H2 + c2, xhi, mask=mv)


@triton.jit(
    do_not_specialize=[
        "B",
        "S",
        "NROWS",
        "NBLK",
        "qs0",
        "qs1",
        "qs2",
        "ks0",
        "ks1",
        "ks2",
        "vs0",
        "vs1",
        "vs2",
    ]
)
def _qknorm_rope_bwd_kernel(
    dq_ptr,
    dk_ptr,
    dv_ptr,
    x_ptr,
    gq_ptr,
    gk_ptr,
    tab_ptr,
    dx_ptr,
    part_ptr,
    B,
    S,
    NROWS,
    NBLK,
    eps,
    qs0,
    qs1,
    qs2,
    ks0,
    ks1,
    ks2,
    vs0,
    vs1,
    vs2,
    NG: tl.constexpr,
    NQ: tl.constexpr,
    R: tl.constexpr,
    ZC: tl.constexpr,
):
    HD: tl.constexpr = 64
    H2: tl.constexpr = 32
    GR: tl.constexpr = NQ + 2
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    c2 = tl.max_contiguous(tl.multiple_of(tl.arange(0, H2), H2), H2)[None, :]
    rl = tl.arange(0, R)
    gqlo = tl.load(gq_ptr + c2).to(tl.float32)
    gqhi = tl.load(gq_ptr + H2 + c2).to(tl.float32)
    gklo = tl.load(gk_ptr + c2).to(tl.float32)
    gkhi = tl.load(gk_ptr + H2 + c2).to(tl.float32)
    if ZC:
        gqlo += 1.0
        gqhi += 1.0
        gklo += 1.0
        gkhi += 1.0
    aqlo = tl.zeros([R, H2], dtype=tl.float32)
    aqhi = tl.zeros([R, H2], dtype=tl.float32)
    aklo = tl.zeros([R, H2], dtype=tl.float32)
    akhi = tl.zeros([R, H2], dtype=tl.float32)
    for blk in range(pid, NBLK, nprog):
        row = blk * R + rl
        rm = row < NROWS
        t = row // (NG * GR)
        g = (row // GR) % NG
        rr = row % GR
        s_id = t // B
        b_id = t % B
        is_q = rr < NQ
        is_k = rr == NQ
        is_v = rr == NQ + 1
        mrow = rm[:, None]
        xrow = x_ptr + row.to(tl.int64) * HD
        flo = tl.load(xrow[:, None] + c2, mask=mrow, other=0.0).to(tl.float32)
        fhi = tl.load(xrow[:, None] + H2 + c2, mask=mrow, other=0.0).to(tl.float32)
        ss = tl.sum(flo * flo, axis=1) + tl.sum(fhi * fhi, axis=1)
        rs = tl.where(is_v, 1.0, tl.rsqrt(ss * (1.0 / HD) + eps))[:, None]
        xhlo = flo * rs
        xhhi = fhi * rs
        glo = tl.where(is_q[:, None], gqlo, tl.where(is_k[:, None], gklo, 1.0))
        ghi = tl.where(is_q[:, None], gqhi, tl.where(is_k[:, None], gkhi, 1.0))
        to = (tl.where(is_q, 0, 2 * S * H2) + s_id * H2)[:, None] + c2
        mrot = (rm & (is_q | is_k))[:, None]
        cs = tl.load(tab_ptr + to, mask=mrot, other=1.0)
        sn = tl.load(tab_ptr + S * H2 + to, mask=mrot, other=0.0)
        mq = (rm & is_q)[:, None]
        mk = (rm & is_k)[:, None]
        mv = (rm & is_v)[:, None]
        s64 = s_id.to(tl.int64)
        b64 = b_id.to(tl.int64)
        dqrow = dq_ptr + s64 * qs0 + b64 * qs1 + (g * NQ + rr).to(tl.int64) * qs2
        dkrow = dk_ptr + s64 * ks0 + b64 * ks1 + g.to(tl.int64) * ks2
        dvrow = dv_ptr + s64 * vs0 + b64 * vs1 + g.to(tl.int64) * vs2
        dlo = (
            tl.load(dqrow[:, None] + c2, mask=mq, other=0.0).to(tl.float32)
            + tl.load(dkrow[:, None] + c2, mask=mk, other=0.0).to(tl.float32)
            + tl.load(dvrow[:, None] + c2, mask=mv, other=0.0).to(tl.float32)
        )
        dhi = (
            tl.load(dqrow[:, None] + H2 + c2, mask=mq, other=0.0).to(tl.float32)
            + tl.load(dkrow[:, None] + H2 + c2, mask=mk, other=0.0).to(tl.float32)
            + tl.load(dvrow[:, None] + H2 + c2, mask=mv, other=0.0).to(tl.float32)
        )
        dylo = dlo * cs + dhi * sn
        dyhi = dhi * cs - dlo * sn
        aqlo += tl.where(mq, dylo * xhlo, 0.0)
        aqhi += tl.where(mq, dyhi * xhhi, 0.0)
        aklo += tl.where(mk, dylo * xhlo, 0.0)
        akhi += tl.where(mk, dyhi * xhhi, 0.0)
        dhlo = dylo * glo
        dhhi = dyhi * ghi
        m = ((tl.sum(dhlo * xhlo, axis=1) + tl.sum(dhhi * xhhi, axis=1)) * (1.0 / HD))[:, None]
        dxlo = tl.where(is_v[:, None], dlo, rs * (dhlo - m * xhlo))
        dxhi = tl.where(is_v[:, None], dhi, rs * (dhhi - m * xhhi))
        dxrow = dx_ptr + row.to(tl.int64) * HD
        tl.store(dxrow[:, None] + c2, dxlo.to(tl.bfloat16), mask=mrow)
        tl.store(dxrow[:, None] + H2 + c2, dxhi.to(tl.bfloat16), mask=mrow)
    cc = tl.arange(0, H2)
    pb = part_ptr + pid.to(tl.int64) * 128
    tl.store(pb + cc, tl.sum(aqlo, axis=0))
    tl.store(pb + H2 + cc, tl.sum(aqhi, axis=0))
    tl.store(pb + 2 * H2 + cc, tl.sum(aklo, axis=0))
    tl.store(pb + 3 * H2 + cc, tl.sum(akhi, axis=0))


@triton.jit(do_not_specialize=["P", "RPP"])
def _qknorm_colsum_kernel(part_ptr, out_ptr, P, RPP, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    c = tl.arange(0, 128)[None, :]
    acc = tl.zeros([128], dtype=tl.float32)
    r_begin = pid * RPP
    r_end = tl.minimum(r_begin + RPP, P)
    for r0 in range(r_begin, r_end, BLOCK_R):
        rr = (r0 + tl.arange(0, BLOCK_R))[:, None]
        m = rr < r_end
        acc += tl.sum(tl.load(part_ptr + rr.to(tl.int64) * 128 + c, mask=m, other=0.0), axis=0)
    tl.store(out_ptr + pid * 128 + tl.arange(0, 128), acc)


_SMS = {}


def _sm_count(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]


def _sincos(q_freqs, k_freqs, s):
    tab = torch.empty((4, s, H2), dtype=torch.float32, device=q_freqs.device)
    _qknorm_sincos_kernel[(triton.cdiv(s, 64),)](q_freqs, k_freqs, tab, s, BS=64, num_warps=4)
    return tab


def _check_inputs(x, q_freqs, k_freqs, gq, gk):
    s, b, ng, gsz = x.shape
    nq = gsz // HD - 2
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and gsz == (nq + 2) * HD and nq >= 1
    assert ng & (ng - 1) == 0 and nq & (nq - 1) == 0 and x.numel() < 2**31
    for f in (q_freqs, k_freqs):
        assert f.dtype == torch.float32 and f.is_contiguous() and f.shape[-1] == HD and f.numel() >= s * HD
    assert gq.numel() == HD and gk.numel() == HD and gq.is_contiguous() and gk.is_contiguous()
    return s, b, ng, nq


def _check_grad(g, shape):
    assert g.shape == shape and g.dtype == torch.bfloat16 and g.stride(3) == 1
    assert sum((n - 1) * st for n, st in zip(shape, g.stride())) + HD < 2**31


@torch.library.custom_op("optimized::qknorm_rope_fwd", mutates_args=(), device_types="cuda")
def qknorm_rope_fwd(
    x: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    gq: torch.Tensor,
    gk: torch.Tensor,
    eps: float,
    zero_centered: bool,
    rows: int,
    num_warps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s, b, ng, nq = _check_inputs(x, q_freqs, k_freqs, gq, gk)
    q = x.new_empty((s, b, ng * nq, HD))
    k = x.new_empty((s, b, ng, HD))
    v = x.new_empty((s, b, ng, HD))
    tab = _sincos(q_freqs, k_freqs, s)
    nrows = s * b * ng * (nq + 2)
    _qknorm_rope_fwd_kernel[(triton.cdiv(nrows, rows),)](
        x, gq, gk, tab, q, k, v, b, s, nrows, eps, NG=ng, NQ=nq, R=rows, ZC=zero_centered, num_warps=num_warps
    )
    return q, k, v


@qknorm_rope_fwd.register_fake
def _(x, q_freqs, k_freqs, gq, gk, eps, zero_centered, rows, num_warps):
    s, b, ng, gsz = x.shape
    nq = gsz // HD - 2
    return x.new_empty((s, b, ng * nq, HD)), x.new_empty((s, b, ng, HD)), x.new_empty((s, b, ng, HD))


@torch.library.custom_op("optimized::qknorm_rope_bwd", mutates_args=(), device_types="cuda")
def qknorm_rope_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    x: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    gq: torch.Tensor,
    gk: torch.Tensor,
    eps: float,
    zero_centered: bool,
    rows: int,
    num_warps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s, b, ng, nq = _check_inputs(x, q_freqs, k_freqs, gq, gk)
    _check_grad(dq, (s, b, ng * nq, HD))
    _check_grad(dk, (s, b, ng, HD))
    _check_grad(dv, (s, b, ng, HD))
    dx = torch.empty_like(x)
    tab = _sincos(q_freqs, k_freqs, s)
    nrows = s * b * ng * (nq + 2)
    nblk = triton.cdiv(nrows, rows)
    nprog = min(nblk, _sm_count(x.device) * _BWD_CTAS)
    part = torch.empty((nprog, 128), dtype=torch.float32, device=x.device)
    _qknorm_rope_bwd_kernel[(nprog,)](
        dq,
        dk,
        dv,
        x,
        gq,
        gk,
        tab,
        dx,
        part,
        b,
        s,
        nrows,
        nblk,
        eps,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        NG=ng,
        NQ=nq,
        R=rows,
        ZC=zero_centered,
        num_warps=num_warps,
    )
    return dx, part


@qknorm_rope_bwd.register_fake
def _(dq, dk, dv, x, q_freqs, k_freqs, gq, gk, eps, zero_centered, rows, num_warps):
    s, b, ng, gsz = x.shape
    nprog = min(triton.cdiv(s * b * gsz // HD, rows), _sm_count(x.device) * _BWD_CTAS)
    return torch.empty_like(x), x.new_empty((nprog, 128), dtype=torch.float32)


@torch.library.custom_op("optimized::qknorm_colsum", mutates_args=(), device_types="cuda")
def qknorm_colsum(part: torch.Tensor) -> torch.Tensor:
    assert part.dtype == torch.float32 and part.is_contiguous() and part.shape[1] == 128
    p = part.shape[0]
    g = max(1, min(128, triton.cdiv(p, 64)))
    mid = torch.empty((g, 128), dtype=torch.float32, device=part.device)
    _qknorm_colsum_kernel[(g,)](part, mid, p, triton.cdiv(p, g), BLOCK_R=64, num_warps=4)
    out = torch.empty(128, dtype=torch.float32, device=part.device)
    _qknorm_colsum_kernel[(1,)](mid, out, g, g, BLOCK_R=128, num_warps=4)
    return out


@qknorm_colsum.register_fake
def _(part):
    return part.new_empty(128)


def _dense(g, like):
    if g is None:
        return torch.zeros(like, dtype=torch.bfloat16, device="cuda")
    return g if g.stride(-1) == 1 else g.contiguous()


class QKNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_freqs, k_freqs, gq, gk, eps, zero_centered):
        q, k, v = torch.ops.optimized.qknorm_rope_fwd(
            x, q_freqs, k_freqs, gq, gk, float(eps), bool(zero_centered), _FWD_ROWS, _FWD_WARPS
        )
        ctx.save_for_backward(x, q_freqs, k_freqs, gq, gk)
        ctx.eps = float(eps)
        ctx.zc = bool(zero_centered)
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        x, q_freqs, k_freqs, gq, gk = ctx.saved_tensors
        s, b, ng, gsz = x.shape
        nq = gsz // HD - 2
        dq = _dense(dq, (s, b, ng * nq, HD))
        dk = _dense(dk, (s, b, ng, HD))
        dv = _dense(dv, (s, b, ng, HD))
        dx, part = torch.ops.optimized.qknorm_rope_bwd(
            dq, dk, dv, x, q_freqs, k_freqs, gq, gk, ctx.eps, ctx.zc, _BWD_ROWS, _BWD_WARPS
        )
        dg = torch.ops.optimized.qknorm_colsum(part)
        return dx, None, None, dg[:HD].to(gq.dtype), dg[HD:].to(gk.dtype), None, None


def qknorm_rope(x, q_freqs, k_freqs, gq, gk, eps, zero_centered=False):
    return QKNormRoPE.apply(x, q_freqs, k_freqs, gq, gk, eps, zero_centered)


def _rope(t, freqs, s):
    f = freqs.float().reshape(-1, HD)[:s]
    c = torch.cos(f)[:, None, None, :]
    sn = torch.sin(f)[:, None, None, :]
    rot = torch.cat((-t[..., H2:], t[..., :H2]), dim=-1)
    return t * c + rot * sn


def _norm(t, g, eps, zero_centered):
    rs = torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps)
    gg = g.float() + (1.0 if zero_centered else 0.0)
    return t * rs * gg


def reference_forward(x, q_freqs, k_freqs, gq, gk, eps, zero_centered=False, out_dtype=torch.bfloat16):
    s, b, ng, gsz = x.shape
    nq = gsz // HD - 2
    xf = x.float()
    q = xf[..., : nq * HD].reshape(s, b, ng * nq, HD)
    k = xf[..., nq * HD : (nq + 1) * HD]
    v = x[..., (nq + 1) * HD :]
    q = _rope(_norm(q, gq, eps, zero_centered), q_freqs, s)
    k = _rope(_norm(k, gk, eps, zero_centered), k_freqs, s)
    return q.to(out_dtype), k.to(out_dtype), v.contiguous()


def reference_backward(x, q_freqs, k_freqs, gq, gk, eps, zero_centered, dq, dk, dv):
    xf = x.detach().float().requires_grad_(True)
    gqf = gq.detach().float().requires_grad_(True)
    gkf = gk.detach().float().requires_grad_(True)
    s, b, ng, gsz = x.shape
    nq = gsz // HD - 2
    q = xf[..., : nq * HD].reshape(s, b, ng * nq, HD)
    k = xf[..., nq * HD : (nq + 1) * HD]
    v = xf[..., (nq + 1) * HD :]
    q = _rope(_norm(q, gqf, eps, zero_centered), q_freqs, s)
    k = _rope(_norm(k, gkf, eps, zero_centered), k_freqs, s)
    dx, dgq, dgk = torch.autograd.grad((q, k, v), (xf, gqf, gkf), (dq.float(), dk.float(), dv.float()))
    return dx, dgq, dgk


def make_freqs(s, offset=0, base=150000.0, device="cuda"):
    inv = 1.0 / (base ** (torch.arange(0, HD, 2, dtype=torch.float32, device=device) / HD))
    pos = torch.arange(offset, offset + s, dtype=torch.float32, device=device)
    f = torch.outer(pos, inv)
    return torch.cat((f, f), dim=-1).reshape(s, 1, 1, HD).contiguous()


SHAPES = [(8192, 2, 8, 8), (8192, 3, 8, 8), (4096, 1, 8, 8), (2048, 1, 8, 8), (2048, 1, 4, 8)]


def get_inputs(shape=(8192, 3, 8, 8), eps=1e-5, zero_centered=False, seed=0, device="cuda"):
    s, b, ng, nq = shape
    gen = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((s, b, ng, (nq + 2) * HD), generator=gen, device=device) * 0.5).to(torch.bfloat16)
    gq = (torch.ones(HD, device=device) + 0.1 * torch.randn(HD, generator=gen, device=device)).to(
        torch.bfloat16
    )
    gk = (torch.ones(HD, device=device) + 0.1 * torch.randn(HD, generator=gen, device=device)).to(
        torch.bfloat16
    )
    if zero_centered:
        gq = gq - 1.0
        gk = gk - 1.0
    return x, make_freqs(s, device=device), make_freqs(s, offset=s, device=device), gq, gk, eps, zero_centered


def kernel_fn(x, q_freqs, k_freqs, gq, gk, eps, zero_centered):
    return qknorm_rope(x, q_freqs, k_freqs, gq, gk, eps, zero_centered)


def reference_fn(x, q_freqs, k_freqs, gq, gk, eps, zero_centered):
    return reference_forward(x, q_freqs, k_freqs, gq, gk, eps, zero_centered)


def _rel(a, b):
    d = (a.float() - b.float()).abs()
    mx = (d.max() / b.float().abs().max().clamp_min(1e-30)).item()
    mean = (d.mean() / b.float().abs().mean().clamp_min(1e-30)).item()
    return mx, mean


def _strided_like(g):
    buf = torch.empty(g.shape[:-1] + (2 * HD,), dtype=g.dtype, device=g.device)
    out = buf[..., :HD]
    out.copy_(g)
    return out


def _check(shape, eps, zero_centered, strided, bwd):
    inputs = get_inputs(shape, eps, zero_centered)
    x, q_freqs, k_freqs, gq, gk, _, _ = inputs
    q, k, v = kernel_fn(*inputs)
    rq, rk, rv = reference_fn(*inputs)
    res = {"q": _rel(q, rq), "k": _rel(k, rk), "v_equal": bool(torch.equal(v, rv))}
    if bwd:
        s, b, ng, nq = shape
        gen = torch.Generator(device="cuda").manual_seed(1)
        dq = torch.randn(q.shape, generator=gen, device="cuda").to(torch.bfloat16)
        dk = torch.randn(k.shape, generator=gen, device="cuda").to(torch.bfloat16)
        dv = torch.randn(v.shape, generator=gen, device="cuda").to(torch.bfloat16)
        if strided:
            dq, dk, dv = _strided_like(dq), _strided_like(dk), _strided_like(dv)
        dx, part = torch.ops.optimized.qknorm_rope_bwd(
            dq, dk, dv, x, q_freqs, k_freqs, gq, gk, eps, zero_centered, _BWD_ROWS, _BWD_WARPS
        )
        dg = torch.ops.optimized.qknorm_colsum(part)
        rdx, rdgq, rdgk = reference_backward(x, q_freqs, k_freqs, gq, gk, eps, zero_centered, dq, dk, dv)
        res["dx"] = _rel(dx, rdx)
        res["dgq"] = _rel(dg[:HD], rdgq)
        res["dgk"] = _rel(dg[HD:], rdgk)
        res["dv_equal"] = bool(torch.equal(dx[..., (nq + 1) * HD :], dv))
    return res


def _run_graph(shape):
    inputs = get_inputs(shape)
    x, q_freqs, k_freqs, gq, gk, eps, zc = inputs
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        xg = x.clone().requires_grad_(True)
        gqg = gq.clone().requires_grad_(True)
        gkg = gk.clone().requires_grad_(True)
        q, k, v = qknorm_rope(xg, q_freqs, k_freqs, gqg, gkg, eps, zc)
        dq, dk, dv = torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)
        torch.autograd.backward((q, k, v), (dq, dk, dv))
        ref = (q.clone(), k.clone(), v.clone(), xg.grad.clone(), gqg.grad.clone(), gkg.grad.clone())
        for _ in range(3):
            xg.grad = None
            gqg.grad = None
            gkg.grad = None
            q, k, v = qknorm_rope(xg, q_freqs, k_freqs, gqg, gkg, eps, zc)
            torch.autograd.backward((q, k, v), (dq, dk, dv))
        xg.grad = None
        gqg.grad = None
        gkg.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        q, k, v = qknorm_rope(xg, q_freqs, k_freqs, gqg, gkg, eps, zc)
        torch.autograd.backward((q, k, v), (dq, dk, dv))
    torch.cuda.current_stream().wait_stream(s)
    ok = True
    for _ in range(2):
        xg.grad.zero_()
        gqg.grad.zero_()
        gkg.grad.zero_()
        graph.replay()
        torch.cuda.synchronize()
        got = (q, k, v, xg.grad, gqg.grad, gkg.grad)
        ok = ok and all(torch.equal(p, r) for p, r in zip(got, ref))
    return ok


def _variants(fn):
    try:
        caches = fn.device_caches
        return sum(len(c[0]) for c in caches.values())
    except Exception:
        return -1


def _bench(shape, fwd_rows, fwd_warps, bwd_rows, bwd_warps):
    inputs = get_inputs(shape)
    x, q_freqs, k_freqs, gq, gk, eps, zc = inputs
    q, k, v = torch.ops.optimized.qknorm_rope_fwd(x, q_freqs, k_freqs, gq, gk, eps, zc, fwd_rows, fwd_warps)
    dq, dk, dv = torch.randn_like(q), torch.randn_like(k), torch.randn_like(v)
    f_us = (
        triton.testing.do_bench(
            lambda: torch.ops.optimized.qknorm_rope_fwd(
                x, q_freqs, k_freqs, gq, gk, eps, zc, fwd_rows, fwd_warps
            )
        )
        * 1e3
    )
    b_us = (
        triton.testing.do_bench(
            lambda: torch.ops.optimized.qknorm_rope_bwd(
                dq, dk, dv, x, q_freqs, k_freqs, gq, gk, eps, zc, bwd_rows, bwd_warps
            )
        )
        * 1e3
    )
    _, part = torch.ops.optimized.qknorm_rope_bwd(
        dq, dk, dv, x, q_freqs, k_freqs, gq, gk, eps, zc, bwd_rows, bwd_warps
    )
    c_us = triton.testing.do_bench(lambda: torch.ops.optimized.qknorm_colsum(part)) * 1e3
    nbytes = x.numel() * 2
    return f_us, 2 * nbytes / f_us / 1e6, b_us, 3 * nbytes / b_us / 1e6, c_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-shapes", action="store_true")
    ap.add_argument("--bwd", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    shapes = SHAPES if args.all_shapes else [SHAPES[1]]
    worst = 0.0
    for shape in shapes:
        for eps in (1e-5, 1e-6):
            for zc in (False, True):
                for strided in (False, True) if args.bwd else (False,):
                    res = _check(shape, eps, zc, strided, args.bwd)
                    worst = max(
                        worst,
                        res["q"][0],
                        res["k"][0],
                        *(res[key][0] for key in ("dx", "dgq", "dgk") if key in res),
                    )
                    print(f"shape={shape} eps={eps} zc={int(zc)} strided={int(strided)} {res}", flush=True)
    nf, nb = _variants(_qknorm_rope_fwd_kernel), _variants(_qknorm_rope_bwd_kernel)
    print(f"max rel diff {worst:.3e}; fwd variants {nf} bwd variants {nb}")
    if args.graph:
        print(f"graph replay matches eager: {_run_graph(SHAPES[1])}", flush=True)
    if args.bench:
        for shape in shapes:
            f_us, f_bw, b_us, b_bw, c_us = _bench(shape, _FWD_ROWS, _FWD_WARPS, _BWD_ROWS, _BWD_WARPS)
            print(
                f"bench shape={shape} fwd {f_us:.1f} us ({f_bw:.2f} TB/s) "
                f"bwd {b_us:.1f} us ({b_bw:.2f} TB/s) colsum {c_us:.1f} us",
                flush=True,
            )
    if args.sweep:
        for rows in (16, 32, 64):
            for w in (2, 4, 8):
                f_us, f_bw, b_us, b_bw, c_us = _bench(SHAPES[1], rows, w, rows, w)
                print(
                    f"sweep rows={rows} warps={w} fwd {f_us:.1f} us ({f_bw:.2f} TB/s) "
                    f"bwd {b_us:.1f} us ({b_bw:.2f} TB/s) colsum {c_us:.1f} us",
                    flush=True,
                )


if __name__ == "__main__":
    sys.exit(main())
