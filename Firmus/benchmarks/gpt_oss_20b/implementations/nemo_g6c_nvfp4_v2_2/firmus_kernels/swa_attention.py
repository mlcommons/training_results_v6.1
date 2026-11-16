"""Triton sliding-window attention with or without learnable softmax sinks (GPT-OSS SWA layers).

Copy of TE's ops/basic/nvfp4_swa_attention.py (transformer_engine_firmus_nvfp4.patch) with
HAS_SINK: when False (softmax_type="vanilla") the softmax has no sink column, there is no
sink gradient, and the online softmax starts from m = -inf, l = 0.

Semantics match TE DotProductAttention(softmax_type="learnable", window_size=(W, 0),
attn_mask_type="causal", qkv_format="sbhd"): query i attends keys [i - W, i]; the
per-head sink is an extra logit column (in scaled-logit units) that joins the
softmax and carries no value. GQA: head h reads KV group h // (H // G).

Backward is deterministic: dK/dV programs own a key block and loop over the
query heads of their group; the sink gradient is written as per-program partials.
"""

import os

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


def _cfg(name, default):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) from NVFP4_SWA_<name>=m,n,w,s."""
    return tuple(int(x) for x in os.environ.get(f"NVFP4_SWA_{name}", default).split(","))


CFG = {"FWD": _cfg("FWD", "64,32,4,3"), "DKDV": _cfg("DKDV", "64,64,4,2"), "DQ": _cfg("DQ", "64,32,4,2")}
# Packed kernels: (HP, BMQ, BLOCK_N, num_warps, num_stages) from NVFP4_SWA_P<name>.
PACKED = [os.environ.get("NVFP4_SWA_PACKED", "1") == "1"]
PCFG = {"FWD": _cfg("PFWD", "4,32,32,4,3"), "DKDV": _cfg("PDKDV", "2,32,64,4,2"), "DQ": _cfg("PDQ", "4,32,32,4,3")}


@triton.jit
def _swa_fwd(
    Q, K, V, SINK, O, LSE,
    sq_s, sq_b, sq_h, sk_s, sk_b, sk_h, sv_s, sv_b, sv_h, so_s, so_b, so_h,
    S, qk_scale,
    H: tl.constexpr, G: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, HAS_SINK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    g = h // (H // G)
    start = pid_m * BLOCK_M
    offs_m = start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < S

    q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    if HAS_SINK:
        sink2 = tl.load(SINK + h).to(tl.float32) * 1.4426950408889634
        m_i = tl.zeros([BLOCK_M], tl.float32) + sink2
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    else:
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    k_base = K + b * sk_b + g * sk_h
    v_base = V + b * sv_b + g * sv_h
    lo = (tl.maximum(start - WINDOW, 0) // BLOCK_N) * BLOCK_N
    hi = tl.minimum(start + BLOCK_M, S)
    for n0 in range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_ok = offs_n < S
        kt = tl.load(k_base + offs_n[None, :] * sk_s + offs_d[:, None], mask=col_ok[None, :], other=0.0)
        s = tl.dot(q, kt) * qk_scale
        valid = (
            (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] >= offs_m[:, None] - WINDOW)
            & col_ok[None, :]
        )
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        if HAS_SINK:
            m_use = m_new
        else:
            # A key block can be fully masked for a row before its first in-window key.
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + offs_n[:, None] * sv_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o = acc / l_i[:, None]
    tl.store(
        O + b * so_b + h * so_h + offs_m[:, None] * so_s + offs_d[None, :],
        o.to(O.dtype.element_ty),
        mask=row_ok[:, None],
    )
    tl.store(LSE + pid_bh * S + offs_m, m_i + tl.log2(l_i), mask=row_ok)


@triton.jit
def _swa_bwd_pre(
    O, DO, DELTA,
    so_s, so_b, so_h, sdo_s, sdo_b, sdo_h,
    S, H: tl.constexpr, BLOCK_M: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < S
    o = tl.load(O + b * so_b + h * so_h + offs_m[:, None] * so_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + b * sdo_b + h * sdo_h + offs_m[:, None] * sdo_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    tl.store(DELTA + pid_bh * S + offs_m, tl.sum(o.to(tl.float32) * do.to(tl.float32), 1), mask=row_ok)


@triton.jit
def _swa_bwd_dkdv(
    Q, K, V, DO, LSE, DELTA, DK, DV,
    sq_s, sq_b, sq_h, sk_s, sk_b, sk_h, sv_s, sv_b, sv_h, sdo_s, sdo_b, sdo_h,
    sdk_s, sdk_b, sdk_h,
    S, qk_scale, sm_scale,
    H: tl.constexpr, G: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = pid_bg // G
    g = pid_bg % G
    HPG: tl.constexpr = H // G
    ks = pid_n * BLOCK_N
    offs_n = ks + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    col_ok = offs_n < S

    k = tl.load(K + b * sk_b + g * sk_h + offs_n[:, None] * sk_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    v = tl.load(V + b * sv_b + g * sv_h + offs_n[:, None] * sv_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], tl.float32)
    dv = tl.zeros([BLOCK_N, D], tl.float32)

    q_lo = (ks // BLOCK_M) * BLOCK_M
    q_hi = tl.minimum(ks + BLOCK_N + WINDOW, S)
    for hh in range(HPG):
        h = g * HPG + hh
        bh = b * H + h
        for m0 in range(q_lo, q_hi, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            row_ok = offs_m < S
            q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
            do = tl.load(DO + b * sdo_b + h * sdo_h + offs_m[:, None] * sdo_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
            lse = tl.load(LSE + bh * S + offs_m, mask=row_ok, other=0.0)
            delta = tl.load(DELTA + bh * S + offs_m, mask=row_ok, other=0.0)
            s = tl.dot(q, tl.trans(k)) * qk_scale
            valid = (
                (offs_n[None, :] <= offs_m[:, None])
                & (offs_n[None, :] >= offs_m[:, None] - WINDOW)
                & col_ok[None, :]
                & row_ok[:, None]
            )
            p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    dk = dk * sm_scale
    base = b * sdk_b + g * sdk_h + offs_n[:, None] * sdk_s + offs_d[None, :]
    tl.store(DK + base, dk.to(DK.dtype.element_ty), mask=col_ok[:, None])
    tl.store(DV + base, dv.to(DV.dtype.element_ty), mask=col_ok[:, None])


@triton.jit
def _swa_bwd_dq(
    Q, K, V, DO, LSE, DELTA, SINK, DQ, DSINK,
    sq_s, sq_b, sq_h, sk_s, sk_b, sk_h, sv_s, sv_b, sv_h, sdo_s, sdo_b, sdo_h,
    sdq_s, sdq_b, sdq_h,
    S, qk_scale, sm_scale, n_mblocks,
    H: tl.constexpr, G: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, HAS_SINK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    g = h // (H // G)
    start = pid_m * BLOCK_M
    offs_m = start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < S

    q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + b * sdo_b + h * sdo_h + offs_m[:, None] * sdo_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    lse = tl.load(LSE + pid_bh * S + offs_m, mask=row_ok, other=0.0)
    delta = tl.load(DELTA + pid_bh * S + offs_m, mask=row_ok, other=0.0)
    dq = tl.zeros([BLOCK_M, D], tl.float32)

    k_base = K + b * sk_b + g * sk_h
    v_base = V + b * sv_b + g * sv_h
    lo = (tl.maximum(start - WINDOW, 0) // BLOCK_N) * BLOCK_N
    hi = tl.minimum(start + BLOCK_M, S)
    for n0 in range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_ok = offs_n < S
        k = tl.load(k_base + offs_n[:, None] * sk_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        v = tl.load(v_base + offs_n[:, None] * sv_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        valid = (
            (offs_n[None, :] <= offs_m[:, None])
            & (offs_n[None, :] >= offs_m[:, None] - WINDOW)
            & col_ok[None, :]
            & row_ok[:, None]
        )
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    dq = dq * sm_scale
    tl.store(
        DQ + b * sdq_b + h * sdq_h + offs_m[:, None] * sdq_s + offs_d[None, :],
        dq.to(DQ.dtype.element_ty),
        mask=row_ok[:, None],
    )
    if HAS_SINK:
        sink2 = tl.load(SINK + h).to(tl.float32) * 1.4426950408889634
        p_sink = tl.where(row_ok, tl.exp2(sink2 - lse), 0.0)
        tl.store(DSINK + pid_bh * n_mblocks + pid_m, -tl.sum(p_sink * delta, 0))


# Packed variants: a tile holds HP query heads of one KV group x BMQ query rows, so
# the MMA M dimension is HP * BMQ without widening the key band of the window.


@triton.jit
def _swa_fwd_packed(
    Q, K, V, SINK, O, LSE,
    sq_s, sq_b, sq_h, sk_s, sk_b, sk_h, sv_s, sv_b, sv_h, so_s, so_b, so_h,
    S, qk_scale,
    H: tl.constexpr, G: tl.constexpr, WINDOW: tl.constexpr,
    HP: tl.constexpr, BMQ: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, HAS_SINK: tl.constexpr,
):
    NP: tl.constexpr = (H // G) // HP
    ROWS: tl.constexpr = HP * BMQ
    pid_m = tl.program_id(0)
    pid = tl.program_id(1)
    b = pid // (G * NP)
    g = (pid % (G * NP)) // NP
    h0 = g * (H // G) + (pid % NP) * HP
    start = pid_m * BMQ
    r = tl.arange(0, ROWS)
    hr = h0 + r // BMQ
    qr = start + r % BMQ
    offs_d = tl.arange(0, D)
    row_ok = qr < S

    q = tl.load(Q + b * sq_b + hr[:, None] * sq_h + qr[:, None] * sq_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    if HAS_SINK:
        m_i = tl.load(SINK + hr).to(tl.float32) * 1.4426950408889634
        l_i = tl.full([ROWS], 1.0, tl.float32)
    else:
        m_i = tl.full([ROWS], float("-inf"), tl.float32)
        l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D], tl.float32)

    k_base = K + b * sk_b + g * sk_h
    v_base = V + b * sv_b + g * sv_h
    lo = (tl.maximum(start - WINDOW, 0) // BLOCK_N) * BLOCK_N
    hi = tl.minimum(start + BMQ, S)
    for n0 in range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_ok = offs_n < S
        kt = tl.load(k_base + offs_n[None, :] * sk_s + offs_d[:, None], mask=col_ok[None, :], other=0.0)
        s = tl.dot(q, kt) * qk_scale
        valid = (offs_n[None, :] <= qr[:, None]) & (offs_n[None, :] >= qr[:, None] - WINDOW) & col_ok[None, :]
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        if HAS_SINK:
            m_use = m_new
        else:
            # A key block can be fully masked for a row before its first in-window key.
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + offs_n[:, None] * sv_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o = acc / l_i[:, None]
    tl.store(
        O + b * so_b + hr[:, None] * so_h + qr[:, None] * so_s + offs_d[None, :],
        o.to(O.dtype.element_ty),
        mask=row_ok[:, None],
    )
    tl.store(LSE + (b * H + hr) * S + qr, m_i + tl.log2(l_i), mask=row_ok)


@triton.jit
def _swa_bwd_dkdv_packed(
    Q, K, V, DO, LSE, DELTA, DK, DV,
    sq_s, sq_b, sq_h, sk_s, sk_b, sk_h, sv_s, sv_b, sv_h, sdo_s, sdo_b, sdo_h,
    sdk_s, sdk_b, sdk_h,
    S, qk_scale, sm_scale,
    H: tl.constexpr, G: tl.constexpr, WINDOW: tl.constexpr,
    HP: tl.constexpr, BMQ: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    HPG: tl.constexpr = H // G
    NP: tl.constexpr = HPG // HP
    ROWS: tl.constexpr = HP * BMQ
    pid_n = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = pid_bg // G
    g = pid_bg % G
    ks = pid_n * BLOCK_N
    offs_n = ks + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    col_ok = offs_n < S
    r = tl.arange(0, ROWS)

    k = tl.load(K + b * sk_b + g * sk_h + offs_n[:, None] * sk_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    v = tl.load(V + b * sv_b + g * sv_h + offs_n[:, None] * sv_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], tl.float32)
    dv = tl.zeros([BLOCK_N, D], tl.float32)

    q_lo = (ks // BMQ) * BMQ
    q_hi = tl.minimum(ks + BLOCK_N + WINDOW, S)
    for hp in range(NP):
        hr = g * HPG + hp * HP + r // BMQ
        for m0 in range(q_lo, q_hi, BMQ):
            qr = m0 + r % BMQ
            row_ok = qr < S
            q = tl.load(Q + b * sq_b + hr[:, None] * sq_h + qr[:, None] * sq_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
            do = tl.load(DO + b * sdo_b + hr[:, None] * sdo_h + qr[:, None] * sdo_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
            lse = tl.load(LSE + (b * H + hr) * S + qr, mask=row_ok, other=0.0)
            delta = tl.load(DELTA + (b * H + hr) * S + qr, mask=row_ok, other=0.0)
            s = tl.dot(q, tl.trans(k)) * qk_scale
            valid = (
                (offs_n[None, :] <= qr[:, None])
                & (offs_n[None, :] >= qr[:, None] - WINDOW)
                & col_ok[None, :]
                & row_ok[:, None]
            )
            p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    dk = dk * sm_scale
    base = b * sdk_b + g * sdk_h + offs_n[:, None] * sdk_s + offs_d[None, :]
    tl.store(DK + base, dk.to(DK.dtype.element_ty), mask=col_ok[:, None])
    tl.store(DV + base, dv.to(DV.dtype.element_ty), mask=col_ok[:, None])


@triton.jit
def _swa_bwd_dq_packed(
    Q, K, V, DO, LSE, DELTA, SINK, DQ, DSINK,
    sq_s, sq_b, sq_h, sk_s, sk_b, sk_h, sv_s, sv_b, sv_h, sdo_s, sdo_b, sdo_h,
    sdq_s, sdq_b, sdq_h,
    S, qk_scale, sm_scale, n_mblocks,
    H: tl.constexpr, G: tl.constexpr, WINDOW: tl.constexpr,
    HP: tl.constexpr, BMQ: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, HAS_SINK: tl.constexpr,
):
    NP: tl.constexpr = (H // G) // HP
    ROWS: tl.constexpr = HP * BMQ
    pid_m = tl.program_id(0)
    pid = tl.program_id(1)
    b = pid // (G * NP)
    g = (pid % (G * NP)) // NP
    h0 = g * (H // G) + (pid % NP) * HP
    start = pid_m * BMQ
    r = tl.arange(0, ROWS)
    hr = h0 + r // BMQ
    qr = start + r % BMQ
    offs_d = tl.arange(0, D)
    row_ok = qr < S

    q = tl.load(Q + b * sq_b + hr[:, None] * sq_h + qr[:, None] * sq_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + b * sdo_b + hr[:, None] * sdo_h + qr[:, None] * sdo_s + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    lse = tl.load(LSE + (b * H + hr) * S + qr, mask=row_ok, other=0.0)
    delta = tl.load(DELTA + (b * H + hr) * S + qr, mask=row_ok, other=0.0)
    dq = tl.zeros([ROWS, D], tl.float32)

    k_base = K + b * sk_b + g * sk_h
    v_base = V + b * sv_b + g * sv_h
    lo = (tl.maximum(start - WINDOW, 0) // BLOCK_N) * BLOCK_N
    hi = tl.minimum(start + BMQ, S)
    for n0 in range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_ok = offs_n < S
        k = tl.load(k_base + offs_n[:, None] * sk_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        v = tl.load(v_base + offs_n[:, None] * sv_s + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        valid = (
            (offs_n[None, :] <= qr[:, None])
            & (offs_n[None, :] >= qr[:, None] - WINDOW)
            & col_ok[None, :]
            & row_ok[:, None]
        )
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)

    dq = dq * sm_scale
    tl.store(
        DQ + b * sdq_b + hr[:, None] * sdq_h + qr[:, None] * sdq_s + offs_d[None, :],
        dq.to(DQ.dtype.element_ty),
        mask=row_ok[:, None],
    )
    if HAS_SINK:
        sink2 = tl.load(SINK + hr).to(tl.float32) * 1.4426950408889634
        contrib = tl.where(row_ok, tl.exp2(sink2 - lse), 0.0) * delta
        part = -tl.sum(tl.reshape(contrib, [HP, BMQ]), 1)
        heads = h0 + tl.arange(0, HP)
        tl.store(DSINK + (b * H + heads) * n_mblocks + pid_m, part)


def _check(q, k, v):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("expected sbhd tensors")
    for t in (q, k, v):
        if t.stride(-1) != 1:
            raise ValueError("last dim must be contiguous")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[3] != k.shape[3]:
        raise ValueError(f"shape mismatch q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.shape[2] % k.shape[2]:
        raise ValueError("query heads must be a multiple of kv heads")


class SWAAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sink, sm_scale, window):
        _check(q, k, v)
        S, B, H, D = q.shape
        G = k.shape[2]
        has_sink = sink is not None
        sink_flat = sink.reshape(-1) if has_sink else q
        o = torch.empty((S, B, H, D), device=q.device, dtype=q.dtype)
        lse = torch.empty((B * H, S), device=q.device, dtype=torch.float32)
        qk_scale = sm_scale * _LOG2E
        args = (
            q, k, v, sink_flat, o, lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            S, qk_scale,
        )
        if PACKED[0]:
            hp, bmq, bn, w, st = PCFG["FWD"]
            _swa_fwd_packed[(triton.cdiv(S, bmq), B * H // hp)](
                *args, H=H, G=G, WINDOW=window, HP=hp, BMQ=bmq, BLOCK_N=bn, D=D, HAS_SINK=has_sink,
                num_warps=w, num_stages=st,
            )
        else:
            bm, bn, w, st = CFG["FWD"]
            _swa_fwd[(triton.cdiv(S, bm), B * H)](
                *args, H=H, G=G, WINDOW=window, BLOCK_M=bm, BLOCK_N=bn, D=D, HAS_SINK=has_sink,
                num_warps=w, num_stages=st,
            )
        ctx.save_for_backward(q, k, v, sink, o, lse)
        ctx.has_sink = has_sink
        ctx.sm_scale = sm_scale
        ctx.window = window
        return o.view(S, B, H * D)

    @staticmethod
    def backward(ctx, dout):
        q, k, v, sink, o, lse = ctx.saved_tensors
        S, B, H, D = q.shape
        G = k.shape[2]
        do = dout.reshape(S, B, H, D)
        if do.stride(-1) != 1:
            do = do.contiguous()
        has_sink = ctx.has_sink
        sink_flat = sink.reshape(-1) if has_sink else q
        qk_scale = ctx.sm_scale * _LOG2E
        delta = torch.empty((B * H, S), device=q.device, dtype=torch.float32)
        _swa_bwd_pre[(triton.cdiv(S, 64), B * H)](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            S, H=H, BLOCK_M=64, D=D, num_warps=4,
        )
        packed = PACKED[0]
        dq_bm = PCFG["DQ"][1] if packed else CFG["DQ"][0]
        n_m = triton.cdiv(S, dq_bm)
        dq = torch.empty((S, B, H, D), device=q.device, dtype=q.dtype)
        dk = torch.empty((S, B, G, D), device=q.device, dtype=k.dtype)
        dv = torch.empty((S, B, G, D), device=q.device, dtype=v.dtype)
        dsink_part = torch.empty((B, H, n_m) if has_sink else (1,), device=q.device, dtype=torch.float32)
        strides = (
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
        )
        kv_args = (q, k, v, do, lse, delta, dk, dv, *strides,
                   dk.stride(0), dk.stride(1), dk.stride(2), S, qk_scale, ctx.sm_scale)
        dq_args = (q, k, v, do, lse, delta, sink_flat, dq, dsink_part, *strides,
                   dq.stride(0), dq.stride(1), dq.stride(2), S, qk_scale, ctx.sm_scale, n_m)
        if packed:
            hp, bmq, bn, w, st = PCFG["DKDV"]
            _swa_bwd_dkdv_packed[(triton.cdiv(S, bn), B * G)](
                *kv_args, H=H, G=G, WINDOW=ctx.window, HP=hp, BMQ=bmq, BLOCK_N=bn, D=D,
                num_warps=w, num_stages=st,
            )
            hp, bmq, bn, w, st = PCFG["DQ"]
            _swa_bwd_dq_packed[(n_m, B * H // hp)](
                *dq_args, H=H, G=G, WINDOW=ctx.window, HP=hp, BMQ=bmq, BLOCK_N=bn, D=D, HAS_SINK=has_sink,
                num_warps=w, num_stages=st,
            )
        else:
            bm, bn, w, st = CFG["DKDV"]
            _swa_bwd_dkdv[(triton.cdiv(S, bn), B * G)](
                *kv_args, H=H, G=G, WINDOW=ctx.window, BLOCK_M=bm, BLOCK_N=bn, D=D,
                num_warps=w, num_stages=st,
            )
            bm, bn, w, st = CFG["DQ"]
            _swa_bwd_dq[(n_m, B * H)](
                *dq_args, H=H, G=G, WINDOW=ctx.window, BLOCK_M=bm, BLOCK_N=bn, D=D, HAS_SINK=has_sink,
                num_warps=w, num_stages=st,
            )
        dsink = dsink_part.sum(dim=(0, 2)).view(sink.shape).to(sink.dtype) if has_sink else None
        return dq, dk, dv, dsink, None, None


def swa_attention(q, k, v, sink, sm_scale, window):
    """q [S, B, H, D], k/v [S, B, G, D], sink with H elements or None; returns [S, B, H * D]."""
    return SWAAttentionFunc.apply(q, k, v, sink, sm_scale, window)


def install():
    """Route eligible Megatron TEDotProductAttention SWA calls to the Triton kernel."""
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType

    orig_forward = TEDotProductAttention.forward
    logged, logged_fp8 = [False], [False]
    full_fp8 = None
    if os.environ.get("NVFP4_FULL_ATTN_FP8", "0") == "1":
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe as te_recipe

        # Local amax only: a world all-reduce per full layer would sit on the critical path.
        full_fp8 = te_recipe.DelayedScaling(
            fp8_format=te_recipe.Format.HYBRID, fp8_dpa=True, fp8_mha=False, reduce_amax=False,
            amax_history_len=int(os.environ.get("NVFP4_FULL_ATTN_FP8_HISTORY", "16")),
        )

    def forward(self, query, key, value, attention_mask, attn_mask_type, attention_bias=None,
                packed_seq_params=None, num_splits=None, bf16_backward=None):
        window = getattr(self, "window_size", None)
        if (
            window is not None
            and tuple(window)[1] == 0
            and tuple(window)[0] > 0
            and getattr(self, "softmax_type", "vanilla") in ("learnable", "vanilla")
            and attn_mask_type == AttnMaskType.causal
            and attention_bias is None
            and packed_seq_params is None
            and self.qkv_format == "sbhd"
            and query.dtype == torch.bfloat16
            and query.shape[-1] in (64, 128)
        ):
            if not logged[0] and int(os.environ.get("RANK", "0")) == 0:
                logged[0] = True
                print(
                    f"NVFP4 SWA Triton attention window={tuple(window)} q={tuple(query.shape)} "
                    f"k={tuple(key.shape)} packed={PACKED[0]} cfg={PCFG if PACKED[0] else CFG}",
                    flush=True,
                )
            sink = self.softmax_offset if self.softmax_type == "learnable" else None
            return swa_attention(
                query, key, value, sink, self.fused_attention.softmax_scale, int(window[0])
            )
        if full_fp8 is not None and (window is None or tuple(window)[0] < 0):
            if not logged_fp8[0] and int(os.environ.get("RANK", "0")) == 0:
                logged_fp8[0] = True
                print(f"NVFP4 full attention FP8 DPA recipe={full_fp8} q={tuple(query.shape)}", flush=True)
            with te.fp8_autocast(enabled=True, fp8_recipe=full_fp8):
                return orig_forward(self, query, key, value, attention_mask, attn_mask_type, attention_bias,
                                    packed_seq_params, num_splits, bf16_backward)
        return orig_forward(self, query, key, value, attention_mask, attn_mask_type, attention_bias,
                            packed_seq_params, num_splits, bf16_backward)

    TEDotProductAttention.forward = forward
