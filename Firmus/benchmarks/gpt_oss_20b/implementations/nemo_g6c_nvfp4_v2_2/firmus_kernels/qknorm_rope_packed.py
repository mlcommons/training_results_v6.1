# Classification: Internal
"""NVIDIA's fused QK-norm + RoPE (opt_kernels/qknorm_rope.py) with Q, K, V packed and the amax.

The forward writes Q, K and V back to back into one buffer and, per program, the max |output|;
the backward is NVIDIA's. The packed buffer and the amax parts are handed to TE's
``nvfp4_qkv_packed.combine_and_quantize``, so the FP8 DPA quantizes Q|K|V in place without the
concatenation copy and without the current-scaling amax pass (NVFP4_QKV_PACKED and
NVFP4_QKV_PACKED_AMAX of the 20261006 model, whose RoPE kernel had no QK norm).

``install()`` replaces ``opt_kernels.qknorm_rope.qknorm_rope``, the function NVIDIA's
``qknorm_rope_patch._dispatch`` calls (also from its self-check), and installs the TE
``combine_and_quantize``.
"""

import os

import torch
import triton
import triton.language as tl

from opt_kernels import qknorm_rope as _nv

HD = 64
H2 = 32
_AMAX = os.environ.get("NVFP4_QKV_PACKED_AMAX", "0") == "1"
_stats = {"calls": 0}


@triton.jit(do_not_specialize=["B", "S", "NROWS"])
def _qknorm_rope_fwd_amax_kernel(
    x_ptr,
    gq_ptr,
    gk_ptr,
    tab_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    parts_ptr,
    B,
    S,
    NROWS,
    eps,
    NG: tl.constexpr,
    NQ: tl.constexpr,
    R: tl.constexpr,
    ZC: tl.constexpr,
):
    # opt_kernels.qknorm_rope._qknorm_rope_fwd_kernel plus the max |output| of the program.
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
    alo = tl.where(is_v[:, None], tl.abs(flo), tl.abs(olo.to(tl.float32)))
    ahi = tl.where(is_v[:, None], tl.abs(fhi), tl.abs(ohi.to(tl.float32)))
    a = tl.where(mrow, tl.maximum(alo, ahi), 0.0)
    tl.store(parts_ptr + pid, tl.max(tl.max(a, axis=1), axis=0))


def packed_fwd(x, q_freqs, k_freqs, gq, gk, eps, zero_centered, amax=_AMAX):
    """Returns q, k, v (consecutive views of one buffer) and the per-program amax parts or None."""
    s, b, ng, nq = _nv._check_inputs(x, q_freqs, k_freqs, gq, gk)
    n_q = s * b * ng * nq * HD
    n_k = s * b * ng * HD
    buf = x.new_empty(n_q + 2 * n_k)
    q = buf[:n_q].view(s, b, ng * nq, HD)
    k = buf[n_q : n_q + n_k].view(s, b, ng, HD)
    v = buf[n_q + n_k :].view(s, b, ng, HD)
    tab = _nv._sincos(q_freqs, k_freqs, s)
    nrows = s * b * ng * (nq + 2)
    grid = (triton.cdiv(nrows, _nv._FWD_ROWS),)
    parts = None
    if amax:
        parts = torch.empty(grid[0], dtype=torch.float32, device=x.device)
        _qknorm_rope_fwd_amax_kernel[grid](
            x, gq, gk, tab, q, k, v, parts, b, s, nrows, eps,
            NG=ng, NQ=nq, R=_nv._FWD_ROWS, ZC=zero_centered, num_warps=_nv._FWD_WARPS,
        )
    else:
        _nv._qknorm_rope_fwd_kernel[grid](
            x, gq, gk, tab, q, k, v, b, s, nrows, eps,
            NG=ng, NQ=nq, R=_nv._FWD_ROWS, ZC=zero_centered, num_warps=_nv._FWD_WARPS,
        )
    return q, k, v, buf, parts


class QKNormRoPEPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q_freqs, k_freqs, gq, gk, eps, zero_centered):
        q, k, v, buf, parts = packed_fwd(x, q_freqs, k_freqs, gq, gk, float(eps), bool(zero_centered))
        if _packed_mod is not None:
            _packed_mod._last[0] = (buf.untyped_storage().data_ptr(), parts) if parts is not None else None
        _stats["calls"] += 1
        ctx.save_for_backward(x, q_freqs, k_freqs, gq, gk)
        ctx.eps = float(eps)
        ctx.zc = bool(zero_centered)
        return q, k, v

    backward = staticmethod(_nv.QKNormRoPE.backward)


def qknorm_rope(x, q_freqs, k_freqs, gq, gk, eps, zero_centered=False):
    return QKNormRoPEPacked.apply(x, q_freqs, k_freqs, gq, gk, eps, zero_centered)


_packed_mod = None


def install():
    """Route NVIDIA's dispatch to the packed kernel and install TE's in-place Q|K|V quantize."""
    global _packed_mod
    from transformer_engine.pytorch.ops.fused import grouped_mlp as _te_grouped_mlp

    _packed_mod = _te_grouped_mlp._v7_module("nvfp4_qkv_packed")
    _packed_mod.install()
    _nv.qknorm_rope = qknorm_rope
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"FIRMUS qknorm_rope packed installed (amax {int(_AMAX)})", flush=True)
