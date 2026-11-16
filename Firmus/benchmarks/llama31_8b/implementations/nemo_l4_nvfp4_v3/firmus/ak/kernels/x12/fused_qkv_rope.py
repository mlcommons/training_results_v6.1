"""Byte-exact fused QKV rotary embedding (Triton) -- whole-file replacement of TE's kernels.

One program per (s, b) position, two warps.  The D elements of a head are treated as two
explicit contiguous halves x0 = x[0:D/2] and x1 = x[D/2:D]; each half is loaded exactly
once and BOTH output halves come out of those two register values:

    y0 = fma(x0, cos_lo, (-x1) * sin_lo)      # d <  D/2:  xr = -x[d + D/2]
    y1 = fma(x1, cos_hi,  x0  * sin_hi)       # d >= D/2:  xr =  x[d - D/2]

so no element is fetched twice: the parent's partner request (a second, permuted load of
every element of the tile) is gone, halving the qkv L1 traffic and the load instruction
count.  TE's contraction order, the sign-bit negation and the libdevice cos/sin are
untouched.

The four fp32 angle slices (cos/sin of each half) are the only table reads; they are
evaluated once per position by a tiny libdevice table kernel, and when q_freqs and
k_freqs are the same tensor (training) the key heads reuse the query's angle registers.
Two warps per position: the angle slices are replicated per thread, so halving the
threads halves the fp32 table traffic (8 KB instead of 16 KB per program).  The backward
is the same shape with the sine taken at the rotated index and negated for the upper
half: y0 = fma(g0, cos_lo, g1 * sin_hi); y1 = fma(g1, cos_hi, (-g0) * sin_lo).  Value
heads are copied unchanged.

2026-10-07 (x12f): with AK_X12_TAB_CACHE=1 the angle table is cached between calls.  Every
layer's forward and backward evaluated the same table from the same freqs tensor (64 table
launches per training step); now the first eager call per freqs tensor computes it and keeps
it, keyed on the freqs' storage (data pointer, shape, strides, dtype, device, the tensor's
version counter) with a strong reference to the freqs, so their storage cannot be freed and
its address reused while the entry lives.  Entries are never evicted (a graph may hold the
table's address); at most 8 are made.  Under CUDA-graph capture a hit is served from the
cache (no kernel in the graph; the cached table was written eagerly and never changes), a
miss computes the table inside the capture without caching it.  The first eager hit is
checked: a fresh table against the cached one, as bytes (a mismatch raises); rank 0 prints
one ARMED line, one line at the first capture hit and one at the first capture miss.
"""

import os

import torch
import triton
import triton.language as tl

_TAB_CACHE = os.environ.get("AK_X12_TAB_CACHE", "0") == "1"
_TAB_STATE = {"entries": {}, "checked": False, "eager_hits": 0, "capture_hits": 0, "capture_misses": 0,
              "computed": 0, "announced_capture": False}


def _rank():
    try:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0

_NUM_WARPS = 2
_TAB_ROWS = 8
_TAB_WARPS = 4


# Exact negation (the sign bit flipped).  Triton's unary minus lowers to ``0 - x``, which turns +0.0
# into +0.0 instead of -0.0; TE's ``-static_cast<float>(src)`` is a true negation.  -2147483648 is
# 0x80000000 as an int32.
@triton.jit
def _neg(x):
    return (x.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)


@triton.jit
def _rope_table_kernel(qf_ptr, kf_ptr, tab_ptr, s, D: tl.constexpr, ROWS: tl.constexpr,
                       SAME: tl.constexpr):
    """tab[r, 0:D] = cos(qf[r]), tab[r, D:2D] = sin(qf[r]) -- libdevice cos/sin, once per
    angle.  When q and k are the same tensor (SAME) the key angles are the query angles and
    the second pair of sections is not evaluated at all."""
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)[:, None]
    d = tl.arange(0, D)[None, :]
    m = r < s
    off = r * D + d
    fq = tl.load(qf_ptr + off, mask=m, other=0.0)
    base = r * (4 * D) + d
    tl.store(tab_ptr + base, tl.cos(fq), mask=m)
    tl.store(tab_ptr + base + D, tl.sin(fq), mask=m)
    if not SAME:
        fk = tl.load(kf_ptr + off, mask=m, other=0.0)
        tl.store(tab_ptr + base + 2 * D, tl.cos(fk), mask=m)
        tl.store(tab_ptr + base + 3 * D, tl.sin(fk), mask=m)


@triton.jit
def _qkv_rope_fwd_kernel(qkv_ptr, tab_ptr, q_ptr, k_ptr, v_ptr, b,
                         D: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr,
                         TOTAL: tl.constexpr, SAME: tl.constexpr):
    pid = tl.program_id(0)                     # one (s, b) position of H*TOTAL elements
    s_id = pid // b
    half: tl.constexpr = D // 2
    dd = tl.arange(0, half)[None, :]           # offset inside a half
    tb = s_id * (4 * D)                        # this position's row of the angle table
    c0 = tl.load(tab_ptr + tb + dd)                     # cos of the low half
    c1 = tl.load(tab_ptr + tb + half + dd)              # cos of the high half
    t0 = tl.load(tab_ptr + tb + D + dd)                 # sin of the low half
    t1 = tl.load(tab_ptr + tb + D + half + dd)          # sin of the high half
    if SAME:
        kc0 = c0                                # same tensor: the key angles are these
        kc1 = c1
        kt0 = t0
        kt1 = t1
    else:
        kc0 = tl.load(tab_ptr + tb + 2 * D + dd)
        kc1 = tl.load(tab_ptr + tb + 2 * D + half + dd)
        kt0 = tl.load(tab_ptr + tb + 3 * D + dd)
        kt1 = tl.load(tab_ptr + tb + 3 * D + half + dd)

    base = pid * (H * TOTAL)
    # ---- query heads: H*NQ rows, each element loaded exactly once
    r = tl.arange(0, H * NQ)[:, None]
    qrow = base + (r // NQ) * TOTAL + (r % NQ) * D
    x0 = tl.load(qkv_ptr + qrow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + qrow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, c0, _neg(x1) * t0)         # d < half : xr = -x[d + half]
    y1 = tl.fma(x1, c1, x0 * t1)               # d >= half: xr =  x[d - half]
    qout = pid * (H * NQ * D) + r * D
    tl.store(q_ptr + qout + dd, y0.to(tl.bfloat16))
    tl.store(q_ptr + qout + half + dd, y1.to(tl.bfloat16))
    # ---- key heads: one row per group, angles from the k table
    g = tl.arange(0, H)[:, None]
    krow = base + g * TOTAL + NQ * D
    x0 = tl.load(qkv_ptr + krow + dd).to(tl.float32)
    x1 = tl.load(qkv_ptr + krow + half + dd).to(tl.float32)
    y0 = tl.fma(x0, kc0, _neg(x1) * kt0)
    y1 = tl.fma(x1, kc1, x0 * kt1)
    kout = pid * (H * D) + g * D
    tl.store(k_ptr + kout + dd, y0.to(tl.bfloat16))
    tl.store(k_ptr + kout + half + dd, y1.to(tl.bfloat16))
    # ---- value heads: copied unchanged
    gv = tl.arange(0, H)[:, None]
    dv = tl.arange(0, D)[None, :]
    tl.store(v_ptr + pid * (H * D) + gv * D + dv,
             tl.load(qkv_ptr + base + gv * TOTAL + (NQ + 1) * D + dv))


@triton.jit
def _qkv_rope_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, tab_ptr, dqkv_ptr, b,
                         D: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr,
                         TOTAL: tl.constexpr, SAME: tl.constexpr):
    pid = tl.program_id(0)
    s_id = pid // b
    half: tl.constexpr = D // 2
    dd = tl.arange(0, half)[None, :]
    tb = s_id * (4 * D)
    c0 = tl.load(tab_ptr + tb + dd)                     # cos of the element itself
    c1 = tl.load(tab_ptr + tb + half + dd)
    t0 = tl.load(tab_ptr + tb + D + dd)                 # sine at the rotated index
    t1 = tl.load(tab_ptr + tb + D + half + dd)
    if SAME:
        kc0 = c0
        kc1 = c1
        kt0 = t0
        kt1 = t1
    else:
        kc0 = tl.load(tab_ptr + tb + 2 * D + dd)
        kc1 = tl.load(tab_ptr + tb + 2 * D + half + dd)
        kt0 = tl.load(tab_ptr + tb + 3 * D + dd)
        kt1 = tl.load(tab_ptr + tb + 3 * D + half + dd)

    # ---- query gradients: dq rows are the H*NQ contiguous rows of the position
    r = tl.arange(0, H * NQ)[:, None]
    qin = pid * (H * NQ * D) + r * D
    g0 = tl.load(dq_ptr + qin + dd).to(tl.float32)
    g1 = tl.load(dq_ptr + qin + half + dd).to(tl.float32)
    y0 = tl.fma(g0, c0, g1 * t1)               # d < half : gr = g[d+half], s = +sin[d+half]
    y1 = tl.fma(g1, c1, _neg(g0) * t0)         # d >= half: gr = g[d-half], s = -sin[d-half]
    qout = pid * (H * TOTAL) + (r // NQ) * TOTAL + (r % NQ) * D
    tl.store(dqkv_ptr + qout + dd, y0.to(tl.bfloat16))
    tl.store(dqkv_ptr + qout + half + dd, y1.to(tl.bfloat16))
    # ---- key gradients
    g = tl.arange(0, H)[:, None]
    kin = pid * (H * D) + g * D
    g0 = tl.load(dk_ptr + kin + dd).to(tl.float32)
    g1 = tl.load(dk_ptr + kin + half + dd).to(tl.float32)
    y0 = tl.fma(g0, kc0, g1 * kt1)
    y1 = tl.fma(g1, kc1, _neg(g0) * kt0)
    kout = pid * (H * TOTAL) + g * TOTAL + NQ * D
    tl.store(dqkv_ptr + kout + dd, y0.to(tl.bfloat16))
    tl.store(dqkv_ptr + kout + half + dd, y1.to(tl.bfloat16))
    # ---- value gradients: copied unchanged
    gv = tl.arange(0, H)[:, None]
    dv = tl.arange(0, D)[None, :]
    tl.store(dqkv_ptr + pid * (H * TOTAL) + gv * TOTAL + (NQ + 1) * D + dv,
             tl.load(dv_ptr + pid * (H * D) + gv * D + dv))


class FusedQKVRoPE:
    """forward(qkv, q_freqs, k_freqs, split) -> (q, k, v); backward(dq, dk, dv, q_freqs, k_freqs, split) -> dqkv.
    sbhd, not interleaved, cp_size 1: the training call."""

    def __init__(self, device="cuda"):
        self.device = device

    def _table(self, qf, kf, s, d, same):
        tab = torch.empty(s, 4 * d, dtype=torch.float32, device=qf.device)
        _rope_table_kernel[(triton.cdiv(s, _TAB_ROWS),)](
            qf, kf, tab, s, D=d, ROWS=_TAB_ROWS, SAME=same, num_warps=_TAB_WARPS)
        return tab

    def _angles(self, q_freqs, k_freqs, s, d):
        """fp32 cos/sin of every angle of every position.  q_freqs and k_freqs are honoured
        separately; when they are the very same tensor (training) only one pair of sections
        is evaluated and the key heads reuse it.  With AK_X12_TAB_CACHE=1 the table of a freqs
        tensor seen before (same storage, unmodified) is the cached one (see the module note)."""
        qf = q_freqs.contiguous()
        kf = k_freqs.contiguous()
        same = qf.data_ptr() == kf.data_ptr()
        if not _TAB_CACHE:
            return self._table(qf, kf, s, d, same), same
        st = _TAB_STATE
        key = (qf.data_ptr(), tuple(qf.shape), tuple(qf.stride()), kf.data_ptr(), tuple(kf.shape), tuple(kf.stride()),
               qf.dtype, kf.dtype, str(qf.device), s, d, same)
        ent = st["entries"].get(key)
        capturing = torch.cuda.is_current_stream_capturing()
        if ent is not None and ent["qv"] == qf._version and ent["kv"] == kf._version:
            if capturing:
                st["capture_hits"] += 1
                if not st["announced_capture"]:
                    st["announced_capture"] = True
                    if _rank() == 0:
                        print("[auto_kernel x12] RoPE angle table served from the cache under graph capture ([{}, {}] "
                              "fp32, the table checked eagerly; capture hits so far {}, computed inside a capture "
                              "{})".format(s, 4 * d, st["capture_hits"], st["capture_misses"]), flush=True)
                return ent["tab"], same
            st["eager_hits"] += 1
            if not st["checked"]:
                st["checked"] = True
                fresh = self._table(qf, kf, s, d, same)
                w = 2 * d if same else 4 * d      # with SAME the key sections are never written nor read
                ok = bool(torch.equal(fresh[:, :w].view(torch.int32), ent["tab"][:, :w].view(torch.int32)))
                if _rank() == 0:
                    print("[auto_kernel x12] RoPE angle table cached ARMED: [{}, {}] fp32 (SAME {}), keyed on the freqs' "
                          "storage and version; a fresh table v the cached one, as bytes: {}".format(
                              s, 4 * d, same, ok), flush=True)
                if not ok:
                    raise RuntimeError("[auto_kernel x12] E0 self-check FAILED: the cached RoPE angle table differs")
            return ent["tab"], same
        tab = self._table(qf, kf, s, d, same)
        if capturing:
            st["capture_misses"] += 1
            if st["capture_misses"] == 1 and _rank() == 0:
                print("[auto_kernel x12] RoPE angle table computed inside a graph capture: no cached table for these "
                      "freqs ([{}, {}] fp32; entries {}, capture hits so far {})".format(
                          s, 4 * d, len(st["entries"]), st["capture_hits"]), flush=True)
        else:
            st["computed"] += 1
            if ent is None and len(st["entries"]) < 8:
                # strong references to the freqs: their storage stays allocated, so its address cannot be reused;
                # an entry is never replaced or dropped (a captured graph may read its table)
                st["entries"][key] = {"tab": tab, "q": qf, "k": kf, "qv": qf._version, "kv": kf._version}
        return tab, same

    def forward(self, qkv, q_freqs, k_freqs, split):
        s, b, h, total = qkv.shape
        qs, ks, vs = split
        d = q_freqs.shape[-1]
        assert total == qs + ks + vs and ks == d and vs == d and qs % d == 0
        nq = qs // d
        q = torch.empty(s, b, h * nq, d, dtype=qkv.dtype, device=qkv.device)
        k = torch.empty(s, b, h, d, dtype=qkv.dtype, device=qkv.device)
        v = torch.empty(s, b, h, d, dtype=qkv.dtype, device=qkv.device)
        tab, same = self._angles(q_freqs, k_freqs, s, d)
        _qkv_rope_fwd_kernel[(s * b,)](qkv, tab, q, k, v, b,
                                       D=d, NQ=nq, H=h, TOTAL=total, SAME=same,
                                       num_warps=_NUM_WARPS)
        return q, k, v

    def backward(self, dq, dk, dv, q_freqs, k_freqs, split):
        s, b, hq, d = dq.shape
        h = dk.shape[2]
        qs, ks, vs = split
        nq = qs // d
        assert hq == h * nq and dk.shape[-1] == d and dv.shape == dk.shape
        dq = dq.contiguous()
        dk = dk.contiguous()
        dv = dv.contiguous()
        dqkv = torch.empty(s, b, h, qs + ks + vs, dtype=dq.dtype, device=dq.device)
        tab, same = self._angles(q_freqs, k_freqs, s, d)
        _qkv_rope_bwd_kernel[(s * b,)](dq, dk, dv, tab, dqkv, b,
                                       D=d, NQ=nq, H=h, TOTAL=qs + ks + vs, SAME=same,
                                       num_warps=_NUM_WARPS)
        return dqkv
