# SPDX-License-Identifier: Apache-2.0
"""Fused RoPE backward + QK RMSNorm backward + QKV join, emitting the NVFP4 amax pair and dbias.

``grad_b`` is a deterministic fixed-order fp32 column sum, not bitwise ``tma_bgrad_amax``'s.
"""

from __future__ import annotations

import functools
import os
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
import torch.library
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T

ROWS = 16                  # rows per tile (= the NVFP4 block)
DH = 128                   # head dim
CHUNK = 8                  # bf16 per thread per row (16 B)
CHUNKS = DH // CHUNK       # 16 threads per row
THREADS = ROWS * CHUNKS    # 256
WARPS = THREADS // 32      # 8
MIN_CTAS_PER_SM = 2        # launch bound: 128-register budget
# Preferred shared-memory carveout, percent; admits two ~64 KB CTAs per SM.
SMEM_CARVEOUT_PCT = 100
PITCH = DH + 8                        # staged row pitch, bf16 (272 B)
STAGE_ELEMS = 3 * ROWS * PITCH        # one staged tile, bf16 (6528)
STAGE_BYTES = STAGE_ELEMS * 2         # 13056
SLICES_PER_WARP = 3                   # 24 16x16 slices per tile / 8 warps
HEADS = 24                            # smem dbias accumulator sized for H = 24
D3_STATIC = 3 * HEADS * DH            # 9216

FIN_THREADS = 256
FIN_GROUPS = 8                        # slot groups per finaliser block
FIN_QUADS = FIN_THREADS // FIN_GROUPS  # 32 column quads per block
FIN_COLS = FIN_QUADS * 4              # 128 columns per block
FIN_BATCH = 4                         # slot loads in flight per group

# Bytes per element of grad_qkv (bf16).
GRAD_ITEMSIZE = 2

# Environment override for fused_above_numel, in elements.
FUSE_ABOVE_ENV = "FLAT_ROPE_BGRAD_FUSE_ABOVE_NUMEL"

F32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32
I64 = cutlass.Int64
I16 = cutlass.Int16

_ONES_BF16X2 = 0x3F803F80             # bf16x2 {1.0, 1.0}
_BF16_HI_MASK = -65536                # 0xFFFF0000 as a signed Int32

# Inline PTX. Raw llvm.inline_asm with explicit register constraints where inline_ptx
# would fold a constant operand into an immediate that ptxas rejects.
_DIV_FULL_ASM = "div.full.f32 $0, $1, $2;"                  # __fdividef
_PACK_BF16X2_ASM = "cvt.rn.bf16x2.f32 $0, $1, $2;"          # d = {hi, lo}, RN both halves
_MMA_16816_BF16 = (
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
)
_MMA_CONSTRAINTS = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
_LDMATRIX_X4_TRANS = (
    "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
    "{ {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];"
)
_MAX_XORSIGN_ABS_BF16X2 = "max.xorsign.abs.bf16x2 {$w0}, {$r0}, {$r1};"


def _div_full(a, b):
    res = llvm.inline_asm(
        T.f32(), [a.ir_value(), b.ir_value()], _DIV_FULL_ASM, "=f,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return F32(res)


def _pack_bf16x2(lo, hi):
    res = llvm.inline_asm(
        T.i32(), [hi.ir_value(), lo.ir_value()], _PACK_BF16X2_ASM, "=r,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return I32(res)


def _mma_16816(a, b0, b1, c):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [v.ir_value() for v in (a[0], a[1], a[2], a[3], b0, b1, c[0], c[1], c[2], c[3])],
        _MMA_16816_BF16, _MMA_CONSTRAINTS,
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(F32(llvm.extractvalue(T.f32(), res, [i])) for i in range(4))


def _ldmatrix_x4_trans(addr):
    return cute.arch.inline_ptx(
        _LDMATRIX_X4_TRANS, write_only_types=[I32] * 4, read_only_args=[addr],
    )


def _max_xorsign_abs(a, b):
    return cute.arch.inline_ptx(
        _MAX_XORSIGN_ABS_BF16X2, write_only_types=[I32], read_only_args=[a, b],
    )


def _bf16_lo(w):
    return (w << 16).bitcast(F32)


def _bf16_hi(w):
    return (w & I32(_BF16_HI_MASK)).bitcast(F32)


def _round_bf16(x):
    return x.to(BF16).to(F32)


def _warp_max(v):
    for off in (16, 8, 4, 2, 1):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=off))
    return v


# Traced helpers (plain Python loops, no dynamic if).
def _load_tile(h, bb, l, rg, H, c, D, Dh, in_stride_b, in_stride_b_h, stride_qkv_l,
               gqp, gkp, gvp32, qkvp, rqp, rkp):
    """The seven ``ld.global.nc`` loads of tile ``(rg, h)``.

    ``grad_v`` comes back as four Int32 words (packed bf16x2), not eight bf16.
    """
    col = h * DH + c * CHUNK
    colh = h * (DH // 2) + c * (CHUNK // 2)
    in_off = bb.to(I64) * in_stride_b + (l * D + col).to(I64)
    vq = (gqp + in_off).load(16, count=CHUNK, is_invariant=True)
    vk = (gkp + in_off).load(16, count=CHUNK, is_invariant=True)
    in_off_h = bb.to(I64) * in_stride_b_h + (l * Dh + colh).to(I64)
    vv = (gvp32 + in_off_h).load(16, count=4, is_invariant=True)
    pre = rg.to(I64) * stride_qkv_l + col.to(I64)
    pq = (qkvp + pre).load(16, count=CHUNK, is_invariant=True)
    pk = (qkvp + (pre + D.to(I64))).load(16, count=CHUNK, is_invariant=True)
    ro = rg * H + h
    rq = (rqp + ro).load(4, is_invariant=True)
    rk = (rkp + ro).load(4, is_invariant=True)
    return vq, vk, vv, pq, pk, rq, rk


def _rowgroup(g, r, L):
    """``(rg, bb, l)`` of row group ``g`` for this thread's row ``r``."""
    rg = g * ROWS + r
    bb = rg // L
    l = rg - bb * L
    return rg, bb, l


def _wrap_select(hn, h_hi, h_cur, last_g, bb, l, rg, bb_n, l_n):
    """Prefetch target: (g, hn) if hn < h_hi, else (g + 1, 0), else the current tile again."""
    w = cutlass.min(cutlass.max(hn - h_hi + 1, 0), 1)     # hn >= h_hi
    over = w * last_g                                     # wrapped past the last g
    sel = w - over                                        # take the next row group
    bb_s = bb + (bb_n - bb) * sel
    l_s = l + (l_n - l) * sel
    rg_s = rg + ROWS * sel
    h_s = hn - hn * sel + (h_cur - hn) * over
    return h_s, bb_s, l_s, rg_s


def _triton_row_sum(gyw, pre, lane):
    """Row sum of ``gyw * pre`` in the exact order Triton 3.8.0 lowers ``tl.sum(axis=0)``."""
    t = gyw * pre
    u = []
    for e in range(CHUNK):
        tr = cute.arch.shuffle_sync_bfly(t[e], 2)
        u.append(cute.math.fma(gyw[e], pre[e], tr))
    v = []
    for e in range(CHUNK):
        ur = cute.arch.shuffle_sync_bfly(u[e], 1)
        v.append(u[e] + ur)
    w0 = v[0] + v[4]
    w1 = v[1] + v[5]
    w2 = v[2] + v[6]
    w3 = v[3] + v[7]
    x0 = w0 + w2
    x1 = w1 + w3
    y = x0 + x1
    yr = cute.arch.shuffle_sync_bfly(y, 8)
    s = y + yr
    sr = cute.arch.shuffle_sync_bfly(s, 4)
    tot = s + sr
    return cute.arch.shuffle_sync(tot, lane & 16)


def _tile_math(cq, ck, cpq, cpk, crq, crk, cs_c, cs_s, wq, wk, lane, acc_q, acc_k):
    """RoPE backward + RMSNorm backward of one tile.

    Returns the packed bf16x2 q and k words and the updated norm-weight partials.
    """
    qf = cq.to(F32)
    kf = ck.to(F32)
    pqf = cpq.to(F32)
    pkf = cpk.to(F32)
    gq_l = []
    gk_l = []
    for i4 in range(4):
        e0 = 2 * i4
        e1 = e0 + 1
        q0 = qf[e0]
        q1 = qf[e1]
        gq_l.append(cute.math.fma(cs_c[e0], q0, cs_s[e0] * q1))
        gq_l.append(cute.math.fma(cs_c[e1], q1, cs_s[e1] * (-q0)))
        k0 = kf[e0]
        k1 = kf[e1]
        gk_l.append(cute.math.fma(cs_c[e0], k0, cs_s[e0] * k1))
        gk_l.append(cute.math.fma(cs_c[e1], k1, cs_s[e1] * (-k0)))
    gradq = cutlass.Vector.from_elements(tuple(gq_l), F32)
    gradk = cutlass.Vector.from_elements(tuple(gk_l), F32)
    gyw_q = gradq * wq
    gyw_k = gradk * wk
    sum_q = _triton_row_sum(gyw_q, pqf, lane)
    sum_k = _triton_row_sum(gyw_k, pkf, lane)
    c_q = _div_full((sum_q * crq) * crq, F32(128.0))
    c_k = _div_full((sum_k * crk) * crk, F32(128.0))
    oq_w = []
    ok_w = []
    aq = []
    ak = []
    for i4 in range(4):
        e0 = 2 * i4
        e1 = e0 + 1
        oq_w.append(_pack_bf16x2(cute.math.fma(-c_q, pqf[e0], gyw_q[e0]) * crq,
                                 cute.math.fma(-c_q, pqf[e1], gyw_q[e1]) * crq))
        ok_w.append(_pack_bf16x2(cute.math.fma(-c_k, pkf[e0], gyw_k[e0]) * crk,
                                 cute.math.fma(-c_k, pkf[e1], gyw_k[e1]) * crk))
    for e in range(CHUNK):
        aq.append(cute.math.fma(gradq[e] * pqf[e], crq, acc_q[e]))
        ak.append(cute.math.fma(gradk[e] * pkf[e], crk, acc_k[e]))
    acc_q = cutlass.Vector.from_elements(tuple(aq), F32)
    acc_k = cutlass.Vector.from_elements(tuple(ak), F32)
    return oq_w, ok_w, acc_q, acc_k


def _rowmax(rmax_packed, oq_w, ok_w, cv):
    """Rowwise amax over the 12 packed bf16x2 output words."""
    m = _max_xorsign_abs(oq_w[0], oq_w[1])
    m = _max_xorsign_abs(m, _max_xorsign_abs(oq_w[2], oq_w[3]))
    m = _max_xorsign_abs(m, _max_xorsign_abs(ok_w[0], ok_w[1]))
    m = _max_xorsign_abs(m, _max_xorsign_abs(ok_w[2], ok_w[3]))
    m = _max_xorsign_abs(m, _max_xorsign_abs(cv[0], cv[1]))
    m = _max_xorsign_abs(m, _max_xorsign_abs(cv[2], cv[3]))
    return _max_xorsign_abs(rmax_packed, m)


def _consume_math(buf_addr, slice_off, bfrag, ones, cmax):
    """Post-RHT amax and column sums of this warp's three 16x16 slices of one staged tile.

    Returns ``(cmax, cs)``; in ``cs`` lanes with ``tq == 0`` hold columns ``gi`` and ``gi + 8``.
    """
    cs = []
    for j in range(SLICES_PER_WARP):
        a = _ldmatrix_x4_trans(buf_addr + slice_off[j])
        zero = (F32(0.0),) * 4
        d0 = _mma_16816(a, bfrag[0][0], bfrag[0][1], zero)
        d1 = _mma_16816(a, bfrag[1][0], bfrag[1][1], zero)
        for q in range(4):
            cmax = cute.arch.fmax(cmax, cute.math.abs(d0[q]))
            cmax = cute.arch.fmax(cmax, cute.math.abs(d1[q]))
        d = _mma_16816(a, ones, ones, zero)
        cs.append((d[0], d[2]))
    return cmax, cs


def _rmw_dbias(s_db, prev_off, col_base, cs):
    """Plain RMW of the smem dbias accumulator (one lane per column per tile)."""
    for j in range(SLICES_PER_WARP):
        p0 = s_db + (col_base[j] + prev_off)
        p0.store(p0.load() + cs[j][0])
        p8 = s_db + (col_base[j] + prev_off + 8)
        p8.store(p8.load() + cs[j][1])


@cute.kernel
def rope_norm_join_bgrad_amax_ctm_kernel(
    gq: cute.Tensor, gk: cute.Tensor, gv: cute.Tensor, qkv: cute.Tensor,
    q_rstd: cute.Tensor, k_rstd: cute.Tensor,
    qn_w: cute.Tensor, kn_w: cute.Tensor, freqs: cute.Tensor,
    out: cute.Tensor, grad_qn_w: cute.Tensor, grad_kn_w: cute.Tensor,
    rht: cute.Tensor, ws_dbias: cute.Tensor, ws_amax: cute.Tensor,
    amax: cute.Tensor, counter: cute.Tensor,
    L: I32, H: I32, in_stride_b: I64, stride_qkv_l: I64, freq_stride_s: I64,
    pos_offset: I32, n_tiles: I32,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    r = tid >> 4
    c = tid & 15
    gi = lane >> 2
    tq = lane & 3
    D = H * DH
    Dh = H * (DH // 2)
    D3h = 3 * Dh
    in_stride_b_h = in_stride_b // 2

    # shared memory
    smem = utils.SmemAllocator()
    s_w = smem.allocate_array(F32, 2 * DH, byte_alignment=16)                 # [2][128]
    stage = smem.allocate_array(BF16, 2 * STAGE_ELEMS, byte_alignment=16)      # [2][3][16][136]
    s_acc = cutlass.Pointer(stage, dtype=F32)         # [2][16][128] fp32, aliases stage
    s_db = cutlass.Pointer(smem.allocate_array(F32, D3_STATIC, byte_alignment=16), dtype=F32)
    mbar = smem.allocate_array(I64, 2, byte_alignment=8)
    amax_stage = cutlass.Pointer(smem.allocate_array(F32, 2 * WARPS, byte_alignment=16), dtype=F32)
    s_flag = cutlass.Pointer(smem.allocate_array(I32, 1, byte_alignment=4), dtype=I32)

    if tid < DH:
        (s_w + tid).store(qn_w.iterator[tid].to(F32))
    else:
        (s_w + tid).store(kn_w.iterator[tid - DH].to(F32))
    for i in cutlass.range(tid, D3_STATIC, THREADS):
        (s_db + i).store(F32(0.0))
    if tid == 0:
        cute.arch.mbarrier_init(mbar, THREADS)
        cute.arch.mbarrier_init(mbar + 1, THREADS)
    cute.arch.mbarrier_init_fence()
    if tid == 0:
        s_flag.store(I32(0))
    # B fragments of R; rht viewed as Int16
    bfrag = []
    ones = I32(_ONES_BF16X2)
    for jb in cutlass.range_constexpr(2):
        nn = jb * 8 + gi
        r0 = rht[(2 * tq) * 16 + nn].to(I32) & 0xFFFF
        r1 = rht[(2 * tq + 1) * 16 + nn].to(I32) & 0xFFFF
        r8 = rht[(2 * tq + 8) * 16 + nn].to(I32) & 0xFFFF
        r9 = rht[(2 * tq + 9) * 16 + nn].to(I32) & 0xFFFF
        bfrag.append((r0 | (r1 << 16), r8 | (r9 << 16)))
    cute.arch.sync_threads()

    gqp = cutlass.Pointer(gq.iterator, dtype=BF16)
    gkp = cutlass.Pointer(gk.iterator, dtype=BF16)
    gvp32 = cutlass.Pointer(gv.iterator, dtype=I32)
    qkvp = cutlass.Pointer(qkv.iterator, dtype=BF16)
    outp32 = cutlass.Pointer(out.iterator, dtype=I32)
    rqp = cutlass.Pointer(q_rstd.iterator, dtype=F32)
    rkp = cutlass.Pointer(k_rstd.iterator, dtype=F32)
    fp = cutlass.Pointer(freqs.iterator, dtype=F32)
    swp = cutlass.Pointer(s_w, dtype=F32)
    wsd = cutlass.Pointer(ws_dbias.iterator, dtype=F32)

    # persistent, contiguous tile range of this CTA (n_tiles * gdim fits Int64)
    nt = n_tiles.to(I64)
    b64 = bidx.to(I64)
    g64 = gdim.to(I64)
    t0 = ((nt * b64) // g64).to(I32)
    t1 = ((nt * (b64 + 1)) // g64).to(I32)

    wq_off = c * CHUNK
    wk_off = DH + c * CHUNK

    # staging geometry: this thread's row (Int32 words) and this warp's three slices.
    # ldmatrix lane map: rows (lane & 7) + 8 * (lane >> 4), cols ((lane >> 3) & 1) * 8.
    row_off32 = r * (PITCH // 2) + c * (CHUNK // 2)
    lane_off = ((lane & 7) + 8 * (lane >> 4)) * (PITCH * 2) + ((lane >> 3) & 1) * 16
    slice_off = []
    col_base = []
    for j in cutlass.range_constexpr(SLICES_PER_WARP):
        s = warp * SLICES_PER_WARP + j
        t = s >> 3
        b = s & 7
        slice_off.append(t * (ROWS * PITCH * 2) + b * 32 + lane_off)
        col_base.append(t * D + b * 16 + gi)
    stage_base = stage.toint()
    stg32 = cutlass.Pointer(stage, dtype=I32)

    zero = F32(0.0)
    acc_q = cutlass.Vector.from_elements((zero,) * CHUNK, F32)
    acc_k = cutlass.Vector.from_elements((zero,) * CHUNK, F32)
    rmax_packed = I32(0)
    cmax = F32(0.0)
    k = I32(0)          # tiles staged so far
    prev_off = I32(0)   # dbias col offset of prev staged tile

    g_lo = t0 // H
    g_hi = (t1 - 1) // H

    # head loop, unrolled by two
    rg0, bb0, l0 = _rowgroup(g_lo, r, L)
    h0 = t0 - g_lo * H
    cq, ck, cv, cpq, cpk, crq, crk = _load_tile(
        h0, bb0, l0, rg0, H, c, D, Dh, in_stride_b, in_stride_b_h, stride_qkv_l,
        gqp, gkp, gvp32, qkvp, rqp, rkp)
    for g in cutlass.range(g_lo, g_hi + 1):
        rg, bb, l = _rowgroup(g, r, L)
        rg_n, bb_n, l_n = _rowgroup(g + 1, r, L)
        last_g = cutlass.min(cutlass.max(g - g_hi + 1, 0), 1)
        # cos/sin for this row group (once per H tiles)
        foff = (l + pos_offset).to(I64) * freq_stride_s + (c * CHUNK).to(I64)
        fv = (fp + foff).load(16, count=CHUNK, is_invariant=True)
        cos_l = []
        sin_l = []
        for e in cutlass.range_constexpr(CHUNK):
            cos_l.append(cute.math.cos(fv[e]))   # libdevice __nv_cosf == tl.math.cos
            sin_l.append(cute.math.sin(fv[e]))
        cs_c = cutlass.Vector.from_elements(tuple(cos_l), F32)
        cs_s = cutlass.Vector.from_elements(tuple(sin_l), F32)
        o_g = rg.to(I64) * D3h.to(I64) + (c * (CHUNK // 2)).to(I64)

        h_lo = cutlass.max(t0 - g * H, 0)
        h_hi = cutlass.min(t1 - g * H, H)
        for h in cutlass.range(h_lo, h_hi - 1, 2):
            # tile A: (g, h) from the C set; prefetch (g, h + 1) into N
            nq, nk, nv, npq, npk, nrq, nrk = _load_tile(
                h + 1, bb, l, rg, H, c, D, Dh, in_stride_b, in_stride_b_h, stride_qkv_l,
                gqp, gkp, gvp32, qkvp, rqp, rkp)
            # weights re-read from smem per tile; volatile keeps the loads inside the loop
            wq = (swp + wq_off).load(32, count=CHUNK, is_volatile=True)
            wk = (swp + wk_off).load(32, count=CHUNK, is_volatile=True)
            oq_w, ok_w, acc_q, acc_k = _tile_math(cq, ck, cpq, cpk, crq, crk, cs_c, cs_s,
                                                  wq, wk, lane, acc_q, acc_k)
            vq = cutlass.Vector.from_elements(tuple(oq_w), I32)
            vk = cutlass.Vector.from_elements(tuple(ok_w), I32)
            o_off = o_g + (h * (DH // 2)).to(I64)
            (outp32 + o_off).store(vq, alignment=16)
            (outp32 + (o_off + Dh.to(I64))).store(vk, alignment=16)
            (outp32 + (o_off + (2 * Dh).to(I64))).store(cv, alignment=16)
            rmax_packed = _rowmax(rmax_packed, oq_w, ok_w, cv)
            # Deferred consume of tile k-1. Every arrival on k-1 followed that warp's
            # consume of k-2 (which read buf[k&1]), so buf[k&1] may be overwritten now.
            if k > 0:
                pkA = k - 1
                cute.arch.mbarrier_wait(mbar + (pkA & 1), (pkA >> 1) & 1)
                cmax, csA = _consume_math(stage_base + (pkA & 1) * STAGE_BYTES, slice_off,
                                          bfrag, ones, cmax)
                if tq == 0:
                    _rmw_dbias(s_db, prev_off, col_base, csA)
            buf32 = stg32 + ((k & 1) * (STAGE_ELEMS // 2) + row_off32)
            buf32.store(vq, alignment=16)
            (buf32 + (ROWS * PITCH // 2)).store(vk, alignment=16)
            (buf32 + (2 * ROWS * PITCH // 2)).store(cv, alignment=16)
            cute.arch.mbarrier_arrive(mbar + (k & 1))
            prev_off = h * DH
            k = k + 1
            # tile B: (g, h + 1) from the N set; prefetch (g, h + 2) or wrap into C
            hB = h + 1
            h_s, bb_s, l_s, rg_s = _wrap_select(hB + 1, h_hi, hB, last_g, bb, l, rg, bb_n, l_n)
            cq, ck, cv, cpq, cpk, crq, crk = _load_tile(
                h_s, bb_s, l_s, rg_s, H, c, D, Dh, in_stride_b, in_stride_b_h, stride_qkv_l,
                gqp, gkp, gvp32, qkvp, rqp, rkp)
            wq = (swp + wq_off).load(32, count=CHUNK, is_volatile=True)
            wk = (swp + wk_off).load(32, count=CHUNK, is_volatile=True)
            oq_w, ok_w, acc_q, acc_k = _tile_math(nq, nk, npq, npk, nrq, nrk, cs_c, cs_s,
                                                  wq, wk, lane, acc_q, acc_k)
            vq = cutlass.Vector.from_elements(tuple(oq_w), I32)
            vk = cutlass.Vector.from_elements(tuple(ok_w), I32)
            o_off = o_g + (hB * (DH // 2)).to(I64)
            (outp32 + o_off).store(vq, alignment=16)
            (outp32 + (o_off + Dh.to(I64))).store(vk, alignment=16)
            (outp32 + (o_off + (2 * Dh).to(I64))).store(nv, alignment=16)
            rmax_packed = _rowmax(rmax_packed, oq_w, ok_w, nv)
            pkB = k - 1
            cute.arch.mbarrier_wait(mbar + (pkB & 1), (pkB >> 1) & 1)
            cmax, csB = _consume_math(stage_base + (pkB & 1) * STAGE_BYTES, slice_off,
                                      bfrag, ones, cmax)
            if tq == 0:
                _rmw_dbias(s_db, prev_off, col_base, csB)
            buf32 = stg32 + ((k & 1) * (STAGE_ELEMS // 2) + row_off32)
            buf32.store(vq, alignment=16)
            (buf32 + (ROWS * PITCH // 2)).store(vk, alignment=16)
            (buf32 + (2 * ROWS * PITCH // 2)).store(nv, alignment=16)
            cute.arch.mbarrier_arrive(mbar + (k & 1))
            prev_off = hB * DH
            k = k + 1
        # tail: an odd head count leaves (g, h_hi - 1) in the C set
        if ((h_hi - h_lo) & 1) == 1:
            hT = h_hi - 1
            h_sT, bb_sT, l_sT, rg_sT = _wrap_select(h_hi, h_hi, hT, last_g, bb, l, rg, bb_n, l_n)
            zq, zk, zv, zpq, zpk, zrq, zrk = _load_tile(
                h_sT, bb_sT, l_sT, rg_sT, H, c, D, Dh, in_stride_b, in_stride_b_h, stride_qkv_l,
                gqp, gkp, gvp32, qkvp, rqp, rkp)
            wqT = (swp + wq_off).load(32, count=CHUNK, is_volatile=True)
            wkT = (swp + wk_off).load(32, count=CHUNK, is_volatile=True)
            oq_wT, ok_wT, acc_q, acc_k = _tile_math(cq, ck, cpq, cpk, crq, crk, cs_c, cs_s,
                                                    wqT, wkT, lane, acc_q, acc_k)
            vqT = cutlass.Vector.from_elements(tuple(oq_wT), I32)
            vkT = cutlass.Vector.from_elements(tuple(ok_wT), I32)
            o_offT = o_g + (hT * (DH // 2)).to(I64)
            (outp32 + o_offT).store(vqT, alignment=16)
            (outp32 + (o_offT + Dh.to(I64))).store(vkT, alignment=16)
            (outp32 + (o_offT + (2 * Dh).to(I64))).store(cv, alignment=16)
            rmax_packed = _rowmax(rmax_packed, oq_wT, ok_wT, cv)
            if k > 0:
                pkT = k - 1
                cute.arch.mbarrier_wait(mbar + (pkT & 1), (pkT >> 1) & 1)
                cmax, csT = _consume_math(stage_base + (pkT & 1) * STAGE_BYTES, slice_off,
                                          bfrag, ones, cmax)
                if tq == 0:
                    _rmw_dbias(s_db, prev_off, col_base, csT)
            buf32T = stg32 + ((k & 1) * (STAGE_ELEMS // 2) + row_off32)
            buf32T.store(vqT, alignment=16)
            (buf32T + (ROWS * PITCH // 2)).store(vkT, alignment=16)
            (buf32T + (2 * ROWS * PITCH // 2)).store(cv, alignment=16)
            cute.arch.mbarrier_arrive(mbar + (k & 1))
            prev_off = hT * DH
            k = k + 1
            cq = zq
            ck = zk
            cv = zv
            cpq = zpq
            cpk = zpk
            crq = zrq
            crk = zrk

    # the last staged tile
    if k > 0:
        pk2 = k - 1
        cute.arch.mbarrier_wait(mbar + (pk2 & 1), (pk2 >> 1) & 1)
        cmax, cs2 = _consume_math(stage_base + (pk2 & 1) * STAGE_BYTES, slice_off,
                                  bfrag, ones, cmax)
        if tq == 0:
            _rmw_dbias(s_db, prev_off, col_base, cs2)

    # CTA epilogue 1: norm-weight partials over the 16 rows (s_acc aliases stage)
    cute.arch.sync_threads()
    for e in cutlass.range_constexpr(CHUNK):
        (s_acc + (r * DH + c * CHUNK + e)).store(acc_q[e])
        (s_acc + (ROWS * DH + r * DH + c * CHUNK + e)).store(acc_k[e])
    cute.arch.sync_threads()
    tau = tid >> 7
    d = tid & (DH - 1)
    s = F32(0.0)
    for rr in cutlass.range_constexpr(ROWS):
        s = s + (s_acc + (tau * (ROWS * DH) + rr * DH + d)).load()
    # cute.arch.atomic_add (unused result lowers to RED); cute.arch.red is rejected by libNVVM.
    if tau == 0:
        cute.arch.atomic_add(grad_qn_w.iterator + d, s)
    else:
        cute.arch.atomic_add(grad_kn_w.iterator + d, s)

    # CTA epilogue 2: dbias partial slot (16 B vectors)
    wsd_cta = wsd + bidx * D3_STATIC
    for i in cutlass.range(tid, D3_STATIC // 4, THREADS):
        v = (s_db + 4 * i).load(16, count=4)
        (wsd_cta + 4 * i).store(v, alignment=16)

    # CTA epilogue 3: amax partial -> last-CTA completion
    rm = _warp_max(cute.arch.fmax(cute.math.abs(_bf16_lo(rmax_packed)),
                                  cute.math.abs(_bf16_hi(rmax_packed))))
    cm = _warp_max(cmax)
    if lane == 0:
        (amax_stage + 2 * warp).store(rm)
        (amax_stage + 2 * warp + 1).store(cm)
    cute.arch.sync_threads()
    if tid == 0:
        rr = F32(0.0)
        cc = F32(0.0)
        for w in cutlass.range_constexpr(WARPS):
            rr = cute.arch.fmax(rr, (amax_stage + 2 * w).load())
            cc = cute.arch.fmax(cc, (amax_stage + 2 * w + 1).load())
        ws_amax[2 * bidx] = rr
        ws_amax[2 * bidx + 1] = cc
        cute.arch.fence_acq_rel_gpu()
        prev = cute.arch.atomic_add(counter.iterator, I32(1))
        if prev == gdim - 1:
            s_flag.store(I32(1))
    cute.arch.sync_threads()
    last = s_flag.load()
    if last == 1:
        # last CTA: the acquire fence makes every CTA's ws_amax writes visible
        cute.arch.fence_acq_rel_gpu()
        rr = F32(0.0)
        cc = F32(0.0)
        for i in cutlass.range(tid, gdim, THREADS):
            rr = cute.arch.fmax(rr, ws_amax[2 * i])
            cc = cute.arch.fmax(cc, ws_amax[2 * i + 1])
        rr = _warp_max(rr)
        cc = _warp_max(cc)
        if lane == 0:
            (amax_stage + 2 * warp).store(rr)
            (amax_stage + 2 * warp + 1).store(cc)
        cute.arch.sync_threads()
        if tid == 0:
            r2 = F32(0.0)
            c2 = F32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                r2 = cute.arch.fmax(r2, (amax_stage + 2 * w).load())
                c2 = cute.arch.fmax(c2, (amax_stage + 2 * w + 1).load())
            amax[0] = r2                   # max of bf16 values: already bf16
            amax[1] = _round_bf16(c2)      # TE rounds to bf16 before the max
            counter[0] = I32(0)            # reset for the next launch


@cute.kernel
def rope_norm_join_bgrad_amax_ctm_fin_kernel(
    ws: cute.Tensor, dbias: cute.Tensor, slots: I32, n: I32,
) -> None:
    """Fixed-order dbias finaliser: ``ws[slots][n]`` fp32 -> ``dbias[n]`` bf16.

    A block owns 128 columns as 32 fp32x4 quads; 8 slot groups each sum every eighth slot,
    then combine through smem in group order. Requires ``n % 128 == 0``.
    """
    tid, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    q = tid & (FIN_QUADS - 1)
    grp = tid >> 5
    col = (bidx * FIN_QUADS + q) * 4
    smem = utils.SmemAllocator()
    red = cutlass.Pointer(smem.allocate_array(F32, FIN_GROUPS * FIN_QUADS * 4, byte_alignment=16), dtype=F32)
    wsp = cutlass.Pointer(ws.iterator, dtype=F32) + col
    s0 = F32(0.0)
    s1 = F32(0.0)
    s2 = F32(0.0)
    s3 = F32(0.0)
    # batches of four slots (k, k+8, k+16, k+24) while k + 24 < slots
    nb = cutlass.max((slots - (FIN_BATCH - 1) * FIN_GROUPS - grp + (FIN_BATCH * FIN_GROUPS - 1)) // (FIN_BATCH * FIN_GROUPS), 0)
    for kb in cutlass.range(0, nb):
        k0 = grp + kb * (FIN_BATCH * FIN_GROUPS)
        v0 = (wsp + k0 * n).load(16, count=4)
        v1 = (wsp + (k0 + FIN_GROUPS) * n).load(16, count=4)
        v2 = (wsp + (k0 + 2 * FIN_GROUPS) * n).load(16, count=4)
        v3 = (wsp + (k0 + 3 * FIN_GROUPS) * n).load(16, count=4)
        s0 = s0 + v0[0]
        s1 = s1 + v0[1]
        s2 = s2 + v0[2]
        s3 = s3 + v0[3]
        s0 = s0 + v1[0]
        s1 = s1 + v1[1]
        s2 = s2 + v1[2]
        s3 = s3 + v1[3]
        s0 = s0 + v2[0]
        s1 = s1 + v2[1]
        s2 = s2 + v2[2]
        s3 = s3 + v2[3]
        s0 = s0 + v3[0]
        s1 = s1 + v3[1]
        s2 = s2 + v3[2]
        s3 = s3 + v3[3]
    # remainder, one slot at a time
    for kk in cutlass.range(grp + nb * (FIN_BATCH * FIN_GROUPS), slots, FIN_GROUPS):
        v = (wsp + kk * n).load(16, count=4)
        s0 = s0 + v[0]
        s1 = s1 + v[1]
        s2 = s2 + v[2]
        s3 = s3 + v[3]
    rp = red + (grp * FIN_QUADS + q) * 4
    rp.store(cutlass.Vector.from_elements((s0, s1, s2, s3), F32), alignment=16)
    cute.arch.sync_threads()
    if grp == 0:
        t = (red + q * 4).load(16, count=4)
        t0 = t[0]
        t1 = t[1]
        t2 = t[2]
        t3 = t[3]
        for g in cutlass.range_constexpr(1, FIN_GROUPS):
            v = (red + (g * FIN_QUADS + q) * 4).load(16, count=4)
            t0 = t0 + v[0]
            t1 = t1 + v[1]
            t2 = t2 + v[2]
            t3 = t3 + v[3]
        dbp = cutlass.Pointer(dbias.iterator, dtype=BF16) + col
        dbp.store(cutlass.Vector.from_elements(
            (t0.to(BF16), t1.to(BF16), t2.to(BF16), t3.to(BF16)), BF16), alignment=8)


@cute.jit
def rope_norm_join_bgrad_amax_ctm_host(
    gq: cute.Tensor, gk: cute.Tensor, gv: cute.Tensor, qkv: cute.Tensor,
    q_rstd: cute.Tensor, k_rstd: cute.Tensor,
    qn_w: cute.Tensor, kn_w: cute.Tensor, freqs: cute.Tensor,
    out: cute.Tensor, grad_qn_w: cute.Tensor, grad_kn_w: cute.Tensor,
    rht: cute.Tensor, ws_dbias: cute.Tensor, ws_amax: cute.Tensor,
    amax: cute.Tensor, counter: cute.Tensor,
    L: I32, H: I32, in_stride_b: I64, stride_qkv_l: I64, freq_stride_s: I64,
    pos_offset: I32, n_tiles: I32, grid_x: I32, stream: cuda.CUstream,
) -> None:
    rope_norm_join_bgrad_amax_ctm_kernel(
        gq, gk, gv, qkv, q_rstd, k_rstd, qn_w, kn_w, freqs, out, grad_qn_w, grad_kn_w,
        rht, ws_dbias, ws_amax, amax, counter,
        L, H, in_stride_b, stride_qkv_l, freq_stride_s, pos_offset, n_tiles,
    ).launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream,
             min_blocks_per_mp=MIN_CTAS_PER_SM,
             preferred_smem_carveout=SMEM_CARVEOUT_PCT)


@cute.jit
def rope_norm_join_bgrad_amax_ctm_fin_host(
    ws: cute.Tensor, dbias: cute.Tensor, slots: I32, n: I32, grid_x: I32,
    stream: cuda.CUstream,
) -> None:
    rope_norm_join_bgrad_amax_ctm_fin_kernel(ws, dbias, slots, n).launch(
        grid=(grid_x, 1, 1), block=(FIN_THREADS, 1, 1), stream=stream)


# Host side
_compiled: dict[int, Callable] = {}
_compiled_fin: dict[int, Callable] = {}
_counters: dict[int, torch.Tensor] = {}
_partials: dict[tuple[int, int], torch.Tensor] = {}
_fin_streams: dict[int, torch.cuda.Stream] = {}
_fin_events: dict[int, torch.cuda.Event] = {}   # one outstanding fork per device


def _compile(device_index: int) -> Callable:
    """Compile once per device; every length and stride stays symbolic."""
    if device_index not in _compiled:
        with torch.cuda.device(device_index):
            def bf(align: int = 16):
                return make_fake_compact_tensor(BF16, (cute.sym_int64(),), assumed_align=align)

            def f32(align: int):
                return make_fake_compact_tensor(F32, (cute.sym_int64(),), assumed_align=align)

            fakes = [bf(), bf(), bf(), bf(),        # gq gk gv qkv
                     f32(4), f32(4),                # q_rstd k_rstd
                     bf(2), bf(2),                  # qn_w kn_w (scalar reads)
                     f32(16),                       # freqs (float4 reads)
                     bf(),                          # out
                     f32(4), f32(4),                # grad_qn_w grad_kn_w
                     make_fake_compact_tensor(I16, (256,), assumed_align=16),   # rht as Int16
                     f32(16), f32(8),               # ws_dbias ws_amax
                     make_fake_compact_tensor(F32, (2,), assumed_align=8),      # amax
                     make_fake_compact_tensor(I32, (2,), assumed_align=8)]      # counter
            ints = [I32(0), I32(0), I64(0), I64(0), I64(0), I32(0), I32(0), I32(0)]
            _compiled[device_index] = cute.compile(
                rope_norm_join_bgrad_amax_ctm_host, *fakes, *ints, make_fake_stream(),
                options="--enable-tvm-ffi")
    return _compiled[device_index]


def _compile_fin(device_index: int) -> Callable:
    if device_index not in _compiled_fin:
        with torch.cuda.device(device_index):
            f_ws = make_fake_compact_tensor(F32, (cute.sym_int64(),), assumed_align=16)
            f_db = make_fake_compact_tensor(BF16, (cute.sym_int64(),), assumed_align=16)
            _compiled_fin[device_index] = cute.compile(
                rope_norm_join_bgrad_amax_ctm_fin_host, f_ws, f_db, I32(0), I32(0), I32(0),
                make_fake_stream(), options="--enable-tvm-ffi")
    return _compiled_fin[device_index]


def max_ctas(device_index: int = 0) -> int:
    """Largest persistent grid the kernel launches: ``SMs * MIN_CTAS_PER_SM``."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count * MIN_CTAS_PER_SM


def grid_for(B: int, L: int, H: int, device_index: int = 0) -> int:
    """``min(n_tiles, max_ctas())``; env ``ROPE_BGRAD_CTM_MAX_CTAS`` caps it (tests only)."""
    n_tiles = B * L // ROWS * H
    cap = int(os.environ.get("ROPE_BGRAD_CTM_MAX_CTAS", "0")) or max_ctas(device_index)
    return min(n_tiles, cap)


@functools.lru_cache(maxsize=None)
def fused_above_numel(device_index: int) -> int:
    """Elements of the cast view above which the fused op is used: L2 bytes / itemsize."""
    override = os.environ.get(FUSE_ABOVE_ENV)
    if override:
        return int(override)
    return torch.cuda.get_device_properties(device_index).L2_cache_size // GRAD_ITEMSIZE


def fused(numel: int, device_index: int | None = None) -> bool:
    """Whether a cast view of ``numel`` elements takes the fused op."""
    if device_index is None:
        device_index = torch.cuda.current_device()
    return numel > fused_above_numel(device_index)


def _flat(t: torch.Tensor) -> torch.Tensor:
    """1-D stride-1 view over the span this tensor addresses (strided slices too)."""
    if t.is_contiguous():
        return t.view(-1)
    span = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()) if s > 0)
    return t.as_strided((span,), (1,))


def _counter_for(device: torch.device) -> torch.Tensor:
    c = _counters.get(device.index)
    if c is None:
        c = torch.zeros(2, dtype=torch.int32, device=device)
        _counters[device.index] = c
    return c


def _partials_for(device: torch.device, slots: int) -> torch.Tensor:
    key = (device.index, slots)
    buf = _partials.get(key)
    if buf is None:
        buf = torch.empty(slots * D3_STATIC, dtype=torch.float32, device=device)
        _partials[key] = buf
    return buf


def _fin_stream_for(device: torch.device) -> torch.cuda.Stream:
    s = _fin_streams.get(device.index)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _fin_streams[device.index] = s
    return s


def _check(grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd,
           qn_weight, kn_weight, grad_qn_weight, grad_kn_weight, freqs, pos_offset, rht):
    B, L, D = grad_q_rope.shape
    if D != H * DH:
        raise ValueError("rope_norm_join_bgrad_amax_ctm: head dim must be 128")
    if L % ROWS:
        raise ValueError(f"rope_norm_join_bgrad_amax_ctm requires L % 16 == 0, got L={L}")
    if B * L * 3 * D >= (1 << 31):
        raise ValueError("rope_norm_join_bgrad_amax_ctm: M x 3D must fit int32")
    if 3 * D != D3_STATIC:
        raise ValueError(f"rope_norm_join_bgrad_amax_ctm: the smem dbias accumulator is sized for 3D = {D3_STATIC}")
    for t in (grad_q_rope, grad_k_rope, grad_v):
        if t.dtype != torch.bfloat16 or t.shape != grad_q_rope.shape:
            raise ValueError("rope_norm_join_bgrad_amax_ctm: grad views must be bf16 [B, L, D]")
        if t.stride(2) != 1 or t.stride(1) != D or t.stride(0) != grad_q_rope.stride(0):
            raise ValueError("rope_norm_join_bgrad_amax_ctm: grad views must be row-contiguous "
                             "[B, L, D] slices of one buffer")
        if t.data_ptr() % 16:
            raise ValueError("rope_norm_join_bgrad_amax_ctm: grad views must be 16-byte aligned")
    if grad_q_rope.stride(0) % CHUNK:
        raise ValueError("rope_norm_join_bgrad_amax_ctm: batch stride must be a multiple of 8")
    if (qkv.dtype != torch.bfloat16 or not qkv.is_contiguous()
            or tuple(qkv.shape) != (B, L, 3 * D)):
        raise ValueError("rope_norm_join_bgrad_amax_ctm: qkv must be contiguous bf16 [B, L, 3D]")
    for t in (q_rstd, k_rstd):
        if t.dtype != torch.float32 or not t.is_contiguous() or t.numel() != B * L * H:
            raise ValueError("rope_norm_join_bgrad_amax_ctm: rstd must be contiguous fp32 [B*L, H]")
    for t in (qn_weight, kn_weight):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.numel() != DH:
            raise ValueError("rope_norm_join_bgrad_amax_ctm: norm weights must be contiguous bf16 [128]")
    for t in (grad_qn_weight, grad_kn_weight):
        if t.dtype != torch.float32 or not t.is_contiguous() or t.numel() != DH:
            raise ValueError("rope_norm_join_bgrad_amax_ctm: norm-weight grads must be contiguous fp32 [128]")
    if (freqs.dtype != torch.float32 or freqs.stride(-1) != 1 or freqs.shape[-1] != DH
            or freqs.stride(0) % 4 or freqs.shape[0] < pos_offset + L
            or freqs.data_ptr() % 16):
        raise ValueError("rope_norm_join_bgrad_amax_ctm: freqs must be fp32 [S >= pos_offset + L, ..., 128], "
                         "row stride a multiple of 4, 16-byte aligned")
    if rht.dtype != torch.bfloat16 or not rht.is_contiguous() or tuple(rht.shape) != (16, 16):
        raise ValueError("rope_norm_join_bgrad_amax_ctm: rht_matrix must be a contiguous bf16 [16, 16]")
    return B, L, D


def _launch(args: tuple, rht: torch.Tensor, out: torch.Tensor, grid: int,
            ws_dbias: torch.Tensor, ws_amax: torch.Tensor, amax: torch.Tensor,
            counter: torch.Tensor) -> None:
    (grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd, qn_weight, kn_weight,
     grad_qn_weight, grad_kn_weight, freqs, pos_offset) = args
    B, L, D = grad_q_rope.shape
    dev = grad_q_rope.device.index
    _compile(dev)(
        _flat(grad_q_rope), _flat(grad_k_rope), _flat(grad_v), qkv.view(-1),
        q_rstd.view(-1), k_rstd.view(-1), qn_weight.view(-1), kn_weight.view(-1),
        _flat(freqs), out.view(-1), grad_qn_weight.view(-1), grad_kn_weight.view(-1),
        rht.view(torch.int16).view(-1), ws_dbias, ws_amax, amax, counter,
        L, H, grad_q_rope.stride(0), qkv.stride(1), freqs.stride(0), pos_offset,
        B * L // ROWS * H, grid, cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream),
    )


def finalize(ws_dbias: torch.Tensor, grad_b: torch.Tensor, slots: int) -> None:
    """Dbias finaliser on the current stream: ``ws_dbias[slots][n]`` -> bf16 ``grad_b[n]``."""
    n = grad_b.numel()
    if grad_b.dtype != torch.bfloat16 or not grad_b.is_contiguous() or n % FIN_COLS:
        raise ValueError("rope_norm_join_bgrad_amax_ctm: dbias must be contiguous bf16 with n % 128 == 0")
    if ws_dbias.dtype != torch.float32 or ws_dbias.numel() < slots * n:
        raise ValueError("rope_norm_join_bgrad_amax_ctm: ws_dbias must be fp32 [slots][n]")
    dev = grad_b.device.index
    _compile_fin(dev)(ws_dbias, grad_b, slots, n, n // FIN_COLS,
                      cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream))


def _produce(args: tuple, rht: torch.Tensor, ws_dbias: torch.Tensor,
             ) -> tuple[torch.Tensor, torch.Tensor, int]:
    (grad_q_rope, *_rest) = args
    dev = grad_q_rope.device
    B, L, D = grad_q_rope.shape
    out = torch.empty(B, L, 3 * D, dtype=torch.bfloat16, device=dev)
    grid = grid_for(B, L, args[4], dev.index)
    ws_amax = torch.empty(grid * 2, dtype=torch.float32, device=dev)
    amax = torch.empty(2, dtype=torch.float32, device=dev)
    _launch(args, rht, out, grid, ws_dbias, ws_amax, amax, _counter_for(dev))
    return out, amax, grid


def rope_norm_join_bgrad_amax_ctm(
    grad_q_rope: torch.Tensor, grad_k_rope: torch.Tensor, grad_v: torch.Tensor,
    qkv: torch.Tensor, H: int, q_rstd: torch.Tensor, k_rstd: torch.Tensor,
    qn_weight: torch.Tensor, kn_weight: torch.Tensor,
    grad_qn_weight: torch.Tensor, grad_kn_weight: torch.Tensor,
    freqs: torch.Tensor, pos_offset: int, rht_matrix: torch.Tensor,
    *, finalize_dbias: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RoPE bwd + QK norm bwd + QKV join, also returning ``grad_b`` and the amax pair.

    Arguments are ``flux.rope_norm_join_ctm``'s plus the bf16 [16, 16] ``rht_matrix``; H must be 24.
    Returns ``(grad_qkv, grad_b, amax)``: bf16 [B, L, 3D], bf16 [3D] fixed-order column sum, and
    fp32 [2] = (max|x|, post-RHT amax). With ``finalize_dbias=False`` ``grad_b`` is undefined.
    """
    args = (grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd, qn_weight, kn_weight,
            grad_qn_weight, grad_kn_weight, freqs, pos_offset)
    B, L, D = _check(*args, rht_matrix)
    dev = grad_q_rope.device
    ws_dbias = torch.empty(grid_for(B, L, H, dev.index) * 3 * D, dtype=torch.float32, device=dev)
    out, amax, grid = _produce(args, rht_matrix, ws_dbias)
    grad_b = torch.empty(3 * D, dtype=torch.bfloat16, device=dev)
    if finalize_dbias:
        finalize(ws_dbias, grad_b, grid)
    return out, grad_b, amax


def rope_norm_join_bgrad_amax_ctm_forked(
    grad_q_rope: torch.Tensor, grad_k_rope: torch.Tensor, grad_v: torch.Tensor,
    qkv: torch.Tensor, H: int, q_rstd: torch.Tensor, k_rstd: torch.Tensor,
    qn_weight: torch.Tensor, kn_weight: torch.Tensor,
    grad_qn_weight: torch.Tensor, grad_kn_weight: torch.Tensor,
    freqs: torch.Tensor, pos_offset: int, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`rope_norm_join_bgrad_amax_ctm` with the finaliser on a per-device side stream.

    Call :func:`join_dbias` before reading ``grad_b`` and before the next call into this op.
    ``grad_b`` is bitwise the inline path's.
    """
    args = (grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd, qn_weight, kn_weight,
            grad_qn_weight, grad_kn_weight, freqs, pos_offset)
    B, L, D = _check(*args, rht_matrix)
    dev = grad_q_rope.device
    ws_dbias = _partials_for(dev, grid_for(B, L, H, dev.index))
    out, amax, grid = _produce(args, rht_matrix, ws_dbias)
    grad_b = torch.empty(3 * D, dtype=torch.bfloat16, device=dev)
    fin = _fin_stream_for(dev)
    fin.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(fin):
        finalize(ws_dbias, grad_b, grid)
        ev = torch.cuda.Event()
        ev.record(fin)
    _fin_events[dev.index] = ev
    grad_b.record_stream(fin)
    return out, grad_b, amax


def join_dbias(device: torch.device | None = None) -> None:
    """Make the current stream wait for the deferred finaliser, if one is pending.

    Waits on the event recorded by the last forked call (not on the finaliser stream) and
    clears it; a no-op when no fork is outstanding.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    ev = _fin_events.pop(device.index, None)
    if ev is not None:
        torch.cuda.current_stream(device).wait_event(ev)


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "rope_norm_join_bgrad_amax_ctm(Tensor grad_q_rope, Tensor grad_k_rope,"
    " Tensor grad_v, Tensor qkv, int H, Tensor q_rstd, Tensor k_rstd,"
    " Tensor qn_weight, Tensor kn_weight, Tensor grad_qn_weight, Tensor grad_kn_weight,"
    " Tensor freqs, int pos_offset, Tensor rht_matrix) -> (Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "rope_norm_join_bgrad_amax_ctm", "CUDA")
def _rope_norm_join_bgrad_amax_ctm_impl(
    grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd, qn_weight, kn_weight,
    grad_qn_weight, grad_kn_weight, freqs, pos_offset, rht_matrix,
):
    return rope_norm_join_bgrad_amax_ctm(
        grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd, qn_weight, kn_weight,
        grad_qn_weight, grad_kn_weight, freqs, pos_offset, rht_matrix,
    )
