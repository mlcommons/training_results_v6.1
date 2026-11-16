# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""torch.ops.flux.dgelu_dbias_amax_ctm: fused dGELU + dbias + amax on the CUTLASS DSL."""

from __future__ import annotations

import os
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T

WARPS = 8
THREADS = 32 * WARPS
WARP_COLS = 64
CTA_COLS = WARPS * WARP_COLS          # 512
GROUP_ROWS = 16
PITCH_BYTES = 1040                    # 1024 B + 16 B pad, conflict-free ldmatrix
TILE_BYTES = GROUP_ROWS * PITCH_BYTES  # 16640; two buffers
MIN_BLOCKS_PER_SM = 2                 # __launch_bounds__(256, 2)
SMEM_CARVEOUT_PCT = 100
#: name -> (min_blocks_per_mp, preferred_smem_carveout).
LAUNCH_VARIANTS: dict[str, tuple[int, int]] = {
    "A": (2, SMEM_CARVEOUT_PCT),
    "C3": (3, SMEM_CARVEOUT_PCT),
}
DEFAULT_VARIANT = "C3"
_ONES_BF16X2 = 0x3F803F80             # bf16x2 {1.0, 1.0}
_BF16_HI_MASK = -65536                # 0xFFFF0000 as a signed Int32
_INT32_MAX = 2**31 - 1

# aten GeluBackward constants folded to fp32 as nvcc does:
# kBeta = sqrt(2/pi), kKappa = 0.044715, 3 * kKappa
_BETA = float.fromhex("0x1.988454p-1")
_KAPPA = float.fromhex("0x1.6e4e26p-5")
_3KAPPA = float.fromhex("0x1.12ba9cp-3")

_CVT_BF16X2 = "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};"
_MAX_XORSIGN_ABS_BF16X2 = "max.xorsign.abs.bf16x2 {$w0}, {$r0}, {$r1};"
_LDMATRIX_X4_TRANS = (
    "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
    "{ {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];"
)
_LD_CG_V2_F32 = "ld.global.cg.v2.f32 { {$w0}, {$w1} }, [{$r0}];"
# Raw llvm.inline_asm with explicit constraints: inline_ptx lets LLVM fold
# constant-zero C operands into immediates, which ptxas rejects for mma.
_MMA_16816_BF16 = (
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
)
_MMA_CONSTRAINTS = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"


def _dgelu_f32(dy, x, fast: bool, fma_left: bool):
    """aten's tanh-GELU backward on fp32 scalars, FMAs placed as nvcc contracts them."""
    beta = cutlass.Float32(_BETA)
    x_sq = x * x
    x_cube = x_sq * x
    inner = beta * cute.math.fma(x_cube, cutlass.Float32(_KAPPA), x)
    if fast:
        t = cute.math.tanh(inner, approx=True)
    else:
        t = cute.math.tanh(inner)
    left = x * cutlass.Float32(0.5)
    right = t + cutlass.Float32(1.0)
    tanh_d = cutlass.Float32(1.0) - t * t        # nvcc: mul.f32 + sub.f32, NOT an fma
    inner_d = beta * cute.math.fma(x_sq, cutlass.Float32(_3KAPPA), cutlass.Float32(1.0))
    if fma_left:
        s = cute.math.fma(right, cutlass.Float32(0.5), (left * tanh_d) * inner_d)
    else:
        s = cute.math.fma(left * tanh_d, inner_d, right * cutlass.Float32(0.5))
    return dy * s


def _bf16_lo(w):
    return (w << 16).bitcast(cutlass.Float32)


def _bf16_hi(w):
    return (w & cutlass.Int32(_BF16_HI_MASK)).bitcast(cutlass.Float32)


def _pack_bf16x2_rn(lo, hi):
    return cute.arch.inline_ptx(
        _CVT_BF16X2, write_only_types=[cutlass.Int32], read_only_args=[hi, lo],
    )


def _max_xorsign_abs(a, b):
    return cute.arch.inline_ptx(
        _MAX_XORSIGN_ABS_BF16X2, write_only_types=[cutlass.Int32], read_only_args=[a, b],
    )


def _ldmatrix_x4_trans(addr):
    return cute.arch.inline_ptx(
        _LDMATRIX_X4_TRANS, write_only_types=[cutlass.Int32] * 4, read_only_args=[addr],
    )


def _ld_cg_v2(addr64):
    """__ldcg of a float2: the L2-coherent read of another CTA's partial."""
    return cute.arch.inline_ptx(
        _LD_CG_V2_F32, write_only_types=[cutlass.Float32] * 2, read_only_args=[addr64],
    )


def _mma_16816(a, b0, b1, c):
    """D = A(16x16 bf16, row) * B(16x8 bf16, col) + C, fp32 accumulate."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [v.ir_value() for v in (a[0], a[1], a[2], a[3], b0, b1, c[0], c[1], c[2], c[3])],
        _MMA_16816_BF16, _MMA_CONSTRAINTS,
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(cutlass.Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(4))


def _round_bf16(x):
    return x.to(cutlass.BFloat16).to(cutlass.Float32)


def _warp_max(v):
    for off in (16, 8, 4, 2, 1):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=off))
    return v


def _cta_of(t, gdim, tasks):
    """The CTA whose contiguous task range contains task t."""
    return ((t + 1) * gdim + tasks - 1) // tasks - 1


def _zeros(n: int):
    return [cutlass.Float32(0.0) for _ in range(n)]


def _load_slabs(ptr, rgi, cb, warp, lane, n_w):
    """The four 16 B slabs of one warp task from ptr (stream mapping), as Int32x4 vectors."""
    vs = []
    for s in range(4):
        row = rgi * GROUP_ROWS + 2 * warp + (s >> 1)
        idx = row * n_w + cb * (CTA_COLS // 2) + (lane + 32 * (s & 1)) * 4
        vs.append((ptr + idx).load(16, count=4, is_invariant=True))
    return vs


def _dgelu_words(gw, pw, fast: bool, fma_left: bool):
    """Four packed bf16x2 output words from four grad and four pre words."""
    outs = []
    for k in range(4):
        lo = _dgelu_f32(_bf16_lo(gw[k]), _bf16_lo(pw[k]), fast, fma_left)
        hi = _dgelu_f32(_bf16_hi(gw[k]), _bf16_hi(pw[k]), fast, fma_left)
        outs.append(_pack_bf16x2_rn(lo, hi))
    return outs


@cute.kernel
def dgelu_dbias_amax_ctm_kernel(
    grad: cute.Tensor, pre: cute.Tensor, out: cute.Tensor,
    dbias: cute.Tensor, amax: cute.Tensor, group_amax: cute.Tensor,
    ws_dbias: cute.Tensor, ws_amax: cute.Tensor, counters: cute.Tensor, rht: cute.Tensor,
    n_w: cutlass.Int32, rg: cutlass.Int32, n_cb: cutlass.Int32, p: cutlass.Int32,
    tasks: cutlass.Int32,
    FAST: cutlass.Constexpr[bool], RHT: cutlass.Constexpr[bool],
    DBIAS: cutlass.Constexpr[bool], PREFETCH: cutlass.Constexpr[bool],
    GROUP: cutlass.Constexpr[bool], FMA_LEFT: cutlass.Constexpr[bool],
) -> None:
    """grad/pre/out: bf16 [M, N] viewed as flat Int32 words; n_w = N / 2 words per row.

    rg = M / 16 row groups, n_cb = N / 512 column blocks, p partial slots per column
    block, tasks = rg * n_cb; rht is the bf16 [16, 16] basis viewed as Int16.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    gi = lane >> 2          # mma "groupID"
    tq = lane & 3           # mma "threadID_in_group"

    gp = cutlass.Pointer(grad.iterator, dtype=cutlass.Int32)
    pp = cutlass.Pointer(pre.iterator, dtype=cutlass.Int32)
    op = cutlass.Pointer(out.iterator, dtype=cutlass.Int32)
    wsd = cutlass.Pointer(ws_dbias.iterator, dtype=cutlass.Float32)
    wsa = cutlass.Pointer(ws_amax.iterator, dtype=cutlass.Float32)
    cnt = cutlass.Pointer(counters.iterator, dtype=cutlass.Int32)
    dbp = cutlass.Pointer(dbias.iterator, dtype=cutlass.BFloat16)

    tile = cute.arch.alloc_smem(cutlass.Int32, 2 * TILE_BYTES // 4, 16)
    tile_p = cutlass.Pointer(tile, dtype=cutlass.Int32)
    tile_base = tile.toint()                     # Int32 shared-window address
    stage = cutlass.Pointer(cute.arch.alloc_smem(cutlass.Float32, 2 * WARPS, 16),
                            dtype=cutlass.Float32)
    flag = cutlass.Pointer(cute.arch.alloc_smem(cutlass.Int32, 1, 4), dtype=cutlass.Int32)

    # B fragments of R: lane holds k = 2tq, 2tq+1 (+8) at n = gi (+8 jb).
    bfrag = []
    for jb in cutlass.range_constexpr(2):
        nn = jb * 8 + gi
        r0 = rht[(2 * tq) * 16 + nn].to(cutlass.Int32) & 0xFFFF
        r1 = rht[(2 * tq + 1) * 16 + nn].to(cutlass.Int32) & 0xFFFF
        r8 = rht[(2 * tq + 8) * 16 + nn].to(cutlass.Int32) & 0xFFFF
        r9 = rht[(2 * tq + 9) * 16 + nn].to(cutlass.Int32) & 0xFFFF
        bfrag.append((r0 | (r1 << 16), r8 | (r9 << 16)))
    ones = cutlass.Int32(_ONES_BF16X2)

    # ldmatrix.x4 lane address inside this warp's 16 x 64 slice of the CTA tile:
    # matrix q = lane/8 supplies row (q>>1)*8 + lane%8 at column offset (q&1)*8.
    ldq = lane >> 3
    ld_off = (warp * (WARP_COLS * 2) + ((ldq >> 1) * 8 + (lane & 7)) * PITCH_BYTES
              + (ldq & 1) * 16)
    # st.shared word offset of this lane's vector for slab 0 of buffer 0 (stream
    # mapping: row 2w + (s>>1), column vector lane + 32(s&1)).
    st_word0 = ((2 * warp) * PITCH_BYTES + lane * 16) // 4

    # CTA b runs tasks [start(b), start(b+1)), start(b) = floor(b * T / grid).
    # Products stay below 2**31 (host-checked).
    t_begin = (bidx * tasks) // gdim
    t_end = ((bidx + 1) * tasks) // gdim

    zero4 = (cutlass.Float32(0.0),) * 4
    cacc = _zeros(16)   # dbias C fragments, 4 blocks x 4
    rmax_packed = cutlass.Int32(0)                      # rowwise amax, packed bf16x2
    cmax = cutlass.Float32(0.0)                         # post-RHT amax, fp32

    # grid <= tasks (host-checked): every CTA has at least one task
    cur_cb = t_begin // rg
    # loaded before the loop so the loop-carried lists have one type
    gv = _load_slabs(gp, t_begin - cur_cb * rg, cur_cb, warp, lane, n_w)
    pv = _load_slabs(pp, t_begin - cur_cb * rg, cur_cb, warp, lane, n_w)

    for t in cutlass.range(t_begin, t_end):
        cb = t // rg
        if cb != cur_cb:
            # flush the dbias partial for cur_cb (uniform across the CTA)
            slot = bidx - _cta_of(cur_cb * rg, gdim, tasks)
            if cutlass.const_expr(DBIAS):
                if tq == 0:
                    part = wsd + (cur_cb * p + slot) * CTA_COLS + warp * WARP_COLS
                    for blk in cutlass.range_constexpr(4):
                        (part + blk * 16 + gi).store(cacc[4 * blk + 0])
                        (part + blk * 16 + gi + 8).store(cacc[4 * blk + 2])
            cacc = _zeros(16)
            cute.arch.fence_acq_rel_gpu()
            cute.arch.sync_threads()
            n_touch = _cta_of(cur_cb * rg + rg - 1, gdim, tasks) - _cta_of(cur_cb * rg, gdim, tasks) + 1
            if tidx == 0:
                prev = cute.arch.atomic_add(cnt + cur_cb, cutlass.Int32(1))
                flag.store(prev - (n_touch - 1))
            cute.arch.sync_threads()
            if flag.load() == 0:
                # last CTA for this column block: fixed-order sum over slots 0..n_touch-1
                cute.arch.fence_acq_rel_gpu()
                s0 = cutlass.Float32(0.0)
                s1 = cutlass.Float32(0.0)
                if cutlass.const_expr(DBIAS):
                    base = wsd + cur_cb * p * CTA_COLS + 2 * tidx
                    for k in cutlass.range(n_touch):
                        vv = _ld_cg_v2((base + k * CTA_COLS).toint())
                        s0 = s0 + vv[0]
                        s1 = s1 + vv[1]
                (dbp + cur_cb * CTA_COLS + 2 * tidx).store(
                    cutlass.Vector.from_elements(
                        (s0.to(cutlass.BFloat16), s1.to(cutlass.BFloat16)), cutlass.BFloat16),
                    alignment=4)
                if tidx == 0:
                    (cnt + cur_cb).store(cutlass.Int32(0))   # self-reset for the next launch
            cur_cb = cb

        rgi = t - cb * rg
        buf = t & 1
        if cutlass.const_expr(not PREFETCH):
            gv = _load_slabs(gp, rgi, cb, warp, lane, n_w)
            pv = _load_slabs(pp, rgi, cb, warp, lane, n_w)

        # dGELU, pack, store (global + smem), rowwise amax
        tmax = cutlass.Float32(0.0)
        for s in cutlass.range_constexpr(4):
            gw = gv[s]
            pw = pv[s]
            o = _dgelu_words((gw[0], gw[1], gw[2], gw[3]), (pw[0], pw[1], pw[2], pw[3]),
                             FAST, FMA_LEFT)
            for k in cutlass.range_constexpr(4):
                rmax_packed = _max_xorsign_abs(rmax_packed, o[k])
            ovec = cutlass.Vector.from_elements((o[0], o[1], o[2], o[3]), cutlass.Int32)
            row = rgi * GROUP_ROWS + 2 * warp + (s >> 1)
            idx = row * n_w + cb * (CTA_COLS // 2) + (lane + 32 * (s & 1)) * 4
            (op + idx).store(ovec, alignment=16)
            if cutlass.const_expr(RHT or DBIAS):
                sw = (st_word0 + buf * (TILE_BYTES // 4)
                      + ((s >> 1) * PITCH_BYTES + (s & 1) * 512) // 4)
                (tile_p + sw).store(ovec, alignment=16)

        if cutlass.const_expr(PREFETCH):
            # prefetch the next task before the epilogue; on the last task this
            # re-reads the current task (clamped, branch-free)
            tn = t + 1 - (t + 1) // t_end
            ncb = tn // rg
            gv = _load_slabs(gp, tn - ncb * rg, ncb, warp, lane, n_w)
            pv = _load_slabs(pp, tn - ncb * rg, ncb, warp, lane, n_w)

        if cutlass.const_expr(RHT or DBIAS):
            # the tile was written by all eight warps; with double buffering one barrier
            # per task suffices (buffer t&1 was last read before the previous barrier)
            cute.arch.sync_threads()
            ld_base = tile_base + ld_off + buf * TILE_BYTES
            new_cacc = []
            for blk in cutlass.range_constexpr(4):
                a = _ldmatrix_x4_trans(ld_base + blk * 32)
                if cutlass.const_expr(RHT):
                    d0 = _mma_16816(a, bfrag[0][0], bfrag[0][1], zero4)
                    d1 = _mma_16816(a, bfrag[1][0], bfrag[1][1], zero4)
                    for j in cutlass.range_constexpr(4):
                        tmax = cute.arch.fmax(tmax, cute.math.abs(d0[j]))
                        tmax = cute.arch.fmax(tmax, cute.math.abs(d1[j]))
                if cutlass.const_expr(DBIAS):
                    c = _mma_16816(a, ones, ones,
                                   (cacc[4 * blk], cacc[4 * blk + 1],
                                    cacc[4 * blk + 2], cacc[4 * blk + 3]))
                    new_cacc.extend(c)
            if cutlass.const_expr(DBIAS):
                cacc = new_cacc
        cmax = cute.arch.fmax(cmax, tmax)
        if cutlass.const_expr(GROUP):
            wmax = _warp_max(tmax)
            if lane == 0:
                gap = cutlass.Pointer(group_amax.iterator, dtype=cutlass.Float32)
                cute.arch.atomic_fmax(gap + rgi, _round_bf16(wmax), sign_bit=False)

    # final flush of the last column block this CTA touched
    slot = bidx - _cta_of(cur_cb * rg, gdim, tasks)
    if cutlass.const_expr(DBIAS):
        if tq == 0:
            part = wsd + (cur_cb * p + slot) * CTA_COLS + warp * WARP_COLS
            for blk in cutlass.range_constexpr(4):
                (part + blk * 16 + gi).store(cacc[4 * blk + 0])
                (part + blk * 16 + gi + 8).store(cacc[4 * blk + 2])
    cute.arch.fence_acq_rel_gpu()
    cute.arch.sync_threads()
    n_touch = _cta_of(cur_cb * rg + rg - 1, gdim, tasks) - _cta_of(cur_cb * rg, gdim, tasks) + 1
    if tidx == 0:
        prev = cute.arch.atomic_add(cnt + cur_cb, cutlass.Int32(1))
        flag.store(prev - (n_touch - 1))
    cute.arch.sync_threads()
    if flag.load() == 0:
        cute.arch.fence_acq_rel_gpu()
        s0 = cutlass.Float32(0.0)
        s1 = cutlass.Float32(0.0)
        if cutlass.const_expr(DBIAS):
            base = wsd + cur_cb * p * CTA_COLS + 2 * tidx
            for k in cutlass.range(n_touch):
                vv = _ld_cg_v2((base + k * CTA_COLS).toint())
                s0 = s0 + vv[0]
                s1 = s1 + vv[1]
        (dbp + cur_cb * CTA_COLS + 2 * tidx).store(
            cutlass.Vector.from_elements(
                (s0.to(cutlass.BFloat16), s1.to(cutlass.BFloat16)), cutlass.BFloat16),
            alignment=4)
        if tidx == 0:
            (cnt + cur_cb).store(cutlass.Int32(0))

    # amax: warp -> CTA -> partial slot -> last CTA reduces all
    rmax = _warp_max(cute.arch.fmax(cute.math.abs(_bf16_lo(rmax_packed)),
                                    cute.math.abs(_bf16_hi(rmax_packed))))
    cmax = _warp_max(cmax)
    if lane == 0:
        (stage + 2 * warp).store(rmax)
        (stage + 2 * warp + 1).store(cmax)
    cute.arch.sync_threads()
    if tidx == 0:
        r = cutlass.Float32(0.0)
        c = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(WARPS):
            r = cute.arch.fmax(r, (stage + 2 * w).load())
            c = cute.arch.fmax(c, (stage + 2 * w + 1).load())
        (wsa + 2 * bidx).store(r)
        (wsa + 2 * bidx + 1).store(c)
        cute.arch.fence_acq_rel_gpu()
        prev = cute.arch.atomic_add(cnt + n_cb, cutlass.Int32(1))
        flag.store(prev - (gdim - 1))
    cute.arch.sync_threads()
    if flag.load() == 0:
        cute.arch.fence_acq_rel_gpu()
        r = cutlass.Float32(0.0)
        c = cutlass.Float32(0.0)
        for i in cutlass.range(tidx, gdim, THREADS):
            vv = _ld_cg_v2((wsa + 2 * i).toint())
            r = cute.arch.fmax(r, vv[0])
            c = cute.arch.fmax(c, vv[1])
        r = _warp_max(r)
        c = _warp_max(c)
        if lane == 0:
            (stage + 2 * warp).store(r)
            (stage + 2 * warp + 1).store(c)
        cute.arch.sync_threads()
        if tidx == 0:
            r = cutlass.Float32(0.0)
            c = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                r = cute.arch.fmax(r, (stage + 2 * w).load())
                c = cute.arch.fmax(c, (stage + 2 * w + 1).load())
            amax[0] = r                    # already on the bf16 grid
            amax[1] = _round_bf16(c)       # rounded to bf16 before the max
            (cnt + n_cb).store(cutlass.Int32(0))   # self-reset for the next launch


@cute.jit
def dgelu_dbias_amax_ctm_host(
    grad: cute.Tensor, pre: cute.Tensor, out: cute.Tensor,
    dbias: cute.Tensor, amax: cute.Tensor, group_amax: cute.Tensor,
    ws_dbias: cute.Tensor, ws_amax: cute.Tensor, counters: cute.Tensor, rht: cute.Tensor,
    n_w: cutlass.Int32, rg: cutlass.Int32, n_cb: cutlass.Int32, p: cutlass.Int32,
    tasks: cutlass.Int32, grid_x: cutlass.Int32, stream: cuda.CUstream,
    FAST: cutlass.Constexpr[bool], RHT: cutlass.Constexpr[bool],
    DBIAS: cutlass.Constexpr[bool], PREFETCH: cutlass.Constexpr[bool],
    GROUP: cutlass.Constexpr[bool], FMA_LEFT: cutlass.Constexpr[bool],
    OCC: cutlass.Constexpr[int], CARVEOUT: cutlass.Constexpr[int],
) -> None:
    dgelu_dbias_amax_ctm_kernel(
        grad, pre, out, dbias, amax, group_amax, ws_dbias, ws_amax, counters, rht,
        n_w, rg, n_cb, p, tasks, FAST, RHT, DBIAS, PREFETCH, GROUP, FMA_LEFT,
    ).launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1),
             min_blocks_per_mp=OCC, preferred_smem_carveout=CARVEOUT, stream=stream)


# Host side

#: Which product nvcc fused into the final add of aten's dgelu (see _dgelu_f32).
FMA_LEFT_DEFAULT = True

_compiled: dict[tuple, Callable] = {}
_sm_count: dict[int, int] = {}
_counters: dict[tuple[int, int], torch.Tensor] = {}


def _compile(device_index: int, fast: bool, rht: bool, dbias: bool, prefetch: bool,
             group: bool, fma_left: bool = FMA_LEFT_DEFAULT,
             variant: str = DEFAULT_VARIANT) -> Callable:
    """Compile once per (device, constexprs, launch variant); every shape stays symbolic."""
    occ, carveout = LAUNCH_VARIANTS[variant]
    key = (device_index, fast, rht, dbias, prefetch, group, fma_left, variant)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            i32, f32, i16, bf = cutlass.Int32, cutlass.Float32, cutlass.Int16, cutlass.BFloat16
            sym = cute.sym_int64
            f_words = [make_fake_compact_tensor(i32, (sym(256, s),), assumed_align=16)
                       for s in ("g", "p", "o")]
            f_dbias = make_fake_compact_tensor(bf, (sym(512, "n"),), assumed_align=16)
            f_amax = make_fake_compact_tensor(f32, (2,), assumed_align=8)
            f_group = make_fake_compact_tensor(f32, (sym(1, "gr"),), assumed_align=4)
            f_wsd = make_fake_compact_tensor(f32, (sym(512, "wd"),), assumed_align=16)
            f_wsa = make_fake_compact_tensor(f32, (sym(2, "wa"),), assumed_align=8)
            f_cnt = make_fake_compact_tensor(i32, (sym(1, "c"),), assumed_align=4)
            f_rht = make_fake_compact_tensor(i16, (256,), assumed_align=16)
            ints = [cutlass.Int32(0)] * 6
            _compiled[key] = cute.compile(
                dgelu_dbias_amax_ctm_host, *f_words, f_dbias, f_amax, f_group, f_wsd, f_wsa,
                f_cnt, f_rht, *ints, make_fake_stream(), fast, rht, dbias, prefetch, group,
                fma_left, occ, carveout, options="--enable-tvm-ffi",
            )
    return _compiled[key]


def geometry(m: int, n: int, device_index: int,
             variant: str = DEFAULT_VARIANT) -> tuple[int, int, int, int, int]:
    """(rg, n_cb, tasks, grid, p) for an [M, N] input on this device.

    rg = M / 16 row groups, n_cb = N / 512 column blocks, tasks = rg * n_cb, persistent
    grid min(SMs * blocks, tasks), p = ceil(grid / n_cb) + 2 partial slots per column block.
    """
    if device_index not in _sm_count:
        _sm_count[device_index] = torch.cuda.get_device_properties(
            device_index).multi_processor_count
    rg = m // GROUP_ROWS
    n_cb = n // CTA_COLS
    tasks = rg * n_cb
    blocks = max(LAUNCH_VARIANTS[variant][0], MIN_BLOCKS_PER_SM)
    grid = max(1, min(_sm_count[device_index] * blocks, tasks))
    p = (grid + n_cb - 1) // n_cb + 2
    return rg, n_cb, tasks, grid, p


def supported(x: torch.Tensor) -> bool:
    """Contiguous 16 B-aligned BF16 [M, N] with M % 16 == 0, N % 512 == 0, M x N < 2**31."""
    return (
        x.dim() == 2 and x.dtype is torch.bfloat16 and x.is_contiguous()
        and x.shape[0] % GROUP_ROWS == 0 and x.shape[0] > 0
        and x.shape[1] % CTA_COLS == 0 and x.shape[1] > 0
        and x.numel() <= _INT32_MAX and x.data_ptr() % 16 == 0
    )


def _check(grad: torch.Tensor, pre: torch.Tensor, rht: torch.Tensor) -> None:
    if not (grad.is_cuda and pre.is_cuda and rht.is_cuda):
        raise ValueError("dgelu_dbias_amax_ctm: all tensors must be CUDA")
    if not supported(grad):
        raise ValueError(
            "dgelu_dbias_amax_ctm: grad must be a contiguous, 16 B-aligned bf16 [M, N] with "
            f"M % 16 == 0, N % 512 == 0 and M x N < 2**31, got {tuple(grad.shape)} {grad.dtype}"
        )
    if not (pre.dtype is torch.bfloat16 and pre.is_contiguous() and pre.shape == grad.shape
            and pre.data_ptr() % 16 == 0):
        raise ValueError("dgelu_dbias_amax_ctm: pre must match grad (bf16, contiguous, same shape)")
    if not (rht.dtype is torch.bfloat16 and rht.is_contiguous() and tuple(rht.shape) == (16, 16)):
        raise ValueError("dgelu_dbias_amax_ctm: rht_matrix must be a contiguous bf16 [16, 16]")


def _counters_for(n_cb: int, device: torch.device) -> torch.Tensor:
    """The persistent zeroed counter buffer for n_cb column blocks on device.

    Zero at entry and exit of every launch, so graph replays reuse it; two launches
    must not run concurrently on different streams of one device.
    """
    key = (device.index, n_cb)
    buf = _counters.get(key)
    if buf is None:
        buf = torch.zeros(n_cb + 1, dtype=torch.int32, device=device)
        _counters[key] = buf
    return buf


def production_fast() -> bool:
    """Read FLAT_TANH_APPROX_DGELU (default on), as the CUDA op does."""
    return os.environ.get("FLAT_TANH_APPROX_DGELU", "1") == "1"


def _run(grad: torch.Tensor, pre: torch.Tensor, rht: torch.Tensor, *, fast: bool,
         group: torch.Tensor | None, rht_on: bool = True, dbias_on: bool = True,
         prefetch: bool = True, fma_left: bool = FMA_LEFT_DEFAULT,
         variant: str = DEFAULT_VARIANT) -> tuple:
    dev = grad.device.index
    m, n = grad.shape
    rg, n_cb, tasks, grid, p = geometry(m, n, dev, variant)
    out = torch.empty_like(grad)
    dbias = torch.empty(n, dtype=torch.bfloat16, device=grad.device)
    amax = torch.empty(2, dtype=torch.float32, device=grad.device)
    ws_dbias = torch.empty(n_cb * p * CTA_COLS, dtype=torch.float32, device=grad.device)
    ws_amax = torch.empty(grid * 2, dtype=torch.float32, device=grad.device)
    g = group if group is not None else torch.empty(1, dtype=torch.float32, device=grad.device)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(dev, fast, rht_on, dbias_on, prefetch, group is not None, fma_left, variant)(
        grad.view(torch.int32).view(-1), pre.view(torch.int32).view(-1),
        out.view(torch.int32).view(-1), dbias, amax, g, ws_dbias, ws_amax,
        _counters_for(n_cb, grad.device), rht.view(torch.int16).view(-1),
        n // 2, rg, n_cb, p, tasks, grid, stream,
    )
    return out, dbias, amax


def dgelu_dbias_amax_ctm(
    grad: torch.Tensor, pre: torch.Tensor, rht_matrix: torch.Tensor, fast: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused tanh-GELU backward, bias gradient and NVFP4 amax pair, CUTLASS-DSL kernel.

    grad, pre: [M, N] bf16 contiguous, M % 16 == 0, N % 512 == 0; rht_matrix: [16, 16] bf16.
    fast: True = tanh.approx.f32, False = tanhf; None reads FLAT_TANH_APPROX_DGELU.
    """
    _check(grad, pre, rht_matrix)
    return _run(grad, pre, rht_matrix, fast=production_fast() if fast is None else fast,
                group=None)


def dgelu_dbias_amax_ctm_pergroup(
    grad: torch.Tensor, pre: torch.Tensor, rht_matrix: torch.Tensor, fast: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """dgelu_dbias_amax_ctm plus the per-16-row-group post-RHT amax.

    Returns (dgelu, dbias, amax, group_amax) with group_amax fp32 [M/16].
    """
    _check(grad, pre, rht_matrix)
    group = torch.zeros(grad.shape[0] // GROUP_ROWS, dtype=torch.float32, device=grad.device)
    out, dbias, amax = _run(grad, pre, rht_matrix,
                            fast=production_fast() if fast is None else fast, group=group)
    return out, dbias, amax, group


def dgelu_dbias_amax_ctm_variant(
    grad: torch.Tensor, pre: torch.Tensor, rht_matrix: torch.Tensor, *, fast: bool,
    rht_on: bool = True, dbias_on: bool = True, prefetch: bool = True,
    fma_left: bool = FMA_LEFT_DEFAULT, variant: str = DEFAULT_VARIANT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Benchmark entry: one kernel variant, to attribute the epilogue cost.

    With rht_on / dbias_on clear the corresponding output is meaningless
    (amax[1] == 0 / dbias unwritten).
    """
    _check(grad, pre, rht_matrix)
    return _run(grad, pre, rht_matrix, fast=fast, group=None, rht_on=rht_on,
                dbias_on=dbias_on, prefetch=prefetch, fma_left=fma_left, variant=variant)


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "dgelu_dbias_amax_ctm(Tensor grad, Tensor pre, Tensor rht_matrix)"
    " -> (Tensor, Tensor, Tensor)"
)
_lib.define(
    "dgelu_dbias_amax_ctm_pergroup(Tensor grad, Tensor pre, Tensor rht_matrix)"
    " -> (Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "dgelu_dbias_amax_ctm", "CUDA")
def _dgelu_dbias_amax_ctm_impl(
    grad: torch.Tensor, pre: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return dgelu_dbias_amax_ctm(grad, pre, rht_matrix)


@torch.library.impl(_lib, "dgelu_dbias_amax_ctm_pergroup", "CUDA")
def _dgelu_dbias_amax_ctm_pergroup_impl(
    grad: torch.Tensor, pre: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return dgelu_dbias_amax_ctm_pergroup(grad, pre, rht_matrix)
