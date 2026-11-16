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

"""torch.ops.flux.mxfp8_rowwise_cast_amax_ctm: MXFP8 rowwise cast + NVFP4 amax, CUTLASS DSL.

One read of a BF16 [rows, cols] tensor emits E4M3 data, GEMM-swizzled E8M0 scales
(32-element blocks) and the per-tensor amax, plain and post-RHT.
"""

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
from cutlass.cutlass_dsl import T as _T

from flat.custom_ops import amax_arena

THREADS = 128
TILE_ROWS = 32
TILE_COLS = 128
NUM_WARPS = THREADS // 32
SMEM_PITCH_B = TILE_COLS * 2 + 16      # 272 B: ldmatrix conflict-free
SMEM_TILE_B = TILE_ROWS * SMEM_PITCH_B  # 8704 B
CTAS_PER_SM_TARGET = int(os.environ.get("FLAT_CAST_CTAS_PER_SM", "12"))
_E4M3_MAX_NORM_RCP = 1.0 / 448.0
_BF16_HI_MASK = -65536                  # 0xFFFF0000 as a signed Int32
_INT32_MAX = 2**31 - 1


_MAX_XORSIGN_ABS_BF16X2 = "max.xorsign.abs.bf16x2 {$w0}, {$r0}, {$r1};"

#: float_to_e8m0 (cvt.rp.satfinite.ue8m0x2.f32, low byte) fused with exp2f_rcp:
#: {$w0} = biased exponent (Int32), {$w1} = 1/scale as Float32 bits; 254 -> 2^-127, 255 -> NaN.
_E8M0_AND_RCP = "\n".join([
    "{",
    ".reg .b16 e16;",
    ".reg .b32 e, sh;",
    ".reg .pred p;",
    "cvt.rp.satfinite.ue8m0x2.f32 e16, 0f00000000, {$r0};",
    "cvt.u32.u16 e, e16;",
    "and.b32 e, e, 255;",
    "mov.b32 {$w0}, e;",
    "mov.b32 sh, 254;",
    "sub.u32 sh, sh, e;",
    "shl.b32 sh, sh, 23;",
    "setp.eq.u32 p, e, 254;",
    "selp.b32 sh, 0x00400000, sh, p;",
    "setp.eq.u32 p, e, 255;",
    "selp.b32 sh, 0x7fffffff, sh, p;",
    "mov.b32 {$w1}, sh;",
    "}",
])

#: mul_cvt_2x on two bf16x2 words: bf16 -> f32, mul.f32x2 by the reciprocal scale,
#: cvt.rn.satfinite.e4m3x2.f32; the four output bytes are elements 0..3 in memory order.
_MUL_CVT_2X = "\n".join([
    "{",
    ".reg .b16 h0, h1, h2, h3, o0, o1;",
    ".reg .b32 f0, f1, f2, f3;",
    ".reg .b64 s2, p0, p1;",
    "mov.b64 s2, { {$r2}, {$r2} };",
    "mov.b32 {h0, h1}, {$r0};",
    "mov.b32 {h2, h3}, {$r1};",
    "cvt.f32.bf16 f0, h0;",
    "cvt.f32.bf16 f1, h1;",
    "cvt.f32.bf16 f2, h2;",
    "cvt.f32.bf16 f3, h3;",
    "mov.b64 p0, {f0, f1};",
    "mov.b64 p1, {f2, f3};",
    "mul.f32x2 p0, p0, s2;",
    "mul.f32x2 p1, p1, s2;",
    "mov.b64 {f0, f1}, p0;",
    "mov.b64 {f2, f3}, p1;",
    "cvt.rn.satfinite.e4m3x2.f32 o0, f1, f0;",
    "cvt.rn.satfinite.e4m3x2.f32 o1, f3, f2;",
    "mov.b32 {$w0}, {o0, o1};",
    "}",
])

_LDMATRIX_X4_TRANS = (
    "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
    "{ {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];"
)
# Raw llvm.inline_asm with explicit constraints: inline_ptx lets LLVM fold
# constant-zero C operands into immediates, which ptxas rejects for mma.
_MMA_16816_BF16 = (
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
)
_MMA_CONSTRAINTS = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"


# Traced device helpers: plain Python control flow only


def _xorsign_abs(a, b):
    return cute.arch.inline_ptx(
        _MAX_XORSIGN_ABS_BF16X2, write_only_types=[cutlass.Int32], read_only_args=[a, b],
    )


def _e8m0_and_rcp(scaled_amax):
    return cute.arch.inline_ptx(
        _E8M0_AND_RCP, write_only_types=[cutlass.Int32, cutlass.Float32],
        read_only_args=[scaled_amax],
    )


def _mul_cvt_2x(w0, w1, inv):
    return cute.arch.inline_ptx(
        _MUL_CVT_2X, write_only_types=[cutlass.Int32], read_only_args=[w0, w1, inv],
    )


def _ldmatrix_x4_trans(addr):
    return cute.arch.inline_ptx(
        _LDMATRIX_X4_TRANS, write_only_types=[cutlass.Int32] * 4, read_only_args=[addr],
    )


def _mma_16816(a, b0, b1, c):
    """D = A(16x16 bf16, row) * B(16x8 bf16, col) + C, fp32 accumulate."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([_T.f32()] * 4),
        [v.ir_value() for v in (a[0], a[1], a[2], a[3], b0, b1, c[0], c[1], c[2], c[3])],
        _MMA_16816_BF16, _MMA_CONSTRAINTS,
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(cutlass.Float32(llvm.extractvalue(_T.f32(), res, [i])) for i in range(4))


def _bf16_lo(w):
    return (w << 16).bitcast(cutlass.Float32)


def _bf16_hi(w):
    return (w & cutlass.Int32(_BF16_HI_MASK)).bitcast(cutlass.Float32)


def _round_bf16(x):
    return x.to(cutlass.BFloat16).to(cutlass.Float32)


def _warp_max(v):
    for off in (16, 8, 4, 2, 1):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=off))
    return v


def _swizzled_scale_idx(i, j, n_tiles_x):
    """cuBLASLt's 128 x 4 scale-atom swizzle, as Int32 math."""
    idx = ((i >> 7) * n_tiles_x + (j >> 2)) * 512
    return idx + (i & 31) * 16 + ((i >> 5) & 3) * 4 + (j & 3)


def _load_block(xp, eoff):
    """The thread's MX block: four 16 B ld.global.nc as 4 x Vector<Int32, 4>."""
    return [
        (xp + eoff + 8 * i).load(16, count=8, is_invariant=True).bitcast(cutlass.Int32)
        for i in range(4)
    ]


@cute.kernel
def mxfp8_cast_amax_ctm_kernel(
    x: cute.Tensor, out: cute.Tensor, scales: cute.Tensor, amax: cute.Tensor,
    ws: cute.Tensor, counter: cute.Tensor, group: cute.Tensor, rht: cute.Tensor,
    cols: cutlass.Int32, n_ctas: cutlass.Int32,
    RHT: cutlass.Constexpr[bool], GROUP: cutlass.Constexpr[bool],
    ZFREE: cutlass.Constexpr[bool], T: cutlass.Constexpr[int],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    r_local = tidx >> 2
    blk = tidx & 3
    col0 = bidx * TILE_COLS + blk * 32
    tile0 = bidy * T
    n_tiles_x = cols >> 7

    xp = cutlass.Pointer(x.iterator, dtype=cutlass.BFloat16)
    op = cutlass.Pointer(out.iterator, dtype=cutlass.Int32)
    sp = cutlass.Pointer(scales.iterator, dtype=cutlass.Uint8)
    red = cute.arch.alloc_smem(cutlass.Float32, 2 * NUM_WARPS, 16)
    rp = cutlass.Pointer(red, dtype=cutlass.Float32)
    flag = cute.arch.alloc_smem(cutlass.Int32, 1, 4)
    fp = cutlass.Pointer(flag, dtype=cutlass.Int32)

    if cutlass.const_expr(RHT):
        sbuf = cute.arch.alloc_smem(cutlass.Int32, 2 * SMEM_TILE_B // 4, 16)
        sb = cutlass.Pointer(sbuf, dtype=cutlass.Int32)
        sbase = sbuf.toint()
        ld_row = lane & 15
        ld_chunk = lane >> 4
        tq = lane & 3
        gi = lane >> 2
        # B fragments of R (mma.m16n8k16.col): k = 2tq, 2tq+1 (+8) at n = gi + 8 jb.
        bfrag = []
        for jb in cutlass.range_constexpr(2):
            nn = jb * 8 + gi
            r0 = rht[(2 * tq) * 16 + nn].to(cutlass.Int32) & 0xFFFF
            r1 = rht[(2 * tq + 1) * 16 + nn].to(cutlass.Int32) & 0xFFFF
            r8 = rht[(2 * tq + 8) * 16 + nn].to(cutlass.Int32) & 0xFFFF
            r9 = rht[(2 * tq + 9) * 16 + nn].to(cutlass.Int32) & 0xFFFF
            bfrag.append((r0 | (r1 << 16), r8 | (r9 << 16)))
        rg = warp >> 1                 # the warp's 16-row group of the tile
        cb0 = (warp & 1) * 4           # its first 16-column block

    plain = cutlass.Float32(0.0)
    cmax = cutlass.Float32(0.0)

    words_next = _load_block(xp, (tile0 * TILE_ROWS + r_local) * cols + col0)
    for t in cutlass.range_constexpr(T):
        words = words_next
        if cutlass.const_expr(t + 1 < T):
            words_next = _load_block(
                xp, ((tile0 + t + 1) * TILE_ROWS + r_local) * cols + col0)
        row = (tile0 + t) * TILE_ROWS + r_local
        eoff = row * cols + col0

        if cutlass.const_expr(RHT):
            # stage the tile (pitch 272 B) for the ldmatrix reads below
            soff = ((t & 1) * SMEM_TILE_B + r_local * SMEM_PITCH_B + blk * 64) >> 2
            for i in cutlass.range_constexpr(4):
                (sb + soff + 4 * i).store(words[i], alignment=16)

        # MXFP8 rowwise half
        flat = []
        for i in cutlass.range_constexpr(4):
            for k in cutlass.range_constexpr(4):
                flat.append(words[i][k])
        m = _xorsign_abs(flat[0], flat[1])
        for i in cutlass.range_constexpr(2, 16):
            m = _xorsign_abs(m, flat[i])
        thread_amax = cute.arch.fmax(cute.math.abs(_bf16_lo(m)), cute.math.abs(_bf16_hi(m)))
        plain = cute.arch.fmax(plain, thread_amax)
        e, inv = _e8m0_and_rcp(thread_amax * cutlass.Float32(_E4M3_MAX_NORM_RCP))
        (sp + _swizzled_scale_idx(row, col0 >> 5, n_tiles_x)).store(e.to(cutlass.Uint8))
        q = []
        for j in cutlass.range_constexpr(8):
            q.append(_mul_cvt_2x(flat[2 * j], flat[2 * j + 1], inv))
        (op + (eoff >> 2)).store(
            cutlass.Vector.from_elements(tuple(q[0:4]), cutlass.Int32), alignment=16)
        (op + (eoff >> 2) + 4).store(
            cutlass.Vector.from_elements(tuple(q[4:8]), cutlass.Int32), alignment=16)

        # post-RHT amax
        if cutlass.const_expr(RHT):
            cute.arch.sync_threads()
            gmax = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(4):
                cb = cb0 + i
                addr = (sbase + (t & 1) * SMEM_TILE_B + (rg * 16 + ld_row) * SMEM_PITCH_B
                        + (cb * 16 + ld_chunk * 8) * 2)
                f0, f1, f2, f3 = _ldmatrix_x4_trans(addr)
                a = (f0, f2, f1, f3)       # A = X^T for mma.sync
                zero = (cutlass.Float32(0.0),) * 4
                d0 = _mma_16816(a, bfrag[0][0], bfrag[0][1], zero)
                d1 = _mma_16816(a, bfrag[1][0], bfrag[1][1], zero)
                tmax = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(4):
                    tmax = cute.arch.fmax(tmax, cute.math.abs(d0[k]))
                    tmax = cute.arch.fmax(tmax, cute.math.abs(d1[k]))
                gmax = cute.arch.fmax(gmax, tmax)
            cmax = cute.arch.fmax(cmax, gmax)
            if cutlass.const_expr(GROUP):
                # group amax: max over the tile's 16-row group; this warp holds 4 of
                # its 8 column blocks, so warp max + atomic (rounded to bf16)
                wmax = _warp_max(gmax)
                if lane == 0:
                    gp = cutlass.Pointer(group.iterator, dtype=cutlass.Float32)
                    cute.arch.atomic_fmax(
                        gp + (tile0 + t) * 2 + rg, _round_bf16(wmax), sign_bit=False)

    # CTA reduction
    p_w = _warp_max(plain)
    h_w = _warp_max(cmax)
    if lane == 0:
        (rp + warp).store(p_w)
        (rp + NUM_WARPS + warp).store(_round_bf16(h_w))
    cute.arch.sync_threads()
    ap = cutlass.Pointer(amax.iterator, dtype=cutlass.Float32)
    if cutlass.const_expr(not ZFREE):
        if tidx == 0:
            p = (rp + 0).load()
            h = (rp + NUM_WARPS).load()
            for w in cutlass.range_constexpr(1, NUM_WARPS):
                p = cute.arch.fmax(p, (rp + w).load())
                h = cute.arch.fmax(h, (rp + NUM_WARPS + w).load())
            cute.arch.atomic_fmax(ap, p, sign_bit=False)
            if cutlass.const_expr(RHT):
                cute.arch.atomic_fmax(ap + 1, h, sign_bit=False)
    else:
        # zero-free: partials to ws, last CTA reduces in fixed order, resets counter
        wsp = cutlass.Pointer(ws.iterator, dtype=cutlass.Float32)
        cp = cutlass.Pointer(counter.iterator, dtype=cutlass.Int32)
        cta = bidy * n_tiles_x + bidx
        if tidx == 0:
            p = (rp + 0).load()
            h = (rp + NUM_WARPS).load()
            for w in cutlass.range_constexpr(1, NUM_WARPS):
                p = cute.arch.fmax(p, (rp + w).load())
                h = cute.arch.fmax(h, (rp + NUM_WARPS + w).load())
            (wsp + 2 * cta).store(p)
            (wsp + 2 * cta + 1).store(h)
            cute.arch.fence_acq_rel_gpu()
            old = cute.arch.atomic_add(cp, cutlass.Int32(1))
            fp.store(old)
        cute.arch.sync_threads()
        old_b = fp.load()
        if old_b == n_ctas - 1:
            cute.arch.fence_acq_rel_gpu()
            r = cutlass.Float32(0.0)
            c = cutlass.Float32(0.0)
            for i in cutlass.range(tidx, n_ctas, THREADS):
                r = cute.arch.fmax(r, (wsp + 2 * i).load(is_volatile=True))
                c = cute.arch.fmax(c, (wsp + 2 * i + 1).load(is_volatile=True))
            r = _warp_max(r)
            c = _warp_max(c)
            if lane == 0:
                (rp + warp).store(r)
                (rp + NUM_WARPS + warp).store(c)
            cute.arch.sync_threads()
            if tidx == 0:
                r = (rp + 0).load()
                c = (rp + NUM_WARPS).load()
                for w in cutlass.range_constexpr(1, NUM_WARPS):
                    r = cute.arch.fmax(r, (rp + w).load())
                    c = cute.arch.fmax(c, (rp + NUM_WARPS + w).load())
                ap.store(r)
                (ap + 1).store(c)
                cp.store(cutlass.Int32(0))


@cute.jit
def mxfp8_cast_amax_ctm_host(
    x: cute.Tensor, out: cute.Tensor, scales: cute.Tensor, amax: cute.Tensor,
    ws: cute.Tensor, counter: cute.Tensor, group: cute.Tensor, rht: cute.Tensor,
    cols: cutlass.Int32, n_ctas: cutlass.Int32, grid_x: cutlass.Int32, grid_y: cutlass.Int32,
    stream: cuda.CUstream, RHT: cutlass.Constexpr[bool], GROUP: cutlass.Constexpr[bool],
    ZFREE: cutlass.Constexpr[bool], T: cutlass.Constexpr[int],
) -> None:
    mxfp8_cast_amax_ctm_kernel(
        x, out, scales, amax, ws, counter, group, rht, cols, n_ctas, RHT, GROUP, ZFREE, T,
    ).launch(grid=(grid_x, grid_y, 1), block=(THREADS, 1, 1), stream=stream)


# Host side

_compiled: dict[tuple[int, bool, bool, bool, int], Callable] = {}
_sm_count: dict[int, int] = {}
_counters: dict[int, torch.Tensor] = {}


def _compile(device_index: int, rht: bool, group: bool, zfree: bool, tiles: int) -> Callable:
    """Compile once per (device, RHT, GROUP, ZFREE, T); every shape stays symbolic."""
    key = (device_index, rht, group, zfree, tiles)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            bf, u8, f32, i16, i32 = (cutlass.BFloat16, cutlass.Uint8, cutlass.Float32,
                                     cutlass.Int16, cutlass.Int32)
            sym = cute.sym_int64
            fx = make_fake_compact_tensor(bf, (sym(8, "nx"),), assumed_align=16)
            fout = make_fake_compact_tensor(u8, (sym(16, "no"),), assumed_align=16)
            fsc = make_fake_compact_tensor(u8, (sym(4, "ns"),), assumed_align=4)
            famax = make_fake_compact_tensor(f32, (2,), assumed_align=8)
            fws = make_fake_compact_tensor(f32, (sym(2, "nw"),), assumed_align=8)
            fcnt = make_fake_compact_tensor(i32, (1,), assumed_align=4)
            fgrp = make_fake_compact_tensor(f32, (sym(1, "ng"),), assumed_align=4)
            frht = make_fake_compact_tensor(i16, (256,), assumed_align=16)
            ints = [i32(0)] * 4
            _compiled[key] = cute.compile(
                mxfp8_cast_amax_ctm_host, fx, fout, fsc, famax, fws, fcnt, fgrp, frht,
                *ints, make_fake_stream(), rht, group, zfree, tiles,
                options="--enable-tvm-ffi",
            )
        if device_index not in _counters:
            # persistent, self-resetting counter (the last CTA writes 0), created
            # outside any later graph capture
            _counters[device_index] = torch.zeros(
                1, dtype=torch.int32, device=torch.device("cuda", device_index))
    return _compiled[key]


def pick_tiles(col_tiles: int, row_tiles: int, device_index: int) -> int:
    """Row tiles per CTA: largest of 8/4/2/1 dividing row_tiles with >= CTAS_PER_SM_TARGET CTAs/SM.

    col_tiles = cols / 128, row_tiles = rows / 32. Returns T in {8, 4, 2, 1}.
    """
    if device_index not in _sm_count:
        _sm_count[device_index] = torch.cuda.get_device_properties(
            device_index).multi_processor_count
    want = _sm_count[device_index] * CTAS_PER_SM_TARGET
    for t in (8, 4, 2):
        if row_tiles % t == 0 and col_tiles * (row_tiles // t) >= want:
            return t
    return 1


def _check(x: torch.Tensor, rht_matrix: torch.Tensor) -> None:
    if not (x.is_cuda and rht_matrix.is_cuda):
        raise ValueError("mxfp8_rowwise_cast_amax_ctm: all tensors must be CUDA")
    if not (x.dim() == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()):
        raise ValueError("mxfp8_rowwise_cast_amax_ctm: x must be a contiguous 2-D BF16 tensor")
    rows, cols = x.shape
    if rows % 128 != 0 or cols % TILE_COLS != 0:
        raise ValueError(
            "mxfp8_rowwise_cast_amax_ctm: rows % 128 == 0 and cols % 128 == 0 required, "
            f"got {rows}x{cols}")
    if rows * cols > _INT32_MAX:
        raise ValueError("mxfp8_rowwise_cast_amax_ctm: numel exceeds the Int32 index range")
    if not (rht_matrix.dtype == torch.bfloat16 and rht_matrix.is_contiguous()
            and tuple(rht_matrix.shape) == (16, 16)):
        raise ValueError(
            "mxfp8_rowwise_cast_amax_ctm: rht_matrix must be a contiguous BF16 [16, 16]")


def _run(
    x: torch.Tensor, rht_matrix: torch.Tensor, rht_amax: bool,
    group: torch.Tensor | None, zfree: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = x.device.index
    rows, cols = x.shape
    col_tiles, row_tiles = cols // TILE_COLS, rows // TILE_ROWS
    tiles = pick_tiles(col_tiles, row_tiles, dev)
    grid_x, grid_y = col_tiles, row_tiles // tiles
    n_ctas = grid_x * grid_y
    u8 = dict(dtype=torch.uint8, device=x.device)
    out = torch.empty(rows, cols, **u8)
    scales = torch.empty(rows, cols // 32, **u8)
    fn = _compile(dev, rht_amax, group is not None, zfree, tiles)
    if zfree:
        amax = torch.empty(2, dtype=torch.float32, device=x.device)
        ws = torch.empty(2 * n_ctas, dtype=torch.float32, device=x.device)
    else:
        amax = amax_arena.alloc(x.device)
        ws = torch.empty(2, dtype=torch.float32, device=x.device)
    g = group if group is not None else torch.empty(1, dtype=torch.float32, device=x.device)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    fn(x.view(-1), out.view(-1), scales.view(-1), amax, ws, _counters[dev], g,
       rht_matrix.view(torch.int16).view(-1), cols, n_ctas, grid_x, grid_y, stream)
    return out, scales, amax


def mxfp8_rowwise_cast_amax_ctm(
    x: torch.Tensor, rht_matrix: torch.Tensor, rht_amax: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One read of x: MXFP8 rowwise half plus the NVFP4 per-tensor amax, CUTLASS-DSL kernel.

    Returns (data, scales, amax) with amax = (max|x|, post-RHT amax or 0), bitwise
    flux.mxfp8_rowwise_cast_amax.
    """
    _check(x, rht_matrix)
    return _run(x, rht_matrix, rht_amax, None, False)


def mxfp8_rowwise_cast_amax_ctm_z(
    x: torch.Tensor, rht_matrix: torch.Tensor, rht_amax: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mxfp8_rowwise_cast_amax_ctm without the accumulator fill kernel; same contract, same bytes.

    Returns (data, scales, amax).
    """
    _check(x, rht_matrix)
    return _run(x, rht_matrix, rht_amax, None, True)


def mxfp8_rowwise_cast_amax_ctm_pergroup(
    x: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """mxfp8_rowwise_cast_amax_ctm plus the per-16-row-group post-RHT amax.

    Returns (data, scales, amax, group_amax) with group_amax fp32 [rows/16].
    """
    _check(x, rht_matrix)
    group = torch.zeros(x.shape[0] // 16, dtype=torch.float32, device=x.device)
    data, scales, amax = _run(x, rht_matrix, True, group, False)
    return data, scales, amax, group


def verify(rows: int = 8192, cols: int = 3072) -> None:
    """Compile, run and check both variants against TE, bitwise.

    The CUDA op this used to compare against has been deleted, so TE is the
    oracle: the MXFP8 half against a straight MXFP8 cast, and the two amaxes
    against the reference reduction and TE's columnwise amax.

    rows, cols: probe tensor shape, both % 128 == 0. Raises AssertionError on
    any difference.
    """
    import transformer_engine.pytorch.cpp_extensions as tex

    from flat import nvfp4_ops as nv

    torch.manual_seed(rows * 31 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    q = nv.role_quantizer("act", columnwise_only=True)
    mx = nv._quantize_mx(x, nv._MX_FPROP_QUANTIZER)
    te_col = tex.quantize(x, q)._amax_columnwise.view(-1)
    for rht_amax in (True, False):
        for fn in (mxfp8_rowwise_cast_amax_ctm, mxfp8_rowwise_cast_amax_ctm_z):
            data, scales, amax = fn(x, q.rht_matrix, rht_amax)
            torch.cuda.synchronize()
            assert torch.equal(data, mx._rowwise_data), (
                f"{fn.__name__} rht_amax={rht_amax}: data")
            assert torch.equal(scales, mx._rowwise_scale_inv), (
                f"{fn.__name__} rht_amax={rht_amax}: scales")
            assert torch.equal(amax[0], x.float().abs().amax()), (
                f"{fn.__name__}: amax[0]")
            if rht_amax:
                assert torch.equal(amax[1:2], te_col), (
                    f"{fn.__name__}: post-RHT amax")
    print(f"verify [{rows},{cols}]: ctm and ctm_z bitwise TE, both roles: PASS")


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "mxfp8_rowwise_cast_amax_ctm(Tensor x, Tensor rht_matrix, bool rht_amax)"
    " -> (Tensor, Tensor, Tensor)"
)
_lib.define(
    "mxfp8_rowwise_cast_amax_ctm_z(Tensor x, Tensor rht_matrix, bool rht_amax)"
    " -> (Tensor, Tensor, Tensor)"
)
_lib.define(
    "mxfp8_rowwise_cast_amax_ctm_pergroup(Tensor x, Tensor rht_matrix)"
    " -> (Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "mxfp8_rowwise_cast_amax_ctm", "CUDA")
def _mxfp8_rowwise_cast_amax_ctm_impl(
    x: torch.Tensor, rht_matrix: torch.Tensor, rht_amax: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mxfp8_rowwise_cast_amax_ctm(x, rht_matrix, rht_amax)


@torch.library.impl(_lib, "mxfp8_rowwise_cast_amax_ctm_z", "CUDA")
def _mxfp8_rowwise_cast_amax_ctm_z_impl(
    x: torch.Tensor, rht_matrix: torch.Tensor, rht_amax: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mxfp8_rowwise_cast_amax_ctm_z(x, rht_matrix, rht_amax)


@torch.library.impl(_lib, "mxfp8_rowwise_cast_amax_ctm_pergroup", "CUDA")
def _mxfp8_rowwise_cast_amax_ctm_pergroup_impl(
    x: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return mxfp8_rowwise_cast_amax_ctm_pergroup(x, rht_matrix)
