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

"""``flux.dual_cast_one_read`` for the weight role on the CUTLASS DSL.

The per-step weight dual cast -- every quantized weight of every block, 304 launches a
step, 10.6 ms/step at ``04x04x32`` and 13.3 ms/step at ``64x04x04`` on the CUDA kernel
it replaced -- read once from a TMA-staged
ring instead of that kernel's register prefetch. Same contract as the deleted
``dual_cast_one_read_kernel.cu`` with ``rht=False`` (weight role):

* ``mx_data``    E4M3 ``[rows, cols]``          MXFP8 rowwise data (fprop operand)
* ``mx_scales``  E8M0 ``[rows, cols/32]``       GEMM-swizzled (cuBLASLt 128 x 4 atom)
* ``col_data``   E2M1 ``[cols, rows/2]`` bytes  NVFP4 columnwise data, transposed
* ``col_scales`` E4M3 ``[cols, rows/16]``       GEMM-swizzled (same atom)

with the per-tensor NVFP4 amax supplied by the caller (T6's optimizer-path weight
amax), 2-D 16 x 16 block scales and no Hadamard transform (TE
``quantize_transpose_nvfp4.cuh`` @ 27486e03). Every emitted byte is bitwise the
CUDA op's and TE's (``tests/test_dual_cast_one_read_ctm.py``).

Structure is ``onepass_act_ctm.py``'s (the act-role one-read cast, shipped on
GB300): one producer warp streams ``TILE_ROWS`` x 128 bf16 tiles as two SWIZZLE_128B
boxes per stage through a ``PipelineTmaAsync`` ring; consumer warp ``w`` casts its rows
to MXFP8 from shared memory (``LDS.128`` x4, one ``st.global.v8`` per 32-element block)
and owns ``8 / CW`` 16-column blocks for the NVFP4 half: ``ldmatrix.x4.trans`` gives
lane ``(gi, q)`` rows ``2q, 2q+1, 2q+8, 2q+9`` of columns ``gi`` and ``gi+8`` of a 16 x 16
block directly -- the weight role needs no ``mma`` -- the block amax is a full-warp
``max.xorsign.abs.bf16x2`` butterfly, the codes are staged warp-privately with
``stmatrix.x2`` and flushed as full lines every ``WINDOW`` rows. The un-permuted row
order (the act kernel permutes ``R``'s columns instead) costs four extra ``prmt`` per
16 B chunk in the flush and nothing in the per-element path.

TE's instructions are emitted verbatim (``onepass_act_ctm_common``); the weight role's
``S_dec = (float)(1.0 / (double)S_enc)`` is ``div.rn.f32 1.0, S_enc`` bitwise (double
rounding is innocuous for binary32 division through binary64; checked on 2 M random
floats, 0 mismatches).

Variants (``VARIANTS``) differ in tile rows, stages, consumer warps, flush window and
the residency knob; the dispatch (``pick``) is on ``numel``, a size expression, and is
fixed from the row's measurement. Public ``nvidia-cutlass-dsl`` surface only.
"""

from __future__ import annotations

import os
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
import torch.library
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from flat.custom_ops.onepass_act_ctm_common import (
    BOX_COLS, E4M3_MAX_NORM_RCP, FLT_MAX, FP4_MAX_INV, INT32_MAX, TILE_COLS,
    bf16_hi, bf16_lo, div_const_rn, e4m3_to_f32, e8m0_and_rcp, f32x2_to_e4m3x2,
    geometry_windows, global_encode_scale, ldmatrix_x4_trans, lds128, lds32,
    make_tma, mul_cvt_2x, mul_cvt_e2m1x4, prmt, ring_layout, sm_count, stg256,
    sts16, sts32, stsm_x2, swizzled_scale_idx, xorsign_abs,
)

#: variant -> (TILE_ROWS, STAGES, consumer warps, flush window rows, CTAs/SM pinned
#: through min_blocks_per_mp -- 0 = no residency knob and no carveout request).
VARIANTS: dict[str, tuple[int, int, int, int, int]] = {
    "A": (64, 4, 8, 256, 2),     # B4's shipped A_stsm geometry, 288 threads
    "A0": (64, 4, 8, 256, 0),    # A without the residency knob / carveout
    "S": (32, 4, 4, 128, 4),     # 160-thread CTAs: co-residency beside collectives
    "N": (32, 4, 4, 64, 4),      # 160 threads, 64-row window: 2x the CTAs of A at [3072,3072]
}
MODE_OVERSUB2 = 2
#: kernel symbol substring the dispatch launches
KERNEL = "dual_cast_one_read_ctm_kernel"
#: ``(numel upto, variant, mode)`` bands, fixed from the measurement: ``A+m0`` was the fastest passing arm at
#: every production weight shape by more than the 2.05 us quantum (warm paired medians
#: 26.3 / 14.5 / 32.1 / 31.8 us vs the runner-up 27.2 / 15.4 / 33.0 / 33.1), so one band.
DISPATCH: tuple[tuple[int, str, int], ...] = (
    (INT32_MAX, "A", 0),
)


@cute.kernel
def dual_cast_one_read_ctm_kernel(
    tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor, smem_layout: cute.ComposedLayout,
    out: cute.Tensor, scales: cute.Tensor, col_data: cute.Tensor, col_scales: cute.Tensor,
    amax_in: cute.Tensor, rows: cutlass.Int32, cols: cutlass.Int32,
    tiles_per_cta: cutlass.Int32,
    TILE_ROWS: cutlass.Constexpr[int], STAGES: cutlass.Constexpr[int],
    CW: cutlass.Constexpr[int], WINDOW: cutlass.Constexpr[int],
) -> None:
    FLUSH_TILES = WINDOW // TILE_ROWS
    RG = TILE_ROWS // 16                     # 16-row groups per tile (2 or 4)
    SUB_BYTES = TILE_ROWS * BOX_COLS * 2     # one 64-column box
    STAGE_BYTES = 2 * SUB_BYTES
    MX_UNITS = TILE_ROWS * 4 // (CW * 32)    # 32-element blocks per thread per tile
    CBW = 8 // CW                            # 16-column blocks per warp
    PIECE = RG * 8                           # E2M1 bytes per column per tile
    COLS_W = CBW * 16                        # columns staged per warp
    COL_BYTES = FLUSH_TILES * PIECE          # per column per window
    COL_PITCH = COL_BYTES + 16
    SF_PITCH = FLUSH_TILES * RG              # scale bytes per column per window
    COLSTAGE_W = COLS_W * COL_PITCH
    SFSTAGE_W = COLS_W * SF_PITCH
    CHUNKS_PER_COL = COL_BYTES // 16
    NCHUNK = COLS_W * CHUNKS_PER_COL
    WORDS_PER_COL = SF_PITCH // 4
    NWORDS = COLS_W * WORDS_PER_COL

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    row_tiles = rows // TILE_ROWS
    tile0 = bidy * tiles_per_cta
    ntiles = cutlass.min(tiles_per_cta, row_tiles - tile0)
    col0 = bidx * TILE_COLS

    smem = utils.SmemAllocator()
    # SWIZZLE_128B is a function of absolute shared-address bits 4-9: 1024 B-align.
    ring = smem.allocate_array(cutlass.BFloat16, STAGES * TILE_ROWS * TILE_COLS, byte_alignment=1024)
    s_tile = cute.make_tensor(
        cute.recast_ptr(ring, smem_layout.inner, dtype=cutlass.BFloat16), smem_layout.outer,
    )
    colstage = smem.allocate_array(cutlass.Uint8, CW * COLSTAGE_W, byte_alignment=16)
    sfstage = smem.allocate_array(cutlass.Uint8, CW * SFSTAGE_W, byte_alignment=16)
    lut = smem.allocate_array(cutlass.Float32, 256, byte_alignment=16)
    bars = smem.allocate_array(cutlass.Int64, 2 * STAGES, byte_alignment=8)

    pipe = pipeline.PipelineTmaAsync.create(
        num_stages=STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, CW),
        tx_count=STAGE_BYTES,
        barrier_storage=bars,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )

    # S_enc / S_dec / LUT: dual_cast_one_read_kernel.cu:493-499, weight role. S_dec is
    # (float)(1.0 / (double)S_enc) there == div.rn.f32 here (module docstring).
    amax = cutlass.Pointer(amax_in.iterator, dtype=cutlass.Float32).load()
    s_enc = global_encode_scale(amax)
    enc_mul = s_enc * cutlass.Float32(FP4_MAX_INV)
    lut_p = cutlass.Pointer(lut, dtype=cutlass.Float32)
    if warp < CW:
        s_dec = div_const_rn(1.0, s_enc)
        for i in cutlass.range(tidx, 256, CW * 32):
            v = div_const_rn(1.0, e4m3_to_f32(i) * s_dec)
            (lut_p + i).store(cute.arch.fmin(v, cutlass.Float32(FLT_MAX)))
    cute.arch.sync_threads()

    # ((TILE_ROWS, 64), n_rb, n_cb64) of the TMA tensor; ((TILE_ROWS, 64), 2*STAGES) of the ring.
    g_tiles = cute.local_tile(tma_tensor, (TILE_ROWS, BOX_COLS), (None, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1),
        cute.group_modes(s_tile, 0, 2), cute.group_modes(g_tiles, 0, 2),
    )

    if warp == CW:
        # ---- producer warp: two 64-column boxes per row tile onto one barrier ----
        pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
        for t in cutlass.range(ntiles):
            pipe.producer_acquire(pstate)
            bar = pipe.producer_get_barrier(pstate)
            rb = tile0 + t
            cute.copy(tma_atom, tAgA[(None, rb, 2 * bidx)], tAsA[(None, 2 * pstate.index)],
                      tma_bar_ptr=bar)
            cute.copy(tma_atom, tAgA[(None, rb, 2 * bidx + 1)], tAsA[(None, 2 * pstate.index + 1)],
                      tma_bar_ptr=bar)
            pstate.advance()
        pipe.producer_tail(pstate)

    if warp < CW:
        # ---- consumers -----------------------------------------------------------
        gi = lane >> 2
        q = lane & 3
        cb0 = warp * CBW
        ring_base = ring.toint()
        cst_base = colstage.toint() + warp * COLSTAGE_W
        sst_base = sfstage.toint() + warp * SFSTAGE_W
        ld_row = lane & 15                   # ldmatrix lane map (T21)
        ld_chunk_hi = lane >> 4
        stsm_lane = lane & 15                # stmatrix row-address lanes: matrix 0 <- 0-7, matrix 1 <- 8-15
        n_tiles_x = cols >> 7
        sf_tiles_x = (rows >> 4) >> 2
        half_rows = rows >> 1
        out_i64 = cutlass.Pointer(out.iterator, dtype=cutlass.Uint8).toint()
        sp = cutlass.Pointer(scales.iterator, dtype=cutlass.Uint8)
        cd_p = cutlass.Pointer(col_data.iterator, dtype=cutlass.Int32)
        cs_p = cutlass.Pointer(col_scales.iterator, dtype=cutlass.Int32)
        flt_max = cutlass.Float32(FLT_MAX)

        cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
        for t in cutlass.range(ntiles):
            pipe.consumer_wait(cstate)
            tile_s = ring_base + cstate.index * STAGE_BYTES
            grow0 = (tile0 + t) * TILE_ROWS

            # ---- MXFP8 rowwise half: thread = (row, 32-element block) ----------------
            # quantize_mxfp8.cuh:310-465 through TE's own instructions (076 / B4, verbatim).
            for k in cutlass.range_constexpr(MX_UNITS):
                u = tidx + k * CW * 32
                r = u >> 2
                b = u & 3
                sub = tile_s + (b >> 1) * SUB_BYTES + r * 128
                words = []
                for i in cutlass.range_constexpr(4):
                    chunk = ((b & 1) * 4 + i) ^ (r & 7)
                    w0, w1, w2, w3 = lds128(sub + (chunk << 4))
                    words.append(w0)
                    words.append(w1)
                    words.append(w2)
                    words.append(w3)
                m = xorsign_abs(words[0], words[1])
                for i in cutlass.range_constexpr(2, 16):
                    m = xorsign_abs(m, words[i])
                thread_amax = cute.arch.fmax(cute.math.abs(bf16_lo(m)), cute.math.abs(bf16_hi(m)))
                e, inv = e8m0_and_rcp(thread_amax * cutlass.Float32(E4M3_MAX_NORM_RCP))
                row = grow0 + r
                (sp + swizzled_scale_idx(row, (col0 >> 5) + b, n_tiles_x)).store(e.to(cutlass.Uint8))
                qw = []
                for j in cutlass.range_constexpr(8):
                    qw.append(mul_cvt_2x(words[2 * j], words[2 * j + 1], inv))
                stg256(out_i64 + (row * cols + col0 + b * 32).to(cutlass.Int64), qw)

            # ---- NVFP4 columnwise half, weight role: 2-D 16 x 16 blocks, no RHT ------
            # quantize_transpose_nvfp4.cuh:964-1110. ldmatrix.x4.trans of the block: lane
            # (gi, q) holds f0 = rows (2q, 2q+1) and f1 = rows (2q+8, 2q+9) of column gi,
            # f2 / f3 the same rows of column gi+8 (dual_cast_one_read_kernel.cu:349, the
            # (m0, m2, m1, m3) fragment order, minus the mma).
            tw = t % FLUSH_TILES
            for cbi in cutlass.range_constexpr(CBW):
                cb = cb0 + cbi
                sub = tile_s + (cb >> 2) * SUB_BYTES
                chunk = ((cb & 3) * 2 + ld_chunk_hi) ^ (ld_row & 7)
                c_a = cbi * 16 + gi                              # column within the warp's staging
                codes_as = []
                codes_bs = []
                sfs = []
                for rgi in cutlass.range_constexpr(RG):
                    f0, f1, f2, f3 = ldmatrix_x4_trans(sub + (rgi * 16 + ld_row) * 128 + (chunk << 4))
                    # block amax over the 256 elements: 8 per lane, then the full warp.
                    m = xorsign_abs(f0, f1)
                    m = xorsign_abs(m, f2)
                    m = xorsign_abs(m, f3)
                    for off in (16, 8, 4, 2, 1):
                        m = xorsign_abs(m, cute.arch.shuffle_sync_bfly(m, offset=off))
                    bamax = cute.arch.fmax(cute.math.abs(bf16_lo(m)), cute.math.abs(bf16_hi(m)))
                    # core_nvfp4.cuh:40-53, :1088-1090: sf = e4m3(min(bamax * S_enc / 6, FLT_MAX))
                    # (the same runtime value in both halves: a float constant is never an
                    # inline_ptx operand)
                    scaled = cute.arch.fmin(bamax * enc_mul, flt_max)
                    sf = f32x2_to_e4m3x2(scaled, scaled) & 0xFF
                    bsi = (lut_p + sf).load()                    # 1 / (e4m3(sf) * S_dec), via the LUT
                    # codes: byte 0 = rows (2q, 2q+1), byte 1 = rows (2q+8, 2q+9), element 0 low nibble
                    codes_as.append(mul_cvt_e2m1x4(f0, f1, bsi) & 0xFFFF)   # column gi
                    codes_bs.append(mul_cvt_e2m1x4(f2, f3, bsi) & 0xFFFF)   # column gi+8
                    sfs.append(sf)
                # Codes: one stmatrix.x2 per row-group pair. Matrix 0 = columns cbi*16 + 0..7
                # (rows addressed by lanes 0-7), matrix 1 = columns cbi*16 + 8..15 (lanes 8-15);
                # lane (gi, q) supplies bytes 4q..4q+3 of its column's 16 B pair chunk =
                # {rg even: byte q, byte q+4 codes} {rg odd: the same} -- un-interleaved at the flush.
                row_addr = cst_base + (cbi * 16 + stsm_lane) * COL_PITCH + tw * PIECE
                for p in cutlass.range_constexpr(RG // 2):
                    wa = codes_as[2 * p] | (codes_as[2 * p + 1] << 16)
                    wb = codes_bs[2 * p] | (codes_bs[2 * p + 1] << 16)
                    stsm_x2(row_addr + p * 16, wa, wb)
                # Scale bytes: the block's sf is shared by its 16 columns (TE
                # quantize_transpose_nvfp4.cuh:1300-1317 stores it per column). Lane q = 0
                # writes column c_a's RG bytes of this tile, q = 1 column c_a + 8's; q = 2, 3 idle.
                c_sel = c_a + cutlass.Int32(cutlass.select_(q == 0, cutlass.Int32(0), cutlass.Int32(8)))
                if cutlass.const_expr(RG == 4):
                    sw = sfs[0] | (sfs[1] << 8) | (sfs[2] << 16) | (sfs[3] << 24)
                    if q < 2:
                        sts32(sst_base + c_sel * SF_PITCH + tw * RG, sw)
                else:
                    sw = sfs[0] | (sfs[1] << 8)
                    if q < 2:
                        sts16(sst_base + c_sel * SF_PITCH + tw * RG, sw)

            # The ring slot is free as soon as this warp's smem reads are done; the
            # flush below reads warp-private staging only.
            cute.arch.sync_warp()
            pipe.consumer_release(cstate)
            cstate.advance()

            # ---- flush: full lines per column every WINDOW rows -------------------------
            if tw == FLUSH_TILES - 1:
                wrow0 = grow0 - (FLUSH_TILES - 1) * TILE_ROWS
                for i in cutlass.range_constexpr(NCHUNK // 32):
                    idx = i * 32 + lane
                    c = idx // CHUNKS_PER_COL
                    ch = idx % CHUNKS_PER_COL
                    w0, w1, w2, w3 = lds128(cst_base + c * COL_PITCH + ch * 16)
                    # staged chunk (bytes) = [q0: e0 e1 o0 o1][q1: ...][q2: ...][q3: ...] where
                    # e/o = rg even/odd, byte 0 = rows 2q..2q+1, byte 1 = rows 2q+8..2q+9;
                    # output = [q0.e0 q1.e0 q2.e0 q3.e0 q0.e1 q1.e1 q2.e1 q3.e1 | the same for o]
                    t01 = prmt(w0, w1, 0x5140)      # [w0.b0 w1.b0 w0.b1 w1.b1]
                    t23 = prmt(w2, w3, 0x5140)      # [w2.b0 w3.b0 w2.b1 w3.b1]
                    u01 = prmt(w0, w1, 0x7362)      # [w0.b2 w1.b2 w0.b3 w1.b3]
                    u23 = prmt(w2, w3, 0x7362)
                    o0 = prmt(t01, t23, 0x5410)     # [q0.e0 q1.e0 q2.e0 q3.e0]
                    o1 = prmt(t01, t23, 0x7632)     # [q0.e1 q1.e1 q2.e1 q3.e1]
                    o2 = prmt(u01, u23, 0x5410)     # [q0.o0 ...]
                    o3 = prmt(u01, u23, 0x7632)     # [q0.o1 ...]
                    c_global = col0 + cb0 * 16 + c
                    dst = c_global * half_rows + (wrow0 >> 1) + ch * 16
                    (cd_p + (dst >> 2)).store(
                        cutlass.Vector.from_elements((o0, o1, o2, o3), cutlass.Int32), alignment=16)
                # col scales: 4 consecutive row blocks = one 4 B word inside a 128x4 atom.
                for i in cutlass.range_constexpr(NWORDS // 32):
                    idx = i * 32 + lane
                    c = idx // WORDS_PER_COL
                    wd = idx % WORDS_PER_COL
                    w = lds32(sst_base + c * SF_PITCH + wd * 4)
                    c_global = col0 + cb0 * 16 + c
                    j = (wrow0 >> 4) + 4 * wd
                    (cs_p + (swizzled_scale_idx(c_global, j, sf_tiles_x) >> 2)).store(w)
                cute.arch.sync_warp()   # staging reads done before this warp's next write


@cute.jit
def dual_cast_one_read_ctm_host(
    x: cute.Tensor, out: cute.Tensor, scales: cute.Tensor, col_data: cute.Tensor,
    col_scales: cute.Tensor, amax_in: cute.Tensor,
    rows: cutlass.Int32, cols: cutlass.Int32, tiles_per_cta: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32, stream: cuda.CUstream,
    TILE_ROWS: cutlass.Constexpr[int], STAGES: cutlass.Constexpr[int],
    CW: cutlass.Constexpr[int], WINDOW: cutlass.Constexpr[int], MIN_CTAS: cutlass.Constexpr[int],
) -> None:
    smem_layout = ring_layout(TILE_ROWS, 2 * STAGES)
    tma_atom, tma_tensor = make_tma(x, TILE_ROWS, smem_layout)
    kern = dual_cast_one_read_ctm_kernel(
        tma_atom, tma_tensor, smem_layout, out, scales, col_data, col_scales, amax_in,
        rows, cols, tiles_per_cta, TILE_ROWS, STAGES, CW, WINDOW,
    )
    if cutlass.const_expr(MIN_CTAS > 0):
        # The register cap of __launch_bounds__((CW+1)*32, MIN_CTAS); the explicit carveout
        # keeps the DSL out of its automatic-carveout path.
        kern.launch(grid=(grid_x, grid_y, 1), block=((CW + 1) * 32, 1, 1), stream=stream,
                    min_blocks_per_mp=MIN_CTAS, preferred_smem_carveout=100)
    else:
        # The driver default: no residency knob, no carveout request.
        kern.launch(grid=(grid_x, grid_y, 1), block=((CW + 1) * 32, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_compiled: dict[tuple[int, str], Callable] = {}


def _compile(device_index: int, variant: str) -> Callable:
    """Compile once per (device, variant); rows, cols and the grid stay symbolic."""
    key = (device_index, variant)
    if key not in _compiled:
        tile_rows, stages, cw, window, min_ctas = VARIANTS[variant]
        with torch.cuda.device(device_index):
            bf, u8, f32, i32 = cutlass.BFloat16, cutlass.Uint8, cutlass.Float32, cutlass.Int32
            sym = cute.sym_int64
            # stride_order=(1, 0): a row-major torch [M, N] (runtime.py:661; 063).
            fx = make_fake_compact_tensor(bf, (sym(16, "m"), sym(16, "n")), stride_order=(1, 0),
                                          assumed_align=16)
            fout = make_fake_compact_tensor(u8, (sym(16, "no"),), assumed_align=16)
            fsc = make_fake_compact_tensor(u8, (sym(4, "ns"),), assumed_align=4)
            fcd = make_fake_compact_tensor(u8, (sym(16, "ncd"),), assumed_align=16)
            fcs = make_fake_compact_tensor(u8, (sym(4, "ncs"),), assumed_align=4)
            famax = make_fake_compact_tensor(f32, (1,), assumed_align=4)
            ints = [i32(0)] * 5
            _compiled[key] = cute.compile(
                dual_cast_one_read_ctm_host, fx, fout, fsc, fcd, fcs, famax, *ints,
                make_fake_stream(), tile_rows, stages, cw, window, min_ctas,
                options="--enable-tvm-ffi",
            )
    return _compiled[key]


def variants() -> list[str]:
    """Variant names."""
    return list(VARIANTS)


def grid(rows: int, cols: int, variant: str = "A", mode: int = 0,
         device_index: int | None = None) -> tuple[int, int, int, int]:
    """``(col_strips, row_splits, tiles_per_cta, resident_ctas)`` the host will launch."""
    tile_rows, _, _, window, min_ctas = VARIANTS[variant]
    dev = torch.cuda.current_device() if device_index is None else device_index
    # the CTA target of geometry_windows is a grid rule, not a residency pin: A0 uses A's
    return geometry_windows(rows, cols, tile_rows, window, min_ctas or 2, sm_count(dev),
                            2 if mode & MODE_OVERSUB2 else 1)


def _check(x: torch.Tensor, amax: torch.Tensor, variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"dual_cast_one_read_ctm: unknown variant {variant!r}; have {list(VARIANTS)}")
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()):
        raise ValueError("dual_cast_one_read_ctm: x must be a contiguous 2-D CUDA BF16 tensor")
    rows, cols = x.shape
    window = VARIANTS[variant][3]
    if rows % window != 0 or cols % TILE_COLS != 0:
        raise ValueError(
            f"dual_cast_one_read_ctm: rows % {window} == 0 and cols % 128 == 0 required, "
            f"got {rows}x{cols}")
    if rows * cols > INT32_MAX:
        raise ValueError("dual_cast_one_read_ctm: numel exceeds the Int32 index range")
    if x.data_ptr() % 16 != 0:
        raise ValueError("dual_cast_one_read_ctm: x must be 16 B aligned")
    if not (amax.is_cuda and amax.dtype == torch.float32 and amax.numel() == 1):
        raise ValueError("dual_cast_one_read_ctm: amax must be a CUDA fp32 [1] tensor")


def dual_cast_one_read_ctm(
    x: torch.Tensor, amax: torch.Tensor, variant: str = "A", mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One TMA-staged read of a BF16 weight: MXFP8 rowwise half + NVFP4 columnwise half.

    The contract of ``flux.dual_cast_one_read(x, rht_matrix, False, amax)``
    (``dual_format_cast.py``), bitwise, on the CUTLASS DSL.

    Args:
        x: BF16 ``[rows, cols]`` contiguous, ``rows`` a multiple of the variant's flush
            window (256 for ``A``/``A0``, 128 for ``S``, 64 for ``N``), ``cols % 128 == 0``.
        amax: FP32 ``[1]`` CUDA -- the **final** per-tensor ``max|x|`` (T6's reduced
            weight amax), complete on the current stream; read once in the prologue.
        variant: One of :data:`VARIANTS`.
        mode: Bit 1 (``MODE_OVERSUB2``) doubles the row splits.

    Returns:
        ``(mx_data, mx_scales, col_data, col_scales)`` -- E4M3 ``[rows, cols]``,
        GEMM-swizzled E8M0 ``[rows, cols/32]``, E2M1 ``[cols, rows/2]`` bytes,
        GEMM-swizzled E4M3 ``[cols, rows/16]``, all uint8.

    Raises:
        ValueError: On an unsupported input or variant (an error, never a fallback).
    """
    _check(x, amax, variant)
    dev = x.device.index
    rows, cols = x.shape
    col_strips, row_splits, tiles_per_cta, _ = grid(rows, cols, variant, mode, dev)
    tile_rows = VARIANTS[variant][0]
    # The CUDA kernel's `if (ntiles <= 0) return;` has no DSL form; it is dead here.
    assert (row_splits - 1) * tiles_per_cta < rows // tile_rows
    u8 = dict(dtype=torch.uint8, device=x.device)
    out = torch.empty(rows, cols, **u8)
    scales = torch.empty(rows, cols // 32, **u8)
    col_data = torch.empty(cols, rows // 2, **u8)
    col_scales = torch.empty(cols, rows // 16, **u8)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(dev, variant)(
        x, out.view(-1), scales.view(-1), col_data.view(-1), col_scales.view(-1), amax,
        rows, cols, tiles_per_cta, col_strips, row_splits, stream,
    )
    return out, scales, col_data, col_scales


def pick(numel: int) -> tuple[str, int]:
    """The dispatch: ``(variant, mode)`` for a weight of ``numel`` elements."""
    for upto, variant, mode in DISPATCH:
        if numel <= upto:
            return variant, mode
    return DISPATCH[-1][1], DISPATCH[-1][2]


def dual_cast_one_read_ctm_auto(
    x: torch.Tensor, amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The cast, dispatched on ``x.numel()`` (decided at trace time under graphs)."""
    variant, mode = pick(x.numel())
    return dual_cast_one_read_ctm(x, amax, variant, mode)


def live_kernel(rows: int, cols: int) -> tuple[str, tuple[int, int]]:
    """``(kernel-name substring, (grid_x, grid_y))`` the dispatch launches for a
    ``[rows, cols]`` weight on this device."""
    variant, mode = pick(rows * cols)
    cs, rs, _, _ = grid(rows, cols, variant, mode)
    return KERNEL, (cs, rs)


def verify(rows: int = 3072, cols: int = 3072) -> None:
    """Compile, run and bit-check every variant in both modes against TE's two casts.

    The ``ctm-write-kernel`` skill's entry point; the full contract (all shapes,
    graphs, the pressure test, the production entry point) is
    ``tests/test_dual_cast_one_read_ctm.py``. The reference is TE itself --
    ``tex.quantize`` with the columnwise-only weight quantizer for the NVFP4 half
    and the rowwise-only MXFP8 quantizer for the MXFP8 half -- the closed-form
    oracle the deleted CUDA kernel was itself held to.

    Args:
        rows: Rows of the BF16 probe weight (``% 256 == 0``).
        cols: Columns (``% 128 == 0``).

    Raises:
        AssertionError: If any output byte differs from TE's.
    """
    import transformer_engine.pytorch.cpp_extensions as tex
    from flat import nvfp4_ops as nv

    torch.manual_seed(rows * 31 + cols)
    w = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.02
    amax = w.abs().amax().float().reshape(1)
    nvt = tex.quantize(w, nv.role_quantizer("weight", columnwise_only=True))
    mx = nv._quantize_mx(w, nv._MX_FPROP_QUANTIZER)
    ref = (mx._rowwise_data, mx._rowwise_scale_inv, nvt._columnwise_data, nvt._columnwise_scale_inv)
    for v in VARIANTS:
        for mode in (0, MODE_OVERSUB2):
            got = dual_cast_one_read_ctm(w, amax, v, mode)
            torch.cuda.synchronize()
            for i, (g, t) in enumerate(zip(got, ref)):
                assert torch.equal(g.reshape(-1), t.reshape(-1).view(torch.uint8)), (
                    f"{v}+m{mode}: output {i} differs from TE")
    print(f"dual_cast_one_read_ctm.verify [{rows},{cols}]: {list(VARIANTS)} x modes (0, 2) "
          f"bitwise TE: PASS")


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "dual_cast_one_read_ctm(Tensor x, Tensor amax, str variant, int mode)"
    " -> (Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "dual_cast_one_read_ctm", "CUDA")
def _dual_cast_one_read_ctm_impl(
    x: torch.Tensor, amax: torch.Tensor, variant: str, mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return dual_cast_one_read_ctm(x, amax, variant, mode)
