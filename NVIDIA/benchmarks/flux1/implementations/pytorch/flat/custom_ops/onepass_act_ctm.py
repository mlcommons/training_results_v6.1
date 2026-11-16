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

"""The one-read GELU-out cast (the CUDA ``onepass_act_tma_kernel.cu``) on the
CUTLASS DSL -- a port plus two levers, all in one kernel body.

Same contract, same numerics, same data movement as the CUDA kernel: one
TMA-staged read of a BF16 ``[rows, cols]`` tensor with the post-RHT amax supplied
by the producer, emitting

* ``mx_data``    E4M3 ``[rows, cols]``          MXFP8 rowwise data
* ``mx_scales``  E8M0 ``[rows, cols/32]``       GEMM-swizzled (128 x 4 atom)
* ``col_data``   E2M1 ``[cols, rows/2]`` bytes  NVFP4 columnwise data, transposed
* ``col_scales`` E4M3 ``[cols, rows/16]``       GEMM-swizzled (same atom)

Structure: a producer warp streams ``TILE_ROWS`` x
128 bf16 tiles as two SWIZZLE_128B boxes per stage through a
``cutlass.pipeline.PipelineTmaAsync`` ring; consumer warp ``w`` casts rows
``8w..8w+7`` of each tile to MXFP8 (four ``LDS.128``, one ``st.global.v8``) and
owns 16-column block ``w`` for the NVFP4 half (``ldmatrix.x4.trans`` + two
``mma.m16n8k16`` against the column-permuted ``R``), staging the transposed bytes
warp-privately and flushing full lines every ``WINDOW`` rows.

The two levers, each a variant of the same body:

* ``A3_*``: 3 stages and ``min_blocks_per_mp=3`` -- the ``t64s3w8x3`` occupancy arm
  (72-register cap) on the DSL toolchain.
* ``*_stsm``: the E2M1 codes of two 16-row groups are packed per lane into one b32
  and staged with ONE ``stmatrix.x2`` per row-group pair (2 per tile per warp,
  replacing 8 ``STS.16``); the four scale bytes of a tile are packed into one b32
  and staged with ONE predicated ``STS.32`` (replacing 8 predicated ``STS.8``). The
  flush un-interleaves each 16 B chunk with four ``PRMT``. Every emitted byte is the
  same code at the same address as the ``sts`` staging; only the staging layout
  differs.

Public ``nvidia-cutlass-dsl`` surface only (4.6.2 spellings, run on 4.8.0). This is the
only implementation of ``flux.onepass_act`` on every architecture; the geometry and grid
factor are chosen per architecture and per size in ``onepass_act.py``.
"""

from __future__ import annotations

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from flat.custom_ops.onepass_act_ctm_common import (
    BOX_COLS, E4M3_MAX_NORM_RCP, FLT_MAX, FP4_MAX_INV, INT32_MAX, TILE_COLS,
    bf16_hi, bf16_lo, div_const_rn, e4m3_to_f32, e8m0_and_rcp, f32x2_to_e4m3x2,
    geometry_windows, global_encode_scale, ldmatrix_x4_trans, lds128, lds32,
    make_tma, mma_16816, mul_cvt_2x, mul_cvt_e2m1x4, pack_bf16x2, prmt, ring_layout,
    sm_count, stg256, sts16, sts32, sts8, stsm_x2, swizzled_scale_idx, xorsign_abs,
)

#: variant -> (TILE_ROWS, STAGES, consumer warps, flush window rows, CTAs/SM pinned
#: through min_blocks_per_mp, stmatrix staging). The two geometries the dispatch in
#: ``onepass_act.py`` names.
VARIANTS: dict[str, tuple[int, int, int, int, int, bool]] = {
    "A_stsm": (64, 4, 8, 256, 2, True),     # the 2-CTA ``t64s4w8`` geometry, stmatrix staging
    "W_stsm": (64, 3, 8, 128, 3, True),     # the 3-CTA ``t64s3w8x3w128`` geometry, stmatrix staging
}
#: grid oversubscription factors the dispatch uses (x1 below
#: 2048 rows on Rubin and everywhere on GB300; x2 = the CUDA kernel's ``+o2``; x4 at the
#: B=32 shapes on Rubin, where the 1.96-wave tail of the 3-CTA grid costs 2-3 us).
FACTORS: tuple[int, ...] = (1, 2, 4)
KERNEL = "onepass_act_ctm_kernel"


@cute.kernel
def onepass_act_ctm_kernel(
    tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor, smem_layout: cute.ComposedLayout,
    out: cute.Tensor, scales: cute.Tensor, col_data: cute.Tensor, col_scales: cute.Tensor,
    amax_in: cute.Tensor, rht: cute.Tensor,
    rows: cutlass.Int32, cols: cutlass.Int32, tiles_per_cta: cutlass.Int32,
    TILE_ROWS: cutlass.Constexpr[int], STAGES: cutlass.Constexpr[int],
    CW: cutlass.Constexpr[int], WINDOW: cutlass.Constexpr[int],
    STSM: cutlass.Constexpr[bool],
) -> None:
    FLUSH_TILES = WINDOW // TILE_ROWS
    RG = TILE_ROWS // 16                     # 16-row groups per tile
    SUB_BYTES = TILE_ROWS * BOX_COLS * 2     # one 64-column box
    STAGE_BYTES = 2 * SUB_BYTES
    MX_UNITS = TILE_ROWS * 4 // (CW * 32)    # 32-element blocks per thread per tile
    CBW = 8 // CW                            # column blocks per warp
    RGW = RG                                 # row groups per warp per tile
    PIECE = RGW * 8                          # E2M1 bytes per column per tile
    COLS_W = CBW * 16                        # columns staged per warp
    COL_BYTES = FLUSH_TILES * PIECE          # per column per window
    COL_PITCH = COL_BYTES + 16
    SF_PITCH = FLUSH_TILES * RGW
    COLSTAGE_W = COLS_W * COL_PITCH
    SFSTAGE_W = COLS_W * SF_PITCH
    CHUNKS_PER_COL = COL_BYTES // 16
    NCHUNK = COLS_W * CHUNKS_PER_COL
    CHUNKS_PER_TILE = PIECE // 16
    WORDS_PER_COL = SF_PITCH // 4
    NWORDS = COLS_W * WORDS_PER_COL
    WORDS_PER_TILE = RGW // 4

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

    # S_enc / LUT (dual_cast_one_read_kernel.cu:493-499), built by the consumers.
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
        k0 = q * 2
        # B fragment of R with the COLUMN ORDER PERMUTED (onepass_act_tma_kernel.cu:225-246):
        # slot h*8 + gi carries R's column 4*(gi/2) + 2*h + (gi%2).
        hfrag = []
        for h in cutlass.range_constexpr(2):
            n = (gi >> 1) * 4 + 2 * h + (gi & 1)
            r0 = rht[k0 * 16 + n].to(cutlass.Int32) & 0xFFFF
            r1 = rht[(k0 + 1) * 16 + n].to(cutlass.Int32) & 0xFFFF
            r8 = rht[(k0 + 8) * 16 + n].to(cutlass.Int32) & 0xFFFF
            r9 = rht[(k0 + 9) * 16 + n].to(cutlass.Int32) & 0xFFFF
            hfrag.append(r0 | (r1 << 16))
            hfrag.append(r8 | (r9 << 16))
        cb0 = warp * CBW
        ring_base = ring.toint()
        cst_base = colstage.toint() + warp * COLSTAGE_W
        sst_base = sfstage.toint() + warp * SFSTAGE_W
        ld_row = lane & 15                   # ldmatrix lane map
        ld_chunk_hi = lane >> 4
        stsm_lane = lane & 15                # stmatrix row-address lane map: matrix 0 <- lanes 0-7, matrix 1 <- 8-15
        n_tiles_x = cols >> 7
        sf_tiles_x = (rows >> 4) >> 2
        half_rows = rows >> 1
        out_i64 = cutlass.Pointer(out.iterator, dtype=cutlass.Uint8).toint()
        sp = cutlass.Pointer(scales.iterator, dtype=cutlass.Uint8)
        cd_p = cutlass.Pointer(col_data.iterator, dtype=cutlass.Int32)
        cs_p = cutlass.Pointer(col_scales.iterator, dtype=cutlass.Int32)
        zero = (cutlass.Float32(0.0),) * 4

        cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
        for t in cutlass.range(ntiles):
            pipe.consumer_wait(cstate)
            tile_s = ring_base + cstate.index * STAGE_BYTES
            grow0 = (tile0 + t) * TILE_ROWS

            # ---- MXFP8 rowwise half: thread = (row, 32-element block) ----------------
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

            # ---- NVFP4 columnwise half ---------------------------------------------
            tw = t % FLUSH_TILES
            for cbi in cutlass.range_constexpr(CBW):
                cb = cb0 + cbi
                sub = tile_s + (cb >> 2) * SUB_BYTES
                chunk = ((cb & 3) * 2 + ld_chunk_hi) ^ (ld_row & 7)
                c_a = cbi * 16 + gi                              # column within the warp's staging
                codes_as = []
                codes_bs = []
                sf_as = []
                sf_bs = []
                for rgi in cutlass.range_constexpr(RGW):
                    a0, a1, a2, a3 = ldmatrix_x4_trans(sub + (rgi * 16 + ld_row) * 128 + (chunk << 4))
                    # A = x_block^T; fragments (m0, m2, m1, m3) as TE / the shipped kernel.
                    d0, d1, d2, d3 = mma_16816((a0, a2, a1, a3), hfrag[0], hfrag[1], zero)
                    d4, d5, d6, d7 = mma_16816((a0, a2, a1, a3), hfrag[2], hfrag[3], zero)
                    # Column c = gi: positions 4q..4q+3 = (d0, d1, d4, d5); column c+8: (d2, d3, d6, d7).
                    pa0 = pack_bf16x2(d0, d1)
                    pa1 = pack_bf16x2(d4, d5)
                    pb0 = pack_bf16x2(d2, d3)
                    pb1 = pack_bf16x2(d6, d7)
                    ma = xorsign_abs(pa0, pa1)
                    mb = xorsign_abs(pb0, pb1)
                    ma = xorsign_abs(ma, prmt(ma, ma, 0x1032))   # both halves = |max| of column c
                    mb = xorsign_abs(mb, prmt(mb, mb, 0x1032))
                    mx = prmt(ma, mb, 0x5410)                    # lo = column c, hi = column c+8
                    mx = xorsign_abs(mx, cute.arch.shuffle_sync_bfly(mx, offset=1))
                    mx = xorsign_abs(mx, cute.arch.shuffle_sync_bfly(mx, offset=2))
                    vmax_a = cute.math.abs(bf16_lo(mx))
                    vmax_b = cute.math.abs(bf16_hi(mx))
                    sf2 = f32x2_to_e4m3x2(vmax_a * enc_mul, vmax_b * enc_mul)
                    sf_a = sf2 & 0xFF
                    sf_b = (sf2 >> 8) & 0xFF
                    acc_a = (lut_p + sf_a).load()
                    acc_b = (lut_p + sf_b).load()
                    codes_a = mul_cvt_e2m1x4(pa0, pa1, acc_a)
                    codes_b = mul_cvt_e2m1x4(pb0, pb1, acc_b)
                    if cutlass.const_expr(STSM):
                        codes_as.append(codes_a & 0xFFFF)        # the high half duplicates the low
                        codes_bs.append(codes_b & 0xFFFF)
                        sf_as.append(sf_a)
                        sf_bs.append(sf_b)
                    else:
                        off = tw * PIECE + rgi * 8 + 2 * q
                        sfo = tw * RGW + rgi
                        sts16(cst_base + c_a * COL_PITCH + off, codes_a)
                        sts16(cst_base + (c_a + 8) * COL_PITCH + off, codes_b)
                        if q == 0:
                            sts8(sst_base + c_a * SF_PITCH + sfo, sf_a)
                            sts8(sst_base + (c_a + 8) * SF_PITCH + sfo, sf_b)
                if cutlass.const_expr(STSM):
                    # Codes: one stmatrix.x2 per row-group pair. Matrix 0 = columns cbi*16 + 0..7
                    # (rows addressed by lanes 0-7), matrix 1 = columns cbi*16 + 8..15 (lanes 8-15);
                    # lane (gi, q) supplies bytes 4q..4q+3 of its column's 16 B pair chunk =
                    # {rg even codes (2 B), rg odd codes (2 B)} -- un-interleaved at the flush.
                    row_addr = cst_base + (cbi * 16 + stsm_lane) * COL_PITCH + tw * PIECE
                    for p in cutlass.range_constexpr(RGW // 2):
                        wa = codes_as[2 * p] | (codes_as[2 * p + 1] << 16)
                        wb = codes_bs[2 * p] | (codes_bs[2 * p + 1] << 16)
                        stsm_x2(row_addr + p * 16, wa, wb)
                    # Scale bytes: the tile's RGW bytes of a column as whole words; lane q = 0
                    # writes column c's, lane q = 1 column c+8's (all four q-lanes hold the same
                    # sf after the butterfly), lanes q = 2, 3 idle.
                    for wi in cutlass.range_constexpr(RGW // 4):
                        swa = sf_as[4 * wi] | (sf_as[4 * wi + 1] << 8) | (sf_as[4 * wi + 2] << 16) | (sf_as[4 * wi + 3] << 24)
                        swb = sf_bs[4 * wi] | (sf_bs[4 * wi + 1] << 8) | (sf_bs[4 * wi + 2] << 16) | (sf_bs[4 * wi + 3] << 24)
                        is_q0 = q == 0
                        sw = cutlass.Int32(cutlass.select_(is_q0, swa, swb))
                        c_sel = c_a + cutlass.Int32(cutlass.select_(is_q0, cutlass.Int32(0), cutlass.Int32(8)))
                        if q < 2:
                            sts32(sst_base + c_sel * SF_PITCH + tw * RGW + wi * 4, sw)

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
                    tt = ch // CHUNKS_PER_TILE
                    within = ch % CHUNKS_PER_TILE
                    w0, w1, w2, w3 = lds128(cst_base + c * COL_PITCH + ch * 16)
                    if cutlass.const_expr(STSM):
                        # chunk = [q0e q0o q1e q1o q2e q2o q3e q3o] (b16) -> [q0e q1e q2e q3e q0o q1o q2o q3o]
                        o0 = prmt(w0, w1, 0x5410)
                        o1 = prmt(w2, w3, 0x5410)
                        o2 = prmt(w0, w1, 0x7632)
                        o3 = prmt(w2, w3, 0x7632)
                    else:
                        o0 = w0
                        o1 = w1
                        o2 = w2
                        o3 = w3
                    c_global = col0 + cb0 * 16 + c
                    dst = c_global * half_rows + (wrow0 >> 1) + tt * (TILE_ROWS // 2) + within * 16
                    (cd_p + (dst >> 2)).store(
                        cutlass.Vector.from_elements((o0, o1, o2, o3), cutlass.Int32), alignment=16)
                # col scales: 4 consecutive row blocks = one 4 B word inside a 128x4 atom.
                for i in cutlass.range_constexpr(NWORDS // 32):
                    idx = i * 32 + lane
                    c = idx // WORDS_PER_COL
                    wd = idx % WORDS_PER_COL
                    tt = wd // WORDS_PER_TILE
                    g = wd % WORDS_PER_TILE
                    w = lds32(sst_base + c * SF_PITCH + wd * 4)
                    c_global = col0 + cb0 * 16 + c
                    j = (wrow0 >> 4) + tt * RG + 4 * g
                    (cs_p + (swizzled_scale_idx(c_global, j, sf_tiles_x) >> 2)).store(w)
                cute.arch.sync_warp()   # staging reads done before this warp's next write


@cute.jit
def onepass_act_ctm_host(
    x: cute.Tensor, out: cute.Tensor, scales: cute.Tensor, col_data: cute.Tensor,
    col_scales: cute.Tensor, amax_in: cute.Tensor, rht: cute.Tensor,
    rows: cutlass.Int32, cols: cutlass.Int32, tiles_per_cta: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32, stream: cuda.CUstream,
    TILE_ROWS: cutlass.Constexpr[int], STAGES: cutlass.Constexpr[int],
    CW: cutlass.Constexpr[int], WINDOW: cutlass.Constexpr[int], MIN_CTAS: cutlass.Constexpr[int],
    STSM: cutlass.Constexpr[bool],
) -> None:
    smem_layout = ring_layout(TILE_ROWS, 2 * STAGES)
    tma_atom, tma_tensor = make_tma(x, TILE_ROWS, smem_layout)
    onepass_act_ctm_kernel(
        tma_atom, tma_tensor, smem_layout, out, scales, col_data, col_scales, amax_in, rht,
        rows, cols, tiles_per_cta, TILE_ROWS, STAGES, CW, WINDOW, STSM,
    ).launch(
        grid=(grid_x, grid_y, 1), block=((CW + 1) * 32, 1, 1), stream=stream,
        # The register cap of __launch_bounds__(288, MIN_CTAS); the explicit carveout
        # keeps the DSL out of its automatic-carveout path.
        min_blocks_per_mp=MIN_CTAS, preferred_smem_carveout=100,
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_compiled: dict[tuple[int, str], Callable] = {}


def _compile(device_index: int, variant: str) -> Callable:
    """Compile once per (device, variant); rows, cols and the grid stay symbolic."""
    key = (device_index, variant)
    if key not in _compiled:
        tile_rows, stages, cw, window, min_ctas, stsm = VARIANTS[variant]
        with torch.cuda.device(device_index):
            bf, u8, f32, i16, i32 = (cutlass.BFloat16, cutlass.Uint8, cutlass.Float32,
                                     cutlass.Int16, cutlass.Int32)
            sym = cute.sym_int64
            # stride_order=(1, 0): a row-major torch [M, N] (runtime.py:661; 063).
            fx = make_fake_compact_tensor(bf, (sym(16, "m"), sym(16, "n")), stride_order=(1, 0),
                                          assumed_align=16)
            fout = make_fake_compact_tensor(u8, (sym(16, "no"),), assumed_align=16)
            fsc = make_fake_compact_tensor(u8, (sym(4, "ns"),), assumed_align=4)
            fcd = make_fake_compact_tensor(u8, (sym(16, "ncd"),), assumed_align=16)
            fcs = make_fake_compact_tensor(u8, (sym(4, "ncs"),), assumed_align=4)
            famax = make_fake_compact_tensor(f32, (1,), assumed_align=4)
            frht = make_fake_compact_tensor(i16, (256,), assumed_align=16)
            ints = [i32(0)] * 5
            _compiled[key] = cute.compile(
                onepass_act_ctm_host, fx, fout, fsc, fcd, fcs, famax, frht, *ints,
                make_fake_stream(), tile_rows, stages, cw, window, min_ctas, stsm,
                options="--enable-tvm-ffi",
            )
    return _compiled[key]


def grid(rows: int, cols: int, variant: str, factor: int,
         device_index: int | None = None) -> tuple[int, int, int, int]:
    """``(col_strips, row_splits, tiles_per_cta, resident_ctas)`` the host will launch."""
    tile_rows, _, _, window, min_ctas, _ = VARIANTS[variant]
    dev = torch.cuda.current_device() if device_index is None else device_index
    return geometry_windows(rows, cols, tile_rows, window, min_ctas, sm_count(dev), factor)


def _check(x: torch.Tensor, rht_matrix: torch.Tensor, amax: torch.Tensor, variant: str,
           factor: int) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"onepass_act_ctm: unknown variant {variant!r}; have {list(VARIANTS)}")
    if factor not in FACTORS:
        raise ValueError(f"onepass_act_ctm: unknown grid factor {factor!r}; have {FACTORS}")
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()):
        raise ValueError("onepass_act_ctm: x must be a contiguous 2-D CUDA BF16 tensor")
    rows, cols = x.shape
    window = VARIANTS[variant][3]
    if rows % window != 0 or cols % TILE_COLS != 0:
        raise ValueError(
            f"onepass_act_ctm: rows % {window} == 0 and cols % 128 == 0 required, got {rows}x{cols}")
    if rows * cols > INT32_MAX:
        raise ValueError("onepass_act_ctm: numel exceeds the Int32 index range")
    if x.data_ptr() % 16 != 0:
        raise ValueError("onepass_act_ctm: x must be 16 B aligned")
    if not (rht_matrix.is_cuda and rht_matrix.dtype == torch.bfloat16 and rht_matrix.is_contiguous()
            and tuple(rht_matrix.shape) == (16, 16)):
        raise ValueError("onepass_act_ctm: rht_matrix must be a contiguous CUDA BF16 [16, 16]")
    if not (amax.is_cuda and amax.dtype == torch.float32 and amax.numel() == 1):
        raise ValueError("onepass_act_ctm: amax must be a CUDA fp32 [1] tensor")


def onepass_act_ctm(
    x: torch.Tensor, rht_matrix: torch.Tensor, amax: torch.Tensor, variant: str, factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One TMA-staged read of ``x``: MXFP8 rowwise half + NVFP4 columnwise half.

    Args:
        x: BF16 ``[rows, cols]`` contiguous, ``rows`` a multiple of the variant's
            flush window (256 for ``A_stsm``, 128 for ``W_stsm``), ``cols % 128 == 0``.
        rht_matrix: ``NVFP4Quantizer.rht_matrix``, BF16 ``[16, 16]``.
        amax: FP32 ``[1]`` CUDA, the final post-RHT columnwise amax, read once in
            the kernel prologue.
        variant: One of :data:`VARIANTS`.
        factor: One of :data:`FACTORS` -- the grid rule's oversubscription of the
            resident-CTA count (:func:`grid`). Every CTA owns whole flush windows,
            so every factor emits the same bytes at the same addresses.

    Returns:
        ``(mx_data, mx_scales, col_data, col_scales)`` as ``flux.onepass_act``.

    Raises:
        ValueError: On an unsupported input, variant or factor.
    """
    _check(x, rht_matrix, amax, variant, factor)
    dev = x.device.index
    rows, cols = x.shape
    col_strips, row_splits, tiles_per_cta, _ = grid(rows, cols, variant, factor, dev)
    tile_rows = VARIANTS[variant][0]
    assert (row_splits - 1) * tiles_per_cta < rows // tile_rows
    u8 = dict(dtype=torch.uint8, device=x.device)
    out = torch.empty(rows, cols, **u8)
    scales = torch.empty(rows, cols // 32, **u8)
    col_data = torch.empty(cols, rows // 2, **u8)
    col_scales = torch.empty(cols, rows // 16, **u8)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(dev, variant)(
        x, out.view(-1), scales.view(-1), col_data.view(-1), col_scales.view(-1), amax,
        rht_matrix.view(torch.int16).view(-1), rows, cols, tiles_per_cta, col_strips, row_splits,
        stream,
    )
    return out, scales, col_data, col_scales


def verify(rows: int = 1024, cols: int = 12288) -> None:
    """Compile, run and bit-check every variant at every factor against today's two-pass
    chain (pass 1 + ``flux.te_nvfp4_quantize_with_amax``, the contract's reference).

    The ``ctm-write-kernel`` skill's entry point; the full contract (all M, graphs,
    the pressure test, the dispatch census) is ``tests/test_onepass_act.py``.

    Raises:
        AssertionError: If any output byte differs.
    """
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from flat import nvfp4_ops as nv

    torch.manual_seed(rows * 31 + cols)
    q = nv.role_quantizer("act", columnwise_only=True)
    y = torch.ops.flux.gelu_tanh_ctm_v2_auto(
        torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16), True)
    mxd, mxs, amax2 = torch.ops.flux.mxfp8_rowwise_cast_amax_ctm(y, q.rht_matrix, True)
    parts = torch.ops.flux.te_nvfp4_quantize_with_amax(
        y, nv._te_ops.quantizer_handle(q), amax2[0:1], amax2[1:2])
    ref = (mxd, mxs, parts[2], parts[3])
    for v in VARIANTS:
        for factor in FACTORS:
            got = onepass_act_ctm(y, q.rht_matrix, amax2[1:2], v, factor)
            torch.cuda.synchronize()
            for i, (g, r) in enumerate(zip(got, ref)):
                assert torch.equal(g, r), f"{v} x{factor}: output {i} differs from today's chain"
    print(f"onepass_act_ctm.verify [{rows},{cols}]: {list(VARIANTS)} x factors {FACTORS} "
          f"bitwise today's chain: PASS")
