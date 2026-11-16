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

"""TE's NVFP4 quantiser (RHT + stochastic-rounding cast) on the CUTLASS DSL.

Replaces ``transformer_engine::detail::row_col_rht_gemm_device`` -- the kernel behind every
production ``flux.te_nvfp4_quantize_with_amax`` (456 launches/step: 266 gradient casts with
both halves and stochastic rounding, 190 activation casts with the columnwise half only) --
with ONE kernel body and two constexpr switches:

* ``ROWWISE``: also emit the plain rowwise NVFP4 half (E2M1 ``[M, N/2]`` + E4M3 ``[M, N/16]``
  GEMM-swizzled scales) from the same shared-memory tile;
* ``SR``: stochastic rounding (``cvt.rs.satfinite.e2m1x4.f32``) with the random bits drawn by
  TE's Philox4x32-10 (``common/util/curanddx.hpp``) under the per-element mapping of TE's
  kernel (``row_cast_col_hadamard_transform_cast_fusion.cu`` @ ``ae34b34f``), so the emitted bytes are bitwise TE's given the same
  ``(seed, offset)``.

The columnwise half is ``onepass_act_ctm.py``'s NVFP4 half verbatim (``ldmatrix.x4.trans`` +
two ``mma.m16n8k16`` against the column-permuted ``R``, bf16 round trip, block amax, E4M3
scale from a 256-entry LUT, E2M1 codes, ``stmatrix`` staging, full-line flush every
``WINDOW`` rows); the MXFP8 half of that kernel is gone. Structure: one producer warp streams
``TILE_ROWS`` x 128 bf16 tiles as two SWIZZLE_128B TMA boxes per stage through a
``cutlass.pipeline.PipelineTmaAsync`` ring; ``CW`` consumer warps.

RNG state: the op owns an ``int64 [2]`` ``(seed, offset)`` per device (plus an ``int32 [1]``
completion counter). Every CTA reads it in its prologue; the last CTA to finish advances
``offset`` by 1024 -- what TE's ``extract_seed_and_offset`` + ``philox_cuda_state(1024)`` do per
cast -- so the op is graph-safe and launches nothing else.

Public ``nvidia-cutlass-dsl`` surface only.
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
    BOX_COLS, FLT_MAX, FP4_MAX_INV, INT32_MAX, TILE_COLS,
    _ptx, bf16_hi, bf16_lo, div_const_rn, e4m3_to_f32, f32x2_to_e4m3x2,
    geometry_windows, global_encode_scale, ldmatrix_x4_trans, lds128, lds32,
    make_tma, mma_16816, mul_cvt_e2m1x4, pack_bf16x2, prmt, ring_layout,
    sm_count, sts32, stsm_x2, swizzled_scale_idx, xorsign_abs,
)

#: variant -> (TILE_ROWS, STAGES, consumer warps, flush window rows, CTAs/SM pinned through
#: min_blocks_per_mp).
VARIANTS: dict[str, tuple[int, int, int, int, int]] = {
    "A": (64, 4, 8, 256, 2),     # B4's shipped act geometry, 288 threads
    "W": (64, 3, 8, 128, 3),     # 3 CTAs/SM
    "S": (32, 4, 4, 128, 4),     # 160-thread CTAs
    "N": (32, 4, 4, 64, 4),      # 160 threads, 64-row window: 2x the CTAs of S
    "A0": (64, 4, 8, 256, 0),    # A without the residency knob / carveout request
    "W0": (64, 3, 8, 128, 0),    # W without the knob
}
FACTORS: tuple[int, ...] = (1, 2, 4)
KERNEL = "nvfp4_rht_cast_ctm_kernel"
#: TE advances the Philox offset by this per cast (cast.cpp / quantizer.cpp: philox_cuda_state(1024)).
OFFSET_PER_CAST = 1024
#: (gtid of the row-quant warps, gtid of the col-quant warps, k_tile
#: increment applied before the row path's rng.init, same for the col path).
H1_MAPPING: tuple[int, int, int, int] = (256, 128, 1, 0)

# --- PTX ---------------------------------------------------------------------------

# TE: cvt.rs.satfinite.e2m1x4.f32 d, {in3, in2, in1, in0}, rbits (:71). Inputs are two
# packed-bf16x2 words scaled by acc, as mul_cvt_e2m1x4; output = 4 codes in the low 16 bits.
_MUL_CVT_RS_E2M1X4 = "\n".join([
    "{",
    ".reg .b16 h0, h1, h2, h3, o;",
    ".reg .b32 f0, f1, f2, f3;",
    ".reg .b64 A, B, S;",
    "mov.b64 S, { {$r2}, {$r2} };",
    "mov.b32 {h0, h1}, {$r0};",
    "mov.b32 {h2, h3}, {$r1};",
    "cvt.f32.bf16 f0, h0;",
    "cvt.f32.bf16 f1, h1;",
    "cvt.f32.bf16 f2, h2;",
    "cvt.f32.bf16 f3, h3;",
    "mov.b64 A, {f0, f1};",
    "mov.b64 B, {f2, f3};",
    "mul.f32x2 A, A, S;",
    "mul.f32x2 B, B, S;",
    "mov.b64 {f0, f1}, A;",
    "mov.b64 {f2, f3}, B;",
    "cvt.rs.satfinite.e2m1x4.f32 o, {f3, f2, f1, f0}, {$r3};",
    "cvt.u32.u16 {$w0}, o;",
    "}",
])


def _philox_ptx(rounds: int) -> str:
    """Philox4x32-``rounds`` on ctr (r0..r3), key (r4, r5) -> four words; curanddx.hpp verbatim
    (single_round: lo0,hi0 = M0*ctr.x; lo1,hi1 = M1*ctr.z; out = {hi1^ctr.y^key.x, lo1,
    hi0^ctr.w^key.y, lo0}; key += W0/W1 between rounds)."""
    lines = ["{", ".reg .b32 c0, c1, c2, c3, k0, k1, lo0, hi0, lo1, hi1, t0, t2;", ".reg .b64 p0, p1;",
             "mov.b32 c0, {$r0};", "mov.b32 c1, {$r1};", "mov.b32 c2, {$r2};", "mov.b32 c3, 0;",
             "mov.b32 k0, {$r3};", "mov.b32 k1, {$r4};"]
    for r in range(rounds):
        lines += [
            "mul.wide.u32 p0, c0, 0xD2511F53;", "mov.b64 {lo0, hi0}, p0;",
            "mul.wide.u32 p1, c2, 0xCD9E8D57;", "mov.b64 {lo1, hi1}, p1;",
            "xor.b32 t0, hi1, c1;", "xor.b32 t0, t0, k0;",
            "xor.b32 t2, hi0, c3;", "xor.b32 t2, t2, k1;",
            "mov.b32 c0, t0;", "mov.b32 c1, lo1;", "mov.b32 c2, t2;", "mov.b32 c3, lo0;",
        ]
        if r != rounds - 1:
            lines += ["add.u32 k0, k0, 0x9E3779B9;", "add.u32 k1, k1, 0xBB67AE85;"]
    lines += ["mov.b32 {$w0}, c0;", "mov.b32 {$w1}, c1;", "mov.b32 {$w2}, c2;", "mov.b32 {$w3}, c3;", "}"]
    return "\n".join(lines)


_PHILOX10 = _philox_ptx(10)
# ctr.x = off_lo + v with the carry into ctr.y (curanddx philox_state_incr, v generate4 calls).
_ADD_CC = "{ add.cc.u32 {$w0}, {$r0}, {$r1}; addc.u32 {$w1}, {$r2}, 0; }"


def mul_cvt_rs_e2m1x4(p01, p23, acc, rbits):
    """Four stochastically rounded E2M1 codes (low 16 bits) of two packed-bf16x2 words * acc."""
    return _ptx(_MUL_CVT_RS_E2M1X4, [cutlass.Int32], [p01, p23, acc, rbits])


def philox10(c0, c1, c2, k0, k1):
    """TE's Philox4x32-10 output for one counter (ctr.w = 0: the sequence is < 2^32 here, and
    no constant may be an inline_ptx operand): (x, y, z, w) as Int32 words."""
    return _ptx(_PHILOX10, [cutlass.Int32] * 4, [c0, c1, c2, k0, k1])


def rbits_for(seq, v, off_lo, off_hi, k0, k1):
    """The uint4 TE's ``rng.init(seed, seq, offset)`` followed by ``v + 1`` ``generate4()`` calls
    returns last: ctr = (off_lo + v [carry], off_hi [+carry], seq, 0), key = seed."""
    x, y = _ptx(_ADD_CC, [cutlass.Int32, cutlass.Int32], [off_lo, v, off_hi])
    return philox10(x, y, seq, k0, k1)


def sel4(words, i):
    """``words[i]`` for a lane-varying ``i`` in 0..3 (select chain; traced helper)."""
    lo = cutlass.Int32(cutlass.select_((i & 1) == 0, words[0], words[1]))
    hi = cutlass.Int32(cutlass.select_((i & 1) == 0, words[2], words[3]))
    return cutlass.Int32(cutlass.select_((i & 2) == 0, lo, hi))


@cute.kernel
def nvfp4_rht_cast_ctm_kernel(
    tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor, smem_layout: cute.ComposedLayout,
    row_data: cute.Tensor, row_scales: cute.Tensor, col_data: cute.Tensor, col_scales: cute.Tensor,
    amax_row: cute.Tensor, amax_col: cute.Tensor, rht: cute.Tensor,
    rng: cute.Tensor, counter: cute.Tensor,
    rows: cutlass.Int32, cols: cutlass.Int32, tiles_per_cta: cutlass.Int32, grid_ctas: cutlass.Int32,
    gtid_row: cutlass.Int32, gtid_col: cutlass.Int32, ktadd_row: cutlass.Int32, ktadd_col: cutlass.Int32,
    lin_stride: cutlass.Int32,
    TILE_ROWS: cutlass.Constexpr[int], STAGES: cutlass.Constexpr[int],
    CW: cutlass.Constexpr[int], WINDOW: cutlass.Constexpr[int],
    ROWWISE: cutlass.Constexpr[bool], SR: cutlass.Constexpr[bool], XCHG: cutlass.Constexpr[bool],
) -> None:
    FLUSH_TILES = WINDOW // TILE_ROWS
    RG = TILE_ROWS // 16                     # 16-row groups per tile (2 or 4)
    SUB_BYTES = TILE_ROWS * BOX_COLS * 2     # one 64-column box
    STAGE_BYTES = 2 * SUB_BYTES
    RW_UNITS = TILE_ROWS * 4 // (CW * 32)    # 32-element rowwise units per thread per tile
    CBW = 8 // CW                            # 16-column blocks per warp
    RGW = RG
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
    THREADS = (CW + 1) * 32

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    row_tiles = rows // TILE_ROWS
    tile0 = bidy * tiles_per_cta
    ntiles = cutlass.min(tiles_per_cta, row_tiles - tile0)
    col0 = bidx * TILE_COLS
    n_strips = cols >> 7

    smem = utils.SmemAllocator()
    ring = smem.allocate_array(cutlass.BFloat16, STAGES * TILE_ROWS * TILE_COLS, byte_alignment=1024)
    s_tile = cute.make_tensor(
        cute.recast_ptr(ring, smem_layout.inner, dtype=cutlass.BFloat16), smem_layout.outer,
    )
    colstage = smem.allocate_array(cutlass.Uint8, CW * COLSTAGE_W, byte_alignment=16)
    sfstage = smem.allocate_array(cutlass.Uint8, CW * SFSTAGE_W, byte_alignment=16)
    lut_c = smem.allocate_array(cutlass.Float32, 256, byte_alignment=16)
    lut_r = smem.allocate_array(cutlass.Float32, 256, byte_alignment=16)
    flag = smem.allocate_array(cutlass.Int32, 1, byte_alignment=4)
    bars = smem.allocate_array(cutlass.Int64, 2 * STAGES, byte_alignment=8)

    pipe = pipeline.PipelineTmaAsync.create(
        num_stages=STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, CW),
        tx_count=STAGE_BYTES,
        barrier_storage=bars,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )

    # ---- prologue: encode scales, LUTs (TE :714-720 / :946-969, div.rn, no fast math), RNG ----
    amax_c = cutlass.Pointer(amax_col.iterator, dtype=cutlass.Float32).load()
    s_enc_c = global_encode_scale(amax_c)
    enc_mul_c = s_enc_c * cutlass.Float32(FP4_MAX_INV)
    lutc_p = cutlass.Pointer(lut_c, dtype=cutlass.Float32)
    lutr_p = cutlass.Pointer(lut_r, dtype=cutlass.Float32)
    enc_mul_r = cutlass.Float32(0.0)
    if cutlass.const_expr(ROWWISE):
        amax_r = cutlass.Pointer(amax_row.iterator, dtype=cutlass.Float32).load()
        s_enc_r = global_encode_scale(amax_r)
        enc_mul_r = s_enc_r * cutlass.Float32(FP4_MAX_INV)
    if warp < CW:
        s_dec_c = div_const_rn(1.0, s_enc_c)
        for i in cutlass.range(tidx, 256, CW * 32):
            v = div_const_rn(1.0, e4m3_to_f32(i) * s_dec_c)
            (lutc_p + i).store(cute.arch.fmin(v, cutlass.Float32(FLT_MAX)))
        if cutlass.const_expr(ROWWISE):
            s_dec_r = div_const_rn(1.0, s_enc_r)
            for i in cutlass.range(tidx, 256, CW * 32):
                v = div_const_rn(1.0, e4m3_to_f32(i) * s_dec_r)
                (lutr_p + i).store(cute.arch.fmin(v, cutlass.Float32(FLT_MAX)))
    # RNG state (seed, offset) as four little-endian Int32 words; read before any CTA can bump it.
    seed_lo = cutlass.Int32(0)
    seed_hi = cutlass.Int32(0)
    off_lo = cutlass.Int32(0)
    off_hi = cutlass.Int32(0)
    if cutlass.const_expr(SR):
        rng32 = cutlass.Pointer(rng.iterator, dtype=cutlass.Int32)
        seed_lo = rng32.load()
        seed_hi = (rng32 + 1).load()
        off_lo = (rng32 + 2).load()
        off_hi = (rng32 + 3).load()
    cute.arch.sync_threads()

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
        ld_row = lane & 15
        ld_chunk_hi = lane >> 4
        stsm_lane = lane & 15
        sf_tiles_x = (rows >> 4) >> 2          # columnwise scales [cols, rows/16]
        rsf_tiles_x = (cols >> 4) >> 2         # rowwise scales [rows, cols/16]
        half_rows = rows >> 1
        half_cols = cols >> 1
        rd_p = cutlass.Pointer(row_data.iterator, dtype=cutlass.Int32)
        rs_p = cutlass.Pointer(row_scales.iterator, dtype=cutlass.Int16)
        cd_p = cutlass.Pointer(col_data.iterator, dtype=cutlass.Int32)
        cs_p = cutlass.Pointer(col_scales.iterator, dtype=cutlass.Int32)
        zero = (cutlass.Float32(0.0),) * 4
        group_base = lane & 28                  # first lane of this lane's gi-group

        cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
        for t in cutlass.range(ntiles):
            pipe.consumer_wait(cstate)
            tile_s = ring_base + cstate.index * STAGE_BYTES
            grow0 = (tile0 + t) * TILE_ROWS
            # TE tile coordinates of this row tile (H1): 1024-row k-tiles of 8 x 128 rows.
            kt = (grow0 & 1023) >> 7
            lin_base = (bidx + (grow0 >> 10) * n_strips) * lin_stride
            seq_row_base = gtid_row + (kt + ktadd_row) * 512 + lin_base
            seq_col_base = gtid_col + (kt + ktadd_col) * 512 + lin_base
            v_col_base = (grow0 & 127) >> 4

            # ---- rowwise NVFP4 half: thread = (row, 32-column unit) = two 16-blocks ----------
            if cutlass.const_expr(ROWWISE):
                for k in cutlass.range_constexpr(RW_UNITS):
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
                    row = grow0 + r
                    seq_r = seq_row_base + 2 * (row & 127) + (b >> 1)
                    out_words = []
                    sf_bytes = []
                    for j2 in cutlass.range_constexpr(2):
                        m = xorsign_abs(words[8 * j2], words[8 * j2 + 1])
                        for i in cutlass.range_constexpr(2, 8):
                            m = xorsign_abs(m, words[8 * j2 + i])
                        bmax = cute.arch.fmax(cute.math.abs(bf16_lo(m)), cute.math.abs(bf16_hi(m)))
                        sfv = bmax * enc_mul_r
                        sf = f32x2_to_e4m3x2(sfv, sfv) & 0xFF
                        acc = (lutr_p + sf).load()
                        sf_bytes.append(sf)
                        if cutlass.const_expr(SR):
                            rb = rbits_for(seq_r, 2 * (b & 1) + j2, off_lo, off_hi, seed_lo, seed_hi)
                            c0 = mul_cvt_rs_e2m1x4(words[8 * j2], words[8 * j2 + 1], acc, rb[0])
                            c1 = mul_cvt_rs_e2m1x4(words[8 * j2 + 2], words[8 * j2 + 3], acc, rb[1])
                            c2 = mul_cvt_rs_e2m1x4(words[8 * j2 + 4], words[8 * j2 + 5], acc, rb[2])
                            c3 = mul_cvt_rs_e2m1x4(words[8 * j2 + 6], words[8 * j2 + 7], acc, rb[3])
                        else:
                            c0 = mul_cvt_e2m1x4(words[8 * j2], words[8 * j2 + 1], acc) & 0xFFFF
                            c1 = mul_cvt_e2m1x4(words[8 * j2 + 2], words[8 * j2 + 3], acc) & 0xFFFF
                            c2 = mul_cvt_e2m1x4(words[8 * j2 + 4], words[8 * j2 + 5], acc) & 0xFFFF
                            c3 = mul_cvt_e2m1x4(words[8 * j2 + 6], words[8 * j2 + 7], acc) & 0xFFFF
                        out_words.append(c0 | (c1 << 16))
                        out_words.append(c2 | (c3 << 16))
                    dst = (row * half_cols + ((col0 + b * 32) >> 1)) >> 2
                    (rd_p + dst).store(
                        cutlass.Vector.from_elements(
                            (out_words[0], out_words[1], out_words[2], out_words[3]), cutlass.Int32),
                        alignment=16)
                    sidx = swizzled_scale_idx(row, (col0 >> 4) + 2 * b, rsf_tiles_x)
                    (rs_p + (sidx >> 1)).store((sf_bytes[0] | (sf_bytes[1] << 8)).to(cutlass.Int16))

            # ---- NVFP4 columnwise half (RHT) ------------------------------------------------
            tw = t % FLUSH_TILES
            for cbi in cutlass.range_constexpr(CBW):
                cb = cb0 + cbi
                sub = tile_s + (cb >> 2) * SUB_BYTES
                chunk = ((cb & 3) * 2 + ld_chunk_hi) ^ (ld_row & 7)
                c_a = cbi * 16 + gi
                # SR: the rbits word q of counter (column, row group) for this lane's two columns.
                rb_a = []
                rb_b = []
                if cutlass.const_expr(SR):
                    seq_a = seq_col_base + cb * 16 + gi
                    seq_b = seq_a + 8
                    if cutlass.const_expr(XCHG):
                        # holder lane (gi, q) owns counters (gi, rg = q) and (gi + 8, rg = q) (RGW = 4)
                        # or (gi + 8 * (q >> 1), rg = q & 1) (RGW = 2); 4 rotation rounds hand word q of
                        # every counter of the group to lane (gi, q).
                        if cutlass.const_expr(RGW == 4):
                            wa = rbits_for(seq_a, v_col_base + q, off_lo, off_hi, seed_lo, seed_hi)
                            wb = rbits_for(seq_b, v_col_base + q, off_lo, off_hi, seed_lo, seed_hi)
                            recv_a = []
                            recv_b = []
                            for rr in cutlass.range_constexpr(4):
                                sel = (rr + q) & 3
                                src = group_base + ((q - rr) & 3)
                                recv_a.append(cute.arch.shuffle_sync(sel4(wa, sel), offset=src))
                                recv_b.append(cute.arch.shuffle_sync(sel4(wb, sel), offset=src))
                            for rg in cutlass.range_constexpr(4):
                                rb_a.append(sel4(recv_a, (q - rg) & 3))
                                rb_b.append(sel4(recv_b, (q - rg) & 3))
                        else:
                            seq_h = cutlass.Int32(cutlass.select_((q & 2) == 0, seq_a, seq_b))
                            wh = rbits_for(seq_h, v_col_base + (q & 1), off_lo, off_hi, seed_lo, seed_hi)
                            recv = []
                            for rr in cutlass.range_constexpr(4):
                                sel = (rr + q) & 3
                                src = group_base + ((q - rr) & 3)
                                recv.append(cute.arch.shuffle_sync(sel4(wh, sel), offset=src))
                            # holder q' = 2 * (col is gi+8) + rg  ->  received in round (q - q') & 3
                            for rg in cutlass.range_constexpr(2):
                                rb_a.append(sel4(recv, (q - rg) & 3))
                                rb_b.append(sel4(recv, (q - 2 - rg) & 3))
                    else:
                        for rg in cutlass.range_constexpr(RGW):
                            wa = rbits_for(seq_a, v_col_base + rg, off_lo, off_hi, seed_lo, seed_hi)
                            wb = rbits_for(seq_b, v_col_base + rg, off_lo, off_hi, seed_lo, seed_hi)
                            rb_a.append(sel4(wa, q))
                            rb_b.append(sel4(wb, q))
                codes_as = []
                codes_bs = []
                sf_as = []
                sf_bs = []
                for rgi in cutlass.range_constexpr(RGW):
                    a0, a1, a2, a3 = ldmatrix_x4_trans(sub + (rgi * 16 + ld_row) * 128 + (chunk << 4))
                    d0, d1, d2, d3 = mma_16816((a0, a2, a1, a3), hfrag[0], hfrag[1], zero)
                    d4, d5, d6, d7 = mma_16816((a0, a2, a1, a3), hfrag[2], hfrag[3], zero)
                    pa0 = pack_bf16x2(d0, d1)
                    pa1 = pack_bf16x2(d4, d5)
                    pb0 = pack_bf16x2(d2, d3)
                    pb1 = pack_bf16x2(d6, d7)
                    ma = xorsign_abs(pa0, pa1)
                    mb = xorsign_abs(pb0, pb1)
                    ma = xorsign_abs(ma, prmt(ma, ma, 0x1032))
                    mb = xorsign_abs(mb, prmt(mb, mb, 0x1032))
                    mx = prmt(ma, mb, 0x5410)
                    mx = xorsign_abs(mx, cute.arch.shuffle_sync_bfly(mx, offset=1))
                    mx = xorsign_abs(mx, cute.arch.shuffle_sync_bfly(mx, offset=2))
                    vmax_a = cute.math.abs(bf16_lo(mx))
                    vmax_b = cute.math.abs(bf16_hi(mx))
                    sf2 = f32x2_to_e4m3x2(vmax_a * enc_mul_c, vmax_b * enc_mul_c)
                    sf_a = sf2 & 0xFF
                    sf_b = (sf2 >> 8) & 0xFF
                    acc_a = (lutc_p + sf_a).load()
                    acc_b = (lutc_p + sf_b).load()
                    if cutlass.const_expr(SR):
                        codes_a = mul_cvt_rs_e2m1x4(pa0, pa1, acc_a, rb_a[rgi])
                        codes_b = mul_cvt_rs_e2m1x4(pb0, pb1, acc_b, rb_b[rgi])
                    else:
                        codes_a = mul_cvt_e2m1x4(pa0, pa1, acc_a) & 0xFFFF
                        codes_b = mul_cvt_e2m1x4(pb0, pb1, acc_b) & 0xFFFF
                    codes_as.append(codes_a)
                    codes_bs.append(codes_b)
                    sf_as.append(sf_a)
                    sf_bs.append(sf_b)
                row_addr = cst_base + (cbi * 16 + stsm_lane) * COL_PITCH + tw * PIECE
                for p in cutlass.range_constexpr(RGW // 2):
                    wa2 = codes_as[2 * p] | (codes_as[2 * p + 1] << 16)
                    wb2 = codes_bs[2 * p] | (codes_bs[2 * p + 1] << 16)
                    stsm_x2(row_addr + p * 16, wa2, wb2)
                if cutlass.const_expr(RGW == 4):
                    swa = sf_as[0] | (sf_as[1] << 8) | (sf_as[2] << 16) | (sf_as[3] << 24)
                    swb = sf_bs[0] | (sf_bs[1] << 8) | (sf_bs[2] << 16) | (sf_bs[3] << 24)
                    is_q0 = q == 0
                    sw = cutlass.Int32(cutlass.select_(is_q0, swa, swb))
                    c_sel = c_a + cutlass.Int32(cutlass.select_(is_q0, cutlass.Int32(0), cutlass.Int32(8)))
                    if q < 2:
                        sts32(sst_base + c_sel * SF_PITCH + tw * RGW, sw)
                else:
                    # RGW = 2: two scale bytes per column per tile; a 4 B word spans two tiles.
                    swa = sf_as[0] | (sf_as[1] << 8)
                    swb = sf_bs[0] | (sf_bs[1] << 8)
                    is_q0 = q == 0
                    sw = cutlass.Int32(cutlass.select_(is_q0, swa, swb))
                    c_sel = c_a + cutlass.Int32(cutlass.select_(is_q0, cutlass.Int32(0), cutlass.Int32(8)))
                    if q < 2:
                        _ptx("{ .reg .b16 h; cvt.u16.u32 h, {$r1}; st.shared.b16 [{$r0}], h; }", [],
                             [sst_base + c_sel * SF_PITCH + tw * RGW, sw])

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
                    f0, f1, f2, f3 = lds128(cst_base + c * COL_PITCH + ch * 16)
                    o0 = prmt(f0, f1, 0x5410)
                    o1 = prmt(f2, f3, 0x5410)
                    o2 = prmt(f0, f1, 0x7632)
                    o3 = prmt(f2, f3, 0x7632)
                    c_global = col0 + cb0 * 16 + c
                    cdst = c_global * half_rows + (wrow0 >> 1) + tt * (TILE_ROWS // 2) + within * 16
                    (cd_p + (cdst >> 2)).store(
                        cutlass.Vector.from_elements((o0, o1, o2, o3), cutlass.Int32), alignment=16)
                for i in cutlass.range_constexpr(NWORDS // 32):
                    idx = i * 32 + lane
                    c = idx // WORDS_PER_COL
                    wd = idx % WORDS_PER_COL
                    fw = lds32(sst_base + c * SF_PITCH + wd * 4)
                    c_global = col0 + cb0 * 16 + c
                    j = (wrow0 >> 4) + 4 * wd
                    (cs_p + (swizzled_scale_idx(c_global, j, sf_tiles_x) >> 2)).store(fw)
                cute.arch.sync_warp()

    # ---- completion: the last CTA advances the Philox offset by one cast --------------------
    if cutlass.const_expr(SR):
        cute.arch.sync_threads()
        cp = cutlass.Pointer(counter.iterator, dtype=cutlass.Int32)
        fp = cutlass.Pointer(flag, dtype=cutlass.Int32)
        if tidx == 0:
            cute.arch.fence_acq_rel_gpu()
            prev = cute.arch.atomic_add(cp, cutlass.Int32(1))
            fp.store(prev)
        cute.arch.sync_threads()
        prev_b = fp.load()
        if prev_b == grid_ctas - 1:
            if tidx == 0:
                cute.arch.fence_acq_rel_gpu()
                rng64 = cutlass.Pointer(rng.iterator, dtype=cutlass.Int64)
                off = (rng64 + 1).load(is_volatile=True)
                (rng64 + 1).store(off + cutlass.Int64(OFFSET_PER_CAST))
                cp.store(cutlass.Int32(0))


@cute.jit
def nvfp4_rht_cast_ctm_host(
    x: cute.Tensor, row_data: cute.Tensor, row_scales: cute.Tensor, col_data: cute.Tensor,
    col_scales: cute.Tensor, amax_row: cute.Tensor, amax_col: cute.Tensor, rht: cute.Tensor,
    rng: cute.Tensor, counter: cute.Tensor,
    rows: cutlass.Int32, cols: cutlass.Int32, tiles_per_cta: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32,
    gtid_row: cutlass.Int32, gtid_col: cutlass.Int32, ktadd_row: cutlass.Int32, ktadd_col: cutlass.Int32,
    lin_stride: cutlass.Int32, stream: cuda.CUstream,
    TILE_ROWS: cutlass.Constexpr[int], STAGES: cutlass.Constexpr[int],
    CW: cutlass.Constexpr[int], WINDOW: cutlass.Constexpr[int], MIN_CTAS: cutlass.Constexpr[int],
    ROWWISE: cutlass.Constexpr[bool], SR: cutlass.Constexpr[bool], XCHG: cutlass.Constexpr[bool],
) -> None:
    smem_layout = ring_layout(TILE_ROWS, 2 * STAGES)
    tma_atom, tma_tensor = make_tma(x, TILE_ROWS, smem_layout)
    k = nvfp4_rht_cast_ctm_kernel(
        tma_atom, tma_tensor, smem_layout, row_data, row_scales, col_data, col_scales,
        amax_row, amax_col, rht, rng, counter, rows, cols, tiles_per_cta, grid_x * grid_y,
        gtid_row, gtid_col, ktadd_row, ktadd_col, lin_stride,
        TILE_ROWS, STAGES, CW, WINDOW, ROWWISE, SR, XCHG,
    )
    if cutlass.const_expr(MIN_CTAS > 0):
        k.launch(
            grid=(grid_x, grid_y, 1), block=((CW + 1) * 32, 1, 1), stream=stream,
            min_blocks_per_mp=MIN_CTAS, preferred_smem_carveout=100,
        )
    else:
        k.launch(grid=(grid_x, grid_y, 1), block=((CW + 1) * 32, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_compiled: dict[tuple, Callable] = {}
_rng_state: dict[int, torch.Tensor] = {}
_counters: dict[int, torch.Tensor] = {}


def _compile(device_index: int, variant: str, rowwise: bool, sr: bool, xchg: bool) -> Callable:
    """Compile once per (device, variant, role switches); rows, cols and the grid stay symbolic."""
    key = (device_index, variant, rowwise, sr, xchg)
    if key not in _compiled:
        tile_rows, stages, cw, window, min_ctas = VARIANTS[variant]
        with torch.cuda.device(device_index):
            bf, u8, f32, i16, i32, i64 = (cutlass.BFloat16, cutlass.Uint8, cutlass.Float32,
                                          cutlass.Int16, cutlass.Int32, cutlass.Int64)
            sym = cute.sym_int64
            fx = make_fake_compact_tensor(bf, (sym(16, "m"), sym(16, "n")), stride_order=(1, 0),
                                          assumed_align=16)
            frd = make_fake_compact_tensor(u8, (sym(16, "nrd"),), assumed_align=16)
            frs = make_fake_compact_tensor(u8, (sym(4, "nrs"),), assumed_align=4)
            fcd = make_fake_compact_tensor(u8, (sym(16, "ncd"),), assumed_align=16)
            fcs = make_fake_compact_tensor(u8, (sym(4, "ncs"),), assumed_align=4)
            famax_r = make_fake_compact_tensor(f32, (1,), assumed_align=4)
            famax_c = make_fake_compact_tensor(f32, (1,), assumed_align=4)
            frht = make_fake_compact_tensor(i16, (256,), assumed_align=16)
            frng = make_fake_compact_tensor(i64, (2,), assumed_align=16)
            fcnt = make_fake_compact_tensor(i32, (1,), assumed_align=4)
            ints = [i32(0)] * 10
            _compiled[key] = cute.compile(
                nvfp4_rht_cast_ctm_host, fx, frd, frs, fcd, fcs, famax_r, famax_c, frht, frng, fcnt,
                *ints, make_fake_stream(), tile_rows, stages, cw, window, min_ctas,
                rowwise, sr, xchg, options="--enable-tvm-ffi",
            )
    return _compiled[key]


_dummy: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _dummies(device_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if device_index not in _dummy:
        dev = torch.device("cuda", device_index)
        _dummy[device_index] = (torch.zeros(2, dtype=torch.int64, device=dev),
                                torch.zeros(1, dtype=torch.int32, device=dev))
    return _dummy[device_index]


def rng_state(device_index: int) -> torch.Tensor:
    """The op's persistent Philox ``(seed, offset)`` for ``device_index`` (int64 [2]).

    Seeded on first use from the torch CUDA generator's current state, so a run's SR stream
    is a function of ``torch.manual_seed`` as TE's is; advanced by :data:`OFFSET_PER_CAST`
    in-kernel per cast (graph replays included). Created outside any capture.
    """
    if device_index not in _rng_state:
        dev = torch.device("cuda", device_index)
        seed, offset = torch_philox_state(device_index)
        _rng_state[device_index] = torch.tensor([seed, offset], dtype=torch.int64, device=dev)
        _counters[device_index] = torch.zeros(1, dtype=torch.int32, device=dev)
    return _rng_state[device_index]


def counter(device_index: int) -> torch.Tensor:
    """The persistent self-resetting completion counter (0 between launches)."""
    rng_state(device_index)
    return _counters[device_index]


def torch_philox_state(device_index: int) -> tuple[int, int]:
    """``(seed, philox_offset)`` of the torch CUDA generator, as TE's ``philox_cuda_state`` reads
    them (``CUDAGeneratorImpl::get_state``: 8 bytes seed, 8 bytes offset)."""
    st = torch.cuda.get_rng_state(device_index)
    seed = int.from_bytes(st[0:8].numpy().tobytes(), "little", signed=True)
    offset = int.from_bytes(st[8:16].numpy().tobytes(), "little", signed=True)
    return seed, offset


def grid(rows: int, cols: int, variant: str, factor: int,
         device_index: int | None = None) -> tuple[int, int, int, int]:
    """``(col_strips, row_splits, tiles_per_cta, resident_ctas)`` the host will launch."""
    tile_rows, _, _, window, min_ctas = VARIANTS[variant]
    dev = torch.cuda.current_device() if device_index is None else device_index
    return geometry_windows(rows, cols, tile_rows, window, min_ctas or 2, sm_count(dev), factor)


def lin_stride_for(cols: int) -> int:
    """TE's ``K_TILE_MAX * 512``: ``K_TILE_MAX = ceil(min(cols, 1024) / 128)``."""
    return 512 * (-(-min(cols, 1024) // 128))


def _check(x: torch.Tensor, rht_matrix: torch.Tensor, amax_row: torch.Tensor | None,
           amax_col: torch.Tensor, variant: str, factor: int) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"nvfp4_rht_cast_ctm: unknown variant {variant!r}; have {list(VARIANTS)}")
    if factor not in FACTORS:
        raise ValueError(f"nvfp4_rht_cast_ctm: unknown grid factor {factor!r}; have {FACTORS}")
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()):
        raise ValueError("nvfp4_rht_cast_ctm: x must be a contiguous 2-D CUDA BF16 tensor")
    rows, cols = x.shape
    window = VARIANTS[variant][3]
    if rows % window != 0 or cols % TILE_COLS != 0 or rows % 1024 != 0:
        raise ValueError(
            f"nvfp4_rht_cast_ctm: rows % {window} == 0, rows % 1024 == 0 (TE's k-tile) and "
            f"cols % 128 == 0 required, got {rows}x{cols}")
    if rows * cols > INT32_MAX:
        raise ValueError("nvfp4_rht_cast_ctm: numel exceeds the Int32 index range")
    if x.data_ptr() % 16 != 0:
        raise ValueError("nvfp4_rht_cast_ctm: x must be 16 B aligned")
    if not (rht_matrix.is_cuda and rht_matrix.dtype == torch.bfloat16 and rht_matrix.is_contiguous()
            and tuple(rht_matrix.shape) == (16, 16)):
        raise ValueError("nvfp4_rht_cast_ctm: rht_matrix must be a contiguous CUDA BF16 [16, 16]")
    for a in (amax_row, amax_col):
        if a is not None and not (a.is_cuda and a.dtype == torch.float32 and a.numel() == 1):
            raise ValueError("nvfp4_rht_cast_ctm: amax must be a CUDA fp32 [1] tensor")


def nvfp4_rht_cast_ctm(
    x: torch.Tensor, rht_matrix: torch.Tensor, amax_row: torch.Tensor | None,
    amax_col: torch.Tensor, variant: str, factor: int, *, rowwise: bool, sr: bool,
    rng: torch.Tensor | None = None, cnt: torch.Tensor | None = None,
    mapping: tuple[int, int, int, int] = H1_MAPPING, xchg: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """One TMA-staged read of ``x``: the NVFP4 columnwise (RHT) half and, with ``rowwise``,
    the plain rowwise half; stochastic rounding with ``sr``.

    Args:
        x: BF16 ``[rows, cols]`` contiguous; ``rows`` a multiple of 1024 and of the variant's
            flush window, ``cols % 128 == 0``.
        rht_matrix: ``NVFP4Quantizer.rht_matrix``, BF16 ``[16, 16]``.
        amax_row: FP32 ``[1]``, ``max|x|`` (rowwise encode scale); ``None`` unless ``rowwise``.
        amax_col: FP32 ``[1]``, the post-RHT columnwise amax.
        variant: One of :data:`VARIANTS`. factor: One of :data:`FACTORS`.
        rowwise: Emit the rowwise half (the gradient role). sr: Stochastic rounding.
        rng: int64 ``[2]`` ``(seed, offset)`` the kernel reads and advances; the device's
            persistent state (:func:`rng_state`) when ``None``. Ignored unless ``sr``.
        cnt: int32 ``[1]`` completion counter; the device's when ``None``.
        mapping: ``(gtid_row, gtid_col, ktadd_row, ktadd_col)`` of TE's rbits mapping (H1).
        xchg: Use the 4-round shuffle exchange for the columnwise rbits (else every lane
            recomputes the Philox it needs; same bytes).

    Returns:
        ``(row_data, row_scales, col_data, col_scales)``; the first two ``None`` unless ``rowwise``.
    """
    _check(x, rht_matrix, amax_row if rowwise else None, amax_col, variant, factor)
    if rowwise and amax_row is None:
        raise ValueError("nvfp4_rht_cast_ctm: the rowwise half needs amax_row")
    dev = x.device.index
    rows, cols = x.shape
    col_strips, row_splits, tiles_per_cta, _ = grid(rows, cols, variant, factor, dev)
    tile_rows = VARIANTS[variant][0]
    assert (row_splits - 1) * tiles_per_cta < rows // tile_rows
    u8 = dict(dtype=torch.uint8, device=x.device)
    col_data = torch.empty(cols, rows // 2, **u8)
    col_scales = torch.empty(cols, rows // 16, **u8)
    if rowwise:
        row_data = torch.empty(rows, cols // 2, **u8)
        row_scales = torch.empty(rows, cols // 16, **u8)
    else:
        row_data = torch.empty(16, **u8)          # never written (ROWWISE is a constexpr)
        row_scales = torch.empty(4, **u8)
        amax_row = amax_col
    if sr:
        rng = rng_state(dev) if rng is None else rng
        cnt = counter(dev) if cnt is None else cnt
    else:
        # never read (SR is a constexpr): persistent per-device dummies, so an RN cast launches
        # exactly one kernel (torch.zeros here cost two FillFunctor launches per act cast)
        rng, cnt = _dummies(dev)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(dev, variant, rowwise, sr, xchg)(
        x, row_data.view(-1), row_scales.view(-1), col_data.view(-1), col_scales.view(-1),
        amax_row, amax_col, rht_matrix.view(torch.int16).view(-1), rng, cnt,
        rows, cols, tiles_per_cta, col_strips, row_splits,
        mapping[0], mapping[1], mapping[2], mapping[3], lin_stride_for(cols), stream,
    )
    if not rowwise:
        return None, None, col_data, col_scales
    return row_data, row_scales, col_data, col_scales


def te_reference(x: torch.Tensor, role: str, amax: torch.Tensor):
    """TE's cast of ``x`` for ``role`` with the amax pair supplied, plus the Philox
    ``(seed, offset)`` it consumed (read from the torch generator just before the call)."""
    from flat import nvfp4_ops as nv

    q = nv.role_quantizer(role, columnwise_only=(role == "act"))
    state = torch_philox_state(x.device.index)
    parts = torch.ops.flux.te_nvfp4_quantize_with_amax(
        x, nv._te_ops.quantizer_handle(q), amax[0:1], amax[1:2])
    return parts, state, q


def verify(rows: int = 2048, cols: int = 3072) -> None:
    """Compile, run and bit-check both roles (variant ``A``, factor 1) against TE at one shape.

    The ``ctm-write-kernel`` skill's entry point; the full contract (every variant, every
    shape, graphs, the pressure test, the census) is ``tests/test_nvfp4_rht_cast_ctm.py``.

    Raises:
        AssertionError: If any output byte differs from TE's.
    """
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from flat import nvfp4_ops as nv

    torch.manual_seed(rows * 31 + cols)
    x = (torch.randn(rows, cols, device="cuda") * 0.05).to(torch.bfloat16)
    rht = nv.role_quantizer("grad").rht_matrix
    _, _, amax = torch.ops.flux.mxfp8_rowwise_cast_amax_ctm(x, rht, True)
    for role in ("act", "grad"):
        parts, (seed, off), _ = te_reference(x, role, amax)
        st = torch.tensor([seed, off], dtype=torch.int64, device="cuda")
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        got = nvfp4_rht_cast_ctm(x, rht, amax[0:1], amax[1:2], "A", 1, rowwise=(role == "grad"),
                                 sr=(role == "grad"), rng=st, cnt=cnt)
        torch.cuda.synchronize()
        ref = (parts[0], parts[1], parts[2], parts[3])
        for i, (g, r) in enumerate(zip(got, ref)):
            if g is None:
                assert r is None, (role, i)
                continue
            assert torch.equal(g.view(-1), r.view(-1)), f"{role}: output {i} differs from TE"
        if role == "grad":
            assert int(st[1].item()) == off + OFFSET_PER_CAST, "offset not advanced by 1024"
    print(f"nvfp4_rht_cast_ctm.verify [{rows},{cols}]: act + grad bitwise TE: PASS")
