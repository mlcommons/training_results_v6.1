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

"""The large-M arm of the NVFP4 cast: TE's NVFP4 RHT (+ SR) cast with the Hadamard transform on the
tensor core (``tcgen05.mma``, accumulators in TMEM), TE's own structure.

Why a second body: the ``mma.sync`` arm of
``nvfp4_rht_cast_ctm.py`` spreads each 16-token column block over four lanes, so every block pays
shuffles for its amax, a Philox exchange for its rbits and a transposed staging pass -- 2.07x TE's
instruction count, ALU-pipe bound at 70 % issue. Here, as in TE's ``row_col_rht_gemm_device``, one
MMA per 16 tokens (M = 128 hidden columns, N = 16 positions, K = 16 tokens, MN-major A straight
from the TMA-swizzled tile, ``R^T`` as a K-major B) leaves the 16 transformed values of one hidden
column in ONE thread's ``tcgen05.ld``: the block amax needs no shuffle, the 16 codes are one 8 B
store into the transposed output, one Philox per block feeds one thread. The rowwise half (grad
role) reads the same shared tile as before.

Roles, RNG state, amax handling, the H1 rbits mapping and the byte layouts are identical to
``nvfp4_rht_cast_ctm.py`` (same helpers), so the contract test is the same: bitwise TE.

Warp roles: 0 TMA producer, 1 MMA issuer (+ TMEM owner), 2-5 columnwise epilogue (TMEM lanes
``(warp % 4) * 32 + lane`` = hidden column), 6.. rowwise epilogue (``RW`` warps, grad only).
Tile 128 tokens x 128 columns per stage (32 KB), ``STAGES`` stages, two 128-column TMEM
accumulator stages (256 columns -> two CTAs per SM fit the 512).

Public ``nvidia-cutlass-dsl`` surface: ``cutlass.experimental.primitives`` (mbarrier, TMA,
tcgen05) + ``cutlass.experimental.cuda`` TensorMap, on the pattern of the primitives examples
``tcgen05/1cta_mma_mn_major.py`` and ``tcgen05/1cta_mma_pipeline.py``.
"""

from __future__ import annotations

from typing import Callable

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as ecuda
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

from flat.custom_ops.nvfp4_rht_cast_ctm import (
    H1_MAPPING, OFFSET_PER_CAST, _dummies, counter, lin_stride_for, mul_cvt_rs_e2m1x4, rbits_for,
    rng_state,
)
from flat.custom_ops.onepass_act_ctm_common import (
    FLT_MAX, FP4_MAX_INV, INT32_MAX, TILE_COLS, _ptx, bf16_hi, bf16_lo, div_const_rn, e4m3_to_f32,
    f32x2_to_e4m3x2, global_encode_scale, lds128, mul_cvt_e2m1x4, pack_bf16x2, sm_count,
    swizzled_scale_idx, xorsign_abs,
)

KERNEL = "nvfp4_rht_cast_tc_kernel"
TILE_ROWS = 128
STAGE_ELEMS = TILE_ROWS * TILE_COLS          # bf16 per stage
BOX_ELEMS = TILE_ROWS * 64                   # one 64-column box
STAGE_BYTES = STAGE_ELEMS * 2
#: TMEM accumulator stages per variant: 2 (256 columns -> 2 CTAs/SM) or 1 (128 -> up to 4 CTAs/SM)
A_LBO = BOX_ELEMS * 2                        # second 64-column group of the MN-major tile
A_SBO = 8 * 64 * 2                           # eight K rows of 128 B
A_KSTEP = 16 * 64 * 2                        # 16 tokens
#: (STAGES, rowwise warps, CTAs/SM pinned, column-epilogue warpgroups) per variant
#: BAND (5th field): tile order. False = each CTA walks a contiguous run of row tiles down ONE column
#: strip (the mma.sync arm's order); True = TE's order: tiles numbered row-band-major
#: (strip fastest), CTA b takes tiles b, b + grid, b + 2 grid, ... so the whole machine sweeps one
#: 128-row band of the tensor at a time (in situ, TE reads its producer-fed input 25 % faster
#: than cold at [16384,12288]; the strip-major DSL kernel does not -- H2: the tile order).
#: fields: (STAGES, rowwise warps, CTAs/SM pinned, column warpgroups, BAND, ACC_STAGES, PAIR)
#: PAIR: two tcgen05.ld in flight per wait in the column epilogue (it cost T 24 % -- 124 regs, spills
#: the loaded vectors -- so it is a knob, default off).
VARIANTS: dict[str, tuple[int, int, int, int, bool, int, bool]] = {
    "T": (3, 8, 2, 1, False, 2, False),     # 3 x 32 KB ring, 4 column warps, 8 rowwise warps -> 448 threads (grad) / 192 (act)
    "T2": (3, 8, 2, 2, False, 2, False),    # two column warpgroups, each half the blocks of a tile -> 576 / 320 threads
    "T2p": (3, 8, 2, 2, False, 2, True),    # T2 with paired loads
    "T4": (4, 8, 1, 1, False, 2, False),    # 4 stages, one CTA per SM
    "TB": (3, 8, 2, 1, True, 2, False),     # T with the band-major tile order
    "TB2": (3, 8, 2, 2, True, 2, False),    # T2 with the band-major tile order
    "T3": (2, 8, 3, 2, False, 1, False),    # 2 stages, one accumulator stage, THREE CTAs/SM (64 KB smem, 128 TMEM cols each)
    "TB3": (2, 8, 3, 2, True, 1, False),    # T3 + band order
}
FACTORS: tuple[int, ...] = (1, 2)
_STS16 = "{ .reg .b16 h; cvt.u16.u32 h, {$r1}; st.shared.b16 [{$r0}], h; }"


@cute.kernel
def nvfp4_rht_cast_tc_kernel(
    tma: cutlass.GridConstant[ecuda.TensorMap],
    row_data: cute.Tensor, row_scales: cute.Tensor, col_data: cute.Tensor, col_scales: cute.Tensor,
    amax_row: cute.Tensor, amax_col: cute.Tensor, rht: cute.Tensor,
    rng: cute.Tensor, cnt: cute.Tensor,
    rows: cutlass.Int32, cols: cutlass.Int32, tiles_per_cta: cutlass.Int32, grid_ctas: cutlass.Int32,
    gtid_row: cutlass.Int32, gtid_col: cutlass.Int32, ktadd_row: cutlass.Int32, ktadd_col: cutlass.Int32,
    lin_stride: cutlass.Int32,
    STAGES: cutlass.Constexpr[int], RW: cutlass.Constexpr[int], CG: cutlass.Constexpr[int],
    ROWWISE: cutlass.Constexpr[bool], SR: cutlass.Constexpr[bool], BAND: cutlass.Constexpr[bool],
    ACC_STAGES: cutlass.Constexpr[int], PAIR: cutlass.Constexpr[bool],
) -> None:
    TMEM_COLS = 128 * ACC_STAGES
    COLW = 4 * CG                            # column-epilogue warps (warps 2 .. 2 + COLW - 1)
    THREADS = (2 + COLW + RW) * 32
    RW_UNITS = (TILE_ROWS * 4) // (RW * 32) if RW > 0 else 0
    NB = 8 // CG                             # 16-token blocks per column thread per tile

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    row_tiles = rows // TILE_ROWS
    n_strips = cols >> 7
    if cutlass.const_expr(BAND):
        # 1-D grid; tile t of this CTA is linear tile bid + t * grid (strip fastest)
        bid = bidx
        ntiles = cutlass.min(tiles_per_cta, (row_tiles * n_strips - bid + grid_ctas - 1) // grid_ctas)
        tile0 = cutlass.Int32(0)
        col0 = cutlass.Int32(0)
    else:
        tile0 = bidy * tiles_per_cta
        ntiles = cutlass.min(tiles_per_cta, row_tiles - tile0)
        col0 = bidx * TILE_COLS
        bid = cutlass.Int32(0)

    # prims-native shared memory (cutlass.Array in the smem space, as the primitives examples):
    # the prims descriptor / TMA / mbarrier / tcgen05 calls take these directly.
    ring = cutlass.Array(cutlass.BFloat16, STAGES * STAGE_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)
    bmat = cutlass.Array(cutlass.BFloat16, 256, space=cutlass.AddressSpace.smem, alignment=1024)
    lut_c = cutlass.Array(cutlass.Float32, 256, space=cutlass.AddressSpace.smem, alignment=16)
    lut_r = cutlass.Array(cutlass.Float32, 256, space=cutlass.AddressSpace.smem, alignment=16)
    flag = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    tmem_slot = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    bars = cutlass.Array(cutlass.Int64, 2 * STAGES + 2 * ACC_STAGES, space=cutlass.AddressSpace.smem, alignment=8)

    # ---- barriers (warp 0), TMEM (warp 1), B = R^T in the canonical K-major core-matrix layout ----
    if warp == 0:
        if prims.elect_sync():
            for s in cutlass.range_constexpr(STAGES):
                prims.mbarrier_init(bars.data_ptr(s), 1)
                prims.mbarrier_init(bars.data_ptr(STAGES + s), 1 + RW)
            for a in cutlass.range_constexpr(ACC_STAGES):
                prims.mbarrier_init(bars.data_ptr(2 * STAGES + a), 1)
                prims.mbarrier_init(bars.data_ptr(2 * STAGES + ACC_STAGES + a), COLW)
    prims.fence_mbarrier_init()
    if warp == 1:
        prims.tcgen05_alloc(tmem_slot, TMEM_COLS, group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    bmat_base = bmat.data_ptr().toint()
    for i in cutlass.range(tidx, 256, THREADS):
        k = i >> 4                       # R[k][n] -> B[n][k]
        n = i & 15
        val = rht[i].to(cutlass.Int32) & 0xFFFF
        off = (n >> 3) * 256 + (k >> 3) * 128 + (n & 7) * 16 + (k & 7) * 2
        _ptx(_STS16, [], [bmat_base + off, val])
    # encode scales and LUTs (TE :714-720 / :946-969, div.rn, no fast math)
    amax_c = cutlass.Pointer(amax_col.iterator, dtype=cutlass.Float32).load()
    s_enc_c = global_encode_scale(amax_c)
    enc_mul_c = s_enc_c * cutlass.Float32(FP4_MAX_INV)
    enc_mul_r = cutlass.Float32(0.0)
    s_dec_c = div_const_rn(1.0, s_enc_c)
    for i in cutlass.range(tidx, 256, THREADS):
        v = div_const_rn(1.0, e4m3_to_f32(i) * s_dec_c)
        lut_c[i] = cute.arch.fmin(v, cutlass.Float32(FLT_MAX))
    if cutlass.const_expr(ROWWISE):
        amax_r = cutlass.Pointer(amax_row.iterator, dtype=cutlass.Float32).load()
        s_enc_r = global_encode_scale(amax_r)
        enc_mul_r = s_enc_r * cutlass.Float32(FP4_MAX_INV)
        s_dec_r = div_const_rn(1.0, s_enc_r)
        for i in cutlass.range(tidx, 256, THREADS):
            v = div_const_rn(1.0, e4m3_to_f32(i) * s_dec_r)
            lut_r[i] = cute.arch.fmin(v, cutlass.Float32(FLT_MAX))
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
    cute.arch.fence_view_async_shared()      # B and the ring are read by the async (tensor) proxy
    prims.barrier_cta_sync(0)
    tmem_addr = tmem_slot.load()
    tmem_base_col = tmem_addr & 0xFFFF

    ring_base = ring.data_ptr().toint()
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cutlass.BFloat16, b_dtype=cutlass.BFloat16,
        n_dim=16, m_dim=128, a_major=1, b_major=0,
    )
    desc_b = prims.Tcgen05SmemDesc.build(
        start_address=bmat, leading_byte_offset=128, stride_byte_offset=256,
        layout=prims.Tcgen05SmemSwizzle.NONE,
    )
    desc_a_root = prims.Tcgen05SmemDesc.build(
        start_address=ring, leading_byte_offset=A_LBO, stride_byte_offset=A_SBO,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )

    if warp == 0:
        # ---- TMA producer: two 64-column boxes per 128-row tile onto one barrier ----
        tma_ptr = tma.get_ptr()
        for t in cutlass.range(ntiles):
            s = t % STAGES
            ph = ((t // STAGES) & 1) ^ 1
            while not prims.mbarrier_try_wait_parity(bars.data_ptr(STAGES + s), ph, time_limit=10_000_000):
                pass
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(bars.data_ptr(s), STAGE_BYTES)
                if cutlass.const_expr(BAND):
                    lin_t = bid + t * grid_ctas
                    row0 = (lin_t // n_strips) * TILE_ROWS
                    pc0 = (lin_t % n_strips) * TILE_COLS
                else:
                    row0 = (tile0 + t) * TILE_ROWS
                    pc0 = col0
                prims.cp_async_bulk_tensor_shared_cta_global(
                    ring.data_ptr(s * STAGE_ELEMS), tma_ptr, [pc0, row0], bars.data_ptr(s))
                prims.cp_async_bulk_tensor_shared_cta_global(
                    ring.data_ptr(s * STAGE_ELEMS + BOX_ELEMS), tma_ptr, [pc0 + 64, row0], bars.data_ptr(s))
    elif warp == 1:
        # ---- MMA issuer: eight 128x16x16 products per tile, one per 16-token block ----
        for t in cutlass.range(ntiles):
            s = t % STAGES
            a = t % ACC_STAGES
            while not prims.mbarrier_try_wait_parity(bars.data_ptr(s), (t // STAGES) & 1, time_limit=10_000_000):
                pass
            while not prims.mbarrier_try_wait_parity(bars.data_ptr(2 * STAGES + ACC_STAGES + a), ((t // ACC_STAGES) & 1) ^ 1, time_limit=10_000_000):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            desc_a_stage = desc_a_root.advance_start_address(cutlass.Int64(s * STAGE_BYTES))
            for b in cutlass.range_constexpr(8):
                if prims.elect_sync():
                    d_ptr = prims.make_tmem_ptr(tmem_addr + a * 128 + b * 16, cutlass.Float32)
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, d_ptr,
                        desc_a_stage.advance_start_address(cutlass.Int64(b * A_KSTEP)), desc_b, idesc,
                        False,
                    )
            if prims.elect_sync():
                prims.tcgen05_commit(bars.data_ptr(2 * STAGES + a), group=prims.CTAGroup.CTA_1)
                prims.tcgen05_commit(bars.data_ptr(STAGES + s), group=prims.CTAGroup.CTA_1)
    elif warp < 2 + COLW:
        # ---- columnwise epilogue: thread = hidden column (warp % 4) * 32 + lane; warpgroup g takes
        # blocks g*NB .. g*NB+NB-1 of every tile (TMEM lanes are per warp position, columns are free) ----
        sp = warp & 3
        g = (warp - 2) >> 2
        h = sp * 32 + lane
        half_rows = rows >> 1
        sf_tiles_x = (rows >> 4) >> 2
        cd_p = cutlass.Pointer(col_data.iterator, dtype=cutlass.Int32)
        cs_p = cutlass.Pointer(col_scales.iterator, dtype=cutlass.Int32)
        for t in cutlass.range(ntiles):
            a = t % ACC_STAGES
            if cutlass.const_expr(BAND):
                lin_t = bid + t * grid_ctas
                row0 = (lin_t // n_strips) * TILE_ROWS
                strip = lin_t % n_strips
            else:
                row0 = (tile0 + t) * TILE_ROWS
                strip = bidx
            h_global = strip * TILE_COLS + h
            kt = (row0 & 1023) >> 7
            lin_base = (strip + (row0 >> 10) * n_strips) * lin_stride
            seq_c = gtid_col + h + (kt + ktadd_col) * 512 + lin_base
            while not prims.mbarrier_try_wait_parity(bars.data_ptr(2 * STAGES + a), (t // ACC_STAGES) & 1, time_limit=10_000_000):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            code_words = []
            sfs = []
            PB = 2 if PAIR else 1
            for pb in cutlass.range_constexpr(NB // PB):
                # PAIR: two 16-value loads in flight per wait; else load-wait-process per block
                vs = []
                for q2 in cutlass.range_constexpr(PB):
                    bb = g * NB + PB * pb + q2
                    tptr = prims.make_tmem_ptr_from_warp_row_col(
                        tmem_addr, sp, tmem_base_col + a * 128 + bb * 16, cutlass.Float32)
                    vs.append(prims.tcgen05_ld("32x32b", tptr, num=16))
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                for q2 in cutlass.range_constexpr(PB):
                    bb = g * NB + PB * pb + q2
                    v = vs[q2]
                    words = []
                    for i in cutlass.range_constexpr(8):
                        words.append(pack_bf16x2(v[2 * i], v[2 * i + 1]))     # TE's bf16 round trip
                    m = xorsign_abs(words[0], words[1])
                    for i in cutlass.range_constexpr(2, 8):
                        m = xorsign_abs(m, words[i])
                    bmax = cute.arch.fmax(cute.math.abs(bf16_lo(m)), cute.math.abs(bf16_hi(m)))
                    sfv = bmax * enc_mul_c
                    sf = f32x2_to_e4m3x2(sfv, sfv) & 0xFF
                    acc = lut_c[sf]
                    sfs.append(sf)
                    if cutlass.const_expr(SR):
                        rb = rbits_for(seq_c, cutlass.Int32(bb) + (row0 & 127) // 16, off_lo, off_hi, seed_lo, seed_hi)
                        c0 = mul_cvt_rs_e2m1x4(words[0], words[1], acc, rb[0])
                        c1 = mul_cvt_rs_e2m1x4(words[2], words[3], acc, rb[1])
                        c2 = mul_cvt_rs_e2m1x4(words[4], words[5], acc, rb[2])
                        c3 = mul_cvt_rs_e2m1x4(words[6], words[7], acc, rb[3])
                    else:
                        c0 = mul_cvt_e2m1x4(words[0], words[1], acc) & 0xFFFF
                        c1 = mul_cvt_e2m1x4(words[2], words[3], acc) & 0xFFFF
                        c2 = mul_cvt_e2m1x4(words[4], words[5], acc) & 0xFFFF
                        c3 = mul_cvt_e2m1x4(words[6], words[7], acc) & 0xFFFF
                    code_words.append(c0 | (c1 << 16))
                    code_words.append(c2 | (c3 << 16))
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_warp()
            if lane == 0:
                prims.mbarrier_arrive(bars.data_ptr(2 * STAGES + ACC_STAGES + a))
            # 8 B of codes per block, NB blocks contiguous: 16 B stores
            dst = (h_global * half_rows + (row0 >> 1) + g * NB * 8) >> 2
            for k in cutlass.range_constexpr(NB // 2):
                (cd_p + dst + 4 * k).store(
                    cutlass.Vector.from_elements(
                        (code_words[4 * k], code_words[4 * k + 1], code_words[4 * k + 2], code_words[4 * k + 3]),
                        cutlass.Int32), alignment=16)
            j0 = (row0 >> 4) + g * NB
            for k in cutlass.range_constexpr(NB // 4):
                w = sfs[4 * k] | (sfs[4 * k + 1] << 8) | (sfs[4 * k + 2] << 16) | (sfs[4 * k + 3] << 24)
                (cs_p + (swizzled_scale_idx(h_global, j0 + 4 * k, sf_tiles_x) >> 2)).store(w)
    else:
        # ---- rowwise epilogue (grad): thread = (row, 32-column unit) from the shared tile ----
        if cutlass.const_expr(ROWWISE):
            rtid = tidx - (2 + COLW) * 32
            half_cols = cols >> 1
            rsf_tiles_x = (cols >> 4) >> 2
            rd_p = cutlass.Pointer(row_data.iterator, dtype=cutlass.Int32)
            rs_p = cutlass.Pointer(row_scales.iterator, dtype=cutlass.Int16)
            for t in cutlass.range(ntiles):
                s = t % STAGES
                if cutlass.const_expr(BAND):
                    lin_t = bid + t * grid_ctas
                    row0 = (lin_t // n_strips) * TILE_ROWS
                    strip = lin_t % n_strips
                else:
                    row0 = (tile0 + t) * TILE_ROWS
                    strip = bidx
                cc0 = strip * TILE_COLS          # (not c0: that name is a code word below)
                kt = (row0 & 1023) >> 7
                lin_base = (strip + (row0 >> 10) * n_strips) * lin_stride
                seq_row_base = gtid_row + (kt + ktadd_row) * 512 + lin_base
                while not prims.mbarrier_try_wait_parity(bars.data_ptr(s), (t // STAGES) & 1, time_limit=10_000_000):
                    pass
                tile_s = ring_base + s * STAGE_BYTES
                for k in cutlass.range_constexpr(RW_UNITS):
                    u = rtid + k * RW * 32
                    r = u >> 2
                    b = u & 3
                    sub = tile_s + (b >> 1) * (BOX_ELEMS * 2) + r * 128
                    words = []
                    for i in cutlass.range_constexpr(4):
                        chunk = ((b & 1) * 4 + i) ^ (r & 7)
                        w0, w1, w2, w3 = lds128(sub + (chunk << 4))
                        words.append(w0)
                        words.append(w1)
                        words.append(w2)
                        words.append(w3)
                    row = row0 + r
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
                        acc = lut_r[sf]
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
                    dst = (row * half_cols + ((cc0 + b * 32) >> 1)) >> 2
                    (rd_p + dst).store(
                        cutlass.Vector.from_elements(
                            (out_words[0], out_words[1], out_words[2], out_words[3]), cutlass.Int32),
                        alignment=16)
                    sidx = swizzled_scale_idx(row, (cc0 >> 4) + 2 * b, rsf_tiles_x)
                    (rs_p + (sidx >> 1)).store((sf_bytes[0] | (sf_bytes[1] << 8)).to(cutlass.Int16))
                cute.arch.sync_warp()
                if lane == 0:
                    prims.mbarrier_arrive(bars.data_ptr(STAGES + s))

    # ---- teardown: TMEM back, and the last CTA advances the Philox offset by one cast ----
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 1:
        prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_addr, cutlass.Float32), TMEM_COLS, group="cta_1")
    if cutlass.const_expr(SR):
        cp = cutlass.Pointer(cnt.iterator, dtype=cutlass.Int32)
        if tidx == 0:
            cute.arch.fence_acq_rel_gpu()
            prev = cute.arch.atomic_add(cp, cutlass.Int32(1))
            flag[0] = prev
        prims.barrier_cta_sync(0)
        prev_b = flag.load()
        if prev_b == grid_ctas - 1:
            if tidx == 0:
                cute.arch.fence_acq_rel_gpu()
                rng64 = cutlass.Pointer(rng.iterator, dtype=cutlass.Int64)
                off = (rng64 + 1).load(is_volatile=True)
                (rng64 + 1).store(off + cutlass.Int64(OFFSET_PER_CAST))
                cp.store(cutlass.Int32(0))


@cute.jit
def nvfp4_rht_cast_tc_host(
    x: cute.Tensor, row_data: cute.Tensor, row_scales: cute.Tensor, col_data: cute.Tensor,
    col_scales: cute.Tensor, amax_row: cute.Tensor, amax_col: cute.Tensor, rht: cute.Tensor,
    rng: cute.Tensor, cnt: cute.Tensor,
    rows: cutlass.Int32, cols: cutlass.Int32, tiles_per_cta: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32,
    gtid_row: cutlass.Int32, gtid_col: cutlass.Int32, ktadd_row: cutlass.Int32, ktadd_col: cutlass.Int32,
    lin_stride: cutlass.Int32, stream: cuda_drv.CUstream,
    STAGES: cutlass.Constexpr[int], RW: cutlass.Constexpr[int], MIN_CTAS: cutlass.Constexpr[int],
    CG: cutlass.Constexpr[int], ROWWISE: cutlass.Constexpr[bool], SR: cutlass.Constexpr[bool],
    BAND: cutlass.Constexpr[bool], ACC_STAGES: cutlass.Constexpr[int], PAIR: cutlass.Constexpr[bool],
) -> None:
    tma = ecuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(), dtype=cutlass.BFloat16,
        global_dims=[x.shape[1], x.shape[0]], global_strides=[x.shape[1] * 2 // 16],
        box_dims=[64, TILE_ROWS], swizzle=ecuda.TensorMapSwizzle.s128b,
    )
    k = nvfp4_rht_cast_tc_kernel(
        tma, row_data, row_scales, col_data, col_scales, amax_row, amax_col, rht, rng, cnt,
        rows, cols, tiles_per_cta, grid_x * grid_y, gtid_row, gtid_col, ktadd_row, ktadd_col,
        lin_stride, STAGES, RW, CG, ROWWISE, SR, BAND, ACC_STAGES, PAIR,
    )
    threads = (2 + 4 * CG + RW) * 32
    if cutlass.const_expr(MIN_CTAS > 0):
        k.launch(grid=(grid_x, grid_y, 1), block=(threads, 1, 1), stream=stream,
                 min_blocks_per_mp=MIN_CTAS, preferred_smem_carveout=100)
    else:
        k.launch(grid=(grid_x, grid_y, 1), block=(threads, 1, 1), stream=stream)


_compiled: dict[tuple, Callable] = {}


def _compile(device_index: int, variant: str, rowwise: bool, sr: bool) -> Callable:
    key = (device_index, variant, rowwise, sr)
    if key not in _compiled:
        stages, rw, min_ctas, cg, band, acc_stages, pair = VARIANTS[variant]
        rw = rw if rowwise else 0
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
                nvfp4_rht_cast_tc_host, fx, frd, frs, fcd, fcs, famax_r, famax_c, frht, frng, fcnt,
                *ints, make_fake_stream(), stages, rw, min_ctas, cg, rowwise, sr, band, acc_stages, pair,
                options="--enable-tvm-ffi",
            )
    return _compiled[key]


def grid(rows: int, cols: int, variant: str, factor: int, device_index: int | None = None) -> tuple[int, int, int]:
    """``(grid_x, grid_y, tiles_per_cta)``: strip-major variants launch ``(strips, row_splits)``, band-major
    ones a 1-D grid of ~``factor`` x the resident CTAs walking the tiles strip-fastest."""
    _, _, min_ctas, _, band, _, _ = VARIANTS[variant]
    dev = torch.cuda.current_device() if device_index is None else device_index
    strips = cols // TILE_COLS
    row_tiles = rows // TILE_ROWS
    resident = (min_ctas or 2) * sm_count(dev) * factor
    if band:
        total = strips * row_tiles
        ctas = max(1, min(total, resident))
        tpc = -(-total // ctas)
        ctas = -(-total // tpc)
        return ctas, 1, tpc
    splits = max(1, min(row_tiles, resident // strips))
    tpc = -(-row_tiles // splits)
    splits = -(-row_tiles // tpc)
    return strips, splits, tpc


def _check(x, rht_matrix, amax_row, amax_col, variant, factor):
    if variant not in VARIANTS:
        raise ValueError(f"nvfp4_rht_cast_tc: unknown variant {variant!r}; have {list(VARIANTS)}")
    if factor not in FACTORS:
        raise ValueError(f"nvfp4_rht_cast_tc: unknown grid factor {factor!r}; have {FACTORS}")
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()):
        raise ValueError("nvfp4_rht_cast_tc: x must be a contiguous 2-D CUDA BF16 tensor")
    rows, cols = x.shape
    if rows % 1024 != 0 or cols % TILE_COLS != 0:
        raise ValueError(f"nvfp4_rht_cast_tc: rows % 1024 == 0 and cols % 128 == 0 required, got {rows}x{cols}")
    if rows * cols > INT32_MAX:
        raise ValueError("nvfp4_rht_cast_tc: numel exceeds the Int32 index range")
    if not (rht_matrix.is_cuda and rht_matrix.dtype == torch.bfloat16 and rht_matrix.is_contiguous()
            and tuple(rht_matrix.shape) == (16, 16)):
        raise ValueError("nvfp4_rht_cast_tc: rht_matrix must be a contiguous CUDA BF16 [16, 16]")
    for a in (amax_row, amax_col):
        if a is not None and not (a.is_cuda and a.dtype == torch.float32 and a.numel() == 1):
            raise ValueError("nvfp4_rht_cast_tc: amax must be a CUDA fp32 [1] tensor")


def nvfp4_rht_cast_tc(
    x: torch.Tensor, rht_matrix: torch.Tensor, amax_row: torch.Tensor | None, amax_col: torch.Tensor,
    variant: str, factor: int, *, rowwise: bool, sr: bool,
    rng: torch.Tensor | None = None, cnt: torch.Tensor | None = None,
    mapping: tuple[int, int, int, int] = H1_MAPPING,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """The tensor-core arm; same interface and outputs as ``nvfp4_rht_cast_ctm.nvfp4_rht_cast_ctm``."""
    _check(x, rht_matrix, amax_row if rowwise else None, amax_col, variant, factor)
    if rowwise and amax_row is None:
        raise ValueError("nvfp4_rht_cast_tc: the rowwise half needs amax_row")
    dev = x.device.index
    rows, cols = x.shape
    strips, splits, tpc = grid(rows, cols, variant, factor, dev)
    u8 = dict(dtype=torch.uint8, device=x.device)
    col_data = torch.empty(cols, rows // 2, **u8)
    col_scales = torch.empty(cols, rows // 16, **u8)
    if rowwise:
        row_data = torch.empty(rows, cols // 2, **u8)
        row_scales = torch.empty(rows, cols // 16, **u8)
    else:
        row_data = torch.empty(16, **u8)
        row_scales = torch.empty(4, **u8)
        amax_row = amax_col
    if sr:
        rng = rng_state(dev) if rng is None else rng
        cnt = counter(dev) if cnt is None else cnt
    else:
        rng, cnt = _dummies(dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(dev, variant, rowwise, sr)(
        x, row_data.view(-1), row_scales.view(-1), col_data.view(-1), col_scales.view(-1),
        amax_row, amax_col, rht_matrix.view(torch.int16).view(-1), rng, cnt,
        rows, cols, tpc, strips, splits,
        mapping[0], mapping[1], mapping[2], mapping[3], lin_stride_for(cols), stream,
    )
    if not rowwise:
        return None, None, col_data, col_scales
    return row_data, row_scales, col_data, col_scales
