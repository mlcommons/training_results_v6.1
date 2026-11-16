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

"""The chain-1 producer on the CUTLASS DSL: tanh-GELU plus the NVFP4 amax pair
(the reference CUDA kernel ``gelu_tanh_amax_tma_kernel.cu``).

  y       = GELU_tanh(x), bf16, the fast (tanh.approx) mode -- bitwise
            flux.gelu_tanh_ctm_v2_auto(x, True)
  amax[0] = max |y|                                             fp32
  amax[1] = max over 16-row groups g and columns n of
            |bf16_rn( y[16g:16g+16, n] @ R )|                   fp32

Structure, 1:1 with the CUDA kernel: a producer warp streams ``16*CW`` x 128 bf16 tiles as
two SWIZZLE_128B boxes per stage through a hand-rolled mbarrier ring (``mbarrier.init`` by
thread 0, a bare ``mbarrier.try_wait.parity`` spin -- CUTLASS C++'s ``ClusterBarrier::wait``
idiom, ``arrive.expect_tx`` by the producer's lane 0, ``arrive`` by each consumer warp's lane
0); consumer warp ``w`` owns rows ``16w..16w+15`` of every tile (no CTA barrier in the loop):
4 passes of (8 lanes per row, lane ``j`` owning 16 B chunk ``j`` of BOTH 64-column halves --
two bank-spread ``LDS.128``, GELU in fp32 with nvcc's FMA placement, two full-line
``STG.128`` of y, y written back in place), ``__syncwarp``, eight ``ldmatrix.x4.trans`` +
2 ``mma.m16n8k16`` against ``R`` for the post-RHT max, ``fence.proxy.async`` before the slot
is released. Completion is the shipped last-CTA pattern.

Against an earlier ``G1_ctm`` DSL variant this differs in exactly two
places: the ring (it used ``cutlass.pipeline.PipelineTmaAsync``,
whose waits are a timed retry loop with ``NANOSLEEP`` back-off) and the GELU lane map (its
``sub``/``pair`` map fetched the two halves in swapped order and swapped them back through
eight ``SEL``s per pass -- the register pressure that spilled a 24 B frame under the
96-register ceiling of two 9-warp CTAs/SM: 63 MB of local traffic per ``[16384]`` launch).
This kernel compiles to STACK 0, no ``STL``/``LDL``, no ``NANOSLEEP`` (1224 SASS instructions
vs 1320 for the earlier variant and 1368 for CUDA) on sm_103a and sm_107a, bitwise the CUDA kernel at every
production M in both modes.

CTA = ``(CW + 1) * 32`` = 288 threads, ``min_blocks_per_mp=2``, ``preferred_smem_carveout=100``
(2 CTAs/SM, 96 registers on both toolchains; the realised 200 704 B configuration against
CUDA's 233 472 is the 1 028 B alignment-slack artefact of the CUDA host, not a lever).

Dispatched from ``onepass_act.gelu_tanh_amax`` on an allow-list of measured architectures; the
contract is ``tests/test_onepass_act.py``.
"""

from __future__ import annotations

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from flat.custom_ops.onepass_act_ctm_common import (
    BOX_COLS, INT32_MAX, TILE_COLS, bf16_hi, bf16_lo, geometry_tiles, ldmatrix_x4_trans,
    lds128, make_tma, mma_16816, pack_bf16x2, ring_layout, round_bf16, sm_count, sts128,
    warp_max, xorsign_abs,
)

#: variant -> (STAGES, consumer warps, CTAs/SM). Tile = 16*CW rows. One geometry: the CUDA
#: kernel's g128s3w8 (the 4-warp geometry loses on the DSL as it did for nvcc).
VARIANTS: dict[str, tuple[int, int, int]] = {
    "G1_ctm": (3, 8, 2),
}
MODE_OVERSUB2 = 2
_C1 = 0.03567741
_C0 = 0.79788456

# The CUDA kernel's wait (gelu_tanh_amax_tma_kernel.cu:60-74) and CUTLASS C++'s
# ClusterBarrier::wait: a bare try_wait.parity spin, no time budget, no back-off. Labels are
# local to the { } scope, so the block may be inlined any number of times per kernel.
_SPIN_WAIT = "\n".join([
    "{",
    ".reg .pred P1;",
    "LAB_WAIT:",
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [{$r0}], {$r1};",
    "@P1 bra DONE;",
    "bra LAB_WAIT;",
    "DONE:",
    "}",
])
_STG128 = "st.global.v4.b32 [{$r0}], { {$r1}, {$r2}, {$r3}, {$r4} };"


def _gelu(f):
    """TE's fast GELU on one fp32 scalar, FMAs placed as nvcc contracts them (072, bitwise)."""
    inner = f * cute.math.fma(f * _C1, f, cutlass.Float32(_C0))
    t = cute.math.tanh(inner, approx=True)
    return f * cute.math.fma(t, cutlass.Float32(0.5), cutlass.Float32(0.5))


def _spin_wait(mbar_addr, parity):
    """Block until the mbarrier at 32-bit shared address ``mbar_addr`` completes ``parity``."""
    cute.arch.inline_ptx(_SPIN_WAIT, read_only_args=[mbar_addr, parity])


def _stg128(addr, w0, w1, w2, w3):
    """One 128-bit ``st.global.v4.b32`` at a 64-bit address."""
    cute.arch.inline_ptx(_STG128, read_only_args=[addr, w0, w1, w2, w3])


@cute.kernel
def gelu_tanh_amax_ctm_kernel(
    tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor, smem_layout: cute.ComposedLayout,
    y: cute.Tensor, ws_amax: cute.Tensor, amax: cute.Tensor, counter: cute.Tensor,
    rht: cute.Tensor, rows: cutlass.Int32, cols: cutlass.Int32, tiles_per_cta: cutlass.Int32,
    STAGES: cutlass.Constexpr[int], CW: cutlass.Constexpr[int],
) -> None:
    TILE_ROWS = 16 * CW                      # one 16-row group per consumer warp
    SUB_BYTES = TILE_ROWS * BOX_COLS * 2
    STAGE_BYTES = 2 * SUB_BYTES
    THREADS = (CW + 1) * 32

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    gdx, gdy, _ = cute.arch.grid_dim()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    row_tiles = rows // TILE_ROWS
    tile0 = bidy * tiles_per_cta
    ntiles = cutlass.min(tiles_per_cta, row_tiles - tile0)
    col0 = bidx * TILE_COLS
    grid = gdx * gdy
    cta = bidy * gdx + bidx

    smem = utils.SmemAllocator()
    ring = smem.allocate_array(cutlass.BFloat16, STAGES * TILE_ROWS * TILE_COLS, byte_alignment=1024)
    s_tile = cute.make_tensor(
        cute.recast_ptr(ring, smem_layout.inner, dtype=cutlass.BFloat16), smem_layout.outer,
    )
    amax_stage = smem.allocate_array(cutlass.Float32, 2 * (CW + 1), byte_alignment=16)
    bars = smem.allocate_array(cutlass.Int64, 2 * STAGES, byte_alignment=8)
    flag = smem.allocate_array(cutlass.Int32, 1, byte_alignment=4)

    g_tiles = cute.local_tile(tma_tensor, (TILE_ROWS, BOX_COLS), (None, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1),
        cute.group_modes(s_tile, 0, 2), cute.group_modes(g_tiles, 0, 2),
    )

    # ---- the ring, 1:1 with the CUDA kernel: full[s] count 1 (+ tx), empty[s] count CW ----
    full = bars
    empty = bars + STAGES
    full_addr = full.toint()
    empty_addr = empty.toint()
    if tidx == 0:
        for s in cutlass.range_constexpr(STAGES):
            cute.arch.mbarrier_init(full + s, 1)
            cute.arch.mbarrier_init(empty + s, CW)
        cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    if warp == CW:
        # ---- producer warp: lane 0 arms the barrier, the copy elects its lane ------------
        for t in cutlass.range(ntiles):
            s = t % STAGES
            r = t // STAGES
            if t >= STAGES:
                _spin_wait(empty_addr + s * 8, (r - 1) & 1)
            if lane == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(full + s, STAGE_BYTES)
            cute.arch.sync_warp()
            rb = tile0 + t
            cute.copy(tma_atom, tAgA[(None, rb, 2 * bidx)], tAsA[(None, 2 * s)],
                      tma_bar_ptr=full + s)
            cute.copy(tma_atom, tAgA[(None, rb, 2 * bidx + 1)], tAsA[(None, 2 * s + 1)],
                      tma_bar_ptr=full + s)
        # The producer warp still takes part in the amax completion below.

    rmax_packed = cutlass.Int32(0)
    cmax = cutlass.Float32(0.0)
    if warp < CW:
        # ---- consumers -------------------------------------------------------------
        gi = lane >> 2
        tq = lane & 3
        # B fragments: B[k][n] = R[k][n], lane holds k = 2tq, 2tq+1 (+8) at n = gi (+8 jb).
        bfrag = []
        for jb in cutlass.range_constexpr(2):
            nn = jb * 8 + gi
            r0 = rht[(2 * tq) * 16 + nn].to(cutlass.Int32) & 0xFFFF
            r1 = rht[(2 * tq + 1) * 16 + nn].to(cutlass.Int32) & 0xFFFF
            r8 = rht[(2 * tq + 8) * 16 + nn].to(cutlass.Int32) & 0xFFFF
            r9 = rht[(2 * tq + 9) * 16 + nn].to(cutlass.Int32) & 0xFFFF
            bfrag.append(r0 | (r1 << 16))
            bfrag.append(r8 | (r9 << 16))
        # GELU pass lane map: 8 lanes per row; lane j owns 16 B chunk j of BOTH 64-column
        # halves, so each LDS.128 / STS.128 has the 8 lanes of a row on 8 distinct bank groups
        # and y goes out as two full 128 B lines per instruction -- no swap-back, no selects.
        prow = lane >> 3
        j8 = lane & 7
        ld_row = lane & 15
        ld_chunk_hi = lane >> 4
        ring_base = ring.toint()
        y_i64 = cutlass.Pointer(y.iterator, dtype=cutlass.BFloat16).toint()
        zero = (cutlass.Float32(0.0),) * 4

        for t in cutlass.range(ntiles):
            slot = t % STAGES
            _spin_wait(full_addr + slot * 8, (t // STAGES) & 1)
            tile_s = ring_base + slot * STAGE_BYTES
            grow0 = (tile0 + t) * TILE_ROWS

            # ---- GELU: rows 16w..16w+15, 4 passes of 4 rows ------------------------
            for p in cutlass.range_constexpr(4):
                rr = warp * 16 + p * 4 + prow
                rowb = tile_s + rr * 128
                phys = (j8 ^ (rr & 7)) << 4
                addr0 = rowb + phys                      # half 0, chunk j8
                addr1 = rowb + SUB_BYTES + phys          # half 1, chunk j8
                v0, v1, v2, v3 = lds128(addr0)
                v4, v5, v6, v7 = lds128(addr1)
                vs = [v0, v1, v2, v3, v4, v5, v6, v7]
                o = []
                for k in cutlass.range_constexpr(8):
                    ow = pack_bf16x2(_gelu(bf16_lo(vs[k])), _gelu(bf16_hi(vs[k])))
                    rmax_packed = xorsign_abs(rmax_packed, ow)
                    o.append(ow)
                sts128(addr0, o[0], o[1], o[2], o[3])
                sts128(addr1, o[4], o[5], o[6], o[7])
                # y: chunk j8 of half 0 at col0 + 8*j8, of half 1 at +64 elements (+128 B).
                eoff = (grow0 + rr) * cols + col0 + j8 * 8
                gaddr = y_i64 + (eoff * 2).to(cutlass.Int64)
                _stg128(gaddr, o[0], o[1], o[2], o[3])
                _stg128(gaddr + 128, o[4], o[5], o[6], o[7])
            cute.arch.sync_warp()

            # ---- post-RHT max over this warp's 16 rows x 128 columns ---------------
            for cb in cutlass.range_constexpr(8):
                sb = tile_s + (cb >> 2) * SUB_BYTES
                chunk = ((cb & 3) * 2 + ld_chunk_hi) ^ (ld_row & 7)
                f0, f1, f2, f3 = ldmatrix_x4_trans(sb + (warp * 16 + ld_row) * 128 + (chunk << 4))
                d0 = mma_16816((f0, f2, f1, f3), bfrag[0], bfrag[1], zero)
                d1 = mma_16816((f0, f2, f1, f3), bfrag[2], bfrag[3], zero)
                for j in cutlass.range_constexpr(4):
                    cmax = cute.arch.fmax(cmax, cute.math.abs(d0[j]))
                    cmax = cute.arch.fmax(cmax, cute.math.abs(d1[j]))
            # Generic writes (the y write-back) must be ordered before the TMA unit
            # refills this slot: cross-proxy fence, then release the slot.
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(empty + slot)

    # ---- amax: warp -> CTA -> partial slot -> last CTA reduces all (T7's pattern) ----
    rmax = warp_max(cute.arch.fmax(cute.math.abs(bf16_lo(rmax_packed)),
                                   cute.math.abs(bf16_hi(rmax_packed))))
    cmax = warp_max(cmax)
    stage = cutlass.Pointer(amax_stage, dtype=cutlass.Float32)
    if lane == 0:
        (stage + 2 * warp).store(rmax)
        (stage + 2 * warp + 1).store(cmax)
    cute.arch.sync_threads()
    wsp = cutlass.Pointer(ws_amax.iterator, dtype=cutlass.Float32)
    ap = cutlass.Pointer(amax.iterator, dtype=cutlass.Float32)
    cp = cutlass.Pointer(counter.iterator, dtype=cutlass.Int32)
    fp = cutlass.Pointer(flag, dtype=cutlass.Int32)
    if tidx == 0:
        r0 = cutlass.Float32(0.0)
        c0 = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(CW):     # the producer warp's slot holds zeros
            r0 = cute.arch.fmax(r0, (stage + 2 * w).load())
            c0 = cute.arch.fmax(c0, (stage + 2 * w + 1).load())
        (wsp + 2 * cta).store(r0)
        (wsp + 2 * cta + 1).store(c0)
        cute.arch.fence_acq_rel_gpu()
        prev = cute.arch.atomic_add(cp, cutlass.Int32(1))
        fp.store(prev)
    cute.arch.sync_threads()
    prev_b = fp.load()
    if prev_b == grid - 1:
        cute.arch.fence_acq_rel_gpu()
        r1 = cutlass.Float32(0.0)
        c1 = cutlass.Float32(0.0)
        for i in cutlass.range(tidx, grid, THREADS):
            r1 = cute.arch.fmax(r1, (wsp + 2 * i).load(is_volatile=True))
            c1 = cute.arch.fmax(c1, (wsp + 2 * i + 1).load(is_volatile=True))
        r1 = warp_max(r1)
        c1 = warp_max(c1)
        if lane == 0:
            (stage + 2 * warp).store(r1)
            (stage + 2 * warp + 1).store(c1)
        cute.arch.sync_threads()
        if tidx == 0:
            r2 = cutlass.Float32(0.0)
            c2 = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(CW + 1):
                r2 = cute.arch.fmax(r2, (stage + 2 * w).load())
                c2 = cute.arch.fmax(c2, (stage + 2 * w + 1).load())
            ap.store(r2)                       # already on the bf16 grid: max of bf16 values
            (ap + 1).store(round_bf16(c2))     # TE rounds the MMA result to bf16 before the max
            cp.store(cutlass.Int32(0))         # self-reset for the next launch


@cute.jit
def gelu_tanh_amax_ctm_host(
    x: cute.Tensor, y: cute.Tensor, ws_amax: cute.Tensor, amax: cute.Tensor,
    counter: cute.Tensor, rht: cute.Tensor,
    rows: cutlass.Int32, cols: cutlass.Int32, tiles_per_cta: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32, stream: cuda.CUstream,
    STAGES: cutlass.Constexpr[int], CW: cutlass.Constexpr[int], MIN_CTAS: cutlass.Constexpr[int],
) -> None:
    smem_layout = ring_layout(16 * CW, 2 * STAGES)
    tma_atom, tma_tensor = make_tma(x, 16 * CW, smem_layout)
    gelu_tanh_amax_ctm_kernel(
        tma_atom, tma_tensor, smem_layout, y, ws_amax, amax, counter, rht,
        rows, cols, tiles_per_cta, STAGES, CW,
    ).launch(
        grid=(grid_x, grid_y, 1), block=((CW + 1) * 32, 1, 1), stream=stream,
        # The register cap of __launch_bounds__(288, MIN_CTAS); the explicit carveout
        # keeps the DSL out of its automatic-carveout path.
        min_blocks_per_mp=MIN_CTAS, preferred_smem_carveout=100,
        # No smem= request: the allocator's 98 428 B realises the 200 704 B configuration
        # (CUDA's 99 456 B realises 233 472). Measured the matched configuration
        # (`smem=` the CUDA host's Geo::SMEM_BYTES, commit 55ca1b9f2a) at the step on two disjoint
        # 4-node sets: +0.015 / +0.015 ms/step vs +0.068 / +0.050 unmatched, all inside drift --
        # the match buys nothing measurable and costs 32 KB of L1 per SM, so it is not requested.
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_compiled: dict[tuple[int, str], Callable] = {}
_counters: dict[int, torch.Tensor] = {}


def _compile(device_index: int, variant: str) -> Callable:
    """Compile once per (device, variant); rows, cols and the grid stay symbolic."""
    key = (device_index, variant)
    if key not in _compiled:
        stages, cw, min_ctas = VARIANTS[variant]
        with torch.cuda.device(device_index):
            bf, f32, i16, i32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int16, cutlass.Int32
            sym = cute.sym_int64
            fx = make_fake_compact_tensor(bf, (sym(16, "m"), sym(16, "n")), stride_order=(1, 0),
                                          assumed_align=16)
            fy = make_fake_compact_tensor(bf, (sym(16, "ny"),), assumed_align=16)
            fws = make_fake_compact_tensor(f32, (sym(2, "nw"),), assumed_align=8)
            famax = make_fake_compact_tensor(f32, (2,), assumed_align=8)
            fcnt = make_fake_compact_tensor(i32, (1,), assumed_align=4)
            frht = make_fake_compact_tensor(i16, (256,), assumed_align=16)
            ints = [i32(0)] * 5
            _compiled[key] = cute.compile(
                gelu_tanh_amax_ctm_host, fx, fy, fws, famax, fcnt, frht, *ints,
                make_fake_stream(), stages, cw, min_ctas, options="--enable-tvm-ffi",
            )
    if device_index not in _counters:
        # Persistent, self-resetting (the last CTA writes 0); created outside any capture.
        _counters[device_index] = torch.zeros(
            1, dtype=torch.int32, device=torch.device("cuda", device_index))
    return _compiled[key]


def counter(device_index: int) -> torch.Tensor:
    """The persistent completion counter for ``device_index`` (0 between launches)."""
    _compile(device_index, next(iter(VARIANTS)))
    return _counters[device_index]


def variants() -> list[str]:
    """Variant names."""
    return list(VARIANTS)


def grid(rows: int, cols: int, variant: str = "G1_ctm", mode: int = 0,
         device_index: int | None = None) -> tuple[int, int, int, int]:
    """``(col_strips, row_splits, tiles_per_cta, resident_ctas)`` the host will launch."""
    _, cw, min_ctas = VARIANTS[variant]
    dev = torch.cuda.current_device() if device_index is None else device_index
    return geometry_tiles(rows, cols, 16 * cw, min_ctas, sm_count(dev), bool(mode & MODE_OVERSUB2))


def _check(x: torch.Tensor, rht_matrix: torch.Tensor, variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"gelu_tanh_amax_ctm: unknown variant {variant!r}; have {list(VARIANTS)}")
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()):
        raise ValueError("gelu_tanh_amax_ctm: x must be a contiguous 2-D CUDA BF16 tensor")
    rows, cols = x.shape
    tile_rows = 16 * VARIANTS[variant][1]
    if rows % tile_rows != 0 or cols % TILE_COLS != 0:
        raise ValueError(
            f"gelu_tanh_amax_ctm: rows % {tile_rows} == 0 and cols % 128 == 0 required, got {rows}x{cols}")
    if rows * cols * 2 > INT32_MAX:
        raise ValueError("gelu_tanh_amax_ctm: byte count exceeds the Int32 index range")
    if x.data_ptr() % 16 != 0:
        raise ValueError("gelu_tanh_amax_ctm: x must be 16 B aligned")
    if not (rht_matrix.is_cuda and rht_matrix.dtype == torch.bfloat16 and rht_matrix.is_contiguous()
            and tuple(rht_matrix.shape) == (16, 16)):
        raise ValueError("gelu_tanh_amax_ctm: rht_matrix must be a contiguous CUDA BF16 [16, 16]")


def gelu_tanh_amax_ctm(
    x: torch.Tensor, rht_matrix: torch.Tensor, variant: str = "G1_ctm", mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast tanh-GELU of ``x`` plus the NVFP4 amax pair of the result, DSL kernel.

    Args:
        x: BF16 ``[rows, cols]`` contiguous, ``rows % 128 == 0``, ``cols % 128 == 0``.
        rht_matrix: ``NVFP4Quantizer.rht_matrix``, BF16 ``[16, 16]``.
        variant: One of :data:`VARIANTS`.
        mode: Bit 1 (``MODE_OVERSUB2``) doubles the row splits.

    Returns:
        ``(y, amax)`` -- ``y`` BF16 bitwise ``flux.gelu_tanh_ctm_v2_auto(x, True)``;
        ``amax`` FP32 ``[2]`` = ``(max|y|, post-RHT columnwise amax)``, bitwise
        ``flux.mxfp8_rowwise_cast_amax_ctm(y, rht_matrix, True)[2]``.

    Raises:
        ValueError: On an unsupported input or variant.
    """
    _check(x, rht_matrix, variant)
    dev = x.device.index
    rows, cols = x.shape
    col_strips, row_splits, tiles_per_cta, _ = grid(rows, cols, variant, mode, dev)
    assert (row_splits - 1) * tiles_per_cta < rows // (16 * VARIANTS[variant][1])
    fn = _compile(dev, variant)
    y = torch.empty_like(x)
    amax = torch.empty(2, dtype=torch.float32, device=x.device)
    ws = torch.empty(col_strips * row_splits * 2, dtype=torch.float32, device=x.device)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    fn(x, y.view(-1), ws, amax, _counters[dev], rht_matrix.view(torch.int16).view(-1),
       rows, cols, tiles_per_cta, col_strips, row_splits, stream)
    return y, amax


def verify(rows: int = 1024, cols: int = 12288) -> None:
    """Compile, run and bit-check the producer against today's GELU and pass 1, both modes.

    The ``ctm-write-kernel`` skill's entry point; the full contract (all M, TE's
    amax, graphs, the pressure test, the dispatch census) is ``tests/test_onepass_act.py``.

    Raises:
        AssertionError: If ``y`` or the amax pair differs.
    """
    from flat import nvfp4_ops as nv

    torch.manual_seed(rows * 31 + cols)
    q = nv.role_quantizer("act", columnwise_only=True)
    pre = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    y_ref = torch.ops.flux.gelu_tanh_ctm_v2_auto(pre, True)
    _, _, amax_ref = torch.ops.flux.mxfp8_rowwise_cast_amax_ctm(y_ref, q.rht_matrix, True)
    for mode in (0, MODE_OVERSUB2):
        y, amax = gelu_tanh_amax_ctm(pre, q.rht_matrix, "G1_ctm", mode)
        torch.cuda.synchronize()
        assert torch.equal(y, y_ref), f"mode {mode}: y differs from gelu_tanh_ctm_v2_auto"
        assert torch.equal(amax, amax_ref), f"mode {mode}: amax pair differs from pass 1"
    print(f"gelu_tanh_amax_ctm.verify [{rows},{cols}]: G1_ctm (144 LW) plain and +o2 bitwise today's GELU and pass 1: PASS")
