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

"""``flux.seq_split_ctm3`` / ``flux.seq_join_ctm3``.

The sequence split (``[B, L1+L2, ...] -> [B, L1, ...], [B, L2, ...]``) and its
inverse, on the CUTLASS Python DSL, public ``cute.arch.*`` API only (proven on
4.6.2 and 4.8.0.dev0). Pure data movement: the
outputs are bitwise the inputs by construction, any dtype (the flat storage is
viewed as 16-bit units; a row must be a multiple of 16 bytes).

Three kernels, one module, both directions (``TO_SPLIT`` is a ``Constexpr``):

* ``seq_tiled_ctm3_kernel[UNROLL]`` -- 1-D grid over ``THREADS x UNROLL x 16 B``
  tiles in joined-tensor order. Each CTA maps a tile to ``(batch, part, offset)``
  once (one integer divide per tile, CTA-uniform branch) and each thread issues
  ``UNROLL`` independent 16-byte loads before its first store, so the bytes in
  flight per SM grow with ``UNROLL`` (Little's law). Requires each part's byte count to be a tile multiple, true at
  every production shape (256 rows x 6144 B = 96 x 16 KB).
* ``seq_bulk_ctm3_kernel[STAGES]`` -- ``cp.async.bulk`` G2S -> 8 KB smem
  ring (``mbarrier complete_tx``) -> ``cp.async.bulk`` S2G under
  ``bulk_group``, one issuing thread per CTA, now launched with an explicit
  ``min_blocks_per_mp`` and ``preferred_smem_carveout`` and a linear 1-D tile grid.
* ``seq_general_ctm3_kernel`` -- the plain kernel, unchanged in structure:
  one 16-byte load per thread, grid ``(x, B)``. The fallback for any shape the
  tiled kernels cannot take, and the control arm for the grid-mapping question.

Composite ops dispatch on a size expression (``pick_variant``); per-variant ops
``flux.seq_{split,join}_ctm3_<arm>`` exist for the bench and the tests.
Registered, **unwired**: nothing in ``flat/`` routes to any of them.
"""

from __future__ import annotations

import math
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

THREADS = 256
VEC = 8  # 16-bit units per 16-byte access
_UNIT = torch.bfloat16  # the 16-bit view every kernel is compiled for
_INT32_MAX = 2**31 - 1
_TILED_CTAS_PER_SM = 8  # thread-limited: 2048 / 256; no smem, so no carveout to pin

# bulk variant
BULK_THREADS = 32
BULK_TILE = 4096  # 16-bit units per bulk tile (8 KB)
BULK_TILE_BYTES = BULK_TILE * 2
#: Bulk CTAs per SM are pinned with ``min_blocks_per_mp`` (the last field of
#: :data:`VARIANTS`): STAGES x 8 KB (+ barriers) per CTA against the 228 KB per SM
#: on sm_103a; 4 x 32 KB x 6 = 192 KB, 8 x 64 KB x 3 = 192 KB.
SMEM_CARVEOUT_PCT = 100
_G2S = ("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[{$r0}], [{$r1}], {$r2}, [{$r3}];")
_S2G = "cp.async.bulk.global.shared::cta.bulk_group [{$r0}], [{$r1}], {$r2};"

#: arm -> (kernel family, parameter, threads per CTA, grid rule, min_blocks_per_mp).
#: ``parameter`` is UNROLL (tiled) or STAGES (bulk). Grid rule ``persistent`` =
#: ``min(n_tiles, SMs x 8)`` (a first pass that assumed 8 CTAs/SM
#: and got 6 / 4); ``oneshot`` = one CTA per tile, no
#: residency assumption at all (the simplest form).
VARIANTS: dict[str, tuple[str, int, int, str, int]] = {
    "general": ("general", 1, THREADS, "persistent", 0),
    "lin1": ("tiled", 1, THREADS, "persistent", 0),
    "u4": ("tiled", 4, THREADS, "persistent", 0),
    "u8": ("tiled", 8, THREADS, "persistent", 0),
    "bulk4": ("bulk", 4, BULK_THREADS, "persistent", 6),
    "bulk8": ("bulk", 8, BULK_THREADS, "persistent", 3),
    # grids that do not assume a residency
    "u4o": ("tiled", 4, THREADS, "oneshot", 0),
    "u8o": ("tiled", 8, THREADS, "oneshot", 0),
    "row3": ("tiled", 3, 128, "oneshot", 0),   # Triton's structure: 128 thr x 3 vec = one 6 KB row per CTA
    "row6": ("tiled", 6, 128, "oneshot", 0),   # two rows per CTA
    "u4p8": ("tiled", 4, THREADS, "persistent", 8),  # min_blocks_per_mp=8: is it a register budget?
    # row3's structure with raw Int64 pointer arguments instead of
    # three cute.Tensors (no dynamic shape/stride in the launch parameter block), and a
    # single copy body reached through an Int64 address merge; ``rowps`` additionally
    # compiles ``tiles_total`` / ``tiles_1`` as constants (shapes are static per CUDA-graph
    # capture), so the per-CTA integer divide becomes a multiply-shift.
    "rowp": ("rawptr", 3, 128, "oneshot", 0),
    "rowps": ("rawptr_static", 3, 128, "oneshot", 0),
}
#: The composite's default, chosen by measurement:
#: Triton's structure -- 128 threads x 3 vectors = one 6 KB row per CTA, B x L CTAs --
#: with raw Int64 pointer arguments and the tile map compiled as constants (one compile
#: per (tiles_total, tiles_1), which is the same pair at every micro-batch). Kernel
#: time <= Triton at every B.
#: Earlier defaults: ``u4o``, ``row3``.
DEFAULT_TILED = "rowps"
KERNEL_NAMES = {"general": "seq_general_ctm3_kernel", "tiled": "seq_tiled_ctm3_kernel",
                "bulk": "seq_bulk_ctm3_kernel", "rawptr": "seq_rawptr_ctm3_kernel",
                "rawptr_static": "seq_rawptr_static_ctm3_kernel"}
_TILED_FAMILIES = ("tiled", "rawptr", "rawptr_static")


def tile_bytes(variant: str) -> int:
    """Bytes per tile the variant needs each part to be a multiple of (16 for ``general``)."""
    family, p, nt, _, _ = VARIANTS[variant]
    if family in _TILED_FAMILIES:
        return nt * p * 16
    if family == "bulk":
        return BULK_TILE_BYTES
    return 16


def pick_variant(part1_bytes: int, part2_bytes: int) -> str:
    """The composite's dispatch: a size expression, never a batch number.

    ``DEFAULT_TILED`` where both parts are whole tiles of it, else ``general``.
    """
    tb = tile_bytes(DEFAULT_TILED)
    if part1_bytes % tb == 0 and part2_bytes % tb == 0:
        return DEFAULT_TILED
    return "general"


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def _copy_tile(src, s_vec, dst, d_vec, unroll: int, nt: int) -> None:
    """``unroll`` 16-byte loads for this thread, all issued before the first store.

    Plain Python ``for``: helpers are traced, not preprocessed, so this unrolls
    at trace time (``gelu_tanh_ctm_v2._gelu_vec`` uses the same idiom).
    """
    vs = []
    for u in range(unroll):
        vs.append((src + (s_vec + u * nt) * VEC).load(16, count=VEC))
    for u in range(unroll):
        (dst + (d_vec + u * nt) * VEC).store(vs[u], alignment=16)


@cute.kernel
def seq_tiled_ctm3_kernel(
    joined: cute.Tensor, part1: cute.Tensor, part2: cute.Tensor,
    tiles_total: cutlass.Int32, tiles_1: cutlass.Int32, n_tiles: cutlass.Int32,
    TO_SPLIT: cutlass.Constexpr[bool], UNROLL: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
) -> None:
    """Grid-stride over the ``n_tiles = B x tiles_total`` tiles in joined order.

    Tile ``t`` is batch element ``t // tiles_total``, tile ``r`` of it; it is
    tile ``r`` of ``part1[b]`` when ``r < tiles_1``, else tile ``r - tiles_1``
    of ``part2[b]``. The branch is CTA-uniform. Within a tile thread ``tidx``
    owns vectors ``tidx + u * NT`` for ``u < UNROLL``, each load a fully
    coalesced ``NT x 16`` bytes per CTA. With a one-shot grid the loop runs once.
    """
    TILE_VEC: cutlass.Constexpr[int] = NT * UNROLL
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    jp = cutlass.Pointer(joined.iterator, dtype=cutlass.BFloat16)
    p1 = cutlass.Pointer(part1.iterator, dtype=cutlass.BFloat16)
    p2 = cutlass.Pointer(part2.iterator, dtype=cutlass.BFloat16)
    tiles_2 = tiles_total - tiles_1
    for t in cutlass.range(bidx, n_tiles, gdim):
        b = t // tiles_total
        r = t - b * tiles_total
        j_vec = t * TILE_VEC + tidx
        if r < tiles_1:
            p_vec = (b * tiles_1 + r) * TILE_VEC + tidx
            if cutlass.const_expr(TO_SPLIT):
                _copy_tile(jp, j_vec, p1, p_vec, UNROLL, NT)
            else:
                _copy_tile(p1, p_vec, jp, j_vec, UNROLL, NT)
        else:
            p_vec = (b * tiles_2 + (r - tiles_1)) * TILE_VEC + tidx
            if cutlass.const_expr(TO_SPLIT):
                _copy_tile(jp, j_vec, p2, p_vec, UNROLL, NT)
            else:
                _copy_tile(p2, p_vec, jp, j_vec, UNROLL, NT)


@cute.jit
def seq_tiled_ctm3_host(
    joined: cute.Tensor, part1: cute.Tensor, part2: cute.Tensor,
    tiles_total: cutlass.Int32, tiles_1: cutlass.Int32, n_tiles: cutlass.Int32,
    grid_x: cutlass.Int32, stream: cuda.CUstream,
    TO_SPLIT: cutlass.Constexpr[bool], UNROLL: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
) -> None:
    seq_tiled_ctm3_kernel(joined, part1, part2, tiles_total, tiles_1, n_tiles,
                          TO_SPLIT, UNROLL, NT).launch(
        grid=(grid_x, 1, 1), block=(NT, 1, 1), stream=stream)


@cute.jit
def seq_tiled_pinned_ctm3_host(
    joined: cute.Tensor, part1: cute.Tensor, part2: cute.Tensor,
    tiles_total: cutlass.Int32, tiles_1: cutlass.Int32, n_tiles: cutlass.Int32,
    grid_x: cutlass.Int32, stream: cuda.CUstream,
    TO_SPLIT: cutlass.Constexpr[bool], UNROLL: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    MINB: cutlass.Constexpr[int],
) -> None:
    """The same kernel with ``min_blocks_per_mp`` pinned -- the B6 register-budget probe."""
    seq_tiled_ctm3_kernel(joined, part1, part2, tiles_total, tiles_1, n_tiles,
                          TO_SPLIT, UNROLL, NT).launch(
        grid=(grid_x, 1, 1), block=(NT, 1, 1), min_blocks_per_mp=MINB, stream=stream)


def _ptr(addr):
    """A 16-byte-aligned global bf16 pointer from an Int64 byte address
    (``fused_sum_cat_ctm2.py``'s idiom)."""
    return cutlass.Pointer(cute.make_ptr(cutlass.BFloat16, addr, cute.AddressSpace.gmem,
                                         assumed_align=16), dtype=cutlass.BFloat16)


@cute.kernel
def seq_rawptr_ctm3_kernel(
    jaddr: cutlass.Int64, p1addr: cutlass.Int64, p2addr: cutlass.Int64,
    tiles_total: cutlass.Int32, tiles_1: cutlass.Int32, n_tiles: cutlass.Int32,
    TO_SPLIT: cutlass.Constexpr[bool], UNROLL: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
) -> None:
    """``row3``'s structure with raw Int64 pointer arguments and a dynamic tile map.

    One tile per CTA (one-shot grid): tile -> (b, part, offset) once, then a single
    copy body reached through an Int64 address merge. The dynamic ``if`` lives in the
    kernel body, not a helper (it does not survive a plain
    Python call boundary).
    """
    TILE_VEC: cutlass.Constexpr[int] = NT * UNROLL
    tidx, _, _ = cute.arch.thread_idx()
    t, _, _ = cute.arch.block_idx()
    b = t // tiles_total
    r = t - b * tiles_total
    tiles_2 = tiles_total - tiles_1
    j_off = (t * TILE_VEC + tidx).to(cutlass.Int64) * (VEC * 2)
    paddr = p1addr + ((b * tiles_1 + r) * TILE_VEC + tidx).to(cutlass.Int64) * (VEC * 2)
    if r >= tiles_1:
        paddr = p2addr + ((b * tiles_2 + (r - tiles_1)) * TILE_VEC + tidx).to(cutlass.Int64) * (VEC * 2)
    jp = _ptr(jaddr + j_off)
    pp = _ptr(paddr)
    if cutlass.const_expr(TO_SPLIT):
        _copy_tile(jp, 0, pp, 0, UNROLL, NT)
    else:
        _copy_tile(pp, 0, jp, 0, UNROLL, NT)


@cute.jit
def seq_rawptr_ctm3_host(
    jaddr: cutlass.Int64, p1addr: cutlass.Int64, p2addr: cutlass.Int64,
    tiles_total: cutlass.Int32, tiles_1: cutlass.Int32, n_tiles: cutlass.Int32,
    grid_x: cutlass.Int32, stream: cuda.CUstream,
    TO_SPLIT: cutlass.Constexpr[bool], UNROLL: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
) -> None:
    seq_rawptr_ctm3_kernel(jaddr, p1addr, p2addr, tiles_total, tiles_1, n_tiles,
                           TO_SPLIT, UNROLL, NT).launch(
        grid=(grid_x, 1, 1), block=(NT, 1, 1), stream=stream)


@cute.kernel
def seq_rawptr_static_ctm3_kernel(
    jaddr: cutlass.Int64, p1addr: cutlass.Int64, p2addr: cutlass.Int64,
    TO_SPLIT: cutlass.Constexpr[bool], UNROLL: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    TILES_TOTAL: cutlass.Constexpr[int], TILES_1: cutlass.Constexpr[int],
) -> None:
    """``rowp`` with the tile map compiled in: the divide by ``TILES_TOTAL`` is a constant."""
    TILE_VEC: cutlass.Constexpr[int] = NT * UNROLL
    TILES_2: cutlass.Constexpr[int] = TILES_TOTAL - TILES_1
    tidx, _, _ = cute.arch.thread_idx()
    t, _, _ = cute.arch.block_idx()
    b = t // TILES_TOTAL
    r = t - b * TILES_TOTAL
    j_off = (t * TILE_VEC + tidx).to(cutlass.Int64) * (VEC * 2)
    paddr = p1addr + ((b * TILES_1 + r) * TILE_VEC + tidx).to(cutlass.Int64) * (VEC * 2)
    if r >= TILES_1:
        paddr = p2addr + ((b * TILES_2 + (r - TILES_1)) * TILE_VEC + tidx).to(cutlass.Int64) * (VEC * 2)
    jp = _ptr(jaddr + j_off)
    pp = _ptr(paddr)
    if cutlass.const_expr(TO_SPLIT):
        _copy_tile(jp, 0, pp, 0, UNROLL, NT)
    else:
        _copy_tile(pp, 0, jp, 0, UNROLL, NT)


@cute.jit
def seq_rawptr_static_ctm3_host(
    jaddr: cutlass.Int64, p1addr: cutlass.Int64, p2addr: cutlass.Int64,
    grid_x: cutlass.Int32, stream: cuda.CUstream,
    TO_SPLIT: cutlass.Constexpr[bool], UNROLL: cutlass.Constexpr[int], NT: cutlass.Constexpr[int],
    TILES_TOTAL: cutlass.Constexpr[int], TILES_1: cutlass.Constexpr[int],
) -> None:
    seq_rawptr_static_ctm3_kernel(jaddr, p1addr, p2addr, TO_SPLIT, UNROLL, NT,
                                  TILES_TOTAL, TILES_1).launch(
        grid=(grid_x, 1, 1), block=(NT, 1, 1), stream=stream)


def _g2s(smem_int, gaddr, bar_int) -> None:
    cute.arch.inline_ptx(_G2S, read_only_args=[smem_int, gaddr, cutlass.Int32(BULK_TILE_BYTES),
                                                bar_int])


def _s2g(gaddr, smem_int) -> None:
    cute.arch.inline_ptx(_S2G, read_only_args=[gaddr, smem_int, cutlass.Int32(BULK_TILE_BYTES)])


@cute.kernel
def seq_bulk_ctm3_kernel(
    joined: cute.Tensor, part1: cute.Tensor, part2: cute.Tensor,
    tiles_total: cutlass.Int32, tiles_1: cutlass.Int32, n_tiles: cutlass.Int32,
    TO_SPLIT: cutlass.Constexpr[bool], STAGES: cutlass.Constexpr[int],
) -> None:
    """Grid-stride over the ``n_tiles`` 8 KB tiles in joined order, one issuing thread.

    Thread 0 keeps ``STAGES - 1`` G2S bulk copies in flight into a smem ring;
    each landed tile (mbarrier ``complete_tx``) is issued back out with an S2G
    bulk copy, and a stage is refilled only after ``wait_group.read`` proves
    the S2G that last read it has finished reading.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    ring = cute.arch.alloc_smem(cutlass.BFloat16, STAGES * BULK_TILE, 128)
    bars = cute.arch.alloc_smem(cutlass.Int64, STAGES, 8)
    tiles_2 = tiles_total - tiles_1
    jbase = joined.iterator.toint()
    p1base = part1.iterator.toint()
    p2base = part2.iterator.toint()
    if tidx == 0:
        for s in cutlass.range_constexpr(STAGES):
            cute.arch.mbarrier_init(bars + s, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    if tidx == 0:
        for p in cutlass.range_constexpr(STAGES - 1):
            k0 = bidx + p * gdim
            if k0 < n_tiles:
                b0 = k0 // tiles_total
                r0 = k0 - b0 * tiles_total
                jaddr0 = jbase + k0.to(cutlass.Int64) * BULK_TILE_BYTES
                paddr0 = p1base + (b0 * tiles_1 + r0).to(cutlass.Int64) * BULK_TILE_BYTES
                if r0 >= tiles_1:
                    paddr0 = p2base + (b0 * tiles_2 + (r0 - tiles_1)).to(cutlass.Int64) * BULK_TILE_BYTES
                cute.arch.mbarrier_arrive_and_expect_tx(bars + p, BULK_TILE_BYTES)
                if cutlass.const_expr(TO_SPLIT):
                    _g2s((ring + p * BULK_TILE).toint(), jaddr0, (bars + p).toint())
                else:
                    _g2s((ring + p * BULK_TILE).toint(), paddr0, (bars + p).toint())
        for k in cutlass.range(bidx, n_tiles, gdim):
            j = (k - bidx) // gdim
            s = j % STAGES
            cute.arch.mbarrier_wait(bars + s, (j // STAGES) % 2)
            b = k // tiles_total
            r = k - b * tiles_total
            jaddr = jbase + k.to(cutlass.Int64) * BULK_TILE_BYTES
            paddr = p1base + (b * tiles_1 + r).to(cutlass.Int64) * BULK_TILE_BYTES
            if r >= tiles_1:
                paddr = p2base + (b * tiles_2 + (r - tiles_1)).to(cutlass.Int64) * BULK_TILE_BYTES
            if cutlass.const_expr(TO_SPLIT):
                _s2g(paddr, (ring + s * BULK_TILE).toint())
            else:
                _s2g(jaddr, (ring + s * BULK_TILE).toint())
            cute.arch.cp_async_bulk_commit_group()
            kn = k + (STAGES - 1) * gdim
            if kn < n_tiles:
                # tile kn goes to stage (j - 1) % STAGES, last read by S2G group j - 1
                cute.arch.cp_async_bulk_wait_group(1, read=True)
                sn = (j + STAGES - 1) % STAGES
                bn = kn // tiles_total
                rn = kn - bn * tiles_total
                jaddrn = jbase + kn.to(cutlass.Int64) * BULK_TILE_BYTES
                paddrn = p1base + (bn * tiles_1 + rn).to(cutlass.Int64) * BULK_TILE_BYTES
                if rn >= tiles_1:
                    paddrn = p2base + (bn * tiles_2 + (rn - tiles_1)).to(cutlass.Int64) * BULK_TILE_BYTES
                cute.arch.mbarrier_arrive_and_expect_tx(bars + sn, BULK_TILE_BYTES)
                if cutlass.const_expr(TO_SPLIT):
                    _g2s((ring + sn * BULK_TILE).toint(), jaddrn, (bars + sn).toint())
                else:
                    _g2s((ring + sn * BULK_TILE).toint(), paddrn, (bars + sn).toint())
        cute.arch.cp_async_bulk_wait_group(0)


@cute.jit
def seq_bulk_ctm3_host(
    joined: cute.Tensor, part1: cute.Tensor, part2: cute.Tensor,
    tiles_total: cutlass.Int32, tiles_1: cutlass.Int32, n_tiles: cutlass.Int32,
    grid_x: cutlass.Int32, stream: cuda.CUstream,
    TO_SPLIT: cutlass.Constexpr[bool], STAGES: cutlass.Constexpr[int],
    OCC: cutlass.Constexpr[int], CARVEOUT: cutlass.Constexpr[int],
) -> None:
    seq_bulk_ctm3_kernel(joined, part1, part2, tiles_total, tiles_1, n_tiles,
                         TO_SPLIT, STAGES).launch(
        grid=(grid_x, 1, 1), block=(BULK_THREADS, 1, 1),
        min_blocks_per_mp=OCC, preferred_smem_carveout=CARVEOUT, stream=stream)


@cute.kernel
def seq_general_ctm3_kernel(
    joined: cute.Tensor, part1: cute.Tensor, part2: cute.Tensor,
    rowvec_total: cutlass.Int32, rowvec_1: cutlass.Int32,
    TO_SPLIT: cutlass.Constexpr[bool],
) -> None:
    """The ``plain`` kernel: one batch element per ``block_idx().y``, grid-stride
    over its 16-byte vectors, one load and one store per thread per iteration."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, b, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()
    jp = cutlass.Pointer(joined.iterator, dtype=cutlass.BFloat16)
    p1 = cutlass.Pointer(part1.iterator, dtype=cutlass.BFloat16)
    p2 = cutlass.Pointer(part2.iterator, dtype=cutlass.BFloat16)
    rowvec_2 = rowvec_total - rowvec_1
    for i in cutlass.range(bidx * bdim + tidx, rowvec_total, gdim * bdim):
        j_off = (b * rowvec_total + i) * VEC
        if i < rowvec_1:
            p_off = (b * rowvec_1 + i) * VEC
            if cutlass.const_expr(TO_SPLIT):
                (p1 + p_off).store((jp + j_off).load(16, count=VEC), alignment=16)
            else:
                (jp + j_off).store((p1 + p_off).load(16, count=VEC), alignment=16)
        else:
            p_off = (b * rowvec_2 + i - rowvec_1) * VEC
            if cutlass.const_expr(TO_SPLIT):
                (p2 + p_off).store((jp + j_off).load(16, count=VEC), alignment=16)
            else:
                (jp + j_off).store((p2 + p_off).load(16, count=VEC), alignment=16)


@cute.jit
def seq_general_ctm3_host(
    joined: cute.Tensor, part1: cute.Tensor, part2: cute.Tensor,
    rowvec_total: cutlass.Int32, rowvec_1: cutlass.Int32,
    grid_x: cutlass.Int32, batch: cutlass.Int32, stream: cuda.CUstream,
    TO_SPLIT: cutlass.Constexpr[bool],
) -> None:
    seq_general_ctm3_kernel(joined, part1, part2, rowvec_total, rowvec_1, TO_SPLIT).launch(
        grid=(grid_x, batch, 1), block=(THREADS, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------

_compiled: dict[tuple, Callable] = {}
_sms: dict[int, int] = {}


def _fakes():
    return [make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int64(divisibility=VEC),),
                                     assumed_align=16) for _ in range(3)]


def _compile(device_index: int, variant: str, to_split: bool,
             static: tuple[int, int] | None = None) -> Callable:
    """Compile once per (device, variant, direction); every size stays symbolic
    except for ``rawptr_static``, which is compiled per ``(tiles_total, tiles_1)``."""
    key = (device_index, variant, to_split, static)
    if key not in _compiled:
        family, p, nt, _, minb = VARIANTS[variant]
        i32 = cutlass.Int32(0)
        i64 = cutlass.Int64(0)
        with torch.cuda.device(device_index):
            if family == "rawptr":
                fn = cute.compile(seq_rawptr_ctm3_host, i64, i64, i64, i32, i32, i32, i32,
                                  make_fake_stream(), to_split, p, nt, options="--enable-tvm-ffi")
            elif family == "rawptr_static":
                fn = cute.compile(seq_rawptr_static_ctm3_host, i64, i64, i64, i32,
                                  make_fake_stream(), to_split, p, nt, static[0], static[1],
                                  options="--enable-tvm-ffi")
            elif family == "tiled" and minb > 0:
                fn = cute.compile(seq_tiled_pinned_ctm3_host, *_fakes(), i32, i32, i32, i32,
                                  make_fake_stream(), to_split, p, nt, minb, options="--enable-tvm-ffi")
            elif family == "tiled":
                fn = cute.compile(seq_tiled_ctm3_host, *_fakes(), i32, i32, i32, i32,
                                  make_fake_stream(), to_split, p, nt, options="--enable-tvm-ffi")
            elif family == "bulk":
                fn = cute.compile(seq_bulk_ctm3_host, *_fakes(), i32, i32, i32, i32,
                                  make_fake_stream(), to_split, p, minb,
                                  SMEM_CARVEOUT_PCT, options="--enable-tvm-ffi")
            else:
                fn = cute.compile(seq_general_ctm3_host, *_fakes(), i32, i32, i32, i32,
                                  make_fake_stream(), to_split, options="--enable-tvm-ffi")
        _compiled[key] = fn
    return _compiled[key]


def _sm_count(device_index: int) -> int:
    if device_index not in _sms:
        _sms[device_index] = torch.cuda.get_device_properties(device_index).multi_processor_count
    return _sms[device_index]


def grid_for(variant: str, device_index: int, n_tiles: int, batch: int, per_batch: int) -> int:
    """CTAs in x. Tiled ``oneshot``: one CTA per tile. Tiled ``persistent``:
    ``min(n_tiles, SMs x 8)``. Bulk: ``min(n_tiles, SMs x pinned)``. General:
    ``max(1, min(ceil(vectors/THREADS), SMs x 8 // B))`` per batch element."""
    family, _, _, mode, minb = VARIANTS[variant]
    sms = _sm_count(device_index)
    if family in _TILED_FAMILIES:
        if mode == "oneshot":
            return max(1, n_tiles)
        return max(1, min(n_tiles, sms * _TILED_CTAS_PER_SM))
    if family == "bulk":
        return max(1, min(n_tiles, sms * minb))
    return max(1, min(-(-per_batch // THREADS), sms * _TILED_CTAS_PER_SM // batch))


def _units(t: torch.Tensor) -> torch.Tensor:
    """The flat 16-bit view of a contiguous tensor (a pure copy is dtype-blind)."""
    return t.view(-1).view(_UNIT)


def _check(name: str, tensors: list[torch.Tensor], row_bytes: int) -> None:
    for t in tensors:
        if not t.is_cuda:
            raise ValueError(f"{name}: tensors must be CUDA")
        if not t.is_contiguous():
            raise ValueError(f"{name}: tensors must be contiguous")
        if t.data_ptr() % 16:
            raise ValueError(f"{name}: base pointer must be 16-byte aligned")
        if t.numel() * t.element_size() > 2 * _INT32_MAX:
            raise ValueError(f"{name}: tensor exceeds the Int32 index range")
    if row_bytes % 16:
        raise ValueError(f"{name}: a row must be a multiple of 16 bytes, got {row_bytes}")


def _stream(dev: int) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)


def _launch(variant: str, to_split: bool, joined: torch.Tensor, part1: torch.Tensor,
            part2: torch.Tensor, name: str) -> None:
    batch, l1, l2 = joined.shape[0], part1.shape[1], part2.shape[1]
    row_bytes = math.prod(joined.shape[2:]) * joined.element_size()
    _check(name, [joined, part1, part2], row_bytes)
    dev = joined.device.index
    family = VARIANTS[variant][0]
    tb = tile_bytes(variant)
    b1, b2 = l1 * row_bytes, l2 * row_bytes
    if family != "general" and (b1 % tb or b2 % tb):
        raise ValueError(f"{name}[{variant}]: parts must be whole {tb}-byte tiles, "
                         f"got {b1} and {b2} bytes")
    if family == "general":
        per_batch = (b1 + b2) // 16
        first = b1 // 16
        gx = grid_for(variant, dev, 0, batch, per_batch)
        _compile(dev, variant, to_split)(
            _units(joined), _units(part1), _units(part2), per_batch, first, gx, batch, _stream(dev))
        return
    tiles_total = (b1 + b2) // tb
    tiles_1 = b1 // tb
    n_tiles = tiles_total * batch
    gx = grid_for(variant, dev, n_tiles, batch, 0)
    if family == "rawptr":
        _compile(dev, variant, to_split)(
            joined.data_ptr(), part1.data_ptr(), part2.data_ptr(), tiles_total, tiles_1, n_tiles,
            gx, _stream(dev))
        return
    if family == "rawptr_static":
        _compile(dev, variant, to_split, (tiles_total, tiles_1))(
            joined.data_ptr(), part1.data_ptr(), part2.data_ptr(), gx, _stream(dev))
        return
    _compile(dev, variant, to_split)(
        _units(joined), _units(part1), _units(part2), tiles_total, tiles_1, n_tiles, gx,
        _stream(dev))


def _split(variant: str | None, tensor: torch.Tensor, split_at: int, name: str):
    if tensor.dim() < 3:
        raise ValueError(f"{name}: expected [B, L, ...], got {tuple(tensor.shape)}")
    if not 0 < split_at < tensor.shape[1]:
        raise ValueError(f"{name}: split_at must be in (0, L={tensor.shape[1]}), got {split_at}")
    trailing = tensor.shape[2:]
    row_bytes = math.prod(trailing) * tensor.element_size()
    if variant is None:
        variant = pick_variant(split_at * row_bytes, (tensor.shape[1] - split_at) * row_bytes)
    out1 = torch.empty(tensor.shape[0], split_at, *trailing, dtype=tensor.dtype,
                       device=tensor.device)
    out2 = torch.empty(tensor.shape[0], tensor.shape[1] - split_at, *trailing,
                       dtype=tensor.dtype, device=tensor.device)
    _launch(variant, True, tensor, out1, out2, name)
    return out1, out2


def _join(variant: str | None, t1: torch.Tensor, t2: torch.Tensor, name: str) -> torch.Tensor:
    if t1.dim() < 3 or t1.shape[0] != t2.shape[0] or t1.shape[2:] != t2.shape[2:]:
        raise ValueError(f"{name}: shapes {tuple(t1.shape)} and {tuple(t2.shape)} do not join")
    if t1.dtype != t2.dtype or t1.device != t2.device:
        raise ValueError(f"{name}: dtype/device mismatch")
    if min(t1.shape[1], t2.shape[1]) == 0:
        raise ValueError(f"{name}: both parts must have L > 0")
    row_bytes = math.prod(t1.shape[2:]) * t1.element_size()
    if variant is None:
        variant = pick_variant(t1.shape[1] * row_bytes, t2.shape[1] * row_bytes)
    out = torch.empty(t1.shape[0], t1.shape[1] + t2.shape[1], *t1.shape[2:],
                      dtype=t1.dtype, device=t1.device)
    _launch(variant, False, out, t1, t2, name)
    return out


def seq_split_ctm3(tensor: torch.Tensor, split_at: int,
                   variant: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``tensor`` on dim 1 at ``split_at`` into two fresh contiguous tensors.

    Bitwise ``flux.seq_split``: ``(tensor[:, :split_at], tensor[:, split_at:])``.

    Args:
        tensor: Contiguous CUDA ``[B, L, ...]`` tensor whose row (``...``) is a
            multiple of 16 bytes, base pointer 16-byte aligned.
        split_at: Rows in the first output, ``0 < split_at < L``.
        variant: One of :data:`VARIANTS`, or ``None`` for :func:`pick_variant`.

    Returns:
        ``(first, second)``, ``[B, split_at, ...]`` and ``[B, L - split_at, ...]``.

    Raises:
        ValueError: On a non-CUDA, non-contiguous, misaligned or under-3-D
            tensor, a ``split_at`` outside ``(0, L)``, or a tiled/bulk variant on
            a shape whose parts are not whole tiles.
    """
    return _split(variant, tensor, split_at, "seq_split_ctm3")


def seq_join_ctm3(t1: torch.Tensor, t2: torch.Tensor,
                  variant: str | None = None) -> torch.Tensor:
    """Concatenate ``t1`` and ``t2`` on dim 1 into one fresh contiguous tensor.

    Bitwise ``flux.seq_join`` (= ``torch.cat([t1, t2], dim=1)``).

    Args:
        t1: Contiguous CUDA ``[B, L1, ...]`` tensor, row a multiple of 16 bytes.
        t2: Contiguous CUDA ``[B, L2, ...]`` tensor of the same dtype and trailing shape.
        variant: One of :data:`VARIANTS`, or ``None`` for :func:`pick_variant`.

    Returns:
        ``[B, L1 + L2, ...]``.

    Raises:
        ValueError: On incompatible shapes/dtypes, an empty part, or the
            alignment/tile conditions of :func:`seq_split_ctm3`.
    """
    return _join(variant, t1, t2, "seq_join_ctm3")


def verify(batch: int = 4) -> None:
    """Compile, run and bit-check every arm and the composite against torch slicing.

    The smoke check the ``ctm-write-kernel`` skill requires of a kernel module;
    the full contract is ``test_seq_split_join_ctm3.py``.

    Raises:
        AssertionError: If any arm is not bitwise the torch reference.
    """
    torch.manual_seed(batch)
    x = torch.randn(batch, 512, 3072, device="cuda", dtype=torch.bfloat16)
    r1, r2 = x[:, :256].contiguous(), x[:, 256:].contiguous()
    for v in list(VARIANTS) + [None]:
        o1, o2 = seq_split_ctm3(x, 256, v)
        assert torch.equal(o1, r1) and torch.equal(o2, r2), f"split {v}: not bitwise"
        assert torch.equal(seq_join_ctm3(r1, r2, v), x), f"join {v}: not bitwise"
    torch.cuda.synchronize()
    print(f"seq_split_join_ctm3.verify B={batch}: {len(VARIANTS)} arms + composite, both "
          f"directions, bitwise torch")


# ---------------------------------------------------------------------------
# torch.library registration
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define("seq_split_ctm3(Tensor tensor, int split_at) -> (Tensor, Tensor)")
_lib.define("seq_join_ctm3(Tensor t1, Tensor t2) -> Tensor")
for _name in VARIANTS:
    _lib.define(f"seq_split_ctm3_{_name}(Tensor tensor, int split_at) -> (Tensor, Tensor)")
    _lib.define(f"seq_join_ctm3_{_name}(Tensor t1, Tensor t2) -> Tensor")


@torch.library.impl(_lib, "seq_split_ctm3", "CUDA")
def _seq_split_ctm3_impl(tensor: torch.Tensor, split_at: int) -> tuple[torch.Tensor, torch.Tensor]:
    return seq_split_ctm3(tensor, split_at)


@torch.library.impl(_lib, "seq_join_ctm3", "CUDA")
def _seq_join_ctm3_impl(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return seq_join_ctm3(t1, t2)


def _split_impl(name: str):
    def impl(tensor: torch.Tensor, split_at: int) -> tuple[torch.Tensor, torch.Tensor]:
        return seq_split_ctm3(tensor, split_at, name)
    return impl


def _join_impl(name: str):
    def impl(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return seq_join_ctm3(t1, t2, name)
    return impl


for _name in VARIANTS:
    _lib.impl(f"seq_split_ctm3_{_name}", _split_impl(_name), "CUDA")
    _lib.impl(f"seq_join_ctm3_{_name}", _join_impl(_name), "CUDA")
