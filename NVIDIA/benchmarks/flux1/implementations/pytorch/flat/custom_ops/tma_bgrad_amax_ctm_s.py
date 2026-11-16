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

"""``torch.ops.flux.tma_bgrad_amax_ctm_s`` -- one-read dbias + amax reader on the
CUTLASS DSL: **one CTA (or one small cluster) per 64-column strip, no partial slots,
no finaliser**.

Fourth CUTLASS-DSL attempt at ``tma_bgrad_amax_kernel.cu``.
Where the op still launches in production -- cast views [1024], [2048], [4096] x 9216,
L2-resident -- the three earlier attempts measured the same two
constants: a 2.9-3.4 us finaliser kernel plus its in-graph launch at every size, and
a two-level last-CTA fold whose five dependent gpu-scope round trips cost the
same 3.3-4.4 us it deleted. The reader body here is TE's fragment math unchanged
(what makes the amax pair bitwise); the *structure* changes:

* **J = 1.** CTA ``cb`` owns column strip ``[64 cb, 64 cb + 64)`` for all M rows (a
  ``CLUSTER`` of 2 or 4 CTAs may split the rows and combine through distributed
  shared memory). The column sums it accumulates *are* dbias: no ``[J][N]`` slot
  workspace, nothing to finalise, one bf16 store per column.
* **One completion level, two dependent round trips.** The only grid-wide
  reduction left is the amax pair (``n_cb x 2`` floats). Each strip's CTA writes
  its pair and issues one ``atom.add.acq_rel.gpu`` on a per-device counter; the
  CTA that observes ``n_cb - 1`` is the grid's last, loads the pairs with
  ``ld.relaxed.gpu`` (morally strong: served from L2, never a stale L1 line),
  writes ``amax`` and resets the counter for the next launch or graph replay.
* **Fewer, deeper CTAs** -- one CTA per
  SM streams 16-64 tiles through a 16-stage (128 KB) ring with eight consumer
  warps (two per 16-column slice), so the per-SM ``mma`` epilogue is not the
  bottleneck four warps would make it.

Occupancy is pinned, not inferred (``min_blocks_per_mp`` per variant and
``preferred_smem_carveout=100`` on every launch). Written on
the container's public ``nvidia-cutlass-dsl`` 4.8.0: ``cute.arch.*``,
``cutlass.utils.SmemAllocator``, ``cutlass.pipeline.NamedBarrier``, the raw
``nvvm.mma_sync`` dialect op , ``cute.arch.atomic_add(sem=,
scope=)`` and, for the cluster variants only, ``cute.arch.cluster_arrive/wait``
and ``cutlass.experimental.primitives.mapa``. ``CUTE_DSL_ARCH`` must stay unset.

Contract (checked in ``tests/test_tma_bgrad_amax_ctm_s.py``): both
amaxes bitwise TE's;
``dbias`` within ``aten.sum`` tolerance and byte-equal across launches and graph
replays (a fixed order of its own); the cast
with the emitted pair bitwise ``tex.quantize``; capture-safe incl. the 48-replay
pressure test; the completion counter back at zero after every call; exactly one
kernel per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
import torch.library
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass._mlir.extras import types as T
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

TILE = 64                        # rows and columns per TMA box (SWIZZLE_128B: 64 bf16 = 128 B)
TILE_BYTES = TILE * TILE * 2     # 8192
SLICES = 4                       # 16-column slices per tile; each has WPS consumer warps
_ROWS_PER_GROUP = 16
_ONES_BF16X2 = 0x3F803F80        # bf16x2 {1.0, 1.0}
_BF16_HI_MASK = -65536           # 0xFFFF0000 as a signed Int32

# TE's fragment instructions (util/ptx.cuh, hadamard_transform_utils.cuh @ 27486e03),
# emitted verbatim: the exact instruction is what makes the amax pair bitwise.
_LDMATRIX_X4_TRANS = (
    "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
    "{ {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];"
)
_MAX_XORSIGN_ABS_BF16X2 = "max.xorsign.abs.bf16x2 {$w0}, {$r0}, {$r1};"
# A morally strong load for the grid-last CTA's read of the other CTAs' amax pairs:
# served at gpu scope, so an L1 line this SM may hold from an earlier launch cannot
# be returned. Address operand only -- no float constant crosses inline_ptx.
_LD_RELAXED_GPU_F32 = "ld.relaxed.gpu.global.f32 {$w0}, [{$r0}];"
_MMA_SHAPE = "#nvvm.shape<m = 16, n = 8, k = 16>"

I32 = cutlass.Int32
I64 = cutlass.Int64
F32 = cutlass.Float32
BF16 = cutlass.BFloat16


@dataclass(frozen=True)
class Variant:
    """One compile-time configuration of the reader.

    Attributes:
        name: ``"S"``, ``"S4"``, ``"SD"``, ``"C2"`` or ``"C4"``.
        cluster: CTAs per column strip = thread-block cluster size along rows.
            1 means no cluster launch and no distributed-shared-memory traffic.
        wps: Consumer warps per 16-column slice (1 or 2).
        stages: Ring depth in 64 x 64 bf16 tiles (8 KB each).
        min_blocks: ``min_blocks_per_mp`` pin for the launch.
    """

    name: str
    cluster: int
    wps: int
    stages: int
    min_blocks: int
    tile_m: int = 64      # rows per TMA box (64 or 128); columns are always 64 (SW128)
    red: bool = False     # round-3 completion: red.max accumulators, thread-0-only last-CTA tail

    @property
    def consumer_warps(self) -> int:
        """Consumer warps per CTA."""
        return SLICES * self.wps

    @property
    def threads(self) -> int:
        """Threads per CTA: the consumer warps plus one TMA producer warp."""
        return 32 * (self.consumer_warps + 1)


VARIANTS: dict[str, Variant] = {
    "S": Variant("S", 1, 2, 16, 1),
    "S4": Variant("S4", 1, 1, 16, 1),
    "SD": Variant("SD", 1, 2, 8, 1),
    "C2": Variant("C2", 2, 2, 8, 2),
    "C4": Variant("C4", 4, 1, 4, 4),
    # the completion tail and the box geometry
    "SR": Variant("SR", 1, 2, 16, 1, tile_m=64, red=True),
    "T": Variant("T", 1, 2, 8, 1, tile_m=128, red=True),
}


#: Elements of the cast view above which C4 is dispatched.
#: A size expression, never a batch number: [2048, 9216] and below take S (one CTA
#: per strip, 16-64 tiles each); [4096, 9216] and above take C4 (four CTAs per strip
#: -- the CUDA op's CTA count -- with the dbias combined through DSMEM). Round 1
#: measured S best at the two small views and C4 best at [4096] on both clusters.
COMPOSITE_S_MAX_NUMEL = 2048 * 9216


def variant_for(m: int, n: int) -> Variant:
    """The dispatched variant.

    Round 1: S everywhere. Round 2: S at ``M * N <= COMPOSITE_S_MAX_NUMEL``, C4 above
    (measured: nothing resolvable over S). Round 3: SR
    everywhere (measured: the completion tail is the atomic round trips, SR = S on
    GB300). Round 4: **T everywhere** -- 128-row TMA
    boxes, half the box issues per byte, the round-3 completion
    """
    return VARIANTS["T"]


# ---------------------------------------------------------------------------
# Device helpers (traced, not preprocessed: plain Python loops only)
# ---------------------------------------------------------------------------


def _ldmatrix_x4_trans(addr):
    return cute.arch.inline_ptx(
        _LDMATRIX_X4_TRANS, write_only_types=[I32] * 4, read_only_args=[addr],
    )


def _mma_16816(a, b0, b1, c):
    """D = A(16x16 bf16, row) * B(16x8 bf16, col) + C, fp32 accumulate -- TE's instruction."""
    res = nvvm.mma_sync(
        llvm.StructType.get_literal([T.f32()] * 4),
        ir.Attribute.parse(_MMA_SHAPE),
        nvvm.MMALayout.row,
        nvvm.MMALayout.col,
        [v.ir_value() for v in a],
        [b0.ir_value(), b1.ir_value()],
        [v.ir_value() for v in c],
        multiplicand_a_ptx_type=nvvm.MMATypes.bf16,
        multiplicand_b_ptx_type=nvvm.MMATypes.bf16,
    )
    return tuple(F32(llvm.extractvalue(T.f32(), res, [i])) for i in range(4))


def _max_xorsign_abs(a, b):
    return cute.arch.inline_ptx(
        _MAX_XORSIGN_ABS_BF16X2, write_only_types=[I32], read_only_args=[a, b],
    )


_LD_RELAXED_GPU_B32 = "ld.relaxed.gpu.global.b32 {$w0}, [{$r0}];"


def _ld_strong_i32(ptr):
    """``ld.relaxed.gpu.global.b32`` through a global pointer (the u32 max accumulators)."""
    return cute.arch.inline_ptx(
        _LD_RELAXED_GPU_B32, write_only_types=[I32], read_only_args=[ptr],
    )


def _ld_strong_f32(ptr):
    """``ld.relaxed.gpu.global.f32`` through a global pointer (see ``_LD_RELAXED_GPU_F32``)."""
    return cute.arch.inline_ptx(
        _LD_RELAXED_GPU_F32, write_only_types=[F32], read_only_args=[ptr],
    )


def _bf16_lo(w):
    return (w << 16).bitcast(F32)


def _bf16_hi(w):
    return (w & I32(_BF16_HI_MASK)).bitcast(F32)


def _round_bf16(x):
    return x.to(BF16).to(F32)


def _warp_max(v):
    """Butterfly max over the warp -- exact, and identical to the CUDA op's."""
    for off in (16, 8, 4, 2, 1):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=off))
    return v


def _swizzled_tile_offset(lane, slice_):
    """Byte offset of this lane's ldmatrix row within a 16-row block of a SW128 tile.

    TE's lane mapping: lanes 0-15 take rows 0-15 of the slice's first 16 B column
    chunk, lanes 16-31 the second; a 16-row block starts at a multiple of 2048 B,
    so the 128B-swizzle term (``row & 7``) is the same for every block.
    """
    row = lane & 15
    chunk = (lane >> 4) + slice_ * 2
    return row * (TILE * 2) + ((chunk ^ (row & 7)) << 4)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@cute.kernel
def tma_bgrad_amax_ctm_s_kernel(
    tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor, smem_layout: cute.ComposedLayout,
    ws_amax: cute.Tensor, rht: cute.Tensor, group_amax: cute.Tensor,
    dbias: cute.Tensor, amax: cute.Tensor, counter: cute.Tensor,
    m: I32, n: I32, n_cb: I32, n_rb: I32,
    STAGES: cutlass.Constexpr[int], WPS: cutlass.Constexpr[int],
    CLUSTER: cutlass.Constexpr[int], GROUP: cutlass.Constexpr[bool],
    TILE_M: cutlass.Constexpr[int], RED: cutlass.Constexpr[bool],
) -> None:
    NW = SLICES * WPS                    # consumer warps
    NT = 32 * NW                         # consumer threads
    RB_PER_WARP = (TILE_M // 16) // WPS  # 16-row blocks per warp per tile
    BOX_BYTES = TILE_M * TILE * 2        # one TMA box

    tidx, _, _ = cute.arch.thread_idx()
    rank, cb, _ = cute.arch.block_idx()  # rank within the strip's cluster (0 if none); strip
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    slice_ = warp % SLICES              # this warp's 16-column slice of the tile
    half = warp // SLICES               # which RB_PER_WARP row-blocks of each tile it takes
    gi = lane >> 2
    tq = lane & 3

    smem = utils.SmemAllocator()
    ring = smem.allocate_array(BF16, STAGES * TILE_M * TILE, byte_alignment=1024)
    s_tile = cute.make_tensor(
        cute.recast_ptr(ring, smem_layout.inner, dtype=BF16), smem_layout.outer,
    )
    full = smem.allocate_array(I64, STAGES, byte_alignment=8)     # TMA landed (tx count)
    empty = smem.allocate_array(I64, STAGES, byte_alignment=8)    # every consumer warp done
    amax_stage = cutlass.Pointer(
        smem.allocate_array(F32, 2 * NW, byte_alignment=16), dtype=F32)
    # [half][64]: the column partials of halves 1.. WPS-1, summed into half 0's in order
    s_part = cutlass.Pointer(
        smem.allocate_array(F32, TILE * WPS, byte_alignment=16), dtype=F32)
    # cluster variants: this CTA's 64 column totals and its amax pair, read by rank 0
    s_tot = cutlass.Pointer(smem.allocate_array(F32, TILE, byte_alignment=16), dtype=F32)
    s_pair = cutlass.Pointer(smem.allocate_array(F32, 2, byte_alignment=8), dtype=F32)
    s_flag = cutlass.Pointer(smem.allocate_array(I32, 1, byte_alignment=8), dtype=I32)

    if tidx == 0:
        for s in cutlass.range_constexpr(STAGES):
            cute.arch.mbarrier_init(full + s, 1)
            cute.arch.mbarrier_init(empty + s, NW)
        s_flag.store(I32(0))
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    g_tiles = cute.local_tile(tma_tensor, (TILE_M, TILE), (None, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1),
        cute.group_modes(s_tile, 0, 2), cute.group_modes(g_tiles, 0, 2),
    )

    if warp == NW:
        # ---- producer warp: this CTA's row tiles of strip cb, through the ring ----
        # PipelineTmaAsync.producer_acquire = wait(empty) + ONE elected arrive.expect_tx
        # on full (its producer_commit is a no-op: the TMA completes the tx bytes).
        # Without the arrive, full never completes its count and every consumer waits
        # forever -- it hung exactly there. The wait is on the fresh barrier's
        # preceding phase (parity 1) the first time round.
        k = I32(0)
        for rb in cutlass.range(rank, n_rb, CLUSTER):
            s = k % STAGES
            ph = (k // STAGES) & 1
            cute.arch.mbarrier_wait(empty + s, ph ^ 1)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(full + s, BOX_BYTES)
            cute.copy(
                tma_atom, tAgA[(None, rb, cb)], tAsA[(None, s)], tma_bar_ptr=full + s,
            )
            k = k + 1

    if warp < NW:
        # ---- consumer warps: slice `slice_`, row-blocks half*RB_PER_WARP.. of each tile ----
        bfrag = []
        for jb in cutlass.range_constexpr(2):
            nn = jb * 8 + gi
            r0 = rht[(2 * tq) * 16 + nn].to(I32) & 0xFFFF
            r1 = rht[(2 * tq + 1) * 16 + nn].to(I32) & 0xFFFF
            r8 = rht[(2 * tq + 8) * 16 + nn].to(I32) & 0xFFFF
            r9 = rht[(2 * tq + 9) * 16 + nn].to(I32) & 0xFFFF
            bfrag.append((r0 | (r1 << 16), r8 | (r9 << 16)))
        ones = I32(_ONES_BF16X2)
        swz = _swizzled_tile_offset(lane, slice_)
        ring_base = ring.toint()

        rmax_packed = I32(0)
        cmax = F32(0.0)
        c0 = F32(0.0)
        c1 = F32(0.0)
        c2 = F32(0.0)
        c3 = F32(0.0)
        k = I32(0)
        for rb in cutlass.range(rank, n_rb, CLUSTER):
            s = k % STAGES
            ph = (k // STAGES) & 1
            cute.arch.mbarrier_wait(full + s, ph)
            tile_addr = ring_base + s * BOX_BYTES + swz
            for i in cutlass.range_constexpr(RB_PER_WARP):
                rb16 = half * RB_PER_WARP + i
                f0, f1, f2, f3 = _ldmatrix_x4_trans(tile_addr + rb16 * (16 * TILE * 2))
                rmax_packed = _max_xorsign_abs(
                    rmax_packed,
                    _max_xorsign_abs(_max_xorsign_abs(f0, f1), _max_xorsign_abs(f2, f3)),
                )
                a = (f0, f2, f1, f3)          # A = X^T for mma.sync: TE's swap
                zero = (F32(0.0),) * 4
                d0 = _mma_16816(a, bfrag[0][0], bfrag[0][1], zero)
                d1 = _mma_16816(a, bfrag[1][0], bfrag[1][1], zero)
                tmax = F32(0.0)
                for q in cutlass.range_constexpr(4):
                    tmax = cute.arch.fmax(tmax, cute.math.abs(d0[q]))
                    tmax = cute.arch.fmax(tmax, cute.math.abs(d1[q]))
                cmax = cute.arch.fmax(cmax, tmax)
                if cutlass.const_expr(GROUP):
                    wmax = _warp_max(tmax)
                    row = rb * TILE_M + rb16 * 16
                    if lane == 0 and row < m:
                        gp = cutlass.Pointer(group_amax.iterator, dtype=F32)
                        cute.arch.atomic_fmax(
                            gp + row // _ROWS_PER_GROUP, _round_bf16(wmax), sign_bit=False,
                        )
                c0, c1, c2, c3 = _mma_16816(a, ones, ones, (c0, c1, c2, c3))
            # ldmatrix is warp-collective, so once this lane has consumed its fragment
            # every lane's read of stage s is done: one arrive per warp frees the slot.
            if lane == 0:
                cute.arch.mbarrier_arrive(empty + s)
            k = k + 1

        # ---- CTA epilogue 1: column partials of halves 1.. into smem ---------------
        if cutlass.const_expr(WPS > 1):
            if half > 0:
                if tq == 0:
                    (s_part + (half * TILE + slice_ * 16 + gi)).store(c0)
                    (s_part + (half * TILE + slice_ * 16 + gi + 8)).store(c2)

        # ---- CTA epilogue 2: warp amax pair into smem ---------------------------------
        rmax = _warp_max(cute.arch.fmax(
            cute.math.abs(_bf16_lo(rmax_packed)), cute.math.abs(_bf16_hi(rmax_packed)),
        ))
        cmax = _warp_max(cmax)
        if lane == 0:
            (amax_stage + 2 * warp).store(rmax)
            (amax_stage + 2 * warp + 1).store(cmax)
        pipeline.NamedBarrier(barrier_id=1, num_threads=NT).arrive_and_wait()

        # ---- CTA epilogue 3: dbias -- half 0 sums the halves in fixed order -----------
        if half == 0:
            if tq == 0:
                t0 = c0
                t8 = c2
                if cutlass.const_expr(WPS > 1):
                    for hh in cutlass.range_constexpr(1, WPS):
                        t0 = t0 + (s_part + (hh * TILE + slice_ * 16 + gi)).load()
                        t8 = t8 + (s_part + (hh * TILE + slice_ * 16 + gi + 8)).load()
                if cutlass.const_expr(CLUSTER == 1):
                    col = cb * TILE + slice_ * 16 + gi
                    if col < n:
                        dbias[col] = t0.to(BF16)
                    if col + 8 < n:
                        dbias[col + 8] = t8.to(BF16)
                else:
                    (s_tot + (slice_ * 16 + gi)).store(t0)
                    (s_tot + (slice_ * 16 + gi + 8)).store(t8)

        # ---- CTA epilogue 4: the CTA's amax pair; no cluster -> release it to the grid --
        if tidx == 0:
            r = F32(0.0)
            c = F32(0.0)
            for w in cutlass.range_constexpr(NW):
                r = cute.arch.fmax(r, (amax_stage + 2 * w).load())
                c = cute.arch.fmax(c, (amax_stage + 2 * w + 1).load())
            if cutlass.const_expr(CLUSTER == 1):
                if cutlass.const_expr(RED):
                    # Round 3: the pair travels through two fire-and-forget u32 max
                    # atomics (non-negative floats order as u32, so the max is exact
                    # and order-free); the release on the counter atomic orders them.
                    # Only the grid-last CTA's thread 0 does anything after this.
                    cute.arch.atomic_max(counter.iterator + 1, r.bitcast(I32), sem="relaxed", scope="gpu")
                    cute.arch.atomic_max(counter.iterator + 2, c.bitcast(I32), sem="relaxed", scope="gpu")
                    prev = cute.arch.atomic_add(
                        counter.iterator, I32(1), sem="acq_rel", scope="gpu",
                    )
                    if prev == n_cb - 1:
                        r_all = _ld_strong_i32(counter.iterator + 1).bitcast(F32)
                        c_all = _ld_strong_i32(counter.iterator + 2).bitcast(F32)
                        amax[0] = r_all                  # already on the bf16 grid
                        amax[1] = _round_bf16(c_all)     # TE rounds the MMA result to bf16 before the max
                        counter[1] = I32(0)
                        counter[2] = I32(0)
                        counter[0] = I32(0)              # the next launch (or replay) starts clean
                else:
                    ws_amax[2 * cb] = r
                    ws_amax[2 * cb + 1] = c
                    # One instruction releases this thread's two stores at gpu scope and
                    # acquires every earlier CTA's -- the whole completion handshake.
                    prev = cute.arch.atomic_add(
                        counter.iterator, I32(1), sem="acq_rel", scope="gpu",
                    )
                    if prev == n_cb - 1:
                        s_flag.store(I32(1))
            else:
                s_pair.store(r)
                (s_pair + 1).store(c)

    if cutlass.const_expr(CLUSTER > 1):
        # ---- cluster: rank 0 pulls the peers' column totals and amax pairs (DSMEM) -----
        # arrive (release) / wait (acquire) at cluster scope order the peers' smem
        # stores before rank 0's mapa loads; the second barrier keeps every peer's smem
        # alive until rank 0 has finished reading it.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp < NW:
            if rank == 0:
                if half == 0:
                    if tq == 0:
                        idx = slice_ * 16 + gi
                        t0 = (s_tot + idx).load()
                        t8 = (s_tot + idx + 8).load()
                        for rr in cutlass.range_constexpr(1, CLUSTER):
                            peer = prims.mapa(s_tot, rr)
                            t0 = t0 + (peer + idx).load()
                            t8 = t8 + (peer + idx + 8).load()
                        col = cb * TILE + idx
                        if col < n:
                            dbias[col] = t0.to(BF16)
                        if col + 8 < n:
                            dbias[col + 8] = t8.to(BF16)
                if tidx == 0:
                    r = s_pair.load()
                    c = (s_pair + 1).load()
                    for rr in cutlass.range_constexpr(1, CLUSTER):
                        pp = prims.mapa(s_pair, rr)
                        r = cute.arch.fmax(r, pp.load())
                        c = cute.arch.fmax(c, (pp + 1).load())
                    ws_amax[2 * cb] = r
                    ws_amax[2 * cb + 1] = c
                    prev = cute.arch.atomic_add(
                        counter.iterator, I32(1), sem="acq_rel", scope="gpu",
                    )
                    if prev == n_cb - 1:
                        s_flag.store(I32(1))
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    if cutlass.const_expr(not RED):
      if warp < NW:
        if rank == 0:
            # ---- grid-last CTA: the amax pair from the n_cb strip pairs ------------------
            pipeline.NamedBarrier(barrier_id=1, num_threads=NT).arrive_and_wait()
            if s_flag.load() == 1:
                gr = F32(0.0)
                gc = F32(0.0)
                for i in cutlass.range(tidx, n_cb, NT):
                    gr = cute.arch.fmax(gr, _ld_strong_f32(ws_amax.iterator + 2 * i))
                    gc = cute.arch.fmax(gc, _ld_strong_f32(ws_amax.iterator + 2 * i + 1))
                gr = _warp_max(gr)
                gc = _warp_max(gc)
                if lane == 0:
                    (amax_stage + 2 * warp).store(gr)
                    (amax_stage + 2 * warp + 1).store(gc)
                pipeline.NamedBarrier(barrier_id=1, num_threads=NT).arrive_and_wait()
                if tidx == 0:
                    r2 = F32(0.0)
                    c2f = F32(0.0)
                    for w in cutlass.range_constexpr(NW):
                        r2 = cute.arch.fmax(r2, (amax_stage + 2 * w).load())
                        c2f = cute.arch.fmax(c2f, (amax_stage + 2 * w + 1).load())
                    amax[0] = r2                    # already on the bf16 grid: max of bf16 values
                    amax[1] = _round_bf16(c2f)      # TE rounds the MMA result to bf16 before the max
                    counter[0] = I32(0)             # the next launch (or graph replay) starts clean


@cute.jit
def tma_bgrad_amax_ctm_s_host(
    x: cute.Tensor, ws_amax: cute.Tensor, rht: cute.Tensor, group_amax: cute.Tensor,
    dbias: cute.Tensor, amax: cute.Tensor, counter: cute.Tensor,
    m: I32, n: I32, n_cb: I32, n_rb: I32, stream: cuda.CUstream,
    STAGES: cutlass.Constexpr[int], WPS: cutlass.Constexpr[int],
    CLUSTER: cutlass.Constexpr[int], MINB: cutlass.Constexpr[int],
    GROUP: cutlass.Constexpr[bool], TILE_M: cutlass.Constexpr[int],
    RED: cutlass.Constexpr[bool],
) -> None:
    atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
    smem_layout = cute.coalesce(
        cute.tile_to_shape(atom, (TILE_M, TILE, STAGES), order=(0, 1, 2)),
        target_profile=(1, 1, 1),
    )
    tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), x, smem_layout, (TILE_M, TILE),
    )
    threads = 32 * (SLICES * WPS + 1)
    kernel = tma_bgrad_amax_ctm_s_kernel(
        tma_atom, tma_tensor, smem_layout, ws_amax, rht, group_amax,
        dbias, amax, counter, m, n, n_cb, n_rb, STAGES, WPS, CLUSTER, GROUP, TILE_M, RED,
    )
    if cutlass.const_expr(CLUSTER > 1):
        kernel.launch(
            grid=(CLUSTER, n_cb, 1), block=(threads, 1, 1), cluster=(CLUSTER, 1, 1),
            stream=stream, min_blocks_per_mp=MINB, preferred_smem_carveout=100,
        )
    else:
        kernel.launch(
            grid=(1, n_cb, 1), block=(threads, 1, 1),
            stream=stream, min_blocks_per_mp=MINB, preferred_smem_carveout=100,
        )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_compiled: dict[tuple[int, str, bool], Callable] = {}
_counters: dict[int, torch.Tensor] = {}
_warned_capture_first = False


def _compile(device_index: int, variant: Variant, group: bool) -> Callable:
    """Compile once per (device, variant, GROUP); M and N stay symbolic."""
    key = (device_index, variant.name, group)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            sym = cute.sym_int64
            m_sym, n_sym = sym(16, "m"), sym(16, "n")
            fx = make_fake_compact_tensor(BF16, (m_sym, n_sym), stride_order=(1, 0), assumed_align=16)
            f_ws_amax = make_fake_compact_tensor(F32, (sym(2, "wa"),), assumed_align=16)
            f_rht = make_fake_compact_tensor(cutlass.Int16, (256,), assumed_align=16)
            f_group = make_fake_compact_tensor(F32, (sym(1, "g"),), assumed_align=4)
            f_dbias = make_fake_compact_tensor(BF16, (n_sym,), assumed_align=16)
            f_amax = make_fake_compact_tensor(F32, (2,), assumed_align=8)
            f_counter = make_fake_compact_tensor(I32, (sym(1, "c"),), assumed_align=8)
            ints = [I32(0)] * 4
            _compiled[key] = cute.compile(
                tma_bgrad_amax_ctm_s_host, fx, f_ws_amax, f_rht, f_group, f_dbias, f_amax,
                f_counter, *ints, make_fake_stream(),
                variant.stages, variant.wps, variant.cluster, variant.min_blocks, group,
                variant.tile_m, variant.red,
                options="--enable-tvm-ffi",
            )
    return _compiled[key]


def geometry(m: int, n: int, variant: Variant) -> tuple[int, int, tuple[int, int], int]:
    """``(n_cb, n_rb, grid, ctas)`` for an ``[M, N]`` input under ``variant``.

    ``n_cb = ceil(N / 64)`` strips, ``n_rb = ceil(M / 64)`` row tiles; the grid is
    ``(cluster, n_cb)`` -- one CTA (or one cluster of ``variant.cluster`` CTAs
    splitting the rows) per strip, independent of the device's SM count.

    Args:
        m: Rows.
        n: Columns.
        variant: The compile-time variant.

    Returns:
        ``(n_cb, n_rb, (grid_x, grid_y), ctas)``.
    """
    n_cb = (n + TILE - 1) // TILE
    n_rb = (m + variant.tile_m - 1) // variant.tile_m
    return n_cb, n_rb, (variant.cluster, n_cb), variant.cluster * n_cb


def counters(device_index: int) -> torch.Tensor:
    """The per-device completion words ``Int32[3]``: ``[0]`` counter, ``[1]``/``[2]`` the
    round-3 u32 max accumulators (``amax[0]``, pre-rounding ``amax[1]``).

    Zero-initialised once; the grid-last CTA resets it, so the buffer is reusable
    across launches and CUDA-graph replays. Read it back to check that invariant
    (the tests do). Concurrent launches of the op on two streams of one device
    would race on it -- production launches are stream-serial and the CUDA op is
    priced the same way; stated, not hidden.
    """
    c = _counters.get(device_index)
    if c is None:
        if torch.cuda.is_current_stream_capturing():
            # A buffer allocated INSIDE a graph capture belongs to that graph's private
            # pool and can alias other graphs' memory once the graph is freed -- the
            # kernel's counter writes would then land in someone else's tensor. Never
            # keep such a buffer: hand this capture a per-call zeroed one (a memset node,
            # correct, ~1 us slower) and allocate the persistent one at the next eager
            # call. Warm the op up eagerly before capturing to avoid the slow path.
            global _warned_capture_first
            if not _warned_capture_first:
                _warned_capture_first = True
                print("[tma_bgrad_amax_ctm_s] first call on device", device_index,
                      "is inside a CUDA-graph capture: using a per-call zeroed counter for "
                      "this capture; call the op once eagerly first for the persistent one",
                      flush=True)
            return torch.zeros(3, dtype=torch.int32, device=torch.device("cuda", device_index))
        c = torch.zeros(3, dtype=torch.int32, device=torch.device("cuda", device_index))
        torch.cuda.current_stream(device_index).synchronize()   # zeroed before any launch can read it
        _counters[device_index] = c
    return c


def supported(x: torch.Tensor) -> bool:
    """The CUDA op's predicate, unchanged: contiguous 16 B-aligned BF16 ``[M, N]``, ``M % 16 == N % 16 == 0``."""
    return (
        x.dim() == 2 and x.dtype is torch.bfloat16 and x.is_contiguous()
        and x.shape[0] % _ROWS_PER_GROUP == 0 and x.shape[0] > 0
        and x.shape[1] % 16 == 0 and x.shape[1] > 0
        and x.data_ptr() % 16 == 0
    )


def _check(x: torch.Tensor, rht: torch.Tensor) -> None:
    if not (x.is_cuda and rht.is_cuda):
        raise ValueError("tma_bgrad_amax_ctm_s: all tensors must be CUDA")
    if not supported(x):
        raise ValueError(
            "tma_bgrad_amax_ctm_s: x must be a contiguous, 16 B-aligned bf16 [M, N] with "
            f"M % 16 == 0 and N % 16 == 0, got {tuple(x.shape)} {x.dtype}"
        )
    if not (rht.dtype is torch.bfloat16 and rht.is_contiguous() and tuple(rht.shape) == (16, 16)):
        raise ValueError("tma_bgrad_amax_ctm_s: rht_matrix must be a contiguous bf16 [16, 16]")


def _run(
    x: torch.Tensor, rht: torch.Tensor, group: torch.Tensor | None, variant: Variant,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = x.device.index
    m, n = x.shape
    n_cb, n_rb, _, _ = geometry(m, n, variant)
    ws_amax = torch.empty(2 * n_cb, dtype=torch.float32, device=x.device)
    dbias = torch.empty(n, dtype=torch.bfloat16, device=x.device)
    amax = torch.empty(2, dtype=torch.float32, device=x.device)
    g = group if group is not None else torch.empty(1, dtype=torch.float32, device=x.device)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(dev, variant, group is not None)(
        x, ws_amax, rht.view(torch.int16).view(-1), g, dbias, amax, counters(dev),
        m, n, n_cb, n_rb, stream,
    )
    return dbias, amax


def tma_bgrad_amax_ctm_s(
    x: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias gradient and NVFP4 amax pair from one TMA read of ``x``, in one kernel.

    Args:
        x: ``[M, N]`` BF16 contiguous, ``M % 16 == 0``, ``N % 16 == 0``, 16 B aligned.
        rht_matrix: The quantizer's ``[16, 16]`` BF16 Hadamard basis.

    Returns:
        ``(dbias, amax)``: BF16 ``[N]`` column sum of ``x`` (fp32 accumulate,
        fixed order, within ``aten.sum`` tolerance, deterministic) and
        FP32 ``[2]`` = ``(max|x|, post-RHT amax)``, both bitwise TE's.

    Raises:
        ValueError: On an unsupported input (see :func:`supported`).
    """
    _check(x, rht_matrix)
    return _run(x, rht_matrix, None, variant_for(*x.shape))


def tma_bgrad_amax_ctm_s_pergroup(
    x: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`tma_bgrad_amax_ctm_s` plus the per-16-row-group post-RHT amax (debug).

    Args:
        x: As :func:`tma_bgrad_amax_ctm_s`.
        rht_matrix: As :func:`tma_bgrad_amax_ctm_s`.

    Returns:
        ``(dbias, amax, group_amax)`` with ``group_amax`` FP32 ``[M/16]``.
    """
    _check(x, rht_matrix)
    group = torch.zeros(x.shape[0] // _ROWS_PER_GROUP, dtype=torch.float32, device=x.device)
    dbias, amax = _run(x, rht_matrix, group, variant_for(*x.shape))
    return dbias, amax, group


def tma_bgrad_amax_ctm_s_variant(
    x: torch.Tensor, rht_matrix: torch.Tensor, variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Benchmark / test entry: run one named variant regardless of the dispatch.

    Args:
        x: As :func:`tma_bgrad_amax_ctm_s`.
        rht_matrix: As :func:`tma_bgrad_amax_ctm_s`.
        variant: A key of :data:`VARIANTS`.

    Returns:
        ``(dbias, amax)``.
    """
    _check(x, rht_matrix)
    return _run(x, rht_matrix, None, VARIANTS[variant])


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define("tma_bgrad_amax_ctm_s(Tensor x, Tensor rht_matrix) -> (Tensor, Tensor)")
_lib.define(
    "tma_bgrad_amax_ctm_s_pergroup(Tensor x, Tensor rht_matrix) -> (Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "tma_bgrad_amax_ctm_s", "CUDA")
def _tma_bgrad_amax_ctm_s_impl(
    x: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tma_bgrad_amax_ctm_s(x, rht_matrix)


@torch.library.impl(_lib, "tma_bgrad_amax_ctm_s_pergroup", "CUDA")
def _tma_bgrad_amax_ctm_s_pergroup_impl(
    x: torch.Tensor, rht_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tma_bgrad_amax_ctm_s_pergroup(x, rht_matrix)
