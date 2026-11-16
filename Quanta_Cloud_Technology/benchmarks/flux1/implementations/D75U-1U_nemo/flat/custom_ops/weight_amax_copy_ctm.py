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

"""The optimizer-path weight amax on the CUTLASS DSL.

A 1:1 structural port of ``weight_amax_copy_kernel.cu`` -- the two kernels
``flux.shard_copy_partial_amax`` and ``flux.weight_amax_reduce`` dispatch to
(``weight_amax_copy.py``), same grid, same block, same per-thread work:

* **copy + partial amax**: 256 threads x 32 elements = 8192 elements per CTA,
  eight warp-contiguous ``float4`` passes (one warp instruction covers 512 B of
  master and 256 B of BF16), each element converted with ``cvt.rn.bf16x2.f32``
  -- the instruction ``__float22bfloat162_rn`` is, emitted verbatim through
  ``inline_ptx`` because the DSL's own ``Float32 -> BFloat16`` conversion is
  the ``.satfinite`` form -- and stored as one 8 B word pair. The per-CTA
  segment list (``<= MAX_SEGMENTS`` quantized weights touching the CTA's
  window, host tables from ``build_segment_tables``) drives a register max
  per segment of ``|bf16_rn(x)|`` (the BF16 value widened to f32, exact);
  warp butterfly max, one shared-memory round, and one integer ``atom.max``
  on the f32 bit pattern per live segment (``cute.arch.atomic_fmax(...,
  sign_bit=False)`` is exactly TE's / the CUDA kernel's trick). Indexing is
  ``Int64`` throughout (the shard passes ``INT32_MAX`` at
  ``shard_size = 4``). A CTA whose window is not fully inside ``n`` takes a
  scalar guarded path; at most one CTA per launch does.
* **reduce**: ``grid = (1,)``, 512 threads, ``[shard_size, n_slots] ->
  [n_slots]`` max, plain stores.

Base ``cute.arch.*`` only -- no TMA, mbarrier or ``cp.async`` (the
``cutlass.experimental`` primitives): the kernel is a streaming copy with a
register epilogue, already at the copy band in production, and an async pipeline has nothing to hide here. Public
``nvidia-cutlass-dsl`` (4.8.0 in the container). The contract is
``tests/test_weight_amax_ag.py``: BF16 output bitwise ``aten copy_``, partials
bitwise ``max|bf16_rn(x)|`` per slice, bitwise the CUDA kernel, graph-safe.
"""

from __future__ import annotations

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

THREADS = 256
ELTS_PER_THREAD = 32
BLOCK_ELTS = THREADS * ELTS_PER_THREAD  # 8192; == weight_amax_copy.BLOCK_ELTS
MAX_SEGMENTS = 4                        # == weight_amax_copy.MAX_SEGMENTS
PASSES = ELTS_PER_THREAD // 4           # 8 float4 passes per thread
WARPS = THREADS // 32
REDUCE_THREADS = 512
#: Kernel-symbol substrings for the by-name launch census.
COPY_KERNEL = "shard_copy_partial_amax_ctm_kernel"
REDUCE_KERNEL = "weight_amax_reduce_ctm_kernel"
#: name -> (min_blocks_per_mp, invariant loads). ``0`` = leave the driver /
#: ptxas default (the CUDA kernel's plain ``__launch_bounds__(256)``).
#: ``invariant`` marks the master loads ``!invariant.load`` (``ld.global.nc``):
#: it licenses LLVM to sink the eight loads to their uses, which trades loads in
#: flight for registers -- measured that form (``A0i``)
#: 1.6 % (GB300) / 7.7 % (Rubin) slower than the CUDA kernel with 57 vs 64
#: registers and identical DRAM bytes; ``A0`` keeps the loads in program order
#: ahead of the first store.
#: The third field, ``prefetch``: read the CTA's two ``seg_offset`` entries (independent
#: of the data) BEFORE the eight data loads, so their round trip overlaps the data's;
#: the dependent ``slot/lo/hi`` reads stay behind the data loads either way.
LAUNCH_VARIANTS: dict[str, tuple[int, bool, bool]] = {
    "A0": (0, False, False), "A0p": (0, False, True), "A0i": (0, True, False), "A4": (4, False, False),
}
DEFAULT_VARIANT = "A0"

_CVT_BF16X2 = "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};"
_BF16_HI_MASK = -65536  # 0xFFFF0000 as a signed Int32


def _pack_bf16x2_rn(lo, hi):
    """``cvt.rn.bf16x2.f32``: ``hi`` in the upper half, ``lo`` in the lower (``__float22bfloat162_rn``)."""
    return cute.arch.inline_ptx(
        _CVT_BF16X2, write_only_types=[cutlass.Int32], read_only_args=[hi, lo],
    )


def _abs_lo(w):
    """``|bf16|`` of the lower half of a packed word, as the exactly widened f32."""
    return cute.math.abs((w << 16).bitcast(cutlass.Float32))


def _abs_hi(w):
    return cute.math.abs((w & cutlass.Int32(_BF16_HI_MASK)).bitcast(cutlass.Float32))


def _warp_max(v):
    for off in (16, 8, 4, 2, 1):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=off))
    return v


@cute.jit
def _seg_update(acc, lo_s, hi_s, lo_l, hi_l, m, w0, w1):
    """Fold one 4-element group ``[lo_l, hi_l)`` into segment ``[lo_s, hi_s)``'s max.

    ``m`` is the group's ``max|bf16|`` (used when the group lies fully inside
    the segment: two integer compares, the interior fast path); ``w0``/``w1``
    are the packed BF16 words for the rare edge group, judged per element.
    """
    if hi_l > lo_s:
        if lo_l < hi_s:
            if lo_l >= lo_s:
                if hi_l <= hi_s:
                    acc = cute.arch.fmax(acc, m)
                else:
                    acc = _edge_update(acc, lo_s, hi_s, lo_l, w0, w1)
            else:
                acc = _edge_update(acc, lo_s, hi_s, lo_l, w0, w1)
    return acc


@cute.jit
def _edge_update(acc, lo_s, hi_s, lo_l, w0, w1):
    """The edge group: each of the four elements judged on its own."""
    acc = _seg_update_scalar(acc, lo_s, hi_s, lo_l, _abs_lo(w0))
    acc = _seg_update_scalar(acc, lo_s, hi_s, lo_l + 1, _abs_hi(w0))
    acc = _seg_update_scalar(acc, lo_s, hi_s, lo_l + 2, _abs_lo(w1))
    acc = _seg_update_scalar(acc, lo_s, hi_s, lo_l + 3, _abs_hi(w1))
    return acc


@cute.jit
def _seg_update_scalar(acc, lo_s, hi_s, li, ax):
    """Fold one element (block-local index ``li``, ``|bf16|`` value ``ax``) into a segment's max."""
    if li >= lo_s:
        if li < hi_s:
            acc = cute.arch.fmax(acc, ax)
    return acc


@cute.jit
def _segments(seg_slot, seg_lo, seg_hi, s0, nseg):
    """This CTA's segment list as 12 scalars: four ``(slot, lo, hi)`` from
    ``seg_slot/lo/hi[s0 : s0 + nseg]``; a dead slot gets an empty window
    ``[BLOCK, BLOCK)``.

    Called AFTER the data loads are issued: the dependent table reads
    (``seg_offset`` -> count -> guarded entries) would otherwise sit between
    CTA start and the first data load (identical
    instructions and bytes, more ``long_scoreboard`` stall per issue, 1.7 % /
    4-7 % slower on GB300 / Rubin; with the loads first: -4 % on GB300).
    """
    slot0 = cutlass.Int32(0)
    slot1 = cutlass.Int32(0)
    slot2 = cutlass.Int32(0)
    slot3 = cutlass.Int32(0)
    lo0 = cutlass.Int32(BLOCK_ELTS)
    lo1 = cutlass.Int32(BLOCK_ELTS)
    lo2 = cutlass.Int32(BLOCK_ELTS)
    lo3 = cutlass.Int32(BLOCK_ELTS)
    hi0 = cutlass.Int32(BLOCK_ELTS)
    hi1 = cutlass.Int32(BLOCK_ELTS)
    hi2 = cutlass.Int32(BLOCK_ELTS)
    hi3 = cutlass.Int32(BLOCK_ELTS)
    if nseg > 0:
        slot0 = seg_slot[s0]
        lo0 = seg_lo[s0]
        hi0 = seg_hi[s0]
    if nseg > 1:
        slot1 = seg_slot[s0 + 1]
        lo1 = seg_lo[s0 + 1]
        hi1 = seg_hi[s0 + 1]
    if nseg > 2:
        slot2 = seg_slot[s0 + 2]
        lo2 = seg_lo[s0 + 2]
        hi2 = seg_hi[s0 + 2]
    if nseg > 3:
        slot3 = seg_slot[s0 + 3]
        lo3 = seg_lo[s0 + 3]
        hi3 = seg_hi[s0 + 3]
    return slot0, slot1, slot2, slot3, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3


@cute.kernel
def shard_copy_partial_amax_ctm_kernel(
    src: cute.Tensor, dst: cute.Tensor, partial: cute.Tensor,
    seg_offset: cute.Tensor, seg_slot: cute.Tensor, seg_lo: cute.Tensor, seg_hi: cute.Tensor,
    n: cutlass.Int64, INVARIANT: cutlass.Constexpr[bool], PREFETCH: cutlass.Constexpr[bool],
) -> None:
    """``src`` f32 ``[n]`` -> ``dst`` bf16 ``[n]`` (``bf16_rn``), ``partial[slot] = max`` per segment."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()

    sp = cutlass.Pointer(src.iterator, dtype=cutlass.Float32)
    dw = cutlass.Pointer(dst.iterator, dtype=cutlass.Int32)    # bf16 pairs
    dh = cutlass.Pointer(dst.iterator, dtype=cutlass.Int16)    # single bf16 (tail)
    pp = cutlass.Pointer(partial.iterator, dtype=cutlass.Float32)
    stage = cute.arch.alloc_smem(cutlass.Float32, MAX_SEGMENTS * WARPS, 16)

    base = bidx.to(cutlass.Int64) * BLOCK_ELTS
    full = base + BLOCK_ELTS <= n

    acc = [cutlass.Float32(0.0) for _ in range(MAX_SEGMENTS)]
    nseg = cutlass.Int32(0)
    s0 = cutlass.Int32(0)
    slot = [cutlass.Int32(0) for _ in range(MAX_SEGMENTS)]
    lo = [cutlass.Int32(BLOCK_ELTS) for _ in range(MAX_SEGMENTS)]
    hi = [cutlass.Int32(BLOCK_ELTS) for _ in range(MAX_SEGMENTS)]
    if cutlass.const_expr(PREFETCH):
        # The two seg_offset reads do not depend on the data: issue them ahead of
        # the data loads so their round trip overlaps the data's.
        s0 = seg_offset[bidx]
        nseg = seg_offset[bidx + 1] - s0
    if full:
        # Warp-contiguous: in pass v thread t holds src[base + v*1024 + 4t .. +4).
        # All eight data loads are issued first -- before the dependent table
        # reads and before any conversion (the CUDA kernel's order).
        f = []
        for v in cutlass.range_constexpr(PASSES):
            off = base + (v * (THREADS * 4) + tidx * 4)
            f.append((sp + off).load(16, count=4, is_invariant=INVARIANT))
        if cutlass.const_expr(not PREFETCH):
            s0 = seg_offset[bidx]
            nseg = seg_offset[bidx + 1] - s0
        s_0, s_1, s_2, s_3, l_0, l_1, l_2, l_3, h_0, h_1, h_2, h_3 = _segments(
            seg_slot, seg_lo, seg_hi, s0, nseg)
        slot = [s_0, s_1, s_2, s_3]
        lo = [l_0, l_1, l_2, l_3]
        hi = [h_0, h_1, h_2, h_3]
        for v in cutlass.range_constexpr(PASSES):
            lo_l = v * (THREADS * 4) + tidx * 4
            off = base + lo_l
            w0 = _pack_bf16x2_rn(f[v][0], f[v][1])
            w1 = _pack_bf16x2_rn(f[v][2], f[v][3])
            (dw + off // 2).store(
                cutlass.Vector.from_elements((w0, w1), cutlass.Int32), alignment=8)
            m = cute.arch.fmax(cute.arch.fmax(_abs_lo(w0), _abs_hi(w0)),
                               cute.arch.fmax(_abs_lo(w1), _abs_hi(w1)))
            acc = [_seg_update(acc[s], lo[s], hi[s], lo_l, lo_l + 4, m, w0, w1)
                   for s in range(MAX_SEGMENTS)]
    else:
        # Ragged tail CTA (at most one per launch): scalar, guarded per element.
        if cutlass.const_expr(not PREFETCH):
            s0 = seg_offset[bidx]
            nseg = seg_offset[bidx + 1] - s0
        s_0, s_1, s_2, s_3, l_0, l_1, l_2, l_3, h_0, h_1, h_2, h_3 = _segments(
            seg_slot, seg_lo, seg_hi, s0, nseg)
        slot = [s_0, s_1, s_2, s_3]
        lo = [l_0, l_1, l_2, l_3]
        hi = [h_0, h_1, h_2, h_3]
        for v in cutlass.range_constexpr(PASSES):
            for k in cutlass.range_constexpr(4):
                li = v * (THREADS * 4) + tidx * 4 + k
                g = base + li
                if g < n:
                    x = (sp + g).load()
                    w = _pack_bf16x2_rn(x, x)
                    (dh + g).store((w & 0xFFFF).to(cutlass.Int16))
                    ax = _abs_lo(w)
                    acc = [_seg_update_scalar(acc[s], lo[s], hi[s], li, ax)
                           for s in range(MAX_SEGMENTS)]

    # Block reduce per segment: warp butterfly, one smem round, one atomic.
    for s in cutlass.range_constexpr(MAX_SEGMENTS):
        wm = _warp_max(acc[s])
        if lane == 0:
            (stage + s * WARPS + warp).store(wm)
    cute.arch.sync_threads()
    if tidx == 0:
        for s in cutlass.range_constexpr(MAX_SEGMENTS):
            if s < nseg:
                m = (stage + s * WARPS).load()
                for w in cutlass.range_constexpr(1, WARPS):
                    m = cute.arch.fmax(m, (stage + s * WARPS + w).load())
                cute.arch.atomic_fmax(pp + slot[s], m, sign_bit=False)


@cute.kernel
def weight_amax_reduce_ctm_kernel(
    gathered: cute.Tensor, out: cute.Tensor, shard_size: cutlass.Int32, n_slots: cutlass.Int32,
) -> None:
    """``out[c] = max_r gathered[r * n_slots + c]``; one CTA by launch geometry."""
    tidx, _, _ = cute.arch.thread_idx()
    gp = cutlass.Pointer(gathered.iterator, dtype=cutlass.Float32)
    op = cutlass.Pointer(out.iterator, dtype=cutlass.Float32)
    for c in cutlass.range(tidx, n_slots, REDUCE_THREADS):
        m = cutlass.Float32(0.0)
        for r in cutlass.range(shard_size):
            m = cute.arch.fmax(m, (gp + (r * n_slots + c)).load())
        (op + c).store(m)


@cute.jit
def shard_copy_partial_amax_ctm_host(
    src: cute.Tensor, dst: cute.Tensor, partial: cute.Tensor,
    seg_offset: cute.Tensor, seg_slot: cute.Tensor, seg_lo: cute.Tensor, seg_hi: cute.Tensor,
    n: cutlass.Int64, grid_x: cutlass.Int32, stream: cuda.CUstream,
    MIN_CTAS: cutlass.Constexpr[int], INVARIANT: cutlass.Constexpr[bool],
    PREFETCH: cutlass.Constexpr[bool],
) -> None:
    kern = shard_copy_partial_amax_ctm_kernel(
        src, dst, partial, seg_offset, seg_slot, seg_lo, seg_hi, n, INVARIANT, PREFETCH)
    if cutlass.const_expr(MIN_CTAS > 0):
        kern.launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1),
                    min_blocks_per_mp=MIN_CTAS, stream=stream)
    else:
        kern.launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def weight_amax_reduce_ctm_host(
    gathered: cute.Tensor, out: cute.Tensor, shard_size: cutlass.Int32,
    n_slots: cutlass.Int32, stream: cuda.CUstream,
) -> None:
    weight_amax_reduce_ctm_kernel(gathered, out, shard_size, n_slots).launch(
        grid=(1, 1, 1), block=(REDUCE_THREADS, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_compiled: dict[tuple, Callable] = {}
_dummy: dict[int, torch.Tensor] = {}


def _compile_copy(device_index: int, variant: str) -> Callable:
    """Compile once per (device, launch variant); ``n`` and every table length stay symbolic."""
    key = ("copy", device_index, variant)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            f32, bf, i32 = cutlass.Float32, cutlass.BFloat16, cutlass.Int32
            sym = cute.sym_int64
            f_src = make_fake_compact_tensor(f32, (sym(1, "n"),), assumed_align=16)
            f_dst = make_fake_compact_tensor(bf, (sym(1, "nd"),), assumed_align=16)
            f_partial = make_fake_compact_tensor(f32, (sym(1, "p"),), assumed_align=4)
            f_tabs = [make_fake_compact_tensor(i32, (sym(1, t),), assumed_align=4)
                      for t in ("so", "ss", "sl", "sh")]
            min_ctas, invariant, prefetch = LAUNCH_VARIANTS[variant]
            _compiled[key] = cute.compile(
                shard_copy_partial_amax_ctm_host, f_src, f_dst, f_partial, *f_tabs,
                cutlass.Int64(0), cutlass.Int32(0), make_fake_stream(),
                min_ctas, invariant, prefetch, options="--enable-tvm-ffi",
            )
    return _compiled[key]


def _compile_reduce(device_index: int) -> Callable:
    key = ("reduce", device_index)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            f32 = cutlass.Float32
            sym = cute.sym_int64
            f_g = make_fake_compact_tensor(f32, (sym(1, "g"),), assumed_align=4)
            f_o = make_fake_compact_tensor(f32, (sym(1, "o"),), assumed_align=4)
            _compiled[key] = cute.compile(
                weight_amax_reduce_ctm_host, f_g, f_o, cutlass.Int32(0), cutlass.Int32(0),
                make_fake_stream(), options="--enable-tvm-ffi",
            )
    return _compiled[key]


def _nonempty(t: torch.Tensor) -> torch.Tensor:
    """A zero-length table is never read (``nseg == 0`` everywhere); hand the runtime a real pointer."""
    if t.numel():
        return t
    d = _dummy.get(t.device.index)
    if d is None:
        d = torch.zeros(1, dtype=torch.int32, device=t.device)
        _dummy[t.device.index] = d
    return d


def n_blocks(total: int) -> int:
    """CTAs the copy launches for a ``total``-element shard: the CUDA host's rule."""
    return (total + BLOCK_ELTS - 1) // BLOCK_ELTS


def shard_copy_partial_amax_ctm(
    src: torch.Tensor, dst: torch.Tensor, partial: torch.Tensor,
    seg_offset: torch.Tensor, seg_slot: torch.Tensor, seg_lo: torch.Tensor, seg_hi: torch.Tensor,
    variant: str = DEFAULT_VARIANT,
) -> None:
    """Fused master f32 -> BF16 shard copy + per-weight partial amax, DSL kernel.

    Args:
        src: f32 flat master shard, contiguous CUDA.
        dst: BF16, same numel -- receives ``bf16_rn(src)``, bitwise ``copy_``.
        partial: ``[n_slots]`` f32, zeroed by the caller (integer atomic max on
            the bit pattern: sound against zeros and non-negative values only).
        seg_offset: From ``weight_amax_copy.build_segment_tables``, int32.
        seg_slot: As ``seg_offset``.
        seg_lo: As ``seg_offset``.
        seg_hi: As ``seg_offset``.
        variant: Key of :data:`LAUNCH_VARIANTS`.

    Raises:
        ValueError: On a dtype/layout/shape mismatch (the CUDA op's checks).
    """
    if not (src.is_cuda and src.dtype is torch.float32 and src.is_contiguous()):
        raise ValueError("src must be contiguous f32 CUDA")
    if not (dst.is_cuda and dst.dtype is torch.bfloat16 and dst.is_contiguous()
            and dst.numel() == src.numel()):
        raise ValueError("dst must be contiguous bf16 with src's numel")
    if not (partial.dtype is torch.float32 and partial.is_contiguous() and partial.numel() > 0):
        raise ValueError("partial must be contiguous f32")
    n = src.numel()
    nb = n_blocks(n)
    if seg_offset.numel() != nb + 1:
        raise ValueError(f"seg_offset must have n_blocks+1 = {nb + 1} entries, got {seg_offset.numel()}")
    for t in (seg_offset, seg_slot, seg_lo, seg_hi):
        if t.dtype is not torch.int32 or not t.is_contiguous():
            raise ValueError("segment tables must be contiguous int32")
    if not (seg_slot.numel() == seg_lo.numel() == seg_hi.numel()):
        raise ValueError("segment arrays must be parallel")
    if nb > 2**31 - 1:
        raise ValueError("grid overflow")
    if nb == 0:
        return
    dev = src.device.index
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile_copy(dev, variant)(
        src, dst, partial, seg_offset, _nonempty(seg_slot), _nonempty(seg_lo), _nonempty(seg_hi),
        n, nb, stream,
    )


def weight_amax_reduce_ctm(gathered: torch.Tensor, out: torch.Tensor, shard_size: int) -> None:
    """``[shard_size, n_slots] -> [n_slots]`` max, one CTA, DSL kernel.

    Args:
        gathered: f32 CUDA, ``shard_size * n_slots`` elements, rank-major.
        out: ``[n_slots]`` f32, fully overwritten.
        shard_size: Ranks in the shard group.

    Raises:
        ValueError: On a dtype/layout/shape mismatch (the CUDA op's checks).
    """
    if not (gathered.is_cuda and gathered.dtype is torch.float32 and gathered.is_contiguous()):
        raise ValueError("gathered must be contiguous f32 CUDA")
    if not (out.dtype is torch.float32 and out.is_contiguous()):
        raise ValueError("out must be contiguous f32")
    if gathered.numel() != shard_size * out.numel():
        raise ValueError(f"gathered numel {gathered.numel()} != shard_size {shard_size} x n_slots {out.numel()}")
    dev = gathered.device.index
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile_reduce(dev)(gathered, out, int(shard_size), out.numel(), stream)
