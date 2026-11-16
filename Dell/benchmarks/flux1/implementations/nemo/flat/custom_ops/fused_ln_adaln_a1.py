# SPDX-License-Identifier: Apache-2.0
"""``fused_ln_adaln`` / ``_bwd`` / ``_bwd_add`` on the CUTLASS DSL.

Three kernel families, registered under NEW names (``flux.fused_ln_adaln_a1*``)
and **unwired**:

* ``ln_adaln_fwd_ring_kernel`` -- the forward arm ``r2`` **verbatim**
  (warp-per-row ``cp.async.bulk`` ring, 8 warps, depth 2, row held in fp32
  registers). Measured -0.58 ms/iter at B=32 on GB300; carried
  as the large-M forward and re-measured here on the 4.8.0.dev0 container.
* ``ln_adaln_fwd_ring2_kernel`` -- NEW. The same ring, but every warp processes
  **two independent rows per iteration**, three passes from shared memory with
  ``sync_warp`` fences between them, so the two reduction chains interleave and
  one row's ``shfl.sync`` latency is covered by the other row's loads and FMAs.
  No CTA barrier. Attacks the mechanism ncu left on the ring forward: 44 % of scheduler cycles
  with no eligible warp at two warps per scheduler.
* ``ln_adaln_fwd_plain_kernel`` / ``ln_adaln_bwd_plain_kernel`` -- NEW. Triton's
  geometry (one row per CTA, plain 128-bit loads, cross-warp sums through a few
  floats of shared memory and ``bar.sync``) with **exact D = 3072** -- the Triton
  kernels pad to ``BLOCK_D = 4096`` and mask a quarter of their lanes -- and with
  the threads per row as a knob (128 / 192 / 384 -> 3 / 2 / 1 vectors per
  thread). The backward reduces ``c1`` and ``c2`` through ONE barrier: both need
  only the forward's ``mean``/``rstd``, so the second pass Triton makes is not
  needed. Fewer registers per row-worker is the occupancy lever the ncu profiles
  point at (Triton bwd and the earlier DSL bwd both sit at ~4 CTAs/SM, 0.4 eligible warps).

* The mb-4 co-residency work adds a third dispatch tier
  ``fwd_tiny`` at ``M <= SMALL_M_ROWS`` and three candidate arms for it: ``pn128``
  (the ``pm128`` body launched with NO occupancy / carveout attribute), ``pc128``
  (``pn128`` with plain instead of ``.nc`` modulation loads) and ``pw32``
  (``ln_adaln_fwd_warp_kernel``: one warp per row, no shared memory, no barrier).
  ``pn128``/``pc128`` are bitwise ``pm128``; ``pw32`` is bitwise ``r2``. An opt-in
  instrumented twin of ``pm128`` (``DBG_FWD_PLAIN``) stamps %globaltimer / %clock64 /
  %smid per CTA for the production co-residency measurement.
* Three more ``fwd_tiny`` candidates: ``pw32s`` (``pw32``
  with the modulation prefetched before the statistics), ``pw128x``
  (``ln_adaln_fwd_warpx_kernel``: four INDEPENDENT warps per 128-thread CTA, one row each,
  no smem, no barrier) and ``pk128`` (the ``pm128`` body launched with
  ``min_blocks_per_mp=9``, i.e. a ptxas register cap in Triton's occupancy class). The warp
  family carries the same per-CTA stamps as a ``DBG`` constexpr (``DBG_FWD_WARP``).
  ``pw32s`` / ``pw128x`` are bitwise ``r2``; ``pk128`` is bitwise ``pm128``.
* The footprint-lever ``fwd_tiny`` candidates of the warp family
  (the warp kernel's ``PREFETCH`` bool became ``MODE``): ``pw32c`` (``x`` held as the loaded bf16
  vectors and re-converted at each use, ``scale`` prefetched before the statistics, ``shift``
  loaded right after ``mean`` -- the "<= ~100 registers" arm), ``pw32b`` (``x`` bf16-held,
  both modulation vectors prefetched) and ``pw32d`` (``x`` bf16-held, nothing prefetched). bf16 ->
  fp32 is exact, so all three are bitwise ``r2``.
* The warp kernel's ``RPC`` (rows per CTA) and the SSB-tier candidates
  ``pw32c2r2`` / ``pw32b2r2`` (the ``pw32c2`` / ``pw32b2`` bodies with two rows of one batch per
  32-thread CTA, grid ``B * ceil(S / 2)``: 1024 CTAs at 2048 rows, one wave on the SMs a resident
  collective leaves), plus the unwired dispatch split ``fwd_tiny`` / ``fwd_tiny2`` at
  ``TINY_SPLIT_ROWS``. Bitwise ``r2`` (per-row arithmetic unchanged).

Numerics: fp32 statistics, plain ``sum(x)`` mean,
exact two-pass centred variance, ``fma((x-mean)*rstd, 1+scale, shift)``;
backward ``((gn - c1) - n*c2) * rstd (+res)``; ``grad_sum = bf16(f32(go)+f32(ge))``
bitwise. No shifted variance anywhere (an arm using it was rejected).

Public ``nvidia-cutlass-dsl`` only: ``cute.arch.*``, ``cutlass.utils.SmemAllocator``,
``cutlass.Pointer``, ``cute.math.*``, ``inline_ptx`` for the bulk copy. Nothing
from ``cutlass.experimental`` is needed (see API-DELTAS.md). ``CUTE_DSL_ARCH``
must stay unset.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
import torch.library
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T

VEC = 8                      # bf16 elements per 128-bit access
D = 3072                     # Flux.1 hidden size; the only width the kernels are built for
VPL = D // (32 * VEC)        # 12 vectors per lane per row (warp-per-row kernels)
ROW_BYTES = D * 2            # one bf16 row = 6144 B = one bulk copy
EPS = 1e-6
_INT32_MAX = 2**31 - 1
_F32 = cutlass.Float32
_BF16 = cutlass.BFloat16
_I32 = cutlass.Int32
_I64 = cutlass.Int64

#: ``preferred_smem_carveout`` for the ring kernels (a request the driver may
#: round; read back from ncu). The plain kernels ask for a SMALL
#: carveout so L1 keeps the modulation vectors (Triton's realised config is
#: 16-32 KB).
SMEM_CARVEOUT_PCT = 100
PLAIN_CARVEOUT_PCT = 8
MOD_BATCHES = 2

#: Kernel symbols -- the symbols launch checks match by name.
KERNEL_FWD_RING = "ln_adaln_fwd_ring_kernel"
KERNEL_FWD_RING2 = "ln_adaln_fwd_ring2_kernel"
KERNEL_FWD_PLAIN = "ln_adaln_fwd_plain_kernel"
KERNEL_FWD_PLAIN_DBG = "ln_adaln_fwd_plain_dbg_kernel"
KERNEL_FWD_WARP = "ln_adaln_fwd_warp_kernel"
KERNEL_FWD_WARPX = "ln_adaln_fwd_warpx_kernel"
KERNEL_BWD_PLAIN = "ln_adaln_bwd_plain_kernel"
KERNEL_TRITON_FWD = "_fused_ln_adaln_fwd_kernel"
KERNEL_TRITON_BWD = "_fused_ln_adaln_bwd_kernel"

#: Arms, fixed before measurement.
#: fwd: name -> (family, *knobs); ring: (warps, depth); ring2: (warps, depth);
#: plain: (tpr, min_blocks, carveout, mode) -- ``min_blocks``/``carveout`` None =
#: launch with NO occupancy / carveout attribute (driver defaults, as Triton and
#: A2's ``rb_alt`` launch); warp: no knobs.
#: The mb-4 work adds the three ``fwd_tiny`` candidates ``pn128`` / ``pc128`` /
#: ``pw32``.
FWD_ARMS: dict[str, tuple] = {
    "r2": ("ring", 8, 2),
    "p84": ("ring2", 8, 4),
    "p64": ("ring2", 6, 4),
    "p44": ("ring2", 4, 4),
    "pl128": ("plain", 128, 1, PLAIN_CARVEOUT_PCT, 0),
    "pl384": ("plain", 384, 1, PLAIN_CARVEOUT_PCT, 0),
    "pm128": ("plain", 128, 1, PLAIN_CARVEOUT_PCT, 1),      # modulation prefetched
    "pn128": ("plain", 128, None, None, 1),                 # pm128 body, no launch attributes
    "pc128": ("plain", 128, None, None, 2),                 # pn128 with plain (non-.nc) modulation loads
    "pw32": ("warp", 0),                                    # one warp per row, no smem, no barrier (x held fp32)
    "pw32s": ("warp", 1),                                   # pw32 with the modulation prefetched before the statistics
    "pw32d": ("warp", 2),                                   # x held as bf16, nothing prefetched (low-register control)
    "pw32c": ("warp", 3),                                   # x held as bf16, scale prefetched, shift loaded after mean (THE candidate)
    "pw32b": ("warp", 4),                                   # x held as bf16, scale AND shift prefetched (placement control)
    "pw32d2": ("warp", 5),                                  # x held as packed words, OPAQUE per-pass conversion, nothing prefetched
    "pw32c2": ("warp", 6),                                  # opaque conversion, scale prefetched, shift after mean (THE candidate)
    "pw32b2": ("warp", 7),                                  # opaque conversion, scale AND shift prefetched
    "pw32c2r2": ("warp", 6, 2),                             # pw32c2 body, TWO rows of one batch per CTA (grid B*ceil(S/2)) -- the SSB tier candidate
    "pw32b2r2": ("warp", 7, 2),                             # pw32b2 body, two rows per CTA (control for the shift trip)
    "pw32c2r2p": ("warp", 6, 2, 12),                        # pw32c2r2 with min_blocks_per_mp=12 (ptxas cap 168 regs = 3 warps per SMSP)
    "pw128x": ("warpx", 4),                                # four INDEPENDENT warps per 128-thread CTA, one row each
    "pk128": ("plain", 128, 9, PLAIN_CARVEOUT_PCT, 1),      # pm128 body, min_blocks_per_mp=9 -> ptxas register cap <= 56 (Triton's class)
}
#: bwd / bwd_add: name -> (tpr, min_blocks, carveout, mode).
BWD_ARMS: dict[str, tuple] = {
    "pl128": (128, 1, PLAIN_CARVEOUT_PCT, 0),
    "pl128c": (128, 5, PLAIN_CARVEOUT_PCT, 0),
    "pl192": (192, 1, PLAIN_CARVEOUT_PCT, 0),
    "pl384": (384, 1, PLAIN_CARVEOUT_PCT, 0),
    "pr128": (128, 1, PLAIN_CARVEOUT_PCT, 1),               # residual before the barrier
    "ph128": (128, 1, PLAIN_CARVEOUT_PCT, 2),               # + 1+scale held
    "pL128": (128, 1, PLAIN_CARVEOUT_PCT, 3),               # stateless second pass
    "pr192": (192, 1, PLAIN_CARVEOUT_PCT, 1),
    "pL192": (192, 1, PLAIN_CARVEOUT_PCT, 3),
}
#: The dispatcher's selection. Initially r2 / pl128 / pl128 / pl128;
#: measurement moved the selection to the arms that won every regime they serve: the stateless second-pass
#: backward ``pL128`` (bwd) / ``pL192`` (bwd_add) and the modulation-prefetching
#: plain forward ``pm128`` below the ring threshold.
SELECT = {"fwd_large": "r2", "fwd_small": "pm128", "fwd_tiny": "pw32c2", "bwd": "pL128", "bwd_add": "pL192"}
#: The tiny tier is split at ``TINY_SPLIT_ROWS`` (different
#: problem sizes may need different implementations): ``M <= TINY_SPLIT_ROWS`` (the mb-4 DSB shape, 1024
#: rows) takes ``fwd_tiny``; ``TINY_SPLIT_ROWS < M <= SMALL_M_ROWS`` (the SSB shape, 2048 rows) takes
#: ``fwd_tiny2``.
#:
#: ``fwd_tiny`` = ``pw32c2`` and ``fwd_tiny2`` = ``pw32c2r2p``, replacing the Triton
#: forward that fixC routed here. Both are bitwise ``r2``. The evidence, at the shipped
#: ``64x04x04`` beside the real ``AllGather_STMC``:
#:
#: * per-launch: the first arm ever measured there that is neither PLACED on the
#:   collective's SM (0 CTAs in 100 % of resident launches, as Triton), nor bimodally hurt (>= 15 us
#:   fraction 0.000 at both shapes), nor uniformly taxed (no tail; ``pw32`` by contrast ran +4.2 us on
#:   every launch);
#: * step, on two crossed distinct node sets -- **+0.054** (ABBA) and **+0.041** ms/step (BAAB),
#:   agreeing to 0.013, with an independent 16-node
#:   measurement reading +0.044. A resolved, uniform ~ +0.045 ms/step (0.046 %) COST, inside the
#:   accepted non-regression band [-0.20, +0.10]. **This is not a win; it is an accepted cost.**
#:
SELECT["fwd_tiny2"] = "pw32c2r2p"
TINY_SPLIT_ROWS = 1024
#: The fitted threshold: the ring takes the forward once ``x + out``
#: (4*D bytes per row) reaches this fraction of L2 -- 6618 rows on GB300, 6452 on Rubin.
RING_L2_FRACTION = 0.6
#: The mb-4 shapes (M = B*S rows) take ``SELECT["fwd_tiny"]`` -- a
#: size expression. The plain forward ran a FIXED ~+18 us per launch
#: beside the 256-rank ``AllGather_STMC`` kernel there.
SMALL_M_ROWS = 2048

#: Instrumentation: when True, the plain forward launches
#: ``ln_adaln_fwd_plain_dbg_kernel`` (the ``pm128`` body plus per-CTA %globaltimer /
#: %clock64 / %smid stamps into a side buffer) and keeps every buffer allocated during
#: a CUDA-graph capture so :func:`dump_dbg` can write them after the profiled replay.
DBG_FWD_PLAIN = False
#: The same per-CTA stamps for the warp family (``pw32`` / ``pw32s`` / ``pw128x``),
#: compiled in as a ``DBG`` constexpr of the same kernel (the kernel NAME does not change).
DBG_FWD_WARP = False
DBG_WORDS = 4  # per row: t0 (globaltimer ns), t1 (globaltimer ns), clock64 delta, smid
_DBG_BUFS: list[tuple[int, int, torch.Tensor]] = []

_GT = "mov.u64 $0, %globaltimer;"
_CLK = "mov.u64 $0, %clock64;"
_SMID = "mov.u32 $0, %smid;"


def _asm(res, code, cons):
    """A zero-input, one-output, side-effecting inline asm (never CSE'd or dropped)."""
    return llvm.inline_asm(res, [], code, cons, has_side_effects=True, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT)


def _globaltimer():
    return _I64(_asm(T.i64(), _GT, "=l"))


def _clock64():
    return _I64(_asm(T.i64(), _CLK, "=l"))


def _smid():
    return _I32(_asm(T.i32(), _SMID, "=r"))

_G2S = ("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[{$r0}], [{$r1}], {$r2}, [{$r3}];")


def _words_to_f32x8(w, sh, mk):
    """Traced helper (plain Python): the 8 bf16 of 4 packed-bf16x2 Int32 words as an fp32x8 Vector.

    Element ``2j`` is word ``j``'s low half (``w << 16``), ``2j+1`` its high half (``w & 0xffff0000``)
    -- exactly ``bf16 -> fp32``. ``sh`` (= 16) and ``mk`` (= 0xffff0000) are RUNTIME kernel parameters,
    one pair per pass, so neither LLVM nor ptxas can prove two passes' conversions equal and keep a
    shared fp32 copy live (the bf16-held modes 2-4 otherwise compiled to their fp32 twins)."""
    el = []
    for j in range(4):
        el.append((w[j] << sh).bitcast(_F32))
        el.append((w[j] & mk).bitcast(_F32))
    return cutlass.Vector.from_elements(tuple(el), _F32)


def _warp_sum(v):
    """Full-warp fp32 sum, fixed 5-step butterfly (``shfl.sync.bfly``)."""
    for offset in (16, 8, 4, 2, 1):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=offset)
    return v


def _warp_sum2(a, b):
    """Two independent full-warp sums, the butterflies interleaved step by step."""
    for offset in (16, 8, 4, 2, 1):
        a = a + cute.arch.shuffle_sync_bfly(a, offset=offset)
        b = b + cute.arch.shuffle_sync_bfly(b, offset=offset)
    return a, b


def _row_copy(dst, src, bar):
    """Lane-0 helper: one 6 KB row, global -> the warp's smem slot, tracked by ``bar``."""
    cute.arch.inline_ptx(_G2S, read_only_args=[dst.toint(), src.toint(), _I32(ROW_BYTES), bar.toint()])


# ---------------------------------------------------------------------------
# Forward, ring (``r2``)
# ---------------------------------------------------------------------------

@cute.kernel
def ln_adaln_fwd_ring_kernel(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor,
    n_rows: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32, chunk: _I32,
    WARPS: cutlass.Constexpr[int], DEPTH: cutlass.Constexpr[int], HOLD: cutlass.Constexpr[bool],
    MODSMEM: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = tidx // 32
    smem = utils.SmemAllocator()
    ring = smem.allocate_array(_BF16, WARPS * DEPTH * D, byte_alignment=128)
    bars = smem.allocate_array(_I64, WARPS * DEPTH, byte_alignment=8)
    my_ring = ring + warp * (DEPTH * D)
    my_bars = bars + warp * DEPTH
    ring_v = cutlass.Pointer(my_ring, dtype=_BF16)
    scp = cutlass.Pointer(sc.iterator, dtype=_BF16)
    shp = cutlass.Pointer(sh.iterator, dtype=_BF16)
    op = cutlass.Pointer(out.iterator, dtype=_BF16)
    mp = cutlass.Pointer(mean.iterator, dtype=_F32)
    rp = cutlass.Pointer(rstd.iterator, dtype=_F32)

    row_begin = bidx * chunk
    row_end = cutlass.min(row_begin + chunk, n_rows)
    rem = row_end - row_begin - warp            # >= -(WARPS-1) when this CTA has rows
    n_my = cutlass.max(rem + (WARPS - 1), 0) // WARPS
    b0 = row_begin // S

    if lane == 0:
        for s in cutlass.range_constexpr(DEPTH):
            cute.arch.mbarrier_init(my_bars + s, 1)
    if cutlass.const_expr(MODSMEM):
        # CTA-wide: fp32 1+scale and shift of batches b0, b0+1 -> smem [bb][g|h][D].
        # (Inlined: a dynamic cutlass.range inside a plain helper does not lower.)
        smod = smem.allocate_array(_F32, MOD_BATCHES * 2 * D, byte_alignment=128)
        smod_v = cutlass.Pointer(smod, dtype=_F32)
        b_last = n_rows // S - 1
        for i in cutlass.range(tidx, MOD_BATCHES * (D // VEC), WARPS * 32):
            bb = i // (D // VEC)
            moff = (i - bb * (D // VEC)) * VEC
            bm = cutlass.min(b0 + bb, b_last)
            gv = (scp + (bm * sc_stride_b + moff)).load(16, count=VEC, is_invariant=True).to(_F32) + 1.0
            hv = (shp + (bm * sc_stride_b + moff)).load(16, count=VEC, is_invariant=True).to(_F32)
            (smod_v + (bb * 2) * D + moff).store(gv, alignment=16)
            (smod_v + (bb * 2 + 1) * D + moff).store(hv, alignment=16)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # prologue: the first DEPTH rows of this warp
    if lane == 0:
        for p in cutlass.range_constexpr(DEPTH):
            if p < n_my:
                row = row_begin + warp + p * WARPS
                b = row // S
                s = row - b * S
                cute.arch.mbarrier_arrive_and_expect_tx(my_bars + p, ROW_BYTES)
                _row_copy(my_ring + p * D, x.iterator + (b * x_stride_b + s * D), my_bars + p)

    for j in cutlass.range(0, n_my, 1):
        slot = j % DEPTH
        phase = (j // DEPTH) % 2
        row = row_begin + warp + j * WARPS
        b = row // S
        if cutlass.const_expr(MODSMEM):
            grow = smod_v + ((b - b0) * 2) * D      # fp32 1+scale of this row's batch
            hrow = grow + D                          # fp32 shift
        else:
            grow = scp + b * sc_stride_b
            hrow = shp + b * sc_stride_b
        orow = op + row * D
        sbase = ring_v + slot * D
        cute.arch.mbarrier_wait(my_bars + slot, phase)
        if cutlass.const_expr(HOLD):
            v = []
            for k in cutlass.range_constexpr(VPL):
                v.append((sbase + (k * 32 + lane) * VEC).load(16, count=VEC).to(_F32))
            s1 = _F32(0.0)
            for k in cutlass.range_constexpr(VPL):
                s1 = s1 + v[k].reduce("add")
            mean_v = _warp_sum(s1) / _F32(D)
            s2 = _F32(0.0)
            for k in cutlass.range_constexpr(VPL):
                d = v[k] - mean_v
                s2 = s2 + (d * d).reduce("add")
            var_v = _warp_sum(s2) / _F32(D)
            rstd_v = _F32(1.0) / cute.math.sqrt(var_v + _F32(EPS))
            for k in cutlass.range_constexpr(VPL):
                off = (k * 32 + lane) * VEC
                xc = v[k] - mean_v
                if cutlass.const_expr(MODSMEM):
                    g = (grow + off).load(16, count=VEC)
                    hh = (hrow + off).load(16, count=VEC)
                else:
                    g = (grow + off).load(16, count=VEC, is_invariant=True).to(_F32) + 1.0
                    hh = (hrow + off).load(16, count=VEC, is_invariant=True).to(_F32)
                res = cute.math.fma(xc * rstd_v, g, hh)
                (orow + off).store(res.to(_BF16), alignment=16)
        else:
            s1 = _F32(0.0)
            for k in cutlass.range_constexpr(VPL):
                s1 = s1 + (sbase + (k * 32 + lane) * VEC).load(16, count=VEC).to(_F32).reduce("add")
            mean_v = _warp_sum(s1) / _F32(D)
            s2 = _F32(0.0)
            for k in cutlass.range_constexpr(VPL):
                d = (sbase + (k * 32 + lane) * VEC).load(16, count=VEC).to(_F32) - mean_v
                s2 = s2 + (d * d).reduce("add")
            var_v = _warp_sum(s2) / _F32(D)
            rstd_v = _F32(1.0) / cute.math.sqrt(var_v + _F32(EPS))
            for k in cutlass.range_constexpr(VPL):
                off = (k * 32 + lane) * VEC
                xc = (sbase + off).load(16, count=VEC).to(_F32) - mean_v
                if cutlass.const_expr(MODSMEM):
                    g = (grow + off).load(16, count=VEC)
                    hh = (hrow + off).load(16, count=VEC)
                else:
                    g = (grow + off).load(16, count=VEC, is_invariant=True).to(_F32) + 1.0
                    hh = (hrow + off).load(16, count=VEC, is_invariant=True).to(_F32)
                res = cute.math.fma(xc * rstd_v, g, hh)
                (orow + off).store(res.to(_BF16), alignment=16)
        if lane == 0:
            (mp + row).store(mean_v)
            (rp + row).store(rstd_v)
        # every lane is done reading the slot before lane 0 refills it
        cute.arch.sync_warp()
        if lane == 0:
            jn = j + DEPTH
            if jn < n_my:
                rown = row_begin + warp + jn * WARPS
                bn = rown // S
                sn = rown - bn * S
                cute.arch.mbarrier_arrive_and_expect_tx(my_bars + slot, ROW_BYTES)
                _row_copy(my_ring + slot * D, x.iterator + (bn * x_stride_b + sn * D), my_bars + slot)


@cute.jit
def ln_adaln_fwd_ring_host(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor,
    n_rows: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32, chunk: _I32, grid_x: _I32,
    stream: cuda.CUstream,
    WARPS: cutlass.Constexpr[int], DEPTH: cutlass.Constexpr[int], HOLD: cutlass.Constexpr[bool],
    MODSMEM: cutlass.Constexpr[bool], MINB: cutlass.Constexpr[int],
) -> None:
    ln_adaln_fwd_ring_kernel(
        x, sc, sh, out, mean, rstd, n_rows, S, x_stride_b, sc_stride_b, chunk, WARPS, DEPTH, HOLD, MODSMEM,
    ).launch(grid=(grid_x, 1, 1), block=(WARPS * 32, 1, 1), stream=stream,
             min_blocks_per_mp=MINB, preferred_smem_carveout=SMEM_CARVEOUT_PCT)


# ---------------------------------------------------------------------------
# Forward, ring2: two independent rows per warp per iteration, three passes from smem
# ---------------------------------------------------------------------------

@cute.kernel
def ln_adaln_fwd_ring2_kernel(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor,
    n_rows: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32, chunk: _I32,
    WARPS: cutlass.Constexpr[int], DEPTH: cutlass.Constexpr[int],
) -> None:
    """Warp ``w`` owns rows ``row_begin + w + j*WARPS``; iteration ``j`` (even) handles
    rows ``j`` and ``j+1`` together. ``DEPTH`` is even and >= 4 so the next pair is in
    flight while this pair computes. A ragged last row is handled without dynamic
    control flow: ``jb = min(j+1, n_my-1)`` makes row *b* alias row *a*, so every
    store of *b* rewrites *a*'s own values (idempotent) and the refill guards fall
    out automatically (``j+DEPTH >= n_my`` whenever ``jb == j``)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = tidx // 32
    smem = utils.SmemAllocator()
    ring = smem.allocate_array(_BF16, WARPS * DEPTH * D, byte_alignment=128)
    bars = smem.allocate_array(_I64, WARPS * DEPTH, byte_alignment=8)
    my_ring = ring + warp * (DEPTH * D)
    my_bars = bars + warp * DEPTH
    ring_v = cutlass.Pointer(my_ring, dtype=_BF16)
    scp = cutlass.Pointer(sc.iterator, dtype=_BF16)
    shp = cutlass.Pointer(sh.iterator, dtype=_BF16)
    op = cutlass.Pointer(out.iterator, dtype=_BF16)
    mp = cutlass.Pointer(mean.iterator, dtype=_F32)
    rp = cutlass.Pointer(rstd.iterator, dtype=_F32)

    row_begin = bidx * chunk
    row_end = cutlass.min(row_begin + chunk, n_rows)
    rem = row_end - row_begin - warp
    n_my = cutlass.max(rem + (WARPS - 1), 0) // WARPS

    if lane == 0:
        for s in cutlass.range_constexpr(DEPTH):
            cute.arch.mbarrier_init(my_bars + s, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    if lane == 0:
        for p in cutlass.range_constexpr(DEPTH):
            if p < n_my:
                row = row_begin + warp + p * WARPS
                b = row // S
                s = row - b * S
                cute.arch.mbarrier_arrive_and_expect_tx(my_bars + p, ROW_BYTES)
                _row_copy(my_ring + p * D, x.iterator + (b * x_stride_b + s * D), my_bars + p)

    for j in cutlass.range(0, n_my, 2):
        jb = cutlass.min(j + 1, n_my - 1)
        sa = j % DEPTH
        sb = jb % DEPTH
        phase = (j // DEPTH) % 2
        row_a = row_begin + warp + j * WARPS
        row_b = row_begin + warp + jb * WARPS
        ba = row_a // S
        bb = row_b // S
        ga = scp + ba * sc_stride_b
        ha = shp + ba * sc_stride_b
        gb = scp + bb * sc_stride_b
        hb = shp + bb * sc_stride_b
        oa = op + row_a * D
        ob = op + row_b * D
        pa = ring_v + sa * D
        pb = ring_v + sb * D
        cute.arch.mbarrier_wait(my_bars + sa, phase)
        cute.arch.mbarrier_wait(my_bars + sb, phase)
        # pass 1: sums
        s1a = _F32(0.0)
        s1b = _F32(0.0)
        for k in cutlass.range_constexpr(VPL):
            off = (k * 32 + lane) * VEC
            s1a = s1a + (pa + off).load(16, count=VEC).to(_F32).reduce("add")
            s1b = s1b + (pb + off).load(16, count=VEC).to(_F32).reduce("add")
        s1a, s1b = _warp_sum2(s1a, s1b)
        mean_a = s1a / _F32(D)
        mean_b = s1b / _F32(D)
        cute.arch.sync_warp()   # compiler fence: pass 2 re-reads shared memory
        # pass 2: exact centred variances
        s2a = _F32(0.0)
        s2b = _F32(0.0)
        for k in cutlass.range_constexpr(VPL):
            off = (k * 32 + lane) * VEC
            da = (pa + off).load(16, count=VEC).to(_F32) - mean_a
            db = (pb + off).load(16, count=VEC).to(_F32) - mean_b
            s2a = s2a + (da * da).reduce("add")
            s2b = s2b + (db * db).reduce("add")
        s2a, s2b = _warp_sum2(s2a, s2b)
        rstd_a = _F32(1.0) / cute.math.sqrt(s2a / _F32(D) + _F32(EPS))
        rstd_b = _F32(1.0) / cute.math.sqrt(s2b / _F32(D) + _F32(EPS))
        cute.arch.sync_warp()
        # pass 3: outputs
        for k in cutlass.range_constexpr(VPL):
            off = (k * 32 + lane) * VEC
            xa = (pa + off).load(16, count=VEC).to(_F32) - mean_a
            xb = (pb + off).load(16, count=VEC).to(_F32) - mean_b
            gav = (ga + off).load(16, count=VEC, is_invariant=True).to(_F32) + 1.0
            hav = (ha + off).load(16, count=VEC, is_invariant=True).to(_F32)
            gbv = (gb + off).load(16, count=VEC, is_invariant=True).to(_F32) + 1.0
            hbv = (hb + off).load(16, count=VEC, is_invariant=True).to(_F32)
            ra = cute.math.fma(xa * rstd_a, gav, hav)
            rb = cute.math.fma(xb * rstd_b, gbv, hbv)
            (oa + off).store(ra.to(_BF16), alignment=16)
            (ob + off).store(rb.to(_BF16), alignment=16)
        if lane == 0:
            (mp + row_a).store(mean_a)
            (rp + row_a).store(rstd_a)
            (mp + row_b).store(mean_b)
            (rp + row_b).store(rstd_b)
        cute.arch.sync_warp()
        if lane == 0:
            jn = j + DEPTH
            if jn < n_my:
                rown = row_begin + warp + jn * WARPS
                bn = rown // S
                sn = rown - bn * S
                cute.arch.mbarrier_arrive_and_expect_tx(my_bars + sa, ROW_BYTES)
                _row_copy(my_ring + sa * D, x.iterator + (bn * x_stride_b + sn * D), my_bars + sa)
            jm = jb + DEPTH
            if jm < n_my:
                rowm = row_begin + warp + jm * WARPS
                bm = rowm // S
                sm = rowm - bm * S
                cute.arch.mbarrier_arrive_and_expect_tx(my_bars + sb, ROW_BYTES)
                _row_copy(my_ring + sb * D, x.iterator + (bm * x_stride_b + sm * D), my_bars + sb)


@cute.jit
def ln_adaln_fwd_ring2_host(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor,
    n_rows: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32, chunk: _I32, grid_x: _I32,
    stream: cuda.CUstream,
    WARPS: cutlass.Constexpr[int], DEPTH: cutlass.Constexpr[int],
) -> None:
    ln_adaln_fwd_ring2_kernel(
        x, sc, sh, out, mean, rstd, n_rows, S, x_stride_b, sc_stride_b, chunk, WARPS, DEPTH,
    ).launch(grid=(grid_x, 1, 1), block=(WARPS * 32, 1, 1), stream=stream,
             min_blocks_per_mp=1, preferred_smem_carveout=SMEM_CARVEOUT_PCT)


# ---------------------------------------------------------------------------
# Plain kernels: one row per CTA, TPR threads per row, exact D
# ---------------------------------------------------------------------------

@cute.kernel
def ln_adaln_fwd_plain_kernel(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor,
    S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    TPR: cutlass.Constexpr[int], MODE: cutlass.Constexpr[int],
) -> None:
    """``MODE`` 0: modulation vectors loaded after the second barrier;
    1: loaded before the first barrier and held as bf16 vectors, so nothing after
    ``bar.sync`` depends on memory (``pm``); 2: as 1 but with plain
    ``ld.global`` instead of ``ld.global.nc`` (``pc128``)."""
    W: cutlass.Constexpr[int] = TPR // 32
    GV: cutlass.Constexpr[int] = D // (TPR * VEC)
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    wid = tidx // 32
    smem = utils.SmemAllocator()
    red = smem.allocate_array(_F32, 2 * W, byte_alignment=16)
    red_v = cutlass.Pointer(red, dtype=_F32)
    b = row // S
    s = row - b * S
    xp = cutlass.Pointer(x.iterator, dtype=_BF16) + (b * x_stride_b + s * D)
    grow = cutlass.Pointer(sc.iterator, dtype=_BF16) + b * sc_stride_b
    hrow = cutlass.Pointer(sh.iterator, dtype=_BF16) + b * sc_stride_b
    orow = cutlass.Pointer(out.iterator, dtype=_BF16) + row * D
    mp = cutlass.Pointer(mean.iterator, dtype=_F32)
    rp = cutlass.Pointer(rstd.iterator, dtype=_F32)

    v = []
    gm = []
    hm = []
    for k in cutlass.range_constexpr(GV):
        v.append((xp + (k * TPR + tidx) * VEC).load(16, count=VEC).to(_F32))
        if cutlass.const_expr(MODE == 1):
            off = (k * TPR + tidx) * VEC
            gm.append((grow + off).load(16, count=VEC, is_invariant=True))
            hm.append((hrow + off).load(16, count=VEC, is_invariant=True))
        if cutlass.const_expr(MODE == 2):
            off = (k * TPR + tidx) * VEC
            gm.append((grow + off).load(16, count=VEC))
            hm.append((hrow + off).load(16, count=VEC))
    s1 = _F32(0.0)
    for k in cutlass.range_constexpr(GV):
        s1 = s1 + v[k].reduce("add")
    s1 = _warp_sum(s1)
    if lane == 0:
        (red_v + wid).store(s1)
    cute.arch.sync_threads()
    t1 = _F32(0.0)
    for w in cutlass.range_constexpr(W):
        t1 = t1 + (red_v + w).load()
    mean_v = t1 / _F32(D)
    s2 = _F32(0.0)
    for k in cutlass.range_constexpr(GV):
        d = v[k] - mean_v
        s2 = s2 + (d * d).reduce("add")
    s2 = _warp_sum(s2)
    if lane == 0:
        (red_v + W + wid).store(s2)
    cute.arch.sync_threads()
    t2 = _F32(0.0)
    for w in cutlass.range_constexpr(W):
        t2 = t2 + (red_v + W + w).load()
    rstd_v = _F32(1.0) / cute.math.sqrt(t2 / _F32(D) + _F32(EPS))
    for k in cutlass.range_constexpr(GV):
        off = (k * TPR + tidx) * VEC
        if cutlass.const_expr(MODE >= 1):
            g = gm[k].to(_F32) + 1.0
            hh = hm[k].to(_F32)
        else:
            g = (grow + off).load(16, count=VEC, is_invariant=True).to(_F32) + 1.0
            hh = (hrow + off).load(16, count=VEC, is_invariant=True).to(_F32)
        res = cute.math.fma((v[k] - mean_v) * rstd_v, g, hh)
        (orow + off).store(res.to(_BF16), alignment=16)
    if tidx == 0:
        (mp + row).store(mean_v)
        (rp + row).store(rstd_v)


@cute.jit
def ln_adaln_fwd_plain_host(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor,
    n_rows: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    stream: cuda.CUstream,
    TPR: cutlass.Constexpr[int], MODE: cutlass.Constexpr[int],
    MINB: cutlass.Constexpr[int], CARVE: cutlass.Constexpr[int],
) -> None:
    ln_adaln_fwd_plain_kernel(
        x, sc, sh, out, mean, rstd, S, x_stride_b, sc_stride_b, TPR, MODE,
    ).launch(grid=(n_rows, 1, 1), block=(TPR, 1, 1), stream=stream,
             min_blocks_per_mp=MINB, preferred_smem_carveout=CARVE)


@cute.jit
def ln_adaln_fwd_plain_host_noattr(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor,
    n_rows: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    stream: cuda.CUstream,
    TPR: cutlass.Constexpr[int], MODE: cutlass.Constexpr[int],
) -> None:
    """The same kernel with NO ``min_blocks_per_mp`` (no ``.minnctasm``) and
    NO ``preferred_smem_carveout`` (no ``cuFuncSetAttribute(PREFERRED_SHARED_MEMORY_CARVEOUT)``),
    i.e. the launch shape of the Triton kernel and of A2's ``rb_alt``."""
    ln_adaln_fwd_plain_kernel(
        x, sc, sh, out, mean, rstd, S, x_stride_b, sc_stride_b, TPR, MODE,
    ).launch(grid=(n_rows, 1, 1), block=(TPR, 1, 1), stream=stream)


@cute.kernel
def ln_adaln_fwd_plain_dbg_kernel(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor, dbg: cute.Tensor,
    S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    TPR: cutlass.Constexpr[int],
) -> None:
    """Instrumented twin: the ``pm128`` body (MODE 1) with per-CTA stamps.
    Thread 0 reads %globaltimer and %clock64 at entry and after its own stores, and
    writes ``dbg[row*4 : row*4+4] = t0, t1, clk1-clk0, smid``. No extra barrier."""
    W: cutlass.Constexpr[int] = TPR // 32
    GV: cutlass.Constexpr[int] = D // (TPR * VEC)
    t0 = _globaltimer()
    c0 = _clock64()
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    wid = tidx // 32
    smem = utils.SmemAllocator()
    red = smem.allocate_array(_F32, 2 * W, byte_alignment=16)
    red_v = cutlass.Pointer(red, dtype=_F32)
    b = row // S
    s = row - b * S
    xp = cutlass.Pointer(x.iterator, dtype=_BF16) + (b * x_stride_b + s * D)
    grow = cutlass.Pointer(sc.iterator, dtype=_BF16) + b * sc_stride_b
    hrow = cutlass.Pointer(sh.iterator, dtype=_BF16) + b * sc_stride_b
    orow = cutlass.Pointer(out.iterator, dtype=_BF16) + row * D
    mp = cutlass.Pointer(mean.iterator, dtype=_F32)
    rp = cutlass.Pointer(rstd.iterator, dtype=_F32)
    dp = cutlass.Pointer(dbg.iterator, dtype=_I64) + row * DBG_WORDS

    v = []
    gm = []
    hm = []
    for k in cutlass.range_constexpr(GV):
        off = (k * TPR + tidx) * VEC
        v.append((xp + off).load(16, count=VEC).to(_F32))
        gm.append((grow + off).load(16, count=VEC, is_invariant=True))
        hm.append((hrow + off).load(16, count=VEC, is_invariant=True))
    s1 = _F32(0.0)
    for k in cutlass.range_constexpr(GV):
        s1 = s1 + v[k].reduce("add")
    s1 = _warp_sum(s1)
    if lane == 0:
        (red_v + wid).store(s1)
    cute.arch.sync_threads()
    t1 = _F32(0.0)
    for w in cutlass.range_constexpr(W):
        t1 = t1 + (red_v + w).load()
    mean_v = t1 / _F32(D)
    s2 = _F32(0.0)
    for k in cutlass.range_constexpr(GV):
        d = v[k] - mean_v
        s2 = s2 + (d * d).reduce("add")
    s2 = _warp_sum(s2)
    if lane == 0:
        (red_v + W + wid).store(s2)
    cute.arch.sync_threads()
    t2 = _F32(0.0)
    for w in cutlass.range_constexpr(W):
        t2 = t2 + (red_v + W + w).load()
    rstd_v = _F32(1.0) / cute.math.sqrt(t2 / _F32(D) + _F32(EPS))
    for k in cutlass.range_constexpr(GV):
        off = (k * TPR + tidx) * VEC
        g = gm[k].to(_F32) + 1.0
        hh = hm[k].to(_F32)
        res = cute.math.fma((v[k] - mean_v) * rstd_v, g, hh)
        (orow + off).store(res.to(_BF16), alignment=16)
    if tidx == 0:
        (mp + row).store(mean_v)
        (rp + row).store(rstd_v)
        tt = _globaltimer()
        cc = _clock64()
        dp.store(t0)
        (dp + 1).store(tt)
        (dp + 2).store(cc - c0)
        (dp + 3).store(_smid().to(_I64))


@cute.jit
def ln_adaln_fwd_plain_dbg_host(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor, dbg: cute.Tensor,
    n_rows: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    stream: cuda.CUstream,
    TPR: cutlass.Constexpr[int], MINB: cutlass.Constexpr[int], CARVE: cutlass.Constexpr[int],
) -> None:
    ln_adaln_fwd_plain_dbg_kernel(
        x, sc, sh, out, mean, rstd, dbg, S, x_stride_b, sc_stride_b, TPR,
    ).launch(grid=(n_rows, 1, 1), block=(TPR, 1, 1), stream=stream,
             min_blocks_per_mp=MINB, preferred_smem_carveout=CARVE)


def _warp_row(lane, xp, orow, grow, hrow, gm, hm, load_mod, cvt, mcvt, flags):
    """Traced helper (plain Python -- no ``range_constexpr`` across a call boundary):
    one warp normalises and modulates ONE 3072-wide row. ``gm`` / ``hm`` are the CTA's modulation
    vector lists: filled here when ``load_mod`` (the CTA's first row), reused as they stand otherwise
    (``RPC = 2``: the second row of the same batch). ``cvt`` = the six runtime x-conversion
    parameters; ``mcvt`` = None (modulation converted with ``.to(F32)``: every first row, so row A is the
    one-row arm's code) or a runtime ``(sh, mk)`` pair (the CTA's second row), through which the held bf16
    modulation vectors are converted so the two rows' conversions cannot be common-subexpressed
    (with ``.to`` on both rows LLVM CSE'd the two rows' identical
    ``gm[k].to(F32) + 1`` / ``hm[k].to(F32)`` and kept 192 fp32 copies live -- 255 registers, STACK 64).
    ``flags`` = (HOLD_BF16, HOLD_WORDS, PRE_G, PRE_H, MID_H) as Python bools. Element-to-lane mapping and
    summation order are ``r2``'s (bf16 -> fp32 is exact either way); returns ``(mean, rstd)``."""
    HOLD_BF16, HOLD_WORDS, PRE_G, PRE_H, MID_H = flags
    sh1, mk1, sh2, mk2, sh3, mk3 = cvt

    def mod_f32(vec):
        if mcvt is None:
            return vec.to(_F32)
        return _words_to_f32x8(vec.bitcast(_I32), mcvt[0], mcvt[1])
    v = []
    for k in range(VPL):
        off = (k * 32 + lane) * VEC
        xv = (xp + off).load(16, count=VEC)
        if HOLD_WORDS:
            v.append(xv.bitcast(_I32))
        elif HOLD_BF16:
            v.append(xv)
        else:
            v.append(xv.to(_F32))
        if PRE_G and load_mod:
            gm.append((grow + off).load(16, count=VEC))
        if PRE_H and load_mod:
            hm.append((hrow + off).load(16, count=VEC))

    s1 = _F32(0.0)
    for k in range(VPL):
        if HOLD_WORDS:
            xk = _words_to_f32x8(v[k], sh1, mk1)
        elif HOLD_BF16:
            xk = v[k].to(_F32)
        else:
            xk = v[k]
        s1 = s1 + xk.reduce("add")
    mean_v = _warp_sum(s1) / _F32(D)
    if MID_H and load_mod:
        for k in range(VPL):
            hm.append((hrow + (k * 32 + lane) * VEC).load(16, count=VEC))
    s2 = _F32(0.0)
    for k in range(VPL):
        if HOLD_WORDS:
            xk = _words_to_f32x8(v[k], sh2, mk2)
        elif HOLD_BF16:
            xk = v[k].to(_F32)
        else:
            xk = v[k]
        d = xk - mean_v
        s2 = s2 + (d * d).reduce("add")
    var_v = _warp_sum(s2) / _F32(D)
    rstd_v = _F32(1.0) / cute.math.sqrt(var_v + _F32(EPS))
    for k in range(VPL):
        off = (k * 32 + lane) * VEC
        if HOLD_WORDS:
            xk = _words_to_f32x8(v[k], sh3, mk3)
        elif HOLD_BF16:
            xk = v[k].to(_F32)
        else:
            xk = v[k]
        if PRE_G:
            g = mod_f32(gm[k]) + 1.0
        else:
            g = (grow + off).load(16, count=VEC).to(_F32) + 1.0
        if PRE_H or MID_H:
            hh = mod_f32(hm[k])
        else:
            hh = (hrow + off).load(16, count=VEC).to(_F32)
        res = cute.math.fma((xk - mean_v) * rstd_v, g, hh)
        (orow + off).store(res.to(_BF16), alignment=16)
    return mean_v, rstd_v


def _warp_stamp(dbg, row, t0, c0):
    """Traced helper (no loops): lane 0's per-row stamp record -- t0, now, clock delta, smid."""
    dp = cutlass.Pointer(dbg.iterator, dtype=_I64) + row * DBG_WORDS
    tt = _globaltimer()
    cc = _clock64()
    dp.store(t0)
    (dp + 1).store(tt)
    (dp + 2).store(cc - c0)
    (dp + 3).store(_smid().to(_I64))


@cute.kernel
def ln_adaln_fwd_warp_kernel(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor, dbg: cute.Tensor,
    S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    sh1: _I32, mk1: _I32, sh2: _I32, mk2: _I32, sh3: _I32, mk3: _I32,
    MODE: cutlass.Constexpr[int], DBG: cutlass.Constexpr[bool], RPC: cutlass.Constexpr[int],
) -> None:
    """``pw32`` / ``pw32s`` / ``pw32d``/``pw32c``/``pw32b`` and
    the opaque-conversion modes 5-7 (``pw32d2``/``pw32c2``/``pw32b2``; ``sh*``/``mk*`` are the per-pass
    conversion constants 16 / 0xffff0000 passed at RUNTIME, see :func:`_words_to_f32x8`; unused below 5):
    ONE warp per row, ``RPC`` rows per 32-thread CTA. The row is held in registers
    (12 vectors per lane, the ring's ``HOLD`` mapping), both reductions are warp shuffles: no
    shared memory, no ``bar.sync``, no launch attribute. Element-to-lane mapping and
    summation order are ``r2``'s, so ``out``, ``mean`` and ``rstd`` are bitwise ``r2``'s.

    ``MODE``: 0 -- ``x`` held as fp32, modulation loaded after the statistics (``pw32``);
    1 -- ``x`` fp32, ``scale`` and ``shift`` prefetched before the statistics and held as bf16
    (``pw32s``: the exposed second round trip ``pm128``'s MODE 1 removed); 2 -- ``x`` held as the
    loaded bf16 vectors and converted at each use (exact), nothing prefetched (``pw32d``);
    3 -- ``x`` bf16-held, ``scale`` prefetched, ``shift`` loaded right after ``mean`` so its trip
    overlaps the variance pass (``pw32c``, the footprint lever); 4 -- ``x`` bf16-held,
    both prefetched (``pw32b``). The arithmetic is identical in every mode.

    ``RPC``: 1 -- grid = n_rows, CTA = row (every arm above). 2 -- grid = B * ceil(S / 2);
    CTA ``c`` owns rows ``(b, s0)`` and ``(b, s0 + 1)`` of ONE batch (``b = c // ceil(S/2)``), processed
    in sequence by the same warp; the second row reuses the modulation vectors the first row loaded
    and is skipped (warp-uniformly) when ``s0 + 1 == S``. Halves the CTA count so the SSB shape
    (2048 rows) runs one wave on the SMs a resident collective leaves (``pw32c2r2`` / ``pw32b2r2``).
    ``DBG``: lane 0 stamps %globaltimer at entry / after its stores, the
    %clock64 delta and %smid into ``dbg[row*4 : row*4+4]`` -- the instrument. With ``RPC = 2``
    both rows' records are ``[t0, t_end]`` of the whole CTA, stamped once after every store."""
    HOLD_BF16: cutlass.Constexpr[bool] = MODE in (2, 3, 4)
    HOLD_WORDS: cutlass.Constexpr[bool] = MODE >= 5
    PRE_G: cutlass.Constexpr[bool] = MODE in (1, 3, 4, 6, 7)
    PRE_H: cutlass.Constexpr[bool] = MODE in (1, 4, 7)
    MID_H: cutlass.Constexpr[bool] = MODE in (3, 6)
    flags = (HOLD_BF16, HOLD_WORDS, PRE_G, PRE_H, MID_H)
    cvt = (sh1, mk1, sh2, mk2, sh3, mk3)
    if cutlass.const_expr(DBG):
        t0 = _globaltimer()
        c0 = _clock64()
    lane, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(RPC == 1):
        b = cta // S
        s = cta - b * S
    else:
        SP = (S + (RPC - 1)) // RPC
        b = cta // SP
        s = (cta - b * SP) * RPC
    row = b * S + s
    xb = cutlass.Pointer(x.iterator, dtype=_BF16) + b * x_stride_b
    grow = cutlass.Pointer(sc.iterator, dtype=_BF16) + b * sc_stride_b
    hrow = cutlass.Pointer(sh.iterator, dtype=_BF16) + b * sc_stride_b
    ob = cutlass.Pointer(out.iterator, dtype=_BF16)
    mp = cutlass.Pointer(mean.iterator, dtype=_F32)
    rp = cutlass.Pointer(rstd.iterator, dtype=_F32)
    gm = []
    hm = []
    mean_v, rstd_v = _warp_row(lane, xb + s * D, ob + row * D, grow, hrow, gm, hm, True, cvt, None, flags)
    if lane == 0:
        (mp + row).store(mean_v)
        (rp + row).store(rstd_v)
        if cutlass.const_expr(DBG and RPC == 1):
            _warp_stamp(dbg, row, t0, c0)
    if cutlass.const_expr(RPC == 2):
        s1 = s + 1
        if s1 < S:
            row1 = row + 1
            mean_1, rstd_1 = _warp_row(lane, xb + s1 * D, ob + row1 * D, grow, hrow, gm, hm, False, cvt, (sh2, mk2), flags)
            if lane == 0:
                (mp + row1).store(mean_1)
                (rp + row1).store(rstd_1)
        if cutlass.const_expr(DBG):
            # ONE stamp per CTA, after every store (as the one-row twin): a mid-CTA %globaltimer read is a
            # scheduling barrier ptxas sinks row A's 12 output stores to, holding 48 output words live
            # (stamped twin 220 regs vs 170 plain). Both rows record [t0, t_end] of the CTA.
            if lane == 0:
                _warp_stamp(dbg, row, t0, c0)
                if s1 < S:
                    _warp_stamp(dbg, row + 1, t0, c0)


@cute.kernel
def ln_adaln_fwd_warpx_kernel(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor, dbg: cute.Tensor,
    n_rows: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    WARPS: cutlass.Constexpr[int], DBG: cutlass.Constexpr[bool],
) -> None:
    """``pw128x``: ``WARPS`` INDEPENDENT warps per CTA, one row each
    (grid = ceil(n_rows / WARPS), block = 32 * WARPS); no shared memory, no barrier, no
    launch attribute -- the ``pw32`` body (no prefetch) at a 128-thread CTA width, so the
    CTA-width and the smem/barrier properties of ``pw32`` can be told apart. Bitwise ``r2``.
    Rows past ``n_rows`` (ragged tail) are skipped by the whole warp."""
    if cutlass.const_expr(DBG):
        t0 = _globaltimer()
        c0 = _clock64()
    tidx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    wid = tidx // 32
    row = cta * WARPS + wid
    if row < n_rows:
        b = row // S
        s = row - b * S
        xp = cutlass.Pointer(x.iterator, dtype=_BF16) + (b * x_stride_b + s * D)
        grow = cutlass.Pointer(sc.iterator, dtype=_BF16) + b * sc_stride_b
        hrow = cutlass.Pointer(sh.iterator, dtype=_BF16) + b * sc_stride_b
        orow = cutlass.Pointer(out.iterator, dtype=_BF16) + row * D
        mp = cutlass.Pointer(mean.iterator, dtype=_F32)
        rp = cutlass.Pointer(rstd.iterator, dtype=_F32)
        v = []
        for k in cutlass.range_constexpr(VPL):
            v.append((xp + (k * 32 + lane) * VEC).load(16, count=VEC).to(_F32))
        s1 = _F32(0.0)
        for k in cutlass.range_constexpr(VPL):
            s1 = s1 + v[k].reduce("add")
        mean_v = _warp_sum(s1) / _F32(D)
        s2 = _F32(0.0)
        for k in cutlass.range_constexpr(VPL):
            d = v[k] - mean_v
            s2 = s2 + (d * d).reduce("add")
        var_v = _warp_sum(s2) / _F32(D)
        rstd_v = _F32(1.0) / cute.math.sqrt(var_v + _F32(EPS))
        for k in cutlass.range_constexpr(VPL):
            off = (k * 32 + lane) * VEC
            g = (grow + off).load(16, count=VEC).to(_F32) + 1.0
            hh = (hrow + off).load(16, count=VEC).to(_F32)
            res = cute.math.fma((v[k] - mean_v) * rstd_v, g, hh)
            (orow + off).store(res.to(_BF16), alignment=16)
        if lane == 0:
            (mp + row).store(mean_v)
            (rp + row).store(rstd_v)
            if cutlass.const_expr(DBG):
                dp = cutlass.Pointer(dbg.iterator, dtype=_I64) + row * DBG_WORDS
                tt = _globaltimer()
                cc = _clock64()
                dp.store(t0)
                (dp + 1).store(tt)
                (dp + 2).store(cc - c0)
                (dp + 3).store(_smid().to(_I64))


@cute.jit
def ln_adaln_fwd_warp_host(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor, dbg: cute.Tensor,
    n_ctas: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    sh1: _I32, mk1: _I32, sh2: _I32, mk2: _I32, sh3: _I32, mk3: _I32,
    stream: cuda.CUstream,
    MODE: cutlass.Constexpr[int], DBG: cutlass.Constexpr[bool], RPC: cutlass.Constexpr[int],
    MINB: cutlass.Constexpr[int],
) -> None:
    k = ln_adaln_fwd_warp_kernel(
        x, sc, sh, out, mean, rstd, dbg, S, x_stride_b, sc_stride_b, sh1, mk1, sh2, mk2, sh3, mk3, MODE, DBG, RPC,
    )
    if cutlass.const_expr(MINB == 0):
        k.launch(grid=(n_ctas, 1, 1), block=(32, 1, 1), stream=stream)
    else:
        # ``pw32c2r2p``: pin the residency (ptxas register cap 65536 / (32 * MINB)); the carveout is
        # passed explicitly as every attributed launch in this file does, smem is 0 here.
        k.launch(grid=(n_ctas, 1, 1), block=(32, 1, 1), stream=stream,
                 min_blocks_per_mp=MINB, preferred_smem_carveout=PLAIN_CARVEOUT_PCT)


@cute.jit
def ln_adaln_fwd_warpx_host(
    x: cute.Tensor, sc: cute.Tensor, sh: cute.Tensor,
    out: cute.Tensor, mean: cute.Tensor, rstd: cute.Tensor, dbg: cute.Tensor,
    n_rows: _I32, n_ctas: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    stream: cuda.CUstream,
    WARPS: cutlass.Constexpr[int], DBG: cutlass.Constexpr[bool],
) -> None:
    ln_adaln_fwd_warpx_kernel(
        x, sc, sh, out, mean, rstd, dbg, n_rows, S, x_stride_b, sc_stride_b, WARPS, DBG,
    ).launch(grid=(n_ctas, 1, 1), block=(32 * WARPS, 1, 1), stream=stream)


@cute.kernel
def ln_adaln_bwd_plain_kernel(
    go: cute.Tensor, x: cute.Tensor, sc: cute.Tensor,
    mean: cute.Tensor, rstd: cute.Tensor,
    gx: cute.Tensor, gs: cute.Tensor,
    res: cute.Tensor, ge: cute.Tensor, gsum: cute.Tensor,
    S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    TPR: cutlass.Constexpr[int],
    HAS_RES: cutlass.Constexpr[bool], HAS_GE: cutlass.Constexpr[bool],
    MODE: cutlass.Constexpr[int],
) -> None:
    """``c1 = mean(gn)`` and ``c2 = mean(gn*n)`` both come from the forward's
    ``mean``/``rstd``, so one pass accumulates both and ONE barrier reduces them.

    ``MODE``: 0 -- ``g``/``n`` held in fp32 across
    the barrier, ``1+scale`` re-read and ``residual`` loaded AFTER it (a
    second dependent DRAM round trip per row); 1 -- ``residual`` loaded before the
    barrier as a held bf16 vector (``pr``); 2 -- as 1 and ``1+scale`` held in fp32
    too, nothing after ``bar.sync`` depends on memory (``ph``); 3 -- stateless
    second pass: only ``residual`` held, ``grad_out``/``grad_extra``/``x``/``scale``
    re-read from L1/L2 after the barrier (``pL``, smallest register footprint)."""
    W: cutlass.Constexpr[int] = TPR // 32
    GV: cutlass.Constexpr[int] = D // (TPR * VEC)
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    wid = tidx // 32
    smem = utils.SmemAllocator()
    red = smem.allocate_array(_F32, 2 * W, byte_alignment=16)
    red_v = cutlass.Pointer(red, dtype=_F32)
    b = row // S
    s = row - b * S
    rowd = row * D
    gop = cutlass.Pointer(go.iterator, dtype=_BF16) + rowd
    xp = cutlass.Pointer(x.iterator, dtype=_BF16) + (b * x_stride_b + s * D)
    grow = cutlass.Pointer(sc.iterator, dtype=_BF16) + b * sc_stride_b
    gxp = cutlass.Pointer(gx.iterator, dtype=_BF16) + rowd
    gsp = cutlass.Pointer(gs.iterator, dtype=_BF16) + rowd
    resp = cutlass.Pointer(res.iterator, dtype=_BF16) + rowd
    gep = cutlass.Pointer(ge.iterator, dtype=_BF16) + rowd
    gsump = cutlass.Pointer(gsum.iterator, dtype=_BF16) + rowd
    mean_v = (cutlass.Pointer(mean.iterator, dtype=_F32) + row).load()
    rstd_v = (cutlass.Pointer(rstd.iterator, dtype=_F32) + row).load()

    gv = []
    nv = []
    g1v = []
    rv = []
    c1 = _F32(0.0)
    c2 = _F32(0.0)
    for k in cutlass.range_constexpr(GV):
        off = (k * TPR + tidx) * VEC
        g = (gop + off).load(16, count=VEC).to(_F32)
        if cutlass.const_expr(HAS_GE):
            g = g + (gep + off).load(16, count=VEC).to(_F32)
            (gsump + off).store(g.to(_BF16), alignment=16)
        g1 = (grow + off).load(16, count=VEC, is_invariant=True).to(_F32) + 1.0
        n = ((xp + off).load(16, count=VEC).to(_F32) - mean_v) * rstd_v
        gn = g * g1
        c1 = c1 + gn.reduce("add")
        c2 = c2 + (gn * n).reduce("add")
        if cutlass.const_expr(MODE != 3):
            gv.append(g)
            nv.append(n)
        if cutlass.const_expr(MODE == 2):
            g1v.append(g1)
        if cutlass.const_expr(HAS_RES and MODE >= 1):
            rv.append((resp + off).load(16, count=VEC))
    c1 = _warp_sum(c1)
    c2 = _warp_sum(c2)
    if lane == 0:
        (red_v + wid).store(c1)
        (red_v + W + wid).store(c2)
    cute.arch.sync_threads()
    t1 = _F32(0.0)
    t2 = _F32(0.0)
    for w in cutlass.range_constexpr(W):
        t1 = t1 + (red_v + w).load()
        t2 = t2 + (red_v + W + w).load()
    c1 = t1 / _F32(D)
    c2 = t2 / _F32(D)
    if cutlass.const_expr(MODE == 3):
        cute.arch.sync_warp()   # compiler fence: pass 2 re-reads memory instead of holding pass 1
    for k in cutlass.range_constexpr(GV):
        off = (k * TPR + tidx) * VEC
        if cutlass.const_expr(MODE == 3):
            g = (gop + off).load(16, count=VEC).to(_F32)
            if cutlass.const_expr(HAS_GE):
                g = g + (gep + off).load(16, count=VEC).to(_F32)
            n = ((xp + off).load(16, count=VEC).to(_F32) - mean_v) * rstd_v
        else:
            g = gv[k]
            n = nv[k]
        if cutlass.const_expr(MODE == 2):
            g1 = g1v[k]
        else:
            g1 = (grow + off).load(16, count=VEC, is_invariant=True).to(_F32) + 1.0
        gn = g * g1
        gxv = ((gn - c1) - n * c2) * rstd_v
        if cutlass.const_expr(HAS_RES):
            if cutlass.const_expr(MODE >= 1):
                gxv = gxv + rv[k].to(_F32)
            else:
                gxv = gxv + (resp + off).load(16, count=VEC).to(_F32)
        (gxp + off).store(gxv.to(_BF16), alignment=16)
        (gsp + off).store((g * n).to(_BF16), alignment=16)


@cute.jit
def ln_adaln_bwd_plain_host(
    go: cute.Tensor, x: cute.Tensor, sc: cute.Tensor,
    mean: cute.Tensor, rstd: cute.Tensor,
    gx: cute.Tensor, gs: cute.Tensor,
    res: cute.Tensor, ge: cute.Tensor, gsum: cute.Tensor,
    n_rows: _I32, S: _I32, x_stride_b: _I32, sc_stride_b: _I32,
    stream: cuda.CUstream,
    TPR: cutlass.Constexpr[int], HAS_RES: cutlass.Constexpr[bool], HAS_GE: cutlass.Constexpr[bool],
    MODE: cutlass.Constexpr[int], MINB: cutlass.Constexpr[int], CARVE: cutlass.Constexpr[int],
) -> None:
    ln_adaln_bwd_plain_kernel(
        go, x, sc, mean, rstd, gx, gs, res, ge, gsum, S, x_stride_b, sc_stride_b, TPR, HAS_RES, HAS_GE, MODE,
    ).launch(grid=(n_rows, 1, 1), block=(TPR, 1, 1), stream=stream,
             min_blocks_per_mp=MINB, preferred_smem_carveout=CARVE)


# ---------------------------------------------------------------------------
# Compile cache, geometry, argument plumbing
# ---------------------------------------------------------------------------

_compiled: dict[tuple, Callable] = {}
_props: dict[int, object] = {}
PLAIN_TPRS = (128, 192, 384)


def _device_props(device_index: int):
    if device_index not in _props:
        _props[device_index] = torch.cuda.get_device_properties(device_index)
    return _props[device_index]


def _fake_bf16():
    return make_fake_compact_tensor(_BF16, (cute.sym_int64(divisibility=VEC),), assumed_align=16)


def _fake_f32():
    return make_fake_compact_tensor(_F32, (cute.sym_int64(divisibility=1),), assumed_align=4)


def _fwd_tens():
    return [_fake_bf16(), _fake_bf16(), _fake_bf16(), _fake_bf16(), _fake_f32(), _fake_f32()]


def _bwd_tens():
    return [_fake_bf16(), _fake_bf16(), _fake_bf16(), _fake_f32(), _fake_f32(),
            _fake_bf16(), _fake_bf16(), _fake_bf16(), _fake_bf16(), _fake_bf16()]


def _compile(key: tuple, build: Callable) -> Callable:
    if key not in _compiled:
        with torch.cuda.device(key[1]):
            _compiled[key] = build()
    return _compiled[key]


def _compile_fwd_ring(dev: int, warps: int, depth: int) -> Callable:
    return _compile(("fwd_ring", dev, warps, depth), lambda: cute.compile(
        ln_adaln_fwd_ring_host, *_fwd_tens(), *([_I32(0)] * 6), make_fake_stream(),
        warps, depth, True, False, 1, options="--enable-tvm-ffi"))


def _compile_fwd_ring2(dev: int, warps: int, depth: int) -> Callable:
    return _compile(("fwd_ring2", dev, warps, depth), lambda: cute.compile(
        ln_adaln_fwd_ring2_host, *_fwd_tens(), *([_I32(0)] * 6), make_fake_stream(),
        warps, depth, options="--enable-tvm-ffi"))


def _compile_fwd_plain(dev: int, tpr: int, minb: int | None, carve: int | None, mode: int) -> Callable:
    if minb is None:
        return _compile(("fwd_plain_noattr", dev, tpr, mode), lambda: cute.compile(
            ln_adaln_fwd_plain_host_noattr, *_fwd_tens(), *([_I32(0)] * 4), make_fake_stream(),
            tpr, mode, options="--enable-tvm-ffi"))
    return _compile(("fwd_plain", dev, tpr, minb, carve, mode), lambda: cute.compile(
        ln_adaln_fwd_plain_host, *_fwd_tens(), *([_I32(0)] * 4), make_fake_stream(),
        tpr, mode, minb, carve, options="--enable-tvm-ffi"))


def _fake_i64():
    return make_fake_compact_tensor(_I64, (cute.sym_int64(divisibility=DBG_WORDS),), assumed_align=16)


def _compile_fwd_plain_dbg(dev: int, tpr: int, minb: int, carve: int) -> Callable:
    return _compile(("fwd_plain_dbg", dev, tpr, minb, carve), lambda: cute.compile(
        ln_adaln_fwd_plain_dbg_host, *_fwd_tens(), _fake_i64(), *([_I32(0)] * 4), make_fake_stream(),
        tpr, minb, carve, options="--enable-tvm-ffi"))


def _compile_fwd_warp(dev: int, mode: int, dbg: bool, rpc: int, minb: int) -> Callable:
    return _compile(("fwd_warp", dev, mode, dbg, rpc, minb), lambda: cute.compile(
        ln_adaln_fwd_warp_host, *_fwd_tens(), _fake_i64(), *([_I32(0)] * 10), make_fake_stream(),
        mode, dbg, rpc, minb, options="--enable-tvm-ffi"))


def _compile_fwd_warpx(dev: int, warps: int, dbg: bool) -> Callable:
    return _compile(("fwd_warpx", dev, warps, dbg), lambda: cute.compile(
        ln_adaln_fwd_warpx_host, *_fwd_tens(), _fake_i64(), *([_I32(0)] * 5), make_fake_stream(),
        warps, dbg, options="--enable-tvm-ffi"))


def _compile_bwd_plain(dev: int, tpr: int, has_res: bool, has_ge: bool, minb: int, carve: int,
                       mode: int) -> Callable:
    return _compile(("bwd_plain", dev, tpr, has_res, has_ge, minb, carve, mode), lambda: cute.compile(
        ln_adaln_bwd_plain_host, *_bwd_tens(), *([_I32(0)] * 4), make_fake_stream(),
        tpr, has_res, has_ge, mode, minb, carve, options="--enable-tvm-ffi"))


def smem_bytes_ring(warps: int, depth: int) -> int:
    """Shared memory one ring CTA asks for (ring + mbarriers)."""
    return warps * depth * ROW_BYTES + warps * depth * 8


def device_smem_limit(device_index: int) -> int:
    """Per-CTA dynamic smem the DSL can launch with on this device
    (Rubin's oversized mode is 334 848 B; elsewhere the driver's opt-in maximum)."""
    if torch.cuda.get_device_capability(device_index) == (10, 7):
        return 334848
    return int(getattr(_device_props(device_index), "shared_memory_per_block_optin", 232448))


def geometry(device_index: int, n_rows: int, warps: int) -> tuple[int, int]:
    """``(grid, chunk)`` for the ring kernels: one CTA per SM, each owning a contiguous
    run of rows; the grid is recomputed from the chunk so no CTA is empty."""
    sms = _device_props(device_index).multi_processor_count
    ctas = max(1, min(sms, math.ceil(n_rows / warps)))
    chunk = math.ceil(n_rows / ctas)
    grid = math.ceil(n_rows / chunk)
    return grid, chunk


def _stream(device_index: int) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(device_index).cuda_stream)


def _flat(t: torch.Tensor, extent: int) -> torch.Tensor:
    """A unit-stride 1-D view of ``t``'s storage from its own base, ``extent`` elements long."""
    return torch.as_strided(t, (extent,), (1,))


def _check_act(t: torch.Tensor, name: str, S: int) -> int:
    """Validate a ``[B, S, D]`` bf16 activation; return its 1-D storage extent."""
    if not t.is_cuda or t.dtype != torch.bfloat16:
        raise ValueError(f"fused_ln_adaln_a1: {name} must be a CUDA bf16 tensor")
    if t.dim() != 3 or t.shape[2] != D or t.shape[1] != S:
        raise ValueError(f"fused_ln_adaln_a1: {name} must be [B, {S}, {D}], got {tuple(t.shape)}")
    if t.stride(2) != 1 or t.stride(1) != D:
        raise ValueError(f"fused_ln_adaln_a1: {name} rows must be contiguous, got strides {t.stride()}")
    if t.stride(0) < S * D or t.stride(0) % VEC:
        raise ValueError(f"fused_ln_adaln_a1: {name} batch stride {t.stride(0)} invalid")
    if t.data_ptr() % 16:
        raise ValueError(f"fused_ln_adaln_a1: {name} base is not 16-byte aligned")
    extent = (t.shape[0] - 1) * t.stride(0) + S * D
    if extent > _INT32_MAX:
        raise ValueError(f"fused_ln_adaln_a1: {name} extent {extent} exceeds the Int32 index range")
    return extent


def _check_mod(t: torch.Tensor, name: str, B: int) -> int:
    """Validate a ``[B, 1, D]`` bf16 modulation view; return its 1-D storage extent."""
    if not t.is_cuda or t.dtype != torch.bfloat16:
        raise ValueError(f"fused_ln_adaln_a1: {name} must be a CUDA bf16 tensor")
    if t.dim() != 3 or t.shape[0] != B or t.shape[1] != 1 or t.shape[2] != D:
        raise ValueError(f"fused_ln_adaln_a1: {name} must be [{B}, 1, {D}], got {tuple(t.shape)}")
    if t.stride(2) != 1 or (B > 1 and (t.stride(0) < D or t.stride(0) % VEC)):
        raise ValueError(f"fused_ln_adaln_a1: {name} strides {t.stride()} invalid")
    if t.data_ptr() % 16:
        raise ValueError(f"fused_ln_adaln_a1: {name} base is not 16-byte aligned")
    return (B - 1) * t.stride(0) + D


def _check_stat(t: torch.Tensor, name: str, n_rows: int) -> None:
    if not t.is_cuda or t.dtype != torch.float32 or not t.is_contiguous() or t.numel() != n_rows:
        raise ValueError(f"fused_ln_adaln_a1: {name} must be a contiguous fp32 [{n_rows}] tensor")


def _check_fwd(x, scale, shift):
    B, S, _ = x.shape
    xe = _check_act(x, "x", S)
    sce = _check_mod(scale, "scale", B)
    she = _check_mod(shift, "shift", B)
    if shift.stride(0) != scale.stride(0):
        raise ValueError("fused_ln_adaln_a1: scale and shift must share a batch stride")
    return B, S, xe, sce, she


def _alloc_fwd(x, n_rows):
    B, S, _ = x.shape
    out = torch.empty(B, S, D, dtype=torch.bfloat16, device=x.device)
    mean = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    return out, mean, rstd


# ---------------------------------------------------------------------------
# Python entry points -- one per family
# ---------------------------------------------------------------------------

def fwd_ring(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
             warps: int = 8, depth: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``r2`` forward (warp-per-row bulk-copy ring, row held in registers).

    Args:
        x: ``[B, S, 3072]`` bf16; rows contiguous, batch stride may exceed ``S*3072``.
        scale: ``[B, 1, 3072]`` bf16, any batch stride (a split view is fine).
        shift: ``[B, 1, 3072]`` bf16, same layout rules as ``scale``.
        warps: Warps (= rows in flight) per CTA.
        depth: Ring depth in rows per warp.

    Returns:
        ``(out [B,S,3072] bf16 contiguous, mean [B*S] fp32, rstd [B*S] fp32)``.

    Raises:
        ValueError: On a layout, dtype, alignment, index-range or shared-memory violation.
    """
    B, S, xe, sce, she = _check_fwd(x, scale, shift)
    dev = x.device.index
    n_rows = B * S
    if warps < 1 or warps > 32 or depth < 1:
        raise ValueError(f"fused_ln_adaln_a1: warps={warps}, depth={depth} invalid")
    if smem_bytes_ring(warps, depth) > device_smem_limit(dev):
        raise ValueError(f"fused_ln_adaln_a1: ring warps={warps} depth={depth} needs "
                         f"{smem_bytes_ring(warps, depth)} B of shared memory, device allows "
                         f"{device_smem_limit(dev)}")
    grid, chunk = geometry(dev, n_rows, warps)
    out, mean, rstd = _alloc_fwd(x, n_rows)
    _compile_fwd_ring(dev, warps, depth)(
        _flat(x, xe), _flat(scale, sce), _flat(shift, she), out.view(-1), mean, rstd,
        n_rows, S, x.stride(0), scale.stride(0), chunk, grid, _stream(dev),
    )
    return out, mean, rstd


def fwd_ring2(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
              warps: int = 8, depth: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward on the ring with two independent rows per warp per iteration.

    Args:
        x: ``[B, S, 3072]`` bf16; rows contiguous, batch stride may exceed ``S*3072``.
        scale: ``[B, 1, 3072]`` bf16, any batch stride.
        shift: ``[B, 1, 3072]`` bf16, same layout rules as ``scale``.
        warps: Warps per CTA (each owns two rows per iteration).
        depth: Ring depth in rows per warp; even and >= 2 (4 for a prefetched pair).

    Returns:
        ``(out, mean, rstd)`` as :func:`fwd_ring`.

    Raises:
        ValueError: On a layout, dtype, alignment, index-range, depth or shared-memory violation.
    """
    B, S, xe, sce, she = _check_fwd(x, scale, shift)
    dev = x.device.index
    n_rows = B * S
    if warps < 1 or warps > 32 or depth < 2 or depth % 2:
        raise ValueError(f"fused_ln_adaln_a1: ring2 warps={warps}, depth={depth} invalid (depth even >= 2)")
    if smem_bytes_ring(warps, depth) > device_smem_limit(dev):
        raise ValueError(f"fused_ln_adaln_a1: ring2 warps={warps} depth={depth} needs "
                         f"{smem_bytes_ring(warps, depth)} B of shared memory, device allows "
                         f"{device_smem_limit(dev)}")
    grid, chunk = geometry(dev, n_rows, warps)
    out, mean, rstd = _alloc_fwd(x, n_rows)
    _compile_fwd_ring2(dev, warps, depth)(
        _flat(x, xe), _flat(scale, sce), _flat(shift, she), out.view(-1), mean, rstd,
        n_rows, S, x.stride(0), scale.stride(0), chunk, grid, _stream(dev),
    )
    return out, mean, rstd


def fwd_plain(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
              tpr: int = 128, min_blocks: int | None = 1, carveout: int | None = PLAIN_CARVEOUT_PCT,
              mode: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward, one row per CTA of ``tpr`` threads, plain loads, exact D.

    Args:
        x: ``[B, S, 3072]`` bf16; rows contiguous, batch stride may exceed ``S*3072``.
        scale: ``[B, 1, 3072]`` bf16, any batch stride.
        shift: ``[B, 1, 3072]`` bf16, same layout rules as ``scale``.
        tpr: Threads per row: 128, 192 or 384 (3, 2 or 1 vectors per thread).
        min_blocks: ``min_blocks_per_mp`` launch hint (register cap); ``None`` launches
            with no occupancy and no carveout attribute at all (``carveout`` must be None too).
        carveout: ``preferred_smem_carveout`` percent, or ``None`` with ``min_blocks=None``.
        mode: 0 = modulation loaded after the reductions; 1 = prefetched before them
            (``ld.global.nc``); 2 = prefetched with plain ``ld.global``.

    Returns:
        ``(out, mean, rstd)`` as :func:`fwd_ring`.

    Raises:
        ValueError: On a layout, dtype, alignment, index-range, ``tpr`` or knob violation.
    """
    B, S, xe, sce, she = _check_fwd(x, scale, shift)
    if tpr not in PLAIN_TPRS or mode not in (0, 1, 2):
        raise ValueError(f"fused_ln_adaln_a1: tpr must be one of {PLAIN_TPRS} and mode 0/1/2, got {tpr}/{mode}")
    if (min_blocks is None) != (carveout is None):
        raise ValueError("fused_ln_adaln_a1: min_blocks and carveout must both be None or both be set")
    dev = x.device.index
    n_rows = B * S
    out, mean, rstd = _alloc_fwd(x, n_rows)
    if DBG_FWD_PLAIN and mode == 1 and min_blocks is not None:
        dbg = torch.zeros(n_rows * DBG_WORDS, dtype=torch.int64, device=x.device)
        if torch.cuda.is_current_stream_capturing():
            _DBG_BUFS.append((B, S, dbg))
        _compile_fwd_plain_dbg(dev, tpr, min_blocks, carveout)(
            _flat(x, xe), _flat(scale, sce), _flat(shift, she), out.view(-1), mean, rstd, dbg,
            n_rows, S, x.stride(0), scale.stride(0), _stream(dev),
        )
        return out, mean, rstd
    _compile_fwd_plain(dev, tpr, min_blocks, carveout, mode)(
        _flat(x, xe), _flat(scale, sce), _flat(shift, she), out.view(-1), mean, rstd,
        n_rows, S, x.stride(0), scale.stride(0), _stream(dev),
    )
    return out, mean, rstd


_DBG_DUMMY: dict[int, torch.Tensor] = {}


def _dbg_buf(x, B, S, n_rows, mean):
    """The per-CTA stamp buffer when :data:`DBG_FWD_WARP` is on (recorded during a graph
    capture for :func:`dump_dbg`), else a cached 32-byte int64 placeholder operand (the
    kernel's ``dbg`` parameter is typed int64; it is never touched when ``DBG`` is off)."""
    if not DBG_FWD_WARP:
        dev = x.device.index
        if dev not in _DBG_DUMMY:
            _DBG_DUMMY[dev] = torch.zeros(DBG_WORDS, dtype=torch.int64, device=x.device)
        return _DBG_DUMMY[dev], False
    dbg = torch.zeros(n_rows * DBG_WORDS, dtype=torch.int64, device=x.device)
    if torch.cuda.is_current_stream_capturing():
        _DBG_BUFS.append((B, S, dbg))
    return dbg, True


WARP_MODES = (0, 1, 2, 3, 4, 5, 6, 7)
WARP_RPCS = (1, 2)
#: The per-pass conversion constants of the opaque modes (bf16 low half << 16; high half & 0xffff0000),
#: passed at runtime so the compilers cannot share one fp32 copy across passes.
_CVT_ARGS = (16, -65536, 16, -65536, 16, -65536)


def warp_grid(B: int, S: int, rpc: int) -> int:
    """CTAs the warp kernel launches: ``B * ceil(S / rpc)`` (rows of one CTA share a batch)."""
    return B * ((S + rpc - 1) // rpc)


def fwd_warp(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
             mode: int = 0, rpc: int = 1, minb: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward, one warp per row and ``rpc`` rows per 32-thread CTA; no shared memory, no
    barrier, no launch attribute (``pw32``; ``pw32s``;
    ``pw32d`` / ``pw32c`` / ``pw32b`` / ``pw32d2`` / ``pw32c2`` / ``pw32b2``;
    ``pw32c2r2`` / ``pw32b2r2``). Bitwise ``r2``'s numerics in every mode.

    Args:
        x: ``[B, S, 3072]`` bf16; rows contiguous, batch stride may exceed ``S*3072``.
        scale: ``[B, 1, 3072]`` bf16, any batch stride.
        shift: ``[B, 1, 3072]`` bf16, same layout rules as ``scale``.
        mode: 0 ``x`` fp32, no prefetch; 1 ``x`` fp32, ``scale``+``shift`` prefetched;
            2 ``x`` bf16-held, no prefetch; 3 ``x`` bf16-held, ``scale`` prefetched and ``shift``
            loaded after ``mean``; 4 ``x`` bf16-held, both prefetched; 5 / 6 / 7 as 2 / 3 / 4 with
            the opaque per-pass conversion (see the kernel docstring).
        rpc: Rows per CTA, 1 or 2 (grid ``B * ceil(S / rpc)``; see the kernel docstring).
        minb: 0 = no launch attribute (every arm but ``pw32c2r2p``); else ``min_blocks_per_mp`` (a ptxas
            register cap of ``65536 / (32 * minb)``).

    Returns:
        ``(out, mean, rstd)`` as :func:`fwd_ring`.

    Raises:
        ValueError: On a layout, dtype, alignment, index-range, ``mode`` or ``rpc`` violation.
    """
    B, S, xe, sce, she = _check_fwd(x, scale, shift)
    mode = int(mode)
    rpc = int(rpc)
    if mode not in WARP_MODES:
        raise ValueError(f"fused_ln_adaln_a1: warp mode must be one of {WARP_MODES}, got {mode}")
    if rpc not in WARP_RPCS:
        raise ValueError(f"fused_ln_adaln_a1: warp rpc must be one of {WARP_RPCS}, got {rpc}")
    minb = int(minb)
    if minb < 0 or minb > 32:
        raise ValueError(f"fused_ln_adaln_a1: warp minb must be 0..32, got {minb}")
    dev = x.device.index
    n_rows = B * S
    out, mean, rstd = _alloc_fwd(x, n_rows)
    dbg, on = _dbg_buf(x, B, S, n_rows, mean)
    _compile_fwd_warp(dev, mode, on, rpc, minb)(
        _flat(x, xe), _flat(scale, sce), _flat(shift, she), out.view(-1), mean, rstd, dbg,
        warp_grid(B, S, rpc), S, x.stride(0), scale.stride(0), *_CVT_ARGS, _stream(dev),
    )
    return out, mean, rstd


def fwd_warpx(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
              warps: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward, ``warps`` independent warps per CTA, one row each; no shared memory, no
    barrier, no launch attribute (``pw128x``). Bitwise ``r2``'s numerics.

    Args:
        x: ``[B, S, 3072]`` bf16; rows contiguous, batch stride may exceed ``S*3072``.
        scale: ``[B, 1, 3072]`` bf16, any batch stride.
        shift: ``[B, 1, 3072]`` bf16, same layout rules as ``scale``.
        warps: Warps (= rows) per CTA, 1..32.

    Returns:
        ``(out, mean, rstd)`` as :func:`fwd_ring`.

    Raises:
        ValueError: On a layout, dtype, alignment, index-range or ``warps`` violation.
    """
    B, S, xe, sce, she = _check_fwd(x, scale, shift)
    if warps < 1 or warps > 32:
        raise ValueError(f"fused_ln_adaln_a1: warps={warps} invalid")
    dev = x.device.index
    n_rows = B * S
    out, mean, rstd = _alloc_fwd(x, n_rows)
    dbg, on = _dbg_buf(x, B, S, n_rows, mean)
    _compile_fwd_warpx(dev, warps, on)(
        _flat(x, xe), _flat(scale, sce), _flat(shift, she), out.view(-1), mean, rstd, dbg,
        n_rows, math.ceil(n_rows / warps), S, x.stride(0), scale.stride(0), _stream(dev),
    )
    return out, mean, rstd


def dump_dbg(dirname: str) -> str | None:
    """Write every per-CTA stamp buffer captured into a CUDA graph.

    Returns the path written, or None when :data:`DBG_FWD_PLAIN` is off or nothing was
    captured. Buffers are listed in capture order, which is the launch order of the
    profiled replay (the k-th entry is the k-th ``ln_adaln_fwd_plain`` launch).
    """
    if not (DBG_FWD_PLAIN or DBG_FWD_WARP) or not _DBG_BUFS:
        return None
    torch.cuda.synchronize()
    path = os.path.join(dirname, "a1_dbg.pt")
    torch.save({"words": DBG_WORDS, "launches": [(B, S) for B, S, _ in _DBG_BUFS],
                "bufs": [t.cpu() for _, _, t in _DBG_BUFS]}, path)
    print(f":::A1DBG wrote {len(_DBG_BUFS)} launches to {path}", flush=True)
    return path


def bwd_plain(grad_out: torch.Tensor, x: torch.Tensor, scale: torch.Tensor,
              mean: torch.Tensor, rstd: torch.Tensor, S: int,
              residual: torch.Tensor | None = None, grad_extra: torch.Tensor | None = None,
              tpr: int = 128, min_blocks: int = 1, carveout: int = PLAIN_CARVEOUT_PCT,
              mode: int = 0) -> tuple[torch.Tensor, ...]:
    """Fused AdaLN bwd + LayerNorm bwd, one row per CTA, with optional residual add and grad add.

    Args:
        grad_out: ``[B, S, 3072]`` bf16 contiguous.
        x: ``[B, S, 3072]`` bf16, same layout rules as the forward.
        scale: ``[B, 1, 3072]`` bf16 view.
        mean: ``[B*S]`` fp32 from the forward.
        rstd: ``[B*S]`` fp32 from the forward.
        S: Sequence length.
        residual: Optional ``[B, S, 3072]`` bf16 added to ``grad_x`` in-kernel.
        grad_extra: Optional ``[B, S, 3072]`` bf16 added to ``grad_out`` first; when
            given, the bf16 sum is returned as a third tensor.
        tpr: Threads per row: 128, 192 or 384.
        min_blocks: ``min_blocks_per_mp`` launch hint (register cap).
        carveout: ``preferred_smem_carveout`` percent.
        mode: 0 residual after the barrier; 1 residual prefetched; 2 + ``1+scale``
            held; 3 stateless second pass (see the kernel docstring).

    Returns:
        ``(grad_x, grad_scale)`` or ``(grad_x, grad_scale, grad_sum)``, all
        ``[B, S, 3072]`` bf16 contiguous. ``grad_shift = grad_out`` is the caller's identity.

    Raises:
        ValueError: On a layout, dtype, alignment, index-range or ``tpr`` violation.
    """
    B = grad_out.shape[0]
    n_rows = B * S
    has_res = residual is not None
    has_ge = grad_extra is not None
    if tpr not in PLAIN_TPRS or mode not in (0, 1, 2, 3):
        raise ValueError(f"fused_ln_adaln_a1: tpr must be one of {PLAIN_TPRS} and mode 0..3, got {tpr}/{mode}")
    if not grad_out.is_contiguous():
        raise ValueError("fused_ln_adaln_a1: grad_out must be contiguous")
    _check_act(grad_out, "grad_out", S)
    xe = _check_act(x, "x", S)
    sce = _check_mod(scale, "scale", B)
    _check_stat(mean, "mean", n_rows)
    _check_stat(rstd, "rstd", n_rows)
    for name, t in (("residual", residual), ("grad_extra", grad_extra)):
        if t is not None:
            if not t.is_contiguous():
                raise ValueError(f"fused_ln_adaln_a1: {name} must be contiguous")
            _check_act(t, name, S)
    dev = x.device.index
    grad_x = torch.empty_like(grad_out)
    grad_scale = torch.empty_like(grad_out)
    grad_sum = torch.empty_like(grad_out) if has_ge else None
    dummy = grad_out.view(-1)
    _compile_bwd_plain(dev, tpr, has_res, has_ge, min_blocks, carveout, mode)(
        grad_out.view(-1), _flat(x, xe), _flat(scale, sce), mean, rstd,
        grad_x.view(-1), grad_scale.view(-1),
        residual.view(-1) if has_res else dummy,
        grad_extra.view(-1) if has_ge else dummy,
        grad_sum.view(-1) if has_ge else dummy,
        n_rows, S, x.stride(0), scale.stride(0), _stream(dev),
    )
    if has_ge:
        return grad_x, grad_scale, grad_sum
    return grad_x, grad_scale


# ---------------------------------------------------------------------------
# Arm plumbing (shared by the contract tests and the bench)
# ---------------------------------------------------------------------------

def fwd_arm(name: str, x, scale, shift):
    """Run forward arm ``name`` from :data:`FWD_ARMS`."""
    fam, *knobs = FWD_ARMS[name]
    if fam == "ring":
        return fwd_ring(x, scale, shift, *knobs)
    if fam == "ring2":
        return fwd_ring2(x, scale, shift, *knobs)
    if fam == "warp":
        return fwd_warp(x, scale, shift, *knobs)
    if fam == "warpx":
        return fwd_warpx(x, scale, shift, *knobs)
    return fwd_plain(x, scale, shift, *knobs)


def bwd_arm(name: str, grad_out, x, scale, mean, rstd, S, residual=None, grad_extra=None):
    """Run backward arm ``name`` from :data:`BWD_ARMS` (``grad_extra`` selects ``bwd_add``)."""
    return bwd_plain(grad_out, x, scale, mean, rstd, S, residual, grad_extra, *BWD_ARMS[name])


def fwd_arms_that_fit(device_index: int = 0) -> list[str]:
    """Forward arms whose shared memory the device admits."""
    out = []
    for name, (fam, *knobs) in FWD_ARMS.items():
        if fam in ("ring", "ring2") and smem_bytes_ring(knobs[0], knobs[1]) > device_smem_limit(device_index):
            continue
        out.append(name)
    return out


def fwd_kernel_name(name: str) -> str:
    fam, *knobs = FWD_ARMS[name]
    if fam == "plain" and DBG_FWD_PLAIN and knobs[3] == 1 and knobs[1] is not None:
        return KERNEL_FWD_PLAIN_DBG
    return {"ring": KERNEL_FWD_RING, "ring2": KERNEL_FWD_RING2, "plain": KERNEL_FWD_PLAIN,
            "warp": KERNEL_FWD_WARP, "warpx": KERNEL_FWD_WARPX}[fam]


# ---------------------------------------------------------------------------
# Dispatch rule (size expression; shapes are static per graph capture)
# ---------------------------------------------------------------------------

def ring_min_rows(device_index: int) -> int:
    """Smallest ``M = B*S`` the ring takes: the row count whose forward working set
    (``x`` + ``out``, ``4*D`` bytes per row) reaches :data:`RING_L2_FRACTION` of
    ``L2_cache_size`` -- the fitted value (6618 rows on GB300, 6452 on Rubin).
    ``FLAT_LN_ADALN_A1_MIN_ROWS`` overrides it for testing."""
    override = os.environ.get("FLAT_LN_ADALN_A1_MIN_ROWS")
    if override:
        return int(override)
    return math.ceil(RING_L2_FRACTION * _device_props(device_index).L2_cache_size / (4 * D))


def fwd_select(x: torch.Tensor) -> str:
    """The forward arm the dispatcher names for ``x`` (four tiers on ``M = B*S``)."""
    M = x.shape[0] * x.shape[1]
    if M <= SMALL_M_ROWS:
        return SELECT["fwd_tiny"] if M <= TINY_SPLIT_ROWS else SELECT["fwd_tiny2"]
    return SELECT["fwd_large"] if M >= ring_min_rows(x.device.index) else SELECT["fwd_small"]


def fused_ln_adaln_a1(x, scale, shift):
    """The composite forward as it would ship (``torch.ops.flux.fused_ln_adaln_a1``)."""
    return fwd_arm(fwd_select(x), x, scale, shift)


def fused_ln_adaln_bwd_a1(grad_out, x, scale, mean, rstd, S, residual=None):
    return bwd_arm(SELECT["bwd"], grad_out, x, scale, mean, rstd, S, residual, None)


def fused_ln_adaln_bwd_add_a1(grad_out, grad_extra, x, scale, mean, rstd, S, residual=None):
    return bwd_arm(SELECT["bwd_add"], grad_out, x, scale, mean, rstd, S, residual, grad_extra)


# ---------------------------------------------------------------------------
# torch.library registration -- new names, unwired
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define("fused_ln_adaln_a1(Tensor x, Tensor scale, Tensor shift) -> (Tensor, Tensor, Tensor)")
_lib.define(
    "fused_ln_adaln_bwd_a1(Tensor grad_out, Tensor x, Tensor scale,"
    " Tensor mean, Tensor rstd, int S, Tensor? residual=None) -> (Tensor, Tensor)"
)
_lib.define(
    "fused_ln_adaln_bwd_add_a1(Tensor grad_out, Tensor grad_extra,"
    " Tensor x, Tensor scale, Tensor mean, Tensor rstd, int S,"
    " Tensor? residual=None) -> (Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "fused_ln_adaln_a1", "CUDA")
def _fla_a1_impl(x, scale, shift):
    return fused_ln_adaln_a1(x, scale, shift)


@torch.library.impl(_lib, "fused_ln_adaln_bwd_a1", "CUDA")
def _flab_a1_impl(grad_out, x, scale, mean, rstd, S, residual=None):
    return fused_ln_adaln_bwd_a1(grad_out, x, scale, mean, rstd, S, residual)


@torch.library.impl(_lib, "fused_ln_adaln_bwd_add_a1", "CUDA")
def _flaba_a1_impl(grad_out, grad_extra, x, scale, mean, rstd, S, residual=None):
    return fused_ln_adaln_bwd_add_a1(grad_out, grad_extra, x, scale, mean, rstd, S, residual)


# ---------------------------------------------------------------------------
# The PRODUCTION ops -- the names ``flat/model.py`` calls
# ---------------------------------------------------------------------------
#
# These three defines and their implementations lived in ``fused_ln_adaln.py``
# (Triton), which owned the forward at
# ``M <= 2048`` and delegated everything else here. With the mb-4 forward now
# served by ``pw32c2`` / ``pw32c2r2p`` (see ``SELECT`` above), that module had no
# live caller and was deleted -- the last non-CUTLASS kernel in ``flat/``.
#
# Every arm reached from here is CUTLASS DSL and bitwise ``r2``. The forward
# tiers on ``fwd_select``; the backward and backward-with-add have been the DSL
# ``pL128`` / ``pL192`` arms and are unchanged.

_lib.define(
    "fused_ln_adaln(Tensor x, Tensor scale, Tensor shift)"
    " -> (Tensor, Tensor, Tensor)"
)
_lib.define(
    "fused_ln_adaln_bwd(Tensor grad_out, Tensor x, Tensor scale,"
    " Tensor mean, Tensor rstd, int S, Tensor? residual=None)"
    " -> (Tensor, Tensor)"
)
_lib.define(
    "fused_ln_adaln_bwd_add(Tensor grad_out, Tensor grad_extra,"
    " Tensor x, Tensor scale, Tensor mean, Tensor rstd, int S,"
    " Tensor? residual=None) -> (Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "fused_ln_adaln", "CUDA")
def _fla_impl(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fused_ln_adaln_a1(x, scale, shift)


@torch.library.impl(_lib, "fused_ln_adaln_bwd", "CUDA")
def _flab_impl(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    S: int,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_ln_adaln_bwd_a1(grad_out, x, scale, mean, rstd, S, residual)


@torch.library.impl(_lib, "fused_ln_adaln_bwd_add", "CUDA")
def _flaba_impl(
    grad_out: torch.Tensor,
    grad_extra: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    S: int,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fused_ln_adaln_bwd_add_a1(grad_out, grad_extra, x, scale, mean, rstd, S, residual)
