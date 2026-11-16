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

"""``torch.ops.flux.gate_bwd_*_ctm`` -- the fused gate backward + NVFP4 amax + dbias producer on the CUTLASS DSL.

A 1:1 structural port of the shipped
variant of the hand-written CUDA kernel it replaced (``gate_bwd_amax_kernel.cu``,
mode ``oneshot+hint1+memset+dbpart`` with the ``fin3v4`` finaliser deferred on a
forked stream; deleted once this module was proven
on both architectures). This is the
only implementation of the op:

* ``out0 = bf16(grad * gate)`` (``grad_proj``), ``out1 = bf16(grad * proj)``
  (``grad_gate``): products exact in fp32, one ``cvt.rn.bf16x2.f32`` -- the
  same bits as the plain DSL ``flux.fused_gate_bwd`` (``gate_residual_a2.gate_bwd_a2``)
  and the closed form ``tests/gate_reference.gate_bwd_ref``;
* ``amax[0] = max|out0|`` (``max.xorsign.abs.bf16x2``), ``amax[1]`` the post-RHT
  columnwise amax over 16-row groups (smem tile, ``ldmatrix.x4.trans`` +
  ``mma.m16n8k16`` against ``R``, fp32 max, one ``rn`` to bf16);
* ``dbias`` = fp32 column sum of the emitted ``out0`` -- a third ``mma`` against
  ones per task, one fp32 partial row per 16-row group, finalised in a **fixed
  order** by :func:`_finalize` (24 blocks x 1024 threads at N = 3072, lane =
  four columns, warp ``w`` sums slots ``w, w+32, ...`` eight loads at a time,
  then warp 0 sums the 32 warp partials ascending) -- the same order as the
  CUDA ``gate_dbias_finalize_kernel<float, 4>``, so it is bitwise that kernel;
* the amax completion is an 8 B memset node on the stream
  and ``red.relaxed.gpu.global.max.u32`` from thread 0 of every CTA;
* L2 policy exactly as shipped: ``createpolicy … evict_first`` on the ``grad``
  and ``proj`` loads and on the ``out1`` stores (the hints buy
  the following cast ~12 us), plain stores on ``out0``.

Structure (one task per CTA): a CTA task is 16 rows x
512 columns; warp ``w`` owns rows ``2w, 2w+1`` across the 512 columns, one 16 B
vector per lane per row-half, four slabs per task, so every load instruction is
512 contiguous bytes of one row. ``grid = (M/16) x (N/512)``.

The deferred op (:func:`gate_bwd_dbias_amax_deferred_ctm`) writes the partials
into one persistent buffer per (device, slots, N) and launches the finaliser on
one persistent per-device side stream after a ``wait_stream``; the caller casts
and then calls :func:`join_dbias_ctm` before consuming ``dbias`` and before its
next call. Under capture the fork and join are graph edges.

Contract (proved in
``tests/test_gate_bwd_amax_ctm.py`` -- a rewrite inherits nothing): ``out0``/``out1``
bitwise the closed form and the plain DSL op; both amaxes bitwise TE's ``HadamardAmaxTmaKernel``
pair; ``dbias`` within ``aten.sum`` tolerance, ``torch.equal`` across
launches, graph replays and the 48-replay pressure test, and bitwise a torch
reference that sums the kernel's own partials in the finaliser's fixed order;
the deferred op bitwise the serial one; capture-safe (no host sync, no
allocation outside the graph pool; the side stream and partial buffers are
created on the first eager call).

The exact instructions that carry the bitwise contract are emitted verbatim
through ``inline_ptx`` / ``llvm.inline_asm``; everything else is the DSL's own
lowering. Public ``nvidia-cutlass-dsl`` API only (``cute.arch.*``), so the
module runs on 4.6.2 as well as the container's 4.8.0.

Restrictions: contiguous BF16 ``[B, S, D]`` ``grad``/``proj``
with ``S % 16 == 0`` and ``D % 512 == 0``; ``gate`` a ``[B, 1, D]`` BF16 view
with ``stride(2) == 1`` and a 16 B-aligned row stride; ``M x N < 2**31``.

Launch variants (:data:`LAUNCH_VARIANTS`): ``P`` the
port (``min_blocks_per_mp=2``, the CUDA kernel's ``__launch_bounds__(256, 2)``),
``P3`` / ``P4`` the occupancy levers, ``NH`` the no-hint diagnostic. Every one
passes ``preferred_smem_carveout=100`` explicitly.

``BL3``: the same body with the eight
``rht`` B-fragment reads moved from the top of the task to after the slab loop
(``BFRAG_LATE``). Nothing reads them before the barrier, and holding their four
packed registers through the slab loop is what put ``P`` at 86 registers on
``sm_107a`` -- one CTA/SM fewer than the CUDA kernel's 80 -- while ``P3``'s
forced 80 split the ten-load burst behind the first stores. ``BL3`` compiles to
72 registers on both targets with the burst intact (SASS: 10 x ``LDG.128``, 8 x
``STG``, then the ``rht`` loads under the stores). Measured against the CUDA op
on two nodes per cluster: GB300 ``[16384]`` -4.2 us/launch, ``[8192]`` -1.6
(-0.29..-0.30 ms/iter at B = 32); Rubin ``[16384]`` +0.2..+0.35 and ``[8192]``
+0.5..+0.6 (from ``P``'s +2.0 / +2.0), every per-shape median inside the 2.05 us
quantum. :func:`select_variant` routes by (device, M): ``BL3`` everywhere on
GB300 and at ``M > 2048`` on Rubin, ``P`` at the Rubin mb-4 shapes where
``BL3`` reads +0.2..+0.5 us and ``P`` +0.1 (both nodes, both instruments).
"""

from __future__ import annotations

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T

I32 = cutlass.Int32
I64 = cutlass.Int64
I16 = cutlass.Int16
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

WARPS = 8
THREADS = 32 * WARPS
WARP_COLS = 64
CTA_COLS = WARPS * WARP_COLS          # 512
GROUP_ROWS = 16
PITCH_BYTES = 1040                    # 1024 B of bf16 + 16 B pad: conflict-free ldmatrix
TILE_BYTES = GROUP_ROWS * PITCH_BYTES  # 16640, one buffer (one task per CTA)
#: Finaliser geometry (the CUDA ``fin3v4``): 32 warps x 32 lanes, four columns
#: per lane, eight loads in flight per round.
FIN_WARPS = 32
FIN_THREADS = 32 * FIN_WARPS
FIN_V = 4
FIN_COLS = 32 * FIN_V                 # 128 columns per block
FIN_ROUND = 8
SMEM_CARVEOUT_PCT = 100
#: name -> (min_blocks_per_mp, preferred_smem_carveout, L2 hints on, volatile hinted loads,
#: rht B fragments read after the slab loop).
#: ``P3nv``: the eight hinted loads are emitted as non-volatile ``inline_asm``
#: (``has_side_effects=False``), so ptxas may reorder them; every store and the
#: ``createpolicy`` stay volatile. Same bits by construction (a load has no side effect).
#: ``BL3``: ``BFRAG_LATE`` at 3 CTAs/SM -- 72 registers, load burst intact.
LAUNCH_VARIANTS: dict[str, tuple[int, int, bool, bool, bool]] = {
    "P": (2, SMEM_CARVEOUT_PCT, True, True, False),
    "P3": (3, SMEM_CARVEOUT_PCT, True, True, False),
    "P4": (4, SMEM_CARVEOUT_PCT, True, True, False),
    "NH": (2, SMEM_CARVEOUT_PCT, False, True, False),
    "P3nv": (3, SMEM_CARVEOUT_PCT, True, False, False),
    "BL3": (3, SMEM_CARVEOUT_PCT, True, True, True),
}
#: Rubin rows at or below this take ``P`` (``BL3`` +0.2..+0.5 us at the mb-4
#: shapes on two nodes, ``P`` +0.1); above it, and on GB300 at every M, ``BL3``.
RUBIN_SMALL_M = 2048


def select_variant(m: int, cc: tuple[int, int] | None = None) -> str:
    """The production launch variant for ``M = B * S`` rows on a device of capability ``cc``.

    Args:
        m: Row count of the call.
        cc: ``(major, minor)``; the current device's when omitted.

    Returns:
        ``"P"`` on Rubin (``(10, 7)``) at ``m <= RUBIN_SMALL_M``, else ``"BL3"``.
    """
    if cc is None:
        cc = tuple(torch.cuda.get_device_capability())
    if tuple(cc) == (10, 7) and m <= RUBIN_SMALL_M:
        return "P"
    return "BL3"
_ONES_BF16X2 = 0x3F803F80             # bf16x2 {1.0, 1.0}
_BF16_HI_MASK = -65536                # 0xFFFF0000 as a signed Int32
_INT32_MAX = 2**31 - 1

_CVT_BF16X2 = "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};"
_MAX_XORSIGN_ABS_ASM = "max.xorsign.abs.bf16x2 $0, $1, $2;"
_MAX_XORSIGN_ABS_CONSTRAINTS = "=r,r,r"
_LDMATRIX_X4_TRANS = (
    "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
    "{ {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];"
)
# Raw llvm.inline_asm with explicit register constraints where inline_ptx would
# let LLVM fold a constant operand into an immediate ptxas rejects
# or where the instruction has no result.
_MMA_16816_BF16 = (
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
)
_MMA_CONSTRAINTS = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
# L2 eviction-priority policy (sm_80+): a 64-bit register operand on
# ld/st.L2::cache_hint. The fraction is spelled inside the asm.
_POLICY_FIRST_ASM = "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;"
_LD16_HINT_ASM = "ld.global.nc.L2::cache_hint.v4.b32 {$0, $1, $2, $3}, [$4], $5;"
_LD16_HINT_CONSTRAINTS = "=r,=r,=r,=r,l,l"
_ST16_HINT_ASM = "st.global.L2::cache_hint.v4.b32 [$0], {$1, $2, $3, $4}, $5;"
_ST16_HINT_CONSTRAINTS = "l,r,r,r,r,l"
_RED_MAX_U32_ASM = "red.relaxed.gpu.global.max.u32 [$0], $1;"
_RED_MAX_U32_CONSTRAINTS = "l,r"
_LD_CG_V4_F32_ASM = "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];"
_LD_CG_V4_F32_CONSTRAINTS = "=f,=f,=f,=f,l"

#: ``__syncwarp``: ``cute.arch.sync_warp`` where the installed DSL has it, else
#: the CTA barrier (correct, one extra ``bar.sync`` per task). Which one ran is
#: printed by :func:`describe`.
_SYNC_WARP = getattr(cute.arch, "sync_warp", None)


# ---------------------------------------------------------------------------
# Device helpers (traced; plain Python loops only)
# ---------------------------------------------------------------------------


def _asm(res, args, code, cons, side_effects: bool):
    return llvm.inline_asm(
        res, [a.ir_value() for a in args], code, cons,
        has_side_effects=side_effects, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _bf16_lo(w):
    return (w << 16).bitcast(F32)


def _bf16_hi(w):
    return (w & I32(_BF16_HI_MASK)).bitcast(F32)


def _pack_bf16x2_rn(lo, hi):
    """``cvt.rn.bf16x2.f32 d, hi, lo``: the correctly rounded pair, hi in the upper half."""
    return cute.arch.inline_ptx(
        _CVT_BF16X2, write_only_types=[I32], read_only_args=[hi, lo],
    )


def _max_xorsign_abs(a, b):
    """``max.xorsign.abs.bf16x2`` with BOTH operands forced into registers.

    In this one-task kernel the body is fully unrolled, so the first call sees
    the literal ``0`` seed; ``cute.arch.inline_ptx`` lets LLVM fold it into an
    immediate (``max.xorsign.abs.bf16x2 %r, 0, %r``) and ptxas rejects the
    module with "Arguments mismatch for instruction 'max'". A persistent loop
    would make the seed a loop-carried value and avoid it.
    """
    return I32(_asm(T.i32(), [a, b], _MAX_XORSIGN_ABS_ASM, _MAX_XORSIGN_ABS_CONSTRAINTS, False))


def _ldmatrix_x4_trans(addr):
    return cute.arch.inline_ptx(
        _LDMATRIX_X4_TRANS, write_only_types=[I32] * 4, read_only_args=[addr],
    )


def _mma_16816(a, b0, b1, c):
    """D = A(16x16 bf16, row) * B(16x8 bf16, col) + C, fp32 accumulate."""
    res = _asm(llvm.StructType.get_literal([T.f32()] * 4),
               [a[0], a[1], a[2], a[3], b0, b1, c[0], c[1], c[2], c[3]],
               _MMA_16816_BF16, _MMA_CONSTRAINTS, True)
    return tuple(F32(llvm.extractvalue(T.f32(), res, [i])) for i in range(4))


def _policy_first():
    """One ``createpolicy … evict_first`` per CTA (the CUDA kernel: once per thread)."""
    return I64(_asm(T.i64(), [], _POLICY_FIRST_ASM, "=l", True))


def _ld16_hint(ptr, pol, volatile: bool = True):
    """16 B ``ld.global.nc.L2::cache_hint`` -> four Int32 words (same bits as ``.load``).

    ``volatile=False`` (variant ``P3nv``) lets LLVM/ptxas schedule the load freely.
    """
    res = _asm(llvm.StructType.get_literal([T.i32()] * 4), [ptr.toint(), pol],
               _LD16_HINT_ASM, _LD16_HINT_CONSTRAINTS, volatile)
    return tuple(I32(llvm.extractvalue(T.i32(), res, [i])) for i in range(4))


def _ld16_nc(ptr):
    """16 B ``ld.global.nc`` (no policy) -> four Int32 words."""
    v = ptr.load(16, count=4, is_invariant=True)
    return (v[0], v[1], v[2], v[3])


def _st16_hint(ptr, words, pol):
    """16 B ``st.global.L2::cache_hint`` of four Int32 words."""
    _asm(None, [ptr.toint(), words[0], words[1], words[2], words[3], pol],
         _ST16_HINT_ASM, _ST16_HINT_CONSTRAINTS, True)


def _red_max_u32(addr, bits):
    """``red.relaxed.gpu.global.max.u32`` -- the fp32 max of non-negative values on their bits."""
    _asm(None, [addr, bits], _RED_MAX_U32_ASM, _RED_MAX_U32_CONSTRAINTS, True)


def _ld_cg_v4(addr):
    """``__ldcg`` of a float4: the L2-coherent read of another kernel's partials."""
    res = _asm(llvm.StructType.get_literal([T.f32()] * 4), [addr],
               _LD_CG_V4_F32_ASM, _LD_CG_V4_F32_CONSTRAINTS, True)
    return tuple(F32(llvm.extractvalue(T.f32(), res, [i])) for i in range(4))


def _round_bf16(x):
    return x.to(BF16).to(F32)


def _warp_max(v):
    for off in (16, 8, 4, 2, 1):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, offset=off))
    return v


def _sync_warp():
    if _SYNC_WARP is not None:
        _SYNC_WARP()
    else:
        cute.arch.sync_threads()


# ---------------------------------------------------------------------------
# The producer kernel
# ---------------------------------------------------------------------------


@cute.kernel
def gate_bwd_amax_ctm_kernel(
    grad: cute.Tensor, proj: cute.Tensor, gate: cute.Tensor,
    out0: cute.Tensor, out1: cute.Tensor,
    amax: cute.Tensor, group_amax: cute.Tensor, ws_dbias: cute.Tensor, rht: cute.Tensor,
    n_w: cutlass.Int32, n_cb: cutlass.Int32, gate_stride_w: cutlass.Int32,
    groups_per_sample: cutlass.Int32,
    DBIAS: cutlass.Constexpr[bool], GROUP: cutlass.Constexpr[bool],
    HINT: cutlass.Constexpr[bool], VOL: cutlass.Constexpr[bool],
    BFRAG_LATE: cutlass.Constexpr[bool],
) -> None:
    """One 16 x 512 task per CTA.

    ``grad``/``proj``/``out0``/``out1`` are the bf16 ``[M, N]`` tensors viewed as
    flat Int32 words (bf16 pairs); ``gate`` the ``[B, 1, N]`` view's storage as
    flat Int32 words with ``gate_stride_w`` words per sample; ``n_w = N / 2``;
    ``n_cb = N / 512``; ``groups_per_sample = S / 16``. ``amax`` is fp32 ``[2]``
    memset to zero on the stream; ``ws_dbias`` fp32 ``[M/16][N]`` (``DBIAS``);
    ``group_amax`` a zeroed fp32 ``[M/16]`` (``GROUP``); ``rht`` the bf16
    ``[16, 16]`` basis viewed as Int16.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    gi = lane >> 2          # mma "groupID"
    tq = lane & 3           # mma "threadID_in_group"
    rgi = bidx // n_cb      # row group (one-shot, row-group-major)
    cb = bidx - rgi * n_cb  # column block

    gp = cutlass.Pointer(grad.iterator, dtype=I32)
    pp = cutlass.Pointer(proj.iterator, dtype=I32)
    gtp = cutlass.Pointer(gate.iterator, dtype=I32)
    o0p = cutlass.Pointer(out0.iterator, dtype=I32)
    o1p = cutlass.Pointer(out1.iterator, dtype=I32)
    wsd = cutlass.Pointer(ws_dbias.iterator, dtype=F32)
    ap = cutlass.Pointer(amax.iterator, dtype=F32)

    tile = cute.arch.alloc_smem(I32, TILE_BYTES // 4, 16)
    tile_p = cutlass.Pointer(tile, dtype=I32)
    tile_base = tile.toint()                     # Int32 shared-window address
    stage = cutlass.Pointer(cute.arch.alloc_smem(F32, 2 * WARPS, 16), dtype=F32)
    # dbias: each warp's 64 column sums of the task, staged so that 16 lanes each
    # pick up one contiguous float4 (one 16 B store per 4 columns).
    dsum = cutlass.Pointer(cute.arch.alloc_smem(F32, WARPS * WARP_COLS, 16), dtype=F32)

    pol = I64(0)
    if cutlass.const_expr(HINT):
        pol = _policy_first()

    # B fragments of R: lane holds k = 2tq, 2tq+1 (+8) at n = gi (+8 jb).
    def read_bfrag():
        out = []
        for jb in cutlass.range_constexpr(2):
            nn = jb * 8 + gi
            r0 = rht[(2 * tq) * 16 + nn].to(I32) & 0xFFFF
            r1 = rht[(2 * tq + 1) * 16 + nn].to(I32) & 0xFFFF
            r8 = rht[(2 * tq + 8) * 16 + nn].to(I32) & 0xFFFF
            r9 = rht[(2 * tq + 9) * 16 + nn].to(I32) & 0xFFFF
            out.append((r0 | (r1 << 16), r8 | (r9 << 16)))
        return out

    bfrag = []
    if cutlass.const_expr(not BFRAG_LATE):
        bfrag = read_bfrag()
    ones = I32(_ONES_BF16X2)

    # ldmatrix.x4 lane address inside this warp's 16 x 64 slice of the CTA tile:
    # matrix q = lane/8 supplies row (q>>1)*8 + lane%8 at column offset (q&1)*8.
    ldq = lane >> 3
    ld_off = (warp * (WARP_COLS * 2) + ((ldq >> 1) * 8 + (lane & 7)) * PITCH_BYTES
              + (ldq & 1) * 16)
    # st.shared word offset of this lane's vector for slab 0 (stream mapping:
    # row 2w + (s>>1), column vector lane + 32(s&1)).
    st_word0 = ((2 * warp) * PITCH_BYTES + lane * 16) // 4

    # S % 16 == 0, so a 16-row group lies inside one sample: one gate row per
    # task, two 16 B vectors per lane (columns lane*8 and (lane+32)*8).
    sample = rgi // groups_per_sample
    gbase = sample * gate_stride_w + cb * (CTA_COLS // 2) + lane * 4
    gv = [_ld16_nc(gtp + gbase), _ld16_nc(gtp + gbase + (CTA_COLS // 4))]

    # The task's eight input loads, all issued before any compute.
    xs = []
    ys = []
    idxs = []
    for s in cutlass.range_constexpr(4):
        row = rgi * GROUP_ROWS + 2 * warp + (s >> 1)
        idx = row * n_w + cb * (CTA_COLS // 2) + (lane + 32 * (s & 1)) * 4
        idxs.append(idx)
        if cutlass.const_expr(HINT):
            xs.append(_ld16_hint(gp + idx, pol, VOL))
            ys.append(_ld16_hint(pp + idx, pol, VOL))
        else:
            xs.append(_ld16_nc(gp + idx))
            ys.append(_ld16_nc(pp + idx))

    # ---- products, pack, rowwise amax, stores (global + smem tile) ----
    rmax_packed = I32(0)                         # rowwise amax, packed bf16x2 (TE idiom)
    for s in cutlass.range_constexpr(4):
        g = gv[s & 1]
        o0 = []
        o1 = []
        for k in cutlass.range_constexpr(4):
            x = xs[s][k]
            y = ys[s][k]
            gk = g[k]
            # A bf16 x bf16 product is exact in fp32; one RN to bf16 is the
            # correctly rounded product (the closed form tests/gate_reference.gate_bwd_ref).
            o0.append(_pack_bf16x2_rn(_bf16_lo(x) * _bf16_lo(gk), _bf16_hi(x) * _bf16_hi(gk)))
            o1.append(_pack_bf16x2_rn(_bf16_lo(x) * _bf16_lo(y), _bf16_hi(x) * _bf16_hi(y)))
            rmax_packed = _max_xorsign_abs(rmax_packed, o0[k])
        v0 = cutlass.Vector.from_elements((o0[0], o0[1], o0[2], o0[3]), I32)
        (o0p + idxs[s]).store(v0, alignment=16)
        if cutlass.const_expr(HINT):
            _st16_hint(o1p + idxs[s], o1, pol)
        else:
            v1 = cutlass.Vector.from_elements((o1[0], o1[1], o1[2], o1[3]), I32)
            (o1p + idxs[s]).store(v1, alignment=16)
        sw = st_word0 + ((s >> 1) * PITCH_BYTES + (s & 1) * 512) // 4
        (tile_p + sw).store(v0, alignment=16)

    if cutlass.const_expr(BFRAG_LATE):
        # Plain loads cannot be hoisted above the preceding stores, so these are issued
        # here: their latency hides under the stores and the barrier, and their four
        # registers are not live through the slab loop (86 -> 72 registers).
        bfrag = read_bfrag()

    # ---- the 16-row tile was written by all eight warps: RHT + dbias epilogue ----
    cute.arch.sync_threads()
    zero4 = (F32(0.0),) * 4
    tmax = F32(0.0)
    for blk in cutlass.range_constexpr(4):
        a = _ldmatrix_x4_trans(tile_base + ld_off + blk * 32)
        d0 = _mma_16816(a, bfrag[0][0], bfrag[0][1], zero4)
        d1 = _mma_16816(a, bfrag[1][0], bfrag[1][1], zero4)
        for j in cutlass.range_constexpr(4):
            tmax = cute.arch.fmax(tmax, cute.math.abs(d0[j]))
            tmax = cute.arch.fmax(tmax, cute.math.abs(d1[j]))
        if cutlass.const_expr(DBIAS):
            # A = tile^T (m = column, k = row), so A x ones = the 16-row column
            # sums, fp32 accumulate; lane (gi, tq) holds column gi in ds[0] and
            # gi + 8 in ds[2] (ds[1], ds[3] are the same values at n = 2tq + 1).
            ds = _mma_16816(a, ones, ones, zero4)
            if tq == 0:
                (dsum + warp * WARP_COLS + blk * 16 + gi).store(ds[0])
                (dsum + warp * WARP_COLS + blk * 16 + gi + 8).store(ds[2])
    if cutlass.const_expr(DBIAS):
        _sync_warp()
        if lane < 16:
            base = dsum + warp * WARP_COLS + lane * 4
            v = cutlass.Vector.from_elements(
                (base.load(), (base + 1).load(), (base + 2).load(), (base + 3).load()), F32)
            col = cb * CTA_COLS + warp * WARP_COLS + lane * 4
            (wsd + rgi * (2 * n_w) + col).store(v, alignment=16)
    if cutlass.const_expr(GROUP):
        wmax = _warp_max(tmax)
        if lane == 0:
            gap = cutlass.Pointer(group_amax.iterator, dtype=F32)
            cute.arch.atomic_fmax(gap + rgi, _round_bf16(wmax), sign_bit=False)

    # ---- amax: warp -> CTA -> red.max into the memset output ----
    r = _warp_max(cute.arch.fmax(cute.math.abs(_bf16_lo(rmax_packed)),
                                 cute.math.abs(_bf16_hi(rmax_packed))))
    c = _warp_max(tmax)
    if lane == 0:
        (stage + 2 * warp).store(r)
        (stage + 2 * warp + 1).store(c)
    cute.arch.sync_threads()
    if tidx == 0:
        r = F32(0.0)
        c = F32(0.0)
        for w in cutlass.range_constexpr(WARPS):
            r = cute.arch.fmax(r, (stage + 2 * w).load())
            c = cute.arch.fmax(c, (stage + 2 * w + 1).load())
        # The output was memset to zero on the stream. Rounding before the max is
        # exact: rn_bf16 is monotone, so rn(max_i c_i) == max_i rn(c_i).
        _red_max_u32(ap.toint(), r.bitcast(I32))
        _red_max_u32((ap + 1).toint(), _round_bf16(c).bitcast(I32))


# ---------------------------------------------------------------------------
# The finaliser kernel (fin3v4): dbias[col] = sum over slots of ws[slot][col]
# ---------------------------------------------------------------------------


@cute.kernel
def gate_dbias_finalize_ctm_kernel(
    ws: cute.Tensor, dbias: cute.Tensor, slots: cutlass.Int32, n: cutlass.Int32,
    OUT_BF16: cutlass.Constexpr[bool],
) -> None:
    """``F2`` (default). Grid ``N / 128``, 1024 threads. Lane ``l`` owns columns
    ``128 b + 4 l .. + 3``; warp ``w`` sums slots ``w, w + 32, ...`` ascending,
    eight loads in flight per round, then warp 0 sums the 32 warp partials
    ascending -- the CUDA ``gate_dbias_finalize_kernel<float, 4>`` order, so
    bitwise run-to-run and bitwise that kernel.

    Addressing (an ncu finding): ``F1`` recomputed a
    64-bit ``(s + 32 j) * n + col`` per load and executed 1.7x the CUDA
    finaliser's instructions (0.341 vs 0.201 M at ``[16384]``). Here one byte
    address is carried through the loop and each of the eight loads adds the
    hoisted row-block stride ``32 * n * 4`` -- the strength reduction nvcc
    applies to the CUDA source. Same loads, same order, same bits.

    ``OUT_BF16``: the same fp32 sum rounded once to bf16
    (``cvt.rn.bf16x2.f32``, aten's RNE) and stored as two 4 B words straight into
    the bias's ``param.grad`` view -- bitwise the ``copy_`` that used to follow.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    red = cutlass.Pointer(cute.arch.alloc_smem(F32, FIN_WARPS * FIN_COLS, 16), dtype=F32)
    wp = cutlass.Pointer(ws.iterator, dtype=F32)
    dp = cutlass.Pointer(dbias.iterator, dtype=F32)
    dw = cutlass.Pointer(dbias.iterator, dtype=I32)
    col = bidx * FIN_COLS + lane * FIN_V

    # Byte address of this thread's first partial (slot = warp) and the strides:
    # one row block of 32 slots, and one round of 8 row blocks.
    addr = (wp + (warp * n + col)).toint()
    stride_b = (n * (FIN_WARPS * 4)).to(I64)
    round_b = stride_b * FIN_ROUND
    acc = [F32(0.0) for _ in range(FIN_V)]
    n_full = (slots + 31 - warp) // (FIN_ROUND * FIN_WARPS)
    for i in cutlass.range(n_full):
        a = addr
        vs = []
        for j in cutlass.range_constexpr(FIN_ROUND):
            vs.append(_ld_cg_v4(a))
            a = a + stride_b
        new_acc = []
        for k in cutlass.range_constexpr(FIN_V):
            t = acc[k]
            for j in cutlass.range_constexpr(FIN_ROUND):
                t = t + vs[j][k]
            new_acc.append(t)
        acc = new_acc
        addr = addr + round_b
    # Tail: fewer than eight slots left on this warp's stream, ascending.
    for s in cutlass.range(warp + n_full * (FIN_ROUND * FIN_WARPS), slots, FIN_WARPS):
        v = _ld_cg_v4(addr)
        new_acc = []
        for k in cutlass.range_constexpr(FIN_V):
            new_acc.append(acc[k] + v[k])
        acc = new_acc
        addr = addr + stride_b
    for k in cutlass.range_constexpr(FIN_V):
        (red + warp * FIN_COLS + lane * FIN_V + k).store(acc[k])
    cute.arch.sync_threads()
    if warp == 0:
        tot = []
        for k in cutlass.range_constexpr(FIN_V):
            t = F32(0.0)
            for w in cutlass.range_constexpr(FIN_WARPS):
                t = t + (red + w * FIN_COLS + lane * FIN_V + k).load()
            tot.append(t)
            if cutlass.const_expr(not OUT_BF16):
                (dp + col + k).store(t)
        if cutlass.const_expr(OUT_BF16):
            (dw + col // 2).store(_pack_bf16x2_rn(tot[0], tot[1]))
            (dw + col // 2 + 1).store(_pack_bf16x2_rn(tot[2], tot[3]))


@cute.kernel
def gate_dbias_finalize_ctm_v1_kernel(
    ws: cute.Tensor, dbias: cute.Tensor, slots: cutlass.Int32, n: cutlass.Int32,
) -> None:
    """``F1``: the first-round finaliser, kept as the attribution control (it read 1.7x the CUDA finaliser's instructions).
    Same order as ``F2``, so bitwise ``F2`` and the CUDA kernel."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    red = cutlass.Pointer(cute.arch.alloc_smem(F32, FIN_WARPS * FIN_COLS, 16), dtype=F32)
    wp = cutlass.Pointer(ws.iterator, dtype=F32)
    dp = cutlass.Pointer(dbias.iterator, dtype=F32)
    col = bidx * FIN_COLS + lane * FIN_V

    acc = [F32(0.0) for _ in range(FIN_V)]
    # Full rounds: the CUDA loop `for (; s + 7*32 < slots; s += 256)` runs
    # ceil((slots - 224 - warp) / 256) times when positive, else 0 -- which is
    # (slots + 31 - warp) // 256 for every slots >= 1, warp <= 31.
    n_full = (slots + 31 - warp) // (FIN_ROUND * FIN_WARPS)
    for i in cutlass.range(n_full):
        s = warp + i * (FIN_ROUND * FIN_WARPS)
        vs = []
        for j in cutlass.range_constexpr(FIN_ROUND):
            vs.append(_ld_cg_v4((wp + (s + j * FIN_WARPS) * n + col).toint()))
        new_acc = []
        for k in cutlass.range_constexpr(FIN_V):
            a = acc[k]
            for j in cutlass.range_constexpr(FIN_ROUND):
                a = a + vs[j][k]
            new_acc.append(a)
        acc = new_acc
    # Tail: fewer than eight slots left on this warp's stream, ascending.
    for s in cutlass.range(warp + n_full * (FIN_ROUND * FIN_WARPS), slots, FIN_WARPS):
        v = _ld_cg_v4((wp + s * n + col).toint())
        new_acc = []
        for k in cutlass.range_constexpr(FIN_V):
            new_acc.append(acc[k] + v[k])
        acc = new_acc
    for k in cutlass.range_constexpr(FIN_V):
        (red + warp * FIN_COLS + lane * FIN_V + k).store(acc[k])
    cute.arch.sync_threads()
    if warp == 0:
        for k in cutlass.range_constexpr(FIN_V):
            t = F32(0.0)
            for w in cutlass.range_constexpr(FIN_WARPS):
                t = t + (red + w * FIN_COLS + lane * FIN_V + k).load()
            (dp + col + k).store(t)


# ---------------------------------------------------------------------------
# Host-side jit wrappers
# ---------------------------------------------------------------------------


@cute.jit
def _producer_host(
    grad: cute.Tensor, proj: cute.Tensor, gate: cute.Tensor,
    out0: cute.Tensor, out1: cute.Tensor,
    amax: cute.Tensor, group_amax: cute.Tensor, ws_dbias: cute.Tensor, rht: cute.Tensor,
    n_w: cutlass.Int32, n_cb: cutlass.Int32, gate_stride_w: cutlass.Int32,
    groups_per_sample: cutlass.Int32, grid_x: cutlass.Int32, stream: cuda.CUstream,
    DBIAS: cutlass.Constexpr[bool], GROUP: cutlass.Constexpr[bool],
    HINT: cutlass.Constexpr[bool], VOL: cutlass.Constexpr[bool],
    BFRAG_LATE: cutlass.Constexpr[bool], OCC: cutlass.Constexpr[int],
    CARVEOUT: cutlass.Constexpr[int],
) -> None:
    gate_bwd_amax_ctm_kernel(
        grad, proj, gate, out0, out1, amax, group_amax, ws_dbias, rht,
        n_w, n_cb, gate_stride_w, groups_per_sample, DBIAS, GROUP, HINT, VOL, BFRAG_LATE,
    ).launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1),
             min_blocks_per_mp=OCC, preferred_smem_carveout=CARVEOUT, stream=stream)


@cute.jit
def _finalize_host(
    ws: cute.Tensor, dbias: cute.Tensor, slots: cutlass.Int32, n: cutlass.Int32,
    grid_x: cutlass.Int32, stream: cuda.CUstream, V1: cutlass.Constexpr[bool],
    OUT_BF16: cutlass.Constexpr[bool],
) -> None:
    if cutlass.const_expr(V1):
        gate_dbias_finalize_ctm_v1_kernel(ws, dbias, slots, n).launch(
            grid=(grid_x, 1, 1), block=(FIN_THREADS, 1, 1), stream=stream)
    else:
        gate_dbias_finalize_ctm_kernel(ws, dbias, slots, n, OUT_BF16).launch(
            grid=(grid_x, 1, 1), block=(FIN_THREADS, 1, 1), stream=stream)


#: Finaliser variants: ``F2`` the strength-reduced default, ``F1`` the first-round control.
FIN_VARIANTS: tuple[str, ...] = ("F2", "F1")
DEFAULT_FIN = "F2"


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_compiled: dict[tuple, Callable] = {}
_compiled_fin: dict[tuple[int, str], Callable] = {}
_dummies: dict[int, torch.Tensor] = {}
#: ``side``: one persistent partial buffer per (device, slots, N) and ONE
#: persistent finaliser stream per device, created on first (eager) use.
_partials: dict[tuple[int, int, int], torch.Tensor] = {}
_fin_streams: dict[int, torch.cuda.Stream] = {}


def describe() -> str:
    """One line of facts for the job log: DSL version, and which warp sync ran."""
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    routing = (f"{select_variant(RUBIN_SMALL_M, cc)} at M<={RUBIN_SMALL_M}, "
               f"{select_variant(RUBIN_SMALL_M + 1, cc)} above") if cc else "<no device>"
    return (f"gate_bwd_amax_ctm: cutlass-dsl {cutlass.__version__}  "
            f"syncwarp={'cute.arch.sync_warp' if _SYNC_WARP is not None else 'sync_threads (fallback)'}  "
            f"variants={list(LAUNCH_VARIANTS)}  routing cc={cc}: {routing}  "
            f"finalisers={list(FIN_VARIANTS)} default={DEFAULT_FIN}")


def _compile(device_index: int, dbias: bool, group: bool, variant: str) -> Callable:
    """Compile once per (device, constexprs, launch variant); every shape stays symbolic."""
    occ, carveout, hint, vol, bfrag_late = LAUNCH_VARIANTS[variant]
    key = (device_index, dbias, group, variant)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            sym = cute.sym_int64
            f_words = [make_fake_compact_tensor(I32, (sym(256, s),), assumed_align=16)
                       for s in ("g", "p", "o0", "o1")]
            f_gate = make_fake_compact_tensor(I32, (sym(8, "gt"),), assumed_align=16)
            f_amax = make_fake_compact_tensor(F32, (2,), assumed_align=8)
            f_group = make_fake_compact_tensor(F32, (sym(1, "gr"),), assumed_align=4)
            f_wsd = make_fake_compact_tensor(F32, (sym(512, "wd"),), assumed_align=16)
            f_rht = make_fake_compact_tensor(I16, (256,), assumed_align=16)
            ints = [I32(0)] * 5
            _compiled[key] = cute.compile(
                _producer_host, f_words[0], f_words[1], f_gate, f_words[2], f_words[3],
                f_amax, f_group, f_wsd, f_rht, *ints, make_fake_stream(),
                dbias, group, hint, vol, bfrag_late, occ, carveout, options="--enable-tvm-ffi",
            )
    return _compiled[key]


def _compile_fin(device_index: int, fin: str = DEFAULT_FIN, out_bf16: bool = False) -> Callable:
    if fin not in FIN_VARIANTS:
        raise ValueError(f"unknown finaliser variant {fin!r}; one of {FIN_VARIANTS}")
    if out_bf16 and fin == "F1":
        raise ValueError("the bf16-output finaliser exists for F2 only")
    key = (device_index, fin, out_bf16)
    if key not in _compiled_fin:
        with torch.cuda.device(device_index):
            sym = cute.sym_int64
            f_ws = make_fake_compact_tensor(F32, (sym(512, "wd"),), assumed_align=16)
            if out_bf16:
                # a param.grad view: 4 B alignment is all the grad buffer guarantees
                f_db = make_fake_compact_tensor(BF16, (sym(128, "db"),), assumed_align=4)
            else:
                f_db = make_fake_compact_tensor(F32, (sym(128, "db"),), assumed_align=16)
            _compiled_fin[key] = cute.compile(
                _finalize_host, f_ws, f_db, I32(0), I32(0), I32(0), make_fake_stream(),
                fin == "F1", out_bf16, options="--enable-tvm-ffi",
            )
    return _compiled_fin[key]


def compile_all(device_index: int = 0, variants: tuple[str, ...] | None = None) -> None:
    """Compile the production constexpr set (dbias, no group) of every variant + both finalisers."""
    for v in (variants or tuple(LAUNCH_VARIANTS)):
        _compile(device_index, True, False, v)
    for fin in FIN_VARIANTS:
        _compile_fin(device_index, fin)


def geometry(m: int, n: int) -> tuple[int, int, int, int, int]:
    """``(rg, n_cb, grid, slots, fin_grid)`` for ``[M, N]``: ``rg = M/16``, ``n_cb = N/512``,
    one-shot ``grid = rg x n_cb``, ``slots = rg`` partial rows, ``fin_grid = N/128``."""
    rg = m // GROUP_ROWS
    n_cb = n // CTA_COLS
    return rg, n_cb, rg * n_cb, rg, n // FIN_COLS


def supported(grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor) -> bool:
    """Whether the DSL kernel accepts these operands (else the feeder uses the plain ``flux.fused_gate_bwd``).

    Args:
        grad: ``[B, S, D]`` BF16.
        gate: ``[B, 1, D]`` BF16 view.
        proj: ``[B, S, D]`` BF16.

    Returns:
        True for contiguous BF16 3-D inputs with ``S % 16 == 0``, ``D % 512 == 0``,
        a 16 B-aligned gate view, and ``M x N < 2**31``.
    """
    if grad.dim() != 3 or grad.shape != proj.shape:
        return False
    b, s, d = grad.shape
    return (
        grad.dtype is torch.bfloat16 and proj.dtype is torch.bfloat16
        and gate.dtype is torch.bfloat16
        and grad.is_contiguous() and proj.is_contiguous()
        and s % GROUP_ROWS == 0 and s > 0 and d % CTA_COLS == 0 and d > 0
        and b * s * d <= _INT32_MAX
        and tuple(gate.shape) == (b, 1, d) and gate.stride(2) == 1
        and gate.stride(0) % 8 == 0 and gate.data_ptr() % 16 == 0
        and grad.data_ptr() % 16 == 0 and proj.data_ptr() % 16 == 0
    )


def _check(grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor, rht: torch.Tensor) -> None:
    if not (grad.is_cuda and gate.is_cuda and proj.is_cuda and rht.is_cuda):
        raise ValueError("gate_bwd_amax_ctm: all tensors must be CUDA")
    if not supported(grad, gate, proj):
        raise ValueError(
            "gate_bwd_amax_ctm: grad/proj must be contiguous bf16 [B, S, D] with S % 16 == 0, "
            "D % 512 == 0, M x N < 2**31, and gate a 16 B-aligned bf16 [B, 1, D] view; got "
            f"grad {tuple(grad.shape)} {grad.dtype}, gate {tuple(gate.shape)} stride {tuple(gate.stride())}"
        )
    if not (rht.dtype is torch.bfloat16 and rht.is_contiguous() and tuple(rht.shape) == (16, 16)):
        raise ValueError("gate_bwd_amax_ctm: rht_matrix must be a contiguous bf16 [16, 16]")


def _dummy(device: torch.device) -> torch.Tensor:
    """A persistent 512-float buffer standing in for an absent ``ws_dbias`` / ``group_amax``."""
    buf = _dummies.get(device.index)
    if buf is None:
        buf = torch.zeros(512, dtype=torch.float32, device=device)
        _dummies[device.index] = buf
    return buf


def _partials_for(device: torch.device, slots: int, d: int) -> torch.Tensor:
    key = (device.index, slots, d)
    buf = _partials.get(key)
    if buf is None:
        buf = torch.empty(slots * d, dtype=torch.float32, device=device)
        _partials[key] = buf
    return buf


def _fin_stream_for(device: torch.device) -> torch.cuda.Stream:
    stream = _fin_streams.get(device.index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _fin_streams[device.index] = stream
    return stream


def _memset_amax(amax: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """The 8 B memset node the ``red.max`` completion accumulates into (captured as such)."""
    (err,) = cuda.cuMemsetD32Async(cuda.CUdeviceptr(amax.data_ptr()), 0, 2,
                                   cuda.CUstream(stream.cuda_stream))
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"gate_bwd_amax_ctm: cuMemsetD32Async failed: {err}")


def _gate_flat_words(gate: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The gate view's storage as a flat Int32 tensor, and its row stride in words."""
    gw = gate.view(torch.int32)                       # [B, 1, D/2], stride(0) = stride/2
    b = gw.shape[0]
    stride_w = gw.stride(0)
    span = (b - 1) * stride_w + gw.shape[2]
    return gw.as_strided((span,), (1,)), stride_w


def _launch(grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor, rht: torch.Tensor, *,
            dbias: bool, group: torch.Tensor | None, variant: str | None,
            ws: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
    """The producer alone: ``(out0, out1, amax, ws_dbias)``; ``ws_dbias`` holds the
    partials when ``dbias`` (allocated here unless given) and is a dummy otherwise.
    ``variant`` ``None`` is :func:`select_variant`'s choice for this M and device."""
    b, s, d = grad.shape
    m = b * s
    dev = grad.device.index
    if variant is None:
        variant = select_variant(m, torch.cuda.get_device_capability(dev))
    rg, n_cb, grid, _, _ = geometry(m, d)
    out0 = torch.empty_like(grad)
    out1 = torch.empty_like(grad)
    amax = torch.empty(2, dtype=torch.float32, device=grad.device)
    stream = torch.cuda.current_stream(dev)
    _memset_amax(amax, stream)
    if dbias:
        ws_t = ws if ws is not None else torch.empty(rg * d, dtype=torch.float32, device=grad.device)
    else:
        ws_t = _dummy(grad.device)
    g = group if group is not None else _dummy(grad.device)
    gate_flat, stride_w = _gate_flat_words(gate)
    _compile(dev, dbias, group is not None, variant)(
        grad.view(torch.int32).view(-1), proj.view(torch.int32).view(-1), gate_flat,
        out0.view(torch.int32).view(-1), out1.view(torch.int32).view(-1),
        amax, g, ws_t, rht.view(torch.int16).view(-1),
        d // 2, n_cb, stride_w, s // GROUP_ROWS, grid, cuda.CUstream(stream.cuda_stream),
    )
    return out0, out1, amax, ws_t


def _finalize(ws: torch.Tensor, dbias: torch.Tensor, slots: int, n: int,
              stream: torch.cuda.Stream, fin: str = DEFAULT_FIN) -> None:
    out_bf16 = dbias.dtype == torch.bfloat16
    if out_bf16 and (not dbias.is_contiguous() or dbias.numel() != n or dbias.data_ptr() % 4):
        raise ValueError("gate dbias finaliser: a bf16 dbias_out must be a contiguous, "
                         f"4-byte aligned [{n}] tensor")
    _compile_fin(dbias.device.index, fin, out_bf16)(
        ws, dbias.view(-1), slots, n, n // FIN_COLS, cuda.CUstream(stream.cuda_stream))


def gate_bwd_amax_ctm(
    grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor, rht_matrix: torch.Tensor,
    variant: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused gate backward and NVFP4 amax pair (the base op) on the DSL.

    Args:
        grad: ``[B, S, D]`` BF16 contiguous, ``S % 16 == 0``, ``D % 512 == 0``.
        gate: ``[B, 1, D]`` BF16 view, any 16 B-aligned row stride.
        proj: ``[B, S, D]`` BF16 contiguous.
        rht_matrix: The quantizer's ``[16, 16]`` BF16 Hadamard basis.
        variant: Key of :data:`LAUNCH_VARIANTS`; :func:`select_variant`'s choice when omitted.

    Returns:
        ``(grad_proj, grad_gate, amax)``: two BF16 ``[B, S, D]`` outputs bitwise
        ``flux.fused_gate_bwd``'s, and FP32 ``[2]`` = ``(max|grad_proj|, post-RHT amax)``.
    """
    _check(grad, gate, proj, rht_matrix)
    return _launch(grad, gate, proj, rht_matrix, dbias=False, group=None, variant=variant)[:3]


def gate_bwd_amax_pergroup_ctm(
    grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor, rht_matrix: torch.Tensor,
    variant: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`gate_bwd_amax_ctm` plus the per-16-row-group post-RHT amax (debug).

    Args:
        grad: As :func:`gate_bwd_amax_ctm`.
        gate: As :func:`gate_bwd_amax_ctm`.
        proj: As :func:`gate_bwd_amax_ctm`.
        rht_matrix: As :func:`gate_bwd_amax_ctm`.
        variant: As :func:`gate_bwd_amax_ctm`.

    Returns:
        ``(grad_proj, grad_gate, amax, group_amax)`` with ``group_amax`` FP32 ``[B * S / 16]``.
    """
    _check(grad, gate, proj, rht_matrix)
    group = torch.zeros(grad.shape[0] * grad.shape[1] // GROUP_ROWS,
                        dtype=torch.float32, device=grad.device)
    out0, out1, amax, _ = _launch(grad, gate, proj, rht_matrix, dbias=False, group=group,
                                  variant=variant)
    return out0, out1, amax, group


def gate_bwd_dbias_amax_ctm(
    grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor, rht_matrix: torch.Tensor,
    variant: str | None = None, fin: str = DEFAULT_FIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`gate_bwd_amax_ctm` plus the bias gradient, finalised serially on the current stream.

    The capturable one-op comparator and the bitwise oracle of the deferred op.

    Args:
        grad: As :func:`gate_bwd_amax_ctm`.
        gate: As :func:`gate_bwd_amax_ctm`.
        proj: As :func:`gate_bwd_amax_ctm`.
        rht_matrix: As :func:`gate_bwd_amax_ctm`.
        variant: As :func:`gate_bwd_amax_ctm`.
        fin: Finaliser variant, one of :data:`FIN_VARIANTS`.

    Returns:
        ``(grad_proj, grad_gate, amax, dbias)`` with ``dbias`` FP32 ``[D]``, the fixed-order
        column sum of the BF16 ``grad_proj``.
    """
    _check(grad, gate, proj, rht_matrix)
    b, s, d = grad.shape
    out0, out1, amax, ws = _launch(grad, gate, proj, rht_matrix, dbias=True, group=None,
                                   variant=variant)
    dbias = torch.empty(d, dtype=torch.float32, device=grad.device)
    _finalize(ws, dbias, b * s // GROUP_ROWS, d, torch.cuda.current_stream(grad.device), fin)
    return out0, out1, amax, dbias


def gate_bwd_dbias_amax_ctm_nofin(
    grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor, rht_matrix: torch.Tensor,
    variant: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BENCH ONLY: the dbias producer without its finaliser (partials written into the
    persistent buffer, ``dbias`` never produced). Prices the finaliser; never an op."""
    _check(grad, gate, proj, rht_matrix)
    b, s, d = grad.shape
    ws = _partials_for(grad.device, b * s // GROUP_ROWS, d)
    return _launch(grad, gate, proj, rht_matrix, dbias=True, group=None, variant=variant,
                   ws=ws)[:3]


def gate_bwd_dbias_amax_deferred_ctm(
    grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor, rht_matrix: torch.Tensor,
    variant: str | None = None, fin: str = DEFAULT_FIN, dbias_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`gate_bwd_dbias_amax_ctm` with the finaliser on a forked stream.

    The producer runs on the current stream and leaves its partials in a
    persistent per-shape buffer; the fixed-order finaliser runs on a per-device
    side stream that waits on the current one, so it overlaps whatever the
    caller launches next (the NVFP4 cast, which does not read the dbias). **The
    caller must call :func:`join_dbias_ctm` before consuming ``dbias`` and
    before its next call into this op.** Same kernels, same order, so ``dbias``
    is bitwise the serial op's. Under capture the fork and join become graph
    edges.

    Args:
        grad: As :func:`gate_bwd_amax_ctm`.
        gate: As :func:`gate_bwd_amax_ctm`.
        proj: As :func:`gate_bwd_amax_ctm`.
        rht_matrix: As :func:`gate_bwd_amax_ctm`.
        variant: As :func:`gate_bwd_amax_ctm`.
        fin: Finaliser variant, one of :data:`FIN_VARIANTS`.
        dbias_out: Optional bf16 ``[D]`` tensor (the bias's ``param.grad`` view): the
            finaliser rounds its fixed-order fp32 sum once (RNE) and writes it there,
            and it is returned as ``dbias``; the fp32 ``dbias`` path
            is unchanged when omitted.

    Returns:
        ``(grad_proj, grad_gate, amax, dbias)``; ``dbias`` is valid after :func:`join_dbias_ctm`.
    """
    _check(grad, gate, proj, rht_matrix)
    b, s, d = grad.shape
    slots = b * s // GROUP_ROWS
    ws = _partials_for(grad.device, slots, d)
    out0, out1, amax, _ = _launch(grad, gate, proj, rht_matrix, dbias=True, group=None,
                                  variant=variant, ws=ws)
    if dbias_out is None:
        dbias = torch.empty(d, dtype=torch.float32, device=grad.device)
    else:
        if dbias_out.dtype != torch.bfloat16:
            raise ValueError("gate_bwd_dbias_amax_deferred_ctm: dbias_out must be bf16")
        dbias = dbias_out
    side = _fin_stream_for(grad.device)
    # Fork: an event recorded on the current stream after the producer, waited on
    # by `side` -- the finaliser sees the partials; under capture, a graph edge.
    side.wait_stream(torch.cuda.current_stream(grad.device))
    with torch.cuda.stream(side):
        _finalize(ws, dbias, slots, d, side, fin)
    # Eager-path insurance only (the caching allocator must not recycle dbias's
    # block on the current stream before `side` is done); under capture the two
    # dependency edges are what make the op safe.
    dbias.record_stream(side)
    return out0, out1, amax, dbias


def join_dbias_ctm(device: torch.device | None = None) -> None:
    """Make the current stream wait for the deferred finaliser.

    Args:
        device: The device whose side stream to join; the current one if omitted.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    fin = _fin_streams.get(device.index)
    if fin is not None:
        torch.cuda.current_stream(device).wait_stream(fin)


# ---------------------------------------------------------------------------
# torch.library registration (unwired: nothing in flat/ imports this module)
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "gate_bwd_amax_ctm(Tensor grad, Tensor gate, Tensor proj, Tensor rht_matrix)"
    " -> (Tensor, Tensor, Tensor)"
)
_lib.define(
    "gate_bwd_amax_pergroup_ctm(Tensor grad, Tensor gate, Tensor proj,"
    " Tensor rht_matrix) -> (Tensor, Tensor, Tensor, Tensor)"
)
_lib.define(
    "gate_bwd_dbias_amax_ctm(Tensor grad, Tensor gate, Tensor proj,"
    " Tensor rht_matrix) -> (Tensor, Tensor, Tensor, Tensor)"
)
_lib.define(
    "gate_bwd_dbias_amax_deferred_ctm(Tensor grad, Tensor gate, Tensor proj,"
    " Tensor rht_matrix, Tensor(a!)? dbias_out=None) -> (Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "gate_bwd_amax_ctm", "CUDA")
def _gate_bwd_amax_ctm_impl(grad, gate, proj, rht_matrix):
    return gate_bwd_amax_ctm(grad, gate, proj, rht_matrix)


@torch.library.impl(_lib, "gate_bwd_amax_pergroup_ctm", "CUDA")
def _gate_bwd_amax_pergroup_ctm_impl(grad, gate, proj, rht_matrix):
    return gate_bwd_amax_pergroup_ctm(grad, gate, proj, rht_matrix)


@torch.library.impl(_lib, "gate_bwd_dbias_amax_ctm", "CUDA")
def _gate_bwd_dbias_amax_ctm_impl(grad, gate, proj, rht_matrix):
    return gate_bwd_dbias_amax_ctm(grad, gate, proj, rht_matrix)


@torch.library.impl(_lib, "gate_bwd_dbias_amax_deferred_ctm", "CUDA")
def _gate_bwd_dbias_amax_deferred_ctm_impl(grad, gate, proj, rht_matrix, dbias_out=None):
    return gate_bwd_dbias_amax_deferred_ctm(grad, gate, proj, rht_matrix, dbias_out=dbias_out)
