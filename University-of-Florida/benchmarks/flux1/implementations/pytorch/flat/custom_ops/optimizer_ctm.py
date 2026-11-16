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

"""The reduced-gradient finaliser and the fused AdamW on the CUTLASS DSL.

Replaces, over one rank's ZeRO-2 shard (11.9 G parameters / G), the step's tail:
``torch.nan_to_num_(bf16)`` -> ``f32.copy_(bf16)`` -> TE ``multi_tensor_l2norm`` (+ ``cleanup``, ``.pow(2)``) and
TE ``multi_tensor_adam_capturable``. Measured at the shipped ``04x04x32`` topology (shard 16): 7.17 + 0.79 + 1.85 +
0.50 ms per step, TE's Adam at 2.9 TB/s and the aten upcast at 2.4 TB/s against the 6.3-6.9 TB/s a DSL streaming
kernel reaches on this same shard (``weight_amax_copy_ctm.py`` -- the template for these kernels).

Three ``@cute.kernel``s on base ``cute.arch.*`` (a streaming pass with a register epilogue and one block
reduction; no TMA / mbarrier / tcgen05 -- an async pipeline has nothing to hide in a kernel
at the copy band). Public ``nvidia-cutlass-dsl`` (4.8.0 in the container).

* ``grad_norm_partial_ctm_kernel`` -- one 512-thread CTA per TE chunk (65 536 elements of one bucket shard; the
  host tables ``chunk_base`` / ``chunk_len`` mirror ``multi_tensor_apply``'s chunking). Reads the **bf16** shard
  (2 B/elt), applies ``nan_to_num(0, 0, 0)`` in registers and accumulates ``vals[ii] = fma(x, x, vals[ii])`` in
  TE's exact per-thread order (thread ``t``, group ``k``: elements ``(t + 512 k) * 4 + ii``), then TE's
  ``reduce_block_into_lanes`` (smem tree 256 / 128 / 64, ``x[tid] + x[tid + 32]``, ``shfl.down`` 16..1); writes
  ``partial[chunk]``; sets ``noop = 1`` on a non-finite chunk sum (TE's semantics).
* ``grad_norm_cleanup_ctm_kernel`` -- one CTA: slot ``s < 320`` accumulates ``partial[s], partial[s + 320], ...``
  in launch order (TE's ``output[blockIdx.x] += final`` across its 320-CTA launches, from 0), the same block
  reduce, then ``r = sqrt.rn(final)``; ``norm_sq = r * r`` (the caller's ``.pow(2)``).
* ``adamw_ctm_kernel`` -- flat elementwise over the whole shard (the four buffers share the shard layout, so the
  18 bucket slices are one span): 256 threads x 8 elements, one 16 B bf16 load + six 16 B f32 loads / stores per
  thread (26 B/elt against TE's 28 -- TE writes the scaled grad back into an f32 buffer nothing reads).
  ``AdamCapturableFunctor``'s math for mode 1 (AdamW) with bias correction, ``lr`` / ``step`` / ``inv_scale`` /
  ``noop`` read from device scalars (graph-safe), the FMA contraction pinned by explicit ``fma.rn`` / ``mul.rn`` /
  ``add.rn`` / ``div.rn`` / ``sqrt.rn`` (:data:`PATTERN`, fixed by the bitwise test against TE).

Bitwise contract (``tests/test_optimizer_ctm.py``): ``norm_sq`` equals the TE chain's, ``p`` / ``m`` / ``v`` equal
``multi_tensor_adam_capturable``'s, as ``int32`` patterns; graph-safe; hard errors, no fallback.
"""

from __future__ import annotations

from typing import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

#: TE ``multi_tensor_apply`` constants mirrored for the norm (l2norm.cu / multi_tensor_apply.cuh).
CHUNK = 65536
NORM_THREADS = 512
NORM_SLOTS = 320
#: The Adam kernel's geometry.
ADAM_THREADS = 256
ADAM_EPT = 8                              # elements per thread: one 16 B bf16 load, two float4 per f32 buffer
ADAM_BLOCK_ELTS = ADAM_THREADS * ADAM_EPT  # 2048
#: Kernel-symbol substrings for the by-name launch census.
NORM_PARTIAL_KERNEL = "grad_norm_partial_ctm_kernel"
NORM_CLEANUP_KERNEL = "grad_norm_cleanup_ctm_kernel"
ADAM_KERNEL = "adamw_ctm_kernel"
#: FMA contraction pattern of ``AdamCapturableFunctor`` as nvcc compiled it:
#: (M, V, U, P) -- M: 0 ``fma(b1, m, (1-b1)*g)``, 1 ``fma(1-b1, g, b1*m)``, 2 unfused;
#: V: 0 ``fma(b2, v, ((1-b2)*g)*g)``, 1 ``fma((1-b2)*g, g, b2*v)``, 2 unfused;
#: U: 0 ``fma(decay, p, m_hat/denom)``, 1 unfused; P: 0 ``fma(-lr, update, p)``, 1 unfused.
PATTERN: tuple[int, int, int, int] = (0, 0, 0, 0)
#: TE compiles ``multi_tensor/adam.cu`` and ``l2norm.cu`` with ``--use_fast_math`` (common/CMakeLists.txt,
#: ``nvte_sources_with_fast_math``): ``-ftz=true -prec-div=false -prec-sqrt=false -fmad=true`` and ``powf ->
#: __powf``. ``FAST = True`` mirrors that lowering -- ``.ftz`` on every fma / mul / add, ``div.full.ftz.f32`` for
#: ``/``, ``sqrt.approx.ftz.f32`` for ``sqrtf`` -- which is what bitwise equality with TE requires (with IEEE ``div.rn`` / ``sqrt.rn`` the update differed in 1.3 % of elements, ``m`` / ``v`` and the
#: norm already matched). ``False`` = IEEE round-to-nearest everywhere (the numerically stricter twin, kept as a
#: knob and for the test's negative control).
FAST = True
#: Bias-correction source: 0 = in-kernel ``1 - pow(beta, float(step))`` (precise ``__nv_powf``); 1 = the host table
#: ``bc_table[0 | 1][step]`` (:func:`build_bc_table`, algorithm :data:`BC_ALGO`); 2 = the in-kernel fast-math
#: ``__powf`` mirror ``1 - ex2.approx(step * lg2.approx(beta))``. CUDA's ``pow(float, int)`` overload is ``powif``
#: (integer power by repeated squaring): nvcc's value did not move under ``--use_fast_math`` (a ``powf`` would have
#: become ``__powf``) and matches neither ``powf`` (6 / 16 of 8193 steps) nor the double promotion (35 / 69), and the
#: DSL's ``fpowi`` segfaults the compiler on a dynamic exponent -- so the table it is. Later testing showed the value IS the double promotion of the float32-rounded beta (``(float)(1 - pow((double)b32, s))``).
BC_MODE = 1
#: Last step the ``BC_MODE == 1`` table covers; ``adamw_ctm_kernel`` indexes it unchecked, so a larger step reads out of bounds.
BC_MAX_STEP = 32768
#: :func:`build_bc_table`'s algorithm when ``BC_MODE == 1``: ``powif_lsb`` = libdevice ``__nv_powif``'s loop (LSB-first
#: repeated squaring, float32); ``powif_msb`` = MSB-first square-and-multiply; ``seq`` = ``step`` sequential multiplies;
#: ``powf`` = torch's device ``powf``; ``double`` = the C++ promotion. The sweep picks the one bitwise nvcc.
BC_ALGO = "double"
#: How ``r_m / beta1_correction`` and ``r_v / beta2_correction`` are lowered. The corrections are per-thread
#: invariants, and under ``-prec-div=false`` nvcc hoists ``x / c`` as ``x * (1 / c)``: 0 = ``div.full(x, bc)``
#: per element; 1 = ``x * rcp.approx.ftz(bc)``; 2 = ``x * div.full(1, bc)``. With 0 the update
#: matched TE only where bc == 1.0 (steps 1000 / 4000) and differed in 2e-4 of elements at step 1, where the
#: correction value itself is exact in every implementation -- the divisor path, not the value, was the miss.
DIV_BC = 0
#: Accumulate the norm as ``fma(x, x, acc)`` (nvcc's contraction of ``vals[ii] += next * next``) or unfused.
NORM_FMA = True
FLT_MAX = 3.4028234663852886e38

_RN = cute.math.RoundingMode.NEAREST_EVEN
_BF16_HI_MASK = -65536      # 0xFFFF0000 as a signed Int32
_EXP_LO = 0x7F80            # bf16 exponent field of the low half-word
_EXP_HI = 0x7F800000        # bf16 exponent field of the high half-word


def _f32(x: float):
    return cutlass.Float32(x)


def _add(a, b, fast=True):
    return cute.math.add(a, b, rounding=_RN, ftz=bool(fast))


def _sub(a, b, fast=True):
    # the public wheel's ``sub`` takes no ``rounding``; ``a - b`` is ``a + (-b)`` in IEEE 754 (negation is exact)
    return cute.math.add(a, cute.math.neg(b), rounding=_RN, ftz=bool(fast))


def _mul(a, b, fast=True):
    return cute.math.mul(a, b, rounding=_RN, ftz=bool(fast))


def _div(a, b, fast=True):
    """``a / b``: nvcc ``-prec-div=false`` (TE's build) is ``div.full.ftz.f32``; IEEE is ``div.rn.f32``."""
    if fast:
        return cute.math.div(a, b, full=True, ftz=True)
    return cute.math.div(a, b, rounding=_RN)


def _sqrt(x, fast=True):
    """``sqrtf(x)``: nvcc ``-prec-sqrt=false`` is ``sqrt.approx.ftz.f32``; IEEE is ``sqrt.rn.f32``."""
    if fast:
        return cute.math.sqrt(x, approx=True, ftz=True)
    return cute.math.sqrt(x, rounding=_RN)


def _fma(a, b, c, fast=True):
    return cute.math.fma(a, b, c, rounding=_RN, ftz=bool(fast))


def _lo_clean(w):
    """bf16 in the low half of a packed word -> f32, exactly, with NaN / +-Inf -> +0.0 (``nan_to_num(0, 0, 0)``)."""
    x = (w << 16).bitcast(cutlass.Float32)
    bad = (w & cutlass.Int32(_EXP_LO)) == cutlass.Int32(_EXP_LO)
    return cutlass.Float32(cutlass.select_(bad, _f32(0.0), x))


def _hi_clean(w):
    x = (w & cutlass.Int32(_BF16_HI_MASK)).bitcast(cutlass.Float32)
    bad = (w & cutlass.Int32(_EXP_HI)) == cutlass.Int32(_EXP_HI)
    return cutlass.Float32(cutlass.select_(bad, _f32(0.0), x))


def _i16_clean(h):
    """One bf16 read as Int16 -> f32, cleaned (the Adam tail path)."""
    return _lo_clean(h.to(cutlass.Int32) & cutlass.Int32(0xFFFF))


@cute.jit
def _reduce_block_512(stage, tid, val, FAST_: cutlass.Constexpr[bool]):
    """TE's ``reduce_block_into_lanes(x, val)`` for a 512-thread CTA; lane 0 of warp 0 holds the sum."""
    (stage + tid).store(val)
    cute.arch.sync_threads()
    for i in cutlass.range_constexpr(3):
        half = 256 >> i
        if tid < half:
            (stage + tid).store(_add((stage + tid).load(), (stage + tid + half).load(), FAST_))
        cute.arch.sync_threads()
    final = _f32(0.0)
    if tid < 32:
        final = _add((stage + tid).load(), (stage + tid + 32).load(), FAST_)
        for off in cutlass.range_constexpr(5):
            final = _add(final, cute.arch.shuffle_sync_down(final, offset=16 >> off), FAST_)
    return final


@cute.kernel
def grad_norm_partial_ctm_kernel(
    g: cute.Tensor, chunk_base: cute.Tensor, chunk_len: cute.Tensor, partial: cute.Tensor, noop: cute.Tensor,
    FMA_ACC: cutlass.Constexpr[bool], FAST_: cutlass.Constexpr[bool],
) -> None:
    """``partial[c] = sum(nan_to_num(g[chunk c]) ** 2)`` in TE's ``L2NormFunctor`` order (aligned path)."""
    tid, _, _ = cute.arch.thread_idx()
    c, _, _ = cute.arch.block_idx()
    gw = cutlass.Pointer(g.iterator, dtype=cutlass.Int32)      # bf16 pairs
    pp = cutlass.Pointer(partial.iterator, dtype=cutlass.Float32)
    np_ = cutlass.Pointer(noop.iterator, dtype=cutlass.Int32)
    stage = cute.arch.alloc_smem(cutlass.Float32, NORM_THREADS, 16)
    base = chunk_base[c]          # Int64 element offset, a multiple of 4
    n = chunk_len[c]              # Int32, a multiple of 4
    v0 = _f32(0.0)
    v1 = _f32(0.0)
    v2 = _f32(0.0)
    v3 = _f32(0.0)
    for k in cutlass.range(tid, n // 4, NORM_THREADS):
        off = base + k * 4
        w = (gw + off // 2).load(8, count=2)
        x0 = _lo_clean(w[0])
        x1 = _hi_clean(w[0])
        x2 = _lo_clean(w[1])
        x3 = _hi_clean(w[1])
        if cutlass.const_expr(FMA_ACC):
            v0 = _fma(x0, x0, v0, FAST_)
            v1 = _fma(x1, x1, v1, FAST_)
            v2 = _fma(x2, x2, v2, FAST_)
            v3 = _fma(x3, x3, v3, FAST_)
        else:
            v0 = _add(v0, _mul(x0, x0, FAST_), FAST_)
            v1 = _add(v1, _mul(x1, x1, FAST_), FAST_)
            v2 = _add(v2, _mul(x2, x2, FAST_), FAST_)
            v3 = _add(v3, _mul(x3, x3, FAST_), FAST_)
    val = _add(_add(_add(v0, v1, FAST_), v2, FAST_), v3, FAST_)
    final = _reduce_block_512(stage, tid, val, FAST_)
    if tid == 0:
        (pp + c).store(final)
        if final != final:
            np_.store(cutlass.Int32(1))
        if cute.math.abs(final) > _f32(FLT_MAX):
            np_.store(cutlass.Int32(1))


@cute.kernel
def grad_norm_cleanup_ctm_kernel(
    partial: cute.Tensor, n_chunks: cutlass.Int32, norm_sq: cute.Tensor, FAST_: cutlass.Constexpr[bool],
) -> None:
    """TE's ``output[s] += final`` over launches, its ``cleanup`` (sqrt of the 320-slot sum), the caller's ``pow(2)``."""
    tid, _, _ = cute.arch.thread_idx()
    pp = cutlass.Pointer(partial.iterator, dtype=cutlass.Float32)
    op = cutlass.Pointer(norm_sq.iterator, dtype=cutlass.Float32)
    stage = cute.arch.alloc_smem(cutlass.Float32, NORM_THREADS, 16)
    acc = _f32(0.0)
    if tid < NORM_SLOTS:
        for idx in cutlass.range(tid, n_chunks, NORM_SLOTS):
            acc = _add(acc, (pp + idx).load(), FAST_)
    final = _reduce_block_512(stage, tid, acc, FAST_)
    if tid == 0:
        r = _sqrt(final, FAST_)
        # the caller's ``norm.pow(2)`` is torch's ``x * x`` -- an IEEE multiply, never flushed
        op.store(cute.math.mul(r, r, rounding=_RN))


@cute.jit
def _adam_elem(x, p, m, v, inv, lr, omb1, omb2, bc1, bc2, beta1, beta2, eps, decay,
               PAT_M: cutlass.Constexpr[int], PAT_V: cutlass.Constexpr[int],
               PAT_U: cutlass.Constexpr[int], PAT_P: cutlass.Constexpr[int], FAST_: cutlass.Constexpr[bool],
               DIVBC: cutlass.Constexpr[int]):
    """One element of ``AdamCapturableFunctor`` (mode 1, bias correction on) with the contraction pinned.

    ``bc1`` / ``bc2`` are the corrections themselves when ``DIVBC == 0`` and their (hoisted) reciprocals otherwise.
    """
    g = _mul(x, inv, FAST_)
    if cutlass.const_expr(PAT_M == 0):
        m = _fma(beta1, m, _mul(omb1, g, FAST_), FAST_)
    else:
        if cutlass.const_expr(PAT_M == 1):
            m = _fma(omb1, g, _mul(beta1, m, FAST_), FAST_)
        else:
            m = _add(_mul(beta1, m, FAST_), _mul(omb1, g, FAST_), FAST_)
    gg = _mul(omb2, g, FAST_)
    if cutlass.const_expr(PAT_V == 0):
        v = _fma(beta2, v, _mul(gg, g, FAST_), FAST_)
    else:
        if cutlass.const_expr(PAT_V == 1):
            v = _fma(gg, g, _mul(beta2, v, FAST_), FAST_)
        else:
            v = _add(_mul(beta2, v, FAST_), _mul(gg, g, FAST_), FAST_)
    if cutlass.const_expr(DIVBC == 0):
        m_hat = _div(m, bc1, FAST_)
        v_hat = _div(v, bc2, FAST_)
    else:
        m_hat = _mul(m, bc1, FAST_)
        v_hat = _mul(v, bc2, FAST_)
    denom = _add(_sqrt(v_hat, FAST_), eps, FAST_)
    q = _div(m_hat, denom, FAST_)
    if cutlass.const_expr(PAT_U == 0):
        upd = _fma(decay, p, q, FAST_)
    else:
        upd = _add(q, _mul(decay, p, FAST_), FAST_)
    if cutlass.const_expr(PAT_P == 0):
        p = _fma(cute.math.neg(lr), upd, p, FAST_)
    else:
        p = _sub(p, _mul(lr, upd, FAST_), FAST_)
    return p, m, v


@cute.kernel
def adamw_ctm_kernel(
    g: cute.Tensor, p: cute.Tensor, m: cute.Tensor, v: cute.Tensor,
    lr_t: cute.Tensor, step_t: cute.Tensor, inv_t: cute.Tensor, noop: cute.Tensor, bc_table: cute.Tensor,
    n: cutlass.Int64, beta1: cutlass.Float32, beta2: cutlass.Float32, eps: cutlass.Float32, decay: cutlass.Float32,
    PAT_M: cutlass.Constexpr[int], PAT_V: cutlass.Constexpr[int], PAT_U: cutlass.Constexpr[int],
    PAT_P: cutlass.Constexpr[int], BC: cutlass.Constexpr[int], FAST_: cutlass.Constexpr[bool],
    DIVBC: cutlass.Constexpr[int],
) -> None:
    """AdamW over ``[0, n)``: ``g`` bf16 (cleaned in registers, scaled by ``inv``), ``p`` / ``m`` / ``v`` f32 in place."""
    tid, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gw = cutlass.Pointer(g.iterator, dtype=cutlass.Int32)      # bf16 pairs
    gh = cutlass.Pointer(g.iterator, dtype=cutlass.Int16)      # single bf16 (tail)
    pp = cutlass.Pointer(p.iterator, dtype=cutlass.Float32)
    mp = cutlass.Pointer(m.iterator, dtype=cutlass.Float32)
    vp = cutlass.Pointer(v.iterator, dtype=cutlass.Float32)
    if noop[0] == 0:
        lr = lr_t[0]
        inv = inv_t[0]
        step = step_t[0]
        step_f = step.to(cutlass.Float32)
        if cutlass.const_expr(BC == 0):
            bc1 = _sub(_f32(1.0), cute.math.pow(beta1, step_f), FAST_)
            bc2 = _sub(_f32(1.0), cute.math.pow(beta2, step_f), FAST_)
        else:
            if cutlass.const_expr(BC == 1):
                bc1 = bc_table[step]
                bc2 = bc_table[bc_table.shape[0] // 2 + step]
            else:
                # nvcc --use_fast_math: powf(a, b) -> __powf(a, b) = ex2.approx(b * lg2.approx(a))
                bc1 = _sub(_f32(1.0), cute.math.exp2(_mul(step_f, cute.math.log2(beta1, approx=True, ftz=True), FAST_),
                                                     approx=True, ftz=True), FAST_)
                bc2 = _sub(_f32(1.0), cute.math.exp2(_mul(step_f, cute.math.log2(beta2, approx=True, ftz=True), FAST_),
                                                     approx=True, ftz=True), FAST_)
        if cutlass.const_expr(DIVBC == 1):
            bc1 = cute.math.rcp(bc1, approx=True, ftz=True)
            bc2 = cute.math.rcp(bc2, approx=True, ftz=True)
        else:
            if cutlass.const_expr(DIVBC == 2):
                bc1 = _div(_f32(1.0), bc1, FAST_)
                bc2 = _div(_f32(1.0), bc2, FAST_)
        omb1 = _sub(_f32(1.0), beta1, FAST_)
        omb2 = _sub(_f32(1.0), beta2, FAST_)
        off = bidx.to(cutlass.Int64) * ADAM_BLOCK_ELTS + tid * ADAM_EPT
        if off + ADAM_EPT <= n:
            w = (gw + off // 2).load(16, count=4)
            pf0 = (pp + off).load(16, count=4)
            pf1 = (pp + off + 4).load(16, count=4)
            mf0 = (mp + off).load(16, count=4)
            mf1 = (mp + off + 4).load(16, count=4)
            vf0 = (vp + off).load(16, count=4)
            vf1 = (vp + off + 4).load(16, count=4)
            xs = [_lo_clean(w[0]), _hi_clean(w[0]), _lo_clean(w[1]), _hi_clean(w[1]),
                  _lo_clean(w[2]), _hi_clean(w[2]), _lo_clean(w[3]), _hi_clean(w[3])]
            ps = [pf0[0], pf0[1], pf0[2], pf0[3], pf1[0], pf1[1], pf1[2], pf1[3]]
            ms = [mf0[0], mf0[1], mf0[2], mf0[3], mf1[0], mf1[1], mf1[2], mf1[3]]
            vs = [vf0[0], vf0[1], vf0[2], vf0[3], vf1[0], vf1[1], vf1[2], vf1[3]]
            for e in cutlass.range_constexpr(ADAM_EPT):
                ps[e], ms[e], vs[e] = _adam_elem(
                    xs[e], ps[e], ms[e], vs[e], inv, lr, omb1, omb2, bc1, bc2, beta1, beta2, eps, decay,
                    PAT_M, PAT_V, PAT_U, PAT_P, FAST_, DIVBC)
            (pp + off).store(cutlass.Vector.from_elements((ps[0], ps[1], ps[2], ps[3]), cutlass.Float32), alignment=16)
            (pp + off + 4).store(cutlass.Vector.from_elements((ps[4], ps[5], ps[6], ps[7]), cutlass.Float32), alignment=16)
            (mp + off).store(cutlass.Vector.from_elements((ms[0], ms[1], ms[2], ms[3]), cutlass.Float32), alignment=16)
            (mp + off + 4).store(cutlass.Vector.from_elements((ms[4], ms[5], ms[6], ms[7]), cutlass.Float32), alignment=16)
            (vp + off).store(cutlass.Vector.from_elements((vs[0], vs[1], vs[2], vs[3]), cutlass.Float32), alignment=16)
            (vp + off + 4).store(cutlass.Vector.from_elements((vs[4], vs[5], vs[6], vs[7]), cutlass.Float32), alignment=16)
        else:
            # Ragged tail (at most one CTA per launch): scalar, guarded per element.
            for e in cutlass.range_constexpr(ADAM_EPT):
                i = off + e
                if i < n:
                    x = _i16_clean((gh + i).load())
                    pe, me, ve = _adam_elem(
                        x, (pp + i).load(), (mp + i).load(), (vp + i).load(), inv, lr, omb1, omb2, bc1, bc2,
                        beta1, beta2, eps, decay, PAT_M, PAT_V, PAT_U, PAT_P, FAST_, DIVBC)
                    (pp + i).store(pe)
                    (mp + i).store(me)
                    (vp + i).store(ve)


@cute.jit
def grad_norm_partial_ctm_host(
    g: cute.Tensor, chunk_base: cute.Tensor, chunk_len: cute.Tensor, partial: cute.Tensor, noop: cute.Tensor,
    n_chunks: cutlass.Int32, stream: cuda.CUstream, FMA_ACC: cutlass.Constexpr[bool], FAST_: cutlass.Constexpr[bool],
) -> None:
    grad_norm_partial_ctm_kernel(g, chunk_base, chunk_len, partial, noop, FMA_ACC, FAST_).launch(
        grid=(n_chunks, 1, 1), block=(NORM_THREADS, 1, 1), stream=stream)


@cute.jit
def grad_norm_cleanup_ctm_host(
    partial: cute.Tensor, n_chunks: cutlass.Int32, norm_sq: cute.Tensor, stream: cuda.CUstream,
    FAST_: cutlass.Constexpr[bool],
) -> None:
    grad_norm_cleanup_ctm_kernel(partial, n_chunks, norm_sq, FAST_).launch(
        grid=(1, 1, 1), block=(NORM_THREADS, 1, 1), stream=stream)


@cute.jit
def adamw_ctm_host(
    g: cute.Tensor, p: cute.Tensor, m: cute.Tensor, v: cute.Tensor,
    lr_t: cute.Tensor, step_t: cute.Tensor, inv_t: cute.Tensor, noop: cute.Tensor, bc_table: cute.Tensor,
    n: cutlass.Int64, grid_x: cutlass.Int32,
    beta1: cutlass.Float32, beta2: cutlass.Float32, eps: cutlass.Float32, decay: cutlass.Float32,
    stream: cuda.CUstream,
    PAT_M: cutlass.Constexpr[int], PAT_V: cutlass.Constexpr[int], PAT_U: cutlass.Constexpr[int],
    PAT_P: cutlass.Constexpr[int], BC: cutlass.Constexpr[int], FAST_: cutlass.Constexpr[bool],
    DIVBC: cutlass.Constexpr[int],
) -> None:
    adamw_ctm_kernel(g, p, m, v, lr_t, step_t, inv_t, noop, bc_table, n, beta1, beta2, eps, decay,
                     PAT_M, PAT_V, PAT_U, PAT_P, BC, FAST_, DIVBC).launch(
        grid=(grid_x, 1, 1), block=(ADAM_THREADS, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_compiled: dict[tuple, Callable] = {}


def _fake(dtype, name: str, align: int):
    return make_fake_compact_tensor(dtype, (cute.sym_int64(1, name),), assumed_align=align)


def _compile_norm(device_index: int, fma_acc: bool, fast: bool) -> tuple[Callable, Callable]:
    key = ("norm", device_index, fma_acc, fast)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            f32, bf, i32, i64 = cutlass.Float32, cutlass.BFloat16, cutlass.Int32, cutlass.Int64
            part = cute.compile(
                grad_norm_partial_ctm_host, _fake(bf, "g", 16), _fake(i64, "cb", 8), _fake(i32, "cl", 4),
                _fake(f32, "pa", 4), _fake(i32, "no", 4), cutlass.Int32(0), make_fake_stream(), fma_acc, fast,
                options="--enable-tvm-ffi")
            clean = cute.compile(
                grad_norm_cleanup_ctm_host, _fake(f32, "pa", 4), cutlass.Int32(0), _fake(f32, "ns", 4),
                make_fake_stream(), fast, options="--enable-tvm-ffi")
            _compiled[key] = (part, clean)
    return _compiled[key]


def _compile_adam(device_index: int, pattern: tuple[int, int, int, int], bc_mode: int, fast: bool,
                  div_bc: int) -> Callable:
    key = ("adam", device_index, tuple(pattern), bc_mode, fast, div_bc)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            f32, bf, i32 = cutlass.Float32, cutlass.BFloat16, cutlass.Int32
            _compiled[key] = cute.compile(
                adamw_ctm_host, _fake(bf, "g", 16), _fake(f32, "p", 16), _fake(f32, "m", 16), _fake(f32, "v", 16),
                _fake(f32, "lr", 4), _fake(i32, "st", 4), _fake(f32, "iv", 4), _fake(i32, "no", 4),
                _fake(f32, "bc", 4), cutlass.Int64(0), cutlass.Int32(0),
                cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0),
                make_fake_stream(), *[int(x) for x in pattern], int(bc_mode), bool(fast), int(div_bc),
                options="--enable-tvm-ffi")
    return _compiled[key]


def build_chunk_tables(
    shards: Sequence[tuple[int, int]], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TE ``multi_tensor_apply``'s chunk list for the bucket shards, as ``(chunk_base int64, chunk_len int32)``.

    Args:
        shards: ``(offset, numel)`` per bucket shard in flat shard-buffer elements, in bucket order (the order
            ``multi_tensor_l2norm`` receives the list). Each ``offset`` must be a multiple of 4 and each ``numel``
            a multiple of 4 (TE's aligned path; production aligns to 16).
        device: CUDA device for the tables.

    Raises:
        ValueError: On an alignment TE's aligned path would not take (no unaligned path exists here).
    """
    base, length = [], []
    for off, numel in shards:
        if off % 4 or numel % 4:
            raise ValueError(f"shard ({off}, {numel}) is not 4-element aligned; TE's aligned path is the contract")
        for c in range((numel + CHUNK - 1) // CHUNK):
            base.append(off + c * CHUNK)
            length.append(min(CHUNK, numel - c * CHUNK))
    if not base:
        raise ValueError("no shards")
    return (torch.tensor(base, dtype=torch.int64, device=device),
            torch.tensor(length, dtype=torch.int32, device=device))


def build_bc_table(beta1: float, beta2: float, max_step: int, device: torch.device,
                   algo: str = BC_ALGO) -> torch.Tensor:
    """``[2 * (max_step + 1)]`` f32: ``1 - beta ** step`` for ``step`` in ``[0, max_step]``, the two betas back to back.

    ``powif_lsb`` reproduces CUDA's ``pow(float, int)`` (libdevice ``__nv_powif``: ``r = 1; while e: if e & 1: r *= a;
    e >>= 1; a *= a``) in float32 on the host; ``powif_msb`` and ``seq`` are the other integer-power orders; ``powf``
    and ``double`` the float / double ``pow`` promotions (torch on the device). Only :data:`BC_MODE` ``1`` reads it.

    Raises:
        ValueError: On an unknown ``algo``.
    """
    import numpy as np
    steps = np.arange(0, max_step + 1)
    out = np.empty((2, max_step + 1), dtype=np.float32)
    one = np.float32(1.0)
    for row, beta in enumerate((beta1, beta2)):
        b = np.float32(beta)
        if algo == "powif_lsb":
            for s in steps:
                e, a, r = int(s), b, one
                while e:
                    if e & 1:
                        r = np.float32(r * a)
                    e >>= 1
                    if e:
                        a = np.float32(a * a)
                out[row, s] = np.float32(one - r)
        elif algo == "powif_msb":
            for s in steps:
                r = one
                for bit in bin(int(s))[2:]:
                    r = np.float32(r * r)
                    if bit == "1":
                        r = np.float32(r * b)
                out[row, s] = np.float32(one - r)
        elif algo == "seq":
            r = one
            for s in steps:
                out[row, s] = np.float32(one - r)
                r = np.float32(r * b)
        elif algo in ("powf", "double"):
            # TE holds ``const float beta``: round to float32 FIRST, then promote. C++ overload resolution sends
            # ``pow(float, int)`` to ``pow(double, double)`` (nvcc oracle: 1 - 0.9f^2 = 0.190000042 is the double
            # square, not powf's 0.190000057); torch's device ``pow`` on float64 is the same libdevice ``pow``.
            dt = torch.float64 if algo == "double" else torch.float32
            b32 = float(np.float32(beta))
            t = 1 - torch.pow(torch.tensor([b32], dtype=dt, device=device), torch.arange(0, max_step + 1, device=device).to(dt))
            out[row] = t.to(torch.float32).cpu().numpy()
        else:
            raise ValueError(f"unknown bias-correction algorithm {algo!r}")
    return torch.from_numpy(out.reshape(-1)).to(device).contiguous()


def _check(t: torch.Tensor, dtype: torch.dtype, name: str, numel: int | None = None, align: int = 16) -> None:
    if not (t.is_cuda and t.dtype == dtype and t.is_contiguous()):
        raise ValueError(f"{name} must be contiguous {dtype} CUDA")
    if numel is not None and t.numel() != numel:
        raise ValueError(f"{name} must have {numel} elements, got {t.numel()}")
    if t.data_ptr() % align:
        raise ValueError(f"{name} must be {align} B aligned")


def grad_norm_sq_ctm(
    grads: torch.Tensor, chunk_base: torch.Tensor, chunk_len: torch.Tensor, partial: torch.Tensor,
    norm_sq: torch.Tensor, noop: torch.Tensor, fma_acc: bool = NORM_FMA, fast: bool = FAST,
) -> None:
    """``norm_sq[0] = (sqrt(sum over chunks of sum(nan_to_num(g)^2)))^2`` in TE's reduction order; two launches.

    Args:
        grads: bf16 flat reduced-grad shard buffer (the whole buffer; the tables index into it).
        chunk_base: From :func:`build_chunk_tables`.
        chunk_len: From :func:`build_chunk_tables`.
        partial: f32 ``[n_chunks]`` scratch, fully overwritten.
        norm_sq: f32 ``[1]``, overwritten.
        noop: int32 ``[1]``; set to 1 on a non-finite chunk sum, never cleared here (TE's flag).
        fma_acc: Accumulate with ``fma`` (default) or unfused.
        fast: :data:`FAST` (TE's fast-math lowering) or IEEE.

    Raises:
        ValueError: On a dtype / layout / shape mismatch.
    """
    _check(grads, torch.bfloat16, "grads", align=8)
    n_chunks = chunk_base.numel()
    _check(chunk_base, torch.int64, "chunk_base", align=8)
    _check(chunk_len, torch.int32, "chunk_len", n_chunks, align=4)
    _check(partial, torch.float32, "partial", n_chunks, align=4)
    _check(norm_sq, torch.float32, "norm_sq", 1, align=4)
    _check(noop, torch.int32, "noop", 1, align=4)
    dev = grads.device.index
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    part, clean = _compile_norm(dev, bool(fma_acc), bool(fast))
    part(grads, chunk_base, chunk_len, partial, noop, n_chunks, stream)
    clean(partial, n_chunks, norm_sq, stream)


def adamw_ctm(
    grads: torch.Tensor, p: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    lr: torch.Tensor, step: torch.Tensor, inv_scale: torch.Tensor, noop: torch.Tensor, bc_table: torch.Tensor,
    beta1: float, beta2: float, eps: float, weight_decay: float,
    pattern: tuple[int, int, int, int] = PATTERN, bc_mode: int = BC_MODE, fast: bool = FAST,
    div_bc: int = DIV_BC,
) -> None:
    """AdamW (TE mode 1, bias correction) over the shard: ``p``, ``m``, ``v`` updated in place from bf16 ``grads``.

    Args:
        grads: bf16 shard, ``n`` elements (cleaned and scaled by ``inv_scale[0]`` in registers; NOT written).
        p: f32 master weights, ``n``.
        m: f32 first moment, ``n``.
        v: f32 second moment, ``n``.
        lr: f32 ``[1]`` device scalar.
        step: int32 ``[1]`` device scalar (1-indexed).
        inv_scale: f32 ``[1]`` device scalar (clip coefficient / world size).
        noop: int32 ``[1]``; ``1`` skips the update (TE's flag).
        bc_table: From :func:`build_bc_table`; read only when ``bc_mode == 1`` (pass any f32 tensor otherwise).
        beta1: As TE.
        beta2: As TE.
        eps: As TE.
        weight_decay: As TE.
        pattern: :data:`PATTERN`.
        bc_mode: :data:`BC_MODE`.
        fast: :data:`FAST`.
        div_bc: :data:`DIV_BC`.

    Raises:
        ValueError: On a dtype / layout / shape mismatch.
    """
    n = grads.numel()
    _check(grads, torch.bfloat16, "grads")
    for t, name in ((p, "p"), (m, "m"), (v, "v")):
        _check(t, torch.float32, name, n)
    _check(lr, torch.float32, "lr", 1, 4)
    _check(step, torch.int32, "step", 1, 4)
    _check(inv_scale, torch.float32, "inv_scale", 1, 4)
    _check(noop, torch.int32, "noop", 1, 4)
    _check(bc_table, torch.float32, "bc_table", align=4)
    if bc_mode == 1 and bc_table.numel() % 2:
        raise ValueError("bc_table must hold both betas back to back")
    grid = (n + ADAM_BLOCK_ELTS - 1) // ADAM_BLOCK_ELTS
    if grid > 2**31 - 1:
        raise ValueError("grid overflow")
    if grid == 0:
        return
    dev = grads.device.index
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile_adam(dev, tuple(pattern), int(bc_mode), bool(fast), int(div_bc))(
        grads, p, m, v, lr, step, inv_scale, noop, bc_table, n, grid,
        float(beta1), float(beta2), float(eps), float(weight_decay), stream)


def prebuild(device_index: int) -> None:
    """Compile the AdamW shard kernel ahead of ``run_start``, without touching any real state.

    ``_prebuild_kernels`` in ``train_flat.py`` runs a synthetic forward/backward and an eval forward,
    which covers every kernel except this one: the optimizer apply is never reached, so
    ``adamw_ctm_host`` compiled inside the MLPerf timing region (~0.3 s on every rank). This compiles
    against fake tensors -- no parameter, moment or step state is read or written.

    Args:
        device_index: CUDA device ordinal to compile for.
    """
    _compile_adam(device_index, tuple(PATTERN), int(BC_MODE), bool(FAST), int(DIV_BC))


def live_kernels() -> tuple[str, str, str]:
    """``(norm partial, norm cleanup, adam)`` kernel-symbol substrings, for the by-name launch census."""
    return NORM_PARTIAL_KERNEL, NORM_CLEANUP_KERNEL, ADAM_KERNEL


# ---------------------------------------------------------------------------
# torch.library boundary (op records in the dispatch census; graph-safe: nothing allocated, no host sync)
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")
_lib.define(
    "grad_norm_sq_ctm(Tensor grads, Tensor chunk_base, Tensor chunk_len, Tensor(a!) partial,"
    " Tensor(b!) norm_sq, Tensor(c!) noop) -> ()"
)
_lib.define(
    "adamw_ctm(Tensor grads, Tensor(a!) p, Tensor(b!) m, Tensor(c!) v, Tensor lr, Tensor step,"
    " Tensor inv_scale, Tensor noop, Tensor bc_table, float beta1, float beta2, float eps, float weight_decay) -> ()"
)


@torch.library.impl(_lib, "grad_norm_sq_ctm", "CUDA")
def _grad_norm_sq_ctm_impl(grads, chunk_base, chunk_len, partial, norm_sq, noop) -> None:
    try:
        grad_norm_sq_ctm(grads, chunk_base, chunk_len, partial, norm_sq, noop)
    except ValueError as e:
        raise RuntimeError(f"grad_norm_sq_ctm: {e}") from e


@torch.library.impl(_lib, "adamw_ctm", "CUDA")
def _adamw_ctm_impl(grads, p, m, v, lr, step, inv_scale, noop, bc_table, beta1, beta2, eps, weight_decay) -> None:
    try:
        adamw_ctm(grads, p, m, v, lr, step, inv_scale, noop, bc_table, beta1, beta2, eps, weight_decay)
    except ValueError as e:
        raise RuntimeError(f"adamw_ctm: {e}") from e
