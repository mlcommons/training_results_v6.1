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

"""The PyTorch ``aten`` elementwise tail of the model step on the CUTLASS DSL.

Six small kernels that replace, by graph adjacency, the 549 ``at::native`` launches
(and TE's ``silu``/``dsilu``) the production CUDA graph still carries at every
micro-batch:

* ``flux.bias_colsum_ctm(x, out)`` -- ``out[n] = bf16(sum_m x[m, n])``, fp32 in a
  FIXED order, written straight into the ``param.grad`` view (no ``aten.sum``, no
  follow-up ``copy_``). Tier S (``M <= 32``: one launch) and tier L (``M`` in the
  thousands: split-M partials + a fixed-order finalise, two launches).
* ``flux.silu_ctm(x)`` / ``flux.dsilu_accum_ctm(acc, g0, x0[, g1, x1])`` -- TE's
  ``silu`` / ``dsilu`` formulas verbatim (``math.h``: ``s = 1/(1+expf(-x))``,
  ``silu = x*s``, ``dsilu = x*(s*(1-s)) + s``), fp32, one bf16 round per TE op, then
  the ``aten.add`` chain's roundings kept exactly (``bf16(t0 + t1)``, ``bf16(acc + s)``).
* ``flux.qknorm_grad_fin_ctm(sq, sk, gq, gk)`` -- the two ``[128]`` fp32 QK-norm
  weight-gradient accumulators to their bf16 ``param.grad`` views (RNE, bitwise
  ``aten copy_``) and re-zeroed for the next block, one launch for four.
* ``flux.pad_join_ctm(x, l1)`` -- ``cat([zeros([B, l1, D]), x], 1)`` in one
  16-byte-vectorised pass (the SSB -> DSB boundary; the split half is the shipped
  ``flux.seq_split_ctm3``).
* ``flux.timestep_embed_ctm(t, 256)`` -- the sinusoidal timestep embedding,
  aten's nine launches as one, same fp32 op order, RNE to bf16.
* ``flux.mse_loss_fwd_bwd_ctm(pred, target, loss_out)`` -- the loss value (fixed
  order, into the static ``_loss_buf``) and ``grad = bf16(norm * (p - t))`` as
  ``aten.mse_loss_backward`` computes it, two launches for the nine of the
  ``pred.float() / mse_loss / mean / mse_loss_backward / .to(bf16)`` chain.
* ``flux.zero_f32_ctm(t)`` -- ``t.zero_()`` for a contiguous fp32 tensor in 16-byte
  stores (the amax arena's one block fill per step, ``amax_arena.py``,
  was the last ``at::native`` launch in the model-side graph).

Plain ``cute.arch.*`` throughout -- every kernel is latency-bound elementwise work or
a <= 32-row reduction; there is no pipeline, TMA or MMA to hide anything behind
(``flat/custom_ops/fused_sum_cat_ctm2.py`` is the reduce template,
``weight_amax_copy_ctm.py`` the RNE-pack template, ``seq_split_join_ctm3.py`` the
copy template). Public ``nvidia-cutlass-dsl`` only. ``FLAT_ATEN_TAIL_CTM=0`` keeps the
aten path at every site (the in-situ baseline arm); the import itself never falls
back.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from flat.custom_ops.weight_amax_copy_ctm import _pack_bf16x2_rn

#: ``FLAT_ATEN_TAIL_CTM=0`` -> every site keeps its aten / TE predecessor (baseline arm).
ENABLED = os.environ.get("FLAT_ATEN_TAIL_CTM", "1") == "1"

VEC = 8            # bf16 elements per 16-byte access
THREADS = 256
_INT32_MAX = 2**31 - 1
_F32 = cutlass.Float32
_BF16 = cutlass.BFloat16
_I32 = cutlass.Int32
_I64 = cutlass.Int64

#: Kernel-symbol substrings for the by-name launch census.
KERNELS = {
    "colsum": "colsum_ctm_kernel",
    "silu": "silu_ctm_kernel",
    "dsilu": "dsilu_accum_ctm_kernel",
    "qknorm": "qknorm_grad_fin_ctm_kernel",
    "pad_join": "pad_join_ctm_kernel",
    "timestep": "timestep_embed_ctm_kernel",
    "mse": "mse_fwd_bwd_ctm_kernel",
    "mse_fin": "mse_fin_ctm_kernel",
    "add": "add_ctm_kernel",
    "zero": "zero_f32_ctm_kernel",
}

_compiled: dict[tuple, Callable] = {}


def _stream(dev: int) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)


def _fake(dtype, div: int = VEC, align: int = 16, symbol: str = "n"):
    return make_fake_compact_tensor(
        dtype, (cute.sym_int64(divisibility=div, symbol=symbol),), assumed_align=align)


def _bf16x8_to_words(v):
    """Eight fp32 -> four packed ``bf16x2`` words, ``cvt.rn.bf16x2.f32`` (aten's RNE)."""
    return [_pack_bf16x2_rn(v[2 * i], v[2 * i + 1]) for i in range(4)]


def _store_words(p_i32, off_words, words, ALIGN16: bool) -> None:
    """Four Int32 words at ``p_i32 + off_words``: one 16 B store, or four 4 B stores."""
    if cutlass.const_expr(ALIGN16):
        (p_i32 + off_words).store(cutlass.Vector.from_elements(tuple(words), _I32), alignment=16)
    else:
        for i in range(4):
            (p_i32 + off_words + i).store(words[i])


# ---------------------------------------------------------------------------
# K1: column sum -> bf16 param.grad
# ---------------------------------------------------------------------------

@cute.kernel
def colsum_ctm_kernel(
    inp: cute.Tensor, out: cute.Tensor,
    N: _I32, rows_per_thread: _I32, rows_per_split: _I32,
    IN_F32: cutlass.Constexpr[bool], OUT_F32: cutlass.Constexpr[bool],
    DT: cutlass.Constexpr[int], TS: cutlass.Constexpr[int], U: cutlass.Constexpr[int],
    ALIGN16: cutlass.Constexpr[bool],
) -> None:
    """CTA ``(bidx, bidy)`` = column tile ``bidx`` (``DT`` columns) of row split ``bidy``.

    ``TS`` slices of ``DT/8`` threads; slice ``ts`` owns rows ``ts, ts+TS, ...`` of its
    split, ``rows_per_thread`` each, 8 columns per thread, fp32 accumulation; the slices
    are reduced through shared memory ascending -- a fixed order, so the output is
    bitwise run-to-run and under graph replay. ``IN_F32`` reads the fp32 partials of a
    previous launch; ``OUT_F32`` writes fp32 partials at ``out[bidy, :]`` instead of the
    rounded bf16 result.
    """
    TD: cutlass.Constexpr[int] = DT // VEC
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    td = tidx % TD
    ts = tidx // TD
    col0 = bidx * DT + td * VEC
    row0 = bidy * rows_per_split + ts
    stride = _I64(TS) * _I64(N)
    first = _I64(row0) * _I64(N) + _I64(col0)

    acc = [_F32(0.0) for _ in range(VEC)]
    if cutlass.const_expr(IN_F32):
        p = cutlass.Pointer(inp.iterator, dtype=_F32) + first
        for k0 in cutlass.range(0, rows_per_thread, U):
            vs = []
            for u in cutlass.range_constexpr(U):
                q = p + (k0 + u) * stride
                vs.append((q.load(16, count=4), (q + 4).load(16, count=4)))
            for u in cutlass.range_constexpr(U):
                lo, hi = vs[u]
                acc = [acc[i] + lo[i] for i in range(4)] + [acc[4 + i] + hi[i] for i in range(4)]
    else:
        p = cutlass.Pointer(inp.iterator, dtype=_BF16) + first
        for k0 in cutlass.range(0, rows_per_thread, U):
            vs = []
            for u in cutlass.range_constexpr(U):
                vs.append((p + (k0 + u) * stride).load(16, count=VEC))
            for u in cutlass.range_constexpr(U):
                v = vs[u]
                acc = [acc[i] + v[i].to(_F32) for i in range(VEC)]

    red = cutlass.Pointer(cute.arch.alloc_smem(_F32, TS * DT, 16), dtype=_F32)
    (red + (ts * DT + td * VEC)).store(
        cutlass.Vector.from_elements(tuple(acc[:4]), _F32), alignment=16)
    (red + (ts * DT + td * VEC + 4)).store(
        cutlass.Vector.from_elements(tuple(acc[4:]), _F32), alignment=16)
    cute.arch.sync_threads()
    if tidx < TD:
        tot = [_F32(0.0) for _ in range(VEC)]
        for j in cutlass.range_constexpr(TS):
            lo = (red + (j * DT + td * VEC)).load(16, count=4)
            hi = (red + (j * DT + td * VEC + 4)).load(16, count=4)
            tot = [tot[i] + lo[i] for i in range(4)] + [tot[4 + i] + hi[i] for i in range(4)]
        if cutlass.const_expr(OUT_F32):
            op = cutlass.Pointer(out.iterator, dtype=_F32) + (_I64(bidy) * _I64(N) + _I64(col0))
            op.store(cutlass.Vector.from_elements(tuple(tot[:4]), _F32), alignment=16)
            (op + 4).store(cutlass.Vector.from_elements(tuple(tot[4:]), _F32), alignment=16)
        else:
            ow = cutlass.Pointer(out.iterator, dtype=_I32)
            _store_words(ow, col0 // 2, _bf16x8_to_words(tot), ALIGN16)


@cute.jit
def colsum_ctm_host(
    inp: cute.Tensor, out: cute.Tensor, N: _I32, rows_per_thread: _I32, rows_per_split: _I32,
    grid_x: _I32, grid_y: _I32, stream: cuda.CUstream,
    IN_F32: cutlass.Constexpr[bool], OUT_F32: cutlass.Constexpr[bool],
    DT: cutlass.Constexpr[int], TS: cutlass.Constexpr[int], U: cutlass.Constexpr[int],
    ALIGN16: cutlass.Constexpr[bool],
) -> None:
    colsum_ctm_kernel(inp, out, N, rows_per_thread, rows_per_split,
                      IN_F32, OUT_F32, DT, TS, U, ALIGN16).launch(
        grid=(grid_x, grid_y, 1), block=((DT // VEC) * TS, 1, 1), stream=stream)


COLSUM_DT = 64          # columns per CTA: 8 threads x 8 columns per slice
COLSUM_ROWS_L = 256     # rows per CTA in tier L (TS = 32 slices x 8 rows) at M <= 8192
COLSUM_MAX_SPLITS = 32  # partials the finaliser reduces; above 32 x 256 rows each split walks M / 32 rows
_partials: dict[tuple, torch.Tensor] = {}


def _compile_colsum(dev: int, in_f32: bool, out_f32: bool, ts: int, u: int, align16: bool) -> Callable:
    key = ("colsum", dev, in_f32, out_f32, ts, u, align16)
    if key not in _compiled:
        with torch.cuda.device(dev):
            fin = _fake(_F32 if in_f32 else _BF16, symbol="i")
            fout = _fake(_F32, symbol="o") if out_f32 else _fake(_BF16, 2, 16 if align16 else 4, "o")
            _compiled[key] = cute.compile(
                colsum_ctm_host, fin, fout, _I32(0), _I32(0), _I32(0), _I32(0), _I32(0),
                make_fake_stream(), in_f32, out_f32, COLSUM_DT, ts, u, align16,
                options="--enable-tvm-ffi")
    return _compiled[key]


def _unroll(rows: int) -> int:
    for u in (8, 4, 2, 1):
        if rows % u == 0:
            return u
    return 1


def colsum_tier(M: int) -> str:
    """``"S"`` (one launch: ``M <= 32``, or ``M < 256`` with ``M % 32 == 0`` -- the
    modulation bias gradients at local batch 64) or ``"L"`` (split-M partials + finalise, ``M % 256 == 0``)."""
    return "S" if M <= 32 or (M < COLSUM_ROWS_L and M % 32 == 0) else "L"


def _check_out(out: torch.Tensor, n: int, name: str) -> bool:
    if out.dtype != torch.bfloat16 or not out.is_contiguous() or out.numel() != n:
        raise ValueError(f"{name}: out must be a contiguous bf16 tensor of {n} elements")
    if out.data_ptr() % 4 != 0:
        raise ValueError(f"{name}: out must be at least 4-byte aligned")
    return out.data_ptr() % 16 == 0


def bias_colsum_ctm(x: torch.Tensor, out: torch.Tensor) -> None:
    """``out[n] = bf16(sum_m x[m, n])`` in fp32, fixed order, into ``out`` in place.

    Args:
        x: bf16 CUDA ``[M, N]`` (or ``[..., N]``, reshaped), contiguous, 16-byte aligned;
            ``N % 64 == 0``; ``M % 4 == 0`` with ``M <= 32``, or ``M % 32 == 0`` with ``M < 256``
            (tier S), or ``M % 256 == 0`` (tier L; above ``M = 8192`` also ``M % 1024 == 0``).
        out: bf16 ``[N]``-element contiguous tensor (a ``param.grad`` view), written.

    Raises:
        ValueError: On a dtype, layout, alignment or shape the kernel does not take.
    """
    if not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        raise ValueError("bias_colsum_ctm: x must be a contiguous, 16-byte aligned bf16 CUDA tensor")
    N = x.shape[-1]
    M = x.numel() // N
    if N % COLSUM_DT or M * N > _INT32_MAX:
        raise ValueError(f"bias_colsum_ctm: N={N} must be a multiple of {COLSUM_DT}")
    align16 = _check_out(out, N, "bias_colsum_ctm")
    dev = x.device.index
    xf = x.view(-1)
    if colsum_tier(M) == "S":
        if M % 4:
            raise ValueError(f"bias_colsum_ctm: M={M} must be a multiple of 4")
        ts = min(M, 32)
        rows = M // ts
        _compile_colsum(dev, False, False, ts, _unroll(rows), align16)(
            xf, out.view(-1), N, rows, M, N // COLSUM_DT, 1, _stream(dev))
        return
    # Above COLSUM_MAX_SPLITS x 256 rows (M > 8192 -- the mb-64 stem bias gradients,
    # M = 16384) the split count stays 32 and each split walks M / 32 rows (16 per thread at
    # 16384); at and below 8192 the geometry is byte-for-byte the original (256 rows per split).
    splits = min(M // COLSUM_ROWS_L, COLSUM_MAX_SPLITS)
    rows_per_split = M // splits if splits else 0
    if M % COLSUM_ROWS_L or rows_per_split % 32:
        raise ValueError(f"bias_colsum_ctm: M={M} must be a multiple of {COLSUM_ROWS_L} "
                         f"and of {32 * COLSUM_MAX_SPLITS} above {COLSUM_ROWS_L * COLSUM_MAX_SPLITS}")
    key = (dev, splits, N)
    ws = _partials.get(key)
    if ws is None:
        ws = torch.empty(splits * N, dtype=torch.float32, device=x.device)
        _partials[key] = ws
    rows = rows_per_split // 32
    _compile_colsum(dev, False, True, 32, _unroll(rows), True)(
        xf, ws, N, rows, rows_per_split, N // COLSUM_DT, splits, _stream(dev))
    _compile_colsum(dev, True, False, splits, 1, align16)(
        ws, out.view(-1), N, 1, splits, N // COLSUM_DT, 1, _stream(dev))


# ---------------------------------------------------------------------------
# K2: TE's silu / dsilu, and the dsilu + accumulate chain
# ---------------------------------------------------------------------------

def _sigmoid(x):
    """TE ``math.h`` ``sigmoid``: ``1.f / (1.f + expf(-x))`` (precise ``expf``, IEEE divide)."""
    return _F32(1.0) / (_F32(1.0) + cute.math.exp(-x))


def _silu(x):
    """TE ``silu``: ``x * sigmoid(x)``."""
    return x * _sigmoid(x)


def _dsilu(x):
    """TE ``dsilu``: ``x * dsigmoid(x) + sigmoid(x)`` with ``dsigmoid = s * (1 - s)``;
    nvcc contracts the ``a*b + c`` into one fma (``--fmad=true`` default)."""
    s = _sigmoid(x)
    return cute.math.fma(x, s * (_F32(1.0) - s), s)


def _bf16_round(x):
    """fp32 -> bf16 (RNE) -> fp32: one rounding, exactly as a stored bf16 re-read.

    Both ``cvt`` operands must be runtime values (a constant becomes an ``n`` immediate the
    inline-PTX constraint rejects), so ``x`` is packed twice
    and the low half is taken.
    """
    w = _pack_bf16x2_rn(x, x)
    return (w << 16).bitcast(_F32)


@cute.kernel
def silu_ctm_kernel(x: cute.Tensor, out: cute.Tensor, n: _I32) -> None:
    """``out = bf16(silu(x))``, eight elements per thread."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * THREADS + tidx) * VEC
    if base < n:
        v = (cutlass.Pointer(x.iterator, dtype=_BF16) + base).load(16, count=VEC)
        ys = [_silu(v[i].to(_F32)) for i in range(VEC)]
        _store_words(cutlass.Pointer(out.iterator, dtype=_I32), base // 2, _bf16x8_to_words(ys), True)


@cute.kernel
def dsilu_accum_ctm_kernel(
    acc: cute.Tensor, g0: cute.Tensor, x0: cute.Tensor, g1: cute.Tensor, x1: cute.Tensor,
    out: cute.Tensor, n: _I32,
    NTERMS: cutlass.Constexpr[int], ACCUM: cutlass.Constexpr[bool], ALIGN16: cutlass.Constexpr[bool],
) -> None:
    """``t_k = bf16(dsilu(x_k) * g_k)`` (TE's ``unary_grad_kernel``), ``s = bf16(t0 + t1)`` when
    ``NTERMS == 2``, then ``out = bf16(acc + s)`` (``ACCUM``) or ``out = s``. Every rounding of the
    TE + ``aten.add`` chain is kept, so the bytes are the chain's given equal ``dsilu``."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * THREADS + tidx) * VEC
    if base < n:
        vg0 = (cutlass.Pointer(g0.iterator, dtype=_BF16) + base).load(16, count=VEC)
        vx0 = (cutlass.Pointer(x0.iterator, dtype=_BF16) + base).load(16, count=VEC)
        s = [_bf16_round(_dsilu(vx0[i].to(_F32)) * vg0[i].to(_F32)) for i in range(VEC)]
        if cutlass.const_expr(NTERMS == 2):
            vg1 = (cutlass.Pointer(g1.iterator, dtype=_BF16) + base).load(16, count=VEC)
            vx1 = (cutlass.Pointer(x1.iterator, dtype=_BF16) + base).load(16, count=VEC)
            t1 = [_bf16_round(_dsilu(vx1[i].to(_F32)) * vg1[i].to(_F32)) for i in range(VEC)]
            s = [_bf16_round(s[i] + t1[i]) for i in range(VEC)]
        if cutlass.const_expr(ACCUM):
            va = (cutlass.Pointer(acc.iterator, dtype=_BF16) + base).load(16, count=VEC)
            s = [va[i].to(_F32) + s[i] for i in range(VEC)]
        _store_words(cutlass.Pointer(out.iterator, dtype=_I32), base // 2, _bf16x8_to_words(s), ALIGN16)


@cute.jit
def silu_ctm_host(x: cute.Tensor, out: cute.Tensor, n: _I32, grid_x: _I32, stream: cuda.CUstream) -> None:
    silu_ctm_kernel(x, out, n).launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def dsilu_accum_ctm_host(
    acc: cute.Tensor, g0: cute.Tensor, x0: cute.Tensor, g1: cute.Tensor, x1: cute.Tensor,
    out: cute.Tensor, n: _I32, grid_x: _I32, stream: cuda.CUstream,
    NTERMS: cutlass.Constexpr[int], ACCUM: cutlass.Constexpr[bool], ALIGN16: cutlass.Constexpr[bool],
) -> None:
    dsilu_accum_ctm_kernel(acc, g0, x0, g1, x1, out, n, NTERMS, ACCUM, ALIGN16).launch(
        grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.kernel
def add_ctm_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, n: _I32) -> None:
    """``out = bf16(f32(a) + f32(b))`` -- ``aten.add`` on two bf16 tensors, eight per thread."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * THREADS + tidx) * VEC
    if base < n:
        va = (cutlass.Pointer(a.iterator, dtype=_BF16) + base).load(16, count=VEC)
        vb = (cutlass.Pointer(b.iterator, dtype=_BF16) + base).load(16, count=VEC)
        s = [va[i].to(_F32) + vb[i].to(_F32) for i in range(VEC)]
        _store_words(cutlass.Pointer(out.iterator, dtype=_I32), base // 2, _bf16x8_to_words(s), True)


@cute.jit
def add_ctm_host(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, n: _I32, grid_x: _I32,
                 stream: cuda.CUstream) -> None:
    add_ctm_kernel(a, b, out, n).launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


def add_ctm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``aten.add(a, b)`` for two bf16 tensors of one shape; a fresh tensor."""
    _check_vec(a, "add_ctm")
    _check_vec(b, "add_ctm", a)
    dev = a.device.index
    key = ("add", dev)
    if key not in _compiled:
        with torch.cuda.device(dev):
            _compiled[key] = cute.compile(add_ctm_host, _fake(_BF16, symbol="a"), _fake(_BF16, symbol="b"),
                                          _fake(_BF16, symbol="o"), _I32(0), _I32(0), make_fake_stream(),
                                          options="--enable-tvm-ffi")
    out = torch.empty_like(a)
    n = a.numel()
    _compiled[key](a.view(-1), b.view(-1), out.view(-1), n, (n + THREADS * VEC - 1) // (THREADS * VEC), _stream(dev))
    return out


def _check_vec(t: torch.Tensor, name: str, like: torch.Tensor | None = None) -> None:
    if not t.is_cuda or t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16:
        raise ValueError(f"{name}: expected a contiguous, 16-byte aligned bf16 CUDA tensor")
    if t.numel() % VEC or t.numel() > _INT32_MAX:
        raise ValueError(f"{name}: numel must be a multiple of {VEC}")
    if like is not None and (t.shape != like.shape or t.device != like.device):
        raise ValueError(f"{name}: operands must share shape and device")


def silu_ctm(x: torch.Tensor) -> torch.Tensor:
    """``bf16(x * sigmoid(x))`` with TE's formula; a fresh tensor like ``x``."""
    _check_vec(x, "silu_ctm")
    dev = x.device.index
    key = ("silu", dev)
    if key not in _compiled:
        with torch.cuda.device(dev):
            _compiled[key] = cute.compile(silu_ctm_host, _fake(_BF16, symbol="x"), _fake(_BF16, symbol="o"),
                                          _I32(0), _I32(0), make_fake_stream(), options="--enable-tvm-ffi")
    out = torch.empty_like(x)
    n = x.numel()
    _compiled[key](x.view(-1), out.view(-1), n, (n + THREADS * VEC - 1) // (THREADS * VEC), _stream(dev))
    return out


def dsilu_accum_ctm(
    acc: torch.Tensor | None, g0: torch.Tensor, x0: torch.Tensor,
    g1: torch.Tensor | None = None, x1: torch.Tensor | None = None,
) -> torch.Tensor:
    """``dsilu`` of one or two ``(grad, input)`` pairs, summed as the aten chain sums them.

    Args:
        acc: bf16 accumulator, updated IN PLACE (``acc = bf16(acc + s)``) and returned; or
            ``None`` for a fresh ``s``.
        g0: Gradient of the first SiLU output, bf16.
        x0: Its pre-activation input, bf16, same shape.
        g1: Optional second gradient; with ``x1`` gives ``s = bf16(t0 + t1)``.
        x1: Its input.

    Returns:
        ``acc`` (in place) or the fresh result.
    """
    _check_vec(g0, "dsilu_accum_ctm")
    _check_vec(x0, "dsilu_accum_ctm", g0)
    nterms = 1
    if g1 is not None or x1 is not None:
        if g1 is None or x1 is None:
            raise ValueError("dsilu_accum_ctm: g1 and x1 come together")
        _check_vec(g1, "dsilu_accum_ctm", g0)
        _check_vec(x1, "dsilu_accum_ctm", g0)
        nterms = 2
    else:
        g1, x1 = g0, x0   # unused slots, never dereferenced
    if acc is None:
        out = torch.empty_like(g0)
        accum, align16 = False, True
        acc = out
    else:
        _check_vec(acc, "dsilu_accum_ctm", g0)
        out = acc
        accum, align16 = True, True
    dev = g0.device.index
    key = ("dsilu", dev, nterms, accum, align16)
    if key not in _compiled:
        with torch.cuda.device(dev):
            fakes = [_fake(_BF16, symbol=s) for s in ("a", "g0", "x0", "g1", "x1", "o")]
            _compiled[key] = cute.compile(dsilu_accum_ctm_host, *fakes, _I32(0), _I32(0), make_fake_stream(),
                                          nterms, accum, align16, options="--enable-tvm-ffi")
    n = g0.numel()
    _compiled[key](acc.view(-1), g0.view(-1), x0.view(-1), g1.view(-1), x1.view(-1), out.view(-1),
                   n, (n + THREADS * VEC - 1) // (THREADS * VEC), _stream(dev))
    return out


# ---------------------------------------------------------------------------
# K3: QK-norm weight-gradient scratch -> bf16 grads, re-zeroed
# ---------------------------------------------------------------------------

QKNORM_D = 128


@cute.kernel
def qknorm_grad_fin_ctm_kernel(
    sq: cute.Tensor, sk: cute.Tensor, gq: cute.Tensor, gk: cute.Tensor,
) -> None:
    """64 threads: lane ``t < 32`` takes ``sq[4t..4t+4)`` -> ``gq``, ``t >= 32`` the ``k`` pair;
    RNE to bf16 (two 4 B word stores), then the four fp32 scratch slots are zeroed."""
    tidx, _, _ = cute.arch.thread_idx()
    zero4 = cutlass.Vector.from_elements((_F32(0.0), _F32(0.0), _F32(0.0), _F32(0.0)), _F32)
    if tidx < 32:
        sp = cutlass.Pointer(sq.iterator, dtype=_F32) + tidx * 4
        gp = cutlass.Pointer(gq.iterator, dtype=_I32) + tidx * 2
        v = sp.load(16, count=4)
        gp.store(_pack_bf16x2_rn(v[0], v[1]))
        (gp + 1).store(_pack_bf16x2_rn(v[2], v[3]))
        sp.store(zero4, alignment=16)
    else:
        sp = cutlass.Pointer(sk.iterator, dtype=_F32) + (tidx - 32) * 4
        gp = cutlass.Pointer(gk.iterator, dtype=_I32) + (tidx - 32) * 2
        v = sp.load(16, count=4)
        gp.store(_pack_bf16x2_rn(v[0], v[1]))
        (gp + 1).store(_pack_bf16x2_rn(v[2], v[3]))
        sp.store(zero4, alignment=16)


@cute.jit
def qknorm_grad_fin_ctm_host(
    sq: cute.Tensor, sk: cute.Tensor, gq: cute.Tensor, gk: cute.Tensor, stream: cuda.CUstream,
) -> None:
    qknorm_grad_fin_ctm_kernel(sq, sk, gq, gk).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def qknorm_grad_fin_ctm(sq: torch.Tensor, sk: torch.Tensor, gq: torch.Tensor, gk: torch.Tensor) -> None:
    """``gq = bf16(sq); gk = bf16(sk); sq[:] = sk[:] = 0`` -- one launch.

    Args:
        sq: fp32 ``[128]`` contiguous accumulator (query-norm weight grad), zeroed on return.
        sk: fp32 ``[128]`` accumulator (key-norm weight grad), zeroed on return.
        gq: bf16 ``[128]`` ``param.grad`` view, 4-byte aligned, written.
        gk: bf16 ``[128]`` ``param.grad`` view, written.
    """
    for t, nm in ((sq, "sq"), (sk, "sk")):
        if not t.is_cuda or t.dtype != torch.float32 or not t.is_contiguous() or t.numel() != QKNORM_D or t.data_ptr() % 16:
            raise ValueError(f"qknorm_grad_fin_ctm: {nm} must be a contiguous, 16-byte aligned fp32 [128] CUDA tensor")
    for t, nm in ((gq, "gq"), (gk, "gk")):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.numel() != QKNORM_D or t.data_ptr() % 4:
            raise ValueError(f"qknorm_grad_fin_ctm: {nm} must be a contiguous, 4-byte aligned bf16 [128] tensor")
    dev = sq.device.index
    key = ("qknorm", dev)
    if key not in _compiled:
        with torch.cuda.device(dev):
            _compiled[key] = cute.compile(
                qknorm_grad_fin_ctm_host, _fake(_F32, 4, 16, "sq"), _fake(_F32, 4, 16, "sk"),
                _fake(_BF16, 2, 4, "gq"), _fake(_BF16, 2, 4, "gk"), make_fake_stream(),
                options="--enable-tvm-ffi")
    _compiled[key](sq.view(-1), sk.view(-1), gq.view(-1), gk.view(-1), _stream(dev))


# ---------------------------------------------------------------------------
# K4b: zero-padded join on dim 1
# ---------------------------------------------------------------------------

PAD_UNROLL = 4
PAD_TILE_ELTS = THREADS * PAD_UNROLL * VEC   # 8192 bf16 = 16 KB per tile


def _copy_vecs(src, s_vec, dst, d_vec) -> None:
    """``PAD_UNROLL`` 16 B loads for this thread, all issued before the first store (a plain
    helper, so the list lives in one scope -- ``seq_split_join_ctm3._copy_tile``'s idiom)."""
    vs = []
    for u in range(PAD_UNROLL):
        vs.append((src + (s_vec + u * THREADS) * VEC).load(16, count=VEC))
    for u in range(PAD_UNROLL):
        (dst + (d_vec + u * THREADS) * VEC).store(vs[u], alignment=16)


def _zero_vecs(dst_words, o_vec) -> None:
    zero = cutlass.Vector.from_elements((_I32(0), _I32(0), _I32(0), _I32(0)), _I32)
    for u in range(PAD_UNROLL):
        (dst_words + (o_vec + u * THREADS) * (VEC // 2)).store(zero, alignment=16)


@cute.kernel
def pad_join_ctm_kernel(
    x: cute.Tensor, out: cute.Tensor, tiles_total: _I32, tiles_1: _I32, n_tiles: _I32,
) -> None:
    """Tile ``t`` of the joined ``[B, L1+L2, D]`` is batch ``t // tiles_total``, row-tile ``r``;
    zeros for ``r < tiles_1``, else a copy of tile ``r - tiles_1`` of ``x[b]``. Thread ``tidx``
    owns 16 B vectors ``tidx + u*256`` of the tile (``PAD_UNROLL`` loads before the first store)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    xp = cutlass.Pointer(x.iterator, dtype=_BF16)
    op = cutlass.Pointer(out.iterator, dtype=_BF16)
    ow = cutlass.Pointer(out.iterator, dtype=_I32)
    tiles_2 = tiles_total - tiles_1
    for t in cutlass.range(bidx, n_tiles, gdim):
        b = t // tiles_total
        r = t - b * tiles_total
        o_vec = t * (THREADS * PAD_UNROLL) + tidx
        if r < tiles_1:
            _zero_vecs(ow, o_vec)
        else:
            s_vec = (b * tiles_2 + (r - tiles_1)) * (THREADS * PAD_UNROLL) + tidx
            _copy_vecs(xp, s_vec, op, o_vec)


@cute.jit
def pad_join_ctm_host(
    x: cute.Tensor, out: cute.Tensor, tiles_total: _I32, tiles_1: _I32, n_tiles: _I32,
    grid_x: _I32, stream: cuda.CUstream,
) -> None:
    pad_join_ctm_kernel(x, out, tiles_total, tiles_1, n_tiles).launch(
        grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


def pad_join_ctm(x: torch.Tensor, l1: int) -> torch.Tensor:
    """``torch.cat([zeros([B, l1, ...]), x], 1)`` in one pass.

    Args:
        x: bf16 CUDA ``[B, L2, ...]`` contiguous, 16-byte aligned; ``l1 * row`` and
            ``L2 * row`` elements must be multiples of 8192 (a 16 KB tile).
        l1: Rows of zeros to prepend.

    Returns:
        bf16 ``[B, l1 + L2, ...]``.
    """
    if x.dim() < 3 or not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr() % 16:
        raise ValueError("pad_join_ctm: x must be a contiguous, 16-byte aligned bf16 CUDA [B, L, ...] tensor")
    B, L2 = x.shape[0], x.shape[1]
    row = math.prod(x.shape[2:])
    if l1 <= 0 or (l1 * row) % PAD_TILE_ELTS or (L2 * row) % PAD_TILE_ELTS:
        raise ValueError(f"pad_join_ctm: l1*row={l1 * row} and L2*row={L2 * row} must be multiples of {PAD_TILE_ELTS}")
    tiles_1 = l1 * row // PAD_TILE_ELTS
    tiles_total = tiles_1 + L2 * row // PAD_TILE_ELTS
    n_tiles = B * tiles_total
    if n_tiles * PAD_TILE_ELTS > _INT32_MAX:
        raise ValueError("pad_join_ctm: tensor too large")
    out = torch.empty(B, l1 + L2, *x.shape[2:], dtype=torch.bfloat16, device=x.device)
    dev = x.device.index
    key = ("pad_join", dev)
    if key not in _compiled:
        with torch.cuda.device(dev):
            _compiled[key] = cute.compile(pad_join_ctm_host, _fake(_BF16, symbol="x"), _fake(_BF16, symbol="o"),
                                          _I32(0), _I32(0), _I32(0), _I32(0), make_fake_stream(),
                                          options="--enable-tvm-ffi")
    _compiled[key](x.view(-1), out.view(-1), tiles_total, tiles_1, n_tiles, n_tiles, _stream(dev))
    return out


# ---------------------------------------------------------------------------
# K6a: sinusoidal timestep embedding
# ---------------------------------------------------------------------------

TS_DIM = 256
TS_HALF = TS_DIM // 2
_TS_TIME_FACTOR = 1000.0
_TS_LOG_COEF = -math.log(10000) / TS_HALF


@cute.kernel
def timestep_embed_ctm_kernel(t: cute.Tensor, out: cute.Tensor) -> None:
    """CTA ``b``, thread ``j < 128``: aten's op order -- ``t_s = bf16(f32(t[b]) * 1000)``
    (``mul.Scalar`` on a bf16 tensor rounds), ``f = expf(j * c)``, ``a = f32(t_s) * f``,
    ``out[b, j] = bf16(cosf(a))``, ``out[b, 128 + j] = bf16(sinf(a))``."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tv = (cutlass.Pointer(t.iterator, dtype=_BF16) + bidx).load()
    ts = _bf16_round(tv.to(_F32) * _F32(_TS_TIME_FACTOR))
    f = cute.math.exp(tidx.to(_F32) * _F32(_TS_LOG_COEF))
    a = ts * f
    cv = cute.math.cos(a)
    sv = cute.math.sin(a)
    c = _pack_bf16x2_rn(cv, cv)   # runtime operands only (see _bf16_round); low half used
    s = _pack_bf16x2_rn(sv, sv)
    op = cutlass.Pointer(out.iterator, dtype=cutlass.Int16) + bidx * TS_DIM
    (op + tidx).store((c & 0xFFFF).to(cutlass.Int16))
    (op + TS_HALF + tidx).store((s & 0xFFFF).to(cutlass.Int16))


@cute.jit
def timestep_embed_ctm_host(t: cute.Tensor, out: cute.Tensor, grid_x: _I32, stream: cuda.CUstream) -> None:
    timestep_embed_ctm_kernel(t, out).launch(grid=(grid_x, 1, 1), block=(TS_HALF, 1, 1), stream=stream)


def timestep_embed_ctm(t: torch.Tensor, dim: int) -> torch.Tensor:
    """``flat.model.FluxModel._timestep_embedding(t, 256)`` for a bf16 ``t`` -- one launch.

    Args:
        t: bf16 CUDA ``[B]`` timesteps.
        dim: Must be 256 (the model's).

    Returns:
        bf16 ``[B, 256]``: ``cat([cos(t*1000*f), sin(t*1000*f)], 1)``.
    """
    if dim != TS_DIM or t.dim() != 1 or not t.is_cuda or t.dtype != torch.bfloat16 or not t.is_contiguous():
        raise ValueError("timestep_embed_ctm: expects a contiguous bf16 CUDA [B] tensor and dim=256")
    dev = t.device.index
    key = ("timestep", dev)
    if key not in _compiled:
        with torch.cuda.device(dev):
            _compiled[key] = cute.compile(timestep_embed_ctm_host, _fake(_BF16, 1, 2, "t"), _fake(_BF16, symbol="o"),
                                          _I32(0), make_fake_stream(), options="--enable-tvm-ffi")
    out = torch.empty(t.shape[0], TS_DIM, dtype=torch.bfloat16, device=t.device)
    _compiled[key](t, out.view(-1), t.shape[0], _stream(dev))
    return out


# ---------------------------------------------------------------------------
# K6b: MSE loss value + gradient
# ---------------------------------------------------------------------------

@cute.kernel
def mse_fwd_bwd_ctm_kernel(
    p_in: cute.Tensor, target: cute.Tensor, grad: cute.Tensor, partial: cute.Tensor,
    n: _I32, norm: _F32,
) -> None:
    """Eight elements per thread (the prediction is ``p_in``: a parameter named ``pred`` is shadowed
    by the DSL's dynamic-if predicate inside the branch): ``d = f32(p) - f32(t)``; ``grad = bf16(norm * d)``
    (``aten.mse_loss_backward`` with ``grad_output = 1``: ``norm * (a - b) * 1``);
    ``sum d*d`` per thread -> smem -> thread 0 sums the 256 slots ascending -> ``partial[bidx]``."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * THREADS + tidx) * VEC
    red = cutlass.Pointer(cute.arch.alloc_smem(_F32, THREADS, 16), dtype=_F32)
    # No value crosses the dynamic `if` (a pre-defined accumulator mutated inside it makes the
    # branch a result-carrying region that mis-flattens the tensor live-ins): each branch
    # computes and stores its own slot.
    if base < n:
        vp = (cutlass.Pointer(p_in.iterator, dtype=_BF16) + base).load(16, count=VEC)
        vt = (cutlass.Pointer(target.iterator, dtype=_BF16) + base).load(16, count=VEC)
        ds = [vp[i].to(_F32) - vt[i].to(_F32) for i in range(VEC)]
        acc = ds[0] * ds[0]
        for i in cutlass.range_constexpr(1, VEC):   # a loop STATEMENT in a kernel body must be range_constexpr
            acc = acc + ds[i] * ds[i]
        _store_words(cutlass.Pointer(grad.iterator, dtype=_I32), base // 2,
                     _bf16x8_to_words([norm * d for d in ds]), True)
        (red + tidx).store(acc)
    else:
        (red + tidx).store(_F32(0.0))
    cute.arch.sync_threads()
    if tidx == 0:
        tot = _F32(0.0)
        for j in cutlass.range_constexpr(THREADS):
            tot = tot + (red + j).load()
        (cutlass.Pointer(partial.iterator, dtype=_F32) + bidx).store(tot)


@cute.kernel
def mse_fin_ctm_kernel(partial: cute.Tensor, loss: cute.Tensor, n_partials: _I32, inv_n: _F32) -> None:
    """One thread: ``loss[0] = (sum_i partial[i]) * inv_n``, ascending."""
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        pp = cutlass.Pointer(partial.iterator, dtype=_F32)
        tot = _F32(0.0)
        for i in cutlass.range(n_partials):
            tot = tot + (pp + i).load()
        cutlass.Pointer(loss.iterator, dtype=_F32).store(tot * inv_n)


@cute.jit
def mse_fwd_bwd_ctm_host(
    p_in: cute.Tensor, target: cute.Tensor, grad: cute.Tensor, partial: cute.Tensor, loss: cute.Tensor,
    n: _I32, norm: _F32, grid_x: _I32, inv_n: _F32, stream: cuda.CUstream,
) -> None:
    mse_fwd_bwd_ctm_kernel(p_in, target, grad, partial, n, norm).launch(
        grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)
    mse_fin_ctm_kernel(partial, loss, grid_x, inv_n).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


_mse_partials: dict[tuple, torch.Tensor] = {}


def mse_loss_fwd_bwd_ctm(pred: torch.Tensor, target: torch.Tensor, loss_out: torch.Tensor) -> torch.Tensor:
    """``loss_out[0] = mean((pred - target)^2)`` (fp32, fixed order) and the bf16 gradient of that
    mean w.r.t. ``pred`` -- ``bf16(f32(2 / N) * (f32(pred) - f32(target)))``.

    Args:
        pred: bf16 CUDA tensor, contiguous, ``numel % 8 == 0``.
        target: bf16, same shape.
        loss_out: fp32 ``[1]`` static buffer, written.

    Returns:
        bf16 gradient, ``pred``'s shape.
    """
    _check_vec(pred, "mse_loss_fwd_bwd_ctm")
    _check_vec(target, "mse_loss_fwd_bwd_ctm", pred)
    if loss_out.dtype != torch.float32 or loss_out.numel() != 1 or not loss_out.is_cuda:
        raise ValueError("mse_loss_fwd_bwd_ctm: loss_out must be a fp32 [1] CUDA tensor")
    n = pred.numel()
    grid = (n + THREADS * VEC - 1) // (THREADS * VEC)
    dev = pred.device.index
    key = (dev, grid)
    ws = _mse_partials.get(key)
    if ws is None:
        ws = torch.empty(grid, dtype=torch.float32, device=pred.device)
        _mse_partials[key] = ws
    ckey = ("mse", dev)
    if ckey not in _compiled:
        with torch.cuda.device(dev):
            _compiled[ckey] = cute.compile(
                mse_fwd_bwd_ctm_host, _fake(_BF16, symbol="p"), _fake(_BF16, symbol="t"), _fake(_BF16, symbol="g"),
                _fake(_F32, 1, 4, "w"), _fake(_F32, 1, 4, "l"), _I32(0), _F32(0.0), _I32(0), _F32(0.0),
                make_fake_stream(), options="--enable-tvm-ffi")
    grad = torch.empty_like(pred)
    # aten: norm = 2. / numel (double) -> scalar_t float; mean = sum / numel.
    _compiled[ckey](pred.view(-1), target.view(-1), grad.view(-1), ws, loss_out.view(-1),
                    n, float(2.0 / n), grid, float(1.0 / n), _stream(dev))
    return grad


# ---------------------------------------------------------------------------
# verify(): the ctm-write-kernel smoke check; the contract is tests/test_aten_tail_ctm.py
# ---------------------------------------------------------------------------

def verify(B: int = 4) -> None:
    """Compile every kernel and check it against its aten / closed-form reference.

    Raises:
        AssertionError: On any mismatch beyond the kernel's stated contract.
    """
    dev = "cuda"
    torch.manual_seed(B)
    # K1 S and L
    for M, N in ((B, 18432), (B, 3072), (B * 256, 3072), (B * 256, 64)):
        x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        out = torch.empty(N, device=dev, dtype=torch.bfloat16)
        bias_colsum_ctm(x, out)
        ref = x.double().sum(0).to(torch.bfloat16)
        assert torch.equal(out, ref) or (out.float() - ref.float()).abs().max() <= 2 ** -7 * ref.float().abs().max(), (M, N)
    # K2
    x = torch.randn(B, 3072, device=dev, dtype=torch.bfloat16)
    g = torch.randn(B, 3072, device=dev, dtype=torch.bfloat16)
    s = silu_ctm(x)
    assert torch.allclose(s.float(), torch.nn.functional.silu(x.float()), atol=2e-2, rtol=2e-2)
    acc = torch.randn(B, 3072, device=dev, dtype=torch.bfloat16)
    acc0 = acc.clone()
    xs = torch.sigmoid(x.float())
    ref = (g.float() * (xs * (1 + x.float() * (1 - xs)))).to(torch.bfloat16)
    dsilu_accum_ctm(acc, g, x)
    assert torch.allclose(acc.float(), (acc0.float() + ref.float()).to(torch.bfloat16).float(), atol=3e-2, rtol=3e-2)
    # K3
    sq, sk = torch.randn(128, device=dev), torch.randn(128, device=dev)
    gq, gk = torch.empty(128, device=dev, dtype=torch.bfloat16), torch.empty(128, device=dev, dtype=torch.bfloat16)
    rq, rk = sq.to(torch.bfloat16), sk.to(torch.bfloat16)
    qknorm_grad_fin_ctm(sq, sk, gq, gk)
    assert torch.equal(gq, rq) and torch.equal(gk, rk) and not sq.any() and not sk.any()
    # K4b
    xi = torch.randn(B, 256, 3072, device=dev, dtype=torch.bfloat16)
    assert torch.equal(pad_join_ctm(xi, 256), torch.cat([torch.zeros_like(xi), xi], 1))
    # K6a
    t = torch.rand(B, device=dev).to(torch.bfloat16)
    e = timestep_embed_ctm(t, 256)
    ts = (t.float() * 1000.0).to(torch.bfloat16).float()
    fr = torch.exp(torch.arange(128, device=dev, dtype=torch.float32) * (-math.log(10000) / 128))
    a = ts[:, None] * fr[None]
    er = torch.cat([torch.cos(a), torch.sin(a)], 1).to(torch.bfloat16)
    assert (e.float() - er.float()).abs().max() <= 2 ** -7, "timestep embedding"
    # K6b
    p = torch.randn(B, 256, 64, device=dev, dtype=torch.bfloat16)
    tg = torch.randn(B, 256, 64, device=dev, dtype=torch.bfloat16)
    lo = torch.zeros(1, device=dev)
    gr = mse_loss_fwd_bwd_ctm(p, tg, lo)
    lref = torch.nn.functional.mse_loss(p.float(), tg.float())
    assert abs(lo.item() - lref.item()) <= 1e-5 * max(1.0, lref.item()), (lo.item(), lref.item())
    gref = torch.ops.aten.mse_loss_backward.default(torch.ones(1, device=dev), p.float(), tg.float(), 1).to(torch.bfloat16)
    assert torch.equal(gr.view(torch.int16), gref.view(torch.int16)), "mse grad"
    torch.cuda.synchronize()
    print(f"aten_tail_ctm.verify B={B}: colsum S/L, silu, dsilu_accum, qknorm_fin, pad_join, timestep, mse ok")


# ---------------------------------------------------------------------------
# K8: zero a contiguous fp32 tensor -- the amax arena's block fill
# ---------------------------------------------------------------------------

@cute.kernel
def zero_f32_ctm_kernel(t: cute.Tensor, n: _I32) -> None:
    """One 16 B zero store per thread (``n % 4 == 0``)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * THREADS + tidx) * 4
    if base < n:
        zero = cutlass.Vector.from_elements((_F32(0.0), _F32(0.0), _F32(0.0), _F32(0.0)), _F32)
        (cutlass.Pointer(t.iterator, dtype=_F32) + base).store(zero, alignment=16)


@cute.jit
def zero_f32_ctm_host(t: cute.Tensor, n: _I32, grid_x: _I32, stream: cuda.CUstream) -> None:
    zero_f32_ctm_kernel(t, n).launch(grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


def zero_f32_ctm(t: torch.Tensor) -> None:
    """``t.zero_()`` for a contiguous, 16 B-aligned fp32 CUDA tensor whose numel is a multiple of 4."""
    if not t.is_cuda or t.dtype != torch.float32 or not t.is_contiguous() or t.data_ptr() % 16 or t.numel() % 4:
        raise ValueError("zero_f32_ctm: expected a contiguous, 16-byte aligned fp32 CUDA tensor with numel % 4 == 0")
    dev = t.device.index
    key = ("zero", dev)
    if key not in _compiled:
        with torch.cuda.device(dev):
            _compiled[key] = cute.compile(zero_f32_ctm_host, _fake(_F32, div=4, symbol="z"), _I32(0), _I32(0),
                                          make_fake_stream(), options="--enable-tvm-ffi")
    n = t.numel()
    _compiled[key](t.view(-1), n, (n + THREADS * 4 - 1) // (THREADS * 4), _stream(dev))


# ---------------------------------------------------------------------------
# torch.library registration
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define("bias_colsum_ctm(Tensor x, Tensor(a!) out) -> ()")
_lib.define("add_ctm(Tensor a, Tensor b) -> Tensor")
_lib.define("fused_sum_cat2_ctm(Tensor t0, Tensor t1) -> Tensor")
_lib.define("silu_ctm(Tensor x) -> Tensor")
_lib.define("dsilu_ctm(Tensor g0, Tensor x0) -> Tensor")
_lib.define("dsilu_accum_ctm(Tensor(a!) acc, Tensor g0, Tensor x0, Tensor? g1, Tensor? x1) -> ()")
_lib.define("qknorm_grad_fin_ctm(Tensor(a!) sq, Tensor(b!) sk, Tensor(c!) gq, Tensor(d!) gk) -> ()")
_lib.define("pad_join_ctm(Tensor x, int l1) -> Tensor")
_lib.define("timestep_embed_ctm(Tensor t, int dim) -> Tensor")
_lib.define("mse_loss_fwd_bwd_ctm(Tensor pred, Tensor target, Tensor(a!) loss_out) -> Tensor")
_lib.define("zero_f32_ctm(Tensor(a!) t) -> ()")


@torch.library.impl(_lib, "bias_colsum_ctm", "CUDA")
def _bias_colsum_impl(x: torch.Tensor, out: torch.Tensor) -> None:
    bias_colsum_ctm(x, out)


@torch.library.impl(_lib, "add_ctm", "CUDA")
def _add_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return add_ctm(a, b)


@torch.library.impl(_lib, "fused_sum_cat2_ctm", "CUDA")
def _fsc2_impl(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    """The shipped ``sum_cat_ctm_kernel`` with two inputs (its ``CONFIGS[DEFAULT_CONFIG]``):
    the LastLayer's two seq-sums and their ``cat`` as one launch."""
    from flat.custom_ops import fused_sum_cat_ctm2 as m
    return m.fused_sum_cat_ctm((t0, t1), *m.CONFIGS[m.DEFAULT_CONFIG])


@torch.library.impl(_lib, "silu_ctm", "CUDA")
def _silu_impl(x: torch.Tensor) -> torch.Tensor:
    return silu_ctm(x)


@torch.library.impl(_lib, "dsilu_ctm", "CUDA")
def _dsilu_impl(g0: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return dsilu_accum_ctm(None, g0, x0)


@torch.library.impl(_lib, "dsilu_accum_ctm", "CUDA")
def _dsilu_accum_impl(acc, g0, x0, g1, x1) -> None:
    dsilu_accum_ctm(acc, g0, x0, g1, x1)


@torch.library.impl(_lib, "qknorm_grad_fin_ctm", "CUDA")
def _qknorm_impl(sq, sk, gq, gk) -> None:
    qknorm_grad_fin_ctm(sq, sk, gq, gk)


@torch.library.impl(_lib, "pad_join_ctm", "CUDA")
def _pad_join_impl(x: torch.Tensor, l1: int) -> torch.Tensor:
    return pad_join_ctm(x, l1)


@torch.library.impl(_lib, "timestep_embed_ctm", "CUDA")
def _timestep_impl(t: torch.Tensor, dim: int) -> torch.Tensor:
    return timestep_embed_ctm(t, dim)


@torch.library.impl(_lib, "mse_loss_fwd_bwd_ctm", "CUDA")
def _mse_impl(pred, target, loss_out) -> torch.Tensor:
    return mse_loss_fwd_bwd_ctm(pred, target, loss_out)


@torch.library.impl(_lib, "zero_f32_ctm", "CUDA")
def _zero_impl(t: torch.Tensor) -> None:
    zero_f32_ctm(t)


if __name__ == "__main__":
    verify()
