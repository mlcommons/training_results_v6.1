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

"""The gated-residual ops of Flux.1 as row-block CUTLASS-DSL kernels, forward and backward
(the backward replaced the Triton ``adaln_modulate.py``).

* ``flux.fused_gate_residual_a2(x, gate, y)          -> x + gate * y``
* ``flux.fused_gate_residual_add_a2(x, gate, y1, y2) -> (x + gate * (y1 + y2), y1 + y2)``
* ``flux.fused_gate_bwd(grad, gate, proj)            -> (grad * gate, grad * proj)`` -- the
  gated-residual backward of the recipes that do not take the fused NVFP4 producer
  (``fp8_ops.py`` MXFP8-only, ``fp8_ops_bf16.py``, and ``nvfp4_ops.py``'s demoted /
  unsupported-width fall-through). Under the shipped R2 recipe it has no launch: every site
  takes ``gate_bwd_amax_ctm``, whose ``out0``/``out1`` are bitwise this op.

``x`` / ``y*`` / ``grad`` / ``proj`` are ``[B, S, D]`` bf16 contiguous; ``gate`` is ``[B, 1, D]``
bf16 broadcast over ``S``, in production a non-contiguous ``split`` view (batch stride 6D or
3D, storage offset 2D).

Structure (no integer division anywhere):

* grid ``(D/8/THREADS, S/U, B)``: ``b = block_idx.z``; a thread's column is
  fixed, so its 8-element gate vector is loaded once per CTA;
* each thread handles ``U`` consecutive rows; all ``2U`` (plain, bwd) / ``3U`` (add)
  16-byte loads are issued before the first store;
* forward: ``fma.rn.bf16x2`` / ``add.rn.bf16x2`` through ``inline_ptx`` -- one rounding of
  the exact sum (``tests/gate_reference.py`` states it in torch);
* backward: fp32 products (exact for two bf16 operands) and one ``cvt.rn.bf16x2.f32`` --
  the producer's arithmetic, bitwise the Triton kernel it replaced.

New here, each a ``Constexpr`` knob so every combination is its own kernel:

* ``HINT`` -- 0: plain ``ld``/``st``. 1: ``createpolicy.fractional.L2::evict_first``
  on the single-use streaming loads (``x``, ``y``, ``y1``, ``y2``) and on the
  ``proj`` stores (dead until the backward), through ``ld/st.global.L2::cache_hint``.
  2: 1 + ``evict_last`` on the ``out`` stores (read by the very next kernel).
  The hinted loads are **non-side-effecting** ``llvm.inline_asm`` (volatile
  hinted loads cost +2-3 us); the policy register is created once per
  thread and may be CSE'd.
* ``MINB`` -- ``min_blocks_per_mp`` with an explicit ``preferred_smem_carveout=0``
  (this kernel has no shared memory); 0 leaves the launch unpinned.

Public ``nvidia-cutlass-dsl`` only (``cute.arch.*``, ``cutlass.Pointer``,
``inline_ptx``, ``llvm.inline_asm``). The forward ops are wired in ``flat/model.py``; the
backward is wired through the recipe seam ``fp8_gate_bwd_quantize``.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T

VEC = 8  # bf16 elements per 16-byte access
WORDS = VEC // 2  # packed bf16x2 words per access
_INT32_MAX = 2**31 - 1
_BF16 = cutlass.BFloat16
_F32 = cutlass.Float32
_I32 = cutlass.Int32
_I64 = cutlass.Int64
_BF16_HI_MASK = -65536  # 0xFFFF0000 as a signed Int32

HINT_NONE, HINT_EF, HINT_EFL = 0, 1, 2

# d = a * b + c, packed bf16x2, one rounding of the exact sum (tests/gate_reference.py).
_FMA_BF16X2 = "fma.rn.bf16x2 {$w0}, {$r0}, {$r1}, {$r2};"
_ADD_BF16X2 = "add.rn.bf16x2 {$w0}, {$r0}, {$r1};"
# two fp32 -> one packed bf16x2 word, RNE, hi in the upper half (the backward's rounding).
_CVT_BF16X2 = "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};"
# L2 eviction-priority policies; the fraction is spelled INSIDE the asm (never a
# float operand on inline asm).
_POLICY_FIRST = "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;"
_POLICY_LAST = "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;"
_LD16_HINT = "ld.global.L2::cache_hint.v4.b32 {$0, $1, $2, $3}, [$4], $5;"
_LD16_HINT_CONS = "=r,=r,=r,=r,l,l"
_ST16_HINT = "st.global.L2::cache_hint.v4.b32 [$0], {$1, $2, $3, $4}, $5;"
_ST16_HINT_CONS = "l,r,r,r,r,l"

with open(__file__, "rb") as _fh:
    SRC_MD5 = hashlib.md5(_fh.read()).hexdigest()
if os.environ.get("A2_QUIET") != "1":
    print(f":::SRC gate_residual_a2 {__file__} md5={SRC_MD5}", flush=True)


# --- traced device helpers (plain Python: no range_constexpr / const_expr here) ------


def _asm(res, args, code, cons, side_effects: bool):
    return llvm.inline_asm(
        res, [a.ir_value() for a in args], code, cons,
        has_side_effects=side_effects, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _policy(last: bool):
    """The per-thread 64-bit L2 policy register (non-volatile: LLVM may CSE it)."""
    return _I64(_asm(T.i64(), [], _POLICY_LAST if last else _POLICY_FIRST, "=l", False))


def _ld(it, ptr, off, pol):
    """One 16-byte load at element offset ``off`` -> list of 4 packed-bf16x2 Int32 words.

    ``it`` is the scalar CuTe iterator (for ``toint()``), ``ptr`` the vector-capable
    ``cutlass.Pointer``; ``pol`` None selects the plain ``.load`` path.
    """
    if pol is None:
        v = (ptr + off).load(16, count=VEC).bitcast(_I32)
        return [v[j] for j in range(WORDS)]
    res = _asm(llvm.StructType.get_literal([T.i32()] * WORDS), [(it + off).toint(), pol],
               _LD16_HINT, _LD16_HINT_CONS, False)
    return [_I32(llvm.extractvalue(T.i32(), res, [j])) for j in range(WORDS)]


def _st(it, ptr, off, words, pol):
    """One 16-byte store of 4 Int32 words at element offset ``off`` (hinted when ``pol``)."""
    if pol is None:
        (ptr + off).store(
            cutlass.Vector.from_elements(tuple(words), _I32).bitcast(_BF16), alignment=16)
    else:
        _asm(None, [(it + off).toint()] + list(words) + [pol], _ST16_HINT, _ST16_HINT_CONS, True)


def _fma2(g, y, x):
    return cute.arch.inline_ptx(_FMA_BF16X2, write_only_types=[_I32], read_only_args=[g, y, x])


def _add2(a, b):
    return cute.arch.inline_ptx(_ADD_BF16X2, write_only_types=[_I32], read_only_args=[a, b])


def _bf16_lo(w):
    return (w << 16).bitcast(_F32)


def _bf16_hi(w):
    return (w & _I32(_BF16_HI_MASK)).bitcast(_F32)


def _mul2(a, b):
    """Packed bf16x2 product: the two fp32 products (exact) then one ``cvt.rn.bf16x2.f32``."""
    return cute.arch.inline_ptx(
        _CVT_BF16X2, write_only_types=[_I32],
        read_only_args=[_bf16_hi(a) * _bf16_hi(b), _bf16_lo(a) * _bf16_lo(b)])


def _policies(hint: int):
    """(load / proj policy, out policy) for a HINT selector; None = plain path."""
    if hint == HINT_NONE:
        return None, None
    ef = _policy(last=False)
    return ef, (_policy(last=True) if hint == HINT_EFL else None)


# --- kernels --------------------------------------------------------------------------


@cute.kernel
def gate_residual_a2_kernel(
    x: cute.Tensor, gate: cute.Tensor, y: cute.Tensor, out: cute.Tensor,
    S: cutlass.Int32, d_vec: cutlass.Int32, gate_stride_b: cutlass.Int32,
    THREADS: cutlass.Constexpr[int], U: cutlass.Constexpr[int], HINT: cutlass.Constexpr[int],
) -> None:
    """``out = x + gate * y``; CTA = (column block, U-row chunk, batch)."""
    tidx, _, _ = cute.arch.thread_idx()
    dcol, chunk, b = cute.arch.block_idx()
    xp = cutlass.Pointer(x.iterator, dtype=_BF16)
    gp = cutlass.Pointer(gate.iterator, dtype=_BF16)
    yp = cutlass.Pointer(y.iterator, dtype=_BF16)
    op = cutlass.Pointer(out.iterator, dtype=_BF16)
    ef, el = _policies(HINT)
    col = dcol * THREADS + tidx
    gw = _ld(gate.iterator, gp, b * gate_stride_b + col * VEC, None)
    base = ((b * S + chunk * U) * d_vec + col) * VEC
    step = d_vec * VEC
    xs = []
    ys = []
    for k in cutlass.range_constexpr(U):
        xs.append(_ld(x.iterator, xp, base + k * step, ef))
        ys.append(_ld(y.iterator, yp, base + k * step, ef))
    for k in cutlass.range_constexpr(U):
        words = []
        for j in cutlass.range_constexpr(WORDS):
            words.append(_fma2(gw[j], ys[k][j], xs[k][j]))
        _st(out.iterator, op, base + k * step, words, el)


@cute.kernel
def gate_residual_add_a2_kernel(
    x: cute.Tensor, gate: cute.Tensor, y1: cute.Tensor, y2: cute.Tensor,
    out: cute.Tensor, proj: cute.Tensor,
    S: cutlass.Int32, d_vec: cutlass.Int32, gate_stride_b: cutlass.Int32,
    THREADS: cutlass.Constexpr[int], U: cutlass.Constexpr[int], HINT: cutlass.Constexpr[int],
) -> None:
    """``proj = y1 + y2`` (rounded to bf16, as stored); ``out = x + gate * proj``."""
    tidx, _, _ = cute.arch.thread_idx()
    dcol, chunk, b = cute.arch.block_idx()
    xp = cutlass.Pointer(x.iterator, dtype=_BF16)
    gp = cutlass.Pointer(gate.iterator, dtype=_BF16)
    y1p = cutlass.Pointer(y1.iterator, dtype=_BF16)
    y2p = cutlass.Pointer(y2.iterator, dtype=_BF16)
    op = cutlass.Pointer(out.iterator, dtype=_BF16)
    pp = cutlass.Pointer(proj.iterator, dtype=_BF16)
    ef, el = _policies(HINT)
    col = dcol * THREADS + tidx
    gw = _ld(gate.iterator, gp, b * gate_stride_b + col * VEC, None)
    base = ((b * S + chunk * U) * d_vec + col) * VEC
    step = d_vec * VEC
    xs = []
    y1s = []
    y2s = []
    for k in cutlass.range_constexpr(U):
        xs.append(_ld(x.iterator, xp, base + k * step, ef))
        y1s.append(_ld(y1.iterator, y1p, base + k * step, ef))
        y2s.append(_ld(y2.iterator, y2p, base + k * step, ef))
    for k in cutlass.range_constexpr(U):
        pw = []
        ow = []
        for j in cutlass.range_constexpr(WORDS):
            pw.append(_add2(y1s[k][j], y2s[k][j]))
        for j in cutlass.range_constexpr(WORDS):
            ow.append(_fma2(gw[j], pw[j], xs[k][j]))
        _st(proj.iterator, pp, base + k * step, pw, ef)
        _st(out.iterator, op, base + k * step, ow, el)


@cute.kernel
def gate_bwd_a2_kernel(
    grad: cute.Tensor, gate: cute.Tensor, proj: cute.Tensor,
    grad_proj: cute.Tensor, grad_gate: cute.Tensor,
    S: cutlass.Int32, d_vec: cutlass.Int32, gate_stride_b: cutlass.Int32,
    THREADS: cutlass.Constexpr[int], U: cutlass.Constexpr[int], HINT: cutlass.Constexpr[int],
) -> None:
    """``grad_proj = grad * gate``, ``grad_gate = grad * proj``; CTA = (column block, U-row chunk, batch).

    ``HINT``: 1 puts ``evict_first`` on the single-use ``grad`` / ``proj`` loads; 2 adds
    ``evict_last`` on the ``grad_proj`` stores (read by the cast that follows).
    """
    tidx, _, _ = cute.arch.thread_idx()
    dcol, chunk, b = cute.arch.block_idx()
    gp = cutlass.Pointer(grad.iterator, dtype=_BF16)
    ap = cutlass.Pointer(gate.iterator, dtype=_BF16)
    pp = cutlass.Pointer(proj.iterator, dtype=_BF16)
    o0p = cutlass.Pointer(grad_proj.iterator, dtype=_BF16)
    o1p = cutlass.Pointer(grad_gate.iterator, dtype=_BF16)
    ef, el = _policies(HINT)
    col = dcol * THREADS + tidx
    gw = _ld(gate.iterator, ap, b * gate_stride_b + col * VEC, None)
    base = ((b * S + chunk * U) * d_vec + col) * VEC
    step = d_vec * VEC
    gs = []
    ps = []
    for k in cutlass.range_constexpr(U):
        gs.append(_ld(grad.iterator, gp, base + k * step, ef))
        ps.append(_ld(proj.iterator, pp, base + k * step, ef))
    for k in cutlass.range_constexpr(U):
        w0 = []
        w1 = []
        for j in cutlass.range_constexpr(WORDS):
            w0.append(_mul2(gs[k][j], gw[j]))
        for j in cutlass.range_constexpr(WORDS):
            w1.append(_mul2(gs[k][j], ps[k][j]))
        _st(grad_proj.iterator, o0p, base + k * step, w0, el)
        _st(grad_gate.iterator, o1p, base + k * step, w1, None)


# --- host JIT entry points (one per (op, pinned?) so MINB never has to mean "unset") ---


@cute.jit
def _host_plain(
    x: cute.Tensor, gate: cute.Tensor, y: cute.Tensor, out: cute.Tensor,
    S: cutlass.Int32, d_vec: cutlass.Int32, gate_stride_b: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32,
    stream: cuda.CUstream,
    THREADS: cutlass.Constexpr[int], U: cutlass.Constexpr[int], HINT: cutlass.Constexpr[int],
) -> None:
    gate_residual_a2_kernel(x, gate, y, out, S, d_vec, gate_stride_b, THREADS, U, HINT).launch(
        grid=(grid_x, grid_y, grid_z), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def _host_plain_mb(
    x: cute.Tensor, gate: cute.Tensor, y: cute.Tensor, out: cute.Tensor,
    S: cutlass.Int32, d_vec: cutlass.Int32, gate_stride_b: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32,
    stream: cuda.CUstream,
    THREADS: cutlass.Constexpr[int], U: cutlass.Constexpr[int], HINT: cutlass.Constexpr[int],
    MINB: cutlass.Constexpr[int],
) -> None:
    gate_residual_a2_kernel(x, gate, y, out, S, d_vec, gate_stride_b, THREADS, U, HINT).launch(
        grid=(grid_x, grid_y, grid_z), block=(THREADS, 1, 1), stream=stream,
        min_blocks_per_mp=MINB, preferred_smem_carveout=0)


@cute.jit
def _host_add(
    x: cute.Tensor, gate: cute.Tensor, y1: cute.Tensor, y2: cute.Tensor,
    out: cute.Tensor, proj: cute.Tensor,
    S: cutlass.Int32, d_vec: cutlass.Int32, gate_stride_b: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32,
    stream: cuda.CUstream,
    THREADS: cutlass.Constexpr[int], U: cutlass.Constexpr[int], HINT: cutlass.Constexpr[int],
) -> None:
    gate_residual_add_a2_kernel(
        x, gate, y1, y2, out, proj, S, d_vec, gate_stride_b, THREADS, U, HINT,
    ).launch(grid=(grid_x, grid_y, grid_z), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def _host_add_mb(
    x: cute.Tensor, gate: cute.Tensor, y1: cute.Tensor, y2: cute.Tensor,
    out: cute.Tensor, proj: cute.Tensor,
    S: cutlass.Int32, d_vec: cutlass.Int32, gate_stride_b: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32,
    stream: cuda.CUstream,
    THREADS: cutlass.Constexpr[int], U: cutlass.Constexpr[int], HINT: cutlass.Constexpr[int],
    MINB: cutlass.Constexpr[int],
) -> None:
    gate_residual_add_a2_kernel(
        x, gate, y1, y2, out, proj, S, d_vec, gate_stride_b, THREADS, U, HINT,
    ).launch(grid=(grid_x, grid_y, grid_z), block=(THREADS, 1, 1), stream=stream,
             min_blocks_per_mp=MINB, preferred_smem_carveout=0)


@cute.jit
def _host_bwd(
    grad: cute.Tensor, gate: cute.Tensor, proj: cute.Tensor,
    grad_proj: cute.Tensor, grad_gate: cute.Tensor,
    S: cutlass.Int32, d_vec: cutlass.Int32, gate_stride_b: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32,
    stream: cuda.CUstream,
    THREADS: cutlass.Constexpr[int], U: cutlass.Constexpr[int], HINT: cutlass.Constexpr[int],
) -> None:
    gate_bwd_a2_kernel(
        grad, gate, proj, grad_proj, grad_gate, S, d_vec, gate_stride_b, THREADS, U, HINT,
    ).launch(grid=(grid_x, grid_y, grid_z), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def _host_bwd_mb(
    grad: cute.Tensor, gate: cute.Tensor, proj: cute.Tensor,
    grad_proj: cute.Tensor, grad_gate: cute.Tensor,
    S: cutlass.Int32, d_vec: cutlass.Int32, gate_stride_b: cutlass.Int32,
    grid_x: cutlass.Int32, grid_y: cutlass.Int32, grid_z: cutlass.Int32,
    stream: cuda.CUstream,
    THREADS: cutlass.Constexpr[int], U: cutlass.Constexpr[int], HINT: cutlass.Constexpr[int],
    MINB: cutlass.Constexpr[int],
) -> None:
    gate_bwd_a2_kernel(
        grad, gate, proj, grad_proj, grad_gate, S, d_vec, gate_stride_b, THREADS, U, HINT,
    ).launch(grid=(grid_x, grid_y, grid_z), block=(THREADS, 1, 1), stream=stream,
             min_blocks_per_mp=MINB, preferred_smem_carveout=0)


# --- host side ------------------------------------------------------------------------

_compiled: dict[tuple, Callable] = {}


def _compile(device_index: int, add: bool, threads: int, u: int, hint: int, minb: int) -> Callable:
    """Compile once per (device, op, THREADS, U, HINT, MINB); every shape stays symbolic."""
    key = (device_index, add, threads, u, hint, minb)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            n = cute.sym_int64(divisibility=VEC)  # x / y* / out / proj share numel
            m = cute.sym_int64(divisibility=VEC)  # the gate's flat extent
            fake = lambda s: make_fake_compact_tensor(_BF16, (s,), assumed_align=16)  # noqa: E731
            ints = [cutlass.Int32(0)] * 6  # S, d_vec, gate_stride_b, grid x/y/z
            if add:
                args = (fake(n), fake(m), fake(n), fake(n), fake(n), fake(n))
                host = _host_add_mb if minb else _host_add
            else:
                args = (fake(n), fake(m), fake(n), fake(n))
                host = _host_plain_mb if minb else _host_plain
            knobs = (threads, u, hint, minb) if minb else (threads, u, hint)
            _compiled[key] = cute.compile(
                host, *args, *ints, make_fake_stream(), *knobs, options="--enable-tvm-ffi")
    return _compiled[key]


def _compile_bwd(device_index: int, threads: int, u: int, hint: int, minb: int) -> Callable:
    """:func:`_compile` for the backward (five tensors, two outputs); same symbolic shapes."""
    key = (device_index, "bwd", threads, u, hint, minb)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            n = cute.sym_int64(divisibility=VEC)
            m = cute.sym_int64(divisibility=VEC)
            fake = lambda s: make_fake_compact_tensor(_BF16, (s,), assumed_align=16)  # noqa: E731
            ints = [cutlass.Int32(0)] * 6
            args = (fake(n), fake(m), fake(n), fake(n), fake(n))
            host = _host_bwd_mb if minb else _host_bwd
            knobs = (threads, u, hint, minb) if minb else (threads, u, hint)
            _compiled[key] = cute.compile(
                host, *args, *ints, make_fake_stream(), *knobs, options="--enable-tvm-ffi")
    return _compiled[key]


#: The measured width/unroll rule: Rubin
#: (cc >= 10.7) holds 32 warps per SM, so a 384-thread CTA caps at 2 per SM and the
#: 128-thread column split with deeper unrolling wins; Blackwell (cc 10.x < 10.7)
#: holds 64 warps and one CTA per row with a shallow unroll is the best arm.
#: (threads, u_plain, u_add).
_RULE_RUBIN = (128, 4, 2)
_RULE_BLACKWELL = (384, 2, 1)
#: The other width, for the ``rb_alt`` arm: (threads, u_plain, u_add).
_ALT_RUBIN = (384, 4, 2)
_ALT_BLACKWELL = (128, 2, 2)
#: ``min_blocks_per_mp`` for the ``rb_mb`` arm, per rule width (the occupancy
#: ncu reported: T384 -> 4-5 CTAs/SM on Blackwell, T128 -> 8 on Rubin).
_MINB = {384: 4, 128: 8}


def is_rubin(cc: tuple[int, int] | None = None) -> bool:
    if cc is None:
        cc = torch.cuda.get_device_capability()
    return tuple(cc) >= (10, 7)


def _width(rule_width: int, D: int) -> int:
    d_vec = D // VEC
    if d_vec % rule_width == 0:
        return rule_width
    cands = [t for t in range(1024, 31, -32) if d_vec % t == 0]
    if not cands:
        raise ValueError(f"fused_gate_residual_a2: D/8 = {d_vec} has no divisor that is a "
                         f"multiple of 32")
    return cands[0]


def _unroll(u0: int, S: int) -> int:
    return next(k for k in (u0, 4, 2, 1) if k <= u0 and S % k == 0)


def pick(S: int, D: int, add: bool = False, cc: tuple[int, int] | None = None) -> tuple[int, int]:
    """The dispatch rule on size expressions and compute capability: ``(threads, u)``.

    Args:
        S: Tokens per sample (rows per batch element).
        D: Hidden size; ``D % 8 == 0``.
        add: ``True`` for the ``_add`` variant (three loads per row).
        cc: Compute capability; defaults to the current device's.

    Returns:
        ``(threads, u)``: the rule's width where it divides ``D/8`` (else the largest
        multiple of 32 <= 1024 that does), and the rule's unroll for the op reduced
        to the largest of it, 4, 2, 1 that divides ``S``.

    Raises:
        ValueError: If ``D/8`` has no divisor that is a multiple of 32.
    """
    width, u_plain, u_add = _RULE_RUBIN if is_rubin(cc) else _RULE_BLACKWELL
    return _width(width, D), _unroll(u_add if add else u_plain, S)


def pick_alt(S: int, D: int, add: bool = False, cc: tuple[int, int] | None = None) -> tuple[int, int]:
    """The other CTA width (the ``rb_alt`` arm), same size reductions as :func:`pick`."""
    width, u_plain, u_add = _ALT_RUBIN if is_rubin(cc) else _ALT_BLACKWELL
    return _width(width, D), _unroll(u_add if add else u_plain, S)


#: arm name -> f(S, D, add, cc) -> (threads, u, hint, minb)
ARMS: dict[str, Callable] = {
    "rb": lambda S, D, add, cc: (*pick(S, D, add, cc), HINT_NONE, 0),
    "rb_alt": lambda S, D, add, cc: (*pick_alt(S, D, add, cc), HINT_NONE, 0),
    "rb_ef": lambda S, D, add, cc: (*pick(S, D, add, cc), HINT_EF, 0),
    "rb_efl": lambda S, D, add, cc: (*pick(S, D, add, cc), HINT_EFL, 0),
    "rb_mb": lambda S, D, add, cc: (lambda t, u: (t, u, HINT_NONE, _MINB.get(t, 0)))(*pick(S, D, add, cc)),
}


def grid(B: int, S: int, D: int, threads: int, u: int) -> tuple[int, int, int]:
    """``(D/8/threads, S/u, B)`` -- the launch grid for the given knobs."""
    d_vec = D // VEC
    if d_vec % threads or S % u:
        raise ValueError(f"fused_gate_residual_a2: threads={threads} must divide D/8={d_vec} and "
                         f"u={u} must divide S={S}")
    if threads % 32 or threads > 1024:
        raise ValueError(f"fused_gate_residual_a2: threads={threads} must be a multiple of 32 <= 1024")
    gy, gz = S // u, B
    if gy > 65535 or gz > 65535:
        raise ValueError("fused_gate_residual_a2: grid y/z exceed 65535")
    return d_vec // threads, gy, gz


def _check_dense(name: str, t: torch.Tensor, like: torch.Tensor) -> None:
    if not t.is_cuda or t.dtype != torch.bfloat16:
        raise ValueError(f"fused_gate_residual_a2: {name} must be a CUDA bf16 tensor")
    if t.shape != like.shape:
        raise ValueError(f"fused_gate_residual_a2: {name} shape {tuple(t.shape)} != {tuple(like.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"fused_gate_residual_a2: {name} must be contiguous")
    if t.data_ptr() % 16:
        raise ValueError(f"fused_gate_residual_a2: {name} must be 16-byte aligned")


def _check(x: torch.Tensor, gate: torch.Tensor, *ys: torch.Tensor) -> tuple[int, int, int]:
    if x.dim() != 3:
        raise ValueError(f"fused_gate_residual_a2: x must be [B, S, D], got {tuple(x.shape)}")
    B, S, D = x.shape
    _check_dense("x", x, x)
    for i, y in enumerate(ys):
        _check_dense(f"y{i + 1}" if len(ys) > 1 else "y", y, x)
    if not gate.is_cuda or gate.dtype != torch.bfloat16:
        raise ValueError("fused_gate_residual_a2: gate must be a CUDA bf16 tensor")
    if tuple(gate.shape) != (B, 1, D):
        raise ValueError(f"fused_gate_residual_a2: gate must be [B, 1, D] = {(B, 1, D)}, "
                         f"got {tuple(gate.shape)}")
    if gate.stride(2) != 1 or gate.stride(0) % VEC or gate.data_ptr() % 16:
        raise ValueError("fused_gate_residual_a2: gate rows must be unit-stride, its batch stride "
                         "a multiple of 8 elements, and its base 16-byte aligned")
    if D % VEC:
        raise ValueError(f"fused_gate_residual_a2: D must be a multiple of 8, got {D}")
    if x.numel() > _INT32_MAX or gate.stride(0) * (B - 1) + D > _INT32_MAX:
        raise ValueError("fused_gate_residual_a2: index range exceeds Int32")
    return B, S, D


def _gate_flat(gate: torch.Tensor, B: int, D: int) -> torch.Tensor:
    """The gate's storage from its base pointer to its last element, as a 1-D view (no copy)."""
    return gate.as_strided((gate.stride(0) * (B - 1) + D,), (1,))


def _stream(dev: int) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)


def _knobs(S, D, add, arm, threads, u, hint, minb):
    t0, u0, h0, m0 = ARMS[arm](S, D, add, None)
    return (threads or t0, u or u0, h0 if hint is None else hint, m0 if minb is None else minb)


def gate_residual_a2(
    x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor, *, arm: str = "rb",
    threads: int | None = None, u: int | None = None, hint: int | None = None,
    minb: int | None = None,
) -> torch.Tensor:
    """``x + gate * y`` -- the row-block CUTLASS-DSL kernel.

    Args:
        x: ``[B, S, D]`` bf16 contiguous CUDA tensor, ``D % 8 == 0``.
        gate: ``[B, 1, D]`` bf16; may be a non-contiguous batch-strided view
            (unit row stride, batch stride a multiple of 8, 16-byte-aligned base).
        y: Same layout as ``x``.
        arm: A key of :data:`ARMS`; the registered op uses ``"rb"``.
        threads: CTA width in threads (must divide ``D/8``); default from ``arm``.
        u: Rows per thread (must divide ``S``); default from ``arm``.
        hint: L2 policy selector (0 / 1 / 2); default from ``arm``.
        minb: ``min_blocks_per_mp`` (0 = unpinned); default from ``arm``.

    Returns:
        ``[B, S, D]`` bf16, bitwise ``flux.fused_gate_residual``.

    Raises:
        ValueError: On any layout, dtype, alignment, divisibility or index-range violation.
    """
    B, S, D = _check(x, gate, y)
    threads, u, hint, minb = _knobs(S, D, False, arm, threads, u, hint, minb)
    gx, gy, gz = grid(B, S, D, threads, u)
    out = torch.empty_like(x)
    dev = x.device.index
    _compile(dev, False, threads, u, hint, minb)(
        x.view(-1), _gate_flat(gate, B, D), y.view(-1), out.view(-1),
        S, D // VEC, gate.stride(0), gx, gy, gz, _stream(dev))
    return out


def gate_residual_add_a2(
    x: torch.Tensor, gate: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, *, arm: str = "rb",
    threads: int | None = None, u: int | None = None, hint: int | None = None,
    minb: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x + gate * (y1 + y2), y1 + y2)`` -- same contract and knobs as :func:`gate_residual_a2`."""
    B, S, D = _check(x, gate, y1, y2)
    threads, u, hint, minb = _knobs(S, D, True, arm, threads, u, hint, minb)
    gx, gy, gz = grid(B, S, D, threads, u)
    out = torch.empty_like(x)
    proj = torch.empty_like(x)
    dev = x.device.index
    _compile(dev, True, threads, u, hint, minb)(
        x.view(-1), _gate_flat(gate, B, D), y1.view(-1), y2.view(-1), out.view(-1), proj.view(-1),
        S, D // VEC, gate.stride(0), gx, gy, gz, _stream(dev))
    return out, proj


def gate_bwd_a2(
    grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor, *, arm: str = "rb",
    threads: int | None = None, u: int | None = None, hint: int | None = None,
    minb: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(grad * gate, grad * proj)`` -- the gated-residual backward, same contract and knobs
    as :func:`gate_residual_a2` (the plain op's unroll rule: two streaming loads per row).

    Args:
        grad: ``[B, S, D]`` bf16 contiguous CUDA tensor, ``D % 8 == 0``.
        gate: ``[B, 1, D]`` bf16; may be a non-contiguous batch-strided view.
        proj: Same layout as ``grad``.
        arm: A key of :data:`ARMS`; the registered op uses ``"rb"``.
        threads: CTA width in threads (must divide ``D/8``); default from ``arm``.
        u: Rows per thread (must divide ``S``); default from ``arm``.
        hint: L2 policy selector (0 / 1 / 2); default from ``arm``.
        minb: ``min_blocks_per_mp`` (0 = unpinned); default from ``arm``.

    Returns:
        ``(grad_proj, grad_gate)``, both ``[B, S, D]`` bf16: each element the fp32 product
        rounded once to bf16 (``tests/gate_reference.gate_bwd_ref``).

    Raises:
        ValueError: On any layout, dtype, alignment, divisibility or index-range violation.
    """
    B, S, D = _check(grad, gate, proj)
    threads, u, hint, minb = _knobs(S, D, False, arm, threads, u, hint, minb)
    gx, gy, gz = grid(B, S, D, threads, u)
    grad_proj = torch.empty_like(grad)
    grad_gate = torch.empty_like(grad)
    dev = grad.device.index
    _compile_bwd(dev, threads, u, hint, minb)(
        grad.view(-1), _gate_flat(gate, B, D), proj.view(-1), grad_proj.view(-1), grad_gate.view(-1),
        S, D // VEC, gate.stride(0), gx, gy, gz, _stream(dev))
    return grad_proj, grad_gate


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define("fused_gate_residual_a2(Tensor x, Tensor gate, Tensor y) -> Tensor")
_lib.define("fused_gate_residual_add_a2(Tensor x, Tensor gate, Tensor y1, Tensor y2)"
            " -> (Tensor, Tensor)")
_lib.define("fused_gate_bwd(Tensor grad, Tensor gate, Tensor proj) -> (Tensor, Tensor)")


#: The mb-4 shapes (M <= SMALL_M_ROWS rows) take the 128-thread-CTA arm (rb_alt):
#: 4.7 us/launch beside the 256-rank NCCL kernels vs rb (384-thread CTAs) 11.8 and Triton 4.8.
SMALL_M_ROWS = 2048


@torch.library.impl(_lib, "fused_gate_residual_a2", "CUDA")
def _fgr_a2_impl(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    arm = "rb_alt" if x.shape[0] * x.shape[1] <= SMALL_M_ROWS else "rb"
    return gate_residual_a2(x, gate, y, arm=arm)


@torch.library.impl(_lib, "fused_gate_residual_add_a2", "CUDA")
def _fgra_a2_impl(
    x: torch.Tensor, gate: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    arm = "rb_alt" if x.shape[0] * x.shape[1] <= SMALL_M_ROWS else "rb"
    return gate_residual_add_a2(x, gate, y1, y2, arm=arm)


@torch.library.impl(_lib, "fused_gate_bwd", "CUDA")
def _fgb_impl(
    grad: torch.Tensor, gate: torch.Tensor, proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gate_bwd_a2(grad, gate, proj)
