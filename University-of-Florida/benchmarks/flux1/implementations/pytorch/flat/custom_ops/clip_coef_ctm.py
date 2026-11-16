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

"""The grad-norm clip coefficient on the CUTLASS DSL .

One kernel, one thread, replacing the **seven** ``at::native`` scalar launches the
production CUDA graph carries at every micro-batch between the grad-norm all-reduce and
``adamw_ctm`` (``train_flat.py:489-498``) **and** the ``_inv_scale`` ``memcpy32_post`` that
followed them -- eight launches for one. Eight production replay traces measured
the family it removes at **9.7-11.7 us/step, flat in micro-batch**: this is launch-count
closure, not a performance term.

Plain ``cute.arch.*`` / ``cute.math.*`` -- one thread doing seven scalar operations has no
pipeline, TMA, mbarrier or MMA for a ``cutlass.experimental`` ``prims`` construct to hide
anything behind (the level rule; ``flat/custom_ops/aten_tail_ctm.py`` made the same choice
for the same class of work, and is this module's scaffolding template).

THE BITWISE CONTRACT
--------------------
The kernel reproduces ATen's **lowering**, not the algebra. Two things make that a real
constraint rather than a formality:

1. **ATen turns both divisions into a reciprocal and a multiply**, and that is not the same
   float as a single divide. ``tensor / world_size`` is ``div_true_kernel_cuda``'s scalar
   path, which computes ``opmath_t inv_b = 1 / b`` on the host and emits a
   ``BUnaryFunctor<MulFunctor>``; ``max_norm / tensor`` is ``reciprocal`` followed by an
   ``AUnaryFunctor<MulFunctor>``. Both are visible by name in the launch trace. The
   ``1/world_size`` reciprocal is therefore computed **on the host, in float** and passed in,
   exactly as ATen does -- the kernel never re-derives it.
2. **ATen is built WITHOUT fast math**, so unlike the optimizer (which had to match
   TE's ``-prec-div=false -prec-sqrt=false -ftz=true``) every operation here is IEEE:
   ``sqrt.rn.f32``, ``div.rn.f32``, and **no** ``.ftz`` -- denormals are preserved.
3. **``clamp``'s NaN guard is load-bearing.** ATen's ``launch_clamp_scalar`` with
   ``ClampLimits::Max`` is ``_isnan(v) ? v : ::min(v, upper)``; a bare ``fminf(NaN, 1.0f)``
   would return ``1.0f`` and silently turn a NaN gradient norm into a clip coefficient of 1.
   ``f = 1.0f if e > 1.0f else e`` reproduces the guarded form exactly for every input --
   including NaN (``setp.gt.f32`` is false, so ``e`` survives) and ``-0.0``.

``tests/test_clip_coef_ctm.py`` asserts **integer-view** equality against the seven-op ATen
chain over a grid that includes 0, denormals, both sides of the clamp, +-Inf and NaN.
Integer-view because ``torch.equal`` is IEEE equality and would pass ``NaN != NaN``.

SEMANTICS, WHICH ARE CONVERGENCE-RELEVANT AND UNCHANGED
-------------------------------------------------------
Gradients are **SUM**-reduced (SHARP cannot fuse AVG); ``max_norm`` is defined against the
**mean** gradient (reference / MLPerf semantics), hence the ``/ world_size`` *before* the
coefficient; and the ``1 / world_size`` average is folded into the **same** coefficient the
optimizer applies, hence the ``/ world_size`` *after*. Both survive, in that order.

**The optimizer is not touched.** This kernel writes the same ``_inv_scale`` scalar
``adamw_ctm`` already reads, so the optimizer's arithmetic ordering -- and with it the
bitwise identity to TE's fused AdamW that is the programme's largest shipped win -- is
unchanged by construction.

``FLAT_CLIP_COEF_CTM=0`` keeps the ATen chain (the in-situ baseline arm). The import itself
never falls back.
"""

from __future__ import annotations

import os
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

#: ``FLAT_CLIP_COEF_CTM=0`` -> the site keeps its seven-launch ATen chain (baseline arm).
ENABLED = os.environ.get("FLAT_CLIP_COEF_CTM", "1") == "1"

_F32 = cutlass.Float32
_RN = cute.math.RoundingMode.NEAREST_EVEN

#: ATen's ``clamp`` epsilon, ``train_flat.py:493``.
EPS = 1e-6

#: Kernel-symbol substring for the by-name launch census.
KERNELS = {"clip_coef": "clip_coef_ctm_kernel"}

_compiled: dict[tuple, Callable] = {}


def _stream(dev: int) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)


def _fake1(symbol: str):
    """A one-element fp32 tensor.

    ``divisibility=1`` / ``assumed_align=4`` because every access is a scalar 4-byte load or
    store -- the same shape ``aten_tail_ctm``'s ``loss_out`` uses (``_fake(_F32, 1, 4, "l")``).
    """
    return make_fake_compact_tensor(
        _F32, (cute.sym_int64(divisibility=1, symbol=symbol),), assumed_align=4)


@cute.kernel
def clip_coef_ctm_kernel(
    norm_sq: cute.Tensor, inv_scale: cute.Tensor, gnorm: cute.Tensor,
    max_norm: _F32, inv_world: _F32,
    WRITE_GNORM: cutlass.Constexpr[bool],
) -> None:
    """One thread: the seven-op ATen clip chain, in ATen's lowering.

    ``inv_scale[0] = clamp(max_norm / (sqrt(norm_sq[0]) / W + 1e-6), max=1) / W`` with every
    step pinned to the instruction ATen emits (see the module docstring), and
    ``gnorm[0] = sqrt(norm_sq[0]) / W`` -- the mean gradient norm -- when asked for.
    """
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        s = (cutlass.Pointer(norm_sq.iterator, dtype=_F32) + 0).load()
        # a: sqrt_kernel_cuda -- IEEE sqrt.rn.f32 (ATen is not built -prec-sqrt=false)
        a = cute.math.sqrt(s, rounding=_RN)
        # b: div.Scalar -> BUnaryFunctor<MulFunctor>, the reciprocal formed on the host
        b = cute.math.mul(a, inv_world, rounding=_RN, ftz=False)
        # c: CUDAFunctorOnSelf_add
        c = cute.math.add(b, _F32(EPS), rounding=_RN, ftz=False)
        # d: reciprocal_kernel_cuda -- IEEE div.rn.f32, NOT rcp.approx
        d = cute.math.div(_F32(1.0), c, rounding=_RN)
        # e: AUnaryFunctor<MulFunctor>
        e = cute.math.mul(d, max_norm, rounding=_RN, ftz=False)
        # f: launch_clamp_scalar, ClampLimits::Max == `_isnan(v) ? v : ::min(v, upper)`.
        # `e > 1.0` is false for NaN, so NaN survives -- which `fminf` would not do.
        f = e
        if e > _F32(1.0):
            f = _F32(1.0)
        # g: div.Scalar -> BUnaryFunctor<MulFunctor> again (the folded 1/world_size average)
        g = cute.math.mul(f, inv_world, rounding=_RN, ftz=False)
        cutlass.Pointer(inv_scale.iterator, dtype=_F32).store(g)
        if cutlass.const_expr(WRITE_GNORM):
            cutlass.Pointer(gnorm.iterator, dtype=_F32).store(b)


@cute.jit
def clip_coef_ctm_host(
    norm_sq: cute.Tensor, inv_scale: cute.Tensor, gnorm: cute.Tensor,
    max_norm: _F32, inv_world: _F32, stream: cuda.CUstream,
    WRITE_GNORM: cutlass.Constexpr[bool],
) -> None:
    clip_coef_ctm_kernel(norm_sq, inv_scale, gnorm, max_norm, inv_world, WRITE_GNORM).launch(
        grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


def _compile(dev: int, write_gnorm: bool) -> Callable:
    key = ("clip_coef", dev, write_gnorm)
    if key not in _compiled:
        with torch.cuda.device(dev):
            _compiled[key] = cute.compile(
                clip_coef_ctm_host, _fake1("s"), _fake1("i"), _fake1("g"),
                _F32(0.0), _F32(0.0), make_fake_stream(), write_gnorm,
                options="--enable-tvm-ffi")
    return _compiled[key]


def _check(t: torch.Tensor, name: str) -> torch.Tensor:
    if not t.is_cuda or t.dtype != torch.float32 or t.numel() != 1 or not t.is_contiguous():
        raise ValueError(f"clip_coef_ctm: {name} must be a contiguous 1-element fp32 CUDA tensor")
    return t.view(1)


def clip_coef_ctm(
    norm_sq: torch.Tensor, inv_scale: torch.Tensor, gnorm: torch.Tensor | None,
    max_norm: float, world_size: int,
) -> None:
    """Write the clip coefficient (and optionally the mean grad norm) in one launch.

    Args:
        norm_sq: 1-element fp32 tensor, the all-reduced squared global gradient norm.
        inv_scale: 1-element fp32 tensor written in place -- the scale ``adamw_ctm`` reads.
        gnorm: 1-element fp32 tensor for the **mean** gradient norm, or ``None`` to skip it.
        max_norm: The clip threshold, against the mean gradient (MLPerf semantics).
        world_size: Ranks the gradients were SUM-reduced over.

    Raises:
        ValueError: On a dtype, shape or layout the kernel does not take.
    """
    ns = _check(norm_sq, "norm_sq")
    iv = _check(inv_scale, "inv_scale")
    gn = _check(gnorm, "gnorm") if gnorm is not None else iv
    dev = ns.device.index
    # ATen's ``div_true_kernel_cuda`` scalar path forms ``1 / b`` in opmath_t (float) on the
    # host and multiplies; forming it here keeps the kernel bitwise that path.
    inv_world = float(torch.tensor(1.0 / world_size, dtype=torch.float32).item())
    _compile(dev, gnorm is not None)(
        ns, iv, gn, float(max_norm), inv_world, _stream(dev))


# ---------------------------------------------------------------------------
# torch.library registration
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "clip_coef_ctm(Tensor norm_sq, Tensor(a!) inv_scale, Tensor(b!)? gnorm, "
    "float max_norm, int world_size) -> ()"
)


@torch.library.impl(_lib, "clip_coef_ctm", "CUDA")
def _clip_coef_impl(norm_sq, inv_scale, gnorm, max_norm: float, world_size: int) -> None:
    clip_coef_ctm(norm_sq, inv_scale, gnorm, max_norm, world_size)
