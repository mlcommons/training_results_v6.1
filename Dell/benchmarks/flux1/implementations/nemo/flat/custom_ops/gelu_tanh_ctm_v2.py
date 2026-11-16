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

"""``torch.ops.flux.gelu_tanh_ctm_v2`` -- second attempt.

The shipped DSL composite (``gelu_tanh_ctm.py``) reads 94 % of the
copy band on Blackwell and 75-77 % on Rubin. This module tests two mechanisms for the
Rubin gap:

* **UNROLL** -- each thread issues ``UNROLL`` independent 128-bit loads per
  iteration, all before its first store, so the bytes in flight per SM grow
  by ``UNROLL`` x (Little's law: a one-load kernel at 8 CTAs/SM tops out at
  32 KB/SM, which Rubin's 10.44 TB/s band cannot be reached from).
* **PACKED** -- the fp32 arithmetic goes through ``cute.arch.mul_packed_f32x2``
  / ``fma_packed_f32x2`` (``mul.rn.f32x2`` / ``fma.rn.f32x2``, sm_100+),
  per-lane IEEE RN and therefore bitwise the scalar ``mul`` / ``fma``.
  ``tanh.approx.f32`` has no packed form and stays scalar.

Both wrappers exist with the same signature in public 4.6.2 and
4.8.0.dev0. The math is TE's expression with nvcc's FMA placement, so
each mode stayed **bitwise** the CUDA kernel ``gelu_tanh_kernel.cu`` it replaced; the
bf16 narrowing is ``.to(BFloat16)`` (RN) -- never ``cvt_f32x2_bf16x2``, which
emits ``.satfinite``. The composite ``gelu_tanh_ctm_v2_auto`` reuses the
shipped bulk-staged kernel unchanged below ``DISPATCH_MAX_NUMEL``.

``gelu_tanh_ctm_v2_auto`` is the production GELU (``nvfp4_ops._gelu``).
"""

from __future__ import annotations

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from flat.custom_ops import gelu_tanh_ctm as _v1

THREADS = 256
VEC = 8  # bf16 elements per 128-bit access
_C1 = 0.03567741
_C0 = 0.79788456
_HALF = 0.5
_INT32_MAX = 2**31 - 1

#: arm -> (UNROLL, PACKED, grid rule).
VARIANTS: dict[str, tuple[int, bool, str]] = {
    "u1o": (1, False, "oneshot"),
    "u1p": (1, True, "persistent"),
    "u2": (2, False, "oneshot"),
    "u4": (4, False, "oneshot"),
    "u2p": (2, True, "oneshot"),
    "u4p": (4, True, "oneshot"),
}
DEFAULT_VARIANT = "u4p"
#: the shipped composite's threshold, unchanged
DISPATCH_MAX_NUMEL = _v1.DISPATCH_MAX_NUMEL


def _tanh(v, fast: bool):
    return cute.math.tanh(v, approx=True) if fast else cute.math.tanh(v)


def _gelu_scalar(f, fast: bool):
    """TE's GELU on one fp32 scalar, FMAs placed as nvcc contracts them."""
    inner = f * cute.math.fma(f * _C1, f, cutlass.Float32(_C0))
    return f * cute.math.fma(_tanh(inner, fast), cutlass.Float32(_HALF), cutlass.Float32(_HALF))


def _gelu_pair(a, b, fast: bool):
    """The same expression on two fp32 lanes through the packed f32x2 wrappers."""
    c1 = (cutlass.Float32(_C1), cutlass.Float32(_C1))
    c0 = (cutlass.Float32(_C0), cutlass.Float32(_C0))
    half = (cutlass.Float32(_HALF), cutlass.Float32(_HALF))
    t = cute.arch.mul_packed_f32x2((a, b), c1)           # f * C1
    t = cute.arch.fma_packed_f32x2(t, (a, b), c0)        # fma(f * C1, f, C0)
    inner = cute.arch.mul_packed_f32x2((a, b), t)        # f * (...)
    th = (_tanh(inner[0], fast), _tanh(inner[1], fast))  # scalar: no packed tanh
    h = cute.arch.fma_packed_f32x2(th, half, half)       # fma(t, 0.5, 0.5)
    return cute.arch.mul_packed_f32x2((a, b), h)         # f * (...)


def _gelu_vec(v, fast: bool, packed: bool):
    """8 x bf16 Vector -> 8 x bf16 Vector. Plain ``for``: helpers are traced, not preprocessed."""
    outs = []
    if packed:
        for k in range(0, VEC, 2):
            o0, o1 = _gelu_pair(v[k].to(cutlass.Float32), v[k + 1].to(cutlass.Float32), fast)
            outs.append(o0.to(cutlass.BFloat16))
            outs.append(o1.to(cutlass.BFloat16))
    else:
        for k in range(VEC):
            outs.append(_gelu_scalar(v[k].to(cutlass.Float32), fast).to(cutlass.BFloat16))
    return cutlass.Vector.from_elements(tuple(outs), cutlass.BFloat16)


@cute.kernel
def gelu_tanh_ctm_v2_kernel(
    x: cute.Tensor, y: cute.Tensor, n_vec: cutlass.Int32, n_full: cutlass.Int32,
    FAST: cutlass.Constexpr[bool], UNROLL: cutlass.Constexpr[int], PACKED: cutlass.Constexpr[bool],
) -> None:
    """Grid-stride over chunks of ``THREADS x UNROLL`` vectors, then a guarded tail.

    Inside a chunk, thread ``t`` owns vectors ``base + u * THREADS`` for
    ``u < UNROLL`` -- each of the ``UNROLL`` load instructions is a fully
    coalesced 4 KB per CTA, and all of them are issued before the first store.
    ``n_full = n_vec // (THREADS x UNROLL)`` full chunks; the remainder (never
    on a production shape) is finished by the direct one-vector path.
    """
    CHUNK: cutlass.Constexpr[int] = THREADS * UNROLL
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    xp = cutlass.Pointer(x.iterator, dtype=cutlass.BFloat16)
    yp = cutlass.Pointer(y.iterator, dtype=cutlass.BFloat16)
    for c in cutlass.range(bidx, n_full, gdim):
        base = c * CHUNK + tidx
        vs = []
        for u in cutlass.range_constexpr(UNROLL):
            vs.append((xp + (base + u * THREADS) * VEC).load(16, count=VEC))
        for u in cutlass.range_constexpr(UNROLL):
            (yp + (base + u * THREADS) * VEC).store(_gelu_vec(vs[u], FAST, PACKED), alignment=16)
    for i in cutlass.range(n_full * CHUNK + bidx * THREADS + tidx, n_vec, gdim * THREADS):
        v = (xp + i * VEC).load(16, count=VEC)
        (yp + i * VEC).store(_gelu_vec(v, FAST, PACKED), alignment=16)


@cute.jit
def gelu_tanh_ctm_v2_host(
    x: cute.Tensor, y: cute.Tensor, n_vec: cutlass.Int32, n_full: cutlass.Int32,
    grid_x: cutlass.Int32, stream: cuda.CUstream,
    FAST: cutlass.Constexpr[bool], UNROLL: cutlass.Constexpr[int], PACKED: cutlass.Constexpr[bool],
) -> None:
    gelu_tanh_ctm_v2_kernel(x, y, n_vec, n_full, FAST, UNROLL, PACKED).launch(
        grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


_compiled: dict[tuple[int, bool, int, bool], Callable] = {}
_resident: dict[int, int] = {}


def _compile(device_index: int, fast: bool, unroll: int, packed: bool) -> Callable:
    """Compile once per (device, mode, UNROLL, PACKED); the shape stays symbolic."""
    key = (device_index, fast, unroll, packed)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            n = cute.sym_int64(divisibility=VEC)
            fx = make_fake_compact_tensor(cutlass.BFloat16, (n,), assumed_align=16)
            fy = make_fake_compact_tensor(cutlass.BFloat16, (n,), assumed_align=16)
            _compiled[key] = cute.compile(
                gelu_tanh_ctm_v2_host, fx, fy, cutlass.Int32(0), cutlass.Int32(0),
                cutlass.Int32(0), make_fake_stream(), fast, unroll, packed,
                options="--enable-tvm-ffi")
    return _compiled[key]


def _grid(device_index: int, n_vec: int, unroll: int, mode: str) -> int:
    """``oneshot``: one CTA per chunk. ``persistent``: the shipped composite's ``min(SMs x 8, chunks)``."""
    chunks = max(1, -(-n_vec // (THREADS * unroll)))
    if mode == "oneshot":
        return chunks
    if device_index not in _resident:
        p = torch.cuda.get_device_properties(device_index)
        _resident[device_index] = (
            p.multi_processor_count * (p.max_threads_per_multi_processor // THREADS))
    return min(_resident[device_index], chunks)


def _check(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("gelu_tanh_ctm_v2: input must be CUDA")
    if x.dtype != torch.bfloat16:
        raise ValueError("gelu_tanh_ctm_v2: input must be bf16")
    if not x.is_contiguous():
        raise ValueError("gelu_tanh_ctm_v2: input must be contiguous")
    if x.numel() % VEC != 0:
        raise ValueError(f"gelu_tanh_ctm_v2: numel must be a multiple of 8, got {x.numel()}")
    if x.numel() > _INT32_MAX:
        raise ValueError(f"gelu_tanh_ctm_v2: numel {x.numel()} exceeds the Int32 index range")


def gelu_tanh_ctm_v2(x: torch.Tensor, fast: bool, variant: str = DEFAULT_VARIANT) -> torch.Tensor:
    """Tanh-GELU of a contiguous BF16 tensor, CUTLASS-DSL kernel (second attempt).

    Args:
        x: Contiguous BF16 CUDA tensor, ``numel % 8 == 0``, ``numel < 2**31``.
        fast: ``False`` = ``tanhf`` (bitwise ``tex.gelu``); ``True`` =
            ``tanh.approx.f32``. Each mode is bitwise the same
            mode of ``flux.gelu_tanh``.
        variant: One of :data:`VARIANTS` -- ``(UNROLL, PACKED, grid rule)``.

    Returns:
        BF16 tensor of the same shape.

    Raises:
        ValueError: On a non-CUDA, non-BF16, non-contiguous tensor, a
            ``numel`` that is not a multiple of 8, or one at or above 2**31.
    """
    _check(x)
    unroll, packed, mode = VARIANTS[variant]
    y = torch.empty_like(x)
    dev = x.device.index
    n_vec = x.numel() // VEC
    n_full = n_vec // (THREADS * unroll)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    _compile(dev, fast, unroll, packed)(
        x.view(-1), y.view(-1), n_vec, n_full, _grid(dev, n_vec, unroll, mode), stream)
    return y


def gelu_tanh_ctm_v2_auto(x: torch.Tensor, fast: bool) -> torch.Tensor:
    """The pre-registered composite: the shipped staged kernel at small shapes, ``u4p`` above."""
    if x.numel() <= DISPATCH_MAX_NUMEL:
        return _v1.gelu_tanh_ctm_staged(x, fast)
    return gelu_tanh_ctm_v2(x, fast, DEFAULT_VARIANT)


def verify(m: int = 1024, n: int = 12288) -> None:
    """Compile, run and bit-check every arm: exact mode against ``tex.gelu``, fast mode
    against the shipped composite (bitwise the same ``tanh.approx`` expression).

    The smoke check the ``ctm-write-kernel`` skill requires of a kernel module;
    the full contract -- including the golden fixture that pins the fast mode's
    ``tanh.approx`` bits on the exhaustive bf16 domain -- is
    ``tests/test_gelu_tanh_ctm.py``.

    Raises:
        AssertionError: If any arm differs from its reference in either mode.
    """
    torch.manual_seed(m * 31 + n)
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    for fast in (False, True):
        ref = _v1.gelu_tanh_ctm(x, True) if fast else torch.ops.flux.te_gelu(x)
        for name in VARIANTS:
            bad = (gelu_tanh_ctm_v2(x, fast, name) != ref).sum().item()
            assert bad == 0, f"{name} fast={fast}: {bad} of {ref.numel()} differ from the reference"
        assert torch.equal(gelu_tanh_ctm_v2_auto(x, fast), ref), f"auto fast={fast}: not bitwise"
    torch.cuda.synchronize()
    print(f"gelu_tanh_ctm_v2.verify [{m},{n}]: {len(VARIANTS)} arms + auto, both modes, bitwise the reference")


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define("gelu_tanh_ctm_v2(Tensor x, bool fast) -> Tensor")
_lib.define("gelu_tanh_ctm_v2_auto(Tensor x, bool fast) -> Tensor")
for _name in VARIANTS:
    _lib.define(f"gelu_tanh_ctm_v2_{_name}(Tensor x, bool fast) -> Tensor")


@torch.library.impl(_lib, "gelu_tanh_ctm_v2", "CUDA")
def _gelu_tanh_ctm_v2_impl(x: torch.Tensor, fast: bool) -> torch.Tensor:
    return gelu_tanh_ctm_v2(x, fast)


@torch.library.impl(_lib, "gelu_tanh_ctm_v2_auto", "CUDA")
def _gelu_tanh_ctm_v2_auto_impl(x: torch.Tensor, fast: bool) -> torch.Tensor:
    return gelu_tanh_ctm_v2_auto(x, fast)


def _variant_impl(name: str) -> Callable[[torch.Tensor, bool], torch.Tensor]:
    def impl(x: torch.Tensor, fast: bool) -> torch.Tensor:
        return gelu_tanh_ctm_v2(x, fast, name)
    return impl


for _name in VARIANTS:
    _lib.impl(f"gelu_tanh_ctm_v2_{_name}", _variant_impl(_name), "CUDA")
