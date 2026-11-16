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

"""torch.ops.flux.gelu_tanh_ctm: tanh-GELU in the CUTLASS Python DSL.

256 threads, one 128-bit (8 x bf16) load and store per thread per iteration, grid-stride
over numel / 8 vectors; both modes are bitwise the CUDA kernel's.
"""

from __future__ import annotations

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

THREADS = 256
VEC = 8  # bf16 elements per 128-bit access
_C1 = 0.03567741
_C0 = 0.79788456
_INT32_MAX = 2**31 - 1


def _gelu_f32(f, fast: bool):
    """TE's GELU on one fp32 scalar, FMAs placed as nvcc contracts them."""
    inner = f * cute.math.fma(f * _C1, f, cutlass.Float32(_C0))
    if fast:
        t = cute.math.tanh(inner, approx=True)
    else:
        t = cute.math.tanh(inner)
    return f * cute.math.fma(t, cutlass.Float32(0.5), cutlass.Float32(0.5))


@cute.kernel
def gelu_tanh_ctm_kernel(
    x: cute.Tensor, y: cute.Tensor, n_vec: cutlass.Int32,
    FAST: cutlass.Constexpr[bool], INVARIANT: cutlass.Constexpr[bool] = False,
) -> None:
    """INVARIANT: load with is_invariant=True (the ld.global.nc path); same bits either way."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()
    xp = cutlass.Pointer(x.iterator, dtype=cutlass.BFloat16)
    yp = cutlass.Pointer(y.iterator, dtype=cutlass.BFloat16)
    start = bidx * bdim + tidx
    stride = gdim * bdim
    for i in cutlass.range(start, n_vec, stride):
        if cutlass.const_expr(INVARIANT):
            v = (xp + i * VEC).load(16, count=VEC, is_invariant=True)
        else:
            v = (xp + i * VEC).load(16, count=VEC)
        outs = []
        for k in cutlass.range_constexpr(VEC):
            outs.append(_gelu_f32(v[k].to(cutlass.Float32), FAST).to(cutlass.BFloat16))
        (yp + i * VEC).store(cutlass.Vector.from_elements(tuple(outs), cutlass.BFloat16),
                             alignment=16)


@cute.jit
def gelu_tanh_ctm_host(
    x: cute.Tensor, y: cute.Tensor, n_vec: cutlass.Int32, grid_x: cutlass.Int32,
    stream: cuda.CUstream, FAST: cutlass.Constexpr[bool],
) -> None:
    gelu_tanh_ctm_kernel(x, y, n_vec, FAST, False).launch(
        grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


@cute.jit
def gelu_tanh_ctm_inv_host(
    x: cute.Tensor, y: cute.Tensor, n_vec: cutlass.Int32, grid_x: cutlass.Int32,
    stream: cuda.CUstream, FAST: cutlass.Constexpr[bool],
) -> None:
    gelu_tanh_ctm_kernel(x, y, n_vec, FAST, True).launch(
        grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


# Variant P: bf16 -> f32 by shift/mask on the packed words, f32 -> bf16 by cvt.rn.bf16x2.f32
# verbatim (the DSL's cvt_f32x2_bf16x2 is the .satfinite form, not bitwise on overflow/NaN).

_CVT_BF16X2 = "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};"


@cute.kernel
def gelu_tanh_ctm_packed_kernel(
    x: cute.Tensor, y: cute.Tensor, n_vec: cutlass.Int32,
    FAST: cutlass.Constexpr[bool],
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()
    xp = cutlass.Pointer(x.iterator, dtype=cutlass.BFloat16)
    yp = cutlass.Pointer(y.iterator, dtype=cutlass.BFloat16)
    start = bidx * bdim + tidx
    stride = gdim * bdim
    for i in cutlass.range(start, n_vec, stride):
        w = (xp + i * VEC).load(16, count=VEC).bitcast(cutlass.Int32)
        outs = []
        for k in cutlass.range_constexpr(VEC // 2):
            wk = w[k]
            lo = (wk << 16).bitcast(cutlass.Float32)
            hi = (wk & cutlass.Int32(-65536)).bitcast(cutlass.Float32)
            outs.append(cute.arch.inline_ptx(
                _CVT_BF16X2, write_only_types=[cutlass.Int32],
                read_only_args=[_gelu_f32(hi, FAST), _gelu_f32(lo, FAST)]))
        o = cutlass.Vector.from_elements(tuple(outs), cutlass.Int32).bitcast(cutlass.BFloat16)
        (yp + i * VEC).store(o, alignment=16)


@cute.jit
def gelu_tanh_ctm_packed_host(
    x: cute.Tensor, y: cute.Tensor, n_vec: cutlass.Int32, grid_x: cutlass.Int32,
    stream: cuda.CUstream, FAST: cutlass.Constexpr[bool],
) -> None:
    gelu_tanh_ctm_packed_kernel(x, y, n_vec, FAST).launch(
        grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


# Variant S: cp.async.bulk G2S into a STAGES-deep smem ring with mbarrier complete_tx,
# compute from smem, st.global.v4 from registers.

TILE = 4096          # bf16 elements per bulk copy (8 KB)
STAGES = 4
STAGED_CTAS_PER_SM = 4
_G2S = ("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[{$r0}], [{$r1}], {$r2}, [{$r3}];")


@cute.kernel
def gelu_tanh_ctm_staged_kernel(
    x: cute.Tensor, y: cute.Tensor, n_tiles: cutlass.Int32, n_vec: cutlass.Int32,
    FAST: cutlass.Constexpr[bool],
) -> None:
    VPT: cutlass.Constexpr[int] = TILE // (THREADS * VEC)
    TILE_BYTES: cutlass.Constexpr[int] = TILE * 2
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()
    xp = cutlass.Pointer(x.iterator, dtype=cutlass.BFloat16)
    sin = cute.arch.alloc_smem(cutlass.BFloat16, STAGES * TILE, 128)
    bars = cute.arch.alloc_smem(cutlass.Int64, STAGES, 8)
    sin_v = cutlass.Pointer(sin, dtype=cutlass.BFloat16)
    yp = cutlass.Pointer(y.iterator, dtype=cutlass.BFloat16)
    if tidx == 0:
        for s in cutlass.range_constexpr(STAGES):
            cute.arch.mbarrier_init(bars + s, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    if tidx == 0:
        for p in cutlass.range_constexpr(STAGES):
            k0 = bidx + p * gdim
            if k0 < n_tiles:
                cute.arch.mbarrier_arrive_and_expect_tx(bars + p, TILE_BYTES)
                cute.arch.inline_ptx(_G2S, read_only_args=[
                    (sin + p * TILE).toint(), (x.iterator + k0 * TILE).toint(),
                    cutlass.Int32(TILE_BYTES), (bars + p).toint()])
    for k in cutlass.range(bidx, n_tiles, gdim):
        j = (k - bidx) // gdim
        s = j % STAGES
        cute.arch.mbarrier_wait(bars + s, (j // STAGES) % 2)
        for vi in cutlass.range_constexpr(VPT):
            off = (vi * THREADS + tidx) * VEC
            v = (sin_v + s * TILE + off).load(16, count=VEC)
            outs = []
            for e in cutlass.range_constexpr(VEC):
                outs.append(_gelu_f32(v[e].to(cutlass.Float32), FAST).to(cutlass.BFloat16))
            (yp + k * TILE + off).store(
                cutlass.Vector.from_elements(tuple(outs), cutlass.BFloat16), alignment=16)
        cute.arch.sync_threads()
        if tidx == 0:
            kn = k + STAGES * gdim
            if kn < n_tiles:
                cute.arch.mbarrier_arrive_and_expect_tx(bars + s, TILE_BYTES)
                cute.arch.inline_ptx(_G2S, read_only_args=[
                    (sin + s * TILE).toint(), (x.iterator + kn * TILE).toint(),
                    cutlass.Int32(TILE_BYTES), (bars + s).toint()])
    # sub-tile remainder (never on a production shape): the direct 128-bit path
    tail = n_tiles * (TILE // VEC)
    for i in cutlass.range(tail + bidx * bdim + tidx, n_vec, gdim * bdim):
        v = (xp + i * VEC).load(16, count=VEC)
        outs = []
        for k in cutlass.range_constexpr(VEC):
            outs.append(_gelu_f32(v[k].to(cutlass.Float32), FAST).to(cutlass.BFloat16))
        (yp + i * VEC).store(cutlass.Vector.from_elements(tuple(outs), cutlass.BFloat16),
                             alignment=16)


@cute.jit
def gelu_tanh_ctm_staged_host(
    x: cute.Tensor, y: cute.Tensor, n_tiles: cutlass.Int32, n_vec: cutlass.Int32,
    grid_x: cutlass.Int32, stream: cuda.CUstream, FAST: cutlass.Constexpr[bool],
) -> None:
    gelu_tanh_ctm_staged_kernel(x, y, n_tiles, n_vec, FAST).launch(
        grid=(grid_x, 1, 1), block=(THREADS, 1, 1), stream=stream)


_HOSTS = {"ldg": (gelu_tanh_ctm_host, VEC), "inv": (gelu_tanh_ctm_inv_host, VEC),
          "packed": (gelu_tanh_ctm_packed_host, VEC), "staged": (gelu_tanh_ctm_staged_host, VEC)}
_compiled: dict[tuple[int, str, bool], Callable] = {}
_blocks_per_sm: dict[int, int] = {}


def _compile(device_index: int, fast: bool, variant: str = "ldg") -> Callable:
    """Compile once per (device, variant, mode); the shape stays symbolic."""
    key = (device_index, variant, fast)
    if key not in _compiled:
        host, div = _HOSTS[variant]
        with torch.cuda.device(device_index):
            n = cute.sym_int64(divisibility=div)
            fx = make_fake_compact_tensor(cutlass.BFloat16, (n,), assumed_align=16)
            fy = make_fake_compact_tensor(cutlass.BFloat16, (n,), assumed_align=16)
            ints = [cutlass.Int32(0)] * (3 if variant == "staged" else 2)
            _compiled[key] = cute.compile(
                host, fx, fy, *ints, make_fake_stream(), fast, options="--enable-tvm-ffi")
    return _compiled[key]


def _grid(device_index: int, n_vec: int) -> int:
    """min(SMs x resident blocks, CTAs)."""
    if device_index not in _blocks_per_sm:
        p = torch.cuda.get_device_properties(device_index)
        _blocks_per_sm[device_index] = (
            p.multi_processor_count * (p.max_threads_per_multi_processor // THREADS))
    ctas = (n_vec + THREADS - 1) // THREADS
    return max(1, min(_blocks_per_sm[device_index], ctas))


def _check(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("gelu_tanh_ctm: input must be CUDA")
    if x.dtype != torch.bfloat16:
        raise ValueError("gelu_tanh_ctm: input must be bf16")
    if not x.is_contiguous():
        raise ValueError("gelu_tanh_ctm: input must be contiguous")
    if x.numel() % VEC != 0:
        raise ValueError(f"gelu_tanh_ctm: numel must be a multiple of 8, got {x.numel()}")
    if x.numel() > _INT32_MAX:
        raise ValueError(f"gelu_tanh_ctm: numel {x.numel()} exceeds the Int32 index range")


def _stream(dev: int) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)


def _run_flat(variant: str, xf: torch.Tensor, yf: torch.Tensor, fast: bool) -> None:
    """Launch variant on flat views xf -> yf (numel % 8 == 0)."""
    dev = xf.device.index
    n_vec = xf.numel() // VEC
    _compile(dev, fast, variant)(xf, yf, n_vec, _grid(dev, n_vec), _stream(dev))


def gelu_tanh_ctm(x: torch.Tensor, fast: bool) -> torch.Tensor:
    """Tanh-GELU of a contiguous BF16 tensor, CUTLASS-DSL kernel.

    x: contiguous bf16 CUDA tensor, numel % 8 == 0, numel < 2**31.
    fast: False = tanhf, True = tanh.approx.f32; each mode is bitwise flux.gelu_tanh's.
    """
    _check(x)
    y = torch.empty_like(x)
    _run_flat("ldg", x.view(-1), y.view(-1), fast)
    return y


def gelu_tanh_ctm_inv(x: torch.Tensor, fast: bool) -> torch.Tensor:
    """Variant I: gelu_tanh_ctm with invariant (ld.global.nc) loads. Same contract."""
    _check(x)
    y = torch.empty_like(x)
    _run_flat("inv", x.view(-1), y.view(-1), fast)
    return y


def gelu_tanh_ctm_packed(x: torch.Tensor, fast: bool) -> torch.Tensor:
    """Variant P: gelu_tanh_ctm with nvcc's packed conversions. Same contract."""
    _check(x)
    y = torch.empty_like(x)
    _run_flat("packed", x.view(-1), y.view(-1), fast)
    return y


def gelu_tanh_ctm_staged(x: torch.Tensor, fast: bool) -> torch.Tensor:
    """Variant S: bulk-copy-staged gelu_tanh_ctm. Same contract.

    Whole TILE-element tiles run through the smem ring; a sub-tile remainder is finished
    by the direct 128-bit path in the same launch.
    """
    _check(x)
    y = torch.empty_like(x)
    dev = x.device.index
    n_tiles = x.numel() // TILE
    p = torch.cuda.get_device_properties(dev)
    grid = max(1, min(p.multi_processor_count * STAGED_CTAS_PER_SM,
                      max(n_tiles, (x.numel() // VEC + THREADS - 1) // THREADS)))
    _compile(dev, fast, "staged")(x.view(-1), y.view(-1), n_tiles, x.numel() // VEC,
                                  grid, _stream(dev))
    return y


DISPATCH_MAX_NUMEL = 2048 * 12288


def gelu_tanh_ctm_auto(x: torch.Tensor, fast: bool) -> torch.Tensor:
    """Composite: variant S at or below DISPATCH_MAX_NUMEL, the plain kernel above."""
    if x.numel() <= DISPATCH_MAX_NUMEL:
        return gelu_tanh_ctm_staged(x, fast)
    return gelu_tanh_ctm(x, fast)


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define("gelu_tanh_ctm(Tensor x, bool fast) -> Tensor")
_lib.define("gelu_tanh_ctm_inv(Tensor x, bool fast) -> Tensor")
_lib.define("gelu_tanh_ctm_packed(Tensor x, bool fast) -> Tensor")
_lib.define("gelu_tanh_ctm_staged(Tensor x, bool fast) -> Tensor")
_lib.define("gelu_tanh_ctm_auto(Tensor x, bool fast) -> Tensor")


@torch.library.impl(_lib, "gelu_tanh_ctm", "CUDA")
def _gelu_tanh_ctm_impl(x: torch.Tensor, fast: bool) -> torch.Tensor:
    return gelu_tanh_ctm(x, fast)


@torch.library.impl(_lib, "gelu_tanh_ctm_inv", "CUDA")
def _gelu_tanh_ctm_inv_impl(x: torch.Tensor, fast: bool) -> torch.Tensor:
    return gelu_tanh_ctm_inv(x, fast)


@torch.library.impl(_lib, "gelu_tanh_ctm_packed", "CUDA")
def _gelu_tanh_ctm_packed_impl(x: torch.Tensor, fast: bool) -> torch.Tensor:
    return gelu_tanh_ctm_packed(x, fast)


@torch.library.impl(_lib, "gelu_tanh_ctm_staged", "CUDA")
def _gelu_tanh_ctm_staged_impl(x: torch.Tensor, fast: bool) -> torch.Tensor:
    return gelu_tanh_ctm_staged(x, fast)


@torch.library.impl(_lib, "gelu_tanh_ctm_auto", "CUDA")
def _gelu_tanh_ctm_auto_impl(x: torch.Tensor, fast: bool) -> torch.Tensor:
    return gelu_tanh_ctm_auto(x, fast)
