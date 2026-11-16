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

"""flux.fused_sum_cat{3,6}_ctm: sum over the sequence axis + concat, CUTLASS Python DSL.

Same semantics as flat/custom_ops/fused_sum_cat.py: N inputs [B, S, D] bf16 -> [B, 1, N*D].
"""

from __future__ import annotations

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.library
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

VEC = 8  # bf16 elements per 128-bit access
_INT32_MAX = 2**31 - 1
_MAX_INPUTS = 6
_F32 = cutlass.Float32
_BF16 = cutlass.BFloat16
_I32 = cutlass.Int32
_I64 = cutlass.Int64


def _groups(ts: int) -> int:
    """Slice groups in the cross-slice reduce: 8 slices per group, 1 group below 8."""
    return ts // 8 if ts >= 8 else 1


@cute.kernel
def sum_cat_ctm_kernel(
    in0: cute.Tensor, in1: cute.Tensor, in2: cute.Tensor,
    in3: cute.Tensor, in4: cute.Tensor, in5: cute.Tensor,
    out: cute.Tensor,
    S: cutlass.Int32, D: cutlass.Int32, n_dtiles: cutlass.Int32, rows_per_thread: cutlass.Int32,
    N: cutlass.Constexpr[int], DT: cutlass.Constexpr[int], TS: cutlass.Constexpr[int],
    U: cutlass.Constexpr[int],
) -> None:
    """One CTA per (b, n, d-tile); TS row slices x DT/8 column-8 groups."""
    TD: cutlass.Constexpr[int] = DT // VEC
    G: cutlass.Constexpr[int] = _groups(TS)
    SPG: cutlass.Constexpr[int] = TS // G  # slices per group

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    dtile = bidx % n_dtiles
    bn = bidx // n_dtiles          # b * N + n
    n = bn % N
    b = bn // N
    td = tidx % TD
    ts = tidx // TD
    d0 = dtile * DT + td * VEC

    # Select the n-th input: merge one Int64 byte address through a runtime if-chain.
    addr = in0.iterator.toint()
    if cutlass.const_expr(N > 1):
        if n == 1:
            addr = in1.iterator.toint()
    if cutlass.const_expr(N > 2):
        if n == 2:
            addr = in2.iterator.toint()
    if cutlass.const_expr(N > 3):
        if n == 3:
            addr = in3.iterator.toint()
    if cutlass.const_expr(N > 4):
        if n == 4:
            addr = in4.iterator.toint()
    if cutlass.const_expr(N > 5):
        if n == 5:
            addr = in5.iterator.toint()
    # element offset of [b, ts, d0]; Int64 so B*S*D need not fit Int32
    first = (_I64(b) * _I64(S) + _I64(ts)) * _I64(D) + _I64(d0)
    p = cutlass.Pointer(
        cute.make_ptr(_BF16, addr + first * 2, cute.AddressSpace.gmem, assumed_align=16),
        dtype=_BF16)
    stride = TS * D  # elements between consecutive rows

    a0 = _F32(0.0)
    a1 = _F32(0.0)
    a2 = _F32(0.0)
    a3 = _F32(0.0)
    a4 = _F32(0.0)
    a5 = _F32(0.0)
    a6 = _F32(0.0)
    a7 = _F32(0.0)
    for k0 in cutlass.range(0, rows_per_thread, U):
        vs = []
        for u in cutlass.range_constexpr(U):
            vs.append((p + (k0 + u) * stride).load(16, count=VEC))
        for u in cutlass.range_constexpr(U):
            v = vs[u]
            a0 = a0 + v[0].to(_F32)
            a1 = a1 + v[1].to(_F32)
            a2 = a2 + v[2].to(_F32)
            a3 = a3 + v[3].to(_F32)
            a4 = a4 + v[4].to(_F32)
            a5 = a5 + v[5].to(_F32)
            a6 = a6 + v[6].to(_F32)
            a7 = a7 + v[7].to(_F32)

    # cross-slice reduction through smem in a fixed order (slices ascending inside
    # a group, then groups ascending): deterministic
    sred = cute.arch.alloc_smem(_F32, TS * DT, 16)
    sp = cutlass.Pointer(sred, dtype=_F32)
    (sp + (ts * DT + td * VEC)).store(
        cutlass.Vector.from_elements((a0, a1, a2, a3, a4, a5, a6, a7), _F32), alignment=16)
    cute.arch.sync_threads()

    op = cutlass.Pointer(out.iterator, dtype=_BF16)
    out_off = bn * D + d0
    if cutlass.const_expr(G == 1):
        if tidx < TD:
            tot = (sp + td * VEC).load(16, count=VEC)
            for j in cutlass.range_constexpr(1, TS):
                tot = tot + (sp + (j * DT + td * VEC)).load(16, count=VEC)
            (op + out_off).store(tot.to(_BF16), alignment=16)
    else:
        sgrp = cute.arch.alloc_smem(_F32, G * DT, 16)
        gp = cutlass.Pointer(sgrp, dtype=_F32)
        if tidx < TD * G:
            g = tidx // TD
            tdg = tidx % TD
            part = (sp + ((g * SPG) * DT + tdg * VEC)).load(16, count=VEC)
            for j in cutlass.range_constexpr(1, SPG):
                part = part + (sp + ((g * SPG + j) * DT + tdg * VEC)).load(16, count=VEC)
            (gp + (g * DT + tdg * VEC)).store(part, alignment=16)
        cute.arch.sync_threads()
        if tidx < TD:
            tot = (gp + td * VEC).load(16, count=VEC)
            for g2 in cutlass.range_constexpr(1, G):
                tot = tot + (gp + (g2 * DT + td * VEC)).load(16, count=VEC)
            (op + out_off).store(tot.to(_BF16), alignment=16)


@cute.jit
def sum_cat_ctm_host(
    in0: cute.Tensor, in1: cute.Tensor, in2: cute.Tensor,
    in3: cute.Tensor, in4: cute.Tensor, in5: cute.Tensor,
    out: cute.Tensor,
    S: cutlass.Int32, D: cutlass.Int32, n_dtiles: cutlass.Int32, rows_per_thread: cutlass.Int32,
    grid_x: cutlass.Int32, stream: cuda.CUstream,
    N: cutlass.Constexpr[int], DT: cutlass.Constexpr[int], TS: cutlass.Constexpr[int],
    U: cutlass.Constexpr[int],
) -> None:
    sum_cat_ctm_kernel(in0, in1, in2, in3, in4, in5, out, S, D, n_dtiles, rows_per_thread,
                       N, DT, TS, U).launch(
        grid=(grid_x, 1, 1), block=((DT // VEC) * TS, 1, 1), stream=stream)


# Host side


CONFIGS: dict[str, tuple[int, int]] = {
    "A": (64, 32),    # 256 threads, B*N*48 CTAs
    "B": (128, 16),   # 256 threads, B*N*24 CTAs
    "C": (64, 64),    # 512 threads, B*N*48 CTAs
    "D": (256, 8),    # 256 threads, B*N*12 CTAs
    "E": (32, 32),    # 128 threads, B*N*96 CTAs
}

DEFAULT_CONFIG = "A"

_compiled: dict[tuple, Callable] = {}


def _unroll_for(rows_per_thread: int) -> int:
    """Largest of 8/4/2/1 dividing the per-thread row count."""
    for u in (8, 4, 2, 1):
        if rows_per_thread % u == 0:
            return u
    return 1


def _compile(device_index: int, n_inputs: int, dt: int, ts: int, unroll: int) -> Callable:
    key = (device_index, n_inputs, dt, ts, unroll)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            fakes = [make_fake_compact_tensor(
                _BF16, (cute.sym_int64(divisibility=VEC, symbol=f"n{i}"),), assumed_align=16)
                for i in range(_MAX_INPUTS)]
            fout = make_fake_compact_tensor(
                _BF16, (cute.sym_int64(divisibility=VEC, symbol="o"),), assumed_align=16)
            ints = [cutlass.Int32(0)] * 5
            _compiled[key] = cute.compile(
                sum_cat_ctm_host, *fakes, fout, *ints, make_fake_stream(),
                n_inputs, dt, ts, unroll, options="--enable-tvm-ffi")
    return _compiled[key]


def _stream(dev: int) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)


def _check(tensors: tuple[torch.Tensor, ...], dt: int, ts: int) -> tuple[int, int, int]:
    t0 = tensors[0]
    if t0.dim() != 3:
        raise ValueError(f"fused_sum_cat_ctm: inputs must be [B, S, D], got {tuple(t0.shape)}")
    B, S, D = t0.shape
    for i, t in enumerate(tensors):
        if not t.is_cuda:
            raise ValueError(f"fused_sum_cat_ctm: input {i} must be CUDA")
        if t.dtype != torch.bfloat16:
            raise ValueError(f"fused_sum_cat_ctm: input {i} must be bf16, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"fused_sum_cat_ctm: input {i} must be contiguous")
        if tuple(t.shape) != (B, S, D):
            raise ValueError(f"fused_sum_cat_ctm: input {i} has shape {tuple(t.shape)}, "
                             f"expected {(B, S, D)}")
        if t.device != t0.device:
            raise ValueError("fused_sum_cat_ctm: inputs must share a device")
        if t.data_ptr() % 16 != 0:
            raise ValueError(f"fused_sum_cat_ctm: input {i} is not 16-byte aligned")
    if D % dt != 0:
        raise ValueError(f"fused_sum_cat_ctm: D={D} must be a multiple of the d-tile {dt}")
    if S % ts != 0:
        raise ValueError(f"fused_sum_cat_ctm: S={S} must be a multiple of the slice count {ts}")
    if B * S * D > _INT32_MAX or (dt // VEC) * ts > 1024:
        raise ValueError("fused_sum_cat_ctm: shape or configuration out of range")
    return B, S, D


def fused_sum_cat_ctm(tensors: tuple[torch.Tensor, ...], dt: int, ts: int) -> torch.Tensor:
    """Sum each [B, S, D] input over S and concatenate along the last axis.

    tensors: 2..6 contiguous bf16 tensors of one shape. dt: column tile (mult. of 8, divides D).
    ts: row slices per CTA (divides S), dt / 8 * ts <= 1024. Returns bf16 [B, 1, N*D].
    """
    n_in = len(tensors)
    if not 2 <= n_in <= _MAX_INPUTS:
        raise ValueError(f"fused_sum_cat_ctm: 2..{_MAX_INPUTS} inputs, got {n_in}")
    B, S, D = _check(tensors, dt, ts)
    out = torch.empty(B, 1, n_in * D, dtype=torch.bfloat16, device=tensors[0].device)
    dev = tensors[0].device.index
    rows = S // ts
    flat = [t.view(-1) for t in tensors]
    flat += [flat[0]] * (_MAX_INPUTS - n_in)  # unused slots; never dereferenced
    _compile(dev, n_in, dt, ts, _unroll_for(rows))(
        *flat, out.view(-1), S, D, D // dt, rows, B * n_in * (D // dt), _stream(dev))
    return out


def _config_for(n_inputs: int, B: int, S: int, D: int) -> tuple[int, int]:
    """Trace-time configuration choice by shape."""
    return CONFIGS[DEFAULT_CONFIG]


def fused_sum_cat3_ctm(t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """3-input fused_sum_cat_ctm with the shape-dispatched configuration."""
    dt, ts = _config_for(3, *t0.shape)
    return fused_sum_cat_ctm((t0, t1, t2), dt, ts)


def fused_sum_cat6_ctm(
    t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor,
    t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor,
) -> torch.Tensor:
    """6-input fused_sum_cat_ctm with the shape-dispatched configuration."""
    dt, ts = _config_for(6, *t0.shape)
    return fused_sum_cat_ctm((t0, t1, t2, t3, t4, t5), dt, ts)


def verify(B: int = 2, S: int = 64, D: int = 128) -> None:
    """Compile, run every configuration on one small shape and check it against fp64.

    Raises AssertionError beyond one bf16 ulp of the fp64 sum, or if two launches differ.
    """
    torch.manual_seed(0)
    for n_in in (3, 6):
        ts_list = [torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(n_in)]
        ref = torch.cat([t.double().sum(1, keepdim=True) for t in ts_list], -1).to(torch.bfloat16)
        for name, (dt, ts) in CONFIGS.items():
            if D % dt or S % ts:
                continue
            out = fused_sum_cat_ctm(tuple(ts_list), dt, ts)
            again = fused_sum_cat_ctm(tuple(ts_list), dt, ts)
            torch.cuda.synchronize()
            assert torch.equal(out, again), f"config {name}, N={n_in}: two launches differ"
            diff = (out.float() - ref.float()).abs()
            ulp = torch.exp2(torch.floor(torch.log2(ref.float().abs().clamp_min(2.0**-126))) - 7)
            bad = (diff > ulp).sum().item()
            assert bad == 0, f"config {name}, N={n_in}: {bad} elements beyond 1 bf16 ulp of fp64"
            print(f"verify N={n_in} config {name} (DT={dt}, TS={ts}) [{B},{S},{D}]: PASS")


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define("fused_sum_cat3_ctm(Tensor t0, Tensor t1, Tensor t2) -> Tensor")
_lib.define("fused_sum_cat6_ctm(Tensor t0, Tensor t1, Tensor t2, Tensor t3, Tensor t4, "
            "Tensor t5) -> Tensor")


@torch.library.impl(_lib, "fused_sum_cat3_ctm", "CUDA")
def _fsc3_impl(t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return fused_sum_cat3_ctm(t0, t1, t2)


@torch.library.impl(_lib, "fused_sum_cat6_ctm", "CUDA")
def _fsc6_impl(
    t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor,
    t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor,
) -> torch.Tensor:
    return fused_sum_cat6_ctm(t0, t1, t2, t3, t4, t5)
