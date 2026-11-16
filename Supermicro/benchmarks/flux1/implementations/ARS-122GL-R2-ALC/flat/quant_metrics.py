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

"""Offline quantization-quality metrics for the NVFP4 recipe ladder."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor

#: A quantizer: high-precision tensor -> quantize/dequantize round-trip.
#: Must be callable repeatedly; stochastic quantizers return a fresh draw.
QuantizeFn = Callable[[Tensor], Tensor]

#: Largest magnitude representable by OCP E2M1 (NVFP4 element format).
E2M1_MAX = 6.0
#: Largest magnitude representable by OCP E4M3 (MXFP8 element format).
E4M3_MAX = 448.0
#: NVFP4 scaling-block size (elements along the last dim).
NVFP4_BLOCK = 16
#: MXFP8 scaling-block size (elements along the last dim).
MXFP8_BLOCK = 32


# Primitive metrics


def relative_mse(ref: Tensor, approx: Tensor) -> float:
    """Relative mean-squared error ||approx - ref||^2 / ||ref||^2; 0.0 for all-zero ref."""
    r = ref.to(torch.float32)
    a = approx.to(torch.float32)
    denom = torch.sum(r * r).item()
    if denom == 0.0:
        return 0.0
    return torch.sum((a - r) ** 2).item() / denom


def cosine(ref: Tensor, approx: Tensor) -> float:
    """Cosine similarity between flattened ref and approx; 0.0 if either is all-zero."""
    r = ref.to(torch.float32).reshape(-1)
    a = approx.to(torch.float32).reshape(-1)
    nr = torch.linalg.vector_norm(r).item()
    na = torch.linalg.vector_norm(a).item()
    if nr == 0.0 or na == 0.0:
        return 0.0
    return (torch.dot(r, a).item() / (nr * na))


def scale_ratio(ref: Tensor, approx: Tensor) -> float:
    """Magnitude drift ||approx|| / ||ref||; 0.0 for an all-zero reference."""
    nr = torch.linalg.vector_norm(ref.to(torch.float32)).item()
    if nr == 0.0:
        return 0.0
    return torch.linalg.vector_norm(approx.to(torch.float32)).item() / nr


def relative_bias(
    ref: Tensor, quantize: QuantizeFn, num_draws: int = 32,
) -> float:
    """Systematic offset ||E[Q(ref)] - ref|| / ||ref||, E[] averaged over num_draws draws."""
    if num_draws < 1:
        raise ValueError("num_draws must be >= 1")
    r = ref.to(torch.float32)
    denom = torch.linalg.vector_norm(r).item()
    if denom == 0.0:
        return 0.0
    acc = torch.zeros_like(r)
    for _ in range(num_draws):
        acc += quantize(ref).to(torch.float32)
    acc /= num_draws
    return torch.linalg.vector_norm(acc - r).item() / denom


def relative_bias_mc_floor(rel_mse: float, num_draws: int) -> float:
    """Monte-Carlo noise floor of relative_bias: sqrt(rel_mse / num_draws)."""
    return math.sqrt(max(rel_mse, 0.0) / num_draws)


def debiased_relative_bias(rel_bias: float, rel_mse: float,
                           num_draws: int) -> float:
    """Remove the Monte-Carlo floor from a measured relBias: sqrt(max(relBias^2 - floor^2, 0))."""
    floor = relative_bias_mc_floor(rel_mse, num_draws)
    return math.sqrt(max(rel_bias ** 2 - floor ** 2, 0.0))


# Block-scale emulation (clipping fraction, dynamic range)


def _as_blocks(x: Tensor, block_size: int) -> Tensor:
    """Reshape the last dim into (..., n_blocks, block_size); last dim must be a multiple."""
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            f"last dim {x.shape[-1]} not a multiple of {block_size}",
        )
    return x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)


def block_amax(x: Tensor, block_size: int) -> Tensor:
    """Per-block max|x| along the last dim, shape (..., n_blocks)."""
    return _as_blocks(x.to(torch.float32).abs(), block_size).amax(dim=-1)


def block_scale_dynamic_range(x: Tensor, block_size: int) -> float:
    """log2(max_block_amax / min_nonzero_block_amax); 0.0 if no block is non-zero."""
    amax = block_amax(x, block_size).reshape(-1)
    nz = amax[amax > 0]
    if nz.numel() == 0:
        return 0.0
    return math.log2(nz.max().item() / nz.min().item())


def _round_to_e4m3(x: Tensor) -> Tensor:
    """Round a non-negative tensor onto the E4M3 grid (RNE, saturating at 448), as float32."""
    return x.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def nvfp4_block_limits(x: Tensor) -> Tensor:
    """Largest magnitude NVFP4 can represent in each 16-element block, shape (..., n_blocks).

    Emulates TE's two-level scaling: a per-tensor FP32 encode scale, then a
    per-block E4M3 scale.
    """
    xf = x.to(torch.float32)
    amax_g = xf.abs().max()
    if amax_g.item() == 0.0:
        return torch.zeros(
            _as_blocks(xf, NVFP4_BLOCK).shape[:-1], device=xf.device,
        )
    encode = (E2M1_MAX * E4M3_MAX) / amax_g
    blk = block_amax(xf, NVFP4_BLOCK)
    blk_scale = _round_to_e4m3(blk * encode / E2M1_MAX)
    return E2M1_MAX * blk_scale / encode


def clipping_fraction(x: Tensor, block_limits: Tensor,
                      block_size: int) -> float:
    """Fraction of elements whose magnitude exceeds their block's limit, in [0, 1].

    block_limits has shape (..., n_blocks), e.g. from nvfp4_block_limits.
    """
    blocks = _as_blocks(x.to(torch.float32).abs(), block_size)
    limit = block_limits.unsqueeze(-1) * (1.0 + 1e-6)
    return (blocks > limit).to(torch.float32).mean().item()


def saturation_fraction(ref: Tensor, approx: Tensor,
                        block_size: int) -> float:
    """Empirical saturation rate of a real round-trip, in [0, 1].

    An element is saturated when it moved toward zero onto its block's largest magnitude.
    """
    r = _as_blocks(ref.to(torch.float32).abs(), block_size)
    a = _as_blocks(approx.to(torch.float32).abs(), block_size)
    top = a.amax(dim=-1, keepdim=True)
    hit = (a >= top) & (top > 0) & (r > a)
    return hit.to(torch.float32).mean().item()


# Rounding-decision metrics (per-slot bias, spatial correlation)

#: OCP E2M1 magnitudes, ascending. The NVFP4 element grid.
E2M1_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4_block_units(x: Tensor) -> tuple[Tensor, Tensor]:
    """Express x in E2M1 block units.

    Returns (v, divisor) with v = x / divisor in [-6, 6] and divisor
    broadcastable against _as_blocks(x, 16).
    """
    xf = x.to(torch.float32)
    amax_g = xf.abs().max()
    if amax_g.item() == 0.0:
        blocks = _as_blocks(xf, NVFP4_BLOCK)
        return blocks, torch.ones_like(blocks[..., :1])
    encode = (E2M1_MAX * E4M3_MAX) / amax_g
    blk_scale = _round_to_e4m3(block_amax(xf, NVFP4_BLOCK) * encode / E2M1_MAX)
    divisor = (blk_scale / encode).unsqueeze(-1).clamp(min=1e-38)
    return _as_blocks(xf, NVFP4_BLOCK) / divisor, divisor


def rounding_residual(ref: Tensor, approx: Tensor,
                      grid: Sequence[float] = E2M1_GRID) -> Tensor:
    """Recover the rounding decision from a round-trip, as 1{up} - p, in grid-step units."""
    v, divisor = _nvfp4_block_units(ref)
    g = torch.tensor(grid, dtype=torch.float32, device=v.device)
    # Clamping here is what makes a saturated element land on hi and so
    # be excluded by the hi != a test below.
    a = v.abs().clamp(max=float(g[-1]))
    w = (_as_blocks(approx.to(torch.float32), NVFP4_BLOCK) / divisor).abs()
    idx = torch.searchsorted(g, a.contiguous())
    hi_i = idx.clamp(max=len(grid) - 1)
    lo_i = (hi_i - 1).clamp(min=0)
    hi, lo = g[hi_i], g[lo_i]
    step = hi - lo
    # Elements already on a grid point (including every saturated one) took
    # no rounding decision and are excluded.
    live = (step > 0) & (hi != a)
    return torch.where(live, (w - a) / step.clamp(min=1e-30),
                       torch.zeros_like(a))


def rounding_residual_mask(ref: Tensor,
                           grid: Sequence[float] = E2M1_GRID) -> Tensor:
    """Bool mask, shaped like rounding_residual, of elements that took a rounding decision."""
    v, _ = _nvfp4_block_units(ref)
    g = torch.tensor(grid, dtype=torch.float32, device=v.device)
    a = v.abs().clamp(max=float(g[-1]))
    idx = torch.searchsorted(g, a.contiguous())
    hi_i = idx.clamp(max=len(grid) - 1)
    lo_i = (hi_i - 1).clamp(min=0)
    return ((g[hi_i] - g[lo_i]) > 0) & (g[hi_i] != a)


def freeze_block_slot(quantize: QuantizeFn, slot: int,
                      block_size: int = NVFP4_BLOCK,
                      grid: Sequence[float] = E2M1_GRID) -> QuantizeFn:
    """Wrap a quantizer so one slot of every scaling block never rounds up."""
    if not 0 <= slot < block_size:
        raise ValueError(f"slot {slot} outside block of {block_size}")

    def fn(x: Tensor) -> Tensor:
        approx = quantize(x)
        v, divisor = _nvfp4_block_units(x)
        g = torch.tensor(grid, dtype=torch.float32, device=v.device)
        a = v.abs().clamp(max=float(g[-1]))
        hi_i = torch.searchsorted(g, a.contiguous()).clamp(max=len(grid) - 1)
        lo_i = (hi_i - 1).clamp(min=0)
        hi, lo = g[hi_i], g[lo_i]
        # Elements on a grid point, or saturated, took no rounding decision;
        # leave them in place.
        live = ((hi - lo) > 0) & (hi != a)
        down = torch.sign(v) * torch.where(live, lo, a) * divisor
        blocks = _as_blocks(approx.to(torch.float32), block_size).clone()
        blocks[..., slot] = down[..., slot]
        return blocks.reshape(approx.shape).to(approx.dtype)

    return fn


def per_slot_relative_bias(ref: Tensor, quantize: QuantizeFn,
                           num_draws: int = 32,
                           block_size: int = NVFP4_BLOCK) -> list[float]:
    """relative_bias computed per position within the scaling block; returns block_size floats."""
    if num_draws < 1:
        raise ValueError("num_draws must be >= 1")
    r = _as_blocks(ref.to(torch.float32), block_size)
    acc = torch.zeros_like(r)
    for _ in range(num_draws):
        acc += _as_blocks(quantize(ref).to(torch.float32), block_size)
    acc /= num_draws
    out = []
    for j in range(block_size):
        rj = r[..., j].reshape(-1)
        denom = torch.linalg.vector_norm(rj).item()
        aj = acc[..., j].reshape(-1)
        out.append(0.0 if denom == 0.0
                   else torch.linalg.vector_norm(aj - rj).item() / denom)
    return out


def slot_bias_ratio(slot_biases: Sequence[float]) -> float:
    """Worst slot over the median slot: max(slot_biases) / median; 1.0 if the median is 0."""
    vals = sorted(float(v) for v in slot_biases)
    if not vals:
        return 1.0
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    return 1.0 if med <= 0.0 else vals[-1] / med


def per_slot_rounding_bias(ref: Tensor, approx: Tensor,
                           ) -> tuple[list[float], list[float]]:
    """Per-slot mean of the rounding decision from a single draw.

    Returns (means, stderrs), each of length NVFP4_BLOCK.
    """
    u = rounding_residual(ref, approx)
    mask = rounding_residual_mask(ref).expand_as(u)
    means: list[float] = []
    errs: list[float] = []
    for j in range(NVFP4_BLOCK):
        uj = u[..., j].reshape(-1)[mask[..., j].reshape(-1)]
        if uj.numel() < 2:
            means.append(0.0)
            errs.append(1.0)
            continue
        means.append(uj.mean().item())
        errs.append(
            (uj.std().item() / math.sqrt(uj.numel())) or 1e-12,
        )
    return means, errs


def slot_rounding_bias_z(means: Sequence[float],
                         stderrs: Sequence[float]) -> float:
    """Worst per-slot rounding bias in standard errors: max_j |mean_j| / stderr_j."""
    return max((abs(m) / e if e > 0 else float("inf"))
               for m, e in zip(means, stderrs))


def residual_autocorrelation(u: Tensor, lags: Sequence[int] = (1, 16),
                             mask: Tensor | None = None) -> dict[int, float]:
    """Spatial correlation of the rounding decision at each lag; returns {lag: rho}."""
    flat = u.reshape(-1).to(torch.float32)
    keep = (torch.ones_like(flat, dtype=torch.bool) if mask is None
            else mask.reshape(-1))
    out: dict[int, float] = {}
    for lag in lags:
        if lag <= 0 or lag >= flat.numel():
            out[int(lag)] = 0.0
            continue
        sel = keep[:-lag] & keep[lag:]
        a, b = flat[:-lag][sel], flat[lag:][sel]
        if a.numel() < 2:
            out[int(lag)] = 0.0
            continue
        a = a - a.mean()
        b = b - b.mean()
        denom = (torch.linalg.vector_norm(a)
                 * torch.linalg.vector_norm(b)).item()
        out[int(lag)] = 0.0 if denom == 0.0 else torch.dot(a, b).item() / denom
    return out


def autocorrelation_noise_floor(n: int, sigma: float = 3.0) -> float:
    """Autocorrelation noise floor sigma / sqrt(n); 1.0 for n < 1."""
    return 1.0 if n < 1 else sigma / math.sqrt(n)


# GEMM-output error


def gemm_output_metrics(lhs: Tensor, rhs: Tensor, quantize_lhs: QuantizeFn,
                        quantize_rhs: QuantizeFn | None = None,
                        num_draws: int = 32) -> dict[str, float]:
    """Score the product quantize_lhs(lhs) @ rhs against lhs @ rhs.

    Returns rel_mse, rel_bias, rel_bias_mc_floor and rel_bias_debiased.
    """
    if num_draws < 1:
        raise ValueError("num_draws must be >= 1")
    a = lhs.to(torch.float32)
    b = rhs.to(torch.float32)
    ref = a @ b
    bq = b if quantize_rhs is None else quantize_rhs(rhs).to(torch.float32)
    denom = torch.linalg.vector_norm(ref).item()
    if denom == 0.0:
        return {"rel_mse": 0.0, "rel_bias": 0.0,
                "rel_bias_mc_floor": 0.0, "rel_bias_debiased": 0.0}
    acc = torch.zeros_like(ref)
    first: Tensor | None = None
    for _ in range(num_draws):
        out = quantize_lhs(lhs).to(torch.float32) @ bq
        if first is None:
            first = out
        acc += out
    acc /= num_draws
    rel_mse = (torch.sum((first - ref) ** 2).item()
               / torch.sum(ref * ref).item())
    bias = torch.linalg.vector_norm(acc - ref).item() / denom
    return {
        "rel_mse": rel_mse,
        "rel_bias": bias,
        "rel_bias_mc_floor": relative_bias_mc_floor(rel_mse, num_draws),
        "rel_bias_debiased": debiased_relative_bias(bias, rel_mse, num_draws),
    }


def cross_cast_residual_corr(ref: Tensor, approx_a: Tensor,
                             approx_b: Tensor) -> float:
    """Pearson correlation between the rounding residuals of two casts of one tensor."""
    mask = rounding_residual_mask(ref).reshape(-1)
    ua = rounding_residual(ref, approx_a).reshape(-1)[mask]
    ub = rounding_residual(ref, approx_b).reshape(-1)[mask]
    if ua.numel() < 2:
        return 0.0
    ua = ua - ua.mean()
    ub = ub - ub.mean()
    denom = (torch.linalg.vector_norm(ua)
             * torch.linalg.vector_norm(ub)).item()
    return 0.0 if denom == 0.0 else torch.dot(ua, ub).item() / denom


# Metric bundles


@dataclass(frozen=True)
class FpropMetrics:
    """Forward-path quantization quality for one tensor."""

    rel_mse: float
    clip_frac: float
    sat_frac: float
    block_scale_range_log2: float


@dataclass(frozen=True)
class BpropMetrics:
    """Backward-path quantization quality for one tensor."""

    cos: float
    rel_bias: float
    rel_bias_debiased: float
    rel_bias_mc_floor: float
    scale_ratio: float
    rel_mse: float
    num_draws: int


def fprop_metrics(ref: Tensor, quantize: QuantizeFn,
                  block_limits_fn: Callable[[Tensor], Tensor],
                  block_size: int) -> FpropMetrics:
    """Score a forward GEMM operand.

    block_limits_fn and block_size must match the recipe quantize implements.
    """
    approx = quantize(ref)
    return FpropMetrics(
        rel_mse=relative_mse(ref, approx),
        clip_frac=clipping_fraction(ref, block_limits_fn(ref), block_size),
        sat_frac=saturation_fraction(ref, approx, block_size),
        block_scale_range_log2=block_scale_dynamic_range(ref, block_size),
    )


def bprop_metrics(ref: Tensor, quantize: QuantizeFn,
                  num_draws: int = 32) -> BpropMetrics:
    """Score a backward GEMM operand (a gradient); num_draws is K for the E[Q(g)] estimate."""
    single = quantize(ref)
    rel_mse = relative_mse(ref, single)
    bias = relative_bias(ref, quantize, num_draws)
    return BpropMetrics(
        cos=cosine(ref, single),
        rel_bias=bias,
        rel_bias_debiased=debiased_relative_bias(bias, rel_mse, num_draws),
        rel_bias_mc_floor=relative_bias_mc_floor(rel_mse, num_draws),
        scale_ratio=scale_ratio(ref, single),
        rel_mse=rel_mse,
        num_draws=num_draws,
    )


@dataclass(frozen=True)
class SRStreamMetrics:
    """Rounding-stream quality for one tensor (unpooled axes)."""

    slot_bias_ratio: float
    slot_bias_max: float
    slot_bias_median: float
    slot_rounding_bias_z: float
    slot_rounding_bias_worst: float
    resid_autocorr_lag1: float
    resid_autocorr_lag16: float
    resid_autocorr_floor: float
    gemm_rel_bias: float
    gemm_rel_bias_floor: float
    gemm_rel_mse: float
    num_draws: int


#: Worst slot must not exceed the median slot by more than this (sound stream: ~1).
SLOT_BIAS_RATIO_LIMIT = 2.5
#: Worst per-slot rounding bias, in standard errors of its own mean.
SLOT_ROUNDING_BIAS_Z_LIMIT = 5.0
#: Multiple of the analytic 3/sqrt(n) autocorrelation floor that counts as real structure.
RESID_AUTOCORR_FLOOR_MULT = 4.0
#: GEMM-output bias must not exceed this multiple of its own Monte-Carlo floor.
GEMM_BIAS_FLOOR_MULT = 3.0


def sr_stream_metrics(ref: Tensor, quantize: QuantizeFn,
                      rhs: Tensor | None = None,
                      num_draws: int = 32) -> SRStreamMetrics:
    """Score a stochastic-rounding stream on the unpooled axes.

    ref is (M, K) with K a multiple of 16; rhs is (K, N), or None for a fixed
    seeded gaussian.
    """
    slots = per_slot_relative_bias(ref, quantize, num_draws)
    ordered = sorted(slots)
    n = len(ordered)
    med = (ordered[n // 2] if n % 2
           else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2]))
    single = quantize(ref)
    mask = rounding_residual_mask(ref)
    u = rounding_residual(ref, single)
    rho = residual_autocorrelation(u, (1, NVFP4_BLOCK), mask)
    slot_means, slot_errs = per_slot_rounding_bias(ref, single)
    if rhs is None:
        gen = torch.Generator(device="cpu").manual_seed(0)
        rhs = torch.randn(ref.shape[-1], 256, generator=gen,
                          dtype=torch.float32).to(ref.device)
    gemm = gemm_output_metrics(ref, rhs, quantize, None, num_draws)
    return SRStreamMetrics(
        slot_bias_ratio=slot_bias_ratio(slots),
        slot_bias_max=max(slots),
        slot_bias_median=med,
        slot_rounding_bias_z=slot_rounding_bias_z(slot_means, slot_errs),
        slot_rounding_bias_worst=max(abs(m) for m in slot_means),
        resid_autocorr_lag1=rho[1],
        resid_autocorr_lag16=rho[NVFP4_BLOCK],
        resid_autocorr_floor=autocorrelation_noise_floor(
            int(mask.sum().item()),
        ),
        gemm_rel_bias=gemm["rel_bias"],
        gemm_rel_bias_floor=gemm["rel_bias_mc_floor"],
        gemm_rel_mse=gemm["rel_mse"],
        num_draws=num_draws,
    )


def sr_stream_verdict(m: SRStreamMetrics) -> tuple[bool, list[str]]:
    """Apply the rounding-stream gate; returns (passed, reasons) with every failing clause."""
    reasons: list[str] = []
    if m.slot_rounding_bias_z > SLOT_ROUNDING_BIAS_Z_LIMIT:
        reasons.append(
            f"per-slot rounding bias: worst slot "
            f"{m.slot_rounding_bias_worst:+.5f} at "
            f"{m.slot_rounding_bias_z:.1f} sigma "
            f"(limit {SLOT_ROUNDING_BIAS_Z_LIMIT})",
        )
    if m.slot_bias_ratio > SLOT_BIAS_RATIO_LIMIT:
        reasons.append(
            f"per-slot bias: worst slot {m.slot_bias_ratio:.2f}x the median "
            f"(limit {SLOT_BIAS_RATIO_LIMIT})",
        )
    lim = RESID_AUTOCORR_FLOOR_MULT * m.resid_autocorr_floor
    for lag, val in ((1, m.resid_autocorr_lag1),
                     (NVFP4_BLOCK, m.resid_autocorr_lag16)):
        if abs(val) > lim:
            reasons.append(
                f"residual autocorrelation at lag {lag}: {val:+.4f} "
                f"(limit +/-{lim:.4f})",
            )
    gemm_lim = GEMM_BIAS_FLOOR_MULT * m.gemm_rel_bias_floor
    if m.gemm_rel_bias > gemm_lim:
        reasons.append(
            f"GEMM-output bias {m.gemm_rel_bias:.5f} exceeds "
            f"{GEMM_BIAS_FLOOR_MULT}x its MC floor ({gemm_lim:.5f})",
        )
    return (not reasons), reasons


# Aggregation

#: Direction in which each metric gets worse: +1 larger, -1 smaller,
#: 0 distance from 1.0, +2 larger absolute value.
METRIC_POLARITY: dict[str, int] = {
    "rel_mse": +1,
    "clip_frac": +1,
    "sat_frac": +1,
    "block_scale_range_log2": +1,
    "cos": -1,
    "rel_bias": +1,
    "rel_bias_debiased": +1,
    "rel_bias_mc_floor": +1,
    "scale_ratio": 0,
    "num_draws": 0,
    # Rounding-stream axes.
    "slot_bias_ratio": +1,
    "slot_bias_max": +1,
    "slot_rounding_bias_z": +1,
    "slot_rounding_bias_worst": +1,
    "gemm_rel_bias": +1,
    "resid_autocorr_lag1": +2,
    "resid_autocorr_lag16": +2,
}


@dataclass(frozen=True)
class MetricRecord:
    """One scored tensor, tagged with everything needed to aggregate it."""

    gemm_class: str
    block_index: int
    role: str
    step: int
    recipe: str
    metrics: dict[str, float]


def _badness(name: str, value: float) -> float:
    """Map a metric value onto a scale where larger is always worse (KeyError if unknown)."""
    pol = METRIC_POLARITY[name]
    if pol == 2:
        return abs(value)
    if pol > 0:
        return value
    if pol < 0:
        return -value
    return abs(value - 1.0)


def worst_case(records: Iterable[MetricRecord],
               group_by: Sequence[str] = ("gemm_class", "role"),
               ) -> dict[tuple, dict[str, float]]:
    """Reduce records to the worst value of every metric per group.

    Returns {group_key_tuple: {metric_name: worst_value}}.
    """
    out: dict[tuple, dict[str, float]] = {}
    for rec in records:
        key = tuple(getattr(rec, f) for f in group_by)
        acc = out.setdefault(key, {})
        for name, value in rec.metrics.items():
            if name not in METRIC_POLARITY:
                continue
            if name not in acc or _badness(name, value) > _badness(
                name, acc[name],
            ):
                acc[name] = value
    return out


def record_to_row(rec: MetricRecord) -> dict[str, float | str | int]:
    """Flatten a record into one dict with the tag fields and every metric."""
    row = asdict(rec)
    metrics = row.pop("metrics")
    row.update(metrics)
    return row
