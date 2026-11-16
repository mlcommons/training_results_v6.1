# SPDX-License-Identifier: Apache-2.0
"""Chain 1 -- the GELU-out dual cast in ONE read.

Registers two ``torch.ops.flux`` ops, both dispatched on ``numel`` at trace time:

* ``flux.gelu_tanh_amax(x, rht_matrix) -> (y, amax)`` -- the producer
  (``gelu_tanh_amax_ctm.py``, the CUTLASS-DSL kernel -- the only
  implementation on every architecture): ``y`` is bitwise
  ``flux.gelu_tanh_ctm_v2_auto(x, True)``; ``amax`` is FP32 ``[2]`` =
  ``(max|y|, post-RHT columnwise amax)``, bitwise pass 1
  (``flux.mxfp8_rowwise_cast_amax_ctm(y, R, True)[2]``) and TE's
  ``_amax_columnwise``.
* ``flux.onepass_act(y, rht_matrix, amax) -> (mx_data, mx_scales, col_data,
  col_scales)`` -- the cast, the CUTLASS-DSL kernel ``onepass_act_ctm.py``:
  one TMA-staged read of ``y`` emitting the MXFP8 rowwise
  half and the NVFP4 columnwise half, bitwise today's two-pass chain (pass 1 +
  ``flux.te_nvfp4_quantize_with_amax``) and bitwise the shipped
  ``flux.dual_cast_one_read``.

Today's chain reads the GELU output from DRAM twice (pass 1 computes the MXFP8
half and the amax, pass 2 the NVFP4 half). With the producer emitting the
post-RHT amax the cast needs, the tensor is read once. Measured worth
(two nodes, warm paired, routed): +31.1 / +30.9 us
per call at ``[8192, 12288]`` and +49.6 / +46.8 at ``[16384, 12288]`` --
+3.07 / +2.95 ms/iter at B=32 op-level.

Dispatch, verified by kernel name AND grid per micro-batch in
``tests/test_onepass_act.py``:

* producer: ``g128s3w8``; row splits doubled (``+o2``) from
  ``numel >= 8192 * 12288``;
* cast: :data:`CAST_TIERS` -- a per-architecture table of ``(variant, grid
  factor)`` on ``numel`` (one kernel does not serve every size).
  GB300 (sm_103): the 2-CTA ``A_stsm`` geometry at every M
  (-3.5..-5.2 us per call against the CUDA cast on three nodes). Rubin
  (sm_107): ``A_stsm`` below 2048 rows, the 3-CTA ``W_stsm`` at the CUDA
  kernel's x2 grid up to 8192 rows, and ``W_stsm`` at a x4 grid from there:
  the 3-CTA grid at x2
  runs 1.96 waves at ``[16384]`` and the tail costs 2-3 us; x4 recovers it and
  costs it back where a CTA's work falls below ~4 flush windows.

Shapes: ``cols % 128 == 0``; rows a multiple of the producer tile (128) and of
the cast variant's flush window (256 for ``A_stsm``, 128 for ``W_stsm``). Every
production GELU-out shape (``B * 256`` and ``B * 512`` rows, 12288 columns)
qualifies; anything else is an error, not a fallback.

The producer's completion counter (``int32 [1]``, self-resetting) is one
persistent buffer per device, allocated on first use. It is sound because
every launch on a device is stream-ordered behind the previous one -- the
compute stream in production, and a captured graph replays in stream order.
"""

from __future__ import annotations

import json
import os
import tempfile

import torch
import torch.library

_ctm_mod = None
_prod_ctm_mod = None

#: Producer dispatch threshold on ``numel``.
#: **Measured on GB300 only**; it chooses a grid mode of one kernel, so it turns on SM
#: count and wave quantisation rather than cache capacity. Override with the variable.
PROD_O2_MIN_NUMEL = int(os.environ.get("FLAT_ONEPASS_PROD_O2_MIN_NUMEL", 0)) or 8192 * 12288
#: ``(variant name, mode)``; mode bit 1 = x2 row splits. One geometry, the CUDA kernel's
#: ``g128s3w8`` (8 consumer warps x 3 stages, 2 CTAs/SM), named ``G1_ctm`` in the DSL module.
PROD_SMALL = ("G1_ctm", 0)
PROD_LARGE = ("G1_ctm", 2)
#: The producer kernel symbol both modes launch; the modes differ
#: only in the grid. The DSL ``LW`` producer is the only implementation on every
#: architecture -- bitwise the retired CUDA kernel at every production M in both modes on
#: sm_103a and sm_107a, at op-level parity, and at step-level parity on two disjoint 4-node
#: sets per cluster.
PROD_KERNEL = "gelu_tanh_amax_ctm_kernel"

#: Cast dispatch: capability -> tiers of ``(max numel, exclusive; variant; grid factor)``,
#: the last tier's bound ``None`` = everything at and above the previous bound. An
#: allow-list of measured architectures: an unlisted capability is an error,
#: never a guess -- chain 1 itself is gated to the same two by ``nvfp4_ops._ONEPASS_ACT_ARCHS``.
CAST_TIERS: dict[tuple[int, int], tuple[tuple[int | None, str, int], ...]] = {
    # A_stsm wins at every M on three GB300 nodes; no size split.
    (10, 3): ((None, "A_stsm", 1),),
    (10, 7): ((2048 * 12288, "A_stsm", 1), (8192 * 12288, "W_stsm", 2), (None, "W_stsm", 4)),
}
#: The cast kernel symbol every tier launches (the variants differ by constexpr, not by name).
CAST_KERNEL = "onepass_act_ctm_kernel"


def _prod_ctm():
    """Import (once) the DSL producer module ``flat.custom_ops.gelu_tanh_amax_ctm``."""
    global _prod_ctm_mod
    if _prod_ctm_mod is None:
        from flat.custom_ops import gelu_tanh_amax_ctm as _m
        _prod_ctm_mod = _m
    return _prod_ctm_mod


def _ctm():
    """Import (once) the DSL cast module ``flat.custom_ops.onepass_act_ctm``."""
    global _ctm_mod
    if _ctm_mod is None:
        from flat.custom_ops import onepass_act_ctm as _m
        _ctm_mod = _m
    return _ctm_mod


def cast_tiers(capability: tuple[int, int] | None = None) -> tuple[tuple[int | None, str, int], ...]:
    """The cast tiers for ``capability`` (this device's when ``None``).

    Raises:
        RuntimeError: On an architecture that has not been measured.
    """
    cc = tuple(torch.cuda.get_device_capability()) if capability is None else tuple(capability)
    tiers = CAST_TIERS.get(cc)
    if tiers is None:
        raise RuntimeError(
            f"onepass_act: no measured cast dispatch for capability {cc}; "
            f"measured: {sorted(CAST_TIERS)}")
    return tiers


def pick_cast(numel: int, capability: tuple[int, int] | None = None) -> tuple[str, int]:
    """``(variant, factor)`` the cast dispatch selects for a tensor of ``numel`` elements."""
    for bound, variant, factor in cast_tiers(capability):
        if bound is None or numel < bound:
            return variant, factor
    raise AssertionError("unreachable: the last tier is unbounded")


def cast_arms(capability: tuple[int, int] | None = None) -> tuple[tuple[str, int], ...]:
    """Every distinct ``(variant, factor)`` the dispatch can launch on ``capability``."""
    return tuple(dict.fromkeys((v, f) for _, v, f in cast_tiers(capability)))


def pick_producer(numel: int) -> tuple[str, int]:
    """``(variant, mode)`` the producer dispatch selects for a tensor of ``numel`` elements."""
    return PROD_LARGE if numel >= PROD_O2_MIN_NUMEL else PROD_SMALL


def onepass_act_variant(
    x: torch.Tensor, rht_matrix: torch.Tensor, amax: torch.Tensor, variant: str, factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The cast with the DSL geometry and grid factor named explicitly.

    Args:
        x: BF16 ``[rows, cols]`` contiguous; ``rows`` a multiple of the variant's
            flush window (256 for ``A_stsm``, 128 for ``W_stsm``), ``cols % 128 == 0``.
        rht_matrix: ``NVFP4Quantizer.rht_matrix``, BF16 ``[16, 16]``.
        amax: FP32 ``[1]`` CUDA -- the **final** post-RHT columnwise amax,
            complete on the current stream; read once in the kernel prologue.
        variant: A key of ``onepass_act_ctm.VARIANTS``.
        factor: One of ``onepass_act_ctm.FACTORS``.

    Returns:
        ``(mx_data, mx_scales, col_data, col_scales)`` -- E4M3 ``[rows, cols]``,
        GEMM-swizzled E8M0 ``[rows, cols/32]``, E2M1 ``[cols, rows/2]`` bytes,
        GEMM-swizzled E4M3 ``[cols, rows/16]``, all uint8; the contract of
        ``flux.dual_cast_one_read(x, rht_matrix, True, amax)``.

    Raises:
        RuntimeError: On an unsupported shape, dtype, variant or factor -- the
            op's contract is an error, never a fallback.
    """
    try:
        return _ctm().onepass_act_ctm(x, rht_matrix, amax, variant, factor)
    except ValueError as e:
        raise RuntimeError(f"onepass_act: {e}") from e


def gelu_tanh_amax_variant(
    x: torch.Tensor, rht_matrix: torch.Tensor, variant: str, mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The producer with the variant and mode named explicitly.

    Args:
        x: Pre-activation, BF16 ``[rows, cols]`` contiguous, ``rows % 128 == 0``,
            ``cols % 128 == 0``.
        rht_matrix: ``NVFP4Quantizer.rht_matrix``, BF16 ``[16, 16]``.
        variant: A key of ``gelu_tanh_amax_ctm.VARIANTS``.
        mode: Bit 1 = row splits x2.

    Returns:
        ``(y, amax)`` -- ``y`` BF16 bitwise ``flux.gelu_tanh_ctm_v2_auto(x, True)``;
        ``amax`` FP32 ``[2]`` = ``(max|y|, post-RHT columnwise amax)``.

    Raises:
        RuntimeError: On an unsupported shape, dtype or variant -- the op's contract is
            an error, never a fallback.
    """
    try:
        return _prod_ctm().gelu_tanh_amax_ctm(x, rht_matrix, variant, mode)
    except ValueError as e:
        raise RuntimeError(f"gelu_tanh_amax: {e}") from e


def live_cast_kernel(rows: int, cols: int) -> tuple[str, tuple[int, int]]:
    """``(kernel-name substring, (grid_x, grid_y))`` :func:`onepass_act` launches for
    a ``[rows, cols]`` input on this device -- the dispatch's own statement of what it
    does, for the by-name / by-grid census."""
    variant, factor = pick_cast(rows * cols)
    cs, rs, _, _ = cast_grid(rows, cols, variant, factor)
    return CAST_KERNEL, (cs, rs)


def onepass_act(
    x: torch.Tensor, rht_matrix: torch.Tensor, amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The cast, dispatched on ``x.numel()`` and this device's capability (:func:`pick_cast`)."""
    variant, factor = pick_cast(x.numel())
    return onepass_act_variant(x, rht_matrix, amax, variant, factor)


def live_producer_kernel(rows: int, cols: int, mode: int | None = None) -> tuple[str, tuple[int, int]]:
    """``(kernel-name substring, (grid_x, grid_y))`` :func:`gelu_tanh_amax` launches for a
    ``[rows, cols]`` input on this device -- the dispatch's own statement of what it does,
    for the by-name / by-grid census. ``mode`` defaults to the routed
    one; pass the other mode to show the two grids differ."""
    if mode is None:
        _, mode = pick_producer(rows * cols)
    cs, rs, _, _ = producer_grid(rows, cols, PROD_SMALL[0], mode)
    return PROD_KERNEL, (cs, rs)


def producer_counter(device: torch.device) -> torch.Tensor:
    """The producer's persistent completion counter (0 between launches)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return _prod_ctm().counter(idx)


def gelu_tanh_amax(x: torch.Tensor, rht_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The producer, dispatched on ``x.numel()`` (:func:`pick_producer`)."""
    variant, mode = pick_producer(x.numel())
    return gelu_tanh_amax_variant(x, rht_matrix, variant, mode)


def cast_grid(rows: int, cols: int, variant: str, factor: int) -> tuple[int, int, int, int]:
    """``(col_strips, row_splits, tiles_per_cta, resident_ctas)`` the cast host launches."""
    return tuple(_ctm().grid(rows, cols, variant, factor))


def producer_grid(rows: int, cols: int, variant: str, mode: int) -> tuple[int, int, int, int]:
    """``(col_strips, row_splits, tiles_per_cta, resident_ctas)`` the producer host launches."""
    return tuple(_prod_ctm().grid(rows, cols, variant, mode))


def launches(fn) -> list[tuple[str, tuple[int, int, int]]]:
    """``(kernel name, grid)`` of every kernel ``fn`` launches, in order.

    Verify the dispatch by kernel name per micro-batch. The
    producer's two modes share one kernel symbol and differ only in the grid,
    and so do the cast's tiers, so the grid is read from the profiler's trace
    as well.
    """
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile("r+", suffix=".json", delete=False) as f:
        path = f.name
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
    finally:
        os.unlink(path)
    kernels = [e for e in events if e.get("cat") == "kernel"]
    kernels.sort(key=lambda e: e["ts"])
    return [(e["name"], tuple(int(v) for v in e["args"]["grid"])) for e in kernels]


# ---------------------------------------------------------------------------
# torch.library registration
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define("gelu_tanh_amax(Tensor x, Tensor rht_matrix) -> (Tensor, Tensor)")
_lib.define(
    "onepass_act(Tensor x, Tensor rht_matrix, Tensor amax)"
    " -> (Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "gelu_tanh_amax", "CUDA")
def _gelu_tanh_amax_impl(x: torch.Tensor, rht_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return gelu_tanh_amax(x, rht_matrix)


@torch.library.impl(_lib, "onepass_act", "CUDA")
def _onepass_act_impl(
    x: torch.Tensor, rht_matrix: torch.Tensor, amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return onepass_act(x, rht_matrix, amax)
