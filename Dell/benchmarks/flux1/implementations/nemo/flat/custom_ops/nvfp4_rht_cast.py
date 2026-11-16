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

"""The amax-supplied NVFP4 cast, size- and arch-tiered.

Registers ``torch.ops.flux.nvfp4_rht_cast_ctm`` -- the CUTLASS-DSL replacement for
``flux.te_nvfp4_quantize_with_amax`` (TE's ``row_col_rht_gemm_device``) -- and
:func:`cast_with_amax`, the ONE production entry every amax-supplied cast site in
``flat/nvfp4_ops.py`` calls. Dispatch: :data:`TIERS` is a
per-(architecture, role) table of ``(rows upto, variant, grid factor)`` on ``rows``
(a size expression; shapes are static per graph capture, so the branch is trace-time),
filled from measurement. An
architecture with no table keeps TE's kernel; ``FLAT_NVFP4_RHT_CAST_CTM=0/1`` forces
TE / the DSL kernel, and neither value falls back silently.

RNG: the DSL op owns its Philox ``(seed, offset)`` per device
(``nvfp4_rht_cast_ctm.rng_state``), seeded once from the torch CUDA generator and
advanced by 1024 per stochastic cast in-kernel -- TE's contract, without
``extract_seed_and_offset``.
"""

from __future__ import annotations

import os

import torch
import torch.library

from flat.custom_ops import nvfp4_rht_cast_ctm as _ctm
from flat.custom_ops import nvfp4_rht_cast_tc as _tc

KERNEL = _ctm.KERNEL

#: (compute capability, role) -> ((rows upto | None, cols upto | None, variant | None, factor), ...);
#: the first entry with rows <= rows_upto and cols <= cols_upto decides; variant None keeps TE's
#: kernel. Written from cold single-op
#: measurements (quantum 2.05 us): the DSL arm is named only where it beat TE by more than the
#: quantum; a shape inside the quantum or a loss keeps TE.
TIERS: dict[tuple[tuple[int, int], str], tuple[tuple[int | None, int | None, str | None, int], ...]] = {
    # GB300 -- IN SITU (4-GPU production replays, one arm at every
    # shape, kernel time vs TE): the tensor-core arm TB2 (nvfp4_rht_cast_tc.py: tcgen05 RHT, two column
    # warpgroups, band-major tile order) beats TE at EVERY shape group at every micro-batch -- mb 4 -1.76 ms
    # of 8.57, mb 8 -2.07 of 11.85, mb 16 -1.94 of 17.59, mb 32 -3.91 of 30.79 -- so one arm serves every
    # size here. The mma.sync arms are kept (N x1 ties TB2 at rows <= 2048: -1.67 vs -1.76 at mb 4) but are
    # not dispatched; the cold sweep's wider table was never the shipping rule (in situ TE runs
    # 1.4-3x faster than cold at N = 3072 and the mma.sync arms tie or lose at every M >= 4096).
    ((10, 3), "act"): ((None, None, "TB2", 1),),
    ((10, 3), "grad"): ((None, None, "TB2", 1),),
    # Rubin (informational, cold measurements only -- no in-situ instrument): T2 wins at every
    # shape cold (-2.7 .. -14.4 us), TB2 only at [16384, 9216|12288]; N x1 at rows <= 2048 is within a tick
    # of T2 there and is the arm the in-situ GB300 data supports at those rows.
    ((10, 7), "act"): ((2048, None, "N", 1), (None, None, "T2", 1)),
    ((10, 7), "grad"): ((2048, None, "N", 1), (16384, 3072, "T2", 1), (8192, None, "T2", 1), (None, None, "TB2", 1)),
}


def _override() -> bool | None:
    v = os.environ.get("FLAT_NVFP4_RHT_CAST_CTM")
    return None if v is None else v == "1"


def _forced_arm() -> tuple[str, int] | None:
    """``FLAT_NVFP4_RHT_CAST_ARM=<variant>,<factor>``: one arm at every eligible shape, both roles
    (the in-situ measuring instrument; never set in production)."""
    v = os.environ.get("FLAT_NVFP4_RHT_CAST_ARM")
    if not v:
        return None
    variant, factor = v.split(",")
    if variant not in _ctm.VARIANTS and variant not in _tc.VARIANTS:
        raise ValueError(f"FLAT_NVFP4_RHT_CAST_ARM: unknown variant {variant!r}")
    return variant, int(factor)


def _window(variant: str) -> int:
    """Rows must be a multiple of this for the variant (the flush window; 1024 for the tensor-core arm)."""
    return 1024 if variant in _tc.VARIANTS else _ctm.VARIANTS[variant][3]


def role_of(quantizer) -> str | None:
    """``"grad"`` (both halves, SR) or ``"act"`` (columnwise only, RN) for a TE ``NVFP4Quantizer``
    this kernel implements; ``None`` for any other quantizer (the 2-D weight role of the eager
    weight route, a rowwise-only or an SR-off both-halves quantizer), which keeps TE's cast."""
    if not getattr(quantizer, "with_rht", False) or not getattr(quantizer, "with_post_rht_amax", False):
        return None
    if getattr(quantizer, "with_2d_quantization", False):
        return None
    if quantizer.rowwise_usage and quantizer.columnwise_usage and quantizer.stochastic_rounding:
        return "grad"
    if not quantizer.rowwise_usage and quantizer.columnwise_usage and not quantizer.stochastic_rounding:
        return "act"
    return None


def select(rows: int, cols: int, role: str, device_index: int | None = None) -> tuple[str, int] | None:
    """The ``(variant, factor)`` the dispatch names for this shape, or ``None`` for TE."""
    dev = torch.cuda.current_device() if device_index is None else device_index
    cc = torch.cuda.get_device_capability(dev)
    tiers = TIERS.get((cc, role))
    forced = _override()
    if forced is False:
        return None
    arm = _forced_arm()
    if arm is not None:
        if rows % 1024 or cols % 128 or rows % _window(arm[0]):
            raise ValueError(f"nvfp4_rht_cast: [{rows},{cols}] is not a shape variant {arm[0]} takes")
        return arm
    if tiers is None:
        if not forced:
            return None
        tiers = ((None, None, "A", 1),)
    if rows % 1024 or cols % 128:
        if forced:
            raise ValueError(f"nvfp4_rht_cast: [{rows},{cols}] is not a shape this kernel takes")
        return None
    for rows_upto, cols_upto, variant, factor in tiers:
        if (rows_upto is None or rows <= rows_upto) and (cols_upto is None or cols <= cols_upto):
            if variant is None:
                return ((("A", 1)) if forced else None)
            if rows % _window(variant) == 0:
                return variant, factor
            return None
    return None


def live_kernel(rows: int, cols: int, role: str) -> tuple[str, tuple[int, int, int]] | None:
    """Kernel symbol substring and grid the dispatch launches (for the by-name / by-grid census)."""
    arm = select(rows, cols, role)
    if arm is None:
        return None
    if arm[0] in _tc.VARIANTS:
        strips, splits, _ = _tc.grid(rows, cols, arm[0], arm[1])
        return _tc.KERNEL, (strips, splits, 1)
    strips, splits, _, _ = _ctm.grid(rows, cols, arm[0], arm[1])
    return KERNEL, (strips, splits, 1)


def nvfp4_rht_cast_ctm(
    x: torch.Tensor, rht_matrix: torch.Tensor, amax_row: torch.Tensor, amax_col: torch.Tensor,
    rowwise: bool, variant: str, factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The op body: grad role when ``rowwise`` (both halves, SR), act role otherwise; the
    ``mma.sync`` body for ``nvfp4_rht_cast_ctm.VARIANTS``, the tensor-core body for ``_tc.VARIANTS``."""
    impl = _tc.nvfp4_rht_cast_tc if variant in _tc.VARIANTS else _ctm.nvfp4_rht_cast_ctm
    rd, rs, cd, cs = impl(
        x, rht_matrix, amax_row, amax_col, variant, factor, rowwise=rowwise, sr=rowwise)
    if rd is None:
        rd = torch.empty(0, dtype=torch.uint8, device=x.device)
        rs = torch.empty(0, dtype=torch.uint8, device=x.device)
    return rd, rs, cd, cs


try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except RuntimeError:
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "nvfp4_rht_cast_ctm(Tensor x, Tensor rht_matrix, Tensor amax_row, Tensor amax_col, "
    "bool rowwise, str variant, int factor) -> (Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "nvfp4_rht_cast_ctm", "CUDA")
def _nvfp4_rht_cast_ctm_cuda(x, rht_matrix, amax_row, amax_col, rowwise, variant, factor):
    return nvfp4_rht_cast_ctm(x, rht_matrix, amax_row, amax_col, rowwise, variant, factor)


def cast_with_amax(x: torch.Tensor, quantizer, amax_row: torch.Tensor, amax_col: torch.Tensor):
    """The production entry: ``x`` cast by ``quantizer`` with the amax pair supplied,
    on the DSL kernel where the tier table names one, else TE's. Returns an ``NVFP4Tensor``."""
    from flat import nvfp4_ops as nv

    role = role_of(quantizer)
    rows, cols = x.shape
    arm = None if role is None else select(rows, cols, role, x.device.index)
    if arm is None:
        parts = torch.ops.flux.te_nvfp4_quantize_with_amax(
            x, nv._te_ops.quantizer_handle(quantizer), amax_row, amax_col)
        return nv._rebuild_nvfp4(parts, x, quantizer)
    rd, rs, cd, cs = torch.ops.flux.nvfp4_rht_cast_ctm(
        x, quantizer.rht_matrix, amax_row, amax_col, role == "grad", arm[0], arm[1])
    if role == "grad":
        parts = (rd, rs, cd, cs, amax_row, amax_col)
    else:
        parts = (None, None, cd, cs, None, amax_col)
    return nv._rebuild_nvfp4(parts, x, quantizer)
