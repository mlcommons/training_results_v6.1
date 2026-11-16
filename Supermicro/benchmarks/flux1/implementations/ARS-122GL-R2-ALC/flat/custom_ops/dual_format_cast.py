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

"""``torch.ops.flux`` boundary for the weight one-read dual-format cast.

Under R2 a tensor consumed by both an MXFP8 fprop GEMM and an NVFP4 backward
GEMM used to be read three times: the MXFP8 cast, TE's NVFP4 amax pass, and
the NVFP4 cast. Pass 1 of the two-read chain -- the MXFP8 rowwise cast with
the NVFP4 amax as its epilogue -- is ``flux.mxfp8_rowwise_cast_amax_ctm``
(``dual_format_cast_ctm.py``, CUTLASS DSL, both shipped architectures);
pass 2 is TE's unchanged columnwise-only cast through
``flux.te_nvfp4_quantize_with_amax`` (``dgelu_dbias_amax.py``).

=========================================  ======================================
op                                         what it does
=========================================  ======================================
``flux.dual_cast_one_read``                ``(x, rht_matrix, rht, amax) ->
                                           (mx_data, mx_scales, col_data,
                                           col_scales)`` -- the one-read
                                           cast: both consumed halves from one
                                           read, the NVFP4 amax supplied by a
                                           producer (the optimizer-path
                                           weight amax); see below
=========================================  ======================================

The one-read cast
-----------------
When a producer already knows the per-tensor NVFP4 amax before the cast --
today the weights, whose amax the optimizer path computes, all-gathers and
reduces before the first quantized weight cast
(``flat/comm_overlap.py``) -- pass 2 has nothing left to wait for and the two
reads collapse to one. ``flux.dual_cast_one_read`` reads the BF16 weight once
and emits the MXFP8 rowwise half and the NVFP4 columnwise half natively,
bitwise TE's (``tests/test_dual_cast_one_read_ctm.py`` at the four production
weight shapes; ``tests/test_weight_amax_ag.py`` through the production entry
point). It reads ``amax`` **once, in its prologue**, so the value must be
final on the current stream before the launch; and TE stores the amax, not
``S_enc``, so the wiring must attach the **same** tensor as the output's
``_amax_columnwise``.

The implementation is the CUTLASS DSL kernel ``dual_cast_one_read_ctm.py``,
on every architecture: op-level -3.88 / -3.81 ms/step
on two GB300 nodes and -2.71 / -2.66 on two Rubin nodes against the CUDA kernel
it replaced, production ``04x04x32`` -2.78 / -2.86 ms/step on two crossed node
sets and ``64x04x04`` -2.50 on one. The CUDA kernel
(``dual_cast_one_read_kernel.cu``) was deleted:
after the DSL was wired on both architectures it had **no live dispatch** -- its
``rht=True`` act arm never had a production caller (the act
role's one-read cast is ``flux.onepass_act``) -- so it was unreachable code,
not a fallback.
``rht=True`` is therefore an error here, never a fallback.

CUDA graphs
-----------
All outputs are ordinary allocations made inside the op, so capture takes them
from the graph pool. No host synchronisation.
"""

from __future__ import annotations

import torch
import torch.library

from flat.custom_ops import dual_cast_one_read_ctm as _ctm


def live_one_read_kernel(rows: int, cols: int) -> tuple[str, tuple[int, int]]:
    """``(kernel-name substring, (grid_x, grid_y))`` :func:`dual_cast_one_read` launches
    for a ``[rows, cols]`` weight on this device -- the dispatch's own statement of what
    it does, for the by-name / by-grid census."""
    return _ctm.live_kernel(rows, cols)


def dual_cast_one_read(
    x: torch.Tensor, rht_matrix: torch.Tensor, rht: bool, amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One read of a BF16 weight: MXFP8 rowwise half and NVFP4 columnwise half.

    Args:
        x: BF16 ``[rows, cols]`` contiguous, ``rows % 256 == 0``,
            ``cols % 128 == 0`` (an error otherwise; every production weight
            shape qualifies).
        rht_matrix: ``NVFP4Quantizer.rht_matrix``, a contiguous CUDA BF16
            ``[16, 16]``. Unread by the weight role; kept in the interface so
            the caller's quantizer is what it passes.
        rht: Must be ``False`` (the 2-D weight role). ``True`` -- the RHT act
            role -- has no implementation here: the act role's one-read cast is
            ``flux.onepass_act``.
        amax: FP32, ``numel() == 1``, CUDA: the **final** per-tensor NVFP4
            amax (``max|x|``), complete on the current stream before this
            call. Read once in the kernel prologue; a partial value is
            silently wrong.

    Returns:
        ``(mx_data, mx_scales, col_data, col_scales)`` -- E4M3 ``[rows,
        cols]``, GEMM-swizzled E8M0 ``[rows, cols/32]``, E2M1 ``[cols,
        rows/2]``, GEMM-swizzled E4M3 ``[cols, rows/16]``, all uint8; bitwise
        what ``tex.quantize`` emits for a rowwise-only MXFP8 quantizer and a
        columnwise-only NVFP4 quantizer given this amax.

    Raises:
        RuntimeError: On ``rht=True`` or any interface violation (the error
            type ``TORCH_CHECK`` raised, which ``tests/test_weight_amax_ag.py``
            pins).
    """
    if rht:
        raise RuntimeError(
            "dual_cast_one_read: the act role (rht=True) has no implementation; "
            "the act role's one-read cast is flux.onepass_act")
    if not (rht_matrix.is_cuda and rht_matrix.dtype == torch.bfloat16
            and rht_matrix.is_contiguous() and tuple(rht_matrix.shape) == (16, 16)):
        raise RuntimeError(
            "dual_cast_one_read: rht_matrix must be a contiguous CUDA BF16 [16, 16]")
    try:
        return _ctm.dual_cast_one_read_ctm_auto(x, amax)
    except ValueError as e:
        # a shape/dtype guard is an error, never a fallback
        raise RuntimeError(f"dual_cast_one_read: {e}") from e


# ---------------------------------------------------------------------------
# torch.library registration
# ---------------------------------------------------------------------------

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

_lib.define(
    "dual_cast_one_read(Tensor x, Tensor rht_matrix, bool rht, Tensor amax)"
    " -> (Tensor, Tensor, Tensor, Tensor)"
)


@torch.library.impl(_lib, "dual_cast_one_read", "CUDA")
def _dual_cast_one_read_impl(
    x: torch.Tensor, rht_matrix: torch.Tensor, rht: bool, amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return dual_cast_one_read(x, rht_matrix, rht, amax)
