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

"""``torch.ops`` wrappers for the TransformerEngine kernels the NVFP4 path uses."""

from __future__ import annotations

from typing import Any

import torch
import torch.library

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (
    MXFP8TensorStorage,
)
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (
    NVFP4TensorStorage,
)

_OUT_DTYPE = torch.bfloat16


# Quantizer registry: device-resident quantizers cross the op boundary by index. The list
# holds a strong reference, so an id() key cannot be recycled by a later quantizer.
_quantizers: list[Any] = []
_handle_of: dict[int, int] = {}


def quantizer_handle(quantizer: Any) -> int:
    """Return the op-boundary handle for a quantizer, registering it if new.

    Handles index the module registry and are stable for the life of the process.
    """
    handle = _handle_of.get(id(quantizer))
    if handle is None:
        handle = len(_quantizers)
        _quantizers.append(quantizer)
        _handle_of[id(quantizer)] = handle
    return handle


def _resolve(handle: int) -> Any:
    """Return the registered quantizer for ``handle``."""
    return _quantizers[handle]


# Host wrappers

def _quantize_to_parts(t: torch.Tensor, handle: int) -> tuple:
    """``tex.quantize``, returning the quantized tensor's constituent tensors."""
    quantizer = _resolve(handle)
    assert quantizer.optimize_for_gemm, (
        "te_*_quantize requires optimize_for_gemm: the GEMM ops rebuild the "
        "operand with with_gemm_swizzled_scales=True, which only holds when "
        "tex.quantize was asked to bake the GEMM scale layout in "
        "(TE cast.cpp:83)."
    )
    # A one-sided quantizer is legitimate: which half a GEMM reads is layout-dependent.
    assert quantizer.rowwise_usage or quantizer.columnwise_usage, (
        "te_*_quantize needs a quantizer with at least one usage flag set; "
        "this one would emit no data at all, which TE's "
        "NVTETensorFromNVFP4Tensor rejects."
    )
    return tex.quantize(t, quantizer)


def te_nvfp4_quantize(
    t: torch.Tensor, quantizer_handle_: int,
) -> tuple[torch.Tensor | None, ...]:
    """Quantize to NVFP4, returning plain tensors.

    Returns ``(rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv,
    amax_rowwise, amax_columnwise)``; the rowwise entries are None for a columnwise-only
    quantizer.
    """
    q = _quantize_to_parts(t, quantizer_handle_)
    assert q._rowwise_data is not None or not _resolve(
        quantizer_handle_
    ).stochastic_rounding, (
        "refusing to emit a columnwise-only NVFP4 tensor from a "
        "stochastic-rounding quantizer: SR marks the gradient role, and both "
        "of a gradient's halves are live (dgrad NN reads rowwise, wgrad NT "
        "reads columnwise)."
    )
    return (
        q._rowwise_data,
        q._rowwise_scale_inv,
        q._columnwise_data,
        q._columnwise_scale_inv,
        q._amax_rowwise,
        q._amax_columnwise,
    )


def te_mxfp8_quantize(
    t: torch.Tensor, quantizer_handle_: int,
) -> tuple[torch.Tensor, ...]:
    """Quantize to MXFP8, returning plain tensors.

    ``t`` is [M, K] contiguous, both dims divisible by 32. Returns ``(rowwise_data,
    rowwise_scale_inv, columnwise_data, columnwise_scale_inv)``; the columnwise pair is
    None for a rowwise-only quantizer.
    """
    q = _quantize_to_parts(t, quantizer_handle_)
    return (
        q._rowwise_data,
        q._rowwise_scale_inv,
        q._columnwise_data,
        q._columnwise_scale_inv,
    )


#: Which half of each NVFP4 operand cuBLASLt reads, per layout: (A, B), True = rowwise.
#: Transcribed from TE's CanonicalizeGemmInput.
_NVFP4_READS_ROWWISE: dict[str, tuple[bool, bool]] = {
    "TN": (True, True),    # fprop: weight rowwise, act rowwise
    "NN": (False, True),   # dgrad: weight columnwise, grad rowwise
    "NT": (False, False),  # wgrad: act columnwise, grad columnwise
}

#: The same for MXFP8 operands, transcribed from the other branch of CanonicalizeGemmInput.
#: Kept separate from _NVFP4_READS_ROWWISE on purpose; a test asserts the two agree.
_MXFP8_READS_ROWWISE: dict[str, tuple[bool, bool]] = {
    "TN": (True, True),    # fprop: A rowwise, B rowwise
    "NN": (False, True),   # dgrad: A columnwise, B rowwise
    "NT": (False, False),  # wgrad: A columnwise, B columnwise
}


def te_nvfp4_gemm(
    a_rowwise_data: torch.Tensor | None,
    a_rowwise_scale_inv: torch.Tensor | None,
    a_columnwise_data: torch.Tensor,
    a_columnwise_scale_inv: torch.Tensor,
    a_amax_rowwise: torch.Tensor | None,
    a_amax_columnwise: torch.Tensor,
    b_rowwise_data: torch.Tensor | None,
    b_rowwise_scale_inv: torch.Tensor | None,
    b_columnwise_data: torch.Tensor,
    b_columnwise_scale_inv: torch.Tensor,
    b_amax_rowwise: torch.Tensor | None,
    b_amax_columnwise: torch.Tensor,
    layout: str,
    nvfp4_e4m3_max: int,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``general_gemm`` on two NVFP4 operands supplied as plain tensors.

    An operand's rowwise half may be None only when this layout does not read it
    (_NVFP4_READS_ROWWISE); that is checked here.
    """
    try:
        a_reads_rowwise, b_reads_rowwise = _NVFP4_READS_ROWWISE[layout]
    except KeyError:
        raise ValueError(
            f"layout {layout!r} is not one of "
            f"{sorted(_NVFP4_READS_ROWWISE)}"
        ) from None
    for side, reads_rowwise, rowwise, columnwise in (
        ("A", a_reads_rowwise, a_rowwise_data, a_columnwise_data),
        ("B", b_reads_rowwise, b_rowwise_data, b_columnwise_data),
    ):
        half = "rowwise" if reads_rowwise else "columnwise"
        if (rowwise if reads_rowwise else columnwise) is None:
            raise ValueError(
                f"te_nvfp4_gemm layout {layout} reads operand {side}'s "
                f"{half} half, but it was quantized without one. A tensor may "
                f"drop a half only when no GEMM reads it -- gradients need "
                f"both (dgrad NN takes rowwise, wgrad NT columnwise)."
            )
    a = NVFP4TensorStorage(
        a_rowwise_data, a_rowwise_scale_inv,
        a_columnwise_data, a_columnwise_scale_inv,
        a_amax_rowwise, a_amax_columnwise,
        tex.DType.kFloat4E2M1, None, True,
        fake_dtype=_OUT_DTYPE, nvfp4_e4m3_max=nvfp4_e4m3_max,
    )
    b = NVFP4TensorStorage(
        b_rowwise_data, b_rowwise_scale_inv,
        b_columnwise_data, b_columnwise_scale_inv,
        b_amax_rowwise, b_amax_columnwise,
        tex.DType.kFloat4E2M1, None, True,
        fake_dtype=_OUT_DTYPE, nvfp4_e4m3_max=nvfp4_e4m3_max,
    )
    result, _, _, _ = general_gemm(
        a, b, out_dtype=_OUT_DTYPE, layout=layout, bias=bias, out=out,
    )
    return result


def _check_mxfp8_operand_halves(
    layout: str,
    a_rowwise_data: torch.Tensor | None,
    a_columnwise_data: torch.Tensor | None,
    b_rowwise_data: torch.Tensor | None,
    b_columnwise_data: torch.Tensor | None,
) -> None:
    """Raise unless both operands carry the half this layout will read."""
    try:
        a_needs_rowwise, b_needs_rowwise = _MXFP8_READS_ROWWISE[layout]
    except KeyError:
        raise ValueError(
            f"layout {layout!r} is not one of {sorted(_MXFP8_READS_ROWWISE)}"
        ) from None
    b_needs_columnwise = not b_needs_rowwise
    missing = []
    if a_needs_rowwise and a_rowwise_data is None:
        missing.append("A rowwise")
    if not a_needs_rowwise and a_columnwise_data is None:
        missing.append("A columnwise")
    if b_needs_columnwise and b_columnwise_data is None:
        missing.append("B columnwise")
    if not b_needs_columnwise and b_rowwise_data is None:
        missing.append("B rowwise")
    if missing:
        raise ValueError(
            f"te_mxfp8_gemm layout {layout} reads {', '.join(missing)}, which "
            f"the caller did not supply. The operand was quantized with that "
            f"half switched off; a one-sided cast is only valid when no GEMM "
            f"reads the missing half. Check the quantizer choice in "
            f"flat/nvfp4_ops.py against this GEMM class's backward format "
            f"(FLAT_NVFP4_FALLBACK demotes a class to MXFP8, and a demoted "
            f"class's dgrad and wgrad do read the columnwise halves)."
        )


def te_mxfp8_gemm(
    a_rowwise_data: torch.Tensor | None,
    a_rowwise_scale_inv: torch.Tensor | None,
    a_columnwise_data: torch.Tensor | None,
    a_columnwise_scale_inv: torch.Tensor | None,
    b_rowwise_data: torch.Tensor | None,
    b_rowwise_scale_inv: torch.Tensor | None,
    b_columnwise_data: torch.Tensor | None,
    b_columnwise_scale_inv: torch.Tensor | None,
    layout: str,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``general_gemm`` on two MXFP8 operands supplied as plain tensors.

    Any operand half may be None when this layout does not read it. ``layout`` is "TN",
    "NN" or "NT"; ``out``, if given, is written in place.
    """
    _check_mxfp8_operand_halves(
        layout, a_rowwise_data, a_columnwise_data,
        b_rowwise_data, b_columnwise_data,
    )
    a = MXFP8TensorStorage(
        a_rowwise_data, a_rowwise_scale_inv,
        a_columnwise_data, a_columnwise_scale_inv,
        tex.DType.kFloat8E4M3, None, True,
        fake_dtype=_OUT_DTYPE,
    )
    b = MXFP8TensorStorage(
        b_rowwise_data, b_rowwise_scale_inv,
        b_columnwise_data, b_columnwise_scale_inv,
        tex.DType.kFloat8E4M3, None, True,
        fake_dtype=_OUT_DTYPE,
    )
    result, _, _, _ = general_gemm(
        a, b, out_dtype=_OUT_DTYPE, layout=layout, bias=bias, out=out,
    )
    return result


def te_gelu(x: torch.Tensor) -> torch.Tensor:
    """``tex.gelu`` with no quantizer: tanh-GELU, BF16 in, BF16 out."""
    return tex.gelu(x, None)


# torch.library registration

try:
    _lib = torch.library.Library("flux", "FRAGMENT")
except Exception:  # pragma: no cover - only when flux is not yet defined
    _lib = torch.library.Library("flux", "DEF")

# The rowwise triple is optional (columnwise-only quantizers); the columnwise one is not.
_lib.define(
    "te_nvfp4_quantize(Tensor t, int quantizer_handle)"
    " -> (Tensor?, Tensor?, Tensor, Tensor, Tensor?, Tensor)"
)
_lib.define(
    # The columnwise pair is optional (rowwise-only quantizers).
    "te_mxfp8_quantize(Tensor t, int quantizer_handle)"
    " -> (Tensor, Tensor, Tensor?, Tensor?)"
)
_lib.define(
    "te_nvfp4_gemm("
    "Tensor? a_rowwise_data, Tensor? a_rowwise_scale_inv,"
    " Tensor a_columnwise_data, Tensor a_columnwise_scale_inv,"
    " Tensor? a_amax_rowwise, Tensor a_amax_columnwise,"
    " Tensor? b_rowwise_data, Tensor? b_rowwise_scale_inv,"
    " Tensor b_columnwise_data, Tensor b_columnwise_scale_inv,"
    " Tensor? b_amax_rowwise, Tensor b_amax_columnwise,"
    " str layout, int nvfp4_e4m3_max,"
    " Tensor? bias=None, Tensor(a!)? out=None) -> Tensor"
)
_lib.define(
    # Every half is optional; _check_mxfp8_operand_halves enforces what the layout reads.
    "te_mxfp8_gemm("
    "Tensor? a_rowwise_data, Tensor? a_rowwise_scale_inv,"
    " Tensor? a_columnwise_data, Tensor? a_columnwise_scale_inv,"
    " Tensor? b_rowwise_data, Tensor? b_rowwise_scale_inv,"
    " Tensor? b_columnwise_data, Tensor? b_columnwise_scale_inv,"
    " str layout,"
    " Tensor? bias=None, Tensor(a!)? out=None) -> Tensor"
)
_lib.define("te_gelu(Tensor x) -> Tensor")


@torch.library.impl(_lib, "te_nvfp4_quantize", "CUDA")
def _te_nvfp4_quantize_impl(
    t: torch.Tensor, quantizer_handle: int,
) -> tuple[torch.Tensor | None, ...]:
    return te_nvfp4_quantize(t, quantizer_handle)


@torch.library.impl(_lib, "te_mxfp8_quantize", "CUDA")
def _te_mxfp8_quantize_impl(
    t: torch.Tensor, quantizer_handle: int,
) -> tuple[torch.Tensor, ...]:
    return te_mxfp8_quantize(t, quantizer_handle)


@torch.library.impl(_lib, "te_nvfp4_gemm", "CUDA")
def _te_nvfp4_gemm_impl(
    a_rowwise_data: torch.Tensor | None,
    a_rowwise_scale_inv: torch.Tensor | None,
    a_columnwise_data: torch.Tensor,
    a_columnwise_scale_inv: torch.Tensor,
    a_amax_rowwise: torch.Tensor | None,
    a_amax_columnwise: torch.Tensor,
    b_rowwise_data: torch.Tensor | None,
    b_rowwise_scale_inv: torch.Tensor | None,
    b_columnwise_data: torch.Tensor,
    b_columnwise_scale_inv: torch.Tensor,
    b_amax_rowwise: torch.Tensor | None,
    b_amax_columnwise: torch.Tensor,
    layout: str,
    nvfp4_e4m3_max: int,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return te_nvfp4_gemm(
        a_rowwise_data, a_rowwise_scale_inv,
        a_columnwise_data, a_columnwise_scale_inv,
        a_amax_rowwise, a_amax_columnwise,
        b_rowwise_data, b_rowwise_scale_inv,
        b_columnwise_data, b_columnwise_scale_inv,
        b_amax_rowwise, b_amax_columnwise,
        layout, nvfp4_e4m3_max, bias, out,
    )


@torch.library.impl(_lib, "te_mxfp8_gemm", "CUDA")
def _te_mxfp8_gemm_impl(
    a_rowwise_data: torch.Tensor | None,
    a_rowwise_scale_inv: torch.Tensor | None,
    a_columnwise_data: torch.Tensor | None,
    a_columnwise_scale_inv: torch.Tensor | None,
    b_rowwise_data: torch.Tensor | None,
    b_rowwise_scale_inv: torch.Tensor | None,
    b_columnwise_data: torch.Tensor | None,
    b_columnwise_scale_inv: torch.Tensor | None,
    layout: str,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return te_mxfp8_gemm(
        a_rowwise_data, a_rowwise_scale_inv,
        a_columnwise_data, a_columnwise_scale_inv,
        b_rowwise_data, b_rowwise_scale_inv,
        b_columnwise_data, b_columnwise_scale_inv,
        layout, bias, out,
    )


@torch.library.impl(_lib, "te_gelu", "CUDA")
def _te_gelu_impl(x: torch.Tensor) -> torch.Tensor:
    return te_gelu(x)
