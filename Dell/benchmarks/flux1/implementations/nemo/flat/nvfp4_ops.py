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

"""NVFP4 linear operations wrapping TransformerEngine CUDA kernels.

Drop-in replacement for flat/fp8_ops.py: same public names, signatures and return shapes.
"""

from __future__ import annotations

import importlib
import functools as _functools
import os as _os
from typing import Any, Iterable

import torch
from torch import Tensor
import torch.nn as nn

import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import (
    MXFP8Quantizer,
    MXFP8Tensor,
)
from transformer_engine.pytorch.tensor.nvfp4_tensor import (
    NVFP4Quantizer,
    NVFP4Tensor,
)

# Registers torch.ops.flux.te_{nvfp4,mxfp8}_{quantize,gemm} and te_gelu.
from flat.custom_ops import te_quantized_gemm as _te_ops

# Registers torch.ops.flux.dgelu_dbias_amax and te_nvfp4_quantize_with_amax.
from flat.custom_ops import dgelu_dbias_amax as _dgelu_ops  # noqa: F401

# Registers torch.ops.flux.dual_cast_one_read.
from flat.custom_ops import dual_format_cast as _dual_ops  # noqa: F401

# Registers the tanh-GELU CUTLASS-DSL kernel variants.
from flat.custom_ops import gelu_tanh_ctm as _gelu_tanh_ctm_ops  # noqa: F401
from flat.custom_ops import gelu_tanh_ctm_v2 as _gelu_tanh_ctm_v2_ops  # noqa: F401
from flat.custom_ops import (  # noqa: F401
    fused_qkv_split_norm_rope_ctm2 as _qkv_rope_ctm_ops,
)
from flat.custom_ops import rope_norm_join_ctm as _rope_join_ctm_ops  # noqa: F401
from flat.custom_ops import fused_sum_cat_ctm2 as _sum_cat_ctm_ops  # noqa: F401
from flat.custom_ops import dual_format_cast_ctm as _dual_cast_ctm_ops  # noqa: F401

from flat.weight_cache import WeightCache

# Registers torch.ops.flux.gate_bwd_dbias_amax_deferred_ctm.
from flat.custom_ops import gate_bwd_amax_ctm as _gate_bwd_ctm_ops

# Registers torch.ops.flux.tma_bgrad_amax_ctm_s.
from flat.custom_ops import tma_bgrad_amax_ctm_s as _tma_bgrad_s_ops

# Registers torch.ops.flux.gelu_tanh_amax and torch.ops.flux.onepass_act.
from flat.custom_ops import onepass_act as _onepass_ops  # noqa: F401

# Registers torch.ops.flux.rope_norm_join_bgrad_amax_ctm.
from flat.custom_ops import rope_norm_join_bgrad_amax_ctm as _rope_bgrad_ops

# The NVFP4 RHT cast. cast_with_amax replaces the TE
# te_nvfp4_quantize_with_amax + _rebuild_nvfp4 pair at every amax-supplied site.
from flat.custom_ops import nvfp4_rht_cast as _rht_cast_ops

FP8 = "fp8"
NVFP4 = "nvfp4"

_DISABLE_SR = _os.environ.get("FLAT_NVFP4_DISABLE_SR", "0") == "1"
_DISABLE_2D = _os.environ.get("FLAT_NVFP4_DISABLE_2D", "0") == "1"

# NVFP4 analogue of FLAT_BF16: the dispatch below still runs, every GEMM runs in BF16.
_FLAT_NVFP4_BF16 = _os.environ.get("FLAT_NVFP4_BF16", "0") == "1"


# Role quantizers

# RHT is on for the activation and gradient roles and off for weights;
# with_post_rht_amax follows with_rht.
_ROLE_FLAGS: dict[str, dict[str, bool]] = {
    "act": {
        "with_rht": True,
        "with_post_rht_amax": True,
        "with_2d_quantization": False,
        "stochastic_rounding": False,
    },
    "weight": {
        "with_rht": False,
        "with_post_rht_amax": False,
        "with_2d_quantization": not _DISABLE_2D,
        "stochastic_rounding": False,
    },
    "grad": {
        "with_rht": True,
        "with_post_rht_amax": True,
        "with_2d_quantization": False,
        "stochastic_rounding": not _DISABLE_SR,
    },
}

# (role, columnwise_only, device index) -> NVFP4Quantizer. Built lazily: the
# constructor allocates tensors on the current device, so no module-scope instances.
_role_quantizers: dict[tuple[str, bool, int], NVFP4Quantizer] = {}


def role_quantizer(
    role: str, *, columnwise_only: bool = False,
) -> NVFP4Quantizer:
    """Return this device's quantizer for one tensor role.

    columnwise_only builds it with rowwise=False, so the cast emits only the columnwise half.
    """
    if columnwise_only and role == "grad":
        raise ValueError(
            "the gradient role cannot be columnwise-only: dgrad reads the "
            "gradient's rowwise half (layout NN) and wgrad its columnwise "
            "half (layout NT), so both are live. Only activation and weight "
            "casts have a dead rowwise half, and only while the fprop GEMM "
            "runs in another format."
        )
    key = (role, columnwise_only, torch.cuda.current_device())
    quantizer = _role_quantizers.get(key)
    if quantizer is None:
        quantizer = NVFP4Quantizer(
            rowwise=not columnwise_only, columnwise=True, **_ROLE_FLAGS[role],
        )
        quantizer.optimize_for_gemm = True
        _role_quantizers[key] = quantizer
    return quantizer


# fp8_ops imports this module from its tail, so it is imported back lazily.
_mx_module: Any = None


def _mx() -> Any:
    """Return the flat.fp8_ops module, importing it on first use."""
    global _mx_module
    if _mx_module is None:
        _mx_module = importlib.import_module("flat.fp8_ops")
    return _mx_module


# MXFP8 quantizer for casts whose columnwise half nothing reads. Safe at module
# scope: MXFP8Quantizer allocates no device tensors.
_MX_FPROP_QUANTIZER = MXFP8Quantizer(
    fp8_dtype=tex.DType.kFloat8E4M3,
    rowwise=True,
    columnwise=False,
)
_MX_FPROP_QUANTIZER.optimize_for_gemm = True


def _mx_quantizer(columnwise: bool) -> object:
    """Return the MXFP8 quantizer emitting the columnwise half or not.

    Only backward GEMMs read columnwise halves: dgrad (NN) the weight's, wgrad (NT)
    the activation's and gradient's. Fprop (TN) reads rowwise only.
    """
    if columnwise:
        return _mx()._fwd_quantizer
    return _MX_FPROP_QUANTIZER


# GEMM classes and precision routing

GEMM_CLASSES: tuple[str, ...] = (
    "dsb.img.qkv", "dsb.txt.qkv",
    "dsb.img.proj", "dsb.txt.proj",
    "dsb.img.fc1", "dsb.txt.fc1",
    "dsb.img.fc2", "dsb.txt.fc2",
    "ssb.qkv", "ssb.fc1", "ssb.proj", "ssb.fc2",
)

_DSB_ATTRS = {
    ("img_attn", "qkv"): "dsb.img.qkv",
    ("txt_attn", "qkv"): "dsb.txt.qkv",
    ("img_attn", "proj"): "dsb.img.proj",
    ("txt_attn", "proj"): "dsb.txt.proj",
}
_DSB_MLP_ATTRS = {
    "_img_mlp_0": "dsb.img.fc1",
    "_txt_mlp_0": "dsb.txt.fc1",
    "_img_mlp_2": "dsb.img.fc2",
    "_txt_mlp_2": "dsb.txt.fc2",
}
_SSB_ATTRS = {
    "linear1_qkv": "ssb.qkv",
    "linear1_mlp": "ssb.fc1",
    "linear2_attn": "ssb.proj",
    "linear2_mlp": "ssb.fc2",
}

# weight -> GEMM class token, populated by register_gemm_classes().
# Keyed on tensor identity, not address.
_class_of_weight = WeightCache()


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Whether needle's components appear in order within haystack."""
    it = iter(haystack)
    return all(part in it for part in needle)


def _expand_class_token(token: str) -> set[str]:
    """Expand one FLAT_NVFP4_FALLBACK token to concrete GEMM classes.

    Accepts an exact class name, a prefix.* wildcard, or a dot-separated ordered
    subsequence of the class name (dsb.qkv, fc1). Raises ValueError on no match.
    """
    if token.endswith(".*"):
        prefix = token[:-1]
        matched = {c for c in GEMM_CLASSES if c.startswith(prefix)}
    elif token in GEMM_CLASSES:
        matched = {token}
    else:
        parts = token.split(".")
        matched = {
            c for c in GEMM_CLASSES
            if _is_subsequence(parts, c.split("."))
        }
    if not matched:
        raise ValueError(
            f"FLAT_NVFP4_FALLBACK: unknown GEMM class {token!r}. "
            f"Known classes: {', '.join(GEMM_CLASSES)}"
        )
    return matched


def _parse_fallback(spec: str) -> frozenset[str]:
    """Parse FLAT_NVFP4_FALLBACK into a set of demoted GEMM classes."""
    demoted: set[str] = set()
    for token in spec.split(","):
        token = token.strip()
        if token:
            demoted |= _expand_class_token(token)
    return frozenset(demoted)


def _flag(name: str, default: bool) -> bool:
    """Read a boolean recipe flag: 1/0, true/false, yes/no."""
    value = _os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={value!r} is not a boolean")


def _formats_from_flags() -> tuple[str, str]:
    """Resolve (forward, backward) GEMM formats from the FP8, FP4 and FP4_SCOPE flags.

    Each is 'fp8', 'nvfp4' or 'bfloat16'.
    """
    fp8 = _flag("FP8", True)
    fp4 = _flag("FP4", True)
    scope = (_os.environ.get("FP4_SCOPE", "").strip().lower() or "grad")
    if scope not in ("grad", "all"):
        raise ValueError(f"FP4_SCOPE={scope!r} is not 'grad' or 'all'")
    base = FP8 if fp8 else "bfloat16"
    if not fp4:
        return base, base
    return (NVFP4 if scope == "all" else base), NVFP4


_FWD_FORMAT, _BWD_FORMAT = _formats_from_flags()
_DEMOTED = _parse_fallback(_os.environ.get("FLAT_NVFP4_FALLBACK", ""))


def nvfp4_enabled() -> bool:
    """Whether any GEMM in this configuration runs in NVFP4."""
    return NVFP4 in (_FWD_FORMAT, _BWD_FORMAT)


def register_gemm_classes(model: nn.Module) -> None:
    """Map every quantized weight tensor to its GEMM class token.

    Only needed when FLAT_NVFP4_FALLBACK is set.
    """
    _class_of_weight.clear()

    def _record(module: nn.Module | None, token: str) -> None:
        weight = getattr(module, "weight", None)
        if weight is not None:
            _class_of_weight.put(weight, token)

    for block in model.modules():
        name = type(block).__name__
        if name == "DoubleStreamBlock":
            for (outer, inner), token in _DSB_ATTRS.items():
                _record(getattr(getattr(block, outer), inner), token)
            for attr, token in _DSB_MLP_ATTRS.items():
                _record(getattr(block, attr, None), token)
        elif name == "SingleStreamBlock":
            for attr, token in _SSB_ATTRS.items():
                _record(getattr(block, attr, None), token)


def _formats(weight: Tensor) -> tuple[str, str]:
    """Resolve the (forward, backward) GEMM format for a weight tensor."""
    if not _DEMOTED:
        return _FWD_FORMAT, _BWD_FORMAT
    klass = _class_of_weight.get(weight)
    if klass is None:
        raise KeyError(
            f"FLAT_NVFP4_FALLBACK is set but the weight of shape "
            f"{tuple(weight.shape)} was never registered; call "
            f"nvfp4_ops.register_gemm_classes(model) after model setup."
        )
    if klass in _DEMOTED:
        return FP8, FP8
    return _FWD_FORMAT, _BWD_FORMAT


# Quantization helpers

class DualQuantized:
    """A saved activation quantized in two formats at once.

    fwd is in the forward GEMM format, bwd in the backward format.
    """

    __slots__ = ("fwd", "bwd")

    def __init__(self, fwd: object, bwd: object) -> None:
        self.fwd = fwd
        self.bwd = bwd


# (weight tensor, format) -> quantized tensor, keyed on tensor identity.
# Cleared by clear_fp8_weight_cache() after the optimizer step.
_weight_cache = WeightCache()


def clear_fp8_weight_cache() -> None:
    """Clear cached quantized weights. Call after optimizer step."""
    _weight_cache.clear()
    _mx()._MXFP8_IMPL["clear_fp8_weight_cache"]()


def _quantize(t: Tensor, quantizer: NVFP4Quantizer) -> object:
    """Quantize a BF16/FP32 tensor to NVFP4; both dims must be divisible by 16."""
    if not t.is_contiguous():
        t = t.contiguous()
    parts = torch.ops.flux.te_nvfp4_quantize(
        t, _te_ops.quantizer_handle(quantizer),
    )
    return _rebuild_nvfp4(parts, t, quantizer)


def _rebuild_nvfp4(
    parts: Iterable[Tensor | None], src: Tensor, quantizer: NVFP4Quantizer,
) -> NVFP4Tensor:
    """Reassemble an NVFP4Tensor from the six plain tensors an op returned; nothing is copied."""
    rowwise_data, rowwise_scale, columnwise_data, columnwise_scale, \
        amax_rowwise, amax_columnwise = parts
    quantized = NVFP4Tensor(
        shape=src.shape,
        dtype=src.dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale,
        amax_rowwise=amax_rowwise,
        amax_columnwise=amax_columnwise,
        fp4_dtype=tex.DType.kFloat4E2M1,
        quantizer=None,
        with_gemm_swizzled_scales=True,
        row_scaled_nvfp4=quantizer.row_scaled_nvfp4,
        nvfp4_use_4over6=quantizer.nvfp4_use_4over6,
        nvfp4_e4m3_max=quantizer.nvfp4_e4m3_max,
    )
    quantized._quantizer = quantizer
    return quantized


def _quantize_mx(t: Tensor, quantizer: object) -> MXFP8Tensor:
    """Quantize a BF16/FP32 tensor to MXFP8 through the torch.ops boundary."""
    if not t.is_contiguous():
        t = t.contiguous()
    parts = torch.ops.flux.te_mxfp8_quantize(
        t, _te_ops.quantizer_handle(quantizer),
    )
    return _rebuild_mxfp8(parts, t, quantizer)


def _rebuild_mxfp8(
    parts: Iterable[Tensor | None], src: Tensor, quantizer: object,
) -> MXFP8Tensor:
    """Reassemble an MXFP8Tensor from the tensors an op returned; nothing is copied.

    parts is (rowwise_data, rowwise_scale_inv, columnwise_data, columnwise_scale_inv);
    a pair is None for an absent half.
    """
    rowwise_data, rowwise_scale, columnwise_data, columnwise_scale = parts
    quantized = MXFP8Tensor(
        shape=src.shape,
        dtype=src.dtype,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale,
        quantizer=None,
        with_gemm_swizzled_scales=True,
    )
    quantized._quantizer = quantizer
    return quantized


#: Compute capabilities on which the one-read GELU-out cast is on by default.
_ONEPASS_ACT_ARCHS = frozenset({
    (10, 3),
    (10, 7),
})


@_functools.lru_cache(maxsize=1)
def _onepass_act_live() -> bool:
    """Whether the one-read GELU-out cast runs; FLAT_ONEPASS_ACT=0/1 overrides the arch list."""
    override = _os.environ.get("FLAT_ONEPASS_ACT")
    if override is not None:
        return override == "1"
    return torch.cuda.get_device_capability() in _ONEPASS_ACT_ARCHS


def _dual_cast(
    x: Tensor, role: str, *, mx_uses, nv_uses,
) -> tuple[MXFP8Tensor, NVFP4Tensor]:
    """Cast x to MXFP8 (rowwise only) and NVFP4 in two reads.

    The MXFP8 cast also emits the amax the NVFP4 cast consumes, so TE's amax pass is skipped.
    """
    if _mxfp8_columnwise_live(mx_uses):
        raise ValueError(
            f"the dual-format cast emits the MXFP8 rowwise half only, but "
            f"slots {tuple(mx_uses)} read the columnwise half. A cast that "
            f"feeds an MXFP8 backward GEMM is a single both-halves MXFP8 cast "
            f"(_quantize_act / _quantize_weight), not a dual-format pair."
        )
    if not x.is_contiguous():
        x = x.contiguous()
    quantizer = role_quantizer(
        role, columnwise_only=not _nvfp4_rowwise_live(nv_uses),
    )
    rht_amax = bool(quantizer.with_rht and quantizer.with_post_rht_amax)
    data, scales, amax = torch.ops.flux.mxfp8_rowwise_cast_amax_ctm(
        x, quantizer.rht_matrix, rht_amax,
    )
    mxfp8 = _rebuild_mxfp8((data, scales, None, None), x, _MX_FPROP_QUANTIZER)
    columnwise_amax = amax[1:2] if rht_amax else amax[0:1]
    return mxfp8, _rht_cast_ops.cast_with_amax(
        x, quantizer, amax[0:1], columnwise_amax,
    )


# Optimizer-path weight amax, registered by flat/comm_overlap.py before the first
# weight cast of the step; a registered weight takes the one-read dual cast.
_weight_amax_buf: Tensor | None = None
_weight_amax_slot = WeightCache()


def set_weight_amax(
    buf: Tensor | None, slots: Iterable[tuple[Tensor, int]],
) -> None:
    """Register the optimizer-path reduced weight amax.

    buf: [n_slots] f32 CUDA buffer, one reduced amax per quantized weight; None clears.
    slots: (weight, index into buf) pairs, keyed on tensor identity.
    """
    global _weight_amax_buf, _weight_amax_slot
    _weight_amax_buf = buf
    _weight_amax_slot = WeightCache.of(slots) if buf is not None else WeightCache()


def weight_amax_for(weight: Tensor) -> Tensor | None:
    """The optimizer-path amax of weight, or None if none is registered.

    FP32, numel() == 1, on the weight's device: max|x| over the whole parameter.
    Final before any weight cast of the step on the compute stream; other streams must wait.
    """
    slot = _weight_amax_slot.get(weight)
    if slot is None:
        return None
    return _weight_amax_buf[slot:slot + 1]


def _dual_cast_supplied(
    x: Tensor, amax: Tensor, *, mx_uses, nv_uses,
) -> tuple[MXFP8Tensor, NVFP4Tensor]:
    """_dual_cast for a weight whose amax the optimizer path supplies: one read.

    Emits the MXFP8 rowwise half and the NVFP4 columnwise half from a single pass.
    """
    if _mxfp8_columnwise_live(mx_uses):
        raise ValueError(
            f"the dual-format cast emits the MXFP8 rowwise half only, but "
            f"slots {tuple(mx_uses)} read the columnwise half. A cast that "
            f"feeds an MXFP8 backward GEMM is a single both-halves MXFP8 cast "
            f"(_quantize_act / _quantize_weight), not a dual-format pair."
        )
    if _nvfp4_rowwise_live(nv_uses):
        raise ValueError(
            f"the one-read dual-format cast emits the NVFP4 columnwise half "
            f"only, but slots {tuple(nv_uses)} read the rowwise half. A "
            f"weight whose NVFP4 cast feeds a rowwise slot is not a "
            f"dual-format weight (_weight_uses)."
        )
    if not x.is_contiguous():
        x = x.contiguous()
    quantizer = role_quantizer("weight", columnwise_only=True)
    data, scales, col_data, col_scales = torch.ops.flux.dual_cast_one_read(
        x, quantizer.rht_matrix, False, amax,
    )
    mxfp8 = _rebuild_mxfp8((data, scales, None, None), x, _MX_FPROP_QUANTIZER)
    nvfp4 = _rebuild_nvfp4(
        (None, None, col_data, col_scales, None, amax), x, quantizer,
    )
    return mxfp8, nvfp4


def _quantize_weight(weight: Tensor, fmt: str, *, uses) -> object:
    """Quantize a weight in fmt, memoised for the step; keyed on tensor identity.

    In-place weight updates are invisible to the key: call clear_fp8_weight_cache()
    after the optimizer step.
    """
    rowwise_live = fmt != NVFP4 or _nvfp4_rowwise_live(uses)
    columnwise_live = fmt == NVFP4 or _mxfp8_columnwise_live(uses)
    cached = _weight_cache.get(weight, fmt)
    if cached is None:
        if fmt == NVFP4:
            cached = _quantize(weight, role_quantizer(
                "weight", columnwise_only=not rowwise_live,
            ))
        elif (_formats(weight) == (FP8, NVFP4)
              and _weight_cache.get(weight, NVFP4) is None):
            # Dual-consumed weight: MXFP8 by fprop, NVFP4 by dgrad. One dual cast
            # fills both cache entries.
            amax = weight_amax_for(weight)
            dual = _dual_cast if amax is None else (
                lambda w, _role, **kw: _dual_cast_supplied(w, amax, **kw)
            )
            cached, nvfp4_cast = dual(
                weight, "weight",
                mx_uses=uses, nv_uses=_weight_uses(weight, NVFP4),
            )
            _weight_cache.put(weight, nvfp4_cast, NVFP4)
        else:
            cached = _quantize_mx(weight, _mx_quantizer(columnwise_live))
        _weight_cache.put(weight, cached, fmt)
    elif rowwise_live and getattr(cached, "_rowwise_data", 0) is None:
        raise ValueError(
            f"the cached {fmt} weight of shape {tuple(weight.shape)} has no "
            "rowwise half, but this GEMM reads one. A weight is cast "
            "columnwise-only only when its fprop GEMM runs in another "
            "format; a caller asking for both halves after that means the "
            "format resolution disagrees between forward and backward."
        )
    elif columnwise_live and getattr(cached, "_columnwise_data", 0) is None:
        raise ValueError(
            f"the cached {fmt} weight of shape {tuple(weight.shape)} has no "
            "columnwise half, but this GEMM reads one. An MXFP8 weight is "
            "cast rowwise-only only when its dgrad GEMM runs in another "
            "format; a caller asking for it after that means the format "
            "resolution disagrees between forward and backward."
        )
    return cached


#: Operand slots a cast's result is used in, as (layout, side) pairs.
WGRAD_A = (("NT", "A"),)
"""The wgrad operand slot: layout NT, A side, read columnwise."""

FPROP_B = (("TN", "B"),)
"""The fprop activation slot: layout TN, B side, read rowwise."""

DGRAD_A = (("NN", "A"),)
"""The dgrad weight slot: layout NN, A side, read columnwise."""

FPROP_A = (("TN", "A"),)
"""The fprop weight slot: layout TN, A side, read rowwise."""


def _weight_uses(weight: Tensor, fmt: str) -> tuple[tuple[str, str], ...]:
    """Every operand slot this weight's cached fmt cast will serve: fprop's and/or dgrad's."""
    fwd_fmt, bwd_fmt = _formats(weight)
    return (
        (FPROP_A if fwd_fmt == fmt else ())
        + (DGRAD_A if bwd_fmt == fmt else ())
    )


def _nvfp4_rowwise_live(uses: Iterable[tuple[str, str]]) -> bool:
    """Whether any GEMM that will read this tensor reads its rowwise half.

    uses: (layout, side) pairs, side "A" or "B".
    """
    return any(
        _te_ops._NVFP4_READS_ROWWISE[layout][0 if side == "A" else 1]
        for layout, side in uses
    )


def _mxfp8_columnwise_live(uses: Iterable[tuple[str, str]]) -> bool:
    """Whether any GEMM that will read this tensor reads its columnwise half.

    Uses the MXFP8 layout table, not the NVFP4 one.
    """
    return any(
        not _te_ops._MXFP8_READS_ROWWISE[layout][0 if side == "A" else 1]
        for layout, side in uses
    )


def _quantize_act(x: Tensor, fmt: str, *, uses) -> object:
    """Quantize an activation in fmt using that format's input recipe.

    uses: every operand slot the result will occupy, as (layout, side) pairs.
    """
    if fmt == NVFP4:
        return _quantize(x, role_quantizer(
            "act", columnwise_only=not _nvfp4_rowwise_live(uses),
        ))
    return _quantize_mx(x, _mx_quantizer(_mxfp8_columnwise_live(uses)))


def _quantize_grad(grad: Tensor, fmt: str) -> object:
    """Quantize a gradient in fmt using that format's gradient recipe (both halves)."""
    if fmt == NVFP4:
        return _quantize(grad, role_quantizer("grad"))
    return _quantize_mx(grad, _mx()._bwd_quantizer)


def _bgrad_quantize_grad(grad_2d: Tensor, fmt: str | None) -> tuple[Tensor, object]:
    """Bias gradient and the gradient cast from one read of grad_2d.

    Under NVFP4 at a supported shape flux.tma_bgrad_amax_ctm_s supplies grad_b and
    both amaxes; otherwise aten.sum plus a separate cast.
    """
    if fmt == NVFP4 and _tma_bgrad_s_ops.supported(grad_2d):
        quantizer = role_quantizer("grad")
        grad_b, amax = torch.ops.flux.tma_bgrad_amax_ctm_s(
            grad_2d, quantizer.rht_matrix,
        )
        return grad_b, _rht_cast_ops.cast_with_amax(
            grad_2d, quantizer, amax[0:1], amax[1:2],
        )
    grad_b = torch.ops.aten.sum.dim_IntList(grad_2d, [0])
    return grad_b, (grad_2d if fmt is None else _quantize_grad(grad_2d, fmt))


def _is_unquantized(t: object) -> bool:
    """Whether t is a plain tensor needing quantization (not a QuantizedTensor subclass)."""
    return isinstance(t, Tensor) and not isinstance(t, QuantizedTensor)


def _check_format(t: object, fmt: str, what: str) -> None:
    """Raise TypeError unless an already-quantized operand is in the expected format."""
    expected = "NVFP4Tensor" if fmt == NVFP4 else "MXFP8Tensor"
    if type(t).__name__ != expected:
        raise TypeError(
            f"{what} is a {type(t).__name__} but this GEMM runs in {fmt} "
            f"and needs a {expected}. A cuBLAS GEMM cannot mix quantization "
            f"formats; check FLAT_NVFP4_FALLBACK and which GEMM class "
            f"produced this tensor."
        )


def _check_grad_role(t: object) -> None:
    """Raise TypeError unless a quantized operand was produced by the gradient role."""
    quantizer = getattr(t, "_quantizer", None)
    if quantizer is role_quantizer("grad") or quantizer is _mx()._bwd_quantizer:
        return
    raise TypeError(
        f"the gradient operand of this GEMM was not quantized by the "
        f"gradient role quantizer but by {quantizer!r}. The gradient role is "
        f"the only one with stochastic rounding on, and only its RHT cancels "
        f"against the activation operand's in wgrad, so an activation "
        f"substituted here is silently wrong rather than an error."
    )


def _reshape_in(x: Tensor) -> tuple[Tensor, list]:
    """Flatten a 3D activation to 2D for the GEMM; return it and its shape."""
    shape = list(x.shape)
    if len(shape) == 3:
        return torch.ops.aten.reshape.default(x, [-1, shape[-1]]), shape
    return x, shape


def _reshape_out(out_2d: Tensor, shape: list, out_features: int) -> Tensor:
    """Restore the leading dims of a GEMM output."""
    if len(shape) == 3:
        return torch.ops.aten.reshape.default(
            out_2d, [shape[0], shape[1], out_features],
        )
    return out_2d


def _shared_act_cast(x_2d: Tensor, formats: tuple[str, str]) -> object:
    """Cast an activation once per distinct format its consumers need."""
    fwd_fmt, bwd_fmt = formats
    if bwd_fmt == fwd_fmt:
        return _quantize_act(x_2d, fwd_fmt, uses=FPROP_B + WGRAD_A)
    if (fwd_fmt, bwd_fmt) == (FP8, NVFP4):
        return DualQuantized(*_dual_cast(
            x_2d, "act", mx_uses=FPROP_B, nv_uses=WGRAD_A,
        ))
    return DualQuantized(
        _quantize_act(x_2d, fwd_fmt, uses=FPROP_B),
        _quantize_act(x_2d, bwd_fmt, uses=WGRAD_A),
    )


def release_fwd(cached: object) -> None:
    """Drop the fprop half of a :class:`DualQuantized`, once fprop has read it.

    Under R2 a cached activation is quantized twice -- MXFP8 for the forward
    GEMM, NVFP4 for the wgrad -- and both halves are then alive for the whole
    forward. Only the backward half is ever read again: the two readers of the
    forward half, :func:`fp8_linear_fwd_save` and :func:`fp8_linear_fwd_preq`,
    both run before this point, and every backward consumer goes through
    :func:`_bwd_operand`. Holding the forward half to the end of the forward
    costs 1.03125 B per element of every dual-cast activation.

    Call it immediately after the LAST forward GEMM that consumes ``cached``.
    A call placed one GEMM early is a correctness bug that still frees the
    memory, so it looks exactly like a working optimization.

    Args:
        cached: A forward feeder's return value -- a :class:`DualQuantized`, a
            single quantized tensor when the two formats coincide, or BF16 when
            the consumers disagreed. Only the first case holds two halves;
            the others are left alone.
    """
    if isinstance(cached, DualQuantized):
        cached.fwd = None


def _fwd_operand(cached: object) -> object:
    """Select the forward-format half of a shared activation; pass anything else through."""
    if isinstance(cached, DualQuantized):
        return cached.fwd
    return cached


def _bwd_operand(cached: object, fmt: str) -> object:
    """Select the backward-format half of a saved activation, quantizing BF16 for wgrad."""
    if isinstance(cached, DualQuantized):
        return cached.bwd
    if _is_unquantized(cached):
        return _quantize_act(cached, fmt, uses=WGRAD_A)
    _check_format(cached, fmt, "saved activation")
    return cached


def _gemm(
    a: object, b: object, layout: str,
    bias: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Issue one quantized GEMM through torch.ops.flux; both operands must share a format."""
    if isinstance(a, NVFP4Tensor) and isinstance(b, NVFP4Tensor):
        return torch.ops.flux.te_nvfp4_gemm(
            a._rowwise_data, a._rowwise_scale_inv,
            a._columnwise_data, a._columnwise_scale_inv,
            a._amax_rowwise, a._amax_columnwise,
            b._rowwise_data, b._rowwise_scale_inv,
            b._columnwise_data, b._columnwise_scale_inv,
            b._amax_rowwise, b._amax_columnwise,
            layout, a._nvfp4_e4m3_max, bias, out,
        )
    if isinstance(a, MXFP8Tensor) and isinstance(b, MXFP8Tensor):
        return torch.ops.flux.te_mxfp8_gemm(
            a._rowwise_data, a._rowwise_scale_inv,
            a._columnwise_data, a._columnwise_scale_inv,
            b._rowwise_data, b._rowwise_scale_inv,
            b._columnwise_data, b._columnwise_scale_inv,
            layout, bias, out,
        )
    raise TypeError(
        f"GEMM operands must both be NVFP4 or both MXFP8, got "
        f"{type(a).__name__} and {type(b).__name__}. A cuBLAS GEMM cannot "
        f"mix quantization formats."
    )


# Forward

def _fwd_gemm(
    x_2d: Tensor | object, weight: Tensor, bias: Tensor | None, fmt: str,
    prequantized: bool = False,
) -> Tensor:
    """Issue the fprop GEMM out = x @ weight^T + bias in fmt.

    x_2d: [M, K], BF16 or already quantized in fmt when prequantized. weight: [N, K] BF16.
    """
    if prequantized:
        _check_format(x_2d, fmt, "forward activation")
        qx = x_2d
    else:
        qx = _quantize_act(x_2d, fmt, uses=FPROP_B)
    qw = _quantize_weight(weight, fmt, uses=_weight_uses(weight, fmt))
    return _gemm(qw, qx, "TN", bias=bias)


def fp8_linear_fwd(
    x: Tensor, weight: Tensor, bias: Tensor,
) -> Tensor:
    """NVFP4 linear forward: out = x @ weight^T + bias.

    x: [*, in_features], weight: [out_features, in_features], bias: [out_features]; all BF16.
    """
    fwd_fmt, _ = _formats(weight)
    x_2d, shape = _reshape_in(x)
    out_2d = _fwd_gemm(x_2d, weight, bias, fwd_fmt)
    return _reshape_out(out_2d, shape, weight.shape[0])


def fp8_linear_fwd_save(
    x: Tensor, weight: Tensor, bias: Tensor,
) -> tuple[Tensor, object]:
    """NVFP4 linear forward that also returns the quantized activation for wgrad.

    Returns (output, cached_x); cached_x is an NVFP4Tensor, or a DualQuantized when
    forward and backward formats differ.
    """
    fwd_fmt, bwd_fmt = _formats(weight)
    x_2d, shape = _reshape_in(x)
    cached = _shared_act_cast(x_2d, (fwd_fmt, bwd_fmt))
    out_2d = _fwd_gemm(
        _fwd_operand(cached), weight, bias, fwd_fmt, prequantized=True,
    )
    # The GEMM above is the ONLY forward reader of the fprop half -- the caller
    # keeps `cached` solely to hand the backward its wgrad operand, which comes
    # from _bwd_operand. Releasing here rather than at each call site covers
    # every fwd_save activation (both DSB qkv/proj/mlp_in pairs and the SSB
    # attn_flat) without model.py having to name them.
    release_fwd(cached)
    return _reshape_out(out_2d, shape, weight.shape[0]), cached


def fp8_linear_fwd_gelu(
    x: Tensor, weight: Tensor, bias: Tensor,
) -> tuple[Tensor, Tensor]:
    """NVFP4 linear forward with GELU fused into the GEMM epilogue (not implemented)."""
    raise NotImplementedError(
        "fp8_linear_fwd_gelu has no verified NVFP4 implementation: the "
        "general_gemm(gelu=True) epilogue is unreachable from flat/model.py "
        "and was therefore never validated against a BF16 reference."
    )


def fp8_linear_fwd_preq(
    fp8_x, weight: Tensor, bias: Tensor, orig_shape: list,
) -> Tensor:
    """NVFP4 linear forward with a pre-quantized input.

    fp8_x: a quantized tensor, a DualQuantized (forward half used), or BF16 (quantized here).
    """
    fwd_fmt, _ = _formats(weight)
    operand = _fwd_operand(fp8_x)
    out_2d = _fwd_gemm(
        operand, weight, bias, fwd_fmt,
        prequantized=not _is_unquantized(operand),
    )
    return _reshape_out(out_2d, orig_shape, weight.shape[0])


# Backward

def _wgrad_delay_unsupported(entry_point: str) -> None:
    """Refuse the deferred-wgrad entry points in NVFP4 mode (WGRAD_DELAY unsupported)."""
    raise NotImplementedError(
        f"{entry_point} has no verified NVFP4 implementation. It is only "
        "reachable with WGRAD_DELAY=1 or WGRAD_DELAY_GRAPH=1, which no "
        "config in this benchmark sets, so the NVFP4 port never validated "
        "it. Run NVFP4 with WGRAD_DELAY=0, or implement and verify the "
        "deferred-wgrad path against a BF16 reference first."
    )


def _dgrad_gemm(qgrad: object, weight: Tensor, fmt: str) -> Tensor:
    """Issue the dgrad GEMM grad_x = grad @ weight in fmt.

    dgrad reads the weight's columnwise half (layout NN, A side).
    """
    _check_grad_role(qgrad)
    return _gemm(
        _quantize_weight(weight, fmt, uses=_weight_uses(weight, fmt)),
        qgrad, "NN",
    )


def _wgrad_gemm(
    qx: object, qgrad: object, wgrad_out: Tensor | None,
) -> Tensor:
    """Issue the wgrad GEMM grad_w = grad^T @ x."""
    _check_grad_role(qgrad)
    return _gemm(qx, qgrad, "NT", out=wgrad_out)


def fp8_linear_bwd(
    grad: Tensor, x: Tensor | None, weight: Tensor,
    wgrad_out: Tensor | None = None,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """NVFP4 linear backward; returns (grad_x, grad_w, grad_b) in BF16.

    x may be None when cached_fp8_x (from fp8_linear_fwd_save) and x_shape are given.
    """
    _, bwd_fmt = _formats(weight)
    grad_2d, _ = _reshape_in(grad)
    grad_b, qgrad = _bgrad_quantize_grad(grad_2d, bwd_fmt)
    return _linear_bwd_from_qgrad(
        qgrad, grad_b, x, weight, wgrad_out, bwd_fmt,
        cached_fp8_x=cached_fp8_x, x_shape=x_shape,
    )


def _linear_bwd_from_qgrad(
    qgrad: object, grad_b: Tensor, x: Tensor | None, weight: Tensor,
    wgrad_out: Tensor | None, bwd_fmt: str,
    *,
    cached_fp8_x: object | None,
    x_shape: list | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """The two backward GEMMs, given an already-quantized gradient.

    Returns (grad_x, grad_w, grad_b). x may be None when cached_fp8_x and x_shape are given.
    """
    shape = x_shape if x_shape is not None else list(x.shape)
    if cached_fp8_x is not None:
        qx = _bwd_operand(cached_fp8_x, bwd_fmt)
    else:
        x_2d, _ = _reshape_in(x)
        qx = _quantize_act(x_2d, bwd_fmt, uses=WGRAD_A)

    grad_x_2d = _dgrad_gemm(qgrad, weight, bwd_fmt)
    grad_w = _wgrad_gemm(qx, qgrad, wgrad_out)

    grad_x = (
        torch.ops.aten.reshape.default(grad_x_2d, shape)
        if len(shape) == 3 else grad_x_2d
    )
    return grad_x, grad_w, grad_b


def _dgelu_dbias(grad_2d: Tensor, pre_2d: Tensor, fmt: str | None) -> tuple:
    """Fused dGELU + dbias + amax, then the amax-supplied NVFP4 cast.

    Returns (grad_b, dgelu); dgelu is an NVFP4Tensor when fmt is nvfp4, else BF16.
    """
    quantizer = role_quantizer("grad")
    dgelu, grad_b, amax = torch.ops.flux.dgelu_dbias_amax(
        grad_2d, pre_2d, quantizer.rht_matrix,
    )
    if fmt != NVFP4:
        return grad_b, dgelu
    return grad_b, _rht_cast_ops.cast_with_amax(
        dgelu, quantizer, amax[0:1], amax[1:2],
    )


def fp8_linear_bwd_dgrad(
    grad: Tensor, x: Tensor | None, weight: Tensor,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, dict]:
    """NVFP4 linear backward, dgrad only; unsupported, see _wgrad_delay_unsupported."""
    _wgrad_delay_unsupported("fp8_linear_bwd_dgrad")


def fp8_linear_bwd_wgrad(
    ctx: dict,
    wgrad_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """NVFP4 linear backward, wgrad + bias only; unsupported, see _wgrad_delay_unsupported."""
    _wgrad_delay_unsupported("fp8_linear_bwd_wgrad")


def fp8_linear_bwd_dgelu(
    grad: Tensor, gelu_input: Tensor,
    x: Tensor, weight: Tensor,
    wgrad_out: Tensor | None = None,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """NVFP4 linear backward for a linear whose output feeds a GELU.

    gelu_input is the saved pre-GELU activation. Returns (grad_x, grad_w, grad_b) in BF16.
    """
    _, bwd_fmt = _formats(weight)
    grad_2d, _ = _reshape_in(grad)
    gelu_in_2d, _ = _reshape_in(gelu_input)
    grad_b, qgrad = _dgelu_dbias(grad_2d, gelu_in_2d, bwd_fmt)
    if bwd_fmt != NVFP4:
        qgrad = _quantize_grad(qgrad, bwd_fmt)
    return _linear_bwd_from_qgrad(
        qgrad, grad_b, x, weight, wgrad_out, bwd_fmt,
        cached_fp8_x=cached_fp8_x, x_shape=x_shape,
    )


def fp8_linear_bwd_dgrad_dgelu(
    grad: Tensor, gelu_input: Tensor,
    x: Tensor, weight: Tensor,
    *,
    cached_fp8_x: object | None = None,
    x_shape: list | None = None,
) -> tuple[Tensor, dict]:
    """NVFP4 linear backward, dgrad only with dgelu; unsupported, see _wgrad_delay_unsupported."""
    _wgrad_delay_unsupported("fp8_linear_bwd_dgrad_dgelu")


def fp8_linear_bwd_dgrad_preq_grad(
    fp8_grad, grad_b, weight: Tensor,
    orig_shape: list,
    *,
    cached_fp8_x: object | None = None,
) -> tuple[Tensor, dict]:
    """NVFP4 dgrad-only backward with a feeder's gradient; see _wgrad_delay_unsupported."""
    _wgrad_delay_unsupported("fp8_linear_bwd_dgrad_preq_grad")


def fp8_linear_bwd_preq_grad(
    fp8_grad, x: Tensor, weight: Tensor,
    wgrad_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """NVFP4 linear backward with the gradient supplied by a feeder.

    Returns (grad_x, grad_w) in BF16; the bias gradient is the caller's.
    """
    return fp8_linear_bwd_preq_both(
        fp8_grad, x, weight, list(x.shape), wgrad_out,
    )


def fp8_linear_bwd_preq_both(
    fp8_grad, fp8_x, weight: Tensor,
    orig_shape: list,
    wgrad_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """NVFP4 linear backward with both operands supplied by feeders.

    Returns (grad_x, grad_w) in BF16; the bias gradient is the caller's.
    """
    _, bwd_fmt = _formats(weight)
    if _is_unquantized(fp8_grad):
        qgrad = _quantize_grad(fp8_grad, bwd_fmt)
    else:
        _check_format(fp8_grad, bwd_fmt, "gradient")
        qgrad = fp8_grad
    qx = _bwd_operand(fp8_x, bwd_fmt)

    grad_x_2d = _dgrad_gemm(qgrad, weight, bwd_fmt)
    grad_w = _wgrad_gemm(qx, qgrad, wgrad_out)

    grad_x = (
        torch.ops.aten.reshape.default(grad_x_2d, orig_shape)
        if len(orig_shape) == 3 else grad_x_2d
    )
    return grad_x, grad_w


# Quantize-only entry points (the _preq feeders). A feeder receives the weights
# of every consuming GEMM and casts once per distinct format they resolve to.

def _shared_act_format(*weights: Tensor) -> tuple[str, str] | None:
    """The (fwd, bwd) formats every consuming GEMM agrees on, or None if they disagree."""
    if not weights:
        raise ValueError(
            "a forward feeder needs at least one consuming weight to resolve "
            "the format to cast in; without one it cannot share the cast."
        )
    fmts = {_formats(w) for w in weights}
    if len(fmts) == 1:
        (agreed,) = fmts
        return agreed
    if not _DEMOTED:
        raise RuntimeError(
            f"the GEMMs consuming this activation disagree on a format "
            f"({sorted(fmts)}) but FLAT_NVFP4_FALLBACK demoted nothing, so "
            f"they should all resolve to ({_FWD_FORMAT}, {_BWD_FORMAT}). The "
            f"feeder cannot share a cast across them and would silently issue "
            f"one per consumer instead."
        )
    return None


def fp8_quantize_fwd(x: Tensor, *weights: Tensor) -> tuple:
    """Quantize a forward activation once for every GEMM that consumes it.

    Returns (fp8_x, orig_shape); fp8_x is quantized, a DualQuantized when forward and
    backward formats differ, or BF16 if the consumers disagree.
    """
    x_2d, shape = _reshape_in(x)
    formats = _shared_act_format(*weights)
    return (
        x_2d if formats is None else _shared_act_cast(x_2d, formats)
    ), shape


def fp8_quantize_bwd(x: Tensor) -> tuple:
    """Pass a saved activation through as (x_2d, orig_shape), still BF16."""
    return _reshape_in(x)


def _gelu(x_2d: Tensor) -> Tensor:
    """The forward tanh-GELU kernel; FLAT_TANH_APPROX_GELU=0 selects TE's."""
    if _os.environ.get("FLAT_TANH_APPROX_GELU", "1") == "1":
        return torch.ops.flux.gelu_tanh_ctm_v2_auto(x_2d, True)
    return torch.ops.flux.te_gelu(x_2d)


def _gelu_dual_cast(x_2d: Tensor) -> tuple[MXFP8Tensor, NVFP4Tensor]:
    """Fast tanh-GELU plus its dual cast, with the GELU output read once.

    Returns (MXFP8 rowwise-only, NVFP4 columnwise-only).
    """
    if _nvfp4_rowwise_live(WGRAD_A):
        raise ValueError(
            f"the one-read GELU-out cast emits the NVFP4 columnwise half only, "
            f"but slots {WGRAD_A} read the rowwise half."
        )
    quantizer = role_quantizer("act", columnwise_only=True)
    if not (quantizer.with_rht and quantizer.with_post_rht_amax):
        raise ValueError(
            "the one-read GELU-out cast consumes the post-RHT amax; the act "
            "quantizer has no RHT or no post-RHT amax."
        )
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    y, amax = torch.ops.flux.gelu_tanh_amax(x_2d, quantizer.rht_matrix)
    data, scales, col_data, col_scales = torch.ops.flux.onepass_act(
        y, quantizer.rht_matrix, amax[1:2],
    )
    mxfp8 = _rebuild_mxfp8((data, scales, None, None), y, _MX_FPROP_QUANTIZER)
    # TE attaches no rowwise amax to a columnwise-only tensor; the GEMM derives
    # its dequant scale from _amax_columnwise.
    nvfp4 = _rebuild_nvfp4(
        (None, None, col_data, col_scales, None, amax[1:2]), y, quantizer,
    )
    return mxfp8, nvfp4


def fp8_gelu_fwd(x: Tensor, *weights: Tensor) -> tuple:
    """Tanh-GELU, quantized once for every GEMM that consumes the result."""
    x_2d, shape = _reshape_in(x)
    formats = _shared_act_format(*weights)
    if (
        formats == (FP8, NVFP4)
        and _os.environ.get("FLAT_TANH_APPROX_GELU", "1") == "1"
        and _onepass_act_live()
    ):
        return DualQuantized(*_gelu_dual_cast(x_2d)), shape
    act = _gelu(x_2d)
    return (
        act if formats is None else _shared_act_cast(act, formats)
    ), shape


def fp8_qgelu_fwd(x: Tensor, *weights: Tensor) -> tuple:
    """Alias of fp8_gelu_fwd."""
    return fp8_gelu_fwd(x, *weights)


def _shared_grad_format(*weights: Tensor) -> str | None:
    """The backward format every consuming GEMM agrees on, or None if they disagree."""
    if not weights:
        raise ValueError(
            "a gradient feeder needs at least one consuming weight to resolve "
            "the format to cast in; without one it cannot share the cast."
        )
    fmts = {_formats(w)[1] for w in weights}
    if len(fmts) == 1:
        (agreed,) = fmts
        return agreed
    if not _DEMOTED:
        raise RuntimeError(
            f"the GEMMs consuming this gradient disagree on a backward format "
            f"({sorted(fmts)}) but FLAT_NVFP4_FALLBACK demoted nothing, so "
            f"they should all resolve to {_BWD_FORMAT}. The feeder cannot "
            f"share a cast across them and would silently issue one per "
            f"consumer instead."
        )
    return None


def fp8_bgrad_quantize(grad: Tensor, *weights: Tensor) -> tuple:
    """Bias gradient, and the gradient cast once for every consuming GEMM.

    Returns (grad_b, fp8_grad, orig_shape); fp8_grad is BF16 if the consumers disagree.
    """
    grad_2d, shape = _reshape_in(grad)
    grad_b, qgrad = _bgrad_quantize_grad(grad_2d, _shared_grad_format(*weights))
    return grad_b, qgrad, shape


def fp8_rope_join_bgrad_quantize(
    grad_q_rope: Tensor, grad_k_rope: Tensor, grad_v: Tensor, qkv: Tensor, H: int,
    q_rstd: Tensor, k_rstd: Tensor, qn_weight: Tensor, kn_weight: Tensor,
    grad_qn_weight: Tensor, grad_kn_weight: Tensor, freqs: Tensor, pos_offset: int,
    *weights: Tensor,
) -> tuple:
    """RoPE/QK-norm backward + QKV join, bias gradient, and the gradient cast once.

    Returns (grad_b, fp8_grad, orig_shape).
    """
    fmt = _shared_grad_format(*weights)
    args = (grad_q_rope, grad_k_rope, grad_v, qkv, H, q_rstd, k_rstd, qn_weight,
            kn_weight, grad_qn_weight, grad_kn_weight, freqs, pos_offset)
    B, L, D = grad_q_rope.shape
    if fmt == NVFP4 and _rope_bgrad_ops.fused(B * L * 3 * D, grad_q_rope.device.index):
        quantizer = role_quantizer("grad")
        grad_qkv, grad_b, amax = torch.ops.flux.rope_norm_join_bgrad_amax_ctm(
            *args, quantizer.rht_matrix,
        )
        grad_2d = grad_qkv.view(B * L, 3 * D)
        qgrad = _rht_cast_ops.cast_with_amax(
            grad_2d, quantizer, amax[0:1], amax[1:2],
        )
        return grad_b, qgrad, [B, L, 3 * D]
    grad_2d, shape = _reshape_in(torch.ops.flux.rope_norm_join_ctm(*args))
    grad_b, qgrad = _bgrad_quantize_grad(grad_2d, fmt)
    return grad_b, qgrad, shape


def fp8_gate_bwd_quantize(
    grad: Tensor, gate: Tensor, proj: Tensor, *weights: Tensor,
    bias_grad: Tensor | None = None,
) -> tuple:
    """Gate backward, bias gradient, and grad_proj cast once for its consumers.

    When bias_grad is given, the bias gradient is written straight into that
    buffer and returned as grad_b; the caller then has nothing to accumulate.

    Returns (grad_b, fp8_grad_proj, orig_shape, grad_gate).
    """
    fmt = _shared_grad_format(*weights)
    if fmt == NVFP4 and _gate_bwd_ctm_ops.supported(grad, gate, proj):
        # The producer also emits grad_b (FP32 [D]) via a finaliser on a forked
        # stream; join_dbias_ctm() below makes it valid for the consumer.
        quantizer = role_quantizer("grad")
        grad_proj, grad_gate, amax, grad_b = (
            torch.ops.flux.gate_bwd_dbias_amax_deferred_ctm(
                grad, gate, proj, quantizer.rht_matrix, bias_grad,
            )
        )
        grad_2d, shape = _reshape_in(grad_proj)
        qgrad = _rht_cast_ops.cast_with_amax(
            grad_2d, quantizer, amax[0:1], amax[1:2],
        )
        _gate_bwd_ctm_ops.join_dbias_ctm(grad.device)
        return grad_b, qgrad, shape, grad_gate
    grad_proj, grad_gate = torch.ops.flux.fused_gate_bwd(grad, gate, proj)
    grad_2d, shape = _reshape_in(grad_proj)
    if bias_grad is not None:
        torch.ops.flux.bias_colsum_ctm(grad_2d, bias_grad)
        grad_b = bias_grad
    else:
        grad_b = torch.ops.aten.sum.dim_IntList(grad_2d, [0])
    return grad_b, (
        grad_2d if fmt is None else _quantize_grad(grad_2d, fmt)
    ), shape, grad_gate


def fp8_dbias_dqgelu_bwd(grad: Tensor, fwd_input: Tensor) -> tuple:
    """dgelu + bias gradient, with the gradient cast for the consuming GEMM.

    Returns (grad_b, dgelu_2d, orig_shape); dgelu_2d is an NVFP4Tensor when every class
    runs NVFP4 backward, else BF16 for the consumer to cast.
    """
    grad_2d, shape = _reshape_in(grad)
    fwd_2d, _ = _reshape_in(fwd_input)
    fmt = None if _DEMOTED else _BWD_FORMAT
    grad_b, dgelu = _dgelu_dbias(grad_2d, fwd_2d, fmt)
    return grad_b, dgelu, shape


SEAM: tuple[str, ...] = (
    "fp8_linear_fwd",
    "fp8_linear_fwd_save",
    "fp8_linear_fwd_gelu",
    "fp8_linear_fwd_preq",
    "fp8_linear_bwd",
    "fp8_linear_bwd_dgrad",
    "fp8_linear_bwd_wgrad",
    "fp8_linear_bwd_dgelu",
    "fp8_linear_bwd_dgrad_dgelu",
    "fp8_linear_bwd_dgrad_preq_grad",
    "fp8_linear_bwd_preq_grad",
    "fp8_linear_bwd_preq_both",
    "fp8_quantize_fwd",
    "fp8_quantize_bwd",
    "release_fwd",
    "fp8_gelu_fwd",
    "fp8_bgrad_quantize",
    "fp8_rope_join_bgrad_quantize",
    "fp8_gate_bwd_quantize",
    "fp8_qgelu_fwd",
    "fp8_dbias_dqgelu_bwd",
    "clear_fp8_weight_cache",
)
"""Names fp8_ops rebinds when NVFP4 is selected."""


def _bind_bf16_bisect(names: Iterable[str]) -> None:
    """Route the NVFP4 path through BF16 for FLAT_NVFP4_BF16 bisection."""
    from flat import fp8_ops_bf16 as bf16
    for name in names:
        globals()[name] = getattr(bf16, name)


if _FLAT_NVFP4_BF16:
    _bind_bf16_bisect(SEAM)


def forward_format() -> str:
    """The forward GEMM format this configuration selected."""
    return _FWD_FORMAT


def backward_format() -> str:
    """The dgrad/wgrad format this configuration selected."""
    return _BWD_FORMAT
