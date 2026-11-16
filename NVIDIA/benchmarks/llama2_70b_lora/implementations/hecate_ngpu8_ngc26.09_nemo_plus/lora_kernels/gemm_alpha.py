# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""NVFP4 GEMMs with the alpha from our quantize kernels (switch LORA_PATCH_GEMM_ALPHA).

Before each NVFP4 GEMM, TE starts a one-thread kernel that computes alpha from the two amax
values. With this patch, the quantize kernel of the GEMM input also writes the alpha. The GEMM
runs through nvte_cublas_gemm_v2 with operands that have no amax, so TE starts no alpha
kernel. The alpha and the GEMM output have the same bits as on the TE path.

The first eager GEMM of each input quantizer finds the weight of that GEMM. From the next
call on, the quantize kernel writes the alpha into a slot of that quantizer, and the GEMM
reads the slot. Other GEMMs and the first call run the TE path.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
import transformer_engine.pytorch.ops.basic.basic_linear as basic_linear
from transformer_engine.pytorch.cpp_extensions.gemm import get_cublas_workspace
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage

import nvte_gemm

ENABLED = os.environ.get("LORA_PATCH_GEMM_ALPHA", "0") == "1"
# Compare the first N routed GEMMs (outside graph capture) with the TE path, bit by bit.
CHECK_CALLS = int(os.environ.get("LORA_GEMM_ALPHA_CHECK", "0"))
# Records that no GEMM used, for example after an exception. Above this count, drop them all.
_MAX_RECORDS = 4096


@dataclass
class _Target:
    """The GEMM of one input quantizer: the weight amax and the alpha slot."""

    layout: str
    weight_amax: torch.Tensor
    alpha: torch.Tensor


_state = {"original": None, "gemm": None, "beta": {}}
# id(quantizer) -> (quantizer, _Target or None). None: the quantizer meets two weights.
_targets: dict = {}
# id(amax tensor of a quantize output) -> (amax tensor, quantizer, wrote_alpha).
_records: dict = {}
counts = {"routed": 0, "learned": 0, "fallback": 0, "conflict": 0, "checked": 0, "mismatch": 0}


def kernel_tensors(quantizer: object) -> tuple:
    """Return (weight amax, alpha slot) for the quantize kernel, or (None, None)."""
    if not ENABLED:
        return None, None
    entry = _targets.get(id(quantizer))
    if entry is None or entry[0] is not quantizer or entry[1] is None:
        return None, None
    return entry[1].weight_amax, entry[1].alpha


def record(amax: torch.Tensor, quantizer: object, wrote_alpha: bool) -> None:
    """Record a quantize output (by its amax tensor) for the next GEMM."""
    if not ENABLED:
        return
    if len(_records) > _MAX_RECORDS:
        _records.clear()
    _records[id(amax)] = (amax, quantizer, wrote_alpha)


def stats() -> dict:
    """Return the counters."""
    return dict(counts, targets=len(_targets), open_records=len(_records))


def clear() -> None:
    """Drop the records that no GEMM used (call it at each train step start)."""
    _records.clear()


def _plain_call(kwargs: dict) -> bool:
    """Return True for a GEMM call with a bf16 output and no epilogue, overlap or scale."""
    accumulate = bool(kwargs.get("accumulate", False))
    return (
        kwargs.get("layout", "TN") in ("TN", "NN")
        and kwargs.get("quantization_params") is None
        and kwargs.get("bias") is None
        and not kwargs.get("gelu", False)
        and kwargs.get("gelu_in") is None
        and kwargs.get("ub") is None
        and kwargs.get("extra_output") is None
        and not kwargs.get("bulk_overlap", False)
        and kwargs.get("alpha", 1.0) in (1.0, None)
        and kwargs.get("beta") in (None, 1.0 if accumulate else 0.0)
        and kwargs.get("out_dtype") in (None, torch.bfloat16)
    )


def _plain_operand(tensor: object) -> bool:
    """Return True for an NVFP4 tensor with GEMM-swizzled scales and one tensor amax."""
    return (
        isinstance(tensor, NVFP4TensorStorage)
        and bool(tensor._with_gemm_swizzled_scales)
        and not getattr(tensor, "_row_scaled_nvfp4", False)
    )


def _operands(a: object, b: object, layout: str) -> tuple | None:
    """Return (a_data, a_scales, a_amax, b_data, b_scales, b_amax) as TE reads them, or None."""
    if layout == "TN":
        a_parts = (a._rowwise_data, a._rowwise_scale_inv, a._amax_rowwise)
    else:
        a_parts = (a._columnwise_data, a._columnwise_scale_inv, a._amax_columnwise)
    parts = a_parts + (b._rowwise_data, b._rowwise_scale_inv, b._amax_rowwise)
    if any(part is None for part in parts):
        return None
    return parts


def _learn(quantizer: object, layout: str, weight_amax: torch.Tensor) -> None:
    """Give the quantizer an alpha slot for this weight. Two weights: no slot."""
    entry = _targets.get(id(quantizer))
    if entry is not None and entry[0] is quantizer:
        target = entry[1]
        if target is not None and (target.weight_amax is not weight_amax or target.layout != layout):
            _targets[id(quantizer)] = (quantizer, None)
            counts["conflict"] += 1
        return
    alpha = torch.zeros(1, dtype=torch.float32, device=weight_amax.device)
    _targets[id(quantizer)] = (quantizer, _Target(layout, weight_amax, alpha))
    counts["learned"] += 1
    # Make the beta constants and the GEMM object here, outside of a graph capture.
    _beta(weight_amax.device)
    if _state["gemm"] is None:
        _state["gemm"] = nvte_gemm.NvteGemm()


def _beta(device: torch.device) -> torch.Tensor:
    """Return the device constants [0, 1] for beta. Make them before graph capture."""
    beta = _state["beta"].get(device)
    if beta is None:
        beta = torch.tensor([0.0, 1.0], dtype=torch.float32, device=device)
        _state["beta"][device] = beta
    return beta


def _output(a_data: torch.Tensor, b_data: torch.Tensor, kwargs: dict) -> torch.Tensor | None:
    """Return the output tensor: the given one if it fits, a new one, or None (no fit)."""
    shape = tuple(b_data.shape[:-1]) + (int(np.prod(a_data.shape[:-1])),)
    out = kwargs.get("out")
    if out is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=b_data.device)
    if out.dtype != torch.bfloat16 or not out.is_contiguous() or out.numel() != int(np.prod(shape)):
        return None
    return out


def _routed_gemm(layout: str, parts: tuple, target: _Target, out: torch.Tensor,
                 kwargs: dict) -> None:
    """Run the GEMM with the alpha slot of the target."""
    a_data, a_scales, _, b_data, b_scales, _ = parts
    beta = _beta(out.device)
    beta = beta[1:] if kwargs.get("accumulate", False) else beta[:1]
    workspace = get_cublas_workspace(out.device.index, False, False)
    options = (bool(kwargs.get("use_split_accumulator", False)), nvte_gemm.math_sm_count(out.device),
               workspace, nvte_gemm.te_workspace_bytes(workspace))
    _state["gemm"].gemm(layout, (a_data, a_scales, b_data, b_scales), out,
                        (target.alpha, beta), options)


def _check(a: object, b: object, out: torch.Tensor, kwargs: dict) -> None:
    """Compare one routed GEMM with the TE path, bit by bit (no accumulate, no capture)."""
    reference = _state["original"](a, b, **dict(kwargs, out=None))[0]
    counts["checked"] += 1
    if not torch.equal(reference.view(torch.int16), out.view(torch.int16)):
        counts["mismatch"] += 1
        print(f"[gemm_alpha] MISMATCH layout={kwargs.get('layout', 'TN')} shape={tuple(out.shape)}",
              flush=True)


def _target_for(rec: tuple, layout: str, weight_amax: torch.Tensor) -> _Target | None:
    """Return the target of a record if its kernel wrote the alpha for this weight, else None."""
    quantizer, wrote_alpha = rec[1], rec[2]
    entry = _targets.get(id(quantizer))
    target = entry[1] if entry is not None and entry[0] is quantizer else None
    if wrote_alpha and target is not None and target.layout == layout \
            and target.weight_amax is weight_amax:
        return target
    return None


def general_gemm(a: object, b: object, *args: object, **kwargs: object) -> tuple:
    """Replacement of general_gemm in te.ops BasicLinear."""
    plain = not args and _plain_call(kwargs) and _plain_operand(a) and _plain_operand(b)
    layout = kwargs.get("layout", "TN")
    parts = _operands(a, b, layout) if plain else None
    rec = None if parts is None else _records.pop(id(parts[5]), None)
    if rec is None or rec[0] is not parts[5]:
        counts["fallback"] += 1
        return _state["original"](a, b, *args, **kwargs)
    target = _target_for(rec, layout, parts[2])
    out = None if target is None else _output(parts[0], parts[3], kwargs)
    if out is None:
        if not torch.cuda.is_current_stream_capturing():
            _learn(rec[1], layout, parts[2])
        counts["fallback"] += 1
        return _state["original"](a, b, *args, **kwargs)
    _routed_gemm(layout, parts, target, out, kwargs)
    counts["routed"] += 1
    capturing = torch.cuda.is_current_stream_capturing()
    if counts["checked"] < CHECK_CALLS and not capturing and not kwargs.get("accumulate", False):
        _check(a, b, out, kwargs)
    return out, None, None, None


def apply() -> None:
    """Install the GEMM replacement in te.ops BasicLinear."""
    if _state["original"] is not None:
        return
    _state["original"] = basic_linear.general_gemm
    basic_linear.general_gemm = general_gemm
