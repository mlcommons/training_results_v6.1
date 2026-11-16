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

"""Tensor capture at the flat/fp8_ops.py recipe seam.

Dumps high-precision GEMM operands from a training run for scripts/quant_study.py.
"""

from __future__ import annotations

import inspect
import json
import math
import os
from typing import Any, Callable

import torch
from torch import Tensor
from transformer_engine.pytorch.tensor import QuantizedTensor

from flat.weight_cache import WeightCache

#: Roles a captured tensor can play in a GEMM.
ROLE_ACT = "act"
ROLE_WEIGHT = "weight"
ROLE_GRAD = "grad"

#: flat/model.py line -> (role, gemm class, source fragment that must be on that
#: line). Key = line of the callee name (what f_lineno reports); "a|b" = two classes.
_LINE_SITES: dict[int, tuple[str, str, str]] = {
    685: (ROLE_ACT, "dsb.img.fc2", "fp8_gelu_fwd("),
    702: (ROLE_ACT, "dsb.txt.fc2", "fp8_gelu_fwd("),
    1192: (ROLE_ACT, "ssb.qkv|ssb.fc1", "fp8_quantize_fwd("),
    1219: (ROLE_ACT, "ssb.fc2", "fp8_gelu_fwd("),
    1386: (ROLE_GRAD, "ssb.qkv", "fp8_bgrad_quantize(grad_qkv,"),
    1396: (ROLE_GRAD, "ssb.fc1", "fp8_dbias_dqgelu_bwd(grad_mlp_act"),
}

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


class _State:
    """Mutable capture state, kept off the module namespace."""

    def __init__(self) -> None:
        self.enabled = False
        self.out_dir = ""
        self.steps: set[int] = set()
        self.blocks: set[int] = set()
        self.rows = 512
        self.step = -1
        self.param_names = WeightCache()
        self.counters: dict[tuple[str, str], int] = {}
        self.manifest: list[dict[str, Any]] = []
        self.seen: set[tuple] = set()
        self.done_reported: set[int] = set()
        self.rejected_quantized: dict[tuple[str, str], int] = {}


_S = _State()


def _tanh_dgelu(grad: Tensor, pre: Tensor) -> Tensor:
    """Gradient of the tanh-approximation GELU w.r.t. its input, in grad's dtype."""
    x = pre.to(torch.float32)
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)
    tanh = torch.tanh(inner)
    dinner = _SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x ** 2)
    dgelu = 0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh ** 2) * dinner
    return (grad.to(torch.float32) * dgelu).to(grad.dtype)


def is_high_precision(t: object) -> bool:
    """Whether t is a tensor that has not already been quantized.

    MXFP8Tensor/NVFP4Tensor subclass torch.Tensor, so isinstance alone is not enough.
    """
    return isinstance(t, Tensor) and not isinstance(t, QuantizedTensor)


def _subsample_rows(t: Tensor, rows: int, chunk: int = 32) -> Tensor:
    """Take up to rows rows of a 2-D tensor as chunk-aligned chunks spread over it."""
    total = t.shape[0]
    if total <= rows:
        return t.contiguous()
    n_chunks = max(1, rows // chunk)
    stride = total // n_chunks
    starts = [min((i * stride) // chunk * chunk, total - chunk)
              for i in range(n_chunks)]
    return torch.cat([t[s:s + chunk] for s in starts], dim=0).contiguous()


def _flatten_2d(t: Tensor) -> Tensor:
    """Collapse everything but the last dim into a 2-D view."""
    return t.reshape(-1, t.shape[-1])


def _site_key(frame_lineno: int, fn_name: str) -> str:
    """Stable call-site identifier "<fn>@<line>"."""
    return f"{fn_name}@{frame_lineno}"


def _record(tensor: Tensor, role: str, gemm_class: str, site: str,
            phase: str) -> None:
    """Persist one operand if this step and block are in the capture set."""
    counter_key = (site, role)
    occurrence = _S.counters.get(counter_key, 0)
    _S.counters[counter_key] = occurrence + 1
    if _S.blocks and occurrence not in _S.blocks:
        return
    dedup = (_S.step, gemm_class, role, occurrence)
    if dedup in _S.seen:
        return
    _S.seen.add(dedup)

    flat = _flatten_2d(tensor.detach())
    amax_full = flat.abs().max().item()
    # Subsample before widening to avoid a full fp32 copy of the tensor.
    sub = _subsample_rows(flat, _S.rows).to(torch.bfloat16)
    name = (f"s{_S.step:04d}_{gemm_class.replace('|', '+')}"
            f"_{role}_b{occurrence:03d}.pt")
    torch.save(sub.cpu(), os.path.join(_S.out_dir, name))
    _S.manifest.append({
        "file": name,
        "step": _S.step,
        "gemm_class": gemm_class,
        "role": role,
        "phase": phase,
        "occurrence": occurrence,
        "site": site,
        "full_shape": list(flat.shape),
        "kept_shape": list(sub.shape),
        "amax_full": amax_full,
        "amax_kept": sub.float().abs().max().item(),
    })


def _record_if_reference(t: object, role: str, gemm_class: str, site: str,
                         phase: str) -> None:
    """Record t only if it is a high-precision reference.

    None is skipped silently; an already-quantized tensor is skipped and counted.
    """
    if t is None:
        return
    if not is_high_precision(t):
        _S.rejected_quantized[(site, role)] = (
            _S.rejected_quantized.get((site, role), 0) + 1
        )
        return
    _record(t, role, gemm_class, site, phase)


def _resolve_class(weight: Tensor) -> str:
    """Look up a weight's GEMM class; "unknown@<ptr>" if unregistered."""
    cls = _S.param_names.get(weight)
    if cls is None:
        return f"unknown@{weight.data_ptr()}"
    return str(cls)


def _caller_line() -> int:
    """Line number of the flat/model.py frame that called the seam, or -1."""
    frame = inspect.currentframe()
    for _ in range(6):
        frame = frame.f_back
        if frame is None:
            return -1
        if frame.f_code.co_filename.endswith("flat/model.py"):
            return frame.f_lineno
    return -1


def _wrap_linear(fn: Callable, act_arg: int | None, weight_arg: int,
                 grad_arg: int | None, phase: str,
                 dgelu_pre_arg: int | None = None,
                 gate_arg: int | None = None, feeder: bool = False) -> Callable:
    """Wrap a seam function that takes a weight, so its class is resolvable.

    *_arg are positional indices into the call; feeder=True means the weights
    from weight_arg on are the consumers and no weight is recorded here.
    """

    def wrapper(*args, **kwargs):
        if _S.enabled and _S.step in _S.steps:
            if feeder:
                cls = "|".join(_resolve_class(w) for w in args[weight_arg:])
            else:
                cls = _resolve_class(args[weight_arg])
            site = _site_key(_caller_line(), fn.__name__)
            if not feeder:
                _record(args[weight_arg], ROLE_WEIGHT, cls, site, phase)
            if act_arg is not None:
                _record_if_reference(args[act_arg], ROLE_ACT, cls, site,
                                     phase)
            if grad_arg is not None:
                g = args[grad_arg]
                if is_high_precision(g) and dgelu_pre_arg is not None:
                    g = _tanh_dgelu(g, args[dgelu_pre_arg])
                if is_high_precision(g) and gate_arg is not None:
                    g = g * args[gate_arg]
                _record_if_reference(g, ROLE_GRAD, cls, site, phase)
        return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _wrap_quantize_only(fn: Callable, transform: str) -> Callable:
    """Wrap a quantize-only helper, keyed by _LINE_SITES.

    transform: "none", "gelu" (operand = gelu(arg0)) or "dgelu" (operand = dgelu(arg0, arg1)).
    """

    def wrapper(*args, **kwargs):
        if _S.enabled and _S.step in _S.steps:
            line = _caller_line()
            entry = _LINE_SITES.get(line)
            if entry is not None:
                role, cls, _ = entry
                if transform == "gelu":
                    operand = torch.nn.functional.gelu(
                        args[0], approximate="tanh",
                    )
                elif transform == "dgelu":
                    operand = _tanh_dgelu(args[0], args[1])
                else:
                    operand = args[0]
                phase = "fwd" if role == ROLE_ACT else "bwd"
                _record(operand, role, cls, _site_key(line, fn.__name__),
                        phase)
        return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


#: Seam functions that take a weight: (act index, weight index, grad index,
#: phase, pre-GELU index[, gate index, feeder]).
_LINEAR_SPECS: dict[str, tuple] = {
    # Gate-backward feeder: grad operand is grad * gate, shared by every
    # consuming weight from index 3 on.
    "fp8_gate_bwd_quantize": (None, 3, 0, "bwd", None, 1, True),
    "fp8_linear_fwd": (0, 1, None, "fwd", None),
    "fp8_linear_fwd_save": (0, 1, None, "fwd", None),
    "fp8_linear_fwd_gelu": (0, 1, None, "fwd", None),
    "fp8_linear_fwd_preq": (None, 1, None, "fwd", None),
    "fp8_linear_bwd": (1, 2, 0, "bwd", None),
    "fp8_linear_bwd_dgrad": (1, 2, 0, "bwd", None),
    "fp8_linear_bwd_dgelu": (2, 3, 0, "bwd", 1),
    "fp8_linear_bwd_dgrad_dgelu": (2, 3, 0, "bwd", 1),
    "fp8_linear_bwd_dgrad_preq_grad": (None, 2, None, "bwd", None),
    "fp8_linear_bwd_preq_both": (None, 2, None, "bwd", None),
    "fp8_linear_bwd_preq_grad": (1, 2, None, "bwd", None),
}

#: Quantize-only seam functions and the transform from arg 0 to the operand.
_QUANTIZE_SPECS: dict[str, str] = {
    "fp8_quantize_fwd": "none",
    "fp8_quantize_bwd": "none",
    "fp8_gelu_fwd": "gelu",
    "fp8_qgelu_fwd": "gelu",
    "fp8_bgrad_quantize": "none",
    "fp8_dbias_dqgelu_bwd": "dgelu",
}


def _verify_line_table(model_py: str) -> None:
    """Raise RuntimeError if flat/model.py has drifted from _LINE_SITES."""
    with open(model_py, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for lineno, (_, _, fragment) in _LINE_SITES.items():
        if lineno > len(lines) or fragment not in lines[lineno - 1]:
            raise RuntimeError(
                f"flat/quant_capture.py line table is stale: expected "
                f"{fragment!r} at {model_py}:{lineno}",
            )


def build_param_map(model: torch.nn.Module) -> WeightCache:
    """Map each GEMM weight to its class name, keyed on the weight's identity."""
    suffix_map = {
        "img_attn.qkv.weight": "dsb.img.qkv",
        "txt_attn.qkv.weight": "dsb.txt.qkv",
        "img_attn.proj.weight": "dsb.img.proj",
        "txt_attn.proj.weight": "dsb.txt.proj",
        "img_mlp.0.weight": "dsb.img.fc1",
        "txt_mlp.0.weight": "dsb.txt.fc1",
        "img_mlp.2.weight": "dsb.img.fc2",
        "txt_mlp.2.weight": "dsb.txt.fc2",
        "linear1_qkv.weight": "ssb.qkv",
        "linear1_mlp.weight": "ssb.fc1",
        "linear2_attn.weight": "ssb.proj",
        "linear2_mlp.weight": "ssb.fc2",
    }
    pairs = []
    for name, param in model.named_parameters():
        for suffix, cls in suffix_map.items():
            if name.endswith(suffix):
                pairs.append((param, cls))
                break
    return WeightCache.of(pairs)


def install_capture(model: torch.nn.Module) -> bool:
    """Patch the flat.model seam bindings to capture GEMM operands.

    Configured by the FLAT_CAPTURE_* environment variables; rank 0 only.
    Returns True if installed.
    """
    out_dir = os.environ.get("FLAT_CAPTURE_DIR", "")
    if not out_dir or os.environ.get("RANK", "0") != "0":
        return False

    from flat import model as flat_model

    _verify_line_table(flat_model.__file__)
    os.makedirs(out_dir, exist_ok=True)

    _S.enabled = True
    _S.out_dir = out_dir
    _S.steps = {
        int(v) for v in
        os.environ.get("FLAT_CAPTURE_STEPS", "3,10,20,35").split(",") if v
    }
    _S.blocks = {
        int(v) for v in
        os.environ.get("FLAT_CAPTURE_BLOCKS", "0,9,18").split(",") if v
    }
    _S.rows = int(os.environ.get("FLAT_CAPTURE_ROWS", "512"))
    _S.param_names = build_param_map(model)

    import flat.fp8_ops as fp8_ops

    for name, spec in _LINEAR_SPECS.items():
        wrapped = _wrap_linear(getattr(fp8_ops, name), *spec)
        setattr(flat_model, name, wrapped)
    for name, transform in _QUANTIZE_SPECS.items():
        if hasattr(flat_model, name):
            setattr(flat_model, name,
                    _wrap_quantize_only(getattr(fp8_ops, name), transform))
    return True


def begin_step(step: int) -> None:
    """Announce the current global_step and reset per-step counters."""
    _S.step = step
    _S.counters.clear()


def finish_capture() -> str:
    """Write manifest.json and return its path, or "" if capture was never installed."""
    if not _S.enabled:
        return ""
    path = os.path.join(_S.out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "steps": sorted(_S.steps),
            "blocks": sorted(_S.blocks),
            "rows": _S.rows,
            "rejected_quantized": {
                f"{site}:{role}": n
                for (site, role), n in _S.rejected_quantized.items()
            },
            "records": _S.manifest,
        }, fh, indent=1)
    return path


def capture_done() -> bool:
    """Whether the step that just ran wrote any operands; True once per capture step."""
    if not (_S.enabled and _S.step in _S.steps):
        return False
    if _S.step in _S.done_reported:
        return False
    _S.done_reported.add(_S.step)
    return True
