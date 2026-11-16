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

"""Faster fused QKV RoPE for the LoRA benchmark, with the same math as TE.

Call apply() before the model runs. It replaces tex.fused_qkv_rope_forward and
tex.fused_qkv_rope_backward. Cases that the fast kernel does not support go to TE.
"""

import importlib
import os
import sys

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine_torch as tex
from torch.utils.cpp_extension import load

_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_rope.cu")
_BUILD_DIR = os.environ.get("LORA_FAST_ROPE_BUILD_DIR", "/tmp/lora_fast_rope_build")
_CUDA_FLAGS = ["-O3", "--expt-relaxed-constexpr"]
_SUPPORTED_FORMATS = (tex.NVTE_QKV_Format.NVTE_SBHD, tex.NVTE_QKV_Format.NVTE_BSHD)

_state = {"ext": None, "forward": None, "backward": None}


def get_extension() -> object:
    """Import the prebuilt extension in _BUILD_DIR, or build it there."""
    if _state["ext"] is None and os.path.exists(os.path.join(_BUILD_DIR, "lora_fast_rope.so")):
        sys.path.insert(0, _BUILD_DIR)
        _state["ext"] = importlib.import_module("lora_fast_rope")
    if _state["ext"] is None:
        os.makedirs(_BUILD_DIR, exist_ok=True)
        _state["ext"] = load(
            name="lora_fast_rope",
            sources=[_SOURCE],
            extra_cuda_cflags=_CUDA_FLAGS,
            build_directory=_BUILD_DIR,
            verbose=False,
        )
    return _state["ext"]


def _options_supported(split: list, options: tuple) -> bool:
    """Return True if the layout options are the ones that the fast kernel handles."""
    qkv_format, interleaved, cp_size, start_positions = options
    simple = not interleaved and cp_size == 1 and start_positions is None
    layout = qkv_format in _SUPPORTED_FORMATS and len(split) == 3
    return simple and layout and split[1] == 128 and split[2] == 128 and split[0] % 128 == 0


def _tensors_supported(tensors: list, freqs: list) -> bool:
    """Return True if all tensors have the dtype and shape that the fast kernel handles."""
    for tensor in tensors:
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() or tensor.dim() != 4:
            return False
    for freq in freqs:
        if freq.dtype != torch.float32 or not freq.is_contiguous() or freq.size(3) != 128:
            return False
    return True


def _is_supported(tensors: list, freqs: list, split: list, options: tuple) -> bool:
    """Return True if the fast kernel gives the same result as TE for these inputs."""
    return _options_supported(split, options) and _tensors_supported(tensors, freqs)


def fast_forward(*args: object) -> tuple:
    """Same signature as tex.fused_qkv_rope_forward."""
    qkv, q_freqs, k_freqs, start_positions, split, qkv_format, interleaved, cp_size, _ = args
    options = (qkv_format, interleaved, cp_size, start_positions)
    if not _is_supported([qkv], [q_freqs, k_freqs], list(split), options):
        return _state["forward"](*args)
    sbhd = qkv_format == tex.NVTE_QKV_Format.NVTE_SBHD
    q_out, k_out, v_out = get_extension().forward(qkv, q_freqs, k_freqs, list(split), sbhd)
    return q_out, k_out, v_out


def fast_backward(*args: object) -> torch.Tensor:
    """Same signature as tex.fused_qkv_rope_backward."""
    dq, dk, dv, q_freqs, k_freqs, split, qkv_format, interleaved, cp_size, _ = args
    options = (qkv_format, interleaved, cp_size, None)
    if not _is_supported([dq, dk, dv], [q_freqs, k_freqs], list(split), options):
        return _state["backward"](*args)
    sbhd = qkv_format == tex.NVTE_QKV_Format.NVTE_SBHD
    return get_extension().backward(dq, dk, dv, q_freqs, k_freqs, list(split), sbhd)


def apply() -> None:
    """Replace the TE QKV RoPE functions with the fast versions."""
    if _state["forward"] is not None:
        return
    get_extension()
    _state["forward"] = tex.fused_qkv_rope_forward
    _state["backward"] = tex.fused_qkv_rope_backward
    tex.fused_qkv_rope_forward = fast_forward
    tex.fused_qkv_rope_backward = fast_backward
