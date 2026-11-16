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

"""Side channel between a producer kernel and the NVFP4 quantize of its output tensor.

A producer kernel writes its bf16 output and the per-block maxima of |output| (partials). It
calls register(tensor, partials). The patched NVFP4 quantize calls take(tensor). If an entry
exists, the quantize kernel reduces the partials itself and no separate amax pass runs.

An entry keeps a reference to the tensor. Thus the allocator cannot free its memory and give
the address to another tensor while the entry exists. A key match therefore always means the
same bytes. Entries go away on take(), on clear() (call it at each train step start), and when
more than MAX_ENTRIES entries exist (the oldest goes away).
"""

from collections import OrderedDict

import torch
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

MAX_ENTRIES = 64
_state = {
    "entries": OrderedDict(),
    "enabled": True,
    "registered": 0,
    "hits": 0,
    "misses": 0,
    "dropped": 0,
}


def _key(tensor: torch.Tensor) -> tuple:
    return (tensor.data_ptr(), tensor.numel(), tensor.dtype, tensor.storage_offset(),
            tensor.device.index)


def nvfp4_recipe_active() -> bool:
    """Return True inside an autocast region with the NVFP4 recipe (the FP4 training phase)."""
    if not FP8GlobalStateManager.is_fp8_enabled():
        return False
    return isinstance(FP8GlobalStateManager.get_fp8_recipe(), NVFP4BlockScaling)


def active() -> bool:
    """Return True if producers should register partials now."""
    return _state["enabled"] and nvfp4_recipe_active()


def set_enabled(flag: bool) -> None:
    """Turn the channel on or off (off: producers register nothing, take() finds nothing)."""
    _state["enabled"] = bool(flag)
    if not flag:
        clear()


def register(tensor: torch.Tensor, partials: torch.Tensor) -> None:
    """Publish the partials of a producer output."""
    if not _state["enabled"]:
        return
    entries = _state["entries"]
    entries[_key(tensor)] = (tensor, partials)
    _state["registered"] += 1
    while len(entries) > MAX_ENTRIES:
        entries.popitem(last=False)
        _state["dropped"] += 1


def take(tensor: torch.Tensor) -> torch.Tensor | None:
    """Return the partials of this tensor and remove the entry, or None."""
    entry = _state["entries"].pop(_key(tensor), None)
    if entry is None:
        _state["misses"] += 1
        return None
    _state["hits"] += 1
    return entry[1]


def clear() -> None:
    """Remove all entries."""
    _state["entries"].clear()


def stats() -> dict:
    """Return the counters and the number of open entries."""
    return {
        "registered": _state["registered"],
        "hits": _state["hits"],
        "misses": _state["misses"],
        "dropped": _state["dropped"],
        "open": len(_state["entries"]),
    }
