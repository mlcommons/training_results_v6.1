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

"""Apply the fused amax producers (switch LORA_PATCH_AMAX_PRODUCERS).

Call apply() after the other kernel patches (nvfp4q, fp8_rope), before the model runs.
Call clear() at each train step start.
"""

import os

import amax_channel
import dequant_patch
import fast_quant_ap
import fork_patch
import rope_patch

_state = {"applied": None}


def apply() -> list:
    """Install the patches. Return the names of the installed parts."""
    if _state["applied"] is not None:
        return _state["applied"]
    names = []
    dequant_patch.apply()
    names.append("fp8_dequantize")
    fork_patch.apply()
    names.append("fork_accumulate")
    if rope_patch.apply():
        names.append("rope_backward")
    fast_quant_ap.apply()
    names.append("quantize_from_partials")
    _state["applied"] = names
    return names


def apply_if_enabled() -> list:
    """Install the patches if LORA_PATCH_AMAX_PRODUCERS=1."""
    if os.environ.get("LORA_PATCH_AMAX_PRODUCERS", "0") != "1":
        return []
    return apply()


def clear() -> None:
    """Remove the open side-channel entries (call it at each train step start)."""
    amax_channel.clear()


def stats() -> dict:
    """Return the side-channel counters and the number of Fork calls."""
    result = amax_channel.stats()
    result["forks"] = fork_patch._state["forks"]
    return result
