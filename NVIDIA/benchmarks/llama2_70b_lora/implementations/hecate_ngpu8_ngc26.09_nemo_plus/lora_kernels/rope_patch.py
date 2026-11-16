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

"""RoPE backward with amax partials, in place of the fp8_rope backward kernel.

fp8_rope._FP8QKVRoPE.backward dequantizes the FP8 dQ, dK and dV and does the RoPE backward in
one kernel. Here the kernel also writes the amax partials of the bf16 dqkv output. The patch
registers them for the stochastic-rounding quantize of the QKV dgrad.
"""

import importlib

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import (
    Float8TensorStorage,
)

import amax_channel
import amax_prod_ext

_state = {"forward": None, "backward": None, "cls": None}


def _forward(ctx: object, *args: object) -> tuple:
    """The fp8_rope forward, plus a note whether the channel is active for this layer."""
    result = _state["forward"](ctx, *args)
    ctx.amax_prod_active = amax_channel.active()
    return result


def _backward(ctx: object, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor) -> tuple:
    """Return the qkv grad with registered partials, or the fp8_rope result."""
    grads = (dq, dk, dv)
    if not all(isinstance(grad, Float8TensorStorage) for grad in grads):
        return _state["backward"](ctx, dq, dk, dv)
    data = [grad._data.contiguous() for grad in grads]
    if not all(item.data_ptr() % 16 == 0 for item in data):
        return _state["backward"](ctx, dq, dk, dv)
    q_freqs, k_freqs = ctx.saved_tensors
    e5m2 = dq._fp8_dtype == tex.DType.kFloat8E5M2
    dqkv, partials = amax_prod_ext.get_module().rope_backward_fp8_partials(
        *data, dq._scale_inv, q_freqs, k_freqs, ctx.split, ctx.tensor_format == "sbhd", e5m2,
    )
    if getattr(ctx, "amax_prod_active", False):
        amax_channel.register(dqkv, partials)
    return dqkv, None, None, None


def apply() -> bool:
    """Install the partials backward in fp8_rope. Return False if fp8_rope is not in use."""
    if _state["cls"] is not None:
        return True
    try:
        fp8_rope = importlib.import_module("fp8_rope")
    except ImportError:
        return False
    if fp8_rope._state.get("rope") is None:
        return False
    amax_prod_ext.get_module()
    cls = fp8_rope._FP8QKVRoPE
    _state["forward"] = cls.forward
    _state["backward"] = cls.backward
    cls.forward = staticmethod(_forward)
    cls.backward = staticmethod(_backward)
    _state["cls"] = cls
    return True
