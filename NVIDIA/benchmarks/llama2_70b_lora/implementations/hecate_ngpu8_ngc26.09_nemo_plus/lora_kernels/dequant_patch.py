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

"""FP8 dequantize to bf16 with amax partials, in place of tex.dequantize for plain FP8 tensors.

The attention output O comes out of cuDNN in FP8 E4M3 with one scale. TE dequantizes it to
bf16 (unary_kernel), and the proj input quantize then runs an amax pass over the bf16 tensor.
Here one kernel does the dequantize with the TE math and writes the amax partials. The patch
registers the partials in the side channel for the quantize.
"""

import torch
import transformer_engine.pytorch  # noqa: F401  (loads transformer_engine_torch)
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import (
    Float8TensorStorage,
)

import amax_channel
import amax_prod_ext

_FP8_TYPES = {tex.DType.kFloat8E4M3: False, tex.DType.kFloat8E5M2: True}
_state = {"original": None}


def _plain_fp8(tensor: Float8TensorStorage) -> bool:
    """Return True for a contiguous, aligned FP8 tensor with one scale."""
    data = tensor._data
    if data is None or not data.is_contiguous() or data.data_ptr() % 16 != 0:
        return False
    scale_inv = tensor._scale_inv
    if scale_inv is None or scale_inv.numel() != 1 or scale_inv.dtype != torch.float32:
        return False
    return data.numel() % 16 == 0 and tensor._fp8_dtype in _FP8_TYPES


def fast_dequantize(*args: object) -> torch.Tensor:
    """Same signature as tex.dequantize(tensor, otype)."""
    tensor, otype = args[0], args[1]
    if otype == tex.DType.kBFloat16 and isinstance(tensor, Float8TensorStorage) and _plain_fp8(tensor):
        e5m2 = _FP8_TYPES[tensor._fp8_dtype]
        out, partials = amax_prod_ext.get_module().fp8_dequantize_partials(
            tensor._data, tensor._scale_inv, e5m2)
        if amax_channel.active():
            amax_channel.register(out, partials)
        return out
    return _state["original"](*args)


def apply() -> None:
    """Replace tex.dequantize."""
    if _state["original"] is not None:
        return
    amax_prod_ext.get_module()
    _state["original"] = tex.dequantize
    tex.dequantize = fast_dequantize
