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

"""Apply the nvfp4q quantize patch and the fused SwiGLU patch together.

- nvfp4q: fast NVFP4 quantize of activations, with GEMM-swizzled scales.
- swiglu: fused SwiGLU forward and backward with amax. Its NVFP4 quantize step uses the
  nvfp4q kernel with the SwiGLU amax.
"""

import torch

import nvfp4q_ext
import nvfp4q_patch
import swiglu_patch

_state = {"applied": False}


def quantize_with_amax(tensor: torch.Tensor, amax: torch.Tensor, quantizer: object) -> object:
    """Quantize the SwiGLU output with its amax, with the nvfp4q kernel if possible."""
    plain = nvfp4q_patch._plain_quantizer(quantizer) and nvfp4q_patch._plain_tensor(tensor)
    if not plain:
        return swiglu_patch.te_quantize(tensor, amax, quantizer)
    module = nvfp4q_ext.get_extension()
    data, scales = module.quantize_with_amax(tensor, amax, bool(quantizer.stochastic_rounding))
    # TE keeps the given amax tensor in the output, so do the same.
    return nvfp4q_patch._make_output(quantizer, tensor, [data, scales, amax])


def apply() -> None:
    """Patch the TE quantize, the te.ops RMSNorm quantize, and the te.ops SwiGLU op."""
    if _state["applied"]:
        return
    nvfp4q_patch.apply()
    swiglu_patch.apply(quantize_with_amax)
    _state["applied"] = True
