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

"""Load the SwiGLU kernels that give the same bits as TE, with the amax.

forward(x) returns (y, amax) and backward(dy, x) returns (dx, amax). The amax is the
max of |value| over the fp32 values before the cast to bf16, as in TE. Call init()
before CUDA graph capture.
forward_fp8(x, quantizer) and backward_fp8(dy, x, quantizer) give the Float8 tensor of
the TE delayed-scaling path (FP8 healing phase): an atomic max into quantizer.amax and
scale_inv = 1 / scale.
"""

import importlib
import os
import sys

import torch
from torch.utils.cpp_extension import load

_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swiglu_fused.cu")
_BUILD_DIR = os.environ.get("LORA_SWIGLU_BUILD_DIR", "/tmp/lora_swiglu_fused_build")
_NAME = "lora_swiglu_fused"
_BEST_FORWARD = int(os.environ.get("LORA_SWIGLU_FWD_VARIANT", "27"))
_BEST_BACKWARD = int(os.environ.get("LORA_SWIGLU_BWD_VARIANT", "24"))
_BEST_FORWARD_FP8 = int(os.environ.get("LORA_SWIGLU_FWD_FP8_VARIANT", "27"))
_BEST_BACKWARD_FP8 = int(os.environ.get("LORA_SWIGLU_BWD_FP8_VARIANT", "31"))
_state = {"module": None}


def _cuda_flags() -> list:
    """Return the nvcc flags. The arch flag stops the default arch flags of torch."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    return ["-O3", "--expt-relaxed-constexpr", f"-gencode=arch=compute_{arch},code=sm_{arch}"]


def get_module() -> object:
    """Import the build in the build folder, or build it."""
    if _state["module"] is None:
        if os.path.exists(os.path.join(_BUILD_DIR, f"{_NAME}.so")):
            sys.path.insert(0, _BUILD_DIR)
            _state["module"] = importlib.import_module(_NAME)
        else:
            os.makedirs(_BUILD_DIR, exist_ok=True)
            _state["module"] = load(name=_NAME, sources=[_SOURCE], extra_cuda_cflags=_cuda_flags(),
                                    build_directory=_BUILD_DIR, verbose=False)
    return _state["module"]


def init() -> None:
    """Make the amax scratch of the current device. Do this before graph capture."""
    get_module().init_workspace()


def forward(x: torch.Tensor, variant: int = _BEST_FORWARD) -> tuple:
    """Return (y, amax) for x = [..., 2 * cols] (act, then gate)."""
    y, amax = get_module().forward(x, variant)
    return y, amax


def backward(dy: torch.Tensor, x: torch.Tensor, variant: int = _BEST_BACKWARD) -> tuple:
    """Return (dx, amax) for dy = [..., cols] and x = [..., 2 * cols]."""
    dx, amax = get_module().backward(dy, x, variant)
    return dx, amax


def _fp8_dtype_flag(quantizer: object) -> bool:
    """Return True for E5M2, False for E4M3."""
    import transformer_engine_torch as tex

    if quantizer.dtype == tex.DType.kFloat8E5M2:
        return True
    if quantizer.dtype == tex.DType.kFloat8E4M3:
        return False
    raise ValueError(f"unsupported FP8 dtype {quantizer.dtype}")


def forward_fp8(x: torch.Tensor, quantizer: object, variant: int = _BEST_FORWARD_FP8) -> object:
    """Return the Float8 tensor of SwiGLU(x), as TE makes it with a delayed-scaling quantizer."""
    shape = list(x.shape)
    shape[-1] //= 2
    out = quantizer.make_empty(shape, dtype=x.dtype, device=x.device)
    get_module().forward_fp8(x, quantizer.scale, quantizer.amax, out._scale_inv, out._data,
                             _fp8_dtype_flag(quantizer), variant)
    return out


def backward_fp8(dy: torch.Tensor, x: torch.Tensor, quantizer: object,
                 variant: int = _BEST_BACKWARD_FP8) -> object:
    """Return the Float8 tensor of dSwiGLU(dy, x), as TE makes it with a delayed-scaling quantizer."""
    out = quantizer.make_empty(list(x.shape), dtype=x.dtype, device=x.device)
    get_module().backward_fp8(dy, x, quantizer.scale, quantizer.amax, out._scale_inv, out._data,
                              _fp8_dtype_flag(quantizer), variant)
    return out
