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

"""Load the lora_attn CUDA extension: import the prebuilt module, or build it one time.

lora_attn: one module with the tuned kernels and the amax producer kernels.
"""

import importlib
import os
import sys

import torch  # noqa: F401  (loads the libraries that the extension needs)
from torch.utils.cpp_extension import load

_NAME = "lora_attn"
_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_attn.cu")
_BUILD_DIR = os.environ.get("LORA_ATTN_BUILD_DIR", "/tmp/lora_attn_build")
# Same optimization flags as TE (no fast math). The 256-bit loads, the FP4 conversion and the
# bulk copies need the "a" targets of sm_100 or later.
_FLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_100a,code=sm_100a",
    "-gencode=arch=compute_103a,code=sm_103a",
    "-gencode=arch=compute_107a,code=sm_107a",
]

_state = {"module": None}


def cuda_flags() -> list:
    """Return the nvcc flags."""
    return list(_FLAGS)


def get_module() -> object:
    """Return the extension module."""
    if _state["module"] is not None:
        return _state["module"]
    if os.path.exists(os.path.join(_BUILD_DIR, f"{_NAME}.so")):
        sys.path.insert(0, _BUILD_DIR)
        _state["module"] = importlib.import_module(_NAME)
    else:
        os.makedirs(_BUILD_DIR, exist_ok=True)
        _state["module"] = load(name=_NAME, sources=[_SOURCE], extra_cuda_cflags=cuda_flags(),
                                build_directory=_BUILD_DIR, verbose=False)
    return _state["module"]
