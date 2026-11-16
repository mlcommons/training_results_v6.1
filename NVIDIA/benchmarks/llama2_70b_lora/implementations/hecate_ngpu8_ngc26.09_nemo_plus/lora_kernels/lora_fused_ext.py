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

"""Load the fused LoRA branch extension: import the prebuilt module, or build it."""

import importlib
import os
import sys

import torch  # noqa: F401  (loads the libraries that the extension needs)
from torch.utils.cpp_extension import load

_NAME = "lora_fused_cuda"
_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_fused.cu")
_BUILD_DIR = os.environ.get("LORA_FUSED_BUILD_DIR", "/tmp/lora_fused_build")
# Same optimization flags as TE (no fast math). The FP4 conversion needs the "a" targets.
_FLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_100a,code=sm_100a",
    "-gencode=arch=compute_103a,code=sm_103a",
    "-gencode=arch=compute_107a,code=sm_107a",
]
_state = {"module": None}


def get_extension() -> object:
    """Import the build in LORA_FUSED_BUILD_DIR, or build it there."""
    if _state["module"] is not None:
        return _state["module"]
    if os.path.exists(os.path.join(_BUILD_DIR, f"{_NAME}.so")):
        sys.path.insert(0, _BUILD_DIR)
        _state["module"] = importlib.import_module(_NAME)
    else:
        os.makedirs(_BUILD_DIR, exist_ok=True)
        _state["module"] = load(name=_NAME, sources=[_SOURCE], extra_cuda_cflags=_FLAGS,
                                build_directory=_BUILD_DIR, verbose=False)
    return _state["module"]
