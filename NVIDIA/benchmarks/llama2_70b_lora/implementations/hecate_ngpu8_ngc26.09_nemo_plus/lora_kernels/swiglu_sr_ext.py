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

"""Build or load the swiglu_sr CUDA extension (two-pass dSwiGLU + NVFP4 quantize)."""

import importlib
import os
import sys

import torch
from torch.utils.cpp_extension import load

_NAME = "swiglu_sr_cuda"
_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swiglu_sr.cu")
_BUILD_DIR = os.environ.get("SWIGLU_SR_BUILD_DIR", "/tmp/swiglu_sr_build")
_ALL_ARCHS = ("100a", "103a", "107a")
_state = {"module": None}


def _cuda_flags() -> list:
    """Return the nvcc flags: the TE optimization flags (no fast math) and the "a" targets.

    SWIGLU_SR_ALL_ARCHS=1 builds for sm_100a, sm_103a and sm_107a. Else the build is only for
    the arch of the current GPU. An arch flag stops the default arch flags of torch.
    """
    if os.environ.get("SWIGLU_SR_ALL_ARCHS", "0") == "1":
        archs = _ALL_ARCHS
    else:
        major, minor = torch.cuda.get_device_capability()
        archs = (f"{major}{minor}a",)
    gencode = [f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in archs]
    return ["-O3", "--expt-relaxed-constexpr", *gencode]


def get_extension() -> object:
    """Import the build in SWIGLU_SR_BUILD_DIR, or build it there."""
    if _state["module"] is None:
        if os.path.exists(os.path.join(_BUILD_DIR, f"{_NAME}.so")):
            sys.path.insert(0, _BUILD_DIR)
            _state["module"] = importlib.import_module(_NAME)
        else:
            os.makedirs(_BUILD_DIR, exist_ok=True)
            _state["module"] = load(name=_NAME, sources=[_SOURCE], extra_cuda_cflags=_cuda_flags(),
                                    build_directory=_BUILD_DIR, verbose=False)
    return _state["module"]
