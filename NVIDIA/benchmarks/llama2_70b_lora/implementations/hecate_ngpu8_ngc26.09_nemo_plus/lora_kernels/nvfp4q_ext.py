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

"""Build or load the nvfp4q CUDA extension (fast NVFP4 quantize, same result as TE)."""

import importlib
import os
import sys

from torch.utils.cpp_extension import load

_NAME = "nvfp4q_cuda"
_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvfp4q.cu")
_BUILD_DIR = os.environ.get("NVFP4Q_BUILD_DIR", "/tmp/nvfp4q_build")
# Same optimization flags as TE (no fast math). The FP4 PTX needs the "a" targets.
_FLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    "-gencode=arch=compute_100a,code=sm_100a",
    "-gencode=arch=compute_103a,code=sm_103a",
    "-gencode=arch=compute_107a,code=sm_107a",
]
_state = {"module": None}


def get_extension() -> object:
    """Import the build in NVFP4Q_BUILD_DIR, or build it there."""
    if _state["module"] is None:
        if os.path.exists(os.path.join(_BUILD_DIR, f"{_NAME}.so")):
            sys.path.insert(0, _BUILD_DIR)
            _state["module"] = importlib.import_module(_NAME)
        else:
            os.makedirs(_BUILD_DIR, exist_ok=True)
            _state["module"] = load(
                name=_NAME,
                sources=[_SOURCE],
                extra_cuda_cflags=_FLAGS,
                build_directory=_BUILD_DIR,
                verbose=False,
            )
    return _state["module"]
