# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

# Packaging metadata only; the training does not import or run this file.
# torch, fbgemm_gpu (ROCm build: fbgemm_gpu_nightly_rocm) and torchrec are not listed in
# install_requires: they come from the container (see Dockerfile and requirements.txt), and
# pip would otherwise replace them with CUDA or CPU wheels from PyPI.

import os

from setuptools import find_packages, setup

_HERE = os.path.dirname(os.path.abspath(__file__))
_README = next(
    (
        os.path.join(_HERE, n)
        for n in ("README.md", "README.MD")
        if os.path.isfile(os.path.join(_HERE, n))
    ),
    None,
)
if _README:
    with open(_README, encoding="utf-8") as _f:
        _LONG_DESCRIPTION = _f.read()
else:
    _LONG_DESCRIPTION = ""

setup(
    name="generative_recommenders",
    version="0.1.0",
    description="Library for generative recommendation algorithms.",
    packages=find_packages(exclude=["configs"]),
    python_requires=">=3.10",
    install_requires=[
        "gin_config>=0.5.0",
        "pandas>=2.2.0",
        "tensorboard>=2.19.0",
        "pybind11",
        "click",
        "matplotlib",
    ],
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/meta-recsys/generative-recommenders",
    license="Apache-2.0",
)
