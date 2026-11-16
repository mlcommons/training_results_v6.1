#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

ARG BASE_IMAGE=docker.io/rocm/primus:v25.11
FROM ${BASE_IMAGE}

WORKDIR /workspace/code/patches
COPY patches/primus_evaluator.patch .
COPY patches/megatron_validation_consumed_samples.patch .

WORKDIR /workspace/deps
# HPE: the reference checks out Primus `main`, which no longer matches the
# patches (primus_evaluator.patch was merged upstream on 2026-02-25 in 8c5bc42).
# Pin to its parent, where both patches' pre-image blobs match exactly:
#   evaluator.py      f7df2870  (Primus 9788c18)
#   data_samplers.py  1bf1bf5e  (Megatron-LM 8477817, the submodule at 9788c18)
ARG PRIMUS_COMMIT=9788c180d82a711d45e0954749e62c792450e2c7
RUN git clone --recursive https://github.com/AMD-AIG-AIMA/Primus.git && \
    cd Primus && \
    git checkout ${PRIMUS_COMMIT} && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    git apply /workspace/code/patches/primus_evaluator.patch 

RUN cd /workspace/deps/Primus/third_party/Megatron-LM && \
    git apply /workspace/code/patches/megatron_validation_consumed_samples.patch && \
    pip install -e . --no-deps

WORKDIR /workspace/code

COPY . .

# HPE: the reference names primus_mllog-0.1.0, but only 0.1.20 is shipped.
# primus_mllog needs mlperf_logging >= 6.0.0-rc3 (adds TENSOR_PARALLELISM);
# the reference installs none and inherits 4.1.50 from Primus, which crashes
# every rank at startup. Pin the same release as llama31_8b.
RUN pip install "git+https://github.com/mlcommons/logging.git@6.0.0-rc5" && \
    pip install primus_mllog-*-py3-none-any.whl && \
    python -c "from mlperf_logging.mllog import constants as c; assert c.TENSOR_PARALLELISM"
