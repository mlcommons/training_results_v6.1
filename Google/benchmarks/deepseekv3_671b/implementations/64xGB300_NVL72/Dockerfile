# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG FROM_IMAGE_NAME=nvcr.io/nvdlfwea/ea_ext_containers/pytorch:26.09-py3
FROM ${FROM_IMAGE_NAME}

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

WORKDIR /workspace/

RUN pip install numcodecs==0.16.3
RUN pip install nvidia-mathdx==25.1.1

## 1. Megatron-Bridge and Megatron-Core
ARG MBRIDGE_REVISION=26.09-alpha.rc1
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge && \
    cd /workspace/Megatron-Bridge && \
    git checkout ${MBRIDGE_REVISION} && \
    git submodule update --init --recursive && \
    echo MBRIDGE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MBRIDGE_COMMIT_HASH.env && \
    cd /workspace/Megatron-Bridge/3rdparty/Megatron-LM && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install -e .

ENV PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"

## 2. cuDNN frontend
ARG CUDNN_FRONTEND_REVISION=v1.29.0
RUN if [ "${CUDNN_FRONTEND_REVISION}" != SKIP ]; then \
    pip uninstall -y cudnn && \
    pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base \
        nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13 && \
    git clone https://github.com/NVIDIA/cudnn-frontend.git cudnn-frontend && \
    cd cudnn-frontend && \
    git checkout ${CUDNN_FRONTEND_REVISION} && \
    echo CUDNN_FRONTEND_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /CUDNN_FRONTEND_COMMIT_HASH.env && \
    git submodule init && git submodule update && \
    pip install .[cutedsl] && \
    # CuTe DSL version with UE5M3 support; installed last so the cudnn-frontend
    # extra cannot swap it, and checked.
    pip install "nvidia-cutlass-dsl[cu13]==4.8.0.dev0" && \
    python3 -c "import importlib.metadata as m; v = m.version('nvidia-cutlass-dsl'); assert v == '4.8.0.dev0', v" \
    ; fi

## 3. Transformer Engine
ARG TE_REVISION=9ef901577c75ccd1cb78e2f42335c82984db67f2
ENV CUSTOM_TE_REVISION ${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    export MAX_JOBS=$(( $(nproc) > 24 ? 24 : $(nproc) )); \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
    cd transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule update --init --recursive && \
    NVTE_BUILD_NUM_PHILOX_ROUNDS=10 NVTE_CUDA_ARCHS="100a;103a;107a" NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi

## 4. HybridEP
RUN git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && \
    git checkout 58397eac493b075b0baea5861ac6e9793d8d7a21 && \
    echo DEEPEP_COMMIT_HASH=$(git rev-parse HEAD) && \
    TORCH_CUDA_ARCH_LIST="10.0;10.3;12.0" pip install --no-build-isolation . "pynvml==13.0.1"

## 5. Benchmark dependencies
COPY requirements.txt .
RUN apt-get remove -y python3-yaml && pip uninstall -y transformers torchao && pip install --no-cache-dir --no-build-isolation -r requirements.txt

## 6. Benchmark code
WORKDIR /workspace/llm
COPY . .
ENV PYTHONPATH="/workspace/llm:${PYTHONPATH}"
