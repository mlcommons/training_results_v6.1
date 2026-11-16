# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.09-py3
FROM ${FROM_IMAGE_NAME}


ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

WORKDIR /workspace/ft-llm

## 1. Megatron-Bridge and Megatron-Core
ARG MBRIDGE_REVISION=v0.6.0
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge && \
    cd /workspace/Megatron-Bridge && \
    git checkout ${MBRIDGE_REVISION} && \
    git submodule update --init --recursive && \
    echo MBRIDGE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MBRIDGE_COMMIT_HASH.env && \
    # Carry NVIDIA-NeMo/Megatron-Bridge#6342 (merged on main: no pre-quantize of the LoRA branch input) until the pin has it.
    git fetch origin 34ddd53b0e023d93f107f29b7dc61f9c5dd76a57 && git cherry-pick -x --keep-redundant-commits FETCH_HEAD && \
    cd /workspace/Megatron-Bridge/3rdparty/Megatron-LM && \
    # Carry the MCore plain residual add (bias None, dropout 0, same bits) until it is upstream.
    git fetch https://github.com/michal2409/Megatron-LM.git 778a6643c92167575b8a67ac33859438b0a9513b && \
    git cherry-pick -x --keep-redundant-commits FETCH_HEAD && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install --no-build-isolation -e .

ENV PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"

## 2. Transformer Engine
# The LoRA NVFP4 kernels (lora_kernels/) copy the math of TE v2.20.0. The base image has TE 2.19.
# Keep the default Philox rounds and no activation fast math, because the kernels copy that math.
ARG TE_REVISION=v2.20.0
ENV CUSTOM_TE_REVISION=${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    export MAX_JOBS=$(( $(nproc) > 24 ? 24 : $(nproc) )); \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git /workspace/transformerengine && \
    cd /workspace/transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    pip uninstall -y flash-attn-4 quack-kernels && \
    git submodule update --init --recursive && \
    pip install "nvidia-cudnn-frontend==1.29.0" && \
    NVTE_CUDA_ARCHS="100a;103a;107a" NVTE_WITH_NCCL_EP=0 NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi

ADD . /workspace/ft-llm
RUN apt-get remove -y python3-yaml && pip uninstall -y transformers torchao && pip install --no-build-isolation -r requirements.txt

## LoRA NVFP4 kernels: apply the TE patches and prebuild the CUDA extensions
RUN bash te_patches/apply.sh && python lora_kernels/build.py
