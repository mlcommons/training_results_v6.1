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

# Azure packaging: pinned dependencies and offline model-config cache.
# Supply the aarch64 NVIDIA PyTorch 26.04 base image by immutable digest.
ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

ARG FROM_IMAGE_NAME
RUN printf '%s\n' "${FROM_IMAGE_NAME}" | grep -Eq '^nvcr.io/nvidia/pytorch@sha256:[0-9a-f]{64}$'

ENV PIP_CONSTRAINT=""

WORKDIR /workspace/

RUN pip install numcodecs==0.16.3
RUN pip install nvidia-mathdx==25.1.1

## 1. Megatron-Bridge and Megatron-Core
ARG MBRIDGE_REVISION=dabf51d9d5e79b093e321a5c6a32f2a7fb39ecc7
ARG MCORE_REVISION=69c486825c893c07e1fad93a6603f7243bb311fe
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge && \
    cd /workspace/Megatron-Bridge && \
    git checkout ${MBRIDGE_REVISION} && \
    git submodule update --init --recursive && \
    echo MBRIDGE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MBRIDGE_COMMIT_HASH.env && \
    cd /workspace/Megatron-Bridge/3rdparty/Megatron-LM && \
    git checkout ${MCORE_REVISION} && \
    git submodule update --init --recursive && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install -e .

ENV PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"

## 2. Transformer Engine
ARG TE_REVISION=4220403e831d29e93868f7793693ea83f6b8b05b
ENV CUSTOM_TE_REVISION=${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    export MAX_JOBS=$(( $(nproc) > 24 ? 24 : $(nproc) )); \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
    cd transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule update --init --recursive && \
    NVTE_BUILD_NUM_PHILOX_ROUNDS=3 NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 NVTE_CUDA_ARCHS="100a;103a" NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi

## 3. HybridEP
ARG HYBRID_EP_REVISION=f725d29699f5bda9ba789456bb9579af69844685
RUN git clone https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && \
    git checkout ${HYBRID_EP_REVISION} && \
    TORCH_CUDA_ARCH_LIST="10.0;10.3" pip install --no-build-isolation .

## 4. cuDNN frontend
ARG CUDNN_FRONTEND_REVISION=e46d7082450264ce05cf898f8740011c4896f817
RUN if [ "${CUDNN_FRONTEND_REVISION}" != SKIP ]; then \
    pip uninstall -y cudnn && \
    pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 && \
    git clone https://github.com/NVIDIA/cudnn-frontend.git cudnn-frontend && \
    cd cudnn-frontend && \
    git checkout ${CUDNN_FRONTEND_REVISION} && \
    echo CUDNN_FRONTEND_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /CUDNN_FRONTEND_COMMIT_HASH.env && \
    git submodule update --init --recursive && \
    pip install .[cutedsl] \
    ; fi

## 5. Benchmark dependencies
COPY requirements.txt .
RUN pip uninstall -y transformers torchao && pip install --no-cache-dir -r requirements.txt

## 6. Benchmark code
WORKDIR /workspace/llm
COPY . .
ENV PYTHONPATH="/workspace/llm:${PYTHONPATH}"

RUN HF_DIR=/workspace/llm/hf_hub/v1 bash data_scripts/download_hf_config.sh