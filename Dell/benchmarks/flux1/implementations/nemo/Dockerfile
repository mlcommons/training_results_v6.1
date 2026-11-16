# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.08-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/flux

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=${GIT_COMMIT_ID}
ENV PIP_CONSTRAINT=""

COPY requirements.txt .
RUN pip install --no-build-isolation -r requirements.txt

ARG DSL_VERSION=4.8.0.dev0
RUN if [ "${DSL_VERSION}" != SKIP ]; then \
    pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base \
        nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 \
        nvidia-cutlass-dsl-libs-cu13 && \
    pip install "nvidia-cutlass-dsl[cu13]==${DSL_VERSION}" && \
    python3 -c "import importlib.metadata as m; v = m.version('nvidia-cutlass-dsl'); assert v == '${DSL_VERSION}', v" \
    ; fi

## Transformer Engine -- rebuilt: the base image's TE 2.18 has an NVFP4 quantizer
## use-after-free that hits our backward cast. Do not SKIP.
ARG TE_REVISION=v2.20.0
ARG TE_REPO_URL=https://github.com/NVIDIA/TransformerEngine.git
ENV CUSTOM_TE_REVISION=${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    export MAX_JOBS=$(( $(nproc) > 24 ? 24 : $(nproc) )); \
    pip uninstall -y transformer-engine transformer_engine && \
    git clone "${TE_REPO_URL}" /workspace/transformerengine && \
    cd /workspace/transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule update --init --recursive && \
    pip install "nvidia-cudnn-frontend==1.29.0" && \
    NVTE_BUILD_NUM_PHILOX_ROUNDS=10 NVTE_CUDA_ARCHS="100a;103a;107a" \
    NVTE_FRAMEWORK=pytorch \
    pip install --no-build-isolation . \
    ; fi

COPY . .

