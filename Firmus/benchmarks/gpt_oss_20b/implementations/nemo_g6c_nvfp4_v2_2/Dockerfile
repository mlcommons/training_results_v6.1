# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Firmus Technologies. All rights reserved.
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

# GPT-OSS 20B NVFP4 (V7) image for GB300 (aarch64, sm_103).
#
# Every source is pinned to an upstream commit; all Firmus changes are the patches in
# docker/patches, applied before the sources are built or installed:
#   te_fused_permutation_2880_2944.patch  TE C++/CUDA: fused permute with 2880->2944 padding
#   transformer_engine_firmus_nvfp4.patch TE PyTorch: NVFP4 grouped MLP and Triton kernels
#   transformer_engine_firmus_v7.patch    TE PyTorch: NVFP4 expert weight all-gather, shard cast,
#                                         dy statistics, residual gradients, packed FP8 QKV;
#                                         64-column fused RHT cast (NVTE_NVFP4_RHT_FUSION_COLS64)
#   megatron_lm_firmus_nvfp4.patch        Megatron-Core: MoE, dispatcher and DDP changes
#   cudnn_frontend_firmus_nvfp4.patch     cuDNN frontend grouped GEMM dGLU/RHT kernels
# Transformer Engine is built for sm_103a with 7 Philox rounds for NVFP4 stochastic rounding.
# docker/constraints pins every Python distribution: official_stack.txt for the Megatron-Bridge
# stack (steps 1-3), final.txt for the CuTe DSL upgrade and later steps (the final image).

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.08-py3@sha256:3becd068f49bd2ad38f90db5f9a4803019a76933a24e63d821376c44e7a9200a
FROM ${FROM_IMAGE_NAME}

ARG MBRIDGE_REVISION=72000e9d4306afa0354fd34e96743d46572b8d41
ARG MCORE_REVISION=f6c33bde4cdc10805ee7157db746d47b758f5f02
ARG TE_REVISION=27486e03cfc1fa41f6932dcecdc47c71c47eac3e
ARG DEEPEP_REVISION=10d4dd7377d5bce900fbb4b80cce863764892b95
ARG MLPERF_LOGGING_REVISION=d197741894957ff281b7b411f256d14581b74f0b
ARG MLPERF_COMMON_REVISION=c02c93c0414071400a10716d9cd5cb8f668b78d4
ARG BUILD_JOBS=8
ARG PIP_C_STACK="-c /opt/firmus/docker/constraints/official_stack.txt"
ARG PIP_C="-c /opt/firmus/docker/constraints/final.txt"

COPY docker /opt/firmus/docker

## 1. Megatron-Bridge and Megatron-Core
RUN rm -rf /workspace/Megatron-Bridge && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge && \
    cd /workspace/Megatron-Bridge && \
    git checkout ${MBRIDGE_REVISION} && \
    git submodule update --init 3rdparty/Megatron-LM && \
    test "$(git -C 3rdparty/Megatron-LM rev-parse HEAD)" = "${MCORE_REVISION}" && \
    git -C 3rdparty/Megatron-LM apply /opt/firmus/docker/patches/megatron_lm_firmus_nvfp4.patch && \
    echo ${MBRIDGE_REVISION} > /MBRIDGE_COMMIT_HASH.env && \
    echo ${MCORE_REVISION} > /MCORE_COMMIT_HASH.env && \
    pip install ${PIP_C_STACK} --no-cache-dir --no-deps -e /workspace/Megatron-Bridge/3rdparty/Megatron-LM && \
    pip install ${PIP_C_STACK} --no-cache-dir --no-deps -e /workspace/Megatron-Bridge

## 2. Megatron-Bridge runtime dependencies and MLPerf logging
RUN pip install ${PIP_C_STACK} --no-cache-dir --ignore-installed "pyyaml==6.0.3" && \
    pip install ${PIP_C_STACK} --no-cache-dir \
      "transformers==5.15.0" "hydra-core==1.3.7" \
      "six==1.17.0" "megatron-energon==7.4.1" \
      "accelerate==1.15.0" "diffusers==0.40.0" \
      "mistral-common==1.11.7" "peft==0.21.0" \
      "qwen-vl-utils==0.0.14" "timm==1.0.29" \
      "open-clip-torch==3.3.0" && \
    pip install ${PIP_C_STACK} --no-cache-dir \
      "git+https://github.com/mlcommons/logging.git@${MLPERF_LOGGING_REVISION}" \
      "git+https://github.com/NVIDIA/mlperf-common.git@${MLPERF_COMMON_REVISION}" \
      "numcodecs==0.16.3" "nvidia-mathdx==25.1.1" && \
    echo ${MLPERF_LOGGING_REVISION} > /MLPERF_LOGGING_COMMIT_HASH.env && \
    echo ${MLPERF_COMMON_REVISION} > /MLPERF_COMMON_COMMIT_HASH.env

## 3. HybridEP (DeepEP) with the Megatron-Bridge patch
RUN export RDMA_CORE_HOME=/opt/rdma-core/build && \
    arch_lib=$(dpkg-architecture -qDEB_HOST_MULTIARCH) && \
    { test -f /usr/lib/${arch_lib}/libmlx5.so || \
      ln -sf /usr/lib/${arch_lib}/libmlx5.so.1 /usr/lib/${arch_lib}/libmlx5.so; } && \
    mkdir -p ${RDMA_CORE_HOME} && \
    ln -sfn /usr/include ${RDMA_CORE_HOME}/include && \
    ln -sfn /usr/lib/${arch_lib} ${RDMA_CORE_HOME}/lib && \
    pip install ${PIP_C_STACK} --no-cache-dir nvidia-nvshmem-cu13==3.6.5 && \
    nvshmem_lib=$(python3 -c 'import importlib.metadata as m; print(m.distribution("nvidia-nvshmem-cu13").locate_file("nvidia/nvshmem/lib"))') && \
    ln -sf ${nvshmem_lib}/libnvshmem_host.so.3 ${nvshmem_lib}/libnvshmem_host.so && \
    rm -rf /opt/DeepEP && \
    git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP && \
    git -C /opt/DeepEP checkout ${DEEPEP_REVISION} && \
    patch -d /opt/DeepEP -p1 < /workspace/Megatron-Bridge/docker/patches/deepep.patch && \
    echo ${DEEPEP_REVISION} > /DEEPEP_COMMIT_HASH.env && \
    apt-get update && \
    apt-get install -y --no-install-recommends "libnvidia-ml-dev=12.0.140~12.0.1-4build4" && \
    HYBRID_EP_MULTINODE=1 TORCH_CUDA_ARCH_LIST="10.0 10.3" MAX_JOBS=${BUILD_JOBS} \
      pip install ${PIP_C_STACK} --no-cache-dir --no-build-isolation -v /opt/DeepEP && \
    apt-get purge -y libnvidia-ml-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig

## 4. Transformer Engine from source
RUN rm -rf /workspace/TransformerEngine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git /workspace/TransformerEngine && \
    cd /workspace/TransformerEngine && \
    git checkout ${TE_REVISION} && \
    git submodule update --init --recursive && \
    git apply /opt/firmus/docker/patches/te_fused_permutation_2880_2944.patch && \
    git apply /opt/firmus/docker/patches/transformer_engine_firmus_nvfp4.patch && \
    git apply /opt/firmus/docker/patches/transformer_engine_firmus_v7.patch && \
    echo ${TE_REVISION} > /TE_COMMIT_HASH.env && \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=0 NVTE_CUDA_ARCHS=103a \
    NVTE_BUILD_NUM_PHILOX_ROUNDS=7 \
    TORCH_CUDA_ARCH_LIST="10.0 10.3" MAX_JOBS=${BUILD_JOBS} \
      pip install ${PIP_C} --no-cache-dir --no-build-isolation --no-deps --force-reinstall -v . && \
    rm -rf /workspace/TransformerEngine/build

## 5. CuTe DSL and FlashInfer
RUN pip install ${PIP_C} --no-cache-dir --force-reinstall "nvidia-cutlass-dsl[cu13]==4.5.0" && \
    pip install ${PIP_C} --no-cache-dir --no-deps --force-reinstall "flashinfer-python==0.6.18.post1"

## 6. cuDNN frontend grouped GEMM kernels (base image ships cudnn-frontend 35fd7b0d)
RUN cd /usr/local/lib/python3.12/dist-packages && \
    sha256sum -c /opt/firmus/docker/checksums/cudnn_frontend_before_patch.sha256 && \
    patch -p1 --forward < /opt/firmus/docker/patches/cudnn_frontend_firmus_nvfp4.patch && \
    sha256sum -c /opt/firmus/docker/checksums/cudnn_frontend_after_patch.sha256

## 7. Benchmark code and its CUDA extensions
WORKDIR /workspace/llm
COPY . .
RUN pip install ${PIP_C} --no-cache-dir --no-build-isolation /workspace/llm/embedding_lib && \
    python3 -c "import nvfp4_clip_adam as m; m.build('/opt/nvfp4_ext', name='nvfp4_clip_adam_ext_fm9', cuda_flags=('-O3', '--use_fast_math'))" && \
    test -s /opt/nvfp4_ext/nvfp4_clip_adam_ext_fm9.so && \
    cd /workspace/TransformerEngine/transformer_engine/pytorch/ops/fused && \
    python3 -c "import nvfp4_shard_cast_cuda as m; m.build('/tmp/ext_shard_cast', arch='10.3a', verbose=False)" && \
    python3 -c "import nvfp4_amax_dbias as m; m.build('/tmp/ext_amax_dbias', arch='10.3', verbose=False)" && \
    cp /tmp/ext_shard_cast/nvfp4_shard_cast_ext.so /tmp/ext_amax_dbias/nvfp4_amax_dbias_ext.so /opt/nvfp4_ext/ && \
    rm -rf /tmp/ext_shard_cast /tmp/ext_amax_dbias && \
    test -s /opt/nvfp4_ext/nvfp4_shard_cast_ext.so && test -s /opt/nvfp4_ext/nvfp4_amax_dbias_ext.so
