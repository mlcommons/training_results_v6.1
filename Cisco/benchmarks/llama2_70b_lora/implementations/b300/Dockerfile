# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 
                                                                         
                                                                            
                                                                           
                                                                          
                                                                       
                                                                      
 
                                                                            
                                                     
 
                                                                            
                                                                          
                                                                         
                                                                            
                                                                         
                                                                     
                           

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${FROM_IMAGE_NAME}

ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a" && \
    git config --global http.proxy ${http_proxy} && \
    git config --global https.proxy ${https_proxy}

ENV PIP_CONSTRAINT=""

WORKDIR /workspace/ft-llm

        
RUN NVARCH=$(uname -m |grep x86_64 2>/dev/null || echo sbsa) && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub > /etc/apt/trusted.gpg.d/cuda.asc && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y libcudnn9-cuda-13=9.21.1.3-1 libcudnn9-dev-cuda-13=9.21.1.3-1 libcudnn9-headers-cuda-13=9.21.1.3-1 && rm -rf /var/lib/apt/lists/*
ENV CUDNN_VERSION=9.21.1.3

       
ARG NCCL_REVISION=v2.30.4-1
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends fakeroot devscripts debhelper && \
    apt-get purge -y libnccl-dev libnccl2 && \
    git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    git checkout ${NCCL_REVISION} && \
    echo "NCCL_REVISION=${NCCL_REVISION}" && \
    make -j24 pkg.debian.build && \
    dpkg -i build/pkg/deb/*.deb && \
    cd .. && rm -rf nccl && \
    apt-get purge -y fakeroot devscripts debhelper
ENV NCCL_VERSION=${NCCL_REVISION}

                                       
ARG MBRIDGE_REVISION=26.04-alpha.rc1
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge && \
    cd /workspace/Megatron-Bridge && \
    git checkout ${MBRIDGE_REVISION} && \
    git submodule update --init --recursive && \
    echo MBRIDGE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MBRIDGE_COMMIT_HASH.env && \
    cd 3rdparty/Megatron-LM && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install -e .

ENV PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"

                        
ARG TE_REVISION=SKIP
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    pip uninstall -y transformer_engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git /workspace/transformerengine && \
    cd /workspace/transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    pip install nvidia-mathdx==25.1.1 && \
    git submodule init && git submodule update && \
    NVTE_CUDA_ARCHS="100a;103a" NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1 NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi


ADD . /workspace/ft-llm
RUN pip install -r requirements.txt

ENV PATH="/root/.local/bin:${PATH}"
