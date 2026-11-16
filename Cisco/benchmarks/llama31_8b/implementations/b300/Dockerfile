# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 
                                                                         
                                                                            
                                                                           
                                                                          
                                                                       
                                                                      
 
                                                                            
                                                     
 
                                                                            
                                                                          
                                                                         
                                                                            
                                                                         
                                                                     
                           

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

        
RUN NVARCH=$(uname -m |grep x86_64 2>/dev/null || echo sbsa) && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub > /etc/apt/trusted.gpg.d/cuda.asc && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y libcudnn9-cuda-13=9.21.1.3-1 && rm -rf /var/lib/apt/lists/*
ENV CUDNN_VERSION=9.21.1.3

WORKDIR /workspace

       
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

                                   
WORKDIR /workspace/pytorch
COPY ./pytorch_ckpt.patch /workspace/pytorch/pytorch_ckpt.patch
                         
                                                                                   
RUN patch --directory=/usr/local/lib/python3.12/dist-packages -p1 < /workspace/pytorch/pytorch_ckpt.patch

WORKDIR /workspace/
RUN pip install numcodecs==0.16.3
RUN pip install nvidia-mathdx==25.1.1

                                       
ARG MBRIDGE_REVISION=26.04-alpha.rc1
ARG MCORE_REVISION=26.04-alpha.rc1
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge && \
    cd /workspace/Megatron-Bridge && \
    git checkout ${MBRIDGE_REVISION} && \
    git submodule update --init --recursive && \
    echo MBRIDGE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MBRIDGE_COMMIT_HASH.env && \
    cd /workspace/Megatron-Bridge/3rdparty/Megatron-LM && \
    git checkout ${MCORE_REVISION} && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install -e .

ENV PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"

                        
ARG TE_REVISION=SKIP
ENV CUSTOM_TE_REVISION ${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
    cd transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule init && git submodule update && \
    NVTE_CUDA_ARCHS="100a;103a" NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi

                            
COPY requirements.txt .
RUN pip uninstall -y transformers torchao && pip install --no-cache-dir -r requirements.txt

                    
WORKDIR /workspace/llm
COPY . .
RUN cd /workspace/llm/embedding_lib && pip install --force-reinstall --no-deps --no-build-isolation .

