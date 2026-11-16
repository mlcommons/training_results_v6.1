# syntax=docker/dockerfile:1
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Qwen3.5 397B GRPO benchmark container.

ARG NEMO_RL_BASE_IMAGE=nvcr.io/nvidia/vllm:26.09-py3
# Build-time only: the development headers, NVTX interop shim and cuDNN engine
# library that the release base trims and the source builds need.
ARG CUDA_DEVEL_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.08-cuda13.4-devel-ubuntu24.04
ARG NEMO_RL_REPO=https://github.com/NVIDIA-NeMo/RL.git
# Official 26.09-alpha integration branch.
ARG NEMO_RL_REVISION=a6ec8a93df4062e5991e225166f60ea981d170e3
ARG MBRIDGE_REPO=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
# 26.09-beta.rc1; pin both Megatron release revisions by commit.
ARG MBRIDGE_REVISION=634ec709062e643f0469610a315b0cfff62cfc85
ARG MCORE_REVISION=094e18d8c3d44aca87573b9bc0f5b56eb6ebd098
ARG TE_REPO_URL=https://github.com/NVIDIA/TransformerEngine.git
# v2.20.2, pinned for reproducible builds.
ARG TE_REPO_REF=6ea2a74a9e98c99e6d7b164a33775cc457520027

FROM ${CUDA_DEVEL_IMAGE} AS cuda_devel

## Stage: base
FROM ${NEMO_RL_BASE_IMAGE} AS base

# The release base keeps nvcc but trims the development headers the source
# builds need (TE: nvtx3; DeepEP: cuda_profiler_api.h, libnvtx3interop).
# Restore the matching CUDA target include tree from the devel image.
RUN --mount=from=cuda_devel,target=/cuda-devel,readonly <<"EOF" bash -exu
RT_INC=$(find /usr/local/cuda/targets -mindepth 2 -maxdepth 2 -type d -name include -print -quit)
CUDA_TARGET=$(basename "$(dirname "${RT_INC}")")
DEV_INC=$(find /cuda-devel/usr/local -type d -path "*/targets/${CUDA_TARGET}/include" -print -quit)
test -n "${RT_INC}" && test -n "${DEV_INC}"
RT_VER=$(awk '$1 == "#define" && $2 == "CUDA_VERSION" {print $3; exit}' "${RT_INC}/cuda.h")
DEV_VER=$(awk '$1 == "#define" && $2 == "CUDA_VERSION" {print $3; exit}' "${DEV_INC}/cuda.h")
test -n "${RT_VER}" && test "${RT_VER}" = "${DEV_VER}"
cp -a "${DEV_INC}/." "${RT_INC}/"
test -f /usr/local/cuda/include/nvtx3/nvToolsExt.h && test -f /usr/local/cuda/include/cuda_profiler_api.h
NVTX_INTEROP=$(find /cuda-devel/usr/local -name 'libnvtx3interop.so*' -print -quit)
test -n "${NVTX_INTEROP}"
cp -a "$(dirname "${NVTX_INTEROP}")"/libnvtx3interop.so* /usr/local/cuda/lib64/
ldconfig
EOF

# rsync: run.sub log collection. apptainer: SWE-agent sandboxes.
RUN sed -i 's|http://ports.ubuntu.com|https://ports.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources && \
    export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install -y --no-install-recommends rsync software-properties-common && \
    add-apt-repository -y ppa:apptainer/ppa && \
    apt-get install -y --no-install-recommends apptainer && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -sf "$(command -v apptainer)" /usr/bin/singularity

ENV CUBLAS_VERSION=13.8.1.7
COPY install_cublas.sh /tmp/install_cublas.sh
RUN bash /tmp/install_cublas.sh && rm /tmp/install_cublas.sh

ARG UV_VERSION=0.11.18
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | XDG_BIN_HOME=/usr/local/bin sh

# NEMO_GYM_VENV_DIR: pre-baked Gym server venvs, reused at runtime.
ENV NRL_CONTAINER=1 NEMO_GYM_VENV_DIR=/opt/gym_venvs \
    UV_PYTHON_PREFERENCE=only-system UV_LINK_MODE=copy

## Stage: nemo-rl-src — pinned NeMo-RL and dependency submodules.
FROM base AS nemo-rl-src
ARG NEMO_RL_REPO NEMO_RL_REVISION MBRIDGE_REPO MBRIDGE_REVISION MCORE_REVISION
RUN <<"EOF" bash -exu
git clone ${NEMO_RL_REPO} /opt/nemo-rl-src
cd /opt/nemo-rl-src
# Pinned commits may no longer be reachable from the branches fetched by clone.
git fetch origin "${NEMO_RL_REVISION}"
git checkout --detach FETCH_HEAD
# Select Bridge directly instead of fetching the integration fork's gitlink.
git -c submodule.3rdparty/Megatron-Bridge.update=none submodule update --init --recursive
MBRIDGE=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git clone "${MBRIDGE_REPO}" "${MBRIDGE}"
git -C "${MBRIDGE}" fetch origin "${MBRIDGE_REVISION}"
git -C "${MBRIDGE}" checkout --detach FETCH_HEAD
git -C "${MBRIDGE}" submodule update --init --recursive
# Bridge RC1's MCore gitlink predates the GRPO GDN/offload changes.
MCORE=${MBRIDGE}/3rdparty/Megatron-LM
git -C "${MCORE}" fetch origin "${MCORE_REVISION}"
git -C "${MCORE}" checkout --detach FETCH_HEAD
EOF

## Stage: swe_setups — SWE-agent toolchains at pinned SHAs (Gym clones any missing tree at start).
FROM base AS swe_setups
ARG SWE_PYTHON_VERSION=3.12.13
COPY --from=nemo-rl-src /opt/nemo-rl-src/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/ /opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/
RUN <<"EOF" bash -exu -o pipefail
unset UV_PYTHON_PREFERENCE
S=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
cd "${S}"; mkdir -p swe_openhands_setup swe_swebench_setup swe_swebench_multilingual_setup swe_r2e_gym_setup swe_rebench_setup
for script in swebench.sh swebench_multilingual.sh r2e_gym.sh; do
    sed -i "s/^uv python install 3\\.12$/uv python install --no-bin ${SWE_PYTHON_VERSION}/" "setup_scripts/${script}"
    sed -i "s/^uv venv --python 3\\.12 venv$/uv venv --managed-python --python ${SWE_PYTHON_VERSION} venv/" "setup_scripts/${script}"
    grep -Fqx "uv python install --no-bin ${SWE_PYTHON_VERSION}" "setup_scripts/${script}"
    grep -Fqx "uv venv --managed-python --python ${SWE_PYTHON_VERSION} venv" "setup_scripts/${script}"
done

# Agent framework.
SETUP_DIR=${S}/swe_openhands_setup MINIFORGE_DIR=${S}/swe_openhands_setup/miniforge3 \
    OPENHANDS_DIR=${S}/swe_openhands_setup/OpenHands \
    AGENT_FRAMEWORK_REPO=https://github.com/michal2409/nv-OpenHands.git \
    AGENT_FRAMEWORK_COMMIT=193be2fbae833c5b9bf3e24989451b6a837180f0 \
    MAKE_BUILD_TIMEOUT_SECONDS=300 setup_scripts/openhands.sh

# Eval harnesses.
SETUP_DIR=${S}/swe_swebench_setup UV_DIR=${S}/swe_swebench_setup/uv PYTHON_DIR=${S}/swe_swebench_setup/python \
    SWEBENCH_DIR=${S}/swe_swebench_setup/SWE-bench \
    SWEBENCH_REPO=https://github.com/HeyyyyyyG/SWE-bench.git \
    SWEBENCH_COMMIT=d546b5e4be7e5fd5eaec6ee67b7588d85306a422 setup_scripts/swebench.sh
SETUP_DIR=${S}/swe_swebench_multilingual_setup UV_DIR=${S}/swe_swebench_multilingual_setup/uv \
    PYTHON_DIR=${S}/swe_swebench_multilingual_setup/python \
    SWEBENCH_DIR=${S}/swe_swebench_multilingual_setup/SWE-bench_Multilingual \
    SWEBENCH_REPO=https://github.com/Kipok/SWE-bench.git \
    SWEBENCH_COMMIT=c6cd63f0f48956fa3db9ba1e0fadff3a4cd2e2c6 setup_scripts/swebench_multilingual.sh
SETUP_DIR=${S}/swe_r2e_gym_setup UV_DIR=${S}/swe_r2e_gym_setup/uv PYTHON_DIR=${S}/swe_r2e_gym_setup/python \
    R2E_GYM_DIR=${S}/swe_r2e_gym_setup/R2E-Gym \
    EVAL_HARNESS_REPO=https://github.com/michal2409/nv-R2E-Gym.git \
    EVAL_HARNESS_COMMIT=4f5f011afdbe70185d71e3e24dfdb47e42c6f465 setup_scripts/r2e_gym.sh
SETUP_DIR=${S}/swe_rebench_setup REBENCH_DIR=${S}/swe_rebench_setup/SWE-rebench-V2 setup_scripts/swe_rebench.sh

# Build-only caches (~2.4 GB).
rm -rf swe_openhands_setup/miniforge3/pkgs swe_openhands_setup/OpenHands/frontend/node_modules
for d in swe_swebench_setup/SWE-bench swe_swebench_multilingual_setup/SWE-bench_Multilingual swe_r2e_gym_setup/R2E-Gym; do
    [[ "$(readlink -f "${d}/venv/bin/python")" == "${S}/${d%%/*}/python/"* ]]
done
swe_r2e_gym_setup/R2E-Gym/venv/bin/python -c "import r2egym"
EOF

## Stage: build_te — Transformer Engine wheel.
FROM base AS build_te
ARG NVTE_BUILD_THREADS_PER_JOB TE_REPO_URL TE_REPO_REF
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG NVTE_CUDA_ARCHS="100a;103a;107a"
RUN --mount=from=cuda_devel,target=/cuda-devel,readonly <<"EOF" bash -exu
SRC=$(find /cuda-devel -name "libcudnn_engines_precompiled.so*" -print -quit)
DST=$(find /usr/lib /usr/local -name libcudnn.so.9 -print -quit)
test -n "${SRC}" && test -n "${DST}"
cp -av "$(dirname "${SRC}")"/libcudnn_engines_precompiled.so* "$(dirname "${DST}")"/
ldconfig
pip install --no-cache-dir pybind11==3.1.0
pip install --no-cache-dir --no-deps nvidia-cudnn-frontend==1.29.0
git init /opt/transformerengine
git -C /opt/transformerengine remote add origin "${TE_REPO_URL}"
git -C /opt/transformerengine fetch --depth 1 origin "${TE_REPO_REF}"
git -C /opt/transformerengine checkout --detach FETCH_HEAD
git -C /opt/transformerengine submodule update --init --recursive --depth 1
test "$(git -C /opt/transformerengine rev-parse HEAD)" = "${TE_REPO_REF}"
mkdir -p /opt/te_wheel
git -C /opt/transformerengine rev-parse HEAD > /opt/te_wheel/TE_COMMIT_HASH.env
printf '%s\n' 1 > /opt/te_wheel/TE_BUILD_NCCL_EP.env
# Build the optional dispatcher; recipes still default to HybridEP.
env NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS}" NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-8}" NVTE_WITH_NCCL_EP=1 \
    pip wheel --no-deps --no-build-isolation --wheel-dir /opt/te_wheel /opt/transformerengine
EOF

## Stage: release — the NeMo-RL venv.
FROM base AS release
WORKDIR /opt/nemo-rl
COPY --from=nemo-rl-src /opt/nemo-rl-src/ /opt/nemo-rl

ARG MAX_JOBS UV_INDEX_URL=https://pypi.org/simple
# sm_100 SASS runs on GB200/GB300/VR. CUDA 13 keeps cub/thrust under include/cccl.
ENV TORCH_CUDA_ARCH_LIST="10.0" CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl \
    UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv

RUN <<"EOF" bash -exu -o pipefail
# Upstream pins python 3.13; the base has 3.12.
sed -i '\#^  "research/flow_grpo",$#d' pyproject.toml
sed -i 's/requires-python = ">=3.13.14,<3.14"/requires-python = ">=3.12,<3.13"/' pyproject.toml
sed -i "s/python_version == '3.13' and sys_platform/python_version == '3.12' and sys_platform/g" pyproject.toml
sed -i 's/cp313-cp313/cp312-cp312/g' pyproject.toml
# tensorrt-llm is not used here.
sed -i "s/\"tensorrt-llm==\(1\.3\.0rc[0-9]\+\)\"/\"tensorrt-llm==\1; python_version >= '3.13'\"/" pyproject.toml
# match-runtime needs torch in the resolution graph; torch comes from the base.
sed -i '\#^\[tool\.uv\.extra-build-dependencies\]#,\#^\[#{/match-runtime[[:space:]]*=[[:space:]]*true/d}' pyproject.toml
sed -i 's/requires-python = ">=3.13.14"/requires-python = ">=3.12"/' \
    research/template_project/pyproject.toml \
    3rdparty/Gym-workspace/Gym/pyproject.toml \
    3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/pyproject.toml
# Gym's vllm extra is not installed here.
sed -Ei 's/"(vllm==[^"]+)"/"\1; python_version >= '\''3.13'\''"/;s/"(flashinfer-python==[^"]+)"/"\1; python_version >= '\''3.13'\''"/' \
    3rdparty/Gym-workspace/Gym/pyproject.toml
printf '3.12\n' > .python-version

uv venv --seed --system-site-packages --python python3 /opt/nemo_rl_venv
export VIRTUAL_ENV=/opt/nemo_rl_venv
uv lock

# The base torch headers need C++20; DeepGEMM/DeepEP pass -std=c++17.
mkdir -p /tmp/cxx20
for tool in g++:/usr/bin/g++ nvcc:/usr/local/cuda/bin/nvcc; do
    printf '#!/usr/bin/env bash\nargs=(); for a in "$@"; do [[ "$a" == -std=c++17 ]] && a=-std=c++20; args+=("$a"); done\nexec %s "${args[@]}"\n' \
        "${tool#*:}" > "/tmp/cxx20/${tool%%:*}"
done
chmod 0755 /tmp/cxx20/*
export CXX=/tmp/cxx20/g++ PYTORCH_NVCC=/tmp/cxx20/nvcc

# Base-provided packages the lock must not replace.
NO_INSTALL="--no-install-package torch --no-install-package torchvision --no-install-package torchaudio --no-install-package torchcodec --no-install-package triton --no-install-package triton-kernels --no-install-package nvidia-cublas --no-install-package nvidia-cuda-cupti --no-install-package nvidia-cuda-nvrtc --no-install-package nvidia-cuda-runtime --no-install-package nvidia-cudnn-cu13 --no-install-package nvidia-cudnn-frontend --no-install-package nvidia-cufft --no-install-package nvidia-cufile --no-install-package nvidia-curand --no-install-package nvidia-cusolver --no-install-package nvidia-cusparse --no-install-package nvidia-cusparselt-cu13 --no-install-package nvidia-nccl-cu13 --no-install-package vllm --no-install-package flash-attn --no-install-package flash-attn-4 --no-install-package numpy --no-install-package nixl --no-install-package nixl-cu13 --no-install-package nixl-cu12 --no-install-package cuda-bindings --no-install-package cuda-python --no-install-package transformer-engine --no-install-package transformer-engine-torch --no-install-package transformer-engine-cu13 --no-install-package nvidia-cutlass-dsl --no-install-package nvidia-cutlass-dsl-libs-base --no-install-package nvidia-cutlass-dsl-libs-core --no-install-package nvidia-cutlass-dsl-libs-cu12 --no-install-package nvidia-cutlass-dsl-libs-cu13 --no-install-package flashinfer-python --no-install-package flashinfer-cubin --no-install-package flashinfer-jit-cache --no-install-package openai --no-install-package quack-kernels"
# vllm and mcore extras are declared conflicting; sync vllm first, training last.
uv sync --locked --inexact --extra vllm --no-install-project ${NO_INSTALL}
uv sync --locked --inexact --extra mcore --extra nemo_gym ${NO_INSTALL}

rm -rf /tmp/cxx20 /root/.cache/uv
EOF

# --no-config: install our TE wheel independently of the upstream pin.
# cuDNN Frontend matches the TE build.
RUN --mount=type=bind,from=build_te,source=/opt/te_wheel,target=/tmp/te,readonly \
    VIRTUAL_ENV=/opt/nemo_rl_venv uv pip install --no-cache-dir --no-config --no-deps /tmp/te/*.whl && \
    VIRTUAL_ENV=/opt/nemo_rl_venv uv pip install --no-cache-dir --no-config --no-deps --reinstall-package nvidia-cudnn-frontend nvidia-cudnn-frontend==1.29.0 && \
    cp /tmp/te/TE_COMMIT_HASH.env /TE_COMMIT_HASH.env && \
    cp /tmp/te/TE_BUILD_NCCL_EP.env /TE_BUILD_NCCL_EP.env

# Megatron imports flash_attn.cute (FA4).
RUN python3 -m pip install --no-cache-dir --no-deps "flash-attn-4==4.0.0b27"

ENV PATH="/opt/nemo_rl_venv/bin:${PATH}"

# Megatron's gradient_accumulation_fusion needs Apex's fused weight-gradient
# extension. Build against the base torch visible in the trainer venv.
ARG APEX_REVISION=26.09
RUN <<"EOF" bash -exu -o pipefail
git clone https://github.com/NVIDIA/apex.git /tmp/apex
git -C /tmp/apex checkout --detach "${APEX_REVISION}"
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 MAX_JOBS="${MAX_JOBS:-8}" NVCC_APPEND_FLAGS="--threads 2" \
    /opt/nemo_rl_venv/bin/python -m pip install --no-cache-dir --no-build-isolation --no-deps /tmp/apex
git -C /tmp/apex rev-parse HEAD > /APEX_COMMIT_HASH.env
rm -rf /tmp/apex
EOF

# All actors use this interpreter; `uv run` must never re-sync.
ENV NEMO_RL_PY_EXECUTABLES_SYSTEM=1 UV_NO_SYNC=1
# aarch64: TLS-heavy libtvm (tilelang) needs these registered first.
ENV LD_PRELOAD="libstdc++.so.6:libgomp.so.1"
# Real libcudart under TileLang's libcudart_stub.so name. /opt/amazon: AWS EFA NCCL plugin.
RUN mkdir -p /opt/cudart-compat && ln -sf /usr/local/cuda/lib64/libcudart.so /opt/cudart-compat/libcudart_stub.so
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/cudart-compat:${LD_LIBRARY_PATH}"

## Stage: mlperf (final)
FROM release AS mlperf
ARG NEMO_RL_BASE_IMAGE NEMO_RL_REPO NEMO_RL_REVISION GIT_COMMIT_ID
RUN test -n "${GIT_COMMIT_ID}"
ENV GIT_COMMIT_ID=${GIT_COMMIT_ID}
LABEL org.opencontainers.image.revision="${GIT_COMMIT_ID}" org.opencontainers.image.base.name="${NEMO_RL_BASE_IMAGE}" \
      com.nvidia.nemo-rl.repository="${NEMO_RL_REPO}" com.nvidia.nemo-rl.revision="${NEMO_RL_REVISION}"

COPY --link --from=swe_setups /opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/ /opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/

# Benchmark files that must live inside the NeMo-RL tree.
COPY mlperf_grpo_logging.py nemo_rl/algorithms/mlperf_grpo_logging.py
COPY conf/ qwen_35/configs/
COPY python/low_precision_recipes.py low_precision_recipes.py

# Pre-bake Gym server venvs once per shard definition, regardless of replicas.
RUN <<"EOF" bash -exu -o pipefail
mkdir -p /tmp/prefetch/model /tmp/prefetch/sif; touch /tmp/prefetch/train.jsonl /tmp/prefetch/validation.jsonl
CONTAINER_HF_CKPT_PATH=/tmp/prefetch/model NEMO_GYM_SWE_SIF_DIR=/tmp/prefetch/sif \
    NEMO_GYM_SWE_TRAIN_DATA_PATH=/tmp/prefetch/train.jsonl NEMO_GYM_SWE_VALIDATION_DATA_PATH=/tmp/prefetch/validation.jsonl \
    uv run python - <<'PYEOF'
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
from nemo_rl.environments.nemo_gym_shards import apply_shard_overlay, parse_shard_plan
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()
config = OmegaConf.to_container(
    load_config("qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml"), resolve=True
)
gym_config = config["env"]["nemo_gym"]
plan = parse_shard_plan(gym_config)
if plan is None:
    raise ValueError("The canonical Gym recipe must define shards")

# The pinned prefetch script expects one merged Gym stack. These temporary
# recipes prepare its venvs only; runtime still uses the canonical shard plan.
prefetch_recipes = []
for shard in plan.shards:
    path = Path("/tmp/prefetch") / f"{shard.name}.yaml"
    gym_stack = apply_shard_overlay(gym_config, plan, shard)
    OmegaConf.save(OmegaConf.create({"env": {"nemo_gym": gym_stack}}), path)
    prefetch_recipes.append(str(path))
subprocess.run(
    [sys.executable, "examples/nemo_gym/prefetch_venvs.py", *prefetch_recipes], check=True
)
PYEOF
rm -rf /tmp/prefetch /root/.cache/uv
# The locked torch must not shadow the base torch.
test ! -e /opt/nemo_rl_venv/lib/python3.12/site-packages/torch
for v in responses_api_models/vllm_model responses_api_agents/swe_agents; do
    "/opt/gym_venvs/${v}/.venv/bin/python" -c "import aiohttp, ray, nemo_gym"
done
EOF

# Allow FA2 head_dim>192 (Qwen3.5 uses 256) on sm_103/sm_107.
RUN <<"EOF" python
import glob, py_compile, sys
OLD, NEW = "((8, 0), (9, 0), (10, 0), (12, 0))", "((8, 0), (9, 0), (10, 0), (10, 3), (10, 7), (12, 0))"
[path] = glob.glob("/opt/nemo_rl_venv/lib/python3*/site-packages/transformer_engine/pytorch/attention/dot_product_attention/utils.py")
src = open(path).read()
if src.count(OLD) != 1:
    sys.exit(f"FA2 arch allowlist count={src.count(OLD)} in {path}; TE changed, re-evaluate")
open(path, "w").write(src.replace(OLD, NEW, 1))
py_compile.compile(path, doraise=True)
EOF

COPY requirements-mlperf.txt /tmp/requirements-mlperf.txt
RUN uv pip install --no-cache-dir -r /tmp/requirements-mlperf.txt

# MLPerf launch tree.
WORKDIR /workspace/llm
COPY . .
ENV PYTHONPATH="/workspace/llm:${PYTHONPATH}"
WORKDIR /opt/nemo-rl

# nemo_rl checks this fingerprint at import.
RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

