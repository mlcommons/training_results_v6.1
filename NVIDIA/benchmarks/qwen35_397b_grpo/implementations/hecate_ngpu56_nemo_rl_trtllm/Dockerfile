# syntax=docker/dockerfile:1
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Qwen3.5 397B GRPO benchmark container with TRT-LLM rollouts.

ARG NEMO_RL_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.08-py3
ARG NEMO_RL_REPO=https://github.com/NVIDIA-NeMo/RL.git
# Also pins the TensorRT-LLM source ref built below
# (3rdparty/TensorRT-LLM-workspace/pyproject.toml [tool.trtllm]).
# Official 26.09-beta branch
ARG NEMO_RL_REVISION=4c823cd8955cd41e09a65399f4abaad6240fe396
ARG MBRIDGE_REPO=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
# 26.09-beta.rc1; pin both Megatron release revisions by commit.
ARG MBRIDGE_REVISION=634ec709062e643f0469610a315b0cfff62cfc85
ARG MCORE_REVISION=0d454e9172641c95a584028bfafd5b112acfe9dc
ARG TE_REPO_URL=https://github.com/NVIDIA/TransformerEngine.git
# v2.20.2, pinned for reproducible builds.
ARG TE_REPO_REF=6ea2a74a9e98c99e6d7b164a33775cc457520027
# GB200 (sm_100), GB300 (sm_103), VR (sm_107).
ARG BUILD_CUSTOM_TRTLLM_ARCH="100-real;103-real;107-real"

## Stage: base
FROM ${NEMO_RL_BASE_IMAGE} AS base

# rsync: run.sub log collection. ccache, libzmq3-dev: TRT-LLM build (libzmq is also
# dlopen()ed by its UCX cache transceiver at run time). apptainer: SWE-agent sandboxes.
RUN sed -i 's|http://ports.ubuntu.com|https://ports.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources && \
    export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install -y --no-install-recommends rsync ccache libzmq3-dev software-properties-common && \
    add-apt-repository -y ppa:apptainer/ppa && \
    apt-get install -y --no-install-recommends apptainer && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -sf "$(command -v apptainer)" /usr/bin/singularity

ENV CUBLAS_VERSION=13.8.1.7
COPY install_cublas.sh /tmp/install_cublas.sh
RUN bash /tmp/install_cublas.sh && rm /tmp/install_cublas.sh

# Preserve the base Apex CUDA extensions for Megatron gradient accumulation
# fusion. Transformer Engine is rebuilt below for the trainer venv.
RUN python3 -m pip uninstall -y transformer_engine

# Rubin (sm_107a) needs CUTLASS DSL >= 4.8.0.dev0; the base ships 4.6.2. quack-kernels
# (imported by FA4, TRT-LLM and modelopt) must match it: the lock's 0.4.1 fails on import
# against 4.8 (cute.core.ThrMma is gone).
ARG CUTLASS_DSL_VERSION=4.8.0.dev0 QUACK_KERNELS_VERSION=0.6.1
RUN unset PIP_CONSTRAINT && python3 -m pip install --no-cache-dir --pre "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" && \
    python3 -m pip install --no-cache-dir --no-deps "quack-kernels==${QUACK_KERNELS_VERSION}" && \
    python3 -c 'import quack; import cutlass.cute as cute; print(quack.__file__)'

# The FlashInfer release the pinned TRT-LLM targets (the base ships none). The cubin and
# jit-cache wheels carry the prebuilt trtllm-gen kernels for sm_100a/103a/107a.
ARG FLASHINFER_VERSION=0.6.18.post1
RUN <<"EOF" bash -exu -o pipefail
unset PIP_CONSTRAINT
python3 -m pip install --no-cache-dir --no-deps "flashinfer-python==${FLASHINFER_VERSION}"
python3 -m pip install --no-cache-dir --no-deps --index-url https://flashinfer.ai/whl "flashinfer-cubin==${FLASHINFER_VERSION}"
python3 -m pip install --no-cache-dir --no-deps --index-url https://flashinfer.ai/whl/cu134 "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu134"
# flashinfer itself is not imported: its GDN module reads the GPU at import and the builder has none.
python3 -c 'import pathlib, flashinfer_cubin; assert list(pathlib.Path(flashinfer_cubin.get_cubin_dir()).rglob("fmha/trtllm-gen/fmhaSm107a*.cubin"))'
EOF

# NIXL for TRT-LLM's KV-cache transceiver (disaggregated prefill/decode), at the prefix
# build-custom-trtllm.sh expects. Same version and options as TensorRT-LLM's
# docker/common/install_nixl.sh, built against the base's UCX.
ARG NIXL_VERSION=v1.4.0
RUN <<"EOF" bash -exu -o pipefail
pip install --no-cache-dir meson==1.12.1
git clone --depth 1 -b "${NIXL_VERSION}" https://github.com/ai-dynamo/nixl.git /tmp/nixl
cd /tmp/nixl
GDS_PATH=/usr/local/cuda/targets/sbsa-linux
[[ "$(uname -m)" != x86_64 ]] || GDS_PATH=/usr/local/cuda/targets/x86_64-linux
# meson's CUDA probe needs a loadable libcuda.so.1; the builder only has the compat copy.
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(dirname "$(find /usr/local -name libcuda.so.1 -print -quit)")" meson setup builddir \
    -Ducx_path=/usr/local/ucx -Dcudapath_lib=/usr/local/cuda/lib64 -Dcudapath_inc=/usr/local/cuda/include \
    -Dgds_path="${GDS_PATH}" -Dinstall_headers=true -Ddisable_plugins=POSIX,GUSLI \
    -Dbuild_tests=false -Dbuild_examples=false --buildtype=release
ninja -C builddir install
rm -rf /tmp/nixl
test -f /opt/nvidia/nvda_nixl/include/nixl.h
# meson picks lib/, lib64/ or lib/<triplet> depending on the prefix; register what it chose.
NIXL_LIB=$(dirname "$(find /opt/nvidia/nvda_nixl -name libnixl.so -print -quit)")
printf '%s\n' "${NIXL_LIB}" "${NIXL_LIB}/plugins" > /etc/ld.so.conf.d/nixl.conf
ldconfig
echo "${NIXL_VERSION}" > /NIXL_VERSION.env
EOF
ENV LD_LIBRARY_PATH="/opt/nvidia/nvda_nixl/lib:/opt/nvidia/nvda_nixl/lib64:/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu:/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:/usr/local/ucx/lib:${LD_LIBRARY_PATH}"

ARG UV_VERSION=0.11.18
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | XDG_BIN_HOME=/usr/local/bin sh

# NEMO_GYM_VENV_DIR: pre-baked Gym server venvs, reused at runtime. TRT-LLM initialises MPI
# (mpi4py) as the container's root user.
ENV NRL_CONTAINER=1 NEMO_GYM_VENV_DIR=/opt/gym_venvs \
    UV_PYTHON_PREFERENCE=only-system UV_LINK_MODE=copy \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PRTE_ALLOW_RUN_AS_ROOT=1 PRTE_ALLOW_RUN_AS_ROOT_CONFIRM=1

## Stage: nemo-rl-src — pinned NeMo-RL and dependency submodules.
FROM base AS nemo-rl-src
ARG NEMO_RL_REPO NEMO_RL_REVISION MBRIDGE_REPO MBRIDGE_REVISION MCORE_REVISION
RUN <<"EOF" bash -exu
git clone "${NEMO_RL_REPO}" /opt/nemo-rl-src
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
COPY --from=nemo-rl-src /opt/nemo-rl-src/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/ /opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/
RUN <<"EOF" bash -exu -o pipefail
S=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents
cd "${S}"; mkdir -p swe_openhands_setup swe_swebench_setup swe_swebench_multilingual_setup swe_r2e_gym_setup swe_rebench_setup

# Agent framework. The commit must equal agent_framework_commit in
# conf/grpo_qwen35_397b_swe_openhands_async.yaml, or Gym re-clones OpenHands at run time
# and the agent containers lose the scratch directories they bind-mount.
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
# only-managed: the eval runs inside the r2e testbed SIFs with only this directory
# bind-mounted, so the venv interpreter must live inside it.
SETUP_DIR=${S}/swe_r2e_gym_setup UV_DIR=${S}/swe_r2e_gym_setup/uv PYTHON_DIR=${S}/swe_r2e_gym_setup/python \
    UV_PYTHON_PREFERENCE=only-managed \
    R2E_GYM_DIR=${S}/swe_r2e_gym_setup/R2E-Gym \
    EVAL_HARNESS_REPO=https://github.com/michal2409/nv-R2E-Gym.git \
    EVAL_HARNESS_COMMIT=4f5f011afdbe70185d71e3e24dfdb47e42c6f465 setup_scripts/r2e_gym.sh
[[ "$(readlink -f swe_r2e_gym_setup/R2E-Gym/venv/bin/python)" == "${S}"/swe_r2e_gym_setup/python/* ]]
SETUP_DIR=${S}/swe_rebench_setup REBENCH_DIR=${S}/swe_rebench_setup/SWE-rebench-V2 setup_scripts/swe_rebench.sh

# Build-only caches (~2.4 GB).
rm -rf swe_openhands_setup/miniforge3/pkgs swe_openhands_setup/OpenHands/frontend/node_modules
EOF

## Stage: build_te — Transformer Engine wheel.
FROM base AS build_te
ARG NVTE_BUILD_THREADS_PER_JOB TE_REPO_URL TE_REPO_REF
ARG NVTE_CUDA_ARCHS="100a;103a;107a"
# Include <cassert> for CUDA kernels that use assert() with this toolchain.
ENV NVCC_APPEND_FLAGS="-include cassert"
RUN <<"EOF" bash -exu
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

## Stage: build_trtllm — TRT-LLM wheel, built by nemo-rl's own script from the source ref in
## 3rdparty/TensorRT-LLM-workspace/pyproject.toml. Its own stage so the ~60 min overlap
## build_te and the venv syncs.
FROM base AS build_trtllm
ARG BUILD_CUSTOM_TRTLLM_ARCH TRTLLM_BUILD_JOBS
# Optional remote ccache storage; empty means the local cache mount only.
ARG CCACHE_REMOTE_STORAGE
ENV CCACHE_REMOTE_STORAGE=${CCACHE_REMOTE_STORAGE} BUILD_CUSTOM_TRTLLM_ARCH=${BUILD_CUSTOM_TRTLLM_ARCH} \
    NIXL_ROOT_DIR=/opt/nvidia/nvda_nixl
# TensorRT-LLM calls assert() without including <cassert> in one kernel.
ENV NVCC_APPEND_FLAGS="-include cassert"
WORKDIR /opt/trtllm-build
# The script derives its build dir from its own location; keep this layout.
COPY --from=nemo-rl-src /opt/nemo-rl-src/tools/build-custom-trtllm.sh ./tools/
COPY --from=nemo-rl-src /opt/nemo-rl-src/3rdparty/TensorRT-LLM-workspace/pyproject.toml ./3rdparty/TensorRT-LLM-workspace/
# Drop the pip NCCL/NVRTC pins from TRT-LLM's requirements.txt right after the script's
# checkout: a pip NCCL that differs from the one the base torch links breaks `import torch`.
RUN <<"EOF" python3
import pathlib
p = pathlib.Path("tools/build-custom-trtllm.sh")
s = p.read_text()
anchor = "sed -i '/^nvidia-modelopt/d' requirements.txt\n"
assert s.count(anchor) == 1, "build-custom-trtllm.sh changed; re-check this hook"
p.write_text(s.replace(anchor, anchor + "sed -i '/^nvidia-nccl-cu13/d;/^nvidia-cuda-nvrtc/d' requirements.txt\n"))
EOF
RUN --mount=type=cache,target=/root/.cache/ccache \
    <<"EOF" bash -exu -o pipefail
# Build the public TensorRT-LLM source at the revision pinned by NeMo-RL.
read -r TRTLLM_URL TRTLLM_REF < <(python3 -c 'import tomllib; t = tomllib.load(open("3rdparty/TensorRT-LLM-workspace/pyproject.toml", "rb"))["tool"]["trtllm"]; print(t["url"], t["ref"])')
# A full build is ~9.8k objects; ccache's 5G default would thrash.
CCACHE_DIR=/root/.cache/ccache ccache -M 20G
CCACHE_DIR=/root/.cache/ccache bash tools/build-custom-trtllm.sh "${TRTLLM_URL}" "${TRTLLM_REF}"
ls /opt/trtllm_wheels/tensorrt_llm-*.whl
# Record the pinned source in /NEMO_RL_PROVENANCE.txt.
{ echo "TRTLLM_REPO=${TRTLLM_URL}"; echo "TRTLLM_REVISION=${TRTLLM_REF}"; echo "TRTLLM_CUDA_ARCHS=${BUILD_CUSTOM_TRTLLM_ARCH}"; } > /opt/trtllm_wheels/TRTLLM_SOURCE.env
EOF

## Stage: release — the NeMo-RL venv, plus TE and TRT-LLM.
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
# match-runtime needs torch in the resolution graph; torch comes from the base.
sed -i '\#^\[tool\.uv\.extra-build-dependencies\]#,\#^\[#{/match-runtime[[:space:]]*=[[:space:]]*true/d}' pyproject.toml
# The TRT-LLM wheel is built here for the venv's own interpreter, so cp312 is fine.
sed -i 's/requires-python = ">=3.13"/requires-python = ">=3.12"/' 3rdparty/TensorRT-LLM-workspace/pyproject.toml
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

# Packages the lock must not replace: base-provided ones, the CUTLASS DSL pairing from the
# base stage, and the two engines installed below into the system layer (tensorrt-llm,
# flash-attn-4). The trtllm extra stays in the resolution so its ~90 deps are lock-pinned.
NO_INSTALL="--no-install-package tensorrt-llm --no-install-package nvidia-nvjitlink --no-install-package torch --no-install-package torchvision --no-install-package torchaudio --no-install-package torchcodec --no-install-package triton --no-install-package triton-kernels --no-install-package nvidia-cublas --no-install-package nvidia-cuda-cupti --no-install-package nvidia-cuda-nvrtc --no-install-package nvidia-cuda-runtime --no-install-package nvidia-cudnn-cu13 --no-install-package nvidia-cudnn-frontend --no-install-package nvidia-cufft --no-install-package nvidia-cufile --no-install-package nvidia-curand --no-install-package nvidia-cusolver --no-install-package nvidia-cusparse --no-install-package nvidia-cusparselt-cu13 --no-install-package nvidia-nccl-cu13 --no-install-package vllm --no-install-package flash-attn --no-install-package flash-attn-4 --no-install-package numpy --no-install-package nixl --no-install-package nixl-cu13 --no-install-package nixl-cu12 --no-install-package cuda-bindings --no-install-package cuda-python --no-install-package transformer-engine --no-install-package transformer-engine-torch --no-install-package transformer-engine-cu13 --no-install-package nvidia-cutlass-dsl --no-install-package nvidia-cutlass-dsl-libs-base --no-install-package nvidia-cutlass-dsl-libs-core --no-install-package nvidia-cutlass-dsl-libs-cu12 --no-install-package nvidia-cutlass-dsl-libs-cu13 --no-install-package apache-tvm-ffi --no-install-package flashinfer-python --no-install-package flashinfer-cubin --no-install-package flashinfer-jit-cache --no-install-package quack-kernels"
# trtllm is declared conflicting with mcore/nemo_gym: sync it first, training last so it
# owns any shared pin; --inexact keeps the earlier packages installed.
uv sync --locked --inexact --extra trtllm --no-install-project ${NO_INSTALL}
uv sync --locked --inexact --extra mcore --extra nemo_gym ${NO_INSTALL}
rm -rf /root/.cache/uv
EOF

# --no-config: install our TE wheel independently of the upstream pin.
# --no-deps: the base owns the CUDA-native
# dependencies. cuDNN Frontend 1.29.0 fixes a VR break in the version the lock pins.
RUN --mount=type=bind,from=build_te,source=/opt/te_wheel,target=/tmp/te,readonly \
    VIRTUAL_ENV=/opt/nemo_rl_venv uv pip install --no-cache-dir --no-config --no-deps /tmp/te/*.whl && \
    VIRTUAL_ENV=/opt/nemo_rl_venv uv pip install --no-cache-dir --no-config --no-deps --reinstall-package nvidia-cudnn-frontend nvidia-cudnn-frontend==1.29.0 && \
    cp /tmp/te/TE_COMMIT_HASH.env /TE_COMMIT_HASH.env && \
    cp /tmp/te/TE_BUILD_NCCL_EP.env /TE_BUILD_NCCL_EP.env

# TRT-LLM goes into the system layer, which the venv must never shadow (see
# NO_INSTALL above). --no-deps: the trtllm extra installed its deps above.
RUN --mount=type=bind,from=build_trtllm,source=/opt/trtllm_wheels,target=/tmp/trtllm,readonly \
    <<"EOF" bash -exu -o pipefail
python3 -m pip install --no-cache-dir --no-deps /tmp/trtllm/tensorrt_llm-*.whl
cp /tmp/trtllm/TRTLLM_SOURCE.env /TRTLLM_SOURCE.env
# The wheel's bundled CCCL predates cuda/std/__tuple_dir/structured_bindings.h, which its
# own FMHA headers include when NVRTC JIT-compiles them; use the toolkit's CCCL instead.
SITE=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
FMHA_CCCL="${SITE}/tensorrt_llm/include/trtllm_gen_kernels/fmha/cuda/cccl"
if [[ -d "${FMHA_CCCL}" && ! -e "${FMHA_CCCL}/cuda/std/__tuple_dir/structured_bindings.h" ]]; then
    test -e /usr/local/cuda/include/cccl/cuda/std/__tuple_dir/structured_bindings.h
    rm -rf "${FMHA_CCCL}" && ln -s /usr/local/cuda/include/cccl "${FMHA_CCCL}"
fi
EOF

# Megatron imports flash_attn.cute (FA4); needs the CUTLASS DSL from the base stage.
RUN python3 -m pip install --no-cache-dir --no-deps "flash-attn-4==4.0.0b27"

ENV PATH="/opt/nemo_rl_venv/bin:${PATH}"
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
# The locked torch and tensorrt-llm must not shadow the system copies.
test ! -e /opt/nemo_rl_venv/lib/python3.12/site-packages/torch
test ! -e /opt/nemo_rl_venv/lib/python3.12/site-packages/tensorrt_llm
for v in responses_api_models/vllm_model responses_api_agents/swe_agents; do
    "/opt/gym_venvs/${v}/.venv/bin/python" -c "import aiohttp, ray, nemo_gym"
done
EOF

COPY requirements-mlperf.txt /tmp/requirements-mlperf.txt
RUN uv pip install --no-cache-dir -r /tmp/requirements-mlperf.txt

# MLPerf launch tree. This directory owns every file the runtime uses.
WORKDIR /workspace/optimized/qwen35_397b_grpo_trtllm/pytorch
COPY . .
ENV PYTHONPATH="/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch:${PYTHONPATH}"
WORKDIR /opt/nemo-rl

# nemo_rl checks this fingerprint at import.
RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

# Provenance surfaced by run.sub.
RUN <<"EOF" bash -eu
printf '%s\n' "${NEMO_RL_REVISION}-dirty" > /NEMO_RL_COMMIT_HASH.env
printf '%s\n' "$(git rev-parse :3rdparty/Gym-workspace/Gym)-dirty" > /NEMO_GYM_COMMIT_HASH.env
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD > /MBRIDGE_COMMIT_HASH.env
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD > /MCORE_COMMIT_HASH.env
git -C 3rdparty/Automodel-workspace/Automodel rev-parse HEAD > /AUTOMODEL_COMMIT_HASH.env
printf '%s\n' "${NEMO_RL_BASE_IMAGE}" > /NEMO_RL_BASE_IMAGE.env
echo '<absent>' > /VLLM_VERSION.env
python -c 'import importlib.metadata as m; print(m.version("transformer-engine"))' > /TRANSFORMER_ENGINE_VERSION.env
{
    echo "OPTIMIZED_GIT_COMMIT_ID=${GIT_COMMIT_ID}"
    echo "NEMO_RL_REPO=${NEMO_RL_REPO}"
    for k in NEMO_RL_COMMIT_HASH NEMO_GYM_COMMIT_HASH MBRIDGE_COMMIT_HASH MCORE_COMMIT_HASH AUTOMODEL_COMMIT_HASH \
             NEMO_RL_BASE_IMAGE TRANSFORMER_ENGINE_VERSION TE_COMMIT_HASH NIXL_VERSION; do echo "$k=$(cat /$k.env)"; done
    cat /TRTLLM_SOURCE.env
    echo "TRANSFORMER_ENGINE_BUILD_NCCL_EP=$(cat /TE_BUILD_NCCL_EP.env)"
    echo "INSTALLED_STACK_BEGIN"
    python - <<'PY'
import importlib.metadata as m
for d in ("apex", "torch", "torchvision", "triton", "tensorrt-llm", "tensorrt", "transformer-engine", "flash-attn",
          "flash-attn-4", "nvidia-cudnn-frontend", "nvidia-cutlass-dsl", "flashinfer-python", "flashinfer-cubin",
          "flashinfer-jit-cache", "megatron-core", "megatron-bridge", "nemo-gym", "deep-ep", "fastokens", "ray", "transformers",
          "mlperf-logging"):
    try: print(f"{d}=={m.version(d)}")
    except Exception: print(f"{d}==<absent>")
PY
    echo "INSTALLED_STACK_END"
    echo "SUBMODULES_BEGIN"; git submodule status --recursive; echo "SUBMODULES_END"
} > /NEMO_RL_PROVENANCE.txt
EOF

## Build-time checks: the rollout engine resolves from the system layer, the toolkit's
## NVRTC is the only one its interpreter can load (torch preloads a pip copy when it finds
## one, and an older NVRTC breaks the FMHA JIT), and the driver plus the TRT-LLM
## recipe load against the pinned stack.
RUN <<"EOF" bash -exu
python - <<'PY'
import importlib.util
for name in ("tensorrt_llm", "tensorrt"):
    spec = importlib.util.find_spec(name)
    assert spec and spec.origin, f"{name} is not importable"
    print(f"{name} -> {spec.origin}")
assert not importlib.util.find_spec("tensorrt_llm").origin.startswith("/opt/nemo_rl_venv/")
PY
test -z "$(find /usr/local/lib/python3.12/dist-packages /opt/nemo_rl_venv -name 'libnvrtc.so*')"
export CONTAINER_HF_CKPT_PATH=/dev/null NEMO_GYM_SWE_SIF_DIR=/dev/null \
    NEMO_GYM_SWE_TRAIN_DATA_PATH=/dev/null NEMO_GYM_SWE_VALIDATION_DATA_PATH=/dev/null
uv run --no-sync --project /opt/nemo-rl python - <<'PYEOF'
import torch
import apex
import fused_weight_gradient_mlp_cuda

for name in ("wgrad_gemm_accum_fp16", "wgrad_gemm_accum_fp32"):
    assert callable(getattr(fused_weight_gradient_mlp_cuda, name)), name

import importlib.util
from omegaconf import OmegaConf
from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
from megatron.core.extensions.transformer_engine import HAVE_TE_GDN
from megatron.core.fusions.fused_gated_norm import fused_gated_norm
from megatron.core.fusions.fused_pre_gated_delta_rule import fused_streamed_pre_gated_delta_rule
from megatron.core.transformer import TransformerConfig
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

assert HAVE_TE_GDN, "The selected TE/cuDNN GDN backend is unavailable"
for name in ("gdn_kernel_backend", "gdn_pre_gated_delta_rule_fusion", "gdn_gated_output_norm_fusion"):
    assert name in TransformerConfig.__dataclass_fields__, f"MCore GDN option missing: {name}"
assert all(callable(fn) for fn in (causal_conv1d_bwd_function, fused_gated_norm, fused_streamed_pre_gated_delta_rule))

spec = importlib.util.spec_from_file_location("mlperf_grpo_driver", "/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/run_grpo_nemo_gym.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
register_omegaconf_resolvers()
recipe = "/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm.yaml"
resolved = OmegaConf.to_container(load_config(recipe), resolve=True)
gdn = resolved["policy"]["megatron_cfg"]["model_overrides"]
assert gdn["gdn_kernel_backend"] == "transformer_engine"
assert gdn["gdn_pre_gated_delta_rule_fusion"] is True
assert gdn["gdn_gated_output_norm_fusion"] is True
config = MasterConfig(**resolved)
assert config.grpo.val_at_start is not None
assert config.policy["generation"]["backend"] == "trtllm"
print("pin/driver compatibility check passed")
PYEOF
EOF

