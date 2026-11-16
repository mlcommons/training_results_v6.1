# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=rocm/primus:v26.5@sha256:3040bf42974d791dd42de2e36b3c919a00869a5754cfc57a06b96d004c55eed1
FROM ${BASE_IMAGE}

ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
ARG PRIMUS_REF=e5e32eced45946f2c57f625c5d3c1958f129ea98
ARG PRIMUS_TURBO_REPO=https://github.com/AMD-AGI/Primus-Turbo.git
ARG PRIMUS_TURBO_REF=c3503f6cd9762b54406ecb30ee2a34322ffcd4b8
ARG TE_REF=a07e607f14a5330807ffdafeeb6224f2d7dffacc
ARG AITER_REF=d32b0cb62ecc32bb1a858e8437d58eb9b3856af6
ARG MLPERF_COMMON_REF=7771aeabc8b91402e95cb7a33f447d9d9941361f
ARG MAX_JOBS=96

WORKDIR /workspace

RUN rm -rf /workspace/Primus && \
    git clone --recursive "${PRIMUS_REPO}" /workspace/Primus && \
    cd /workspace/Primus && \
    git fetch origin "${PRIMUS_REF}" && \
    git checkout --detach "${PRIMUS_REF}" && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache

# Replace the base image's primus-mllog with the one from mlperf-common.
RUN python3 -m pip uninstall -y primus-mllog 2>/dev/null || true && \
    python3 -m pip install --no-cache-dir \
      "git+https://github.com/AMD-AGI/mlperf-common.git@${MLPERF_COMMON_REF}"

WORKDIR /workspace/deps

# TransformerEngine with the CK fused-attention backend, built for gfx950.
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git && \
    cd TransformerEngine && git checkout "${TE_REF}" && \
    git submodule update --init --recursive && \
    sed -i "s|^aiter_commit = .*|aiter_commit = \"${AITER_REF}\"|" \
      transformer_engine/common/ck_fused_attn/qola_manifest.toml && \
    PYTHONPATH="$PWD/3rdparty/QoLA:${PYTHONPATH}" \
      python3 -m qola.cli checkout \
        --manifest transformer_engine/common/ck_fused_attn/qola_manifest.toml \
        --aiter-root /workspace/deps/te-aiter && \
    python3 -m pip uninstall -y \
      transformer_engine transformer_engine_rocm7 transformer_engine_rocm_torch transformer-engine || true && \
    NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_FUSED_ATTN_CK=1 \
    NVTE_CK_JIT=1 \
    NVTE_CK_FUSED_ATTN_PATH="" \
    NVTE_AITER_SOURCE_DIR=/workspace/deps/te-aiter \
    NVTE_BUILD_MAX_JOBS="${MAX_JOBS}" \
    NVTE_FRAMEWORK=pytorch \
    NVTE_ROCM_ARCH=gfx950 \
    NVTE_USE_HIPBLASLT=1 \
    PYTORCH_ROCM_ARCH=gfx950 \
    CU_NUM=304 \
    NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \
    CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
    MAX_JOBS="${MAX_JOBS}" \
      python3 -m pip install --no-build-isolation --no-deps -v . && \
    cd .. && rm -rf TransformerEngine te-aiter

RUN git clone --recursive "${PRIMUS_TURBO_REPO}" /workspace/deps/Primus-Turbo && \
    cd /workspace/deps/Primus-Turbo && \
    git fetch origin "${PRIMUS_TURBO_REF}" && \
    git checkout --detach "${PRIMUS_TURBO_REF}" && \
    git submodule update --init --recursive && \
    python3 -m pip install -r requirements.txt && \
    PRIMUS_TURBO_FRAMEWORK=PYTORCH \
    GPU_ARCHS=gfx950 \
    MAX_JOBS="${MAX_JOBS}" \
      python3 -m pip install --no-build-isolation --no-deps -v .

WORKDIR /workspace

# Patches to the Megatron-Bridge and Megatron-LM submodules, shipped in the Primus recipe.
RUN P=/workspace/Primus/primus/backends/megatron_bridge/recipes/mlperf_llama2_70b/source_patches && \
    git config --global --add safe.directory /workspace/Primus && \
    git config --global --add safe.directory /workspace/Primus/third_party/Megatron-Bridge && \
    cd /workspace/Primus/third_party/Megatron-Bridge && \
    git apply "$P/megatron_nemo_lora_only.patch" && \
    git apply "$P/megatron_bridge_validation_consumed_samples.patch" && \
    git apply "$P/megatron_bridge_deterministic_eval.patch" && \
    git apply "$P/sft_attention_mask_cache.patch" && \
    cd /workspace/Primus/third_party/Megatron-Bridge/3rdparty/Megatron-LM && \
    git apply "$P/megatron_lm_mxfp4_recipe.patch" && \
    git apply "$P/megatron_lm_tegrouped_mlp_submodules.patch"

# Editable install, so the patched Megatron-Bridge source is what runs.
RUN cd /workspace/Primus/third_party/Megatron-Bridge && \
    pip install -e . --no-deps --no-build-isolation

# Pre-install the Megatron-Bridge posttrain hook's dependencies so the hook is mostly a no-op.
RUN pip install "onnx==1.20.0rc1" "datasets>=2.14.0" "transformers==4.57.6" \
    "qwen-vl-utils" "timm" "open-clip-torch>=3.2.0" "flash-linear-attention" \
    "megatron-energon" "bitstring" "filetype" \
    && pip install nvidia_resiliency_ext || true \
    && rm -rf /root/.cache

# Megatron-Bridge imports modelopt at module level; stub it.
RUN mkdir -p /opt/venv/lib/python3.12/site-packages/modelopt/torch/distill/plugins \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/opt/plugins \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/quantization \
    && touch /opt/venv/lib/python3.12/site-packages/modelopt/__init__.py \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/__init__.py \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/distill/__init__.py \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/distill/plugins/__init__.py \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/quantization/__init__.py \
    && printf '%s\n' \
       'def get_tensor_shapes_adjust_fn_for_distillation(*a, **kw): return None' \
       > /opt/venv/lib/python3.12/site-packages/modelopt/torch/distill/plugins/megatron.py \
    && printf '%s\n' \
       'def save_modelopt_state(*a, **kw): pass' \
       'def save_sharded_modelopt_state(*a, **kw): pass' \
       'def restore_sharded_modelopt_state(*a, **kw): pass' \
       'def restore_modelopt_state(*a, **kw): pass' \
       > /opt/venv/lib/python3.12/site-packages/modelopt/torch/opt/__init__.py \
    && cp /opt/venv/lib/python3.12/site-packages/modelopt/torch/opt/__init__.py \
          /opt/venv/lib/python3.12/site-packages/modelopt/torch/opt/plugins/__init__.py

WORKDIR /workspace/code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Model preset and recipe modules; run_and_time.sh copies them again from /workspace/code.
RUN cp src/llama2_70b_lora.yaml /workspace/Primus/primus/configs/models/megatron_bridge/llama2_70b_lora.yaml && \
    for d in /workspace/Primus/primus/recipes \
             /workspace/Primus/primus/backends/megatron_bridge/recipes/mlperf_llama2_70b; do \
      mkdir -p "$d" && \
      cp src/llama2_custom.py src/_log_suppression.py "$d/" || exit 1; \
    done
