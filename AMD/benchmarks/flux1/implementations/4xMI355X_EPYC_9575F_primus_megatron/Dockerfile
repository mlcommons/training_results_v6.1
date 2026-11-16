# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# MLPerf Training v6.1 flux1 -- Primus (Megatron backend) with MXFP6 GEMMs on MI355X.
#
# Everything is built from public sources at pinned commits on top of the public
# rocm/primus image:
#   aiter        -- A6W6 (MXFP6) GEMM kernels and their tuned table, FMHA v3
#   Primus-Turbo -- MXFP6 packers (Hadamard + E2M3 quantisation) and fused prologues
#   Primus       -- training framework, Flux model, MLPerf recipe and logging
#
# Build:  docker build -t <image> .
# No GPU is needed to build; the GPU architecture is set explicitly below.

ARG BASE_IMAGE=docker.io/rocm/primus:v26.7@sha256:68b7eb7d4db99ecfa18bd7972e5b6d8b78e56219f0a144a8ef5e9a993e57f195
FROM ${BASE_IMAGE}

ARG GPU_ARCHS=gfx950
ENV GPU_ARCHS=${GPU_ARCHS} \
    PYTORCH_ROCM_ARCH=${GPU_ARCHS} \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Hold the image's ROCm builds of torch / TransformerEngine / Triton: every pip install
# below runs under constraints that pin them, and each step re-checks they are unchanged.
COPY scripts/hold_rocm_stack.py /opt/mlperf/hold_rocm_stack.py
RUN python3 /opt/mlperf/hold_rocm_stack.py record /opt/mlperf/constraints.txt /opt/mlperf/held.json
ENV PIP_CONSTRAINT=/opt/mlperf/constraints.txt

# ---- aiter -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_COMMIT=c0164029ebe4fd6480b2151c5ac43fdc6a9b8239
RUN rm -rf /workspace/aiter && git clone ${AITER_REPO} /workspace/aiter \
    && cd /workspace/aiter && git checkout ${AITER_COMMIT} \
    && git submodule update --init --recursive \
    && (pip uninstall -y amd-aiter || true) \
    && AITER_USE_SYSTEM_TRITON=1 pip install --no-build-isolation -e . \
    && python3 /opt/mlperf/hold_rocm_stack.py check /opt/mlperf/held.json "aiter install"

# ---- Primus-Turbo ----------------------------------------------------------------------
ARG TURBO_REPO=https://github.com/AMD-AGI/Primus-Turbo.git
ARG TURBO_COMMIT=33239012d7206e1cb96ef848b005969fa3b5b247
RUN rm -rf /workspace/Primus-Turbo && git clone ${TURBO_REPO} /workspace/Primus-Turbo \
    && cd /workspace/Primus-Turbo && git checkout ${TURBO_COMMIT} \
    && git submodule update --init --recursive \
    && pip install --no-build-isolation -e . \
    && python3 /opt/mlperf/hold_rocm_stack.py check /opt/mlperf/held.json "Primus-Turbo install"

# ---- Primus ----------------------------------------------------------------------------
ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
ARG PRIMUS_COMMIT=d697aa625cf9f1dc04cd9a4c080ac5bbdcdc2575
RUN rm -rf /workspace/Primus && git clone ${PRIMUS_REPO} /workspace/Primus \
    && cd /workspace/Primus && git checkout ${PRIMUS_COMMIT} \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt \
    && pip install --no-build-isolation --no-deps -e third_party/Emerging-Optimizers \
    && python3 /opt/mlperf/hold_rocm_stack.py check /opt/mlperf/held.json "Primus requirements"

# ---- aiter JIT modules used by the Flux path, compiled into the image ------------------
# Importing aiter queries the GPU through rocminfo, and a docker build has no GPU. For this
# step only, a stub rocminfo on the PATH reports the target architecture; it is not kept.
COPY precompile_aiter.py /opt/mlperf/precompile_aiter.py
RUN mkdir -p /tmp/buildbin \
    && printf '#!/bin/sh\necho "  Name:                    %s"\n' "${GPU_ARCHS%%;*}" > /tmp/buildbin/rocminfo \
    && chmod +x /tmp/buildbin/rocminfo \
    && PATH=/tmp/buildbin:${PATH} python3 /opt/mlperf/precompile_aiter.py \
    && rm -rf /tmp/buildbin

# apex (from the base image) compiles amp_C, fused_weight_gradient_mlp_cuda and syncbn on first
# use, which happens while the model is being set up. Compile them here instead, into the
# default extension cache, so a run spends no time building them.
RUN mkdir -p /tmp/buildbin \
    && printf '#!/bin/sh\necho "  Name:                    %s"\n' "${GPU_ARCHS%%;*}" > /tmp/buildbin/rocminfo \
    && chmod +x /tmp/buildbin/rocminfo \
    && PATH=/tmp/buildbin:${PATH} python3 -c "import amp_C, fused_weight_gradient_mlp_cuda, syncbn; [m._load_module() for m in (amp_C, fused_weight_gradient_mlp_cuda, syncbn)]; print('apex extensions built')" \
    && rm -rf /tmp/buildbin

# ---- RCCL-SDMA (opt-in) ------------------------------------------------------------------
# RCCL_SDMA=1 builds the image for a development variant (frameworks/Primus-LM/dev/scratch/mxfp6/pkg_variants/
# config_MI355X_04x08x32_sdma.sh in tiger-training-internal; not a scored config): the distributed optimizer's
# parameter all-gather on a dedicated zero-CTA RCCL copy-engine communicator. It replaces the
# image's RCCL and ROCr with builds from ROCm/rocm-systems at pinned revisions, compiled against
# this image's ROCm SDK (its compilers and bundled rocm_sysdeps libraries) so their sonames match
# what torch loads:
#   RCCL  -- hierarchical CE collectives and the cuMem-probe fix
#   ROCr  -- fixes for imported VMM handles (01d70be664 and related), which RCCL's hierarchical
#            CE path uses; the image's ROCr predates them and faults on that path.
# RCCL_SDMA=0 (default) leaves the image's RCCL and ROCr untouched.
ARG RCCL_SDMA=0
ARG RCCL_REPO=https://github.com/ROCm/rocm-systems.git
ARG RCCL_REV=d75507948fa94106ea2adf5c158af31e6fbda6ef
ARG RCCL_MAX_JOBS=96
RUN set -eux; \
    if [ "${RCCL_SDMA}" != 1 ]; then exit 0; fi; \
    SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel; \
    export PATH="${SDK}/bin:/opt/venv/bin:${PATH}" ROCM_PATH="${SDK}"; \
    git clone --filter=blob:none --sparse "${RCCL_REPO}" /tmp/rccl-src; \
    git -C /tmp/rccl-src sparse-checkout set projects/rccl; \
    git -C /tmp/rccl-src fetch origin "${RCCL_REV}"; \
    git -C /tmp/rccl-src checkout --detach "${RCCL_REV}"; \
    grep -q "symAvailable && cfgAllowsSymk" /tmp/rccl-src/projects/rccl/src/scheduler/symmetric_sched.cc; \
    grep -q "static const int runtimeSupported" /tmp/rccl-src/projects/rccl/src/misc/rocmwrap.cc; \
    mkdir -p /tmp/rccl-build && cd /tmp/rccl-build; \
    cmake -G Ninja /tmp/rccl-src/projects/rccl \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${SDK}/bin/amdclang++" \
        -DCMAKE_C_COMPILER="${SDK}/bin/amdclang" \
        -DCMAKE_PREFIX_PATH="${SDK}" \
        -DROCM_PATH="${SDK}" \
        -DGPU_TARGETS="${GPU_ARCHS}" \
        -DBUILD_TESTS=OFF \
        -DROCTX=ON; \
    ninja -j "${RCCL_MAX_JOBS}"; \
    DST=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librccl.so.1; \
    test -f "${DST}"; \
    cp -L /tmp/rccl-build/librccl.so.1 "${DST}"; \
    test "$(strings "${DST}" | grep -c 'Hierarchical CE')" -gt 0; \
    readelf -d "${DST}" | grep -E 'NEEDED.*(amd_smi|amdhip64|hsa-runtime)'; \
    rm -rf /tmp/rccl-src /tmp/rccl-build

ARG ROCR_REPO=https://github.com/ROCm/rocm-systems.git
ARG ROCR_REV=2cd17fbbdc252291da3d922e5d192b9242a9fdeb
RUN set -eux; \
    if [ "${RCCL_SDMA}" != 1 ]; then exit 0; fi; \
    SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel; \
    SYSDEPS="${SDK}/lib/rocm_sysdeps"; \
    export PATH="${SDK}/bin:${SDK}/lib/llvm/bin:/opt/venv/bin:${PATH}" ROCM_PATH="${SDK}"; \
    export PKG_CONFIG_PATH="${SYSDEPS}/lib/pkgconfig"; \
    git clone --filter=blob:none --sparse "${ROCR_REPO}" /tmp/rocr-src; \
    git -C /tmp/rocr-src sparse-checkout set projects/rocr-runtime; \
    git -C /tmp/rocr-src fetch origin "${ROCR_REV}"; \
    git -C /tmp/rocr-src checkout --detach "${ROCR_REV}"; \
    mkdir -p /tmp/rocr-build && cd /tmp/rocr-build; \
    cmake -G Ninja /tmp/rocr-src/projects/rocr-runtime \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${SDK};${SDK}/lib/llvm;${SYSDEPS}" \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN:$ORIGIN/rocm_sysdeps/lib:$ORIGIN/../../_rocm_sdk_core/lib/llvm/lib:$ORIGIN/../../_rocm_sdk_core/lib/rocm_sysdeps/lib'; \
    ninja -j "${RCCL_MAX_JOBS}"; \
    SRC=$(find /tmp/rocr-build -name 'libhsa-runtime64.so.1.*' -type f | head -1); \
    test -n "${SRC}"; \
    readelf -d "${SRC}" | grep -E 'SONAME.*libhsa-runtime64\.so\.1'; \
    readelf -d "${SRC}" | grep -E 'NEEDED.*librocm_sysdeps_(drm|drm_amdgpu|elf|numa)\.so' | wc -l | grep -qx 4; \
    DST=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1; \
    test -e "${DST}"; \
    cp "${SRC}" "$(readlink -f "${DST}")"; \
    if ldd "${DST}" | grep 'not found'; then exit 1; fi; \
    rm -rf /tmp/rocr-src /tmp/rocr-build

# Record exactly what was built, for the system description and for reviewers.
RUN { echo "aiter=$(git -C /workspace/aiter rev-parse HEAD)"; \
      echo "Primus-Turbo=$(git -C /workspace/Primus-Turbo rev-parse HEAD)"; \
      echo "Primus=$(git -C /workspace/Primus rev-parse HEAD)"; \
      if [ "${RCCL_SDMA}" = 1 ]; then echo "rccl=${RCCL_REPO}@${RCCL_REV}"; echo "rocr=${ROCR_REPO}@${ROCR_REV}"; fi; \
      cat /opt/mlperf/held.json; } > /opt/mlperf/stack.txt

# ---- Submission code ---------------------------------------------------------------------
WORKDIR /workspace/code
COPY . .
RUN pip install -r requirements.txt \
    && python3 /opt/mlperf/hold_rocm_stack.py check /opt/mlperf/held.json "submission requirements"

