# Classification: Internal (CONFIDENTIAL until MLCommons publication)
#
# Firmus MLPerf Training v6.1 -- llama31_8b, workstream L4 v3 "Kernel opt NVFP4 v3 (aggressive, no gather overlap)"
#   system_desc_id  firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3
#   implementation  nemo_l4_nvfp4_v3      (this file goes to Firmus/benchmarks/llama31_8b/implementations/nemo_l4_nvfp4_v3/)
#
# DRAFT 2026-10-09.  Not yet built.  See llama31/2026_10_08_AGGRESSIVE_SUBMISSION.md (Phase A/B) for how to build it
# and llama31/KERNEL_SUPPLEMENTARY_INFORMATION_L4.md for what every Firmus kernel replaces and why it is equivalent.
#
# What this is: NVIDIA's own llama31_8b Dockerfile of the 20261006 release, VERBATIM except for the lines marked
# "FIRMUS" below.  NVIDIA's file was read from nvdlfwea+mlperftv61+llama31_8b-arm+20261006.sqsh
# (sha256 7f139c9b12a06b128b0abaf6e617f3110ac5e9cdf0d572cdefac2005f0443da6) at /workspace/llm/Dockerfile; it differs
# from the 20260929 release in one line only (TE_REVISION 131157b5 -> v2.20.0).  The image NVIDIA provides
# (nvcr.io/nvdlfwea/mlperftv61/llama31_8b-arm:20261006) is NVIDIA's own build of this same Dockerfile; it is our
# reference, not our base (SR-15, gate G3; gap B-3, unblocked 2026-10-09): this file rebuilds the same stack from the
# public NGC PyTorch image -- every input verified public on 2026-10-09 -- and the Firmus kernels are compiled here
# from the source in ./firmus/ (KERNEL_CODE_IN_DOCKERFILE.md).
#
# FIRMUS changes to NVIDIA's file, in order:
#   F0  the base image pinned by digest; Megatron-Bridge and Transformer Engine pinned by commit, not by tag; the two
#       git dependencies of requirements.txt likewise (recipe diff 0003)
#       (the same commits NVIDIA's 20261006 image records in /MBRIDGE_COMMIT_HASH.env and /TE_COMMIT_HASH.env)   SR-9
#   F1  Megatron-Core patches (./firmus/patches/mcore/*.patch) applied before Megatron-LM's editable install
#   F2  Transformer Engine patches (./firmus/patches/te/*.patch) applied after the checkout, before TE is built
#   F3  two of the recipe's own files are edited in this directory (nothing in the build applies a diff to them):
#       embedding_lib/custom_embedding/custom_embedding.py (emb2) and pretrain.py, whose first import is `firmus_ak`
#       (gap B-4: the training program enables the Firmus kernels explicitly; no .pth auto-loading, no argv test).
#       The diffs against NVIDIA's recipe are ./firmus/patches/recipe/*.diff (documentation)
#   F5  section 5: the Firmus CUDA/C++ extensions compiled from ./firmus/ak/csrc, the Python kernels (Triton) and the
#       patch modules installed as the package `firmus_ak`; build fails if any expected output is missing
#
# Nothing is mounted into the container at run time except data, results and the tokenizer (checklist G3).

# F0: nvcr.io/nvidia/pytorch:26.09-py3, public NGC.  Index digest read from nvcr.io on 2026-10-09:
#     sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6
#     (arm64 manifest sha256:307c449df7dabed6b36e017a2b2f2e45bdec9dc203b81507111315f1e971de6e)
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6
FROM ${FROM_IMAGE_NAME}

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

WORKDIR /workspace/

RUN pip install numcodecs==0.16.3
RUN pip install nvidia-mathdx==25.1.1

# FIRMUS F1/F2: the patch files must exist before sections 1 and 2 build the libraries they change.
COPY firmus/patches/ /workspace/firmus_patches/

## 1. Megatron-Bridge and Megatron-Core
# FIRMUS F0: NVIDIA's 20261006 file says MBRIDGE_REVISION=26.09-alpha.rc1; that tag is this commit (checked on
# github.com on 2026-10-09, and /MBRIDGE_COMMIT_HASH.env of the 20261006 image).  Megatron-LM is its submodule at
# f6c33bde4cdc10805ee7157db746d47b758f5f02 (/MCORE_COMMIT_HASH.env), asserted below.
ARG MBRIDGE_REVISION=ea5dded83078cf7e61ef91960273053cf564d5ac
ARG MCORE_EXPECTED=f6c33bde4cdc10805ee7157db746d47b758f5f02
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge && \
    cd /workspace/Megatron-Bridge && \
    git checkout ${MBRIDGE_REVISION} && \
    git submodule update --init --recursive && \
    echo MBRIDGE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MBRIDGE_COMMIT_HASH.env && \
    cd /workspace/Megatron-Bridge/3rdparty/Megatron-LM && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    test "$(git rev-parse HEAD)" = "${MCORE_EXPECTED}" && \
    # FIRMUS F1: distrib_optimizer / clip_grads (X9), distributed_data_parallel / param_and_grad_buffer (X7),
    #            extensions/transformer_engine.py (TCE + TCEV).  git apply refuses a patch that does not apply cleanly.
    for p in /workspace/firmus_patches/mcore/*.patch; do git apply --verbose "$p" || exit 1; done && \
    git diff --stat > /FIRMUS_MCORE_PATCHES.txt && \
    pip install --no-build-isolation -e .

ENV PYTHONPATH="/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"

## 2. Transformer Engine
# FIRMUS F0: NVIDIA's 20261006 file says TE_REVISION=v2.20.0; that tag is this commit (/TE_COMMIT_HASH.env of the
# 20261006 image; `git tag --points-at` gives v2.20 and v2.20.0).
ARG TE_REVISION=5b28d7aa2bd639ff0541356cc025f62954f96ce3
ENV CUSTOM_TE_REVISION ${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    export MAX_JOBS=$(( $(nproc) > 24 ? 24 : $(nproc) )); \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
    cd transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    # FIRMUS F2: transformer_engine/pytorch/attention/rope.py and .../dot_product_attention/utils.py (X12 + QKV pack).
    { for p in /workspace/firmus_patches/te/*.patch; do git apply --verbose "$p" || exit 1; done; } && \
    git diff --stat > /FIRMUS_TE_PATCHES.txt && \
    pip uninstall -y flash-attn-4 quack-kernels && \
    git submodule update --init --recursive && \
    pip install "nvidia-cudnn-frontend==1.29.0" && \
    NVTE_BUILD_NUM_PHILOX_ROUNDS=3 NVTE_CUDA_ARCHS="100a;103a;107a" NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1 NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi
# NOTE: /workspace/transformerengine (the source tree, incl. its 3rdparty/cutlass) is kept, as in NVIDIA's image; the
# Firmus extensions rht7 / rht8 / w2d7 compile against its headers in section 5.


## 3. Benchmark dependencies
COPY requirements.txt .
RUN apt-get remove -y python3-yaml && pip uninstall -y transformers torchao && pip install --no-cache-dir --no-build-isolation -r requirements.txt

## 4. Benchmark code
WORKDIR /workspace/llm
COPY . .
RUN cd /workspace/llm/embedding_lib && pip install --force-reinstall --no-deps --no-build-isolation .

## 5. FIRMUS F5: auto_kernel kernels for L4 (source: ./firmus/ak/, built here, nothing prebuilt is copied)
#
# The extension list, per-module nvcc flags and arch, and the expected outputs are in ./firmus/ak/build_manifest.json;
# build_all.py builds each with torch.utils.cpp_extension and exits non-zero if any expected file is missing.
# Per-module flags are the ones the validated modules were built with (L4_SOURCE_MANIFEST.tsv); none adds
# --use_fast_math except ak_l8b_nvfp4p (SwiGLU/dSwiGLU), which matches TE's NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1.
ARG FIRMUS_AK_PREFIX=/opt/firmus/ak
ENV FIRMUS_AK_PREFIX=${FIRMUS_AK_PREFIX}
# Where each patch module finds its compiled extension or kernel file.  Set in the image itself, so every process in
# the container has them.  The code's own defaults (used when a variable is unset) are the same /opt/firmus/ak paths
# (gap B-5, fixed 2026-10-09: no /home, /shared or /vast path anywhere in the Firmus code); check_install below fails
# the build if any variable is unset or resolves outside ${FIRMUS_AK_PREFIX}, or if any Firmus file names those paths.
# The switches that turn the kernels ON are not set here: they are in config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh
# (SR-10 "Enablement").
ENV AK_X12_KERNEL_DIR=${FIRMUS_AK_PREFIX}/x12 \
    AK_EMB_KERNEL_DIR=${FIRMUS_AK_PREFIX}/emb \
    AK_X9_KERNEL_DIR=${FIRMUS_AK_PREFIX}/x9 \
    AK_NVFP4P_EXT_DIR=${FIRMUS_AK_PREFIX}/lib \
    AK_NVFP4P_W2D_EXT_DIR=${FIRMUS_AK_PREFIX}/lib \
    AK_NVFP4P_NORMCAST_EXT_DIR=${FIRMUS_AK_PREFIX}/lib \
    AK_NVFP4P_RHT8_EXT_DIR=${FIRMUS_AK_PREFIX}/lib \
    AK_NVFP4P_BX_EXT_DIR=${FIRMUS_AK_PREFIX}/lib \
    AK_FP8CS_EXT_DIR=${FIRMUS_AK_PREFIX}/lib \
    AK_FP4G_EXT_DIR=${FIRMUS_AK_PREFIX}/lib \
    AK_LMH_DIR=${FIRMUS_AK_PREFIX}/lmh
RUN cd /workspace/llm/firmus/ak && \
    MAX_JOBS=16 TE_SRC=/workspace/transformerengine \
    python3 build_all.py --manifest build_manifest.json --out ${FIRMUS_AK_PREFIX}/lib && \
    python3 build_all.py --manifest build_manifest.json --out ${FIRMUS_AK_PREFIX}/lib --verify-only && \
    rm -rf /tmp/firmus_ak_build
# (the build tree is /tmp/firmus_ak_build, outside the source directory, and is removed once the outputs verify)
# build_all.py also lays out the kernel directories the patches read: ${FIRMUS_AK_PREFIX}/x12 (the Triton RoPE/QKV
# kernels + ak_l8b_qkvx.so), /emb (Triton embedding gradient), /x9/harness/tier2/x9_adamw (Triton AdamW + norm),
# /lmh (ak_l8b_lmh.so + lmh_pin.json), /lib (every other extension).  Triton kernels are JIT-compiled at run time from
# these source files (allowed: KERNEL_CODE_IN_DOCKERFILE.md "JIT-compiled kernels"; triton is the base image's).
# The eight patch modules install as the package `firmus_ak`, with NO .pth file (gap B-4).  pretrain.py's first
# import, `import firmus_ak`, loads them in the order the validated runs' zz-ak-l8b-*.pth files did (site.py's sorted
# order: fp4g, fp8cs, gsa, lmh, nvfp4p, opswz, projcol, smallk2), before torch / TE / Megatron are imported.  Each arms
# on its AK_* switch alone.  check_install: no .pth of ours in site-packages; no /home, /shared or /vast literal in firmus_ak; every AK_*_DIR variable set and inside
# ${FIRMUS_AK_PREFIX}; every expected extension and kernel file present.  It runs as a plain script (standard library
# only): importing the package runs firmus_ak.banner, which needs the records written below.
# (pip builds in the directory it is given, so it installs from a copy and the source tree stays as shipped)
RUN cp -a /workspace/llm/firmus/ak/pkg /tmp/firmus_ak_pkg && \
    pip install --no-deps --no-build-isolation /tmp/firmus_ak_pkg && rm -rf /tmp/firmus_ak_pkg && \
    python3 /workspace/llm/firmus/ak/pkg/firmus_ak/check_install.py --prefix ${FIRMUS_AK_PREFIX}
# Provenance recorded in the image beside NVIDIA's *_COMMIT_HASH.env files, and the records firmus_ak.banner re-checks
# on every rank at the start of every run (it raises on any difference):
#   /FIRMUS_AK_BUILD_SHA256.txt  every file under ${FIRMUS_AK_PREFIX} (the built extensions, Triton kernels, lmh pins)
#   /FIRMUS_STACK_SHA256.txt     every installed upstream / recipe file a Firmus patch or recipe edit changed; the build
#                                fails if one is missing or does not carry the Firmus change
#   /FIRMUS_AK_SOURCE_SHA256.txt the Firmus source tree this image was built from
RUN set -e; cd ${FIRMUS_AK_PREFIX}; \
    find . -type f -not -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum > /FIRMUS_AK_BUILD_SHA256.txt; \
    SP=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    ML=/workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core; \
    CE=$(cd / && python3 -c 'import importlib.util as u; print(u.find_spec("custom_embedding").submodule_search_locations[0])')/custom_embedding.py; \
    STACK="$SP/transformer_engine/pytorch/attention/rope.py $SP/transformer_engine/pytorch/attention/dot_product_attention/utils.py \
           $ML/optimizer/distrib_optimizer.py $ML/optimizer/clip_grads.py $ML/distributed/distributed_data_parallel.py \
           $ML/distributed/param_and_grad_buffer.py $ML/extensions/transformer_engine.py \
           /workspace/llm/embedding_lib/custom_embedding/custom_embedding.py $CE /workspace/llm/pretrain.py"; \
    for f in $STACK; do test -f "$f" || { echo "FIRMUS: $f missing"; exit 1; }; \
        grep -qE 'auto_kernel|firmus_ak' "$f" || { echo "FIRMUS: $f does not carry the Firmus change"; exit 1; }; done; \
    sha256sum $STACK > /FIRMUS_STACK_SHA256.txt; \
    cd /workspace/llm/firmus && find . -type f -not -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum > /FIRMUS_AK_SOURCE_SHA256.txt; \
    echo "FIRMUS_AK_SOURCE_TREE_SHA256=$(sha256sum /FIRMUS_AK_SOURCE_SHA256.txt | cut -c1-64)" > /FIRMUS_AK.env; \
    echo "FIRMUS_AK_BUILD_SHA256=$(sha256sum /FIRMUS_AK_BUILD_SHA256.txt | cut -c1-64)" >> /FIRMUS_AK.env; \
    echo "FIRMUS_STACK_SHA256=$(sha256sum /FIRMUS_STACK_SHA256.txt | cut -c1-64)" >> /FIRMUS_AK.env; \
    cat /FIRMUS_AK.env /FIRMUS_STACK_SHA256.txt
