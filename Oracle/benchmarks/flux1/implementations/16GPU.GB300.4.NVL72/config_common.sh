# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export HYDRA_FULL_ERROR=1
export TQDM_DISABLE=True

export EXP_NAME=${EXP_NAME:-flux-train-$(date +%y%m%d%H%M%S%N)}
# Target evaluation frequency in samples (rounded to a whole step by GBS)
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}
export LOG_EVERY_N_STEPS=${LOG_EVERY_N_STEPS:-16}

export HF_HUB_OFFLINE=1 # disable network requests for HF transfers

export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0
export NCCL_PAT_ENABLE=1  # Pattern-Aware Transport

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor_cache}  # Node-local: avoids Lustre stale file handles

# Data loading
export NUM_WORKERS=${NUM_WORKERS:-4}  # Fewer workers reduces CPU thread contention

export MODEL_TFLOP_PER_SAMPLE=20.3

# Precision flags read by the code; the MLPERF_* variables below are derived
# from them and are logging only.
#
# FP8            MXFP8 (OCP microscaling, E4M3) linear GEMMs.
# FP4            NVFP4 linear GEMMs, stochastic rounding on the casts, RHT on the wgrad path.
# FP4_SCOPE      grad = NVFP4 for dgrad/wgrad only (ships); all = NVFP4 forward too.
# FP8_ATTENTION  MXFP8 attention core; off = bf16 attention (ships).
export FP8=${FP8:-True}
export FP4=${FP4:-True}
export FP4_SCOPE=${FP4_SCOPE:-grad}
export FP8_ATTENTION=${FP8_ATTENTION:-False}

# lowest_numerical_precision_in_* disclosure values, emitted by utils/mlperf_logger.py.
# linear: forward GEMM format ("fp8" is the allow-list token; there is no "mxfp8").
# grad:   dgrad/wgrad format; disclosed for linear when it is the lower one.
# comm:   the replicate all-reduce runs in bf16.
if [[ "${FP4}" == "True" && "${FP4_SCOPE}" == "all" ]]; then
    export MLPERF_LINEAR_PRECISION="nvfp4"
elif [[ "${FP8}" == "True" ]]; then
    export MLPERF_LINEAR_PRECISION="fp8"
else
    export MLPERF_LINEAR_PRECISION="bfloat16"
fi
if [[ "${FP4}" == "True" ]]; then
    export MLPERF_GRAD_PRECISION="nvfp4"
else
    export MLPERF_GRAD_PRECISION="${MLPERF_LINEAR_PRECISION}"
fi
if [[ "${FP8_ATTENTION}" == "True" ]]; then
    export MLPERF_ATTN_PRECISION="fp8"
else
    export MLPERF_ATTN_PRECISION="bfloat16"
fi
export MLPERF_COMM_PRECISION="bfloat16"
