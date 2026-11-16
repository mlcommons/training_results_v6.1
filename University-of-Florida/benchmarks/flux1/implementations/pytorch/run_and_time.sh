#!/bin/bash
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

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HOME=/hf_home

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}"

if [[ "${USE_SYNTHETIC_DATA:-0}" -eq 1 ]]; then
    echo "Synthetic data is not supported by train_flat.py" >&2
    exit 1
fi

if [[ -z "${SEED}" ]]; then
    echo "SEED is not set!"
    exit 1
fi

: "${BATCHSIZE:?BATCHSIZE not set}"
: "${LEARNING_RATE:?LEARNING_RATE not set}"
: "${WARMUP_STEPS:?WARMUP_STEPS not set}"
: "${VAL_CHECK_INTERVAL:?VAL_CHECK_INTERVAL not set}"
: "${LOG_EVERY_N_STEPS:?LOG_EVERY_N_STEPS not set}"

if [ "${PROFILE:-0}" -gt 0 ]; then
    export NSYS_START_STEP="${NSYS_START_STEP:-3}"
    export NSYS_END_STEP="${NSYS_END_STEP:-5}"
    NSYSCMD="nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --cuda-graph-trace=node --cuda-event-trace=false --backtrace=none --cudabacktrace=false --capture-range=cudaProfilerApi --capture-range-end=repeat --kill=none --stats=false --export=sqlite --force-overwrite=true --output ${NSYS_PREFIX}${SLURM_PROCID}${NSYS_SUFFIX}"
    if [ "${NSYS_METRICS:-0}" -gt 0 ]; then
        NSYSCMD="${NSYSCMD} --gpu-metrics-devices=${local_rank} --gpu-metrics-set=${NSYS_METRICS_SET} --gpu-metrics-frequency=50000"
    fi
fi

declare -a CMD
if [[ "${LOCAL_WORLD_SIZE}" -gt 1 ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( ${NSYSCMD} 'python' '-u')
else
    # interactive run on single node, no need to bind
    CMD=( ${NSYSCMD} 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi

: "${LOGGER:=""}"
if [[ -n "${APILOG_DIR:-}" ]]; then
    if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]; then
      LOGGER="apiLog.sh -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
    fi
fi

# Assert $RANDOM is usable
if [[ -z "${RANDOM}" ]]; then
    echo "RANDOM is not set!" >&2
    exit 1
fi

if [[ "${node_rank}" -eq 0 && "${local_rank}" -eq 0 ]]; then
    echo "SEED=${SEED}"
fi


# Virtual NVLink clique: split the physical NVLink domain so that ranks in
# different cliques must talk over IB. Lets a small allocation exercise the
# D>1 inter-domain path that would otherwise need multiple racks. Unset =>
# natural physical cliques (no-op). Must agree with SEGMENT: the clique size
# in GPUs should equal SEGMENT * DGXNGPU.
if [ -n "${FLAT_CLIQUE_GPUS:-}" ]; then
    export NCCL_MNNVL_CLIQUE_ID=$(( ${SLURM_PROCID:-0} / FLAT_CLIQUE_GPUS ))
    if [ "${SLURM_PROCID:-0}" = "0" ]; then
        echo "VIRTUAL_CLIQUE: FLAT_CLIQUE_GPUS=${FLAT_CLIQUE_GPUS} rank0 cliqueId=${NCCL_MNNVL_CLIQUE_ID}"
    fi
fi

# NVLS multicast AllGather for the intra-domain parameter all-gather.
# Only meaningful when there is more than one NVLink domain (D>1); the code
# gates FLAT_NVLS_AG_REGISTER on is_hierarchical, and we gate the NCCL knobs
# here too so single-domain configs (small/med) are bit-for-bit untouched.
#
# These are exported INSIDE the container (run_and_time.sh is launched via
# `slurm2pytorch ./run_and_time.sh` under --container-name), so they do not
# need to appear in run.sub's srun --container-env allowlist -- pyxis only
# drops *host* env vars that were never set in the container.
#
# NCCL_WIN_ENABLE + NCCL_CUMEM_ENABLE are both required for
# comm->symmetricSupport, without which ncclCommWindowRegister cannot build
# the symmetric windows and the multicast AllGather kernel is never dispatched.
# NCCL_SYM_CE_THRESHOLD selects copy-engine vs STMC multicast BY MESSAGE SIZE:
# messages above it use the copy engine (O(N) peer copies), below it use the
# multimem multicast kernel. Our bucket all-gathers are 100s of MB, so the
# small default routes them to the copy engine; setting the threshold above
# the largest all-gather forces every one onto the multicast path.
_segment="${SEGMENT:-0}"
if [ "${_segment}" -gt 0 ] 2>/dev/null; then
    _n_domains=$(( DGXNNODES / _segment ))
else
    _n_domains=1
fi
if [ "${_n_domains}" -gt 1 ]; then
    export FLAT_NVLS_AG_REGISTER=${FLAT_NVLS_AG_REGISTER:-1}
    export NCCL_WIN_ENABLE=${NCCL_WIN_ENABLE:-1}
    export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-1}
    export NCCL_SYM_CE_THRESHOLD=${NCCL_SYM_CE_THRESHOLD:-1099511627776}
    if [[ "${node_rank}" -eq 0 && "${local_rank}" -eq 0 ]]; then
        echo "NVLS_AG: D=${_n_domains} FLAT_NVLS_AG_REGISTER=${FLAT_NVLS_AG_REGISTER}" \
             "NCCL_WIN_ENABLE=${NCCL_WIN_ENABLE} NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE}" \
             "NCCL_SYM_CE_THRESHOLD=${NCCL_SYM_CE_THRESHOLD}"
    fi
fi

${LOGGER:-} ${CMD[@]} train_flat.py \
    --dataset-root /dataset/energon \
    --max-steps "${MAX_STEPS:-10000}" \
    --seed "${SEED}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --results-dir /results \
    --num-nodes "${DGXNNODES}" \
    --learning-rate "${LEARNING_RATE}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --batch-size "${BATCHSIZE}" \
    --val-check-interval "${VAL_CHECK_INTERVAL}" \
    --log-every-n-steps "${LOG_EVERY_N_STEPS}" \
    --segment-nodes "${SEGMENT:-0}" \
    --replicate-inflight "${HSDP_REPLICATE_INFLIGHT:-32}"

