#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=8:00:00
#SBATCH --job-name=bruno-gptoss-v265
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# AMD's MLPerf GPT-OSS-20B recipe exactly as shipped INSIDE rocm/primus:v26.5
# (/workspace/Primus/examples/mlperf/gpt_oss_20b): FP8, GBS 32. Code and
# libraries come from the same image, so they cannot mismatch.
#
# Why not Primus main: its launcher needs newer primus_turbo than v26.5
# (gemm_fp8() fuse_bgrad_accum_pattern TypeError), and its rccl_sdma param
# gather fails on lodin (HIP 'invalid argument' in RCCL p2p transport).
#
# /data must be writable: Megatron writes its sample-index cache under
# <prefix>/cache/GPTDataset_indices/ (hashed names, safe next to llama31's).
#
# Tuning overrides (both unconstrained by the rules for gpt_oss_20b):
#   LR=<base lr>          end LR is forced to LR*0.1 (rule-fixed)
#   WARMUP_ITERS=<iters>  decay iters forced to 1,200,000 - warmup (rule-fixed)
# Logging override: MLLOG_TRAIN_LOSS_LOG_FREQ so speed can be measured.
# Seed: from /dev/urandom (allowed source), 31-bit, so repeats across runs are
# practically impossible (bash $RANDOM has only 32,768 values).
#
# Paths: since the 2026-10-02 maintenance the compute nodes mount the NAS
# root at /nas, so striker's /nas/SHARED/MLPERF is /nas/data/SHARED/MLPERF
# there. Detected below. Slurm logs go to /home (same path everywhere).
#
# Newer Primus: set CONT (e.g. rocm/primus:v26.7) and optionally PRIMUS_HOST, a
# host Primus checkout mounted over the image's /workspace/Primus (e.g. a
# patched copy). Without PRIMUS_HOST the image's own recipe is used.
#
# Usage: sbatch --export=ALL,TAG=x,LR=6e-4[,CONT=..,PRIMUS_HOST=..] run_amd_primus_v265.sbatch [NEXP]
set -uo pipefail

if [[ -d /nas/data/SHARED/MLPERF/results ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
export NEXP=${1:-${NEXP:-1}}
CONT=${CONT:-rocm/primus:v26.5}
DATADIR=${DATADIR:-/lvol0/mlperf/llama31_8b/data}
MODELDIR=${MODELDIR:-/lvol0/mlperf/llama31_8b/model}
LOGDIR=${LOGDIR:-${NAS}/results/gpt_oss_20b/${TAG:-amd-v265}}
LR=${LR:-}
WARMUP_ITERS=${WARMUP_ITERS:-}
PRIMUS_HOST=${PRIMUS_HOST:-}
MNT=(); [[ -n "${PRIMUS_HOST}" ]] && MNT=(-v "${PRIMUS_HOST}:/workspace/Primus")
NAME=bruno_gptoss_${SLURM_JOB_ID:-$$}

echo "=== gpt_oss_20b v26.5 in-image recipe $(date -u) on $(hostname), NEXP=${NEXP}"
echo "    CONT=${CONT}  PRIMUS_HOST=${PRIMUS_HOST:-image}  LOGDIR=${LOGDIR}  LR=${LR:-recipe}  WARMUP_ITERS=${WARMUP_ITERS:-recipe}"
for _ in $(seq 1 60); do
    [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
    tar="${NAS}/containers/$(echo "${CONT#rocm/}" | tr '/:' '__').tar"
    if [[ -f "${tar}" ]]; then echo "loading ${CONT} from ${tar} ($(date -u +%H:%M))"; docker load -q -i "${tar}"; fi
    docker image inspect "${CONT}" >/dev/null 2>&1 || docker pull -q "${CONT}"
    echo "image ready ($(date -u +%H:%M))"
fi
docker image inspect "${CONT}" >/dev/null 2>&1 || { echo "FATAL: no image ${CONT}"; exit 1; }
mkdir -p "${LOGDIR}"
docker rm -f ${NAME} >/dev/null 2>&1
trap 'docker rm -f ${NAME} >/dev/null 2>&1' EXIT

for i in $(seq 1 "${NEXP}"); do
    sync; sudo /sbin/sysctl -q vm.drop_caches=3
    log="${LOGDIR}/$(date +%y%m%d%H%M%S)_${i}.log"
    echo "--- trial ${i}/${NEXP} -> ${log}"
    docker run --rm --name ${NAME} \
        --device /dev/dri --device /dev/kfd --device /dev/infiniband \
        --network host --ipc host --group-add video --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined --privileged --shm-size 128G \
        --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
        -v "${DATADIR}:/data" -v "${MODELDIR}:/model:ro" -v "${LOGDIR}:/results" "${MNT[@]}" \
        -e MODEL=/model -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e SEED=$(( $(od -An -N4 -tu4 /dev/urandom) % 2147483647 )) \
        -e HPE_LR="${LR}" -e HPE_WARMUP="${WARMUP_ITERS}" \
        "${CONT}" bash -c '
            cd /workspace/Primus/examples/mlperf/gpt_oss_20b &&
            echo "Primus in image: $(cd /workspace/Primus && git log -1 --format="%h %cs" 2>/dev/null)" &&
            source config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh &&
            if [[ -n "$HPE_LR" ]]; then
                export PRIMUS_LR=$HPE_LR
                export PRIMUS_MIN_LR=$(python3 -c "print(\"%g\" % ($HPE_LR*0.1))")
            fi &&
            if [[ -n "$HPE_WARMUP" ]]; then
                export PRIMUS_LR_WARMUP_ITERS=$HPE_WARMUP
                export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS - HPE_WARMUP))
            fi &&
            echo "HPE settings: lr=$PRIMUS_LR min_lr=$PRIMUS_MIN_LR warmup=$PRIMUS_LR_WARMUP_ITERS decay=$PRIMUS_LR_DECAY_ITERS gbs=$PRIMUS_GLOBAL_BATCH_SIZE" &&
            export MODEL=/model PRIMUS_NUM_WORKERS=0 MLLOG_TRAIN_LOSS_LOG_FREQ=32 &&
            export HF_HOME=/results/hf_home PRIMUS_WORKSPACE=/results/primus &&
            mkdir -p "$HF_HOME" "$PRIMUS_WORKSPACE" &&
            bash run_and_time.sh' 2>&1 | tee "${log}"
done
echo "=== done $(date -u)"
