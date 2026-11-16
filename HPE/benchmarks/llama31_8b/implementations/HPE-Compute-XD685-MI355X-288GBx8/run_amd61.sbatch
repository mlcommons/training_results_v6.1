#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=6:00:00
#SBATCH --job-name=bruno-amd61
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Run AMD's MLPerf 6.1 partner code drops (7512b71 / 253c561: Llama 3.1, GPT-OSS;
# 3b590b6 / 253c561: DLRMv4; 253c561: Llama 2 70B LoRA) for one benchmark, with the
# HPE adaptations needed on lodin:
#   * image loaded from /nas/.../containers if the node does not have it
#   * MEGATRON_PARAM_GATHER_BACKEND=nccl (the drop's rccl_sdma default fails on
#     lodin with HIP "invalid argument" in RCCL p2p)
#   * submission org/platform = HPE, no kernel tunables, logs on /nas
# Everything else is the drop's MI355X config. Host VAR=VAL arguments are
# exported and take precedence over the config (the drop's launcher only
# applies config values for variables that are not already set).
#
# HPE_NUMA_BIND=1 adds Primus --numa (rank pinned to its GPU's NUMA node).
#
# Usage: sbatch -w <node> --export=ALL,BENCH=llama31_8b|gpt_oss_20b|dlrmv4|llama2_70b_lora,TAG=<tag>[,NEXP=n] \
#               run_amd61.sbatch [VAR=VAL ...]
set -uo pipefail
BENCH=${BENCH:?set BENCH=llama31_8b|gpt_oss_20b|dlrmv4|llama2_70b_lora}
TAG=${TAG:?set TAG}
export NEXP=${NEXP:-1}
if [[ -d /nas/data/SHARED/MLPERF/results ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
REPO=${REPO:-/home/hpebench/mlcommons_mlperf_training_61a}
IMPL=${REPO}/HPE/benchmarks/${BENCH}/implementations/MI355X_release_6.1

case "${BENCH}" in
  llama31_8b)  export CONT=${CONT:-rocm/amd-mlperf:llama31_8b_training_6.1};  export DGXSYSTEM=MI355X_1x8x1 ;;
  gpt_oss_20b) export CONT=${CONT:-rocm/amd-mlperf:gpt_oss_20b_training_6.1}; export DGXSYSTEM=MI355X_1x8x1_tp1pp1ep1_gbs32 ;;
  dlrmv4)      export CONT=${CONT:-rocm/amd-mlperf:dlrmv4_training_6.1};      export DGXSYSTEM=MI355X_1x8x1 ;;
  llama2_70b_lora) export CONT=${CONT:-rocm/amd-mlperf:llama2_70b_training_6.1-253c561}; export DGXSYSTEM=MI355X_1x8x1 ;;
  *) echo "unknown BENCH ${BENCH}"; exit 1 ;;
esac
if [[ "${BENCH}" == llama2_70b_lora ]]; then
  # The drop mounts ${DATADIR}/data -> /data and ${MODELDIR} -> /ckpt, and reads
  # /ckpt/megatron_checkpoints/Llama-2-70b-hf/iter_0000000; both live in one tree here.
  export DATADIR=${DATADIR:-/lvol0/mlperf/llama2}
  export MODELDIR=${MODELDIR:-/lvol0/mlperf/llama2/data}
  need=("${DATADIR}/data/train.npy" "${DATADIR}/data/validation.npy" "${DATADIR}/data/packed_metadata.jsonl"
        "${MODELDIR}/megatron_checkpoints/Llama-2-70b-hf/iter_0000000/common.pt"
        "${DATADIR}/data/huggingface/hub/models--meta-llama--Llama-2-70b-hf")
  # The recipe loads the gated meta-llama/Llama-2-70b-hf config/tokenizer: use the
  # offline copy staged under /data/huggingface instead of the network.
  export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export HPE_FORWARD="${HPE_FORWARD:-} HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE"
elif [[ "${BENCH}" == dlrmv4 ]]; then
  export DATADIR=${DATADIR:-/lvol0/mlperf/dlrmv4/data}
  need=("${DATADIR}/processed_5b/hstu_cache_L4086/_READY" "${DATADIR}/shared_metadata/album_item_mapping.parquet")
else
  # Llama 3.1 and GPT-OSS use the Llama 3.1 C4 data and tokenizer.
  export DATADIR=${DATADIR:-/lvol0/mlperf/llama31_8b/data}
  export MODELDIR=${MODELDIR:-/lvol0/mlperf/llama31_8b/model}
  need=("${DATADIR}/c4-train.en_6_text_document.bin" "${MODELDIR}/tokenizer.json")
fi
export LOGDIR=${NAS}/results/${BENCH}/${TAG}
export CONT_NAME=mlperf_${BENCH}_61_${SLURM_JOB_ID}
# In-container check. The 7512b71 images predate the 6.1.0 rules; the DLRMv4
# image pins mlcommons/logging 8a0cc05, which has them (6.0.0 has no dlrmv4).
if [[ "${BENCH}" == dlrmv4 || "${BENCH}" == llama2_70b_lora ]]; then export MLPERF_RULESET=${MLPERF_RULESET:-6.1.0}
else export MLPERF_RULESET=${MLPERF_RULESET:-6.0.0}; fi
export RUN_RUNTIME_TUNABLES=0
# The drop's launcher forwards host values of config variables; the config
# builds PYTHONPATH for the container, so a host one must not leak in.
unset PYTHONPATH
for kv in "$@"; do export "${kv?}"; done

echo "=== ${BENCH} 6.1 $(date -u) on $(hostname): TAG=${TAG} NEXP=${NEXP} CONT=${CONT} overrides: $*"
for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
for p in "${need[@]}" "${IMPL}/run_with_docker.sh"; do
  [[ -r "$p" ]] || { echo "FATAL: missing $p"; exit 1; }
done
if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
  tar=${NAS}/containers/amd-mlperf_${CONT##*:}.tar
  [[ -f "$tar" ]] || { echo "FATAL: image ${CONT} not on node and no ${tar}"; exit 1; }
  echo "loading ${tar}"; docker load -i "$tar" || exit 1
fi
mkdir -p "${LOGDIR}"
{ date -Is; echo "job=${SLURM_JOB_ID} node=$(hostname -s) image=$(docker image inspect --format '{{.Id}}' "${CONT}") config=${DGXSYSTEM}"
  echo "overrides: HPE_PARAM_GATHER=${HPE_PARAM_GATHER:-nccl} HPE_NCCL_CUMEM=${HPE_NCCL_CUMEM:-0} HPE_NUMA_BIND=${HPE_NUMA_BIND:-0} $*"
  (cd "${REPO}" && git log -1 --format='repo %h %cs'); } >> "${LOGDIR}/launch.txt"
# Run from a per-job copy kept with the results: records the exact code and
# lets the checkout change while jobs run.
CODE=${LOGDIR}/code-${SLURM_JOB_ID}
cp -a "${IMPL}" "${CODE}" && cd "${CODE}" || exit 1
# The GPT-OSS launcher does not read its config (the README sources it first);
# source it here for both benchmarks, then re-apply the HPE overrides.
source "config_${DGXSYSTEM}.sh"
export MLLOG_SUBMISSION_ORG=${HPE_ORG:-HPE} MLLOG_SUBMISSION_PLATFORM=${HPE_PLATFORM:-XD685_MI355X}
export MLPERF_SUBMISSION_PLATFORM=${HPE_PLATFORM:-XD685_MI355X}   # DLRMv4 reads this name
# The GPT-OSS config logs every RCCL collective (NCCL_DEBUG=INFO, SUBSYS COLL)
# to /results: ~220 MB per rank per 35 min on /nas. Warnings only.
export NCCL_DEBUG=${HPE_NCCL_DEBUG:-WARN}
unset NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE
# With the drop's custom RCCL, GPT-OSS crashes at its first all-reduce on lodin
# (HIP "invalid argument" in RCCL p2p) unless the parameter all-gather is plain
# NCCL *and* cuMem allocation is off (Joshua's job 396; our job 398 failed
# with only the first). The config forces NCCL_CUMEM_ENABLE=1, so set it after.
export MEGATRON_PARAM_GATHER_BACKEND=${HPE_PARAM_GATHER:-nccl}
[[ "${BENCH}" == gpt_oss_20b || "${BENCH}" == llama2_70b_lora ]] && export NCCL_CUMEM_ENABLE=${HPE_NCCL_CUMEM:-0}
for kv in "$@"; do export "${kv?}"; done
# Relax strict memory overcommit for this job (user-authorized 2026-10-12): with
# vm.overcommit_memory=2 DLRMv4 can fork only ~6 loader workers per GPU. The
# node puts 2 back by itself, so set it here. Never restore 2 at exit: the
# container can outlive the batch script (cancel), and switching to strict mode
# while ~3 TB is committed makes every allocation on the node fail (lodin1 was
# drained that way, 2026-10-12 08:03). Default: keep the node's setting.
if [[ -n "${HPE_OVERCOMMIT:-}" ]]; then
  _oc_old=$(sysctl -n vm.overcommit_memory)
  sudo -n sysctl -qw vm.overcommit_memory="${HPE_OVERCOMMIT}" && echo "vm.overcommit_memory ${_oc_old} -> $(sysctl -n vm.overcommit_memory)"
fi
# THP defrag: lodin1/2 still have "always" from AMD's runtime_tunables.sh (run by
# our older launchers); "madvise" is the kernel default (lodin4). Opt-in.
if [[ -n "${HPE_THP_DEFRAG:-}" ]]; then
  echo "${HPE_THP_DEFRAG}" | sudo -n tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null
fi
echo "kernel: zone_reclaim_mode=$(sysctl -n vm.zone_reclaim_mode) overcommit=$(sysctl -n vm.overcommit_memory) thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled) defrag=$(cat /sys/kernel/mm/transparent_hugepage/defrag)"
# Names of the VAR=VAL overrides, forwarded into the container by launchers
# that only pass their config's variables (DLRMv4).
export HPE_FORWARD="${HPE_FORWARD:-} $(for kv in "$@"; do printf '%s ' "${kv%%=*}"; done)"
# Host memory commit (strict overcommit on lodin: forked loader workers can hit
# CommitLimit, seen as ENOMEM or ENOSPC on /dev/shm). One line every 10 s.
( while sleep 10; do echo "$(date -u +%T) $(awk '/^Committed_AS/{c=$2} /^CommitLimit/{l=$2} /^MemAvailable/{a=$2} END{printf "committed %d GB of %d GB, available %d GB", c/1048576, l/1048576, a/1048576}' /proc/meminfo)"; done ) \
    > "${LOGDIR}/commit-${SLURM_JOB_ID}.log" 2>&1 &
monpid=$!
bash run_with_docker.sh
rc=$?
kill ${monpid} 2>/dev/null
echo "=== done rc=${rc} $(date -u)"
exit ${rc}
