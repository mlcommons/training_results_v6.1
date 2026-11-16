#!/bin/bash
#
# Unattended MLPerf Llama 3.1 8B scored set using the softmax32 recipe.
#
# Recipe under test (vs the recipe used for the current scored set):
#   - conf/llama3.1_8B-pretrain-fp4-softmax32.yaml : attention_softmax_in_fp32=true
#   - PRIMUS_MICRO_BATCH_SIZE=4                    : gradient accumulation 1 (+3.5%)
#   - MLPERF_NUMA_BIND=1                           : rank pinned to its GPU's NUMA node (+1.8%)
#
# Why: our runs miss the 184,320-sample eval checkpoint by as little as 0.005
# eval loss, and each miss costs a full 12,288-sample interval (~5.7 min). In a
# 2-run pilot, softmax32 converged at 184,320 both times, while the stock recipe
# lands on 196,608 roughly 40% of the time. Throughput cost measured at ~0.1%.
#
# Safe to run detached and safe to re-run: it counts completed runs and only
# launches the remainder, so an interruption costs at most one run.
#
# Usage:
#   run_softmax32_set.sh                 # allocate a node, run 10, analyse, package
#   run_softmax32_set.sh --dry-run       # preflight checks only, no allocation
#   run_softmax32_set.sh --runs 10 --node lodin2 --tag softmax32_final
#   run_softmax32_set.sh --no-alloc      # reuse an allocation you already hold

set -uo pipefail

RUNS=10
NODE="lodin3"
TAG="softmax32_final"
PARTITION="dmoise"
WALLTIME="1-00:00:00"
JOBNAME="mlperf-sm32"
DO_ALLOC=1
DRY_RUN=0
KEEP_ALLOC=0
AUTO_PREP=1

WORK=/nas/SHARED/MLPERF/work
RESULTS_ROOT=/nas/SHARED/MLPERF/results
SCRATCH=/nas/SHARED/MLPERF/scratch
DATA_ROOT=/lvol0/mlperf/llama31_8b
EXP_YAML=/workspace/code/conf/llama3.1_8B-pretrain-fp4-softmax32.yaml
IMAGE="rocm/amd-mlperf:llama31_8b_training_6.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2;;
    --node) NODE="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --time) WALLTIME="$2"; shift 2;;
    --no-alloc) DO_ALLOC=0; shift;;
    --keep-alloc) KEEP_ALLOC=1; shift;;
    --no-prep) AUTO_PREP=0; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

LOGDIR="${RESULTS_ROOT}/llama31_8b/${TAG}"
MAINLOG="${SCRATCH}/${TAG}_driver.log"
JOBID=""

log(){ echo "[$(date '+%F %T')] $*" | tee -a "${MAINLOG}"; }
ssh_node(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "${NODE}" "$@"; }

# Stage the C4 dataset + tokenizer to node-local NVMe and fetch the container.
# Needed the first time a node is used; idempotent afterwards. Staging to local
# durable storage before the timed region is explicitly permitted by the rules.
prep_node() {
  log "--- prep ${NODE}: staging dataset + image (one-off, ~40-60 min) ---"
  ssh_node "bash -s" <<PREP 2>&1 | sed 's/^/    /' | tee -a "${MAINLOG}"
set -u
SRC=/nas/SHARED/MLPERF/mlperf_llama31_8b
DST=${DATA_ROOT}
sudo mkdir -p "\$DST"/data "\$DST"/model
sudo chown -R 64000:100 /lvol0/mlperf

( docker image inspect ${IMAGE} >/dev/null 2>&1 || docker pull ${IMAGE} >/dev/null 2>&1 ) &
PULL=\$!

for f in c4-train.en_6_text_document.bin c4-train.en_6_text_document.idx \\
         c4-validation-91205-samples.en_text_document.bin \\
         c4-validation-91205-samples.en_text_document.idx \\
         llama-3-1-8b-preprocessed-c4-dataset.md5 LICENSE.txt NOTICE.txt; do
  s="\$SRC/data/\$f"; d="\$DST/data/\$f"
  if [ -e "\$d" ] && [ "\$(stat -c%s "\$s")" = "\$(stat -c%s "\$d")" ]; then continue; fi
  cp -f "\$s" "\$d" && echo "staged \$f"
done
for f in config.json generation_config.json special_tokens_map.json \\
         tokenizer.json tokenizer_config.json model.safetensors.index.json; do
  cp -f "\$SRC/model/\$f" "\$DST/model/\$f" 2>/dev/null
done
wait \$PULL
cd "\$DST/data" && md5sum -c <(grep -E 'c4-' llama-3-1-8b-preprocessed-c4-dataset.md5) 2>&1
PREP
  # md5sum -c prints "FAILED" on mismatch; treat that as fatal
  ! grep -q "FAILED" "${MAINLOG}" || { log "checksum mismatch after staging"; return 1; }
  return 0
}

cleanup() {
  if [[ -n "${JOBID}" && ${KEEP_ALLOC} -eq 0 ]]; then
    log "releasing allocation ${JOBID}"
    scancel "${JOBID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "${SCRATCH}"
log "=== softmax32 scored set: ${RUNS} runs, tag=${TAG}, node=${NODE} ==="

# ----------------------------------------------------------------- preflight
log "--- preflight ---"
PRE_OK=1
check(){ if eval "$2" >/dev/null 2>&1; then log "  OK   $1"; else log "  FAIL $1"; PRE_OK=0; fi; }

check "driver script present"      "[ -x ${WORK}/llama31_8b/run.llama31-8b.training.sh ]"
check "softmax32 YAML present"     "[ -f ${WORK}/llama31_8b/code/conf/llama3.1_8B-pretrain-fp4-softmax32.yaml ]"
check "NUMA module present"        "[ -f ${WORK}/llama31_8b/code/src/_numa_affinity.py ]"
check "submission tooling present" "[ -x ${WORK}/submission/make_submission.sh ]"
check "node reachable"             "ssh_node true"

# An unallocated node has its docker daemon stopped, which is the normal idle
# state; only a *running* container means the node is genuinely busy.
NODE_BUSY=$(ssh_node 'docker ps -q 2>/dev/null | wc -l' 2>/dev/null || echo 0)
if [[ "${NODE_BUSY}" -eq 0 ]]; then
  log "  OK   node idle (no running container)"
else
  log "  FAIL node busy: ${NODE_BUSY} container(s) running on ${NODE}"
  PRE_OK=0
fi

# Refuse to collide with someone else's allocation.
HOLDER=$(squeue -h -w "${NODE}" -o '%i %u %j' 2>/dev/null | head -1)
if [[ -z "${HOLDER}" ]]; then
  log "  OK   ${NODE} not allocated to any job"
elif [[ "${DO_ALLOC}" -eq 0 ]]; then
  log "  OK   ${NODE} held by: ${HOLDER} (reusing, --no-alloc)"
else
  log "  FAIL ${NODE} already allocated to: ${HOLDER}"
  PRE_OK=0
fi

if [[ ${DO_ALLOC} -eq 0 ]]; then
  check "dataset staged on ${NODE}" "ssh_node '[ -s ${DATA_ROOT}/data/c4-train.en_6_text_document.bin ]'"
  check "tokenizer staged"          "ssh_node '[ -s ${DATA_ROOT}/model/tokenizer.json ]'"
  check "image present"             "ssh_node 'docker image inspect ${IMAGE} >/dev/null'"
else
  log "  ..  dataset/image checks deferred until the node is allocated"
fi

# the softmax32 knob must actually be set, otherwise this run is a duplicate
if grep -qE '^\s*attention_softmax_in_fp32:\s*true' \
     "${WORK}/llama31_8b/code/conf/llama3.1_8B-pretrain-fp4-softmax32.yaml" 2>/dev/null; then
  log "  OK   attention_softmax_in_fp32: true"
else
  log "  FAIL attention_softmax_in_fp32 not enabled in the softmax32 YAML"
  PRE_OK=0
fi

DONE_NOW=$(grep -l 'run_stop' "${LOGDIR}"/*.log 2>/dev/null | wc -l)
log "  ..   completed runs already in ${LOGDIR}: ${DONE_NOW}"

if [[ ${PRE_OK} -ne 1 ]]; then
  log "PREFLIGHT FAILED - not starting"
  exit 1
fi
if [[ ${DRY_RUN} -eq 1 ]]; then
  log "dry run only; preflight passed. Estimated wall time for ${RUNS} runs: ~$(( RUNS * 90 / 60 )) h"
  KEEP_ALLOC=1
  exit 0
fi

# ---------------------------------------------------------------- allocation
if [[ ${DO_ALLOC} -eq 1 ]]; then
  log "--- requesting allocation on ${NODE} (${WALLTIME}) ---"
  JOBID=$(salloc --no-shell -N1 -n8 --exclusive --comment='docker' \
                 -p "${PARTITION}" -w "${NODE}" --time="${WALLTIME}" -J "${JOBNAME}" 2>&1 \
          | grep -oE 'job allocation [0-9]+' | grep -oE '[0-9]+')
  if [[ -z "${JOBID}" ]]; then log "could not obtain an allocation"; exit 1; fi
  log "allocation ${JOBID} granted"

  log "waiting for docker daemon on ${NODE}..."
  for _ in $(seq 1 60); do
    ssh_node 'systemctl is-active docker' 2>/dev/null | grep -q active && break
    sleep 15
  done
  ssh_node 'systemctl is-active docker' 2>/dev/null | grep -q active \
    || { log "docker never came up on ${NODE}"; exit 1; }
  log "docker active"

  for c in "dataset:[ -s ${DATA_ROOT}/data/c4-train.en_6_text_document.bin ]" \
           "tokenizer:[ -s ${DATA_ROOT}/model/tokenizer.json ]" \
           "image:docker image inspect ${IMAGE} >/dev/null 2>&1"; do
    n="${c%%:*}"; t="${c#*:}"
    if ssh_node "${t}"; then
      log "  OK   ${n} on ${NODE}"
    elif [[ ${AUTO_PREP} -eq 1 ]]; then
      log "  ..   ${n} missing on ${NODE}; running prep (this is a fresh node)"
      prep_node || { log "prep failed for ${n}"; exit 1; }
      ssh_node "${t}" || { log "  FAIL ${n} still missing after prep"; exit 1; }
      log "  OK   ${n} after prep"
    else
      log "  FAIL ${n} on ${NODE} (use --prep or stage manually)"
      exit 1
    fi
  done
fi

# ---------------------------------------------------------------------- runs
mkdir -p "${LOGDIR}"
chmod 777 "${LOGDIR}" 2>/dev/null || true

DONE=$(grep -l 'run_stop' "${LOGDIR}"/*.log 2>/dev/null | wc -l)
NEED=$(( RUNS - DONE ))
log "--- runs: ${DONE} complete, launching ${NEED} more ---"

if [[ ${NEED} -gt 0 ]]; then
  ssh_node "cd ${WORK}/llama31_8b && LOGDIR=${LOGDIR} ./run.llama31-8b.training.sh ${TAG} ${NEED} \
      EXP=${EXP_YAML} \
      PRIMUS_MICRO_BATCH_SIZE=4 \
      MLPERF_NUMA_BIND=1 \
      MLLOG_MICRO_BATCH_SIZE=4" >> "${SCRATCH}/${TAG}_runs.log" 2>&1
  log "run batch returned $?"
fi

DONE=$(grep -l 'run_stop' "${LOGDIR}"/*.log 2>/dev/null | wc -l)
log "completed runs: ${DONE}/${RUNS}"

# ------------------------------------------------------------------ analysis
log "--- analysis ---"
python3 "${WORK}/analyze_results.py" llama31_8b "${LOGDIR}" 2>&1 | tee -a "${MAINLOG}"

if [[ ${DONE} -lt ${RUNS} ]]; then
  log "INCOMPLETE (${DONE}/${RUNS}). Re-run this script to continue; it resumes automatically."
  exit 2
fi

# ------------------------------------------------- package + full validation
log "--- building and validating submission package ---"
PKG=/nas/SHARED/MLPERF/submission_${TAG}
"${WORK}/submission/make_submission.sh" --out "${PKG}" \
    --llama31-results "${LOGDIR}" \
    --llama2-results "${RESULTS_ROOT}/llama2_70b/final" 2>&1 | tee -a "${MAINLOG}"

ssh_node "cd ${WORK} && ./submission/validate_submission.sh --package ${PKG}" 2>&1 | tee -a "${MAINLOG}"
RC=${PIPESTATUS[0]}

log "=== DONE. validation exit=${RC}; package at ${PKG} ==="
log "    driver log : ${MAINLOG}"
log "    run log    : ${SCRATCH}/${TAG}_runs.log"
exit "${RC}"
