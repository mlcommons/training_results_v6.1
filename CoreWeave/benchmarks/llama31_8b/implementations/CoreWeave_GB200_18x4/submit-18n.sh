#!/usr/bin/env bash
set -euo pipefail

[[ "$(id -u)" -eq 0 ]] && { echo "LAUNCH BLOCKED: uid 0" >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")"
grep -q 'llama31_8b-arm-20261012.sqsh' site.env || { echo "LAUNCH BLOCKED: site.env CONT is not the 20261012 image" >&2; exit 2; }

NODES_18N="${NODES_18N:?set NODES_18N to the 18-node allocation}"
NEXP="${NEXP:-10}"
TAG="${TAG:-b}"
[[ "${NEXP}" =~ ^[0-9]+$ && "${NEXP}" -ge 10 ]] || { echo "LAUNCH BLOCKED: NEXP=${NEXP}; a scored set is at least 10 seeds in one job" >&2; exit 2; }

submit() {
  local launcher="$1" name="$2" seed="$3" nodelist="$4" cfg="$5"
  local out
  out="$(NODELIST="${nodelist}" NEXP="${NEXP}" SEED_BASE="${seed}" CONFIG_FILE="${cfg}" LLAMA_18N_LONG_WARMUP=0 \
    LOGDIR="${HOME}/llama31-8b-logs/${name}-n${NEXP}" \
    bash "${launcher}" | tail -1)"
  echo "${name} ${out} seed=${seed} cfg=${cfg}"
}

submit ./llama-18n.sh "llama-18n-1012-${TAG}" 31900 "${NODES_18N}" \
  config_GB200_18x4x1xtp1pp1cp1_8b_fp4.sh
