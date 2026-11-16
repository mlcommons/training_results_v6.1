#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -u)" -eq 0 ]] && { echo "LAUNCH BLOCKED: uid 0" >&2; exit 2; }
cd "${HOME}/dsv3-671b"
[[ -f fabric.sets ]] || { echo "missing fabric.sets" >&2; exit 2; }

export ARM="${ARM:-baseline}" NEXP="${NEXP:-3}" SHORT=0 LOCAL_COPY=1 CHECK_COMPLIANCE=1
export SETS_FILE="${HOME}/dsv3-671b/fabric.sets"
export WALLTIME_RUNANDTIME="${WALLTIME_RUNANDTIME:-90}"
export LOGDIR="${LOGDIR:-${HOME}/dsv3-671b/logs/score-$(date -u +%Y%m%d)-${ARM}-n${NEXP}}"
if [[ -n "${AFTER:-}" ]]; then
  export DEPENDENCY="afterany:${AFTER}" ALLOW_BUSY_NODES=1
fi
REAL="${REAL:-0}" bash dsv3-64n.sh "$@"
