#!/usr/bin/env bash
set -euo pipefail
[[ "$(id -u)" -eq 0 ]] && { echo "LAUNCH BLOCKED: uid 0" >&2; exit 2; }

L=/mnt/data/mlperftv61/codedrop/llama31-8b-20261012
L18CFG="${L18CFG:-config_VR200_18x4x1xtp1pp1cp1_8b_ub_avgcoll.sh}"
NEXP18="${NEXP18:-10}"
L18TAG="${L18CFG#config_VR200_18x4x1xtp1pp1cp1_8b_}"; L18TAG="${L18TAG%.sh}"
LLOG="$HOME/llama31-8b-logs"
MANIFEST="$HOME/llama31-8b-logs/manifest.tsv"
HCA='=ibp0p0,=ibp1p0,=ibp2p0,=ibp3p0,=ibp4p0,=ibp5p0,=ibp6p0,=ibp7p0'
DRY="${DRY_RUN:-0}"
D=$(date -u +%Y%m%d)
N18="${NODES_18N:?set NODES_18N to the 18-node allocation}"
AFTER18="${AFTER18:-}"

for f in "$L/$L18CFG" "$L/llama-18n.sh" "$L/site.env"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
done
grep -q 'llama31_8b-arm-20261012.sqsh' "$L/site.env" || { echo "$L/site.env CONT is not the 20261012 image" >&2; exit 2; }
grep -qx 'export NCCL_IB_HCA=ibp' "$L/site.env" || { echo "$L/site.env NCCL_IB_HCA is not the expected 'ibp' line" >&2; exit 2; }
DATA=$(sed -n 's/^export DATADIR=//p' "$L/site.env"); [[ -d "$DATA/8b" ]] || { echo "DATADIR $DATA/8b missing" >&2; exit 2; }
n=$(sinfo -h -N -o %N -n "$N18" | sort -u | wc -l); [[ "$n" -eq 18 ]] || { echo "NODES_18N: $n of 18 nodes registered" >&2; exit 2; }
if [[ -z "$AFTER18" ]]; then
  idle=$(sinfo -h -N -t idle -n "$N18" -o %N | sort -u | wc -l)
  [[ "$idle" -eq 18 ]] || { echo "NODES_18N: $idle of 18 idle and no AFTER18 given" >&2; exit 2; }
else
  squeue -h -j "${AFTER18//:/,}" -o %i | grep -q . || { echo "AFTER18=$AFTER18 is not in the queue" >&2; exit 2; }
fi

cp "$L/site.env" "$L/site.env.hca-bak"
trap 'cp "$L/site.env.hca-bak" "$L/site.env"' EXIT
python3 - "$L/site.env" "$HCA" <<'PY'
import sys
p, hca = sys.argv[1], sys.argv[2]
s = open(p).read().replace("export NCCL_IB_HCA=ibp\n", f'export NCCL_IB_HCA="{hca}"\n', 1)
open(p, "w").write(s)
PY
grep -q "NCCL_IB_HCA=\"$HCA\"" "$L/site.env" || { echo "HCA patch failed" >&2; exit 2; }

submit() {
  local tag=$1 launcher=$2 logdir=$3 nodes=$4 dep=$5; shift 5
  mkdir -p "$logdir"
  local out jid
  out=$(env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" PATH="$PATH" LOGDIR="$logdir" NODELIST="$nodes" DRY_RUN="$DRY" \
        ${dep:+SBATCH_DEPENDENCY="afterany:$dep"} "$@" bash "$L/$launcher" 2>&1) || { echo "$out" >&2; exit 1; }
  echo "$out" | grep -E "^(CONFIG|SLOW_DATADIR|LOCAL_COPY|DRY_RUN|Submitted)" || { echo "$out" >&2; exit 1; }
  jid=$(echo "$out" | sed -n 's/^Submitted batch job \([0-9]*\).*/\1/p')
  [[ -n "$jid" ]] || { echo "  -> $tag on $nodes, dependency ${dep:-none}"; return 0; }
  [[ -n "$dep" ]] && scontrol update jobid="$jid" dependency="afterany:$dep"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$jid" "${tag%%-*}" L "${tag#*-}" "$nodes" "$tag" "$logdir" >> "$MANIFEST"
  echo "  -> $tag job $jid on $nodes, dependency ${dep:-none}"
  LAST_JID=$jid
}

LAST_JID=
echo "=== 18x4 $L18TAG, $NEXP18 seeds, local copy, NODES_18N${AFTER18:+ after $AFTER18}"
submit "18n-score-$L18TAG" llama-18n.sh "$LLOG/18n-score-$D-${L18TAG//_/-}-n$NEXP18" "$N18" "$AFTER18" \
  NEXP="$NEXP18" SLOW_DATADIR_SRC="$DATA" CONFIG_FILE="$L18CFG"
echo "18x4 job: ${LAST_JID:-dry}"
