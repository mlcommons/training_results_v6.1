#!/bin/bash
# Render + run ONE Flux.1 v6.1 compliance run and collect its result file.
#
#   CTX=<kube-context> NS=<namespace> bash run-flux-compliance.sh <SET> <SEED> [RESULT_IDX]
#
#   SET A : GBS 2048 / LR 2.5e-4 / warmup 0    (RCP row 2048)
#   SET B : GBS 4096 / LR 4.0e-4 / warmup WU   (RCP row 4096; WU defaults to 400)
#           warmup 100 converged BELOW the RCP window (14.81M < 15.02M min); warmup 400
#           is what the v6.1 submission used (see package/README.md for the RCP note).
#   SET C : GBS 1024 / LR 2.5e-4 / warmup 800  (RCP row 1024)
#   MBS = GBS / (N*8): no gradient accumulation (A at 32 nodes = MBS 8, at 8 nodes = MBS 32).
#
# Ends with one line "VERDICT <PASS|FAIL|FAIL_PRESTART> <job> bench=flux1 nn= gbs= seed= ttt_s=
# last_eval= job= run=" (llama/scheduler.py drives Flux sets through it). ADOPT_J/ADOPT_RUN
# re-attach to a job a previous runner launched instead of creating one.
#
# Required: IMAGE (the flux1 trainer image), SELKEY/SELVAL (a label on exactly the
#           validated nodes to run on; must NOT be amd.com/gpu.product-name, which the
#           template already selects on).
# Optional: PULL_SECRET (registry pull secret name), N (nodes, default 32), WU (set-B
#           warmup), RCCL_TOPO_PATH, OUT_DIR, TIMEOUT (s, default 10800).
# With RESULT_IDX, the rank-0 MLLOG stream is saved as OUT_DIR/result_<RESULT_IDX>.txt.
#
# Verdict is read from the MLLOG stream, not the Job status: a run that never
# reaches the target still exits 0 when it hits its own stop condition.
set -uo pipefail
cd "$(dirname "$0")"
: "${CTX:?set CTX=<kube-context>}"; : "${NS:?set NS=<namespace>}"
SET="${1:?usage: run-flux-compliance.sh A|B SEED [RESULT_IDX]}"; SEED="${2:?seed}"; RIDX="${3:-}"
N="${N:-32}"
: "${IMAGE:?set IMAGE=<registry>/<repo>/flux1-libionic:<tag>}"
: "${SELKEY:?set SELKEY=<node-label-key> (label on the validated nodes)}"; : "${SELVAL:?set SELVAL=<node-label-value>}"
[ "$SELKEY" = amd.com/gpu.product-name ] && { echo "!! SELKEY must be a pool label, not amd.com/gpu.product-name (already in the template)"; exit 1; }
TOPO="${RCCL_TOPO_PATH:-/etc/crusoe/rccl_topo/mi355x-288gb-ib.xml}"
KC(){ kubectl --context "$CTX" -n "$NS" "$@"; }
ts(){ echo "$(date -u +'[%H:%M:%SZ]') $*"; }

case "$SET" in
  A) GBS=2048; LR=2.5e-4; WARM=0 ;;
  B) GBS=4096; LR=4.0e-4; WARM="${WU:-400}" ;;
  C) GBS=1024; LR=2.5e-4; WARM=800 ;;                  # RCP row 1024
  *) echo "!! SET must be A, B or C"; exit 1 ;;
esac
# No gradient accumulation: every GPU takes GBS/GPUs samples per step (MBS 8/16 for A/B at 32 nodes).
MBS=$(( GBS / (N * 8) ))
# MBS x GPUs must divide GBS, or the trainer silently uses gradient accumulation
# that the log does not disclose the way the RCP row assumes.
[ $(( GBS % (MBS * N * 8) )) -eq 0 ] || { echo "!! GBS=$GBS not divisible by MBS=$MBS x $((N*8)) GPUs"; exit 1; }

TAG=$(printf '%s' "$SET" | tr 'ABC' 'abc')
J="flux-${N}n-${TAG}-s${SEED}"
RUN="${J}-$(date -u +%m%d%H%M)"
OUT_DIR="${OUT_DIR:-results/set${SET}/${N}xMI355X-288GB_2xEPYC-9575F/flux1}"

have=$(kubectl --context "$CTX" get nodes -l "$SELKEY=$SELVAL" -o json | jq '[.items[]|select(.spec.unschedulable != true)]|length')
[ "$have" -ge "$N" ] || { echo "!! only $have schedulable nodes carry $SELKEY=$SELVAL, need $N"; exit 1; }

if [ -n "${ADOPT_J:-}" ]; then
  # re-attach to a job a previous runner launched (scheduler restart): follow it, don't recreate it
  J=$ADOPT_J; RUN=${ADOPT_RUN:?}
  ts "ADOPT set=$SET seed=$SEED N=$N job=$J run=$RUN"
else
ts "set=$SET seed=$SEED N=$N GBS=$GBS MBS=$MBS LR=$LR warmup=$WARM  job=$J run=$RUN"
KC delete job,svc "$J" --ignore-not-found >/dev/null 2>&1; sleep 5
sed "s/__J__/$J/g; s/__SEED__/$SEED/g; s/__RUN__/$RUN/g; s/__N__/$N/g;
     s/__GBS__/$GBS/g; s/__MBS__/$MBS/g; s/__LR__/$LR/g; s/__WARMUP__/$WARM/g;
     s|__SELKEY__|$SELKEY|g; s|__SELVAL__|$SELVAL|g; s|__IMAGE__|$IMAGE|g;
     s|/etc/crusoe/rccl_topo/mi355x-288gb-ib.xml|$TOPO|g" train-32node-compliance.template.yaml \
  | if [ -n "${PULL_SECRET:-}" ]; then sed "s|__PULL_SECRET__|$PULL_SECRET|"; else grep -v '__PULL_SECRET__'; fi \
  | KC apply -f - >/dev/null || { echo "!! apply failed"; exit 1; }
fi

P0=""; DL=$(( $(date +%s) + ${TIMEOUT:-10800} ))
while [ "$(date +%s)" -lt $DL ]; do
  P0=$(KC get pods -l job-name="$J",batch.kubernetes.io/job-completion-index=0 -o name 2>/dev/null | head -1)
  st=$(KC get job "$J" -o jsonpath='{.status.succeeded}/{.status.failed}' 2>/dev/null)
  run=$(KC get pods -l job-name="$J" --no-headers 2>/dev/null | grep -c Running)
  ev=$( [ -n "$P0" ] && KC logs "${P0#pod/}" --tail=400 2>/dev/null | grep -oE '"key": "eval_accuracy", "value": [0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
  ts "running=$run/$N succeeded/failed=${st:-/} last_eval=${ev:-none}"
  # done when every index succeeded, or any pod failed (backoffLimit 0 = the run is over)
  s_=${st%%/*}; f_=${st#*/}
  { [ "${s_:-0}" -ge "$N" ] || [ "${f_:-0}" -gt 0 ]; } 2>/dev/null && break
  KC logs "${P0#pod/}" --tail=50 2>/dev/null | grep -q '"key": "run_stop"' && break
  sleep 120
done

LOG="${TMPDIR:-/tmp}/${RUN}-rank0.log"
FULL="${TMPDIR:-/tmp}/${RUN}-rank0.full.log"
# full framework log (training_rules "Result files"): rank 0 tees it to the shared volume, which
# survives a failed Indexed Job deleting its pods; fall back to the pod log only if that is missing.
# The tee on the shared volume can trail the pod log by a few seconds, so when the pod already
# shows run_stop, wait (up to 3 min) for the shared copy to have it too before reading the verdict.
podstop=$(KC logs "${P0#pod/}" --tail=50 2>/dev/null | grep -c '"key": "run_stop"')
for _ in $(seq 1 36); do
  KC exec "${READER_POD:-log-reader}" -- cat "/shared/logs/${RUN}/rank0.log" > "$FULL" 2>/dev/null
  { [ "${podstop:-0}" -eq 0 ] || grep -q '"key": "run_stop"' "$FULL"; } && break
  sleep 5
done
[ -s "$FULL" ] || KC logs "${P0#pod/}" --tail=-1 > "$FULL" 2>/dev/null
grep ':::MLLOG' "$FULL" > "$LOG"
status=$(grep '"key": "run_stop"' "$LOG" | grep -oE '"status": "[a-z]+"' | tail -1 | cut -d'"' -f4)
python3 - "$LOG" <<'PY'
import json,sys
t0=t1=None; ev=[]
for l in open(sys.argv[1]):
    j=json.loads(l.split(':::MLLOG',1)[1]); k=j['key']; m=j.get('metadata') or {}
    if k=='run_start': t0=j['time_ms']
    if k=='run_stop':  t1=j['time_ms']
    if k=='eval_accuracy': ev.append((j['value'], m.get('samples_count')))
print(f"  evals={len(ev)} last={ev[-1] if ev else None}  ttt_min={round((t1-t0)/60000,2) if t0 and t1 else None}")
PY
ts "run_stop status=${status:-MISSING}"
# Keep EVERY run that reached run_start (pass or fail) for the M>N consecutive-window selection
# (training_rules "Scoring"): failed runs count as infinite time and can't be silently skipped.
if [ -n "${OUT_DIR:-}" ] && grep -q '"key": "run_start"' "$LOG"; then
  mkdir -p "$OUT_DIR/all-runs"; cp "$FULL" "$OUT_DIR/all-runs/${RUN}.txt"
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$RUN" "${status:-failed}" >> "$OUT_DIR/all-runs/runs.tsv"
fi
if [ "$status" = success ] && [ -n "$RIDX" ]; then
  mkdir -p "$OUT_DIR"; cp "$FULL" "$OUT_DIR/result_${RIDX}.txt"; ts "saved $OUT_DIR/result_${RIDX}.txt (full log)"
fi
KC delete job,svc "$J" --ignore-not-found >/dev/null 2>&1
# One-line verdict in the same format as llama/run-llama-mlperf.sh, so llama/scheduler.py can drive
# Flux sets. FAIL_PRESTART = never reached run_start (infrastructure, not a scored run).
ttt=$(python3 - "$LOG" <<'PY'
import json,sys
t={}
for l in open(sys.argv[1]):
    try: j=json.loads(l.split(":::MLLOG",1)[1])
    except Exception: continue
    if j.get("key") in ("run_start","run_stop"): t[j["key"]]=j["time_ms"]
print(round((t["run_stop"]-t["run_start"])/1000) if len(t)==2 else "NA")
PY
)
last=$(grep -oE '"key": "eval_accuracy", "value": [0-9.]+' "$LOG" | tail -1 | grep -oE '[0-9.]+$')
if ! grep -q '"key": "run_start"' "$LOG"; then v=FAIL_PRESTART
elif [ "$status" = success ]; then v=PASS; else v=FAIL; fi
echo "VERDICT $v $J bench=flux1 nn=$N gbs=$GBS seed=$SEED ttt_s=$ttt last_eval=${last:-NA} job=$J run=$RUN"
[ "$status" = success ]
