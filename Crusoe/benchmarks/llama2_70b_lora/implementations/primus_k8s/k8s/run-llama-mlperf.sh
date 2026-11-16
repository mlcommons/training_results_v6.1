#!/bin/bash
# Launch ONE MLPerf v6.1 Llama run and print a one-line verdict.
#
#   CTX=<ctx> NS=<ns> IMAGE=<image> BENCH=llama31_8b|llama2_70b_lora NN=<nodes> GBS=<gbs> \
#     [MBS=1] [CP=1] [TP=1] [SEED=<n>] [TAG=<name>] [SELKEY=<label> SELVAL=<value>] [PULL_SECRET=<secret>] \
#     [LOGROOT=<local mount of the shared volume's llama-v61/logs>] bash run-llama-mlperf.sh
#
# Logs: every rank tees to <pvc>/llama-v61/logs/<run>/rank<i>.log (written in-pod, so nothing
# depends on this shell staying alive). Verdict = rank0's :::MLLOG run_stop status, never the
# Job's exit code. TTT = run_stop - run_start (the MLPerf-scored interval).
set -uo pipefail
cd "$(dirname "$0")"
: "${CTX:?}" "${NS:?}" "${IMAGE:?}" "${BENCH:?}" "${NN:?}" "${GBS:?}"
MBS=${MBS:-1}; SEED=${SEED:-$RANDOM}; TAG=${TAG:-run}; CP=${CP:-1}; TP=${TP:-1}
SELKEY=${SELKEY:-amd.com/gpu.product-name}; SELVAL=${SELVAL:-AMD_Instinct_MI355_OAM}
TIMEOUT=${TIMEOUT:-14400}
K(){ kubectl --context "$CTX" -n "$NS" "$@"; }
ts(){ date -u +%H:%M:%SZ; }
short=$([ "$BENCH" = llama31_8b ] && echo l8b || echo lora)
J="${short}-${NN}n-s${SEED}-$(date -u +%m%d%H%M%S)"; RUN="${TAG}/${J}"
case "$BENCH" in
  llama31_8b) DSUB=datasets/llama31_8b ;;
  llama2_70b_lora) DSUB=datasets/llama2_70b_lora ;;
  *) echo "!! unknown BENCH=$BENCH"; exit 2 ;;
esac

if [ -n "${ADOPT_J:-}" ]; then
  # re-attach to a job launched by an earlier (killed) runner: no apply, just wait + verdict
  J=$ADOPT_J; RUN=${ADOPT_RUN:?}
  echo "[$(ts)] adopted $J run=$RUN"
else
K create configmap llama-v61-launch --from-file=launch-llama.sh=launch-llama.sh --dry-run=client -o yaml | K apply -f - >/dev/null
sed -e "s|__J__|$J|g; s|__N__|$NN|g; s|__BENCH__|$BENCH|g; s|__GBS__|$GBS|g; s|__MBS__|$MBS|g" \
    -e "s|__SEED__|$SEED|g; s|__RUN__|$RUN|g; s|__IMAGE__|$IMAGE|g; s|__DATA_SUBPATH__|$DSUB|g" \
    -e "s|__SELKEY__|$SELKEY|g; s|__SELVAL__|$SELVAL|g; s|__CP__|$CP|g; s|__TP__|$TP|g; s|__LR__|${LR:-}|g; s|__MGC__|${MANUAL_GC:-0}|g; s|__HF_SECRET__|${HF_SECRET:-hf-token}|g" train-llama.template.yaml \
  | { if [ -n "${PULL_SECRET:-}" ]; then sed "s|__PULL_SECRET__|$PULL_SECRET|"; else grep -v __PULL_SECRET__; fi; } \
  | K apply -f - >/dev/null || { echo "!! apply failed"; exit 1; }
echo "[$(ts)] launched $J bench=$BENCH nn=$NN gbs=$GBS mbs=$MBS cp=$CP tp=$TP seed=$SEED run=$RUN"
fi

t0=$(date +%s); st=""
while :; do
  st=$(K get job "$J" -o jsonpath='{.status.succeeded}/{.status.failed}' 2>/dev/null) || {
    # job gone (another runner already reaped it, or it was deleted): judge from the logs
    # (capture first: under pipefail, `kubectl | grep -q` fails on SIGPIPE even when grep matches)
    o=$(K get job "$J" 2>&1); case "$o" in *NotFound*) st="gone"; break ;; esac; }
  s_=${st%%/*}; f_=${st#*/}
  { [ "${s_:-0}" -ge "$NN" ] || [ "${f_:-0}" -gt 0 ]; } 2>/dev/null && break
  [ $(( $(date +%s) - t0 )) -gt "$TIMEOUT" ] && { st="timeout"; break; }
  # a NaN eval never recovers: stop now (non-converged run) instead of holding nodes to TIMEOUT
  if K exec "${READER_POD:-log-reader}" -- sh -c "grep -q '\"key\": \"eval_accuracy\", \"value\": NaN' /shared/llama-v61/logs/$RUN/rank0.log" 2>/dev/null; then
    echo "[$(ts)] NaN eval detected -- stopping $J (non-converged)"; K delete job,svc "$J" --wait=false >/dev/null 2>&1; st="nan"; break
  fi
  sleep 30
done

# verdict from rank0 log on the shared volume (read through any pod that mounts it)
R0=$(K exec "${READER_POD:-log-reader}" -- sh -c "cat /shared/llama-v61/logs/$RUN/rank0.log 2>/dev/null" 2>/dev/null)
stop=$(echo "$R0" | grep -oE '"key": "run_stop".*' | head -1)
status=$(echo "$stop" | grep -oE '"status": "[a-z]+"' | grep -oE '[a-z]+"$' | tr -d '"')
t_start=$(echo "$R0" | grep -oE '"time_ms": [0-9]+, "event_type": "INTERVAL_START", "key": "run_start"' | grep -oE '[0-9]{13}' | head -1)
t_stop=$(echo "$stop" | grep -oE '"time_ms": [0-9]+' | grep -oE '[0-9]+' | head -1)
[ -z "$t_stop" ] && t_stop=$(echo "$R0" | grep -E '"key": "run_stop"' | grep -oE '"time_ms": [0-9]+' | grep -oE '[0-9]+' | head -1)
ttt=""; [ -n "$t_start" ] && [ -n "$t_stop" ] && ttt=$(( (t_stop - t_start) / 1000 ))
acc=$(echo "$R0" | grep -E '"key": "eval_accuracy"' | grep -oE '"value": [0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
v=FAIL; [ "$status" = success ] && v=PASS
# FAIL before run_start = harness/config problem, not a node fault (scheduler won't blame nodes)
[ "$v" = FAIL ] && [ -z "$t_start" ] && v=FAIL_PRESTART
echo "[$(ts)] VERDICT $v $J bench=$BENCH nn=$NN gbs=$GBS seed=$SEED ttt_s=${ttt:-NA} last_eval=${acc:-NA} job=$st run=$RUN"
[ "${KEEP:-0}" = 1 ] || { K delete job,svc "$J" --ignore-not-found >/dev/null 2>&1; }
[ "$v" = PASS ]
