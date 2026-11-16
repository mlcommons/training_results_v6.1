#!/bin/bash
# Render + run one AMD v6.1 customer-drop (flux1 | dlrmv4) test and print a one-line verdict.
#   CTX=<ctx> NS=<ns> IMAGE=<image> BENCH=flux1|dlrmv4 NN=<nodes> GBS=<gbs> MBS=<mbs> [EXTRA=<VAR=val ...>] \
#   SELKEY=<label-key> SELVAL=<label-value> [DEADLINE=<s>] [SEED=<n>] [LR=<lr>] [PULL_SECRET=<secret>] bash run-cdrop.sh
# Verdict line: VERDICT <CONVERGED|RUNNING_AT_DEADLINE|FAIL> bench= nn= gbs= ttt_s= last_eval= evals= samples_per_s= run=
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${CTX:?}" "${NS:?}" "${IMAGE:?}" "${BENCH:?}" "${NN:?}" "${GBS:?}" "${MBS:?}" "${SELKEY:?}" "${SELVAL:?}"
DEADLINE=${DEADLINE:-14400}; SEED=${SEED:-$RANDOM}; LR=${LR:-}; PULL_SECRET=${PULL_SECRET:-ccr}
KC(){ kubectl --context "$CTX" -n "$NS" --request-timeout=30s "$@"; }
ts(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
TAG=$BENCH; [ "$BENCH" = flux1 ] && TAG=flux; [ "$BENCH" = dlrmv4 ] && TAG=dlrm
case "$BENCH" in flux1) BLOCK=262144 ;; dlrmv4) BLOCK=2290000 ;; *) BLOCK=12288 ;; esac
J="v61-${TAG}-${NN}n${JSUF:-}"; RUN="${J}-$(date -u +%m%d%H%M)-s${SEED}"

if [ -n "${FOLLOW_RUN:-}" ]; then
  # Re-attach to a job that is already running (its runner died): no delete/apply, just follow + verdict.
  RUN=$FOLLOW_RUN; SEED=${RUN##*-s}; ts "following $J (run=$RUN)"
else
KC create configmap cdrop-launch --from-file=launch-cdrop.sh="$HERE/launch-cdrop.sh" --dry-run=client -o yaml | KC apply -f - >/dev/null
KC delete job "$J" --ignore-not-found --wait=true >/dev/null 2>&1; KC delete svc "$J" --ignore-not-found >/dev/null 2>&1
sed -e "s|__J__|$J|g; s|__N__|$NN|g; s|__BENCH__|$BENCH|g; s|__GBS__|$GBS|g; s|__MBS__|$MBS|g; s|__SEED__|$SEED|g" \
    -e "s|__RUN__|$RUN|g; s|__IMAGE__|$IMAGE|g; s|__SELKEY__|$SELKEY|g; s|__SELVAL__|$SELVAL|g; s|__DEADLINE__|$DEADLINE|g" \
    -e "s|__LR__|$LR|g; s|__PULL_SECRET__|$PULL_SECRET|g; s|__EXTRA__|${EXTRA:-}|g; s|__PRIV__|${PRIV:-false}|g" "$HERE/train-cdrop.template.yaml" | KC apply -f - >/dev/null || { echo "!! apply failed"; exit 1; }
ts "applied $J (bench=$BENCH nn=$NN gbs=$GBS mbs=$MBS seed=$SEED deadline=${DEADLINE}s run=$RUN)"
fi

# rank-0 output: pod log, plus gpt_oss's full trainer log (its run_and_time.sh tees the trainer there and
# filters stdout); after the pod exits, the launcher's copy on the shared volume via the log-reader pod.
r0(){ P=$(KC get pods -l job-name="$J" -o name 2>/dev/null | grep -- '-0-' | head -1)
      { KC logs "$P" --tail=-1 2>/dev/null; KC exec "${P#pod/}" -c train -- sh -c 'cat /results/artifacts/*.full.log 2>/dev/null' 2>/dev/null \
        || KC exec log-reader -- sh -c "cat /shared/v61drop/logs/$RUN/*.full.log /shared/v61drop/logs/$RUN/rank0.log 2>/dev/null" 2>/dev/null; } \
        | sed 's/\x1b\[[0-9;]*m//g' | awk '!seen[$0]++'; }     # two copies of the rank-0 log: count each MLLOG event once
while :; do
  s=$(KC get job "$J" -o jsonpath='{.status.succeeded}{" "}{.status.failed}{" "}{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  read -r ok bad cond <<<"$s"
  # Gone only on an explicit NotFound: API timeouts / a sleeping client must not end a live run (2026-10-14 false FAILs).
  # (Captured, not piped into grep -q: with pipefail, kubectl's own exit 1 would mask the match.)
  if [[ "$(KC get job "$J" 2>&1 >/dev/null)" == *NotFound* ]]; then
    sleep 30; [[ "$(KC get job "$J" 2>&1 >/dev/null)" == *NotFound* ]] && { ts "job $J no longer exists"; cond=Failed; break; }
  fi
  ev=$(r0 | grep -oE '"key": "eval_accuracy", "value": [0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
  it=$(r0 | grep -oE 'iteration +[0-9]+/ *[0-9]+' | tail -1)
  ts "succeeded=${ok:-0} failed=${bad:-0} ${cond:-} ${it:-} last_eval=${ev:-none}"
  case "$cond" in *Complete*|*Failed*) break ;; esac
  sleep 120
done

L=$(r0)
start=$(echo "$L" | grep -oE '"time_ms": [0-9]+, "event_type": "INTERVAL_START", "key": "run_start"' | grep -oE '[0-9]{13}' | head -1)
stop=$(echo "$L" | grep -E '"key": "run_stop"' | grep -oE '"time_ms": [0-9]+' | grep -oE '[0-9]+' | tail -1)
status=$(echo "$L" | grep -E '"key": "run_stop"' | grep -oE '"status": "[a-z]+"' | grep -oE '"[a-z]+"$' | tr -d '"' | tail -1)
evs=$(echo "$L" | grep -oE '"key": "eval_accuracy", "value": [0-9.]+' | grep -oE '[0-9.]+$' | tr '\n' ',' | sed 's/,$//')
nev=$(echo "$evs" | tr ',' '\n' | grep -c .); last=$(echo "$evs" | tr ',' '\n' | tail -1)
sps=$(echo "$L" | grep -oE '"key": "throughput", "value": \{"throughput": [0-9.]+' | grep -oE '[0-9.]+$' | tail -5 | awk '{s+=$1;n++} END{if(n) printf "%.1f", s/n}')
ttt=""; [ -n "$start" ] && [ -n "$stop" ] && ttt=$(( (stop - start) / 1000 ))
if [ "$status" = success ]; then v=CONVERGED
elif echo "$cond" | grep -q Failed && echo "$L" | grep -q "DeadlineExceeded\|run_start" && [ -z "$status" ] && [ "${nev:-0}" -gt 0 ]; then v=RUNNING_AT_DEADLINE
else v=FAIL; fi
KC get job "$J" -o jsonpath='{.status.conditions[?(@.status=="True")].reason}' 2>/dev/null | grep -q DeadlineExceeded && [ "$v" != CONVERGED ] && [ "${nev:-0}" -gt 0 ] && v=RUNNING_AT_DEADLINE
# Throughput from MLLOG training blocks: each block_start..block_stop spans one eval interval of training
# (12,288 samples), eval excluded. Median of blocks after the first (warm-up/compile). Also Megatron's
# per-step time when log_interval is on (perf mode).
bsps=$(echo "$L" | python3 -c '
import sys,re,statistics as st
s=None; d=[]
for ln in sys.stdin:
    m=re.search(r"\"time_ms\": (\d+), \"event_type\": \"INTERVAL_(START|END)\", \"key\": \"block_(start|stop)\"",ln)
    if not m: continue
    if m.group(3)=="start": s=int(m.group(1))
    elif s: d.append((int(m.group(1))-s)/1000.0); s=None
d=d[1:] if len(d)>1 else d
print("%.1f" % (float(sys.argv[1])/st.median(d)) if d else "none")' $BLOCK)
step_ms=$(echo "$L" | grep -oE "elapsed time per iteration \(ms\): [0-9.]+" | tail -n +4 | awk '{print $NF}' | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]; else print "none"}')
echo "VERDICT $v bench=$BENCH nn=$NN gbs=$GBS mbs=$MBS seed=$SEED ttt_s=${ttt:-none} last_eval=${last:-none} evals=${nev:-0} samples_per_s=${bsps:-none} step_ms=${step_ms:-none} tp=${TP:-1} cp=${CP:-1} extra=[${EXTRA:-}] run=$RUN job=$J"
echo "EVALS $RUN $evs"
