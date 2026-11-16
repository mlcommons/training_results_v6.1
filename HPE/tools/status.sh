#!/bin/bash
# Compact cluster status, times in Paris local.
# Usage: status.sh   (run on striker as hpebench)
set -uo pipefail
R=/nas/SHARED/MLPERF/results
TZP="Europe/Paris"

echo "STATUS  $(TZ=$TZP date '+%H:%M %Z')   (UTC $(date -u '+%H:%M'))"
echo
printf '%-5s %-15s %-3s %-9s %-9s %s\n' JOB NAME ST ELAPSED LEFT NODE
squeue -u hpebench -h -o '%i|%j|%t|%M|%L|%R' | while IFS='|' read -r i j t m l r; do
    printf '%-5s %-15s %-3s %-9s %-9s %s\n' "$i" "${j:0:15}" "$t" "$m" "$l" "$r"
done
echo

# llama2 sweep: convergence hit-rate is the figure of merit, not raw TTT.
REPO=${REPO:-$HOME/mlcommons_mlperf_training}
IMPL=${REPO}/HPE/benchmarks/llama2_70b_lora/implementations/MI355X_release_6.0
if compgen -G "$R/llama2_70b/lrsweep-0.*/*.log" > /dev/null 2>&1; then
    python3 "$IMPL/llama2_convergence.py" $R/llama2_70b/lrsweep-0.*/ 2>/dev/null | head -6
    echo
fi

# flux1 conversion progress on whichever node is converting
for n in lodin1 lodin2; do
    s=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$n" \
        'ls /lvol0/mlperf/flux1/energon/train/*.tar 2>/dev/null | wc -l; ls -d /lvol0/mlperf/flux1/energon/.nv-meta >/dev/null 2>&1 && echo prepared || echo raw' 2>/dev/null)
    [ -z "$s" ] && continue
    printf 'flux1 %-7s shards %3s/110  %s\n' "$n" $(echo "$s" | tr '\n' ' ')
done

b=$(du -sm /nas/SHARED/MLPERF/flux1_energon 2>/dev/null | cut -f1)
[ -n "$b" ] && printf 'flux1 backup   %s of 2282000 MB (%d%%)\n' "$b" $((b*100/2282000))
