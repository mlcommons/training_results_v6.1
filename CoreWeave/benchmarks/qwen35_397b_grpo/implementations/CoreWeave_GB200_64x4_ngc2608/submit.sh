#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export NEXP="${NEXP:-3}"
TRIAL_SEEDS=$(python3 - <<'PYSEEDS'
import os
import re
import secrets
import sys

def fail(message):
    sys.exit("ERROR: " + message)

raw_n = os.environ["NEXP"]
if not re.fullmatch(r"[1-9][0-9]*", raw_n):
    fail("NEXP must be a positive integer without leading zeroes")
n = int(raw_n)
maximum = 2147483647
if n > maximum:
    fail("NEXP exceeds the number of distinct supported seeds")
plan = os.environ.get("TRIAL_SEEDS", "").split()
if plan:
    if len(plan) != n:
        fail("TRIAL_SEEDS must contain exactly NEXP seeds")
    if any(not re.fullmatch(r"[1-9][0-9]{0,9}", s) for s in plan):
        fail("TRIAL_SEEDS must contain positive integers without leading zeroes")
    seeds = [int(s) for s in plan]
    if any(s > maximum for s in seeds):
        fail("TRIAL_SEEDS values must be <= 2147483647")
    if len(set(seeds)) != n:
        fail("duplicate seed in TRIAL_SEEDS")
else:
    seeds, seen = [], set()
    while len(seeds) < n:
        seed = secrets.randbelow(maximum) + 1
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
print(" ".join(map(str, seeds)))
PYSEEDS
)
export TRIAL_SEEDS
export SEED="${TRIAL_SEEDS%% *}"
source ./config_site.sh
NODELIST="${NODELIST:?set NODELIST to the 64-node allocation}"
printf 'Nodes=%s training=%s generation=%s walltime_min=%s training_min=%s eval_min=%s\n' "$DGXNNODES" "$TRAIN_NODES" "$GEN_NODES" "$WALLTIME" "$WALLTIME_RUNANDTIME" "$WALLTIME_OFFLINE_EVAL"
printf 'NEXP=%s\nTRIAL_SEEDS=%s\nLOGDIR=%s\n' "$NEXP" "$TRIAL_SEEDS" "$LOGDIR"
if [[ "${1:-}" == --show ]]; then exit 0; fi
mkdir -p "$LOGDIR"
_seed_plan_file=$(mktemp "$LOGDIR/seed-plan.XXXXXX")
printf 'export NEXP=%q\nexport TRIAL_SEEDS=%q\n' "$NEXP" "$TRIAL_SEEDS" > "$_seed_plan_file"
printf 'Seed plan saved to %s\n' "$_seed_plan_file"
sbatch --parsable ${SLURM_ACCOUNT:+"--account=$SLURM_ACCOUNT"} --partition=gb200 --gres=gpu:4 \
  --no-requeue --nodes="$DGXNNODES" --nodelist="$NODELIST" \
  --time="$WALLTIME" --job-name=qwen35-grpo \
  --output="$LOGDIR/slurm-%j.out" --error="$LOGDIR/slurm-%j.out" \
  "$@" "$PWD/job.sub"
