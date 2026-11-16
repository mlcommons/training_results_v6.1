#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodelist=lodin2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=8:00:00
#SBATCH --job-name=llama2-lrsweep
#SBATCH --output=/nas/SHARED/MLPERF/results/llama2_70b/lrsweep-%j.out
#SBATCH --error=/nas/SHARED/MLPERF/results/llama2_70b/lrsweep-%j.err
#
# Llama 2 LoRA learning-rate sweep, measuring convergence RELIABILITY.
#
# Figure of merit is p(land at 3,072 samples), NOT raw TTT. Convergence is
# quantised to the 384-sequence eval grid, and measured effective scores are
# 3,072 -> 8.437 min versus 3,456 -> 9.490 min, so every run that slips to
# 3,456 is a large loss. Current rate is 11/15 = 73%.
#
# The misses look convertible: runs landing at 3,072 show eval loss
# 0.9224-0.9243 against a 0.925 target, i.e. barely inside. A small convergence
# improvement should pull most of the 27% under the bar.
#
# opt_base_learning_rate is explicitly *unconstrained* for llama2_70b_lora in
# training_rules.adoc. Reference is 4e-4; AMD ships 6e-4 (our control arm).
#
# CAUTION: higher LR may start landing runs at 2,688. That is NOT automatically
# good -- the olympic mean must stay >= min_epochs (2,961), so more than ~29% of
# the olympic-8 at 2,688 HARD FAILS the whole submission. llama2_convergence.py
# reports the 2,688 count explicitly for this reason.
#
# Seeds stay random per run (rules forbid seed selection); this measures a rate
# over 5 runs per arm, not a single lucky run.

set -euo pipefail

# Path to this benchmark's implementation directory. The repo mirrors the
# MLCommons submission tree, so this is deep; override IMPL to relocate.
# NOTE: sbatch may execute from a spool copy, so $0 is not reliable here.
IMPL=${IMPL:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}   # submit from this directory
NEXP_PER=${NEXP_PER:-5}
LRS=${LRS:-"0.0006 0.0007 0.0008"}

# Without this, run_and_time.sh discards stderr and filters stdout to three
# patterns, so a crashed rank produces a ~3 KB log and nothing else. This is
# the single most important variable when a llama2 run "hangs".
export MLLOG_VERBOSE_LOGS=1

echo "=== llama2 LR sweep: $(date -u) on $(hostname) ==="
echo "    impl: ${IMPL}"
echo "    arms: ${LRS}   NEXP per arm: ${NEXP_PER}"

[[ -x "${IMPL}/run.llama2-70b.training.sh" ]] \
    || { echo "FATAL: runner not found under ${IMPL}" >&2; exit 1; }

# A job still in CG on this node will run the epilog's disable-docker.sh and
# destroy our container mid-run, with no traceback anywhere. Jobs 55 and 57
# died exactly this way. Wait it out rather than racing it.
for _ in $(seq 1 60); do
    others=$(squeue -h -w "$(hostname -s)" -o '%i %t' | awk -v me="${SLURM_JOB_ID:-0}" '$1!=me && $2=="CG"' | wc -l)
    [[ "${others}" -eq 0 ]] && break
    echo "    waiting: another job is still CG on this node..."
    sleep 10
done

# Confirm the daemon the prolog was supposed to start is actually usable.
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available (is --comment=docker set?)" >&2; exit 1; }
echo "    docker OK, node clear"

# Smoke test one run before committing ~3 h to the full sweep. Job 55 burned an
# allocation discovering a launch failure that a single run would have caught.
echo
echo "--- smoke test: 1 run at the control LR ($(date -u)) ---"
# One fresh folder per job: a stale log would satisfy the check below, and
# old folders hold root-owned Primus logs that hpebench cannot delete.
smoke=/nas/SHARED/MLPERF/results/llama2_70b/smoke-${SLURM_JOB_ID:-$$}
mkdir -p "${smoke}"
if LOGDIR="${smoke}" NEXP_OVERRIDE=1 LR_OVERRIDE=0.0006 \
       bash "${IMPL}/run.llama2-70b.training.sh"; then
    if compgen -G "${smoke}/*.log" > /dev/null; then
        echo "smoke OK: $(ls "${smoke}"/*.log | wc -l) log(s) produced"
    else
        echo "FATAL: smoke run exited 0 but produced no log" >&2
        exit 1
    fi
else
    echo "FATAL: smoke run failed; aborting before the full sweep" >&2
    exit 1
fi

for lr in ${LRS}; do
    tag="${TAG_PREFIX:-lrsweep}-${lr}"
    out=/nas/SHARED/MLPERF/results/llama2_70b/${tag}
    echo
    echo "--- arm LR=${lr} -> ${out} ($(date -u)) ---"
    mkdir -p "${out}"

    # LR_OVERRIDE, not LR: config_MI355X_1x8x1.sh does an unconditional
    # `export LR=0.0006` when sourced, so a plain LR= is silently discarded.
    LOGDIR="${out}" \
    NEXP_OVERRIDE="${NEXP_PER}" \
    LR_OVERRIDE="${lr}" \
        bash "${IMPL}/run.llama2-70b.training.sh" || echo "arm ${lr} exited $?"
done

echo
echo "=== sweep done: $(date -u) ==="
python3 "${IMPL}/llama2_convergence.py" \
    /nas/SHARED/MLPERF/results/llama2_70b/lrsweep-*/ 2>&1 || true
