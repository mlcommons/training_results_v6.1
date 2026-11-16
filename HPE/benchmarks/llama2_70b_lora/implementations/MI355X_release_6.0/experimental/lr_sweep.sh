#!/bin/bash
#
# Llama 2 LoRA convergence-reliability sweep.
#
# WHY: convergence is quantised to 384 sequences (eval cadence at GBS 8), so the
# only reachable points near the RCP mean of 3,179 are 2,688 / 3,072 / 3,456.
# Measured effective scores: 3,072 -> 8.437 min, 3,456 -> 9.490 min. The optimum
# is therefore to land EVERY run at 3,072, the lowest legal point.
#
# Currently only 11/15 (73%) do. The runs that miss are marginal: eval loss at
# 3,072 lands at 0.9224-0.9243 against a 0.925 target, so a small convergence
# improvement should convert most of the 27% failures.
#
# LEVER: opt_base_learning_rate is explicitly *unconstrained* by training_rules
# for llama2_70b_lora. The reference is 4e-4; AMD already ships 6e-4.
#
# RISK: too high an LR starts landing runs at 2,688. That is not automatically
# good — the olympic mean must stay >= min_epochs (2,961), so more than ~29% of
# the olympic-8 at 2,688 is a HARD FAIL of the whole submission. This sweep
# therefore reports the 2,688 rate explicitly rather than optimising blindly.
#
# METRIC: this measures p(converge at 3072) and TTT-within-class. It is NOT a
# scored run. Seeds stay random, as required.
#
# Usage:  bash lr_sweep.sh [NEXP_PER_POINT]      (default 5, ~45 min per point)

set -euo pipefail
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
NEXP_PER=${1:-5}

# 6e-4 is the current production value and acts as the control arm.
LRS=${LRS:-"0.0006 0.0007 0.0008"}

for lr in ${LRS}; do
    tag="lrsweep-${lr}"
    echo "=== llama2 LR ${lr}, NEXP=${NEXP_PER} -> tag ${tag} ==="
    LOGDIR=/nas/SHARED/MLPERF/results/llama2_70b/${tag} \
    NEXP_OVERRIDE="${NEXP_PER}" \
    LR_OVERRIDE="${lr}" \
        bash "${SCRIPT_DIR}/../run.llama2-70b.training.sh"
done

echo
echo "Analyse with:"
echo "  python3 analyze_results.py llama2_70b_lora /nas/SHARED/MLPERF/results/llama2_70b/lrsweep-*/"
echo "and compare p(3072) per arm - that, not raw TTT, is the figure of merit."
