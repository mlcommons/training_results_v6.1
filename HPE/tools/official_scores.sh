#!/bin/bash
# Official MLPerf scores, exactly as MLCommons computes them: builds a throwaway
# submission package from the given result folders and runs the official
# result summarizer (mlcommons/logging, ruleset 6.1.0). The summarizer runs the
# RCP check and writes each benchmark's scaling.json on its first pass and
# applies it on the second, so it is run twice; the second pass is the
# official, RCP-normalised score.
#
# Run on striker. Each folder must hold one run log per trial (*.log); with more
# than 10 runs, select_runs.py picks the 10-run window closest to the median.
#
# Usage: official_scores.sh <bench>=<dir> [<bench>=<dir> ...]
#   bench: llama31_8b llama2_70b_lora flux1 gpt_oss_20b dlrmv4
# Example:
#   official_scores.sh dlrmv4=/nas/SHARED/MLPERF/results/dlrmv4/amd61-set2
set -euo pipefail
TOOLS=$(dirname "$(readlink -f "$0")")
REPO=$(dirname "$(dirname "${TOOLS}")")
LOGGING=${LOGGING:-${HOME}/mlperf-logging-latest}
PY=${PY:-${HOME}/venv-mlperf/bin/python}
RULESET=${RULESET:-6.1.0}
SID=${SID:-HPE-Compute-XD685-MI355X-288GBx8}
[[ $# -gt 0 ]] || { sed -n 2,15p "$0"; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT
BASE=${WORK}/closed/HPE
mkdir -p "${BASE}/systems" "${BASE}/results/${SID}"
cp "${REPO}/HPE/systems/${SID}.json" "${BASE}/systems/"

benches=()
for arg in "$@"; do
    b=${arg%%=*}; d=${arg#*=}
    dest=${BASE}/results/${SID}/${b}
    mkdir -p "${dest}"
    python3 "${TOOLS}/select_runs.py" "${b}" "${d}" --json |
        python3 -c 'import json,sys,shutil,os
s=json.load(sys.stdin)
for i,f in enumerate(s["files"],1): shutil.copyfile(f, os.path.join(sys.argv[1], f"result_{i}.txt"))' "${dest}"
    benches+=("${b}")
done

cd "${LOGGING}"
for pass in 1 2; do
    PYTHONPATH=. "${PY}" -m mlperf_logging.result_summarizer "${BASE}" training "${RULESET}" \
        --quiet -csv "${WORK}/summary.csv" > "${WORK}/summarizer.${pass}.log" 2>&1 || true
done
grep -h "ERROR" "${WORK}"/summarizer.2.log | sort -u || true

python3 - "${WORK}" "${benches[@]}" <<'PY'
import csv, json, os, sys
work, benches = sys.argv[1], sys.argv[2:]
rows = list(csv.DictReader(open(os.path.join(work, "summary.csv"))))
res = os.path.join(work, "closed/HPE/results")
print(f"{'benchmark':<16} {'official score (min)':>21} {'RCP scaling':>12}")
for b in benches:
    val = next((r.get(b) for r in rows if r.get(b)), "")
    sid = os.listdir(res)[0]
    sc = os.path.join(res, sid, b, "scaling.json")
    f = json.load(open(sc))["scaling_factor"] if os.path.exists(sc) else 1.0
    print(f"{b:<16} {float(val) if val else float('nan'):>21.3f} {f:>12.4f}")
PY
