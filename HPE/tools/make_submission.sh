#!/bin/bash
#
# Assemble an MLPerf Training submission package.
#
# Layout produced (what mlperf_logging.package_checker expects, with the
# division/submitter prefix required by the submission rules):
#
#   <out>/<division>/<submitter>/
#       systems/<system_id>.json
#       results/<system_id>/<benchmark>/result_1.txt .. result_10.txt
#       benchmarks/<benchmark>/implementations/<system_id>/<source tree>
#
# The 10 scored runs are chosen by select_runs.py, which takes the window of 10
# consecutive runs closest to the MEDIAN (never the fastest -- that is
# explicitly prohibited).
#
# Usage:
#   make_submission.sh --out DIR [--division closed] [--submitter HPE]
#                      [--system-id HPE-Compute-XD685-MI355X-288GBx8]
#                      [--llama31-results DIR...] [--llama2-results DIR...]
#                      [--flux1-results DIR...] [--gptoss-results DIR...] [--dlrmv4-results DIR...]
#                      [--primus-host DIR]   (Primus v26.7.0 checkout used by Llama 2 and GPT-OSS)
#                      [--code BENCH=DIR]    code that actually ran (e.g. the per-job copy
#                                            <logdir>/code-<jobid>); default: repo implementation
#                      [--extra BENCH=FILE]  extra file to ship (launcher, patch)
#                      [--primus-recipe BENCH=RECIPE] ship Primus examples/mlperf/RECIPE plus
#                                            every locally modified Primus file (if given, replaces
#                                            the old Llama 2 / GPT-OSS defaults)
#                      [--system-json FILE]  system description (default HPE/systems/<id>.json)

set -euo pipefail
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
REPO_ROOT=$(dirname "$(dirname "${SCRIPT_DIR}")")   # HPE/tools -> repo root

OUT=""
DIVISION="closed"
SUBMITTER="HPE"
SYSTEM_ID="HPE-Compute-XD685-MI355X-288GBx8"
IMPL_NAME="MI355X_release_6.0"   # source folder under HPE/benchmarks/<bench>/implementations/
RESULTS_ROOT="/nas/SHARED/MLPERF/results"
LLAMA31_DIRS=()
LLAMA2_DIRS=()
FLUX1_DIRS=()
GPTOSS_DIRS=()
DLRMV4_DIRS=()
PRIMUS_HOST="/home/hpebench/primus/v26.7.0"
declare -A CODE_DIR=()
EXTRAS=()
PRIMUS_RECIPES=()
SYS_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --division) DIVISION="$2"; shift 2;;
    --submitter) SUBMITTER="$2"; shift 2;;
    --system-id) SYSTEM_ID="$2"; shift 2;;
    --results-root) RESULTS_ROOT="$2"; shift 2;;
    --llama31-results) LLAMA31_DIRS+=("$2"); shift 2;;
    --llama2-results) LLAMA2_DIRS+=("$2"); shift 2;;
    --flux1-results) FLUX1_DIRS+=("$2"); shift 2;;
    --gptoss-results) GPTOSS_DIRS+=("$2"); shift 2;;
    --dlrmv4-results) DLRMV4_DIRS+=("$2"); shift 2;;
    --primus-host) PRIMUS_HOST="$2"; shift 2;;
    --code) CODE_DIR["${2%%=*}"]="${2#*=}"; shift 2;;
    --extra) EXTRAS+=("$2"); shift 2;;
    --primus-recipe) PRIMUS_RECIPES+=("$2"); shift 2;;
    --system-json) SYS_JSON="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

: "${OUT:?--out is required}"
[[ ${#LLAMA31_DIRS[@]} -eq 0 ]] && LLAMA31_DIRS=("${RESULTS_ROOT}/llama31_8b/final")
[[ ${#LLAMA2_DIRS[@]}  -eq 0 ]] && LLAMA2_DIRS=("${RESULTS_ROOT}/llama2_70b/final")

BASE="${OUT}/${DIVISION}/${SUBMITTER}"
echo "=== building submission at ${BASE} ==="
rm -rf "${OUT}"
mkdir -p "${BASE}/systems" "${BASE}/results/${SYSTEM_ID}" "${BASE}/benchmarks"

# ---------------------------------------------------------------- system desc
# Prefer the real system description committed under HPE/systems/; fall back
# to the template only for a system that has none yet.
SYS_SRC="${SYS_JSON:-${REPO_ROOT}/HPE/systems/${SYSTEM_ID}.json}"
[[ -f "${SYS_SRC}" ]] || SYS_SRC="${SCRIPT_DIR}/system_desc_template.json"
python3 - "${SYS_SRC}" \
         "${BASE}/systems/${SYSTEM_ID}.json" \
         "$DIVISION" "$SUBMITTER" <<'PY'
import json, sys
src, dst, division, submitter = sys.argv[1:5]
d = json.load(open(src))
d["division"] = division
d["submitter"] = submitter
# The checker requires every value to be a string or number; flatten the nested
# software-stack dict into a single descriptive string.
if isinstance(d.get("other_software_stack"), dict):
    d["other_software_stack"] = ", ".join(f"{k}: {v}" for k, v in d["other_software_stack"].items())
json.dump(d, open(dst, "w"), indent=2)
print(f"  wrote {dst}")
PY

# ------------------------------------------------------------------- results
INCOMPLETE=0
copy_results () {
  local bench="$1"; shift
  local dirs=("$@")
  local dest="${BASE}/results/${SYSTEM_ID}/${bench}"
  echo ""
  echo "--- ${bench} ---"
  local sel
  # Skip rather than abort when a benchmark does not yet have a full set, so the
  # package can be assembled and validated incrementally while runs finish.
  if ! sel=$(python3 "${SCRIPT_DIR}/select_runs.py" "${bench}" "${dirs[@]}" --json 2>&1); then
    echo "  SKIPPED: ${sel}"
    INCOMPLETE=1
    return 0
  fi
  mkdir -p "${dest}"
  echo "${sel}" | python3 -c "
import json,sys,shutil,os
s=json.load(sys.stdin); dest=sys.argv[1]
for i,f in enumerate(s['files'],1):
    shutil.copyfile(f, os.path.join(dest, f'result_{i}.txt'))
print(f\"  selected window starting at run {s['window_start_index']+1}\")
print(f\"  olympic score  : {s['olympic_score_min']:.3f} min\")
print(f\"  olympic samples: {s['olympic_samples']:.0f}\")
print(f\"  copied {len(s['files'])} result files\")
" "${dest}"
}

copy_results llama31_8b      "${LLAMA31_DIRS[@]}"
copy_results llama2_70b_lora "${LLAMA2_DIRS[@]}"
[[ ${#FLUX1_DIRS[@]}  -gt 0 ]] && copy_results flux1       "${FLUX1_DIRS[@]}"
[[ ${#GPTOSS_DIRS[@]} -gt 0 ]] && copy_results gpt_oss_20b "${GPTOSS_DIRS[@]}"
[[ ${#DLRMV4_DIRS[@]} -gt 0 ]] && copy_results dlrmv4      "${DLRMV4_DIRS[@]}"

# ---------------------------------------------------------------------- code
# The seed checker scans these sources to verify no two runs log the same seed
# from the same file:line, so the real training sources must be included.
copy_code () {
  local bench="$1" srcdir="$2"
  local dest="${BASE}/benchmarks/${bench}/implementations/${SYSTEM_ID}"
  mkdir -p "${dest}"
  # Ship sources only. The 6.1 repo_checker rejects names starting with "."
  # (.dockerignore, Hydra's outputs/*/.hydra). Exclude Primus run artifacts (code/output/... per-rank
  # logs land in the bind-mounted tree), caches, images and staged data, which
  # would otherwise bloat the package by gigabytes.
  rsync -a \
        --exclude '__pycache__' --exclude '*.pyc' \
        --exclude '*.tar' --exclude '*.tar.gz' \
        --exclude 'output/' --exclude 'results/' --exclude 'probes/' \
        --exclude '*.log' --exclude '.git/' --exclude 'experimental/' \
        --exclude '.*' --exclude 'outputs/' --exclude 'nemo_experiments/' --exclude 'logs/' \
        "${srcdir}/" "${dest}/"
  local n
  n=$(find "${dest}" -type f | wc -l)
  echo "  ${bench}: ${n} files from ${srcdir}"
}
echo ""
echo "--- source code ---"
# The repo already mirrors the MLCommons tree, so the implementation sources
# are copied straight across rather than rearranged.
IMPL_ROOT="${REPO_ROOT}/HPE/benchmarks"
src_for () { echo "${CODE_DIR[$1]:-${IMPL_ROOT}/$1/implementations/${IMPL_NAME}}"; }
copy_code llama31_8b      "$(src_for llama31_8b)"
copy_code llama2_70b_lora "$(src_for llama2_70b_lora)"
[[ ${#FLUX1_DIRS[@]}  -gt 0 ]] && copy_code flux1       "$(src_for flux1)"
[[ ${#GPTOSS_DIRS[@]} -gt 0 ]] && copy_code gpt_oss_20b "$(src_for gpt_oss_20b)"
# DLRMv4 exists only as the 6.1 drop.
[[ ${#DLRMV4_DIRS[@]} -gt 0 ]] && copy_code dlrmv4      "${CODE_DIR[dlrmv4]:-${IMPL_ROOT}/dlrmv4/implementations/MI355X_release_6.1}"
for e in "${EXTRAS[@]}"; do
  b="${e%%=*}"; f="${e#*=}"
  [[ -f "$f" ]] || { echo "  ERROR: extra file $f missing" >&2; exit 1; }
  cp "$f" "${BASE}/benchmarks/${b}/implementations/${SYSTEM_ID}/"
  echo "  ${b}: + $(basename "$f")"
done

# Llama 2 and GPT-OSS scored sets ran AMD's Primus v26.7.0 MLPerf recipes from a
# host checkout (mounted over the image's /workspace/Primus). Ship that recipe
# code plus the exact Primus revision next to our launchers.
add_primus_recipe () {
  local bench="$1" recipe="$2" launcher="$3"
  local dest="${BASE}/benchmarks/${bench}/implementations/${SYSTEM_ID}/primus_v26.7.0"
  [[ -d "${PRIMUS_HOST}/examples/mlperf/${recipe}" ]] || { echo "  WARNING: no ${PRIMUS_HOST}/examples/mlperf/${recipe}"; return 0; }
  mkdir -p "${dest}/examples/mlperf"
  rsync -a --exclude data --exclude logs --exclude '*.log' --exclude output --exclude '.*' --exclude __pycache__ \
        "${PRIMUS_HOST}/examples/mlperf/${recipe}" "${dest}/examples/mlperf/"
  ( cd "${PRIMUS_HOST}" && echo "Primus $(git log -1 --format='%H %cs %D')"; git status --short | sed 's/^/local change: /' ) > "${dest}/PRIMUS_REVISION.txt"
  # Locally modified Primus files (outside the recipe folder too), same paths.
  ( cd "${PRIMUS_HOST}" && git diff --name-only ) | while read -r f; do
    [[ -f "${PRIMUS_HOST}/$f" ]] && { mkdir -p "${dest}/$(dirname "$f")"; cp "${PRIMUS_HOST}/$f" "${dest}/$f"; }
  done
  ( cd "${PRIMUS_HOST}" && git diff ) > "${dest}/PRIMUS_LOCAL_CHANGES.patch"
  [[ -n "${launcher}" && -f "${launcher}" ]] && cp "${launcher}" "${dest}/"
  echo "  ${bench}: + Primus v26.7.0 recipe examples/mlperf/${recipe}"
}
if [[ ${#PRIMUS_RECIPES[@]} -gt 0 ]]; then
  for r in "${PRIMUS_RECIPES[@]}"; do add_primus_recipe "${r%%=*}" "${r#*=}" ""; done
else
  add_primus_recipe llama2_70b_lora llama2_70b "${REPO_ROOT}/HPE/tools/run_primus_llama.sbatch"
  [[ ${#GPTOSS_DIRS[@]} -gt 0 ]] && add_primus_recipe gpt_oss_20b gpt_oss_20b "${IMPL_ROOT}/gpt_oss_20b/implementations/${IMPL_NAME}/experimental/run_amd_primus_v265.sbatch"
fi

echo ""
echo "=== package tree ==="
find "${OUT}" -maxdepth 5 -type d | sed "s|${OUT}|.|" | sort
echo ""
echo "result files:"
find "${BASE}/results" -name 'result_*.txt' | wc -l
if [[ ${INCOMPLETE} -eq 1 ]]; then
  echo ""
  echo "WARNING: package is INCOMPLETE - at least one benchmark lacks a full run set."
  echo "         Validation will report missing benchmarks until all runs finish."
fi
echo ""
echo "Next: tools/validate_submission.sh --package ${OUT}"
