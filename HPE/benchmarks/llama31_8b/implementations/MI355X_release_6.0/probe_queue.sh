#!/bin/bash
#
# Run a sequence of throughput probes back to back and print a summary table.
# Each entry is: "<label>|<VAR=VAL> <VAR=VAL> ..."
#
# Usage: probe_queue.sh <queuefile>

set -uo pipefail
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
QUEUE="${1:?usage: probe_queue.sh <queuefile>}"
PROBE_ROOT=${PROBE_ROOT:-/nas/SHARED/MLPERF/probes}

while IFS= read -r line; do
  [[ -z "${line// }" || "${line:0:1}" == "#" ]] && continue
  label="${line%%|*}"
  overrides="${line#*|}"
  [[ "$overrides" == "$line" ]] && overrides=""
  echo ""
  echo "############################################################"
  echo "# $(date +%T)  probe: ${label}  [${overrides}]"
  echo "############################################################"
  # shellcheck disable=SC2086
  "${SCRIPT_DIR}/probe.sh" "${label}" ${overrides} 2>&1 | tail -20
done < "${QUEUE}"

echo ""
echo "================= PROBE SUMMARY ================="
python3 - "${PROBE_ROOT}" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(root, "*", "result.json")):
    try:
        rows.append(json.load(open(f)))
    except Exception:
        pass
rows.sort(key=lambda r: -r["median_steady_samples_per_s"])
base = next((r for r in rows if r["label"] == "stock_local"), None)
print(f"{'label':<22}{'median samp/s':>14}{'best':>9}{'proj TTT min':>14}{'vs stock':>10}")
for r in rows:
    d = ""
    if base and base["median_steady_samples_per_s"]:
        d = f"{100*(r['median_steady_samples_per_s']/base['median_steady_samples_per_s']-1):+.2f}%"
    print(f"{r['label']:<22}{r['median_steady_samples_per_s']:>14.2f}"
          f"{r['best_steady_samples_per_s']:>9.2f}{r['projected_ttt_min']:>14.2f}{d:>10}")
PY
