#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:?WORKDIR required}"
SRC="${WORKDIR}/run.sub"
DST="${WORKDIR}/run.sub.use2a"

[[ -f "${SRC}" ]] || { echo "LAUNCH BLOCKED: missing ${SRC}" >&2; exit 2; }

sed 's/ --network=sharp//' "${SRC}" > "${DST}"

python3 - "${DST}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
text, dropped = re.subn(r"[ \t]*--container-env=\S*MASTER_ADDR\S*", "", text)
if dropped == 0 and "--container-env=" in text and "MASTER_ADDR" in text:
    raise SystemExit("LAUNCH BLOCKED: MASTER_ADDR allowlist not removed")
print("dropped container-env tokens", dropped)
needle = 'echo "using MASTER_ADDR \\"${MASTER_ADDR}\\" of list \\"${SLURM_JOB_NODELIST}\\""'
insert = r'''echo "using MASTER_ADDR \"${MASTER_ADDR}\" of list \"${SLURM_JOB_NODELIST}\""
if [[ ! "${MASTER_ADDR}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  _resolved="$(getent ahostsv4 "${MASTER_ADDR}" | awk '{print $1; exit}')"
  if [[ -z "${_resolved}" ]]; then
    echo "LAUNCH BLOCKED: cannot resolve MASTER_ADDR=${MASTER_ADDR} to IPv4" >&2
    exit 2
  fi
  echo "resolved MASTER_ADDR ${MASTER_ADDR} -> ${_resolved}"
  export MASTER_ADDR="${_resolved}"
fi'''
if needle not in text:
    raise SystemExit("LAUNCH BLOCKED: MASTER_ADDR echo line not found in run.sub")
if "resolved MASTER_ADDR" not in text:
    text = text.replace(needle, insert, 1)
pyxis = "PYXIS_DEFAULTS=( '--no-container-mount-home' '--container-remap-root' '--container-writable' )"
override = pyxis + r'''
if [[ -n "${NCCL_PROFILER_PLUGIN_OVERRIDE:-}" ]]; then
  export NCCL_PROFILER_PLUGIN="${NCCL_PROFILER_PLUGIN_OVERRIDE}"
  PYXIS_DEFAULTS+=( '--container-env=NCCL_PROFILER_PLUGIN' )
fi'''
if pyxis not in text:
    raise SystemExit("LAUNCH BLOCKED: PYXIS_DEFAULTS line not found in run.sub")
if "NCCL_PROFILER_PLUGIN_OVERRIDE" not in text:
    text = text.replace(pyxis, override, 1)
path.write_text(text)
print(path)
PY

chmod 775 "${DST}"

grep -q -- '--network=sharp' "${DST}" && { echo "LAUNCH BLOCKED: sharp remains" >&2; exit 2; }
grep -E -q -- '--container-env=\S*MASTER_ADDR' "${DST}" && { echo "LAUNCH BLOCKED: allowlist remains" >&2; exit 2; }
grep -E -q '^[[:space:]]*,' "${DST}" && { echo "LAUNCH BLOCKED: dangling comma argument" >&2; exit 2; }
grep -q 'resolved MASTER_ADDR' "${DST}" || { echo "LAUNCH BLOCKED: ipv4 resolve missing" >&2; exit 2; }
grep -q 'NCCL_PROFILER_PLUGIN_OVERRIDE' "${DST}" || { echo "LAUNCH BLOCKED: profiler plugin override missing" >&2; exit 2; }
echo "OK ${DST}"
