#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:?WORKDIR required}"
SRC="${WORKDIR}/run.sub"
DST="${WORKDIR}/run.sub.vr"

[[ -f "${SRC}" ]] || { echo "LAUNCH BLOCKED: missing ${SRC}" >&2; exit 2; }

python3 - "${SRC}" "${DST}" <<'PY'
from pathlib import Path
import re
import sys

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text()
text, dropped = re.subn(r"[ \t]*--container-env=\S*MASTER_ADDR\S*", "", text)
if dropped == 0:
    raise SystemExit("LAUNCH BLOCKED: MASTER_ADDR allowlist not found in run.sub")
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
text = text.replace(needle, insert, 1)
dst.write_text(text)
print(dst)
PY

chmod 775 "${DST}"

grep -E -q -- '--container-env=\S*MASTER_ADDR' "${DST}" && { echo "LAUNCH BLOCKED: allowlist remains" >&2; exit 2; }
grep -E -q '^[[:space:]]*,' "${DST}" && { echo "LAUNCH BLOCKED: dangling comma argument" >&2; exit 2; }
grep -q 'resolved MASTER_ADDR' "${DST}" || { echo "LAUNCH BLOCKED: ipv4 resolve missing" >&2; exit 2; }
echo "OK ${DST}"
