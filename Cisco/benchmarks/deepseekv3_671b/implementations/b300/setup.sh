#!/usr/bin/env bash
set -euo pipefail
command -v sbatch >/dev/null
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/config_Zeus_8xB300_SXM_288GB_multi_planar.sh"
export HYBRID_EP_SCRIPT_DIR="${SCRIPT_DIR}"
export CONT="${CONT:?Set CONT}"
export HYBRID_EP_TOKENS_PER_RANK=4096 HYBRID_EP_PAD_MULTIPLE=256
sbatch --export=ALL --partition="${SBATCH_PARTITION:?Set SBATCH_PARTITION}" --nodelist="${COMPILE_NODE:?Set COMPILE_NODE}" "${SCRIPT_DIR}/precompile_hybridep_b300.sub"
