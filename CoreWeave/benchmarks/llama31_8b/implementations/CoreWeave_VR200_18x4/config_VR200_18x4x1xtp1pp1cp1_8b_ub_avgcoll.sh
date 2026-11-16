source $(dirname ${BASH_SOURCE[0]})/config_VR200_18x4x1xtp1pp1cp1_8b_ub.sh

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AVERAGE_IN_COLLECTIVE=True
export EXTRA_MOUNTS="${_HERE}/patches/pretrain_avgcoll.py:/workspace/llm/pretrain.py:ro"
