#!/usr/bin/env bash
set -euo pipefail
command -v sbatch >/dev/null
printf 'Provision the container and datasets listed in run.sh, then run bash run.sh.\n'
