#!/bin/bash
exec numactl --cpunodebind=$(( ${SLURM_LOCALID:-0} / 2 )) "$@"
