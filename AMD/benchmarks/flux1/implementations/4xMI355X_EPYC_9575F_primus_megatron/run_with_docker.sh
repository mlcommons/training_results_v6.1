#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Run on a single node without Slurm (development; the submission config is multi-node).
# Same variables as run_with_docker_slurm.sh.
export NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} MASTER_ADDR=${MASTER_ADDR:-localhost}
exec bash "$(dirname "$(readlink -f "$0")")/run_with_docker_slurm.sh" "$@"
