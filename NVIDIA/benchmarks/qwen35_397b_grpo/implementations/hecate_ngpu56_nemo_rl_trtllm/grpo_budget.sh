#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Teardown budget shared by allocation sizing (config_common.sh, sourced on the
# login node with DGXNNODES) and the batch script (cleanup_grpo_scratch.sh and
# grpo_teardown.sh, with SLURM_JOB_NUM_NODES), so the allocation and the
# deferred-launch check use one calculation. Pure functions, no side effects.

# Deadline, in seconds, for one srun sweep over every node of an allocation of
# $1 nodes (container inventory, `enroot remove`, tmpfs and node-local
# checkpoint prepare/cleanup); the sweep takes longer on larger allocations.
# GRPO_NODE_SWEEP_TIMEOUT overrides the node-scaled default.
grpo_node_sweep_timeout_for() {
    echo "${GRPO_NODE_SWEEP_TIMEOUT:-$(( 90 + ${1:-1} ))}"
}

grpo_validate_node_sweep_timeout() {
    if [[ -n "${GRPO_NODE_SWEEP_TIMEOUT:-}" && ! "${GRPO_NODE_SWEEP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: GRPO_NODE_SWEEP_TIMEOUT must be a positive integer (seconds)" >&2
        return 1
    fi
}

# Seconds one experiment's teardown may take on $1 nodes with $2 whole-allocation
# sweeps: the Ray and Slurm drains (15 + 60 s), every sweep at its deadline plus
# 2 s of kill grace, and 195 s for the synchronous shared-tree deletion. Deletion
# time is unbounded: this is a reserve, not a cleanup-completion guarantee.
grpo_teardown_reserve_seconds_for() {
    echo $(( 15 + 60 + ${2} * ($(grpo_node_sweep_timeout_for "${1}") + 2) + 195 ))
}

# The same reserve in whole minutes, rounded up, for allocation sizing.
grpo_teardown_reserve_minutes_for() {
    echo $(( ($(grpo_teardown_reserve_seconds_for "${1}" "${2}") + 59) / 60 ))
}
