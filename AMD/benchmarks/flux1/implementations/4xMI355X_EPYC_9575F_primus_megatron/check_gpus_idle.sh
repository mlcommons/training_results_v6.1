#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Fail if any GPU on this node already has memory in use (as in AMD's MLPerf v6.0 entry).
set -uo pipefail
smi_output="$(rocm-smi --showmemuse --csv 2>/dev/null)"
busy=$(printf '%s\n' "$smi_output" | awk -F, '$1 ~ /^card/ && $2+0 > 0')
if [[ -z "$busy" ]]; then
    echo "[$(hostname)] check_gpus_idle: OK"
    exit 0
fi
echo "[$(hostname)] GPU IDLE CHECK FAILED" >&2
printf '%s\n' "$smi_output" >&2
exit 1
