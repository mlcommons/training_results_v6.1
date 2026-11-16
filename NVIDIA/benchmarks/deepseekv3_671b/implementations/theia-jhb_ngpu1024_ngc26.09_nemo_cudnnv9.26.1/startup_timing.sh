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

startup_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

startup_epoch_ms() {
    date +%s%3N
}

startup_iso() {
    date -u +'%Y-%m-%dT%H:%M:%S.%3NZ'
}

startup_should_log() {
    startup_truthy "${STARTUP_TIMING:-0}" \
        && { [[ -z "${RANK:-}" ]] \
             || startup_truthy "${STARTUP_TIMING_ALL_RANKS:-0}" \
             || [[ "${RANK}" -eq 0 ]]; }
}

startup_mark() {
    startup_should_log || return 0
    local phase="$1"
    local event="${2:-mark}"
    local start_ms="${3:-}"
    local now_ms elapsed=""
    now_ms="$(startup_epoch_ms)"
    if [[ -n "${start_ms}" ]]; then
        elapsed=" elapsed_ms=$((now_ms - start_ms))"
    fi
    shift $(( $# < 3 ? $# : 3 ))
    echo "STARTUP_TIMING phase=${phase} event=${event} rank=${RANK:-launcher} epoch_ms=${now_ms} iso=$(startup_iso)${elapsed}${*:+ $*}"
}

startup_time_cmd() {
    local phase="$1"
    shift
    local start_ms rc restore_errexit=0
    start_ms="$(startup_epoch_ms)"
    startup_mark "${phase}" start "${start_ms}"
    [[ "$-" == *e* ]] && restore_errexit=1
    set +e
    "$@"
    rc=$?
    [[ "${restore_errexit}" -eq 0 ]] || set -e
    startup_mark "${phase}" end "${start_ms}" "status=${rc}"
    return "${rc}"
}
