#!/usr/bin/env bash

dlrm_select_rdma_ports() {
    local node=${1:?Expected physical hostname} gpu plane
    local mode=${DLRM_HCA_MODE:-full32}
    local -a selected=()
    [[ "$node" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || return 1
    case "$mode" in full32|exclude2|exclude8) ;; *) echo 'Unknown DLRM_HCA_MODE' >&2; return 1;; esac
    for gpu in 0 1 2 3 4 5 6 7; do
        for plane in 0 1 2 3; do
            if [[ "$mode" == exclude2 && "$plane" == 1 && ( "$gpu" == 1 || "$gpu" == 5 ) ]]; then continue; fi
            if [[ "$mode" == exclude8 && ( "$gpu" == 1 || "$gpu" == 5 ) ]]; then continue; fi
            selected+=("gpu${gpu}.${plane}_rdma:1:${plane}:${plane}")
        done
    done
    export NCCL_IB_HCA="=$(IFS=,; echo "${selected[*]}")"
    export DLRM_SELECTED_HCA_COUNT=${#selected[@]}
}
