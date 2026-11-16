#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${script_dir}/select_rdma_ports.sh"
node=$(hostname -s)
dlrm_select_rdma_ports "$node"
                                                                               
IFS=, read -r -a selected <<< "${NCCL_IB_HCA#=}"
for entry in "${selected[@]}"; do
    dev=${entry%%:*}
    path=/sys/class/infiniband/${dev}/ports/1
    read -r state < "${path}/state"
    read -r gid < "${path}/gids/3"
    read -r gid_type < "${path}/gid_attrs/types/3"
    [[ "$state" == '4: ACTIVE' && "$gid_type" == 'RoCE v2' &&
       "$gid" != 0000:0000:0000:0000:0000:0000:0000:0000 ]] || {
        printf 'Unhealthy permitted port: %s %s %s %s\n' "$node" "$dev" "$state" "$gid" >&2
        exit 1
    }
done
if [[ "${LOCAL_RANK:-0}" == 0 ]]; then
    printf 'DLRM_RDMA_MASK host=%s count=%s NCCL_IB_HCA=%s\n' "$node" "$DLRM_SELECTED_HCA_COUNT" "$NCCL_IB_HCA"
fi
exec "$@"
