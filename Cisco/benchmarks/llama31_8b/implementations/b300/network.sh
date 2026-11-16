#!/bin/bash
                                                             
 
                                                                            
                                                                        

_b300_quad_ucx=()
_b300_quad_hca=()
for _b300_gpu in {0..7}; do
    for _b300_plane in {0..3}; do
        _b300_quad_ucx+=("gpu${_b300_gpu}.${_b300_plane}_eth")
        _b300_quad_hca+=("gpu${_b300_gpu}.${_b300_plane}_rdma")
    done
done

if [[ ${#_b300_quad_ucx[@]} -ne 32 || ${#_b300_quad_hca[@]} -ne 32 ]]; then
    echo "ERROR: failed to construct the 32-device B300 quad-plane lists" >&2
    return 1 2>/dev/null || exit 1
fi

export UCX_NET_DEVICES="$(IFS=,; echo "${_b300_quad_ucx[*]}")"
export NCCL_IB_HCA="$(IFS=,; echo "${_b300_quad_hca[*]}")"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export B300_NETWORK_PLANES=4

unset _b300_quad_ucx _b300_quad_hca _b300_gpu _b300_plane
