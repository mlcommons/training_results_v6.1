# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Settings shared by every config_MI355X_<nodes>x<gpus>x<mbs>.sh. Sourced, not executed.
# Every exported variable is passed into the training container.

# The configuration name comes from the including file's name (config_<MLPERF_CONFIG>.sh).
export MLPERF_CONFIG=$(basename "$(readlink -f "${BASH_SOURCE[1]}")" | sed 's/^config_//; s/\.sh$//')

# ---- What is run -------------------------------------------------------------------------
# The recipe (EXP) is set by the system config. conf/flux1_mlperf_4n_gbs1024.yaml is a copy of the recipe versioned in
# Primus (examples/megatron/configs/MI355X/diffusion/...mxfp6_fp4_fast_mlperf_4n_gbs1024.yaml), except that the package
# sets the run (a 128 MB head bucket, forward healing from step 7325, exp_name); scripts/check_conf_sync.sh proves they
# are otherwise identical.
export PRIMUS_PATH=/workspace/Primus
# The dataset root inside DATADIR (mounted at /data): mlperf_flux1 as scripts/prepare_data.sh writes it, or another
# directory name for the same data (DATASET, set in job.sh).
export PRIMUS_DIFFUSION_DATA_PATH=/data/${DATASET:-mlperf_flux1}
export PRIMUS_SKIP_PIP=1

# ---- MLPerf logging ----------------------------------------------------------------------
export MLLOG_SUBMISSION_ORG=AMD
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_PLATFORM=MI355X
export MLLOG_SUBMISSION_STATUS=cloud
# Numerical formats (training_rules.adoc, "Numerical formats"): the transformer linears run
# MXFP6 (OCP E2M3, one E8M0 scale per 32 elements, 32-point Hadamard on both GEMM operands),
# attention and communication bf16.
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR=mxfp6
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN=bfloat16
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM=bfloat16
# The page cache is dropped on every node before every run (run_with_docker_slurm.sh and
# run_and_time.sh); a run that cannot drop it does not start.
export MLPERF_CLEAR_CACHES=true

# ---- Process placement --------------------------------------------------------------------
# Each training rank, and the data-loader workers it starts, runs on the CPUs and allocates from the memory of the
# NUMA node its GPU is attached to (run_and_time.sh -> primus-cli direct --numa), which keeps the host side of the
# step steady. The GPU-to-node map assumes all of the node's GPUs are visible to the run (NODE_GPUS unset).
export PRIMUS_NUMA_BIND=${PRIMUS_NUMA_BIND:-1}

# ---- Multi-node network (AMD AI NIC / Pollara, RCCL over RDMA) --------------------------
# Site-specific: set the NIC list and the socket interface for your cluster.
export USING_AINIC=${USING_AINIC:-1}
export NCCL_IB_HCA=${NCCL_IB_HCA:-ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1}
# RoCE v2 GID of the AI NICs. Primus's base environment exports index 3 before its AI NIC hook runs, and the hook only
# fills in a default, so without this the ionic NICs get index 3; on hosts whose GID table has only 0 and 1 (IPv6)
# ibv_query_gid then fails and the queue pairs never connect.
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-1}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export RCCL_USE_AMD_SMI_LIB=${RCCL_USE_AMD_SMI_LIB:-1}
# GPU-direct RDMA (dmabuf registration). With it, data-loader workers must not fork after
# GPU memory is registered, so they start through forkserver (also set in the recipe).
export PRIMUS_RCCL_GDR=${PRIMUS_RCCL_GDR:-1}
export ENERGON_MP_CONTEXT=${ENERGON_MP_CONTEXT:-forkserver}
export ENERGON_MP_PICKLER=${ENERGON_MP_PICKLER:-stock}
# RCCL topology file (optional): the path ON THE HOST of an RCCL/NCCL topology XML for this
# node type, if the site provides one. run_with_docker_slurm.sh mounts it read-only into the
# container and sets NCCL_TOPO_FILE to it; without it RCCL discovers the topology itself.
export RCCL_TOPO_FILE=${RCCL_TOPO_FILE:-}

# ---- Runtime ------------------------------------------------------------------------------
export PRIMUS_TURBO_CHECK_NONFINITE=0
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE=0
# Compile caches on node-local storage (set in run_with_docker_slurm.sh's mounts).
export TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
export TRITON_CACHE_DIR=/cache/triton
export PYTHONFAULTHANDLER=1
