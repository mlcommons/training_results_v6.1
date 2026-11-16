# HiPerGator flags

# Read by mlperf_logging_utils.py for submission_org and submission_platform
# (platform is logged as <nodes>x<value>)
export MLPERF_SUBMISSION_ORG="University-of-Florida"
export MLPERF_SUBMISSION_PLATFORM="hpg.dgxb200x8"
# MPI does not start inside apptainer here (flux job 44936447), so the MPI NCCL tests are off
export NCCL_TEST=${NCCL_TEST:-0}
# expected-mounts.csv lists the parquets, which our data dir does not have
export VERIFY_MOUNTS=${VERIFY_MOUNTS:-0}

# Pin NCCL to the compute HCAs and the bridge interface (HCA order fix from v5.1)
export NCCL_IB_HCA=mlx5_15,mlx5_10,mlx5_14,mlx5_13,mlx5_8,mlx5_7,mlx5_9,mlx5_4
export NCCL_SOCKET_IFNAME=bridge-1145
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib

# Triton kernel cache on the per-job, node-local /tmp. The default ~/.triton/cache is on
# NFS: in the 8-node smoke (job 44997221) a rank read a cache file while another node
# wrote it ("OSError: [Errno 116] Stale file handle") and the run failed.
export TRITON_CACHE_DIR=/tmp/triton_cache
