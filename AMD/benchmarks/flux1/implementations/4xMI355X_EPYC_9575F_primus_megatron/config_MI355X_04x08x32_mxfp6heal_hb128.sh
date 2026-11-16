# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# The submitted system config: 4 nodes x 8 MI355X, micro batch 32 per GPU, global batch 1024.
# Recipe conf/flux1_mlperf_4n_gbs1024.yaml: MXFP6 forward GEMMs except the joint blocks' text stream (MXFP4), MXFP4
# backward GEMMs rounded to nearest, CPU-free step boundary; from step 7325 the text-stream forwards go back to MXFP6
# (forward healing, "mxfp6heal"); 128 MB head bucket ("hb128"). The file name is the config_filename in the results.
# Schedule: the GBS-1024 reference convergence point (lr 2.5e-4, 800 warmup steps,
# evaluation every 262,144 samples), set as the recipe's defaults.
source "$(dirname "${BASH_SOURCE[0]}")/config_common.sh"
export MLLOG_CONFIG_FILENAME=$(basename "$(readlink -f "${BASH_SOURCE[0]}")")

export NUM_NODES=4
export NUM_GPUS=8
export MICRO_BATCH_SIZE=32
export GLOBAL_BATCH_SIZE=1024

# Wall-clock budget per run, minutes (job.sh sizes the allocation from it).
export WALLTIME_RUNANDTIME=${WALLTIME_RUNANDTIME:-120}

export EXP=/workspace/code/conf/flux1_mlperf_4n_gbs1024.yaml
# MXFP4 in the backward GEMMs and the text-stream forward: the lowest linear-layer precision is MXFP4.
export MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR=mxfp4

# RCCL over the AI NICs: 2 queue pairs per connection, no data split, 1 MB buffers. Without the CPU sync at the
# step boundary, back-to-back collectives with the default 4 MB buffers drop packets on the fabric
# (retransmissions, slow steps); 1 MB buffers keep it clean at the same step time.
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-2}
export NCCL_IB_SPLIT_DATA_ON_QPS=${NCCL_IB_SPLIT_DATA_ON_QPS:-0}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-1048576}
