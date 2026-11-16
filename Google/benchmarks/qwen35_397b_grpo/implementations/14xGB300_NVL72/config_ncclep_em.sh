# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# NCCL-EP high-throughput expert-major pull/push; the EP group must fit
# within one NVLink domain. Source explicitly to opt in.
export MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-flex}
export MOE_FLEX_DISPATCHER_BACKEND=${MOE_FLEX_DISPATCHER_BACKEND:-ncclep}
export NCCL_EP_HT_EM_PULL_PUSH=${NCCL_EP_HT_EM_PULL_PUSH:-1}
export NCCL_EP_ENV_VERBOSE=${NCCL_EP_ENV_VERBOSE:-1}
