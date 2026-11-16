                                                                  
unset NCCL_NET NCCL_IB_DISABLE NCCL_CFG_PATH NCCL_MIN_NCHANNELS NCCL_MIN_CTAS NCCL_MAX_CTAS
unset NCCL_P2P_NET_CHUNKSIZE NCCL_WORK_FIFO_DEPTH NCCL_SHARP_GROUP_SIZE_THRESH TP_PP_DP_MAPPING
unset RANDOM_INIT_POLICY PROXY_NUM_LAYERS MOCK_ROLLOUT CONTAINER_IMAGE_OVERRIDE
#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
                                                                                               
export ACTIVATION_CHECKPOINTING=true
export CHECK_COMPLIANCE=1
export CLEAR_CACHES=1
export COLOCATED_GENERATION=0
export CONTAINER_LD_LIBRARY_PATH_PREPEND=/usr/lib/x86_64-linux-gnu
export CUDA_DEVICE_MAX_CONNECTIONS=32
export DGXNGPU=8
export DGXNNODES=8
export DGXSYSTEM=B300_qwen35_tp2pp4ep8_mbs1_mxfp8_expandable_mgmt_pinned_offline_eval
export EP=8
export ETP=1
export EXTRA_ARGS='++grpo.val_period=0 ++grpo.val_at_start=false ++grpo.val_at_end=false ++grpo.deferred_evaluation.enabled=false ++policy.megatron_cfg.model_overrides.batch_p2p_sync=false ++policy.megatron_cfg.model_overrides.overlap_dispatch_backward_with_experts_wgrad=true ++policy.megatron_cfg.model_overrides.overlap_moe_expert_parallel_comm=false ++policy.megatron_cfg.model_overrides.delay_wgrad_compute=false ++policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=true ++policy.megatron_cfg.distributed_data_parallel_config.align_param_gather=true ++policy.megatron_cfg.distributed_data_parallel_config.bucket_size=134217728 ++policy.megatron_cfg.moe_permute_fusion=true ++policy.megatron_cfg.apply_rope_fusion=false ++policy.megatron_cfg.model_overrides.moe_router_fusion=true ++policy.megatron_cfg.model_overrides.moe_hybridep_num_sms_preprocessing=32 ++policy.megatron_cfg.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=${oc.env:CUDA_DEVICE_MAX_CONNECTIONS} ++policy.megatron_cfg.env_vars.TORCH_NCCL_HIGH_PRIORITY=${oc.env:TORCH_NCCL_HIGH_PRIORITY} ++policy.megatron_cfg.env_vars.NVTE_NORM_FWD_USE_CUDNN=${oc.env:NVTE_NORM_FWD_USE_CUDNN} ++policy.megatron_cfg.env_vars.NVTE_NORM_BWD_USE_CUDNN=${oc.env:NVTE_NORM_BWD_USE_CUDNN} ++policy.megatron_cfg.env_vars.NVTE_FWD_LAYERNORM_SM_MARGIN=${oc.env:NVTE_FWD_LAYERNORM_SM_MARGIN} ++policy.megatron_cfg.env_vars.NVTE_BWD_LAYERNORM_SM_MARGIN=${oc.env:NVTE_BWD_LAYERNORM_SM_MARGIN} ++policy.megatron_cfg.env_vars.PRINT_MEMORY_USAGE=${oc.env:PRINT_MEMORY_USAGE} ++policy.megatron_cfg.env_vars.NCCL_DEBUG=${oc.env:NCCL_DEBUG} ++policy.megatron_cfg.env_vars.NCCL_DEBUG_SUBSYS=${oc.env:NCCL_DEBUG_SUBSYS} ++policy.megatron_cfg.env_vars.NCCL_DEBUG_FILE=${oc.env:NCCL_DEBUG_FILE} ++policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF=${oc.env:QWEN_POLICY_ALLOC_CONF} ++grpo.val_period=1 ++grpo.val_at_start=false ++grpo.val_at_end=false ++grpo.deferred_evaluation.enabled=false ++grpo.max_val_samples=null ++grpo.val_num_generations_per_prompt=4 ++grpo.stop_at_validation_threshold=0.69 ++grpo.val_period=0 ++grpo.deferred_evaluation.enabled=true ++checkpointing.enabled=true ++checkpointing.save_period=10 ++checkpointing.keep_top_k=2 ++checkpointing.save_optimizer=false ++policy.megatron_cfg.checkpoint.async_save=true'
export FINE_GRAINED_ACTIVATION_OFFLOADING=false
export FORCE_SUCCESS_STATUS=0
export GBS=512
export GEN_NODES=4
export GLOO_SOCKET_IFNAME=mgmt_eth
export GYM_EVAL_CONCURRENCY=1024
export GYM_TRAIN_CONCURRENCY=1536
export HF_CACHE_DIR="${HF_CACHE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/cache/runtime}"
export HF_CKPT_PATH="${HF_CKPT_PATH:?Set HF_CKPT_PATH}"
export HYDRA_FULL_ERROR=1
export INIT_EXPECTED_MOUNTS=0
export LEARNING_RATE=1.4142135624e-06
export MAX_GRAD_NORM=0.08838834765
export MAX_STEPS=11
export MAX_TRAJECTORY_AGE_STEPS=4
export MCORE_GDN_FUSION=1
export MEGATRON_GDN_BACKEND=transformer_engine
export MLPERF_ATTN_PRECISION=bfloat16
export MLPERF_BENCHMARK_NAME=qwen35_397b_grpo
export MLPERF_COMM_PRECISION=bfloat16
export MLPERF_LINEAR_PRECISION=fp8
export MLPERF_TARGET_ACCURACY=0.69
export MLPERF_WARMUP_TRAIN=0
export MLPERF_WARMUP_TRAIN_TIMEOUT_S=3600
export MOE_FLEX_DISPATCHER_BACKEND=hybridep
export MOE_SHARED_EXPERT_OVERLAP=false
export MOE_USE_GROUPED_TENSOR=true
export NCCL_CROSS_NIC=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=/results/nccl/%h.%p.log
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA==gpu0.0_rdma,gpu0.1_rdma,gpu0.2_rdma,gpu0.3_rdma,gpu1.0_rdma,gpu1.1_rdma,gpu1.2_rdma,gpu1.3_rdma,gpu2.0_rdma,gpu2.1_rdma,gpu2.2_rdma,gpu2.3_rdma,gpu3.0_rdma,gpu3.1_rdma,gpu3.2_rdma,gpu3.3_rdma,gpu4.0_rdma,gpu4.1_rdma,gpu4.2_rdma,gpu4.3_rdma,gpu5.0_rdma,gpu5.1_rdma,gpu5.2_rdma,gpu5.3_rdma,gpu6.0_rdma,gpu6.1_rdma,gpu6.2_rdma,gpu6.3_rdma,gpu7.0_rdma,gpu7.1_rdma,gpu7.2_rdma,gpu7.3_rdma
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_SOCKET_IFNAME=mgmt_eth
export NCCL_TEST=0
export NEMORL_TRAIN_SEGMENT_SIZE=1
export NEMO_GYM_NUM_WORKERS=4
export NEMO_GYM_REPLICAS=4
export NEMO_GYM_SWE_SIF_DIR="${NEMO_GYM_SWE_SIF_DIR:?Set NEMO_GYM_SWE_SIF_DIR}"
export NEMO_GYM_SWE_TRAIN_DATA_PATH="${NEMO_GYM_SWE_TRAIN_DATA_PATH:?Set NEMO_GYM_SWE_TRAIN_DATA_PATH}"
export NEMO_GYM_SWE_VALIDATION_DATA_PATH="${NEMO_GYM_SWE_VALIDATION_DATA_PATH:?Set NEMO_GYM_SWE_VALIDATION_DATA_PATH}"
export NEXP=1
export NRL_CONTAINER_ENV_FORWARD=DGXNGPU,DGXNNODES,TRAIN_NODES,GEN_NODES,COLOCATED_GENERATION,TP,PP,EP,ETP,RECIPE,GBS,NUM_PROMPTS_PER_STEP,NUM_GENERATIONS_PER_PROMPT,LEARNING_RATE,MAX_GRAD_NORM,MAX_STEPS,VAL_START_AT,ACTIVATION_CHECKPOINTING,RECOMPUTE_GRANULARITY,RECOMPUTE_MODULES,FINE_GRAINED_ACTIVATION_OFFLOADING,OPTIMIZER_CPU_OFFLOAD,OVERLAP_GRAD_REDUCE,MOE_SHARED_EXPERT_OVERLAP,MOE_FLEX_DISPATCHER_BACKEND,MOE_USE_GROUPED_TENSOR,NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN,USE_MNNVL,NVLINK_DOMAIN_SIZE,NEMORL_TRAIN_SEGMENT_SIZE,TRTLLM_TP,TRTLLM_MEP,TRTLLM_GEN_PRECISION,TRTLLM_GEN_IS_MX,NRL_REFIT_BUFFER_MEMORY_RATIO,NRL_USE_FASTOKENS,CONTAINER_LD_LIBRARY_PATH_PREPEND,MEGATRON_GDN_BACKEND,MCORE_GDN_FUSION,NEMO_GYM_REPLICAS,NEMO_GYM_NUM_WORKERS,GYM_TRAIN_CONCURRENCY,GYM_EVAL_CONCURRENCY,VAL_BATCH_SIZE,TRTLLM_HYBRID_DISAGG_REUSE,TLLM_MAMBA_FOLD_SAVE_LAST,TLLM_ADP_PACKED_SYNC,UCX_TLS,UCX_TCP_AF_PRIO,NCCL_NVLS_ENABLE,NCCL_IB_HCA,UCX_NET_DEVICES,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME,NCCL_IB_GID_INDEX,NCCL_CROSS_NIC,NCCL_IGNORE_CPU_AFFINITY,NVTE_UB_SOCKET_IFNAME,TE_UB_FT_COMM_ID,HYDRA_FULL_ERROR,OFFLINE_EVAL,EXTRA_ARGS,CUDA_DEVICE_MAX_CONNECTIONS,TORCH_NCCL_HIGH_PRIORITY,NVTE_NORM_FWD_USE_CUDNN,NVTE_NORM_BWD_USE_CUDNN,NVTE_FWD_LAYERNORM_SM_MARGIN,NVTE_BWD_LAYERNORM_SM_MARGIN,PRINT_MEMORY_USAGE,MLPERF_WARMUP_TRAIN_TIMEOUT_S,NCCL_DEBUG,NCCL_DEBUG_SUBSYS,NCCL_DEBUG_FILE,RAY_DEDUP_LOGS,QWEN_POLICY_ALLOC_CONF,QWEN_FULL_EVAL,QWEN_B300_NODE_ORDER,MLPERF_WARMUP_TRAIN
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:?Set NRL_MEGATRON_CHECKPOINT_DIR}"
export NRL_OFFLOAD_BEFORE_REFIT_EMPTY_CACHE=1
export NRL_OFFLOAD_BEFORE_REFIT_GC=1
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.01
export NRL_SOURCE_OVERLAY=0
export NRL_USE_FASTOKENS=1
export NUM_GENERATIONS_PER_PROMPT=16
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_PROMPTS_PER_STEP=32
export NVLINK_DOMAIN_SIZE=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVTE_CPU_OFFLOAD_V1=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_UB_SOCKET_IFNAME=mgmt_eth
export OFFLINE_EVAL=1
export OPTIMIZER_CPU_OFFLOAD=false
export OVERLAP_GRAD_REDUCE=true
export PP=4
export PRINT_MEMORY_USAGE=1
export QWEN_B300_NODE_ORDER=172.17.27.78,172.17.27.76,172.17.27.79,172.17.27.77,172.17.27.73,172.17.27.74,172.17.27.75,172.17.27.80
export QWEN_B300_RUN_MINUTES=180
export QWEN_COMPILE_CACHE_ROOT=/var/tmp/qwen35-b300-jit/qwen35_trtllm_b300_x86_64_20261012_tp2_pp2_ep8_mbs1
export QWEN_FULL_EVAL=1
export QWEN_IPC_LOGIN_GUARD=1
export QWEN_POLICY_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export RAY_LOG_SYNC_FREQUENCY=15
export RECIPE=/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm_mxfp8.yaml
export RECOMPUTE_GRANULARITY=full
export RECOMPUTE_MODULES=null
export RETAIN_GRPO_CHECKPOINTS=1
export RETAIN_GRPO_SCRATCH=1
export SEGMENT=1
export TE_UB_FT_COMM_ID=nccl
export TLLM_ADP_PACKED_SYNC=1
export TLLM_MAMBA_FOLD_SAVE_LAST=1
export TORCH_DIST_USE_NCCL2=0
export TORCH_NCCL_HIGH_PRIORITY=1
export TP=2
export TRAIN_NODES=4
export TRTLLM_GEN_IS_MX=true
export TRTLLM_GEN_PRECISION=fp8
export TRTLLM_GPU_UTIL=0.7
export TRTLLM_HYBRID_DISAGG_REUSE=1
export TRTLLM_MEP=4
export TRTLLM_TP=4
export UCX_NET_DEVICES=all
export UCX_TLS=all
export USE_MNNVL=0
export VAL_BATCH_SIZE=null
export VAL_START_AT=10
export VERIFY_MOUNTS=1
export WALLTIME=390
export WALLTIME_OFFLINE_EVAL=180
export WALLTIME_RUNANDTIME=180

                                                                   
_qwen_submission_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export QWEN_OFFICIAL_DIR="${_qwen_submission_dir}"
export NRL_RUNTIME_PATCH="${_qwen_submission_dir}/patches/ddp_performance_memory_pinned_placement.patch"
export QWEN_CPU_PREFLIGHT_LOG="${_qwen_submission_dir}/results/preflight/cpu_preflight.log"
export SUBMISSION_ORG=Cisco SUBMISSION_PLATFORM=Zeus_8xB300_SXM_288GB_multi_planar
export NRL_CONTAINER_ENV_FORWARD="${NRL_CONTAINER_ENV_FORWARD},SUBMISSION_ORG,SUBMISSION_PLATFORM"
unset _qwen_submission_dir
