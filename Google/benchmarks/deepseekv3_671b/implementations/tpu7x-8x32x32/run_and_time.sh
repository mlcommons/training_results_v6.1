#!/bin/bash
# MLPerf Training v6.1, DeepSeek-V3 671B, MaxText on one tpu7x-8x32x32 slice.
# Launches ONE run with the configuration used for the submitted runs; every XLA flag
# and MaxText override is identical to those runs. Only the per-run values (workload
# name, artifact dir, seed) and the environment-specific ones (project, cluster, zone,
# output dir, image, dataset and checkpoint paths, xpk namespace) are parameterized.
# WORKLOAD_IMAGE is the image built from public sources by setup.sh (Dockerfile; MaxText
# source: ../maxtext/src of this package), given as --docker-image.
#
# Usage: set PROJECT_ID, CLUSTER_NAME, ZONE, BASE_OUTPUT_DIR (gs://...), WORKLOAD_IMAGE,
#        DATASET_PATH (gs://...), CHECKPOINT_PATH (gs://...) and WORKLOAD_NAME, then run
#        bash run_and_time.sh   (SEED and NAMESPACE are optional)
set -e
set -o pipefail

# --- Environment (required inputs: set them for your project and cluster) ---
export PROJECT_ID="${PROJECT_ID:?set PROJECT_ID (GCP project of the GKE cluster)}"
export CLUSTER_NAME="${CLUSTER_NAME:?set CLUSTER_NAME (GKE cluster with the TPU slices)}"
export ZONE="${ZONE:?set ZONE (zone or region of the cluster)}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:?set BASE_OUTPUT_DIR (gs:// bucket/dir for run artifacts and logs)}"
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:?set WORKLOAD_IMAGE (image built by setup.sh)}"
export DATASET_PATH="${DATASET_PATH:?set DATASET_PATH (gs:// tokenized C4 dataset, see init_datasets.sh)}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:?set CHECKPOINT_PATH (gs:// initial checkpoint, see init_datasets.sh)}"
export WORKLOAD_NAME="${WORKLOAD_NAME:?set WORKLOAD_NAME}"
export ARTIFACT_DIR="${BASE_OUTPUT_DIR}/${WORKLOAD_NAME}"
# Seed: drawn once per run from /dev/urandom, passed as data_shuffle_seed and init_weights_seed
# (logged as the MLLOG "seed" event; MaxText writes ${ARTIFACT_DIR}/mllog_<seed>.log).
export SEED="${SEED:-$(( $(od -An -N4 -tu4 /dev/urandom) & 0x7fffffff ))}"

# XLA Flags
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_enable_offloading_copy_to_sparsecore=false \
  --xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=false \
  --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false \
  --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=500000 \
  --xla_all_gather_latency_bound_threshold_in_bytes=1000000 \
  --xla_tpu_pipeline_memory_host_transfers=false \
  --xla_tpu_enable_data_parallel_all_reduce_opt=false \
  --xla_tpu_dcn_all_reduce_combiner_threshold_bytes=134217728 \
  --xla_tpu_dvfs_p_state=7 \
  --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false \
  --xla_tpu_enable_async_ragged_all_to_all=true \
  --xla_tpu_force_async_all_to_all=true "

# MaxText Workload Overrides
MAXTEXT_ARGS="\
model_name=deepseek3-671b-lineage \
override_model_config=True \
enable_mllog=True \
data_shuffle_seed=${SEED} \
init_weights_seed=${SEED} \
first_num_dense_layers=3 \
use_lineage=True \
learning_rate=2.4e-05 \
dtype=bfloat16 \
mu_dtype=bfloat16 \
grad_dtype=bfloat16 \
attention=flash \
attention_type=mla \
dataset_type=c4_mlperf \
dataset_path=${DATASET_PATH} \
dataset_name=c4/en:3.0.5 \
eval_dataset_name=c4/en:3.0.9 \
train_data_columns='[ids]' \
eval_data_columns='[ids]' \
tokenize_train_data=False \
tokenize_eval_data=False \
enable_checkpointing=True \
load_parameters_path=${CHECKPOINT_PATH} \
target_eval_loss=3.6 \
lr_schedule_type=cosine \
learning_rate_schedule_steps=12000 \
warmup_steps_fraction=0.00034 \
learning_rate_final_fraction=0.000417 \
eval_start_step=42 \
eval_interval=1 \
eval_steps=1 \
save_checkpoint_on_start=False \
save_checkpoint_on_completion=False \
abort_on_nan_loss=False \
abort_on_inf_loss=False \
profile_periodically_period=10000 \
tokenizer_path=src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken \
tokenizer_type=tiktoken \
async_checkpointing=False \
per_device_batch_size=1 \
max_target_length=4096 \
skip_jax_distributed_system=False \
jax_distributed_initialization_timeout=900 \
block_state_before_run_start=True \
warm_input_reshard_before_run_start=True \
warmup_programs_in_init=True \
train_shard_in_read=True \
train_interleave_cycle_length=4 \
use_iota_embed=False \
opt_type=adamw \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_kv_compute=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
ici_parallelism='[1,8,32,32,2]' \
dcn_parallelism='[1,1,1,1,1]' \
n_routing_groups=8 \
topk_routing_group=4 \
eval_per_device_batch_size=0.25 \
steps=100 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"



echo "=== Creating XPK Workload: $WORKLOAD_NAME ==="
xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=high \
  --max-restarts=3 \
  --device-type=tpu7x-8x32x32 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  ${NAMESPACE:+--namespace=${NAMESPACE}} --restart-on-exit-codes=1 \
   \
  --workload="${WORKLOAD_NAME}" \
   \
  --command="set -e && set -o pipefail && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export ARTIFACT_DIR='${ARTIFACT_DIR}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
export PYTHONPATH='/app/src' && \
export M_BASE_NUM_DECODER_LAYERS='61' && \
export M_COMPILE_XLA_FLAGS='' && \
export JAX_COMPILATION_CACHE_DIR='${BASE_OUTPUT_DIR}/JAX_XLA_CACHE/maxtext/deepseek_v3' && \
ulimit -c 0 && \
set +e; \
python3 -u -m maxtext.trainers.pre_train.train maxtext/configs/base.yml ${MAXTEXT_ARGS} | tee train.log; \
TRAIN_EXIT_CODE=\${PIPESTATUS[0]}; \
if [ -s train.log ]; then \
  timeout 30s gcloud storage cp --no-user-output-enabled train.log \${ARTIFACT_DIR}/logs/train-\${TPU_WORKER_ID:-0}.log || true; \
fi; \
exit \${TRAIN_EXIT_CODE}"
