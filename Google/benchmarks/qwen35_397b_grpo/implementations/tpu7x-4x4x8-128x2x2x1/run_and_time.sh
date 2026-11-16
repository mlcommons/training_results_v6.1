#!/bin/bash
# MLPerf Training v6.1 / Qwen3.5-397B-A17B GRPO (qwen35_397b_grpo) / tpu7x-4x4x8-128x2x2x1
# Launch recipes: recipes/mlperf_gbs1024_submission.sh (training) and
# recipes/evals/mlperf_397b_v7x_eval.sh (evaluation).
#   - Trainer: 1x tpu7x:4x4x8 (32 hosts, 128 chips / 256 devices, FSDP=32, TP=1, EP=2, CP=4)
#   - Rollout: 128x tpu7x:2x2x1 (128 hosts, 512 chips / 1024 devices, DP=1, EP=8, TP=1, jobset.tpu.yaml)
#   - Global batch size 1,152: 72 prompts x 16 generations per step, 10 steps
# Launches ONE complete two-phase RL run (Phase 1: Online GRPO Training;
# Phase 2: Sequential Offline Checkpoint Evaluation & RCP MLLOG completion).
#
# Usage:
#   PROJECT=<gcp-project> REGION=<region> CLUSTER=<tpu7x-gke-cluster> \
#   K8S_NAMESPACE=<namespace> BUCKET=gs://<bucket> \
#   TUNIX_IMAGE=<image> PATHWAYS_SERVER_IMAGE=<image> PATHWAYS_PROXY_IMAGE=<image> \
#   OPENHANDS_SERVER_IMAGE=<image> \
#   CPU_NODEPOOL=<cpu-node-pool> DATASET_PATH=gs://<dataset> \
#   MAXTEXT_CKPT=gs://<checkpoint>/items \
#   [EVAL_TUNIX_IMAGE=<image>] [EVAL_PATHWAYS_SERVER_IMAGE=<image>] \
#   [EVAL_PATHWAYS_PROXY_IMAGE=<image>] [PREPULL_SANDBOX_IMAGES=true|false] \
#   [TPU_RESERVATION=<reservation>] [IMAGE_REWRITE_PREFIX=<registry-mirror>/] \
#   JOB_PREFIX=<name> [RUN_INDEX=0] [DRY_RUN=true] bash run_and_time.sh
#   The settings without brackets are required and have no default. Pathways
#   scratch data goes to GCS_SCRATCH_LOCATION (default ${BUCKET}/pathways-tmp).
#   Phase 2 runs with EVAL_TUNIX_IMAGE, EVAL_PATHWAYS_SERVER_IMAGE and
#   EVAL_PATHWAYS_PROXY_IMAGE (default: the training images) on the same cluster
#   unless EVAL_PROJECT, EVAL_REGION, EVAL_CLUSTER and EVAL_K8S_NAMESPACE name
#   another TPU7x cluster. Optional Phase 2 overrides: EVAL_TPU_RESERVATION,
#   EVAL_IMAGE_REWRITE_PREFIX, EVAL_SANDBOX_NODE_SELECTOR_VAL, EVAL_RECIPE,
#   EVAL_ROLLOUT_REPLICAS.
# Exit code: 0 when a checkpoint reaches the quality target, 2 when none does
# (a run that did not converge), any other non-zero value when training or an
# evaluation failed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Tunix source tree; the recipes take k8s_launcher.sh and the JAX compilation
# cache settings from it.
export TUNIX_ROOT="${TUNIX_ROOT:-${SCRIPT_DIR}/../tunix}"
RECIPE_DIR="${RECIPE_DIR:-${SCRIPT_DIR}/recipes}"

export JOB_PREFIX="${JOB_PREFIX:?set JOB_PREFIX (e.g. run-0)}"
export SEED="${SEED:-$(( $(od -An -N4 -tu4 /dev/urandom) & 0x7fffffff ))}"
export DRY_RUN="${DRY_RUN:-false}"

# --- Cluster, Storage, Images & Data (site-specific; no defaults) ---
# Phase 1 trains on a TPU7x GKE cluster. Phase 2 evaluates on the same cluster
# unless the EVAL_* variables name another TPU7x cluster. Every required setting
# is checked here so that a missing one fails before Phase 1 starts.
export PROJECT="${PROJECT:?set PROJECT to the GCP project of the TPU7x GKE cluster}"
export REGION="${REGION:?set REGION to the region of the TPU7x GKE cluster}"
export CLUSTER="${CLUSTER:?set CLUSTER to the name of the TPU7x GKE cluster}"
export K8S_NAMESPACE="${K8S_NAMESPACE:?set K8S_NAMESPACE to the Kubernetes namespace for training}"
export BUCKET="${BUCKET:?set BUCKET to the gs:// bucket for checkpoints, MLLOG and eval results}"
export GCS_SCRATCH_LOCATION="${GCS_SCRATCH_LOCATION:-${BUCKET}/pathways-tmp}"
EVAL_PROJECT="${EVAL_PROJECT:-${PROJECT}}"
EVAL_REGION="${EVAL_REGION:-${REGION}}"
EVAL_CLUSTER="${EVAL_CLUSTER:-${CLUSTER}}"
EVAL_K8S_NAMESPACE="${EVAL_K8S_NAMESPACE:-${K8S_NAMESPACE}}"
export MAXTEXT_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR:-${BUCKET}/maxtext/${JOB_PREFIX}}"
export METRIC_LOGGER_DIR="${METRIC_LOGGER_DIR:-${MAXTEXT_OUTPUT_DIR}/mllog}"
# Container image built from ../tunix/tunix/Dockerfile (provenance of the scored
# runs' images: ../tunix/SOURCE_VERSION.txt). The scored runs used the same
# container image and Pathways build for training and offline evaluation.
export TUNIX_IMAGE="${TUNIX_IMAGE:?set TUNIX_IMAGE to the container image built from ../tunix/tunix/Dockerfile}"
EVAL_TUNIX_IMAGE="${EVAL_TUNIX_IMAGE:-${TUNIX_IMAGE}}"
export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:?set PATHWAYS_SERVER_IMAGE to the Pathways server image}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:?set PATHWAYS_PROXY_IMAGE to the Pathways proxy image}"
EVAL_PATHWAYS_SERVER_IMAGE="${EVAL_PATHWAYS_SERVER_IMAGE:-${PATHWAYS_SERVER_IMAGE}}"
EVAL_PATHWAYS_PROXY_IMAGE="${EVAL_PATHWAYS_PROXY_IMAGE:-${PATHWAYS_PROXY_IMAGE}}"
export OPENHANDS_SERVER_IMAGE="${OPENHANDS_SERVER_IMAGE:?set OPENHANDS_SERVER_IMAGE to the OpenHands agent-server image}"
# GKE CPU node pool for the RL controller and the Pathways heads (the scored runs:
# c3d-standard-60 nodes; see the README).
export CPU_NODEPOOL="${CPU_NODEPOOL:?set CPU_NODEPOOL to the GKE CPU node pool for the RL controller and the Pathways heads}"
export CPU_MACHINE="${CPU_MACHINE:-c3d-standard-60}"
# Prepared dataset, with sandbox_images.txt, and converted base checkpoint
# (init_datasets.sh).
export DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to the prepared R2E-Gym dataset}"
export MAXTEXT_CKPT="${MAXTEXT_CKPT:?set MAXTEXT_CKPT to the converted base checkpoint}"
# Optional: TPU7x reservation, and a registry mirror of the per-task sandbox images.
export TPU_RESERVATION="${TPU_RESERVATION:-}"
export IMAGE_REWRITE_PREFIX="${IMAGE_REWRITE_PREFIX:-}"
# Sandbox image pre-pull before init_start (not timed). `true`: the RL
# controller pulls the 700 task images of ${DATASET_PATH}/sandbox_images.txt
# (and the agent-server image) onto every sandbox node before init_start.
# `false` (the scored runs): a node pulls a task image the first time a sandbox
# of that task starts on it.
PREPULL_SANDBOX_IMAGES="${PREPULL_SANDBOX_IMAGES:-false}"
case "${PREPULL_SANDBOX_IMAGES}" in
  true | false) ;;
  *) echo "ERROR: PREPULL_SANDBOX_IMAGES must be true or false, got '${PREPULL_SANDBOX_IMAGES}'" >&2; exit 1 ;;
esac

# Record pre-launch metadata
if [[ "${DRY_RUN}" != "true" ]]; then
  printf '%s\n' "${SEED}" | gcloud storage cp --no-user-output-enabled - "${MAXTEXT_OUTPUT_DIR}/seed.txt"
  {
    echo "workload_name: ${JOB_PREFIX}"
    echo "run_index: ${RUN_INDEX:-0}"
    echo "seed: ${SEED}"
    echo "launch_time_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "image: ${TUNIX_IMAGE}"
    echo "pathways_images: ${PATHWAYS_SERVER_IMAGE} ${PATHWAYS_PROXY_IMAGE}"
    echo "prepull_sandbox_images: ${PREPULL_SANDBOX_IMAGES}"
    echo "eval_image: ${EVAL_TUNIX_IMAGE}"
    echo "eval_pathways_images: ${EVAL_PATHWAYS_SERVER_IMAGE} ${EVAL_PATHWAYS_PROXY_IMAGE}"
    echo "cluster: ${CLUSTER}"
    echo "region: ${REGION}"
  } | gcloud storage cp --no-user-output-enabled - "${MAXTEXT_OUTPUT_DIR}/launch_info.txt"
  gcloud storage cp --no-user-output-enabled "${BASH_SOURCE[0]}" "${MAXTEXT_OUTPUT_DIR}/recipe/run_and_time.sh"
fi

# --- Phase 1: Online GRPO Training (mlperf_gbs1024_submission.sh -> mlperf_397b_1024_v7x.sh) ---
# Runs in a subshell so that the training-only exports below (for example
# NUM_GENERATIONS=16, ROLLOUT_REPLICAS=128, ROLLOUT_TPU_SLICE, ROLLOUT_FP8 and
# TRAINER_FP8) do not leak into the Phase 2 evaluation recipe, which reads them
# as `${VAR:-default}` overrides. The exports pin the scored training
# configuration: the recipes set most of these values only as `${VAR:-default}`
# defaults, which a variable inherited from the caller's environment would
# override.
(
  export RCP_LOGGING="true"
  export WEIGHT_SYNC_DISABLE_TIMEOUTS="1"
  export RAIDEN_BROADCAST_HOST_RATIO="1.0"
  export RAIDEN_BROADCAST_PIPELINE_STAGES="4"
  export FAIL_FAST="true"
  export WANDB_RESUME="allow"
  export PATHWAYS_CHECKPOINTING_IMPL="persistence"
  export CKPT_USE_OCDBT="0"
  export CHECKPOINT_SAVE_INTERVAL_STEPS="1"
  export CHECKPOINT_ABANDON_FAILED_SAVES="true"
  export CHECKPOINT_OPTIMIZER_INTERVAL_STEPS="0"
  # Logging only: result_1.txt and result_2.txt ran with these defaults
  # (false / true); result_0.txt ran with true / false.
  export TUNIX_DEBUG_INFERENCE_LOGS="${TUNIX_DEBUG_INFERENCE_LOGS:-false}"
  export DISABLE_TRAJECTORY_STORE="${DISABLE_TRAJECTORY_STORE:-true}"
  export MAX_STEPS="10"
  export PYTHONFAULTHANDLER="1"
  export ASYNC_WEIGHT_SYNC="true"
  export PIPELINE_TRAIN_MICROBATCHES="true"
  export IN_FLIGHT_WEIGHT_UPDATES="true"
  export MAX_STALENESS="2"
  export ROLLOUT_PRIORITY_SCHEDULING="true"
  export OPENHANDS_MULTI_TOOL_CALLS="true"
  export WEIGHT_SYNC_PARALLEL_H2H="1"
  export VERIFY_WEIGHTS="false"
  # Global batch size 1,152 = 72 prompts x 16 generations; learning rate
  # 1e-6 * sqrt(1152 / 256) and gradient clip 0.125 * sqrt(256 / 1152).
  export BATCH_SIZE="72"
  export NUM_GENERATIONS="16"
  export LEARNING_RATE="2.121320343559642e-06"
  export MAX_GRAD_NORM="0.05892556509887896"
  # Start-up: rollout batches start 75 s apart; the trainer compiles on a
  # synthetic micro-batch before run_start.
  export INITIAL_DISPATCH_FIRST_STAGGER_S="75"
  export INITIAL_DISPATCH_STAGGER_S="75"
  export TRAINER_WARMUP_COMPILE="true"
  export SANDBOX_WAIT_INITIAL="false"
  # Sandbox image pre-pull before init_start (see above).
  export PREPULL_SANDBOX_IMAGES
  export TRAJECTORY_GROUP_ORDER="prompt_batch"
  export USE_DEFAULT_SQL_STORE="false"
  # Dropless MoE trainer: ragged buffer factor 2.75, and a micro-batch whose
  # buffer overflows is rerun with the dropless program.
  export DROPLESS="true"
  export RAGGED_BUFFER_FACTOR="2.75"
  export RETRY_WHEN_TOKENS_DROPPED="true"
  export MOE_EXCLUDE_PADDING="true"
  export LOG_MOE_DROPS="true"
  export COMPUTE_LOGPS_CHUNK_SIZE="1024"
  export ROLLOUT_ATTENTION_DP="false"
  export TRAINER_FP8="true"
  export ROLLOUT_FP8="true"
  export ROLLOUT_REPLICAS="128"
  export ROLLOUT_TPU_SLICE="tpu7x:2x2x1"
  export VLLM_DATA_PARALLEL_SIZE="1"
  export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
  export MAX_WARMPOOL_REPLICAS="16"
  export MAX_CONCURRENCY="4096"
  export USER_CONTAINER_MEMORY="160G"
  export CPU_MEMORY="220Gi"

  echo "=== [Phase 1] Launching GRPO Training: JOB_PREFIX=${JOB_PREFIX} SEED=${SEED} ==="
  bash "${RECIPE_DIR}/mlperf_gbs1024_submission.sh" start

  if [[ "${DRY_RUN}" != "true" ]]; then
    echo "Waiting for Orchestrator JobSet ${JOB_PREFIX}-orch to complete ${MAX_STEPS} steps..."
    ORCH_EXIT=""
    while true; do
      # A failed API call (non-zero exit) is retried; an empty answer means the
      # JobSet is gone (for example torn down by the fail-fast reaper).
      if ! JOBSET="$(kubectl get jobset "${JOB_PREFIX}-orch" -n "${K8S_NAMESPACE}" \
          --ignore-not-found -o name 2>/dev/null)"; then
        sleep 15
        continue
      fi
      if [[ -z "${JOBSET}" ]]; then
        echo "ERROR: JobSet ${JOB_PREFIX}-orch no longer exists." >&2
        ORCH_EXIT="1"
        break
      fi
      ORCH_EXIT="$(kubectl get pods -n "${K8S_NAMESPACE}" \
        -l "jobset.sigs.k8s.io/jobset-name=${JOB_PREFIX}-orch" \
        -o jsonpath='{.items[0].status.containerStatuses[?(@.name=="main")].state.terminated.exitCode}' 2>/dev/null || true)"
      if [[ -n "${ORCH_EXIT}" ]]; then
        echo "Orchestrator exited with code ${ORCH_EXIT}."
        break
      fi
      JOBSET_STATE="$(kubectl get jobset "${JOB_PREFIX}-orch" -n "${K8S_NAMESPACE}" \
        -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || true)"
      if [[ " ${JOBSET_STATE} " == *" Failed "* ]]; then
        echo "ERROR: JobSet ${JOB_PREFIX}-orch failed." >&2
        ORCH_EXIT="1"
        break
      fi
      if [[ " ${JOBSET_STATE} " == *" Completed "* ]]; then
        echo "JobSet ${JOB_PREFIX}-orch completed."
        ORCH_EXIT="0"
        break
      fi
      sleep 15
    done
    bash "${RECIPE_DIR}/mlperf_gbs1024_submission.sh" stop || true
    if [[ "${ORCH_EXIT}" != "0" ]]; then
      echo "ERROR: training failed (orchestrator exit code ${ORCH_EXIT}); no offline evaluation." >&2
      exit "${ORCH_EXIT}"
    fi
    gcloud storage cp "${METRIC_LOGGER_DIR}/seed_${SEED}.out" \
      "${MAXTEXT_OUTPUT_DIR}/eval/mllog/seed_${SEED}.out.orig"
  fi
)

# --- Phase 2: Sequential Offline Checkpoint Evaluation & RCP MLLOG Completion ---
# The defaults reproduce the scored evaluations: recipes/evals/mlperf_397b_v7x_eval.sh
# on TPU7x (32 tpu7x:2x2x4 Pathways replicas, DP=2, EP=16), BF16 weights, pass@4
# over the 251 R2E-Gym validation tasks, with the same container image and
# Pathways build as training (unless EVAL_TUNIX_IMAGE, EVAL_PATHWAYS_SERVER_IMAGE
# or EVAL_PATHWAYS_PROXY_IMAGE override them).
export EVAL_RECIPE="${EVAL_RECIPE:-${RECIPE_DIR}/evals/mlperf_397b_v7x_eval.sh}"
export TUNIX_IMAGE="${EVAL_TUNIX_IMAGE}"
export PATHWAYS_SERVER_IMAGE="${EVAL_PATHWAYS_SERVER_IMAGE}"
export PATHWAYS_PROXY_IMAGE="${EVAL_PATHWAYS_PROXY_IMAGE}"
export PROJECT="${EVAL_PROJECT}"
export REGION="${EVAL_REGION}"
export CLUSTER="${EVAL_CLUSTER}"
export K8S_NAMESPACE="${EVAL_K8S_NAMESPACE}"
export TPU_RESERVATION="${EVAL_TPU_RESERVATION:-${TPU_RESERVATION}}"
export IMAGE_REWRITE_PREFIX="${EVAL_IMAGE_REWRITE_PREFIX:-${IMAGE_REWRITE_PREFIX}}"
export SANDBOX_NODE_SELECTOR_VAL="${EVAL_SANDBOX_NODE_SELECTOR_VAL:-${SANDBOX_NODE_SELECTOR_VAL:-sandbox}}"
export RCP_LOGGING="true"
export ROLLOUT_FP8="false"  # BF16 checkpoint evaluation
export TRAINER_FP8="false"
export NUM_GENERATIONS="4"  # pass@4: 4 attempts per validation task
# Attention data parallelism across each replica's 2 DP ranks, 16 sequences per
# engine (the recipe default is false, as in training).
export ROLLOUT_ATTENTION_DP="true"
# Both training and offline evaluation run every tool call of each model turn,
# as in the reference. See the README.
export OPENHANDS_MULTI_TOOL_CALLS="true"
export TUNIX_DEBUG_INFERENCE_LOGS="true"
export USE_DEFAULT_SQL_STORE="false"
# Validation sampling seed of the scored evaluations (k8s_launcher.sh default).
# The run's training seed stays in seed.txt and in the MLLOG `seed` event.
export SEED="42"
# Throughput only; the evaluated tasks and sampling do not change. The scored
# evaluations used 32 replicas (batch 128, concurrency 512) and 160G head-pod
# memory.
export ROLLOUT_REPLICAS="${EVAL_ROLLOUT_REPLICAS:-32}"
export BATCH_SIZE="$((4 * ROLLOUT_REPLICAS))"
export MAX_CONCURRENCY="$((16 * ROLLOUT_REPLICAS))"
export MAX_WARMPOOL_REPLICAS="16"
export USER_CONTAINER_MEMORY="160G"
export CHECKPOINT_MANIFEST_FILE="${METRIC_LOGGER_DIR}/eval_checkpoints.jsonl"
export EVAL_OUTPUT_DIR="${MAXTEXT_OUTPUT_DIR}/eval/eval_results/${JOB_PREFIX}"
export TARGET_ACCURACY="0.69"

echo "=== [Phase 2] Launching Sequential Offline Evaluation from ${CHECKPOINT_MANIFEST_FILE} ==="
# Exits 0 when a checkpoint reaches TARGET_ACCURACY, 2 when none does, and
# another non-zero code when an evaluation fails (recipes/mlperf_base.sh).
bash "${EVAL_RECIPE}" eval
