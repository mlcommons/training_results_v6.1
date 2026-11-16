#!/usr/bin/env bash
# GB200 formal MLPerf Qwen3.5-397B training, using the existing October 6 image.
# Usage: bash launch_gb200_train64_1006_no_opt_offload.sh [/absolute/path/new-image.sqsh]
# Stages the image and its launch files on Lustre, then submits 64 nodes.
set -euo pipefail
umask 077
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ $# -le 1 ]] || fail 'Usage: bash launch_gb200_train64_1006_no_opt_offload.sh [image.sqsh]'
image_source=${1:-$HOME/images/qwen35_397b_grpo-arm+20261006.sqsh}
[[ -f "$image_source" && -r "$image_source" ]] || fail "Missing image: $image_source"
image_source=$(realpath -e -- "$image_source")
site="/lustre/$USER/mlperf-rl/site.sh"
[[ -r "$site" ]] || fail "Missing $site"
source "$site"
: "${RL_ROOT:?}" "${DATADIR:?}" "${HF_CKPT_PATH:?}"
: "${NEMO_GYM_SWE_TRAIN_DATA_PATH:?}" "${NEMO_GYM_SWE_VALIDATION_DATA_PATH:?}" "${NEMO_GYM_SWE_SIF_DIR:?}"
for cmd in python3 unsquashfs sbatch realpath; do
    command -v "$cmd" >/dev/null || fail "Missing command: $cmd"
done
[[ "$RL_ROOT" == /lustre/* ]] || fail 'RL_ROOT must be on shared Lustre'
for p in "$RL_ROOT" "$HF_CKPT_PATH" "$NEMO_GYM_SWE_TRAIN_DATA_PATH" "$NEMO_GYM_SWE_VALIDATION_DATA_PATH" "$NEMO_GYM_SWE_SIF_DIR" "$image_source"; do
    [[ "$p" == /* && "$p" != *[[:space:],:]* ]] || fail "Unsupported mount path: $p"
done
mkdir -p "$RL_ROOT/images" "$RL_ROOT/results/train"
export GB200_RUN_DIR
GB200_RUN_DIR=$(mktemp -d "$RL_ROOT/results/train/gb200_64n_gbs512_1006_no_opt_offload_$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")

# Keep the new dependency stack's converted model separate from the old eval cache.
export NRL_MEGATRON_CHECKPOINT_DIR="$RL_ROOT/mcore_ckpt_1006_gb200_tp4pp2ep32"
mkdir -p "$NRL_MEGATRON_CHECKPOINT_DIR"
if [[ "$image_source" == /lustre/* ]]; then
    export CONT="$image_source"
else
    image_dir=$(mktemp -d "$RL_ROOT/images/train64-1006-image.XXXXXX")
    echo "Staging image: $image_dir/$(basename -- "$image_source")"
    cp --reflink=auto -- "$image_source" "$image_dir/image.sqsh.part"
    mv -- "$image_dir/image.sqsh.part" "$image_dir/$(basename -- "$image_source")"
    export CONT="$image_dir/$(basename -- "$image_source")"
fi

# Read the Slurm launch files baked into this image.
bench_relative=workspace/optimized/qwen35_397b_grpo_trtllm/pytorch
payload="$GB200_RUN_DIR/image_payload"
unsquashfs -no-progress -no-xattrs -processors 4 -d "$payload" \
    "$CONT" "$bench_relative" > "$GB200_RUN_DIR/extract_image.log" 2>&1
export GB200_LLMDIR="$payload/$bench_relative"

# Draw once from the OS RNG; freeze the seed so sourcing config twice cannot change it.
# run_and_time.sh passes it to grpo.seed; the benchmark logs it through mllog.
export SEED
SEED=$(python3 -c 'import secrets; print(secrets.randbelow(2**31))')
printf '%s\n' "$SEED" > "$GB200_RUN_DIR/seed.txt"

# Freeze public paths and settings, without saving the caller's entire environment.
for key in RL_ROOT DATADIR HF_CKPT_PATH NRL_MEGATRON_CHECKPOINT_DIR NEMO_GYM_SWE_TRAIN_DATA_PATH NEMO_GYM_SWE_VALIDATION_DATA_PATH NEMO_GYM_SWE_SIF_DIR GB200_RUN_DIR GB200_LLMDIR CONT SEED; do
    printf 'export %s=%q\n' "$key" "${!key}"
done > "$GB200_RUN_DIR/config.sh"
cat >> "$GB200_RUN_DIR/config.sh" <<'CONFIG'
unset RECIPE EXTRA_ARGS SBATCH_SEGMENT SBATCH_SEGMENT_SIZE
unset RANDOM_INIT_POLICY PROXY_NUM_LAYERS MOCK_ROLLOUT GRPO_LOCAL_CHECKPOINT_DIR
unset CONTAINER_IMAGE_OVERRIDE EXTRA_MOUNTS NRL_RUNTIME_PATCH NRL_RUNTIME_PATCH_CONTAINER
unset NRL_GPU_CPU_AFFINITY_FILE CHECKPOINT_DIR CHECKPOINTS_DIR HF_CACHE_DIR
unset GYM_TRAIN_CONCURRENCY NRL_SOURCE_OVERLAY REPO_LOCATION
unset NUM_LAYERS_IN_FIRST_PIPELINE_STAGE NUM_LAYERS_IN_LAST_PIPELINE_STAGE
unset RAY_ADDRESS RAY_TMPDIR TMPDIR
export NEXP=1 GBS=512 OFFLINE_EVAL=1 FORCE_SUCCESS_STATUS=0
export WALLTIME_RUNANDTIME=360 WALLTIME_OFFLINE_EVAL=180
export NEMO_GYM_REPLICAS=4 NEMO_GYM_NUM_WORKERS=4 MAX_TRAJECTORY_AGE_STEPS=4
export TP=4 PP=2 EP=32 ETP=1 NEMORL_TRAIN_SEGMENT_SIZE=8 SEGMENT=16
export MCORE_GDN_FUSION=1 MEGATRON_GDN_BACKEND=transformer_engine
export MOE_TOKEN_DISPATCHER_TYPE=flex MOE_FLEX_DISPATCHER_BACKEND=hybridep MOE_USE_GROUPED_TENSOR=true
export NRL_SOURCE_OVERLAY=0 MLPERF_TARGET_ACCURACY=0.69
# Use the image's common/batch defaults without depending on a hardware-profile filename.
source "$GB200_LLMDIR/config_common.sh"
source "$GB200_LLMDIR/config_GBS512.sh"
export RECIPE=/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch/conf/grpo_qwen35_397b_swe_openhands_async_trtllm_mxfp8.yaml
export TRAIN_NODES=32 GEN_NODES=32 DGXNNODES=64 DGXNGPU=4 COLOCATED_GENERATION=0
export MLPERF_LINEAR_PRECISION=fp8 NRL_USE_FASTOKENS=1
export WALLTIME=$((21 + NEXP * (WALLTIME_RUNANDTIME + 5)))
add_grpo_eval_walltime
export NUM_LAYERS_IN_FIRST_PIPELINE_STAGE=null NUM_LAYERS_IN_LAST_PIPELINE_STAGE=null

# NVIDIA 64-node recipe defaults; both optimizer offload modes disabled.
export ACTIVATION_CHECKPOINTING=true RECOMPUTE_GRANULARITY=full RECOMPUTE_MODULES=null
export FINE_GRAINED_ACTIVATION_OFFLOADING=false OFFLOAD_MODULES=null
export OPTIMIZER_CPU_OFFLOAD=false OPTIMIZER_OFFLOAD_FRACTION=0.0
export CHUNKED_OPTIMIZER_STATE_OFFLOAD=false OPTIMIZER_STATE_OFFLOAD_FRACTION=1.0
export OPTIMIZER_STATE_OFFLOAD_CHUNK_SIZE_MB=256 MOE_SHARED_EXPERT_OVERLAP=true
# Explicit 64-node allocation: 32 training nodes plus 32 generation nodes.
# NVIDIA baseline: 32 independent TP4/EP4 engines, 128 generation GPUs.
export TRTLLM_TP=4 TRTLLM_MEP=4 TRTLLM_MAX_BATCH=256
export TRTLLM_GPU_UTIL=0.7 TRTLLM_MOE_BACKEND=CUTEDSL
unset CTX_GPU_UTIL CTX_MAX_BATCH GEN_MAX_BATCH
unset CTX_ENGINES GEN_ENGINES CTX_TP CTX_MEP CTX_MTP GEN_TP GEN_MEP GEN_MTP
export TRTLLM_GEN_PRECISION=fp8 TRTLLM_GEN_IS_MX=true
export VAL_BATCH_SIZE=null GYM_EVAL_CONCURRENCY=1024
export GYM_TRAIN_CONCURRENCY=1536
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.01 OVERLAP_GRAD_REDUCE=false
# Respect the actual Slurm CPU allocation instead of GB300's 144-CPU preset.
unset CPUS_PER_WORKER
export UCX_TLS=all UCX_NET_DEVICES=all UCX_TCP_AF_PRIO=inet
export MLPERF_WARMUP=1 NCCL_TEST=0 RUN_ONLY_NCCL=0 CHECK_COMPLIANCE=1
export CLEAR_CACHES=1 SET_PEAK_MEM_CLK=0 DROPCACHE_CMD='sudo -n /sbin/sysctl vm.drop_caches=3'
export RETAIN_GRPO_SCRATCH=1 RETAIN_GRPO_CHECKPOINTS=1
export NRL_ENABLE_GPU_CPU_AFFINITY=1 VERIFY_MOUNTS=0 INIT_EXPECTED_MOUNTS=0
export ENABLE_WANDB=0
unset SEED_BASE
export LOGDIR="$GB200_RUN_DIR/results"
export CHECKPOINTS_DIR="$GB200_RUN_DIR/checkpoints"
export HF_CACHE_DIR="$GB200_RUN_DIR/hf_cache"
export MODEL_NAME=qwen35_397b_grpo_gb200_64n
export DGXSYSTEM=GB200_64x4_t32g32_tp4pp2ep32_gtp4_mxfp8
export MLPERF_CLUSTER_NAME=simra4xslu
# mlperf-common adds the node-count prefix, producing submission_platform=64xGB200.
export MLPERF_SUBMISSION_ORG=Google MLPERF_SUBMISSION_PLATFORM=GB200
export RUNSUB_DIR="$GB200_LLMDIR"
export BENCHMARK_DIR=/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch
export RUN_AND_TIME_SCRIPT="$BENCHMARK_DIR/run_and_time.sh" WORK_DIR=/opt/nemo-rl
export CONTAINER_HF_CKPT_PATH="$HF_CKPT_PATH"
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR=/inputs/nemo_gym/mcore_ckpt
export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH=/inputs/nemo_gym/data/train.jsonl
export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH=/inputs/nemo_gym/data/validation.jsonl
export CONTAINER_NEMO_GYM_SWE_SIF_DIR=/inputs/nemo_gym/sif
export EXTRA_ARGS=''
# Make these host values win over defaults baked into the container.
export NRL_CONTAINER_ENV_FORWARD=MLPERF_SUBMISSION_ORG,MLPERF_SUBMISSION_PLATFORM,UCX_TLS,UCX_NET_DEVICES,UCX_TCP_AF_PRIO,TRTLLM_HYBRID_DISAGG_REUSE,TLLM_MAMBA_FOLD_SAVE_LAST,TLLM_ADP_PACKED_SYNC,NRL_OFFLOAD_BEFORE_REFIT_GC,NRL_OFFLOAD_BEFORE_REFIT_EMPTY_CACHE,NRL_REFIT_BUFFER_MEMORY_RATIO,NRL_USE_FASTOKENS,ACTIVATION_CHECKPOINTING,RECOMPUTE_GRANULARITY,RECOMPUTE_MODULES,FINE_GRAINED_ACTIVATION_OFFLOADING,OFFLOAD_MODULES,OPTIMIZER_CPU_OFFLOAD,OPTIMIZER_OFFLOAD_FRACTION,CHUNKED_OPTIMIZER_STATE_OFFLOAD,OPTIMIZER_STATE_OFFLOAD_FRACTION,OPTIMIZER_STATE_OFFLOAD_CHUNK_SIZE_MB,MOE_SHARED_EXPERT_OVERLAP,MOE_TOKEN_DISPATCHER_TYPE,MOE_FLEX_DISPATCHER_BACKEND,MOE_USE_GROUPED_TENSOR,MEGATRON_GDN_BACKEND,MCORE_GDN_FUSION,NVTE_CPU_OFFLOAD_V1,TRTLLM_GPU_UTIL,TRTLLM_TP,TRTLLM_MEP,TRTLLM_MOE_BACKEND,TRTLLM_MAX_BATCH,TRTLLM_GEN_PRECISION,TRTLLM_GEN_IS_MX,VAL_BATCH_SIZE,GYM_EVAL_CONCURRENCY,OFFLINE_EVAL,MLPERF_WARMUP,MAX_STEPS,VAL_START_AT,SEED,TP,PP,EP,ETP,NUM_LAYERS_IN_FIRST_PIPELINE_STAGE,NUM_LAYERS_IN_LAST_PIPELINE_STAGE,NUM_PROMPTS_PER_STEP,NUM_GENERATIONS_PER_PROMPT,LEARNING_RATE,MAX_GRAD_NORM,MAX_TRAJECTORY_AGE_STEPS,OVERLAP_GRAD_REDUCE,NEMORL_TRAIN_SEGMENT_SIZE,NEMO_GYM_REPLICAS,NEMO_GYM_NUM_WORKERS,GYM_TRAIN_CONCURRENCY,BENCHMARK_DIR,RECIPE,ENABLE_WANDB,CONTAINER_HF_CKPT_PATH,CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR,CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH,CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH,CONTAINER_NEMO_GYM_SWE_SIF_DIR
CONFIG
source "$GB200_RUN_DIR/config.sh"
mkdir -p "$LOGDIR" "$CHECKPOINTS_DIR" "$HF_CACHE_DIR"

# Patch only the per-run Slurm launcher; the supplied image stays unchanged.
python3 - "$GB200_LLMDIR/run.sub" "$GB200_RUN_DIR/run.gb200.sub" <<'PY'
import pathlib, subprocess, sys
src, dst = map(pathlib.Path, sys.argv[1:])
s = src.read_text()
old = 'COMMON_SRUN_ARGS+=" -A $SLURM_JOB_ACCOUNT"'
assert s.count(old) == 1, 'Unexpected account clause'
s = s.replace(old, 'COMMON_SRUN_ARGS+="${SLURM_JOB_ACCOUNT:+ -A $SLURM_JOB_ACCOUNT}"', 1)
start, end = 'TOPO_RANK=""\n', '\nsource "${BENCHMARK_DIR}/utility/slurm_helpers.sh"'
assert s.count(start) == s.count(end) == 1, 'Unexpected topology block'
a, b = s.index(start), s.index(end)
assert b > a and '10000000000' in s[a:b], 'Unrecognized topology formula'
replacement = r'''# Positive, bounded ordering; real ClusterUUID still defines each NVLink domain.
if [[ ! "\${RAY_NODE_RANK:-}" =~ ^([0-9]|[1-5][0-9]|6[0-3])$ ]]; then
    echo "INVALID_RAY_NODE_RANK: \${RAY_NODE_RANK:-unset}" >&2
    exit 1
fi
TOPO_RANK=\$(( \${RAY_NODE_RANK} + 1 ))
echo "GB200_TOPOLOGY_RANK host=\$(hostname) rank=\$TOPO_RANK"
'''
s = s[:a] + replacement + s[b:]
subprocess.run(['bash', '-n'], input=s, text=True, check=True)
with dst.open('x') as f: f.write(s)
print('GB200_LAUNCHER_READY account=optional topo_rank=1..64')
PY

cat > "$GB200_RUN_DIR/job.sh" <<'JOB'
#!/usr/bin/env bash
#SBATCH --exclusive
#SBATCH --mem=0
set -euo pipefail
source "${GB200_RUN_DIR:?}/config.sh"
echo "GB200_TRAIN_START job=$SLURM_JOB_ID GBS=512 train_GPUs=128 gen_GPUs=128"
cd "$GB200_LLMDIR"
exec bash "$GB200_RUN_DIR/run.gb200.sub"
JOB
nodes='simra4xslu-a4xnodeset-[0-9,11-16],simra4xslu-a4xnodeset2-[0-6,8-9,11-17],simra4xslu-a4xnodeset3-[0-8,10-16],simra4xslu-a4xnodeset4-[0-15]'
export SLURM_EXPORT_ENV=ALL
job=$(sbatch --parsable --export=ALL --partition=a4x --nodes=64 --segment=16 --ntasks-per-node=1 --gres=gpu:4 \
    --nodelist="$nodes" --exclusive --mem=0 --no-requeue --time=10:00:00 \
    --job-name=qwen397b-gb200-64-noopt --chdir="$GB200_LLMDIR" \
    --output="$GB200_RUN_DIR/slurm-%j.out" --error="$GB200_RUN_DIR/slurm-%j.out" "$GB200_RUN_DIR/job.sh")
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response: $job"
printf '%s\n' "$GB200_RUN_DIR" > "$RL_ROOT/last_train64_dir"
printf '%s\n' "$GB200_RUN_DIR/slurm-$job.out" > "$RL_ROOT/last_train64_log"
printf '%s\n' "$job" > "$RL_ROOT/last_train64_job"
echo "Job: $job"
echo "Log: $GB200_RUN_DIR/slurm-$job.out"
echo "Results: $GB200_RUN_DIR/results"
echo "Seed: $SEED"
echo "MLPerf metadata: org=$MLPERF_SUBMISSION_ORG platform=${DGXNNODES}x${MLPERF_SUBMISSION_PLATFORM}"
echo 'Official deferred schedule: train through step 11; evaluate checkpoints 10 and 11 as needed.'
echo 'Allocation limit 10 hours (a cap, not an expected duration). TensorBoard/MLPerf logs enabled.'

