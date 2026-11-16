#!/usr/bin/env bash
# GB200 adaptation of NVIDIA's October 13 GB300 14-node profile.
# Variant: one TP8/EP8 context engine (the disagg YAML default) instead of the
# profile's two TP4/EP4 engines, plus GYM_TRAIN_CONCURRENCY=640. Same 6 generation
# nodes: 8 context GPUs + 16 decode GPUs. On GB200 the TP4 context engines hold only
# 1.03M KV tokens per rank, so prefix reuse collapses above ~512 rollouts (job 98).
# Plus CTX_MAX_BATCH=96: TRT-LLM retains save-last recurrent snapshots for only
# max_batch_size (default 64) conversations per context rank; 640 rollouts on 8
# context ranks is 80 per rank, which made prefix reuse decay in job 100.
# 8 training + 6 generation nodes, GBS512. No git, rebuild, or GPU preflight.
# Retains NVIDIA's chunked optimizer offload; uses FULL recomputation on GB200
# (selective recompute OOMed in trainer warmup on 184 GB GPUs, job 68).
# GB200 memory fit is not established by the static checks in this launcher.
# Usage: bash launch_gb200_train14_1013_ctx8_c640_mb96.sh [/absolute/path/image.sqsh]
set -euo pipefail
umask 077
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ $# -le 1 ]] || fail 'Usage: bash launch_gb200_train14_1013_ctx8_c640_mb96.sh [image.sqsh]'
site="/lustre/$USER/mlperf-rl/site.sh"
[[ -r "$site" ]] || fail "Missing existing cluster settings: $site"
source "$site"
: "${RL_ROOT:?}" "${DATADIR:?}" "${HF_CKPT_PATH:?}"
: "${NEMO_GYM_SWE_TRAIN_DATA_PATH:?}" "${NEMO_GYM_SWE_VALIDATION_DATA_PATH:?}" "${NEMO_GYM_SWE_SIF_DIR:?}"
image_source=${1:-/lustre/$USER/mlperf-rl/images/train64-1006-image.p8HANa/qwen35_397b_grpo-arm+20261013.sqsh}
[[ -f "$image_source" && -r "$image_source" ]] || fail "Missing image: $image_source"
for cmd in python3 unsquashfs sbatch realpath; do
    command -v "$cmd" >/dev/null || fail "Missing command: $cmd"
done
image_source=$(realpath -e -- "$image_source")
[[ "$RL_ROOT" == /lustre/* ]] || fail 'RL_ROOT must be on Lustre'
for p in "$RL_ROOT" "$HF_CKPT_PATH" "$NEMO_GYM_SWE_TRAIN_DATA_PATH" "$NEMO_GYM_SWE_VALIDATION_DATA_PATH" "$NEMO_GYM_SWE_SIF_DIR" "$image_source"; do
    [[ "$p" == /* && "$p" != *[[:space:],:]* ]] || fail "Unsupported mount path: $p"
done
[[ -d "$HF_CKPT_PATH" && -r "$HF_CKPT_PATH/config.json" ]] || fail "Model config missing: $HF_CKPT_PATH/config.json"
[[ -d "$NEMO_GYM_SWE_SIF_DIR" ]] || fail "SIF directory missing: $NEMO_GYM_SWE_SIF_DIR"
for p in "$NEMO_GYM_SWE_TRAIN_DATA_PATH" "$NEMO_GYM_SWE_VALIDATION_DATA_PATH"; do
    [[ -r "$p" && -s "$p" ]] || fail "Dataset missing or empty: $p"
done
mkdir -p "$RL_ROOT/images" "$RL_ROOT/results/train" "$RL_ROOT/enroot-cache"
export GB200_RUN_DIR
GB200_RUN_DIR=$(mktemp -d "$RL_ROOT/results/train/gb200_14n_gbs512_1013_ctx8_c640_mb96_$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
if [[ "$image_source" == /lustre/* ]]; then
    export CONT="$image_source"
else
    image_dir=$(mktemp -d "$RL_ROOT/images/train14-1013-image.XXXXXX")
    echo "Staging image: $image_dir/$(basename -- "$image_source")"
    cp --reflink=auto -- "$image_source" "$image_dir/image.sqsh.part"
    mv -- "$image_dir/image.sqsh.part" "$image_dir/$(basename -- "$image_source")"
    export CONT="$image_dir/$(basename -- "$image_source")"
fi

# Use the launch files supplied by this exact image, without git or a rebuild.
bench_relative=workspace/optimized/qwen35_397b_grpo_trtllm/pytorch
payload="$GB200_RUN_DIR/image_payload"
unsquashfs -no-progress -no-xattrs -processors 4 -d "$payload" \
    "$CONT" "$bench_relative" > "$GB200_RUN_DIR/extract_image.log" 2>&1
export GB200_LLMDIR="$payload/$bench_relative"
profile=''
for candidate in \
    config_GB300_14x4_t8g6_tp4pp2ep16_2p4d16_mxfp8.sh \
    config_GB300_14x4_t8g6_tp4pp2ep16_2p4d16_disagg_mxfp8.sh; do
    if [[ -r "$GB200_LLMDIR/$candidate" ]]; then
        profile="$GB200_LLMDIR/$candidate"
        break
    fi
done
[[ -n "$profile" ]] || fail "NVIDIA 14-node GB300 profile missing from supplied image; see $GB200_LLMDIR"
printf '%s\n' "$profile" > "$GB200_RUN_DIR/source_profile.txt"
printf '%s\n' "$CONT" > "$GB200_RUN_DIR/image_path.txt"
echo "Source profile (adapted for GB200): $(basename -- "$profile")"
export SEED
SEED=$(python3 -c 'import secrets; print(secrets.randbelow(2**31))')
printf '%s\n' "$SEED" > "$GB200_RUN_DIR/seed.txt"

# Resolve defaults in a clean environment, then freeze them for this job. This
# prevents an earlier 64-node/no-offload run from overriding NVIDIA's profile.
# Time limits are generous caps for first conversion and evaluation, not TTT.
python3 - "$profile" "$GB200_RUN_DIR/profile.env.sh" "$GB200_RUN_DIR/profile.forward" "$SEED" <<'PY'
import os, pathlib, shlex, subprocess, sys
profile, output, forward, seed = sys.argv[1:]
env = {'PATH': os.environ['PATH'], 'GBS': '512', 'NEXP': '1',
       'SEED': seed, 'OFFLINE_EVAL': '1', 'SEGMENT': '14',
       'WALLTIME_RUNANDTIME': '360', 'WALLTIME_OFFLINE_EVAL': '180'}
raw = subprocess.check_output(
    ['bash', '-euc', 'source "$1"; env -0', '_', profile], env=env)
values = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
for key in ('PATH', 'PWD', 'OLDPWD', 'SHLVL', '_'):
    values.pop(key, None)
expected = {'DGXNNODES': '14', 'DGXNGPU': '4', 'TRAIN_NODES': '8', 'GEN_NODES': '6',
            'TP': '4', 'PP': '2', 'EP': '16', 'ETP': '1',
            'CTX_ENGINES': '2', 'GEN_ENGINES': '1', 'CTX_TP': '4', 'GEN_TP': '16',
            'CTX_MEP': '4', 'GEN_MEP': '16', 'GBS': '512',
            'NUM_PROMPTS_PER_STEP': '32', 'NUM_GENERATIONS_PER_PROMPT': '16',
            'OPTIMIZER_CPU_OFFLOAD': 'false', 'CHUNKED_OPTIMIZER_STATE_OFFLOAD': 'true',
            'RECOMPUTE_GRANULARITY': 'selective', 'NEMORL_TRAIN_SEGMENT_SIZE': '8'}
for key, expected_value in expected.items():
    if values.get(key) != expected_value:
        raise SystemExit(f'Unexpected image profile: {key}={values.get(key)!r}; expected {expected_value}')
pathlib.Path(output).write_text(''.join(
    f'export {key}={shlex.quote(value)}\n' for key, value in sorted(values.items())))
pathlib.Path(forward).write_text(','.join(sorted(values)) + '\n')
print('GB200_PROFILE_OK source=GB300_14nodes train=8 gen=6 context=2xTP4 decode=1xTP16 chunked_offload=true')
PY

export NRL_MEGATRON_CHECKPOINT_DIR="$RL_ROOT/mcore_ckpt_1013_gb200_tp4pp2ep16"
mkdir -p "$NRL_MEGATRON_CHECKPOINT_DIR"
for key in RL_ROOT DATADIR HF_CKPT_PATH NRL_MEGATRON_CHECKPOINT_DIR NEMO_GYM_SWE_TRAIN_DATA_PATH NEMO_GYM_SWE_VALIDATION_DATA_PATH NEMO_GYM_SWE_SIF_DIR GB200_RUN_DIR GB200_LLMDIR CONT; do
    printf 'export %s=%q\n' "$key" "${!key}"
done > "$GB200_RUN_DIR/paths.sh"
cat > "$GB200_RUN_DIR/config.sh" <<'CONFIG'
# Clear inherited benchmark tuning before loading this job's frozen profile.
for _name in $(compgen -e); do
    case "$_name" in
        NRL_*|NEMO_GYM_*|NEMORL_*|TRTLLM_*|CTX_*|GEN_*|MOE_*|MLPERF_*|CONTAINER_*|GYM_*|OPTIMIZER_*|RECOMPUTE_*|OFFLOAD_*|NUM_LAYERS_*|RAY_*|TLLM_*) unset "$_name" ;;
    esac
done
unset _name RECIPE EXTRA_ARGS RANDOM_INIT_POLICY PROXY_NUM_LAYERS MOCK_ROLLOUT
unset GRPO_LOCAL_CHECKPOINT_DIR REPO_LOCATION EXTRA_MOUNTS EXTRA_ASSETS
unset CHECKPOINT_DIR CHECKPOINTS_DIR HF_CACHE_DIR CPUS_PER_WORKER
unset TP PP EP ETP GBS SEGMENT SEED SEED_BASE MAX_STEPS VAL_START_AT VAL_BATCH_SIZE
unset TRAIN_NODES COLOCATED_GENERATION MAX_TRAJECTORY_AGE_STEPS
unset ACTIVATION_CHECKPOINTING FINE_GRAINED_ACTIVATION_OFFLOADING
unset CHUNKED_OPTIMIZER_STATE_OFFLOAD OVERLAP_GRAD_REDUCE
unset NUM_PROMPTS_PER_STEP NUM_GENERATIONS_PER_PROMPT LEARNING_RATE MAX_GRAD_NORM
unset MCORE_GDN_FUSION MEGATRON_GDN_BACKEND NVTE_CPU_OFFLOAD_V1 TORCH_DIST_USE_NCCL2
unset SBATCH_SEGMENT SBATCH_SEGMENT_SIZE TMPDIR UV_CACHE_DIR
unset USER_DROPCACHE DATESTAMP LOGBASE
unset SBATCH_NODELIST SBATCH_EXCLUDE SBATCH_CONSTRAINT SBATCH_GRES SBATCH_GPUS_PER_NODE
source "${GB200_RUN_DIR:?}/paths.sh"
source "$GB200_RUN_DIR/profile.env.sh"
# Honor Slurm's actual CPU allocation instead of the profile's 144-CPU preset.
unset CPUS_PER_WORKER
# GB200 override: selective recompute OOMed (job 68); match the proven 64-node setting.
export RECOMPUTE_GRANULARITY=full RECOMPUTE_MODULES=null
# Generation layout for GB200: 1 x TP8/EP8 context engine + 1 x TP16 decode engine.
export CTX_ENGINES=1 CTX_TP=8 CTX_MEP=8
# Context max_batch_size also sizes the retained Mamba snapshot chains per rank.
export CTX_MAX_BATCH=96
# Rollout concurrency (profile default is GBS=512). Forwarded to the containers below.
export GYM_TRAIN_CONCURRENCY=640
export ENROOT_CACHE_PATH="$RL_ROOT/enroot-cache"
export NRL_MEGATRON_CHECKPOINT_RW=1
export UCX_TLS=all UCX_NET_DEVICES=all UCX_TCP_AF_PRIO=inet
export MLPERF_SUBMISSION_ORG=Google MLPERF_SUBMISSION_PLATFORM=GB200
export MLPERF_CLUSTER_NAME=simra4xslu
export MODEL_NAME=qwen35_397b_grpo_gb200_14n
export FORCE_SUCCESS_STATUS=0 CHECK_COMPLIANCE=1
export MLPERF_WARMUP=1 NCCL_TEST=0 RUN_ONLY_NCCL=0
export CLEAR_CACHES=1 SET_PEAK_MEM_CLK=0
export DROPCACHE_CMD='sudo -n /sbin/sysctl vm.drop_caches=3'
export RETAIN_GRPO_SCRATCH=1 RETAIN_GRPO_CHECKPOINTS=1
export NRL_SOURCE_OVERLAY=0 NRL_ENABLE_GPU_CPU_AFFINITY=1
export VERIFY_MOUNTS=0 INIT_EXPECTED_MOUNTS=0 ENABLE_WANDB=0
# Keep the current cluster workaround and forward it to all Ray containers.
export NCCL_RAS_ENABLE=0
export LOGDIR="$GB200_RUN_DIR/results"
export CHECKPOINTS_DIR="$GB200_RUN_DIR/checkpoints"
export HF_CACHE_DIR="$GB200_RUN_DIR/hf_cache"
export RUNSUB_DIR="$GB200_LLMDIR"
export BENCHMARK_DIR=/workspace/optimized/qwen35_397b_grpo_trtllm/pytorch
export RUN_AND_TIME_SCRIPT="$BENCHMARK_DIR/run_and_time.sh" WORK_DIR=/opt/nemo-rl
export CONTAINER_HF_CKPT_PATH="$HF_CKPT_PATH"
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR=/inputs/nemo_gym/mcore_ckpt
export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH=/inputs/nemo_gym/data/train.jsonl
export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH=/inputs/nemo_gym/data/validation.jsonl
export CONTAINER_NEMO_GYM_SWE_SIF_DIR=/inputs/nemo_gym/sif
export EXTRA_ARGS=''
# Force all selected profile variables (including disaggregated inference) over
# defaults baked into the image. run.sub drops unset entries such as CPUS_PER_WORKER.
export NRL_CONTAINER_ENV_FORWARD
NRL_CONTAINER_ENV_FORWARD="$(cat "$GB200_RUN_DIR/profile.forward"),NCCL_RAS_ENABLE,UCX_TLS,UCX_NET_DEVICES,UCX_TCP_AF_PRIO,MLPERF_SUBMISSION_ORG,MLPERF_SUBMISSION_PLATFORM,MLPERF_CLUSTER_NAME,FORCE_SUCCESS_STATUS,MLPERF_WARMUP,NRL_SOURCE_OVERLAY,NRL_ENABLE_GPU_CPU_AFFINITY,ENABLE_WANDB,BENCHMARK_DIR,CONTAINER_HF_CKPT_PATH,CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR,CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH,CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH,CONTAINER_NEMO_GYM_SWE_SIF_DIR,GYM_TRAIN_CONCURRENCY,CTX_MAX_BATCH"
CONFIG
source "$GB200_RUN_DIR/config.sh"
[[ "$CTX_ENGINES/$CTX_TP/$CTX_MEP/$GEN_ENGINES/$GEN_TP/$GEN_NODES" == 1/8/8/1/16/6 ]] \
    || fail "Unexpected generation layout: CTX $CTX_ENGINES x TP$CTX_TP EP$CTX_MEP, GEN $GEN_ENGINES x TP$GEN_TP, nodes $GEN_NODES"
_fwd=",$NRL_CONTAINER_ENV_FORWARD,"
for _v in GYM_TRAIN_CONCURRENCY CTX_ENGINES CTX_TP CTX_MEP CTX_MAX_BATCH; do
    [[ "$_fwd" == *",$_v,"* ]] || fail "$_v is not forwarded to the containers"
done
[[ "${GYM_TRAIN_CONCURRENCY:-}" == 640 ]] || fail "GYM_TRAIN_CONCURRENCY=${GYM_TRAIN_CONCURRENCY:-unset}, expected 640"
[[ "${CTX_MAX_BATCH:-}" == 96 ]] || fail "CTX_MAX_BATCH=${CTX_MAX_BATCH:-unset}, expected 96"
unset _fwd _v
mkdir -p "$LOGDIR" "$CHECKPOINTS_DIR" "$HF_CACHE_DIR"

# Site-only Slurm fixes; retain the real NVLink ClusterUUID and model recipe.
python3 - "$GB200_LLMDIR/run.sub" "$GB200_RUN_DIR/run.gb200.sub" <<'PY'
import pathlib, subprocess, sys
src, dst = map(pathlib.Path, sys.argv[1:])
s = src.read_text()
old = 'COMMON_SRUN_ARGS+=" -A $SLURM_JOB_ACCOUNT"'
assert s.count(old) == 1, 'Unexpected account clause in image run.sub'
s = s.replace(old, 'COMMON_SRUN_ARGS+="${SLURM_JOB_ACCOUNT:+ -A $SLURM_JOB_ACCOUNT}"', 1)
start, end = 'TOPO_RANK=""\n', '\nsource "${BENCHMARK_DIR}/utility/slurm_helpers.sh"'
assert s.count(start) == s.count(end) == 1, 'Unexpected topology block in image run.sub'
a, b = s.index(start), s.index(end)
assert b > a and '10000000000' in s[a:b], 'Unrecognized topology formula'
replacement = r'''# Positive, bounded rank; the real ClusterUUID defines the NVLink domain.
if [[ ! "\${RAY_NODE_RANK:-}" =~ ^([0-9]|1[0-3])$ ]]; then
    echo "INVALID_RAY_NODE_RANK: \${RAY_NODE_RANK:-unset}" >&2
    exit 1
fi
TOPO_RANK=\$(( \${RAY_NODE_RANK} + 1 ))
echo "GB200_TOPOLOGY_RANK host=\$(hostname) rank=\$TOPO_RANK"
'''
s = s[:a] + replacement + s[b:]
subprocess.run(['bash', '-n'], input=s, text=True, check=True)
with dst.open('x') as f:
    f.write(s)
print('GB200_LAUNCHER_READY account=optional topo_rank=1..14')
PY

cat > "$GB200_RUN_DIR/job.sh" <<'JOB'
#!/usr/bin/env bash
#SBATCH --exclusive
#SBATCH --mem=0
set -euo pipefail
source "${GB200_RUN_DIR:?}/config.sh"
echo "GB200_TRAIN_START job=$SLURM_JOB_ID GBS=512 train_GPUs=32 gen_GPUs=24 seed=$SEED ctx=${CTX_ENGINES}xTP${CTX_TP}/EP${CTX_MEP} gen=${GEN_ENGINES}xTP${GEN_TP} gym_train_concurrency=${GYM_TRAIN_CONCURRENCY:-unset} ctx_max_batch=${CTX_MAX_BATCH:-unset}"
echo "PROFILE=$DGXSYSTEM chunked_offload=$CHUNKED_OPTIMIZER_STATE_OFFLOAD recompute=$RECOMPUTE_GRANULARITY"
cd "$GB200_LLMDIR"
exec bash "$GB200_RUN_DIR/run.gb200.sub"
JOB
# Slurm chooses one 14-node segment; no stale hard-coded node list.
export SLURM_EXPORT_ENV=ALL
job=$(sbatch --parsable --export=ALL --partition=a4x \
    --nodes=14 --segment=14 --ntasks=14 --ntasks-per-node=1 --gpus-per-node=4 \
    --exclusive --mem=0 --no-requeue --time=10:00:00 \
    --job-name=qwen397b-gb200-14-mb96 --chdir="$GB200_LLMDIR" \
    --output="$GB200_RUN_DIR/slurm-%j.out" --error="$GB200_RUN_DIR/slurm-%j.out" "$GB200_RUN_DIR/job.sh")
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response: $job"
printf '%s\n' "$GB200_RUN_DIR" > "$RL_ROOT/last_train14_mb96_gb200_dir"
printf '%s\n' "$GB200_RUN_DIR/slurm-$job.out" > "$RL_ROOT/last_train14_mb96_gb200_log"
printf '%s\n' "$job" > "$RL_ROOT/last_train14_mb96_gb200_job"
echo "Job: $job"
echo "Log: $GB200_RUN_DIR/slurm-$job.out"
echo "Results: $GB200_RUN_DIR/results"
echo "Image: $CONT"
echo "Seed: $SEED"
echo 'GB200 trial: 8 trainer nodes; generation = 1 TP8/EP8 context engine + 1 TP16 decode engine (profile had 2 TP4 context engines).'
echo 'Rollout concurrency: GYM_TRAIN_CONCURRENCY=640 (NVIDIA 14-node default: 512); CTX_MAX_BATCH=96 (default 64).'
echo 'Optimizer: chunked state offload ON; regular optimizer CPU offload OFF. FULL recompute (GB200 override).'
echo 'GBS512; deferred evaluation; random seed; first-conversion checkpoint mount writable.'
echo 'Time caps: training driver 360 min, evaluation 180 min, allocation 10 hours.'

