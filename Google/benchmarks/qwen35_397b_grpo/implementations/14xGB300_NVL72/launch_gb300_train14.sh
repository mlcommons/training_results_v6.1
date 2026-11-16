#!/usr/bin/env bash
# NVIDIA October 13 GB300 profile: 14 nodes, 8 training + 6 generation, GBS512.
# Usage: bash launch_gb300_train14_1013.sh [/absolute/path/image.sqsh]
set -euo pipefail
umask 077
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ $# -le 1 ]] || fail 'Usage: bash launch_gb300_train14_1013.sh [image.sqsh]'
tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
site="/lustre/$USER/mlperf-rl/site.sh"
[[ -r "$site" ]] || fail "Missing $site; run prepare_gb300_1013.sh first"
source "$site"
: "${RL_ROOT:?}" "${DATADIR:?}" "${HF_CKPT_PATH:?}"
: "${NEMO_GYM_SWE_TRAIN_DATA_PATH:?}" "${NEMO_GYM_SWE_VALIDATION_DATA_PATH:?}" "${NEMO_GYM_SWE_SIF_DIR:?}"
image_source=${1:-${CONT:-$HOME/images/qwen35_397b_grpo-arm+20261013.sqsh}}
[[ -f "$image_source" && -r "$image_source" ]] || fail "Missing image: $image_source"
for cmd in python3 unsquashfs sbatch realpath; do
    command -v "$cmd" >/dev/null || fail "Missing command: $cmd"
done
image_source=$(realpath -e -- "$image_source")
[[ "$RL_ROOT" == /lustre/* ]] || fail 'RL_ROOT must be on Lustre'
for p in "$RL_ROOT" "$HF_CKPT_PATH" "$NEMO_GYM_SWE_TRAIN_DATA_PATH" "$NEMO_GYM_SWE_VALIDATION_DATA_PATH" "$NEMO_GYM_SWE_SIF_DIR" "$image_source"; do
    [[ "$p" == /* && "$p" != *[[:space:],:]* ]] || fail "Unsupported mount path: $p"
done
mkdir -p "$RL_ROOT/images" "$RL_ROOT/results/train"
export GB300_RUN_DIR
GB300_RUN_DIR=$(mktemp -d "$RL_ROOT/results/train/gb300_14n_gbs512_1013_$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
python3 "$tools_dir/validate_gb300_inputs.py" "$RL_ROOT" \
    --report "$GB300_RUN_DIR/assets_check.json"
if [[ "$image_source" == /lustre/* ]]; then
    export CONT="$image_source"
else
    image_dir=$(mktemp -d "$RL_ROOT/images/train16-1013-image.XXXXXX")
    echo "Staging image: $image_dir/$(basename -- "$image_source")"
    cp --reflink=auto -- "$image_source" "$image_dir/image.sqsh.part"
    mv -- "$image_dir/image.sqsh.part" "$image_dir/$(basename -- "$image_source")"
    export CONT="$image_dir/$(basename -- "$image_source")"
fi

# Use the launch files supplied by this exact image, without git or a rebuild.
bench_relative=workspace/optimized/qwen35_397b_grpo_trtllm/pytorch
payload="$GB300_RUN_DIR/image_payload"
unsquashfs -no-progress -no-xattrs -processors 4 -d "$payload" \
    "$CONT" "$bench_relative" > "$GB300_RUN_DIR/extract_image.log" 2>&1
export GB300_LLMDIR="$payload/$bench_relative"
profile="$GB300_LLMDIR/config_GB300_14x4_t8g6_tp4pp2ep16_2p4d16_mxfp8.sh"
[[ -r "$profile" ]] || fail "Official 14-node launch profile missing from image: $profile"
export SEED
SEED=$(python3 -c 'import secrets; print(secrets.randbelow(2**31))')
printf '%s\n' "$SEED" > "$GB300_RUN_DIR/seed.txt"

# Resolve defaults in a clean environment, then freeze them for this job. This
# prevents an earlier 64-node/no-offload run from overriding NVIDIA's profile.
# Time limits are generous caps for first conversion and evaluation, not TTT.
python3 - "$profile" "$GB300_RUN_DIR/profile.env.sh" "$GB300_RUN_DIR/profile.forward" "$SEED" <<'PY'
import os, pathlib, shlex, subprocess, sys
profile, output, forward, seed = sys.argv[1:]
env = {'PATH': os.environ['PATH'], 'GBS': '512', 'NEXP': '1',
       'SEED': seed, 'OFFLINE_EVAL': '1',
       'WALLTIME_RUNANDTIME': '360', 'WALLTIME_OFFLINE_EVAL': '180'}
raw = subprocess.check_output(
    ['bash', '-euc', 'source "$1"; env -0', '_', profile], env=env)
values = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if item)
for key in ('PATH', 'PWD', 'OLDPWD', 'SHLVL', '_'):
    values.pop(key, None)
expected = {'DGXNNODES': '14', 'DGXNGPU': '4', 'TRAIN_NODES': '8', 'GEN_NODES': '6',
            'TP': '4', 'PP': '2', 'EP': '16', 'ETP': '1',
            'CTX_ENGINES': '2', 'GEN_ENGINES': '1', 'CTX_TP': '4', 'GEN_TP': '16'}
for key, expected_value in expected.items():
    if values.get(key) != expected_value:
        raise SystemExit(f'Unexpected image profile: {key}={values.get(key)!r}; expected {expected_value}')
pathlib.Path(output).write_text(''.join(
    f'export {key}={shlex.quote(value)}\n' for key, value in sorted(values.items())))
pathlib.Path(forward).write_text(','.join(sorted(values)) + '\n')
print('OFFICIAL_GB300_PROFILE_OK train=8 gen=6 context=2xTP4 decode=1xTP16')
PY

export NRL_MEGATRON_CHECKPOINT_DIR="$RL_ROOT/mcore_ckpt_1013_gb300_tp4pp2ep16"
mkdir -p "$NRL_MEGATRON_CHECKPOINT_DIR"
for key in RL_ROOT DATADIR HF_CKPT_PATH NRL_MEGATRON_CHECKPOINT_DIR NEMO_GYM_SWE_TRAIN_DATA_PATH NEMO_GYM_SWE_VALIDATION_DATA_PATH NEMO_GYM_SWE_SIF_DIR GB300_RUN_DIR GB300_LLMDIR CONT; do
    printf 'export %s=%q\n' "$key" "${!key}"
done > "$GB300_RUN_DIR/paths.sh"
cat > "$GB300_RUN_DIR/config.sh" <<'CONFIG'
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
unset USER_DROPCACHE
source "${GB300_RUN_DIR:?}/paths.sh"
source "$GB300_RUN_DIR/profile.env.sh"
# Honor Slurm's actual CPU allocation instead of the profile's 144-CPU preset.
unset CPUS_PER_WORKER
export ENROOT_CACHE_PATH="$RL_ROOT/enroot-cache"
export NRL_MEGATRON_CHECKPOINT_RW=1
export UCX_TLS=all UCX_NET_DEVICES=all UCX_TCP_AF_PRIO=inet
export MLPERF_SUBMISSION_ORG=Google MLPERF_SUBMISSION_PLATFORM=GB300
export MLPERF_CLUSTER_NAME=xtyslurma4
export MODEL_NAME=qwen35_397b_grpo_gb300_14n
export FORCE_SUCCESS_STATUS=0 CHECK_COMPLIANCE=1
export MLPERF_WARMUP=1 NCCL_TEST=0 RUN_ONLY_NCCL=0
export CLEAR_CACHES=1 SET_PEAK_MEM_CLK=0
export DROPCACHE_CMD='sudo -n /sbin/sysctl vm.drop_caches=3'
export RETAIN_GRPO_SCRATCH=1 RETAIN_GRPO_CHECKPOINTS=1
export NRL_SOURCE_OVERLAY=0 NRL_ENABLE_GPU_CPU_AFFINITY=1
export VERIFY_MOUNTS=0 INIT_EXPECTED_MOUNTS=0 ENABLE_WANDB=0
# Work around NCCL 2.30 RAS use-after-free SIGSEGV (ras/peers.cc rasMsgHandlePeersUpdate) on GB300.
export NCCL_RAS_ENABLE=0
export LOGDIR="$GB300_RUN_DIR/results"
export CHECKPOINTS_DIR="$GB300_RUN_DIR/checkpoints"
export HF_CACHE_DIR="$GB300_RUN_DIR/hf_cache"
export RUNSUB_DIR="$GB300_LLMDIR"
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
NRL_CONTAINER_ENV_FORWARD="$(cat "$GB300_RUN_DIR/profile.forward"),NCCL_RAS_ENABLE,UCX_TLS,UCX_NET_DEVICES,UCX_TCP_AF_PRIO,MLPERF_SUBMISSION_ORG,MLPERF_SUBMISSION_PLATFORM,MLPERF_CLUSTER_NAME,FORCE_SUCCESS_STATUS,MLPERF_WARMUP,NRL_SOURCE_OVERLAY,NRL_ENABLE_GPU_CPU_AFFINITY,ENABLE_WANDB,BENCHMARK_DIR,CONTAINER_HF_CKPT_PATH,CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR,CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH,CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH,CONTAINER_NEMO_GYM_SWE_SIF_DIR"
CONFIG
source "$GB300_RUN_DIR/config.sh"
mkdir -p "$LOGDIR" "$CHECKPOINTS_DIR" "$HF_CACHE_DIR"

# Site-only Slurm fixes; retain the real NVLink ClusterUUID and model recipe.
python3 - "$GB300_LLMDIR/run.sub" "$GB300_RUN_DIR/run.gb300.sub" <<'PY'
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
echo "GB300_TOPOLOGY_RANK host=\$(hostname) rank=\$TOPO_RANK"
'''
s = s[:a] + replacement + s[b:]
subprocess.run(['bash', '-n'], input=s, text=True, check=True)
with dst.open('x') as f:
    f.write(s)
print('GB300_LAUNCHER_READY account=optional topo_rank=1..14')
PY

cat > "$GB300_RUN_DIR/job.sh" <<'JOB'
#!/usr/bin/env bash
#SBATCH --exclusive
#SBATCH --mem=0
set -euo pipefail
source "${GB300_RUN_DIR:?}/config.sh"
echo "GB300_TRAIN_START job=$SLURM_JOB_ID GBS=512 train_GPUs=32 gen_GPUs=24 seed=$SEED"
echo "PROFILE=$DGXSYSTEM chunked_offload=$CHUNKED_OPTIMIZER_STATE_OFFLOAD recompute=$RECOMPUTE_GRANULARITY"
cd "$GB300_LLMDIR"
exec bash "$GB300_RUN_DIR/run.gb300.sub"
JOB
nodes='xtyslurma4-a4xmaxnodeset1-[0,2-14]'
export SLURM_EXPORT_ENV=ALL
job=$(sbatch --parsable --export=ALL --partition=a4xmaxpartition \
    --nodes=14 --segment=14 --ntasks-per-node=1 --gpus-per-node=4 \
    --nodelist="$nodes" --exclusive --mem=0 --no-requeue --time=10:00:00 \
    --job-name=qwen397b-gb300-14 --chdir="$GB300_LLMDIR" \
    --output="$GB300_RUN_DIR/slurm-%j.out" --error="$GB300_RUN_DIR/slurm-%j.out" "$GB300_RUN_DIR/job.sh")
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response: $job"
printf '%s\n' "$GB300_RUN_DIR" > "$RL_ROOT/last_train14_gb300_dir"
printf '%s\n' "$GB300_RUN_DIR/slurm-$job.out" > "$RL_ROOT/last_train14_gb300_log"
printf '%s\n' "$job" > "$RL_ROOT/last_train14_gb300_job"
echo "Job: $job"
echo "Log: $GB300_RUN_DIR/slurm-$job.out"
echo "Results: $GB300_RUN_DIR/results"
echo "Image: $CONT"
echo "Seed: $SEED"
echo 'NVIDIA 14-node profile retained: 8 trainer nodes, 2 TP4 context engines + 1 TP16 decode engine.'
echo 'GBS512; deferred evaluation; random seed; first-conversion checkpoint mount writable.'
echo 'Time caps: training driver 360 min, evaluation 180 min, allocation 10 hours.'
