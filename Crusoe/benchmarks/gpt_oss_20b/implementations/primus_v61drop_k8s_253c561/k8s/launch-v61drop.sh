#!/bin/bash
# In-pod launcher for AMD's MLPerf v6.1 partner-drop recipes (llama31_8b, gpt_oss_20b) on N MI355X
# nodes (mounted from ConfigMap v61drop-launch). Env from the Job: BENCH NN GBS MBS SEED RUN_NAME JOB IDX.
#
# Base = AMD's in-image single-node config (the drop ships only 1x8x1). On top of it this sets the
# multi-node rendezvous and the batch-size-dependent hyperparameters. LR / warmup are free in the
# 6.1.0 closed rules for both benchmarks; we use AMD's recipe at GBS 32 and the RCP-table row
# (mlperf_logging/rcp_checker/training_6.1.0/rcps_<bench>.json) at every other GBS. Fixed by the
# checker and honoured here: eval_samples == 1024, decay_steps == 1_200_000 - warmup_steps.
set -uo pipefail
ulimit -n 1048576 2>/dev/null || ulimit -n 262144 2>/dev/null || true
: "${BENCH:?}" "${NN:?}" "${GBS:?}" "${MBS:?}" "${SEED:?}" "${RUN_NAME:?}" "${JOB:?}" "${IDX:?}"
LOGD=/joblogs/${RUN_NAME}; mkdir -p "$LOGD"
exec > >(tee "$LOGD/rank${IDX}.log") 2>&1
say(){ echo "[launch $(date -u +%H:%M:%SZ) node${IDX}] $*"; }
die(){ say "!! $*"; sleep 3; exit 2; }

cd /workspace/code
case "$BENCH" in
  llama31_8b)  source ./config_MI355X_1x8x1.sh; PORT=29502 ;;
  gpt_oss_20b) source ./config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh; PORT=29501 ;;
  *) die "unknown BENCH=$BENCH" ;;
esac

# Multi-node RCCL budget: the ionic NIC allows 4096 QPs. AMD's 32 channels per network peer x 4 QPs per
# connection (rccl-roce-env) exhausts it at 8 nodes (ibv_create_qp: No space left, or a silent hang).
# Safe defaults for multi-node; EXTRA_EXPORTS below can override them for A/B tests.
if [ "$NN" -gt 1 ]; then export NCCL_IB_QPS_PER_CONNECTION=1 NCCL_NCHANNELS_PER_NET_PEER=8; fi

# Experiment knobs: EXTRA_EXPORTS="K=V -K2 ..." applied on top of AMD's config (-K unsets K).
for kv in ${EXTRA_EXPORTS:-}; do case "$kv" in -*) unset "${kv#-}" ;; *) export "$kv" ;; esac; done
[ -n "${EXTRA_EXPORTS:-}" ] && say "EXTRA_EXPORTS: $EXTRA_EXPORTS"

# Parallel layout: TP (tensor, within a node), CP (context: splits each 8192-token sequence), EP (experts,
# gpt_oss only; carved out of DP). DP = GPUs / (TP*CP). GBS = DP * MBS * grad_accum.
TP=${TP:-1} CP=${CP:-1} EP=${EP:-1}
[ $(( 8 * NN % (TP * CP) )) -eq 0 ] || die "TP*CP=$((TP*CP)) does not divide $((8*NN)) GPUs"
[ "$TP" -le 8 ] || die "TP must stay within a node"
DP=$(( 8 * NN / (TP * CP) ))
[ $(( GBS % (DP * MBS) )) -eq 0 ] || die "GBS=$GBS not a multiple of DP*MBS=$((DP*MBS)) (NN=$NN TP=$TP CP=$CP)"
case "$BENCH:$GBS" in
  llama31_8b:32)   LR=$PRIMUS_LR; WARM=$PRIMUS_LR_WARMUP_ITERS ;;   # as shipped in the image's AMD config (7512b71/253c561: 8e-4, 64)
  llama31_8b:64)   LR=8e-4;   WARM=$((6144/64)) ;;   # RCP ref_64
  llama31_8b:96)   LR=1e-3;   WARM=$((16348/96)) ;;  # RCP ref_96
  llama31_8b:128)  LR=2e-3;   WARM=$((32768/128)) ;; # RCP ref_128
  gpt_oss_20b:32)  LR=$PRIMUS_LR; WARM=$PRIMUS_LR_WARMUP_ITERS ;;   # as shipped (7512b71: 5.7e-4, 117; 253c561: 4.5e-4, 234)
  gpt_oss_20b:16)  LR=4e-4;   WARM=$((2048/16)) ;;   # RCP ref_16
  gpt_oss_20b:64)  LR=1e-3;   WARM=$((12288/64)) ;;  # RCP ref_64
  gpt_oss_20b:128) LR=1e-3;   WARM=$((12288/128)) ;; # no RCP row (largest is 64): exploratory 16n DP-only run, not for the package
  *) die "GBS=$GBS has no RCP row for $BENCH" ;;
esac
LR=${LR_OVERRIDE:-$LR}; WARM=${WARM_OVERRIDE:-$WARM}   # e.g. the RCP row's own hyperparameters at GBS 32

export DGXSYSTEM=MI355X_${NN}x8x1 MLLOG_CONFIG_FILENAME=config_MI355X_${NN}x8x1.sh
export MLLOG_SUBMISSION_ORG=Crusoe MLLOG_SUBMISSION_PLATFORM=MI355X
export SEED NNODES=$NN NODE_RANK=$IDX GPUS_PER_NODE=8 MASTER_ADDR=${JOB}-0.${JOB} MASTER_PORT=$PORT
export SLURM_NNODES=$NN SLURM_NODEID=$IDX            # AMD's run_and_time.sh multi-node path
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export PRIMUS_GLOBAL_BATCH_SIZE=$GBS PRIMUS_MICRO_BATCH_SIZE=$MBS MLLOG_MICRO_BATCH_SIZE=$MBS
export PRIMUS_LR=$LR PRIMUS_MIN_LR=$(python3 -c "print($LR/10)") PRIMUS_LR_WARMUP_ITERS=$WARM
export PRIMUS_EVAL_INTERVAL=$((12288 / GBS))         # eval every 12288 samples
export PRIMUS_EVAL_ITERS=$((1024 / GBS)) EVAL_ITERS=$((1024 / GBS))   # 1024 eval samples
export PRIMUS_LR_DECAY_ITERS=$((1200000 - WARM))
mkdir -p /results/artifacts
if [ "$BENCH" = llama31_8b ]; then
  # yaml hardcodes lr_decay_iters (1199936 = AMD's warmup 64); keep decay == 1.2M - warmup
  sed "s/^\( *lr_decay_iters:\).*/\1 $PRIMUS_LR_DECAY_ITERS/" "$EXP" > conf/run.yaml; export EXP=/workspace/code/conf/run.yaml
else
  cp "$EXP" conf/run.yaml; export EXP=/workspace/code/conf/run.yaml
  export MLPERF_VERBOSE_LOGS=1                     # keep the full trainer output in the pod log
  export NCCL_DEBUG=WARN; unset NCCL_DEBUG_FILE    # AMD's config writes one INFO file per process
fi
# Parallelism into the run config (values, not env refs, so the trainer and MLLOG agree).
sed -i "s/^\( *tensor_model_parallel_size:\).*/\1 $TP/; s/^\( *expert_model_parallel_size:\).*/\1 $EP/" conf/run.yaml
if [ "$CP" -gt 1 ]; then
  grep -q "^ *context_parallel_size:" conf/run.yaml && sed -i "s/^\( *context_parallel_size:\).*/\1 $CP/" conf/run.yaml \
    || sed -i "s/^\( *\)tensor_model_parallel_size:.*$/&\n\1context_parallel_size: $CP/" conf/run.yaml
fi
if [ "$TP" -gt 1 ]; then
  grep -q "^ *sequence_parallel:" conf/run.yaml && sed -i "s/^\( *sequence_parallel:\).*/\1 true/" conf/run.yaml \
    || sed -i "s/^\( *\)tensor_model_parallel_size:.*$/&\n\1sequence_parallel: true/" conf/run.yaml
fi
# Experiment overrides into the run config: YAML_SET="key=value,key2=value2" replaces each key where it
# appears, or inserts it next to micro_batch_size (modules.pre_trainer.overrides) when it is absent.
# Separator is ';' when present (values may contain commas, e.g. lists), else ','.
YS_SEP=','; case "${YAML_SET:-}" in *\;*) YS_SEP=';' ;; esac
for kv in $(echo "${YAML_SET:-}" | tr "$YS_SEP" ' '); do
  k=${kv%%=*}; v=${kv#*=}
  if grep -q "^ *$k:" conf/run.yaml; then sed -i "s|^\( *$k:\).*|\1 $v|" conf/run.yaml
  else sed -i "s|^\( *\)micro_batch_size:\(.*\)$|&\n\1$k: $v|" conf/run.yaml; fi
done
[ -n "${YAML_SET:-}" ] && say "YAML_SET: $YAML_SET"
export PRIMUS_TP=$TP PRIMUS_EP=$EP MLLOG_TENSOR_PARALLELISM=$TP MLLOG_CONTEXT_PARALLELISM=$CP MLLOG_EXPERT_PARALLELISM=$EP
# Megatron manual GC (math unchanged): all ranks collect at the same point instead of stalling at random.
if [ "${MANUAL_GC:-0}" = 1 ]; then
  sed -i "s/^\( *\)overlap_grad_reduce:.*$/&\n\1manual_gc: true\n\1manual_gc_interval: 0\n\1manual_gc_eval: true/" conf/run.yaml
fi
# Perf mode: a fixed number of samples, per-step timing in the log. Not a compliant run.
if [ -n "${PERF_SAMPLES:-}" ]; then
  export PRIMUS_TRAIN_ITERS=$(( PERF_SAMPLES / GBS )) PRIMUS_LOG_INTERVAL=10 LOG_INTERVAL=10
fi
grep -nE "tensor_model_parallel_size|context_parallel_size|expert_model_parallel_size|sequence_parallel|manual_gc|num_workers|cp_comm_type|dispatcher|deepep" conf/run.yaml

# Multi-node: once RCCL registers memory with the ionic NICs, DataLoader workers forked from the
# trainer SIGSEGV at the first real batch (seen on every node, never at 1 node). Primus ships a patch
# for exactly this (megatron.dataloader_mp_context) but AMD's yaml leaves it null; start workers
# from a forkserver instead. Worker start method only - no effect on the data or the math.
if [ "$NN" -gt 1 ]; then
  sed -i "s/^\( *\)micro_batch_size:\(.*\)$/&\n\1dataloader_mp_context: forkserver/" conf/run.yaml
  grep -nE "dataloader_mp_context" conf/run.yaml || die "could not set dataloader_mp_context"
  # forkserver workers initialise HIP themselves, and with AMD's HSA_ENABLE_INTERRUPT=0 each one runs
  # 2 busy-polling runtime threads. llama inherits Primus' 8 workers per rank, and eval adds a second
  # iterator: 128 workers x 2 spinning threads per node > 236 cores, so the trainer starves after the
  # first eval (2n: eval 171 s vs 8 s at 1n). 2 per rank, as AMD ships for gpt_oss. Loader count only.
  NW=${NUM_WORKERS:-2}
  if grep -q "^ *num_workers:" conf/run.yaml; then sed -i "s/^\( *num_workers:\).*/\1 $NW/" conf/run.yaml
  else sed -i "s/^\( *\)micro_batch_size:\(.*\)$/&\n\1num_workers: $NW/" conf/run.yaml; fi
  export PRIMUS_NUM_WORKERS=$NW
fi
say "libionic: $(ls /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav*.so 2>/dev/null | tr "\n" " ") rccl: $(readlink -f /opt/rocm/lib/librccl.so.1)"
say "layout tp=$TP cp=$CP ep=$EP dp=$DP perf_samples=${PERF_SAMPLES:-none} manual_gc=${MANUAL_GC:-0} nccl_net_peer_ch=${NCCL_NCHANNELS_PER_NET_PEER:-} qps=${NCCL_IB_QPS_PER_CONNECTION:-}"
say "bench=$BENCH nodes=$NN gbs=$GBS mbs=$MBS dp=$DP grad_accum=$((GBS/(DP*MBS))) lr=$LR warmup=$WARM eval_int=$PRIMUS_EVAL_INTERVAL eval_iters=$PRIMUS_EVAL_ITERS seed=$SEED master=$MASTER_ADDR"

for i in $(seq 0 $((NN - 1))); do                 # every rank resolves every peer first
  until getent hosts "${JOB}-${i}.${JOB}" >/dev/null 2>&1; do sleep 2; done
done
say "all $NN peers resolvable"

bash ./run_and_time.sh
rc=$?
cp -f /results/mlperf_logging.out "$LOGD/mlperf_logging.node${IDX}.out" 2>/dev/null || true
cp -f /results/artifacts/*.full.log "$LOGD/" 2>/dev/null || true
cp -f /workspace/code/train*.log "$LOGD/train.node${IDX}.log" 2>/dev/null || true
say "exit rc=$rc"; sleep 3
exit $rc
