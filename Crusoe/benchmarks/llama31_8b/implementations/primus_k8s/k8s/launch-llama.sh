#!/bin/bash
# In-pod launcher for the MLPerf v6.1 Llama runs (mounted from ConfigMap llama-v61-launch).
# Env from the Job: BENCH (llama31_8b|llama2_70b_lora) NN GBS MBS SEED RUN_NAME JOB IDX.
#
# Base recipe = AMD's in-image config_MI355X_1x8x1.sh (the v6.0 submission). On top of it this
# sets the multi-node rendezvous and the batch-size-dependent hyperparameters, taken from the
# v6.1 RCP tables (mlperf_logging/rcp_checker/training_6.1.0/rcps_<bench>.json) so every
# scale is on an RCP row. Only GBS values that have an RCP row are accepted.
set -uo pipefail
ulimit -n 1048576 2>/dev/null || ulimit -n 262144 2>/dev/null || true
: "${BENCH:?}" "${NN:?}" "${GBS:?}" "${MBS:?}" "${SEED:?}" "${RUN_NAME:?}" "${JOB:?}" "${IDX:?}"
LOGD=/joblogs/${RUN_NAME}; mkdir -p "$LOGD"
exec > >(tee "$LOGD/rank${IDX}.log") 2>&1
say(){ echo "[launch $(date -u +%H:%M:%SZ) node${IDX}] $*"; }
die(){ say "!! $*"; sleep 3; exit 2; }        # sleep: let tee flush before the pod exits
REQ_CP=${CP:-1}; REQ_TP=${TP:-1}              # AMD's config below exports CP=1/TP=1; keep ours

cd /workspace/code
source ./config_MI355X_1x8x1.sh
export DGXSYSTEM=MI355X_${NN}x8x1
export MLLOG_CONFIG_FILENAME=config_${DGXSYSTEM}.sh
export MLLOG_SUBMISSION_ORG=Crusoe MLLOG_SUBMISSION_PLATFORM=MI355X
export SEED
export NNODES=$NN NODE_RANK=$IDX GPUS_PER_NODE=8
export MASTER_ADDR=${JOB}-0.${JOB} MASTER_PORT=29502
export SLURM_NNODES=$NN SLURM_NODEID=$IDX          # AMD's run_and_time.sh multi-node path
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export MLLOG_OUTPUT_FILE=/results/mlperf_logging.out; mkdir -p /results
# Keep stderr: AMD's wrappers send it to /dev/null, which hides every failure. Result files
# only take the :::MLLOG lines, so the extra output does not affect submission logs.
export LLAMA_DEBUG=${LLAMA_DEBUG:-1} MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-1}

case "$BENCH" in
  llama31_8b)
    # RCP rows (6.1): BS -> lr, warmup samples. BS32 keeps AMD's exact v6.0 recipe.
    case "$GBS" in
      32)  LR=8e-4; WARM_IT=64 ;;                         # AMD v6.0 1x8x1 as submitted
      64)  LR=8e-4; WARM_IT=$((6144/64)) ;;
      96)  LR=1e-3; WARM_IT=$((16348/96)) ;;
      128) LR=2e-3; WARM_IT=$((32768/128)) ;;
      *) die "GBS=$GBS has no llama31_8b RCP row (16/32/64/96/128)" ;;
    esac
    LR=${LR_OVERRIDE:-$LR}                              # tuned LR (logged by mllog as opt_base_learning_rate)
    export PRIMUS_GLOBAL_BATCH_SIZE=$GBS PRIMUS_MICRO_BATCH_SIZE=$MBS MLLOG_MICRO_BATCH_SIZE=$MBS
    export PRIMUS_LR=$LR PRIMUS_MIN_LR=$(python3 -c "print($LR/10)") PRIMUS_LR_WARMUP_ITERS=$WARM_IT
    export PRIMUS_EVAL_INTERVAL=$((12288 / GBS))        # eval every 12288 samples
    export PRIMUS_EVAL_ITERS=$((1024 / GBS))            # 1024 eval samples (checker: eval_samples == 1024)
    # compliance: opt_learning_rate_decay_steps == 1_200_000 - warmup_steps (yaml hardcodes 1199936)
    sed "s/^\( *lr_decay_iters:\).*/\1 $((1200000 - WARM_IT))/" conf/llama3.1_8B-pretrain-fp4.yaml > conf/run.yaml
    # Model parallelism (default DP only). The largest RCP batch is 128, so at 32 nodes (256 GPUs)
    # 8B needs TP2 or CP2 to keep DP*MBS <= 128. AMD's yaml has tensor_model_parallel_size but no
    # context_parallel_size, so CP is inserted next to it.
    TP=$REQ_TP; CP=$REQ_CP
    DP=$(( 8 * NN / (TP * CP) ))
    [ $(( GBS % (DP * MBS) )) -eq 0 ] || die "GBS=$GBS not a multiple of DP*MBS=$((DP*MBS)) (NN=$NN TP=$TP CP=$CP)"
    sed -i "s/^\( *\)tensor_model_parallel_size:.*/\1tensor_model_parallel_size: $TP/" conf/run.yaml
    [ "$TP" -gt 1 ] && sed -i "s/^\( *\)tensor_model_parallel_size: $TP$/&\n\1sequence_parallel: true/" conf/run.yaml
    [ "$CP" -gt 1 ] && sed -i "s/^\( *\)tensor_model_parallel_size: $TP$/&\n\1context_parallel_size: $CP/" conf/run.yaml
    export MLLOG_TENSOR_PARALLELISM=$TP MLLOG_CONTEXT_PARALLELISM=$CP
    grep -nE "tensor_model_parallel_size|sequence_parallel|context_parallel_size" conf/run.yaml
    # Optional: Megatron manual GC (no effect on the math). Python's automatic gen-2 collections
    # hit ranks at different times, so every collective waits for the slowest rank; disabling
    # them and collecting at eval (all ranks pause together) removes that straggler noise.
    if [ "${MANUAL_GC:-0}" = 1 ]; then
      sed -i "s/^\( *\)overlap_grad_reduce: true/&\n\1manual_gc: true\n\1manual_gc_interval: 0\n\1manual_gc_eval: true/" conf/run.yaml
      grep -nE "manual_gc" conf/run.yaml
    fi
    export EXP=/workspace/code/conf/run.yaml
    ;;
  llama2_70b_lora)
    case "$GBS" in
      8)   LR=0.0006; STEPS=550 ;;                        # AMD v6.0 1x8x1 as submitted
      16|32) LR=0.0004; STEPS=1024 ;;
      128) LR=0.001; STEPS=1024 ;;
      *) die "GBS=$GBS has no llama2_70b_lora RCP row (8/16/32/128)" ;;
    esac
    LR=${LR_OVERRIDE:-$LR}                              # tuned LR (logged by mllog as opt_base_learning_rate)
    # Scale-out is by context parallelism (CP), like every multi-node v6.0 LoRA submission:
    # it keeps GBS small (fastest convergence). GBS must equal (8*NN/(TP*CP)) * MBS * grad-accum.
    CP=$REQ_CP; TP=$REQ_TP
    DP=$(( 8 * NN / (TP * CP) ))
    [ $(( GBS % (DP * MBS) )) -eq 0 ] || die "GBS=$GBS not a multiple of DP*MBS=$((DP*MBS)) (NN=$NN TP=$TP CP=$CP)"
    export PRIMUS_GLOBAL_BATCH_SIZE=$GBS PRIMUS_MICRO_BATCH_SIZE=$MBS MLLOG_MICRO_BATCH_SIZE=$MBS
    export MLLOG_CONTEXT_PARALLELISM=$CP MLLOG_TENSOR_PARALLELISM=$TP CP TP
    export LR PRIMUS_TRAIN_ITERS=$STEPS
    export PRIMUS_EVAL_INTERVAL=$((384 / GBS > 0 ? 384 / GBS : 1))   # eval every 384 samples
    export PRIMUS_EVAL_ITERS=$(( (192 + GBS - 1) / GBS ))             # >= full validation set
    sed -e "s/^\( *lr_decay_iters:\).*/\1 $STEPS/" \
        -e "s/^\( *context_parallel_size:\).*/\1 $CP/" \
        -e "s/^\( *tensor_model_parallel_size:\).*/\1 $TP/" \
        -e "s/^\( *sequence_parallel:\).*/\1 $([ "$TP" -gt 1 ] && echo true || echo false)/" \
        conf/llama2_70b_lora_posttrain.yaml > conf/run.yaml
    # Megatron-Bridge refuses CP>1 fine-tuning without per-token loss (NeMo's CP LoRA sets it
    # too). AMD's recipe builds model_cfg itself and drops unknown YAML keys, so patch the recipe
    # (idempotent): calculate_per_token_loss = CP > 1. average_in_collective is already False.
    R=/workspace/Primus/primus/recipes/llama2_custom.py
    grep -q "model_cfg.calculate_per_token_loss" "$R" || \
      sed -i 's/^\( *\)model_cfg.context_parallel_size = context_parallel_size$/&\n\1model_cfg.calculate_per_token_loss = context_parallel_size > 1/' "$R"
    # Manual GC (same knob as the 8B path): the recipe hardcodes it off in TrainingConfig.
    if [ "${MANUAL_GC:-0}" = 1 ]; then
      sed -i -e 's/^\( *\)manual_gc=False,/\1manual_gc=True,/' -e 's/^\( *\)manual_gc_eval=False,/\1manual_gc_eval=True,/' "$R"
      grep -nE "manual_gc" "$R" | grep -v import
    fi
    export EXP=/workspace/code/conf/run.yaml
    ;;
  *) die "unknown BENCH=$BENCH" ;;
esac
say "bench=$BENCH nodes=$NN gbs=$GBS mbs=$MBS tp=${TP:-1} cp=${CP:-1} lr=${PRIMUS_LR:-$LR} seed=$SEED master=$MASTER_ADDR"

# Readiness barrier: every rank resolves every peer before torchrun's rendezvous opens.
for i in $(seq 0 $((NN - 1))); do
  until getent hosts "${JOB}-${i}.${JOB}" >/dev/null 2>&1; do sleep 2; done
done
say "all $NN peers resolvable"

# v6.1-required keys the v6.0 images don't emit - rank 0 only.
#   cache_clear: logged ONLY if this run's cache-flush initContainer left evidence for every
#   node (it drops the host page cache, incl. NFS client cache, before the run).
#   Precision disclosures (the images log only the older single lowest_numerical_precision_linear
#   key): linear = mxfp4 (FP4 recipe); attn = bfloat16 (FP8 dot-product attention is off in both
#   configs); comm = bfloat16 (grads reduced in bf16, accumulate_allreduce_grads_in_fp32=false).
if [ "$IDX" = 0 ]; then
  for _ in $(seq 1 180); do            # other nodes' initContainers may still be flushing
    n_flush=$(ls "$LOGD"/cacheflush-node*.txt 2>/dev/null | wc -l)
    [ "$n_flush" -ge "$NN" ] && break; sleep 5
  done
  echo "[cache-flush evidence: $n_flush/$NN nodes]"; cat "$LOGD"/cacheflush-node*.txt 2>/dev/null
  [ "$n_flush" -eq "$NN" ] || die "cache flush evidence missing ($n_flush/$NN nodes) - refusing to log cache_clear"
  PREC_LINEAR=${MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR:-mxfp4} PREC_ATTN=${PREC_ATTN:-bfloat16} PREC_COMM=${PREC_COMM:-bfloat16} \
  python3 - <<'PY'
import json, os, time
def ev(k, v):
    print(":::MLLOG " + json.dumps({"namespace": "", "time_ms": int(time.time() * 1000), "event_type": "POINT_IN_TIME",
                                    "key": k, "value": v, "metadata": {"file": "launch-llama.sh", "lineno": 1}}), flush=True)
ev("cache_clear", True)
ev("lowest_numerical_precision_in_linear", os.environ["PREC_LINEAR"])
ev("lowest_numerical_precision_in_attn", os.environ["PREC_ATTN"])
ev("lowest_numerical_precision_in_comm", os.environ["PREC_COMM"])
PY
fi

if [ "${LLAMA_DEBUG:-0}" = 1 ] && [ "$BENCH" = llama31_8b ]; then
  # AMD's run_and_time.sh torchrun, with stderr kept and STATIC rendezvous: the c10d rdzv
  # backend timed out across pods (TCPStore socket timeout on every rank at 8 nodes), while
  # static master_addr/port rendezvous is what the working LoRA path (primus-cli) uses.
  torchrun --nproc_per_node=8 --nnodes=$NN --node_rank=$IDX --master_addr=$MASTER_ADDR \
           --master_port=$MASTER_PORT src/train.py
else
  MLLOG_VERBOSE_LOGS=${LLAMA_DEBUG:-0} bash ./run_and_time.sh
fi
rc=$?
cp -f /results/mlperf_logging.out "$LOGD/mlperf_logging.node${IDX}.out" 2>/dev/null || true
say "exit rc=$rc"; sleep 3
exit $rc
