#!/bin/bash
# In-pod launcher for AMD's MLPerf v6.1 customer-drop recipes (flux1 = Primus diffusion, dlrmv4 = PyTorch
# generative-recommenders) on N MI355X nodes. Mounted from ConfigMap cdrop-launch.
# Env from the Job: BENCH NN GBS MBS SEED RUN_NAME JOB IDX [EXTRA_EXPORTS LR_OVERRIDE WARM_OVERRIDE].
# AMD ships 1- and 4-node configs (dlrmv4 also 2-node); this writes the equivalent config for any node count
# with hyperparameters from AMD's config at that GBS, or the RCP row (training_6.1.0) otherwise.
set -uo pipefail
ulimit -n 1048576 2>/dev/null || ulimit -n 262144 2>/dev/null || true
: "${BENCH:?}" "${NN:?}" "${GBS:?}" "${MBS:?}" "${SEED:?}" "${RUN_NAME:?}" "${JOB:?}" "${IDX:?}"
LOGD=/joblogs/${RUN_NAME}; mkdir -p "$LOGD"
exec > >(tee "$LOGD/rank${IDX}.log") 2>&1
say(){ echo "[launch $(date -u +%H:%M:%SZ) node${IDX}] $*"; }
die(){ say "!! $*"; sleep 3; exit 2; }
cd /workspace/code || die "no /workspace/code"

# Multi-node RCCL budget (ionic: 4096 QPs/NIC). AMD's flux config asks for 32 channels per net peer.
if [ "$NN" -gt 1 ]; then export NCCL_IB_QPS_PER_CONNECTION=1 NCCL_NCHANNELS_PER_NET_PEER=8; fi
for kv in ${EXTRA_EXPORTS:-}; do case "$kv" in -*) unset "${kv#-}" ;; *) export "$kv" ;; esac; done
[ -n "${EXTRA_EXPORTS:-}" ] && say "EXTRA_EXPORTS: $EXTRA_EXPORTS"
export SEED NNODES=$NN NODE_RANK=$IDX GPUS_PER_NODE=8 SLURM_NNODES=$NN SLURM_NODEID=$IDX
export MASTER_ADDR=${JOB}-0.${JOB} MASTER_PORT=29503
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME_OVERRIDE:-eth0} GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME_OVERRIDE:-eth0}
export MLLOG_SUBMISSION_ORG=Crusoe MLLOG_SUBMISSION_PLATFORM=MI355X MLPERF_SUBMISSION_PLATFORM=MI355X
mkdir -p /results/artifacts

case "$BENCH" in
flux1)
  # Hyperparameters per GBS: AMD's configs at 256 (1n) and 1024 (4n); RCP rows otherwise.
  case "$GBS" in
    256)  LR=2.5e-4; WARM=800 ;;   # AMD 1x8x32 (no RCP row; dev only)
    512)  LR=2e-4;   WARM=1600 ;;  # RCP
    1024) LR=2.5e-4; WARM=1600 ;;  # AMD 4x8x32
    2048) LR=2.5e-4; WARM=0 ;;     # RCP
    4096) LR=4e-4;   WARM=400 ;;   # v6.0 set B (RCP row warmup 100 converged too fast)
    *) die "flux1 GBS=$GBS has no config" ;;
  esac
  LR=${LR_OVERRIDE:-$LR}; WARM=${WARM_OVERRIDE:-$WARM}
  [ $(( GBS % (8 * NN * MBS) )) -eq 0 ] || die "GBS=$GBS not a multiple of $((8*NN))*MBS=$MBS"
  GA=$(( GBS / (8 * NN * MBS) ))
  SYS=MI355X_${NN}x8x${MBS}
  # Drop 253c561+: node configs set their values BEFORE sourcing config_common.sh (it derives FP4/FP8 all-gather
  # settings from them) and add a mid-run switch of the forward pass to FP8 (AMD: 4n 7325 = 7.5M samples, 8n 4395 =
  # 9.0M). FPROP_SWITCH="" disables it. Older drops: source first, no switch.
  if grep -q FLUX_FPROP_SWITCH_PRECOMPILE config_common.sh 2>/dev/null; then
    case "$GBS" in 1024) FPS=7325 ;; 2048) FPS=4395 ;; *) FPS= ;; esac
    FPS=${FPROP_SWITCH-$FPS}; PRE=""; POST="source \"\$(dirname -- \"\${BASH_SOURCE[0]}\")/config_common.sh\""
    say "flux1 new-drop config: FLUX_FPROP_SWITCH_STEP=${FPS:-off}"
  else
    FPS=; PRE="source \"\$(dirname -- \"\${BASH_SOURCE[0]}\")/config_common.sh\""; POST=""
  fi
  cat > "config_${SYS}.sh" <<EOF
$PRE
export NNODES=$NN
export DGXNNODES=$NN
export LOCAL_BATCH_SIZE=$MBS
export GRADIENT_ACCUMULATION_STEPS=$GA
export EVAL_BATCH_SIZE=\${EVAL_BATCH_SIZE:-$MBS}
export GLOBAL_BATCH_SIZE=$GBS
export LR=$LR
export WARMUP_STEPS=$WARM
${FPS:+export FLUX_FPROP_SWITCH_STEP=$FPS}
$POST
EOF
  export DGXSYSTEM=$SYS
  # Multi-node: AMD's rccl_sdma all-gather hook exports NCCL_CUMEM_ENABLE=1, which makes RCCL set up
  # symmetric-memory RMA QPs on the ionic NICs; those fail ("local access violation work queue error") and
  # RCCL's cleanup SIGSEGVs (ncclRmaIbProxyDestroyContext). Skip the RMA proxy; transport-only, math unchanged.
  # Root cause and repro: v61drop/flux-debug/.
  [ "$NN" -gt 1 ] && export NCCL_RMA_DISABLE=${NCCL_RMA_DISABLE:-1}
  # INDUCTOR_LOCAL=1: per-node cache in the container (cold compile) instead of the shared NFS cache, which
  # 32 ranks on 4 nodes write at once (a rank hit "could not get source code" reading a cached kernel there).
  # Prefer the node's RAID0 (/data, persists across runs) so each node compiles once; a cold cache recompiles the
  # eval graph at the first eval inside the timed region, where an inductor compile-worker hang stalled a run.
  if [ "${INDUCTOR_LOCAL:-0}" = 1 ]; then
    if mkdir -p /data/inductor-${SYS} 2>/dev/null && [ -w /data/inductor-${SYS} ]; then export TORCHINDUCTOR_CACHE_DIR=/data/inductor-${SYS}
    else export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-${SYS}; fi
  else export TORCHINDUCTOR_CACHE_DIR=/inductor-cache/flux1-${SYS}; fi
  [ -n "${PERF_SAMPLES:-}" ] && export MAX_STEPS=$(( PERF_SAMPLES / GBS ))
  # FLUX_LOCAL_DATA=1: read the dataset from the node-local RAID0 copy made by flux-stage.sh (AMD's
  # scripts/stage_dataset_local.sh) instead of NFS; AMD's own launcher refuses to train from NFS. The copy is made
  # before the run, so this run's init-container page-cache flush still sets the data state at start.
  if [ "${FLUX_LOCAL_DATA:-0}" = 1 ]; then
    FLD=/data/flux1
    df -P /data | awk 'NR==2{print $1}' | grep -q '^/dev/md' || die "/data is not on the md RAID0 array; cannot use local Flux data"
    grep -qx 'source=/dataset' "$FLD/.stage_complete" 2>/dev/null || die "no complete local Flux copy at $FLD on $(hostname) (run flux-stage.sh for this node)"
    export DATASET_PATH=$FLD/cc12m_preprocessed EVAL_DATASET_PATH=$FLD/coco_preprocessed EMPTY_ENCODINGS_PATH=$FLD/empty_encodings
    say "flux1 data: local $FLD ($(tr '\n' ' ' < $FLD/.stage_complete))"
  else
    for d in cc12m_preprocessed coco_preprocessed empty_encodings; do [ -d /dataset/$d ] || die "missing /dataset/$d"; done
  fi
  say "flux1 sys=$SYS nodes=$NN gbs=$GBS mbs=$MBS ga=$GA lr=$LR warmup=$WARM fprop_switch=${FPS:-off} data=${DATASET_PATH:-/dataset/cc12m_preprocessed} seed=$SEED max_steps=${MAX_STEPS:-30000} ag=${FSDP_ALL_GATHER_BACKEND:-rccl_sdma} nch=${NCCL_NCHANNELS_PER_NET_PEER:-} qps=${NCCL_IB_QPS_PER_CONNECTION:-}"
  ;;
dlrmv4)
  # GBS rows 8192/16384/32768; LR scales linearly (AMD: 1e-6 at 8192); SKIP_EVAL_EPOCH_PCT from AMD's config
  # at the same GBS (the first eval must land on the checker's grid).
  case "$GBS" in
    8192)  SKIP=0.025 ;; 16384) SKIP=0.0303 ;; 32768) SKIP=0.041 ;;
    *) die "dlrmv4 GBS=$GBS has no config" ;;
  esac
  BS=$(( GBS / (8 * NN) )); [ $(( BS * 8 * NN )) -eq "$GBS" ] || die "GBS=$GBS not divisible by $((8*NN)) ranks"
  LR=${LR_OVERRIDE:-$(python3 -c "print(1e-6*$GBS/8192)")}
  export DGXSYSTEM=MI355X_${NN}x8x1 WORLD_SIZE=$((8*NN)) BATCH_SIZE=$BS DENSE_LR=$LR SPARSE_LR=$LR
  export LR_WARMUP_STEPS=${WARM_OVERRIDE:-24000} LR_WARMUP_START_LR=0 SKIP_EVAL_EPOCH_PCT=$SKIP
  export START_TS=0 NUM_TRAIN_TS=${NUM_TRAIN_TS:-299} NUM_TRAIN_BATCHES=${NUM_TRAIN_BATCHES:-0} NUM_EVAL_BATCHES=0
  export EVAL_EVERY_N_WINDOWS=0 EVAL_EVERY_DATA_PCT=0.001 AUC_THRESHOLD=0.75 MLPERF_LOGGING=1
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=WARN
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MLPERF_VERBOSE_LOGS=${MLPERF_VERBOSE_LOGS:-1}
  [ "$NN" -eq 1 ] && export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1
  # AMD's image force-installs torchaudio 2.11 over torch 2.12; its native lib (libtorchaudio.so) is missing,
  # and torchmetrics imports torchaudio when present -> import-time OSError. DLRM never uses audio, so remove it
  # (torchmetrics then treats it as unavailable). No change to the training code.
  python3 -c "import torchaudio" 2>/dev/null || { pip uninstall -y torchaudio >/dev/null 2>&1; say "removed broken torchaudio"; }
  # MLLOG submission_org: AMD's logger defaults submitter_name to "AMD" (mlperf_logging_utils.py:80,756).
  sed -i 's/submitter_name: str = "AMD"/submitter_name: str = "Crusoe"/' generative_recommenders/dlrm_v4/train/mlperf_logging_utils.py
  grep -q 'submitter_name: str = "Crusoe"' generative_recommenders/dlrm_v4/train/mlperf_logging_utils.py || die "submitter patch failed"
  # Stage the shared dataset onto node-local RAID0 /local/dlrmv4 once per node (AMD refuses to train from NFS).
  export DLRM_DATA_PATH=/data
  # /data must be the node-local RAID0 (/local = md array). If the array is not mounted, /local is a plain directory
  # on the ~130 GB OS disk and staging 229 GB fills it (kubelet DiskPressure, node unusable): refuse instead.
  df -P /data | awk 'NR==2{print $1}' | grep -q '^/dev/md' || die "/data is not on the md RAID0 array ($(df -P /data | awk 'NR==2{print $1}')); is /local mounted on this node?"
  if [ ! -f /data/.stage_complete ]; then
    say "staging /dlrm_shared -> /data"
    t0=$(date +%s)
    ( cd /dlrm_shared && find . -type d -exec mkdir -p /data/{} \; && find . -type f ! -name '*.aria2' -print0 | xargs -0 -P 16 -I{} cp {} /data/{} ) || die "staging failed"
    touch /data/.stage_complete "$LOGD/.staged-node${IDX}"; say "staged $(du -sh /data | cut -f1) in $(( $(date +%s) - t0 ))s"
    # MLPerf "Data State at Start of Run": the init container flushed the page cache BEFORE this copy, so the
    # staged dataset is now cached. Flush again (needs a privileged pod) and leave evidence, or refuse to run.
    b=$(grep -E '^Cached:' /proc/meminfo | tr -s ' ')
    sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || die "post-staging cache flush failed (run DLRM with PRIV=true); not training on a warm cache"
    a=$(grep -E '^Cached:' /proc/meminfo | tr -s ' ')
    echo "$(date -u +%FT%TZ) node=$(hostname) post-staging drop_caches=3 before=[$b] after=[$a]" | tee "$LOGD/cacheflush-poststage-node${IDX}.txt"
  fi
  # Drop 253c561+ logs cache_clear=True only when MLPERF_CACHE_CLEARED=1 (else False plus a "not a compliant timed run"
  # warning). Set it only on the evidence: this node's init-container flush, and the post-staging flush if it staged.
  if grep -q 'drop_caches=3' "$LOGD/cacheflush-node${IDX}.txt" 2>/dev/null && \
     { [ ! -f "$LOGD/.staged-node${IDX}" ] || grep -q 'drop_caches=3' "$LOGD/cacheflush-poststage-node${IDX}.txt" 2>/dev/null; }; then
    export MLPERF_CACHE_CLEARED=1
  else export MLPERF_CACHE_CLEARED=0; say "!! no cache-flush evidence in $LOGD: run will log cache_clear=False"; fi
  say "dlrmv4 nodes=$NN gbs=$GBS local_bs=$BS lr=$LR warmup=$LR_WARMUP_STEPS skip=$SKIP cache_cleared=$MLPERF_CACHE_CLEARED seed=$SEED train_batches=${NUM_TRAIN_BATCHES} nch=${NCCL_NCHANNELS_PER_NET_PEER:-} qps=${NCCL_IB_QPS_PER_CONNECTION:-}"
  ;;
*) die "unknown BENCH=$BENCH" ;;
esac
say "libionic: $(ls /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav*.so 2>/dev/null | tr '\n' ' ')"

for i in $(seq 0 $((NN - 1))); do until getent hosts "${JOB}-${i}.${JOB}" >/dev/null 2>&1; do sleep 2; done; done
say "all $NN peers resolvable"
bash ./run_and_time.sh
rc=$?
cp -f /results/mlperf_logging.out "$LOGD/mlperf_logging.node${IDX}.out" 2>/dev/null || true
cp -f /results/artifacts/*.full.log /results/artifacts/*.mlperf.log "$LOGD/" 2>/dev/null || true
say "exit rc=$rc"; sleep 3
exit $rc
