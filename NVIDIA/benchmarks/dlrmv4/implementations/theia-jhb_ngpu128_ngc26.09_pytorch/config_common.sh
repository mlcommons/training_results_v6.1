# Hyperparameters map to the gin env bindings in
# generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin.

export DATASET="yambda-5b"
export TRAIN_MODE="streaming-train-eval"

# Workers per loader per rank; prefetch factors are batches per worker.
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"
export WORKER_NUM_THREADS="${WORKER_NUM_THREADS:-4}"
export TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-2}"
export EVAL_PREFETCH_FACTOR="${EVAL_PREFETCH_FACTOR:-8}"
export PERSISTENT_LOADER="${PERSISTENT_LOADER:-1}"
# Double buffering uses two train worker pools; eval has a separate pool.
export DOUBLE_BUFFER="${DOUBLE_BUFFER:-1}"
export USE_PIPELINE="${USE_PIPELINE:-1}"
export PACKED_H2D="${PACKED_H2D:-1}"

# must match the staged hstu_cache_L2039 or the first run rebuilds it
export HISTORY_LENGTH=4086
export MIN_HISTORY=4086
export MAX_SEQ_LEN=4096

# bf16 is only numerically safe with the fused TRITON kernels
export HSTU_HAMMER_KERNEL=TRITON
# Use the SM10x cuDNN Frontend HSTU attention implementation.
export HSTU_USE_CUDNN_ATTENTION="${HSTU_USE_CUDNN_ATTENTION:-1}"
# Select cuDNN LMSD independently of the attention backend.
export HSTU_USE_CUDNN_LMSD="${HSTU_USE_CUDNN_LMSD:-1}"
# Compute only the suffix candidate in the last HSTU layer, without recompute.
# Set to 0 for a full-layer baseline. Dropout uses compact row numbering.
export HSTU_LAST_LAYER_CANDIDATE_ONLY="${HSTU_LAST_LAYER_CANDIDATE_ONLY:-1}"
# the pinned autotune configs are MI350X-tuned
export TRITON_FULL_AUTOTUNE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Constrain every Yambda embedding table to row-wise sharding by default.
export EMB_SHARDING_OVERRIDES="${EMB_SHARDING_OVERRIDES:-item_id=row_wise,\
artist_id=row_wise,album_id=row_wise,uid=row_wise,user_x_artist=row_wise,\
user_x_album=row_wise,user_x_hour=row_wise,item_x_hour=row_wise,\
artist_x_hour=row_wise,user_x_is_organic=row_wise,\
user_x_artist_x_hour=row_wise}"

export COMPACTED_PREPROCESSOR="${COMPACTED_PREPROCESSOR:-1}"
export EC_INDEX_DEDUP="${EC_INDEX_DEDUP:-1}"
# Use torch.gather instead of index_select to restore deduplicated embedding
# outputs. The same knob enables the TorchRec index-dedup path it depends on.
export TORCHREC_USE_GATHER_SELECT="${TORCHREC_USE_GATHER_SELECT:-0}"

# Compacted preprocessing already deduplicates content keys.
case "${COMPACTED_PREPROCESSOR,,}" in
    1|true)
        export COMPACTED_PREPROCESSOR=1
        export EC_INDEX_DEDUP=0
        export TORCHREC_USE_GATHER_SELECT=0
        ;;
    0|false) export COMPACTED_PREPROCESSOR=0 ;;
    *)
        echo "COMPACTED_PREPROCESSOR must be 0/1 or True/False" >&2
        return 1
        ;;
esac

# run.sub launches one task per GPU, so bindpcie applies independently to each
# rank and places its worker threads near the assigned GPU.
export BINDCMD="${BINDCMD:-bindpcie --cpu=node}"

# reference convergence target (upstream README target: 0.80275; reference default: 0.75)
export AUC_THRESHOLD=0.75
export MLPERF_LOGGING=1
# Synthetic full-model forward/backward passes before MLPerf RUN_START. Warmup
# uses a throwaway embedding collection and never steps the dense optimizer.
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-20}"

# Optional diagnostic logging. yambda_5b.gin resolves these through env_int and
# injects them into make_model/MetricsLogger; set either to 1 to restore the
# corresponding legacy behavior.
export ENABLE_TFLOPS_LOGGING="${ENABLE_TFLOPS_LOGGING:-0}"
export ENABLE_TRAIN_METRICS="${ENABLE_TRAIN_METRICS:-0}"

# lowest_numerical_precision_in_* disclosure (training_6.1.0/common.yaml).
# linear/attn: make_model.bf16_training=True autocasts the whole HSTU
# transducer to bfloat16; dlrmv4 has no fp8/fp4 path.
# comm: the sparse embedding all-to-all is quantized via TorchRec QCommsConfig.
# MUST stay in sync with SPARSE_A2A_FWD/SPARSE_A2A_BWD (gin default fp16/fp16);
# set both of those to fp32 to disable quantization and this becomes fp32.
# Dense gradients are not compressed (no DDP comm hook), so they go out at fp32.
export MLPERF_LINEAR_PRECISION="bfloat16"
export MLPERF_ATTN_PRECISION="bfloat16"
export MLPERF_COMM_PRECISION="fp16"
# cadence (train steps) of the MLPerf tracked_stats train_step_time event
export MLPERF_STEP_TIME_EVERY=10

# CKPT_PATH is unset, so the hourly checkpoint clock would only add a per-step
# NCCL broadcast + host sync; literal 0 disables it ("" falls back to 3600)
export CKPT_TIME_INTERVAL_S=0
