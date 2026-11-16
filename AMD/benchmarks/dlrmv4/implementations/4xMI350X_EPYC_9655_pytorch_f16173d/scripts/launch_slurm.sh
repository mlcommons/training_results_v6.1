#!/bin/bash
#SBATCH --job-name=yambda_slurm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=<YOUR_PARTITION>   # [CLUSTER-SPECIFIC] partition name
#SBATCH --output=yambda_slurm.%j.out
# =============================================================================
# launch_slurm.sh — single entry point for the yambda-5b trainer on N>=1 nodes.
#
# Consolidates what used to be separate scripts so multi-node enablement is
# ONE committable script (plus the train_ranker.py / utils.py python changes):
#   * orchestrate phase (host SLURM glue) — formerly sbatch_smoke_multinode.sh
#   * provision   phase (container + RDMA) — formerly _provision_yambda_primus.sh
#   * worker      phase (in-container train) — now inlined below
#
# PHASES (auto-detected from context; force with LAUNCH_SLURM_PHASE=<phase>):
#   orchestrate  Runs on the SLURM batch host (no /.dockerenv). Resolves the
#                rendezvous (MASTER_ADDR/PORT), stages the dataset to node-local
#                NVMe (STAGE_DATA=1 by default; falls back to NFS if any node
#                cannot take it), ensures the container on every node (provision
#                phase), then `docker exec`s the worker phase on every node, one
#                task per node.
#   provision    Runs on a compute-node host. Ensures the `yambda_primus`
#                container is up (loads the pre-baked image if present — no
#                internet/pip — else builds from the base image) and stages the
#                host RDMA userspace overlay on shared NFS.
#   worker       Runs INSIDE the container. Sets the distributed topology +
#                NCCL/RDMA env and spawns this node's GPU ranks via train_ranker.
#                N==1 transparently uses the legacy single-node path (localhost,
#                node_rank 0), byte-for-byte as before, so the streaming-e2e
#                supervisor's direct `bash scripts/launch_slurm.sh` is unchanged.
#
# USAGE
#   Multi-node (N>=1):  sbatch --nodes=2 scripts/launch_slurm.sh
#   Single-node direct: bash scripts/launch_slurm.sh   (already inside container;
#                       what run_streaming_e2e.sh invokes per relaunch)
#   Perf pair (compare_node_perf.sh parses the 'Step N perf' lines, which only
#   exist with METRIC_LOG_FREQ>0; the gin default 0 prints none). THROUGHPUT
#   RULE: every 200-step / ladder / rate perf arm sets METRIC_LOG_FREQ
#   explicitly, same value in control and candidate (add
#   MLPERF_TRAIN_LOSS_LOG_FREQ=1 when per-step MLLOG timing is needed):
#     LOG=/shared/perf_1node.log NUM_TRAIN_BATCHES=200 NUM_EVAL_BATCHES=0 \
#       EVAL_EVERY_N_WINDOWS=0 METRIC_LOG_FREQ=20 \
#       sbatch --nodes=1 --job-name=y1 scripts/launch_slurm.sh
#     LOG=/shared/perf_2node.log NUM_TRAIN_BATCHES=200 NUM_EVAL_BATCHES=0 \
#       EVAL_EVERY_N_WINDOWS=0 METRIC_LOG_FREQ=20 \
#       sbatch --nodes=2 --job-name=y2 scripts/launch_slurm.sh
#     # then: bash scripts/compare_node_perf.sh /shared/perf_1node.log /shared/perf_2node.log
#
# DATA LOCATION
#   Default: the training set is copied to node-local NVMe ($STAGE_DEST) before
#   training and read from there, so the shared filer is off the hot path. First
#   run on a node pays ~140 s / 372 GB; later runs skip files already staged.
#   If staging cannot be done, the run ABORTS rather than quietly reading NFS —
#   a silent switch back would change what the run measures with nothing in the
#   log to say so. The stager clears its own stale scratch before giving up.
#   STAGE_DATA=0            read straight from $DATA_MOUNT over NFS (explicit)
#   DLRM_DATA_PATH=<path>   read from <path>; suppresses staging entirely
#   STAGE_DEST=<path>       stage somewhere other than /tmp/$USER/dlrm_data
#   STAGE_CLEAN=0           do not reclaim stale scratch when space is short
#   scripts/stage_data_local.sh --purge   drop the local copy (it is scratch:
#                                         reproducible from NFS in ~140 s)
#
# ONE-TIME IMAGE BAKE (so fresh nodes skip the multi-GB torch download + pip):
#   BAKE_IMAGE=1 LAUNCH_SLURM_PHASE=provision bash scripts/launch_slurm.sh
#   (commits the deps-installed container to $BAKED_IMAGE and `docker save`s it to
#    $BAKED_TAR on NFS; subsequent provisions `docker load` it offline.)
#
# -----------------------------------------------------------------------------
# PORTABILITY — what to change for a DIFFERENT cluster / network / hardware.
# Every such knob is also tagged inline with "[CLUSTER-SPECIFIC]" (grep for it).
# All are env-overridable, so you can adapt without editing this file.
#
#  A) SLURM / scheduler
#     - #SBATCH --partition=PARTITION  : partition name. CHANGE per cluster.
#     - #SBATCH --time / --exclusive : policy; adjust to taste.
#
#  B) Filesystems (must be shared/NFS across ALL nodes — this script re-invokes
#     itself and reads the overlay + data from these paths cluster-wide)
#     - REPO_MOUNT (repo + this script, e.g. /home/<user>) is bind-mounted rw;
#       DATA_MOUNT (e.g. /shared) holds the read-only dataset + overlay +
#       baked tar + pip tarball; SCRATCH (e.g. /home/<user>/yambda_runs) is the
#       writable log/output root. Override any via env — nothing is user-hardwired.
#
#  C) Container image / GPU software stack (tied to the GPU arch + ROCm version)
#     - IMAGE=rocm/primus:v26.3        : base image. ROCm/AMD-specific.
#     - docker run --device=/dev/kfd --device=/dev/dri --group-add video : AMD ROCm
#       device passthrough. For NVIDIA this is --gpus all / nvidia runtime instead.
#     - --ulimit memlock=-1 : REQUIRED for RDMA QP registration (do not drop).
#     - TORCH_IDX (rocm7.2), torch/vision/audio ==*+rocm7.2, FBGEMM_WHL (a gfx950
#       wheel), torchrec pin : the whole deps set is arch/ROCm-version-specific.
#
#  D) Network fabric — THE trickiest part; defaults are PROVEN on PARTITION cluster
#     (Broadcom bnxt_re RoCEv2). On a different fabric these almost certainly change
#     (see the worker-phase block for the full rationale):
#     - NCCL_SOCKET_IFNAME=fenic0 : the ONE routable host NIC for TCP bootstrap.
#       Find yours with `ip -br addr`; the per-GPU RDMA NICs are usually NOT
#       routable for plain TCP, so auto-detect hangs init — you MUST pin this.
#     - NCCL_IB_HCA=bnxt_re0..7 : the RDMA HCA device names. List with `ibv_devices`.
#       Different NIC vendor (e.g. mlx5_*, ionic_*) => different names AND a
#       different userspace provider, which changes the RDMA overlay below.
#     - NCCL_IB_GID_INDEX=3 : RoCEv2 IPv4 GID index. Check `show_gids`; v1/v2 and
#       IPv4/IPv6 live at different indices per port.
#     - NCCL_IB_TC=104 : RoCE lossless (PFC) traffic class. Fabric/switch-specific.
#     - RDMA overlay (provision phase): only needed when the CONTAINER's rdma-core
#       is older than the HOST kernel driver's uapi (our bnxt_re v34-vs-v59 case).
#       Different NIC/host => different /usr/lib64 provider .so to stage, or the
#       overlay may be unnecessary entirely (set RDMA_OVERLAY= to disable). If RDMA
#       can't be made to work, NCCL_NET_TRANSPORT=socket falls back to TCP.
#
#  E) Not cluster-specific (auto-derived): GPUS_PER_NODE (torch.cuda.device_count),
#     NNODES/NODE_RANK/MASTER_ADDR (from SLURM), WORLD_SIZE.
# =============================================================================
set -uo pipefail

# Absolute path to THIS script so the orchestrate phase can re-invoke it on every
# node (home is shared NFS, so the same path resolves cluster-wide).
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# ---- phase detection --------------------------------------------------------
PHASE="${LAUNCH_SLURM_PHASE:-}"
if [ -z "$PHASE" ]; then
  if [ -f /.dockerenv ]; then PHASE=worker; else PHASE=orchestrate; fi
fi

# ---- shared config (env-overridable) ----------------------------------------
CONTAINER=${CONTAINER:-yambda_primus}
# Whether the caller named the repo, as opposed to it being defaulted here. The
# orchestrate phase normally re-derives it from the batch script's shared-NFS
# path (see there), and must not do so when the caller was explicit.
REPO_EXPLICIT=${REPO:+1}
REPO=${REPO:-$REPO_ROOT}                       # repo path inside the container
IMAGE=${IMAGE:-rocm/primus:v26.3}              # [CLUSTER-SPECIFIC] ROCm/arch base image
# The stack this run is DEFINED as -- the three components docs/training_recipe.md
# replaces in the base image (torch/vision/audio, fbgemm_gpu, torchrec). Asserted in
# both the provision and worker phases, because a failed install or a fresh container
# silently substitutes the image's natives (torch 2.10 / fbgemm 2026.5.14 / torchrec
# 1.4.0) and every measurement moves. See
# torch_212_index_scatter_regression.md (git history at 33610d2).
TORCH_PIN=${TORCH_PIN:-2.12.0+rocm7.2}
TORCHVISION_PIN=${TORCHVISION_PIN:-0.27.0+rocm7.2}  # ABI must match torch 2.12
TORCHAUDIO_PIN=${TORCHAUDIO_PIN:-2.11.0+rocm7.2}    # ABI must match torch 2.12
FBGEMM_PIN=${FBGEMM_PIN:-2026.6.2}          # FBGEMM commit 10b77573, gfx950
TORCHREC_PIN=${TORCHREC_PIN:-1.7.0a0+bf55480}   # git tag v2026.06.01.00

# Proof-of-provisioning marker, written INSIDE the container only after the dep
# install + import smoke both succeed, and checked before a container is reused.
#
# WHY: the reuse test used to be `docker exec $CONTAINER true`, i.e. it treated
# "responds" as "provisioned". Those came apart on 2026-08-25: a container was
# created from the base image at 23:43:27 with none of the pinned deps installed,
# and the next run found it alive, logged "container already up", skipped
# provisioning and trained on the image's torch 2.10 / torchrec 1.4.0 /
# fbgemm 2026.5.14. Nine percent of step time moved and nothing said a word.
# The stamp carries the pinned versions, so a container provisioned for a
# DIFFERENT pin set also fails the check instead of being silently adopted.
PROVISION_MARKER=${PROVISION_MARKER:-/opt/.launch_slurm_provisioned}
PROVISION_STAMP="torch=$TORCH_PIN torchvision=$TORCHVISION_PIN torchaudio=$TORCHAUDIO_PIN fbgemm=$FBGEMM_PIN torchrec=$TORCHREC_PIN"
BAKED_IMAGE=${BAKED_IMAGE:-yambda_primus_baked:latest}
BAKED_TAR=${BAKED_TAR:-${DATA_ROOT:-/data}/yambda_primus_baked.tar}   # [CLUSTER-SPECIFIC] shared-NFS path
USE_BAKED=${USE_BAKED:-1}
OVERLAY=${RDMA_OVERLAY:-${DATA_ROOT:-/data}/rdma_host_el9_new}        # [CLUSTER-SPECIFIC] shared-NFS RDMA overlay

REPO_MOUNT=${REPO_MOUNT:-$HOME}              # NFS home holding the repo (must contain $REPO); override if your repo lives elsewhere
DATA_MOUNT=${DATA_MOUNT:-${DATA_ROOT:-/data}}        # shared dataset + RDMA overlay + pip/fbgemm assets (read-only)
SCRATCH=${SCRATCH:-$HOME/yambda_runs}        # writable output root (logs / tb / traces)

# Copy the training set to node-local NVMe before training, and read it from
# there. DEFAULT ON. The streaming loaders mmap a ~350 GB flat-event cache and
# re-scan it at every data-window boundary; served from shared NFS that boundary
# step pays for the faults, and the run is also hostage to whatever else is
# hammering the filer. Staging takes the shared filesystem off the hot path.
#
# It does NOT move the steady-state rate (356.61 vs 359.72 ms/step measured
# 2026-08-25, i.e. nothing) -- it buys tail behaviour and independence from NFS
# contention, not mean throughput. The cost is one-time per node and idempotent:
# stage_data_local.sh skips files already present at the right size, so the
# second run on a node pays nothing.
#
# STAGE_DATA=0 opts out. An explicit DLRM_DATA_PATH also opts out, so a caller
# that already points somewhere specific is never second-guessed.
STAGE_DATA=${STAGE_DATA:-1}
# Where the local copy goes: node-local durable disk (/var/tmp, NVMe root on this
# cluster). Not /tmp: Slurm may give jobs a private /tmp, and the rules require
# data on durable storage, not RAM.
STAGE_DEST=${STAGE_DEST:-/var/tmp/$(id -un)/dlrm_data}   # [CLUSTER-SPECIFIC] node-local path

# =============================================================================
# PHASE: orchestrate  (SLURM batch host)
# =============================================================================
orchestrate() {
  # When run as the SLURM batch script, $0 is the node-local staged copy
  # (/var/spool/slurmd/job<ID>/slurm_script), so $SELF / $REPO_ROOT are WRONG
  # here (they don't exist on other nodes). Resolve the REAL shared-NFS script
  # path + repo root from SLURM so we can re-invoke this script on every node and
  # `cd` to the right repo inside the container.
  # The allocation's batch script is the right thing to re-invoke only when it IS
  # this script. Reusing an allocation created by some OTHER sbatch -- a node
  # holder, say -- otherwise re-runs that foreign script on every node: it was
  # found to re-run a node-holder sbatch from a subdirectory, which resolved REPO
  # to that subdirectory, tore down the running containers and left
  # nothing but a `provision rc=127` behind. So require the basename to match,
  # and honour an explicit override first.
  _want=launch_slurm.sh
  if [ ! -f "${SCRIPT_PATH:-}" ] || [ "$(basename "${SCRIPT_PATH:-}")" != "$_want" ]; then
    SCRIPT_PATH=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -oP 'Command=\K\S+')
    if [ ! -f "${SCRIPT_PATH:-}" ] || [ "$(basename "$SCRIPT_PATH")" != "$_want" ]; then
      SCRIPT_PATH="${SLURM_SUBMIT_DIR:-$REPO_ROOT}/scripts/$_want"
    fi
    [ -f "$SCRIPT_PATH" ] || SCRIPT_PATH="$SELF"
  fi
  # An explicit REPO wins: with a foreign batch script the path above cannot be
  # derived, and guessing silently is what made this fail invisibly.
  [ -n "$REPO_EXPLICIT" ] || REPO=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)

  mkdir -p "$SCRATCH" 2>/dev/null || true
  LOG=${LOG:-$SCRATCH/yambda_slurm.${SLURM_JOB_ID:-manual}.log}

  # Run shape: the full MLPerf run by default (all 299 windows, full train and
  # eval batches). SMOKE=1 selects the old quick smoke shape; any explicit value
  # still wins.
  MODE=${MODE:-streaming-train-eval}
  if [ "${SMOKE:-0}" = 1 ]; then
    START_TS=${START_TS:-150}; NUM_TRAIN_TS=${NUM_TRAIN_TS:-1}
    NUM_TRAIN_BATCHES=${NUM_TRAIN_BATCHES:-20}; NUM_EVAL_BATCHES=${NUM_EVAL_BATCHES:-10}
  fi
  START_TS=${START_TS:-0}
  NUM_TRAIN_TS=${NUM_TRAIN_TS:-299}
  NUM_TRAIN_BATCHES=${NUM_TRAIN_BATCHES:-0}
  NUM_EVAL_BATCHES=${NUM_EVAL_BATCHES:-0}
  # Default eval cadence: per-window OFF (0), data-fraction every 0.1% of data
  # (0.001, matching the RCP sweep). The two are mutually exclusive (both >0
  # raises a ValueError).
  EVAL_EVERY_N_WINDOWS=${EVAL_EVERY_N_WINDOWS:-0}
  EVAL_EVERY_DATA_PCT=${EVAL_EVERY_DATA_PCT:-0.001}
  # Console/TB train-metric cadence (ship-4 item 7, 2026-10-05). NOT defaulted
  # here: the gin file (yambda_5b.gin mlf/env_int) is the single source, default
  # 0 = training-side metric logging OFF: the MLPerf submission setting (no
  # per-step torchmetrics update, no train compute_and_log, so no "Step N
  # metrics/perf" or LR console lines; eval and every MLLOG event unchanged).
  # Each metric-log step drains the host-sync-free run-ahead queue
  # (torchmetrics gather: ~21 barriers + ~210 syncs, ~25-35 ms GPU idle at 1n):
  # cadence 5 cost ~4% of 1n train rate vs 50 (lab iter-12, job N), and
  # OFF is a further -0.48% ms/step vs 50 (job N). Logging only: no
  # training math changes. METRIC_LOG_FREQ is forwarded only when set and
  # non-empty (an explicit 0 passes). Opt-ins: 50 = coarse debugging metrics;
  # 20 = perf pairs (usage examples above); 5 = exact b97d9d9 cadence.
  # THROUGHPUT RULE: 200-step / ladder / rate perf runs set METRIC_LOG_FREQ
  # explicitly in every arm (same value in control and candidate).
  # MLPerf train_loss event cadence: likewise forwarded only when set; the gin
  # default (mltlf/env_int) is 0 = OFF (submission logging profile; MLPerf 6.1
  # does not require train_loss). Throughput runs set it explicitly
  # (MLPERF_TRAIN_LOSS_LOG_FREQ=1 for per-step MLLOG timing, or 20).
  FORCE_PROVISION=${FORCE_PROVISION:-0}

  # Truncate the metrics log on a FRESH run; APPEND on a supervised relaunch
  # (APPEND_LOG=1) so the full-run NE/AUC history survives crash/node-failover
  # resubmits instead of being wiped on every attempt (mirrors the single-node
  # supervisor's init-once/append model).
  if [ "${APPEND_LOG:-0}" = "1" ]; then
    echo "[$(date)] === resume: appending to existing $LOG (APPEND_LOG=1) ===" >> "$LOG"
  else
    : > "$LOG"
  fi
  echo "[$(date)] launch_slurm/orchestrate: job=${SLURM_JOB_ID:-?} nodes=${SLURM_JOB_NODELIST:-?} nnodes=${SLURM_NNODES:-1}" | tee -a "$LOG"
  echo "[$(date)] resolved SCRIPT_PATH=$SCRIPT_PATH REPO=$REPO" | tee -a "$LOG"

  # Persistent kernel compile cache, SEED + WRITEBACK design (ship-12 gin item
  # (34); KCACHE_DIR, default none = off). Triton keeps compiled kernels
  # under ~/.triton/cache in the CONTAINER's writable layer, so a freshly
  # provisioned container recompiles every Triton kernel inside the timed region:
  # +9.2..9.7 s in the first train window and +4.3..4.5 s at the bf16->FP8 switch
  # per run on the ship7 logs (eval/eval_cold_jit.py). With KCACHE_DIR set, the
  # worker copies $KCACHE_DIR/triton into the container-LOCAL cache before the
  # trainer starts (untimed; no dataset touched) and, after the trainer exits
  # (after RUN_STOP), publishes the entries this run compiled back with one atomic
  # directory rename each. Nothing reads or writes shared storage while the clock
  # runs -- the iter-12/13 variant pointed TRITON_CACHE_DIR straight at NFS and hit
  # root-squash EACCES on /home plus NFS writes inside the timed region. Triton
  # keys entries on kernel source hash + specialization/constexprs + target +
  # Triton build (+ cache-invalidating env), so a stale entry for what that key
  # covers is a miss. The key does NOT cover the ROCm runtime or the image: keep
  # one KCACHE_DIR per image/stack (name it after the image). MLPerf:
  # training_policies issue #399 (WG reading, non-binding): "If you can
  # pre-compile and then cache (such as to disk) and then load (such as from
  # disk) before the run, then this pre-compilation would not be limited in
  # time"; the L237 flush is about DATA (the page cache is still dropped below)
  # -- the cache holds compiled kernels only, no dataset content. A scored set
  # should seed a bundle populated by a NON-scored run (KCACHE_PUBLISH=1) and run
  # with the default KCACHE_PUBLISH=0; the seed logs the bundle's content digest.
  # KCACHE_FLYDSL=1 also seeds/publishes FlyDSL's disk cache (per file, temp +
  # rename; its in-process writes stay container-local).
  KCACHE_DIR=${KCACHE_DIR:-none}
  # KCACHE_PUBLISH=0 (default) seeds only: the shared cache stays read-only to
  # the run (a scored run whose bundle a dedicated non-scored run populated); the
  # bundle is then never created or chmod-ed here. KCACHE_PUBLISH=1 (populating
  # runs) = seed + writeback after RUN_STOP. Both knobs are normalized to 0/1
  # (spliced unquoted below); an unknown KCACHE_PUBLISH value fails closed to 0.
  case "${KCACHE_PUBLISH:-0}" in
    1|true|yes|on) KCACHE_PUB=1 ;;
    0|false|no|off) KCACHE_PUB=0 ;;
    *) KCACHE_PUB=0
       echo "[$(date)] [kcache] WARNING: KCACHE_PUBLISH=$KCACHE_PUBLISH is not 0/1/false/true/no/yes/off/on; publish off (read-only seed)" | tee -a "$LOG" ;;
  esac
  case "${KCACHE_FLYDSL:-0}" in 1) KCACHE_FLY=1 ;; *) KCACHE_FLY=0 ;; esac
  KCACHE_SRC=""
  if [ "$KCACHE_DIR" != "none" ]; then
    # KCACHE_SRC is spliced UNQUOTED into the srun/docker exec string below
    # (-e KCACHE_SRC=...), like the [fabric] values: only an absolute path of
    # [A-Za-z0-9_./+-] arms; anything else leaves the cache off (fail open --
    # this is an optimization, the run itself is unaffected).
    if [[ "$KCACHE_DIR" =~ ^/[A-Za-z0-9_./+-]+$ ]]; then
    case "$KCACHE_DIR" in
      "$REPO_MOUNT"/*|"$DATA_MOUNT"/*)
        if [ "$KCACHE_PUB" = "0" ]; then
          # Read-only seed: use an existing bundle as is (no mkdir, no chmod).
          [ -d "$KCACHE_DIR/triton" ] && KCACHE_SRC="$KCACHE_DIR"
        elif mkdir -p "$KCACHE_DIR/triton" 2>/dev/null; then
          # The container's root is squashed to an unprivileged uid on /home NFS:
          # publishing needs world-writable cache dirs (create + rename only).
          chmod a+rwx "$KCACHE_DIR/triton" 2>/dev/null || true
          if [ "$KCACHE_FLY" = "1" ] && mkdir -p "$KCACHE_DIR/flydsl" 2>/dev/null; then
            chmod a+rwx "$KCACHE_DIR/flydsl" 2>/dev/null || true
          fi
          KCACHE_SRC="$KCACHE_DIR"
        fi ;;
    esac
    fi
    if [ -n "$KCACHE_SRC" ]; then
      echo "[$(date)] [kcache] armed: seed+writeback $KCACHE_SRC publish=$KCACHE_PUB triton_entries=$(ls "$KCACHE_SRC/triton" 2>/dev/null | wc -l) flydsl=$KCACHE_FLY" | tee -a "$LOG"
    else
      echo "[$(date)] [kcache] WARNING: KCACHE_DIR=$KCACHE_DIR is not a usable dir under REPO_MOUNT/DATA_MOUNT (characters outside [A-Za-z0-9_./+-], not creatable, or with KCACHE_PUBLISH=0 no existing $KCACHE_DIR/triton); kernel cache stays container-local" | tee -a "$LOG"
    fi
  fi
  # SHIP-13 CANDIDATE (dlstart) TSPEC, Triton specialization table
  # (dlrm_v4/train/triton_spec.py; gin apply_env_bootstrap.TRITON_SPEC_REPLAY,
  # default "0" = off): TRITON_SPEC_RECORD=<dir> (non-scored record runs),
  # TRITON_SPEC_REPLAY=<merged table>|0, TRITON_SPEC_REPLAY_PHASES=post|all,
  # TRITON_SPEC_MONITOR=1. Forwarded into the container only when set (docker
  # exec block below, next to FP8_WARMUP). The values are spliced UNQUOTED into
  # the srun/docker command, so only [A-Za-z0-9_./=+-] passes; any other value
  # is dropped with a warning and that knob stays off (fail open: the run is
  # unaffected). Absolute paths must be visible in the container (under
  # REPO_MOUNT or DATA_MOUNT) and may not contain a ".." component (dropped); a
  # record dir there is created world-writable, because the container's root is
  # squashed on NFS (same reason as KCACHE above). Relative paths are resolved
  # inside the container and not checked here.
  for _ts_v in TRITON_SPEC_RECORD TRITON_SPEC_REPLAY TRITON_SPEC_REPLAY_PHASES TRITON_SPEC_MONITOR; do
    eval "_ts_val=\${$_ts_v:-}"
    [ -n "$_ts_val" ] || continue
    if [[ ! "$_ts_val" =~ ^[A-Za-z0-9_./=+-]+$ ]]; then
      echo "[$(date)] [triton-spec] WARNING: $_ts_v has characters outside [A-Za-z0-9_./=+-]; not forwarded ($_ts_v stays off)" | tee -a "$LOG"
      unset "$_ts_v"
    elif [[ "$_ts_v" != *_PHASES && "$_ts_v" != *_MONITOR && "/$_ts_val/" == */../* ]]; then
      echo "[$(date)] [triton-spec] WARNING: $_ts_v=$_ts_val contains a '..' component; not forwarded ($_ts_v stays off)" | tee -a "$LOG"
      unset "$_ts_v"
    else
      echo "[$(date)] [triton-spec] forwarded: $_ts_v=$_ts_val" | tee -a "$LOG"
    fi
  done
  for _ts_v in TRITON_SPEC_RECORD TRITON_SPEC_REPLAY; do
    eval "_ts_val=\${$_ts_v:-}"
    # off spellings as in triton_spec.py (any case): forwarded, nothing to check
    case "${_ts_val,,}" in ""|0|off|false|no|none) continue ;; esac
    case "$_ts_val" in
      /*) ;;
      *) # relative: triton_spec.py resolves a table against the container's CWD,
         # then recommendation/ (the form a table shipped in-tree uses); a record
         # dir against the CWD. Nothing to check on the host.
         echo "[$(date)] [triton-spec] $_ts_v=$_ts_val is relative: resolved inside the container (CWD, then recommendation/ for a table)" | tee -a "$LOG"
         continue ;;
    esac
    case "$_ts_val" in
      "$REPO_MOUNT"/*|"$DATA_MOUNT"/*) ;;
      *) echo "[$(date)] [triton-spec] WARNING: $_ts_v=$_ts_val is not under REPO_MOUNT/DATA_MOUNT; the container may not see it" | tee -a "$LOG"
         continue ;;
    esac
    if [ "$_ts_v" = "TRITON_SPEC_RECORD" ]; then
      { mkdir -p "$_ts_val" && chmod a+rwx "$_ts_val"; } 2>/dev/null \
        || echo "[$(date)] [triton-spec] WARNING: cannot create a world-writable $_ts_val; the trainer logs NOT armed if it cannot write there" | tee -a "$LOG"
    elif [ ! -r "$_ts_val" ]; then
      echo "[$(date)] [triton-spec] WARNING: TRITON_SPEC_REPLAY=$_ts_val is not readable here; the trainer logs NOT armed and runs without it" | tee -a "$LOG"
    fi
  done
  echo "[$(date)] config: MODE=$MODE START_TS=$START_TS NUM_TRAIN_TS=$NUM_TRAIN_TS NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES NUM_EVAL_BATCHES=$NUM_EVAL_BATCHES METRIC_LOG_FREQ=${METRIC_LOG_FREQ:-<unset: gin default 0>} MLPERF_TRAIN_LOSS_LOG_FREQ=${MLPERF_TRAIN_LOSS_LOG_FREQ:-<unset: gin default 0 = OFF>}" | tee -a "$LOG"

  # Rendezvous resolved on the HOST (the container image has no SLURM client).
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
  MASTER_ADDR=${MASTER_ADDR:-localhost}
  MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
  echo "[$(date)] rendezvous: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT" | tee -a "$LOG"

  # Optional NCCL/RCCL fabric overrides — forwarded into the container only when
  # set at submit time (docker exec does NOT inherit the srun task env). The
  # worker phase applies its own validated multi-node bnxt_re defaults when these
  # are unset. Common: NCCL_NET_TRANSPORT=socket (TCP fallback), NCCL_DEBUG=INFO.
  #
  # Beyond the fixed names, every EXPORTED variable with prefix NCCL_, RCCL_,
  # MSCCL_ or TORCH_NCCL_ (plus GPU_MAX_HW_QUEUES, HIP_FORCE_DEV_KERNARG and
  # HSA_ENABLE_IPC_MODE_LEGACY) is forwarded too. Before this, any RCCL knob
  # outside the fixed list (channels, ALGO/PROTO, BUFFSIZE, P2P chunk size,
  # ...) was dropped SILENTLY and the arm trained the default path. Rules:
  #   * The accumulator is FABRIC_ENV_ARGS -- deliberately outside the forwarded
  #     prefixes, so prefix enumeration cannot pick up the accumulator itself.
  #   * Each name is forwarded once. The four TORCH_NCCL_* names already passed
  #     explicitly in the docker exec block below are skipped here.
  #   * Values are spliced UNQUOTED into the double-quoted `srun ... bash -c "..."`
  #     string, so a space, ';', quote, '$', '*', '%' or backtick would break or
  #     rewrite the docker command. Every value must match [A-Za-z0-9_.,:/=^+-]+;
  #     anything else ABORTS the job (fail closed). One collective per
  #     NCCL_ALGO/NCCL_PROTO value, e.g. NCCL_ALGO=allreduce:ring.
  #   * Every forwarded name/value is logged as '[fabric] forwarded: NAME=VAL', and
  #     an empty set is logged as '[fabric] forwarded: <none>', so the control arm
  #     proves no ambient submit-shell variable leaked into the container.
  #   * Kill switch: LAUNCH_FABRIC_FWD=0 restores the legacy behavior exactly
  #     (fixed RCCL list only, no prefix scan, no value validation). The switch
  #     name is outside the forwarded prefixes and is read here on the submit
  #     side, so it needs no forwarding itself. HSA_ENABLE_IPC_MODE_LEGACY is
  #     on both fixed lists, so a caller's value (e.g. its kill switch 0)
  #     reaches the worker's multi-node default in either mode.
  FABRIC_ENV_ARGS=""
  if [ "${LAUNCH_FABRIC_FWD:-1}" = "0" ]; then
  echo "[$(date)] [fabric] LAUNCH_FABRIC_FWD=0: prefix forwarding disabled, legacy fixed list only" | tee -a "$LOG"
  for v in NCCL_NET_TRANSPORT NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX \
           NCCL_IB_TC NCCL_IB_TIMEOUT NCCL_IGNORE_CPU_AFFINITY RCCL_MSCCL_ENABLE NCCL_NET_GDR_LEVEL \
           NCCL_IB_PCI_RELAXED_ORDERING NCCL_IB_USE_INLINE NCCL_IB_QPS_PER_CONNECTION \
           NCCL_IB_ECE_ENABLE NCCL_DMABUF_ENABLE NCCL_GDRCOPY_ENABLE NCCL_GDR_FLUSH_DISABLE \
           NCCL_PXN_DISABLE NCCL_CHECKS_DISABLE NCCL_CROSS_NIC RDMA_OVERLAY \
           HSA_ENABLE_IPC_MODE_LEGACY; do
    eval "val=\${$v:-}"
    if [ -n "$val" ]; then FABRIC_ENV_ARGS="$FABRIC_ENV_ARGS -e $v=$val"; fi
  done
  else
  _fabric_names=" TORCH_NCCL_TRACE_BUFFER_SIZE TORCH_NCCL_DUMP_ON_TIMEOUT TORCH_NCCL_TRACE_CPP_STACK TORCH_NCCL_DEBUG_INFO_TEMP_FILE "
  _fabric_re='^[A-Za-z0-9_.,:/=^+-]+$'
  _fabric_n=0
  _fabric_add() {
    local _n=$1 _v
    case "$_fabric_names" in *" $_n "*) return 0 ;; esac
    eval "_v=\${$_n:-}"
    [ -n "$_v" ] || return 0
    if ! [[ "$_v" =~ $_fabric_re ]]; then
      echo "[$(date)] [fabric] FATAL: $_n has a value with characters outside [A-Za-z0-9_.,:/=^+-] -- it would be spliced unquoted into the srun/docker command. Aborting." | tee -a "$LOG"
      exit 1
    fi
    FABRIC_ENV_ARGS="$FABRIC_ENV_ARGS -e $_n=$_v"
    _fabric_names="$_fabric_names$_n "
    _fabric_n=$((_fabric_n + 1))
    echo "[$(date)] [fabric] forwarded: $_n=$_v" | tee -a "$LOG"
  }
  for v in NCCL_NET_TRANSPORT NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX \
           NCCL_IB_TC NCCL_IB_TIMEOUT NCCL_IGNORE_CPU_AFFINITY RCCL_MSCCL_ENABLE NCCL_NET_GDR_LEVEL \
           NCCL_IB_PCI_RELAXED_ORDERING NCCL_IB_USE_INLINE NCCL_IB_QPS_PER_CONNECTION \
           NCCL_IB_ECE_ENABLE NCCL_DMABUF_ENABLE NCCL_GDRCOPY_ENABLE NCCL_GDR_FLUSH_DISABLE \
           NCCL_PXN_DISABLE NCCL_CHECKS_DISABLE NCCL_CROSS_NIC RDMA_OVERLAY \
           GPU_MAX_HW_QUEUES HIP_FORCE_DEV_KERNARG HSA_ENABLE_IPC_MODE_LEGACY; do
    _fabric_add "$v"
  done
  for v in $(compgen -e); do
    case "$v" in
      NCCL_*|RCCL_*|MSCCL_*|TORCH_NCCL_*) _fabric_add "$v" ;;
    esac
  done
  [ "$_fabric_n" -gt 0 ] || echo "[$(date)] [fabric] forwarded: <none>" | tee -a "$LOG"
  fi

  # FP8 HSTU knobs. Forwarded by PREFIX rather than by name: the set grows with
  # the path (fusion stages, scale mode, calibration), and a knob this file has
  # not been taught about is worse than useless -- the run trains on the BF16
  # path while its log says the FP8 arm was requested. Same rule as the NCCL
  # block above: only what the caller actually set is passed through.
  FP8_ENV_ARGS=""
  for v in "${!DLRM_HSTU_FP8_@}"; do
    eval "val=\${$v:-}"
    if [ -n "$val" ]; then FP8_ENV_ARGS="$FP8_ENV_ARGS -e $v=$val"; fi
  done
  if [ -n "$FP8_ENV_ARGS" ]; then
    echo "[$(date)] FP8 knobs forwarded:$FP8_ENV_ARGS" | tee -a "$LOG"
  fi

  # TRICKY — variable expansion inside the `srun ... bash -c "..."` blocks below:
  # the string is double-quoted, so PLAIN $VAR expands NOW on the batch host (e.g.
  # $MASTER_ADDR, $CONTAINER, $SCRIPT_PATH — values computed above), while
  # BACKSLASH-escaped \$VAR is passed through literally and expands LATER on each
  # compute node inside the srun task (e.g. \$SLURM_NODEID, \$(hostname)) where the
  # per-node SLURM_* env actually lives. Mixing these up sends every rank the
  # wrong node id or breaks the docker exec — keep the \$ on per-node values.

  # --- node-health preflight: per-GPU VRAM capacity on every host -------------
  # A host whose GPUs report reduced HBM capacity is a memory-bandwidth straggler.
  # node reports 270,566,162,432 B/GPU against 309,220,868,096 on
  # healthy hosts. It runs HBM-bound kernels 14-24% slower on all 8 GPUs and was
  # the rank-26 straggler behind ~11 ms/step of 4-node collective wait (lab
  # rccl campaign iter 6, eval/eval_node_screen.py). Nothing in the job said so.
  # This logs '[nodehealth] <host> gpus=N vram_min=.. vram_max=..' per node and
  # flags any host whose smallest GPU is below NODE_HEALTH_VRAM_MIN_B
  # (integer bytes, default 300000000000), or that shows fewer than
  # NODE_HEALTH_MIN_GPUS (default 8) GPUs, and any node that did not report.
  # WARN-ONLY by default. It runs before staging and container setup and passes
  # nothing to the trainer. NODE_HEALTH_STRICT=1 aborts the job on a flagged
  # host instead.
  # Read-only host sysfs: no GPU work and no container involvement.
  # NODE_HEALTH_CHECK=0 skips it.
  if [ "${NODE_HEALTH_CHECK:-1}" = "1" ]; then
    _nh_min_b=${NODE_HEALTH_VRAM_MIN_B:-300000000000}
    _nh_min_g=${NODE_HEALTH_MIN_GPUS:-8}
    case "$_nh_min_b$_nh_min_g" in ""|*[!0-9]*)
      echo "[$(date)] [nodehealth] WARN: NODE_HEALTH_VRAM_MIN_B/NODE_HEALTH_MIN_GPUS must be integers -- using defaults" | tee -a "$LOG"
      _nh_min_b=300000000000; _nh_min_g=8 ;;
    esac
    _nh_out=$(timeout -k 10 90 srun --ntasks-per-node=1 bash -c '
      _mn=""; _mx=0; _n=0
      for _f in /sys/class/drm/card*/device/mem_info_vram_total; do
        [ -r "$_f" ] || continue
        _v=$(cat "$_f" 2>/dev/null) || continue
        case "$_v" in ""|*[!0-9]*) continue ;; esac
        _n=$((_n+1))
        [ -z "$_mn" ] || [ "$_v" -lt "$_mn" ] && _mn=$_v
        [ "$_v" -gt "$_mx" ] && _mx=$_v
      done
      echo "[nodehealth] $(hostname) gpus=$_n vram_min=${_mn:-0} vram_max=$_mx"
    ' 2>/dev/null || true)
    _nh_bad=""; _nh_seen=0
    while read -r _nh_tag _nh_host _nh_g _nh_lo _nh_hi; do
      [ "$_nh_tag" = "[nodehealth]" ] || continue
      echo "[$(date)] [nodehealth] $_nh_host $_nh_g $_nh_lo $_nh_hi" | tee -a "$LOG"
      _nh_seen=$((_nh_seen + 1))
      _nh_v=${_nh_lo#vram_min=}; _nh_n=${_nh_g#gpus=}
      case "$_nh_v$_nh_n" in ""|*[!0-9]*) _nh_bad="$_nh_bad $_nh_host(unparsed)"; continue ;; esac
      if [ "$_nh_n" -lt "$_nh_min_g" ]; then _nh_bad="$_nh_bad $_nh_host(gpus=$_nh_n)"
      elif [ "$_nh_v" -lt "$_nh_min_b" ]; then _nh_bad="$_nh_bad $_nh_host(vram=$_nh_v)"; fi
    done <<< "$_nh_out"
    if [ "$_nh_seen" -eq 0 ]; then
      echo "[$(date)] [nodehealth] WARN: preflight produced no output (srun/sysfs unavailable) -- not checked" | tee -a "$LOG"
    elif [ "$_nh_seen" -lt "${SLURM_NNODES:-1}" ] && [ -z "$_nh_bad" ]; then
      echo "[$(date)] [nodehealth] WARN: only $_nh_seen of ${SLURM_NNODES:-1} nodes reported -- partially checked" | tee -a "$LOG"
    elif [ -n "$_nh_bad" ]; then
      echo "[$(date)] [nodehealth] WARN: degraded host(s) (min ${_nh_min_g} GPUs, ${_nh_min_b} B/GPU):${_nh_bad} -- expect a memory-bandwidth straggler; exclude it for multi-node timing" | tee -a "$LOG"
      if [ "${NODE_HEALTH_STRICT:-0}" = "1" ]; then
        echo "[$(date)] [nodehealth] FATAL: NODE_HEALTH_STRICT=1 -- refusing to start on a degraded host" | tee -a "$LOG"
        exit 1
      fi
    else
      echo "[$(date)] [nodehealth] OK: $_nh_seen/${SLURM_NNODES:-1} hosts have >= ${_nh_min_g} GPUs at >= ${_nh_min_b} B/GPU" | tee -a "$LOG"
    fi
  fi

  # --- step 0: stage the dataset to node-local NVMe --------------------------
  # Must run BEFORE the container step: docker fixes bind mounts at `run`, so the
  # local path has to exist and be known here or the container cannot see it.
  #
  # ALL-OR-NOTHING across nodes. DLRM_DATA_PATH is one value handed to every
  # rank, so a partial success would point the ranks on an unstaged node at a
  # path that is empty there. srun's exit code is non-zero if ANY task failed,
  # which is the semantics wanted.
  #
  # A staging failure is FATAL, deliberately. Quietly reading from NFS instead
  # would produce a run that looks fine and measures differently -- the
  # boundary-step behaviour staging exists to provide would just be absent, with
  # nothing in the log saying so. That is the same class of silent substitution
  # as the unprovisioned-container bug above, and it gets the same treatment:
  # refuse to start. Before giving up the stager reclaims its own stale scratch,
  # and only as much of it as the shortfall needs, so "not enough space" means
  # the node is genuinely full of data that is not ours to delete (node was
  # found on 2026-08-25 with 48 GB free of 3.4 TB). A node with room is left
  # untouched. Read data over NFS by asking for it: STAGE_DATA=0.
  if [ "$STAGE_DATA" = "1" ] && [ -z "${DLRM_DATA_PATH:-}" ]; then
    _stager="$(dirname "$SCRIPT_PATH")/stage_data_local.sh"
    if [ ! -f "$_stager" ]; then
      echo "[$(date)] FATAL: STAGE_DATA=1 but $_stager is missing" | tee -a "$LOG"
      echo "[$(date)]   set STAGE_DATA=0 to read from $DATA_MOUNT over NFS instead" | tee -a "$LOG"
      exit 1
    fi
    echo "[$(date)] staging dataset to $STAGE_DEST on all nodes (STAGE_DATA=1)" | tee -a "$LOG"
    # gin item (17b): the staging mode comes from the gin the trainer parses
    # ($REPO's yambda_5b.gin), never from an inherited STAGE_GIN.
    _stage_gin="$REPO/generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin"
    echo "[$(date)] [stage] item (17b) mode from STAGE_GIN=$_stage_gin (env DLRM_HSTU_FP8_STAGE_REF_ONLY=${DLRM_HSTU_FP8_STAGE_REF_ONLY:-<unset>} wins if set); stager $_stager" | tee -a "$LOG"
    _stage_l0=$(wc -l < "$LOG")
    # `pipefail` (set at the top of this script) is what makes srun's failure
    # visible through the tee — without it this would only ever see tee's rc.
    if ! STAGE_GIN="$_stage_gin" srun --ntasks-per-node=1 bash "$_stager" "$STAGE_DEST" \
           ${HISTORY_LENGTH:+"$HISTORY_LENGTH"} 2>&1 | tee -a "$LOG"; then
      echo "[$(date)] FATAL: dataset staging failed on at least one node — refusing to start" | tee -a "$LOG"
      echo "[$(date)]   see the [stage] lines above for which node and why" | tee -a "$LOG"
      echo "[$(date)]   to run on NFS instead, ask for it explicitly: STAGE_DATA=0" | tee -a "$LOG"
      exit 1
    fi
    LOCAL_DATA_MOUNT=$STAGE_DEST
    DLRM_DATA_PATH=$STAGE_DEST
    echo "[$(date)] staged OK on all nodes — reading data from $DLRM_DATA_PATH" | tee -a "$LOG"
    _stage_n() { tail -n +"$((_stage_l0 + 1))" "$LOG" | grep -c "^\[stage\] $1"; }
    echo "[$(date)] [stage] item (17b) applied by this launch on ${SLURM_NNODES:-1} node(s): reference-only ON $(_stage_n 'reference-only staging ON'), end state verified $(_stage_n 'data dir is reference-only'), OFF $(_stage_n 'reference-only staging OFF')" | tee -a "$LOG"
  elif [ -n "${DLRM_DATA_PATH:-}" ]; then
    echo "[$(date)] DLRM_DATA_PATH set by caller ($DLRM_DATA_PATH) — not staging" | tee -a "$LOG"
    echo "[$(date)] [stage] item (17b) NOT applied by this launch: the data directory is the caller's (the trainer's 'env bootstrap: DLRM_HSTU_FP8_STAGE_REF_ONLY=' line is the gin/env value, not a staging record)" | tee -a "$LOG"
  else
    echo "[$(date)] STAGE_DATA=$STAGE_DATA — not staging; [stage] item (17b) NOT applied by this launch" | tee -a "$LOG"
  fi

  # --- step 1: ensure the container is up on every node ----------------------
  echo "[$(date)] ensuring container '$CONTAINER' on all nodes (force=$FORCE_PROVISION)" | tee -a "$LOG"
  srun --ntasks-per-node=1 bash -c "
    # Reap stale/foreign GPU containers from prior jobs BEFORE (re)provisioning.
    # The node is allocated --exclusive, so any GPU container other than
    # '$CONTAINER' is an orphan left by a previous job (its container outlives the
    # SLURM allocation). We remove every such container that has GPU access
    # (/dev/kfd or /dev/dri) — running OR stopped, whether or not it currently
    # pins VRAM ('docker ps -aq' includes stopped ones) — since idle orphans can
    # still hold device handles or wake up; leaked HBM from these has caused both
    # OOMs and RCCL collective hangs. We deliberately SKIP non-GPU containers
    # (e.g. 'k8s-node-services-*' and other cluster system services) so we don't
    # disrupt node infrastructure. docker teardown lets the driver reclaim HBM.
    for _c in \$(docker ps -aq 2>/dev/null); do
      _nm=\$(docker inspect -f '{{.Name}}' \"\$_c\" 2>/dev/null | sed 's#^/##')
      [ \"\$_nm\" = \"$CONTAINER\" ] && continue
      _dev=\$(docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' \"\$_c\" 2>/dev/null)
      case \"\$_dev\" in
        *kfd*|*dri*)
          echo \"[\$(hostname)] reaping stale GPU container \$_nm (\$_c)\"
          docker rm -f \"\$_c\" >/dev/null 2>&1 || true ;;
        *)
          echo \"[\$(hostname)] keeping non-GPU/system container \$_nm (\$_c)\" ;;
      esac
    done
    # Reuse a STOPPED '$CONTAINER' (its installed deps persist in the container
    # fs) instead of destructively re-provisioning from the base image + pip.
    # Harmless no-op on a fresh node (no such container) -> falls through to
    # provision below. Repo code is bind-mounted, so live edits are still picked up.
    docker start $CONTAINER >/dev/null 2>&1 || true
    # A live container is NOT evidence of a provisioned one -- see PROVISION_STAMP.
    # Reuse requires the stamp to match the pins this run is defined as.
    _alive=0; docker exec $CONTAINER true >/dev/null 2>&1 && _alive=1
    _stamp=\$(docker exec $CONTAINER cat $PROVISION_MARKER 2>/dev/null || true)
    # A container created before staging existed has no bind mount for the local
    # copy, so DLRM_DATA_PATH would name a path that is simply absent inside it.
    # Mounts are fixed at \`docker run\`, so the only fix is to recreate it.
    _mountok=1
    if [ -n \"${LOCAL_DATA_MOUNT:-}\" ] && [ \"\$_alive\" = 1 ]; then
      case \" \$(docker inspect -f '{{range .Mounts}}{{.Source}} {{end}}' $CONTAINER 2>/dev/null) \" in
        *\" ${LOCAL_DATA_MOUNT:-} \"*) _mountok=1 ;;
        *) _mountok=0
           echo \"[\$(hostname)] container lacks the ${LOCAL_DATA_MOUNT:-} mount — re-provisioning to add it\" ;;
      esac
    fi
    if [ \"\$_alive\" = 1 ] && [ \"\$_stamp\" != \"$PROVISION_STAMP\" ]; then
      if [ -z \"\$_stamp\" ]; then
        echo \"[\$(hostname)] container is live but UNPROVISIONED (no $PROVISION_MARKER) — re-provisioning\"
      else
        echo \"[\$(hostname)] container provisioned for a different pin set — re-provisioning\"
        echo \"[\$(hostname)]   have: \$_stamp\"
        echo \"[\$(hostname)]   want: $PROVISION_STAMP\"
      fi
    fi
    if [ \"$FORCE_PROVISION\" = \"1\" ] || [ \"\$_alive\" != 1 ] \
       || [ \"\$_stamp\" != \"$PROVISION_STAMP\" ] || [ \"\$_mountok\" != 1 ]; then
      echo \"[\$(hostname)] (re)provisioning container\"
      LAUNCH_SLURM_PHASE=provision CONTAINER=$CONTAINER IMAGE=$IMAGE \
        BAKED_IMAGE=$BAKED_IMAGE BAKED_TAR=$BAKED_TAR USE_BAKED=$USE_BAKED \
        BAKE_IMAGE=${BAKE_IMAGE:-0} RDMA_OVERLAY=$OVERLAY REPO=$REPO \
        REPO_MOUNT=$REPO_MOUNT DATA_MOUNT=$DATA_MOUNT SCRATCH=$SCRATCH \
        TORCH_PIN=$TORCH_PIN TORCHVISION_PIN=$TORCHVISION_PIN \
        TORCHAUDIO_PIN=$TORCHAUDIO_PIN FBGEMM_PIN=$FBGEMM_PIN \
        TORCHREC_PIN=$TORCHREC_PIN PROVISION_MARKER=$PROVISION_MARKER \
        LOCAL_DATA_MOUNT=${LOCAL_DATA_MOUNT:-} bash $SCRIPT_PATH
    else
      # Container persists across jobs; the reap above only removes FOREIGN GPU
      # containers, so our own '$CONTAINER' can still pin HBM via stray trainer
      # ranks left by a prior OOM/crash (this caused repeated 'CUDA out of memory'
      # on relaunch onto the same node). Restart it to kill every exec'd proc and
      # let the driver reclaim HBM — cheap (keeps the installed deps in the
      # container fs; NFS RDMA overlay also persists), no full re-provision.
      echo \"[\$(hostname)] container already up — restarting to free any leaked HBM before launch\"
      docker restart $CONTAINER >/dev/null 2>&1 || true
      # Readiness gate: a bare 'docker exec true' can pass while the runtime is
      # still settling, so the SUBSEQUENT (heavier) worker exec races the restart
      # and dies with 'container is not running' / OCI 'setns' errors (observed on
      # two nodes -> the peer never joins rendezvous -> master 600s
      # TCPStore timeout). Require State.Running=true AND a successful probe, then
      # a short settle, before considering the container ready.
      for _w in \$(seq 1 30); do
        [ \"\$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)\" = \"true\" ] \
          && docker exec $CONTAINER true >/dev/null 2>&1 && break
        sleep 2
      done
      sleep 2
      echo \"[\$(hostname)] container restarted (HBM reclaimed; running=\$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null))\"
    fi
  " 2>&1 | tee -a "$LOG"

  # --- step 2: launch the worker (trainer) inside the container on every node -
  echo "[$(date)] launching trainer (worker phase) on all nodes" | tee -a "$LOG"
  srun --ntasks-per-node=1 bash -c "
    # Pre-flight readiness gate (per node): step 1 ran in a SEPARATE srun, so the
    # container can still be settling here. Wait for State.Running=true + a probe
    # before the worker exec so we don't race a just-restarted container.
    for _w in \$(seq 1 30); do
      [ \"\$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)\" = \"true\" ] \
        && docker exec $CONTAINER true >/dev/null 2>&1 && break
      [ \$_w -eq 1 ] && echo \"[\$(hostname)] worker pre-flight: waiting for container to be ready...\"
      sleep 2
    done
    # MLPerf Training rules (Data State at Start of Run): 'You must flush the
    # cache or restart the system prior to benchmarking ... This explicitly
    # excludes RAM.' Drop the host page cache on every node right before the
    # trainer starts (as the NVIDIA/AMD reference launchers do); container
    # restarts do not touch it, and the staged dataset fits in host RAM. The
    # trainer logs cache_clear with the outcome (MLPERF_CACHE_CLEARED).
    # CLEAR_CACHES=0 skips it (development runs only); a failed flush aborts.
    _cc=0
    if [ \"${CLEAR_CACHES:-1}\" = 1 ]; then
      _cb=\$(awk '/^Cached:/{print \$2}' /proc/meminfo)
      if sync && sudo -n /sbin/sysctl -q vm.drop_caches=3; then
        _cc=1; echo \"[\$(hostname)] page cache flushed (CLEAR_CACHES=1): Cached \${_cb} kB -> \$(awk '/^Cached:/{print \$2}' /proc/meminfo) kB\"
      else
        echo \"[\$(hostname)] FATAL: CLEAR_CACHES=1 but the page-cache flush failed\"; exit 97
      fi
    fi
    # Retry wrapper: docker exec startup failures (rc 125 daemon 'container is not
    # running', 126/127 OCI/setns 'exec failed') mean the container wasn't ready,
    # NOT that the trainer ran and failed. Restart + re-gate + retry a few times.
    # Any OTHER rc (the trainer actually started and exited) is propagated so the
    # supervisor's resume-from-checkpoint logic owns real failures.
    _wattempt=0
    while : ; do
    _wattempt=\$((_wattempt+1))
    docker exec \
      -e LAUNCH_SLURM_PHASE=worker \
      -e SCRATCH=$SCRATCH \
      -e SLURM_NNODES=\$SLURM_NNODES \
      -e SLURM_NODEID=\$SLURM_NODEID \
      -e SLURM_PROCID=\$SLURM_PROCID \
      -e SLURM_JOB_NODELIST=\"\$SLURM_JOB_NODELIST\" \
      -e SLURM_JOB_ID=\$SLURM_JOB_ID \
      -e MASTER_ADDR=$MASTER_ADDR \
      -e MASTER_PORT=$MASTER_PORT \
      -e HSTU_HAMMER_KERNEL=${HSTU_HAMMER_KERNEL:-TRITON} \
      ${HSTU_ATTN_BWD:+-e HSTU_ATTN_BWD=$HSTU_ATTN_BWD} \
      ${HSTU_ATTN_BWD_CHECK:+-e HSTU_ATTN_BWD_CHECK=$HSTU_ATTN_BWD_CHECK} \
      ${HSTU_ATTN_FWD:+-e HSTU_ATTN_FWD=$HSTU_ATTN_FWD} \
      ${HSTU_ATTN_FWD_CHECK:+-e HSTU_ATTN_FWD_CHECK=$HSTU_ATTN_FWD_CHECK} \
      ${HSTU_ATTN_FWD_DUMP_LENS:+-e HSTU_ATTN_FWD_DUMP_LENS=$HSTU_ATTN_FWD_DUMP_LENS} \
      ${HSTU_DW_GEMM:+-e HSTU_DW_GEMM=$HSTU_DW_GEMM} \
      ${TRITON_FULL_AUTOTUNE:+-e TRITON_FULL_AUTOTUNE=$TRITON_FULL_AUTOTUNE} \
      ${TRITON_HSTU_FAST_INTERIOR:+-e TRITON_HSTU_FAST_INTERIOR=$TRITON_HSTU_FAST_INTERIOR} \
      ${TRITON_HSTU_LAST_LAYER_TARGETS_ONLY:+-e TRITON_HSTU_LAST_LAYER_TARGETS_ONLY=$TRITON_HSTU_LAST_LAYER_TARGETS_ONLY} \
      ${TRITON_HSTU_TARGETS_ONLY_EVAL:+-e TRITON_HSTU_TARGETS_ONLY_EVAL=$TRITON_HSTU_TARGETS_ONLY_EVAL} \
      ${GR_PIN_BF16_BLAS_SOLUTIONS:+-e GR_PIN_BF16_BLAS_SOLUTIONS=$GR_PIN_BF16_BLAS_SOLUTIONS} \
      ${GR_PIN_BF16_BLAS_DEBUG:+-e GR_PIN_BF16_BLAS_DEBUG=$GR_PIN_BF16_BLAS_DEBUG} \
      ${TRANSDUCER_BF16_DTYPE:+-e TRANSDUCER_BF16_DTYPE=$TRANSDUCER_BF16_DTYPE} \
      ${PYTORCH_TUNABLEOP_ENABLED:+-e PYTORCH_TUNABLEOP_ENABLED=$PYTORCH_TUNABLEOP_ENABLED} \
      ${PYTORCH_TUNABLEOP_TUNING:+-e PYTORCH_TUNABLEOP_TUNING=$PYTORCH_TUNABLEOP_TUNING} \
      -e MODE=$MODE \
      -e START_TS=$START_TS \
      -e NUM_TRAIN_TS=$NUM_TRAIN_TS \
      -e EVAL_EVERY_N_WINDOWS=$EVAL_EVERY_N_WINDOWS \
      ${EVAL_EVERY_DATA_PCT:+-e EVAL_EVERY_DATA_PCT=$EVAL_EVERY_DATA_PCT} \
      -e NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES \
      -e NUM_EVAL_BATCHES=$NUM_EVAL_BATCHES \
      ${DIE_AT_STEP:+-e DIE_AT_STEP=$DIE_AT_STEP} \
      ${METRIC_LOG_FREQ:+-e METRIC_LOG_FREQ=$METRIC_LOG_FREQ} \
      ${EVAL_INTERIM_LOG:+-e EVAL_INTERIM_LOG=$EVAL_INTERIM_LOG} \
      ${SKIP_EVAL_EPOCH_PCT:+-e SKIP_EVAL_EPOCH_PCT=$SKIP_EVAL_EPOCH_PCT} \
      ${MLPERF_LOGGING:+-e MLPERF_LOGGING=$MLPERF_LOGGING} \
      ${MLPERF_TRAIN_LOSS_LOG_FREQ:+-e MLPERF_TRAIN_LOSS_LOG_FREQ=$MLPERF_TRAIN_LOSS_LOG_FREQ} \
      ${STREAMING_SHUFFLE_FRACTION:+-e STREAMING_SHUFFLE_FRACTION=$STREAMING_SHUFFLE_FRACTION} \
      ${STREAMING_SHUFFLE_SEED:+-e STREAMING_SHUFFLE_SEED=$STREAMING_SHUFFLE_SEED} \
      ${NUM_WORKERS:+-e NUM_WORKERS=$NUM_WORKERS} \
      ${EVAL_NUM_WORKERS:+-e EVAL_NUM_WORKERS=$EVAL_NUM_WORKERS} \
      ${PREFETCH_FACTOR:+-e PREFETCH_FACTOR=$PREFETCH_FACTOR} \
      ${DL_PIN_MEMORY:+-e DL_PIN_MEMORY=$DL_PIN_MEMORY} \
      ${DL_PIN_FREE_SHM:+-e DL_PIN_FREE_SHM=$DL_PIN_FREE_SHM} \
      ${DL_PRESTART:+-e DL_PRESTART=$DL_PRESTART} \
      ${DL_PRESTART_RANK_TIMELINE:+-e DL_PRESTART_RANK_TIMELINE=$DL_PRESTART_RANK_TIMELINE} \
      ${DL_FAST_BATCH:+-e DL_FAST_BATCH=$DL_FAST_BATCH} \
      ${DL_FAST_BATCH_EVAL:+-e DL_FAST_BATCH_EVAL=$DL_FAST_BATCH_EVAL} \
      ${DL_FAST_BATCH_ONESTORE:+-e DL_FAST_BATCH_ONESTORE=$DL_FAST_BATCH_ONESTORE} \
      ${DL_COLD_PREFETCH:+-e DL_COLD_PREFETCH=$DL_COLD_PREFETCH} \
      ${DL_COLD_PREFETCH_STREAM:+-e DL_COLD_PREFETCH_STREAM=$DL_COLD_PREFETCH_STREAM} \
      ${DL_COLD_PREFETCH_LEAD:+-e DL_COLD_PREFETCH_LEAD=$DL_COLD_PREFETCH_LEAD} \
      ${DL_COLD_PREFETCH_HORIZON:+-e DL_COLD_PREFETCH_HORIZON=$DL_COLD_PREFETCH_HORIZON} \
      ${DL_COLD_PREFETCH_GROUP:+-e DL_COLD_PREFETCH_GROUP=$DL_COLD_PREFETCH_GROUP} \
      ${DL_COLD_PREFETCH_BATCHES:+-e DL_COLD_PREFETCH_BATCHES=$DL_COLD_PREFETCH_BATCHES} \
      ${DL_COLD_PREFETCH_THREADS:+-e DL_COLD_PREFETCH_THREADS=$DL_COLD_PREFETCH_THREADS} \
      ${DL_COLD_PREFETCH_EVAL:+-e DL_COLD_PREFETCH_EVAL=$DL_COLD_PREFETCH_EVAL} \
      ${DL_COLD_PREFETCH_EVAL_LEAD:+-e DL_COLD_PREFETCH_EVAL_LEAD=$DL_COLD_PREFETCH_EVAL_LEAD} \
      ${DL_COLD_PREFETCH_EVAL_GAP_S:+-e DL_COLD_PREFETCH_EVAL_GAP_S=$DL_COLD_PREFETCH_EVAL_GAP_S} \
      ${DL_SINGLE_POOL:+-e DL_SINGLE_POOL=$DL_SINGLE_POOL} \
      ${DL_WBND_DIAG:+-e DL_WBND_DIAG=$DL_WBND_DIAG} \
      ${DLQ_LOG_EVERY:+-e DLQ_LOG_EVERY=$DLQ_LOG_EVERY} \
      ${BLAS_WARMUP:+-e BLAS_WARMUP=$BLAS_WARMUP} \
      ${BLAS_WARMUP_M:+-e BLAS_WARMUP_M=$BLAS_WARMUP_M} \
      ${DL_WINDOW_INDEX:+-e DL_WINDOW_INDEX=$DL_WINDOW_INDEX} \
      ${TRAIN_PIPELINE:+-e TRAIN_PIPELINE=$TRAIN_PIPELINE} \
      ${TP_DEFER_INPUT_DIST:+-e TP_DEFER_INPUT_DIST=$TP_DEFER_INPUT_DIST} \
      ${TP_WARM_WINDOWS:+-e TP_WARM_WINDOWS=$TP_WARM_WINDOWS} \
      ${TP_EVAL_PIPELINE:+-e TP_EVAL_PIPELINE=$TP_EVAL_PIPELINE} \
      ${FP8_WARMUP:+-e FP8_WARMUP=$FP8_WARMUP} \
      ${FP8_WARMUP_STORES:+-e FP8_WARMUP_STORES=$FP8_WARMUP_STORES} \
      ${FP8_WARMUP_VERIFY:+-e FP8_WARMUP_VERIFY=$FP8_WARMUP_VERIFY} \
      ${FP8_WARMUP_SKIP:+-e FP8_WARMUP_SKIP=$FP8_WARMUP_SKIP} \
      ${FP8_WARMUP_QUIET:+-e FP8_WARMUP_QUIET=$FP8_WARMUP_QUIET} \
      ${FP8_WARMUP_N11:+-e FP8_WARMUP_N11=$FP8_WARMUP_N11} \
      ${TRITON_SPEC_RECORD:+-e TRITON_SPEC_RECORD=$TRITON_SPEC_RECORD} \
      ${TRITON_SPEC_REPLAY:+-e TRITON_SPEC_REPLAY=$TRITON_SPEC_REPLAY} \
      ${TRITON_SPEC_REPLAY_PHASES:+-e TRITON_SPEC_REPLAY_PHASES=$TRITON_SPEC_REPLAY_PHASES} \
      ${TRITON_SPEC_MONITOR:+-e TRITON_SPEC_MONITOR=$TRITON_SPEC_MONITOR} \
      ${DLRM_DATA_PATH:+-e DLRM_DATA_PATH=$DLRM_DATA_PATH} \
      ${DIAG_UNIQUE_EMB:+-e DIAG_UNIQUE_EMB=$DIAG_UNIQUE_EMB} \
      ${DIAG_EMB_STEPS:+-e DIAG_EMB_STEPS=$DIAG_EMB_STEPS} \
      ${OMP_NUM_THREADS:+-e OMP_NUM_THREADS=$OMP_NUM_THREADS} \
      ${MKL_NUM_THREADS:+-e MKL_NUM_THREADS=$MKL_NUM_THREADS} \
      ${OMP_WAIT_POLICY:+-e OMP_WAIT_POLICY=$OMP_WAIT_POLICY} \
      ${OMP_PROC_BIND:+-e OMP_PROC_BIND=$OMP_PROC_BIND} \
      ${OUTPUT_TRACE:+-e OUTPUT_TRACE=$OUTPUT_TRACE} \
      ${TRACE_DIR:+-e TRACE_DIR=$TRACE_DIR} \
      ${LIB_OVERLAY:+-e LIB_OVERLAY=$LIB_OVERLAY} \
      ${TORCH_PIN:+-e TORCH_PIN=$TORCH_PIN} \
      ${TORCHVISION_PIN:+-e TORCHVISION_PIN=$TORCHVISION_PIN} \
      ${TORCHAUDIO_PIN:+-e TORCHAUDIO_PIN=$TORCHAUDIO_PIN} \
      ${FBGEMM_PIN:+-e FBGEMM_PIN=$FBGEMM_PIN} \
      ${TORCHREC_PIN:+-e TORCHREC_PIN=$TORCHREC_PIN} \
      ${MIN_HISTORY:+-e MIN_HISTORY=$MIN_HISTORY} \
      ${SEED:+-e SEED=$SEED} \
      ${DENSE_LR:+-e DENSE_LR=$DENSE_LR} \
      ${SPARSE_LR:+-e SPARSE_LR=$SPARSE_LR} \
      ${GRAD_CLIP_NORM:+-e GRAD_CLIP_NORM=$GRAD_CLIP_NORM} \
      ${LR_WARMUP_STEPS:+-e LR_WARMUP_STEPS=$LR_WARMUP_STEPS} \
      ${LR_WARMUP_START_LR:+-e LR_WARMUP_START_LR=$LR_WARMUP_START_LR} \
      ${HSTU_NUM_LAYERS:+-e HSTU_NUM_LAYERS=$HSTU_NUM_LAYERS} \
      ${MAX_SEQ_LEN:+-e MAX_SEQ_LEN=$MAX_SEQ_LEN} \
      ${HISTORY_LENGTH:+-e HISTORY_LENGTH=$HISTORY_LENGTH} \
      ${BATCH_SIZE:+-e BATCH_SIZE=$BATCH_SIZE} \
      ${CKPT_TIME_INTERVAL_S:+-e CKPT_TIME_INTERVAL_S=$CKPT_TIME_INTERVAL_S} \
      ${KEEP_LAST_N:+-e KEEP_LAST_N=$KEEP_LAST_N} \
      ${IN_WINDOW_CKPT_FREQ:+-e IN_WINDOW_CKPT_FREQ=$IN_WINDOW_CKPT_FREQ} \
      ${CKPT_STEP_FREQ:+-e CKPT_STEP_FREQ=$CKPT_STEP_FREQ} \
      -e TRAIN_SPLIT_PERCENTAGE=${TRAIN_SPLIT_PERCENTAGE:-1.0} \
      -e AUC_THRESHOLD=${AUC_THRESHOLD:-0.75} \
      ${MLPERF_SUBMISSION_PLATFORM:+-e MLPERF_SUBMISSION_PLATFORM=$MLPERF_SUBMISSION_PLATFORM} \
      -e SPLIT_SALT=${SPLIT_SALT:-0} \
      -e EVAL_HOLDOUT_TS=${EVAL_HOLDOUT_TS:--1} \
      -e EVAL_HOLDOUT_NUM_WINDOWS=${EVAL_HOLDOUT_NUM_WINDOWS:-1} \
      ${WORKER_CMD:+-e WORKER_CMD=\"$WORKER_CMD\"} \
      ${RUN_NAME:+-e RUN_NAME=$RUN_NAME} \
      ${TENSORBOARD_LOG_PATH:+-e TENSORBOARD_LOG_PATH=$TENSORBOARD_LOG_PATH} \
      ${MLPERF_LOG_PATH:+-e MLPERF_LOG_PATH=$MLPERF_LOG_PATH} \
      -e MLPERF_CACHE_CLEARED=\$_cc \
      ${KCACHE_SRC:+-e KCACHE_SRC=$KCACHE_SRC} \
      ${KCACHE_SRC:+-e KCACHE_FLYDSL=$KCACHE_FLY} \
      ${KCACHE_SRC:+-e KCACHE_PUBLISH=$KCACHE_PUB} \
      ${CKPT_PATH:+-e CKPT_PATH=$CKPT_PATH} \
      ${SPARSE_A2A_FWD:+-e SPARSE_A2A_FWD=$SPARSE_A2A_FWD} \
      ${SPARSE_A2A_BWD:+-e SPARSE_A2A_BWD=$SPARSE_A2A_BWD} \
      ${QCOMM_LOWMEM_CODEC:+-e QCOMM_LOWMEM_CODEC=$QCOMM_LOWMEM_CODEC} \
      ${EC_INDEX_DEDUP:+-e EC_INDEX_DEDUP=$EC_INDEX_DEDUP} \
      ${EC_GATHER_SELECT:+-e EC_GATHER_SELECT=$EC_GATHER_SELECT} \
      ${EC_BF16_OUTPUT:+-e EC_BF16_OUTPUT=$EC_BF16_OUTPUT} \
      ${DLRMV4_EC_BF16_MIN_ROWS:+-e DLRMV4_EC_BF16_MIN_ROWS=$DLRMV4_EC_BF16_MIN_ROWS} \
      ${DLRMV4_EC_BF16_BWD_CHUNK:+-e DLRMV4_EC_BF16_BWD_CHUNK=$DLRMV4_EC_BF16_BWD_CHUNK} \
      ${EC_SEGMENT_REDUCE:+-e EC_SEGMENT_REDUCE=$EC_SEGMENT_REDUCE} \
      ${FUSE_LINEAR_NORM:+-e FUSE_LINEAR_NORM=$FUSE_LINEAR_NORM} \
      ${FUSE_LN_RESIDUAL:+-e FUSE_LN_RESIDUAL=$FUSE_LN_RESIDUAL} \
      ${EMB_SHARDING_OVERRIDES:+-e EMB_SHARDING_OVERRIDES=$EMB_SHARDING_OVERRIDES} \
      ${EMB_PLACEMENT_OVERRIDES:+-e EMB_PLACEMENT_OVERRIDES=$EMB_PLACEMENT_OVERRIDES} \
      ${EMB_PLACEMENT:+-e EMB_PLACEMENT=$EMB_PLACEMENT} \
      ${DL_TEARDOWN_HARD_EXIT:+-e DL_TEARDOWN_HARD_EXIT=$DL_TEARDOWN_HARD_EXIT} \
      ${PG_TIMEOUT_S:+-e PG_TIMEOUT_S=$PG_TIMEOUT_S} \
      ${DL_TEARDOWN_CLOSE_POOLS:+-e DL_TEARDOWN_CLOSE_POOLS=$DL_TEARDOWN_CLOSE_POOLS} \
      ${PYTHONFAULTHANDLER:+-e PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER} \
      ${TORCH_NCCL_TRACE_BUFFER_SIZE:+-e TORCH_NCCL_TRACE_BUFFER_SIZE=$TORCH_NCCL_TRACE_BUFFER_SIZE} \
      ${TORCH_NCCL_DUMP_ON_TIMEOUT:+-e TORCH_NCCL_DUMP_ON_TIMEOUT=$TORCH_NCCL_DUMP_ON_TIMEOUT} \
      ${TORCH_NCCL_TRACE_CPP_STACK:+-e TORCH_NCCL_TRACE_CPP_STACK=$TORCH_NCCL_TRACE_CPP_STACK} \
      ${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:+-e TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$TORCH_NCCL_DEBUG_INFO_TEMP_FILE} \
      -e LOG=$LOG \
      $FABRIC_ENV_ARGS \
      $FP8_ENV_ARGS \
      $CONTAINER bash -lc 'umask 000 && cd $REPO && LAUNCH_SLURM_PHASE=worker bash scripts/launch_slurm.sh'
    _wrc=\$?
    if { [ \$_wrc -eq 125 ] || [ \$_wrc -eq 126 ] || [ \$_wrc -eq 127 ]; } && [ \$_wattempt -lt 5 ]; then
      echo \"[\$(hostname)] worker exec failed to START (rc=\$_wrc, attempt \$_wattempt/5) — container not ready; restarting + retrying\"
      docker restart $CONTAINER >/dev/null 2>&1 || true
      for _w in \$(seq 1 30); do
        [ \"\$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)\" = \"true\" ] \
          && docker exec $CONTAINER true >/dev/null 2>&1 && break
        sleep 2
      done
      sleep 3
      continue
    fi
    exit \$_wrc
    done
  " 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "[$(date)] launch_slurm/orchestrate finished rc=$rc" | tee -a "$LOG"
  exit $rc
}

# =============================================================================
# PHASE: provision  (compute-node host)
# =============================================================================
provision() {
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
  DOCKER=$(command -v docker 2>/dev/null || true); DOCKER=${DOCKER:-/usr/bin/docker}
  FBGEMM_WHL=${FBGEMM_WHL:-${DATA_ROOT:-/data}/FBGEMM/fbgemm_gpu/dist/fbgemm_gpu_nightly_rocm-2026.6.2-cp312-cp312-linux_x86_64.whl}  # [CLUSTER-SPECIFIC] gfx950/ROCm wheel
  TORCH_IDX=${TORCH_IDX:-https://download.pytorch.org/whl/rocm7.2}  # [CLUSTER-SPECIFIC] ROCm version index
  echo "[provision] host=$(hostname) container=$CONTAINER docker=$DOCKER"

  # Resolve which image to run + whether deps must be installed. Prefer a pre-baked
  # image (deps already installed) to skip the multi-GB torch download + pip /
  # torchrec-from-git build on every fresh node:
  #   1) baked image in this node's docker -> use it, skip deps
  #   2) baked image tar on NFS            -> docker load (local, no internet)
  #   3) neither                           -> base image + pip (slow path, which
  #                                           can then be baked via BAKE_IMAGE=1)
  NEED_DEPS=1
  RUN_IMAGE="$IMAGE"
  if [ "$USE_BAKED" = "1" ]; then
    if "$DOCKER" image inspect "$BAKED_IMAGE" >/dev/null 2>&1; then
      echo "[provision] using baked image $BAKED_IMAGE (deps preinstalled, no download)"
      RUN_IMAGE="$BAKED_IMAGE"; NEED_DEPS=0
    elif [ -f "$BAKED_TAR" ]; then
      echo "[provision] loading baked image from $BAKED_TAR (local, no internet)..."
      if "$DOCKER" load -i "$BAKED_TAR" >/dev/null 2>&1 && "$DOCKER" image inspect "$BAKED_IMAGE" >/dev/null 2>&1; then
        RUN_IMAGE="$BAKED_IMAGE"; NEED_DEPS=0; echo "[provision] baked image loaded"
      else
        echo "[provision] WARNING: docker load failed; falling back to base-image + pip"
      fi
    else
      # Neither artifact exists. This used to fall through in SILENCE, which is how
      # a run ends up on the base image's torch: the slow path below needs the
      # pytorch index, the compute nodes have no route to it, the install fails
      # non-fatally (no `set -e` at this level) and the smoke test only PRINTS the
      # version. Result: training proceeds on an unpinned stack and the numbers are
      # quietly incomparable to every previous run. Say so loudly.
      echo "[provision] WARNING: USE_BAKED=1 but neither the baked image" >&2
      echo "[provision]   ($BAKED_IMAGE) nor the tar ($BAKED_TAR) exists." >&2
      echo "[provision]   Falling back to base image + live pip, which needs a route" >&2
      echo "[provision]   to $TORCH_IDX. If that fails the run uses the BASE image's" >&2
      echo "[provision]   torch, not $TORCH_PIN. Re-bake with:" >&2
      echo "[provision]     BAKE_IMAGE=1 LAUNCH_SLURM_PHASE=provision bash scripts/launch_slurm.sh" >&2
    fi
  fi
  if ! "$DOCKER" image inspect "$RUN_IMAGE" >/dev/null 2>&1; then
    echo "[provision] pulling $RUN_IMAGE (this can take a while)..."; "$DOCKER" pull "$RUN_IMAGE"
  fi

  echo "[provision] (re)starting container $CONTAINER from $RUN_IMAGE"
  "$DOCKER" rm -f "$CONTAINER" >/dev/null 2>&1 || true
  "$DOCKER" run -d --name "$CONTAINER" \
    --network=host --ipc=host --shm-size=64g \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    `# [CLUSTER-SPECIFIC] AMD ROCm device passthrough; NVIDIA uses --gpus all / nvidia runtime` \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
    `# memlock=-1 is REQUIRED for RDMA QP memory registration — do not drop` \
    --security-opt seccomp=unconfined --privileged \
    -v "$REPO_MOUNT:$REPO_MOUNT" \
    -v "$DATA_MOUNT:$DATA_MOUNT" \
    ${LOCAL_DATA_MOUNT:+-v "$LOCAL_DATA_MOUNT:$LOCAL_DATA_MOUNT"} \
    `# shared-NFS bind mounts: repo home (REPO_MOUNT, rw) + dataset/build assets (DATA_MOUNT).` \
    `# LOCAL_DATA_MOUNT is the optional node-local NVMe copy staged by` \
    `# scripts/stage_data_local.sh; point DLRM_DATA_PATH at it to read the dataset` \
    `# off local disk instead of NFS. Docker fixes mounts at run, so switching this` \
    `# on for an existing container needs FORCE_PROVISION=1.` \
    -w "$REPO" \
    "$RUN_IMAGE" sleep infinity

  # --- RDMA userspace overlay for in-container RCCL (bnxt_re) -----------------
  # The image (rocm/primus, rdma-core 50/libbnxt_re-rdmav34) ships an OLDER RDMA
  # userspace than the host kernel bnxt_re driver. The stock v34 provider faults
  # RCCL's deep-queue create_qp (max_send_wr=256) against the newer kernel uapi
  # -> "ibv_create_qp ... Bad address". Fix: stage the host's matched rdma-core
  # (libibverbs v61 + libbnxt_re-rdmav59 + libnl) on NFS so the worker phase makes
  # RCCL load it via LD_PRELOAD + LD_LIBRARY_PATH. The UNVERSIONED libibverbs.so
  # symlink is essential (import torch pulls the unversioned soname; without it
  # the lookup falls through to the container v34 lib and the fix regresses).
  if [ "${FORCE_OVERLAY:-0}" != "1" ] && ls "$OVERLAY/lib/libibverbs/"libbnxt_re-rdmav*.so >/dev/null 2>&1 && [ -L "$OVERLAY/lib/libibverbs.so" ]; then
    echo "[provision] host RDMA overlay already staged at $OVERLAY (shared NFS) — skipping"
  else
    echo "[provision] staging host RDMA userspace overlay -> $OVERLAY"
    rm -rf "${OVERLAY}.tmp" 2>/dev/null
    mkdir -p "${OVERLAY}.tmp/lib/libibverbs" "${OVERLAY}.tmp/libibverbs.d"
    cp -L /usr/lib64/libibverbs.so.1 /usr/lib64/libnl-3.so.200 /usr/lib64/libnl-route-3.so.200 "${OVERLAY}.tmp/lib/" 2>/dev/null || true
    ln -sf libibverbs.so.1 "${OVERLAY}.tmp/lib/libibverbs.so"
    cp -L /usr/lib64/libibverbs/*.so "${OVERLAY}.tmp/lib/libibverbs/" 2>/dev/null || true
    cp /etc/libibverbs.d/*.driver "${OVERLAY}.tmp/libibverbs.d/" 2>/dev/null || true
    if ls "${OVERLAY}.tmp/lib/libibverbs/"libbnxt_re-rdmav*.so >/dev/null 2>&1; then
      rm -rf "$OVERLAY" 2>/dev/null
      mv "${OVERLAY}.tmp" "$OVERLAY" 2>/dev/null || { mkdir -p "$OVERLAY"; cp -a "${OVERLAY}.tmp/." "$OVERLAY/"; }
      echo "[provision] host RDMA overlay staged: $(ls "$OVERLAY/lib/libibverbs" | wc -l) providers + libibverbs.so symlink"
    else
      echo "[provision] WARNING: host bnxt_re provider not found at /usr/lib64/libibverbs — multi-node RDMA will fail 'Bad address'; use NCCL_NET_TRANSPORT=socket"
    fi
  fi

  if [ "$NEED_DEPS" = "0" ]; then
    echo "[provision] baked image — deps preinstalled; verifying imports only"
    # All five replaced components, same as the worker-phase guard: a baked image
    # is only useful if it actually carries the pinned stack, and this is the last
    # point where a wrong one can be rejected cheaply.
    "$DOCKER" exec \
      -e TORCH_PIN="$TORCH_PIN" -e TORCHVISION_PIN="$TORCHVISION_PIN" \
      -e TORCHAUDIO_PIN="$TORCHAUDIO_PIN" -e FBGEMM_PIN="$FBGEMM_PIN" \
      -e TORCHREC_PIN="$TORCHREC_PIN" "$CONTAINER" bash -lc '
python -c "
import os, sys, torch, torchvision, torchaudio, fbgemm_gpu, torchrec, polars, xxhash, gin
print(\"imports OK,\", torch.__version__, torch.version.hip, torch.cuda.device_count(), \"gpus\")
want = {k: os.environ[e] for k, e in ((\"torch\",\"TORCH_PIN\"),
        (\"torchvision\",\"TORCHVISION_PIN\"), (\"torchaudio\",\"TORCHAUDIO_PIN\"),
        (\"fbgemm_gpu\",\"FBGEMM_PIN\"), (\"torchrec\",\"TORCHREC_PIN\"))}
got = {\"torch\": torch.__version__, \"torchvision\": torchvision.__version__,
       \"torchaudio\": torchaudio.__version__, \"fbgemm_gpu\": fbgemm_gpu.__version__,
       \"torchrec\": torchrec.__version__}
bad = {k: (got[k], v) for k, v in want.items() if v not in got[k]}
if bad:
    sys.exit(\"baked image stack mismatch: \" + \"; \".join(
        f\"{k} {g} != pinned {w}\" for k, (g, w) in bad.items()))
"' || {
      echo "[provision] ERROR: baked-image smoke FAILED (imports or pinned versions)." >&2
      echo "[provision]   Refusing to stamp a baked image that is not the pinned stack;" >&2
      echo "[provision]   re-bake with BAKE_IMAGE=1 or set USE_BAKED=0 to build from base." >&2
      exit 1
    }
  else
    echo "[provision] installing recipe deps (base image, slow path)"
    # Install misc deps FIRST, then pin the rocm torch stack + fbgemm + torchrec
    # LAST with --no-deps so nothing pulls a CUDA torch over the rocm build.
    "$DOCKER" exec "$CONTAINER" bash -lc '
set -e
echo "=== native torch ==="; python -c "import torch;print(torch.__version__)" || true
echo "=== misc python deps ==="
pip install --no-cache-dir polars-u64-idx pyarrow pyyaml tqdm psutil numba xxhash gin-config \
  absl-py pandas tensorboard torchmetrics tensordict pyre-extensions iopath typing-inspect 2>&1 | tail -3 || true
echo "=== rocm torch stack (force, no-deps, LAST) ==="
pip install --force-reinstall --no-deps --index-url '"$TORCH_IDX"' \
  torch=='"$TORCH_PIN"' torchvision==0.27.0+rocm7.2 torchaudio==2.11.0+rocm7.2
echo "=== fbgemm (local gfx950 wheel) ==="
pip install --force-reinstall --no-deps '"$FBGEMM_WHL"'
echo "=== torchrec v2026.06.01.00 (force, no-deps) ==="
pip install --force-reinstall --no-deps "git+https://github.com/pytorch/torchrec.git@v2026.06.01.00"
echo "=== import smoke ==="
python -c "
import torch, fbgemm_gpu, torchrec, polars, xxhash, gin
pin = \"'"$TORCH_PIN"'\"
print(\"imports OK,\", torch.__version__, torch.version.hip, torch.cuda.device_count(), \"gpus\")
assert torch.__version__ == pin, f\"torch {torch.__version__} != pinned {pin}\"
"
' || {
      echo "[provision] ERROR: dep install/smoke failed. The container may be on the" >&2
      echo "[provision]   BASE image's torch instead of $TORCH_PIN, which trains but" >&2
      echo "[provision]   makes results incomparable. Refusing to continue silently." >&2
      exit 1
    }
  fi

  # Stamp the container as provisioned. Written LAST and only on success, so an
  # interrupted or failed provision leaves no stamp and the next run re-provisions
  # instead of adopting a half-built container.
  if "$DOCKER" exec "$CONTAINER" bash -lc \
       "printf '%s' '$PROVISION_STAMP' > '$PROVISION_MARKER'"; then
    echo "[provision] stamped $PROVISION_MARKER: $PROVISION_STAMP"
  else
    echo "[provision] WARNING: could not write $PROVISION_MARKER; container will be re-provisioned next run" >&2
  fi

  # --- one-time bake: snapshot the deps-installed container into a reusable image
  # and save it to NFS so future nodes skip the download/pip path entirely.
  if [ "${BAKE_IMAGE:-0}" = "1" ]; then
    echo "[provision] baking: docker commit $CONTAINER -> $BAKED_IMAGE"
    if "$DOCKER" commit "$CONTAINER" "$BAKED_IMAGE" >/dev/null; then
      echo "[provision] saving $BAKED_IMAGE -> $BAKED_TAR (one-time, tens of GB)"
      if "$DOCKER" save "$BAKED_IMAGE" -o "${BAKED_TAR}.tmp.$$" && mv -f "${BAKED_TAR}.tmp.$$" "$BAKED_TAR"; then
        echo "[provision] bake done: $(ls -lh "$BAKED_TAR" 2>/dev/null | awk '{print $5}')"
      else
        echo "[provision] WARNING: docker save failed"; rm -f "${BAKED_TAR}.tmp.$$" 2>/dev/null
      fi
    else
      echo "[provision] WARNING: docker commit failed"
    fi
  fi
  echo "[provision] DONE"
}

# =============================================================================
# PHASE: worker  (inside the container)
# =============================================================================
worker() {
  cd "$REPO_ROOT"
  mkdir -p "$SCRATCH" 2>/dev/null || true
  LOG=${LOG:-$SCRATCH/yambda_5b_8gpu.log}
  # Append (not truncate): under the streaming-e2e supervisor a run may relaunch
  # many times into the SAME $LOG; the supervisor initializes it once at run start.
  # MLPerf compliance log (rank 0 writes it). Per-job filename so each standalone
  # sbatch gets a clean log; the e2e supervisor pins MLPERF_LOG_PATH itself.
  export MLPERF_LOG_PATH=${MLPERF_LOG_PATH:-$SCRATCH/mlperf/yambda_5b_mlperf.${SLURM_JOB_ID:-manual}.log}
  echo "[$(date)] REPO_ROOT=$REPO_ROOT" | tee -a "$LOG"

  # polars-u64-idx (NOT stock polars) — yambda parquet's flat-explode overruns
  # 32-bit row index. Reserved node has no outbound DNS, so install from a
  # pre-staged tarball under /shared/. Override PIP_LOCAL_TGZ for other hosts.
  PIP_LOCAL_TGZ=${PIP_LOCAL_TGZ:-${DATA_ROOT:-/data}/pip_local_yambda.tgz}   # [CLUSTER-SPECIFIC] shared-NFS path
  PIP_LOCAL_DIR=${PIP_LOCAL_DIR:-/tmp/pip_local}
  if [ ! -f "$PIP_LOCAL_DIR/lib/python3.12/site-packages/polars/__init__.py" ]; then
    rm -rf "$PIP_LOCAL_DIR"
    mkdir -p "$PIP_LOCAL_DIR" && tar xzf "$PIP_LOCAL_TGZ" -C "$(dirname "$PIP_LOCAL_DIR")" 2>&1 | tail -3 | tee -a "$LOG"
  fi

  # The pinned torch/fbgemm/torchrec stack, staged on NFS as a PYTHONPATH overlay.
  # WHY THIS EXISTS: provision installs torch==$TORCH_PIN from download.pytorch.org,
  # which these nodes cannot reach, so on any FRESHLY provisioned container that
  # install fails and python silently falls back to the BASE IMAGE's torch (2.10).
  # That trains fine and is ~9% faster because torch 2.12 regressed index_add_ 2.4x
  # (torch_212_index_scatter_regression.md, git history at 33610d2) -- i.e. the wrong
  # stack is both silent AND flattering, which is the worst combination. Overlaying
  # here is what keeps an offline node on the same stack as production.
  # Set LIB_OVERLAY=none to opt out (e.g. to reproduce the 2.10 comparison).
  #
  # COLON-SEPARATED LIST, because the recipe replaces FIVE components and they are
  # staged in two trees: the root-owned one holds torch/fbgemm/torchrec, the second
  # holds torchvision/torchaudio. Both are required -- torch 2.12 next to the image's
  # torchvision 0.25 (built against 2.10) dies at import with "partially initialized
  # module 'torchvision' has no attribute 'extension'" as soon as torchmetrics probes
  # its version, which is how this was found.
  LIB_OVERLAY=${LIB_OVERLAY:-${DATA_ROOT:-/data}/lib_overlay_c10_v212:${DATA_ROOT:-/data}/lib_overlay_v212_tv}
  OVERLAY_PP=""
  if [ "$LIB_OVERLAY" != "none" ]; then
    IFS=':' read -ra _ov_dirs <<< "$LIB_OVERLAY"
    for _ov in "${_ov_dirs[@]}"; do
      [ -n "$_ov" ] || continue
      if [ -d "$_ov" ]; then
        OVERLAY_PP="$OVERLAY_PP$_ov:"
      else
        echo "[$(date)] WARNING: LIB_OVERLAY entry $_ov not found; skipping" | tee -a "$LOG"
      fi
    done
    [ -n "$OVERLAY_PP" ] \
      && echo "[$(date)] lib overlay: ${OVERLAY_PP%:} (pinned stack)" | tee -a "$LOG" \
      || echo "[$(date)] WARNING: no LIB_OVERLAY entry resolved; using the container's own stack" | tee -a "$LOG"
  fi

  export PYTHONPATH="$OVERLAY_PP$PIP_LOCAL_DIR/lib/python3.12/site-packages:$REPO_ROOT:${PYTHONPATH:-}"
  export HOME=${HOME:-/tmp}
  # FlyDSL target arch (ship13 K10). FlyDSL 0.1.1 resolves it once per process:
  # FLYDSL_GPU_ARCH, else `rocm_agent_enumerator -name` (5 s timeout, errors
  # swallowed), else gfx942. Under full-node load the enumerator can fail or
  # exceed 5 s; that process then builds every FlyDSL kernel for gfx942 and
  # aborts (rc 139, no traceback) at the first gfx950-only MFMA intrinsic.
  # Pinned here, every process in the container inherits it. A value already
  # in this environment wins (the orchestrate phase's docker exec does not
  # forward one set at submit time). [CLUSTER-SPECIFIC] GPU arch: MI350X=gfx950.
  export FLYDSL_GPU_ARCH=${FLYDSL_GPU_ARCH:-gfx950}
  echo "[$(date)] [flydsl] FLYDSL_GPU_ARCH=$FLYDSL_GPU_ARCH (FlyDSL arch pin, ship13 K10)" | tee -a "$LOG"
  echo "[$(date)] PYTHONPATH=$PYTHONPATH" | tee -a "$LOG"
  python -c "import torch, fbgemm_gpu, torchrec, polars, xxhash, gin; print('imports OK,', torch.__version__, torch.cuda.device_count(),'gpus')" 2>&1 | tee -a "$LOG"
  # Fail loudly rather than silently measuring a different stack. ALL FIVE components
  # docs/training_recipe.md replaces are checked, not just torch: a partial
  # overlay/install can pair the pinned torch with the image's torchrec 1.4.0,
  # fbgemm 2026.5.14 or torchvision 0.25, which is either an inconsistent measurement
  # or (for torchvision) a hard import crash.
  python -c "
import sys, torch, torchvision, torchaudio, fbgemm_gpu, torchrec
want = {'torch': '${TORCH_PIN}', 'torchvision': '${TORCHVISION_PIN}',
        'torchaudio': '${TORCHAUDIO_PIN}', 'fbgemm_gpu': '${FBGEMM_PIN}',
        'torchrec': '${TORCHREC_PIN}'}
got = {'torch': torch.__version__, 'torchvision': torchvision.__version__,
       'torchaudio': torchaudio.__version__, 'fbgemm_gpu': fbgemm_gpu.__version__,
       'torchrec': torchrec.__version__}
bad = {k: (got[k], v) for k, v in want.items() if v not in got[k]}
if bad:
    sys.exit('FATAL: stack does not match docs/training_recipe.md: '
             + '; '.join(f'{k} {g} != pinned {w}' for k, (g, w) in bad.items())
             + '. Set LIB_OVERLAY to the staged pinned stack, or override the *_PIN vars.')
print('stack OK (recipe pins):', got)
" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" != "0" ]; then
    echo "[$(date)] aborting: stack does not match the pinned recipe" | tee -a "$LOG"; exit 1
  fi

  export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

  # --- distributed topology ---------------------------------------------------
  GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
  # Multi-node when launched one-task-per-node under SLURM (SLURM_NNODES>1);
  # otherwise fall through to legacy single-node defaults (localhost, node_rank 0).
  if [ "${SLURM_NNODES:-1}" -gt 1 ] && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    NNODES=${SLURM_NNODES}
    NODE_RANK=${SLURM_NODEID:-${SLURM_PROCID:-0}}
    # PREFER a MASTER_ADDR/PORT forwarded from the orchestrate phase (resolved on
    # the host, which has scontrol); the container image carries no SLURM client.
    if [ -z "${MASTER_ADDR:-}" ]; then
      MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
      MASTER_ADDR=${MASTER_ADDR:-localhost}
    fi
    MASTER_PORT=${MASTER_PORT:-$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))}
  else
    NNODES=${NNODES:-1}
    NODE_RANK=${NODE_RANK:-0}
    # Single-node: all ranks live on THIS host, so rendezvous over loopback and
    # do NOT use the SLURM hostname. On some nodes the hostname resolves to a
    # non-routable per-GPU RoCE /31 (benic 192.168.x) address; using it makes the
    # NCCL bootstrap fail with "No route to host". localhost is node-independent.
    MASTER_ADDR=localhost
    MASTER_PORT=${MASTER_PORT:-}     # empty => train_ranker picks a free port
  fi
  export NNODES NODE_RANK GPUS_PER_NODE MASTER_ADDR MASTER_PORT
  export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
  echo "[$(date)] topology: nnodes=$NNODES node_rank=$NODE_RANK gpus_per_node=$GPUS_PER_NODE world_size=$WORLD_SIZE master=$MASTER_ADDR:${MASTER_PORT:-<auto>}" | tee -a "$LOG"

  # NCCL bootstrap NIC — pin for BOTH single- and multi-node. The container is
  # --network=host so RCCL sees ALL host interfaces; if left to auto-detect, NCCL
  # can pick a non-routable per-GPU RoCE /31 (benic* 192.168.x) link and fail
  # bootstrap with "No route to host" (this is node-dependent: it happened to
  # work on some nodes and not others, causing repetitive single-node init
  # failures). Pinning the routable host NIC fixes it everywhere.
  # [CLUSTER-SPECIFIC] routable host NIC for TCP bootstrap (find via `ip -br addr`).
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic0}

  # Multi-node additionally needs the RDMA data-plane (bnxt_re HCAs) configured;
  # single-node uses intra-node P2P (XGMI/PCIe) so only the bootstrap NIC matters.
  if [ "$NNODES" -gt 1 ]; then
    NCCL_NET_TRANSPORT=${NCCL_NET_TRANSPORT:-ib}
    if [ "$NCCL_NET_TRANSPORT" = "socket" ]; then
      export NCCL_IB_DISABLE=1
      echo "[$(date)] NCCL: IB disabled — allreduce over TCP (fenic0). Functional, not RDMA-fast." | tee -a "$LOG"
    else
      # bnxt_re userspace provider ABI overlay (REQUIRED for RCCL). The stock v34
      # provider faults RCCL's create_qp (256 WRs) against the host kernel uapi
      # ("Bad address"); the host v61/v59 set staged by the provision phase works.
      # The libibverbs.so (UNVERSIONED) symlink + LD_PRELOAD are both required so
      # the torch process maps ONLY the host lib (see provision phase comment).
      if [ -e "$OVERLAY/lib/libibverbs.so.1" ]; then
        [ -e "$OVERLAY/lib/libibverbs.so" ] || ln -sf libibverbs.so.1 "$OVERLAY/lib/libibverbs.so" 2>/dev/null || true
        export LD_LIBRARY_PATH="$OVERLAY/lib:$OVERLAY/lib/libibverbs:${LD_LIBRARY_PATH:-}"
        export LD_PRELOAD="$OVERLAY/lib/libibverbs.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "[$(date)] NCCL: bnxt_re provider overlay -> $OVERLAY (host rdma-core v61/v59; symlink+LD_PRELOAD so RCCL binds the host lib for QP creation)" | tee -a "$LOG"
      else
        echo "[$(date)] WARNING: RDMA overlay $OVERLAY missing — RCCL QP creation will fail 'Bad address' on stock v34 provider; set RDMA_OVERLAY or use NCCL_NET_TRANSPORT=socket" | tee -a "$LOG"
      fi
      # MINIMAL bnxt_re set PROVEN on these PARTITION cluster nodes (vendor RCCL
      # benchmarks + confirmed e2e here). NCCL_IB_TC=104 (RoCE lossless PFC class)
      # is required; do NOT add the ionic-AINIC QPS/ECE/DMABUF block.
      # [CLUSTER-SPECIFIC] RDMA HCA names (`ibv_devices`); other vendors => mlx5_*/ionic_*
      export NCCL_IB_HCA=${NCCL_IB_HCA:-bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7}
      export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}    # [CLUSTER-SPECIFIC] RoCEv2 IPv4 GID idx (`show_gids`)
      export NCCL_IB_TC=${NCCL_IB_TC:-104}                # [CLUSTER-SPECIFIC] RoCE lossless/PFC traffic class
      export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-14}
      export NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY:-1}
      export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-0}
      # GPU-Direct RDMA: ENABLED by default. The brcmrdma host kernel ships the
      # inbox peer-memory client (`ib_register_peer_memory_client` in
      # /proc/kallsyms), so RCCL does true GPU<->NIC DMA over bnxt_re instead of
      # bouncing through host memory. Measured ~+22% throughput at 2 nodes
      # (65.7%->79.8% weak-scaling efficiency) vs the old host-staged path.
      # GDR_LEVEL=5 (most permissive) is required so GDR is used even when the GPU
      # and NIC cross the CPU root complex. NCCL_DMABUF_ENABLE=1 is a harmless
      # no-op here (kernel lacks CONFIG_DMABUF_MOVE_NOTIFY/CONFIG_PCI_P2PDMA, so
      # peermem carries it). Enabling is non-fatal: if peermem is ever absent RCCL
      # just logs "GDR 0" and falls back to host staging. Override with
      # NCCL_NET_GDR_LEVEL=0 to force the legacy host-staged path.
      export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-5}
      export NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-1}
      echo "[$(date)] NCCL: RDMA over bnxt_re (GID idx ${NCCL_IB_GID_INDEX}, TC ${NCCL_IB_TC}, GDR_LEVEL=${NCCL_NET_GDR_LEVEL}, DMABUF=${NCCL_DMABUF_ENABLE}; bnxt_re config, validated)" | tee -a "$LOG"
    fi
    # ROCr IPC mode (SHIP-13 item S13-h; multi-node only, either transport).
    # Multi-node starts hit an intermittent RCCL init crash: one local rank
    # segfaults in ROCr's DMA-BUF IPC import (libdrm amdgpu_bo_query_info <-
    # ROCr <- hipIpcOpenMemHandle <- RCCL ncclP2pImportShareableBuffer) while
    # the communicators connect (ring, tree or PXN), and its node peers abort
    # with "HIP failure: 'invalid device pointer'" on their own failed imports.
    # It has not been seen on one node. The same RCCL build's upstream issue
    # ROCm/rocm-systems#8178 was closed with this workaround:
    # HSA_ENABLE_IPC_MODE_LEGACY=1 makes ROCr export and import IPC handles
    # through the KFD IPC ioctls instead of passing DMA-BUF fds over a
    # per-exporter socket. Those ioctls exist only in the DKMS amdgpu KFD,
    # which publishes /sys/module/amdgpu/version (the inbox amdgpu does not,
    # and with it every legacy IPC export fails), so the default applies only
    # where that file is non-empty. This block runs inside the container
    # (worker phase); sysfs is visible there. IPC mappings are made when RCCL
    # connects peers: during communicator setup and, lazily, at a
    # communicator's first point-to-point use (e.g. the first all-to-all,
    # which can be after RUN_START). The mapped memory and the kernels are the
    # same in both modes, so collective arithmetic is unchanged. An explicit
    # HSA_ENABLE_IPC_MODE_LEGACY (0 or 1; forwarded by the orchestrate fabric
    # block, in both LAUNCH_FABRIC_FWD modes) always wins;
    # HSA_ENABLE_IPC_MODE_LEGACY=0 restores ROCr's default DMA-BUF IPC. ROCr
    # treats only "1" as on. Logs one line per node. Trainer-side proof of the
    # mode in effect: the "[ipc] ROCr IPC mode=" line (dlrm_v4/train/ipc_mode.py).
    _amdgpu_ver=$(cat /sys/module/amdgpu/version 2>/dev/null | tr -cd 'A-Za-z0-9._+-')
    if [ -n "${HSA_ENABLE_IPC_MODE_LEGACY:-}" ]; then
      _hsa_how="caller-set"
      case "$HSA_ENABLE_IPC_MODE_LEGACY" in
        0|1) ;;
        *) _hsa_how="caller-set; WARNING: value is not 0 or 1, ROCr treats it as off (DMA-BUF IPC)" ;;
      esac
      if [ "$HSA_ENABLE_IPC_MODE_LEGACY" = "1" ] && [ -z "$_amdgpu_ver" ]; then
        _hsa_how="caller-set; WARNING: /sys/module/amdgpu/version is empty or absent (no DKMS amdgpu KFD), legacy IPC exports may fail"
      fi
    elif [ -n "$_amdgpu_ver" ]; then
      export HSA_ENABLE_IPC_MODE_LEGACY=1
      _hsa_how="multi-node default, DKMS amdgpu KFD"
    else
      _hsa_how="not defaulted: /sys/module/amdgpu/version is empty or absent, i.e. no DKMS amdgpu KFD, which the legacy IPC ioctls need"
    fi
    echo "[$(date)] HSA: ROCr IPC mode legacy=${HSA_ENABLE_IPC_MODE_LEGACY:-unset} node_rank=$NODE_RANK host=$(hostname) amdgpu=${_amdgpu_ver:-none} ($_hsa_how; 0 or unset = ROCr DMA-BUF IPC)" | tee -a "$LOG"
  elif [ -n "${HSA_ENABLE_IPC_MODE_LEGACY:-}" ]; then
    # One node: no default (the init crash has not been seen there); a caller's
    # value is forwarded and honoured, so log it.
    echo "[$(date)] HSA: ROCr IPC mode legacy=$HSA_ENABLE_IPC_MODE_LEGACY node_rank=$NODE_RANK host=$(hostname) (caller-set, single node; 0 or unset = ROCr DMA-BUF IPC)" | tee -a "$LOG"
  fi
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export HSTU_HAMMER_KERNEL=${HSTU_HAMMER_KERNEL:-}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

  # --- GPU clock sanity guard -------------------------------------------------
  # A leftover perf_determinism cap (half clock) silently slows every kernel ~1.9x.
  # Log the perf level + a live sclk sample and try to restore boost (non-fatal).
  if command -v rocm-smi >/dev/null 2>&1; then
    echo "[$(date)] GPU perf-level check:" | tee -a "$LOG"
    rocm-smi --showperflevel 2>/dev/null | grep -iE "GPU\[[0-9]+\]" | tee -a "$LOG" || true
    if rocm-smi --showperflevel 2>/dev/null | grep -iqE "Performance Level: *(perf_determinism|manual|low)"; then
      echo "[$(date)] WARNING: GPUs not in 'auto' perf level — attempting --setperflevel auto" | tee -a "$LOG"
      rocm-smi --setperflevel auto 2>/dev/null | grep -iE "set to auto" | tee -a "$LOG" \
        || echo "[$(date)] WARNING: could not set perf level (no permission?). Run 'rocm-smi --setperflevel auto' on the HOST before benchmarking — clocks may be capped." | tee -a "$LOG"
    fi
    echo "[$(date)] sclk sample (GPU0):$(rocm-smi -d 0 --showclocks 2>/dev/null | grep -i 'sclk clock level' | sed -E 's/.*sclk clock level//')" | tee -a "$LOG" || true
  fi

  # --- stray-trainer / leaked-VRAM guard -------------------------------------
  # The trainer runs via `docker exec` into a long-lived container, so its procs
  # live in the container PID namespace, NOT the SLURM job cgroup. If a prior job
  # OOM'd/crashed, a rank can leak and keep holding ~half of every GPU's VRAM,
  # which persists across jobs (container survives) and guarantees the next
  # attempt OOMs. Before launching, reap any pre-existing trainer procs (there
  # should be none at this point) and wait for VRAM to drain. [g]-guard avoids
  # self-match. Non-fatal.
  if pgrep -f '[g]enerative_recommenders' >/dev/null 2>&1; then
    echo "[$(date)] WARNING: leaked trainer procs found pre-launch — killing." | tee -a "$LOG"
    pkill -9 -f '[g]enerative_recommenders' 2>/dev/null || true
    for _i in $(seq 1 15); do
      pgrep -f '[g]enerative_recommenders' >/dev/null 2>&1 || break
      sleep 2
    done
    sleep 5  # let the driver release VRAM after process exit
    if command -v rocm-smi >/dev/null 2>&1; then
      echo "[$(date)] post-cleanup GPU0 used GiB:$(rocm-smi --showmeminfo vram 2>/dev/null | awk -F: '/Used/{printf " %.0f", $3/1073741824; exit}')" | tee -a "$LOG"
    fi
  fi

  # WORKER_CMD override: run an arbitrary in-container command (e.g. an a2a/RCCL
  # micro-benchmark) instead of the trainer, REUSING all the NCCL/RDMA/topology
  # setup above so it exercises the exact transport the trainer uses. The
  # supervisor never sets WORKER_CMD, so the training path is unchanged.
  if [ -n "${WORKER_CMD:-}" ]; then
    echo "[$(date)] WORKER_CMD override (WORLD_SIZE=$WORLD_SIZE): $WORKER_CMD" | tee -a "$LOG"
    bash -lc "cd $REPO_ROOT && $WORKER_CMD" 2>&1 | tee -a "$LOG"
    return
  fi

  echo "[$(date)] launching train_ranker with WORLD_SIZE=$WORLD_SIZE" | tee -a "$LOG"
  # Drop hipBLASLt's untuned-shape chatter before it reaches disk rather than
  # trimming it afterwards. It emits one line per GEMM shape missing from its
  # tuning table -- on the order of a million lines a run, and with eight ranks
  # sharing the stream they tear each other's writes into unparseable
  # fragments. Left alone it reached 84 GB across 51 files and took a 101 GB
  # home to zero bytes free, which killed six convergence runs in both arms at
  # once. A trimming daemon handled this before; filtering in the pipe needs no
  # daemon to stay alive, and nothing reads these lines.
  #
  # sed rather than grep -v because `pipefail` is set at the top of this script
  # and grep exits 1 when it emits nothing, which under `set -e` would turn a
  # fully-filtered run into a spurious failure. sed exits 0 whenever it
  # processed its input. Either way pipefail still surfaces a trainer crash,
  # since any non-zero in the pipeline wins.
  #
  # Deliberately NOT sed -u. Unbuffered means a read syscall per line, and with
  # ~100 MB of chatter a run the trainer ends up blocking on a full pipe inside
  # the HIP call that emitted the line, which serialises against GPU work: the
  # first attempt at this filter used -u and inflated step time from ~337 ms to
  # ~490 ms in both arms. Block buffering costs a slightly stale tail if a run
  # is killed, which is not worth 45% of step time.
  #
  # Strip the chatter, never the text around it (r16 iteration 44; filter in
  # _hipblaslt_chatter_filter below). hipBLASLt writes each warning token by
  # token through an unbuffered std::cerr, and eight ranks share this pipe, so
  # a token often waits without its newline; a Python line written in that
  # window shares one physical line with it. Deleting every line that held a
  # phrase therefore also deleted arming lines (ship14 lab items 30/31: B3,
  # intglue and x21 lines missing on 1-3 ranks, absent from .log and .stdout).
  kcache_seed
  echo "[$(date)] hipBLASLt chatter filter: $(_hipblaslt_chatter_filter --describe)" | tee -a "$LOG"
  python -m generative_recommenders.dlrm_v4.train.train_ranker \
      --dataset yambda-5b --mode "${MODE:-streaming-train-eval}" 2>&1 \
    | _hipblaslt_chatter_filter \
    | tee -a "$LOG"
  local _train_rc=$?
  # After RUN_STOP: publish newly compiled kernels (never on a failed run, so a
  # crash mid-compile cannot leave a half-written entry behind; never with
  # KCACHE_PUBLISH=0, the read-only seed of a scored run).
  if [ "$_train_rc" -eq 0 ]; then
    if [ "${KCACHE_PUBLISH:-0}" != "1" ]; then
      if [ -n "${KCACHE_SRC:-}" ]; then
        echo "[$(date)] [kcache] publish skipped on $(hostname): KCACHE_PUBLISH=0 (read-only seed)" | tee -a "$LOG"
      fi
    else
      kcache_publish
    fi
  fi
  return $_train_rc
}

# Persistent kernel cache helpers (orchestrate KCACHE_DIR block). Both run in the
# container OUTSIDE the trainer process: seed before it starts, publish after it
# exits. Failures only log; they never fail the run.
# hipBLASLt's untuned-shape warning, as torn by eight ranks sharing one pipe
# (worker() pipes the trainer's stdout+stderr through this). The tokens are the
# literal strings of libhipblaslt.so: "Warning: Latency not found for MI_M=",
# ", MI_N=", ", MI_K=", ", mi_input_type=" and ". Returning latency value of
# 32 (really slow)." (its split pieces are covered too), between the values and
# the type name (e.g. BFloat8Float8_fnuz).
#   1. A whole, untorn warning line is dropped (one anchored match).
#   2. A line without any token passes through unchanged.
#   3. Any other line has every token removed (s///g). It is dropped only when
#      nothing but chatter residue is left (whitespace, values, type names);
#      otherwise its text is kept, possibly after a residue prefix such as
#      "16128BFloat8Float8_fnuz", which marker greps tolerate.
# perl (perl-base is part of every Debian/Ubuntu image) because glibc's regex
# engine runs this logic 5-50x slower than perl's; measured on 112 MB of clean
# warnings: perl 0.60 s, the old two-pattern sed 0.30 s, the sed form of this
# logic 16 s. Block-buffered like the old sed. Without perl, or with
# DLRM_HSTU_FP8_LOG_CHATTER_STRIP=0, the old whole-line delete is used.
_hipblaslt_chatter_filter() {
  local _mode=strip
  [ "${DLRM_HSTU_FP8_LOG_CHATTER_STRIP:-1}" = 0 ] && _mode=delete
  command -v perl >/dev/null 2>&1 || _mode=delete
  if [ "${1:-}" = --describe ]; then
    case "$_mode" in
      strip) echo "perl token strip (whole warnings dropped, other text kept)";;
      *) echo "sed whole-line delete (DLRM_HSTU_FP8_LOG_CHATTER_STRIP=${DLRM_HSTU_FP8_LOG_CHATTER_STRIP:-1}, perl $(command -v perl >/dev/null 2>&1 && echo present || echo missing))";;
    esac
    return 0
  fi
  if [ "$_mode" = strip ]; then
    exec perl -ne 'next if /^Warning: Latency not found for MI_M=[0-9]+, MI_N=[0-9]+, MI_K=[0-9]+, mi_input_type=[A-Za-z0-9_]+\. Returning latency value of [0-9]+ \(really slow\)\.\n?\z/; if (!/Latency not found for|Returning latency value of| \(really slow\)\.|, MI_[NK]=|, mi_input_type=/) { print; next } s/Warning: Latency not found for MI_M=|, MI_N=|, MI_K=|, mi_input_type=|\. Returning latency value of (?:[0-9]+ \(really slow\)\.)?| \(really slow\)\.//g; (my $r = $_) =~ s/(?:B?Float|Half|Double|Int)[A-Za-z0-9_]*|[0-9]+|\s+//g; print if length $r;'
  fi
  exec sed -e '/Latency not found for/d' -e '/Returning latency value of/d'
}

_kcache_local_triton() { echo "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"; }
_kcache_local_flydsl() { echo "$HOME/.flydsl_kcache"; }
kcache_seed() {
  [ -n "${KCACHE_SRC:-}" ] || return 0
  local t0=$SECONDS lt; lt=$(_kcache_local_triton)
  mkdir -p "$lt" || { echo "[$(date)] [kcache] WARNING: cannot create $lt; seed skipped" | tee -a "$LOG"; return 0; }
  export TRITON_CACHE_DIR="$lt"
  # Publish later sends back only entries changed after this stamp (this run's
  # compiles), not every stale entry a long-lived container has accumulated.
  touch "$lt/.kc_seed_stamp" 2>/dev/null || true
  local before after e tmp tag="$(hostname).$$"
  before=$(ls "$lt" 2>/dev/null | wc -l)
  # Shared entries are complete (published by rename); the glob skips in-flight
  # publish temps (.kc_tmp.*, dotfiles). Copy each to a local temp and rename,
  # so an interrupted/failed NFS read can never leave a partial entry under its
  # real name (Triton would then fail to load it instead of missing). Each copy
  # is time-bounded so an NFS stall cannot hang the launch. Plain cp -R: any
  # --preserve=mode also copies NFSv4 ACLs and fails ENOTSUP from /home (probe
  # job N: rc=1 on every entry although the bytes arrived).
  # Total budget (120 s) on top of the per-copy bound: at 2n/4n every node seeds
  # at once and the other nodes' ranks wait in rendezvous for the slowest one, so
  # a slow filer must not hold the launch. A cut seed only means more misses.
  local cut=0
  for e in "$KCACHE_SRC/triton"/*; do
    [ -d "$e" ] || continue
    # Only compiled code crosses runs, never Triton autotune RESULTS
    # (*.autotune.json; written only with TRITON_CACHE_AUTOTUNING=1 or
    # cache_results=True, neither set in this tree): those are timing decisions
    # made on real batches (MLPerf gray area, lab notes).
    compgen -G "$e/*.autotune.json" >/dev/null && continue
    [ -e "$lt/${e##*/}" ] && continue
    [ $((SECONDS-t0)) -ge 120 ] && { cut=1; break; }
    tmp="$lt/.kc_tmp.${e##*/}.$tag"
    timeout -k 5 60 cp -R "$e" "$tmp" 2>/dev/null && mv -T "$tmp" "$lt/${e##*/}" 2>/dev/null
    rm -rf "$tmp" 2>/dev/null || true
  done
  # Entries newer than this marker were compiled (missed) by this run; publish
  # reports their count per node (compiled_this_run), a JIT census over every
  # Triton kernel and launcher stub, not just the triton_hstu_attention marker.
  touch "$lt/.kc_seed_done" 2>/dev/null || true
  after=$(ls "$lt" 2>/dev/null | wc -l)
  if [ "${KCACHE_FLYDSL:-0}" = "1" ] && [ -d "$KCACHE_SRC/flydsl" ]; then
    export FLYDSL_RUNTIME_CACHE_DIR; FLYDSL_RUNTIME_CACHE_DIR=$(_kcache_local_flydsl)
    mkdir -p "$FLYDSL_RUNTIME_CACHE_DIR" && touch "$FLYDSL_RUNTIME_CACHE_DIR/.kc_seed_stamp" 2>/dev/null
    local f d
    while IFS= read -r -d '' f; do
      [ $((SECONDS-t0)) -ge 120 ] && { cut=1; break; }
      [ -e "$FLYDSL_RUNTIME_CACHE_DIR/$f" ] && continue
      d="$FLYDSL_RUNTIME_CACHE_DIR/$(dirname "$f")"; mkdir -p "$d" 2>/dev/null || continue
      tmp="$d/.kc_tmp.$(basename "$f").$tag"
      timeout -k 5 60 cp "$KCACHE_SRC/flydsl/$f" "$tmp" 2>/dev/null && mv -T "$tmp" "$FLYDSL_RUNTIME_CACHE_DIR/$f" 2>/dev/null
      rm -f "$tmp" 2>/dev/null || true
    done < <(cd "$KCACHE_SRC/flydsl" 2>/dev/null && timeout -k 5 60 find . -type f ! -name '.kc_tmp.*' ! -name '.kc_seed_stamp' -print0)
  fi
  # Content digest of the shared bundle (sorted per-file sha256 list), so every
  # run's log records exactly which bundle it seeded (audit for a scored set).
  local manifest dig nfiles
  manifest=$(cd "$KCACHE_SRC/triton" 2>/dev/null && timeout -k 5 60 find . -type f ! -path '*/.kc_tmp.*' -print0 2>/dev/null | LC_ALL=C sort -z | timeout -k 5 90 xargs -0 -r sha256sum 2>/dev/null)
  dig=$(printf '%s\n' "$manifest" | sha256sum | cut -c1-16)
  nfiles=$(printf '%s\n' "$manifest" | grep -c .)
  echo "[$(date)] [kcache] seeded $(hostname): triton local $before -> $after entries from $KCACHE_SRC/triton in $((SECONDS-t0)) s; TRITON_CACHE_DIR=$TRITON_CACHE_DIR flydsl=${FLYDSL_RUNTIME_CACHE_DIR:-off} seed_cut=$cut bundle_sha=$dig files=$nfiles publish=${KCACHE_PUBLISH:-0}" | tee -a "$LOG"
  # For the trainer's rank-0 arming line (train_ranker, gin item (34)): what
  # this node seeded, in the trainer's own log.
  export KCACHE_SEEDED="entries=$after files=$nfiles bundle_sha=$dig seed_cut=$cut publish=${KCACHE_PUBLISH:-0}"
}
kcache_publish() {
  [ -n "${KCACHE_SRC:-}" ] || return 0
  local t0=$SECONDS lt n=0 e tmp m=0 cut=0; lt=$(_kcache_local_triton)
  local tag="$(hostname).$$"
  for e in "$lt"/*; do
    [ -d "$e" ] || continue
    compgen -G "$e/*.autotune.json" >/dev/null && continue   # never publish autotune results (see kcache_seed)
    [ -e "$lt/.kc_seed_done" ] && [ "$e" -nt "$lt/.kc_seed_done" ] && m=$((m+1))
    [ -e "$KCACHE_SRC/triton/${e##*/}" ] && continue
    [ -e "$lt/.kc_seed_stamp" ] && ! [ "$e" -nt "$lt/.kc_seed_stamp" ] && continue
    # Budget (300 s, after RUN_STOP): a stalled filer must not hold the job.
    # The rest stay unpublished: the next run re-stamps .kc_seed_stamp, so only
    # a run in a fresh container (which recompiles them) publishes them. Keep
    # counting m.
    [ $((SECONDS-t0)) -ge 300 ] && { cut=1; continue; }
    tmp="$KCACHE_SRC/triton/.kc_tmp.${e##*/}.$tag"
    # Copy aside, then one rename(2). mv -T never nests into an existing dir:
    # a concurrent publisher of the same entry makes it fail; drop ours then.
    if timeout -k 5 60 cp -R "$e" "$tmp" 2>/dev/null && mv -T "$tmp" "$KCACHE_SRC/triton/${e##*/}" 2>/dev/null; then
      n=$((n+1))
    fi
    timeout -k 5 30 rm -rf "$tmp" 2>/dev/null || true
  done
  local nf=0 f
  if [ "${KCACHE_FLYDSL:-0}" = "1" ] && [ -n "${FLYDSL_RUNTIME_CACHE_DIR:-}" ] && [ -d "$KCACHE_SRC/flydsl" ]; then
    while IFS= read -r -d '' f; do
      [ $((SECONDS-t0)) -ge 300 ] && { cut=1; break; }
      [ -e "$KCACHE_SRC/flydsl/$f" ] && continue
      [ "$FLYDSL_RUNTIME_CACHE_DIR/$f" -nt "$FLYDSL_RUNTIME_CACHE_DIR/.kc_seed_stamp" ] || continue
      mkdir -p "$KCACHE_SRC/flydsl/$(dirname "$f")" 2>/dev/null || continue
      tmp="$KCACHE_SRC/flydsl/$(dirname "$f")/.kc_tmp.$(basename "$f").$tag"
      timeout -k 5 60 cp "$FLYDSL_RUNTIME_CACHE_DIR/$f" "$tmp" 2>/dev/null && mv -T "$tmp" "$KCACHE_SRC/flydsl/$f" 2>/dev/null && nf=$((nf+1))
      rm -f "$tmp" 2>/dev/null || true
    done < <(cd "$FLYDSL_RUNTIME_CACHE_DIR" && find . -type f ! -name '.kc_tmp.*' ! -name '.kc_seed_stamp' -print0)
  fi
  echo "[$(date)] [kcache] published $(hostname): $n new triton entries, $nf flydsl files -> $KCACHE_SRC in $((SECONDS-t0)) s; compiled_this_run=$m publish_cut=$cut" | tee -a "$LOG"
}

# ---- dispatch ---------------------------------------------------------------
case "$PHASE" in
  orchestrate) orchestrate ;;
  provision)   provision ;;
  worker)      worker ;;
  *) echo "launch_slurm.sh: unknown LAUNCH_SLURM_PHASE='$PHASE'" >&2; exit 2 ;;
esac
