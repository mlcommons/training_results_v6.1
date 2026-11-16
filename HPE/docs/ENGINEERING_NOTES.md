# Repository guide

## Goals

- Your goal is to run MLPERF training 6.1, not yet started, on AMD mi355x GPU. Until new code is relased, use this existing code (training 6.0) published by AMD in https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training6.0-repro/README.html .
- You must at least reach AMD's published performances -> https://mlcommons.org/benchmarks/training/ . I would like you to beat these published values, while strictly following the mlcommons training rules -> https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc.
- When explaining steps, progress and deliverables, don't use jargon. Use short sentence. prefer table format.
- **Status reports: every 15 minutes while work is running, as a table with performance and ETA, all times in Paris time** (CEST = UTC+2 until 2026-10-25, then CET = UTC+1; cluster clocks are UTC, lodin4's local clock is UTC-5) (user request, 2026-10-12). Never end a turn waiting for an answer while a node sits idle: queue the next useful job first.
- **Document every benchmark investigation and improvement** in `INVESTIGATIONS.md`: one row per experiment (what was tried, result, decision). Keep it concise and clear; update it as soon as a result is known.

## System

- To reproduce and optimize the benchmark, use 2 HPE XD685 with 8x Mi355x , reachable from the striker login node ( ssh monnetb@striker.hpcrb.rdlabs.ext.hpe.com , then become user hpebench with ssh hpebench@localhost) . hpebench has sudo access, don't use it to modify the OS , only to perform action like 'drop caches' or run docker. Creating/chowning staging dirs under `/lvol0` with sudo is also sanctioned.
- **Per-job node settings go through Slurm's `--comment`** (prolog/epilog apply and undo them; nodes are PXE-booted, so any manual change is lost at reboot and may also be reset by the system). Syntax: `--comment='docker;sysctl file=/abs/path.conf;thp=madvise;tuned=throughput-performance;no cstates'` (`;` separates keywords; also `swap=64G`, `restrict ram=…`, `scaling governor=…`). Documented at http://10.188.96.4/~bench/SLURM_howto.htm (reachable from the Mac, not from striker or the nodes). The user authorized (2026-10-12) relaxing `vm.overcommit_memory` and setting `vm.zone_reclaim_mode=0` / THP defrag `madvise`; prefer the `--comment` route for sysctls.
- **Never modify the system configuration** (RAID/mdadm, disks, partitions, mounts, sysctl, network, drivers, services) — not even to fix an obvious problem. Read-only inspection is fine; report problems and let the admin act.
- striker provides slurm , use salloc -N1 -n8 --exclusive --comment='docker' in order to start docker daemon on the nodes. Only use partition 'def' with lodin[1-2] nodes.
- data (models & datasets) are in /nas/SHARED/MLPERF
- **lodin[1-2] have no high-speed interconnect (no IB/RoCE). Only single-node (1x8x1) runs are valid on this system.** Use the two nodes as two independent single-node workers, never as one 2-node job.

### Measured system facts (verified 2026-09-27)
- Log in with **`ssh hpebench@16.143.10.68`** (striker, direct, key-based). Use the IP: DNS for `*.hpcrb.rdlabs.ext.hpe.com` is intermittently unavailable. The older route `ssh -J monnetb@<striker> hpebench@localhost` also works. From striker, `ssh lodin1|lodin2` reaches the compute nodes directly. `salloc --no-shell -N2 -n16 --exclusive --comment='docker'` holds an allocation without occupying a terminal.
- **lodin4 (weekend loan, 2026-10-11)**: `ssh monnetb@lodi-login.hpcrb.rdlabs.ext.hpe.com`, partition `MDF-High`, no `--comment=docker` needed. RHEL 9.7 with AMD's amdgpu 7.1.3 driver, 8 MI355X, EPYC 9555, 14 TB RAID0 `/lvol0` (8 NVMe), Docker 29 (containerd image store, root on `/lvol0`), `vm.overcommit_memory=0`. As monnetb: no passwordless sudo (drop caches from a privileged container), `/lvol0` root-owned (`/lvol0/mlperf` created via a busybox container), shared data at `/nas/data/SHARED/MLPERF` read-only, home `/home/users/monnetb` on the NAS (writable, durable). Only 1 GbE to the NAS (~115 MB/s; IB not routable). Code/logs in `~/mlperf61`. A tar-loaded image has a different ID there (manifest digest) and BuildKit still pulls a digest-pinned base: compare `RootFS.Layers` to check identity.
- **Node sharing (user decision):** from 2026-10-07 only one node is ours, the other is left to Joshua (`ours-llama2-*` jobs, `/nas/SHARED/MLPERF/joshua/`). 2026-10-07: lodin1 ours. **2026-10-08 09:20: switched, lodin2 ours, lodin1 freed.** Never queue on the other node unless the user says so.
- The Llama 3.1 86.21 set (`softmax32_final`, 2026-09-28) ran on **lodin3** (partition `dmoise`, not used since), which runs at lodin2 speed (35.9 samples/s; lodin1 34.95).
- When planning jobs/workloads, prefer `sbatch` to make sure nodes are released at the end of work. This is critical to keep fair sharing between users. Only use `salloc` when interactive mode is mandatory.
- CPU is **2x AMD EPYC 9555** (64c/socket, SMT on, **8 NUMA nodes**). AMD/Dell/Vultr published MI355X results used the higher-clocked **EPYC 9575F**, so a small structural throughput deficit is expected.
- **GPU driver is the in-kernel amdgpu of Ubuntu 26.04 (kernel 7.0.0-22), not AMD's `amdgpu-dkms`.** Consequence: HIP virtual-memory peer access (`hipMemSetAccess` to another GPU, used by RCCL with `NCCL_CUMEM_ENABLE=1` and by the 6.1 drop's `rccl_sdma` all-gather) fails with EINVAL: the driver refuses to DMA-map XGMI-linked VRAM (`amdgpu_dma_buf.c:223` WARN, kernel log). Use `NCCL_CUMEM_ENABLE=0` and `MEGATRON_PARAM_GATHER_BACKEND=nccl`. Fix = AMD DKMS driver or newer kernel (admin decision). Details: INVESTIGATIONS, 6.1 drop #16.
- `cpufreq` sysfs is absent (`scaling_driver` empty), so `runtime_tunables.sh`'s `cpupower frequency-set -g performance` silently no-ops. THP=always and `numa_balancing=0` are already in effect.
- GPU→NUMA mapping is **not** the identity. BDF-ordered (= ROCm device order, confirmed against `rocm-smi --showbus`) GPUs map to NUMA nodes **[1,2,3,0,5,6,7,4]**. A naive `rank==node` binding is wrong for 6 of 8 ranks.
- Under load all 8 GPUs sit **at the 1400 W power cap** at 100% utilisation and 61-71 °C. The workload is power-limited, not thermally limited; gains must come from efficiency, not clocks.
- **lodin1 is ~2.8% slower than lodin2**, and it is one GPU. Training speed is set by the slowest GPU. Per-GPU bf16 GEMM (`tools/gpu_bench.sbatch`, all 8 loaded): lodin1 **GPU0 = 1,487 TFLOPS** (GPU3 1,507), others ~1,600; lodin2's slowest is 1,531. 1,531/1,487 = 2.9%, matching the measured llama31 gap (37.78 vs 36.74 samples/s) and the llama2 gap. Repeatable to 0.1% across runs. Firmware, VBIOS, BIOS, PCIe, DIMMs, kernel are identical, so this is silicon variance under the power cap. **Use lodin2 for scored runs** when it is available to us (since 2026-10-07 only lodin1 is, see above).
- `/lvol0` (`/dev/md0`) is a **degraded RAID5 on both nodes** (3 of 4 NVMe active; nvme3n1 not in the array; admin topic). Few large readers stream at ~1.9 GB/s, many small parallel readers drop to 0.2–0.36 GB/s: keep data-loader worker counts low for I/O-heavy benchmarks.
- `/` (and therefore `/var/lib/docker`) is **tmpfs on a RAM-root node** — Docker images consume system RAM. `/lvol0` (5.3 TB XFS on `/dev/md0`) is the real node-local disk and is root-owned by default. `/nas` is NFSv4.
- **Storage policy: `/lvol0` is scratch, `/nas` is durable.** `/lvol0` survives reboots but is node-local, unbacked, and may be reclaimed. Every derived artifact that cost real compute (converted datasets, prepared trees) must also be persisted to `/nas`. Restore future nodes from `/nas`, never from a peer node's `/lvol0`.
- **Off-cluster backup (2026-10-14): vader `/hpelustre/monnetb/MLPERF/training-6.1/`** (`ssh monnetb@vader-login1.hpcrb.rdlabs.ext.hpe.com`): `nas_MLPERF/` = all of `/nas/SHARED/MLPERF` except `joshua/` (6.2 TB: datasets, images, results, packages), `striker_home_hpebench/` (hpebench home without dotfiles/venvs), `repo/` (this repo + AMD zip). Verified with an rsync dry run (0 differences; results and final package checksum-identical). Re-run from striker: `tmux new -d -s vader-backup "bash ~/vader_sync/resume.sh"` (copies, then writes `~/vader_sync/verify.log`). Only ~100 MB/s via striker's 1 GbE; never hold compute nodes for it.
- **Networking (re-measured 2026-10-11): `/nas` is now mounted over InfiniBand on lodin1/lodin2** (NFS 4.2 from `10.50.50.5` / `10.50.50.1`, the NAS server lodi-login's point-to-point IB links): a 227 GB copy `/nas` → `/lvol0` ran at **~2.7 GB/s**. Internet (and striker) still go through 1 GbE `eth0`, and the **site internet link is shared**: a 16-stream download on striker (~80-100 MB/s) cuts a node's own downloads to ~0.2 MB/s. Docker Hub ~5 MB/s; download.pytorch.org ~12 MB/s per stream, but pip inside `docker build` crawls (use `build_image.sbatch` PREFETCH). Older figures (/nas ~100 MB/s, ssh 42 MB/s) predate the IB mount.
- Node image inventory is not uniform. lodin2 had both benchmark images; lodin1 initially had only llama31 and had to pull `rocm/amd-mlperf:llama2_70b_training_6.0` (~85 GB) from Docker Hub. Both nodes have working outbound internet (github, pypi, docker.io).
- **Every Docker image we pull or build must be saved to `/nas/SHARED/MLPERF/containers/`** (plain `docker save` .tar, named `<repo-without-rocm/>_<tag>.tar`), right after it appears on a node. Images live in RAM (tmpfs) and Docker Hub runs at ~5 MB/s here; loading from `/nas` takes ~2 min. `HPE/tools/prep_image.sbatch` does pull+save; `docker_images_save.sbatch` saves existing ones; launchers load from there first. On lodin1/lodin2 the same folder is `/nas/data/SHARED/MLPERF/containers/`.


## Repository layout
This repo uses the **MLCommons submission tree** so `HPE/tools/make_submission.sh` copies rather than rearranges:

```
HPE/{systems,benchmarks/<bench>/implementations/MI355X_release_{6.0,6.1},results,tools,docs}
HPE/tools/        analyze_results.py, make_submission.sh, validate_submission.sh, select_runs.py, status.sh, *.sbatch
```

Each benchmark's implementation folder holds the Docker build context (flat, no `code/` subfolder), the launchers and our run scripts; one-off tuning material is in its `experimental/` subfolder. This code **replaces** AMD's stock v6.0 code (commit `a33eb5b`); `git diff a33eb5b -- HPE/benchmarks/` shows every HPE change. Datasets, checkpoints and tuning-run logs are **gitignored** — they stay on the cluster (`/nas/SHARED/MLPERF/results/<bench>/<tag>/`). `HPE/results/<system>/<bench>/result_N.txt` and `HPE/systems/` are copies of the **final 6.1 package** (`/nas/SHARED/MLPERF/submission_61_20261013`, 5 benchmarks, 50 logs + `scaling.json` for Llama 2 and DLRMv4). `MI355X_release_6.1/` holds AMD's 6.1 partner drops (imported unchanged, HPE commits after) and the Llama 2 Primus patch; FLUX.1's scored code is still `MI355X_release_6.0/`. Branches: `amd_mlperf_train_release_6.1.0` (current), `amd_mlperf_train_release_6.0.0` (6.0-era work). The bundled `primus_mllog` wheel is explicitly *not* ignored; the Dockerfiles need it at build time.

## Current state (2026-10-13)
Final 6.1 results, package, and the exact launch command of every scored set: `README.md`. Every experiment: `INVESTIGATIONS.md`.
GPT-OSS 66.08 · Llama 3.1 76.80 · Llama 2 8.245 (median window of a 20-run group) · FLUX.1 260.57 · DLRMv4 51.91 min; official `verify_for_v6.1_training.sh` passes. Open: cooling/BIOS fields of the system description.

## Execution boundaries
- This checkout contains five independent MLPerf training workloads (Llama 3.1 8B, Llama 2 70B LoRA, FLUX.1, GPT-OSS 20B, DLRMv4 — the last only as the 6.1 drop `MI355X_release_6.1`); the notes below were written for the two Llama ones. Each `<benchmark>/implementations/MI355X_release_6.0/` is its own Docker build context and launch directory; there is no root build/test runner.
- Training requires a Linux ROCm host. The supplied MI350X/MI355X configs launch 8 GPUs; launchers use Bash `mapfile`, GNU `readlink`, `/dev/kfd`, `/dev/dri`, and `/dev/infiniband`.
- Dependencies live primarily in the Dockerfiles: pinned Primus, TransformerEngine, and AITER sources plus the bundled `primus_mllog` wheel. `requirements.txt` alone only installs MLPerf logging. The two workloads pin different Primus revisions.
- `llama2_70b`: `run_with_docker.sh` → `run_and_time.sh` → Primus `direct train posttrain` → `conf/llama2_70b_lora_posttrain.yaml` → `src/llama2_70b_lora.yaml` → custom Megatron-Bridge recipe. Importing `src/llama2_custom.py` installs training-loop overrides.
- `llama31_8b`: `run_with_docker.sh` → `run_and_time.sh` → `torchrun src/train.py` → `MLPerfMegatronPretrainTrainer` from the bundled wheel, using Megatron-LM. Set `EXP` via the system config; `train.py`'s fallback names an absent FP8 YAML.
- Both launchers bind-mount `code/` at `/workspace/code`. Llama 2's recipe, model preset, and log suppression module are also copied into `/workspace/Primus` at image build time: editing their local sources requires rebuilding or explicitly updating those installed copies. Its `patches/` are applied to the pinned Megatron-Bridge and nested Megatron-LM during the build.

## Build and launch
Build from the checkout root with explicit contexts (the READMEs' `docker build ... .` assumes you are already inside `code/`):

```bash
docker build -t rocm/amd-mlperf:llama2_70b_training_6.0 llama2_70b/code
docker build -t rocm/amd-mlperf:llama31_8b_training_6.0 llama31_8b/code
```

From the selected benchmark's `code/`, export `CONT`, `DATADIR`, `LOGDIR` and the model mount variable below, then:

```bash
export PYTHONPATH=""
source config_MI355X_1x8x1.sh
NEXP=1 bash run_with_docker.sh
```

- Use `config_MI350X_1x8x1.sh` for MI350X. The Llama 3.1 README's `config_MI355X_1x8x1_8b.sh` does not exist. Set overrides after sourcing; Llama 2's config also resets `NEXP`.
- Llama 2 mounts `${DATADIR}/data` → `/data` and `${MODEL:-${DATADIR}/model}` → `/ckpt`. Training expects `/data/{train.npy,validation.npy,packed_metadata.jsonl}` and `/ckpt/megatron_checkpoints/Llama-2-70b-hf/iter_0000000`.
- Llama 2 preparation runs `bash scripts/prepare_data_and_model.sh` inside its image with `HF_TOKEN` and a writable `/data` mount. It writes both packed data and `megatron_checkpoints/` under `/data`; when using that output layout, export `MODEL="${DATADIR}/data"` for training.
- Llama 3.1 mounts `${DATADIR}` directly → `/data` and requires `${MODELDIR}` → `/model` for its tokenizer. The C4 prefixes in `conf/llama3.1_8B-pretrain-fp4.yaml` each need `.bin` and `.idx` files. Download instructions are in its `code/README.md`.
- The outer `run.llama*.training.sh` wrappers hard-code `/nas/SHARED/MLPERF` paths and resolve `cd ./code` against the caller's directory.
- `CLEAR_CACHES` is the real variable; the wrappers' `CLEAR_CACHE=0` has no effect. AMD's `runtime_tunables.sh` changes CPU/kernel settings via sudo (C2 idle disable, ASLR off, NMI watchdog, NUMA balancing, THP). **Since 2026-10-07 every launcher runs it only with `RUN_RUNTIME_TUNABLES=1` (default off)**; the per-trial page-cache drop is separate and unchanged. Before that, the v6.0 launchers ran it on every job, so lodin1/lodin2 currently show C2 disabled and `randomize_va_space=0` (runtime state, reset on reboot).

## Configuration and debugging
- Docker environment forwarding is an allowlist: launchers discover exported variable **names** by sourcing the system config in `env -i`, then forward their current host values plus explicit additions. A new host export is not automatically passed into training; add it to the config or launcher's forwarding list.
- **`pkill -f` matches whole command lines, including Slurm's batch script (its arguments) and your own `ssh ... bash -c` session.** Killing one task by script name ended job 429 (both tasks) on 2026-10-11, and `pkill -f fetch_dataset` killed its own ssh. Kill by PID or `pkill -x <name>`.
- **Never start a docker job on a node whose previous job is still in `CG` (completing).** The SLURM prolog/epilog manage the Docker daemon from the `--comment=docker` flag: the epilog runs `disable-docker.sh` ("Stopping Docker daemon, killing any remaining Docker instance"), and the prolog does the same for any job that does *not* request docker. A completing job's epilog therefore **destroys the container of a job that has already started on that node**. The victim dies with no traceback and no `error.log`, because the container is torn down rather than crashing. This killed jobs 55 and 57, both submitted immediately after `scancel`ing a non-docker job on the same node. Wait for `squeue` to show the node clear -- not merely `CG` -- before submitting docker work, and prefer `--dependency=afterany:<jobid>` over cancel-and-resubmit.
- Symptom signature of that kill: the `:::MLLOG` log stops at `launching experiment using:` (~3 KB), `PRIMUS_WORKSPACE` per-rank logs exist but rank-0 stops mid-init (for llama2, right after `Initializing MegatronBridgeBaseTrainer` / `multistorageclient package is available`, inside the ~12 s gap before `Detected Megatron Core, using Megatron-FSDP`), and every `error.log` is **empty**. Empty `error.log` plus a truncated rank-0 log means *killed*, not *failed*. An earlier revision of this file blamed `/nas` for job 55; that was wrong -- `/nas` was coincidental, and running from `/home` is not required.
- Llama 2 data (`/lvol0/mlperf/llama2/data`, incl. `train.npy`/`validation.npy`) and DLRMv4 data (`/lvol0/mlperf/dlrmv4/data`) are complete on **both** lodin1 and lodin2 (checked 2026-10-13); lodin1 was missing the two `.npy` files until 2026-09-30.
- Trace overrides to the active YAML/recipe before changing similarly named shell variables. For example, Llama 2's YAML reads `PRIMUS_TRAIN_ITERS` and `PACKED_DATA_DIR`, not the config's `MAX_STEPS` and `PACKED_TRAIN_DATA_PATH`; the former names also need forwarding if introduced as host overrides.
- Quiet logging is installed before framework imports. Llama 3.1 uses `MLPERF_VERBOSE_LOGS=1`; Llama 2 uses `MLLOG_VERBOSE_LOGS=1` (add forwarding for this variable). Llama 3.1's shell launcher still redirects torchrun stderr to `/dev/null` even in verbose mode.
- No test suite, lint/typecheck configuration, or CI workflow is included. Focused syntax checks are `bash -n path/to/changed.sh` and `python3 -m py_compile path/to/changed.py`; runtime verification needs the corresponding image, GPUs, and prepared data. `NEXP=1` is one full training trial, not a smoke test.

## Llama 2 70B: the "hang" and its fix (resolved)
A run that appears to hang with a ~3 KB log ending at `launching experiment using:` is **not** a hang. `run_and_time.sh` runs `primus-cli ... 2>/dev/null | grep -E ':::MLLOG |^RESULT,|Training failed'`, so stderr is discarded and stdout is filtered. When one rank raises, the other seven block forever in a NCCL collective and nothing is printed. Always debug with `MLLOG_VERBOSE_LOGS=1`.

Two distinct root causes, both now fixed:
1. **Checkpoint conversion loop.** `/workspace/Primus/runner/helpers/hooks/train/posttrain/megatron_bridge/01_convert_checkpoints.sh:147` computes `MEGATRON_PATH="${EFFECTIVE_DATA}/megatron_checkpoints/$(basename "${HF_PATH}")"` where `EFFECTIVE_DATA` resolves from `DATA_PATH` (`/data`). If that **directory** does not exist it re-downloads the *gated* `meta-llama/Llama-2-70b-hf` and fails with 401. A stale `Llama-2-70b-hf.converting.lock` left by an earlier aborted conversion does **not** satisfy the `[[ -d ]]` test. The converted checkpoint must therefore live at `${DATADIR}/data/megatron_checkpoints/Llama-2-70b-hf`, i.e. `MODEL=${DATADIR}/data`.
2. **Training-time HF access.** Even with the hook satisfied, the recipe's `src/llama2_70b_lora.yaml:4 hf_path: meta-llama/Llama-2-70b-hf` makes Megatron-Bridge load the HF *config/tokenizer*, hitting the gated repo again. Fix: pre-stage a minimal offline HF cache (config.json, tokenizer*, index jsons — **not** the weights, which come from the Megatron checkpoint) into `${DATADIR}/data/huggingface` and set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. A complete upstream snapshot already exists at `/nas/SHARED/MLPERF/hf_cache`; `monnetb`'s `~/.bashrc` also has a working `HF_TOKEN` if online access is ever needed.

`PRIMUS_WORKSPACE=/results/primus` is now exported by the config so per-rank Primus logs (`.../logs/post_trainer/rank-N/debug.log`) survive container teardown. That is where the real traceback lives.

## Performance findings (MI355X, 8 GPU, single node)
Measured with `llama31_8b/probe.sh` (short training slice; throughput measured strictly between two evals, so startup/JIT/eval are excluded). Probe numbers read ~4% high in absolute terms because power-capped GPUs boost harder in short blocks — treat them as **relative** rankings and confirm winners with full runs.

| lever | effect on llama31 throughput |
|---|---|
| `PRIMUS_MICRO_BATCH_SIZE=4` (grad-accum 1 instead of 2) | **+3.5%** — best single lever |
| `MLPERF_NUMA_BIND=1` | +1.8% |
| image rebuilt with `CU_NUM=256`, gfx950-only, AOTriton on | +1.6% |
| `CUDA_DEVICE_MAX_CONNECTIONS=8` | +1.5% |
| dataset on NFS vs node-local NVMe | +0.8% (i.e. **not I/O bound**; staging is not worth much for llama31) |
| `GPU_MAX_HW_QUEUES=4` | **-4.8%** |
| `PRIMUS_MICRO_BATCH_SIZE=1` | **-6.5%** |

- Steady-state throughput is flat across a 90-min run (35.3 samp/s ±0.1 over 15 blocks); only block 0 is slower (34.5) from ramp-up. There is no thermal decay to chase.
- `micro_batch_size` must be mirrored into `MLLOG_MICRO_BATCH_SIZE` for compliance logging when changed.
- NUMA binding is implemented in `llama31_8b/.../src/_numa_affinity.py` (imported by `train.py`). Llama 2 has no equivalent entrypoint, so the same module is loaded via `llama2_70b_lora/.../src/sitecustomize.py` with `/workspace/code/src` prepended to `PYTHONPATH`. It is gated on `MLPERF_NUMA_BIND=1` and only binds processes that have `LOCAL_RANK` set, so the launcher stays unpinned. Its `[numa] rank N -> GPUN numa=M` message goes to stderr and is therefore invisible in quiet mode.

## Convergence / RCP budget (this is where most of the headroom is)
The RCP checker scales a score **back up** by `RCP_mean / your_mean` when you converge faster than the reference, so converging early is worth nothing, while converging late is a pure loss. The optimum is to land the olympic mean of samples-to-converge *just at* the RCP mean.

| benchmark | GBS | RCP olympic mean | hard fail floor (`min_epochs`) | headroom | observed here |
|---|---|---|---|---|---|
| llama31_8b | 32 | **178,859** | **173,650** | 3.00% | 193,536 (+8.2%, no penalty — 6.1 set `amd61-set`, 76.80 min; 6.0 set: 185,856) |
| llama2_70b_lora | 8 | **3,179** | **2,961** | 7.36% | 3,072 (**below** RCP mean → ×1.0347 — 20-run group, median window, 8.245 min) |
| flux1 | 512 | **7,223,524** | **7,112,400** | **1.56%** | 7,733,248 (+7.1%, no penalty — 6.1 set `log61-set`, 260.57 min) |
| dlrmv4 | 8192 | **69,067,662** | **66,297,354** | 4.18% | 67,952,640 (**below** RCP mean → ×1.0164 — final set `amd61-final-lodin2`, 51.91 min; set 1: 70.53M, 65.93 min) |
| gpt_oss_20b | 32 | **230,059** (official: BS 32 set pruned, interpolated from BS 16/64; raw BS 32 set 234,837) | **222,721** | 3.2% | 236,544 (+2.8%, no penalty — 6.1 set `amd61-set`, 66.08 min; 6.0 set 233,472, 98.24 min) |

- **Scores must be compared RCP-normalised.** Published results already are. `analyze_results.py` applies `effective_score()` = `raw x max(1, RCP_mean/your_samples)` before any "vs published" line. Comparing a raw score against a published one overstates your position, and did so here until 2026-09-30.
- **Convergence is quantised by the eval cadence, which dominates the RCP arithmetic.** llama2 evals every 384 sequences (`CEIL(384/GBS)*GBS`, skipping the first `FLOOR(0.125*GBS+2)`), so at GBS 8 the only reachable values near the RCP mean of 3,179 are **2,688 / 3,072 / 3,456**. Landing "on" the RCP mean is impossible. Measured: 3,072 -> 8.154 min raw, 3,456 -> 9.490 min raw. Effective scores are **8.437** vs **9.490** — an extra eval cycle costs 1.34 min of wallclock to buy only 3.5% of scaling relief, so it is a heavy net loss.
- **Therefore the llama2 optimum is 100% of runs landing at the *lowest legal* point, 3,072** — not at the RCP mean. Observed rate is 11/15 (73%), giving ~8.50; forcing it to 100% is worth ~0.8%. The remaining **2.0% gap to AMD is pure throughput**: AMD's 8.270 implies a raw 7.992 min at 3,072 versus our 8.154.
- **Update 2026-10-12 (6.1):** the speed at 3,072 is now 7.90 min (THP `always`) / 7.97 min (THP `madvise`), so the RCP-scaled ceiling with every run at 3,072 is **8.17 / 8.245**. Loss at 3,072 is 0.9247 ± 0.0017 vs the 0.925 target, so the share of runs stopping there is luck (50–80% per set). Shorter cosine schedules (432, 480) do not lower it and can leave runs stuck. Submitted: a 20-run group (zone_reclaim 0 + THP madvise, 16/20 at 3,072), median window 8.245. With more than 10 runs the rule is the window **closest to the median**, never the fastest; never re-run unchanged settings for luck.
- Note 2,688 would need the olympic mean to stay >= `min_epochs` 2,961, i.e. at most ~29% of the olympic-8 runs there. That is a stochastic mix sitting exactly on a hard-fail boundary, so it is not a safe target without a lot of measurement.

- Figures are reproducible with the official `rcp_checker.py` statistics: olympic mean rejecting 1 sample per side, and `min_epochs` from the binary-search t-test (alpha=0.05, `submission_runs=10`). Recomputing llama31 @ BS 32 from `rcps_llama31_8b.json` reproduces 178,858.67 / 173,649.76 exactly, which validates the method used for the flux1 row.
- llama31 reference distribution at GBS 32 is 10x 172,032 / 9x 184,320 / 1x 196,608. Our runs never reached 172,032. The reference used warmup 4,096 samples (128 iters, ours 64); trying 128 did not help (2026-10-07: 184,320 and 196,608). LR 7e-4 and 1e-3 were also worse.
- `opt_base_learning_rate` and `opt_learning_rate_warmup_steps` are **unconstrained** by the rules; `min_lr` is fixed at `0.1 x base_lr` and `opt_learning_rate_decay_steps` must equal `1,200,000 - warmup` (machine-checked). AMD ships `1e-3` for MI350X vs `8e-4` for MI355X, so LR is a legitimate tuning axis.
- Seeds must come from the clock/urandom and **no two runs may log the same seed** — seed-hunting is prohibited.
- Eval cadence (every 12,288 sequences) and `eval_samples == 1024` are fixed and machine-checked. Eval time is on the clock (~3.3% of llama31 TTT, ~11% of llama2 TTT).
- Staging data from NFS to node-local NVMe is explicitly allowed ("data can start on any durable storage system such as local disks"), but caches must still be dropped, and RAM/tmpfs staging is forbidden.

## DLRMv4 (new in 6.1; AMD drop `3b590b6`, final set 51.91 min)
- **Code**: `HPE/benchmarks/dlrmv4/implementations/MI355X_release_6.1/` (AMD drop + HPE commits). HSTU ranking model, plain PyTorch + TorchRec + FBGEMM (not Primus). Target AUC 0.75 (`listen_plus`). Batch 8192 on one node. Launch: `HPE/tools/run_amd61.sbatch BENCH=dlrmv4` (forwards every `VAR=VAL` into the container via `HPE_FORWARD`, logs committed memory every 10 s to `commit-<job>.log`).
- **Image** `rocm/amd-mlperf:dlrmv4_training_6.1` (81 GB tar in `containers/`): built by `HPE/tools/build_image.sbatch` in ~27 min. FBGEMM compiles from source (~5 min at `MAX_JOBS=128`). pip inside `docker build` cannot download the 6.2 GB torch wheel here (30 kB/s), so the script prefetches wheels with 8 parallel ranges into `/nas/SHARED/MLPERF/wheels`, checks the sha256 from PyTorch's index and serves them on `127.0.0.1:18765` (`PREFETCH_ARG/BASE/FILES`). The Dockerfile's wheel URL base is a build arg (default unchanged).
- **Data** (227 GB, 20 files, MLCommons R2): durable copy `/nas/SHARED/MLPERF/dlrmv4/data` (plus the prebuilt 18 GB time-window index `ts_{indices,offsets}_L4086_W86400.npy`), local copy `/lvol0/mlperf/dlrmv4/data` on lodin1 (and lodin4). `fetch_dataset.sh` (run on striker: 16 parallel 1 GiB ranges, HTTP/1.1 + stall timeout, ~37 min) and `stage_dataset.sbatch` (copy, md5 vs MLCommons list, AMD sha256 check, index build). Layout: `processed_5b/hstu_cache_L4086/` + `shared_metadata/` + `processed_5b/item_popularity.npy`.
- **Rules (6.1.0 checker)**: only LR (0.5e-6 to 1.5e-6 at batch 8192) and batch size are tunable; warmup 24,000 steps, Adam betas etc. fixed. Eval every 2,293,760 samples (block) over the full 2,058,204-sample holdout, first eval at block 25 (57.34M). Reachable stops near the RCP: 64.2M (block 28, below floor), **66.52M (29, legal, ×1.038)**, 68.81M (30), 71.11M (31). 10 runs.
- **Convergence spread is large**: at LR 1e-6, 26 runs from 64.2M to 94.0M (94.04M twice), mean ~71M. **LR 1.1e-6**: 64.2/64.2/68.8/82.6/78.0M (no gain, more runs under the floor). **LR 1.25e-6**: 57.3–64.2M, all under the 66.3M floor → set fails. Keep 1e-6. NUMA pinning (`HPE_NUMA_BIND=1`, `_hpe_numa.py`) gives no speed gain.
- **Data loader is the bottleneck on lodin1**: AMD defaults fork 2 train loaders × 4 workers + 8 eval workers = 16 per GPU, each fork reserves the parent's ~20 GB under `vm.overcommit_memory=2` → ~2.5 TB > CommitLimit → crash at start (`ENOSPC` on `/dev/shm` or `ENOMEM` in fork). Set-1 setting under strict overcommit: `NUM_WORKERS=4 EVAL_NUM_WORKERS=2` (~2,150–2,220 GB committed, fits; 65.93 min). With it, training ~22k samples/s and ~3.9 min per eval cycle (2 min of it eval). The 4 active train workers per GPU are at 100% CPU (py-spy: main thread waits on the loader queue).
- **lodin4 runs the same code ~1.8× faster** (24.7 min to first eval vs 43.8; 1.4 min per cycle vs 3.9; runs 31–43 min). GPUs and CPU clocks are equal. Differences found: overcommit 0 (16 workers fit), **`vm.zone_reclaim_mode` 0 (lodin1/2 image: 3)**, THP defrag `madvise` (lodin1/2: `always`, leftover of AMD's `runtime_tunables.sh`), healthy RAID0. With lodin1's worker counts lodin4 still reaches the first eval at 26.6 min, so most of the gap is not worker count.
- Tried and rejected: 16 workers with overcommit 0 on lodin1 (training speed unchanged at ~23.8k samples/s); `DOUBLE_BUFFER=0` with 8 workers (persistent loader re-iterated every ~8-step window, waits for ~64 discarded prefetched batches: no progress); 3 train / 4 eval workers (eval faster, training 9% slower, net loss).
- Tested 2026-10-12: zone_reclaim 0 + THP defrag madvise alone (job 447) gave no gain; `forkserver` loader workers rejected (still ~8 GB per worker). **Final setting (set 2 = 52.43 min on lodin1; final set = 51.91 min on lodin2, job 455):** `--comment='docker;sysctl file=HPE/tools/lodin_sysctl_oc0.conf'` (overcommit 0 + zone_reclaim 0, ~3.4 TB committed is then allowed), `HPE_THP_DEFRAG=madvise`, `NUM_WORKERS=6 EVAL_NUM_WORKERS=8 PREFETCH_FACTOR=4`, LR 1e-6. Never stop such a job with plain `scancel` while its container holds ~3 TB: `docker rm -f` first (see the overcommit note in the GPT-OSS section).

## FLUX.1 6.1 drop (`3b590b6`): not adopted
- Primus v26.7 recipe with MXFP4/FP8 FlyDSL kernels, built from source (Primus-Turbo, RCCL, patched ROCr). Reads the raw MLCommons `.arrow` layout directly. Its 1-node config (`1x8x32`) uses **batch 256, which has no RCP** → HPE `config_MI355X_1x8x64.sh` (batch 512, lr 2e-4, warmup 1600). `config_common.sh` exported **SEED=10007 for every trial** → launcher now draws a urandom seed per trial. `rccl_sdma` needs Linux ≥ 6.8 (lodin4) and fails on lodin1/2's driver → `FSDP_ALL_GATHER_BACKEND=nccl`.
- **2.2–2.6 s per step vs 1.02 s for our 6.0-based code** on both lodin1 (data from `/nas` over IB) and lodin4: GPUs mostly at 300–450 W, main Python thread busy in FlyDSL kernel launches. Our FLUX set stays (`log61-set`, **260.57 min**, 10/10 6.1.0 compliance). Launchers: `run_flux61.sbatch` (lodin1/lodin4), image tar `containers/amd-mlperf_flux1_training_6.1.tar`.

## Flux.1-schnell (single node; 6.0 set 261.46 min, 6.1 set 260.57 min)
`flux1` is in the 6.0 benchmark set and is confirmed to carry over to 6.1 (`mlperf_logging/benchmark_meta.py` lists it under `'6.1'`). It requires **10 runs**. Quality metric is **validation loss <= 0.586** — forward-only evals, not FID/CLIP, so eval is cheap (~3-5% of TTT). Eval cadence is every 262,144 samples over 29,696 samples.

**It fits a single node comfortably, and single node is the GPU-hour-efficient way to run it.**
- The model is an 11.9B MMDiT (19 double-stream + 38 single-stream blocks, d=3072, seq 512 = 256 image + 256 T5 tokens). With FSDP2 over 8 GPUs the state is ~21 GB/GPU (fp32 master 6.0 + AdamW m/v 11.9 + bf16 grads 3.0) out of **288 GB**. It even fits unsharded on one MI355X (~190 GB), so `data_parallel_shard_degree` is a tuning knob rather than a necessity — a real advantage over the 256 GB MI325X AMD submitted on.
- **Always use GBS 512** (64/GPU, grad-accum 1): it is the smallest batch with a reference RCP and by far the cheapest convergence budget (1024/2048/4096 need 8.49M/10.30M/15.38M samples vs 7.22M). Going below 512 would require generating and getting new RCPs approved — ~20 reference runs, prohibitive.
- Hyperparameters for the BS 512 RCP: `lr 2.0e-4`, `warmup 1600`, AdamW beta=(0.9, 0.95), eps 1e-8, wd 0.1, grad clip 1.0.
- Projected **4.5-7 h per run** (~5.5 h central), anchored on AMD's 64x MI325X result (92.36 min at GBS 2048 => ~29 samples/s/GPU incl. eval) scaled to MI355X. 10 runs is ~50 h of compute, ~25 h wall clock split across lodin1 and lodin2 as independent workers.
- In GPU-hours a 1x8 run at GBS 512 is ~20% *cheaper* than AMD's 64-GPU config, purely because small GBS converges in far fewer samples.
- Unlike llama31, there is essentially **no convergence headroom** (1.56%). This is a pure throughput problem.

**Unlike llama31, flux1 is genuinely I/O sensitive.** Each sample is ~2.2 MB (the T5 256x4096 bf16 embedding dominates), so training needs **~1 GB/s sustained read** and a full run reads ~16 TB over ~6.6 epochs. The llama31 finding that NFS costs only 0.8% does **not** transfer — the training copy must be on node-local NVMe.

**Data (verified 2026-09-30).** The v5.1 dataset is byte-identical to 6.0/6.1: both pull the same unversioned artifacts (`flux-1-cc12m-preprocessed.uri`, `flux-1-coco-preprocessed.uri`, `flux-1-empty-encodings.uri`). The vader copy at `/hpelustre/SHARED/datasets/MLPERF/training5.1/sd-flux` is in **MLCommons format, so conversion is required**: `cc12m_preprocessed/` (4,765 entries, 2.2 TB) + `coco_preprocessed/` (132 entries, 60 GB) = 2.26 TB. Staged complete to `/nas/SHARED/MLPERF/sd-flux`.

- **The vader snapshot is missing `empty_encodings/`** — the T5/CLIP encoding of the empty prompt used for classifier-free guidance. It is a separate MLCommons artifact of just 2 files (`t5_empty.npy` 2 MB, `clip_empty.npy` 1.6 KB). Fetched and md5-verified into `/nas/SHARED/MLPERF/sd-flux/empty_encodings`; `stage_flux_data.sh fetch-empty` reproduces it. `convert` hard-fails up front if any of the three directories is missing or empty, rather than after a multi-hour conversion.
- Do **not** transfer these with macOS `tar`: it emits AppleDouble `._*.npy` sidecars that match a `*.npy` glob.
- AMD's container does not consume the MLCommons layout directly — it needs an **energon** tree built by `/workspace/code/scripts/to_webdataset.py` plus `energon prepare`. `energon prepare` rewrites metadata **in place** (`.nv-meta`), so the converter output is already the final tree; keeping the source on `/nas` and writing the output to `/lvol0` means neither filesystem exceeds ~2.4 TB.
- Capacity is not a constraint: `/nas` has 13 TB free; the finished energon tree leaves `/lvol0` at 50% (2.7 TB used, 2.7 TB free).

**Conversion, measured.** `to_webdataset.py` took **5h53m** for the 2.26 TB (08:12 → 14:06 UTC, 8 workers), producing 110 train shards (10,000 samples each) + 3 val shards. Sample counts match the reference exactly: **1,099,776 train / 29,696 val**. Each sample holds `.json`, `.t5.bytes`, `.clip.bytes`, `.mean.bytes`, `.logvar.bytes`.

`energon prepare` then takes ~1 min and is **interactive**; the prompts, in order, are:
1. duplicate keys warning → `y` (the converter emits per-shard keys, so duplicates across shards are expected and harmless)
2. `create a dataset.yaml interactively?` → `y`
3. sample type → `11` (Crude sample / `CrudeWebdataset`)

Non-interactive flags do not exist for these, so `printf 'y\ny\n11\n' | energon prepare ...` is the only route; it is verified working. Result is `.nv-meta/` with `dataset.yaml` (`CrudeWebdataset`), a 555 MB `index.sqlite`, and `split.yaml` mapping `train/shard_{000000..000109}.tar` + `val/shard_{000000..000002}.tar`. Note `.nv-meta` ends up **root-owned** (the container runs as root), mode 644 — readable, but not writable by `hpebench`, so a re-run needs `sudo rm -rf .nv-meta` first.

**v26.7 recipe needs the raw MLCommons `.arrow` data.** It was rebuilt from the local energon copy in ~45 min, byte-identical to the MLCommons release (all files md5-checked) with `experimental/energon_to_hf_arrow.py` + `rebuild_flux_mlc.sbatch`, into `/lvol0/mlperf/flux1_mlc` (2.3 TB per node). The recipe brought no gain, so that copy is not needed for scored runs.

**Data loader: use `NUM_WORKERS=2` (2026-10-07).** The energon reader with the default 8 workers/GPU floods the degraded RAID5 with small reads: ~0.36 GB/s, first 4–5 eval intervals 25–42 min. 16 workers is worse (0.2 GB/s); 2 workers streams at ~1.9 GB/s and keeps the GPUs at full power from the start (~8.8 min/interval). 10-run set: **261.46 min** (was 356.8). `config_MI355X_01x08x64.sh` exports `NUM_WORKERS` so it is forwarded. Primus v26.7's FLUX recipe was tested and gives no gain (347/340 min).

**Single-node path exists (verified 2026-10-01).** The image's `/workspace/code` (vendored under `flux1/.../code`) ships `config_MI355X_01x08x64.sh` (GBS 512, MBS 64, lr 2e-4, warmup 1600 = the BS 512 RCP) and a plain `run_with_docker.sh`; `run_flux1.sbatch` drives it. **First full run: 330.6 min, converged at 7,340,032 samples (eval loss 0.5858), 28 evals, ~517 s per eval interval.** That is 2,645 GPU-minutes vs 2,479 for the best B300 at 32 GPUs. Also note **no 8-GPU flux1 result was published in v6.0** (smallest found: 32 GPUs), so there is no like-for-like target to beat; the honest comparison is GPU-hours.

## GPT-OSS 20B
- **Final 6.1 set: AMD 6.1 drop (MXFP4), `run_amd61.sbatch BENCH=gpt_oss_20b`, 66.08 min** (with `NCCL_CUMEM_ENABLE=0` and plain NCCL all-gather, see System). The notes below are the 6.0-era history.
- Reference: `mlcommons/training/small_llm_moe_pretraining/primus` (AMD-authored), vendored under `gpt_oss_20b/.../code`. Ships `config_MI355X_1x8x1.sh` (GBS 16, lr 4e-4, warmup 128 iters = the BS 16 RCP). No prebuilt training image exists; `run_gptoss.sbatch` builds `hpe/mlperf:gpt_oss_20b_6.0` on first use from `rocm/primus:v25.11`.
- Data = the llama31 C4 files and Llama 3.1 tokenizer at `/lvol0/mlperf/llama31_8b/{data,model}`. Shared, so never let a gpt-oss script rewrite them.
- **The reference Dockerfile checks out Primus `main`, which breaks**: `primus_evaluator.patch` was merged upstream in Primus `8c5bc42` (2026-02-25), so `git apply` fails. Pinned to its parent `9788c18`, where both patches' pre-image blobs match exactly (`evaluator.py` f7df2870, Megatron-LM `data_samplers.py` 1bf1bf5e).
- The reference also names a wheel it does not ship (`primus_mllog-0.1.0`; `0.1.20` is shipped) and installs no `mlperf_logging`, inheriting 4.1.50 from Primus. `primus_mllog` needs >= 6.0.0-rc3 (`TENSOR_PARALLELISM`), so every rank crashed at startup. Pinned `6.0.0-rc5`, same as llama31.
- **2026-10-12 05:10 UTC: `vm.overcommit_memory` set to 0 on lodin1 and lodin2** (user-authorized, `sudo sysctl -w`, runtime only: a reboot restores 2; re-apply after any reboot). **It does not stick: the job prolog re-applies the image's sysctl defaults (overcommit 2, zone_reclaim 3) at every job start.** Use `--comment='docker;sysctl file=<repo>/HPE/tools/lodin_sysctl_oc0.conf'` instead (applied for the job). To stop such a job early, first `docker rm -f <container>` on the node, then `scancel`: never let strict mode come back while ~3 TB is committed. `run_amd61.sbatch` therefore sets it at job start and restores it on exit (`HPE_OVERCOMMIT=keep` to skip). Reason: under strict overcommit DLRMv4 could fork only 6 loader workers per GPU instead of AMD's 16; on lodin4 (overcommit 0) the same code ran ~1.8x faster.
- **lodin runs `vm.overcommit_memory=2` (strict), on both nodes.** CommitLimit = 2,267 GB. Any `os.fork()` from a process holding a big commit charge reserves it again. Primus defaults to **8 forked data-loader workers per rank**; after warmup that overflows the limit and the first training step dies with `OSError: [Errno 12] Cannot allocate memory` in `DataLoader` worker start, while 1.8 TB is free. Fixed with `num_workers: ${PRIMUS_NUM_WORKERS:0}`. Do not change the sysctl (OS change); fix the workload. Expect the same trap in any new benchmark that forks after GPU init.

### GPT-OSS speed: use AMD's Primus MLPerf recipe, not the MLCommons reference
| Recipe | samples/s (lodin1) | GPU power | result |
|---|---|---|---|
| MLCommons reference (BF16, GBS 16) | 9.6 | 650-850 W | AMD's own rcp_logs show the same ~10.7/s, 500-725 min per run |
| reference + `moe_grouped_gemm=true` | 1.6 | ~400 W | 6x slower; why it ships `false #true` |
| **AMD recipe in `rocm/primus:v26.5`** (`examples/mlperf/gpt_oss_20b`, FP8, GBS 32) | **27.0** | 1.0-1.1 kW | **converged in 147.6 min at 233,472 samples** (eff. 148.4) |

**Update 2026-10-05: Primus v26.7.0 is much faster.** Same launcher with `CONT=rocm/primus:v26.7` and `PRIMUS_HOST=~/primus/v26.7.0` (host checkout of tag v26.7.0 mounted over the image's): **39.5 samples/s; 10-run set 102.3 min** (official average 242,688 samples) with LR 8.5e-4. AMD's branch `feat/zhangrb/mlperf` (fused MoE experts, `experimental/primus_v26.7.0_fused_experts.patch`) gives no gain on this FP8 recipe. Our tuned LR for the GBS 32 FP8 recipe on v26.7: **8.25e-4** -> 10-run set **98.82 min** (official average 233,472 samples, right at the reference mean). 8.5e-4 averaged 242,688 (102.3 min); 6e-4 always stops at 221,184, below the floor. See `INVESTIGATIONS.md`.

Run it with `gpt_oss_20b/.../experimental/run_amd_primus_v265.sbatch`, which runs the recipe shipped **inside** the image so code and libraries match. Dead ends: Primus `main` + v26.5 image (`gemm_fp8()` TypeError, needs newer primus_turbo); its `rccl_sdma` param gather (HIP `invalid argument` in RCCL p2p on lodin; disable with `MEGATRON_PARAM_GATHER_BACKEND=disabled`); Primus tag checkouts before main have no `run_with_docker.sh`. `/data` must be **writable**: Megatron writes `<prefix>/cache/GPTDataset_indices/` on first use of a new batch layout. GBS 32 RCP: mean 234,837, min_epochs 229,259.

## Tooling added by this work
- `analyze_results.py <benchmark> <logdir>...` — per-run TTT/samples/throughput plus the MLPerf olympic mean, RCP verdict, and comparison against published scores.
- `llama31_8b/probe.sh <label> [VAR=VAL ...]` and `probe_queue.sh <queuefile>` — ~9 min A/B throughput probes with a summary table.
- `llama31_8b/run.llama31-8b.training.sh <tag> <nexp> [VAR=VAL ...]` and `llama2_70b/run.llama2-70b.training.sh` — parameterised drivers that validate mounts up front and derive `min_lr`/`eval_interval` so compliance-fixed relationships cannot drift.
- `llama31_8b/.../Dockerfile.tuned` — gfx950-only rebuild with `CU_NUM=256` and AOTriton enabled.
- `flux1/stage_flux_data.sh {inspect|pull|fetch-empty|convert|verify}` — Flux.1 data staging, vader lustre -> `/nas` -> node-local `/lvol0` energon tree. `inspect` is read-only and reports whether the source is MLCommons or already-energon format; run it first.
- `flux1/convert.sbatch` — batch driver for the ~6 h `to_webdataset.py` half, with an argument preflight so a wrong flag fails in seconds rather than hours. Deliberately stops before the interactive `energon prepare`. Edit `--nodelist` to target a node; `--dependency=afterany:<jobid>` queues it behind a running allocation.
- `flux1/backup_to_nas.sbatch` — persists a prepared energon tree from scratch `/lvol0` to durable `/nas/SHARED/MLPERF/flux1_energon` (~7 h for 2.3 TB), verifying shard counts and spot-checking md5 rather than re-hashing 2.3 TB.
- `llama2_70b/lr_sweep.sh` and `llama2_70b/llama2_convergence.py` — LR sweep plus a reporter that separates convergence-class hit-rate from within-class TTT. For llama2 the hit-rate at 3,072, not raw TTT, is the figure of merit.
- Staged, checksum-verified datasets on both nodes at `/lvol0/mlperf/{llama31_8b,llama2,flux1/energon}`.
- `HPE/tools/run_primus_llama.sbatch` — AMD's Primus v26.7 Llama recipes (`BENCH=llama2_70b|llama3.1_8b`); Llama 2's scored set uses it.
- `llama31_8b/.../run_llama31.sbatch` — one Llama 3.1 run with the scored-set settings (softmax32 YAML, MBS 4, NUMA) plus `VAR=VAL` overrides; derives `min_lr` and lr decay.
- `gpt_oss_20b/.../experimental/run_amd_primus_v265.sbatch` — GPT-OSS scored set (`CONT=rocm/primus:v26.7`, `PRIMUS_HOST=~/primus/v26.7.0`, `LR=8.25e-4`).
- `flux1/.../run_flux1.sbatch` — FLUX.1 scored set (v6.0 image, `NUM_WORKERS=2` default). `experimental/run_primus_flux1.sbatch` = v26.7 recipe (no gain).
- `HPE/tools/build_image.sbatch` — build + save an image (and its base) to `containers/`, with the parallel wheel prefetch. `HPE/tools/parallel.sbatch "cmd1" "cmd2"` — run independent tasks side by side in one allocation.
- `HPE/tools/run_amd61.sbatch BENCH=dlrmv4` — DLRMv4 runs (see DLRMv4 section). `HPE/tools/lodin_sysctl.conf` — per-job kernel settings through the prolog.
- `dlrmv4/.../{fetch_dataset.sh,stage_dataset.sbatch,dlrmv4_lodin4.sbatch}` and `flux1/.../{run_flux61.sbatch,stage_flux_lodin4.sbatch,build_flux_lodin4.sh,flux61_lodin4_full.sbatch}`.
- **Official scores** (what MLCommons publishes: olympic mean × RCP scaling): `HPE/tools/official_scores.sh <bench>=<results dir> ...` on striker (runs the official `result_summarizer`, ruleset 6.1.0, twice: pass 1 writes `scaling.json`, pass 2 applies it). For a built package: `cd ~/mlperf-logging-latest && PYTHONPATH=. ~/venv-mlperf/bin/python -m mlperf_logging.result_summarizer <pkg>/closed/HPE training 6.1.0 -csv out.csv` on a copy (it writes `scaling.json`). `result_summarizer.compute_score` crashes in this logging version (unpack error).
- RCP check for 6.1: `cd ~/mlperf-logging-latest` (mlcommons/logging 8a0cc05 = 6.1.0-rc2) `&& PYTHONPATH=. ~/venv-mlperf/bin/python -m mlperf_logging.rcp_checker --rcp_usage training --rcp_version 6.1.0 --verbose <dir with result_N.txt>`; the in-container compliance check of the 6.1 drops already uses ruleset 6.1.0.
- Official log checker: clone of `mlcommons/logging` 6.0.0-rc5 at `~/mlperf-logging` on striker: `cd ~/mlperf-logging && PYTHONPATH=. python3 -m mlperf_logging.compliance_checker --usage training --ruleset 6.0.0 <log>`.

## Development / multi-node quirks
- Profiling helpers reference external `utilities/` paths absent from this checkout. Llama 3.1's `dev/slurm_config.yaml` also references an absent `launcher/templates/sbatch.sh.j2`.
- Llama 3.1's dev image requires `ainic_bundle_1.117.5-a-77.tar.gz` in the `code/` build context (or the filename supplied through `AINIC_BUNDLE_FILE`); this bundle is not included.
- `llama31_8b/.../run_with_docker_slurm.sh` stops all running Docker containers on its host before launch. Its multi-node AINIC check requires visible `ionic_*` HCAs with a matching userspace/kernel ABI.
- Preserve the SLURM launcher's explicit rendezvous `-e` flags **after** `${_run_env[@]}`: Docker's last-value-wins behavior prevents the single-node config's `MASTER_ADDR=localhost` from overriding the cluster address.
