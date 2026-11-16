# Running AMD dlrmv4 (HSTU on Yambda-5b) PyTorch MLPerf Training Benchmark

This folder is the `2xMI350X_EPYC_9655_pytorch` implementation: 2 nodes, 16 GPUs, global batch
size 16384 (per-GPU batch 1024).

The benchmark trains an HSTU sequential ranking model on the Yambda-5b music-listening dataset.
Given a user's chronologically ordered history of plays, likes and skips, the model ranks which
track the user will genuinely listen to next (the binary `listen_plus` label: played for at least
50% of the track, versus a skip). Training streams over fixed-duration time windows, and every
evaluation scores the strictly future held-out window, so there is no future leakage.

Files generated for this system:

| File | Purpose |
|---|---|
| `config_common.sh` | settings shared by all dlrmv4 systems |
| `config_MI350X_02x08x1024.sh` | this system: 2 node(s) x 8 GPUs x per-GPU batch 1024 |
| `job.sh` | Slurm job file (`sbatch job.sh`) |
| `run_with_docker_slurm.sh` | per-node launcher started by `job.sh` |
| `run_and_time.sh` | timed run inside the container (cache flush, trainer, MLPerf log) |
| `Dockerfile` | container image |
| `download_dataset.sh`, `verify_dataset.sh` | dataset preparation and checksum verification |

## 1. Hardware Requirements

- 2 node(s) with 8 AMD Instinct MI350X GPUs each (16 GPUs). The full system description is `systems/2xMI350X_EPYC_9655_pytorch.json` of this submission.
- GPU: AMD Instinct MI350X; GPU memory: 288 GB; CPU: AMD EPYC 9655; host memory: 3.0 TB; networking: 8 x Broadcom Inc. and subsidiaries BCM57608 400Gb Ethernet (RoCEv2, one per GPU).
- Disk for the dataset: the prepared download (227.41 GB in 19 files) plus the time-window index (about 18.3 GB). Building the dataset from the raw release instead needs about 313 GB of disk at peak (275 GB afterwards) and about 190 GB of host RAM for the cache build.
- RoCEv2 RDMA network between the nodes, with GPUDirect RDMA. The submitted runs used 8 Broadcom bnxt_re HCAs per node; [docs/multi_node_config.md](docs/multi_node_config.md) describes the fabric settings and the userspace RDMA library requirements.

## 2. Software Requirements

- [Docker](https://docs.docker.com/get-docker/), with the user able to run `docker` (or `sudo docker`).
- Passwordless `sudo` on the host for the page-cache flush before each run (`sysctl vm.drop_caches=3`).
- The amdgpu kernel driver with ROCm support on the host.
- Slurm (`sbatch`, `srun`, `scontrol`) for the multi-node launch.
- For the multi-node launch, the DKMS amdgpu driver (it publishes `/sys/module/amdgpu/version`). The configuration then sets `HSA_ENABLE_IPC_MODE_LEGACY=1`, as the submitted runs did.

Software stack of the container the submitted runs used:

| Component | Version |
|---|---|
| base image | `rocm/primus:v26.3` (`sha256:1a02b74a94d82131f3f119a94ccbed45bb5b4cd77a0616cb4e6a29e64a048482`) |
| ROCm | 7.2.1 |
| Python | 3.12 |
| PyTorch | 2.12.0+rocm7.2 |
| torchvision / torchaudio | 0.27.0+rocm7.2 / 2.11.0+rocm7.2 |
| fbgemm_gpu | 2026.6.2 (FBGEMM commit 10b77573, built for gfx950) |
| torchrec | 1.7.0a0+bf55480 (tag v2026.06.01.00) |
| Triton | 3.6.0 |
| RCCL | 2.27.7 |

## 3. Set up

### 3.1 Build the container

The submitted runs used the base image at the digest above. The tag can move, so pull the base by digest and tag it before building:

```bash
docker pull rocm/primus@sha256:1a02b74a94d82131f3f119a94ccbed45bb5b4cd77a0616cb4e6a29e64a048482
docker tag rocm/primus@sha256:1a02b74a94d82131f3f119a94ccbed45bb5b4cd77a0616cb4e6a29e64a048482 rocm/primus:v26.3
docker build -t rocm/amd-mlperf:dlrmv4_training .
```

The `Dockerfile` installs the same PyTorch, torchvision, torchaudio and torchrec versions as the image the submitted runs used. It compiles fbgemm_gpu from source at the same FBGEMM commit (10b77573, for gfx950); the submitted runs used a prebuilt wheel of that commit. A source build reports its build date as the fbgemm_gpu version, so the version string differs from the table above.

`scripts/launch_slurm.sh` checks the container's package versions against its pins and stops on a mismatch; its fbgemm_gpu pin defaults to the prebuilt wheel's `2026.6.2`. When launching an image built from this `Dockerfile` through that script, set `FBGEMM_PIN` to the version the build reports:

```bash
export FBGEMM_PIN=$(docker run --rm rocm/amd-mlperf:dlrmv4_training python -c 'import importlib.metadata as m; print(m.version("fbgemm_gpu_nightly_rocm"))')
```

Make the image available on every node (push it to a registry, or `docker save` / `docker load`), and set `CONT` to its name below.

### 3.2 Download, verify and prepare the dataset

The trainer reads the flat-event cache `processed_5b/hstu_cache_L4086/`, its time-window index and the two item mappings in `shared_metadata/`. Prepare them once on storage that every node can read (or copy them to each node), then point `DATADIR` at that folder.

Download the prepared dataset with the MLCommons R2 downloader (it checks every file's md5):

```bash
export DATADIR=/path/to/dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d "$DATADIR" \
    https://training.mlcommons-storage.org/metadata/dlrmv4_dataset.uri
```

Verify the files against the checksum lists shipped here and prebuild the time-window index (otherwise the first training run builds it during dataset initialization):

```bash
docker run --rm -it -v "$DATADIR":/data/mlperf_dlrm_v4 -v "$(pwd)":/workspace/recommendation -w /workspace/recommendation \
    rocm/amd-mlperf:dlrmv4_training bash
# inside the container:
./verify_dataset.sh
C=$DLRM_DATA_PATH/processed_5b/hstu_cache_L4086
python3 scripts/ts_index_tool.py prebuild --cache-dir "$C" --history-length 4086
python3 scripts/ts_index_tool.py verify   --cache-dir "$C" --history-length 4086
exit
```

`verify_dataset.sh` checks the files against `md5sums_yambda_5b_processed.txt` and `sha256sums_yambda_5b_cache.txt`. The submitted runs read 21 files: the 18 files of processed_5b/hstu_cache_L4086/ (the flat-event cache, including the time-window index ts_indices_L4086_W86400.npy and ts_offsets_L4086_W86400.npy), processed_5b/item_popularity.npy and shared_metadata/{artist,album}_item_mapping.parquet.

To build the dataset from the upstream HuggingFace release instead, run `./download_dataset.sh` in the container (it downloads `yandex/yambda` and preprocesses it), then `./verify_dataset.sh`. The cache is built during the first training run; it is deterministic and verifies against the same checksums.

## 4. Launch training (single node)

This system runs on 2 nodes: launch it with Slurm (section 5).
The single-node system and its launcher are in `../MI350X_EPYC_9655_pytorch/`.

## 5. Launch training (multi-node via Slurm)

Edit `job.sh`: set `CONT`, `DATADIR`, `LOGDIR` and `--partition`, and size `--time` to the number of runs. Check the `[SITE-SPECIFIC]` RDMA settings in `config_MI350X_02x08x1024.sh`. Then submit from this folder:

```bash
sbatch job.sh
NEXP=10 sbatch job.sh      # a submission set
```

`job.sh` allocates 2 nodes exclusively and starts `run_with_docker_slurm.sh` once per node (`srun --ntasks-per-node=1`). On each node it starts the container and runs `run_and_time.sh`; the trainer spawns 8 ranks per node (no torchrun), and the nodes rendezvous at the first node of the allocation, port `20000 + SLURM_JOB_ID % 20000`. `NNODES` and `NODE_RANK` come from `SLURM_NNODES` and `SLURM_NODEID`. Every run uses one seed on all nodes (`ROOT_SEED + 100 x SLURM_ARRAY_TASK_ID + n`). The MLPerf log of run n is `$LOGDIR/<job id>_<n>.log`, written by node 0; each node's output goes to `$LOGDIR/<job id>_<n>_node<k>.out`.

## 6. Run configuration

| Setting | Value |
|---|---|
| nodes x GPUs | 2 x 8 |
| global batch size | 16384 |
| per-GPU batch (`BATCH_SIZE`) | 1024 |
| learning rate (dense and sparse, after warmup) | 0.000002 |
| warmup | 24000 steps from 0.0 |
| optimizers | Adam (dense), RowWiseAdagrad (sparse) |
| eval cadence (`EVAL_EVERY_DATA_PCT`) | 0.001 |
| first evaluation (`SKIP_EVAL_EPOCH_PCT`) | 0.0303 of the epoch |
| seed | random per run |

Everything else, including the model, numerics and kernel selection, is set by `generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin`. The trainer logs that file as the MLPerf `config_filename`; the `config_*.sh` files only set the launch environment.

The submitted runs (`results/<system>/dlrmv4/` of this submission) were launched with `scripts/launch_slurm.sh`, which also distributed the image, staged the dataset to node-local disk and added the RDMA userspace libraries described in `docs/multi_node_config.md`. The launchers in this folder use the same container flags, environment, trainer command and rank layout, and leave those site steps to you. The submitted runs applied no host tuning beyond the page-cache flush and GPU performance level `auto`, so no `runtime_tunables.sh` is included.

## 7. Evaluation

### Quality metric

AUC (window_auc of the listen_plus task) on the held-out future time window.

### Quality target

Eval AUC >= 0.75. The run stops at the first evaluation that reaches it.

### Evaluation frequency

Every 2,293,760 training samples (140 steps at global batch 16384): `EVAL_EVERY_DATA_PCT` = 0.001 of `train_samples` = 2,290,835,423, rounded up to whole steps. Evaluations before 0.0303 of the epoch (step 4237) are skipped (`SKIP_EVAL_EPOCH_PCT`).

### Evaluation thoroughness

Every evaluation scores the whole held-out window: 2,058,204 samples (MLPerf `eval_samples`).
