# DLRMv4 (HSTU): University of Florida files

This dir is NVIDIA's MLPerf Training v6.1 implementation, unchanged, from the image
`nvcr.io/nvdlfwea/mlperftv61/dlrmv4-arm:20261006` (`GIT_COMMIT_ID=2647c2168340a97ec5b94ae56eb539194bf8a961`), rebuilt for x86 as `dlrmv4-amd_20261006.sif` (see `BUILD_UF.md`).
NVIDIA's guide is `README.md`. Our files are the configs below and these launcher files:

| File | Purpose |
|---|---|
| `launch.sh`, `dlrmv4.sub`, `config_hpg.sh` | HPG B200: `launch.sh` submits `dlrmv4.sub` (apptainer port of NVIDIA's `run.sub`); `config_hpg.sh` has the site settings |
| `make_datadir.sh` | makes our data dir from the shared dataset (one time) |

## What to run

One config per result set. Each config sources NVIDIA's `config_common.sh` and uses the hyperparameters of NVIDIA's `config_GB300_2x4x1024.sh` (GBS 8192) or `config_GB300_16x4x256.sh` (GBS 16384), except the values in the table. All configs set `EVAL_PREFETCH_FACTOR=2`.

| System | Config | GBS | Batch per GPU | LR (dense, sparse) |
|---|---|---|---|---|
| hpg.dgxb200x8.n8 | `config_DGXB200_8x8x128.sh` | 8192 | 128 | 1e-6 |
| hpg.dgxb200x8.n16 | `config_DGXB200_16x8x128.sh` | 16384 | 128 | 2e-6 |
| hpg.dgxb200x8.n32 | `config_DGXB200_32x8x64.sh` | 16384 | 64 | 2e-6 |

## How to run

From this dir:

```bash
NEXP=10 ./launch.sh <config>    # HPG B200 (Slurm); RESERVATION=<name> is optional
```

Options: `NEXP=<n>` (runs per job, default 10, each with a new `$RANDOM` seed), `STAGE_DATA=0` (HPG: read the dataset from `/blue`), `LOGDIR`, `KIT`, `CONT`. The container's own code runs (`/workspace/dlrmv4`); the config reaches it through the environment.

## Why we changed NVIDIA's settings

- GBS and LR (closed division: only the GBS and the LR, 0.5–1.5 × 1e-6 × GBS/8192, may change): 8 nodes use GBS 8192, 16 and 32 nodes GBS 16384. All runs end inside the 24000-step LR warmup.
- `HBM_CAP_GB=170` on B200 (the gin default 260 GiB is for GB300; B200 has 179 GiB).
- `EVAL_PREFETCH_FACTOR=2` (NVIDIA default 8): the number of batches that each eval-loader worker preloads. After each eval, NVIDIA's `_do_eval_db` (`utils.py`) restarts the eval loader at once, so with 8 its workers take CPU from the train data workers. It is not a hyperparameter: the training, the eval set and the AUC do not change.
- Data dir: the v6.1 code reads `<DATADIR>/processed_5b/hstu_cache/`. The shared dataset has the same cache files in `hstu_cache_L4086/` and no parquets. `make_datadir.sh` makes our data dir (symlinks to the cache files and the offline time-window indices). On HPG, `dlrmv4.sub` copies it to node-local `/tmp` on every node before the runs (not timed). `VERIFY_MOUNTS=0`: `expected-mounts.csv` lists the parquets.
- `config_hpg.sh`: Triton cache in node-local `/tmp` (the NFS home gave "Stale file handle" with 8 nodes); `NCCL_TEST=0` (MPI does not start inside apptainer on HPG); NCCL InfiniBand settings.

## Result files

`result_<k>.txt` = run k+1 of the set.
