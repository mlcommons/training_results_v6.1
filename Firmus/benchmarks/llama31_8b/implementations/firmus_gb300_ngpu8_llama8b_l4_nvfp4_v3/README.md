# firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3

Firmus, MLPerf Training v6.1, closed division, Available on-premise. Benchmark `llama31_8b`, 8 NVIDIA GB300 on 2 nodes (4 per node) in one GB300 NVL72 NVLink domain.

## Implementation and config

- **Implementation:** `benchmarks/llama31_8b/implementations/nemo_l4_nvfp4_v3/`. This is NVIDIA's NeMo / Megatron-Bridge recipe from the 20261006 release, rebuilt from the public NGC PyTorch 26.09 base with Firmus kernels compiled from source. It is L4 v2 (`nemo_l4_nvfp4_v2`, an earlier Firmus entry, withdrawn from this submission) with one change: no forward parameter gather overlaps compute (`AK_FP4G_NOOVERLAP=1`). Its [README.md](../nemo_l4_nvfp4_v3/README.md) explains the changes, and [KERNELS.md](../nemo_l4_nvfp4_v3/KERNELS.md) is the kernel supplementary information.
- **Config:** `config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh`. It is NVIDIA's `config_GB300_2x4x2xtp1pp1cp1_8b.sh`, sourced unchanged, plus the `AK_*` switches of the Firmus kernels (L4 v2's and `AK_FP4G_NOOVERLAP=1`):
  - 2 nodes × 4 GPUs, TP 1, PP 1, CP 1;
  - micro-batch 2, global batch 16;
  - NVIDIA's NVFP4 recipe.
- **Launcher:** `run.sub` in the implementation directory is the Slurm launcher that ran the jobs. [run.sub.vs_nvidia1006.diff](run.sub.vs_nvidia1006.diff) lists every difference from NVIDIA's `run.sub` (`run.sub.vendor1006`).

## Files in this directory

| file | what |
|---|---|
| `setup.sh` | builds the container from the implementation directory's Dockerfile and imports it for Enroot |
| `init_datasets.sh` | NVIDIA's dataset download (`data_scripts/download_8b.sh`) and md5 check |
| `run_and_time.sh` | one training run: sets the submission identity, sources the L4 leaf, submits `run.sub` |
| `firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3_nemo_l4_nvfp4_v3.json` | system-implementation metadata (image, software versions, config, kernels) |
| `BUILD_CONTEXT.sha256` | sha256 of the 142 files the scored image was built from |
| `run.sub.vs_nvidia1006.diff` | the launcher's differences from NVIDIA's `run.sub` |
| `redact_logs.sh` | the log post-processing: makes each `result_N.txt` from its run's raw log (see Results) |

## The image that produced the results

- **Build.** Built on 2026-10-14 from `nemo_l4_nvfp4_v3/Dockerfile` alone, with build context = `BUILD_CONTEXT.sha256`:
  - base `nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6`;
  - docker image `sha256:bc859ba89be7…`;
  - squashfs sha256 `1d55d6d50d540c7688046dc1cb0b09ae13ef5745ec0ae1c18ec08f236d484942`.
- **Software:**
  - Transformer Engine 2.20.0 (`5b28d7aa`);
  - Megatron-Bridge `ea5dded8` (26.09-alpha.rc1);
  - Megatron-Core `f6c33bde` (0.20.0).
- **Firmus build records** (`/FIRMUS_AK.env`):
  - `FIRMUS_AK_BUILD_SHA256=d041cf2e1a22bf4551a464a56f1e3000ceaaa43ef7bcc5e79d611a8e84ec65ec`;
  - stack record `024dd9bc…`.

  Each result log prints both, followed by `[auto_kernel banner] MATCH`.

### Rebuilding from the exact build context

The implementation directory differs from the build context in five files and adds one directory, `KERNELS/` (the per-item write-ups), all listed in its [README](../nemo_l4_nvfp4_v3/README.md) ("This directory compared with the build context"). None is code that runs. This recreates the build context from the package:

```bash
cp -a ../nemo_l4_nvfp4_v3 ctx && cd ctx
mv README_NVIDIA.md README.md                 # the build's README.md, renamed
mv run.sub.vendor1006 run.sub                 # the build's run.sub (NVIDIA's); the launcher that ran is host-side
mv KERNELS.md KERNELS ../                     # written after the build
cat > .dockerignore <<'EOF'
# Firmus L4: no build output enters the build context.  Every binary in the image is compiled by this Dockerfile from
# source (KERNEL_CODE_IN_DOCKERFILE.md; checklist G3 / SR-15).  This also keeps out the build/ and *.egg-info left in
# embedding_lib by NVIDIA's own image build (the recipe directory was read from that image), so section 4's
# `pip install .` of embedding_lib compiles embedding_funcs.cu itself instead of finding a prebuilt object.
**/build/
**/*.egg-info/
**/__pycache__/
**/*.pyc
**/*.o
**/*.so
**/.ninja_deps
**/.ninja_log
EOF
sha256sum -c ../BUILD_CONTEXT.sha256 | grep -v ': OK$'
```

The check passes for all 142 files and prints nothing, and no other file is left in `ctx`.

A rebuild from this context runs the same build steps on the same sources, but it does not reproduce the docker image ID or the squashfs sha256: image layers carry timestamps, and nvcc output is not byte-identical from one build to the next. For the same reason `FIRMUS_AK_BUILD_SHA256`, which hashes the compiled Firmus extensions, changes. The stack record (`024dd9bc…`) hashes only patched Python files and depends on the sources alone.

## Launch

```bash
TAG=llama31_8b_l4 SQSH=$PWD/llama31_8b_l4.sqsh bash setup.sh
DATADIR=/path/to/data bash init_datasets.sh
CONT=$PWD/llama31_8b_l4.sqsh DATADIR=/path/to/data LOGDIR=/path/to/results SEED=1198273789 bash run_and_time.sh
```

`run_and_time.sh` submits:

```bash
sbatch -N 2 --time=$WALLTIME --exclusive --mem=0 --gpus-per-node=4 run.sub
```

It does so after sourcing `config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh`. Inside the container on every rank, `run.sub` runs NVIDIA's `run_and_time.sh`, then the MLPerf compliance checker on the log.

**Hosts.** The jobs ran on two GB300 compute trays, each with 2 Grace CPUs and 4 GB300. The dataset was on node-local NVMe: on both nodes `DATADIR=/tmp/mlperf` is on the root file system, ext4 on an NVMe partition, and no tmpfs or ramfs is mounted at or under `/tmp` (`findmnt -T`, `df -T`; recorded during the runs, on file systems unchanged since the nodes booted, on 2026-10-06 and 2026-09-10). Before each run, `run.sub` cleared the page cache on both nodes and would have failed the job if it could not; each `slurm-*.out` records this. No prolog NCCL test was run (`NCCL_TEST=0`).

## Results

`results/firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/llama31_8b/result_0.txt` … `result_9.txt` are the complete logs of the ten runs, in run order. Each holds the framework output and every MLLOG line.
- Each is its run's raw log after [redact_logs.sh](redact_logs.sh), which redacts internal storage paths, as training rules TR-11.2 allows.
- On lines that are not MLLOG lines, every path prefix made of `/shared` and the next two directory names becomes `[internal storage]`. That changes 4 lines per log: two `:::DLPAL` lines, `SLURM_HOSTFILE` and the `srun --container-mounts` line.
- MLLOG lines are copied unchanged, so every key and value the MLPerf checkers read is the raw log's. The unredacted raw logs are kept and will be provided on request.

| result | Slurm job | seed | evals to 3.3 | time to train (min) |
|---|---|---|---|---|
| result_0 | 22462 | 617231358 | 15 | 54.21 |
| result_1 | 22475 | 1027662530 | 14 | 50.56 |
| result_2 | 22490 | 1434106694 | 14 | 50.56 |
| result_3 | 22523 | 1794876020 | 15 | 54.19 |
| result_4 | 22526 | 1516803325 | 14 | 50.60 |
| result_5 | 22528 | 1715502533 | 14 | 50.55 |
| result_6 | 22533 | 1058952939 | 14 | 50.59 |
| result_7 | 22545 | 954373187 | 15 | 54.25 |
| result_8 | 22556 | 66254591 | 15 | 54.27 |
| result_9 | 22560 | 1058566433 | 15 | 54.31 |

- All ten runs reached the target (`run_stop` status success), and each passed the MLPerf compliance checker (6.1.0).
- Olympic mean: **52.402 min** (the fastest and slowest dropped).
- RCP check (6.1.0, pruned RCPs): PASSED, submission mean 178,176 samples.
- The runs ran from 2026-10-14 16:30 to 2026-10-15 02:29 AEDT.

## Seeds and the set

- The ten seeds above were drawn from `/dev/urandom` on 2026-10-14 at 16:08 AEDT, with four spares, after the image's tests and smoke and before the first run.
- The set was declared in advance as one set of ten runs on these two nodes with this image and config. The rules for runs that end before `run_start` and for failed runs were also fixed in advance.
- The seeds and the declaration were committed to Firmus's seed ledger and pushed before attempt 1 (attempt 1 was submitted at 16:30 AEDT).
- All ten ran in order, none failed, and no spare was used. Every seed appears exactly once in the logs.
