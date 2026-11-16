# firmus_gb300_ngpu8_gptoss_g1_1012

Firmus, MLPerf Training v6.1, closed division, Available on-premise. Benchmark `gpt_oss_20b`, 8 NVIDIA GB300 on 2 nodes (4 per node) in one GB300 NVL72 NVLink domain.

## Implementation and config

- **Implementation:** `benchmarks/gpt_oss_20b/implementations/nemo_g1_1012/`. This is NVIDIA's NeMo / Megatron-Bridge recipe from the 20261012 release, with its code, Dockerfile and config unchanged. Its [README.md](../nemo_g1_1012/README.md) explains the config, the logged values and how the 20261012 recipe differs from the 20261006 one that G1 ran (G1: Firmus's earlier entry `firmus_gb300_ngpu8_gptoss_g1_base`, which ran that release unchanged and was withdrawn from this submission).
- **Config:** `config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh`, NVIDIA's own GB300 2-node leaf from the same release, unchanged:
  - 2 nodes × 4 GPUs, TP 1, PP 1, CP 1, EP 1;
  - micro-batch 3, global batch 24;
  - LR 5e-4 with 256 warm-up steps, evaluation every 512 steps (12,288 samples);
  - NVIDIA's MXFP8 recipe with FP8 attention, and the HybridEP MoE dispatcher.
- **Launcher:** `run.sub` in the implementation directory is the Slurm launcher that ran the jobs. [run.sub.vs_nvidia1012.diff](run.sub.vs_nvidia1012.diff) lists every difference from NVIDIA's `run.sub` (`run.sub.vendor1012`). It is the same launcher and the same diff as G1's.

## Files in this directory

| file | what |
|---|---|
| `setup.sh` | recreates the exact build context from the implementation directory, builds NVIDIA's Dockerfile on the public base pinned by digest with Megatron-Bridge and Transformer Engine pinned by commit, fails unless the image holds the commits below, and imports the image for Enroot |
| `init_datasets.sh` | NVIDIA's dataset download (`data_scripts/download.sh`) and md5 check |
| `run_and_time.sh` | one training run: sets the submission identity, sources the config leaf, submits `run.sub` |
| `firmus_gb300_ngpu8_gptoss_g1_1012_nemo_g1_1012.json` | system-implementation metadata (image, software versions, config) |
| `BUILD_CONTEXT.sha256` | sha256 of the 44 files of the build context |
| `run.sub.vs_nvidia1012.diff` | the launcher's differences from NVIDIA's `run.sub` |
| `redact_logs.sh` | the log post-processing: makes each `result_N.txt` from its run's raw log (see Results) |

## The image that produced the results

- **NVIDIA's build of the packaged Dockerfile.** The ten runs used the image NVIDIA built from this recipe's Dockerfile, unchanged:
  - squashfs sha256 `3b3a1c6669bb3cf25bc5027d01b101d9673c30c169d8d354f0653f38831397a6`, 25,965,891,584 bytes. The driver checked this sha256 before the smoke and attempt 1, and every submit checked its size; it was hashed again on 2026-10-14 after the set and matched.
  - Provenance only: NVIDIA distributed it as the early-access image `nvcr.io/nvdlfwea/mlperftv61/gpt_oss_20b-arm:20261012`. The system description names only public components.
- **Software in the image,** all public (SR-15), the same commits as G1's image:
  - base `nvcr.io/nvidia/pytorch:26.09-py3`: CUDA 13.4.1, cuDNN 9.26.0.51, NCCL 2.31.2, cuBLAS 13.8.0.4;
  - Transformer Engine 2.19.0 (`5e52befd`, tag `v2.19`);
  - Megatron-Bridge `ea5dded8` (tag `26.09-alpha.rc1`);
  - Megatron-Core 0.20.0 (`f6c33bde`, the Megatron-Bridge submodule);
  - DeepEP 1.2.1 (`58397eac`, branch `hybrid-ep`), the HybridEP MoE dispatcher;
  - `mlperf-logging` `8a0cc055` (tag `6.1.0-rc2`) and `mlperf-common` `b5014f3c` (tag `training-v6.1-rc0`).
- **Not rebuilt.** Firmus has not rebuilt this Dockerfile and compared the result with the image that ran (open by decision, as for G1). What was checked, by reading the squashfs without starting a container:
  - the commit records (`/TE_COMMIT_HASH.env`, `/MBRIDGE_COMMIT_HASH.env`, `/MCORE_COMMIT_HASH.env`, DeepEP's checkout `HEAD`, and pip's `direct_url.json` of `mlperf-logging` and `mlperf-common`) are the commits above;
  - the recipe directory `/workspace/llm` is, apart from NVIDIA's 12 build-output files, exactly the 44 files of `BUILD_CONTEXT.sha256`, with the same content and modes as the implementation directory;
  - the Dockerfile and `requirements.txt` are byte-identical to the 20261006 release's. For G1, Firmus checked that release's image statically on 2026-10-10: every pin of the Dockerfile and `requirements.txt` was what the image held, and the source trees the Dockerfile compiles (Transformer Engine with its submodules, and DeepEP) matched the public commits file for file (8,212 of 8,212 tracked files). That check was of the 20261006 image, not this one;
  - inside the running container, every run hashed 36 Transformer Engine, Megatron-LM, cuDNN-frontend and recipe files (`:::STACK_SHA256` lines, including the three `opt_kernels/` files); all ten logs equal the expectation computed from the squashfs.
  - This does not show that this image's compiled binaries (Transformer Engine 2.19 and DeepEP, which is on the training path) come from those sources with the Dockerfile's flags; only a rebuild would.
- **`submission_platform`.** The logs record `submission_platform = 2xfirmus_ngpu8_ngc26.09_nemo`: the node count and the launcher's platform string for NGC 26.09 NeMo. It is not the `system_desc_id`; the system is `firmus_gb300_ngpu8_gptoss_g1_1012`.

### Rebuilding from the exact build context

The implementation directory holds every file of the build context, byte-identical, plus three Firmus files, one of them under the build's name. Its [README](../nemo_g1_1012/README.md) ("This directory compared with the build context") lists the differences. [setup.sh](setup.sh) does the following, and also checks the commits in the built image; by hand:

```bash
mkdir ctx && cp -a ../nemo_g1_1012/. ctx/ && cd ctx
mv README.md ../README.firmus.md                       # the Firmus README, not in the build
mv run.sub ../run.sub.firmus                           # the launcher that ran; host-side, not in the build
mv stack_files.txt ..                                  # the STACK list; read on the host, not in the build
mv README_NVIDIA.md README.md                          # the build's README.md, renamed
mv run.sub.vendor1012 run.sub                          # the build's run.sub (NVIDIA's)
sha256sum -c ../BUILD_CONTEXT.sha256 | grep -v ': OK$'; find . -type f | wc -l   # no failures; 44 files
docker build --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6 \
             --build-arg MBRIDGE_REVISION=ea5dded83078cf7e61ef91960273053cf564d5ac \
             --build-arg TE_REVISION=5e52befd5262c06289106338c308079d6adb391f -t gpt_oss_20b_g1_1012 .
```

There is no `.dockerignore`: the recipe directory in NVIDIA's image has none, and the context contains no build output that would need one.

No rebuild reproduces the docker image ID or the squashfs sha256 of the image that ran: image layers carry build times, `pip` records install times, bytecode carries source timestamps, compiled extensions are not byte-identical from one build to the next, and the squashfs depends on how it was packed. `setup.sh` checks the commits.

## Launch

```bash
TAG=gpt_oss_20b_g1_1012 SQSH=$PWD/gpt_oss_20b_g1_1012.sqsh bash setup.sh
DATADIR=/path/to/data bash init_datasets.sh
CONT=$PWD/gpt_oss_20b_g1_1012.sqsh DATADIR=/path/to/data LOGDIR=/path/to/results SEED=1443849258 bash run_and_time.sh
```

`run_and_time.sh` submits:

```bash
sbatch -N 2 --time=$WALLTIME --exclusive --mem=0 --gpus-per-node=4 --job-name=mlperf-gpt_oss_20b-2x4 run.sub
```

It does so after sourcing `config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh`. Inside the container on every rank, `run.sub` runs NVIDIA's `run_and_time.sh`, then the MLPerf compliance checker on the log. The runs also pinned the partition and the two nodes (`SBATCH_EXTRA` passes such options).

**Hosts.** The jobs ran on two GB300 compute trays, each with 2 Grace CPUs and 4 GB300: the same two nodes as G1 and as the llama31_8b 20261012 set, which ran just before on them. All ten runs used them exclusively, with the same image, GPU driver 580.105.08 and VBIOS 97.10.4A.00.1A, both nodes in one NVLink clique (each `slurm-*.out`). A read-only probe of both nodes on 2026-10-14 found them identical in CPU, caches, NUMA, GPU clocks and memory, VBIOS, driver and kernel.
- The dataset was on node-local NVMe, as for G1.
- Before each run, `run.sub` cleared the page cache on both nodes and would have failed the job if it could not; each `slurm-*.out` records this.
- No prolog NCCL test was run (`NCCL_TEST=0`).

## Results

`results/firmus_gb300_ngpu8_gptoss_g1_1012/gpt_oss_20b/result_0.txt` … `result_9.txt` are the complete logs of the ten runs, in run order. Each holds the framework output and every MLLOG line.
- Each is its run's raw log after [redact_logs.sh](redact_logs.sh), which redacts internal storage paths, as training rules TR-11.2 allows.
- On lines that are not MLLOG lines, every path prefix made of `/shared` and the next two directory names becomes `[internal storage]`. That changes 6 lines per log: two `:::DLPAL` lines, the two `run.sub` lines that read `stack_files.txt`, `SLURM_HOSTFILE` and the `srun --container-mounts` line. The two `:::DLPAL` lines keep the rest of the container path, which ends in the image's file name.
- MLLOG lines are copied unchanged, so every key and value the MLPerf checkers read is the raw log's. The unredacted raw logs are kept and will be provided on request.

| result | Slurm job | seed | evals to 3.34 | time to train (min) |
|---|---|---|---|---|
| result_0 | 22270 | 1443849258 | 18 | 70.59 |
| result_1 | 22271 | 1599531824 | 18 | 70.57 |
| result_2 | 22272 | 1324113487 | 18 | 70.60 |
| result_3 | 22274 | 655146381 | 18 | 70.56 |
| result_4 | 22280 | 2021125161 | 18 | 70.61 |
| result_5 | 22281 | 1512914239 | 16 | 62.74 |
| result_6 | 22297 | 1614451486 | 17 | 66.66 |
| result_7 | 22313 | 786224576 | 17 | 66.68 |
| result_8 | 22365 | 653603880 | 18 | 70.56 |
| result_9 | 22456 | 1485274712 | 18 | 70.59 |

- All ten runs reached the target (`run_stop` status success), and each passed the MLPerf compliance checker (6.1.0).
- Olympic mean: **69.600 min** (`result_summarizer`: 69.60006458333334 min; the fastest, result_5, and the slowest, result_4, dropped).
- RCP check (6.1.0, pruned RCPs): PASSED. There is no RCP at global batch 24; the checker interpolates between the RCPs at batch 16 and 64 to an RCP mean of 212,309 samples and a minimum of 205,774. The submission's olympic mean is 218,112 samples, above the RCP mean, so no normalisation applies and no `scaling.json` ships.

## Seeds and the set

- The fourteen seeds (the ten above, then four spares: 66984861, 340122104, 77881054, 1387708106) and the smoke seed were drawn from `/dev/urandom` and written to the set's seeds files at 21:16:42 AEDT on 2026-10-13. The set's seed ledger, with the attempt policy below, records the same time ("Written 2026-10-13T21:16:42+11:00 ... before any job of this set was submitted").
- The attempt policy: the scoring set is attempts 1-10 in submission order; at most one may fail, and it is kept and trimmed as the slowest; a second failure invalidates the set; a job that ends before `run_start` is not an attempt.
- The set was queued behind the llama31_8b 20261012 set on the same two nodes. Its driver started at 03:53 AEDT on 2026-10-14; the smoke (job 22269, seed 500814723, not scoring) passed at 04:10, before attempt 1.
- All ten attempts ran in order, from 04:10 to 17:37 AEDT on 2026-10-14; none failed and no spare was used. Every seed appears exactly once in the logs, and none of the set's or the smoke's seeds appears in any other package of this submission.
