# firmus_gb300_ngpu8_llama8b_l1_1012

Firmus, MLPerf Training v6.1, closed division, Available on-premise. Benchmark `llama31_8b`, 8 NVIDIA GB300 on 2 nodes (4 per node) in one GB300 NVL72 NVLink domain.

## Implementation and config

- **Implementation:** `benchmarks/llama31_8b/implementations/nemo_l1_1012/`. This is NVIDIA's NeMo / Megatron-Bridge recipe from the 20261012 release, with its code, Dockerfile and config unchanged. Its [README.md](../nemo_l1_1012/README.md) explains the config and the logged values.
- **Config:** `config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh`, NVIDIA's own GB300 2-node leaf from the same release, unchanged:
  - 2 nodes × 4 GPUs, TP 1, PP 1, CP 1;
  - micro-batch 2, global batch 16, LR 4e-4 with 16 warm-up steps;
  - NVIDIA's NVFP4 recipe with FP8 attention, with NVIDIA's Transformer Engine settings for this benchmark.
- **Launcher:** `run.sub` in the implementation directory is the Slurm launcher that ran the jobs. [run.sub.vs_nvidia1012.diff](run.sub.vs_nvidia1012.diff) lists every difference from NVIDIA's `run.sub` (`run.sub.vendor1012`). It is the same launcher and the same diff as L1's (L1: Firmus's earlier entry `firmus_gb300_ngpu8_llama8b_l1_base`, which ran NVIDIA's 20261006 release and was withdrawn from this submission).

**L1 and L1_1012.** Both run NVIDIA's llama31_8b recipe unchanged, with the same Dockerfile, launcher, recorded commits and two nodes. L1 ran NVIDIA's 20261006 release, which has no GB300 leaf, under a renamed copy of its VR200 leaf; L1_1012 runs NVIDIA's 20261012 release with its own GB300 leaf.

## Files in this directory

| file | what |
|---|---|
| `setup.sh` | recreates the exact build context from the implementation directory, builds NVIDIA's Dockerfile on the public base pinned by digest with its git revisions pinned to commits, checks the built image's commits, and imports the image for Enroot |
| `init_datasets.sh` | NVIDIA's dataset download (`data_scripts/download_8b.sh`) and md5 check |
| `run_and_time.sh` | one training run: sets the submission identity, sources the leaf, submits `run.sub` |
| `firmus_gb300_ngpu8_llama8b_l1_1012_nemo_l1_1012.json` | system-implementation metadata (image, software versions, config) |
| `BUILD_CONTEXT.sha256` | sha256 of the 48 files of the build context |
| `run.sub.vs_nvidia1012.diff` | the launcher's differences from NVIDIA's `run.sub` |
| `redact_logs.sh` | the log post-processing: makes each `result_N.txt` from its run's raw log (see Results) |

## The image that produced the results

- **NVIDIA's build of the packaged Dockerfile.** The ten runs used the image NVIDIA built from this recipe's Dockerfile, unchanged:
  - squashfs sha256 `97a310af019f1006d9aa60196ce81122795852d26ba979d80bbe411e0b8e0c2b`, 26,031,730,688 bytes. The driver checked this sha256 before the smokes and attempt 1, and every submit checked its size; it was hashed again on 2026-10-14 after the set and matched.
  - Provenance only: NVIDIA distributed it as the early-access image `nvcr.io/nvdlfwea/mlperftv61/llama31_8b-arm:20261012`. The system description names only public components.
- **Software in the image,** all public (SR-15), the same commits as L1's image:
  - base `nvcr.io/nvidia/pytorch:26.09-py3`: CUDA 13.4.1, cuDNN 9.26.0.51, NCCL 2.31.2, cuBLAS 13.8.0.4;
  - Transformer Engine 2.20.0 (`5b28d7aa`, tag `v2.20.0`);
  - Megatron-Bridge `ea5dded8` (tag `26.09-alpha.rc1`);
  - Megatron-Core 0.20.0 (`f6c33bde`, the Megatron-Bridge submodule);
  - `mlperf-logging` `8a0cc055` (tag `6.1.0-rc2`) and `mlperf-common` `b5014f3c` (tag `training-v6.1-rc0`).
- **Not rebuilt.** Firmus has not rebuilt this Dockerfile and compared the result with the image that ran (open by decision). What was checked, by reading the squashfs without starting a container:
  - the commit records (`/TE_COMMIT_HASH.env`, `/MBRIDGE_COMMIT_HASH.env`, `/MCORE_COMMIT_HASH.env`, and pip's `direct_url.json` of `mlperf-logging` and `mlperf-common`) are the commits above;
  - the recipe directory `/workspace/llm` is, apart from NVIDIA's 12 build-output files, exactly the 48 files of `BUILD_CONTEXT.sha256`, with the same content and modes as the implementation directory;
  - the Dockerfile and `requirements.txt` are byte-identical to the 20261006 release's. For L1, Firmus rebuilt that Dockerfile, unchanged, from the same pinned public base on 2026-10-10 and compared the rebuild's squashfs with NVIDIA's 20261006 image entry by entry: the same 283 Python packages and versions and the same commits; cuDNN, NCCL, cuBLAS, apex and all 46 recipe files byte-identical; identical GPU machine code in all cubins of the extensions the Dockerfile compiles; no difference in any performance-relevant component. That comparison was not repeated for this image.
  - This does not show that this image's compiled binaries come from those sources with the Dockerfile's flags; only a rebuild would.
- **`submission_platform`.** The logs record `submission_platform = 2xfirmus_ngpu8_ngc26.09_nemo`: the node count and the launcher's platform string for NGC 26.09 NeMo. It is not the `system_desc_id`; the system is `firmus_gb300_ngpu8_llama8b_l1_1012`.

### Rebuilding from the exact build context

The implementation directory holds every file of the build context, byte-identical, plus two Firmus files, one of them under the build's name. Its [README](../nemo_l1_1012/README.md) ("This directory compared with the build context") lists the differences. [setup.sh](setup.sh) does the following; by hand:

```bash
mkdir ctx && cp -a ../nemo_l1_1012/. ctx/ && cd ctx
mv README.md ../README.firmus.md              # the Firmus README, not in the build
mv run.sub ../run.sub.firmus                  # the launcher that ran; host-side, not in the build
mv README_NVIDIA.md README.md                 # the build's README.md, renamed
mv run.sub.vendor1012 run.sub                 # the build's run.sub (NVIDIA's)
sha256sum -c ../BUILD_CONTEXT.sha256 | grep -v ': OK$'; find . -type f | wc -l   # no failures; 48 files
docker build --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.09-py3@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6 \
             --build-arg MBRIDGE_REVISION=ea5dded83078cf7e61ef91960273053cf564d5ac \
             --build-arg TE_REVISION=5b28d7aa2bd639ff0541356cc025f62954f96ce3 \
             -t llama31_8b_l1_1012 .
```

There is no `.dockerignore`: the recipe directory in NVIDIA's image has none, and the context contains no build output that would need one.

**Commits.** The two revision build-args are the Dockerfile's own (`ARG MBRIDGE_REVISION`, `ARG TE_REVISION`). Their defaults, the tags `26.09-alpha.rc1` and `v2.20.0`, name the same two commits; passing the commits means a rebuild cannot pick up a tag that has moved. Megatron-Core is the Megatron-Bridge commit's submodule. `mlperf-logging` and `mlperf-common` are named by tag in NVIDIA's `requirements.txt`, which is used unchanged. After the build, `setup.sh` reads the five commit records from the image and stops before the Enroot import unless all five equal the commits of the image that ran. With the commit build-arg, the image's `CUSTOM_TE_REVISION` variable holds `5b28d7aa…` where the image that ran holds `v2.20.0`; nothing in the recipe reads it.

No rebuild reproduces the docker image ID or the squashfs sha256 of the image that ran: image layers carry build times, `pip` records install times, bytecode carries source timestamps, compiled extensions are not byte-identical from one build to the next, and the squashfs depends on how it was packed.

## Launch

```bash
TAG=llama31_8b_l1_1012 SQSH=$PWD/llama31_8b_l1_1012.sqsh bash setup.sh
DATADIR=/path/to/data bash init_datasets.sh
CONT=$PWD/llama31_8b_l1_1012.sqsh DATADIR=/path/to/data LOGDIR=/path/to/results SEED=410147210 bash run_and_time.sh
```

`run_and_time.sh` submits:

```bash
sbatch -N 2 --time=$WALLTIME --exclusive --mem=0 --gpus-per-node=4 --job-name=mlperf-llama31_8b-2x4 run.sub
```

It does so after sourcing `config_GB300_2x4x2xtp1pp1cp1_8b_fp4.sh`. Inside the container on every rank, `run.sub` runs NVIDIA's `run_and_time.sh`, then the MLPerf compliance checker on the log. The runs also pinned the partition and the two nodes (`SBATCH_EXTRA` passes such options).

**Hosts.** The jobs ran on two GB300 compute trays, each with 2 Grace CPUs and 4 GB300: the same two nodes as L1. All ten runs used them exclusively, with the same image, GPU driver 580.105.08 and VBIOS 97.10.4A.00.1A, both nodes in one NVLink clique (each `slurm-*.out`). A read-only probe of both nodes on 2026-10-14 found them identical in CPU, caches, NUMA, GPU clocks and memory, VBIOS, driver and kernel.
- The dataset was on node-local NVMe (`DATADIR=/tmp/mlperf`, as for L1); its md5 sums were checked on both nodes before the set.
- Before each run, `run.sub` cleared the page cache on both nodes and would have failed the job if it could not; each `slurm-*.out` records this.
- No prolog NCCL test was run (`NCCL_TEST=0`).

## Results

`results/firmus_gb300_ngpu8_llama8b_l1_1012/llama31_8b/result_0.txt` … `result_9.txt` are the complete logs of the ten runs, in run order. Each holds the framework output and every MLLOG line.
- Each is its run's raw log after [redact_logs.sh](redact_logs.sh), which redacts internal storage paths, as training rules TR-11.2 allows.
- On lines that are not MLLOG lines, every path prefix made of `/shared` and the next two directory names becomes `[internal storage]`. That changes 4 lines per log: two `:::DLPAL` lines, `SLURM_HOSTFILE` and the `srun --container-mounts` line. The two `:::DLPAL` lines keep the rest of the container path, which ends in the image's file name.
- MLLOG lines are copied unchanged, so every key and value the MLPerf checkers read is the raw log's. The unredacted raw logs are kept and will be provided on request.

| result | Slurm job | seed | evals to 3.3 | time to train (min) |
|---|---|---|---|---|
| result_0 | 22164 | 410147210 | 15 | 64.66 |
| result_1 | 22187 | 366536338 | 14 | 60.38 |
| result_2 | 22192 | 1916308639 | 14 | 60.35 |
| result_3 | 22196 | 1978309967 | 14 | 60.35 |
| result_4 | 22205 | 1492468934 | 14 | 60.35 |
| result_5 | 22214 | 1122020640 | 14 | 60.34 |
| result_6 | 22259 | 168724501 | 16 | 69.25 |
| result_7 | 22265 | 576229135 | 14 | 60.33 |
| result_8 | 22266 | 980230703 | 14 | 60.31 |
| result_9 | 22268 | 297000207 | 14 | 60.32 |

- All ten runs reached the target (`run_stop` status success), and each passed the MLPerf compliance checker (6.1.0).
- Olympic mean: **60.884 min** (`result_summarizer`: 60.88363125 min; the fastest, result_8, and the slowest, result_6, dropped).
- RCP check (6.1.0, pruned RCPs): PASSED. The RCP at global batch 16 applies directly: RCP mean 161,109 samples, minimum 157,920. The submission's olympic mean is 173,568 samples, above the RCP mean, so no normalisation applies and no `scaling.json` ships.

## Seeds and the set

- The fourteen seeds (the ten above, then four spares: 716430481, 466251745, 139623132, 1276861782) were drawn from `/dev/urandom` and written to the set's seeds file by 16:10:30 AEDT on 2026-10-13, before the first smoke (16:12) and attempt 1. The set's seed ledger, with the attempt policy below, records the same time ("Written 2026-10-13T16:10:30+11:00 ... before any job of this set was submitted").
- The attempt policy: the scoring set is attempts 1-10 in submission order; at most one may fail, and it is kept and trimmed as the slowest; a second failure invalidates the set; a job that ends before `run_start` is not an attempt.
- Two smokes ran before attempt 1, with their own seeds. Smoke 1 (job 22161, seed 550204033) ran cleanly, but the driver stopped at 16:23 on two of its own log checks (`EXPECT` lines) that were written wrongly; the checks were corrected and recorded in the ledger (16:24:59), a new smoke seed was drawn, and smoke 2 (job 22162, seed 1969149985) passed at 16:39. Neither smoke is an attempt or scores.
- All ten attempts ran in order, from 16:39 AEDT on 2026-10-13 to 03:52 on 2026-10-14; none failed and no spare was used. Every seed appears exactly once in the logs, and none of the set's or the smokes' seeds appears in any other package of this submission.
