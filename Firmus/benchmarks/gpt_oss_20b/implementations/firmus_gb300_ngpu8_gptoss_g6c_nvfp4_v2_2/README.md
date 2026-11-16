# firmus_gb300_ngpu8_gptoss_g6c_nvfp4_v2_2

NVFP4 "V7" recipe (gpt_oss_20b, 8x NVIDIA GB300, 2 nodes x 4 GPUs) on NVIDIA's 20261012 reference
model: no attention or expert biases, SiLU-gated experts, vanilla softmax without sinks, QK RMSNorm,
and the reference evaluation set. The kernels that assumed the earlier model were reworked for it
([`KERNELS.md`](KERNELS.md), section 10); NVIDIA's fused QK-norm + RoPE (`opt_kernels/`) is included
unchanged.

## Implementation

Code: `benchmarks/gpt_oss_20b/implementations/nemo_g6c_nvfp4_v2_2/` (recipe description in `README_nvfp4_v7.md`)
Config: `benchmarks/gpt_oss_20b/implementations/nemo_g6c_nvfp4_v2_2/config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh`

Kernel and numerical changes: [`KERNELS.md`](KERNELS.md). Padded expert entries: [`verification/padding_zero/`](verification/padding_zero/README.md). Optimizer equivalence test: [`verification/optimizer_ab/`](verification/optimizer_ab/README.md). All three are kept outside the Docker build context so that `nemo_g6c_nvfp4_v2_2/` builds the image used for the results.

The V7 patch's helper-module loader has a fallback directory outside the image (`NVFP4_V7_OV_DIR`, default `/opt/v7ov`) that the runs never use: every helper they load is installed in the image, the variable is not set, and no such directory is mounted. [`KERNELS.md`](KERNELS.md) (introduction) gives the details. It is accepted for this set.

Evaluation uses 1024 validation samples (`VAL_SAMPLES=1024`) every 512 steps; "Full validation set every 512 steps (12288 sequences)" in `README_nvfp4_v7.md` should read "1024 validation samples every 512 steps".

Before `run_start`, `callback_logging.py` builds the dataset's `.npy` document, sample and shuffle indices from the `.idx` metadata (document count and sizes) and reads them once into the page cache. It does not read training tokens.

## Build the container

`setup.sh` builds the image from `nemo_g6c_nvfp4_v2_2/Dockerfile` and imports it as an enroot squashfs:

```bash
IMAGE=firmus/gpt-oss-20b-nvfp4-g6c:v61v2 SQSH=/path/to/containers/firmus+gpt-oss-20b-nvfp4-g6c-v2+v61.sqsh \
DOCKER_BUILD_ARGS="--build-arg BUILD_JOBS=48" ./setup.sh
```

Image used for the results: ID `sha256:d69e3e51f4a18273d977c9c0767d7dae680f73c0bde166995ec0101120076869`, squashfs sha256 `fd815c589765300f2f559f6635c791b3a16f59aed319524abd64f789399aea7a`.

## Dataset

`init_datasets.sh` downloads the preprocessed C4 dataset and tokenizer with the MLCommons reference scripts and verifies the checksums.

## Launch

`run_and_time.sh` sources the config, draws the seed base once from `/dev/urandom` and submits 10 runs as one Slurm job. Before each run, `run.sub` flushes the page cache on every node with `sudo -n /sbin/sysctl vm.drop_caches=3`, aborts the job if the flush fails on any node, and logs `cache_clear` with the per-node `Cached:` values before and after.

```bash
CONT=/path/to/containers/firmus+gpt-oss-20b-nvfp4-g6c-v2+v61.sqsh DATADIR=/path/to/data LOGDIR=/path/to/logs \
NODES=node1,node2 ./run_and_time.sh
```

## Results

- `result_0.txt` .. `result_9.txt` are runs 1 to 10 of Slurm job 22801 (`NEXP=10`), in launch
  order, on two GB300 compute trays with the image above. No run was
  dropped, reordered or replaced. An earlier set with the same package and image (job 22794) was
  cancelled during its first run, whose loss had collapsed to ln(vocab) from the first logged step;
  no run of it is used or combined with this set.
- Seeds: `run_and_time.sh` drew the seed base 1654593198 once from `/dev/urandom` and passed it as
  `SEED_BASE`; run *i* uses `SEED_BASE - 1 + i`, so the seeds are 1654593198 .. 1654593207, each
  logged once with the `seed` key.
- Each result file is its run's raw log after [`redact_logs.sh`](redact_logs.sh), which replaces
  internal storage path prefixes (the first two directories under `/shared/`) on non-MLLOG lines with `[internal storage]`, as
  training rules TR-11.2 allows. MLLOG lines are copied unchanged, so every key and value the
  MLPerf checkers read is the raw log's; [`verification/redaction_check/`](verification/redaction_check/)
  compares every result file with its raw log. The unredacted raw logs are kept and will be
  provided on request.

## Files in this directory

- `firmus_gb300_ngpu8_gptoss_g6c_nvfp4_v2_2_nemo_g6c_nvfp4_v2_2.json`: system-implementation metadata
- `setup.sh`: one-time configuration
- `init_datasets.sh`: one-time dataset preparation
- `run_and_time.sh`: runs the benchmark and produces a result
- `redact_logs.sh`: the log post-processing that makes each `result_N.txt` from its run's raw log
- `BUILD_CONTEXT.sha256`: sha256 of the 57 files in `nemo_g6c_nvfp4_v2_2/`, the build context of the image used for the results
- `KERNELS.md`: every kernel and numerical change, what it replaces, equivalence, enabling switch
- `verification/padding_zero/`: check that every padded expert entry stays zero, and its output
- `verification/optimizer_ab/`: optimizer equivalence test and its output
- `verification/redaction_check/`: per-file comparison of the result files with the raw logs
- `verification/refmodel_check/`: model structure, parameter and eval-window dump from a 30-step run of the submitted image (job 22789), and its output
- `LICENSE`: Apache License 2.0
