# TTA — Llama 3.1 8B, 4x H100 NVL, Closed + Power

MLPerf Training v6.1 submission notes for this system. The two NVIDIA files in this
directory (`README.md`, `README_8b.md`) are inherited upstream documents written for
NVIDIA reference platforms; **`README.md` documents Llama 3.1 405B, not this benchmark.**
Where they disagree with the description below, this file is authoritative for our runs.

---

## 1. System

| | |
|---|---|
| Node count | 1 |
| Accelerators | **4x NVIDIA H100 NVL** (94 GB HBM3, 400 W power limit, 1785 MHz max SM) |
| Accelerator interconnect | **Two independent 2-GPU NVLink Bridge pairs** — GPU0-GPU1 and GPU2-GPU3, NV12 (~212 GB/s busbw per pair). Cross-pair traffic traverses PCIe Gen5 x16 / UPI |
| Host CPU | 2x Intel Xeon Platinum 8558 (48 cores each, 192 vCPU) |
| Host memory | 1.0 TB, 16x 64 GB DDR5-4800 RDIMM |
| Chassis / board / BIOS | ASUS ESC8000-E11 / Z13PG-D32 / 2803 |
| Training storage | **Lustre 2.15.8 on ZFS 2.3.5** — 1x MDT + 2x OST on NVMe, `stripe_count=2 stripe_size=1M`. MDS/OSS run on this same node |
| OS / kernel | Rocky Linux 10.2, kernel 6.12.0-211.16.1.el10_2.0.1.x86_64 |
| Driver / Slurm | NVIDIA 580.126.20 / Slurm 25.05.0 with Pyxis + Enroot 3.4.1 |
| Cooling | Air cooled |
| Power supplies | 4x 3000 W redundant, **80 PLUS Titanium** — chassis vendor specification (ASUS ESC8000-E11); `dmidecode -t 39` reports 4x GOSPOWER G1236-3000WNA, 3000 W each |

### NUMA layout matters here

```
node 0  cpus 0-23,96-119      <- IPMI power sampler is pinned here
node 1  cpus 24-47,120-143    <- resource monitoring is pinned here
node 2  cpus 48-71,144-167    <- GPU0, GPU1
node 3  cpus 72-95,168-191    <- GPU2, GPU3
```

Because the NVLink bridges are pairwise, **the TP group must stay inside one physical
pair.** Verified at runtime from the training log:

```
All tensor model parallel group ranks: [[0, 1], [2, 3]]
Rank 0 has data parallel group : [0, 2]
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM
```

TP crosses no bridge; only the DP gradient all-reduce crosses pairs.

## 2. Dataset

Preprocessed C4 (`c4/en/3.0.1`) plus the Mixtral tokenizer, laid out exactly as the
upstream reference expects. The corpus is byte-identical to the one used in v5.1 — the
benchmark's dataset did not change between rounds.

```bash
export DATADIR=<path/on/lustre>/mltraining_llama31_8b
ls $DATADIR/8b/     # 13 files: c4-train/validation .bin/.idx, tokenizer/, md5, LICENSE, NOTICE
```

Dataset integrity is checked against `llama-3-1-8b-preprocessed-c4-dataset.md5`.

> Upstream switched the download path from rclone to an R2 downloader after v5.1; see
> the reference repository's `small_llm_pretraining/nemo/README.md` for current steps.
> `data_scripts/download_8b.sh` in this directory is the v5.1-era script.

## 3. Container

We do **not** build the image. The image validated in v5.1 is restored from a preserved
archive so the training environment stays bit-identical across rounds.

```bash
zstd -dc <archive>/mlperf_v51_llama31_8b_cb0f14cbe7cb.tar.zst | docker load
docker tag <loaded> <registry>/mlperf-nvidia:training5.1_llama31_8b-pyt
export CONT=<registry>/mlperf-nvidia:training5.1_llama31_8b-pyt
```

| | |
|---|---|
| **config digest** | `sha256:cb0f14cbe7cb83d7fa310dc4fadda43cd7a33e36870c02f7513170defd4a95fc` |
| manifest digest | `sha256:d68cb785a9077022a3ab30d5277f1426e79ea0b5aad3f23da77300058df74170` |

> Docker 29 with the containerd image store prints the **manifest** digest as the image
> ID; older Docker printed the **config** digest. Both are listed so the image can be
> identified either way. Verify with:
> ```bash
> ctr -n moby content get <manifest-digest> \
>   | python3 -c "import json,sys;print(json.load(sys.stdin)['config']['digest'])"
> ```

### `pretrain.py` overlay — required

v6.1 adds nine mandatory mllog keys. Rather than rebuild the image (which would change
the training environment), we overlay **only** the patched `pretrain.py`:

```bash
export EXTRA_MOUNTS="$PWD/pretrain.py:/workspace/llm/pretrain.py:ro"
```

`config_mounts.sh:51-53` prepends `EXTRA_MOUNTS` to `_cont_mounts`, which is what
`run.sub` passes to `--container-mounts`. A single-file bind mount is enough — the
unpatched `pretrain.py` baked into the image is shadowed at exactly that path:

```bash
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
    _cont_mounts="${EXTRA_MOUNTS},${_cont_mounts}"
fi
```

`REMOUNT_WORKDIR` (`config_mounts.sh:72-75`) is a **separate** knob that replaces the
whole `/workspace/llm` directory. It is **not** needed here and we do not set it.

**This is verifiable in the submitted logs.** `run.sub:218` echoes the final mount list,
and the line is present in all ten `result_*.txt`:

```
$ grep -o 'pretrain\.py:/workspace/llm/pretrain\.py:ro' results/*/llama31_8b/result_*.txt | wc -l
10
$ grep -m1 -o '[^,=]*pretrain\.py:/workspace/llm/pretrain\.py:ro' results/*/llama31_8b/result_0.txt
<impl-dir>/pretrain.py:/workspace/llm/pretrain.py:ro
```

So the `pretrain.py` shipped in this directory is provably the file that ran, and the
nine keys are emitted from it (`"file": "/workspace/llm/pretrain.py", "lineno": 667` in
every log).

The unpatched original is preserved as `pretrain.py.container-original`
(`md5 54fc1f533b7b…`; patched is `7055d6a4d881…`). The delta is **+67 / -1 lines, all
logging** — no change to the training computation. Diff the two files to confirm.

The nine keys and their values on this system:

| key | value | derived from |
|---|---|---|
| `lowest_numerical_precision_in_linear` | `bfloat16` | `FP8=False`, `trainer.precision=bf16`, `params_dtype=torch.bfloat16` |
| `lowest_numerical_precision_in_attn` | `bfloat16` | `FP8_DPA=False` |
| `lowest_numerical_precision_in_comm` | `bfloat16` | `grad_reduce_in_fp32=False`, `FP8_PARAM_GATHER=False` |
| `tensor_parallelism` | `2` | `TENSOR_MODEL_PARALLEL` |
| `pipeline_parallelism` | `1` | `PIPELINE_MODEL_PARALLEL` |
| `context_parallelism` | `1` | `CONTEXT_PARALLEL` |
| `expert_parallelism` | `1` | dense model; rules require unused dimensions logged as 1 |
| `micro_batch_size` | `1` | `MICRO_BATCH_SIZE` |
| `config_filename` | `config_NVLink_H100_BF16_RCP_BS32_4GPU.sh` | `DGXSYSTEM` |

Precision values are derived from the live config rather than hardcoded, so a config
that does enable FP8 cannot silently log `bfloat16`.

## 4. Configuration and launch

Our config does **not** follow the upstream
`config_<SYSTEM>_<NODES>x<GPUS>x<BATCH>xtpXppYcpZ.sh` naming convention:

```
config_NVLink_H100_BF16_RCP_BS32_4GPU.sh
```

It sources `config_common.sh`, `config_common_8b.sh`, `config_common_cg.sh` and
`config_common_fp8attn.sh`, then overrides the FP8 switches back to `False`.

| knob | value |
|---|---|
| `DGXNNODES` / `DGXNGPU` | 1 / 4 |
| `MINIBS` | 16 (per-DP mini-batch) |
| `TENSOR_MODEL_PARALLEL` | 2 |
| `PIPELINE_MODEL_PARALLEL` / `CONTEXT_PARALLEL` | 1 / 1 |
| `MICRO_BATCH_SIZE` | 1 |
| global batch size | **32** = `MINIBS × (DGXNGPU / TP)` = 16 × (4/2) |
| `gradient_accumulation_steps` | 16 = `GBS / (MBS × DP)` |
| `LR` | 1e-3 |
| `WARMUP_STEPS` | 511 |
| `VAL_CHECK_INTERVAL` | 384 steps (= 12,288 sequences, per the v6.1 rules) |
| `MAX_STEPS` | 1,200,000 |
| precision | BF16 (`FP8=False`, `TRANSFORMER_ENGINE=False`) |
| `BINDCMD` | `bindpcie --cpu=node --mem=node` (pins each rank to its GPU's NUMA node) |

> `OVERWRITTEN_NUM_LAYERS` is 32 for 8B. The upstream note about a default of 126 and
> `PP * VP` divisibility applies to 405B, not here.

Launch:

```bash
export DATADIR=<as above>
export LOGDIR=<path/on/lustre>/log
export CONT=<as above>
export EXTRA_MOUNTS="$PWD/pretrain.py:/workspace/llm/pretrain.py:ro"
export MLPERF_SUBMISSION_ORG=<org>
export MLPERF_SUBMISSION_PLATFORM=<system-id>
source config_NVLink_H100_BF16_RCP_BS32_4GPU.sh
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub
```

⚠️ **Do not set `SEED`.** `run.sub` derives it as `${SEED_BASE:=${SEED-$RANDOM}}`, so an
inherited `SEED` would freeze it and every run would log the same seed, which makes the
seed checker reject the whole results directory. Leaving it unset gives a fresh
`$RANDOM` per job.

### Local additions to `run.sub`

Three additions, none touching the training path:

1. **Mount guard** — `lustre_mount_guard.sh` runs before training and fails the job if the
   data path is not a real Lustre mount. The data path is a symlink onto a local
   directory; if the Lustre client is not mounted, every write silently lands on local
   disk instead.
2. **Resource monitoring** — five independently switchable collectors at 10 s
   (Lustre client/server stats, `zpool iostat`, `iostat`, `nvidia-smi dmon`), pinned to
   NUMA node 1 under a single `sudo`. Measured cost: 0.35 % of one core out of 192 vCPU;
   step time and power were unchanged versus monitoring-off runs. Output is written
   **outside** the results tree. Set `MON_ENABLE=0` to disable.
3. Two variable declarations for the above.

## 5. Power measurement

```
sampler    : ipmitool dcmi power reading, 1 s interval, whole script under ONE sudo,
             pinned to NUMA node 0 (no GPUs, and not the monitoring node)
raw log    : "Instantaneous power reading: <W> Watts  IPMI timestamp: <ctime>"
converter  : mlperf_logging/mllog/examples/power/power_measurement.py  (official tool)
             --log-type IPMI --start-with-readings --convertion-coef 0.94
metric     : mlperf_logging/mllog/examples/power/compute_metric_example.py
output     : power/result_N/node_0.txt
```

Three points worth stating explicitly:

- **The sampler takes one `sudo` for the whole script, not one per sample.** A per-sample
  `sudo` stretches the interval to 2-3 s.
- **`conversion_eff` 0.94 is applied as a multiplication** (AC measurement -> DC). Per the
  MLPerf power rules, 1000 W AC with 10 % loss is 900 W DC. The PSU is 80 PLUS Titanium —
  chassis vendor specification (ASUS ESC8000-E11, "up to four 3000W Titanium redundant
  power supplies"); `dmidecode -t 39` confirms four 3000 W units are installed.
  `conversion_eff` is set to 0.94, the efficiency that the 80 PLUS Titanium standard
  guarantees at its 20 % load point, which is the load band this submission operates in.
- **The el10 `ipmitool` prints `MM/DD/YYYY HH:MM:SS AM/PM TZ`**, not the ctime form the
  official parser expects (`%a %b %d %H:%M:%S %Y`). `power_monitor.sh` normalises it so
  the output format is identical across OS versions. Without this the entire power log
  fails to parse.

Sampling quality across the ten runs: 65,591-84,878 samples per run, mean interval
1.048-1.050 s, intervals are only ever 1 s or 2 s (never 3 s or more), and 2 s gaps are
4.85-4.94 %. The gaps come from loop drift (`sleep 1` plus the `ipmitool` execution time
accumulating), not from contention — runs with monitoring disabled show the same
interval. Energy is integrated trapezoidally from the actual timestamps, so a missing
sample does not bias the result.

## 6. Reproducing the 10-run campaign

```bash
NRUNS=10 ./submit_4gpu_chain.sh     # afterany-chained, one run at a time
```

Per run, roughly 19-25 h depending on how many evaluation cycles convergence takes.

### Checking each run

```bash
./scripts/check_run.sh <jobid>            # rule items and reference figures
./scripts/check_seed_uniqueness.sh        # seeds must differ between runs
./process_power_log.sh <jobid>            # energy
```

Rule items (pass/fail): `eval_accuracy <= 3.3`, `run_stop status == success`,
compliance against ruleset 6.1.0, and seed uniqueness. Everything else — samples to
converge, energy, step time — is a reference figure, not a rule.

### What we observed

| | |
|---|---|
| Convergence | 208,896 (4 runs) / 221,184 (5) / 270,336 (1) samples |
| Olympic mean | **216,576** (RCP lower bound for BS=32 is 173,649.76) |
| Step time | 10.226-10.276 s |
| Run time | 19.05-24.66 h |
| Energy per run | 152.25-196.98 MJ; **7.28-7.43 MJ per 10k samples** across all ten |

Convergence is quantised to the evaluation interval: 12,288 samples = 384 steps at
GBS 32, so every value is an exact multiple (17, 18 and 22 cycles above). One cycle is
about 1.05 h and 8.4 MJ.

The slowest run (270,336 samples, 196.98 MJ) is a valid run — RCP tests only a lower
bound, so slow convergence cannot violate it, and its normalised energy
(7.286 MJ/10k samples) matched every other run. The result summariser drops the fastest
and slowest runs' power values along with their performance scores, so it does not enter
the power score either.

### Final verification

```bash
cd <mlperf-logging checkout>
./scripts/verify_for_v6.1_training.sh <submission>/TTA
```

This wrapper is what the submission UI runs: `package_checker`, then
`result_summarizer`, then `repo_checker`. Note the argument is the **submitter**
directory; passing its parent fails with `FileNotFoundError: .../results`.

`package_checker` emits `Unknown files in result directory` for any Power submission —
it globs `*` in the benchmark directory and counts the `power/` directory as an extra
entry, while the same file looks for `power/` at exactly that path. It is a warning, not
an error.

## 7. Quality

| | |
|---|---|
| Metric | Log perplexity |
| Target | 3.3 |
| Evaluation frequency | every 12,288 sequences |
| Evaluation thoroughness | 1,024-sequence validation subset |
| Checkpoint | none — trained from scratch |
