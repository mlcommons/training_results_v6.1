# MLPerf Training 6.1 on AMD MI355X — Analysis, Fixes and Optimization Report

**System:** 2x HPE XD685, 8x AMD Instinct MI355X (288 GB HBM3e) per node, 2x AMD EPYC 9555, 2.2 TB RAM (all results: 1 node)
**Software:** AMD MLPerf 6.1 partner code (Llama 3.1, GPT-OSS, DLRMv4), AMD Primus v26.7.0 + HPE patch (Llama 2), AMD v6.0 FLUX.1 code + HPE 6.1 logging
**Ruleset:** MLPerf Training 6.1.0, Closed division (mlcommons/logging `8a0cc05`)
**Updated:** 2026-10-13 (section 0 is current; sections 0-old to 3 are the 6.0-era history)

---

## 0. Final status 2026-10-13 (MLPerf 6.1)

| Benchmark | Score (official method) | Best AMD MI355X | Best NVIDIA 8× B300 | What made the difference |
|---|---|---|---|---|
| GPT-OSS 20B | **66.08 min** | none published | 82.55 (we are 20.0% faster) | AMD 6.1 drop: MXFP4 + fused kernels, +51% samples/s over Primus v26.7 FP8 (98.24 → 66.08) |
| Llama 3.1 8B | **76.80 min** | 86.84 (we are 11.6% faster) | 72.01 (we are 6.6% slower) | AMD 6.1 drop: FlyDSL attention/GEMM and fused layers, +16% samples/s (86.21 → 76.80) |
| Llama 2 70B LoRA | **8.245 min** | 8.27 (we are 0.3% faster) | 6.58 (we are 25.3% slower) | 6.1 logging patch; 20-run group with `zone_reclaim_mode=0` + THP madvise: 16/20 runs at the earliest stop (3,072 samples); median window submitted |
| FLUX.1 | **260.57 min** = 2,085 GPU-min | no 8-GPU result | no 8-GPU result; 32× B300 2,479 GPU-min (we use 16% less) | 6.1 logging fix; 2 data-loader workers per GPU (the AMD 6.1 FLUX drop is 2.2× slower here, not used) |
| DLRMv4 (new in 6.1) | **51.91 min** | none | none | AMD 6.1 drop; data loader was the bottleneck: overcommit 0 + zone_reclaim 0 via the Slurm prolog, 6 train + 8 eval workers, prefetch 4 (65.93 → 51.91) |

Package `/nas/SHARED/MLPERF/submission_61_20261013`: official `verify_for_v6.1_training.sh` **ALL CHECKS PASSED**,
50/50 logs compliant, RCP 5/5. Every experiment is in `INVESTIGATIONS.md`; how to rebuild and re-check the package is in `README.md`.

Key findings of the 6.1 phase:

| Finding | Consequence |
|---|---|
| The 6.1 log rules make three precision fields mandatory | All 6.0-era sets became invalid; every benchmark was re-run |
| The GPU driver on lodin is Ubuntu's in-kernel amdgpu; it refuses GPU-to-GPU virtual-memory sharing (`amdgpu_dma_buf.c:223`) | AMD's `rccl_sdma` mode and `NCCL_CUMEM_ENABLE=1` crash; we use plain NCCL. Fix = AMD's DKMS driver (admin decision) |
| Llama 2 scores are capped by the RCP scaling (faster-than-reference runs are scaled up) | Best possible ≈ 8.17 with THP `always`, 8.245 with THP `madvise` (THP madvise costs ~0.9% per run) |
| Shorter Llama 2 cosine schedules (432, 480) | No lower loss at 3,072 samples, and runs can get stuck: rejected |
| Kernel settings on lodin are reset by the Slurm prolog at every job start | Per-job settings go through `--comment` (`sysctl file=…`, `thp=…`) and are disclosed per benchmark in the system description |

## 0-old. Status 2026-10-07 (6.0 era, history)

| Benchmark | Score (10 runs, official method) | Best published | What made the difference |
|---|---|---|---|
| Llama 3.1 8B | **86.21 min** | AMD MI355X 86.84; B300 72.01 | Micro-batch 4, NUMA binding, FP32 attention softmax. Primus v26.7: same speed, no gain. Warmup 128: no gain |
| Llama 2 70B LoRA | **8.244 min** | AMD MI355X 8.27; B300 6.58 | AMD's Primus v26.7 recipe (0.8% faster than v6.0 image) |
| GPT-OSS 20B | **98.24 min** | no MI355X; B300 82.55 | AMD's Primus MLPerf recipe instead of the MLCommons reference (~5x faster), Primus v26.7 (+46% over v26.5), LR 8.25e-4 |
| FLUX.1 | **261.46 min** (2,092 GPU-min) | no 8-GPU result; B300 32 GPUs 2,479 GPU-min | Data loader 2 workers/GPU instead of 8: the degraded RAID streams at 1.9 GB/s instead of 0.36 (was 356.8 min) |

All 4 sets pass the official MLCommons log checker; seeds are unique.

**Rules finding (2026-10-07).** AMD's v6.0 launchers ran `runtime_tunables.sh` on every job, which changes kernel settings (C2 idle disable, ASLR off, NMI watchdog, NUMA balancing, THP) via sudo. Our rule forbids system changes, so all launchers now skip it unless `RUN_RUNTIME_TUNABLES=1`; the page-cache drop that the MLPerf rules require still runs before every trial. The nodes still show the changed settings until the next reboot.

**Trial package (2026-10-09):** built from these 4 sets; every official MLCommons check passes (package checker: NO ERRORS). Open items: logs say organization "AMD", cooling/BIOS fields to verify, 6.1 rules. See `README.md`.

## 1. Executive summary (2026-09-30)

Scores below are **RCP-normalised effective scores**, which is what published results are. A raw wallclock that converged faster than the RCP mean is multiplied back up by `RCP_mean / your_mean`; comparing a raw score against a published one is not like-for-like.

| Benchmark | Before | After (effective) | Best published (AMD MI355X) | Verdict |
|---|---|---|---|---|
| **Llama 3.1 8B** | 93.09 min (olympic mean, 4 runs) | **86.214 min** (10 runs, scaling 1.0) | 86.84 min | **beats AMD by 0.72%** — valid 10-run set |
| **Llama 2 70B LoRA** | never ran (silent hang) | **8.244 min** (10 runs, Primus v26.7, raw 8.217 x 1.0034) | 8.27 min | **beats AMD by 0.31%** |
| **FLUX.1** | not runnable on 1 node | **261.46 min** (10 runs, 2 data-loader workers/GPU; was 356.8) | no 8-GPU result published | 2,092 GPU-min vs B300 32-GPU 2,479 (−15.6%), GB300 NVL72 32-GPU 2,107 (−0.7%) |
| **GPT-OSS 20B** | reference recipe ~500-725 min | **98.24 min** (10 runs, Primus v26.7, LR 8.25e-4; official RCP check: no scaling) | none on MI355X; best B300 82.55 | AMD Primus v26.7 FP8 recipe + our LR (v26.5 set 154.1, v26.7 at LR 8.5e-4 102.3) |

> **Correction (2026-09-30).** An earlier version of this table claimed Llama 2 beat AMD by 0.67% with 8.215 min. That was wrong on two counts: it used only 3 runs (10 are required), and it compared a **raw** score against RCP-normalised published figures. Those 3 runs converged in 3,072 samples against an RCP mean of 3,179, so they carry a 1.0347 scaling penalty — an effective **8.500 min, i.e. 2.8% behind AMD**, not ahead. The bug that produced the flattering number was in `analyze_results.py`'s "vs published" comparison and is now fixed; it applies `effective_score()` everywhere.

Two independent problems were identified and separated for each benchmark:

- **Throughput** — pure hardware/software efficiency. Unrestricted by the rules.
- **Convergence** — how many samples are needed to hit the quality target. Tightly bounded by the Reference Convergence Points (RCP) checker.

The single most important insight of this work is that **converging faster than the reference earns nothing**: the RCP checker multiplies your score back up by `RCP_mean / your_mean`. The optimum is to land exactly on the RCP mean.

This cuts both ways. Llama 3.1 sits 3.9% *above* its RCP mean, so it takes no penalty and its raw gains are real.

Llama 2 is subtler, and an intermediate version of this report got it wrong. Because **convergence is quantised by the eval cadence** (every 384 sequences at GBS 8), the only reachable values near the 3,179 RCP mean are 2,688 / 3,072 / 3,456. Landing *on* the mean is impossible. Measured: 3,072 costs 8.154 min raw (effective **8.437** after 1.0347 scaling), 3,456 costs 9.490 min (effective **9.490**). Buying scaling relief with an extra eval cycle is a heavy net loss, so **the optimum is the lowest legal convergence point, 3,072 — not the RCP mean.** Llama 2's remaining work is therefore (a) reliability, lifting the 3,072 hit-rate from 73% to ~100%, worth ~0.8%, and (b) ~2.0% of raw throughput.

---

## 2. Llama 2 70B LoRA — from broken to running (history; now 8.244 min, beats AMD)

### 2.1 Symptom

Runs produced a ~3 KB log that stopped immediately after the `launching experiment using: ...` line and then hung indefinitely. No error, no `RESULT,` line.

### 2.2 Why it looked like a hang

`llama2_70b_lora/.../run_and_time.sh` invokes:

```bash
./runner/primus-cli direct train posttrain --config ${EXP} 2>/dev/null \
  | grep --line-buffered -E ':::MLLOG |^RESULT,|Training failed'
```

stderr is discarded and stdout is filtered to three patterns. When rank 0 raised an exception, the remaining seven ranks blocked forever inside a NCCL collective, and **nothing was printed**. The first fix was therefore diagnostic, not functional: make failures visible.

- Added `export MLLOG_VERBOSE_LOGS=${MLLOG_VERBOSE_LOGS:-0}` to `config_MI355X_1x8x1.sh`. This is mandatory — Docker env forwarding is a *name allowlist* harvested from the config file, so a host-only export is silently dropped.
- Added `export PRIMUS_WORKSPACE=${PRIMUS_WORKSPACE:-/results/primus}` so per-rank Primus logs land on the mounted results volume instead of dying with the container. The real traceback lives in `.../logs/post_trainer/rank-N/debug.log`.

### 2.3 Root cause 1 — checkpoint conversion against a gated repo

The Primus hook `01_convert_checkpoints.sh` decides whether to convert:

```bash
MEGATRON_PATH="${EFFECTIVE_DATA}/megatron_checkpoints/$(basename "${HF_PATH}")"   # line 147
if [[ -d "$MEGATRON_PATH" ]]; then
    echo "extra.pretrained_checkpoint=${MEGATRON_PATH}"; exit 0
fi
# ... otherwise download meta-llama/Llama-2-70b-hf and convert
```

`EFFECTIVE_DATA` resolves to `DATA_PATH` = `/data`, which is the **`${DATADIR}/data`** bind mount.

What was on disk:

| Expected at `/data` | Actually present |
|---|---|
| `train.npy`, `validation.npy`, `packed_metadata.jsonl` | absent — they were one directory *above*, at `/nas/SHARED/MLPERF/mlperf_llama2/` |
| `megatron_checkpoints/Llama-2-70b-hf/` | only a stale `Llama-2-70b-hf.converting.lock` from an earlier aborted attempt |

A lock *file* does not satisfy `[[ -d ]]`, so the hook attempted a fresh download of the **gated** `meta-llama/Llama-2-70b-hf` and died with `401 Unauthorized`. Meanwhile the real, fully converted 129 GB checkpoint was sitting at `/nas/SHARED/MLPERF/mlperf_llama2/megatron_checkpoints/` — outside the mount.

**Fix:** place the converted checkpoint and the packed data in the same directory and point both mounts at it:

```bash
DATADIR=/lvol0/mlperf/llama2          # -> ${DATADIR}/data mounted at /data
MODEL=/lvol0/mlperf/llama2/data       # -> mounted at /ckpt  (same directory)
```

This is the layout `scripts/prepare_data_and_model.sh` produces natively.

### 2.4 Root cause 2 — training-time HuggingFace access

With the hook satisfied, training still failed. `src/llama2_70b_lora.yaml:4` declares `hf_path: meta-llama/Llama-2-70b-hf`, and Megatron-Bridge loads the HF **config and tokenizer** from it — hitting the gated repo a second time.

**Fix:** the model *weights* come from the Megatron checkpoint, so only config + tokenizer are needed. A minimal 2.5 MB offline HF cache was staged into `${DATADIR}/data/huggingface`:

```
hub/models--meta-llama--Llama-2-70b-hf/
  refs/main
  snapshots/3aba440b.../{config.json, tokenizer.json, tokenizer.model,
                         tokenizer_config.json, special_tokens_map.json,
                         generation_config.json, *.index.json}
```

plus `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the config. This makes the benchmark fully reproducible offline and removes any dependency on a token. (A complete upstream snapshot also exists at `/nas/SHARED/MLPERF/hf_cache`, and `monnetb`'s `~/.bashrc` carries a working `HF_TOKEN` should online access ever be required.)

### 2.5 Results

Three consecutive runs, node-local data, caches dropped between trials:

| Run | TTT | Samples to converge | samp/s | Eval time | Final eval loss | Status |
|---|---|---|---|---|---|---|
| 1 | 9.586 min | 3,456 | 6.01 | 66.5 s | 0.9237 | success |
| 2 | 8.215 min | 3,072 | 6.23 | 53.4 s | 0.9226 | success |
| 3 | 8.211 min | 3,072 | 6.24 | 53.4 s | 0.9246 | success |

- **Olympic mean (MLPerf scoring method) = 8.215 min**, versus AMD 8.27, Dell XE9785L 8.640, Dell XE9785 8.585.
- Quality target (eval loss ≤ 0.925) met in every run.
- Mean samples to converge = **3,200**, exactly the RCP olympic mean for GBS 8.

NUMA binding was also tested (2 runs, both 3,456 samples): 6.06 and 6.07 samp/s versus 6.01 for the base run at the same sample count — a **+1.0%** throughput gain, small but consistent.

### 2.6 Caveats

- Three runs is not a score. A valid submission needs **10**, scored as the mean of the middle 8.
- Run-to-run spread is dominated by convergence quantisation (3,072 vs 3,456 samples is a 12.5% swing), not by throughput.
- With only 3 runs the olympic-mean *sample* count (3,072) sits below the RCP mean (3,200) and would incur a 1.0417 scaling factor. Across 10 runs the mean should regress toward the true value (our 3-run arithmetic mean is already exactly 3,200), but this must be verified with the real 10-run set.

---

## 3. Llama 3.1 8B — decomposing a 7.2% deficit

### 3.1 Baseline

Four completed runs (all passing the compliance checker):

| Run | TTT | Samples | Overall samp/s | Eval time | Eval % |
|---|---|---|---|---|---|
| 1 | 96.19 min | 196,608 | 34.07 | 191.0 s | 3.3% |
| 2 | 96.29 min | 196,608 | 34.03 | 190.4 s | 3.3% |
| 3 | 89.99 min | 184,320 | 34.14 | 175.0 s | 3.2% |
| 4 | 89.85 min | 184,320 | 34.19 | 176.8 s | 3.3% |

Olympic mean **93.09 min** vs AMD **86.84 min** — 7.2% behind.

### 3.2 The reference number, pinned down

AMD's reproduction blog quotes "about 35.6 samples per second" and gives an example run of `run_stop - run_start = 5,178,140 ms`. Since `184,320 / 5178.14 = 35.60`, AMD's quoted figure is **overall throughput** (samples ÷ wall-clock TTT), directly comparable to ours.

- AMD: 35.60 samp/s
- Ours: 34.11 samp/s
- **Gap: 4.3%**

The remaining ~2.9% of the 7.2% total deficit comes from convergence, not speed.

### 3.3 Throughput investigation

Steady-state throughput is remarkably stable — 15 consecutive training blocks in a 90-minute run:

```
block 0 : 34.54 samp/s   (ramp-up)
blocks 1-14: 35.34 samp/s  (sigma ~0.1)
```

There is no thermal or memory-fragmentation decay to recover. Under load all 8 GPUs sit **at the 1400 W power cap**, 100% utilisation, 61–71 °C junction. The workload is **power-limited**, so gains must come from doing less work per sample, not from higher clocks.

A probe harness (`llama31_8b/probe.sh`) was built that measures throughput strictly between two evaluations, excluding startup, JIT warmup and eval cost — roughly 9 minutes per data point instead of a 90-minute run.

| Config | samp/s | vs stock |
|---|---|---|
| `PRIMUS_MICRO_BATCH_SIZE=4` | **37.88** | **+3.46%** |
| NUMA bind + MBS=4 | 37.72 | +3.02% |
| `MLPERF_NUMA_BIND=1` | 37.28 | +1.81% |
| rebuilt image (`CU_NUM=256`, gfx950-only, AOTriton on) | 37.22 | +1.64% |
| `CUDA_DEVICE_MAX_CONNECTIONS=8` | 37.17 | +1.49% |
| dataset on NFS | 36.92 | +0.84% |
| stock, node-local NVMe | 36.62 | baseline |
| `GPU_MAX_HW_QUEUES=4` | 34.88 | −4.75% |
| `PRIMUS_MICRO_BATCH_SIZE=1` | 34.22 | −6.54% |

Key conclusions:

- **MBS=4 is the best single lever.** At GBS 32 across 8 GPUs it drives gradient accumulation from 2 down to 1, removing a full extra forward/backward boundary per optimizer step. `micro_batch_size` is an unrestricted implementation detail in the Closed division and is invisible to the RCP checker, so this is a zero-risk optimization. It must be mirrored into `MLLOG_MICRO_BATCH_SIZE` for compliance logging.
- **The workload is not I/O bound.** Staging 84 GB of C4 from NFS to node-local NVMe changed throughput by less than 1% (in fact NFS measured marginally *faster*, i.e. within noise). This was the hypothesis with the largest expected payoff and it was wrong — Megatron's prefetching fully hides NFS latency. Staging is still retained because it is free, rules-legal, and removes a shared-filesystem variable.
- **GPU→NUMA mapping is not the identity.** BDF-ordered GPUs map to NUMA nodes `[1,2,3,0,5,6,7,4]`. A naive `rank == node` binding would be wrong for 6 of 8 ranks, which is why the implementation detects the mapping from sysfs at runtime.
- Probe absolute values read ~4% high versus a full run (36.6 vs 35.3) because power-capped GPUs boost harder in short blocks. Probes are reliable for **ranking**, and winners are confirmed with full runs.

### 3.4 Convergence — the larger remaining opportunity

Reading the official RCP file (`rcps_llama31_8b.json`, `training_6.0.0`) for GBS 32:

- Reference distribution (20 points): 10x 172,032 / 9x 184,320 / 1x 196,608
- **RCP olympic mean = 178,858.7 samples**
- **`min_epochs` hard fail floor = 173,649.7 samples**

Our mean was **190,464** — 6.5% beyond the optimum, and we never once reached 172,032 while the reference does so half the time.

The scoring logic in `rcp_checker.py` is:

```python
norm_factor = max(1.0, RCP_Mean / mean_subm_epochs)
if mean_subm_epochs >= Min_Epochs:
    ... pass, scaling_factor = norm_factor
else:
    ... FAIL
```

So converging *faster* than the reference is neutralised by an exactly compensating score inflation, and converging faster than `min_epochs` is an outright rejection. **The optimum is to land on 178,859 and no lower.** Closing this gap alone is worth ~6% of TTT.

Available, rules-legal levers:

- `opt_base_learning_rate` — **unconstrained** by the hyperparameter table. AMD themselves ship `1e-3` for MI350X versus `8e-4` for MI355X, establishing LR as a tuned axis.
- `opt_learning_rate_warmup_steps` — **unconstrained**. Our config uses 64 iterations (2,048 samples); the RCP reference used 4,096 samples (128 iterations).
- Constraints that must be preserved: `min_lr = 0.1 x base_lr`, `opt_learning_rate_decay_steps = 1,200,000 - warmup` (machine-checked), betas/eps/weight-decay/clip-norm all fixed.
- Seeds must be clock- or urandom-derived, and no two runs may log the same seed — deliberate seed selection is prohibited.

The likely reason we converge slower than the BF16 reference is the **MXFP4** training recipe, which the RCP check exists precisely to police. Dataset integrity was ruled out: all C4 and tokenizer md5s verify against the published manifests.

### 3.5 Status

A 3-run set with `MBS=4 + NUMA` is executing. Expected effect: +3 to +3.5% throughput, taking ~93.1 min toward ~90 min at unchanged convergence, or ~85–87 min once convergence is also corrected.

---

## 4. Rules compliance notes

All optimizations applied are legal in the Closed division, verified against `training_rules.adoc`:

| Change | Rule basis |
|---|---|
| NFS → node-local NVMe staging | "Data can start on any durable storage system such as local disks." RAM/tmpfs excluded; cache flush retained. |
| `micro_batch_size`, gradient accumulation | listed as **unconstrained**; pure implementation detail |
| NUMA/CPU affinity, env vars, kernel and NCCL tuning | unrestricted; must be consistent across all runs and disclosed |
| Image rebuild (`CU_NUM`, arch, AOTriton) | permitted provided every component stays from an official repo (required for *Available* status) |
| LR / warmup tuning | **unconstrained** in the hyperparameter table, bounded in practice by the RCP checker |
| Eval cadence and `eval_samples=1024` | **untouched** — both are fixed and machine-checked |
| Cache dropping | **retained** before every trial — required by the rules. AMD's `runtime_tunables.sh` (kernel settings) is **not** run since 2026-10-07: it changes the system |
| Data-loader worker count (FLUX) | implementation detail; model, data, hyperparameters unchanged |
| Data-loader worker count and prefetch (DLRMv4) | implementation detail; model, data, hyperparameters unchanged |
| Per-job kernel settings (`vm.zone_reclaim_mode`, `vm.overcommit_memory`, THP) via the Slurm prolog | system tuning is allowed; applied identically to every run of the set and **disclosed per benchmark** in the system description |
| Llama 2: 20 runs, submit 10 | "run M>N independent runs as a group and designate N consecutive runs … closest to the median result": runs ordered by start time, the median 10-run window (runs 5–14) is submitted; all 20 logs kept for reviewers |
| Llama 2 cosine schedule length, DLRMv4 LR | `opt_learning_rate_training_steps` and LR are unconstrained; tested variants that failed were not used, with the decision rule written down before the results |
| DLRMv4: which set to submit | the lodin2 set was declared final before it ran (set 2 = fallback only), so no choice among sets by score |
| `scaling.json` in the Llama 2 and DLRMv4 result folders | "the scaling factor must be provided in a json file under the name scaling.json"; written by the official RCP checker |

Not done, and deliberately so: seed selection, eval-frequency reduction, any change to fixed optimizer hyperparameters, re-running a set with unchanged settings to get a luckier result, or picking the fastest 10-run window.

---

## 5. Reproduction

See `README.md` ("Running"): one `sbatch` per run, with the launcher used for each scored set. Score with `HPE/tools/analyze_results.py` and check logs with the MLCommons compliance checker.

---

## 6. Remaining work (2026-10-07, history — superseded by section 0)

1. **Package blockers** (see `README.md`): replace AMD's logs in `HPE/results/`, HPE values in the logs (re-run final sets), HPE system description, add the Primus v26.7 code to the implementation folders, then `make_submission.sh` + `validate_submission.sh`.
2. **MLPerf Training 6.1**: re-check rules, logging version and RCPs when published.
3. **Llama 3.1** is the largest gap to B300 among tunable benchmarks; LR (7e-4, 1e-3) and warmup (128) tuning did not help. Convergence sits 3.9% above the RCP mean with no penalty.
4. **Llama 2**: 3 of 10 runs missed the lowest point (3,072 samples); ~0.8% possible from reliability.
