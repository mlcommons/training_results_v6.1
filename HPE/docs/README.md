# HPE XD685 / AMD MI355X — MLPerf Training 6.1

MLPerf Training 6.1 on **1 node × 8 AMD Instinct MI355X** (HPE ProLiant Compute XD685),
closed division, five benchmarks.

| Git branch (github.hpe.com) | Content |
|---|---|
| **`amd_mlperf_train_release_6.1.0`** (this branch) | 6.1 work: AMD 6.1 partner code (Llama 3.1, GPT-OSS, DLRMv4), 6.1 logging fixes (Llama 2, FLUX.1), final 6.1 results |
| `amd_mlperf_train_release_6.0.0` | Earlier work on AMD's v6.0 code (6.0-era results, they fail the 6.1 log rules) |

## Final results (6.1 rules, 10 runs each, official MLCommons scoring)

| Benchmark | HPE score | Best AMD MI355X (v6.0) | Best NVIDIA 8× B300 (v6.0) | vs AMD | vs NVIDIA |
|---|---|---|---|---|---|
| GPT-OSS 20B | **66.08 min** | none published | 82.55 | first MI355X result | ✅ 20.0% faster |
| Llama 3.1 8B | **76.80 min** | 86.84 | 72.01 | ✅ 11.6% faster | 6.6% slower |
| Llama 2 70B LoRA | **8.245 min** | 8.27 | 6.58 | ✅ 0.3% faster | 25.3% slower |
| FLUX.1 | **260.57 min** = 2,085 GPU-min | no 8-GPU result | no 8-GPU result (32× B300: 2,479 GPU-min) | only 8-GPU result | ✅ 16% less GPU time than 32× B300 |
| DLRMv4 | **51.91 min** | new in 6.1 | new in 6.1 | — | — |

Scores are RCP-normalised (the official way), so they compare directly with published
results: Llama 2 = raw 7.969 × 1.0347, DLRMv4 = raw 51.070 × 1.0164, others ×1.0.

## Submission package

**`/nas/SHARED/MLPERF/submission_61_20261013`** (on the cluster) holds all five sets, the code
each job actually ran, and the system description. `HPE/results/` and `HPE/systems/` in git
are identical copies of its results and system description.

| Check (mlcommons/logging `8a0cc05`, latest upstream, ruleset 6.1.0) | Result |
|---|---|
| `scripts/verify_for_v6.1_training.sh` | ✅ ALL CHECKS PASSED, package checker NO ERRORS |
| Log compliance | ✅ 50/50 |
| Convergence (RCP) | ✅ 5/5 |
| Warnings | 4, all from the `scaling.json` files in the Llama 2 and DLRMv4 result folders. The rules require that file ("the scaling factor must be provided in a json file under the name scaling.json"); the checker only notes a file not named `result_*.txt` |

| Benchmark | Log set (`/nas/SHARED/MLPERF/results/…`) | Code shipped | Notes |
|---|---|---|---|
| GPT-OSS 20B | `gpt_oss_20b/amd61-set` | `code-403` (AMD 6.1 drop + HPE fixes) | MXFP4 |
| Llama 3.1 8B | `llama31_8b/amd61-set` | `code-404` (AMD 6.1 drop + HPE fixes) | MXFP4 |
| Llama 2 70B LoRA | `llama2_70b/g20-zr0-thpmadvise` | Primus v26.7.0 recipe + HPE patch | **20 runs as one group; submitted = runs 5–14, the 10 consecutive runs closest to the median** (MLPerf rule for M>N runs). All 20 logs kept for reviewers |
| FLUX.1 | `flux1/log61-set` | v6.0 code + HPE 6.1 logging | |
| DLRMv4 | `dlrmv4/amd61-final-lodin2` | `code-455` (AMD 6.1 drop + HPE fixes) | Set declared "final" before it ran (set 2, 52.43 min, was the fallback) |

Kernel settings are stated per benchmark in the system description: Llama 2 ran with
`vm.zone_reclaim_mode=0` and transparent huge pages `madvise`; DLRMv4 with
`vm.zone_reclaim_mode=0`, `vm.overcommit_memory=0` and THP defrag `madvise`; the others with
the node defaults. All were applied per job by the Slurm prolog, not by the benchmark code.

Rebuild and check the package (on striker):

```bash
R=/nas/SHARED/MLPERF/results; A=~/mlcommons_mlperf_training_61a
bash HPE/tools/make_submission.sh --out <dir> --system-json HPE/systems/HPE-Compute-XD685-MI355X-288GBx8.json \
  --llama31-results $R/llama31_8b/amd61-set --code llama31_8b=$R/llama31_8b/amd61-set/code-404 --extra llama31_8b=$A/HPE/tools/run_amd61.sbatch \
  --gptoss-results  $R/gpt_oss_20b/amd61-set --code gpt_oss_20b=$R/gpt_oss_20b/amd61-set/code-403 --extra gpt_oss_20b=$A/HPE/tools/run_amd61.sbatch \
  --llama2-results  $R/llama2_70b/g20-zr0-thpmadvise --code llama2_70b_lora=$A/HPE/benchmarks/llama2_70b_lora/implementations/MI355X_release_6.1 \
                    --extra llama2_70b_lora=$A/HPE/tools/run_primus_llama.sbatch \
                    --primus-host ~/primus/v26.7.0-log61 --primus-recipe llama2_70b_lora=llama2_70b \
  --flux1-results   $R/flux1/log61-set --code flux1=$A/HPE/benchmarks/flux1/implementations/MI355X_release_6.0 \
  --dlrmv4-results  $R/dlrmv4/amd61-final-lodin2 --code dlrmv4=$R/dlrmv4/amd61-final-lodin2/code-455 \
                    --extra dlrmv4=$A/HPE/tools/run_amd61.sbatch --extra dlrmv4=$A/HPE/tools/lodin_sysctl_oc0.conf
bash HPE/tools/validate_submission.sh --package <dir> --local ~/mlperf-logging-latest --python ~/venv-mlperf/bin/python
cd ~/mlperf-logging-latest && PATH=~/venv-mlperf/bin:$PATH PYTHONPATH=. bash scripts/verify_for_v6.1_training.sh <dir>/closed/HPE
```

`select_runs.py` picks the scored window (the median one when there are more than 10 runs;
failed runs count as infinite time). Older packages are kept for the record:
`submission_61` (4 benchmarks), `submission_61_prev`, `submission_trial` (6.0 sets).

## How each final set was run (on striker, from `~/mlcommons_mlperf_training_61a`)

```bash
cd HPE/tools
# GPT-OSS 20B and Llama 3.1 8B: AMD 6.1 drop, lodin fixes in the launcher
sbatch -w lodin2 --time=16:00:00 --export=ALL,BENCH=gpt_oss_20b,TAG=amd61-set,NEXP=10 run_amd61.sbatch
sbatch -w lodin2 --time=18:00:00 --export=ALL,BENCH=llama31_8b,TAG=amd61-set,NEXP=10 run_amd61.sbatch

# Llama 2 70B LoRA: Primus v26.7.0 + HPE patch, 20 runs, kernel settings via the Slurm prolog
sbatch -w lodin2 --time=6:00:00 --comment="docker;sysctl file=/home/hpebench/sysctl/zone_reclaim0.conf;thp=madvise" \
  --export=ALL,BENCH=llama2_70b,TAG=<tag>,PRIMUS_HOST=/home/hpebench/primus/v26.7.0-log61 run_primus_llama.sbatch 20

# DLRMv4: AMD 6.1 drop, overcommit 0 + zone_reclaim 0 via the prolog, THP defrag madvise, more loader workers
sbatch -w lodin2 --time=12:00:00 --comment="docker;sysctl file=$PWD/lodin_sysctl_oc0.conf" \
  --export=ALL,BENCH=dlrmv4,TAG=<tag>,NEXP=10,HPE_THP_DEFRAG=madvise run_amd61.sbatch NUM_WORKERS=6 EVAL_NUM_WORKERS=8 PREFETCH_FACTOR=4

# FLUX.1: v6.0 code + 6.1 logging, one run per job (chain 10 with --dependency=afterany:<id>)
cd ../benchmarks/flux1/implementations/MI355X_release_6.0
sbatch -w lodin2 --export=ALL,TAG=<tag> run_flux1.sbatch 1
```

| Launcher fix needed on lodin (in `run_amd61.sbatch`) | Why |
|---|---|
| plain NCCL parameter all-gather, `NCCL_CUMEM_ENABLE=0` (GPT-OSS) | the in-kernel amdgpu driver refuses GPU-to-GPU virtual-memory sharing; AMD's `rccl_sdma` mode needs AMD's DKMS driver |
| RCCL debug logging off | the GPT-OSS config wrote ~220 MB per GPU per 35 min |
| no kernel tunables (`RUN_RUNTIME_TUNABLES=1` to enable) | system changes go through the Slurm `--comment` only |
| random seeds from `/dev/urandom`, organization HPE | every run must log a different seed |

Score a set and check its logs (on striker):

```bash
python3 HPE/tools/analyze_results.py <llama31_8b|llama2_70b_lora|flux1|gpt_oss_20b|dlrmv4> <logdir>
cd ~/mlperf-logging-latest && PYTHONPATH=. ~/venv-mlperf/bin/python -m mlperf_logging.compliance_checker --usage training --ruleset 6.1.0 <run>.log
```

## Layout

| Path | Content |
|---|---|
| `HPE/benchmarks/<bench>/implementations/MI355X_release_6.1/` | 6.1 code: AMD partner drop (Llama 3.1, GPT-OSS, DLRMv4; FLUX.1 drop kept but not used) imported unchanged, HPE changes in later commits; Llama 2: HPE Primus patch, README, sysctl file |
| `HPE/benchmarks/<bench>/implementations/MI355X_release_6.0/` | AMD v6.0 code with HPE changes (FLUX.1 final set uses it; `git diff a33eb5b -- HPE/benchmarks/` shows HPE's v6.0 changes) |
| `.../experimental/` | one-off tuning and test scripts |
| `HPE/tools/` | launchers, scoring, packaging (`make_submission.sh`, `select_runs.py`, `validate_submission.sh`), sysctl files |
| `HPE/docs/` | this file, **`INVESTIGATIONS.md`** (every experiment and decision), `REPORT.md`, `ENGINEERING_NOTES.md` |
| `HPE/systems/`, `HPE/results/` | system description and the 50 scored logs of the final package |

Datasets, checkpoints and tuning logs are **not** in git; they are on the cluster
(`/nas/SHARED/MLPERF`, durable; `/lvol0`, node-local scratch).

## Still open before submitting

| Item | Who |
|---|---|
| System description: cooling and BIOS power settings marked "to verify" | you or the admin |
| Re-run the official check if MLCommons publishes a newer 6.1 logging release before the deadline | on request |

## Read before changing anything

`ENGINEERING_NOTES.md` holds the system facts and the traps that cost the most time:

| Trap | Effect |
|---|---|
| Starting a Docker job on a node whose previous job is still completing | The old job's cleanup kills the new container. No error is logged |
| Changing kernel settings by hand | The nodes reset them (prolog, reboot). Use Slurm `--comment` (`sysctl file=…`, `thp=…`) |
| Llama 2 hides errors by default | A crash looks like a hang. Set `MLLOG_VERBOSE_LOGS=1` |
| Comparing raw times with published scores | Published scores are RCP-normalised; raw times look better than they are |
| Choosing between run sets by score | Not allowed. With more than 10 runs, submit the median 10-run window |
