# Llama 3.1 8B: University of Florida files

This dir is NVIDIA's MLPerf Training v6.1 implementation, unchanged, from the image
`nvcr.io/nvdlfwea/mlperftv61/llama31_8b-arm:20261006` (`GIT_COMMIT_ID=2647c2168340a97ec5b94ae56eb539194bf8a961`), rebuilt for x86 as `llama31_8b-amd_20261006.sif` (see `BUILD_UF.md`).
NVIDIA's guide is `README.md`. Our files are the configs below and `uf_launcher/`.

## What to run

One config per result set. Each config sources NVIDIA's `config_VR200_2x4x2xtp1pp1cp1_8b.sh` and changes only the values in the table.

| System | Config | GBS | Batch per GPU | LR, warmup |
|---|---|---|---|---|
| hpg.dgxb300x8.n1 | `config_DGXB300_1x8x2xtp1pp1cp1_8b.sh` | 16 | 2 | 4e-4, 16 steps |
| hpg.dgxb200x8.n1 | `config_DGXB200_1x8x2xtp1pp1cp1_8b.sh` | 16 | 2 | 4e-4, 16 steps |
| hpg.dgxb200x8.n2 | `config_DGXB200_2x8x2xtp1pp1cp1_8b_borrow.sh` | 32 | 2 | 8e-4, 64 steps |
| hpg.dgxb200x8.n4 | `config_DGXB200_4x8x1xtp1pp1cp1_8b_borrow.sh` | 32 | 1 | 8e-4, 64 steps |
| hpg.dgxb200x8.n8 | `config_DGXB200_8x8x1xtp1pp1cp1_8b.sh` | 64 | 1 | 8e-4, 96 steps |
| hpg.dgxb200x8.n16 | `config_DGXB200_16x8x1xtp1pp1cp2_8b.sh` | 64 | 1 (CP 2) | 8e-4, 96 steps |
| hpg.dgxb200x8.n32 | `config_DGXB200_32x8x1xtp4pp1cp1_8b.sh` | 64 | 1 (TP 4) | 8e-4, 96 steps |

## How to run

`uf_launcher/` holds the launcher files that started the runs. Copy them to `<kit>/launcher/`, where `<kit>/University-of-Florida/` is this tree, then run from `<kit>`:

```bash
NEXP=10 launcher/run_hpg_llama3.sh llama31_8b <config>    # HPG B200 (Slurm); RESERVATION=<name> is optional
nohup launcher/run_b300.sh llama31_8b config_DGXB300_1x8x2xtp1pp1cp1_8b.sh > b300.out 2>&1 &    # B300 node (no Slurm)
```

`kit_defaults.sh` sets the image, the data dir and this dir. The container's own code runs. `run_hpg_llama3.sub` does the steps of NVIDIA's `run.sub` with apptainer (cache drop per run, one rank per GPU through `srun`, compliance checker). `run_b300.sh` starts one rank per GPU through the container's `slurm2pytorch`.

## Why we changed NVIDIA's settings

- 1 node: an 8-GPU B200 or B300 node has the GPUs of NVIDIA's 2 x 4 VR200 config, so the per-GPU batch and the hyperparameters stay NVIDIA's.
- 2 to 32 nodes: 1 sample per GPU, as in NVIDIA's 72-GPU config. LR and warmup are the RCP values of the GBS. Eval every 12,288 samples (`VAL_CHECK_INTERVAL` = 12288 / GBS).
- 16 and 32 nodes: CP 2 and TP 4 keep GBS at 64, the batch of the RCP.
- The 2- to 32-node configs were rewritten after the runs as single files (before, each sourced the 1-node config). The shell variables after `source` are the same (checked with `declare -p`).

## Result files

`result_<k>.txt` = run k+1 of the set.

- hpg.dgxb200x8.n1: run 2 (`result_1.txt`) crashed at start (`EADDRINUSE` on port 53394, before `init_stop`). It is the 1 failed run that olympic scoring allows.
- hpg.dgxb200x8.n2 (resubmitted 2026-10-28 in the review period, hyperparameter borrowing; issue #265): GBS 32 (2 sequences per GPU), LR 8e-4, end LR 8e-5 and warmup 64 steps borrowed from HPE `HPE-Cray-XD690-B300-SXM6-288GB_n2_ngc26.04` (16 GPUs). `config_DGXB200_2x8x2xtp1pp1cp1_8b_borrow.sh` sources `config_DGXB200_2x8x1xtp1pp1cp1_8b.sh` and changes only these values (eval every 384 steps = 12,288 samples). 10 runs of one Slurm job, 10 x compliance SUCCESS, `rcp_checker` PASSED (no scaling).
- hpg.dgxb200x8.n4 (resubmitted 2026-10-28 in the review period, hyperparameter borrowing; issue #52): warmup 64 steps borrowed from HPE `HPE-GB300-ngpu32_ngc26.09` (32 GPUs, GBS 32); the other hyperparameters were already equal. `config_DGXB200_4x8x1xtp1pp1cp1_8b_borrow.sh` sources `config_DGXB200_4x8x1xtp1pp1cp1_8b.sh`. 10 runs of one Slurm job, 10 x compliance SUCCESS, `rcp_checker` PASSED (no scaling).
