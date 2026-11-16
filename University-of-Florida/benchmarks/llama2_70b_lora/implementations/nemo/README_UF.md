# Llama 2 70B LoRA: University of Florida files

This dir is NVIDIA's MLPerf Training v6.1 implementation, unchanged, from the image
`nvcr.io/nvdlfwea/mlperftv61/llama2_70b_lora-arm:20261006` (`GIT_COMMIT_ID=2647c2168340a97ec5b94ae56eb539194bf8a961`), rebuilt for x86 as `llama2_70b_lora-amd_20261006.sif` (see `BUILD_UF.md`).
NVIDIA's guide is `README.md`. Our files are the configs below and `uf_launcher/`.

## What to run

One config per result set. Each config sources NVIDIA's `config_VR200_2x4x1xtp1pp1cp1_fp4.sh` and changes only the values in the table. All sets use the v6.1 dataset and checkpoint (`scripts/download_dataset.py`, `scripts/download_model.py`).

| System | Config | GBS | Batch per GPU | LR, max steps, healing step |
|---|---|---|---|---|
| hpg.dgxb300x8.n1 | `config_DGXB300_1x8x1xtp1pp1cp1_fp4.sh` | 8 | 1 | 6e-4, 550, 350 |
| hpg.dgxb200x8.n1 | `config_DGXB200_1x8x2xtp1pp1cp2_fp4.sh` | 8 | 1 x 2 grad. acc. (CP 2) | 6e-4, 550, 350 |
| hpg.dgxb200x8.n2 | `config_DGXB200_2x8x1xtp1pp1cp2_fp4.sh` | 8 | 1 (CP 2) | 6e-4, 550, 350 |
| hpg.dgxb200x8.n4 | `config_DGXB200_4x8x1xtp1pp1cp2_fp4.sh` | 16 | 1 (CP 2) | 4e-4, 1024, 175 |
| hpg.dgxb200x8.n8 | `config_DGXB200_8x8x1xtp1pp1cp2_fp4_borrow.sh` | 32 | 1 (CP 2) | 6e-4, 600, 88 |
| hpg.dgxb200x8.n16 | `config_DGXB200_16x8x1xtp1pp1cp2_fp4_borrow.sh` | 64 | 1 (CP 2) | 6e-4, 600, 88 |
| hpg.dgxb200x8.n32 | `config_DGXB200_32x8x1xtp1pp1cp2_fp4_borrow.sh` | 128 | 1 (CP 2) | 6e-4, 600, 88 |

## How to run

`uf_launcher/` holds the launcher files that started the runs. Copy them to `<kit>/launcher/`, where `<kit>/University-of-Florida/` is this tree, then run from `<kit>`:

```bash
NEXP=10 launcher/run_hpg.sh llama2_70b_lora <config>    # HPG B200 (Slurm); RESERVATION=<name> is optional
nohup launcher/run_b300.sh llama2_70b_lora config_DGXB300_1x8x1xtp1pp1cp1_fp4.sh > b300.out 2>&1 &    # B300 node (no Slurm)
```

`kit_defaults.sh` sets the image, the dataset dir, the checkpoint dir (`MODEL`) and this dir. The container's own code runs. `run_hpg.sub` does the steps of NVIDIA's `run.sub` with apptainer (cache drop per run, one rank per GPU through `srun`, compliance checker); it binds a writable host dir over `${WORK_DIR}/outputs` (Hydra writes there; the SIF is read-only). `run_b300.sh` starts one rank per GPU through the container's `slurm2pytorch`.

## Why we changed NVIDIA's settings

- 1 node: an 8-GPU node has the GPUs of NVIDIA's 2 x 4 VR200 config, so the hyperparameters stay NVIDIA's.
- B200 uses CP 2: with CP 1 (NVIDIA) the first forward pass runs out of memory on 178 GiB. At 1 node, 2 micro-batches per GPU keep GBS at 8.
- 4 and 8 nodes: LR and max steps of the RCP of the GBS. 16 and 32 nodes: the LR is unconstrained for this benchmark; LR 5e-4 at GBS 64 is the LR of NVIDIA's 72-GPU config.
- Healing step (FP4 to FP8): NVIDIA heals at step 350 at GBS 8 (2,800 samples). The 4- and 8-node configs keep about the same sample point (step 175, step 88). The 16- and 32-node configs keep step 88. NVIDIA has no FP4 LoRA config at scale.
- The 2- to 32-node configs were rewritten after the runs as single files (before, each sourced the 1-node config). The shell variables after `source` are the same (checked with `declare -p`).

## Result files

`result_<k>.txt` = run k+1 of the set.

- hpg.dgxb200x8.n8, n16, n32 (resubmitted 2026-10-28 in the review period, hyperparameter borrowing; issues #264, #263, #262): LR 6e-4 and 600 training steps borrowed from NVIDIA `hecate_ngpu512_ngc26.09_nemo` (512 GPUs, GBS 64). The batch sizes stay ours (32, 64, 128: 1 sequence per GPU with CP 2; the rule allows the batch size to differ). The `_borrow` configs source the October configs (`config_DGXB200_8x8x1xtp1pp1cp2_fp4.sh`, `config_DGXB200_16x8x1xtp1pp1cp2_fp4_lr5e4.sh`, `config_DGXB200_32x8x1xtp1pp1cp2_fp4.sh`) and change only these two values; software, precision and parallelism are unchanged. Each set: 10 runs of one Slurm job, 10 x compliance SUCCESS, `rcp_checker` PASSED (no scaling).
