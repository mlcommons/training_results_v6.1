# GPT-OSS 20B: University of Florida files

This dir is NVIDIA's MLPerf Training v6.1 implementation, unchanged, from the image
`nvcr.io/nvdlfwea/mlperftv61/gpt_oss_20b-arm:20261006` (`GIT_COMMIT_ID=2647c2168340a97ec5b94ae56eb539194bf8a961`), rebuilt for x86 as `gpt_oss_20b-amd_20261006.sif` (see `BUILD_UF.md`).
NVIDIA's guide is `README.md`. Our files are the configs below and `uf_launcher/`.

## What to run

One config per result set. Each config sources one NVIDIA config (column "Base") and changes only the values in the table and the notes below.

| System | Config | GBS | Batch per GPU, parallelism | LR, warmup | Base |
|---|---|---|---|---|---|
| hpg.dgxb200x8.n2 | `config_DGXB200_2x8x1xtp1pp1cp1ep4_mxfp8.sh` | 16 | 1, EP 4 | 4e-4, 128 steps | `config_VR200_2x4x3xtp1pp1cp1ep1_mxfp8.sh` |
| hpg.dgxb200x8.n4 | `config_DGXB200_4x8x1xtp1pp1cp1ep4_mxfp8_lr4e4.sh` | 32 | 1, EP 4 | 4e-4, 256 steps | `config_VR200_2x4x3xtp1pp1cp1ep1_mxfp8.sh` |
| hpg.dgxb200x8.n8 | `config_DGXB200_8x8x1xtp1pp1cp1ep4_mxfp8_borrow.sh` | 64 | 1, EP 4 | 5.2e-4, 32 steps | `config_VR200_2x4x3xtp1pp1cp1ep1_mxfp8.sh` (via `config_DGXB200_8x8x1xtp1pp1cp1ep4_mxfp8_lr4e4.sh`) |
| hpg.dgxb200x8.n16 | `config_DGXB200_16x8x1xtp1pp1cp2ep4_mxfp8_ub_borrow.sh` | 64 | 1, CP 2, EP 4 | 5.2e-4, 32 steps | `config_VR200_18x4x1xtp1pp1cp2ep4_mxfp8.sh` (via `config_DGXB200_16x8x1xtp1pp1cp2ep4_mxfp8_ub_lr4e4.sh`) |
| hpg.dgxb200x8.n32 | `config_DGXB200_32x8x1xtp1pp1cp4ep4_mxfp8_lr4e4.sh` | 64 | 1, CP 4, EP 4 | 4e-4, 32 steps | `config_VR200_18x4x1xtp1pp1cp2ep4_mxfp8.sh` |

## How to run

`uf_launcher/` holds the launcher files that started the runs. Copy them to `<kit>/launcher/`, where `<kit>/University-of-Florida/` is this tree, then run from `<kit>`:

```bash
NEXP=10 launcher/run_hpg.sh gpt_oss_20b <config>    # HPG B200 (Slurm); RESERVATION=<name> is optional
```

`kit_defaults.sh` sets the image, the data dir and this dir. The container's own code runs. `run_hpg.sub` does the steps of NVIDIA's `run.sub` with apptainer (cache drop per run, one rank per GPU through `srun`, compliance checker). It binds writable host dirs over `${WORK_DIR}/outputs` (Hydra) and, with `MCORE_BUILD_BIND=1`, over Megatron-Core's `megatron/core/build` (the SIF is read-only).

Runs 1-2 of the n32 set ran an earlier `run_hpg.sub` without the 3 exports that move build caches to node-local `/tmp` (`TORCH_EXTENSIONS_DIR`, `CUDA_CACHE_PATH`, `HYBRID_EP_CACHE_DIR`). No training setting changed.

## Why we changed NVIDIA's settings

- All configs: `USE_MNNVL=0` (HGX nodes have no multi-node NVLink) and `MCORE_BUILD_BIND=1` (`NCCL_UB_DP=True` builds the NCCL allocator into `megatron/core/build`).
- 2 to 32 nodes: EP 4 inside one node with capacity factor 5, as in NVIDIA's EP 4 configs, and FP8 parameter gather.
- 16 and 32 nodes: CP 2 and CP 4 keep GBS at 64, the largest batch in the RCP table. They use NVIDIA's CP config (BF16 attention under CP) with NCCL user buffers on.
- LR: GBS 16 uses the RCP values. GBS 32 and 64 use LR 4e-4 (warmup 256 and 32 steps): with the RCP LR this code converges faster than the RCP minimum allows. LR and warmup are unconstrained for gpt_oss_20b. v6.0 sets with the same values: Cisco Zeus_16xH100 (GBS 64, LR 4e-4, warmup 32); Dell 16xXE9712x4GB300 and QCT D75U-1U_ngpu64 (GBS 32, LR 4e-4, warmup 256).
- Eval every 12,288 samples (`VAL_CHECK_INTERVAL` = 12288 / GBS).
- The 2- to 32-node configs were rewritten after the runs as single files (before, each sourced 1 or 2 of our base configs). The shell variables after `source` are the same (checked with `declare -p`).

## Result files

`result_<k>.txt` = run k+1 of the set. Sets with runs from more than one Slurm job (the same config): `result_<k>.txt` = the runs ordered by `run_start`.

- hpg.dgxb200x8.n2 (4 jobs at the same time), n4 (2 jobs), n8 (4 jobs): the runs of each set ran at the same time on separate nodes.
- hpg.dgxb200x8.n4: `result_4.txt` failed in initialization, before `run_start` (`EADDRINUSE` on port 53394). It has no `run_start`, so its `init_start` puts it in the order. It is the 1 failed run that olympic scoring allows.
- hpg.dgxb200x8.n32: runs 1-2 of one job and runs 1-8 of a second job (M = N = 10, fixed before the second job started).
- hpg.dgxb200x8.n8 (resubmitted 2026-10-28 in the review period, hyperparameter borrowing; issue #61): LR 5.2e-4 (end LR 5.2e-5) borrowed from NVIDIA `hecate_ngpu64_ngc26.09_nemo` (64 GPUs, GBS 64; Crusoe `crusoe_b300_n16_ngc26.04_nemo` uses the same values on the same pre-fix code as ours); warmup 32 steps unchanged. 10 runs of one Slurm job, 10 x compliance SUCCESS, `rcp_checker` PASSED; the set converges faster than the RCP mean, so `scaling.json` holds the factor 1.0051.
- hpg.dgxb200x8.n16 (resubmitted 2026-10-28 in the review period, hyperparameter borrowing; issue #59): LR 5.2e-4 (end LR 5.2e-5) borrowed from Crusoe `crusoe_b300_n16_ngc26.04_nemo` (128 GPUs, GBS 64, the same pre-fix code as ours); warmup 32 steps unchanged. 10 runs of one Slurm job, 10 x compliance SUCCESS, `rcp_checker` PASSED; `scaling.json` holds the factor 1.0262. The `_borrow` configs source the October `_lr4e4` configs and change only the LR; software, precision and parallelism are unchanged.
