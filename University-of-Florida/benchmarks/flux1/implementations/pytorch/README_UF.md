# Flux.1: University of Florida files

This dir is NVIDIA's MLPerf Training v6.1 implementation, unchanged, from the image
`nvcr.io/nvdlfwea/mlperftv61/flux1-arm:20261006` (`GIT_COMMIT_ID=2647c2168340a97ec5b94ae56eb539194bf8a961`), rebuilt for x86 as `flux1-amd_20261006.sif` (see `BUILD_UF.md`).
NVIDIA's guide is `README.md`. Our files are the configs below and `uf_launcher/`.

## What to run

One config per result set. Each config sources NVIDIA's `config_GB300_02x04x64.sh` and changes only the values in the table and the notes below.

| System | Config | GBS | Batch per GPU | LR, warmup |
|---|---|---|---|---|
| hpg.dgxb300x8.n1 | `config_DGXB300_01x08x64.sh` | 512 | 64 | 2e-4, 2400 steps |
| hpg.dgxb200x8.n2 | `config_DGXB200_02x08x32.sh` | 512 | 32 | 2e-4, 2400 steps |
| hpg.dgxb200x8.n4 | `config_DGXB200_04x08x32.sh` | 1024 | 32 | 2.5e-4, 800 steps |
| hpg.dgxb200x8.n8 | `config_DGXB200_08x08x16.sh` | 1024 | 16 | 2.5e-4, 800 steps |
| hpg.dgxb200x8.n16 | `config_DGXB200_16x08x16.sh` | 2048 | 16 | 2.5e-4, 0 steps |
| hpg.dgxb200x8.n32 | `config_DGXB200_32x08x16.sh` | 4096 | 16 | 4e-4, 100 steps |

## How to run

`uf_launcher/` holds the launcher files that started the runs. Copy them to `<kit>/launcher/`, where `<kit>/University-of-Florida/` is this tree, then run from `<kit>`:

```bash
NEXP=10 launcher/run_hpg.sh flux1 <config>    # HPG B200 (Slurm); RESERVATION=<name> is optional
nohup launcher/run_b300.sh flux1 config_DGXB300_01x08x64.sh > b300.out 2>&1 &    # B300 node (no Slurm)
```

`kit_defaults.sh` sets the image, the data dir and this dir. The container's own code runs. `run_hpg.sub` does the steps of NVIDIA's `run.sub` with apptainer (cache drop per run, one rank per GPU through `srun`, compliance checker); it binds a writable host dir over `/hf_home` (the SIF is read-only). `run_b300.sh` starts one rank per GPU through the container's `slurm2pytorch`.

## Why we changed NVIDIA's settings

- GBS 512: NVIDIA's `config_common_gbs_512.sh` values. GBS 1024, 2048 and 4096: the RCP values of the GBS (flux_ref_1024 = NVIDIA's v6.0 GBS-1024 values).
- `train_flat.py` has no gradient accumulation: per-GPU batch = GBS / GPUs. `SEGMENT=1` (one NVLink domain per HGX node).
- `FLAT_AMAX_ARENA=0` in all configs: NVIDIA's `flat/custom_ops/amax_arena.py` calls CUDA 13 `cudaStreamGetCaptureInfo` through ctypes with 6 of its 7 arguments, which segfaults at step 1 on x86.

## Result files

`result_<k>.txt` = run k+1 of the set.

- hpg.dgxb200x8.n2: the 10 runs came from 3 Slurm jobs with the same config that ran at the same time (4 + 3 + 3 runs); `result_<k>.txt` = the runs ordered by `run_start`.
