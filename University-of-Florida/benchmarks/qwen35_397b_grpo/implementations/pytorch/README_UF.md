# Qwen3.5-397B GRPO: University of Florida files

This dir is NVIDIA's MLPerf Training v6.1 implementation, unchanged, from the image
`nvcr.io/nvdlfwea/mlperftv61/qwen35_397b_grpo_trtllm-arm:20261006` (NeMo-RL `4c823cd`, NGC PyTorch 26.09-beta base), rebuilt for x86 as `qwen35_397b_grpo_trtllm-amd_20261006.sif` (see `BUILD_UF.md`).
NVIDIA's guide is `README.md`. Our files are the configs below, `run_hpg_qwen.sub` (apptainer port of NVIDIA's `run.sub` for Slurm on HiPerGator: same Ray cluster start, same `run_and_time.sh`, same compliance checker call) and `run_hpg_qwen.sh` (submits it).

## What to run

One config per result set. Each config is a port of NVIDIA's `config_GB300_14x4_t8g6_tp4pp2ep16_2p4d16_mxfp8.sh`: it sources NVIDIA's `config_common.sh`, `config_optimizer_cpu_offload.sh` and `config_GBS512.sh` (LR 1.4142e-6, 32 prompts x 16 generations).

| System | Config | GBS | Training nodes | Generation nodes |
|---|---|---|---|---|
| hpg.dgxb200x8.n16 | `config_B200_16x8_t8g8_tp4pp4ep8_4p4d8_mxfp8.sh` | 512 | 8 (TP 4, PP 4, EP 8, DP 4) | 8 (4 TP4 context + 6 TP8 decode engines) |
| hpg.dgxb200x8.n32 | `config_B200_32x8_t16g16_tp4pp4ep8_8p4d8_mxfp8.sh` | 512 | 16 (TP 4, PP 4, EP 8, DP 8) | 16 (8 TP4 context + 12 TP8 decode engines) |

## How to run

From this dir:

```bash
./run_hpg_qwen.sh config_B200_16x8_t8g8_tp4pp4ep8_4p4d8_mxfp8.sh             # 16 nodes, 1 run per job (NEXP=1, the default); submit it 3 times for a set
NEXP=3 ./run_hpg_qwen.sh config_B200_32x8_t16g16_tp4pp4ep8_8p4d8_mxfp8.sh     # 32 nodes, 3 runs in 1 job
```

`RESERVATION=<name>` is optional. Each run: NVIDIA's untimed warmup, then `run_start` -> 11 training steps -> `run_stop`, then NVIDIA's deferred evaluation of the step-10 and step-11 checkpoints. Target: pass@4 >= 0.69 on the 251 validation tasks.

## Why we changed NVIDIA's settings

1. **Full activation recompute.** With NVIDIA's selective recompute, the first pipeline stage holds about 90 GB of activations and the warmup training step runs out of memory on a 178 GiB B200. The B200 configs set `RECOMPUTE_GRANULARITY=full` and `MOE_SHARED_EXPERT_OVERLAP=false`. NVIDIA's optimizer-state CPU offload and MXFP8 recipe are unchanged.
2. **EP 8 inside one node.** The image builds the HybridEP dispatcher for one NVLink domain, and a B200 node has 8 GPUs. So EP 8 (NVIDIA: EP 16 over an NVL72 rack) with PP 4 instead of PP 2. The expert load per GPU is the same.
3. **Sandbox user name.** libtmux rejects tmux session names with a period, and HiPerGator user names have one. `run_hpg_qwen.sub` sets `USER`/`LOGNAME` inside the task container to the user name with `.` replaced by `_`.
4. **Node-local Gym copy check.** At each job start, `run_hpg_qwen.sub` compares the OpenHands site-packages of the node-local Gym copy with the image and copies it again if it differs. After NVIDIA's `run_openhands.py` patch, the site-packages dirs are made read-only.
5. **No NVL72.** `SEGMENT=2`, `NEMORL_TRAIN_SEGMENT_SIZE=2` (node pairs). UCX and NCCL use the 8 GPU-local NDR rails of a B200 node. Disaggregated-serving frontend ports start at 7300.
6. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the trainer (fragmentation on 178 GiB without it). `MLPERF_WARMUP_TRAIN_TIMEOUT_S=1800`.
7. Apptainer: the SIF is read-only; `VERIFY_MOUNTS=0` (NVIDIA's `expected-mounts.csv` lists the aarch64 bundle). Task SIFs run as nested Apptainer containers.

## Result files

3 runs per set (the 6.1 rule for this benchmark). `result_<k>.txt` = run k+1 ordered by `run_start`.

- hpg.dgxb200x8.n16: 3 Slurm jobs with 1 run each (`NEXP=1`), submitted with `sbatch` on `run_hpg_qwen.sub` (the same file that `run_hpg_qwen.sh` submits). Each result file comes from a different job (3 DATESTAMPs).
- hpg.dgxb200x8.n32: 1 Slurm job with 3 runs (`NEXP=3`).
