This is NVIDIA's MLPerf Training v6.1 reference implementation for gpt_oss_20b
(NeMo Framework / Megatron-Bridge), with Crusoe's v6.1 logging updates applied
on top of the v6.0 base.

## Scales submitted

- 8 GPU (1 node x 8 GPU): `config_DGXB300_1x8x3xtp1pp1cp1ep1_mxfp8.sh`
- 64 GPU (8 nodes x 8 GPU): `config_DGXB300_8x8x1xtp1pp1cp2ep4_mxfp8.sh`
- 128 GPU (16 nodes x 8 GPU): `config_DGXB300_16x8x1xtp1pp1cp2ep4_mxfp8.sh`

Each config sources `config_common.sh`, `config_common_mxfp8.sh`, and
`config_common_cg.sh` before setting its own hyperparameters and node count.

## Build

```bash
docker build -t mlperf-nvidia:gpt_oss_20b-pyt .
enroot import -o gpt_oss_20b.sqsh dockerd://mlperf-nvidia:gpt_oss_20b-pyt
```

## Data

See `data_scripts/download.sh` (data) and `data_scripts/cleanup.sh`
(post-processing) for retrieving and preparing the C4 dataset used by this
benchmark.

## Run

Submit via Slurm + Pyxis/Enroot using `run.sub` with the appropriate
`config_DGXB300_*.sh` sourced first, per the standard NVIDIA MLPerf launch
convention (see `run_and_time.sh`).
