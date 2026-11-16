This is NVIDIA's MLPerf Training v6.1 reference implementation for llama31_8b
(NeMo Framework / Megatron-Bridge), with Crusoe's v6.1 logging updates applied
on top of the v6.0 base.

## Scale submitted

- 8 GPU (1 node x 8 GPU): `config_DGXB300_1x8x2xtp1pp1cp1_8b_fp4.sh`

This config sources `config_common.sh`, `config_common_fp4.sh`,
`config_common_cg.sh`, `config_common_8b.sh`, and `config_common_fp8attn.sh`
before setting its own hyperparameters and node count.

## Build

```bash
docker build -t mlperf-nvidia:llama31_8b-pyt .
enroot import -o llama31_8b.sqsh dockerd://mlperf-nvidia:llama31_8b-pyt
```

## Data

See `data_scripts/download_8b.sh` (data and tokenizer) and
`data_scripts/cleanup_8b.sh` (post-processing) for retrieving and preparing
the C4 dataset and tokenizer used by this benchmark, and `download_model.py`
for retrieving the base checkpoint.

## Run

Submit via Slurm + Pyxis/Enroot using `run.sub` with the appropriate
`config_DGXB300_*.sh` sourced first, per the standard NVIDIA MLPerf launch
convention (see `run_and_time.sh`).
