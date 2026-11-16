This is NVIDIA's MLPerf Training v6.1 reference implementation for
llama2_70b_lora (NeMo Framework), with Crusoe's v6.1 logging updates applied
on top of the v6.0 base.

## Scales submitted

- 8 GPU (1 node x 8 GPU): `config_DGXB300_1x8x1xtp1pp1cp1_fp4.sh`
- 16 GPU (2 nodes x 8 GPU): `config_DGXB300_2x8x1xtp1pp1cp1_fp4.sh`
- 32 GPU (4 nodes x 8 GPU): `config_DGXB300_4x8x1xtp1pp1cp4_fp4.sh`

Each config sources `config_common.sh` and `config_common_fp4.sh` before
setting its own hyperparameters and node count.

## Build

```bash
docker build -t mlperf-nvidia:llama2_70b_lora-pyt .
enroot import -o llama2_70b_lora.sqsh dockerd://mlperf-nvidia:llama2_70b_lora-pyt
```

## Data

See `scripts/download_dataset.py` and `scripts/download_model.py` for
retrieving the dataset and base checkpoint, and `scripts/convert_model.py`
for converting the checkpoint to the format expected by training.

## Run

Submit via Slurm + Pyxis/Enroot using `run.sub` with the appropriate
`config_DGXB300_*.sh` sourced first, per the standard NVIDIA MLPerf launch
convention (see `run_and_time.sh`).
