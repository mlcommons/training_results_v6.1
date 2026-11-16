# Model built by the run, and evaluation samples

Classification: Internal (becomes public with the MLPerf Training v6.1 results)

Run evidence that the submitted image and config build NVIDIA's 20261012 reference model (no
linear biases, SiLU-gated experts without clamp, vanilla softmax without sinks, QK RMSNorm), and
that every evaluation reads the same validation samples.

| File | Content |
|---|---|
| `refmodel_check.py` | The check, hooked into the submission's logging callback |
| `sitecustomize.py` | Loads the hook when `REFCHECK=1` |
| `launch_refmodel_check.sh` | Submits one run of the submitted image and config |
| `refmodel_check.out` | The check's output from that run (rank-0 `REFCHECK` lines) |

## How it runs

`launch_refmodel_check.sh` sources `config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh` and submits the
unchanged `run.sub` → `run_and_time.sh` → `pretrain.py` of `../../../nemo_g6c_nvfp4_v2_2/` on the
submitted image, with this directory mounted read-only at `/refcheck` and put on `PYTHONPATH`.
Differences from a scoring run, for this check only: evaluation every 10 steps
(`VAL_CHECK_INTERVAL=10`; submitted value 512), the run stops after step 30, one run (`NEXP=1`).

What is recorded is read from the built model, not from the config files: the
`TransformerConfig` fields that define the architecture (`REFCHECK_CONFIG`), per layer the core
attention class, `window_size`, `softmax_type` and the QK norm classes (`REFCHECK_LAYER`), every
parameter name with its shape and count (`REFCHECK_PARAMS`), and, for every evaluation, the union
over ranks of the sample indices the validation sampler yields (`REFCHECK_EVAL`).

## Result

Job 22789, 2026-10-16, two nodes (4 GB300 GPUs each), image
`firmus+gpt-oss-20b-nvfp4-g6c-v2+v61.sqsh`, image ID
`sha256:d69e3e51f4a18273d977c9c0767d7dae680f73c0bde166995ec0101120076869`, squashfs sha256
`fd815c589765300f2f559f6635c791b3a16f59aed319524abd64f789399aea7a`.

- `REFCHECK_CONFIG`: `add_bias_linear: false`, `add_qkv_bias: false`,
  `activation_func: torch.nn.functional.silu`, `activation_func_clamp_value: null`,
  `gated_linear_unit: true`, `glu_linear_offset: 0.0`, `softmax_type: vanilla`,
  `qk_layernorm: true`, `normalization: RMSNorm`, `layernorm_epsilon: 1e-06`, 24 layers, hidden
  2880, 64 attention heads, 8 query groups, `kv_channels` 64, 32 experts, top-4, router
  `softmax`, `moe_router_load_balancing_type: none`, RoPE base 150000, `window_attn_skip_freq` 2,
  `dataset_eod_mask_loss: true`.
- `REFCHECK_LAYER`: all 24 layers `TEDotProductAttention`, `softmax_type` `vanilla`, q and k norm
  `_QKNormImpl` with eps 1e-06. `window_size` `[128, 0]` on layers 0, 2, …, 22 and `[-1, 0]` on
  layers 1, 3, …, 23. (`REFCHECK_MODEL` `sliding_window_layers` lists all 24 because the script
  counts every `window_size` other than `None` and `(-1, -1)`.)
- `REFCHECK_MODEL`: `bias_parameters: []`, `softmax_offset_or_sink_parameters: []`, 24 q-norm and
  24 k-norm layers.
- `REFCHECK_PARAMS`: 12 parameter groups, none a bias; per layer `self_attention.q_layernorm.weight`
  and `k_layernorm.weight` `[64]`, `linear_qkv.weight` `[5120, 2880]`, `linear_proj.weight`
  `[2880, 4096]`, router `[32, 2880]`, expert FC1 `[5888, 2944]` and FC2 `[2944, 2944]` (padded,
  see `../padding_zero/`).
- `REFCHECK_EVAL`: two evaluations (steps 10 and 20), each 1,032 samples, 1,032 unique, indices
  0–1031, 129 per rank, sorted-list sha256 `dea1e890…c4cd` both times; `REFCHECK_SUMMARY`
  `every_eval_same_samples: true`.

The raw log of the run is kept with the submission's run evidence, not in this package.
