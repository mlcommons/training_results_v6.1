# Expert padding stays zero

Classification: Internal (becomes public with the MLPerf Training v6.1 results)

Run evidence for [`../../KERNELS.md` 3.1](../../KERNELS.md#31-expert-width-2880--2944): every
padded entry of the 2880 → 2944 expert weights is exactly zero, in the model parameters,
their gradients and the optimizer state, before and during training and around evaluation.

| File | Content |
|---|---|
| `pad_zero_check.py` | The check, hooked into the submission's logging and warmup callbacks |
| `sitecustomize.py` | Loads the hook when `PADCHECK=1`; chains any existing `sitecustomize` |
| `launch_pad_zero_check.sh` | Submits one run of the submitted image and config |
| `pad_zero_check.out` | The check's output from that run |

## How it runs

`launch_pad_zero_check.sh` sources `config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_v7.sh` and submits the
unchanged `run.sub` → `run_and_time.sh` → `pretrain.py` of `../../../nemo_g6c_nvfp4_v2_2/` on the
submitted image. This directory is mounted read-only at `/padcheck` (`EXTRA_MOUNTS`) and put on
`PYTHONPATH`; `sitecustomize.py` then replaces `MLPerfLoggingCallback` and `WarmupCallback` with
subclasses that call the original methods and add the check. Differences from a scoring run, all
for this check only:

- evaluation every 10 steps (`VAL_CHECK_INTERVAL=10`; submitted value 512), so that a short run
  passes through evaluation twice;
- the run stops after step 30 and logs `run_stop` with status `aborted`;
- one run (`NEXP=1`).

## What is checked

For each expert module (24 layers × 32 experts, all on every rank), the padded region is:

| Tensor | Shape | Padded entries |
|---|---|---|
| FC1 weight | `[5888, 2944]` | rows 5760–5887 (the padded FFN channels after GLU interleave 32; the layout is asserted at run time) and input columns 2880–2943 |
| FC2 weight | `[2944, 2944]` | output rows 2880–2943 and input columns 2880–2943 |

The model has no expert biases (`add_bias_linear=False`); the check covers biases when they exist
and finds none here.

For each of these, the check reads the BF16 parameter and its `main_grad` on every rank, and the
distributed optimizer's shards of the FP32 master remainder (`master_param`, int16), `exp_avg` and
`exp_avg_sq` (FP32). The FP32 master value is the BF16 parameter plus the remainder, so both being
zero means the master is zero. For each tensor it counts nonzero entries, negative zeros and the
maximum absolute value over the padded entries, summed (counts) or maximized over the 8 ranks.

Check points: before warmup (`init_before_warmup`, parameters and gradients only; the optimizer
state does not exist yet), at train start, at the end of steps 1, 2, 10, 11, 20, 21, 30, and at
the start and end of both evaluations (steps 10 and 20). `PADCHECK_SUMMARY` counts the 12 check
points of the logging callback; `init_before_warmup` runs in the warmup callback.

As a control that training is moving the weights, rank 0 also records, for expert 0 of each layer,
the largest absolute change of the FC1 weight since train start, separately for the logical
region, the padded rows and the padded columns.

## Result

Job 22790, 2026-10-16, two nodes (4 GB300 GPUs each), image
`firmus+gpt-oss-20b-nvfp4-g6c-v2+v61.sqsh`, image ID
`sha256:d69e3e51f4a18273d977c9c0767d7dae680f73c0bde166995ec0101120076869`, squashfs sha256
`fd815c589765300f2f559f6635c791b3a16f59aed319524abd64f789399aea7a`.

- All 124 tensor checks report `nonzero: 0`, `negative_zero: 0`, `max_abs: 0.0`;
  `PADCHECK_SUMMARY` `all_padded_entries_zero_at_every_check: true`, 12 checks.
- Padded entries checked per check point, summed over ranks: FC1 weight parameter and gradient
  4,580,179,968; FC2 weight 2,290,089,984. Optimizer shards (each element held by one rank):
  FC1 weight 572,522,496; FC2 weight 286,261,248.
- Expert 0 FC1 weight change since train start, maximum over layers (check point at the end of
  the step; at steps 10 and 20 also at the start and end of evaluation):

| Step | Logical | Padded rows | Padded columns |
|---|---|---|---|
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 10 (end, eval start, eval end) | 4.77e-7 | 0 | 0 |
| 11 | 4.77e-7 | 0 | 0 |
| 20 (end) | 9.54e-7 | 0 | 0 |
| 20 (eval start, eval end) | 1.91e-6 | 0 | 0 |
| 21 | 1.91e-6 | 0 | 0 |
| 30 | 1.91e-6 | 0 | 0 |

The raw log of the run is kept with the submission's run evidence, not in this package.
