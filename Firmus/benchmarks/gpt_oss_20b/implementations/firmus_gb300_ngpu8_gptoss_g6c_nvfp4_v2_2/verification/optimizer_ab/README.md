# Optimizer A/B comparison

Classification: Internal (becomes public with the MLPerf Training v6.1 results)

Demonstrates, for training rule "Hyperparameters" (alternate optimizer implementations must
"describe the optimizer's equation and demonstrate equivalence"), that the fused clip+Adam path
used by this submission produces the same parameters and optimizer state as the reference
optimizer path and as `apex.optimizers.FusedAdam`. The equations are in [`../../KERNELS.md`](../../KERNELS.md#1-optimizer).

| File | Content |
|---|---|
| `ab_clip_adam.py` | The comparison |
| `ab_clip_adam.out` | Its output on one GB300 GPU with the submitted container image |

## What is compared

All three sides start from the same parameters and receive the same gradients for 20 steps.
Gradient magnitudes alternate so that 10 steps are clipped (global norm about 17.4) and 10 are
not (about 0.17). Learning rate follows the submitted warmup (peak 5e-4, 256 warmup steps);
betas (0.9, 0.95), epsilon 1e-5, clip norm 1.0, weight decay 0.1 for one parameter group and
0.0 for the other. Nine tensors, 33,788,488 elements, including the model's expert weight shapes
and odd sizes that use the kernel's unaligned tail and multi-chunk paths.

| Side | Code |
|---|---|
| ref | Megatron-Core `get_grad_norm_fp32` + `clip_grad_by_total_norm_fp32`, then Transformer Engine `FusedAdam` configured as Megatron builds it for the precision-aware optimizer (`master_weights=True`, FP32 master stored as BF16 param + int16 remainder, `use_decoupled_grad=True`, `adam_w_mode=True`) |
| ext | The same `FusedAdam` with the submission's switches: `NVFP4_FAST_L2NORM`, `NVFP4_CLIP_IN_ADAM`, `NVFP4_ADAM_AMAX`, `NVFP4_PHASED_ADAM_AG=4`, using the extension built in the `Dockerfile` (`/opt/nvfp4_ext/nvfp4_clip_adam_ext_fm9.so`) |
| apex | `apex.optimizers.FusedAdam(adam_w_mode=True)` on FP32 copies of the parameters, given the unclipped gradients, clipped on its own copy with the same Megatron-Core functions |

The FP32-parameter kernel of the extension (no remainder) is compared the same way. Each
comparison is run on the recipe's values and on edge inputs (+0, -0 and subnormal parameters
and gradients, and lr = 0 on steps 5, 10, 15, 20).

Run on one NVIDIA GB300 GPU in the submitted image (image ID
`sha256:d69e3e51f4a18273d977c9c0767d7dae680f73c0bde166995ec0101120076869`, squashfs sha256
`fd815c589765300f2f559f6635c791b3a16f59aed319524abd64f789399aea7a`), Slurm job 22924: torch 2.14.0a0+4fdf77b940.nv26.08, Transformer Engine 2.18.0+27486e03,
apex 0.1.

## Result

| Comparison | Recipe values, 20 steps | Edge inputs, 20 steps |
|---|---|---|
| ext vs ref: gradient norm | bitwise equal every step | bitwise equal every step |
| ext vs ref: BF16 params, int16 remainders, `exp_avg`, `exp_avg_sq` | 0 differing elements | 0 differing elements |
| ext vs ref: amax folded into the Adam kernel vs `param.abs().amax()` | 0 of 9 slots differ, every step | 0 of 9 slots differ, every step |
| ext vs apex: FP32 master weights | 0 ulp, 0 sign differences | 0 ulp, 0 sign differences |
| ext vs apex: `exp_avg`, `exp_avg_sq` | 0 relative difference | 0 relative difference |
| ext vs apex: BF16 param vs round-to-nearest-even of apex's FP32 master | 42 to 261 elements per step differ; all are exact BF16 ties (the reference TE storage rounds ties away from zero) | 52 to 108 elements per step differ; all are exact BF16 ties |
| FP32-parameter kernel: ext vs ref | 0 differing elements | 0 differing elements |
| FP32-parameter kernel: ext vs apex | 0 ulp, 0 sign differences | 0 ulp, 0 sign differences |
| Fused launches / fallbacks to the unfused path | 400 / 0, both columns together | |

## Reproduce

```bash
srun -N1 --gpus-per-node=1 --container-image=<image> \
  --container-mounts=$PWD:/verify --container-workdir=/workspace/llm \
  env AB_IMAGE="<image ID, squashfs sha256>" python /verify/ab_clip_adam.py > ab_clip_adam.out
```

`AB_STEPS` and `AB_PHASES` change the number of steps and optimizer phases.
