# Krai kernel changes since Recipe A

This document lists every performance change Krai made to AMD's MLPerf Training v5.1 reference implementation
(`README_upstream_AMD_v5.1.md`) for the llama2_70b_lora benchmark on 8× AMD Instinct MI300X.  Changes are
grouped by the submission version in which they first appeared.  Krai submitted **v3.1** and **v4.2** (from the
MLCommons v6.1 training round); **v6** is the version in this folder.

All changes are performance-only.  Hyperparameters, optimizer, model initialisation, data traversal order,
evaluation frequency and convergence criterion are unchanged from AMD's reference.

---

## v3.1 (first submitted)

Olympic mean: ~26 min on 8× MI300X.

| Variable / patch | What it does | Numerics |
|---|---|---|
| `ACL=3` | 3 activation-checkpointed layers instead of 21; frees HBM for larger activations in the no-recompute layers without the full memory impact of `ACG=null` | unchanged (recompute only) |
| `FP8_ACT_STORE=1`, `HS_ACT_STORE_FUSED=1` (`hs_act_store.py`, `hs_swiglu_fp8_triton.py`) | SwiGLU input stored as FP8 e4m3 for the backward pass (Megatron `activation_func_fp8_input_store`); SwiGLU forward/backward fused in Triton | bit-identical to TE's kernels |
| `HS_LORA_EPILOGUE_ADD/RESIDUAL/BWD=1` (`hs_lora_epilogue.py`) | LoRA output, scale and residual adds folded into single passes | BF16 add order changes (≤ 1 ulp) |
| `HS_SWIGLU_BWD_FP8=1` (`hs_dswiglu_e5m2.py`) | SwiGLU backward writes the FP8 e5m2 fc1 gradient directly with fc1's delayed-scaling quantizer | bit-identical to the BF16 gradient + TE cast it replaces |
| `HS_ROPE_FUSED=1` (`hs_rope_qkv.py`) | qkv split + RoPE fused (forward); RoPE backward + concat fused | bit-identical to Megatron/apex |
| `HS_GRAD_CAST_ONLY=1` (`te_patches/`, copied by `Dockerfile`) | grad-output FP8 cast without the unused transposed copy (frozen LoRA adapter base weights) | unchanged |
| `NVTE_CK_HOW_V3_BF16_CVT=2` | round-toward-zero BF16 conversion in v3 CK attention backward (matches AMD's v5.1 config) | as v5.1 |
| `HS_EMPTY_CACHE_AT_EVAL=1` | allocator pool released before and after each evaluation step to avoid OOM from evaluation's larger activation footprint | unchanged |

---

## v4.2 (second submitted)

Olympic mean: 23.89 min.  Changes relative to v3.1:

| Variable / patch | What it does | Numerics |
|---|---|---|
| `ACG=null ACM=null ACL=null` | Full no-activation-recomputation (all 80 layers); `lora_dropout_packed` frees the extra HBM needed | unchanged (recompute only) |
| `lora_dropout_packed` module (`lora_dropout_packed.py`) | LoRA dropout masks stored bit-packed (1 bit per element instead of 1 byte), freeing ~10 GB HBM to allow full no-recompute; backward reads packed bits directly | bit-identical; masks from `torch.native_dropout` RNG |
| `HS_LORA_EPILOGUE_FAST=1` (`hs_lora_epilogue.py`) | LoRA up-projection epilogue kernel restructured for memory bandwidth (more efficient tile layout) | bit-identical to the kernel it replaces |
| `HS_RMSNORM_GRID=1` (`hs_rmsnorm_grid.py`) | Transformer Engine's Triton RMSNorm forward launched with a larger fixed grid (avoids grid under-occupancy at this batch size) | bit-identical to TE's launch |
| `HS_SWIGLU_FWD_FP8=1` (`hs_swiglu_fwd_e4m3.py`) | SwiGLU forward writes fc2's FP8 e4m3 input directly with fc2's delayed-scaling quantizer | bit-identical to the BF16 output + TE cast it replaces |
| `HS_TCACHE=qkv,proj` (`hs_transpose_cache.py`) | TE's FP8 weight-transpose cache kept across steps for the frozen qkv and proj weights (built once at step 1 instead of every backward) | unchanged (same bytes) |
| `HS_RESGRAD_FOLD=1` (`hs_resgrad_fold.py`) | Residual-stream gradient accumulation and the e5m2 grad cast of proj/fc2 folded into the RMSNorm backward (copy of TE's kernel with an epilogue) | bit-identical to TE's norm backward + BF16 add + TE cast |
| `HS_QKVGRAD_E5M2=1` (`hs_resgrad_fold.py`) | Fused RoPE backward also writes the e5m2 copy of the qkv gradient for the qkv GEMM | bit-identical to the TE cast it replaces |
| `HS_QKVNORM_E4M3=1` (`hs_qkvnorm_e4m3.py`) | qkv RMSNorm writes its BF16 output (for LoRA) and the e4m3 GEMM input in one pass (copy of TE's kernel with an epilogue) | bit-identical to TE's norm + TE cast |
| `HS_HOST_SYNC=1` (`hs_host_sync.py`) | No host synchronisation at the step boundary: on-stream waits for asynchronous gradient all-reduces; logging scalars created on device; no progress-bar readback; batch tensors pinned | unchanged (scheduling only) |
| `TE_HIPBLASLT_ALGO_LOAD=patches/te_algos_fix7.csv` | hipBLASLt solution 322884 selected for the fc2 forward FP8 GEMM (8192×8192×28672); tuned offline with `TE_HIPBLASLT_TUNING_RUN_COUNT=10 ALGO_COUNT=100` | accumulation order of that GEMM changes (library algorithm choice) |

---

## v6 (this submission)

Score: 22.83 min (olympic mean 22.76 min × RCP normalisation 1.0034; −1.06 min vs v4.2).  Changes relative to v4.2:

| Variable / patch | What it does | Numerics |
|---|---|---|
| `HS_LORA_DGRAD_PACKED_FAST=1` (`hs_lora_dgrad_packed.py`) | LoRA dgrad kernel restructured for memory bandwidth when reading packed dropout masks: 1-D streaming grid (32×128 tiles, 4 warps, `.cg` loads), mask expanded from packed bits in registers; replaces the 2-D copy-of-fix-2 kernel that `lora_dropout_packed` used for the packed-mask path; 76 µs instead of 302 µs per call | bit-identical to the packed kernel it replaces |
| `eval_trim` module (`eval_trim.py`) | Each validation row is cut after its last loss token, rounded up to 512 tokens; rows served longest-first for load balance across the 8 ranks; a token count TE's Triton cast/transpose and RMSNorm kernels have not seen reuses the configuration tuned for the training shape instead of autotuning inside the eval (which produced NaN); evals take 22.0 s instead of 29.5 s, about 40–49 s per run | validation loss unchanged: cut positions are dead computation under causal attention; `sample_weight: constant` loss is the mean of per-row means, unaffected by row order |

Not in v6: fused LoRA dropout (`hs_lora_dropout_fast.py`) and the extra hipBLASLt solutions (`te_algos_all_gemm.csv`).
