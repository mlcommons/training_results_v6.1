# Kernel supplementary information: the Firmus kernels of nemo_l4_nvfp4_v3 (llama31_8b)

Classification: Internal (becomes public with the MLPerf Training v6.1 results)

This implementation is NVIDIA's MLPerf Training v6.1 `llama31_8b` recipe (release 20261006) with 16 Firmus kernels
(K1-K16) and 16 patch, wiring and configuration items (N1-N16) ([README.md](README.md)). This file is the index:
section 1 lists every item with its class, rule basis and result; sections 2-4 give the properties and facts that apply
across items; section 5 lists the tests. K1-K16 and N1-N13 each have a write-up under [`KERNELS/`](KERNELS/); N14-N16
are described in section 1.

**L4 v3.** This implementation is L4 v2 (`nemo_l4_nvfp4_v2`, system `firmus_gb300_ngpu8_llama8b_l4_nvfp4_v2`, an earlier
Firmus entry, withdrawn from this submission) with one change, in N11: with `AK_FP4G_NOOVERLAP=1` no parameter gather of
the forward overlaps compute ([N11](KERNELS/N11_fp4g_patch.md), "L4 v3: no gather overlap"). It removes the forward-side
run-to-run departure seen in L4 v2 ([Run-to-run reproducibility](#run-to-run-reproducibility)). L4 v3 has its own image
and its own ten scored runs.
- Changed: N11's module `ak_l8b_fp4g_patch.py` (the mode), `firmus_ak/run_start_gate.py` (one more gate row) and the
  config leaf (one more switch). Four more files differ only in a comment or docstring that names the implementation.
  The change record (`firmus/ak/pkg/CHANGES_vs_validated.md`, `firmus/ak/pkg/diffs/ak_l8b_fp4g_patch.diff`) and the
  documentation describe the change.
- Rebuilt: the image. Three extensions came out with new bytes from unchanged sources (K10, K11, K12); their kits passed
  on the new image.
- Each write-up states at its top how L4 v3 relates to L4 v2 for that item. Its Evidence lists the tests on this image
  under "Tests on the submitted image" and the tests that ran on L4 v2's image under "Tests on L4 v2's image".
- L4 v2's package, and the L3b packages (`nemo_l3b_nvfp4_v1_1`, `nemo_l3b_nvfp4_v1_2`, `nemo_l3b_nvfp4_v4`) that some
  records cite ("L3b", "the sibling package"), were withdrawn from this submission and are no longer in this repository.
  Evidence collected on L4 v2's image, and the records that name those packages' files, are kept as cited; reviewers can
  request them (TR-12.1e).

**How to read a write-up.** Each starts with a short summary ("In brief"), then gives, under the names of
`docs/KERNEL_SUPPLEMENTARY_INFORMATION.md` in brackets: what it replaces (Replaces) and where the code is (Source); why
the result is the same (Equivalence), with the equations on both sides; Numerics; Optimizer and Gradient clipping; the
inputs it accepts and the assumptions no code checks (Shape specialisation); how it is switched on and checked at run
time (Enablement); the rule it relies on (Approval); the evidence (Evidence); and the Firmus records behind it.

**The image.** Every result in this submission, and every test listed as on the submitted image, ran on the image built
from this directory's [Dockerfile](Dockerfile) (the submitted image):
- squashfs sha256 `1d55d6d50d540c7688046dc1cb0b09ae13ef5745ec0ae1c18ec08f236d484942`;
- `/FIRMUS_AK.env`: `FIRMUS_AK_BUILD_SHA256=d041cf2e1a22bf4551a464a56f1e3000ceaaa43ef7bcc5e79d611a8e84ec65ec`.

The tests listed as on L4 v2's image ran on that image (squashfs `23469bf7…`, build record `5b695487…`), built from the
same sources except the files named above. Evidence from dev image 0929 is development evidence (section 8). The test
kits and the logs of every test below are available to reviewers on request.

Paths are relative to this directory (`nemo_l4_nvfp4_v3/`). "config L" is a line of
`config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh`; "patch L" a line of a `.patch` / `.diff` file in `firmus/patches/`; "file L" a
line of the named file. Shas are sha256 prefixes of the shipped file; for a compiled extension, of the binary in the
image. The files in `KERNELS/` are generated from the central copies in `kernels/llama_nvfp4_lw/` of the submission
repository by `tools/build_kernel_docs.py`; `KERNELS/SOURCES.sha256` records the source and output sha256 of each.

**Contents**

1. [Summary: every change, its class and rule basis](#1-summary-every-change-its-class-and-rule-basis)
2. [Assumptions that no code checks](#2-assumptions-that-no-code-checks)
3. [Verifying which kernels are on](#3-verifying-which-kernels-are-on)
4. [Facts that apply to every kernel](#4-facts-that-apply-to-every-kernel)
5. [Tests on the built image](#5-tests-on-the-built-image)
6. [Optimizer, gradient clipping and numeric formats](#6-optimizer-gradient-clipping-and-numeric-formats)
7. [Training evidence and modules not shipped](#7-training-evidence-and-modules-not-shipped)
8. [Glossary](#8-glossary)
- [Records available on request](#records-available-on-request)

## 1. Summary: every change, its class and rule basis

**Classes.** How an item's output compares with the NVIDIA code it replaces, on the same inputs:

- **E0**: byte-identical on every input that was tested.
- **E0-restricted** (N4, and K6's gradient-norm pass): byte-identical except on a stated input domain. The write-up
  states the domain, what differs there, and why training stays outside it.
- **E0 in value** (N7 only): byte-identical except that an exact-zero gradient element can come out as −0.0 where
  NVIDIA's code gives +0.0 (NVIDIA adds the gradient onto a zeroed buffer, and +0.0 + (−0.0) = +0.0; N7 stores it). The
  two compare equal. The gradient norm and Adam's second moment are byte-identical either way; Adam's first moment and
  the update can differ only in the sign of an exact-zero result, never in a non-zero value. No negative zero occurred
  in any test.

No item is E1 (a different floating-point order compared within a tolerance).

**Rule basis.** The section of `training_rules.adoc` an item relies on, with the Firmus checklist id in brackets:

- **Graph Execution (TR-8.3):** "Frameworks are free to optimize the non-weight parts of the computation graph provided
  that the changes are mathematically equivalent."
- **Numerical formats (TR-5.2):** pre-approved formats (TR-5.2a), and numerically safe techniques that leave the
  real-valued function of the model unchanged (TR-5.2b). RHT and stochastic rounding are the recipe's own.
- **Hyperparameters, optimizer (TR-9.1c):** an alternate optimizer implementation must have its equation described and
  its equivalence with an approved optimizer shown (K6).
- Graph Definition (TR-8.1), Benchmark detection is not allowed (TR-2.3), SR-9 (N16) and SR-10 (training inspection) are
  also addressed.

**No item asks for an exception** under "Equivalence exceptions" (TR-9.5).

**First-call check.** The first eager call of each replaced operation also runs the stock operation on the same inputs,
in NVIDIA's untimed warm-up, before `run_start`. Every output buffer is compared byte for byte, including scales,
amaxes, flags and, for stochastic rounding, the generator offset after the call. Rank 0 prints a line ending `: True`
(the two X12 lines add the per-output results, `: True ([True, …])`, and the TCE line ends `loss True; gradient True`).
A mismatch raises, so the job ends before the clock starts ([Enablement and fail-loud
behaviour](#enablement-and-fail-loud-behaviour)).

The column "result on the submitted image" gives each write-up's result in short; each write-up also lists the tests
that ran on L4 v2's image.

| id | write-up | what NVIDIA's code does there, and what changes | switch (config L) | class | rule basis | first-call check | result on the submitted image | submitted runs |
|---|---|---|---|---|---|---|---|---|
| K1 | [x12f angle table](KERNELS/K01_x12f_rope_table_cache.md) | TE's fused QKV RoPE computes `sincosf` of the frequencies in every call; the fp32 angle table is computed once per frequency tensor and read by K2's and K4's passes (Triton). The file's own RoPE kernels never run | `AK_X12_ROPE` `AK_X12_TAB_CACHE` (L10) | E0 | Graph Execution | first cache hit against a fresh table, as bytes | PASS: kit; every run's first-call checks against TE | on (table and cache) |
| K2 | [QKVA](KERNELS/K02_x12a_qkv_rope_amax.md) | in eager steps, TE's fused QKV RoPE forward, plus the amax pass of TE's FP8 current-scaling quantize of q\|k\|v: one Triton kernel writes q, k, v and per-position partial maxima | `AK_QKV_AMAX` (L10) | E0 | Graph Execution | X12 forward against TE; K14's partials check | PASS: kit; X12's forward check in every run | on |
| K3 | [x12d](KERNELS/K03_x12d_qkv_direct_triton.md) | the amax pass and cast of TE's FP8 quantize of q\|k\|v as Triton passes: the fallback for a buffer K4 declines | `AK_QKV_DIRECT_P1` `_W1` `_W2` `_REV` (L11) `_ASM` (L12); imported by `AK_QKV_DIRECT` (L11) | E0 | Graph Execution | — | imported but never run in training; where kits ran its passes, the bytes equal TE's quantize | imported, never launched |
| K4 | [qkvx](KERNELS/K04_qkvx_qkv_direct_rope_bwd.md) | under capture, TE's fused RoPE forward + the FP8 cast of q\|k\|v as two CUDA passes; in every backward, TE's RoPE backward + the Hadamard-amax pass of the QKV gradient | `AK_QKV_DIRECT` `AK_QKV_DIRECT_IMPL` (L11), `AK_QKVX_*_VER` (L10, L12), `AK_QKV_GRAD_AMAX` (L10), `AK_QKV_DIRECT_P1_REV` `_P2_REV` (L12) | E0 | Graph Execution | QKV-direct and gradient-amax lines; X12 backward | PASS: kit; every first-call check in every run | on |
| K5 | [emb2](KERNELS/K05_emb2_embedding_wgrad.md) | the recipe's `embedding_ops.embedding_bwd` + Megatron's `main_grad +=`: one Triton kernel writes the embedding gradient into `main_grad` | `AK_EMB` (L14) | E0 | Graph Execution | two (the gradient on a slice; `main_grad` whole) | PASS: kits of K5 and its call site; both checks in every run | on |
| K6 | [X9 fused AdamW](KERNELS/K06_x9step3_fused_adamw.md) | TE FusedAdam's L2 norm, clip scale and AdamW step, the copies around them and the next weight-amax pass, fused into a norm pass and one step kernel | `AK_X9_E0` `AK_X9_FOLD` `AK_X9_WAMAX` (L20) | **E0-restricted** (the norm pass: identical except where every element of a partial sum has \|g\| < 2^-63; the AdamW step E0) | Hyperparameters, optimizer; Graph Execution; Numerical formats | epilogue amaxes; the norm pass is a gate part; the fused step has none | PASS (E0-restricted): kit; the epilogue check in every run; no compared case or step showed a difference in the norm | on |
| K7 | [nvfp4p SwiGLU](KERNELS/K07_nvfp4p_swiglu.md) | `tex.swiglu` / `tex.dswiglu` + TE's Hadamard-amax pass | `AK_NVFP4P_KVER` (L23) | E0 | Graph Execution | per (operation, usage), RNG replayed | PASS on every shape training sends: kit; three checks in every run | on |
| K8 | [nvfp4n add + norm](KERNELS/K08_nvfp4n_add_rmsnorm.md) | Megatron's bias-dropout-add (a plain add) + the statistics of `tex.rmsnorm_fwd`; the RMSNorm forward where K11 does not apply | `AK_NVFP4P_NORM` `_NORM_VER` (L23), `_ADDNORM` `_ADDSTATS_VER` (L24) | E0 | Graph Execution | the add; the norm's output and `rsigma` | PASS on every shape training sends: kit; five checks in every run | on |
| K9 | [nvfp4w weight amax](KERNELS/K09_nvfp4w_weight_amax.md) | the full-weight amax pass of TE's 2D weight quantize, from the optimizer shards | `AK_NVFP4P_WEIGHTS` (L23) `AK_NVFP4P_WAMAX` (L24) | E0 | Graph Execution | at the second optimizer step: 21 weights here, 109 in K15's check | PASS: kits; first-call and step-2 checks in every run | on |
| K10 | [w2d7 weight cast](KERNELS/K10_w2d7_weight_cast.md) | TE's `quantize_transpose_nvfp4_2D_kernel`, register-tile schedule (the 19 GEMM weights K15 does not gather; the cached workspaces) | `AK_NVFP4P_WQ` (L25) `AK_NVFP4P_W2D` (L26) | E0 | Graph Execution | weight casts; workspaces at the next step | PASS: kit on this image's build; first-call and workspace checks in every run | on |
| K11 | [rht7 norm apply](KERNELS/K11_rht7_norm_apply.md) | the apply half of `tex.rmsnorm_fwd` with an NVFP4 output, persistent schedule | `AK_NVFP4P_NORMCAST` `_NORMCAST_MODE` `_RHTCAST_VP` (L25), `_NORMCAST_APPLY_VER` (L26) | E0 | Graph Execution | per norm site | PASS on every shape training sends: kit on this image's build; first-call checks in every run | on |
| K12 | [rht8 RHT cast](KERNELS/K12_rht8_rht_cast.md) | TE's fused row + column-RHT NVFP4 cast (`row_col_rht_gemm_device`), rescheduled | `AK_NVFP4P_RHTCAST` (L25) `AK_NVFP4P_RHT8` (L26) | E0 | Graph Execution; Numerical formats | per shape, RNG replayed | PASS: kit on this image's build; each cast's first call against TE in every run | on |
| K13 | [nvfp4bx norm backward](KERNELS/K13_nvfp4bx_norm_bwd.md) | `tex.rmsnorm_bwd` + the residual-gradient add + the next quantize's amax pass + DDP's `main_grad.add_(dgamma)`; the Hadamard-amax pass of TE's other RHT quantizes | `AK_NVFP4P_BWDNORM` (L23), `_BWDNORM_VER` `_RHTAMAX` (L24), `_DGFOLD` `_BX` (L26) | E0 | Graph Execution | per site | PASS: kit; first-call checks in every run | on |
| K14 | [fp8q FP8 quantize](KERNELS/K14_fp8q_fp8_quantize.md) | TE's `Float8CurrentScalingQuantizer` for dO; for q\|k\|v, its scale, and its cast in eager forwards | `AK_FP8CS_AMAX` `_CAST` `_AMAX_CFG` (L32) | E0 | Graph Execution | per (path, shape, format), plus one QKV-direct check | PASS: kits of K14 and K2; first-call checks in every run | on |
| K15 | [fp4g parameter gather](KERNELS/K15_fp4g_param_allgather.md) | the bf16 parameter all-gather + TE's 2D weight cast, for 109 of 128 weights: each rank casts its shard and the NVFP4 images are gathered | `AK_FP4G` `AK_FP4G_CBPC` (L34) | E0 | Graph Execution | every gathered weight at the second optimizer step | PASS: kit and the patch-module kit's L4 v3 version; every run's check, 0 bytes differ | on |
| K16 | [lmh LM head](KERNELS/K16_lmh_lm_head.md) | the output layer's forward and dgrad matmuls, through cuBLASLt on pinned algorithms | `AK_LMH` (L40) | E0 | Graph Execution | forward and dgrad | PASS: kit; both first-call checks in every run | on |
| N1 | [TE rope.py + utils.py](KERNELS/N01_qkv_pack.md) | X12 dispatch in TE's RoPE; the `torch.cat` of q\|k\|v replaced by a flat view of one buffer | `AK_QKV_PACK` (L10), with K1-K4's | E0 | Graph Execution | the pack against the cat path | PASS: kit; the pack's and both X12 checks in every run | on |
| N2 | [recipe custom_embedding.py](KERNELS/N02_emb2_call_site.md) | K5's call site; the gradient written into `main_grad` | `AK_EMB_MAIN_GRAD` (L14), with `AK_EMB` | E0 | Graph Execution | as K5 | PASS: kit; first-call checks in every run | on |
| N3 | [TCE + TCEV](KERNELS/N03_tce_cross_entropy.md) | the logits copy of TE's cross-entropy, in training and evaluation | `AK_TCE_OVERWRITE` `AK_TCEV` (L16) | E0 | Graph Execution | training (first 256 positions); evaluation | PASS: both checks in every run; identity loops equal stock's reference | on |
| N4 | [GSA](KERNELS/N04_gsa_avg_reduce_scatter.md) | the 1/DP pre-multiply + SUM reduce-scatter becomes NCCL's `ReduceOp.AVG` | `AK_GSA` (L18) | **E0-restricted** (identical except where some rank's nonzero gradient element has \|g\| < DP·2^-126) | Graph Execution; Numerical formats | the first bucket, both ways, every rank | PASS (E0-restricted): the first bucket both ways, 0 differing elements in every run | on |
| N5 | [X9 wiring](KERNELS/N05_x9_wiring.md) | Megatron-Core's optimizer step, local norm and parameter all-gather, wired to K6; the next forward's first bucket gathered inside the step | `AK_X9_EARLY_AG` (L20), with K6's | as K6 | as K6 | the early all-gather against a synchronous re-gather | PASS (as K6): 0 of 1,221,623,808 elements differ in every run | on |
| N6 | [nvfp4p patch](KERNELS/N06_nvfp4p_patch.md) | wraps TE's quantize, activation, norm and cast calls and Megatron's bias-dropout-add (the wiring of K7-K13) | `AK_NVFP4P` (L23), with K7-K13's | E0 | Graph Execution | per site | PASS on every shape training and evaluation send: 37 checks in every run; the 24 kits | on |
| N7 | [X7 dense](KERNELS/N07_x7_dense_store.md) | the dense weights' `main_grad` zero-fill + accumulate becomes a store | `AK_X7_SKIP_GRAD_ZERO` (L28) | **E0 in value** | Graph Execution | sentinel count at the first gradient sync (49 of the 129 stored weights) | PASS (E0 in value): kit; the sentinel count in every run | on |
| N8 | [PROJCOL](KERNELS/N08_projcol.md) | the backward re-quantize of `linear_proj`'s input | `AK_PROJCOL` (L30) | E0 | Graph Execution | the combined quantize against two separate ones | PASS: kit; first-call check in every run | on |
| N9 | [OPSWZ](KERNELS/N09_opswz.md) | the per-GEMM scale swizzle of the fc1 / fc2 weights | `AK_OPSWZ` (L30) | E0 | Graph Execution | per weight shape | PASS: kit; two first-call checks in every run | on |
| N10 | [fp8cs patch](KERNELS/N10_fp8cs_patch.md) | TE FP8 attention's `tex.quantize` of q\|k\|v and dO, routed to K14 and K2 / K4 (the wiring of K2-K4, K14) | `AK_FP8CS` (L32), with K14's | E0 | Graph Execution | as K14 | PASS: kits; four first-call checks in every run | on |
| N11 | [fp4g patch](KERNELS/N11_fp4g_patch.md) | `_ParamAndGradBucketGroup.start_param_sync` (the wiring of K15); in L4 v3, every forward parameter gather is also finished at its dispatch (no overlap with compute) | `AK_FP4G_CHECK` `AK_FP4G_POISON` (L34), `AK_FP4G_NOOVERLAP` (L36), with K15's | E0 | Graph Execution | as K15; the no-overlap mode is a gate part | PASS: the patch-module kit's L4 v3 version, 9 of 9 checks per run; every run's check, 0 bytes differ | on |
| N12 | [smallk2](KERNELS/N12_smallk2.md) | the embedding gathered in [s, b] order; TE's per-region delayed-scaling updates batched | `AK_SMALLK2` (L38) | E0 | Graph Execution | two | PASS (E0) in the submitted configuration, with N5's early all-gather on: kit; identity loops; both checks in every run | on |
| N13 | [lmh patch](KERNELS/N13_lmh_patch.md) | the output layer's forward and dgrad lines (the wiring of K16) | uses K16's `AK_LMH` (L40) | E0 | Graph Execution | as K16 | PASS: both GEMMs equal torch's matmul at their first call in every run; K16's kit | on |

**Upstream patches.** "TE" is Transformer Engine 2.20.0 at `5b28d7aa`. "Megatron-Core" is Megatron-LM `f6c33bde`, the
submodule of Megatron-Bridge `ea5dded8`.
- The Dockerfile applies each patch with `git apply` before that library is installed (steps F1 / F2).
- The two recipe files are edited in place; their diffs against NVIDIA's files are `firmus/patches/recipe/`.
- Each patch's stock file equals the public upstream file at the pinned commit.

### N14. The L4 config leaf

`config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh` (`ff81b648`) sources NVIDIA's `config_GB300_2x4x2xtp1pp1cp1_8b.sh` (`5bdb2dfe`)
unchanged, then exports 61 `AK_*` switches (L10-40) and nothing else. They are L4 v2's 60 and `AK_FP4G_NOOVERLAP=1`
(L36).
- NVIDIA's leaf is the 20261006 release's `config_VR200_2x4x2xtp1pp1cp1_8b.sh` under a GB300 name, with one comment line
  added at the top (`config_GB300_2x4x2xtp1pp1cp1_8b.sh` L1).
- There is no change to `MAX_STEPS`, the bucket size, the batch size or any hyperparameter: global batch 16,
  micro-batch 2, LR 4e-4, 16 warm-up steps, evaluation every 768 steps, `BUCKET_SIZE` 768,000,000.
- The 61 switches are those of the summary table. Each belongs to exactly one write-up (its front-matter line); the build
  and `tools/check_package.py` (KD-2) check that for every enabled switch. Three are exported with the value 0, which
  leaves their path off: `AK_QKV_DIRECT_ASM`, `AK_QKV_DIRECT_P1_REV` and `AK_FP4G_POISON`.
- Seven are launch and direction knobs of K3 and K4: `AK_QKV_DIRECT_P1=2 AK_QKV_DIRECT_W1=2 AK_QKV_DIRECT_W2=2
  AK_QKV_DIRECT_REV=1 AK_QKV_DIRECT_ASM=0` (K3, not executed) and `AK_QKV_DIRECT_P1_REV=0 AK_QKV_DIRECT_P2_REV=1` (K4).
- The six variables the leaf leaves to a code default that turns a path on are listed in section 3.

### N15. Explicit enable: `pretrain.py` and the `firmus_ak` package

- **Source.** `pretrain.py` (`4e46d219`) is NVIDIA's with one change (`firmus/patches/recipe/0002-*.diff`): a comment
  and `import firmus_ak` as the first import. The package `firmus/ak/pkg/firmus_ak/`: `__init__.py` `f06ea0c3`,
  `banner.py` `652b404c`, `run_start_gate.py` `0c1c576b`, `check_install.py` `145ba772`. Against L4 v2's: `__init__.py`
  differs only in its first docstring line (the implementation's name); `run_start_gate.py` adds the row
  `fp4g no-overlap` (L58-62).
- **Role.** No numerical role. `banner` checks the image's bytes first; `__init__` then imports the eight patch modules
  in a fixed order; `run_start_gate` makes the end of the warm-up raise if an armed part never checked its first call
  (section 4).
- **Enablement.** Always imported. Each module arms on its own switch. The gate is on by the code default
  `FIRMUS_AK_RUN_START_GATE=1` (`run_start_gate.py` L23); `FIRMUS_AK_RUN_START_GATE=0` would turn it off, and the config
  does not set it.

### N16. `requirements.txt`: git dependencies pinned by commit

`firmus/patches/recipe/0003-*.diff` replaces the two tag references in NVIDIA's `requirements.txt` by the commits those
tags name: `mlcommons/logging` `8a0cc055…` (tag `6.1.0-rc2`); `NVIDIA/mlperf-common` `b5014f3c…` (tag
`training-v6.1-rc0`). This is the same code, identified uniquely (SR-9). No numerical role.

**Which tests cover the submitted items.**
- *Unit tests of the whole bundle on L4 v2's image* (section 5): every called path of every kernel and patch is
  compared with the stock path by at least one test.
- *Tests on this image of what L4 v3 changed* (section 5): the kits of the three rebuilt extensions (with all the
  others), the fp4g patch-module kit's L4 v3 version (N11's no-overlap mode), and the identity and drift loops.
- *A first-call check in every scored run:* every armed replacement except K6's fused step and norm pass (the norm
  pass's gate row records only that it ran); N7 at its first gradient sync (49 of the 129 stored weights).
- *An in-training shadow at DP 8, on L4 v2's image:* K6 and N5 in graph mode, the shipped mode; the NVFP4, FP8 and
  LM-head sites in eager mode only (section 5).
- K3 is never launched.

## 2. Assumptions that no code checks

Every item tests its inputs with a gate, and a call that fails the gate runs NVIDIA's code. The conditions below are
different: the items rely on them, but neither the gate nor any assert checks them. Each holds in the submitted
configuration, for the reason given. The main ones are below; each write-up has its full list under "Assumptions not
checked in code".

**Conditions whose failure would give a wrong result without an error.** These matter most.

| item | assumption | why it holds in the submitted configuration | if it failed |
|---|---|---|---|
| N12 (with N5) | The embedding weight is gathered before N12 reads it. N12 reads `word_embeddings.weight` without waiting for its parameter all-gather; N5's early all-gather gathers that bucket first. | `AK_X9_EARLY_AG=1` (config L20); its proof line "early all-gather ARMED: bucket group 8 of 9 … True" is in every log. | The other ranks' embedding rows would be one step stale. Measured: with the early all-gather off, training departs at step 3. |
| N7, N2, K5 | One microbatch per optimizer step, and no output layer that shares the embedding weight: the stored gradients are written once per step, not accumulated. | Global batch 16 = DP 8 × micro-batch 2; every log shows gradient accumulation 1 and untied embeddings. | A stored weight would keep only the last microbatch's gradient. |
| N4 | No rank's nonzero gradient element has \|g\| < DP·2^-126 (2^-123 ≈ 9.4e-38 at DP 8), and the data-parallel world is a power of two. | A census of a whole run at the submitted layout found 0 in-band elements of 6.93e14 (one run, one seed); DP is 8. | AVG and the stock pre-multiply + SUM can round such elements differently. |
| K6 (with N5) | No partial sum of the gradient-norm pass is a nonzero subnormal (every contributing \|g\| < 2^-63). | No compared case or step showed a difference: the in-training shadows and the closed loops found the norm byte-identical at every checked step. | The local norm, and so the clip coefficient, could differ from TE's in the last bits. |
| K6 (with N5) | The optimizer is AdamW (decoupled weight decay) with `capturable` off; TE's `step()` has advanced the step count; the gradient and parameter buffers are never reallocated. | Megatron-Core's default `decoupled_weight_decay=True` and `optimizer_cuda_graph=False` (both in every log's optimizer config); stock call order; Megatron allocates the buffers once. | AdamW would be computed where TE computes L2-regularised Adam; stale bias correction or addresses. |
| K15, N11 | Nothing reads the other ranks' stale bf16 copies of the 109 gathered weights; every consumer runs after its group's wait. | Every GEMM reads the gathered NVFP4 storage; poison tests found no reader. | A stale weight would be read. |
| K10, N6 | A live workspace's scale tensors have a fresh one's layout, and a workspace's amax (a view of K9's slot) is read before the next optimizer step rewrites it. | The workspaces are TE's, rewritten in place every step; the workspace check at the next optimizer step compares every buffer with TE's cast. | Wrong scale offsets or amax for the cast. |
| K13 | Megatron's DDP hook skips its add when `grad_added_to_main_grad` is set. | TE's own mechanism; Megatron's hook and N7's patched hook honour it. | The norm weights' gradient would be added twice. |
| K13 | The norm backward's first-call check is keyed on the calling module only, not the row count or the fold flag. | Training folds on every call, at one shape. | A later call of another kind would run unchecked. |
| K7, K8, K11, N6 | Later calls have their key's first row count, and it is one at which the route equals TE's: the routes equal TE's at multiples of 1,024 (K8's norm at every multiple of 64 tested). The gates of K7 and of the norm wrapper K8 and K11 share admit any multiple of 32; K11's own gate needs a multiple of 1,024 and leaves other rows to K8's route. | Every NVFP4 site's row count in training and evaluation is 16,384 (the row-count census, section 5). | At the odd multiples of 32 tested, the scales come back in another layout than TE's, or N6's own check raises (section 4, "Shape specialisation"). |
| N6 (K7, K8, K11, K13) | Every first-call check's key is first called outside a CUDA-graph capture. Inside a capture a key that was not checked eagerly stays unchecked: K7's, K8's and K13's routes then run their own kernels, a norm site keeps K8's route instead of K11's, an RHT-cast key gets TE's cast and a weight-amax key TE's cast with K9's amax. | NVIDIA's warm-up runs an eager training step and eager validation warm-ups before it captures each graph; in all ten logs the first-call comparison lines print outside both captures. | That key's route would run without ever being compared. |
| K2, K14 (N10) | The q\|k\|v registration identifies the buffer by data pointer and element count only, and its partial maxima describe its current contents. | K2 / K4 write the buffer and register it; the quantize follows at once. | A wrong FP8 scale. |
| K3, K4, N10 | Under capture the unwritten bf16 q\|k\|v buffer has the quantize as its only reader; the QKV-direct check runs once per process, on whichever passes the first buffer takes. | TE's FP8 attention keeps only the FP8 q, k, v; the recipe has one shape. | Garbage read, or K3's passes run unchecked. |
| K1, N1 | The RoPE frequencies are [s', 1, 1, D] with s' ≥ s, and are not written in place between graph replays. | The frequencies are a constant of the model. | Wrong rows or an out-of-bounds read; a stale angle table. |
| K5 | Ids in [0, V) and 2,048 ≤ h ≤ 4,096. | Dataset token ids, vocabulary 128,256, hidden 4,096. | Out-of-bounds access or a wrong gradient. |
| K8, K11 | Nothing writes the residual sum between K8's add and its norm without bumping its version counter; nothing reads the attention or MLP output after the evaluation add (stock writes into it in place; K8 does not). | No such writer or reader in the recipe. | A stale `rsigma`; a reader would see the unchanged output. |
| K9 | Only the optimizer gives the bf16 weights new values between forwards, and the bucket buffers keep their addresses. | Megatron's optimizer is the only writer; buffers allocated once. | Stale weight amaxes. |
| K12 | Its output buffers have TE's layout for the input. | Its only caller passes buffers TE has just created. | Writes out of place. |
| K16 | Calls after the first of each problem compute as the first. | A fixed algorithm and descriptor; shown by tests. | A wrong LM-head output. |
| N3 | Nothing writes or reads the logits between the loss forward and backward, or after the backward; later evaluation calls send inputs TE accepts. | No other reader in the recipe; one evaluation shape. | A read through autograd raises; a direct read would see the gradient. |
| K1 | The frequencies are written only by operations that bump PyTorch's version counter. | They are a model constant, not written after setup. | A stale angle table. |
| K13 | Nothing writes the residual sum between the norm backward and the quantize that consumes its amaxes without bumping its version counter. | No such writer in the recipe. | Amaxes of stale data. |
| K15, N11 | The gathered buffers' addresses and storage stay fixed after setup. | Megatron allocates the buffers once. | Stale addresses (a replaced workspace raises). |
| N1 | Under capture, the q\|k\|v pack a forward registers is consumed by its quantize before the next forward. | The quantize follows each forward. | Raises at the next packed forward; silent only after the last forward of the last capture. |
| K2, K3, K5 and other Triton kernels | 32-bit offsets: every tensor they index has fewer than 2^31 elements. | The largest is K5's 525,336,576-element embedding gradient. | Offsets would wrap. |
| N7 | TE's weight-gradient GEMM writes the same bf16 values when it stores as when it adds onto zeros, and stores for every tagged weight; every element of a stored weight is written each step; padding between parameters is never read. | The kit tested the four weight shapes; the model-level comparison with N7 off matched; the sentinel count at the first gradient sync (49 of 129 weights) found no unwritten element. | A stale or doubled gradient element. |
| N5 | Megatron's `align_param_gather` and `overlap_param_gather_with_optimizer_step` are off, and nothing reads or writes the early-gathered bucket group between its dispatch and its wait. | Both are False in every log's DDP config; no such reader in the recipe. | Not fully analysed; an early-gathered weight could be read before its gather completes. |
| N4 | NCCL adds AVG's contributions in the same order as SUM's (taken from measurement, not from NCCL's code). | The image pins NCCL 2.31.2; the tests found 0 differing bytes outside the domain, and so did every run's first-bucket check. | AVG and SUM would differ by rounding. |
| N8 | Nothing writes `linear_proj`'s input between the forward and the backward without autograd seeing it. | No such writer in the recipe. | The backward would use a column-wise copy of stale data. |
| N9 | One first-call check per weight shape stands for every quantize of that shape; TE's GEMM-side swizzle and the pre-swizzled quantize write the same layout. | Same code and shapes; TE's own rule. | Wrong scale layout at a GEMM. |
| N10 | The quantizer's settings, read by attribute name with defaults, are the same on later calls as at the first call of each key. | One recipe quantizer per site. | A later call with other settings would run unchecked. |
| N11 | `_post_param_sync()`, which N11's wrapper skips on a forced or synchronous sync, has nothing to do for a pure group. | Pure groups hold bf16 parameters only; `reuse_grad_buf_for_mxfp8_param_ag=False` in every log. | A post-gather step would be missed. |
| N12 | Only a layer's own attention reads its S and dP scales, and only that attention and TE's update write their histories. | TE's delayed-scaling design; the warm-up learns and checks the schedule. | A scale read between updates would be stale. |
| N13 | The branch of Megatron's `_linear_forward` that the wrapper replaces computes `torch.matmul(x, W.t())`. | The image's Megatron (`f6c33bde`) does; the eager shadow compared the wrapper with Megatron's own function on training calls (0 of 82 differed). | The LM head would keep computing the plain product. |

**A condition of L4 v3's no-overlap mode.** Its failure would change no value. The gathers N11 waits for at their
dispatch (the 7 pure groups' and bucket group 0's) are all the gathers the forward runs beside compute; bucket group 8,
the embedding's, is gathered and waited for inside the optimizer step by N5's early all-gather. Setup raises unless
exactly one stock group has no next gather, and the gate's `fp4g no-overlap` row requires both kinds of wait in the
warm-up; no code checks that no other stock group's gather overlaps a GEMM. If one did, that gather would overlap compute
as in L4 v2 ([N11](KERNELS/N11_fp4g_patch.md), "Assumptions not checked in code").

**Conditions whose failure would raise an error.** Listed for completeness.

| item | assumption | if it failed |
|---|---|---|
| K1, N1 | Head counts and head dim are powers of two; the packed layout's sizes match. | Assertion or compile error. |
| K4 | Version 3's grid (rows / 16) fits where the gate allows it. | The launch raises. |
| K7 | TE's activation kernels are built with fast math; the SwiGLU row count fits the grid (≤ 2,097,120). | The first-call check raises; the launch check raises. |
| K10 | `_wq_swizzled` matches TE's swizzle rule. | The workspace check raises (or the stock cast runs). |
| K13 | TE's grid has 3 CTAs per SM for the `dgamma` partials; the post-RHT amax grid fits. | The first call raises; the launch raises. |
| K15 | K9's all-reduce group equals the gather group. | The check at the second optimizer step raises. |
| K16 | The pinned algorithm is in the process's heuristic list; the first call is eager; Megatron's backward source has the recorded sha. | The install or first call raises. |
| N12 | Each phase has the length learned in the warm-up. | Raises. |
| N8 | The quantizer is TE's `NVFP4Quantizer`. | The run-start gate raises. |
| N9 | TE honours `optimize_for_gemm` on this GPU. | The first-call check raises. |
| all | The checks raise on a mismatch, and no caller catches the exception. Some first-call checks mark their key checked before comparing. | A caught exception would leave that route unchecked, and the gate would count the part as checked. |

## 3. Verifying which kernels are on

The on/off state of every item can be checked from the package alone.

1. **The config that ran.** Every `result_*.txt` logs `config_filename = config_GB300_2x4x2xtp1pp1cp1_8b.sh`, NVIDIA's
   base leaf. The config that ran is `config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh`, which sources it unchanged and adds the
   `AK_*` switches (L10-40). NVIDIA's `pretrain.py` logs `config_${DGXSYSTEM}.sh`, and `DGXSYSTEM` is set by NVIDIA's leaf
   ([README.md](README.md), "`config_filename`").
2. **The code's defaults.** Every enable switch is read with default `"0"`, so a switch the config does not export leaves
   NVIDIA's code in place. Six variables the config does not set have code defaults that select a path that runs:
   - `AK_LMH_DGRAD=1` (`ak_l8b_lmh_patch.py` L39): K16's dgrad;
   - `AK_SMALLK2_EMB=1` and `AK_SMALLK2_DS=1` (`ak_l8b_smallk2_patch.py` L48-49): N12's two parts;
   - `FIRMUS_AK_RUN_START_GATE=1` (`firmus_ak/run_start_gate.py` L23): the run-start gate;
   - `AK_NVFP4P_RHTAMAX_VER=2` (`ak_l8b_nvfp4p_patch.py` L219, used at L1360): the version of K13's `rht_amax`;
   - `AK_NVFP4P_BWDNORM_CPS=3` (L239, used at L1160): K13's CTAs per SM in the norm backward.

   Every other `AK_*` or `FIRMUS_AK_*` variable the code reads and the config does not set is a directory, a build-time
   setting, or a switch of a path that does not run ([Code present but not
   executed](#code-present-but-not-executed-with-the-submitted-switches)). The Dockerfile `ENV` sets kernel directories
   only, never a switch.
3. **The logs.** Every armed replacement prints an ARMED line on rank 0, most with its module's sha and its first-call
   verdict. In each of the ten logs:
   - **63 ARMED lines.** 40 name their module with a sha. The other **23 carry no sha**: they are printed by the patch
     modules and the patched upstream files, and five of them are K8's `ak_l8b_nvfp4n.so` lines, which name the module
     with no path or sha. K8's bytes are still bound: the banner's `tree_sha256` is the build record that `banner.py`
     re-hashes at import, and `lib/ak_l8b_nvfp4n.so` is in it. L4 v3 adds one line to L4 v2's 62: N11's no-overlap
     ARMED line, which has no sha and no verdict (its check is the gate part `fp4g no-overlap`).
   - **64 verdict lines, all True**, before the gate line: 61 ending `: True`, the two X12 lines ending `: True ([…])` and
     the TCE line ending `loss True; gradient True`. 53 are first-call checks (45 in the forward and backward of
     warm-up iteration 0, 8 in the validation warm-up); 5 are checks at the first gradient sync (GSA's, and four N7
     sentinel counts); 5 are checks in the warm-up's optimizer steps (N5's early all-gather, K15's gathered weights,
     K9's weight amaxes, K10's workspaces and K6's epilogue amaxes); 1 is N12's check of the delayed-scaling schedule
     learned in iteration 0.
   - Six ARMED lines end without a verdict. Two have their verdict in a separate check line: K1's X12 line and K15's
     gather line. Four have no verdict line: K6's norm pass and fused step (both name `fused_adamw_master.py`
     `3035ea35`), N6's residual taps (a count) and N11's no-overlap line. K6's norm pass and N11's no-overlap mode are
     parts of the run-start gate; K6's fused step has no in-run first-call check (K6).
   - **The run-start gate line** (L946 in every log, before `run_start` at L950): `19 armed Firmus parts (gsa, projcol,
     opswz, fp8cs, fp4g, fp4g no-overlap, lmh forward, lmh dgrad, nvfp4p, nvfp4p rht cast, nvfp4p weight amax, nvfp4p
     norm backward, smallk2 embedding, smallk2 delayed scaling, x12 rope, qkv pack, emb, emb main_grad, x9 norm fold);
     every armed part checked its first call: True`.
   - `[auto_kernel banner] MATCH`, with `tree_sha256=d041cf2e…` and `stack_sha256=024dd9bc…` (`result_1.txt` L67-70).
   - **56 proof lines.** Each run's post-run check matched all 56 proof-line patterns of the submitted image (section
     8, "Proof line").

   Run from the directory that holds `Firmus/`:

```bash
R=Firmus/results/firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/llama31_8b
grep -c 'ARMED' $R/result_*.txt                                     # 63 in every file
grep 'ARMED' $R/result_*.txt | grep -vc 'sha256='                    # 230: 23 per file
grep -c 'auto_kernel banner\] MATCH' $R/result_*.txt                 # 1 in every file
grep -c '19 armed Firmus parts.*first call: True$' $R/result_*.txt   # 1 in every file (the run-start gate)
grep '^0: \[auto_kernel' $R/result_1.txt | grep -v 'auto_kernel gate' | grep -c ': True$'   # 61
```

Only rank 0's lines reach the log. That the replacements arm on the other ranks follows from identical switches and
shapes on every rank and from installs and checks that raise on failure; the log does not show it. A reading of this
package's ten logs: 40 ARMED lines per log carry a sha, 14 distinct, and each equals the sha of that file in the image's
build or source record; no verdict is `False`.

## 4. Facts that apply to every kernel

### Provenance

The kernels were developed and verified on NVIDIA's 20260929 release of this recipe (dev image 0929). That release has
the same Megatron-Core, Megatron-Bridge, cuBLASLt 13.8.0.4, cuDNN 9.26.0 and NCCL 2.31.2 as the 20261006 release this
implementation rebuilds.

The two releases differ only in Transformer Engine (7 commits) and NVIDIA's `run.sub`. Those commits touch the
experimental CuTeDSL MXFP8 backend, which NVIDIA's Dockerfile does not build; the MoE grouped GEMM / grouped MLP; and the
Triton permutation kernel. None of these is on this recipe's path, and none is code these kernels reproduce. The stock
originals of every patched source file (TE, Megatron-Core, `custom_embedding.py`, `pretrain.py`) are byte-identical in
the two releases.

All equivalence tests were then run again on L4 v2's image, on that image's own bytes (section 5). On this image the
kits, the fp4g patch-module kit's L4 v3 version, the import checks, and the identity and drift loops ran again
(section 5).

### Numeric formats: no hyperparameter, format or algorithm change

Every MLPerf hyperparameter, the precision recipe, the evaluation and the stopping rule are NVIDIA's (N14). Formats:
- **Linear layers:** NVFP4 (E2M1 values, an E4M3 scale per 16 elements, an fp32 per-tensor scale from the amax).
  Activations: 1D 16-element blocks, with the 16×16 random Hadamard transform (RHT) on the column-wise copy. Weights: 2D
  16×16 blocks. Gradients: RHT, plus stochastic rounding with TE's Philox4x32 (3 rounds).
- **Attention:** FP8 current scaling (q|k|v E4M3, dO E5M2), and delayed scaling with history 1 for S and dP.
- **LM head:** BF16. **Optimizer:** fp32 master weights and moments, bf16 parameters.

The kernels reproduce these formats bit for bit, including TE's own approximations: fast-math `div.full` /
`sqrt.approx` / FTZ in the optimizer, and the fast-math activation build. No `NVTE_*` switch is added. NVIDIA's
`config_common_fp4.sh` sets `NVTE_NVFP4_DISABLE_RHT=0`, `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0` and
`NVTE_NVFP4_DISABLE_2D_QUANTIZATION=0` (L10-12), so all three stay on.

### Build: compiler flags and architecture

`firmus/ak/build_all.py` compiles every extension from `firmus/ak/csrc/` with `torch.utils.cpp_extension.load`. The
flags and target architecture are recorded per extension in `firmus/ak/build_manifest.json`:

| extension | arch (`TORCH_CUDA_ARCH_LIST`) | flags |
|---|---|---|
| K4 `ak_l8b_qkvx` | 10.3 (`sm_103`) | `-O3 -lineinfo` |
| K7 `ak_l8b_nvfp4p` | 10.3 | `-O3 --use_fast_math -lineinfo`. TE builds its activation kernels with fast math (`NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1`); the same flag is what makes the bytes equal |
| K8 `ak_l8b_nvfp4n` | 10.3 | `-O3 -lineinfo` (no fast math, like TE's norms) |
| K9 `ak_l8b_nvfp4w` | 10.3a (`sm_103a`) | `-O3 -lineinfo` |
| K10 `ak_l8b_w2d7` | 10.3a | TE's own flags (`-O3 --expt-relaxed-constexpr -DNDEBUG -DNVTE_BUILD_NUM_PHILOX_ROUNDS=3`, the half-precision operator flags), `-lineinfo`, `-DAK_W2D_HAVE_V21 -DAK_W2D_HAVE_V23`; TE headers from the image's TE source tree; the only extension that links the image's `libtransformer_engine` |
| K11 `ak_l8b_rht7` | 10.3a | TE's own flags (as K10) and `-lineinfo`; TE, TE Hadamard-transform and CUTLASS headers from the image's TE source tree; does not link `libtransformer_engine` |
| K12 `ak_l8b_rht8` | 10.3a | as K11 |
| K13 `ak_l8b_nvfp4bx` | 10.3a | `-O3 -lineinfo` |
| K14 `ak_l8b_fp8q` | 10.3 | `-O3 -lineinfo` |
| K15 `ak_l8b_fp4g` | 10.3a | `-O3 -lineinfo` |
| K16 `ak_l8b_lmh` | host code only | `-O2`, `-lcublasLt` |

**Where the flags come from.** They are the build calls of the development builds that were verified; for K14, its
development test script. **Architecture.** The `sm_103a` builds are the kernels that use arch-specific instructions
(for example `cvt.e2m1x2`) or TE's `sm_103a` code. All of them target GB300 only.

**JIT kernels.** The Triton kernels (K1, K2, K3, K5, K6) are compiled at run time by the image's Triton; K3's kernels
and K1's forward and backward kernels are never launched in the submitted runs. K6 uses inline PTX for its fp32 chain
(K6). A rebuild compiles the extensions again; compiled binaries are not always byte-reproducible (this image's build
gave new bytes for K10, K11 and K12 from unchanged sources), so the image records what it built (next subsection).

### Enablement and fail-loud behaviour

**Explicit enable (N15).** The recipe's `pretrain.py` imports `firmus_ak` as its first import, before torch, TE and
Megatron.
- `firmus_ak` imports the eight patch modules in a fixed order.
- A module whose `AK_*` switch is 1 installs import hooks that patch the named TE / Megatron function when that module
  loads. A module whose switch is unset changes nothing.
- Nothing depends on the program's name or on the benchmark. Other processes do not import `firmus_ak`: the mount check,
  the NCCL test and the compliance checker.

**In-image paths.** Every directory a module loads from defaults, in the code, to the in-image path the Dockerfile builds
(`/opt/firmus/ak/...`). The image `ENV` sets the same paths. The build runs `firmus_ak.check_install`, which fails the
build if any directory variable is unset or outside `/opt/firmus/ak`, if any Firmus file names a host path, or if a
`.pth` loader is installed.

**Build records and the integrity banner.** The Dockerfile records the sha256 of every file it installs under
`/opt/firmus/ak` (22 files: the extensions, the Triton kernels and `lmh_pin.json`; `/FIRMUS_AK_BUILD_SHA256.txt`) and
of every patched upstream or recipe file as installed (10 files; `/FIRMUS_STACK_SHA256.txt`). At `import firmus_ak`,
every rank re-hashes those 32 files and checks that no other `.so`, `.py` or `.json` was added under `/opt/firmus/ak`.
Any difference raises before the model is built, and rank 0 prints `[auto_kernel banner] MATCH`.

Not re-hashed at run time: the eight installed patch modules (`ak_l8b_*.py`) and the `firmus_ak` package. They are pinned
by:
- the image's sha256;
- the image's source record (`/FIRMUS_AK_SOURCE_SHA256.txt`, 92 files);
- the image check the kit runner ran before the kits: each of the 8 installed modules equals its packaged copy (all 8
  matched, on this image as on L4 v2's). None of the 24 re-targeted kits imports the installed GSA, NVFP4P or lmh
  modules; the two patch-module kits (section 5) test the installed NVFP4P module (on L4 v2's image; the module is the
  same file in both) and fp4g module (its L4 v3 version, on this image).

Separately, the smallk2 module's sha is a proof line in every run.

**First-call checks.** Every armed replacement except K6's fused step and norm pass checks its first eager call
(section 1) or, for N7, its first gradient sync, and raises on a mismatch. K6's step and norm pass rest on the tests on
the image (K6). Any switch whose module cannot honour it raises at import.

**The run-start gate.** At the end of NVIDIA's warm-up, `firmus_ak/run_start_gate.py` lists the armed parts and raises
if any armed part has not completed its check.
- A part is a module or patch, not a call site, and its row tests one condition (`run_start_gate.py` L37-109). For
  example, the `nvfp4p` row passes once any one of N6's sites has run its first-call check (L73), and the `x9 norm fold`
  row once K6's norm pass has run (L105-108). The per-site record is each site's own first-call line.
- L4 v3 adds the row `fp4g no-overlap` (L58-62). It passes once N11 has finished at least one pure group's gather at its
  dispatch and waited for bucket group 0's at its dispatch in the warm-up.
- No row covers N3, N7, K6's fused step or epilogue, N5's early all-gather, or K1's table cache. All of them but the
  fused step print their own check lines.
- In the submitted runs the gate ran before `run_start`, and each of the ten printed its 19 armed parts and `True`
  (section 3).

**What a failure does, rank by rank.** The gate's verdict is per rank and is not all-reduced (`run_start_gate.py`
L120-131): each rank checks its own parts, prints if it fails, and raises.
- When every rank fails, every rank raises. In the one test that tripped the gate (on L4 v2's image; N7 switched off with
  K5's `main_grad` path on, a configuration the gate refuses by design), all 4 ranks raised the gate error. The job then
  hung for about 25 minutes, in Megatron-Bridge's process-group abort (a 600 s timeout) and a 480 s NCCL-watchdog stall,
  before SIGABRT ended every rank.
- When only some ranks fail, the passing ranks can log `run_start` and then block in a collective the failed ranks never
  join.
- Either way the failure happens before the clock starts: no scored attempt begins.
- The import-integrity failure in section 5 (on L4 v2's image) is different: a corrupted `.so` is the same on every
  rank, and all 8 ranks raised at `import firmus_ak`.

### Shape specialisation (TR-2.3)

Each replacement arms only for calls that match the shapes, dtypes, layouts and quantizer settings it was proven on, and
every other call runs the stock code. Each write-up quotes its gate. No gate reads the benchmark's identity.

**Gates on the identity of the replaced code.** Three gates check what code they are replacing, which is a property of
the software stack, not of the benchmark:
- **N7 and K15** select parameters by Megatron's layer names (`linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`):
  the dense GEMM weights whose gradients TE writes in one pass.
- **N12** arms only when the embedding in use is the recipe's own `custom_embed_forward`, the code it reproduces.
- **N13** builds its dgrad replacement only from Megatron's backward source with a recorded sha256.

**Gates that admit shapes training never sends.** Training and evaluation send 16,384 rows at every NVFP4 site and four
weight shapes, all multiples of 128 (the row-count census, section 5). Some gates admit more:
- **K7** (rows a multiple of 32): at 1,056, 2,080 and 4,064 rows SwiGLU and dSwiGLU return GEMM-swizzled, fully written
  scales where TE's are compact; the FP4 data and both amaxes are equal.
- **The norm wrapper K8 and K11 share** (rows a multiple of 32): K11's own gate needs rows a multiple of 1,024, so the
  other rows run K8's route. That route equals TE's at every multiple of 64 tested. At a multiple of 32 that is not a
  multiple of 64, N6's first-call check raises with a row + column quantizer; with a row-only quantizer the scales come
  back GEMM-swizzled where TE's are compact (FP4 data, amax and `rsigma` equal).
- **N6's weight-amax route** (any 2-D weight whose element count is a multiple of 8; K10 takes only dimensions that are
  multiples of 128): with compact scales and a dimension a multiple of 16 but not of 128 (e.g. k = 4,080), its
  first-call check compares whole scale tensors, including padding TE's cast leaves unwritten, and raises. Every defined
  byte equals TE's.

The conditions an item relies on but no code asserts are listed in [section 2](#2-assumptions-that-no-code-checks).

### State kept between calls (TR-8.3)

No kernel keeps a result keyed on data, and no kernel or wiring returns a stored value in place of computing it. What
is kept between calls:

| item | kept between calls | keyed on | why replaying identical inputs cannot hide it |
|---|---|---|---|
| K1 | the fp32 cos/sin table of the RoPE frequencies (at most 8 entries) | the frequency tensor's storage (pointer, shape, strides, dtype, device) and its version counter; a strong reference keeps the storage alive | the table depends only on the frequency tensor, a constant of the model; a modified tensor (new version) is recomputed; the first cache hit is compared with a fresh table, as bytes |
| K2, K4, K14, N10 | the last packed q\|k\|v buffer's pointer, size and partial maxima, until the next quantize | pointer and size of that buffer; single use | a quantize of any other tensor computes its own amax (K14's amax path); the next forward replaces an unused registration of a written buffer, and raises if a buffer K4 left unwritten under capture was never quantized (patch `te/0001` L205-206) |
| K4, K8, K13 (N6) | single-use hand-offs between consecutive operations: a fused sum and its amaxes; a residual add and its rsigma; a residual gradient | the tensor object, its storage pointer, size and version counter (and epsilon) | a mismatch drops the hand-off, and the stock computation runs; an unconsumed residual gradient or a pointer mismatch raises |
| K6 (N5) | per parameter group, a fused instance with its device address tables; the clip coefficient for the current step; the weight-amax slot lists | the shard data pointers (rebuilt when any pointer changes) | addresses and sizes only; the clip coefficient must be stashed fresh each step (an assert) and is cleared after use |
| K9 | one fp32 amax per 2D bf16 weight, written after each optimizer step and read by the next forward's weight casts | the parameter (pointer, size) | the optimizer's write of the bf16 weights is their only writer, and the buffer is rewritten after every step; at the second optimizer step the amaxes of the 21 weights K15 does not gather are compared with a fresh full-weight pass (0 differ), and those of the 109 it gathers are in K15's check |
| K10 | the cached QKV / proj weight workspaces, which TE itself keeps, updated in place | TE's workspace objects | rewritten every step from the current weights; checked at the next optimizer step against TE's cast of the same weight |
| K13 | which norm weights had their gradient add folded this step (`grad_added_to_main_grad`) | the parameter | reset by TE's path whenever a later backward for that weight is not fused |
| K15 (N11) | persistent FP4, tile-scale and norm images per bucket, and each gathered weight's NVFP4 storage; the other ranks' part of the bf16 parameter buffer goes stale | parameter identity and the bucket layout (set once) | the images are rewritten from the current shards every step; the stale bf16 part is never read (K15); every gathered weight is compared with TE's cast of the re-gathered bf16 weight at the second optimizer step (the first after setup), in the same call as K9's check |
| K16 (N13) | the pinned algorithm per problem; a 32 MiB cuBLASLt workspace | (forward or dgrad, rows T, device) | a fixed implementation choice, the same for any data |
| N7 | the stored weights' `main_grad` is not zeroed between steps | — | every element is written each step: TE's weight-gradient GEMM writes its whole output, and K5 every element of the embedding's; checked once, at a bucket group's first gradient sync outside graph capture, where the first step's NaN sentinel must have 0 survivors in its stored weights; in the submitted runs that count covers 49 of the 129 (N7) |
| N12 | the delayed-scaling (S, dP) updates of skipped calls, applied at a later call | the call index within a forward / backward phase, learned in the warm-up | every scale an attention kernel reads equals the stock value at the moment it is read (N12); a phase of another length raises |
| all | which first calls have been checked | operation, usage, call site, shape | the check compares against the stock computation on the same inputs |

### Run-to-run reproducibility

- **Neither NVIDIA's stock recipe nor L4 is bit-reproducible run to run in the submitted layout** (2 trays, DP 8):
  repeated loops with the same seed depart from each other, stock as well as L4.
- **With a device synchronize at the end of every training step**, the loops shared one trajectory, and on it L4 equals
  stock bit for bit: every loss, the evaluation and every rank's gradient norm (6 of 7 loops of L4 v2 and stock at DP 8;
  the seventh, an L4 v2 loop, departed on the gradient side). On this image two such loops equal stock's reference in all
  1,152 losses and the evaluation ([Tests before the scored runs](#tests-before-the-scored-runs)).
- **Two kinds of departure were seen in L4 v2.** (a) Gradient-side, shared with stock: every rank's forward is identical
  and one reduced gradient differs. (b) Forward-side, L4 v2 only: on one rank, an fc1 GEMM (TE's `general_gemm`, stock
  cuBLASLt code) gives a different output from identical inputs while K15's next parameter gather overlaps it, and the
  fc1 output changes by more than rounding. The cause of (b) is not established.
- **Removing the parameter-gather overlap removed (b) in every loop tested.** L4 v3 is that configuration (N11's
  no-overlap mode). In the 8 drift loops on this image (5 at DP 4, 3 in the submitted layout) no departure of either kind
  occurred ([Tests before the scored runs](#tests-before-the-scored-runs)).
- **The gradient-side departure, shared with stock, remains:** L4 v3 does not change it. The submitted runs are not
  claimed to be bit-reproducible run to run, as the reference recipe is not in this layout.
- **No kernel's input or output differed at any departure.** The bytes of the items at the optimizer-step boundary (K6,
  K10, K15, N5, N6, N11) were identical up to every departure.
- The submitted runs converged (section 7).

### The evaluation coupling

An evaluation at step k changes step k+1 deterministically, in NVIDIA's stock release as in L4, through two pieces of
TE 2.20 state: (i) FP8 attention's delayed-scaling amax for the softmax tensor S (`dot_product_attention.py`
L1243-1262, history 1): evaluation forwards write `amax_history[0]` under `no_grad`, where the scale update is skipped,
and the next training forward's layer exits consume it; (ii) fused attention's Philox reservation (2^19 offsets per call,
`cpp_extensions/fused_attn.py` L356-362), which advances the default generator that NVFP4 gradient stochastic rounding
draws from. **L4 adds nothing:** L4 v2 and stock are bit-identical with and without the evaluation, saving and restoring
(i) and (ii) around the evaluation removes the effect in both, and N12's batched delayed-scaling updates reproduce stock
exactly through it; L4 v3's mode moves no evaluation work (a forced sync is unchanged, N11). It is the reference
recipe's behaviour, so it is not changed here. A bit-identity comparison must hold the evaluation schedule fixed:
NVIDIA's 2-node leaf sets `VAL_CHECK_INTERVAL=768` (`config_GB300_2x4x2xtp1pp1cp1_8b.sh` L18), so every run, stock and
L4, evaluates every 768 steps.

### Code present but not executed with the submitted switches

Each path below is shipped but never runs in the submitted jobs. Most are reached only by a switch value the config does
not set. The others are reached only by calls or table entries the recipe never produces, or by no shipped code. The
switches that would raise instead of running are marked.

| path | file | reached only with | with the submitted config |
|---|---|---|---|
| X8: deferred small gradient adds (`_foreach_add_`) | patch `mcore/0003` L18-20, L87-102; `mcore/0004` (`_ak_flush_pending_adds`) | `AK_X8_FOREACH_ADD=1` | not set; the flush function runs on each gradient sync and returns at once with nothing pending |
| the stochastic-rounding cast extension `ak_l8b_nvfp4s` (`_sr_cast_ok`, `_sr_cast`, `_sr_quantize`) | `ak_l8b_nvfp4p_patch.py` L487-517, L1322-1352 | `AK_NVFP4P_SR=1` | not built; the switch **raises** at import (L864-866) |
| the 2D weight quantize of weight mode 1 (`_weight_ok`, `_weight_new` -> `ak_l8b_nvfp4w.quantize_2d`) | `ak_l8b_nvfp4p_patch.py` L1280, L1307-1320 | `AK_NVFP4P_WEIGHTS=1` | the config sets `amax`, so `_weight_ok` returns False |
| the "fused" norm cast (`ak_l8b_rht7.rmsnorm_amax`, `rht_cast_norm`) | `ak_l8b_nvfp4p_patch.py` L606-616 | `AK_NVFP4P_NORMCAST_MODE=fused` | the config sets `apply7` |
| the SR cast variants 79 / 99 of `ak_l8b_rht7` (`rht_cast_vp`; `inst_p79.cu`, `inst_p99.cu`) | `ak_l8b_nvfp4p_patch.py` L548, L665-667, L690-693 | `AK_NVFP4P_RHTCAST_VP=1` (set) and an SR RHT cast whose width is not in K12's table (4096, 6144, 28672) | the recipe's SR casts are all in the table, so these variants never run (every run's five RHT-cast lines name `ak_l8b_rht8`); with K12's table on, only variant 79 (the default for such a width) could be selected, and variant 99's only entry, width 28,672, is K12's (`ak_l8b_nvfp4p_patch.py` L526-528, L557-558, L619-625); such a call would get a first-call check against TE |
| the SR variants 335 / 355 of `ak_l8b_rht8` (`inst8_sr335.cu`, `inst8_sr355.cu`) | `firmus/ak/csrc/rht8/binding8.cu` L73-74 | — | compiled into K12's binary, but no entry of K12's table (`ak_l8b_nvfp4p_patch.py` L557-558) selects them |
| K1's forward and backward kernels (`_qkv_rope_fwd_kernel`, `_qkv_rope_bwd_kernel`; `FusedQKVRoPE.forward` / `.backward`) | `fused_qkv_rope.py` L88-137, L140-188, L257-286; patch `te/0001` L232-235, L257-258, L280-283 | forward: `AK_QKV_PACK=0` or `AK_QKV_AMAX=0`; backward: `AK_QKV_GRAD_AMAX=0`, or a gradient K4 declines (patch `te/0001` L150-159) | the config sets all three to 1, so every forward runs K2's kernel (eager) or K4's passes (captured), and every backward runs K4 |
| the Triton QKV direct passes (K3) | patch `te/0001` L100-102, L136-145, L215-222; `ak_l8b_fp8cs_patch.py` L87-121 | a packed q\|k\|v that K4's CUDA passes decline (head dim ≠ 128, micro-batch outside 1-4, non-contiguous, misaligned, not bf16) | imported because `AK_QKV_DIRECT=1` (L100-102), never launched: the recipe's head dim 128 and micro-batch 2 always get K4's callables (L136-145, L215-217). K3's five knobs (`AK_QKV_DIRECT_P1`, `_W1`, `_W2`, `_REV`, `_ASM`) are read at import (patch `te/0001` L98-99; `ak_l8b_fp8cs_patch.py` L42-47) and used only on these paths (patch `te/0001` L219-222; `ak_l8b_fp8cs_patch.py` L93-97, L107-109, L116-120), so they have no effect |
| earlier kernel versions inside the extensions (`AK_NVFP4P_KVER`, `_NORM_VER`, `_ADDSTATS_VER`, `_BWDNORM_VER`, `_RHTAMAX_VER`, `_NORMCAST_APPLY_VER`, `AK_QKVX_P1_VER`, `_P2_VER`, `_BWD_VER`) | the K4, K7, K8, K11, K13 sources | other version values | the config selects the version of every site but one, and the first-call lines name it (`v8`; `v6` and `add_stats v6`; `v7`; `apply7 v7`; K4's `v34` and `v37 / 37`). `AK_NVFP4P_RHTAMAX_VER` is not set: K13's `rht_amax` runs the code default, version 2 (`ak_l8b_nvfp4p_patch.py` L219, used at L1360), and its "RHT amax" line prints no version |
| exported functions no shipped code calls | `ak_l8b_rht7.sm_hog` (test only); `ak_l8b_nvfp4bx.rht_amax_post`; `ak_l8b_fp8q.amax_partials`; `ak_l8b_fp4g` versions 1 / 2 (`split_amax`, `shard_quant`, `post_ag`, `post_ag2`) and `sdec_check` (test only) | — | never called |
| the poison mode and the `all` / `off` check modes of K15 | `ak_l8b_fp4g_patch.py` L62-63, L501-512, L550-556 | `AK_FP4G_POISON=1`; `AK_FP4G_CHECK=all` or `off` | `POISON=0`, `CHECK=first` |
| the delayed-scaling digest writer of N12 (development) | `ak_l8b_smallk2_patch.py` L52, L339-383, L414-415 | `AK_SMALLK2_DUMP` set to a file path | not set |
| fixed development overrides | `ak_l8b_nvfp4p_patch.py` L529 (`_RHTCAST_FORCE = ""`), L757 (`_W2D_FORCE = ""`) | — | constants: no environment variable is read, and the branches they guard are never taken |
| the run-start gate switched off | `firmus_ak/run_start_gate.py` L23, L152 | `FIRMUS_AK_RUN_START_GATE=0` | not set; the code default is 1, and every run prints the gate line |

The modules' docstrings still describe these paths, and the three development modules this implementation does not ship
(section 7). The code that would load a module that is not shipped raises instead (`ak_l8b_nvfp4p_patch.py` L213-229,
L864-875).

## 5. Tests on the built image

The tests under "Tests before the scored runs" ran on the submitted image, against its own installed files; those under
"Tests on L4 v2's image" ran on L4 v2's submitted image. None is a scored run. The kits and the logs of every test are
available on request (records R3-R5, R7).

### Tests before the scored runs

| test | what it compares | items | result | conditions |
|---|---|---|---|---|
| Build-time install check (`firmus_ak.check_install`) | the installed tree against the build's rules | all | PASSED: no `.pth` loader, the 11 kernel-directory variables inside `/opt/firmus/ak`, no host path in any Firmus file, every expected output present | the image build |
| Import of `firmus_ak` in the clean image | the banner's re-hash against the build records | all built files | `[auto_kernel banner] MATCH`; all 8 modules import; N11's FP4G and no-overlap mode armed | the submitted switches |
| Import with one byte of `ak_l8b_nvfp4n.so` flipped | the banner's re-hash | banner | raises `[auto_kernel banner] MISMATCH` at import | a modified copy of the binary |
| Import with `AK_FP4G_NOOVERLAP=1` and `AK_FP4G` unset | N11's switch check | N11 | raises `AK_FP4G_NOOVERLAP=1 needs AK_FP4G=1` at import | — |
| The 24 frozen development kits, re-targeted at the image | each kit's frozen tests on the image's files (what each compares: its write-up; "The kits", below) | K1-K16, N1, N2, N6 (the kit of the module's predecessor version; the shipped module's own kit is under L4 v2's image), N7-N10, N12, N13 (K16's kit, which tests the hooks with its own development copy of N13's module) | **48 / 48 PASS**, including the kits of K10, K11 and K12, whose binaries this image rebuilt | one GPU each; each kit in two forms: as written, and with PyTorch's memory pools first filled with random bytes |
| fp4g patch-module kit, L4 v3 version | the gathered weights against TE's 2D cast of the re-gathered bf16 weights; losses and evaluations against a control with the gather off | N11, K15 | **9 / 9 checks per run with 0 differing bytes; no NaN; the no-overlap ARMED line and the gate's `fp4g no-overlap` part in the two no-overlap runs only; the 8 training losses and 2 evaluation values of all three gather runs bit-identical to the control**. Its L4 v2 version passed on L4 v2's image | real training on one tray (DP 4, 32 layers, synthetic data, 8 steps, evaluations after steps 4 and 8) with the L4 v3 leaf's switches; full-iteration graph and eager with the no-overlap mode on, and graph with it off (L4 v2's overlapped gather), each with the stale bf16 weight copies poisoned to NaN and a byte check at every step; the control with the parameter gather off |
| Arithmetic-identity loops | every loss and the evaluation against the stock recipe's reference loop | all | 2 of 2: the 1,152 losses (md5 `5ee7a6f05839`) and the evaluation (5.51950740814209) equal the stock reference exactly, on both trays | seed 4243, 1,152 steps, DP 4, a device synchronize at every step end, evaluation at step 768 (the layout of the stock reference); one loop on each of the two trays |
| Drift loops, one tray | every loss, evaluation and rank-0 gradient norm against the clean reference trajectory of the earlier probes | N11, all | 5 / 5 identical: 0 forward-side, 0 gradient-side departures | seed 4243, 1,152 steps, DP 4, evaluation every 192 steps, no sync, with a light localiser and pre-reduce gradient digests |
| Drift loops, the submitted layout | the same streams against one loop of the same layout with the step-end sync | N11, all | 3 / 3 identical: 0 forward-side, 0 gradient-side departures | 2 trays, DP 8, global batch 16, seed 4243, 1,152 steps, the same instrument and evaluation interval, no sync |
| 2 × 4 smoke run | the proof lines and the gate | all | every proof line True (56); run-start gate True over 19 armed parts | 800 steps, the submitted config, two trays |
| The ten submitted runs | the first-call checks, the banner and the gate | all | each: banner MATCH; all 64 verdict lines True; run-start gate True over 19 armed parts (section 3) | the submitted config |

**The kits.** 26 kits in all: the 24 frozen development kits, re-targeted at the image's files with their tests and
inputs unchanged, and the 2 patch-module kits written for L4 v2's image.
- The nvfp4p patch-module kit runs the two-layer test of the predecessor module's frozen kit against the shipped module
  and against stock TE, and compares the two (on L4 v2's image).
- The fp4g patch-module kit repeats the module's pre-deployment training selftest with the shipped module. Its L4 v3
  version adds a run with the no-overlap mode off and checks the mode's ARMED line and gate part.
- **What a re-targeted kit checks, and what else it runs.** Before its tests, each re-targeted kit checks every image file
  and binary it loads against the image's build records, and exits on any difference. The kits were frozen during
  development, and many also carry code that is not in the image, which some tests run as a reference or as context:
  copies of the dropped modules `ak_l8b_wq`, `ak_l8b_rht6` and `ak_l8b_nvfp4b`, an earlier `ak_l8b_fp8cs` and an
  earlier `ak_l8b_nvfp4p`, and earlier versions of shipped files (for example `fused_qkv_rope.py` `a91b35c1` and earlier
  NVFP4P patch modules). A carried file is pinned to its frozen sha256 and checked before the tests. So some kit
  comparisons are against that code, not TE: for example K4's kit compares the QKV-gradient amaxes with the kit's own
  `ak_l8b_nvfp4b`, K7's kit compares versions 7-10 of the image's module with its version 7, and K12's kit compares its
  CUDA-graph replays with `ak_l8b_rht6` / `ak_l8b_rht7`. The kits of N8 and N9 do not run the shipped patch module:
  their tests apply the module's change to TE themselves. Each write-up says what its kit compares.

### Tests on L4 v2's image

L4 v2's submitted image; its files differ from L4 v3's only in N11's module, the run-start gate, the leaf's one added
switch, four comments or docstrings, and the three rebuilt extensions (K10, K11, K12). These tests ran L4 v2's builds of
those files; the kits above re-tested this image's builds, and the fp4g kit's L4 v3 version this image's N11. Unless a
test names a test-only mount, each ran with the image's own files and no file mounted over them, armed the image's modules
as `pretrain.py` does with L4 v2's leaf's 60 switches, recorded the sha of every file it loaded and printed the banner
`MATCH`. Some compare with NVIDIA's 20261006 release image (stock), as named.

| test | what it compares | items | result | conditions |
|---|---|---|---|---|
| Unit tests of the whole bundle, with the kits, the call-site tasks, multi-rank routes and model-level comparisons | every output byte against the stock path | all 29 items | **the 35 package files byte-identical in L4 v2's image**; the build records verify; banner MATCH in every route; every called item equal to the stock path on every case of the shapes training and evaluation send (N4 outside its restricted domain; off those shapes, section 4); K3 never called; no failure | visible, reseeded, hidden, shape-variant and edge cases, in fresh processes and under CUDA-graph replay (K10's unit test in a separate job) |
| Coverage of every called path | which test grades each path the runs call | all | every called path of every kernel and patch is graded by a test above that found it equal to the stock path | the paths the runs call |
| Closed optimizer loops | the optimizer state after every step against TE's chain | K6, N5 | **E0 in all six**; an earlier X9 kernel and an FTZ-stripped variant were caught at steps 550-614 | six loops of 2,002 optimizer steps |
| In-training shadow on live tensors | the stock computation re-run beside the live one on the same tensors | K6, N5 in graph and eager mode; K7, K8, K11, K12, K14, K16, N6, N10, N13 in eager mode only | **0 mismatches**; a planted mismatch caught on every rank | DP 8 |
| Real training tensors as hidden cases | captured cases against the stock path | K5, K6 | **6 / 6 and 17 / 17 captured cases E0** | cases captured from a stock DP-8 run (NVIDIA's 20261006 release image) to its final evaluation |
| Census of N4's input domain | the count of in-band gradient elements | N4 | **0 in-band elements of 6.93e14** nonzero gradient elements | one armed L4 v2 run at the submitted DP-8 layout, the L4 v2 leaf unchanged, to its final evaluation; every bucket at every step on 8 ranks; one run, one seed |
| Census of the NVFP4 sites' row counts | the row count of every NVFP4 call | K7, K8, K11-K13, N6 | every row count 16,384, none off the 1,024-row family: training and evaluation never send the shapes of section 4's "Gates that admit shapes training never sends" | DP 8, graph mode, 1,152 steps, rank 0 counted (24 row families); the eager shadow run (27); a one-rank graph-mode run of 48 steps for the weight shapes |
| ARMED-line reader on L4 v2's ten scored logs | each ARMED-line sha against L4 v2's documents | all | every ARMED-line sha equals L4 v2's document's; 0 `False`; K8's five lines carry no sha (this package's logs: section 3) | — |
| The Firmus call-site verifier's tasks for K1, K5 and K6 | the kernel against the stock call | K1, K5, K6 | **6 / 6 E0** | two forms each: as written, and with the allocator's memory pools first filled with random bytes; K1's task runs the earlier X12 module `a91b35c1`, whose four Triton kernels are text-identical to K1's file, without the cache |
| Call-site checks for K1, K5, K6 | every timed call verified on fresh contents; a stricter static check; the launch plan and streams | K1, K5, K6 | **E0** with every timed call verified (2.356x, 4.523x, 2.617x); the stricter static check passes with each kernel's declared allowances; the launch-plan check records K5's forked-and-joined side stream | the stricter static check and the launch-plan check are test checks, not the verifier's default gates |
| Self-test of the call-site verifier | whether each of its gates is necessary | the verifier, no kernel | every gate of the verifier shown necessary; no result changed | — |
| Deliberate failure in a 2 × 4 job of the submitted config | the banner, fail-loud | banner | **every one of the 8 ranks raises `[auto_kernel banner] MISMATCH … changed /opt/firmus/ak/lib/ak_l8b_nvfp4n.so` at `import firmus_ak`; no `init_start`, no `run_start`** | a test-only mount replaces `ak_l8b_nvfp4n.so` with a copy that has one byte inverted |
| `make_pin.py --check` | the LM-head pin, enumerated again on the image's cuBLASLt | K16 | **the pin holds** | K16's binary is the same file in both images |
| Optimizer A/B | the image's X9 kernel against TE `FusedAdam` and `apex.optimizers.FusedAdam` | K6 | **192 comparisons, 0 differing bytes** (K6) | 100 steps, 4 configurations |
| Optimizer shadow in training | K6's step against TE's chain | K6, N5 | **16 / 16 checked steps, on all 8 ranks, byte-equal to TE's chain** (K6) | a 2 × 4 run of the submitted config for 1,536 steps, with test-only mounts |
| GSA collective test at world 8 | pre-scale + SUM against AVG | N4 | **0 differing bytes** (N4) | 6 data patterns; its 16,777,216-element shards are below the size at which differences in N4's restricted domain appeared |
| nvfp4p patch-module kit | two mock transformer layers of the recipe's TE pieces: the installed module (L4 switches) against stock TE (no Firmus module) | N6, K7-K13 | **byte-identical: output, input gradient, all 12 main_grads and the RNG offset, in every case** | eager (2 steps) and replays of a captured CUDA graph (2); two forms: as written, and with the allocator filled with random bytes |
| fp4g patch-module kit, L4 v2 version | as its L4 v3 version, without the no-overlap mode | N11, K15 | **9 / 9 checks per mode with 0 differing bytes; no NaN; the 8 losses and 2 evaluation values bit-identical to the control** | L4 v2's module and leaf; graph and eager with the stale bf16 copies poisoned to NaN, and the control with the gather off |
| Same-seed comparisons of repeated loops, and the reproducibility probes | losses, gradient norms and activations of repeated loops; stock against L4 | all | [Run-to-run reproducibility](#run-to-run-reproducibility) and [The evaluation coupling](#the-evaluation-coupling) | DP 8 and DP 4, seed 4243, 1,152 steps; NVIDIA's 20261006 release image for stock |

## 6. Optimizer, gradient clipping and numeric formats

- **Optimizer (TR-9.1c, TR-9.5).** X9 is K6 (the kernel) + N5 (the wiring). The recipe's optimizer is Megatron-Core's
  `DistributedOptimizer` with TE's `FusedAdam` in AdamW mode, fp32 master weights and moments, and bf16 parameters. K6
  gives the update's equations on both sides, the hyperparameters (β 0.9 / 0.95, ε 1e-5, weight decay 0.1, clip 1.0,
  peak LR 4e-4, 16 warm-up steps, cosine to LR / 10, all NVIDIA's), the A/B against TE and apex `FusedAdam`, the
  in-training shadow and the closed loops. No epsilon, weight-decay or clipping change, so no TR-9.5 exception.
- **Gradient clipping (TR-5.2b).** The coefficient is computed by Megatron's stock code; X9 changes only how the local
  norm is computed (TE's exact reduction order, byte-identical except on K6's stated subnormal domain, where no compared
  case or step showed a difference) and applies the coefficient inside the fused step. The gradients that enter the norm
  are byte-identical to stock (K5, K13; N4 outside its restricted domain; N7 up to the sign of exact zeros). K6 and N5
  give the details.
- **Numeric formats and logged precision (TR-5.2a, TR-5.3).**

| quantity | stock recipe | this implementation |
|---|---|---|
| linear-layer inputs / weights / gradients | NVFP4 (RHT + RN / 2D RN / RHT + SR) | identical (K7-K13, K15) |
| attention | FP8 E4M3 / E5M2 current scaling; delayed scaling for S / dP | identical (K2-K4, K14, N12) |
| LM head | bf16 | bf16 (K16, the same GEMM) |
| optimizer | fp32 master / m / v, bf16 parameters | identical (K6) |
| parameter all-gather | bf16 | NVFP4 for 109 of 128 weights (K15); bf16 for the LM head, embedding and norms |
| gradient reduce-scatter | bf16, pre-scale + SUM | bf16, NCCL AVG (N4): the same bytes outside its restricted domain, which a whole-run census found empty |

The MLLOG precision keys are NVIDIA's: `lowest_numerical_precision_in_linear = nvfp4`, `_in_attn = fp8`,
`_in_comm = nvfp4`. Because of K15, `_in_comm = nvfp4` is literally true here.

## 7. Training evidence and modules not shipped

**Convergence (context only, not equivalence evidence).** Unchanged convergence is not offered as proof of equivalence;
the write-ups are. For context:
- The ten submitted runs reached the target in 14 evaluations (5 runs) or 15 (5 runs); olympic time to train 52.402 min.
- NVIDIA's recipe without these kernels, run as a ten-run set on NVIDIA's 20260929 release on two other trays of the
  same rack, took 14 (8 runs) or 15 (2 runs).
- L4 v2's ten, on those two other trays, took 15 (7 runs) or 14 (3 runs); olympic 52.868 min.
- The RCP check passes with a submission mean of 178,176 samples.

**Modules not shipped.** Three modules of the development stack never executed in it, and are not built here:

| module | why it never executed | here |
|---|---|---|
| `ak_l8b_wq` (2D cast into the cached workspaces) | every workspace went to K10 | not built. A workspace K10 declines stays on TE's quantize; one that reaches the in-place cast raises |
| `ak_l8b_rht6` (the earlier RHT cast) | the fallback for a (width, rounding) pair outside K12's table; all five recipe casts are in it | not built. A round-to-nearest cast outside the table runs TE |
| `ak_l8b_nvfp4b` (the earlier norm backward) | superseded by K13, which carries its kernels | not built. Switches that would load it raise |

Copies of all three are carried by some of the frozen kits (section 5).

## 8. Glossary

- **The submitted image.** Built from this directory's Dockerfile on 2026-10-14; tag `l4v3_20261014`, squashfs sha256
  `1d55d6d5…` ([README.md](README.md), "Build"). Its TE is 2.20.0 at `5b28d7aa` (`Dockerfile` L82-84).
- **L4 v2's image.** The image of `nemo_l4_nvfp4_v2` (withdrawn from this submission), built on 2026-10-09; tag `l4_20261009b`, squashfs sha256
  `23469bf7…`. The tests under "Tests on L4 v2's image" ran on it.
- **dev image 0929.** NVIDIA's 2026-09-29 release image of this recipe, on which the kernels were developed, with the
  kernel files mounted into the container (the development stack). Its TE is at `131157b5` (`Dockerfile` L13); some
  write-ups cite TE line numbers read on it, and say so. Evidence from it is labelled development evidence.
- **NVIDIA's 20261006 release image.** The stock release this package rebuilds, with no Firmus code: the stock side of
  the comparisons with stock. Its TE is the same commit as the submitted image's.
- **Tests on L4 v2's image.** In a write-up, the tests that ran on L4 v2's submitted image, under one label line that
  names what differs from this image.
- **The Firmus call-site verifier.** Firmus's internal test harness. It runs a kernel and the stock call on recorded and
  edge-case inputs and compares the output bytes. It is not part of the package.
- **Kit.** A frozen development test suite for one kernel or patch, re-targeted at the image's installed files and run
  on the image in two forms: as written, and with PyTorch's memory pools first filled with random bytes (section 5).
- **Unit test.** A test that arms the image's own modules as `pretrain.py` does, with the leaf's switches, and compares
  every output byte with the stock path on visible, reseeded, hidden, shape-variant and edge cases.
- **Model-level comparison.** Short DP-4 training loops on mock data: the bundle against the bundle with one item
  switched off, and against stock.
- **ARMED line.** The `[auto_kernel <tag>] … ARMED …` line rank 0 prints when a replacement arms; most carry its module's
  sha and its first-call verdict (section 3).
- **Proof line.** A line every scored log must contain. Firmus's post-run check matched 56 such patterns (L4 v2's 54,
  with the three rebuilt binaries' shas, and two for the no-overlap mode), from a pattern file in this package's
  evidence (record R4); every scored log matched all of them.
- **Nicknames** (the tags in the ARMED lines and switch names): x12f / X12 → K1; QKVA, x12a → K2; x12d → K3; qkvx → K4;
  emb2 → K5, N2; X9, x9step3, x9e0, x9fold → K6 (with N5); nvfp4p → K7 and N6; nvfp4n → K8; nvfp4w → K9; w2d7 → K10; rht7
  → K11; rht8 → K12; nvfp4bx → K13; fp8q, fp8cs → K14, N10; fp4g → K15, N11; lmh → K16, N13; x12pack, qkvpack → N1;
  TCE, TCEV, tce0929 → N3; GSA → N4; X7 → N7; PROJCOL → N8; OPSWZ → N9; smallk2 → N12.
- **Records available on request.** Each write-up ends with the Firmus records behind its evidence (experiment-log
  sections, the validated-kernels catalogue, kit and test logs). They are not part of the package. Reviewers can request
  them (TR-12.1e).

## Records available on request

- R1: The test plan of 2026-10-12 (items V1-V31) behind the tests on L4 v2's image: `2026_10_12_COMPLETE_TESTS_PLAN.md`
  of Firmus's Llama kernel repository. Its items, by section 5 row: the unit tests of the whole bundle V24 with V23 (job
  22116, 2026-10-13; K10's unit test job 22133); coverage V30; closed optimizer loops V18; in-training shadow V21; real
  training tensors V28; the censuses V22 (N4's: job 22189); ARMED-line reader V20; call-site checks V1, V3, V7;
  verifier self-test V25; same-seed comparison of repeated loops V31.
- R2: The results of those tests: that repository's `EXPERIMENT_LOG.md` (commit `93e7934`), sections L17-L21; the
  validated-kernels catalogue `VALIDATED_KERNELS_CATALOGUE_L8B.json`, keys `wave2` (the bundle unit tests, V24;
  coverage, V30), `wave3`, `wave4`.
- R3: The reproducibility probes of 2026-10-13 and 2026-10-14, each with its report: the first whole-run comparison and
  its review; the race probe (DP 4, one tray); the DP-8 probe (the submitted layout, with and without a step-end sync);
  the ordering probe (graph structure, device timestamps, addresses); the evaluation-coupling probe; the localisation
  probe (where the drift enters; final). Also the probe of K8 and K15 against TE's stock operations. The reports number
  the probes E1-E8; E4 is the stock reference of the identity loops, E6 the no-overlap probe (the DP-4 drift loops
  compare with its clean reference trajectory), and E8 the localisation probe. L4 v3's identity and drift loops (jobs
  22278 on CN010435, 22279 on CN010431, 22368 on CN010431 + CN010435), their code and logs are in this package's
  evidence (`tests/probes/`).
- R4: The 56 proof-line patterns of the submitted image: `proofs_l4v3_l4v3_20261014.txt` (Firmus's post-run check), in
  this package's evidence (`launch/`); each run's check, `postten/per_attempt_checks.txt` L1-10.
- R5: The kit and test logs of section 5 (the kits, the call-site tasks, the A/B and shadow runs, the GSA collective test,
  the forced-failure run). Those run on this image are in this package's evidence (`image/`, `tests/`, `smoke/`): the 24
  kits, job 22275 on CN010435 (`tests/kits/`); the fp4g patch-module kit's L4 v3 version, job 22276 on CN010435
  (`tests/fp4g_kit/`); the image build, job 22273 on CN010431, and the three image checks (`image/`).
- R7: The ten scored runs and the smoke run of this package: CN010431 + CN010435; the smoke, job 22409
  (`smoke/run_1/`). The stock ten-run set on NVIDIA's 20260929 release and L4 v2's ten ran on CN010412 + CN010413
  (`postten/compare_stock0929_v_L4v3.*`, `postten/compare_L4v2_v_L4v3.*`).
