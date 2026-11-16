# nemo_l4_nvfp4_v3: Llama 3.1 8B, NVIDIA's NeMo / Megatron-Bridge recipe with Firmus kernels (Firmus, MLPerf Training v6.1)

System: `firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3`, 2 nodes × 4 NVIDIA GB300 (one GB300 NVL72 NVLink domain).

## L4 v3 compared with L4 v2

L4 v3 is the L4 v2 implementation (`nemo_l4_nvfp4_v2`, an earlier Firmus entry, withdrawn from this submission) with one change: **the forward's parameter all-gathers no longer overlap compute.**
- **Why.** In L4 v2's layout, one rank's forward occasionally computed a different fc1 GEMM output from identical inputs, different by more than rounding. That GEMM was a stock cuBLASLt NVFP4 GEMM of the first layer to use an FP4G parameter group, and it ran while the next group's gather (K15) ran beside it on FP4G's side stream. Its cause is not established. With every forward gather finished right after its dispatch, the departure did not occur. [KERNELS.md, "Run-to-run reproducibility"](KERNELS.md#run-to-run-reproducibility), gives the evidence.
- **What changed.** Only where two waits happen, in `firmus/ak/pkg/ak_l8b_fp4g_patch.py`:
  - each FP4G group's gather is finished (all-gather wait, `post_ag3`, norm unpack, join) right after its dispatch;
  - bucket group 0's stock bf16 all-gather is waited for right after its dispatch.

  The same kernels run on the same buffers in the same order, so the arithmetic is unchanged.
- **The switch.** `AK_FP4G_NOOVERLAP=1`, set in [config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh](config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh). Each log prints `[auto_kernel fp4g] no-overlap parameter gather ARMED (AK_FP4G_NOOVERLAP=1): ...`, and the run-start gate lists the part `fp4g no-overlap`. It is checked only when both waits fired during the warm-up.
- **Fail loud.** The switch raises at import without `AK_FP4G=1`. Setup raises if the wrapper is not installed or the stock group is not found.
- **Nothing else changes.** Every other kernel, setting, hyperparameter and recipe file is L4 v2's. [firmus/ak/pkg/CHANGES_vs_validated.md](firmus/ak/pkg/CHANGES_vs_validated.md) lists the 8 edits. [firmus/ak/pkg/diffs/ak_l8b_fp4g_patch.diff](firmus/ak/pkg/diffs/ak_l8b_fp4g_patch.diff) is the module against its validated source.

## What this implementation is

This is NVIDIA's `llama31_8b` reference implementation from the MLPerf Training v6.1 release of 2026-10-06. NVIDIA's original instructions are in [README_NVIDIA.md](README_NVIDIA.md).

**What Firmus changed:**
- **The image is rebuilt from the public base.** The [Dockerfile](Dockerfile) is NVIDIA's own Dockerfile for this recipe, kept verbatim as `docker/Dockerfile.nvidia_1006`. Firmus pinned it to the public NGC base `nvcr.io/nvidia/pytorch:26.09-py3` by digest and added three Firmus steps, marked F1, F2 and section 5.
- **Patches to the upstream libraries** (F1, F2). The files in `firmus/patches/` are applied with `git apply` before each library is installed:
  - Transformer Engine 2.20.0 (`5b28d7aa`): `firmus/patches/te/`;
  - Megatron-Core (`f6c33bde`, the submodule of Megatron-Bridge `ea5dded8`): `firmus/patches/mcore/`.
  - Two recipe files are edited in place: `embedding_lib/custom_embedding/custom_embedding.py` and `pretrain.py`. Their diffs against NVIDIA's files are `firmus/patches/recipe/`.
- **The Firmus kernels** (section 5). They are compiled from the sources in `firmus/ak/` by `firmus/ak/build_all.py`, and installed with the `firmus_ak` package under `/opt/firmus/ak`. No binary enters the image from outside: the build context excludes every build output (`.dockerignore`).
- **A switches leaf.** [config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh](config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh) sources NVIDIA's 2-node GB300 leaf `config_GB300_2x4x2xtp1pp1cp1_8b.sh` unchanged and adds only `AK_*` switches ([KERNELS.md, section 1, N14](KERNELS.md#n14-the-l4-config-leaf)). Most of them turn the Firmus kernels on or select their versions and launch shapes. Six affect no code that runs with the recipe's shapes: K3's five launch knobs, and `AK_NVFP4P_RHTCAST_VP=1`, which arms K11's stochastic-rounding cast variants 79 / 99 (never reached). K3's own switch, `AK_QKV_DIRECT=1`, is also the switch of K4's forward passes, which run; K3's kernels are imported but never launched. The hyperparameters, batch size, precision recipe, evaluation and stopping rule are NVIDIA's.

**What the kernels do.** Every kernel computes exactly what the stock code computes, bit for bit (equivalence class E0), except K6's gradient-norm pass: it keeps subnormal partial sums that TE's fast-math kernel flushes, so it is byte-identical except on a stated subnormal domain, where no compared case or step showed a difference (E0-restricted; [KERNELS/K06_x9step3_fused_adamw.md](KERNELS/K06_x9step3_fused_adamw.md)). Each one only runs it faster: it fuses passes, removes copies, or uses a better-scheduled kernel for the same arithmetic.

**Where to read more.** [KERNELS.md](KERNELS.md) is the kernel supplementary information: an index, with one write-up per item under [KERNELS/](KERNELS/) for the 16 kernels (K3 is shipped but never launched with the recipe's shapes, and of K1 only the angle table runs) and 13 of the 16 non-kernel items (N14-N16 are in the index). Each write-up gives:
- the source, what it replaces and the equivalence argument;
- its numerics, the optimizer and gradient-clipping behaviour and shape specialisation;
- how it is enabled, and the tests on this image and on L4 v2's image, with their conditions and results.

The write-ups in `KERNELS/` are generated by `tools/build_kernel_docs.py` of the submission repository from its central copies (`kernels/llama_nvfp4_lw/`), which L4 v2's package shared; `KERNELS/SOURCES.sha256` records the sha256 of each source and output. Each write-up starts with a note on how L4 v3 relates to L4 v2 for that item.

**What KERNELS.md states.**
- N4 (GSA) is E0-restricted, and a census of one whole run (on L4 v2's image) found no gradient element in its domain. K6's gradient-norm pass is E0-restricted, on a subnormal domain where no compared case or step showed a difference. N7 is "E0 in value".
- Neither NVIDIA's recipe nor L4 is bit-reproducible run to run in this layout, and the runs are not claimed to be. L4 v3 removes only the forward-side departure seen in L4 v2 (above); the gradient-side departure, which stock shows too, remains ([KERNELS.md, "Run-to-run reproducibility"](KERNELS.md#run-to-run-reproducibility)).
- The evaluation coupling is stock TE behaviour ([KERNELS.md, "The evaluation coupling"](KERNELS.md#the-evaluation-coupling)), and the run-start gate's verdict is per rank.

**Revision of 2026-10-15.** `KERNELS.md` and the write-ups in `KERNELS/` were rewritten again for a reader outside Firmus: each item states what it changes, why its result equals NVIDIA's code, the tests on the submitted image with their conditions and results, and the properties a reviewer needs (class and domain, input gates, assumptions no code checks). Internal record identifiers appear only in each file's records list. No class, code, config, image or result changed.

## How the kernels are switched on

- **The training program enables them explicitly.** `pretrain.py`'s first import is `import firmus_ak`, which imports the eight Firmus patch modules.
- **Each module arms on its `AK_*` switch alone.** Once armed, a module patches the named Transformer Engine / Megatron functions as they are imported. A patched function uses the Firmus kernel only for calls that pass its shape, dtype and quantizer checks; every other call runs the stock code.
- **Nothing depends on the program's name or on the benchmark.** Processes other than training never load the modules: the mount check, the NCCL test and the compliance checker.
- **Every default directory in the Firmus code is inside the image** (`/opt/firmus/ak/...`). The Dockerfile's `ENV` sets the same paths. The install check fails the build if any directory variable points outside `/opt/firmus/ak`, or if any Firmus file names a host path.
- **Fail loud.** The build records the sha256 of every installed Firmus file and every patched upstream file (`/FIRMUS_AK_BUILD_SHA256.txt`, `/FIRMUS_STACK_SHA256.txt`).
  - At `import firmus_ak`, every rank re-hashes them, and any difference raises before the model is built.
  - Every armed kernel except K6 checks its first call against the stock path byte for byte, and raises on a mismatch. K6's fused optimizer step and norm pass are checked by tests on the image ([KERNELS/K06_x9step3_fused_adamw.md](KERNELS/K06_x9step3_fused_adamw.md)).
  - A gate at the end of the warm-up raises if any armed part never ran its check. All of this happens before `run_start`. The gate's verdict is per rank, not all-reduced ([KERNELS.md, "Enablement and fail-loud behaviour"](KERNELS.md#enablement-and-fail-loud-behaviour)).

## Modules not shipped

Three modules of the development stack never executed in it, and are not built here: `ak_l8b_wq`, `ak_l8b_rht6` and `ak_l8b_nvfp4b`. Where they would have been a fallback, the stock Transformer Engine code runs instead. Details: [KERNELS.md, section 7](KERNELS.md#7-training-evidence-and-modules-not-shipped), "Modules not shipped".

## Logged precision

The MLLOG keys `lowest_numerical_precision_in_linear = nvfp4`, `_in_attn = fp8` and `_in_comm = nvfp4` are NVIDIA's values for this recipe.
- For this implementation, `_in_comm = nvfp4` is literally true: the Firmus parameter all-gather (kernel K15) sends 109 of the 128 weights as NVFP4. See [KERNELS/K15_fp4g_param_allgather.md](KERNELS/K15_fp4g_param_allgather.md).
- The quantization formats themselves are NVIDIA's: NVFP4 with 1D-16 and 2D-16×16 block scales, the random Hadamard transform, and FP8 current scaling for attention.

## Build

```bash
docker build -t llama31_8b_l4 .
enroot import -o llama31_8b_l4.sqsh dockerd://llama31_8b_l4
```

The results in `results/firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/llama31_8b/` came from the image built from this directory on 2026-10-14:
- tag `l4v3_20261014`, docker image `sha256:bc859ba89be7…`;
- squashfs sha256 `1d55d6d50d540c7688046dc1cb0b09ae13ef5745ec0ae1c18ec08f236d484942`, 26,061,021,184 bytes;
- `/FIRMUS_AK.env`: `FIRMUS_AK_BUILD_SHA256=d041cf2e1a22bf4551a464a56f1e3000ceaaa43ef7bcc5e79d611a8e84ec65ec`.

Every training log prints the build record's sha256 as `[auto_kernel banner] tree_sha256=…`.

**This directory compared with the build context.** Every code, config and build file is byte-identical to that build. [../firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/BUILD_CONTEXT.sha256](../firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/BUILD_CONTEXT.sha256) lists the 142 files as built. The only differences are five files and one added directory, none of them code that runs:
- `README.md` (this file) and `KERNELS.md`: written after the build.
- `KERNELS/` (the per-item write-ups and `SOURCES.sha256`): written after the build. It is not in the build context or the image, no build step or run reads it, and it is generated by `tools/build_kernel_docs.py` of the submission repository.
- `README_NVIDIA.md` is the build's `README.md`, renamed.
- `run.sub`: the Slurm launcher that ran the jobs. It is host-side and not used inside the image. NVIDIA's `run.sub`, the one in the build, is kept as `run.sub.vendor1006`, and every difference is listed in `../firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/run.sub.vs_nvidia1006.diff`.
- `.dockerignore` is left out, because the package may not contain hidden files.
  - It only kept build outputs out of the build context: `**/build/`, `**/*.egg-info/`, `**/__pycache__/`, `**/*.pyc`, `**/*.o`, `**/*.so`, `**/.ninja_deps` and `**/.ninja_log`.
  - This directory contains none of them, so a build from it is unchanged.

The change records `firmus/ak/pkg/CHANGES_vs_validated.md` and `firmus/ak/pkg/diffs/*.diff` were built as they are here: in the build too, internal home-directory paths in their removed lines (the old kernel-directory defaults) are replaced by `[campaign directory]`.

[../firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/README.md](../firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/README.md) ("Rebuilding from the exact build context") recreates the build context from this directory, including the `.dockerignore`, and moves `KERNELS.md` and `KERNELS/` out; all 142 files then match `BUILD_CONTEXT.sha256`.

## Launch

See [../firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/README.md](../firmus_gb300_ngpu8_llama8b_l4_nvfp4_v3/README.md): `setup.sh`, `init_datasets.sh` and `run_and_time.sh`. In short:

```bash
export CONT=$PWD/llama31_8b_l4.sqsh DATADIR=/path/to/data LOGDIR=/path/to/results   # DATADIR: README_NVIDIA.md 3.2
source config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub
```

**`config_filename`.** The logs record `config_filename = config_GB300_2x4x2xtp1pp1cp1_8b.sh`. The config that ran is `config_GB300_2x4x2xtp1pp1cp1_8b_l4.sh`.
- NVIDIA's `pretrain.py` logs `config_${DGXSYSTEM}.sh`.
- `DGXSYSTEM=GB300_2x4x2xtp1pp1cp1_8b` is set by NVIDIA's leaf, which the L4 v3 leaf sources unchanged and then adds only the `AK_*` switches.
- Both files are in this directory. The training rules ask that `config_filename` be "named so that it can be matched to the submitted config"; the logged name is the submitted NVIDIA leaf, and this note matches it to the L4 v3 leaf that sources it.
- The ten runs are filed as run, with this note.

## Data indices built before `run_start`

NVIDIA's `callback_logging.py` (`on_data_init_start`, L128-161) prepares the dataset indices before it logs `run_start`. Each log shows this as `> building train, validation, and test datasets for GPT ...` up to `> finished creating GPT datasets`, before `run_start`.
- Rank 0 builds Megatron's document, sample and shuffle indices (`.npy` files in a fresh per-job directory) from the `.idx` metadata: the document count and the document lengths.
- Every rank then reads the `.npy` files once, so they are in the page cache (`_warm_npy_page_cache`, L71-112).
- No training token file (`.bin`) is read; the first token read is in the timed run.

Training rules L256-258 allow metadata such as document counts and lengths to be used before the clock starts. L240 asks that preprocessing repeated on every run be timed. The shuffle index depends on the run's seed, but it only orders documents: no token is transformed, augmented or read. This is NVIDIA's code, unchanged; their entry builds its indices the same way.
