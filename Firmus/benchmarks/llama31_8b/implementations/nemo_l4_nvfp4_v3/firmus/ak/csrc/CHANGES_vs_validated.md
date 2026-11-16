# firmus/ak/csrc: changes against the validated files

Every file under `firmus/ak/csrc/` and `firmus/ak/kernels/` was copied from the Llama 3.1 8B campaign
(`auto_kernel_llama31_8b`) and checked against its validated sha256 (`VALIDATED_KERNELS_CATALOGUE_L8B.json`, the deployed
overlays, the w2d7 source tarball `4cd52c90`). `../SOURCES.sha256` lists every shipped file's sha256 as shipped.

The table below covers every file whose shipped bytes differ from the validated bytes. **No kernel source differs.**
That covers every `.cu`, `.cuh` and `.cpp` file that is compiled, every Triton kernel, and `lmh_pin.json`. Only Python
tools changed: the build script, the header generators and the lmh test/pin enumerator.

## 2026-10-09: no `sys.argv`, no `*_FORCE` (check_impl_hygiene H2; docs/KERNEL_CODE_IN_DOCKERFILE.md C2)

**Why.** Rule C2 says Firmus code must contain no process-name gating:
- no `sys.argv` anywhere;
- no `*_FORCE` override.

These files are tools, and none of them gates a kernel on the program name: each read its command-line arguments with
`sys.argv`. They are changed so that the rule holds without exceptions.

**How.** Each command line is now parsed with `argparse`, with the same positional arguments in the same order. The
only behaviour added is argparse's own:
- `-h`;
- an error and exit 2 when a positional argument is missing or an extra one is given (the old code ignored extras).

Nothing else in these files changed.

| file | validated sha256 (source) | shipped sha256 | change |
|---|---|---|---|
| `rht7/gen_ext.py` | `85cbb397f6c355ec49786da07d7c74444cabf9752772b3025564eb23f69d5b7b` (`local/l8b/rhtdev/gen_ext.py`) | `60f8d49acf73926cc4468fa12f2c83f28380c3cff5d1093020adb5b035f905ef` | `src, out = sys.argv[1], sys.argv[2]` becomes argparse positionals `src out`; `import sys` becomes `import argparse` |
| `rht7/gen_variants.py` | `1bea3e01ceb41433f72340353be079f2ee3c77bd77c9ad3626ae46265aba174d` (`rhtdev/gen_variants.py`) | `0496f9a0ac1cf627f4d5053c416f9ea25246cbce13b9c40aaee795f693c49f02` | same |
| `rht7/gen_norm.py` | `e1531b3e105061b2168644a378455d64569cdc302d0570b10d719e2f18dd17bc` (`p14dev/gen_norm.py`) | `0375ae8332de333e50af34b0148c06e39981d4f192a8429e5cddc54c69739511` | same |
| `rht7/gen_philox.py` | `00781813a003347ba56c8c073588afa2d41216158cb491e9d213146011519066` (`p14dev/gen_philox.py`) | `418e14683b58c9d09f9b3a1a8bdb51a03ff661fc81ca1cdca32ae8965b97d3d7` | same |
| `rht8/gen_variants8.py` | `b7d983e64b182ec19ee467293cfbb77c986542f0fdca357d6b06049e4543ce2e` (`rx26dev/gen_variants8.py`) | `6b241eed89f55341a7d9ed3c63fa8ec145bf9af326f54941461ae7651b318fa9` | same |
| `rht8/gen_common8.py` | `771fad0a4b65d5c8b288778298420daf59e35dac244288319f1b89ab53e3eb26` (`rx26dev/gen_common8.py`) | `1d4ef4b1553979fef4a5862fc655ff22c0537356d4c77ed195103c1f5cd76a70` | same |
| `rht8/gen_philox8.py` | `e447b206c4d29f9e1d07461713748bc908f8ea02645f1ceea1038b96abac0be6` (`rx26dev/gen_philox8.py`) | `2888425b00133bdb6dca248f58060f768b9aa3f7300b83ea35cd97a2613c41f2` | same |
| `w2d7/gen_w2d.py` | `06b7c35483b8a1bcff00aeaef1884de3a51dcd5d9f423a652f467a443199cf96` (`w2ddev/gen_w2d.py` = member of `ak_l8b_w2d7_src.tar.gz` `4cd52c90`) | `725ab54440e9eb4cb0609625b0395fc86851053bbc288b03f75f62201493b937` | same |
| `lmh/build_lmh.py` | `2771039debaa61068ce70597fc08cd6bdd5eae5dd27f3e7d3dcc350ae02069ad` (`lmh29dev/build_lmh.py` = the deployed lmh overlay's `src/`) | `0ff461e672cec6c6237a5c5e3e335a7e68a25a1aeffed30d58bb0c3390275379` | `src, build, ext = sys.argv[1:4]` becomes argparse positionals `src build ext`. Not used by the image build (`build_all.py` builds lmh with the same `load()` call); shipped as the build's record |
| `lmh/test_lmh.py` | `c5cedef10e77cdcd08fa0d3a84360d259bb26975f8828a269b741d8a2872ab63` (`lmh29dev/test_lmh.py` = the deployed lmh overlay's `src/`) | `b86fc8fcd5fc0b7682a5611a3ad597f593f0e16d437b15184780dfd5f620e7f8` | see the three bullets below |

The three changes to `lmh/test_lmh.py`:
1. **The `__main__` block.** `mode, d = sys.argv[1], sys.argv[2]` and `sys.argv[3] if len(sys.argv) > 3 else None` become argparse positionals: `mode d [pin_out]`. As before, `enum` writes `pin_out` when it is given, and any other mode runs `test`.
2. **`patch_test`.** It sets `AK_LMH=1, AK_LMH_DIR=d`, without `AK_LMH_FORCE=1`. The lmh patch module shipped in `firmus_ak` arms on `AK_LMH=1` alone, with no `argv` test and no FORCE override (gap B-4), so the override has nothing to bypass.
3. **`import argparse`** is added. `sys` stays, for `sys.path` and `sys.exit`.

Every function is byte-identical to the validated file. That includes `enum()`, which wrote the shipped `lmh_pin.json`,
and which `make_pin.py` calls.

**Generated headers still reproduce.** All seven generated-header checks of `build_manifest.json` ("regen_checks") were
run with the changed generators: `python3 -I`, an empty working directory, and TE `5b28d7aa`'s two source files as
input. Each regenerated file equals the shipped bytes. `build_all.py` repeats the checks at image build time, against
`/workspace/transformerengine`, and stops on any difference.

| regenerated | sha256 |
|---|---|
| `rht7/variants.cuh` (TE `row_cast_col_hadamard_transform_cast_fusion.cu`, then `gen_ext.py`, then `gen_variants.py`) | `cb94d8d15a983b14618e0f082e37f7cce575176318e83157d31048ec5423595a` |
| `rht7/norm_variant.cuh` | `05acdd66692f9bd869635db20d706aa6245421d6b8faca86f497eb8f0c0fa810` |
| `rht7/philox_variant.cuh` | `b5979b81143611873abaa9e1fd3c0fc9f685a199a0394757c1d58c33cff12457` |
| `rht8/variants8.cuh` | `c8fec0f94aca67fb59beed7c02dc24d90f96fb9756d756a888ff16b3ba4a9305` |
| `rht8/philox_variant8.cuh` | `42e5662721d2258070a675a0e859fb071c98cc00980721fd9ce7987624e9f9dd` |
| `rht8/ak_rht8_common.cuh` | `2a868cada0a08c7a4bb5548283fb084315050403c01ec6cd6d4cb704b6a0624e` |
| `w2d7/w2d_kernel.cuh` (TE `quantize_transpose_nvfp4.cuh`, then `gen_w2d.py`) | `84e960b75dfbd465140c1c4c913384c64642025bfefac7bd067dcddbe8ba91ee` |

## New files (not in the validated set)

- `lmh/make_pin.py`: the pin generator and checker (gap M-1). It wraps `test_lmh.py`'s unchanged `enum()`.
- `../build_all.py`, `../build_manifest.json`, `../EXPECTED_OUTPUTS.json`, `../SOURCES.sha256`: the image build's
  machinery.
