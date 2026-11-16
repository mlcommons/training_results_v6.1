# firmus_ak (L4): the eight patch modules against their validated L2x bytes

Written 2026-10-09 (Phase A5). L4 v3 (`nemo_l4_nvfp4_v3`, 2026-10-14) changes two files of this directory, both for the no-overlap mode `AK_FP4G_NOOVERLAP`: `ak_l8b_fp4g_patch.py` (8 more edits, listed under "ak_l8b_fp4g_patch, L4 v3") and `firmus_ak/run_start_gate.py` (one more gate row). Every other file is L4 v2's.

**What this directory is.** The eight auto_kernel patch modules of the validated stack L2x, installed as top-level modules by `pip install firmus/ak/pkg`, plus the package `firmus_ak`. The recipe's `pretrain.py` imports `firmus_ak` first.

**How the files were made.** `scratchpad/a5/make_pkg.py` made every file from its validated source:
- it sha256-checks each source against `llama31/L4_SOURCE_MANIFEST.tsv` (`sha256_validated`, also the deployed sha);
- it then applies only the listed edits, each an exact string replacement that must match exactly once.

The unified diffs are in [diffs/](diffs/).

**Edit classes:**

| class | what it changes |
|---|---|
| B-4 | arms on the `AK_*` switch alone: no `sys.argv` test, no `AK_LMH_FORCE`, no `.pth` loader (docstrings updated) |
| B-5 | every default directory is the in-image path the Dockerfile `ENV` sets (`/opt/firmus/ak/lib`, `/opt/firmus/ak/lmh`) |
| M-10 | `ak_l8b_wq`, `ak_l8b_rht6` and `ak_l8b_nvfp4b` (38f42c75) dropped, and `ak_l8b_nvfp4s` not built: their loaders are removed or raise; their calls route to TE's own path |
| C5 | fail loud, fail early (`docs/KERNEL_CODE_IN_DOCKERFILE.md` C5, gap M-14): an armed switch whose kernel cannot load, cannot arm or was never checked raises; it never runs silently on the stock path. Per-call shape routing (a shape the kernel does not cover goes to the stock path) is the validated design and stays |
| C2 / H2 | the two development `*_FORCE` overrides are no longer read from the environment |

**What did not change.** The arithmetic, the first-call checks, the ARMED / proof-line formats, the import hooks and every other byte.

## The eight modules

| module | validated source (repo `auto_kernel_llama31_8b`) | validated sha256 | L4 sha256 | edits |
|---|---|---|---|---|
| ak_l8b_fp4g_patch | local/l8b/fp4gather/ak_l8b_fp4g_patch.py | 92666d1133668b79dbaf7d771c41cd0a4a2b4f17006c96953f8e2b45dc2a39a8 | 5a2e302ab3ab678a492497feeebe8c98196f1110ee8f9e55ed3a69841ef4714a (L4 v2: 0c04037122da6b3b3b7383459c3870ebe348c552b3c299f4788b9cb1a30a2b1c) | 6 + 8 (v3) |
| ak_l8b_fp8cs_patch | local/l8b/overlays/fp8cs3/ak_l8b_fp8cs_patch.py | f54467d05ef03033a8f02ec164d9cc61a6fec3e648e0ee86179447bd13911f55 | 1aca679a3cf54db7947fa0fa81ff127feeab3b7bbcad50a89a8230e16354144f | 3 |
| ak_l8b_gsa | local/l8b/overlays/gsa/ak_l8b_gsa.py | cea81e440fcab15061783ff01dc17ae7c35dbaebaf9123accb42efbf410393df | 27e114fbc0e706ce09a4552088adc48bdc0b1637bc3c4e462687fe4dec8a6f8a | 2 |
| ak_l8b_lmh_patch | local/l8b/lmh29dev/ak_l8b_lmh_patch.py | 69d551d9b6a621938ba4a449c6fa86d1a5bafe4608bfc03103a241798d0fd730 | 2b32b7cec41dba73c7f45290f28c3f2261c8217463a074d8b85b4917b0789af3 | 9 |
| ak_l8b_nvfp4p_patch | local/l8b/ci27dev/nvfp4p21_dev/ak_l8b_nvfp4p_patch.py | 07ac727d6244dbd5b448b0fdced606b7eeabd9f5ed34b6e33efea20711cae1b1 | 4c012630ed6267a61818a4b81ed8463bdbe21eeaa60a310a6aa12d4074c9274c | 22 |
| ak_l8b_opswz | local/l8b/overlays/opswz/ak_l8b_opswz.py | e429fc49e1d38dfa7a18ac95315c53a4bdabaed009699b322fa99c11c3789414 | b49a4eace5c3311c06a333c57eb7ab030fba8fa267d34ca95fec585e2fc0ae56 | 2 |
| ak_l8b_projcol | local/l8b/overlays/projcol/ak_l8b_projcol.py | 0616a01caf559f8596e1346947caefbe1cfcc5b3b71e0e249b670accbbf8da84 | 17f0403a71ad567a01b6d65193c36fd993ad5b5e998f266c0e443674fd182010 | 2 |
| ak_l8b_smallk2_patch | local/l8b/overlays/smallk2/ak_l8b_smallk2_patch.py | a7184add69a11ba362879dd0d8e025d1f664fae449ed29810204d0a3161c01bb | 38549f51a521ca4815bd3fad6507c2a9872beecb4c6084fdb3d033cad9693c6f | 6 |

### ak_l8b_gsa, ak_l8b_projcol, ak_l8b_opswz
- **B-4:** `_ARMED` loses `and "pretrain" in os.path.basename(sys.argv[0] ...)`.
- **B-4:** the docstring now says "Loaded by `import firmus_ak` ...; armed by AK_*=1 alone".

### ak_l8b_fp8cs_patch
- **B-4:** the argv test is removed; the docstring is updated.
- **B-5:** the `AK_FP8CS_EXT_DIR` default `[campaign directory]/.../fp8cs2/ext` becomes `/opt/firmus/ak/lib`.

### ak_l8b_fp4g_patch
- **B-4:** the argv test is removed; the docstring is updated.
- **C5 (new, fail early):** an armed FP4G refuses at import a configuration in which NVFP4P's weight-amax update, the only caller of its `setup`, does not run (`AK_NVFP4P=1`, `AK_NVFP4P_WAMAX=1`, `AK_NVFP4P_WEIGHTS=amax`). Before, it silently never gathered in FP4.
- **B-5:** the `AK_FP4G_EXT_DIR` default becomes `/opt/firmus/ak/lib`.
- **C5:** `setup` with more than one model chunk raises. It used to print `NOT armed: ...` and return, leaving a stock gather.
- **C5:** `setup` finding no pure bucket group raises, listing the reasons. It used to set `active=False` and run a stock gather.

### ak_l8b_fp4g_patch, L4 v3 (`AK_FP4G_NOOVERLAP=1`, set in v3's config leaf)
The 8 edits below are the whole difference between L4 v2's file (`0c040371`) and v3's (`5a2e302a`). They move waits only: the same kernels run on the same buffers in the same order, so the arithmetic is unchanged. The tested reference is probe E6's import-time wrapper `e6_fix.py` (mode `nooverlap`), which probes E6 and E8 ran; these edits put the same waits inside this module.
- **Docstring:** a paragraph on the no-overlap mode.
- **The switch (C5, fail early):** `AK_FP4G_NOOVERLAP` must be `0` or `1`, and `1` without `AK_FP4G=1` raises at import.
- **`_STATE`:** three new keys: `patched` (the wrapper is on the class) and the two wait counts `nooverlap_fp4g` and `nooverlap_g0`.
- **`_NoopHandle`:** a handle whose `wait()` does nothing. `finish_param_sync` gets it for a gather that is already finished, and still dispatches the next group.
- **The stock group's wait:** for a group without FP4G, after the stock `start_param_sync`, when it was not forced, the group has no next gather (bucket group 0) and a handle is pending: the handle's `wait()` runs at once on the current stream.
- **The FP4G group's wait:** right after `_dispatch`, the handle's `wait()` runs at once (the side stream's all-gather wait, `post_ag3`, the norm unpack and the current stream's join), and the group keeps `_NOOP_HANDLE`.
- **`setup` (C5):** raises unless the wrapper is on the class and, with `overlap_param_gather` on, exactly one stock group has no next gather. Rank 0 then prints the ARMED line `[auto_kernel fp4g] no-overlap parameter gather ARMED (AK_FP4G_NOOVERLAP=1): ...`.
- **The import hook (C5):** with the switch on, a `param_and_grad_buffer` without `_ParamAndGradBucketGroup` raises; a successful wrap sets `patched`.

### ak_l8b_lmh_patch
- **B-4:**
  - `_ARMED` loses the argv test and `AK_LMH_FORCE`.
  - The docstring is updated.
- **B-5:** `_DIR` defaults to `/opt/firmus/ak/lmh`. It used to default to the module's own `ext/` sibling.
- **C5 changes in `_run`:**
  - A pinned problem first met under graph capture raises. It used to return stock.
  - An unpinned T still routes to Megatron's code; the pin lookup now comes before the capture test.
  - If the pinned algorithm is absent from cuBLASLt's heuristic list, `_run` raises. It used to print a "NOT ARMED ... stock torch matmul" line and run stock.
- **C5 changes to the dgrad copy:**
  - `_patch_layers` no longer swallows a failed `_install_dgrad` in `try/except Exception`. It used to print "dgrad NOT ARMED" and keep the stock dgrad.
  - `_install_dgrad` raises when the stock backward's source sha256 differs from `lmh_pin.json`'s `bwd_src_sha256`. It used to print "NOT ARMED" and return.
  - Both failures surface at import of `megatron.core.tensor_parallel.layers`, at process start.
  - `AK_LMH_DGRAD=0` still keeps the dgrad stock on purpose.
- The docstring is updated for all of the above.

### ak_l8b_smallk2_patch
- **B-4:** the argv test is removed; the docstring is updated.
- **C5:** a first delayed-scaling update inside graph capture raises. It used to set `learn=False` and leave stock updates for the whole run, with no line printed.
- **C5:** `_ds_arm` raises when the schedule cannot arm, for example on unmet preconditions. It used to print "NOT armed ...; stock updates kept". That print loses the "stock updates kept" clause and is followed by the raise.
- The docstring is updated: "and raises otherwise (AK_SMALLK2_DS=0 keeps TE's updates)".

### ak_l8b_nvfp4p_patch

**B-4**
- The argv test is removed.
- The docstrings ("Armed by AK_NVFP4P=1 alone (firmus_ak)" and `_fp4g`'s ".pth sorts before" note) are updated.

**B-5:** every remaining default directory is `/opt/firmus/ak/lib`:
- `_ext`, `_ext_norm` and `_ext_w` (`AK_NVFP4P_EXT_DIR`);
- `_ext_b` (`AK_NVFP4P_BX_EXT_DIR`);
- `_ext_r8` (`AK_NVFP4P_RHT8_EXT_DIR`);
- `_ext_r7` (`AK_NVFP4P_NORMCAST_EXT_DIR`);
- `_ext_2d` (`AK_NVFP4P_W2D_EXT_DIR`).

**M-10: ak_l8b_nvfp4s and ak_l8b_nvfp4b**
- `_ext_s` (`ak_l8b_nvfp4s`, the SR cast; not built) now only raises. Its `/home` default is gone.
- `_ext_b` keeps only the BX branch (`ak_l8b_nvfp4bx`). Without `AK_NVFP4P_BX=1` it raises, because `ak_l8b_nvfp4b` 38f42c75 is dropped.

**M-10: ak_l8b_wq**
- `_ext_wq` (`ak_l8b_wq`) is removed.
- `_wq_slot` returns None (TE's own in-place quantize) for a workspace `_w2d_ok` does not admit. In L2x all 10 workspaces were `ak_l8b_w2d7`'s ("our 2D cast into 10 of them").
- `_wq_into`'s `ak_l8b_wq` branch now raises. It is unreachable after `_wq_slot`'s routing.
- `_wq_check`'s print no longer imports `ak_l8b_wq`. The "TE's 2D cast" fragment reads `TE's 2D cast (tex.quantize)`; the proof regex covers it with `.*`.
- `_wamax_update`'s eager load (its first use is inside the graph capture) loads `ak_l8b_w2d7` instead of `ak_l8b_wq`.

**M-10: ak_l8b_rht6**
- `_ext_r` (`ak_l8b_rht6`) is removed.
- `_rhtcast` returns None (TE's cast) for a cast that is neither in `ak_l8b_rht8`'s table nor `ak_l8b_rht7`'s VP case. In L2x all 5 recipe casts are rht8's.
- `_rhtcast`'s rht6 call is replaced by an unreachable raise.
- `_rhtcast_module` returns rht7 for the non-rht8 case.

**C2 / H2:** `_RHTCAST_FORCE` and `_W2D_FORCE` (development overrides, empty in every L2x run) are constants `""`, no longer read from `AK_NVFP4P_RHTCAST_FORCE` / `AK_NVFP4P_W2D_FORCE`. Their comment is updated.

**C5 (new, fail early):** at import, an armed nvfp4p refuses any switch that would need a dropped extension:
- `AK_NVFP4P_SR=1`;
- `AK_NVFP4P_BWDNORM` or `AK_NVFP4P_RHTAMAX` without `AK_NVFP4P_BX=1`;
- `AK_NVFP4P_WQ=1` without `AK_NVFP4P_W2D=1`;
- `AK_NVFP4P_RHTCAST=1` without `AK_NVFP4P_RHT8=1`.

Each of these would otherwise run silently on TE's path.

**Left as validated (shape routing, not load or check failures):**
- `rhtcast_capture_fallback` / `normcast_capture_fallback`: a (rows, width, SR) key first seen inside a graph capture runs TE's cast, which is byte-correct.
- `_check` returns None under capture (ours runs; the key was checked in the warm-up).
- Every `_tensor_ok` / `_quantizer_ok` / `_w2d_ok` route.

## firmus_ak (new)

- **`__init__.py`:** `from . import banner` first. It then imports the eight modules in the validated `.pth` order (fp4g, fp8cs, gsa, lmh, nvfp4p, opswz, projcol, smallk2), then `from . import run_start_gate`.
- **`banner.py`:** the coordinator's file, copied unchanged (sha256 `652b404c…`). It re-hashes `${FIRMUS_AK_PREFIX}` against `/FIRMUS_AK_BUILD_SHA256.txt` and the patched upstream files against `/FIRMUS_STACK_SHA256.txt`, and raises on any difference.
- **`run_start_gate.py` (new, C5 fail early):** wraps the recipe's `callback_warmup.WarmupCallback.on_data_init_start`. When the warm-up returns, before `MLPerfLoggingCallback` logs `run_start`, every armed part must report its first-call check done, or the rank raises. The job then ends before `run_start`, which is not an attempt.
  - **Parts covered:**
    - the 8 modules, from their own `_STATE` flags;
    - TE rope.py X12, from `_AK_X12_CHECKED` fwd and bwd;
    - TE dot_product_attention/utils.py QKV pack;
    - the recipe's custom_embedding EMB and EMB-MG;
    - mcore clip_grads.py X9 fold.

    A patch whose switch is on but whose module was never imported counts as unchecked.
  - **Output:** rank 0 prints `[auto_kernel gate] warm-up done, before run_start: N armed Firmus parts (...); every armed part checked its first call: True`.
  - **Off switch:** `FIRMUS_AK_RUN_START_GATE=0` disables it; the L4 config does not set it.
  - **L4 v3:** with `AK_FP4G_NOOVERLAP=1` the gate has one more part, `fp4g no-overlap`. It is checked when both wait counts are above 0: at least one FP4G gather finished at its dispatch and at least one bucket-group-0 gather waited for at its dispatch during the warm-up.
  - **Status:** addition beyond the A5 list. Validate it in the Phase C smoke, and add its line to `proofs_l4_pin.txt`.
- **`check_install.py`:** stdlib only. It is run as a script, so it never imports `firmus_ak`, whose banner needs build records that do not exist yet:

  ```bash
  python3 /workspace/llm/firmus/ak/pkg/firmus_ak/check_install.py --prefix P
  ```

  It locates the installed package and modules with `find_spec`, which does not execute them. It fails on any of:
  - a Firmus `.pth` in site-packages;
  - any of the 11 `AK_*_DIR` variables unset, outside P, or not `P/<env_dirs[var]>`;
  - an extension not in its `env_dir`;
  - any of EXPECTED_OUTPUTS.json's `files` missing or empty;
  - a cluster path in firmus_ak or the 8 modules.

  It reads `P/EXPECTED_OUTPUTS.json` in the build fork's schema (`firmus-ak-expected-outputs/1`). A missing file is a failure; there is no packaged fallback.
- **`pyproject.toml`:** setuptools, `firmus-ak 6.1.0+l4`, the 8 py-modules plus the `firmus_ak` package. No `.pth`, no data files.

## Effect on the proof lines (Phase C: `proofs_l4_pin.txt`)

- `[auto_kernel smallk2] embedding gathered ...` prints the patch file's own sha256. It was `a7184add69a11ba3`; it is now `38549f51a521ca48`.
- `nvfp4p ... module weight casts ...`: only the fragment after "our 2D cast into 10 of them (...)," changes, and the regex covers it with `.*`.
- The new gate line `[auto_kernel gate] ... every armed part checked its first call: True$` should become a proof line.
- L4 v3: the new line `[auto_kernel fp4g] no-overlap parameter gather ARMED (AK_FP4G_NOOVERLAP=1): ...` is a proof line of v3's list.
- Every module sha printed from a `.so` changes with the Dockerfile build (Phase C replaces them).
