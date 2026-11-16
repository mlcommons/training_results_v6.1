"""auto_kernel NVFP4P (Llama 3.1 8B campaign, 2026-10-02): SwiGLU / dSwiGLU that hand TE's NVFP4 quantize its amaxes.

Version 21 (2026-10-08) = version 20 plus AK_NVFP4P_BX=1: the norm backward through ak_l8b_nvfp4bx
(AK_NVFP4P_BX_EXT_DIR), the deployed ak_l8b_nvfp4b.cu with one more kernel, AK_NVFP4P_BWDNORM_VER=7 --
rmsnorm_bwd_add_v4: version 2's grid, rows, staging and arithmetic with fewer instructions per element and the
same bytes ((bf16(gamma) + shift) once per CTA in shared memory instead of per row, dx's and the sum's bf16 by
the packed round-to-nearest convert, the pre-RHT amax by TE's packed max.xorsign.abs.bf16x2), then version 6's
post-RHT amax pass.  The norm backward's first-call check (against tex.rmsnorm_bwd + torch's add) covers it; its
ARMED line names the module.  The RHT-amax quantize also imports ak_l8b_nvfp4bx (its rht_amax kernels are
ak_l8b_nvfp4b's, AK_NVFP4P_RHTAMAX_VER selects among them; default 2, the deployed one).

Version 20 (2026-10-08) = version 18 plus AK_NVFP4P_RHT8=1: the recipe's five RHT casts through ak_l8b_rht8
(AK_NVFP4P_RHT8_EXT_DIR) -- versions 12 / 14's variants of TE's fused row + column-RHT cast with two more changes in
the epilogue warps that leave every byte as it was: (a) each 16-element block's reciprocal scale,
min(1 / (float(e4m3 scale) * decode), FLT_MAX), read from a 256-entry table that each CTA builds once with exactly those
operations (it depends only on the scale's byte and the call's decode scale); (b) the stochastic-rounding draws of a
k-tile with the multiplies they share done once -- while the call's Philox offset x leaves x + v unwrapped for every
draw v of a k-tile (x <= 0xFFFFFFF8, the same in every thread), round 1's M1 * z and round 2's M0 * c1.x are the same
for every draw; the epilogue warps' work loop is instantiated for both cases and the launch takes one, so a call at a
2^32 boundary of the offset takes the variant's own 128-bit counter add.  Table (width, SR) -> variant / tokens per
tile: 4096 RN 326/256, 14336 RN 359/256 (version 12's 70 / 103 + the table), 4096 SR 847/512, 6144 SR 847/256
(version 14's 79 + both), 28672 SR 867/1024 (99 + both).  Other widths keep version 12 / 14's route.  Version 12's
first-call check per (rows, width, SR) covers them (every buffer, the swizzle flag, the class, the generator offset
against TE's cast from the same RNG state); its ARMED line names the module.

Version 18 (2026-10-08) = version 14 plus three held small items, each E0.  (1) nvfp4p17's extension set (no patch
change; selected by its mounts: AK_NVFP4P_KVER=8, the dSwiGLU v8 build of ak_l8b_nvfp4p; AK_NVFP4P_BWDNORM_VER=6, the
norm backward's post-RHT pass in reversed row order).  (2) Version 13's 2D weight cast (AK_NVFP4P_W2D, ak_l8b_w2d7),
merged three-way (base version 12): with the FP4G overlay it covers the 19 NVFP4 weights the parameter gather leaves --
the gathered weights' quantize returns their gathered storage first (fp4g.weight_quantize), so they never reach it.
(3) AK_NVFP4P_DGFOLD=1: the norm weight's gradient accumulation folded into the fused norm backward's finalize -- when
the norm weight (the gamma tensor TE passes, or its base) carries a bf16 [4096] main_grad, rmsnorm_bwd_add's finalize
also writes main_grad = bf16(float(main_grad) + float(dgamma)) (the DDP hook's main_grad.add_(param.grad), torch's
CUDAFunctor_add<BFloat16>) and the patch sets grad_added_to_main_grad, so the hook skips its add (TE's own mechanism
for fused-wgrad weights); a norm backward that falls back to TE's path clears the flag of a weight folded before, so
the hook adds as shipped.  The first eager call per call site compares main_grad against main_grad + dgamma by torch's
add (and dgamma against TE's) as bytes; a mismatch raises.

Version 14 (2026-10-07) = version 13g plus two paths of ak_l8b_rht7 (AK_NVFP4P_NORMCAST_EXT_DIR).
(1) AK_NVFP4P_NORMCAST=1: the training norms' forward (the recipe's quantizer with row-wise and column-wise usage,
optimize_for_gemm, TE's exact math, rows % 1024 == 0; evaluation keeps version 13g's path).  Mode "apply7" (the default):
rmsnorm_apply_amax7 is version 13g's apply pass (ak_l8b_nvfp4n apply v6: z = bf16(gamma * (rs * x)) written, with its
pre/post-RHT amaxes, the same arithmetic and amax epilogue) in persistent CTAs whose warps prefetch their next block, the
16-row blocks in reverse order and the work split into 4 CTAs per resident slot (AK_NVFP4P_NORMCAST_APPLY_VER=7; 5 is
one CTA per slot, faster on an idle GPU but not when SMs are held by a concurrent kernel): the rows the residual add
wrote last are read while still in L2, and version 12's cast, which walks forward, finds the first rows of z there.  Mode "fused": z is never
written -- rmsnorm_amax takes the two amaxes only and rht_cast_norm is version 12's cast whose row-quant warps form z
from each TMA-loaded stage of x (written back into the stage for the column-wise RHT MMA, one mbarrier per stage);
exact, but the cast becomes issue-bound and the pair is no faster than "apply7".  rsigma comes from the residual add
(version 7) or from rmsnorm_stats (TE's, as rmsnorm_fwd_amax computes it; layer 0's norm).  The first eager call per
(call site, after the add or not) runs tex.rmsnorm_fwd with the quantizer and the new path on the same input: every
buffer, the swizzle flag, the class, rsigma; rank 0 prints one ARMED line, a mismatch raises; a site first seen inside
a CUDA-graph capture keeps version 13g's path.
(2) AK_NVFP4P_RHTCAST_VP=1: version 12's stochastic-rounding casts (variants 79 and 99) through rht_cast_vp, which hands
the generator's Philox state (the same philox_cuda_state(1024) per call) to the cast by value and unpacks it in the
cast's epilogue warps, instead of the one-thread unpack kernel version 12's binding launches in front of every such
cast.  Version 12's first-call check per (rows, width, SR) covers it (bytes and generator offset against TE's cast).

Version 13g (2026-10-07) = version 12 plus the hooks of the FP4G overlay (ak_l8b_fp4g_patch, the NVFP4 parameter gather;
nothing changes without it): the tex.quantize wrapper first asks fp4g.weight_quantize (an FP4-gathered weight's quantize
returns its gathered storage); the first weight-amax update runs fp4g.setup (it appends the cut tile-rows' slots to the
weight-amax buffer) and every update runs fp4g.before_allreduce (this rank's partial tile amaxes into those slots) before
the max-all-reduce; the step hook runs fp4g.check before the optimizer writes; the weight-amax and workspace checks skip
the FP4-gathered weights (their bf16 copy outside this rank's shard is stale; fp4g.check covers them).

Stock (TE 2.20, op fuser `ops/basic/swiglu.py`): `tex.swiglu(x, q)` / `tex.dswiglu(dy, x, q)` with an NVFP4 quantizer
(RHT, post-RHT amax) write the BF16 result, then quantize it: `HadamardAmaxTmaKernel` reads it once for the two amaxes,
`row_col_rht_gemm_device` reads it again for the cast (PROF-L0: 76 + 110 us forward, 141 + 322 us backward per layer).
Here `ak_l8b_nvfp4p.swiglu_amax` / `.dswiglu_amax` compute the same BF16 result with TE's arithmetic and both amaxes in
the same pass, and `tex.nvfp4_quantize_with_amax` runs TE's own cast with them: the amax pass is gone.

Armed by AK_NVFP4P=1 alone (firmus_ak).  Only the recipe's quantizer configuration takes the new path (NVFP4,
RHT with post-RHT amax, 1D, no 4over6 / row scaling / amax reduction, BF16 contiguous input, rows % 32 == 0 and
N % 256 == 0); anything else calls TE unchanged.  The first eager call of each (op, usage) pair is checked against TE's
own call on the same inputs, every buffer of the quantized result as bytes (row and column data, scales, amaxes), with
the CUDA RNG state replayed for the stochastic-rounding quantize; rank 0 prints one ARMED line per pair, a mismatch
raises.  The RMSNorm forward (hidden 4096; ak_l8b_nvfp4n, TE's per-row arithmetic, no fast math) takes the same route
unless AK_NVFP4P_NORM=0; its check also compares rsigma, once per call site (the op fuser's RMSNorm, LayerNormLinear's).
Weights are accepted as plain tensors or as nn.Parameters holding one, never as quantized tensors.  AK_NVFP4P_EXT_DIR
names the directory holding the built extension (ak_l8b_nvfp4p*.so).

With AK_NVFP4P_SR=1 the stochastic-rounding casts of the gradients are ours too (ak_l8b_nvfp4s, from
AK_NVFP4P_SR_EXT_DIR): the row-wise + column-wise RHT cast reproduces TE's fused row_col_rht_gemm_device byte for byte,
its Philox4x32-3 draws included (one philox_cuda_state(1024) per call from the default CUDA generator, as TE).  It
replaces tex.nvfp4_quantize_with_amax after the fused dSwiGLU, and tex.quantize(dY, q) for the recipe's other gradient
quantizers (SR, RHT, row + column, T % 1024 == 0, H % 256 == 0), whose two amaxes then come from ak_l8b_nvfp4s.rht_amax.
Each path's first eager call per shape is checked against TE with the RNG state replayed, every buffer as bytes.

With AK_NVFP4P_BWDNORM=1 (version 6) each RMSNorm backward of a transformer layer also does the residual-gradient add
that autograd would do right after it, and takes the two amaxes of the sum for the next linear's gradient quantize
(ak_l8b_nvfp4b: TE's tuned kernel reproduced with its grid and row order, dgamma through TE's finalize, then
bf16(float(dx_bf16) + float(g_res)) as autograd's add).  Megatron's TransformerLayer bias-dropout-add steps pass the
residual through a tap (_ResidualTap) whose backward hands its gradient to the norm backward that consumes the same
tensor next, instead of returning it to autograd; the norm backward returns the sum.  The fused sum is registered, and
the next tex.quantize of exactly that tensor (the same object or a view of it, unmodified) runs TE's cast with those
amaxes (tex.nvfp4_quantize_with_amax).  First eager calls: the sum and dgamma against TE's rmsnorm_bwd followed by
torch's add, per call site; the registered-amax quantize against tex.quantize of the same tensor, RNG replayed.  A
residual gradient left unconsumed raises.

With AK_NVFP4P_RHTAMAX=1 (version 8) every other tex.quantize(t, q) with the recipe's 1D RHT quantizer and a
column-wise usage on a BF16 tensor (rows % 32, N % 256: the attention output feeding linear_proj, the QKV gradient from
the RoPE backward; a row-only quantize needs no post-RHT amax and stays TE's) takes its two
amaxes from ak_l8b_nvfp4b.rht_amax (TE's HadamardAmaxTmaKernel contract with its own amax instructions) and runs TE's
cast with them (tex.nvfp4_quantize_with_amax).  First eager call per (shape, usage, SR): against tex.quantize, every
buffer, the RNG replayed.

With AK_NVFP4P_WAMAX=1 (version 9, needs AK_NVFP4P_WEIGHTS=amax) the weights' amaxes come from the distributed
optimizer's shards instead of a pass over every full weight in the forward: after each _copy_main_params_to_model_params
(the only writer of the bf16 weights; X9's fused step runs inside it) ak_l8b_nvfp4w.multi_amax_bf16 takes max |w| over
this rank's shard of every 2D bf16 parameter (an eighth of the weights) and an all-reduce (MAX) over the shards' group
completes it; the weight quantizes then run TE's 2D cast with those amaxes.  The forward before the first optimizer step
uses the full-weight pass.  At the second optimizer step, before it writes, every amax of the first is checked against
amax_bf16 of the gathered full weight (rank 0 prints the result, a mismatch raises).

With AK_NVFP4P_ADDNORM=1 (version 7, needs version 6) the bias-dropout-add itself is ours when it is a plain residual
add (no bias, dropout probability 0): h = bf16(float(r) + float(o)), as the compiled add, written by
ak_l8b_nvfp4n.rmsnorm_add_stats together with TE's rsigma of h; the RMSNorm forward that consumes h next runs only the
apply pass (rmsnorm_apply_amax) with that rsigma.  The add's backward gives the block output its gradient and hands the
residual's to the norm backward (version 6).  First eager calls: h against the original bias-dropout-add, and the
norm's quantized output and rsigma against tex.rmsnorm_fwd.  AK_NVFP4P_ADDSTATS_VER (default 1) selects the
rmsnorm_add_stats kernel version (2026-10-06: 2..7 the pipelined v2 variants; same arithmetic, same order).

With AK_NVFP4P_WQ=1 (needs AK_NVFP4P_WAMAX=1) the module weights' cached workspaces (QKV's and proj's: TE's
quantize_weight updates them in place each step, tex.quantize(w, q, workspace, None), which is TE's full-weight amax pass
and then its 2D cast) take the weight's amax from the optimizer-shard buffer too: ak_l8b_wq.quantize_2d_into runs TE's
own cast (nvte_quantize_v2, the tensor wrapper and quantization config TE builds for that workspace) without the amax
pass, and the workspace's amaxes alias the buffer's slot, as tex.nvfp4_quantize_with_amax aliases them.  Only for
128-aligned weights (no scale padding) whose workspace already holds exactly the buffers its usage needs, with no no-op
flag, when TE would not swizzle after the cast and fast math is off; anything else stays TE's.  These calls happen inside
CUDA-graph capture, so each workspace seen is registered and checked eagerly at the next optimizer step, before it
writes (warm-up step 1 for the training graph): a scratch workspace filled by our cast with the slot's amax against
tex.quantize of the same weight with a copy of the module's quantizer, every buffer and the swizzle flag as bytes; rank 0
prints one ARMED line, a mismatch raises.

With AK_NVFP4P_RHTCAST=1 (version 12) the RHT cast after the patch's amax kernels -- every tex.nvfp4_quantize_with_amax
the patch runs (SwiGLU, dSwiGLU, the norm forwards, the registered residual-gradient quantize, the RHT-amax quantize) --
is ours: ak_l8b_rht6 (AK_NVFP4P_RHTCAST_EXT_DIR) is TE's fused kernel (row_col_rht_gemm_device and its launcher,
hadamard_transform/row_cast_col_hadamard_transform_cast_fusion.cu) copied verbatim and built as TE builds that file, with
schedule variants that change no result (a packed f32x2 multiply, the column copy's bf16 round trip in one conversion,
each Philox draw from its counter directly, 256- or 512-token tiles with the stochastic-rounding subsequences of TE's
1024-token tiling, the block amax as a tree of TE's max instruction); one variant per (width, SR), the fastest of the
dev tests.  Only for TE's fused-kernel case with GEMM-swizzled scales: the recipe's 1D RHT quantizer with row-wise and
column-wise usage and optimize_for_gemm, TE's exact math, a contiguous bf16 input with width % 128 == 0 and
rows % 1024 == 0; the output is the quantizer's own create_tensor with the amaxes set to the given tensors, as TE's;
anything else (the 2D weight casts included) stays TE's.  The first eager call per (rows, width, SR) runs TE's cast and
ours on the same input with the CUDA RNG state replayed: every buffer, the swizzle flag, the class and the generator
offset after the call; rank 0 prints one ARMED line, a mismatch raises.  A shape first seen inside a CUDA-graph capture
gets TE's cast there.

With AK_NVFP4P_W2D=1 (version 13) the 2D weight casts are ours: ak_l8b_w2d7 (AK_NVFP4P_W2D_EXT_DIR) computes what TE's
quantize_transpose_nvfp4_2D_kernel writes for a bf16 weight with TE's own device code on the same operands -- the 16 x 16
block amax with TE's max instruction from 0 (bf16 max |w| of the block, NaN skipped), TE's global and block scale
arithmetic per direction, TE's bf16 -> e2m1 multiply-convert on the same four elements per call (four row neighbours for
the row-wise data, four column neighbours for the column-wise data) -- on a register-tile schedule: each thread a
16 x 8 piece of the weight straight from global memory, two lanes per 16 x 16 block, the row-wise data in 4-byte and the
column-wise data in 8-byte pieces, the scales staged in shared memory and written as 16-byte pieces of the
GEMM-swizzled layout (or as rows of the compact one); one variant per weight shape, the fastest of the dev tests.  Only
for the recipe's 2D weight quantizer (no RHT, round to nearest, no 4over6 / row scaling, E4M3 max 448) with row-wise
usage (column-wise optional), TE's exact math, a contiguous 2D bf16 weight whose rows and cols are multiples of 128;
anything else stays TE's.  The two call sites: the op fuser's weight quantize (fc1 / fc2, tex.quantize(w, q) through the
weight-amax path) is ours only for a (shape, usage, layout) whose first eager call passed the weight-amax check against
tex.quantize (its ARMED line names the module), TE's cast for one first seen inside a capture; the module workspaces
(QKV / proj, AK_NVFP4P_WQ) are ours inside the capture and each is checked eagerly at the next optimizer step, before it
writes, as in version 11 (its ARMED line names the module).
"""
import os
import sys

_ARMED = os.environ.get("AK_NVFP4P", "0") == "1"
_STATE = {"checked": set(), "fast_calls": 0, "ext": None}
# kernel version of the SwiGLU pair (job 19966: v1 446.8 / 243.5 us, v2 497.0 / 242.2 us for dSwiGLU / SwiGLU; v4 the
# padded staging; v6 the epilogue of ak_l8b_rht_epi.cuh; v7 = dSwiGLU in warp-owned blocks, SwiGLU as v6)
_KVER = int(os.environ.get("AK_NVFP4P_KVER", "1"))
# RMSNorm kernel version (job 19983: v4 = stats + padded apply, 117.4 against stock 123.2 us with the cast; v2 130.0;
# v5 = v4 with the epilogue of ak_l8b_rht_epi.cuh)
_NVER = int(os.environ.get("AK_NVFP4P_NORM_VER", "4"))
# 2D weight quantize kernel version (ak_l8b_nvfp4w: 1 = tile-wide loops, 2 = one thread per tile row / column)
_WVER = int(os.environ.get("AK_NVFP4P_WEIGHTS_VER", "2"))
_BUFS = ("_rowwise_data", "_rowwise_scale_inv", "_columnwise_data", "_columnwise_scale_inv", "_amax_rowwise",
         "_amax_columnwise")


def _ext():
    if _STATE["ext"] is None:
        d = os.environ.get("AK_NVFP4P_EXT_DIR", "/opt/firmus/ak/lib")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ak_l8b_nvfp4p
        _STATE["ext"] = ak_l8b_nvfp4p
    return _STATE["ext"]


def _ext_norm():
    if _STATE.get("ext_norm") is None:
        d = os.environ.get("AK_NVFP4P_EXT_DIR", "/opt/firmus/ak/lib")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ak_l8b_nvfp4n
        _STATE["ext_norm"] = ak_l8b_nvfp4n
    return _STATE["ext_norm"]


def _ext_w():
    if _STATE.get("ext_w") is None:
        d = os.environ.get("AK_NVFP4P_EXT_DIR", "/opt/firmus/ak/lib")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ak_l8b_nvfp4w
        _STATE["ext_w"] = ak_l8b_nvfp4w
    return _STATE["ext_w"]


def _ext_s():
    # L4: ak_l8b_nvfp4s (AK_NVFP4P_SR) is not part of this build; the import-time check below refuses the switch
    raise RuntimeError("[auto_kernel nvfp4p] ak_l8b_nvfp4s is not part of this build (AK_NVFP4P_SR)")


_BX_ON = os.environ.get("AK_NVFP4P_BX", "0") == "1"           # version 21: ak_l8b_nvfp4bx in place of ak_l8b_nvfp4b
_RAVER = int(os.environ.get("AK_NVFP4P_RHTAMAX_VER", "2"))     # version 21: 3 = rht_amax_kernel_r2 (persistent)


def _ext_b():
    if _STATE.get("ext_b") is None:
        if not _BX_ON:  # L4: ak_l8b_nvfp4b is not part of this build (ak_l8b_nvfp4bx carries its kernels)
            raise RuntimeError("[auto_kernel nvfp4p] ak_l8b_nvfp4b is not part of this build: AK_NVFP4P_BX=1 is "
                               "required")
        d = os.environ.get("AK_NVFP4P_BX_EXT_DIR", "/opt/firmus/ak/lib")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ak_l8b_nvfp4bx
        _STATE["ext_b"] = ak_l8b_nvfp4bx
    return _STATE["ext_b"]


# version 6: the RMSNorm backward + residual add + amaxes (ak_l8b_nvfp4b kernel version; TE's CTAs per SM for its tuned
# rmsnorm_bwd kernel -- its NVRTC build has 167 registers, 3 CTAs of 128 threads per SM, PROF-L0 grid 408)
_BWD_ON = os.environ.get("AK_NVFP4P_BWDNORM", "0") == "1"
_BVER = int(os.environ.get("AK_NVFP4P_BWDNORM_VER", "2"))   # 3: v2 + the post-RHT amax with ak_l8b_rht_epi.cuh
_BCPS = int(os.environ.get("AK_NVFP4P_BWDNORM_CPS", "3"))
_DGFOLD_ON = os.environ.get("AK_NVFP4P_DGFOLD", "0") == "1"      # version 18: the norm weights' gradient add folded
_STATE_DG = {"dg_folded": set(), "dg_fold_calls": 0, "dg_unfolded": 0}
_STATE.update({"bwd_enabled": _BWD_ON, "res_stash": None, "bwd_reg": None, "taps": 0, "bwd_fused": 0, "bwd_consumed": 0,
               "bwd_stock": 0})
_STATE.update(_STATE_DG)                                          # version 18
_TAP = []
# version 7: the residual add + the norm statistics of its result (needs version 6's tap)
_ADDNORM_ON = os.environ.get("AK_NVFP4P_ADDNORM", "0") == "1" and _BWD_ON
_SVER = int(os.environ.get("AK_NVFP4P_ADDSTATS_VER", "1"))   # ak_l8b_nvfp4n.rmsnorm_add_stats kernel version
_STATE.update({"addnorm_enabled": _ADDNORM_ON, "addnorm_reg": None, "addnorm_adds": 0, "addnorm_norms": 0})
_ADDF = []


def _tap_summary():
    """at the first forward after the first backward with taps: what that backward did (rank 0)"""
    if _STATE["taps"] and not _STATE.get("tap_summary") and _rank() == 0:
        _STATE["tap_summary"] = True
        print("[auto_kernel nvfp4p] residual taps ARMED: first backward: {} residual gradients stashed, {} fused norm "
              "backwards (+ add + amaxes), {} consumed by the next gradient quantize, {} stock norm backwards with a "
              "residual".format(_STATE["taps"], _STATE["bwd_fused"], _STATE["bwd_consumed"], _STATE["bwd_stock"]),
              flush=True)


def residual_tap(t):
    """the residual branch of a transformer layer's bias-dropout-add, applied after the block's forward: its gradient
    goes to the norm backward that consumes the same tensor next (version 6), not back through autograd"""
    if not _STATE["bwd_enabled"] or not _TAP:
        return t
    import torch
    if (not torch.is_grad_enabled() or type(t) is not torch.Tensor or not t.requires_grad or not t.is_cuda
            or t.dtype != torch.bfloat16 or not t.is_contiguous() or t.size(-1) != 4096):
        return t
    if _STATE["res_stash"] is not None:
        raise RuntimeError("[auto_kernel nvfp4p] a residual gradient from the previous backward was never consumed")
    _tap_summary()
    return _TAP[0].apply(t)


_SR_ON = os.environ.get("AK_NVFP4P_SR", "0") == "1"
_RHTAMAX_ON = os.environ.get("AK_NVFP4P_RHTAMAX", "0") == "1"
_WAMAX_ON = os.environ.get("AK_NVFP4P_WAMAX", "0") == "1" and os.environ.get("AK_NVFP4P_WEIGHTS", "0") == "amax"
_STATE.update({"wamax_buf": None, "wamax_slot": {}, "wamax_valid": False, "wamax_calls": 0, "wamax_used": 0})
_WAMAX_CHUNK = 1 << 18
# the module weights' in-place workspace updates with the shard amaxes (AK_NVFP4P_WQ; needs WAMAX)
_WQ_ON = os.environ.get("AK_NVFP4P_WQ", "0") == "1" and _WAMAX_ON
_STATE.update({"wq_reg": {}, "wq_checked": set(), "wq_calls": 0})


def _env_off(name):
    v = os.environ.get(name, "")
    return v in ("", "0")


def _fp4g():
    """version 13g: the FP4G overlay's module when it is armed (firmus_ak imports it before this patch), else None"""
    m = sys.modules.get("ak_l8b_fp4g_patch")
    return m if (m is not None and getattr(m, "_ARMED", False)) else None


def _wq_swizzled(q, n, k):
    """TE's nvfp4_emits_gemm_swizzled_scales for the 2D, non-RHT, not row-scaled, 4over6-free weight quantizer"""
    import torch
    cc = torch.cuda.get_device_capability()
    sm = cc[0] * 10 + cc[1]
    return bool(getattr(q, "optimize_for_gemm", False)) and n % 128 == 0 and k % 128 == 0 and 100 <= sm <= 110


def _wq_into(tensor, q, output, slot, row, col):
    """TE's 2D cast of `tensor` into the workspace `output` with the slot's amax (no amax pass); the amaxes alias the slot"""
    import torch
    a = _STATE["wamax_buf"][slot:slot + 1]
    n, k = tensor.shape
    swz = _wq_swizzled(q, n, k)
    e = _STATE.get("wq_empty")
    if e is None or e.device != tensor.device:
        e = _STATE["wq_empty"] = torch.empty(0, dtype=torch.uint8, device=tensor.device)
    if _w2d_ok(tensor, q, row, col):
        _w2d_cast(tensor, output._rowwise_data, output._rowwise_scale_inv, output._columnwise_data if col else e,
                  output._columnwise_scale_inv if col else e, a, swz)
    else:  # L4: ak_l8b_wq is not part of this build; _wq_slot admits only the weights ak_l8b_w2d7 covers
        raise RuntimeError("[auto_kernel nvfp4p] a module weight workspace outside ak_l8b_w2d7's cases reached the "
                           "in-place cast ({} row {} col {})".format(list(tensor.shape), row, col))
    if row:
        output._amax_rowwise = a
    if col:
        output._amax_columnwise = a
    output._with_gemm_swizzled_scales = swz
    for nm, v in (("_row_scaled_nvfp4", False), ("_nvfp4_use_4over6", False), ("_nvfp4_e4m3_max", int(q.nvfp4_e4m3_max))):
        if hasattr(output, nm):
            setattr(output, nm, v)
    return output


def _wq_check():
    """every registered workspace path not yet checked: our cast with the slot's amax into a scratch workspace against
    tex.quantize of the same weight with a copy of the module's quantizer (eager, before the optimizer writes)"""
    import torch
    import transformer_engine_torch as tex
    F = _fp4g()
    todo = [(key, v) for key, v in _STATE["wq_reg"].items() if key not in _STATE["wq_checked"]
            and not (F is not None and F.is_fp4_weight(v[0]))]
    if not todo:
        return
    orig_quantize = _STATE["orig_quantize"]
    shapes, bad, ours = {}, [], 0
    for key, (w, row, col, ofg, q) in todo:
        _STATE["wq_checked"].add(key)
        q2 = q.copy()
        q2.set_usage(rowwise=row, columnwise=col)
        q2.optimize_for_gemm = ofg
        ours += bool(_w2d_ok(w, q2, row, col))
        slot = _STATE["wamax_slot"].get((w.data_ptr(), w.numel()))
        ref = orig_quantize(w, q2)
        new = tex.create_empty_quantized_tensor(q2, list(w.shape), w.dtype, w.device, False)
        _wq_into(w, q2, new, slot, row, col)
        ok = _compare(ref, new)
        ok["swizzled_flag"] = (getattr(ref, "_with_gemm_swizzled_scales", None)
                               == getattr(new, "_with_gemm_swizzled_scales", None))
        ok["type"] = type(ref) is type(new)
        tag = "{} row {} col {} swizzled {}".format(list(w.shape), row, col, bool(new._with_gemm_swizzled_scales))
        shapes[tag] = shapes.get(tag, 0) + 1
        if not all(ok.values()):
            bad.append((tag, {kk.lstrip("_"): vv for kk, vv in ok.items() if not vv}))
    good = not bad
    if _rank() == 0:
        te_cast = "TE's 2D cast (tex.quantize)"   # L4: ak_l8b_wq dropped; such workspaces stay on TE's own path
        if ours:
            cast = "our 2D cast into {} of them ({}; variants {}), {} into the other {}; no amax pass".format(
                ours, _w2d_desc(), sorted({_w2d_pick(*w.shape) for _, (w, _r, _c, _o, _q) in todo}), te_cast,
                len(todo) - ours)
        else:
            cast = "{} into each workspace, no amax pass".format(te_cast)
        print("[auto_kernel nvfp4p] module weight casts with the optimizer-shard amaxes ARMED: {} workspaces ({}): {}; "
              "byte-exact v tex.quantize {}: {}".format(len(todo), "; ".join("{} x{}".format(t, c) for t, c in
                                                                              sorted(shapes.items())),
                                                        cast, bad[:2] if bad else "every buffer", good), flush=True)
    if not good:
        raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the module weight casts: {}".format(bad[:4]))


def _wamax_build(opt):
    """the chunk table of this rank's shards of every 2D bf16 parameter, slots in one rank-independent order"""
    import torch
    params = [p for chunk in opt.model_chunks for p in chunk.parameters()
              if p.dim() == 2 and p.dtype == torch.bfloat16]
    slot_of = {id(p): i for i, p in enumerate(params)}
    ptrs, lens, slots, group = [], [], [], None
    for shard_main_group, model_group in zip(opt.shard_fp32_from_float16_groups, opt.model_float16_groups):
        for shard_main_param, model_param in zip(shard_main_group, model_group):
            if id(model_param) not in slot_of:
                continue
            world_range = opt._get_model_param_range_map(model_param)["gbuf_world_in_bucket"]
            gbuf_index, _, bucket_id = opt.model_param_gbuf_map[model_param]
            buf = opt.buffers[gbuf_index].buckets[bucket_id].param_data
            g = opt.buffers[gbuf_index].data_parallel_group
            if group is None:
                group = g
            elif g is not group:
                raise RuntimeError("[auto_kernel nvfp4p] the weight shards span two process groups")
            view = buf.view(-1)[world_range.start: world_range.end]
            assert view.dtype == torch.bfloat16 and view.numel() == shard_main_param.nelement()
            for c in range(0, view.numel(), _WAMAX_CHUNK):
                ptrs.append(view.data_ptr() + 2 * c)
                lens.append(min(_WAMAX_CHUNK, view.numel() - c))
                slots.append(slot_of[id(model_param)])
    dev = torch.device("cuda", torch.cuda.current_device())
    _STATE["wamax_table"] = (torch.tensor(ptrs, dtype=torch.int64, device=dev),
                             torch.tensor(lens, dtype=torch.int32, device=dev),
                             torch.tensor(slots, dtype=torch.int32, device=dev))
    _STATE["wamax_buf"] = torch.zeros(len(params), dtype=torch.float32, device=dev)
    _STATE["wamax_params"] = params
    _STATE["wamax_slot"] = {(p.data_ptr(), p.numel()): i for i, p in enumerate(params)}
    _STATE["wamax_group"] = group if group is not None else opt.data_parallel_group


def _wamax_update(opt):
    """after the optimizer wrote this rank's bf16 shards: their amaxes, then the max over the shards' group"""
    import torch
    if _STATE.get("wamax_table") is None:
        _wamax_build(opt)
        _STATE["wamax_opt"] = opt
        F = _fp4g()
        if F is not None:
            F.setup(opt, _STATE)        # may replace _STATE["wamax_buf"] (the slot region appended)
    elif _STATE["wamax_opt"] is not opt:
        raise RuntimeError("[auto_kernel nvfp4p] a second DistributedOptimizer instance: the weight-amax buffer covers one")
    _STATE["wamax_calls"] += 1
    if _WQ_ON:
        _ext_2d()                   # loaded eagerly: its first call is inside the training graph's capture (L4: w2d7)
    cptr, clen, cslot = _STATE["wamax_table"]
    if getattr(opt, "_ak_x9_wamax_done", False):
        # version 10: the fused AdamW step (x9step3, AK_X9_WAMAX=1) already folded this rank's shard amaxes into the
        # buffer while writing the bf16 shards; checked once against the shard pass, byte for byte, before the all-reduce
        opt._ak_x9_wamax_done = False
        if not _STATE.get("wamax_x9_checked"):
            n_par = len(_STATE["wamax_params"])      # 13g: the buffer may carry fp4g's slot region after the weights
            ref = torch.empty_like(_STATE["wamax_buf"][:n_par])
            _ext_w().multi_amax_bf16(cptr, clen, cslot, ref)
            n_bad = int((ref.view(torch.int32) != _STATE["wamax_buf"][:n_par].view(torch.int32)).sum().item())
            ok = n_bad == 0
            if _rank() == 0:
                print("[auto_kernel nvfp4p] weight amaxes from the fused AdamW epilogue ARMED: {} weights; equal to the "
                      "shard pass multi_amax_bf16 ({} differ): {}".format(ref.numel(), n_bad, ok), flush=True)
            if not ok:
                raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the fused-step weight amaxes "
                                   "({} of {} differ)".format(n_bad, ref.numel()))
            _STATE["wamax_x9_checked"] = True
            del ref
    else:
        _ext_w().multi_amax_bf16(cptr, clen, cslot, _STATE["wamax_buf"])
    F = _fp4g()
    if F is not None:
        F.before_allreduce()
    torch.distributed.all_reduce(_STATE["wamax_buf"], op=torch.distributed.ReduceOp.MAX, group=_STATE["wamax_group"])
    _STATE["wamax_valid"] = True


def _wamax_check():
    """the buffer (from the previous step's shards) against a full-weight pass over the gathered weights"""
    import torch
    F = _fp4g()
    n_par = len(_STATE["wamax_params"])
    buf = _STATE["wamax_buf"][:n_par]
    ref = buf.clone()
    n_skip = 0
    for i, p in enumerate(_STATE["wamax_params"]):
        if F is not None and F.is_fp4_weight(p):
            n_skip += 1                              # its bf16 copy is stale outside this shard: fp4g.check covers it
            continue
        if p.numel() % 8 == 0 and p.is_contiguous():
            _ext_w().amax_bf16(p.data, ref[i:i + 1])
        else:                                        # TE's semantics: NaN ignored, inf kept
            ref[i] = torch.nan_to_num(p.data.float().abs(), nan=0.0, posinf=float("inf")).amax()
    eq = bool(torch.equal(buf.view(torch.int32), ref.view(torch.int32)))
    n_bad = int((buf.view(torch.int32) != ref.view(torch.int32)).sum().item())
    if _rank() == 0:
        import hashlib
        path = _ext_w().__file__
        print("[auto_kernel nvfp4p] weight amaxes from the optimizer shards ARMED: {} 2D bf16 weights ({} FP4-gathered "
              "left to fp4g's check), {} chunks on this rank, module {} sha256={}; every amax equal to amax_bf16 of the "
              "gathered weight ({} differ): {}".format(
                  buf.numel(), n_skip, _STATE["wamax_table"][0].numel(), os.path.basename(path),
                  hashlib.sha256(open(path, "rb").read()).hexdigest()[:16], n_bad, eq), flush=True)
    if not eq:
        raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the optimizer-shard weight amaxes "
                           "({} of {} differ)".format(n_bad, buf.numel()))
# Philox subsequence mapping of TE's fused kernel (row variant, column variant; test_srfit.py), and K_TILE_MAX (0: TE's)
_SR_RV = int(os.environ.get("AK_NVFP4P_SR_RV", "0"))
_SR_CV = int(os.environ.get("AK_NVFP4P_SR_CV", "0"))
_SR_VER = int(os.environ.get("AK_NVFP4P_SR_VER", "2"))   # ak_l8b_nvfp4s kernel version (1: tile per CTA, 2: persistent)


def _sr_cast_ok(x, q):
    """the recipe's gradient quantizer on a tensor our cast covers"""
    if not _SR_ON or not _quantizer_ok(q) or not getattr(q, "stochastic_rounding", False):
        return False
    if not (getattr(q, "rowwise_usage", False) and getattr(q, "columnwise_usage", False)):
        return False
    import torch
    if type(x) is not torch.Tensor or not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.dim() < 2:
        return False
    h = x.size(-1)
    return h % 256 == 0 and (x.numel() // h) % 1024 == 0


def _sr_cast(x, q, a_pre, a_post):
    """TE's tex.nvfp4_quantize_with_amax(x, q, a_pre, a_post) for an SR gradient quantizer, with our cast"""
    import transformer_engine_torch as tex
    out = tex.create_empty_quantized_tensor(q, list(x.shape), x.dtype, x.device, False)
    out._amax_rowwise = a_pre.view(-1)[:1]          # aliased, as tex.nvfp4_quantize_with_amax does (cast.cpp:120-127)
    out._amax_columnwise = a_post.view(-1)[:1]
    swz = bool(getattr(q, "optimize_for_gemm", False))
    _ext_s().rht_cast(x, out._amax_rowwise, out._amax_columnwise, out._rowwise_data, out._rowwise_scale_inv,
                      out._columnwise_data, out._columnwise_scale_inv, int(getattr(q, "rht_matrix_random_sign_mask_t")),
                      True, swz, _SR_RV, _SR_CV, 0, _SR_VER)
    if hasattr(out, "_with_gemm_swizzled_scales"):
        out._with_gemm_swizzled_scales = swz
    return out


# ---- version 12: the RHT cast ours (AK_NVFP4P_RHTCAST; ak_l8b_rht6 from AK_NVFP4P_RHTCAST_EXT_DIR) ----------------
_RHTCAST_ON = os.environ.get("AK_NVFP4P_RHTCAST", "0") == "1"
# (width, SR) -> (variant, tokens per tile): the fastest exact variant of each recipe call in the dev tests (job 20771,
# us per call v TE's: 4096 RN 37.3 v 41.3, 4096 SR 48.1 v 56.0, 6144 SR 67.5 v 78.4, 14336 RN 112.6 v 118.8,
# 28672 SR 272.1 v 293.9); other widths take the default of their rounding.  (L4, C2: the
# development override that forced one variant for every call is not read.)
_RHTCAST_TABLE = {(4096, False): (70, 256), (4096, True): (79, 512), (6144, True): (79, 256),
                  (14336, False): (103, 256), (28672, True): (99, 1024)}
_RHTCAST_DEFAULT = {False: (70, 256), True: (79, 512)}
_RHTCAST_FORCE = ""          # L4 (C2): the development override is not read; the table decides
_FAST_MATH = os.environ.get("NVTE_USE_FAST_MATH", "0") not in ("", "0")
_STATE.update({"rhtcast_checked": set(), "rhtcast_calls": 0, "rhtcast_capture_fallback": 0, "rhtcast_results": {}})


# ---- version 14: the training norms' apply rescheduled or folded into the RHT cast (AK_NVFP4P_NORMCAST; ak_l8b_rht7) --
_NORMCAST_ON = os.environ.get("AK_NVFP4P_NORMCAST", "0") == "1" and _RHTCAST_ON
# "apply7": ak_l8b_rht7.rmsnorm_apply_amax7 (apply v6's arithmetic, z written, in a persistent schedule whose block
# order, AK_NVFP4P_NORMCAST_APPLY_VER, reads the rows the residual add wrote last first) and version 12's cast after it;
# "fused": ak_l8b_rht7.rmsnorm_amax (z not written) and rht_cast_norm (the cast forming z itself)
_NC_MODE = os.environ.get("AK_NVFP4P_NORMCAST_MODE", "apply7")
_NC_APPLY_VER = int(os.environ.get("AK_NVFP4P_NORMCAST_APPLY_VER", "7"))   # 7: 16-row blocks reversed, 4 CTAs per slot
_NCVER = int(os.environ.get("AK_NVFP4P_NORMCAST_VER", "4"))         # ak_l8b_rht7.rmsnorm_amax version (4: reversed)
# rht_cast_norm variant and tokens per tile: 70 = version 12's variant 70 with the norm in front, 198 = 70 with gamma
# staged per work tile in shared memory as fp32
_NC_VAR, _NC_KT = (int(v) for v in os.environ.get("AK_NVFP4P_NORMCAST_CAST", "70:256").split(":"))
# version 14 also: the stochastic-rounding casts of version 12 with the generator's Philox state passed to the cast by
# value (ak_l8b_rht7.rht_cast_vp, variants 79 and 99), without the one-thread unpack kernel version 12's binding launches
# in front of every such cast; the same unpack, inside the cast's epilogue warps
_RHTCAST_VP = os.environ.get("AK_NVFP4P_RHTCAST_VP", "0") == "1" and _RHTCAST_ON
_STATE.update({"normcast_calls": 0, "normcast_capture_fallback": 0})

# ---- version 20: the recipe's RHT casts through ak_l8b_rht8 (AK_NVFP4P_RHT8; AK_NVFP4P_RHT8_EXT_DIR) -------------------
_RHT8_ON = os.environ.get("AK_NVFP4P_RHT8", "0") == "1" and _RHTCAST_ON
# (width, SR) -> (variant, tokens per tile) of ak_l8b_rht8: version 12 / 14's variant + 256 (the reciprocal-scale table)
# and, with SR, + 512 (the draws' shared multiplies once, the work loop unswitched per launch); exactness and idle
# timings: nvfp4dev/rx26 dev rx26c (us, deployed -> ours: 4096 RN 38.4 -> 37.2, 14336 RN 113.0 -> 111.4, 4096 SR
# 46.5 -> 37.2, 6144 SR 65.7 -> 53.0, 28672 SR 272.1 -> 223.2)
_RHT8_TABLE = {(4096, False): (326, 256), (14336, False): (359, 256), (4096, True): (847, 512),
               (6144, True): (847, 256), (28672, True): (867, 1024)}


def _ext_r8():
    if _STATE.get("ext_r8") is None:
        d = os.environ.get("AK_NVFP4P_RHT8_EXT_DIR", "/opt/firmus/ak/lib")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ak_l8b_rht8
        _STATE["ext_r8"] = ak_l8b_rht8
    return _STATE["ext_r8"]


def _ext_r7():
    if _STATE.get("ext_r7") is None:
        d = os.environ.get("AK_NVFP4P_NORMCAST_EXT_DIR", "/opt/firmus/ak/lib")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ak_l8b_rht7
        _STATE["ext_r7"] = ak_l8b_rht7
    return _STATE["ext_r7"]


def _normcast_ok(x, q):
    """the norm forward's quantize is version 12's RHT cast (_rhtcast_ok with z's shape, which is x's): x a contiguous bf16
    [.., 4096] tensor with rows % 1024 == 0, the recipe's 1D RHT quantizer with both usages and optimize_for_gemm"""
    import torch
    if not _NORMCAST_ON or not _quantizer_ok(q):
        return False
    if not (getattr(q, "rowwise_usage", False) and getattr(q, "columnwise_usage", False)):
        return False
    if not getattr(q, "optimize_for_gemm", False) or _FAST_MATH or bool(getattr(q, "stochastic_rounding", False)):
        return False
    m = getattr(q, "rht_matrix", None)
    if (type(m) is not torch.Tensor or m.dtype != torch.bfloat16 or m.numel() != 256 or not m.is_contiguous()
            or m.device != x.device or x.device.index != torch.cuda.current_device()):
        return False
    return x.size(-1) == 4096 and (x.numel() // 4096) % 1024 == 0


def _normcast(tex, x, w, rs, q, mask):
    """(tex.rmsnorm_fwd's quantized output, None, rs): "apply7" -- rmsnorm_apply_amax7 and version 12's cast dispatch;
    "fused" -- rmsnorm_amax + rht_cast_norm; None when the output would not take GEMM-swizzled scales (the caller then
    runs version 13g's path)"""
    if _NC_MODE == "apply7":
        z, a_pre, a_post = _ext_r7().rmsnorm_apply_amax7(x, w, rs, mask, _NC_APPLY_VER)
        _STATE["normcast_calls"] += 1
        return (_STATE["rhtcast_wrapped"](z, q, a_pre, a_post), None, rs)
    out = tex.create_empty_quantized_tensor(q, list(x.shape), x.dtype, x.device, False)
    if getattr(out, "_with_gemm_swizzled_scales", None) is not True:
        return None
    r7 = _ext_r7()
    a_pre, a_post = r7.rmsnorm_amax(x, w, rs, mask, _NCVER)
    out._amax_rowwise = a_pre                      # aliased, as tex.nvfp4_quantize_with_amax sets them
    out._amax_columnwise = a_post
    r7.rht_cast_norm(x.view(-1, 4096), w, rs, a_pre, a_post, out._rowwise_data, out._rowwise_scale_inv,
                     out._columnwise_data, out._columnwise_scale_inv, q.rht_matrix, _NC_KT, _NC_VAR)
    _STATE["normcast_calls"] += 1
    return (out, None, rs)


def _rhtcast_pick(width, sr):
    if _RHTCAST_FORCE:
        v, k = _RHTCAST_FORCE.split(":")
        return int(v), int(k)
    if _RHT8_ON and (width, sr) in _RHT8_TABLE:
        return _RHT8_TABLE[(width, sr)]
    return _RHTCAST_TABLE.get((width, sr), _RHTCAST_DEFAULT[sr])


def _rhtcast_r8(width, sr):
    """version 20: this cast through ak_l8b_rht8"""
    return _RHT8_ON and not _RHTCAST_FORCE and (width, sr) in _RHT8_TABLE


def _rhtcast_module(width, sr, var):
    """the extension module a cast of this (width, SR, variant) runs in"""
    if _rhtcast_r8(width, sr):
        return _ext_r8()
    return _ext_r7()                       # L4: _rhtcast sends every other cast to TE (ak_l8b_rht6 dropped)


def _rhtcast_ok(x, q, a_pre, a_post):
    """tex.nvfp4_quantize_with_amax(x, q, a_pre, a_post) as TE runs it through its fused RHT cast kernel with
    GEMM-swizzled scales: the recipe's 1D RHT quantizer with row-wise and column-wise usage and optimize_for_gemm, TE's
    exact math, a contiguous bf16 CUDA input of >= 2 dims with width % 128 == 0 and rows % 1024 == 0 (TE's 1024-token
    tiling then has 8 full 128-token blocks, whose subsequences the variants reproduce), one-element fp32 amaxes"""
    import torch
    if not _quantizer_ok(q) or not (getattr(q, "rowwise_usage", False) and getattr(q, "columnwise_usage", False)):
        return False
    if not getattr(q, "optimize_for_gemm", False) or _FAST_MATH:
        return False
    m = getattr(q, "rht_matrix", None)
    if (type(x) is not torch.Tensor or not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.dim() < 2 or x.numel() == 0 or type(m) is not torch.Tensor or m.dtype != torch.bfloat16
            or m.numel() != 256 or not m.is_contiguous() or m.device != x.device
            or x.device.index != torch.cuda.current_device()):
        return False
    h = x.size(-1)
    if h % 128 != 0 or (x.numel() // h) % 1024 != 0:
        return False
    for a in (a_pre, a_post):
        if type(a) is not torch.Tensor or a.dtype != torch.float32 or a.numel() != 1 or a.device != x.device:
            return False
    return True


def _rhtcast_vp(sr, var):
    """version 14: this cast through ak_l8b_rht7.rht_cast_vp (the Philox state by value)"""
    return sr and _RHTCAST_VP and var in (79, 99)


def _rhtcast(tex, x, q, a_pre, a_post):
    """tex.nvfp4_quantize_with_amax (cast.cpp:93-136) with the cast ours: the output from the quantizer's own
    create_tensor, its amaxes set to the given tensors (as TE sets them), ak_l8b_rht8's (or rht7's) copy of TE's fused
    kernel; None when the output would not take GEMM-swizzled scales or neither module covers the cast (TE then runs)"""
    h = x.size(-1)
    sr = bool(getattr(q, "stochastic_rounding", False))
    var, kt = _rhtcast_pick(h, sr)
    if not _rhtcast_r8(h, sr) and not _rhtcast_vp(sr, var):
        return None                          # L4: ak_l8b_rht6 is not part of this build: TE's cast
    out = tex.create_empty_quantized_tensor(q, list(x.shape), x.dtype, x.device, False)
    if getattr(out, "_with_gemm_swizzled_scales", None) is not True:
        return None
    out._amax_rowwise = a_pre
    out._amax_columnwise = a_post
    if _rhtcast_r8(h, sr):
        r8 = _ext_r8()
        fn = r8.rht_cast_vp if sr else r8.rht_cast_v
        fn(x.view(-1, h), a_pre, a_post, out._rowwise_data, out._rowwise_scale_inv, out._columnwise_data,
           out._columnwise_scale_inv, q.rht_matrix, sr, kt, var)
        return out
    if _rhtcast_vp(sr, var):
        _ext_r7().rht_cast_vp(x.view(-1, h), a_pre, a_post, out._rowwise_data, out._rowwise_scale_inv,
                              out._columnwise_data, out._columnwise_scale_inv, q.rht_matrix, sr, kt, var)
        return out
    raise RuntimeError("[auto_kernel nvfp4p] an RHT cast outside ak_l8b_rht8 / ak_l8b_rht7 reached the cast")


def _make_rhtcast(tex, te_cast):
    """the patch's tex.nvfp4_quantize_with_amax: ours for the calls _rhtcast_ok admits, TE's for everything else"""
    def quant_with_amax(x, q, a_pre, a_post):
        import torch
        if not _RHTCAST_ON or not _rhtcast_ok(x, q, a_pre, a_post):
            return te_cast(x, q, a_pre, a_post)
        h = x.size(-1)
        sr = bool(getattr(q, "stochastic_rounding", False))
        key = (x.numel() // h, h, sr)
        if key in _STATE["rhtcast_checked"]:
            out = _rhtcast(tex, x, q, a_pre, a_post)
            if out is None:
                return te_cast(x, q, a_pre, a_post)
            _STATE["rhtcast_calls"] += 1
            return out
        if torch.cuda.is_current_stream_capturing():
            _STATE["rhtcast_capture_fallback"] += 1          # never checked: TE's cast inside the capture
            return te_cast(x, q, a_pre, a_post)
        # the first eager call: TE's cast and ours on the same input, the stochastic-rounding draws replayed
        gen = torch.cuda.default_generators[x.device.index]
        st = torch.cuda.get_rng_state(x.device) if sr else None
        ref = te_cast(x, q, a_pre, a_post)
        off_ref = gen.get_offset()
        if sr:
            torch.cuda.set_rng_state(st, x.device)
        new = _rhtcast(tex, x, q, a_pre, a_post)
        if new is None:
            return ref
        off_new = gen.get_offset()
        ok = _compare(ref, new)
        ok["swizzled_flag"] = (getattr(ref, "_with_gemm_swizzled_scales", None)
                               == getattr(new, "_with_gemm_swizzled_scales", None))
        ok["type"] = type(ref) is type(new)
        ok["generator_offset"] = off_ref == off_new
        good = all(ok.values())
        _STATE["rhtcast_results"][key] = good
        if _rank() == 0:
            import hashlib
            var, kt = _rhtcast_pick(h, sr)
            path = _rhtcast_module(h, sr, var).__file__
            print("[auto_kernel nvfp4p] RHT cast ARMED ([{}, {}] SR {}): module {} sha256={} variant {}/k{}; byte-exact v "
                  "tex.nvfp4_quantize_with_amax {}: {}".format(
                      key[0], h, sr, os.path.basename(path), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                      var, kt, {kk.lstrip("_"): v for kk, v in ok.items()}, good), flush=True)
        if not good:
            raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the RHT cast: {}".format(ok))
        _STATE["rhtcast_checked"].add(key)
        _STATE["rhtcast_calls"] += 1
        return new
    quant_with_amax.__wrapped__ = te_cast
    return quant_with_amax


# version 13: the 2D weight cast ours (ak_l8b_w2d7: TE's quantize_transpose_nvfp4_2D_kernel arithmetic on a register-tile
# schedule, 64 x 128 per CTA; variant 21 loads with ld.global.nc, 23 adds L1::no_allocate); the variant per weight shape
# (rows, cols) from the dev tests (nvfp4dev/w1 round 6, job 20817: 13.0 / 9.2 / 57.9 / 30.9 us against TE's 20.0 / 14.0 /
# 84.0 / 43.1 us)
_W2D_ON = os.environ.get("AK_NVFP4P_W2D", "0") == "1"
_W2D_TABLE = {(6144, 4096): 23, (4096, 4096): 23, (28672, 4096): 21, (4096, 14336): 23}
_W2D_DEFAULT = 23
_W2D_FORCE = ""              # L4 (C2): the development override is not read; the table decides
_STATE.update({"w2d_calls": 0})


def _ext_2d():
    if _STATE.get("ext_2d") is None:
        d = os.environ.get("AK_NVFP4P_W2D_EXT_DIR", "/opt/firmus/ak/lib")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ak_l8b_w2d7
        _STATE["ext_2d"] = ak_l8b_w2d7
    return _STATE["ext_2d"]


def _w2d_pick(n, k):
    if _W2D_FORCE:
        return int(_W2D_FORCE)
    return _W2D_TABLE.get((n, k), _W2D_DEFAULT)


def _w2d_ok(tensor, q, row, col):
    """TE's 2D weight cast (quantize_transpose_nvfp4_2D_kernel) as ak_l8b_w2d7 covers it: the recipe's 2D weight
    quantizer (no RHT, round to nearest, no 4over6 / row scaling, E4M3 max 448), row-wise usage (column-wise optional),
    TE's exact math, a contiguous 2D bf16 CUDA weight on the current device with rows % 128 == 0 and cols % 128 == 0"""
    import torch
    if not _W2D_ON or not row or not (_env_off("NVTE_USE_FAST_MATH") and _env_off("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH")):
        return False
    if q is None or type(q).__name__ != "NVFP4Quantizer" or not getattr(q, "with_2d_quantization", False):
        return False
    if (getattr(q, "with_rht", False) or getattr(q, "stochastic_rounding", False) or getattr(q, "nvfp4_use_4over6", False)
            or getattr(q, "row_scaled_nvfp4", False) or int(getattr(q, "nvfp4_e4m3_max", 448)) != 448):
        return False
    return (_plain(tensor) and tensor.is_cuda and tensor.dtype == torch.bfloat16 and tensor.dim() == 2
            and tensor.is_contiguous() and tensor.size(0) % 128 == 0 and tensor.size(1) % 128 == 0
            and tensor.device.index == torch.cuda.current_device())


def _w2d_cast(tensor, rd, rs, cd, cs, a, swz):
    """ak_l8b_w2d7's cast of `tensor` with the amax `a` for both directions into the given buffers (cd, cs empty: row-wise
    only), GEMM-swizzled scales when swz"""
    n, k = tensor.shape
    _ext_2d().w2d_cast(tensor, rd, rs, cd, cs, a, a, swz, _w2d_pick(n, k))
    _STATE["w2d_calls"] += 1


def _w2d_desc():
    import hashlib
    path = _ext_2d().__file__
    return "{} sha256={}".format(os.path.basename(path), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16])


def _rank():
    try:
        import torch
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get("RANK", "0"))
    except Exception:  # noqa: BLE001
        return 0


def _quantizer_ok(q):
    if q is None or type(q).__name__ != "NVFP4Quantizer":
        return False
    g = lambda k, d=None: getattr(q, k, d)  # noqa: E731
    return (g("with_rht") is True and g("with_post_rht_amax") is True and not g("with_2d_quantization", False)
            and not g("nvfp4_use_4over6", False) and not g("row_scaled_nvfp4", False)
            and not g("with_amax_reduction", False) and g("amax_reduction_group") is None
            and (g("rowwise_usage") or g("columnwise_usage")))


def _plain(t):
    """a plain tensor or an nn.Parameter holding one (a module's own weight), never a quantized tensor subclass"""
    import torch
    return type(t) is torch.Tensor or type(t) is torch.nn.Parameter


def _norm_site():
    """the module whose code called the norm wrapper: the op fuser's ops/basic/rmsnorm.py ("rmsnorm"), or the module behind
    module/_common.py's apply_normalization ("layernorm_linear")"""
    f = sys._getframe(2)
    while f is not None and f.f_globals.get("__name__", "") == "transformer_engine.pytorch.module._common":
        f = f.f_back
    return f.f_globals.get("__name__", "?").rsplit(".", 1)[-1] if f is not None else "?"


def _tensor_ok(t, last):
    import torch
    return (type(t) is torch.Tensor and t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous()
            and t.dim() >= 2 and t.size(-1) == last and (t.numel() // last) % 32 == 0)


def _same(a, b):
    import torch
    if a is None or b is None:
        return a is None and b is None
    return (tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype
            and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))))


def _compare(ref, new):
    out = {}
    for k in _BUFS:
        out[k] = _same(getattr(ref, k, None), getattr(new, k, None))
    return out


# L4 (2026-10-09, C5 fail early): the extensions dropped from this build (ak_l8b_wq, ak_l8b_rht6, ak_l8b_nvfp4b,
# ak_l8b_nvfp4s) leave switches that would otherwise run silently on TE's path; an armed patch refuses them here.
if _ARMED:
    if _SR_ON:
        raise RuntimeError("[auto_kernel nvfp4p] AK_NVFP4P_SR=1: ak_l8b_nvfp4s is not part of this build")
    if (_BWD_ON or _RHTAMAX_ON) and not _BX_ON:
        raise RuntimeError("[auto_kernel nvfp4p] AK_NVFP4P_BWDNORM / AK_NVFP4P_RHTAMAX need AK_NVFP4P_BX=1: "
                           "ak_l8b_nvfp4b is not part of this build")
    if _WQ_ON and not _W2D_ON:
        raise RuntimeError("[auto_kernel nvfp4p] AK_NVFP4P_WQ=1 needs AK_NVFP4P_W2D=1: ak_l8b_wq is not part of "
                           "this build")
    if _RHTCAST_ON and not _RHT8_ON:
        raise RuntimeError("[auto_kernel nvfp4p] AK_NVFP4P_RHTCAST=1 needs AK_NVFP4P_RHT8=1: ak_l8b_rht6 is not "
                           "part of this build")

if _ARMED:
    import importlib.abc
    import importlib.util

    def _install(tex):
        orig_swiglu = tex.swiglu
        orig_dswiglu = tex.dswiglu
        quant_with_amax_te = tex.nvfp4_quantize_with_amax
        # version 12: the patch's casts after its amax kernels through the RHT cast dispatch (TE's for the rest)
        quant_with_amax = _make_rhtcast(tex, quant_with_amax_te) if _RHTCAST_ON else quant_with_amax_te
        _STATE["rhtcast_te"] = quant_with_amax_te
        _STATE["rhtcast_wrapped"] = quant_with_amax

        def _check(tag, q, run_ref, run_new, stochastic):
            import torch
            key = (tag, bool(getattr(q, "rowwise_usage", False)), bool(getattr(q, "columnwise_usage", False)))
            if key in _STATE["checked"] or torch.cuda.is_current_stream_capturing():
                return None
            _STATE["checked"].add(key)
            st = torch.cuda.get_rng_state() if stochastic else None
            ref = run_ref()
            if stochastic:
                torch.cuda.set_rng_state(st)
            new = run_new()
            ok = _compare(ref, new)
            good = all(ok.values())
            if _rank() == 0:
                import hashlib
                path = _ext().__file__
                sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
                cast = "TE's cast"
                if _STATE.get("ext_s") is not None and stochastic and _SR_ON:
                    sp = _STATE["ext_s"].__file__
                    cast = "our SR cast {} sha256={} (v{}, variants {} {})".format(
                        os.path.basename(sp), hashlib.sha256(open(sp, "rb").read()).hexdigest()[:16], _SR_VER, _SR_RV,
                        _SR_CV)
                print("[auto_kernel nvfp4p] {} + pre/post-RHT amax ARMED (row {} col {}, SR {}): module {} sha256={} v{}, {}; "
                      "byte-exact v tex.{} with an NVFP4 quantizer {}: {}".format(
                          tag, key[1], key[2], bool(stochastic), os.path.basename(path), sha, _KVER, cast, tag,
                          {k[1:]: v for k, v in ok.items()}, good), flush=True)
            if not good:
                raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for {}: {}".format(tag, ok))
            return new

        def swiglu(input_, quantizer, *a, **k):
            if a or k or not _quantizer_ok(quantizer) or not _tensor_ok(input_, input_.size(-1)) \
                    or input_.size(-1) % 512 != 0:
                return orig_swiglu(input_, quantizer, *a, **k)
            mask = int(getattr(quantizer, "rht_matrix_random_sign_mask_t"))

            def run_new():
                y, a_pre, a_post = _ext().swiglu_amax(input_, mask, _KVER)
                return quant_with_amax(y, quantizer, a_pre, a_post)
            checked = _check("swiglu", quantizer, lambda: orig_swiglu(input_, quantizer), run_new,
                             bool(getattr(quantizer, "stochastic_rounding", False)))
            _STATE["fast_calls"] += 1
            return checked if checked is not None else run_new()

        def dswiglu(grad, input_, quantizer, *a, **k):
            if a or k or not _quantizer_ok(quantizer) or not _tensor_ok(input_, input_.size(-1)) \
                    or input_.size(-1) % 512 != 0 or not _tensor_ok(grad, input_.size(-1) // 2) \
                    or grad.numel() * 2 != input_.numel():
                return orig_dswiglu(grad, input_, quantizer, *a, **k)
            mask = int(getattr(quantizer, "rht_matrix_random_sign_mask_t"))

            def run_new():
                dx, a_pre, a_post = _ext().dswiglu_amax(grad, input_, mask, _KVER)
                if _sr_cast_ok(dx, quantizer):
                    return _sr_cast(dx, quantizer, a_pre, a_post)
                return quant_with_amax(dx, quantizer, a_pre, a_post)
            checked = _check("dswiglu", quantizer, lambda: orig_dswiglu(grad, input_, quantizer), run_new,
                             bool(getattr(quantizer, "stochastic_rounding", False)))
            _STATE["fast_calls"] += 1
            return checked if checked is not None else run_new()

        orig_rmsnorm_fwd = tex.rmsnorm_fwd
        norm_on = os.environ.get("AK_NVFP4P_NORM", "1") == "1"

        def _addnorm_take(input_, eps):
            reg = _STATE["addnorm_reg"]
            _STATE["addnorm_reg"] = None          # single use: the next norm forward consumes or drops it
            base = reg["ref"]()
            if base is None or not (input_ is base or getattr(input_, "_base", None) is base):
                return None
            if (input_.data_ptr() != base.data_ptr() or input_.numel() != base.numel()
                    or input_._version != reg["ver"] or float(eps) != reg["eps"]):
                return None
            return reg

        def rmsnorm_fwd(input_, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma, *a, **k):
            import torch
            reg = _addnorm_take(input_, eps) if _STATE["addnorm_reg"] is not None else None
            if (not norm_on or a or k or ln_out is not None or zero_centered_gamma or not _quantizer_ok(quantizer)
                    or not _tensor_ok(input_, 4096) or (input_.numel() // 4096) % 32 != 0
                    or not _plain(weight) or weight.dtype != torch.bfloat16
                    or not weight.is_contiguous() or weight.numel() != 4096):
                return orig_rmsnorm_fwd(input_, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma,
                                        *a, **k)
            mask = int(getattr(quantizer, "rht_matrix_random_sign_mask_t"))
            if _normcast_ok(input_, quantizer):
                site = _norm_site()
                key = ("rmsnorm_fwd_normcast", site) + (("after_add",) if reg is not None else ())

                def run_nc():
                    rs = reg["rs"] if reg is not None else _ext_r7().rmsnorm_stats(input_, float(eps))
                    out = _normcast(tex, input_, weight, rs, quantizer, mask)
                    if out is not None and reg is not None:
                        _STATE["addnorm_norms"] += 1
                    return out
                if key in _STATE["checked"]:
                    new = run_nc()
                    if new is not None:
                        return new
                elif torch.cuda.is_current_stream_capturing():
                    _STATE["normcast_capture_fallback"] += 1        # never checked: version 13g's path in the capture
                else:
                    ref = orig_rmsnorm_fwd(input_, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma)
                    new = run_nc()
                    if new is not None:
                        ok = _compare(ref[0], new[0])
                        ok["swizzled_flag"] = (getattr(ref[0], "_with_gemm_swizzled_scales", None)
                                               == getattr(new[0], "_with_gemm_swizzled_scales", None))
                        ok["type"] = type(ref[0]) is type(new[0])
                        ok["rsigma"] = _same(ref[2], new[2])
                        ok["mu_none"] = ref[1] is None and new[1] is None
                        good = all(ok.values())
                        if _rank() == 0:
                            import hashlib
                            path = _ext_r7().__file__
                            what = ("norm apply v7 (persistent, L2-ordered) + the RHT cast ARMED" if _NC_MODE == "apply7"
                                    else "norm apply folded into the RHT cast ARMED")
                            how = ("apply7 v{}".format(_NC_APPLY_VER) if _NC_MODE == "apply7" else
                                   "amax v{}, cast variant {}/k{}".format(_NCVER, _NC_VAR, _NC_KT))
                            print("[auto_kernel nvfp4p] {} (row True col True{}): module {} sha256={} {}; site {}; "
                                  "byte-exact v tex.rmsnorm_fwd with an NVFP4 quantizer {}: {}".format(
                                      what, ", rsigma from the residual add" if reg is not None else "",
                                      os.path.basename(path), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                                      how, site, {kk.lstrip("_"): v for kk, v in ok.items()}, good), flush=True)
                        if not good:
                            raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the version-14 norm "
                                               "forward ({}): {}".format(_NC_MODE, ok))
                        _STATE["checked"].add(key)
                        return new

            def run_new():
                _STATE["norm_calls"] = _STATE.get("norm_calls", 0) + 1
                if reg is not None:
                    _STATE["addnorm_norms"] += 1
                    if _NVER >= 5:
                        z, a_pre, a_post = _ext_norm().rmsnorm_apply_amax(input_, weight, reg["rs"], mask, _NVER)
                    else:
                        z, a_pre, a_post = _ext_norm().rmsnorm_apply_amax(input_, weight, reg["rs"], mask)
                    return (quant_with_amax(z, quantizer, a_pre, a_post), None, reg["rs"])
                z, rs, a_pre, a_post = _ext_norm().rmsnorm_fwd_amax(input_, weight, float(eps), mask, _NVER)
                return (quant_with_amax(z, quantizer, a_pre, a_post), None, rs)

            def run_ref():
                return orig_rmsnorm_fwd(input_, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma)
            site = _norm_site()
            key = ("rmsnorm_fwd", bool(getattr(quantizer, "rowwise_usage", False)),
                   bool(getattr(quantizer, "columnwise_usage", False)), site) + (("after_add",) if reg is not None else ())
            if key not in _STATE["checked"] and not torch.cuda.is_current_stream_capturing():
                _STATE["checked"].add(key)
                ref = run_ref()
                new = run_new()
                ok = _compare(ref[0], new[0])
                ok["swizzled_flag"] = (getattr(ref[0], "_with_gemm_swizzled_scales", None)
                                       == getattr(new[0], "_with_gemm_swizzled_scales", None))
                ok["type"] = type(ref[0]) is type(new[0])
                ok["rsigma"] = _same(ref[2], new[2])
                ok["mu_none"] = ref[1] is None and new[1] is None
                good = all(ok.values())
                if _rank() == 0:
                    print("[auto_kernel nvfp4p] rmsnorm_fwd + pre/post-RHT amax ARMED (row {} col {}{}): module {} v{}; "
                          "site {}; byte-exact v tex.rmsnorm_fwd with an NVFP4 quantizer {}: {}".format(
                              key[1], key[2], ", rsigma from the residual add" if reg is not None else "",
                              os.path.basename(_ext_norm().__file__), _NVER, site,
                              {kk.lstrip("_"): v for kk, v in ok.items()}, good), flush=True)
                if not good:
                    raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for rmsnorm_fwd: {}".format(ok))
                return new
            return run_new()

        # ---- version 6: the RMSNorm backward with the residual add and the next quantize's amaxes ----
        orig_rmsnorm_bwd = tex.rmsnorm_bwd
        import torch as _torch

        class _ResidualTap(_torch.autograd.Function):
            @staticmethod
            def forward(ctx, residual):
                ctx.ptr = residual.data_ptr()
                ctx.numel = residual.numel()
                return residual.view_as(residual)

            @staticmethod
            def backward(ctx, g):
                if _STATE["res_stash"] is not None:
                    raise RuntimeError("[auto_kernel nvfp4p] a residual gradient was never consumed by its norm backward")
                _STATE["res_stash"] = (g, ctx.ptr, ctx.numel)
                _STATE["taps"] += 1
                return None
        _TAP.append(_ResidualTap)

        class _AddResidualStats(_torch.autograd.Function):
            """h = residual + out (the bias-dropout-add with no bias and dropout 0) with TE's rsigma of h"""
            @staticmethod
            def forward(ctx, residual, out, eps, tap):
                if _SVER != 1:
                    h, rs = _ext_norm().rmsnorm_add_stats(residual, out, float(eps), _SVER)
                else:
                    h, rs = _ext_norm().rmsnorm_add_stats(residual, out, float(eps))
                import weakref
                _STATE["addnorm_reg"] = {"ref": weakref.ref(h), "ver": h._version, "rs": rs, "eps": float(eps)}
                _STATE["addnorm_adds"] += 1
                ctx.tap = bool(tap)
                ctx.ptr = residual.data_ptr()
                ctx.numel = residual.numel()
                return h

            @staticmethod
            def backward(ctx, g):
                if not ctx.tap:
                    return g, g, None, None
                if _STATE["res_stash"] is not None:
                    raise RuntimeError("[auto_kernel nvfp4p] a residual gradient was never consumed by its norm backward")
                _STATE["res_stash"] = (g, ctx.ptr, ctx.numel)
                _STATE["taps"] += 1
                return None, g, None, None
        _ADDF.append(_AddResidualStats)

        def _bwd_fast_ok(dz, x, rsigma, gamma, g, zero_centered_gamma):
            def bf(t):
                return _plain(t) and t.is_cuda and t.dtype == _torch.bfloat16 and t.is_contiguous()
            return (not zero_centered_gamma and bf(dz) and bf(x) and bf(gamma) and bf(g) and x.size(-1) == 4096
                    and gamma.numel() == 4096 and dz.numel() == x.numel() and g.numel() == x.numel()
                    and (x.numel() // 4096) % 32 == 0 and type(rsigma) is _torch.Tensor
                    and rsigma.dtype == _torch.float32 and rsigma.is_contiguous()
                    and rsigma.numel() == x.numel() // 4096)

        def _dg_target(gamma):
            """version 18: the norm weight and its main_grad when its gradient add can be folded into the finalize --
            the gamma TE passes (or its base) carries a bf16 contiguous [4096] main_grad on gamma's device"""
            if not _DGFOLD_ON:
                return None, None
            p = gamma if hasattr(gamma, "main_grad") else getattr(gamma, "_base", None)
            if p is None or not hasattr(p, "main_grad") or getattr(p, "zero_out_wgrad", False):
                return None, None
            mg = p.main_grad
            if (mg is None or type(mg) is not _torch.Tensor or mg.dtype != _torch.bfloat16 or not mg.is_contiguous()
                    or mg.numel() != 4096 or mg.device != gamma.device or gamma.numel() != 4096):
                return None, None
            return p, mg

        def _dg_unfold(gamma):
            """version 18: a norm backward on TE's path for a weight folded before: the hook adds its gradient again"""
            if _DGFOLD_ON and _STATE["dg_folded"]:
                p = gamma if hasattr(gamma, "main_grad") else getattr(gamma, "_base", None)
                if p is not None and id(p) in _STATE["dg_folded"]:
                    p.grad_added_to_main_grad = False
                    _STATE["dg_unfolded"] += 1

        def rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma, *a, **k):
            st = _STATE["res_stash"]
            if st is None:
                _dg_unfold(gamma)
                return orig_rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma, *a, **k)
            _STATE["res_stash"] = None
            g, ptr, numel = st
            if x.data_ptr() != ptr or x.numel() != numel:
                raise RuntimeError("[auto_kernel nvfp4p] the pending residual gradient belongs to another tensor "
                                   "({:#x}, {}) than this norm backward's input ({:#x}, {})".format(
                                       ptr, numel, x.data_ptr(), x.numel()))
            if a or k or not _bwd_fast_ok(dz, x, rsigma, gamma, g, zero_centered_gamma):
                # TE's norm backward, then the add autograd would have done
                _STATE["bwd_stock"] += 1
                _dg_unfold(gamma)
                dx, dg = orig_rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma, *a, **k)
                return [dx + g.reshape(dx.shape), dg]
            dev = x.device
            sms = _STATE.setdefault("sms", {}).get(dev.index)
            if sms is None:
                sms = _torch.cuda.get_device_properties(dev).multi_processor_count
                _STATE["sms"][dev.index] = sms
            cpc = (sms - int(sm_margin)) * _BCPS
            mask = int(_STATE.get("rht_mask", 0xD7E8))
            gv = g.view(x.shape)

            pw, mg = _dg_target(gamma)

            def run_new():
                if mg is None:
                    return _ext_b().rmsnorm_bwd_add(dz, x, rsigma, gamma, gv, cpc, False, mask, True, _BVER)
                return _ext_b().rmsnorm_bwd_add(dz, x, rsigma, gamma, gv, cpc, False, mask, True, _BVER, mg)
            site = sys._getframe(1).f_globals.get("__name__", "?")
            key = ("rmsnorm_bwd_add", site)
            if key not in _STATE["checked"] and not _torch.cuda.is_current_stream_capturing():
                _STATE["checked"].add(key)
                dx_ref, dg_ref = orig_rmsnorm_bwd(dz, x, rsigma, gamma, sm_margin, zero_centered_gamma)
                s_ref = dx_ref + gv
                mg0 = mg.clone() if mg is not None else None
                out, dg, am = run_new()
                pre = _torch.nan_to_num(s_ref.float().abs(), nan=0.0, posinf=float("inf")).max()
                ok = {"sum": _same(s_ref, out), "dgamma": _same(dg_ref, dg), "amax_pre": bool(am[0] == pre)}
                if mg is not None:
                    ok["main_grad"] = _same(mg0.add(dg_ref), mg)       # the DDP hook's main_grad.add_(grad)
                good = all(ok.values())
                if _rank() == 0:
                    import hashlib
                    path = _ext_b().__file__
                    print("[auto_kernel nvfp4p] rmsnorm_bwd + residual add + pre/post-RHT amax ARMED ({}, {} rows, "
                          "ctas_per_col {}): module {} sha256={} v{}; byte-exact v tex.rmsnorm_bwd + torch's add {}: {}"
                          .format(site.rsplit(".", 1)[-1], x.numel() // 4096, cpc, os.path.basename(path),
                                  hashlib.sha256(open(path, "rb").read()).hexdigest()[:16], _BVER, ok, good),
                          flush=True)
                    if mg is not None:
                        print("[auto_kernel nvfp4p] norm weight gradient add folded into the finalize ARMED ({}): "
                              "main_grad [4096] bf16 += dgamma in rmsnorm_bwd_finalize_mg, grad_added_to_main_grad set "
                              "(the DDP hook's add skipped); byte-exact v main_grad.add_(dgamma): {}".format(
                                  site.rsplit(".", 1)[-1], ok["main_grad"]), flush=True)
                if not good:
                    raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for rmsnorm_bwd + add: {}".format(ok))
                del dx_ref, dg_ref, s_ref, mg0
            else:
                out, dg, am = run_new()
            if pw is not None:
                pw.grad_added_to_main_grad = True            # the DDP hook skips its add (TE's fused-wgrad mechanism)
                _STATE["dg_folded"].add(id(pw))
                _STATE["dg_fold_calls"] += 1
            import weakref
            _STATE["bwd_reg"] = {"ref": weakref.ref(out), "ver": out._version, "a_pre": am[0:1], "a_post": am[1:2],
                                 "mask": mask, "site": site.rsplit(".", 1)[-1]}
            _STATE["bwd_fused"] += 1
            return [out, dg]

        def _bwd_reg_take(tensor, quantizer):
            """the registered fused sum, if `tensor` is it (or a view of it) and the quantizer is the recipe's"""
            reg = _STATE["bwd_reg"]
            _STATE["bwd_reg"] = None        # single use: the next quantize call consumes or drops it
            base = reg["ref"]()
            if base is None or type(tensor) is not _torch.Tensor or not _quantizer_ok(quantizer):
                return None
            if not (tensor is base or getattr(tensor, "_base", None) is base):
                return None
            if (tensor.data_ptr() != base.data_ptr() or tensor.numel() != base.numel() or tensor._version != reg["ver"]
                    or not tensor.is_contiguous() or tensor.dtype != _torch.bfloat16
                    or int(getattr(quantizer, "rht_matrix_random_sign_mask_t", -1)) != reg["mask"]):
                return None
            return reg

        # ---- the NVFP4 2D weight quantization (in-place update of a cached weight: tex.quantize(w, q, dst, None)) ----
        orig_quantize = tex.quantize
        weights_mode = os.environ.get("AK_NVFP4P_WEIGHTS", "0")
        weights_on = weights_mode == "1"
        # "amax": only the op fuser's allocating weight quantize (fc1 / fc2), with our amax pass (ak_l8b_nvfp4w.amax_bf16)
        # and TE's own 2D cast through tex.nvfp4_quantize_with_amax; the in-place module weights stay TE's.
        weights_amax = weights_mode == "amax"

        def _weight_ok_any(tensor, quantizer):
            import torch
            if quantizer is None or type(quantizer).__name__ != "NVFP4Quantizer" \
                    or not getattr(quantizer, "with_2d_quantization", False):
                return False
            if (getattr(quantizer, "with_rht", False) or getattr(quantizer, "stochastic_rounding", False)
                    or getattr(quantizer, "nvfp4_use_4over6", False) or getattr(quantizer, "row_scaled_nvfp4", False)
                    or getattr(quantizer, "with_amax_reduction", False)
                    or getattr(quantizer, "amax_reduction_group", None) is not None):
                return False
            return (_plain(tensor) and tensor.is_cuda and tensor.dtype == torch.bfloat16
                    and tensor.is_contiguous() and tensor.dim() == 2 and tensor.numel() % 8 == 0
                    and (bool(getattr(quantizer, "rowwise_usage", False)) or bool(getattr(quantizer, "columnwise_usage", False))))

        def _wq_slot(tensor, quantizer, output):
            """the weight's slot in the shard-amax buffer when tex.quantize(tensor, quantizer, output, None) is TE's amax
            pass + 2D cast into a workspace that already holds exactly the buffers this usage needs; else None"""
            import torch
            if not _weight_ok_any(tensor, quantizer) or not (_env_off("NVTE_USE_FAST_MATH")
                                                             and _env_off("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH")):
                return None
            if type(output).__name__ not in ("NVFP4Tensor", "NVFP4TensorStorage"):
                return None
            n, k = tensor.shape
            if n % 128 or k % 128:       # no scale padding (the Llama weights); TE's path for anything else
                return None
            row, col = bool(quantizer.rowwise_usage), bool(quantizer.columnwise_usage)
            for flag, names in ((row, ("_rowwise_data", "_rowwise_scale_inv", "_amax_rowwise")),
                                (col, ("_columnwise_data", "_columnwise_scale_inv", "_amax_columnwise"))):
                for nm in names:
                    if (getattr(output, nm, None) is not None) != flag:
                        return None
            for nm in ("_amax_rowwise", "_amax_columnwise"):
                t = getattr(output, nm, None)
                if t is not None and (t.numel() != 1 or t.dtype != torch.float32 or not t.is_cuda):
                    return None
            if row and tuple(output._rowwise_data.shape) != (n, k // 2):
                return None
            if col and tuple(output._columnwise_data.shape) != (k, n // 2):
                return None
            if bool(getattr(quantizer, "optimize_for_gemm", False)) and not _wq_swizzled(quantizer, n, k):
                return None              # TE swizzles the scales after the cast
            if not _w2d_ok(tensor, quantizer, row, col):
                return None              # L4: TE's own quantize (ak_l8b_wq dropped); the recipe's are all w2d7's
            return _STATE["wamax_slot"].get((tensor.data_ptr(), tensor.numel()))

        def _weight_ok(tensor, quantizer, output, noop):
            import torch
            if not weights_on or noop is not None or quantizer is None:
                return False
            if type(quantizer).__name__ != "NVFP4Quantizer" or not getattr(quantizer, "with_2d_quantization", False):
                return False
            if (getattr(quantizer, "with_rht", False) or getattr(quantizer, "stochastic_rounding", False)
                    or getattr(quantizer, "nvfp4_use_4over6", False) or getattr(quantizer, "row_scaled_nvfp4", False)
                    or getattr(quantizer, "with_amax_reduction", False)
                    or getattr(quantizer, "amax_reduction_group", None) is not None):
                return False
            if (not _plain(tensor) or not tensor.is_cuda or tensor.dtype != torch.bfloat16
                    or not tensor.is_contiguous() or tensor.dim() != 2
                    or tensor.size(0) % 128 != 0 or tensor.size(1) % 128 != 0):
                return False
            row, col = bool(getattr(quantizer, "rowwise_usage", False)), bool(getattr(quantizer, "columnwise_usage", False))
            if output is None:          # the op-fuser weights (fc1 / fc2): a new tensor each call, built by TE itself
                return row or col
            if row and (getattr(output, "_rowwise_data", None) is None or getattr(output, "_rowwise_scale_inv", None) is None
                        or getattr(output, "_amax_rowwise", None) is None):
                return False
            if col and (getattr(output, "_columnwise_data", None) is None
                        or getattr(output, "_columnwise_scale_inv", None) is None
                        or getattr(output, "_amax_columnwise", None) is None):
                return False
            return row or col

        def _weight_new(tensor, quantizer, output):
            import torch
            row, col = bool(quantizer.rowwise_usage), bool(quantizer.columnwise_usage)
            amax = output._amax_rowwise if row else output._amax_columnwise
            _ext_w().amax_bf16(tensor, amax)
            if row and col:
                output._amax_columnwise.copy_(output._amax_rowwise)
            e = torch.empty(0, device=tensor.device, dtype=torch.uint8)
            _ext_w().quantize_2d(tensor, amax, output._rowwise_data if row else e, output._rowwise_scale_inv if row else e,
                                 output._columnwise_data if col else e, output._columnwise_scale_inv if col else e,
                                 row, col, bool(getattr(quantizer, "optimize_for_gemm", False)), _WVER)
            if hasattr(output, "_with_gemm_swizzled_scales"):
                output._with_gemm_swizzled_scales = bool(getattr(quantizer, "optimize_for_gemm", False))
            return output

        def _sr_quantize(tensor, quantizer):
            import torch
            mask = int(getattr(quantizer, "rht_matrix_random_sign_mask_t"))

            def run_new():
                am = _ext_s().rht_amax(tensor, mask)
                return _sr_cast(tensor, quantizer, am[0:1], am[1:2])
            key = ("quantize_sr", tensor.size(-1), tensor.numel() // tensor.size(-1))
            if key in _STATE["checked"] or torch.cuda.is_current_stream_capturing():
                return run_new()
            _STATE["checked"].add(key)
            st = torch.cuda.get_rng_state()
            ref = orig_quantize(tensor, quantizer)
            torch.cuda.set_rng_state(st)
            new = run_new()
            ok = _compare(ref, new)
            ok["swizzled_flag"] = (getattr(ref, "_with_gemm_swizzled_scales", None)
                                   == getattr(new, "_with_gemm_swizzled_scales", None))
            ok["type"] = type(ref) is type(new)
            good = all(ok.values())
            if _rank() == 0:
                import hashlib
                sp = _ext_s().__file__
                print("[auto_kernel nvfp4p] gradient SR quantize ARMED ({} -> [{}, {}], row True col True, SR True): our amax + "
                      "SR cast {} sha256={} (v{}, variants {} {}); byte-exact v tex.quantize {}: {}".format(
                          list(tensor.shape), key[2], key[1], os.path.basename(sp),
                          hashlib.sha256(open(sp, "rb").read()).hexdigest()[:16], _SR_VER, _SR_RV, _SR_CV,
                          {kk.lstrip("_"): v for kk, v in ok.items()}, good), flush=True)
            if not good:
                raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the SR gradient quantize: {}".format(ok))
            return new

        def _rhtamax_quantize(tensor, quantizer):
            """tex.quantize(tensor, q) for the recipe's 1D RHT quantizer: our two amaxes, TE's cast"""
            import torch
            mask = int(getattr(quantizer, "rht_matrix_random_sign_mask_t"))

            def run_new():
                am = _ext_b().rht_amax(tensor, mask, _RAVER)
                return quant_with_amax(tensor, quantizer, am[0:1], am[1:2])
            sr = bool(getattr(quantizer, "stochastic_rounding", False))
            key = ("rht_amax_quantize", tuple(tensor.shape), bool(quantizer.rowwise_usage),
                   bool(quantizer.columnwise_usage), sr)
            _STATE["rhtamax_calls"] = _STATE.get("rhtamax_calls", 0) + 1
            if key in _STATE["checked"] or torch.cuda.is_current_stream_capturing():
                return run_new()
            _STATE["checked"].add(key)
            st = torch.cuda.get_rng_state() if sr else None
            ref = orig_quantize(tensor, quantizer)
            if sr:
                torch.cuda.set_rng_state(st)
            new = run_new()
            ok = _compare(ref, new)
            ok["swizzled_flag"] = (getattr(ref, "_with_gemm_swizzled_scales", None)
                                   == getattr(new, "_with_gemm_swizzled_scales", None))
            ok["type"] = type(ref) is type(new)
            good = all(ok.values())
            if _rank() == 0:
                import hashlib
                path = _ext_b().__file__
                print("[auto_kernel nvfp4p] RHT amax + TE's cast ARMED ({} row {} col {} SR {}): module {} sha256={}; "
                      "byte-exact v tex.quantize {}: {}".format(
                          list(tensor.shape), key[2], key[3], sr, os.path.basename(path),
                          hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
                          {kk.lstrip("_"): v for kk, v in ok.items()}, good), flush=True)
            if not good:
                raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the RHT-amax quantize: {}".format(ok))
            return new

        def _weight_cast(tensor, quantizer, a, ours):
            """tex.nvfp4_quantize_with_amax(tensor, quantizer, a, a) (cast.cpp:93-136), the cast ak_l8b_w2d7's when `ours`
            and _w2d_ok: the quantizer's own create_tensor, its amaxes set to `a` as TE sets them; TE's whenever TE
            would swizzle the scales after its cast"""
            import torch
            row, col = bool(quantizer.rowwise_usage), bool(quantizer.columnwise_usage)
            if not ours or not _w2d_ok(tensor, quantizer, row, col):
                return quant_with_amax(tensor, quantizer, a, a)
            out = tex.create_empty_quantized_tensor(quantizer, list(tensor.shape), tensor.dtype, tensor.device, False)
            swz = bool(getattr(out, "_with_gemm_swizzled_scales", False))
            if bool(getattr(quantizer, "optimize_for_gemm", False)) and not swz:
                return quant_with_amax(tensor, quantizer, a, a)
            if getattr(out, "_amax_rowwise", None) is not None:
                out._amax_rowwise = a
            if getattr(out, "_amax_columnwise", None) is not None:
                out._amax_columnwise = a
            e = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            _w2d_cast(tensor, out._rowwise_data, out._rowwise_scale_inv, out._columnwise_data if col else e,
                      out._columnwise_scale_inv if col else e, a, swz)
            return out

        def _weight_amax_quantize(tensor, quantizer, ours=False):
            import torch
            if _WAMAX_ON and _STATE["wamax_valid"]:
                slot = _STATE["wamax_slot"].get((tensor.data_ptr(), tensor.numel()))
                if slot is not None:
                    _STATE["wamax_used"] += 1
                    a = _STATE["wamax_buf"][slot:slot + 1]
                    return _weight_cast(tensor, quantizer, a, ours)
            amax = torch.empty(1, dtype=torch.float32, device=tensor.device)
            _ext_w().amax_bf16(tensor, amax)
            return _weight_cast(tensor, quantizer, amax, ours)

        def quantize(tensor, quantizer, output=None, noop=None, *a, **k):
            import torch
            F = _fp4g()
            if F is not None and not a and not k and F.active():
                r = F.weight_quantize(tensor, quantizer, output, noop)
                if r is not None:
                    return r
            if _STATE["bwd_reg"] is not None:
                reg = _bwd_reg_take(tensor, quantizer) if (not a and not k and output is None and noop is None) else None
                if reg is None:
                    _STATE["bwd_reg"] = None
                else:
                    a_pre, a_post = reg["a_pre"], reg["a_post"]

                    def run_reg():
                        if _sr_cast_ok(tensor, quantizer):
                            return _sr_cast(tensor, quantizer, a_pre, a_post)
                        return quant_with_amax(tensor, quantizer, a_pre, a_post)
                    _STATE["bwd_consumed"] += 1
                    sr = bool(getattr(quantizer, "stochastic_rounding", False))
                    key = ("bwd_reg_quantize", reg["site"], bool(quantizer.rowwise_usage),
                           bool(quantizer.columnwise_usage), sr)
                    if key in _STATE["checked"] or torch.cuda.is_current_stream_capturing():
                        return run_reg()
                    _STATE["checked"].add(key)
                    st = torch.cuda.get_rng_state() if sr else None
                    ref = orig_quantize(tensor, quantizer)
                    if sr:
                        torch.cuda.set_rng_state(st)
                    new = run_reg()
                    ok = _compare(ref, new)
                    ok["swizzled_flag"] = (getattr(ref, "_with_gemm_swizzled_scales", None)
                                           == getattr(new, "_with_gemm_swizzled_scales", None))
                    ok["type"] = type(ref) is type(new)
                    good = all(ok.values())
                    if _rank() == 0:
                        print("[auto_kernel nvfp4p] residual-gradient quantize with the norm backward's amaxes ARMED "
                              "(from {}, {} row {} col {} SR {}): TE's cast (tex.nvfp4_quantize_with_amax); byte-exact "
                              "v tex.quantize {}: {}".format(reg["site"], list(tensor.shape), key[2], key[3], sr,
                                                             {kk.lstrip("_"): v for kk, v in ok.items()}, good),
                              flush=True)
                    if not good:
                        raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the registered-amax quantize: "
                                           "{}".format(ok))
                    return new
            if not a and not k and output is None and noop is None and _sr_cast_ok(tensor, quantizer):
                _STATE["fast_calls"] += 1
                return _sr_quantize(tensor, quantizer)
            if (_RHTAMAX_ON and not a and not k and output is None and noop is None and _quantizer_ok(quantizer)
                    and getattr(quantizer, "columnwise_usage", False)
                    and _tensor_ok(tensor, tensor.size(-1)) and tensor.size(-1) % 256 == 0
                    and tensor.numel() // tensor.size(-1) <= 32 * 65535):
                return _rhtamax_quantize(tensor, quantizer)
            if (weights_amax and not a and not k and output is None and noop is None
                    and _weight_ok_any(tensor, quantizer)):
                key = ("weight_amax", tuple(tensor.shape), bool(quantizer.rowwise_usage),
                       bool(quantizer.columnwise_usage), bool(getattr(quantizer, "optimize_for_gemm", False)))
                if key in _STATE["checked"] or torch.cuda.is_current_stream_capturing():
                    # version 13: our 2D cast only for a checked key (an unchecked one inside a capture: TE's cast)
                    return _weight_amax_quantize(tensor, quantizer, key in _STATE["checked"])
                ref = orig_quantize(tensor, quantizer)
                c0 = _STATE["w2d_calls"]
                new = _weight_amax_quantize(tensor, quantizer, True)
                ours = _STATE["w2d_calls"] - c0 == 1
                ok = _compare(ref, new)
                ok["swizzled_flag"] = (getattr(ref, "_with_gemm_swizzled_scales", None)
                                       == getattr(new, "_with_gemm_swizzled_scales", None))
                ok["type"] = type(ref) is type(new)
                good = all(ok.values())
                if _rank() == 0:
                    cast = ("our 2D cast ({} variant {})".format(_w2d_desc(), _w2d_pick(*tensor.shape)) if ours
                            else "TE's 2D cast (tex.nvfp4_quantize_with_amax)")
                    print("[auto_kernel nvfp4p] weight amax ARMED ({} row {} col {} swizzled {}): our amax ({}) + {}; "
                          "byte-exact v tex.quantize {}: {}".format(
                              list(tensor.shape), key[2], key[3], key[4], os.path.basename(_ext_w().__file__), cast,
                              {kk.lstrip("_"): v for kk, v in ok.items()}, good), flush=True)
                if not good:
                    raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the weight amax path: {}".format(ok))
                _STATE["checked"].add(key)          # version 13: only after the check passed
                return new
            if (_WQ_ON and weights_amax and _STATE["wamax_valid"] and not a and not k and output is not None
                    and noop is None):
                slot = _wq_slot(tensor, quantizer, output)
                if slot is not None:
                    row, col = bool(quantizer.rowwise_usage), bool(quantizer.columnwise_usage)
                    _STATE["wq_calls"] += 1
                    key = (id(output), row, col)
                    if key not in _STATE["wq_reg"]:
                        _STATE["wq_reg"][key] = (tensor, row, col, bool(getattr(quantizer, "optimize_for_gemm", False)),
                                                 quantizer)
                    return _wq_into(tensor, quantizer, output, slot, row, col)
            if a or k or not _weight_ok(tensor, quantizer, output, noop):
                return orig_quantize(tensor, quantizer, output, noop, *a, **k)
            if output is None:
                # what tex.quantize(tensor, q) allocates: the quantizer's C++ create_tensor (cast.cpp:149)
                output = tex.create_empty_quantized_tensor(quantizer, list(tensor.shape), tensor.dtype, tensor.device,
                                                           False)
            key = ("weight2d", tuple(tensor.shape), bool(quantizer.rowwise_usage), bool(quantizer.columnwise_usage),
                   bool(getattr(quantizer, "optimize_for_gemm", False)))
            if key not in _STATE["checked"] and not torch.cuda.is_current_stream_capturing():
                _STATE["checked"].add(key)
                ref = orig_quantize(tensor, quantizer)          # TE's own allocation and quantize
                _weight_new(tensor, quantizer, output)
                ok = _compare(ref, output)
                ok["swizzled_flag"] = (getattr(ref, "_with_gemm_swizzled_scales", None)
                                       == getattr(output, "_with_gemm_swizzled_scales", None))
                good = all(ok.values())
                if _rank() == 0:
                    print("[auto_kernel nvfp4p] weight 2D quantize ARMED ({} row {} col {} swizzled {}): module {} v{}; "
                          "byte-exact v tex.quantize {}: {}".format(
                              list(tensor.shape), key[2], key[3], key[4], os.path.basename(_ext_w().__file__), _WVER,
                              {kk.lstrip("_"): v for kk, v in ok.items()}, good), flush=True)
                if not good:
                    raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the weight quantize: {}".format(ok))
                return output
            return _weight_new(tensor, quantizer, output)

        quantize.__wrapped__ = orig_quantize
        _STATE["orig_quantize"] = orig_quantize
        tex.quantize = quantize
        rmsnorm_bwd.__wrapped__ = orig_rmsnorm_bwd
        if _BWD_ON:
            tex.rmsnorm_bwd = rmsnorm_bwd
        rmsnorm_fwd.__wrapped__ = orig_rmsnorm_fwd
        tex.rmsnorm_fwd = rmsnorm_fwd
        swiglu.__wrapped__ = orig_swiglu
        dswiglu.__wrapped__ = orig_dswiglu
        tex.swiglu = swiglu
        tex.dswiglu = dswiglu
        # A module that did `from transformer_engine_torch import rmsnorm_fwd` before this point would keep the original
        # function: re-bind such names to the wrappers.  TE loads its extension before importing any of its modules, so
        # in training and in the selftests the list is empty (the op fuser's ops/basic/rmsnorm.py binds the wrappers).
        repl = {id(orig_quantize): quantize, id(orig_rmsnorm_fwd): rmsnorm_fwd, id(orig_swiglu): swiglu,
                id(orig_dswiglu): dswiglu}
        if _BWD_ON:
            repl[id(orig_rmsnorm_bwd)] = rmsnorm_bwd
        rebound = []
        for mname, mod in list(sys.modules.items()):
            if not mname.startswith("transformer_engine.") or mod is None:
                continue
            md = getattr(mod, "__dict__", None)
            if not isinstance(md, dict):
                continue
            for k, v in list(md.items()):
                if id(v) in repl and v is not repl[id(v)]:
                    setattr(mod, k, repl[id(v)])
                    rebound.append("%s.%s" % (mname, k))
        _STATE["rebound"] = rebound
        if _BWD_ON and _rank() == 0:
            print("[auto_kernel nvfp4p] names re-bound to the wrappers after import: {}".format(sorted(rebound)),
                  flush=True)

    def _wrap_common(mod):
        orig = mod.load_framework_extension

        def load_framework_extension(framework, *a, **k):
            r = orig(framework, *a, **k)
            if framework == "torch":
                _install(sys.modules["transformer_engine_torch"])
            return r
        mod.load_framework_extension = load_framework_extension

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "transformer_engine.common":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _wrap_common(module)
            spec.loader.exec_module = exec_module
            return spec

    if "transformer_engine_torch" in sys.modules:
        _install(sys.modules["transformer_engine_torch"])
    else:
        sys.meta_path.insert(0, _Finder())

    # version 6: Megatron's TransformerLayer passes the residual of each bias-dropout-add through the tap.  The tap is
    # applied after the block's forward, so its autograd node runs before the block's (higher sequence number).
    def _make_bda(orig_bda, layer):
        """version 7: the layer's bias-dropout-add builder, with the plain residual add (no bias, dropout 0) ours"""
        def bda(training, fused):
            inner = orig_bda(training, fused)

            def call(x_with_bias, residual, prob):
                import torch
                x, bias = x_with_bias
                ok = (_STATE["addnorm_enabled"] and _ADDF and bias is None and float(prob) == 0.0 and type(x) is torch.Tensor
                      and type(residual) is torch.Tensor and x.is_cuda and x.dtype == torch.bfloat16
                      and residual.dtype == torch.bfloat16 and x.shape == residual.shape and x.is_contiguous()
                      and residual.is_contiguous() and x.size(-1) == 4096 and (x.numel() // 4096) % 32 == 0)
                if not ok:
                    return inner(x_with_bias, residual_tap(residual), prob)
                tap = bool(_STATE["bwd_enabled"] and _TAP and torch.is_grad_enabled() and residual.requires_grad)
                if tap and _STATE["res_stash"] is not None:
                    raise RuntimeError("[auto_kernel nvfp4p] a residual gradient from the previous backward was never "
                                       "consumed")
                if tap:
                    _tap_summary()
                eps = float(layer.config.layernorm_epsilon)
                key = ("residual_add", bool(training))
                if key not in _STATE["checked"] and not torch.cuda.is_current_stream_capturing():
                    _STATE["checked"].add(key)
                    with torch.no_grad():       # a copy of x: in eval mode the stock add works in place
                        ref = inner((x.detach().clone(), None), residual.detach(), prob)
                    h = _ADDF[0].apply(residual, x, eps, tap)
                    good = _same(ref, h)
                    if _rank() == 0:
                        print("[auto_kernel nvfp4p] residual add + RMSNorm statistics ARMED (training {}, {}): module {} "
                              "add_stats v{}; byte-exact v the bias-dropout-add: {}".format(
                                  bool(training), list(h.shape), os.path.basename(_ext_norm().__file__), _SVER, good),
                              flush=True)
                    if not good:
                        raise RuntimeError("[auto_kernel nvfp4p] E0 self-check FAILED for the residual add")
                    return h
                return _ADDF[0].apply(residual, x, eps, tap)
            return call
        bda.__wrapped__ = orig_bda
        return bda

    def _patch_layer(mod):
        cls = mod.TransformerLayer
        if _ADDNORM_ON:
            orig_init = cls.__init__

            def __init__(self, *a, **k):
                orig_init(self, *a, **k)
                import torch
                for attr in ("self_attn_bda", "mlp_bda"):
                    f = getattr(self, attr, None)
                    if callable(f) and not isinstance(f, torch.nn.Module):
                        setattr(self, attr, _make_bda(f, self))
                        _STATE["bda_wrapped"] = _STATE.get("bda_wrapped", 0) + 1
            __init__.__wrapped__ = orig_init
            cls.__init__ = __init__
        else:
            for name in ("_apply_self_attn_bda_step", "_apply_mlp_bda_step"):
                orig = getattr(cls, name)

                def wrapped(self, out_with_bias, residual, *a, _orig=orig, **k):
                    return _orig(self, out_with_bias, residual_tap(residual), *a, **k)
                wrapped.__wrapped__ = orig
                setattr(cls, name, wrapped)
        _STATE["layer_patched"] = True

    class _LayerFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "megatron.core.transformer.transformer_layer":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _patch_layer(module)
            spec.loader.exec_module = exec_module
            return spec

    def _patch_distopt(mod):
        cls = mod.DistributedOptimizer
        orig = cls._copy_main_params_to_model_params

        def _copy_main_params_to_model_params(self, *a, **k):
            stub = getattr(self, "is_stub_optimizer", False)
            F = _fp4g()
            if not stub and F is not None:
                F.check(self, _STATE["orig_quantize"])  # every gathered weight, before this write (13g)
            if not stub and _STATE["wamax_valid"] and _STATE["wamax_calls"] == 1:
                _wamax_check()      # before this write: the buffer of the first step against the gathered weights
            if not stub and _WQ_ON and _STATE["wamax_valid"] and len(_STATE["wq_checked"]) < len(_STATE["wq_reg"]):
                _wq_check()         # the workspaces our cast filled since (inside capture), before this write
            r = orig(self, *a, **k)
            if not stub:
                _wamax_update(self)
            return r
        _copy_main_params_to_model_params.__wrapped__ = orig
        cls._copy_main_params_to_model_params = _copy_main_params_to_model_params
        _STATE["distopt_patched"] = True

    class _DistOptFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "megatron.core.optimizer.distrib_optimizer":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _patch_distopt(module)
            spec.loader.exec_module = exec_module
            return spec

    if _WAMAX_ON:
        if "megatron.core.optimizer.distrib_optimizer" in sys.modules:
            _patch_distopt(sys.modules["megatron.core.optimizer.distrib_optimizer"])
        else:
            sys.meta_path.insert(0, _DistOptFinder())

    if _BWD_ON:
        if "megatron.core.transformer.transformer_layer" in sys.modules:
            _patch_layer(sys.modules["megatron.core.transformer.transformer_layer"])
        else:
            sys.meta_path.insert(0, _LayerFinder())
