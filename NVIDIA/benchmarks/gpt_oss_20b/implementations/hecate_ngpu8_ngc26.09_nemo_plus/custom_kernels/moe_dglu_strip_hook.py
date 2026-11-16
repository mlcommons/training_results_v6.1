# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strip the fc1 bias gradient and the routing-probability work out of the expert MLP's cuDNN grouped-GEMM epilogues.
TE's fused grouped MLP (transformer_engine/pytorch/ops/fused/grouped_mlp.py, _GroupedMLP_CuTeGEMMBase) scales the fc1
GLU output by the token's routing probability (prob_tensor), adds the probability-scaled bias in the fc2 epilogue
(scale_bias), reduces the fc1 bias gradient (generate_dbias), re-applies the probability and reduces dprob in the
fc2-dgrad dGLU epilogue, and re-reads the bf16 fc2 output gradient in a Triton pass (compute_grouped_dbias_dscales) for
the fc2 bias gradient.  The dGLU kernel costs 1.190 ms/call with all of that on, 0.711 ms with dbias and prob/dprob off.
This module rebinds GroupedMLP_CuTeGEMMGLU's activation / quant / dactivation kernels and compute_grouped_dbias_dscales
(details at the kernel wrappers): the epilogues emit the unscaled h = GLU(pre), z = h W2^T + b2 and d(pre); the
probability multiply and dprob move into the combine (residual_bias_hook); the fc1 bias gradient comes from a
grouped wgrad GEMM with a static ones operand over the columnwise MXFP8 d(pre) (_ones_colsum); the fc2 bias gradient
from the source MOE_DGLU_STRIP_DBIAS2 selects.  The model's math is unchanged: out = res + sum_k p_k (W2 GLU(W1 x + b1) + b2).
Gates: MOE_DGLU_STRIP=0 leaves the reference path (residual_bias_hook and moe_qperm_hook read the same
variable); MOE_DGLU_STRIP_SELFCHECK=0 skips the first-call checks (ones-GEMM bias gradients vs a torch column sum of the
dequantized MXFP8 operand, 'bf16' dbias2 vs a torch fp32 column sum); a failed check raises, there is no silent fallback.
"""
import os
import sys

import torch

_INSTALLED = False
_ORIG = {}
_STATS = {"glu_calls": 0, "quant_calls": 0, "dglu_calls": 0, "dbias1_gemm": 0, "dbias2_gemm": 0, "dbias2_bf16": 0,
          "dbias2_fused": 0, "dbias2_plain": 0}
_ONES = {}                 # (P, M1, device index) -> (e4m3 ones data [P*M1], e8m0 ones scales)
_WORKSPACE = {}            # (E, device index) -> caller-owned descriptor workspace of the cuDNN wgrad op (see _ones_colsum)

_CHECKED = {"dbias1": False, "dbias2": False}

ENABLED = os.getenv("MOE_DGLU_STRIP", "1") == "1"
SELFCHECK = os.getenv("MOE_DGLU_STRIP_SELFCHECK", "1") == "1"
MONES = int(os.getenv("MOE_DGLU_STRIP_MONES", "32"))   # M1 rows of the ones operand (one 32-row quantization block)
MAX_LOG = 3
REL_TOL = 2.0 ** -7        # bf16 output of the GEMM vs an fp32 torch reference of the same products


def _say(msg: str) -> None:
    rank = os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))
    sys.__stdout__.write(f"MOE_DGLU_STRIP rank={rank} {msg}\n")
    sys.__stdout__.flush()


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _ones_operand(P: int, M1: int, device):
    key = (P, M1, device.index if device.index is not None else -1)
    ent = _ONES.get(key)
    if ent is None:
        data = torch.full((P * M1,), 0x38, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)      # e4m3 1.0
        scales = torch.full(((P // 32) * _round_up(M1, 128),), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
        ent = (data, scales)
        _ONES[key] = ent
        _say(f"ones operand allocated: [{P}, {M1}] e4m3 + {scales.numel()} E8M0 scale bytes on {device}")
    return ent


def _descriptor_workspace(E: int, device) -> torch.Tensor:
    """TMA-descriptor workspace of the cuDNN grouped wgrad op for E groups, sized once per (E, device) by the wrapper's
    own get_grouped_gemm_wgrad_workspace_size_sm100 (128 B x 4 slots x E).  Caller-owned so the wrapper keys its op cache
    on operand shapes/dtypes alone (see _ones_colsum); the kernel takes it as a static-shape argument, so the size must match."""
    key = (E, device.index if device.index is not None else -1)
    ws = _WORKSPACE.get(key)
    if ws is None:
        nbytes = int(_ORIG["workspace_size"](E))
        ws = torch.empty((nbytes,), dtype=torch.uint8, device=device)
        _WORKSPACE[key] = ws
        _say(f"cuDNN wgrad descriptor workspace: {nbytes} bytes for E={E} on {device}")
    return ws


def _ones_colsum(cw, cws, offsets_i32, P: int, N: int, stream) -> torch.Tensor:
    """[E, N] bf16: per-group column sums of the columnwise MXFP8 operand (cw [P*N] e4m3 bytes, cws its swizzled E8M0
    scales), computed by the cuDNN grouped wgrad GEMM with a static [P, M1] ones operand (e4m3 1.0, E8M0 1.0 block scales,
    M1 = MOE_DGLU_STRIP_MONES) on the A side, exactly as _cudnn_compute_wgrad calls it; row 0 of each group is returned."""
    wgrad = _ORIG["wgrad"]
    M1 = MONES
    dev = cw.device
    ones_data, ones_scales = _ones_operand(P, M1, dev)
    a = ones_data.view(P, M1).T                                                     # (M1, P), as DY^T in _cudnn_compute_wgrad
    sfa = ones_scales.view(_round_up(M1, 128), -1)
    b = cw.view(dtype=torch.float8_e4m3fn).view(P, N)
    sfb = cws.view(dtype=torch.float8_e8m0fnu).view(_round_up(N, 128), -1)
    E = offsets_i32.shape[0]
    out = torch.empty((E, M1, N), dtype=torch.bfloat16, device=dev)
    # Dense [E, M1, N] output (all M1 rows identical) rather than an extra column block of main_grad: the parameter's
    # main_grad has row stride in_features, so a wider wgrad result cannot land in it in place.  The caller-owned
    # descriptor workspace takes the output's data pointer out of the wrapper's op-cache key (cudnn/gemm/cutedsl/grouped/
    # wgrad/api.py otherwise binds a block-scaled dense-output op to one output buffer, so every fresh output would rebuild
    # the op, and its check_support() copies the offsets to the host, which the CUDA graph capture forbids; TE's own wgrads
    # write into the persistent main_grad and never hit this).  Both ones-GEMMs run on the chain stream, so one workspace
    # per E suffices and the bias gradients flow through autograd and the DDP hooks exactly as before.
    wgrad(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        offsets_tensor=offsets_i32,
        acc_dtype=torch.float32,
        sf_vec_size=32,
        accumulate_on_output=False,
        current_stream=stream if stream is not None else torch.cuda.current_stream(dev).cuda_stream,
        output_mode="dense",
        wgrad_tensor=out,
        wgrad_dtype=out.dtype,
        descriptor_workspace=_descriptor_workspace(E, dev),
    )
    return out[:, 0, :]


def _colsum_reference(cw, cws, offsets_i32, P: int, N: int) -> torch.Tensor:
    """fp32 [E, N]: column sums of the dequantized columnwise MXFP8 operand, group by group (TE grouped scale layout 3,
    decoded with moe_qperm.grouped_columnwise_positions)."""
    import custom_kernels.moe_qperm as K
    dev = cw.device
    ends = offsets_i32.to(torch.int64).tolist()
    E = len(ends)
    padded = torch.tensor([ends[0]] + [ends[i] - ends[i - 1] for i in range(1, E)], dtype=torch.int64, device=dev)
    out = torch.zeros((E, N), dtype=torch.float32, device=dev)
    data = cw.view(dtype=torch.float8_e4m3fn).view(P, N)
    cws_u8 = cws.view(dtype=torch.uint8).reshape(-1)
    start = 0
    for g in range(E):
        m_g = int(padded[g].item())
        if m_g > 0:
            rows = data[start:start + m_g].float()                                  # [m_g, N]
            k = torch.arange(N, device=dev)[:, None].expand(N, m_g // 32)
            mb = torch.arange(start // 32, (start + m_g) // 32, device=dev)[None, :].expand(N, m_g // 32)
            pos = K.grouped_columnwise_positions(k, mb, padded, N)                  # [N, m_g/32] byte offsets
            e = cws_u8[pos.reshape(-1)].reshape(N, m_g // 32).to(torch.int32)
            scale = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - 127).t()   # [m_g/32, N]
            rows = rows.view(m_g // 32, 32, N) * scale[:, None, :]
            out[g] = rows.sum(dim=(0, 1))
        start += m_g
    return out


def _check(tag: str, got: torch.Tensor, cw, cws, offsets_i32, P: int, N: int) -> None:
    ref = _colsum_reference(cw, cws, offsets_i32, P, N)
    d = (got.float() - ref).abs().max().item()
    s = ref.abs().max().item()
    ok = d <= REL_TOL * max(s, 1e-30)
    _say(f"{tag} self-check: ones-operand grouped wgrad GEMM vs torch column sum of the dequantized MXFP8 operand: "
         f"max|diff|={d:.3e} max|ref|={s:.3e} ({'OK' if ok else 'FAIL'}); P={P} N={N} E={offsets_i32.shape[0]} M1={MONES}")
    if not ok:
        raise RuntimeError(f"MOE_DGLU_STRIP: {tag} self-check FAILED (max|diff| {d:.3e} vs max|ref| {s:.3e}); refusing to continue")


# ------------------------------------------------------------------------------------------------------------
# kernel wrappers
#   _glu_wrapper / _quant_wrapper: prob_tensor=None -- fc1 emits the unscaled h = GLU(pre), fc2 the unscaled z = h W2^T + b2.
#   _dglu_wrapper: generate_dbias=False, prob_tensor=None, dprob_tensor=None -- the epilogue does only the dGLU math and
#       emits d(pre) MXFP8 rowwise + columnwise.  dbias1[e] = ones^T . d(pre)_e comes from the ones-operand grouped wgrad
#       GEMM over the columnwise copy it just wrote; dprob goes back as the zero tensor TE handed in (the real dprob =
#       dres . z, dy.b2 term included, is computed in the combine backward, gpt_oss::moe_qperm_gather_quant_scaled).
#   _compute_grouped_dbias_dscales replaces TE's Triton pass.  dbias2[e] = sum of the dz rows of expert e, from the source
#       MOE_DGLU_STRIP_DBIAS2 selects (moe_qperm_hook.DBIAS2_MODE):
#       'fused' (default): the combine-backward gather kernel accumulates the fp32 column sum of the bf16 dz rows (the
#           gradient before its MXFP8 quantization) while writing the quantized operands and parks it beside them.
#       'bf16': the gather kernel also writes the bf16 dz rows and TE's own compute_grouped_dbias (fp32 atomics) reduces them.
#       'gemm': the ones-operand grouped wgrad GEMM over the columnwise MXFP8 copy of dz (a column sum of the quantized grad).
#       Whatever the source, TE's two memsets, the bf16 re-read of dz, the dscales pass and the fp32->bf16 cast disappear.
# ------------------------------------------------------------------------------------------------------------
def _glu_wrapper(**kw):
    kw["prob_tensor"] = None                      # h = GLU(pre), unscaled: the probability is applied in the combine
    _STATS["glu_calls"] += 1
    if _STATS["glu_calls"] <= MAX_LOG:
        _say("fc1 GLU epilogue: probability multiply off")
    return _ORIG["glu"](**kw)


def _quant_wrapper(**kw):
    kw["prob_tensor"] = None                      # fc2: z = acc + b2, the bias unscaled; fc1 dgrad: already None
    _STATS["quant_calls"] += 1
    if _STATS["quant_calls"] <= MAX_LOG:
        _say("grouped quant GEMM (fc2 forward / fc1 dgrad): prob_tensor off")
    return _ORIG["quant"](**kw)


def _dglu_wrapper(**kw):
    dprob_in = kw.get("dprob_tensor")
    kw["prob_tensor"] = None
    kw["dprob_tensor"] = None
    kw["generate_dbias"] = False
    out = _ORIG["dglu"](**kw)
    a = kw["a_tensor"]
    P = int(a.shape[0])
    offs = kw["padded_offsets"]
    if offs.dtype != torch.int32:
        offs = offs.to(torch.int32)
    d_col = out["d_col_tensor"]
    N = d_col.numel() // P
    d_col_flat = d_col.view(P, N).reshape(-1)
    sfd_col = out["sfd_col_tensor"].permute(5, 2, 4, 0, 1, 3).reshape(-1)
    dbias1 = _ones_colsum(d_col_flat, sfd_col, offs, P, N, kw.get("current_stream"))      # [E, N] bf16
    if SELFCHECK and not _CHECKED["dbias1"]:
        _CHECKED["dbias1"] = True
        _check("fc1 dbias", dbias1, d_col_flat, sfd_col, offs, P, N)
    out["dbias_tensor"] = dbias1.unsqueeze(-1)                                               # TE: squeeze(-1).to(dtype)
    out["dprob_tensor"] = dprob_in                                                           # the zeros TE handed in
    _STATS["dglu_calls"] += 1
    _STATS["dbias1_gemm"] += 1
    if _STATS["dglu_calls"] <= MAX_LOG:
        _say(f"fc2 dgrad + dGLU epilogue: dbias / prob / dprob off; fc1 dbias from the ones-operand grouped wgrad GEMM "
             f"over d(pre) columnwise [{P}, {N}] (M1={MONES})")
    return out


def _compute_grouped_dbias_dscales(dy, scales, bias, offsets, dbias=None, dscales=None):
    import custom_kernels.moe_qperm_hook as QH
    st = QH.pop_backward_operand(dy.data_ptr())
    if st is None:
        if QH._is_placeholder(dy):
            raise RuntimeError("MOE_DGLU_STRIP: the fc2 output-gradient placeholder reached compute_grouped_dbias_dscales "
                               "without its parked operands; refusing to continue")
        _STATS["dbias2_plain"] += 1
        if _STATS["dbias2_plain"] <= MAX_LOG:
            _say("fc2 dbias: no parked columnwise operand for this gradient (plain path); TE's Triton dbias/dscales pass")
        return _ORIG["compute_grouped_dbias_dscales"](dy, scales, bias, offsets, dbias=dbias, dscales=dscales)
    mode = QH.DBIAS2_MODE
    if mode == "fused":
        dbias2 = st.dbias                                                                    # [E, H] fp32, from the gather kernel
        if dbias2 is None or dbias2.shape != (st.E, st.H):
            raise RuntimeError("MOE_DGLU_STRIP: DBIAS2 mode 'fused' but the parked operands carry no [E, H] column sum; "
                               "refusing to continue")
        _STATS["dbias2_fused"] += 1
        if _STATS["dbias2_fused"] <= MAX_LOG:
            _say(f"fc2 dbias = the fp32 column sum of the bf16 dz rows fused into the combine-backward gather [{st.P}, {st.H}]; "
                 "the grouped dbias/dscales Triton pass, its memsets and its cast are gone")
    elif mode == "bf16":
        # dy IS the bf16 dz rows the gather wrote (park_backward keyed the operands under them): TE's own pass
        dbias2 = _ORIG["compute_grouped_dbias"](dy, offsets, st.E)                          # [E, H] fp32
        if SELFCHECK and not _CHECKED["dbias2"]:
            _CHECKED["dbias2"] = True
            import custom_kernels.moe_qperm as K
            padded = (offsets[1:] - offsets[:-1]) if offsets.numel() == st.E + 1 else None
            if padded is None:
                raise RuntimeError("MOE_DGLU_STRIP: offsets are not the (E+1) cumulative form; refusing to continue")
            ref = K.grouped_colsum_reference(dy, padded.to(torch.int32), 1)
            d = (dbias2 - ref).abs().max().item(); sc = ref.abs().max().item()
            ok = d <= 2.0 ** -12 * max(sc, 1e-30)
            _say(f"fc2 dbias self-check [bf16]: TE compute_grouped_dbias over the bf16 dz rows vs torch fp32 column sum: "
                 f"max|diff|={d:.3e} max|ref|={sc:.3e} ({'OK' if ok else 'FAIL'}); P={st.P} H={st.H} E={st.E}")
            if not ok:
                raise RuntimeError("MOE_DGLU_STRIP: fc2 dbias [bf16] self-check FAILED; refusing to continue")
        _STATS["dbias2_bf16"] += 1
        if _STATS["dbias2_bf16"] <= MAX_LOG:
            _say(f"fc2 dbias from TE's compute_grouped_dbias over the bf16 dz rows [{st.P}, {st.H}] (split run: the "
                 "unquantized gradient, fp32 atomics); the dscales half of HEAD's pass and its memset are gone")
    else:
        offs = offsets[1:].to(torch.int32) if offsets.numel() == st.E + 1 else offsets.to(torch.int32)
        dbias2 = _ones_colsum(st.cw, st.cws, offs, st.P, st.H, None)                         # [E, H] bf16
        if SELFCHECK and not _CHECKED["dbias2"]:
            _CHECKED["dbias2"] = True
            _check("fc2 dbias", dbias2, st.cw, st.cws, offs, st.P, st.H)
        _STATS["dbias2_gemm"] += 1
        if _STATS["dbias2_gemm"] <= MAX_LOG:
            _say(f"fc2 dbias from the ones-operand grouped wgrad GEMM over dz columnwise [{st.P}, {st.H}] (M1={MONES}); "
                 "the grouped dbias/dscales Triton pass, its memsets and its cast are gone")
    if dscales is None:
        dscales = torch.zeros((st.P,), dtype=torch.float32, device=dy.device)
    return dbias2, dscales


# ------------------------------------------------------------------------------------------------------------
def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    if not ENABLED:
        _say("disabled by MOE_DGLU_STRIP=0 (control path: probability in the GEMM epilogues, dbias in the dGLU epilogue, "
             "TE's grouped dbias/dscales pass)")
        return False
    try:
        import custom_kernels.moe_qperm_hook as QH
        import custom_kernels.residual_bias_hook as RBF
        from transformer_engine.pytorch.ops.fused import grouped_mlp as GM
    except Exception as ex:  # noqa
        _say(f"not installed: import failed {type(ex).__name__}: {ex}")
        return False
    if not (RBF._INSTALLED and RBF.DGLU and RBF.COMBINE and "hybrid_ep_combine" in RBF._ORIG):
        # without the combine applying the probability the epilogues must keep it: nothing is patched (reference path)
        _say("not installed: residual_bias_hook's combine (the probability multiply) is not installed; "
             "the GEMM epilogues keep the probability and the dbias / dprob work")
        return False
    if not (QH._INSTALLED and QH.DGLU):
        _say("not installed: moe_qperm_hook is not installed with MOE_DGLU_STRIP=1 (the combine backward parks through it)")
        return False
    cls = getattr(GM, "GroupedMLP_CuTeGEMMGLU", None)
    if cls is None or not hasattr(GM, "compute_grouped_dbias_dscales") or not hasattr(GM, "compute_grouped_dbias"):
        _say("not installed: grouped_mlp has no GroupedMLP_CuTeGEMMGLU / compute_grouped_dbias_dscales / compute_grouped_dbias")
        return False
    try:
        glu = cls.grouped_gemm_activation_kernel()
        quant = cls.grouped_gemm_quant_kernel()
        dglu = cls.grouped_gemm_dactivation_kernel()
        wgrad = cls.grouped_gemm_wgrad_kernel()
    except Exception as ex:  # noqa
        _say(f"not installed: cuDNN grouped GEMM wrappers unavailable ({type(ex).__name__}: {ex})")
        return False
    if wgrad is None:
        _say("not installed: the cuDNN grouped wgrad kernel is disabled (NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP)")
        return False
    try:
        from cudnn import get_grouped_gemm_wgrad_workspace_size_sm100 as ws_size   # the wgrad wrapper's own size function
    except ImportError:
        import importlib
        ws_size = getattr(importlib.import_module(wgrad.__module__), "get_grouped_gemm_wgrad_workspace_size_sm100", None)
    if ws_size is None:
        _say("not installed: the cuDNN wgrad wrapper exposes no get_grouped_gemm_wgrad_workspace_size_sm100 (the ones-GEMM "
             "needs a caller-owned descriptor workspace of the op's exact size)")
        return False
    _ORIG.update(glu=glu, quant=quant, dglu=dglu, wgrad=wgrad, workspace_size=ws_size,
                 compute_grouped_dbias_dscales=GM.compute_grouped_dbias_dscales, compute_grouped_dbias=GM.compute_grouped_dbias)
    cls.grouped_gemm_activation_kernel = classmethod(lambda c: _glu_wrapper)
    cls.grouped_gemm_quant_kernel = classmethod(lambda c: _quant_wrapper)
    cls.grouped_gemm_dactivation_kernel = classmethod(lambda c: _dglu_wrapper)
    GM.compute_grouped_dbias_dscales = _compute_grouped_dbias_dscales
    QH.RECORD_BWD_OPERANDS = True
    QH._DGLU_ENGAGED[0] = True
    RBF._DGLU_ENGAGED[0] = True
    _INSTALLED = True
    db2 = {"fused": "fc2 dbias = fp32 column sum fused into the combine-backward gather",
           "bf16": "fc2 dbias = TE compute_grouped_dbias over the bf16 dz rows [split run]",
           "gemm": "fc2 dbias ones-GEMM"}[QH.DBIAS2_MODE]
    _say(f"installed: GroupedMLP_CuTeGEMMGLU.grouped_gemm_activation_kernel / grouped_gemm_quant_kernel (prob off), "
         f"grouped_gemm_dactivation_kernel (dbias / prob / dprob off + fc1 dbias ones-GEMM), "
         f"grouped_mlp.compute_grouped_dbias_dscales ({db2}; MOE_DGLU_STRIP_DBIAS2={QH.DBIAS2_MODE}); M1={MONES} selfcheck={SELFCHECK}")
    return True


def stats():
    return dict(_STATS)
