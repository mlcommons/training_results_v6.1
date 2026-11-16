# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import atexit
import itertools
import os
from typing import Any, Dict, Optional, Tuple

import torch

KNOB: str = "DLRM_HSTU_FP8_BWD_ERM"
ERM_SITE_ATTR: str = "_hstu_dy_erm_site"
_REQUESTED: bool = os.environ.get(KNOB, "0") == "1"
_override: Optional[bool] = None
_broken: Optional[str] = None


_prebuilt: bool = False
_inert: Optional[str] = None
_inert_checked: bool = False
_tokens = itertools.count(1)
_once: set = set()
_counts: Dict[str, int] = {
    "sites": 0,
    "claimed": 0,
    "tickets": 0,
    "emitted": 0,
    "consumed": 0,
    "declined": 0,
}
_reasons: Dict[str, int] = {}
_atexit_registered: bool = False
D: int = 512


M_RM_MAX: int = (2**31 - 1) // D


_bwd = None
_w3 = None
_tln = None
_snr = None


def _mods() -> Tuple[Any, Any, Any, Any]:
    global _bwd, _w3, _tln, _snr
    if _bwd is None:
        from generative_recommenders.ops.triton import bwd_fp8 as b

        _bwd = b
    if _w3 is None:
        from generative_recommenders.ops.triton import wgrad_ck_w3 as w

        _w3 = w
    if _tln is None:
        from generative_recommenders.ops.triton import (
            triton_layer_norm as t,
        )

        _tln = t
    if _snr is None:
        from generative_recommenders.ops.triton import fp8_wgrad_snr as s

        _snr = s
    return _bwd, _w3, _tln, _snr


def _print(msg: str) -> None:
    print(f"[hstu-fp8-bwd] {msg}", flush=True)


def _arm_print(msg: str) -> None:
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(f"[hstu-fp8-bwd] {msg}")
    except Exception:
        _print(msg)


def _note(key: str, msg: str) -> None:
    _reasons[key] = _reasons.get(key, 0) + 1
    if key not in _once:
        _once.add(key)
        _arm_print(msg)


def _register_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        _atexit_registered = True


        atexit.register(_final)


def requested() -> bool:
    return _REQUESTED if _override is None else _override


def set_requested_for_testing(value: Optional[bool]) -> None:
    global _override
    _override = value


def broken() -> Optional[str]:
    return _broken


def enabled() -> bool:
    return requested() and _broken is None


def set_broken(reason: str) -> None:
    global _broken
    if _broken is None:
        _broken = reason
        _arm_print(f"erm DISABLED: {reason}")


def mark_prebuilt() -> None:
    global _prebuilt
    _prebuilt = True


def prebuilt() -> bool:
    return _prebuilt


def inert_reason() -> Optional[str]:
    global _inert, _inert_checked
    if _inert_checked:
        return _inert
    missing = []
    try:
        b, w, t, _s = _mods()
        from generative_recommenders.ops.triton import (
            triton_fp8 as tf8,
            triton_hstu_linear as thl,
        )

        if not b.scale_controller_enabled():
            missing.append("DLRM_HSTU_FP8_BWD")
        if not (getattr(b, "_OUT_WGRAD_ARMED", False) and getattr(thl, "_FP8_OUT_WGRAD", False)):
            missing.append("DLRM_HSTU_FP8_BWD_OUT_WGRAD")
        if getattr(thl, "_FP8_OUT_WGRAD_IMPL", "nn") != "nn":
            missing.append("DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL=nn")
        if not (getattr(b, "_OUT_WGRAD_T_E4M3", False) and b._GRAD_DTYPE is b._FP8_E5M2):
            missing.append("DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT=e4m3 with e5m2 grads")
        if not getattr(b, "_BWD_HELPERS_FUSED", False):
            missing.append("DLRM_HSTU_FP8_BWD_HELPERS_FUSED")
        if not t._wln_bwd_flydsl_enabled():
            missing.append("DLRM_HSTU_FP8_WLN_BWD_FLYDSL")
        if not w.requested():
            missing.append("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3")
        if not w.tiled_requested():
            missing.append("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED")
        if not w.t256s_requested():
            missing.append("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S")
        if not tf8.hstu_fp8_fuse_residual_enabled():
            missing.append("DLRM_HSTU_FP8_FUSE_RESIDUAL")
    except Exception as error:
        missing.append(f"prerequisite check failed: {type(error).__name__}: {error}")
    _inert = ", ".join(missing) if missing else None
    _inert_checked = True
    return _inert


def _requested_line() -> Optional[str]:
    why = inert_reason()
    if "requested" not in _once:
        _once.add("requested")
        _arm_print("requested: erm=1" + (f" (INERT without {why})" if why else ""))
    return why


class ErmSite:
    __slots__ = ("key", "device", "claimed", "consumed", "record")

    def __init__(self, key: int, device: torch.device) -> None:
        self.key = int(key)
        self.device = device
        self.claimed = False
        self.consumed = False
        self.record: Optional["ErmEmit"] = None


class ErmTicket:
    __slots__ = ("state", "calls", "t_shift", "inv_scale", "stream")

    def __init__(
        self, state: Any, calls: int, t_shift: int, inv_scale: torch.Tensor, stream: Any
    ) -> None:
        self.state = state
        self.calls = int(calls)
        self.t_shift = int(t_shift)
        self.inv_scale = inv_scale
        self.stream = stream


class ErmEmit:
    def __init__(
        self,
        tk: ErmTicket,
        dx: torch.Tensor,
        g8: torch.Tensor,
        x_rm: torch.Tensor,
        partials: torch.Tensor,
    ) -> None:
        self.ticket = tk
        self.g8: Optional[torch.Tensor] = g8
        self.x_rm: Optional[torch.Tensor] = x_rm
        self.partials: Optional[torch.Tensor] = partials
        self.dx_ptr = int(dx.data_ptr())
        self.dx_version = int(dx._version)
        self.shape = tuple(dx.shape)
        self.stride = tuple(dx.stride())
        self.dtype = dx.dtype
        self.used = False


        self.layout = "rm"
        self.token = next(_tokens)

    def release(self) -> None:
        self.g8 = None
        self.x_rm = None
        self.partials = None


def _stream(device: torch.device) -> Any:
    return torch.cuda.current_stream(device) if device.type == "cuda" else None


def make_site(
    output_weight: torch.Tensor,
    *,
    fp8_y_saved: bool,
    y_token_major: bool,
    dout_token_major: bool,
    out_wgrad_nn: bool,
    need_bwd: bool,
    layer_tag: int,
    helpers_fused_ok: bool,
) -> Optional[ErmSite]:


    if not enabled():
        return None
    if _requested_line() is not None:
        return None
    if not (fp8_y_saved and dout_token_major and out_wgrad_nn) or y_token_major:
        return None
    if not need_bwd:
        return None
    if int(layer_tag) != 0:
        return None
    if not (output_weight.is_leaf and output_weight.requires_grad and helpers_fused_ok):
        return None
    _counts["sites"] += 1
    _register_atexit()
    return ErmSite(output_weight.data_ptr(), output_weight.device)


def claim(x: torch.Tensor, *, residual_proxy: bool, targets_only: bool) -> Optional[ErmSite]:
    if not enabled():
        return None
    site = getattr(x, ERM_SITE_ATTR, None)
    if not isinstance(site, ErmSite):
        return None
    if targets_only or not residual_proxy:
        return None
    if not (torch.is_grad_enabled() and x.requires_grad):
        return None
    if site.claimed:
        return None
    site.claimed = True
    _counts["claimed"] += 1
    return site


def producer_eligible(
    x: torch.Tensor,
    dy: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> Optional[str]:


    _, _, t, _ = _mods()
    why = t.flydsl_ln_bwd_decline_reason(
        x, dy, weight, bias, mean, rstd, residual, is_swish=False, sum_dx=False
    )
    if why is not None:
        return f"shipped FlyDSL LN bwd not taken: {why}"
    if weight.dtype != torch.bfloat16 or int(x.shape[1]) != D:
        return f"ERM variant covers D=512 bf16 weights only ({weight.dtype}, D={int(x.shape[1])})"
    if int(x.shape[0]) > M_RM_MAX:
        return f"M={int(x.shape[0])} > {M_RM_MAX} (x_rm 32-bit offsets)"
    if not _prebuilt:
        return "ERM FlyDSL variant not prebuilt before RUN_START"
    return None


def ticket(
    site: Optional[ErmSite],
    x: torch.Tensor,
    dy: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    fold_dropout: bool,
) -> Optional[ErmTicket]:

    if site is None or not enabled():
        return None
    if site.consumed or site.record is not None:
        return None
    if fold_dropout or residual is None:
        return None
    if torch.compiler.is_compiling() or os.environ.get("AOTI_LOWER"):
        _note("compile", "erm NOT used (compiling/AOTI)")
        return None
    b, w, _, s = _mods()
    if not (
        b.scale_controller_enabled()
        and b._OUT_WGRAD_ARMED
        and b._OUT_WGRAD_T_E4M3
        and b._BWD_HELPERS_FUSED
        and b._GRAD_DTYPE is b._FP8_E5M2
    ):
        _note("inert", "erm NOT used (prerequisites: BWD, OUT_WGRAD e4m3 copy, HELPERS_FUSED, e5m2 grads)")
        return None
    if bool(getattr(s, "SNR_ENABLED", False)):
        _note("snr", "erm NOT used (SNR probe on: fp8_wgrad_snr.SNR_ENABLED)")
        return None
    st = b._peek_site_state("output_projection", site.key)
    if st is None or not st.published or st.t_clipped_windows is None:
        return None
    why_w3 = w.rm_inactive_reason(x.device)
    if why_w3 is not None:
        _note("w3:" + why_w3.split(" (")[0], f"erm NOT used (W3: {why_w3})")
        return None
    if (
        x.dtype != torch.bfloat16
        or dy.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or x.dim() != 2
        or int(x.shape[1]) != D
        or int(x.shape[0]) < 1
        or dy.shape != x.shape
        or residual.shape != x.shape
        or not (x.is_contiguous() and dy.is_contiguous() and residual.is_contiguous())
    ):
        _note("shape", "erm NOT used (producer operands outside the contract)")
        return None
    why_p = producer_eligible(x, dy, residual, weight, bias, mean, rstd)
    if why_p is not None:
        _note("producer:" + why_p.split(":")[0].split(" (")[0], f"erm NOT used ({why_p})")
        return None
    _counts["tickets"] += 1
    return ErmTicket(st, st.calls, st.t_shift, st.inv_scale, _stream(x.device))


def _ln_bwd_erm():
    from generative_recommenders.ops.flydsl.ln_bwd import ln_bwd_erm

    return ln_bwd_erm


def _empty_x_rm(m: int, device: torch.device) -> torch.Tensor:
    from generative_recommenders.ops.triton import fp8_tm_alloc

    return fp8_tm_alloc.empty_tm((m, D), torch.float8_e4m3fn, device)


def produce(
    site: ErmSite,
    tk: ErmTicket,
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    residual: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:


    try:
        x_rm = _empty_x_rm(int(x.shape[0]), x.device)
        dx, dw, db, g8, xrm, amaxp = _ln_bwd_erm()(
            dy, x, weight, bias, mean, rstd, residual, tk.inv_scale, tk.t_shift,
            out={"xrm": x_rm},
        )
    except Exception as error:
        set_broken(f"producer: {type(error).__name__}: {str(error)[:200]}")
        return None
    rec = ErmEmit(tk, dx, g8, xrm, amaxp)
    site.record = rec
    _counts["emitted"] += 1
    if "armed" not in _once:
        _once.add("armed")
        _arm_print(
            f"ARMED erm on {x.device}: output_projection dY from "
            f"ln_bwd_flydsl_d512_res_erm_x32_c3: e5m2 g8 {list(g8.shape)} + e4m3 x_rm "
            f"{list(xrm.shape)} + {int(amaxp.numel())} amax partials (bottom layer); "
            f"dual cast skipped (token {rec.token}, t_shift {tk.t_shift})"
        )
    return dx, dw, db


def run_producer(
    site: Optional[ErmSite],
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    residual: Optional[torch.Tensor],
    fold_dropout: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    tk = ticket(site, x, dy, residual, weight, bias, mean, rstd, fold_dropout)
    if tk is None:
        return None
    assert site is not None and residual is not None
    return produce(site, tk, dy, x, weight, bias, mean, rstd, residual)


def take(site: Optional[ErmSite]) -> Optional[ErmEmit]:
    if site is None:
        return None
    rec = site.record
    site.record = None
    site.consumed = True
    return rec


def _drop(rec: ErmEmit, why: str) -> None:
    rec.release()
    _counts["declined"] += 1
    _note("drop:" + why.split(" (")[0], f"erm record dropped ({why}) -> dual cast")
    set_broken(f"unexpected decline: {why}")


def consumable(
    rec: Optional[ErmEmit],
    state: Any,
    gradient: torch.Tensor,
    token_major: bool,
    t_e4m3: bool,
    t_tiled: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:


    if rec is None:
        return None
    tk = rec.ticket
    why: Optional[str] = None
    if rec.g8 is None or rec.used:
        why = "record released"
    elif tk.state is not state or not state.published:
        why = "site state replaced"
    elif state.calls != tk.calls:
        why = f"generation moved ({tk.calls} -> {state.calls})"
    elif state.t_shift != tk.t_shift:
        why = f"t_shift moved ({tk.t_shift} -> {state.t_shift})"
    elif not (token_major and t_e4m3):
        why = "consumer does not take the e4m3 token-major copy"
    elif not t_tiled:
        why = "consumer would not tile dout_t (knob-off W3 path is not x21)"
    elif (
        gradient.dtype != rec.dtype
        or int(gradient.data_ptr()) != rec.dx_ptr
        or int(gradient._version) != rec.dx_version
        or tuple(gradient.shape) != rec.shape
        or tuple(gradient.stride()) != rec.stride
    ):
        why = "dout is not the tensor the producer stored"
    elif _stream(gradient.device) != tk.stream:
        why = "stream differs"
    else:
        _, w, _, _ = _mods()
        r = w.rm_inactive_reason(gradient.device)
        if r is not None:
            why = f"W3 row-major entry not active since the emission ({r})"
    if why is not None:
        _drop(rec, why)
        return None
    payload = (rec.g8, rec.x_rm, rec.partials)
    rec.release()
    rec.used = True
    _counts["consumed"] += 1
    if "consumed" not in _once:
        _once.add("consumed")
        _arm_print(
            f"erm consumed on {gradient.device}: {getattr(state, 'site', 'output_projection')} "
            f"(generation {tk.calls}, t_shift {tk.t_shift})"
        )
    return payload


def finish(rec: Optional[ErmEmit]) -> None:
    if rec is None or rec.used:
        return
    if rec.g8 is None:
        return
    _drop(rec, "consumer did not reach the published fused cast")


def prebuild(device: torch.device) -> str:
    if not requested():
        return ""
    if _broken is not None:
        return "erm=off(latched)"
    why = _requested_line()
    if why is not None:
        return "erm=inert"
    try:
        m = 1024
        x = torch.zeros((m, D), dtype=torch.bfloat16, device=device)
        dy = torch.zeros_like(x)
        res = torch.zeros_like(x)
        w = torch.ones((D,), dtype=torch.bfloat16, device=device)
        b = torch.zeros((D,), dtype=torch.bfloat16, device=device)
        mean = torch.zeros((m,), dtype=torch.float32, device=device)
        rstd = torch.ones((m,), dtype=torch.float32, device=device)
        inv = torch.ones((1,), dtype=torch.float32, device=device)

        x_rm = torch.empty((m, D), dtype=torch.float8_e4m3fn, device=device)
        _ln_bwd_erm()(dy, x, w, b, mean, rstd, res, inv, 5, out={"xrm": x_rm})
    except Exception as error:
        set_broken(f"startup prebuild: {type(error).__name__}: {str(error)[:200]}")
        return "erm=error"
    mark_prebuilt()
    return "erm=1"


def _final() -> None:
    if not requested():
        return
    if any(_counts.values()) or _broken is not None:
        _arm_print(
            f"erm final=1 sites={_counts['sites']} claimed={_counts['claimed']} "
            f"tickets={_counts['tickets']} emitted={_counts['emitted']} "
            f"consumed={_counts['consumed']} declined={_counts['declined']} "
            f"latched={int(_broken is not None)} reasons={dict(sorted(_reasons.items()))}"
        )


def counters() -> Dict[str, int]:
    return dict(_counts)


def reset_for_testing() -> None:
    global _broken, _prebuilt, _inert, _inert_checked
    _broken = None
    _prebuilt = False
    _inert = None
    _inert_checked = False
    _once.clear()
    _reasons.clear()
    for k in _counts:
        _counts[k] = 0
