# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import collections
import hashlib
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.fx._symbolic_trace import is_fx_tracing

logger: logging.Logger = logging.getLogger(__name__)

API_VERSION = 2

ENV_ON = "DLRM_HSTU_FP8_M4"
ENV_EMA = "DLRM_HSTU_FP8_M4_EMA"
ENV_MIN_STEPS = "DLRM_HSTU_FP8_M4_MIN_STEPS"
ENV_OFFSETS = "DLRM_HSTU_FP8_M4_OFFSETS_MS"
ENV_FREEZE = "DLRM_HSTU_FP8_M4_FREEZE_AFTER"
ENV_SPAN = "DLRM_HSTU_FP8_M4_SPAN"
ENV_LOG_ALL = "DLRM_HSTU_FP8_M4_LOG_ALL"


COST_B = 38.157e-6
COST_C = 13.509e-9
COST_B_MS = COST_B
COST_C_MS = COST_C

CLAMP_LO = 0.9
CLAMP_HI = 1.1
DEFAULT_ALPHA = 0.5
DEFAULT_MIN_STEPS = 32
CLIP_FRAC = 0.025
STALL_FRAC = 0.10
STALE_MIN_STEPS = 8
DEFAULT_TIMEOUT_S = 600.0


MAX_LIVE = 64

MAX_DONE = 1 << 16

MAX_ACC = 1 << 16
_POOL_MAX = 2 * MAX_LIVE + 8

_ROW = 5

LOG_COUNTERS = ("incomplete", "orphan", "hook_error", "no_grad", "live_overflow")
_BROW = _ROW + 1 + len(LOG_COUNTERS)

_AROW = 5
FLAG_OK = 1.0
FLAG_OFF = 0.0
FLAG_ERR = 2.0
MAX_SPREAD = 4.0
LOG_MAX = 32
SPANS = ("mf", "stu")


_MEMBER_TAGS = (
    "_dlrmv4_ec_dgrad_port",
    "_dlrmv4_ec_indexed_l1",
    "_dlrmv4_uih_cand_premerged",
    "_dlrmv4_catfused",
)


class M4FactorTimeout(RuntimeError):
    pass


def anchor_costs(n: Any) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    return COST_B * n + COST_C * n * (n + 1.0) * 0.5


def requested() -> bool:
    return os.environ.get(ENV_ON, "0").strip() == "1"


def _env_alpha() -> float:
    raw = os.environ.get(ENV_EMA, "").strip()
    if not raw:
        return DEFAULT_ALPHA
    a = float(raw)
    if not (0.0 < a <= 1.0):
        raise ValueError(f"{ENV_EMA}={raw!r} must be in (0, 1]")
    return a


def _env_min_steps() -> int:
    raw = os.environ.get(ENV_MIN_STEPS, "").strip()
    if not raw:
        return DEFAULT_MIN_STEPS
    v = int(raw)
    if v < 1:
        raise ValueError(f"{ENV_MIN_STEPS}={raw!r} must be >= 1")
    return v


def _env_freeze() -> int:
    raw = os.environ.get(ENV_FREEZE, "").strip()
    if not raw:
        return 0
    v = int(raw)
    if v < 0:
        raise ValueError(f"{ENV_FREEZE}={raw!r} must be >= 0 (0 = off)")
    return v


def _env_span() -> str:
    raw = os.environ.get(ENV_SPAN, "").strip().lower()
    if raw in ("", "mf", "p_mf"):
        return "mf"
    if raw in ("stu", "p_stu"):
        return "stu"
    raise ValueError(f"{ENV_SPAN}={raw!r} must be 'mf' (default) or 'stu' (diagnostic)")


def parse_offsets(raw: Optional[str], w: int) -> Tuple[int, Optional[np.ndarray], str]:
    raw = (raw or "").strip()
    if not raw:
        return 0, None, ""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != int(w):
        return -1, None, f"{len(parts)} values for world size {int(w)}"
    try:
        vals = [float(p) for p in parts]
    except ValueError as e:
        return -1, None, f"not a list of floats ({e})"
    if not all(math.isfinite(v) and v >= 0.0 for v in vals):
        return -1, None, "every value must be finite and >= 0"
    return 1, np.array(vals, dtype=np.float64), ""


class _Rec:
    __slots__ = ("idx", "start", "end", "stream", "ended", "dead", "handles")

    def __init__(self, idx: int, start: Any, end: Any) -> None:
        self.idx = idx
        self.start = start
        self.end = end
        self.stream: Any = None
        self.ended = False
        self.dead = False
        self.handles: List[Any] = []


class _State:
    def __init__(self) -> None:
        self.cond = threading.Condition()


        self.local_off: Optional[str] = None
        self.global_off: Optional[str] = None
        self.off_from: Optional[int] = None

        self.armed: Optional[bool] = None
        self.inert = False
        self.stopped: Optional[str] = None
        self.installed = False
        self.hooked: Optional[str] = None
        self.span = "mf"
        self.require_cuda = True
        self.hooks_active = False
        self.remove_pending = False
        self.module_handles: List[Any] = []

        self.step_idx = 0
        self.live: Deque[_Rec] = collections.deque()
        self.done: List[Tuple[int, float]] = []
        self.pool: List[Any] = []
        self.counters: Dict[str, int] = collections.Counter()
        self.cnt_snap: Dict[str, int] = {}
        self.maxmed = float("nan")

        self.acc: Deque[float] = collections.deque(maxlen=MAX_ACC)
        self.ref: Optional[float] = None

        self.costs: Dict[int, np.ndarray] = {}
        self.noted_k = False
        self.published: Dict[int, Optional[np.ndarray]] = {}
        self.offsets_pub: Dict[int, Optional[np.ndarray]] = {}
        self.last_k = -1
        self.callbacks: List[Callable[[int], None]] = []

        self.boundary = 0
        self.ema: Optional[List[float]] = None
        self.n_valid = 0
        self.frozen: Optional[List[float]] = None
        self.alpha = DEFAULT_ALPHA
        self.min_steps = DEFAULT_MIN_STEPS
        self.freeze_after = 0
        self.log_all = False
        self.offsets: Optional[np.ndarray] = None
        self.offsets_desc = "none"
        self.last_rows: Optional[List[List[float]]] = None
        self.warned: set = set()


_S = _State()


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0") or "0")


def _world() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _print_all_ranks(msg: str) -> None:
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks
    except Exception:
        print(f"{msg} [rank {_rank()}]", flush=True)
        return
    arm_print_all_ranks(msg)


def _vec_sha(v: Optional[np.ndarray]) -> str:
    if v is None:
        return "-"
    return hashlib.sha256(np.ascontiguousarray(v, dtype="<f8").tobytes()).hexdigest()[:16]


def _hash48(b: bytes) -> float:
    return float(int.from_bytes(hashlib.sha256(b).digest()[:6], "little"))


def _config_hash(st: _State) -> float:
    s = (
        f"m4v{API_VERSION}|span={st.span}|alpha={st.alpha!r}|min={st.min_steps}|"
        f"freeze={st.freeze_after}|clip={CLIP_FRAC!r}|stall={STALL_FRAC!r}|"
        f"clamp={CLAMP_LO!r},{CLAMP_HI!r}|b={COST_B!r}|c={COST_C!r}"
    )
    return _hash48(s.encode())


def _median(v: Sequence[float]) -> float:
    return float(np.median(np.asarray(v, dtype=np.float64)))


def _warn_once(st: _State, key: str, msg: str, *args: Any) -> None:
    if key not in st.warned:
        st.warned.add(key)
        logger.warning(msg, *args)


def _take_event(st: _State) -> Any:
    if st.pool:
        return st.pool.pop()

    return torch.cuda.Event(enable_timing=True)


def _recycle(st: _State, *events: Any) -> None:
    for ev in events:
        if len(st.pool) < _POOL_MAX:
            st.pool.append(ev)


def _in_backward() -> bool:
    gt = getattr(torch._C, "_current_graph_task_id", None)
    if gt is not None:
        try:
            return gt() != -1
        except Exception:
            pass
    return threading.current_thread() is not threading.main_thread()


def _is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _drop_rec(st: _State, rec: _Rec, recycle: bool) -> None:
    for h in rec.handles:
        try:
            h.remove()
        except Exception:
            pass
    rec.handles = []
    if recycle:
        _recycle(st, rec.start, rec.end)


def _harvest(st: _State, final: bool = False) -> None:
    live = st.live
    while live:
        rec = live[0]
        if rec.dead:
            live.popleft()
            st.counters["hook_error"] += 1
            _drop_rec(st, rec, False)
            continue
        if not rec.ended:
            if final or len(live) > 1:
                live.popleft()
                st.counters["orphan"] += 1
                _drop_rec(st, rec, False)
                continue
            return
        if not (rec.start.query() and rec.end.query()):
            if final:
                live.popleft()
                st.counters["incomplete"] += 1
                _drop_rec(st, rec, False)
                continue
            return
        ms = float(rec.start.elapsed_time(rec.end))
        live.popleft()
        if len(st.done) < MAX_DONE:
            st.done.append((rec.idx, ms))
        else:
            st.counters["done_overflow"] += 1
        _drop_rec(st, rec, True)


def _make_end_hook(rec: _Rec) -> Callable[[Any], None]:
    def _m4_end_hook(_grads: Any) -> None:
        if rec.ended or rec.dead:
            return None
        if _S.hooks_active:
            try:
                if rec.stream is not None:
                    rec.end.record(rec.stream)
                else:
                    rec.end.record()
                rec.ended = True
            except Exception:
                rec.dead = True
        return None

    return _m4_end_hook


def _register_end_hooks(rec: _Rec, tensors: List[torch.Tensor]) -> None:
    fn = _make_end_hook(rec)
    if len(tensors) > 1:
        multi = getattr(torch.autograd.graph, "register_multi_grad_hook", None)
        if multi is not None:
            rec.handles.append(multi(list(tensors), fn, mode="all"))
            return
    rec.handles.append(tensors[0].register_hook(fn))


def _begin_step(st: _State, tensors: List[torch.Tensor]) -> None:
    _harvest(st)
    idx = st.step_idx
    st.step_idx = idx + 1
    if not tensors:
        st.counters["no_grad"] += 1
        return
    if st.require_cuda and not all(t.is_cuda for t in tensors):
        st.counters["not_cuda"] += 1
        return
    if len(st.live) >= MAX_LIVE:
        _drop_rec(st, st.live.popleft(), False)
        st.counters["live_overflow"] += 1
    rec = _Rec(idx, _take_event(st), _take_event(st))
    try:
        if st.require_cuda:
            rec.stream = torch.cuda.current_stream()
            rec.start.record(rec.stream)
        else:
            rec.start.record()
        _register_end_hooks(rec, tensors)
    except Exception as e:
        rec.dead = True
        _drop_rec(st, rec, False)
        st.counters["hook_error"] += 1
        _warn_once(st, "hook_error", "[m4] speed estimator: step not measured (%s: %s)",
                   type(e).__name__, e)
        return
    st.live.append(rec)


def _member_tagged(t: torch.Tensor) -> bool:
    return any(getattr(t, a, None) is not None for a in _MEMBER_TAGS)


def main_forward_begin(module: torch.nn.Module, seq_embeddings: Any) -> None:
    st = _S
    if not st.hooks_active or st.span != "mf":
        return None
    if not module.training or not torch.is_grad_enabled():
        return None
    if _is_compiling() or is_fx_tracing() or _in_backward():
        return None
    cfg = getattr(module, "_hstu_configs", None)
    names = getattr(cfg, "item_embedding_feature_names", None)
    if not names:
        st.counters["no_item_names"] += 1
        return None
    tensors: List[torch.Tensor] = []
    for c in names:
        try:
            t = getattr(seq_embeddings[c], "embedding", None)
        except Exception:
            return None
        if not isinstance(t, torch.Tensor):
            return None
        if t.requires_grad:
            if _member_tagged(t):
                st.counters["member_skip"] += 1
            else:
                tensors.append(t)
    _begin_step(st, tensors)
    return None


def _stu_pre_hook(module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    st = _S
    if not st.hooks_active or st.span != "stu":
        return None
    if not module.training or not torch.is_grad_enabled():
        return None
    if _is_compiling() or is_fx_tracing() or _in_backward():
        return None
    x = kwargs.get("x") if kwargs else None
    if x is None and args:
        x = args[0]
    if not isinstance(x, torch.Tensor):
        return None
    _begin_step(st, [x] if x.requires_grad else [])
    return None


def _remove_hooks(st: _State) -> None:
    st.hooks_active = False
    for h in st.module_handles:
        try:
            h.remove()
        except Exception:
            pass
    st.module_handles = []
    for rec in st.live:
        rec.dead = True
        _drop_rec(st, rec, False)
    st.live.clear()
    st.done = []
    st.remove_pending = False


def install(model: torch.nn.Module, require_cuda: bool = True) -> bool:
    st = _S
    if not requested():
        return False
    if st.local_off is not None or st.global_off is not None or st.armed is False:
        return False
    if st.installed:
        return True
    st.span = _env_span()
    st.alpha = _env_alpha()
    st.min_steps = _env_min_steps()
    st.freeze_after = _env_freeze()
    st.log_all = os.environ.get(ENV_LOG_ALL, "0").strip() == "1"
    if require_cuda and not torch.cuda.is_available():
        set_enabled(False, "CUDA not available")
        return False
    if st.span == "mf":
        found = None
        for name, m in model.named_modules():
            fn = getattr(type(m), "main_forward", None)
            if callable(fn):
                found = (name, m, fn)
                break
        if found is None:
            set_enabled(False, "no module with a main_forward (DlrmHSTU) in the model")
            return False
        name, m, fn = found
        code = getattr(fn, "__code__", None)
        if code is None or "main_forward_begin" not in code.co_names:
            set_enabled(
                False,
                f"{type(m).__name__}.main_forward does not call "
                "m4_speed.main_forward_begin (dlrm_hstu.py hunk missing)",
            )
            return False
        st.hooked = f"{name or '<model>'}.main_forward ({type(m).__name__})"
    else:
        found = None
        for name, m in model.named_modules():
            layers = getattr(m, "_stu_layers", None)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                found = (name, m, layers)
                break
        if found is None:
            set_enabled(False, "no STU stack (_stu_layers) in the model")
            return False
        name, stack, layers = found
        st.module_handles.append(
            layers[0].register_forward_pre_hook(_stu_pre_hook, with_kwargs=True)
        )
        st.hooked = (
            f"{name or '<model>'}._stu_layers[0] ({type(layers[0]).__name__}) of a "
            f"{len(layers)}-layer {type(stack).__name__}"
        )
        if len(layers) < 2:
            logger.warning(
                "[m4] speed estimator (span=stu): 1-layer stack; with the "
                "targets-only last layer the layer-0 pre-hook never fires"
            )
    st.require_cuda = bool(require_cuda)
    st.installed = True
    st.hooks_active = True
    if _rank() == 0:
        logger.info(
            "[m4] speed estimator installed on %s (span=%s); arms at the first "
            "window boundary",
            st.hooked,
            "P_mf" if st.span == "mf" else "P_stu (diagnostic)",
        )
    return True


def enabled() -> bool:
    st = _S
    return (
        requested()
        and not st.inert
        and st.local_off is None
        and st.global_off is None
        and st.armed is not False
    )


def set_enabled(on: bool, reason: str = "") -> bool:
    if on:
        return enabled()
    st = _S
    with st.cond:
        first = st.local_off is None
        if first:
            st.local_off = reason or "set_enabled(False)"
        st.hooks_active = False
        st.remove_pending = True
    if threading.current_thread() is threading.main_thread():
        _remove_hooks(st)
    if first and requested() and not st.inert and st.armed is not False:
        _print_all_ranks(
            f"[m4] speed estimator disabled on this rank: {st.local_off} "
            + (
                "(all ranks go off at the next window boundary)"
                if st.armed
                else "(all ranks stay OFF: decided at the first window boundary)"
            )
        )
    return False


def shutdown(reason: str = "shutdown") -> None:
    st = _S
    with st.cond:
        if st.stopped is None:
            st.stopped = reason
        st.cond.notify_all()


def register_boundary_callback(fn: Callable[[int], None]) -> None:
    if not callable(fn):
        raise TypeError("m4_speed.register_boundary_callback needs a callable")
    with _S.cond:
        _S.callbacks.append(fn)


def note_window_costs(j: int, costs: Any) -> None:
    st = _S
    if st.inert or not requested() or st.global_off is not None:
        return
    c = np.array(costs, dtype=np.float64, copy=True).reshape(-1)
    if c.size and (not np.all(np.isfinite(c)) or np.any(c < 0.0)):
        raise ValueError(f"m4_speed.note_window_costs(j={j}): invalid costs")
    with st.cond:
        if int(j) < st.boundary - 1:
            st.counters["late_costs"] += 1
            return
        st.costs[int(j)] = c


def _wait_decided(k: int, timeout_s: float, what: str) -> Optional[np.ndarray]:
    st = _S
    k = int(k)
    if k < 0 or st.inert or not requested():
        return None
    deadline = time.monotonic() + float(timeout_s)
    with st.cond:
        while True:
            if k in st.published:
                v = (st.published if what == "factors" else st.offsets_pub).get(k)
                return None if v is None else v.copy()
            if st.off_from is not None and k >= st.off_from:
                return None
            if st.stopped is not None:
                return None
            if threading.current_thread() is threading.main_thread():
                raise RuntimeError(
                    f"m4_speed.{what}_for({k}) on the main thread before boundary {k} "
                    f"is decided (last decided k={st.last_k}): the main thread decides "
                    "it at the barrier of window k+1, so waiting here would deadlock"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise M4FactorTimeout(
                    f"m4_speed.{what}_for({k}) timed out after {float(timeout_s):.0f} s: "
                    f"boundary {k} not decided (last decided k={st.last_k}, "
                    f"boundaries seen={st.boundary}); kill switch {ENV_ON}=0"
                )
            st.cond.wait(remaining)


def factors_for(k: int, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[np.ndarray]:
    return _wait_decided(k, timeout_s, "factors")


def offsets_for(k: int, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[np.ndarray]:
    return _wait_decided(k, timeout_s, "offsets")


def decide(
    k: int,
    rows: List[List[float]],
    ema_prev: Optional[List[float]],
    alpha: float,
    min_steps: int,
) -> Tuple[bool, bool, Optional[List[float]], Optional[List[float]]]:


    if not rows or any(len(r) != _ROW for r in rows):
        raise RuntimeError(f"m4_speed: malformed gathered rows {rows!r}")
    if any(r[4] != FLAG_OK for r in rows):
        return True, False, ema_prev, None
    valid = k >= 1 and all(
        r[1] >= min_steps and math.isfinite(r[0]) and r[0] > 0.0 for r in rows
    )
    ema = ema_prev
    if valid:
        w = len(rows)
        mean = math.fsum(r[0] for r in rows) / w
        v = [r[0] / mean for r in rows]
        if ema_prev is None or len(ema_prev) != w:
            ema = v
        else:
            ema = [alpha * a + (1.0 - alpha) * b for a, b in zip(v, ema_prev)]
    pub = None
    if ema is not None:
        m = math.fsum(ema) / len(ema)
        pub = [min(CLAMP_HI, max(CLAMP_LO, e / m)) for e in ema]
    return False, valid, ema, pub


def _local_row(st: _State, k: int) -> List[float]:
    _harvest(st, final=True)
    done = st.done
    with st.cond:
        costs = st.costs.pop(k, None)
        for old in [j for j in st.costs if j < k]:
            del st.costs[old]
    st.noted_k = costs is not None
    n_costs = 0 if costs is None else int(costs.shape[0])
    st.maxmed = float("nan")
    if k >= 1 and n_costs:
        rs = [
            ms / float(costs[idx])
            for idx, ms in done
            if 1 <= idx < n_costs and costs[idx] > 0.0 and ms > 0.0
        ]
        if rs:
            st.maxmed = max(rs) / _median(rs)
        if st.ref is not None and rs:
            ref = st.ref
            keep = [r for r in rs if abs(r / ref - 1.0) <= STALL_FRAC]
            if len(rs) >= STALE_MIN_STEPS and 2 * len(keep) < len(rs):
                st.counters["stall_ref_stale"] += 1
                keep = rs
            st.counters["stall"] += len(rs) - len(keep)
            rs = keep
        if rs:
            m = _median(rs)
            lo, hi = m * (1.0 - CLIP_FRAC), m * (1.0 + CLIP_FRAC)
            st.counters["clipped"] += sum(1 for r in rs if r < lo or r > hi)
            st.acc.extend(min(hi, max(lo, r)) for r in rs)
    n_acc = len(st.acc)
    stat = _median(st.acc) if n_acc else float("nan")
    return [stat, float(n_acc), float(len(done)), float(n_costs), FLAG_OK]


def _log_fields(st: _State) -> List[float]:
    snap = st.cnt_snap
    vals = [st.maxmed] + [
        float(st.counters.get(c, 0) - snap.get(c, 0)) for c in LOG_COUNTERS
    ]
    st.cnt_snap = dict(st.counters)
    return vals


def _sanitize(pub: Optional[List[float]], w: int) -> str:
    if pub is None:
        return ""
    if len(pub) != w:
        return f"length {len(pub)} != world {w}"
    if not all(math.isfinite(x) and x > 0.0 for x in pub):
        return "non-finite or non-positive entry"
    if max(pub) > MAX_SPREAD * min(pub):
        return f"spread > {MAX_SPREAD:g}x"
    return ""


def _reset_window(st: _State) -> None:
    for rec in st.live:
        rec.dead = True
        _drop_rec(st, rec, False)
    st.live.clear()
    st.done = []
    st.step_idx = 0


def _all_gather_rows(vals: List[float], device: Optional[torch.device]) -> List[List[float]]:
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return [list(vals)]
    w = dist.get_world_size()
    if w <= 1:
        return [list(vals)]
    dev = torch.device(device) if device is not None else torch.device("cpu")
    host = torch.tensor(vals, dtype=torch.float64)
    if dev.type == "cuda":

        inp = host.pin_memory().to(dev, non_blocking=True)
    else:
        inp = host
    out = torch.empty(w * len(vals), dtype=torch.float64, device=dev)
    dist.all_gather_into_tensor(out, inp)

    return out.view(w, len(vals)).tolist()


def _global_off(st: _State, i: int, reason: str) -> None:
    with st.cond:
        if st.global_off is None:
            st.global_off = reason
        st.off_from = max(0, i - 1)
        st.hooks_active = False
        st.cond.notify_all()
    _remove_hooks(st)
    st.acc.clear()


def on_window_boundary(device: Optional[torch.device] = None) -> None:
    st = _S
    if st.inert:
        return
    if st.armed is None and not requested():
        st.inert = True
        st.armed = False
        return
    i = st.boundary
    st.boundary = i + 1
    if i == 0:
        _arm(st, device)
        return
    if st.armed and st.global_off is None:
        _close_window(st, i, device)
    elif st.remove_pending or st.live:
        _remove_hooks(st)
    if st.armed:


        _run_callbacks(st, i)


def _arm(st: _State, device: Optional[torch.device]) -> None:
    w = _world()
    if st.local_off is not None:
        why = st.local_off
    elif not st.installed:
        why = "no hook point (install() not called or it declined)"
    else:
        why = ""
    ostat, ovec, oreason = parse_offsets(os.environ.get(ENV_OFFSETS, ""), w)
    ohash = _hash48(np.ascontiguousarray(ovec, dtype="<f8").tobytes()) if ovec is not None else 0.0
    row = [_config_hash(st), ohash, float(ostat), float(w), FLAG_OFF if why else FLAG_OK]
    rows = _all_gather_rows(row, device)
    if not rows or any(len(r) != _AROW for r in rows):
        raise RuntimeError(f"m4_speed: malformed arming rows {rows!r}")
    bad = [r for r, x in enumerate(rows) if x[4] != FLAG_OK]
    if bad or len({x[0] for x in rows}) > 1:
        if bad:
            reason = f"not armed on rank(s) {bad}" + (f" (this rank: {why})" if why else "")
        else:
            reason = f"diagnostic M4 knobs differ across ranks (config hashes {[int(x[0]) for x in rows]})"
        st.armed = False
        _global_off(st, 0, reason)
        _print_all_ranks(f"[m4] speed estimator OFF: {reason}")
        return
    st.armed = True
    stats = [int(x[2]) for x in rows]
    if all(s == 0 for s in stats):
        st.offsets, st.offsets_desc = None, "none"
    elif all(s == 1 for s in stats) and len({x[1] for x in rows}) == 1 and ovec is not None:
        st.offsets, st.offsets_desc = ovec, _vec_sha(ovec)
    else:
        st.offsets, st.offsets_desc = None, "refused"
        _print_all_ranks(
            f"[m4] offsets REFUSED ({ENV_OFFSETS}): status by rank {stats} "
            "(1 valid, 0 unset, -1 invalid)"
            + ("; values differ across ranks" if all(s == 1 for s in stats) else "")
            + f"; this rank: {oreason or 'valid'}; factors are published without offsets"
        )
    _reset_window(st)
    st.cnt_snap = dict(st.counters)
    _print_all_ranks(
        "armed OK: [m4] speed estimator ON span="
        + ("P_mf" if st.span == "mf" else "P_stu (DIAGNOSTIC)")
        + f" (API v{API_VERSION}): {st.hooked}; E_end "
        + ("when all item-candidate embedding grads are final" if st.span == "mf"
           else "at the grad of the layer-0 input")
        + f"; world={w} min_steps={st.min_steps} ema_alpha={st.alpha:g} "
        f"clamp=[{CLAMP_LO:g},{CLAMP_HI:g}] clip={CLIP_FRAC:g} stall={STALL_FRAC:g} "
        f"freeze={st.freeze_after or 'off'} offsets={st.offsets_desc}; factors from "
        f"boundary k>=1; kill switch {ENV_ON}=0"
    )
    _run_callbacks(st, 0)


def _close_window(st: _State, i: int, device: Optional[torch.device]) -> None:
    k = i - 1
    if st.remove_pending:
        _remove_hooks(st)
    try:
        row = _local_row(st, k)
    except Exception as e:
        row = [float("nan"), 0.0, 0.0, 0.0, FLAG_ERR]
        _warn_once(st, "row_error", "[m4] speed estimator: window k=%d measurement "
                   "failed on this rank (%s: %s); M4 goes off on every rank", k,
                   type(e).__name__, e)
    if st.local_off is not None:
        row[4] = FLAG_OFF
    row = row + _log_fields(st)
    _reset_window(st)
    rows = _all_gather_rows(row, device)
    if not rows or any(len(r) != _BROW for r in rows):
        raise RuntimeError(f"m4_speed: malformed gathered rows {rows!r}")
    st.last_rows = rows
    off, valid, ema, pub = decide(k, [r[:_ROW] for r in rows], st.ema, st.alpha, st.min_steps)
    if off:
        dis = [r for r, x in enumerate(rows) if x[4] == FLAG_OFF]
        err = [r for r, x in enumerate(rows) if x[4] not in (FLAG_OK, FLAG_OFF)]
        reason = "; ".join(
            x for x in (
                f"disabled on rank(s) {dis}" if dis else "",
                f"measurement error on rank(s) {err}" if err else "",
            ) if x
        )
        reason = f"{reason} before boundary k={k}"
        _global_off(st, i, reason)
        _print_all_ranks(f"[m4] speed estimator OFF from window i={i}: {reason}")
        return
    st.ema = ema
    if valid:
        st.n_valid += 1
        st.ref = rows[_rank()][0] if len(rows) > 1 else rows[0][0]
        st.acc.clear()
        if st.frozen is None and st.freeze_after > 0 and st.n_valid >= st.freeze_after:
            st.frozen = pub
    if st.frozen is not None:
        pub = st.frozen
    bad = _sanitize(pub, len(rows))
    if bad:
        st.counters["unpublishable"] += 1
        if _rank() == 0:
            logger.warning("[m4] WARNING: factors k=%d not publishable (%s): %r -> None", k, bad, pub)
        pub = None
    vec = None if pub is None else np.array(pub, dtype=np.float64)
    offs = None if (vec is None or st.offsets is None) else st.offsets.copy()
    with st.cond:
        st.published[k] = vec
        st.offsets_pub[k] = offs
        st.last_k = k
        st.cond.notify_all()
    if _rank() == 0:
        _log_boundary(st, k, valid, vec, offs, rows)


def _fmt(vals: Sequence[float], fmt: str = "%.4f") -> str:
    v = list(vals)
    txt = ",".join("nan" if not math.isfinite(x) else fmt % x for x in v[:LOG_MAX])
    return "[" + txt + (",..." if len(v) > LOG_MAX else "") + "]"


def _log_boundary(
    st: _State,
    k: int,
    valid: bool,
    vec: Optional[np.ndarray],
    offs: Optional[np.ndarray],
    rows: List[List[float]],
) -> None:
    n = [int(r[1]) for r in rows]
    meas = [int(r[2]) for r in rows]
    nc = [int(r[3]) for r in rows]
    cnt = ""
    for j, c in enumerate(LOG_COUNTERS):
        per = [int(r[_ROW + 1 + j]) for r in rows]
        if any(per):
            cnt += " %s=%s" % (c, per[:LOG_MAX])
    extra = ""
    if st.log_all:
        extra = " n_r=%s meas=%s costs=%s" % (n[:LOG_MAX], meas[:LOG_MAX], nc[:LOG_MAX])
    logger.warning(
        "[m4] factors k=%d valid=%d n=%d..%d min=%s max=%s vec=%s offsets=%s "
        "nvalid=%d frozen=%d stat=%s maxmed=%s pub=%s%s%s",
        k,
        int(valid),
        min(n),
        max(n),
        "-" if vec is None else "%.4f" % float(vec.min()),
        "-" if vec is None else "%.4f" % float(vec.max()),
        _vec_sha(vec),
        "none" if offs is None else _vec_sha(offs),
        st.n_valid,
        int(st.frozen is not None),
        _fmt([r[0] for r in rows], "%.5f"),
        _fmt([r[_ROW] for r in rows], "%.3f"),
        "-" if vec is None else _fmt(vec.tolist(), "%.5f"),
        cnt,
        extra,
    )
    if k >= 1 and max(meas) > 0 and max(nc) == 0:
        if st.noted_k:


            if "empty_costs" not in st.warned:
                st.warned.add("empty_costs")
                logger.info("[m4] window k=%d: empty costs noted (no sample; expected "
                            "beyond the N table / resumed partial window)", k)
        else:
            _warn_once(st, "no_costs", "[m4] WARNING: window k=%d trained but "
                       "note_window_costs was never called for it (a path without an "
                       "M4 dealer must call m4_speed.set_enabled(False, reason))", k)
    if min(nc) == 0 < max(nc):
        _warn_once(st, "costs_partial", "[m4] WARNING: window k=%d: costs noted on some "
                   "ranks only %s (the dealer path should be rank-identical)", k, nc)
    if k >= 1 and max(nc) > 0 and max(meas) == 0:
        _warn_once(st, "no_steps", "[m4] WARNING: window k=%d has costs but no measured "
                   "step on any rank (span end hooks not reached?)", k)
    if any(m != c for m, c in zip(meas, nc) if m > 0 and c > 0):
        _warn_once(st, "mismatch", "[m4] WARNING: window k=%d measured steps %s != cost "
                   "entries %s on some rank (j<->k mapping, an extra training forward, "
                   "or lost samples)", k, meas[:LOG_MAX], nc[:LOG_MAX])


def _run_callbacks(st: _State, i: int) -> None:
    with st.cond:
        cbs = list(st.callbacks)
    for fn in cbs:
        fn(i)


def status() -> Dict[str, Any]:
    st = _S
    with st.cond:
        return {
            "requested": requested(),
            "enabled": enabled(),
            "armed": st.armed,
            "inert": st.inert,
            "local_off": st.local_off,
            "global_off": st.global_off,
            "off_from": st.off_from,
            "stopped": st.stopped,
            "installed": st.installed,
            "hooked": st.hooked,
            "span": st.span,
            "n_module_hooks": len(st.module_handles),
            "hooks_active": st.hooks_active,
            "boundary": st.boundary,
            "last_k": st.last_k,
            "n_valid": st.n_valid,
            "frozen": None if st.frozen is None else list(st.frozen),
            "published": {
                k: (None if v is None else v.copy()) for k, v in st.published.items()
            },
            "offsets_published": {
                k: (None if v is None else v.copy()) for k, v in st.offsets_pub.items()
            },
            "offsets": None if st.offsets is None else st.offsets.copy(),
            "offsets_desc": st.offsets_desc,
            "ema": None if st.ema is None else list(st.ema),
            "ref": st.ref,
            "acc": len(st.acc),
            "counters": dict(st.counters),
            "live": len(st.live),
            "live_handles": sum(len(r.handles) for r in st.live),
            "done": len(st.done),
            "costs_pending": sorted(st.costs),
            "callbacks": len(st.callbacks),
            "params": {
                "alpha": st.alpha,
                "min_steps": st.min_steps,
                "freeze_after": st.freeze_after,
                "log_all": st.log_all,
            },
        }


def _reset_for_tests() -> None:
    global _S
    old = _S
    _remove_hooks(old)
    with old.cond:
        old.stopped = old.stopped or "reset"
        old.cond.notify_all()
    _S = _State()
