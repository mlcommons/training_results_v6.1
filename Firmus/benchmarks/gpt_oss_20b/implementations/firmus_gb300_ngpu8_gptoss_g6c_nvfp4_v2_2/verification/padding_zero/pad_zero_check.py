# Classification: Internal
"""Checks that every padded expert entry stays exactly zero during training.

Hooked into the image's unmodified pretrain.py by sitecustomize.py: after callback_logging and
callback_warmup are imported, their callbacks are replaced by subclasses that call the parent
method and run ``check``. Nothing else in the run changes.

Padded entries of each TEGroupedMLP expert (logical width H = F = 2880, physical width 2944):
  FC1 weight [2*2944, 2944]: rows of the padded FFN channels, and input columns H..2943 of the
                             other rows. With GLU interleave 32 and F a multiple of 32 the padded
                             channels are exactly rows 2F..2*2944-1 (checked at start-up).
  FC1 bias   [2*2944]:       entries 2F..2*2944-1
  FC2 weight [2944, 2944]:   output rows H..2943, and input columns F..2943 of the other rows
  FC2 bias   [2944]:         entries H..2943
For each, the check reads the BF16 parameter, its main_grad, and every optimizer state tensor of
the rank's distributed-optimizer shard as stored (exp_avg, exp_avg_sq, and the int16 remainder of
the FP32 master weight; the master weight is zero exactly when the BF16 parameter and the remainder
are both zero). Only the padded slices are copied to the host, so the check allocates almost no
device memory. Non-zero counts, negative zeros and the largest magnitude are reduced over all
ranks and printed by rank 0 as ``PADCHECK {json}``. As a control, the L2 norm of each whole
tensor is reported (all of it comes from the logical entries when the padded ones are zero), and
rank 0 reports how far expert 0's FC1 weight of every layer moved since the start of training.

Environment: PADCHECK_STEPS (comma-separated steps to check after), PADCHECK_STOP_STEP (stop the
run after this step).
"""

import importlib.abc
import importlib.machinery
import json
import os
import sys

STEPS = {int(s) for s in os.environ.get("PADCHECK_STEPS", "1,2,10,11,20,21,30").split(",") if s}
STOP_STEP = int(os.environ.get("PADCHECK_STOP_STEP", "30"))
PARAMS = ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias")
TENSORS = ("param", "main_grad", "master_param", "exp_avg", "exp_avg_sq")

_shards = None
_snapshot = {}
_layout_checked = False


def _is_pretrain():
    return bool(sys.argv) and os.path.basename(sys.argv[0]) == "pretrain.py"


def _rank():
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def _print0(tag, payload):
    if _rank() == 0:
        print(f"{tag} {json.dumps(payload, sort_keys=True)}", flush=True)


def _layout(kind, is_bias, shape, module):
    """Returns (rows, cols, first padded row, first padded column) of the flattened 2-D layout."""
    H = module.logical_hidden_size
    F = module.logical_moe_ffn_hidden_size
    if is_bias:
        return 1, shape[-1], 1, (2 * F if kind == "fc1" else H)
    rows, cols = shape[-2], shape[-1]
    return (rows, cols, 2 * F, H) if kind == "fc1" else (rows, cols, H, F)


def _check_fc1_layout(module, rows):
    """The padded FFN channels of FC1 (GLU interleave) must be exactly rows 2F.. ."""
    import torch

    F = module.logical_moe_ffn_hidden_size
    il = module.config.moe_mlp_glu_interleave_size
    channel = torch.arange(rows)
    if module.config.gated_linear_unit and il:
        channel = (channel // (2 * il)) * il + channel % il
    elif module.config.gated_linear_unit:
        channel = channel % (rows // 2)
    padded = channel >= F
    expected = torch.arange(rows) >= 2 * F
    if not torch.equal(padded, expected):
        raise RuntimeError("padded FC1 rows are not the tail rows 2F..; the check would be incomplete")
    return {"glu_interleave": il, "fc1_padded_rows": [2 * F, rows - 1], "logical_hidden": module.logical_hidden_size,
            "logical_ffn": F, "physical": module.padded_expert_hidden_size}


def _padded_slices(flat, start, end, rows, cols, row_pad, col_pad):
    """Padded entries of a 2-D [rows, cols] tensor whose flat range [start, end) is ``flat``."""
    out = []
    a, b = max(start, row_pad * cols), end
    if a < b:
        out.append(flat[a - start : b - start])
    first, last = start // cols, min(row_pad, -(-end // cols))
    r = first
    while r < last:
        lo, hi = max(start, r * cols + col_pad), min(end, (r + 1) * cols)
        if r * cols >= start and (r + 1) * cols <= end:
            n = min(last, end // cols) - r
            if n > 0:
                out.append(flat[r * cols - start : (r + n) * cols - start].view(n, cols)[:, col_pad:])
                r += n
                continue
        if lo < hi:
            out.append(flat[lo - start : hi - start])
        r += 1
    return out


def _slice_stats(slices, acc):
    """Adds non-zero count, negative zeros, element count and max |x| of the slices to ``acc``.

    Each slice is at most a few MB, so the temporaries fit in the little device memory left free.
    """
    import torch

    for s in slices:
        if s.numel() == 0:
            continue
        acc[0] += torch.count_nonzero(s)
        if s.is_floating_point():
            acc[1] += ((s == 0) & torch.signbit(s)).sum()
        acc[2] += s.numel()
        acc[3] = torch.maximum(acc[3], s.abs().amax().double())


def _expert_params(model):
    for chunk in model:
        for mname, module in chunk.named_modules():
            if not getattr(module, "padded_expert_hidden_size", 0):
                continue
            for kind in ("fc1", "fc2"):
                linear = getattr(module, f"linear_{kind}")
                for pname, param in linear.named_parameters():
                    is_bias = "bias" in pname.rsplit(".", 1)[-1]
                    yield mname, module, kind, is_bias, pname, param


def _flatten(optimizer):
    stack, out = [optimizer], []
    while stack:
        o = stack.pop()
        if hasattr(o, "chained_optimizers"):
            stack.extend(o.chained_optimizers)
        else:
            out.append(o)
    return out


def _build_shards(optimizer):
    """Maps id(model param) -> list of (distributed optimizer, shard param, attribute name)."""
    shards = {}
    for o in _flatten(optimizer):
        for shard_attr, model_attr in (
            ("shard_fp32_from_float16_groups", "model_float16_groups"),
            ("shard_float16_groups", "model_float16_groups"),
            ("shard_fp32_groups", "model_fp32_groups"),
        ):
            for mg, sg in zip(getattr(o, model_attr, None) or [], getattr(o, shard_attr, None) or []):
                for mp, sp in zip(mg, sg):
                    if sp is not None:
                        shards.setdefault(id(mp), []).append((o, sp, shard_attr))
    return shards


def check(context, tag, step):
    import torch
    import torch.distributed as dist

    global _shards, _layout_checked
    torch.cuda.synchronize()
    if not _shards:
        _shards = _build_shards(context.optimizer)
    keys = [(p, t) for p in PARAMS for t in TENSORS]
    # per key: non-zero, negative zero, checked, max |x|, whole-tensor L2^2
    acc = torch.zeros(len(keys), 5, dtype=torch.float64, device="cuda")
    row = {k: acc[i] for i, k in enumerate(keys)}
    modules = set()
    info = {}
    changes = []
    for mname, module, kind, is_bias, pname, param in _expert_params(context.model):
        modules.add(mname)
        pkind = f"{kind}_{'bias' if is_bias else 'weight'}"
        rows, cols, row_pad, col_pad = _layout(kind, is_bias, param.shape, module)
        if kind == "fc1" and not is_bias and not _layout_checked:
            info["layout"] = _check_fc1_layout(module, rows)
            _layout_checked = True
        numel = param.numel()
        tensors = [("param", param.detach().reshape(-1), 0, numel)]
        grad = getattr(param, "main_grad", None)
        if grad is not None:
            tensors.append(("main_grad", grad.detach().reshape(-1), 0, numel))
        for o, shard, shard_attr in _shards.get(id(param), []):
            inner = getattr(o, "optimizer", o)
            state = inner.state.get(shard)
            if not state:
                continue
            rng = o._get_model_param_range_map(param)["param"]
            for key, value in state.items():
                if torch.is_tensor(value) and value.numel() == rng.end - rng.start:
                    tensors.append((key, value.detach().reshape(-1), rng.start, rng.end))
            info.setdefault(
                "optimizer",
                {"optimizer": type(o).__name__, "inner": type(inner).__name__, "shard_attr": shard_attr,
                 "state": {k: str(v.dtype) for k, v in state.items() if torch.is_tensor(v)}},
            )
            break
        for name, flat, s, e in tensors:
            if (pkind, name) not in row:
                continue
            r = row[(pkind, name)]
            _slice_stats(_padded_slices(flat, s, e, rows, cols, row_pad, col_pad), r)
            if name in ("param", "main_grad"):
                r[4] += flat.norm().double() ** 2
        if pkind == "fc1_weight" and pname.endswith("weight0") and _rank() == 0:
            host = param.detach().to("cpu")
            if tag == "train_start":
                _snapshot[mname] = host.clone()
            elif mname in _snapshot:
                delta = (host.float() - _snapshot[mname].float()).abs()
                changes.append((float(delta[:row_pad, :col_pad].max()), float(delta[row_pad:].max()),
                                float(delta[:row_pad, col_pad:].max())))
    sums = acc[:, [0, 1, 2, 4]].contiguous()
    mx = acc[:, 3].contiguous()
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    world = dist.get_world_size()
    all_zero = True
    lines = []
    for i, (pkind, name) in enumerate(keys):
        nz, neg0, checked, l2sq = (float(v) for v in sums[i].cpu())
        if checked == 0:
            continue
        all_zero &= nz == 0
        entry = {"tensor": f"{pkind}.{name}", "nonzero": int(nz), "negative_zero": int(neg0),
                 "max_abs": float(mx[i]), "checked_all_ranks": int(checked)}
        if name in ("param", "main_grad"):
            entry["whole_tensor_l2_rank_mean"] = (l2sq / world) ** 0.5
        lines.append(entry)
    head = {"tag": tag, "step": step, "rank0_expert_modules": len(modules),
            "all_padded_entries_zero": bool(all_zero), "tensors_checked": len(lines)}
    if changes:
        head["rank0_fc1_weight0_max_abs_change_since_train_start"] = {
            "logical": max(c[0] for c in changes),
            "padded_rows": max(c[1] for c in changes),
            "padded_columns": max(c[2] for c in changes),
            "layers": len(changes),
        }
    _print0("PADCHECK", head)
    for entry in lines:
        _print0("PADCHECK_TENSOR", {"tag": tag, "step": step, **entry})
    if info:
        _print0("PADCHECK_INFO", info)
    return bool(all_zero)


def _patch_logging(mod):
    base = mod.MLPerfLoggingCallback

    class PadCheckLoggingCallback(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pad_all_zero = True
            self._pad_checks = 0

        def _check(self, context, tag, step):
            self._pad_all_zero &= check(context, tag, step)
            self._pad_checks += 1

        def on_train_start(self, context):
            super().on_train_start(context)
            self._check(context, "train_start", 0)

        def on_train_step_end(self, context):
            step = context.state.train_state.step + 1
            super().on_train_step_end(context)
            if step in STEPS or step == STOP_STEP:
                self._check(context, "step_end", step)
            if step == STOP_STEP:
                _print0("PADCHECK_SUMMARY", {"stop_step": step, "checks": self._pad_checks,
                                             "all_padded_entries_zero_at_every_check": bool(self._pad_all_zero)})
                mod.mllogger.end(
                    mod.mllogger.constants.RUN_STOP,
                    metadata={mod.mllogger.constants.SAMPLES_COUNT: self._get_samples_count(context.state),
                              "status": "aborted"},
                )
                # The training loop increments the step after this callback and runs while
                # step < train_iters.
                context.state.should_stop = True
                context.state.train_state.step = context.state.cfg.train.train_iters
                context.state.train_state.do_valid = False
                context.state.train_state.do_test = False

        def on_eval_start(self, context):
            self._check(context, "eval_start", context.state.train_state.step)
            super().on_eval_start(context)

        def on_eval_end(self, context):
            super().on_eval_end(context)
            self._check(context, "eval_end", context.state.train_state.step)

    mod.MLPerfLoggingCallback = PadCheckLoggingCallback


def _patch_warmup(mod):
    base = mod.WarmupCallback

    class PadCheckWarmupCallback(base):
        def on_data_init_start(self, context):
            check(context, "init_before_warmup", 0)
            super().on_data_init_start(context)

    mod.WarmupCallback = PadCheckWarmupCallback


_PATCHES = {"callback_logging": _patch_logging, "callback_warmup": _patch_warmup}


class _Loader(importlib.abc.Loader):
    def __init__(self, loader, patch):
        self._loader, self._patch = loader, patch

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _PATCHES or not _is_pretrain():
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None:
            spec.loader = _Loader(spec.loader, _PATCHES[name])
        return spec


def install():
    sys.meta_path.insert(0, _Finder())
