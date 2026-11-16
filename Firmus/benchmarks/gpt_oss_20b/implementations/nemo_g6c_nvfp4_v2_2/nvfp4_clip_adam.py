# Copyright (c) 2026, Firmus Technologies. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Classification: Internal
"""Fold the gradient-clip scale into the FusedAdam step.

`defer_clip(grads, coeff)` records the clip coefficient instead of scaling the
gradients. `install(fused_adam)` wraps every FusedAdam multi-tensor Adam op:
the fp32 and bf16+remainder ops scale each recorded gradient as they load it,
the others scale recorded gradients before running. Gradients that no Adam op
reads are scaled by `finish()` after the optimizer step.
"""

import importlib.util
import os

import torch

EXT_NAME = "nvfp4_clip_adam_ext"
_ext = None
_meta_cache = {}
_pending = {"scale": None, "grads": {}}
counters = {"fused": 0, "fallback": 0, "leftover": 0, "l2norm": 0, "l2norm_early": 0,
            "l2_prefetch": 0}


def build(build_dir, name=EXT_NAME, cuda_flags=("-O3",), verbose=True):
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_adam_ext.cu")
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name=name,
        sources=[src],
        build_directory=build_dir,
        extra_cuda_cflags=list(cuda_flags),
        verbose=verbose,
    )


def load_prebuilt(path):
    global _ext
    name = os.path.basename(path).split(".")[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ext = mod
    return mod


def set_ext(mod):
    global _ext
    _ext = mod


def set_max_ctas(n):
    """Cap the grid of later fused Adam launches; returns False if the extension can't."""
    if not hasattr(_ext, "set_max_ctas"):
        return False
    _ext.set_max_ctas(int(n))
    return True


def _meta(lists, device):
    key = tuple((t.data_ptr(), t.numel()) for lst in lists for t in lst)
    hit = _meta_cache.get(key)
    if hit is not None:
        return hit
    g = lists[0]
    chunk = int(_ext.CHUNK)
    ptrs, sizes, bt, bc = [], [], [], []
    for i in range(len(g)):
        n = g[i].numel()
        ptrs.extend(lst[i].data_ptr() for lst in lists)
        sizes.append(n)
        c = (n + chunk - 1) // chunk
        bt.extend([i] * c)
        bc.extend(range(c))
    meta = (
        torch.tensor(ptrs, dtype=torch.int64).to(device),
        torch.tensor(sizes, dtype=torch.int64).to(device),
        torch.tensor(bt, dtype=torch.int32).to(device),
        torch.tensor(bc, dtype=torch.int32).to(device),
    )
    if len(_meta_cache) > 64:
        _meta_cache.clear()
    _meta_cache[key] = meta
    return meta


_GRAD_DTYPES = (torch.float32, torch.bfloat16)
_DTYPES = {
    "plain": (_GRAD_DTYPES, torch.float32, torch.float32, torch.float32),
    "remainder": (_GRAD_DTYPES, torch.bfloat16, torch.float32, torch.float32, torch.int16),
}
_FNS = {
    ("plain", torch.float32): "clip_adam",
    ("plain", torch.bfloat16): "clip_adam_bf16g",
    ("remainder", torch.float32): "clip_adam_rem",
    ("remainder", torch.bfloat16): "clip_adam_rem_bf16g",
}


def _fusable(kind, lists):
    dtypes = _DTYPES.get(kind)
    if dtypes is None or len(lists) != len(dtypes):
        return False
    gdt = lists[0][0].dtype
    if gdt not in dtypes[0] or not hasattr(_ext, _FNS[(kind, gdt)]):
        return False
    for lst, dt in zip(lists, (gdt,) + dtypes[1:]):
        for t in lst:
            if t.dtype != dt or not t.is_contiguous() or not t.is_cuda:
                return False
    n = len(lists[0])
    return all(lists[0][i].numel() == lst[i].numel() for lst in lists[1:] for i in range(n))


# Optional fold of the updated params' amax into per-weight slots (remainder kernel only):
# `slot_of(param_tensor)` gives each tensor's slot or -1; `covered` collects the slots written.
_amax = {"buf": None, "slot_of": None, "covered": set(), "cache": {}}


def set_amax_target(amax_buf, slot_of):
    if slot_of is not _amax.get("last_slot_of"):
        _amax["cache"].clear()
        _amax["last_slot_of"] = slot_of
    _amax.update(buf=amax_buf, slot_of=slot_of, covered=set())


def clear_amax_target():
    """Stop folding; returns the set of slots folded since set_amax_target."""
    covered = _amax["covered"]
    _amax.update(buf=None, slot_of=None, covered=set())
    return covered


def _tensor_slots(params, device):
    key = tuple((t.data_ptr(), t.numel()) for t in params)
    hit = _amax["cache"].get(key)
    if hit is None:
        slots = [_amax["slot_of"](t) for t in params]
        hit = (torch.tensor(slots, dtype=torch.int32).to(device), {s for s in slots if s >= 0})
        if len(_amax["cache"]) > 64:
            _amax["cache"].clear()
        _amax["cache"][key] = hit
    return hit


def fused_adam(kind, lists, scale, lr, beta1, beta2, eps, step, mode, bias_correction, weight_decay):
    ptrs, sizes, bt, bc = _meta(lists, lists[0][0].device)
    name = _FNS[(kind, lists[0][0].dtype)]
    args = (ptrs, sizes, bt, bc, scale, lr, beta1, beta2, eps, step, mode, bias_correction, weight_decay)
    if kind == "remainder" and _amax["buf"] is not None and hasattr(_ext, name + "_amax"):
        slots, slot_set = _tensor_slots(lists[1], lists[0][0].device)
        getattr(_ext, name + "_amax")(*args, slots, _amax["buf"])
        _amax["covered"] |= slot_set
        return
    getattr(_ext, name)(*args)


def _scale(grads, scale):
    """Scale in place exactly as the reference clip does (FP32 math, one rounding to dtype)."""
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_scale_tensor

    noop = torch.zeros(1, dtype=torch.int, device=grads[0].device)
    multi_tensor_applier(multi_tensor_scale_tensor, noop, [grads, grads], scale)


def defer_clip(grads, coeff):
    """Record one clip; returns False if a clip is already pending (caller scales itself)."""
    if _pending["scale"] is not None:
        return False
    _pending["scale"] = coeff
    _pending["grads"] = {g.data_ptr(): g for g in grads if g.numel() > 0}
    return True


def wrap_adam_op(orig_op, kind="plain"):
    """Wrap one FusedAdam multi-tensor op.

    `kind` is "plain" (fp32 g/p/m/v), "remainder" (bf16 p + int16 remainder) or anything
    else, in which case pending gradients are scaled before the original op runs.
    """

    def op(chunk_size, noop, lists, *args):
        scale = _pending["scale"]
        pend = _pending["grads"]
        if scale is None or not lists or not lists[0]:
            return orig_op(chunk_size, noop, lists, *args)
        hit = [g.numel() > 0 and g.data_ptr() in pend for g in lists[0]]
        if not any(hit):
            return orig_op(chunk_size, noop, lists, *args)
        if len(args) != 8 or not _fusable(kind, lists):
            grads = [g for g, h in zip(lists[0], hit) if h]
            _scale(grads, scale)
            for g in grads:
                pend.pop(g.data_ptr(), None)
            counters["fallback"] += 1
            return orig_op(chunk_size, noop, lists, *args)
        fused = [[lst[i] for i in range(len(hit)) if hit[i]] for lst in lists]
        rest = [[lst[i] for i in range(len(hit)) if not hit[i]] for lst in lists]
        for g in fused[0]:
            pend.pop(g.data_ptr(), None)
        fused_adam(kind, fused, scale, *args)
        counters["fused"] += 1
        if rest[0]:
            orig_op(chunk_size, noop, rest, *args)

    op._nvfp4_clip_adam = True
    return op


_OP_KINDS = {
    "multi_tensor_adam": "plain",
    "multi_tensor_adam_param_remainder": "remainder",
    "multi_tensor_adam_fp8": None,
    "multi_tensor_adam_capturable": None,
    "multi_tensor_adam_capturable_master": None,
}


def install(fused_adam_optimizer):
    """Wrap every Adam op of a FusedAdam instance, so no op reads an unscaled pending grad."""
    for name, kind in _OP_KINDS.items():
        orig = getattr(fused_adam_optimizer, name, None)
        if orig is not None and not getattr(orig, "_nvfp4_clip_adam", False):
            setattr(fused_adam_optimizer, name, wrap_adam_op(orig, kind))


# multi_tensor_apply's launch limits for one tensor list.
_L2_MAX_TENSORS = 110
_L2_MAX_BLOCKS = 320
_l2_cache = {}


def _l2_meta(tensors, chunk_size, device):
    key = (chunk_size,) + tuple((t.data_ptr(), t.numel()) for t in tensors)
    hit = _l2_cache.get(key)
    if hit is not None:
        return hit
    ptrs, sizes, bt, bc, slots, cur, loc = [], [], [], [], [], [], 0
    last = len(tensors) - 1
    for i, t in enumerate(tensors):
        n = t.numel()
        ptrs.append(t.data_ptr())
        sizes.append(n)
        loc += 1
        c = (n + chunk_size - 1) // chunk_size
        for k in range(c):
            cur.append(len(bt))
            bt.append(i)
            bc.append(k)
            end = k == c - 1
            if (loc == _L2_MAX_TENSORS and end) or len(cur) == _L2_MAX_BLOCKS or (i == last and end):
                slots.extend(cur + [-1] * (_L2_MAX_BLOCKS - len(cur)))
                cur = []
                loc = 0 if end else 1
    meta = (
        torch.tensor(ptrs, dtype=torch.int64).to(device),
        torch.tensor(sizes, dtype=torch.int64).to(device),
        torch.tensor(bt, dtype=torch.int32).to(device),
        torch.tensor(bc, dtype=torch.int32).to(device),
        torch.tensor(slots, dtype=torch.int32).to(device),
        torch.empty(len(bt), dtype=torch.float32, device=device),
    )
    if len(_l2_cache) > 16:
        _l2_cache.clear()
    _l2_cache[key] = meta
    return meta


# Early chunk sums: the grad-norm list is the same every step, so once a norm has run, the
# chunk sums of the tensors inside a reduced grad buffer can be computed as soon as that
# buffer's reduce-scatter is waited on (`l2_prefetch`); the norm then only computes the
# remaining blocks and assembles. `l2_prefetch_begin` starts a new round for the next norm.
l2_check = {"left": 0, "log": print}
_l2_pre = {"key": None, "meta": None, "host": None, "dtype": None, "chunk": 0,
           "active": False, "done": [], "sel": {}, "noop": None, "pending": False}


def _l2_key(tensors, chunk_size):
    return (chunk_size,) + tuple((t.data_ptr(), t.numel()) for t in tensors)


def _l2_drop_pending():
    """Clear the flag of early chunks that no norm consumed (the list changed)."""
    if _l2_pre["pending"]:
        _l2_pre["noop"].zero_()
        _l2_pre["pending"] = False


def l2_prefetch_begin():
    _l2_drop_pending()
    _l2_pre["done"] = []
    _l2_pre["active"] = _l2_pre["key"] is not None and hasattr(_ext, "l2norm_part")


def _l2_select(tag, ranges):
    # Keyed by the ranges too: only tensors of the last norm that lie inside the given
    # (live) buffers are read, even if the norm's list has changed since.
    tag = (tag, tuple(ranges))
    sel = _l2_pre["sel"].get(tag)
    if sel is None:
        spans, bt = _l2_pre["host"]
        inside = [any(lo <= a and b <= hi for lo, hi in ranges) for a, b in spans]
        idx = [k for k, t in enumerate(bt) if inside[t]]
        sel = (torch.tensor(idx, dtype=torch.int32).to(_l2_pre["meta"][0].device), set(idx))
        _l2_pre["sel"][tag] = sel
    return sel


def l2_prefetch(tag, ranges):
    """Compute the chunk sums of the norm tensors lying inside `ranges` [(lo, hi) bytes)."""
    if not _l2_pre["active"] or any(t[0] == tag for t in _l2_pre["done"]):
        return False
    sel, idx = _l2_select(tag, ranges)
    _l2_pre["done"].append((tag, tuple(ranges)))
    if not idx:
        return False
    ptrs, sizes, bt, bc, slots, sums = _l2_pre["meta"]
    if _l2_pre["noop"] is None:
        _l2_pre["noop"] = torch.zeros(1, dtype=torch.int, device=sums.device)
    _ext.l2norm_part(ptrs, sizes, bt, bc, sel, _l2_pre["chunk"],
                     _l2_pre["dtype"] == torch.bfloat16, sums, _l2_pre["noop"])
    _l2_pre["pending"] = True
    counters["l2_prefetch"] += 1
    return True


def _l2_rest_sel():
    done = tuple(_l2_pre["done"])
    rest = _l2_pre["sel"].get(("rest",) + done)
    if rest is None:
        covered = set().union(*(_l2_pre["sel"][t][1] for t in done))
        idx = [k for k in range(len(_l2_pre["host"][1])) if k not in covered]
        dev = _l2_pre["meta"][0].device
        rest = torch.tensor(idx, dtype=torch.int32).to(dev) if idx else None
        _l2_pre["sel"][("rest",) + done] = rest
    return rest


def wrap_l2norm(orig_op):
    """multi_tensor_l2norm with per_tensor=False in two launches, bitwise equal to orig_op."""

    def op(chunk_size, noop, lists, per_tensor=False, *rest):
        if per_tensor or rest or len(lists) != 1 or not lists[0] or not hasattr(_ext, "l2norm"):
            return orig_op(chunk_size, noop, lists, per_tensor, *rest)
        tensors = lists[0]
        dt = tensors[0].dtype
        if dt not in _GRAD_DTYPES or any(
            t.dtype != dt or not t.is_cuda or t.numel() == 0 or t.numel() >= 2**31 for t in tensors
        ):
            return orig_op(chunk_size, noop, lists, per_tensor, *rest)
        device = tensors[0].device
        key = _l2_key(tensors, chunk_size)
        ret = torch.empty(1, dtype=torch.float32, device=device)
        early = _l2_pre["active"] and _l2_pre["done"] and key == _l2_pre["key"]
        _l2_pre["active"] = False
        if early:
            ptrs, sizes, bt, bc, slots, sums = _l2_pre["meta"]
            if _l2_pre["noop"] is None:
                _l2_pre["noop"] = torch.zeros(1, dtype=torch.int, device=device)
            rest_sel = _l2_rest_sel()
            if rest_sel is not None:
                _ext.l2norm_part(ptrs, sizes, bt, bc, rest_sel, chunk_size,
                                 dt == torch.bfloat16, sums, noop)
            _ext.l2norm_finish(sums, slots, _l2_pre["noop"], noop, ret)
            _l2_pre["pending"] = False
            counters["l2norm_early"] += 1
            if l2_check["left"] > 0:
                l2_check["left"] -= 1
                ref, _ = orig_op(chunk_size, torch.zeros_like(noop), lists, False)
                equal = torch.equal(ref.view(torch.int32), ret.view(torch.int32))
                l2_check["log"](f"early l2norm check: equal={bool(equal)} {counters}")
        else:
            _l2_drop_pending()
            meta = _l2_meta(tensors, chunk_size, device)
            ptrs, sizes, bt, bc, slots, sums = meta
            _ext.l2norm(ptrs, sizes, bt, bc, slots, chunk_size, dt == torch.bfloat16, sums, noop, ret)
            if key != _l2_pre["key"]:
                es = tensors[0].element_size()
                spans = [(t.data_ptr(), t.data_ptr() + t.numel() * es) for t in tensors]
                bt_host = []
                for i, t in enumerate(tensors):
                    bt_host.extend([i] * ((t.numel() + chunk_size - 1) // chunk_size))
                _l2_pre.update(key=key, meta=meta, host=(spans, bt_host), dtype=dt,
                               chunk=chunk_size, sel={})
        counters["l2norm"] += 1
        return ret, torch.empty(0, dtype=torch.float32, device=device)

    op._nvfp4_l2norm = True
    return op


def finish():
    scale = _pending["scale"]
    left = list(_pending["grads"].values())
    if scale is not None and left:
        _scale(left, scale)
        counters["leftover"] += len(left)
    _pending["scale"] = None
    _pending["grads"] = {}
