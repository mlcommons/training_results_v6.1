"""Firmus L4 run_start gate: at the end of the warm-up, every Firmus kernel whose switch is on must have armed.

Every Firmus module and patch checks its first eager call against the stock path, as bytes, and raises on a mismatch;
the modules also refuse a switch they cannot honour.  What no single module can see is a hook that never fired: an armed
module whose target was never reached (a renamed class, a code path the recipe no longer takes) would leave the run on
the stock path with no error, and only the proof regexes after the run would notice.  This gate closes that gap before
the clock starts.  When the recipe's WarmupCallback.on_data_init_start returns -- the training and validation warm-ups
are done, and MLPerfLoggingCallback.on_data_init_start logs run_start next (pretrain.py's callback order) -- every armed
part must report its first-call check done.  Rank 0 prints one line; a rank that finds an armed part unchecked raises,
so the job ends before run_start (not an attempt: docs/RUN_POLICY.md rule 1).

The gate only reads each part's own state; it recomputes nothing and launches nothing.  It covers the eight firmus_ak
modules and the Firmus patches whose state is a module flag: Transformer Engine's rope.py (X12) and
dot_product_attention/utils.py (QKV pack), Megatron-Core's clip_grads.py (X9 norm fold) and the recipe's
custom_embedding.py (EMB).  FIRMUS_AK_RUN_START_GATE=0 turns it off (the L4 config does not set it).
"""
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "callback_warmup"
_ON = os.environ.get("FIRMUS_AK_RUN_START_GATE", "1") == "1"
_PATCH_SWITCHES = ("AK_X12_ROPE", "AK_QKV_PACK", "AK_EMB", "AK_X9_FOLD")
_DONE = {"ran": False}


def _on(name):
    return os.environ.get(name, "0") == "1"


def _armed_module(name):
    m = sys.modules.get(name)
    return m if (m is not None and getattr(m, "_ARMED", False)) else None


def _status():
    """[(part, checked, detail)] for every armed Firmus part"""
    rows = []
    m = _armed_module("ak_l8b_gsa")
    if m is not None:
        rows.append(("gsa", m._STATE["checked"] is True, "configs forced %d" % m._STATE["forced"]))
    m = _armed_module("ak_l8b_projcol")
    if m is not None:
        rows.append(("projcol", m._STATE["checked"] is True and m._STATE["cleared"] > 0,
                     "modules cleared %d" % m._STATE["cleared"]))
    m = _armed_module("ak_l8b_opswz")
    if m is not None:
        rows.append(("opswz", len(m._STATE["checked"]) > 0, "weight shapes checked %d" % len(m._STATE["checked"])))
    m = _armed_module("ak_l8b_fp8cs_patch")
    if m is not None:
        rows.append(("fp8cs", len(m._STATE["checked"]) > 0, "paths checked %d" % len(m._STATE["checked"])))
    m = _armed_module("ak_l8b_fp4g_patch")
    if m is not None:
        st = m._STATE
        rows.append(("fp4g", bool(st["active"]) and (m._CHECK == "off" or st["checks"] > 0),
                     "active %s, gather checks %d" % (bool(st["active"]), st["checks"])))
        if getattr(m, "_NOOVERLAP", False):
            # L4 v3: the forward's gathers were finished at their dispatch (both kinds) in the warm-up
            rows.append(("fp4g no-overlap", st["nooverlap_fp4g"] > 0 and st["nooverlap_g0"] > 0,
                         "FP4G gathers finished at dispatch %d, bucket group 0 gathers waited at dispatch %d"
                         % (st["nooverlap_fp4g"], st["nooverlap_g0"])))
    m = _armed_module("ak_l8b_lmh_patch")
    if m is not None:
        kinds = {k[0] for k in m._S["checked"]}
        rows.append(("lmh forward", 0 in kinds, "problems checked %d" % len(m._S["checked"])))
        if m._DGRAD_ON:
            rows.append(("lmh dgrad", bool(m._S["dgrad_installed"]) and 1 in kinds,
                         "installed %s" % bool(m._S["dgrad_installed"])))
    m = _armed_module("ak_l8b_nvfp4p_patch")
    if m is not None:
        st = m._STATE
        rows.append(("nvfp4p", len(st["checked"]) > 0, "first-call checks %d" % len(st["checked"])))
        if m._RHTCAST_ON:
            rows.append(("nvfp4p rht cast", len(st["rhtcast_checked"]) > 0,
                         "casts checked %d" % len(st["rhtcast_checked"])))
        if m._WAMAX_ON:
            rows.append(("nvfp4p weight amax", bool(st["wamax_valid"]), "updates %d" % st["wamax_calls"]))
        if m._BWD_ON:
            rows.append(("nvfp4p norm backward", st["bwd_fused"] > 0, "fused %d" % st["bwd_fused"]))
    m = _armed_module("ak_l8b_smallk2_patch")
    if m is not None:
        if m._EMB_ON:
            rows.append(("smallk2 embedding", m._EMB["checked"] is True, "calls %d" % m._EMB["calls"]))
        if m._DS_ON:
            rows.append(("smallk2 delayed scaling", m._DS["armed"] is True, "why %s" % m._DS["why"]))
    # the Firmus patches applied in the Dockerfile (their module flags; the switch is read from the environment, so a
    # patched module that was never imported counts as unchecked)
    if _on("AK_X12_ROPE"):
        m = sys.modules.get("transformer_engine.pytorch.attention.rope")
        c = getattr(m, "_AK_X12_CHECKED", {}) if m is not None else {}
        rows.append(("x12 rope", bool(getattr(m, "_AK_X12_ROPE", False)) and c.get("fwd") is True
                     and c.get("bwd") is True, "fwd %s bwd %s" % (c.get("fwd"), c.get("bwd"))))
    if _on("AK_QKV_PACK"):
        m = sys.modules.get("transformer_engine.pytorch.attention.dot_product_attention.utils")
        st = getattr(m, "_AK_QKV_PACK_STATE", {}) if m is not None else {}
        rows.append(("qkv pack", st.get("checked") is True, "hits %s" % st.get("hits")))
    if _on("AK_EMB"):
        m = sys.modules.get("embedding_lib.custom_embedding.custom_embedding")
        st = getattr(m, "_AK_EMB_STATE", {}) if m is not None else {}
        rows.append(("emb", st.get("checked") is True, "checked %s" % st.get("checked")))
        if _on("AK_EMB_MAIN_GRAD"):
            st = getattr(m, "_AK_EMB_STATE_MG", {}) if m is not None else {}
            rows.append(("emb main_grad", st.get("checked") is True, "checked %s" % st.get("checked")))
    if _on("AK_X9_FOLD"):
        m = sys.modules.get("megatron.core.optimizer.clip_grads")
        rows.append(("x9 norm fold", getattr(m, "_ak_fold_announced", False) is True,
                     "module %s" % ("loaded" if m is not None else "not loaded")))
    return rows


def _rank():
    try:
        import torch
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get("RANK", "0"))
    except Exception:  # noqa: BLE001 -- only chooses the printing rank
        return 0


def check():
    """raise unless every armed Firmus part has checked its first call; rank 0 prints the result"""
    rows = _status()
    bad = [(name, detail) for name, ok, detail in rows if not ok]
    _DONE["ran"] = True
    if _rank() == 0 or bad:
        print("[auto_kernel gate] warm-up done, before run_start: {} armed Firmus parts ({}); every armed part checked "
              "its first call: {}{}".format(len(rows), ", ".join(name for name, _, _ in rows), not bad,
                                           "; unchecked: {}".format(bad) if bad else ""), flush=True)
    if bad:
        raise RuntimeError("[auto_kernel gate] armed but never checked in the warm-up (the run would use the stock "
                           "path): {}".format(bad))


def _patch(mod):
    cls = mod.WarmupCallback
    orig = cls.on_data_init_start

    def on_data_init_start(self, *a, **k):
        r = orig(self, *a, **k)
        check()
        return r
    on_data_init_start.__wrapped__ = orig
    cls.on_data_init_start = on_data_init_start


def _anything_armed():
    names = ("ak_l8b_gsa", "ak_l8b_projcol", "ak_l8b_opswz", "ak_l8b_fp8cs_patch", "ak_l8b_fp4g_patch",
             "ak_l8b_lmh_patch", "ak_l8b_nvfp4p_patch", "ak_l8b_smallk2_patch")
    return any(_armed_module(n) is not None for n in names) or any(_on(s) for s in _PATCH_SWITCHES)


if _ON and _anything_armed():
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                raise RuntimeError("[auto_kernel gate] the recipe module %s cannot be hooked" % fullname)
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _patch(module)
            spec.loader.exec_module = exec_module
            return spec

    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
