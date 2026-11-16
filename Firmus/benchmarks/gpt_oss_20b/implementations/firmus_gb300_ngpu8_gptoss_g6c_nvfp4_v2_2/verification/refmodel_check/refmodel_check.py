# Classification: Internal
"""Records the model the run actually built and the validation samples every evaluation reads.

Hooked into the image's unmodified pretrain.py by sitecustomize.py: after callback_logging is
imported, MLPerfLoggingCallback is replaced by a subclass that calls the parent method and then
reports. Nothing else in the run changes.

At train start rank 0 prints, from the built model (not from the config files):
  REFCHECK_CONFIG  the TransformerConfig fields that define the architecture, and eod_mask_loss of
                   the dataset config;
  REFCHECK_LAYER   per layer: core attention class, window_size, softmax_type, and the class and
                   eps of q_layernorm / k_layernorm;
  REFCHECK_PARAMS  every parameter name with the layer index replaced by N, its shape and count;
  REFCHECK_MODEL   the summary checks (no bias parameter, no softmax offset/sink parameter, ...).
For every evaluation each rank records the sample indices its validation sampler yields between
eval start and eval end; rank 0 prints REFCHECK_EVAL with the union over all ranks (count, unique
count, min, max, sha256 of the sorted list) and whether it equals the first evaluation's.

Environment: REFCHECK_STOP_STEP (stop the run after this step, default 30).
"""

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import re
import sys

STOP_STEP = int(os.environ.get("REFCHECK_STOP_STEP", "30"))
_recording = False
_indices = []
_first_eval = None
_eval_ok = True
_evals = 0


def _is_pretrain():
    return bool(sys.argv) and os.path.basename(sys.argv[0]) == "pretrain.py"


def _rank():
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def _print0(tag, payload):
    if _rank() == 0:
        print(f"{tag} {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _unwrap(m):
    while hasattr(m, "module"):
        m = m.module
    return m


def _patch_sampler():
    from megatron.bridge.data import samplers

    orig_iter = samplers.MegatronPretrainingSampler.__iter__

    def recording_iter(self):
        for batch in orig_iter(self):
            if _recording:
                _indices.extend(int(i) for i in batch)
            yield batch

    samplers.MegatronPretrainingSampler.__iter__ = recording_iter


def report_model(context):
    model = _unwrap(context.model[0])
    cfg = model.config
    fields = ("num_layers", "hidden_size", "ffn_hidden_size", "moe_ffn_hidden_size", "num_moe_experts",
              "moe_router_topk", "num_attention_heads", "num_query_groups", "kv_channels", "normalization",
              "layernorm_epsilon", "init_method_std", "add_bias_linear", "add_qkv_bias", "gated_linear_unit",
              "glu_linear_offset", "activation_func_clamp_value", "qk_layernorm", "softmax_type",
              "window_size", "window_attn_skip_freq", "rotary_base", "position_embedding_type",
              "apply_rope_fusion", "fused_single_qkv_rope", "yarn_rotary_scaling_factor",
              "moe_router_score_function", "moe_router_pre_softmax", "moe_router_load_balancing_type")
    info = {f: getattr(cfg, f, "<absent>") for f in fields}
    act = getattr(cfg, "activation_func", None)
    info["activation_func"] = f"{getattr(act, '__module__', '')}.{getattr(act, '__name__', act)}"
    info["model_position_embedding_type"] = getattr(model, "position_embedding_type", None)
    info["rotary_pos_emb_class"] = type(getattr(model, "rotary_pos_emb", None)).__name__
    info["dataset_eod_mask_loss"] = getattr(context.state.cfg.dataset, "eod_mask_loss", "<absent>")
    _print0("REFCHECK_CONFIG", info)

    windows = []
    for name, mod in model.named_modules():
        m = re.fullmatch(r"decoder\.layers\.(\d+)\.self_attention", name)
        if not m:
            continue
        core = getattr(mod, "core_attention", None)
        qn, kn = getattr(mod, "q_layernorm", None), getattr(mod, "k_layernorm", None)
        layer = {
            "layer": int(m.group(1)),
            "core_attention": type(core).__name__,
            "window_size": getattr(core, "window_size", "<absent>"),
            "softmax_type": getattr(core, "softmax_type", "<absent>"),
            "q_layernorm": type(qn).__name__ if qn is not None else None,
            "q_layernorm_eps": getattr(qn, "eps", None),
            "k_layernorm": type(kn).__name__ if kn is not None else None,
            "k_layernorm_eps": getattr(kn, "eps", None),
        }
        windows.append(layer)
        _print0("REFCHECK_LAYER", layer)

    params = {}
    for name, p in model.named_parameters():
        key = re.sub(r"layers\.\d+\.", "layers.N.", name)
        key = re.sub(r"(weight|bias)\d+$", r"\1<expert>", key)
        entry = params.setdefault(key, {"shape": list(p.shape), "count": 0})
        entry["count"] += 1
    for key in sorted(params):
        _print0("REFCHECK_PARAMS", {"name": key, **params[key]})
    names = list(params)
    summary = {
        "parameter_groups": len(names),
        "bias_parameters": [n for n in names if re.search(r"bias", n.rsplit(".", 1)[-1])],
        "softmax_offset_or_sink_parameters": [n for n in names if re.search(r"softmax_offset|sink", n)],
        "q_layernorm_layers": sum(1 for w in windows if w["q_layernorm"]),
        "k_layernorm_layers": sum(1 for w in windows if w["k_layernorm"]),
        "sliding_window_layers": [w["layer"] for w in windows if w["window_size"] not in (None, (-1, -1), "<absent>")],
        "window_sizes": sorted({str(w["window_size"]) for w in windows}),
        "softmax_types": sorted({str(w["softmax_type"]) for w in windows}),
        "layers": len(windows),
    }
    _print0("REFCHECK_MODEL", summary)


def report_eval(step):
    import torch.distributed as dist

    global _first_eval, _eval_ok, _evals
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, list(_indices))
    if _rank() != 0:
        return
    union = sorted(i for g in gathered for i in g)
    digest = hashlib.sha256(json.dumps(union).encode()).hexdigest()
    if _first_eval is None:
        _first_eval = digest
    same = digest == _first_eval
    _eval_ok &= same
    _evals += 1
    _print0("REFCHECK_EVAL", {
        "step": step, "eval": _evals, "samples": len(union), "unique": len(set(union)),
        "min": union[0] if union else None, "max": union[-1] if union else None,
        "per_rank": [len(g) for g in gathered], "sha256_sorted": digest, "same_as_first_eval": same,
    })


def _patch_logging(mod):
    base = mod.MLPerfLoggingCallback
    _patch_sampler()

    class RefCheckLoggingCallback(base):
        def on_train_start(self, context):
            super().on_train_start(context)
            report_model(context)

        def on_eval_start(self, context):
            global _recording
            _indices.clear()
            _recording = True
            super().on_eval_start(context)

        def on_eval_end(self, context):
            global _recording
            super().on_eval_end(context)
            _recording = False
            report_eval(context.state.train_state.step)

        def on_train_step_end(self, context):
            step = context.state.train_state.step + 1
            super().on_train_step_end(context)
            if step == STOP_STEP:
                _print0("REFCHECK_SUMMARY", {"stop_step": step, "evals": _evals,
                                             "every_eval_same_samples": bool(_eval_ok)})
                mod.mllogger.end(
                    mod.mllogger.constants.RUN_STOP,
                    metadata={mod.mllogger.constants.SAMPLES_COUNT: self._get_samples_count(context.state),
                              "status": "aborted"},
                )
                context.state.should_stop = True
                context.state.train_state.step = context.state.cfg.train.train_iters
                context.state.train_state.do_valid = False
                context.state.train_state.do_test = False

    mod.MLPerfLoggingCallback = RefCheckLoggingCallback


class _Loader(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch_logging(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "callback_logging" or not _is_pretrain():
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    sys.meta_path.insert(0, _Finder())
