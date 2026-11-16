"""auto_kernel SMALLK2 (Llama 3.1 8B campaign, 2026-10-08): two E0 removals of small work from the training step.

(1) The embedding gathered in [s, b, h] (AK_SMALLK2_EMB, default on).  Stock, with the recipe's custom_embed_forward
(callback_warmup.py) at TP 1 and Megatron's LanguageModelEmbedding.forward: embedding_ops.embedding_fwd gathers the rows
in [b, s] order (weight.index_select, 62.6 us per step in PROF-L2q), CustomEmbedding.forward clones the result (a 134 MB
device copy, 57.2 us) and LanguageModelEmbedding.forward makes its [b, s, h] -> [s, b, h] transpose contiguous (a
non-vectorized copy, 231.5 us).  Here one index_select gathers the same rows in [s, b] order straight into a contiguous
[s, b, h] tensor -- the same bytes -- and the backward hands the recipe's CustomEmbedding.backward (emb2's) exactly what
autograd hands it in the stock chain: the incoming gradient's transpose(0, 1) view, with the same saved ids and ctx
fields.  Only the recipe's case takes it (custom_embed_forward installed, TP 1, no reduce-scatter, no position or
token-type embedding, no MuP multiplier, no fp32 residual, no sequence parallelism, int64 [b, s] ids); anything else
runs Megatron's forward.  The first call (eager) compares the output with embedding_fwd + clone + transpose(0, 1)
.contiguous() byte for byte; a mismatch raises.

(2) TE's delayed-scaling updates batched (AK_SMALLK2_DS, default on).  FP8 attention runs Float8CurrentScaling for QKV,
O, dO, dQKV and a DelayedScaling recipe (amax history 1, "most_recent") for S and dP; each of the 32 per-layer autocast
regions' exits calls FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=True), and each layer's first module's
backward calls it with forward=False -- every call concatenates the 32 DPA modules' amaxes (CatArrayBatchedCopy) and runs
the fused update over all 32 (amax_and_scale_update_impl::kernel_bulk): 64 + 64 launches, 0.38 ms per step.  The fused
update is a per-module function of the module's own (amax_history, scale) with no communication here (the amax-reduction
group has one rank), and it leaves a module untouched when its amax_history[0] is 0 (no roll, scale kept); a module's
history is written only by its own attention (cuDNN's amax output) and by the update, and its scale is read only by its
own attention.  So, per step:
  - forward call 1 (layer 0's exit): the stock full forward update -- it also consumes whatever the forward amaxes hold
    from in between (an eval's forward writes every layer's S amax), as stock -- and the full backward update the
    previous step's backward deferred;
  - forward calls 2 .. n-1: skipped -- layer k's S amax, written by its attention, waits untouched for call n;
  - forward call n (the last region's exit, before any attention backward reads an S scale): the full forward update;
  - backward calls: skipped -- each layer's dP amax waits untouched for the next step's forward call 1, before any
    attention backward reads a dP scale again.
Every scale an attention kernel reads is then the stock value.  Eager iteration 0 runs TE's updates unchanged and
learns n (the forward calls per step, 32); for each of its calls it records which modules' tensors the update changed
and requires that a module changed by a call that will be skipped is changed by no later call of its phase (each
module consumed once per phase) -- else it raises.  The schedule arms from the next step (the graph capture) when every
delayed-scaling key takes TE's fused update without an amax all-reduce, and raises otherwise (L4: AK_SMALLK2_DS=0
keeps TE's updates); a forward phase of another length then raises.
Proof lines: "[auto_kernel smallk2] embedding gathered in [s, b, h] ARMED: ... True" and
"[auto_kernel smallk2] delayed-scaling updates batched ARMED: ... True".  Loaded by `import firmus_ak`; armed by
AK_SMALLK2=1 alone.
"""
import importlib
import importlib.abc
import importlib.util
import os
import sys

_ARMED = os.environ.get("AK_SMALLK2", "0") == "1"
_EMB_ON = _ARMED and os.environ.get("AK_SMALLK2_EMB", "1") == "1"
_DS_ON = _ARMED and os.environ.get("AK_SMALLK2_DS", "1") == "1"
# development only: per-replay digests of the delayed-scaling state (the backward key's deferred update applied to
# clones), written by rank 0 to this path
_DUMP = os.environ.get("AK_SMALLK2_DUMP", "")


def _rank():
    try:
        import torch
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get("RANK", "0"))
    except Exception:  # noqa: BLE001
        return 0


def _bytes(t):
    import torch
    return t.detach().reshape(-1).view(torch.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# (1) the embedding gathered in [s, b, h]
_EMB = {"fn": None, "checked": False, "calls": 0}


def _emb_case(self, input_ids, tokentype_ids):
    """the recipe's CustomEmbedding class when this call is the recipe's case, else None"""
    we = getattr(self, "word_embeddings", None)
    f = getattr(getattr(we, "forward", None), "__func__", None)
    if f is None or f.__name__ != "custom_embed_forward" or not f.__module__.endswith("callback_warmup"):
        return None
    ce = f.__globals__.get("CustomEmbedding")
    if ce is None or "embedding_ops" not in ce.forward.__globals__:
        return None
    cfg = self.config
    if (getattr(self, "add_position_embedding", True) or getattr(self, "reduce_scatter_embeddings", True)
            or tokentype_ids is not None or getattr(self, "tokentype_embeddings", None) is not None
            or (getattr(cfg, "use_mup", False) and getattr(cfg, "mup_embedding_mult", 1.0) != 1.0)
            or getattr(cfg, "fp32_residual_connection", True) or getattr(cfg, "sequence_parallel", True)
            or getattr(we, "reduce_scatter_embeddings", True) or we.tp_group.size() != 1):
        return None
    import torch
    w = we.weight
    if (input_ids.dim() != 2 or input_ids.dtype != torch.int64 or w.dim() != 2 or not w.is_contiguous()
            or not input_ids.is_cuda):
        return None
    return ce


def _emb_fn(ce):
    import torch
    g = ce.forward.__globals__

    class EmbSBH(torch.autograd.Function):
        """CustomEmbedding.apply(weight, ids, n).transpose(0, 1).contiguous() as one gather; the same backward"""

        @staticmethod
        def forward(ctx, weight, indices, num_embeddings):
            ctx.num_embeddings = num_embeddings          # the ctx fields CustomEmbedding.forward sets
            ctx.save_for_backward(indices)
            if g.get("_AK_EMB_MG", False):
                ctx.ak_weight = weight
            b, s = indices.shape
            return weight.index_select(0, indices.t().reshape(-1)).view(s, b, weight.shape[1])

        @staticmethod
        def backward(ctx, grad_output):
            # stock: CloneBackward0 passes the gradient through, TransposeBackward0 hands CustomEmbedding.backward
            # grad.transpose(0, 1)
            return ce.backward(ctx, grad_output.transpose(0, 1))

    return EmbSBH


def _emb_check(ce, we, ids, out):
    import hashlib
    import torch
    ops = ce.forward.__globals__["embedding_ops"]
    with torch.no_grad():
        ref = ops.embedding_fwd(we.weight, ids).clone().transpose(0, 1).contiguous()
    ok = (tuple(ref.shape) == tuple(out.shape) and ref.dtype == out.dtype and out.is_contiguous()
          and bool(torch.equal(_bytes(ref), _bytes(out))))
    if _rank() == 0:
        sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
        print("[auto_kernel smallk2] embedding gathered in [s, b, h] ARMED: weight {} {}, ids {} {} -> {}; one "
              "index_select in [s, b] order (patch {} sha256={}); backward: the recipe's CustomEmbedding.backward on "
              "grad.transpose(0, 1); byte-exact v embedding_fwd + clone + transpose(0, 1).contiguous(): {}".format(
                  list(we.weight.shape), str(we.weight.dtype).replace("torch.", ""), list(ids.shape),
                  str(ids.dtype).replace("torch.", ""), list(out.shape), os.path.basename(__file__), sha, ok),
              flush=True)
    if not ok:
        raise RuntimeError("[auto_kernel smallk2] E0 self-check FAILED for the [s, b, h] embedding gather")


def _patch_lme(mod):
    cls = mod.LanguageModelEmbedding
    if getattr(cls.forward, "_ak_smallk2", False):
        return
    orig = cls.forward

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        ce = _emb_case(self, input_ids, tokentype_ids)
        if ce is None:
            return orig(self, input_ids, position_ids, tokentype_ids)
        import torch
        if _EMB["fn"] is None:
            _EMB["fn"] = _emb_fn(ce)
        we = self.word_embeddings
        out = _EMB["fn"].apply(we.weight, input_ids, we.num_embeddings_per_partition)
        if not _EMB["checked"]:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("[auto_kernel smallk2] the [s, b, h] embedding first called inside graph capture: "
                                   "its check never ran")
            _emb_check(ce, we, input_ids, out)
            _EMB["checked"] = True
        _EMB["calls"] += 1
        return self.embedding_dropout(out)      # the stock non-sequence-parallel tail (dropout p 0: the same tensor)

    forward.__wrapped__ = orig
    forward._ak_smallk2 = True
    cls.forward = forward


# ---------------------------------------------------------------------------------------------------------------------
# (2) the delayed-scaling updates batched
_DS = {"phase": None, "k": 0, "n_f": None, "n_b": 0, "learn": True, "armed": False, "rec": [], "skipped": [0, 0],
       "why": None, "mod": None, "orig": None}


def _ds_tensors(qstate, forward):
    """[((key, i), amax_history, scale)] for the keys of one direction, in buffer order"""
    out = []
    for key in sorted(qstate.global_amax_buffer.keys()):
        if (key.split("_", 1)[0] == "forward") != forward:
            continue
        for i, (h, s) in enumerate(zip(qstate.global_amax_history_buffer[key], qstate.global_scale_buffer[key])):
            out.append(((key, i), h, s))
    return out


def _ds_snapshot(qstate, forward):
    return {k: (h.detach().clone(), s.detach().clone()) for k, h, s in _ds_tensors(qstate, forward)}


def _ds_changed(before, qstate, forward):
    import torch
    ch = set()
    for k, h, s in _ds_tensors(qstate, forward):
        if k not in before:
            ch.add(k)
            continue
        h0, s0 = before[k]
        if not (torch.equal(_bytes(h0), _bytes(h)) and torch.equal(_bytes(s0), _bytes(s))):
            ch.add(k)
    return ch


def _ds_preconditions(mod, qstate):
    """None when every delayed-scaling key takes TE's fused update with no amax all-reduce, else the reason"""
    import torch
    if bool(int(os.getenv("NVTE_UNFUSED_FP8_UPDATE", "0"))):
        return "NVTE_UNFUSED_FP8_UPDATE set"
    if not qstate.global_amax_buffer:
        return "no delayed-scaling tensors registered"
    for key in qstate.global_amax_buffer:
        _, autocast_key = key.split("_", 1)
        recipe, group = qstate.autocast_arguments[autocast_key]
        if callable(recipe.amax_compute_algo) or callable(recipe.scaling_factor_compute_algo):
            return "custom amax / scale algorithm ({})".format(key)
        if (recipe.reduce_amax and torch.distributed.is_initialized()
                and torch.distributed.get_world_size(group=group) > 1):
            return "amax all-reduce over {} ranks ({})".format(torch.distributed.get_world_size(group=group), key)
    return None


def _ds_learn_check(forward):
    """iteration 0: a module changed by a call the schedule will skip is changed by no later call of its phase"""
    rec = [c for (f, c) in _DS["rec"] if f == forward]
    n = len(rec)
    skip = range(1, n - 1) if forward else range(0, n)      # 0-based: forward calls 2 .. n-1, every backward call
    bad = 0
    for j in skip:
        for later in rec[j + 1:]:
            bad += len(rec[j] & later)
    mods = sorted(len(c) for c in rec)
    return bad, n, mods


def _ds_phase_end(forward_ended):
    st = _DS
    if forward_ended is None:
        return
    if forward_ended:
        if st["learn"]:
            st["n_f"] = st["k"]
        elif st["armed"] and st["k"] != st["n_f"]:
            raise RuntimeError("[auto_kernel smallk2] a forward phase of {} delayed-scaling updates, learned {}: the "
                               "batched schedule no longer holds".format(st["k"], st["n_f"]))
    else:
        if st["learn"]:
            st["n_b"] = st["k"]
            _ds_arm()


def _ds_arm():
    """end of iteration 0's backward phase: check the recorded calls, arm the schedule from the next step"""
    import hashlib
    st = _DS
    st["learn"] = False
    mod = st["mod"]
    qstate = mod.FP8GlobalStateManager.quantization_state
    why = _ds_preconditions(mod, qstate)
    bf, nf, mf = _ds_learn_check(True)
    bb, nb, mb = _ds_learn_check(False)
    ok = bf == 0 and bb == 0
    if why is None and nf is not None and nf >= 1 and nb >= 1:
        st["armed"] = ok
    else:
        st["why"] = why or "no forward / backward updates seen in iteration 0"
    if _rank() == 0:
        n_mod = len(_ds_tensors(qstate, True))
        sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
        if st["armed"]:
            print("[auto_kernel smallk2] delayed-scaling updates batched ARMED: {} forward-key and {} backward-key "
                  "modules; {} forward updates per step (calls 1 and {} full, {} deferred to call {}; call 1 also runs "
                  "the backward key's full update) and {} backward updates (all deferred to the next step's forward "
                  "call 1); no amax all-reduce, TE's fused update (patch {} sha256={}); iteration 0: modules changed "
                  "per call {}..{} (forward), {}..{} (backward), a module changed by a deferred call changed again "
                  "later in its phase: {} / {} times: {}".format(
                      n_mod, len(_ds_tensors(qstate, False)), nf, nf, max(nf - 2, 0), nf, nb,
                      os.path.basename(__file__), sha, mf[0] if mf else 0, mf[-1] if mf else 0,
                      mb[0] if mb else 0, mb[-1] if mb else 0, bf, bb, ok), flush=True)
        else:
            print("[auto_kernel smallk2] delayed-scaling updates batched NOT armed: {}".format(
                st["why"] or "iteration 0 check failed ({} / {})".format(bf, bb)), flush=True)
    if why is None and not ok:
        raise RuntimeError("[auto_kernel smallk2] E0 self-check FAILED for the delayed-scaling schedule: a module was "
                           "changed by two calls of one phase ({} forward, {} backward)".format(bf, bb))
    if not st["armed"]:
        raise RuntimeError("[auto_kernel smallk2] delayed-scaling updates batched NOT armed: {}".format(st["why"]))


def _patch_q(mod):
    M = mod.FP8GlobalStateManager
    if getattr(M.reduce_and_update_fp8_tensors, "_ak_smallk2", False):
        return
    orig = M.reduce_and_update_fp8_tensors.__func__
    _DS["mod"] = mod
    _DS["orig"] = orig

    def reduce_and_update_fp8_tensors(cls, forward=True):
        st = _DS
        forward = bool(forward)
        if st["phase"] != forward:
            _ds_phase_end(st["phase"])
            st["phase"] = forward
            st["k"] = 0
        st["k"] += 1
        k = st["k"]
        if st["learn"]:
            import torch
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("[auto_kernel smallk2] delayed-scaling updates batched NOT armed: the first "
                                   "update came inside graph capture (the iteration-0 check never ran)")
            qstate = cls.quantization_state
            before = _ds_snapshot(qstate, forward)
            r = orig(cls, forward=forward)
            st["rec"].append((forward, _ds_changed(before, qstate, forward)))
            return r
        if not st["armed"]:
            return orig(cls, forward=forward)
        if forward:
            if k > st["n_f"]:
                raise RuntimeError("[auto_kernel smallk2] forward delayed-scaling update {} of a phase learned at {}"
                                   .format(k, st["n_f"]))
            if k == 1:
                orig(cls, forward=False)            # the previous step's deferred backward updates
                return orig(cls, forward=True)
            if k == st["n_f"]:
                return orig(cls, forward=True)
            st["skipped"][0] += 1
            return None
        st["skipped"][1] += 1
        return None

    reduce_and_update_fp8_tensors._ak_smallk2 = True
    M.reduce_and_update_fp8_tensors = classmethod(reduce_and_update_fp8_tensors)


# ---------------------------------------------------------------------------------------------------------------------
# development: per-replay digests of the delayed-scaling state
def _dump_install(graphs):
    import hashlib
    lines = []

    def digest(forward, pending):
        import torch
        mod = sys.modules.get("transformer_engine.pytorch.quantization")
        if mod is None:
            return "-"
        qstate = mod.FP8GlobalStateManager.quantization_state
        tens = _ds_tensors(qstate, forward)
        if pending and tens:
            # the update the next forward call 1 will run, applied to clones (as TE runs it, key by key)
            hs = {k: h.detach().clone() for k, h, s in tens}
            ss = {k: s.detach().clone() for k, h, s in tens}
            for key in sorted({k[0] for k, h, s in tens}):
                ks = sorted(k for k in hs if k[0] == key)
                _, autocast_key = key.split("_", 1)
                recipe, _group = qstate.autocast_arguments[autocast_key]
                amax = torch.cat([hs[k][0] for k in ks])
                mod.tex.fused_amax_and_scale_update_after_reduction(
                    amax, [hs[k] for k in ks], [ss[k] for k in ks], recipe.amax_compute_algo,
                    mod.get_fp8_te_dtype(recipe, forward), recipe.margin)
            tens = [(k, hs[k], ss[k]) for k, h, s in tens]
        hsh = hashlib.sha256()
        for k, h, s in tens:
            hsh.update(_bytes(h).cpu().numpy().tobytes())
            hsh.update(_bytes(s).cpu().numpy().tobytes())
        return hsh.hexdigest()[:16]

    orig_replay = graphs.CUDAGraph.replay

    def replay(self, *a, **kw):
        import torch
        r = orig_replay(self, *a, **kw)
        torch.cuda.synchronize()
        if _rank() == 0:
            lines.append(1)
            with open(_DUMP, "a") as f:          # appended per replay: the process may end without atexit
                f.write("replay {} fwd {} bwd {} skipped {} {}\n".format(
                    len(lines), digest(True, False), digest(False, _DS["armed"]), _DS["skipped"][0],
                    _DS["skipped"][1]))
        return r

    graphs.CUDAGraph.replay = replay


# ---------------------------------------------------------------------------------------------------------------------
def _install(name, patch):
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != name:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                patch(module)
            spec.loader.exec_module = exec_module
            return spec

    if name in sys.modules:
        patch(sys.modules[name])
    else:
        sys.meta_path.insert(0, _Finder())


if _EMB_ON:
    _install("megatron.core.models.common.embeddings.language_model_embedding", _patch_lme)
if _DS_ON:
    _install("transformer_engine.pytorch.quantization", _patch_q)
if _ARMED and _DUMP:
    _install("torch.cuda.graphs", _dump_install)
