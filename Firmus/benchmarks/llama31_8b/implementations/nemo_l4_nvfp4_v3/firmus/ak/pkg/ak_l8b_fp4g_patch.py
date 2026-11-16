"""auto_kernel FP4G (Llama 3.1 8B campaign, 2026-10-07): the NVFP4 parameter gather on the stock DDP layout.

The recipe all-gathers every bf16 weight each step and then, in the forward, TE re-quantizes each of the 128 linear
weights with its 2D NVFP4 weight quantizer (5.2-5.7 ms per step; again in each of the eval's forwards).  Here, for the
bucket groups whose buckets hold only those weights and RMSNorm weights ("pure" groups; the embedding's and the output
layer's buckets stay as they are), the parameter all-gather moves FP4 instead of bf16, and the cast happens on each
rank's shard only:

  * optimizer step (eager; NVFP4P 13g's weight-amax update calls `before_allreduce`): this rank's partial tile amaxes of
    the tile-rows that shard boundaries cut (ak_l8b_fp4g.split_amax2) into a slot region appended to NVFP4P's weight-amax
    buffer, max-all-reduced with the weights' amaxes in the one all-reduce that runs today;
  * start_param_sync of a pure group (in the forward, so inside the full-iteration CUDA graph's capture), on a side
    stream forked from the current stream: ak_l8b_fp4g.shard_quant2 over this rank's shards (the FP4 nibbles of its own
    elements into the bucket's FP4 image -- byte (bucket element e) / 2 -- and the E4M3 scales of the tiles whose anchor
    element 256 t it owns into the bucket's tile-scale image -- byte e / 256), this rank's RMSNorm elements into the
    bucket's norm image, then ONE coalesced all-gather of the three images (equal chunks: the bucket's shard size S is a
    multiple of 256 and every weight starts on a multiple of 256);
  * the handle's wait (finish_param_sync, a forced sync, anything that waits on the handle): the side stream waits for
    the gather, ak_l8b_fp4g.post_ag3 writes the column-wise data and both GEMM-swizzled scale tensors into each weight's
    persistent NVFP4 storage, the norms go back into the bf16 parameter buffer, and the current stream joins the side
    stream -- the fork and the join both happen inside the capture (no event recorded outside it);
  * TE's weight quantize for these weights (NVFP4P 13g's tex.quantize wrapper asks `weight_quantize` first): the cached
    module workspace (QKV, proj) IS the persistent storage -- its row-wise data a view of the FP4 image, its amaxes the
    weight-amax buffer's slot -- so the in-place update is a no-op; an allocating quantize (the op fuser's fc1 / fc2, an
    uncached eval forward) gets a fresh wrapper over the same buffers.
The arithmetic is TE's 2D cast's, per element and per tile (dev g2-g4: every byte equal on the 128 weights at the stock
layout's shard cuts, DP 4 and 8); the gradient buffer, the reduce-scatter, the bucket layout and the mixed groups are
untouched.  The bf16 parameter buffer of a pure bucket is written only in this rank's shard (by the optimizer, as
today); the rest goes stale.  AK_FP4G_POISON=1 fills that rest with NaN at setup (proof runs only): any reader of the
stale bf16 weights then turns the loss into NaN.

Setup runs once, at the first weight-amax update (the end of the first optimizer step: TE has built its workspaces in the
first forward).  Checks (AK_FP4G_CHECK=first|all|off): at the start of an optimizer step after setup, before the
optimizer writes, every pure bucket's bf16 shards are re-gathered into a scratch buffer and every gathered weight is
compared with TE's own 2D cast of its re-gathered bf16 weight (tex.quantize with a copy of the weight's quantizer):
row-wise data, row-wise scales, column-wise data, column-wise scales and both amaxes, every byte; the norms against the
re-gathered values.  Rank 0 prints the result, a mismatch on any rank raises.

Armed by AK_FP4G=1 alone; AK_FP4G_EXT_DIR holds ak_l8b_fp4g.so; AK_FP4G_CBPC (16) is shard_quant2's
256-column blocks per CTA.  Loaded by `import firmus_ak`; it wraps _ParamAndGradBucketGroup.start_param_sync when
megatron.core.distributed.param_and_grad_buffer is imported (x9step3 then wraps this wrapper).  Needs NVFP4P 13g.

L4 v3, AK_FP4G_NOOVERLAP=1 (needs AK_FP4G=1): no parameter gather of the forward overlaps compute.  A pure group's
gather is finished right after its dispatch -- the handle's wait (the side stream's all-gather wait, post_ag3, the norm
unpack, the current stream's join) runs at once, and finish_param_sync gets a handle with nothing left to wait for (it
still dispatches the next group) -- and bucket group 0's stock bf16 all-gather (the one group with no next gather) is
waited for on the current stream right after its dispatch.  The same kernels on the same buffers in the same order; only
the waits move, so the arithmetic is unchanged.  Probes E6 / E8 (2026-10-13): in L4 v2's layout one rank's fc1 GEMM of
an FP4G group-first consumer layer occasionally returned a different output from identical inputs while the next group's
gather ran beside it; with the gathers finished at dispatch that forward-side departure did not occur.
"""
import os
import sys

_ARMED = os.environ.get("AK_FP4G", "0") == "1"
# L4 (C5, fail early): the setup runs only inside NVFP4P's weight-amax update; without it an armed FP4G would
# silently never gather in FP4
if _ARMED and not (os.environ.get("AK_NVFP4P", "0") == "1" and os.environ.get("AK_NVFP4P_WAMAX", "0") == "1"
                   and os.environ.get("AK_NVFP4P_WEIGHTS", "0") == "amax"):
    raise RuntimeError("[auto_kernel fp4g] AK_FP4G=1 needs NVFP4P's weight-amax update (AK_NVFP4P=1, "
                       "AK_NVFP4P_WAMAX=1, AK_NVFP4P_WEIGHTS=amax), which runs its setup")
_CHECK = os.environ.get("AK_FP4G_CHECK", "first")
_POISON = os.environ.get("AK_FP4G_POISON", "0") == "1"
_CBPC = int(os.environ.get("AK_FP4G_CBPC", "16"))
# L4 v3 (C5, fail early): the no-overlap mode lives in this module's start_param_sync wrapper
_NOOVERLAP_SW = os.environ.get("AK_FP4G_NOOVERLAP", "0")
if _NOOVERLAP_SW not in ("0", "1"):
    raise RuntimeError("[auto_kernel fp4g] AK_FP4G_NOOVERLAP=%r: must be 0 or 1" % _NOOVERLAP_SW)
_NOOVERLAP = _NOOVERLAP_SW == "1"
if _NOOVERLAP and not _ARMED:
    raise RuntimeError("[auto_kernel fp4g] AK_FP4G_NOOVERLAP=1 needs AK_FP4G=1 (the no-overlap mode is FP4G's "
                       "start_param_sync wrapper)")
_SUFFIXES = (".self_attention.linear_qkv.weight", ".self_attention.linear_proj.weight", ".mlp.linear_fc1.weight",
             ".mlp.linear_fc2.weight")
_STATE = {"active": False, "setup_done": False, "ext": None, "groups": [], "by_ptr": {}, "fp4_ids": set(),
          "k1": None, "n_params": 0, "slot_floats": 0, "checks": 0, "noop_calls": 0, "wrap_calls": 0, "gathers": 0,
          "side": None, "summary": None, "patched": False, "nooverlap_fp4g": 0, "nooverlap_g0": 0}


def _rank():
    import torch
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _say(msg):
    if _rank() == 0:
        print("[auto_kernel fp4g] " + msg, flush=True)


def _ext():
    if _STATE["ext"] is None:
        import torch  # noqa: F401  (the extension links libtorch)
        d = os.environ.get("AK_FP4G_EXT_DIR", "/opt/firmus/ak/lib")
        if d not in sys.path:
            sys.path.insert(0, d)
        import ak_l8b_fp4g
        _STATE["ext"] = ak_l8b_fp4g
    return _STATE["ext"]


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def active():
    return _STATE["active"]


def is_fp4_weight(p):
    """True for a weight gathered in FP4 (its bf16 copy is stale outside this rank's shard)"""
    return _STATE["active"] and id(p) in _STATE["fp4_ids"]


# ---------------------------------------------------------------------------------------------------- the weights
class _Weight:
    """one FP4-gathered weight: its persistent NVFP4 storage (proto), the row-wise view of the FP4 image"""

    def __init__(self, p, name, kind, proto, q_ref, img_view, slot):
        self.p, self.name, self.kind, self.proto, self.q_ref, self.img_view, self.slot = p, name, kind, proto, q_ref, \
            img_view, slot

    def wrap(self, quantizer):
        """a fresh NVFP4 object over the persistent buffers, with the usages the caller's quantizer asks for (TE's ops
        drop usages on the object they get, e.g. the op fuser keeps only the column-wise data for the backward)"""
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor, _make_nvfp4_tensor_in_reduce_ex
        pr = self.proto
        if isinstance(pr, NVFP4Tensor):
            fn, args = pr.__reduce_ex__(2)
            args = list(args)
            args[9] = quantizer  # the caller's quantizer (copied by the constructor), as TE's own quantize would carry
            obj = _make_nvfp4_tensor_in_reduce_ex(*args)
        else:                    # NVFP4TensorStorage (an internal quantizer's result): its constructor takes the metadata
            meta = pr.get_metadata()
            meta["quantizer"] = quantizer
            obj = type(pr)(**meta)
        if not bool(getattr(quantizer, "rowwise_usage", True)):
            obj._rowwise_data = obj._rowwise_scale_inv = obj._amax_rowwise = None
        if not bool(getattr(quantizer, "columnwise_usage", True)):
            obj._columnwise_data = obj._columnwise_scale_inv = obj._amax_columnwise = None
        return obj


def weight_quantize(tensor, quantizer, output, noop):
    """NVFP4P 13g's tex.quantize wrapper asks this first: the gathered storage for an FP4-gathered weight, else None"""
    if not _STATE["active"]:
        return None
    W = _STATE["by_ptr"].get((tensor.data_ptr(), tensor.numel()))
    if W is None:
        return None
    if type(quantizer).__name__ != "NVFP4Quantizer" or not getattr(quantizer, "with_2d_quantization", False):
        return None
    if output is not None:
        rd = getattr(output, "_rowwise_data", None)
        if output is not W.proto and (rd is None or rd.data_ptr() != W.img_view.data_ptr()):
            raise RuntimeError("[auto_kernel fp4g] %s: TE updates a weight workspace that is not the gathered storage "
                               "(the module's workspace was replaced)" % W.name)
        _STATE["noop_calls"] += 1
        return output
    _STATE["wrap_calls"] += 1
    return W.wrap(quantizer)


# ---------------------------------------------------------------------------------------------------- the gather
class _Handle:
    """the side stream's gather of one pure bucket group; wait() finishes it (post_ag3, norms) and joins"""

    def __init__(self, G, cm):
        self.G, self.cm = G, cm

    def wait(self):
        import torch
        G = self.G
        side = _STATE["side"]
        with torch.cuda.stream(side):
            if self.cm is not None:
                self.cm.wait()
                self.cm = None
            ext = _ext()
            ext.post_ag3(G["k3"][0], G["k3"][1], G["k3"][2], 2, 0)
            if G["unpack_dst"]:
                torch._foreach_copy_(G["unpack_dst"], G["unpack_src"])
        torch.cuda.current_stream().wait_stream(side)


def _dispatch(G):
    import torch
    from torch.distributed.distributed_c10d import _coalescing_manager
    side = _STATE["side"]
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        ext = _ext()
        if G["k2"][2]:
            ext.shard_quant2(G["k2"][0], G["k2"][1], G["k2"][2], _CBPC, 1)
        if G["pack_dst"]:
            torch._foreach_copy_(G["pack_dst"], G["pack_src"])
        with _coalescing_manager(G["group"], async_ops=True) as cm:
            for out, inp in G["ag"]:
                torch.distributed.all_gather_into_tensor(out, inp, group=G["group"], async_op=True)
    _STATE["gathers"] += 1
    return _Handle(G, cm)


class _NoopHandle:
    """AK_FP4G_NOOVERLAP: the handle finish_param_sync gets for a gather already finished at its dispatch"""

    def wait(self):
        return None


_NOOP_HANDLE = _NoopHandle()


def _wrap_start_param_sync(orig):
    def start_param_sync(self, force_sync: bool = False):
        G = getattr(self, "_ak_fp4g", None)
        if G is None or not _STATE["active"]:
            r = orig(self, force_sync=force_sync)
            if (_NOOVERLAP and not force_sync and G is None and self.next_param_gather_bucket_group is None
                    and self.param_gather_handle is not None):
                # bucket group 0's stock gather: the current stream waits for it right after its dispatch
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                _STATE["nooverlap_g0"] += 1
            return r
        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                return
        else:
            assert self.param_gather_handle is None
        h = _dispatch(G)
        if _NOOVERLAP:
            h.wait()  # the gather, post_ag3, the norm unpack and the join, before any later compute
            h = _NOOP_HANDLE
            _STATE["nooverlap_fp4g"] += 1
        if force_sync or not self.ddp_config.overlap_param_gather:
            h.wait()
            self.param_gather_handle = None
        else:
            self.param_gather_handle = h
        self.param_gather_dispatched = True
    start_param_sync.__wrapped__ = orig
    return start_param_sync


# ---------------------------------------------------------------------------------------------------- setup
def _owners(ddp, fp4):
    """weight id -> (TE source module, fused-op BasicLinear or None)"""
    src, ops = {}, {}
    for m in ddp.modules():
        for nm, p in m._parameters.items():
            if p is not None and nm == "weight" and id(p) in fp4:
                src.setdefault(id(p), m)
        fi = getattr(m, "_fused_impl", None)
        if isinstance(fi, tuple) and fi:
            for op in fi[0]:
                w = getattr(op, "weight", None)
                if w is not None and id(w) in fp4:
                    ops[id(w)] = op
    return src, ops


def _proto_for(p, src_mod, op):
    """the persistent NVFP4 storage of one weight (not yet re-pointed: see _adopt), or (None, None, reason)"""
    import torch
    import transformer_engine_torch as tex
    M, K = p.shape
    if op is not None:
        q = op.get_quantizer("forward", 1)
        if q is None or type(q).__name__ != "NVFP4Quantizer" or not getattr(q, "with_2d_quantization", False):
            return None, None, "fused op without the 2D NVFP4 weight quantizer"
        q2 = q.copy()
        q2.set_usage(rowwise=True, columnwise=True)
        q2.optimize_for_gemm = bool(getattr(q, "optimize_for_gemm", False))
        proto = tex.create_empty_quantized_tensor(q2, [M, K], torch.bfloat16, p.device, False)
        q_ref = q2
    else:
        ws = getattr(src_mod, "_fp8_workspaces", {}).get("weight") if src_mod is not None else None
        if ws is None or type(ws).__name__ not in ("NVFP4Tensor", "NVFP4TensorStorage"):
            return None, None, "no cached NVFP4 workspace"
        if ws._columnwise_data is None or ws._rowwise_data is None:
            return None, None, "workspace without both usages"
        proto = ws
        q_ref = ws._get_quantizer().copy() if hasattr(ws, "_get_quantizer") else ws._quantizer.copy()
        q_ref.set_usage(rowwise=True, columnwise=True)
        q_ref.optimize_for_gemm = bool(getattr(ws, "_with_gemm_swizzled_scales", False))
    if not getattr(proto, "_with_gemm_swizzled_scales", False):
        return None, None, "scales not GEMM-swizzled"
    if getattr(proto, "_row_scaled_nvfp4", False) or getattr(proto, "_nvfp4_use_4over6", False):
        return None, None, "row-scaled / 4over6"
    shp = {"_rowwise_scale_inv": (M, K // 16), "_columnwise_scale_inv": (K, M // 16), "_columnwise_data": (K, M // 2)}
    for nm, want in shp.items():
        t = getattr(proto, nm, None)
        if t is None or tuple(t.shape) != want or t.dtype != torch.uint8 or not t.is_contiguous():
            return None, None, "%s shape %s" % (nm, None if t is None else tuple(t.shape))
    return proto, q_ref, None


def _adopt(proto, img_view, slot_amax):
    """re-point the storage: its row-wise data the FP4 image's view, its amaxes the weight-amax buffer's slot"""
    proto._rowwise_data = img_view
    proto._amax_rowwise = slot_amax
    proto._amax_columnwise = slot_amax


def setup(opt, P):
    """once, inside NVFP4P's first weight-amax update (P: its _STATE, already holding the buffer and its slots)"""
    import torch
    if _STATE["setup_done"] or not _ARMED:
        return
    _STATE["setup_done"] = True
    chunks = list(getattr(opt, "model_chunks", []))
    if len(chunks) != 1:
        raise RuntimeError("[auto_kernel fp4g] NOT armed: %d model chunks" % len(chunks))
    ddp = chunks[0]
    named = list(ddp.named_parameters())
    fp4 = {id(p): (n, p) for n, p in named if p.dim() == 2 and p.dtype == torch.bfloat16 and n.endswith(_SUFFIXES)}
    src, ops = _owners(ddp, fp4)
    slot_of = P["wamax_slot"]
    n_params = P["wamax_buf"].numel()
    dev = torch.device("cuda", torch.cuda.current_device())
    ext = _ext()

    # ---- classify the buckets; lay out the images and the split tile-rows (rank-independent order)
    plan, reasons = [], {}
    for gi, bg in enumerate(ddp.bucket_groups):
        dp = bg.intra_distributed_optimizer_instance_size
        r = bg.intra_distributed_optimizer_instance_rank
        gplan, ok = [], True
        for b in bg.buckets:
            numel = b.param_data.numel() if b.param_data is not None else 0
            why = None
            if b.param_data is None or b.param_data.dtype != torch.bfloat16:
                why = "no bf16 parameter buffer"
            elif numel % (dp * 256):
                why = "bucket size %d not a multiple of 256 * DP" % numel
            ws_, ns_ = [], []
            for p in b.params_list:
                s, e = b.param_to_index[p]
                if id(p) in fp4:
                    M, K = p.shape
                    if s % 256 or M % 256 or K % 256:
                        why = why or "weight %s at %d not 256-aligned" % (fp4[id(p)][0], s)
                    ws_.append((p, s, e))
                elif p.dim() == 1 and p.dtype == torch.bfloat16:
                    ns_.append((p, s, e))
                else:
                    why = why or "holds %s" % ([n for n, q in named if q is p] or ["?"])[0]
            if not ws_:
                why = why or "no NVFP4 weight"
            if why is not None:
                ok = False
                reasons[gi] = why
                break
            gplan.append((b, numel, dp, r, ws_, ns_))
        if ok:
            plan.append((gi, bg, gplan))

    # split tile-rows of every pure bucket (touched by two or more ranks), in a fixed order
    slots = {}
    n_slot = 0
    for gi, bg, gplan in plan:
        for b, numel, dp, r, ws_, ns_ in gplan:
            S = numel // dp
            for p, s, e in ws_:
                K = p.shape[1]
                touch = {}
                for rr in range(dp):
                    lo, hi = max(s, rr * S) - s, min(e, (rr + 1) * S) - s
                    if lo < hi:
                        for ti in {lo // (16 * K), (hi - 1) // (16 * K)}:
                            touch.setdefault(ti, set()).add(rr)
                for ti in sorted(touch):
                    if len(touch[ti]) >= 2:
                        slots[(id(p), ti)] = n_params + n_slot
                        n_slot += K // 16
    # the weight-amax buffer with the slot region (nobody has cached the buffer yet: x9step3 does at its next step)
    old = P["wamax_buf"]
    buf = torch.zeros(n_params + n_slot, dtype=torch.float32, device=old.device)
    buf[:n_params].copy_(old)
    P["wamax_buf"] = buf
    _STATE["n_params"], _STATE["slot_floats"] = n_params, n_slot
    _STATE["buf"] = buf

    k1_rows, k1_begin, n1 = [], [], 0
    groups, n_w, img_bytes, ts_bytes, norm_bytes, skipped = [], 0, 0, 0, 0, []
    for gi, bg, gplan in plan:
        G = {"index": gi, "group": bg.intra_distributed_optimizer_instance_group, "ag": [], "pack_dst": [],
             "pack_src": [], "unpack_dst": [], "unpack_src": [], "buckets": [], "k2": None, "k3": None}
        k2_rows, k2_begin, n2 = [], [], 0
        k3_rows, k3_begin, n3 = [], [], 0
        fine = True
        built = []
        for b, numel, dp, r, ws_, ns_ in gplan:
            S = numel // dp
            img = torch.empty(numel // 2, dtype=torch.uint8, device=dev)
            tsi = torch.empty(numel // 256, dtype=torch.uint8, device=dev)
            Wb = []
            for p, s, e in ws_:
                M, K = p.shape
                slot = slot_of.get((p.data_ptr(), p.numel()))
                if slot is None:
                    fine = False
                    reasons[gi] = "weight %s has no weight-amax slot" % fp4[id(p)][0]
                    break
                view = img[s // 2: e // 2].view(M, K // 2)
                proto, q_ref, why = _proto_for(p, src.get(id(p)), ops.get(id(p)))
                if proto is None:
                    fine = False
                    reasons[gi] = "%s: %s" % (fp4[id(p)][0], why)
                    break
                Wb.append(_Weight(p, fp4[id(p)][0], "op" if id(p) in ops else "module", proto, q_ref, view, slot))
            if not fine:
                break
            built.append((b, numel, dp, r, S, img, tsi, Wb, ns_))
        if not fine:
            skipped.append(gi)
            continue
        for b, numel, dp, r, S, img, tsi, Wb, ns_ in built:
            for W in Wb:
                _adopt(W.proto, W.img_view, buf[W.slot:W.slot + 1])
            for W in Wb:
                p = W.p
                s, e = b.param_to_index[p]
                M, K = p.shape
                lo, hi = max(s, r * S) - s, min(e, (r + 1) * S) - s
                amax_ptr = buf.data_ptr() + 4 * W.slot
                if lo < hi:
                    ti0, ti1 = lo // (16 * K), (hi - 1) // (16 * K)
                    sf = slots.get((id(p), ti0))
                    sl_ = slots.get((id(p), ti1))
                    src_ptr = p.data_ptr() + 2 * lo
                    for ti in sorted({ti0, ti1}):
                        so = slots.get((id(p), ti))
                        if so is not None:
                            k1_rows.append([src_ptr, 0, 0, 0, buf.data_ptr() + 4 * so, 0, K, lo, hi, ti, ti])
                            k1_begin.append(n1)
                            n1 += -(-(K // 256) // _CBPC)
                    k2_rows.append([src_ptr, img.data_ptr() + s // 2, tsi.data_ptr() + s // 256, amax_ptr,
                                    0 if sf is None else buf.data_ptr() + 4 * sf,
                                    0 if sl_ is None else buf.data_ptr() + 4 * sl_, K, lo, hi, ti0, ti1])
                    k2_begin.append(n2)
                    n2 += (ti1 - ti0 + 1) * -(-(K // 256) // _CBPC)
                pr = W.proto
                k3_rows.append([img.data_ptr() + s // 2, tsi.data_ptr() + s // 256, pr._columnwise_data.data_ptr(),
                                pr._rowwise_scale_inv.data_ptr(), pr._columnwise_scale_inv.data_ptr(), M, K])
                k3_begin.append(n3)
                n3 += (M // 256) * (K // 128)
            # norms: per rank a fixed chunk of NB elements, its owned norm segments packed in parameter order
            own = [[] for _ in range(dp)]
            for p, s, e in ns_:
                for rr in range(dp):
                    a, z = max(s, rr * S), min(e, (rr + 1) * S)
                    if a < z:
                        own[rr].append((a, z))
            NB = max([sum(z - a for a, z in o) for o in own] + [0])
            NB = -(-NB // 8) * 8
            nimg = torch.empty(dp * NB, dtype=torch.bfloat16, device=dev) if NB else None
            if NB:
                for rr in range(dp):
                    off = rr * NB
                    for a, z in own[rr]:
                        L = z - a
                        if rr == r:
                            G["pack_dst"].append(nimg[off: off + L])
                            G["pack_src"].append(b.param_data[a:z])
                        else:
                            G["unpack_dst"].append(b.param_data[a:z])
                            G["unpack_src"].append(nimg[off: off + L])
                        off += L
            G["ag"].append((img, img[r * (S // 2): (r + 1) * (S // 2)]))
            G["ag"].append((tsi, tsi[r * (S // 256): (r + 1) * (S // 256)]))
            if NB:
                G["ag"].append((nimg.view(torch.uint8), nimg[r * NB: (r + 1) * NB].view(torch.uint8)))
            G["buckets"].append({"bucket": b, "numel": numel, "S": S, "r": r, "img": img, "tsi": tsi, "nimg": nimg,
                                 "weights": Wb, "norms": ns_})
            img_bytes += img.numel()
            ts_bytes += tsi.numel()
            norm_bytes += 0 if nimg is None else nimg.numel() * 2
            for W in Wb:
                _STATE["by_ptr"][(W.p.data_ptr(), W.p.numel())] = W
                _STATE["fp4_ids"].add(id(W.p))
                n_w += 1
        mk = lambda rows, nf: (torch.tensor(rows, dtype=torch.int64, device=dev) if rows  # noqa: E731
                               else torch.zeros((0, nf), dtype=torch.int64, device=dev))
        G["k2"] = (mk(k2_rows, ext.SEG_FIELDS), torch.tensor(k2_begin, dtype=torch.int32, device=dev), n2)
        G["k3"] = (mk(k3_rows, ext.POST_FIELDS), torch.tensor(k3_begin, dtype=torch.int32, device=dev), n3)
        bg._ak_fp4g = G
        groups.append(G)
    _STATE["groups"] = groups
    if k1_rows:
        _STATE["k1"] = (torch.tensor(k1_rows, dtype=torch.int64, device=dev),
                        torch.tensor(k1_begin, dtype=torch.int32, device=dev), n1)
    _STATE["side"] = torch.cuda.Stream(device=dev)
    if not groups:
        raise RuntimeError("[auto_kernel fp4g] NOT armed: no bucket group holds only NVFP4 weights and norms (%s)"
                           % {k: v for k, v in sorted(reasons.items())})
    _STATE["active"] = bool(groups)
    # poison: the stale part of every pure bucket's bf16 buffer (the other ranks' shards) set to NaN
    if _POISON and groups:
        n_poison = 0
        for G in groups:
            for B in G["buckets"]:
                S, r = B["S"], B["r"]
                pd = B["bucket"].param_data.view(torch.int16)
                for rr in range(pd.numel() // S):
                    if rr != r:
                        pd[rr * S: (rr + 1) * S].fill_(0x7FC0)
                        n_poison += S
        _STATE["poisoned"] = n_poison
    n_fp4 = len(fp4)
    _STATE["summary"] = (n_w, n_fp4, len(groups), len(ddp.bucket_groups))
    so = ext.__file__
    _say("NVFP4 parameter gather ARMED: %d of %d weights in %d of %d bucket groups (%s); FP4 images %.3f GB, tile-scale "
         "images %.1f MB, norm images %.1f KB; %d split tile-rows (%d slot floats in the weight-amax buffer); side "
         "stream; module %s sha256=%s; mixed groups %s; poison %s (%d elements); check %s" % (
             n_w, n_fp4, len(groups), len(ddp.bucket_groups),
             "buckets per group %s" % sorted({len(G["buckets"]) for G in groups}), img_bytes / 1e9, ts_bytes / 1e6,
             norm_bytes / 1e3, len(slots), n_slot, os.path.basename(so), _sha(so),
             {k: v for k, v in sorted(reasons.items())}, _POISON, _STATE.get("poisoned", 0), _CHECK))
    if _NOOVERLAP:
        # L4 v3 (C5): the wrapper must be on the class, and the stock group it waits for must exist
        last = [gi for gi, bg in enumerate(ddp.bucket_groups)
                if bg.next_param_gather_bucket_group is None and getattr(bg, "_ak_fp4g", None) is None]
        ovl = bool(ddp.ddp_config.overlap_param_gather)
        if not _STATE["patched"] or (ovl and len(last) != 1):
            raise RuntimeError("[auto_kernel fp4g] AK_FP4G_NOOVERLAP=1 cannot install: start_param_sync wrapped %s, "
                               "stock bucket groups with no next gather %s" % (_STATE["patched"], last))
        _say("no-overlap parameter gather ARMED (AK_FP4G_NOOVERLAP=1): each of the %d FP4G groups' gathers is finished "
             "at its dispatch (all-gather wait, post_ag3, norm unpack, join) and bucket group %s's stock all-gather is "
             "waited for at its dispatch; overlap_param_gather %s; the same kernels, buffers and order, only the waits "
             "move" % (len(groups), last[0] if last else None, ovl))


def before_allreduce():
    """NVFP4P 13g's weight-amax update, before its max-all-reduce: this rank's partial amaxes of the cut tile-rows"""
    if not _STATE["active"]:
        return
    buf = _STATE["buf"]
    if _STATE["slot_floats"]:
        buf[_STATE["n_params"]:].zero_()
    if _STATE["k1"] is not None:
        k = _STATE["k1"]
        _ext().split_amax2(k[0], k[1], k[2], _CBPC)


# ---------------------------------------------------------------------------------------------------- the check
def check(opt, orig_quantize):
    """at the start of an optimizer step, before the optimizer writes: every gathered weight against TE's 2D cast of its
    re-gathered bf16 weight, and the norms against the re-gathered values"""
    import torch
    _STATE["opt_calls"] = _STATE.get("opt_calls", 0) + 1
    if not _STATE["active"] or _CHECK == "off" or (_CHECK == "first" and _STATE["checks"] > 0):
        return
    _STATE["checks"] += 1
    torch.cuda.synchronize()
    bad, nw, nn, first_bad = 0, 0, 0, None
    group = None
    for G in _STATE["groups"]:
        group = G["group"]
        for B in G["buckets"]:
            S, r = B["S"], B["r"]
            pd = B["bucket"].param_data
            scratch = torch.empty_like(pd)
            torch.distributed.all_gather_into_tensor(scratch, pd[r * S: (r + 1) * S], group=G["group"])
            for W in B["weights"]:
                s, e = B["bucket"].param_to_index[W.p]
                M, K = W.p.shape
                ref = orig_quantize(scratch[s:e].view(M, K), W.q_ref)
                pr = W.proto
                pairs = (("rowwise_data", ref._rowwise_data, W.img_view),
                         ("rowwise_scale_inv", ref._rowwise_scale_inv, pr._rowwise_scale_inv),
                         ("columnwise_data", ref._columnwise_data, pr._columnwise_data),
                         ("columnwise_scale_inv", ref._columnwise_scale_inv, pr._columnwise_scale_inv),
                         ("amax_rowwise", ref._amax_rowwise, pr._amax_rowwise),
                         ("amax_columnwise", ref._amax_columnwise, pr._amax_columnwise))
                for nm, a, b in pairs:
                    if a is None or b is None or a.numel() * a.element_size() != b.numel() * b.element_size():
                        d = 1
                    else:
                        d = int((a.contiguous().view(torch.uint8).reshape(-1)
                                 != b.contiguous().view(torch.uint8).reshape(-1)).sum())
                    if d and first_bad is None:
                        first_bad = (W.name, nm, d)
                    bad += d
                if not (getattr(ref, "_with_gemm_swizzled_scales", None) is True and pr._with_gemm_swizzled_scales):
                    bad += 1
                    first_bad = first_bad or (W.name, "swizzle flag", 1)
                nw += 1
                del ref
            for p, s, e in B["norms"]:
                d = int((p.data.reshape(-1).view(torch.int16) != scratch[s:e].view(torch.int16)).sum())
                if d and first_bad is None:
                    first_bad = ("norm %d" % s, "values", d)
                bad += d
                nn += 1
            del scratch
    t = torch.tensor([bad], dtype=torch.int64, device="cuda")
    if group is not None:
        torch.distributed.all_reduce(t, group=group)
    tot = int(t.item())
    ok = tot == 0
    _say("gathered weights checked (check %d, optimizer step %d): %d weights x 6 buffers and %d norms against TE's 2D cast "
         "of the re-gathered bf16 weights (tex.quantize, the weight's quantizer): %d bytes differ over all ranks%s; "
         "gathers %d, workspace updates skipped %d, wrappers %d: %s" % (
             _STATE["checks"], _STATE["opt_calls"] - 1, nw, nn, tot,
             "" if first_bad is None else " (rank 0's first: %s)" % (first_bad,), _STATE["gathers"],
             _STATE["noop_calls"], _STATE["wrap_calls"], ok))
    if not ok:
        raise RuntimeError("[auto_kernel fp4g] E0 self-check FAILED: %d bytes differ (first on rank 0: %s)" %
                           (tot, first_bad))


# ---------------------------------------------------------------------------------------------------- the import hook
if _ARMED:
    import importlib.abc
    import importlib.util

    _TARGET = "megatron.core.distributed.param_and_grad_buffer"

    def _patch(mod):
        if _NOOVERLAP and not hasattr(mod, "_ParamAndGradBucketGroup"):
            raise RuntimeError("[auto_kernel fp4g] AK_FP4G_NOOVERLAP=1 cannot install: %s has no "
                               "_ParamAndGradBucketGroup" % _TARGET)
        cls = mod._ParamAndGradBucketGroup
        if getattr(cls.start_param_sync, "_ak_fp4g", False):
            return
        w = _wrap_start_param_sync(cls.start_param_sync)
        w._ak_fp4g = True
        cls.start_param_sync = w
        _STATE["patched"] = True

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
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
