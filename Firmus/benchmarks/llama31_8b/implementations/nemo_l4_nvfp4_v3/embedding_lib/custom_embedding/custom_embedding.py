# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from typing import Tuple, Union
import custom_embedding.embedding_ops as embedding_ops
import os as _ak_os

# --- auto_kernel EMB (2026-09-26, EXPERIMENT_LOG block 10 sec 101): the embedding weight gradient in Triton, byte-exact --
# Armed by AK_EMB=1.  The backward below zero-fills the persistent [128256, 2880] bf16 gradient (183 us), copies the
# [b, s, h] view of the gradient contiguous (167 us) and runs embedding_backward_feature_kernel (90 blocks walking all
# 24,576 rows: 1,180 us).  AK_EMB_KERNEL_DIR/embedding_bwd.py (the embedding_bwd call-site task's module, graded byte for
# byte against this extension) writes the same bytes into the same persistent buffer.  Only the recipe's call takes the
# fast path (bf16 gradient that is the [b, s, h] view of a contiguous [s, b, h] tensor, int64 [b, s] ids).  The first call
# (outside graph capture) byte-compares both on a compact slice (the first 1024 positions of every sequence, ids renumbered
# to 0..U-1) -- no second 739 MB buffer; rank 0 prints the ARMED line and a mismatch raises.
# (sec 168: ported to the 20260929 image -- this file and callback_warmup.py's custom_embed_forward swap are unchanged
# there; the hunks are the 0828 ones.)
_AK_EMB =_ak_os.environ.get("AK_EMB", "0") == "1"
# --- auto_kernel EMB-MG (2026-10-07, Llama 3.1 8B): the gradient written straight into the weight's main_grad ----------
# With AK_EMB_MAIN_GRAD=1 (and the x7dense2 overlay, which then tags the embedding as a store-path weight: its main_grad is
# not zeroed each step and its first write is sentinel-checked) the backward writes the gradient into weight.main_grad,
# marks grad_added_to_main_grad and returns a detached alias, so Megatron's hook adds nothing: the shipped
# `zeroed main_grad += gradient` (0.45 ms on the path to the last bucket's reduce-scatter, plus the main_grad's 525 M-element
# zeroing at the step start) becomes one write.  0 + x = x except -0.0 -> +0.0; the kernel stores -0.0 as +0.0 there
# (normz).  The first call (outside capture) compares main_grad with zeros + the stock-path gradient, byte for byte.
_AK_EMB_MG = _AK_EMB and _ak_os.environ.get("AK_EMB_MAIN_GRAD", "0") == "1"
_AK_EMB_STATE_MG = {"checked": None}
_AK_EMB_STATE = {"inst": None, "path": None, "checked": None}


def _ak_emb_rank():
    try:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else int(_ak_os.environ.get("RANK", "0"))
    except Exception:  # noqa: BLE001
        return 0


def _ak_emb_instance():
    st = _AK_EMB_STATE
    if st["inst"] is None:
        import sys
        kdir = _ak_os.environ.get("AK_EMB_KERNEL_DIR", "/opt/firmus/ak/emb")
        if kdir not in sys.path:
            sys.path.insert(0, kdir)
        import embedding_bwd as _mod
        st["inst"] = _mod.EmbeddingBwd(device="cuda")
        st["path"] = _mod.__file__
    return st["inst"]


def _ak_emb_selfcheck(g_sbh, indices, h):
    n = min(1024, g_sbh.shape[0])
    gs = g_sbh[:n].contiguous()
    ids = indices[:, :n].contiguous()
    uniq = torch.unique(ids)
    ids_c = torch.searchsorted(uniq, ids).to(ids.dtype).contiguous()
    u = int(uniq.numel())
    ref = torch.empty((u, h), dtype=gs.dtype, device=gs.device)
    ref.zero_()
    ref = embedding_ops.embedding_bwd(gs.transpose(0, 1), ids_c, u, ref)
    mine = torch.full((u, h), 7.0, dtype=gs.dtype, device=gs.device)
    mine = _ak_emb_instance()(gs, ids_c, u, mine)
    ok = bool(torch.equal(ref.view(torch.int16), mine.view(torch.int16)))
    if _ak_emb_rank() == 0:
        import hashlib
        sha = hashlib.sha256(open(_AK_EMB_STATE["path"], "rb").read()).hexdigest()[:16]
        print("[auto_kernel emb] embedding weight gradient ARMED: module {} sha256={} byte-exact vs embedding_ops.embedding_bwd "
              "on [{}, {}, {}] ({} tokens): {}".format(_AK_EMB_STATE["path"], sha, n, gs.shape[1], h, u, ok), flush=True)
    if not ok:
        raise RuntimeError("[auto_kernel emb] self-check against the extension FAILED")
    return ok


def _ak_emb_ok(grad_output, indices, num_embeddings, buf):
    if not _AK_EMB or grad_output.dim() != 3 or indices.dim() != 2 or grad_output.dtype != torch.bfloat16:
        return False
    g_sbh = grad_output.transpose(0, 1)
    return (g_sbh.is_contiguous() and indices.dtype == torch.int64 and indices.is_contiguous()
            and tuple(indices.shape) == tuple(grad_output.shape[:2]) and buf.is_contiguous()
            and tuple(buf.shape) == (num_embeddings, grad_output.shape[2]) and buf.dtype == grad_output.dtype)

global_embed_wgrad = None
class CustomEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight,
        indices,
        num_embeddings,
    ):
        ctx.num_embeddings = num_embeddings
        ctx.save_for_backward(indices)
        if _AK_EMB_MG:
            ctx.ak_weight = weight
        return embedding_ops.embedding_fwd(weight, indices).clone()

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        (indices,) = ctx.saved_tensors
        global global_embed_wgrad
        w = getattr(ctx, "ak_weight", None)
        mg = getattr(w, "main_grad", None) if w is not None else None
        if (_AK_EMB_MG and mg is not None and getattr(w, "_ak_x7_store", False)
                and _ak_emb_ok(grad_output, indices, ctx.num_embeddings, mg)):  # auto_kernel EMB-MG
            g_sbh = grad_output.transpose(0, 1)
            if _AK_EMB_STATE["checked"] is None and not torch.cuda.is_current_stream_capturing():
                _AK_EMB_STATE["checked"] = _ak_emb_selfcheck(g_sbh, indices, grad_output.shape[2])
            if _AK_EMB_STATE["checked"] is None:
                raise RuntimeError("[auto_kernel emb] first call inside graph capture: the self-check never ran")
            if _AK_EMB_STATE_MG["checked"] is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("[auto_kernel emb] main_grad path first called inside graph capture: its check never ran")
                if global_embed_wgrad is None:
                    global_embed_wgrad = torch.empty_like(mg)
                ref_g = _ak_emb_instance()(g_sbh, indices, ctx.num_embeddings, global_embed_wgrad)
                ref = torch.zeros_like(mg)
                ref.add_(ref_g)
                _ak_emb_instance()(g_sbh, indices, ctx.num_embeddings, mg, normz=True)
                n_bad = int((ref.view(torch.int16) != mg.view(torch.int16)).sum().item())
                n_negz = int((ref_g.view(torch.int16) == -32768).sum().item())
                ok = n_bad == 0
                if _ak_emb_rank() == 0:
                    print("[auto_kernel emb] embedding gradient written into main_grad ARMED: {} {}; -0.0 in the stock-path "
                          "gradient {}; byte-exact v zeros + the stock-path gradient ({} differ): {}".format(
                              list(mg.shape), mg.dtype, n_negz, n_bad, ok), flush=True)
                del ref, ref_g
                if not ok:
                    raise RuntimeError("[auto_kernel emb] E0 self-check FAILED for the main_grad write ({} of {} differ)"
                                       .format(n_bad, mg.numel()))
                _AK_EMB_STATE_MG["checked"] = True
                global_embed_wgrad = None          # the persistent buffer is not used again
            else:
                _ak_emb_instance()(g_sbh, indices, ctx.num_embeddings, mg, normz=True)
            w.grad_added_to_main_grad = True
            return (mg.detach(), None, None,)
        if _AK_EMB_MG and mg is not None and getattr(w, "_ak_x7_store", False):
            # a store-path main_grad is not zeroed each step, so the shipped add is not available here
            raise RuntimeError("[auto_kernel emb] AK_EMB_MAIN_GRAD: the backward is not the recipe's call "
                               "(grad {} {}, ids {} {}); unset AK_EMB_MAIN_GRAD".format(
                                   tuple(grad_output.shape), grad_output.dtype, tuple(indices.shape), indices.dtype))
        if global_embed_wgrad is None:
            global_embed_wgrad = torch.empty((ctx.num_embeddings, grad_output.size(-1)), dtype=grad_output.dtype, device=grad_output.device)
        if _ak_emb_ok(grad_output, indices, ctx.num_embeddings, global_embed_wgrad):  # auto_kernel EMB
            g_sbh = grad_output.transpose(0, 1)
            if _AK_EMB_STATE["checked"] is None and not torch.cuda.is_current_stream_capturing():
                _AK_EMB_STATE["checked"] = _ak_emb_selfcheck(g_sbh, indices, grad_output.shape[2])
            if _AK_EMB_STATE["checked"] is None:
                raise RuntimeError("[auto_kernel emb] first call inside graph capture: the self-check never ran")
            global_embed_wgrad = _ak_emb_instance()(g_sbh, indices, ctx.num_embeddings, global_embed_wgrad)
            return (global_embed_wgrad, None, None,)
        global_embed_wgrad.detach().zero_()
        global_embed_wgrad = embedding_ops.embedding_bwd(grad_output, indices, ctx.num_embeddings, global_embed_wgrad)
        return (global_embed_wgrad, None, None,)
