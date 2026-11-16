# Copyright (c) 2026 Krai Ltd. All Rights Reserved.
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
#
# The step() method installed by this file re-implements MainParamsOptimizerWrapper.step of NVIDIA NeMo
# (nemo/core/optim/optimizer_with_main_params.py), Copyright (c) 2021, NVIDIA CORPORATION. Licensed under the
# Apache License, Version 2.0.

"""Fix 11: remove the host-side synchronisations at the step boundary (env HS_HOST_SYNC=1; no numerics change).

The recipe-A trace (docs/analysis/recipeA_profile_geak_eval.md 2.2) shows 19 ms/step of GPU idle around the
optimizer step: every CPU-only phase (NeMo's grad bookkeeping, the optimizer and LR-scheduler Python, zero_grad,
Lightning's logging, the next batch's preparation, TE's prepare_forward) runs with an empty GPU queue because a
few host-blocking calls stop the CPU from running ahead of the GPU:
  1. NeMo MainParamsOptimizerWrapper.step() calls torch.cuda.synchronize() so that the async data-parallel grad
     all-reduces (issued in the backward hooks with async_op=True, their Work handles discarded) are complete
     before Adam reads the main grads. HS_HOST_SYNC_OPT=1: the hooks keep the Work handles and they are waited
     (work.wait(): a stream wait on the collective's end event, no host block) as soon as the forward/backward of
     the step returns (MegatronGPTModel.training_step_fwd_bwd_step_call), i.e. BEFORE anything reads the main
     grads: Lightning's gradient clipping (configure_gradient_clipping, clip_grad_norm over the main grads) runs
     before optimizer.step(). step() waits on anything left and falls back to the synchronize if no handle was
     captured. (Version 1 waited only in step(): the clipping then read and rescaled grads while RCCL was still
     writing them; the 10-run set 261013-174239-8fc7 converged at 3840-4992 samples. The stock recipe hides the
     same race behind the blocking log copies of item 2, which drain the device before the clipping.)
  2. Lightning's self.log(python_number) builds the value with torch.tensor(value, device=cuda): a pageable
     host-to-device memcpy that blocks the CPU until the GPU queue drains (3-4 per step: lr, global_step,
     consumed_samples, global_batch_size). HS_HOST_SYNC_LOG=1: torch.full((), value, device, dtype) instead, a
     fill kernel with the scalar as a kernel argument (same value, dtype, device, shape).
  3. Metrics logged with prog_bar=True are read back with .item() at the end of every step by Lightning's logger
     connector (3 per step) although the progress bar is disabled in the recipe (enable_progress_bar False).
     HS_HOST_SYNC_LOG=1: prog_bar is forced off when the trainer has no progress-bar callback (callback_metrics
     and logged values are unchanged; only the unused progress_bar_metrics dict loses its entries).
  4. The batch tensors reach the GPU through val.cuda(non_blocking=True) on pageable memory, i.e. 7 synchronous
     memcpys per step. HS_HOST_SYNC_PIN=1: the batch dict is pinned (tensor.pin_memory(), 65 KB each) right
     before NeMo splits it into micro-batches, so the copies are asynchronous.
"""

import os

import torch
import torch.distributed as dist

_ON = os.getenv("HS_HOST_SYNC", "0") == "1"
OPT = _ON and os.getenv("HS_HOST_SYNC_OPT", "1") == "1"
LOG = _ON and os.getenv("HS_HOST_SYNC_LOG", "1") == "1"
PIN = _ON and os.getenv("HS_HOST_SYNC_PIN", "1") == "1"
STATS = {"works": 0, "waits": 0, "waited_after_bwd": 0, "fallback_sync": 0, "log_fill": 0, "pbar_dropped": 0, "pinned": 0}


# --------------------------------------------------------------------------------------- 1. optimizer step

def _install_optimizer():
    from nemo.core.optim import optimizer_with_main_params as owmp

    W = owmp.MainParamsOptimizerWrapper
    orig_make = W._make_param_hook

    def _make_param_hook(self, *args, **kwargs):
        hook = orig_make(self, *args, **kwargs)

        def param_hook(*unused):
            real = dist.all_reduce

            def capture(*a, **k):
                work = real(*a, **k)
                if work is not None:
                    self.__dict__.setdefault("_hs_works", []).append(work)
                    STATS["works"] += 1
                return work

            dist.all_reduce = capture   # the hook calls torch.distributed.all_reduce by name
            try:
                hook(*unused)
            finally:
                dist.all_reduce = real

        return param_hook

    def step(self, **kwargs):
        """NeMo's step without the torch.cuda.synchronize(): the async grad all-reduces are waited on-stream
        (normally already at the end of the backward, see wait_pending)."""
        if self._async_grad_allreduce:
            waited = wait_pending(self) or self.__dict__.pop("_hs_waited", False)
            if not waited:
                torch.cuda.synchronize()
                STATS["fallback_sync"] += 1
        self.__dict__.pop("_hs_waited", None)
        closure = kwargs.pop('closure', None)
        assert closure is None, f"closure should be None but was passed {closure}"
        with torch.no_grad():
            self.optimizer.step(closure=None, **kwargs)
            self._copy_main_params_to_model_params()
        return True

    W._make_param_hook = _make_param_hook
    W.step = step

    # wait for the all-reduces as soon as the backward of the step is done, before gradient clipping reads them
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

    orig_fwd_bwd = MegatronGPTModel.training_step_fwd_bwd_step_call

    def training_step_fwd_bwd_step_call(self, dataloader_iter, forward_only):
        out = orig_fwd_bwd(self, dataloader_iter, forward_only)
        if not forward_only:
            opt = getattr(self, "_optimizer", None)
            if opt is not None and getattr(opt, "_async_grad_allreduce", False) and wait_pending(opt):
                opt._hs_waited = True
                STATS["waited_after_bwd"] += 1
        return out

    MegatronGPTModel.training_step_fwd_bwd_step_call = training_step_fwd_bwd_step_call


def wait_pending(opt):
    """Enqueue a stream wait on every captured all-reduce of `opt`; True if there were any."""
    works = opt.__dict__.pop("_hs_works", None)
    if not works:
        return False
    for w in works:
        w.wait()
    STATS["waits"] += len(works)
    return True


# ------------------------------------------------------------------------------------ 2 + 3. Lightning logging

def _prog_bar(module, prog_bar):
    if prog_bar and getattr(getattr(module, "_trainer", None), "progress_bar_callback", None) is None:
        STATS["pbar_dropped"] += 1
        return False
    return prog_bar


def _install_lightning():
    from lightning.pytorch.core.module import LightningModule
    from lightning.pytorch.trainer.connectors.logger_connector.result import _get_default_dtype

    mangled = "_LightningModule__to_tensor"
    orig_to_tensor = getattr(LightningModule, mangled)

    def __to_tensor(self, value, name):
        if isinstance(value, torch.Tensor) or isinstance(value, bool) or not isinstance(value, (int, float)):
            return orig_to_tensor(self, value, name)
        STATS["log_fill"] += 1
        return torch.full((), value, device=self.device, dtype=_get_default_dtype())   # no host-to-device copy

    setattr(LightningModule, mangled, __to_tensor)

    orig_log = LightningModule.log

    def log(self, name, value, prog_bar=False, *args, **kwargs):
        return orig_log(self, name, value, _prog_bar(self, prog_bar), *args, **kwargs)

    LightningModule.log = log


# ------------------------------------------------------------------------------------------- 4. pinned batch

def pin_batch(batch):
    if isinstance(batch, dict):
        for key, v in batch.items():
            if isinstance(v, torch.Tensor) and v.device.type == "cpu" and not v.is_pinned():
                batch[key] = v.pin_memory()
                STATS["pinned"] += 1
    return batch


def _install_pin():
    from nemo.collections.nlp.models.language_modeling import megatron_gpt_sft_model as sft

    orig_split = sft.get_iterator_k_split

    def get_iterator_k_split(batch, *args, **kwargs):
        return orig_split(pin_batch(batch), *args, **kwargs)

    sft.get_iterator_k_split = get_iterator_k_split   # looked up by name in fwd_bwd_step


if OPT:
    _install_optimizer()
if LOG:
    _install_lightning()
if PIN:
    _install_pin()
print(f"[hyperspindle] hs_host_sync: optimizer waits {'on' if OPT else 'off'}, log fills / no prog_bar readback "
      f"{'on' if LOG else 'off'}, pinned batch {'on' if PIN else 'off'} (HS_HOST_SYNC=1 enables all)", flush=True)


def report():
    return f"[hyperspindle] hs_host_sync: {STATS}"
