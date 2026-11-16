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

"""Tests for fix 11 (hs_host_sync.py, HS_HOST_SYNC=1).

  docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ipc=host \\
      -e ROCR_VISIBLE_DEVICES=<n> -e HIP_VISIBLE_DEVICES=0 -v $PWD/patches:/p:ro --entrypoint /opt/venv/bin/python \\
      hyperspindle/llama2_70b_lora:v51-hs /p/test_hs_host_sync.py

1. Optimizer: NeMo's MainParamsOptimizerWrapper around AdamW on a BF16 model with the async grad all-reduce path
   forced on (world size 1): the hooks capture one Work per parameter, step() waits on them (no synchronize), and
   the updated parameters equal NeMo's original step() from the same state, bit for bit.
2. Lightning: the logged scalar built by the patched __to_tensor equals the original's (value, dtype, device,
   shape) for int / float / bool / tensor and launches no host-to-device memcpy; prog_bar is dropped only when the
   trainer has no progress-bar callback.
3. Pinned batch: the dict's CPU tensors come out pinned and equal.
"""

import os
import sys

os.environ.setdefault("HS_HOST_SYNC", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import hs_host_sync as hsy  # noqa: E402  (patches NeMo / Lightning classes before they are used)
import test_hs_lora_epilogue as T  # noqa: E402

DEV = "cuda"


def _wrapper(seed):
    from nemo.core.optim.optimizer_with_main_params import MainParamsOptimizerWrapper
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 64)).to(DEV, torch.bfloat16)
    opt = torch.optim.AdamW([{"params": list(model.parameters())}], lr=1e-3)
    w = MainParamsOptimizerWrapper(opt, fp32_grad_accum=True, contiguous_grad_bucket=True, async_grad_allreduce=True,
                                   grad_div_ar_fusion=False, grad_allreduce_chunk_size_mb=0)
    # world size 1 turns the async path off; force it so the hooks issue the (1-rank) all-reduces
    w._async_grad_allreduce = True
    w._require_backward_grad_sync = True
    w._grad_allreduce_chunk_size_mb = 0
    return model, w


def _train(model, w, steps, orig_step=None):
    gen = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(steps):
        x = torch.randn(32, 256, generator=gen, device=DEV).to(torch.bfloat16)
        w.zero_grad()
        model(x).float().pow(2).mean().backward()
        if orig_step is not None:
            orig_step(w)
        else:
            w.step()
    return [p.detach().clone() for p in model.parameters()]


def test_optimizer_waits_and_matches():
    T._init_parallel()
    from nemo.core.optim.optimizer_with_main_params import MainParamsOptimizerWrapper
    assert MainParamsOptimizerWrapper.step is not None
    import inspect
    patched_src = inspect.getsource(MainParamsOptimizerWrapper.step)
    assert "wait_pending" in patched_src, "hs_host_sync did not patch step()"

    def nemo_step(w):   # NeMo's original step: synchronize, then the optimizer and the main->model copy
        torch.cuda.synchronize()
        w.optimizer.step(closure=None)
        with torch.no_grad():
            w._copy_main_params_to_model_params()

    m0, w0 = _wrapper(1)
    p_ref = _train(m0, w0, 3, orig_step=nemo_step)
    st = dict(hsy.STATS)
    m1, w1 = _wrapper(1)
    p_new = _train(m1, w1, 3)
    d = {k: hsy.STATS[k] - st[k] for k in hsy.STATS}
    n_params = len(list(m1.parameters()))
    nd = [int((a.view(torch.int16) != b.view(torch.int16)).sum()) for a, b in zip(p_ref, p_new)]
    print(f"  {n_params} params, 3 steps: works captured {d['works']}, waited {d['waits']}, fallback syncs "
          f"{d['fallback_sync']}; param differences {nd}")
    assert d["works"] == 3 * n_params and d["waits"] == 3 * n_params and d["fallback_sync"] == 0, d
    assert all(n == 0 for n in nd)
    print("PASS optimizer: all-reduce handles waited on-stream, parameters bit-identical to NeMo's step()")


def test_waits_before_clipping():
    """The all-reduces must be waited when the step's forward/backward returns, i.e. before Lightning's gradient
    clipping reads the main grads (version 1 waited only in optimizer.step(), after the clipping)."""
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    import inspect
    assert "wait_pending" in inspect.getsource(MegatronGPTModel.training_step_fwd_bwd_step_call)
    T._init_parallel()
    model, w = _wrapper(3)
    x = torch.randn(32, 256, device=DEV).to(torch.bfloat16)
    order = []

    class _W:
        def __init__(self, inner): self.inner = inner
        def wait(self):
            order.append("wait")
            return self.inner.wait()

    class Fake:
        _optimizer = w

    def fake_fwd_bwd(self, it, forward_only):
        w.zero_grad()
        model(x).float().pow(2).mean().backward()
        w._hs_works = [_W(k) for k in w.__dict__.get("_hs_works", [])]
        return torch.zeros(())

    wrapped = MegatronGPTModel.training_step_fwd_bwd_step_call
    orig = wrapped.__closure__[[c.cell_contents.__name__ if callable(c.cell_contents) else "" for c in wrapped.__closure__].index("training_step_fwd_bwd_step_call")]
    saved = orig.cell_contents
    try:
        orig.cell_contents = fake_fwd_bwd
        st = dict(hsy.STATS)
        wrapped(Fake(), None, False)
        order.append("clip")          # what Lightning does next: configure_gradient_clipping
        w.step()
        d = {k: hsy.STATS[k] - st[k] for k in hsy.STATS}
    finally:
        orig.cell_contents = saved
    n = len(list(model.parameters()))
    print(f"  order {order[:2]} ... {order[-2:]}; stats {d}")
    assert order.index("clip") == n and order.count("wait") == n, order
    assert d["waited_after_bwd"] == 1 and d["fallback_sync"] == 0, d
    print("PASS all-reduces waited at the end of the backward, before clipping; step() adds no synchronize")


def test_lightning_to_tensor():
    from lightning.pytorch.core.module import LightningModule

    class M(LightningModule):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(2, 2)

    m = M().to(DEV)
    mangled = "_LightningModule__to_tensor"
    patched = getattr(LightningModule, mangled)
    orig = patched.__closure__[0].cell_contents if patched.__closure__ else None
    # the original is the closure's first cell (orig_to_tensor); find it by type
    orig = next(c.cell_contents for c in patched.__closure__ if callable(c.cell_contents) and c.cell_contents is not patched
                and getattr(c.cell_contents, "__name__", "") == "__to_tensor")
    vals = [3, 0.1, -2.5, True, torch.tensor(1.5, device=DEV), torch.tensor([7.0])]
    for v in vals:
        a = orig(m, v, "x")
        b = patched(m, v, "x")
        assert a.dtype == b.dtype and a.device == b.device and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()), (v, a, b)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for v in (3, 0.1, -2.5):
            patched(m, v, "x")
        torch.cuda.synchronize()
    copies = [e.key for e in p.key_averages() if "memcpy" in e.key.lower() or "htod" in e.key.lower()]
    print(f"  patched __to_tensor on numbers: device events {[e.key for e in p.key_averages() if e.self_device_time_total > 0]}")
    assert not copies, copies

    class Tr:
        progress_bar_callback = None
    m._trainer = Tr()
    assert hsy._prog_bar(m, True) is False and hsy._prog_bar(m, False) is False
    Tr.progress_bar_callback = object()
    assert hsy._prog_bar(m, True) is True
    print("PASS lightning: scalars logged with a fill kernel (no memcpy), prog_bar dropped only without a bar")


def test_pin_batch():
    batch = {"tokens": torch.randint(0, 100, (8, 64)), "loss_mask": torch.rand(8, 64), "meta": [1, 2],
             "gpu": torch.zeros(2, device=DEV)}
    ref = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    out = hsy.pin_batch(batch)
    assert out is batch
    for k in ("tokens", "loss_mask"):
        assert out[k].is_pinned() and torch.equal(out[k], ref[k])
    assert out["gpu"].device.type == "cuda" and out["meta"] == [1, 2]
    print("PASS pinned batch: CPU tensors pinned and equal, others untouched")


if __name__ == "__main__":
    for t in (test_optimizer_waits_and_matches, test_waits_before_clipping, test_lightning_to_tensor, test_pin_batch):
        print(f"== {t.__name__}", flush=True)
        t()
    print("ALL PASSED")
