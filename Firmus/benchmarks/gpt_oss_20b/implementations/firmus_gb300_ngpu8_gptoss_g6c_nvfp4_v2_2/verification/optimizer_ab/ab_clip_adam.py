# Copyright (c) 2026, Firmus Technologies. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Classification: Internal
# AI-assisted (Cursor) - reviewed by Firmus.
"""A/B comparison of the fused clip+Adam extension against the reference optimizer path.

Three optimizers start from identical parameters and see identical gradients for STEPS steps:

  ref   Megatron-Core clip (get_grad_norm_fp32 + clip_grad_by_total_norm_fp32) followed by
        Transformer Engine FusedAdam, configured as Megatron builds it for the precision-aware
        optimizer (BF16 params, FP32 master stored as int16 remainders, decoupled grads).
  ext   The same TE FusedAdam with the submission's code paths enabled exactly as in
        config_GB300_2x4x3xtp1pp1cp1ep1_nvfp4_ag6.sh: NVFP4_FAST_L2NORM (single-launch norm),
        NVFP4_CLIP_IN_ADAM (clip coefficient applied inside the Adam kernel), NVFP4_ADAM_AMAX
        (amax of the updated weights folded into the kernel) and NVFP4_PHASED_ADAM_AG (the step
        runs on parameter subsets one after another).
  apex  apex.optimizers.FusedAdam (adam_w_mode=True) on FP32 copies of the parameters, fed the
        unclipped gradients and clipped on its own copy with the same Megatron-Core functions.

ext is compared with ref bit for bit; ext is compared with apex numerically. The FP32-parameter
kernel of the extension (no remainders) is checked the same way. Each path is run twice: on the
recipe's values, and on edge inputs (+0, -0 and subnormal parameters and gradients, and steps
with lr = 0).
"""

import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "/workspace/llm")

import nvfp4_clip_adam as clip_adam  # noqa: E402
from apex.optimizers import FusedAdam as ApexFusedAdam  # noqa: E402
from megatron.core.optimizer import clip_grads  # noqa: E402
from transformer_engine.pytorch.optimizers import FusedAdam as TEFusedAdam  # noqa: E402

EXT = os.environ.get("NVFP4_CLIP_ADAM_EXT", "/opt/nvfp4_ext/nvfp4_clip_adam_ext_fm9.so")
STEPS = int(os.environ.get("AB_STEPS", "20"))
PHASES = int(os.environ.get("AB_PHASES", "4"))

# Hyperparameters from pretrain.py get_optimizer_config / config_*.sh.
LR = 5e-4
WARMUP = 256
BETAS = (0.9, 0.95)
EPS = 1e-5
WD = 0.1
MAX_NORM = 1.0

# Weight shapes from the model (hidden 2880, padded expert width 2944, FC1 output 2 x 2880),
# plus odd sizes that exercise the unaligned tail of the vectorised kernel and multi-chunk tensors.
WD_SHAPES = [(5760, 2944), (2880, 2944), (128256, 64), (65536 * 2 + 3,), (131,)]
NO_WD_SHAPES = [(2880,), (5760,), (1,), (4097,)]


def lr_at(step, edge=False):
    if edge and step % 5 == 0:
        return 0.0
    return LR * step / WARMUP


def edge_fill(t, values):
    """Set elements i, i + n, i + 2n, ... of t to values[i]; the other elements keep their value."""
    flat = t.view(-1)
    n = len(values) + 3
    for i, v in enumerate(values):
        flat[i::n] = v
    return t


def master_fp32(p_bf16, remainder):
    """FP32 master weight held by TE as BF16 high half + int16 remainder."""
    hi = p_bf16.view(torch.int16).to(torch.int32)
    rem = remainder.to(torch.int32)
    hi = hi - (rem < 0).to(torch.int32)
    bits = (hi << 16) | (rem & 0xFFFF)
    return bits.view(torch.float32)


def ulp_diff(a, b):
    ai, bi = a.view(torch.int32).to(torch.int64), b.view(torch.int32).to(torch.int64)
    same_sign = (ai < 0) == (bi < 0)
    d = (ai - bi).abs()
    return int(d[same_sign].max().item()) if same_sign.any() else 0, int((~same_sign).sum().item())


def rel_diff(a, b):
    a, b = a.double(), b.double()
    diff = (a - b).abs()
    return float(diff.max().item()), float((diff / b.abs().clamp_min(1e-30)).max().item())


GRAD_EDGE = (0.0, -0.0, 1e-41, -1e-41, 1e-39)
PARAM_EDGE = (0.0, -0.0, 1e-39, -1e-39)


def grads_for_step(gen, shapes, step, device, edge=False):
    # Alternate gradient magnitudes so that some steps are clipped (norm > 1) and some are not.
    scale = 3e-3 if step % 2 else 3e-5
    grads = [torch.randn(s, generator=gen, device=device) * scale for s in shapes]
    if edge:
        grads = [edge_fill(g, GRAD_EDGE) for g in grads]
    return grads


def init_params(gen, shapes, device, dtype, edge=False):
    init = [torch.randn(s, generator=gen, device=device) * 0.02 for s in shapes]
    if edge:
        init = [edge_fill(t, PARAM_EDGE) for t in init]
    return [t.to(dtype) for t in init]


def apex_step(opt_apex, p_apex, grads):
    for p, g in zip(p_apex, grads):
        p.decoupled_grad = g.clone()
    norm = clip_grads.get_grad_norm_fp32([p.decoupled_grad for p in p_apex])
    clip_grads.clip_grad_by_total_norm_fp32(p_apex, MAX_NORM, norm, use_decoupled_grad=True)
    for p in p_apex:
        p.grad = p.decoupled_grad
    opt_apex.step()


def make_groups(params):
    n_wd = len(WD_SHAPES)
    return [
        {"params": params[:n_wd], "weight_decay": WD},
        {"params": params[n_wd:], "weight_decay": 0.0},
    ]


def ref_clip(params):
    grads = [p.decoupled_grad for p in params]
    norm = clip_grads.get_grad_norm_fp32(grads)
    clip_grads.clip_grad_by_total_norm_fp32(params, MAX_NORM, norm, use_decoupled_grad=True)
    return norm


def ext_clip(params):
    """pretrain.py: _clip_by_norm_deferred with NVFP4_FAST_L2NORM's l2_norm_impl."""
    grads = [p.decoupled_grad for p in params]
    norm = clip_grads.get_grad_norm_fp32(grads)
    assert isinstance(norm, torch.Tensor), "pretrain.py only defers the clip for a tensor norm"
    coeff = MAX_NORM / (norm + 1.0e-6)
    coeff.clamp_max_(1.0)
    assert clip_adam.defer_clip([g.detach() for g in grads], coeff)
    return norm


def phased_step(opt, params):
    """pretrain.py: _install_phased_adam.run(), stepping parameter subsets in turn."""
    orig_groups = opt.param_groups
    new_steps = [None] * len(orig_groups)
    order = list(params)
    bounds = [round(i * len(order) / PHASES) for i in range(PHASES + 1)]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        keep = {id(p) for p in order[lo:hi]}
        subs = [dict(g, params=[p for p in g["params"] if id(p) in keep]) for g in orig_groups]
        opt.param_groups = subs
        try:
            opt.step()
        finally:
            opt.param_groups = orig_groups
        for i, sub in enumerate(subs):
            if sub["params"] and "step" in sub:
                new_steps[i] = sub["step"]
    for g, s in zip(orig_groups, new_steps):
        if s is not None:
            g["step"] = s


def run_remainder(device, edge=False):
    gen = torch.Generator(device=device).manual_seed(20261009)
    shapes = WD_SHAPES + NO_WD_SHAPES
    init = init_params(gen, shapes, device, torch.bfloat16, edge)

    p_ref = [torch.nn.Parameter(t.clone()) for t in init]
    p_ext = [torch.nn.Parameter(t.clone()) for t in init]
    p_apex = [torch.nn.Parameter(t.float().clone()) for t in init]

    te_kwargs = dict(lr=LR, betas=BETAS, eps=EPS, adam_w_mode=True, master_weights=True,
                     master_weight_dtype=torch.float32, use_decoupled_grad=True,
                     store_param_remainders=True)
    opt_ref = TEFusedAdam(make_groups(p_ref), **te_kwargs)
    opt_ext = TEFusedAdam(make_groups(p_ext), **te_kwargs)
    opt_apex = ApexFusedAdam(make_groups(p_apex), lr=LR, betas=BETAS, eps=EPS, adam_w_mode=True)

    init_master = torch.cat([t.float().flatten() for t in init])
    slot = {p.data_ptr(): i for i, p in enumerate(p_ext)}
    amax_buf = torch.zeros(len(p_ext), dtype=torch.float32, device=device)

    rows = []
    for step in range(1, STEPS + 1):
        grads = grads_for_step(gen, shapes, step, device, edge)
        for opt in (opt_ref, opt_ext, opt_apex):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, edge)
        for p, g in zip(p_ref, grads):
            p.decoupled_grad = g.clone()
        for p, g in zip(p_ext, grads):
            p.decoupled_grad = g.clone()

        norm_ref = ref_clip(p_ref)
        opt_ref.step()

        clip_grads.l2_norm_impl = l2_wrapped
        try:
            norm_ext = ext_clip(p_ext)
        finally:
            clip_grads.l2_norm_impl = l2_orig
        clip_adam.install(opt_ext)
        amax_buf.zero_()
        clip_adam.set_amax_target(amax_buf, lambda t: slot.get(t.data_ptr(), -1))
        phased_step(opt_ext, p_ext)
        covered = clip_adam.clear_amax_target()
        leftover = len(clip_adam._pending["grads"])
        clip_adam.finish()

        apex_step(opt_apex, p_apex, grads)
        torch.cuda.synchronize()

        def st(opt, p, k):
            return opt.state[p][k]

        mism = {"param": 0, "remainder": 0, "exp_avg": 0, "exp_avg_sq": 0}
        for a, b in zip(p_ref, p_ext):
            mism["param"] += int((a.view(torch.int16) != b.view(torch.int16)).sum())
            for k, key in (("remainder", "master_param"), ("exp_avg", "exp_avg"),
                           ("exp_avg_sq", "exp_avg_sq")):
                x, y = st(opt_ref, a, key), st(opt_ext, b, key)
                mism[k] += int((x.view(torch.int32 if x.dtype == torch.float32 else torch.int16)
                                != y.view(torch.int32 if y.dtype == torch.float32 else torch.int16)).sum())
        amax_mism = sum(
            int(not torch.equal(amax_buf[i : i + 1], p.detach().abs().amax().float().view(1)))
            for i, p in enumerate(p_ext)
        )

        m_ext = torch.cat([master_fp32(p, st(opt_ext, p, "master_param")).flatten() for p in p_ext])
        m_apex = torch.cat([p.detach().flatten() for p in p_apex])
        ea_ext = torch.cat([st(opt_ext, p, "exp_avg").flatten() for p in p_ext])
        ea_apex = torch.cat([opt_apex.state[p]["exp_avg"].flatten() for p in p_apex])
        es_ext = torch.cat([st(opt_ext, p, "exp_avg_sq").flatten() for p in p_ext])
        es_apex = torch.cat([opt_apex.state[p]["exp_avg_sq"].flatten() for p in p_apex])
        bf_ext = torch.cat([p.detach().flatten() for p in p_ext])
        ulp_max, sign_flips = ulp_diff(m_ext, m_apex)
        bf_differs = bf_ext.view(torch.int16) != m_apex.to(torch.bfloat16).view(torch.int16)
        # TE stores the BF16 param rounded half away from zero; torch's cast rounds half to even.
        bf_tie = (m_apex.view(torch.int32) & 0xFFFF) == 0x8000
        rows.append({
            "step": step,
            "lr": lr_at(step, edge),
            "grad_norm": float(norm_ref),
            "clipped": float(norm_ref) > MAX_NORM,
            "norm_bitwise_equal": torch.tensor(float(norm_ref)).view(torch.int32).item()
            == torch.tensor(float(norm_ext)).view(torch.int32).item(),
            "ext_vs_ref_mismatches": mism,
            "amax_slots_mismatched": amax_mism,
            "amax_slots_covered": len(covered),
            "grads_left_for_finish": leftover,
            "ext_vs_apex": {
                "master_max_abs": rel_diff(m_ext, m_apex)[0],
                "master_max_rel": rel_diff(m_ext, m_apex)[1],
                "master_max_ulp": ulp_max,
                "master_sign_differences": sign_flips,
                "exp_avg_max_rel": rel_diff(ea_ext, ea_apex)[1],
                "exp_avg_sq_max_rel": rel_diff(es_ext, es_apex)[1],
                "bf16_param_differs_from_rne_of_apex_master": int(bf_differs.sum()),
                "of_which_exact_bf16_ties": int((bf_differs & bf_tie).sum()),
            },
            "master_max_abs_change_from_init": float(
                (m_ext - init_master).abs().max().item()),
            "master_elements_changed_from_init": int((m_ext != init_master).sum().item()),
        })
    return rows, sum(t.numel() for t in init)


def run_plain(device, edge=False):
    gen = torch.Generator(device=device).manual_seed(20261010)
    shapes = WD_SHAPES + NO_WD_SHAPES
    init = init_params(gen, shapes, device, torch.float32, edge)
    p_ref = [torch.nn.Parameter(t.clone()) for t in init]
    p_ext = [torch.nn.Parameter(t.clone()) for t in init]
    p_apex = [torch.nn.Parameter(t.clone()) for t in init]
    te_kwargs = dict(lr=LR, betas=BETAS, eps=EPS, adam_w_mode=True, use_decoupled_grad=True)
    opt_ref = TEFusedAdam(make_groups(p_ref), **te_kwargs)
    opt_ext = TEFusedAdam(make_groups(p_ext), **te_kwargs)
    opt_apex = ApexFusedAdam(make_groups(p_apex), lr=LR, betas=BETAS, eps=EPS, adam_w_mode=True)
    rows = []
    for step in range(1, STEPS + 1):
        grads = grads_for_step(gen, shapes, step, device, edge)
        for opt in (opt_ref, opt_ext, opt_apex):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, edge)
        for p, g in zip(p_ref, grads):
            p.decoupled_grad = g.clone()
        for p, g in zip(p_ext, grads):
            p.decoupled_grad = g.clone()
        norm_ref = ref_clip(p_ref)
        opt_ref.step()
        clip_grads.l2_norm_impl = l2_wrapped
        try:
            ext_clip(p_ext)
        finally:
            clip_grads.l2_norm_impl = l2_orig
        clip_adam.install(opt_ext)
        phased_step(opt_ext, p_ext)
        clip_adam.finish()
        apex_step(opt_apex, p_apex, grads)
        torch.cuda.synchronize()
        mism = 0
        for a, b in zip(p_ref, p_ext):
            mism += int((a.detach().view(torch.int32) != b.detach().view(torch.int32)).sum())
            for key in ("exp_avg", "exp_avg_sq"):
                mism += int((opt_ref.state[a][key].view(torch.int32)
                             != opt_ext.state[b][key].view(torch.int32)).sum())
        m_ext = torch.cat([p.detach().flatten() for p in p_ext])
        m_apex = torch.cat([p.detach().flatten() for p in p_apex])
        ulp_max, sign_flips = ulp_diff(m_ext, m_apex)
        rows.append({
            "step": step,
            "clipped": float(norm_ref) > MAX_NORM,
            "ext_vs_ref_mismatches": mism,
            "ext_vs_apex_param_max_rel": rel_diff(m_ext, m_apex)[1],
            "ext_vs_apex_param_max_ulp": ulp_max,
            "ext_vs_apex_sign_differences": sign_flips,
        })
    return rows


def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29571", rank=0, world_size=1)
    clip_adam.load_prebuilt(EXT)

    global l2_orig, l2_wrapped
    l2_orig = clip_grads.l2_norm_impl
    l2_wrapped = clip_adam.wrap_l2norm(l2_orig)

    rem_rows, numel = run_remainder(device)
    fused_before_plain = dict(clip_adam.counters)
    plain_rows = run_plain(device)
    rem_edge_rows, _ = run_remainder(device, edge=True)
    plain_edge_rows = run_plain(device, edge=True)

    from importlib import metadata

    import transformer_engine

    def dist_version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not found"

    report = {
        "image": os.environ.get("AB_IMAGE", "not given"),
        "extension": EXT,
        "torch": torch.__version__,
        "transformer_engine": transformer_engine.__version__,
        "apex": dist_version("apex"),
        "gpu": torch.cuda.get_device_name(device),
        "config": {"lr_peak": LR, "warmup": WARMUP, "betas": BETAS, "eps": EPS,
                   "weight_decay": [WD, 0.0], "max_norm": MAX_NORM, "steps": STEPS,
                   "phases": PHASES, "elements": numel,
                   "edge": {"grad_values": GRAD_EDGE, "param_values": PARAM_EDGE,
                            "lr_zero_steps": [s for s in range(1, STEPS + 1) if s % 5 == 0]}},
        "counters_after_remainder_test": fused_before_plain,
        "counters_final": dict(clip_adam.counters),
        "remainder_path": rem_rows,
        "fp32_path": plain_rows,
        "remainder_path_edge": rem_edge_rows,
        "fp32_path_edge": plain_edge_rows,
    }
    print(json.dumps(report, indent=1))

    def summary(label, rem, plain):
        total = sum(sum(r["ext_vs_ref_mismatches"].values()) + r["amax_slots_mismatched"]
                    + (0 if r["norm_bitwise_equal"] else 1) for r in rem)
        total += sum(r["ext_vs_ref_mismatches"] for r in plain)
        print(f"SUMMARY[{label}] ext_vs_ref_total_mismatches={total} "
              f"ext_vs_apex_master_max_rel={max(r['ext_vs_apex']['master_max_rel'] for r in rem):.3e} "
              f"ext_vs_apex_master_max_ulp={max(r['ext_vs_apex']['master_max_ulp'] for r in rem)} "
              f"ext_vs_apex_master_sign_differences={sum(r['ext_vs_apex']['master_sign_differences'] for r in rem)} "
              f"bf16_non_tie_differences={sum(r['ext_vs_apex']['bf16_param_differs_from_rne_of_apex_master'] - r['ext_vs_apex']['of_which_exact_bf16_ties'] for r in rem)} "
              f"fp32_path_vs_apex_max_ulp={max(r['ext_vs_apex_param_max_ulp'] for r in plain)} "
              f"fp32_path_vs_apex_sign_differences={sum(r['ext_vs_apex_sign_differences'] for r in plain)}")

    summary("recipe", rem_rows, plain_rows)
    summary("edge", rem_edge_rows, plain_edge_rows)
    print(f"SUMMARY fallbacks={clip_adam.counters['fallback']} "
          f"fused_launches={clip_adam.counters['fused']}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
