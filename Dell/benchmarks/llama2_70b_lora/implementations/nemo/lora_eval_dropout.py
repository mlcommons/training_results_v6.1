# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Turn off the LoRA dropout in eval.

Megatron Bridge TEFusedLoRALinear keeps its te.ops branches in a tuple, so they are not
submodules. Thus model.eval() does not reach the te.ops Dropout of the LoRA branch, and the eval
loss uses dropout. The reference (HF PEFT) has no dropout in eval. apply() gives that Dropout
the train or eval mode of its TEFusedLoRALinear module.
"""

import transformer_engine.pytorch as te
from megatron.bridge.peft.lora_layers import TEFusedLoRALinear

_original = {}


def _set_dropout_mode(module: TEFusedLoRALinear, branches: tuple) -> None:
    """Give the te.ops Dropout of the LoRA branch the mode of the module."""
    lora_branch = branches[1]
    for op in lora_branch.modules():
        if isinstance(op, te.ops.Dropout):
            op.train(module.training)


def _train(self: TEFusedLoRALinear, mode: bool = True) -> TEFusedLoRALinear:
    """Replacement of TEFusedLoRALinear.train (model.train() and model.eval() call it)."""
    result = _original["train"](self, mode)
    if self._fused_branches is not None:
        _set_dropout_mode(self, self._fused_branches)
    return result


def _make_fused_branches(self: TEFusedLoRALinear) -> tuple:
    """Replacement of TEFusedLoRALinear._make_fused_branches (the first forward calls it)."""
    branches = _original["make_fused_branches"](self)
    _set_dropout_mode(self, branches)
    return branches


def apply() -> None:
    """Patch TEFusedLoRALinear. Call it before the first forward."""
    if _original:
        return
    _original["train"] = TEFusedLoRALinear.train
    _original["make_fused_branches"] = TEFusedLoRALinear._make_fused_branches
    TEFusedLoRALinear.train = _train
    TEFusedLoRALinear._make_fused_branches = _make_fused_branches
