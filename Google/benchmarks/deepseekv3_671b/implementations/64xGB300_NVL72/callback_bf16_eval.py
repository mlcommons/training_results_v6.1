"""Graphed BF16-only eval: the production eval runs in BF16 instead of FP4/FP8.

Gated by BF16_EVAL=1. Quantization (config.fp4/fp8) is disabled around every
validation forward: during the warmup validation passes (where the validation
CUDA graph is captured, baking BF16 compute into the graph) and during real
evals (so eager fallback paths, e.g. PagedStashRunner overflow re-runs, match
the graph). Training forwards are untouched. eval_accuracy, run_stop, and all
MLPerf logging then natively use the BF16-scored loss.

Weights stay in their quantized storage; TE dequantizes them on the fly for
the BF16 GEMMs. Attention keeps its module-level MXFP8 DPA recipe.
"""

import os
from contextlib import contextmanager

from megatron.core.utils import get_model_config


def bf16_eval_enabled() -> bool:
    return os.environ.get("BF16_EVAL", "0") == "1"


@contextmanager
def bf16_forward_ctx(model):
    """Disable quantized linear compute (config-driven) for the duration."""
    model_config = get_model_config(model[0])
    saved_fp4, saved_fp8 = model_config.fp4, model_config.fp8
    model_config.fp4 = False
    model_config.fp8 = None
    try:
        yield
    finally:
        model_config.fp4 = saved_fp4
        model_config.fp8 = saved_fp8


class BF16EvalCallback:
    """Flip quantization off for the span of each production eval."""

    def __init__(self):
        self._ctx = None

    def on_eval_start(self, context):
        self._ctx = bf16_forward_ctx(context.model)
        self._ctx.__enter__()

    def on_eval_end(self, context):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
