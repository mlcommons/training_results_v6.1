
"""Application-side helpers for the unmodified TorchRec sparse pipeline."""

from typing import Any, Callable, Optional

import torch


class ClippingOptimizer:
    """Clip after all backward work and immediately before the dense update.

    The training loop retains the original optimizer for LR warmup and
    checkpointing. Only the pipeline receives this delegating optimizer.
    """

    def __init__(
        self,
        optimizer: Any,
        model: torch.nn.Module,
        max_norm: float,
        zero_grad_pre_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        self._optimizer = optimizer
        self._model = model
        self._max_norm = max_norm
        self._zero_grad_pre_hook = zero_grad_pre_hook

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        if self._zero_grad_pre_hook is not None:
            self._zero_grad_pre_hook()
        self._optimizer.zero_grad(*args, **kwargs)

    def step(self) -> None:
        if self._max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_norm)
        self._optimizer.step()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._optimizer, name)


def pipeline_segment_batches(
    global_step: int,
    window_batch: int,
    window_limit: Optional[int],
    eval_interval: int,
    skip_eval_until_step: int,
    die_at_step: int = -1,
) -> Optional[int]:
    """Limit prefetch to the next eval, window cap, or deliberate test exit.

    Feeding a finite islice to execute_all_batches=True drains the pipeline
    exactly at the boundary, without discarding batches or issuing input_dist
    twice when detach()/attach() is used for evaluation. A new islice per
    segment also resets TorchRec's EOF detection for persistent DataLoaders,
    whose iter(loader) may return the very same iterator object each window.
    """
    limits = []
    if window_limit:
        limits.append(max(0, window_limit - window_batch))
    if eval_interval > 0:
        next_eval = (global_step // eval_interval + 1) * eval_interval
        first_eval = (
            (skip_eval_until_step + eval_interval - 1) // eval_interval * eval_interval
        )
        limits.append(max(next_eval, first_eval) - global_step)
    if die_at_step >= 0:
        limits.append(max(1, die_at_step - global_step))
    return min(limits) if limits else None
