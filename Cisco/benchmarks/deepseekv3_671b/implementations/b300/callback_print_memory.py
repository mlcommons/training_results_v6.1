''                                                                

from __future__ import annotations

import os
from typing import Any

import torch
from megatron.bridge.training.callbacks import Callback, CallbackContext


_GIB = 1024**3

                                                                            
                                                                            
                                                                            
_HYBRIDEP_PROBE: dict[str, Any] = {
    "installed": False,
    "calls": 0,
    "samples": 0,
    "recv_sum": None,
    "recv_max": None,
    "expert_max": None,
}


def _gib(value: int) -> float:
    return value / _GIB


def _parallel_rank(name: str) -> int | str:
    try:
        from megatron.core import parallel_state

        getter = getattr(parallel_state, f"get_{name}_rank")
        return getter()
    except (AttributeError, RuntimeError):
        return "unknown"


def _diagnostics_enabled() -> bool:
    return os.environ.get("SPIKE_DIAGNOSTICS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _install_hybridep_probe() -> None:
    ''                                                                      
    if not _diagnostics_enabled() or _HYBRIDEP_PROBE["installed"]:
        return

    from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager

    original_dispatch = _HybridEPManager.dispatch
    interval = max(int(os.environ.get("SPIKE_DIAGNOSTICS_SAMPLE_INTERVAL", "128")), 1)

    def dispatch_with_probe(self, *args, **kwargs):
        output = original_dispatch(self, *args, **kwargs)
        _HYBRIDEP_PROBE["calls"] += 1
        if _HYBRIDEP_PROBE["calls"] % interval == 0:
            tokens = self.tokens_per_expert
            total = tokens.sum(dtype=torch.int64)
            largest = tokens.max()
            if _HYBRIDEP_PROBE["recv_sum"] is None:
                _HYBRIDEP_PROBE["recv_sum"] = torch.zeros_like(total)
                _HYBRIDEP_PROBE["recv_max"] = torch.zeros_like(total)
                _HYBRIDEP_PROBE["expert_max"] = torch.zeros_like(largest)
            _HYBRIDEP_PROBE["recv_sum"].add_(total)
            torch.maximum(_HYBRIDEP_PROBE["recv_max"], total, out=_HYBRIDEP_PROBE["recv_max"])
            torch.maximum(
                _HYBRIDEP_PROBE["expert_max"], largest, out=_HYBRIDEP_PROBE["expert_max"]
            )
            _HYBRIDEP_PROBE["samples"] += 1
        return output

    _HybridEPManager.dispatch = dispatch_with_probe
    _HYBRIDEP_PROBE["installed"] = True


def _reset_hybridep_probe() -> None:
    _HYBRIDEP_PROBE["samples"] = 0
    for name in ("recv_sum", "recv_max", "expert_max"):
        value = _HYBRIDEP_PROBE[name]
        if value is not None:
            value.zero_()


def _format_hybridep_probe(step: int) -> str | None:
    samples = _HYBRIDEP_PROBE["samples"]
    if not _diagnostics_enabled() or not samples or _HYBRIDEP_PROBE["recv_sum"] is None:
        return None
                                                                                
                                                                                 
    recv_sum = int(_HYBRIDEP_PROBE["recv_sum"].item())
    recv_max = int(_HYBRIDEP_PROBE["recv_max"].item())
    expert_max = int(_HYBRIDEP_PROBE["expert_max"].item())
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "unknown"))
    interval = max(int(os.environ.get("SPIKE_DIAGNOSTICS_SAMPLE_INTERVAL", "128")), 1)
    return (
        f"SPIKE_MOE step={step} rank={rank} samples={samples} sample_interval={interval}"
        f" recv_tokens_mean={recv_sum / samples:.3f} recv_tokens_max={recv_max}"
        f" local_expert_tokens_max={expert_max}"
    )


class PrintMemoryUsageCallback(Callback):
    ''                                                                  

                                                                             
                                                                         
                                                                              
                                     
       

    def __init__(self) -> None:
        self._last_step = 0
        self._previous_allocator_stats: dict[str, int] = {}
        _install_hybridep_probe()

    def _print(self, phase: str, step: int | None = None) -> None:
        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        allocator_stats = torch.cuda.memory_stats(device)
        counter_names = (
            "num_alloc_retries",
            "num_ooms",
            "num_sync_all_streams",
            "num_device_alloc",
            "num_device_free",
        )
        fields = {
            "phase": phase,
            "step": "none" if step is None else step,
            "rank": os.environ.get("RANK", os.environ.get("SLURM_PROCID", "unknown")),
            "local_rank": os.environ.get("LOCAL_RANK", "unknown"),
            "pp_rank": _parallel_rank("pipeline_model_parallel"),
            "cp_rank": _parallel_rank("context_parallel"),
            "dp_rank": _parallel_rank("data_parallel"),
            "allocated_gib": f"{_gib(torch.cuda.memory_allocated(device)):.3f}",
            "reserved_gib": f"{_gib(torch.cuda.memory_reserved(device)):.3f}",
            "peak_allocated_gib": f"{_gib(torch.cuda.max_memory_allocated(device)):.3f}",
            "peak_reserved_gib": f"{_gib(torch.cuda.max_memory_reserved(device)):.3f}",
            "device_free_gib": f"{_gib(free):.3f}",
            "device_total_gib": f"{_gib(total):.3f}",
        }
        for name in counter_names:
            value = int(allocator_stats.get(name, 0))
            fields[name] = value
            fields[f"delta_{name}"] = value - self._previous_allocator_stats.get(name, value)
        self._previous_allocator_stats = {
            name: int(allocator_stats.get(name, 0)) for name in counter_names
        }
        print(
            "PRINT_MEMORY_USAGE "
            + " ".join(f"{key}={value}" for key, value in fields.items()),
            flush=True,
        )

    def on_train_start(self, context: CallbackContext) -> None:
        self._print("after_checkpoint_load", step=context.state.train_state.step)
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        self._print("train_peak_reset", step=context.state.train_state.step)

    def on_train_step_start(self, context: CallbackContext) -> None:
        self._last_step = context.state.train_state.step + 1
        _reset_hybridep_probe()
        self._print("train_step_start", step=self._last_step)

    def on_train_step_end(self, context: CallbackContext) -> None:
        self._last_step = context.state.train_state.step + 1
        self._print("train_step_end", step=self._last_step)
        probe_line = _format_hybridep_probe(self._last_step)
        if probe_line is not None:
            print(probe_line, flush=True)

    def on_train_end(self, context: CallbackContext) -> None:
        self._print("train_end", step=context.state.train_state.step)

    def cleanup_after_oom(self) -> None:
        self._print("exception", step=self._last_step or None)
