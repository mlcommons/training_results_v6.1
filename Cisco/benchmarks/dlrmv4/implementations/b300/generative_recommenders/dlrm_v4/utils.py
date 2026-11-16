# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

             
'' 
                                           
   

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gin
import tensorboard                                                                                                      
import torch
from generative_recommenders.dlrm_v4.datasets.dataset import DLRMv4RandomDataset
from generative_recommenders.dlrm_v4.datasets.kuairand import DLRMv4KuaiRandDataset
from generative_recommenders.dlrm_v4.datasets.movie_lens import DLRMv4MovieLensDataset
from generative_recommenders.dlrm_v4.datasets.synthetic_movie_lens import (
    DLRMv4SyntheticMovieLensDataset,
)
from generative_recommenders.dlrm_v4.datasets.synthetic_streaming import (
    DLRMv4SyntheticStreamingDataset,
)
from generative_recommenders.dlrm_v4.datasets.yambda import DLRMv4YambdaDataset
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torch.profiler import profile, profiler, ProfilerActivity                    
from torch.utils.tensorboard import SummaryWriter
from torchrec.metrics.accuracy import AccuracyMetricComputation
from torchrec.metrics.auc import AUCMetricComputation, compute_auc
from torchrec.metrics.gauc import GAUCMetricComputation
from torchrec.metrics.mae import MAEMetricComputation
from torchrec.metrics.metrics_namespace import MetricName, MetricPrefix
from torchrec.metrics.mse import MSEMetricComputation
from torchrec.metrics.ne import NEMetricComputation
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetricComputation,
)


class LifetimeAUCMetricComputation(AUCMetricComputation):
    ''                                                                         
                    

                                                                                
                                                                               
                                                           
                                                                               
                                                                                
                         

                                                                             
                                                                            
                                                                          
                                                                               
                                                                                 
                                                                              
                                                                            
                                                                               
                                                                              
                                                      
       

                                                                             
                                                                             
    _LIFETIME_KEY_PREFIX: str = "_lifetime_"

    def _compute(self) -> List[MetricComputationReport]:
        from typing import cast as _cast
        from torchrec.metrics.auc import LABELS, PREDICTIONS, WEIGHTS
        return [
            MetricComputationReport(
                name=MetricName.AUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_auc(
                    self._n_tasks,
                    _cast(List[torch.Tensor], getattr(self, PREDICTIONS)),
                    _cast(List[torch.Tensor], getattr(self, LABELS)),
                    _cast(List[torch.Tensor], getattr(self, WEIGHTS)),
                    self._apply_bin,
                ),
            )
        ]

    def lifetime_sample_count(self) -> int:
        ''                                                                   
        return int(getattr(self, "_num_samples", 0))

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        from torchrec.metrics.auc import LABELS, PREDICTIONS, WEIGHTS

        destination = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
                                                                                 
                                                                              
                                                               
        for attr in (PREDICTIONS, LABELS, WEIGHTS):
            buf = getattr(self, attr)
            if isinstance(buf, (list, tuple)) and len(buf) > 0:
                flat = torch.cat([t for t in buf], dim=-1)
            elif isinstance(buf, torch.Tensor):
                flat = buf
            else:
                flat = torch.empty(0)
            destination[prefix + self._LIFETIME_KEY_PREFIX + attr] = (
                flat.detach().cpu().clone()
            )
        destination[prefix + self._LIFETIME_KEY_PREFIX + "num_samples"] = (
            torch.tensor(int(getattr(self, "_num_samples", 0)), dtype=torch.long)
        )
        return destination

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> Any:
        from torchrec.metrics.auc import LABELS, PREDICTIONS, WEIGHTS

                                                                              
                                                                        
        remaining = dict(state_dict)
        saved_bufs: Dict[str, torch.Tensor] = {}
        for attr in (PREDICTIONS, LABELS, WEIGHTS):
            key = self._LIFETIME_KEY_PREFIX + attr
            if key in remaining:
                saved_bufs[attr] = remaining.pop(key)
        num_key = self._LIFETIME_KEY_PREFIX + "num_samples"
        saved_num = remaining.pop(num_key, None)

        result = super().load_state_dict(remaining, strict=strict)

        if saved_bufs:
                                                                               
                                                                         
            existing = getattr(self, PREDICTIONS)
            dev = (
                existing[0].device
                if isinstance(existing, (list, tuple)) and len(existing) > 0
                else torch.device("cpu")
            )
            for attr, val in saved_bufs.items():
                setattr(self, attr, [val.to(dev)])
            if saved_num is not None:
                self._num_samples = int(saved_num.item())
        return result


                                                                               
                                                                               
                                                                            
                                                                             
UNBOUNDED_WINDOW: int = 1 << 60


class BinnedCumulativeAUC(RecMetricComputation):
    ''                                                                         

                                                                               
                                                                                 
                                                                                      
                                                                                   
                                                                                 
                                                                               
                                                                                   
                                                                                
                                                                           

                                                                                
                                                                                  
                                        
       

    def __init__(self, *args, num_bins: int = 100_000, **kwargs) -> None:
                                                                           
        super().__init__(*args, **kwargs)
        self._num_bins: int = int(num_bins)
        self._add_state(
            "pos_hist",
            torch.zeros((self._n_tasks, self._num_bins), dtype=torch.float64),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "neg_hist",
            torch.zeros((self._n_tasks, self._num_bins), dtype=torch.float64),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=True,
        )

    def cumulative_sample_count(self) -> int:
        ''                                                                         
        return int((self.pos_hist.sum() + self.neg_hist.sum()).item())

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        if predictions is None or weights is None:
            raise ValueError(
                "BinnedCumulativeAUC.update requires predictions and weights"
            )
        preds = predictions.float().clamp_(0.0, 1.0)                         
        labels = labels.float()
        weights = weights.float()
                                                                               
        idx = (preds * self._num_bins).long().clamp_(0, self._num_bins - 1)
        pos_w = (weights * labels).to(self.pos_hist.dtype)
        neg_w = (weights * (1.0 - labels)).to(self.neg_hist.dtype)
        self.pos_hist.scatter_add_(1, idx, pos_w)
        self.neg_hist.scatter_add_(1, idx, neg_w)

    def _compute(self) -> List[MetricComputationReport]:
                                                                                
                                                               
        pos = self.pos_hist                       
        neg = self.neg_hist
        total_pos = pos.sum(dim=1)
        total_neg = neg.sum(dim=1)
                                                                            
                                                                              
                                                                       
        neg_below = torch.cumsum(neg, dim=1) - neg
        numerator = (pos * (neg_below + 0.5 * neg)).sum(dim=1)
        denom = total_pos * total_neg
        auc = torch.where(
            denom > 0,
            numerator / denom,
            torch.full_like(numerator, 0.5),
        ).to(torch.float32)
        return [
            MetricComputationReport(
                name=MetricName.AUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=auc,
            )
        ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utils")


def _trim_warmup_from_trace(path: str, keep_n_active: int) -> None:
    ''                                                                    

                                                                           
                                                                        
                                                                          

                                                          
                                                                         
                                          
                                                                           
                                                                  

                              
       
    import json as _json
    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

                                                                          
                                                                             
                                                                      
                                                                          
                                                                         
                                           
    name_to_span: Dict[str, tuple] = {}
    for e in events:
        nm = e.get("name", "")
        if "ProfilerStep" not in nm or e.get("ph") != "X" or "ts" not in e:
            continue
        ts = e["ts"]
        end = ts + e.get("dur", 0)
        prev = name_to_span.get(nm)
        if prev is None:
            name_to_span[nm] = (ts, end)
        else:
            name_to_span[nm] = (min(prev[0], ts), max(prev[1], end))
    if len(name_to_span) <= keep_n_active:
        return
    sorted_spans = sorted(name_to_span.values())
    active = sorted_spans[-keep_n_active:]
    t_start = min(s for s, _ in active)
    t_end = max(e for _, e in active)

    def _keep(e: dict) -> bool:
        if e.get("ph") == "M":
            return True
        ts = e.get("ts")
        if ts is None:
            return True
        return t_start <= ts < t_end

    kept = [e for e in events if _keep(e)]
    d["traceEvents"] = kept
    with open(path, "w") as f:
        _json.dump(d, f)
    logger.warning(
        f"Trimmed WARMUP events from {path}: {len(events):,} -> {len(kept):,} "
        f"(kept active range [{t_start:.0f}, {t_end:.0f}] us)"
    )


                                                                             
_GPU_KERNEL_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})


def _is_rocm() -> bool:
    ''                                                                         

                                                                              
                                                                                
                                                                                 
                                                         
       
    return getattr(torch.version, "hip", None) is not None


def _normalize_profilerstep_layout(path: str) -> None:
    ''                                                                          

                                                                                 
                                                                              
                                                                               
                                                                          
                                                                             
                                                                              
                                                                            
                                                                 

                                                                                
                                                                                 
                                                                               
                                                                                 
                                                                             
                                                             
       
    import json as _json

    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

                                                                           
    stream_busy: Dict[tuple, int] = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in _GPU_KERNEL_CATS:
            dur = e.get("dur", 0)
            if dur > 0:
                key = (e.get("pid"), e.get("tid"))
                stream_busy[key] = stream_busy.get(key, 0) + dur
    if not stream_busy:
        return
    busiest = max(stream_busy, key=lambda k: stream_busy[k])

                                                                       
    gpu_ps_streams = set()
    template = None
    for e in events:
        if e.get("cat") == "gpu_user_annotation" and str(
            e.get("name", "")
        ).startswith("ProfilerStep"):
            gpu_ps_streams.add((e.get("pid"), e.get("tid")))
            if template is None:
                template = e
                                                                            
    if len(gpu_ps_streams) <= 1:
        return

                                                               
    cpu_win: Dict[str, list] = {}
    for e in events:
        if (
            e.get("cat") == "user_annotation"
            and e.get("ph") == "X"
            and str(e.get("name", "")).startswith("ProfilerStep")
        ):
            ts = e.get("ts", 0)
            end = ts + e.get("dur", 0)
            w = cpu_win.get(e["name"])
            if w is None:
                cpu_win[e["name"]] = [ts, end]
            else:
                w[0] = min(w[0], ts)
                w[1] = max(w[1], end)

                                                                    
    gpu_kernels = [
        (e.get("ts", 0), e.get("ts", 0) + e.get("dur", 0))
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") in _GPU_KERNEL_CATS
        and e.get("dur", 0) > 0
    ]

    new_spans = []
    for sname, (cs, ce) in cpu_win.items():
        ks = [(ts, end) for ts, end in gpu_kernels if end > cs and ts < ce]
        if not ks:
            continue
        gmin = min(ts for ts, _ in ks)
        gmax = max(end for _, end in ks)
        span = dict(template) if template else {}
        span.update(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": sname,
                "pid": busiest[0],
                "tid": busiest[1],
                "ts": gmin,
                "dur": gmax - gmin,
                "args": {"normalized_profilerstep": True},
            }
        )
        new_spans.append(span)

    if not new_spans:
        return

    out = [
        e
        for e in events
        if not (
            e.get("cat") == "gpu_user_annotation"
            and str(e.get("name", "")).startswith("ProfilerStep")
        )
    ]
    dropped = len(events) - len(out)
    out.extend(new_spans)
    d["traceEvents"] = out
    with open(path, "w") as f:
        _json.dump(d, f)
    logger.warning(
        f"Normalized GPU ProfilerStep layout in {path}: dropped {dropped} "
        f"fragmented span(s) across {len(gpu_ps_streams)} stream(s), wrote "
        f"{len(new_spans)} span(s) on busiest stream pid={busiest[0]} "
        f"tid={busiest[1]}"
    )


def _deoverlap_gpu_slices(path: str, max_snap_us: float = 5.0) -> None:
    ''                                                                       

                                                                                
                                                                                
                                                                             
                                                                          
                                                                               
                                                                      
                                                                               
                                                                             
                                                                              
                                                                                
         

                                                                            
                                                                               
                                                                                 
                                                                           
                                                                           
                                                         

                                                                             
                                                                         
                                                                            
                                                                               
                                                                                
                                                                              
       
    import json as _json
    from collections import defaultdict

                                                                          
                                                                           
                                                   
    _GAP_US = 0.001

    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

    tracks: Dict[tuple, list] = defaultdict(list)
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("cat") in _GPU_KERNEL_CATS
            and e.get("dur", 0) > 0
        ):
            tracks[(e.get("pid"), e.get("tid"))].append(e)

    snapped = 0
    max_clip = 0.0
    for sl in tracks.values():
                                                                              
                                                                                
                                     
        sl.sort(key=lambda e: (e["ts"], -e["dur"]))
        for i in range(len(sl) - 1):
            a = sl[i]
            b = sl[i + 1]
            a_end = a["ts"] + a["dur"]
            b_end = b["ts"] + b["dur"]
                                                                                
                                                                                 
                                
            if b["ts"] <= a_end < b_end:
                desired_end = b["ts"] - _GAP_US
                clip = a_end - desired_end
                if a["ts"] < desired_end and 0 < clip < max_snap_us:
                    a["dur"] = desired_end - a["ts"]
                    snapped += 1
                    if clip > max_clip:
                        max_clip = clip

    if snapped:
        with open(path, "w") as f:
            _json.dump(d, f)
        logger.warning(
            f"De-overlapped GPU slices in {path}: snapped {snapped} sub-us "
            f"overlap(s) (max {max_clip:.3f}us) so Perfetto renders every kernel"
        )


def _deoverlap_gpu_annotations(path: str, max_snap_us: float = 5.0) -> None:
    ''                                                                          
                                                        

                                                                           
                                                                       
                                                                            
                                                                           
                                                                                 
                                                                                 
                                                                                
                                                                                
                                                                             
                                                                                
                                                                                  
                                                                   

                                                                            
                                                                                  
                                                                                
                                                                             
                                                                                 
                                                                          
                                                                                 
                                                                           
                                                                                   
                                           
       
    import json as _json
    from collections import defaultdict

                                                                          
                                                                              
                                                                            
                                                                               
    _GAP_US = 0.002

    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

                                                                               
                                                                             
                                                  
    _ANN = "gpu_user_annotation"
    tracks: Dict[tuple, list] = defaultdict(list)
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("dur", 0) > 0
            and (e.get("cat") in _GPU_KERNEL_CATS or e.get("cat") == _ANN)
        ):
            tracks[(e.get("pid"), e.get("tid"))].append(e)

    snapped = 0
    max_clip = 0.0
    for sl in tracks.values():
                                                                            
        sl.sort(key=lambda e: (e["ts"], -e["dur"]))
                                                                               
                                                              
        stack: list = []
        for b in sl:
            b_ts = b["ts"]
            b_end = b_ts + b["dur"]
            while stack:
                a = stack[-1][0]
                a_end = a["ts"] + a["dur"]
                if a_end < b_ts:
                                                                                 
                    frame = stack.pop()
                    eff = frame[0]["ts"] + frame[0]["dur"]
                    if stack:
                        stack[-1][1] = max(stack[-1][1], eff, frame[1])
                    continue
                if a_end < b_end:
                                                                               
                                                                                
                                                                                  
                                                                      
                                                                           
                                                                              
                                                                               
                                                                               
                                                                    
                    desired_end = b_ts - _GAP_US
                    clip = a_end - desired_end
                                                                             
                                                                          
                                                                               
                                                                              
                                                                              
                                                                                 
                                                          
                    if (
                        a["ts"] < desired_end
                        and stack[-1][1] <= b_ts
                        and 0 < clip < max_snap_us
                    ):
                        a["dur"] = desired_end - a["ts"]
                        snapped += 1
                        if clip > max_clip:
                            max_clip = clip
                    frame = stack.pop()
                    eff = frame[0]["ts"] + frame[0]["dur"]
                    if stack:
                        stack[-1][1] = max(stack[-1][1], eff, frame[1])
                    continue
                                                                           
                break
            stack.append([b, b_ts])

    if snapped:
        with open(path, "w") as f:
            _json.dump(d, f)
        logger.warning(
            f"De-overlapped GPU annotations in {path}: snapped {snapped} sub-us "
            f"sibling overlap(s) (max {max_clip:.3f}us) so Perfetto renders every "
            f"annotation full width"
        )


def _on_trace_ready_fn(
    rank: Optional[int] = None,
    trace_dir: str = "/tmp/dlrm_v4_traces",
    keep_n_active: Optional[int] = None,
    trace_steps: Optional[List[int]] = None,
) -> Callable[[torch.profiler.profile], None]:
    ''                                                                        

                                                                       
                                                                          
                   

                         

                                                                          
                                                                    
                                                                     
                          
                                                                       
                                                                      
           

                                                                            
                                                                          
       
    state = {"fire_count": 0}

    def handle_fn(p: torch.profiler.profile) -> None:
        os.makedirs(trace_dir, exist_ok=True)
        if trace_steps:
            i = state["fire_count"]
            step_label = (
                trace_steps[i] if i < len(trace_steps) else getattr(p, "step_num", 0)
            )
        else:
            step_label = getattr(p, "step_num", 0)
        state["fire_count"] += 1
        rank_str = f"_rank{rank}" if rank is not None else ""
        file_name = f"trace_step{step_label}{rank_str}.json"
        path = os.path.join(trace_dir, file_name)
        logger.warning(
            p.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total"
            )
        )
                                                                               
                                                                           
                                                                              
                                          
        try:
            p.export_chrome_trace(path)
            logger.warning(f"Trace written to: {path}")
            if keep_n_active is not None and keep_n_active > 0:
                _trim_warmup_from_trace(path, keep_n_active)
                                                                                
                                                                          
                                                                               
                                                          
            if _is_rocm():
                                                                           
                                                                            
                _normalize_profilerstep_layout(path)
                                                                             
                                                 
                _deoverlap_gpu_slices(path)
                                                                            
                                                                           
                _deoverlap_gpu_annotations(path)
        except Exception as exc:
            logger.warning(f"Trace export/trim failed for {path}: {exc!r} (skipping)")

    return handle_fn


def profiler_or_nullcontext(
    enabled: bool, with_stack: bool, trace_dir: str = "/tmp/dlrm_v4_traces"
):
    ''                                                                 
    return (
        profile(
                                                                                    
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=_on_trace_ready_fn(trace_dir=trace_dir),
            with_stack=with_stack,
        )
        if enabled
        else contextlib.nullcontext()
    )


def _multi_window_schedule(
    trace_steps,
    warmup: int,
    active: int,
):
    ''                                                                   

                
                                        
                                        
                                                 
       
    windows = [(s - warmup, s, s + active) for s in sorted(trace_steps)]

    def schedule_fn(step: int) -> torch.profiler.ProfilerAction:
        for warmup_start, active_start, active_end in windows:
            if warmup_start <= step < active_start:
                return torch.profiler.ProfilerAction.WARMUP
            if active_start <= step < active_end - 1:
                return torch.profiler.ProfilerAction.RECORD
            if step == active_end - 1:
                return torch.profiler.ProfilerAction.RECORD_AND_SAVE
        return torch.profiler.ProfilerAction.NONE

    return schedule_fn


@gin.configurable
class Profiler:
    ''                                                                    

                            

                                                                           
                                                                   
                                                                              
                                                         

                                                   

                                                           
                                                
                           
                            
       

    def __init__(
        self,
        rank: int,
        active: int = 50,
        wait: int = 10,
        warmup: int = 20,
        repeat: int = 1,
        trace_steps: Optional[List[int]] = None,
        trace_dir: str = "/tmp/dlrm_v4_traces",
        trim_warmup: bool = True,
        record_shapes: bool = True,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
    ) -> None:
        self.rank = rank
        self.trace_dir = trace_dir
        if trace_steps:
            sched = _multi_window_schedule(trace_steps, warmup, active)
        else:
            sched = torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            )
        keep_n = active if trim_warmup else None
        self._profiler: profiler.profile = torch.profiler.profile(
            schedule=sched,
            on_trace_ready=_on_trace_ready_fn(
                self.rank, trace_dir, keep_n, trace_steps
            ),
                                                                                    
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
        )

    def step(self) -> None:
        ''                                          
        self._profiler.step()


class _NoOpSummaryWriter:
    ''                                                                     
                                                                            
                                                                               
                                                                                
                                                                             
                                                                               
                             

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


@gin.configurable
class MetricsLogger:
    '' 
                                                             

                                                                            
                                                        

         
                                                                             
                                                      
                                                                
                                                  
                                                    
                                                                    
       

    def __init__(
        self,
        multitask_configs: List[TaskConfig],
        batch_size: int,
        window_size: int,
        device: torch.device,
        rank: int,
        tensorboard_log_path: str = "",
        world_size: int = 1,
        auc_threshold: Optional[float] = None,
        num_flops_per_sample: float = 0.0,
        gpu_peak_flops: float = 0.0,
        model: Optional[torch.nn.Module] = None,
        eval_cumulative: bool = False,
        cumulative_auc_bins: int = 100_000,
        train_lifetime_auc_mode: str = "binned",
        eval_lifetime_auc_mode: str = "binned",
        lifetime_auc_window: int = 10_000_000,
        mlperf_step_time_every: int = 10,
        enable_train_metrics: bool = False,
    ) -> None:
                                                                               
                                                                                
                                                                           
                                                                                      
        self._num_flops_per_sample: float = max(0.0, float(num_flops_per_sample))
        self._gpu_peak_flops: float = max(0.0, float(gpu_peak_flops))
        self._model_ref: Optional[torch.nn.Module] = model
        if rank == 0 and self._num_flops_per_sample > 0 and self._gpu_peak_flops > 0:
            logger.info(
                f"FLOPS reporting enabled: {self._num_flops_per_sample / 1e9:.1f} "
                f"GFLOP/sample (dense fwd+bwd), GPU peak {self._gpu_peak_flops / 1e12:.0f} TFLOPS"
            )
        self.multitask_configs: List[TaskConfig] = multitask_configs
        all_classification_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type != MultitaskTaskType.REGRESSION
        ]
        all_regression_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type == MultitaskTaskType.REGRESSION
        ]
        assert all_classification_tasks + all_regression_tasks == [
            task.task_name for task in multitask_configs
        ]
        self.task_names: List[str] = all_classification_tasks + all_regression_tasks

                                
                                                                                  
                                                                            
                                               
                                                                                 
                                                                              
                                                                           
                                                                               
                                                                               
         
                                                                                
                                                                                 
                                                                            
                                                               
                                                                                  
                                                                               
                                              
        self._eval_cumulative: bool = eval_cumulative
        self._cumulative_auc_bins: int = int(cumulative_auc_bins)
        self._train_lifetime_auc_mode: str = str(train_lifetime_auc_mode)
        self._eval_lifetime_auc_mode: str = str(eval_lifetime_auc_mode)
        self._lifetime_auc_window: int = int(lifetime_auc_window)
        self._enable_train_metrics: bool = bool(enable_train_metrics)
        n_cls = len(all_classification_tasks)
        n_reg = len(all_regression_tasks)

        def _make_lifetime_auc(mode: str) -> RecMetricComputation:
            if mode == "binned":
                                                                             
                return BinnedCumulativeAUC(
                    my_rank=rank, batch_size=batch_size, n_tasks=n_cls,
                    window_size=0, num_bins=self._cumulative_auc_bins,
                ).to(device)
            if mode == "capped":
                return LifetimeAUCMetricComputation(
                    my_rank=rank, batch_size=batch_size, n_tasks=n_cls,
                    window_size=self._lifetime_auc_window,
                ).to(device)
            raise ValueError(
                f"lifetime_auc_mode must be 'binned' or 'capped', got {mode!r}"
            )

        def _make_class(ws: int, lifetime_mode: Optional[str]) -> List[RecMetricComputation]:
            mets: List[RecMetricComputation] = [
                NEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
                AccuracyMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
                GAUCMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
                AUCMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
            ]
            if lifetime_mode is not None:
                mets.append(_make_lifetime_auc(lifetime_mode))
            return mets

        def _make_class_cumulative() -> List[RecMetricComputation]:
                                                                               
                                                                        
            return [
                NEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=window_size).to(device),
                AccuracyMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=window_size).to(device),
                GAUCMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=window_size).to(device),
                _make_lifetime_auc(self._eval_lifetime_auc_mode),
            ]

        def _make_reg(ws: int) -> List[RecMetricComputation]:
            return [
                MSEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_reg, window_size=ws).to(device),
                MAEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_reg, window_size=ws).to(device),
            ]

        self.class_metrics: Dict[str, List[RecMetricComputation]] = {"train": [], "eval": []}
        self.regression_metrics: Dict[str, List[RecMetricComputation]] = {"train": [], "eval": []}
        if eval_cumulative:
            self.class_metrics["eval_cum"] = []
            self.regression_metrics["eval_cum"] = []

        if all_classification_tasks:
            if self._enable_train_metrics:
                self.class_metrics["train"] = _make_class(
                    window_size, lifetime_mode=self._train_lifetime_auc_mode
                )
            if eval_cumulative:
                self.class_metrics["eval"] = _make_class(UNBOUNDED_WINDOW, lifetime_mode=None)
                self.class_metrics["eval_cum"] = _make_class_cumulative()
            else:
                self.class_metrics["eval"] = _make_class(window_size, lifetime_mode=self._eval_lifetime_auc_mode)

        if all_regression_tasks:
            if self._enable_train_metrics:
                self.regression_metrics["train"] = _make_reg(window_size)
            if eval_cumulative:
                self.regression_metrics["eval"] = _make_reg(UNBOUNDED_WINDOW)
                self.regression_metrics["eval_cum"] = _make_reg(window_size)
            else:
                self.regression_metrics["eval"] = _make_reg(window_size)

        self.global_step: Dict[str, int] = {"train": 0, "eval": 0}
                                                                                 
                                                                     
        self.cumulative_train_samples: int = 0
                                                                                  
                                                                              
                                                                                
                                                                               
                                                                        
        self.mlperf_run_started: bool = False
        self._rank: int = int(rank)
                                                                                 
                                                                                   
        self.mlperf_logger: Optional[Any] = None
        self.lr_getter: Optional[Callable[[], float]] = None
        self.tb_logger: Optional[SummaryWriter] = None
        if tensorboard_log_path != "":
            self.tb_logger = SummaryWriter(log_dir=tensorboard_log_path, purge_step=0)
            self.tb_logger.flush()
        else:
                                                                                 
                                                                                  
                                                                        
            self.tb_logger = _NoOpSummaryWriter()

                                                                             
                                                                    
        self._world_size: int = max(1, int(world_size))
        self._auc_threshold: Optional[float] = auc_threshold
        self._time_to_target_logged: bool = False
        self._perf_t_start: float = time.perf_counter()
        self._perf_t_window: float = self._perf_t_start
        self._perf_steps_in_window: int = 0
        self._perf_total_samples: int = 0
        self._perf_samples_counter: torch.Tensor = torch.zeros(
            1, dtype=torch.long, device=device
        )
                                                                           
                                                                               
                                                                           
                                                                               
                                                               
                                                                                   
        self._perf_excluded: Dict[str, float] = {"eval": 0.0, "ckpt": 0.0}
        self._perf_pause: Dict[str, Optional[float]] = {}
                                                                           
                                                                        
        self._mlperf_step_time_every: int = int(mlperf_step_time_every)
        self._mlperf_step_time_step: int = 0

    def pause_perf(self, category: str) -> None:
        ''                                                                     
                                                                         
                                                                    
        if self._perf_pause.get(category) is None:
            self._perf_pause[category] = time.perf_counter()

    def resume_perf(self, category: str) -> None:
        ''                                                                 
                                                                   
        t0 = self._perf_pause.get(category)
        if t0 is not None:
            self._perf_excluded[category] = (
                self._perf_excluded.get(category, 0.0)
                + (time.perf_counter() - t0)
            )
            self._perf_pause[category] = None

    @property
    def auc_threshold(self) -> Optional[float]:
        ''                                                            
        return self._auc_threshold

    @property
    def all_metrics(self) -> Dict[str, List[RecMetricComputation]]:
        '' 
                                                 

                
                                                                                    
           
        out = {
            "train": self.class_metrics["train"] + self.regression_metrics["train"],
            "eval": self.class_metrics["eval"] + self.regression_metrics["eval"],
        }
        if "eval_cum" in self.class_metrics or "eval_cum" in self.regression_metrics:
            out["eval_cum"] = self.class_metrics.get(
                "eval_cum", []
            ) + self.regression_metrics.get("eval_cum", [])
        return out

    def update(
        self,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        labels: torch.Tensor,
        num_candidates: torch.Tensor,
        mode: str = "train",
    ) -> None:
        '' 
                                                                

             
                                                 
                                          
                                              
                                                                       
                                           
           
                                                                               
                                           
        update_targets: List[RecMetricComputation] = []
        if mode != "train" or self._enable_train_metrics:
            update_targets = list(self.all_metrics[mode])
            if mode == "eval" and "eval_cum" in self.all_metrics:
                update_targets = update_targets + self.all_metrics["eval_cum"]
        for metric in update_targets:
            if isinstance(metric, GAUCMetricComputation):
                metric.update(
                    predictions=predictions,
                    labels=labels,
                    weights=weights,
                    num_candidates=num_candidates,
                )
            else:
                metric.update(
                    predictions=predictions,
                    labels=labels,
                    weights=weights,
                )
        self.global_step[mode] += 1
        if mode == "train":
                                                                             
                                                             
            self._perf_samples_counter += num_candidates.sum().to(
                self._perf_samples_counter.dtype
            )
            self._perf_steps_in_window += 1
                                                                                 
                                                                              
            self.cumulative_train_samples += int(num_candidates.numel()) * self._world_size

    def compute(self, mode: str = "train") -> Dict[str, float]:
        '' 
                                                              

             
                                           

                
                                                                     
           
        all_computed_metrics = {}

        if mode == "eval" and "eval_cum" in self.all_metrics:
                                                                                 
                                                                               
                                                                          
                                                                        
                                                                            
            def _emit(
                metrics: List[RecMetricComputation], keep_prefix: str
            ) -> None:
                for metric in metrics:
                    for computed in metric.compute():
                        pfx = str(computed.metric_prefix)
                        name = str(computed.name)
                        if pfx != keep_prefix or name.endswith("num_samples"):
                            continue
                        all_values = computed.value.cpu()
                        for i, task_name in enumerate(self.task_names):
                            if i >= len(all_values):
                                break
                            all_computed_metrics[f"metric/{pfx}{name}/{task_name}"] = (
                                all_values[i]
                            )

            _emit(self.all_metrics["eval"], "window_")
            _emit(self.all_metrics["eval_cum"], "lifetime_")
        else:
            for metric in self.all_metrics[mode]:
                computed_metrics = metric.compute()
                for computed in computed_metrics:
                    all_values = computed.value.cpu()
                    for i, task_name in enumerate(self.task_names):
                        if i >= len(all_values):
                            break
                        key = f"metric/{str(computed.metric_prefix) + str(computed.name)}/{task_name}"
                        all_computed_metrics[key] = all_values[i]

        logger.info(
            f"{mode} - Step {self.global_step[mode]} metrics: {all_computed_metrics}"
        )
        return all_computed_metrics

    def _global_mean_loss(self, loss_terms: Dict[str, torch.Tensor]) -> float:
        ''                                               

                                                                                 
                                                                                
                            
           
        loss_t = torch.stack(
            [v.detach().float().sum() for v in loss_terms.values()]
        ).sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.SUM)
            loss_t = loss_t / self._world_size
        return float(loss_t)

    def maybe_log_mlperf_train_loss(
        self, aux_losses: Dict[str, torch.Tensor], every: int
    ) -> None:
        ''                                                         

                                                                                
                                                                           
                                                                               
                                                                               
                                                                              
                                                                           
                                                                    
           
        if self.mlperf_logger is None or every <= 0 or not aux_losses:
            return
        if self.global_step["train"] % every != 0:
            return
        train_loss = self._global_mean_loss(aux_losses)
        c = self.mlperf_logger.constants
        md: Dict[str, Any] = {c.SAMPLES_COUNT: self.cumulative_train_samples}
        if self.lr_getter is not None:
            try:
                md["lr"] = float(self.lr_getter())
            except Exception:
                pass
        self.mlperf_logger.event(
            key=getattr(c, "TRAIN_LOSS", "train_loss"),
            value=train_loss,
            metadata=md,
        )

    def compute_and_log(
        self,
        mode: str = "train",
        additional_logs: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        '' 
                                               

             
                                           
                                                             

                
                                                                     

               
                                                                    
           
        assert self.tb_logger is not None
                                                                              
                                                                           
                                                                     
        all_computed_metrics = (
            self.compute(mode)
            if mode != "train" or self._enable_train_metrics
            else {}
        )
        for k, v in all_computed_metrics.items():
            self.tb_logger.add_scalar(                    
                f"{mode}_{k}",
                v,
                global_step=self.global_step[mode],
            )

        if additional_logs is not None:
            for tag, data in additional_logs.items():
                for data_name, data_value in data.items():
                    self.tb_logger.add_scalar(
                        f"{tag}/{mode}_{data_name}",
                        data_value.detach().clone().cpu(),
                        global_step=self.global_step[mode],
                    )

                                                                                 
                                                                                 
                                                                                
                                                                                   
                                                       
        if mode == "train" and additional_logs is not None and "losses" in additional_logs:
            loss_terms = additional_logs["losses"]
            if loss_terms:
                train_loss = self._global_mean_loss(loss_terms)
                self.tb_logger.add_scalar(
                    "train_loss", train_loss, global_step=self.global_step["train"]
                )
                if self._rank == 0:
                    logger.info(
                        f"train - Step {self.global_step['train']} "
                        f"train_loss={train_loss:.5f}"
                    )

                                                                      
        if mode == "train" and self._perf_steps_in_window > 0:
            now = time.perf_counter()
            wall_dt = max(now - self._perf_t_window, 1e-6)
                                                                             
                                                                            
                                                                             
                                                                                  
            eval_s = self._perf_excluded.get("eval", 0.0)
            ckpt_s = self._perf_excluded.get("ckpt", 0.0)
            dt = max(wall_dt - eval_s - ckpt_s, 1e-6)
            n_samples = int(self._perf_samples_counter.item())
            self._perf_total_samples += n_samples
            local_sps = n_samples / dt
            global_sps = local_sps * self._world_size
            step_ms = dt * 1000.0 / self._perf_steps_in_window
            wall_step_ms = wall_dt * 1000.0 / self._perf_steps_in_window
            eval_ms = eval_s * 1000.0
            ckpt_ms = ckpt_s * 1000.0
            elapsed = now - self._perf_t_start
            step = self.global_step["train"]
            self.tb_logger.add_scalar(
                "perf/train_samples_per_sec_local", local_sps, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_samples_per_sec_global", global_sps, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_step_time_ms", step_ms, global_step=step
            )
                                                                            
                                                                              
                                                         
            if (
                self.mlperf_logger is not None
                and self._mlperf_step_time_every > 0
                and step - self._mlperf_step_time_step >= self._mlperf_step_time_every
            ):
                self._mlperf_step_time_step = step
                self.mlperf_logger.event(
                    key="tracked_stats",
                    value={"train_step_time": step_ms / 1000.0},
                    metadata={
                        self.mlperf_logger.constants.SAMPLES_COUNT: self.cumulative_train_samples,
                        "step": step,
                    },
                )
                                                                            
                                                             
            self.tb_logger.add_scalar(
                "perf/train_wall_step_time_ms", wall_step_ms, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/window_eval_time_ms", eval_ms, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/window_ckpt_time_ms", ckpt_ms, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_total_samples", self._perf_total_samples, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_elapsed_sec", elapsed, global_step=step
            )
                                                                             
                                                                              
                                                                         
                                                           
                                                                        
                                                                       
                                        
                                                                               
                                                                         
                                                                              
                                                                 
            tflops_str = ""
            if self._num_flops_per_sample > 0 and self._gpu_peak_flops > 0:
                local_flops = self._num_flops_per_sample * local_sps
                tflops_algo = local_flops / 1e12
                mfu = 100.0 * local_flops / self._gpu_peak_flops
                self.tb_logger.add_scalar("perf/train_tflops_algo_gpu", tflops_algo, global_step=step)
                self.tb_logger.add_scalar("perf/train_mfu_pct", mfu, global_step=step)
                tflops_str = f" tflops_algo/gpu={tflops_algo:.1f} mfu={mfu:.1f}%"
                jagged_t = None
                executed_t = None
                m = self._model_ref
                if m is not None:
                    inner = m.module if hasattr(m, "module") else m
                    jagged_t = getattr(inner, "_last_jagged_flops_per_sample", None)
                    executed_t = getattr(
                        inner, "_last_executed_flops_per_sample", None
                    )
                if jagged_t is not None:
                    if executed_t is None:
                        executed_t = jagged_t
                                                      
                    jagged, executed = torch.stack([jagged_t, executed_t]).tolist()
                    if 0 < executed <= jagged < self._num_flops_per_sample:
                        tflops_real = executed * local_sps / 1e12
                        hfu = 100.0 * executed * local_sps / self._gpu_peak_flops
                        fill = 100.0 * jagged / self._num_flops_per_sample
                        self.tb_logger.add_scalar("perf/train_tflops_real_gpu", tflops_real, global_step=step)
                        self.tb_logger.add_scalar("perf/train_hfu_pct", hfu, global_step=step)
                        self.tb_logger.add_scalar("perf/train_fill_pct", fill, global_step=step)
                        tflops_str += f" tflops_real/gpu={tflops_real:.1f} hfu={hfu:.1f}% fill={fill:.1f}%"
            logger.info(
                f"train - Step {step} perf: local_sps={local_sps:.1f} "
                f"global_sps={global_sps:.1f} step_ms={step_ms:.2f} "
                f"elapsed_sec={elapsed:.1f} total_samples={self._perf_total_samples} "
                f"wall_step_ms={wall_step_ms:.2f} eval_ms={eval_ms:.1f} "
                f"ckpt_ms={ckpt_ms:.1f}"
                + tflops_str
            )
            self._perf_t_window = now
            self._perf_steps_in_window = 0
            self._perf_samples_counter.zero_()
                                                                           
                                                                                
                                                                              
                                                           
            self._perf_excluded = {"eval": 0.0, "ckpt": 0.0}
            self._perf_pause = {}

                                                                                 
                                                                                      
        if (
            self._auc_threshold is not None
            and not self._time_to_target_logged
        ):
            for key, val in all_computed_metrics.items():
                metric_short = key.split("/")[-2] if "/" in key else key
                if metric_short.endswith("auc") and not metric_short.endswith("gauc"):
                    if float(val) >= self._auc_threshold:
                        ttt = time.perf_counter() - self._perf_t_start
                        self.tb_logger.add_scalar(
                            f"perf/time_to_auc_{self._auc_threshold:.5f}_sec",
                            ttt,
                            global_step=self.global_step[mode],
                        )
                        logger.info(
                            f"REACHED AUC>={self._auc_threshold} on {key}="
                            f"{float(val):.6f} at elapsed_sec={ttt:.2f} "
                            f"step={self.global_step[mode]}"
                        )
                        self._time_to_target_logged = True
                        break

        return all_computed_metrics

    def reset(self, mode: str = "train"):
        '' 
                                           

             
                                           
           
        for metric in self.all_metrics[mode]:
            metric.reset()


                         
SUPPORTED_DATASETS = [
    "debug",
    "movielens-1m",
    "movielens-20m",
    "movielens-13b",
    "movielens-18b",
    "kuairand-1k",
    "streaming-400m",
    "streaming-200b",
    "streaming-100b",
    "sampled-streaming-100b",
    "yambda-5b",
]


@gin.configurable
def env_path(key: str = "", default: str = "") -> str:
    ''                                                                

                                                                           
                                            

                               
                                       
                                               
                                                                
       
    return os.environ.get(key, default) if key else default


@gin.configurable
def env_str(key: str = "", default: str = "") -> str:
    ''                                                                  

                                                                               
                                                

                                                               
                                                        
                                       
       
    raw = os.environ.get(key) if key else None
    return raw if raw else default


@gin.configurable
def env_str_map(
    key: str = "",
    default: Optional[Dict[str, str]] = None,
    merge: bool = False,
) -> Dict[str, str]:
    ''                                                              

                                                                               
                                                                            
                      

                                                                               
                                                       
                                

                                                                                

                                                                       
                                                                             
                                                                 
                                                                             
                                                                             
                                                                  
       
    base = dict(default or {})
    raw = os.environ.get(key) if key else None
    if not raw:
        return base
    parsed: Dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, _, v = pair.partition("=")
        parsed[k.strip()] = v.strip()
    return {**base, **parsed} if merge else parsed


@gin.configurable
def env_int(key: str = "", default: int = 0) -> int:
    ''                                                                

                                                                     

                                                        
                                  
                             
       
    raw = os.environ.get(key) if key else None
    return int(raw) if raw else default


@gin.configurable
def env_float(key: str = "", default: float = 0.0) -> float:
    ''                                                                 

                                                                    
                                                            

                                                                           
                                                  
                                  
       
    raw = os.environ.get(key) if key else None
    return float(raw) if raw else default


_GPU_PEAK_FLOPS_TABLE: Dict[str, Dict[str, float]] = {
                                                                               
                                                                               
                                                 
    "MI355X": {"bf16": 2300e12, "fp32": 575e12},
    "MI350X": {"bf16": 2300e12, "fp32": 575e12},
    "MI300X": {"bf16": 1300e12, "fp32": 653e12},
    "MI325X": {"bf16": 1300e12, "fp32": 653e12},
    "B200":   {"bf16": 2250e12, "fp32": 1125e12},
    "H100":   {"bf16": 990e12,  "fp32": 67e12},
    "A100":   {"bf16": 312e12,  "fp32": 19.5e12},
}


def get_gpu_peak_flops(dtype: str = "bf16") -> float:
    ''                                                   

                                                                             
                                                                            
       
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(0)
    for gpu_key, peaks in _GPU_PEAK_FLOPS_TABLE.items():
        if gpu_key in name:
            return peaks.get(dtype, peaks["bf16"])
    logger.warning(
        f"Unknown GPU for peak FLOPS: {name}; defaulting to MI350X bf16 (2300 TF)"
    )
    return _GPU_PEAK_FLOPS_TABLE["MI350X"]["bf16"]


@gin.configurable
def run_results_dir(run_name: str = "default", subdir: str = "results") -> str:
    ''                                                                            

                                                                              
                                                                      

                       

                              
                                     
                                    
                                            
                                               
       
                                                                                  
                                             
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / subdir / run_name)


@gin.configurable
def get_dataset(
    name: str,
    new_path_prefix: str = "",
    history_length: Optional[int] = None,
    min_history: Optional[int] = None,
    history_strategy: str = "interleaved",
    streaming_window_seconds: int = 86400,
    streaming_sort_within_window: bool = False,
    streaming_shuffle_fraction: float = 0.0,
    streaming_shuffle_seed: int = 0,
    train_split_percentage: float = 1.0,
    split_salt: int = 0,
    cache_dir: Optional[str] = None,
    use_offline_ts_indices: bool = False,
):
    '' 
                                                

         
                                                                 
                                                                  

            
                                                                        

           
                                                         
       
    assert name in SUPPORTED_DATASETS, f"dataset {name} not supported"
    if name == "debug":
        return DLRMv4RandomDataset, {}
    if name == "movielens-1m":
        return (
            DLRMv4MovieLensDataset,
            {
                "ratings_file": os.path.join(
                    new_path_prefix, "data/ml-1m/sasrec_format.csv"
                ),
            },
        )
    if name == "movielens-20m":
        return (
            DLRMv4MovieLensDataset,
            {
                "ratings_file": os.path.join(
                    new_path_prefix, "data/ml-20m/sasrec_format.csv"
                ),
            },
        )
    if name == "movielens-13b":
        return (
            DLRMv4SyntheticMovieLensDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/ml-13b/16x16384"
                ),
            },
        )
    if name == "movielens-18b":
        return (
            DLRMv4SyntheticMovieLensDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/ml-18b/20x36864"
                ),
            },
        )
    if name == "kuairand-1k":
        return (
            DLRMv4KuaiRandDataset,
            {
                "seq_logs_file": os.path.join(
                    new_path_prefix, "data/KuaiRand-1K/data/processed_seqs.csv"
                ),
            },
        )
    if name == "streaming-400m":
        return (
            DLRMv4SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-400m/"
                ),
                "train_ts": 8,
                "total_ts": 10,
                "num_files": 3,
                "num_users": 150_000,
                "num_items": 1_500_000,
                "num_categories": 128,
            },
        )
    if name == "streaming-200b":
        return (
            DLRMv4SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-200b/"
                ),
                "train_ts": 90,
                "total_ts": 100,
                "num_files": 100,
                "num_users": 10_000_000,
                "num_items": 1_000_000_000,
                "num_categories": 128,
            },
        )
    if name == "streaming-100b":
        return (
            DLRMv4SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-100b/"
                ),
                "train_ts": 90,
                "total_ts": 100,
                "num_files": 100,
                "num_users": 5_000_000,
                "num_items": 1_000_000_000,
                "num_categories": 128,
            },
        )
    if name == "yambda-5b":
        from generative_recommenders.dlrm_v4.configs import YAMBDA_5B_CROSS_SPECS

        return (
            DLRMv4YambdaDataset,
            {
                                                                                     
                                                                                            
                                                                            
                                                                                
                                                                            
                                                                              
                                                                      
                                                                    
                "processed_dir": os.path.join(new_path_prefix, "processed_5b"),
                "metadata_dir": os.path.join(new_path_prefix, "shared_metadata"),
                "cache_dir": cache_dir
                or os.path.join(new_path_prefix, "processed_5b", "hstu_cache"),
                                                                             
                                                                       
                "history_length": history_length if history_length is not None else 4096,
                "scan_window": 20000,
                                                                             
                                                                      
                                                                            
                                                                           
                                                                            
                "min_history": min_history,
                                                                        
                                                                          
                                                                              
                                                                            
                                                                            
                                                          
                "history_strategy": history_strategy,
                "cross_specs": YAMBDA_5B_CROSS_SPECS,
                                                                  
                                                                                
                "streaming_window_seconds": streaming_window_seconds,
                "streaming_sort_within_window": streaming_sort_within_window,
                "use_offline_ts_indices": use_offline_ts_indices,
                                                                                    
                                                                                 
                                              
                "streaming_shuffle_fraction": streaming_shuffle_fraction,
                "streaming_shuffle_seed": streaming_shuffle_seed,
                                                                             
                                                                             
                                                                
                "train_split_percentage": train_split_percentage,
                "split_salt": split_salt,
            },
        )
    if name == "sampled-streaming-100b":
        return (
            DLRMv4SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-100b/sampled_data/"
                ),
                "train_ts": 90,
                "total_ts": 100,
                "num_files": 1,
                "num_users": 50_000,
                "num_items": 1_000_000_000,
                "num_categories": 128,
            },
        )
