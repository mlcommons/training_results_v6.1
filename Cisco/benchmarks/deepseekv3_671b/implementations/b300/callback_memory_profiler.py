# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


import os
import traceback
from typing import List, Optional

import torch
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.utils.common_utils import get_rank_safe, get_world_size_safe


def _format_bytes(b: int) -> str:
    ''                                                               
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GiB"
    if b >= 1024**2:
        return f"{b / 1024**2:.2f} MiB"
    return f"{b} B"


def log_device_memory_stats(label: str, all_ranks: bool = False) -> None:
    ''                                                                      

                                                                                 

                         
                                                                                   
                                                                                  
                                                                                   
                                                                               
                                                                                   
                                                                                  
                                                                     

               
                                                                                
                                                                           
                                                                                  
                                                                                  
                                                                 

                                                                               
                                                                            

                                                                          

                                                                                   
                                                                                     
       
    if not torch.cuda.is_available():
        return
    try:
        stats = torch.cuda.memory_stats()
        retries = stats.get("num_alloc_retries", 0)
        ooms = stats.get("num_ooms", 0)
        rank = get_rank_safe()
                                                                          
                                                                          
        print_memory_usage = os.environ.get("PRINT_MEMORY_USAGE", "0").strip().lower()
        verbose = (
            os.environ.get("MEMORY_STATS_LOG", "0") == "1"
            or print_memory_usage in ("1", "true", "yes", "on")
        )
        if not (retries or ooms) and not (verbose and (all_ranks or rank == 0)):
            return
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        print(
            f"[MEMSTATS] rank={rank} at={label}"
            f" alloc={_format_bytes(torch.cuda.memory_allocated())}"
            f" reserved={_format_bytes(reserved)}"
            f" peak_alloc={_format_bytes(torch.cuda.max_memory_allocated())}"
            f" peak_reserved={_format_bytes(torch.cuda.max_memory_reserved())}"
            f" nonpool={_format_bytes(max(total - free - reserved, 0))}"
            f" dev_free={_format_bytes(free)}"
            f" dev_total={_format_bytes(total)}"
            f" retries={retries}"
            f" ooms={ooms}"
            f" syncs={stats.get('num_sync_all_streams', 0)}"
            f" seg_alloc={stats.get('num_device_alloc', 0)}"
            f" seg_free={stats.get('num_device_free', 0)}",
            flush=True,
        )
    except Exception:
        pass


class MemoryProfileCallback(Callback):
    '' 
                                                                          
                                                                                     
                                                                              
                                                                     

                                                                                
                                                                   
                                                                          
                                                                            
                                                                         
                                                                                        
                                    

                                                      
                                                                                    
                                                                                      
                                                
                                                                                    
                                                                         
                                                                                  
                                                                                    
                                                                     
                                                                                 
                                                               
                                                                                               

                                                                                                  
                                                                
       

    def __init__(
        self,
        enabled: bool = True,
        file_prefix: str = "memdump",
        output_dir: str = "/mem_dump",
        max_entries: int = 1000000,
        rank_0_only: bool = True,
        profile_ranks: Optional[List[int]] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        force_oom_before_stop: bool = False,
    ):
        self.enabled = enabled
        self.file_prefix = file_prefix
        self.output_dir = output_dir
        self.max_entries = max_entries
        self.force_oom_before_stop = force_oom_before_stop
        self.start_step = start_step                               
        self.end_step = end_step                            

        global_rank = get_rank_safe()
        world_size = get_world_size_safe()
        if profile_ranks is not None:
            ranks = profile_ranks
        else:
            ranks = [0] if rank_0_only else list(range(world_size))
        self.do_profile = enabled and (global_rank in ranks)
        self._rank = global_rank

        self._recording = False

    @staticmethod
    def _format_bytes(b: int) -> str:
        ''                                                               
        return _format_bytes(b)

    def _log_step_memory(self, step: int, label: str) -> None:
        ''                                                                    
        try:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            print_str = f"Rank {self._rank} step {step} {label}: \
            Allocated: {self._format_bytes(allocated)} \
            Reserved: {self._format_bytes(reserved)} \
            Max Allocated: {self._format_bytes(max_allocated)} \
            Max Reserved: {self._format_bytes(max_reserved)}"
            print(print_str, flush=True)
        except Exception:
            pass

    def _start_recording(self) -> None:
        if not self.do_profile or self._recording:
            return
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        self._recording = True

    def _stop_and_dump(self, suffix: str = "") -> None:
        if not self.do_profile or not self._recording:
            return
        try:
            if self.force_oom_before_stop:
                self._force_oom()
            os.makedirs(self.output_dir, exist_ok=True)
            filename = os.path.join(self.output_dir, f"{self.file_prefix}_{self._rank}{suffix}.pickle")
                                                                                           
                                                                                         
            torch.cuda.synchronize()
            torch.cuda.memory._dump_snapshot(filename)
        finally:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
            self._recording = False

    def _force_oom(self) -> None:
        ''                                                                   
        try:
            torch.cuda.empty_cache()
            _ = torch.empty(2**32, device="cuda", dtype=torch.uint8)
        except Exception:
            pass

    def cleanup_after_oom(self) -> None:
        '' 
                                                                   
                                                                                       
           
        if not self.do_profile:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            memdump_filename = os.path.join(self.output_dir, f"{self.file_prefix}_failed_{self._rank}.pickle")
            traceback_filename = os.path.join(self.output_dir, f"{self.file_prefix}_failed_{self._rank}.traceback")
            with open(traceback_filename, "w") as f:
                traceback.print_exc(file=f)
            if self._recording:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.memory._dump_snapshot(memdump_filename)
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
            self._recording = False
        except Exception:
            pass

    def on_train_start(self, context: CallbackContext) -> None:
        if self.start_step is None:
            self._start_recording()

    def on_train_end(self, context: CallbackContext) -> None:
                                                                                           
                                                                                     
        if self.end_step is None or self._recording:
            self._stop_and_dump()

    def on_train_step_start(self, context: CallbackContext) -> None:
        step = context.state.train_state.step
        self._log_step_memory(step, "start")
        if self.start_step is not None and step == self.start_step:
            self._start_recording()

    def on_train_step_end(self, context: CallbackContext) -> None:
        step = context.state.train_state.step
        if self.end_step is not None and step == self.end_step:
            self._stop_and_dump()
        self._log_step_memory(step, "end")
