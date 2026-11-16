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

#!/usr/bin/env python3

                        

import functools
import os

import torch


class _PlainFuncWrapper:
    ''                                                                       
                                                                      
                                                                      
                                                                   

    def __init__(self, func):
        self._func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def register_fake(self, fake_func):
        return fake_func

    def register_kernel(self, device):
        def inner(func):
            return func

        return inner


def maybe_register_custom_op(op_name, mutates_args):
    '' 
                                                            

                                                                       
                                                                       
                                                                      
       

    def decorator(func):
        if os.environ.get("AOTI_LOWER"):
            return _PlainFuncWrapper(func)
        return torch.library.custom_op(op_name, func, mutates_args=mutates_args)

    return decorator


def is_sm100_plus() -> bool:
    '' 
                                                                           
                   

                                                                  
                                                                            
                                                                        
       
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    if props.major != 10:
        return False
    if props.minor <= 3:
        return True
    return props.minor == 7 and os.environ.get("HSTU_RUBIN_USE_SM100_PATHS", "1") == "1"


def is_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 9 and props.minor == 0


def is_sm90_plus() -> bool:
    return is_sm100_plus() or is_sm90()


@functools.lru_cache(maxsize=None)
def is_amd_mi350() -> bool:
    ''                                                                    

                                                                                
                                                                             
                                                 
       
    if not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None) is None:
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName or ""
    except (AssertionError, RuntimeError, AttributeError):
        return False
    return "gfx950" in arch


def use_separated_rng_ln_mul_dropout() -> bool:
    ''                                                                              
                                                                            
                                             

                                                                                 
                                                                                   
                                                                       
       
    return is_sm100_plus() or is_amd_mi350()


def copy_if_different_ptr(dst: torch.Tensor, src: torch.Tensor) -> None:
    if torch.compiler.is_compiling():
                                    
        dst.copy_(src)
    else:
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src)
