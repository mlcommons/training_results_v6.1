''                                

                                                                      
                                                                        
                                                                       
                                                                       
                                                                          
                  

                                                                     
                                                                           
                             
   

import os
from typing import Callable, List

import triton


def pinned_or_full(
    pinned: List[triton.Config],
    full_configs_fn: Callable[[], List[triton.Config]],
) -> List[triton.Config]:
    if os.environ.get("TRITON_FULL_AUTOTUNE", "0") == "1":
        return full_configs_fn()
    return pinned
