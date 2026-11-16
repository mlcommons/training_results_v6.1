''                                                                             

                                                                            
                                                                             
                                                                           
                                                                             
                                                                             
                                                      

                                                                           
                                                                  
   

import os
from contextlib import contextmanager

from megatron.core.utils import get_model_config


def bf16_eval_enabled() -> bool:
    return os.environ.get("BF16_EVAL", "0") == "1"


@contextmanager
def bf16_forward_ctx(model):
    ''                                                                      
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
    ''                                                               

    def __init__(self):
        self._ctx = None

    def on_eval_start(self, context):
        self._ctx = bf16_forward_ctx(context.model)
        self._ctx.__enter__()

    def on_eval_end(self, context):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
