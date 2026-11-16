''                              

                                                                        
                                                                        
                                                                        

                                                                           
                                                            
                                                                          
                                                        
   

import logging
import os
from typing import Optional

import gin

logger: logging.Logger = logging.getLogger(__name__)


@gin.configurable
def apply_env_bootstrap(
    TRITON_FULL_AUTOTUNE: Optional[bool] = None,
) -> None:
                                                                          
                                                                            
                                                                        
                                                                           
                                                                         
    if "TRITON_FULL_AUTOTUNE" in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set TRITON_FULL_AUTOTUNE=%s (overrides gin binding)",
            os.environ["TRITON_FULL_AUTOTUNE"],
        )
    elif TRITON_FULL_AUTOTUNE is not None:
        os.environ["TRITON_FULL_AUTOTUNE"] = "1" if TRITON_FULL_AUTOTUNE else "0"
        logger.info("env bootstrap: TRITON_FULL_AUTOTUNE=%s", os.environ["TRITON_FULL_AUTOTUNE"])
