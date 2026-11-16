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

import transformer_engine as te


def make_nvfp4_ue5m3_quantizer(
    role: te.pytorch.QuantizerRole,
) -> te.pytorch.NVFP4Quantizer:
    ''                                                  

                                                                     
                            

                                                                    
           

                                                                     
                                      

                                                                          
                                                                    
                                

                                                                          
                                                                       
                                                                          
            

                                                                         
                                                                        
                                                                         
                                                                     
                                                                      
       
    tensor_type = role.tensor_type if role is not None else None
    if not tensor_type:
        tensor_type = "input"
    with_rht = tensor_type in ("input", "grad_output")
    return te.pytorch.NVFP4Quantizer(
        scale_dtype=te.pytorch.DType.kFloat8UE5M3,
        with_rht=with_rht,
        with_post_rht_amax=with_rht,
        with_2d_quantization=tensor_type == "weight",
        stochastic_rounding=tensor_type == "grad_output",
        with_random_sign_mask=with_rht,
        disable_second_level_scale=tensor_type == "input",
    )
