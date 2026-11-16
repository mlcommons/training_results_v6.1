# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

import argparse

import torch
from megatron.bridge import AutoBridge

                                                   
                                                                                         
                                    
                                                  
              
           
                                            
                 
                
                     
 
                                                                               
                                 
                                                            
 
                                                                      
                                                                                 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="/ckpt",
        help="Path to save the imported checkpoint",
    )

    args = parser.parse_args()

    AutoBridge.import_ckpt(
        hf_model_id="meta-llama/Llama-2-70B-hf",
        megatron_path=args.save_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
