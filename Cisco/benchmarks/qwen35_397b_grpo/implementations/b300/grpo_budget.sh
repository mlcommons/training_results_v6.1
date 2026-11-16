#!/bin/bash

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

                                                                               
                                                                              
                                                                        
                                                                             

                                                                              
                                                                      
                                                                            
                                                            
grpo_node_sweep_timeout_for() {
    echo "${GRPO_NODE_SWEEP_TIMEOUT:-$(( 90 + ${1:-1} ))}"
}

grpo_validate_node_sweep_timeout() {
    if [[ -n "${GRPO_NODE_SWEEP_TIMEOUT:-}" && ! "${GRPO_NODE_SWEEP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: GRPO_NODE_SWEEP_TIMEOUT must be a positive integer (seconds)" >&2
        return 1
    fi
}

                                                                                 
                                                                                
                                                                                 
                                                                           
grpo_teardown_reserve_seconds_for() {
    echo $(( 15 + 60 + ${2} * ($(grpo_node_sweep_timeout_for "${1}") + 2) + 195 ))
}

                                                                       
grpo_teardown_reserve_minutes_for() {
    echo $(( ($(grpo_teardown_reserve_seconds_for "${1}" "${2}") + 59) / 60 ))
}
