#!/bin/bash

# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

                                                                           
                                                                              
       
 
                                                                           
 
                                                                            
                                                          
                                                                      
                                                                              
                                                                    
 
                                                                   
 
                                            
                                          
 
                                                                            
                                  
                                                                      
                                                                         
                                                       
                                                                           

                                                   
: "${RANK:?RANK not set}"
: "${LOCAL_RANK:?LOCAL_RANK not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${LOCAL_WORLD_SIZE:?LOCAL_WORLD_SIZE not set}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"
: "${NSYS_METRICS:=0}"

readonly NODE_RANK=$(( RANK / LOCAL_WORLD_SIZE ))

: "${LOGGER:=""}"
if [[ -n "${APILOG_DIR:-}" ]]; then
    if [[ "$RANK" -eq 0 ]]; then
      LOGGER="apiLog.sh -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
    fi
fi

NSYS_OUT="/results/${NSYS_PREFIX:="lora"}_${SLURM_JOBID}_n${NODE_RANK}_p${LOCAL_RANK}"
NSYSCMD=""
profile_this_rank=0
if [ "${NVTX_FLAG:-0}" -gt 0 ]; then
    if [ -z "${PROFILE_RANKS:-}" ]; then
        profile_this_rank=1
    else
        case ",${PROFILE_RANKS}," in
            *",${RANK},"*) profile_this_rank=1 ;;
        esac
    fi
fi
if [ "${profile_this_rank}" -eq 1 ]; then
    NSYSCMD="nsys profile --sample=process-tree --cpuctxsw=process-tree --cudabacktrace=all --python-backtrace=cuda --trace=cuda-sw,nvtx --cuda-graph-trace=node --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --output ${NSYS_OUT}"
    if [ "${NSYS_PYTHON_SAMPLING:-0}" -gt 0 ]; then
        NSYSCMD="${NSYSCMD} --python-sampling=true --python-sampling-frequency=${NSYS_PYTHON_SAMPLING_HZ:-200} --stats=true"
    fi
    if [ ${NSYS_METRICS:-0} -gt 0 ]; then
        NSYSCMD="${NSYSCMD} --gpu-metrics-devices=${LOCAL_RANK} --gpu-metrics-set=${NSYS_METRICS_SET} --gpu-metrics-frequency=50000"
    fi
fi

declare -a CMD
if [[ ${LOCAL_WORLD_SIZE} -gt 1 ]]; then
                                                                     
    CMD=( ${NSYSCMD} 'python' '-u')
else
                                                     
    CMD=( ${NSYSCMD} 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi
${LOGGER:-} ${BINDCMD:-} ${CMD[@]} train.py; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi
