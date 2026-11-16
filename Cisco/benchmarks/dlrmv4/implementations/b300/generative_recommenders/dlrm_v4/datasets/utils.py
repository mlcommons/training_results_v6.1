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
                                         

                                                                     
                               
   

import json
import struct
from typing import Dict, List, Sequence, Tuple

import numpy as np
import xxhash


def json_loads(
    x: str | int | List[int],
) -> List[int]:
    '' 
                                                     

                                                                         
                               

         
                                                                   
                                         

            
                                               
       
    if isinstance(x, str):
        if x[0] != "[" and x[-1] != "]":
            x = "[" + x + "]"
        y = json.loads(x)
    else:
        y = x
    y_list = [y] if type(y) == int else list(y)
    return y_list


def separate_uih_candidates(
    x: str | int | List[int],
    candidates_max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    '' 
                                                                           

                                                                         
                                                       

         
                                                                      
                                                                                

            
                                                                    
       
    if isinstance(x, str):
        if x[0] != "[" and x[-1] != "]":
            x = "[" + x + "]"
        y = json.loads(x)
    else:
        y = x
    y_list = [y] if type(y) == int else list(y)
    candidates, uih = (
        y_list[-candidates_max_seq_len:],
        y_list[:-candidates_max_seq_len],
    )
    return uih, candidates


def maybe_truncate_seq(
    y: List[int],
    max_seq_len: int,
) -> List[int]:
    '' 
                                                         

         
                                                  
                                                     

            
                                                                  
       
    y_len = len(y)
    if y_len > max_seq_len:
        y = y[:max_seq_len]
    return y


def xxhash_cross(
    anchor: Dict[str, int],
    keys: Sequence[str],
    table_size: int,
    salt: int = 0,
) -> int:
    ''                                                                                              

                                                                              
                                                 
       
    n = len(keys)
    assert n >= 2, f"xxhash_cross needs >=2 keys, got {n}"
    digest = xxhash.xxh64(seed=salt)
    digest.update(struct.Struct(f"<{n}q").pack(*(int(anchor[k]) for k in keys)))
    return digest.intdigest() % table_size


def xxhash_cross_batch(
    arr_by_key: Dict[str, np.ndarray],
    keys: Sequence[str],
    table_size: int,
    salt: int = 0,
) -> np.ndarray:
    ''                                                                         
    n = len(keys)
    assert n >= 2
    cols = [np.asarray(arr_by_key[k], dtype=np.int64).ravel() for k in keys]
    length = cols[0].shape[0]
    for c in cols:
        assert c.shape[0] == length
    pack = struct.Struct(f"<{n}q").pack
    digest_cls = xxhash.xxh64
    out = np.empty(length, dtype=np.int64)
    for i in range(length):
        d = digest_cls(seed=salt)
        d.update(pack(*(int(c[i]) for c in cols)))
        out[i] = d.intdigest() % table_size
    return out
