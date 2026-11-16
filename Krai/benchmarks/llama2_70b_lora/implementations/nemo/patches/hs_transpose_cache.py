# Copyright (c) 2026 Krai Ltd. All Rights Reserved.
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

"""Selective FP8 weight-transpose cache for the frozen base weights (fix 6b, env HS_TCACHE).

The recipe runs with Megatron's keep_fp8_weight_transpose_cache=False, so every backward of every TE Linear /
LayerNormLinear transposes its frozen FP8 weight (rocm_utils.create_fp8_weight_transpose_cache -> tex.fp8_transpose)
before the dgrad GEMM and frees it after: hipBLASLt runs FP8 GEMMs only as TN on gfx942. Measured per layer (lane
l4_transpose): fc1 315, fc2 163, qkv 90, proj 69 us, ~51 ms/step at 80 layers. The weights never change (LoRA), so
the transpose is the same bytes every step. Caching all of them would cost another ~65 GB per GPU (TP 1); this module
turns TE's own cache on (module.keep_fp8_weight_transpose_cache = True, read by TE at every forward) for a chosen
subset: TE then builds the transpose once (Float8Tensor.update_usage(columnwise_usage=True) in the forward) and the
backward skips its create/clear pair. The dgrad GEMM reads the same bytes either way: bit-exact.

  HS_TCACHE=qkv,proj         kinds to cache (qkv, proj, fc1, fc2); "kind:N" caches only the first N modules of that
                             kind built (e.g. fc2:20). Default qkv,proj = 151 MB per layer, 12.1 GB at 80 layers,
                             the cheapest per byte (qkv 1.07, proj 1.03 us/MB vs fc1 0.67, fc2 0.69).

Only modules built under fp8_model_init (primary FP8 weights, the frozen base model) are touched. Kinds are told
apart by TE class and shape: LayerNormLinear with out > 4 x in is fc1, else qkv; Linear with
in > out is fc2, else proj (Llama MLP and attention projections).
"""

import os

import transformer_engine.pytorch as te

_SPEC = {}
for _item in filter(None, (s.strip() for s in os.getenv("HS_TCACHE", "qkv,proj").split(","))):
    _kind, _, _n = _item.partition(":")
    assert _kind in ("qkv", "proj", "fc1", "fc2"), f"HS_TCACHE: unknown kind {_kind!r}"
    _SPEC[_kind] = int(_n) if _n else None
STATS = {"cached": {}, "bytes": 0, "seen": {}}


def _kind(module):
    i, o = module.in_features, module.out_features
    if isinstance(module, te.LayerNormLinear):
        return "fc1" if o > 4 * i else "qkv"
    return "fc2" if i > o else "proj"


def _wrap_init(cls):
    orig = cls.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if not getattr(self, "primary_weights_in_fp8", False):
            return   # only FP8 primary (frozen base) weights: a BF16 weight's FP8 workspace is re-cast every step
        kind = _kind(self)
        n = STATS["seen"][kind] = STATS["seen"].get(kind, 0) + 1
        if kind in _SPEC and (_SPEC[kind] is None or n <= _SPEC[kind]):
            self.keep_fp8_weight_transpose_cache = True
            STATS["cached"][kind] = STATS["cached"].get(kind, 0) + 1
            STATS["bytes"] += self.in_features * self.out_features   # one FP8 byte per weight element
            if os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0")) == "0" and STATS["cached"][kind] % 20 == 0:
                print(report(), flush=True)

    __init__.__wrapped__ = orig
    cls.__init__ = __init__


for _cls in (te.Linear, te.LayerNormLinear):
    _wrap_init(_cls)
print(f"[hyperspindle] hs_transpose_cache: caching FP8 weight transposes for {_SPEC}", flush=True)


def report():
    return (f"[hyperspindle] hs_transpose_cache: cached {STATS['cached']} of {STATS['seen']} modules, "
            f"{STATS['bytes'] / 2**30:.2f} GiB of transposes")
