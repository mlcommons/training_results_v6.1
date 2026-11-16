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
    """Construct a TE NVFP4 quantizer with UE5M3 scales.

    Tensor scaling is disabled for activation tensors and enabled for
    weight and grad tensors.

    This factory function is compatible with the custom quantization
    recipe.

    Dispatch mirrors ``te.pytorch.custom_recipes.nvfp4_factory``. The
    deliberate deviations from it are:

    * UE5M3 scale factors instead of E4M3 (hardware-gated to Rubin/SM107).
    * ``disable_second_level_scale`` for activations, which drop the
      per-tensor scale entirely.

    Stochastic rounding on gradients and the random sign mask with RHT are
    kept as in the stock recipe. Without SR the 64-node DSv3 run needed
    10+ more steps than NVFP4 to reach the target; with it both hit 3.6 at
    step 51.

    Note that, unlike the TE factories, this keys only on ``tensor_type``
    and ignores ``role.module_type``. An absent or empty ``tensor_type``
    is treated as ``"input"``, which also turns second-level scaling off.
    That is safe for the linear/grouped_linear modules this recipe is
    applied to, but would need revisiting before reusing it elsewhere.
    """
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
