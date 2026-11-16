# SPDX-License-Identifier: Apache-2.0
"""Custom fused ops for the Flux.1 flat model.

Import this package to register all torch.library custom ops under the ``flux``
namespace (e.g. ``torch.ops.flux.fused_gate_residual_a2``).

Every import below is unconditional. There is no env flag that swaps a module for
a sibling and falls back on ``ImportError``: such a gate silently swallows a real
import failure and reports success while the old path runs. If one of these
imports fails, the process fails.

Registration belongs to the package, not to one recipe's module. ``flat/model.py``
calls ``flux.qkv_split_norm_rope_inplace_ctm``, ``flux.fused_sum_cat{3,6}_ctm`` and
``flux.rope_norm_join_ctm`` unconditionally, so registering them from
``flat/nvfp4_ops.py`` -- which ``fp8_ops.py`` imports only for the NVFP4 recipe --
would leave them undefined under MXFP8-only (``FP4=0``) and BF16 (``FLAT_BF16=1``).

These are all CUTLASS-DSL modules and cost nothing at import beyond the
``torch.library`` defines.
"""

# The layernorm + adaLN modulation kernels. Owns both the ``flux.fused_ln_adaln_a1``
# names and the plain ``flux.fused_ln_adaln*`` names that flat/model.py calls.
from flat.custom_ops import fused_ln_adaln_a1 as _fused_ln_adaln_a1  # noqa: F401

# The row-block gate+residual kernels, registered as flux.fused_gate_residual_a2 /
# _add_a2 (called from flat/model.py) and flux.fused_gate_bwd (the recipe seam
# fp8_gate_bwd_quantize).
from flat.custom_ops import gate_residual_a2 as _gate_residual_a2  # noqa: F401

# The sequence split / join (default ``rowps``: one row per CTA with raw pointers
# and a constant tile map), registered as flux.seq_split_ctm3 / seq_join_ctm3.
from flat.custom_ops import seq_split_join_ctm3 as _seq_split_join_ctm3  # noqa: F401

from flat.custom_ops import (  # noqa: F401
    fused_qkv_split_norm_rope_ctm2 as _qkv_split_norm_rope_ctm2,
)
from flat.custom_ops import fused_sum_cat_ctm2 as _fused_sum_cat_ctm2  # noqa: F401

# The aten-tail kernels: bias_colsum, add, fused_sum_cat2, silu / dsilu /
# dsilu_accum, qknorm_grad_fin, pad_join, timestep_embed, mse_loss_fwd_bwd, zero_f32.
from flat.custom_ops import aten_tail_ctm as _aten_tail_ctm  # noqa: F401

from flat.custom_ops import rope_norm_join_ctm as _rope_norm_join_ctm  # noqa: F401
