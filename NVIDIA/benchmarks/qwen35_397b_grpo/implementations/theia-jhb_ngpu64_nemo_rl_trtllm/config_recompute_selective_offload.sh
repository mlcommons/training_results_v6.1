# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

export RECOMPUTE_MODULES=${RECOMPUTE_MODULES:-"[layernorm,attention,gdn,moe_act]"}
source "$(dirname "${BASH_SOURCE[0]}")/config_recompute_selective.sh"
export FINE_GRAINED_ACTIVATION_OFFLOADING=${FINE_GRAINED_ACTIVATION_OFFLOADING:-true}
export OFFLOAD_MODULES=${OFFLOAD_MODULES:-"[attn_norm,mlp_norm]"}
