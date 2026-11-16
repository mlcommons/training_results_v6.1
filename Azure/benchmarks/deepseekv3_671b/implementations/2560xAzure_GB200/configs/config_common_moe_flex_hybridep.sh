if [ -z "${EXPERT_PARALLEL:-}" ]; then
    echo "EXPERT_PARALLEL must be set before sourcing config_common_moe_flex_hybridep.sh" >&2
    exit 1
fi

export MOE_TOKEN_DISPATCHER_TYPE=flex
export MOE_FLEX_DISPATCHER_BACKEND=hybridep
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
export USE_MNNVL=1

# Override the default preprocessing allocation of 108 SMs.
export MOE_HYBRIDEP_NUM_SMS_PREPROCESSING=32
