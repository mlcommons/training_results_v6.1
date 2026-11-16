#!/bin/bash
# ==============================================================================
# Pathways Shared Configuration for MLPerf Recipes
# ==============================================================================

export PATHWAYS_SERVER_IMAGE="${PATHWAYS_SERVER_IMAGE:?set PATHWAYS_SERVER_IMAGE to the Pathways server image}"
export PATHWAYS_PROXY_IMAGE="${PATHWAYS_PROXY_IMAGE:?set PATHWAYS_PROXY_IMAGE to the Pathways proxy image}"
export PATHWAYS_PROXY_MEMORY_LIMIT="${PATHWAYS_PROXY_MEMORY_LIMIT:-160G}"
export USER_CONTAINER_MEMORY="${USER_CONTAINER_MEMORY:-260G}"
export USER_CONTAINER_MEMORY_LIMIT="${USER_CONTAINER_MEMORY_LIMIT:-260G}"
