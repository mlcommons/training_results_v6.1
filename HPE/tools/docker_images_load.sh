#!/bin/bash
# Load Docker images saved in /nas/SHARED/MLPERF/containers (plain .tar).
# Run on a node with a Docker daemon (inside an allocation made with
# --comment=docker). Loads only images missing from this node.
#
# Usage: docker_images_load.sh [<name filter>]     e.g.  docker_images_load.sh llama2
set -uo pipefail
SRC=${SRC:-/nas/SHARED/MLPERF/containers}
for f in "${SRC}"/*.tar; do
    read -r tag cfg < <(tar -xOf "$f" manifest.json | python3 -c \
        'import json,sys; m=json.load(sys.stdin)[0]; print((m.get("RepoTags") or ["?"])[0], m["Config"].split("/")[-1].replace(".json",""))')
    [[ -n "${1:-}" && "${tag}" != *"$1"* ]] && continue
    if [[ "$(docker image inspect --format '{{.Id}}' "${tag}" 2>/dev/null)" == "sha256:${cfg}" ]]; then
        echo "present: ${tag}"; continue
    fi
    echo "loading: ${tag}  ($(du -h "$f" | cut -f1))"
    docker load -i "$f"
done
