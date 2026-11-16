#!/bin/bash
# MLPerf Training v6.1 / Qwen3.5-397B-A17B GRPO (qwen35_397b_grpo) / tpu7x-4x4x8-128x2x2x1
# One-time setup: GKE cluster credentials, optional CPU node pools, and the
# container image (built from the code tree in ../tunix and pushed).
# Usage: PROJECT=<gcp-project> REGION=<region> CLUSTER=<tpu7x-gke-cluster> \
#   K8S_NAMESPACE=<namespace> TUNIX_IMAGE=<registry/image:tag> \
#   [EVAL_PROJECT=<gcp-project> EVAL_REGION=<region> EVAL_CLUSTER=<tpu7x-gke-cluster>] \
#   [CREATE_NODEPOOLS=1 ZONE=<zone>] [BUILD_IMAGE=0] bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT="${PROJECT:?set PROJECT to the GCP project of the TPU7x GKE cluster}"
export REGION="${REGION:?set REGION to the region of the TPU7x GKE cluster}"
export CLUSTER="${CLUSTER:?set CLUSTER to the name of the TPU7x GKE cluster}"
export K8S_NAMESPACE="${K8S_NAMESPACE:?set K8S_NAMESPACE to the Kubernetes namespace for training}"
export TUNIX_IMAGE="${TUNIX_IMAGE:?set TUNIX_IMAGE to the registry/image:tag to build and push}"

# Credentials for a separate Phase 2 evaluation cluster (run_and_time.sh EVAL_*),
# fetched first so that the training cluster stays the current kubectl context.
if [[ -n "${EVAL_CLUSTER:-}" ]]; then
  gcloud container clusters get-credentials "${EVAL_CLUSTER}" \
    --region="${EVAL_REGION:?set EVAL_REGION}" --project="${EVAL_PROJECT:?set EVAL_PROJECT}"
fi
gcloud container clusters get-credentials "${CLUSTER}" --region="${REGION}" --project="${PROJECT}"
kubectl config set-context --current --namespace="${K8S_NAMESPACE}"

# Optional one-time GKE node pool provisioning on ${CLUSTER} (${ZONE}), matching
# the pools of the scored runs (508 sandbox nodes: 198 + 310):
if [[ "${CREATE_NODEPOOLS:-0}" == "1" ]]; then
  export ZONE="${ZONE:?set ZONE to the node-pool zone}"
  # 1. OpenHands sandbox pools (c3d-standard-60-lssd and c3d-standard-90-lssd,
  #    local SSD ephemeral storage, Ubuntu node image), labeled and tainted
  #    workload=sandbox
  gcloud container node-pools create sandbox-np \
    --cluster="${CLUSTER}" --project="${PROJECT}" --region="${REGION}" \
    --node-locations="${ZONE}" --machine-type=c3d-standard-60-lssd \
    --num-nodes="${SANDBOX_NODES_60:-198}" --max-pods-per-node=128 --enable-image-streaming \
    --image-type=UBUNTU_CONTAINERD \
    --ephemeral-storage-local-ssd=count=4 --node-labels=workload=sandbox \
    --node-taints=workload=sandbox:NoSchedule
  gcloud container node-pools create sandbox-c3d-np-2 \
    --cluster="${CLUSTER}" --project="${PROJECT}" --region="${REGION}" \
    --node-locations="${ZONE}" --machine-type=c3d-standard-90-lssd \
    --num-nodes="${SANDBOX_NODES_90:-310}" --max-pods-per-node=192 --enable-image-streaming \
    --image-type=UBUNTU_CONTAINERD \
    --ephemeral-storage-local-ssd=count=8 --node-labels=workload=sandbox \
    --node-taints=workload=sandbox:NoSchedule

  # 2. agent-sandbox controller pool (1 x c3d-standard-60-lssd; C3D supports
  #    pd-ssd and pd-balanced boot disks in addition to Hyperdisk, whereas the
  #    160 tpu7x-standard-4t hosts in systems/*.json use 100 GB Hyperdisk Balanced)
  gcloud container node-pools create sandbox-controller \
    --cluster="${CLUSTER}" --project="${PROJECT}" --region="${REGION}" \
    --node-locations="${ZONE}" --machine-type=c3d-standard-60-lssd --num-nodes=1 \
    --ephemeral-storage-local-ssd=count=4 --disk-type=pd-ssd --disk-size=200 \
    --node-labels=workload=sandbox-controller \
    --node-taints=workload=sandbox-controller:NoSchedule

  # 3. RL controller and Pathways head pool (c3d-standard-60; run_and_time.sh
  #    CPU_NODEPOOL=cpu-np)
  gcloud container node-pools create cpu-np \
    --cluster="${CLUSTER}" --project="${PROJECT}" --region="${REGION}" \
    --node-locations="${ZONE}" --machine-type=c3d-standard-60 --num-nodes=40 \
    --node-labels=workload=cpu-user \
    --scopes=https://www.googleapis.com/auth/cloud-platform
fi

# Populate the Tunix source tree (../tunix/tunix) per ../tunix/SOURCE_VERSION.txt.
TUNIX_SRC="${TUNIX_SRC:-${SCRIPT_DIR}/../tunix/tunix}"
if ! grep -q 'is_partial=True' "${TUNIX_SRC}/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh" 2>/dev/null; then
  rm -rf "${TUNIX_SRC}"
  git clone https://github.com/google/tunix.git "${TUNIX_SRC}"
  git -C "${TUNIX_SRC}" checkout 592380f2963d81cb710082c05f00b538afdb2aa2
  sed -i "s|agent-sandbox.git#subdirectory=|agent-sandbox.git@7fd162d9d5c33fd0c8af9fbb15c6453ccf5ddd58#subdirectory=|g" "${TUNIX_SRC}/Dockerfile"
  cp -a "${SCRIPT_DIR}/recipes/." "${TUNIX_SRC}/tunix/experimental/examples/recipes/"
  sed -i \
    -e '/VLLM_TPU_USING_PATHWAYS=1/,/ENABLE_PATHWAYS_PERSISTENCE=/{ /BOOTSTRAP_CMD/d }' \
    -e '/--worker_container_port="\${eval_port}"/,/PYTHONUNBUFFERED=1/ s|--worker_startup_command=" \\|--worker_startup_command=" \\\n        sed -i '\''s/is_partial=False/is_partial=True/g'\'' /opt/venv/lib/python3.12/site-packages/maxtext/utils/model_creation_utils.py; \\\n        \${BOOTSTRAP_CMD} \\|' \
    "${TUNIX_SRC}/tunix/experimental/examples/deepswe_dist/k8s_launcher.sh"
fi

# Build the container image from the Tunix source tree and push it.
# INSTALL_RAIDEN installs the tpu_sync_jax wheel in raiden_wheels/ (the wheel
# of the scored runs' training image; see ../tunix/SOURCE_VERSION.txt).
if [[ "${BUILD_IMAGE:-1}" == "1" ]]; then
  WHEEL_NAME="tpu_sync_jax-0.0.1.dev20261013212200-0110-cp312-cp312-manylinux_2_31_x86_64.whl"

  # 1. Clone the public TPU Sync (Raiden) repository
  git clone https://github.com/google/tpu-sync.git
  cd tpu-sync

  # 2. Check out the exact commit corresponding to 20261013212200
  git checkout e5f7f2ed4c1cd5a295498c6359ca52b7b1567304

  # 3. Build the JAX wheel hermetically in Docker
  # - RAIDEN_JAX_VERSION=0.11.0 matches the MLPerf v6.1 JAX stack
  # - (Optional) WHEEL_VERSION_EXTRAS reproduces the exact filename stamp
  RAIDEN_JAX_VERSION=0.11.0 \
  WHEEL_VERSION_EXTRAS=.dev20261013212200 \
  bash ci/build_wheel.sh jax

  cd ..
  mkdir -p "${TUNIX_SRC}/raiden_wheels"
  cp "tpu-sync/dist/${WHEEL_NAME}" "${TUNIX_SRC}/raiden_wheels/${WHEEL_NAME}"

  docker build --network=host \
    --build-arg INSTALL_MAXTEXT=true \
    --build-arg INSTALL_RAIDEN=true \
    --build-arg INSTALL_DEEPSWE_DEPS=true \
    -t "${TUNIX_IMAGE}" "${TUNIX_SRC}"
  docker push "${TUNIX_IMAGE}"
fi
