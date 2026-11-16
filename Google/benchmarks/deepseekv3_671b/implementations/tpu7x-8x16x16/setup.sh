#!/bin/bash
# MLPerf Training v6.1 / DeepSeek-V3 671B / tpu7x-8x16x16 / MaxText on TPU7x
# Environment setup: cluster and container image. See README.md sections 1-3.1.
#
# Replace the <placeholders>.
set -euo pipefail
cd "$(dirname "$0")"

export PROJECT_ID="${PROJECT_ID:?set PROJECT_ID (<project>)}"
export CLUSTER_NAME="${CLUSTER_NAME:?set CLUSTER_NAME (<cluster>)}"
export ZONE="${ZONE:?set ZONE (<zone>)}"
export REGION="${REGION:-${ZONE%-*}}"
# Image that run_and_time.sh runs, in a registry the cluster can pull from.
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE:?set WORKLOAD_IMAGE (<your-registry>/mlperf-google:deepseekv3_671b-maxtext)}"

# 1. GKE cluster with a static node pool of 512 x tpu7x-standard-4t hosts (one tpu7x-8x16x16
#    slice, 2048 chips). The submitted runs used an existing cluster of this shape; to create one:
#
#    # 1a. Custom VPC with a large MTU, plus Cloud NAT so that nodes can pull images.
#    NETWORK="${CLUSTER_NAME}-net"
#    SUBNET="${CLUSTER_NAME}-subnet-${REGION}"
#    gcloud compute networks create "${NETWORK}" --project="${PROJECT_ID}" \
#      --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom
#    gcloud compute networks subnets create "${SUBNET}" --project="${PROJECT_ID}" \
#      --network="${NETWORK}" --region="${REGION}" --range=10.11.0.0/18
#    gcloud compute firewall-rules create "${NETWORK}-allow-internal" --project="${PROJECT_ID}" \
#      --network="${NETWORK}" --allow=tcp,udp,icmp
#    gcloud compute routers create "${NETWORK}-router" --project="${PROJECT_ID}" \
#      --network="${NETWORK}" --region="${REGION}"
#    gcloud compute routers nats create "${NETWORK}-nat" --project="${PROJECT_ID}" \
#      --router="${NETWORK}-router" --region="${REGION}" \
#      --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges
#
#    # 1b. Regional cluster.
#    GKE_VERSION=1.36.4-gke.1247000
#    gcloud container clusters create "${CLUSTER_NAME}" --project="${PROJECT_ID}" \
#      --location="${REGION}" --cluster-version="${GKE_VERSION}" \
#      --network="${NETWORK}" --subnetwork="${SUBNET}" \
#      --machine-type=n2d-standard-16
#    gcloud container clusters get-credentials "${CLUSTER_NAME}" --project="${PROJECT_ID}" \
#      --region="${REGION}"
#
#    # 1c. TPU7x node pool in ${ZONE}: 512 tpu7x-standard-4t hosts (one 8x16x16 slice).
#    gcloud container node-pools create "${CLUSTER_NAME}-tpu7x" --project="${PROJECT_ID}" \
#      --cluster="${CLUSTER_NAME}" --location="${REGION}" --node-locations="${ZONE}" \
#      --node-version="${GKE_VERSION}" --machine-type=tpu7x-standard-4t \
#      --tpu-topology=8x16x16 --num-nodes=512 --reservation-affinity=specific \
#      --reservation=<reservation>
#
#    # 1d. JobSet, used by XPK to launch run_and_time.sh.
#    kubectl apply --server-side -f \
#      https://github.com/kubernetes-sigs/jobset/releases/download/v0.10.1/manifests.yaml

# 2. Container image from public sources (Dockerfile): public MaxText base image, pinned PyPI
#    packages, the MaxText source fetched from GitHub at the commit of ../maxtext/SOURCE_VERSION.txt.
docker build -f Dockerfile -t "${WORKLOAD_IMAGE}" .
docker push "${WORKLOAD_IMAGE}"
