ARG BASE_IMAGE
FROM $BASE_IMAGE

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y \
    uuid-runtime \
    sqlite3 libsqlite3-dev \
    libfmt-dev nlohmann-json3-dev

# Install library dependencies
WORKDIR /workspace/deps

# RPD
RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData \
    && cd rocmProfileData \
    && git checkout 13e7600ca75b34078927bbc5a9f9191882e7048e \
    && cd rocpd_python \
    && python3 setup.py bdist_wheel \
    && pip install dist/*.whl \
    && cd .. \
    && cd rpd_tracer \
    && python3 setup.py bdist_wheel \
    && pip3 install dist/*.whl \
    && cd .. \
    && sed -i 's/pip install --user \./pip install ./g' rocpd_python/Makefile \
    && sed -i 's/pip install --user \./pip install ./g' rpd_tracer/Makefile \
    && sed -i '/strncpy(row\.completionSignal/d' rpd_tracer/CuptiDataSource.cpp \
    && make; make install

RUN pip install git+https://github.com/AMD-AGI/TraceLens.git

# AINIC driver (Pensando ionic NIC userspace + libibverbs provider). Required
# inside the container for multi-node RDMA via NCCL_IB_HCA=ionic_*. The bundle
# tarball is downloaded into the build context by mlperf_primus_dev.yaml's
# "Download AINIC bundle from S3" step before the dev image is built.
ARG AINIC_BUNDLE_FILE=ainic_bundle_1.117.5-a-77.tar.gz

COPY Dockerfile ${AINIC_BUNDLE_FILE} /tmp/ainic/
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        jq dpkg-dev kmod xz-utils ethtool iproute2 \
        libibverbs-dev ibverbs-utils infiniband-diags && \
    rm -rf /var/lib/apt/lists/* && \
    cd /tmp/ainic && tar zxf ${AINIC_BUNDLE_FILE} && \
    cd "${AINIC_BUNDLE_FILE%.tar.gz}" && tar zxf host_sw_pkg.tar.gz && \
    cd host_sw_pkg && ./install.sh --domain=user -y && \
    rm -rf /tmp/ainic

WORKDIR /workspace/code
