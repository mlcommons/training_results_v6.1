# MLPerf Training v6.1 / DeepSeek-V3 671B / tpu7x-8x16x32 / MaxText on TPU7x
# Container image for run_and_time.sh, built from public sources only. MaxText mlperf-dsv3-v6.1
# (commit 24e75daf8e16; see ../maxtext/SOURCE_VERSION.txt) is fetched from GitHub at that commit;
# the build context is not used. setup.sh builds and pushes it.

# Public MaxText 0.2.5 pre-training image (Python 3.12 and the MaxText dependencies), pinned by digest.
FROM us-docker.pkg.dev/cloud-tpu-images/maxtext-images/tpu_pre_training@sha256:b152355a1425c28e99841c014703495059705de1733df8ac837ba43b4fe52fcf

# Pinned package versions (see ../maxtext/requirements_frozen.txt).
RUN pip install -q --upgrade 'jax==0.11.1' 'jaxlib==0.11.1' 'libtpu==0.0.48' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install -q --upgrade 'mlperf_logging==4.1.67' && \
    pip install -q --no-deps 'typeguard==2.13.3'

# Orbax checkpoint-restore barrier timeout raised from 300 s to 1800 s (restoring the 671B
# checkpoint on hundreds of hosts can take longer).
RUN SP=$(python3 -c 'import orbax.checkpoint, os; print(os.path.dirname(orbax.checkpoint.__file__))') && \
    sed -i 's/^_DEFAULT_BARRIER_TIMEOUT = 300$/_DEFAULT_BARRIER_TIMEOUT = 1800/' \
      "$SP/experimental/v1/_src/synchronization/multihost.py" "$SP/_src/multihost/multihost.py" && \
    grep -qx '_DEFAULT_BARRIER_TIMEOUT = 1800' "$SP/experimental/v1/_src/synchronization/multihost.py" && \
    grep -qx '_DEFAULT_BARRIER_TIMEOUT = 1800' "$SP/_src/multihost/multihost.py"

# MaxText source (src/ of the pinned commit), at the path the run command uses (PYTHONPATH=/app/src).
ARG MAXTEXT_COMMIT=24e75daf8e16a644d3a2db5a36212cdfd9bc4331
WORKDIR /app
RUN python3 -c "import tarfile, urllib.request; \
url = 'https://github.com/AI-Hypercomputer/maxtext/archive/${MAXTEXT_COMMIT}.tar.gz'; \
tarfile.open(urllib.request.urlretrieve(url, '/tmp/maxtext.tar.gz')[0]).extractall('/tmp/maxtext')" && \
    mv "/tmp/maxtext/maxtext-${MAXTEXT_COMMIT}/src" /app/src && \
    rm -rf /tmp/maxtext /tmp/maxtext.tar.gz && \
    test -f /app/src/maxtext/configs/base.yml
