#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --comment=docker
#SBATCH --time=8:00:00
#SBATCH --job-name=bruno-build-image
#SBATCH --output=/home/hpebench/slurm-logs/%x-%j.out
#
# Build a Docker image from a benchmark implementation folder and keep durable
# copies of it (and of its base image, which the build pulls) on /nas.
# Skips the build if the image is already on the node or saved on /nas.
#
# Usage: sbatch -w <node> --export=ALL,CONT=rocm/amd-mlperf:dlrmv4_training_6.1,CTX=<impl dir> \
#               build_image.sbatch [docker build args ...]
set -uo pipefail
if [[ -d /nas/data/SHARED/MLPERF/containers ]]; then NAS=/nas/data/SHARED/MLPERF; else NAS=/nas/SHARED/MLPERF; fi
CONT=${CONT:?set CONT}
CTX=${CTX:?set CTX to the build context}
tarname() { echo "${NAS}/containers/$(echo "${1#rocm/}" | sed 's/@.*//' | tr '/:' '__').tar"; }
save() {  # image tar
    [[ -f "$2" ]] && return 0
    docker save -o "$2.part" "$1" && mv "$2.part" "$2" && chmod 644 "$2" && echo "saved $2 ($(du -h "$2" | cut -f1))"
}

for _ in $(seq 1 60); do [[ $(squeue -h -w "$(hostname -s)" -t CG | wc -l) -eq 0 ]] && break; sleep 10; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not available"; exit 1; }
echo "=== $(date -u +%H:%M) ${CONT} from ${CTX} on $(hostname), $(docker --version)"
tar=$(tarname "${CONT}")
if ! docker image inspect "${CONT}" >/dev/null 2>&1 && [[ -f "${tar}" ]]; then
    docker load -q -i "${tar}"
fi
if ! docker image inspect "${CONT}" >/dev/null 2>&1; then
    base=$(sed -n 's/^ARG BASE_IMAGE=//p; s/^FROM \([^ ]*\).*/\1/p' "${CTX}/Dockerfile" | grep -v '^\(stack\|fbgemm\)$' | head -1)
    btag=${base%@*}
    btar=$(tarname "${btag}")
    if [[ -n "${base}" ]] && ! docker image inspect "${btag}" >/dev/null 2>&1; then
        if [[ -f "${btar}" ]]; then docker load -q -i "${btar}"
        else echo "=== $(date -u +%H:%M) pulling base ${base}"; docker pull -q "${base}" && docker tag "${base}" "${btag}"; fi
    fi
    echo "=== $(date -u +%H:%M) build"
    # Optional: fetch large files with parallel range requests into /nas/wheels
    # and serve them locally; pip's single stream is too slow on lodin.
    # PREFETCH_ARG=<build-arg name> PREFETCH_BASE=<remote base URL> PREFETCH_FILES="<url-encoded names>"
    if [[ -n "${PREFETCH_ARG:-}" ]]; then
        wdir=${NAS}/wheels; mkdir -p "${wdir}"
        for f in ${PREFETCH_FILES:?}; do
            out=${wdir}/$(python3 -c 'import sys,urllib.parse; print(urllib.parse.unquote(sys.argv[1]))' "$f")
            size=$(curl -fsSI "${PREFETCH_BASE}/${f}" | awk 'tolower($1)=="content-length:" {print $2+0}')
            [[ -f "${out}" && $(stat -c %s "${out}") -eq ${size} ]] && continue
            echo "prefetch ${f} (${size} bytes)"
            part=$(( size / 8 + 1 ))
            for i in $(seq 0 7); do
                ( a=$(( i * part )); b=$(( (i + 1) * part - 1 < size - 1 ? (i + 1) * part - 1 : size - 1 ))
                  for n in 1 2 3 4 5 6; do
                      [[ -f "${out}.p${i}" && $(stat -c %s "${out}.p${i}") -eq $(( b - a + 1 )) ]] && exit 0
                      curl -fsS --retry 5 --http1.1 -r "${a}-${b}" "${PREFETCH_BASE}/${f}" -o "${out}.p${i}"
                      sleep $(( n * 5 ))
                  done; exit 1 ) &
            done
            wait
            cat "${out}".p? > "${out}.part" && rm -f "${out}".p? && mv "${out}.part" "${out}"
            [[ $(stat -c %s "${out}") -eq ${size} ]] || { echo "FATAL: prefetch size mismatch ${f}"; exit 1; }
            want=$(curl -fsS "${PREFETCH_BASE}/${f%%-*}/" | grep -o "${f}#sha256=[0-9a-f]*" | head -1 | cut -d= -f2)
            if [[ -n "${want}" ]]; then
                [[ $(sha256sum "${out}" | cut -d' ' -f1) == "${want}" ]] || { echo "FATAL: sha256 mismatch ${f}"; rm -f "${out}"; exit 1; }
                echo "sha256 OK ${f}"
            fi
        done
        python3 -m http.server 18765 --bind 127.0.0.1 -d "${wdir}" >/dev/null 2>&1 &
        srv=$!; trap 'kill ${srv} 2>/dev/null' EXIT; sleep 2
        set -- --build-arg "${PREFETCH_ARG}=http://127.0.0.1:18765" "$@"
    fi
    DOCKER_BUILDKIT=1 docker build --network=host --progress=plain -t "${CONT}" "$@" "${CTX}" \
        || { echo "FATAL: build failed"; exit 1; }
    if [[ -n "${base}" ]]; then save "${btag}" "${btar}" & savepid=$!; fi
fi
echo "=== $(date -u +%H:%M) image present: $(docker image inspect --format '{{.Id}}' "${CONT}")"
save "${CONT}" "${tar}"
# Only the base-image save: a bare wait would also wait for the wheel server.
[[ -n "${savepid:-}" ]] && wait "${savepid}"
[[ -n "${srv:-}" ]] && kill "${srv}" 2>/dev/null
docker run --rm "${CONT}" bash -c '
  echo "ROCm (image): $(cat /opt/rocm/.info/version 2>/dev/null)"
  pip list 2>/dev/null | grep -iE "^(torch|torchrec|fbgemm|triton|mlperf|primus)"'
echo "=== $(date -u +%H:%M) done"
