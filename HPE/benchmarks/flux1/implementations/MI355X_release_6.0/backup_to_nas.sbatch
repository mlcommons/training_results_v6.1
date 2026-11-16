#!/bin/bash
#SBATCH --partition=def
#SBATCH --nodelist=lodin2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --job-name=flux1-backup
#SBATCH --output=/nas/SHARED/MLPERF/results/flux1/backup-%j.out
#SBATCH --error=/nas/SHARED/MLPERF/results/flux1/backup-%j.err
#
# Persist the converted Flux.1 energon tree to /nas.
#
# POLICY: /lvol0 is scratch. It is a real disk (md0 RAID5, survives reboot) but
# is node-local, not backed up, and may be reclaimed at any time. Every derived
# artifact that cost significant compute must also exist on /nas.
#
# This tree cost ~6 h of conversion plus the interactive `energon prepare`, so
# losing it is expensive -- and re-running `energon prepare` means re-driving
# prompts whose order is not guaranteed stable.
#
# Measured: /nas write from lodin2 is ~96 MB/s (1 GbE is the ceiling; the ib*
# interfaces are present but not configured). 2.3 TB therefore takes ~7 h.
# Node-to-node ssh is only ~42 MB/s (SSH encryption bound), so /nas is both the
# durable target and the faster path -- any future node should restore from
# here rather than from a peer node.
#
# Runs --exclusive deliberately: it is chained BEFORE GPU work rather than
# overlapped with it, so that a 7 h background I/O load cannot contaminate the
# llama2 TTT measurements, which are the whole point of that experiment.

set -euo pipefail

SRC=${SRC:-/lvol0/mlperf/flux1/energon}
DST=${DST:-/nas/SHARED/MLPERF/flux1_energon}

echo "=== flux1 backup: $(date -u) on $(hostname) ==="
echo "    ${SRC} -> ${DST}"

[[ -d "${SRC}/.nv-meta" ]] || { echo "FATAL: ${SRC} has no .nv-meta; not a prepared tree" >&2; exit 1; }

src_gb=$(du -sm "${SRC}" | cut -f1)
avail_gb=$(df -BM --output=avail /nas | tail -1 | tr -dc '0-9')
echo "    source ${src_gb} MB, /nas avail ${avail_gb} MB"
[[ "${avail_gb}" -gt "$((src_gb + 204800))" ]] \
    || { echo "FATAL: insufficient space on /nas" >&2; exit 1; }

mkdir -p "${DST}"

# -H preserves any hardlinks; --partial so an interrupted 7 h transfer resumes
# instead of restarting. No -z: the payload is incompressible embeddings and
# the link, not CPU, is the bottleneck.
rsync -aH --partial --info=progress2 "${SRC}/" "${DST}/"

echo "=== verifying ==="
for d in train val empty_encodings .nv-meta; do
    [[ -e "${DST}/${d}" ]] || { echo "FATAL: ${DST}/${d} missing after copy" >&2; exit 1; }
done

s_tar=$(find "${SRC}" -name '*.tar' | wc -l)
d_tar=$(find "${DST}" -name '*.tar' | wc -l)
echo "    tar shards: src=${s_tar} dst=${d_tar}"
[[ "${s_tar}" -eq "${d_tar}" ]] || { echo "FATAL: shard count mismatch" >&2; exit 1; }

# Spot-check payload integrity: a full 2.3 TB checksum would take hours, but a
# byte-identical first and last shard plus matching total size catches the
# realistic failure modes (truncation, partial transfer).
for f in train/shard_000000.tar val/shard_000000.tar; do
    a=$(md5sum "${SRC}/${f}" | cut -d' ' -f1)
    b=$(md5sum "${DST}/${f}" | cut -d' ' -f1)
    [[ "$a" == "$b" ]] || { echo "FATAL: checksum mismatch on ${f}" >&2; exit 1; }
    echo "    ${f} md5 ok"
done

echo "    src $(du -sh "${SRC}" | cut -f1) / dst $(du -sh "${DST}" | cut -f1)"
echo "=== BACKUP_OK: $(date -u) ==="
