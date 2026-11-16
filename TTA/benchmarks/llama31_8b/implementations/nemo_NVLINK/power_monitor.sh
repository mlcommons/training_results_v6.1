#!/bin/bash
# IPMI power monitor — launched by run.sub as:
#   srun --overlap --ntasks-per-node=1 sudo -n numactl --cpunodebind=0 --membind=0 \
#       bash $POWERCMDDIR/power_monitor.sh "${POWERLOGDIR}" "${SLURM_JOB_ID}"
# Runs whole-script as root (single sudo — no per-iteration sudo overhead).
# sudoers env_reset strips POWERLOGDIR/SLURM_JOB_ID, so they are passed as args.
# Output line format MUST NOT change — parsed by mlperf-logging power_measurement.py.
OUTDIR="${1:-.}"; JOBID="${2:-manual}"
[ -d "$OUTDIR" ] || OUTDIR="."
OUT="${OUTDIR}/IPMIPower_${JOBID}.txt"
while true; do
    output=$(ipmitool dcmi power reading)
    power=$(echo "$output" | grep "Instantaneous" | awk '{print $4}')
    timestamp=$(echo "$output" | grep "IPMI timestamp" | cut -d':' -f2- | xargs)
    # el10 ipmitool (>=1.8.19) prints "MM/DD/YYYY HH:MM:SS AM/PM TZ ..." instead of the
    # el8 ctime form — normalize so the output format stays identical across OS versions.
    case "$timestamp" in
        */*)
            ts_clean=$(echo "$timestamp" | sed -E 's/ Sampling period.*//; s/ [A-Z]{2,5}$//')
            ts_legacy=$(LC_ALL=C date -d "$ts_clean" '+%a %b %d %H:%M:%S %Y' 2>/dev/null)
            [ -n "$ts_legacy" ] && timestamp="$ts_legacy"
            ;;
    esac
    if [ -n "$power" ]; then
        echo "Instantaneous power reading:                   ${power} Watts     IPMI timestamp:                           ${timestamp}"
    fi
    sleep 1
done >> "$OUT"
