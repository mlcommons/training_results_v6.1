#!/bin/bash
# redact_logs.sh -- the log post-processing of firmus_gb300_ngpu8_gptoss_g1_1012: makes result_<i>.txt from the raw
# log of run i+1 ($LOGDIR/<DATESTAMP>_1.log, see run_and_time.sh).
#
#   bash redact_logs.sh <raw log> > result_<i>.txt
#
# One change, of the kind training_rules.adoc TR-11.2 allows (internal paths may be redacted): on every line that is not
# an MLLOG line, each path prefix /shared/<dir>/<dir> (the cluster's internal storage) becomes "[internal storage]".
# MLLOG lines are copied byte for byte, so every key and value the MLPerf checkers read is the raw log's.
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 <raw log>" >&2; exit 2; }
sed -E '/:::MLLOG/!s#/shared/[^/[:space:]]+/[^/[:space:]]+#[internal storage]#g' "$1"
