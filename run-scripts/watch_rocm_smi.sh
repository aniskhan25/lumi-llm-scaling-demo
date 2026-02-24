#!/bin/bash
set -euo pipefail

OUT_FILE="${1:-logs/rocm_smi_watch.log}"
INTERVAL="${2:-1}"
mkdir -p "$(dirname "$OUT_FILE")"

echo "Writing rocm-smi telemetry to $OUT_FILE (interval ${INTERVAL}s)"

while true; do
  TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "=== $TS ==="
    rocm-smi --showuse --showmemuse --showtemp --showpower
    echo
  } | tee -a "$OUT_FILE"
  sleep "$INTERVAL"
done
