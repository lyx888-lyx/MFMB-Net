#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/usr/miniconda3/envs/mfmb/bin/python
LOGDIR=results/logs
mkdir -p "$LOGDIR" results/results/mide_r3

for m in 0.0 0.3 0.5; do
  echo "===== $(date) mide-r3 missing=$m ====="
  $PY run.py --missing "$m" --mide_enable --exp_tag mide_r3 2>&1 | tee "$LOGDIR/mide_r3_m${m}.log"
done

echo "MIDE r3 complete."
