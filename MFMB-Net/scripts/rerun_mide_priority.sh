#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/usr/miniconda3/envs/mfmb/bin/python
LOGDIR=results/logs
mkdir -p "$LOGDIR" results/results/mide

for m in 0.0 0.3 0.5; do
  rm -f "results/results/mide/mosi-regression-${m}.csv"
  echo "===== $(date) mide-v2 missing=$m ====="
  $PY run.py --missing "$m" --mide_enable --exp_tag mide 2>&1 | tee "$LOGDIR/mide_v2_m${m}.log"
done

$PY scripts/compare_auilc.py \
  --baseline_dir results/results/baseline \
  --mide_dir results/results/mide \
  --output results/auilc_comparison.csv

$PY scripts/generate_mide_report.py \
  --mide_log results/logs/mide_v2_m0.3.log \
  --output results/mide_report.md

echo "MIDE v2 rerun complete."
