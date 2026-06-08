#!/usr/bin/env bash
# Full experiment suite: baseline + MIDE on MOSI missing rates.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-/usr/miniconda3/envs/mfmb/bin/python}"
RATES="${RATES:-0.0 0.3 0.5}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"

run_one() {
  local mode="$1" missing="$2"
  local extra=()
  if [[ "$mode" == "mide" ]]; then
    extra=(--mide_enable --exp_tag mide)
  else
    extra=(--exp_tag baseline)
  fi
  echo "[$(date)] START $mode missing=$missing"
  $PY run.py --missing "$missing" "${extra[@]}" 2>&1 | tee "$LOGDIR/${mode}_m${missing}.log"
  echo "[$(date)] DONE $mode missing=$missing"
}

for m in $RATES; do
  run_one baseline "$m"
done

for m in $RATES; do
  run_one mide "$m"
done

$PY scripts/compare_auilc.py \
  --baseline_dir results/results/baseline \
  --mide_dir results/results/mide \
  --output results/auilc_comparison.csv

echo "Suite complete."
