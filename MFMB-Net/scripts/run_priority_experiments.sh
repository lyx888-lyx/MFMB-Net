#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/usr/miniconda3/envs/mfmb/bin/python
LOGDIR=results/logs
mkdir -p "$LOGDIR"

run_exp() {
  local mode="$1" missing="$2"
  local -a extra=()
  if [[ "$mode" == "mide" ]]; then
    extra=(--mide_enable --exp_tag mide)
  else
    extra=(--exp_tag baseline)
  fi
  echo "===== $(date) $mode missing=$missing ====="
  $PY run.py --missing "$missing" "${extra[@]}" 2>&1 | tee "$LOGDIR/${mode}_m${missing}.log"
}

# Skip baseline 0.0 if CSV already exists
if [[ ! -f results/results/baseline/mosi-regression-0.0.csv ]]; then
  run_exp baseline 0.0
fi

for m in 0.3 0.5; do
  if [[ ! -f "results/results/baseline/mosi-regression-${m}.csv" ]]; then
    run_exp baseline "$m"
  fi
done

for m in 0.0 0.3 0.5; do
  if [[ ! -f "results/results/mide/mosi-regression-${m}.csv" ]]; then
    run_exp mide "$m"
  fi
done

$PY scripts/compare_auilc.py \
  --baseline_dir results/results/baseline \
  --mide_dir results/results/mide \
  --output results/auilc_comparison.csv

$PY scripts/generate_mide_report.py \
  --mide_log results/logs/mide_m0.3.log \
  --output results/mide_report.md

echo "Priority experiments done at $(date)"
