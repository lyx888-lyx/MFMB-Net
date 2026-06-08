#!/usr/bin/env bash
# Sweep MOSI missing rates for baseline or MIDE.
# Usage:
#   bash scripts/run_mide_sweep.sh baseline
#   bash scripts/run_mide_sweep.sh mide
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-baseline}"
RATES="${2:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"

EXTRA_ARGS=()
if [[ "$MODE" == "mide" ]]; then
  EXTRA_ARGS+=(--mide_enable --exp_tag mide)
else
  EXTRA_ARGS+=(--exp_tag baseline)
fi

for m_r in $RATES; do
  echo "===== Running $MODE missing=$m_r ====="
  python run.py --missing "$m_r" "${EXTRA_ARGS[@]}"
done
