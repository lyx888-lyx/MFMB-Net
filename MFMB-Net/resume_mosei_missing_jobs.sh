#!/usr/bin/env bash
# Resume only MOSEI (datasetName mosei) job combinations that are not COMPLETE.
# COMPLETE = all three non-empty files exist:
#   results/predictions/miss_${miss}_${center}_${lte}/predictions_mosi_fc_${center}_seed{111,1111,11111}_test*.csv
# Run this script from the directory that contains run.py (same as original commands).

set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

if [[ ! -f "run.py" ]]; then
  echo "Error: run.py not found in ${ROOT}. Run this script from the project directory that contains run.py." >&2
  exit 1
fi

LOGDIR="results/logs/mosei_resume"
mkdir -p "$LOGDIR"

# Returns 0 if combo is COMPLETE, 1 otherwise.
is_complete() {
  local miss="$1" center="$2" lte="$3"
  local d="results/predictions/miss_${miss}_${center}_${lte}"
  local s seed
  for seed in 111 1111 11111; do
    shopt -s nullglob
    local files=( "$d"/predictions_mosi_fc_"${center}"_seed${seed}_test*.csv )
    shopt -u nullglob
    if [[ ${#files[@]} -eq 0 ]]; then
      return 1
    fi
    local f found=0
    for f in "${files[@]}"; do
      if [[ -s "$f" ]]; then
        found=1
        break
      fi
    done
    if [[ $found -ne 1 ]]; then
      return 1
    fi
  done
  return 0
}

TS="$(date +%Y%m%d-%H%M%S)"
any_run=0

for miss in 0.0 0.1 0.2 0.3 0.4 0.5 0.6; do
  for center in text dynamic_missing; do
    for lte in legacy mstcn; do
      if is_complete "$miss" "$center" "$lte"; then
        echo "SKIP (COMPLETE): miss=$miss center=$center lte=$lte"
        continue
      fi
      any_run=1
      out="$LOGDIR/mosei_${miss}_${center}_${lte}_${TS}.log"
      echo "==== RUN: miss=$miss center=$center lte=$lte  log=$out  ====" | tee "$out"
      {
        echo "==== START $(date -Is) ===="
        echo "CWD: $(pwd)"
        python run.py \
          --missing "$miss" \
          --fusion_center_modality "$center" \
          --local_temporal_encoder_type "$lte" \
          --datasetName mosei \
          --debug_data_inspect \
          --export_test_predictions \
          --export_pred_dir "results/predictions/miss_${miss}_${center}_${lte}" \
          2>&1
        echo "==== END $(date -Is) exit=$? ===="
      } 2>&1 | tee -a "$out"
    done
  done
done

if [[ "$any_run" -eq 0 ]]; then
  echo "All 28 combinations are already COMPLETE (3 non-empty prediction CSVs each). No jobs to run."
fi
