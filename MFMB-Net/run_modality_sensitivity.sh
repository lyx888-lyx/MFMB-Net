#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-mosei}"
MODEL="${2:-mfmb_net}"
RUN_TAG="${3:-modality_sensitivity}"
GPU_IDS="${GPU_IDS:-}"
SEEDS="${SEEDS:-111,1111,11111}"
SINGLE_STEPS="${SINGLE_STEPS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
PAIR_STEPS="${PAIR_STEPS:-0.0 0.25 0.5 0.75 1.0}"

run_case() {
  local mt="$1"
  local ma="$2"
  local mv="$3"
  echo "[RUN] dataset=${DATASET} t=${mt} a=${ma} v=${mv} tag=${RUN_TAG}"
  python run.py \
    --datasetName "${DATASET}" \
    --modelName "${MODEL}" \
    --missing_t "${mt}" \
    --missing_a "${ma}" \
    --missing_v "${mv}" \
    --gpu_ids "${GPU_IDS}" \
    --seeds "${SEEDS}" \
    --run_tag "${RUN_TAG}"
}

# 1) baseline
run_case 0.0 0.0 0.0

# 2) single-modality sensitivity curves
for r in ${SINGLE_STEPS}; do
  run_case "${r}" 0.0 0.0
  run_case 0.0 "${r}" 0.0
  run_case 0.0 0.0 "${r}"
done

# 3) pairwise heatmaps: hold one modality complete while the other two degrade
for a in ${PAIR_STEPS}; do
  for v in ${PAIR_STEPS}; do
    run_case 0.0 "${a}" "${v}"
  done
done

for t in ${PAIR_STEPS}; do
  for v in ${PAIR_STEPS}; do
    run_case "${t}" 0.0 "${v}"
  done
done

for t in ${PAIR_STEPS}; do
  for a in ${PAIR_STEPS}; do
    run_case "${t}" "${a}" 0.0
  done
done
