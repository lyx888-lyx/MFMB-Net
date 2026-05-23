#!/usr/bin/env bash
set -euo pipefail

PY_BIN="${PY_BIN:-python}"
if command -v conda >/dev/null 2>&1; then
  PY_CMD=(conda run -n mfmb python)
else
  PY_CMD=(${PY_BIN})
fi

DATASET="mosi"
GPU_IDS="0"
ROUTER_BIAS="2.0"
EXPORT_ANCHOR="1"
PHASE="full"

MISSING_LIST=(0.4 0.5 0.2 0.1 0.0 0.3)
MODES=(text audio vision dynamic_soft dynamic_soft_moe)

for miss in "${MISSING_LIST[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] run missing=${miss} mode=${mode}"
    "${PY_CMD[@]}" scripts/auto_run_anchor_experiments.py \
      --datasetName "${DATASET}" \
      --phase "${PHASE}" \
      --missing_list "${miss}" \
      --modes "${mode}" \
      --export_anchor_weights "${EXPORT_ANCHOR}" \
      --router_missing_bias "${ROUTER_BIAS}" \
      --gpu_ids "${GPU_IDS}" \
      --resume 1
  done
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] full prioritized sweep finished"
