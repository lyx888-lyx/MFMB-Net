#!/usr/bin/env bash
set -e

# Resume core runs for missing=0.3 only, one mode per call so reports refresh after each mode.
for MODE in audio vision dynamic_soft dynamic_soft_moe; do
  conda run -n mfmb python scripts/auto_run_anchor_experiments.py \
    --datasetName mosi \
    --phase core \
    --missing_list 0.3 \
    --modes ${MODE} \
    --export_anchor_weights 1 \
    --router_missing_bias 2.0 \
    --gpu_ids 0 \
    --resume 1
done
