#!/usr/bin/env bash
set -e

conda run -n mfmb python scripts/auto_run_anchor_experiments.py \
  --datasetName mosi \
  --phase full \
  --missing_list 0.3,0.4,0.5 \
  --modes text,audio,vision,dynamic_soft,dynamic_soft_moe \
  --export_anchor_weights 1 \
  --router_missing_bias 2.0 \
  --gpu_ids 0 \
  --resume 1 \
  --train_drop_last 1 \
  --eval_drop_last 0 \
  --test_drop_last 0 \
  --output_dir results/auto_anchor_runs_fulltest
