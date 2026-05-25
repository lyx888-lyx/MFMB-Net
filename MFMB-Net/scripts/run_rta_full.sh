#!/usr/bin/env bash
set -euo pipefail

conda run -n mfmb python scripts/auto_run_anchor_experiments.py \
  --datasetName mosi \
  --phase full \
  --missing_list 0.1,0.2,0.3,0.4,0.5 \
  --modes dynamic_rta \
  --export_anchor_weights 1 \
  --export_task_router_info 1 \
  --router_missing_bias 2.0 \
  --task_router_lambda 0.1 \
  --center_aux_lambda 0.05 \
  --router_oracle_temperature 0.8 \
  --router_oracle_type soft \
  --use_reliability_task_gate 1 \
  --gate_balance_lambda 0.01 \
  --gate_target 0.5 \
  --gate_hidden_dim 32 \
  --gate_dropout 0.1 \
  --gpu_ids 0 \
  --resume 1 \
  --train_drop_last 1 \
  --eval_drop_last 0 \
  --test_drop_last 0 \
  --output_dir results/auto_anchor_runs_rta

conda run -n mfmb python scripts/generate_rta_outputs.py \
  --fulltest_dir results/auto_anchor_runs_fulltest \
  --taskrouter_final_dir results/auto_anchor_runs_taskrouter_final \
  --rta_dir results/auto_anchor_runs_rta
