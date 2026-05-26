#!/usr/bin/env bash
set -e

OUT=results/auto_anchor_runs_stress_06_10
DATASET=mosi
MISSING=0.6,0.7,0.8,0.9,1.0

# text
conda run -n mfmb python scripts/auto_run_anchor_experiments.py   --datasetName ${DATASET} --phase full --missing_list ${MISSING} --modes text   --gpu_ids 0 --resume 1   --train_drop_last 1 --eval_drop_last 0 --test_drop_last 0   --output_dir ${OUT}

# vision
conda run -n mfmb python scripts/auto_run_anchor_experiments.py   --datasetName ${DATASET} --phase full --missing_list ${MISSING} --modes vision   --gpu_ids 0 --resume 1   --train_drop_last 1 --eval_drop_last 0 --test_drop_last 0   --output_dir ${OUT}

# dynamic_soft
conda run -n mfmb python scripts/auto_run_anchor_experiments.py   --datasetName ${DATASET} --phase full --missing_list ${MISSING} --modes dynamic_soft   --export_anchor_weights 1 --router_missing_bias 2.0   --gpu_ids 0 --resume 1   --train_drop_last 1 --eval_drop_last 0 --test_drop_last 0   --output_dir ${OUT}

# Ours: dynamic_rta_pred_residual
conda run -n mfmb python scripts/auto_run_anchor_experiments.py   --datasetName ${DATASET} --phase full --missing_list ${MISSING} --modes dynamic_rta_pred_residual   --export_anchor_weights 1 --export_task_router_info 1 --router_missing_bias 2.0   --use_reliability_task_gate 1 --use_prediction_gate_supervision 1 --rta_pred_residual 1   --gate_supervision_mode margin --gate_margin 0.05   --task_router_lambda 0.1 --center_aux_lambda 0.05   --router_oracle_temperature 0.8 --router_oracle_type soft   --gate_oracle_temperature 0.8 --gate_task_lambda 0.02   --gate_balance_lambda 0.0 --gate_target 0.5 --gate_init_bias -2.0   --gate_hidden_dim 32 --gate_dropout 0.1   --gpu_ids 0 --resume 1   --train_drop_last 1 --eval_drop_last 0 --test_drop_last 0   --output_dir ${OUT}

# post-process (planned)
# conda run -n mfmb python scripts/generate_stress_outputs.py --input_dir ${OUT}
