# Result Source Report

## 1. Paper Table Eligible Sources
- OK: `results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv`
- OK: `results/auto_anchor_runs_taskrouter_final/final_main_comparison.csv`
- OK: `results/auto_anchor_runs_rta/rta_final_comparison.csv`

## 2. Debug-Only Sources
- `results/results/normals/*.csv`
- WARNING: normals CSV does not contain sufficient experiment metadata and should not be used for paper tables.

## 3. Official CSV Audit
### `results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv`
- rows: `25`
- columns: `['dataset', 'phase', 'missing', 'mode', 'protocol_tag', 'train_drop_last', 'eval_drop_last', 'test_drop_last', 'task_router_lambda', 'center_aux_lambda', 'router_oracle_temperature', 'router_oracle_type', 'gate_balance_lambda', 'gate_target', 'MAE_mean', 'MAE_std', 'Corr_mean', 'Corr_std', 'Non0_acc_2_mean', 'Non0_acc_2_std', 'Non0_F1_score_mean', 'Non0_F1_score_std', 'Mult_acc_5_mean', 'Mult_acc_5_std', 'Mult_acc_7_mean', 'Mult_acc_7_std', 'effective_test_samples']`
- field check missing: OK
- field check mode/method: OK
- field check MAE: OK
- field check Corr: OK
- field check Non0_acc_2: OK
- field check Non0_F1_score: OK
- missing coverage: `[0.1, 0.2, 0.3, 0.4, 0.5]`
- mode coverage: `['audio', 'dynamic_soft', 'dynamic_soft_moe', 'text', 'vision']`

### `results/auto_anchor_runs_taskrouter_final/final_main_comparison.csv`
- rows: `30`
- columns: `['missing', 'method', 'MAE', 'Corr', 'Non0_acc_2', 'Non0_F1_score', 'Mult_acc_5', 'Mult_acc_7']`
- field check missing: OK
- field check mode/method: OK
- field check MAE: OK
- field check Corr: OK
- field check Non0_acc_2: OK
- field check Non0_F1_score: OK
- missing coverage: `[0.1, 0.2, 0.3, 0.4, 0.5]`
- method coverage: `['audio', 'dynamic_soft', 'dynamic_soft_moe', 'dynamic_task_soft_tuned', 'text', 'vision']`

### `results/auto_anchor_runs_rta/anchor_experiment_agg.csv`
- rows: `5`
- columns: `['dataset', 'phase', 'missing', 'mode', 'train_drop_last', 'eval_drop_last', 'test_drop_last', 'MAE_mean', 'MAE_std', 'Corr_mean', 'Corr_std', 'Non0_acc_2_mean', 'Non0_acc_2_std', 'Non0_F1_score_mean', 'Non0_F1_score_std', 'Mult_acc_5_mean', 'Mult_acc_5_std', 'Mult_acc_7_mean', 'Mult_acc_7_std', 'effective_test_samples']`
- field check missing: OK
- field check mode/method: OK
- field check MAE: OK
- field check Corr: OK
- field check Non0_acc_2: OK
- field check Non0_F1_score: OK
- missing coverage: `[0.1, 0.2, 0.3, 0.4, 0.5]`
- mode coverage: `['dynamic_rta']`

### `results/auto_anchor_runs_rta/rta_final_comparison.csv`
- rows: `35`
- columns: `['missing', 'method', 'MAE', 'Corr', 'Non0_acc_2', 'Non0_F1_score', 'Mult_acc_5', 'Mult_acc_7']`
- field check missing: OK
- field check mode/method: OK
- field check MAE: OK
- field check Corr: OK
- field check Non0_acc_2: OK
- field check Non0_F1_score: OK
- missing coverage: `[0.1, 0.2, 0.3, 0.4, 0.5]`
- method coverage: `['audio', 'dynamic_rta', 'dynamic_soft', 'dynamic_soft_moe', 'dynamic_task_soft_tuned', 'text', 'vision']`

## 4. Duplicate / Protocol Mixing Risk
- No protocol mixing detected in fulltest agg by `(missing, mode)`.

## 5. Recommendation
- Safe paper sources (priority):
  1. `results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv`
  2. `results/auto_anchor_runs_taskrouter_final/final_main_comparison.csv`
  3. `results/auto_anchor_runs_rta/rta_final_comparison.csv`
- Do not use `results/results/normals/*.csv` except debugging.