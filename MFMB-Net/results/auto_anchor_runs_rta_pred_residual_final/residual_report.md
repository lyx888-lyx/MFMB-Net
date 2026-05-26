# Residual Prediction-level RTA Report

## 1. Motivation
Feature-level RTA was unstable. Residual prediction-level gating keeps reliability route as backbone and applies task-aware correction only when useful.

## 2. Method
`pred_final = pred_rel + g * (pred_task - pred_rel)` with margin-supervised gate.

## 3. Main Results
See `results/auto_anchor_runs_rta_pred_residual_final/residual_final_comparison.csv` and main tables.

## 4. Comparison with Dynamic Soft
Dynamic residual RTA beats dynamic_soft on MAE in 3/5 missing settings.

## 5. Comparison with Task-aware Router
Dynamic residual RTA beats dynamic_task_soft_tuned on MAE in 3/5 missing settings.

## 6. Comparison with Best Fixed
Dynamic residual RTA beats best_fixed on MAE in 2/5 missing settings.

## 7. Gate Diagnostics
See `results/auto_anchor_runs_rta_pred_residual_final/residual_gate_diagnostics.csv` for `g_task/g_rel`, margin mask ratio, and oracle-match trends.

## 8. Conclusion
Use residual gate as final Ours if it preserves 0.4 strength and improves 0.5 while keeping stable average rank.