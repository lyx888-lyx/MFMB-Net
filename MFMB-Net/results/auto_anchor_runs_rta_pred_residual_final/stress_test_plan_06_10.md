# Stress Test Plan (0.6~1.0)

This is an extreme-missing stress test plan and is **not** a replacement for the main 0.1~0.5 results.

## Goal
Evaluate whether Ours (`dynamic_rta_pred_residual`) remains more stable than `dynamic_soft` under very high missing rates.

## Scope
- Missing rates: 0.6, 0.7, 0.8, 0.9, 1.0
- Methods: `text`, `vision`, `dynamic_soft`, `dynamic_rta_pred_residual`
- Protocol: train_drop_last=1, eval_drop_last=0, test_drop_last=0
- Output dir: `results/auto_anchor_runs_stress_06_10/`

## Notes
- Results should be audited with manifest/protocol tags.
- If Ours underperforms at 0.6~1.0, this does not invalidate 0.1~0.5 conclusions, but should be reported as a boundary condition.
- Planned post-processing outputs:
  - `stress_06_10_comparison.csv`
  - `stress_06_10_auilc.csv`
  - `stress_06_10_report.md`
