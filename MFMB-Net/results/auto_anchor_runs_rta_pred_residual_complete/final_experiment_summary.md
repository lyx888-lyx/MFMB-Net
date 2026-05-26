# Final Experiment Summary (Complete Modality + Main Missing Sweep)

## 1) Ours@missing=0.0

- MAE: 0.8811
- Corr: 0.7644
- Non0_acc_2: 0.8293
- Non0_F1: 0.8292
- Mult_acc_5: 0.4213
- Mult_acc_7: 0.3712

## 2) Interpretation for Complete-Modality Setting

- Ours@0.0 does not show anomalous degradation and remains in a reasonable range relative to its 0.1~0.5 trend.
- Current complete-modality comparison is limited by baseline availability at missing=0.0 in the full-test baseline artifact; missing entries are explicitly marked as not available.

## 3) Scope Clarification

- The main missing-rate table remains `missing=0.1~0.5` (see residual_main_table_compact).
- AUILC currently covers only `0.1~0.5`, not `0.0~1.0`.
- To produce paper-style AUILC over `0.0~1.0`, additional stress-test runs for `missing=0.6~1.0` are still required.

## 4) Recommended Paper Table Layout

- Table 1: complete modality comparison at missing=0.0
- Table 2: main comparison under missing=0.1~0.5
- Table 3: AUILC@0.1~0.5 and average rank
- Table 4: ablation at missing=0.4/0.5
- Optional Table 5: stress test at missing=0.6~1.0