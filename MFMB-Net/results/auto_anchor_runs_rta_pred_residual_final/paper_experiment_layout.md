# Suggested Paper Experiment Layout

## Table 1
Complete modality comparison at missing=0.0.
- If `Ours@0.0` is not available, mark it as pending and avoid forcing it into final claims.

## Table 2
Main comparison under missing=0.1~0.5.
- Methods: Text/Audio/Vision, Dynamic Soft, Task-aware Dynamic, Ours.

## Table 3
AUILC@0.1~0.5 + average metrics + average rank.
- Emphasize average robustness and ranking stability.

## Table 4
Ablation at key missing rates (0.4/0.5).
- Include dynamic_soft, dynamic_soft_moe, task-aware, feature-RTA, pred-RTA, residual pred-RTA.

## Figure 1
Metric curves over missing=0.1~0.5 (MAE/Corr/Non0-F1 etc.).

## Optional Table 5
Extreme missing stress test 0.6~1.0.

## Important rule
Do **not** report AUILC over 0.0~1.0 unless Ours is fully evaluated at both missing=0.0 and 0.6~1.0.
