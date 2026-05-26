# Figure Captions

1. `residual_mae_vs_missing.png`
- EN: MAE trends over missing rates (0.1–0.5) under the full-test protocol, comparing fixed-center, dynamic baselines, and the residual prediction-level RTA.
- 中文：展示 0.1~0.5 缺失率下 MAE 曲线，对比固定中心、动态基线与残差预测级 RTA。

2. `residual_corr_vs_missing.png`
- EN: Correlation trends across missing rates, showing robustness differences among routing strategies.
- 中文：展示相关系数随缺失率变化，体现不同路由策略的鲁棒性差异。

3. `residual_non0_f1_vs_missing.png`
- EN: Non0 F1-score comparison over missing rates, highlighting classification-oriented behavior under missing modalities.
- 中文：展示 Non0 F1 在不同缺失率下的变化，反映缺失条件下分类相关性能。

4. `residual_gate_weight_vs_missing.png`
- EN: Mean gate allocation (`g_task` vs `g_rel`) across missing rates for residual prediction-level RTA.
- 中文：展示残差预测级 RTA 的平均门控分配（`g_task` 与 `g_rel`）随缺失率变化。

5. `residual_average_rank.png`
- EN: Average rank across core metrics, indicating relative overall stability rather than single-point superiority.
- 中文：展示核心指标平均排名，强调整体稳定性而非单点绝对最优。

6. `residual_delta_mae_vs_dynamic_soft.png`
- EN: MAE delta of residual RTA against dynamic_soft; values below zero indicate improvement.
- 中文：展示残差 RTA 相对 dynamic_soft 的 MAE 差值，低于 0 表示提升。
