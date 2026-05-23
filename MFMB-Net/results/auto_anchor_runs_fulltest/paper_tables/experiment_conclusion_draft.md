## 中文版本
在 CMU-MOSI 的 official full-test 协议（train_drop_last=1, eval_drop_last=0, test_drop_last=0）下，我们系统比较了 fixed text/audio/vision、dynamic_soft 与 dynamic_soft_moe。结果表明，dynamic_soft 在中等缺失区间具有明确潜力：在 missing=0.4 时，dynamic_soft 在 MAE、Corr、Non0 Acc-2、Non0 F1 等核心指标上均取得最优，说明“缺失感知动态锚点”在该区间能够带来有效收益。

但该优势并非全区间稳定。随着缺失率继续升高到 missing=0.5，dynamic_soft 出现退化，fixed vision/text 再次变得更有竞争力。这说明当前动态路由策略虽有可行性，但在高缺失极端场景下仍存在泛化不足。

对于 Anchor-MoE，实验显示其在 missing=0.3 可优于 dynamic_soft，但在 0.4/0.5 区间表现不稳定，尚不足以作为主结论模块。综合来看，本工作验证了动态锚点机制的可行性与潜力，同时也揭示了其局限：后续需要引入 task-aware router（例如任务损失驱动的路由监督或置信度约束）以提升跨缺失区间的一致性与可解释性。

## English Version
Under the official CMU-MOSI full-test protocol (train_drop_last=1, eval_drop_last=0, test_drop_last=0), we compared fixed text/audio/vision anchors with dynamic_soft and dynamic_soft_moe. The results demonstrate the feasibility of missing-aware dynamic anchoring: dynamic_soft achieves the strongest overall performance at missing=0.4 across core metrics (MAE, Corr, Non0 Acc-2, and Non0 F1).

However, the gain is not uniform across all missing rates. At missing=0.5, dynamic_soft degrades, while fixed vision/text baselines become more competitive again. This suggests that the current routing strategy is promising but not yet robust in highly sparse regimes.

For Anchor-MoE, improvements are observed at missing=0.3, but the behavior is unstable at 0.4/0.5 and therefore insufficient as a primary contribution in its current form. Overall, the study demonstrates the potential of dynamic anchoring while also revealing limitations, motivating a task-aware router design in future work (e.g., performance-aware supervision or confidence-guided routing constraints).
