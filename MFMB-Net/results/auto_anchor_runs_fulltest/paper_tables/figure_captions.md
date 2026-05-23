## 1) mae_vs_missing.png
EN: MAE across missing rates under the official full-test protocol. Dynamic-soft shows its strongest gain around missing=0.4, while performance degrades at missing=0.5.
ZH: 该图展示不同缺失率下的 MAE。dynamic_soft 在 missing=0.4 附近优势最明显，但在 missing=0.5 出现退化。

## 2) corr_vs_missing.png
EN: Correlation trends under increasing missing rates. Dynamic routing improves correlation at specific mid-missing regimes but does not dominate all ranges.
ZH: 该图展示 Corr 随缺失率变化趋势。动态锚点在中等缺失区间有提升，但并非所有区间都占优。

## 3) non0_acc2_vs_missing.png
EN: Non0 Acc-2 comparison for robustness under missing modalities. Fixed baselines remain competitive at low/high missing, while dynamic_soft peaks around missing=0.4.
ZH: 该图展示 Non0 Acc-2 的鲁棒性对比。低缺失和高缺失区间固定中心仍较强，dynamic_soft 在 0.4 附近达到峰值。

## 4) non0_f1_vs_missing.png
EN: Non0 F1 trends show that dynamic_soft_moe is not consistently better than dynamic_soft, indicating unstable expert collaboration.
ZH: 该图显示 Non0 F1 上 MoE 并未稳定优于 dynamic_soft，说明专家协同仍不稳定。

## 5) router_weights_vs_missing_dynamic_soft.png
EN: Router weight evolution for dynamic_soft. The router is missing-aware (positive availability-weight correlations), yet anchor preference can become biased at certain missing rates (e.g., text tendency at 0.4).
ZH: 该图展示 dynamic_soft 的路由权重变化。虽然 availability-weight 相关性为正，具备缺失感知，但在部分区间会出现偏置（如 0.4 的 text 倾向）。

## 6) router_weights_vs_missing_dynamic_soft_moe.png
EN: Router behavior with Anchor-MoE. The routing often leans toward audio-centric weighting, but this does not always translate to superior task metrics.
ZH: 该图展示 Anchor-MoE 下的路由行为。路由常偏向 audio，但这种偏置并不总能转化为更优任务指标。
