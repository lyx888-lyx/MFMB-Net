# Extreme Missing-rate Stress Test Report

## 1. Settings
- dataset=MOSI
- missing=0.6~1.0
- methods=text, vision, dynamic_soft, Ours
- protocol=train_drop_last=1, eval_drop_last=0, test_drop_last=0
- effective_test_samples=686

## 2. Main Results
- See `stress_06_10_comparison.csv` and `stress_06_10_table.md/.tex`.

## 3. Comparison with Dynamic Soft
- Ours beats dynamic_soft on MAE at 4/5 missing rates: 0.6, 0.7, 0.8, 0.9.
- dynamic_soft shows clear degradation as missing increases, especially near 0.9~1.0.

## 4. Comparison with Vision Center
- Ours beats vision on MAE at 4/5 missing rates: 0.6, 0.7, 0.8, 0.9.
- Vision remains a strong baseline at some extreme points.

## 5. Comparison with Text Center
- Ours beats text on MAE at 3/5 missing rates: 0.6, 0.8, 0.9.
- Text is competitive at selected extreme points (e.g., 0.7/1.0 in MAE).

## 6. AUILC and Average Rank
- Best MAE-AUILC@0.6~1.0: Ours.
- Best Corr-AUILC@0.6~1.0: Vision center.
- Best Non0_F1-AUILC@0.6~1.0: Text center.
- Best AvgRank_core: Text center; Best AvgRank_all: Text center.
- Ours' strongest advantage is on MAE robustness over most high-missing points.

## 7. Ours Curve from 0.0 to 1.0
- See `ours_curve_00_10.*` and `ours_auilc_00_10.*`.
- Ours degrades as missing increases, with a notable boundary effect near missing=1.0.
- Corr at missing=1.0 is -0.0325; this near-zero/negative behavior is an expected boundary under extreme missing.

## 8. Conclusion
- Ours is a credible high-missing robust candidate in MAE trend (majority wins vs dynamic_soft/vision/text).
- The 0.6~1.0 stress evidence supports robustness claims, but not universal dominance on every metric.
- To claim full-range fairness, adding audio/task-aware/MoE in 0.6~1.0 is recommended as supplemental baselines.
- A full-method AUILC@0.0~1.0 is currently not recommended because not all baselines have official complete 0.0~1.0 coverage.
- Next step: a small-scale MOSEI validation is recommended before large-scale expansion.