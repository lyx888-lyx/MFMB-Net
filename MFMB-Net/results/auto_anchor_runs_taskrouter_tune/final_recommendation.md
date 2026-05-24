# Final Recommendation

1. 0.4 MAE best: lambda=0.10, temp=0.8, MAE=1.0896
2. 0.4 Corr best: lambda=0.20, temp=0.3, Corr=0.5472
3. 0.4 Acc/F1 best: Acc->(0.10,0.8)=0.7256; F1->(0.10,0.8)=0.7248
4. 0.5 MAE best: lambda=0.20, temp=0.5, MAE=1.1668
5. 0.5 Corr best: lambda=0.20, temp=0.5, Corr=0.4670
6. 0.5 Acc/F1 best: Acc->(0.20,0.5)=0.7012; F1->(0.20,0.5)=0.7019
7. Avg MAE best: lambda=0.10, temp=0.8, Avg MAE=1.1509
8. Avg Corr best: lambda=0.20, temp=0.5, Avg Corr=0.5062
9. Avg Non0_F1 best: lambda=0.20, temp=0.5, Avg F1=0.7107
10. Comprehensive recommended: lambda=0.10, temp=0.8, center_aux_lambda=0.05

Answer to key question:
Yes, under the current selection rule, we still recommend task_router_lambda=0.1 and router_oracle_temperature=0.8.

missing=0.4: tuned vs original -> MAE 1.0896 vs 1.1603, Corr 0.5194 vs 0.5401, F1 0.7248 vs 0.7114.
missing=0.4: tuned vs dynamic_soft deltas -> MAE -0.0025, Corr -0.0330, Acc -0.0010, F1 -0.0003.
missing=0.4: tuned vs best-fixed-by-MAE (text) -> MAE -0.0163, Corr -0.0303, Acc +0.0203, F1 +0.0203.
missing=0.5: tuned vs original -> MAE 1.2121 vs 1.2424, Corr 0.4663 vs 0.4499, F1 0.6861 vs 0.6714.
missing=0.5: tuned vs dynamic_soft deltas -> MAE -0.0866, Corr +0.0874, Acc +0.0615, F1 +0.0450.
missing=0.5: tuned vs best-fixed-by-MAE (vision) -> MAE -0.0373, Corr +0.0043, Acc +0.0147, F1 +0.0098.