| missing | best_MAE_method | best_Corr_method | best_Non0_acc_2_method | best_Non0_F1_method | main_observation |
|---:|---|---|---|---|---|
| 0.1 | vision | vision | text | text | Under low missing, fixed vision/text remain competitive. |
| 0.2 | audio | audio | vision | vision | At mild missing, fixed audio/vision are stronger while dynamic routing is not yet beneficial. |
| 0.3 | vision | vision | vision | vision | At moderate missing, fixed vision remains strongest overall; MoE helps over dynamic_soft but is still below vision. |
| 0.4 | dynamic_soft | dynamic_soft | dynamic_soft | dynamic_soft | At higher missing, dynamic_soft achieves the best performance across all core metrics. |
| 0.5 | vision | text | vision | vision | At very high missing, dynamic_soft degrades and fixed baselines (especially vision/text) recover strength. |
