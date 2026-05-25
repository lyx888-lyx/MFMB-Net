# Final Task-aware Dynamic Anchor Router Report

## 1. Settings
- dataset=MOSI
- protocol=train_drop_last=1, eval_drop_last=0, test_drop_last=0 (686 test samples)
- tuned parameters: task_router_lambda=0.1, center_aux_lambda=0.05, router_oracle_temperature=0.8, router_oracle_type=soft
- missing=0.1~0.5
- baselines included: text/audio/vision/dynamic_soft/dynamic_soft_moe

## 2. Main Results
- dynamic_task_soft_tuned outperforms dynamic_soft on MAE in 2/5 missing rates.
- dynamic_task_soft_tuned outperforms dynamic_soft_moe on MAE in 3/5 missing rates.
- dynamic_task_soft_tuned outperforms best fixed center on MAE in 2/5 missing rates.

## 3. Comparison with Dynamic Soft
- tuned dynamic_task_soft improves robustness at high missing (especially 0.5) compared with dynamic_soft.
- dynamic_soft still keeps a strong advantage on some medium-missing settings (notably around 0.4 in prior runs).

## 4. Comparison with Best Fixed Center
- best fixed center remains strong on some missing rates.
- tuned dynamic_task_soft is competitive but not uniformly superior to best fixed center.

## 5. Comparison with MoE
- tuned dynamic_task_soft is generally more stable than dynamic_soft_moe.
- MoE is not recommended as the main module under this protocol.

## 6. Router Diagnostics
- router_oracle_match_rate over missing 0.1~0.5: 0.301, 0.302, 0.304, 0.370, 0.280.
- task-aware supervision helps, but router-oracle alignment is still moderate and not sufficient alone to guarantee best task performance.

## 7. Conclusion
- tuned dynamic_task_soft is a strong primary candidate due to improved high-missing robustness and better stability than MoE.
- further oracle/router refinement is still needed to consistently exceed the best fixed center across all missing rates.
- next step: keep tuned setting as default and improve task-aware supervision design before extending to MOSEI/asymmetric missing.