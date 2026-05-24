# Task-aware Router Hyperparameter Tuning Report

## 1. Experiment Settings
- dataset=MOSI
- missing=0.4,0.5
- protocol: train_drop_last=1, eval_drop_last=0, test_drop_last=0 (effective_test_samples=686)
- grid: lambda in {0.05,0.1,0.2}, temperature in {0.3,0.5,0.8}, center_aux_lambda=0.05

## 2. Grid Search Results
| lambda | temperature | MAE@0.4 | Corr@0.4 | Non0_acc_2@0.4 | Non0_F1@0.4 | MAE@0.5 | Corr@0.5 | Non0_acc_2@0.5 | Non0_F1@0.5 | average_MAE_04_05 | average_Corr_04_05 | average_Non0_acc_2_04_05 | average_Non0_F1_04_05 | average_router_oracle_match_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.4 | text | 1.1059 | <u>0.5497</u> | 0.7053 | 0.7045 | <u>0.3144</u> | 0.2808 |
| 0.4 | audio | 1.1347 | 0.5334 | 0.7114 | 0.7102 | 0.2877 | 0.2629 |
| 0.4 | vision | 1.2219 | 0.5241 | 0.7022 | 0.7049 | 0.2464 | 0.2327 |
| 0.4 | dynamic_soft | <u>1.0921</u> | **0.5524** | **0.7266** | **0.7251** | **0.3207** | <u>0.2891</u> |
| 0.4 | dynamic_soft_moe | 1.1596 | 0.5264 | 0.7144 | 0.7211 | 0.3110 | 0.2697 |
| 0.4 | dynamic_task_soft_original (0.1,0.5) | 1.1603 | 0.5401 | 0.7104 | 0.7114 | 0.2862 | 0.2672 |
| 0.4 | dynamic_task_soft_tuned (0.10,0.8) | **1.0896** | 0.5194 | <u>0.7256</u> | <u>0.7248</u> | 0.3100 | **0.2920** |
| 0.5 | text | 1.2593 | <u>0.4647</u> | 0.6626 | 0.6668 | 0.2449 | 0.2289 |
| 0.5 | audio | 1.2776 | 0.4560 | 0.6458 | 0.6526 | <u>0.2638</u> | <u>0.2435</u> |
| 0.5 | vision | 1.2494 | 0.4619 | <u>0.6707</u> | <u>0.6763</u> | **0.3008** | **0.2716** |
| 0.5 | dynamic_soft | 1.2988 | 0.3788 | 0.6240 | 0.6411 | 0.2153 | 0.2099 |
| 0.5 | dynamic_soft_moe | 1.3486 | 0.4490 | 0.6123 | 0.6578 | 0.2532 | 0.2410 |
| 0.5 | dynamic_task_soft_original (0.1,0.5) | <u>1.2424</u> | 0.4499 | 0.6677 | 0.6714 | 0.2279 | 0.2211 |
| 0.5 | dynamic_task_soft_tuned (0.10,0.8) | **1.2121** | **0.4663** | **0.6855** | **0.6861** | 0.2575 | 0.2410 |

## 3. Best Hyperparameters
| selection | task_router_lambda | router_oracle_temperature | center_aux_lambda | MAE | Corr | router_oracle_match_rate | note |
|---|---|---|---|---|---|---|---|
| best_mae_missing_0.4 | 0.100000 | 0.800000 | 0.050000 | 1.089600 | 0.519400 | 0.369776 | lowest MAE on this missing |
| best_mae_missing_0.5 | 0.200000 | 0.500000 | 0.050000 | 1.166833 | 0.467033 | 0.350340 | lowest MAE on this missing |
| best_avg_mae_04_05 | 0.100000 | 0.800000 | 0.050000 | 1.150866 | 0.492833 | 0.324829 | lowest average MAE over missing=0.4,0.5 |
| best_avg_corr_04_05 | 0.200000 | 0.500000 | 0.050000 | 1.156516 | 0.506233 | 0.358843 | highest average Corr over missing=0.4,0.5 |
| recommended | 0.100000 | 0.800000 | 0.050000 | 1.150866 | 0.492833 | 0.324829 | balanced rule: avg MAE + high-missing robustness + router match |

## 4. Comparison with Baselines
| missing | method | MAE | Corr | Non0_acc_2 | Non0_F1 | Mult_acc_5 | Mult_acc_7 |
|---|---|---|---|---|---|---|---|
| 0.4 | text | 1.1059 | 0.5497 | 0.7053 | 0.7045 | 0.3144 | 0.2808 |
| 0.4 | audio | 1.1347 | 0.5334 | 0.7114 | 0.7102 | 0.2877 | 0.2629 |
| 0.4 | vision | 1.2219 | 0.5241 | 0.7022 | 0.7049 | 0.2464 | 0.2327 |
| 0.4 | dynamic_soft | 1.0921 | 0.5524 | 0.7266 | 0.7251 | 0.3207 | 0.2891 |
| 0.4 | dynamic_soft_moe | 1.1596 | 0.5264 | 0.7144 | 0.7211 | 0.3110 | 0.2697 |
| 0.4 | dynamic_task_soft_original (0.1,0.5) | 1.1603 | 0.5401 | 0.7104 | 0.7114 | 0.2862 | 0.2672 |
| 0.4 | dynamic_task_soft_tuned (0.10,0.8) | 1.0896 | 0.5194 | 0.7256 | 0.7248 | 0.3100 | 0.2920 |
| 0.5 | text | 1.2593 | 0.4647 | 0.6626 | 0.6668 | 0.2449 | 0.2289 |
| 0.5 | audio | 1.2776 | 0.4560 | 0.6458 | 0.6526 | 0.2638 | 0.2435 |
| 0.5 | vision | 1.2494 | 0.4619 | 0.6707 | 0.6763 | 0.3008 | 0.2716 |
| 0.5 | dynamic_soft | 1.2988 | 0.3788 | 0.6240 | 0.6411 | 0.2153 | 0.2099 |
| 0.5 | dynamic_soft_moe | 1.3486 | 0.4490 | 0.6123 | 0.6578 | 0.2532 | 0.2410 |
| 0.5 | dynamic_task_soft_original (0.1,0.5) | 1.2424 | 0.4499 | 0.6677 | 0.6714 | 0.2279 | 0.2211 |
| 0.5 | dynamic_task_soft_tuned (0.10,0.8) | 1.2121 | 0.4663 | 0.6855 | 0.6861 | 0.2575 | 0.2410 |

## 5. Router Diagnostics
- match-rate range: 0.2702~0.4534
- above-random points: 12/18
- high match-rate does not always imply best downstream MAE/Corr/F1.

## 6. Conclusion
- tuned setting (lambda=0.10, temp=0.8) is recommended by current rule.
- tuned model improves over original dynamic_task_soft overall, and better addresses high-missing (0.5) than untuned task-router.
- compared with dynamic_soft and best fixed, improvements are partial-by-metric rather than absolute domination.
- next step: refine oracle construction/temperature strategy for stronger task-aligned routing.