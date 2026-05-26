## A. Full-sweep Ablation (0.1~0.5)

| missing | method | MAE | Corr | Non0_F1_score |
|---|---|---|---|---|
| 0.1 | dynamic_soft | **0.8649** | **0.7350** | 0.8039 |
| 0.1 | dynamic_task_soft_tuned | _0.8828_ | _0.7238_ | _0.8077_ |
| 0.1 | dynamic_rta_feature | 0.9504 | 0.7209 | **0.8119** |
| 0.1 | dynamic_rta_pred_residual | 0.9068 | 0.7181 | 0.7924 |
| 0.2 | dynamic_soft | 1.0409 | 0.6402 | 0.7667 |
| 0.2 | dynamic_task_soft_tuned | 1.0936 | _0.6547_ | 0.7444 |
| 0.2 | dynamic_rta_feature | _0.9497_ | 0.6546 | _0.7760_ |
| 0.2 | dynamic_rta_pred_residual | **0.9356** | **0.6700** | **0.7844** |
| 0.3 | dynamic_soft | _1.0610_ | _0.6053_ | _0.7395_ |
| 0.3 | dynamic_task_soft_tuned | 1.0803 | 0.5968 | 0.7356 |
| 0.3 | dynamic_rta_feature | 1.1093 | 0.5847 | 0.7284 |
| 0.3 | dynamic_rta_pred_residual | **1.0224** | **0.6168** | **0.7475** |
| 0.4 | dynamic_soft | _1.0921_ | **0.5524** | **0.7251** |
| 0.4 | dynamic_task_soft_tuned | **1.0896** | 0.5194 | _0.7248_ |
| 0.4 | dynamic_rta_feature | 1.1803 | _0.5414_ | 0.7043 |
| 0.4 | dynamic_rta_pred_residual | 1.1534 | 0.5352 | 0.7095 |
| 0.5 | dynamic_soft | 1.2988 | 0.3788 | 0.6411 |
| 0.5 | dynamic_task_soft_tuned | 1.2121 | **0.4663** | 0.6861 |
| 0.5 | dynamic_rta_feature | _1.1886_ | _0.4590_ | _0.6868_ |
| 0.5 | dynamic_rta_pred_residual | **1.1875** | 0.4542 | **0.7010** |


## B. Key-point Ablation (0.4, 0.5)

| missing | method | MAE | Corr | Non0_F1_score |
|---|---|---|---|---|
| 0.4 | dynamic_soft | _1.0921_ | **0.5524** | **0.7251** |
| 0.4 | dynamic_soft_moe | 1.1596 | 0.5264 | 0.7211 |
| 0.4 | dynamic_task_soft_tuned | **1.0896** | 0.5194 | _0.7248_ |
| 0.4 | dynamic_rta_feature | 1.1803 | 0.5414 | 0.7043 |
| 0.4 | dynamic_rta_pred | 1.1625 | _0.5496_ | 0.7003 |
| 0.4 | dynamic_rta_pred_residual | 1.1534 | 0.5352 | 0.7095 |
| 0.5 | dynamic_soft | 1.2988 | 0.3788 | 0.6411 |
| 0.5 | dynamic_soft_moe | 1.3486 | 0.4490 | 0.6578 |
| 0.5 | dynamic_task_soft_tuned | 1.2121 | _0.4663_ | 0.6861 |
| 0.5 | dynamic_rta_feature | _1.1886_ | 0.4590 | 0.6868 |
| 0.5 | dynamic_rta_pred | 1.2226 | **0.4780** | **0.7149** |
| 0.5 | dynamic_rta_pred_residual | **1.1875** | 0.4542 | _0.7010_ |
