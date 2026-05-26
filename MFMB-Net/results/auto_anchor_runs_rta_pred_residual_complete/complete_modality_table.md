Table X. Comparison of complete-modality setting on CMU-MOSI under full-test protocol (missing=0.0).

| missing | method | MAE | Corr | Non0_acc_2 | Non0_F1 | Mult_acc_5 | Mult_acc_7 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0 | text | not available | not available | not available | not available | not available | not available |
| 0.0 | audio | not available | not available | not available | not available | not available | not available |
| 0.0 | vision | not available | not available | not available | not available | not available | not available |
| 0.0 | dynamic_soft | not available | not available | not available | not available | not available | not available |
| 0.0 | dynamic_task_soft_tuned | not available | not available | not available | not available | not available | not available |
| 0.0 | dynamic_rta_pred_residual | **0.8811** | **0.7644** | **0.8293** | **0.8292** | **0.4213** | **0.3712** |

Note: Full-test protocol: train_drop_last=1, eval_drop_last=0, test_drop_last=0. MOSI test set is evaluated on all 686 samples.
Note: Methods marked `not available` do not have official missing=0.0 runs in current full-test baseline artifacts.