# RTA Pred Residual Quick Report

Protocol: train_drop_last=1, eval_drop_last=0, test_drop_last=0; missing=0.4/0.5; seed=111.

## Method Comparison

 missing                    method    MAE   Corr  Non0_acc_2  Non0_F1_score  Mult_acc_5  Mult_acc_7 best_fixed_name
  0.4000              dynamic_soft 1.0921 0.5524      0.7266         0.7251      0.3207      0.2891            text
  0.4000   dynamic_task_soft_tuned 1.0896 0.5194      0.7256         0.7248      0.3100      0.2920            text
  0.4000       dynamic_rta_feature 1.1803 0.5414      0.6951         0.7043      0.2483      0.2366            text
  0.4000          dynamic_rta_pred 1.1625 0.5496      0.6966         0.7003      0.3251      0.2843            text
  0.4000 dynamic_rta_pred_residual 1.0826 0.5418      0.7348         0.7345      0.2930      0.2770            text
  0.4000                best_fixed 1.1059 0.5497      0.7053         0.7045      0.3144      0.2808            text
  0.5000              dynamic_soft 1.2988 0.3788      0.6240         0.6411      0.2153      0.2099          vision
  0.5000   dynamic_task_soft_tuned 1.2121 0.4663      0.6855         0.6861      0.2575      0.2410          vision
  0.5000       dynamic_rta_feature 1.1886 0.4590      0.6865         0.6868      0.2886      0.2673          vision
  0.5000          dynamic_rta_pred 1.2226 0.4780      0.7165         0.7149      0.2259      0.2187          vision
  0.5000 dynamic_rta_pred_residual 1.1599 0.4615      0.6936         0.6932      0.2682      0.2580          vision
  0.5000                best_fixed 1.2494 0.4619      0.6707         0.6763      0.3008      0.2716          vision

## Residual Model Summary

 missing    MAE   Corr  Non0_acc_2  Non0_F1_score  Mult_acc_5  Mult_acc_7  delta_MAE_vs_dynamic_soft  delta_Corr_vs_dynamic_soft  delta_Non0_F1_vs_dynamic_soft  delta_MAE_vs_dynamic_task_soft_tuned  delta_Corr_vs_dynamic_task_soft_tuned  delta_Non0_F1_vs_dynamic_task_soft_tuned  delta_MAE_vs_dynamic_rta_pred  delta_Corr_vs_dynamic_rta_pred  delta_Non0_F1_vs_dynamic_rta_pred  delta_MAE_vs_best_fixed  delta_Corr_vs_best_fixed  delta_Non0_F1_vs_best_fixed  mean_g_task  mean_g_rel  gate_margin_mask_ratio  gate_oracle_match_rate
  0.4000 1.0826 0.5418      0.7348         0.7345      0.2930      0.2770                    -0.0095                     -0.0106                         0.0094                               -0.0070                                 0.0224                                    0.0097                        -0.0799                         -0.0078                             0.0342                  -0.0233                   -0.0079                       0.0300       0.4764      0.5236                  0.9767                  0.4300
  0.5000 1.1599 0.4615      0.6936         0.6932      0.2682      0.2580                    -0.1389                      0.0827                         0.0521                               -0.0522                                -0.0048                                    0.0071                        -0.0627                         -0.0165                            -0.0217                  -0.0895                   -0.0004                       0.0169       0.4486      0.5514                  0.8878                  0.4752

## Acceptance Check
- c1_04_close_dynamic_soft: True
- c2_05_better_dynamic_soft_by_0.03: True
- c3_05_corr_not_worse_taskaware: True
- c3_05_f1_not_worse_taskaware: True
- c4_gate_not_collapsed: True
- c5_margin_mask_ratio_gt_0.1: True

## Recommendation
- Pass quick criteria; it is worth running 0.1~0.5 with 3 seeds.