# Residual RTA Gate-Margin Check Report

Settings: dynamic_rta_pred_residual, seed=111, missing=0.4/0.5, protocol trainDL1/evalDL0/testDL0.

## Summary (All Margins)

 gate_margin  missing  seed    MAE   Corr  Non0_acc_2  Non0_F1_score  Mult_acc_5  Mult_acc_7  mean_g_task  mean_g_rel  gate_oracle_match  margin_mask_ratio                                                                                                                                                    log_path
      0.0500   0.4000   111 1.0826 0.5418      0.7348         0.7345      0.2930      0.2770       0.4764      0.5236             0.4300             0.9767                               results/auto_anchor_runs_rta_pred_residual/logs/mosi_m0.4_dynamic_rta_pred_residual_seed111_trainDL1_evalDL0_testDL0_full.log
      0.1000   0.4000   111 1.0770 0.5284      0.7043         0.7028      0.3309      0.3003       0.5282      0.4718             0.4592             0.1676 results/auto_anchor_runs_rta_pred_residual_margin_check/gate_margin_0.10/logs/mosi_m0.4_dynamic_rta_pred_residual_seed111_trainDL1_evalDL0_testDL0_full.log
      0.1500   0.4000   111 1.1594 0.5508      0.6890         0.6943      0.3163      0.2813       0.5301      0.4699             0.4927             0.8120 results/auto_anchor_runs_rta_pred_residual_margin_check/gate_margin_0.15/logs/mosi_m0.4_dynamic_rta_pred_residual_seed111_trainDL1_evalDL0_testDL0_full.log
      0.0500   0.5000   111 1.1599 0.4615      0.6936         0.6932      0.2682      0.2580       0.4486      0.5514             0.4752             0.8878                               results/auto_anchor_runs_rta_pred_residual/logs/mosi_m0.5_dynamic_rta_pred_residual_seed111_trainDL1_evalDL0_testDL0_full.log
      0.1000   0.5000   111 1.2453 0.4445      0.6585         0.6617      0.2085      0.2085       0.4627      0.5373             0.4752             0.8382 results/auto_anchor_runs_rta_pred_residual_margin_check/gate_margin_0.10/logs/mosi_m0.5_dynamic_rta_pred_residual_seed111_trainDL1_evalDL0_testDL0_full.log
      0.1500   0.5000   111 1.2644 0.4800      0.6616         0.6650      0.2682      0.2478       0.3953      0.6047             0.5510             0.8294 results/auto_anchor_runs_rta_pred_residual_margin_check/gate_margin_0.15/logs/mosi_m0.5_dynamic_rta_pred_residual_seed111_trainDL1_evalDL0_testDL0_full.log

## Comparison vs Baselines (dynamic_soft / dynamic_task_soft_tuned)

 gate_margin  missing    MAE   Corr  Non0_acc_2  Non0_F1_score  margin_mask_ratio  gate_oracle_match  delta_MAE_vs_dynamic_soft  delta_MAE_vs_task_tuned  delta_Corr_vs_task_tuned  delta_Non0_F1_vs_task_tuned
      0.0500   0.4000 1.0826 0.5418      0.7348         0.7345             0.9767             0.4300                    -0.0095                  -0.0070                    0.0224                       0.0097
      0.1000   0.4000 1.0770 0.5284      0.7043         0.7028             0.1676             0.4592                    -0.0151                  -0.0126                    0.0090                      -0.0220
      0.1500   0.4000 1.1594 0.5508      0.6890         0.6943             0.8120             0.4927                     0.0673                   0.0698                    0.0314                      -0.0305
      0.0500   0.5000 1.1599 0.4615      0.6936         0.6932             0.8878             0.4752                    -0.1389                  -0.0522                   -0.0048                       0.0071
      0.1000   0.5000 1.2453 0.4445      0.6585         0.6617             0.8382             0.4752                    -0.0535                   0.0332                   -0.0218                      -0.0244
      0.1500   0.5000 1.2644 0.4800      0.6616         0.6650             0.8294             0.5510                    -0.0344                   0.0523                    0.0137                      -0.0211

## Best Margin by MAE
- missing=0.4: best MAE margin = 0.10
- missing=0.5: best MAE margin = 0.05
- avg MAE over 0.4/0.5: best margin = 0.05

## Feasibility Checks
- margin=0.05: c1(0.4 close to dynamic_soft)=True, c2(0.5 better than dynamic_soft by >=0.03)=True, c3(0.5 Corr/F1 not worse than tuned task-aware)=True
- margin=0.10: c1(0.4 close to dynamic_soft)=True, c2(0.5 better than dynamic_soft by >=0.03)=True, c3(0.5 Corr/F1 not worse than tuned task-aware)=False
- margin=0.15: c1(0.4 close to dynamic_soft)=False, c2(0.5 better than dynamic_soft by >=0.03)=True, c3(0.5 Corr/F1 not worse than tuned task-aware)=False

## Recommendation
- Recommended gate_margin: 0.05
- Reason: best overall MAE; although mask ratio is high, performance remains strongest/most stable.