# Dynamic RTA Gate Ablation Report

## Summary

missing,rta_gate_mode,seed,MAE,Corr,Non0_acc_2,Non0_F1_score,Mult_acc_5,Mult_acc_7,mean_g_task,mean_g_rel,log_path
0.4,fixed_half,111,1.2711,0.3372,0.6387,0.6464,0.2332,0.2172,,,results/auto_anchor_runs_rta_gate_ablation/mode_fixed_half/logs/mosi_m0.4_dynamic_rta_seed111_trainDL1_evalDL0_testDL0_full.log
0.4,force_rel,111,1.1033,0.5444,0.7256,0.7313,0.2843,0.2711,,,results/auto_anchor_runs_rta_gate_ablation/mode_force_rel/logs/mosi_m0.4_dynamic_rta_seed111_trainDL1_evalDL0_testDL0_full.log
0.4,force_task,111,1.0549,0.5509,0.7363,0.7368,0.328,0.312,,,results/auto_anchor_runs_rta_gate_ablation/mode_force_task/logs/mosi_m0.4_dynamic_rta_seed111_trainDL1_evalDL0_testDL0_full.log
0.4,learned,111,1.0585,0.5421,0.7271,0.7283,0.3265,0.3047,,,results/auto_anchor_runs_rta_gate_ablation/mode_learned/logs/mosi_m0.4_dynamic_rta_seed111_trainDL1_evalDL0_testDL0_full.log
0.5,fixed_half,111,1.1235,0.5165,0.7226,0.721,0.2872,0.2784,,,results/auto_anchor_runs_rta_gate_ablation/mode_fixed_half/logs/mosi_m0.5_dynamic_rta_seed111_trainDL1_evalDL0_testDL0_full.log
0.5,force_rel,111,1.1636,0.4567,0.7073,0.7076,0.3076,0.2697,,,results/auto_anchor_runs_rta_gate_ablation/mode_force_rel/logs/mosi_m0.5_dynamic_rta_seed111_trainDL1_evalDL0_testDL0_full.log
0.5,force_task,111,1.1249,0.4828,0.7058,0.7072,0.2726,0.2624,,,results/auto_anchor_runs_rta_gate_ablation/mode_force_task/logs/mosi_m0.5_dynamic_rta_seed111_trainDL1_evalDL0_testDL0_full.log
0.5,learned,111,1.3157,0.465,0.6174,0.6333,0.1822,0.1808,,,results/auto_anchor_runs_rta_gate_ablation/mode_learned/logs/mosi_m0.5_dynamic_rta_seed111_trainDL1_evalDL0_testDL0_full.log


## Key Findings
- missing=0.4:
  - force_rel vs dynamic_soft MAE delta: +0.0112
  - force_task vs dynamic_task_soft_tuned MAE delta: -0.0347
  - learned MAE - fixed_half MAE: -0.2126 (negative means learned better)
- missing=0.5:
  - force_rel vs dynamic_soft MAE delta: -0.1352
  - force_task vs dynamic_task_soft_tuned MAE delta: -0.0872
  - learned MAE - fixed_half MAE: +0.1922 (negative means learned better)

## Interpretation
- If force_rel is far from dynamic_soft, H_rel path may be non-equivalent.
- If force_task is far from dynamic_task_soft_tuned, H_task path may be non-equivalent.
- If force_rel/force_task are reasonable but learned is bad, gate learning is likely unstable.
- If fixed_half beats learned consistently, learned gate may be overfitting/noisy.