# MIDE Split-AUP Ablation Report

## 文件修改摘要表

# MIDE Split-AUP 文件修改摘要（修改后）

| 文件 | 修改内容 | 修改原因 | 是否影响兼容性 |
|------|----------|----------|----------------|
| `models/missingTask/MFMB_NET/mide.py` | 新增 `SplitAUPController`、`build_targets`、`compute_d_eff`、分阶段 loss；保留 `OldMIDEController` | 修正 A/U/P 监督与 density gate；支持 old/split 变体 | 是（MIDE 接口扩展） |
| `models/missingTask/MFMB_NET/fusion_599.py` | true LOO（mask raw + 重算 AV fusion）；单次 D_eff 缩放；train/eval gate 一致 | 修复假 LOO、double scaling、warmup/eval 不一致 | 是 |
| `models/missingTask/MFMB_NET/model.py` | 向 fusion 传递 `mide_variant` | 显式上下文，避免隐式全局状态 | 否 |
| `trains/missingTask/MFMB_NET.py` | AMP、分阶段 MIDE loss、Corr tie-break、延后 early-stop、epoch_stats JSON | 训练稳定性与可复现日志 | 否 |
| `config/config_regression.py` | 新增 split-AUP 超参；batch/lr override | 统一默认与 CLI 覆盖 | 否 |
| `run.py` | CLI（amp、mide_*、override）；日志路径 `results/logs/` | 消融实验与结果组织 | 否 |
| `scripts/run_mide_ablation.sh` | 四组实验 × 3 missing | 自动化消融 | 否 |
| `scripts/collect_ablation.py` | 汇总 CSV + markdown 报告 | 结果对比与诚实结论 | 否 |


## Ablation 实验矩阵

| exp_tag | missing | Has0_acc_2 | Non0_acc_2 | MAE | Corr | Loss |
|---------|--------:|-----------:|-----------:|----:|-----:|-----:|
| baseline_fixed | 0.0 | 81.15 | 82.57 | 75.11 | 78.15 | 74.4800 |
| baseline_fixed | 0.3 | 74.59 | 75.71 | 102.85 | 60.97 | 101.4000 |
| baseline_fixed | 0.5 | 67.06 | 67.58 | 119.44 | 46.36 | 118.0500 |
| mide_old_repro | 0.0 | 80.03 | 81.30 | 129.35 | 77.77 | 129.5000 |
| mide_old_repro | 0.3 | 71.23 | 71.80 | 109.60 | 57.05 | 108.3700 |
| mide_old_repro | 0.5 | 67.88 | 68.75 | 131.22 | 45.66 | 130.1100 |
| mide_split_aup | 0.0 | 80.90 | 81.61 | 88.52 | 77.71 | 88.4500 |
| mide_split_aup | 0.3 | 73.13 | 73.22 | 102.77 | 59.76 | 101.7800 |
| mide_split_aup | 0.5 | 64.04 | 65.09 | 146.18 | 47.08 | 145.2900 |
| mide_split_plus | 0.0 | 81.00 | 82.42 | 87.44 | 77.70 | 87.3900 |
| mide_split_plus | 0.3 | 73.76 | 74.24 | 116.90 | 60.88 | 114.8700 |
| mide_split_plus | 0.5 | 67.30 | 67.78 | 152.46 | 46.51 | 151.0400 |

## 推荐超参数表

| exp_tag | batch | lr_other | lr_bert | amp | mide_variant | notes |
|---------|------:|---------:|--------:|:---:|--------------|-------|
| baseline_fixed | 32 | 0.0025 | 1e-5 | on | off | split-AUP fixes |
| mide_old_repro | 32 | 0.0025 | 1e-5 | on | old | split-AUP fixes |
| mide_split_aup | 32 | 0.0025 | 1e-5 | on | split_aup | split-AUP fixes |
| mide_split_plus | 40 | 0.0028 | 1e-5 | on | split_aup_plus | split-AUP fixes |

## Baseline vs MIDE 对比（相对 baseline 的 Has0/MAE/Corr 变化）


### mide_old_repro

| missing | ΔHas0 | ΔMAE(better↑) | ΔCorr |
|--------:|------:|--------------:|------:|
| 0.0 | -1.12 | -54.24 | -0.38 |
| 0.3 | -3.36 | -6.75 | -3.92 |
| 0.5 | +0.82 | -11.78 | -0.70 |

### mide_split_aup

| missing | ΔHas0 | ΔMAE(better↑) | ΔCorr |
|--------:|------:|--------------:|------:|
| 0.0 | -0.25 | -13.41 | -0.44 |
| 0.3 | -1.46 | +0.08 | -1.21 |
| 0.5 | -3.02 | -26.74 | +0.72 |

### mide_split_plus

| missing | ΔHas0 | ΔMAE(better↑) | ΔCorr |
|--------:|------:|--------------:|------:|
| 0.0 | -0.15 | -12.33 | -0.45 |
| 0.3 | -0.83 | -14.05 | -0.09 |
| 0.5 | +0.24 | -33.02 | +0.15 |

## AUILC 表

| metric | baseline_auilc | mide_auilc | delta_mide_minus_baseline | higher_is_better | mide_wins |
|---|---|---|---|---|---|
| MAE | -48.923 | -57.587 | 8.664000000000009 | False | False |
| Corr | 31.601 | 31.526000000000003 | -0.0749999999999957 | True | False |
| Has0_acc_2 | 37.526 | 37.32 | -0.206000000000003 | True | False |
| Has0_F1_score | 37.554 | 37.362 | -0.1920000000000001 | True | False |
| Non0_acc_2 | 38.071 | 37.70099999999999 | -0.3700000000000045 | True | False |
| Non0_F1_score | 38.0615 | 37.7 | -0.3614999999999995 | True | False |
| Mult_acc_5 | 18.5625 | 19.223 | 0.660499999999999 | True | True |
| Mult_acc_7 | 16.6105 | 14.481 | -2.1294999999999984 | True | False |


## 修复项分析

- **True LOO**: `_mask_raw_inputs` + 重算 `audio_visual_fusion` / pairwise stacks / y_without_m_ref
- **Warmup/eval 一致**: `gate_enabled(epoch)` 在 train/eval 共用；epoch ≤ gate_start 时 D_eff=1
- **单次 density 缩放**: 主 rep × D_eff；stack × sqrt(D_i D_j)；AV × sqrt(D_a D_v)
- **A/U/P 监督**: `build_targets` 分离 uni/loo utility 与 pollution target

## Batch size 影响

- missing=0.0: plus(40) vs aup(32) MAE 87.44 vs 88.52, Corr 77.70 vs 77.71
- missing=0.3: plus(40) vs aup(32) MAE 116.90 vs 102.77, Corr 60.88 vs 59.76
- missing=0.5: plus(40) vs aup(32) MAE 152.46 vs 146.18, Corr 46.51 vs 47.08

## 最终结论

- **未全面超越 baseline_fixed**。主 checkpoint 均按 best_by_loss 选取，无 cherry-pick。
- missing=0.0：所有 MIDE 变体 MAE 均差于 baseline（split_aup -13.4，split_plus -12.3，old -54.2）。
- missing=0.3：**mide_split_aup** 最接近 baseline（MAE 102.77 vs 102.85，几乎持平；Corr -1.21）。
- missing=0.5：Has0 上 split_plus (+0.24) / old (+0.82) 略好，但 MAE 全面劣于 baseline（split_aup -26.7，split_plus -33.0）。
- AUILC（baseline vs mide_split_plus）：MAE/Corr/Has0 均未胜出，仅 Mult_acc_5 略胜。
- **工程修复已生效**（true LOO、gate 一致、单次缩放、BCE logits），但当前超参下 split-AUP 尚未在 MAE/Corr 上稳定超越 baseline。
- 若继续迭代，优先保留 **mide_split_aup**（batch=32），而非 batch=40 的 plus 版本。
