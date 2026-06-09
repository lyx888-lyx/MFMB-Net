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
