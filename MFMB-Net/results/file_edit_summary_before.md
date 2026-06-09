# 修改前文件概览

| 文件 | 当前作用 | 当前问题 | 计划修改点 |
|------|----------|----------|------------|
| `mide.py` | MIDE 头、A/U/P、损失 | bias=1.5 与注释不一致；KL 监督 U；无 D_eff gate；无分阶段训练 | 重构 split A/U/P；build_targets；D_eff；BCE 监督；L_sep；三阶段 loss |
| `fusion_599.py` | Fusion + 伪 LOO | y_without_m 未重算 AV fusion；common 与 stack 双重缩放；train/eval gate 不一致 | true LOO 重算路径；单次缩放；统一 gate 逻辑 |
| `model.py` | 主模型 forward | 未传 variant | 保持接口，传完整 mide_context |
| `MFMB_NET.py` (trainer) | 训练循环 | 无 AMP；early stop 从 epoch1；无 epoch stats；单 checkpoint | AMP、Corr tie-break、分阶段 early stop、json 统计 |
| `config_regression.py` | 超参 | 缺 split 参数与 override | 新增 mide_* 默认与 batch/lr override |
| `run.py` | 入口 | CLI 不全；日志路径 | 扩展 CLI、日志目录 |
| `load_data.py` | DataLoader | 已修复 valid/test | 保持不变 |
| `AMIO.py` | 包装 | 已支持 kwargs | 保持不变 |
| `compare_auilc.py` | AUILC | parse 已修 | 复用 |
