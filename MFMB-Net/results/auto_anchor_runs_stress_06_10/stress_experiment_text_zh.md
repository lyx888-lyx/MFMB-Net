本节给出极高缺失率(0.6~1.0)压力测试结果。采用 full-test 协议（train_drop_last=1, eval_drop_last=0, test_drop_last=0），并在 MOSI 全测试集686样本上评估。
结果显示，Ours 在 MAE 上对 dynamic_soft 为 4/5 缺失率更优，对 vision 为 4/5 更优，对 text 为 3/5 更优，说明其在高缺失区间能够有效缓解退化。
同时，missing=1.0 时所有方法都出现明显性能恶化，相关系数可接近零或为负，这反映了极端缺失下的自然边界，而非单一方法失效。
因此，stress test 主要用于鲁棒性边界分析，不替代主表结论。主比较仍建议以 0.1~0.5 区间为核心。