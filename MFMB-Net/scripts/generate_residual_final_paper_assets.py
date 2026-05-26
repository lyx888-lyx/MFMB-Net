#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('results/auto_anchor_runs_rta_pred_residual_final')

CORE_METHODS = ['text','audio','vision','dynamic_soft','dynamic_task_soft_tuned','dynamic_rta_pred_residual']
MAIN_RENAME = {
    'text':'Text center',
    'audio':'Audio center',
    'vision':'Vision center',
    'dynamic_soft':'Dynamic Soft',
    'dynamic_task_soft_tuned':'Task-aware Dynamic',
    'dynamic_rta_pred_residual':'Ours',
}
ALL_METRICS = ['MAE','Corr','Non0_acc_2','Non0_F1_score','Mult_acc_5','Mult_acc_7']
CORE_METRICS = ['MAE','Corr','Non0_acc_2','Non0_F1_score']


def guard_no_normals(paths):
    for p in paths:
        s = str(p)
        if 'results/results/normals' in s.replace('\\\\','/'):
            raise RuntimeError('WARNING: normals CSV does not contain enough metadata and must not be used for paper tables.')


def fmt4(x):
    if pd.isna(x):
        return 'NaN'
    return f"{float(x):.4f}"


def mark_best_second(df, group_col, metrics, lower_is_better=('MAE',)):
    out = df.copy()
    for g, idxs in out.groupby(group_col).groups.items():
        sub = out.loc[list(idxs)]
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors='coerce')
            if vals.notna().sum() == 0:
                continue
            asc = m in lower_is_better
            ord_idx = vals.sort_values(ascending=asc).index.tolist()
            # initialize plain strings
            for i in ord_idx:
                out.loc[i, m] = fmt4(vals.loc[i])
            if len(ord_idx) >= 1:
                i = ord_idx[0]
                out.loc[i, m] = f"**{fmt4(vals.loc[i])}**"
            if len(ord_idx) >= 2:
                i = ord_idx[1]
                out.loc[i, m] = f"_{fmt4(vals.loc[i])}_"
    return out


def md_to_tex_cell(s):
    s = str(s)
    if s.startswith('**') and s.endswith('**'):
        return f"\\textbf{{{s[2:-2]}}}"
    if s.startswith('_') and s.endswith('_'):
        return f"\\underline{{{s[1:-1]}}}"
    return s


def write_md_table(df, path, cols):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|' + '|'.join(['---']*len(cols)) + '|\n')
        for _, r in df.iterrows():
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')


def write_tex_table(df, path, cols, caption, label, align=None):
    if align is None:
        align = 'l' * len(cols)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\\begin{table*}[t]\n\\centering\n\\small\n')
        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write('\\toprule\n')
        f.write(' & '.join(cols) + ' \\\\ \n')
        f.write('\\midrule\n')
        for _, r in df.iterrows():
            vals = [md_to_tex_cell(r[c]) for c in cols]
            f.write(' & '.join(vals) + ' \\\\ \n')
        f.write('\\bottomrule\n\\end{tabular}\n')
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write('\\end{table*}\n')


def compute_auilc(comp):
    methods = CORE_METHODS
    rows = []
    xs = np.array([0.1,0.2,0.3,0.4,0.5])
    for m in methods:
        d = comp[comp['method']==m].sort_values('missing')
        d = d[d['missing'].isin(xs)]
        if len(d) != 5:
            continue
        row = {'method': m}
        for met in ALL_METRICS:
            y = d[met].values.astype(float)
            row[f'{met}_AUILC'] = float(np.trapz(y, xs))
            row[f'{met}_avg'] = float(np.mean(y))
        rows.append(row)
    return pd.DataFrame(rows)


def make_auilc_outputs(comp):
    a = compute_auilc(comp)
    a['method'] = a['method'].astype(str)
    a['method'] = pd.Categorical(a['method'], categories=CORE_METHODS, ordered=True)
    a = a.sort_values('method').reset_index(drop=True)
    a.to_csv(ROOT/'residual_auilc_01_05.csv', index=False, float_format='%.6f')

    view = a[['method','MAE_AUILC','Corr_AUILC','Non0_acc_2_AUILC','Non0_F1_score_AUILC','Mult_acc_5_AUILC','Mult_acc_7_AUILC','MAE_avg','Corr_avg','Non0_acc_2_avg','Non0_F1_score_avg']].copy()
    # mark best/second per column globally
    for c in view.columns[1:]:
        vals = pd.to_numeric(view[c], errors='coerce')
        asc = c.startswith('MAE')
        ord_idx = vals.sort_values(ascending=asc).index.tolist()
        for i in ord_idx:
            view.loc[i, c] = fmt4(vals.loc[i])
        if len(ord_idx)>=1:
            i=ord_idx[0]; view.loc[i,c]=f"**{fmt4(vals.loc[i])}**"
        if len(ord_idx)>=2:
            i=ord_idx[1]; view.loc[i,c]=f"_{fmt4(vals.loc[i])}_"
    view['method'] = view['method'].astype(str).map(MAIN_RENAME).fillna(view['method'].astype(str))

    md_cols = list(view.columns)
    write_md_table(view, ROOT/'residual_auilc_01_05.md', md_cols)

    caption = ('AUILC over missing rates 0.1--0.5 under the full-test protocol. '
               'Lower is better for MAE-AUILC, while higher is better for correlation and accuracy/F1 AUILC.')
    write_tex_table(view, ROOT/'residual_auilc_01_05.tex', md_cols, caption, 'tab:residual_auilc_01_05', align='l'+'c'*(len(md_cols)-1))


def make_compact_main(comp):
    d = comp[comp['method'].isin(CORE_METHODS)].copy()
    d['method'] = d['method'].map(MAIN_RENAME)
    method_order = [MAIN_RENAME[m] for m in CORE_METHODS]
    d['method'] = pd.Categorical(d['method'], categories=method_order, ordered=True)
    d = d[['missing','method']+CORE_METRICS].sort_values(['missing','method'])
    d_mark = mark_best_second(d, 'missing', CORE_METRICS, lower_is_better=('MAE',))

    write_md_table(d_mark, ROOT/'residual_main_table_compact.md', ['missing','method']+CORE_METRICS)

    caption = ('Full-test protocol: train_drop_last=1, eval_drop_last=0, test_drop_last=0. '
               'The MOSI test set is evaluated on all 686 samples.')
    write_tex_table(d_mark, ROOT/'residual_main_table_compact.tex', ['missing','method']+CORE_METRICS, caption, 'tab:residual_main_compact', align='c l c c c c')


def make_average_summary(avgm, avgr):
    keep = pd.DataFrame({'method': CORE_METHODS})
    s = keep.merge(avgm, on='method', how='left').merge(avgr[['method','avg_rank_core','avg_rank_all']], on='method', how='left')
    s['method'] = s['method'].map(MAIN_RENAME)
    s = s.rename(columns={
        'Non0_F1_avg':'Non0_F1_avg',
        'avg_rank_core':'AvgRank_core',
        'avg_rank_all':'AvgRank_all'
    })
    cols = ['method','MAE_avg','Corr_avg','Non0_acc_2_avg','Non0_F1_avg','AvgRank_core','AvgRank_all']
    v = s[cols].copy()
    method_order = [MAIN_RENAME[m] for m in CORE_METHODS]
    v['method'] = pd.Categorical(v['method'], categories=method_order, ordered=True)
    v = v.sort_values('method').reset_index(drop=True)
    for c in cols[1:]:
        vals = pd.to_numeric(v[c], errors='coerce')
        asc = c in ['MAE_avg','AvgRank_core','AvgRank_all']
        ord_idx = vals.sort_values(ascending=asc).index.tolist()
        for i in ord_idx:
            v.loc[i,c]=fmt4(vals.loc[i])
        if len(ord_idx)>=1:
            i=ord_idx[0]; v.loc[i,c]=f"**{fmt4(vals.loc[i])}**"
        if len(ord_idx)>=2:
            i=ord_idx[1]; v.loc[i,c]=f"_{fmt4(vals.loc[i])}_"

    write_md_table(v, ROOT/'residual_average_summary.md', cols)
    caption = 'Average metrics and average rank over missing rates 0.1--0.5 under the full-test protocol.'
    write_tex_table(v, ROOT/'residual_average_summary.tex', cols, caption, 'tab:residual_avg_summary', align='l c c c c c c')

    # append one-line conclusion
    best_mae_method = s.loc[s['MAE_avg'].idxmin(),'method'] if s['MAE_avg'].notna().any() else 'N/A'
    best_rank_method = s.loc[s['AvgRank_core'].idxmin(),'method'] if s['AvgRank_core'].notna().any() else 'N/A'
    with open(ROOT/'residual_average_summary.md','a',encoding='utf-8') as f:
        f.write('\n\n')
        f.write(f"Conclusion: {best_mae_method} obtains the best average MAE, while {best_rank_method} remains competitive in average rank.\n")


def make_ablation_compact(comp):
    # A) full sweep
    m_full = ['dynamic_soft','dynamic_task_soft_tuned','dynamic_rta_feature','dynamic_rta_pred_residual']
    df_full = comp[comp['method'].isin(m_full)].copy()
    df_full['method'] = pd.Categorical(df_full['method'], categories=m_full, ordered=True)
    df_full = df_full[['missing','method','MAE','Corr','Non0_F1_score']].sort_values(['missing','method'])
    df_full_mark = mark_best_second(df_full, 'missing', ['MAE','Corr','Non0_F1_score'], lower_is_better=('MAE',))

    # B) key point 0.4/0.5
    m_key = ['dynamic_soft','dynamic_soft_moe','dynamic_task_soft_tuned','dynamic_rta_feature','dynamic_rta_pred','dynamic_rta_pred_residual']
    df_key = comp[(comp['method'].isin(m_key)) & (comp['missing'].isin([0.4,0.5]))].copy()
    df_key['method'] = pd.Categorical(df_key['method'], categories=m_key, ordered=True)
    df_key = df_key[['missing','method','MAE','Corr','Non0_F1_score']].sort_values(['missing','method'])
    df_key_mark = mark_best_second(df_key, 'missing', ['MAE','Corr','Non0_F1_score'], lower_is_better=('MAE',))

    with open(ROOT/'residual_ablation_compact.md','w',encoding='utf-8') as f:
        f.write('## A. Full-sweep Ablation (0.1~0.5)\n\n')
    write_md_table(df_full_mark, ROOT/'residual_ablation_compact.md.tmp', ['missing','method','MAE','Corr','Non0_F1_score'])
    with open(ROOT/'residual_ablation_compact.md','a',encoding='utf-8') as f:
        f.write((ROOT/'residual_ablation_compact.md.tmp').read_text(encoding='utf-8'))
        f.write('\n\n## B. Key-point Ablation (0.4, 0.5)\n\n')
    write_md_table(df_key_mark, ROOT/'residual_ablation_compact_key.md.tmp', ['missing','method','MAE','Corr','Non0_F1_score'])
    with open(ROOT/'residual_ablation_compact.md','a',encoding='utf-8') as f:
        f.write((ROOT/'residual_ablation_compact_key.md.tmp').read_text(encoding='utf-8'))
    (ROOT/'residual_ablation_compact.md.tmp').unlink(missing_ok=True)
    (ROOT/'residual_ablation_compact_key.md.tmp').unlink(missing_ok=True)

    with open(ROOT/'residual_ablation_compact.tex','w',encoding='utf-8') as f:
        f.write('\\begin{table*}[t]\n\\centering\n\\small\n')
        f.write('\\begin{tabular}{c l c c c}\n\\toprule\n')
        f.write('Missing & Method & MAE$\\downarrow$ & Corr$\\uparrow$ & Non0 F1$\\uparrow$ \\\\ \n\\midrule\n')
        for _,r in df_full_mark.iterrows():
            f.write(f"{r['missing']} & {r['method']} & {md_to_tex_cell(r['MAE'])} & {md_to_tex_cell(r['Corr'])} & {md_to_tex_cell(r['Non0_F1_score'])} \\\\ \n")
        f.write('\\midrule\n')
        f.write('\\multicolumn{5}{c}{Key-point Ablation (0.4, 0.5)} \\\\ \n\\midrule\n')
        for _,r in df_key_mark.iterrows():
            f.write(f"{r['missing']} & {r['method']} & {md_to_tex_cell(r['MAE'])} & {md_to_tex_cell(r['Corr'])} & {md_to_tex_cell(r['Non0_F1_score'])} \\\\ \n")
        f.write('\\bottomrule\n\\end{tabular}\n')
        f.write('\\caption{Compact ablation: full-sweep and key-point comparisons.}\\label{tab:residual_ablation_compact}\n')
        f.write('\\end{table*}\n')


def make_caption_and_text():
    cap = ROOT/'residual_paper_captions.md'
    cap.write_text(
"""# Figure Captions

1. `residual_mae_vs_missing.png`
- EN: MAE trends over missing rates (0.1–0.5) under the full-test protocol, comparing fixed-center, dynamic baselines, and the residual prediction-level RTA.
- 中文：展示 0.1~0.5 缺失率下 MAE 曲线，对比固定中心、动态基线与残差预测级 RTA。

2. `residual_corr_vs_missing.png`
- EN: Correlation trends across missing rates, showing robustness differences among routing strategies.
- 中文：展示相关系数随缺失率变化，体现不同路由策略的鲁棒性差异。

3. `residual_non0_f1_vs_missing.png`
- EN: Non0 F1-score comparison over missing rates, highlighting classification-oriented behavior under missing modalities.
- 中文：展示 Non0 F1 在不同缺失率下的变化，反映缺失条件下分类相关性能。

4. `residual_gate_weight_vs_missing.png`
- EN: Mean gate allocation (`g_task` vs `g_rel`) across missing rates for residual prediction-level RTA.
- 中文：展示残差预测级 RTA 的平均门控分配（`g_task` 与 `g_rel`）随缺失率变化。

5. `residual_average_rank.png`
- EN: Average rank across core metrics, indicating relative overall stability rather than single-point superiority.
- 中文：展示核心指标平均排名，强调整体稳定性而非单点绝对最优。

6. `residual_delta_mae_vs_dynamic_soft.png`
- EN: MAE delta of residual RTA against dynamic_soft; values below zero indicate improvement.
- 中文：展示残差 RTA 相对 dynamic_soft 的 MAE 差值，低于 0 表示提升。
""", encoding='utf-8')

    zh = ROOT/'residual_experiment_text_zh.md'
    zh.write_text(
"""在统一 full-test 协议（train_drop_last=1, eval_drop_last=0, test_drop_last=0, MOSI test=686）下，我们对比了固定中心、Dynamic Soft、Task-aware Dynamic 以及提出的残差预测级 RTA。结果表明，Ours 在 0.1~0.5 区间取得了最低的平均 MAE，并在高缺失区间对 Dynamic Soft 的退化表现出更好的修复能力。与此同时，fixed vision 在多个缺失点上的排名仍然具有竞争力，使得 Ours 并非在所有缺失率和所有指标上都绝对最优。

这说明残差预测级门控在“稳定主干 + 任务修正”方向上是有效的：它避免了 feature-level 直接混合的不稳定，并在高缺失样本上引入更有针对性的校正。然而，我们也观察到 average rank 上 fixed vision 仍然较强，提示当前方法的优势主要体现在平均误差控制与高缺失鲁棒性，而不是全指标统治。作为对照，MoE 在本任务设置下波动较大，适合作为消融参考而非主线方案。""", encoding='utf-8')

    en = ROOT/'residual_experiment_text_en.md'
    en.write_text(
"""Under the unified full-test protocol (train_drop_last=1, eval_drop_last=0, test_drop_last=0, all 686 MOSI test samples), we compared fixed-center baselines, Dynamic Soft, Task-aware Dynamic, and the proposed residual prediction-level RTA. The results show that Ours achieves the best average MAE over missing rates 0.1–0.5 and demonstrates improved robustness in high-missing regimes where Dynamic Soft degrades.

At the same time, fixed vision remains competitive in average rank across several settings, indicating that Ours is not uniformly optimal on every metric at every missing level. These findings suggest that residual prediction-level gating is a practical compromise: it preserves a reliable backbone while applying task-aware correction when beneficial. The evidence supports the feasibility and potential of this direction, while also indicating room for further refinement. In contrast, the MoE variant remains less stable in this setup and is better treated as an ablation baseline rather than the primary solution.""", encoding='utf-8')


def make_stress_plan_and_cmds():
    plan = ROOT/'stress_test_plan_06_10.md'
    plan.write_text(
"""# Stress Test Plan (0.6~1.0)

This is an extreme-missing stress test plan and is **not** a replacement for the main 0.1~0.5 results.

## Goal
Evaluate whether Ours (`dynamic_rta_pred_residual`) remains more stable than `dynamic_soft` under very high missing rates.

## Scope
- Missing rates: 0.6, 0.7, 0.8, 0.9, 1.0
- Methods: `text`, `vision`, `dynamic_soft`, `dynamic_rta_pred_residual`
- Protocol: train_drop_last=1, eval_drop_last=0, test_drop_last=0
- Output dir: `results/auto_anchor_runs_stress_06_10/`

## Notes
- Results should be audited with manifest/protocol tags.
- If Ours underperforms at 0.6~1.0, this does not invalidate 0.1~0.5 conclusions, but should be reported as a boundary condition.
- Planned post-processing outputs:
  - `stress_06_10_comparison.csv`
  - `stress_06_10_auilc.csv`
  - `stress_06_10_report.md`
""", encoding='utf-8')

    sh = ROOT/'stress_test_commands_06_10.sh'
    sh.write_text(
"""#!/usr/bin/env bash
set -e

OUT=results/auto_anchor_runs_stress_06_10
DATASET=mosi
MISSING=0.6,0.7,0.8,0.9,1.0

# text
conda run -n mfmb python scripts/auto_run_anchor_experiments.py \
  --datasetName ${DATASET} --phase full --missing_list ${MISSING} --modes text \
  --gpu_ids 0 --resume 1 \
  --train_drop_last 1 --eval_drop_last 0 --test_drop_last 0 \
  --output_dir ${OUT}

# vision
conda run -n mfmb python scripts/auto_run_anchor_experiments.py \
  --datasetName ${DATASET} --phase full --missing_list ${MISSING} --modes vision \
  --gpu_ids 0 --resume 1 \
  --train_drop_last 1 --eval_drop_last 0 --test_drop_last 0 \
  --output_dir ${OUT}

# dynamic_soft
conda run -n mfmb python scripts/auto_run_anchor_experiments.py \
  --datasetName ${DATASET} --phase full --missing_list ${MISSING} --modes dynamic_soft \
  --export_anchor_weights 1 --router_missing_bias 2.0 \
  --gpu_ids 0 --resume 1 \
  --train_drop_last 1 --eval_drop_last 0 --test_drop_last 0 \
  --output_dir ${OUT}

# Ours: dynamic_rta_pred_residual
conda run -n mfmb python scripts/auto_run_anchor_experiments.py \
  --datasetName ${DATASET} --phase full --missing_list ${MISSING} --modes dynamic_rta_pred_residual \
  --export_anchor_weights 1 --export_task_router_info 1 --router_missing_bias 2.0 \
  --use_reliability_task_gate 1 --use_prediction_gate_supervision 1 --rta_pred_residual 1 \
  --gate_supervision_mode margin --gate_margin 0.05 \
  --task_router_lambda 0.1 --center_aux_lambda 0.05 \
  --router_oracle_temperature 0.8 --router_oracle_type soft \
  --gate_oracle_temperature 0.8 --gate_task_lambda 0.02 \
  --gate_balance_lambda 0.0 --gate_target 0.5 --gate_init_bias -2.0 \
  --gate_hidden_dim 32 --gate_dropout 0.1 \
  --gpu_ids 0 --resume 1 \
  --train_drop_last 1 --eval_drop_last 0 --test_drop_last 0 \
  --output_dir ${OUT}

# post-process (planned)
# conda run -n mfmb python scripts/generate_stress_outputs.py --input_dir ${OUT}
""", encoding='utf-8')
    sh.chmod(0o755)


def make_layout():
    (ROOT/'paper_experiment_layout.md').write_text(
"""# Suggested Paper Experiment Layout

## Table 1
Complete modality comparison at missing=0.0.
- If `Ours@0.0` is not available, mark it as pending and avoid forcing it into final claims.

## Table 2
Main comparison under missing=0.1~0.5.
- Methods: Text/Audio/Vision, Dynamic Soft, Task-aware Dynamic, Ours.

## Table 3
AUILC@0.1~0.5 + average metrics + average rank.
- Emphasize average robustness and ranking stability.

## Table 4
Ablation at key missing rates (0.4/0.5).
- Include dynamic_soft, dynamic_soft_moe, task-aware, feature-RTA, pred-RTA, residual pred-RTA.

## Figure 1
Metric curves over missing=0.1~0.5 (MAE/Corr/Non0-F1 etc.).

## Optional Table 5
Extreme missing stress test 0.6~1.0.

## Important rule
Do **not** report AUILC over 0.0~1.0 unless Ours is fully evaluated at both missing=0.0 and 0.6~1.0.
""", encoding='utf-8')


def main():
    input_paths = [
        ROOT/'residual_final_comparison.csv',
        ROOT/'residual_average_metrics.csv',
        ROOT/'residual_average_rank.csv',
        ROOT/'residual_gate_diagnostics.csv',
        ROOT/'residual_main_table.md',
        ROOT/'residual_main_table.tex',
        ROOT/'residual_ablation_table.md',
        ROOT/'residual_ablation_table.tex',
    ]
    guard_no_normals(input_paths)
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    comp = pd.read_csv(ROOT/'residual_final_comparison.csv')
    avgm = pd.read_csv(ROOT/'residual_average_metrics.csv')
    avgr = pd.read_csv(ROOT/'residual_average_rank.csv')

    # restrict expected missing and methods
    comp = comp[comp['missing'].isin([0.1,0.2,0.3,0.4,0.5])].copy()

    make_auilc_outputs(comp)
    make_compact_main(comp)
    make_average_summary(avgm, avgr)
    make_ablation_compact(comp)
    make_caption_and_text()
    make_stress_plan_and_cmds()
    make_layout()
    print('[DONE] residual paper assets generated.')


if __name__ == '__main__':
    main()
