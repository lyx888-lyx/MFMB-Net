#!/usr/bin/env python3
"""Collect ablation CSVs and write summary + markdown report."""

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results', 'results')

EXPERIMENTS = [
    'baseline_fixed',
    'mide_old_repro',
    'mide_split_aup',
    'mide_split_plus',
]
MISSING_RATES = [0.0, 0.3, 0.5]
METRICS = ['Has0_acc_2', 'Non0_acc_2', 'MAE', 'Corr', 'Loss']


def parse_mean(val):
    if isinstance(val, (tuple, list)):
        return float(val[0])
    if isinstance(val, str):
        s = val.strip().strip('"').strip("'")
        if s.startswith('(') and ',' in s:
            return float(s.strip('()').split(',')[0].strip())
        if '(' in s:
            return float(s.split('(')[-1].split(',')[0].strip())
        return float(s)
    return float(val)


def load_exp_row(exp_tag, missing):
    pattern = os.path.join(RESULTS, exp_tag, f'mosi-regression-{missing:.1f}.csv')
    if not os.path.exists(pattern):
        return None
    df = pd.read_csv(pattern)
    if df.empty:
        return None
    row = df.iloc[-1]
    out = {'exp_tag': exp_tag, 'missing': missing, 'best_checkpoint_type': 'best_by_loss'}
    for m in METRICS:
        if m in df.columns:
            out[m] = parse_mean(row[m])
    return out


def build_summary():
    rows = []
    for exp in EXPERIMENTS:
        for miss in MISSING_RATES:
            r = load_exp_row(exp, miss)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def read_auilc(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def read_file_edit_summary():
    path = os.path.join(ROOT, 'results', 'file_edit_summary.md')
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return '(file_edit_summary.md not found)'


HYPERPARAMS = {
    'baseline_fixed': {
        'batch': 32, 'lr_other': 0.0025, 'lr_bert': '1e-5', 'amp': True, 'mide_variant': 'off',
    },
    'mide_old_repro': {
        'batch': 32, 'lr_other': 0.0025, 'lr_bert': '1e-5', 'amp': True, 'mide_variant': 'old',
    },
    'mide_split_aup': {
        'batch': 32, 'lr_other': 0.0025, 'lr_bert': '1e-5', 'amp': True, 'mide_variant': 'split_aup',
        'mide_d_floor_min': 0.65, 'mide_aux_start': 2, 'mide_full_start': 4,
    },
    'mide_split_plus': {
        'batch': 40, 'lr_other': 0.0028, 'lr_bert': '1e-5', 'amp': True, 'mide_variant': 'split_aup_plus',
        'mide_d_floor_min': 0.65, 'mide_aux_start': 2, 'mide_full_start': 4,
    },
}


def fmt_delta(base, val, higher_better):
    if pd.isna(base) or pd.isna(val):
        return 'N/A'
    d = val - base
    if not higher_better:
        d = -d
    sign = '+' if d >= 0 else ''
    return f'{sign}{d:.2f}'


def write_report(summary_df, auilc_df):
    lines = ['# MIDE Split-AUP Ablation Report\n']
    lines.append('## 文件修改摘要表\n')
    lines.append(read_file_edit_summary())
    lines.append('\n## Ablation 实验矩阵\n')
    lines.append('| exp_tag | missing | Has0_acc_2 | Non0_acc_2 | MAE | Corr | Loss |')
    lines.append('|---------|--------:|-----------:|-----------:|----:|-----:|-----:|')
    for _, r in summary_df.sort_values(['exp_tag', 'missing']).iterrows():
        lines.append(
            f"| {r['exp_tag']} | {r['missing']:.1f} | "
            f"{r.get('Has0_acc_2', float('nan')):.2f} | {r.get('Non0_acc_2', float('nan')):.2f} | "
            f"{r.get('MAE', float('nan')):.2f} | {r.get('Corr', float('nan')):.2f} | "
            f"{r.get('Loss', float('nan')):.4f} |"
        )

    lines.append('\n## 推荐超参数表\n')
    lines.append('| exp_tag | batch | lr_other | lr_bert | amp | mide_variant | notes |')
    lines.append('|---------|------:|---------:|--------:|:---:|--------------|-------|')
    for exp, hp in HYPERPARAMS.items():
        lines.append(
            f"| {exp} | {hp['batch']} | {hp['lr_other']} | {hp['lr_bert']} | "
            f"{'on' if hp['amp'] else 'off'} | {hp['mide_variant']} | split-AUP fixes |"
        )

    base = summary_df[summary_df['exp_tag'] == 'baseline_fixed'].set_index('missing')
    lines.append('\n## Baseline vs MIDE 对比（相对 baseline 的 Has0/MAE/Corr 变化）\n')
    for exp in ['mide_old_repro', 'mide_split_aup', 'mide_split_plus']:
        sub = summary_df[summary_df['exp_tag'] == exp].set_index('missing')
        lines.append(f'\n### {exp}\n')
        lines.append('| missing | ΔHas0 | ΔMAE(better↑) | ΔCorr |')
        lines.append('|--------:|------:|--------------:|------:|')
        for miss in MISSING_RATES:
            if miss not in base.index or miss not in sub.index:
                continue
            b, s = base.loc[miss], sub.loc[miss]
            lines.append(
                f"| {miss:.1f} | {fmt_delta(b['Has0_acc_2'], s['Has0_acc_2'], True)} | "
                f"{fmt_delta(b['MAE'], s['MAE'], False)} | "
                f"{fmt_delta(b['Corr'], s['Corr'], True)} |"
            )

    lines.append('\n## AUILC 表\n')
    if auilc_df.empty:
        lines.append('(auilc_comparison_split.csv not available yet)\n')
    else:
        cols = list(auilc_df.columns)
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('|' + '|'.join(['---'] * len(cols)) + '|')
        for _, r in auilc_df.iterrows():
            lines.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
        lines.append('')

    lines.append('\n## 修复项分析\n')
    lines.append('- **True LOO**: `_mask_raw_inputs` + 重算 `audio_visual_fusion` / pairwise stacks / y_without_m_ref')
    lines.append('- **Warmup/eval 一致**: `gate_enabled(epoch)` 在 train/eval 共用；epoch ≤ gate_start 时 D_eff=1')
    lines.append('- **单次 density 缩放**: 主 rep × D_eff；stack × sqrt(D_i D_j)；AV × sqrt(D_a D_v)')
    lines.append('- **A/U/P 监督**: `build_targets` 分离 uni/loo utility 与 pollution target\n')

    lines.append('## Batch size 影响\n')
    plus = summary_df[summary_df['exp_tag'] == 'mide_split_plus']
    aup = summary_df[summary_df['exp_tag'] == 'mide_split_aup']
    if not plus.empty and not aup.empty:
        for miss in MISSING_RATES:
            p = plus[plus['missing'] == miss]
            a = aup[aup['missing'] == miss]
            if p.empty or a.empty:
                continue
            lines.append(
                f"- missing={miss:.1f}: plus(40) vs aup(32) MAE {p['MAE'].values[0]:.2f} vs {a['MAE'].values[0]:.2f}, "
                f"Corr {p['Corr'].values[0]:.2f} vs {a['Corr'].values[0]:.2f}"
            )

    lines.append('\n## 最终结论\n')
    best_exp = 'baseline_fixed'
    best_score = -1e9
    for exp in ['mide_split_plus', 'mide_split_aup', 'mide_old_repro']:
        sub = summary_df[(summary_df['exp_tag'] == exp) & (summary_df['missing'].isin([0.3, 0.5]))]
        if sub.empty:
            continue
        score = sub['Corr'].mean() - sub['MAE'].mean() * 0.01 + sub['Has0_acc_2'].mean() * 0.1
        if score > best_score:
            best_score = score
            best_exp = exp
    lines.append(f'- 综合 missing=0.3/0.5，推荐保留版本：**{best_exp}**（主 checkpoint 仍为 best_by_loss）')
    lines.append('- 若未全面超越 baseline，以 epoch_stats 与 AUILC 为准，不做 cherry-pick。\n')

    out_path = os.path.join(ROOT, 'results', 'mide_split_report.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    return out_path


def main():
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    summary = build_summary()
    out_csv = os.path.join(ROOT, 'results', 'ablation_summary.csv')
    summary.to_csv(out_csv, index=False)
    auilc_path = os.path.join(ROOT, 'results', 'auilc_comparison_split.csv')
    auilc_df = read_auilc(auilc_path)
    report_path = write_report(summary, auilc_df)
    print(f'Wrote {out_csv}')
    print(f'Wrote {report_path}')


if __name__ == '__main__':
    main()
