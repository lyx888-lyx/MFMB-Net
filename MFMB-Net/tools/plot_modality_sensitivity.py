#!/usr/bin/env python3
import argparse
import ast
import glob
import math
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [
    'Has0_acc_2',
    'Has0_F1_score',
    'Non0_acc_2',
    'Non0_F1_score',
    'Mult_acc_5',
    'Mult_acc_7',
    'MAE',
    'Corr',
    'Loss',
]
DIV100_METRICS = {'MAE', 'Corr', 'Loss'}
HIGHER_BETTER = {
    'Has0_acc_2': True,
    'Has0_F1_score': True,
    'Non0_acc_2': True,
    'Non0_F1_score': True,
    'Mult_acc_5': True,
    'Mult_acc_7': True,
    'Corr': True,
    'MAE': False,
    'Loss': False,
}
MODALITIES = ['text', 'audio', 'vision']
FILE_RE = re.compile(r'(?P<dataset>[^/]+)-(?P<mode>[^/]+)-t(?P<t>[0-9.]+)_a(?P<a>[0-9.]+)_v(?P<v>[0-9.]+)(?:-(?P<tag>.+))?\.csv$')


def parse_tuple_cell(cell):
    if pd.isna(cell):
        return float('nan'), float('nan')
    if isinstance(cell, (int, float)):
        return float(cell), float('nan')
    text = str(cell).strip()
    if text.startswith('(') and text.endswith(')'):
        try:
            mean, std = ast.literal_eval(text)
            return float(mean), float(std)
        except Exception:
            pass
    return float(text), float('nan')


def maybe_scale(metric, value):
    if metric in DIV100_METRICS and not pd.isna(value):
        return value / 100.0
    return value


def load_results(pattern: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        m = FILE_RE.match(name)
        if not m:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = {
            'path': path,
            'dataset': m.group('dataset'),
            'train_mode': m.group('mode'),
            'missing_t': float(m.group('t')),
            'missing_a': float(m.group('a')),
            'missing_v': float(m.group('v')),
            'run_tag': m.group('tag') or '',
        }
        last = df.iloc[-1].to_dict()
        for metric in METRICS:
            if metric in last:
                mean, std = parse_tuple_cell(last[metric])
                row[metric] = maybe_scale(metric, mean)
                row[f'{metric}_std'] = maybe_scale(metric, std)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f'No result CSV matched pattern: {pattern}')
    return out.sort_values(['missing_t', 'missing_a', 'missing_v']).reset_index(drop=True)


def area_under_curve(xs, ys):
    if len(xs) < 2:
        return float('nan')
    integrator = getattr(np, 'trapezoid', np.trapz)
    return float(integrator(ys, xs) / (xs[-1] - xs[0]))


def compute_single_modality_scores(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    baseline_row = df[(df['missing_t'] == 0) & (df['missing_a'] == 0) & (df['missing_v'] == 0)]
    if baseline_row.empty:
        raise ValueError('Baseline result t0_a0_v0 is required for sensitivity ranking.')
    baseline = float(baseline_row.iloc[0][metric])
    higher_is_better = HIGHER_BETTER[metric]
    rows = []
    for modality in MODALITIES:
        if modality == 'text':
            sub = df[(df['missing_a'] == 0) & (df['missing_v'] == 0)].copy().sort_values('missing_t')
            xs = sub['missing_t'].to_numpy()
        elif modality == 'audio':
            sub = df[(df['missing_t'] == 0) & (df['missing_v'] == 0)].copy().sort_values('missing_a')
            xs = sub['missing_a'].to_numpy()
        else:
            sub = df[(df['missing_t'] == 0) & (df['missing_a'] == 0)].copy().sort_values('missing_v')
            xs = sub['missing_v'].to_numpy()
        ys = sub[metric].to_numpy(dtype=float)
        keep = ~np.isnan(xs) & ~np.isnan(ys)
        xs = xs[keep]
        ys = ys[keep]
        if len(xs) == 0:
            continue
        if higher_is_better:
            drops = baseline - ys
            drop_at_100 = baseline - ys[np.argmax(xs)]
        else:
            drops = ys - baseline
            drop_at_100 = ys[np.argmax(xs)] - baseline
        avg_drop = area_under_curve(xs, drops)
        rows.append({
            'modality': modality,
            'metric': metric,
            'baseline': baseline,
            'drop_at_max_missing': float(drop_at_100),
            'avg_drop_auc': float(avg_drop),
            'sensitivity_score': float(avg_drop),
        })
    out = pd.DataFrame(rows).sort_values('sensitivity_score', ascending=False).reset_index(drop=True)
    return out


def plot_single_metric(df: pd.DataFrame, metric: str, out_path: str):
    plt.figure(figsize=(7, 5))
    for modality, xcol, filters in [
        ('text', 'missing_t', (df['missing_a'] == 0) & (df['missing_v'] == 0)),
        ('audio', 'missing_a', (df['missing_t'] == 0) & (df['missing_v'] == 0)),
        ('vision', 'missing_v', (df['missing_t'] == 0) & (df['missing_a'] == 0)),
    ]:
        sub = df[filters].sort_values(xcol)
        plt.plot(sub[xcol], sub[metric], marker='o', label=modality)
    plt.xlabel('Missing rate')
    plt.ylabel(metric)
    plt.title(f'Single-modality sensitivity ({metric})')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(df: pd.DataFrame, fixed_modality: str, metric: str, out_path: str):
    if fixed_modality == 'text':
        sub = df[df['missing_t'] == 0].copy()
        xcol, ycol = 'missing_a', 'missing_v'
        xlabel, ylabel = 'audio missing', 'vision missing'
    elif fixed_modality == 'audio':
        sub = df[df['missing_a'] == 0].copy()
        xcol, ycol = 'missing_t', 'missing_v'
        xlabel, ylabel = 'text missing', 'vision missing'
    else:
        sub = df[df['missing_v'] == 0].copy()
        xcol, ycol = 'missing_t', 'missing_a'
        xlabel, ylabel = 'text missing', 'audio missing'

    pivot = sub.pivot_table(index=ycol, columns=xcol, values=metric, aggfunc='first')
    pivot = pivot.sort_index().sort_index(axis=1)
    if pivot.empty:
        return

    plt.figure(figsize=(6, 5))
    im = plt.imshow(pivot.values, aspect='auto', origin='lower')
    plt.colorbar(im, label=metric)
    plt.xticks(range(len(pivot.columns)), [f'{v:.2f}' for v in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [f'{v:.2f}' for v in pivot.index])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{metric} heatmap (keep {fixed_modality} intact)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot modality sensitivity results and rank which modality is most critical.')
    parser.add_argument('--pattern', required=True, help='Glob pattern for result CSVs.')
    parser.add_argument('--out_dir', default='analysis/modality_sensitivity', help='Directory to save figures and summary CSVs.')
    parser.add_argument('--ranking_metric', default='Corr', choices=METRICS, help='Primary metric used for the final ranking.')
    parser.add_argument('--plot_metrics', default='Corr,MAE,Non0_F1_score', help='Comma-separated metrics to draw line plots for.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_results(args.pattern)
    df.to_csv(os.path.join(args.out_dir, 'all_conditions_summary.csv'), index=False)

    ranking = compute_single_modality_scores(df, args.ranking_metric)
    ranking.to_csv(os.path.join(args.out_dir, f'modality_ranking_{args.ranking_metric}.csv'), index=False)

    plot_metrics = [m.strip() for m in args.plot_metrics.split(',') if m.strip()]
    for metric in plot_metrics:
        plot_single_metric(df, metric, os.path.join(args.out_dir, f'single_modality_{metric}.png'))
        for fixed_modality in MODALITIES:
            plot_heatmap(df, fixed_modality, metric, os.path.join(args.out_dir, f'heatmap_{metric}_keep_{fixed_modality}.png'))

    with open(os.path.join(args.out_dir, 'ranking_report.txt'), 'w', encoding='utf-8') as f:
        if ranking.empty:
            f.write('No valid single-modality results were found.\n')
        else:
            f.write(f'Primary ranking metric: {args.ranking_metric}\n')
            f.write(ranking.to_string(index=False))
            f.write('\n\nInterpretation:\n')
            top = ranking.iloc[0]
            f.write(f"Most critical modality under {args.ranking_metric}: {top['modality']}\n")
            f.write('Higher sensitivity_score means performance deteriorates faster when that modality is removed.\n')

    print(f'[OK] saved analysis to: {args.out_dir}')
    if not ranking.empty:
        print('\nTop ranking:')
        print(ranking.to_string(index=False))


if __name__ == '__main__':
    main()
