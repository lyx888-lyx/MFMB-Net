#!/usr/bin/env python3
import argparse
import ast
import glob
import os
import re
from typing import Dict, List, Tuple

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
FILE_RE = re.compile(
    r'(?P<dataset>[^/]+)-(?P<mode>[^/]+)-t(?P<t>[0-9.]+)_a(?P<a>[0-9.]+)_v(?P<v>[0-9.]+)(?:-(?P<tag>.+))?\.csv$'
)


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

def plot_fixed_value_heatmap(df: pd.DataFrame, fixed_modality: str, fixed_value: float, metric: str, out_path: str):
    if fixed_modality == 'text':
        sub = df[np.isclose(df['missing_t'], fixed_value)].copy()
        xcol, ycol = 'missing_a', 'missing_v'
        xlabel, ylabel = 'audio missing', 'vision missing'
    elif fixed_modality == 'audio':
        sub = df[np.isclose(df['missing_a'], fixed_value)].copy()
        xcol, ycol = 'missing_t', 'missing_v'
        xlabel, ylabel = 'text missing', 'vision missing'
    else:
        sub = df[np.isclose(df['missing_v'], fixed_value)].copy()
        xcol, ycol = 'missing_t', 'missing_a'
        xlabel, ylabel = 'text missing', 'audio missing'

    pivot = sub.pivot_table(index=ycol, columns=xcol, values=metric, aggfunc='mean')
    pivot = pivot.sort_index().sort_index(axis=1)
    if pivot.empty:
        print(f'[WARN] empty fixed-value heatmap: {fixed_modality}={fixed_value}, metric={metric}')
        return

    plt.figure(figsize=(6, 5))
    im = plt.imshow(pivot.values, aspect='auto', origin='lower')
    plt.colorbar(im, label=metric)
    plt.xticks(range(len(pivot.columns)), [f'{v:.2f}' for v in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [f'{v:.2f}' for v in pivot.index])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{metric} heatmap ({fixed_modality}={fixed_value:.2f})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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
    if len(xs) < 2 or xs[-1] == xs[0]:
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
        if len(xs) < 2:
            continue
        max_idx = np.argmax(xs)
        if higher_is_better:
            drops = baseline - ys
            drop_at_100 = baseline - ys[max_idx]
        else:
            drops = ys - baseline
            drop_at_100 = ys[max_idx] - baseline
        avg_drop = area_under_curve(xs, drops)
        rows.append({
            'modality': modality,
            'metric': metric,
            'baseline': baseline,
            'drop_at_max_missing': float(drop_at_100),
            'avg_drop_auc': float(avg_drop),
            'sensitivity_score': float(avg_drop),
        })
    if not rows:
        return pd.DataFrame(columns=['modality', 'metric', 'baseline', 'drop_at_max_missing', 'avg_drop_auc', 'sensitivity_score'])
    out = pd.DataFrame(rows).sort_values('sensitivity_score', ascending=False).reset_index(drop=True)
    return out


def plot_single_metric(df: pd.DataFrame, metric: str, out_path: str):
    plt.figure(figsize=(7, 5))
    configs = [
        ('text', 'missing_t', (df['missing_a'] == 0) & (df['missing_v'] == 0)),
        ('audio', 'missing_a', (df['missing_t'] == 0) & (df['missing_v'] == 0)),
        ('vision', 'missing_v', (df['missing_t'] == 0) & (df['missing_a'] == 0)),
    ]
    plotted = False
    for modality, xcol, filters in configs:
        sub = df[filters].sort_values(xcol)[[xcol, metric]].dropna()
        sub = sub.drop_duplicates(subset=[xcol], keep='first')
        if len(sub) < 2:
            print(f'[WARN] skip {modality} in single-modality plot: only {len(sub)} point(s)')
            continue
        plt.plot(sub[xcol], sub[metric], marker='o', label=modality)
        plotted = True
    if not plotted:
        plt.close()
        return
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

    pivot = sub.pivot_table(index=ycol, columns=xcol, values=metric, aggfunc='mean')
    pivot = pivot.sort_index().sort_index(axis=1)
    if pivot.empty:
        return
    if len(pivot.columns) < 2 or len(pivot.index) < 2:
        print(f'[WARN] Degenerate heatmap for keep_{fixed_modality}: shape={pivot.shape}. This usually means the pattern only matched one experiment subset.')

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


def _slice_config(fixed_modality: str) -> Tuple[pd.DataFrame, str, str, str, str]:
    raise RuntimeError('not used directly')


def plot_slice_lines(df: pd.DataFrame, fixed_modality: str, metric: str, out_dir: str):
    if fixed_modality == 'text':
        sub = df[df['missing_t'] == 0].copy()
        var1, var2 = 'missing_a', 'missing_v'
        label1, label2 = 'audio missing', 'vision missing'
    elif fixed_modality == 'audio':
        sub = df[df['missing_a'] == 0].copy()
        var1, var2 = 'missing_t', 'missing_v'
        label1, label2 = 'text missing', 'vision missing'
    else:
        sub = df[df['missing_v'] == 0].copy()
        var1, var2 = 'missing_t', 'missing_a'
        label1, label2 = 'text missing', 'audio missing'

    if sub.empty:
        return

    # Plot var1 on x-axis, one line per fixed var2
    def _make_plot(xcol: str, linecol: str, xlabel: str, line_label: str, filename_suffix: str):
        plot_df = sub[[xcol, linecol, metric]].dropna().copy()
        if plot_df.empty:
            return
        unique_x = sorted(plot_df[xcol].unique())
        unique_line = sorted(plot_df[linecol].unique())
        if len(unique_x) < 2 or len(unique_line) < 2:
            return

        plt.figure(figsize=(7, 5))
        plotted = False
        for lv in unique_line:
            cur = plot_df[plot_df[linecol] == lv].sort_values(xcol)
            cur = cur.drop_duplicates(subset=[xcol], keep='first')
            if len(cur) < 2:
                continue
            plt.plot(cur[xcol], cur[metric], marker='o', label=f'{line_label}={lv:.2f}')
            plotted = True
        if not plotted:
            plt.close()
            return
        plt.xlabel(xlabel)
        plt.ylabel(metric)
        plt.title(f'{metric} slices (keep {fixed_modality} intact)')
        plt.xlim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'slice_{metric}_keep_{fixed_modality}_{filename_suffix}.png'), dpi=200)
        plt.close()

    _make_plot(var1, var2, label1, label2, 'xvar1_linesvar2')
    _make_plot(var2, var1, label2, label1, 'xvar2_linesvar1')



def main():
    parser = argparse.ArgumentParser(description='Plot modality sensitivity results, heatmaps, and slice line plots.')
    parser.add_argument('--pattern', required=True, help='Glob pattern for result CSVs.')
    parser.add_argument('--out_dir', default='analysis/modality_sensitivity', help='Directory to save figures and summary CSVs.')
    parser.add_argument('--ranking_metric', default='Corr', choices=METRICS, help='Primary metric used for the final ranking.')
    parser.add_argument('--plot_metrics', default='Corr,MAE,Non0_F1_score', help='Comma-separated metrics to draw line plots for.')
    parser.add_argument('--make_slice_plots', action='store_true', help='Also draw slice line plots for keep_text / keep_audio / keep_vision grids.')
    parser.add_argument('--fixed_modality', type=str, default='', choices=['', 'text', 'audio', 'vision'],
                    help='plot a heatmap slice with one modality fixed to a specified value')
    parser.add_argument('--fixed_value', type=float, default=None,
                        help='the fixed missing rate value used with --fixed_modality')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_results(args.pattern)
    df.to_csv(os.path.join(args.out_dir, 'all_conditions_summary.csv'), index=False)

    ranking = compute_single_modality_scores(df, args.ranking_metric)
    ranking.to_csv(os.path.join(args.out_dir, f'modality_ranking_{args.ranking_metric}.csv'), index=False)

    plot_metrics = [m.strip() for m in args.plot_metrics.split(',') if m.strip()]
    for metric in plot_metrics:
        plot_single_metric(df, metric, os.path.join(args.out_dir, f'single_modality_{metric}.png'))

        if args.fixed_modality and args.fixed_value is not None:
            plot_fixed_value_heatmap(
                df,
                args.fixed_modality,
                args.fixed_value,
                metric,
                os.path.join(args.out_dir, f'heatmap_{metric}_{args.fixed_modality}{args.fixed_value:.2f}.png')
            )
        else:
            for fixed_modality in MODALITIES:
                plot_heatmap(df, fixed_modality, metric, os.path.join(args.out_dir, f'heatmap_{metric}_keep_{fixed_modality}.png'))

    with open(os.path.join(args.out_dir, 'ranking_report.txt'), 'w', encoding='utf-8') as f:
        if ranking.empty:
            f.write('No valid complete single-modality results were found.\n')
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
