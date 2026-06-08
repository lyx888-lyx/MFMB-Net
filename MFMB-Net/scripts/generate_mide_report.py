#!/usr/bin/env python3
"""Generate results/mide_report.md from baseline and MIDE experiment outputs."""

import argparse
import glob
import os
import re
from datetime import datetime

import pandas as pd

from compare_auilc import load_csv_dir, compute_auilc, METRICS, HIGHER_BETTER


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


def load_results_table(csv_dir):
    df = load_csv_dir(csv_dir)
    if df.empty:
        return df
    for c in METRICS:
        if c in df.columns:
            df[c] = df[c].apply(parse_mean)
    return df


def summarize_modality_stats(log_path):
  if not os.path.exists(log_path):
    return None
  d_means, c_means = [], []
  with open(log_path) as f:
    for line in f:
      if 'D_mean=' in line:
        m = re.search(r"D_mean=\[([^\]]+)\]", line)
        if m:
          d_means.append([float(x.strip()) for x in m.group(1).split(',')])
      if 'C_mean=' in line:
        m = re.search(r"C_mean=\[([^\]]+)\]", line)
        if m:
          c_means.append([float(x.strip()) for x in m.group(1).split(',')])
  if not d_means:
    return None
  import numpy as np
  d = np.mean(d_means, axis=0)
  c = np.mean(c_means, axis=0) if c_means else [0, 0, 0]
  names = ['text', 'audio', 'vision']
  return {
    'D_mean': dict(zip(names, d.tolist())),
    'C_mean': dict(zip(names, c.tolist())),
    'top_utility': names[int(d.argmax())] if len(d) else 'n/a',
    'top_pollution': names[int(c.argmin())] if len(c) else 'n/a',
  }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', default='results/results/baseline')
    parser.add_argument('--mide_dir', default='results/results/mide')
    parser.add_argument('--output', default='results/mide_report.md')
    parser.add_argument('--mide_log', default='results/logs/mide_m0.3.log')
    args = parser.parse_args()

    base = load_results_table(args.baseline_dir)
    mide = load_results_table(args.mide_dir)
    auilc_base = compute_auilc(base)
    auilc_mide = compute_auilc(mide)

    lines = [
        '# MIDE Experiment Report',
        '',
        f'Generated: {datetime.now().isoformat(timespec="seconds")}',
        '',
        '## Engineering Fixes (not MIDE innovation)',
        '- Removed broken gradient accumulation; per-batch `zero_grad/backward/step`.',
        '- `valid/test` DataLoader: `shuffle=False`, `drop_last=False`.',
        '- Dynamic batch reshape in `fusion_599.py` (no fixed `batch_size`).',
        '- Replaced hardcoded `.cuda()` with device-aware tensors.',
        '- Data root priority: `--data_root` > `MFMB_DATA_ROOT` > default path.',
        '',
        '## Baseline Results (MOSI)',
        '',
    ]

    def df_to_md(df):
        cols = list(df.columns)
        header = '| ' + ' | '.join(cols) + ' |'
        sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
        rows = []
        for _, row in df.iterrows():
            rows.append('| ' + ' | '.join(str(row[c]) for c in cols) + ' |')
        return '\n'.join([header, sep] + rows)

    if not base.empty:
        lines.append(df_to_md(base))
    else:
        lines.append('_No baseline CSV found._')

    lines += ['', '## MIDE Results (MOSI)', '']
    if not mide.empty:
        lines.append(df_to_md(mide))
    else:
        lines.append('_No MIDE CSV found._')

    lines += ['', '## AUILC Comparison', '']
    rows = []
    for metric in METRICS:
        if metric not in auilc_base and metric not in auilc_mide:
            continue
        b = auilc_base.get(metric, float('nan'))
        m = auilc_mide.get(metric, float('nan'))
        delta = m - b
        if not HIGHER_BETTER.get(metric, True):
            better = delta < 0
            delta_report = -delta
        else:
            better = delta > 0
            delta_report = delta
        rows.append({
            'metric': metric,
            'baseline_auilc': round(b, 4),
            'mide_auilc': round(m, 4),
            'abs_improvement': round(delta_report, 4),
            'mide_better': better,
        })
    if rows:
        lines.append(df_to_md(pd.DataFrame(rows)))
    else:
        lines.append('_Insufficient sweep for AUILC._')

    stats = summarize_modality_stats(args.mide_log)
    lines += ['', '## Training Stability / MIDE Statistics', '']
    if stats:
        lines.append(f"- Mean D_task: text={stats['D_mean']['text']:.3f}, audio={stats['D_mean']['audio']:.3f}, vision={stats['D_mean']['vision']:.3f}")
        lines.append(f"- Mean contribution C: text={stats['C_mean']['text']:.3f}, audio={stats['C_mean']['audio']:.3f}, vision={stats['C_mean']['vision']:.3f}")
        lines.append(f"- Highest utility modality (by D): **{stats['top_utility']}**")
        lines.append(f"- Most negative contribution (pollution proxy): **{stats['top_pollution']}**")
    else:
        lines.append('_MIDE stat logs not found yet._')

    lines += [
        '',
        '## Conclusion',
        '- Compare tables above for per-missing-rate gains.',
        '- AUILC summarizes robustness across missing rates (trapezoidal integration).',
        '- Engineering fixes affect both baseline and MIDE equally; metric deltas attribute to MIDE when baseline uses the same fixed trainer.',
        '',
    ]

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
