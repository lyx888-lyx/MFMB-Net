#!/usr/bin/env python3
"""Compute AUILC (area under missing-rate curve) via trapezoidal integration."""

import argparse
import os
import re
import glob
import pandas as pd
import numpy as np


METRICS = [
    'MAE', 'Corr', 'Has0_acc_2', 'Has0_F1_score',
    'Non0_acc_2', 'Non0_F1_score', 'Mult_acc_5', 'Mult_acc_7',
]

HIGHER_BETTER = {
    'MAE': False,
    'Corr': True,
    'Has0_acc_2': True,
    'Has0_F1_score': True,
    'Non0_acc_2': True,
    'Non0_F1_score': True,
    'Mult_acc_5': True,
    'Mult_acc_7': True,
}


def parse_mean(val):
    if isinstance(val, (tuple, list)):
        return float(val[0])
    if isinstance(val, str):
        s = val.strip().strip('"').strip("'")
        if not s:
            return float('nan')
        if s.startswith('(') and ',' in s:
            return float(s.strip('()').split(',')[0].strip())
        if '(' in s:
            return float(s.split('(')[-1].split(',')[0].strip())
        return float(s)
    return float(val)


def load_csv_dir(csv_dir, pattern='*.csv'):
    rows = []
    for path in sorted(glob.glob(os.path.join(csv_dir, pattern))):
        m = re.search(r'-(\d+\.\d+)\.csv$', path)
        if not m:
            continue
        missing = float(m.group(1))
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[-1]
        entry = {'missing': missing}
        for col in df.columns:
            if col == 'Model':
                continue
            entry[col] = parse_mean(row[col])
        rows.append(entry)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('missing')


def trapezoid_auc(x, y):
    return float(np.trapz(y, x))


def compute_auilc(df):
    results = {}
    if df.empty:
        return results
    x = df['missing'].values
    for metric in METRICS:
        if metric not in df.columns:
            continue
        y = df[metric].values
        if not HIGHER_BETTER.get(metric, True):
            y = -y
        results[metric] = trapezoid_auc(x, y)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', default='results/results/baseline')
    parser.add_argument('--mide_dir', default='results/results/mide')
    parser.add_argument('--output', default='results/auilc_comparison.csv')
    args = parser.parse_args()

    base_df = load_csv_dir(args.baseline_dir)
    mide_df = load_csv_dir(args.mide_dir)

    base_auilc = compute_auilc(base_df)
    mide_auilc = compute_auilc(mide_df)

    rows = []
    for metric in METRICS:
        if metric not in base_auilc and metric not in mide_auilc:
            continue
        b = base_auilc.get(metric, np.nan)
        m = mide_auilc.get(metric, np.nan)
        delta = m - b
        if not HIGHER_BETTER.get(metric, True):
            delta = -delta
        rows.append({
            'metric': metric,
            'baseline_auilc': b,
            'mide_auilc': m,
            'delta_mide_minus_baseline': delta,
            'higher_is_better': HIGHER_BETTER.get(metric, True),
            'mide_wins': delta > 0 if HIGHER_BETTER.get(metric, True) else delta < 0,
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(out_df.to_string(index=False))
    print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
