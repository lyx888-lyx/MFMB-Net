#!/usr/bin/env python3
"""
Offline metric recomputation from exported test predictions CSV.

Usage (from MFMB-Net/ directory):
  python tools/recompute_metrics_from_csv.py --csv results/predictions/predictions_mosi_fc_text_seed111_test.csv

Outputs:
  - Terminal: (1) metricsTop-style using utils.metricsTop (same as training)
  - Terminal: (2) explicit formulas (documented below)
  - Writes <csv_basename>_recompute_summary.json next to the csv

Explicit definitions (set 2):
  MAE_raw: mean(|pred_raw - truth|)
  Corr_raw: Pearson correlation between pred_raw and truth (numpy corrcoef)
  Mult_acc_7: clip pred and truth to [-3,3], then mean(round(pred) == round(truth))
  Mult_acc_5: clip pred and truth to [-2,2], then mean(round(pred) == round(truth))
  MAE_clip: clip both pred and truth to [-3,3], then mean(|pred - truth|)

metricsTop MOSI regression delegates to __eval_mosei_regression (see utils/metricsTop.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def explicit_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = pred.astype(np.float64).reshape(-1)
    truth = truth.astype(np.float64).reshape(-1)
    mae_raw = float(np.mean(np.abs(pred - truth)))
    corr_raw = float(np.corrcoef(pred, truth)[0, 1])

    pc = np.clip(pred, -3.0, 3.0)
    tc = np.clip(truth, -3.0, 3.0)
    mae_clip = float(np.mean(np.abs(pc - tc)))

    p7, t7 = np.clip(pred, -3.0, 3.0), np.clip(truth, -3.0, 3.0)
    mult_acc_7 = float(np.mean(np.round(p7) == np.round(t7)))

    p5, t5 = np.clip(pred, -2.0, 2.0), np.clip(truth, -2.0, 2.0)
    mult_acc_5 = float(np.mean(np.round(p5) == np.round(t5)))

    return {
        'MAE_raw': round(mae_raw, 6),
        'Corr_raw': round(corr_raw, 6),
        'MAE_clip_m3_p3': round(mae_clip, 6),
        'Mult_acc_7_explicit': round(mult_acc_7, 6),
        'Mult_acc_5_explicit': round(mult_acc_5, 6),
    }


def metrics_top_style(pred: np.ndarray, truth: np.ndarray, dataset_name: str) -> dict:
    import torch
    from utils.metricsTop import MetricsTop

    p = torch.tensor(pred, dtype=torch.float32).view(-1, 1)
    t = torch.tensor(truth, dtype=torch.float32).view(-1, 1)
    fn = MetricsTop('regression').getMetics(dataset_name.upper())
    out = fn(p, t)
    # make JSON-serializable
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Exported predictions CSV path')
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name for MetricsTop (default: read from CSV column datasetName or mosi)',
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if 'pred_raw' not in df.columns or 'truth' not in df.columns:
        raise SystemExit('CSV must contain columns pred_raw and truth')

    pred = df['pred_raw'].values
    truth = df['truth'].values
    ds = args.dataset or (df['datasetName'].iloc[0] if 'datasetName' in df.columns else 'mosi')

    print('=== (1) MetricsTop (same code path as training / do_test) ===')
    mt = metrics_top_style(pred, truth, ds)
    for k in sorted(mt.keys()):
        print(f'  {k}: {mt[k]}')

    print('\n=== (2) Explicit standalone formulas (documented in this script) ===')
    ex = explicit_metrics(pred, truth)
    for k in sorted(ex.keys()):
        print(f'  {k}: {ex[k]}')

    print('\n=== Comparison notes ===')
    print('  - MetricsTop MAE/Corr use raw pred/truth (see utils/metricsTop.py __eval_mosei_regression).')
    print('  - Mult_acc_7/5 in MetricsTop use clip + round + __multiclass_acc (same as explicit if formulas match).')
    print('  - MAE_clip_m3_p3 is NOT printed by MetricsTop; use it only to compare with papers that clip first.')

    out_path = args.csv.replace('.csv', '_recompute_summary.json')
    summary = {
        'csv': os.path.abspath(args.csv),
        'datasetName_used': ds,
        'n_rows': int(len(df)),
        'metricsTop_style': mt,
        'explicit': ex,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
