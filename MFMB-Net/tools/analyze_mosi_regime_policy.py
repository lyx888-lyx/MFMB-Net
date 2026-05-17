#!/usr/bin/env python3
"""
MOSI / MOSEI missing-rate analysis: deployable branches, regime-aware policy (tau1/tau2),
AUILC / slopes, and paper figures.

Learnable routers are not part of the policy or primary plots.

Run from the MFMB-Net package root (directory containing ``utils/`` and ``results/``):
  python tools/analyze_mosi_regime_policy.py --dataset mosi
  python tools/analyze_mosi_regime_policy.py --dataset mosei

See ``analysis_report.md`` in the output directory for caveats (test-set threshold search, oracle).
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.rebuild_mosei_normals_from_predictions import eval_mosei_regression_from_arrays

# Policy search uses exactly three branches (independent tau search per --dataset).
POLICY_BRANCH_KEYS = ['text', 'text_mstcn', 'dynamic_rule']

# All branches loaded into tables / AUILC bars (policy excludes dynamic_lte_mstcn).
ALL_METRIC_BRANCH_KEYS = ['text', 'dynamic_rule', 'text_mstcn', 'dynamic_lte_mstcn']

# Canonical slugs for reporting (actual normals folder may use legacy fallback).
METHOD_SLUG_PRIMARY = {
    'text': 'text',
    'text_mstcn': 'text_lte_mstcn',
    'dynamic_rule': 'dynamic_missing_router_rule',
    'dynamic_lte_mstcn': 'dynamic_missing_lte_mstcn_router_rule',
}

# Plot / table labels (English).
MAIN_LABELS_EN = {
    'text': 'text + legacy LTE',
    'text_mstcn': 'text + MSTCN',
    'dynamic_rule': 'dynamic missing + rule router + legacy LTE',
    'dynamic_lte_mstcn': 'dynamic missing + LTE MSTCN + rule router',
    'policy': 'Regime-aware policy',
}

# Ordered normals subdirs to try (new slug first, then legacy fallback where specified).
NORMAL_SUBDIR_CANDIDATES = {
    'text': ['text'],
    'text_mstcn': ['text_lte_mstcn'],
    'dynamic_rule': ['dynamic_missing_router_rule', 'dynamic_missing'],
    'dynamic_lte_mstcn': ['dynamic_missing_lte_mstcn_router_rule', 'dynamic_missing_lte_mstcn'],
}

# Prediction folder miss_<rate>_<center>_<lte>/ (try candidates until CSVs found).
PRED_SUBFOLDER_CANDIDATES: Dict[str, List[Tuple[str, str]]] = {
    'text': [('text', 'legacy')],
    'text_mstcn': [('text', 'mstcn')],
    'dynamic_rule': [('dynamic_missing', 'legacy')],
    'dynamic_lte_mstcn': [
        ('dynamic_missing_lte_mstcn', 'router_rule'),
        ('dynamic_missing_lte_mstcn', 'legacy'),
    ],
}

MAIN_LINE_COLORS = {
    'text': '#1f4e79',
    'dynamic_rule': '#2f6690',
    'text_mstcn': '#447a9c',
    'policy': '#c45c26',
}

# Paper figures: concise legend names (do not use bare "Policy", "Dynamic", etc.).
METHOD_DISPLAY_SHORT = {
    'text': 'Baseline',
    'dynamic_rule': 'Ours-Dynamic',
    'text_mstcn': 'Ours-MSTCN',
    'dynamic_lte_mstcn': 'Ours-Dynamic+MSTCN',
    'regime_policy': 'Ours-Policy',
}

# Curve / bar colors (CCF-style).
PAPER_LINE_COLORS = {
    'text': '#1f4e79',  # Baseline — deep blue
    'dynamic_rule': '#5b7c8f',  # Ours-Dynamic — gray-blue
    'text_mstcn': '#8fa9b8',  # Ours-MSTCN — light gray-blue
    'dynamic_lte_mstcn': '#6b8e9f',
    'policy': '#d35400',  # Ours-Policy — orange
    'regime_policy': '#d35400',  # same curve as policy (AUILC bar label Ours-Policy)
}

# Full tick grid for missing-rate axes (subset shown if data span is narrower).
STANDARD_MISSING_RATE_TICKS = (
    0.0,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
)

# Main performance / schedule x-axis: include 0.7–1.0 for paper-style full range.
MAIN_DISPLAY_X_TICKS = (
    0.0,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)

FIGURE_DPI = 300
LINE_WIDTH = 2.0
MARKER_SIZE = 5.5
Y_SCALE_DISPLAY = 100.0

AUILC_COMPARE_METRICS = ['Mult_acc_5', 'Mult_acc_7', 'MAE', 'Corr']
AUILC_SUMMARY_CURVES = [
    'text',
    'dynamic_rule',
    'text_mstcn',
    'dynamic_lte_mstcn',
    'regime_policy',
    'oracle_best_point',
]

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
HIGHER_BETTER = {
    'Has0_acc_2',
    'Has0_F1_score',
    'Non0_acc_2',
    'Non0_F1_score',
    'Mult_acc_5',
    'Mult_acc_7',
    'Corr',
}
LOWER_BETTER = {'MAE', 'Loss'}

DEFAULT_RATES = [
    0.0,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
]

DATASET_SPECS: Dict[str, Dict[str, str]] = {
    'mosi': {
        'csv_stem': 'mosi-regression',
        'pred_glob': 'predictions_mosi_fc_*.csv',
        'display': 'MOSI',
        'default_analysis_dir': os.path.join(ROOT, 'results', 'analysis', 'mosi_regime_policy'),
    },
    'mosei': {
        'csv_stem': 'mosei-regression',
        'pred_glob': 'predictions_mosei_fc_*.csv',
        'display': 'MOSEI',
        'default_analysis_dir': os.path.join(ROOT, 'results', 'analysis', 'mosei_regime_policy'),
    },
}


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def format_missing_slug(r: float) -> str:
    """Match common result filenames (0.0, 0.1, 0.15, ...)."""
    if abs(r - round(r, 1)) < 1e-9:
        return f'{r:.1f}'
    s = f'{r:.4f}'.rstrip('0').rstrip('.')
    return s if s else '0'


def parse_csv_mean_cell(cell: Any) -> float:
    """Training CSV stores (mean*100, std*100) per run.py — return mean in [~0,1] native metric space."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return float('nan')
    s = str(cell).strip()
    if not s or s.lower() in ('nan', 'none'):
        return float('nan')
    if s.startswith('('):
        try:
            t = ast.literal_eval(s)
            return float(t[0]) / 100.0
        except (SyntaxError, ValueError, TypeError, IndexError):
            return float('nan')
    try:
        return float(s) / 100.0
    except ValueError:
        return float('nan')


def load_normals_csv(path: str) -> Dict[str, float]:
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    out: Dict[str, float] = {}
    for m in METRICS:
        if m not in df.columns:
            out[m] = float('nan')
            continue
        vals = [parse_csv_mean_cell(v) for v in df[m].values]
        vals = [v for v in vals if not math.isnan(v)]
        out[m] = float(np.mean(vals)) if vals else float('nan')
    return out


def glob_prediction_csvs(
    pred_root: str,
    miss: float,
    method_key: str,
    pred_glob: str,
) -> List[str]:
    """Collect test prediction CSVs across candidate miss_* folder layouts."""
    ms = format_missing_slug(miss)
    found: set = set()
    for center, lte in PRED_SUBFOLDER_CANDIDATES[method_key]:
        pdir = os.path.join(pred_root, f'miss_{ms}_{center}_{lte}')
        if not os.path.isdir(pdir):
            continue
        for p in sorted(glob.glob(os.path.join(pdir, pred_glob))):
            if 'test' in os.path.basename(p).lower():
                found.add(p)
    return sorted(found)


def metrics_from_prediction_csv(path: str) -> Dict[str, float]:
    df = pd.read_csv(path)
    if 'pred_raw' not in df.columns or 'truth' not in df.columns:
        return {m: float('nan') for m in METRICS}
    pred = df['pred_raw'].values.astype(np.float64)
    truth = df['truth'].values.astype(np.float64)
    ev = eval_mosei_regression_from_arrays(pred, truth)
    ev = {k: float(v) for k, v in ev.items()}
    # Loss is batch L1 in do_test — not recoverable from row-wise export
    ev['Loss'] = float('nan')
    for m in METRICS:
        if m not in ev:
            ev[m] = float('nan')
    return ev


def aggregate_predictions(
    pred_root: str, miss: float, method_key: str, pred_glob: str
) -> Dict[str, float]:
    paths = glob_prediction_csvs(pred_root, miss, method_key, pred_glob)
    if not paths:
        return {m: float('nan') for m in METRICS}
    per_seed: List[Dict[str, float]] = []
    for p in paths:
        per_seed.append(metrics_from_prediction_csv(p))
    agg: Dict[str, float] = {}
    for m in METRICS:
        xs = [d[m] for d in per_seed if m in d and not math.isnan(d[m])]
        agg[m] = float(np.mean(xs)) if xs else float('nan')
    return agg


def load_method_at_rate(
    normals_root: str,
    pred_root: str,
    miss: float,
    method_key: str,
    csv_stem: str,
    pred_glob: str,
) -> Tuple[Dict[str, float], str]:
    slug = format_missing_slug(miss)
    for sub in NORMAL_SUBDIR_CANDIDATES[method_key]:
        npath = os.path.join(normals_root, sub, f'{csv_stem}-{slug}.csv')
        if os.path.isfile(npath):
            note = f'normals:{os.path.relpath(npath, ROOT)}'
            if sub != NORMAL_SUBDIR_CANDIDATES[method_key][0]:
                note += ' (fallback slug)'
            return load_normals_csv(npath), note
    agg = aggregate_predictions(pred_root, miss, method_key, pred_glob)
    if any(not math.isnan(agg.get(m, float('nan'))) for m in METRICS if m != 'Loss'):
        g = glob_prediction_csvs(pred_root, miss, method_key, pred_glob)
        return agg, f'predictions:{g[0] if g else "glob"}'
    return {m: float('nan') for m in METRICS}, 'missing'


def trapezoid_auilc(rates: np.ndarray, y: np.ndarray) -> float:
    """AUILC = sum_i 0.5 * (y_i + y_{i+1}) * (r_{i+1} - r_i)."""
    mask = ~(np.isnan(rates) | np.isnan(y))
    r = rates[mask]
    v = y[mask]
    if len(r) < 2:
        return float('nan')
    order = np.argsort(r)
    r = r[order]
    v = v[order]
    return float(np.sum(0.5 * (v[1:] + v[:-1]) * (r[1:] - r[:-1])))


def slice_interval(rates: np.ndarray, y: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (rates >= lo) & (rates <= hi) & ~np.isnan(y)
    return rates[mask], y[mask]


def interp_rates_flat_extrap(
    r_obs: np.ndarray,
    y_obs: np.ndarray,
    r_query: np.ndarray,
) -> np.ndarray:
    """
    Piecewise-linear interpolation on sorted observed knots; flat extrapolation
    outside [min(r_obs), max(r_obs)] (numpy.interp semantics).
    """
    rq = np.asarray(r_query, dtype=float)
    ok = np.isfinite(r_obs) & np.isfinite(y_obs)
    r = np.asarray(r_obs[ok], dtype=float)
    y = np.asarray(y_obs[ok], dtype=float)
    if r.size == 0:
        return np.full(rq.shape, np.nan, dtype=float)
    order = np.argsort(r)
    r, y = r[order], y[order]
    return np.interp(rq, r, y, left=float(y[0]), right=float(y[-1]))


def merge_ticks_for_axis(
    x_min: float,
    x_max: float,
    base_ticks: Sequence[float],
    extra_rates: Optional[np.ndarray] = None,
) -> List[float]:
    acc = set()
    for t in base_ticks:
        tf = float(t)
        if x_min - 1e-12 <= tf <= x_max + 1e-12:
            acc.add(round(tf, 4))
    if extra_rates is not None:
        for tf in np.asarray(extra_rates, dtype=float).ravel():
            if np.isfinite(tf) and x_min - 1e-12 <= float(tf) <= x_max + 1e-12:
                acc.add(round(float(tf), 4))
    return sorted(acc)


def rankdata_higher_better(x: np.ndarray) -> np.ndarray:
    """Average ranks, 1 = best (largest value). NaNs get worst ranks."""
    s = pd.Series(np.asarray(x, dtype=float))
    out = np.full(len(s), np.nan)
    ok = s.notna()
    if not ok.any():
        return out
    out[ok.to_numpy()] = s[ok].rank(ascending=False, method='average').to_numpy()
    worst = float(np.nanmax(out) + 1.0)
    out[~ok.to_numpy()] = worst
    return out


def rankdata_lower_better(x: np.ndarray) -> np.ndarray:
    """1 = best (smallest value)."""
    s = pd.Series(np.asarray(x, dtype=float))
    out = np.full(len(s), np.nan)
    ok = s.notna()
    if not ok.any():
        return out
    out[ok.to_numpy()] = s[ok].rank(ascending=True, method='average').to_numpy()
    worst = float(np.nanmax(out) + 1.0)
    out[~ok.to_numpy()] = worst
    return out


def select_best_primary(
    table: Dict[str, Dict[str, float]],
    rate: float,
    tie_mult7: float = 0.002,
) -> Tuple[str, str]:
    """
    Mode A: best Mult_acc_7; tie if within tie_mult7 (in [0,1] metric space, ~0.2 on CSV x100 scale)
    then lower MAE, then higher Corr.
    """
    rows = []
    for mk in POLICY_BRANCH_KEYS:
        m = table.get(mk, {})
        rows.append(
            (
                mk,
                m.get('Mult_acc_7', float('nan')),
                m.get('MAE', float('nan')),
                m.get('Corr', float('nan')),
            )
        )
    rows.sort(
        key=lambda z: (
            -z[1] if not math.isnan(z[1]) else float('-inf'),
            z[2] if not math.isnan(z[2]) else float('inf'),
            -z[3] if not math.isnan(z[3]) else float('-inf'),
        )
    )
    if not rows:
        return 'text', 'empty'
    top_m7 = rows[0][1]
    tie = [
        z
        for z in rows
        if not math.isnan(z[1]) and not math.isnan(top_m7) and abs(z[1] - top_m7) < tie_mult7
    ]
    if not tie:
        tie = [rows[0]]
    tie.sort(
        key=lambda z: (
            z[2] if not math.isnan(z[2]) else float('inf'),
            -z[3] if not math.isnan(z[3]) else float('-inf'),
        )
    )
    best = tie[0][0]
    rationale = f'Mult_acc_7={top_m7:.6f}'
    if len(tie) > 1:
        rationale += f'; tie-break within ±{tie_mult7} using MAE then Corr'
    return best, rationale


def select_best_rank_sum(table: Dict[str, Dict[str, float]], rate: float) -> Tuple[str, np.ndarray]:
    """Mode B: rank sum over Mult_acc_5, Mult_acc_7, MAE (lower better), Corr (higher better)."""
    M = np.full((len(POLICY_BRANCH_KEYS), 4), np.nan)
    for i, mk in enumerate(POLICY_BRANCH_KEYS):
        m = table.get(mk, {})
        M[i, 0] = m.get('Mult_acc_5', float('nan'))
        M[i, 1] = m.get('Mult_acc_7', float('nan'))
        M[i, 2] = m.get('MAE', float('nan'))
        M[i, 3] = m.get('Corr', float('nan'))
    r5 = rankdata_higher_better(M[:, 0])
    r7 = rankdata_higher_better(M[:, 1])
    r_mae = rankdata_lower_better(M[:, 2])
    r_c = rankdata_higher_better(M[:, 3])
    rs = r5 + r7 + r_mae + r_c
    if np.all(np.isnan(rs)):
        return POLICY_BRANCH_KEYS[0], rs
    j = int(np.nanargmin(rs))
    return POLICY_BRANCH_KEYS[j], rs


def policy_pick_method(r: float, tau1: float, tau2: float) -> str:
    if r <= tau1:
        return 'dynamic_rule'
    if r <= tau2:
        return 'text_mstcn'
    return 'text'


def format_policy_definition(tau1: float, tau2: float) -> str:
    """Human-readable policy line for CSV/report."""

    def fmt(x: float) -> str:
        if math.isnan(x):
            return 'nan'
        if abs(x - round(x, 2)) < 1e-9:
            s = f'{x:.2f}'.rstrip('0').rstrip('.')
            return s if s else '0'
        return f'{x:.4g}'

    return (
        f'if r <= {fmt(tau1)}: dynamic_rule; '
        f'elif r <= {fmt(tau2)}: text_mstcn; else: text'
    )


def composite_at_point(vals: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Per-method composite: min–max norm across the three policy branches at this rate."""
    eps = 1e-12
    score: Dict[str, float] = {}
    keys = POLICY_BRANCH_KEYS
    def mm(arr):
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return 0.0, 1.0
        return float(np.min(a)), float(np.max(a))

    m5s = [vals[k].get('Mult_acc_5', float('nan')) for k in keys]
    m7s = [vals[k].get('Mult_acc_7', float('nan')) for k in keys]
    crs = [vals[k].get('Corr', float('nan')) for k in keys]
    maes = [vals[k].get('MAE', float('nan')) for k in keys]
    lo5, hi5 = mm(m5s)
    lo7, hi7 = mm(m7s)
    loc, hic = mm(crs)
    lom, him = mm(maes)
    for mk in POLICY_BRANCH_KEYS:
        m5 = vals[mk].get('Mult_acc_5', float('nan'))
        m7 = vals[mk].get('Mult_acc_7', float('nan'))
        cr = vals[mk].get('Corr', float('nan'))
        mae = vals[mk].get('MAE', float('nan'))
        if any(math.isnan(v) for v in (m5, m7, cr, mae)):
            score[mk] = float('nan')
            continue
        z5 = (m5 - lo5) / (hi5 - lo5 + eps)
        z7 = (m7 - lo7) / (hi7 - lo7 + eps)
        zc = (cr - loc) / (hic - loc + eps)
        zm_bad = (mae - lom) / (him - lom + eps)
        score[mk] = z5 + z7 + zc - zm_bad
    return score


def search_policy(
    rates: Sequence[float],
    by_rate: Dict[float, Dict[str, Dict[str, float]]],
    objective: str,
) -> Tuple[float, float, float, str]:
    """
    Search tau1 <= tau2 from observed rates.
    objective: 'auilc_mult7' | 'composite'
    Returns tau1, tau2, best_score, objective_name
    """
    R = sorted(set(float(x) for x in rates))
    best = (-1.0, None, None, '')
    for tau1 in R:
        for tau2 in R:
            if tau1 > tau2:
                continue
            ys = []
            rs = []
            comp_series = []
            for r in R:
                row = by_rate.get(r)
                if not row:
                    continue
                mk = policy_pick_method(r, tau1, tau2)
                m = row.get(mk)
                if not m:
                    continue
                m7 = m.get('Mult_acc_7', float('nan'))
                if objective == 'auilc_mult7':
                    if not math.isnan(m7):
                        ys.append(m7)
                        rs.append(r)
                else:
                    comp = composite_at_point(row)
                    cv = comp.get(mk, float('nan'))
                    if not math.isnan(cv):
                        comp_series.append(cv)
                        rs.append(r)
            ra = np.array(rs, dtype=float)
            if objective == 'auilc_mult7':
                yv = np.array(ys, dtype=float)
                au = trapezoid_auilc(ra, yv)
            else:
                yv = np.array(comp_series, dtype=float)
                au = trapezoid_auilc(ra, yv)
            if math.isnan(au):
                continue
            if au > best[0]:
                best = (au, tau1, tau2, objective)
    return (best[1], best[2], best[0], objective) if best[1] is not None else (float('nan'), float('nan'), float('nan'), objective)


def build_tables(
    rates: Sequence[float],
    normals_root: str,
    pred_root: str,
    csv_stem: str,
    pred_glob: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    prov = {}
    for r in rates:
        by_m = {}
        srcs = {}
        for mk in ALL_METRIC_BRANCH_KEYS:
            met, src = load_method_at_rate(normals_root, pred_root, r, mk, csv_stem, pred_glob)
            by_m[mk] = met
            srcs[mk] = src
        prov[str(r)] = srcs
        for mk in ALL_METRIC_BRANCH_KEYS:
            rec = {
                'missing_rate': r,
                'method_name': mk,
                'method_label': MAIN_LABELS_EN[mk],
            }
            for m in METRICS:
                rec[m] = by_m[mk].get(m, float('nan'))
            rows.append(rec)
    df = pd.DataFrame(rows)
    return df, prov


def slopes_for_method(
    rates: np.ndarray, series: np.ndarray
) -> List[Dict[str, Any]]:
    out = []
    order = np.argsort(rates)
    r = rates[order]
    y = series[order]
    for i in range(len(r) - 1):
        dr = r[i + 1] - r[i]
        if dr <= 0:
            continue
        slope = (y[i + 1] - y[i]) / dr
        out.append(
            {
                'r_lo': float(r[i]),
                'r_hi': float(r[i + 1]),
                'slope': float(slope),
            }
        )
    return out


def mean_slope_in_range(
    slopes: List[Dict[str, Any]], lo: float, hi_bound: float, _metric_higher_better: bool
) -> float:
    """Mean slope over segments whose **midpoint** lies in [lo, hi_bound] (inclusive)."""
    acc = []
    for s in slopes:
        a, b = s['r_lo'], s['r_hi']
        mid = 0.5 * (a + b)
        if lo <= mid <= hi_bound:
            if not math.isnan(s['slope']):
                acc.append(s['slope'])
    if not acc:
        return float('nan')
    return float(np.mean(acc))


def setup_matplotlib():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times', 'DejaVu Serif'],
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'axes.grid': False,
            'axes.edgecolor': '#333333',
            'axes.linewidth': 0.8,
        }
    )
    return plt


# Delta bands vs zero (pairwise figures are line-only; no red/green between curves).
DELTA_FILL_POS = '#c8e6c9'
DELTA_FILL_NEG = '#ffcdd2'


def _metric_axis_short(metric: str) -> str:
    """Axis / title fragment: MAE, Mult-Acc7, or Corr."""
    if metric == 'Mult_acc_7':
        return 'Mult-Acc7'
    if metric == 'MAE':
        return 'MAE'
    if metric == 'Corr':
        return 'Corr'
    return metric


def _pairwise_ylabel(metric: str) -> str:
    return f'{_metric_axis_short(metric)} (×100)'


def _apply_light_grid(ax) -> None:
    ax.grid(True, which='major', linestyle='-', linewidth=0.55, alpha=0.38, color='#bfbfbf')


def _configure_missing_rate_xaxis(
    ax,
    x_min: float,
    x_max: float,
    *,
    base_ticks: Sequence[float] = STANDARD_MISSING_RATE_TICKS,
    extra_rates: Optional[np.ndarray] = None,
) -> None:
    """Numeric x, flush ends, tick marks include 0.15/0.25/… and any observed rates in range."""
    import matplotlib.ticker as mticker

    ticks = merge_ticks_for_axis(x_min, x_max, base_ticks, extra_rates)
    if not ticks:
        ticks = [x_min, x_max]
    ax.set_xlim(float(x_min), float(x_max))
    ax.margins(x=0)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha('right')


def save_figure_paper(fig, base_path_no_ext: str) -> None:
    """Save PNG (dpi>=300) and PDF with bbox_inches=tight."""
    d = os.path.dirname(base_path_no_ext)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(f'{base_path_no_ext}.png', dpi=FIGURE_DPI, bbox_inches='tight')
    fig.savefig(f'{base_path_no_ext}.pdf', dpi=FIGURE_DPI, bbox_inches='tight')


def plot_figure_main_policy_vs_baseline_mosi(
    plt,
    rates: np.ndarray,
    y_baseline_native: Dict[str, np.ndarray],
    y_policy_native: Dict[str, np.ndarray],
    out_base: str,
    x_axis_hi: float = 1.0,
    dataset_display: str = 'MOSI',
) -> None:
    """Figure A1: Baseline vs Ours-Policy on [0, x_axis_hi] with flat extrapolation beyond measured rates."""
    import matplotlib.ticker as mtick

    metrics_cfg = [
        ('Mult_acc_7', '(a) Mult-Acc7', 'Mult-Acc7 (×100)'),
        ('Corr', '(b) Correlation', 'Corr (×100)'),
        ('MAE', '(c) MAE', 'MAE (×100)'),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.1), dpi=FIGURE_DPI, constrained_layout=False)
    fig.suptitle(
        f'Overall Performance on {dataset_display} under Symmetric Missing Rates',
        fontsize=13,
        fontweight='normal',
        y=1.02,
    )
    x_hi = float(x_axis_hi)
    for ax, (mkey, sub_title, ylab) in zip(axs, metrics_cfg):
        yb = np.asarray(y_baseline_native[mkey], dtype=float) * Y_SCALE_DISPLAY
        yp = np.asarray(y_policy_native[mkey], dtype=float) * Y_SCALE_DISPLAY
        ax.plot(
            rates,
            yb,
            '-o',
            label=METHOD_DISPLAY_SHORT['text'],
            color=PAPER_LINE_COLORS['text'],
            lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            zorder=4,
        )
        ax.plot(
            rates,
            yp,
            '-s',
            label=METHOD_DISPLAY_SHORT['regime_policy'],
            color=PAPER_LINE_COLORS['policy'],
            lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            zorder=5,
        )
        ax.set_title(sub_title, fontsize=11.5)
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel(ylab)
        _configure_missing_rate_xaxis(
            ax,
            0.0,
            x_hi,
            base_ticks=MAIN_DISPLAY_X_TICKS,
            extra_rates=rates,
        )
        _apply_light_grid(ax)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6))
    h0, l0 = axs[0].get_legend_handles_labels()
    fig.legend(
        h0,
        l0,
        loc='lower center',
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor='#333333',
        bbox_to_anchor=(0.5, 0.06),
    )
    fig.text(
        0.5,
        0.01,
        'For high missing rates, Ours-Policy falls back to the baseline branch, so curves may overlap.',
        ha='center',
        fontsize=8.5,
        style='italic',
        color='#444444',
    )
    fig.subplots_adjust(bottom=0.28, top=0.88, wspace=0.28)
    save_figure_paper(fig, out_base)
    plt.close(fig)


def fmt_auilc_interval_label(lo: float, hi: float) -> str:
    """Bracket label for figure titles / report (two decimals)."""
    return f'[{lo:.2f}, {hi:.2f}]'


def plot_figure_auilc_summary_mosi(
    plt,
    auilc_by_method: Dict[str, Dict[str, float]],
    out_base: str,
    auilc_lo: float,
    auilc_hi: float,
    dataset_display: str = 'MOSI',
) -> None:
    """Figure A2: grouped bars; suptitle documents trapezoid-AUILC integration interval."""
    methods_order = ['text', 'dynamic_rule', 'text_mstcn', 'dynamic_lte_mstcn', 'regime_policy']
    labels = [METHOD_DISPLAY_SHORT[k] for k in methods_order]
    colors = [PAPER_LINE_COLORS[k] for k in methods_order]
    interval_tag = fmt_auilc_interval_label(auilc_lo, auilc_hi)
    metrics_plot = [
        ('Mult_acc_7', 'Mult-Acc7', 'Higher is better', 'AUILC (Mult-Acc7)'),
        ('Corr', 'Corr', 'Higher is better', 'AUILC (Corr)'),
        ('MAE', 'MAE', 'Lower is better', 'AUILC (MAE)'),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(17.5, 4.2), dpi=FIGURE_DPI)
    fig.suptitle(f'AUILC Summary on {dataset_display} ({interval_tag})', fontsize=13, y=1.02)
    x = np.arange(len(methods_order), dtype=float)
    width = 0.42
    for ax, (mkey, sub_title, note, ylab) in zip(axs, metrics_plot):
        vals = [float(auilc_by_method[k][mkey]) for k in methods_order]
        ax.bar(
            x,
            vals,
            width,
            color=colors,
            edgecolor='#222222',
            linewidth=0.6,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right')
        ax.set_title(sub_title, fontsize=11.5)
        ax.set_ylabel(ylab)
        ax.text(
            0.5,
            0.02,
            note,
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=8.5,
            style='italic',
            color='#444444',
        )
        _apply_light_grid(ax)
        ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.30, bottom=0.2, top=0.86)
    save_figure_paper(fig, out_base)
    plt.close(fig)


def _policy_branch_ycode(mk: str) -> int:
    return {'dynamic_rule': 2, 'text_mstcn': 1, 'text': 0}[mk]


def plot_figure_policy_schedule_mosi(
    plt,
    tau1: float,
    tau2: float,
    pick_fn,
    out_base: str,
    x_axis_hi: float = 1.0,
    dataset_display: str = 'MOSI',
) -> None:
    """Figure A3: branch vs missing rate (step) on [0, x_axis_hi]; τ from threshold search."""
    x_hi = float(x_axis_hi)
    R = np.linspace(0.0, x_hi, max(401, int(x_hi * 400) + 1), dtype=float)
    y = np.array([_policy_branch_ycode(pick_fn(float(r))) for r in R], dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 3.8), dpi=FIGURE_DPI)
    ax.step(R, y, where='post', color='#2c3e50', lw=LINE_WIDTH + 0.3, zorder=3)
    for tau, tag in ((tau1, r'$\tau_1$'), (tau2, r'$\tau_2$')):
        if math.isnan(tau):
            continue
        if not (-1e-12 <= tau <= x_hi + 1e-12):
            continue
        ax.axvline(tau, color='#c0392b', ls='--', lw=1.2, zorder=2)
        ax.text(
            tau,
            2.08,
            tag,
            ha='center',
            va='bottom',
            fontsize=10,
            color='#c0392b',
        )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(
        [METHOD_DISPLAY_SHORT['text'], METHOD_DISPLAY_SHORT['text_mstcn'], METHOD_DISPLAY_SHORT['dynamic_rule']]
    )
    ax.set_ylim(-0.35, 2.45)
    ax.set_xlabel('Missing Rate')
    ax.set_ylabel('Selected branch')
    ax.set_title(f'Missing-Rate-Aware Branch Selection Policy on {dataset_display}')
    _configure_missing_rate_xaxis(ax, 0.0, x_hi, base_ticks=MAIN_DISPLAY_X_TICKS)
    _apply_light_grid(ax)
    fig.tight_layout()
    save_figure_paper(fig, out_base)
    plt.close(fig)


def plot_figure_pairwise_baseline_vs_method_mosi(
    plt,
    rates: np.ndarray,
    y_baseline_native: Dict[str, np.ndarray],
    y_method_native: Dict[str, np.ndarray],
    method_key: str,
    suptitle: str,
    out_base: str,
) -> None:
    """Supplementary: 1×3 pairwise, two lines only (no band fill)."""
    import matplotlib.ticker as mtick

    metrics_cfg = [
        ('Mult_acc_7', '(a) Mult-Acc7'),
        ('Corr', '(b) Correlation'),
        ('MAE', '(c) MAE'),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.0), dpi=FIGURE_DPI)
    fig.suptitle(suptitle, fontsize=12.8, y=1.02)
    r_obs = np.asarray(rates, dtype=float)
    x0, x1 = float(np.min(r_obs)), float(np.max(r_obs))
    for ax, (mkey, sub_title) in zip(axs, metrics_cfg):
        yt = np.asarray(y_baseline_native[mkey], dtype=float) * Y_SCALE_DISPLAY
        ym = np.asarray(y_method_native[mkey], dtype=float) * Y_SCALE_DISPLAY
        ax.plot(
            r_obs,
            yt,
            '-o',
            label=METHOD_DISPLAY_SHORT['text'],
            color=PAPER_LINE_COLORS['text'],
            lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            zorder=4,
        )
        ax.plot(
            r_obs,
            ym,
            '-s',
            label=METHOD_DISPLAY_SHORT[method_key],
            color=PAPER_LINE_COLORS[method_key],
            lw=LINE_WIDTH,
            ms=MARKER_SIZE,
            zorder=5,
        )
        ax.set_title(sub_title, fontsize=11.5)
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel(_pairwise_ylabel(mkey))
        _configure_missing_rate_xaxis(
            ax,
            x0,
            x1,
            base_ticks=MAIN_DISPLAY_X_TICKS,
            extra_rates=r_obs,
        )
        _apply_light_grid(ax)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6))
    h0, l0 = axs[0].get_legend_handles_labels()
    fig.legend(
        h0,
        l0,
        loc='lower center',
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor='#333333',
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.28)
    save_figure_paper(fig, out_base)
    plt.close(fig)


def plot_figure_delta_improvement_mosi(
    plt,
    rates: np.ndarray,
    y_baseline_native: Dict[str, np.ndarray],
    y_dynamic_native: Dict[str, np.ndarray],
    y_mstcn_native: Dict[str, np.ndarray],
    out_base: str,
) -> None:
    """Supplementary: Δ vs Baseline; green/red fill vs zero (pairwise figures stay line-only)."""
    import matplotlib.ticker as mtick

    def delta_series(metric: str, y_method: np.ndarray) -> np.ndarray:
        b = np.asarray(y_baseline_native[metric], dtype=float) * Y_SCALE_DISPLAY
        m = np.asarray(y_method, dtype=float) * Y_SCALE_DISPLAY
        if metric in LOWER_BETTER:
            return b - m
        return m - b

    metrics_cfg = [
        ('Mult_acc_7', '(a) Δ Mult-Acc7'),
        ('Corr', '(b) Δ Corr'),
        ('MAE', '(c) Δ MAE Improvement'),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.0), dpi=FIGURE_DPI)
    fig.suptitle('Relative Improvement over Baseline', fontsize=12.8, y=1.02)
    r_obs = np.asarray(rates, dtype=float)
    x0, x1 = float(np.min(r_obs)), float(np.max(r_obs))
    for ax, (mkey, sub_title) in zip(axs, metrics_cfg):
        d_dyn = delta_series(mkey, y_dynamic_native[mkey])
        d_ms = delta_series(mkey, y_mstcn_native[mkey])
        ax.axhline(0.0, color='#000000', ls='--', lw=1.05, zorder=2)
        for delta, color, lab in (
            (d_dyn, PAPER_LINE_COLORS['dynamic_rule'], METHOD_DISPLAY_SHORT['dynamic_rule']),
            (d_ms, PAPER_LINE_COLORS['text_mstcn'], METHOD_DISPLAY_SHORT['text_mstcn']),
        ):
            ax.fill_between(
                r_obs,
                0.0,
                delta,
                where=(delta >= 0),
                color=DELTA_FILL_POS,
                alpha=0.48,
                interpolate=True,
                zorder=0,
            )
            ax.fill_between(
                r_obs,
                0.0,
                delta,
                where=(delta < 0),
                color=DELTA_FILL_NEG,
                alpha=0.48,
                interpolate=True,
                zorder=0,
            )
            ax.plot(
                r_obs,
                delta,
                '-o' if lab == METHOD_DISPLAY_SHORT['dynamic_rule'] else '-^',
                color=color,
                lw=LINE_WIDTH,
                ms=MARKER_SIZE - 0.5,
                label=lab,
                zorder=4,
            )
        ax.set_title(sub_title, fontsize=11.5)
        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('Δ (×100)')
        _configure_missing_rate_xaxis(
            ax,
            x0,
            x1,
            base_ticks=MAIN_DISPLAY_X_TICKS,
            extra_rates=r_obs,
        )
        _apply_light_grid(ax)
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6))
        ax.text(
            0.98,
            0.98,
            'Positive is better',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=8.5,
            style='italic',
            color='#333333',
        )
    h0, l0 = axs[0].get_legend_handles_labels()
    fig.legend(
        h0,
        l0,
        loc='lower center',
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor='#333333',
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.text(
        0.5,
        0.02,
        'MAE: Δ = MAE(Baseline) − MAE(Method); positive means lower error.',
        ha='center',
        fontsize=8.5,
        style='italic',
        color='#444444',
    )
    fig.subplots_adjust(bottom=0.24, top=0.86, wspace=0.28)
    save_figure_paper(fig, out_base)
    plt.close(fig)


def print_data_presence_audit(df_all: pd.DataFrame, rates: Sequence[float]) -> List[str]:
    """Print per-rate / per-method presence; return human-readable missing lines for the report."""
    core = ('Mult_acc_7', 'MAE', 'Corr')
    miss_lines: List[str] = []
    col_w = max(12, max(len(METHOD_DISPLAY_SHORT[mk]) for mk in ALL_METRIC_BRANCH_KEYS) + 2)
    print('\n=== Data presence (core metrics from normals or predictions) ===')
    header = f"{'rate':>7}" + ''.join(f'{METHOD_DISPLAY_SHORT[mk]:>{col_w}}' for mk in ALL_METRIC_BRANCH_KEYS)
    print(header)
    print('-' * len(header))
    for r in rates:
        row_bits = [f'{r:>7}']
        for mk in ALL_METRIC_BRANCH_KEYS:
            sub = df_all[(df_all['missing_rate'] == r) & (df_all['method_name'] == mk)]
            ok = False
            if len(sub):
                ok = any(
                    not math.isnan(float(sub.iloc[0][c]))
                    for c in core
                    if c in sub.columns
                )
            row_bits.append(f'{"OK" if ok else "MISSING":>{col_w}}')
            if not ok:
                miss_lines.append(
                    f'- missing_rate **{r}** / **{mk}** ({METHOD_DISPLAY_SHORT[mk]}): '
                    f'no usable normals or predictions for {", ".join(core)}'
                )
        print(''.join(row_bits))
    print('=== End data presence ===\n')
    return miss_lines


def main():
    parser = argparse.ArgumentParser(
        description='MOSI / MOSEI regime-aware policy, AUILC, and paper figures'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mosi', 'mosei'],
        default='mosi',
        help='Dataset: normals/predictions stem (mosi-regression-* vs mosei-regression-*)',
    )
    parser.add_argument(
        '--rates',
        type=str,
        default=None,
        help='Comma-separated missing rates (default: built-in grid 0.0..0.6)',
    )
    parser.add_argument(
        '--normals-root',
        type=str,
        default=os.path.join(ROOT, 'results', 'results', 'normals'),
        help='Directory containing text/, dynamic_missing/, ...',
    )
    parser.add_argument(
        '--predictions-root',
        type=str,
        default=os.path.join(ROOT, 'results', 'predictions'),
        help='Directory with miss_<rate>_<center>_<lte>/ prediction CSVs',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Output directory (default: results/analysis/<dataset>_regime_policy/)',
    )
    parser.add_argument(
        '--primary-tie-eps',
        type=float,
        default=0.002,
        help='Mult_acc_7 tie band in native metric (~0.2 on CSV ×100 scale)',
    )
    parser.add_argument(
        '--extend-auilc-to-1',
        action='store_true',
        help='If data include missing_rate>=1, also report AUILC on [0,1]',
    )
    parser.add_argument(
        '--auilc-lo',
        type=float,
        default=0.1,
        help='Lower bound for trapezoid AUILC in bar figures (default 0.1; use 0.0 for [0, hi])',
    )
    parser.add_argument(
        '--auilc-hi',
        type=float,
        default=1.0,
        help='Upper bound for AUILC bar figures (default 1.0; y flat-extrapolates beyond max observed rate)',
    )
    parser.add_argument(
        '--auilc-interval',
        type=str,
        default=None,
        help='Override --auilc-lo/--auilc-hi with "lo,hi" (e.g. 0.1,1.0 or 0.1,0.6 or 0,0.6)',
    )
    parser.add_argument(
        '--main-x-hi',
        type=float,
        default=1.0,
        help='Right x-axis limit for main performance + policy schedule (default 1.0)',
    )
    parser.add_argument(
        '--allow-extrapolate-auilc',
        action='store_true',
        help=(
            'If set, AUILC bar integration uses flat extrapolation beyond max observed missing rate. '
            'If unset (default), requested AUILC hi above max observed rate is clamped (no silent [0.1,1.0] on 0–0.6 only).'
        ),
    )
    args = parser.parse_args()

    ds = str(args.dataset).lower()
    if ds not in DATASET_SPECS:
        raise SystemExit(f'Unknown dataset {ds!r}')
    spec = DATASET_SPECS[ds]
    csv_stem = spec['csv_stem']
    pred_glob = spec['pred_glob']
    dataset_display = spec['display']

    auilc_lo, auilc_hi = float(args.auilc_lo), float(args.auilc_hi)
    if args.auilc_interval:
        parts = [p.strip() for p in args.auilc_interval.split(',') if p.strip()]
        if len(parts) == 2:
            auilc_lo, auilc_hi = float(parts[0]), float(parts[1])
    if auilc_lo > auilc_hi:
        auilc_lo, auilc_hi = auilc_hi, auilc_lo
    main_x_hi = float(args.main_x_hi)

    if args.rates:
        rates = [float(x.strip()) for x in args.rates.split(',') if x.strip()]
    else:
        rates = list(DEFAULT_RATES)

    r_max_obs = float(max(rates)) if rates else 0.0
    auilc_hi_requested = float(auilc_hi)
    auilc_clamp_note = ''
    if not args.allow_extrapolate_auilc and auilc_hi > r_max_obs + 1e-12:
        new_hi = r_max_obs
        auilc_clamp_note = (
            f'AUILC upper bound was requested as {auilc_hi_requested:.4g} but max observed missing rate '
            f'is {r_max_obs:.4g}. **Clamped AUILC integration to [{auilc_lo:.2f}, {new_hi:.2f}]** '
            f'(no flat extrapolation). Re-run with `--allow-extrapolate-auilc` to extrapolate, '
            f'or set e.g. `--auilc-interval 0.1,{r_max_obs:.2f}` for MOSEI-style grids.'
        )
        print(f'[AUILC] {auilc_clamp_note}')
        auilc_hi = new_hi
    auilc_lo = min(auilc_lo, auilc_hi)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.abspath(spec['default_analysis_dir'])
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    df_all, provenance = build_tables(
        rates, args.normals_root, args.predictions_root, csv_stem, pred_glob
    )
    df_all.to_csv(os.path.join(out_dir, 'all_methods_metrics.csv'), index=False)

    missing_audit_lines = print_data_presence_audit(df_all, rates)

    by_rate: Dict[float, Dict[str, Dict[str, float]]] = {}
    for r in rates:
        by_rate[float(r)] = {}
        for mk in ALL_METRIC_BRANCH_KEYS:
            sub = df_all[(df_all['missing_rate'] == r) & (df_all['method_name'] == mk)]
            if len(sub):
                by_rate[float(r)][mk] = {m: float(sub.iloc[0][m]) for m in METRICS}

    # Best selections
    primary_rows = []
    rank_rows = []
    for r in rates:
        tbl = by_rate[float(r)]
        bpm, why = select_best_primary(tbl, r, tie_mult7=args.primary_tie_eps)
        brs, rsvec = select_best_rank_sum(tbl, r)
        primary_rows.append(
            {
                'missing_rate': r,
                'best_method_name': bpm,
                'note': why,
                'Mult_acc_7': tbl[bpm].get('Mult_acc_7', float('nan')),
                'MAE': tbl[bpm].get('MAE', float('nan')),
                'Corr': tbl[bpm].get('Corr', float('nan')),
            }
        )
        rank_rows.append(
            {
                'missing_rate': r,
                'best_method_name': brs,
                'rank_sum': float(np.nansum(rsvec)),
                **{f'rank_{POLICY_BRANCH_KEYS[i]}': float(rsvec[i]) for i in range(len(POLICY_BRANCH_KEYS))},
            }
        )
    pd.DataFrame(primary_rows).to_csv(os.path.join(out_dir, 'best_by_primary_metric.csv'), index=False)
    pd.DataFrame(rank_rows).to_csv(os.path.join(out_dir, 'best_by_rank_sum.csv'), index=False)

    # Policy search (both objectives)
    tau_a1, tau_a2, sc_a, _ = search_policy(rates, by_rate, 'auilc_mult7')
    tau_c1, tau_c2, sc_c, _ = search_policy(rates, by_rate, 'composite')
    pd.DataFrame(
        [
            {
                'objective_name': 'max_auilc_Mult_acc_7',
                'best_tau1': tau_a1,
                'best_tau2': tau_a2,
                'objective_value': sc_a,
                'policy_definition': format_policy_definition(tau_a1, tau_a2),
            },
            {
                'objective_name': 'max_auilc_composite_normalized',
                'best_tau1': tau_c1,
                'best_tau2': tau_c2,
                'objective_value': sc_c,
                'policy_definition': format_policy_definition(tau_c1, tau_c2),
            },
        ]
    ).to_csv(os.path.join(out_dir, 'policy_threshold_search.csv'), index=False)

    # Use Mult_acc_7 AUILC objective for main policy tables / plots unless user wants composite —
    # we document both; primary policy = auilc_mult7
    tau1, tau2 = tau_a1, tau_a2
    policy_rows = []
    for r in sorted(by_rate.keys()):
        mk = policy_pick_method(r, tau1, tau2)
        m = by_rate[r][mk]
        row: Dict[str, Any] = {
            'missing_rate': r,
            'selected_method': mk,
            'selected_slug': METHOD_SLUG_PRIMARY[mk],
        }
        for k in METRICS:
            row[k] = m.get(k, float('nan'))
        policy_rows.append(row)
    pd.DataFrame(policy_rows).to_csv(os.path.join(out_dir, 'policy_metrics_by_missing.csv'), index=False)

    R = np.array(sorted(by_rate.keys()), dtype=float)

    def oracle_curve(metric: str) -> np.ndarray:
        ys = []
        for r in R:
            vals = [by_rate[r][mk].get(metric, float('nan')) for mk in POLICY_BRANCH_KEYS]
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                ys.append(float('nan'))
            elif metric in LOWER_BETTER:
                ys.append(float(np.min(vals)))
            else:
                ys.append(float(np.max(vals)))
        return np.array(ys, dtype=float)

    def method_curve(mk: str, metric: str) -> np.ndarray:
        return np.array([by_rate[r][mk].get(metric, float('nan')) for r in R], dtype=float)

    def policy_curve(metric: str) -> np.ndarray:
        out = []
        for r in R:
            mk = policy_pick_method(r, tau1, tau2)
            out.append(by_rate[r][mk].get(metric, float('nan')))
        return np.array(out, dtype=float)

    auilc_rows: List[Dict[str, Any]] = []
    for mk in list(POLICY_BRANCH_KEYS) + ['dynamic_lte_mstcn']:
        for metric in AUILC_COMPARE_METRICS:
            y = method_curve(mk, metric)
            for lo, hi, tag in [(0.0, 0.6, '0.0_0.6'), (0.1, 0.6, '0.1_0.6')]:
                rr, yy = slice_interval(R, y, lo, hi)
                auilc_rows.append(
                    {
                        'curve': mk,
                        'metric': metric,
                        'interval': tag,
                        'auilc': trapezoid_auilc(rr, yy),
                        'better_direction': 'higher' if metric in HIGHER_BETTER else 'lower',
                    }
                )
            if args.extend_auilc_to_1 and np.nanmax(R) >= 1.0 - 1e-9:
                rr, yy = slice_interval(R, y, 0.0, 1.0)
                auilc_rows.append(
                    {
                        'curve': mk,
                        'metric': metric,
                        'interval': '0.0_1.0',
                        'auilc': trapezoid_auilc(rr, yy),
                        'better_direction': 'higher' if metric in HIGHER_BETTER else 'lower',
                    }
                )
    for name, curve_fn in [
        ('oracle_best_point', oracle_curve),
        ('regime_policy', policy_curve),
    ]:
        for metric in AUILC_COMPARE_METRICS:
            y = curve_fn(metric)
            for lo, hi, tag in [(0.0, 0.6, '0.0_0.6'), (0.1, 0.6, '0.1_0.6')]:
                rr, yy = slice_interval(R, y, lo, hi)
                auilc_rows.append(
                    {
                        'curve': name,
                        'metric': metric,
                        'interval': tag,
                        'auilc': trapezoid_auilc(rr, yy),
                        'better_direction': 'higher' if metric in HIGHER_BETTER else 'lower',
                    }
                )
            if args.extend_auilc_to_1 and np.nanmax(R) >= 1.0 - 1e-9:
                rr, yy = slice_interval(R, y, 0.0, 1.0)
                auilc_rows.append(
                    {
                        'curve': name,
                        'metric': metric,
                        'interval': '0.0_1.0',
                        'auilc': trapezoid_auilc(rr, yy),
                        'better_direction': 'higher' if metric in HIGHER_BETTER else 'lower',
                    }
                )

    pd.DataFrame(auilc_rows).to_csv(os.path.join(out_dir, 'policy_auilc_summary.csv'), index=False)

    # Slopes (fixed branches + regime_policy curve)
    slope_rows = []
    avg_rows = []
    for mk in list(POLICY_BRANCH_KEYS) + ['dynamic_lte_mstcn']:
        for metric in METRICS:
            y = method_curve(mk, metric)
            sl = slopes_for_method(R, y)
            for s in sl:
                slope_rows.append({'method_name': mk, 'metric': metric, **s})
            slopes_list = sl
            avg_rows.append(
                {
                    'method_name': mk,
                    'metric': metric,
                    'mean_slope_all_segments': float(
                        np.mean([s['slope'] for s in slopes_list])
                    )
                    if slopes_list
                    else float('nan'),
                    'mean_abs_slope_all_segments': float(
                        np.mean([abs(s['slope']) for s in slopes_list])
                    )
                    if slopes_list
                    else float('nan'),
                }
            )
    for metric in METRICS:
        y = policy_curve(metric)
        sl = slopes_for_method(R, y)
        for s in sl:
            slope_rows.append({'method_name': 'regime_policy', 'metric': metric, **s})
        slopes_list = sl
        avg_rows.append(
            {
                'method_name': 'regime_policy',
                'metric': metric,
                'mean_slope_all_segments': float(np.mean([s['slope'] for s in slopes_list]))
                if slopes_list
                else float('nan'),
                'mean_abs_slope_all_segments': float(
                    np.mean([abs(s['slope']) for s in slopes_list])
                )
                if slopes_list
                else float('nan'),
            }
        )
    pd.DataFrame(slope_rows).to_csv(os.path.join(out_dir, 'slope_by_method.csv'), index=False)
    pd.DataFrame(avg_rows).to_csv(os.path.join(out_dir, 'average_degradation_slope.csv'), index=False)

    # Highlights (Mult_acc_7): slowest decline = largest mean slope among Acc-like metrics
    def stability_highlights(metric: str = 'Mult_acc_7'):
        higher_better = metric in HIGHER_BETTER
        rep = {}
        for mk in POLICY_BRANCH_KEYS:
            y = method_curve(mk, metric)
            sl = slopes_for_method(R, y)
            rep[mk] = sl

        def pick_range(lo: float, hi_bound: float):
            best_mk, best_v = None, float('-inf') if higher_better else float('inf')
            for mk in POLICY_BRANCH_KEYS:
                v = mean_slope_in_range(rep[mk], lo, hi_bound, higher_better)
                if math.isnan(v):
                    continue
                if higher_better:
                    if v > best_v:
                        best_v, best_mk = v, mk
                else:
                    if v < best_v:
                        best_v, best_mk = v, mk
            if best_mk is None:
                return None, float('nan')
            return best_mk, best_v

        m13, v13 = pick_range(0.1, 0.3)
        m35, v35 = pick_range(0.3, 0.5)
        m16, v16 = pick_range(0.1, 0.6)
        return {
            'slowest_decline_0.1_0.3_Mult_acc_7': m13,
            'mean_slope_band_0.1_0.3': v13,
            'slowest_decline_0.3_0.5_Mult_acc_7': m35,
            'mean_slope_band_0.3_0.5': v35,
            'slowest_decline_0.1_0.6_Mult_acc_7': m16,
            'mean_slope_band_0.1_0.6': v16,
        }

    highlights = stability_highlights('Mult_acc_7')

    policy_branch_lines = []
    for r in sorted(by_rate.keys()):
        mk = policy_pick_method(r, tau1, tau2)
        policy_branch_lines.append(
            f'- missing_rate **{r}** → `{mk}` (slug `{METHOD_SLUG_PRIMARY[mk]}`)'
        )
    policy_branch_md = '\n'.join(policy_branch_lines)

    # Paper figures: main curves to main_x_hi (policy = branch interp + flat tail); AUILC bars on [auilc_lo, auilc_hi]
    plt = setup_matplotlib()

    R_sorted = np.array(sorted(by_rate.keys()), dtype=float)

    def interp_branch_native(mk: str, met: str, rq: np.ndarray) -> np.ndarray:
        yv = np.array([by_rate[float(rr)][mk].get(met, float('nan')) for rr in R_sorted], dtype=float)
        return interp_rates_flat_extrap(R_sorted, yv, np.asarray(rq, dtype=float))

    def interp_policy_native(met: str, rq: np.ndarray) -> np.ndarray:
        rq_flat = np.asarray(rq, dtype=float).reshape(-1)
        out = np.empty(rq_flat.shape[0], dtype=float)
        for i in range(rq_flat.shape[0]):
            mk = policy_pick_method(float(rq_flat[i]), tau1, tau2)
            out[i] = float(interp_branch_native(mk, met, np.array([rq_flat[i]]))[0])
        return out

    def auilc_trapezoid_interval_flat(mk: str, metric: str, lo: float, hi: float) -> float:
        inner_vals = R_sorted[(R_sorted >= lo - 1e-12) & (R_sorted <= hi + 1e-12)]
        knots = np.sort(np.unique(np.concatenate([[lo], inner_vals, [hi]])))
        knots = knots[(knots >= lo - 1e-12) & (knots <= hi + 1e-12)]
        if knots.size < 2:
            return float('nan')
        if mk == 'regime_policy':
            yy = interp_policy_native(metric, knots)
        else:
            yy = interp_branch_native(mk, metric, knots)
        return trapezoid_auilc(knots, yy)

    R_main_plot = np.array(
        [t for t in MAIN_DISPLAY_X_TICKS if -1e-12 <= t <= main_x_hi + 1e-12],
        dtype=float,
    )
    y_baseline_main = {m: interp_branch_native('text', m, R_main_plot) for m in ('Mult_acc_7', 'MAE', 'Corr')}
    y_policy_main = {m: interp_policy_native(m, R_main_plot) for m in ('Mult_acc_7', 'MAE', 'Corr')}

    fig_id = ds
    plot_figure_main_policy_vs_baseline_mosi(
        plt,
        R_main_plot,
        y_baseline_main,
        y_policy_main,
        os.path.join(fig_dir, f'fig_main_policy_vs_baseline_{fig_id}'),
        x_axis_hi=main_x_hi,
        dataset_display=dataset_display,
    )

    auilc_by_method: Dict[str, Dict[str, float]] = {}
    for mk in ('text', 'dynamic_rule', 'text_mstcn', 'dynamic_lte_mstcn', 'regime_policy'):
        auilc_by_method[mk] = {
            'Mult_acc_7': auilc_trapezoid_interval_flat(mk, 'Mult_acc_7', auilc_lo, auilc_hi),
            'Corr': auilc_trapezoid_interval_flat(mk, 'Corr', auilc_lo, auilc_hi),
            'MAE': auilc_trapezoid_interval_flat(mk, 'MAE', auilc_lo, auilc_hi),
        }

    plot_figure_auilc_summary_mosi(
        plt,
        auilc_by_method,
        os.path.join(fig_dir, f'fig_auilc_summary_{fig_id}'),
        auilc_lo,
        auilc_hi,
        dataset_display=dataset_display,
    )

    plot_figure_policy_schedule_mosi(
        plt,
        tau1,
        tau2,
        lambda r: policy_pick_method(float(r), tau1, tau2),
        os.path.join(fig_dir, f'fig_policy_schedule_{fig_id}'),
        x_axis_hi=main_x_hi,
        dataset_display=dataset_display,
    )

    y_baseline = {m: method_curve('text', m) for m in ('Mult_acc_7', 'MAE', 'Corr')}
    y_dynamic = {m: method_curve('dynamic_rule', m) for m in ('Mult_acc_7', 'MAE', 'Corr')}
    y_mstcn = {m: method_curve('text_mstcn', m) for m in ('Mult_acc_7', 'MAE', 'Corr')}

    plot_figure_pairwise_baseline_vs_method_mosi(
        plt,
        R,
        y_baseline,
        y_dynamic,
        'dynamic_rule',
        'Baseline vs Ours-Dynamic',
        os.path.join(fig_dir, f'fig_pairwise_dynamic_vs_baseline_{fig_id}'),
    )
    plot_figure_pairwise_baseline_vs_method_mosi(
        plt,
        R,
        y_baseline,
        y_mstcn,
        'text_mstcn',
        'Baseline vs Ours-MSTCN',
        os.path.join(fig_dir, f'fig_pairwise_mstcn_vs_baseline_{fig_id}'),
    )

    plot_figure_delta_improvement_mosi(
        plt,
        R,
        y_baseline,
        y_dynamic,
        y_mstcn,
        os.path.join(fig_dir, f'fig_delta_improvement_{fig_id}'),
    )

    # JSON summary
    summary = {
        'dataset': ds,
        'dataset_display': dataset_display,
        'csv_stem': csv_stem,
        'rates_requested': rates,
        'provenance': provenance,
        'policy_primary_mult_acc_7_auilc': {
            'tau1': tau_a1,
            'tau2': tau_a2,
            'objective_score': sc_a,
            'policy_definition': format_policy_definition(tau_a1, tau_a2),
        },
        'policy_composite_normalized_auilc': {
            'tau1': tau_c1,
            'tau2': tau_c2,
            'objective_score': sc_c,
            'policy_definition': format_policy_definition(tau_c1, tau_c2),
        },
        'figure_settings': {
            'auilc_bar_interval': [auilc_lo, auilc_hi],
            'auilc_bar_interval_label': fmt_auilc_interval_label(auilc_lo, auilc_hi),
            'auilc_hi_requested': auilc_hi_requested,
            'allow_extrapolate_auilc': bool(args.allow_extrapolate_auilc),
            'main_and_schedule_x_hi': main_x_hi,
        },
        'missing_metric_cells': missing_audit_lines,
        'auilc_clamp_note': auilc_clamp_note or None,
        'highlights_Mult_acc_7_mean_slope_bands': highlights,
        'oracle_disclaimer': (
            'oracle_best_point chooses the best of {text, dynamic_rule, text_mstcn} at each rate using '
            'recorded test metrics — an oracle upper bound, not a single model.'
        ),
        'regime_policy_note': (
            'Thresholds tau1/tau2 are searched on the same recorded curves by default (test-set regime analysis). '
            'Lock thresholds on validation for a formal deployable policy.'
        ),
    }
    with open(os.path.join(out_dir, 'analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(_sanitize_for_json(summary), f, indent=2)

    auilc_df = pd.DataFrame(auilc_rows)
    _sel = auilc_df['curve'].isin(AUILC_SUMMARY_CURVES) & auilc_df['metric'].isin(
        AUILC_COMPARE_METRICS
    )
    auilc_block = auilc_df.loc[_sel].sort_values(['curve', 'metric', 'interval']).to_string(index=False)

    missing_audit_md = (
        '\n'.join(missing_audit_lines)
        if missing_audit_lines
        else '*No missing cells: Mult_acc_7 / MAE / Corr present for every listed method and rate (see console table).*'
    )
    auilc_clamp_md = (
        auilc_clamp_note
        if auilc_clamp_note
        else '*No clamp: requested AUILC upper bound is at or below the maximum observed missing rate, or `--allow-extrapolate-auilc` was set.*'
    )

    # Markdown report (static body avoids f-string parsing LaTeX braces)
    report = """# __DATASET_DISPLAY__ regime-aware policy analysis (three policy branches)

## Dataset and inputs

- **Dataset:** __DATASET_DISPLAY__ (`__CSV_STEM__-<missing>.csv` under each normals subfolder).
- **MOSEI vs MOSI:** this script **never** mixes stems: `--dataset mosei` only reads `mosei-regression-*.csv` and `predictions_mosei_fc_*.csv`.

## Data availability audit

__MISSING_AUDIT__

## AUILC integration note

__AUILC_CLAMP_MD__

## Role of each curve

1. **oracle_best_point** — At each missing rate, picks the best among **`text`**, **`dynamic_rule`**, **`text_mstcn`** using the recorded **test** metrics. This is a **test oracle upper bound** on segmentation-by-rate; it is **not** a single deployable model and must not be reported as such.

2. **regime_policy** — Follows the fixed template below with thresholds **`tau1`, `tau2`** chosen by search on the **same recorded curves** (default in this script: **test-set regime analysis**). This is still **analysis / policy study**, not a classical single-model test score, unless thresholds are locked on **validation** and then applied once on test.

3. **Proper reporting** — To present a **legitimate** regime policy: choose `tau1`, `tau2` on a **validation** protocol, freeze them, then evaluate **one** policy on **test**.

---

## Automatically selected thresholds

**Objective A (default for tables/figures):** maximize **AUILC** of **Mult_acc_7** along the policy curve on the observed rate grid.

**Objective B (supplementary):** maximize **AUILC** of a **composite** score per rate:
\\[ z(\\mathrm{Mult\\_acc\\_5}) + z(\\mathrm{Mult\\_acc\\_7}) + z(\\mathrm{Corr}) - z\_{\\mathrm{bad}}(\\mathrm{MAE}) \\]
where each \\(z\\) is **min–max normalized across the three branches** at that rate.

| Item | Objective A (Mult_acc_7 AUILC) | Objective B (composite AUILC) |
|------|-------------------------------|-------------------------------|
| **best_tau1** | __TAU_A1__ | __TAU_C1__ |
| **best_tau2** | __TAU_A2__ | __TAU_C2__ |
| **objective_value** | __SC_A__ | __SC_C__ |
| **policy_definition** | __POLICY_DEF_A__ | __POLICY_DEF_C__ |

**Primary policy used in `policy_metrics_by_missing.csv` and figures:** Objective **A**.

---

## Policy definition (primary)

__POLICY_DEF_A__

Branches:

- **`dynamic_rule`** → normals slug **`dynamic_missing_router_rule`** (fallback **`dynamic_missing`**).
- **`text_mstcn`** → **`text_lte_mstcn`**.
- **`text`** → **`text`** (legacy LTE).

---

## Which branch at each missing rate (primary policy)

__POLICY_BRANCH_MD__

---

## AUILC summary (subset)

Curves: `text`, `dynamic_rule`, `text_mstcn`, `regime_policy`, `oracle_best_point`.  
Metrics: Mult_acc_5, Mult_acc_7, MAE, Corr.  
Intervals: **0.0–0.6**, **0.1–0.6** (and **0.0–1.0** if `--extend-auilc-to-1` and rates reach 1).

```
__AUILC_BLOCK__
```

Higher AUILC is better for accuracy / correlation metrics; **lower** is better for MAE / Loss.

---

## Slopes (Mult_acc_7)

Mean slope over segments whose **midpoint** lies in each band (larger = slower decline for Mult_acc_7):

| Band | Slowest decline (method) | Mean slope in band |
|------|--------------------------|--------------------|
| 0.1 – 0.3 | __H13__ | __V13__ |
| 0.3 – 0.5 | __H35__ | __V35__ |
| 0.1 – 0.6 | __H16__ | __V16__ |

Full segment table: `slope_by_method.csv`. Aggregates: `average_degradation_slope.csv`.

---

## Auxiliary CSV exports

- `best_by_primary_metric.csv` / `best_by_rank_sum.csv` — reference only (three branches).
- `all_methods_metrics.csv` — four fixed branches (plus policy rows are not duplicated here; see `policy_metrics_by_missing.csv`).

---

## Paper writing checklist

- Do **not** describe **oracle_best_point** as a single trained model’s test result.
- If **tau1/tau2** were searched on test (this script’s default), describe **regime_policy** as **analysis / policy study** or **upper-bound-style regime exploration**, not a fully validatable reported score.
- For a **defensible** claim: **validate** thresholds, then report **one** locked policy on test.

---

## Data sources

Normals prefer `results/results/normals/<subdir>/__CSV_STEM__-<miss>.csv` (means ×100 in cells).  
Fallback: recompute from `results/predictions/miss_<miss>_<center>_<lte>/` using the dataset-specific prediction glob (MOSI: `predictions_mosi_fc_*test*.csv`; MOSEI: `predictions_mosei_fc_*test*.csv`). **Loss** is not recovered from row-wise export (NaN).

---

## Figure guide (paper-oriented exports)

All experiments use **symmetric missing rates** (the same missing rate is applied jointly to the three modalities).

1. **Ours-Policy is not a separate trained model** — it is a **missing-rate-aware branch selection policy** that switches among the three fixed branches using thresholds **τ₁**, **τ₂** (searched on the observed rate grid in this script by default).

2. **Main performance figure** — `fig_main_policy_vs_baseline___FIG_ID__.png` / `.pdf` plots **Baseline** vs **Ours-Policy** from missing rate **0** up to **__MAIN_X_HI__** on the x-axis (ticks include 0.15–0.55 and 0.7–1.0 when in range). Where no measurements exist yet, curves use **flat extrapolation** of each branch from the last observed rate. **Ours-Policy** at any rate uses the metric of the **selected branch**; when that branch is **Baseline** (`text`), the orange curve coincides with the blue curve. In particular, **for high missing rates, Ours-Policy falls back to the baseline branch, so the two curves may overlap** — this is expected.

3. **AUILC bar figure** — `fig_auilc_summary___FIG_ID__.png` / `.pdf` reports trapezoid **AUILC** on **__AUILC_INTERVAL_LABEL__** (configurable via `--auilc-lo` / `--auilc-hi` or `--auilc-interval`). **Bars include Baseline, Ours-Dynamic, Ours-MSTCN, Ours-Dynamic+MSTCN, and Ours-Policy.** Without `--allow-extrapolate-auilc`, the script **does not** silently integrate past the maximum observed missing rate (see *AUILC integration note* above). With `--allow-extrapolate-auilc`, each branch is **held flat** beyond the last observed rate. For **Mult-Acc7** and **Corr**, higher integrated area is better; for **MAE**, **lower** integrated area is better.

4. **Policy schedule** — `fig_policy_schedule___FIG_ID__.png` / `.pdf` is a **step chart** on **[0, __MAIN_X_HI__]** showing which branch the policy selects, with **τ₁**, **τ₂** as vertical guides (when finite).

5. **Pairwise supplements** — `fig_pairwise_*___FIG_ID__` show **raw** Baseline vs **Ours-Dynamic** / **Ours-MSTCN** as **two lines only** (no red/green shading between curves).

6. **Delta supplement** — `fig_delta_improvement___FIG_ID__.png` / `.pdf` highlights **relative improvement vs Baseline** with **green** for Δ>0 and **red** for Δ<0 (MAE: Δ = MAE(Baseline) − MAE(Method), so positive is still better).

7. **Legend mapping** — `text` → **Baseline**; `dynamic_rule` → **Ours-Dynamic**; `text_mstcn` → **Ours-MSTCN**; `dynamic_lte_mstcn` → **Ours-Dynamic+MSTCN**; **Ours-Policy** = regime policy curve. Optional caption text: Baseline = MFMB-Net with text-centered local fusion; Ours-Dynamic = proposed dynamic-anchor branch; Ours-MSTCN = proposed MSTCN-enhanced branch; Ours-Policy = missing-rate-aware branch selection.

8. **x-axis** — Curve panels use **`set_xlim(x_min, x_max)`** and **`margins(x=0)`**. Tick labels use the **0.00, 0.10, 0.15, …** format; every **observed** missing rate in the span is included in the tick set.

---

## Figures produced (under `figures/`)

| File | Role | Visual form |
|------|------|----------------|
| `fig_main_policy_vs_baseline___FIG_ID__` | Main | Line (1×3), x to __MAIN_X_HI__ |
| `fig_auilc_summary___FIG_ID__` | Main | Bar (1×3), five methods + policy; AUILC on __AUILC_INTERVAL_LABEL__ |
| `fig_policy_schedule___FIG_ID__` | Main | Step + τ; x to __MAIN_X_HI__ |
| `fig_pairwise_dynamic_vs_baseline___FIG_ID__` | Supplement | Two lines (1×3) |
| `fig_pairwise_mstcn_vs_baseline___FIG_ID__` | Supplement | Two lines (1×3) |
| `fig_delta_improvement___FIG_ID__` | Supplement | Δ lines + fill vs 0 (1×3) |

Each base name is saved as **`.png`** (300 dpi, `bbox_inches='tight'`) and **`.pdf`**.

Y-axes for raw metric curves use the **×100** display convention. Serif fonts via matplotlib rcParams.

"""
    report = (
        report.replace('__PRIMARY_TIE__', str(args.primary_tie_eps))
        .replace('__DATASET_DISPLAY__', dataset_display)
        .replace('__CSV_STEM__', csv_stem)
        .replace('__FIG_ID__', ds)
        .replace('__MISSING_AUDIT__', missing_audit_md)
        .replace('__AUILC_CLAMP_MD__', auilc_clamp_md)
        .replace('__TAU_A1__', str(tau_a1))
        .replace('__TAU_A2__', str(tau_a2))
        .replace('__SC_A__', str(sc_a))
        .replace('__TAU_C1__', str(tau_c1))
        .replace('__TAU_C2__', str(tau_c2))
        .replace('__SC_C__', str(sc_c))
        .replace('__POLICY_DEF_A__', format_policy_definition(tau_a1, tau_a2))
        .replace('__POLICY_DEF_C__', format_policy_definition(tau_c1, tau_c2))
        .replace('__POLICY_BRANCH_MD__', policy_branch_md)
        .replace('__AUILC_BLOCK__', auilc_block)
        .replace('__AUILC_INTERVAL_LABEL__', fmt_auilc_interval_label(auilc_lo, auilc_hi))
        .replace('__MAIN_X_HI__', str(main_x_hi))
        .replace('__H13__', str(highlights['slowest_decline_0.1_0.3_Mult_acc_7']))
        .replace('__V13__', str(highlights['mean_slope_band_0.1_0.3']))
        .replace('__H35__', str(highlights['slowest_decline_0.3_0.5_Mult_acc_7']))
        .replace('__V35__', str(highlights['mean_slope_band_0.3_0.5']))
        .replace('__H16__', str(highlights['slowest_decline_0.1_0.6_Mult_acc_7']))
        .replace('__V16__', str(highlights['mean_slope_band_0.1_0.6']))
    )
    with open(os.path.join(out_dir, 'analysis_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'Wrote analysis under {out_dir}')
    print(json.dumps(_sanitize_for_json(highlights), indent=2))
    print(f'Policy (AUILC Mult_acc_7): tau1={tau_a1}, tau2={tau_a2}')
    print(
        f'Figures: main/schedule x∈[0,{main_x_hi}]; AUILC bars on {fmt_auilc_interval_label(auilc_lo, auilc_hi)} '
        f'({dataset_display})'
    )


if __name__ == '__main__':
    main()
