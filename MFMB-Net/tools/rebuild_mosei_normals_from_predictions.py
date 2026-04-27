#!/usr/bin/env python3
"""
Offline rebuild of mosei-regression-*.csv from exported **MOSEI** prediction CSVs
(three seeds per miss x fusion_center x lte) without retraining.

**Only** files named `predictions_mosei_fc_*.csv` with a sidecar `*_meta.json` passing
`datasetName=mosei`, `MOSEI` in `dataPath_runtime`, and `n_samples != 686` are used.
MOSI exports (`predictions_mosi_fc_*`, 686 samples) are **never** written into
mosei-regression-*.csv; they are logged as rejected.

Metrics match utils.metricsTop.__eval_mosei_regression (MOSI/MOSEI share it).

Test Loss (batch-averaged L1 in do_test) is not recoverable from per-row exports;
new/rewritten cells use "(N/A, N/A)". Skips overwrite if the first eight columns
match within --tol (compare_loss=False).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MISSING_RATES = [f'{(i / 10.0):.1f}' for i in range(0, 7)]
CENTERS = ['text', 'dynamic_missing']
LTES = ['legacy', 'mstcn']
SEEDS = ['111', '1111', '11111']

RECOVER = [
    'Has0_acc_2',
    'Has0_F1_score',
    'Non0_acc_2',
    'Non0_F1_score',
    'Mult_acc_5',
    'Mult_acc_7',
    'MAE',
    'Corr',
]
CSV_COLUMNS = ['Model'] + RECOVER + ['Loss']
LOSS_UNSET = '(N/A, N/A)'
DEFAULT_TOL = 0.5
# Typical MOSI test count; must not be used for mosei-regression rebuild
MOSI_TEST_N = 686


def _accuracy_eq(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float(np.mean(y_true == y_pred))


def _f1_weighted_binary(y_true, y_pred) -> float:
    """
    Sklearn-like f1 average='weighted' for {0,1} labels
    (matches sklearn.metrics.f1_score on binary 0/1 from bool casts).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = (0, 1)
    wsum = 0.0
    tsum = 0.0
    for lab in labels:
        tp = int(np.sum((y_true == lab) & (y_pred == lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        sup = int(np.sum(y_true == lab))
        wsum += f1 * sup
        tsum += sup
    if tsum == 0:
        return 0.0
    return float(wsum / tsum)


def _fusion_center_slug(center: str, lte: str) -> str:
    if lte == 'legacy':
        return center
    return f'{center}_lte_{lte}'


def _multiclass_acc(preds: np.ndarray, truths: np.ndarray) -> float:
    return float(np.sum(np.round(preds) == np.round(truths)) / float(len(truths)))


_METRICS_BACKEND: Optional[str] = None


def eval_mosei_regression_from_arrays(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """
    Match utils.metricsTop.__eval_mosei_regression, in order:
    1) torch + MetricsTop (same as training)
    2) numpy + sklearn (same code path as in metricsTop.py)
    3) numpy-only fallback (best-effort if torch/sklearn missing)
    """
    global _METRICS_BACKEND
    test_preds = np.asarray(pred, dtype=np.float64).reshape(-1)
    test_truth = np.asarray(truth, dtype=np.float64).reshape(-1)

    # (1) Preferred: same stack as do_test
    try:
        import torch
        from utils.metricsTop import MetricsTop

        p = torch.from_numpy(test_preds).float().view(-1, 1)
        t = torch.from_numpy(test_truth).float().view(-1, 1)
        out = MetricsTop('regression').getMetics('MOSEI')(p, t)
        _METRICS_BACKEND = 'torch+MetricsTop'
        return {k: float(v) for k, v in out.items()}
    except Exception:
        pass

    # (2) Numpy + sklearn: literal port of __eval_mosei_regression
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        accuracy_score = None  # type: ignore
        f1_score = None  # type: ignore
    if accuracy_score is not None and f1_score is not None:
        test_preds_a7 = np.clip(test_preds, -3.0, 3.0)
        test_truth_a7 = np.clip(test_truth, -3.0, 3.0)
        test_preds_a5 = np.clip(test_preds, -2.0, 2.0)
        test_truth_a5 = np.clip(test_truth, -2.0, 2.0)

        mae = float(np.mean(np.abs(test_preds - test_truth)))
        ccm = np.corrcoef(test_preds, test_truth)
        corr = float(ccm[0, 1])
        mult_a7 = _multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = _multiclass_acc(test_preds_a5, test_truth_a5)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        if len(non_zeros) == 0:
            nza, nzf = float('nan'), float('nan')
        else:
            non_zeros_binary_truth = (test_truth[non_zeros] > 0)
            non_zeros_binary_preds = (test_preds[non_zeros] > 0)
            nza = float(accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth))
            nzf = float(
                f1_score(
                    non_zeros_binary_preds, non_zeros_binary_truth, average='weighted'
                )
            )

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = float(accuracy_score(binary_preds, binary_truth))
        f_sc = float(f1_score(binary_preds, binary_truth, average='weighted'))
        _METRICS_BACKEND = 'numpy+sklearn (metricsTop-equivalent)'
        return {
            'Has0_acc_2': round(acc2, 4),
            'Has0_F1_score': round(f_sc, 4),
            'Non0_acc_2': round(nza, 4),
            'Non0_F1_score': round(nzf, 4),
            'Mult_acc_5': round(mult_a5, 4),
            'Mult_acc_7': round(mult_a7, 4),
            'MAE': round(mae, 4),
            'Corr': round(corr, 4),
        }

    # (3) Pure numpy
    _METRICS_BACKEND = 'numpy-fallback (install torch+sklearn for exact match)'
    mae = float(np.mean(np.abs(test_preds - test_truth)))
    ccm = np.corrcoef(test_preds, test_truth)
    corr = float(ccm[0, 1]) if ccm.size > 1 and not np.isnan(ccm[0, 1]) else float('nan')
    test_preds_a7 = np.clip(test_preds, -3.0, 3.0)
    test_truth_a7 = np.clip(test_truth, -3.0, 3.0)
    test_preds_a5 = np.clip(test_preds, -2.0, 2.0)
    test_truth_a5 = np.clip(test_truth, -2.0, 2.0)
    mult_a7 = _multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = _multiclass_acc(test_preds_a5, test_truth_a5)
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0], dtype=int)
    if len(non_zeros) == 0:
        nza, nzf = float('nan'), float('nan')
    else:
        nzt = (test_truth[non_zeros] > 0).astype(int)
        nzp = (test_preds[non_zeros] > 0).astype(int)
        nza = _accuracy_eq(nzt, nzp)
        nzf = _f1_weighted_binary(nzt, nzp)
    bint = (test_truth >= 0).astype(int)
    binp = (test_preds >= 0).astype(int)
    acc2 = _accuracy_eq(bint, binp)
    f_score = _f1_weighted_binary(bint, binp)
    cr = float('nan') if (isinstance(corr, float) and np.isnan(corr)) else round(corr, 4)
    return {
        'Has0_acc_2': round(acc2, 4),
        'Has0_F1_score': round(f_score, 4),
        'Non0_acc_2': round(nza, 4),
        'Non0_F1_score': round(nzf, 4),
        'Mult_acc_5': round(mult_a5, 4),
        'Mult_acc_7': round(mult_a7, 4),
        'MAE': round(mae, 4),
        'Corr': cr,
    }


def _load_meta_json(csv_path: str) -> Optional[Dict[str, Any]]:
    meta_path = csv_path.replace('.csv', '_meta.json') if csv_path.endswith(
        '.csv'
    ) else csv_path + '_meta.json'
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def is_valid_mosei_meta(meta: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    if not meta:
        return False, 'missing or empty meta'
    if meta.get('datasetName') != 'mosei':
        return False, f"datasetName={meta.get('datasetName')!r} (expected 'mosei')"
    dp = str(meta.get('dataPath_runtime') or '')
    if 'MOSEI' not in dp:
        return False, f"dataPath_runtime must contain 'MOSEI', got {dp!r}"
    n = meta.get('n_samples')
    if n is None:
        return False, 'n_samples missing in meta'
    if n == MOSI_TEST_N:
        return (
            False,
            f'n_samples=={MOSI_TEST_N} (MOSI-scale test), refused for mosei-regression',
        )
    return True, 'ok'


def find_mosei_prediction_csv(
    pred_dir: str, center: str, seed: str, root: str
) -> Tuple[Optional[str], List[str]]:
    """
    Match predictions_mosei_fc_{center}_seed{seed}_test*.csv with valid meta.
    If predictions_mosei_fc_* exist but meta fails, log reasons; if only
    predictions_mosi_fc_* exist, add explicit rejection messages.
    """
    notes: List[str] = []
    if not os.path.isdir(pred_dir):
        return None, notes
    mosei_pat = os.path.join(
        pred_dir, f'predictions_mosei_fc_{center}_seed{seed}_test*.csv'
    )
    for f in sorted(glob.glob(mosei_pat)):
        if not (os.path.isfile(f) and os.path.getsize(f) > 0):
            continue
        meta = _load_meta_json(f)
        ok, reason = is_valid_mosei_meta(meta)
        if ok:
            return f, notes
        notes.append(
            f"reject {os.path.relpath(f, root)}: {reason}"
        )
    mosi_pat = os.path.join(
        pred_dir, f'predictions_mosi_fc_{center}_seed{seed}_test*.csv'
    )
    for f in sorted(glob.glob(mosi_pat)):
        if os.path.isfile(f) and os.path.getsize(f) > 0:
            rel = os.path.relpath(f, root)
            notes.append(
                f"MOSI prediction ignored for mosei-regression (use MOSEI export + meta): {rel}"
            )
    return None, notes


def _read_pred_truth(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if 'pred_raw' not in df.columns or 'truth' not in df.columns:
        raise ValueError(f'{path}: need columns pred_raw, truth')
    return df['pred_raw'].values, df['truth'].values


def per_seed_metrics(
    pred_dir: str,
    center: str,
    rel_paths: List[str],
    root: str,
    all_notes: List[str],
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for s in SEEDS:
        p, notes = find_mosei_prediction_csv(pred_dir, center, s, root)
        all_notes.extend(notes)
        if p is None:
            if verbose:
                print(
                    f'[rebuild_mosei_normals] no valid MOSEI export seed={s} '
                    f'dir={os.path.relpath(pred_dir, root)}',
                    file=sys.stderr,
                )
            continue
        rel_paths.append(os.path.relpath(p, root))
        pr, tr = _read_pred_truth(p)
        out[s] = eval_mosei_regression_from_arrays(pr, tr)
    return out


def _agg_mean_std(values: List[float]) -> Tuple[float, float]:
    clean = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return float('nan'), float('nan')
    arr = np.array(clean, dtype=np.float64)
    return (
        round(float(np.mean(arr)) * 100.0, 2),
        round(float(np.std(arr, ddof=0)) * 100.0, 2),
    )


def build_row(seed_m: Dict[str, Dict[str, float]], model_name: str) -> List[str]:
    if len(seed_m) < 3:
        raise ValueError('Need 3 seeds')
    row: List[str] = [model_name]
    for k in RECOVER:
        ser = [seed_m[s][k] for s in SEEDS if s in seed_m]
        m, s = _agg_mean_std(ser)
        row.append(f'({m}, {s})')
    row.append(LOSS_UNSET)
    return row


def _parse_pair_str(cell: Any) -> Optional[Tuple[float, float]]:
    if cell is None:
        return None
    t = str(cell).strip().strip('"').strip("'")
    m = re.match(r'^\s*\(\s*([^,()]+?)\s*,\s*([^)]+?)\s*\)\s*$', t)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    if a.upper() == 'N/A' or b.upper() == 'N/A':
        return None
    try:
        return float(a), float(b)
    except ValueError:
        return None


def rows_close(
    a: List[str], b: List[str], *, tol: float, compare_loss: bool
) -> bool:
    if len(a) != len(b) or a[0] != b[0]:
        return False
    n = 9 if compare_loss else 8
    for i in range(1, 1 + n):
        pa, pb = _parse_pair_str(a[i]), _parse_pair_str(b[i])
        if pa is None and pb is None:
            continue
        if pa is None or pb is None:
            return False
        if abs(pa[0] - pb[0]) > tol or abs(pa[1] - pb[1]) > tol:
            return False
    return True


def is_file_bad(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return True
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return True
    return df.shape[0] == 0 or df.shape[1] < 10


def read_first_mfmb_row(path: str) -> Optional[List[str]]:
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or len(header) < 10:
            return None
        for row in r:
            if not row or len(row) < 10:
                continue
            if (row[0] or '').strip() == 'mfmb_net':
                return row
        for row in r:
            if row and len(row) >= 10:
                return row
    return None


def write_csv(path: str, data_row: List[str]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(CSV_COLUMNS)
        w.writerow(data_row)


def scan_meta_any(pred_paths: List[str], root: str) -> Optional[Dict[str, Any]]:
    for p in pred_paths:
        ap = os.path.join(root, p) if not os.path.isabs(p) else p
        mp = ap.replace('.csv', '_meta.json')
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp, encoding='utf-8') as f:
                j = json.load(f)
            return {
                'n_samples': j.get('n_samples'),
                'datasetName': j.get('datasetName'),
                'dataPath_runtime': j.get('dataPath_runtime'),
            }
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return None


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default=ROOT, help='Project root')
    ap.add_argument('--tol', type=float, default=DEFAULT_TOL)
    ap.add_argument(
        '--force', action='store_true', help='Always overwrite regardless of match'
    )
    ap.add_argument('--dry-run', action='store_true', help='No file writes')
    ap.add_argument(
        '--report',
        default='results/results/normals/rebuild_mosei_normals_report.json',
    )
    ap.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-seed messages to stderr (default: one line per failed combo only)',
    )
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    dnorm = os.path.join(root, 'results', 'results', 'normals')
    out_report = os.path.join(root, args.report)

    rep: Dict[str, Any] = {
        'project_root': root,
        'strict_mosei_source': {
            'filename_glob': 'predictions_mosei_fc_{center}_seed{seed}_test*.csv',
            'meta': 'datasetName=mosei, MOSEI in dataPath_runtime, n_samples != 686',
        },
        'naming_note': (
            "MOSI training exports: predictions_mosi_fc_* (datasetName=mosi). "
            "MOSEI training exports: predictions_mosei_fc_* (datasetName=mosei). "
            "mosei-regression-*.csv is rebuilt only from the latter, validated by meta."
        ),
        'loss_note': 'Loss = batch-mean L1 in training; not reproduced here; (N/A, N/A) for new rows.',
        'mosei_prediction_files_accepted': 0,
        'rejected_mosi_for_mosei_normals': [],
        'combinations': [],
        'normals_wrote': [],
        'normals_unchanged': [],
        'incomplete': [],
    }

    n_pred = 0
    for m in MISSING_RATES:
        for c in CENTERS:
            for lte in LTES:
                pdir = os.path.join(
                    root, f'results/predictions/miss_{m}_{c}_{lte}'
                )
                for s in SEEDS:
                    p, _n = find_mosei_prediction_csv(pdir, c, s, root)
                    if p is not None:
                        n_pred += 1
    rep['mosei_prediction_files_accepted'] = n_pred

    for m in MISSING_RATES:
        for c in CENTERS:
            for lte in LTES:
                slug = _fusion_center_slug(c, lte)
                pdir = os.path.join(
                    root, f'results/predictions/miss_{m}_{c}_{lte}'
                )
                out_p = os.path.join(dnorm, slug, f'mosei-regression-{m}.csv')
                pl: List[str] = []
                val_notes: List[str] = []
                one: Dict[str, Any] = {
                    'missing': m,
                    'center': c,
                    'lte': lte,
                    'fc_slug': slug,
                    'normals_relpath': os.path.relpath(out_p, root),
                }
                try:
                    sm = per_seed_metrics(
                        pdir, c, pl, root, val_notes, verbose=args.verbose
                    )
                except Exception as e:  # pylint: disable=broad-except
                    one['status'] = f'error: {e}'
                    rep['incomplete'].append(one)
                    rep['combinations'].append(one)
                    continue
                if val_notes:
                    one['validation_notes'] = val_notes
                    rep['rejected_mosi_for_mosei_normals'].extend(val_notes)
                if rep.get('metrics_backend') is None and _METRICS_BACKEND is not None:
                    rep['metrics_backend'] = _METRICS_BACKEND
                one['prediction_csv'] = pl
                one['meta'] = scan_meta_any(pl, root) or {}
                if len(sm) < 3:
                    ex = val_notes[:5]
                    if len(val_notes) > 5:
                        ex = ex + [f'... +{len(val_notes) - 5} more']
                    print(
                        f'[rebuild_mosei_normals] WARNING: skip '
                        f'mosei-regression {m} {c} {lte} — need 3 valid MOSEI exports, '
                        f'found seeds={list(sm)}. Example notes: {ex}',
                        file=sys.stderr,
                    )
                    one['status'] = (
                        f'cannot_build_mosei_normals: need 3 valid MOSEI seeds, '
                        f'found {list(sm)} — see validation_notes (MOSI not used).'
                    )
                    rep['incomplete'].append(one)
                    rep['combinations'].append(one)
                    continue
                new_row = build_row(sm, 'mfmb_net')
                if args.dry_run:
                    one['status'] = 'dry'
                    if args.force or is_file_bad(out_p) or (not os.path.isfile(out_p)):
                        one['action'] = 'would_write'
                    else:
                        oldr = read_first_mfmb_row(out_p)
                        if oldr and rows_close(
                            oldr, new_row, tol=args.tol, compare_loss=False
                        ):
                            one['action'] = 'skip_match'
                        else:
                            one['action'] = 'would_write_mismatch'
                    rep['combinations'].append(one)
                    continue
                if args.force:
                    write_csv(out_p, new_row)
                    one['action'] = 'wrote_force'
                    rep['normals_wrote'].append(one['normals_relpath'])
                elif is_file_bad(out_p) or (not os.path.isfile(out_p)):
                    write_csv(out_p, new_row)
                    one['action'] = 'wrote_new_or_repair'
                    rep['normals_wrote'].append(one['normals_relpath'])
                else:
                    oldr = read_first_mfmb_row(out_p)
                    if oldr and rows_close(
                        oldr, new_row, tol=args.tol, compare_loss=False
                    ):
                        one['action'] = 'unchanged'
                        rep['normals_unchanged'].append(
                            one['normals_relpath']
                        )
                    else:
                        write_csv(out_p, new_row)
                        one['action'] = 'wrote_replace_mismatch'
                        rep['normals_wrote'].append(
                            one['normals_relpath']
                        )
                one['status'] = 'ok'
                rep['combinations'].append(one)
    d = os.path.dirname(out_report)
    if d:
        os.makedirs(d, exist_ok=True)
    if not args.dry_run:
        with open(out_report, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
    else:
        rep['dry_run'] = True
        with open(out_report, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
    print(f'Wrote {out_report}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
