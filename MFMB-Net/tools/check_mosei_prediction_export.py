#!/usr/bin/env python3
"""
Quick smoke check: verify a MOSEI test prediction export (CSV + meta) is real MOSEI.

Pass either:
  - path to a predictions_mosei_fc_*.csv file, or
  - a directory (finds first matching CSV in that directory)

Exit code 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load_meta(csv_path: str) -> Optional[dict]:
    mp = csv_path.replace('.csv', '_meta.json') if csv_path.endswith('.csv') else csv_path + '_meta.json'
    if not os.path.isfile(mp):
        return None
    with open(mp, encoding='utf-8') as f:
        return json.load(f)


def validate_mosei_meta(meta: dict) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if meta.get('datasetName') != 'mosei':
        errs.append(
            f"datasetName={meta.get('datasetName')!r} (expected 'mosei')"
        )
    dp = str(meta.get('dataPath_runtime') or '')
    if 'MOSEI' not in dp:
        errs.append(f"dataPath_runtime should contain 'MOSEI', got: {dp!r}")
    n = meta.get('n_samples')
    if n is None:
        errs.append('n_samples missing in meta')
    elif n == 686:
        errs.append('n_samples==686 (typical MOSI test size) — not MOSEI')
    return (len(errs) == 0, errs)


def check_csv_path(csv_path: str) -> int:
    print('File:', os.path.abspath(csv_path))
    base = os.path.basename(csv_path)
    if not base.startswith('predictions_mosei_fc_'):
        print('FAIL: filename must start with predictions_mosei_fc_ (got MOSI-style mosi if starts with predictions_mosi_fc_)')
        return 1
    meta = _load_meta(csv_path)
    if not meta:
        print('FAIL: missing sidecar meta', csv_path.replace('.csv', '_meta.json'))
        return 1
    print('meta.json datasetName:', meta.get('datasetName'))
    print('meta.json dataPath_runtime:', meta.get('dataPath_runtime'))
    print('meta.json n_samples:', meta.get('n_samples'))
    ok, errs = validate_mosei_meta(meta)
    if not ok:
        for e in errs:
            print('FAIL:', e)
        return 1
    print('OK: looks like a MOSEI export (filename + meta + n_samples not 686).')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        'path',
        type=str,
        help='Path to predictions_mosei_fc_*.csv or a directory to scan',
    )
    args = ap.parse_args()
    p = os.path.abspath(args.path)
    if os.path.isdir(p):
        pat = os.path.join(p, 'predictions_mosei_fc_*_test*.csv')
        cands = sorted(glob.glob(pat))
        if not cands:
            print('FAIL: no predictions_mosei_fc_*_test*.csv under', p)
            return 1
        return check_csv_path(cands[0])
    if not os.path.isfile(p):
        print('FAIL: not a file or directory:', p)
        return 1
    return check_csv_path(p)


if __name__ == '__main__':
    sys.exit(main())
