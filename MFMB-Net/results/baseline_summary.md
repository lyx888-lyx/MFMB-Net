# Fixed-Baseline Summary (MOSI, engineering fixes applied, MIDE disabled)

Seeds: `111, 1111, 11111` (same as `run.py` defaults)

| missing | Has0_acc_2 | Has0_F1 | Non0_acc_2 | Non0_F1 | Mult_acc_5 | Mult_acc_7 | MAE | Corr | Loss |
|--------:|-----------:|--------:|-----------:|--------:|-----------:|-----------:|----:|-----:|-----:|
| 0.0 | 82.26±0.30 | 82.37±0.33 | 83.94±0.56 | 83.98±0.64 | 49.47±1.07 | 43.39±0.70 | 74.67±1.52 | 78.19±0.76 | 74.27±1.58 |
| 0.3 | 72.30±0.41 | 72.61±0.10 | 73.58±0.40 | 73.76±0.77 | 34.94±1.91 | 31.78±1.44 | 102.26±1.54 | 61.35±0.69 | 101.46±1.59 |
| 0.5 | 66.33±1.95 | 66.82±1.53 | 66.72±2.45 | 67.10±2.08 | 27.50±1.50 | 25.76±1.31 | 119.95±5.78 | 47.11±1.16 | 119.10±5.67 |

Artifacts:
- CSV: `results/results/baseline/mosi-regression-{missing}.csv`
- Checkpoints: `results/models/baseline/m{missing}/`
- Logs: `results/logs/baseline_m*.log`

Engineering fixes included in this baseline (not algorithm changes):
- Per-batch optimizer step (removed broken `update_epochs` accumulation)
- `valid/test` loaders: `shuffle=False`, `drop_last=False`
- Dynamic batch reshape + device-safe tensors in fusion
- Optional `--data_root` / `MFMB_DATA_ROOT`
