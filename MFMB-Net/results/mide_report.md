# MIDE Experiment Report (MFMB-Net / MOSI)

Generated: 2026-06-08

## 1. Modified Files

| File | Core changes |
|------|----------------|
| `models/missingTask/MFMB_NET/mide.py` | **New.** `MIDEController`, availability helper, unimodal/utility/pollution heads, MIDE losses |
| `models/missingTask/MFMB_NET/fusion_599.py` | MIDE integration, LOO fusion predictions, density-weighted fusion, dynamic batch reshape, device-safe tensors |
| `models/missingTask/MFMB_NET/model.py` | Pass masks/labels/epoch to fusion; return `mide_aux` |
| `models/AMIO.py` | Forward kwargs for MIDE context |
| `trains/missingTask/MFMB_NET.py` | Fix per-batch training; combine MIDE losses with warmup |
| `config/config_regression.py` | MIDE hyperparameters; `--data_root` support |
| `data/load_data.py` | `valid/test`: `shuffle=False`, `drop_last=False` |
| `run.py` | `--data_root`, `--mide_enable`, `--exp_tag`; GPU remap under `CUDA_VISIBLE_DEVICES` |
| `scripts/run_mide_sweep.sh` | Missing-rate sweep helper |
| `scripts/compare_auilc.py` | Trapezoidal AUILC baseline vs MIDE |
| `scripts/generate_mide_report.py` | Auto-report generator |
| `scripts/run_priority_experiments.sh` | Baseline + MIDE priority suite |
| `scripts/rerun_mide_priority.sh` | MIDE v2 (stability) rerun |
| `scripts/rerun_mide_r3.sh` | MIDE round-3 tuning rerun |

## 2. Engineering Fixes (not MIDE innovation)

1. **Trainer gradient accumulation removed** — each batch: `zero_grad → backward → step` (old `update_epochs` logic was incorrect).
2. **DataLoader** — train: `shuffle=True, drop_last=True`; valid/test: `shuffle=False, drop_last=False`.
3. **Fusion reshape** — uses `bs = text_rep.size(0)` instead of `self.args.batch_size`.
4. **Device safety** — replaced `.cuda()` with `device=` / `to(device)` in `BottleAttentionNet`.
5. **Data root** — priority: `--data_root` > `MFMB_DATA_ROOT` > default `/sharefile/...`.

## 3. Commands Executed

```bash
# Baseline (MIDE off)
/usr/miniconda3/envs/mfmb/bin/python run.py --missing 0.0 --exp_tag baseline
/usr/miniconda3/envs/mfmb/bin/python run.py --missing 0.3 --exp_tag baseline
/usr/miniconda3/envs/mfmb/bin/python run.py --missing 0.5 --exp_tag baseline

# MIDE (final kept config: round-3)
/usr/miniconda3/envs/mfmb/bin/python run.py --missing 0.0 --mide_enable --exp_tag mide_r3
/usr/miniconda3/envs/mfmb/bin/python run.py --missing 0.3 --mide_enable --exp_tag mide_r3
/usr/miniconda3/envs/mfmb/bin/python run.py --missing 0.5 --mide_enable --exp_tag mide_r3

# Analysis
/usr/miniconda3/envs/mfmb/bin/python scripts/compare_auilc.py \
  --baseline_dir results/results/baseline \
  --mide_dir results/results/mide \
  --output results/auilc_comparison.csv
```

Seeds: `111, 1111, 11111` (unchanged from `run.py`).

## 4. Baseline Results (fixed-baseline, MIDE off)

| missing | Has0_acc_2 | Has0_F1 | Non0_acc_2 | Non0_F1 | Mult_acc_5 | Mult_acc_7 | MAE | Corr |
|--------:|-----------:|--------:|-----------:|--------:|-----------:|-----------:|----:|-----:|
| 0.0 | 82.26 | 82.37 | 83.94 | 83.98 | 49.47 | 43.39 | **74.67** | **78.19** |
| 0.3 | 72.30 | 72.61 | 73.58 | 73.76 | 34.94 | 31.78 | **102.26** | **61.35** |
| 0.5 | 66.33 | 66.82 | 66.72 | 67.10 | 27.50 | 25.76 | **119.95** | **47.11** |

CSV: `results/results/baseline/` | Summary: `results/baseline_summary.md`

## 5. MIDE Results (kept: round-3 config)

Final hyperparameters: `tau=1.0`, utility bias init `+1.5`, `λ_uni=0.1`, `λ_uce=0.15`, `λ_rank=0.05`, `λ_poll=0.05`, warmup=1 epoch (no density weighting / no MIDE loss in epoch 1).

| missing | Has0_acc_2 | Has0_F1 | Non0_acc_2 | Non0_F1 | Mult_acc_5 | Mult_acc_7 | MAE | Corr |
|--------:|-----------:|--------:|-----------:|--------:|-----------:|-----------:|----:|-----:|
| 0.0 | 81.24 | 81.30 | 82.93 | 82.92 | 47.86 | 42.03 | 77.17 | 77.96 |
| 0.3 | **73.42** | **73.42** | **74.34** | **74.25** | 32.56 | 30.32 | 105.76 | 60.66 |
| 0.5 | 64.62 | 65.07 | 64.23 | 64.52 | 22.16 | 21.28 | 130.34 | 44.98 |

CSV copied to `results/results/mide/` from best tuning round.

## 6. Per-Rate Delta (MIDE r3 − Baseline)

| missing | ΔHas0_acc_2 | ΔMAE | ΔCorr | ΔMult_acc_7 |
|--------:|------------:|-----:|------:|------------:|
| 0.0 | −1.02 | +2.50 | −0.23 | −1.36 |
| 0.3 | **+1.12** | +3.50 | −0.69 | −1.46 |
| 0.5 | −1.71 | +10.39 | −2.13 | −4.48 |

## 7. AUILC (missing ∈ {0.0, 0.3, 0.5}, trapezoid)

| metric | baseline | MIDE r3 | Δ (MIDE−base) | MIDE better? |
|--------|----------|---------|---------------|--------------|
| MAE | −48.76 | −51.05 | +2.29 (worse) | No |
| Corr | 31.78 | 31.36 | −0.42 | No |
| Has0_acc_2 | 37.05 | 37.00 | −0.04 | No |
| Has0_F1 | 37.19 | 37.06 | −0.13 | No |
| Non0_acc_2 | 37.66 | 37.45 | −0.21 | No |
| Mult_acc_5 | 18.91 | 17.54 | −1.37 | No |
| Mult_acc_7 | 17.03 | 16.01 | −1.02 | No |

Full table: `results/auilc_comparison.csv`

## 8. What Improved / What Did Not

**Improved (pointwise):**
- `missing=0.3`: Has0_acc_2 **+1.12%**, Non0_acc_2 **+0.76%**, Has0_F1 **+0.81%**

**Not improved:**
- `missing=0.0`: all core metrics slightly worse (MAE +2.5, Has0_acc −1.0)
- `missing=0.5`: Has0_acc −1.7, MAE +10.4, Corr −2.1
- AUILC across {0.0,0.3,0.5}: no metric wins overall

**Likely reasons:**
- Density scaling `D·rep` attenuates fusion inputs; even with neutral init, D≈0.6–0.7 shrinks effective representation magnitude.
- `utility_rank` + `pollution` losses are active early and compete with `L_pred`.
- At high missing (0.5), generator + MIDE auxiliary losses add noise; MAE degrades most.

## 9. Training Stability

**Loss trends (MIDE r3, missing=0.3, late epochs):**
- `uni_pred`: 2.74 → ~0.41 (decreasing)
- `utility_ce`: ~0.02–0.05 (stable after warmup)
- `utility_rank`: ~0.27–0.30 (dominant auxiliary term)
- `pollution`: ~0.11–0.17
- `total_mide`: ~0.07–0.26

**Density / contribution (epoch≈5, missing=0.3):**
- `D_mean` ≈ [0.61, 0.61, 0.61] (t, a, v)
- `C_mean` ≈ [0.05, 0.005, 0.03] — **text** highest leave-one-out contribution; **audio** near-zero / slight pollution

**Modality interpretation:**
- Highest utility density: **text** (also highest C)
- Most pollution risk: **audio** (lowest/negative C in several batches)

## 10. Tuning Rounds (max 3)

| Round | Change | Outcome |
|-------|--------|---------|
| v1 | Default λ, density from epoch 1 | NaN at missing=0.3; m0.0 Has0 collapse |
| v2 | Warmup w/o density weighting, clamp D, lower λ | Stable; m0.3 Has0 +0.98; m0.0 still weak |
| **r3 (kept)** | Utility bias +1.5, τ=1.0, lower λ | Best overall; m0.3 Has0 **+1.12**, m0.0 MAE closest to baseline |

## 11. Engineering Fix Impact on Baseline

The fixed trainer (no bogus grad accumulation) and valid/test loaders produce a **reproducible** baseline under the same seeds. These fixes are shared by MIDE runs; observed MIDE deltas are not caused by different training mechanics.

## 12. Conclusion

- **MIDE is implemented as specified** (A from masks, U from unimodal error + LOO ranking, P from pollution, D=A·U·(1−P) enters fusion).
- **Partial success:** at **missing=0.3**, classification-style metrics improve (+1.1% Has0_acc_2), but **MAE/Corr do not beat baseline** at target rates overall.
- **Most influential loss:** `utility_rank` (pairwise contribution ordering).
- **Best kept checkpoint config:** MIDE round-3 (`results/results/mide/`, `results/models/mide_r3/`).
- **Honest assessment:** full acceptance criterion (MIDE beats baseline on core metrics at 0.0/0.3/0.5) is **not met** for MAE/Corr; Has0_acc_2 improves only at 0.3.

Artifacts:
- Logs: `results/logs/baseline_m*.log`, `results/logs/mide_r3_m*.log`
- Checkpoints: `results/models/baseline/`, `results/models/mide_r3/`
