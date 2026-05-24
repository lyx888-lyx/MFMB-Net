# Center Fairness Diagnostics (0.4 collapse case)

## Scope
This note checks whether `dynamic_soft` is fairly comparable to fixed `text` when router collapses to text. No model change is applied here.

## Code-path checks
1. Fixed text center and dynamic text-center branch share the same text-center micro builder:
   - `_build_center_micro('text', ...)` in `models/missingTask/MFMB_NET/fusion_599.py`.
2. Fixed text uses pair BN keys `tv` and `ta`.
3. Dynamic text branch also uses `tv` and `ta` for the text-center branch.
4. When `use_anchor_moe=0`, both fixed and dynamic use the same shared predictor `classifier2` (`BatchNorm1d -> Linear -> ReLU -> Dropout -> Linear`).

## Important non-equivalences
1. `dynamic_soft` always computes all three centers (`text/audio/vision`) and uses router-weighted fusion; fixed text only uses text center.
2. `dynamic_soft` has extra trainable router parameters (`ReliabilityAnchorRouter`), so optimization is not identical.
3. Even with collapse, soft routing is not exact one-hot in floating point.
4. Shared upstream encoders/projections are trained under a multi-center objective in `dynamic_soft`, so text-branch features can diverge from fixed-text training.

## Observed 0.4 outcome (official 3-seed)
- text: MAE=1.106567, Corr=0.548067, Non0_acc_2=0.705400, Non0_F1=0.704467
- dynamic_soft: MAE=1.069033, Corr=0.557033, Non0_acc_2=0.732333, Non0_F1=0.730867
- delta(dynamic_soft - text):
  - MAE -0.037533 (lower is better)
  - Corr +0.008967
  - Non0_acc_2 +0.026933
  - Non0_F1 +0.026400

## Fairness interpretation
- Inference behavior is close to text-center when collapsed, but training is not strictly identical to fixed text because router + multi-center joint training modifies learned representations.
- Therefore, `dynamic_soft > text` at missing=0.4 can happen without code bug, via optimization/regularization effects of dynamic training.

## Recommended follow-up (no code change in this round)
1. Add a strict ablation mode: compute only text center inside `dynamic_soft` with router frozen to one-hot text.
2. Log per-epoch router weights to inspect whether collapse is early or late.
3. Compare with BN-stat control (e.g., freeze pair BN stats) for stronger fairness control.
