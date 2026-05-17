# MOSEI regime-aware policy analysis (three policy branches)

## Dataset and inputs

- **Dataset:** MOSEI (`mosei-regression-<missing>.csv` under each normals subfolder).
- **MOSEI vs MOSI:** this script **never** mixes stems: `--dataset mosei` only reads `mosei-regression-*.csv` and `predictions_mosei_fc_*.csv`.

## Data availability audit

- missing_rate **0.15** / **text** (Baseline): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.15** / **dynamic_rule** (Ours-Dynamic): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.15** / **text_mstcn** (Ours-MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.15** / **dynamic_lte_mstcn** (Ours-Dynamic+MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.25** / **text** (Baseline): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.25** / **dynamic_rule** (Ours-Dynamic): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.25** / **text_mstcn** (Ours-MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.25** / **dynamic_lte_mstcn** (Ours-Dynamic+MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.35** / **text** (Baseline): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.35** / **dynamic_rule** (Ours-Dynamic): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.35** / **text_mstcn** (Ours-MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.35** / **dynamic_lte_mstcn** (Ours-Dynamic+MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.45** / **text** (Baseline): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.45** / **dynamic_rule** (Ours-Dynamic): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.45** / **text_mstcn** (Ours-MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.45** / **dynamic_lte_mstcn** (Ours-Dynamic+MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.55** / **text** (Baseline): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.55** / **dynamic_rule** (Ours-Dynamic): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.55** / **text_mstcn** (Ours-MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr
- missing_rate **0.55** / **dynamic_lte_mstcn** (Ours-Dynamic+MSTCN): no usable normals or predictions for Mult_acc_7, MAE, Corr

## AUILC integration note

AUILC upper bound was requested as 1 but max observed missing rate is 0.6. **Clamped AUILC integration to [0.10, 0.60]** (no flat extrapolation). Re-run with `--allow-extrapolate-auilc` to extrapolate, or set e.g. `--auilc-interval 0.1,0.60` for MOSEI-style grids.

## Role of each curve

1. **oracle_best_point** — At each missing rate, picks the best among **`text`**, **`dynamic_rule`**, **`text_mstcn`** using the recorded **test** metrics. This is a **test oracle upper bound** on segmentation-by-rate; it is **not** a single deployable model and must not be reported as such.

2. **regime_policy** — Follows the fixed template below with thresholds **`tau1`, `tau2`** chosen by search on the **same recorded curves** (default in this script: **test-set regime analysis**). This is still **analysis / policy study**, not a classical single-model test score, unless thresholds are locked on **validation** and then applied once on test.

3. **Proper reporting** — To present a **legitimate** regime policy: choose `tau1`, `tau2` on a **validation** protocol, freeze them, then evaluate **one** policy on **test**.

---

## Automatically selected thresholds

**Objective A (default for tables/figures):** maximize **AUILC** of **Mult_acc_7** along the policy curve on the observed rate grid.

**Objective B (supplementary):** maximize **AUILC** of a **composite** score per rate:
\[ z(\mathrm{Mult\_acc\_5}) + z(\mathrm{Mult\_acc\_7}) + z(\mathrm{Corr}) - z\_{\mathrm{bad}}(\mathrm{MAE}) \]
where each \(z\) is **min–max normalized across the three branches** at that rate.

| Item | Objective A (Mult_acc_7 AUILC) | Objective B (composite AUILC) |
|------|-------------------------------|-------------------------------|
| **best_tau1** | 0.1 | 0.0 |
| **best_tau2** | 0.6 | 0.0 |
| **objective_value** | 0.296855 | 1.1073508999604234 |
| **policy_definition** | if r <= 0.1: dynamic_rule; elif r <= 0.6: text_mstcn; else: text | if r <= 0: dynamic_rule; elif r <= 0: text_mstcn; else: text |

**Primary policy used in `policy_metrics_by_missing.csv` and figures:** Objective **A**.

---

## Policy definition (primary)

if r <= 0.1: dynamic_rule; elif r <= 0.6: text_mstcn; else: text

Branches:

- **`dynamic_rule`** → normals slug **`dynamic_missing_router_rule`** (fallback **`dynamic_missing`**).
- **`text_mstcn`** → **`text_lte_mstcn`**.
- **`text`** → **`text`** (legacy LTE).

---

## Which branch at each missing rate (primary policy)

- missing_rate **0.0** → `dynamic_rule` (slug `dynamic_missing_router_rule`)
- missing_rate **0.1** → `dynamic_rule` (slug `dynamic_missing_router_rule`)
- missing_rate **0.15** → `text_mstcn` (slug `text_lte_mstcn`)
- missing_rate **0.2** → `text_mstcn` (slug `text_lte_mstcn`)
- missing_rate **0.25** → `text_mstcn` (slug `text_lte_mstcn`)
- missing_rate **0.3** → `text_mstcn` (slug `text_lte_mstcn`)
- missing_rate **0.35** → `text_mstcn` (slug `text_lte_mstcn`)
- missing_rate **0.4** → `text_mstcn` (slug `text_lte_mstcn`)
- missing_rate **0.45** → `text_mstcn` (slug `text_lte_mstcn`)
- missing_rate **0.5** → `text_mstcn` (slug `text_lte_mstcn`)
- missing_rate **0.55** → `text_mstcn` (slug `text_lte_mstcn`)
- missing_rate **0.6** → `text_mstcn` (slug `text_lte_mstcn`)

---

## AUILC summary (subset)

Curves: `text`, `dynamic_rule`, `text_mstcn`, `regime_policy`, `oracle_best_point`.  
Metrics: Mult_acc_5, Mult_acc_7, MAE, Corr.  
Intervals: **0.0–0.6**, **0.1–0.6** (and **0.0–1.0** if `--extend-auilc-to-1` and rates reach 1).

```
            curve     metric interval    auilc better_direction
dynamic_lte_mstcn       Corr  0.0_0.6 0.390785           higher
dynamic_lte_mstcn       Corr  0.1_0.6 0.316825           higher
dynamic_lte_mstcn        MAE  0.0_0.6 0.379020            lower
dynamic_lte_mstcn        MAE  0.1_0.6 0.322430            lower
dynamic_lte_mstcn Mult_acc_5  0.0_0.6 0.300460           higher
dynamic_lte_mstcn Mult_acc_5  0.1_0.6 0.246670           higher
dynamic_lte_mstcn Mult_acc_7  0.0_0.6 0.293240           higher
dynamic_lte_mstcn Mult_acc_7  0.1_0.6 0.241235           higher
     dynamic_rule       Corr  0.0_0.6 0.386240           higher
     dynamic_rule       Corr  0.1_0.6 0.313095           higher
     dynamic_rule        MAE  0.0_0.6 0.378905            lower
     dynamic_rule        MAE  0.1_0.6 0.322165            lower
     dynamic_rule Mult_acc_5  0.0_0.6 0.302125           higher
     dynamic_rule Mult_acc_5  0.1_0.6 0.248575           higher
     dynamic_rule Mult_acc_7  0.0_0.6 0.294730           higher
     dynamic_rule Mult_acc_7  0.1_0.6 0.242770           higher
oracle_best_point       Corr  0.0_0.6 0.391580           higher
oracle_best_point       Corr  0.1_0.6 0.318015           higher
oracle_best_point        MAE  0.0_0.6 0.374750            lower
oracle_best_point        MAE  0.1_0.6 0.318590            lower
oracle_best_point Mult_acc_5  0.0_0.6 0.305535           higher
oracle_best_point Mult_acc_5  0.1_0.6 0.251405           higher
oracle_best_point Mult_acc_7  0.0_0.6 0.298535           higher
oracle_best_point Mult_acc_7  0.1_0.6 0.245890           higher
    regime_policy       Corr  0.0_0.6 0.387310           higher
    regime_policy       Corr  0.1_0.6 0.314165           higher
    regime_policy        MAE  0.0_0.6 0.376345            lower
    regime_policy        MAE  0.1_0.6 0.319605            lower
    regime_policy Mult_acc_5  0.0_0.6 0.303980           higher
    regime_policy Mult_acc_5  0.1_0.6 0.250430           higher
    regime_policy Mult_acc_7  0.0_0.6 0.296855           higher
    regime_policy Mult_acc_7  0.1_0.6 0.244895           higher
             text       Corr  0.0_0.6 0.391365           higher
             text       Corr  0.1_0.6 0.318015           higher
             text        MAE  0.0_0.6 0.376860            lower
             text        MAE  0.1_0.6 0.320305            lower
             text Mult_acc_5  0.0_0.6 0.303495           higher
             text Mult_acc_5  0.1_0.6 0.249525           higher
             text Mult_acc_7  0.0_0.6 0.296535           higher
             text Mult_acc_7  0.1_0.6 0.243985           higher
       text_mstcn       Corr  0.0_0.6 0.387530           higher
       text_mstcn       Corr  0.1_0.6 0.314090           higher
       text_mstcn        MAE  0.0_0.6 0.376110            lower
       text_mstcn        MAE  0.1_0.6 0.319765            lower
       text_mstcn Mult_acc_5  0.0_0.6 0.304480           higher
       text_mstcn Mult_acc_5  0.1_0.6 0.250695           higher
       text_mstcn Mult_acc_7  0.0_0.6 0.296940           higher
       text_mstcn Mult_acc_7  0.1_0.6 0.244880           higher
```

Higher AUILC is better for accuracy / correlation metrics; **lower** is better for MAE / Loss.

---

## Slopes (Mult_acc_7)

Mean slope over segments whose **midpoint** lies in each band (larger = slower decline for Mult_acc_7):

| Band | Slowest decline (method) | Mean slope in band |
|------|--------------------------|--------------------|
| 0.1 – 0.3 | None | nan |
| 0.3 – 0.5 | None | nan |
| 0.1 – 0.6 | None | nan |

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

Normals prefer `results/results/normals/<subdir>/mosei-regression-<miss>.csv` (means ×100 in cells).  
Fallback: recompute from `results/predictions/miss_<miss>_<center>_<lte>/` using the dataset-specific prediction glob (MOSI: `predictions_mosi_fc_*test*.csv`; MOSEI: `predictions_mosei_fc_*test*.csv`). **Loss** is not recovered from row-wise export (NaN).

---

## Figure guide (paper-oriented exports)

All experiments use **symmetric missing rates** (the same missing rate is applied jointly to the three modalities).

1. **Ours-Policy is not a separate trained model** — it is a **missing-rate-aware branch selection policy** that switches among the three fixed branches using thresholds **τ₁**, **τ₂** (searched on the observed rate grid in this script by default).

2. **Main performance figure** — `fig_main_policy_vs_baseline_mosei.png` / `.pdf` plots **Baseline** vs **Ours-Policy** from missing rate **0** up to **1.0** on the x-axis (ticks include 0.15–0.55 and 0.7–1.0 when in range). Where no measurements exist yet, curves use **flat extrapolation** of each branch from the last observed rate. **Ours-Policy** at any rate uses the metric of the **selected branch**; when that branch is **Baseline** (`text`), the orange curve coincides with the blue curve. In particular, **for high missing rates, Ours-Policy falls back to the baseline branch, so the two curves may overlap** — this is expected.

3. **AUILC bar figure** — `fig_auilc_summary_mosei.png` / `.pdf` reports trapezoid **AUILC** on **[0.10, 0.60]** (configurable via `--auilc-lo` / `--auilc-hi` or `--auilc-interval`). **Bars include Baseline, Ours-Dynamic, Ours-MSTCN, Ours-Dynamic+MSTCN, and Ours-Policy.** Without `--allow-extrapolate-auilc`, the script **does not** silently integrate past the maximum observed missing rate (see *AUILC integration note* above). With `--allow-extrapolate-auilc`, each branch is **held flat** beyond the last observed rate. For **Mult-Acc7** and **Corr**, higher integrated area is better; for **MAE**, **lower** integrated area is better.

4. **Policy schedule** — `fig_policy_schedule_mosei.png` / `.pdf` is a **step chart** on **[0, 1.0]** showing which branch the policy selects, with **τ₁**, **τ₂** as vertical guides (when finite).

5. **Pairwise supplements** — `fig_pairwise_*_mosei` show **raw** Baseline vs **Ours-Dynamic** / **Ours-MSTCN** as **two lines only** (no red/green shading between curves).

6. **Delta supplement** — `fig_delta_improvement_mosei.png` / `.pdf` highlights **relative improvement vs Baseline** with **green** for Δ>0 and **red** for Δ<0 (MAE: Δ = MAE(Baseline) − MAE(Method), so positive is still better).

7. **Legend mapping** — `text` → **Baseline**; `dynamic_rule` → **Ours-Dynamic**; `text_mstcn` → **Ours-MSTCN**; `dynamic_lte_mstcn` → **Ours-Dynamic+MSTCN**; **Ours-Policy** = regime policy curve. Optional caption text: Baseline = MFMB-Net with text-centered local fusion; Ours-Dynamic = proposed dynamic-anchor branch; Ours-MSTCN = proposed MSTCN-enhanced branch; Ours-Policy = missing-rate-aware branch selection.

8. **x-axis** — Curve panels use **`set_xlim(x_min, x_max)`** and **`margins(x=0)`**. Tick labels use the **0.00, 0.10, 0.15, …** format; every **observed** missing rate in the span is included in the tick set.

---

## Figures produced (under `figures/`)

| File | Role | Visual form |
|------|------|----------------|
| `fig_main_policy_vs_baseline_mosei` | Main | Line (1×3), x to 1.0 |
| `fig_auilc_summary_mosei` | Main | Bar (1×3), five methods + policy; AUILC on [0.10, 0.60] |
| `fig_policy_schedule_mosei` | Main | Step + τ; x to 1.0 |
| `fig_pairwise_dynamic_vs_baseline_mosei` | Supplement | Two lines (1×3) |
| `fig_pairwise_mstcn_vs_baseline_mosei` | Supplement | Two lines (1×3) |
| `fig_delta_improvement_mosei` | Supplement | Δ lines + fill vs 0 (1×3) |

Each base name is saved as **`.png`** (300 dpi, `bbox_inches='tight'`) and **`.pdf`**.

Y-axes for raw metric curves use the **×100** display convention. Serif fonts via matplotlib rcParams.

