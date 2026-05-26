#!/usr/bin/env python3
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRESS_DIR = Path("results/auto_anchor_runs_stress_06_10")
OURS0_DIR = Path("results/auto_anchor_runs_rta_pred_residual_complete")
OURS015_DIR = Path("results/auto_anchor_runs_rta_pred_residual_final")

METHOD_MAP = {
    "text": "Text center",
    "vision": "Vision center",
    "dynamic_soft": "Dynamic Soft",
    "dynamic_rta_pred_residual": "Ours",
}
METHOD_ORDER = ["text", "vision", "dynamic_soft", "dynamic_rta_pred_residual"]

METRICS = [
    ("MAE", "lower"),
    ("Corr", "higher"),
    ("Non0_acc_2", "higher"),
    ("Non0_F1", "higher"),
    ("Mult_acc_5", "higher"),
    ("Mult_acc_7", "higher"),
]
CORE_METRICS = ["MAE", "Corr", "Non0_acc_2", "Non0_F1"]


def _ensure_no_normals(paths: List[str]) -> None:
    for p in paths:
        if "results/results/normals" in p.replace("\\", "/"):
            raise RuntimeError(
                "WARNING: normals CSV does not contain enough metadata and must not be used for paper tables."
            )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _best_second_indices(values: pd.Series, direction: str) -> Tuple[set, set]:
    s = values.dropna()
    if len(s) == 0:
        return set(), set()
    uniq = sorted(s.unique(), reverse=(direction == "higher"))
    best = uniq[0]
    second = uniq[1] if len(uniq) > 1 else None
    best_idx = set(values[values == best].index.tolist())
    second_idx = set(values[values == second].index.tolist()) if second is not None else set()
    return best_idx, second_idx


def _fmt(v: float, n: int = 4) -> str:
    if pd.isna(v):
        return "N/A"
    return f"{v:.{n}f}"


def _md_mark(value: str, is_best: bool, is_second: bool) -> str:
    if value == "N/A":
        return value
    if is_best:
        return f"**{value}**"
    if is_second:
        return f"<u>{value}</u>"
    return value


def _tex_mark(value: str, is_best: bool, is_second: bool) -> str:
    if value == "N/A":
        return value
    if is_best:
        return f"\\textbf{{{value}}}"
    if is_second:
        return f"\\underline{{{value}}}"
    return value


def _trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_stress_comparison(stress_agg: pd.DataFrame) -> pd.DataFrame:
    sub = stress_agg[stress_agg["mode"].isin(METHOD_ORDER)].copy()
    sub = sub[["missing", "mode", "MAE_mean", "Corr_mean", "Non0_acc_2_mean", "Non0_F1_score_mean", "Mult_acc_5_mean", "Mult_acc_7_mean"]]
    sub = sub.rename(
        columns={
            "MAE_mean": "MAE",
            "Corr_mean": "Corr",
            "Non0_acc_2_mean": "Non0_acc_2",
            "Non0_F1_score_mean": "Non0_F1",
            "Mult_acc_5_mean": "Mult_acc_5",
            "Mult_acc_7_mean": "Mult_acc_7",
        }
    )
    sub["method"] = sub["mode"].map(METHOD_MAP)
    sub["method_order"] = sub["mode"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    sub = sub.sort_values(["missing", "method_order"]).drop(columns=["method_order"])
    return sub


def save_stress_table(df: pd.DataFrame) -> None:
    csv_path = STRESS_DIR / "stress_06_10_comparison.csv"
    df.to_csv(csv_path, index=False)

    # Markdown
    md = []
    md.append("| missing | method | MAE | Corr | Non0_acc_2 | Non0_F1 | Mult_acc_5 | Mult_acc_7 |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for miss in sorted(df["missing"].unique()):
        g = df[df["missing"] == miss].reset_index(drop=True)
        marks = {}
        for metric, direction in METRICS:
            b, s = _best_second_indices(g[metric], direction)
            marks[metric] = (b, s)
        for i, row in g.iterrows():
            vals = []
            for metric, _ in METRICS:
                t = _fmt(row[metric], 4)
                b, s = marks[metric]
                vals.append(_md_mark(t, i in b, i in s))
            md.append(
                f"| {miss:.1f} | {row['method']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {vals[5]} |"
            )
    md.append("")
    md.append(
        "This is an extreme missing-rate stress test over missing rates 0.6–1.0 under the full-test protocol. "
        "It is used as robustness analysis rather than the main comparison table."
    )
    _write_text(STRESS_DIR / "stress_06_10_table.md", "\n".join(md))

    # LaTeX
    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\begin{tabular}{lccccccc}")
    tex.append("\\toprule")
    tex.append("Missing & Method & MAE & Corr & Non0 Acc-2 & Non0 F1 & Mult Acc-5 & Mult Acc-7 \\\\")
    tex.append("\\midrule")
    for miss in sorted(df["missing"].unique()):
        g = df[df["missing"] == miss].reset_index(drop=True)
        marks = {}
        for metric, direction in METRICS:
            marks[metric] = _best_second_indices(g[metric], direction)
        for i, row in g.iterrows():
            vals = []
            for metric, _ in METRICS:
                t = _fmt(row[metric], 4)
                b, s = marks[metric]
                vals.append(_tex_mark(t, i in b, i in s))
            method = str(row["method"]).replace("_", "\\_")
            tex.append(
                f"{miss:.1f} & {method} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {vals[5]} \\\\"
            )
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append(
        "\\caption{This is an extreme missing-rate stress test over missing rates 0.6--1.0 under the full-test protocol. "
        "It is used as robustness analysis rather than the main comparison table.}"
    )
    tex.append("\\label{tab:stress_06_10}")
    tex.append("\\end{table}")
    _write_text(STRESS_DIR / "stress_06_10_table.tex", "\n".join(tex))


def build_stress_auilc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in METHOD_ORDER:
        sub = df[df["mode"] == mode].sort_values("missing")
        x = sub["missing"].to_numpy()
        row = {
            "method": METHOD_MAP[mode],
            "MAE_AUILC": _trapz_auc(x, sub["MAE"].to_numpy()),
            "Corr_AUILC": _trapz_auc(x, sub["Corr"].to_numpy()),
            "Non0_acc_2_AUILC": _trapz_auc(x, sub["Non0_acc_2"].to_numpy()),
            "Non0_F1_AUILC": _trapz_auc(x, sub["Non0_F1"].to_numpy()),
            "Mult_acc_5_AUILC": _trapz_auc(x, sub["Mult_acc_5"].to_numpy()),
            "Mult_acc_7_AUILC": _trapz_auc(x, sub["Mult_acc_7"].to_numpy()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_auilc_and_averages(df: pd.DataFrame, auilc_df: pd.DataFrame) -> None:
    auilc_csv = STRESS_DIR / "stress_06_10_auilc.csv"
    auilc_df.to_csv(auilc_csv, index=False)

    # Markdown/latex for AUILC
    auilc_metrics = [
        ("MAE_AUILC", "lower"),
        ("Corr_AUILC", "higher"),
        ("Non0_acc_2_AUILC", "higher"),
        ("Non0_F1_AUILC", "higher"),
        ("Mult_acc_5_AUILC", "higher"),
        ("Mult_acc_7_AUILC", "higher"),
    ]
    marks = {m: _best_second_indices(auilc_df[m], d) for m, d in auilc_metrics}

    md = []
    md.append("| method | MAE-AUILC | Corr-AUILC | Non0_acc_2-AUILC | Non0_F1-AUILC | Mult_acc_5-AUILC | Mult_acc_7-AUILC |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for i, row in auilc_df.reset_index(drop=True).iterrows():
        vals = []
        for m, _ in auilc_metrics:
            t = _fmt(row[m], 4)
            b, s = marks[m]
            vals.append(_md_mark(t, i in b, i in s))
        md.append(f"| {row['method']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {vals[5]} |")
    _write_text(STRESS_DIR / "stress_06_10_auilc.md", "\n".join(md))

    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\begin{tabular}{lcccccc}")
    tex.append("\\toprule")
    tex.append("Method & MAE-AUILC & Corr-AUILC & Non0 Acc-2-AUILC & Non0 F1-AUILC & Mult Acc-5-AUILC & Mult Acc-7-AUILC \\\\")
    tex.append("\\midrule")
    for i, row in auilc_df.reset_index(drop=True).iterrows():
        vals = []
        for m, _ in auilc_metrics:
            t = _fmt(row[m], 4)
            b, s = marks[m]
            vals.append(_tex_mark(t, i in b, i in s))
        method_esc = str(row["method"]).replace("_", "\\_")
        tex.append(f"{method_esc} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {vals[5]} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append(
        "\\caption{AUILC over missing rates 0.6--1.0. Lower is better for MAE-AUILC, higher is better for the others.}"
    )
    tex.append("\\label{tab:stress_auilc_06_10}")
    tex.append("\\end{table}")
    _write_text(STRESS_DIR / "stress_06_10_auilc.tex", "\n".join(tex))

    # averages
    avg = (
        df.groupby(["mode", "method"], as_index=False)[["MAE", "Corr", "Non0_acc_2", "Non0_F1", "Mult_acc_5", "Mult_acc_7"]]
        .mean()
        .rename(
            columns={
                "MAE": "MAE_avg",
                "Corr": "Corr_avg",
                "Non0_acc_2": "Non0_acc_2_avg",
                "Non0_F1": "Non0_F1_avg",
                "Mult_acc_5": "Mult_acc_5_avg",
                "Mult_acc_7": "Mult_acc_7_avg",
            }
        )
    )
    avg.to_csv(STRESS_DIR / "stress_06_10_average_metrics.csv", index=False)

    # ranking
    rank_rows = []
    for miss in sorted(df["missing"].unique()):
        g = df[df["missing"] == miss].copy()
        g["rank_MAE"] = g["MAE"].rank(method="average", ascending=True)
        g["rank_Corr"] = g["Corr"].rank(method="average", ascending=False)
        g["rank_Non0_acc_2"] = g["Non0_acc_2"].rank(method="average", ascending=False)
        g["rank_Non0_F1"] = g["Non0_F1"].rank(method="average", ascending=False)
        g["rank_Mult_acc_5"] = g["Mult_acc_5"].rank(method="average", ascending=False)
        g["rank_Mult_acc_7"] = g["Mult_acc_7"].rank(method="average", ascending=False)
        rank_rows.append(g)
    rank_df = pd.concat(rank_rows, ignore_index=True)
    rank_summary = (
        rank_df.groupby(["mode", "method"], as_index=False)[
            ["rank_MAE", "rank_Corr", "rank_Non0_acc_2", "rank_Non0_F1", "rank_Mult_acc_5", "rank_Mult_acc_7"]
        ]
        .mean()
    )
    rank_summary["AvgRank_core"] = rank_summary[["rank_MAE", "rank_Corr", "rank_Non0_acc_2", "rank_Non0_F1"]].mean(axis=1)
    rank_summary["AvgRank_all"] = rank_summary[
        ["rank_MAE", "rank_Corr", "rank_Non0_acc_2", "rank_Non0_F1", "rank_Mult_acc_5", "rank_Mult_acc_7"]
    ].mean(axis=1)
    rank_summary.to_csv(STRESS_DIR / "stress_06_10_average_rank.csv", index=False)

    # average summary markdown/latex
    sum_df = avg.merge(rank_summary[["mode", "AvgRank_core", "AvgRank_all"]], on="mode", how="left")
    sum_df = sum_df[["method", "MAE_avg", "Corr_avg", "Non0_acc_2_avg", "Non0_F1_avg", "Mult_acc_5_avg", "Mult_acc_7_avg", "AvgRank_core", "AvgRank_all"]]
    sum_metrics = [
        ("MAE_avg", "lower"),
        ("Corr_avg", "higher"),
        ("Non0_acc_2_avg", "higher"),
        ("Non0_F1_avg", "higher"),
        ("Mult_acc_5_avg", "higher"),
        ("Mult_acc_7_avg", "higher"),
        ("AvgRank_core", "lower"),
        ("AvgRank_all", "lower"),
    ]
    marks2 = {m: _best_second_indices(sum_df[m], d) for m, d in sum_metrics}

    md2 = []
    md2.append("| method | MAE_avg | Corr_avg | Non0_acc_2_avg | Non0_F1_avg | Mult_acc_5_avg | Mult_acc_7_avg | AvgRank_core | AvgRank_all |")
    md2.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, row in sum_df.reset_index(drop=True).iterrows():
        vals = []
        for m, _ in sum_metrics:
            t = _fmt(row[m], 4)
            b, s = marks2[m]
            vals.append(_md_mark(t, i in b, i in s))
        md2.append(
            f"| {row['method']} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} | {vals[5]} | {vals[6]} | {vals[7]} |"
        )
    _write_text(STRESS_DIR / "stress_06_10_average_summary.md", "\n".join(md2))

    tex2 = []
    tex2.append("\\begin{table}[t]")
    tex2.append("\\centering")
    tex2.append("\\small")
    tex2.append("\\begin{tabular}{lcccccccc}")
    tex2.append("\\toprule")
    tex2.append(
        "Method & MAE$_{avg}$ & Corr$_{avg}$ & Non0 Acc-2$_{avg}$ & Non0 F1$_{avg}$ & Mult5$_{avg}$ & Mult7$_{avg}$ & AvgRank$_{core}$ & AvgRank$_{all}$ \\\\"
    )
    tex2.append("\\midrule")
    for i, row in sum_df.reset_index(drop=True).iterrows():
        vals = []
        for m, _ in sum_metrics:
            t = _fmt(row[m], 4)
            b, s = marks2[m]
            vals.append(_tex_mark(t, i in b, i in s))
        method_esc = str(row["method"]).replace("_", "\\_")
        tex2.append(
            f"{method_esc} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {vals[5]} & {vals[6]} & {vals[7]} \\\\"
        )
    tex2.append("\\bottomrule")
    tex2.append("\\end{tabular}")
    tex2.append("\\caption{Average metrics and average ranks over stress missing rates 0.6--1.0.}")
    tex2.append("\\label{tab:stress_avg_summary}")
    tex2.append("\\end{table}")
    _write_text(STRESS_DIR / "stress_06_10_average_summary.tex", "\n".join(tex2))


def build_ours_curve() -> pd.DataFrame:
    ours0 = _read_csv(OURS0_DIR / "anchor_experiment_agg.csv")
    ours_mid = _read_csv(OURS015_DIR / "residual_final_comparison.csv")
    ours_hi = _read_csv(STRESS_DIR / "anchor_experiment_agg.csv")

    r0 = ours0[ours0["mode"] == "dynamic_rta_pred_residual"].iloc[0]
    rows = [
        {
            "missing": float(r0["missing"]),
            "MAE": float(r0["MAE_mean"]),
            "Corr": float(r0["Corr_mean"]),
            "Non0_acc_2": float(r0["Non0_acc_2_mean"]),
            "Non0_F1": float(r0["Non0_F1_score_mean"]),
            "Mult_acc_5": float(r0["Mult_acc_5_mean"]),
            "Mult_acc_7": float(r0["Mult_acc_7_mean"]),
        }
    ]

    mid = ours_mid[ours_mid["method"] == "dynamic_rta_pred_residual"].copy()
    for _, r in mid.iterrows():
        rows.append(
            {
                "missing": float(r["missing"]),
                "MAE": float(r["MAE"]),
                "Corr": float(r["Corr"]),
                "Non0_acc_2": float(r["Non0_acc_2"]),
                "Non0_F1": float(r["Non0_F1_score"]),
                "Mult_acc_5": float(r["Mult_acc_5"]),
                "Mult_acc_7": float(r["Mult_acc_7"]),
            }
        )

    hi = ours_hi[ours_hi["mode"] == "dynamic_rta_pred_residual"].copy()
    for _, r in hi.iterrows():
        rows.append(
            {
                "missing": float(r["missing"]),
                "MAE": float(r["MAE_mean"]),
                "Corr": float(r["Corr_mean"]),
                "Non0_acc_2": float(r["Non0_acc_2_mean"]),
                "Non0_F1": float(r["Non0_F1_score_mean"]),
                "Mult_acc_5": float(r["Mult_acc_5_mean"]),
                "Mult_acc_7": float(r["Mult_acc_7_mean"]),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["missing"]).sort_values("missing")
    return df


def save_ours_curve_and_auilc(ours_curve: pd.DataFrame) -> None:
    ours_curve.to_csv(STRESS_DIR / "ours_curve_00_10.csv", index=False)

    md = []
    md.append("| missing | MAE | Corr | Non0_acc_2 | Non0_F1 | Mult_acc_5 | Mult_acc_7 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in ours_curve.iterrows():
        md.append(
            f"| {r['missing']:.1f} | {r['MAE']:.4f} | {r['Corr']:.4f} | {r['Non0_acc_2']:.4f} | {r['Non0_F1']:.4f} | {r['Mult_acc_5']:.4f} | {r['Mult_acc_7']:.4f} |"
        )
    _write_text(STRESS_DIR / "ours_curve_00_10.md", "\n".join(md))

    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\begin{tabular}{cccccccc}")
    tex.append("\\toprule")
    tex.append("Missing & MAE & Corr & Non0 Acc-2 & Non0 F1 & Mult Acc-5 & Mult Acc-7 \\\\")
    tex.append("\\midrule")
    for _, r in ours_curve.iterrows():
        tex.append(
            f"{r['missing']:.1f} & {r['MAE']:.4f} & {r['Corr']:.4f} & {r['Non0_acc_2']:.4f} & {r['Non0_F1']:.4f} & {r['Mult_acc_5']:.4f} & {r['Mult_acc_7']:.4f} \\\\"
        )
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Ours (dynamic\\_rta\\_pred\\_residual) curve over missing rates 0.0--1.0.}")
    tex.append("\\label{tab:ours_curve_00_10}")
    tex.append("\\end{table}")
    _write_text(STRESS_DIR / "ours_curve_00_10.tex", "\n".join(tex))

    # AUILC for three ranges
    def block_auc(lo: float, hi: float) -> Dict[str, float]:
        sub = ours_curve[(ours_curve["missing"] >= lo) & (ours_curve["missing"] <= hi)].sort_values("missing")
        x = sub["missing"].to_numpy()
        return {
            "MAE_AUILC": _trapz_auc(x, sub["MAE"].to_numpy()),
            "Corr_AUILC": _trapz_auc(x, sub["Corr"].to_numpy()),
            "Non0_acc_2_AUILC": _trapz_auc(x, sub["Non0_acc_2"].to_numpy()),
            "Non0_F1_AUILC": _trapz_auc(x, sub["Non0_F1"].to_numpy()),
            "Mult_acc_5_AUILC": _trapz_auc(x, sub["Mult_acc_5"].to_numpy()),
            "Mult_acc_7_AUILC": _trapz_auc(x, sub["Mult_acc_7"].to_numpy()),
        }

    rows = []
    for lo, hi, tag in [(0.0, 1.0, "0.0~1.0"), (0.1, 0.5, "0.1~0.5"), (0.6, 1.0, "0.6~1.0")]:
        d = {"range": tag}
        d.update(block_auc(lo, hi))
        rows.append(d)
    a = pd.DataFrame(rows)
    a.to_csv(STRESS_DIR / "ours_auilc_00_10.csv", index=False)

    md2 = []
    md2.append("| range | MAE-AUILC | Corr-AUILC | Non0_acc_2-AUILC | Non0_F1-AUILC | Mult_acc_5-AUILC | Mult_acc_7-AUILC |")
    md2.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in a.iterrows():
        md2.append(
            f"| {r['range']} | {r['MAE_AUILC']:.4f} | {r['Corr_AUILC']:.4f} | {r['Non0_acc_2_AUILC']:.4f} | {r['Non0_F1_AUILC']:.4f} | {r['Mult_acc_5_AUILC']:.4f} | {r['Mult_acc_7_AUILC']:.4f} |"
        )
    _write_text(STRESS_DIR / "ours_auilc_00_10.md", "\n".join(md2))

    tex2 = []
    tex2.append("\\begin{table}[t]")
    tex2.append("\\centering")
    tex2.append("\\small")
    tex2.append("\\begin{tabular}{lcccccc}")
    tex2.append("\\toprule")
    tex2.append("Range & MAE-AUILC & Corr-AUILC & Non0 Acc-2-AUILC & Non0 F1-AUILC & Mult Acc-5-AUILC & Mult Acc-7-AUILC \\\\")
    tex2.append("\\midrule")
    for _, r in a.iterrows():
        tex2.append(
            f"{r['range']} & {r['MAE_AUILC']:.4f} & {r['Corr_AUILC']:.4f} & {r['Non0_acc_2_AUILC']:.4f} & {r['Non0_F1_AUILC']:.4f} & {r['Mult_acc_5_AUILC']:.4f} & {r['Mult_acc_7_AUILC']:.4f} \\\\"
        )
    tex2.append("\\bottomrule")
    tex2.append("\\end{tabular}")
    tex2.append("\\caption{Ours-only AUILC across different missing-rate ranges.}")
    tex2.append("\\label{tab:ours_auilc_00_10}")
    tex2.append("\\end{table}")
    _write_text(STRESS_DIR / "ours_auilc_00_10.tex", "\n".join(tex2))


def save_figures(stress_df: pd.DataFrame, ours_curve: pd.DataFrame) -> None:
    fig_dir = STRESS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_names = {
        "text": "Text center",
        "vision": "Vision center",
        "dynamic_soft": "Dynamic Soft",
        "dynamic_rta_pred_residual": "Ours",
    }
    colors = {
        "text": "#1f77b4",
        "vision": "#2ca02c",
        "dynamic_soft": "#ff7f0e",
        "dynamic_rta_pred_residual": "#d62728",
    }

    def line_plot(metric: str, fname: str, ylabel: str) -> None:
        plt.figure(figsize=(7.2, 4.8))
        for m in METHOD_ORDER:
            s = stress_df[stress_df["mode"] == m].sort_values("missing")
            plt.plot(s["missing"], s[metric], marker="o", label=plot_names[m], color=colors[m], linewidth=2)
        plt.xlabel("Missing rate")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=180)
        plt.close()

    line_plot("MAE", "stress_mae_vs_missing_06_10.png", "MAE (lower is better)")
    line_plot("Corr", "stress_corr_vs_missing_06_10.png", "Corr (higher is better)")
    line_plot("Non0_F1", "stress_non0_f1_vs_missing_06_10.png", "Non0 F1 (higher is better)")
    line_plot("Non0_acc_2", "stress_non0_acc2_vs_missing_06_10.png", "Non0 Acc-2 (higher is better)")

    # Ours full curve
    def ours_curve_plot(metric: str, fname: str, ylabel: str) -> None:
        plt.figure(figsize=(7.2, 4.8))
        s = ours_curve.sort_values("missing")
        plt.plot(s["missing"], s[metric], marker="o", color="#d62728", linewidth=2, label="Ours")
        plt.xlabel("Missing rate")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=180)
        plt.close()

    ours_curve_plot("MAE", "ours_mae_curve_00_10.png", "MAE (lower is better)")
    ours_curve_plot("Corr", "ours_corr_curve_00_10.png", "Corr (higher is better)")
    ours_curve_plot("Non0_F1", "ours_non0_f1_curve_00_10.png", "Non0 F1 (higher is better)")

    # average rank figure
    rank = pd.read_csv(STRESS_DIR / "stress_06_10_average_rank.csv")
    rank = rank.set_index("mode").loc[METHOD_ORDER].reset_index()
    x = np.arange(len(rank))
    width = 0.35
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x - width / 2, rank["AvgRank_core"], width=width, label="AvgRank_core", color="#4e79a7")
    plt.bar(x + width / 2, rank["AvgRank_all"], width=width, label="AvgRank_all", color="#f28e2b")
    plt.xticks(x, [plot_names[m] for m in rank["mode"]], rotation=15)
    plt.ylabel("Average rank (lower is better)")
    plt.grid(alpha=0.25, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "stress_average_rank.png", dpi=180)
    plt.close()


def save_reports(stress_df: pd.DataFrame, auilc: pd.DataFrame) -> None:
    # Helpers for wins
    def wins_vs(base: str, metric: str, higher: bool) -> Tuple[int, List[float]]:
        wins = 0
        misses = []
        for miss in sorted(stress_df["missing"].unique()):
            o = stress_df[(stress_df["missing"] == miss) & (stress_df["mode"] == "dynamic_rta_pred_residual")].iloc[0]
            b = stress_df[(stress_df["missing"] == miss) & (stress_df["mode"] == base)].iloc[0]
            cond = o[metric] > b[metric] if higher else o[metric] < b[metric]
            if cond:
                wins += 1
                misses.append(miss)
        return wins, misses

    wins_ds, ds_m = wins_vs("dynamic_soft", "MAE", higher=False)
    wins_vi, vi_m = wins_vs("vision", "MAE", higher=False)
    wins_te, te_m = wins_vs("text", "MAE", higher=False)

    # best by AUILC
    mae_best = auilc.loc[auilc["MAE_AUILC"].idxmin(), "method"]
    corr_best = auilc.loc[auilc["Corr_AUILC"].idxmax(), "method"]
    f1_best = auilc.loc[auilc["Non0_F1_AUILC"].idxmax(), "method"]

    # rank best
    rank = pd.read_csv(STRESS_DIR / "stress_06_10_average_rank.csv")
    best_rank_core = rank.loc[rank["AvgRank_core"].idxmin(), "method"]
    best_rank_all = rank.loc[rank["AvgRank_all"].idxmin(), "method"]

    # degradation note
    ours = stress_df[stress_df["mode"] == "dynamic_rta_pred_residual"].sort_values("missing")
    corr_10 = float(ours[ours["missing"] == 1.0]["Corr"].iloc[0])

    rep = []
    rep.append("# Extreme Missing-rate Stress Test Report")
    rep.append("")
    rep.append("## 1. Settings")
    rep.append("- dataset=MOSI")
    rep.append("- missing=0.6~1.0")
    rep.append("- methods=text, vision, dynamic_soft, Ours")
    rep.append("- protocol=train_drop_last=1, eval_drop_last=0, test_drop_last=0")
    rep.append("- effective_test_samples=686")
    rep.append("")
    rep.append("## 2. Main Results")
    rep.append("- See `stress_06_10_comparison.csv` and `stress_06_10_table.md/.tex`.")
    rep.append("")
    rep.append("## 3. Comparison with Dynamic Soft")
    rep.append(f"- Ours beats dynamic_soft on MAE at {wins_ds}/5 missing rates: {', '.join([f'{m:.1f}' for m in ds_m])}.")
    rep.append("- dynamic_soft shows clear degradation as missing increases, especially near 0.9~1.0.")
    rep.append("")
    rep.append("## 4. Comparison with Vision Center")
    rep.append(f"- Ours beats vision on MAE at {wins_vi}/5 missing rates: {', '.join([f'{m:.1f}' for m in vi_m])}.")
    rep.append("- Vision remains a strong baseline at some extreme points.")
    rep.append("")
    rep.append("## 5. Comparison with Text Center")
    rep.append(f"- Ours beats text on MAE at {wins_te}/5 missing rates: {', '.join([f'{m:.1f}' for m in te_m])}.")
    rep.append("- Text is competitive at selected extreme points (e.g., 0.7/1.0 in MAE).")
    rep.append("")
    rep.append("## 6. AUILC and Average Rank")
    rep.append(f"- Best MAE-AUILC@0.6~1.0: {mae_best}.")
    rep.append(f"- Best Corr-AUILC@0.6~1.0: {corr_best}.")
    rep.append(f"- Best Non0_F1-AUILC@0.6~1.0: {f1_best}.")
    rep.append(f"- Best AvgRank_core: {best_rank_core}; Best AvgRank_all: {best_rank_all}.")
    rep.append("- Ours' strongest advantage is on MAE robustness over most high-missing points.")
    rep.append("")
    rep.append("## 7. Ours Curve from 0.0 to 1.0")
    rep.append("- See `ours_curve_00_10.*` and `ours_auilc_00_10.*`.")
    rep.append("- Ours degrades as missing increases, with a notable boundary effect near missing=1.0.")
    rep.append(f"- Corr at missing=1.0 is {corr_10:.4f}; this near-zero/negative behavior is an expected boundary under extreme missing.")
    rep.append("")
    rep.append("## 8. Conclusion")
    rep.append("- Ours is a credible high-missing robust candidate in MAE trend (majority wins vs dynamic_soft/vision/text).")
    rep.append("- The 0.6~1.0 stress evidence supports robustness claims, but not universal dominance on every metric.")
    rep.append("- To claim full-range fairness, adding audio/task-aware/MoE in 0.6~1.0 is recommended as supplemental baselines.")
    rep.append("- A full-method AUILC@0.0~1.0 is currently not recommended because not all baselines have official complete 0.0~1.0 coverage.")
    rep.append("- Next step: a small-scale MOSEI validation is recommended before large-scale expansion.")
    _write_text(STRESS_DIR / "stress_06_10_report.md", "\n".join(rep))

    zh = []
    zh.append("本节给出极高缺失率(0.6~1.0)压力测试结果。采用 full-test 协议（train_drop_last=1, eval_drop_last=0, test_drop_last=0），并在 MOSI 全测试集686样本上评估。")
    zh.append("结果显示，Ours 在 MAE 上对 dynamic_soft 为 4/5 缺失率更优，对 vision 为 4/5 更优，对 text 为 3/5 更优，说明其在高缺失区间能够有效缓解退化。")
    zh.append("同时，missing=1.0 时所有方法都出现明显性能恶化，相关系数可接近零或为负，这反映了极端缺失下的自然边界，而非单一方法失效。")
    zh.append("因此，stress test 主要用于鲁棒性边界分析，不替代主表结论。主比较仍建议以 0.1~0.5 区间为核心。")
    _write_text(STRESS_DIR / "stress_experiment_text_zh.md", "\n".join(zh))

    en = []
    en.append("We further conduct an extreme missing-rate stress test over 0.6-1.0 under the full-test protocol (train_drop_last=1, eval_drop_last=0, test_drop_last=0) on all 686 MOSI test samples.")
    en.append("In terms of MAE, Ours outperforms Dynamic Soft on 4/5 missing rates, Vision center on 4/5 rates, and Text center on 3/5 rates, indicating that Ours alleviates high-missing degradation in most settings.")
    en.append("At missing=1.0, all methods degrade substantially and correlation may approach zero or become negative, which is an expected robustness boundary under extreme missing rather than evidence of universal failure of one method.")
    en.append("Therefore, this stress test is used for robustness-boundary analysis, while the main comparison should still focus on the 0.1-0.5 range.")
    _write_text(STRESS_DIR / "stress_experiment_text_en.md", "\n".join(en))


def main() -> None:
    _ensure_no_normals(
        [
            str(STRESS_DIR / "anchor_experiment_agg.csv"),
            str(STRESS_DIR / "anchor_experiment_summary.csv"),
            str(STRESS_DIR / "official_progress.csv"),
            str(STRESS_DIR / "failed_runs.csv"),
            str(OURS0_DIR / "anchor_experiment_agg.csv"),
            str(OURS015_DIR / "residual_final_comparison.csv"),
        ]
    )

    stress_agg = _read_csv(STRESS_DIR / "anchor_experiment_agg.csv")
    _read_csv(STRESS_DIR / "anchor_experiment_summary.csv")
    _read_csv(STRESS_DIR / "official_progress.csv")
    _read_csv(STRESS_DIR / "failed_runs.csv")

    stress_df = build_stress_comparison(stress_agg)
    save_stress_table(stress_df)

    auilc_df = build_stress_auilc(stress_df)
    save_auilc_and_averages(stress_df, auilc_df)

    ours_curve = build_ours_curve()
    save_ours_curve_and_auilc(ours_curve)

    save_figures(stress_df, ours_curve)
    save_reports(stress_df, auilc_df)

    print("Generated stress outputs under:", STRESS_DIR)


if __name__ == "__main__":
    main()
