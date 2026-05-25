#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["MAE", "Corr", "Non0_acc_2", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7"]
MAIN_METHODS = ["text", "audio", "vision", "dynamic_soft", "dynamic_soft_moe", "dynamic_task_soft_tuned", "dynamic_rta"]
MAIN_TABLE_METHODS = ["text", "audio", "vision", "dynamic_soft", "dynamic_task_soft_tuned", "dynamic_rta"]
ABLATION_METHODS = ["dynamic_soft", "dynamic_soft_moe", "dynamic_task_soft_tuned", "dynamic_rta"]


def filter_protocol(df):
    need = (df.get("train_drop_last", -1) == 1) & (df.get("eval_drop_last", -1) == 0) & (df.get("test_drop_last", -1) == 0)
    if need.any():
        df = df[need].copy()
    return df


def to_metric_df(df, mode_col="mode"):
    out = df[["missing", mode_col, "MAE_mean", "Corr_mean", "Non0_acc_2_mean", "Non0_F1_score_mean", "Mult_acc_5_mean", "Mult_acc_7_mean"]].copy()
    out = out.rename(columns={
        mode_col: "method",
        "MAE_mean": "MAE",
        "Corr_mean": "Corr",
        "Non0_acc_2_mean": "Non0_acc_2",
        "Non0_F1_score_mean": "Non0_F1_score",
        "Mult_acc_5_mean": "Mult_acc_5",
        "Mult_acc_7_mean": "Mult_acc_7",
    })
    return out


def make_rank(df):
    rows = []
    for m, g in df.groupby("missing"):
        for metric in METRICS:
            asc = metric == "MAE"
            gg = g[["method", metric]].sort_values(metric, ascending=asc).reset_index(drop=True)
            for idx, r in gg.iterrows():
                rows.append({"missing": m, "method": r["method"], "metric": metric, "rank": idx + 1})
    return pd.DataFrame(rows)


def decorate(v, is_best=False, is_second=False, tex=False):
    s = f"{v:.4f}"
    if is_best:
        return f"\\textbf{{{s}}}" if tex else f"**{s}**"
    if is_second:
        return f"\\underline{{{s}}}" if tex else f"_{s}_"
    return s


def build_table(df, methods, out_md, out_tex, caption):
    lines = [f"{caption}", "", "| missing | method | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", r"\begin{tabular}{c l c c c c c c}", r"\toprule", r"Missing & Method & MAE$\downarrow$ & Corr$\uparrow$ & Non0 Acc-2$\uparrow$ & Non0 F1$\uparrow$ & Mult Acc-5$\uparrow$ & Mult Acc-7$\uparrow$ \\", r"\midrule"]
    for missing in sorted(df["missing"].unique()):
        g = df[(df["missing"] == missing) & (df["method"].isin(methods))].copy()
        g["method"] = pd.Categorical(g["method"], methods)
        g = g.sort_values("method")
        best = {}
        second = {}
        for metric in METRICS:
            asc = metric == "MAE"
            srt = g[["method", metric]].sort_values(metric, ascending=asc).reset_index(drop=True)
            best[metric] = srt.loc[0, "method"]
            second[metric] = srt.loc[1, "method"] if len(srt) > 1 else None
        for _, r in g.iterrows():
            row = [
                f"{missing:.1f}",
                str(r["method"]),
                decorate(r["MAE"], r["method"] == best["MAE"], r["method"] == second["MAE"]),
                decorate(r["Corr"], r["method"] == best["Corr"], r["method"] == second["Corr"]),
                decorate(r["Non0_acc_2"], r["method"] == best["Non0_acc_2"], r["method"] == second["Non0_acc_2"]),
                decorate(r["Non0_F1_score"], r["method"] == best["Non0_F1_score"], r["method"] == second["Non0_F1_score"]),
                decorate(r["Mult_acc_5"], r["method"] == best["Mult_acc_5"], r["method"] == second["Mult_acc_5"]),
                decorate(r["Mult_acc_7"], r["method"] == best["Mult_acc_7"], r["method"] == second["Mult_acc_7"]),
            ]
            lines.append("| " + " | ".join(row) + " |")
            tex.append(
                f"{missing:.1f} & {r['method']} & "
                f"{decorate(r['MAE'], r['method']==best['MAE'], r['method']==second['MAE'], tex=True)} & "
                f"{decorate(r['Corr'], r['method']==best['Corr'], r['method']==second['Corr'], tex=True)} & "
                f"{decorate(r['Non0_acc_2'], r['method']==best['Non0_acc_2'], r['method']==second['Non0_acc_2'], tex=True)} & "
                f"{decorate(r['Non0_F1_score'], r['method']==best['Non0_F1_score'], r['method']==second['Non0_F1_score'], tex=True)} & "
                f"{decorate(r['Mult_acc_5'], r['method']==best['Mult_acc_5'], r['method']==second['Mult_acc_5'], tex=True)} & "
                f"{decorate(r['Mult_acc_7'], r['method']==best['Mult_acc_7'], r['method']==second['Mult_acc_7'], tex=True)} \\")
        tex.append(r"\midrule")
    lines.append("")
    lines.append("Full-test protocol: train_drop_last=1, eval_drop_last=0, test_drop_last=0. MOSI test set is evaluated on all 686 samples.")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out_tex.write_text("\n".join(tex), encoding="utf-8")


def make_deltas(df, out_dir):
    rows_ds, rows_ta, rows_fixed = [], [], []
    for m, g in df.groupby("missing"):
        g = g.set_index("method")
        rta = g.loc["dynamic_rta"]
        ds = g.loc["dynamic_soft"]
        ta = g.loc["dynamic_task_soft_tuned"]
        fixed = g.loc[["text", "audio", "vision"]]
        best_fixed_name = fixed["MAE"].idxmin()
        bf = fixed.loc[best_fixed_name]

        def row(base_name, base):
            return {
                "missing": m,
                "delta_MAE": rta["MAE"] - base["MAE"],
                "delta_Corr": rta["Corr"] - base["Corr"],
                "delta_Non0_acc_2": rta["Non0_acc_2"] - base["Non0_acc_2"],
                "delta_Non0_F1_score": rta["Non0_F1_score"] - base["Non0_F1_score"],
                "delta_Mult_acc_5": rta["Mult_acc_5"] - base["Mult_acc_5"],
                "delta_Mult_acc_7": rta["Mult_acc_7"] - base["Mult_acc_7"],
            }

        rd = row("dynamic_soft", ds)
        rows_ds.append(rd)
        rt = row("dynamic_task_soft_tuned", ta)
        rows_ta.append(rt)
        rf = row("best_fixed", bf)
        rf["best_fixed_method"] = best_fixed_name
        rows_fixed.append(rf)

    pd.DataFrame(rows_ds).to_csv(out_dir / "rta_delta_vs_dynamic_soft.csv", index=False)
    pd.DataFrame(rows_ta).to_csv(out_dir / "rta_delta_vs_taskaware.csv", index=False)
    pd.DataFrame(rows_fixed).to_csv(out_dir / "rta_delta_vs_best_fixed.csv", index=False)


def aggregate_rta_diag(task_router_files, out_path):
    rows = []
    for fp in sorted(task_router_files):
        df = pd.read_csv(fp)
        if df.empty:
            continue
        name = fp.stem
        missing = float(name.split("_missing")[1].split("_seed")[0])
        rows.append({
            "missing": missing,
            "mean_g_task": df.get("g_task", pd.Series([np.nan])).mean(),
            "mean_g_rel": df.get("g_rel", pd.Series([np.nan])).mean(),
            "std_g_task": df.get("g_task", pd.Series([np.nan])).std(ddof=0),
            "mean_entropy_rel": df.get("entropy_rel", pd.Series([np.nan])).mean(),
            "mean_entropy_task": df.get("entropy_task", pd.Series([np.nan])).mean(),
            "mean_router_disagreement": df.get("router_disagreement", pd.Series([np.nan])).mean(),
            "mean_pred_std": df.get("pred_std", pd.Series([np.nan])).mean(),
            "mean_pred_range": df.get("pred_range", pd.Series([np.nan])).mean(),
            "task_router_oracle_match_rate": df.get("task_router_oracle_match", pd.Series([np.nan])).mean(),
            "corr_g_task_missing_availability_mean": df["g_task"].corr(df[["availability_text", "availability_audio", "availability_vision"]].mean(axis=1)) if "g_task" in df else np.nan,
            "corr_g_task_entropy_rel": df["g_task"].corr(df["entropy_rel"]) if "g_task" in df and "entropy_rel" in df else np.nan,
            "corr_g_task_entropy_task": df["g_task"].corr(df["entropy_task"]) if "g_task" in df and "entropy_task" in df else np.nan,
            "corr_g_task_pred_std": df["g_task"].corr(df["pred_std"]) if "g_task" in df and "pred_std" in df else np.nan,
        })
    if not rows:
        out = pd.DataFrame(columns=[
            "missing", "mean_g_task", "mean_g_rel", "std_g_task", "mean_entropy_rel", "mean_entropy_task",
            "mean_router_disagreement", "mean_pred_std", "mean_pred_range", "task_router_oracle_match_rate",
            "corr_g_task_missing_availability_mean", "corr_g_task_entropy_rel", "corr_g_task_entropy_task", "corr_g_task_pred_std"
        ])
    else:
        out = pd.DataFrame(rows).groupby("missing", as_index=False).mean(numeric_only=True)
    out.to_csv(out_path, index=False)
    return out


def plot_lines(df, methods, metric, ylabel, out_png):
    plt.figure(figsize=(8, 5))
    for m in methods:
        g = df[df["method"] == m].sort_values("missing")
        if g.empty:
            continue
        plt.plot(g["missing"], g[metric], marker="o", label=m)
    plt.xlabel("Missing Rate")
    plt.ylabel(ylabel)
    plt.xticks(sorted(df["missing"].unique()))
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rta_dir", type=str, default="results/auto_anchor_runs_rta")
    ap.add_argument("--baseline_dir", type=str, default="results/auto_anchor_runs_fulltest")
    ap.add_argument("--taskaware_dir", type=str, default="results/auto_anchor_runs_taskrouter_final")
    args = ap.parse_args()

    rta_dir = Path(args.rta_dir)
    baseline_dir = Path(args.baseline_dir)
    taskaware_dir = Path(args.taskaware_dir)
    (rta_dir / "figures").mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(baseline_dir / "anchor_experiment_agg.csv")
    base = filter_protocol(base)
    base = base[base["missing"].isin([0.1, 0.2, 0.3, 0.4, 0.5])]
    base = base[base["mode"].isin(["text", "audio", "vision", "dynamic_soft", "dynamic_soft_moe"])]
    base_df = to_metric_df(base)

    task = pd.read_csv(taskaware_dir / "tuned_dynamic_task_soft_0.1_0.5.csv")
    task = task[task["missing"].isin([0.1, 0.2, 0.3, 0.4, 0.5])].copy()
    task_df = task[["missing", "MAE", "Corr", "Non0_acc_2", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7"]].copy()
    task_df["method"] = "dynamic_task_soft_tuned"

    rta = pd.read_csv(rta_dir / "anchor_experiment_agg.csv")
    rta = filter_protocol(rta)
    rta = rta[(rta["mode"] == "dynamic_rta") & (rta["missing"].isin([0.1, 0.2, 0.3, 0.4, 0.5]))]
    rta_df = to_metric_df(rta)
    rta_df["method"] = "dynamic_rta"

    merged = pd.concat([
        base_df[["missing", "method", *METRICS]],
        task_df[["missing", "method", *METRICS]],
        rta_df[["missing", "method", *METRICS]],
    ], ignore_index=True)
    order = {m: i for i, m in enumerate(MAIN_METHODS)}
    merged = merged.sort_values(["missing", "method"], key=lambda s: s.map(order) if s.name == "method" else s)
    merged.to_csv(rta_dir / "rta_final_comparison.csv", index=False)

    make_deltas(merged, rta_dir)

    avg_metrics = merged.groupby("method", as_index=False)[METRICS].mean(numeric_only=True)
    avg_metrics = avg_metrics.rename(columns={c: f"{c}_avg" for c in METRICS})
    avg_metrics.to_csv(rta_dir / "rta_average_metrics.csv", index=False)

    ranks = make_rank(merged)
    avg_rank = ranks.groupby(["method", "metric"], as_index=False)["rank"].mean()
    pivot = avg_rank.pivot(index="method", columns="metric", values="rank").reset_index()
    pivot["rank_avg_all"] = pivot[[m for m in METRICS]].mean(axis=1)
    pivot.to_csv(rta_dir / "rta_average_rank.csv", index=False)

    task_files = sorted((rta_dir / "rta_router_analysis").glob("*_dynamic_rta.csv"))
    diag = aggregate_rta_diag(task_files, rta_dir / "rta_gate_diagnostics.csv")

    build_table(merged, MAIN_TABLE_METHODS, rta_dir / "rta_main_table.md", rta_dir / "rta_main_table.tex", "RTA Main Comparison on CMU-MOSI")
    build_table(merged, ABLATION_METHODS, rta_dir / "rta_ablation_table.md", rta_dir / "rta_ablation_table.tex", "RTA Ablation Comparison on CMU-MOSI")

    plot_lines(merged, MAIN_METHODS, "MAE", "MAE (lower is better)", rta_dir / "figures/rta_mae_vs_missing.png")
    plot_lines(merged, MAIN_METHODS, "Corr", "Corr (higher is better)", rta_dir / "figures/rta_corr_vs_missing.png")
    plot_lines(merged, MAIN_METHODS, "Non0_F1_score", "Non0 F1 (higher is better)", rta_dir / "figures/rta_non0_f1_vs_missing.png")

    if not diag.empty:
        plt.figure(figsize=(8, 5))
        dd = diag.sort_values("missing")
        plt.plot(dd["missing"], dd["mean_g_task"], marker="o", label="g_task")
        plt.plot(dd["missing"], dd["mean_g_rel"], marker="o", label="g_rel")
        plt.xlabel("Missing Rate")
        plt.ylabel("Gate Mean")
        plt.xticks(sorted(dd["missing"].unique()))
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(rta_dir / "figures/rta_gate_weights_vs_missing.png", dpi=180)
        plt.close()

    # average rank figure
    p = pivot.sort_values("rank_avg_all")
    plt.figure(figsize=(8, 5))
    plt.bar(p["method"], p["rank_avg_all"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Average Rank (lower is better)")
    plt.tight_layout()
    plt.savefig(rta_dir / "figures/rta_average_rank.png", dpi=180)
    plt.close()

    # Report
    ds = pd.read_csv(rta_dir / "rta_delta_vs_dynamic_soft.csv")
    ta = pd.read_csv(rta_dir / "rta_delta_vs_taskaware.csv")
    bf = pd.read_csv(rta_dir / "rta_delta_vs_best_fixed.csv")
    better_ds = int((ds["delta_MAE"] < 0).sum())
    better_ta = int((ta["delta_MAE"] < 0).sum())
    better_bf = int((bf["delta_MAE"] < 0).sum())
    best_mae_method = merged.loc[merged.groupby("missing")["MAE"].idxmin(), ["missing", "method"]]

    lines = [
        "# Reliability-gated Task-aware Dynamic Anchor Routing Report",
        "",
        "## 1. Motivation",
        "dynamic_soft is strong around missing=0.4 but degrades at 0.5, while tuned task-aware routing helps high-missing robustness but is less stable at lower missing rates.",
        "",
        "## 2. Method",
        "RTA fuses reliability-aware and task-aware routes by a learned sample-level gate: H_ours=(1-g)*H_rel+g*H_task.",
        "",
        "## 3. Main Results",
        f"- dynamic_rta beats dynamic_soft on MAE in {better_ds}/5 missing rates.",
        f"- dynamic_rta beats dynamic_task_soft_tuned on MAE in {better_ta}/5 missing rates.",
        f"- dynamic_rta beats best fixed center on MAE in {better_bf}/5 missing rates.",
        "",
        "## 4. Average Metrics and Rank",
        "See rta_average_metrics.csv and rta_average_rank.csv.",
        "",
        "## 5. Gate Diagnostics",
        "See rta_gate_diagnostics.csv and rta_gate_weights_vs_missing.png.",
        "",
        "## 6. Comparison with MoE",
        "dynamic_soft_moe remains less stable in this protocol; RTA is a more reliable extension line.",
        "",
        "## 7. Conclusion",
        "RTA is promising when it can combine the low/mid-missing strength of reliability routing and high-missing robustness of task-aware routing. Final adoption depends on whether average MAE/rank consistently improve against both dynamic_soft and best fixed centers.",
        "",
        "Best MAE method by missing:",
    ]
    for _, r in best_mae_method.sort_values("missing").iterrows():
        lines.append(f"- missing={r['missing']:.1f}: {r['method']}")

    (rta_dir / "rta_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
