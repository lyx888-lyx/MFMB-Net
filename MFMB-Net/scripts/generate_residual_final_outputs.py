#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CORE_METRICS = ["MAE", "Corr", "Non0_acc_2", "Non0_F1_score"]
ALL_METRICS = ["MAE", "Corr", "Non0_acc_2", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _norm_from_agg(df: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["missing", "method"] + ALL_METRICS)
    out = df.copy()
    if "mode" not in out.columns and "method" in out.columns:
        out["mode"] = out["method"]
    cols = {
        "MAE_mean": "MAE",
        "Corr_mean": "Corr",
        "Non0_acc_2_mean": "Non0_acc_2",
        "Non0_F1_score_mean": "Non0_F1_score",
        "Mult_acc_5_mean": "Mult_acc_5",
        "Mult_acc_7_mean": "Mult_acc_7",
    }
    for k, v in cols.items():
        if k in out.columns and v not in out.columns:
            out[v] = out[k]
    keep = ["missing", "mode"] + [m for m in ALL_METRICS if m in out.columns]
    out = out[keep].copy()
    out["method"] = mode_name if mode_name else out["mode"]
    return out[["missing", "method"] + ALL_METRICS]


def _from_mode_rows(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["missing", "method"] + ALL_METRICS)
    out = []
    tmp = df.copy()
    mode_col = "mode" if "mode" in tmp.columns else "method"
    for mode, method in mapping.items():
        d = tmp[tmp[mode_col] == mode]
        if d.empty:
            continue
        row = d[["missing"]].copy()
        row["method"] = method
        for m in ALL_METRICS:
            if m in d.columns:
                row[m] = d[m].values
            elif f"{m}_mean" in d.columns:
                row[m] = d[f"{m}_mean"].values
            else:
                row[m] = np.nan
        out.append(row)
    if not out:
        return pd.DataFrame(columns=["missing", "method"] + ALL_METRICS)
    return pd.concat(out, ignore_index=True)


def _delta(df: pd.DataFrame, target: str, baseline: str, name: str) -> pd.DataFrame:
    t = df[df["method"] == target].copy()
    b = df[df["method"] == baseline].copy()
    if t.empty or b.empty:
        return pd.DataFrame()
    m = t.merge(b, on="missing", suffixes=("_target", "_base"))
    rows = []
    for _, r in m.iterrows():
        row = {"missing": r["missing"], "target_method": target, "baseline_method": baseline}
        for metric in CORE_METRICS:
            tv = r.get(f"{metric}_target", np.nan)
            bv = r.get(f"{metric}_base", np.nan)
            row[f"{metric}_target"] = tv
            row[f"{metric}_{baseline}"] = bv
            row[f"delta_{metric}"] = tv - bv if pd.notna(tv) and pd.notna(bv) else np.nan
        for metric in ["Mult_acc_5", "Mult_acc_7"]:
            tv = r.get(f"{metric}_target", np.nan)
            bv = r.get(f"{metric}_base", np.nan)
            row[f"{metric}_target"] = tv
            row[f"{metric}_{baseline}"] = bv
            row[f"delta_{metric}"] = tv - bv if pd.notna(tv) and pd.notna(bv) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out["delta_name"] = name
    return out


def _best_fixed(df: pd.DataFrame) -> pd.DataFrame:
    fixed = df[df["method"].isin(["text", "audio", "vision"])].copy()
    if fixed.empty:
        return pd.DataFrame(columns=["missing", "best_fixed_method", "MAE", "Corr", "Non0_acc_2", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7"])
    idx = fixed.groupby("missing")["MAE"].idxmin()
    best = fixed.loc[idx].copy()
    best = best.rename(columns={"method": "best_fixed_method"})
    return best[["missing", "best_fixed_method"] + ALL_METRICS]


def _average_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("method", as_index=False)[ALL_METRICS].mean().rename(columns={
        "MAE": "MAE_avg", "Corr": "Corr_avg", "Non0_acc_2": "Non0_acc_2_avg", "Non0_F1_score": "Non0_F1_avg",
        "Mult_acc_5": "Mult_acc_5_avg", "Mult_acc_7": "Mult_acc_7_avg"
    })


def _average_rank(df: pd.DataFrame) -> pd.DataFrame:
    long_rows = []
    for missing, g in df.groupby("missing"):
        for metric in ALL_METRICS:
            asc = metric == "MAE"
            rr = g[["method", metric]].copy().dropna(subset=[metric])
            if rr.empty:
                continue
            rr["rank"] = rr[metric].rank(ascending=asc, method="average")
            rr["missing"] = missing
            rr["metric"] = metric
            long_rows.append(rr[["missing", "method", "metric", "rank"]])
    if not long_rows:
        return pd.DataFrame()
    long_df = pd.concat(long_rows, ignore_index=True)
    merged = long_df.pivot_table(index=["missing", "method"], columns="metric", values="rank").reset_index()
    merged = merged.rename(columns={m: f"rank_{m}" for m in ALL_METRICS if m in merged.columns})
    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    out = merged.groupby("method", as_index=False)[rank_cols].mean(numeric_only=True)
    core_cols = [f"rank_{m}" for m in CORE_METRICS]
    all_cols = [f"rank_{m}" for m in ALL_METRICS]
    out["avg_rank_core"] = out[core_cols].mean(axis=1)
    out["avg_rank_all"] = out[all_cols].mean(axis=1)
    return out.sort_values(["avg_rank_core", "avg_rank_all"]).reset_index(drop=True)


def _gate_diag(analysis_dir: Path, gate_margin: float = 0.05) -> pd.DataFrame:
    files = sorted(analysis_dir.glob("*.csv"))
    rows = []
    for fp in files:
        try:
            d = pd.read_csv(fp)
        except Exception:
            continue
        if d.empty:
            continue
        # parse missing from filename
        s = fp.name
        miss = np.nan
        try:
            part = s.split("missing", 1)[1].split("_", 1)[0]
            miss = float(part)
        except Exception:
            pass
        av_mean = d[[c for c in ["availability_text", "availability_audio", "availability_vision"] if c in d.columns]].mean(axis=1) if set(["availability_text","availability_audio","availability_vision"]).intersection(d.columns) else pd.Series(dtype=float)
        if "abs_err_diff" not in d.columns and set(["y_true", "pred_rel", "pred_task"]).issubset(d.columns):
            d["err_rel"] = (d["pred_rel"] - d["y_true"]).abs()
            d["err_task"] = (d["pred_task"] - d["y_true"]).abs()
            d["abs_err_diff"] = (d["err_rel"] - d["err_task"]).abs()

        if "gate_margin_mask" in d.columns:
            mm_ratio = d["gate_margin_mask"].mean()
        elif "abs_err_diff" in d.columns:
            mm_ratio = (d["abs_err_diff"] > gate_margin).mean()
        elif set(["err_rel", "err_task"]).issubset(d.columns):
            mm_ratio = ((d["err_rel"] - d["err_task"]).abs() > gate_margin).mean()
        else:
            mm_ratio = np.nan

        rows.append({
            "missing": miss,
            "mean_g_task": d["g_task"].mean() if "g_task" in d.columns else np.nan,
            "mean_g_rel": d["g_rel"].mean() if "g_rel" in d.columns else np.nan,
            "std_g_task": d["g_task"].std() if "g_task" in d.columns else np.nan,
            "mean_err_rel": d["err_rel"].mean() if "err_rel" in d.columns else np.nan,
            "mean_err_task": d["err_task"].mean() if "err_task" in d.columns else np.nan,
            "mean_abs_err_diff": d["abs_err_diff"].mean() if "abs_err_diff" in d.columns else np.nan,
            "margin_mask_ratio": mm_ratio,
            "gate_oracle_match_rate": d["gate_oracle_match"].mean() if "gate_oracle_match" in d.columns else np.nan,
            "mean_oracle_gate_rel": d["oracle_gate_rel"].mean() if "oracle_gate_rel" in d.columns else np.nan,
            "mean_oracle_gate_task": d["oracle_gate_task"].mean() if "oracle_gate_task" in d.columns else np.nan,
            "mean_w_rel_text": d["w_rel_text"].mean() if "w_rel_text" in d.columns else np.nan,
            "mean_w_rel_audio": d["w_rel_audio"].mean() if "w_rel_audio" in d.columns else np.nan,
            "mean_w_rel_vision": d["w_rel_vision"].mean() if "w_rel_vision" in d.columns else np.nan,
            "mean_w_task_text": d["w_task_text"].mean() if "w_task_text" in d.columns else np.nan,
            "mean_w_task_audio": d["w_task_audio"].mean() if "w_task_audio" in d.columns else np.nan,
            "mean_w_task_vision": d["w_task_vision"].mean() if "w_task_vision" in d.columns else np.nan,
            "task_router_oracle_match_rate": d["task_router_oracle_match"].mean() if "task_router_oracle_match" in d.columns else np.nan,
            "corr_g_task_abs_err_diff": d[["g_task", "abs_err_diff"]].corr().iloc[0,1] if set(["g_task","abs_err_diff"]).issubset(d.columns) and len(d)>1 else np.nan,
            "corr_g_task_availability_mean": pd.concat([d["g_task"], av_mean], axis=1).corr().iloc[0,1] if "g_task" in d.columns and len(av_mean)>1 and len(d)>1 else np.nan,
            "corr_g_task_entropy_rel": d[["g_task", "entropy_rel"]].corr().iloc[0,1] if set(["g_task","entropy_rel"]).issubset(d.columns) and len(d)>1 else np.nan,
            "corr_g_task_entropy_task": d[["g_task", "entropy_task"]].corr().iloc[0,1] if set(["g_task","entropy_task"]).issubset(d.columns) and len(d)>1 else np.nan,
            "corr_g_task_abs_pred_rel_task": d[["g_task", "abs_pred_rel_task"]].corr().iloc[0,1] if set(["g_task","abs_pred_rel_task"]).issubset(d.columns) and len(d)>1 else np.nan,
        })
    if not rows:
        return pd.DataFrame()
    gdf = pd.DataFrame(rows)
    agg = gdf.groupby("missing", as_index=False).mean(numeric_only=True)
    return agg.sort_values("missing")


def _fmt(v):
    return "NaN" if pd.isna(v) else f"{v:.4f}"


def _mark_table(dfm: pd.DataFrame, methods: list) -> pd.DataFrame:
    out = dfm.copy()
    for missing, idxs in out.groupby("missing").groups.items():
        sub = out.loc[list(idxs)]
        for metric in ALL_METRICS:
            vals = sub[metric]
            if vals.notna().sum() == 0:
                continue
            if metric == "MAE":
                order = vals.sort_values(ascending=True)
            else:
                order = vals.sort_values(ascending=False)
            best_idx = order.index[0]
            second_idx = order.index[1] if len(order) > 1 else None
            out.loc[best_idx, metric] = f"**{_fmt(out.loc[best_idx, metric])}**"
            if second_idx is not None:
                out.loc[second_idx, metric] = f"_{_fmt(float(sub.loc[second_idx, metric]))}_"
            for i in order.index[2:] if len(order) > 2 else []:
                out.loc[i, metric] = _fmt(float(sub.loc[i, metric]))
    return out


def _to_md_table(df: pd.DataFrame, path: Path, methods: list):
    rows = []
    for m in sorted(df["missing"].unique()):
        s = df[df["missing"] == m].copy()
        s["method"] = pd.Categorical(s["method"], categories=methods, ordered=True)
        s = s.sort_values("method")
        rows.append(s)
    if rows:
        d = pd.concat(rows, ignore_index=True)
    else:
        d = pd.DataFrame(columns=["missing", "method"] + ALL_METRICS)
    marked = _mark_table(d, methods)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| missing | method | MAE | Corr | Non0 Acc-2 | Non0 F1 | Mult Acc-5 | Mult Acc-7 |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for _, r in marked.iterrows():
            f.write(
                f"| {r['missing']:.1f} | {r['method']} | {r['MAE']} | {r['Corr']} | {r['Non0_acc_2']} | {r['Non0_F1_score']} | {r['Mult_acc_5']} | {r['Mult_acc_7']} |\n"
            )


def _to_tex_table(df: pd.DataFrame, path: Path, caption: str, label: str, methods: list):
    rows = []
    for m in sorted(df["missing"].unique()):
        s = df[df["missing"] == m].copy()
        s["method"] = pd.Categorical(s["method"], categories=methods, ordered=True)
        s = s.sort_values("method")
        rows.append(s)
    d = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["missing", "method"]+ALL_METRICS)
    marked = _mark_table(d, methods)
    def md_to_tex(v):
        s = str(v)
        if s.startswith("**") and s.endswith("**") and len(s) >= 4:
            return f"\\textbf{{{s[2:-2]}}}"
        if s.startswith("_") and s.endswith("_") and len(s) >= 2:
            return f"\\underline{{{s[1:-1]}}}"
        return s

    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{c l c c c c c c}\n")
        f.write("\\toprule\n")
        f.write("Missing & Method & MAE$\\downarrow$ & Corr$\\uparrow$ & Non0 Acc-2$\\uparrow$ & Non0 F1$\\uparrow$ & Mult Acc-5$\\uparrow$ & Mult Acc-7$\\uparrow$ \\\\ \n")
        f.write("\\midrule\n")
        for _, r in marked.iterrows():
            vals = [
                md_to_tex(r['MAE']),
                md_to_tex(r['Corr']),
                md_to_tex(r['Non0_acc_2']),
                md_to_tex(r['Non0_F1_score']),
                md_to_tex(r['Mult_acc_5']),
                md_to_tex(r['Mult_acc_7']),
            ]
            f.write(f"{r['missing']:.1f} & {r['method']} & " + " & ".join(vals) + " \\\\ \n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table*}\n")


def _plot_metric(df: pd.DataFrame, metric: str, methods: list, out: Path, ylabel: str):
    plt.figure(figsize=(7, 4.5))
    for method in methods:
        d = df[df["method"] == method].sort_values("missing")
        if d.empty:
            continue
        plt.plot(d["missing"], d[metric], marker="o", label=method)
    plt.xlabel("Missing Rate")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual_dir", default="results/auto_anchor_runs_rta_pred_residual_final")
    ap.add_argument("--fulltest_csv", default="results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv")
    ap.add_argument("--taskaware_csv", default="results/auto_anchor_runs_taskrouter_final/tuned_dynamic_task_soft_0.1_0.5.csv")
    ap.add_argument("--rta_feature_csv", default="results/auto_anchor_runs_rta/rta_final_comparison.csv")
    ap.add_argument("--rta_feature_fallback", default="results/auto_anchor_runs_rta/anchor_experiment_agg.csv")
    ap.add_argument("--rta_pred_csv", default="results/auto_anchor_runs_rta_pred/anchor_experiment_agg.csv")
    args = ap.parse_args()

    residual_dir = Path(args.residual_dir)
    residual_dir.mkdir(parents=True, exist_ok=True)
    (residual_dir / "figures").mkdir(parents=True, exist_ok=True)

    df_res_agg = _read_csv(residual_dir / "anchor_experiment_agg.csv")
    if df_res_agg.empty:
        raise SystemExit(f"Missing residual agg: {residual_dir / 'anchor_experiment_agg.csv'}")

    # residual as method
    df_res = _norm_from_agg(df_res_agg, "dynamic_rta_pred_residual")

    # fulltest baselines
    df_full = _read_csv(Path(args.fulltest_csv))
    mapping_full = {
        "text": "text",
        "audio": "audio",
        "vision": "vision",
        "dynamic_soft": "dynamic_soft",
        "dynamic_soft_moe": "dynamic_soft_moe",
    }
    df_base = _from_mode_rows(df_full, mapping_full)

    # taskaware tuned
    df_ta = _read_csv(Path(args.taskaware_csv))
    if not df_ta.empty:
        if "method" not in df_ta.columns:
            df_ta["method"] = "dynamic_task_soft_tuned"
        else:
            df_ta["method"] = "dynamic_task_soft_tuned"
        for m in ALL_METRICS:
            if m not in df_ta.columns:
                df_ta[m] = np.nan
        df_ta = df_ta[["missing", "method"] + ALL_METRICS]
    else:
        df_ta = pd.DataFrame(columns=["missing", "method"] + ALL_METRICS)

    # feature-level rta
    df_rf = _read_csv(Path(args.rta_feature_csv))
    if df_rf.empty:
        df_rf = _read_csv(Path(args.rta_feature_fallback))
    if not df_rf.empty:
        mode_col = "method" if "method" in df_rf.columns else "mode"
        if mode_col in df_rf.columns:
            if "MAE_mean" in df_rf.columns and "MAE" not in df_rf.columns:
                for m in ALL_METRICS:
                    if f"{m}_mean" in df_rf.columns:
                        df_rf[m] = df_rf[f"{m}_mean"]
            if "mode" not in df_rf.columns:
                df_rf["mode"] = df_rf[mode_col]
            df_rf = df_rf[df_rf["mode"].astype(str).str.contains("dynamic_rta")]
            if not df_rf.empty:
                df_rf["method"] = "dynamic_rta_feature"
                df_rf = df_rf[["missing", "method"] + ALL_METRICS]
    else:
        df_rf = pd.DataFrame(columns=["missing", "method"] + ALL_METRICS)

    # pred quick
    df_rp = _read_csv(Path(args.rta_pred_csv))
    if not df_rp.empty:
        if "MAE_mean" in df_rp.columns and "MAE" not in df_rp.columns:
            for m in ALL_METRICS:
                if f"{m}_mean" in df_rp.columns:
                    df_rp[m] = df_rp[f"{m}_mean"]
        df_rp["method"] = "dynamic_rta_pred"
        df_rp = df_rp[["missing", "method"] + ALL_METRICS]
    else:
        df_rp = pd.DataFrame(columns=["missing", "method"] + ALL_METRICS)

    comp = pd.concat([df_base, df_ta, df_rf, df_rp, df_res], ignore_index=True)
    comp = comp.drop_duplicates(subset=["missing", "method"], keep="last").sort_values(["missing", "method"]).reset_index(drop=True)
    comp.to_csv(residual_dir / "residual_final_comparison.csv", index=False, float_format="%.6f")

    # deltas
    dv_text = _delta(comp, "dynamic_rta_pred_residual", "text", "vs_text")
    dv_ds = _delta(comp, "dynamic_rta_pred_residual", "dynamic_soft", "vs_dynamic_soft")
    dv_ta = _delta(comp, "dynamic_rta_pred_residual", "dynamic_task_soft_tuned", "vs_taskaware")
    bf = _best_fixed(comp)
    if not bf.empty:
        tgt = comp[comp["method"] == "dynamic_rta_pred_residual"].merge(bf, on="missing", suffixes=("_target", "_base"))
        rows = []
        for _, r in tgt.iterrows():
            row = {"missing": r["missing"], "target_method": "dynamic_rta_pred_residual", "baseline_method": r["best_fixed_method"]}
            for metric in ALL_METRICS:
                tv = r.get(f"{metric}_target", np.nan)
                bv = r.get(f"{metric}_base", np.nan)
                row[f"{metric}_target"] = tv
                row[f"{metric}_best_fixed"] = bv
                row[f"delta_{metric}"] = tv - bv if pd.notna(tv) and pd.notna(bv) else np.nan
            rows.append(row)
        dv_bf = pd.DataFrame(rows)
    else:
        dv_bf = pd.DataFrame()

    dv_text.to_csv(residual_dir / "residual_delta_vs_text.csv", index=False, float_format="%.6f")
    dv_ds.to_csv(residual_dir / "residual_delta_vs_dynamic_soft.csv", index=False, float_format="%.6f")
    dv_ta.to_csv(residual_dir / "residual_delta_vs_taskaware.csv", index=False, float_format="%.6f")
    dv_bf.to_csv(residual_dir / "residual_delta_vs_best_fixed.csv", index=False, float_format="%.6f")

    avgm = _average_metrics(comp)
    avgm.to_csv(residual_dir / "residual_average_metrics.csv", index=False, float_format="%.6f")
    avgr = _average_rank(comp)
    avgr.to_csv(residual_dir / "residual_average_rank.csv", index=False, float_format="%.6f")

    gdiag = _gate_diag(residual_dir / "rta_pred_residual_analysis", gate_margin=0.05)
    gdiag.to_csv(residual_dir / "residual_gate_diagnostics.csv", index=False, float_format="%.6f")

    main_methods = ["text", "audio", "vision", "dynamic_soft", "dynamic_task_soft_tuned", "dynamic_rta_pred_residual"]
    main_df = comp[comp["method"].isin(main_methods)].copy()
    _to_md_table(main_df, residual_dir / "residual_main_table.md", main_methods)
    _to_tex_table(main_df, residual_dir / "residual_main_table.tex",
                  "Main comparison under full-test protocol.", "tab:residual_main", main_methods)

    ablation_methods = ["dynamic_soft", "dynamic_soft_moe", "dynamic_task_soft_tuned", "dynamic_rta_feature", "dynamic_rta_pred", "dynamic_rta_pred_residual"]
    abl_df = comp[comp["method"].isin(ablation_methods)].copy()
    _to_md_table(abl_df, residual_dir / "residual_ablation_table.md", ablation_methods)
    _to_tex_table(abl_df, residual_dir / "residual_ablation_table.tex",
                  "Ablation comparison for routing variants.", "tab:residual_ablation", ablation_methods)

    # figures
    _plot_metric(main_df, "MAE", main_methods, residual_dir / "figures/residual_mae_vs_missing.png", "MAE (lower better)")
    _plot_metric(main_df, "Corr", main_methods, residual_dir / "figures/residual_corr_vs_missing.png", "Corr (higher better)")
    _plot_metric(main_df, "Non0_acc_2", main_methods, residual_dir / "figures/residual_non0_acc2_vs_missing.png", "Non0 Acc-2")
    _plot_metric(main_df, "Non0_F1_score", main_methods, residual_dir / "figures/residual_non0_f1_vs_missing.png", "Non0 F1")

    if not gdiag.empty:
        plt.figure(figsize=(7,4.5))
        plt.plot(gdiag["missing"], gdiag["mean_g_task"], marker="o", label="mean_g_task")
        plt.plot(gdiag["missing"], gdiag["mean_g_rel"], marker="o", label="mean_g_rel")
        plt.xlabel("Missing Rate")
        plt.ylabel("Gate Weight")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(residual_dir / "figures/residual_gate_weight_vs_missing.png", dpi=180)
        plt.close()

    if not avgr.empty:
        rr = avgr.sort_values("avg_rank_core")
        plt.figure(figsize=(8,4.8))
        plt.bar(rr["method"], rr["avg_rank_core"])
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Average Rank (core metrics)")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(residual_dir / "figures/residual_average_rank.png", dpi=180)
        plt.close()

    if not dv_ds.empty:
        plt.figure(figsize=(7,4.5))
        d = dv_ds.sort_values("missing")
        plt.plot(d["missing"], d["delta_MAE"], marker="o")
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Missing Rate")
        plt.ylabel("Delta MAE vs dynamic_soft")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(residual_dir / "figures/residual_delta_mae_vs_dynamic_soft.png", dpi=180)
        plt.close()

    # report
    report = []
    report.append("# Residual Prediction-level RTA Report")
    report.append("")
    report.append("## 1. Motivation")
    report.append("Feature-level RTA was unstable. Residual prediction-level gating keeps reliability route as backbone and applies task-aware correction only when useful.")
    report.append("")
    report.append("## 2. Method")
    report.append("`pred_final = pred_rel + g * (pred_task - pred_rel)` with margin-supervised gate.")
    report.append("")
    report.append("## 3. Main Results")
    report.append(f"See `{(residual_dir / 'residual_final_comparison.csv').as_posix()}` and main tables.")
    report.append("")
    report.append("## 4. Comparison with Dynamic Soft")
    if not dv_ds.empty:
        better = int((dv_ds["delta_MAE"] < 0).sum())
        report.append(f"Dynamic residual RTA beats dynamic_soft on MAE in {better}/{len(dv_ds)} missing settings.")
    else:
        report.append("Dynamic soft baseline rows were missing for direct delta comparison.")
    report.append("")
    report.append("## 5. Comparison with Task-aware Router")
    if not dv_ta.empty:
        better = int((dv_ta["delta_MAE"] < 0).sum())
        report.append(f"Dynamic residual RTA beats dynamic_task_soft_tuned on MAE in {better}/{len(dv_ta)} missing settings.")
    else:
        report.append("Task-aware tuned baseline rows were missing for direct delta comparison.")
    report.append("")
    report.append("## 6. Comparison with Best Fixed")
    if not dv_bf.empty:
        better = int((dv_bf["delta_MAE"] < 0).sum())
        report.append(f"Dynamic residual RTA beats best_fixed on MAE in {better}/{len(dv_bf)} missing settings.")
    else:
        report.append("Best fixed baseline could not be constructed.")
    report.append("")
    report.append("## 7. Gate Diagnostics")
    report.append(f"See `{(residual_dir / 'residual_gate_diagnostics.csv').as_posix()}` for `g_task/g_rel`, margin mask ratio, and oracle-match trends.")
    report.append("")
    report.append("## 8. Conclusion")
    report.append("Use residual gate as final Ours if it preserves 0.4 strength and improves 0.5 while keeping stable average rank.")

    (residual_dir / "residual_report.md").write_text("\n".join(report), encoding="utf-8")

    print("[DONE] Generated residual outputs in", residual_dir)


if __name__ == "__main__":
    main()
