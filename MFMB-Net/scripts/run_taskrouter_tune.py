#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PRIORITY_GRID: List[Tuple[float, float]] = [
    (0.05, 0.3),
    (0.1, 0.3),
    (0.2, 0.3),
    (0.1, 0.8),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasetName", type=str, default="mosi")
    p.add_argument("--python_bin", type=str, default="conda run -n mfmb python")
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--output_root", type=str, default="results/auto_anchor_runs_taskrouter_tune")
    p.add_argument("--router_missing_bias", type=float, default=2.0)
    p.add_argument("--center_aux_lambda", type=float, default=0.05)
    p.add_argument("--router_oracle_type", type=str, default="soft", choices=["soft", "hard"])
    p.add_argument("--missing_list", type=str, default="0.4,0.5")
    p.add_argument("--resume", type=int, default=1)
    p.add_argument("--run_all", type=int, default=0, help="1 -> run full 3x3 grid; 0 -> run priority 4")
    p.add_argument("--dry_run", type=int, default=0)
    return p.parse_args()


def fmt_float(v: float) -> str:
    s = f"{v:.2f}"
    return s.rstrip("0").rstrip(".")


def combo_tag(lam: float, temp: float) -> str:
    return f"lambda{fmt_float(lam)}_temp{fmt_float(temp)}"


def run_cmd(cmd: List[str], dry_run: int = 0):
    line = " ".join(shlex.quote(x) for x in cmd)
    print("[CMD]", line)
    if int(dry_run) == 1:
        return 0
    return subprocess.run(cmd).returncode


def completed_combo(combo_out: Path, missing_vals: List[float]) -> bool:
    p = combo_out / "official_progress.csv"
    if not p.exists():
        return False
    try:
        df = pd.read_csv(p)
    except Exception:
        return False
    need = set((float(m), "dynamic_task_soft") for m in missing_vals)
    got = set()
    for _, r in df.iterrows():
        key = (float(r.get("missing", np.nan)), str(r.get("mode", "")))
        if str(r.get("status", "")) == "completed" and int(r.get("completed_seeds", 0)) >= int(r.get("expected_seeds", 3)):
            got.add(key)
    return need.issubset(got)


def gather_combo_rows(combo_out: Path, lam: float, temp: float, center_aux_lambda: float) -> pd.DataFrame:
    cmp_p = combo_out / "taskrouter_vs_fulltest_baselines.csv"
    diag_p = combo_out / "task_router_diagnostics.csv"
    if (not cmp_p.exists()) or (not diag_p.exists()):
        return pd.DataFrame()

    cmp_df = pd.read_csv(cmp_p)
    diag_df = pd.read_csv(diag_p)

    rows = []
    for missing in sorted(cmp_df["missing"].unique()):
        sub = cmp_df[cmp_df["missing"] == missing].copy()
        dt = sub[sub["mode"] == "dynamic_task_soft"]
        ds = sub[sub["mode"] == "dynamic_soft"]
        fixed = sub[sub["mode"].isin(["text", "audio", "vision"])]
        if dt.empty or ds.empty or fixed.empty:
            continue
        dt = dt.iloc[0]
        ds = ds.iloc[0]
        best_fixed = fixed.loc[fixed["MAE"].idxmin()]
        d = diag_df[diag_df["missing"] == missing]
        match = float(d.iloc[0]["router_oracle_match_rate"]) if not d.empty else np.nan

        rows.append({
            "missing": float(missing),
            "task_router_lambda": float(lam),
            "center_aux_lambda": float(center_aux_lambda),
            "router_oracle_temperature": float(temp),
            "router_oracle_type": "soft",
            "MAE": float(dt["MAE"]),
            "Corr": float(dt["Corr"]),
            "Non0_acc_2": float(dt["Non0_acc_2"]),
            "Non0_F1_score": float(dt["Non0_F1_score"]),
            "Mult_acc_5": float(dt["Mult_acc_5"]),
            "Mult_acc_7": float(dt["Mult_acc_7"]),
            "delta_MAE_vs_dynamic_soft": float(dt["MAE"] - ds["MAE"]),
            "delta_Corr_vs_dynamic_soft": float(dt["Corr"] - ds["Corr"]),
            "delta_Non0_acc_2_vs_dynamic_soft": float(dt["Non0_acc_2"] - ds["Non0_acc_2"]),
            "delta_Non0_F1_vs_dynamic_soft": float(dt["Non0_F1_score"] - ds["Non0_F1_score"]),
            "delta_MAE_vs_best_fixed": float(dt["MAE"] - best_fixed["MAE"]),
            "delta_Corr_vs_best_fixed": float(dt["Corr"] - best_fixed["Corr"]),
            "delta_Non0_acc_2_vs_best_fixed": float(dt["Non0_acc_2"] - best_fixed["Non0_acc_2"]),
            "delta_Non0_F1_vs_best_fixed": float(dt["Non0_F1_score"] - best_fixed["Non0_F1_score"]),
            "router_oracle_match_rate": match,
        })
    return pd.DataFrame(rows)


def build_best(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary_df.empty:
        return pd.DataFrame(columns=["selection", "task_router_lambda", "router_oracle_temperature", "center_aux_lambda", "note"])

    for missing in [0.4, 0.5]:
        s = summary_df[summary_df["missing"] == missing]
        if s.empty:
            continue
        b = s.loc[s["MAE"].idxmin()]
        rows.append({
            "selection": f"best_mae_missing_{missing}",
            "task_router_lambda": float(b["task_router_lambda"]),
            "router_oracle_temperature": float(b["router_oracle_temperature"]),
            "center_aux_lambda": float(b["center_aux_lambda"]),
            "MAE": float(b["MAE"]),
            "Corr": float(b["Corr"]),
            "router_oracle_match_rate": float(b["router_oracle_match_rate"]),
            "note": "lowest MAE on this missing",
        })

    grp = summary_df.groupby(["task_router_lambda", "router_oracle_temperature", "center_aux_lambda"], as_index=False).agg(
        avg_MAE=("MAE", "mean"),
        avg_Corr=("Corr", "mean"),
        avg_Non0_acc_2=("Non0_acc_2", "mean"),
        avg_Non0_F1=("Non0_F1_score", "mean"),
        avg_delta_MAE_vs_dynamic_soft=("delta_MAE_vs_dynamic_soft", "mean"),
        avg_router_match=("router_oracle_match_rate", "mean"),
    )
    by_mae = grp.loc[grp["avg_MAE"].idxmin()]
    rows.append({
        "selection": "best_avg_mae_04_05",
        "task_router_lambda": float(by_mae["task_router_lambda"]),
        "router_oracle_temperature": float(by_mae["router_oracle_temperature"]),
        "center_aux_lambda": float(by_mae["center_aux_lambda"]),
        "MAE": float(by_mae["avg_MAE"]),
        "Corr": float(by_mae["avg_Corr"]),
        "router_oracle_match_rate": float(by_mae["avg_router_match"]),
        "note": "lowest average MAE over missing=0.4,0.5",
    })
    by_corr = grp.loc[grp["avg_Corr"].idxmax()]
    rows.append({
        "selection": "best_avg_corr_04_05",
        "task_router_lambda": float(by_corr["task_router_lambda"]),
        "router_oracle_temperature": float(by_corr["router_oracle_temperature"]),
        "center_aux_lambda": float(by_corr["center_aux_lambda"]),
        "MAE": float(by_corr["avg_MAE"]),
        "Corr": float(by_corr["avg_Corr"]),
        "router_oracle_match_rate": float(by_corr["avg_router_match"]),
        "note": "highest average Corr over missing=0.4,0.5",
    })

    # Comprehensive recommendation:
    # 1) average MAE low, 2) 0.5 not degraded vs dynamic_soft, 3) 0.4 not much worse than dynamic_soft, 4) higher match.
    s04 = summary_df[summary_df["missing"] == 0.4][["task_router_lambda", "router_oracle_temperature", "delta_MAE_vs_dynamic_soft"]].rename(
        columns={"delta_MAE_vs_dynamic_soft": "d04"}
    )
    s05 = summary_df[summary_df["missing"] == 0.5][["task_router_lambda", "router_oracle_temperature", "delta_MAE_vs_dynamic_soft"]].rename(
        columns={"delta_MAE_vs_dynamic_soft": "d05"}
    )
    rank_df = grp.merge(s04, on=["task_router_lambda", "router_oracle_temperature"], how="left").merge(
        s05, on=["task_router_lambda", "router_oracle_temperature"], how="left"
    )
    cand = rank_df[(rank_df["d05"] < 0) & (rank_df["d04"] <= 0.05)].copy()
    if cand.empty:
        cand = rank_df.copy()
    cand = cand.sort_values(["avg_MAE", "avg_router_match", "avg_Corr"], ascending=[True, False, False])
    c = cand.iloc[0]
    rows.append({
        "selection": "recommended",
        "task_router_lambda": float(c["task_router_lambda"]),
        "router_oracle_temperature": float(c["router_oracle_temperature"]),
        "center_aux_lambda": float(c["center_aux_lambda"]),
        "MAE": float(c["avg_MAE"]),
        "Corr": float(c["avg_Corr"]),
        "router_oracle_match_rate": float(c["avg_router_match"]),
        "note": "balanced rule: avg MAE + high-missing robustness + router match",
    })
    return pd.DataFrame(rows)


def write_report(report_path: Path, summary_df: pd.DataFrame, best_df: pd.DataFrame, done_combos: List[str], all_combos: List[str]):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Task-aware Router Hyperparameter Tuning Report\n\n")
        f.write("## 1. Settings\n")
        f.write("- dataset: mosi\n")
        f.write("- missing: 0.4, 0.5\n")
        f.write("- mode: dynamic_task_soft\n")
        f.write("- protocol: train_drop_last=1, eval_drop_last=0, test_drop_last=0 (effective_test_samples=686)\n")
        f.write("- grid: task_router_lambda in {0.05,0.1,0.2}, router_oracle_temperature in {0.3,0.5,0.8}, center_aux_lambda=0.05\n")
        f.write(f"- completed combos: {len(done_combos)}/{len(all_combos)}\n\n")

        f.write("## 2. Results by Hyperparameter\n")
        if summary_df.empty:
            f.write("(empty)\n\n")
        else:
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("## 3. Best Settings\n")
        if best_df.empty:
            f.write("(empty)\n\n")
        else:
            f.write(best_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("## 4. Comparison with Baselines\n")
        if summary_df.empty:
            f.write("(empty)\n\n")
        else:
            better_ds = int((summary_df["delta_MAE_vs_dynamic_soft"] < 0).sum())
            better_fixed = int((summary_df["delta_MAE_vs_best_fixed"] < 0).sum())
            f.write(f"- MAE better than dynamic_soft on {better_ds}/{len(summary_df)} points.\n")
            f.write(f"- MAE better than best fixed center on {better_fixed}/{len(summary_df)} points.\n\n")

        f.write("## 5. Router Diagnostics\n")
        if summary_df.empty:
            f.write("(empty)\n")
        else:
            f.write(
                f"- router_oracle_match_rate range: {summary_df['router_oracle_match_rate'].min():.4f} ~ {summary_df['router_oracle_match_rate'].max():.4f}\n"
            )
            f.write("- Higher match generally indicates stronger task-aware routing alignment.\n")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    missing_vals = [float(x.strip()) for x in args.missing_list.split(",") if x.strip()]
    full_grid = [(lam, temp) for lam in [0.05, 0.1, 0.2] for temp in [0.3, 0.5, 0.8]]
    grid = full_grid if int(args.run_all) == 1 else PRIORITY_GRID

    done = []
    all_tags = []
    for lam, temp in grid:
        tag = combo_tag(lam, temp)
        all_tags.append(tag)
        combo_out = output_root / tag
        combo_out.mkdir(parents=True, exist_ok=True)

        if int(args.resume) == 1 and completed_combo(combo_out, missing_vals):
            print(f"[SKIP] {tag} already completed.")
            done.append(tag)
            continue

        cmd = shlex.split(args.python_bin) + [
            "scripts/auto_run_anchor_experiments.py",
            "--datasetName", args.datasetName,
            "--phase", "full",
            "--missing_list", ",".join(str(x) for x in missing_vals),
            "--modes", "dynamic_task_soft",
            "--export_anchor_weights", "1",
            "--export_task_router_info", "1",
            "--router_missing_bias", str(args.router_missing_bias),
            "--task_router_lambda", str(lam),
            "--center_aux_lambda", str(args.center_aux_lambda),
            "--router_oracle_temperature", str(temp),
            "--router_oracle_type", args.router_oracle_type,
            "--gpu_ids", str(args.gpu_ids),
            "--resume", str(args.resume),
            "--train_drop_last", "1",
            "--eval_drop_last", "0",
            "--test_drop_last", "0",
            "--output_dir", str(combo_out),
        ]
        rc = run_cmd(cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"[WARN] combo {tag} returned non-zero: {rc}")
        if completed_combo(combo_out, missing_vals):
            done.append(tag)

    # Merge summaries over completed combos.
    all_rows = []
    for lam, temp in grid:
        tag = combo_tag(lam, temp)
        combo_out = output_root / tag
        df = gather_combo_rows(combo_out, lam, temp, args.center_aux_lambda)
        if not df.empty:
            all_rows.append(df)
    summary_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["missing", "task_router_lambda", "router_oracle_temperature"]
        ).reset_index(drop=True)

    summary_path = output_root / "taskrouter_tune_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")

    best_df = build_best(summary_df)
    best_path = output_root / "taskrouter_tune_best.csv"
    best_df.to_csv(best_path, index=False, float_format="%.6f")

    report_path = output_root / "taskrouter_tune_report.md"
    write_report(report_path, summary_df, best_df, done, all_tags)

    print("Saved:")
    print("-", summary_path)
    print("-", best_path)
    print("-", report_path)
    print(f"Completed combos: {len(done)}/{len(all_tags)}")


if __name__ == "__main__":
    main()
