#!/usr/bin/env python3
import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


MODES: List[str] = ["text", "audio", "vision", "dynamic_soft", "dynamic_soft_moe"]
MODE_LABEL = {
    "text": "Fixed Text",
    "audio": "Fixed Audio",
    "vision": "Fixed Vision",
    "dynamic_soft": "Dynamic Soft",
    "dynamic_soft_moe": "Dynamic Soft + MoE",
}
MODE_COLOR = {
    "text": "#1f77b4",
    "audio": "#ff7f0e",
    "vision": "#2ca02c",
    "dynamic_soft": "#d62728",
    "dynamic_soft_moe": "#9467bd",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_metric(agg: pd.DataFrame, metric_col: str, ylabel: str, out_path: str):
    plt.figure(figsize=(7.6, 5.0), dpi=150)
    for mode in MODES:
        sub = agg[agg["mode"] == mode].sort_values("missing")
        if sub.empty:
            continue
        x = sub["missing"].astype(float).tolist()
        y = sub[metric_col].astype(float).tolist()
        plt.plot(
            x, y, marker="o", linewidth=2.0, markersize=4.5,
            label=MODE_LABEL.get(mode, mode), color=MODE_COLOR.get(mode, None)
        )
    plt.xlabel("Missing Rate")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_router_weights(router_df: pd.DataFrame, mode: str, out_path: str):
    sub = router_df[router_df["mode"] == mode].sort_values("missing")
    if sub.empty:
        return
    plt.figure(figsize=(7.6, 5.0), dpi=150)
    x = sub["missing"].astype(float).tolist()
    plt.plot(x, sub["mean_w_text"].astype(float).tolist(), marker="o", linewidth=2.0, label="w_text", color="#1f77b4")
    plt.plot(x, sub["mean_w_audio"].astype(float).tolist(), marker="o", linewidth=2.0, label="w_audio", color="#ff7f0e")
    plt.plot(x, sub["mean_w_vision"].astype(float).tolist(), marker="o", linewidth=2.0, label="w_vision", color="#2ca02c")
    plt.xlabel("Missing Rate")
    plt.ylabel("Average Router Weight")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, fontsize=9)
    plt.title(MODE_LABEL.get(mode, mode))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results/auto_anchor_runs")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    input_dir = args.input_dir
    fig_dir = args.output_dir if args.output_dir else os.path.join(input_dir, "figures")
    agg_path = os.path.join(input_dir, "anchor_experiment_agg.csv")
    router_path = os.path.join(input_dir, "anchor_router_analysis.csv")
    ensure_dir(fig_dir)

    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"Missing agg CSV: {agg_path}")

    agg = pd.read_csv(agg_path)
    if "missing" not in agg.columns or "mode" not in agg.columns:
        raise ValueError("anchor_experiment_agg.csv missing required columns.")

    plot_metric(agg, "MAE_mean", "MAE (lower is better)", os.path.join(fig_dir, "mae_vs_missing.png"))
    plot_metric(agg, "Corr_mean", "Corr (higher is better)", os.path.join(fig_dir, "corr_vs_missing.png"))
    plot_metric(agg, "Non0_acc_2_mean", "Non0_acc_2 (higher is better)", os.path.join(fig_dir, "non0_acc2_vs_missing.png"))
    plot_metric(agg, "Non0_F1_score_mean", "Non0_F1_score (higher is better)", os.path.join(fig_dir, "non0_f1_vs_missing.png"))

    if os.path.exists(router_path):
        router_df = pd.read_csv(router_path)
        if {"missing", "mode", "mean_w_text", "mean_w_audio", "mean_w_vision"}.issubset(router_df.columns):
            plot_router_weights(
                router_df,
                "dynamic_soft",
                os.path.join(fig_dir, "router_weights_vs_missing_dynamic_soft.png"),
            )
            plot_router_weights(
                router_df,
                "dynamic_soft_moe",
                os.path.join(fig_dir, "router_weights_vs_missing_dynamic_soft_moe.png"),
            )

    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
