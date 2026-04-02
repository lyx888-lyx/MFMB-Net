#!/usr/bin/env python3
"""
汇总「te 扫描」目录下所有 mosi-regression-*.csv，绘制 text vs dynamic 随 te 变化曲线。

用法（在 MFMB-Net 目录下）:
  python scripts/plot_mosi_te_curves.py --root results/bench_te_sweep_20260401
  python scripts/plot_mosi_te_curves.py --root results/bench_te_sweep_20260401 --out figures/te_trends.png

依赖: matplotlib, pandas, numpy（与项目一致）
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("请安装 matplotlib: pip install matplotlib") from e

# 文件名示例: mosi-regression-0.0-0.0-0.0-fctext-tc0.0-te0.2-mix.csv
FNAME_RE = re.compile(
    r"mosi-regression-.+?-fc(?P<fc>text|dynamic|audio|vision)-tc(?P<tc>[\d.]+)-te(?P<te>[\d.]+)-(?P<mode>\w+)\.csv$"
)


def parse_tuple_cell(s: str) -> tuple[float, float]:
    s = str(s).strip()
    if not s:
        return float("nan"), float("nan")
    try:
        t = ast.literal_eval(s)
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            return float(t[0]), float(t[1])
        return float(t), 0.0
    except (ValueError, SyntaxError):
        return float("nan"), float("nan")


def collect_csvs(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("mosi-regression*.csv"):
        if p.is_file():
            out.append(p)
    return sorted(out)


def load_rows(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        m = FNAME_RE.search(p.name)
        if not m:
            continue
        fc = m.group("fc")
        te = float(m.group("te"))
        tc = float(m.group("tc"))
        df = pd.read_csv(p)
        if df.empty:
            continue
        r = df.iloc[0].to_dict()
        row = {"path": str(p), "fc": fc, "te": te, "tc": tc, "mode": m.group("mode")}
        for col in df.columns:
            if col == "Model":
                row["model"] = r[col]
                continue
            mean, std = parse_tuple_cell(r[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
        rows.append(row)
    if not rows:
        raise SystemExit(f"未在 {root} 下解析到任何 mosi-regression-*-fc*-tc*-te*.csv")
    return pd.DataFrame(rows)


def plot_df(df: pd.DataFrame, out_path: Path, title_prefix: str = "MOSI") -> None:
    metrics = [
        ("MAE_mean", "MAE_std", "MAE (lower is better)", False),
        ("Corr_mean", "Corr_std", "Corr (higher is better)", True),
        ("Mult_acc_5_mean", "Mult_acc_5_std", "Mult_acc_5 (higher is better)", True),
        ("Mult_acc_7_mean", "Mult_acc_7_std", "Mult_acc_7 (higher is better)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.ravel()
    colors = {"text": "#1f77b4", "dynamic": "#ff7f0e", "audio": "#2ca02c", "vision": "#d62728"}
    styles = {"text": "-o", "dynamic": "-s"}

    for ax, (m_mean, m_std, ylabel, higher_better) in zip(axes, metrics):
        for fc in sorted(df["fc"].unique()):
            sub = df[df["fc"] == fc].sort_values("te")
            if sub.empty:
                continue
            x = sub["te"].values
            y = sub[m_mean].values
            yerr = sub[m_std].values if m_std in sub.columns else np.zeros_like(y)
            lbl = fc
            c = colors.get(fc, None)
            sty = styles.get(fc, "-o")
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                label=lbl,
                color=c,
                fmt=sty,
                capsize=3,
                markersize=5,
                linewidth=1.5,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_title(f"{title_prefix}: text perturbation (eval) sweep — tc fixed per run")
    for ax in axes[2:]:
        ax.set_xlabel(r"text_corrupt_eval ($te$)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="bench_te_sweep_* 根目录（递归搜 CSV）")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出 png 路径，默认 <root>/te_sweep_curves.png",
    )
    ap.add_argument("--title", type=str, default="MOSI")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out_png = Path(args.out) if args.out else root / "te_sweep_curves.png"

    paths = collect_csvs(root)
    print(f"Found {len(paths)} csv files under {root}")
    df = load_rows(paths)
    df = df.sort_values(["te", "fc"])
    summary_csv = root / "te_sweep_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Saved table: {summary_csv}")

    plot_df(df, out_png, title_prefix=args.title)
    print("Done.")


if __name__ == "__main__":
    main()
