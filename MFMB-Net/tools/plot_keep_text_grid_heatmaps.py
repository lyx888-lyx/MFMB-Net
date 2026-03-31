#!/usr/bin/env python3
"""
Build annotated heatmaps for MOSI regression grid: fixed text missing rate t,
audio missing (rows) vs vision missing (columns).

Reads CSVs like: mosi-regression-t0.5_a{audio}_v{vision}-keep_text_grid.csv
Uses the same tuple parsing and /100 scaling as plot_modality_sensitivity.py.
"""
from __future__ import annotations

import argparse
import ast
import glob
import math
import os
import re
from typing import Dict, List, Tuple

# Avoid matplotlib writing to ~/.config when that path is not writable (CI, containers).
_mpl_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib-cache")
os.makedirs(_mpl_cfg, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cfg)

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns

    HAS_SNS = True
except ImportError:
    HAS_SNS = False

FILE_RE = re.compile(
    r"(?P<dataset>[^/]+)-(?P<mode>[^/]+)-t(?P<t>[0-9.]+)_a(?P<a>[0-9.]+)_v(?P<v>[0-9.]+)"
    r"(?:-(?P<tag>.+))?\.csv$"
)
DIV100_METRICS = {"MAE", "Corr", "Loss"}
HIGHER_BETTER = {
    "MAE": False,
    "Corr": True,
    "Loss": False,
    "Non0_F1_score": True,
    "Has0_acc_2": True,
    "Non0_acc_2": True,
    "Has0_F1_score": True,
    "Mult_acc_5": True,
    "Mult_acc_7": True,
}


def parse_tuple_cell(cell) -> Tuple[float, float]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return float("nan"), float("nan")
    if isinstance(cell, (int, float)):
        return float(cell), float("nan")
    text = str(cell).strip()
    if text.startswith("(") and text.endswith(")"):
        try:
            mean, std = ast.literal_eval(text)
            return float(mean), float(std)
        except Exception:
            pass
    try:
        return float(text), float("nan")
    except ValueError:
        return float("nan"), float("nan")


def maybe_scale(metric: str, value: float) -> float:
    if metric in DIV100_METRICS and not math.isnan(value):
        return value / 100.0
    return value


def load_grid(pattern: str, fixed_t: float) -> List[Dict]:
    """Returns structured rows: missing_a, missing_v, metrics dict."""
    rows: List[Dict] = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        m = FILE_RE.match(name)
        if not m:
            continue
        t = float(m.group("t"))
        if abs(t - fixed_t) > 1e-9:
            continue
        import csv

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            lines = list(reader)
        if not lines:
            continue
        last = lines[-1]
        row: Dict = {
            "missing_a": float(m.group("a")),
            "missing_v": float(m.group("v")),
            "path": path,
        }
        for key in last:
            if key in ("Model", "MissingText", "MissingAudio", "MissingVision"):
                continue
            mean, _ = parse_tuple_cell(last[key])
            row[key] = maybe_scale(key, mean)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No CSV matched t={fixed_t}: {pattern}")
    return rows


def pivot_metric(rows: List[Dict], metric: str) -> Tuple[List[List[float]], List[float], List[float]]:
    """Pivot: rows = vision (y-axis), columns = audio (x-axis), matching common paper figures."""
    aset = sorted({r["missing_a"] for r in rows})
    vset = sorted({r["missing_v"] for r in rows})
    nan = float("nan")
    mat = [[nan for _ in aset] for _ in vset]
    for r in rows:
        ia = aset.index(r["missing_a"])
        iv = vset.index(r["missing_v"])
        mat[iv][ia] = r.get(metric, nan)
    return mat, aset, vset


def _cell_color(
    value: float,
    vmin: float,
    vmax: float,
    higher_better: bool,
) -> Tuple[float, float, float]:
    """Map scalar to RGB in [0,1] using a blue-white-red style (good = calm, bad = warm)."""
    if math.isnan(value) or vmax <= vmin:
        return 0.85, 0.85, 0.85
    if higher_better:
        t = (value - vmin) / (vmax - vmin)
    else:
        t = (vmax - value) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    # low t -> bad (red tint), high t -> good (blue-green tint)
    r = 0.85 * (1.0 - t) + 0.15 * t
    g = 0.35 + 0.45 * t
    b = 0.25 + 0.65 * t
    return r, g, b


def save_svg_heatmap(
    mat: List[List[float]],
    a_labels: List[float],
    v_labels: List[float],
    metric: str,
    out_path: str,
    title: str,
    fmt: str,
    higher_better: bool,
) -> None:
    flat = [x for row in mat for x in row if not math.isnan(x)]
    vmin, vmax = min(flat), max(flat)
    nrows = len(mat)
    ncols = len(mat[0]) if mat else 0
    cell_w, cell_h = 88.0, 52.0
    left_margin, top_margin = 72.0, 56.0
    w = left_margin + ncols * cell_w + 40
    h = top_margin + nrows * cell_h + 72

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}">',
        '<style> .axis { font: 14px "IBM Plex Sans", "Segoe UI", sans-serif; fill: #222; } '
        '.title { font: 16px sans-serif; font-weight: 600; fill: #111; } '
        '.celltxt { font: 13px "IBM Plex Mono", "Consolas", monospace; fill: #111; } '
        '.hdr { font: 13px sans-serif; fill: #333; } </style>',
        f'<text x="{w/2:.0f}" y="28" text-anchor="middle" class="title">{esc(title)}</text>',
        f'<text x="{w/2:.0f}" y="46" text-anchor="middle" class="hdr">'
        f'{esc(metric)} — {"higher is better" if higher_better else "lower is better"} '
        f"(min={vmin:{fmt}}, max={vmax:{fmt}})</text>",
    ]

    for j, aa in enumerate(a_labels):
        cx = left_margin + j * cell_w + cell_w / 2
        lines.append(
            f'<text x="{cx:.1f}" y="{top_margin - 8:.1f}" text-anchor="middle" class="axis">a={aa:g}</text>'
        )

    for i, vv in enumerate(v_labels):
        cy = top_margin + i * cell_h + cell_h / 2 + 5
        lines.append(
            f'<text x="{left_margin - 10:.1f}" y="{cy:.1f}" text-anchor="end" class="axis">v={vv:g}</text>'
        )

    for i in range(nrows):
        for j in range(ncols):
            x = left_margin + j * cell_w
            y = top_margin + i * cell_h
            val = mat[i][j]
            r, g, b = _cell_color(val, vmin, vmax, higher_better)
            fill = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 2:.1f}" height="{cell_h - 2:.1f}" '
                f'rx="4" fill="{fill}" stroke="#bbb" stroke-width="1"/>'
            )
            label = "—" if math.isnan(val) else format(val, fmt)
            tx = x + (cell_w - 2) / 2
            ty = y + (cell_h - 2) / 2 + 5
            lines.append(
                f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" class="celltxt">{esc(label)}</text>'
            )

    lines.append(
        f'<text x="{left_margin:.1f}" y="{h - 28:.1f}" class="hdr">Rows: vision missing; '
        f"Columns: audio missing.</text>"
    )
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_svg_combined_vertical(
    panels: List[Tuple[List[List[float]], List[float], List[float], str, str, bool]],
    out_path: str,
    main_title: str,
) -> None:
    """Stack multiple metric heatmaps vertically (same a/v grid)."""
    cell_w, cell_h = 88.0, 52.0
    left_margin = 72.0
    panel_title_h = 24.0
    col_hdr = 22.0
    gap = 20.0

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    ncols = len(panels[0][0][0]) if panels and panels[0][0] else 0
    max_w = left_margin + ncols * cell_w + 40
    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_w:.0f}" height="800" '
        f'viewBox="0 0 {max_w:.0f} 800">',
        '<style> .axis { font: 13px "Segoe UI", sans-serif; fill: #222; } '
        '.title { font: 17px sans-serif; font-weight: 600; fill: #111; } '
        '.celltxt { font: 12px "Consolas", monospace; fill: #111; } '
        '.phdr { font: 14px sans-serif; font-weight: 600; fill: #222; } '
        '.hdr { font: 12px sans-serif; fill: #444; } </style>',
        f'<text x="{max_w/2:.0f}" y="28" text-anchor="middle" class="title">{esc(main_title)}</text>',
    ]

    y = 48.0
    for mat, a_labels, v_labels, metric, fmt, higher_better in panels:
        flat = [x for row in mat for x in row if not math.isnan(x)]
        vmin, vmax = min(flat), max(flat)
        nrows = len(mat)
        ncols = len(mat[0]) if mat else 0
        lines.append(
            f'<text x="{left_margin:.1f}" y="{y:.1f}" class="phdr">{esc(metric)} — '
            f'{"higher is better" if higher_better else "lower is better"} '
            f"(min={vmin:{fmt}}, max={vmax:{fmt}})</text>"
        )
        y += panel_title_h
        for j, aa in enumerate(a_labels):
            cx = left_margin + j * cell_w + cell_w / 2
            lines.append(
                f'<text x="{cx:.1f}" y="{y:.1f}" text-anchor="middle" class="axis">a={aa:g}</text>'
            )
        y += col_hdr
        grid_top = y
        for i, vv in enumerate(v_labels):
            cy = grid_top + i * cell_h + cell_h / 2 + 4
            lines.append(
                f'<text x="{left_margin - 8:.1f}" y="{cy:.1f}" text-anchor="end" class="axis">v={vv:g}</text>'
            )
        for i in range(nrows):
            for j in range(ncols):
                x = left_margin + j * cell_w
                yy = grid_top + i * cell_h
                val = mat[i][j]
                r, g, b_chan = _cell_color(val, vmin, vmax, higher_better)
                fill = f"#{int(r*255):02x}{int(g*255):02x}{int(b_chan*255):02x}"
                lines.append(
                    f'<rect x="{x:.1f}" y="{yy:.1f}" width="{cell_w - 2:.1f}" height="{cell_h - 2:.1f}" '
                    f'rx="4" fill="{fill}" stroke="#bbb" stroke-width="1"/>'
                )
                label = "—" if math.isnan(val) else format(val, fmt)
                tx = x + (cell_w - 2) / 2
                ty = yy + (cell_h - 2) / 2 + 4
                lines.append(
                    f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" class="celltxt">{esc(label)}</text>'
                )
        y = grid_top + nrows * cell_h + gap

    total_h = y + 28
    lines[0] = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_w:.0f}" height="{total_h:.0f}" '
        f'viewBox="0 0 {max_w:.0f} {total_h:.0f}">'
    )
    lines.append(
        f'<text x="{left_margin:.1f}" y="{total_h - 12:.1f}" class="hdr">'
        f"Rows: audio missing; columns: vision missing. "
        f"MAE/Corr/Loss values are raw CSV means divided by 100 (same convention as plot_modality_sensitivity).</text>"
    )
    lines.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_mpl_figure(
    mat: List[List[float]],
    a_labels: List[float],
    v_labels: List[float],
    metric: str,
    out_path: str,
    title: str,
    fmt: str,
    higher_better: bool,
) -> None:
    """Publication-style PNG: x = audio missing, y = vision missing, viridis + annotations."""
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for PNG output.")

    flat = [x for row in mat for x in row if not math.isnan(x)]
    vmin, vmax = min(flat), max(flat)
    cmap_name = "viridis" if higher_better else "viridis_r"
    xlabs = [f"{a:.2f}" for a in a_labels]
    ylabs = [f"{v:.2f}" for v in v_labels]

    if HAS_SNS:
        try:
            import numpy as np

            data = np.asarray(mat, dtype=float)
        except ImportError:
            data = mat
        sns.set_theme(style="white", font_scale=1.05)
        fig, ax = plt.subplots(figsize=(7.4, 6.0))
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.75,
            linecolor="white",
            cbar_kws={"shrink": 0.82, "label": metric, "aspect": 28},
            ax=ax,
            annot_kws={"size": 11, "weight": "semibold"},
            xticklabels=xlabs,
            yticklabels=ylabs,
        )
        ax.set_xlabel("audio missing", fontsize=12, labelpad=8)
        ax.set_ylabel("vision missing", fontsize=12, labelpad=8)
        ax.set_title(title, fontsize=14, fontweight="600", pad=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.invert_yaxis()
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return

    # Fallback: matplotlib only (no seaborn/pandas)
    try:
        import numpy as np

        arr = np.asarray(mat, dtype=float)
    except ImportError:
        arr = mat  # type: ignore[assignment]

    nrows = len(mat)
    ncols = len(mat[0]) if mat else 0
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    try:
        cmap = plt.colormaps[cmap_name]
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
    cbar.set_label(metric, fontsize=11)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(xlabs)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(ylabs)
    ax.set_xlabel("audio missing", fontsize=12)
    ax.set_ylabel("vision missing", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="600", pad=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(nrows):
        for j in range(ncols):
            v = mat[i][j]
            if math.isnan(v):
                continue
            t = format(v, fmt)
            if higher_better:
                frac = (v - vmin) / (vmax - vmin + 1e-12)
            else:
                frac = (vmax - v) / (vmax - vmin + 1e-12)
            frac = max(0.0, min(1.0, frac))
            txt_color = "white" if frac < 0.42 else "#1a1a1a"
            ax.text(j, i, t, ha="center", va="center", color=txt_color, fontsize=11, fontweight="600")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def default_fmt(metric: str) -> str:
    if metric in ("MAE", "Loss"):
        return ".3f"
    if metric == "Corr":
        return ".3f"
    return ".2f"


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotated heatmaps for keep-text grid (fixed t).")
    parser.add_argument(
        "--pattern",
        default="results/results/normals/mosi-regression-t0.5_a*_v*-keep_text_grid.csv",
        help="Glob for result CSVs.",
    )
    parser.add_argument("--fixed_t", type=float, default=0.5, help="Fixed text missing rate.")
    parser.add_argument(
        "--metrics",
        default="MAE,Corr,Non0_F1_score",
        help="Comma-separated metrics.",
    )
    parser.add_argument("--out_dir", default="results/results/normals/heatmaps_t0.5", help="Output directory.")
    parser.add_argument(
        "--formats",
        default="svg,png" if HAS_MPL else "svg",
        help="Comma-separated: svg, png (png needs matplotlib).",
    )
    parser.add_argument(
        "--combined",
        default="MAE,Corr",
        help="Comma-separated metrics for one vertical combined SVG; use empty string to skip.",
    )
    args = parser.parse_args()

    rows = load_grid(args.pattern, args.fixed_t)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    fmts = {m: default_fmt(m) for m in metrics}

    os.makedirs(args.out_dir, exist_ok=True)
    title_base = f"MOSI regression — text missing = {args.fixed_t:g} (keep_text_grid)"

    want = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    for metric in metrics:
        mat, aset, vset = pivot_metric(rows, metric)
        higher = HIGHER_BETTER.get(metric, True)
        fmt = fmts[metric]
        if "svg" in want:
            save_svg_heatmap(
                mat,
                aset,
                vset,
                metric,
                os.path.join(args.out_dir, f"heatmap_{metric}_t{args.fixed_t:g}.svg"),
                title_base,
                fmt,
                higher,
            )
        if "png" in want:
            if not HAS_MPL:
                raise RuntimeError("matplotlib is required for PNG output.")
            png_title = f"{metric} heatmap (keep text intact)"
            save_mpl_figure(
                mat,
                aset,
                vset,
                metric,
                os.path.join(args.out_dir, f"heatmap_{metric}_t{args.fixed_t:g}.png"),
                png_title,
                fmt,
                higher,
            )

    combined = [x.strip() for x in str(args.combined).split(",") if x.strip()]
    if combined and "svg" in want:
        panels: List[Tuple[List[List[float]], List[float], List[float], str, str, bool]] = []
        for metric in combined:
            mat, aset, vset = pivot_metric(rows, metric)
            fmt = fmts.get(metric, default_fmt(metric))
            higher = HIGHER_BETTER.get(metric, True)
            panels.append((mat, aset, vset, metric, fmt, higher))
        tag = "_".join(combined)
        save_svg_combined_vertical(
            panels,
            os.path.join(args.out_dir, f"heatmap_combined_{tag}_t{args.fixed_t:g}.svg"),
            title_base,
        )

    print(f"[OK] wrote heatmaps to: {args.out_dir}")


if __name__ == "__main__":
    main()
