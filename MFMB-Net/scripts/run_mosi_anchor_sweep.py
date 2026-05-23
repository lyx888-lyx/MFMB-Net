#!/usr/bin/env python3
import os
import shlex
import subprocess
import pandas as pd

MISSING_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MODES = [
    ("text", 0, "text"),
    ("audio", 0, "audio"),
    ("vision", 0, "vision"),
    ("dynamic_soft", 0, "dynamic_soft"),
    ("dynamic_soft", 1, "dynamic_soft+moe"),
]


def run_one(missing, fusion_mode, use_anchor_moe):
    cmd = [
        "python",
        "run.py",
        "--datasetName",
        "mosi",
        "--train_mode",
        "regression",
        "--missing",
        str(missing),
        "--fusion_center_mode",
        fusion_mode,
        "--use_anchor_moe",
        str(use_anchor_moe),
    ]
    if fusion_mode.startswith("dynamic"):
        cmd.extend(["--router_missing_bias", "2.0", "--export_anchor_weights", "1"])

    print("Running:", " ".join(shlex.quote(x) for x in cmd), flush=True)
    before_rows = None
    result_csv = f"results/results/normals/mosi-regression-{missing}.csv"
    if os.path.exists(result_csv):
        before_rows = len(pd.read_csv(result_csv))

    proc = subprocess.run(cmd)

    appended = {}
    if os.path.exists(result_csv):
        df = pd.read_csv(result_csv)
        if len(df) > 0:
            if before_rows is not None and len(df) > before_rows:
                row = df.iloc[-1].to_dict()
            else:
                row = df.iloc[-1].to_dict()
            appended = row

    return proc.returncode, result_csv, appended


def main():
    os.makedirs("results", exist_ok=True)
    summary_rows = []
    for missing in MISSING_VALUES:
        for fusion_mode, use_anchor_moe, mode_label in MODES:
            code, result_csv, appended = run_one(missing, fusion_mode, use_anchor_moe)
            row = {
                "missing": missing,
                "mode": mode_label,
                "fusion_center_mode": fusion_mode,
                "use_anchor_moe": use_anchor_moe,
                "return_code": code,
                "result_csv": result_csv,
            }
            for k, v in appended.items():
                row[f"metric_{k}"] = v
            summary_rows.append(row)

    summary_path = "results/anchor_sweep_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
