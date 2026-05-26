#!/usr/bin/env python3
import argparse
import glob
import os
import re
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

METRICS = [
    "Has0_acc_2",
    "Has0_F1_score",
    "Non0_acc_2",
    "Non0_F1_score",
    "Mult_acc_5",
    "Mult_acc_7",
    "MAE",
    "Corr",
    "Loss",
]
DEFAULT_SEEDS = [111, 1111, 11111]
SUPPORTED_MODES = ["text", "audio", "vision", "dynamic_soft", "dynamic_soft_moe", "dynamic_task_soft", "dynamic_rta", "dynamic_rta_pred", "dynamic_rta_pred_residual"]


@dataclass
class RunConfig:
    dataset: str
    missing: float
    mode: str
    fusion_center_mode: str
    use_anchor_moe: int
    train_drop_last: int
    eval_drop_last: int
    test_drop_last: int
    tag: str = ""


@dataclass
class ParseResult:
    completed: bool
    has_traceback: bool
    test_rows: List[Dict[str, float]]
    reason: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetName", type=str, default="mosi")
    parser.add_argument("--phase", type=str, default="full", choices=["quick", "core", "full", "protocol_check"])
    parser.add_argument("--missing_list", type=str, default=None)
    parser.add_argument("--modes", type=str, default=None)

    parser.add_argument("--export_anchor_weights", type=int, default=1)
    parser.add_argument("--router_missing_bias", type=float, default=2.0)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--python_bin", type=str, default=sys.executable)

    parser.add_argument("--quick_single_seed", type=int, default=0)
    parser.add_argument("--quick_epochs", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--analyze_only", type=int, default=0)
    parser.add_argument("--train_drop_last", type=int, default=1)
    parser.add_argument("--eval_drop_last", type=int, default=0)
    parser.add_argument("--test_drop_last", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/auto_anchor_runs")
    parser.add_argument("--task_router_lambda", type=float, default=0.1)
    parser.add_argument("--center_aux_lambda", type=float, default=0.05)
    parser.add_argument("--router_oracle_temperature", type=float, default=0.5)
    parser.add_argument("--router_oracle_type", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--export_task_router_info", type=int, default=1)
    parser.add_argument("--task_router_output_dir", type=str, default="")
    parser.add_argument("--use_reliability_task_gate", type=int, default=0)
    parser.add_argument("--gate_balance_lambda", type=float, default=0.01)
    parser.add_argument("--gate_target", type=float, default=0.5)
    parser.add_argument("--gate_hidden_dim", type=int, default=32)
    parser.add_argument("--gate_dropout", type=float, default=0.1)
    parser.add_argument("--gate_init_bias", type=float, default=0.0)
    parser.add_argument("--rta_gate_mode", type=str, default="learned", choices=["learned", "force_rel", "force_task", "fixed_half"])
    parser.add_argument("--use_prediction_gate_supervision", type=int, default=1)
    parser.add_argument("--gate_oracle_temperature", type=float, default=0.8)
    parser.add_argument("--gate_task_lambda", type=float, default=0.02)
    parser.add_argument("--rta_pred_residual", type=int, default=0)
    parser.add_argument("--gate_supervision_mode", type=str, default="full", choices=["full", "margin"])
    parser.add_argument("--gate_margin", type=float, default=0.05)
    return parser.parse_args()


def ensure_dirs(out_root: str):
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "logs"), exist_ok=True)


def get_git_meta() -> Tuple[str, str]:
    branch = "unknown"
    commit = "unknown"
    try:
        p = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10)
        if p.returncode == 0 and p.stdout.strip():
            branch = p.stdout.strip()
    except Exception:
        pass
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if p.returncode == 0 and p.stdout.strip():
            commit = p.stdout.strip()
    except Exception:
        pass
    return branch, commit


def make_protocol_tag(train_drop_last: int, eval_drop_last: int, test_drop_last: int) -> str:
    return f"trainDL{int(train_drop_last)}_evalDL{int(eval_drop_last)}_testDL{int(test_drop_last)}"


def make_run_id(dataset: str, missing: float, mode: str, seed: str,
                train_drop_last: int, eval_drop_last: int, test_drop_last: int,
                created_time: Optional[datetime] = None) -> str:
    ts = (created_time or datetime.utcnow()).strftime("%Y%m%d_%H%M")
    return (
        f"{dataset}_m{missing}_{mode}_seed{seed}_"
        f"trainDL{int(train_drop_last)}_evalDL{int(eval_drop_last)}_testDL{int(test_drop_last)}_{ts}"
    )


def parse_start_time(log_path: str) -> Optional[datetime]:
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("[START] "):
                    s = line.replace("[START] ", "").strip().replace("Z", "")
                    return datetime.fromisoformat(s)
    except Exception:
        return None
    return None


def parse_batch_size(log_path: str) -> Optional[int]:
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        m = re.findall(r"'batch_size':\s*(\d+)", txt)
        if m:
            return int(m[-1])
        m = re.findall(r"batch_size:\s*(\d+)", txt)
        if m:
            return int(m[-1])
    except Exception:
        return None
    return None


def resolve_lists(args):
    default_modes = "text,audio,vision,dynamic_soft,dynamic_soft_moe"
    if args.phase == "protocol_check":
        default_missing = "0.0,0.3,0.4,0.5"
        default_modes = "text"
    elif args.phase == "full":
        default_missing = "0.0,0.1,0.2,0.3,0.4,0.5"
    else:
        default_missing = "0.0,0.3"

    missing_str = args.missing_list if args.missing_list is not None else default_missing
    modes_str = args.modes if args.modes is not None else default_modes

    missing_list = [float(x.strip()) for x in missing_str.split(",") if x.strip()]
    modes = [x.strip() for x in modes_str.split(",") if x.strip()]
    return missing_str, modes_str, missing_list, modes


def parse_seeds(seed_str: str) -> List[int]:
    if seed_str and str(seed_str).strip():
        return [int(x.strip()) for x in str(seed_str).split(",") if x.strip()]
    return DEFAULT_SEEDS[:]


def expected_seed_list(args) -> List[int]:
    if int(args.quick_single_seed) == 1:
        seeds = parse_seeds(args.seeds)
        return [seeds[0]] if len(seeds) > 0 else [111]
    return parse_seeds(args.seeds)


def mode_to_flags(
    mode: str,
    export_anchor_weights: int,
    router_missing_bias: float,
    export_task_router_info: int = 0,
    task_router_lambda: float = 0.1,
    center_aux_lambda: float = 0.05,
    router_oracle_temperature: float = 0.5,
    router_oracle_type: str = "soft",
    task_router_output_dir: str = "",
    use_reliability_task_gate: int = 0,
    gate_balance_lambda: float = 0.01,
    gate_target: float = 0.5,
    gate_hidden_dim: int = 32,
    gate_dropout: float = 0.1,
    gate_init_bias: float = 0.0,
    rta_gate_mode: str = "learned",
    use_prediction_gate_supervision: int = 1,
    gate_oracle_temperature: float = 0.8,
    gate_task_lambda: float = 0.02,
    rta_pred_residual: int = 0,
    gate_supervision_mode: str = "full",
    gate_margin: float = 0.05,
) -> Tuple[str, int, List[str]]:
    if mode == "text":
        return "text", 0, []
    if mode == "audio":
        return "audio", 0, []
    if mode == "vision":
        return "vision", 0, []
    if mode == "dynamic_soft":
        flags = ["--router_missing_bias", str(router_missing_bias)]
        if int(export_anchor_weights) == 1:
            flags += ["--export_anchor_weights", "1"]
        return "dynamic_soft", 0, flags
    if mode == "dynamic_soft_moe":
        flags = ["--router_missing_bias", str(router_missing_bias), "--use_anchor_moe", "1"]
        if int(export_anchor_weights) == 1:
            flags += ["--export_anchor_weights", "1"]
        return "dynamic_soft", 1, flags
    if mode == "dynamic_task_soft":
        flags = [
            "--router_missing_bias", str(router_missing_bias),
            "--use_task_aware_router", "1",
            "--task_router_lambda", str(task_router_lambda),
            "--center_aux_lambda", str(center_aux_lambda),
            "--router_oracle_temperature", str(router_oracle_temperature),
            "--router_oracle_type", str(router_oracle_type),
        ]
        if int(export_anchor_weights) == 1:
            flags += ["--export_anchor_weights", "1"]
        if int(export_task_router_info) == 1:
            flags += ["--export_task_router_info", "1"]
        if task_router_output_dir:
            flags += ["--task_router_output_dir", task_router_output_dir]
        return "dynamic_task_soft", 0, flags
    if mode == "dynamic_rta":
        flags = [
            "--router_missing_bias", str(router_missing_bias),
            "--use_task_aware_router", "1",
            "--use_reliability_task_gate", str(int(use_reliability_task_gate)),
            "--task_router_lambda", str(task_router_lambda),
            "--center_aux_lambda", str(center_aux_lambda),
            "--router_oracle_temperature", str(router_oracle_temperature),
            "--router_oracle_type", str(router_oracle_type),
            "--gate_balance_lambda", str(gate_balance_lambda),
            "--gate_target", str(gate_target),
            "--gate_hidden_dim", str(gate_hidden_dim),
            "--gate_dropout", str(gate_dropout),
            "--gate_init_bias", str(gate_init_bias),
            "--rta_gate_mode", str(rta_gate_mode),
        ]
        if int(export_anchor_weights) == 1:
            flags += ["--export_anchor_weights", "1"]
        if int(export_task_router_info) == 1:
            flags += ["--export_task_router_info", "1"]
        if task_router_output_dir:
            flags += ["--task_router_output_dir", task_router_output_dir]
        return "dynamic_rta", 0, flags
    if mode == "dynamic_rta_pred":
        flags = [
            "--router_missing_bias", str(router_missing_bias),
            "--use_task_aware_router", "1",
            "--use_reliability_task_gate", str(int(use_reliability_task_gate)),
            "--use_prediction_gate_supervision", str(int(use_prediction_gate_supervision)),
            "--task_router_lambda", str(task_router_lambda),
            "--center_aux_lambda", str(center_aux_lambda),
            "--router_oracle_temperature", str(router_oracle_temperature),
            "--router_oracle_type", str(router_oracle_type),
            "--gate_oracle_temperature", str(gate_oracle_temperature),
            "--gate_task_lambda", str(gate_task_lambda),
            "--gate_balance_lambda", str(gate_balance_lambda),
            "--gate_target", str(gate_target),
            "--gate_hidden_dim", str(gate_hidden_dim),
            "--gate_dropout", str(gate_dropout),
            "--gate_init_bias", str(gate_init_bias),
            "--rta_gate_mode", str(rta_gate_mode),
        ]
        if int(export_anchor_weights) == 1:
            flags += ["--export_anchor_weights", "1"]
        if int(export_task_router_info) == 1:
            flags += ["--export_task_router_info", "1"]
        if task_router_output_dir:
            flags += ["--task_router_output_dir", task_router_output_dir]
        return "dynamic_rta_pred", 0, flags
    if mode == "dynamic_rta_pred_residual":
        flags = [
            "--router_missing_bias", str(router_missing_bias),
            "--use_task_aware_router", "1",
            "--use_reliability_task_gate", str(int(use_reliability_task_gate)),
            "--use_prediction_gate_supervision", str(int(use_prediction_gate_supervision)),
            "--rta_pred_residual", str(int(rta_pred_residual)),
            "--gate_supervision_mode", str(gate_supervision_mode),
            "--gate_margin", str(gate_margin),
            "--task_router_lambda", str(task_router_lambda),
            "--center_aux_lambda", str(center_aux_lambda),
            "--router_oracle_temperature", str(router_oracle_temperature),
            "--router_oracle_type", str(router_oracle_type),
            "--gate_oracle_temperature", str(gate_oracle_temperature),
            "--gate_task_lambda", str(gate_task_lambda),
            "--gate_balance_lambda", str(gate_balance_lambda),
            "--gate_target", str(gate_target),
            "--gate_hidden_dim", str(gate_hidden_dim),
            "--gate_dropout", str(gate_dropout),
            "--gate_init_bias", str(gate_init_bias),
            "--rta_gate_mode", str(rta_gate_mode),
        ]
        if int(export_anchor_weights) == 1:
            flags += ["--export_anchor_weights", "1"]
        if int(export_task_router_info) == 1:
            flags += ["--export_task_router_info", "1"]
        if task_router_output_dir:
            flags += ["--task_router_output_dir", task_router_output_dir]
        return "dynamic_rta_pred_residual", 0, flags
    raise ValueError(f"Unsupported mode: {mode}")


def make_run_config(
    dataset: str,
    missing: float,
    mode: str,
    export_anchor_weights: int,
    router_missing_bias: float,
    train_drop_last: int,
    eval_drop_last: int,
    test_drop_last: int,
    tag: str = "",
    export_task_router_info: int = 0,
    task_router_lambda: float = 0.1,
    center_aux_lambda: float = 0.05,
    router_oracle_temperature: float = 0.5,
    router_oracle_type: str = "soft",
    task_router_output_dir: str = "",
    use_reliability_task_gate: int = 0,
    gate_balance_lambda: float = 0.01,
    gate_target: float = 0.5,
    gate_hidden_dim: int = 32,
    gate_dropout: float = 0.1,
    gate_init_bias: float = 0.0,
    rta_gate_mode: str = "learned",
    use_prediction_gate_supervision: int = 1,
    gate_oracle_temperature: float = 0.8,
    gate_task_lambda: float = 0.02,
    rta_pred_residual: int = 0,
    gate_supervision_mode: str = "full",
    gate_margin: float = 0.05,
) -> RunConfig:
    center_mode, use_moe, _ = mode_to_flags(
        mode,
        export_anchor_weights,
        router_missing_bias,
        export_task_router_info=export_task_router_info,
        task_router_lambda=task_router_lambda,
        center_aux_lambda=center_aux_lambda,
        router_oracle_temperature=router_oracle_temperature,
        router_oracle_type=router_oracle_type,
        task_router_output_dir=task_router_output_dir,
        use_reliability_task_gate=use_reliability_task_gate,
        gate_balance_lambda=gate_balance_lambda,
        gate_target=gate_target,
        gate_hidden_dim=gate_hidden_dim,
        gate_dropout=gate_dropout,
        gate_init_bias=gate_init_bias,
        rta_gate_mode=rta_gate_mode,
        use_prediction_gate_supervision=use_prediction_gate_supervision,
        gate_oracle_temperature=gate_oracle_temperature,
        gate_task_lambda=gate_task_lambda,
        rta_pred_residual=rta_pred_residual,
        gate_supervision_mode=gate_supervision_mode,
        gate_margin=gate_margin,
    )
    return RunConfig(
        dataset=dataset,
        missing=missing,
        mode=mode,
        fusion_center_mode=center_mode,
        use_anchor_moe=use_moe,
        train_drop_last=int(train_drop_last),
        eval_drop_last=int(eval_drop_last),
        test_drop_last=int(test_drop_last),
        tag=tag,
    )


def detect_run_help_flags(python_bin: str) -> str:
    cmd = shlex.split(python_bin) + ["run.py", "--help"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        return ""


def build_command(cfg: RunConfig, args, seed_list: List[int], run_help_text: str) -> List[str]:
    if args.task_router_output_dir:
        task_router_dir = args.task_router_output_dir
    elif cfg.mode == "dynamic_rta":
        task_router_dir = os.path.join(args.output_dir, "rta_router_analysis")
    elif cfg.mode == "dynamic_rta_pred":
        task_router_dir = os.path.join(args.output_dir, "rta_pred_analysis")
    elif cfg.mode == "dynamic_rta_pred_residual":
        task_router_dir = os.path.join(args.output_dir, "rta_pred_residual_analysis")
    else:
        task_router_dir = os.path.join(args.output_dir, "task_router_analysis")
    _, _, extra_flags = mode_to_flags(
        cfg.mode,
        args.export_anchor_weights,
        args.router_missing_bias,
        export_task_router_info=args.export_task_router_info,
        task_router_lambda=args.task_router_lambda,
        center_aux_lambda=args.center_aux_lambda,
        router_oracle_temperature=args.router_oracle_temperature,
        router_oracle_type=args.router_oracle_type,
        task_router_output_dir=task_router_dir,
        use_reliability_task_gate=args.use_reliability_task_gate,
        gate_balance_lambda=args.gate_balance_lambda,
        gate_target=args.gate_target,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_dropout=args.gate_dropout,
        gate_init_bias=args.gate_init_bias,
        rta_gate_mode=args.rta_gate_mode,
        use_prediction_gate_supervision=args.use_prediction_gate_supervision,
        gate_oracle_temperature=args.gate_oracle_temperature,
        gate_task_lambda=args.gate_task_lambda,
        rta_pred_residual=args.rta_pred_residual,
        gate_supervision_mode=args.gate_supervision_mode,
        gate_margin=args.gate_margin,
    )
    base = shlex.split(args.python_bin) + [
        "run.py",
        "--datasetName", cfg.dataset,
        "--train_mode", "regression",
        "--missing", str(cfg.missing),
        "--fusion_center_mode", cfg.fusion_center_mode,
        "--gpu_ids", str(args.gpu_ids),
        "--train_drop_last", str(cfg.train_drop_last),
        "--eval_drop_last", str(cfg.eval_drop_last),
        "--test_drop_last", str(cfg.test_drop_last),
    ]

    if seed_list:
        base += ["--seeds", ",".join(str(x) for x in seed_list)]

    # quick_epochs is optional and only appended if run.py already supports a known epoch flag.
    if args.quick_epochs is not None and str(args.quick_epochs).strip() and args.phase == "quick":
        for k in ["--epochs", "--num_epochs", "--train_epochs"]:
            if k in run_help_text:
                base += [k, str(args.quick_epochs)]
                break

    return base + extra_flags


def metric_regex(metric: str):
    return re.compile(rf"{re.escape(metric)}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_test_line(line: str) -> Dict[str, float]:
    row = {}
    for m in METRICS:
        rm = metric_regex(m).search(line)
        row[m] = float(rm.group(1)) if rm else np.nan
    return row


def parse_log(log_path: str) -> ParseResult:
    if not os.path.exists(log_path):
        return ParseResult(False, False, [], "log_not_found")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    has_tb = "Traceback (most recent call last)" in text
    test_rows = []
    for line in text.splitlines():
        if "TEST-(" in line and ">>" in line:
            test_rows.append(parse_test_line(line))

    has_finalize = "Results are added to" in text
    completed = (len(test_rows) > 0) and has_finalize and (not has_tb)

    if completed:
        reason = "completed"
    elif has_tb:
        reason = "traceback"
    elif len(test_rows) > 0 and not has_finalize:
        reason = "incomplete_log"
    else:
        reason = "no_test_metrics"

    return ParseResult(completed, has_tb, test_rows, reason)


def parse_effective_samples(log_path: str) -> Dict[str, float]:
    result = {
        "effective_train_samples": np.nan,
        "effective_valid_samples": np.nan,
        "effective_test_samples": np.nan,
    }
    if not os.path.exists(log_path):
        return result
    pat_map = {
        "effective_train_samples": re.compile(r"effective_train_samples:\s*(\d+)"),
        "effective_valid_samples": re.compile(r"effective_valid_samples:\s*(\d+)"),
        "effective_test_samples": re.compile(r"effective_test_samples:\s*(\d+)"),
    }
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    for k, pat in pat_map.items():
        m = pat.findall(text)
        if m:
            result[k] = float(m[-1])
    return result


def parse_drop_last_flags(log_path: str) -> Dict[str, Optional[int]]:
    out = {"train_drop_last": None, "eval_drop_last": None, "test_drop_last": None}
    if not os.path.exists(log_path):
        return out
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        pats = {
            "train_drop_last": re.compile(r"train_drop_last:\s*([01])"),
            "eval_drop_last": re.compile(r"eval_drop_last:\s*([01])"),
            "test_drop_last": re.compile(r"test_drop_last:\s*([01])"),
        }
        for k, p in pats.items():
            m = p.findall(text)
            if m:
                out[k] = int(m[-1])
    except Exception:
        return out
    return out


def parse_logged_command(log_path: str) -> str:
    if not os.path.exists(log_path):
        return ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("[CMD] "):
                    return line.replace("[CMD] ", "").strip()
    except Exception:
        return ""
    return ""


def parse_cmd_flag_value(cmd: str, flag: str) -> Optional[str]:
    if not cmd:
        return None
    try:
        toks = shlex.split(cmd)
    except Exception:
        toks = cmd.split()
    for i, t in enumerate(toks):
        if t == flag and i + 1 < len(toks):
            return toks[i + 1]
    return None


def run_single(cfg: RunConfig, args, out_root: str, seed_list: List[int], run_help_text: str) -> Dict[str, str]:
    missing_tag = str(cfg.missing)
    stage_tag = args.phase
    protocol_tag = make_protocol_tag(cfg.train_drop_last, cfg.eval_drop_last, cfg.test_drop_last)
    seed_tag = "seed" + ",".join(str(x) for x in seed_list) if seed_list else "seedNA"
    tag_suffix = f"_{cfg.tag}" if cfg.tag else ""
    log_path = os.path.join(
        out_root,
        "logs",
        f"{cfg.dataset}_m{missing_tag}_{cfg.mode}_{seed_tag}_{protocol_tag}{tag_suffix}_{stage_tag}.log",
    )
    cmd = build_command(cfg, args, seed_list, run_help_text)

    existing = parse_log(log_path)
    if int(args.resume) == 1 and existing.completed and len(existing.test_rows) >= len(seed_list):
        return {
            "dataset": cfg.dataset,
            "missing": str(cfg.missing),
            "mode": cfg.mode,
            "phase": args.phase,
            "protocol_tag": protocol_tag,
            "train_drop_last": str(cfg.train_drop_last),
            "eval_drop_last": str(cfg.eval_drop_last),
            "test_drop_last": str(cfg.test_drop_last),
            "seeds": ",".join(str(x) for x in seed_list),
            "tag": cfg.tag,
            "status": "skipped_completed",
            "reason": "existing_complete_log",
            "log_path": log_path,
            "command": " ".join(shlex.quote(x) for x in cmd),
        }

    if int(args.dry_run) == 1:
        print("[DRY RUN]", " ".join(shlex.quote(x) for x in cmd))
        return {
            "dataset": cfg.dataset,
            "missing": str(cfg.missing),
            "mode": cfg.mode,
            "phase": args.phase,
            "protocol_tag": protocol_tag,
            "train_drop_last": str(cfg.train_drop_last),
            "eval_drop_last": str(cfg.eval_drop_last),
            "test_drop_last": str(cfg.test_drop_last),
            "seeds": ",".join(str(x) for x in seed_list),
            "tag": cfg.tag,
            "status": "dry_run",
            "reason": "not_executed",
            "log_path": log_path,
            "command": " ".join(shlex.quote(x) for x in cmd),
        }

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    project_log_path = os.path.join("logs", f"mfmb_net-{cfg.dataset}.log")
    project_log_start = os.path.getsize(project_log_path) if os.path.exists(project_log_path) else 0

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[START] {datetime.utcnow().isoformat()}Z\n")
        f.write(f"[PHASE] {args.phase}\n")
        f.write(f"[SEEDS] {','.join(str(x) for x in seed_list)}\n")
        f.write("[CMD] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n[END] return_code={proc.returncode} time={datetime.utcnow().isoformat()}Z\n")

        f.write("\n[PROJECT_LOG_APPEND_BEGIN]\n")
        if os.path.exists(project_log_path):
            project_log_end = os.path.getsize(project_log_path)
            if project_log_end > project_log_start:
                with open(project_log_path, "rb") as pf:
                    pf.seek(project_log_start)
                    chunk = pf.read(project_log_end - project_log_start)
                f.write(chunk.decode("utf-8", errors="ignore"))
            else:
                f.write("(no appended project log chunk)\n")
        else:
            f.write(f"(project log not found: {project_log_path})\n")
        f.write("[PROJECT_LOG_APPEND_END]\n")

    parsed = parse_log(log_path)
    if proc.returncode == 0 and parsed.completed and len(parsed.test_rows) >= len(seed_list):
        status = "success"
        reason = "completed"
    elif proc.returncode != 0:
        status = "failed"
        reason = f"return_code_{proc.returncode}_{parsed.reason}"
    elif len(parsed.test_rows) > 0:
        status = "failed"
        reason = f"insufficient_seed_rows_{len(parsed.test_rows)}_expected_{len(seed_list)}"
    else:
        status = "failed"
        reason = parsed.reason

    return {
        "dataset": cfg.dataset,
        "missing": str(cfg.missing),
        "mode": cfg.mode,
        "phase": args.phase,
        "protocol_tag": protocol_tag,
        "train_drop_last": str(cfg.train_drop_last),
        "eval_drop_last": str(cfg.eval_drop_last),
        "test_drop_last": str(cfg.test_drop_last),
        "seeds": ",".join(str(x) for x in seed_list),
        "tag": cfg.tag,
        "status": status,
        "reason": reason,
        "log_path": log_path,
        "command": " ".join(shlex.quote(x) for x in cmd),
    }


def build_summary_rows(run_records: List[Dict[str, str]], args, seed_list: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, failed = [], []
    def _safe_int(v, default=-1):
        try:
            return int(float(v))
        except Exception:
            return int(default)

    for rec in run_records:
        cfg_missing = float(rec["missing"])
        parsed = parse_log(rec["log_path"])
        eff = parse_effective_samples(rec["log_path"])
        dl = parse_drop_last_flags(rec["log_path"])
        def _dl_norm(v, fallback):
            try:
                iv = int(float(v))
                if iv in (0, 1):
                    return iv
            except Exception:
                pass
            return int(fallback)
        td = _dl_norm(rec.get("train_drop_last", args.train_drop_last),
                      dl["train_drop_last"] if dl["train_drop_last"] is not None else args.train_drop_last)
        ed = _dl_norm(rec.get("eval_drop_last", args.eval_drop_last),
                      dl["eval_drop_last"] if dl["eval_drop_last"] is not None else args.eval_drop_last)
        xd = _dl_norm(rec.get("test_drop_last", args.test_drop_last),
                      dl["test_drop_last"] if dl["test_drop_last"] is not None else args.test_drop_last)
        bs = parse_batch_size(rec["log_path"])
        logged_cmd = rec.get("command", "") or parse_logged_command(rec["log_path"])
        center_mode, use_moe, _ = mode_to_flags(rec["mode"], args.export_anchor_weights, args.router_missing_bias)
        protocol_tag = rec.get("protocol_tag") or make_protocol_tag(td, ed, xd)

        task_router_lambda = parse_cmd_flag_value(logged_cmd, "--task_router_lambda")
        center_aux_lambda = parse_cmd_flag_value(logged_cmd, "--center_aux_lambda")
        router_oracle_temperature = parse_cmd_flag_value(logged_cmd, "--router_oracle_temperature")
        router_oracle_type = parse_cmd_flag_value(logged_cmd, "--router_oracle_type")
        gate_balance_lambda = parse_cmd_flag_value(logged_cmd, "--gate_balance_lambda")
        gate_target = parse_cmd_flag_value(logged_cmd, "--gate_target")
        rta_gate_mode = parse_cmd_flag_value(logged_cmd, "--rta_gate_mode")
        use_prediction_gate_supervision = parse_cmd_flag_value(logged_cmd, "--use_prediction_gate_supervision")
        gate_oracle_temperature = parse_cmd_flag_value(logged_cmd, "--gate_oracle_temperature")
        gate_task_lambda = parse_cmd_flag_value(logged_cmd, "--gate_task_lambda")
        router_missing_bias = parse_cmd_flag_value(logged_cmd, "--router_missing_bias")
        use_task_aware_router = parse_cmd_flag_value(logged_cmd, "--use_task_aware_router")
        use_reliability_task_gate = parse_cmd_flag_value(logged_cmd, "--use_reliability_task_gate")
        export_anchor_weights = parse_cmd_flag_value(logged_cmd, "--export_anchor_weights")
        export_task_router_info = parse_cmd_flag_value(logged_cmd, "--export_task_router_info")
        output_dir = parse_cmd_flag_value(logged_cmd, "--task_router_output_dir") or args.output_dir

        def _f(v, d):
            try:
                return float(v)
            except Exception:
                return float(d)

        def _i(v, d):
            try:
                return int(float(v))
            except Exception:
                return int(d)

        if rec["status"] in ["failed", "dry_run"] or len(parsed.test_rows) == 0:
            failed.append({
                "dataset": rec["dataset"],
                "missing": cfg_missing,
                "mode": rec["mode"],
                "phase": rec["phase"],
                "status": rec["status"],
                "reason": rec["reason"],
                "log_path": rec["log_path"],
                "command": rec["command"],
            })

            row = {
                "dataset": rec["dataset"],
                "missing": cfg_missing,
                "mode": rec["mode"],
                "phase": rec["phase"],
                "protocol_tag": protocol_tag,
                "fusion_center_mode": center_mode,
                "use_anchor_moe": use_moe,
                "seed": np.nan,
                "train_drop_last": td,
                "eval_drop_last": ed,
                "test_drop_last": xd,
                "task_router_lambda": _f(task_router_lambda, args.task_router_lambda),
                "center_aux_lambda": _f(center_aux_lambda, args.center_aux_lambda),
                "router_oracle_temperature": _f(router_oracle_temperature, args.router_oracle_temperature),
                "router_oracle_type": (router_oracle_type or args.router_oracle_type),
                "router_missing_bias": _f(router_missing_bias, args.router_missing_bias),
                "gate_balance_lambda": _f(gate_balance_lambda, args.gate_balance_lambda),
                "gate_target": _f(gate_target, args.gate_target),
                "rta_gate_mode": (rta_gate_mode or args.rta_gate_mode),
                "use_prediction_gate_supervision": _i(use_prediction_gate_supervision, args.use_prediction_gate_supervision),
                "gate_oracle_temperature": _f(gate_oracle_temperature, args.gate_oracle_temperature),
                "gate_task_lambda": _f(gate_task_lambda, args.gate_task_lambda),
                "use_task_aware_router": _i(use_task_aware_router, 0),
                "use_reliability_task_gate": _i(use_reliability_task_gate, args.use_reliability_task_gate),
                "export_anchor_weights": _i(export_anchor_weights, args.export_anchor_weights),
                "export_task_router_info": _i(export_task_router_info, args.export_task_router_info),
                "train_batch_size": bs if bs is not None else np.nan,
                "eval_batch_size": bs if bs is not None else np.nan,
                "test_batch_size": bs if bs is not None else np.nan,
                "output_dir": output_dir,
                "effective_train_samples": eff["effective_train_samples"],
                "effective_valid_samples": eff["effective_valid_samples"],
                "effective_test_samples": eff["effective_test_samples"],
                "status": rec["status"],
                "log_path": rec["log_path"],
            }
            row["run_id"] = make_run_id(
                row["dataset"], row["missing"], row["mode"], "NA",
                row["train_drop_last"], row["eval_drop_last"], row["test_drop_last"],
                created_time=parse_start_time(rec["log_path"]),
            )
            for m in METRICS:
                row[m] = np.nan
            rows.append(row)
            continue

        for i, mrow in enumerate(parsed.test_rows):
            seed_val = seed_list[i] if i < len(seed_list) else (i + 1)
            row = {
                "dataset": rec["dataset"],
                "missing": cfg_missing,
                "mode": rec["mode"],
                "phase": rec["phase"],
                "protocol_tag": protocol_tag,
                "fusion_center_mode": center_mode,
                "use_anchor_moe": use_moe,
                "seed": seed_val,
                "train_drop_last": td,
                "eval_drop_last": ed,
                "test_drop_last": xd,
                "task_router_lambda": _f(task_router_lambda, args.task_router_lambda),
                "center_aux_lambda": _f(center_aux_lambda, args.center_aux_lambda),
                "router_oracle_temperature": _f(router_oracle_temperature, args.router_oracle_temperature),
                "router_oracle_type": (router_oracle_type or args.router_oracle_type),
                "router_missing_bias": _f(router_missing_bias, args.router_missing_bias),
                "gate_balance_lambda": _f(gate_balance_lambda, args.gate_balance_lambda),
                "gate_target": _f(gate_target, args.gate_target),
                "rta_gate_mode": (rta_gate_mode or args.rta_gate_mode),
                "use_prediction_gate_supervision": _i(use_prediction_gate_supervision, args.use_prediction_gate_supervision),
                "gate_oracle_temperature": _f(gate_oracle_temperature, args.gate_oracle_temperature),
                "gate_task_lambda": _f(gate_task_lambda, args.gate_task_lambda),
                "use_task_aware_router": _i(use_task_aware_router, 0),
                "use_reliability_task_gate": _i(use_reliability_task_gate, args.use_reliability_task_gate),
                "export_anchor_weights": _i(export_anchor_weights, args.export_anchor_weights),
                "export_task_router_info": _i(export_task_router_info, args.export_task_router_info),
                "train_batch_size": bs if bs is not None else np.nan,
                "eval_batch_size": bs if bs is not None else np.nan,
                "test_batch_size": bs if bs is not None else np.nan,
                "output_dir": output_dir,
                "effective_train_samples": eff["effective_train_samples"],
                "effective_valid_samples": eff["effective_valid_samples"],
                "effective_test_samples": eff["effective_test_samples"],
                "status": "success" if parsed.completed else "partial",
                "log_path": rec["log_path"],
            }
            row["run_id"] = make_run_id(
                row["dataset"], row["missing"], row["mode"], str(seed_val),
                row["train_drop_last"], row["eval_drop_last"], row["test_drop_last"],
                created_time=parse_start_time(rec["log_path"]),
            )
            for m in METRICS:
                row[m] = mrow.get(m, np.nan)
            rows.append(row)

        if len(parsed.test_rows) < len(seed_list):
            failed.append({
                "dataset": rec["dataset"],
                "missing": cfg_missing,
                "mode": rec["mode"],
                "phase": rec["phase"],
                "status": "partial",
                "reason": f"insufficient_seed_rows_{len(parsed.test_rows)}_expected_{len(seed_list)}",
                "log_path": rec["log_path"],
                "command": rec["command"],
            })

    summary_df = pd.DataFrame(rows)
    failed_df = pd.DataFrame(failed)

    if not summary_df.empty:
        for c in METRICS:
            if c in summary_df.columns:
                summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce")
    return summary_df, failed_df


def build_agg(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    ok = summary_df[summary_df["status"].isin(["success", "partial"])].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = [
        "dataset", "phase", "missing", "mode", "protocol_tag",
        "train_drop_last", "eval_drop_last", "test_drop_last",
        "task_router_lambda", "center_aux_lambda",
        "router_oracle_temperature", "router_oracle_type",
        "gate_balance_lambda", "gate_target",
        "rta_gate_mode", "use_prediction_gate_supervision",
        "gate_oracle_temperature", "gate_task_lambda",
    ]
    agg_items = {}
    for m in ["MAE", "Corr", "Non0_acc_2", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7"]:
        agg_items[f"{m}_mean"] = (m, "mean")
        agg_items[f"{m}_std"] = (m, "std")

    agg_df = ok.groupby(group_cols, as_index=False).agg(**agg_items)
    if "effective_test_samples" in ok.columns:
        eff = ok.groupby(group_cols, as_index=False).agg(
            effective_test_samples=("effective_test_samples", "max")
        )
        agg_df = agg_df.merge(eff, on=group_cols, how="left")
    return agg_df


def build_delta_vs_text(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()

    base = agg_df[agg_df["mode"] == "text"].copy()
    if base.empty:
        return pd.DataFrame()

    base = base.rename(columns={
        "MAE_mean": "MAE_text",
        "Corr_mean": "Corr_text",
        "Non0_acc_2_mean": "Non0_acc_2_text",
        "Non0_F1_score_mean": "Non0_F1_score_text",
        "Mult_acc_5_mean": "Mult_acc_5_text",
        "Mult_acc_7_mean": "Mult_acc_7_text",
    })

    merge_keys = ["dataset", "phase", "missing", "protocol_tag", "train_drop_last", "eval_drop_last", "test_drop_last"]
    merged = agg_df.merge(
        base[[
            "dataset", "phase", "missing", "protocol_tag", "train_drop_last", "eval_drop_last", "test_drop_last", "MAE_text", "Corr_text",
            "Non0_acc_2_text", "Non0_F1_score_text", "Mult_acc_5_text", "Mult_acc_7_text"
        ]],
        on=merge_keys,
        how="left",
    )

    merged["delta_MAE"] = merged["MAE_mean"] - merged["MAE_text"]
    merged["delta_Corr"] = merged["Corr_mean"] - merged["Corr_text"]
    merged["delta_Non0_acc_2"] = merged["Non0_acc_2_mean"] - merged["Non0_acc_2_text"]
    merged["delta_Non0_F1_score"] = merged["Non0_F1_score_mean"] - merged["Non0_F1_score_text"]
    merged["delta_Mult_acc_5"] = merged["Mult_acc_5_mean"] - merged["Mult_acc_5_text"]
    merged["delta_Mult_acc_7"] = merged["Mult_acc_7_mean"] - merged["Mult_acc_7_text"]

    cols = [
        "dataset", "phase", "missing", "mode", "protocol_tag", "train_drop_last", "eval_drop_last", "test_drop_last",
        "MAE_mean", "MAE_text", "delta_MAE",
        "Corr_mean", "Corr_text", "delta_Corr",
        "Non0_acc_2_mean", "Non0_acc_2_text", "delta_Non0_acc_2",
        "Non0_F1_score_mean", "Non0_F1_score_text", "delta_Non0_F1_score",
        "Mult_acc_5_mean", "Mult_acc_5_text", "delta_Mult_acc_5",
        "Mult_acc_7_mean", "Mult_acc_7_text", "delta_Mult_acc_7",
    ]
    return merged[cols]


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def analyze_anchor_router(output_dir: str, missing_filter: Optional[List[float]] = None) -> pd.DataFrame:
    candidates = []
    for pattern in [
        "results/anchor_weights/*.csv",
        os.path.join(output_dir, "*.csv"),
        os.path.join(output_dir, "**", "*.csv"),
    ]:
        candidates.extend(glob.glob(pattern, recursive=True))

    files = []
    for p in sorted(set(candidates)):
        n = os.path.basename(p).lower()
        p_low = p.lower()
        if n.endswith(".csv") and (("anchor" in n) or ("anchor_weights" in p_low)) and ("weight" in n or "dynamic_soft" in n or "moe" in n):
            files.append(p)

    rows = []
    file_re = re.compile(r"(?P<dataset>[a-zA-Z0-9]+)_missing(?P<missing>[^_]+)_seed(?P<seed>[^_]+)_(?P<mode>[^.]+)\.csv")

    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not {"w_text", "w_audio", "w_vision"}.issubset(df.columns):
            continue

        m = file_re.search(os.path.basename(p))
        dataset = m.group("dataset") if m else "unknown"
        missing_raw = m.group("missing") if m else "nan"
        mode = m.group("mode") if m else "unknown"
        if mode not in {"dynamic_soft", "dynamic_soft_moe", "dynamic_task_soft", "dynamic_rta", "dynamic_rta_pred"}:
            continue
        seed = m.group("seed") if m else "unknown"

        try:
            missing = float(missing_raw)
        except Exception:
            missing = np.nan
        if missing_filter is not None and len(missing_filter) > 0:
            if np.isnan(missing) or (float(missing) not in [float(x) for x in missing_filter]):
                continue

        wt = pd.to_numeric(df["w_text"], errors="coerce")
        wa = pd.to_numeric(df["w_audio"], errors="coerce")
        wv = pd.to_numeric(df["w_vision"], errors="coerce")

        if "selected_anchor" in df.columns:
            sel = df["selected_anchor"].astype(str).str.lower()
            sel_t = float((sel == "text").mean())
            sel_a = float((sel == "audio").mean())
            sel_v = float((sel == "vision").mean())
        else:
            arr = np.vstack([wt.fillna(0), wa.fillna(0), wv.fillna(0)]).T
            idx = np.argmax(arr, axis=1)
            sel_t = float(np.mean(idx == 0))
            sel_a = float(np.mean(idx == 1))
            sel_v = float(np.mean(idx == 2))

        entropy = -(wt * np.log(wt + 1e-8) + wa * np.log(wa + 1e-8) + wv * np.log(wv + 1e-8))

        row = {
            "dataset": dataset,
            "missing": missing,
            "mode": mode,
            "seed": seed,
            "file_path": p,
            "mean_w_text": float(wt.mean()),
            "mean_w_audio": float(wa.mean()),
            "mean_w_vision": float(wv.mean()),
            "selected_text_ratio": sel_t,
            "selected_audio_ratio": sel_a,
            "selected_vision_ratio": sel_v,
            "entropy": float(entropy.mean()),
            "corr_availability_text_w_text": np.nan,
            "corr_availability_audio_w_audio": np.nan,
            "corr_availability_vision_w_vision": np.nan,
        }

        if "availability_text" in df.columns:
            row["corr_availability_text_w_text"] = safe_corr(pd.to_numeric(df["availability_text"], errors="coerce"), wt)
        if "availability_audio" in df.columns:
            row["corr_availability_audio_w_audio"] = safe_corr(pd.to_numeric(df["availability_audio"], errors="coerce"), wa)
        if "availability_vision" in df.columns:
            row["corr_availability_vision_w_vision"] = safe_corr(pd.to_numeric(df["availability_vision"], errors="coerce"), wv)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "dataset",
            "missing",
            "mode",
            "mean_w_text",
            "mean_w_audio",
            "mean_w_vision",
            "selected_text_ratio",
            "selected_audio_ratio",
            "selected_vision_ratio",
            "entropy",
            "corr_availability_text_w_text",
            "corr_availability_audio_w_audio",
            "corr_availability_vision_w_vision",
            "is_router_collapsed",
            "dominant_anchor",
            "dominant_anchor_ratio",
            "seed_count",
        ])

    per_file = pd.DataFrame(rows)
    grouped = per_file.groupby(["dataset", "missing", "mode"], as_index=False).agg(
        mean_w_text=("mean_w_text", "mean"),
        mean_w_audio=("mean_w_audio", "mean"),
        mean_w_vision=("mean_w_vision", "mean"),
        selected_text_ratio=("selected_text_ratio", "mean"),
        selected_audio_ratio=("selected_audio_ratio", "mean"),
        selected_vision_ratio=("selected_vision_ratio", "mean"),
        entropy=("entropy", "mean"),
        corr_availability_text_w_text=("corr_availability_text_w_text", "mean"),
        corr_availability_audio_w_audio=("corr_availability_audio_w_audio", "mean"),
        corr_availability_vision_w_vision=("corr_availability_vision_w_vision", "mean"),
        seed_count=("seed", "count"),
    )
    dominant = grouped[["mean_w_text", "mean_w_audio", "mean_w_vision"]].to_numpy()
    idx = np.argmax(dominant, axis=1)
    anchor_names = np.array(["text", "audio", "vision"])
    grouped["dominant_anchor"] = anchor_names[idx]
    grouped["dominant_anchor_ratio"] = dominant[np.arange(len(grouped)), idx]
    grouped["is_router_collapsed"] = grouped["dominant_anchor_ratio"] > 0.85
    return grouped


def fmt_pm(mean_v, std_v):
    if pd.isna(mean_v):
        return "NaN"
    if pd.isna(std_v):
        return f"{mean_v:.6f}"
    return f"{mean_v:.6f} ± {std_v:.6f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append("NaN" if pd.isna(v) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def build_progress_table(summary_df: pd.DataFrame, run_records: List[Dict[str, str]], dataset: str,
                         phase: str, missing_list: List[float], modes: List[str],
                         out_root: str,
                         expected_seeds: List[int]) -> pd.DataFrame:
    # Build quick lookup from summary rows.
    ok = summary_df[summary_df["status"].isin(["success", "partial"])].copy() if not summary_df.empty else pd.DataFrame()
    key_to_seed_count = {}
    if not ok.empty:
        for (ds, miss, mode), g in ok.groupby(["dataset", "missing", "mode"]):
            key_to_seed_count[(str(ds), float(miss), str(mode))] = int(g["seed"].notna().sum())

    key_to_rec = {}
    for rec in run_records:
        key = (str(rec.get("dataset", "")), float(rec.get("missing", "nan")), str(rec.get("mode", "")))
        key_to_rec[key] = rec

    rows = []
    expected_n = len(expected_seeds)
    for miss in missing_list:
        for mode in modes:
            if mode not in SUPPORTED_MODES:
                continue
            key = (str(dataset), float(miss), str(mode))
            completed = int(key_to_seed_count.get(key, 0))
            rec = key_to_rec.get(key)
            log_path = rec["log_path"] if rec else os.path.join(
                out_root, "logs",
                f"{dataset}_missing{miss}_{mode}_{phase}.log"
            )

            if completed >= expected_n and expected_n > 0:
                status = "completed"
                need_rerun = 0
            else:
                if rec is None:
                    status = "not_started"
                    need_rerun = 1
                else:
                    rstatus = rec.get("status", "")
                    if rstatus in {"failed"}:
                        status = "failed"
                        need_rerun = 1
                    elif completed > 0:
                        status = "incomplete"
                        need_rerun = 1
                    else:
                        status = "not_started"
                        need_rerun = 1

            rows.append({
                "missing": float(miss),
                "mode": mode,
                "completed_seeds": completed,
                "expected_seeds": expected_n,
                "status": status,
                "need_rerun": need_rerun,
                "log_path": log_path,
            })
    return pd.DataFrame(rows).sort_values(["missing", "mode"]).reset_index(drop=True)


def collect_existing_records(out_root: str, dataset_name: str, phases: List[str], expected_seed_list: List[int]) -> List[Dict[str, str]]:
    records = []
    expected_n = len(expected_seed_list)
    mode_candidates = sorted(SUPPORTED_MODES, key=len, reverse=True)
    for ph in phases:
        phase_suffix = f"_{ph}.log"
        logs_glob = glob.glob(os.path.join(out_root, "logs", f"*{phase_suffix}"))
        for lp in sorted(set(logs_glob)):
            bn = os.path.basename(lp)
            if not bn.endswith(phase_suffix):
                continue
            stem = bn[: -len(phase_suffix)]  # remove _{phase}.log
            ds = None
            miss = None
            mode = None
            tag = ""
            protocol_tag = ""
            seeds = ""

            # new format:
            # {dataset}_m{missing}_{mode}_seed{...}_trainDLx_evalDLy_testDLz[_tag]
            new_re = re.match(
                r"^(?P<dataset>[A-Za-z0-9]+)_m(?P<missing>[^_]+)_(?P<mode>[A-Za-z0-9_]+)_seed(?P<seeds>[^_]+)_(?P<protocol>trainDL[01]_evalDL[01]_testDL[01])(?:_(?P<tag>.+))?$",
                stem,
            )
            if new_re:
                ds = new_re.group("dataset")
                try:
                    miss = float(new_re.group("missing"))
                except Exception:
                    miss = None
                mode = new_re.group("mode")
                seeds = new_re.group("seeds") or ""
                protocol_tag = new_re.group("protocol") or ""
                tag = new_re.group("tag") or ""
            else:
                # old format compatibility:
                # {dataset}_missing{missing}_{mode_and_tag}
                old_re = re.match(r"^(?P<dataset>[A-Za-z0-9]+)_missing(?P<missing>[^_]+)_(?P<mode_and_tag>.+)$", stem)
                if not old_re:
                    continue
                ds = old_re.group("dataset")
                try:
                    miss = float(old_re.group("missing"))
                except Exception:
                    miss = None
                mode_and_tag = old_re.group("mode_and_tag")
                for cand in mode_candidates:
                    if mode_and_tag == cand:
                        mode = cand
                        tag = ""
                        break
                    if mode_and_tag.startswith(cand + "_"):
                        mode = cand
                        tag = mode_and_tag[len(cand) + 1:]
                        break
                m_drop = re.match(r"eval(?P<e>[01])_test(?P<t>[01])$", tag)
                if m_drop:
                    protocol_tag = f"trainDL1_evalDL{m_drop.group('e')}_testDL{m_drop.group('t')}"

            if ds is None or miss is None or mode is None:
                continue
            if ds != dataset_name:
                continue
            parsed = parse_log(lp)
            status = "success" if parsed.completed and len(parsed.test_rows) >= expected_n else (
                "partial" if len(parsed.test_rows) > 0 else "failed"
            )
            train_drop_last = -1
            eval_drop_last = -1
            test_drop_last = -1
            m_protocol = re.match(r"trainDL(?P<t>[01])_evalDL(?P<e>[01])_testDL(?P<x>[01])$", protocol_tag)
            if m_protocol:
                train_drop_last = int(m_protocol.group("t"))
                eval_drop_last = int(m_protocol.group("e"))
                test_drop_last = int(m_protocol.group("x"))

            records.append({
                "dataset": ds,
                "missing": str(miss),
                "mode": mode,
                "phase": ph,
                "tag": tag,
                "protocol_tag": protocol_tag,
                "train_drop_last": str(train_drop_last),
                "eval_drop_last": str(eval_drop_last),
                "test_drop_last": str(test_drop_last),
                "seeds": seeds,
                "status": status,
                "reason": parsed.reason,
                "log_path": lp,
                "command": "",
            })
    return records


def choose_official_phase(summary_df: pd.DataFrame, expected_seed_list: List[int]) -> pd.DataFrame:
    # Official = completed multi-seed from full/core only; prefer full when both completed.
    if summary_df.empty:
        return pd.DataFrame(columns=["dataset", "missing", "mode", "phase", "completed_seeds"])
    expected_n = len(expected_seed_list)
    ok = summary_df[
        summary_df["phase"].isin(["core", "full"]) &
        summary_df["status"].isin(["success", "partial"]) &
        summary_df["seed"].notna()
    ].copy()
    if ok.empty:
        return pd.DataFrame(columns=["dataset", "missing", "mode", "phase", "completed_seeds"])

    grp = ok.groupby(["dataset", "missing", "mode", "phase"], as_index=False).agg(
        completed_seeds=("seed", "nunique")
    )
    grp = grp[grp["completed_seeds"] >= expected_n].copy()
    if grp.empty:
        return pd.DataFrame(columns=["dataset", "missing", "mode", "phase", "completed_seeds"])

    phase_rank = {"full": 2, "core": 1}
    grp["phase_rank"] = grp["phase"].map(phase_rank).fillna(0)
    grp = grp.sort_values(["dataset", "missing", "mode", "phase_rank"], ascending=[True, True, True, False])
    chosen = grp.drop_duplicates(subset=["dataset", "missing", "mode"], keep="first").copy()
    return chosen[["dataset", "missing", "mode", "phase", "completed_seeds"]]


def build_official_progress(summary_df: pd.DataFrame, run_records: List[Dict[str, str]], dataset: str,
                            missing_list: List[float], modes: List[str], expected_seed_list: List[int]) -> pd.DataFrame:
    expected_n = len(expected_seed_list)
    chosen = choose_official_phase(summary_df, expected_seed_list)
    chosen_map = {
        (str(r["dataset"]), float(r["missing"]), str(r["mode"])): str(r["phase"])
        for _, r in chosen.iterrows()
    }

    # latest record by key+phase for log path hints
    rec_map = {}
    for rec in run_records:
        key = (str(rec.get("dataset", "")), float(rec.get("missing", "nan")), str(rec.get("mode", "")), str(rec.get("phase", "")))
        rec_map[key] = rec

    # existing activity map (any phase core/full)
    activity = {}
    for rec in run_records:
        key = (str(rec.get("dataset", "")), float(rec.get("missing", "nan")), str(rec.get("mode", "")))
        activity.setdefault(key, []).append(rec)

    rows = []
    for miss in missing_list:
        for mode in modes:
            if mode not in SUPPORTED_MODES:
                continue
            key = (str(dataset), float(miss), str(mode))
            chosen_phase = chosen_map.get(key)
            if chosen_phase is not None:
                chosen_rec = rec_map.get((key[0], key[1], key[2], chosen_phase))
                log_path = chosen_rec["log_path"] if chosen_rec else ""
                rows.append({
                    "missing": float(miss),
                    "mode": mode,
                    "completed_seeds": expected_n,
                    "expected_seeds": expected_n,
                    "status": "completed",
                    "need_rerun": 0,
                    "source_phase": chosen_phase,
                    "log_path": log_path,
                })
                continue

            recs = activity.get(key, [])
            if len(recs) == 0:
                status = "not_started"
                log_path = ""
            else:
                # If any log exists but not complete, mark incomplete for rerun.
                status = "incomplete"
                log_path = recs[-1].get("log_path", "")
            rows.append({
                "missing": float(miss),
                "mode": mode,
                "completed_seeds": 0,
                "expected_seeds": expected_n,
                "status": status,
                "need_rerun": 1,
                "source_phase": "",
                "log_path": log_path,
            })
    return pd.DataFrame(rows).sort_values(["missing", "mode"]).reset_index(drop=True)


def build_official_summary(summary_df: pd.DataFrame, expected_seed_list: List[int]) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=summary_df.columns)
    chosen = choose_official_phase(summary_df, expected_seed_list)
    if chosen.empty:
        return pd.DataFrame(columns=summary_df.columns)
    chosen_key = chosen.rename(columns={"phase": "chosen_phase"})[["dataset", "missing", "mode", "chosen_phase"]]
    ok = summary_df[
        summary_df["phase"].isin(["core", "full"]) &
        summary_df["status"].isin(["success", "partial"]) &
        summary_df["seed"].notna()
    ].copy()
    merged = ok.merge(
        chosen_key,
        left_on=["dataset", "missing", "mode", "phase"],
        right_on=["dataset", "missing", "mode", "chosen_phase"],
        how="inner",
    )
    if "chosen_phase" in merged.columns:
        merged = merged.drop(columns=["chosen_phase"])
    return merged.reset_index(drop=True)


def build_paper_tables(agg_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if agg_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    use = agg_df.copy()
    metric_pairs = [
        ("MAE_mean", "MAE"),
        ("Corr_mean", "Corr"),
        ("Non0_acc_2_mean", "Non0_acc_2"),
        ("Non0_F1_score_mean", "Non0_F1_score"),
    ]

    rows = []
    for miss in sorted(use["missing"].dropna().unique()):
        sub = use[use["missing"] == miss]
        row = {"missing": float(miss)}
        for mode in SUPPORTED_MODES:
            mode_sub = sub[sub["mode"] == mode]
            for col_mean, short in metric_pairs:
                val = np.nan
                if len(mode_sub) > 0 and col_mean in mode_sub.columns:
                    val = float(mode_sub.iloc[0][col_mean])
                row[f"{mode}_{short}"] = val
        rows.append(row)
    table_missing = pd.DataFrame(rows).sort_values("missing").reset_index(drop=True)

    best_rows = []
    for miss in sorted(use["missing"].dropna().unique()):
        sub = use[use["missing"] == miss].copy()
        if sub.empty:
            continue
        r = {"missing": float(miss)}
        if sub["MAE_mean"].notna().any():
            i = sub["MAE_mean"].idxmin()
            r["best_MAE_method"] = sub.loc[i, "mode"]
            r["best_MAE"] = float(sub.loc[i, "MAE_mean"])
        else:
            r["best_MAE_method"], r["best_MAE"] = np.nan, np.nan
        if sub["Corr_mean"].notna().any():
            i = sub["Corr_mean"].idxmax()
            r["best_Corr_method"] = sub.loc[i, "mode"]
            r["best_Corr"] = float(sub.loc[i, "Corr_mean"])
        else:
            r["best_Corr_method"], r["best_Corr"] = np.nan, np.nan
        if sub["Non0_acc_2_mean"].notna().any():
            i = sub["Non0_acc_2_mean"].idxmax()
            r["best_Non0_acc_2_method"] = sub.loc[i, "mode"]
            r["best_Non0_acc_2"] = float(sub.loc[i, "Non0_acc_2_mean"])
        else:
            r["best_Non0_acc_2_method"], r["best_Non0_acc_2"] = np.nan, np.nan
        if sub["Non0_F1_score_mean"].notna().any():
            i = sub["Non0_F1_score_mean"].idxmax()
            r["best_Non0_F1_method"] = sub.loc[i, "mode"]
            r["best_Non0_F1"] = float(sub.loc[i, "Non0_F1_score_mean"])
        else:
            r["best_Non0_F1_method"], r["best_Non0_F1"] = np.nan, np.nan
        best_rows.append(r)
    table_best = pd.DataFrame(best_rows).sort_values("missing").reset_index(drop=True)
    return table_missing, table_best


def build_router_trend(router_df: pd.DataFrame) -> pd.DataFrame:
    if router_df.empty:
        return pd.DataFrame(columns=[
            "missing", "mode",
            "mean_w_text", "mean_w_audio", "mean_w_vision",
            "selected_text_ratio", "selected_audio_ratio", "selected_vision_ratio",
            "entropy", "is_router_collapsed", "dominant_anchor", "dominant_anchor_ratio",
            "corr_availability_text_w_text", "corr_availability_audio_w_audio",
            "corr_availability_vision_w_vision",
        ])
    keep_modes = {"dynamic_soft", "dynamic_soft_moe"}
    keep_missing = {0.3, 0.4, 0.5}
    use = router_df[
        router_df["mode"].isin(keep_modes) &
        router_df["missing"].isin(keep_missing)
    ].copy()
    cols = [
        "missing", "mode",
        "mean_w_text", "mean_w_audio", "mean_w_vision",
        "selected_text_ratio", "selected_audio_ratio", "selected_vision_ratio",
        "entropy", "is_router_collapsed", "dominant_anchor", "dominant_anchor_ratio",
        "corr_availability_text_w_text", "corr_availability_audio_w_audio",
        "corr_availability_vision_w_vision",
    ]
    for c in cols:
        if c not in use.columns:
            use[c] = np.nan
    return use[cols].sort_values(["missing", "mode"]).reset_index(drop=True)


def _reduce_metrics(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["missing", "mode", "MAE_mean", "Corr_mean", "Non0_acc_2_mean", "Non0_F1_score_mean", "Mult_acc_5_mean", "Mult_acc_7_mean"]
    out = df[keep].copy()
    return out.rename(columns={
        "MAE_mean": "MAE",
        "Corr_mean": "Corr",
        "Non0_acc_2_mean": "Non0_acc_2",
        "Non0_F1_score_mean": "Non0_F1_score",
        "Mult_acc_5_mean": "Mult_acc_5",
        "Mult_acc_7_mean": "Mult_acc_7",
    })


def build_taskrouter_comparisons(task_agg_df: pd.DataFrame, baseline_agg_path: str, out_root: str):
    def _select_official_rows(df: pd.DataFrame) -> pd.DataFrame:
        use = df.copy()
        needed = {"train_drop_last", "eval_drop_last", "test_drop_last"}
        if needed.issubset(set(use.columns)):
            off = use[
                (pd.to_numeric(use["train_drop_last"], errors="coerce") == 1)
                & (pd.to_numeric(use["eval_drop_last"], errors="coerce") == 0)
                & (pd.to_numeric(use["test_drop_last"], errors="coerce") == 0)
            ].copy()
            if not off.empty:
                use = off
        # Always keep only one row per (missing, mode) for stable comparisons.
        use = use.sort_values(["missing", "mode"]).drop_duplicates(
            subset=["missing", "mode"], keep="last"
        )
        return use.reset_index(drop=True)

    if task_agg_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not os.path.exists(baseline_agg_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    miss_keep = {0.3, 0.4, 0.5}
    task_sub = task_agg_df[
        (task_agg_df["mode"] == "dynamic_task_soft")
        & (task_agg_df["missing"].isin(miss_keep))
    ].copy()
    task_sub = _select_official_rows(task_sub)
    if task_sub.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    base_df = pd.read_csv(baseline_agg_path)
    base_sub = base_df[
        base_df["mode"].isin(["text", "audio", "vision", "dynamic_soft", "dynamic_soft_moe"])
        & base_df["missing"].isin(miss_keep)
    ].copy()
    base_sub = _select_official_rows(base_sub)

    merged = pd.concat([base_sub, task_sub], ignore_index=True, sort=False)
    table = _reduce_metrics(merged).sort_values(["missing", "mode"]).drop_duplicates(
        subset=["missing", "mode"], keep="last"
    ).reset_index(drop=True)
    table.to_csv(os.path.join(out_root, "taskrouter_vs_fulltest_baselines.csv"), index=False, float_format="%.6f")

    # delta vs dynamic_soft
    ds = table[table["mode"] == "dynamic_soft"].rename(columns={
        "MAE": "MAE_ds",
        "Corr": "Corr_ds",
        "Non0_acc_2": "Non0_acc_2_ds",
        "Non0_F1_score": "Non0_F1_score_ds",
        "Mult_acc_5": "Mult_acc_5_ds",
        "Mult_acc_7": "Mult_acc_7_ds",
    })
    dt = table[table["mode"] == "dynamic_task_soft"].copy()
    dvds = dt.merge(ds[["missing", "MAE_ds", "Corr_ds", "Non0_acc_2_ds", "Non0_F1_score_ds", "Mult_acc_5_ds", "Mult_acc_7_ds"]], on="missing", how="left")
    dvds["delta_MAE"] = dvds["MAE"] - dvds["MAE_ds"]
    dvds["delta_Corr"] = dvds["Corr"] - dvds["Corr_ds"]
    dvds["delta_Non0_acc_2"] = dvds["Non0_acc_2"] - dvds["Non0_acc_2_ds"]
    dvds["delta_Non0_F1_score"] = dvds["Non0_F1_score"] - dvds["Non0_F1_score_ds"]
    dvds["delta_Mult_acc_5"] = dvds["Mult_acc_5"] - dvds["Mult_acc_5_ds"]
    dvds["delta_Mult_acc_7"] = dvds["Mult_acc_7"] - dvds["Mult_acc_7_ds"]
    dvds.to_csv(os.path.join(out_root, "taskrouter_delta_vs_dynamic_soft.csv"), index=False, float_format="%.6f")

    # delta vs best fixed (selected by MAE among text/audio/vision)
    fixed = table[table["mode"].isin(["text", "audio", "vision"])].copy()
    best_rows = []
    for miss in sorted(fixed["missing"].unique()):
        s = fixed[fixed["missing"] == miss]
        if s.empty:
            continue
        idx = s["MAE"].idxmin()
        r = s.loc[idx]
        best_rows.append({
            "missing": float(miss),
            "best_fixed_mode": r["mode"],
            "MAE_fixed": float(r["MAE"]),
            "Corr_fixed": float(r["Corr"]),
            "Non0_acc_2_fixed": float(r["Non0_acc_2"]),
            "Non0_F1_score_fixed": float(r["Non0_F1_score"]),
            "Mult_acc_5_fixed": float(r["Mult_acc_5"]),
            "Mult_acc_7_fixed": float(r["Mult_acc_7"]),
        })
    best_fixed = pd.DataFrame(best_rows)
    dvbf = dt.merge(best_fixed, on="missing", how="left")
    dvbf["delta_MAE"] = dvbf["MAE"] - dvbf["MAE_fixed"]
    dvbf["delta_Corr"] = dvbf["Corr"] - dvbf["Corr_fixed"]
    dvbf["delta_Non0_acc_2"] = dvbf["Non0_acc_2"] - dvbf["Non0_acc_2_fixed"]
    dvbf["delta_Non0_F1_score"] = dvbf["Non0_F1_score"] - dvbf["Non0_F1_score_fixed"]
    dvbf["delta_Mult_acc_5"] = dvbf["Mult_acc_5"] - dvbf["Mult_acc_5_fixed"]
    dvbf["delta_Mult_acc_7"] = dvbf["Mult_acc_7"] - dvbf["Mult_acc_7_fixed"]
    dvbf.to_csv(os.path.join(out_root, "taskrouter_delta_vs_best_fixed.csv"), index=False, float_format="%.6f")
    return table, dvds, dvbf


def analyze_task_router_exports(task_router_dir: str, missing_filter: Optional[List[float]] = None) -> pd.DataFrame:
    if not task_router_dir:
        return pd.DataFrame()
    files = sorted(glob.glob(os.path.join(task_router_dir, "*.csv")))
    rows = []
    for p in files:
        name = os.path.basename(p)
        if "dynamic_task_soft" not in name:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        need_cols = {
            "w_text", "w_audio", "w_vision",
            "oracle_w_text", "oracle_w_audio", "oracle_w_vision",
            "router_selected_anchor", "oracle_selected_anchor", "router_oracle_match",
            "availability_text", "availability_audio", "availability_vision",
        }
        if not need_cols.issubset(df.columns):
            continue
        m = re.search(r"_missing([^_]+)_seed", name)
        try:
            miss = float(m.group(1)) if m else np.nan
        except Exception:
            miss = np.nan
        if missing_filter is not None and len(missing_filter) > 0:
            if np.isnan(miss) or miss not in [float(x) for x in missing_filter]:
                continue

        wt = pd.to_numeric(df["w_text"], errors="coerce")
        wa = pd.to_numeric(df["w_audio"], errors="coerce")
        wv = pd.to_numeric(df["w_vision"], errors="coerce")
        ot = pd.to_numeric(df["oracle_w_text"], errors="coerce")
        oa = pd.to_numeric(df["oracle_w_audio"], errors="coerce")
        ov = pd.to_numeric(df["oracle_w_vision"], errors="coerce")
        sel = df["router_selected_anchor"].astype(str).str.lower()
        osl = df["oracle_selected_anchor"].astype(str).str.lower()

        entr_r = -(wt * np.log(wt + 1e-8) + wa * np.log(wa + 1e-8) + wv * np.log(wv + 1e-8))
        entr_o = -(ot * np.log(ot + 1e-8) + oa * np.log(oa + 1e-8) + ov * np.log(ov + 1e-8))
        rows.append({
            "missing": miss,
            "mode": "dynamic_task_soft",
            "router_oracle_match_rate": float(pd.to_numeric(df["router_oracle_match"], errors="coerce").mean()),
            "mean_w_text": float(wt.mean()),
            "mean_w_audio": float(wa.mean()),
            "mean_w_vision": float(wv.mean()),
            "mean_oracle_w_text": float(ot.mean()),
            "mean_oracle_w_audio": float(oa.mean()),
            "mean_oracle_w_vision": float(ov.mean()),
            "selected_text_ratio": float((sel == "text").mean()),
            "selected_audio_ratio": float((sel == "audio").mean()),
            "selected_vision_ratio": float((sel == "vision").mean()),
            "oracle_text_ratio": float((osl == "text").mean()),
            "oracle_audio_ratio": float((osl == "audio").mean()),
            "oracle_vision_ratio": float((osl == "vision").mean()),
            "entropy_router": float(entr_r.mean()),
            "entropy_oracle": float(entr_o.mean()),
            "corr_router_oracle_text": safe_corr(wt, ot),
            "corr_router_oracle_audio": safe_corr(wa, oa),
            "corr_router_oracle_vision": safe_corr(wv, ov),
            "corr_availability_text_w_text": safe_corr(pd.to_numeric(df["availability_text"], errors="coerce"), wt),
            "corr_availability_audio_w_audio": safe_corr(pd.to_numeric(df["availability_audio"], errors="coerce"), wa),
            "corr_availability_vision_w_vision": safe_corr(pd.to_numeric(df["availability_vision"], errors="coerce"), wv),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).groupby(["missing", "mode"], as_index=False).mean(numeric_only=True)
    return out.sort_values(["missing"]).reset_index(drop=True)


def generate_taskrouter_report(out_root: str, compare_df: pd.DataFrame, delta_ds_df: pd.DataFrame,
                               delta_fixed_df: pd.DataFrame, diag_df: pd.DataFrame):
    p = os.path.join(out_root, "taskrouter_report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("# Task-aware Dynamic Anchor Router Report\n\n")
        f.write("## 1. Motivation\n")
        f.write("Missing-aware routing mainly captures modality availability, but availability does not always match task contribution. We add center-wise task supervision to improve router decisions.\n\n")
        f.write("## 2. Method\n")
        f.write("- Three auxiliary center predictors (text/audio/vision).\n")
        f.write("- Oracle center distribution from center-wise prediction error.\n")
        f.write("- Router supervised by KL/CE against oracle during training.\n")
        f.write("- Test-time routing does not use labels.\n\n")
        f.write("## 3. Results\n")
        f.write(markdown_table(compare_df) + "\n\n")
        f.write("## 4. Delta Analysis\n")
        f.write("### dynamic_task_soft vs dynamic_soft\n")
        f.write(markdown_table(delta_ds_df) + "\n\n")
        f.write("### dynamic_task_soft vs best fixed center\n")
        f.write(markdown_table(delta_fixed_df) + "\n\n")
        f.write("## 5. Router Diagnostics\n")
        f.write(markdown_table(diag_df) + "\n\n")
        f.write("## 6. Conclusion\n")
        if not delta_ds_df.empty:
            mae_better = int((delta_ds_df["delta_MAE"] < 0).sum())
            corr_better = int((delta_ds_df["delta_Corr"] > 0).sum())
            f.write(f"- dynamic_task_soft vs dynamic_soft: MAE better on {mae_better}/{len(delta_ds_df)}, Corr better on {corr_better}/{len(delta_ds_df)}.\n")
        if not delta_fixed_df.empty:
            fixed_mae_better = int((delta_fixed_df["delta_MAE"] < 0).sum())
            f.write(f"- dynamic_task_soft vs best fixed: MAE better on {fixed_mae_better}/{len(delta_fixed_df)}.\n")
        if not diag_df.empty and "router_oracle_match_rate" in diag_df.columns:
            f.write(f"- mean router-oracle match rate: {float(diag_df['router_oracle_match_rate'].mean()):.4f}.\n")
        f.write("- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.\n")


def generate_report(out_root: str, args, missing_str: str, modes_str: str,
                    summary_df: pd.DataFrame, agg_df: pd.DataFrame, delta_df: pd.DataFrame,
                    router_df: pd.DataFrame, seed_list: List[int]):
    report_path = os.path.join(out_root, "anchor_experiment_report.md")
    env_path = os.path.join(out_root, "env_check.txt")
    env_text = ""
    if os.path.exists(env_path):
        env_text = open(env_path, "r", encoding="utf-8", errors="ignore").read().strip()

    smoke_plan = [(0.0, "text"), (0.3, "audio"), (0.3, "vision"), (0.3, "dynamic_soft"), (0.3, "dynamic_soft_moe")]
    smoke_rows = []
    for miss, mode in smoke_plan:
        sub = summary_df[(summary_df["missing"] == miss) & (summary_df["mode"] == mode) & (summary_df["status"].isin(["success", "partial"]))]
        smoke_rows.append({"missing": miss, "mode": mode, "status": "PASS" if len(sub) > 0 else "FAIL"})
    smoke_df = pd.DataFrame(smoke_rows)

    overall_df = pd.DataFrame()
    if not agg_df.empty:
        t = agg_df.copy()
        for m in ["MAE", "Corr", "Non0_acc_2", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7"]:
            t[m] = [fmt_pm(a, b) for a, b in zip(t[f"{m}_mean"], t[f"{m}_std"])]
        overall_df = t[["phase", "missing", "mode", "MAE", "Corr", "Non0_acc_2", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7"]].sort_values(["missing", "mode"])

    def summarize_deltas(mode_name: str):
        if delta_df.empty:
            return "N/A"
        sub = delta_df[delta_df["mode"] == mode_name]
        if sub.empty:
            return "N/A"
        return (
            f"MAE better on {(sub['delta_MAE'] < 0).sum()}/{len(sub)}; "
            f"Corr better on {(sub['delta_Corr'] > 0).sum()}/{len(sub)}; "
            f"Non0_acc_2 better on {(sub['delta_Non0_acc_2'] > 0).sum()}/{len(sub)}; "
            f"Non0_F1_score better on {(sub['delta_Non0_F1_score'] > 0).sum()}/{len(sub)}."
        )

    moe_vs_soft = "N/A"
    if not agg_df.empty:
        l = agg_df[agg_df["mode"] == "dynamic_soft_moe"]
        r = agg_df[agg_df["mode"] == "dynamic_soft"]
        if len(l) and len(r):
            m = l.merge(r, on=["dataset", "phase", "missing"], suffixes=("_moe", "_soft"))
            if len(m):
                moe_vs_soft = (
                    f"MAE better {(m['MAE_mean_moe'] < m['MAE_mean_soft']).sum()}/{len(m)}; "
                    f"Corr better {(m['Corr_mean_moe'] > m['Corr_mean_soft']).sum()}/{len(m)}; "
                    f"Non0_acc_2 better {(m['Non0_acc_2_mean_moe'] > m['Non0_acc_2_mean_soft']).sum()}/{len(m)}; "
                    f"Non0_F1_score better {(m['Non0_F1_score_mean_moe'] > m['Non0_F1_score_mean_soft']).sum()}/{len(m)}."
                )

    collapse_note = "No router CSV found."
    if not router_df.empty:
        n = int((router_df["mean_w_text"] > 0.85).sum())
        collapse_note = f"router may collapse to text in {n} grouped settings." if n > 0 else "no obvious text collapse by mean_w_text>0.85."

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MOSI Dynamic Anchor Experiment Report\n\n")
        f.write("## 1. Experiment Settings\n\n")
        f.write(f"- phase: {args.phase}\n")
        f.write(f"- dataset: {args.datasetName}\n")
        f.write(f"- missing rates: {missing_str}\n")
        f.write(f"- modes: {modes_str}\n")
        f.write(f"- seeds in this run: {seed_list}\n")
        f.write(f"- quick_single_seed: {args.quick_single_seed}\n")
        f.write(f"- router_missing_bias: {args.router_missing_bias}\n")
        f.write(f"- train_drop_last: {args.train_drop_last}\n")
        f.write(f"- eval_drop_last: {args.eval_drop_last}\n")
        f.write(f"- test_drop_last: {args.test_drop_last}\n")
        f.write("- symmetric missing only\n")
        f.write("This experiment only uses symmetric missing rates via --missing. The asymmetric missing parameters --missing_t, --missing_a, and --missing_v are reserved but not used in this stage.\n\n")
        if args.phase == "full" and int(args.train_drop_last) == 1 and int(args.eval_drop_last) == 0 and int(args.test_drop_last) == 0:
            f.write("This official full-test protocol uses train_drop_last=1, eval_drop_last=0, and test_drop_last=0. The test set is evaluated on all 686 MOSI samples. Previous exploratory results under drop_last=True are not mixed with this official protocol.\n\n")
        if env_text:
            f.write("```text\n" + env_text + "\n```\n\n")

        f.write("## 2. Smoke Test Results\n\n")
        f.write(markdown_table(smoke_df) + "\n\n")

        f.write("## 3. Overall Results\n\n")
        f.write(markdown_table(overall_df) + "\n\n")

        f.write("## 4. Delta vs Fixed Text Baseline\n\n")
        f.write("- delta_MAE < 0 is better\n")
        f.write("- delta_Corr > 0 is better\n")
        f.write("- delta_Acc/F1 > 0 is better\n\n")
        f.write(markdown_table(delta_df.sort_values(["missing", "mode"]) if not delta_df.empty else delta_df) + "\n\n")

        f.write("## 5. Best Method by Missing Rate\n\n")
        if not agg_df.empty:
            for miss in sorted(agg_df["missing"].dropna().unique()):
                sub = agg_df[agg_df["missing"] == miss]
                f.write(f"missing={miss}:\n")
                if sub["MAE_mean"].notna().any():
                    f.write(f"- Best MAE: {sub.loc[sub['MAE_mean'].idxmin(), 'mode']}\n")
                if sub["Corr_mean"].notna().any():
                    f.write(f"- Best Corr: {sub.loc[sub['Corr_mean'].idxmax(), 'mode']}\n")
                if sub["Non0_acc_2_mean"].notna().any():
                    f.write(f"- Best Non0_acc_2: {sub.loc[sub['Non0_acc_2_mean'].idxmax(), 'mode']}\n")
                if sub["Non0_F1_score_mean"].notna().any():
                    f.write(f"- Best Non0_F1_score: {sub.loc[sub['Non0_F1_score_mean'].idxmax(), 'mode']}\n")
                f.write("\n")
        else:
            f.write("No complete aggregated results available.\n\n")

        f.write("## 6. Dynamic Anchor vs Fixed Text\n\n")
        f.write(f"- dynamic_soft vs text: {summarize_deltas('dynamic_soft')}\n")
        f.write(f"- dynamic_soft_moe vs text: {summarize_deltas('dynamic_soft_moe')}\n\n")

        f.write("## 7. Anchor-MoE Effect\n\n")
        f.write(f"- dynamic_soft_moe vs dynamic_soft: {moe_vs_soft}\n")
        f.write("- If MoE is not better, possible reasons include small MOSI size, overfitting risk, weak expert split, and limited center feature gap.\n\n")

        f.write("## 8. Router Behavior Analysis\n\n")
        f.write(markdown_table(router_df.sort_values(["missing", "mode"]) if not router_df.empty else router_df) + "\n\n")
        f.write(f"- Router note: {collapse_note}\n\n")

        f.write("## 9. Conclusion\n\n")
        f.write(f"- dynamic_soft effectiveness: {summarize_deltas('dynamic_soft')}\n")
        f.write(f"- Anchor-MoE effectiveness: {moe_vs_soft}\n")
        f.write("- Quick single-seed functional check and core/full multi-seed evaluation should be interpreted separately.\n")
        f.write("- Suggested next step: finish full symmetric sweep before asymmetric missing experiments.\n")


def generate_protocol_check_outputs(out_root: str, summary_df: pd.DataFrame, agg_df: pd.DataFrame):
    csv_path = os.path.join(out_root, "drop_last_protocol_check.csv")
    md_path = os.path.join(out_root, "drop_last_protocol_check.md")

    if agg_df.empty:
        empty = pd.DataFrame(columns=[
            "missing", "mode", "eval_drop_last", "test_drop_last", "effective_test_samples",
            "MAE", "Corr", "Non0_acc_2", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7",
        ])
        empty.to_csv(csv_path, index=False, float_format="%.6f")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Drop Last Protocol Check\n\nNo complete protocol_check results found.\n")
        return csv_path, md_path

    use = agg_df[agg_df["phase"] == "protocol_check"].copy()
    if use.empty:
        use = agg_df.copy()
    use = use[use["eval_drop_last"].isin([0, 1]) & use["test_drop_last"].isin([0, 1])].copy()

    out = use[[
        "missing", "mode", "eval_drop_last", "test_drop_last", "effective_test_samples",
        "MAE_mean", "Corr_mean", "Non0_acc_2_mean", "Non0_F1_score_mean", "Mult_acc_5_mean", "Mult_acc_7_mean",
    ]].rename(columns={
        "MAE_mean": "MAE",
        "Corr_mean": "Corr",
        "Non0_acc_2_mean": "Non0_acc_2",
        "Non0_F1_score_mean": "Non0_F1_score",
        "Mult_acc_5_mean": "Mult_acc_5",
        "Mult_acc_7_mean": "Mult_acc_7",
    })
    out = out.groupby(["missing", "mode", "eval_drop_last", "test_drop_last"], as_index=False).agg(
        effective_test_samples=("effective_test_samples", "max"),
        MAE=("MAE", "mean"),
        Corr=("Corr", "mean"),
        Non0_acc_2=("Non0_acc_2", "mean"),
        Non0_F1_score=("Non0_F1_score", "mean"),
        Mult_acc_5=("Mult_acc_5", "mean"),
        Mult_acc_7=("Mult_acc_7", "mean"),
    ).sort_values(["missing", "eval_drop_last", "test_drop_last"]).reset_index(drop=True)
    out.to_csv(csv_path, index=False, float_format="%.6f")

    notes = []
    by_missing = out.groupby(["missing"], as_index=False)
    for _, g in by_missing:
        miss = float(g["missing"].iloc[0])
        row_true = g[(g["eval_drop_last"] == 1) & (g["test_drop_last"] == 1)]
        row_false = g[(g["eval_drop_last"] == 0) & (g["test_drop_last"] == 0)]
        if len(row_true) and len(row_false):
            a = row_true.iloc[0]
            b = row_false.iloc[0]
            dropped = int(float(b["effective_test_samples"]) - float(a["effective_test_samples"])) \
                if pd.notna(a["effective_test_samples"]) and pd.notna(b["effective_test_samples"]) else -1
            notes.append({
                "missing": miss,
                "dropped_test_samples": dropped,
                "delta_MAE_B_minus_A": float(b["MAE"] - a["MAE"]),
                "delta_Corr_B_minus_A": float(b["Corr"] - a["Corr"]),
                "delta_Non0_acc_2_B_minus_A": float(b["Non0_acc_2"] - a["Non0_acc_2"]),
                "delta_Non0_F1_score_B_minus_A": float(b["Non0_F1_score"] - a["Non0_F1_score"]),
            })

    notes_df = pd.DataFrame(notes)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Drop Last Protocol Check\n\n")
        f.write("Protocol A: eval/test drop_last=True\n\n")
        f.write("Protocol B: eval/test drop_last=False\n\n")
        f.write("## Raw Results\n\n")
        f.write(markdown_table(out) + "\n\n")
        f.write("## A vs B Delta (B - A)\n\n")
        f.write(markdown_table(notes_df) + "\n\n")
        f.write("## Notes\n\n")
        f.write("- `drop_last=True` drops the final incomplete batch in eval/test.\n")
        f.write("- If MOSI test size is 686 and batch size is 24, `drop_last=True` uses 672 and drops 14 samples.\n")
        f.write("- Official protocol is recommended as `eval_drop_last=0` and `test_drop_last=0`.\n")
        f.write("- If metric drift is noticeable, prior results with dropped eval/test samples should be retested.\n")
    return csv_path, md_path


def upsert_experiment_manifest(out_root: str, summary_df: pd.DataFrame, summary_path: str, agg_path: str) -> str:
    manifest_path = os.path.join("results", "experiment_manifest.csv")
    branch, commit = get_git_meta()
    host = socket.gethostname()
    now_iso = datetime.utcnow().isoformat() + "Z"
    rows = []

    for _, r in summary_df.iterrows():
        log_path = str(r.get("log_path", ""))
        start_ts = parse_start_time(log_path)
        if start_ts is None and log_path and os.path.exists(log_path):
            start_ts = datetime.utcfromtimestamp(os.path.getmtime(log_path))
        created = (start_ts.isoformat() + "Z") if start_ts is not None else now_iso
        seed_val = r.get("seed", np.nan)
        if pd.isna(seed_val):
            seed_str = "NA"
        else:
            try:
                seed_str = str(int(float(seed_val)))
            except Exception:
                seed_str = str(seed_val)
        train_dl = int(pd.to_numeric(r.get("train_drop_last", -1), errors="coerce"))
        eval_dl = int(pd.to_numeric(r.get("eval_drop_last", -1), errors="coerce"))
        test_dl = int(pd.to_numeric(r.get("test_drop_last", -1), errors="coerce"))
        protocol_tag = str(r.get("protocol_tag", make_protocol_tag(train_dl, eval_dl, test_dl)))
        run_id = str(r.get("run_id", "")).strip()
        if not run_id:
            run_id = make_run_id(str(r.get("dataset", "")), float(r.get("missing", np.nan)), str(r.get("mode", "")),
                                 seed_str, train_dl, eval_dl, test_dl, created_time=start_ts)

        rows.append({
            "run_id": run_id,
            "created_time": created,
            "server_name": host,
            "git_branch": branch,
            "git_commit": commit,
            "dataset": r.get("dataset", np.nan),
            "missing": r.get("missing", np.nan),
            "mode": r.get("mode", np.nan),
            "fusion_center_mode": r.get("fusion_center_mode", np.nan),
            "seed": seed_str,
            "protocol_tag": protocol_tag,
            "train_drop_last": train_dl,
            "eval_drop_last": eval_dl,
            "test_drop_last": test_dl,
            "train_batch_size": r.get("train_batch_size", np.nan),
            "eval_batch_size": r.get("eval_batch_size", np.nan),
            "test_batch_size": r.get("test_batch_size", np.nan),
            "router_missing_bias": r.get("router_missing_bias", np.nan),
            "task_router_lambda": r.get("task_router_lambda", np.nan),
            "center_aux_lambda": r.get("center_aux_lambda", np.nan),
            "router_oracle_temperature": r.get("router_oracle_temperature", np.nan),
            "router_oracle_type": r.get("router_oracle_type", np.nan),
            "gate_balance_lambda": r.get("gate_balance_lambda", np.nan),
            "gate_target": r.get("gate_target", np.nan),
            "rta_gate_mode": r.get("rta_gate_mode", np.nan),
            "use_prediction_gate_supervision": r.get("use_prediction_gate_supervision", np.nan),
            "gate_oracle_temperature": r.get("gate_oracle_temperature", np.nan),
            "gate_task_lambda": r.get("gate_task_lambda", np.nan),
            "use_task_aware_router": r.get("use_task_aware_router", np.nan),
            "use_reliability_task_gate": r.get("use_reliability_task_gate", np.nan),
            "export_anchor_weights": r.get("export_anchor_weights", np.nan),
            "export_task_router_info": r.get("export_task_router_info", np.nan),
            "output_dir": r.get("output_dir", out_root),
            "log_path": log_path,
            "summary_csv": summary_path,
            "agg_csv": agg_path,
            "status": r.get("status", np.nan),
        })

    new_df = pd.DataFrame(rows)
    if os.path.exists(manifest_path):
        try:
            old_df = pd.read_csv(manifest_path)
        except Exception:
            old_df = pd.DataFrame()
        if not old_df.empty:
            new_df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
            # Normalize key dtypes before dedup.
            if "seed" in new_df.columns:
                new_df["seed"] = new_df["seed"].astype(str)
            if "dataset" in new_df.columns:
                new_df["dataset"] = new_df["dataset"].astype(str)
            if "mode" in new_df.columns:
                new_df["mode"] = new_df["mode"].astype(str)
            if "output_dir" in new_df.columns:
                new_df["output_dir"] = new_df["output_dir"].astype(str)
            if "missing" in new_df.columns:
                new_df["missing"] = pd.to_numeric(new_df["missing"], errors="coerce")
            new_df = new_df.drop_duplicates(subset=["run_id"], keep="last")
            # Also deduplicate by semantic experiment key to avoid stale protocol=-1 rows.
            if {"dataset", "missing", "mode", "seed", "output_dir"}.issubset(new_df.columns):
                def _valid_dl(r):
                    try:
                        t = int(float(r.get("train_drop_last", -1)))
                        e = int(float(r.get("eval_drop_last", -1)))
                        x = int(float(r.get("test_drop_last", -1)))
                    except Exception:
                        return 0
                    return int(t in [0, 1] and e in [0, 1] and x in [0, 1])
                new_df["_valid_dl"] = new_df.apply(_valid_dl, axis=1)
                new_df = new_df.sort_values(["_valid_dl", "created_time"], ascending=[True, True])
                new_df = new_df.drop_duplicates(subset=["dataset", "missing", "mode", "seed", "output_dir"], keep="last")
                new_df = new_df.drop(columns=["_valid_dl"])
    new_df = new_df.sort_values(["dataset", "missing", "mode", "seed"], na_position="last").reset_index(drop=True)
    new_df.to_csv(manifest_path, index=False, float_format="%.6f")
    return manifest_path


def build_result_source_report() -> str:
    report_path = os.path.join("results", "result_source_report.md")
    official_paths = [
        "results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv",
        "results/auto_anchor_runs_taskrouter_final/final_main_comparison.csv",
        "results/auto_anchor_runs_rta/anchor_experiment_agg.csv",
        "results/auto_anchor_runs_rta/rta_final_comparison.csv",
    ]
    paper_allowed = [
        "results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv",
        "results/auto_anchor_runs_taskrouter_final/final_main_comparison.csv",
        "results/auto_anchor_runs_rta/rta_final_comparison.csv",
    ]
    debug_only = ["results/results/normals/*.csv"]

    lines = ["# Result Source Report", ""]
    lines.append("## 1. Paper Table Eligible Sources")
    for p in paper_allowed:
        exists = os.path.exists(p)
        lines.append(f"- {'OK' if exists else 'MISSING'}: `{p}`")
    lines.append("")
    lines.append("## 2. Debug-Only Sources")
    lines.append("- `results/results/normals/*.csv`")
    lines.append("- WARNING: normals CSV does not contain sufficient experiment metadata and should not be used for paper tables.")
    lines.append("")

    lines.append("## 3. Official CSV Audit")
    hard_fail = False
    for p in official_paths:
        lines.append(f"### `{p}`")
        if not os.path.exists(p):
            lines.append("- **MISSING**")
            if p in paper_allowed:
                hard_fail = True
            lines.append("")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            lines.append(f"- **READ ERROR**: `{e}`")
            lines.append("")
            continue

        cols = list(df.columns)
        lines.append(f"- rows: `{len(df)}`")
        lines.append(f"- columns: `{cols}`")
        has_missing = "missing" in cols
        has_mode_like = ("mode" in cols) or ("method" in cols)
        # support mean columns in agg files
        has_mae = ("MAE" in cols) or ("MAE_mean" in cols)
        has_corr = ("Corr" in cols) or ("Corr_mean" in cols)
        has_acc = ("Non0_acc_2" in cols) or ("Non0_acc_2_mean" in cols)
        has_f1 = ("Non0_F1_score" in cols) or ("Non0_F1_score_mean" in cols)

        lines.append(f"- field check missing: {'OK' if has_missing else 'MISSING'}")
        lines.append(f"- field check mode/method: {'OK' if has_mode_like else 'MISSING'}")
        lines.append(f"- field check MAE: {'OK' if has_mae else 'MISSING'}")
        lines.append(f"- field check Corr: {'OK' if has_corr else 'MISSING'}")
        lines.append(f"- field check Non0_acc_2: {'OK' if has_acc else 'MISSING'}")
        lines.append(f"- field check Non0_F1_score: {'OK' if has_f1 else 'MISSING'}")
        if has_missing:
            try:
                miss_vals = sorted(pd.to_numeric(df["missing"], errors="coerce").dropna().unique().tolist())
                lines.append(f"- missing coverage: `{miss_vals}`")
                if len(miss_vals) == 0:
                    hard_fail = True
            except Exception:
                lines.append("- missing coverage: `N/A`")
                hard_fail = True
        if "mode" in cols:
            lines.append(f"- mode coverage: `{sorted(df['mode'].dropna().astype(str).unique().tolist())}`")
        elif "method" in cols:
            lines.append(f"- method coverage: `{sorted(df['method'].dropna().astype(str).unique().tolist())}`")

        if not (has_missing and has_mode_like and has_mae and has_corr and has_acc and has_f1):
            hard_fail = True
            lines.append("- **BLOCKER**: metadata/metric fields are insufficient for safe paper-table generation.")
        lines.append("")

    lines.append("## 4. Duplicate / Protocol Mixing Risk")
    fp = "results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv"
    if os.path.exists(fp):
        try:
            fa = pd.read_csv(fp)
            if {"missing", "mode", "train_drop_last", "eval_drop_last", "test_drop_last"}.issubset(fa.columns):
                mix = fa.groupby(["missing", "mode"], as_index=False).agg(
                    protocol_count=("train_drop_last", lambda s: len(set(zip(
                        s.tolist(),
                        fa.loc[s.index, "eval_drop_last"].tolist(),
                        fa.loc[s.index, "test_drop_last"].tolist(),
                    ))))
                )
                mixed = mix[mix["protocol_count"] > 1]
                if mixed.empty:
                    lines.append("- No protocol mixing detected in fulltest agg by `(missing, mode)`.")
                else:
                    lines.append("- **Risk**: mixed protocols detected:")
                    lines.append("")
                    lines.append(mixed.to_markdown(index=False))
            else:
                lines.append("- Protocol columns not present in fulltest agg; cannot verify mixing.")
        except Exception as e:
            lines.append(f"- Unable to check protocol mixing: `{e}`")
    else:
        lines.append("- fulltest agg missing; cannot check protocol mixing.")
    lines.append("")

    lines.append("## 5. Recommendation")
    if hard_fail:
        lines.append("- **Do not generate final paper LaTeX tables yet.** Resolve missing official fields/files first.")
    else:
        lines.append("- Safe paper sources (priority):")
        lines.append("  1. `results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv`")
        lines.append("  2. `results/auto_anchor_runs_taskrouter_final/final_main_comparison.csv`")
        lines.append("  3. `results/auto_anchor_runs_rta/rta_final_comparison.csv`")
        lines.append("- Do not use `results/results/normals/*.csv` except debugging.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    args = parse_args()
    out_root = args.output_dir
    ensure_dirs(out_root)

    missing_str, modes_str, missing_list, modes = resolve_lists(args)
    seed_list = expected_seed_list(args)
    run_help_text = detect_run_help_flags(args.python_bin)

    if args.phase == "quick":
        # Quick phase follows fixed 5-case smoke plan:
        # text@0.0, audio/vision/dynamic_soft/dynamic_soft_moe@0.3
        quick_plan = [
            (0.0, "text"),
            (0.3, "audio"),
            (0.3, "vision"),
            (0.3, "dynamic_soft"),
            (0.3, "dynamic_soft_moe"),
        ]
        run_configs = [
            make_run_config(
                args.datasetName, miss, mode, args.export_anchor_weights, args.router_missing_bias,
                args.train_drop_last, args.eval_drop_last, args.test_drop_last,
                export_task_router_info=args.export_task_router_info,
                task_router_lambda=args.task_router_lambda,
                center_aux_lambda=args.center_aux_lambda,
                router_oracle_temperature=args.router_oracle_temperature,
                router_oracle_type=args.router_oracle_type,
                task_router_output_dir=(args.task_router_output_dir or os.path.join(out_root, "task_router_analysis")),
                rta_gate_mode=args.rta_gate_mode,
                use_prediction_gate_supervision=args.use_prediction_gate_supervision,
                gate_oracle_temperature=args.gate_oracle_temperature,
                gate_task_lambda=args.gate_task_lambda,
                rta_pred_residual=args.rta_pred_residual,
                gate_supervision_mode=args.gate_supervision_mode,
                gate_margin=args.gate_margin,
            )
            for (miss, mode) in quick_plan
            if (miss in missing_list) and (mode in modes)
        ]
    elif args.phase == "protocol_check":
        # Protocol check compares eval/test drop_last=True vs False using fixed text only.
        protocol_pairs = [(1, 1, "eval1_test1"), (0, 0, "eval0_test0")]
        run_configs = []
        for miss in missing_list:
            for eval_drop, test_drop, tag in protocol_pairs:
                run_configs.append(
                    make_run_config(
                        args.datasetName, miss, "text", args.export_anchor_weights, args.router_missing_bias,
                        args.train_drop_last, eval_drop, test_drop, tag=tag,
                        export_task_router_info=args.export_task_router_info,
                        task_router_lambda=args.task_router_lambda,
                        center_aux_lambda=args.center_aux_lambda,
                        router_oracle_temperature=args.router_oracle_temperature,
                        router_oracle_type=args.router_oracle_type,
                        task_router_output_dir=(args.task_router_output_dir or os.path.join(out_root, "task_router_analysis")),
                        rta_gate_mode=args.rta_gate_mode,
                        use_prediction_gate_supervision=args.use_prediction_gate_supervision,
                        gate_oracle_temperature=args.gate_oracle_temperature,
                        gate_task_lambda=args.gate_task_lambda,
                        rta_pred_residual=args.rta_pred_residual,
                        gate_supervision_mode=args.gate_supervision_mode,
                        gate_margin=args.gate_margin,
                    )
                )
    else:
        run_configs = [
            make_run_config(
                args.datasetName, miss, mode, args.export_anchor_weights, args.router_missing_bias,
                args.train_drop_last, args.eval_drop_last, args.test_drop_last,
                export_task_router_info=args.export_task_router_info,
                task_router_lambda=args.task_router_lambda,
                center_aux_lambda=args.center_aux_lambda,
                router_oracle_temperature=args.router_oracle_temperature,
                router_oracle_type=args.router_oracle_type,
                task_router_output_dir=(args.task_router_output_dir or os.path.join(out_root, "task_router_analysis")),
                rta_gate_mode=args.rta_gate_mode,
                use_prediction_gate_supervision=args.use_prediction_gate_supervision,
                gate_oracle_temperature=args.gate_oracle_temperature,
                gate_task_lambda=args.gate_task_lambda,
                rta_pred_residual=args.rta_pred_residual,
                gate_supervision_mode=args.gate_supervision_mode,
                gate_margin=args.gate_margin,
            )
            for miss in missing_list
            for mode in modes
        ]

    run_records = []
    if int(args.analyze_only) != 1:
        for cfg in run_configs:
            print(f"[RUN] phase={args.phase} dataset={cfg.dataset} missing={cfg.missing} mode={cfg.mode}")
            rec = run_single(cfg, args, out_root, seed_list, run_help_text)
            run_records.append(rec)
            print(f"  -> status={rec['status']} reason={rec['reason']}")

    # Official results aggregate core + full only, excluding quick smoke/single-seed runs.
    if args.phase == "quick":
        phases_for_collect = ["quick"]
    elif args.phase == "protocol_check":
        phases_for_collect = ["protocol_check"]
    elif args.phase == "core":
        phases_for_collect = ["core"]
    else:
        phases_for_collect = ["core", "full"]
    existing_records = collect_existing_records(
        out_root=out_root,
        dataset_name=args.datasetName,
        phases=phases_for_collect,
        expected_seed_list=seed_list,
    )

    # Prefer current invocation record over existing snapshot if duplicated.
    merged_records = {}
    for rec in existing_records + run_records:
        key = (
            rec["dataset"], rec["missing"], rec["mode"], rec["phase"],
            rec.get("train_drop_last", ""), rec.get("eval_drop_last", ""), rec.get("test_drop_last", ""),
            rec.get("tag", ""),
        )
        merged_records[key] = rec
    run_records = list(merged_records.values())

    raw_summary_df, failed_df = build_summary_rows(run_records, args, seed_list)
    if args.phase == "full":
        summary_df = build_official_summary(raw_summary_df, seed_list)
    else:
        summary_df = raw_summary_df

    required_summary_cols = [
        "run_id", "protocol_tag", "fusion_center_mode",
        "task_router_lambda", "center_aux_lambda", "router_oracle_temperature", "router_oracle_type",
        "router_missing_bias", "gate_balance_lambda", "gate_target",
        "rta_gate_mode", "use_prediction_gate_supervision", "gate_oracle_temperature", "gate_task_lambda",
        "use_task_aware_router", "use_reliability_task_gate",
        "export_anchor_weights", "export_task_router_info",
        "train_batch_size", "eval_batch_size", "test_batch_size",
        "output_dir",
    ]
    for c in required_summary_cols:
        if c not in summary_df.columns:
            summary_df[c] = np.nan
    agg_df = build_agg(summary_df)
    delta_df = build_delta_vs_text(agg_df)
    router_df = analyze_anchor_router(output_dir=out_root, missing_filter=missing_list)
    if args.phase == "full":
        progress_df = build_official_progress(
            summary_df=raw_summary_df,
            run_records=run_records,
            dataset=args.datasetName,
            missing_list=missing_list,
            modes=modes,
            expected_seed_list=seed_list,
        )
    elif args.phase == "protocol_check":
        progress_rows = []
        for rec in run_records:
            if rec.get("phase") != "protocol_check":
                continue
            parsed = parse_log(rec["log_path"])
            progress_rows.append({
                "missing": float(rec["missing"]),
                "mode": rec["mode"],
                "eval_drop_last": int(float(rec.get("eval_drop_last", -1))),
                "test_drop_last": int(float(rec.get("test_drop_last", -1))),
                "completed_seeds": len(parsed.test_rows),
                "expected_seeds": len(seed_list),
                "status": rec["status"],
                "need_rerun": 0 if rec["status"] in ["success", "skipped_completed"] else 1,
                "log_path": rec["log_path"],
            })
        progress_df = pd.DataFrame(progress_rows).sort_values(
            ["missing", "mode", "eval_drop_last", "test_drop_last"]
        ).reset_index(drop=True)
    else:
        progress_df = build_progress_table(
            summary_df=summary_df,
            run_records=run_records,
            dataset=args.datasetName,
            phase=args.phase,
            missing_list=missing_list,
            modes=modes,
            out_root=out_root,
            expected_seeds=seed_list,
        )
    paper_sweep_df, paper_best_df = build_paper_tables(agg_df)
    router_trend_df = build_router_trend(router_df)

    if args.phase == "protocol_check":
        protocol_summary_path = os.path.join(out_root, "drop_last_protocol_check_summary.csv")
        protocol_agg_path = os.path.join(out_root, "drop_last_protocol_check_agg.csv")
        protocol_progress_path = os.path.join(out_root, "drop_last_protocol_check_progress.csv")
        protocol_failed_path = os.path.join(out_root, "drop_last_protocol_check_failed_runs.csv")
        summary_df.to_csv(protocol_summary_path, index=False, float_format="%.6f")
        agg_df.to_csv(protocol_agg_path, index=False, float_format="%.6f")
        progress_df.to_csv(protocol_progress_path, index=False)
        failed_df.to_csv(protocol_failed_path, index=False)
        manifest_path = upsert_experiment_manifest(out_root, summary_df, protocol_summary_path, protocol_agg_path)
        source_report_path = build_result_source_report()
        protocol_csv_path, protocol_md_path = generate_protocol_check_outputs(out_root, summary_df, agg_df)
        print("Saved:")
        print("-", protocol_summary_path)
        print("-", protocol_agg_path)
        print("-", protocol_progress_path)
        print("-", protocol_failed_path)
        print("-", protocol_csv_path)
        print("-", protocol_md_path)
        print("-", manifest_path)
        print("-", source_report_path)
        return

    summary_path = os.path.join(out_root, "anchor_experiment_summary.csv")
    agg_path = os.path.join(out_root, "anchor_experiment_agg.csv")
    delta_path = os.path.join(out_root, "anchor_experiment_delta_vs_text.csv")
    router_path = os.path.join(out_root, "anchor_router_analysis.csv")
    router_trend_path = os.path.join(out_root, "router_trend_0345.csv")
    failed_path = os.path.join(out_root, "failed_runs.csv")
    phase_progress_path = os.path.join(out_root, f"{args.phase}_progress.csv")
    full_progress_path = os.path.join(out_root, "full_progress.csv")
    official_progress_path = os.path.join(out_root, "official_progress.csv")
    paper_sweep_path = os.path.join(out_root, "paper_table_missing_sweep.csv")
    paper_best_path = os.path.join(out_root, "paper_table_best_by_missing.csv")

    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    if agg_df.empty:
        agg_df = pd.DataFrame(columns=[
            "dataset", "phase", "missing", "mode", "protocol_tag",
            "train_drop_last", "eval_drop_last", "test_drop_last",
            "task_router_lambda", "center_aux_lambda", "router_oracle_temperature", "router_oracle_type",
            "gate_balance_lambda", "gate_target",
            "rta_gate_mode", "use_prediction_gate_supervision", "gate_oracle_temperature", "gate_task_lambda",
            "MAE_mean", "MAE_std",
            "Corr_mean", "Corr_std",
            "Non0_acc_2_mean", "Non0_acc_2_std",
            "Non0_F1_score_mean", "Non0_F1_score_std",
            "Mult_acc_5_mean", "Mult_acc_5_std",
            "Mult_acc_7_mean", "Mult_acc_7_std",
            "effective_test_samples",
        ])
    agg_df.to_csv(agg_path, index=False, float_format="%.6f")
    if delta_df.empty:
        delta_df = pd.DataFrame(columns=[
            "dataset", "phase", "missing", "mode", "protocol_tag", "train_drop_last", "eval_drop_last", "test_drop_last",
            "MAE_mean", "MAE_text", "delta_MAE",
            "Corr_mean", "Corr_text", "delta_Corr",
            "Non0_acc_2_mean", "Non0_acc_2_text", "delta_Non0_acc_2",
            "Non0_F1_score_mean", "Non0_F1_score_text", "delta_Non0_F1_score",
            "Mult_acc_5_mean", "Mult_acc_5_text", "delta_Mult_acc_5",
            "Mult_acc_7_mean", "Mult_acc_7_text", "delta_Mult_acc_7",
        ])
    delta_df.to_csv(delta_path, index=False, float_format="%.6f")
    router_df.to_csv(router_path, index=False, float_format="%.6f")
    router_trend_df.to_csv(router_trend_path, index=False, float_format="%.6f")
    progress_df.to_csv(phase_progress_path, index=False)
    if args.phase == "full":
        progress_df.to_csv(full_progress_path, index=False)
        progress_df.to_csv(official_progress_path, index=False)
    if paper_sweep_df.empty:
        paper_sweep_df = pd.DataFrame(columns=[
            "missing",
            "text_MAE", "audio_MAE", "vision_MAE", "dynamic_soft_MAE", "dynamic_soft_moe_MAE",
            "text_Corr", "audio_Corr", "vision_Corr", "dynamic_soft_Corr", "dynamic_soft_moe_Corr",
            "text_Non0_acc_2", "audio_Non0_acc_2", "vision_Non0_acc_2", "dynamic_soft_Non0_acc_2", "dynamic_soft_moe_Non0_acc_2",
            "text_Non0_F1_score", "audio_Non0_F1_score", "vision_Non0_F1_score", "dynamic_soft_Non0_F1_score", "dynamic_soft_moe_Non0_F1_score",
        ])
    paper_sweep_df.to_csv(paper_sweep_path, index=False, float_format="%.6f")
    if paper_best_df.empty:
        paper_best_df = pd.DataFrame(columns=[
            "missing",
            "best_MAE_method", "best_MAE",
            "best_Corr_method", "best_Corr",
            "best_Non0_acc_2_method", "best_Non0_acc_2",
            "best_Non0_F1_method", "best_Non0_F1",
        ])
    paper_best_df.to_csv(paper_best_path, index=False, float_format="%.6f")
    if failed_df.empty:
        failed_df = pd.DataFrame(columns=[
            "dataset", "missing", "mode", "phase", "status", "reason", "log_path", "command"
        ])
    failed_df.to_csv(failed_path, index=False)

    generate_report(out_root, args, missing_str, modes_str, summary_df, agg_df, delta_df, router_df, seed_list)

    # Task-aware router specific analysis (optional, generated when dynamic_task_soft exists).
    taskrouter_compare = pd.DataFrame()
    taskrouter_delta_ds = pd.DataFrame()
    taskrouter_delta_fixed = pd.DataFrame()
    taskrouter_diag = pd.DataFrame()
    if (not agg_df.empty) and ("dynamic_task_soft" in set(agg_df["mode"].astype(str).tolist())):
        baseline_agg_path = os.path.join("results", "auto_anchor_runs_fulltest", "anchor_experiment_agg.csv")
        taskrouter_compare, taskrouter_delta_ds, taskrouter_delta_fixed = build_taskrouter_comparisons(
            task_agg_df=agg_df,
            baseline_agg_path=baseline_agg_path,
            out_root=out_root,
        )
        task_router_dir = args.task_router_output_dir or os.path.join(out_root, "task_router_analysis")
        taskrouter_diag = analyze_task_router_exports(task_router_dir, missing_filter=missing_list)
        taskrouter_diag_path = os.path.join(out_root, "task_router_diagnostics.csv")
        taskrouter_diag.to_csv(taskrouter_diag_path, index=False, float_format="%.6f")
        generate_taskrouter_report(out_root, taskrouter_compare, taskrouter_delta_ds, taskrouter_delta_fixed, taskrouter_diag)

    protocol_csv_path = None
    protocol_md_path = None
    if args.phase == "protocol_check":
        protocol_csv_path, protocol_md_path = generate_protocol_check_outputs(out_root, summary_df, agg_df)

    print("Saved:")
    print("-", summary_path)
    print("-", agg_path)
    print("-", delta_path)
    print("-", router_path)
    print("-", router_trend_path)
    print("-", phase_progress_path)
    if args.phase == "full":
        print("-", full_progress_path)
        print("-", official_progress_path)
    print("-", paper_sweep_path)
    print("-", paper_best_path)
    print("-", failed_path)
    print("-", os.path.join(out_root, "anchor_experiment_report.md"))
    if not taskrouter_compare.empty:
        print("-", os.path.join(out_root, "taskrouter_vs_fulltest_baselines.csv"))
    if not taskrouter_delta_ds.empty:
        print("-", os.path.join(out_root, "taskrouter_delta_vs_dynamic_soft.csv"))
    if not taskrouter_delta_fixed.empty:
        print("-", os.path.join(out_root, "taskrouter_delta_vs_best_fixed.csv"))
    if not taskrouter_diag.empty:
        print("-", os.path.join(out_root, "task_router_diagnostics.csv"))
        print("-", os.path.join(out_root, "taskrouter_report.md"))
    if protocol_csv_path and protocol_md_path:
        print("-", protocol_csv_path)
        print("-", protocol_md_path)

    manifest_path = upsert_experiment_manifest(out_root, summary_df, summary_path, agg_path)
    source_report_path = build_result_source_report()
    print("-", manifest_path)
    print("-", source_report_path)


if __name__ == "__main__":
    main()
