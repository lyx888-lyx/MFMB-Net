#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import pandas as pd

METRICS = ["MAE", "Corr", "Non0_acc_2", "Non0_F1_score", "Mult_acc_5", "Mult_acc_7"]


def run_cmd(cmd):
    print('[RUN]', ' '.join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def build_report(out_dir, summary_df):
    path = os.path.join(out_dir, 'gate_ablation_report.md')
    lines = ['# Dynamic RTA Gate Ablation Report', '']
    if summary_df.empty:
        lines.append('No valid rows found.')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return path

    lines.append('## Summary')
    lines.append('')
    lines.append(summary_df.to_csv(index=False))
    lines.append('')

    # reference values from existing official files
    ds_ref = {}
    ta_ref = {}
    try:
        base = pd.read_csv('results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv')
        base = base[(base['mode'] == 'dynamic_soft') & (base['missing'].isin([0.4, 0.5]))]
        for _, r in base.iterrows():
            ds_ref[float(r['missing'])] = float(r['MAE_mean'])
    except Exception:
        pass
    try:
        ta = pd.read_csv('results/auto_anchor_runs_taskrouter_final/tuned_dynamic_task_soft_0.1_0.5.csv')
        ta = ta[ta['missing'].isin([0.4, 0.5])]
        for _, r in ta.iterrows():
            ta_ref[float(r['missing'])] = float(r['MAE'])
    except Exception:
        pass

    lines.append('## Key Findings')
    for miss in sorted(summary_df['missing'].unique()):
        blk = summary_df[summary_df['missing'] == miss].set_index('rta_gate_mode')
        lines.append(f'- missing={miss:.1f}:')
        if 'force_rel' in blk.index and miss in ds_ref:
            d = float(blk.loc['force_rel', 'MAE']) - ds_ref[miss]
            lines.append(f'  - force_rel vs dynamic_soft MAE delta: {d:+.4f}')
        if 'force_task' in blk.index and miss in ta_ref:
            d = float(blk.loc['force_task', 'MAE']) - ta_ref[miss]
            lines.append(f'  - force_task vs dynamic_task_soft_tuned MAE delta: {d:+.4f}')
        if 'fixed_half' in blk.index and 'learned' in blk.index:
            d = float(blk.loc['learned', 'MAE']) - float(blk.loc['fixed_half', 'MAE'])
            lines.append(f'  - learned MAE - fixed_half MAE: {d:+.4f} (negative means learned better)')

    lines.append('')
    lines.append('## Interpretation')
    lines.append('- If force_rel is far from dynamic_soft, H_rel path may be non-equivalent.')
    lines.append('- If force_task is far from dynamic_task_soft_tuned, H_task path may be non-equivalent.')
    lines.append('- If force_rel/force_task are reasonable but learned is bad, gate learning is likely unstable.')
    lines.append('- If fixed_half beats learned consistently, learned gate may be overfitting/noisy.')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--python_bin', default=sys.executable)
    ap.add_argument('--output_dir', default='results/auto_anchor_runs_rta_gate_ablation')
    ap.add_argument('--dataset', default='mosi')
    ap.add_argument('--missing_list', default='0.4,0.5')
    ap.add_argument('--seeds', default='111')
    ap.add_argument('--gpu_ids', default='0')
    ap.add_argument('--resume', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    modes = ['force_rel', 'force_task', 'fixed_half', 'learned']
    mode_dirs = {}

    for gm in modes:
        mode_out = os.path.join(args.output_dir, f'mode_{gm}')
        mode_dirs[gm] = mode_out
        cmd = [
            args.python_bin, 'scripts/auto_run_anchor_experiments.py',
            '--datasetName', args.dataset,
            '--phase', 'full',
            '--missing_list', args.missing_list,
            '--modes', 'dynamic_rta',
            '--export_anchor_weights', '1',
            '--export_task_router_info', '1',
            '--router_missing_bias', '2.0',
            '--task_router_lambda', '0.1',
            '--center_aux_lambda', '0.05',
            '--router_oracle_temperature', '0.8',
            '--router_oracle_type', 'soft',
            '--use_reliability_task_gate', '1',
            '--rta_gate_mode', gm,
            '--gate_balance_lambda', '0.01',
            '--gate_target', '0.5',
            '--gate_hidden_dim', '32',
            '--gate_dropout', '0.1',
            '--seeds', args.seeds,
            '--gpu_ids', str(args.gpu_ids),
            '--resume', str(args.resume),
            '--train_drop_last', '1',
            '--eval_drop_last', '0',
            '--test_drop_last', '0',
            '--output_dir', mode_out,
        ]
        run_cmd(cmd)

    rows = []
    for gm, mode_out in mode_dirs.items():
        agg_path = os.path.join(mode_out, 'anchor_experiment_agg.csv')
        if not os.path.exists(agg_path):
            continue
        agg = pd.read_csv(agg_path)
        agg = agg[(agg['mode'] == 'dynamic_rta') & (agg['missing'].isin([0.4, 0.5]))].copy()
        if agg.empty:
            continue
        diag_path = os.path.join(mode_out, 'rta_gate_diagnostics.csv')
        diag = pd.read_csv(diag_path) if os.path.exists(diag_path) else pd.DataFrame(columns=['missing','mean_g_task','mean_g_rel'])
        prog_path = os.path.join(mode_out, 'official_progress.csv')
        prog = pd.read_csv(prog_path) if os.path.exists(prog_path) else pd.DataFrame(columns=['missing','mode','log_path'])

        for _, r in agg.iterrows():
            miss = float(r['missing'])
            drow = diag[diag['missing'] == miss]
            prow = prog[(prog['missing'] == miss) & (prog['mode'] == 'dynamic_rta')]
            rows.append({
                'missing': miss,
                'rta_gate_mode': gm,
                'seed': int(args.seeds.split(',')[0]),
                'MAE': float(r['MAE_mean']),
                'Corr': float(r['Corr_mean']),
                'Non0_acc_2': float(r['Non0_acc_2_mean']),
                'Non0_F1_score': float(r['Non0_F1_score_mean']),
                'Mult_acc_5': float(r['Mult_acc_5_mean']),
                'Mult_acc_7': float(r['Mult_acc_7_mean']),
                'mean_g_task': float(drow.iloc[0]['mean_g_task']) if len(drow) else float('nan'),
                'mean_g_rel': float(drow.iloc[0]['mean_g_rel']) if len(drow) else float('nan'),
                'log_path': str(prow.iloc[0]['log_path']) if len(prow) else '',
            })

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit('No valid ablation rows collected.')

    summary = summary.sort_values(['missing', 'rta_gate_mode']).reset_index(drop=True)
    out_csv = os.path.join(args.output_dir, 'gate_ablation_summary.csv')
    summary.to_csv(out_csv, index=False, float_format='%.6f')
    rpt = build_report(args.output_dir, summary)
    print('Saved:', out_csv)
    print('Saved:', rpt)


if __name__ == '__main__':
    main()
