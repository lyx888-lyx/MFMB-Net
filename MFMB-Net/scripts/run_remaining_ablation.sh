#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/usr/miniconda3/envs/mfmb/bin/python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MFMB_DATA_ROOT="${MFMB_DATA_ROOT:-/sharefile/lyx_model/MMSA/Datasets}"

run_exp() {
  local tag="$1"
  shift
  for miss in 0.0 0.3 0.5; do
    echo "===== ${tag} missing=${miss} ====="
    "$PYTHON" run.py --missing "$miss" --exp_tag "$tag" "$@" 2>&1 | tee -a "results/logs/${tag}_m${miss}.log"
  done
}

run_exp mide_split_aup \
  --mide_enable --mide_variant split_aup \
  --batch_size_override 32 --lr_other_override 0.0025 --lr_bert_override 1e-5 --amp \
  --mide_tau_uni 1.0 --mide_tau_loo 0.6 --mide_tau_p 0.5 \
  --mide_beta_uni 0.4 --mide_beta_loo 0.6 \
  --mide_margin 0.05 --mide_pos_threshold 0.03 --mide_neg_threshold 0.01 \
  --mide_d_floor_min 0.65 \
  --mide_aux_start_epoch 2 --mide_gate_start_epoch 2 --mide_full_start_epoch 4 \
  --mide_lambda_uni 0.08 --mide_lambda_util_bce 0.08 --mide_lambda_rank 0.03 \
  --mide_lambda_poll_bce 0.03 --mide_lambda_noinfo 0.02 --mide_lambda_poll 0.03 \
  --mide_lambda_sparse 0.02 --mide_lambda_sep 0.01

run_exp mide_split_plus \
  --mide_enable --mide_variant split_aup_plus \
  --batch_size_override 40 --lr_other_override 0.0028 --lr_bert_override 1e-5 --amp \
  --mide_tau_uni 1.0 --mide_tau_loo 0.6 --mide_tau_p 0.5 \
  --mide_beta_uni 0.4 --mide_beta_loo 0.6 \
  --mide_margin 0.05 --mide_pos_threshold 0.03 --mide_neg_threshold 0.01 \
  --mide_d_floor_min 0.65 \
  --mide_aux_start_epoch 2 --mide_gate_start_epoch 2 --mide_full_start_epoch 4 \
  --mide_lambda_uni 0.08 --mide_lambda_util_bce 0.08 --mide_lambda_rank 0.03 \
  --mide_lambda_poll_bce 0.03 --mide_lambda_noinfo 0.02 --mide_lambda_poll 0.03 \
  --mide_lambda_sparse 0.02 --mide_lambda_sep 0.01

"$PYTHON" scripts/compare_auilc.py \
  --baseline_dir results/results/baseline_fixed \
  --mide_dir results/results/mide_split_plus \
  --output results/auilc_comparison_split.csv

"$PYTHON" scripts/collect_ablation.py

echo "Remaining ablation finished."
