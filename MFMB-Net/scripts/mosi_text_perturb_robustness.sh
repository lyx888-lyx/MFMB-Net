#!/usr/bin/env bash
# 文本扰动下对比：固定文本锚点(text) vs 动态锚点(dynamic)。
# 常见设置：训练阶段不扰动（tc=0），仅在验证/测试阶段扰动（te>0），便于看泛化鲁棒性。
#
#   cd MFMB-Net && bash scripts/mosi_text_perturb_robustness.sh
#
# 环境变量：TE=测试集扰动强度，MODE=mix|token|span，OUT=结果子目录名

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

TE="${TE:-0.2}"
MODE="${MODE:-mix}"
OUT="${OUT:-text_perturb_compare}"

RES="results/results/${OUT}"
MOD="results/models/${OUT}"
mkdir -p "$RES" "$MOD"

run_exp() {
  local name="$1"
  local fc="$2"
  echo "=== $name (fusion_center=$fc) te=$TE mode=$MODE ==="
  "$PY" run.py \
    --datasetName mosi \
    --modelName "${name}" \
    --fusion_center_modality "$fc" \
    --text_corrupt_train 0 \
    --text_corrupt_eval "$TE" \
    --text_corrupt_mode "$MODE" \
    --res_save_dir "$RES" \
    --model_save_dir "$MOD"
}

# 固定文本锚点 + 文本扰动（基线）
run_exp "mfmb_net_text_anchor" "text"

# 动态锚点 + 同一文本扰动
run_exp "mfmb_net_dynamic_anchor" "dynamic"

echo "结果 CSV 见: $RES/mosi-regression-*-fc*.csv"
