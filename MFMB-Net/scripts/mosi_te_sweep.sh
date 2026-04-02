#!/usr/bin/env bash
# 仅测试集文本扰动强度 te 扫描：tc=0，无缺失；对比 fusion_center = text vs dynamic
#
# 用法（在 MFMB-Net 目录下）:
#   bash scripts/mosi_te_sweep.sh
#
# 环境变量:
#   TE_LIST   空格分隔，默认 "0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5"
#   OUT_ROOT  结果根目录，默认 results/bench_te_sweep_日期
#   TC        训练扰动，默认 0（只扫测试 te）
#   MODE      text_corrupt_mode，默认 mix
#   PY        python，默认 python3

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"

STAMP="${STAMP:-$(date +%Y%m%d)}"
OUT_ROOT="${OUT_ROOT:-results/bench_te_sweep_${STAMP}}"
TE_LIST="${TE_LIST:-0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5}"
TC="${TC:-0}"
MODE="${MODE:-mix}"

mkdir -p "$OUT_ROOT"

run_one() {
  local te="$1"
  local fc="$2"
  local te_tag
  te_tag=$(echo "$te" | tr '.' '_')
  local tag="A1_te${te_tag}_fc${fc}"
  local res="${OUT_ROOT}/${tag}"
  local mod="${OUT_ROOT}_models/${tag}"
  mkdir -p "$res" "$mod"
  echo ">>> te=$te fc=$fc -> $tag"
  "$PY" run.py \
    --datasetName mosi \
    --modelName "mfmb_net_${tag}" \
    --fusion_center_modality "$fc" \
    --text_corrupt_train "$TC" \
    --text_corrupt_eval "$te" \
    --text_corrupt_mode "$MODE" \
    --missing_t 0 --missing_a 0 --missing_v 0 \
    --res_save_dir "$res" \
    --model_save_dir "$mod"
}

for te in $TE_LIST; do
  for fc in text dynamic; do
    run_one "$te" "$fc"
  done
done

echo "完成。结果目录: $OUT_ROOT/*/normals/"
echo "绘图: $PY scripts/plot_mosi_te_curves.py --root $OUT_ROOT --out ${OUT_ROOT}/te_sweep_curves.png"
