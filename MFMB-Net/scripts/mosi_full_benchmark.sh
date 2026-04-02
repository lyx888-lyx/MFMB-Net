#!/usr/bin/env bash
# =============================================================================
# MOSI 完整实验矩阵（可分段运行，避免一次跑完时间过长）
#
# 用法（在 MFMB-Net 目录下）:
#   chmod +x scripts/mosi_full_benchmark.sh
#   bash scripts/mosi_full_benchmark.sh A0          # 仅跑阶段 A0
#   bash scripts/mosi_full_benchmark.sh A1          # 仅跑阶段 A1
#   bash scripts/mosi_full_benchmark.sh B1          # 对称缺失
#   bash scripts/mosi_full_benchmark.sh B2          # 两固定一扫描（只扫文本）
#   bash scripts/mosi_full_benchmark.sh B3          # 一固定两同步（固定文本）
#   bash scripts/mosi_full_benchmark.sh C1          # 训练阶段也加文本扰动
#   bash scripts/mosi_full_benchmark.sh ALL           # 顺序跑全部（极耗时）
#
# 环境变量:
#   OUT_ROOT   结果根目录，默认 results/bench_mosi_日期
#   TE_LIST    测试扰动强度列表，默认 "0 0.1 0.2 0.3"
#   MISS_LIST  对称缺失率列表，默认 "0.1 0.2 0.3"
#   PY         python 命令，默认 python3
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3}"

STAMP="${STAMP:-$(date +%Y%m%d)}"
OUT_ROOT="${OUT_ROOT:-results/bench_mosi_${STAMP}}"
TE_LIST="${TE_LIST:-0 0.1 0.2 0.3}"
MISS_LIST="${MISS_LIST:-0.1 0.2 0.3}"
MODE="${TEXT_CORRUPT_MODE:-mix}"

run_exp() {
  local tag="$1"
  local fc="$2"
  local tc="$3"
  local te="$4"
  local mt="$5"
  local ma="$6"
  local mv="$7"
  local res="${OUT_ROOT}/${tag}"
  local mod="${OUT_ROOT}_models/${tag}"
  mkdir -p "$res" "$mod"
  echo ">>> [$tag] fc=$fc tc=$tc te=$te miss=($mt,$ma,$mv)"
  "$PY" run.py \
    --datasetName mosi \
    --modelName "mfmb_net_${tag}" \
    --fusion_center_modality "$fc" \
    --text_corrupt_train "$tc" \
    --text_corrupt_eval "$te" \
    --text_corrupt_mode "$MODE" \
    --missing_t "$mt" --missing_a "$ma" --missing_v "$mv" \
    --res_save_dir "$res" \
    --model_save_dir "$mod"
}

phase_A0() {
  echo "======== A0: 无文本扰动、无缺失 — 锚点类型基线（text / dynamic / audio）========"
  for fc in text dynamic audio; do
    run_exp "A0_fc${fc}" "$fc" 0 0 0 0 0
  done
}

phase_A1() {
  echo "======== A1: 仅验证/测试文本扰动（tc=0），扫描 te × 锚点（text vs dynamic）========"
  for te in $TE_LIST; do
    te_tag=$(echo "$te" | tr '.' '_')
    for fc in text dynamic; do
      run_exp "A1_te${te_tag}_fc${fc}" "$fc" 0 "$te" 0 0 0
    done
  done
}

phase_B1() {
  echo "======== B1: 对称缺失（三模态同一 missing），无文本扰动，对比 text vs dynamic ========"
  for r in $MISS_LIST; do
    rt=$(echo "$r" | tr '.' '_')
    for fc in text dynamic; do
      run_exp "B1_miss${rt}_fc${fc}" "$fc" 0 0 "$r" "$r" "$r"
    done
  done
}

phase_B2() {
  echo "======== B2: 控制变量 — 两路固定 0.1，只增大文本缺失率 ========"
  local fix="0.1"
  for mt in $MISS_LIST; do
    mtt=$(echo "$mt" | tr '.' '_')
    for fc in text dynamic; do
      run_exp "B2_fixAV_${fix}_sweepT_${mtt}_fc${fc}" "$fc" 0 0 "$mt" "$fix" "$fix"
    done
  done
}

phase_B3() {
  echo "======== B3: 控制变量 — 文本缺失固定 0，音/视同步增大 ========"
  local ft="0.0"
  for s in $MISS_LIST; do
    st=$(echo "$s" | tr '.' '_')
    for fc in text dynamic; do
      run_exp "B3_fixT_${ft}_sweepAV_${st}_fc${fc}" "$fc" 0 0 "$ft" "$s" "$s"
    done
  done
}

phase_C1() {
  echo "======== C1: 训练阶段轻度扰动 + 测试扰动（数据增强式），仅对比 text vs dynamic ========"
  local tc=0.1
  local te=0.2
  for fc in text dynamic; do
    run_exp "C1_tc${tc}_te${te}_fc${fc}" "$fc" "$tc" "$te" 0 0 0
  done
}

phase_ALL() {
  phase_A0
  phase_A1
  phase_B1
  phase_B2
  phase_B3
  phase_C1
}

case "${1:-}" in
  A0) phase_A0 ;;
  A1) phase_A1 ;;
  B1) phase_B1 ;;
  B2) phase_B2 ;;
  B3) phase_B3 ;;
  C1) phase_C1 ;;
  ALL) phase_ALL ;;
  *)
    echo "用法: $0 {A0|A1|B1|B2|B3|C1|ALL}"
    echo "当前 OUT_ROOT=$OUT_ROOT"
    exit 1
    ;;
esac

echo "完成。CSV 位于: ${OUT_ROOT}/*/normals/mosi-regression-*.csv"
