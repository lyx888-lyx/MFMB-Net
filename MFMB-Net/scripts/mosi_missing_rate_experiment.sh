#!/usr/bin/env bash
# =============================================================================
# MOSI 缺失率实验（非对称，适合观察「动态锚点」：T/A/V 缺失率不必相同）
#
# 两种控制变量设计（你描述的）：
#   A) fix1_sweep2 — 固定一个模态的缺失率，另外两个模态缺失率同步逐步调大
#   B) fix2_sweep1 — 固定两个模态的缺失率，剩下一个模态缺失率逐步调大
#
# 另保留 symmetric：三模态同一缺失率（仅作基线对比，可选）
#
# 在 MFMB-Net 目录下执行：
#   bash scripts/mosi_missing_rate_experiment.sh
#
# 指定模式与参数示例见下方 case 分支。
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

FUSION_CENTER="${FUSION_CENTER:-dynamic}"
MODALITY_GATE="${MODALITY_GATE:-integrity}"
FUSION_PROMPT_DIM="${FUSION_PROMPT_DIM:-0}"

# 扫描步长（缺省与原先一致）
SWEEP_RATES="${SWEEP_RATES:-0.0 0.1 0.2 0.3 0.5}"

# 实验模式：symmetric | fix1_sweep2 | fix2_sweep1
EXP_MODE="${EXP_MODE:-fix2_sweep1}"

OUT_TAG="${OUT_TAG:-mosi_missing_${FUSION_CENTER}_gate-${MODALITY_GATE}}"
RES_DIR="results/results/${OUT_TAG}"
MODEL_DIR="results/models/${OUT_TAG}"

mkdir -p "$RES_DIR" "$MODEL_DIR"

run_one() {
  local mt="$1" ma="$2" mv="$3"
  local tag="T${mt}_A${ma}_V${mv}"
  echo "======== missing (T,A,V) = (${mt},${ma},${mv})  [${tag}] ========"
  "$PY" run.py \
    --datasetName mosi \
    --modelName mfmb_net \
    --fusion_center_modality "$FUSION_CENTER" \
    --modality_gate "$MODALITY_GATE" \
    --fusion_prompt_dim "$FUSION_PROMPT_DIM" \
    --missing_t "$mt" \
    --missing_a "$ma" \
    --missing_v "$mv" \
    --res_save_dir "$RES_DIR" \
    --model_save_dir "$MODEL_DIR"
}

echo "res=$RES_DIR  ckpt=$MODEL_DIR"
echo "EXP_MODE=$EXP_MODE  fusion_center=$FUSION_CENTER  modality_gate=$MODALITY_GATE"
echo "SWEEP_RATES=$SWEEP_RATES"

case "$EXP_MODE" in
  symmetric)
    # 三模态同一缺失率（基线）
    for r in $SWEEP_RATES; do
      run_one "$r" "$r" "$r"
    done
    ;;

  fix1_sweep2)
    # 固定一个模态为 FIXED_RATE，另外两个模态同步取 SWEEP_RATES 中的同一值
    # FIXED_MODALITY: t | a | v
    FIXED_MODALITY="${FIXED_MODALITY:-t}"
    FIXED_RATE="${FIXED_RATE:-0.0}"
    for s in $SWEEP_RATES; do
      case "$FIXED_MODALITY" in
        t) run_one "$FIXED_RATE" "$s" "$s" ;;
        a) run_one "$s" "$FIXED_RATE" "$s" ;;
        v) run_one "$s" "$s" "$FIXED_RATE" ;;
        *) echo "FIXED_MODALITY must be t, a, or v"; exit 1 ;;
      esac
    done
    ;;

  fix2_sweep1)
    # 两个模态固定为 PAIR_FIXED_RATE（可相同），第三个模态沿 SWEEP_RATES 扫描
    # SWEEP_MODALITY: t | a | v
    SWEEP_MODALITY="${SWEEP_MODALITY:-t}"
    PAIR_FIXED_RATE="${PAIR_FIXED_RATE:-0.1}"
    for s in $SWEEP_RATES; do
      case "$SWEEP_MODALITY" in
        t) run_one "$s" "$PAIR_FIXED_RATE" "$PAIR_FIXED_RATE" ;;
        a) run_one "$PAIR_FIXED_RATE" "$s" "$PAIR_FIXED_RATE" ;;
        v) run_one "$PAIR_FIXED_RATE" "$PAIR_FIXED_RATE" "$s" ;;
        *) echo "SWEEP_MODALITY must be t, a, or v"; exit 1 ;;
      esac
    done
    ;;

  *)
    echo "Unknown EXP_MODE=$EXP_MODE (use symmetric | fix1_sweep2 | fix2_sweep1)"
    exit 1
    ;;
esac

echo "Done. CSV under: $RES_DIR/mosi-regression-*.csv"

# -----------------------------------------------------------------------------
# 可选：一次跑完 6 条控制曲线（每组用独立 OUT_TAG 子目录，避免混在一起）
#
#   # A) 一固定两同步：分别固定 T、A、V
#   for m in t a v; do
#     EXP_MODE=fix1_sweep2 FIXED_MODALITY=$m FIXED_RATE=0.0 \
#       OUT_TAG="mosi_${FUSION_CENTER:-dynamic}_fix1fix-${m}" \
#       bash scripts/mosi_missing_rate_experiment.sh
#   done
#
#   # B) 两固定一扫描：分别只扫 T、只扫 A、只扫 V（另两路固定为 PAIR_FIXED_RATE）
#   for m in t a v; do
#     EXP_MODE=fix2_sweep1 SWEEP_MODALITY=$m PAIR_FIXED_RATE=0.1 \
#       OUT_TAG="mosi_${FUSION_CENTER:-dynamic}_fix2sweep-${m}" \
#       bash scripts/mosi_missing_rate_experiment.sh
#   done
#
# 单次覆盖示例：
#   EXP_MODE=fix1_sweep2 FIXED_MODALITY=a FIXED_RATE=0.0 SWEEP_RATES="0 0.2 0.4" bash scripts/mosi_missing_rate_experiment.sh
#   EXP_MODE=fix2_sweep1 SWEEP_MODALITY=v PAIR_FIXED_RATE=0.05 SWEEP_RATES="0 0.1 0.3" bash scripts/mosi_missing_rate_experiment.sh
# -----------------------------------------------------------------------------
