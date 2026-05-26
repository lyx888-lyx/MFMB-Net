# MOSI Dynamic Anchor Experiment Report

## 1. Experiment Settings

- phase: full
- dataset: mosi
- missing rates: 0.0
- modes: dynamic_rta_pred_residual
- seeds in this run: [111, 1111, 11111]
- quick_single_seed: 0
- router_missing_bias: 2.0
- train_drop_last: 1
- eval_drop_last: 0
- test_drop_last: 0
- symmetric missing only
This experiment only uses symmetric missing rates via --missing. The asymmetric missing parameters --missing_t, --missing_a, and --missing_v are reserved but not used in this stage.

This official full-test protocol uses train_drop_last=1, eval_drop_last=0, and test_drop_last=0. The test set is evaluated on all 686 MOSI samples. Previous exploratory results under drop_last=True are not mixed with this official protocol.

## 2. Smoke Test Results

| missing | mode | status |
| --- | --- | --- |
| 0.0 | text | FAIL |
| 0.3 | audio | FAIL |
| 0.3 | vision | FAIL |
| 0.3 | dynamic_soft | FAIL |
| 0.3 | dynamic_soft_moe | FAIL |

## 3. Overall Results

| phase | missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 0.0 | dynamic_rta_pred_residual | 0.881133 ± 0.147862 | 0.764367 ± 0.047585 | 0.829267 ± 0.013068 | 0.829233 ± 0.013530 | 0.421300 ± 0.094783 | 0.371200 ± 0.068753 |

## 4. Delta vs Fixed Text Baseline

- delta_MAE < 0 is better
- delta_Corr > 0 is better
- delta_Acc/F1 > 0 is better

(empty)

## 5. Best Method by Missing Rate

missing=0.0:
- Best MAE: dynamic_rta_pred_residual
- Best Corr: dynamic_rta_pred_residual
- Best Non0_acc_2: dynamic_rta_pred_residual
- Best Non0_F1_score: dynamic_rta_pred_residual

## 6. Dynamic Anchor vs Fixed Text

- dynamic_soft vs text: N/A
- dynamic_soft_moe vs text: N/A

## 7. Anchor-MoE Effect

- dynamic_soft_moe vs dynamic_soft: N/A
- If MoE is not better, possible reasons include small MOSI size, overfitting risk, weak expert split, and limited center feature gap.

## 8. Router Behavior Analysis

(empty)

- Router note: No router CSV found.

## 9. Conclusion

- dynamic_soft effectiveness: N/A
- Anchor-MoE effectiveness: N/A
- Quick single-seed functional check and core/full multi-seed evaluation should be interpreted separately.
- Suggested next step: finish full symmetric sweep before asymmetric missing experiments.
