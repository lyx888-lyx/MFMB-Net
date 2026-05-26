# MOSI Dynamic Anchor Experiment Report

## 1. Experiment Settings

- phase: full
- dataset: mosi
- missing rates: 0.4,0.5
- modes: dynamic_rta_pred_residual
- seeds in this run: [111]
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
| full | 0.4 | dynamic_rta_pred_residual | 1.159400 | 0.550800 | 0.689000 | 0.694300 | 0.316300 | 0.281300 |
| full | 0.5 | dynamic_rta_pred_residual | 1.264400 | 0.480000 | 0.661600 | 0.665000 | 0.268200 | 0.247800 |

## 4. Delta vs Fixed Text Baseline

- delta_MAE < 0 is better
- delta_Corr > 0 is better
- delta_Acc/F1 > 0 is better

(empty)

## 5. Best Method by Missing Rate

missing=0.4:
- Best MAE: dynamic_rta_pred_residual
- Best Corr: dynamic_rta_pred_residual
- Best Non0_acc_2: dynamic_rta_pred_residual
- Best Non0_F1_score: dynamic_rta_pred_residual

missing=0.5:
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

| dataset | missing | mode | mean_w_text | mean_w_audio | mean_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | entropy | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision | seed_count | dominant_anchor | dominant_anchor_ratio | is_router_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mosi | 0.4 | dynamic_soft | 0.6668740382732946 | 0.001148396834912145 | 0.3319775600976311 | 0.6666666666666666 | 0.0 | 0.3333333333333333 | 0.014474570542774672 | 0.5954153004646883 | 0.4000625121572034 | 0.45006107603795803 | 3 | text | 0.6668740382732946 | False |
| mosi | 0.4 | dynamic_soft_moe | 0.009483527622252977 | 0.539917171832912 | 0.45059929947328126 | 0.0 | 0.6584062196307094 | 0.34159378036929056 | 0.2541894362764196 | 0.5554751328998743 | 0.44097794746998925 | 0.4749327283233029 | 3 | audio | 0.539917171832912 | False |
| mosi | 0.5 | dynamic_soft | 0.328988862298858 | 0.340057032350195 | 0.33095410453116964 | 0.3333333333333333 | 0.3333333333333333 | 0.3333333333333333 | 0.045133119844568255 | 0.6304190390767941 | 0.5440873914586765 | 0.3928003563819453 | 3 | audio | 0.340057032350195 | False |
| mosi | 0.5 | dynamic_soft_moe | 0.08722109015755063 | 0.5576959450316844 | 0.35508296192703565 | 0.0009718172983479105 | 0.6656948493683187 | 0.3333333333333333 | 0.26808485331094073 | 0.6053629708095937 | 0.4296147750127716 | 0.5014652557689543 | 3 | audio | 0.5576959450316844 | False |

- Router note: no obvious text collapse by mean_w_text>0.85.

## 9. Conclusion

- dynamic_soft effectiveness: N/A
- Anchor-MoE effectiveness: N/A
- Quick single-seed functional check and core/full multi-seed evaluation should be interpreted separately.
- Suggested next step: finish full symmetric sweep before asymmetric missing experiments.
