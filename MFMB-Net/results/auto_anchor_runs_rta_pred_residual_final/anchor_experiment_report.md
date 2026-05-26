# MOSI Dynamic Anchor Experiment Report

## 1. Experiment Settings

- phase: full
- dataset: mosi
- missing rates: 0.1,0.2,0.3,0.4,0.5
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
| full | 0.1 | dynamic_rta_pred_residual | 0.906800 ± 0.027406 | 0.718067 ± 0.022442 | 0.793200 ± 0.006864 | 0.792433 ± 0.007053 | 0.416400 ± 0.035953 | 0.358100 ± 0.012573 |
| full | 0.2 | dynamic_rta_pred_residual | 0.935633 ± 0.012126 | 0.669967 ± 0.005888 | 0.784533 ± 0.006357 | 0.784367 ± 0.007382 | 0.413500 ± 0.015441 | 0.361500 ± 0.011918 |
| full | 0.3 | dynamic_rta_pred_residual | 1.022433 ± 0.028307 | 0.616767 ± 0.015318 | 0.748467 ± 0.007910 | 0.747533 ± 0.008458 | 0.353233 ± 0.030646 | 0.312400 ± 0.018533 |
| full | 0.4 | dynamic_rta_pred_residual | 1.153433 ± 0.075990 | 0.535233 ± 0.007946 | 0.708833 ± 0.029472 | 0.709500 ± 0.027161 | 0.275000 ± 0.015743 | 0.259967 ± 0.014907 |
| full | 0.5 | dynamic_rta_pred_residual | 1.187467 ± 0.026697 | 0.454233 ± 0.013816 | 0.694600 ± 0.006161 | 0.700967 ± 0.008594 | 0.265300 ± 0.005023 | 0.257033 ± 0.004430 |

## 4. Delta vs Fixed Text Baseline

- delta_MAE < 0 is better
- delta_Corr > 0 is better
- delta_Acc/F1 > 0 is better

(empty)

## 5. Best Method by Missing Rate

missing=0.1:
- Best MAE: dynamic_rta_pred_residual
- Best Corr: dynamic_rta_pred_residual
- Best Non0_acc_2: dynamic_rta_pred_residual
- Best Non0_F1_score: dynamic_rta_pred_residual

missing=0.2:
- Best MAE: dynamic_rta_pred_residual
- Best Corr: dynamic_rta_pred_residual
- Best Non0_acc_2: dynamic_rta_pred_residual
- Best Non0_F1_score: dynamic_rta_pred_residual

missing=0.3:
- Best MAE: dynamic_rta_pred_residual
- Best Corr: dynamic_rta_pred_residual
- Best Non0_acc_2: dynamic_rta_pred_residual
- Best Non0_F1_score: dynamic_rta_pred_residual

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
| mosi | 0.1 | dynamic_soft | 0.675958840428074 | 0.023888267317431395 | 0.30015289182094945 | 0.6666666666666666 | 0.0 | 0.3333333333333333 | 0.13944506701052314 | 0.7394499738815699 | 0.3687889985672326 | 0.42829745275497083 | 3 | text | 0.675958840428074 | False |
| mosi | 0.1 | dynamic_soft_moe | 0.028657572522685585 | 0.37188613605659127 | 0.5994562913615716 | 0.0 | 0.33430515063168126 | 0.6656948493683187 | 0.4554785288344263 | 0.6096604338509736 | 0.4533725301720945 | 0.4257734026789632 | 3 | vision | 0.5994562913615716 | False |
| mosi | 0.2 | dynamic_soft | 0.00040461064920449064 | 0.3338753631178348 | 0.6657200218563926 | 0.0 | 0.3333333333333333 | 0.6666666666666666 | 0.007935240271307216 | 0.5485572572176601 | 0.4465494840463209 | 0.4883719113165497 | 3 | vision | 0.6657200218563926 | False |
| mosi | 0.2 | dynamic_soft_moe | 0.10854009394810717 | 0.4387945658147545 | 0.4526653362892365 | 0.10349854227405247 | 0.38338192419825073 | 0.5131195335276968 | 0.3627223425824668 | 0.6108285253761563 | 0.6169848523689964 | 0.47323470244398064 | 3 | vision | 0.4526653362892365 | False |
| mosi | 0.3 | dynamic_soft | 0.3333071701560912 | 0.33353761192397674 | 0.3331552182269338 | 0.3333333333333333 | 0.3333333333333333 | 0.3333333333333333 | 0.002422184929430427 | 0.5309201112612253 | 0.5180920616268136 | 0.46624445279558063 | 3 | audio | 0.33353761192397674 | False |
| mosi | 0.3 | dynamic_soft_moe | 0.19853862259901045 | 0.6957354412221353 | 0.1057259372162262 | 0.18464528668610303 | 0.815354713313897 | 0.0 | 0.6222525947694221 | 0.7833768387181766 | 0.7209705031214435 | 0.6496943877236864 | 3 | audio | 0.6957354412221353 | False |
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
