# MOSI Dynamic Anchor Experiment Report

## 1. Experiment Settings

- phase: full
- dataset: mosi
- missing rates: 0.1,0.2,0.3
- modes: dynamic_task_soft
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
| full | 0.1 | dynamic_task_soft | 0.882800 ± 0.008940 | 0.723800 ± 0.013783 | 0.806433 ± 0.008051 | 0.807700 ± 0.009086 | 0.430033 ± 0.031589 | 0.357600 ± 0.008525 |
| full | 0.1 | dynamic_task_soft | 0.882800 ± 0.008940 | 0.723800 ± 0.013783 | 0.806433 ± 0.008051 | 0.807700 ± 0.009086 | 0.430033 ± 0.031589 | 0.357600 ± 0.008525 |
| full | 0.2 | dynamic_task_soft | 1.093633 ± 0.089004 | 0.654700 ± 0.007503 | 0.744433 ± 0.030279 | 0.744400 ± 0.028999 | 0.356167 ± 0.057054 | 0.309033 ± 0.042250 |
| full | 0.2 | dynamic_task_soft | 1.093633 ± 0.089004 | 0.654700 ± 0.007503 | 0.744433 ± 0.030279 | 0.744400 ± 0.028999 | 0.356167 ± 0.057054 | 0.309033 ± 0.042250 |
| full | 0.3 | dynamic_task_soft | 1.080300 ± 0.050364 | 0.596800 ± 0.015670 | 0.733233 ± 0.027752 | 0.735600 ± 0.027692 | 0.318733 ± 0.017519 | 0.296867 ± 0.010736 |
| full | 0.3 | dynamic_task_soft | 1.080300 ± 0.050364 | 0.596800 ± 0.015670 | 0.733233 ± 0.027752 | 0.735600 ± 0.027692 | 0.318733 ± 0.017519 | 0.296867 ± 0.010736 |

## 4. Delta vs Fixed Text Baseline

- delta_MAE < 0 is better
- delta_Corr > 0 is better
- delta_Acc/F1 > 0 is better

(empty)

## 5. Best Method by Missing Rate

missing=0.1:
- Best MAE: dynamic_task_soft
- Best Corr: dynamic_task_soft
- Best Non0_acc_2: dynamic_task_soft
- Best Non0_F1_score: dynamic_task_soft

missing=0.2:
- Best MAE: dynamic_task_soft
- Best Corr: dynamic_task_soft
- Best Non0_acc_2: dynamic_task_soft
- Best Non0_F1_score: dynamic_task_soft

missing=0.3:
- Best MAE: dynamic_task_soft
- Best Corr: dynamic_task_soft
- Best Non0_acc_2: dynamic_task_soft
- Best Non0_F1_score: dynamic_task_soft

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

- Router note: no obvious text collapse by mean_w_text>0.85.

## 9. Conclusion

- dynamic_soft effectiveness: N/A
- Anchor-MoE effectiveness: N/A
- Quick single-seed functional check and core/full multi-seed evaluation should be interpreted separately.
- Suggested next step: finish full symmetric sweep before asymmetric missing experiments.
