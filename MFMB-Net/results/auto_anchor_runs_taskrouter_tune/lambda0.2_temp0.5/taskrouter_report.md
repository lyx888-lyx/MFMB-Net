# Task-aware Dynamic Anchor Router Report

## 1. Motivation
Missing-aware routing mainly captures modality availability, but availability does not always match task contribution. We add center-wise task supervision to improve router decisions.

## 2. Method
- Three auxiliary center predictors (text/audio/vision).
- Oracle center distribution from center-wise prediction error.
- Router supervised by KL/CE against oracle during training.
- Test-time routing does not use labels.

## 3. Results
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | audio | 1.084633 | 0.611333 | 0.7449 | 0.7457 | 0.340633 | 0.2993 |
| 0.3 | dynamic_soft | 1.060967 | 0.6053 | 0.740867 | 0.739533 | 0.336233 | 0.3047 |
| 0.3 | dynamic_soft_moe | 1.0008 | 0.614767 | 0.7495 | 0.749733 | 0.3421 | 0.3129 |
| 0.3 | text | 1.077033 | 0.616667 | 0.729167 | 0.7334 | 0.329433 | 0.297367 |
| 0.3 | vision | 0.972433 | 0.620167 | 0.764733 | 0.7673 | 0.398933 | 0.343533 |
| 0.4 | audio | 1.134667 | 0.533367 | 0.711367 | 0.710167 | 0.287667 | 0.262867 |
| 0.4 | dynamic_soft | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 |
| 0.4 | dynamic_soft_moe | 1.1596 | 0.526367 | 0.714433 | 0.7211 | 0.311 | 0.2697 |
| 0.4 | dynamic_task_soft | 1.1462 | 0.5454333333333333 | 0.7129 | 0.7194333333333334 | 0.2556 | 0.2488 |
| 0.4 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 |
| 0.4 | vision | 1.221933 | 0.524067 | 0.702233 | 0.704867 | 0.246367 | 0.232733 |
| 0.5 | audio | 1.2776 | 0.456 | 0.645833 | 0.652633 | 0.263833 | 0.243467 |
| 0.5 | dynamic_soft | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 |
| 0.5 | dynamic_soft_moe | 1.348567 | 0.449033 | 0.6123 | 0.657767 | 0.253167 | 0.241033 |
| 0.5 | dynamic_task_soft | 1.1668333333333332 | 0.46703333333333336 | 0.7012 | 0.7018666666666666 | 0.26286666666666664 | 0.25366666666666665 |
| 0.5 | text | 1.259267 | 0.464733 | 0.6626 | 0.666833 | 0.2449 | 0.2289 |
| 0.5 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 |

## 4. Delta Analysis
### dynamic_task_soft vs dynamic_soft
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE_ds | Corr_ds | Non0_acc_2_ds | Non0_F1_score_ds | Mult_acc_5_ds | Mult_acc_7_ds | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.1462 | 0.5454333333333333 | 0.7129 | 0.7194333333333334 | 0.2556 | 0.2488 | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 | 0.05406700000000009 | -0.0069666666666666766 | -0.013732999999999995 | -0.0056996666666666584 | -0.06506699999999999 | -0.04033299999999998 |
| 0.5 | dynamic_task_soft | 1.1668333333333332 | 0.46703333333333336 | 0.7012 | 0.7018666666666666 | 0.26286666666666664 | 0.25366666666666665 | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 | -0.13193366666666684 | 0.08820033333333338 | 0.07720000000000005 | 0.060766666666666636 | 0.04759966666666665 | 0.04373366666666664 |

### dynamic_task_soft vs best fixed center
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | best_fixed_mode | MAE_fixed | Corr_fixed | Non0_acc_2_fixed | Non0_F1_score_fixed | Mult_acc_5_fixed | Mult_acc_7_fixed | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.1462 | 0.5454333333333333 | 0.7129 | 0.7194333333333334 | 0.2556 | 0.2488 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 | 0.040333000000000174 | -0.004233666666666691 | 0.007633000000000001 | 0.014900333333333404 | -0.058767000000000014 | -0.032033000000000006 |
| 0.5 | dynamic_task_soft | 1.1668333333333332 | 0.46703333333333336 | 0.7012 | 0.7018666666666666 | 0.26286666666666664 | 0.25366666666666665 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 | -0.08259966666666685 | 0.005100333333333373 | 0.030467000000000022 | 0.025566666666666626 | -0.03790033333333337 | -0.017966333333333362 |

## 5. Router Diagnostics
| missing | mode | router_oracle_match_rate | mean_w_text | mean_w_audio | mean_w_vision | mean_oracle_w_text | mean_oracle_w_audio | mean_oracle_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | oracle_text_ratio | oracle_audio_ratio | oracle_vision_ratio | entropy_router | entropy_oracle | corr_router_oracle_text | corr_router_oracle_audio | corr_router_oracle_vision | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 0.36734693877551017 | 0.32217146563764565 | 0.3261287531827235 | 0.3516997814909768 | 0.33577124257476965 | 0.33551817684931246 | 0.3287105806193616 | 0.3032069970845481 | 0.228377065111759 | 0.4684159378036929 | 0.38338192419825073 | 0.32118561710398447 | 0.2954324586977648 | 1.0793567888173412 | 1.0958397250309042 | 0.01032178761769724 | -0.012976357582025757 | 0.024896524681961917 | 0.8892780867518896 | 0.7762535763266 | 0.7870973588866162 |
| 0.5 | dynamic_task_soft | 0.35034013605442177 | 0.3318181715912907 | 0.32424878373206295 | 0.3439330458749661 | 0.33479384004895957 | 0.3321646257467474 | 0.33304153633303035 | 0.3464528668610301 | 0.25704567541302237 | 0.3965014577259475 | 0.44509232264334303 | 0.3644314868804665 | 0.19047619047619047 | 1.0815973146705253 | 1.096592469156408 | 0.03789913899654145 | 0.045729519133780615 | 0.01680684034483497 | 0.885495622200867 | 0.7541055715799062 | 0.7560533174764402 |

## 6. Conclusion
- dynamic_task_soft vs dynamic_soft: MAE better on 1/2, Corr better on 1/2.
- dynamic_task_soft vs best fixed: MAE better on 1/2.
- mean router-oracle match rate: 0.3588.
- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.
