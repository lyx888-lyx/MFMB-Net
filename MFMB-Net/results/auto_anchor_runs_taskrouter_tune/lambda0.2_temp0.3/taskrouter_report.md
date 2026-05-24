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
| 0.4 | dynamic_task_soft | 1.1277333333333333 | 0.5472 | 0.7205333333333334 | 0.7202666666666667 | 0.28473333333333334 | 0.2682333333333333 |
| 0.4 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 |
| 0.4 | vision | 1.221933 | 0.524067 | 0.702233 | 0.704867 | 0.246367 | 0.232733 |
| 0.5 | audio | 1.2776 | 0.456 | 0.645833 | 0.652633 | 0.263833 | 0.243467 |
| 0.5 | dynamic_soft | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 |
| 0.5 | dynamic_soft_moe | 1.348567 | 0.449033 | 0.6123 | 0.657767 | 0.253167 | 0.241033 |
| 0.5 | dynamic_task_soft | 1.2500666666666669 | 0.4498333333333333 | 0.6499 | 0.6683333333333333 | 0.24586666666666668 | 0.22646666666666668 |
| 0.5 | text | 1.259267 | 0.464733 | 0.6626 | 0.666833 | 0.2449 | 0.2289 |
| 0.5 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 |

## 4. Delta Analysis
### dynamic_task_soft vs dynamic_soft
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE_ds | Corr_ds | Non0_acc_2_ds | Non0_F1_score_ds | Mult_acc_5_ds | Mult_acc_7_ds | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.1277333333333333 | 0.5472 | 0.7205333333333334 | 0.7202666666666667 | 0.28473333333333334 | 0.2682333333333333 | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 | 0.035600333333333234 | -0.005199999999999982 | -0.006099666666666614 | -0.004866333333333306 | -0.03593366666666664 | -0.02089966666666665 |
| 0.5 | dynamic_task_soft | 1.2500666666666669 | 0.4498333333333333 | 0.6499 | 0.6683333333333333 | 0.24586666666666668 | 0.22646666666666668 | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 | -0.04870033333333312 | 0.07100033333333333 | 0.025900000000000034 | 0.02723333333333333 | 0.03059966666666669 | 0.01653366666666667 |

### dynamic_task_soft vs best fixed center
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | best_fixed_mode | MAE_fixed | Corr_fixed | Non0_acc_2_fixed | Non0_F1_score_fixed | Mult_acc_5_fixed | Mult_acc_7_fixed | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.1277333333333333 | 0.5472 | 0.7205333333333334 | 0.7202666666666667 | 0.28473333333333334 | 0.2682333333333333 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 | 0.02186633333333332 | -0.002466999999999997 | 0.015266333333333382 | 0.015733666666666757 | -0.02963366666666667 | -0.012599666666666676 |
| 0.5 | dynamic_task_soft | 1.2500666666666669 | 0.4498333333333333 | 0.6499 | 0.6683333333333333 | 0.24586666666666668 | 0.22646666666666668 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 | 0.0006336666666668656 | -0.012099666666666675 | -0.02083299999999999 | -0.007966666666666677 | -0.05490033333333333 | -0.045166333333333336 |

## 5. Router Diagnostics
| missing | mode | router_oracle_match_rate | mean_w_text | mean_w_audio | mean_w_vision | mean_oracle_w_text | mean_oracle_w_audio | mean_oracle_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | oracle_text_ratio | oracle_audio_ratio | oracle_vision_ratio | entropy_router | entropy_oracle | corr_router_oracle_text | corr_router_oracle_audio | corr_router_oracle_vision | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 0.3216715257531584 | 0.31396282513863955 | 0.34289864617951066 | 0.343138528888207 | 0.3343131304714029 | 0.3307009815329829 | 0.33498588844453164 | 0.2468415937803693 | 0.3726919339164237 | 0.38046647230320696 | 0.337220602526725 | 0.2662779397473275 | 0.3965014577259475 | 1.0820468039694864 | 1.0976465632658217 | -0.011858896954161586 | 0.00496991992527841 | -0.015725588948738142 | 0.8889276969266264 | 0.7782347826167962 | 0.7920257053274876 |
| 0.5 | dynamic_task_soft | 0.33916423712342086 | 0.3414076785145163 | 0.3404210951646285 | 0.3181712258284942 | 0.34578970370794293 | 0.33078098680871104 | 0.3234293094978273 | 0.3760932944606414 | 0.33770651117589895 | 0.28620019436345967 | 0.3688046647230321 | 0.2725947521865889 | 0.35860058309037895 | 1.076887146826274 | 1.082529823661044 | -0.018371664580276992 | -0.02491888816261766 | -0.004399710997416693 | 0.8821119986367374 | 0.7497009083700373 | 0.7596785841231698 |

## 6. Conclusion
- dynamic_task_soft vs dynamic_soft: MAE better on 1/2, Corr better on 1/2.
- dynamic_task_soft vs best fixed: MAE better on 0/2.
- mean router-oracle match rate: 0.3304.
- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.
