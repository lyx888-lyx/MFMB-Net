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
| 0.4 | dynamic_task_soft | 1.0896 | 0.5194 | 0.7256 | 0.7248333333333333 | 0.3100333333333333 | 0.2920333333333333 |
| 0.4 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 |
| 0.4 | vision | 1.221933 | 0.524067 | 0.702233 | 0.704867 | 0.246367 | 0.232733 |
| 0.5 | audio | 1.2776 | 0.456 | 0.645833 | 0.652633 | 0.263833 | 0.243467 |
| 0.5 | dynamic_soft | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 |
| 0.5 | dynamic_soft_moe | 1.348567 | 0.449033 | 0.6123 | 0.657767 | 0.253167 | 0.241033 |
| 0.5 | dynamic_task_soft | 1.2121333333333333 | 0.46626666666666666 | 0.6854666666666667 | 0.6860666666666666 | 0.25753333333333334 | 0.241 |
| 0.5 | text | 1.259267 | 0.464733 | 0.6626 | 0.666833 | 0.2449 | 0.2289 |
| 0.5 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 |

## 4. Delta Analysis
### dynamic_task_soft vs dynamic_soft
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE_ds | Corr_ds | Non0_acc_2_ds | Non0_F1_score_ds | Mult_acc_5_ds | Mult_acc_7_ds | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.0896 | 0.5194 | 0.7256 | 0.7248333333333333 | 0.3100333333333333 | 0.2920333333333333 | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 | -0.0025330000000001185 | -0.03300000000000003 | -0.0010329999999999506 | -0.00029966666666669806 | -0.010633666666666652 | 0.002900333333333338 |
| 0.5 | dynamic_task_soft | 1.2121333333333333 | 0.46626666666666666 | 0.6854666666666667 | 0.6860666666666666 | 0.25753333333333334 | 0.241 | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 | -0.08663366666666672 | 0.08743366666666669 | 0.06146666666666667 | 0.0449666666666666 | 0.04226633333333335 | 0.031066999999999984 |

### dynamic_task_soft vs best fixed center
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | best_fixed_mode | MAE_fixed | Corr_fixed | Non0_acc_2_fixed | Non0_F1_score_fixed | Mult_acc_5_fixed | Mult_acc_7_fixed | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.0896 | 0.5194 | 0.7256 | 0.7248333333333333 | 0.3100333333333333 | 0.2920333333333333 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 | -0.01626700000000003 | -0.030267000000000044 | 0.020333000000000045 | 0.020300333333333365 | -0.00433366666666668 | 0.011200333333333312 |
| 0.5 | dynamic_task_soft | 1.2121333333333333 | 0.46626666666666666 | 0.6854666666666667 | 0.6860666666666666 | 0.25753333333333334 | 0.241 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 | -0.03729966666666673 | 0.00433366666666668 | 0.014733666666666645 | 0.00976666666666659 | -0.04323366666666667 | -0.03063300000000002 |

## 5. Router Diagnostics
| missing | mode | router_oracle_match_rate | mean_w_text | mean_w_audio | mean_w_vision | mean_oracle_w_text | mean_oracle_w_audio | mean_oracle_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | oracle_text_ratio | oracle_audio_ratio | oracle_vision_ratio | entropy_router | entropy_oracle | corr_router_oracle_text | corr_router_oracle_audio | corr_router_oracle_vision | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 0.36977648202137997 | 0.349167934891318 | 0.31695200261760387 | 0.3338800644659527 | 0.3265168829173921 | 0.3330190096655091 | 0.3404641069899032 | 0.4096209912536443 | 0.2560738581146744 | 0.33430515063168126 | 0.35860058309037895 | 0.22351797862001943 | 0.4178814382896016 | 1.0799032135935407 | 1.0939550508352744 | -0.03406842640650381 | -0.020105605750774636 | -0.0006053127269129525 | 0.8906035322186804 | 0.7691323462123892 | 0.7778652474414144 |
| 0.5 | dynamic_task_soft | 0.27988338192419826 | 0.38739853633243676 | 0.2942810173321106 | 0.3183204455896705 | 0.3327708479064084 | 0.3349639355586036 | 0.33226521647706325 | 0.5126336248785228 | 0.26822157434402333 | 0.21914480077745382 | 0.26919339164237127 | 0.4241982507288629 | 0.30660835762876576 | 1.0578938847440682 | 1.0981337773913997 | 0.011753036601191954 | -0.030662503775905365 | 0.012690784479436355 | 0.8864336013068845 | 0.7428305407966728 | 0.7370081114670232 |

## 6. Conclusion
- dynamic_task_soft vs dynamic_soft: MAE better on 2/2, Corr better on 1/2.
- dynamic_task_soft vs best fixed: MAE better on 2/2.
- mean router-oracle match rate: 0.3248.
- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.
