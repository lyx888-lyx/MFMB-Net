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
| 0.4 | dynamic_task_soft | 1.2050333333333334 | 0.5156333333333333 | 0.6717333333333334 | 0.6760999999999999 | 0.2702 | 0.2483 |
| 0.4 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 |
| 0.4 | vision | 1.221933 | 0.524067 | 0.702233 | 0.704867 | 0.246367 | 0.232733 |
| 0.5 | audio | 1.2776 | 0.456 | 0.645833 | 0.652633 | 0.263833 | 0.243467 |
| 0.5 | dynamic_soft | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 |
| 0.5 | dynamic_soft_moe | 1.348567 | 0.449033 | 0.6123 | 0.657767 | 0.253167 | 0.241033 |
| 0.5 | dynamic_task_soft | 1.3057666666666667 | 0.41619999999999996 | 0.5675666666666667 | 0.6240333333333333 | 0.23423333333333332 | 0.22643333333333335 |
| 0.5 | text | 1.259267 | 0.464733 | 0.6626 | 0.666833 | 0.2449 | 0.2289 |
| 0.5 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 |

## 4. Delta Analysis
### dynamic_task_soft vs dynamic_soft
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE_ds | Corr_ds | Non0_acc_2_ds | Non0_F1_score_ds | Mult_acc_5_ds | Mult_acc_7_ds | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.2050333333333334 | 0.5156333333333333 | 0.6717333333333334 | 0.6760999999999999 | 0.2702 | 0.2483 | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 | 0.11290033333333338 | -0.036766666666666725 | -0.05489966666666657 | -0.049033000000000104 | -0.050466999999999984 | -0.04083299999999998 |
| 0.5 | dynamic_task_soft | 1.3057666666666667 | 0.41619999999999996 | 0.5675666666666667 | 0.6240333333333333 | 0.23423333333333332 | 0.22643333333333335 | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 | 0.006999666666666737 | 0.037366999999999984 | -0.056433333333333335 | -0.017066666666666674 | 0.018966333333333335 | 0.01650033333333334 |

### dynamic_task_soft vs best fixed center
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | best_fixed_mode | MAE_fixed | Corr_fixed | Non0_acc_2_fixed | Non0_F1_score_fixed | Mult_acc_5_fixed | Mult_acc_7_fixed | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.2050333333333334 | 0.5156333333333333 | 0.6717333333333334 | 0.6760999999999999 | 0.2702 | 0.2483 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 | 0.09916633333333347 | -0.03403366666666674 | -0.03353366666666657 | -0.02843300000000004 | -0.04416700000000001 | -0.032533000000000006 |
| 0.5 | dynamic_task_soft | 1.3057666666666667 | 0.41619999999999996 | 0.5675666666666667 | 0.6240333333333333 | 0.23423333333333332 | 0.22643333333333335 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 | 0.056333666666666726 | -0.045733000000000024 | -0.10316633333333336 | -0.052266666666666683 | -0.06653366666666669 | -0.045199666666666666 |

## 5. Router Diagnostics
| missing | mode | router_oracle_match_rate | mean_w_text | mean_w_audio | mean_w_vision | mean_oracle_w_text | mean_oracle_w_audio | mean_oracle_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | oracle_text_ratio | oracle_audio_ratio | oracle_vision_ratio | entropy_router | entropy_oracle | corr_router_oracle_text | corr_router_oracle_audio | corr_router_oracle_vision | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 0.33090379008746357 | 0.3056186465012494 | 0.2891924020583561 | 0.40518895074348643 | 0.31532595050679585 | 0.3809444961021315 | 0.3037295545314677 | 0.11467444120505343 | 0.3172983479105928 | 0.5680272108843538 | 0.3216715257531584 | 0.44314868804664725 | 0.2351797862001944 | 0.9828886840128429 | 1.0356210165871709 | -0.010733764207387262 | -0.028539187669028156 | 0.023711134881332235 | 0.8680296888674534 | 0.7661712464522319 | 0.7554564197385014 |
| 0.5 | dynamic_task_soft | 0.4533527696793003 | 0.6074965890457146 | 0.22041474000528236 | 0.17208868282795184 | 0.34839999427036017 | 0.3252336891554477 | 0.3263663177616511 | 0.7240038872691934 | 0.12099125364431486 | 0.15500485908649173 | 0.4314868804664724 | 0.3255587949465501 | 0.24295432458697763 | 0.6934608378624247 | 1.0511499195591139 | 0.019367459424011923 | 0.015774847364711672 | -0.10561884308412156 | 0.6518854621264963 | 0.5913496131661874 | 0.6222457562067288 |

## 6. Conclusion
- dynamic_task_soft vs dynamic_soft: MAE better on 0/2, Corr better on 1/2.
- dynamic_task_soft vs best fixed: MAE better on 0/2.
- mean router-oracle match rate: 0.3921.
- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.
