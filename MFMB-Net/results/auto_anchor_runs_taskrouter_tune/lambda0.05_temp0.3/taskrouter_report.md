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
| 0.4 | dynamic_task_soft | 1.1106 | 0.5445 | 0.7230666666666666 | 0.7243333333333334 | 0.28523333333333334 | 0.26776666666666665 |
| 0.4 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 |
| 0.4 | vision | 1.221933 | 0.524067 | 0.702233 | 0.704867 | 0.246367 | 0.232733 |
| 0.5 | audio | 1.2776 | 0.456 | 0.645833 | 0.652633 | 0.263833 | 0.243467 |
| 0.5 | dynamic_soft | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 |
| 0.5 | dynamic_soft_moe | 1.348567 | 0.449033 | 0.6123 | 0.657767 | 0.253167 | 0.241033 |
| 0.5 | dynamic_task_soft | 1.3127333333333333 | 0.46216666666666667 | 0.6118 | 0.6435 | 0.1968 | 0.1890333333333333 |
| 0.5 | text | 1.259267 | 0.464733 | 0.6626 | 0.666833 | 0.2449 | 0.2289 |
| 0.5 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 |

## 4. Delta Analysis
### dynamic_task_soft vs dynamic_soft
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE_ds | Corr_ds | Non0_acc_2_ds | Non0_F1_score_ds | Mult_acc_5_ds | Mult_acc_7_ds | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.1106 | 0.5445 | 0.7230666666666666 | 0.7243333333333334 | 0.28523333333333334 | 0.26776666666666665 | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 | 0.01846700000000001 | -0.007900000000000018 | -0.003566333333333338 | -0.000799666666666643 | -0.03543366666666664 | -0.02136633333333332 |
| 0.5 | dynamic_task_soft | 1.3127333333333333 | 0.46216666666666667 | 0.6118 | 0.6435 | 0.1968 | 0.1890333333333333 | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 | 0.013966333333333303 | 0.0833336666666667 | -0.012199999999999989 | 0.0023999999999999577 | -0.018466999999999983 | -0.020899666666666705 |

### dynamic_task_soft vs best fixed center
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | best_fixed_mode | MAE_fixed | Corr_fixed | Non0_acc_2_fixed | Non0_F1_score_fixed | Mult_acc_5_fixed | Mult_acc_7_fixed | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.1106 | 0.5445 | 0.7230666666666666 | 0.7243333333333334 | 0.28523333333333334 | 0.26776666666666665 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 | 0.004733000000000098 | -0.005167000000000033 | 0.017799666666666658 | 0.01980033333333342 | -0.02913366666666667 | -0.013066333333333346 |
| 0.5 | dynamic_task_soft | 1.3127333333333333 | 0.46216666666666667 | 0.6118 | 0.6435 | 0.1968 | 0.1890333333333333 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 | 0.06330033333333329 | 0.00023366666666668756 | -0.05893300000000001 | -0.03280000000000005 | -0.103967 | -0.08259966666666671 |

## 5. Router Diagnostics
| missing | mode | router_oracle_match_rate | mean_w_text | mean_w_audio | mean_w_vision | mean_oracle_w_text | mean_oracle_w_audio | mean_oracle_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | oracle_text_ratio | oracle_audio_ratio | oracle_vision_ratio | entropy_router | entropy_oracle | corr_router_oracle_text | corr_router_oracle_audio | corr_router_oracle_vision | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 0.37657920310981535 | 0.3391401797127562 | 0.3595647077113245 | 0.30129511303931794 | 0.33593533578373136 | 0.3321033176559633 | 0.33196134711059117 | 0.3396501457725947 | 0.4873663751214772 | 0.17298347910592807 | 0.39261418853255586 | 0.31341107871720114 | 0.29397473275024294 | 1.0778852275270114 | 1.094186775326376 | 0.03183498170111749 | 0.001461171310849524 | -0.01328238593039602 | 0.8898301676710068 | 0.7667515452667623 | 0.7796333634751532 |
| 0.5 | dynamic_task_soft | 0.3381924198250729 | 0.27551439350545115 | 0.38209252595264315 | 0.3423930796042476 | 0.34195883224914675 | 0.33416562559276786 | 0.32387554238616434 | 0.13702623906705538 | 0.35471331389698735 | 0.5082604470359572 | 0.3401360544217687 | 0.3513119533527697 | 0.30855199222546165 | 1.0265550418365512 | 1.0585270217596958 | -0.0026247695340020083 | 0.012791723683225337 | 0.00517178626446738 | 0.84943399490027 | 0.7591269398849082 | 0.7500871515107767 |

## 6. Conclusion
- dynamic_task_soft vs dynamic_soft: MAE better on 0/2, Corr better on 1/2.
- dynamic_task_soft vs best fixed: MAE better on 0/2.
- mean router-oracle match rate: 0.3574.
- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.
