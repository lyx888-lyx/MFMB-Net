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
| 0.4 | dynamic_task_soft | 1.191 | 0.5222333333333333 | 0.6880333333333333 | 0.6928333333333333 | 0.26580000000000004 | 0.24539999999999998 |
| 0.4 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 |
| 0.4 | vision | 1.221933 | 0.524067 | 0.702233 | 0.704867 | 0.246367 | 0.232733 |
| 0.5 | audio | 1.2776 | 0.456 | 0.645833 | 0.652633 | 0.263833 | 0.243467 |
| 0.5 | dynamic_soft | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 |
| 0.5 | dynamic_soft_moe | 1.348567 | 0.449033 | 0.6123 | 0.657767 | 0.253167 | 0.241033 |
| 0.5 | dynamic_task_soft | 1.3242666666666667 | 0.37966666666666665 | 0.5991 | 0.6130666666666666 | 0.19096666666666665 | 0.18803333333333336 |
| 0.5 | text | 1.259267 | 0.464733 | 0.6626 | 0.666833 | 0.2449 | 0.2289 |
| 0.5 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 |

## 4. Delta Analysis
### dynamic_task_soft vs dynamic_soft
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE_ds | Corr_ds | Non0_acc_2_ds | Non0_F1_score_ds | Mult_acc_5_ds | Mult_acc_7_ds | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.191 | 0.5222333333333333 | 0.6880333333333333 | 0.6928333333333333 | 0.26580000000000004 | 0.24539999999999998 | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 | 0.09886700000000004 | -0.030166666666666675 | -0.0385996666666667 | -0.032299666666666726 | -0.054866999999999944 | -0.043732999999999994 |
| 0.5 | dynamic_task_soft | 1.3242666666666667 | 0.37966666666666665 | 0.5991 | 0.6130666666666666 | 0.19096666666666665 | 0.18803333333333336 | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 | 0.025499666666666698 | 0.000833666666666677 | -0.024900000000000033 | -0.028033333333333355 | -0.02430033333333334 | -0.02189966666666665 |

### dynamic_task_soft vs best fixed center
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | best_fixed_mode | MAE_fixed | Corr_fixed | Non0_acc_2_fixed | Non0_F1_score_fixed | Mult_acc_5_fixed | Mult_acc_7_fixed | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.191 | 0.5222333333333333 | 0.6880333333333333 | 0.6928333333333333 | 0.26580000000000004 | 0.24539999999999998 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 | 0.08513300000000013 | -0.02743366666666669 | -0.017233666666666703 | -0.011699666666666664 | -0.04856699999999997 | -0.03543300000000002 |
| 0.5 | dynamic_task_soft | 1.3242666666666667 | 0.37966666666666665 | 0.5991 | 0.6130666666666666 | 0.19096666666666665 | 0.18803333333333336 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 | 0.07483366666666669 | -0.08226633333333333 | -0.07163300000000006 | -0.06323333333333336 | -0.10980033333333336 | -0.08359966666666666 |

## 5. Router Diagnostics
| missing | mode | router_oracle_match_rate | mean_w_text | mean_w_audio | mean_w_vision | mean_oracle_w_text | mean_oracle_w_audio | mean_oracle_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | oracle_text_ratio | oracle_audio_ratio | oracle_vision_ratio | entropy_router | entropy_oracle | corr_router_oracle_text | corr_router_oracle_audio | corr_router_oracle_vision | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 0.38629737609329445 | 0.31803409595212623 | 0.4651901592057926 | 0.2167757438474033 | 0.33141111822181585 | 0.337979692918682 | 0.3306091890767202 | 0.37998056365403304 | 0.5393586005830904 | 0.08066083576287658 | 0.4480077745383868 | 0.34305150631681247 | 0.2089407191448008 | 0.9417901978620483 | 1.0942880030083464 | -0.012277638837143614 | -0.021816961703862995 | 0.026479166682176845 | 0.8734051296460441 | 0.753238283497426 | 0.7704657037213153 |
| 0.5 | dynamic_task_soft | 0.27016520894071916 | 0.4537134270597361 | 0.30828007815182806 | 0.23800649258950812 | 0.33548446775814994 | 0.33257635406230235 | 0.3319391786646681 | 0.33673469387755106 | 0.5364431486880467 | 0.1268221574344023 | 0.3819241982507289 | 0.26141885325558795 | 0.35665694849368323 | 0.6774233389446596 | 1.094470587189014 | 0.007220420927516448 | -0.05410140167225184 | -0.014198653795905474 | 0.6950576675205129 | 0.5644956871325438 | 0.58510495093757 |

## 6. Conclusion
- dynamic_task_soft vs dynamic_soft: MAE better on 0/2, Corr better on 1/2.
- dynamic_task_soft vs best fixed: MAE better on 0/2.
- mean router-oracle match rate: 0.3282.
- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.
