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
| 0.3 | dynamic_task_soft | 1.0803 | 0.5968 | 0.7332333333333333 | 0.7355999999999999 | 0.31873333333333337 | 0.29686666666666667 |
| 0.3 | text | 1.077033 | 0.616667 | 0.729167 | 0.7334 | 0.329433 | 0.297367 |
| 0.3 | vision | 0.972433 | 0.620167 | 0.764733 | 0.7673 | 0.398933 | 0.343533 |
| 0.4 | audio | 1.134667 | 0.533367 | 0.711367 | 0.710167 | 0.287667 | 0.262867 |
| 0.4 | dynamic_soft | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 |
| 0.4 | dynamic_soft_moe | 1.1596 | 0.526367 | 0.714433 | 0.7211 | 0.311 | 0.2697 |
| 0.4 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 |
| 0.4 | vision | 1.221933 | 0.524067 | 0.702233 | 0.704867 | 0.246367 | 0.232733 |
| 0.5 | audio | 1.2776 | 0.456 | 0.645833 | 0.652633 | 0.263833 | 0.243467 |
| 0.5 | dynamic_soft | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 |
| 0.5 | dynamic_soft_moe | 1.348567 | 0.449033 | 0.6123 | 0.657767 | 0.253167 | 0.241033 |
| 0.5 | text | 1.259267 | 0.464733 | 0.6626 | 0.666833 | 0.2449 | 0.2289 |
| 0.5 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 |

## 4. Delta Analysis
### dynamic_task_soft vs dynamic_soft
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE_ds | Corr_ds | Non0_acc_2_ds | Non0_F1_score_ds | Mult_acc_5_ds | Mult_acc_7_ds | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | dynamic_task_soft | 1.0803 | 0.5968 | 0.7332333333333333 | 0.7355999999999999 | 0.31873333333333337 | 0.29686666666666667 | 1.060967 | 0.6053 | 0.740867 | 0.739533 | 0.336233 | 0.3047 | 0.019333000000000045 | -0.008499999999999952 | -0.007633666666666761 | -0.003933000000000075 | -0.017499666666666636 | -0.007833333333333359 |

### dynamic_task_soft vs best fixed center
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | best_fixed_mode | MAE_fixed | Corr_fixed | Non0_acc_2_fixed | Non0_F1_score_fixed | Mult_acc_5_fixed | Mult_acc_7_fixed | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | dynamic_task_soft | 1.0803 | 0.5968 | 0.7332333333333333 | 0.7355999999999999 | 0.31873333333333337 | 0.29686666666666667 | vision | 0.972433 | 0.620167 | 0.764733 | 0.7673 | 0.398933 | 0.343533 | 0.10786700000000005 | -0.023367000000000027 | -0.031499666666666704 | -0.03170000000000006 | -0.08019966666666661 | -0.04666633333333331 |

## 5. Router Diagnostics
| missing | mode | router_oracle_match_rate | mean_w_text | mean_w_audio | mean_w_vision | mean_oracle_w_text | mean_oracle_w_audio | mean_oracle_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | oracle_text_ratio | oracle_audio_ratio | oracle_vision_ratio | entropy_router | entropy_oracle | corr_router_oracle_text | corr_router_oracle_audio | corr_router_oracle_vision | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | dynamic_task_soft | 0.3012633624878523 | 0.33748873655907846 | 0.3536362187363084 | 0.3088750431551241 | 0.33408961655794706 | 0.33353942967191036 | 0.33237095359636815 | 0.5665694849368319 | 0.37366375121477163 | 0.0597667638483965 | 0.3449951409135082 | 0.3454810495626823 | 0.30952380952380953 | 1.0753168902084778 | 1.0980306194936318 | -0.01357435340274911 | -0.026791040398308718 | -0.0227823883108043 | 0.899522120147665 | 0.7613834440203707 | 0.7619960305569252 |
| 0.2 | dynamic_task_soft | 0.30174927113702626 | 0.32227270599125196 | 0.3380449652360468 | 0.339682327892968 | 0.3331760586147985 | 0.3333967973498955 | 0.33342714170383175 | 0.27648202137998057 | 0.30709426627793973 | 0.4164237123420797 | 0.3790087463556851 | 0.3129251700680272 | 0.3080660835762877 | 1.0849467090156562 | 1.0979987766324122 | -0.0680644819641954 | -0.042224739480206754 | -0.014952762812418011 | 0.8973941345555753 | 0.7818739389087225 | 0.7687899939525596 |
| 0.3 | dynamic_task_soft | 0.30369290573372204 | 0.3215965656789908 | 0.34342972472295585 | 0.33497371014833915 | 0.33474794552961057 | 0.33134696592800356 | 0.3339050873983706 | 0.29203109815354716 | 0.3828960155490768 | 0.32507288629737613 | 0.3381924198250729 | 0.3075801749271137 | 0.3542274052478134 | 1.0834261155621878 | 1.0977017229323087 | -0.03504281188084401 | 0.01263373006622183 | -0.03936022159965055 | 0.879790345763943 | 0.7783519463853064 | 0.7878271879865576 |

## 6. Conclusion
- dynamic_task_soft vs dynamic_soft: MAE better on 0/1, Corr better on 0/1.
- dynamic_task_soft vs best fixed: MAE better on 0/1.
- mean router-oracle match rate: 0.3022.
- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.
