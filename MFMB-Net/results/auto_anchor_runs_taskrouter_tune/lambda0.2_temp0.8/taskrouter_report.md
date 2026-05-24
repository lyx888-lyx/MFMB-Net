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
| 0.4 | dynamic_task_soft | 1.1570666666666667 | 0.5432 | 0.7088666666666666 | 0.7149666666666666 | 0.31486666666666663 | 0.2828 |
| 0.4 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 |
| 0.4 | vision | 1.221933 | 0.524067 | 0.702233 | 0.704867 | 0.246367 | 0.232733 |
| 0.5 | audio | 1.2776 | 0.456 | 0.645833 | 0.652633 | 0.263833 | 0.243467 |
| 0.5 | dynamic_soft | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 |
| 0.5 | dynamic_soft_moe | 1.348567 | 0.449033 | 0.6123 | 0.657767 | 0.253167 | 0.241033 |
| 0.5 | dynamic_task_soft | 1.3198666666666667 | 0.39083333333333337 | 0.5747 | 0.6283 | 0.22303333333333333 | 0.21673333333333333 |
| 0.5 | text | 1.259267 | 0.464733 | 0.6626 | 0.666833 | 0.2449 | 0.2289 |
| 0.5 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 |

## 4. Delta Analysis
### dynamic_task_soft vs dynamic_soft
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE_ds | Corr_ds | Non0_acc_2_ds | Non0_F1_score_ds | Mult_acc_5_ds | Mult_acc_7_ds | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.1570666666666667 | 0.5432 | 0.7088666666666666 | 0.7149666666666666 | 0.31486666666666663 | 0.2828 | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 | 0.06493366666666667 | -0.009199999999999986 | -0.01776633333333333 | -0.010166333333333388 | -0.005800333333333352 | -0.0063329999999999775 |
| 0.5 | dynamic_task_soft | 1.3198666666666667 | 0.39083333333333337 | 0.5747 | 0.6283 | 0.22303333333333333 | 0.21673333333333333 | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 | 0.02109966666666674 | 0.01200033333333339 | -0.04930000000000001 | -0.012800000000000034 | 0.007766333333333347 | 0.006800333333333325 |

### dynamic_task_soft vs best fixed center
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | best_fixed_mode | MAE_fixed | Corr_fixed | Non0_acc_2_fixed | Non0_F1_score_fixed | Mult_acc_5_fixed | Mult_acc_7_fixed | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.1570666666666667 | 0.5432 | 0.7088666666666666 | 0.7149666666666666 | 0.31486666666666663 | 0.2828 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 | 0.051199666666666754 | -0.0064670000000000005 | 0.0035996666666666677 | 0.010433666666666674 | 0.0004996666666666205 | 0.0019669999999999965 |
| 0.5 | dynamic_task_soft | 1.3198666666666667 | 0.39083333333333337 | 0.5747 | 0.6283 | 0.22303333333333333 | 0.21673333333333333 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 | 0.07043366666666673 | -0.07109966666666662 | -0.09603300000000004 | -0.04800000000000004 | -0.07773366666666667 | -0.05489966666666668 |

## 5. Router Diagnostics
| missing | mode | router_oracle_match_rate | mean_w_text | mean_w_audio | mean_w_vision | mean_oracle_w_text | mean_oracle_w_audio | mean_oracle_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | oracle_text_ratio | oracle_audio_ratio | oracle_vision_ratio | entropy_router | entropy_oracle | corr_router_oracle_text | corr_router_oracle_audio | corr_router_oracle_vision | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 0.3177842565597668 | 0.3206064723924425 | 0.3358729190986867 | 0.34352060781739313 | 0.3329633016309423 | 0.3331028045208508 | 0.3339338937613197 | 0.2891156462585034 | 0.3172983479105928 | 0.3935860058309038 | 0.3012633624878523 | 0.31875607385811466 | 0.37998056365403304 | 1.0818032766355812 | 1.0982987645998874 | -0.0562949775165593 | -0.06618055258903922 | -0.03514681168506314 | 0.8905142331464423 | 0.7767839427076791 | 0.7870713044895412 |
| 0.5 | dynamic_task_soft | 0.3532555879494655 | 0.5689060354589026 | 0.23487114484354085 | 0.19622283034741297 | 0.3377031835033201 | 0.3266088824187006 | 0.3356879351713104 | 0.6224489795918368 | 0.31341107871720114 | 0.0641399416909621 | 0.30369290573372204 | 0.3143828960155491 | 0.3819241982507289 | 0.7191728411153658 | 1.0897416698331799 | 0.04656242664275694 | 0.055590710584686766 | -0.06820787327030245 | 0.6537323520777943 | 0.6037187332442278 | 0.6357211385095386 |

## 6. Conclusion
- dynamic_task_soft vs dynamic_soft: MAE better on 0/2, Corr better on 1/2.
- dynamic_task_soft vs best fixed: MAE better on 0/2.
- mean router-oracle match rate: 0.3355.
- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.
