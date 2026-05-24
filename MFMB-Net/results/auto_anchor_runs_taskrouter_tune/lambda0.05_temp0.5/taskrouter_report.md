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
| 0.4 | dynamic_task_soft | 1.0952333333333335 | 0.5351666666666667 | 0.7159333333333334 | 0.7181666666666667 | 0.2949333333333333 | 0.2726 |
| 0.4 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 |
| 0.4 | vision | 1.221933 | 0.524067 | 0.702233 | 0.704867 | 0.246367 | 0.232733 |
| 0.5 | audio | 1.2776 | 0.456 | 0.645833 | 0.652633 | 0.263833 | 0.243467 |
| 0.5 | dynamic_soft | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 |
| 0.5 | dynamic_soft_moe | 1.348567 | 0.449033 | 0.6123 | 0.657767 | 0.253167 | 0.241033 |
| 0.5 | dynamic_task_soft | 1.2241666666666668 | 0.44203333333333333 | 0.6839333333333334 | 0.6845 | 0.24053333333333335 | 0.2322666666666667 |
| 0.5 | text | 1.259267 | 0.464733 | 0.6626 | 0.666833 | 0.2449 | 0.2289 |
| 0.5 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 |

## 4. Delta Analysis
### dynamic_task_soft vs dynamic_soft
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE_ds | Corr_ds | Non0_acc_2_ds | Non0_F1_score_ds | Mult_acc_5_ds | Mult_acc_7_ds | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.0952333333333335 | 0.5351666666666667 | 0.7159333333333334 | 0.7181666666666667 | 0.2949333333333333 | 0.2726 | 1.092133 | 0.5524 | 0.726633 | 0.725133 | 0.320667 | 0.289133 | 0.0031003333333334826 | -0.017233333333333323 | -0.010699666666666552 | -0.006966333333333297 | -0.025733666666666655 | -0.016532999999999964 |
| 0.5 | dynamic_task_soft | 1.2241666666666668 | 0.44203333333333333 | 0.6839333333333334 | 0.6845 | 0.24053333333333335 | 0.2322666666666667 | 1.298767 | 0.378833 | 0.624 | 0.6411 | 0.215267 | 0.209933 | -0.07460033333333316 | 0.06320033333333336 | 0.059933333333333394 | 0.043399999999999994 | 0.025266333333333363 | 0.022333666666666696 |

### dynamic_task_soft vs best fixed center
| missing | mode | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | best_fixed_mode | MAE_fixed | Corr_fixed | Non0_acc_2_fixed | Non0_F1_score_fixed | Mult_acc_5_fixed | Mult_acc_7_fixed | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score | delta_Mult_acc_5 | delta_Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 1.0952333333333335 | 0.5351666666666667 | 0.7159333333333334 | 0.7181666666666667 | 0.2949333333333333 | 0.2726 | text | 1.105867 | 0.549667 | 0.705267 | 0.704533 | 0.314367 | 0.280833 | -0.01063366666666643 | -0.014500333333333337 | 0.010666333333333444 | 0.013633666666666766 | -0.019433666666666682 | -0.00823299999999999 |
| 0.5 | dynamic_task_soft | 1.2241666666666668 | 0.44203333333333333 | 0.6839333333333334 | 0.6845 | 0.24053333333333335 | 0.2322666666666667 | vision | 1.249433 | 0.461933 | 0.670733 | 0.6763 | 0.300767 | 0.271633 | -0.02526633333333317 | -0.01989966666666665 | 0.01320033333333337 | 0.008199999999999985 | -0.06023366666666666 | -0.03936633333333331 |

## 5. Router Diagnostics
| missing | mode | router_oracle_match_rate | mean_w_text | mean_w_audio | mean_w_vision | mean_oracle_w_text | mean_oracle_w_audio | mean_oracle_w_vision | selected_text_ratio | selected_audio_ratio | selected_vision_ratio | oracle_text_ratio | oracle_audio_ratio | oracle_vision_ratio | entropy_router | entropy_oracle | corr_router_oracle_text | corr_router_oracle_audio | corr_router_oracle_vision | corr_availability_text_w_text | corr_availability_audio_w_audio | corr_availability_vision_w_vision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | dynamic_task_soft | 0.36929057337220605 | 0.436402887936801 | 0.21922497482711503 | 0.3443721367419128 | 0.3349417447899012 | 0.33298375274337993 | 0.33207450248120013 | 0.43440233236151604 | 0.18221574344023325 | 0.38338192419825073 | 0.3751214771622935 | 0.272108843537415 | 0.3527696793002915 | 0.8434760659344541 | 1.0966360671600892 | 0.01982401100045765 | 0.010843232736282557 | 0.0017851494003167194 | 0.8543042791404277 | 0.7046641079666146 | 0.722122508217461 |
| 0.5 | dynamic_task_soft | 0.3032069970845481 | 0.34166696271610214 | 0.29923220960273106 | 0.35910082682677563 | 0.3372030088241756 | 0.33201674010651683 | 0.3307802516413152 | 0.3425655976676385 | 0.15792031098153547 | 0.49951409135082603 | 0.3411078717201166 | 0.36151603498542273 | 0.29737609329446063 | 1.0776795320286368 | 1.0959606078581787 | -0.011747691515230509 | -0.05139272022166779 | -0.04027763908749207 | 0.8795582522193978 | 0.7539257424306279 | 0.738902862651638 |

## 6. Conclusion
- dynamic_task_soft vs dynamic_soft: MAE better on 1/2, Corr better on 1/2.
- dynamic_task_soft vs best fixed: MAE better on 2/2.
- mean router-oracle match rate: 0.3362.
- Next step: tune task_router_lambda / center_aux_lambda / oracle temperature for robust gains at high missing rates.
