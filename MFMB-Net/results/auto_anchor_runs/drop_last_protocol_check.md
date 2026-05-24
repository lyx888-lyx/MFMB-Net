# Drop Last Protocol Check

Protocol A: eval/test drop_last=True

Protocol B: eval/test drop_last=False

## Raw Results

| missing | mode | eval_drop_last | test_drop_last | effective_test_samples | MAE | Corr | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | text | 0 | 0 | 686.0 | 0.8068 | 0.7933 | 0.8277 | 0.8267 | 0.4752 | 0.4213 |
| 0.0 | text | 1 | 1 | 672.0 | 0.8056 | 0.793 | 0.8245 | 0.8232 | 0.4792 | 0.4271 |
| 0.3 | text | 0 | 0 | 686.0 | 1.1056 | 0.6258 | 0.7088 | 0.7114 | 0.3411 | 0.312 |
| 0.3 | text | 1 | 1 | 672.0 | 1.0978 | 0.5801 | 0.715 | 0.7136 | 0.3527 | 0.2961 |
| 0.4 | text | 0 | 0 | 686.0 | 1.0962 | 0.5702 | 0.7149 | 0.7133 | 0.309 | 0.2857 |
| 0.4 | text | 1 | 1 | 672.0 | 1.0989 | 0.5668 | 0.7118 | 0.71 | 0.311 | 0.2887 |
| 0.5 | text | 0 | 0 | 686.0 | 1.2292 | 0.4603 | 0.6768 | 0.6782 | 0.2376 | 0.2318 |
| 0.5 | text | 1 | 1 | 672.0 | 1.2765 | 0.454 | 0.6454 | 0.6478 | 0.25 | 0.2321 |

## A vs B Delta (B - A)

| missing | dropped_test_samples | delta_MAE_B_minus_A | delta_Corr_B_minus_A | delta_Non0_acc_2_B_minus_A | delta_Non0_F1_score_B_minus_A |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 14.0 | 0.0011999999999999789 | 0.00029999999999996696 | 0.0031999999999999806 | 0.0034999999999999476 |
| 0.3 | 14.0 | 0.007799999999999807 | 0.045700000000000074 | -0.006199999999999983 | -0.0021999999999999797 |
| 0.4 | 14.0 | -0.0026999999999999247 | 0.0034000000000000696 | 0.0030999999999999917 | 0.0033000000000000806 |
| 0.5 | 14.0 | -0.0472999999999999 | 0.006299999999999972 | 0.031399999999999983 | 0.030399999999999983 |

## Notes

- `drop_last=True` drops the final incomplete batch in eval/test.
- If MOSI test size is 686 and batch size is 24, `drop_last=True` uses 672 and drops 14 samples.
- Official protocol is recommended as `eval_drop_last=0` and `test_drop_last=0`.
- If metric drift is noticeable, prior results with dropped eval/test samples should be retested.
