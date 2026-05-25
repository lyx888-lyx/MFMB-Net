# Reliability-gated Task-aware Dynamic Anchor Routing Report

## 1. Motivation
dynamic_soft is strong around missing=0.4 but degrades at 0.5, while tuned task-aware routing helps high-missing robustness but is less stable at lower missing rates.

## 2. Method
RTA fuses reliability-aware and task-aware routes by a learned sample-level gate: H_ours=(1-g)*H_rel+g*H_task.

## 3. Main Results
- dynamic_rta beats dynamic_soft on MAE in 2/5 missing rates.
- dynamic_rta beats dynamic_task_soft_tuned on MAE in 2/5 missing rates.
- dynamic_rta beats best fixed center on MAE in 1/5 missing rates.

## 4. Average Metrics and Rank
See rta_average_metrics.csv and rta_average_rank.csv.

## 5. Gate Diagnostics
See rta_gate_diagnostics.csv and rta_gate_weights_vs_missing.png.

## 6. Comparison with MoE
dynamic_soft_moe remains less stable in this protocol; RTA is a more reliable extension line.

## 7. Conclusion
RTA is promising when it can combine the low/mid-missing strength of reliability routing and high-missing robustness of task-aware routing. Final adoption depends on whether average MAE/rank consistently improve against both dynamic_soft and best fixed centers.

Best MAE method by missing:
- missing=0.1: vision
- missing=0.2: audio
- missing=0.3: vision
- missing=0.4: dynamic_task_soft_tuned
- missing=0.5: dynamic_rta