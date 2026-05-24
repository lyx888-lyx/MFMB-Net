# Router Match-rate Analysis
1. Highest match-rate appears at lambda=0.10, temp=0.3, missing=0.5, match=0.4534.
2. This is not the same as global best task-metric setting, so match-rate and downstream quality are correlated but not equivalent.
3. Match-rate range: 0.2702~0.4534; random baseline is ~0.3333.
4. Above-random points: 12/18.
5. Interpretation: oracle supervision helps, but oracle quality/temperature calibration should be improved for better performance alignment.