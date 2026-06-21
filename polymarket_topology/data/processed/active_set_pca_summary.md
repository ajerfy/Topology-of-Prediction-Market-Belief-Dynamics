ACTIVE-SET PCA FORECASTING SUMMARY

1. Dataset:
- number of markets: 171
- number of supervised rows: 532,979
- YES rate by unique market: 0.111
- active market count distribution: min 17, median 34.0, mean 36.8, max 68

2. Baseline:
- Brier: 0.0489
- log loss: 0.1673

3. Best active-filled PCA:
- model: var_90_standard
- components: median 33, range 23-39
- Brier: 0.0467
- log loss: 0.1800
- improves over baseline: Brier yes, log loss no

4. Best family-level PCA:
- model: fixed_2_standard
- components: median 2, range 2-2
- Brier: 0.0445
- log loss: 0.1625
- improves over baseline: Brier yes, log loss yes

5. Calibration:
- best log-loss model: family_pca / fixed_2_standard with log loss 0.1625
- class weighting helps on mean Brier across PCA configs: no
- Calibration should be judged against the decile output file; log loss remains the stricter warning signal for overconfident rare-positive errors.

6. Interpretation:
- Active-set PCA removes the structural-missingness artifact by building supervised rows only for active markets.
- Active-filled PCA treats inactive markets as zero centered deviations rather than as missing probabilities.
- Family-level PCA gives a fixed belief-state representation that is independent of exact market membership.
- Evidence for predictive information beyond market probability is present only if PCA improves both Brier and log loss; otherwise it may still be exploiting class balance or calibration shrinkage.

7. Recommendation:
- A) Proceed to persistent homology using active-set representation

Justification:
- Persistent homology should only follow if this active-set PCA benchmark is stable and fair.
- If PCA still improves only Brier but not log loss, the next step is benchmark refinement rather than topology.
