TDA FORECASTING SUMMARY

1. Dataset
- number of markets: 171
- number of timestamps: 14,464
- number of supervised rows: 532,979
- YES rate by unique market: 0.111

2. Baselines
- market probability Brier/log loss: 0.0489 / 0.1673
- family-level PCA Brier/log loss: 0.0445 / 0.1625

3. Best TDA model
- window size: 24h
- feature set: H0/H1 count, total persistence, max persistence, persistence entropy
- Brier: 0.0497
- log loss: 0.1930

4. Comparison
- does TDA beat market probability? no
- does TDA beat family-level PCA? no

5. Topological findings
- are H1 features common or rare? common
- 24h: H1 nontrivial rate 0.871, avg H0 persistence 3.142, avg H1 persistence 0.027
- 72h: H1 nontrivial rate 1.000, avg H0 persistence 8.960, avg H1 persistence 0.151
- 168h: H1 nontrivial rate 1.000, avg H0 persistence 20.325, avg H1 persistence 0.475
- persistence statistics are stable if their time-series plots show gradual movement rather than isolated spikes; inspect data/processed/figures/tda.
- the point cloud is topologically nontrivial only if H1 persists regularly and improves forecasting beyond PCA.

6. Interpretation
- topology adds predictive information beyond PCA: not yet
- support for the paper thesis: inconclusive with scalar persistence summaries

7. Recommendation
- B) Try richer topology constructions (persistence images, landscapes, kernels, etc.)

Justification:
- The basic scalar persistence summaries do not yet beat the PCA benchmark; stopping now would only test a weak TDA representation, not topology as a class.
- The comparison used the same active-set family-state representation and chronological folds as the successful PCA benchmark, so differences are attributable to the compression features rather than a changed data object.

MOST IMPORTANT QUESTION
- Can topological summaries outperform the strongest PCA benchmark? No, not with this first scalar-feature construction.
