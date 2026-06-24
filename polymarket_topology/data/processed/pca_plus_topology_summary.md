PCA + TOPOLOGY FORECASTING SUMMARY

Research framing:
- This is not PCA vs topology.
- The benchmark is market-implied probability, then PCA enhancement, then PCA + topology enhancement.

Dataset:
- markets: 171
- supervised rows: 532,979
- YES rate by unique market: 0.111

Prior benchmarks:
- market probability: Brier 0.0489, log loss 0.1673
- family-level PCA: Brier 0.0445, log loss 0.1625
- scalar TDA alone: Brier 0.0497, log loss 0.1930

This run:
- market probability: market_probability Brier 0.0489, log loss 0.1673
- family PCA: family_pca_fixed_2_standard Brier 0.0445, log loss 0.1625
- scalar TDA: scalar_tda_24h_standard Brier 0.0497, log loss 0.1930
- PCA + scalar TDA: pca_plus_scalar_tda_24h_standard Brier 0.0445, log loss 0.1624

Incremental topology test:
- PCA + topology beats family PCA on both Brier and log loss: yes
- PCA + topology Brier delta vs prior family PCA: -0.0000
- PCA + topology log-loss delta vs prior family PCA: -0.0000

Interpretation:
- Scalar topological features add detectable but extremely small incremental predictive information beyond market probability and family-level PCA.

Recommendation:
- Treat PCA + scalar topology as a marginal enhancement, then test richer or regularized topology features.
