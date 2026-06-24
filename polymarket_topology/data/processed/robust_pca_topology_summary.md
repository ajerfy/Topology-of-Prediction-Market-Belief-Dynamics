ROBUST PCA + TOPOLOGY BENCHMARK SUMMARY

Dataset:
- markets: 171
- supervised rows: 532,979
- YES rate by unique market: 0.111
- timestamp range: 2024-08-06 17:00:00+00:00 to 2026-04-01 08:00:00+00:00

Locked holdout results:
- market probability: market_probability Brier 0.0464, log loss 0.1531, ECE 0.0224
- PCA-only: pca_only_fixed_5_C0.01 Brier 0.0467, log loss 0.1699, ECE 0.0114
- TDA-only: tda_only_168h_h0_h1_C0.01 Brier 0.0687, log loss 0.2432, ECE 0.0554
- PCA+TDA: pca_plus_tda_var_85_24h_h0_h1_C1 Brier 0.0467, log loss 0.1710, ECE 0.0191

Locked configs selected before holdout:
- PCA-only: pca_only_fixed_5_C0.01
- TDA-only: tda_only_168h_h0_h1_C0.01
- PCA+TDA: pca_plus_tda_var_85_24h_h0_h1_C1

Success criteria:
- Does PCA+TDA beat PCA-only on locked holdout Brier? no (+0.000030)
- Does PCA+TDA beat PCA-only on locked holdout log loss? no (+0.001179)
- Is the gain larger than 0.0005? no
- Is the gain consistent across design folds? no
- Does real TDA beat shuffled/future-shift placebo? no
- Are H1 features adding beyond H0? no
- Is topology improving calibration? mostly shifts probabilities without clear calibration improvement (ECE delta +0.007634)

Paired holdout loss test:
- rows: 120760
- Brier delta CI: [-0.000031, +0.000087]
- Log-loss delta CI: [+0.000859, +0.001496]

Design-fold selected models:
- PCA-only: [('pca_only_fixed_5_C0.01', 5), ('pca_only_fixed_2_C0.01', 4), ('pca_only_var_85_C1', 3)]
- TDA-only: [('tda_only_168h_h0_h1_C0.01', 9), ('tda_only_24h_h0_h1_C0.01', 4), ('tda_only_24h_h0_h1_C0.1', 1)]
- PCA+TDA: [('pca_plus_tda_var_85_24h_h0_h1_C1', 3), ('pca_plus_tda_fixed_5_24h_h0_h1_C0.01', 3), ('pca_plus_tda_fixed_2_24h_h0_h1_C0.01', 3)]

Recommendation:
- D) Stop topology development and focus paper on PCA
