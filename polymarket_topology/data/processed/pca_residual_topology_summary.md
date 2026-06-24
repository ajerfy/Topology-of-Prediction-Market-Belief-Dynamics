PCA RESIDUAL TOPOLOGY SUMMARY

Dataset:
- markets: 171
- supervised rows: 532,979
- YES rate by unique market: 0.111

Model comparison:
- PCA-only: pca_only_fixed_5_C0.01 Brier 0.0444, log loss 0.1621, ECE 0.0314
- best residual PH: pca_residual_image_ph_72h Brier 0.0444, log loss 0.1620, ECE 0.0298
- best placebo log loss: 0.1589

Answers:
- Does residual PH improve over PCA-only? yes (Brier delta -0.000004, log-loss delta -0.000079)
- Is the improvement larger than 0.0005? no
- Is improvement consistent across folds? no
- Are residual topological features nontrivial? yes (mean H1 nontrivial rate 0.954, mean H1 total persistence 0.1433)
- Is any improvement robust to placebo checks? no

Recommendation:
- Abandon this topology path for the current paper and focus on PCA/market-implied forecasts.
