==================================================
PCA DIAGNOSTIC SUMMARY
==================================================

Dataset:
- Panel analyzed: core_plus_satellites, because core-only has no positive labels.
- Core-only panel: 21 markets; unique YES rate 0.000. All supervised PCA logistic fits are invalid because every clean resolved core market is NO.
- Core+satellite panel: 30 markets and 58,228 future test market-time observations in the supervised prediction file.
- Unique-market class balance in core+satellites: 0.033 YES, 0.967 NO.
- Market-time row class balance in core+satellites: 0.039 YES, 0.961 NO.
- The forecasting task is dominated by class imbalance. The universe contains only one clean YES market among the selected core+satellite markets, so repeated hourly rows for that one market supply nearly all positive observations.

Forecasting results:
- Raw market-probability baseline: Brier 0.0410, log loss 0.1320, ECE-10 0.0523.
- Best overall Brier score: fixed_2 with Brier 0.0370.
- Best PCA Brier score: fixed_2 with Brier 0.0370; delta vs baseline -0.0040.
- Best overall log loss: market_probability with log loss 0.1320.
- Best PCA log loss: var_90 with log loss 0.1583; delta vs baseline +0.0263.
- PCA Brier improvements are not a log-loss victory: the best-Brier PCA model changes log loss by +0.0264 relative to baseline.
- Fold consistency: PCA Brier improvement occurs in 3 of 7 folds for the best fold-count model, while log-loss improvement occurs in 2 of 7 folds.

Calibration:
- Polymarket raw probabilities are already strong for this small universe, especially by log loss. The baseline is not perfectly calibrated, but it avoids the largest tail penalties.
- PCA logistic generally shifts probabilities toward the dominant NO class. That can reduce squared error on the many NO rows, improving Brier score.
- Log loss worsens because the same shift can make the model too confident or too low on the scarce YES rows. Log loss punishes confident errors more sharply than Brier score.
- The result is consistent with class-imbalance-driven calibration movement, not a robust proof that PCA produces better probability forecasts.

Feature importance:
- Average absolute standardized effect of p_i,t across PCA logistic models: 0.2532.
- Average absolute standardized effect of an individual PCA component: 0.6777.
- Mean ratio p_i,t effect / mean PCA-component effect: 0.44.
- The fitted logistic models lean heavily on PCA state variables in standardized-effect terms. However, because the positive class comes from only one unique YES market, this should be treated as model-exploited structure in the current sample rather than robust independent forecasting signal.

PCA information content:
- PCA components capture shared movement in the selected crypto market panel, but the component loadings are dominated by a small number of BTC/ETH price-threshold and related satellite markets.
- Dominant fixed_5 loading families by average absolute loading: crypto_policy (0.334), btc_price (0.305), microstrategy (0.258), eth_price (0.241).
- This means the PCA state is economically interpretable as broad crypto-market-level movement, but it is not yet a broad prediction-market belief space.

Interpretation:
- Why Brier improved: PCA logistic nudges forecasts toward the empirically common outcome, NO, reducing average squared error across a heavily imbalanced row set.
- Why log loss worsened: the model sacrifices probability calibration on rare YES observations and/or high-impact tail cases; log loss penalizes those misses strongly.
- Whether PCA contains predictive information beyond market probability: the coefficients and Brier shift suggest the PCA state is model-relevant, but the current evidence is weak. The benchmark is too label-imbalanced and too dependent on one YES market to support a strong conclusion about general predictive information.

Recommendation:
- B) Improve supervised forecasting setup before topology.

Justification:
- Do not proceed to persistent homology as the main comparison yet. The current supervised benchmark is not statistically sturdy enough: core-only is unusable, core+satellites has only one unique YES market, and PCA's apparent Brier gain is likely entangled with class imbalance and calibration shrinkage.
- Next data work should expand or revise the market universe to include more resolved YES and NO outcomes, then rerun this diagnostic. Topology will be much more meaningful once the supervised benchmark has credible label diversity and fold-level stability.

==================================================
