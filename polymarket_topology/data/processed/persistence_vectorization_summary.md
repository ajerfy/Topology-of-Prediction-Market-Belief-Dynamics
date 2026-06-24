PERSISTENCE VECTORIZATION SUMMARY

Dataset:
- markets: 171
- supervised rows: 532,979
- YES rate by unique market: 0.111

Benchmarks:
- market probability: Brier 0.0489, log loss 0.1673
- family-level PCA: Brier 0.0445, log loss 0.1625
- scalar TDA: Brier 0.0497, log loss 0.1930

Best richer topology models:
- persistence image: pi_72h_10x10_standard Brier 0.0692, log loss 0.2504
- persistence landscape: pl_24h_L3_S50_standard Brier 0.0693, log loss 0.2509
- best topology overall: pi_72h_10x10_standard Brier 0.0692, log loss 0.2504

Answers:
1. Do persistence images outperform scalar TDA? no
2. Do persistence landscapes outperform scalar TDA? no
3. Does either topology representation beat family-level PCA? no
4. If not, how close do they get? Best topology Brier gap vs PCA +0.0247, log-loss gap vs PCA +0.0880.
5. Is topology providing unique forecasting information? Not convincingly unless it improves over scalar TDA and narrows the PCA gap under the same folds.

Recommendation:
- D) Conclude PCA is the superior compression

Justification:
- Neither persistence images nor landscapes improved on scalar TDA or approached the family-level PCA benchmark in this supervised test.
- This is a representation test, not a search for a topology win; PCA remains the comparator to beat.
