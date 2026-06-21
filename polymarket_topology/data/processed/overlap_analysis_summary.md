==================================================
OVERLAP ANALYSIS SUMMARY
==================================================

Original Universe B:

- market count: 171
- missingness: 0.785
- structural missingness: 0.774 of all cells, 0.987 of missing cells
- data missingness after active forward-fill: 0.010 of all cells, 0.013 of missing cells
- raw active-period sparsity before forward-fill: 0.049 of active cells

Overlap statistics:

- mean overlap: 0.011
- median overlap: 0.000
- minimum overlap: 0.000
- maximum overlap: 0.195

Candidate universes:

| universe | market_count | yes_rate | class_entropy | panel_missingness | median_active_markets | mean_pairwise_overlap | pc_90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Universe B | 171 | 0.111 | 0.503 | 0.785 | 34.000 | 0.011 | 49.000 |
| Universe B-25 | 0 |  | 0.000 |  | 0.000 |  |  |
| Universe B-40 | 0 |  | 0.000 |  | 0.000 |  |  |
| Universe B-50 | 0 |  | 0.000 |  | 0.000 |  |  |

Interpretation:

- The 78.5% panel missingness is mostly structural.
- Active-period data gaps are small relative to structural gaps; the remaining NaNs are dominated by markets not coexisting in time.

Recommendation:

- A) Keep original Universe B

Justification:

- Selected universe: Universe B
- It keeps 171 markets, YES rate 0.111, entropy 0.503, missingness 0.785, and PC90 49.
- This choice offers the best current tradeoff between reducing construction artifacts and preserving enough markets, outcome diversity, and latent-factor structure for a PCA-vs-topology comparison.

Most important question:

Can we reduce missingness substantially while preserving enough markets, enough outcome diversity, and enough latent-factor structure to make the PCA-vs-topology comparison scientifically meaningful?

Answer: not with the requested average-overlap thresholds. Under the literal pairwise Jaccard-overlap definition, B-25/B-40/B-50 are empty, so they reduce missingness only by destroying the universe. The valid next move is to keep Universe B for now, but build future analysis on active-window-aware methods rather than treating the full rectangular panel as equally observed.

==================================================
