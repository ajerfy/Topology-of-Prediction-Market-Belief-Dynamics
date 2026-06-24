TOPOLOGY RESCUE SWEEP SUMMARY

H1_local_topology:
- best: pca_plus_top_corr_k20_local_graph on Y_i (log loss 0.139994, folds 17)
- delta vs best PCA comparator: -0.022082
- beats best placebo: yes

H2_graph_topology:
- best: pca_plus_graph_topology on Y_i (log loss 0.162078, folds 17)
- delta vs best PCA comparator: +0.000002
- beats best placebo: no

H3_regime_volatility:
- best: pca_plus_topology_abs_move_24h on abs_move_24h (MSE 0.000487, folds 17)
- delta vs best PCA comparator: -0.000001
- beats best placebo: not tested

H4_topology_change:
- best: pca_plus_topology_change on Y_i (log loss 0.162083, folds 17)
- delta vs best PCA comparator: +0.000007
- beats best placebo: no

H5_interactions:
- best: xgboost_interactions on Y_i (log loss 0.160297, folds 17)
- delta vs best PCA comparator: -0.001780
- beats best placebo: not tested

H6_uncertainty_signal:
- best: pca_plus_topology_error_market_squared_error on market_squared_error (MSE 0.018482, folds 17)
- delta vs best PCA comparator: +0.000066
- beats best placebo: not tested

Best primary-outcome topology result:
- H1_local_topology / pca_plus_top_corr_k20_local_graph / target Y_i / metric 0.139994

Best auxiliary topology result:
- H3_regime_volatility / pca_plus_topology_abs_move_24h / target abs_move_24h / metric 0.000487
- survives placebo checks: yes

Most promising topology role:
- B) local topology

Final recommendation:
- B) refine topology
