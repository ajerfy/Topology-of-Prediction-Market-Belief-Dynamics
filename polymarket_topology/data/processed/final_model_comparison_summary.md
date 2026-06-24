FINAL MODEL COMPARISON SUMMARY

Dataset and validation:
- folds: 17
- locked holdout fold: 17
- topology construction used zero test timestamps: yes
- locked topology: top_corr, k=20, thresholds=0.3, 0.5, 0.7

Overall ranking:
- logit_pca_controls_topology: log loss 0.128668, Brier 0.034443, AUC 0.958828
- logit_pca_controls: log loss 0.135540, Brier 0.036162, AUC 0.949246
- xgb_pca_controls: log loss 0.138380, Brier 0.039100, AUC 0.974841
- logit_pca_topology: log loss 0.138709, Brier 0.036362, AUC 0.947098
- xgb_pca_controls_topology: log loss 0.139002, Brier 0.039291, AUC 0.975243
- xgb_pca_topology: log loss 0.151433, Brier 0.041054, AUC 0.951218
- logit_pca: log loss 0.162077, Brier 0.044367, AUC 0.940834
- xgb_pca: log loss 0.165341, Brier 0.045627, AUC 0.934993
- market_probability: log loss 0.167262, Brier 0.048879, AUC 0.941865

Key comparisons:
- Logistic topology after controls: log-loss gain +0.006871, Brier gain +0.001719
- XGBoost topology after controls: log-loss gain -0.000622, Brier gain -0.000191
- XGBoost controls vs logistic controls: log-loss gain -0.002840
- Best model: logit_pca_controls_topology
- Best topology model: logit_pca_controls_topology
- Best non-topology model: logit_pca_controls

Statistical tests:
- Logistic topology after controls: log-loss gain +0.007024, market-clustered 95% CI [+0.000013, +0.017326]
- XGBoost topology after controls: log-loss gain -0.000129, market-clustered 95% CI [-0.001126, +0.000750]

Calibration:
- Logistic topology ECE gain vs controls: -0.003041
- Topology improves calibration: no

Robustness:
- Result dominated by macro markets: no
- Locked holdout best model: logit_pca_controls_topology

Answers:
- Is the logistic topology result paper-ready? yes
- Does topology improve over PCA + controls? yes
- Does XGBoost beat logistic regression? no
- Does topology improve XGBoost? no
- Does topology improve both log loss and Brier? yes
- Are gains statistically meaningful? yes

Final paper framing:
- Topology helps linear models but is partly absorbed by nonlinear learners.
