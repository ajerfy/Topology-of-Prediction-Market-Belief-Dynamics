XGBOOST FORECASTING TOPOLOGY SUMMARY

Overall ranking:
- logit_pca_controls_static_topology: log loss 0.128668, Brier 0.034442, AUC 0.958828
- logit_pca_controls: log loss 0.135540, Brier 0.036162, AUC 0.949246
- logit_pca_controls_static_dynamic_topology: log loss 0.137280, Brier 0.036215, AUC 0.955087
- xgb_pca_controls: log loss 0.138198, Brier 0.039105, AUC 0.974754
- xgb_pca_controls_dynamic_topology: log loss 0.138363, Brier 0.038945, AUC 0.974835
- xgb_pca_controls_static_dynamic_topology: log loss 0.139005, Brier 0.039312, AUC 0.975640
- xgb_pca_controls_static_topology: log loss 0.139038, Brier 0.039381, AUC 0.975474
- logit_pca_controls_dynamic_topology: log loss 0.144773, Brier 0.038154, AUC 0.946480
- market_probability: log loss 0.167262, Brier 0.048879, AUC 0.941865

Key answers:
- Does dynamic topology improve XGBoost final-resolution forecasting? no
- XGBoost dynamic topology gain: log loss -0.000166, Brier +0.000160
- Does dynamic topology improve logistic final-resolution forecasting? no
- Logistic dynamic topology gain: log loss -0.009233, Brier -0.001991
- Does dynamic topology beat static topology for XGBoost? yes
- XGBoost dynamic-vs-static log-loss gain: +0.000675
- Does topology improve calibration? no
- XGBoost ECE gain: -0.002072
- Market-clustered XGBoost CI excludes zero: no
- Result dominated by macro markets: yes
- Zero future timestamp leakage diagnostics: yes
- External CPU supported: yes, via Colab/Kaggle setup instructions and fold split/combine CLI.

Final paper framing:
- Dynamic topology does not improve beyond the existing final comparison.
