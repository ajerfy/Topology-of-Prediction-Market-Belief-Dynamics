LOCAL TOPOLOGY VALIDATION SUMMARY

Locked candidate:
- method: top_corr
- k: 20
- locked holdout fold: 17
- graph thresholds: 0.3, 0.5, 0.7

Overall performance:
- PCA-only log loss: 0.162077
- local topology log loss: 0.138709
- log loss gain vs PCA: +0.023368
- PCA-only Brier: 0.044367
- local topology Brier: 0.036362
- Brier gain vs PCA: +0.008005
- folds improved: log loss 12/17, Brier 14/17

Locked holdout:
- log loss gain vs PCA: +0.040370
- Brier gain vs PCA: +0.004595
- clears meaningful thresholds: yes

Placebo and leakage checks:
- beats all valid placebos: yes
- invalid test-period topology explicitly used test timestamps: yes
- invalid test-period topology log loss: 0.155145
- invalid test-period topology outperforms valid locked topology: no
- timestamp-within-market and future-shift placebos: not applicable because locked topology features are fold-static market features

Ablation and interpretation:
- beats simple neighborhood controls: no
- topology still helps after controls: yes
- strongest domain gain: macro (+0.014933 log-loss gain)
- fragility flag: no

Bootstrap log-loss gain vs PCA:
- row bootstrap: mean +0.024257, 95% CI [+0.023420, +0.025144]
- market_id bootstrap: mean +0.024257, 95% CI [+0.003761, +0.045055]
- timestamp bootstrap: mean +0.024257, 95% CI [+0.023395, +0.025279]

Answers:
- Does local topology beat PCA-only on log loss? yes
- Does local topology beat PCA-only on Brier? yes
- Is the locked-holdout improvement meaningful? yes
- Does it beat all valid placebos? yes
- Is there evidence of leakage in the valid candidate? no

Paper framing decision:
- main topology-enhanced model
