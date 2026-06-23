# Universe Recommendation

## Executive Decision

Recommended universe: **Universe B: Macro + crypto**.

Do not implement persistent homology yet. The next topology/PCA comparison should use the recommended universe after its final panel is rebuilt and the supervised benchmark is rerun.

## Why This Universe

- Panel-ready markets: 171
- YES rate: 0.111
- NO rate: 0.889
- Class entropy: 0.503 bits
- Panel missingness: 0.785
- Median active markets per timestamp: 34.0
- Mean absolute pairwise correlation: 0.416
- PCs needed for 85% / 90% / 95% variance: 40 / 49 / 64

The best universe is the one that most improves label variation without destroying latent-factor structure. It is large enough to avoid the previous one-YES-market failure, still has correlated probability movement, and remains compressible enough for PCA and topological summaries to be meaningfully compared.

## Candidate Universe Comparison

| label | panel_market_count | yes_rate | class_entropy | panel_missingness | mean_abs_pairwise_corr | pc_90 | overall_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Universe B: Macro + crypto | 171 | 0.111 | 0.503 | 0.785 | 0.416 | 49 | 0.540 |
| Universe C: Multi-domain | 171 | 0.111 | 0.503 | 0.785 | 0.416 | 49 | 0.515 |
| Universe A: Crypto diversified | 59 | 0.068 | 0.358 | 0.612 | 0.572 | 18 | 0.332 |

## Market-Family Audit Highlights

| broad_family | broad_domain | markets | panel_ready_markets | yes_rate | class_entropy | avg_volume | forecasting_usefulness_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_microstrategy | crypto | 5 | 5 | 0.400 | 0.971 | 82170353.782 | 0.691 |
| macro_fed_rates | macro | 101 | 101 | 0.188 | 0.698 | 38689370.470 | 0.658 |
| crypto_btc | crypto | 124 | 118 | 0.169 | 0.656 | 7903999.114 | 0.497 |
| policy_other | policy_other | 16 | 0 | 0.625 | 0.954 | 17275288.497 | 0.417 |
| crypto_other | crypto | 7 | 0 | 0.429 | 0.985 | 6794262.976 | 0.411 |
| elections | elections | 854 | 0 | 0.226 | 0.771 | 17959716.795 | 0.382 |
| macro_growth_policy | macro | 3 | 0 | 0.333 | 0.918 | 9213351.671 | 0.379 |
| other | other | 349 | 0 | 0.327 | 0.911 | 15359784.705 | 0.369 |
| sports | sports | 234 | 0 | 0.047 | 0.274 | 28661654.642 | 0.311 |
| crypto_eth | crypto | 27 | 23 | 0.148 | 0.605 | 7386623.863 | 0.273 |
| crypto_policy | crypto | 2 | 2 | 0.000 | 0.000 | 14425556.083 | 0.202 |
| crypto_etf | crypto | 2 | 2 | 1.000 | 0.000 | 12924465.375 | 0.190 |
| crypto_regulation | crypto | 1 | 1 | 0.000 | 0.000 | 7603059.085 | 0.126 |
| crypto_stablecoin | crypto | 1 | 0 | 0.000 | 0.000 | 6194780.658 | 0.100 |
| weather | weather | 1 | 0 | 0.000 | 0.000 | 9466873.291 | 0.084 |

## Answer To The Research-Design Question

**If we want to test whether topological compression preserves more forecasting information than PCA, the best current market universe is the recommended universe above.**

It gives the best chance of detecting a real difference because it contains more outcome variation than the crypto-only core, retains enough cross-market dependence to make compression meaningful, and is already represented in the available price-history panel. A too-narrow crypto-only universe repeats the class-imbalance problem; a broad multi-domain universe is attractive conceptually but is not panel-ready with the current data because elections/sports coverage is sparse in the existing historical price pull.

## Recommended Next Data Step

Rebuild the supervised forecasting dataset using this universe, then rerun the PCA supervised baseline and diagnostics. Only after that benchmark is stable should persistent homology be added as the competing compression method.

## Recommended Market IDs

| market_id | broad_domain | broad_family | resolved_outcome | volume | question |
| --- | --- | --- | --- | --- | --- |
| 516861 | crypto | crypto_btc | No | 31394607.518 | Will Bitcoin reach $1,000,000 by December 31, 2025? |
| 1082748 | crypto | crypto_btc | No | 26660909.269 | Will Bitcoin reach $150,000 in January? |
| 1473040 | crypto | crypto_btc | No | 24149352.753 | Will Bitcoin reach $150,000 in March? |
| 255229 | crypto | crypto_btc | Yes | 22807235.892 | Will Bitcoin hit $100k in 2024? |
| 516863 | crypto | crypto_btc | No | 19275707.737 | Will Bitcoin reach $200,000 by December 31, 2025? |
| 516864 | crypto | crypto_btc | No | 16414060.383 | Will Bitcoin reach $150,000 by December 31, 2025? |
| 516862 | crypto | crypto_btc | No | 15857522.579 | Will Bitcoin reach $250,000 by December 31, 2025? |
| 618949 | crypto | crypto_btc | No | 15617197.158 | Will Bitcoin reach $200k in October? |
| 659233 | crypto | crypto_btc | No | 13588620.834 | Will Bitcoin reach $200,000 in November? |
| 516865 | crypto | crypto_btc | No | 13211425.235 | Will Bitcoin reach $130,000 by December 31, 2025? |
| 1082781 | crypto | crypto_btc | No | 13025594.702 | Will Bitcoin reach $100,000 in January? |
| 687200 | crypto | crypto_btc | No | 9479018.595 | Will Bitcoin reach $100,000 by December 31, 2025? |
| 255322 | crypto | crypto_btc | No | 9422713.949 | Will Bitcoin hit $250k in 2024? |
| 540408 | crypto | crypto_btc | No | 8925263.984 | Will Bitcoin dip to $90k in May? |
| 574072 | crypto | crypto_btc | No | 8804123.734 | Will Bitcoin reach $140,000 by December 31, 2025? |
| 1473066 | crypto | crypto_btc | No | 8681375.205 | Will Bitcoin reach $80,000 in March? |
| 516871 | crypto | crypto_btc | No | 8605610.759 | Will Bitcoin dip to $70,000 by December 31, 2025? |
| 666217 | crypto | crypto_btc | No | 8257060.300 | Will Bitcoin reach $120,000 by December 31, 2025? |
| 620015 | crypto | crypto_btc | No | 8066294.941 | Will Bitcoin dip to $80,000 by December 31, 2025? |
| 574073 | crypto | crypto_btc | No | 7721957.288 | Will Bitcoin reach $170,000 by December 31, 2025? |
| 1473074 | crypto | crypto_btc | No | 7334705.631 | Will Bitcoin dip to $60,000 in March? |
| 1082757 | crypto | crypto_btc | No | 7080988.614 | Will Bitcoin reach $125,000 in January? |
| 515506 | crypto | crypto_btc | No | 6976571.855 | Will Bitcoin reach $110,000 by March 31? |
| 1082777 | crypto | crypto_btc | No | 6931678.684 | Will Bitcoin reach $105,000 in January? |
| 1473075 | crypto | crypto_btc | No | 6531627.932 | Will Bitcoin dip to $55,000 in March? |
| 515502 | crypto | crypto_btc | No | 6211260.143 | Will Bitcoin reach $200,000 by March 31? |
| 687778 | crypto | crypto_btc | No | 6056873.976 | Will Bitcoin reach $110,000 by December 31, 2025? |
| 618952 | crypto | crypto_btc | No | 5978400.521 | Will Bitcoin reach $130k in October? |
| 1082772 | crypto | crypto_btc | No | 5848826.618 | Will Bitcoin reach $110,000 in January? |
| 659241 | crypto | crypto_btc | No | 5622333.157 | Will Bitcoin reach $115,000 in November? |
| 557762 | crypto | crypto_btc | No | 5372746.646 | Will Bitcoin reach $150K in July? |
| 548616 | crypto | crypto_btc | No | 5337732.561 | Will Bitcoin reach $115K in June? |
| 1473060 | crypto | crypto_btc | No | 5327306.986 | Will Bitcoin reach $90,000 in March? |
| 618950 | crypto | crypto_btc | No | 5109930.292 | Will Bitcoin reach $150k in October? |
| 532807 | crypto | crypto_btc | No | 5083115.724 | Will Bitcoin reach $100k in April? |
| 1082801 | crypto | crypto_btc | No | 5040901.869 | Will Bitcoin dip to $75,000 in January? |
| 515557 | crypto | crypto_btc | No | 5033087.296 | Will Bitcoin dip to $70,000 by March 31? |
| 516872 | crypto | crypto_btc | No | 4969076.583 | Will Bitcoin dip to $50,000 by December 31, 2025? |
| 540403 | crypto | crypto_btc | No | 4863907.520 | Will Bitcoin reach $125k in May? |
| 540404 | crypto | crypto_btc | No | 4748065.896 | Will Bitcoin reach $115k in May? |
| 1473053 | crypto | crypto_btc | No | 4676220.956 | Will Bitcoin reach $100,000 in March? |
| 558316 | crypto | crypto_btc | No | 4612360.040 | Will Bitcoin reach $200K in July? |
| 540401 | crypto | crypto_btc | No | 4586719.352 | Will Bitcoin reach $200k in May? |
| 1473063 | crypto | crypto_btc | No | 4542101.699 | Will Bitcoin reach $85,000 in March? |
| 548615 | crypto | crypto_btc | No | 4400436.535 | Will Bitcoin reach $120K in June? |
| 558317 | crypto | crypto_btc | No | 4367871.623 | Will Bitcoin reach $130K in July? |
| 1082763 | crypto | crypto_btc | No | 4349534.835 | Will Bitcoin reach $120,000 in January? |
| 1473078 | crypto | crypto_btc | No | 4295129.025 | Will Bitcoin dip to $50,000 in March? |
| 515503 | crypto | crypto_btc | No | 4273926.842 | Will Bitcoin reach $150,000 by March 31? |
| 515505 | crypto | crypto_btc | No | 4250183.440 | Will Bitcoin reach $120,000 by March 31? |
| 1082753 | crypto | crypto_btc | No | 4218206.464 | Will Bitcoin reach $130,000 in January? |
| 1082805 | crypto | crypto_btc | No | 4150722.768 | Will Bitcoin dip to $70,000 in January? |
| 618965 | crypto | crypto_btc | No | 4135952.555 | Will Bitcoin dip to $100k in October? |
| 526179 | crypto | crypto_btc | No | 4097021.611 | Will Bitcoin dip to $75,000 by March 31? |
| 516873 | crypto | crypto_btc | No | 4003570.186 | Will Bitcoin dip to $20,000 by December 31, 2025? |
| 548614 | crypto | crypto_btc | No | 3930132.868 | Will Bitcoin reach $150K in June? |
| 516878 | crypto | crypto_eth | No | 18467737.400 | Will Ethereum hit $5,000 by December 31? |
| 1082749 | crypto | crypto_eth | No | 13219840.034 | Will Ethereum reach $6,000 in January? |
| 516877 | crypto | crypto_eth | No | 9995564.196 | Will Ethereum hit $6,000 by December 31? |
| 516874 | crypto | crypto_eth | No | 9488801.523 | Will Ethereum hit $10,000 by December 31? |
| 516876 | crypto | crypto_eth | No | 7570762.534 | Will Ethereum hit $7,000 by December 31? |
| 557770 | crypto | crypto_eth | No | 7517688.703 | Will Ethereum reach $4000 in July? |
| 502128 | crypto | crypto_eth | No | 7059606.462 | Will Ethereum hit $10k in 2024? |
| 576410 | crypto | crypto_eth | No | 6416296.560 | Will Ethereum hit $17,000 by December 31? |
| 1473042 | crypto | crypto_eth | No | 6377509.959 | Will Ethereum reach $4,000 in March? |
| 516875 | crypto | crypto_eth | No | 5745770.449 | Will Ethereum hit $8,000 by December 31? |
| 574071 | crypto | crypto_eth | No | 5145125.772 | Will Ethereum hit $14,000 by December 31? |
| 618977 | crypto | crypto_eth | No | 4662515.127 | Will Ethereum reach $5000 in October? |
| 618978 | crypto | crypto_eth | No | 4087360.807 | Will Ethereum reach $4800 in October? |
| 618972 | crypto | crypto_eth | No | 3900458.869 | Will Ethereum reach $8000 in October? |
| 516926 | crypto | crypto_microstrategy | No | 17976157.530 | MicroStrategy sells any Bitcoin in 2025? |
| 692258 | crypto | crypto_microstrategy | Yes | 8418661.008 | MicroStrategy sells any Bitcoin by June 30, 2026? |
| 512250 | crypto | crypto_policy | No | 23324353.367 | Will Trump create Bitcoin reserve in first 100 days? |
| 516937 | crypto | crypto_policy | No | 5526758.798 | US national Bitcoin reserve in 2025? |
| 510854 | crypto | crypto_regulation | No | 7603059.085 | MSFT shareholders vote for Bitcoin investment?  |
| 601697 | macro | macro_fed_rates | No | 235065166.637 | Fed decreases interest rates by 50+ bps after January 2026 meeting? |
| 601700 | macro | macro_fed_rates | No | 216455743.154 | Fed increases interest rates by 25+ bps after January 2026 meeting? |
| 654415 | macro | macro_fed_rates | No | 172972056.633 | Will the Fed increase interest rates by 25+ bps after the March 2026 meeting? |
| 570360 | macro | macro_fed_rates | No | 161574382.129 | Fed decreases interest rates by 50+ bps after December 2025 meeting? |
| 504494 | macro | macro_fed_rates | No | 133955589.250 | Fed increases interest rates by 25+ bps after November 2024 meeting? |
| 570363 | macro | macro_fed_rates | No | 133173605.453 | Fed increases interest rates by 25+ bps after December 2025 meeting? |
| 572473 | macro | macro_fed_rates | No | 127684064.952 | Will Trump nominate Judy Shelton as the next Fed chair? |
| 601699 | macro | macro_fed_rates | Yes | 106767096.072 | No change in Fed interest rates after January 2026 meeting? |
| 553813 | macro | macro_fed_rates | No | 102216689.131 | Fed increases interest rates by 25+ bps after October 2025 meeting? |
| 601698 | macro | macro_fed_rates | No | 101211542.668 | Fed decreases interest rates by 25 bps after January 2026 meeting? |
| 654413 | macro | macro_fed_rates | No | 87103283.726 | Will the Fed decrease interest rates by 25 bps after the March 2026 meeting? |
| 669662 | macro | macro_fed_rates | Yes | 76909226.493 | Will there be no change in Fed interest rates after the April 2026 meeting? |
| 669663 | macro | macro_fed_rates | No | 75978206.161 | Will the Fed increase interest rates by 25+ bps after the April 2026 meeting? |
| 542539 | macro | macro_fed_rates | No | 67368762.497 | No change in Fed interest rates after September 2025 meeting? |
| 542540 | macro | macro_fed_rates | No | 67002488.504 | Fed increases interest rates by 25+ bps after September 2025 meeting? |
| 669660 | macro | macro_fed_rates | No | 66025383.922 | Will the Fed decrease interest rates by 50+ bps after the April 2026 meeting? |
| 669661 | macro | macro_fed_rates | No | 65311948.716 | Will the Fed decrease interest rates by 25 bps after the April 2026 meeting? |
| 572469 | macro | macro_fed_rates | Yes | 59907150.952 | Will Trump nominate Kevin Warsh as the next Fed chair? |
| 512325 | macro | macro_fed_rates | No | 56978076.060 | Fed increases interest rates by 25+ bps after January 2025 meeting? |
| 570361 | macro | macro_fed_rates | Yes | 54634506.683 | Fed decreases interest rates by 25 bps after December 2025 meeting? |
| 553810 | macro | macro_fed_rates | No | 52589989.132 | Fed decreases interest rates by 50+ bps after October 2025 meeting? |
| 553811 | macro | macro_fed_rates | Yes | 51626217.761 | Fed decreases interest rates by 25 bps after October 2025 meeting? |
| 542537 | macro | macro_fed_rates | No | 49890284.071 | Fed decreases interest rates by 50+ bps after September 2025 meeting? |
| 512320 | macro | macro_fed_rates | No | 48840781.094 | Fed decreases interest rates by 75+ bps after January 2025 meeting? |
| 553812 | macro | macro_fed_rates | No | 46052653.993 | No change in Fed interest rates after October 2025 meeting? |
| 570362 | macro | macro_fed_rates | No | 44523455.111 | No change in Fed interest rates after December 2025 meeting? |
| 528975 | macro | macro_fed_rates | No | 43479262.352 | Fed increases interest rates by 25+ bps after July 2025 meeting? |
| 906974 | macro | macro_fed_rates | Yes | 43061416.208 | Will there be no change in Fed interest rates after the June 2026 meeting? |
| 512321 | macro | macro_fed_rates | No | 42815445.766 | Fed decreases interest rates by 50 bps after January 2025 meeting? |
| 528972 | macro | macro_fed_rates | No | 40912544.016 | Fed decreases interest rates by 50+ bps after July 2025 meeting? |
| 572481 | macro | macro_fed_rates | No | 38740980.446 | Will Trump nominate Scott Bessent as the next Fed chair? |
| 906972 | macro | macro_fed_rates | No | 37866883.423 | Will the Fed decrease interest rates by 50+ bps after the June 2026 meeting? |
| 542538 | macro | macro_fed_rates | Yes | 36329925.673 | Fed decreases interest rates by 25 bps after September 2025 meeting? |
| 572470 | macro | macro_fed_rates | No | 36122905.931 | Will Trump nominate Kevin Hassett as the next Fed chair? |
| 572485 | macro | macro_fed_rates | No | 35603624.852 | Will Trump nominate Rick Rieder as the next Fed chair? |
| 906975 | macro | macro_fed_rates | No | 31037182.121 | Will the Fed increase interest rates by 25 bps after the June 2026 meeting? |
| 906973 | macro | macro_fed_rates | No | 30934915.761 | Will the Fed decrease interest rates by 25 bps after the June 2026 meeting? |
| 521834 | macro | macro_fed_rates | No | 29564378.909 | Fed decreases interest rates by 50+ bps after June 2025 meeting? |
| 572471 | macro | macro_fed_rates | No | 29229521.621 | Will Trump nominate Christopher Waller as the next Fed chair? |
| 528974 | macro | macro_fed_rates | Yes | 28670116.640 | No change in Fed interest rates after July 2025 meeting? |
| 521836 | macro | macro_fed_rates | Yes | 28071372.791 | No change in Fed interest rates after June 2025 meeting? |
| 572478 | macro | macro_fed_rates | No | 27875356.845 | Will Trump nominate Jerome Powell as the next Fed chair? |
| 521837 | macro | macro_fed_rates | No | 26912558.981 | Fed increases interest rates by 25+ bps after June 2025 meeting? |
| 572486 | macro | macro_fed_rates | No | 26000156.804 | Will Trump nominate Michelle Bowman as the next Fed chair? |
| 572506 | macro | macro_fed_rates | No | 24724502.621 | Will Trump nominate no one before 2027? |
| 572472 | macro | macro_fed_rates | No | 24537484.808 | Will Trump nominate Bill Pulte as the next Fed chair? |
| 519784 | macro | macro_fed_rates | No | 24000661.536 | Fed increases interest rates by 25+ bps after May 2025 meeting? |
| 572494 | macro | macro_fed_rates | No | 23577646.036 | Will Trump nominate himself as the next Fed chair? |
| 528973 | macro | macro_fed_rates | No | 23551704.396 | Fed decreases interest rates by 25 bps after July 2025 meeting? |
| 504071 | macro | macro_fed_rates | No | 23499210.730 | No change in Fed interest rates after 2024 September meeting? |
| 572480 | macro | macro_fed_rates | No | 22847634.771 | Will Trump nominate Stephen Miran as the next Fed chair? |
| 521835 | macro | macro_fed_rates | No | 22347675.301 | Fed decreases interest rates by 25 bps after June 2025 meeting? |
| 516008 | macro | macro_fed_rates | No | 22150914.603 | Fed decreases interest rates by 50+ bps after March 2025 meeting? |
| 519776 | macro | macro_fed_rates | No | 22089574.500 | Fed decreases interest rates by 50+ bps after May 2025 meeting? |
| 906976 | macro | macro_fed_rates | No | 21740489.034 | Will the Fed increase interest rates by 50+ bps after the June 2026 meeting? |
| 519777 | macro | macro_fed_rates | No | 21653201.775 | Fed decreases interest rates by 25 bps after May 2025 meeting? |
| 516011 | macro | macro_fed_rates | No | 21544318.533 | Fed increases interest rates by 25+ bps after March 2025 meeting? |
| 572489 | macro | macro_fed_rates | No | 21325692.070 | Will Trump nominate Janet Yellen as the next Fed chair? |
| 512323 | macro | macro_fed_rates | No | 21247360.000 | Fed decreases interest rates by 25 bps after January 2025 meeting? |
| 572476 | macro | macro_fed_rates | No | 21154894.069 | Will Trump nominate Arthur Laffer as the next Fed chair? |
| 512324 | macro | macro_fed_rates | Yes | 21033386.279 | No change in Fed interest rates after January 2025 meeting? |
| 572492 | macro | macro_fed_rates | No | 20907360.963 | Will Trump nominate Barron Trump as the next Fed chair? |
| 519778 | macro | macro_fed_rates | Yes | 20629387.641 | No change in Fed interest rates after May 2025 meeting? |
| 255335 | macro | macro_fed_rates | Yes | 20345317.826 | Fed rate cut by September 18? |
| 254580 | macro | macro_fed_rates | No | 20169086.259 | Will Fed cut interest rates 6+ times in 2024? |
| 507418 | macro | macro_fed_rates | No | 19150629.539 | Will the FED change rates to another level after Nov meeting? |
| 572484 | macro | macro_fed_rates | No | 17922566.565 | Will Trump nominate David Zervos as the next Fed chair? |
| 504072 | macro | macro_fed_rates | No | 17625581.465 | Fed increases interest rates by 25+ bps after September 2024 meeting? |
| 516009 | macro | macro_fed_rates | No | 17180913.164 | Fed decreases interest rates by 25 bps after March 2025 meeting? |
| 516010 | macro | macro_fed_rates | Yes | 17161804.938 | No change in Fed interest rates after March 2025 meeting? |
| 504490 | macro | macro_fed_rates | No | 15031954.754 | Fed decreases interest rates by 75+ bps after November 2024 meeting? |
| 504584 | macro | macro_fed_rates | No | 13304399.016 | Fed decreases interest rates by 75+ bps after December 2024 meeting? |
| 572479 | macro | macro_fed_rates | No | 12652617.885 | Will Trump nominate Ron Paul as the next Fed chair? |
| 520930 | macro | macro_fed_rates | No | 11760101.336 | Jerome Powell out as Fed Chair in 2025? |
| 507419 | macro | macro_fed_rates | No | 11155452.805 | Will the FED change rates to another level after December meeting? |
| 504069 | macro | macro_fed_rates | Yes | 10906333.422 | Fed decreases interest rates by 50+ bps after September 2024 meeting? |
| 572488 | macro | macro_fed_rates | No | 10813098.952 | Will Trump nominate Philip Jefferson as the next Fed chair? |
| 572477 | macro | macro_fed_rates | No | 10383485.454 | Will Trump nominate Larry Kudlow as the next Fed chair? |
| 504588 | macro | macro_fed_rates | No | 10228496.464 | Fed increases interest rates by 25+ bps after December 2024 meeting? |
| 504585 | macro | macro_fed_rates | No | 9192309.532 | Fed decreases interest rates by 50 bps after December 2024 meeting? |
| 504491 | macro | macro_fed_rates | No | 8393267.791 | Fed decreases interest rates by 50 bps after November 2024 meeting? |
| 254577 | macro | macro_fed_rates | No | 8337156.919 | Will Fed cut interest rates 3 times in 2024? |
| 504493 | macro | macro_fed_rates | No | 7995206.759 | No change in Fed interest rates after 2024 November meeting? |
| 581247 | macro | macro_fed_rates | No | 7598984.954 | Lisa Cook out as Fed Governor by September 30? |
| 504586 | macro | macro_fed_rates | Yes | 7490697.066 | Fed decreases interest rates by 25 bps after December 2024 meeting? |
| 504587 | macro | macro_fed_rates | No | 7400313.752 | No change in Fed interest rates after December 2024 meeting? |
| 254578 | macro | macro_fed_rates | Yes | 6812524.971 | Will Fed cut interest rates 4 times in 2024? |
| 254579 | macro | macro_fed_rates | No | 6683117.513 | Will Fed cut interest rates 5 times in 2024? |
| 504070 | macro | macro_fed_rates | No | 6660914.250 | Fed decreases interest rates by 25 bps after September 2024 meeting? |
| 572474 | macro | macro_fed_rates | No | 6336304.771 | Will Trump nominate David Malpass as the next Fed chair? |
| 572491 | macro | macro_fed_rates | No | 6059959.379 | Will Trump nominate Larry Lindsey as the next Fed chair? |
| 516730 | macro | macro_fed_rates | No | 6045992.738 | Will 7 Fed rate cuts happen in 2025? |
| 504492 | macro | macro_fed_rates | Yes | 5010507.074 | Fed decreases interest rates by 25 bps after November 2024 meeting? |
| 572483 | macro | macro_fed_rates | No | 4328814.775 | Will Trump nominate Marc Sumerlin as the next Fed chair? |
| 254576 | macro | macro_fed_rates | No | 3982525.829 | Will Fed cut interest rates 2 times in 2024? |
| 516729 | macro | macro_fed_rates | No | 3970078.575 | Will 8+ Fed rate cuts happen in 2025? |
