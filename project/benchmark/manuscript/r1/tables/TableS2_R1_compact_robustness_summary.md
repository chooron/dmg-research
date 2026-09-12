# Table S2. Compact R1 robustness summary

| check | metric | primary_value | alternative_value | contrast | notes |
| --- | --- | --- | --- | --- | --- |
| primary_full_36 | IC ensemble median KGE | 0.621513 |  |  | Median aggregation across 36 model-level basin medians. |
| primary_full_36 | dPL ensemble median KGE | 0.609271 |  |  | Median aggregation across 36 model-level basin medians. |
| primary_full_36 | median model-level ΔKGE | -0.010415 |  |  | ΔKGE = dPL − IC. |
| exclude_simhyd | median model-level ΔKGE | -0.010415 | -0.010776 | -0.000361 | Full 36 versus exclude-simhyd 35; sensitivity only. |
| mean_vs_median | IC ensemble aggregation | 0.621513 | 0.555005 | -0.066508 | Basin median versus basin mean; model remains the ensemble unit. |
| mean_vs_median | dPL ensemble aggregation | 0.609271 | 0.535508 | -0.073763 | Basin median versus basin mean; model remains the ensemble unit. |
| mean_vs_median | median model-level ΔKGE | -0.010415 | -0.008880 | 0.001535 | Basin median versus basin mean sensitivity. |
| temporal_basin | Spearman rho of basin M_A/M_B | 0.345336 |  |  | N=531 basins; descriptive temporal persistence. |
| temporal_basin | same-sign basin fraction | 0.655367 |  |  | Neutral count and denominator are in the R1X temporal summary. |
| tolerance | comparable-or-better C_b(τ) | τ=0, 0.01, 0.02, 0.05 |  | tolerance-dependent | Full sensitivity table remains R1X internal support; do not treat τ>0 as a primary threshold. |

Recommendation: retain as internal support unless the manuscript explicitly cites these sensitivity checks.
