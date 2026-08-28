# Table S3 — Controlled-recovery distributions and generating-field robustness

## Panel A — Denominator, recovery-fraction, and paired-distribution properties

Overall rows report the test-period distribution across 531 basins (or the denominator-valid subset where applicable). Stratum rows report denominator validity for S1–S5. `share_of_all_invalid_basins_pct` has all invalid basins as its denominator; `invalid_rate_within_stratum_pct` has the stratum total as its denominator. Recovery fractions are unclipped, so values below 0 or above 1 are retained.

| section | paradigm | period | snow_stratum | metric | N_total | N_invalid | N_valid | median | Q25 | Q75 | P05 | P95 | P_below_0 | P_above_1 | P_delta_F_gt_0 | P_F_TGD_lt_F_close | invalid_rate_within_stratum_pct | share_of_all_invalid_basins_pct | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | IC | test | All | D | 531 | 104 | 427 | 0.1555 | 0.0428 | 0.4232 | 0.0072 | 1.0771 | nan | nan | nan | nan | nan | nan | recovery_denominator_tail_audit.csv |
| overall | IC | test | All | F_close | 531 | 104 | 427 | 0.1015 | -0.0476 | 0.176 | -1.0436 | 0.3795 | 0.3162 | 0.0047 | nan | nan | nan | nan | recovery_denominator_tail_audit.csv |
| overall | IC | test | All | F_TGD | 531 | 104 | 427 | 0.5456 | 0.3708 | 0.7171 | -0.3883 | 0.9117 | 0.0843 | 0.0398 | nan | nan | nan | nan | recovery_denominator_tail_audit.csv |
| overall | IC | test | All | Delta F | 531 | 104 | 427 | 0.46 | nan | nan | nan | nan | nan | nan | 0.9157 | 0.0843 | nan | nan | recovery_denominator_tail_audit.csv |
| overall | dPL | test | All | D | 531 | 71 | 460 | 0.131 | 0.0365 | 0.4005 | 0.0053 | 1.0723 | nan | nan | nan | nan | nan | nan | recovery_denominator_tail_audit.csv |
| overall | dPL | test | All | F_close | 531 | 71 | 460 | 0.1019 | 0.0002 | 0.1701 | -0.6274 | 0.3651 | 0.25 | 0.0022 | nan | nan | nan | nan | recovery_denominator_tail_audit.csv |
| overall | dPL | test | All | F_TGD | 531 | 71 | 460 | 0.524 | 0.3608 | 0.6615 | -0.0294 | 0.8693 | 0.0587 | 0.0065 | nan | nan | nan | nan | recovery_denominator_tail_audit.csv |
| overall | dPL | test | All | Delta F | 531 | 71 | 460 | 0.4432 | nan | nan | nan | nan | nan | nan | 0.9283 | 0.0717 | nan | nan | recovery_denominator_tail_audit.csv |
| stratum | IC | test | S1 | D validity | 165 | 95 | 70 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 57.58 | 91.35 | invalid_denominator_strata_breakdown.csv |
| stratum | IC | test | S2 | D validity | 156 | 8 | 148 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 5.13 | 7.69 | invalid_denominator_strata_breakdown.csv |
| stratum | IC | test | S3 | D validity | 121 | 1 | 120 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.83 | 0.96 | invalid_denominator_strata_breakdown.csv |
| stratum | IC | test | S4 | D validity | 34 | 0 | 34 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.0 | 0.0 | invalid_denominator_strata_breakdown.csv |
| stratum | IC | test | S5 | D validity | 55 | 0 | 55 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.0 | 0.0 | invalid_denominator_strata_breakdown.csv |
| stratum | dPL | test | S1 | D validity | 165 | 65 | 100 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 39.39 | 91.55 | invalid_denominator_strata_breakdown.csv |
| stratum | dPL | test | S2 | D validity | 156 | 5 | 151 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 3.21 | 7.04 | invalid_denominator_strata_breakdown.csv |
| stratum | dPL | test | S3 | D validity | 121 | 1 | 120 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.83 | 1.41 | invalid_denominator_strata_breakdown.csv |
| stratum | dPL | test | S4 | D validity | 34 | 0 | 34 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.0 | 0.0 | invalid_denominator_strata_breakdown.csv |
| stratum | dPL | test | S5 | D validity | 55 | 0 | 55 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.0 | 0.0 | invalid_denominator_strata_breakdown.csv |

## Panel B — Generating-field construction sensitivity

The canonical row uses the PCA/SVD-ridge field. The alternative row uses the direct basin-wise calibrated CN-IC parameter field. The latter is a generating-field construction sensitivity, not a real-catchment truth validation.

| generating_field | paradigm | period | G_Base median | G_TGD median | F_close median | F_TGD* median | Delta F median | N_total | N_valid | P(Delta F > 0) | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical PCA/SVD-ridge field | IC | test | 0.0026088991818091 | 0.0385621948047482 | 0.1014937993294627 | 0.5456445335373832 | 0.46 | 531 | 427 | 0.9157 | reviewer2_robustness/summaries/canonical_registry.csv |
| direct basin-wise calibrated CN-IC parameter field | IC | test | -0.01 | 0.0005 | -0.1033 | 0.0072 | 0.1953 | 531 | 522 | 0.7241 | reviewer2_robustness/alt_generating_field/alt_generating_field_summary.json |
| canonical PCA/SVD-ridge field | dPL | test | 0.007324709902259 | 0.0359718301931317 | 0.1008926437083189 | 0.5226589950851255 | 0.4432 | 531 | 460 | 0.9283 | reviewer2_robustness/summaries/canonical_registry.csv |
| direct basin-wise calibrated CN-IC parameter field | dPL | test | -0.017 | 0.194 | -0.0531 | 0.584 | 0.7012 | 531 | 123 | 0.9106 | reviewer2_robustness/alt_generating_field/alt_generating_field_summary.json |
