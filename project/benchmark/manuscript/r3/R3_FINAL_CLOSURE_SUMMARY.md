# R3 Final Closure Summary

## Decision

> **R3 DATA CLOSURE COMPLETE — START FIGURE DESIGN**

No training, calibration, dPL seed generation, threshold adjustment, or new paper story was added.

## Required answers

1. **Are the 902 sets identical?** Yes: `S_stable` and `S_absrho20` have intersection 902 and symmetric difference 0; Jaccard `1.000000000000`.
2. **Can F5 use `5420 -> 902 -> 712 -> 692`?** Yes. The 902 cells are one identical set under both rules; 712 are strong in both IC and dPL, 692 of those retain sign, and 20 are sign-flipped.
3. **Bidirectional conditioning:** `P(dPL strong | IC strong) = 712/902 = 0.789356984479`; `P(IC strong | dPL strong) = 712/2163 = 0.329172445677`. This is association asymmetry, not dPL validity or effectiveness.
4. **Was the IC-self profile benchmark constructed?** Yes, as a within-IC calibration reference using canonical best versus an independent performance-comparable alternative restart.
5. **Models/basins:** R results use 35 all-model-available models and 23 R2-strict models; A results use 34 and 22, respectively. Basin support is model-specific, with at least 400 of 531 for included models; exact values are in the model CSV.
6. **`R_paired_IC_self`:** see `R3_F6_IC_SELF_REFERENCE_SUMMARY.csv` and the within-IC report; values are computed on matched `B_m^self`.
7. **Matched-support `R_paired_cross`:** reported beside the self value in the same summary.
8. **`A_diag_IC_self`:** reported for the A-eligible model sets in the same summary; `collie1` is excluded because it has no off-diagonal contrast.
9. **Matched-support `A_diag_cross`:** reported beside the self value in the same summary.
10. **Differences/model consistency:** paired self-minus-cross differences, model-bootstrap intervals, positive-model counts, and per-model rows are exported.
11. **Blockers:** no numerical or support blocker remains for F5/F6. Provenance is explicit: archived ten-start payloads, exact canonical-best matching, frozen 0.01 rule, model-specific common basin support, and frozen 20-D representation.
12. **Figure readiness:** yes; formally proceed to R3 figure design.

## Required files

- `R3_CELL_SET_IDENTITY_AUDIT.md`
- `R3_IC_SELF_PROFILE_REFERENCE.md`
- `R3_FINAL_CLOSURE_SUMMARY.md`
- `tables/R3_CELL_SET_IDENTITY_DIFF.csv`
- `tables/R3_IC_SELF_RESTART_SUPPORT.csv`
- `tables/R3_F5_NESTED_LEDGER.csv`
- `tables/R3_F5_BIDIRECTIONAL_CONDITIONING.csv`
- `tables/R3_F6_IC_SELF_REFERENCE_MODEL.csv`
- `tables/R3_F6_IC_SELF_REFERENCE_SUMMARY.csv`
- `tables/R3_F6_CROSS_MATCHED_REFERENCE_MODEL.csv`
- `tables/R3_IC_SELF_PROFILE_VALUES_LONG.csv` and `R3_IC_SELF_CORRESPONDENCE_LONG.csv`
- `cache/R3_IC_SELF_PROFILE_REFERENCE/*.npz`

The within-IC reference is a calibration-scale comparator only. It must not be labeled an upper bound, ceiling, theoretical maximum, proof of parameter identity, or causal mechanism.
