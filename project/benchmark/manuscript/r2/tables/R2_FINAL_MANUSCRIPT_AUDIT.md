R2 FINAL AUDIT = BLOCKED

## HISTORICAL / SUPERSEDED NOTICE

> This report records the pre-resolution audit and is retained for provenance. Its BLOCKED verdict is superseded by `tables/R2_BLOCKER_RESOLUTION_REPORT.md`; use the blocker-resolution report as the active manuscript-readiness gate.

F2 numerical consistency:
PASS

- Current frozen values agree across `cache/separation/summary.csv`, `F2_ICSELF_AUDIT_REPORT.md`, `F2_MODEL_EXCESS_RECHECK.csv`, and the F2 build note: D_cross model-equal median = 0.3843747256; D_self model-equal median = 0.0734436079; paired model-level excess = 0.2187745308; 95% CI = [0.2091360196, 0.2281453316]; 36/36 model-level excesses are positive.
- The paired estimand is `median_m{median_b[D_cross-D_self]}` = 0.2187745308. The difference of marginal medians is 0.2530788322. The artifacts explicitly distinguish these quantities; they are not interchangeable.
- The strict common-subset CR values agree: canonical = 0.6140173622, consensus = 0.9794559561, IC-self = 1.0043244360, all on the identical 23/36 subset. The current wording correctly describes 1.004 as near unity/comparable spread, not meaningful expansion.
- The repository validator and master checksum check pass.

F3 numerical consistency:
PASS

- Full36 rank coverage is 36 models and 271 coordinates. Current values are model-equal median R_rank = 0.406901464622; coordinate-level fractions R_rank < 0 = 0.0442804428044, R_rank >= 0.5 = 0.328413284133, R_rank >= 0.8 = 0.0369003690037, and R_rank >= 0.9 = 0.0110701107011. These are coordinate-level fractions, not model-level threshold proportions.
- The localization hierarchy is explicit and consistent: within model x basin, square normalized coordinate displacement weights; compute C_eff/top-k; take basin medians within model; then take the equal-model median. Current all36 localization values are C_eff = 0.3458437727, top-1 = 0.54561860045, and top-2 = 0.84824760785.
- The strict rank and localization subsets are exactly identical: 23/36 models. Strict rank values are R_cross = 0.487560126579, R_self = 0.797361671925, median DeltaR = 0.211845558402, with 23/23 DeltaR > 0.
- Exact-matched strict localization values are C_eff 0.34521687865 -> 0.2933820389, top-1 0.5711769539 -> 0.6726335350, and top-2 0.8799996185 -> 0.9417612386. The expected direction occurs in 22/23 models for each metric. The one unchanged model is collie1.
- The current F3 script and `F3_PREPLOT_AUDIT_REPORT.md` use the exact-common-support values, not the earlier all-basins raw values.

F4 exemplar consistency:
FAIL

- Selection coverage passes: the complete audit contains 271/271 coordinates across 36/36 models, and the plotting set contains the requested 12 model-coordinate pairs. The plotted M and R values are read from `tables/F4_ALL_COORDINATE_DIAGNOSTICS.csv`; the normalized IC/dPL arrays are used only for the plotted basin points. No alternate aggregation is performed in the plotting script.
- The following 11 pairs agree with the canonical audit values at the requested precision: alpine1:Smax, gr4j:x1, newzealand1:s1max, mopex2:s2max, vic:ishift, mopex3:s3max, flexi:imax, simhyd:smsc, hymod:smax, wetland:swmax, and ihacres:d.
- Blocking mismatch: the canonical rows in `F4_ALL_COORDINATE_DIAGNOSTICS.csv`, `F4_CANDIDATE_POOL.csv`, and `F4_CANDIDATE_RANKINGS.csv` all give `modhydrolog:k3` M = 0.388019247502 and R = 0.0162892039821. The plotting badge therefore displays M = 0.388 and R = 0.016. The requested audit values M = 0.3879 and R = 0.0164 do not occur in the current canonical source. This is not an alternative plotting aggregation: the plotting input and candidate audit agree with each other, so the expected value or a prior source artifact is stale and must be reconciled before writing.

Cross-figure scientific logic:
PASS (subject to the artifact synchronization blocker below)

- Current roles are coherent: F2 = displacement magnitude/reference context; F3 = coordinate organization and strict benchmark; F4 = concrete coordinate-level realizations.
- F2 does not claim localization, F3 does not claim parameter mechanisms, and F4 does not claim that 12 exemplars establish full36 prevalence.
- The interpretation boundary is supported: basin-wise displacement is concentrated in a small number of coordinates while fixed-coordinate cross-catchment rank preservation is only moderate. Figure 3 does not test whether the coordinates with the largest displacement are the same coordinates with the largest rank loss.
- F4 semantics pass: x = normalized IC realization, y = normalized dPL realization, one point = one basin, dashed line = equality, top histogram = IC marginal, right histogram = dPL marginal.
- Boundary values are genuine bound-normalized realizations rather than plot clipping. Among the displayed panels, visible IC upper-bound accumulation occurs for newzealand1:s1max (12.1%, 64/531), flexi:imax (16.8%, 89/531), and wetland:swmax (14.5%, 77/531); no displayed dPL coordinate has comparable bound accumulation. Recommended caption sentence: “Because coordinates are normalized to their physical bounds, values at 0 and 1 denote realizations at the lower and upper parameter bounds, respectively; visible boundary accumulation is retained as data, not clipping.”

Terminology / interpretation boundary:
FAIL

- The current scientific notes correctly prohibit importance, sensitivity, physical dominance, causal attribution, and functional-role claims, and the F3 interpretation note explicitly preserves the rank/localization boundary.
- However, the current F2 build note and Figure 2 plotting script retain a `collie1†` annotation/open-square distinction because it is labeled `MIXED` in the restart-provenance audit. That marker is not needed for the frozen F2 estimands and is not explained as a one-parameter structural exception. It risks implying a data or restart defect. Remove it, or redefine it explicitly as a structural one-parameter note before writing.
- Exact forbidden manuscript phrases such as “key parameters,” “important parameters,” “physical importance,” “true parameter values,” “dPL identifies better parameters,” and “IC parameters are wrong” were not found. The remaining audit-only phrase “parameter sensitivity” appears in a caveat that explicitly says no sensitivity claim is made; manuscript prose should nevertheless use “parameter realization,” “high-displacement coordinate,” “coordinate localization,” “rank preservation,” and “rank reorganization.”

collie1 handling:
FAIL

- The data support one calibrated parameter: `collie1` has only `Smax` (`parameter_count = 1`) in `cache/inputs/model_parameter_alignment.csv`. Its F3 localization values are algebraically invariant: C_eff = top-1 = top-2 = 1, and its exact-matched raw-to-adjusted changes are all zero.
- The restart audit does not support calling collie1 corrupted or a cache/restart failure: it found the expected ten restart slots and no duplicate issue at the checkpoint/file/shape/key level; CR availability is complete and strict eligibility is true. The `MIXED` label is a provenance classification, not evidence of corruption.
- Plain-language manuscript recommendation: “collie1 is a one-parameter model; its exact localization invariance is structural, so it is retained in the complete summaries but should not be used to characterize multi-parameter localization.” For the direction count, use “22/22 multi-parameter eligible models” in the main text, with “22/23 including collie1” if the full strict denominator is also reported.
- The current dagger/open-square marker is therefore not manuscript-ready until removed or redefined as the structural one-parameter note.

Notation mapping for the 12 F4 panels:

| model_id | raw_parameter_name | manuscript_display_symbol |
|---|---|---|
| alpine1 | Smax | S_max |
| gr4j | x1 | x_1 |
| newzealand1 | s1max | s_{1,max} |
| mopex2 | s2max | s_{2,max} |
| vic | ishift | i_{shift} |
| mopex3 | s3max | s_{3,max} |
| flexi | imax | i_max |
| modhydrolog | k3 | k_3 |
| simhyd | smsc | s_{msc} |
| hymod | smax | s_max |
| wetland | swmax | s_{w,max} |
| ihacres | d | d |

Remaining blockers:

1. Reconcile `modhydrolog:k3` F4 M/R: current canonical/plotting values are 0.388019247502 and 0.0162892039821, while the requested audit values are 0.3879 and 0.0164. Do not write the caption until the authoritative value is identified.
2. Remove or scientifically redefine the current `collie1†`/open-square marker; describe collie1 as a one-parameter structural case, not a corrupted or failed restart/cache case.
3. Synchronize stale figure-audit artifacts with the current exact-common-support F3 definition. `R2_FIGURE_DATA_AUDIT.md` still says Figure 3a is BLOCKED, Figure 3c is PARTIAL, and overall plotting is NOT READY; `R2_FIGURE_DATA_MANIFEST.csv` retains those statuses; and `R2_HEADLINE_REPRO_CHECK.csv` retains two FAIL rows comparing the superseded all-basins raw values 0.346893/0.573874 against the current exact-support values 0.34521687865/0.5711769539. The current F3 script and preplot audit use the exact-support branch, so this requires artifact relabeling/synchronization, not a new experiment.

Validation performed:

- `cd project/benchmark/manuscript/r2 && ../../../../.venv/bin/python scripts/99_validate_frozen_results.py` -> `R2_VALIDATION_PASS`.
- `cd project/benchmark/manuscript/r2 && sha256sum -c cache/final/MASTER_CHECKSUMS.sha256` -> all listed checks `OK`.
- Read-only table checks confirmed F2, F3, strict-subset, collie1, F4 coverage, F4 values, and F4 bound-occupancy claims above.
- No source products, training, calibration, simulation, or new experiment was run.

Review:

- An independent read-only reviewer was attempted but timed out; no reviewer result was used as evidence.
