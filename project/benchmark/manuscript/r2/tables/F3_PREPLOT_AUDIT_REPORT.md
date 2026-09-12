F3 PRE-PLOT AUDIT = READY

Panel (a) rank atlas:
READY

Panel (b) localization:
READY-B1

Panel (c) strict rank benchmark:
READY

Panel (d) strict adjusted localization:
READY

## Scope and overall decision

No Figure 3, panel mockup, PDF, PNG, or other figure export was generated.
The four requested panel data units are now available from canonical artifacts.
Panel (b) follows Branch B1 because model-level C_eff, top-1, and cumulative
top-2 values are frozen for all 36 models. Panel (d) uses exact common
model×basin support for raw and adjusted localization; this replaces the prior
unmatched raw headline in the panel comparison.

## Audit A — model-level top-k availability

`F3_TOPK_MODEL_LEVEL_AUDIT.csv` contains one canonical row for each of 36 model
IDs, with parameter count, model-level basin-median C_eff, top-1 share, and
cumulative top-2 share. The source is
`project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_C_model_equal_claim_audit/tables/frozen_C_eff_model_summaries.csv`, computed by `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_parameter_axis_audit_20260906/agent_C_coordinate_concentration/analyze_coordinate_concentration.py`.

Coverage is 36/36 models and 531 basin rows per model. Recomputed model-equal
medians are C_eff=`0.345843772700`, top-1=`0.545618600450`, and
top-2=`0.848247607850`; these reproduce the frozen values
0.3458437727, 0.54561860045, and 0.84824760785.

MODEL-LEVEL TOP-K STATUS = READY

The prior blocked composition fingerprint and a 36×top-k display use the same
underlying frozen basin-wise squared coordinate-weight values. The 36-row
model summaries are valid for a 36-row top-k display. A model-level stacked
composition is only the arithmetic summary decomposition `top1`, `top2-top1`,
`1-top2`; it is not a new globally ranked coordinate vector and does not claim
that one coordinate is dominant across all basins.

## Audit B — exact top-k decomposition level

`F3_TOPK_DECOMPOSITION_DEFINITION.md` records the formula and hierarchy. In
brief, the pipeline computes squared normalized displacement weights within
each model×basin, ranks coordinates within that basin, calculates C_eff and
cumulative top-k shares, takes a within-model median over basins, and then
uses equal model weighting. It uses `w = DeltaTheta^2 / sum(DeltaTheta^2)`,
`N_eff = 1/sum(w^2)`, and `C_eff=N_eff/P_m`; variable parameter counts are
handled by the explicit `P_m` normalization and `min(P_m,k)` top-k limit.

## Audit C — exact matched strict23 localization

`F3_LOCALIZATION_EXACT_MATCHED_MODEL.csv` reconstructs each strict model using
only rows with adjusted status `PASS`; the raw metrics are taken from those
same model×basin rows. This gives exact common support of 23 models, with
`n_common_basins` ranging from 481 to 530.
The exact-support model-equal medians are:

- raw C_eff = `0.345216878650`; adjusted C_eff = `0.293382038900`;
- raw top-1 = `0.571176953900`; adjusted top-1 = `0.672633535000`;
- raw top-2 = `0.879999618500`; adjusted top-2 = `0.941761238600`.

Adjusted C_eff is lower for 22/23 models and adjusted
 top-1 is higher for 22/23. Top-2 is higher for
22/23. The previous raw values 0.346893 and 0.573874
were obtained by restricting all-basins raw model summaries to the strict
model set; they are not the raw metrics on the exact adjusted-valid basin
support. They are therefore not used in panel (d).

STRICT23 LOCALIZATION STATUS = READY

## Audit D/E — coordinate and model rank data

`F3_RANK_COORDINATE_AUDIT.csv` contains the 271 valid model-native coordinate
rows (`sum_m P_m = 271`) with model ID, native coordinate ID, parameter name,
R_rank, and a display-only within-model descending rank. The coordinate-level
summary is in `F3_RANK_COORDINATE_SUMMARY.csv`. Coordinate-level threshold
fractions are not the same as the frozen model-level threshold percentages.
The frozen 27.78%, 2.78%, and 2.78% values refer to 10/36, 1/36, and 1/36
model summaries reaching 0.5, 0.8, and 0.9, respectively.

`F3_RANK_MODEL_SUMMARY.csv` contains all 36 model medians. All 36 are positive,
and their model-equal median is `0.406901464622`.

## Audit F — strict23 rank benchmark

`F3_STRICT_RANK_MODEL.csv` contains the 23 paired model rows from
`project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_A_rank_self_reference/tables/model_rank_self_primary_5000.csv` computed by `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_A_rank_self_reference/rank_self_reference.py`. The strict summaries
are R_cross=`0.487560126579`, R_self=`0.797361671925`,
and median delta_R=`0.211845558402`; 23/23
delta_R values are positive.

The rank strict set and localization strict set have intersection 23 and
symmetric difference 0. This is a strict23 reference check, not a full36
inference.

## Audit G — scientific tension

`F3_RANK_LOCALIZATION_INTERPRETATION_NOTE.md` documents the coherent descriptive
interpretation: top-k concentration is basin-level, so dominant coordinate
identity can vary across basins while fixed-coordinate rank correspondence
remains moderate. Figure 3 does not test whether coordinates with larger
displacement are the same coordinates with lower rank correspondence.

## Audit H/I — rendering decisions

Panel (b) is READY-B1. Recommended structure: a separate C_eff block with a
common 0–1 axis and cue “lower → stronger concentration”, plus a displacement
composition block using largest-coordinate share, second-largest-coordinate
share (`top2-top1`), and remaining-coordinate share (`1-top2`). These labels do
not imply parameter importance.

Panel (d) is READY. Use exact-support raw versus adjusted endpoint/interval
forms preserving absolute levels: C_eff on its original axis and top-1 share on
its original axis. Do not reduce the panel to deltas and do not add top-2 to
the rendering unless the exact-support value is specifically needed; it is
available in the audit table but not required for the recommended panel.

## Model order, CI, and provenance

`F3_MODEL_ORDER_AND_COVERAGE.csv` freezes the F2 order for all 36 models and
uses one strict23 flag for both lower-row checks. No separate coverage legends
are needed. No bootstrap CI is recommended in the main F3. The archived C_eff
CI, if reported in text or SI, is a 1,000-draw paired-basin bootstrap using
the same basin indices across models (seed 20260906), followed by the
model-equal median; it is not an iid claim about 36 model structures.

## Required outputs

- `tables/F3_TOPK_MODEL_LEVEL_AUDIT.csv`
- `tables/F3_TOPK_DECOMPOSITION_DEFINITION.md`
- `tables/F3_LOCALIZATION_EXACT_MATCHED_MODEL.csv`
- `tables/F3_LOCALIZATION_EXACT_MATCHED_SUMMARY.csv`
- `tables/F3_RANK_COORDINATE_AUDIT.csv`
- `tables/F3_RANK_COORDINATE_SUMMARY.csv`
- `tables/F3_RANK_MODEL_SUMMARY.csv`
- `tables/F3_STRICT_RANK_MODEL.csv`
- `tables/F3_RANK_LOCALIZATION_INTERPRETATION_NOTE.md`
- `tables/F3_MODEL_ORDER_AND_COVERAGE.csv`

| Panel | Scientific question | Valid model set | Exact estimand | Data status | Recommended rendering |
|---|---|---|---|---|---|
| a | cross-catchment ordering | 36 | Native-coordinate tie-corrected Spearman R_rank over 531 basins; atlas plus within-model median summary | READY | Coordinate-level atlas; keep model-native coordinate IDs and distinguish coordinate rows from model summaries |
| b | displacement concentration | 36 | Basin-level squared normalized displacement weights; C_eff=N_eff/P_m and cumulative top-1/top-2 shares, median within model then equal model median | READY-B1 | Separate C_eff block and stacked top-1/second/remaining summary block |
| c | within-IC rank reference | strict23 | Per-model R_cross and IC-self R_self rank summaries with delta_R=R_self-R_cross | READY | Paired strict23 benchmark; visibly qualified as reference check, not full36 inference |
| d | coordinate-specific IC-self adjustment | strict23 | Exact-common-support per-model basin medians of raw versus E_plus-adjusted C_eff/top-1; same model×basin rows | READY | Raw versus adjusted absolute-level endpoint/interval blocks for C_eff and top-1 |
