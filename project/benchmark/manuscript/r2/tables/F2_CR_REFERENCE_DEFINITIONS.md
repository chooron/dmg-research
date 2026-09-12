# F2 CR reference definitions

## Scope

The canonical contraction-robustness pipeline evaluates the same coordinate-wise diagnostic for three alternative IC reference constructions. It uses 271 parameter coordinates per model and 531 basins per model. The frozen primary restart rule is archived non-canonical IC training KGE within `0.01` of the basin best/canonical IC restart; the canonical restart itself is excluded and no canonical fallback is allowed.

## Exact CR calculation

For a model `m` and parameter coordinate `j`, define the IC interquartile range:

```text
IQR_IC(m,j) = Q75_b(IC[m,b,j]) - Q25_b(IC[m,b,j])
```

For any reference field `F[m,b,j]`, the coordinate-level contraction ratio is:

```text
CR(m,j) = IQR_b(F[m,b,j]) / IQR_IC(m,j)
```

There is no epsilon added to the denominator. If the IC IQR is zero or non-finite, that coordinate's CR is unavailable. The model-level CR is the median across the model's parameter coordinates of the coordinate-level CR values (for IC-self, the coordinate-level CR is first summarized by the median across the prescribed synthetic draws, then the model-coordinate median is taken).

### canonical

`F` is the canonical dPL parameter field, using the dPL seed-42 normalized parameter matrix. Thus `CR_canonical` is `IQR_b(dPL[m,b,j]) / IQR_b(IC[m,b,j])`, followed by the median across coordinates. It is computable from the canonical dPL and IC matrices for all 36 models and does not require an IC restart pool.

### consensus

`F` is the basin-wise median across eligible non-canonical primary IC restarts for each model, basin, and coordinate. Basins with no eligible non-canonical restart remain missing. The resulting field's basin IQR is divided by the canonical IC basin IQR, then the median across coordinates is taken. The primary model set requires at least 90% of the 531 basins to have an eligible non-canonical primary restart.

### IC-self

`F` is each fixed synthetic IC-self field from `primary_draw_indices`: one eligible non-canonical restart is selected per basin and the same restart realization supplies all coordinates, preserving the within-basin joint structure. For each of 5,000 draws, coordinate-wise basin IQR/IC-IQR produces a CR field; the median across draws is then taken per coordinate, followed by the median across coordinates for the model-level CR. The same primary eligibility and no-fallback rule applies.

## Why the current table has 23 rows

The corrected primary analysis intentionally filters the 36 upstream model diagnostics to the 23 models with primary non-canonical restart coverage at least 0.90. The 13 excluded models are: `australia, flexb, gsfb, hbv96, mopex4, mopex5, newzealand2, penman, plateau, smar, susannah2, tcm, topmodel`. Numeric rows retained in the upstream `model_summary.csv` for those models are marked `INSUFFICIENT_REFERENCE`; using them would mix incomplete primary reference fields into the formal model-equal estimand. The all-valid restart pool is a sensitivity product and is not substituted for the frozen primary pool.
