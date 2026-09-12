# collie1 structural audit

## Structural facts

Canonical metadata in `cache/inputs/model_parameter_alignment.csv` gives:

```text
model_id       collie1
parameter     Smax
parameter_count P = 1
strict23       True
IC generation 300
dPL seed       42
```

The F2 restart audit reports the expected ten restart slots, ten found slots, and no checkpoint/file/shape/key duplicate issue. `F2_CR_AVAILABILITY_MATRIX.csv` reports canonical, consensus, and IC-self availability as `AVAILABLE` with strict eligibility `True`. The `MIXED` label in `F2_REFERENCE_PROVENANCE.csv` is a provenance classification only; it is not evidence of corrupted data, a cache failure, or a restart failure.

## F3 localization consequences

The one-parameter geometry forces the localization metrics to be invariant:

```text
C_eff       = 1
Top-1       = 1
Top-2       = 1
```

These values are present in `tables/F3_TOPK_MODEL_LEVEL_AUDIT.csv`. In `tables/F3_LOCALIZATION_EXACT_MATCHED_MODEL.csv`, collie1 has zero raw-to-adjusted change for C_eff, top-1, and top-2. It is the only strict model with zero change for the C_eff and top-1 expected-direction checks; the other 22 strict models are multi-parameter models and move in the expected direction.

The manuscript-ready wording is:

```text
22/22 eligible multi-parameter models changed in the expected direction; collie1 (P=1) is invariant by construction.
```

If the complete strict denominator is also shown, this is equivalent to `22/23 including collie1`.

## F2 relevance and marker decision

Collie1 remains a valid member of the frozen F2 model-level estimand. Excluding it is only a sensitivity check (`F2_COLLIE1_SENSITIVITY.csv`: 35/35 positive model-level paired excesses); no F2 numerical value is changed.

The prior Figure 2 script used a dagger label and open square for collie1 because it was the only model classified `MIXED` in restart provenance. That annotation has no remaining scientific purpose for the frozen F2 estimand and could be read as a data-quality warning. It has therefore been removed from:

- `scripts/plot_r2_figure2_unit_hist_cr_deviation.py` y-axis labels;
- the panel (b) marker treatment;
- the generated F2 build-note wording.

The structural exception is documented here instead of encoded as a defect marker in the figure.

## Verdict

```text
COLLIE1_F2_MARKER = REMOVE
```

Recommended manuscript sentence:

> collie1 is a one-parameter model; its exact localization invariance is structural, so it is retained in the complete summaries but does not characterize multi-parameter localization.
