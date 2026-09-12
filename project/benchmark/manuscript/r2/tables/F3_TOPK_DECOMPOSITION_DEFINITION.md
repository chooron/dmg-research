# F3 top-k localization decomposition definition

## Canonical sources

- Coordinate contributions: `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_parameter_axis_audit_20260906/agent_C_coordinate_concentration/tables/basin_coordinate_weights.csv`
- Computation script: `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_parameter_axis_audit_20260906/agent_C_coordinate_concentration/analyze_coordinate_concentration.py`
- The model-level values used in the top-k audit are the canonical rows in `project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_final_robustness_20260906/agent_C_model_equal_claim_audit/tables/frozen_C_eff_model_summaries.csv`.

## Exact estimand

For model $m$, basin $b$, and its $P_m$ native parameter coordinates, let
$\Delta\Theta_{m,b,p}$ be the frozen normalized dPL-minus-IC coordinate
change.  The squared coordinate contribution and normalized weight are

$$
q_{m,b,p} = (\Delta\Theta_{m,b,p})^2,
\qquad
w_{m,b,p} = \frac{q_{m,b,p}}{\sum_{j=1}^{P_m} q_{m,b,j}}.
$$

Rows with zero total squared displacement are not divided; they are marked
`zero_displacement` and their normalized metrics are unavailable.  No such
rows occur in the frozen 36-model product used here.

The effective coordinate number and normalized effective coordinate number are

$$
N_{eff,m,b} = \frac{1}{\sum_p w_{m,b,p}^2},
\qquad
C_{eff,m,b} = \frac{N_{eff,m,b}}{P_m}.
$$

For $k=1,2$, coordinates are sorted in descending order of $w$ **within each
model\times basin** and the cumulative top-$k$ share is

$$
T_{k,m,b} = \sum_{r=1}^{\min(k,P_m)} w_{m,b,(r)}.
$$

Thus `top1_share` is $T_1$ and `top2_share` is the cumulative $T_2$; it is
not the share of a model-level second coordinate.  The canonical model summary
is the median of the 531 basin values within each model.  The ensemble headline
is the median of the 36 model summaries, with one model receiving one weight.

## Hierarchy and variable parameter counts

This is **Case 1: basin-level decomposition**.  Squaring, normalization,
coordinate ranking, $C_{eff}$, and top-k calculation occur at the model\times
basin level.  Basin-level metrics are then reduced to a model median, and
model medians are then reduced to the model-equal ensemble median.  The
normalization by $P_m$ makes $C_{eff}$ comparable across models with different
numbers of native coordinates; top-k uses `min(P_m,k)`.

The 36-row model-level values therefore exist and are frozen as summaries of
basin-level decompositions.  They do not represent a single total displacement
vector with one globally dominant coordinate per model.

## Composition fingerprint versus 36\times top-k display

The previously blocked composition fingerprint and the 36\times top-k display
use the same underlying frozen basin-wise squared coordinate weights.  The
canonical 36-row `C_eff`, top-1, and cumulative top-2 values are available and
reproduce the frozen ensemble medians.  A stacked display may therefore use
`top1`, `top2 - top1`, and `1 - top2`; this is arithmetic decomposition of two
already-frozen model summary values and adds no new estimand.  It must not be
labeled as a single globally dominant-coordinate composition.  The raw
basin-wise coordinate identity table remains the appropriate source for any
basin-specific identity display.

**INTERPRETATION:** “Top-1 share” is the fraction of squared normalized
coordinate displacement carried by the largest coordinate within a basin,
and “top-2 share” is the cumulative fraction carried by the two largest
coordinates within that basin, each summarized by a within-model basin median
before equal weighting across models; they are not parameter importance or
sensitivity measures.
