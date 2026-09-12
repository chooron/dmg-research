#!/usr/bin/env python3
"""Read-only provenance and estimand audit for the proposed R2 Figure 3.

This script creates audit CSV/Markdown tables only.  It does not import a
plotting library and does not create figure exports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
R2 = HERE.parents[1]
BENCHMARK = HERE.parents[3]
REPO = BENCHMARK.parents[1]
RESULT = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905"
TABLES = R2 / "tables"

ORDER = TABLES / "F2_MODEL_ORDER.csv"
RANK_COORD = RESULT / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/tables/R_rank_all_271.csv"
RANK_MODEL = RESULT / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/tables/R_rank_model_summaries.csv"
RANK_SELF = RESULT / "r2_final_robustness_20260906/agent_A_rank_self_reference/tables/model_rank_self_primary_5000.csv"
RAW_MODEL = RESULT / "r2_final_robustness_20260906/agent_C_model_equal_claim_audit/tables/frozen_C_eff_model_summaries.csv"
RAW_EQUAL = RESULT / "r2_final_robustness_20260906/agent_C_model_equal_claim_audit/tables/frozen_C_eff_model_equal_summary.csv"
ADJUSTED_BASIN = RESULT / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/basin_adjusted_summary.csv"
WEIGHTS = RESULT / "r2_parameter_axis_audit_20260906/agent_C_coordinate_concentration/tables/basin_coordinate_weights.csv"

WEIGHT_SCRIPT = RESULT / "r2_parameter_axis_audit_20260906/agent_C_coordinate_concentration/analyze_coordinate_concentration.py"
RANK_SCRIPT = RESULT / "r2_parameter_axis_audit_20260906/agent_A_rank_preservation/rank_preservation.py"
RANK_SELF_SCRIPT = RESULT / "r2_final_robustness_20260906/agent_A_rank_self_reference/rank_self_reference.py"
ADJUSTED_SCRIPT = RESULT / "r2_coordinate_icself_final_audit_20260906/agent_B_adjusted_localization/adjusted_localization.py"

EXPECTED_MODELS = 36
EXPECTED_COORDINATES = 271
EXPECTED_STRICT = 23
TOL = 1e-8


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def check_close(observed: float, expected: float, label: str, tol: float = TOL) -> None:
    if not np.isfinite(observed) or abs(float(observed) - float(expected)) > tol:
        raise AssertionError(f"{label}: observed={observed} expected={expected}")


def check_source_set() -> None:
    for path in [ORDER, RANK_COORD, RANK_MODEL, RANK_SELF, RAW_MODEL,
                 RAW_EQUAL, ADJUSTED_BASIN, WEIGHTS, WEIGHT_SCRIPT,
                 RANK_SCRIPT, RANK_SELF_SCRIPT, ADJUSTED_SCRIPT]:
        require(path)


def load_order() -> pd.DataFrame:
    order = pd.read_csv(ORDER)
    required = {"model_id", "plot_order"}
    if not required.issubset(order.columns):
        raise ValueError(f"model order missing {required - set(order.columns)}")
    if len(order) != EXPECTED_MODELS or order.model_id.nunique() != EXPECTED_MODELS:
        raise AssertionError("F2 model order is not a unique 36-model order")
    if sorted(order.plot_order.tolist()) != list(range(1, EXPECTED_MODELS + 1)):
        raise AssertionError("F2 plot order is not 1..36")
    return order.sort_values("plot_order").reset_index(drop=True)


def audit_topk(order: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    raw = pd.read_csv(RAW_MODEL)
    required = {"model", "parameter_count", "n_basins", "C_eff_median",
                "top1_share_median", "top2_share_median"}
    if not required.issubset(raw.columns):
        raise ValueError(f"canonical top-k source missing {required - set(raw.columns)}")
    if raw.model.nunique() != EXPECTED_MODELS or len(raw) != EXPECTED_MODELS:
        raise AssertionError("canonical model-level top-k source is not 36 rows")
    if set(raw.model) != set(order.model_id):
        raise AssertionError("top-k model IDs do not match the frozen 36-model order")
    if (raw.n_basins != 531).any() or raw[["C_eff_median", "top1_share_median", "top2_share_median"]].isna().any().any():
        raise AssertionError("top-k source has incomplete 531-basin model summaries")
    if (raw.top2_share_median < raw.top1_share_median).any():
        raise AssertionError("top-2 cumulative share is below top-1 share")

    rows = raw.rename(columns={"model": "model_id", "parameter_count": "n_parameters"})[
        ["model_id", "n_parameters", "C_eff_median", "top1_share_median", "top2_share_median"]
    ].copy()
    rows = rows.rename(columns={
        "C_eff_median": "C_eff",
        "top1_share_median": "top1_share",
        "top2_share_median": "top2_share",
    })
    rows["source_file"] = rel(RAW_MODEL)
    rows["source_script"] = rel(WEIGHT_SCRIPT)
    rows["coverage_status"] = "PASS: 1 model summary from 531 basin rows"
    rows["frozen_status"] = "YES: canonical model-equal basin-median summary"
    rows = order[["model_id"]].merge(rows, on="model_id", how="left", validate="one_to_one")
    rows.to_csv(TABLES / "F3_TOPK_MODEL_LEVEL_AUDIT.csv", index=False, float_format="%.12g")

    equal = pd.read_csv(RAW_EQUAL)
    eq = equal.set_index("metric")["median"]
    expected = {
        "C_eff": 0.3458437727,
        "top1_share": 0.54561860045,
        "top2_share": 0.84824760785,
    }
    for metric, value in expected.items():
        check_close(float(eq.loc[metric]), value, f"ensemble {metric}")
        check_close(float(rows[metric].median()), value, f"recomputed ensemble {metric}")
    if len(rows) != EXPECTED_MODELS:
        raise AssertionError("top-k output coverage is not 36/36")
    return rows, expected


def audit_rank_atlas(order: pd.DataFrame, strict_models: set[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    coord = pd.read_csv(RANK_COORD)
    required = {"model", "parameter_index", "parameter", "R_rank", "status"}
    if not required.issubset(coord.columns):
        raise ValueError(f"rank coordinate source missing {required - set(coord.columns)}")
    if len(coord) != EXPECTED_COORDINATES or coord.model.nunique() != EXPECTED_MODELS:
        raise AssertionError("rank coordinate coverage is not 271 rows across 36 models")
    if coord.R_rank.isna().any() or (~np.isfinite(coord.R_rank)).any():
        raise AssertionError("rank atlas contains nonfinite R_rank values")
    if coord.duplicated(["model", "parameter_index"]).any():
        raise AssertionError("rank atlas has duplicate model-native coordinates")

    coord = coord.rename(columns={"model": "model_id", "parameter_index": "coordinate_id", "parameter": "parameter_name"})
    coord["within_model_rank"] = coord.groupby("model_id")["R_rank"].rank(method="min", ascending=False).astype(int)
    coord["strict23_flag"] = coord.model_id.isin(strict_models)
    atlas = coord[["model_id", "coordinate_id", "parameter_name", "R_rank", "within_model_rank", "strict23_flag"]]
    atlas = order[["model_id"]].merge(atlas, on="model_id", how="right", validate="one_to_many")
    atlas.to_csv(TABLES / "F3_RANK_COORDINATE_AUDIT.csv", index=False, float_format="%.12g")

    model_summary = pd.read_csv(RANK_MODEL).rename(columns={"model": "model_id", "median": "median_R_rank", "n": "n_coordinates"})
    if len(model_summary) != EXPECTED_MODELS or set(model_summary.model_id) != set(order.model_id):
        raise AssertionError("rank model summary is not 36-model complete")
    check_close(float(model_summary.median_R_rank.median()), 0.406901464622, "model-equal R_rank")
    if (model_summary.median_R_rank <= 0).any():
        raise AssertionError("not all model-level rank summaries are positive")
    if int(model_summary.n_coordinates.sum()) != EXPECTED_COORDINATES:
        raise AssertionError("model coordinate counts do not sum to 271")

    valid = atlas.R_rank.notna()
    summary = pd.DataFrame([{
        "n_models": int(atlas.loc[valid, "model_id"].nunique()),
        "n_coordinates_total": int(valid.sum()),
        "fraction_negative_coordinate_level": float((atlas.loc[valid, "R_rank"] < 0).mean()),
        "fraction_ge_0.5_coordinate_level": float((atlas.loc[valid, "R_rank"] >= 0.5).mean()),
        "fraction_ge_0.8_coordinate_level": float((atlas.loc[valid, "R_rank"] >= 0.8).mean()),
        "fraction_ge_0.9_coordinate_level": float((atlas.loc[valid, "R_rank"] >= 0.9).mean()),
        "model_equal_median_summary": float(model_summary.median_R_rank.median()),
    }])
    summary.to_csv(TABLES / "F3_RANK_COORDINATE_SUMMARY.csv", index=False, float_format="%.12g")
    model_summary["strict23_flag"] = model_summary.model_id.isin(strict_models)
    model_summary[["model_id", "n_coordinates", "median_R_rank", "strict23_flag"]].to_csv(
        TABLES / "F3_RANK_MODEL_SUMMARY.csv", index=False, float_format="%.12g"
    )
    return atlas, summary.iloc[0].to_dict()


def audit_strict_rank(strict_models: set[str]) -> dict[str, float | int]:
    rank = pd.read_csv(RANK_SELF)
    rank = rank[(rank.pool == "primary") & (rank.prefix == 5000) & (rank.primary_reference_status == "PASS")].copy()
    if len(rank) != EXPECTED_STRICT or rank.model.nunique() != EXPECTED_STRICT:
        raise AssertionError("strict rank source is not exactly 23 PASS model rows")
    if set(rank.model) != strict_models:
        raise AssertionError("strict rank model IDs do not match strict localization IDs")
    rank = rank.rename(columns={
        "model": "model_id",
        "R_cross_median": "R_cross",
        "self_median_median": "R_self",
        "DeltaR_self_minus_cross_median": "delta_R",
    })
    rank["strict23_flag"] = True
    rank[["model_id", "R_cross", "R_self", "delta_R", "strict23_flag"]].to_csv(
        TABLES / "F3_STRICT_RANK_MODEL.csv", index=False, float_format="%.12g"
    )
    result = {
        "n_models": len(rank),
        "R_cross": float(rank.R_cross.median()),
        "R_self": float(rank.R_self.median()),
        "delta_R": float(rank.delta_R.median()),
        "positive_delta": int((rank.delta_R > 0).sum()),
    }
    check_close(result["R_cross"], 0.487560126579, "strict R_cross")
    check_close(result["R_self"], 0.797361671925, "strict R_self")
    check_close(result["delta_R"], 0.211845558402, "strict delta_R")
    if result["positive_delta"] != EXPECTED_STRICT:
        raise AssertionError("strict delta_R direction count is not 23/23")
    return result


def audit_exact_localization(strict_models: set[str], order: pd.DataFrame) -> dict[str, float | int]:
    basin = pd.read_csv(ADJUSTED_BASIN, dtype={"basin_id": str})
    if set(basin.model) != strict_models or basin.model.nunique() != EXPECTED_STRICT:
        raise AssertionError("adjusted localization basin support is not the strict 23-model set")
    required = {"model", "basin_id", "status", "C_eff_excess", "C_eff_raw",
                "top1_excess_share", "top1_share_raw", "top2_excess_share", "top2_share_raw"}
    if not required.issubset(basin.columns):
        raise ValueError(f"adjusted basin source missing {required - set(basin.columns)}")

    rows = []
    for model, group in basin.groupby("model", sort=False):
        # Adjusted localization is defined only on PASS basins.  Raw values
        # are taken from those exact rows, not from the all-raw basin set.
        needed = ["C_eff_excess", "C_eff_raw", "top1_excess_share", "top1_share_raw", "top2_excess_share", "top2_share_raw"]
        common = group[group.status.eq("PASS")].copy()
        if common[needed].isna().any().any():
            raise AssertionError(f"{model}: PASS support has missing matched metrics")
        if not len(common):
            raise AssertionError(f"{model}: no exact adjusted-valid basins")
        raw_c = float(common.C_eff_raw.median())
        adj_c = float(common.C_eff_excess.median())
        raw_1 = float(common.top1_share_raw.median())
        adj_1 = float(common.top1_excess_share.median())
        raw_2 = float(common.top2_share_raw.median())
        adj_2 = float(common.top2_excess_share.median())
        rows.append({
            "model_id": model,
            "n_common_basins": int(len(common)),
            "Ceff_raw_exact": raw_c,
            "Ceff_adjusted": adj_c,
            "delta_Ceff": adj_c - raw_c,
            "top1_raw_exact": raw_1,
            "top1_adjusted": adj_1,
            "delta_top1": adj_1 - raw_1,
            "top2_raw_exact": raw_2,
            "top2_adjusted": adj_2,
            "delta_top2": adj_2 - raw_2,
        })
    exact = pd.DataFrame(rows)
    exact = order[["model_id"]].merge(exact, on="model_id", how="right", validate="one_to_one")
    exact.to_csv(TABLES / "F3_LOCALIZATION_EXACT_MATCHED_MODEL.csv", index=False, float_format="%.12g")

    metrics = [
        ("C_eff", "Ceff_raw_exact", "Ceff_adjusted", "delta_Ceff", "<"),
        ("top1_share", "top1_raw_exact", "top1_adjusted", "delta_top1", ">"),
        ("top2_share", "top2_raw_exact", "top2_adjusted", "delta_top2", ">"),
    ]
    summary_rows = []
    for metric, raw_col, adj_col, delta_col, direction in metrics:
        delta = exact[delta_col]
        direction_count = int((delta < 0).sum()) if direction == "<" else int((delta > 0).sum())
        summary_rows.append({
            "metric": metric,
            "n_models": len(exact),
            "raw_model_equal_median": float(exact[raw_col].median()),
            "adjusted_model_equal_median": float(exact[adj_col].median()),
            "direction_count": direction_count,
            "direction_fraction": direction_count / len(exact),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "F3_LOCALIZATION_EXACT_MATCHED_SUMMARY.csv", index=False, float_format="%.12g")
    result = {
        "n_models": len(exact),
        "Ceff_raw": float(exact.Ceff_raw_exact.median()),
        "Ceff_adjusted": float(exact.Ceff_adjusted.median()),
        "top1_raw": float(exact.top1_raw_exact.median()),
        "top1_adjusted": float(exact.top1_adjusted.median()),
        "top2_raw": float(exact.top2_raw_exact.median()),
        "top2_adjusted": float(exact.top2_adjusted.median()),
        "Ceff_lower_count": int((exact.delta_Ceff < 0).sum()),
        "top1_higher_count": int((exact.delta_top1 > 0).sum()),
        "top2_higher_count": int((exact.delta_top2 > 0).sum()),
        "n_common_min": int(exact.n_common_basins.min()),
        "n_common_max": int(exact.n_common_basins.max()),
    }
    check_close(result["Ceff_adjusted"], 0.2933820389, "exact adjusted C_eff")
    check_close(result["top1_adjusted"], 0.672633535, "exact adjusted top1")
    check_close(result["top2_adjusted"], 0.9417612386, "exact adjusted top2")
    if result["Ceff_lower_count"] != 22 or result["top1_higher_count"] != 22:
        raise AssertionError("exact matched direction counts do not reproduce 22/23")
    return result


def decomposition_definition() -> None:
    text = f"""# F3 top-k localization decomposition definition

## Canonical sources

- Coordinate contributions: `{rel(WEIGHTS)}`
- Computation script: `{rel(WEIGHT_SCRIPT)}`
- The model-level values used in the top-k audit are the canonical rows in `{rel(RAW_MODEL)}`.

## Exact estimand

For model $m$, basin $b$, and its $P_m$ native parameter coordinates, let
$\\Delta\\Theta_{{m,b,p}}$ be the frozen normalized dPL-minus-IC coordinate
change.  The squared coordinate contribution and normalized weight are

$$
q_{{m,b,p}} = (\\Delta\\Theta_{{m,b,p}})^2,
\\qquad
w_{{m,b,p}} = \\frac{{q_{{m,b,p}}}}{{\\sum_{{j=1}}^{{P_m}} q_{{m,b,j}}}}.
$$

Rows with zero total squared displacement are not divided; they are marked
`zero_displacement` and their normalized metrics are unavailable.  No such
rows occur in the frozen 36-model product used here.

The effective coordinate number and normalized effective coordinate number are

$$
N_{{eff,m,b}} = \\frac{{1}}{{\\sum_p w_{{m,b,p}}^2}},
\\qquad
C_{{eff,m,b}} = \\frac{{N_{{eff,m,b}}}}{{P_m}}.
$$

For $k=1,2$, coordinates are sorted in descending order of $w$ **within each
model\\times basin** and the cumulative top-$k$ share is

$$
T_{{k,m,b}} = \\sum_{{r=1}}^{{\\min(k,P_m)}} w_{{m,b,(r)}}.
$$

Thus `top1_share` is $T_1$ and `top2_share` is the cumulative $T_2$; it is
not the share of a model-level second coordinate.  The canonical model summary
is the median of the 531 basin values within each model.  The ensemble headline
is the median of the 36 model summaries, with one model receiving one weight.

## Hierarchy and variable parameter counts

This is **Case 1: basin-level decomposition**.  Squaring, normalization,
coordinate ranking, $C_{{eff}}$, and top-k calculation occur at the model\\times
basin level.  Basin-level metrics are then reduced to a model median, and
model medians are then reduced to the model-equal ensemble median.  The
normalization by $P_m$ makes $C_{{eff}}$ comparable across models with different
numbers of native coordinates; top-k uses `min(P_m,k)`.

The 36-row model-level values therefore exist and are frozen as summaries of
basin-level decompositions.  They do not represent a single total displacement
vector with one globally dominant coordinate per model.

## Composition fingerprint versus 36\\times top-k display

The previously blocked composition fingerprint and the 36\\times top-k display
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
"""
    (TABLES / "F3_TOPK_DECOMPOSITION_DEFINITION.md").write_text(text)


def interpretation_note() -> None:
    text = """# F3 rank/localization interpretation note

DECOMPOSITION LEVEL:

Top-k localization is decomposed at the model × basin level.  For each basin,
normalized squared coordinate contributions are ranked within that basin.  The
resulting C_eff and cumulative top-k shares are first summarized across basins
within each model and only then across the 36 models.  There is no frozen
model-level total vector whose coordinate identity is ranked once across all
basins.

WHAT CAN BE SAID:

High basin-wise top-1/top-2 shares can coexist with only moderate coordinate-wise
cross-catchment rank correspondence because the identity of the largest
contributors can vary among basins.  In that situation, displacement is
concentrated within individual basin decompositions while the rank ordering of
any fixed coordinate across catchments is only partly retained.  This is a
coherent descriptive reading of the two estimands, not a contradiction.

The rank atlas measures tie-corrected Spearman correspondence for each native
model-coordinate across the 531 basins.  The localization summaries measure
concentration of squared normalized displacement within each basin.  They use
different reductions and should be read side by side without treating either as
an explanation of the other.

WHAT CANNOT BE SAID:

Figure 3 does not test whether coordinates with larger displacement are the
same coordinates with lower rank correspondence.

Neither top-k share nor C_eff establishes parameter importance, sensitivity,
physical dominance, mechanism, or a model-family/functional-role claim.
The strict23 rows remain reference-qualified descriptive checks and cannot be
upgraded to a full36 inference.  No causal relation between rank loss and
localization is tested here.
"""
    (TABLES / "F3_RANK_LOCALIZATION_INTERPRETATION_NOTE.md").write_text(text)


def model_order_coverage(order: pd.DataFrame, strict_models: set[str], atlas: pd.DataFrame) -> None:
    counts = atlas.groupby("model_id").size().rename("n_parameters")
    out = order[["plot_order", "model_id"]].rename(columns={"plot_order": "display_order"}).copy()
    out["strict23_flag"] = out.model_id.isin(strict_models)
    out["n_parameters"] = out.model_id.map(counts).astype(int)
    if len(out) != EXPECTED_MODELS or out.n_parameters.sum() != EXPECTED_COORDINATES:
        raise AssertionError("model order/coverage output failed")
    out[["display_order", "model_id", "strict23_flag", "n_parameters"]].to_csv(
        TABLES / "F3_MODEL_ORDER_AND_COVERAGE.csv", index=False
    )


def write_report(order: pd.DataFrame, topk: pd.DataFrame, rank_summary: dict[str, float],
                 strict_rank: dict[str, float | int], loc: dict[str, float | int]) -> None:
    top1_model = float(topk.top1_share.median())
    top2_model = float(topk.top2_share.median())
    report = f"""F3 PRE-PLOT AUDIT = READY

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
`{rel(RAW_MODEL)}`, computed by `{rel(WEIGHT_SCRIPT)}`.

Coverage is 36/36 models and 531 basin rows per model. Recomputed model-equal
medians are C_eff=`{topk.C_eff.median():.12f}`, top-1=`{top1_model:.12f}`, and
top-2=`{top2_model:.12f}`; these reproduce the frozen values
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
`n_common_basins` ranging from {loc['n_common_min']} to {loc['n_common_max']}.
The exact-support model-equal medians are:

- raw C_eff = `{loc['Ceff_raw']:.12f}`; adjusted C_eff = `{loc['Ceff_adjusted']:.12f}`;
- raw top-1 = `{loc['top1_raw']:.12f}`; adjusted top-1 = `{loc['top1_adjusted']:.12f}`;
- raw top-2 = `{loc['top2_raw']:.12f}`; adjusted top-2 = `{loc['top2_adjusted']:.12f}`.

Adjusted C_eff is lower for {loc['Ceff_lower_count']}/23 models and adjusted
 top-1 is higher for {loc['top1_higher_count']}/23. Top-2 is higher for
{loc['top2_higher_count']}/23. The previous raw values 0.346893 and 0.573874
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
and their model-equal median is `{rank_summary['model_equal_median_summary']:.12f}`.

## Audit F — strict23 rank benchmark

`F3_STRICT_RANK_MODEL.csv` contains the 23 paired model rows from
`{rel(RANK_SELF)}` computed by `{rel(RANK_SELF_SCRIPT)}`. The strict summaries
are R_cross=`{strict_rank['R_cross']:.12f}`, R_self=`{strict_rank['R_self']:.12f}`,
and median delta_R=`{strict_rank['delta_R']:.12f}`; {strict_rank['positive_delta']}/23
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
"""
    (TABLES / "F3_PREPLOT_AUDIT_REPORT.md").write_text(report)


def main() -> None:
    check_source_set()
    order = load_order()
    # Strict localization support is taken from the exact adjusted basin source.
    adjusted = pd.read_csv(ADJUSTED_BASIN, usecols=["model"])
    loc_models = set(adjusted.model.astype(str))
    rank_self = pd.read_csv(RANK_SELF, usecols=["pool", "prefix", "model", "primary_reference_status"])
    rank_models = set(rank_self.loc[
        rank_self.pool.eq("primary") & rank_self.prefix.eq(5000) & rank_self.primary_reference_status.eq("PASS"), "model"
    ].astype(str))
    if len(loc_models) != EXPECTED_STRICT or len(rank_models) != EXPECTED_STRICT:
        raise AssertionError("strict model support is not 23/23")
    if loc_models != rank_models:
        raise AssertionError(f"strict support mismatch: {sorted(loc_models ^ rank_models)}")
    strict_models = loc_models

    topk, _ = audit_topk(order)
    atlas, rank_summary = audit_rank_atlas(order, strict_models)
    strict_rank = audit_strict_rank(strict_models)
    loc = audit_exact_localization(strict_models, order)
    decomposition_definition()
    interpretation_note()
    model_order_coverage(order, strict_models, atlas)
    write_report(order, topk, rank_summary, strict_rank, loc)

    print({
        "status": "PASS",
        "figures_generated": False,
        "topk_models": len(topk),
        "rank_coordinates": len(atlas),
        "strict_models": len(strict_models),
        "strict_intersection": len(strict_models & rank_models),
        "strict_symmetric_difference": len(strict_models ^ rank_models),
        "exact_support_min_max": [loc["n_common_min"], loc["n_common_max"]],
        "exact_Ceff_raw_adjusted": [loc["Ceff_raw"], loc["Ceff_adjusted"]],
        "exact_top1_raw_adjusted": [loc["top1_raw"], loc["top1_adjusted"]],
    })


if __name__ == "__main__":
    main()
