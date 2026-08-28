"""Render the four-panel Figure 7 v2 from frozen cached panel tables.


The figure is a real-catchment corroboration display, not a truth-validation
or causal-proving display.  The main figure contains only:
    (a) conditional phase x external-SWE burden interaction matrix;
    (b) condition-resolved effect sizes;
    (c) ECDF heterogeneity;
    (d) descriptive response composition.

The drawing layer does not recompute statistics or rewrite cached data tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.scripts.r4.common import default_results_root  # noqa: E402
from manuscript.scripts.r4.robustness_analysis import (  # noqa: E402
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
    bootstrap_median_ci,
)
from manuscript.scripts.r4.soil_analysis import calendar_month_anomaly  # noqa: E402
from manuscript.scripts.shared.r1_plot_style import (  # noqa: E402
    COLOR_BASE,
    COLOR_CN,
    COLOR_DARK_NEUTRAL,
    COLOR_LIGHT_REF,
    COLOR_TGD,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)

FIGURES_DIR = HERE.parents[1] / "figures"
CANONICAL_DPL = "dPL_seed42"
REGIMES = ["IC_fused", CANONICAL_DPL]
STRUCTURES = ["TGD", "CN"]
STRUCTURE_DISPLAY = {"TGD": "TGD", "CN": "CN"}
PHASE_ORDER = [
    "Phase_1_Snow_Accumulation",
    "Phase_2_Active_Melt_Recharge",
    "Phase_3_Post_Melt_Transition",
    "Phase_4_Summer_Dry_Down",
]
PHASE_TICK_LABELS = ["Accumulation", "Active melt", "Post-melt", "Dry-down"]
PHASE_SHORT = dict(zip(PHASE_ORDER, PHASE_TICK_LABELS))
MIN_EXTERNAL_SWE_MM = 5.0
MIN_BASIN_BURDEN_MM = 20.0
MIN_PHASE_DAYS = 30
DELTA_LABEL = "Δ anomaly correlation relative to Base"
DESCRIPTIVE_EQUIVALENCE_BAND = 0.02
COMPOSITION_TAU = 0.10
COMPOSITION_DELTA = 0.05
FIG_W_CM = 15.0
FIG_H_CM = 13.2
FIG_DPI = 400

REGIME_CFG = {
    "IC_fused": {
        "label": "IC fused",
        "short": "IC",
        "ls": "-",
        "marker": "o",
    },
    CANONICAL_DPL: {
        "label": "dPL seed 42",
        "short": "dPL",
        "ls": (0, (4.0, 2.0)),
        "marker": "^",
    },
}
STRUCTURE_CFG = {
    "TGD": {
        "label": "TGD − Base",
        "color": COLOR_TGD,
        "delta_col": "delta_tgd2_base_anomaly",
        "absolute_col": "tgd2_anomaly_corr",
    },
    "CN": {
        "label": "CN − Base",
        "color": COLOR_CN,
        "delta_col": "delta_cn_base_anomaly",
        "absolute_col": "cn_anomaly_corr",
    },
}
FULL_BIN_LABELS = ["No/trace", "Low", "Middle", "High", "Very high"]
PHASE_BIN_LABELS = ["Low", "Middle", "High"]
EXTERNAL_PHASE_NOTE = (
    "External Caravan/Snow-17 SWE defines burden and phase masks: annual peak, "
    "SWE ≥ 5 mm, melt-out; ERA5-Land SM100 is a process-state reference."
)


# TGD disk identity remains TGD2; F7 uses the reconstructed seed-42 replay.
MODEL_ARRAY_FILES = {
    CANONICAL_DPL: {
        "Base": "r4_official_dpl_XAJ_seed42/official_dpl_XAJ_seed42_full_arrays.npz",
        "TGD": "r4_replay_dpl_XAJ_TGD2_seed42/reconstructed_dpl_XAJ_TGD2_seed42_full_arrays.npz",
        "CN": "r4_official_dpl_XAJ_CN_seed42/official_dpl_XAJ_CN_seed42_full_arrays.npz",
    },
    "IC_fused": {
        "Base": "r4_ic_fused_XAJ/ic_fused_XAJ_full_arrays.npz",
        "TGD": "r4_ic_fused_XAJ_TGD2/ic_fused_XAJ_TGD2_full_arrays.npz",
        "CN": "r4_ic_fused_XAJ_CN/ic_fused_XAJ_CN_full_arrays.npz",
    },
}
TGD_PARAMETER_FILES = {
    CANONICAL_DPL: "r4_replay_dpl_XAJ_TGD2_seed42/reconstructed_best_parameters_physical.npz",
    "IC_fused": "r4_ic_fused_XAJ_TGD2/best_parameters_physical.npz",
}

# Matrix magnitude is neutral and non-semantic: structure colors are reserved
# for the structure blocks and line series in panels (b)-(f).
MATRIX_CMAP = LinearSegmentedColormap.from_list(
    "figure7_neutral_magnitude",
    ["#FFFFFF", "#D9DCDD", "#858A8E", COLOR_DARK_NEUTRAL],
    N=256,
)
MATRIX_VMAX = 0.55

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Finite Pearson correlation with the canonical zero-variance convention."""
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 2:
        return float("nan")
    av = a[valid]
    bv = b[valid]
    if np.std(av) == 0.0 or np.std(bv) == 0.0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _external_phase_array(dates: pd.DatetimeIndex, swe: np.ndarray) -> np.ndarray:
    """Construct canonical phase codes from external SWE only.

    No Base, TGD, CN, W_total, or internal snow state enters this function.
    """
    water_year = np.where(dates.month >= 10, dates.year + 1, dates.year).astype(int)
    phase = np.zeros(len(dates), dtype=np.int8)

    for water_year_id in np.unique(water_year):
        indices = np.flatnonzero(water_year == water_year_id)
        sw = swe[indices]
        finite = np.isfinite(sw)
        if not finite.any() or np.nanmax(sw) < MIN_EXTERNAL_SWE_MM:
            continue

        rel = np.arange(len(indices))
        peak_rel = int(np.nanargmax(sw))
        acc_rel = np.flatnonzero(finite & (sw >= MIN_EXTERNAL_SWE_MM))
        acc_start_rel = int(acc_rel[0]) if len(acc_rel) else 0
        post_peak = np.flatnonzero(
            (rel > peak_rel) & (sw < MIN_EXTERNAL_SWE_MM)
        )
        melt_end_rel = int(post_peak[0]) if len(post_peak) else len(indices) - 1
        months = dates[indices].month.to_numpy()

        phase[indices[(rel >= acc_start_rel) & (rel <= peak_rel)]] = 1
        phase[indices[(rel > peak_rel) & (rel <= melt_end_rel)]] = 2
        phase[indices[(rel > melt_end_rel) & (months <= 6)]] = 3
        phase[indices[np.isin(months, [7, 8, 9])]] = 4

    return phase


def _load_w_total(
    path: Path,
    expected_basin_ids: list[str],
    expected_full_dates: pd.DatetimeIndex,
    test_slice: slice,
) -> np.ndarray:
    """Load W_total and verify basin/date axes before phase statistics."""
    with np.load(path, allow_pickle=False) as archive:
        basin_ids = [str(x).zfill(8) for x in archive["basin_ids"]]
        if basin_ids != expected_basin_ids:
            raise ValueError(f"Basin ordering mismatch in {path}")
        archive_dates = pd.DatetimeIndex(pd.to_datetime(archive["dates"].astype(str)))
        if not archive_dates.equals(expected_full_dates):
            raise ValueError(f"Date axis mismatch in {path}")
        arrays = [archive[name] for name in ("wu", "wl", "wd")]
        if any(a.shape != (len(expected_basin_ids), len(expected_full_dates)) for a in arrays):
            raise ValueError(f"State-array shape mismatch in {path}")
        return (arrays[0][:, test_slice] + arrays[1][:, test_slice] + arrays[2][:, test_slice]).astype(
            np.float64
        )

def _invalid_tgd_parameter_ids(
    results_root: Path, regime: str, expected_basin_ids: list[str]
    ) -> list[str]:
    """Identify rows whose source TGD parameters are non-finite."""
    parameter_path = results_root / TGD_PARAMETER_FILES[regime]
    with np.load(parameter_path, allow_pickle=False) as archive:
        params = np.asarray(archive["params"], dtype=np.float64)
    if params.shape[0] != len(expected_basin_ids):
        raise ValueError(f"TGD parameter rows do not match basin axis in {parameter_path}")
    invalid = ~np.isfinite(params).all(axis=1)
    return [basin_id for basin_id, bad in zip(expected_basin_ids, invalid, strict=True) if bad]


def _validate_external_phase_table(df_phase: pd.DataFrame) -> None:
    """Gate A: ensure the official phase table is the external-SWE table."""
    required = {
        "regime",
        "basin_id",
        "phase_code",
        "phase_name",
        "n_days",
        "base_anomaly_corr",
        "cn_anomaly_corr",
        "delta_anomaly_corr",
        "snow_burden_swe_mm",
    }
    missing = required.difference(df_phase.columns)
    if missing:
        raise ValueError(f"Official external phase table missing columns: {sorted(missing)}")
    if not set(PHASE_ORDER).issubset(set(df_phase["phase_name"].unique())):
        raise ValueError("Official external phase table is missing a canonical phase")
    if not (pd.to_numeric(df_phase["n_days"], errors="coerce") >= MIN_PHASE_DAYS).all():
        raise ValueError("Official phase table contains rows below the minimum phase-day gate")
    if df_phase["snow_burden_swe_mm"].isna().any():
        raise ValueError("Official external phase table contains missing SWE burden")

def _load_external_structure_phase_rows(
    results_root: Path,
    df_paired: pd.DataFrame,
    structure: str,
    regimes: list[str],
 ) -> pd.DataFrame:
    """Recompute Base-vs-structure phase rows from current Caravan SWE masks."""
    if structure not in ("CN", "TGD"):
        raise ValueError(f"Unsupported phase structure: {structure}")
    caravan_path = results_root / "r4_caravan_soil_reference_v1" / "caravan_soil_ensemble.npz"
    with np.load(caravan_path, allow_pickle=False) as caravan:
        expected_basin_ids = [str(x).zfill(8) for x in caravan["basin_ids"]]
        expected_full_dates = pd.DatetimeIndex(pd.to_datetime(caravan["dates"].astype(str)))
        test_slice = slice(int(caravan["test_slice_start"]), int(caravan["test_slice_stop"]))
        dates = expected_full_dates[test_slice]
        sm100 = caravan["SM100"][:, test_slice].astype(np.float64)
        swe = caravan["caravan_swe"][:, test_slice].astype(np.float64)
    months = dates.month.to_numpy()
    basin_index = {basin_id: i for i, basin_id in enumerate(expected_basin_ids)}
    rows: list[dict] = []
    for regime in regimes:
        source = df_paired[df_paired["regime"] == regime].drop_duplicates("basin_id")
        if len(source) != 531:
            raise ValueError(f"{regime} paired table does not contain 531 basins")
        burden = source.set_index("basin_id")["snow_burden_swe_mm"].astype(float).to_dict()
        eligible_ids = sorted(
            basin_id for basin_id, value in burden.items() if value >= MIN_BASIN_BURDEN_MM
        )
        paths = MODEL_ARRAY_FILES[regime]
        base = _load_w_total(results_root / paths["Base"], expected_basin_ids, expected_full_dates, test_slice)
        structure_w = _load_w_total(results_root / paths[structure], expected_basin_ids, expected_full_dates, test_slice)
        if structure == "TGD":
            invalid_ids = _invalid_tgd_parameter_ids(results_root, regime, expected_basin_ids)
            structure_w[np.isin(expected_basin_ids, invalid_ids), :] = np.nan
        for basin_id in eligible_ids:
            if basin_id not in basin_index:
                raise ValueError(f"Basin {basin_id} is absent from Caravan reference")
            i = basin_index[basin_id]
            phase = _external_phase_array(dates, swe[i])
            ref = sm100[i]
            base_w = base[i]
            structure_series = structure_w[i]
            for phase_code, phase_name in enumerate(PHASE_ORDER, start=1):
                mask = phase == phase_code
                if int(mask.sum()) < MIN_PHASE_DAYS:
                    continue
                phase_months = months[mask]
                ref_anom = calendar_month_anomaly(ref[mask], phase_months)
                base_anom = calendar_month_anomaly(base_w[mask], phase_months)
                base_corr = _corr(base_anom, ref_anom)
                if np.isfinite(structure_series[mask]).sum() == 0:
                    structure_corr = float("nan")
                else:
                    structure_anom = calendar_month_anomaly(structure_series[mask], phase_months)
                    structure_corr = _corr(structure_anom, ref_anom)
                row = {
                    "regime": regime,
                    "basin_id": basin_id,
                    "phase_code": phase_code,
                    "phase_name": phase_name,
                    "n_days": int(mask.sum()),
                    "base_anomaly_corr": base_corr,
                    "snow_burden_swe_mm": burden[basin_id],
                }
                if structure == "CN":
                    row["cn_anomaly_corr"] = structure_corr
                    row["delta_anomaly_corr"] = structure_corr - base_corr
                else:
                    row["tgd_anomaly_corr"] = structure_corr
                    row["delta_tgd2_base_anomaly"] = structure_corr - base_corr
                rows.append(row)
    return pd.DataFrame(rows)

def _finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)[np.isfinite(np.asarray(values, dtype=float))]


def _rank_quantile_labels(values: pd.Series, n_groups: int, labels: list[str]) -> pd.Series:
    """Equal-count deterministic groups with explicit tie handling."""
    ranks = values.rank(method="first")
    return pd.qcut(ranks, n_groups, labels=labels)


def _build_full_burden_bins(df_paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build five full-population bins, reserving a tied-zero endpoint."""
    source = df_paired[df_paired["regime"] == CANONICAL_DPL].copy()
    source = source.drop_duplicates("basin_id")[["basin_id", "snow_burden_swe_mm"]]
    if len(source) != 531:
        raise ValueError(f"Full-basin burden population is {len(source)}, expected 531")

    for regime in REGIMES:
        other = df_paired[df_paired["regime"] == regime].drop_duplicates("basin_id")
        merged = source.merge(other[["basin_id", "snow_burden_swe_mm"]], on="basin_id", suffixes=("_primary", "_other"))
        if len(merged) != 531 or not np.allclose(
            merged["snow_burden_swe_mm_primary"], merged["snow_burden_swe_mm_other"], equal_nan=False
        ):
            raise ValueError(f"External SWE burden mismatch across regimes: {regime}")

    source["burden_group"] = ""
    source["burden_group_index"] = -1
    zero = np.isclose(source["snow_burden_swe_mm"].to_numpy(float), 0.0)
    source.loc[zero, "burden_group"] = FULL_BIN_LABELS[0]
    source.loc[zero, "burden_group_index"] = 0

    positive = source.loc[~zero, "snow_burden_swe_mm"]
    positive_labels = _rank_quantile_labels(positive, 4, FULL_BIN_LABELS[1:])
    source.loc[positive.index, "burden_group"] = positive_labels.astype(str)
    source.loc[positive.index, "burden_group_index"] = positive_labels.cat.codes.to_numpy() + 1

    rows: list[dict] = []
    for group_index, group in enumerate(FULL_BIN_LABELS):
        sub = source[source["burden_group"] == group]
        rows.append(
            {
                "group_index": group_index,
                "burden_group": group,
                "assignment_rule": "SWE == 0" if group_index == 0 else "positive SWE rank(method='first') qcut(4)",
                "n_basins": int(len(sub)),
                "burden_min_mm": float(sub["snow_burden_swe_mm"].min()),
                "burden_max_mm": float(sub["snow_burden_swe_mm"].max()),
                "rank_min": float(sub["snow_burden_swe_mm"].rank(method="first").min()),
                "rank_max": float(sub["snow_burden_swe_mm"].rank(method="first").max()),
            }
        )
    return source, pd.DataFrame(rows)


def _build_phase_burden_terciles(
    df_phase_cn: pd.DataFrame,
    df_phase_tgd: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build common three-group burden definitions for all four phases."""
    eligible_sets: list[set[str]] = []
    for regime in REGIMES:
        sub = df_phase_cn[df_phase_cn["regime"] == regime]
        counts = sub.groupby("basin_id")["phase_name"].nunique()
        eligible_sets.append(set(counts[counts == len(PHASE_ORDER)].index))
    common_ids = sorted(set.intersection(*eligible_sets))
    if len(common_ids) < 3:
        raise ValueError("Too few common all-phase eligible basins for burden terciles")

    burden = (
        df_phase_cn[df_phase_cn["basin_id"].isin(common_ids)]
        .drop_duplicates("basin_id")[["basin_id", "snow_burden_swe_mm"]]
        .copy()
    )
    labels = _rank_quantile_labels(burden["snow_burden_swe_mm"], 3, PHASE_BIN_LABELS)
    burden["phase_burden_group"] = labels.astype(str)
    burden["phase_burden_group_index"] = labels.cat.codes
    if burden["phase_burden_group"].value_counts().reindex(PHASE_BIN_LABELS).isna().any():
        raise ValueError("Phase tercile construction produced an empty group")

    # TGD is derived from the same official CN basin/phase masks; verify that
    # all common IDs have the same phase support before plotting.
    for regime in REGIMES:
        for structure_df, name in [(df_phase_cn, "CN"), (df_phase_tgd, "TGD")]:
            sub = structure_df[
                (structure_df["regime"] == regime) & structure_df["basin_id"].isin(common_ids)
            ]
            counts = sub.groupby("basin_id")["phase_name"].nunique()
            if len(counts) != len(common_ids) or not (counts == len(PHASE_ORDER)).all():
                raise ValueError(f"{name} {regime} phase support differs within common tercile population")

    rows: list[dict] = []
    for group_index, group in enumerate(PHASE_BIN_LABELS):
        sub = burden[burden["phase_burden_group"] == group]
        rows.append(
            {
                "group_index": group_index,
                "burden_group": group,
                "assignment_rule": "rank(method='first') qcut(3) across common all-phase eligible basins",
                "n_basins": int(len(sub)),
                "burden_min_mm": float(sub["snow_burden_swe_mm"].min()),
                "burden_max_mm": float(sub["snow_burden_swe_mm"].max()),
                "rank_min": float(sub["snow_burden_swe_mm"].rank(method="first").min()),
                "rank_max": float(sub["snow_burden_swe_mm"].rank(method="first").max()),
            }
        )
    return burden, pd.DataFrame(rows), common_ids



def _build_complete_case_phase_terciles(
    phase_long_all: pd.DataFrame, phase_assignments: pd.DataFrame
 ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build a common-support complete-case burden population for panel (a)."""
    expected_rows = len(PHASE_ORDER) * len(STRUCTURES) * len(REGIMES)
    support = phase_long_all.groupby("basin_id")["delta_r"].agg(
        n_rows="size",
        n_finite=lambda values: int(np.isfinite(values.to_numpy(float)).sum()),
    )
    complete_ids = sorted(support.index[(support["n_rows"] == expected_rows) & (support["n_finite"] == expected_rows)])
    if len(complete_ids) < 3:
        raise ValueError("Too few common complete-case basins for burden terciles")
    burden = phase_assignments[phase_assignments["basin_id"].isin(complete_ids)].copy()
    labels = _rank_quantile_labels(burden["snow_burden_swe_mm"], 3, PHASE_BIN_LABELS)
    burden["phase_burden_group"] = labels.astype(str)
    burden["phase_burden_group_index"] = labels.cat.codes
    rows: list[dict] = []
    for group_index, group in enumerate(PHASE_BIN_LABELS):
        sub = burden[burden["phase_burden_group"] == group]
        rows.append(
            {
                "group_index": group_index,
                "burden_group": group,
                "assignment_rule": "rank(method='first') qcut(3) across common complete-case basins",
                "n_basins": int(len(sub)),
                "burden_min_mm": float(sub["snow_burden_swe_mm"].min()),
                "burden_max_mm": float(sub["snow_burden_swe_mm"].max()),
                "rank_min": float(sub["snow_burden_swe_mm"].rank(method="first").min()),
                "rank_max": float(sub["snow_burden_swe_mm"].rank(method="first").max()),
            }
        )
    return burden, pd.DataFrame(rows), complete_ids


def _phase_long_table(df_phase_cn: pd.DataFrame, df_phase_tgd: pd.DataFrame) -> pd.DataFrame:
    """Put both structures on one basin/phase-level paired-effect table."""
    cn = df_phase_cn[df_phase_cn["regime"].isin(REGIMES)].copy()
    cn["structure"] = "CN"
    cn["delta_r"] = cn["delta_anomaly_corr"]
    tgd = df_phase_tgd[df_phase_tgd["regime"].isin(REGIMES)].copy()
    tgd["structure"] = "TGD"
    tgd["delta_r"] = tgd["delta_tgd2_base_anomaly"]
    cols = [
        "regime",
        "basin_id",
        "phase_code",
        "phase_name",
        "n_days",
        "snow_burden_swe_mm",
        "structure",
        "delta_r",
    ]
    return pd.concat([cn[cols], tgd[cols]], ignore_index=True)


def _summary_row(values: np.ndarray) -> tuple[float, float, float, int]:
    finite = _finite_values(values)
    med, lo, hi = bootstrap_median_ci(finite)
    return med, lo, hi, int(len(finite))

def _build_panel_a_table(phase_long: pd.DataFrame) -> pd.DataFrame:
    """Summarize phase-by-tercile effects for one explicitly defined support."""
    rows: list[dict] = []
    for regime in REGIMES:
        for structure in STRUCTURES:
            sub = phase_long[(phase_long["regime"] == regime) & (phase_long["structure"] == structure)]
            for phase_name in PHASE_ORDER:
                for group_index, group in enumerate(PHASE_BIN_LABELS):
                    vals = sub[
                        (sub["phase_name"] == phase_name)
                        & (sub["phase_burden_group"] == group)
                    ]["delta_r"].to_numpy(float)
                    med, lo, hi, n = _summary_row(vals)
                    rows.append(
                        {
                            "regime": regime,
                            "structure": structure,
                            "phase_name": phase_name,
                            "phase_label": PHASE_SHORT[phase_name],
                            "burden_group": group,
                            "burden_group_index": group_index,
                            "median_delta_r": med,
                            "ci_lower": lo,
                            "ci_upper": hi,
                            "n_basins": n,
                        }
                    )
    return pd.DataFrame(rows)


def _build_forest_table(full_long: pd.DataFrame, panel_c: pd.DataFrame) -> pd.DataFrame:
    """Build categorical forest rows without connecting categorical phases."""
    rows: list[dict] = []
    for structure in STRUCTURES:
        for regime in REGIMES:
            sub = full_long[full_long["regime"] == regime]
            delta_col = STRUCTURE_CFG[structure]["delta_col"]
            med, lo, hi, n = _summary_row(sub[delta_col].to_numpy(float))
            rows.append(
                {
                    "row_index": 0,
                    "row_label": "All days",
                    "phase_name": "All days",
                    "burden_endpoint": "All days",
                    "structure": structure,
                    "regime": regime,
                    "median_delta_r": med,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_basins": n,
                }
            )
    phase_index = {phase: i for i, phase in enumerate(PHASE_ORDER)}
    endpoint_index = {"High": 0, "Low": 1}
    for row in panel_c.itertuples(index=False):
        rows.append(
            {
                "row_index": 1 + 2 * phase_index[row.phase_name] + endpoint_index[row.burden_endpoint],
                "row_label": f"{row.phase_label} · {row.burden_endpoint} burden",
                "phase_name": row.phase_name,
                "burden_endpoint": row.burden_endpoint,
                "structure": row.structure,
                "regime": row.regime,
                "median_delta_r": row.median_delta_r,
                "ci_lower": row.ci_lower,
                "ci_upper": row.ci_upper,
                "n_basins": int(row.n_basins),
            }
        )
    return pd.DataFrame(rows)


def _composition_labels(active: pd.Series, dry: pd.Series, tau: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Assign descriptive composition categories without substituting invalid rows."""
    active_values = active.to_numpy(float)
    dry_values = dry.to_numpy(float)
    valid = np.isfinite(active_values) & np.isfinite(dry_values)
    category = np.full(len(active_values), "invalid", dtype=object)
    category[valid & (active_values >= tau) & (np.abs(dry_values) < delta)] = "melt_specific"
    category[valid & (active_values >= tau) & (dry_values >= delta)] = "generic"
    category[valid & (active_values < tau) & (np.abs(dry_values) < delta)] = "no_change"
    category[valid & (category == "invalid")] = "mixed"
    return category, valid



def _composition_counts(active: pd.Series, dry: pd.Series, tau: float, delta: float) -> tuple[dict[str, int], int]:
    category, valid = _composition_labels(active, dry, tau, delta)
    counts = {key: int(np.sum(category == key)) for key in ("melt_specific", "generic", "no_change", "mixed")}
    return counts, int(valid.sum())


def _build_joint_phase_table(phase_long_all: pd.DataFrame) -> pd.DataFrame:
    """Pair active-melt and dry-down contrasts for both structures and regimes."""
    active = phase_long_all[phase_long_all["phase_name"] == PHASE_ORDER[1]][["basin_id", "regime", "structure", "snow_burden_swe_mm", "delta_r"]].rename(columns={"delta_r": "active_delta_r"})
    dry = phase_long_all[phase_long_all["phase_name"] == PHASE_ORDER[3]][["basin_id", "regime", "structure", "delta_r"]].rename(columns={"delta_r": "dry_down_delta_r"})
    return active.merge(dry, on=["basin_id", "regime", "structure"], how="inner", validate="one_to_one")


def _build_composition_tables(panel_e_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build descriptive burden-stratified response compositions and SI sensitivity."""
    data = _with_paired_burden_groups(panel_e_all)
    aggregate_rows: list[dict] = []
    for structure in STRUCTURES:
        for regime in REGIMES:
            for group_index, group in enumerate(PHASE_BIN_LABELS):
                sub = data[(data["paired_burden_group"] == group) & (data["structure"] == structure) & (data["regime"] == regime)]
                counts, n = _composition_counts(sub["active_delta_r"], sub["dry_down_delta_r"], COMPOSITION_TAU, COMPOSITION_DELTA)
                row = {"structure": structure, "regime": regime, "burden_group": group, "burden_group_index": group_index, "tau": COMPOSITION_TAU, "delta": COMPOSITION_DELTA, "n": n}
                row.update({f"{key}_count": value for key, value in counts.items()})
                row.update({f"{key}_pct": 100.0 * value / n if n else float("nan") for key, value in counts.items()})
                aggregate_rows.append(row)
    sensitivity_rows: list[dict] = []
    for tau in (0.05, 0.10, 0.15):
        for delta in (0.03, 0.05, 0.08):
            for structure in STRUCTURES:
                for regime in REGIMES:
                    for group_index, group in enumerate(PHASE_BIN_LABELS):
                        sub = data[(data["paired_burden_group"] == group) & (data["structure"] == structure) & (data["regime"] == regime)]
                        counts, n = _composition_counts(sub["active_delta_r"], sub["dry_down_delta_r"], tau, delta)
                        row = {"structure": structure, "regime": regime, "burden_group": group, "burden_group_index": group_index, "tau": tau, "delta": delta, "n": n}
                        row.update({f"{key}_count": value for key, value in counts.items()})
                        row.update({f"{key}_pct": 100.0 * value / n if n else float("nan") for key, value in counts.items()})
                        sensitivity_rows.append(row)
    return pd.DataFrame(aggregate_rows), pd.DataFrame(sensitivity_rows)

def _with_paired_burden_groups(panel_e_all: pd.DataFrame) -> pd.DataFrame:
    """Attach the canonical rank-first burden terciles used by panel (d)."""
    data = panel_e_all.copy()
    basin_burden = data[["basin_id", "snow_burden_swe_mm"]].drop_duplicates("basin_id")
    labels = _rank_quantile_labels(basin_burden["snow_burden_swe_mm"], 3, PHASE_BIN_LABELS)
    basin_burden["paired_burden_group"] = labels.astype(str)
    basin_burden["paired_burden_group_index"] = labels.cat.codes
    return data.merge(
        basin_burden,
        on=["basin_id", "snow_burden_swe_mm"],
        how="left",
        validate="many_to_one",
    )



def _build_composition_audit_tables(
    panel_e_all: pd.DataFrame,
    tgd_parameter_invalid_ids: set[str],
 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit valid and conservative-denominator TGD dPL High composition support."""
    data = _with_paired_burden_groups(panel_e_all)
    sub = data[
        (data["structure"] == "TGD")
        & (data["regime"] == CANONICAL_DPL)
        & (data["paired_burden_group"] == "High")
    ].copy()
    category, valid = _composition_labels(
        sub["active_delta_r"], sub["dry_down_delta_r"], COMPOSITION_TAU, COMPOSITION_DELTA
    )
    sub["valid_pair"] = valid
    sub["parameter_nonfinite"] = sub["basin_id"].isin(tgd_parameter_invalid_ids)
    sub["category"] = np.where(valid, category, "invalid")
    sub["failure_mode"] = np.where(
        sub["valid_pair"],
        "defined paired effect",
        np.where(sub["parameter_nonfinite"], "non-finite TGD parameter; invalid state", "other undefined paired effect"),
    )
    support = sub[[
        "basin_id", "snow_burden_swe_mm", "paired_burden_group",
        "active_delta_r", "dry_down_delta_r", "valid_pair",
        "parameter_nonfinite", "category", "failure_mode",
    ]].sort_values("basin_id").reset_index(drop=True)
    valid_category = category[valid]
    valid_counts = {key: int(np.sum(valid_category == key)) for key in ("melt_specific", "generic", "no_change", "mixed")}
    n_valid = int(valid.sum())
    n_invalid = int((~valid).sum())
    n_total = n_valid + n_invalid
    conservative_counts = valid_counts.copy()
    conservative_counts["no_change"] += n_invalid
    row = {
        "structure": "TGD",
        "regime": CANONICAL_DPL,
        "burden_group": "High",
        "tau": COMPOSITION_TAU,
        "delta": COMPOSITION_DELTA,
        "valid_n": n_valid,
        "invalid_n": n_invalid,
        "conservative_denominator_n": n_total,
        "invalid_assignment": "No detectable change for denominator sensitivity only; not observed or imputed",
    }
    for key in ("melt_specific", "generic", "no_change", "mixed"):
        row[f"{key}_valid_count"] = valid_counts[key]
        row[f"{key}_valid_pct"] = 100.0 * valid_counts[key] / n_valid if n_valid else float("nan")
        row[f"{key}_conservative_count"] = conservative_counts[key]
        row[f"{key}_conservative_pct"] = 100.0 * conservative_counts[key] / n_total if n_total else float("nan")
    support_by_group_rows: list[dict] = []
    for group in PHASE_BIN_LABELS:
        group_sub = data[
            (data["structure"] == "TGD")
            & (data["regime"] == CANONICAL_DPL)
            & (data["paired_burden_group"] == group)
        ].copy()
        group_category, group_valid = _composition_labels(
            group_sub["active_delta_r"], group_sub["dry_down_delta_r"], COMPOSITION_TAU, COMPOSITION_DELTA
        )
        group_valid_categories = {key: int(np.sum(group_category[group_valid] == key)) for key in ("melt_specific", "generic", "no_change", "mixed")}
        invalid_ids = sorted(group_sub.loc[~group_valid, "basin_id"].astype(str))
        support_by_group_rows.append(
            {
                "structure": "TGD",
                "regime": CANONICAL_DPL,
                "burden_group": group,
                "eligible_n": int(len(group_sub)),
                "valid_n": int(group_valid.sum()),
                "invalid_n": int((~group_valid).sum()),
                "parameter_invalid_n": int(group_sub.loc[~group_valid, "basin_id"].isin(tgd_parameter_invalid_ids).sum()),
                "invalid_ids": ";".join(invalid_ids),
                **{f"{key}_valid_count": value for key, value in group_valid_categories.items()},
            }
        )
    return support, pd.DataFrame([row]), pd.DataFrame(support_by_group_rows)



def _build_cn_high_drydown_audit(panel_e_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the CN High-burden dry-down distribution and fixed-bin histogram."""
    data = _with_paired_burden_groups(panel_e_all)
    summary_rows: list[dict] = []
    histogram_rows: list[dict] = []
    bins = np.array([-np.inf, -0.20, -0.10, -0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08, 0.10, 0.20, np.inf])
    for regime in REGIMES:
        sub = data[
            (data["structure"] == "CN")
            & (data["regime"] == regime)
            & (data["paired_burden_group"] == "High")
        ]
        values = sub["dry_down_delta_r"].to_numpy(float)
        values = values[np.isfinite(values)]
        summary_rows.append(
            {
                "structure": "CN",
                "regime": regime,
                "burden_group": "High",
                "n_valid": int(len(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
                "fraction_abs_lt_0_03": float(np.mean(np.abs(values) < 0.03)),
                "fraction_abs_lt_0_05": float(np.mean(np.abs(values) < 0.05)),
                "fraction_abs_lt_0_08": float(np.mean(np.abs(values) < 0.08)),
                "fraction_delta_ge_0_05": float(np.mean(values >= 0.05)),
                "threshold_status": "descriptive; delta=0.05 was not demonstrably prespecified",
            }
        )
        counts, _ = np.histogram(values, bins=bins)
        for i, count in enumerate(counts):
            histogram_rows.append(
                {
                    "structure": "CN",
                    "regime": regime,
                    "burden_group": "High",
                    "bin_lower": float(bins[i]),
                    "bin_upper": float(bins[i + 1]),
                    "count": int(count),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(histogram_rows)



def _build_all_days_tgd_dpl_audit(
    results_root: Path,
    df_paired: pd.DataFrame,
    df_phase_tgd: pd.DataFrame,
    expected_basin_ids: list[str],
    expected_full_dates: pd.DatetimeIndex,
    tgd_parameter_invalid_ids: set[str],
 ) -> tuple[pd.DataFrame, dict]:
    """Reconcile all-days and active-melt TGD dPL undefined correlations."""
    paired = df_paired[df_paired["regime"] == CANONICAL_DPL]
    all_days_invalid = set(paired.loc[~np.isfinite(paired["delta_tgd2_base_anomaly"]), "basin_id"])
    active_rows = df_phase_tgd[
        (df_phase_tgd["regime"] == CANONICAL_DPL)
        & (df_phase_tgd["phase_name"] == PHASE_ORDER[1])
    ]
    active_invalid = set(active_rows.loc[~np.isfinite(active_rows["delta_tgd2_base_anomaly"]), "basin_id"])
    state_path = results_root / MODEL_ARRAY_FILES[CANONICAL_DPL]["TGD"]
    with np.load(state_path, allow_pickle=False) as archive:
        archive_ids = [str(x).zfill(8) for x in archive["basin_ids"]]
        if archive_ids != expected_basin_ids:
            raise ValueError(f"TGD dPL state basin ordering mismatch in {state_path}")
        archive_dates = pd.DatetimeIndex(pd.to_datetime(archive["dates"].astype(str)))
        if not archive_dates.equals(expected_full_dates):
            raise ValueError(f"TGD dPL state date axis mismatch in {state_path}")
        state_arrays = [archive[name] for name in ("wu", "wl", "wd")]
        expected_shape = (len(expected_basin_ids), len(expected_full_dates))
        if any(array.shape != expected_shape for array in state_arrays):
            raise ValueError(f"TGD dPL state-array shape mismatch in {state_path}")
        w_total = (state_arrays[0].astype(np.float64) + state_arrays[1].astype(np.float64) + state_arrays[2].astype(np.float64))
    rows: list[dict] = []
    index = {basin_id: i for i, basin_id in enumerate(expected_basin_ids)}
    for basin_id in sorted(all_days_invalid):
        values = np.asarray(w_total[index[basin_id]], dtype=float)
        finite = values[np.isfinite(values)]
        all_zero = bool(len(finite) and np.all(finite == 0.0))
        parameter_nonfinite = basin_id in tgd_parameter_invalid_ids
        rows.append(
            {
                "basin_id": basin_id,
                "all_days_invalid": True,
                "active_melt_invalid": basin_id in active_invalid,
                "parameter_nonfinite": parameter_nonfinite,
                "n_finite_state_values": int(len(finite)),
                "state_min": float(np.min(finite)) if len(finite) else float("nan"),
                "state_max": float(np.max(finite)) if len(finite) else float("nan"),
                "state_all_zero": all_zero,
                "state_constant": bool(len(finite) and np.ptp(finite) == 0.0),
                "insufficient_finite_overlap": False,
                "date_or_support_mismatch": False,
                "failure_mode": "non-finite physical parameter; legacy export is all-zero constant W_total" if parameter_nonfinite and all_zero else "other undefined all-days correlation",
            }
        )
    reconciliation = {
        "all_days_invalid_n": len(all_days_invalid),
        "active_melt_invalid_n": len(active_invalid),
        "intersection_ids": sorted(all_days_invalid & active_invalid),
        "all_days_only_ids": sorted(all_days_invalid - active_invalid),
        "active_melt_only_ids": sorted(active_invalid - all_days_invalid),
        "parameter_invalid_ids": sorted(tgd_parameter_invalid_ids),
        "all_days_invalid_equals_parameter_invalid": all_days_invalid == tgd_parameter_invalid_ids,
        "active_melt_invalid_is_parameter_invalid_subset": active_invalid.issubset(tgd_parameter_invalid_ids),
        "diagnostic_rule": "all-days undefined rows are parameter-invalid/all-zero W_total; active-melt support is the eligible subset of the same invalid parameter rows",
    }
    return pd.DataFrame(rows), reconciliation
def _build_orthogonality_table(full_long: pd.DataFrame, complete_assignments: pd.DataFrame, complete_bins: pd.DataFrame) -> pd.DataFrame:
    """Summarize absolute consistency versus paired effect on common support."""
    data = full_long.merge(
        complete_assignments[["basin_id", "phase_burden_group", "phase_burden_group_index"]],
        on="basin_id",
        how="inner",
        validate="many_to_one",
    )
    group_sizes = complete_bins.set_index("group_index")["n_basins"].to_dict()
    rows: list[dict] = []
    for structure in STRUCTURES:
        for regime in REGIMES:
            abs_col = STRUCTURE_CFG[structure]["absolute_col"]
            delta_col = STRUCTURE_CFG[structure]["delta_col"]
            for group_index, group in enumerate(PHASE_BIN_LABELS):
                sub = data[(data["regime"] == regime) & (data["phase_burden_group"] == group)]
                absolute = sub[abs_col].to_numpy(float)
                paired = sub[delta_col].to_numpy(float)
                valid = np.isfinite(absolute) & np.isfinite(paired)
                rows.append(
                    {
                        "structure": structure,
                        "regime": regime,
                        "burden_group": group,
                        "burden_group_index": group_index,
                        "absolute_median_r": float(np.nanmedian(absolute[valid])),
                        "paired_median_delta_r": float(np.nanmedian(paired[valid])),
                        "n_basins": int(valid.sum()),
                        "n_group": int(group_sizes[group_index]),
                    }
                )
    return pd.DataFrame(rows)

def _build_base_orthogonality_table(
    full_long: pd.DataFrame,
    complete_assignments: pd.DataFrame,
    complete_bins: pd.DataFrame,
 ) -> pd.DataFrame:
    """Provide Base absolute-r references for the panel (e) context strip."""
    data = full_long.merge(
        complete_assignments[["basin_id", "phase_burden_group", "phase_burden_group_index"]],
        on="basin_id",
        how="inner",
        validate="many_to_one",
    )
    group_sizes = complete_bins.set_index("group_index")["n_basins"].to_dict()
    rows: list[dict] = []
    for group_index, group in enumerate(PHASE_BIN_LABELS):
        sub = data[(data["phase_burden_group"] == group) & (data["regime"] == CANONICAL_DPL)]
        values = sub["base_anomaly_corr"].to_numpy(float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "structure": "Base",
                "burden_group": group,
                "burden_group_index": group_index,
                "absolute_median_r": float(np.median(finite)),
                "n_basins": int(len(finite)),
                "n_group": int(group_sizes[group_index]),
            }
        )
    return pd.DataFrame(rows)


def _build_statistics(
    df_paired: pd.DataFrame,
    df_phase_cn: pd.DataFrame,
    df_phase_tgd: pd.DataFrame,
 ) -> dict[str, pd.DataFrame | dict]:
    """Build every machine-readable table needed to reconstruct Figure 7."""
    full_assignments, full_bins = _build_full_burden_bins(df_paired)
    phase_assignments, phase_bins, phase_eligible_ids = _build_phase_burden_terciles(
        df_phase_cn, df_phase_tgd
    )
    phase_long_all = _phase_long_table(df_phase_cn, df_phase_tgd)
    complete_assignments, complete_bins, complete_ids = _build_complete_case_phase_terciles(
        phase_long_all, phase_assignments
    )
    phase_long = phase_long_all.merge(
        phase_assignments[["basin_id", "phase_burden_group", "phase_burden_group_index"]],
        on="basin_id",
        how="inner",
        validate="many_to_one",
    )
    phase_long_complete = phase_long_all.merge(
        complete_assignments[["basin_id", "phase_burden_group", "phase_burden_group_index"]],
        on="basin_id",
        how="inner",
        validate="many_to_one",
    )
    full_long = df_paired[df_paired["regime"].isin(REGIMES)].merge(
        full_assignments[["basin_id", "burden_group", "burden_group_index"]],
        on="basin_id",
        how="inner",
        validate="many_to_one",
    )

    panel_a = _build_panel_a_table(phase_long_complete)
    panel_a_available = _build_panel_a_table(phase_long)

    panel_b_rows: list[dict] = []
    group_sizes = full_bins.set_index("group_index")["n_basins"].to_dict()
    for regime in REGIMES:
        for structure in STRUCTURES:
            sub = full_long[full_long["regime"] == regime]
            delta_col = STRUCTURE_CFG[structure]["delta_col"]
            for group_index, group in enumerate(FULL_BIN_LABELS):
                vals = sub[sub["burden_group"] == group][delta_col].to_numpy(float)
                finite = _finite_values(vals)
                med, lo, hi, n = _summary_row(vals)
                q25, q75 = (np.nanpercentile(finite, [25, 75]) if len(finite) else (float("nan"), float("nan")))
                panel_b_rows.append(
                    {
                        "regime": regime,
                        "structure": structure,
                        "burden_group": group,
                        "burden_group_index": group_index,
                        "median_delta_r": med,
                        "ci_lower": lo,
                        "ci_upper": hi,
                        "iqr_lower": float(q25) if structure == "CN" and regime == CANONICAL_DPL else float("nan"),
                        "iqr_upper": float(q75) if structure == "CN" and regime == CANONICAL_DPL else float("nan"),
                        "n_basins": n,
                        "n_group": int(group_sizes[group_index]),
                    }
                )
    panel_b = pd.DataFrame(panel_b_rows)

    panel_c_rows: list[dict] = []
    for regime in REGIMES:
        for structure in STRUCTURES:
            sub = phase_long[(phase_long["regime"] == regime) & (phase_long["structure"] == structure)]
            for phase_name in PHASE_ORDER:
                for group in ("High", "Low"):
                    vals = sub[
                        (sub["phase_name"] == phase_name)
                        & (sub["phase_burden_group"] == group)
                    ]["delta_r"].to_numpy(float)
                    med, lo, hi, n = _summary_row(vals)
                    panel_c_rows.append(
                        {
                            "regime": regime,
                            "structure": structure,
                            "phase_name": phase_name,
                            "phase_label": PHASE_SHORT[phase_name],
                            "burden_endpoint": group,
                            "median_delta_r": med,
                            "ci_lower": lo,
                            "ci_upper": hi,
                            "n_basins": n,
                        }
                    )
    panel_c = pd.DataFrame(panel_c_rows)

    panel_d_rows: list[dict] = []
    for regime in REGIMES:
        for structure in STRUCTURES:
            active = phase_long_all[
                (phase_long_all["regime"] == regime)
                & (phase_long_all["structure"] == structure)
                & (phase_long_all["phase_name"] == PHASE_ORDER[1])
            ][["basin_id", "delta_r"]].dropna().sort_values("delta_r").reset_index(drop=True)
            vals = active["delta_r"].to_numpy(float)
            panel_d_rows.extend(
                {
                    "curve": f"active_melt_{structure}_{REGIME_CFG[regime]['short']}",
                    "condition": "Active melt",
                    "regime": regime,
                    "structure": structure,
                    "basin_id": row["basin_id"],
                    "delta_r": float(row["delta_r"]),
                    "ecdf_y": (i + 1) / len(active),
                    "positive_fraction": float(np.mean(vals > 0.0)),
                }
                for i, (_, row) in enumerate(active.iterrows())
            )
    dry_cn = phase_long_all[
        (phase_long_all["regime"] == CANONICAL_DPL)
        & (phase_long_all["structure"] == "CN")
        & (phase_long_all["phase_name"] == PHASE_ORDER[3])
    ][["basin_id", "delta_r"]].dropna().sort_values("delta_r").reset_index(drop=True)
    panel_d_rows.extend(
        {
            "curve": "dry_down_CN_dPL",
            "condition": "Dry-down",
            "regime": CANONICAL_DPL,
            "structure": "CN",
            "basin_id": row["basin_id"],
            "delta_r": float(row["delta_r"]),
            "ecdf_y": (i + 1) / len(dry_cn),
            "positive_fraction": float(np.mean(dry_cn["delta_r"].to_numpy(float) > 0.0)),
        }
        for i, (_, row) in enumerate(dry_cn.iterrows())
    )
    panel_d = pd.DataFrame(panel_d_rows)

    active = phase_long_all[
        (phase_long_all["regime"] == CANONICAL_DPL)
        & (phase_long_all["phase_name"] == PHASE_ORDER[1])
    ][["basin_id", "structure", "delta_r", "snow_burden_swe_mm"]].rename(
        columns={"delta_r": "active_delta_r"}
    )
    dry = phase_long_all[
        (phase_long_all["regime"] == CANONICAL_DPL)
        & (phase_long_all["phase_name"] == PHASE_ORDER[3])
    ][["basin_id", "structure", "delta_r"]].rename(columns={"delta_r": "dry_down_delta_r"})
    panel_e = active.merge(dry, on=["basin_id", "structure"], how="inner", validate="one_to_one")
    panel_e = panel_e.pivot(index=["basin_id", "snow_burden_swe_mm"], columns="structure", values=["active_delta_r", "dry_down_delta_r"]).reset_index()
    panel_e.columns = [
        "basin_id" if c[0] == "basin_id" else "snow_burden_swe_mm" if c[0] == "snow_burden_swe_mm" else f"{c[0]}_{c[1]}"
        for c in panel_e.columns
    ]
    panel_e = panel_e.rename(
        columns={
            "active_delta_r_TGD": "tgd_active_delta_r",
            "active_delta_r_CN": "cn_active_delta_r",
            "dry_down_delta_r_TGD": "tgd_dry_down_delta_r",
            "dry_down_delta_r_CN": "cn_dry_down_delta_r",
        }
    )
    panel_e_all = _build_joint_phase_table(phase_long_all)
    panel_b_forest = _build_forest_table(full_long, panel_c)
    panel_d_composition, panel_d_composition_sensitivity = _build_composition_tables(panel_e_all)
    panel_e_orthogonality = _build_orthogonality_table(full_long, complete_assignments, complete_bins)
    panel_e_base_context = _build_base_orthogonality_table(full_long, complete_assignments, complete_bins)
    panel_f_rows: list[dict] = []
    for regime in REGIMES:
        sub = full_long[full_long["regime"] == regime]
        for structure in ["Base", "TGD", "CN"]:
            absolute_col = "base_anomaly_corr" if structure == "Base" else STRUCTURE_CFG[structure]["absolute_col"]
            for group_index, group in enumerate(FULL_BIN_LABELS):
                vals = sub[sub["burden_group"] == group][absolute_col].to_numpy(float)
                med = float(np.nanmedian(vals))
                panel_f_rows.append(
                    {
                        "regime": regime,
                        "structure": structure,
                        "burden_group": group,
                        "burden_group_index": group_index,
                        "median_absolute_anomaly_corr": med,
                        "n_basins": int(np.isfinite(vals).sum()),
                        "n_group": int(group_sizes[group_index]),
                    }
                )
    panel_f = pd.DataFrame(panel_f_rows)

    eligibility_rows: list[dict] = []
    for regime in REGIMES:
        for structure in STRUCTURES:
            for phase_name in PHASE_ORDER:
                sub = phase_long_all[
                    (phase_long_all["regime"] == regime)
                    & (phase_long_all["structure"] == structure)
                    & (phase_long_all["phase_name"] == phase_name)
                ]
                eligibility_rows.append(
                    {
                        "regime": regime,
                        "structure": structure,
                        "phase_name": phase_name,
                        "n_basins": int(sub["basin_id"].nunique()),
                        "n_valid_delta": int(np.isfinite(sub["delta_r"].to_numpy(float)).sum()),
                        "n_days_total": int(sub["n_days"].sum()),
                        "external_phase_rule": "Caravan SWE only",
                    }
                )
    eligibility = pd.DataFrame(eligibility_rows)
    eligible_set = set(phase_eligible_ids)
    phase_group_map = phase_assignments.set_index("basin_id")["phase_burden_group"].to_dict()

    def missing_ids(regime: str, structure: str, phase_name: str) -> list[str]:
        sub = phase_long_all[
            (phase_long_all["regime"] == regime)
            & (phase_long_all["structure"] == structure)
            & (phase_long_all["phase_name"] == phase_name)
        ]
        finite_ids = set(sub.loc[np.isfinite(sub["delta_r"]), "basin_id"])
        return sorted(eligible_set - finite_ids)

    def missing_ids_within_phase_rows(regime: str, structure: str, phase_name: str) -> list[str]:
        sub = phase_long_all[
            (phase_long_all["regime"] == regime)
            & (phase_long_all["structure"] == structure)
            & (phase_long_all["phase_name"] == phase_name)
        ]
        finite_ids = set(sub.loc[np.isfinite(sub["delta_r"]), "basin_id"])
        return sorted(set(sub["basin_id"]) - finite_ids)

    tgd_dpl_active_missing = missing_ids(CANONICAL_DPL, "TGD", PHASE_ORDER[1])
    tgd_dpl_active_missing_all_phase_rows = missing_ids_within_phase_rows(CANONICAL_DPL, "TGD", PHASE_ORDER[1])
    tgd_ic_active_missing = missing_ids("IC_fused", "TGD", PHASE_ORDER[1])
    dropped_complete = sorted(eligible_set - set(complete_ids))
    support_audit = {
        "phase_eligible_all_phase_n": len(phase_eligible_ids),
        "panel_a_complete_case_n": len(complete_ids),
        "panel_a_complete_case_dropped_ids": dropped_complete,
        "panel_a_complete_case_burden_counts": complete_bins[["burden_group", "n_basins"]].to_dict("records"),
        "tgd_dpl_active_melt_missing_ids": tgd_dpl_active_missing,
        "tgd_dpl_active_melt_missing_ids_within_phase_rows": tgd_dpl_active_missing_all_phase_rows,
        "tgd_dpl_active_melt_missing_burden_bins": [
            {"basin_id": basin_id, "burden_group": phase_group_map.get(basin_id)}
            for basin_id in tgd_dpl_active_missing
        ],
        "tgd_ic_active_melt_degenerate_ids": tgd_ic_active_missing,
        "panel_e_paired_basin_n": int(len(panel_e)),
        "panel_e_valid_pair_n": {
            structure: int(
                (
                    np.isfinite(panel_e[f"{structure.lower()}_active_delta_r"]).to_numpy(bool)
                    & np.isfinite(panel_e[f"{structure.lower()}_dry_down_delta_r"]).to_numpy(bool)
                ).sum()
            )
            for structure in STRUCTURES
        },
        "panel_e_condition_specific_fraction": {
            structure: {
                "n": int(
                    (
                        np.isfinite(panel_e[f"{structure.lower()}_active_delta_r"]).to_numpy(bool)
                        & np.isfinite(panel_e[f"{structure.lower()}_dry_down_delta_r"]).to_numpy(bool)
                    ).sum()
                ),
                "count": int(
                    (
                        (panel_e[f"{structure.lower()}_active_delta_r"] > 0.1)
                        & (panel_e[f"{structure.lower()}_dry_down_delta_r"].abs() < 0.05)
                    )
                    .fillna(False)
                    .sum()
                ),
            }
            for structure in STRUCTURES
        },
        "composition_threshold_status": "descriptive; no prespecification record located",
        "composition_thresholds": {"tau": COMPOSITION_TAU, "delta": COMPOSITION_DELTA, "equivalence_band": DESCRIPTIVE_EQUIVALENCE_BAND},
        "panel_d_composition_rows": int(len(panel_d_composition)),
        "panel_d_composition_counts_sum_to_n": bool(
            (
                panel_d_composition[["melt_specific_count", "generic_count", "no_change_count", "mixed_count"]].sum(axis=1)
                == panel_d_composition["n"]
            ).all()
        ),
    }

    return {
        "full_assignments": full_assignments,
        "full_bins": full_bins,
        "phase_assignments": phase_assignments,
        "phase_bins": phase_bins,
        "complete_phase_assignments": complete_assignments,
        "complete_phase_bins": complete_bins,
        "panel_a": panel_a,
        "panel_a_available": panel_a_available,
        "panel_b": panel_b,
        "panel_b_forest": panel_b_forest,
        "panel_c": panel_c,
        "panel_d": panel_d,
        "panel_d_composition": panel_d_composition,
        "panel_d_composition_sensitivity": panel_d_composition_sensitivity,
        "panel_e": panel_e,
        "panel_e_all": panel_e_all,
        "panel_e_orthogonality": panel_e_orthogonality,
        "panel_e_base_context": panel_e_base_context,
        "panel_f": panel_f,
        "eligibility": eligibility,
        "common_phase_ids": phase_eligible_ids,
        "complete_phase_ids": complete_ids,
        "support_audit": support_audit,
    }


def _write_statistics(results_dir: Path, stats: dict, phase_cn: pd.DataFrame, phase_tgd: pd.DataFrame) -> Path:
    """Write reproducibility tables beside the existing canonical R4 results."""
    files = {
        "full_assignments": "figure7_full_basin_assignments.csv",
        "full_bins": "figure7_full_burden_bins.csv",
        "phase_assignments": "figure7_phase_eligible_assignments.csv",
        "phase_bins": "figure7_phase_burden_terciles.csv",
        "complete_phase_assignments": "figure7_phase_complete_case_assignments.csv",
        "complete_phase_bins": "figure7_phase_complete_case_bins.csv",
        "panel_a": "figure7_panel_a_phase_burden_matrix.csv",
        "panel_a_available": "figure7_panel_a_available_case.csv",
        "panel_b": "figure7_panel_b_full_basin_gradient.csv",
        "panel_b_forest": "figure7_panel_b_forest.csv",
        "panel_c": "figure7_panel_c_phase_endpoints.csv",
        "panel_d": "figure7_panel_d_ecdf_values.csv",
        "panel_d_composition": "figure7_panel_d_composition.csv",
        "panel_d_composition_sensitivity": "figure7_panel_d_composition_sensitivity.csv",
        "panel_d_tgd_dpl_high_support": "figure7_tgd_dpl_high_support.csv",
        "panel_d_tgd_dpl_support_by_group": "figure7_tgd_dpl_support_by_burden.csv",
        "panel_d_tgd_dpl_conservative": "figure7_tgd_dpl_high_conservative_denominator.csv",
        "panel_d_cn_high_drydown_summary": "figure7_cn_high_drydown_summary.csv",
        "panel_d_cn_high_drydown_histogram": "figure7_cn_high_drydown_histogram.csv",
        "all_days_tgd_dpl_audit": "figure7_all_days_tgd_dpl_invalid_audit.csv",
        "panel_e": "figure7_panel_e_active_drydown_joint.csv",
        "panel_e_all": "figure7_panel_e_active_drydown_all_regimes.csv",
        "panel_e_orthogonality": "figure7_panel_e_orthogonality.csv",
        "panel_e_base_context": "figure7_panel_e_base_context.csv",
        "panel_f": "figure7_panel_f_absolute_context.csv",
        "eligibility": "figure7_phase_eligibility.csv",
    }
    for key, filename in files.items():
        stats[key].to_csv(results_dir / filename, index=False)
    phase_rows = []
    for frame, structure, delta_col in [
        (phase_cn, "CN", "delta_anomaly_corr"),
        (phase_tgd, "TGD", "delta_tgd2_base_anomaly"),
    ]:
        block = frame.copy()
        block["structure"] = structure
        block["delta_r"] = block[delta_col]
        phase_rows.append(block)
    pd.concat(phase_rows, ignore_index=True, sort=False).to_csv(
        results_dir / "figure7_external_phase_rows.csv", index=False
    )

    phase_source_counts = []
    for frame, structure, delta_col in [
        (phase_cn, "CN", "delta_anomaly_corr"),
        (phase_tgd, "TGD", "delta_tgd2_base_anomaly"),
    ]:
        for regime in REGIMES:
            for phase_name in PHASE_ORDER:
                sub = frame[(frame["regime"] == regime) & (frame["phase_name"] == phase_name)]
                phase_source_counts.append(
                    {
                        "regime": regime,
                        "structure": structure,
                        "phase_name": phase_name,
                        "n_rows": int(len(sub)),
                        "n_basins": int(sub["basin_id"].nunique()),
                        "delta_column": delta_col,
                        "source": "canonical Base/CN/TGD full arrays recomputed under current Caravan SWE mask",
                    }
                )

    metadata = {
        "figure": "Figure7_R4",
        "status": "interim_tgd2_provenance_qualified",
        "architecture": "four_panel_overview_effect_heterogeneity_composition",
        "main_panels": ["a", "b", "c", "d"],
        "omitted_main_panel": "e (absolute-versus-paired audit retained in exported tables)",
        "bootstrap": {"method": "basin bootstrap median CI", "rounds": int(BOOTSTRAP_ROUNDS), "seed": int(BOOTSTRAP_SEED)},
        "support_audit": stats["support_audit"],
        "canvas": {"width_cm": FIG_W_CM, "height_cm": FIG_H_CM, "dpi": FIG_DPI, "target_print_width_cm": 12.0, "savefig_bbox": "tight"},
        "panel_calibration": {
            "panel_a_matrix_scale": {"type": "neutral_sequential_positive_magnitude", "vmin": 0.0, "vmax": MATRIX_VMAX, "negative_cell_cue": "subtle neutral hatch", "display_numeric_threshold": 0.10},
            "panel_b_xlim": [-0.15, 0.78],
            "panel_c_xlim": [-0.8, 1.08],
            "panel_d_composition": {"xlim_pct": [0, 100], "bar_data_xlim_pct": [0, 100], "thresholds": "descriptive"},
        },
        "threshold_provenance": {
            "equivalence_band": {"value": DESCRIPTIVE_EQUIVALENCE_BAND, "status": "descriptive; no prespecification record located"},
            "composition_tau": {"value": COMPOSITION_TAU, "status": "descriptive; no prespecification record located"},
            "composition_delta": {"value": COMPOSITION_DELTA, "status": "descriptive; no prespecification record located"},
            "sensitivity_grid": {"tau": [0.05, 0.10, 0.15], "delta": [0.03, 0.05, 0.08]},
        },
        "external_phase_rule": {
            "source": "results/r4_caravan_soil_reference_v1/caravan_soil_ensemble.npz",
            "swe_variable": "caravan_swe",
            "threshold_mm": MIN_EXTERNAL_SWE_MM,
            "basin_burden_eligibility": {"source": "paired structural effects snow_burden_swe_mm", "threshold_mm": MIN_BASIN_BURDEN_MM, "rule": "burden >= threshold"},
            "annual_peak": True,
            "melt_out": "first post-peak SWE < threshold",
            "minimum_phase_days": MIN_PHASE_DAYS,
            "model_state_inputs_used_for_eligibility": [],
            "calendar_only_primary_rule": False,
            "cn_phase_statistics_rebuilt_from_current_caravan_masks": True,
            "upstream_phase_table_audited_but_not_used_for_panel_values": True,
            "upstream_phase_table_note": "Stored phase-day support differed from the current Caravan archive; Figure 7 recomputes CN and TGD phase rows from canonical arrays under current Caravan SWE masks.",
        },
        "tgd_provenance": {
            "paper_label": "TGD* (interim)",
            "disk_label": "TGD2",
            "model_class": "XAJWithTGD2Lite",
            "structure_version": "temperature_dependent_generic_delay2_v1",
            "state_quantity": "W_total = wu + wl + wd",
            "dPL_seed42": {
                "array": "results/" + str(MODEL_ARRAY_FILES[CANONICAL_DPL]["TGD"]),
                "parameter_source": "results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_42/best_parameters_physical.npz",
                "checkpoint_source": "results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2/seed_42/best_checkpoint.pt",
                "config": "training/dpl/generated_configs/xaj_tgd2_lite_v3_seed_42.json",
                "seed": 42,
                "selection": "best validation-median-KGE checkpoint recorded in best_checkpoint.pt (epoch 50); available epoch history ends at epoch 60 and does not contain COMPLETE/final summary",
                "qualification": "training-incomplete interim source; no numerical substitution performed",
                "nonfinite_parameter_rows_marked_invalid": True,
                "export_cleanup_note": "The legacy exporter converted NaN states to zero; Figure 7 now identifies non-finite parameter rows and marks their exported states invalid instead of treating zero as a valid W_total.",
            },
            "IC_fused": {
                "array": "results/" + str(MODEL_ARRAY_FILES["IC_fused"]["TGD"]),
                "parameter_source": "results/r4_ic_fused_XAJ_TGD2/best_parameters_physical.npz",
                "selection_source": "results/r4_ic_fused_XAJ/ic_fused_XAJ_manifest.json and ic_cmaes_recalibration_p20_p25_20260728_fused/XAJ/per_start.csv",
                "selection": "best train-period KGE restart per basin; lowest start breaks ties",
                "starts": 5,
            },
            "alignment": {"n_basins": 531, "n_full_days": 12418, "test_period": "1995-10-01..2010-09-30", "basin_date_axis_checked": True},
        },
        "phase_source_counts": phase_source_counts,
        "common_phase_tercile_population": {
            "n_basins": len(stats["common_phase_ids"]),
            "phase_support": "all four phases for both selected regimes and structures",
        },
        "panel_a_complete_case_population": {
            "n_basins": len(stats["complete_phase_ids"]),
            "burden_bins": stats["complete_phase_bins"][["burden_group", "n_basins"]].to_dict("records"),
            "support": "finite paired contrasts in all four structure-by-regime blocks across all four external phases",
        },
        "caption_support_notes": [
            "ERA5-Land SM100 is a model-derived external process-state consistency reference, not truth.",
            "External Snow-17/Caravan SWE defines burden and phase conditions; no model state defines eligibility.",
            "Phase rows use the basin-level eligibility gate snow_burden_swe_mm >= 20 mm; this is separate from the daily SWE >= 5 mm phase threshold.",
            "Panel (a) uses a 331-basin common-support complete-case mask across all four blocks and all four phases; the 344-basin available-case table is retained separately.",
            "Panel (b) summarizes all-days and high/low phase-conditioned effects without connecting categorical phases by trend lines.",
            "Panel (c) shows population ECDFs; the gray dry-down curve is the complementary-condition reference.",
            "Panel (d) uses descriptive, not prespecified, tau=0.10 and delta=0.05 composition thresholds; the sensitivity grid is retained separately.",
            "CN High-burden dry-down distributions are audited separately for IC and dPL; the delta=0.05 rule cuts through the distribution body and is not treated as a confirmatory boundary.",
            "TGD dPL High composition is valid-support n=109 of 117 eligible basins; eight invalid rows are shown in conservative denominator sensitivity only.",
            "The 16 all-days TGD dPL undefined rows equal the non-finite-parameter set; 11 intersect active-melt invalid rows and five are all-days-only because they are not active-melt eligible.",
            "Absolute-versus-paired statistics are retained in exported audit tables but omitted from the four-panel main figure.",
            "Invalid dPL-TGD parameter/state rows and constant IC-TGD rows are excluded without imputation and are not interpreted mechanistically.",
            "Base defines zero for paired contrasts; color is structure, IC is filled/solid, and dPL is open/dashed.",
            "No CN−TGD main estimand is generated.",
        ],
    }
    metadata_path = results_dir / "figure7_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata_path


def _plot_panel_a(fig, panel_spec, cells: pd.DataFrame) -> None:
    """Draw panel (a) as four aligned heatmap axes with shared annotations."""
    outer = panel_spec.subgridspec(
        7,
        1,
        height_ratios=(0.20, 3.40, 0.22, 0.16, 0.10, 0.26, 0.20),
        hspace=0.12,
    )
    heat_grid = outer[1, 0].subgridspec(2, 2, wspace=0.06, hspace=0.07)
    norm = mcolors.Normalize(vmin=0.0, vmax=MATRIX_VMAX, clip=False)
    panel_title_ax = fig.add_subplot(outer[0, 0])
    panel_title_ax.set_axis_off()
    panel_title_ax.set_title("(a) Conditional interaction overview", loc="left", pad=6, fontsize=13, fontweight="bold")
    heat_axes = []
    phase_labels = ["Accumulation", "Active melt", "Post-melt", "Dry-down"]
    for row_index, regime in enumerate(REGIMES):
        row_axes = []
        for col_index, structure in enumerate(STRUCTURES):
            ax = fig.add_subplot(heat_grid[row_index, col_index])
            row_axes.append(ax)
            sub = cells[(cells["structure"] == structure) & (cells["regime"] == regime)]
            matrix = np.zeros((len(PHASE_ORDER), len(PHASE_BIN_LABELS)), dtype=float)
            negative = np.zeros_like(matrix, dtype=bool)
            values = np.full_like(matrix, np.nan)
            for item in sub.itertuples(index=False):
                phase_index = PHASE_ORDER.index(item.phase_name)
                burden_index = int(item.burden_group_index)
                value = float(item.median_delta_r)
                values[phase_index, burden_index] = value
                if np.isfinite(value) and value >= DESCRIPTIVE_EQUIVALENCE_BAND:
                    matrix[phase_index, burden_index] = value
                elif np.isfinite(value) and value <= -DESCRIPTIVE_EQUIVALENCE_BAND:
                    negative[phase_index, burden_index] = True
            ax.imshow(matrix, cmap=MATRIX_CMAP, norm=norm, aspect="auto", interpolation="nearest", origin="upper")
            for phase_index, burden_index in zip(*np.where(negative)):
                ax.add_patch(Rectangle(
                    (burden_index - 0.5, phase_index - 0.5),
                    1.0,
                    1.0,
                    facecolor="#F2F3F3",
                    edgecolor="#C5C9CB",
                    linewidth=0.35,
                    hatch="///",
                    zorder=2,
                ))
            ax.set_xticks(np.arange(-0.5, len(PHASE_BIN_LABELS), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(PHASE_ORDER), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlim(-0.5, len(PHASE_BIN_LABELS) - 0.5)
            ax.set_ylim(len(PHASE_ORDER) - 0.5, -0.5)
            if row_index == 0:
                ax.xaxis.set_ticks_position("top")
                ax.set_xticks(range(len(PHASE_BIN_LABELS)), PHASE_BIN_LABELS)
                ax.tick_params(axis="x", labeltop=True, labelbottom=False, labelsize=9.5, length=3.0, width=0.8, direction="out", pad=1.5)
            else:
                ax.set_xticks([])
            if col_index == 0:
                ax.set_yticks(range(len(PHASE_ORDER)), phase_labels)
                ax.tick_params(axis="y", labelsize=9.5, length=3.0, width=0.8, direction="out", pad=3)
            else:
                ax.set_yticks([])
                ax.annotate(
                    REGIME_CFG[regime]["short"],
                    xy=(1.06, 0.5),
                    xycoords="axes fraction",
                    rotation=270,
                    ha="left",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=COLOR_DARK_NEUTRAL,
                    annotation_clip=False,
                )
            for phase_index, burden_index in np.ndindex(values.shape):
                value = values[phase_index, burden_index]
                if np.isfinite(value) and abs(value) >= 0.10:
                    text_color = "white" if value >= 0.35 else COLOR_DARK_NEUTRAL
                    ax.text(burden_index, phase_index, f"{value:+.2f}", ha="center", va="center", fontsize=9, color=text_color, zorder=3)
        heat_axes.append(row_axes)
    structure_grid = outer[2, 0].subgridspec(1, 2, wspace=0.06)
    for col_index, structure in enumerate(STRUCTURES):
        str_ax = fig.add_subplot(structure_grid[0, col_index])
        str_ax.set_axis_off()
        str_ax.text(
            0.5,
            0.50,
            STRUCTURE_CFG[structure]["label"],
            ha="center",
            va="center",
            color=STRUCTURE_CFG[structure]["color"],
            fontsize=10.5,
            fontweight="bold",
        )
    cbar_ax = fig.add_subplot(outer[3, 0])
    sm = ScalarMappable(norm=norm, cmap=MATRIX_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.2, 0.4, MATRIX_VMAX])
    cbar.ax.tick_params(labelsize=9, length=2, pad=1)
    spacer_ax = fig.add_subplot(outer[4, 0])
    spacer_ax.set_axis_off()
    legend_ax = fig.add_subplot(outer[5, 0])
    legend_ax.set_axis_off()
    hatch_handles = [
        Patch(facecolor=COLOR_DARK_NEUTRAL, edgecolor=COLOR_DARK_NEUTRAL, label="positive"),
        Patch(facecolor="#FFFFFF", edgecolor="#BFC4C7", label="equivalence"),
        Patch(facecolor="#F2F3F3", edgecolor="#C5C9CB", hatch="///", label="negative"),
    ]
    legend_ax.legend(
        handles=hatch_handles,
        frameon=False,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.5, 0.50),
        fontsize=8.0,
        columnspacing=1.20,
        handlelength=1.35,
        handletextpad=0.45,
        borderaxespad=0.0,
        labelspacing=0.4,
    )
    label_ax = fig.add_subplot(outer[6, 0])
    label_ax.set_axis_off()
    label_ax.text(0.5, 0.30, "External SWE burden tertile", ha="center", va="center", fontsize=11, clip_on=False)
def _plot_panel_b(ax, data: pd.DataFrame) -> None:
    """Draw the condition-resolved forest with spaced rows and dodged regimes."""
    apply_clean_spines(ax)
    sections = [
        ("All days", [(0, "All days")]),
        ("High burden", [
            (1, "Accumulation"),
            (3, "Active melt"),
            (5, "Post-melt"),
            (7, "Dry-down"),
        ]),
        ("Low burden", [
            (2, "Accumulation"),
            (4, "Active melt"),
            (6, "Post-melt"),
            (8, "Dry-down"),
        ]),
    ]
    y_for_row = {}
    row_labels_map = {}
    section_spans = {}
    cursor = 0.0
    row_step = 1.00
    section_gap = 0.80
    for sec_name, rows in reversed(sections):
        sec_y_values = []
        for row_idx, label in reversed(rows):
            y_for_row[row_idx] = cursor
            row_labels_map[row_idx] = label
            sec_y_values.append(cursor)
            cursor += row_step
        section_spans[sec_name] = (min(sec_y_values), max(sec_y_values))
        cursor += section_gap
    max_y = cursor - section_gap
    bg_colors = {
        "All days": ("#F6F8FA", 0.50),
        "High burden": ("#EEF2F5", 0.60),
        "Low burden": ("#F6F8FA", 0.50),
    }
    for sec_name, (y_min, y_max) in section_spans.items():
        bg_col, bg_alpha = bg_colors[sec_name]
        ax.axhspan(y_min - 0.42, y_max + 0.42, color=bg_col, alpha=bg_alpha, zorder=0)
    ax.axvspan(-DESCRIPTIVE_EQUIVALENCE_BAND, DESCRIPTIVE_EQUIVALENCE_BAND, color=COLOR_LIGHT_REF, alpha=0.16, zorder=0)
    regime_offsets = {"IC_fused": 0.22, CANONICAL_DPL: -0.22}
    structure_offsets = {"TGD": -0.08, "CN": 0.08}
    order = [("TGD", "IC_fused"), ("TGD", CANONICAL_DPL), ("CN", "IC_fused"), ("CN", CANONICAL_DPL)]
    all_row_indices = [idx for sec in sections for idx, _ in sec[1]]
    for row_index in all_row_indices:
        row_data = data[data["row_index"] == row_index]
        y0 = y_for_row[row_index]
        ax.axhline(y0, color="#E6EAEB", lw=0.5, ls=":", zorder=0)
        for structure, regime in order:
            point = row_data[(row_data["structure"] == structure) & (row_data["regime"] == regime)].iloc[0]
            x = float(point["median_delta_r"])
            lo = float(point["ci_lower"])
            hi = float(point["ci_upper"])
            color = STRUCTURE_CFG[structure]["color"]
            filled = regime == "IC_fused"
            y = y0 + structure_offsets[structure] + regime_offsets[regime]
            ax.errorbar(
                x,
                y,
                xerr=[[x - lo], [hi - x]],
                fmt="o",
                color=color,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=5.0,
                elinewidth=1.0,
                capsize=2.0,
                linestyle="None",
                zorder=3,
            )
    # Dashed separator lines between categories within High burden and Low burden
    for sec_name, rows in sections:
        if sec_name in ["High burden", "Low burden"]:
            sec_y_list = [y_for_row[idx] for idx, _ in rows]
            for k in range(len(sec_y_list) - 1):
                y_div = (sec_y_list[k] + sec_y_list[k + 1]) / 2.0
                ax.axhline(y_div, color="#D5DBDF", lw=0.6, ls="--", zorder=1)

    div_low_high = (section_spans["Low burden"][1] + section_spans["High burden"][0]) / 2.0
    div_high_all = (section_spans["High burden"][1] + section_spans["All days"][0]) / 2.0
    ax.axhline(div_low_high, color="#CAD0D5", lw=0.6, ls="-", zorder=1)
    ax.axhline(div_high_all, color="#CAD0D5", lw=0.6, ls="-", zorder=1)
    ax.axvline(0, color=COLOR_ZERO_LINE, lw=0.8, ls="--", zorder=1)
    sorted_row_indices = sorted(all_row_indices, key=lambda idx: y_for_row[idx])
    ax.set_yticks([y_for_row[idx] for idx in sorted_row_indices])
    ax.set_yticklabels([row_labels_map[idx] for idx in sorted_row_indices], fontsize=9.5)
    for sec_name, (y_min, y_max) in section_spans.items():
        if sec_name != "All days":
            short_sec = "High" if "High" in sec_name else "Low"
            ax.text(
                1.02,
                (y_min + y_max) / 2.0,
                short_sec,
                transform=ax.get_yaxis_transform(),
                rotation=270,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=COLOR_DARK_NEUTRAL,
                clip_on=False,
            )
    ax.tick_params(axis="y", pad=3, length=3.0, width=0.8, direction="out")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(-0.15, 0.78)
    ax.set_ylim(-0.6, max_y + 0.6)
    ax.set_xlabel(DELTA_LABEL, fontsize=11)
    ax.set_title("(b) Condition-resolved effect sizes", loc="left", pad=6, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.25, color=COLOR_LIGHT_REF)
    legend_handles = [
        Line2D([0], [0], color=COLOR_TGD, marker="o", markerfacecolor=COLOR_TGD, markeredgecolor=COLOR_TGD, markersize=4.5, lw=1.0, label="TGD* − Base"),
        Line2D([0], [0], color=COLOR_CN, marker="o", markerfacecolor=COLOR_CN, markeredgecolor=COLOR_CN, markersize=4.5, lw=1.0, label="CN − Base"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, marker="o", markerfacecolor=COLOR_DARK_NEUTRAL, markeredgecolor=COLOR_DARK_NEUTRAL, markersize=4.5, lw=1.0, label="IC"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, marker="o", markerfacecolor="white", markeredgecolor=COLOR_DARK_NEUTRAL, markersize=4.5, lw=1.0, label="dPL"),
        Patch(facecolor=COLOR_LIGHT_REF, edgecolor="none", alpha=0.16, label="|Δr| < 0.02"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=True,
        facecolor="white",
        edgecolor="#E0E4E7",
        framealpha=0.92,
        fontsize=7.5,
        ncol=1,
        loc="upper right",
        handlelength=1.1,
        borderpad=0.35,
        labelspacing=0.25,
    )

def _plot_panel_phase_endpoints(ax, data: pd.DataFrame) -> None:
    apply_clean_spines(ax)
    x = np.arange(len(PHASE_ORDER), dtype=float)
    offsets = {
        ("TGD", "IC_fused"): -0.12,
        ("TGD", CANONICAL_DPL): -0.04,
        ("CN", "IC_fused"): 0.04,
        ("CN", CANONICAL_DPL): 0.12,
    }
    ax.axvspan(0.5, 1.5, color="#EEF2F4", alpha=0.65, zorder=0)
    for endpoint, alpha, lw, zorder in [("Low", 0.30, 0.8, 2), ("High", 0.95, 1.15, 4)]:
        for structure in STRUCTURES:
            for regime in REGIMES:
                sub = data[
                    (data["structure"] == structure)
                    & (data["regime"] == regime)
                    & (data["burden_endpoint"] == endpoint)
                ].sort_values("phase_name", key=lambda s: s.map({p: i for i, p in enumerate(PHASE_ORDER)}))
                y = sub["median_delta_r"].to_numpy(float)
                lo = sub["ci_lower"].to_numpy(float)
                hi = sub["ci_upper"].to_numpy(float)
                cfg = REGIME_CFG[regime]
                scfg = STRUCTURE_CFG[structure]
                ax.errorbar(
                    x + offsets[(structure, regime)],
                    y,
                    yerr=[y - lo, hi - y],
                    color=scfg["color"],
                    linestyle=cfg["ls"],
                    marker="o",
                    lw=lw,
                    ms=2.8 if endpoint == "Low" else 3.3,
                    capsize=1.5,
                    elinewidth=0.65,
                    alpha=alpha,
                    zorder=zorder,
                )
    ax.axhline(0, color=COLOR_ZERO_LINE, lw=0.75, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE_TICK_LABELS, fontsize=5.0)
    ax.set_xlabel("External SWE phase", fontsize=5.8)
    ax.set_ylabel(DELTA_LABEL, fontsize=6.0)
    ax.set_ylim(-0.10, 0.62)
    ax.grid(True, axis="y", linestyle=":", alpha=0.28, color=COLOR_LIGHT_REF)
    ax.set_title("(c) Phase-resolved contrast\n(phase-eligible basins)", loc="left", pad=4, fontsize=7.8, fontweight="bold")
    legend_handles = [
        Line2D([0], [0], color=COLOR_TGD, lw=1.1, label="TGD* − Base"),
        Line2D([0], [0], color=COLOR_CN, lw=1.1, label="CN − Base"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, lw=1.0, ls="-", marker="o", ms=2.7, label="IC"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, lw=1.0, ls=REGIME_CFG[CANONICAL_DPL]["ls"], marker="o", ms=2.7, label="dPL"),
        Line2D([0], [0], color=COLOR_TGD, lw=1.2, alpha=0.95, marker="o", ms=2.7, label="High burden"),
        Line2D([0], [0], color=COLOR_TGD, lw=1.0, alpha=0.30, marker="o", ms=2.7, label="Low burden"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=3.9, ncol=2, loc="upper right", handlelength=1.4, columnspacing=0.6)


def _plot_panel_c(ax, data: pd.DataFrame) -> None:
    apply_clean_spines(ax)
    curves = []
    for structure in STRUCTURES:
        for regime in REGIMES:
            curve = f"active_melt_{structure}_{REGIME_CFG[regime]['short']}"
            sub = data[data["curve"] == curve].sort_values("delta_r")
            if len(sub):
                curves.append(
                    (
                        sub["delta_r"].to_numpy(float),
                        sub["ecdf_y"].to_numpy(float),
                        STRUCTURE_CFG[structure]["color"],
                        REGIME_CFG[regime]["ls"],
                        f"{STRUCTURE_CFG[structure]['label']} · {REGIME_CFG[regime]['short']}",
                    )
                )
    dry = data[data["curve"] == "dry_down_CN_dPL"].sort_values("delta_r")
    if len(dry):
        curves.append(
            (
                dry["delta_r"].to_numpy(float),
                dry["ecdf_y"].to_numpy(float),
                "#70777D",
                REGIME_CFG[CANONICAL_DPL]["ls"],
                "CN − Base · dry-down · dPL",
            )
        )
    for values, y, color, ls, label in curves:
        ax.step(values, y, where="post", color=color, ls=ls, lw=1.2 if "dry-down" in label else 1.1, alpha=0.95, label=label)
    ax.axvline(0, color=COLOR_ZERO_LINE, lw=0.8, ls="--")
    ax.set_xlim(-0.8, 1.08)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(DELTA_LABEL, fontsize=11)
    ax.set_ylabel("ECDF", fontsize=11)
    ax.set_title("(c) Heterogeneity", loc="left", pad=6, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.25, color=COLOR_LIGHT_REF)
    fraction_lines = []
    for structure in STRUCTURES:
        parts = []
        for regime in REGIMES:
            sub = data[data["curve"] == f"active_melt_{structure}_{REGIME_CFG[regime]['short']}"]
            parts.append(f"{REGIME_CFG[regime]['short']} {100 * float(sub['positive_fraction'].iloc[0]):.0f}%")
        fraction_lines.append(f"{STRUCTURE_DISPLAY[structure]}: " + " · ".join(parts))
    ax.text(0.03, 0.90, "\n".join(fraction_lines), transform=ax.transAxes, fontsize=9, color=COLOR_DARK_NEUTRAL, va="top")
    ax.text(0.63, 0.80, "dry-down\n(complementary)", transform=ax.transAxes, fontsize=9, color="#6F777D", ha="left", va="center")
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#E0E4E7",
        framealpha=0.92,
        fontsize=7.5,
        ncol=1,
        loc="lower right",
        handlelength=1.4,
        borderpad=0.35,
        labelspacing=0.25,
    )


def _plot_panel_d(ax, data: pd.DataFrame) -> None:
    """Draw burden-stratified response composition with separated groups."""
    apply_clean_spines(ax)
    category_order = ["melt_specific", "generic", "mixed", "no_change"]
    category_labels = {
        "melt_specific": "Melt-specific",
        "generic": "Generic",
        "mixed": "Mixed",
        "no_change": "No change",
    }
    category_colors = {
        "melt_specific": COLOR_BASE,
        "generic": COLOR_TGD,
        "mixed": COLOR_CN,
        "no_change": "#D9D9D9",
    }
    row_order = [(group, structure, regime) for group in PHASE_BIN_LABELS for structure in ("CN", "TGD") for regime in REGIMES]
    y_positions = {}
    group_bounds = {}
    cursor = 0.0
    group_gap = 0.6
    for group in PHASE_BIN_LABELS:
        group_keys = [key for key in row_order if key[0] == group]
        start = cursor
        for key in group_keys:
            y_positions[key] = cursor
            cursor += 1.0
        group_bounds[group] = (start, cursor - 1.0)
        cursor += group_gap
    max_y = cursor - group_gap
    for group_index, group in enumerate(PHASE_BIN_LABELS):
        start, end = group_bounds[group]
        if group_index % 2 == 1:
            ax.axhspan(start - 0.4, end + 0.4, color=COLOR_LIGHT_REF, alpha=0.04, zorder=0)
    for key in row_order:
        group, structure, regime = key
        sub = data[(data["burden_group"] == group) & (data["structure"] == structure) & (data["regime"] == regime)].iloc[0]
        y = y_positions[key]
        left = 0.0
        for category in category_order:
            width = float(sub[f"{category}_pct"])
            ax.barh(y, width, left=left, height=0.72, color=category_colors[category], edgecolor="white", linewidth=0.45, zorder=2)
            if width >= 8.0:
                ax.text(left + width / 2.0, y, f"{width:.0f}%", ha="center", va="center", fontsize=9, color=COLOR_DARK_NEUTRAL, zorder=2)
            left += width
    ax.axvline(50.0, color=COLOR_LIGHT_REF, lw=0.6, ls=":", zorder=0)
    ax.set_yticks([y_positions[key] for key in row_order])
    ax.set_yticklabels([f"{STRUCTURE_DISPLAY[structure]} · {REGIME_CFG[regime]['short']}" for _, structure, regime in row_order], fontsize=9.5)
    ax.tick_params(axis="y", length=0, pad=3)
    for group, (start, end) in group_bounds.items():
        ax.text(
            102.5,
            (start + end) / 2.0,
            group,
            rotation=270,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=COLOR_DARK_NEUTRAL,
            clip_on=False,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, max_y + 0.5)
    ax.invert_yaxis()
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Valid paired basins (%)", fontsize=11)
    ax.set_title("(d) Response composition", loc="left", pad=6, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.25, color=COLOR_LIGHT_REF, zorder=1)
    category_handles = [Patch(facecolor=category_colors[key], edgecolor="white", label=category_labels[key]) for key in category_order]
    ax.legend(
        handles=category_handles,
        frameon=True,
        facecolor="white",
        edgecolor="#E0E4E7",
        framealpha=0.95,
        fontsize=7.4,
        ncol=4,
        loc="center",
        bbox_to_anchor=(0.5, 0.032),
        handlelength=0.9,
        handletextpad=0.3,
        borderpad=0.25,
        columnspacing=0.6,
    )

def _plot_panel_e(ax, data: pd.DataFrame, base_context: pd.DataFrame) -> None:
    """Plot absolute external-state consistency against paired structural effect."""
    apply_clean_spines(ax)
    burden_short = {"Low": "L", "Middle": "M", "High": "H"}
    ordered = data.sort_values(["structure", "regime", "burden_group_index"])
    for (structure, regime), sub in ordered.groupby(["structure", "regime"], sort=False):
        sub = sub.sort_values("burden_group_index")
        ax.plot(
            sub["absolute_median_r"],
            sub["paired_median_delta_r"],
            color=STRUCTURE_CFG[structure]["color"],
            lw=0.55,
            alpha=0.20,
            zorder=1,
        )
    for row in ordered.itertuples(index=False):
        filled = row.regime == "IC_fused"
        color = STRUCTURE_CFG[row.structure]["color"]
        ax.plot(
            row.absolute_median_r,
            row.paired_median_delta_r,
            marker="o",
            linestyle="None",
            markersize=3.4,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            zorder=3,
        )
        ax.text(
            row.absolute_median_r,
            row.paired_median_delta_r + 0.008,
            burden_short[row.burden_group],
            ha="center",
            va="bottom",
            fontsize=3.1,
            color=COLOR_DARK_NEUTRAL,
            zorder=4,
        )
    base_y = -0.062
    base = base_context.sort_values("burden_group_index")
    ax.plot(base["absolute_median_r"], [base_y] * len(base), color=COLOR_BASE, lw=0.7, alpha=0.85, zorder=2)
    for row in base.itertuples(index=False):
        ax.plot(row.absolute_median_r, base_y, marker="|", markersize=7, markeredgewidth=0.9, color=COLOR_BASE, zorder=3)
        ax.text(row.absolute_median_r, base_y + 0.006, burden_short[row.burden_group], ha="center", va="bottom", fontsize=3.0, color=COLOR_BASE)
    ax.text(0.505, base_y - 0.005, "Base", ha="left", va="top", fontsize=3.5, color=COLOR_BASE)
    ax.axhline(0, color=COLOR_ZERO_LINE, lw=0.75, ls="--")
    ax.set_xlim(0.50, 0.90)
    ax.set_ylim(-0.08, 0.20)
    ax.set_xlabel("Absolute median r(W, SM100)", fontsize=5.8)
    ax.set_ylabel("Paired Δr", fontsize=5.8, labelpad=4)
    ax.yaxis.set_label_coords(-0.11, 0.5)
    ax.set_title("(e) Absolute vs paired", loc="left", pad=4, fontsize=7.5, fontweight="bold")
    ax.text(0.98, 0.96, "context only", transform=ax.transAxes, ha="right", va="top", fontsize=4.2, color=COLOR_DARK_NEUTRAL)
    n_lines = []
    for structure in STRUCTURES:
        for regime in REGIMES:
            sub = ordered[(ordered["structure"] == structure) & (ordered["regime"] == regime)].sort_values("burden_group_index")
            values = "/".join(str(int(value)) for value in sub["n_basins"])
            n_lines.append(f"{STRUCTURE_DISPLAY[structure]}·{REGIME_CFG[regime]['short']} {values}")
    ax.text(0.02, 0.90, "n (L/M/H): " + "\n".join(n_lines), transform=ax.transAxes, ha="left", va="top", fontsize=3.1, color=COLOR_DARK_NEUTRAL)
    ax.grid(True, linestyle=":", alpha=0.25, color=COLOR_LIGHT_REF)
    legend_handles = [
        Line2D([0], [0], color=COLOR_TGD, marker="o", markerfacecolor=COLOR_TGD, markersize=3.4, lw=0.8, label="TGD* (interim)"),
        Line2D([0], [0], color=COLOR_CN, marker="o", markerfacecolor=COLOR_CN, markersize=3.4, lw=0.8, label="CN"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, marker="o", markerfacecolor=COLOR_DARK_NEUTRAL, markersize=3.4, lw=0.8, label="IC"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, marker="o", markerfacecolor="white", markersize=3.4, lw=0.8, label="dPL"),
        Line2D([0], [0], color=COLOR_BASE, marker="|", markersize=6.5, lw=0.8, label="Base absolute-r reference"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, lw=0, label="labels: L / M / H burden"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=3.5, ncol=2, loc="lower right", handlelength=1.0, columnspacing=0.5)

def _symmetric_limits(values: np.ndarray, floor: float = -0.25, ceiling: float = 0.35) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if not len(finite):
        return floor, ceiling
    extent = max(abs(float(finite.min())), abs(float(finite.max())), abs(floor), abs(ceiling))
    extent = max(0.05, extent * 1.08)
    return -extent, extent


def _set_dynamic_xlim(ax, minimum: float, maximum: float, floor: float = -0.25, ceiling: float = 0.35) -> None:
    lo, hi = _symmetric_limits(np.array([minimum, maximum]), floor=floor, ceiling=ceiling)
    ax.set_xlim(lo, hi)


def _set_dynamic_ylim(ax, minimum: float, maximum: float, floor: float = -0.25, ceiling: float = 0.35, pad: float = 0.05) -> None:
    values = np.array([minimum, maximum], dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        ax.set_ylim(floor, ceiling)
        return
    span = max(float(np.ptp(finite)), 0.05)
    lo = min(float(finite.min()) - pad * span, floor)
    hi = max(float(finite.max()) + pad * span, ceiling if float(finite.max()) >= 0.0 else ceiling * 0.6)
    ax.set_ylim(lo, hi)


_CACHED_PLOT_FILES = {
    "panel_a": "figure7_panel_a_phase_burden_matrix.csv",
    "panel_b": "figure7_panel_b_forest.csv",
    "panel_c": "figure7_panel_d_ecdf_values.csv",
    "panel_d": "figure7_panel_d_composition.csv",
}


def _load_cached_plot_data(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load frozen Figure 7 panel tables without recomputing statistics."""
    paths = {key: results_dir / filename for key, filename in _CACHED_PLOT_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cached Figure 7 panel tables are required for plot-only rendering: " + ", ".join(missing)
        )
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    required_columns = {
        "panel_a": {"regime", "structure", "phase_name", "burden_group_index", "median_delta_r"},
        "panel_b": {"row_index", "row_label", "structure", "regime", "median_delta_r", "ci_lower", "ci_upper"},
        "panel_c": {"curve", "delta_r", "ecdf_y", "positive_fraction"},
        "panel_d": {"structure", "regime", "burden_group", "burden_group_index", "melt_specific_pct", "generic_pct", "mixed_pct", "no_change_pct"},
    }
    for key, columns in required_columns.items():
        missing_columns = columns.difference(frames[key].columns)
        if missing_columns:
            raise ValueError(f"Cached {key} table is missing columns: {sorted(missing_columns)}")
    frames["panel_b"]["row_label"] = frames["panel_b"]["row_label"].replace({"All days · n varies": "All days"})
    return frames


def _update_plot_metadata(results_dir: Path) -> None:
    """Update layout metadata only; never regenerate or rewrite statistic tables."""
    metadata_path = results_dir / "figure7_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cached Figure 7 metadata is required: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["architecture"] = "four_panel_overview_effect_heterogeneity_composition"
    metadata["main_panels"] = ["a", "b", "c", "d"]
    metadata["omitted_main_panel"] = "e (absolute-versus-paired audit retained in exported tables)"
    calibration = dict(metadata.get("panel_calibration", {}))
    calibration.pop("panel_e_xlim", None)
    calibration.pop("panel_e_ylim", None)
    calibration["panel_d_composition"] = {
        "xlim_pct": [0, 100],
        "bar_data_xlim_pct": [0, 100],
        "thresholds": "descriptive",
    }
    metadata["panel_calibration"] = calibration
    notes = metadata.setdefault("caption_support_notes", [])
    omitted_note = "Absolute-versus-paired statistics are retained in exported audit tables but omitted from the four-panel main figure."
    if omitted_note not in notes:
        notes.append(omitted_note)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def generate_figure7(results_root: Path, out_dir: Path) -> Path:
    """Render v2 Figure 7 from cached panel tables without changing data."""
    setup_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = results_root / "r4_phase1_soil_official"
    plot_data = _load_cached_plot_data(results_dir)
    fig = plt.figure(figsize=(23.5 / 2.54, 20.0 / 2.54))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.05, 1.0),
        height_ratios=(1.45, 1.0),
        hspace=0.24,
        wspace=0.38,
        left=0.08,
        right=0.94,
        top=0.96,
        bottom=0.06,
    )
    _plot_panel_a(fig, gs[0, 0], plot_data["panel_a"])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_panel_b(ax_b, plot_data["panel_b"])
    _plot_panel_c(ax_c, plot_data["panel_c"])
    _plot_panel_d(ax_d, plot_data["panel_d"])
    png_path = out_dir / "figure7_r4_soil_consistency.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Generated Figure 7 (PNG, 300 dpi):\n  {png_path}")
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 7 from cached statistics (plot-only)")
    parser.add_argument("--results-root", type=Path, default=default_results_root())
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    generate_figure7(args.results_root, args.out_dir)


if __name__ == "__main__":
    main()
