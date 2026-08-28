"""Generate Figure 8: Secondary soil-water timing/trajectory characterization (R4).

Figure 7 localizes the paired Base–TGD–CN contrast under external SWE phases. Figure 8
characterizes the corresponding trajectory and timing offsets relative to the
ERA5-Land SM100 process-state reference; it is not truth validation.

Layout (6-panel asymmetric composite layout following HESS / WRR guidelines):
  - (a) Snow accumulation and depletion (illustrative external-SWE-active water year)
  - (b) Standardized soil-water trajectories (same water year, shared x-axis with a)
  - (c) MAIN: Spring soil-water timing markers (Jan–Jun)
  - (d) Basin-level spring wet-up timing offsets (Q3 subset)
  - (e) Basin-level soil-water peak-timing offsets (Q3 subset)
  - (f) Timing-definition sensitivity under external-SWE-active years

Outputs:
    manuscript/figures/figure8_r4_soil_timing.png (300 DPI, PNG only - no PDF)
    manuscript/scripts/r4/figure8_r4_selection_audit.json (audit of rule-based illustrative case)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.scripts.shared.r1_plot_style import (  # noqa: E402
    COLOR_BASE,
    COLOR_CN,
    COLOR_DARK_NEUTRAL,
    COLOR_LIGHT_REF,
    COLOR_OBSERVATION,
    COLOR_TGD,
    COLOR_ZERO_LINE,
    MODEL_MARKERS,
    apply_clean_spines,
    setup_publication_style,
)
from manuscript.scripts.r4.common import default_results_root  # noqa: E402
from manuscript.scripts.r4.soil_analysis import calendar_month_anomaly  # noqa: E402

FIGURES_DIR = HERE.parents[1] / "figures"
CANONICAL_DPL = "dPL_seed42"
REGIMES = [CANONICAL_DPL, "IC_fused"]
REG_SHORT = ["dPL-42", "IC fused"]
REGIME_CFG = {
    CANONICAL_DPL: {
        "label": "dPL (seed 42)",
        "short_label": "dPL-42",
        "color": COLOR_DARK_NEUTRAL,
        "marker": "^",
        "ls": (0, (4.0, 2.0)),
    },
    "IC_fused": {
        "label": "IC (fused)",
        "short_label": "IC fused",
        "color": COLOR_DARK_NEUTRAL,
        "marker": "o",
        "ls": "-",
    },
}
STATE_CFG = {
    "Reference": {
        "label": "ERA5-Land SM$_{100}$",
        "color": COLOR_OBSERVATION,
        "ls": "-",
        "lw": 1.2,
        "marker": "o",
    },
    "Base": {
        "label": "Base $W_{total}$",
        "color": COLOR_BASE,
        "ls": "--",
        "lw": 1.2,
        "marker": MODEL_MARKERS.get("Base", "o"),
    },
    "TGD": {
        "label": "TGD $W_{total}$",
        "color": COLOR_TGD,
        "ls": "-.",
        "lw": 1.2,
        "marker": MODEL_MARKERS.get("TGD", "^"),
    },
    "CN": {
        "label": "CN $W_{total}$",
        "color": COLOR_CN,
        "ls": "-",
        "lw": 1.3,
        "marker": MODEL_MARKERS.get("CN", "s"),
    },
}
Q3_QUANTILE = 0.75
EXTERNAL_SWE_THRESHOLD_MM = 5.0


def _standardized_month_anomaly(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    months = pd.to_datetime(dates).month.to_numpy()
    anomaly = calendar_month_anomaly(values.astype(float), months)
    return (anomaly - np.nanmean(anomaly)) / (np.nanstd(anomaly) + 1e-12)


def _water_year_doy_to_date(water_year: int, doy: float) -> pd.Timestamp:
    return pd.Timestamp(year=int(water_year) - 1, month=10, day=1) + pd.Timedelta(
        days=float(doy) - 1
    )


def _load_external_timing_inputs(results_root: Path):
    """Load Base/TGD/CN, SM100, and external Snow-17 SWE for Figure 8."""
    caravan = np.load(results_root / "r4_caravan_soil_reference_v1" / "caravan_soil_ensemble.npz")
    test = slice(int(caravan["test_slice_start"]), int(caravan["test_slice_stop"]))
    dates = pd.to_datetime(caravan["dates"][test])
    basin_ids = [str(x).zfill(8) for x in caravan["basin_ids"]]
    sm100 = caravan["SM100"][:, test].astype(float)
    swe = np.load(results_root / "r4_swe_reference_v1" / "swe_ensemble.npz")["swe_median"][:, test].astype(float)

    # Use reconstructed TGD replay for dPL if present, else official
    tgd_dpl_path = (
        results_root / "r4_replay_dpl_XAJ_TGD2_seed42/reconstructed_dpl_XAJ_TGD2_seed42_full_arrays.npz"
    )
    if not tgd_dpl_path.exists():
        tgd_dpl_path = (
            results_root / "r4_official_dpl_XAJ_TGD2_seed42/official_dpl_XAJ_TGD2_seed42_full_arrays.npz"
        )

    model_paths = {
        "dPL_seed42": {
            "Base": results_root / "r4_official_dpl_XAJ_seed42/official_dpl_XAJ_seed42_full_arrays.npz",
            "TGD": tgd_dpl_path,
            "CN": results_root / "r4_official_dpl_XAJ_CN_seed42/official_dpl_XAJ_CN_seed42_full_arrays.npz",
        },
        "IC_fused": {
            "Base": results_root / "r4_ic_fused_XAJ/ic_fused_XAJ_full_arrays.npz",
            "TGD": results_root / "r4_ic_fused_XAJ_TGD2/ic_fused_XAJ_TGD2_full_arrays.npz",
            "CN": results_root / "r4_ic_fused_XAJ_CN/ic_fused_XAJ_CN_full_arrays.npz",
        },
    }
    models = {}
    for regime, struct_paths in model_paths.items():
        models[regime] = {}
        for struct, path in struct_paths.items():
            npz = np.load(path)
            models[regime][struct] = (
                npz["wu"][:, test] + npz["wl"][:, test] + npz["wd"][:, test]
            ).astype(float)
    return basin_ids, dates, sm100, swe, models


def _external_phase_codes(dates: pd.DatetimeIndex, swe: np.ndarray) -> np.ndarray:
    """Return the external-SWE phase mask used by robustness_analysis.py."""
    d = pd.DatetimeIndex(dates)
    wy = np.where(d.month >= 10, d.year + 1, d.year).astype(int)
    phases = np.zeros(len(d), dtype=np.int8)
    for water_year in np.unique(wy):
        mask = wy == water_year
        sw = swe[mask]
        if np.isfinite(sw).sum() == 0 or np.nanmax(sw) < EXTERNAL_SWE_THRESHOLD_MM:
            continue
        rel = np.arange(len(sw))
        peak = int(np.nanargmax(sw))
        acc = np.flatnonzero(np.isfinite(sw) & (sw >= EXTERNAL_SWE_THRESHOLD_MM))
        acc_start = int(acc[0]) if len(acc) else 0
        post_peak = np.flatnonzero((rel > peak) & np.isfinite(sw) & (sw < EXTERNAL_SWE_THRESHOLD_MM))
        melt_end = int(post_peak[0]) if len(post_peak) else len(sw) - 1
        indices = np.flatnonzero(mask)
        for j, abs_idx in enumerate(indices):
            month = d[abs_idx].month
            if acc_start <= j <= peak:
                phases[abs_idx] = 1
            elif peak < j <= melt_end:
                phases[abs_idx] = 2
            elif j > melt_end and month <= 6:
                phases[abs_idx] = 3
            elif month in (7, 8, 9):
                phases[abs_idx] = 4
    return phases


def _shade_external_phase(
    ax, dates: pd.Series | pd.DatetimeIndex, phases: np.ndarray, phase_code: int, **kwargs
) -> None:
    """Shade contiguous intervals from the external-SWE phase mask."""
    d = pd.DatetimeIndex(dates)
    selected = np.asarray(phases) == phase_code
    start = None
    for i, flag in enumerate(np.r_[selected, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ax.axvspan(d[start], d[i - 1] + pd.Timedelta(days=1), **kwargs)
            start = None


def _timing_records_for_series(
    dates: pd.DatetimeIndex,
    swe: np.ndarray,
    sm_ref: np.ndarray,
    model: np.ndarray,
    basin_id: str,
    structure: str,
    wetup_window_days: int = 14,
    peak_window: str = "full",
) -> list[dict]:
    """Compute timing markers using external SWE-only year inclusion."""
    d = pd.DatetimeIndex(dates)
    wy = np.where(d.month >= 10, d.year + 1, d.year).astype(int)
    records = []
    half_window = wetup_window_days // 2

    def argmax_or_nan(values):
        valid = np.isfinite(values)
        return int(np.nanargmax(np.where(valid, values, -np.inf))) if valid.any() else None

    for water_year in np.unique(wy):
        mask = wy == water_year
        if mask.sum() < 300:
            continue
        sw = swe[mask]
        if np.isfinite(sw).sum() < 300 or np.nanmax(sw) < EXTERNAL_SWE_THRESHOLD_MM:
            continue
        dw = np.arange(mask.sum(), dtype=float) + 1.0
        dates_w = d[mask]
        ref_w = sm_ref[mask]
        model_w = model[mask]
        if not np.isfinite(ref_w).any() or not np.isfinite(model_w).any():
            continue
        peak_window_mask = np.ones(len(dw), dtype=bool)
        if peak_window == "spring_summer":
            peak_window_mask = (dates_w.month >= 3) & (dates_w.month <= 8)
        ref_peak = argmax_or_nan(np.where(peak_window_mask, ref_w, np.nan))
        model_peak = argmax_or_nan(np.where(peak_window_mask, model_w, np.nan))
        if ref_peak is None or model_peak is None:
            continue
        spring_mask = (dates_w.month >= 1) & (dates_w.month <= 6)

        def rate(arr):
            series = pd.Series(arr)
            return (series.shift(-half_window) - series.shift(half_window)).to_numpy(float)

        ref_rate = rate(ref_w)
        model_rate = rate(model_w)
        ref_wetup = argmax_or_nan(np.where(spring_mask, ref_rate, np.nan))
        model_wetup = argmax_or_nan(np.where(spring_mask, model_rate, np.nan))
        records.append({
            "basin_id": basin_id,
            "water_year": int(water_year),
            "structure": structure,
            "external_snow_year": True,
            "annual_swe_max_mm": float(np.nanmax(sw)),
            "peak_doy_ref": float(dw[ref_peak]),
            "peak_doy_model": float(dw[model_peak]),
            "peak_timing_error_days": float(dw[model_peak] - dw[ref_peak]),
            "wetup_doy_ref": float(dw[ref_wetup]) if ref_wetup is not None else np.nan,
            "wetup_doy_model": float(dw[model_wetup]) if model_wetup is not None else np.nan,
            "wetup_timing_error_days": (
                float(dw[model_wetup] - dw[ref_wetup])
                if ref_wetup is not None and model_wetup is not None else np.nan
            ),
        })
    return records


def _external_timing_summary(results_root: Path, basin_ids: set[str]) -> pd.DataFrame:
    """Summarize Base/TGD/CN timing offsets for Q3 basins under external SWE years."""
    all_ids, dates, sm100, swe, models = _load_external_timing_inputs(results_root)
    wanted = set(str(b).zfill(8) for b in basin_ids)
    records = []
    for i, basin_id in enumerate(all_ids):
        if basin_id not in wanted:
            continue
        for regime, structure_models in models.items():
            for structure, model in structure_models.items():
                rows_for_regime = _timing_records_for_series(
                    dates, swe[i], sm100[i], model[i], basin_id, structure
                )
                for row in rows_for_regime:
                    row["regime"] = regime
                records.extend(rows_for_regime)
    rows = []
    if not records:
        return pd.DataFrame()
    data = pd.DataFrame(records)
    for (regime, basin_id, structure), group in data.groupby(["regime", "basin_id", "structure"]):
        wet = group["wetup_timing_error_days"].dropna().to_numpy(float)
        peak = group["peak_timing_error_days"].dropna().to_numpy(float)
        rows.append({
            "regime": regime,
            "basin_id": basin_id,
            "structure": structure,
            "n_valid_snow_years": int(len(group)),
            "median_wetup_error_days": float(np.median(wet)) if len(wet) else np.nan,
            "median_abs_wetup_error_days": float(np.median(np.abs(wet))) if len(wet) else np.nan,
            "median_peak_error_days": float(np.median(peak)) if len(peak) else np.nan,
            "median_abs_peak_error_days": float(np.median(np.abs(peak))) if len(peak) else np.nan,
        })
    return pd.DataFrame(rows)


def _select_illustrative_basin_year(results_root: Path) -> tuple[str, int, dict]:
    """Select a reproducible Q3 basin-year using external SWE only."""
    state = pd.read_csv(
        results_root / "r4_phase1_soil_official" / "basin_state_consistency.csv",
        dtype={"basin_id": str},
    )
    basin_swe = state[
        (state["regime"] == CANONICAL_DPL) & (state["structure"] == "Base")
    ].drop_duplicates("basin_id")[["basin_id", "snow_burden_swe_mm"]]
    q3_threshold = float(basin_swe["snow_burden_swe_mm"].quantile(Q3_QUANTILE))
    q3 = basin_swe[basin_swe["snow_burden_swe_mm"] >= q3_threshold].copy()
    q3_n = len(q3)
    basin_ids, dates, _, swe, _ = _load_external_timing_inputs(results_root)
    wy = np.where(dates.month >= 10, dates.year + 1, dates.year).astype(int)
    candidates = []
    for basin_id in q3["basin_id"]:
        i = basin_ids.index(str(basin_id).zfill(8))
        for water_year in np.unique(wy):
            sw = swe[i, wy == water_year]
            if np.isfinite(sw).sum() < 300 or np.nanmax(sw) < EXTERNAL_SWE_THRESHOLD_MM:
                continue
            candidates.append({
                "basin_id": str(basin_id).zfill(8),
                "water_year": int(water_year),
                "annual_swe_max_mm": float(np.nanmax(sw)),
                "snow_burden_swe_mm": float(
                    q3.loc[q3["basin_id"].eq(basin_id), "snow_burden_swe_mm"].iloc[0]
                ),
            })
    candidate_df = pd.DataFrame(candidates)
    target = candidate_df[["annual_swe_max_mm", "snow_burden_swe_mm"]].median()
    scale = candidate_df[["annual_swe_max_mm", "snow_burden_swe_mm"]].std().replace(0, 1)
    candidate_df["selection_score"] = (
        (candidate_df["annual_swe_max_mm"] - target["annual_swe_max_mm"]).abs() / scale["annual_swe_max_mm"]
        + (candidate_df["snow_burden_swe_mm"] - target["snow_burden_swe_mm"]).abs() / scale["snow_burden_swe_mm"]
    )
    selected = candidate_df.sort_values(["selection_score", "basin_id", "water_year"]).iloc[0]
    basin_id = str(selected["basin_id"]).zfill(8)
    water_year = int(selected["water_year"])
    audit = {
        "case_description": "Illustrative Q3 basin-year under external SWE conditioning",
        "selection_rule": (
            "Among Q3 basins and externally snow-active water years (annual SWE max >= 5 mm, "
            "at least 300 finite SWE days), minimize standardized distance from the joint median "
            "of annual SWE peak and basin SWE burden; tie-break by basin ID then water year. "
            "No model W_total threshold is used for inclusion."
        ),
        "condition_source": "r4_swe_reference_v1/swe_ensemble.npz::swe_median; external SWE only",
        "q3_threshold_mm": q3_threshold,
        "q3_basin_count": q3_n,
        "selected_basin_id": basin_id,
        "selected_water_year": water_year,
        "selected_year_annual_swe_max_mm": float(selected["annual_swe_max_mm"]),
        "selected_basin_swe_burden_mm": float(selected["snow_burden_swe_mm"]),
    }
    return basin_id, water_year, audit


def _load_illustrative_series(results_root: Path, basin_id: str, water_year: int):
    basin_ids, dates, sm100, swe_all, models = _load_external_timing_inputs(results_root)
    idx = basin_ids.index(str(basin_id).zfill(8))
    ref = sm100[idx]
    swe = swe_all[idx]
    regime_models = models[CANONICAL_DPL]
    base = regime_models["Base"][idx]
    tgd = regime_models["TGD"][idx]
    cn = regime_models["CN"][idx]
    timing_rows = []
    for structure, model in (("Base", base), ("TGD", tgd), ("CN", cn)):
        timing_rows.extend(_timing_records_for_series(
            dates, swe, ref, model, str(basin_id).zfill(8), structure
        ))
    timing = pd.DataFrame(timing_rows)
    timing = timing[timing["water_year"].eq(water_year)].copy()
    return dates, ref, base, tgd, cn, swe, timing


def _compute_timing_definition_sensitivity(
    dates: pd.DatetimeIndex,
    sm100: np.ndarray,
    swe: np.ndarray,
    models: dict[str, dict[str, np.ndarray]],
    basin_ids: list[str],
) -> pd.DataFrame:
    """Compute Base-TGD and Base-CN MAE improvements across timing definitions."""
    d = pd.DatetimeIndex(dates)
    wy = np.where(d.month >= 10, d.year + 1, d.year).astype(int)
    doy = np.zeros(len(d), dtype=float)
    for w in np.unique(wy):
        mask = wy == w
        doy[mask] = np.arange(mask.sum(), dtype=float) + 1.0

    timing_variants = [
        ("Wet-up 7 d", "Peak_Annual_FullWY", "Wetup_07d_Spring", 7, "full", "wetup"),
        ("Wet-up 14 d", "Peak_Annual_FullWY", "Wetup_14d_Spring", 14, "full", "wetup"),
        ("Wet-up 21 d", "Peak_Annual_FullWY", "Wetup_21d_Spring", 21, "full", "wetup"),
        ("Peak full WY", "Peak_Annual_FullWY", "Wetup_14d_Spring", 14, "full", "peak"),
        ("Peak Mar–Aug", "Peak_SpringSummer_MarAug", "Wetup_14d_Spring", 14, "spring_summer", "peak"),
    ]

    summary_rows = []
    for regime in REGIMES:
        wb_all = models[regime]["Base"]
        wt_all = models[regime]["TGD"]
        wc_all = models[regime]["CN"]
        for label, peak_name, wetup_name, wetup_window_days, peak_window, metric_type in timing_variants:
            p_errs_base = []
            p_errs_tgd = []
            p_errs_cn = []
            w_errs_base = []
            w_errs_tgd = []
            w_errs_cn = []

            half_w = wetup_window_days // 2

            def calc_rate(arr):
                s = pd.Series(arr)
                return (s.shift(-half_w) - s.shift(half_w)).to_numpy()

            for i in range(len(basin_ids)):
                wb = wb_all[i]
                wt = wt_all[i]
                wc = wc_all[i]
                ref = sm100[i]
                sw_arr = swe[i]

                for w in np.unique(wy):
                    mask = wy == w
                    if mask.sum() < 300:
                        continue
                    sw = sw_arr[mask]
                    if np.nanmax(sw) < EXTERNAL_SWE_THRESHOLD_MM:
                        continue
                    dw = doy[mask]
                    dates_w = d[mask]
                    ref_w = ref[mask]
                    wb_w = wb[mask]
                    wt_w = wt[mask]
                    wc_w = wc[mask]

                    if peak_window == "spring_summer":
                        p_win_mask = (dates_w.month >= 3) & (dates_w.month <= 8)
                    else:
                        p_win_mask = np.ones(len(dw), dtype=bool)

                    ref_p_sub = np.where(p_win_mask, ref_w, -999.0)
                    wb_p_sub = np.where(p_win_mask, wb_w, -999.0)
                    wt_p_sub = np.where(p_win_mask, wt_w, -999.0)
                    wc_p_sub = np.where(p_win_mask, wc_w, -999.0)

                    doy_p_ref = float(dw[int(np.nanargmax(ref_p_sub))])
                    doy_p_base = float(dw[int(np.nanargmax(wb_p_sub))])
                    doy_p_tgd = float(dw[int(np.nanargmax(wt_p_sub))])
                    doy_p_cn = float(dw[int(np.nanargmax(wc_p_sub))])

                    p_errs_base.append(doy_p_base - doy_p_ref)
                    p_errs_tgd.append(doy_p_tgd - doy_p_ref)
                    p_errs_cn.append(doy_p_cn - doy_p_ref)

                    spring_mask = (dates_w.month >= 1) & (dates_w.month <= 6)
                    diff_ref = calc_rate(ref_w)
                    diff_wb = calc_rate(wb_w)
                    diff_wt = calc_rate(wt_w)
                    diff_wc = calc_rate(wc_w)

                    diff_ref_sp = np.where(spring_mask, diff_ref, -999.0)
                    diff_wb_sp = np.where(spring_mask, diff_wb, -999.0)
                    diff_wt_sp = np.where(spring_mask, diff_wt, -999.0)
                    diff_wc_sp = np.where(spring_mask, diff_wc, -999.0)

                    doy_w_ref = float(dw[int(np.nanargmax(diff_ref_sp))])
                    doy_w_base = float(dw[int(np.nanargmax(diff_wb_sp))])
                    doy_w_tgd = float(dw[int(np.nanargmax(diff_wt_sp))])
                    doy_w_cn = float(dw[int(np.nanargmax(diff_wc_sp))])

                    w_errs_base.append(doy_w_base - doy_w_ref)
                    w_errs_tgd.append(doy_w_tgd - doy_w_ref)
                    w_errs_cn.append(doy_w_cn - doy_w_ref)

            pb = np.array(p_errs_base)
            pt = np.array(p_errs_tgd)
            pc = np.array(p_errs_cn)
            wb_arr = np.array(w_errs_base)
            wt_arr = np.array(w_errs_tgd)
            wc_arr = np.array(w_errs_cn)

            if metric_type == "wetup":
                b_mae = float(np.nanmedian(np.abs(wb_arr)))
                t_mae = float(np.nanmedian(np.abs(wt_arr)))
                c_mae = float(np.nanmedian(np.abs(wc_arr)))
            else:
                b_mae = float(np.nanmedian(np.abs(pb)))
                t_mae = float(np.nanmedian(np.abs(pt)))
                c_mae = float(np.nanmedian(np.abs(pc)))

            summary_rows.append({
                "label": label,
                "peak_definition": peak_name,
                "wetup_definition": wetup_name,
                "regime": regime,
                "metric_type": metric_type,
                "Base_MAE": b_mae,
                "TGD_MAE": t_mae,
                "CN_MAE": c_mae,
                "Base_TGD_gain": b_mae - t_mae,
                "Base_CN_gain": b_mae - c_mae,
                "n_valid_basin_years": len(pb),
            })
    return pd.DataFrame(summary_rows)


def generate_figure8(results_root: Path, out_dir: Path) -> Path:
    setup_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    official = results_root / "r4_phase1_soil_official"

    # Illustrative case
    basin_id, water_year, audit = _select_illustrative_basin_year(results_root)
    (HERE / "figure8_r4_selection_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    dates, ref, base, tgd, cn, swe, timing = _load_illustrative_series(
        results_root, basin_id, water_year
    )

    ref_z = _standardized_month_anomaly(ref, dates.to_numpy())
    base_z = _standardized_month_anomaly(base, dates.to_numpy())
    tgd_z = _standardized_month_anomaly(tgd, dates.to_numpy())
    cn_z = _standardized_month_anomaly(cn, dates.to_numpy())

    df = pd.DataFrame({
        "date": dates,
        "ref_z": ref_z,
        "base_z": base_z,
        "tgd_z": tgd_z,
        "cn_z": cn_z,
        "swe": swe,
    })
    df["wy"] = df["date"].apply(lambda d: d.year if d.month < 10 else d.year + 1)
    df_wy = df[df["wy"] == water_year].copy()
    phase_codes = _external_phase_codes(
        pd.DatetimeIndex(df_wy["date"]), df_wy["swe"].to_numpy(float)
    )

    # Timing markers for illustrative year
    row_base = timing[timing["structure"] == "Base"].iloc[0]
    row_tgd = timing[timing["structure"] == "TGD"].iloc[0]
    row_cn = timing[timing["structure"] == "CN"].iloc[0]

    d_ref_w = _water_year_doy_to_date(water_year, row_base["wetup_doy_ref"])
    d_base_w = _water_year_doy_to_date(water_year, row_base["wetup_doy_model"])
    d_tgd_w = _water_year_doy_to_date(water_year, row_tgd["wetup_doy_model"])
    d_cn_w = _water_year_doy_to_date(water_year, row_cn["wetup_doy_model"])

    d_ref_p = _water_year_doy_to_date(water_year, row_base["peak_doy_ref"])
    d_base_p = _water_year_doy_to_date(water_year, row_base["peak_doy_model"])
    d_tgd_p = _water_year_doy_to_date(water_year, row_tgd["peak_doy_model"])
    d_cn_p = _water_year_doy_to_date(water_year, row_cn["peak_doy_model"])

    # Q3 population
    paired = pd.read_csv(
        official / "paired_structural_effects.csv", dtype={"basin_id": str}
    )
    swe_q3 = paired[paired["regime"] == CANONICAL_DPL].drop_duplicates("basin_id")
    q3_threshold = float(swe_q3["snow_burden_swe_mm"].quantile(Q3_QUANTILE))
    q3_basins = set(swe_q3[swe_q3["snow_burden_swe_mm"] >= q3_threshold]["basin_id"])
    q3_n = len(q3_basins)
    summary = _external_timing_summary(results_root, q3_basins)

    # All models for sensitivity calculation
    all_basin_ids, all_dates, all_sm100, all_swe, all_models = _load_external_timing_inputs(results_root)
    sens_df = _compute_timing_definition_sensitivity(
        all_dates, all_sm100, all_swe, all_models, all_basin_ids
    )

    # Build wide composite figure (7.2 x 6.4 inches -> aspect ratio ~1.12 : 1)
    fig = plt.figure(figsize=(7.2, 7.15))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[3.5, 1.45, 1.50],
        hspace=0.38,
        left=0.075,
        right=0.965,
        top=0.96,
        bottom=0.06,
    )

    # ---------------------------------------------------------------------------
    # Tier 1: Process Dynamics (a, b, c) ~61% area
    # ---------------------------------------------------------------------------
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.26)
    gs_left = gs_top[0].subgridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.20)

    # (a) Snow accumulation and depletion (illustrative external SWE context)
    ax_a = fig.add_subplot(gs_left[0])
    apply_clean_spines(ax_a)
    _shade_external_phase(
        ax_a,
        df_wy["date"],
        phase_codes,
        phase_code=2,
        color=COLOR_LIGHT_REF,
        alpha=0.55,
        zorder=0,
    )
    ax_a.plot(df_wy["date"], df_wy["swe"], color=COLOR_CN, lw=1.1, zorder=3)
    ax_a.fill_between(
        df_wy["date"], 0, df_wy["swe"], color=COLOR_LIGHT_REF, alpha=0.25, zorder=2
    )
    ax_a.set_ylabel("SWE [mm]", fontsize=7.2)
    ax_a.set_title(
        "(a) Snow accumulation and depletion",
        loc="left",
        fontweight="bold",
        fontsize=8.0,
    )
    ax_a.tick_params(axis="x", labelbottom=False)
    ax_a.set_xlim(
        pd.Timestamp(f"{water_year - 1}-10-01"), pd.Timestamp(f"{water_year}-09-30")
    )
    ax_a.text(
        0.97,
        0.88,
        "Phase 2\n(active melt)",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color=COLOR_DARK_NEUTRAL,
    )

    # (b) Soil-water trajectories (same water year, shared x-axis)
    ax_b = fig.add_subplot(gs_left[1], sharex=ax_a)
    apply_clean_spines(ax_b)
    _shade_external_phase(
        ax_b,
        df_wy["date"],
        phase_codes,
        phase_code=2,
        color=COLOR_LIGHT_REF,
        alpha=0.55,
        zorder=0,
    )
    ax_b.axhline(0, color=COLOR_LIGHT_REF, ls=":", lw=0.7, zorder=1)
    for key in ["Reference", "Base", "TGD", "CN"]:
        cfg = STATE_CFG[key]
        col_name = (
            "ref_z" if key == "Reference"
            else ("base_z" if key == "Base"
            else ("tgd_z" if key == "TGD"
            else "cn_z"))
        )
        ax_b.plot(
            df_wy["date"],
            df_wy[col_name],
            color=cfg["color"],
            ls=cfg["ls"],
            lw=cfg["lw"],
            label=cfg["label"],
            zorder=4 if key != "CN" else 5,
        )
    ax_b.set_ylabel("Standardized anomaly", fontsize=7.2)
    ax_b.set_title(
        "(b) Soil-water trajectories", loc="left", fontweight="bold", fontsize=8.0
    )
    ax_b.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_b.set_xlabel(f"Water year {water_year} (Basin {basin_id}, dPL-42)", fontsize=7.0)
    ax_b.legend(loc="upper left", ncol=2, fontsize=5.2, frameon=True, framealpha=0.92)

    # (c) MAIN spring zoom + timing rails
    gs_c = gs_top[1].subgridspec(2, 1, height_ratios=[2.4, 1.05], hspace=0.15)
    ax_c_top = fig.add_subplot(gs_c[0])
    ax_c_bot = fig.add_subplot(gs_c[1], sharex=ax_c_top)
    apply_clean_spines(ax_c_top)
    apply_clean_spines(ax_c_bot)

    zoom_start = pd.Timestamp(f"{water_year}-01-01")
    zoom_end = pd.Timestamp(f"{water_year}-06-30")
    df_zoom = df_wy[(df_wy["date"] >= zoom_start) & (df_wy["date"] <= zoom_end)]

    # Trajectory zoom
    _shade_external_phase(
        ax_c_top,
        df_zoom["date"],
        phase_codes[df_wy["date"].isin(df_zoom["date"])],
        phase_code=2,
        color=COLOR_LIGHT_REF,
        alpha=0.55,
        zorder=0,
    )
    ax_c_top.axhline(0, color=COLOR_LIGHT_REF, ls=":", lw=0.7, zorder=1)
    for key in ["Reference", "Base", "TGD", "CN"]:
        cfg = STATE_CFG[key]
        col_name = (
            "ref_z" if key == "Reference"
            else ("base_z" if key == "Base"
            else ("tgd_z" if key == "TGD"
            else "cn_z"))
        )
        ax_c_top.plot(
            df_zoom["date"],
            df_zoom[col_name],
            color=cfg["color"],
            ls=cfg["ls"],
            lw=cfg["lw"],
            zorder=4 if key != "CN" else 5,
        )
    ax_c_top.set_ylabel("Standardized anomaly", fontsize=7.2)
    ax_c_top.set_title(
        "(c) Spring soil-water timing markers",
        loc="left",
        fontweight="bold",
        fontsize=8.0,
    )
    ax_c_top.tick_params(axis="x", labelbottom=False)
    ax_c_top.set_xlim(zoom_start, zoom_end)

    # Timing rails
    _shade_external_phase(
        ax_c_bot,
        df_zoom["date"],
        phase_codes[df_wy["date"].isin(df_zoom["date"])],
        phase_code=2,
        color=COLOR_LIGHT_REF,
        alpha=0.55,
        zorder=0,
    )
    ax_c_bot.axhline(1.0, color=COLOR_LIGHT_REF, ls="-", lw=0.8, zorder=1)
    ax_c_bot.axhline(0.0, color=COLOR_LIGHT_REF, ls="-", lw=0.8, zorder=1)

    # Wet-up markers
    ax_c_bot.plot([d_base_w, d_ref_w], [1.0, 1.0], color=COLOR_BASE, ls=":", lw=0.9, alpha=0.7, zorder=2)
    ax_c_bot.plot([d_tgd_w, d_ref_w], [1.0, 1.0], color=COLOR_TGD, ls=":", lw=0.9, alpha=0.7, zorder=2)
    ax_c_bot.plot([d_cn_w, d_ref_w], [1.0, 1.0], color=COLOR_CN, ls=":", lw=0.9, alpha=0.7, zorder=2)

    ax_c_bot.plot(
        d_ref_w, 1.0, marker="o", color="none",
        markeredgecolor=COLOR_OBSERVATION, markeredgewidth=1.6, ms=6.8, zorder=4
    )
    ax_c_bot.plot(d_base_w, 1.0, marker=STATE_CFG["Base"]["marker"], color=COLOR_BASE, ms=5.2, zorder=5)
    ax_c_bot.plot(d_tgd_w, 1.0, marker=STATE_CFG["TGD"]["marker"], color=COLOR_TGD, ms=5.2, zorder=5)
    ax_c_bot.plot(d_cn_w, 1.0, marker=STATE_CFG["CN"]["marker"], color=COLOR_CN, ms=4.8, zorder=6)

    ax_c_bot.text(
        d_base_w, 1.24, f"{int(row_base['wetup_timing_error_days']):+d} d",
        ha="center", va="bottom", fontsize=5.5, color=COLOR_BASE, fontweight="bold"
    )
    ax_c_bot.text(
        d_tgd_w, 1.24, f"TGD: {int(row_tgd['wetup_timing_error_days']):+d} d",
        ha="center", va="bottom", fontsize=5.2, color=COLOR_TGD
    )
    ax_c_bot.text(
        d_cn_w, 0.76, f"CN: {int(row_cn['wetup_timing_error_days']):+d} d",
        ha="center", va="top", fontsize=5.2, color=COLOR_CN
    )

    # Peak markers
    ax_c_bot.plot([d_base_p, d_ref_p], [0.0, 0.0], color=COLOR_BASE, ls=":", lw=0.9, alpha=0.7, zorder=2)
    ax_c_bot.plot([d_tgd_p, d_ref_p], [0.0, 0.0], color=COLOR_TGD, ls=":", lw=0.9, alpha=0.7, zorder=2)
    ax_c_bot.plot([d_cn_p, d_ref_p], [0.0, 0.0], color=COLOR_CN, ls=":", lw=0.9, alpha=0.7, zorder=2)

    ax_c_bot.plot(
        d_ref_p, 0.0, marker="o", color="none",
        markeredgecolor=COLOR_OBSERVATION, markeredgewidth=1.6, ms=6.8, zorder=4
    )
    ax_c_bot.plot(d_base_p, 0.0, marker=STATE_CFG["Base"]["marker"], color=COLOR_BASE, ms=5.2, zorder=5)
    ax_c_bot.plot(d_tgd_p, 0.0, marker=STATE_CFG["TGD"]["marker"], color=COLOR_TGD, ms=5.2, zorder=5)
    ax_c_bot.plot(d_cn_p, 0.0, marker=STATE_CFG["CN"]["marker"], color=COLOR_CN, ms=4.8, zorder=6)

    ax_c_bot.text(
        d_base_p, 0.24, f"{int(row_base['peak_timing_error_days']):+d} d",
        ha="center", va="bottom", fontsize=5.5, color=COLOR_BASE, fontweight="bold"
    )
    ax_c_bot.text(
        d_cn_p, -0.24, f"CN: {int(row_cn['peak_timing_error_days']):+d} d",
        ha="center", va="top", fontsize=5.2, color=COLOR_CN
    )
    ax_c_bot.text(
        d_tgd_p, 0.24, f"TGD: {int(row_tgd['peak_timing_error_days']):+d} d",
        ha="center", va="bottom", fontsize=5.2, color=COLOR_TGD
    )

    ax_c_bot.set_yticks([0.0, 1.0])
    ax_c_bot.set_yticklabels(["Peak", "Wet-up"], fontsize=6.8)
    ax_c_bot.set_ylim(-0.52, 1.52)
    ax_c_bot.xaxis.set_major_locator(mdates.MonthLocator())
    ax_c_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_c_bot.set_xlabel(f"Spring {water_year}", fontsize=7.0)

    # Legend for (c)
    leg_c = [
        Line2D(
            [0], [0], marker="o", color="w", markeredgecolor=COLOR_OBSERVATION,
            markeredgewidth=1.5, markerfacecolor="none", ms=5.5, label="ERA5-Land SM$_{100}$"
        ),
        Line2D(
            [0], [0], marker=STATE_CFG["Base"]["marker"], color="w",
            markerfacecolor=COLOR_BASE, ms=5.0, label="Base"
        ),
        Line2D(
            [0], [0], marker=STATE_CFG["TGD"]["marker"], color="w",
            markerfacecolor=COLOR_TGD, ms=5.0, label="TGD"
        ),
        Line2D(
            [0], [0], marker=STATE_CFG["CN"]["marker"], color="w",
            markerfacecolor=COLOR_CN, ms=4.8, label="CN"
        ),
    ]
    ax_c_top.legend(
        handles=leg_c, loc="upper left", ncol=2, fontsize=5.2, frameon=True, framealpha=0.92
    )

    # ---------------------------------------------------------------------------
    # Tier 2: Basin-level Timing Populations (d, e) ~24% area
    # ---------------------------------------------------------------------------
    gs_mid = gs[1].subgridspec(1, 2, wspace=0.24)
    ax_d = fig.add_subplot(gs_mid[0])
    ax_e = fig.add_subplot(gs_mid[1])

    # (d) Spring wet-up timing ECDF
    apply_clean_spines(ax_d)
    ax_d.axvline(0, color=COLOR_ZERO_LINE, ls="--", lw=0.8, zorder=1)
    for reg, reg_name in [("dPL_seed42", "dPL-42"), ("IC_fused", "IC fused")]:
        cfg = REGIME_CFG[reg]
        sub = summary[
            (summary["regime"] == reg) & (summary["basin_id"].isin(q3_basins))
        ].copy()
        for struct, col in [
            ("Base", COLOR_BASE),
            ("TGD", COLOR_TGD),
            ("CN", COLOR_CN),
        ]:
            vals = (
                sub[sub["structure"] == struct]["median_wetup_error_days"]
                .dropna()
                .to_numpy()
            )
            sx = np.sort(vals)
            sy = np.linspace(0, 1, len(sx))
            ax_d.step(
                sx,
                sy,
                where="post",
                color=col,
                ls=cfg["ls"],
                lw=1.2,
                alpha=0.9,
                label=f"{struct} ({reg_name})",
            )

    ax_d.set_title(
        "(d) Basin-level spring wet-up timing offset",
        loc="left",
        fontweight="bold",
        fontsize=8.0,
    )
    ax_d.set_xlabel(
        "Wet-up timing offset relative to ERA5-Land SM$_{100}$ [days]",
        fontsize=7.0,
    )
    ax_d.set_ylabel(f"Cumulative fraction (Q3, n = {q3_n})", fontsize=7.0)
    ax_d.set_xlim(-140, 110)
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.legend(loc="upper left", ncol=2, fontsize=5.0, frameon=True, framealpha=0.92)

    # (e) Soil-water peak timing ECDF
    apply_clean_spines(ax_e)
    ax_e.axvline(0, color=COLOR_ZERO_LINE, ls="--", lw=0.8, zorder=1)
    for reg, reg_name in [("dPL_seed42", "dPL-42"), ("IC_fused", "IC fused")]:
        cfg = REGIME_CFG[reg]
        sub = summary[
            (summary["regime"] == reg) & (summary["basin_id"].isin(q3_basins))
        ].copy()
        for struct, col in [
            ("Base", COLOR_BASE),
            ("TGD", COLOR_TGD),
            ("CN", COLOR_CN),
        ]:
            vals = (
                sub[sub["structure"] == struct]["median_peak_error_days"]
                .dropna()
                .to_numpy()
            )
            sx = np.sort(vals)
            sy = np.linspace(0, 1, len(sx))
            ax_e.step(
                sx,
                sy,
                where="post",
                color=col,
                ls=cfg["ls"],
                lw=1.2,
                alpha=0.9,
                label=f"{struct} ({reg_name})",
            )

    ax_e.set_title(
        "(e) Basin-level soil-water peak-timing offset",
        loc="left",
        fontweight="bold",
        fontsize=8.0,
    )
    ax_e.set_xlabel(
        "Peak-timing offset relative to ERA5-Land SM$_{100}$ [days]",
        fontsize=7.0,
    )
    ax_e.set_ylabel(f"Cumulative fraction (Q3, n = {q3_n})", fontsize=7.0)
    ax_e.set_xlim(-210, 40)
    ax_e.set_ylim(-0.02, 1.02)
    ax_e.legend(loc="upper left", ncol=2, fontsize=5.0, frameon=True, framealpha=0.92)
    ax_e.text(
        0.97,
        0.08,
        "Q3 snow years\nmedian offsets",
        transform=ax_e.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=COLOR_DARK_NEUTRAL,
        bbox=dict(
            boxstyle="square,pad=0.25",
            facecolor="#FAFAFA",
            edgecolor="#E0E0E0",
            alpha=0.9,
        ),
    )

    # ---------------------------------------------------------------------------
    # Tier 3: Definition Sensitivity (f) ~15% area
    # ---------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[2])
    apply_clean_spines(ax_f)
    ax_f.axvline(0, color=COLOR_ZERO_LINE, ls="--", lw=0.8, zorder=1)

    sens_rows = [
        ("Wet-up 7 d", "Peak_Annual_FullWY", "Wetup_07d_Spring"),
        ("Wet-up 14 d", "Peak_Annual_FullWY", "Wetup_14d_Spring"),
        ("Wet-up 21 d", "Peak_Annual_FullWY", "Wetup_21d_Spring"),
        ("Peak full WY", "Peak_Annual_FullWY", "Wetup_14d_Spring"),
        ("Peak Mar–Aug", "Peak_SpringSummer_MarAug", "Wetup_14d_Spring"),
    ]
    y_pos = np.array([4.0, 3.0, 2.0, 0.6, -0.4])

    for y, (label, peak_def, wet_def) in zip(y_pos, sens_rows):
        for reg in REGIMES:
            row = sens_df[
                (sens_df["regime"] == reg)
                & (sens_df["label"] == label)
            ].iloc[0]

            # Base - TGD improvement (Green)
            ax_f.plot(
                row["Base_TGD_gain"],
                y - 0.12 if reg == CANONICAL_DPL else y + 0.12,
                marker=REGIME_CFG[reg]["marker"],
                color=COLOR_TGD,
                ms=4.8,
                zorder=4,
            )
            # Base - CN improvement (Blue)
            ax_f.plot(
                row["Base_CN_gain"],
                y - 0.12 if reg == CANONICAL_DPL else y + 0.12,
                marker=REGIME_CFG[reg]["marker"],
                color=COLOR_CN,
                ms=4.8,
                zorder=4,
            )

    for y_tick in y_pos:
        ax_f.axhline(y_tick, color="#D5DBDF", ls="--", lw=0.65, zorder=0)
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels([r[0] for r in sens_rows], fontsize=6.8)
    ax_f.axhline(1.3, color=COLOR_LIGHT_REF, lw=0.8, zorder=1)
    ax_f.set_xlabel(
        "Timing MAE reduction relative to Base [days]  (Base MAE − Model MAE; positive = improved alignment)",
        fontsize=7.0,
    )
    ax_f.set_xlim(-10, 45)
    ax_f.set_title(
        "(f) Timing-definition sensitivity",
        loc="left",
        fontweight="bold",
        fontsize=8.0,
    )
    ax_f.text(
        0.01,
        0.90,
        "Wet-up definitions",
        transform=ax_f.transAxes,
        fontsize=5.5,
        color=COLOR_DARK_NEUTRAL,
    )
    ax_f.text(
        0.01,
        0.25,
        "Peak windows",
        transform=ax_f.transAxes,
        fontsize=5.5,
        color=COLOR_DARK_NEUTRAL,
    )

    leg_f = [
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=COLOR_TGD,
            ms=5.0, label="Base − TGD gain"
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=COLOR_CN,
            ms=5.0, label="Base − CN gain"
        ),
        Line2D(
            [0], [0], marker=REGIME_CFG["IC_fused"]["marker"], color=COLOR_DARK_NEUTRAL,
            ls="none", ms=4.5, label="IC (fused)"
        ),
        Line2D(
            [0], [0], marker=REGIME_CFG[CANONICAL_DPL]["marker"], color=COLOR_DARK_NEUTRAL,
            ls="none", ms=4.5, label="dPL (seed 42)"
        ),
    ]
    ax_f.legend(
        handles=leg_f,
        loc="lower right",
        fontsize=5.2,
        ncol=2,
        frameon=True,
        framealpha=0.92,
    )

    out = out_dir / "figure8_r4_soil_timing.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(
        f"Generated Figure 8 (PNG only, 300 dpi):\n  {out}\n  basin={basin_id}, water_year={water_year}"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    generate_figure8(args.results_root or default_results_root(), args.out_dir)
