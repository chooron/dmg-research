"""R4 formal real-basin shared soil-water state consistency analysis.

Evaluates Base (XAJ) vs CN (XAJ_CN) shared downstream tension water storage
W_total = wu + wl + wd against the Caravan v1.1 CAMELS-US ERA5-Land soil moisture
reference (primary: SM100 = 0.07*L1 + 0.21*L2 + 0.72*L3; sensitivity: SM289).

Analyses:
1. Standardized & anomaly dynamics:
   - raw daily Pearson & Spearman correlation
   - 7-day centered smoothed Pearson correlation
   - calendar-month anomaly Pearson correlation
   - z-score normalized trajectory error (NRMSE)
   - calendar-monthly mean Pearson correlation (robustness)
2. Seasonal timing diagnostics:
   - annual soil-water peak timing (doy of water year) & timing error
   - spring recharge / wetup timing (max 14d rate of increase) & timing error
   - signed error, absolute error, IQR, valid basin-year count
3. Paired structural contrast:
   - DeltaC(CN - Base) per basin across all metrics
4. Snow burden gradients:
   - DeltaC ~ Snow-17 median annual max SWE, SWE-positive days, frac_snow
   - Spearman rho, Theil-Sen slope, basin-level bootstrap 95% CIs (2000 reps)
   - snow burden quartile summaries Q0..Q3
5. Multi-scale & multi-regime consistency (dPL seeds 42 & 123 + IC fused sensitivity).

Outputs:
    results/r4_phase1_soil_official/
        basin_state_consistency.csv
        paired_structural_effects.csv
        timing_metrics_basin_year.csv
        timing_metrics_basin_summary.csv
        snow_burden_quartile_summary.csv
        r4_phase1_soil_official_report.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.scripts.r4 import (  # noqa: E402
    IC_FUSED_5x200_SENSITIVITY,
    OFFICIAL_DPL_OBSERVATION_TRAINED,
)
from manuscript.scripts.r4.common import default_data_root, default_results_root, load_bundle, zfill8  # noqa: E402

OUT_DIR = default_results_root() / "r4_phase1_soil_official"
SWE_REF_DIR = default_results_root() / "r4_swe_reference_v1"
CARAVAN_REF_DIR = default_results_root() / "r4_caravan_soil_reference_v1"

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20260730

# Timing validity thresholds
MIN_ANNUAL_SWE_PEAK_MM = 5.0
MIN_ANNUAL_W_PEAK_MM = 0.1
MIN_VALID_YEARS_PER_BASIN = 5


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------


def calendar_month_anomaly(series: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Subtract the calendar-month climatological mean from the daily series."""
    anom = np.empty_like(series, dtype=np.float64)
    for m in range(1, 13):
        idx = months == m
        if np.any(idx):
            anom[idx] = series[idx] - np.nanmean(series[idx])
        else:
            anom[idx] = np.nan
    return anom


def smooth_7d(series: np.ndarray) -> np.ndarray:
    """7-day centered rolling mean with min_periods=4, boundary-padded."""
    s = pd.Series(series)
    return s.rolling(7, center=True, min_periods=4).mean().to_numpy(dtype=np.float64)


def monthly_aggregate(series: np.ndarray, dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate daily series to calendar monthly means."""
    df = pd.DataFrame({"val": series, "date": dates})
    df["ym"] = pd.to_datetime(df["date"]).dt.to_period("M")
    agg = df.groupby("ym")["val"].mean()
    return agg.to_numpy(dtype=np.float64), agg.index.to_numpy()


def zscore_nrmse(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized trajectory error on standardized series (z-score NRMSE)."""
    sx = np.nanstd(x)
    sy = np.nanstd(y)
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    zx = (x - np.nanmean(x)) / sx
    zy = (y - np.nanmean(y)) / sy
    return float(np.sqrt(np.nanmean((zx - zy) ** 2)))


def theil_sen_bootstrap(
    x: np.ndarray, y: np.ndarray, n_boot: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Theil-Sen slope & Spearman correlation with paired basin bootstrap 95% CIs."""
    valid = np.isfinite(x) & np.isfinite(y)
    xv = x[valid].astype(np.float64)
    yv = y[valid].astype(np.float64)
    n = len(xv)
    if n < 10 or len(np.unique(xv)) < 5:
        return {
            "n": n, "spearman_rho": np.nan, "spearman_p": np.nan, "rho_ci_lower": np.nan, "rho_ci_upper": np.nan,
            "theil_sen_slope": np.nan, "theil_sen_intercept": np.nan, "slope_ci_lower": np.nan, "slope_ci_upper": np.nan,
        }
    rho, p = stats.spearmanr(xv, yv)
    slope, intercept, _, _ = stats.theilslopes(yv, xv)

    # Basin-level bootstrap
    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_boot, dtype=np.float64)
    boot_slopes = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx, by = xv[idx], yv[idx]
        if len(np.unique(bx)) < 3:
            boot_rhos[b] = rho
            boot_slopes[b] = slope
            continue
        br, _ = stats.spearmanr(bx, by)
        bs, _, _, _ = stats.theilslopes(by, bx)
        boot_rhos[b] = br
        boot_slopes[b] = bs

    rho_ci = np.nanquantile(boot_rhos, [0.025, 0.975])
    slope_ci = np.nanquantile(boot_slopes, [0.025, 0.975])

    return {
        "n": n,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "rho_ci_lower": float(rho_ci[0]),
        "rho_ci_upper": float(rho_ci[1]),
        "theil_sen_slope": float(slope),
        "theil_sen_intercept": float(intercept),
        "slope_ci_lower": float(slope_ci[0]),
        "slope_ci_upper": float(slope_ci[1]),
    }


# ---------------------------------------------------------------------------
# Timing metrics helper
# ---------------------------------------------------------------------------


def compute_basin_year_timing(
    basin_id: str,
    dates: np.ndarray,
    w_model: np.ndarray,
    sm_ref: np.ndarray,
    swe_ref: np.ndarray,
) -> List[Dict[str, Any]]:
    """Compute peak timing and spring wet-up timing per water year."""
    d = pd.to_datetime(dates)
    wy = np.where(d.month >= 10, d.year + 1, d.year).astype(int)
    starts = np.array([np.datetime64(f"{int(w) - 1}-10-01", "D") for w in wy])
    doy = ((d.values - starts) / np.timedelta64(1, "D")).astype(float) + 1  # Oct 1 = 1

    records = []
    for w in np.unique(wy):
        mask = wy == w
        if mask.sum() < 300:
            continue
        dw = doy[mask]
        dates_w = d[mask]
        wm_w = w_model[mask]
        sm_w = sm_ref[mask]
        swe_w = swe_ref[mask]

        # Snow activity filter for the year
        annual_swe_max = float(np.nanmax(swe_w))
        annual_wm_max = float(np.nanmax(wm_w))
        annual_sm_max = float(np.nanmax(sm_w))
        is_snow_year = (annual_swe_max >= MIN_ANNUAL_SWE_PEAK_MM) and (annual_wm_max >= MIN_ANNUAL_W_PEAK_MM)

        # 1. Soil-water peak timing (doy of maximum)
        peak_idx_ref = int(np.nanargmax(sm_w))
        peak_idx_mod = int(np.nanargmax(wm_w))
        peak_doy_ref = float(dw[peak_idx_ref])
        peak_doy_mod = float(dw[peak_idx_mod])
        peak_timing_error = float(peak_doy_mod - peak_doy_ref) if is_snow_year else np.nan

        # 2. Spring wet-up / recharge timing
        # Window: Jan 1 to Jun 30 (months 1..6)
        spring_mask = (dates_w.month >= 1) & (dates_w.month <= 6)
        if spring_mask.sum() >= 60 and is_snow_year:
            # 14-day rate of increase: x[t+7] - x[t-7]
            def calc_14d_diff(arr):
                s = pd.Series(arr)
                diff14 = s.shift(-7) - s.shift(7)
                return diff14.to_numpy()

            diff_ref = calc_14d_diff(sm_w)
            diff_mod = calc_14d_diff(wm_w)

            # Restrict search to spring window
            diff_ref_spring = np.where(spring_mask, diff_ref, -999.0)
            diff_mod_spring = np.where(spring_mask, diff_mod, -999.0)

            wetup_idx_ref = int(np.nanargmax(diff_ref_spring))
            wetup_idx_mod = int(np.nanargmax(diff_mod_spring))
            wetup_doy_ref = float(dw[wetup_idx_ref])
            wetup_doy_mod = float(dw[wetup_idx_mod])
            wetup_timing_error = float(wetup_doy_mod - wetup_doy_ref)
        else:
            wetup_doy_ref = np.nan
            wetup_doy_mod = np.nan
            wetup_timing_error = np.nan

        records.append({
            "basin_id": basin_id,
            "water_year": int(w),
            "is_snow_year": is_snow_year,
            "annual_swe_max_mm": annual_swe_max,
            "peak_doy_ref": peak_doy_ref,
            "peak_doy_model": peak_doy_mod,
            "peak_timing_error_days": peak_timing_error,
            "wetup_doy_ref": wetup_doy_ref,
            "wetup_doy_model": wetup_doy_mod,
            "wetup_timing_error_days": wetup_timing_error,
        })
    return records


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_soil_consistency_analysis() -> Dict[str, Any]:
    print("=" * 80)
    print("R4 FORMAL REAL-BASIN SOIL-WATER STATE CONSISTENCY ANALYSIS")
    print("=" * 80)

    # 1. Load Caravan soil reference & Snow-17 burden
    caravan = np.load(CARAVAN_REF_DIR / "caravan_soil_ensemble.npz")
    basin_ids = [str(b).zfill(8) for b in caravan["basin_ids"]]
    n_basins = len(basin_ids)
    dates_full = caravan["dates"]

    test_start_idx = int(caravan["test_slice_start"])
    test_stop_idx = int(caravan["test_slice_stop"])
    test_sl = slice(test_start_idx, test_stop_idx)
    test_dates = dates_full[test_sl]
    months_test = pd.to_datetime(test_dates).month.values

    sm100_test = caravan["SM100"][:, test_sl].astype(np.float64)
    sm289_test = caravan["SM289"][:, test_sl].astype(np.float64)
    caravan_swe_test = caravan["caravan_swe"][:, test_sl].astype(np.float64)

    burden_df = pd.read_csv(SWE_REF_DIR / "swe_basin_burden_test.csv", dtype={"basin_id": str}).set_index("basin_id")
    bundle = load_bundle(default_data_root())
    frac_snow = bundle.raw_attributes[:, 3]

    print(f"Loaded Caravan soil reference: {n_basins} basins x {len(test_dates)} test days (1995-10-01 .. 2010-09-30)")

    # 2. Define models to analyze
    # Canonical dPL (seeds 42, 123) + IC fused sensitivity
    regimes = [
        ("dPL_seed42", "official_dpl", 42, OFFICIAL_DPL_OBSERVATION_TRAINED),
        ("dPL_seed123", "official_dpl", 123, OFFICIAL_DPL_OBSERVATION_TRAINED),
        ("IC_fused", "ic_fused", None, IC_FUSED_5x200_SENSITIVITY),
    ]

    consistency_rows = []
    paired_rows = []
    timing_year_rows = []
    timing_summary_rows = []

    res_root = default_results_root()

    for regime_name, prefix, seed, tag in regimes:
        print(f"\nProcessing regime: {regime_name} (tag={tag})...")

        # Load Base & CN W_total
        if seed is not None:
            base_dir = res_root / f"r4_{prefix}_XAJ_seed{seed}"
            cn_dir = res_root / f"r4_{prefix}_XAJ_CN_seed{seed}"
            base_npz = np.load(base_dir / f"{prefix}_XAJ_seed{seed}_full_arrays.npz")
            cn_npz = np.load(cn_dir / f"{prefix}_XAJ_CN_seed{seed}_full_arrays.npz")
        else:
            base_dir = res_root / f"r4_{prefix}_XAJ"
            cn_dir = res_root / f"r4_{prefix}_XAJ_CN"
            base_npz = np.load(base_dir / f"{prefix}_XAJ_full_arrays.npz")
            cn_npz = np.load(cn_dir / f"{prefix}_XAJ_CN_full_arrays.npz")

        w_base = (base_npz["wu"][:, test_sl] + base_npz["wl"][:, test_sl] + base_npz["wd"][:, test_sl]).astype(np.float64)
        w_cn = (cn_npz["wu"][:, test_sl] + cn_npz["wl"][:, test_sl] + cn_npz["wd"][:, test_sl]).astype(np.float64)

        # Per-basin consistency calculations
        for i, b in enumerate(basin_ids):
            b_burden_swe = float(burden_df.loc[b, "median_annual_max_swe_mm"])
            b_swe_days = float(burden_df.loc[b, "median_swe_positive_days"])
            b_fs = float(frac_snow[i])
            is_snow_active = b_burden_swe >= 20.0

            ref_sm100 = sm100_test[i]
            ref_sm289 = sm289_test[i]
            ref_swe = caravan_swe_test[i]

            wb = w_base[i]
            wc = w_cn[i]

            # 7-day smoothing
            wb_7d = smooth_7d(wb)
            wc_7d = smooth_7d(wc)
            ref_7d = smooth_7d(ref_sm100)

            # Anomaly series (monthly climatology removed)
            wb_anom = calendar_month_anomaly(wb, months_test)
            wc_anom = calendar_month_anomaly(wc, months_test)
            ref_anom = calendar_month_anomaly(ref_sm100, months_test)

            # Monthly aggregation
            wb_m, m_dates = monthly_aggregate(wb, test_dates)
            wc_m, _ = monthly_aggregate(wc, test_dates)
            ref_m, _ = monthly_aggregate(ref_sm100, test_dates)

            # Metrics for Base vs SM100
            r_daily_base = float(stats.pearsonr(wb, ref_sm100)[0])
            rho_daily_base = float(stats.spearmanr(wb, ref_sm100)[0])
            r_7d_base = float(stats.pearsonr(wb_7d, ref_7d)[0])
            r_anom_base = float(stats.pearsonr(wb_anom, ref_anom)[0])
            nrmse_base = zscore_nrmse(wb, ref_sm100)
            r_m_base = float(stats.pearsonr(wb_m, ref_m)[0])

            # Sensitivity vs SM289
            r_daily_base_sm289 = float(stats.pearsonr(wb, ref_sm289)[0])
            r_anom_base_sm289 = float(stats.pearsonr(wb_anom, calendar_month_anomaly(ref_sm289, months_test))[0])

            # Metrics for CN vs SM100
            r_daily_cn = float(stats.pearsonr(wc, ref_sm100)[0])
            rho_daily_cn = float(stats.spearmanr(wc, ref_sm100)[0])
            r_7d_cn = float(stats.pearsonr(wc_7d, ref_7d)[0])
            r_anom_cn = float(stats.pearsonr(wc_anom, ref_anom)[0])
            nrmse_cn = zscore_nrmse(wc, ref_sm100)
            r_m_cn = float(stats.pearsonr(wc_m, ref_m)[0])

            # Sensitivity vs SM289
            r_daily_cn_sm289 = float(stats.pearsonr(wc, ref_sm289)[0])
            r_anom_cn_sm289 = float(stats.pearsonr(wc_anom, calendar_month_anomaly(ref_sm289, months_test))[0])

            # Store consistency rows
            for struct, r_d, rho_d, r_7, r_a, nrmse, r_mo, r_d289, r_a289 in [
                ("Base", r_daily_base, rho_daily_base, r_7d_base, r_anom_base, nrmse_base, r_m_base, r_daily_base_sm289, r_anom_base_sm289),
                ("CN", r_daily_cn, rho_daily_cn, r_7d_cn, r_anom_cn, nrmse_cn, r_m_cn, r_daily_cn_sm289, r_anom_cn_sm289),
            ]:
                consistency_rows.append({
                    "regime": regime_name,
                    "structure": struct,
                    "tag": tag,
                    "basin_id": b,
                    "raw_daily_corr": r_d,
                    "raw_daily_spearman": rho_d,
                    "smoothed_7d_corr": r_7,
                    "monthly_anomaly_corr": r_a,
                    "zscore_nrmse": nrmse,
                    "monthly_mean_corr": r_mo,
                    "sm289_raw_daily_corr": r_d289,
                    "sm289_anomaly_corr": r_a289,
                    "snow_burden_swe_mm": b_burden_swe,
                    "swe_positive_days": b_swe_days,
                    "frac_snow": b_fs,
                    "is_snow_active": is_snow_active,
                })

            # Paired structural contrast Delta(CN - Base)
            paired_rows.append({
                "regime": regime_name,
                "tag": tag,
                "basin_id": b,
                "delta_raw_daily_corr": r_daily_cn - r_daily_base,
                "delta_7d_corr": r_7d_cn - r_7d_base,
                "delta_anomaly_corr": r_anom_cn - r_anom_base,
                "delta_nrmse": nrmse_cn - nrmse_base,  # Negative delta = CN has lower error
                "delta_monthly_corr": r_m_cn - r_m_base,
                "delta_sm289_daily_corr": r_daily_cn_sm289 - r_daily_base_sm289,
                "delta_sm289_anomaly_corr": r_anom_cn_sm289 - r_anom_base_sm289,
                "snow_burden_swe_mm": b_burden_swe,
                "swe_positive_days": b_swe_days,
                "frac_snow": b_fs,
                "is_snow_active": is_snow_active,
            })

            # Timing calculations (Base and CN)
            timing_years_base = compute_basin_year_timing(b, test_dates, wb, ref_sm100, ref_swe)
            timing_years_cn = compute_basin_year_timing(b, test_dates, wc, ref_sm100, ref_swe)

            for rec in timing_years_base:
                timing_year_rows.append({"regime": regime_name, "structure": "Base", **rec})
            for rec in timing_years_cn:
                timing_year_rows.append({"regime": regime_name, "structure": "CN", **rec})

            # Basin-level timing summaries
            def summarize_timing(t_rows):
                snow_rows = [r for r in t_rows if r["is_snow_year"] and np.isfinite(r["peak_timing_error_days"])]
                n_v = len(snow_rows)
                if n_v < MIN_VALID_YEARS_PER_BASIN:
                    return {
                        "n_valid_snow_years": n_v,
                        "median_peak_error_days": np.nan,
                        "mean_peak_error_days": np.nan,
                        "median_abs_peak_error_days": np.nan,
                        "iqr_peak_error_days": np.nan,
                        "median_wetup_error_days": np.nan,
                        "mean_wetup_error_days": np.nan,
                        "median_abs_wetup_error_days": np.nan,
                        "iqr_wetup_error_days": np.nan,
                    }
                p_errs = np.array([r["peak_timing_error_days"] for r in snow_rows])
                w_errs = np.array([r["wetup_timing_error_days"] for r in snow_rows if np.isfinite(r["wetup_timing_error_days"])])

                return {
                    "n_valid_snow_years": n_v,
                    "median_peak_error_days": float(np.nanmedian(p_errs)),
                    "mean_peak_error_days": float(np.nanmean(p_errs)),
                    "median_abs_peak_error_days": float(np.nanmedian(np.abs(p_errs))),
                    "iqr_peak_error_days": float(np.nanpercentile(p_errs, 75) - np.nanpercentile(p_errs, 25)),
                    "median_wetup_error_days": float(np.nanmedian(w_errs)) if len(w_errs) >= MIN_VALID_YEARS_PER_BASIN else np.nan,
                    "mean_wetup_error_days": float(np.nanmean(w_errs)) if len(w_errs) >= MIN_VALID_YEARS_PER_BASIN else np.nan,
                    "median_abs_wetup_error_days": float(np.nanmedian(np.abs(w_errs))) if len(w_errs) >= MIN_VALID_YEARS_PER_BASIN else np.nan,
                    "iqr_wetup_error_days": float(np.nanpercentile(w_errs, 75) - np.nanpercentile(w_errs, 25)) if len(w_errs) >= MIN_VALID_YEARS_PER_BASIN else np.nan,
                }

            timing_summary_rows.append({"regime": regime_name, "structure": "Base", "basin_id": b, **summarize_timing(timing_years_base)})
            timing_summary_rows.append({"regime": regime_name, "structure": "CN", "basin_id": b, **summarize_timing(timing_years_cn)})

    df_consistency = pd.DataFrame(consistency_rows)
    df_paired = pd.DataFrame(paired_rows)
    df_timing_year = pd.DataFrame(timing_year_rows)
    df_timing_summary = pd.DataFrame(timing_summary_rows)

    # 3. Snow-burden regression & quantile analysis
    report_summary: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_basins": n_basins,
        "n_test_days": len(test_dates),
        "primary_reference": "Caravan v1.1 CAMELS-US ERA5-Land SM100 (0-100 cm depth-weighted composite)",
        "sensitivity_reference": "SM289 (0-289 cm full profile)",
        "primary_model_state": "W_total = wu + wl + wd [mm]",
        "tgd2_status": "TGD2_PENDING (no observation-trained canonical TGD2 checkpoint available; not substituted with legacy TGD)",
        "regimes": {},
    }

    quantile_rows = []

    for regime_name in ["dPL_seed42", "dPL_seed123", "IC_fused"]:
        sub_p = df_paired[df_paired["regime"] == regime_name]
        sub_c = df_consistency[df_consistency["regime"] == regime_name]
        sub_t = df_timing_summary[df_timing_summary["regime"] == regime_name]

        # Snow burden regressions for primary metrics
        reg_results = {}
        for target_col in ["delta_anomaly_corr", "delta_7d_corr", "delta_raw_daily_corr", "delta_monthly_corr", "delta_nrmse"]:
            reg_results[f"{target_col}_vs_swe_burden"] = theil_sen_bootstrap(
                sub_p["snow_burden_swe_mm"].to_numpy(), sub_p[target_col].to_numpy()
            )
            reg_results[f"{target_col}_vs_swe_days"] = theil_sen_bootstrap(
                sub_p["swe_positive_days"].to_numpy(), sub_p[target_col].to_numpy()
            )
            reg_results[f"{target_col}_vs_frac_snow"] = theil_sen_bootstrap(
                sub_p["frac_snow"].to_numpy(), sub_p[target_col].to_numpy()
            )
            # High-snow active subset
            active_mask = sub_p["is_snow_active"].to_numpy()
            reg_results[f"{target_col}_vs_swe_burden_snow_active_only"] = theil_sen_bootstrap(
                sub_p.loc[active_mask, "snow_burden_swe_mm"].to_numpy(), sub_p.loc[active_mask, target_col].to_numpy()
            )

        # Quantile breakdown (Q0..Q3 by Snow-17 SWE burden)
        q_labels = ["Q0 (0-2 mm)", "Q1 (2-35 mm)", "Q2 (35-212 mm)", "Q3 (212-1866 mm)"]
        q_bins = pd.qcut(sub_p["snow_burden_swe_mm"], 4, labels=q_labels, duplicates="drop")
        sub_p = sub_p.copy()
        sub_p["swe_quartile"] = q_bins

        q_summary = {}
        for qv in q_labels:
            q_idx = sub_p["swe_quartile"] == qv
            b_ids_q = sub_p.loc[q_idx, "basin_id"].values
            n_q = int(q_idx.sum())

            # Base & CN median consistency
            c_base_q = sub_c[(sub_c["structure"] == "Base") & (sub_c["basin_id"].isin(b_ids_q))]
            c_cn_q = sub_c[(sub_c["structure"] == "CN") & (sub_c["basin_id"].isin(b_ids_q))]

            delta_anom_q = sub_p.loc[q_idx, "delta_anomaly_corr"].values
            delta_7d_q = sub_p.loc[q_idx, "delta_7d_corr"].values
            delta_daily_q = sub_p.loc[q_idx, "delta_raw_daily_corr"].values
            delta_nrmse_q = sub_p.loc[q_idx, "delta_nrmse"].values

            q_entry = {
                "n": n_q,
                "swe_burden_median_mm": float(np.nanmedian(sub_p.loc[q_idx, "snow_burden_swe_mm"])),
                "base_median_anomaly_corr": float(np.nanmedian(c_base_q["monthly_anomaly_corr"])),
                "cn_median_anomaly_corr": float(np.nanmedian(c_cn_q["monthly_anomaly_corr"])),
                "delta_anomaly_corr_median": float(np.nanmedian(delta_anom_q)),
                "delta_anomaly_corr_iqr": float(np.nanpercentile(delta_anom_q, 75) - np.nanpercentile(delta_anom_q, 25)),
                "base_median_7d_corr": float(np.nanmedian(c_base_q["smoothed_7d_corr"])),
                "cn_median_7d_corr": float(np.nanmedian(c_cn_q["smoothed_7d_corr"])),
                "delta_7d_corr_median": float(np.nanmedian(delta_7d_q)),
                "delta_raw_daily_corr_median": float(np.nanmedian(delta_daily_q)),
                "delta_nrmse_median": float(np.nanmedian(delta_nrmse_q)),
            }
            q_summary[qv] = q_entry

            quantile_rows.append({
                "regime": regime_name,
                "quartile": qv,
                **q_entry,
            })

        # Overall state consistency summary
        c_base = sub_c[sub_c["structure"] == "Base"]
        c_cn = sub_c[sub_c["structure"] == "CN"]
        c_base_act = sub_c[(sub_c["structure"] == "Base") & (sub_c["is_snow_active"])]
        c_cn_act = sub_c[(sub_c["structure"] == "CN") & (sub_c["is_snow_active"])]

        t_base = sub_t[sub_t["structure"] == "Base"]
        t_cn = sub_t[sub_t["structure"] == "CN"]

        report_summary["regimes"][regime_name] = {
            "overall_all_531_basins": {
                "base_median_anomaly_corr": float(np.nanmedian(c_base["monthly_anomaly_corr"])),
                "cn_median_anomaly_corr": float(np.nanmedian(c_cn["monthly_anomaly_corr"])),
                "delta_anomaly_corr_median": float(np.nanmedian(sub_p["delta_anomaly_corr"])),
                "base_median_7d_corr": float(np.nanmedian(c_base["smoothed_7d_corr"])),
                "cn_median_7d_corr": float(np.nanmedian(c_cn["smoothed_7d_corr"])),
                "delta_7d_corr_median": float(np.nanmedian(sub_p["delta_7d_corr"])),
                "base_median_raw_daily_corr": float(np.nanmedian(c_base["raw_daily_corr"])),
                "cn_median_raw_daily_corr": float(np.nanmedian(c_cn["raw_daily_corr"])),
                "delta_raw_daily_corr_median": float(np.nanmedian(sub_p["delta_raw_daily_corr"])),
                "base_median_monthly_corr": float(np.nanmedian(c_base["monthly_mean_corr"])),
                "cn_median_monthly_corr": float(np.nanmedian(c_cn["monthly_mean_corr"])),
                "delta_monthly_corr_median": float(np.nanmedian(sub_p["delta_monthly_corr"])),
                "base_median_nrmse": float(np.nanmedian(c_base["zscore_nrmse"])),
                "cn_median_nrmse": float(np.nanmedian(c_cn["zscore_nrmse"])),
                "delta_nrmse_median": float(np.nanmedian(sub_p["delta_nrmse"])),
            },
            "snow_active_subset_352_basins": {
                "base_median_anomaly_corr": float(np.nanmedian(c_base_act["monthly_anomaly_corr"])),
                "cn_median_anomaly_corr": float(np.nanmedian(c_cn_act["monthly_anomaly_corr"])),
                "delta_anomaly_corr_median": float(np.nanmedian(sub_p.loc[sub_p["is_snow_active"], "delta_anomaly_corr"])),
                "base_median_7d_corr": float(np.nanmedian(c_base_act["smoothed_7d_corr"])),
                "cn_median_7d_corr": float(np.nanmedian(c_cn_act["smoothed_7d_corr"])),
                "delta_7d_corr_median": float(np.nanmedian(sub_p.loc[sub_p["is_snow_active"], "delta_7d_corr"])),
                "base_median_raw_daily_corr": float(np.nanmedian(c_base_act["raw_daily_corr"])),
                "cn_median_raw_daily_corr": float(np.nanmedian(c_cn_act["raw_daily_corr"])),
                "delta_raw_daily_corr_median": float(np.nanmedian(sub_p.loc[sub_p["is_snow_active"], "delta_raw_daily_corr"])),
                "base_median_monthly_corr": float(np.nanmedian(c_base_act["monthly_mean_corr"])),
                "cn_median_monthly_corr": float(np.nanmedian(c_cn_act["monthly_mean_corr"])),
                "delta_monthly_corr_median": float(np.nanmedian(sub_p.loc[sub_p["is_snow_active"], "delta_monthly_corr"])),
            },
            "timing_errors": {
                "base_median_peak_error_days": float(np.nanmedian(t_base["median_peak_error_days"])),
                "base_median_abs_peak_error_days": float(np.nanmedian(t_base["median_abs_peak_error_days"])),
                "cn_median_peak_error_days": float(np.nanmedian(t_cn["median_peak_error_days"])),
                "cn_median_abs_peak_error_days": float(np.nanmedian(t_cn["median_abs_peak_error_days"])),
                "base_median_wetup_error_days": float(np.nanmedian(t_base["median_wetup_error_days"])),
                "base_median_abs_wetup_error_days": float(np.nanmedian(t_base["median_abs_wetup_error_days"])),
                "cn_median_wetup_error_days": float(np.nanmedian(t_cn["median_wetup_error_days"])),
                "cn_median_abs_wetup_error_days": float(np.nanmedian(t_cn["median_abs_wetup_error_days"])),
            },
            "snow_burden_regressions": reg_results,
            "snow_burden_quartiles": q_summary,
        }

    df_quantiles = pd.DataFrame(quantile_rows)

    # 4. Save clean figure-ready tables
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_consistency.to_csv(OUT_DIR / "basin_state_consistency.csv", index=False)
    df_paired.to_csv(OUT_DIR / "paired_structural_effects.csv", index=False)
    df_timing_year.to_csv(OUT_DIR / "timing_metrics_basin_year.csv", index=False)
    df_timing_summary.to_csv(OUT_DIR / "timing_metrics_basin_summary.csv", index=False)
    df_quantiles.to_csv(OUT_DIR / "snow_burden_quartile_summary.csv", index=False)

    report_path = OUT_DIR / "r4_phase1_soil_official_report.json"
    report_path.write_text(json.dumps(report_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSaved all figure-ready tables and JSON report to {OUT_DIR}/")

    return report_summary


if __name__ == "__main__":
    run_soil_consistency_analysis()
