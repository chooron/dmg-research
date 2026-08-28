#!/usr/bin/env python3
"""Build canonical population audit dataset for Supplementary Figure S1.

Calculates the outcome-independent snowiest-year consistency metrics across all
531 CAMELS-US catchments evaluated under R4, producing:
1. `manuscript/supplement/figures/FigureS1_R4_population_audit.csv`
2. `manuscript/supplement/figures/FigureS1_R4_selection_audit.json`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results"
SUPPLEMENT_FIG_DIR = PROJECT_ROOT / "manuscript" / "supplement" / "figures"
SUPPLEMENT_FIG_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SHARED = PROJECT_ROOT / "manuscript" / "scripts" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
from manuscript.scripts.r4.soil_analysis import calendar_month_anomaly

MIN_SNOW_ACTIVE_DAYS = 10  # Minimum valid days with SWE >= 5.0 mm in snowiest water year
SWE_ACTIVE_THRESHOLD_MM = 5.0
EXAMPLE_BASIN_IDS = ["02472000", "07195800", "05495000", "03473000", "12167000", "08377900"]


def calc_standardized_anomaly(arr: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Compute standardized calendar-month anomalies along axis 1."""
    months = dates.month.to_numpy()
    anom = np.empty_like(arr, dtype=float)
    for m in range(1, 13):
        m_mask = (months == m)
        anom[:, m_mask] = arr[:, m_mask] - np.nanmean(arr[:, m_mask], axis=1, keepdims=True)
    stds = np.nanstd(anom, axis=1, keepdims=True)
    stds[stds < 1e-9] = 1.0
    return (anom - np.nanmean(anom, axis=1, keepdims=True)) / stds


def build_audit_data():
    caravan = np.load(RESULTS_ROOT / "r4_caravan_soil_reference_v1" / "caravan_soil_ensemble.npz")
    test = slice(int(caravan["test_slice_start"]), int(caravan["test_slice_stop"]))
    dates = pd.to_datetime(caravan["dates"][test])
    basin_ids = [str(x).zfill(8) for x in caravan["basin_ids"]]
    sm100 = caravan["SM100"][:, test].astype(float)
    swe = np.load(RESULTS_ROOT / "r4_swe_reference_v1" / "swe_ensemble.npz")["swe_median"][:, test].astype(float)

    tgd_dpl_path = RESULTS_ROOT / "r4_replay_dpl_XAJ_TGD2_seed42/reconstructed_dpl_XAJ_TGD2_seed42_full_arrays.npz"
    if not tgd_dpl_path.exists():
        tgd_dpl_path = RESULTS_ROOT / "r4_official_dpl_XAJ_TGD2_seed42/official_dpl_XAJ_TGD2_seed42_full_arrays.npz"

    npz_base = np.load(RESULTS_ROOT / "r4_official_dpl_XAJ_seed42/official_dpl_XAJ_seed42_full_arrays.npz")
    npz_tgd = np.load(tgd_dpl_path)
    npz_cn = np.load(RESULTS_ROOT / "r4_official_dpl_XAJ_CN_seed42/official_dpl_XAJ_CN_seed42_full_arrays.npz")

    w_base = (npz_base["wu"][:, test] + npz_base["wl"][:, test] + npz_base["wd"][:, test]).astype(float)
    w_tgd = (npz_tgd["wu"][:, test] + npz_tgd["wl"][:, test] + npz_tgd["wd"][:, test]).astype(float)
    w_cn = (npz_cn["wu"][:, test] + npz_cn["wl"][:, test] + npz_cn["wd"][:, test]).astype(float)

    d = pd.DatetimeIndex(dates)
    wy = np.where(d.month >= 10, d.year + 1, d.year).astype(int)
    water_years = np.unique(wy)

    anom_sm = calc_standardized_anomaly(sm100, d)
    anom_base = calc_standardized_anomaly(w_base, d)
    anom_tgd = calc_standardized_anomaly(w_tgd, d)
    anom_cn = calc_standardized_anomaly(w_cn, d)

    # 1. Long-term mean annual peak SWE
    mean_peak_swe = []
    for i in range(len(basin_ids)):
        b_swe = swe[i]
        maxs = [np.nanmax(b_swe[wy == y]) for y in water_years]
        mean_peak_swe.append(float(np.nanmean(maxs)))
    mean_peak_swe = np.array(mean_peak_swe)

    t1, t2 = np.percentile(mean_peak_swe, [33.333, 66.667])

    records = []
    for i, bid in enumerate(basin_ids):
        s_mean = mean_peak_swe[i]
        if s_mean <= t1:
            grp = "Low"
        elif s_mean <= t2:
            grp = "Middle"
        else:
            grp = "High"

        b_swe = swe[i]
        yearly_peak = {int(y): float(np.nanmax(b_swe[wy == y])) for y in water_years}
        best_wy = max(yearly_peak, key=yearly_peak.get)
        pk_val = yearly_peak[best_wy]

        wy_mask = (wy == best_wy)
        snow_active_mask = wy_mask & (swe[i] >= SWE_ACTIVE_THRESHOLD_MM)
        n_days = int(snow_active_mask.sum())

        if n_days >= MIN_SNOW_ACTIVE_DAYS:
            y_ref = anom_sm[i, snow_active_mask]
            y_b = anom_base[i, snow_active_mask]
            y_t = anom_tgd[i, snow_active_mask]
            y_c = anom_cn[i, snow_active_mask]

            r_b = float(np.corrcoef(y_ref, y_b)[0, 1]) if np.std(y_b) > 1e-6 and np.std(y_ref) > 1e-6 else np.nan
            r_t = float(np.corrcoef(y_ref, y_t)[0, 1]) if np.std(y_t) > 1e-6 and np.std(y_ref) > 1e-6 else np.nan
            r_c = float(np.corrcoef(y_ref, y_c)[0, 1]) if np.std(y_c) > 1e-6 and np.std(y_ref) > 1e-6 else np.nan

            d_cn_b = r_c - r_b if np.isfinite(r_c) and np.isfinite(r_b) else np.nan
            d_tgd_b = r_t - r_b if np.isfinite(r_t) and np.isfinite(r_b) else np.nan
            eligible = bool(np.isfinite(d_cn_b) and np.isfinite(d_tgd_b))
        else:
            r_b, r_t, r_c = np.nan, np.nan, np.nan
            d_cn_b, d_tgd_b = np.nan, np.nan
            eligible = False

        records.append({
            "basin_id": bid,
            "swe_burden_group": grp,
            "mean_annual_peak_swe_mm": round(float(s_mean), 2),
            "snowiest_water_year": int(best_wy),
            "snowiest_year_peak_swe_mm": round(float(pk_val), 2),
            "snow_active_days": int(n_days),
            "eligible": eligible,
            "r_Base": round(float(r_b), 4) if np.isfinite(r_b) else np.nan,
            "r_TGD": round(float(r_t), 4) if np.isfinite(r_t) else np.nan,
            "r_CN": round(float(r_c), 4) if np.isfinite(r_c) else np.nan,
            "delta_r_CN_Base": round(float(d_cn_b), 4) if np.isfinite(d_cn_b) else np.nan,
            "delta_r_TGD_Base": round(float(d_tgd_b), 4) if np.isfinite(d_tgd_b) else np.nan,
            "is_example_basin": bool(bid in EXAMPLE_BASIN_IDS),
        })

    df_pop = pd.DataFrame(records)
    csv_path = SUPPLEMENT_FIG_DIR / "FigureS1_R4_population_audit.csv"
    df_pop.to_csv(csv_path, index=False)

    # Build updated JSON selection audit
    def pick_two_basins(group_name):
        sub = df_pop[df_pop["swe_burden_group"] == group_name].sort_values("mean_annual_peak_swe_mm").reset_index(drop=True)
        r1 = sub.iloc[int(len(sub) * 0.33)]
        r2 = sub.iloc[int(len(sub) * 0.67)]
        return [
            {
                "group": group_name,
                "basin_id": str(r1["basin_id"]),
                "mean_annual_peak_swe_mm": float(r1["mean_annual_peak_swe_mm"]),
                "selected_water_year": int(r1["snowiest_water_year"]),
                "water_year_peak_swe_mm": float(r1["snowiest_year_peak_swe_mm"]),
            },
            {
                "group": group_name,
                "basin_id": str(r2["basin_id"]),
                "mean_annual_peak_swe_mm": float(r2["mean_annual_peak_swe_mm"]),
                "selected_water_year": int(r2["snowiest_water_year"]),
                "water_year_peak_swe_mm": float(r2["snowiest_year_peak_swe_mm"]),
            },
        ]

    selected_basins = pick_two_basins("Low") + pick_two_basins("Middle") + pick_two_basins("High")

    # Bootstrap CIs for JSON summary
    rng = np.random.default_rng(20260730)
    group_summaries = {}
    for grp in ["Low", "Middle", "High"]:
        sub_el = df_pop[(df_pop["swe_burden_group"] == grp) & (df_pop["eligible"])]
        d_cn = sub_el["delta_r_CN_Base"].dropna().values
        d_tgd = sub_el["delta_r_TGD_Base"].dropna().values
        
        boot_cn = [float(np.median(rng.choice(d_cn, size=len(d_cn), replace=True))) for _ in range(2000)]
        boot_tgd = [float(np.median(rng.choice(d_tgd, size=len(d_tgd), replace=True))) for _ in range(2000)]
        
        group_summaries[grp] = {
            "total_basins": int(len(df_pop[df_pop["swe_burden_group"] == grp])),
            "eligible_basins": int(len(sub_el)),
            "delta_r_CN_Base": {
                "median": round(float(np.median(d_cn)), 4),
                "ci_95": [round(float(np.percentile(boot_cn, 2.5)), 4), round(float(np.percentile(boot_cn, 97.5)), 4)],
                "prop_positive": round(float((d_cn > 0).mean()), 3),
            },
            "delta_r_TGD_Base": {
                "median": round(float(np.median(d_tgd)), 4),
                "ci_95": [round(float(np.percentile(boot_tgd, 2.5)), 4), round(float(np.percentile(boot_tgd, 97.5)), 4)],
                "prop_positive": round(float((d_tgd > 0).mean()), 3),
            },
        }

    audit_payload = {
        "selection_rule": "Deterministic outcome-independent external Snow-17 SWE tercile ranking (fixed ~33% and ~67% intra-tercile ranks)",
        "evaluation_period": "1995-10-01 to 2010-09-30 (15 water years)",
        "tercile_thresholds_mm": {"low_tercile_max": round(float(t1), 2), "mid_tercile_max": round(float(t2), 2)},
        "eligibility_rule": f"At least {MIN_SNOW_ACTIVE_DAYS} snow-active days (SWE >= {SWE_ACTIVE_THRESHOLD_MM} mm) in snowiest water year",
        "selected_example_basins": selected_basins,
        "population_summaries": group_summaries,
    }

    json_path = SUPPLEMENT_FIG_DIR / "FigureS1_R4_selection_audit.json"
    json_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    print(f"Population audit generated successfully:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    build_audit_data()
