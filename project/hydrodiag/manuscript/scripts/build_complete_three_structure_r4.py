#!/usr/bin/env python3
"""Build complete Three-Structure (Base, TGD2, CN) R4 statistical and robustness tables."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "project" / "hydrodiag"
sys.path.insert(0, str(PROJECT))

from r4.common import default_data_root, default_results_root, load_bundle, zfill8
from r4.soil_analysis import (
    calendar_month_anomaly,
    smooth_7d,
    zscore_nrmse,
    compute_basin_year_timing,
    MIN_VALID_YEARS_PER_BASIN,
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
)
from r4.robustness_analysis import bootstrap_median_ci


PHASE_SPECS = [
    ("Phase_1_Snow_Accumulation", 1, lambda m, swe: (m >= 10) | (m <= 2)),
    ("Phase_2_Active_Melt_Recharge", 2, lambda m, swe: (m >= 3) & (m <= 5)),
    ("Phase_3_Post_Melt_Transition", 3, lambda m, swe: (m >= 6) & (m <= 7)),
    ("Phase_4_Summer_Dry_Down", 4, lambda m, swe: (m >= 8) & (m <= 9)),
]


def run_full_three_structure_analysis():
    results_root = default_results_root()
    r4_dir = results_root / "r4_phase1_soil_official"
    caravan = np.load(results_root / "r4_caravan_soil_reference_v1" / "caravan_soil_ensemble.npz")
    basin_ids = [str(b).zfill(8) for b in caravan["basin_ids"]]
    n_basins = len(basin_ids)
    dates_full = caravan["dates"]
    
    test_start_idx = int(caravan["test_slice_start"])
    test_stop_idx = int(caravan["test_slice_stop"])
    test_sl = slice(test_start_idx, test_stop_idx)
    test_dates = pd.to_datetime(dates_full[test_sl])
    months_test = test_dates.month.values
    
    sm100_test = caravan["SM100"][:, test_sl].astype(np.float64)
    swe_test = np.load(results_root / "r4_swe_reference_v1" / "swe_ensemble.npz")["swe_median"][:, test_sl].astype(np.float64)
    burden_df = pd.read_csv(results_root / "r4_swe_reference_v1" / "swe_basin_burden_test.csv", dtype={"basin_id": str}).set_index("basin_id")
    
    # 1. Regimes to evaluate
    regime_configs = [
        ("dPL_seed42", {
            "Base": results_root / "r4_official_dpl_XAJ_seed42/official_dpl_XAJ_seed42_full_arrays.npz",
            "CN": results_root / "r4_official_dpl_XAJ_CN_seed42/official_dpl_XAJ_CN_seed42_full_arrays.npz",
            "TGD2": results_root / "r4_official_dpl_XAJ_TGD2_seed42/official_dpl_XAJ_TGD2_seed42_full_arrays.npz",
        }),
        ("dPL_seed123", {
            "Base": results_root / "r4_official_dpl_XAJ_seed123/official_dpl_XAJ_seed123_full_arrays.npz",
            "CN": results_root / "r4_official_dpl_XAJ_CN_seed123/official_dpl_XAJ_CN_seed123_full_arrays.npz",
            "TGD2": results_root / "r4_official_dpl_XAJ_TGD2_seed123/official_dpl_XAJ_TGD2_seed123_full_arrays.npz",
        }),
        ("dPL_seed2026", {
            "Base": results_root / "r4_official_dpl_XAJ_seed42/official_dpl_XAJ_seed42_full_arrays.npz",
            "CN": results_root / "r4_official_dpl_XAJ_CN_seed42/official_dpl_XAJ_CN_seed42_full_arrays.npz",
            "TGD2": results_root / "r4_official_dpl_XAJ_TGD2_seed2026/official_dpl_XAJ_TGD2_seed2026_full_arrays.npz",
        }),
        ("IC_fused", {
            "Base": results_root / "r4_ic_fused_XAJ/ic_fused_XAJ_full_arrays.npz",
            "CN": results_root / "r4_ic_fused_XAJ_CN/ic_fused_XAJ_CN_full_arrays.npz",
            "TGD2": results_root / "r4_ic_fused_XAJ_TGD2/ic_fused_XAJ_TGD2_full_arrays.npz",
        }),
    ]

    all_basin_state_rows = []
    all_paired_rows = []
    all_phase_rows = []
    all_timing_year_rows = []
    all_timing_summary_rows = []
    all_decile_rows = []

    for reg_name, path_dict in regime_configs:
        print(f"--- Analyzing Regime: {reg_name} ---", flush=True)
        w_dict = {}
        for struct_name, npz_path in path_dict.items():
            if npz_path.exists():
                npz = np.load(npz_path)
                w_dict[struct_name] = (npz["wu"][:, test_sl] + npz["wl"][:, test_sl] + npz["wd"][:, test_sl]).astype(np.float64)
            else:
                print(f"Warning: {npz_path} does not exist!")

        # Per-basin consistency
        anom_corrs = {}
        daily_corrs = {}
        
        for struct_name, w_mod in w_dict.items():
            anom_corrs[struct_name] = np.zeros(n_basins)
            daily_corrs[struct_name] = np.zeros(n_basins)
            
            for i, b_id in enumerate(basin_ids):
                sm_ref = sm100_test[i]
                wm = w_mod[i]
                swe_i = swe_test[i]
                
                ref_anom = calendar_month_anomaly(sm_ref, months_test)
                mod_anom = calendar_month_anomaly(wm, months_test)
                anom_corr = float(np.corrcoef(ref_anom, mod_anom)[0, 1])
                anom_corrs[struct_name][i] = anom_corr
                
                ref_7d = smooth_7d(sm_ref)
                mod_7d = smooth_7d(wm)
                valid_7d = np.isfinite(ref_7d) & np.isfinite(mod_7d)
                corr_7d = float(np.corrcoef(ref_7d[valid_7d], mod_7d[valid_7d])[0, 1])
                
                raw_corr = float(np.corrcoef(sm_ref, wm)[0, 1])
                daily_corrs[struct_name][i] = raw_corr
                nrmse = float(zscore_nrmse(wm, sm_ref))
                
                swe_max = float(burden_df.loc[b_id, "median_annual_max_swe_mm"])
                swe_pos = float(burden_df.loc[b_id, "median_swe_positive_days"])
                
                all_basin_state_rows.append({
                    "regime": reg_name,
                    "structure": struct_name,
                    "basin_id": b_id,
                    "snow_burden_swe_mm": swe_max,
                    "swe_positive_days": swe_pos,
                    "daily_corr": raw_corr,
                    "smoothed_7d_corr": corr_7d,
                    "monthly_anomaly_corr": anom_corr,
                    "nrmse": nrmse,
                })
                
                # Timing for this structure
                timing_recs = compute_basin_year_timing(
                    b_id, test_dates, wm, sm_ref, swe_i,
                )
                for r in timing_recs:
                    r["regime"] = reg_name
                    r["structure"] = struct_name
                    all_timing_year_rows.append(r)

        # Paired contrasts
        if "Base" in anom_corrs and "CN" in anom_corrs and "TGD2" in anom_corrs:
            for i, b_id in enumerate(basin_ids):
                swe_max = float(burden_df.loc[b_id, "median_annual_max_swe_mm"])
                all_paired_rows.append({
                    "regime": reg_name,
                    "basin_id": b_id,
                    "snow_burden_swe_mm": swe_max,
                    "base_anomaly_corr": anom_corrs["Base"][i],
                    "cn_anomaly_corr": anom_corrs["CN"][i],
                    "tgd2_anomaly_corr": anom_corrs["TGD2"][i],
                    "delta_cn_base_anomaly": anom_corrs["CN"][i] - anom_corrs["Base"][i],
                    "delta_tgd2_base_anomaly": anom_corrs["TGD2"][i] - anom_corrs["Base"][i],
                    "delta_cn_tgd2_anomaly": anom_corrs["CN"][i] - anom_corrs["TGD2"][i],
                })

        # Process phase consistency (snow active basins SWE >= 20 mm)
        for i, b_id in enumerate(basin_ids):
            swe_max = float(burden_df.loc[b_id, "median_annual_max_swe_mm"])
            if swe_max < 20.0:
                continue
            sm_ref = sm100_test[i]
            swe_i = swe_test[i]
            
            for p_name, p_code, p_mask_fn in PHASE_SPECS:
                mask = p_mask_fn(months_test, swe_i)
                if mask.sum() < 30:
                    continue
                
                ref_sub = sm_ref[mask]
                ref_anom_sub = calendar_month_anomaly(sm_ref, months_test)[mask]
                
                base_anom_p = float(np.corrcoef(ref_anom_sub, calendar_month_anomaly(w_dict["Base"][i], months_test)[mask])[0, 1])
                cn_anom_p = float(np.corrcoef(ref_anom_sub, calendar_month_anomaly(w_dict["CN"][i], months_test)[mask])[0, 1])
                tgd2_anom_p = float(np.corrcoef(ref_anom_sub, calendar_month_anomaly(w_dict["TGD2"][i], months_test)[mask])[0, 1])
                
                all_phase_rows.append({
                    "regime": reg_name,
                    "basin_id": b_id,
                    "snow_burden_swe_mm": swe_max,
                    "phase_code": p_code,
                    "phase_name": p_name,
                    "n_days": int(mask.sum()),
                    "base_anomaly_corr": base_anom_p,
                    "cn_anomaly_corr": cn_anom_p,
                    "tgd2_anomaly_corr": tgd2_anom_p,
                    "delta_anomaly_corr": cn_anom_p - base_anom_p,
                    "delta_tgd2_base_anomaly": tgd2_anom_p - base_anom_p,
                })

        # Decile shapes for CN-Base and TGD2-Base
        swe_vals = np.array([float(burden_df.loc[b, "median_annual_max_swe_mm"]) for b in basin_ids])
        decile_bins = np.percentile(swe_vals, np.linspace(0, 100, 11))
        decile_bins[-1] += 1e-3
        decile_assignments = np.digitize(swe_vals, decile_bins) - 1
        decile_assignments = np.clip(decile_assignments, 0, 9)
        
        for d in range(10):
            mask_d = decile_assignments == d
            # CN - Base
            vals_cn = anom_corrs["CN"][mask_d] - anom_corrs["Base"][mask_d]
            m_cn, l_cn, h_cn = bootstrap_median_ci(vals_cn)
            # TGD2 - Base
            vals_tgd = anom_corrs["TGD2"][mask_d] - anom_corrs["Base"][mask_d]
            m_tgd, l_tgd, h_tgd = bootstrap_median_ci(vals_tgd)
            
            all_decile_rows.append({
                "regime": reg_name,
                "decile": d + 1,
                "n_basins": int(mask_d.sum()),
                "swe_min_mm": float(decile_bins[d]),
                "swe_max_mm": float(decile_bins[d+1]),
                "delta_cn_base_median": m_cn,
                "delta_cn_base_ci_lower": l_cn,
                "delta_cn_base_ci_upper": h_cn,
                "delta_tgd2_base_median": m_tgd,
                "delta_tgd2_base_ci_lower": l_tgd,
                "delta_tgd2_base_ci_upper": h_tgd,
            })

    # Convert to DataFrames and save
    df_all_basin = pd.DataFrame(all_basin_state_rows)
    df_all_paired = pd.DataFrame(all_paired_rows)
    df_all_phase = pd.DataFrame(all_phase_rows)
    df_all_decile = pd.DataFrame(all_decile_rows)
    df_all_timing_year = pd.DataFrame(all_timing_year_rows)
    
    # Timing summary per basin
    for (reg, struct, b_id), group in df_all_timing_year.groupby(["regime", "structure", "basin_id"]):
        snow_group = group[group["is_snow_year"]]
        n_valid = len(snow_group)
        if n_valid >= MIN_VALID_YEARS_PER_BASIN:
            peak_errs = snow_group["peak_timing_error_days"].dropna().to_numpy()
            wetup_errs = snow_group["wetup_timing_error_days"].dropna().to_numpy()
            all_timing_summary_rows.append({
                "regime": reg,
                "structure": struct,
                "basin_id": b_id,
                "n_valid_snow_years": n_valid,
                "median_peak_error_days": float(np.median(peak_errs)) if len(peak_errs) else np.nan,
                "median_abs_peak_error_days": float(np.median(np.abs(peak_errs))) if len(peak_errs) else np.nan,
                "iqr_peak_error_days": float(stats.iqr(peak_errs)) if len(peak_errs) else np.nan,
                "median_wetup_error_days": float(np.median(wetup_errs)) if len(wetup_errs) else np.nan,
                "median_abs_wetup_error_days": float(np.median(np.abs(wetup_errs))) if len(wetup_errs) else np.nan,
                "iqr_wetup_error_days": float(stats.iqr(wetup_errs)) if len(wetup_errs) else np.nan,
            })
    df_all_timing_summary = pd.DataFrame(all_timing_summary_rows)

    df_all_basin.to_csv(r4_dir / "three_structure_basin_state_consistency.csv", index=False)
    df_all_paired.to_csv(r4_dir / "three_structure_paired_structural_effects.csv", index=False)
    df_all_phase.to_csv(r4_dir / "three_structure_process_phase_consistency.csv", index=False)
    df_all_decile.to_csv(r4_dir / "three_structure_swe_decile_shape.csv", index=False)
    df_all_timing_year.to_csv(r4_dir / "three_structure_timing_metrics_basin_year.csv", index=False)
    df_all_timing_summary.to_csv(r4_dir / "three_structure_timing_metrics_basin_summary.csv", index=False)

    print("\nSUCCESS: All Three-Structure R4 Statistical & Robustness Tables Built!", flush=True)


if __name__ == "__main__":
    run_full_three_structure_analysis()
