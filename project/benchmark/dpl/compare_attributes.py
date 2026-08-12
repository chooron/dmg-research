"""
Quantitative Statistical Comparison Script: Caravan vs CAMELS 35 Catchment Attributes.
Computes distribution metrics (mean, std, min, max, median, IQR),
error/bias metrics (MAE, RMSE, Rel Bias %), correlation (Pearson r, Spearman rho),
and NaN rates across 531 CAMELS basins.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# 1. Paths Setup
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
CARAVAN_PARQUET = Path("/home/jingxin/code/hydro-model-agent/knowledge/marrmot/basin_attributes.parquet")
OUTPUT_DIR = BENCHMARK_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 35 CAMELS attribute names
CAMELS_35_NAMES = [
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "elev_mean", "slope_mean", "area_gages2",
    "frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff",
    "dom_land_cover_frac", "dom_land_cover",
    "root_depth_50", "soil_depth_pelletier", "soil_depth_statsgo",
    "soil_porosity", "soil_conductivity", "max_water_content",
    "sand_frac", "silt_frac", "clay_frac",
    "geol_1st_class", "glim_1st_class_frac", "geol_2nd_class",
    "glim_2nd_class_frac", "carbonate_rocks_frac", "geol_porosity", "geol_permeability"
]

# Mapping CAMELS attribute names to corresponding Caravan column names (if available in HydroATLAS / Caravan)
CARAVAN_MAPPING = {
    "p_mean": "p_mean",
    "pet_mean": "pet_mean",
    "p_seasonality": "seasonality",
    "frac_snow": "frac_snow",
    "aridity": "aridity",
    "high_prec_freq": "high_prec_freq",
    "high_prec_dur": "high_prec_dur",
    "low_prec_freq": "low_prec_freq",
    "low_prec_dur": "low_prec_dur",
    "elev_mean": "ele_mt_sav",
    "slope_mean": "slp_dg_sav",
    "area_gages2": "area",
    "frac_forest": "for_pc_sse",
    "clay_frac": "cly_pc_sav",
    "sand_frac": "snd_pc_sav",
    "silt_frac": "slt_pc_sav",
    "soil_organic": "soc_th_sav",
}

def main():
    print("=== Loading Data for Numerical Statistical Comparison ===")
    
    # Load CAMELS dataset pickle
    with open(DATA_DIR / "camels_dataset", "rb") as f:
        _, _, raw_camels_attrs = pickle.load(f)
    
    gage_ids = np.load(DATA_DIR / "gage_id.npy")
    gage_ids_str = [str(g).zfill(8) for g in gage_ids]

    with open(DATA_DIR / "531sub_id.txt") as f:
        sub531_ids = json.loads(f.read().strip())
    sub531_str = [str(b).zfill(8) for b in sub531_ids]

    camels_indices = [gage_ids_str.index(b) for b in sub531_str]
    camels_531_mat = raw_camels_attrs[camels_indices]  # Shape: (531, 35)

    # Load Caravan parquet dataframe
    caravan_df = pd.read_parquet(CARAVAN_PARQUET)
    caravan_ids = [str(g).replace("camels_", "").zfill(8) for g in caravan_df["gauge_id"]]
    caravan_sub_df = caravan_df.set_index(pd.Index(caravan_ids)).reindex(sub531_str)

    stats_rows = []

    for col_idx, attr_name in enumerate(CAMELS_35_NAMES):
        c_vals = camels_531_mat[:, col_idx]
        
        # Calculate CAMELS stats
        c_nan_cnt = np.isnan(c_vals).sum()
        c_valid = c_vals[~np.isnan(c_vals)]
        
        c_mean = float(np.mean(c_valid)) if len(c_valid) > 0 else np.nan
        c_std = float(np.std(c_valid)) if len(c_valid) > 0 else np.nan
        c_min = float(np.min(c_valid)) if len(c_valid) > 0 else np.nan
        c_p25 = float(np.percentile(c_valid, 25)) if len(c_valid) > 0 else np.nan
        c_median = float(np.median(c_valid)) if len(c_valid) > 0 else np.nan
        c_p75 = float(np.percentile(c_valid, 75)) if len(c_valid) > 0 else np.nan
        c_max = float(np.max(c_valid)) if len(c_valid) > 0 else np.nan
        c_iqr = c_p75 - c_p25

        caravan_col = CARAVAN_MAPPING.get(attr_name)
        
        if caravan_col and caravan_col in caravan_sub_df.columns:
            k_vals = caravan_sub_df[caravan_col].to_numpy(dtype=float)
            k_nan_cnt = np.isnan(k_vals).sum()
            
            # Mask paired valid values
            pair_mask = ~np.isnan(c_vals) & ~np.isnan(k_vals)
            c_pair = c_vals[pair_mask]
            k_pair = k_vals[pair_mask]
            
            k_mean = float(np.mean(k_pair)) if len(k_pair) > 0 else np.nan
            k_std = float(np.std(k_pair)) if len(k_pair) > 0 else np.nan
            k_min = float(np.min(k_pair)) if len(k_pair) > 0 else np.nan
            k_p25 = float(np.percentile(k_pair, 25)) if len(k_pair) > 0 else np.nan
            k_median = float(np.median(k_pair)) if len(k_pair) > 0 else np.nan
            k_p75 = float(np.percentile(k_pair, 75)) if len(k_pair) > 0 else np.nan
            k_max = float(np.max(k_pair)) if len(k_pair) > 0 else np.nan
            k_iqr = k_p75 - k_p25
            
            mae = float(np.mean(np.abs(k_pair - c_pair))) if len(c_pair) > 0 else np.nan
            rmse = float(np.sqrt(np.mean((k_pair - c_pair) ** 2))) if len(c_pair) > 0 else np.nan
            rel_bias = float((k_mean - c_mean) / (abs(c_mean) + 1e-6) * 100.0)
            
            if len(c_pair) > 2 and np.std(c_pair) > 1e-6 and np.std(k_pair) > 1e-6:
                pr_val, _ = pearsonr(c_pair, k_pair)
                sr_val, _ = spearmanr(c_pair, k_pair)
            else:
                pr_val, sr_val = np.nan, np.nan
                
            status = "Matched"
        else:
            k_nan_cnt, k_mean, k_std, k_min, k_p25, k_median, k_p75, k_max, k_iqr = [np.nan] * 9
            mae, rmse, rel_bias, pr_val, sr_val = [np.nan] * 5
            status = "CAMELS Only (HydroATLAS Derived)"

        # Classification category
        if status == "Matched":
            if pr_val > 0.95 and abs(rel_bias) < 15.0:
                category = "A: High Consistency"
            elif pr_val > 0.80 and abs(rel_bias) >= 15.0:
                category = "B: Scaled / Unit Shifted"
            else:
                category = "C: Discrepant / Method Diff"
        else:
            category = "D: Non-Caravan Direct Field"

        stats_rows.append({
            "attr_name": attr_name,
            "caravan_mapped_col": caravan_col if caravan_col else "N/A",
            "category": category,
            "camels_nan_cnt": c_nan_cnt,
            "camels_mean": c_mean,
            "camels_std": c_std,
            "camels_min": c_min,
            "camels_median": c_median,
            "camels_max": c_max,
            "camels_iqr": c_iqr,
            "caravan_nan_cnt": k_nan_cnt,
            "caravan_mean": k_mean,
            "caravan_std": k_std,
            "caravan_min": k_min,
            "caravan_median": k_median,
            "caravan_max": k_max,
            "caravan_iqr": k_iqr,
            "mae": mae,
            "rmse": rmse,
            "rel_bias_pct": rel_bias,
            "pearson_r": pr_val,
            "spearman_rho": sr_val,
        })

    res_df = pd.DataFrame(stats_rows)
    csv_path = OUTPUT_DIR / "caravan_vs_camels_numeric_stats.csv"
    res_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved full stats CSV to {csv_path}")

    # Generate Markdown Summary
    md_path = OUTPUT_DIR / "caravan_vs_camels_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Caravan vs CAMELS 35 Basin Attributes Numerical Comparison Report\n\n")
        f.write(f"**Total Analyzed Basins**: 531 | **Total Attributes**: 35\n\n")
        f.write("## Summary Statistics Table (Matched Attributes)\n\n")
        f.write("| Attribute | Caravan Col | Category | CAMELS Mean (±Std) | Caravan Mean (±Std) | Rel Bias (%) | Pearson r | Spearman rho |\n")
        f.write("|:---|:---|:---|:---|:---|:---|:---|:---|\n")
        for r in stats_rows:
            if r["caravan_mapped_col"] != "N/A":
                c_str = f"{r['camels_mean']:.3f} (±{r['camels_std']:.3f})"
                k_str = f"{r['caravan_mean']:.3f} (±{r['caravan_std']:.3f})"
                f.write(f"| `{r['attr_name']}` | `{r['caravan_mapped_col']}` | {r['category']} | {c_str} | {k_str} | {r['rel_bias_pct']:+.1f}% | {r['pearson_r']:.4f} | {r['spearman_rho']:.4f} |\n")
    print(f"Saved summary report to {md_path}")

if __name__ == "__main__":
    main()
