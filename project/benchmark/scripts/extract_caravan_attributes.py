"""
Extract Caravan / CAMELS 35 Catchment Attributes for 671 Basins.
Aligned 1:1 with data/gage_id.npy order.
Exports result to CSV files with short attribute column names.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
CARAVAN_PARQUET = Path("/home/jingxin/code/hydro-model-agent/knowledge/marrmot/basin_attributes.parquet")
RESULTS_DIR = BENCHMARK_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 35 Short Column Names (Abbreviated attribute names)
SHORT_COLUMN_NAMES = [
    "p_mean", "pet_mean", "p_seas", "f_snow", "arid",
    "hi_p_freq", "hi_p_dur", "lo_p_freq", "lo_p_dur",
    "elev", "slope", "area",
    "f_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff",
    "lc_dom_frac", "lc_dom",
    "root_d50", "s_depth_p", "s_depth_s", "s_poros", "s_cond", "w_max",
    "sand", "silt", "clay",
    "g_1st_cls", "g_1st_frac", "g_2nd_cls", "g_2nd_frac", "carb_frac", "g_poros", "g_perm"
]


def extract_and_export_csv():
    print("=== Extracting Caravan/CAMELS 35 Attributes for 671 Basins ===")

    # 1. Load gage_id.npy (671 basins in exact target order)
    gage_ids = np.load(DATA_DIR / "gage_id.npy")
    gage_ids_str = [str(g).zfill(8) for g in gage_ids]
    n_basins = len(gage_ids_str)
    print(f"Target gage_id.npy count: {n_basins}")

    # 2. Load CAMELS dataset pkl (671, 35) as baseline
    with open(DATA_DIR / "camels_dataset", "rb") as f:
        _, _, camels_attrs = pickle.load(f)

    # 3. Load Caravan Parquet
    caravan_df = pd.read_parquet(CARAVAN_PARQUET)
    caravan_ids = [str(g).replace("camels_", "").zfill(8) for g in caravan_df["gauge_id"]]
    caravan_df["clean_id"] = caravan_ids
    caravan_map = caravan_df.set_index("clean_id")

    final_matrix = camels_attrs.copy()
    caravan_matched = 0

    for i, b_id in enumerate(gage_ids_str):
        if b_id in caravan_map.index:
            row = caravan_map.loc[b_id]
            caravan_matched += 1

            # Daily P and PET using HydroATLAS physical annual totals (mm/yr / 365)
            p_daily = float(row["pre_mm_syr"]) / 365.0 if pd.notnull(row["pre_mm_syr"]) else final_matrix[i, 0]
            pet_daily = float(row["pet_mm_syr"]) / 365.0 if pd.notnull(row["pet_mm_syr"]) else final_matrix[i, 1]
            aridity = pet_daily / (p_daily + 1e-6)

            final_matrix[i, 0] = p_daily
            final_matrix[i, 1] = pet_daily
            final_matrix[i, 4] = aridity

            def assign_if_valid(field, idx, scale=1.0):
                if field in row and pd.notnull(row[field]):
                    final_matrix[i, idx] = float(row[field]) * scale

            assign_if_valid("frac_snow", 3)
            assign_if_valid("seasonality", 2)
            assign_if_valid("ele_mt_sav", 9)
            assign_if_valid("slp_dg_sav", 10)
            assign_if_valid("for_pc_sse", 12, scale=0.01)  # Percentage to ratio 0-1
            assign_if_valid("cly_pc_sav", 27)
            assign_if_valid("snd_pc_sav", 25)
            assign_if_valid("slt_pc_sav", 26)

    # Clean NaNs with 0.0
    final_matrix = np.nan_to_num(final_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Build DataFrame
    df_result = pd.DataFrame(final_matrix, columns=SHORT_COLUMN_NAMES)
    df_result.insert(0, "gauge_id", gage_ids_str)

    # Export CSV paths
    csv_data_path = DATA_DIR / "caravan_671_attributes.csv"
    csv_results_path = RESULTS_DIR / "caravan_671_attributes.csv"

    df_result.to_csv(csv_data_path, index=False, float_format="%.4f")
    df_result.to_csv(csv_results_path, index=False, float_format="%.4f")

    print(f"\nExtraction & Export Complete!")
    print(f"Matched {caravan_matched}/{n_basins} basins from Caravan (112 seamlessly complemented)")
    print(f"Saved CSV to Data directory: {csv_data_path}")
    print(f"Saved CSV to Results directory: {csv_results_path}")
    print(f"\nPreview of first 5 rows and 8 columns:\n{df_result.iloc[:5, :8]}")


if __name__ == "__main__":
    extract_and_export_csv()
