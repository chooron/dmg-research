#!/usr/bin/env python3
"""Prepare source tables for the frozen R1 Figure 1 and minimal supplement."""
from __future__ import annotations

import shutil
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

from R1X_common import canonical_ids, load_full_table
from r1_config import CACHE_DIR, DATA_ROOT, FIGURES_DIR, MODEL_REGISTRY, REPO, TABLES_DIR
from src.model_registry import get_spec

LOCATION_SOURCE = DATA_ROOT / "camels_loc" / "camels_671_loc.dbf"
STATE_SOURCE = REPO / "project" / "hydrodiag" / "manuscript" / "cache" / "gis" / "us_states.geojson"
STATE_CACHE = CACHE_DIR / "gis" / "us_states.geojson"


def load_locations() -> pd.DataFrame:
    if not LOCATION_SOURCE.is_file():
        raise RuntimeError(f"missing CAMELS location source: {LOCATION_SOURCE}")
    locations = pyogrio.read_dataframe(LOCATION_SOURCE)
    locations["basin_id"] = locations["gage_id"].map(lambda x: str(int(float(x))).zfill(8))
    locations = locations[["basin_id", "lat", "lon"]].rename(columns={"lat": "latitude", "lon": "longitude"})
    if locations.duplicated("basin_id").any():
        raise RuntimeError("CAMELS location source contains duplicate basin IDs")
    ids = canonical_ids()
    locations = locations[locations["basin_id"].isin(ids)].copy()
    if len(locations) != len(ids) or not np.isfinite(locations[["latitude", "longitude"]].to_numpy(float)).all():
        raise RuntimeError("CAMELS location join is incomplete or contains non-finite coordinates")
    return locations


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_CACHE.is_file():
        if not STATE_SOURCE.is_file():
            raise RuntimeError(f"missing read-only hydrodiag GIS source: {STATE_SOURCE}")
        STATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(STATE_SOURCE, STATE_CACHE)

    full = load_full_table()
    model_summary = pd.read_csv(TABLES_DIR / "R1_model_performance_summary.csv")
    tendency = pd.read_csv(TABLES_DIR / "R1X_model_estimator_tendency.csv")
    temporal = pd.read_csv(TABLES_DIR / "R1_temporal_AB_model_summary.csv")
    params = {model: int(get_spec(model, device="cpu").dimension) for model in MODEL_REGISTRY}

    fig1a = model_summary[["model", "IC_median", "dPL_median", "delta_median"]].rename(columns={
        "IC_median": "IC_median_KGE", "dPL_median": "dPL_median_KGE", "delta_median": "delta_model_median",
    }).copy()
    fig1a["plot_order"] = fig1a["model"].map({model: i for i, model in enumerate(MODEL_REGISTRY)})
    fig1a["n_params"] = fig1a["model"].map(params)
    fig1a["n_states"] = pd.NA
    fig1a = fig1a.sort_values("plot_order")
    fig1a.to_csv(TABLES_DIR / "Fig1a_model_aggregate_performance.csv", index=False, float_format="%.10f")

    fig1b = tendency[["model", "delta_Q10", "delta_Q25", "delta_median", "delta_Q75", "delta_Q90", "frac_dPL_gt_IC"]].rename(columns={
        "frac_dPL_gt_IC": "P_m_positive",
    }).copy()
    fig1b["plot_order"] = fig1b["model"].map({model: i for i, model in enumerate(MODEL_REGISTRY)})
    fig1b = fig1b.sort_values("plot_order")
    fig1b.to_csv(TABLES_DIR / "Fig1b_model_delta_distribution.csv", index=False, float_format="%.10f")

    locations = load_locations()
    basin_rows = []
    for basin_id in canonical_ids():
        values = full.loc[full["basin_id"] == basin_id, "delta_KGE"].to_numpy(float)
        loc = locations.loc[locations["basin_id"] == basin_id].iloc[0]
        median_delta = float(np.median(values))
        mad_delta = float(np.median(np.abs(values - median_delta)))
        basin_rows.append({
            "basin_id": basin_id,
            "latitude": float(loc.latitude), "longitude": float(loc.longitude),
            "n_models": int(values.size), "n_dPL_higher": int((values > 0).sum()),
            "n_IC_higher": int((values < 0).sum()), "n_ties": int((values == 0).sum()),
            "P_b_dPL": float(np.mean(values > 0)),
            "M_b_median_deltaKGE": median_delta, "S_b_MAD_deltaKGE": mad_delta,
            "geometry_status": "gauge_point",
        })
    fig1c = pd.DataFrame(basin_rows).sort_values(["longitude", "latitude", "basin_id"], kind="stable").reset_index(drop=True)
    fig1c["plot_order"] = np.arange(len(fig1c))
    fig1c.to_csv(TABLES_DIR / "Fig1c_basin_crossmodel_tendency.csv", index=False, float_format="%.10f")

    # v6 GIS: median basin effect is the sole map color; MAD is inset-only spread.
    fig1c["abs_M_b"] = fig1c["M_b_median_deltaKGE"].abs()
    q95_abs_m = float(np.quantile(fig1c["abs_M_b"], 0.95))
    color_limit = float(np.ceil(q95_abs_m * 100.0) / 100.0)
    if color_limit <= 0.0:
        color_limit = 0.01
    fig1c["color_clipped_flag"] = np.select(
        [fig1c["M_b_median_deltaKGE"] < -color_limit, fig1c["M_b_median_deltaKGE"] > color_limit],
        ["low", "high"],
        default="",
    )
    fig1c_v6 = fig1c[[
        "basin_id", "latitude", "longitude", "n_models",
        "M_b_median_deltaKGE", "S_b_MAD_deltaKGE", "P_b_dPL",
        "abs_M_b", "color_clipped_flag", "plot_order",
    ]].rename(columns={"n_models": "N_models", "P_b_dPL": "P_b_positive"})
    fig1c_v6.to_csv(TABLES_DIR / "Fig1c_basin_effect_spread.csv", index=False, float_format="%.10f")
    map_metadata = {
        "map_variable": "M_b_median_deltaKGE",
        "spread_variable": "S_b_MAD_deltaKGE",
        "spread_definition": "median absolute deviation across 36 model structures",
        "selected_color_limit_L": color_limit,
        "color_limit_rule": "ceil(Q95(abs(M_b)) to 0.01)",
        "q95_abs_M_b": q95_abs_m,
        "number_clipped_low": int((fig1c["M_b_median_deltaKGE"] < -color_limit).sum()),
        "number_clipped_high": int((fig1c["M_b_median_deltaKGE"] > color_limit).sum()),
        "number_clipped_total": int((fig1c["abs_M_b"] > color_limit).sum()),
        "n_basins": int(len(fig1c_v6)),
        "n_models": 36,
        "median_abs_M_b": float(fig1c["abs_M_b"].median()),
        "fraction_abs_M_b_less_than_S_b": float((fig1c["abs_M_b"] < fig1c["S_b_MAD_deltaKGE"]).mean()),
        "map_markers_equal_size": True,
        "map_uses_spatial_interpolation": False,
    }
    (CACHE_DIR / "Fig1c_map_metadata.json").write_text(json.dumps(map_metadata, indent=2) + "\n")

    decomposition = pd.read_csv(TABLES_DIR / "R1X_two_way_decomposition.csv")
    decomposition.to_csv(TABLES_DIR / "Fig1d_two_way_decomposition.csv", index=False, float_format="%.12f")

    fig1e = temporal[["model", "delta_median_A", "delta_median_B", "same_sign_non_neutral"]].copy()
    fig1e["plot_order"] = fig1e["model"].map({model: i for i, model in enumerate(MODEL_REGISTRY)})
    fig1e = fig1e.sort_values("plot_order")
    fig1e.to_csv(TABLES_DIR / "Fig1e_temporal_model_persistence.csv", index=False, float_format="%.10f")

    print(f"PASS: prepared Figure 1 source tables; GIS points={len(fig1c)}; state cache={STATE_CACHE}")


if __name__ == "__main__":
    main()
