#!/usr/bin/env python3
"""Agent C: parameter realization distance and archived IC restart identifiability."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    ALL_MODELS,
    RESULTS,
    SEED,
    corr,
    load_inputs,
    load_ic_restart,
    load_status,
    write_csv,
)


def main() -> None:
    ids, paired, params, distance, _, _, _ = load_inputs()
    paired = paired.copy()
    paired["G_seen"] = paired.KGE_IC - paired.KGE_dPL
    distance = distance.copy()
    distance["D_theta_rms"] = distance.normalized_l2_distance / np.sqrt(distance.parameter_count)
    distance["D_theta"] = distance["D_theta_rms"]
    distance["distance_definition"] = "sqrt(mean_j((u_IC-u_dPL)^2)); u=(theta-L)/(U-L)"
    write_csv(RESULTS / "agent_C/C01_PARAMETER_REALIZATION_DISTANCE.csv", distance)

    c02 = distance.groupby("model", sort=True).agg(
        basins=("basin_id", "nunique"), parameter_count=("parameter_count", "first"),
        D_theta_median=("D_theta_rms", "median"), D_theta_mean=("D_theta_rms", "mean"),
        D_theta_q25=("D_theta_rms", lambda x: x.quantile(.25)), D_theta_q75=("D_theta_rms", lambda x: x.quantile(.75)),
        normalized_mean_abs_median=("normalized_mean_abs_distance", "median"), normalized_max_abs_median=("normalized_max_abs_distance", "median"),
        physical_mean_abs_median=("physical_mean_abs_difference", "median"),
    ).reset_index()
    c02["label"] = "SEEN_BASIN_PARAMETER_REALIZATION_DISTANCE"
    write_csv(RESULTS / "agent_C/C02_MODEL_PARAMETER_DISTANCE_SUMMARY.csv", c02)

    model_rows, pooled_x, pooled_y = [], [], []
    for model in ALL_MODELS:
        g = paired[paired.model == model].set_index("basin_id").G_seen
        d = distance[distance.model == model].set_index("basin_id").D_theta
        joined = pd.concat([g, d], axis=1).dropna()
        rho, p, n = corr(joined.G_seen, joined.D_theta)
        model_rows.append({"analysis_level": "within_model", "model": model, "rho_G_seen_D_theta": rho, "p_value": p, "n": n, "label": "G_SEEN_VS_PARAMETER_DISTANCE"})
        pooled_x.extend(joined.G_seen.tolist()); pooled_y.extend(joined.D_theta.tolist())
    rho, p, n = corr(pooled_x, pooled_y)
    model_rows.append({"analysis_level": "pooled_model_basin", "model": "ALL_36", "rho_G_seen_D_theta": rho, "p_value": p, "n": n, "label": "G_SEEN_VS_PARAMETER_DISTANCE"})
    # Rank within model before pooling, which removes model-level location/scale effects.
    rank_frame = pd.DataFrame({"G_seen": pooled_x, "D_theta": pooled_y})
    rank_frame["model"] = np.repeat(ALL_MODELS, [len(paired[paired.model == m]) for m in ALL_MODELS])
    rank_frame["G_rank_within_model"] = rank_frame.groupby("model").G_seen.rank(method="average")
    rank_frame["D_rank_within_model"] = rank_frame.groupby("model").D_theta.rank(method="average")
    rho, p, n = corr(rank_frame.G_rank_within_model, rank_frame.D_rank_within_model)
    model_rows.append({"analysis_level": "within_model_rank_pooled", "model": "ALL_36", "rho_G_seen_D_theta": rho, "p_value": p, "n": n, "label": "G_SEEN_VS_PARAMETER_DISTANCE"})
    write_csv(RESULTS / "agent_C/C03_GAP_VS_PARAMETER_DISTANCE.csv", pd.DataFrame(model_rows))

    model_gap = paired.groupby("model").G_seen.median().rename("G_seen_median")
    model_dist = distance.groupby("model").D_theta.median().rename("D_theta_median")
    frame = pd.concat([model_gap, model_dist], axis=1).reset_index()
    rho, p, n = corr(frame.G_seen_median, frame.D_theta_median)
    write_csv(RESULTS / "agent_C/C04_MODEL_LEVEL_GAP_DISTANCE.csv", pd.DataFrame([{"analysis_level": "model_level_median", "rho_G_seen_D_theta": rho, "p_value": p, "n": n, "label": "MODEL_LEVEL_G_SEEN_VS_PARAMETER_DISTANCE"}]))

    status = load_status()
    parameter_rows, basin_rows, availability = [], [], []
    for model in ALL_MODELS:
        u, physical, fitness, meta = load_ic_restart(model, ids, status)
        availability.append({"model": model, "ic_generation": meta["generation"], "basin_count": len(ids), "starts_per_basin": meta["starts"], "parameter_count": u.shape[2], "restart_latent_archived": True, "restart_fitness_archived": True, "coverage": "complete canonical 531 basins", "label": "IC_RESTART_DATA_AVAILABILITY"})
        names = meta["parameter_names"]
        for b, basin in enumerate(ids):
            order = np.argsort(fitness[b])[::-1]
            best, second = order[0], order[1]
            best_second_u_distance = float(np.linalg.norm(u[b, best] - u[b, second]) / np.sqrt(u.shape[2]))
            fitness_sorted = fitness[b, order]
            basin_rows.append({"model": model, "basin_id": str(int(basin)).zfill(8), "ic_generation": meta["generation"], "restart_count": 10, "fitness_best": float(fitness_sorted[0]), "fitness_second": float(fitness_sorted[1]), "fitness_best_second_gap": float(fitness_sorted[0] - fitness_sorted[1]), "fitness_spread": float(fitness[b].max() - fitness[b].min()), "selected_start_index": int(best), "best_second_u_rms_distance": best_second_u_distance, "mean_restart_u_sd": float(u[b].std(axis=0).mean()), "mean_restart_u_iqr": float(np.mean(np.quantile(u[b], .75, axis=0) - np.quantile(u[b], .25, axis=0))), "max_restart_u_sd": float(u[b].std(axis=0).max()), "label": "IC_RESTART_BASIN_IDENTIFIABILITY"})
            for p, name in enumerate(names):
                up = u[b, :, p]
                pp = physical[b, :, p]
                parameter_rows.append({"model": model, "basin_id": str(int(basin)).zfill(8), "parameter_index": p, "parameter": name, "ic_generation": meta["generation"], "restart_count": 10, "restart_u_sd": float(np.std(up)), "restart_u_iqr": float(np.quantile(up, .75) - np.quantile(up, .25)), "restart_u_range": float(np.max(up) - np.min(up)), "restart_physical_sd": float(np.std(pp)), "restart_physical_iqr": float(np.quantile(pp, .75) - np.quantile(pp, .25)), "best_second_u_distance": float(abs(up[best] - up[second])), "fitness_best_second_gap": float(fitness_sorted[0] - fitness_sorted[1]), "label": "IC_RESTART_PARAMETER_UNCERTAINTY"})
    write_csv(RESULTS / "agent_C/C05_IC_RESTART_PARAMETER_UNCERTAINTY.csv", pd.DataFrame(parameter_rows))
    c06 = pd.DataFrame(basin_rows)
    write_csv(RESULTS / "agent_C/C06_IC_RESTART_BASIN_IDENTIFIABILITY.csv", c06)
    write_csv(RESULTS / "agent_C/C05_RESTART_DATA_AVAILABILITY_GATE.csv", pd.DataFrame(availability))

    # Model-level identifiability linkage is kept here for D and for deterministic QC.
    model_restart = c06.groupby("model", sort=True).agg(restart_mean_u_sd=("mean_restart_u_sd", "median"), restart_mean_u_iqr=("mean_restart_u_iqr", "median"), restart_max_u_sd=("max_restart_u_sd", "median"), best_second_u_distance=("best_second_u_rms_distance", "median"), fitness_best_second_gap=("fitness_best_second_gap", "median"), fitness_spread=("fitness_spread", "median")).reset_index()
    model_restart.to_csv(RESULTS / "agent_C/C07_MODEL_RESTART_SUMMARY.csv", index=False, float_format="%.10f")
    print("Agent C complete", len(distance), len(parameter_rows), len(basin_rows))


if __name__ == "__main__":
    main()
