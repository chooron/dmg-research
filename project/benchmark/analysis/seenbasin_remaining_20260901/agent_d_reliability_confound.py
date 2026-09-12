#!/usr/bin/env python3
"""Agent D: reproducibility bridges, admissible reliability, and confounds."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ALL_MODELS, RESULTS, SEED, bootstrap_corr, corr, load_inputs, write_csv  # noqa: E402


def main() -> None:
    _, paired, params, distance, _, _, _ = load_inputs()
    paired["G_seen"] = paired.KGE_IC - paired.KGE_dPL
    distance["D_theta"] = distance.normalized_l2_distance / np.sqrt(distance.parameter_count)
    atlas = pd.read_csv(Path(RESULTS).parent / "ic_dpl_seenbasin_formal_20260901/10_PARAMETER_ATTRIBUTE_REPRODUCIBILITY.csv")
    atlas["model"] = atlas.model.astype(str)
    model_r = atlas.groupby("model", sort=True).agg(
        reproducibility_median=("profile_spearman", "median"), reproducibility_mean=("profile_spearman", "mean"),
        reproducibility_q25=("profile_spearman", lambda x: x.quantile(.25)), reproducibility_q75=("profile_spearman", lambda x: x.quantile(.75)),
        dominant_control_agreement_model=("dominant_control_agreement", "mean"), sign_agreement_model=("sign_agreement_valid", "mean"),
        parameter_count=("parameter_index", "nunique"),
    ).reset_index()
    gap = paired.groupby("model", sort=True).agg(G_seen_median=("G_seen", "median"), G_seen_mean=("G_seen", "mean"), G_seen_q25=("G_seen", lambda x: x.quantile(.25)), G_seen_q75=("G_seen", lambda x: x.quantile(.75))).reset_index()
    dist = distance.groupby("model", sort=True).agg(D_theta_median=("D_theta", "median"), D_theta_mean=("D_theta", "mean")).reset_index()
    bridge = model_r.merge(gap, on="model").merge(dist, on="model")
    rho, p, n = corr(bridge.G_seen_median, bridge.reproducibility_median)
    ci_low, ci_high = bootstrap_corr(bridge.G_seen_median.to_numpy(), bridge.reproducibility_median.to_numpy(), 5000, SEED)
    rows = [{"analysis_level": "model_level", "rho_G_seen_reproducibility": rho, "p_value": p, "n": n, "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high, "bootstrap_seed": SEED, "bootstrap_replicates": 5000, "label": "REPRODUCIBILITY_VS_MODEL_G_SEEN"}]
    for excluded in ALL_MODELS:
        x = bridge[bridge.model != excluded]
        r, pv, nn = corr(x.G_seen_median, x.reproducibility_median)
        rows.append({"analysis_level": "leave_one_model_out", "excluded_model": excluded, "rho_G_seen_reproducibility": r, "p_value": pv, "n": nn, "bootstrap_ci95_low": np.nan, "bootstrap_ci95_high": np.nan, "bootstrap_seed": SEED, "bootstrap_replicates": 0, "label": "REPRODUCIBILITY_VS_MODEL_G_SEEN"})
    write_csv(RESULTS / "agent_D/D01_REPRODUCIBILITY_VS_MODEL_GAP.csv", pd.DataFrame(rows))

    rows = []
    for xname, yname, label in [("G_seen_median", "D_theta_median", "G_SEEN_VS_PARAMETER_DISTANCE"), ("reproducibility_median", "D_theta_median", "REPRODUCIBILITY_VS_PARAMETER_DISTANCE")]:
        r, pv, nn = corr(bridge[xname], bridge[yname])
        rows.append({"x": xname, "y": yname, "rho": r, "p_value": pv, "n": nn, "label": label})
    write_csv(RESULTS / "agent_D/D02_REPRODUCIBILITY_VS_PARAMETER_DISTANCE.csv", pd.DataFrame([rows[1]]))

    pdelta = params.pivot_table(index=["model", "basin_id", "parameter_index", "parameter"], columns="method", values="normalized_u", aggfunc="first").reset_index()
    pdelta["abs_u_difference"] = (pdelta.IC - pdelta.dPL).abs()
    pprofile = atlas.groupby(["model", "parameter_index", "parameter"], sort=True).agg(relationship_vector_similarity=("profile_spearman", "first"), dominant_control_agreement=("dominant_control_agreement", "first"), sign_agreement=("sign_agreement_valid", "first")).reset_index()
    pdist = pdelta.groupby(["model", "parameter_index", "parameter"], sort=True).agg(parameter_distance_median_abs_u=("abs_u_difference", "median"), parameter_distance_mean_abs_u=("abs_u_difference", "mean")).reset_index()
    d03 = pprofile.merge(pdist, on=["model", "parameter_index", "parameter"])
    r, pv, nn = corr(d03.relationship_vector_similarity, d03.parameter_distance_median_abs_u)
    d03["pooled_rho_relationship_similarity_parameter_distance"] = r
    d03["pooled_p_value"] = pv
    d03["pooled_n"] = nn
    d03["label"] = "PARAMETER_LEVEL_RELIABILITY_VS_DISTANCE"
    write_csv(RESULTS / "agent_D/D03_PARAMETER_LEVEL_RELIABILITY_DISTANCE.csv", d03)

    admissible = pd.read_csv(RESULTS / "agent_A/A04_PREDICTIVELY_ADMISSIBLE_BASIN_LEVEL.csv", dtype={"basin_id": str})
    model_lookup = bridge.set_index("model")
    drows = []
    for tau, group in admissible.groupby("tau", sort=True):
        for _, row in group.iterrows():
            names = row.admissible_models.split(";") if row.admissible_models else []
            # Low/high is relative to the admissible-set G_seen median; ties are in low and high is strictly above.
            for gclass in ("admissible_low_G_seen", "admissible_high_G_seen"):
                # Reliability groups are defined by G_seen, not by R; use the basin-specific values.
                vals = paired[(paired.basin_id == row.basin_id) & (paired.model.isin(names))][["model", "G_seen"]].merge(model_lookup.reset_index(), on="model")
                midpoint = vals.G_seen.median()
                selected = vals[vals.G_seen <= midpoint] if gclass.endswith("low_G_seen") else vals[vals.G_seen > midpoint]
                if len(selected):
                    drows.append({"tau": tau, "basin_id": row.basin_id, "group": gclass, "model_count": len(selected), "G_seen_median": float(selected.G_seen.median()), "reproducibility_median": float(selected.reproducibility_median.median()), "dominant_control_agreement": float(selected.dominant_control_agreement_model.mean()), "D_theta_median": float(selected.D_theta_median.median()), "label": "ADMISSIBLE_G_SEEN_RELIABILITY"})
    d04 = pd.DataFrame(drows)
    if len(d04):
        d04 = d04.groupby(["tau", "group"], as_index=False).agg(basin_count=("basin_id", "nunique"), total_model_basin_rows=("model_count", "sum"), G_seen_median=("G_seen_median", "median"), reproducibility_median=("reproducibility_median", "median"), dominant_control_agreement=("dominant_control_agreement", "median"), D_theta_median=("D_theta_median", "median"))
        d04["label"] = "ADMISSIBLE_G_SEEN_RELIABILITY"
    write_csv(RESULTS / "agent_D/D04_ADMISSIBLE_MODELS_PARAMETER_RELIABILITY.csv", d04)

    ic_summary = pd.read_csv(Path(RESULTS).parent / "ic_dpl_seenbasin_formal_20260901/03_MODEL_PERFORMANCE_SUMMARY.csv")
    ic_summary = ic_summary[(ic_summary.row_type == "MODEL") & (ic_summary.population == "STRUCTURAL_ENSEMBLE_36")][["model", "ic_median", "dpl_median"]]
    conf = bridge.merge(ic_summary, on="model")
    pairs = [("parameter_count", "G_seen_median"), ("parameter_count", "D_theta_median"), ("parameter_count", "reproducibility_median"), ("ic_median", "G_seen_median"), ("ic_median", "reproducibility_median")]
    rows = []
    for xname, yname in pairs:
        r, pv, nn = corr(conf[xname], conf[yname])
        rows.append({"x": xname, "y": yname, "rho": r, "p_value": pv, "n": nn, "label": "MODEL_COMPLEXITY_BASELINE_CONFOUND"})
    write_csv(RESULTS / "agent_D/D05_MODEL_COMPLEXITY_CONFOUND.csv", pd.DataFrame(rows))

    restart = pd.read_csv(RESULTS / "agent_C/C07_MODEL_RESTART_SUMMARY.csv")
    d06 = restart.merge(bridge[["model", "reproducibility_median", "dominant_control_agreement_model"]], on="model")
    rows = []
    for xname in ["restart_mean_u_sd", "restart_mean_u_iqr", "restart_max_u_sd", "best_second_u_distance", "fitness_spread"]:
        for yname in ["D_theta_median", "reproducibility_median", "dominant_control_agreement_model"]:
            if yname == "D_theta_median":
                yy = bridge.set_index("model").D_theta_median.reindex(d06.model).to_numpy()
            else:
                yy = d06[yname].to_numpy()
            xx = d06[xname].to_numpy()
            r, pv, nn = corr(xx, yy)
            rows.append({"x": xname, "y": yname, "rho": r, "p_value": pv, "n": nn, "label": "IDENTIFIABILITY_VS_RELATIONSHIP_RELIABILITY"})
    write_csv(RESULTS / "agent_D/D06_IDENTIFIABILITY_VS_RELATIONSHIP_RELIABILITY.csv", pd.DataFrame(rows))
    write_csv(RESULTS / "agent_D/D00_MODEL_BRIDGE_TABLE.csv", bridge)
    print("Agent D complete", len(bridge), len(d03), len(d04), len(rows))


if __name__ == "__main__":
    main()
