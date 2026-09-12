"""Module 3: raw coordinate localization and coordinate-specific IC-self adjustment."""
import pandas as pd
from r2_common import CACHE, ensure_inputs, finalize_group, read_csv, write_csv, write_json, require_close


def metric(frame, name):
    row = frame[frame.metric == name].iloc[0]
    return float(row.value if "value" in row else row.model_equal_median)


def main() -> None:
    ensure_inputs()
    raw_models = read_csv("raw_localization_models")
    raw_summary = read_csv("raw_localization_summary")
    raw = raw_summary[raw_summary.estimand.eq("model_equal")]
    raw_vals = {m: float(raw[(raw["metric"] == m) & (raw["statistic"] == "median")]["value"].iloc[0]) for m in ["C_eff_median", "top1_share_median", "top2_share_median"]}
    raw_models["uniform_baseline"] = 1.0 / raw_models.parameter_count
    write_csv(CACHE / "localization/raw_model_summary.csv", raw_models)
    excess = read_csv("coord_excess")
    adjusted = read_csv("adjusted_localization")
    adjusted_raw = read_csv("adjusted_raw")
    e = {row["metric"]: float(row["model_equal_median"]) for _, row in excess.iterrows()}
    a = {row["metric"]: float(row["median"]) for _, row in adjusted.iterrows()}
    require_close(raw_vals["C_eff_median"], 0.3458437727, 1e-8, "raw C_eff")
    require_close(raw_vals["top1_share_median"], 0.54561860045, 1e-8, "raw top1")
    require_close(raw_vals["top2_share_median"], 0.84824760785, 1e-8, "raw top2")
    require_close(e["median_M_cross"], 0.1340612825, 1e-8, "M_cross")
    require_close(e["median_M_self"], 0.000164794113, 1e-8, "M_self")
    require_close(e["median_E_coord"], 0.07319252485, 1e-8, "E_coord")
    require_close(a["fraction_any_positive_excess"], 0.9604519774, 1e-8, "positive excess fraction")
    require_close(a["C_eff_excess_median"], 0.2933820389, 1e-8, "adjusted C_eff")
    require_close(a["top1_excess_share_median"], 0.672633535, 1e-8, "adjusted top1")
    require_close(a["top2_excess_share_median"], 0.9417612386, 1e-8, "adjusted top2")
    l1 = read_csv("adjusted_l1")
    l1_rows = l1[["metric", "median", "n_models"]].rename(columns={"median": "value"})
    require_close(float(l1.loc[l1.metric == "C_eff_excess_L1_median", "median"].iloc[0]), 0.435695457, 1e-8, "L1 C_eff")
    require_close(float(l1.loc[l1.metric == "top1_excess_share_L1_median", "median"].iloc[0]), 0.5099751912, 1e-8, "L1 top1")
    require_close(float(l1.loc[l1.metric == "top2_excess_share_L1_median", "median"].iloc[0]), 0.7818176063, 1e-8, "L1 top2")
    write_csv(CACHE / "localization/coordinate_excess_model_equal.csv", excess)
    write_csv(CACHE / "localization/adjusted_basin_model_equal.csv", adjusted)
    write_csv(CACHE / "localization/raw_vs_adjusted_model_equal.csv", adjusted_raw)
    write_csv(CACHE / "localization/L1_sensitivity.csv", l1_rows)
    summary = pd.DataFrame([
        {"metric": "raw_C_eff", "value": raw_vals["C_eff_median"], "scope": "all36"},
        {"metric": "raw_top1", "value": raw_vals["top1_share_median"], "scope": "all36"},
        {"metric": "raw_top2", "value": raw_vals["top2_share_median"], "scope": "all36"},
        {"metric": "M_cross", "value": e["median_M_cross"], "scope": "strict23"},
        {"metric": "M_self", "value": e["median_M_self"], "scope": "strict23"},
        {"metric": "E_coord", "value": e["median_E_coord"], "scope": "strict23"},
        {"metric": "positive_excess_basin_fraction", "value": a["fraction_any_positive_excess"], "scope": "strict23"},
        {"metric": "adjusted_C_eff", "value": a["C_eff_excess_median"], "scope": "strict23"},
        {"metric": "adjusted_top1", "value": a["top1_excess_share_median"], "scope": "strict23"},
        {"metric": "adjusted_top2", "value": a["top2_excess_share_median"], "scope": "strict23"},
    ])
    write_csv(CACHE / "localization/summary.csv", summary)
    write_json(CACHE / "localization/summary.json", {"verdict": "COORDINATE_LOCALIZATION_PERSISTS_BEYOND_IC_SELF_VARIABILITY", "wording": "displacement shows some coordinate concentration", "M_self_caveat": "coordinate/basin median archived IC-self reference; not vector RMS D_self"})
    finalize_group("localization", ["raw_localization_models", "raw_localization_summary", "coord_excess", "coord_overlap", "adjusted_localization", "adjusted_raw", "adjusted_l1", "adjusted_all36"], [])


if __name__ == "__main__":
    main()
