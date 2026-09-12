"""Module 4: reference-dependent distribution geometry."""
import pandas as pd
from r2_common import CACHE, ensure_inputs, finalize_group, read_csv, write_csv, write_json, require_close


def main() -> None:
    ensure_inputs()
    table = read_csv("distribution")
    matched = read_csv("distribution_matched")
    rows = table[table["scope"] == "PRIMARY_MATCHED_23"].copy()
    expected = {
        "canonical_median_CR": 0.6140173622,
        "consensus_median_CR": 0.9794559561,
        "selfref_median_CR": 1.0043244360,
        "canonical_median_DeltaIQR": -0.08893983765,
        "consensus_median_DeltaIQR": -0.009411529494,
        "selfref_median_DeltaIQR": 0.0009412229992,
    }
    for metric, value in expected.items():
        observed = float(rows.loc[rows["metric"] == metric, "median"].iloc[0])
        require_close(observed, value, 1e-8, metric)
    dcons = rows[rows["metric"] == "Delta_CR_consensus"].iloc[0]
    dself = rows[rows["metric"] == "Delta_CR_selfref"].iloc[0]
    require_close(float(dcons["median"]), 0.3290120169, 1e-8, "consensus delta")
    require_close(float(dself["median"]), 0.3799390248, 1e-8, "self delta")
    write_csv(CACHE / "distribution/model_equal_summary_primary23.csv", rows)
    write_csv(CACHE / "distribution/matched_model_comparison_23.csv", matched)
    summary = pd.DataFrame([
        {"metric": "canonical_CR", "value": expected["canonical_median_CR"], "n_models": 23},
        {"metric": "consensus_CR", "value": expected["consensus_median_CR"], "n_models": 23},
        {"metric": "self_reference_CR", "value": expected["selfref_median_CR"], "n_models": 23},
        {"metric": "consensus_minus_canonical", "value": float(dcons["median"]), "ci_low": float(dcons["ci_low"]), "ci_high": float(dcons["ci_high"]), "positive_count": int(dcons["positive_model_count"]), "n_models": 23},
        {"metric": "self_reference_minus_canonical", "value": float(dself["median"]), "ci_low": float(dself["ci_low"]), "ci_high": float(dself["ci_high"]), "positive_count": int(dself["positive_model_count"]), "n_models": 23},
    ])
    count_metrics = ["canonical_models_median_CR_below_1", "canonical_models_contraction_fraction_above_0.5", "consensus_models_median_CR_below_1", "consensus_models_contraction_fraction_above_0.5", "selfref_models_median_CR_below_1", "selfref_models_contraction_fraction_above_0.5"]
    count_rows = pd.DataFrame([{ "metric": m, "value": int(rows.loc[rows["metric"] == m, "value"].iloc[0]), "denominator": int(rows.loc[rows["metric"] == m, "denominator"].iloc[0]), "n_models": 23 } for m in count_metrics])
    summary = pd.concat([summary, count_rows], ignore_index=True, sort=False)
    write_csv(CACHE / "distribution/summary.csv", summary)
    write_json(CACHE / "distribution/summary.json", {"verdict": "CANONICAL_IC_DEPENDENT", "primary_models": 23, "thresholds": {"contraction": "CR < 0.95", "near_one": "0.95 <= CR <= 1.05", "expansion": "CR > 1.05"}})
    finalize_group("distribution", ["distribution", "distribution_matched"], [])


if __name__ == "__main__":
    main()
