"""Module 6: supporting hostile audit linking R2 rank continuity to R3 information specificity."""
import pandas as pd
from r2_common import CACHE, ensure_inputs, finalize_group, read_csv, write_csv, write_json, require_close


def main() -> None:
    ensure_inputs()
    corr = read_csv("linkage_corr")
    unc = read_csv("linkage_uncertainty").iloc[0]
    wanted = corr[corr.unit.isin(["pooled_coordinates", "model_centered_coordinate", "within_model_coordinate_median", "pooled_pairs_with_model_labels", "model_centered_pair", "within_model_pair_median"])].copy()
    require_close(float(wanted.loc[wanted.unit == "pooled_coordinates", "spearman"].iloc[0]), 0.743829615657, 1e-8, "pooled R linkage")
    require_close(float(wanted.loc[wanted.unit == "model_centered_coordinate", "spearman"].iloc[0]), 0.683141802071, 1e-8, "centered R linkage")
    require_close(float(wanted.loc[wanted.unit == "within_model_coordinate_median", "spearman"].iloc[0]), 0.678571428571, 1e-8, "within-model R linkage")
    require_close(float(unc.observed_model_equal_median), 0.295238095238, 1e-8, "A_info|rank")
    require_close(float(unc.bootstrap_ci_low), 0.254385964912, 1e-8, "A_info|rank CI low")
    require_close(float(unc.bootstrap_ci_high), 0.429072681704, 1e-8, "A_info|rank CI high")
    require_close(float(unc.sign_flip_null_p_ge_observed), 0.000399920016, 1e-10, "sign flip p")
    write_csv(CACHE / "r2_r3_linkage/correlation_summaries.csv", wanted)
    write_csv(CACHE / "r2_r3_linkage/uncertainty_summary.csv", pd.DataFrame([unc]))
    write_json(CACHE / "r2_r3_linkage/summary.json", {"role": "supporting_audit_only", "interpretation": "same-coordinate information specificity remains beyond raw rank continuity", "no_causal_or_physical_claim": True})
    finalize_group("r2_r3_linkage", ["linkage_corr", "linkage_uncertainty"], [])


if __name__ == "__main__":
    main()
