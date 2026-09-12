"""Module 2: cross-catchment rank reorganization and strict IC-self subset audit."""
import pandas as pd
from r2_common import CACHE, ELIGIBLE_MODELS, INSUFFICIENT_MODELS, ensure_inputs, finalize_group, read_csv, write_csv, write_json, require_close


def main() -> None:
    ensure_inputs()
    all36 = read_csv("rank_all36")
    rank = all36[all36.summary == "model_equal_primary"].iloc[0]
    selftab = read_csv("rank_self")
    strict = selftab[(selftab.pool == "primary") & (selftab.prefix == 5000)].iloc[0]
    coverage = read_csv("rank_coverage")
    coverage["primary_status"] = coverage["primary_coverage"].ge(0.90).map({True: "PASS", False: "INSUFFICIENT_REFERENCE"})
    require_close(float(rank["median"]), 0.406901464622, 1e-8, "all36 R_rank")
    require_close(float(strict.R_cross_median), 0.487560126579, 1e-8, "strict R_cross")
    require_close(float(strict.self_median_median), 0.797361671925, 1e-8, "strict R_self")
    require_close(float(strict.DeltaR_self_minus_cross_median), 0.211845558402, 1e-8, "strict DeltaR")
    n_positive = int(strict.n_DeltaR_positive_models)
    n_below_q05 = int(round(float(strict.fraction_cross_below_self_q05_models) * int(strict.n_models_valid)))
    if n_positive != 23 or n_below_q05 != 22 or int(strict.n_models_valid) != 23:
        raise AssertionError("strict IC-self rank denominator/count gate failed")
    write_csv(CACHE / "rank/coverage.csv", coverage)
    write_csv(CACHE / "rank/model_summaries_all36.csv", read_csv("rank_models"))
    summary = pd.DataFrame([
        {"scope": "all36_descriptive", "metric": "R_rank_median", "value": float(rank["median"]), "n_models": 36, "verdict": "PARTIAL_RANK_PRESERVATION_WITH_REORGANIZATION"},
        {"scope": "strict23", "metric": "R_cross_median", "value": float(strict.R_cross_median), "n_models": 23, "verdict": "INCONCLUSIVE"},
        {"scope": "strict23", "metric": "R_self_median", "value": float(strict.self_median_median), "n_models": 23, "verdict": "INCONCLUSIVE"},
        {"scope": "strict23", "metric": "DeltaR_self_minus_cross", "value": float(strict.DeltaR_self_minus_cross_median), "n_models": 23, "verdict": "INCONCLUSIVE"},
        {"scope": "strict23", "metric": "DeltaR_positive_count", "value": n_positive, "n_models": 23, "verdict": "INCONCLUSIVE"},
        {"scope": "strict23", "metric": "R_cross_below_self_q05_count", "value": n_below_q05, "n_models": 23, "verdict": "INCONCLUSIVE"},
    ])
    write_csv(CACHE / "rank/summary.csv", summary)
    write_json(CACHE / "rank/summary.json", {"strict_verdict": "INCONCLUSIVE", "eligible_models": ELIGIBLE_MODELS, "insufficient_models": INSUFFICIENT_MODELS, "strict_denominator": 23, "all36_is_descriptive_only": True})
    finalize_group("rank", ["rank_all36", "rank_models", "rank_self", "rank_coverage"], [])


if __name__ == "__main__":
    main()
