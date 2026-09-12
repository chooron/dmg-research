"""Module 5: short within-model outlet-performance/parameter bridge."""
import pandas as pd
from r2_common import CACHE, ensure_inputs, finalize_group, read_csv, write_csv, write_json, require_close


def main() -> None:
    ensure_inputs()
    summary = read_csv("performance")
    models = read_csv("performance_models")
    row = summary[summary["conditioning"] == "all_basins"].iloc[0]
    d = models[pd.to_numeric(models.rho_absDeltaKGE_D_theta, errors="coerce").notna()].copy()
    rho = pd.to_numeric(d.rho_absDeltaKGE_D_theta, errors="coerce")
    positive = int((rho > 0).sum())
    require_close(float(row["model_equal_median_rho"]), 0.2411895986, 1e-8, "bridge rho")
    require_close(float(row["bootstrap_ci_low"]), 0.1973121964, 1e-8, "bridge CI low")
    require_close(float(row["bootstrap_ci_high"]), 0.2586388471, 1e-8, "bridge CI high")
    if positive != 34:
        raise AssertionError(f"bridge positive model count {positive} != 34")
    write_csv(CACHE / "performance_bridge/model_association_all.csv", d)
    out = pd.DataFrame([{"metric": "within_model_rho_absDeltaKGE_D_theta", "value": float(row["model_equal_median_rho"]), "ci_low": float(row["bootstrap_ci_low"]), "ci_high": float(row["bootstrap_ci_high"]), "positive_count": positive, "n_models": 36}])
    write_csv(CACHE / "performance_bridge/summary.csv", out)
    write_json(CACHE / "performance_bridge/summary.json", {"role": "supporting_bridge_only", "interpretation": "moderate observational within-model association; not one-to-one or causal", "dpl_seed": 42})
    finalize_group("performance_bridge", ["performance", "performance_models"], [])


if __name__ == "__main__":
    main()
