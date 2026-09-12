#!/usr/bin/env python3
"""Step 7: lightweight R1 sensitivities (simhyd exclusion; mean vs median)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from r1_config import TABLES_DIR
from r1_utils import pearson, spearman, summary
from src.model_registry import NPARAM_INFO_36


def main() -> None:
    pair = pd.read_csv(TABLES_DIR / "R1_model_basin_delta_kge.csv", dtype={"model": str, "basin_id": str})
    model = pd.read_csv(TABLES_DIR / "R1_model_performance_summary.csv")
    temporal = pd.read_csv(TABLES_DIR / "R1_temporal_AB_model_summary.csv")
    rows: list[dict[str, object]] = []

    def static_metric(df: pd.DataFrame, metric: str) -> float:
        return float(df[metric].median())

    full_model = model
    no_simhyd = model.loc[model.model != "simhyd"].copy()
    for metric, label in [
        ("IC_median", "Fig1a_IC_median_of_model_medians"),
        ("dPL_median", "Fig1a_dPL_median_of_model_medians"),
        ("delta_median", "Fig1a_delta_median_of_model_medians"),
    ]:
        full = static_metric(full_model, metric)
        reduced = static_metric(no_simhyd, metric)
        rows.append({"analysis": "Fig1a_aggregate_summary", "metric": label, "full_36": full,
                     "exclude_simhyd_35": reduced, "exclude_minus_full": reduced - full,
                     "N_full": 36, "N_exclude": 35, "notes": "equal-weight median of model-level basin medians"})
    for q in ("Q10", "Q25", "Q75", "Q90"):
        full = float(full_model.delta_median.quantile(float(q[1:]) / 100))
        reduced = float(no_simhyd.delta_median.quantile(float(q[1:]) / 100))
        rows.append({"analysis": "Fig1b_model_level_delta_distribution", "metric": f"delta_median_{q}",
                     "full_36": full, "exclude_simhyd_35": reduced, "exclude_minus_full": reduced - full,
                     "N_full": 36, "N_exclude": 35, "notes": "quantile of fixed-order model-level delta medians"})

    def temporal_metrics(df: pd.DataFrame) -> dict[str, float]:
        a, b = df.delta_median_A.to_numpy(float), df.delta_median_B.to_numpy(float)
        neutral = (a == 0) | (b == 0)
        nonneutral = ~neutral
        same = (np.sign(a[nonneutral]) == np.sign(b[nonneutral])).sum()
        return {
            "rho_spearman": spearman(a, b), "rho_pearson": pearson(a, b),
            "same_sign_fraction_excluding_neutral": float(same / nonneutral.sum()) if nonneutral.any() else np.nan,
            "same_sign_fraction_including_neutral": float(same / len(df)),
            "neutral_model_count": float(neutral.sum()),
        }

    t_full = temporal_metrics(temporal)
    t_reduced = temporal_metrics(temporal.loc[temporal.model != "simhyd"])
    for metric in t_full:
        rows.append({"analysis": "Fig1c_temporal_AB", "metric": metric, "full_36": t_full[metric],
                     "exclude_simhyd_35": t_reduced[metric], "exclude_minus_full": t_reduced[metric] - t_full[metric],
                     "N_full": 36, "N_exclude": 35, "notes": "model-level temporal summary; exact zero neutral"})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "R1_sensitivity_exclude_simhyd.csv", index=False, float_format="%.10f")

    # Compare aggregation choices without replacing the primary median result.
    mean_rows: list[dict[str, object]] = []
    for model_name, g in pair.groupby("model", sort=False):
        median_delta, mean_delta = float(g.delta_KGE.median()), float(g.delta_KGE.mean())
        mean_rows.append({
            "model": model_name, "N": int(len(g)),
            "IC_basin_median": float(g.KGE_IC.median()), "IC_basin_mean": float(g.KGE_IC.mean()),
            "dPL_basin_median": float(g.KGE_dPL.median()), "dPL_basin_mean": float(g.KGE_dPL.mean()),
            "delta_basin_median": median_delta, "delta_basin_mean": mean_delta,
            "delta_mean_minus_median": mean_delta - median_delta,
            "aggregation_comparison": "basin median versus basin mean", "notes": "model is the observation unit for ensemble aggregation",
        })
    per_model = pd.DataFrame(mean_rows)
    mean_rows.append({
        "model": "__ENSEMBLE__", "N": 36,
        "IC_basin_median": float(per_model.IC_basin_median.median()), "IC_basin_mean": float(per_model.IC_basin_mean.median()),
        "dPL_basin_median": float(per_model.dPL_basin_median.median()), "dPL_basin_mean": float(per_model.dPL_basin_mean.median()),
        "delta_basin_median": float(per_model.delta_basin_median.median()), "delta_basin_mean": float(per_model.delta_basin_mean.median()),
        "delta_mean_minus_median": float(per_model.delta_basin_mean.median() - per_model.delta_basin_median.median()),
        "aggregation_comparison": "ensemble median across model-level summaries", "notes": "headline aggregation sensitivity; no pooled-row inference",
    })
    pd.DataFrame(mean_rows).to_csv(TABLES_DIR / "R1_sensitivity_mean_vs_median.csv", index=False, float_format="%.10f")

    # Existing, low-complexity confound checks are reported descriptively only.
    confound = model.copy()
    confound["parameter_count"] = confound.model.map(NPARAM_INFO_36)
    confound_rows = []
    for covariate, values in [("parameter_count", confound.parameter_count), ("baseline_IC_median", confound.IC_median)]:
        confound_rows.append({
            "covariate": covariate, "outcome": "model_level_delta_median", "N_models": 36,
            "pearson": pearson(values, confound.delta_median), "spearman": spearman(values, confound.delta_median),
            "notes": "descriptive robustness check; no causal or model-selection interpretation",
        })
    pd.DataFrame(confound_rows).to_csv(TABLES_DIR / "R1_sensitivity_confound_checks.csv", index=False, float_format="%.10f")
    print(f"PASS: wrote {TABLES_DIR / 'R1_sensitivity_exclude_simhyd.csv'}")
    print(f"PASS: wrote {TABLES_DIR / 'R1_sensitivity_mean_vs_median.csv'}")
    print(f"PASS: wrote {TABLES_DIR / 'R1_sensitivity_confound_checks.csv'}")


if __name__ == "__main__":
    main()
