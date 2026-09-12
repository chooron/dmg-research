#!/usr/bin/env python3
"""Step 6: summarize full-test and temporal R1 estimands."""
from __future__ import annotations

import numpy as np
import pandas as pd

from r1_config import MODEL_REGISTRY, TABLES_DIR
from r1_utils import registry_sort, spearman, pearson, summary

POP = "STRUCTURAL_ENSEMBLE_36"


def main() -> None:
    pair_path = TABLES_DIR / "R1_model_basin_delta_kge.csv"
    temporal_path = TABLES_DIR / "R1_temporal_AB_model_basin.csv"
    pair = pd.read_csv(pair_path, dtype={"model": str, "basin_id": str})
    temporal = pd.read_csv(temporal_path, dtype={"model": str, "basin_id": str, "partition": str})
    if set(pair.model) != set(MODEL_REGISTRY) or len(pair) != 36 * 531:
        raise RuntimeError("full-test R1 table is not exactly 36x531")
    if len(temporal) != 36 * 531 * 2 or set(temporal.partition) != {"A", "B"}:
        raise RuntimeError("temporal R1 table is not exactly 36x531x2")

    model_rows: list[dict[str, object]] = []
    for model in MODEL_REGISTRY:
        g = pair.loc[pair.model == model]
        if len(g) != 531 or g.basin_id.nunique() != 531:
            raise RuntimeError(f"{model}: full-test basin count is not 531")
        delta = g["delta_KGE"].to_numpy(dtype=float)
        if not np.isfinite(g[["KGE_IC", "KGE_dPL", "delta_KGE"]].to_numpy(dtype=float)).all():
            raise RuntimeError(f"{model}: non-finite full-test KGE")
        model_rows.append({
            "model": model, "N": int(len(g)),
            "IC_median": float(g.KGE_IC.median()), "IC_mean": float(g.KGE_IC.mean()),
            "IC_Q25": float(g.KGE_IC.quantile(.25)), "IC_Q75": float(g.KGE_IC.quantile(.75)),
            "dPL_median": float(g.KGE_dPL.median()), "dPL_mean": float(g.KGE_dPL.mean()),
            "dPL_Q25": float(g.KGE_dPL.quantile(.25)), "dPL_Q75": float(g.KGE_dPL.quantile(.75)),
            "delta_median": float(np.median(delta)), "delta_mean": float(np.mean(delta)),
            "delta_Q10": float(np.quantile(delta, .10)), "delta_Q25": float(np.quantile(delta, .25)),
            "delta_Q75": float(np.quantile(delta, .75)), "delta_Q90": float(np.quantile(delta, .90)),
            "fraction_delta_gt_0": float(np.mean(delta > 0)),
            "fraction_delta_lt_0": float(np.mean(delta < 0)),
            "fraction_delta_eq_0": float(np.mean(delta == 0)),
        })
    model_summary = registry_sort(pd.DataFrame(model_rows))
    model_summary.to_csv(TABLES_DIR / "R1_model_performance_summary.csv", index=False, float_format="%.10f")

    temporal_rows: list[dict[str, object]] = []
    wide = []
    for model in MODEL_REGISTRY:
        g = temporal.loc[temporal.model == model]
        if len(g) != 2 * 531 or g.groupby("partition").basin_id.nunique().to_dict() != {"A": 531, "B": 531}:
            raise RuntimeError(f"{model}: temporal A/B basin coverage is not 531+531")
        row: dict[str, object] = {"model": model}
        for partition in ("A", "B"):
            h = g.loc[g.partition == partition]
            values = h.delta_KGE.to_numpy(dtype=float)
            if not np.isfinite(h[["KGE_IC", "KGE_dPL", "delta_KGE"]].to_numpy(dtype=float)).all():
                raise RuntimeError(f"{model}/{partition}: non-finite temporal KGE")
            s = summary(values)
            suffix = partition
            row.update({
                f"N_{suffix}": s["N"], f"delta_median_{suffix}": s["median"], f"delta_mean_{suffix}": s["mean"],
                f"delta_Q10_{suffix}": s["Q10"], f"delta_Q25_{suffix}": s["Q25"],
                f"delta_Q75_{suffix}": s["Q75"], f"delta_Q90_{suffix}": s["Q90"],
                f"fraction_delta_gt_0_{suffix}": float(np.mean(values > 0)),
                f"fraction_delta_lt_0_{suffix}": float(np.mean(values < 0)),
                f"fraction_delta_eq_0_{suffix}": float(np.mean(values == 0)),
            })
        a = float(row["delta_median_A"]); b = float(row["delta_median_B"])
        row["sign_A"] = "positive" if a > 0 else "negative" if a < 0 else "neutral"
        row["sign_B"] = "positive" if b > 0 else "negative" if b < 0 else "neutral"
        row["same_sign_non_neutral"] = bool(a != 0 and b != 0 and np.sign(a) == np.sign(b))
        temporal_rows.append(row)
        wide.append({"model": model, "delta_median_A": a, "delta_median_B": b})
    temporal_summary = registry_sort(pd.DataFrame(temporal_rows))
    temporal_summary.to_csv(TABLES_DIR / "R1_temporal_AB_model_summary.csv", index=False, float_format="%.10f")
    wide_df = pd.DataFrame(wide)
    neutral = (wide_df.delta_median_A == 0) | (wide_df.delta_median_B == 0)
    nonneutral = ~neutral
    same_sign_n = int(((np.sign(wide_df.loc[nonneutral, "delta_median_A"]) == np.sign(wide_df.loc[nonneutral, "delta_median_B"]))).sum())
    temporal_global = {
        "N_models": int(len(wide_df)), "rho_spearman": spearman(wide_df.delta_median_A, wide_df.delta_median_B),
        "rho_pearson": pearson(wide_df.delta_median_A, wide_df.delta_median_B),
        "same_sign_numerator_excluding_neutral": same_sign_n,
        "same_sign_denominator_excluding_neutral": int(nonneutral.sum()),
        "same_sign_fraction_excluding_neutral": float(same_sign_n / nonneutral.sum()) if nonneutral.any() else np.nan,
        "neutral_model_count": int(neutral.sum()),
        "same_sign_fraction_including_neutral": float(same_sign_n / len(wide_df)),
    }

    ensemble_rows: list[dict[str, object]] = []
    def add(scope: str, metric: str, value: object, n_models: int, n_rows: int, notes: str) -> None:
        ensemble_rows.append({"summary_scope": scope, "population": POP, "metric": metric,
                              "value": value, "N_models": n_models, "N_model_basin_rows": n_rows, "notes": notes})

    for metric in ("IC_median", "dPL_median", "delta_median", "IC_mean", "dPL_mean", "delta_mean"):
        add("model_level", f"{metric}_across_model_values_median", float(model_summary[metric].median()), 36, 36 * 531,
            "equal-weight median across 36 model-level basin summaries")
        add("model_level", f"{metric}_across_model_values_mean", float(model_summary[metric].mean()), 36, 36 * 531,
            "equal-weight mean across 36 model-level basin summaries")
    for metric in ("delta_median", "delta_mean"):
        for q in ("Q10", "Q25", "Q75", "Q90"):
            add("model_level", f"{metric}_across_model_values_{q}", float(model_summary[metric].quantile(float(q[1:]) / 100)), 36, 36 * 531,
                "distribution of model-level estimates; models are the observation units")
    for metric in ("delta_KGE", "KGE_IC", "KGE_dPL"):
        values = pair[metric].to_numpy(dtype=float)
        s = summary(values)
        for label in ("median", "mean", "Q10", "Q25", "Q75", "Q90"):
            add("pooled_model_basin", f"{metric}_{label}", s[label], 36, len(pair),
                "descriptive pooled rows; not independent inferential samples")
    add("pooled_model_basin", "delta_fraction_gt_0", float(np.mean(pair.delta_KGE > 0)), 36, len(pair), "descriptive pooled rows")
    add("pooled_model_basin", "delta_fraction_lt_0", float(np.mean(pair.delta_KGE < 0)), 36, len(pair), "descriptive pooled rows")
    add("temporal_model_level", "rho_spearman", temporal_global["rho_spearman"], 36, 36,
        "Spearman correlation of model median delta_KGE A vs B; models are observation units")
    add("temporal_model_level", "rho_pearson_secondary", temporal_global["rho_pearson"], 36, 36,
        "secondary Pearson correlation of model median delta_KGE A vs B")
    add("temporal_model_level", "same_sign_fraction_excluding_neutral", temporal_global["same_sign_fraction_excluding_neutral"], 36, 36,
        "exact zero is neutral; neutral models excluded from denominator")
    add("temporal_model_level", "same_sign_fraction_including_neutral", temporal_global["same_sign_fraction_including_neutral"], 36, 36,
        "exact zero is neutral and not counted as same-sign; all models in denominator")
    add("temporal_model_level", "same_sign_numerator_excluding_neutral", temporal_global["same_sign_numerator_excluding_neutral"], 36, 36, "integer count")
    add("temporal_model_level", "same_sign_denominator_excluding_neutral", temporal_global["same_sign_denominator_excluding_neutral"], 36, 36, "integer count")
    add("temporal_model_level", "neutral_model_count", temporal_global["neutral_model_count"], 36, 36, "exact-zero neutral count")
    add("temporal_model_level", "delta_A_median_across_models", float(temporal_summary.delta_median_A.median()), 36, 36 * 531, "equal-weight median of model-level A summaries")
    add("temporal_model_level", "delta_B_median_across_models", float(temporal_summary.delta_median_B.median()), 36, 36 * 531, "equal-weight median of model-level B summaries")
    ensemble = pd.DataFrame(ensemble_rows)
    ensemble.to_csv(TABLES_DIR / "R1_ensemble_summary.csv", index=False, float_format="%.10f")

    print(f"PASS: wrote {TABLES_DIR / 'R1_model_performance_summary.csv'}")
    print(f"PASS: wrote {TABLES_DIR / 'R1_temporal_AB_model_summary.csv'}")
    print(f"PASS: wrote {TABLES_DIR / 'R1_ensemble_summary.csv'}")
    print(f"RESULT: IC_model_median={model_summary.IC_median.median():.10f}; dPL_model_median={model_summary.dPL_median.median():.10f}; delta_model_median={model_summary.delta_median.median():.10f}; temporal_rho={temporal_global['rho_spearman']:.10f}; same_sign={temporal_global['same_sign_fraction_excluding_neutral']:.10f}")


if __name__ == "__main__":
    main()
