"""Module 1: bound-normalized IC-to-dPL separation versus archived IC-self reference."""
from pathlib import Path
import pandas as pd
from r2_common import CACHE, ensure_inputs, finalize_group, read_csv, write_csv, write_json, require_close, SOURCE


def main() -> None:
    ensure_inputs()
    models = read_csv("separation_models")
    boot = read_csv("separation_bootstrap")
    primary = models[(models.row_type == "ALL_MODELS") & (models.threshold == "within_0.01")].iloc[0]
    b = boot[boot.threshold == "within_0.01"].iloc[0]
    model_rows = models[(models.row_type == "model") & (models.threshold == "within_0.01")].copy()
    positive = int((model_rows.cross_minus_self_median > 0).sum())
    require_close(float(primary.D_cross_model_equal_median), 0.3843747256, 1e-8, "D_RMS")
    require_close(float(primary.cross_minus_self_model_equal_median), 0.2187745308, 1e-8, "cross-minus-self")
    require_close(float(b.difference_ci_low), 0.2091360196, 1e-8, "difference CI low")
    require_close(float(b.difference_ci_high), 0.2281453316, 1e-8, "difference CI high")
    write_csv(CACHE / "separation/model_summary.csv", model_rows[["model", "D_cross_median", "D_self_median_of_basin_medians", "cross_minus_self_median", "n_insufficient"]])
    summary = pd.DataFrame([
        {"metric": "D_RMS_model_equal_median", "value": float(primary.D_cross_model_equal_median), "n_models": int(primary.n_models)},
        {"metric": "D_self_model_equal_median", "value": float(primary.D_self_model_equal_median), "n_models": int(primary.n_models)},
        {"metric": "cross_minus_self", "value": float(primary.cross_minus_self_model_equal_median), "n_models": int(primary.n_models)},
        {"metric": "cross_minus_self_ci_low", "value": float(b.difference_ci_low), "n_models": int(b.n_boot)},
        {"metric": "cross_minus_self_ci_high", "value": float(b.difference_ci_high), "n_models": int(b.n_boot)},
        {"metric": "positive_model_count", "value": positive, "n_models": len(model_rows)},
    ])
    write_csv(CACHE / "separation/summary.csv", summary)
    write_json(CACHE / "separation/summary.json", {"verdict": "SEPARATION_EXCEEDS_ARCHIVED_IC_SELF_REFERENCE", "one_sided": True, "dpl_seed": 42, "source": [str(SOURCE["separation_models"]), str(SOURCE["separation_bootstrap"])]})
    finalize_group("separation", ["separation_models", "separation_bootstrap", "displacement_model"], [])


if __name__ == "__main__":
    main()
