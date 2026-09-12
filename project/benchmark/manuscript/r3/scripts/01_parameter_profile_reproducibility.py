#!/usr/bin/env python3
"""Freeze and compute the R3 cross-estimator parameter-profile estimand.

Primary: for each model and common parameter, Spearman correlation between its IC
and dPL information-cluster relationship profiles, then median over parameters
within model and median over models (model-equal). Secondary: the same estimator
using raw attributes, plus model-flattened and cell-equal summaries. Historical
0.658/0.733 values are explicitly mapped to their old aggregation definitions.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from r3_common import (
    IC_STABLE_SIGN_THRESHOLD, MODEL_ORDER, R2, R3, R3_CACHE, R3_TABLES, STRICT_MODELS,
    relationship_matrix, write_audit, write_csv, write_json,
)

SPACES = ("information_cluster", "raw_attribute")


def profile_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return np.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def feature_bootstrap(a: np.ndarray, b: np.ndarray, seed: int, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed); values = []
    for _ in range(n):
        idx = rng.integers(0, len(a), size=len(a)); values.append(profile_corr(a[idx], b[idx]))
    values = np.asarray(values, float)
    return float(np.nanquantile(values, .025)), float(np.nanquantile(values, .975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.n_bootstrap < 1000:
        raise ValueError("formal profile uncertainty requires >=1000 feature bootstrap replicates")
    started = time.time()
    long = pd.read_csv(R3 / "tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
    profiles = []; model_rows = []; overall_rows = []
    for space in SPACES:
        ic = relationship_matrix(long, "IC", space, "rho"); dpl = relationship_matrix(long, "dPL", space, "rho")
        ic_long = long[(long.method == "IC") & (long.space == space)].set_index(["model", "parameter_index", "feature_index"]).sort_index()
        for model in MODEL_ORDER:
            for p in range(ic[model].shape[0]):
                value = profile_corr(ic[model][p], dpl[model][p])
                anchor_rows = ic_long.loc[model, p]
                anchor_mask = (anchor_rows.rho.abs() >= .20) & (anchor_rows.bootstrap_sign_probability >= .95)
                anchored_features = anchor_mask.index[anchor_mask.to_numpy()]
                anchored_value = profile_corr(ic[model][p, anchored_features], dpl[model][p, anchored_features]) if len(anchored_features) >= 3 else np.nan
                low, high = feature_bootstrap(ic[model][p], dpl[model][p], seed=20260902 + p + list(MODEL_ORDER).index(model) * 100, n=args.n_bootstrap)
                profiles.append({"space": space, "model": model, "population": "all36", "parameter_index": p, "profile_reproducibility": value, "profile_bootstrap_ci_low": low, "profile_bootstrap_ci_high": high, "n_features": ic[model].shape[1], "ic_anchored_profile_reproducibility": anchored_value, "ic_anchored_feature_count": len(anchored_features), "profile_correlation": "Spearman across relationship features", "parameter_coordinate": "normalized_u"})
        profile_frame = pd.DataFrame([x for x in profiles if x["space"] == space])
        for model, group in profile_frame.groupby("model", sort=False):
            flattened = profile_corr(ic[model].ravel(), dpl[model].ravel())
            anchored_model = float(group.ic_anchored_profile_reproducibility.median()) if group.ic_anchored_profile_reproducibility.notna().any() else float("nan")
            model_rows.append({"space": space, "model": model, "population": "all36", "n_parameters": int(len(group)), "model_reproducibility": float(group.profile_reproducibility.median()), "model_reproducibility_mean": float(group.profile_reproducibility.mean()), "model_reproducibility_q025": float(group.profile_reproducibility.quantile(.025)), "model_reproducibility_q975": float(group.profile_reproducibility.quantile(.975)), "flattened_profile_reproducibility": flattened, "ic_anchored_reproducibility": anchored_model, "ic_anchored_parameter_count": int(group.ic_anchored_profile_reproducibility.notna().sum()), "model_aggregation": "median across common parameters; flattened is one profile over all parameter-feature cells"})
        for population, pop_models in (("all36", MODEL_ORDER), ("exclude_simhyd", STRICT_MODELS)):
            selected_profiles = profile_frame[profile_frame.model.isin(pop_models)].profile_reproducibility.to_numpy(float)
            selected_models = pd.DataFrame([x for x in model_rows if x["space"] == space and x["model"] in pop_models])
            overall_rows.extend([
                {"space": space, "population": population, "estimand": "parameter_profile_model_equal_median", "value": float(selected_models.model_reproducibility.median()), "denominator": len(pop_models), "definition": "median over models of median parameter-profile Spearman"},
                {"space": space, "population": population, "estimand": "parameter_profile_model_equal_mean", "value": float(selected_models.model_reproducibility.mean()), "denominator": len(pop_models), "definition": "mean over model medians"},
                {"space": space, "population": population, "estimand": "model_flattened_profile_median", "value": float(selected_models.flattened_profile_reproducibility.median()), "denominator": len(pop_models), "definition": "median of per-model flattened relationship-profile Spearman correlations"},
                {"space": space, "population": population, "estimand": "cell_equal_profile_median", "value": float(np.nanmedian(selected_profiles)), "denominator": len(selected_profiles), "definition": "median over all model-parameter profile correlations"},
                {"space": space, "population": population, "estimand": "ic_anchored_profile_model_equal_median", "value": float(selected_models.ic_anchored_reproducibility.median()), "denominator": int(selected_models.ic_anchored_reproducibility.notna().sum()), "definition": "median over models of median parameter profiles restricted to IC |rho|>=0.20 and IC bootstrap sign probability>=0.95"},
                {"space": space, "population": population, "estimand": "ic_anchored_cell_equal_median", "value": float(np.nanmedian(profile_frame[profile_frame.model.isin(pop_models)].ic_anchored_profile_reproducibility)), "denominator": int(profile_frame[profile_frame.model.isin(pop_models)].ic_anchored_profile_reproducibility.notna().sum()), "definition": "median over IC-anchored parameter profiles"},
            ])
    profile_frame = pd.DataFrame(profiles); model_frame = pd.DataFrame(model_rows); overall = pd.DataFrame(overall_rows)
    write_csv(R3_TABLES / "R3_PARAMETER_PROFILE_REPRODUCIBILITY.csv", profile_frame)
    write_csv(R3_TABLES / "R3_MODEL_REPRODUCIBILITY.csv", model_frame)
    write_csv(R3_TABLES / "R3_OVERALL_REPRODUCIBILITY.csv", overall)
    write_json(R3_CACHE / "R3_PROFILE_MANIFEST.json", {"analysis": "R3-B parameter-profile reproducibility", "primary_space": "information_cluster", "primary_estimand": "median_models(median_parameters(Spearman(IC_profile,dPL_profile)))", "secondary_space": "raw_attribute", "n_features_cluster": int(profile_frame[profile_frame.space == "information_cluster"].n_features.iloc[0]), "feature_bootstrap": args.n_bootstrap, "seed": 20260902, "populations": {"all36": list(MODEL_ORDER), "exclude_simhyd": list(STRICT_MODELS)}, "runtime_seconds": time.time() - started})
    primary = overall[(overall.space == "information_cluster") & (overall.population == "all36") & (overall.estimand == "parameter_profile_model_equal_median")].value.iloc[0]
    raw = overall[(overall.space == "raw_attribute") & (overall.population == "all36") & (overall.estimand == "parameter_profile_model_equal_median")].value.iloc[0]
    anchored = overall[(overall.space == "information_cluster") & (overall.population == "all36") & (overall.estimand == "ic_anchored_profile_model_equal_median")].value.iloc[0]
    result = f"Unfiltered information-cluster model-equal median reproducibility was {primary:.6f}; IC-anchored profile sensitivity was {anchored:.6f}; raw-attribute sensitivity was {raw:.6f}. Per-parameter, per-model profiles, feature-bootstrap CIs, IC-anchored counts, model medians, cell-equal summaries, and all36/exclude_simhyd are saved."
    write_audit(R3 / "R3_REPRODUCIBILITY_DEFINITION_AUDIT.md", "R3-B Reproducibility Definition Audit", "What exactly is the primary IC--dPL reproducibility estimand, and what do the historical approximately 0.658 and 0.733 numbers mean?", f"Relationship matrices `{R3 / 'tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv'}` built from the canonical R2 table. The primary feature space is the data-defined 0.70 information-cluster space; raw attributes are sensitivity.", "Primary R_m,p = Spearman across information-cluster relationship vectors for each common model parameter; R_m = median_p R_m,p; R_overall = median_m R_m. Secondary model-flattened, mean, cell-equal, raw-attribute, and exclude_simhyd estimands are retained.", "Primary denominator: 36 model medians, each model contributing equally; each model's denominator is its own registry parameter count. Profile feature denominator is 20 primary clusters (varies only if clustering contract changes, which is prohibited here).", "Feature-bootstrap CIs use 1000 resamples to quantify profile-feature sampling variability. No basin or parameter is silently dropped; missing/nonfinite profile cells fail the matrix builder. Historical mapping: formal current report's ~0.6584 is the median of 36 model-level flattened profile correlations; the ~0.733 value corresponds to the model-equal median of per-model parameter-profile medians (~0.7326 in the current formal table), not the cell-equal 0.7527 parameter-cell median. Exact historical values and changed exploratory definitions are retained in `project/benchmark/results/ic_dpl_seenbasin_formal_20260901/EXPLORATORY_ATLAS_COMPARISON.csv`.", result, "The primary is frozen before sign, dominant-control, and null results are interpreted. Raw attribute, model-flattened, cell-equal, all36/exclude_simhyd, IC-anchored, and feature-bootstrap sensitivity are not substituted for it.", "Profile correlation is reproducibility of association patterns, not agreement of parameter estimates or causal effects. A reviewer can challenge feature-bootstrap uncertainty, rank ties, common attribute confounding, dPL construction, and unequal parameter counts; those are addressed in separate audits.", "R3_B_PRIMARY_ESTIMAND_FROZEN", time.time() - started)
    dictionary = f"""# R3 Estimand Dictionary\n\n## Primary\n\nFor every model `m` and common parameter `p`, `R_m,p = Spearman(rho_IC[m,p,k], rho_dPL[m,p,k])` over the frozen information clusters `k` at attribute-clustering threshold 0.70. Then `R_m = median_p R_m,p` and `R_overall = median_m R_m`. Models are equal; parameters are equal only within model through the within-model median.\n\n## IC-anchored sensitivity\n\nFor each model-parameter profile, retain only information-cluster features with IC `|rho| >= 0.20` and IC bootstrap sign probability `>= 0.95`; calculate the same IC-versus-dPL profile correlation when at least three such features remain. This denominator is selected by IC only and is the narrower persistence check used in the dPL construction-artifact audit.\n\n## Secondary\n\n- Raw-attribute parameter-profile reproducibility: same operation over 35 raw attributes.\n- Model-flattened profile: per-model correlation after flattening all parameter × feature cells, then median over models.\n- Cell-equal profile: median of all model × parameter profile correlations.\n- `all36` retains accepted SIMHYD generation 280; `exclude_simhyd` is sensitivity.\n\n## Historical mapping\n\n- Approximately `0.658`: legacy median of per-model flattened profile correlations (formal label `model_flattened_profile_spearman_median`).\n- Approximately `0.733`: legacy/model-equal median of per-model parameter-profile medians (current formal table gives approximately `0.7326` under that aggregation).\n- Approximately `0.7527`: cell-equal median of per-parameter profile correlations in the current formal 36-model table.\n\nThese numbers are not interchangeable because their observation units and aggregation differ.\n"""
    (R3 / "R3_ESTIMAND_DICTIONARY.md").write_text(dictionary)


if __name__ == "__main__":
    main()
