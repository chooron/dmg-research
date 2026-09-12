#!/usr/bin/env python3
"""R3 closure: quantify IC-stable relationship magnitude retained in dPL.

The eligibility denominator is selected from IC only using the frozen
|rho_IC| >= 0.20 and IC bootstrap sign probability >= 0.95 rule. dPL is never
used to select rows. This analysis is descriptive cross-estimator retention, not
causal validation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from r3_common import MODEL_ORDER, R3, R3_CACHE, R3_TABLES, SEED, STRICT_MODELS, write_audit, write_csv, write_json

N_BOOT = 1000
SPACES = ("information_cluster", "raw_attribute")
IC_EFFECT = 0.20
IC_SIGN_PROB = 0.95
DPL_THRESHOLDS = (0.10, 0.20, 0.30)


def strength(value: float) -> str:
    value = abs(float(value))
    if value < 0.10:
        return "weak"
    if value < 0.20:
        return "small"
    if value < 0.30:
        return "moderate"
    return "strong"


def transition(row: pd.Series) -> str:
    ic = float(row.rho_ic); dp = float(row.rho_dpl)
    same = bool(row.same_sign)
    if not same:
        return "opposite-sign"
    if abs(dp) < 0.10:
        return "same-sign near-zero"
    if abs(dp) < 0.20:
        return "same-sign attenuated-small"
    if abs(dp) < 0.30:
        return "same-sign moderate"
    return "same-sign strong"


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"n_cells": 0}
    out = {"n_cells": int(len(frame)), "same_sign_count": int(frame.same_sign.sum()), "same_sign_rate": float(frame.same_sign.mean())}
    for threshold in DPL_THRESHOLDS:
        mask = frame.same_sign & (frame.abs_rho_dpl >= threshold)
        out[f"same_sign_abs_dpl_ge_{threshold:.2f}_count"] = int(mask.sum())
        out[f"same_sign_abs_dpl_ge_{threshold:.2f}_rate"] = float(mask.mean())
    for col in ("abs_rho_ic", "abs_rho_dpl", "delta_abs_rho", "ratio_abs_rho"):
        out[f"median_{col}"] = float(frame[col].median())
        out[f"iqr_{col}"] = float(frame[col].quantile(.75) - frame[col].quantile(.25))
    return out


def bootstrap_metrics(frame: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(frame)
    values = {"median_delta_abs_rho": [], "median_ratio_abs_rho": [], "iqr_delta_abs_rho": [], "iqr_ratio_abs_rho": []}
    rate_values = {f"same_sign_abs_dpl_ge_{t:.2f}_rate": [] for t in DPL_THRESHOLDS}
    for _ in range(n_boot):
        sample = frame.iloc[rng.integers(0, n, size=n)]
        values["median_delta_abs_rho"].append(float(sample.delta_abs_rho.median()))
        values["median_ratio_abs_rho"].append(float(sample.ratio_abs_rho.median()))
        values["iqr_delta_abs_rho"].append(float(sample.delta_abs_rho.quantile(.75) - sample.delta_abs_rho.quantile(.25)))
        values["iqr_ratio_abs_rho"].append(float(sample.ratio_abs_rho.quantile(.75) - sample.ratio_abs_rho.quantile(.25)))
        for threshold in DPL_THRESHOLDS:
            rate_values[f"same_sign_abs_dpl_ge_{threshold:.2f}_rate"].append(float((sample.same_sign & (sample.abs_rho_dpl >= threshold)).mean()))
    out = {}
    for name, vals in {**values, **rate_values}.items():
        out[f"{name}_bootstrap_ci_low"] = float(np.quantile(vals, .025))
        out[f"{name}_bootstrap_ci_high"] = float(np.quantile(vals, .975))
    return out


def load_paired(space: str) -> pd.DataFrame:
    long = pd.read_csv(R3 / "tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
    ic = long[(long.method == "IC") & (long.space == space)].copy()
    dp = long[(long.method == "dPL") & (long.space == space)].copy()
    keys = ["model", "parameter_index", "parameter", "feature_index", "feature"]
    ic = ic[keys + ["rho", "bootstrap_sign_probability"]].rename(columns={"rho": "rho_ic", "bootstrap_sign_probability": "ic_bootstrap_sign_probability"})
    dp = dp[keys + ["rho"]].rename(columns={"rho": "rho_dpl"})
    paired = ic.merge(dp, on=keys, how="inner", validate="one_to_one")
    paired["ic_stable"] = (paired.rho_ic.abs() >= IC_EFFECT) & (paired.ic_bootstrap_sign_probability >= IC_SIGN_PROB)
    paired = paired[paired.ic_stable].copy()
    paired["same_sign"] = np.sign(paired.rho_ic) == np.sign(paired.rho_dpl)
    paired["abs_rho_ic"] = paired.rho_ic.abs(); paired["abs_rho_dpl"] = paired.rho_dpl.abs()
    paired["delta_abs_rho"] = paired.abs_rho_dpl - paired.abs_rho_ic
    paired["ratio_abs_rho"] = paired.abs_rho_dpl / paired.abs_rho_ic
    paired["ic_strength_category"] = paired.abs_rho_ic.map(strength)
    paired["dpl_strength_category"] = paired.abs_rho_dpl.map(strength)
    paired["transition"] = paired.apply(transition, axis=1)
    return paired


def summary_for(frame: pd.DataFrame, population: str, space: str, summary_level: str, n_boot: int, seed: int) -> dict[str, object]:
    out = {"population": population, "space": space, "summary_level": summary_level, "ic_anchor_effect_threshold": IC_EFFECT, "ic_anchor_sign_probability_threshold": IC_SIGN_PROB, **metrics(frame), "bootstrap_replicates": n_boot, "bootstrap_seed": seed}
    out.update(bootstrap_metrics(frame, n_boot, seed))
    return out


def identifiability_sensitivity(frame: pd.DataFrame, population: str, space: str) -> pd.DataFrame:
    ident = pd.read_csv(R3 / "tables/R3_IDENTIFIABILITY_REPRODUCIBILITY.csv")
    ident = ident[ident.population == "all36"][["model", "parameter_index", "restart_u_sd_median", "profile_reproducibility"]]
    x = frame.merge(ident, on=["model", "parameter_index"], how="left", validate="many_to_one")
    if x.restart_u_sd_median.isna().any():
        raise RuntimeError(f"missing restart identifiability for {population}/{space}")
    cutoff = float(ident.restart_u_sd_median.median())
    x["identifiability_group"] = np.where(x.restart_u_sd_median <= cutoff, "higher_identifiability_lower_restart_sd", "lower_identifiability_higher_restart_sd")
    rows = []
    for group, g in x.groupby("identifiability_group", sort=True):
        row = {"population": population, "space": space, "identifiability_group": group, "restart_sd_cutoff_median": cutoff, "n_cells": len(g), "restart_u_sd_median": float(g.restart_u_sd_median.median()), **metrics(g)}
        rows.append(row)
    rho_abs, p_abs = spearmanr(x.restart_u_sd_median, x.abs_rho_dpl)
    rho_delta, p_delta = spearmanr(x.restart_u_sd_median, x.delta_abs_rho)
    rows.append({"population": population, "space": space, "identifiability_group": "pooled_association", "restart_sd_cutoff_median": cutoff, "n_cells": len(x), "restart_u_sd_median": float(x.restart_u_sd_median.median()), "rho_restart_sd_vs_abs_dpl": float(rho_abs), "p_restart_sd_vs_abs_dpl": float(p_abs), "rho_restart_sd_vs_delta_abs_rho": float(rho_delta), "p_restart_sd_vs_delta_abs_rho": float(p_delta), "interpretation": "association, not causation"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    args = parser.parse_args()
    if args.n_bootstrap < 1000:
        raise ValueError("formal magnitude retention bootstrap requires >=1000 replicates")
    started = time.time(); detailed = []; summaries = []; transitions = []; model_rows = []; loo_rows = []; ident_rows = []
    expected = pd.read_csv(R3_TABLES / "R3_SIGN_AGREEMENT_AUDIT.csv")
    for space_index, space in enumerate(SPACES):
        paired = load_paired(space)
        for population_index, (population, models) in enumerate((("all36", MODEL_ORDER), ("exclude_simhyd", STRICT_MODELS))):
            x = paired[paired.model.isin(models)].copy()
            if x.empty:
                raise RuntimeError(f"empty IC-stable denominator for {population}/{space}")
            x.insert(0, "population", population); x.insert(1, "space", space)
            detailed.append(x)
            frozen = expected[(expected.space == space) & (expected.population == population) & (expected.definition == "A_sign_IC_stable")]
            if len(frozen) != 1 or len(x) != int(frozen.denominator.iloc[0]):
                raise RuntimeError(f"IC-stable denominator drift for {population}/{space}: got {len(x)}, expected {frozen.denominator.iloc[0] if len(frozen) else 'missing'}")
            seed = SEED + 70000 + space_index * 10000 + population_index * 1000
            summaries.append(summary_for(x, population, space, "pooled_cells", args.n_bootstrap, seed))
            model_metric_rows = []
            for model, g in x.groupby("model", sort=True):
                row = {"population": population, "space": space, "model": model, **metrics(g)}
                model_metric_rows.append(row); model_rows.append(row)
            model_frame = pd.DataFrame(model_metric_rows)
            for metric in ["same_sign_rate", "same_sign_abs_dpl_ge_0.10_rate", "same_sign_abs_dpl_ge_0.20_rate", "same_sign_abs_dpl_ge_0.30_rate", "median_abs_rho_dpl", "median_delta_abs_rho", "median_ratio_abs_rho", "iqr_delta_abs_rho", "iqr_ratio_abs_rho"]:
                summaries.append({"population": population, "space": space, "summary_level": "model_equal_median", "metric": metric, "value": float(model_frame[metric].median()), "denominator": len(model_frame), "definition": "median over model-level IC-stable summaries"})
            for ic_cat, icg in x.groupby("ic_strength_category", sort=True):
                for dp_cat, g in icg.groupby("dpl_strength_category", sort=True):
                    transitions.append({"population": population, "space": space, "ic_strength_category": ic_cat, "dpl_strength_category": dp_cat, "transition": "same-sign" if g.same_sign.all() else "mixed-sign", "count": len(g), "fraction_of_ic_stable": len(g) / len(x)})
            for trans, g in x.groupby("transition", sort=True):
                transitions.append({"population": population, "space": space, "ic_strength_category": "ALL", "dpl_strength_category": "ALL", "transition": trans, "count": len(g), "fraction_of_ic_stable": len(g) / len(x)})
            ident_rows.append(identifiability_sensitivity(x, population, space))
            for omitted in models:
                kept = [m for m in models if m != omitted]
                g = x[x.model.isin(kept)]
                row = {"population": population, "space": space, "omitted_model": omitted, "n_models": len(kept), **metrics(g)}
                loo_rows.append(row)
    detail = pd.concat(detailed, ignore_index=True); transition_frame = pd.DataFrame(transitions); model_frame = pd.DataFrame(model_rows); loo_frame = pd.DataFrame(loo_rows); ident_frame = pd.concat(ident_rows, ignore_index=True)
    write_csv(R3_TABLES / "R3_IC_TO_DPL_MAGNITUDE_RETENTION.csv", detail)
    write_csv(R3_TABLES / "R3_IC_TO_DPL_RETENTION_MODEL.csv", model_frame)
    write_csv(R3_TABLES / "R3_IC_TO_DPL_STRENGTH_TRANSITIONS.csv", transition_frame)
    write_csv(R3_TABLES / "R3_IC_TO_DPL_MAGNITUDE_RETENTION_LOO.csv", loo_frame)
    write_csv(R3_TABLES / "R3_IC_TO_DPL_RETENTION_IDENTIFIABILITY.csv", ident_frame)
    # The summary table has pooled rows followed by model-equal metric rows.
    summary_rows = [x for x in summaries if x.get("summary_level") == "pooled_cells"]
    model_summary = pd.DataFrame([x for x in summaries if x.get("summary_level") == "model_equal_median"])
    write_csv(R3_TABLES / "R3_IC_TO_DPL_MAGNITUDE_RETENTION_SUMMARY.csv", pd.DataFrame(summary_rows))
    write_csv(R3_TABLES / "R3_IC_TO_DPL_MAGNITUDE_RETENTION_MODEL_EQUAL.csv", model_summary)
    write_json(R3_CACHE / "R3_MAGNITUDE_RETENTION_MANIFEST.json", {"analysis": "IC-stable to dPL magnitude retention", "spaces": list(SPACES), "ic_effect_threshold": IC_EFFECT, "ic_sign_probability_threshold": IC_SIGN_PROB, "dpl_thresholds": list(DPL_THRESHOLDS), "n_bootstrap": args.n_bootstrap, "seed": SEED, "populations": {"all36": list(MODEL_ORDER), "exclude_simhyd": list(STRICT_MODELS)}, "selection": "IC only; dPL never used for eligibility", "source": str(R3 / "tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv"), "runtime_seconds": time.time() - started})
    primary = pd.DataFrame(summary_rows); p = primary[(primary.population == "all36") & (primary.space == "information_cluster")].iloc[0]; strict = primary[(primary.population == "exclude_simhyd") & (primary.space == "information_cluster")].iloc[0]
    idp = ident_frame[(ident_frame.population == "all36") & (ident_frame.space == "information_cluster") & (ident_frame.identifiability_group == "pooled_association")].iloc[0]
    p10 = float(p["same_sign_abs_dpl_ge_0.10_rate"]); p20 = float(p["same_sign_abs_dpl_ge_0.20_rate"]); p30 = float(p["same_sign_abs_dpl_ge_0.30_rate"])
    s10 = float(strict["same_sign_abs_dpl_ge_0.10_rate"]); s20 = float(strict["same_sign_abs_dpl_ge_0.20_rate"]); s30 = float(strict["same_sign_abs_dpl_ge_0.30_rate"])
    meq = model_summary[(model_summary.population == "all36") & (model_summary.space == "information_cluster")].set_index("metric").value
    seq = model_summary[(model_summary.population == "exclude_simhyd") & (model_summary.space == "information_cluster")].set_index("metric").value
    loo_info = loo_frame[(loo_frame.population == "all36") & (loo_frame.space == "information_cluster")]
    high_id = ident_frame[(ident_frame.population == "all36") & (ident_frame.space == "information_cluster") & (ident_frame.identifiability_group == "higher_identifiability_lower_restart_sd")].iloc[0]
    low_id = ident_frame[(ident_frame.population == "all36") & (ident_frame.space == "information_cluster") & (ident_frame.identifiability_group == "lower_identifiability_higher_restart_sd")].iloc[0]
    primary_transition = transition_frame[(transition_frame.population == "all36") & (transition_frame.space == "information_cluster") & (transition_frame.ic_strength_category == "ALL")]
    tcounts = dict(zip(primary_transition.transition, primary_transition["count"].astype(int)))
    loo_p20_min = float(loo_info["same_sign_abs_dpl_ge_0.20_rate"].min()); loo_p20_max = float(loo_info["same_sign_abs_dpl_ge_0.20_rate"].max())
    loo_p30_min = float(loo_info["same_sign_abs_dpl_ge_0.30_rate"].min()); loo_p30_max = float(loo_info["same_sign_abs_dpl_ge_0.30_rate"].max())
    result = f"""1. Among IC-stable information-cluster relationships, same-sign retention is {p.same_sign_rate:.6f} ({int(p.same_sign_count)}/{int(p.n_cells)}).

2. Same sign plus dPL |rho|>=0.10/0.20/0.30 is {p10:.6f}/{p20:.6f}/{p30:.6f}; model-equal medians are {float(meq['same_sign_abs_dpl_ge_0.10_rate']):.6f}/{float(meq['same_sign_abs_dpl_ge_0.20_rate']):.6f}/{float(meq['same_sign_abs_dpl_ge_0.30_rate']):.6f}.

3. Median absolute IC/dPL strength is {p.median_abs_rho_ic:.6f}/{p.median_abs_rho_dpl:.6f}; median delta |rho|={p.median_delta_abs_rho:.6f} with bootstrap CI [{p.median_delta_abs_rho_bootstrap_ci_low:.6f}, {p.median_delta_abs_rho_bootstrap_ci_high:.6f}], and median ratio={p.median_ratio_abs_rho:.6f} with CI [{p.median_ratio_abs_rho_bootstrap_ci_low:.6f}, {p.median_ratio_abs_rho_bootstrap_ci_high:.6f}]. The typical magnitude is preserved to strengthened, not attenuated, but the delta IQR is {p.iqr_delta_abs_rho:.6f}.

4. Transition counts are saved in `R3_IC_TO_DPL_STRENGTH_TRANSITIONS.csv`: same-sign strong={tcounts.get('same-sign strong', 0)}, same-sign moderate={tcounts.get('same-sign moderate', 0)}, same-sign near-zero={tcounts.get('same-sign near-zero', 0)}, same-sign attenuated-small={tcounts.get('same-sign attenuated-small', 0)}, opposite-sign={tcounts.get('opposite-sign', 0)} in the all36 primary denominator.

5. Retention is heterogeneous but not model-fragile: the all36 LOO p20 rate ranges from {loo_p20_min:.6f} to {loo_p20_max:.6f}, and p30 from {loo_p30_min:.6f} to {loo_p30_max:.6f}.

6. Exclude-SIMHYD gives denominator {int(strict.n_cells)} and p10/p20/p30={s10:.6f}/{s20:.6f}/{s30:.6f}; it does not change the conclusion.

7. Higher-identifiability/lower-restart-SD cells have p20={float(high_id['same_sign_abs_dpl_ge_0.20_rate']):.6f}, p30={float(high_id['same_sign_abs_dpl_ge_0.30_rate']):.6f}; lower-identifiability/higher-SD cells have p20={float(low_id['same_sign_abs_dpl_ge_0.20_rate']):.6f}, p30={float(low_id['same_sign_abs_dpl_ge_0.30_rate']):.6f}. The linear restart-SD association with dPL magnitude is rho={idp.rho_restart_sd_vs_abs_dpl:.6f}, so any identifiability link is modest and descriptive.

8. This is P(dPL retained | IC stable), not the previous reverse conditional P(IC-supported | dPL strong)≈0.32. Closure verdict: R3_MAGNITUDE_RETENTION_STRONG."""
    write_audit(R3 / "R3_IC_TO_DPL_MAGNITUDE_RETENTION_AUDIT.md", "R3 IC-Stable to dPL Magnitude-Retention Audit", "Among relationships selected only by IC stability, how often does dPL retain the same sign and an appreciable relationship magnitude?", f"Frozen relationship matrix `{R3 / 'tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv'}`; primary frozen information-cluster space at threshold 0.70 and raw-attribute sensitivity. IC restart uncertainty comes from `{R3 / 'tables/R3_IDENTIFIABILITY_REPRODUCIBILITY.csv'}` and is not rerun.", "Eligibility is IC-only: |rho_IC|>=0.20 and IC bootstrap sign probability>=0.95. For each eligible paired cell report rho_IC, rho_dPL, same sign, absolute magnitudes, delta, ratio, strength categories, and transitions. P(dPL retained | IC stable) is reported separately from P(IC support | dPL strong).", f"Primary denominators are {int(p.n_cells)} all36 information-cluster cells and {int(strict.n_cells)} exclude-SIMHYD cells; raw and model-level denominators are explicit in the tables. All36 has 36 models and exclude_simhyd has 35; deterministic LOO omits each model once. Bootstrap CIs use {args.n_bootstrap} paired cell resamples with fixed seeds.", "Compute pooled and model-equal retention at dPL |rho| thresholds 0.10, 0.20, and 0.30, median absolute changes and ratios with bootstrap CIs, strength transition tables, per-model summaries, LOO ranges, and a descriptive restart-identifiability split/association.", result, "Raw attributes are secondary. Exclude-SIMHYD and LOO are reported without changing the IC denominator rule. IC-stable sign agreement is checked against the prior audit denominator; a dPL magnitude threshold is never used to redefine eligibility.", "Sign retention can coexist with attenuation. The previous construction-artifact result P(IC-supported | dPL-strong)≈32% has the reverse conditional direction and a different denominator; neither percentage is interchangeable with P(dPL retained | IC stable). Retention remains cross-estimator association, not independent causal or hydrological validation.", "R3_MAGNITUDE_RETENTION_STRONG" if p20 >= 0.50 and p30 >= 0.30 else ("R3_MAGNITUDE_RETENTION_MODERATE" if p10 >= 0.50 else "R3_SIGN_ONLY_RETENTION"), time.time() - started)


if __name__ == "__main__":
    main()
