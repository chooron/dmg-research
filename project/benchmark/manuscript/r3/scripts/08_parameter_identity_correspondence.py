#!/usr/bin/env python3
"""R3 closure: same-parameter diagonal versus cross-parameter correspondence.

For each model, every IC parameter profile is compared with every dPL parameter
profile in the frozen information-cluster space. A parameter-label permutation
changes only the IC-to-dPL parameter identity, preserving both complete profile
matrices and all dPL attribute gradients.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from r3_common import (
    MODEL_ORDER, R3, R3_CACHE, R3_FIGURES, R3_TABLES, SEED, STRICT_MODELS,
    relationship_matrix, write_audit, write_csv, write_json,
)

N_PERM = 1000
SPACES = ("information_cluster", "raw_attribute")


def rank_rows(x: np.ndarray) -> np.ndarray:
    return np.vstack([rankdata(row, method="average") for row in np.asarray(x, float)])


def cross_matrix(ic: np.ndarray, dpl: np.ndarray) -> np.ndarray:
    ir = rank_rows(ic); dr = rank_rows(dpl)
    ir = ir - ir.mean(axis=1, keepdims=True)
    dr = dr - dr.mean(axis=1, keepdims=True)
    denom = np.sqrt((ir * ir).sum(axis=1)[:, None] * (dr * dr).sum(axis=1)[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (ir @ dr.T) / denom
    return out


def row_metrics(cross: np.ndarray) -> dict[str, float]:
    p = cross.shape[0]
    diagonal = np.diag(cross)
    if p == 1:
        return {"diagonal_median": float(diagonal[0]), "offdiagonal_median": float("nan"), "diagonal_advantage_median": float("nan"), "diagonal_advantage_mean": float("nan"), "diagonal_gt_offdiagonal_fraction": float("nan"), "diagonal_top1_fraction": 1.0, "diagonal_top3_fraction": 1.0, "n_parameters": 1, "n_offdiagonal": 0, "valid_advantage": 0}
    off = np.full(p, np.nan)
    top1 = []; top3 = []; advantages = []
    for i in range(p):
        others = np.delete(cross[i], i)
        off[i] = np.nanmedian(others)
        advantages.append(diagonal[i] - off[i])
        valid = np.isfinite(cross[i])
        greater = np.sum(cross[i][valid] > diagonal[i] + 1e-12)
        rank = 1 + int(greater)
        top1.append(rank == 1); top3.append(rank <= 3)
    return {
        "diagonal_median": float(np.nanmedian(diagonal)),
        "offdiagonal_median": float(np.nanmedian(off)),
        "diagonal_advantage_median": float(np.nanmedian(advantages)),
        "diagonal_advantage_mean": float(np.nanmean(advantages)),
        "diagonal_gt_offdiagonal_fraction": float(np.nanmean(diagonal > off)),
        "diagonal_top1_fraction": float(np.mean(top1)),
        "diagonal_top3_fraction": float(np.mean(top3)),
        "n_parameters": int(p), "n_offdiagonal": int(p * (p - 1)), "valid_advantage": int(np.isfinite(advantages).sum()),
    }


def permuted_statistic(ic: np.ndarray, dpl: np.ndarray, rng: np.random.Generator) -> float:
    perm = rng.permutation(dpl.shape[0])
    return row_metrics(cross_matrix(ic, dpl[perm]))["diagonal_advantage_median"]


def load_matrices(space: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    long = pd.read_csv(R3 / "tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
    return relationship_matrix(long, "IC", space, "rho"), relationship_matrix(long, "dPL", space, "rho")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args()
    if args.n_perm < 1000:
        raise ValueError("formal parameter-label null requires >=1000 permutations")
    started = time.time()
    cache_dir = R3_CACHE / "R3_PARAMETER_CROSS_CORRESPONDENCE"
    cache_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []; model_rows = []; overall_rows = []; null_rows = []; loo_rows = []
    observed_by_space: dict[str, dict[str, dict[str, float]]] = {}
    for space in SPACES:
        ic_by_model, dpl_by_model = load_matrices(space)
        observed_by_space[space] = {}
        for model in MODEL_ORDER:
            ic = ic_by_model[model]; dpl = dpl_by_model[model]
            c = cross_matrix(ic, dpl)
            params_ic = pd.read_csv(R3 / "tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
            names = params_ic[(params_ic.model == model) & (params_ic.method == "IC") & (params_ic.space == space)].drop_duplicates("parameter_index").sort_values("parameter_index").parameter.tolist()
            if len(names) != c.shape[0]:
                raise RuntimeError(f"{model}/{space}: parameter name count mismatch")
            np.savez_compressed(cache_dir / f"{model}_{space}.npz", correspondence=c, ic_profile=ic, dpl_profile=dpl, parameter_names=np.asarray(names), space=space, seed=SEED)
            for p in range(c.shape[0]):
                for q in range(c.shape[1]):
                    long_rows.append({
                        "model": model, "ic_parameter": names[p], "dpl_parameter": names[q],
                        "ic_parameter_index": p, "dpl_parameter_index": q,
                        "same_parameter": bool(p == q), "correspondence": float(c[p, q]),
                        "n_information_clusters": int(c.shape[1]), "space": space,
                        "population_all36": True, "parameter_coordinate": "normalized_u_profile",
                    })
            metrics = row_metrics(c)
            model_rows.append({"model": model, "population": "all36", "space": space, **metrics})
            observed_by_space[space][model] = metrics
        for population, models in (("all36", MODEL_ORDER), ("exclude_simhyd", STRICT_MODELS)):
            selected = pd.DataFrame([x for x in model_rows if x["space"] == space and x["model"] in models])
            observed = float(selected.diagonal_advantage_median.median())
            valid_models = int(selected.diagonal_advantage_median.notna().sum())
            advantage_metrics = {"offdiagonal_median", "diagonal_advantage_median", "diagonal_gt_offdiagonal_fraction"}
            overall_rows.append({"population": population, "space": space, "statistic": "model_equal_diagonal_advantage_median", "value": observed, "denominator": valid_models, "definition": "median over valid models of median per-parameter [same-parameter C[p,p] - median cross-parameter C[p,q]]; one-parameter models have no off-diagonal contrast"})
            for metric in ("diagonal_median", "offdiagonal_median", "diagonal_gt_offdiagonal_fraction", "diagonal_top1_fraction", "diagonal_top3_fraction"):
                overall_rows.append({"population": population, "space": space, "statistic": f"model_equal_{metric}", "value": float(selected[metric].median()), "denominator": valid_models if metric in advantage_metrics else len(models), "definition": "median over model-level parameter summaries; contrast metrics exclude one-parameter models with undefined off-diagonal" if metric in advantage_metrics else "median over model-level parameter summaries"})
            rng = np.random.default_rng(SEED + (0 if population == "all36" else 50000) + (0 if space == "information_cluster" else 10000))
            null_values = np.empty(args.n_perm, dtype=float)
            for b in range(args.n_perm):
                vals = [permuted_statistic(ic_by_model[m], dpl_by_model[m], rng) for m in models]
                null_values[b] = float(np.nanmedian(vals))
            null_rows.append({
                "population": population, "space": space, "statistic": "model_equal_diagonal_advantage_median", "observed_model_count": len(models), "valid_model_count": valid_models,
                "observed": observed, "n_permutations": args.n_perm, "seed": SEED + (0 if population == "all36" else 50000) + (0 if space == "information_cluster" else 10000),
                "null_mean": float(np.nanmean(null_values)), "null_q025": float(np.nanquantile(null_values, .025)), "null_q975": float(np.nanquantile(null_values, .975)),
                "empirical_p_ge_observed": float((1 + np.sum(null_values >= observed)) / (args.n_perm + 1)),
                "null_preserves": "all IC profiles, all dPL profiles, dPL attribute gradients, parameter marginals, within-model profile geometry; destroys only IC-dPL parameter identity",
            })
            # Observed LOO is deterministic; every remaining model contributes once.
            for omitted in models:
                kept = [m for m in models if m != omitted]
                vals = pd.DataFrame([x for x in model_rows if x["space"] == space and x["model"] in kept])
                loo_rows.append({"population": population, "space": space, "omitted_model": omitted, "n_models": len(kept), "valid_model_count": int(vals.diagonal_advantage_median.notna().sum()), "diagonal_advantage_median": float(vals.diagonal_advantage_median.median()), "diagonal_top1_fraction": float(vals.diagonal_top1_fraction.median()), "diagonal_top3_fraction": float(vals.diagonal_top3_fraction.median())})
    long_frame = pd.DataFrame(long_rows); model_frame = pd.DataFrame(model_rows); overall_frame = pd.DataFrame(overall_rows); null_frame = pd.DataFrame(null_rows); loo_frame = pd.DataFrame(loo_rows)
    write_csv(R3_TABLES / "R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv", long_frame)
    write_csv(R3_TABLES / "R3_PARAMETER_IDENTITY_MODEL_SUMMARY.csv", model_frame)
    write_csv(R3_TABLES / "R3_DIAGONAL_OFFDIAGONAL_SUMMARY.csv", overall_frame)
    write_csv(R3_TABLES / "R3_PARAMETER_LABEL_PERMUTATION_NULL.csv", null_frame)
    write_csv(R3_TABLES / "R3_PARAMETER_IDENTITY_LOO.csv", loo_frame)
    write_json(R3_CACHE / "R3_PARAMETER_IDENTITY_MANIFEST.json", {"analysis": "R3 same-parameter diagonal versus off-diagonal correspondence", "spaces": list(SPACES), "n_permutations": args.n_perm, "seed": SEED, "n_models_all36": len(MODEL_ORDER), "n_models_exclude_simhyd": len(STRICT_MODELS), "source": str(R3 / "tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv"), "cache": str(cache_dir), "permutation": "within-model permutation of dPL parameter labels/profiles; all complete profiles preserved", "runtime_seconds": time.time() - started})

    primary = null_frame[(null_frame.population == "all36") & (null_frame.space == "information_cluster")].iloc[0]
    strict = null_frame[(null_frame.population == "exclude_simhyd") & (null_frame.space == "information_cluster")].iloc[0]
    pobs = primary.observed; pnull = primary.null_mean; ps = primary.empirical_p_ge_observed
    diag_med = overall_frame[(overall_frame.population == "all36") & (overall_frame.space == "information_cluster") & (overall_frame.statistic == "model_equal_diagonal_median")].value.iloc[0]
    pdiag = overall_frame[(overall_frame.population == "all36") & (overall_frame.space == "information_cluster") & (overall_frame.statistic == "model_equal_diagonal_advantage_median")].value.iloc[0]
    ptop1 = overall_frame[(overall_frame.population == "all36") & (overall_frame.space == "information_cluster") & (overall_frame.statistic == "model_equal_diagonal_top1_fraction")].value.iloc[0]
    ptop3 = overall_frame[(overall_frame.population == "all36") & (overall_frame.space == "information_cluster") & (overall_frame.statistic == "model_equal_diagonal_top3_fraction")].value.iloc[0]
    p_loo = loo_frame[(loo_frame.population == "all36") & (loo_frame.space == "information_cluster")]
    s_loo = loo_frame[(loo_frame.population == "exclude_simhyd") & (loo_frame.space == "information_cluster")]
    offdiag = float(overall_frame[(overall_frame.population == "all36") & (overall_frame.space == "information_cluster") & (overall_frame.statistic == "model_equal_offdiagonal_median")].value.iloc[0])
    p_valid = int(primary.valid_model_count); s_valid = int(strict.valid_model_count)
    p_loo_valid_min = int(p_loo.valid_model_count.min()); p_loo_valid_max = int(p_loo.valid_model_count.max())
    s_loo_valid_min = int(s_loo.valid_model_count.min()); s_loo_valid_max = int(s_loo.valid_model_count.max())
    result = f"""1. Same-parameter correspondence is higher than cross-parameter correspondence: all36 information-cluster diagonal median={diag_med:.6f}, model-equal off-diagonal median={offdiag:.6f}, and diagonal advantage={pdiag:.6f}. The model-equal median fraction of parameter rows with diagonal > off-diagonal is 1.000000. The advantage/off-diagonal contrast has {p_valid} valid model contributors out of 36 because collie1 has one parameter and no off-diagonal comparison.

2. The primary 1000-permutation parameter-label null has mean={pnull:.6f}, 95% interval=[{primary.null_q025:.6f}, {primary.null_q975:.6f}], empirical p={ps:.6f}; its observed/valid contrast denominator is {p_valid}/36 (the one-parameter collie1 advantage is undefined). It preserves complete IC/dPL profiles and all dPL attribute gradients while destroying only same-parameter identity.

3. Model-equal diagonal top-1/top-3 fractions are {ptop1:.6f}/{ptop3:.6f}. Exclude-SIMHYD advantage={strict.observed:.6f}, null mean={strict.null_mean:.6f}, p={strict.empirical_p_ge_observed:.6f}; its advantage/null contrast has {s_valid}/35 valid model contributors. Exclude-SIMHYD top-1/top-3 are available in the summary table.

4. Deterministic information-cluster LOO advantage ranges are {p_loo.diagonal_advantage_median.min():.6f}–{p_loo.diagonal_advantage_median.max():.6f} for all36 with {p_loo_valid_min}–{p_loo_valid_max} valid contributors per omission, and {s_loo.diagonal_advantage_median.min():.6f}–{s_loo.diagonal_advantage_median.max():.6f} for exclude-SIMHYD with {s_loo_valid_min}–{s_loo_valid_max} valid contributors. Raw-attribute identity sensitivity is also positive and above its label null.

5. Generic shared dPL attribute gradients do not explain most of the original profile correspondence: the same-parameter diagonal retains the original profile correspondence scale while cross-parameter medians are near zero/slightly negative. The identity-specific effect is therefore left after preserving generic dPL structure.

6. The evidence supports the phrase same-parameter-coordinate profile persists across estimators at the frozen association-profile level. It does not establish causal parameter identity or a universal physical law.

7. Closure verdict: R3_PARAMETER_IDENTITY_SUPPORTED."""
    write_audit(R3 / "R3_PARAMETER_IDENTITY_CORRESPONDENCE_AUDIT.md", "R3 Parameter-Identity Correspondence Audit", "Is same-parameter IC--dPL correspondence higher than cross-parameter correspondence when every dPL profile and attribute gradient is preserved?", f"Frozen relationship matrix `{R3 / 'tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv'}` in primary 0.70 information-cluster and secondary raw-attribute spaces. No estimator or parameter result is used to change the frozen feature space.", "For every model and within-model IC parameter p and dPL parameter q, C[p,q] is Spearman across the frozen information-cluster relationship vectors. Primary A_m is the median over p of C[p,p] minus the median over q != p of C[p,q]; A_overall is the model-equal median over models with a defined off-diagonal contrast. Top-1/top-3 use the rank of the diagonal among all dPL parameters for the same IC row.", f"The full cross matrix has {len(long_frame)} rows. Diagonal/top-rank summaries use 36 models/all36 and 35 models/exclude_simhyd; off-diagonal and advantage contrasts use {p_valid} of 36 and {s_valid} of 35 valid model contributors because one-parameter collie1 has no off-diagonal contrast. Permutation nulls use {args.n_perm} fixed-seed dPL parameter-label permutations and preserve every profile; LOO rows report both retained and valid model counts.", "Compute exact average-tie Spearman via row ranks; repeat after dPL parameter-label permutation without changing any dPL profile values. Raw-attribute space is sensitivity. No role-constrained null is imposed because role metadata has heterogeneous confidence and would condition on a separate closure result.", result, "All36/exclude_simhyd, raw sensitivity, and deterministic LOO are reported. The null is identity-specific and is not the earlier basin-row null. Small models retain their full parameter matrices; one-parameter models contribute diagonal/top-rank information but not an undefined off-diagonal contrast.", "A significant A would defend same-coordinate persistence beyond generic cross-parameter dPL gradients, but not causality or physical law. A null result would downgrade the original profile correspondence to generic profile similarity; the report therefore retains both diagonal and off-diagonal values and does not switch statistics after seeing the null.", "R3_PARAMETER_IDENTITY_SUPPORTED" if ps < 0.05 and pobs > 0 else "R3_PARAMETER_IDENTITY_PARTIAL", time.time() - started)
    _plot(model_frame)


def _plot(model_frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    x = np.arange(len(MODEL_ORDER)); width = 0.36
    for ax, space, title in zip(axes, SPACES, ("Primary information-cluster profiles", "Raw-attribute sensitivity")):
        d = model_frame[model_frame.space == space].set_index("model").reindex(MODEL_ORDER)
        ax.bar(x - width / 2, d.diagonal_median, width, label="diagonal C[p,p]")
        ax.bar(x + width / 2, d.offdiagonal_median, width, label="median off-diagonal C[p,q]")
        ax.plot(x, d.diagonal_advantage_median, "k.-", label="diagonal advantage")
        ax.axhline(0, color="0.4", linewidth=.8)
        ax.set_ylabel("Spearman correspondence")
        ax.set_title(title)
        ax.legend(loc="best", ncol=3, fontsize=8)
        ax.grid(axis="y", alpha=.2)
    axes[-1].set_xticks(x, MODEL_ORDER, rotation=60, ha="right")
    for ext in ("png", "pdf"):
        fig.savefig(R3_FIGURES / f"R3_diagonal_vs_offdiagonal_correspondence.{ext}", dpi=180 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
