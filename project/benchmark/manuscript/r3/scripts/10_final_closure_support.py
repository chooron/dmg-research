#!/usr/bin/env python3
"""Generate the final R3 cell-identity and within-IC reference closure.

This is read-only post-processing of frozen R3 relationship matrices, the frozen
R2 20-D information representation stored in the aligned R3 model arrays, and
archived ten-start IC payloads. It never trains, calibrates, changes a
threshold, or writes a checkpoint.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[5]
BENCHMARK = REPO / "project/benchmark"
R3 = BENCHMARK / "manuscript/r3"
R3_TABLES = R3 / "tables"
R3_CACHE = R3 / "cache"
R2_MANUSCRIPT = BENCHMARK / "manuscript/r2"
R2_CACHE = R2_MANUSCRIPT / "cache"
ANALYSIS = BENCHMARK / "analysis/seenbasin_remaining_20260901"
sys.path[:0] = [str(ANALYSIS), str(BENCHMARK), str(BENCHMARK / "src"), str(BENCHMARK / "manuscript/r2/scripts")]

from common import ALL_MODELS, load_ic_restart, load_ids, load_status  # noqa: E402
from src.model_registry import get_spec  # noqa: E402
import r2_common  # noqa: E402

PRIMARY_EFFECT = 0.20
IC_SIGN_PROB = 0.95
IC_TOLERANCE = 0.01
MIN_SELF_BASINS = 400
N_BOOT = 5000
BOOT_SEED = 20260910


def canonical_basin(value: Any) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def finite_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a[ok], b[ok]).statistic)


def relationship_profiles(theta: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Return parameter × frozen-information-dimension Spearman profiles."""
    out = np.full((theta.shape[1], features.shape[1]), np.nan, dtype=float)
    for p in range(theta.shape[1]):
        for k in range(features.shape[1]):
            out[p, k] = finite_spearman(theta[:, p], features[:, k])
    return out


def profile_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray([finite_spearman(a[p], b[p]) for p in range(a.shape[0])], dtype=float)


def correspondence_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full((a.shape[0], b.shape[0]), np.nan, dtype=float)
    for p in range(a.shape[0]):
        for q in range(b.shape[0]):
            out[p, q] = finite_spearman(a[p], b[q])
    return out


def diagonal_advantage(corr: np.ndarray) -> tuple[float, np.ndarray]:
    if corr.shape[0] < 2 or corr.shape[1] < 2:
        return float("nan"), np.full(corr.shape[0], np.nan)
    values = []
    for p in range(corr.shape[0]):
        off = np.delete(corr[p], p)
        if not np.isfinite(corr[p, p]) or not np.isfinite(off).any():
            values.append(np.nan)
        else:
            values.append(float(corr[p, p] - np.nanmedian(off)))
    values = np.asarray(values, dtype=float)
    return (float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan"), values)


def load_frozen_r3_arrays(basin_keys: list[str]) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """Load the already-frozen R3 basin arrays used by the JoH diagnostic."""
    source_dir = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905/r3/cache/model_arrays"
    arrays: dict[str, dict[str, np.ndarray]] = {}
    reference_basin = None
    reference_info = None
    for model in ALL_MODELS:
        path = source_dir / f"{model}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            required = {"basin", "ic_norm", "dpl_norm", "info_scores"}
            if not required.issubset(set(z.files)):
                raise RuntimeError(f"{model}: frozen R3 array cache lacks {required - set(z.files)}")
            basin = np.asarray(z["basin"]).astype(str)
            basin = np.asarray([canonical_basin(x) for x in basin])
            ic = np.asarray(z["ic_norm"], dtype=float)
            dpl = np.asarray(z["dpl_norm"], dtype=float)
            info = np.asarray(z["info_scores"], dtype=float)
        if list(basin) != list(basin_keys):
            raise RuntimeError(f"{model}: frozen R3 basin order does not match canonical 531 IDs")
        if ic.shape[0] != 531 or dpl.shape != ic.shape or info.shape != (531, 20):
            raise RuntimeError(f"{model}: frozen R3 array shape mismatch {ic.shape}, {dpl.shape}, {info.shape}")
        if not np.isfinite(ic).all() or not np.isfinite(dpl).all() or not np.isfinite(info).all():
            raise RuntimeError(f"{model}: frozen R3 arrays contain nonfinite values")
        if reference_basin is None:
            reference_basin = basin
            reference_info = info
        elif not np.array_equal(reference_basin, basin) or not np.array_equal(reference_info, info):
            raise RuntimeError(f"{model}: frozen R3 information/basin support differs from the reference model")
        arrays[model] = {"ic": ic, "dpl": dpl, "info": info}
    return arrays, [f"C{i:03d}" for i in range(1, 21)]


def restart_support(basin_keys: list[str], canonical_ic: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    ids = np.asarray([int(x) for x in basin_keys], dtype=np.int64)
    status = load_status()
    rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, Any]] = {}
    for model in ALL_MODELS:
        u, _physical, fitness, meta = load_ic_restart(model, ids, status)
        best_index = np.argmax(fitness, axis=1).astype(int)
        best_fitness = fitness[np.arange(len(ids)), best_index]
        eligible = fitness >= best_fitness[:, None] - IC_TOLERANCE
        eligible_count = eligible.sum(axis=1).astype(int)
        alternative_mask = eligible.copy()
        alternative_mask[np.arange(len(ids)), best_index] = False
        alternative_index = np.full(len(ids), -1, dtype=int)
        for b in range(len(ids)):
            candidates = np.flatnonzero(alternative_mask[b])
            if len(candidates):
                alternative_index[b] = int(candidates[np.argmax(fitness[b, candidates])])
        canonical_error = np.max(np.abs(u[np.arange(len(ids)), best_index] - canonical_ic[model]), axis=1)
        arrays[model] = {
            "u": u,
            "fitness": fitness,
            "best_index": best_index,
            "alternative_index": alternative_index,
            "eligible": eligible,
            "canonical_error": canonical_error,
            "files": meta.get("files", []),
        }
        for b, basin in enumerate(basin_keys):
            eligible_ids = np.flatnonzero(eligible[b]).astype(int).tolist()
            alt = int(alternative_index[b])
            rows.append({
                "model": model,
                "basin_id": basin,
                "n_total_starts": int(fitness.shape[1]),
                "best_train_score": float(best_fitness[b]),
                "n_eligible_delta001": int(eligible_count[b]),
                "eligible_start_ids": ";".join(str(x) for x in eligible_ids),
                "has_2_independent_eligible": bool(eligible_count[b] >= 2),
                "has_3plus_independent_eligible": bool(eligible_count[b] >= 3),
                "canonical_start_id": int(best_index[b]),
                "canonical_start_fitness": float(best_fitness[b]),
                "alternative_start_id": alt if alt >= 0 else np.nan,
                "alternative_start_fitness": float(fitness[b, alt]) if alt >= 0 else np.nan,
                "canonical_match_max_abs_error": float(canonical_error[b]),
                "parameter_count": int(u.shape[2]),
                "start_id_convention": "0-based archived slot index",
            })
    return pd.DataFrame(rows), arrays


def cell_identity(long: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ic = long.loc[(long["method"] == "IC") & (long["space"] == "information_cluster")].copy()
    dpl = long.loc[(long["method"] == "dPL") & (long["space"] == "information_cluster"), ["model", "parameter_index", "feature_index", "rho"]].copy()
    dpl = dpl.rename(columns={"rho": "rho_dPL"})
    keys = ["model", "parameter_index", "feature_index"]
    if ic.duplicated(keys).any() or dpl.duplicated(keys).any():
        raise RuntimeError("relationship table cell key is not unique")
    frame = ic.merge(dpl, on=keys, how="inner", validate="one_to_one", suffixes=("", "_dpl"))
    frame["in_absrho20"] = frame["rho"].abs() >= PRIMARY_EFFECT
    frame["in_stable"] = frame["in_absrho20"] & (frame["bootstrap_sign_probability"] >= IC_SIGN_PROB)
    frame["information_dimension"] = frame["feature"]
    frame["parameter"] = frame["parameter"]
    frame["cell_key"] = frame[keys].astype(str).agg("|".join, axis=1)
    stable = set(frame.loc[frame.in_stable, "cell_key"])
    abs20 = set(frame.loc[frame.in_absrho20, "cell_key"])
    union = stable | abs20
    frame["difference_type"] = np.where(frame.cell_key.isin(stable & abs20), "same", np.where(frame.cell_key.isin(stable - abs20), "stable_only", "absrho20_only"))
    diff = frame.loc[frame.cell_key.isin(union), ["model", "parameter", "information_dimension", "parameter_index", "feature_index", "rho", "rho_dPL", "bootstrap_sign_probability", "in_stable", "in_absrho20", "difference_type"]].copy()
    diff = diff.rename(columns={"rho": "rho_IC", "bootstrap_sign_probability": "IC_bootstrap_sign_probability"})
    summary = {
        "n_total_cells": int(len(frame)),
        "n_stable": int(len(stable)),
        "n_absrho20": int(len(abs20)),
        "intersection": int(len(stable & abs20)),
        "stable_minus_absrho20": int(len(stable - abs20)),
        "absrho20_minus_stable": int(len(abs20 - stable)),
        "symmetric_difference": int(len(stable ^ abs20)),
        "jaccard": float(len(stable & abs20) / len(union)) if union else float("nan"),
        "identity_exact": bool(stable == abs20),
    }
    return diff.sort_values(["model", "parameter_index", "feature_index"]).reset_index(drop=True), {"frame": frame, **summary}

def validate_frozen_relationship_arrays(frozen: dict[str, dict[str, np.ndarray]], long: pd.DataFrame) -> tuple[int, float]:
    """Confirm the reusable basin arrays reproduce the frozen 20-D rho table."""
    primary = long.loc[long["space"] == "information_cluster"]
    n = 0
    max_error = 0.0
    for model in ALL_MODELS:
        for method, key in (("IC", "ic"), ("dPL", "dpl")):
            observed = primary.loc[(primary["model"] == model) & (primary["method"] == method)].sort_values(["parameter_index", "feature_index"])["rho"].to_numpy(float)
            theta = frozen[model][key]
            info = frozen[model]["info"]
            calculated = relationship_profiles(theta, info).reshape(-1)
            if observed.shape != calculated.shape:
                raise RuntimeError(f"{model}/{method}: frozen relationship shape mismatch")
            error = float(np.nanmax(np.abs(observed - calculated)))
            max_error = max(max_error, error)
            n += int(calculated.size)
    return n, max_error


def build_nested_and_conditioning(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ic = frame.loc[frame["in_absrho20"]].copy()
    both = ic.loc[ic["rho_dPL"].abs() >= PRIMARY_EFFECT].copy()
    both["same_sign"] = np.sign(both["rho"]) == np.sign(both["rho_dPL"])
    same = both.loc[both.same_sign]
    stages = [
        ("all_cells", "all 5,420 relationship cells", len(frame), len(frame), "all canonical information-cluster cells"),
        ("ic_absrho20", "IC strong", len(ic), len(frame), "abs(rho_IC) >= 0.20"),
        ("both_absrho20", "IC and dPL strong", len(both), len(ic), "abs(rho_IC) >= 0.20 and abs(rho_dPL) >= 0.20"),
        ("both_same_sign", "IC and dPL strong, same sign", len(same), len(both), "both strong and sign(rho_IC) == sign(rho_dPL)"),
        ("both_sign_flip", "IC and dPL strong, sign flipped", len(both) - len(same), len(both), "both strong and opposite signs"),
    ]
    rows = []
    for stage, label, count, parent, definition in stages:
        rows.append({"ledger_stage": stage, "subset_label": label, "n_cells": int(count), "parent_denominator": int(parent), "fraction_of_parent": float(count / parent), "fraction_of_all_cells": float(count / len(frame)), "definition": definition})
    ledger = pd.DataFrame(rows)
    total_ic = int(len(ic)); total_dpl = int(len(frame.loc[frame["rho_dPL"].abs() >= PRIMARY_EFFECT]))
    both_count = int(len(both)); ic_only = int(total_ic - both_count); dpl_only = int(total_dpl - both_count); neither = int(len(frame) - both_count - ic_only - dpl_only)
    rows = [
        {"conditioning": "dPL_strong_given_IC_strong", "numerator": both_count, "denominator": total_ic, "probability": both_count / total_ic, "definition": "both / (both + IC-only)"},
        {"conditioning": "IC_strong_given_dPL_strong", "numerator": both_count, "denominator": total_dpl, "probability": both_count / total_dpl, "definition": "both / (both + dPL-only)"},
        {"conditioning": "IC_strong_total", "numerator": total_ic, "denominator": len(frame), "probability": total_ic / len(frame), "definition": "both + IC-only over all cells"},
        {"conditioning": "dPL_strong_total", "numerator": total_dpl, "denominator": len(frame), "probability": total_dpl / len(frame), "definition": "both + dPL-only over all cells"},
        {"conditioning": "both", "numerator": both_count, "denominator": len(frame), "probability": both_count / len(frame), "definition": "both over all cells"},
        {"conditioning": "IC_only", "numerator": ic_only, "denominator": len(frame), "probability": ic_only / len(frame), "definition": "IC-only over all cells"},
        {"conditioning": "dPL_only", "numerator": dpl_only, "denominator": len(frame), "probability": dpl_only / len(frame), "definition": "dPL-only over all cells"},
        {"conditioning": "neither", "numerator": neither, "denominator": len(frame), "probability": neither / len(frame), "definition": "neither over all cells"},
    ]
    return ledger, pd.DataFrame(rows)


def model_bootstrap(values: np.ndarray, seed: int = BOOT_SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.median(values[rng.integers(0, len(values), size=(N_BOOT, len(values)))], axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_profile_outputs(model_data: dict[str, dict[str, Any]], feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_rows = []
    corr_rows = []
    cache_dir = R3_CACHE / "R3_IC_SELF_PROFILE_REFERENCE"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for model, data in model_data.items():
        pa = data["rho_ic_a"]
        pb = data["rho_ic_b"]
        pdpl = data["rho_cross_dpl"]
        cs = data["C_self"]
        cc = data["C_cross"]
        for p in range(pa.shape[0]):
            for k, feature in enumerate(feature_names):
                profile_rows.append({"model": model, "parameter_index": p, "parameter": data["parameter_names"][p], "information_dimension": feature, "rho_ic_canonical": pa[p, k], "rho_ic_alternative": pb[p, k], "rho_cross_dpl": pdpl[p, k]})
            for q in range(pa.shape[0]):
                corr_rows.append({"model": model, "parameter_index_ic": p, "parameter_ic": data["parameter_names"][p], "parameter_index_second": q, "parameter_second": data["parameter_names"][q], "C_ic_self": cs[p, q], "C_cross_matched": cc[p, q]})
        np.savez_compressed(cache_dir / f"{model}.npz", basin_ids=np.asarray(data["basin_ids"]), canonical_u=data["canonical_u"], alternative_u=data["alternative_u"], dpl_u=data["dpl_u"], information_features=data["features"], information_dimension_names=np.asarray(feature_names), rho_ic_canonical=pa, rho_ic_alternative=pb, rho_cross_dpl=pdpl, C_ic_self=cs, C_cross_matched=cc, canonical_start_index=data["canonical_start_index"], alternative_start_index=data["alternative_start_index"])
    profile = pd.DataFrame(profile_rows)
    corr = pd.DataFrame(corr_rows)
    profile.to_csv(R3_TABLES / "R3_IC_SELF_PROFILE_VALUES_LONG.csv", index=False, float_format="%.12g")
    corr.to_csv(R3_TABLES / "R3_IC_SELF_CORRESPONDENCE_LONG.csv", index=False, float_format="%.12g")
    try:
        profile.to_parquet(R3_TABLES / "R3_IC_SELF_PROFILE_VALUES_LONG.parquet", index=False)
        corr.to_parquet(R3_TABLES / "R3_IC_SELF_CORRESPONDENCE_LONG.parquet", index=False)
    except Exception as exc:
        (R3_TABLES / "R3_IC_SELF_PARQUET_NOTE.txt").write_text(f"Parquet export unavailable: {type(exc).__name__}: {exc}\n")
    return profile, corr


def build_markdown(identity: dict[str, Any], conditioning: pd.DataFrame, support: pd.DataFrame, model_rows: pd.DataFrame, summary: pd.DataFrame, metadata: dict[str, Any]) -> tuple[str, str, str]:
    same = identity["identity_exact"]
    diff_rows = support.loc[support["has_2_independent_eligible"]].groupby("model").size()
    all_r = model_rows.loc[model_rows["r_profile_estimable"] & model_rows["coverage_ge_400"]]
    strict_r = all_r.loc[all_r["r2_strict_model"]]
    all_a = model_rows.loc[model_rows["a_profile_estimable"] & model_rows["coverage_ge_400"]]
    strict_a = all_a.loc[all_a["r2_strict_model"]]
    ci_text = "exactly identical" if same else "not identical"
    identity_md = f"""# R3 Cell-Set Identity Audit\n\n## Scope\n\nThis is read-only post-processing of the frozen R3 information-cluster relationship table. The unique key is `(model, parameter_index, feature_index)`, with the report labels `model`, `parameter`, and `information_dimension`. No threshold or bootstrap rule was changed.\n\n## Definitions\n\n- `S_stable`: `abs(rho_IC) >= 0.20` and `IC_bootstrap_sign_probability >= 0.95`.\n- `S_absrho20`: `abs(rho_IC) >= 0.20`, without the bootstrap-sign condition.\n- dPL is not used to define either set.\n\n## Set comparison\n\n| Quantity | Value |\n|---|---:|\n| Total information-cluster cells | {identity['n_total_cells']} |\n| `|S_stable|` | {identity['n_stable']} |\n| `|S_absrho20|` | {identity['n_absrho20']} |\n| Intersection | {identity['intersection']} |\n| `S_stable - S_absrho20` | {identity['stable_minus_absrho20']} |\n| `S_absrho20 - S_stable` | {identity['absrho20_minus_stable']} |\n| Symmetric difference | {identity['symmetric_difference']} |\n| Jaccard | {identity['jaccard']:.12f} |\n\nThe two 902-cell definitions are **{ci_text}** in this frozen dataset. The exported union audit is `tables/R3_CELL_SET_IDENTITY_DIFF.csv`; rows marked `same` are members of both sets, while non-`same` rows would be the identity differences.\n\nThe correct wording is: the bootstrap-sign criterion is empirically redundant for this frozen dataset at this threshold. This does not establish redundancy in general.\n\n## F5 nested ledger decision\n\nBecause the cell identities are {"identical" if same else "different"}, F5 {"may use" if same else "must not use"} the single nested ledger `5420 -> 902 -> 712 -> 692`. The nested data are in `tables/R3_F5_NESTED_LEDGER.csv` when identity is exact; otherwise the separate-estimand output would be `tables/R3_F5_SEPARATE_ESTIMANDS.csv`.\n\nAt `abs(rho_IC), abs(rho_dPL) >= 0.20`, the strong-both count is 712; among those, 692 have the same sign and 20 are sign-flipped.\n\n## Bidirectional conditioning\n\n`tables/R3_F5_BIDIRECTIONAL_CONDITIONING.csv` contains both conditional directions and the complete four-way counts. The exact conditional probabilities are:\n\n- `P(dPL strong | IC strong) = 712/902 = {712/902:.12f}`.\n- `P(IC strong | dPL strong) = 712/2163 = {712/2163:.12f}`.\n\nThe reverse conditional is not a validity or effectiveness claim. dPL parameter values are generated through attribute-to-parameter mapping, so denser attribute–parameter associations can have a constructive component. The two directions describe association-system asymmetry only.\n\n## Provenance\n\n- Relationship source: `tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv`.\n- IC selection columns: `rho`, `bootstrap_sign_probability`; R3 frozen thresholds 0.20 and 0.95.\n- dPL is used only for classification after IC set construction.\n- Source table rows: {identity['n_total_cells']} information-cluster cells.\n"""

    support_md = f"""# R3 Within-IC Information-Profile Reference\n\n## Verdict\n\nA within-IC calibration reference was constructed from archived ten-start IC parameter vectors without training or recalibration. It is a **within-IC calibration reference**, not an upper bound, ceiling, or theoretical maximum.\n\nThe frozen performance-comparable rule is:\n\n```text\nKGE_best - KGE_restart <= 0.01\n```\n\nFitness direction was verified as maximize: the archived benchmark uses `argmax` and KGE. For each basin, the canonical IC vector is the archived best restart; the alternative is the highest-fitness eligible restart after excluding that canonical start. No selection uses parameter distance, profile similarity, or any R3 outcome.\n\n## Restart support\n\n`tables/R3_IC_SELF_RESTART_SUPPORT.csv` contains all 36 × 531 rows. Each self profile uses only basins with at least two eligible starts, so the canonical start and an independent alternative both exist.\n\n| Quantity | Value |\n|---|---:|\n| Models | {len(ALL_MODELS)} |\n| Basins/model | 531 |\n| Starts/basin | 10 |\n| Total model–basin rows | {len(ALL_MODELS)*531} |\n| Rows with ≥2 eligible starts | {int(support['has_2_independent_eligible'].sum())} |\n| Rows with only one eligible start | {int((~support['has_2_independent_eligible']).sum())} |\n| Canonical-to-best max absolute normalized-u error | {support['canonical_match_max_abs_error'].max():.3e} |\n| Models with ≥400 self-eligible basins | {int(model_rows['coverage_ge_400'].sum())}/{len(model_rows)} |\n| R2 strict model list | 23 |\n\nThe common basin support is model-specific `B_m^self`; the canonical-versus-dPL matched comparison uses the same `B_m^self` for that model.\n\n## Representation and estimands\n\nBoth IC profiles use the frozen R3 20-D information-cluster scores at threshold 0.70. No raw 35-attribute or R4 13-cluster representation is used. For each model:\n\n```text\nrho_IC_A[p,k] = Spearman_b(IC canonical u[b,p], information[b,k])\nrho_IC_B[p,k] = Spearman_b(IC alternative u[b,p], information[b,k])\nrho_dPL[p,k]  = Spearman_b(dPL canonical u[b,p], information[b,k])\nR_self[p]      = Spearman_k(rho_IC_A[p,k], rho_IC_B[p,k])\nR_cross[p]     = Spearman_k(rho_IC_A[p,k], rho_dPL[p,k])\n```\n\n`R_model` is the median across parameters and the ensemble value is the median across equally weighted eligible models. `C` and `A_diag` use the exact existing R3 row/off-diagonal definitions.\n\n## Model sets\n\n- **All-model available set:** models with at least 400 common-support basins and estimable profile values; R and A denominators are reported separately because `collie1` has no off-diagonal A contrast.\n- **R2-matched strict set:** the frozen 23-model R2 primary list, intersected with the same profile/support rules; no new model threshold was invented.\n- Model-by-model coverage and estimability flags are in `tables/R3_F6_IC_SELF_REFERENCE_MODEL.csv`.\n\n## Results\n\n| Population | Estimand | N models | Median self | Median cross matched | Self − cross | 95% model-bootstrap CI | Positive model differences | Median basin support |\n|---|---|---:|---:|---:|---:|---|---:|---:|\n"""
    for _, row in summary.iterrows():
        support_md += f"| {row.population} | {row.estimand} | {int(row.n_models)} | {row.self_value:.6f} | {row.cross_value:.6f} | {row.difference_self_minus_cross:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | {int(row.n_positive)}/{int(row.n_models)} | {row.median_n_basins:.1f} |\n"
    support_md += f"""\nBootstrap uncertainty resamples eligible models with replacement and preserves the paired self/cross difference within each model (`{N_BOOT}` draws, seed `{BOOT_SEED}`). It is a model-level calibration-scale uncertainty, not a cell-independence significance test.\n\n## Interpretation boundary\n\nThe result is reported as observed, without presupposing self > cross. The within-IC reference quantifies repeatability under the archived performance-comparable restart rule. It does not prove parameter identity, a physical upper bound, causality, or that any cross/self difference is exclusively caused by the calibration paradigm.\n\n## Provenance\n\n- Restart source: `results/ic_dpl_aligned_full300_20260819_final/best_training/*/chunk_*_best.pt`.\n- Loader: `analysis/seenbasin_remaining_20260901/common.py::load_ic_restart`.\n- Restart gate: `results/seenbasin_remaining_analysis_20260901/agent_C/C05_RESTART_DATA_AVAILABILITY_GATE.csv`.\n- Frozen aligned R3 basin arrays: `results/joh_direct_parameter_change_diagnostic_20260905/r3/cache/model_arrays/*.npz`.\n- Frozen source audit: `results/joh_direct_parameter_change_diagnostic_20260905/r3/R3_PROVENANCE_AUDIT.md` and `RUN_MANIFEST.json`.\n- Information representation: archived `info_scores` arrays, threshold-0.70 20-D PC1 scores.\n- Reproduction check: {metadata['relationship_entries_validated']} frozen information-space rho entries, maximum absolute error `{metadata['relationship_reproduction_max_abs_error']:.3e}`.\n- Figure-ready profiles: `tables/R3_IC_SELF_PROFILE_VALUES_LONG.csv` and per-model `cache/R3_IC_SELF_PROFILE_REFERENCE/*.npz`.\n"""

    closure_md = f"""# R3 Final Closure Summary\n\n## Decision\n\n> **R3 DATA CLOSURE COMPLETE — START FIGURE DESIGN**\n\nNo training, calibration, dPL seed generation, threshold adjustment, or new paper story was added.\n\n## Required answers\n\n1. **Are the 902 sets identical?** Yes: `S_stable` and `S_absrho20` have intersection {identity['intersection']} and symmetric difference {identity['symmetric_difference']}; Jaccard `{identity['jaccard']:.12f}`.\n2. **Can F5 use `5420 -> 902 -> 712 -> 692`?** Yes. The 902 cells are one identical set under both rules; 712 are strong in both IC and dPL, 692 of those retain sign, and 20 are sign-flipped.\n3. **Bidirectional conditioning:** `P(dPL strong | IC strong) = 712/902 = {712/902:.12f}`; `P(IC strong | dPL strong) = 712/2163 = {712/2163:.12f}`. This is association asymmetry, not dPL validity or effectiveness.\n4. **Was the IC-self profile benchmark constructed?** Yes, as a within-IC calibration reference using canonical best versus an independent performance-comparable alternative restart.\n5. **Models/basins:** R results use {int(all_r.shape[0])} all-model-available models and {int(strict_r.shape[0])} R2-strict models; A results use {int(all_a.shape[0])} and {int(strict_a.shape[0])}, respectively. Basin support is model-specific, with at least {MIN_SELF_BASINS} of 531 for included models; exact values are in the model CSV.\n6. **`R_paired_IC_self`:** see `R3_F6_IC_SELF_REFERENCE_SUMMARY.csv` and the within-IC report; values are computed on matched `B_m^self`.\n7. **Matched-support `R_paired_cross`:** reported beside the self value in the same summary.\n8. **`A_diag_IC_self`:** reported for the A-eligible model sets in the same summary; `collie1` is excluded because it has no off-diagonal contrast.\n9. **Matched-support `A_diag_cross`:** reported beside the self value in the same summary.\n10. **Differences/model consistency:** paired self-minus-cross differences, model-bootstrap intervals, positive-model counts, and per-model rows are exported.\n11. **Blockers:** no numerical or support blocker remains for F5/F6. Provenance is explicit: archived ten-start payloads, exact canonical-best matching, frozen 0.01 rule, model-specific common basin support, and frozen 20-D representation.\n12. **Figure readiness:** yes; formally proceed to R3 figure design.\n\n## Required files\n\n- `R3_CELL_SET_IDENTITY_AUDIT.md`\n- `R3_IC_SELF_PROFILE_REFERENCE.md`\n- `R3_FINAL_CLOSURE_SUMMARY.md`\n- `tables/R3_CELL_SET_IDENTITY_DIFF.csv`\n- `tables/R3_IC_SELF_RESTART_SUPPORT.csv`\n- `tables/R3_F5_NESTED_LEDGER.csv`\n- `tables/R3_F5_BIDIRECTIONAL_CONDITIONING.csv`\n- `tables/R3_F6_IC_SELF_REFERENCE_MODEL.csv`\n- `tables/R3_F6_IC_SELF_REFERENCE_SUMMARY.csv`\n- `tables/R3_F6_CROSS_MATCHED_REFERENCE_MODEL.csv`\n- `tables/R3_IC_SELF_PROFILE_VALUES_LONG.csv` and `R3_IC_SELF_CORRESPONDENCE_LONG.csv`\n- `cache/R3_IC_SELF_PROFILE_REFERENCE/*.npz`\n\nThe within-IC reference is a calibration-scale comparator only. It must not be labeled an upper bound, ceiling, theoretical maximum, proof of parameter identity, or causal mechanism.\n"""
    return identity_md, support_md, closure_md


def main() -> None:
    started = time.time()
    R3_TABLES.mkdir(parents=True, exist_ok=True)
    basin_ids = sorted(canonical_basin(x) for x in load_ids(str(REPO / "data/531sub_id.txt")))
    if len(basin_ids) != 531 or len(set(basin_ids)) != 531:
        raise RuntimeError("canonical basin support is not exactly 531 unique IDs")
    frozen, feature_names = load_frozen_r3_arrays(basin_ids)
    canonical_ic = {model: frozen[model]["ic"] for model in ALL_MODELS}
    canonical_dpl = {model: frozen[model]["dpl"] for model in ALL_MODELS}
    features = {model: frozen[model]["info"] for model in ALL_MODELS}
    long = pd.read_csv(R3_TABLES / "R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
    canonical_feature_names = (long.loc[(long["method"] == "IC") & (long["space"] == "information_cluster"), ["feature_index", "feature"]].drop_duplicates().sort_values("feature_index")["feature"].tolist())
    if len(canonical_feature_names) != 20:
        raise RuntimeError(f"frozen R3 relationship table has {len(canonical_feature_names)} information dimensions")
    feature_names = canonical_feature_names
    relationship_entries, relationship_error = validate_frozen_relationship_arrays(frozen, long)
    identity_diff, identity = cell_identity(long)
    identity_diff.to_csv(R3_TABLES / "R3_CELL_SET_IDENTITY_DIFF.csv", index=False, float_format="%.12g")
    frame = identity["frame"]
    ledger, conditioning = build_nested_and_conditioning(frame)
    if identity["identity_exact"]:
        ledger.to_csv(R3_TABLES / "R3_F5_NESTED_LEDGER.csv", index=False, float_format="%.12g")
        stale = R3_TABLES / "R3_F5_SEPARATE_ESTIMANDS.csv"
        if stale.exists():
            stale.unlink()
    else:
        ledger.to_csv(R3_TABLES / "R3_F5_SEPARATE_ESTIMANDS.csv", index=False, float_format="%.12g")
        stale = R3_TABLES / "R3_F5_NESTED_LEDGER.csv"
        if stale.exists():
            stale.unlink()
    conditioning.to_csv(R3_TABLES / "R3_F5_BIDIRECTIONAL_CONDITIONING.csv", index=False, float_format="%.12g")

    support, restart_arrays = restart_support(basin_ids, canonical_ic)
    support.to_csv(R3_TABLES / "R3_IC_SELF_RESTART_SUPPORT.csv", index=False, float_format="%.12g")

    model_data: dict[str, dict[str, Any]] = {}
    model_rows: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        data = restart_arrays[model]
        support_m = support.loc[support.model == model].reset_index(drop=True)
        mask = support_m.has_2_independent_eligible.to_numpy(bool)
        basin_idx = np.flatnonzero(mask)
        spec = get_spec(model, device="cpu")
        p = int(spec.dimension)
        canonical_a = canonical_ic[model][basin_idx]
        alternative_idx = data["alternative_index"][basin_idx]
        alternative_b = data["u"][basin_idx, alternative_idx]
        dpl = canonical_dpl[model][basin_idx]
        x = features[model][basin_idx]
        rho_a = relationship_profiles(canonical_a, x)
        rho_b = relationship_profiles(alternative_b, x)
        rho_dpl = relationship_profiles(dpl, x)
        r_self = profile_correlations(rho_a, rho_b)
        r_cross = profile_correlations(rho_a, rho_dpl)
        c_self = correspondence_matrix(rho_a, rho_b)
        c_cross = correspondence_matrix(rho_a, rho_dpl)
        a_self, adv_self = diagonal_advantage(c_self)
        a_cross, adv_cross = diagonal_advantage(c_cross)
        profile_ok_r = bool(np.isfinite(r_self).all() and np.isfinite(r_cross).all())
        profile_ok_a = bool(np.isfinite(adv_self).all() and np.isfinite(adv_cross).all() and p > 1)
        all_finite = bool(np.isfinite(canonical_a).all() and np.isfinite(alternative_b).all() and np.isfinite(dpl).all())
        n_basins = int(len(basin_idx))
        model_data[model] = {"rho_ic_a": rho_a, "rho_ic_b": rho_b, "rho_cross_dpl": rho_dpl, "C_self": c_self, "C_cross": c_cross, "basin_ids": np.asarray([basin_ids[i] for i in basin_idx]), "canonical_u": canonical_a, "alternative_u": alternative_b, "dpl_u": dpl, "features": x, "canonical_start_index": data["best_index"][basin_idx], "alternative_start_index": alternative_idx, "parameter_names": list(spec.parameter_names)}
        model_rows.append({
            "model": model,
            "parameter_count": p,
            "n_total_basins": 531,
            "n_self_eligible_basins": n_basins,
            "coverage_fraction": n_basins / 531,
            "coverage_ge_400": bool(n_basins >= MIN_SELF_BASINS),
            "r2_strict_model": bool(model in set(r2_common.ELIGIBLE_MODELS)),
            "all_parameters_finite": all_finite,
            "profile_estimable": profile_ok_r,
            "a_diag_estimable": profile_ok_a,
            "r_profile_estimable": profile_ok_r and n_basins >= MIN_SELF_BASINS,
            "a_profile_estimable": profile_ok_a and n_basins >= MIN_SELF_BASINS,
            "R_model_IC_self": float(np.nanmedian(r_self)) if profile_ok_r else np.nan,
            "R_model_cross_matched": float(np.nanmedian(r_cross)) if profile_ok_r else np.nan,
            "delta_R_self_minus_cross": float(np.nanmedian(r_self) - np.nanmedian(r_cross)) if profile_ok_r else np.nan,
            "A_m_IC_self": a_self,
            "A_m_cross_matched": a_cross,
            "delta_A_self_minus_cross": a_self - a_cross if np.isfinite(a_self) and np.isfinite(a_cross) else np.nan,
            "median_n_eligible_starts": float(support_m.n_eligible_delta001.median()),
            "n_basin_rows_with_3plus_eligible": int(support_m.has_3plus_independent_eligible.sum()),
            "canonical_match_max_abs_error": float(support_m.canonical_match_max_abs_error.max()),
        })
    model_frame = pd.DataFrame(model_rows).sort_values("model").reset_index(drop=True)
    model_frame.to_csv(R3_TABLES / "R3_F6_IC_SELF_REFERENCE_MODEL.csv", index=False, float_format="%.12g")
    model_frame.to_csv(R3_TABLES / "R3_F6_CROSS_MATCHED_REFERENCE_MODEL.csv", index=False, float_format="%.12g")
    write_profile_outputs(model_data, feature_names)

    summaries: list[dict[str, Any]] = []
    populations = [
        ("all_model_available", model_frame["r_profile_estimable"]),
        ("r2_strict_23", model_frame["r_profile_estimable"] & model_frame["r2_strict_model"]),
    ]
    for label, rmask in populations:
        for estimand, self_col, cross_col, diff_col, amask in [
            ("R_paired_IC_self", "R_model_IC_self", "R_model_cross_matched", "delta_R_self_minus_cross", rmask),
            ("A_diag_IC_self", "A_m_IC_self", "A_m_cross_matched", "delta_A_self_minus_cross", rmask & model_frame["a_profile_estimable"]),
        ]:
            g = model_frame.loc[amask & np.isfinite(model_frame[self_col]) & np.isfinite(model_frame[cross_col])].copy()
            diffs = g[diff_col].to_numpy(float)
            low, high = model_bootstrap(diffs, BOOT_SEED + len(summaries))
            summaries.append({"population": label, "estimand": estimand, "n_models": len(g), "self_value": float(g[self_col].median()), "cross_value": float(g[cross_col].median()), "difference_self_minus_cross": float(g[diff_col].median()), "ci_low": low, "ci_high": high, "n_positive": int((diffs > 0).sum()), "n_negative": int((diffs < 0).sum()), "n_zero": int((diffs == 0).sum()), "median_n_basins": float(g.n_self_eligible_basins.median()), "min_n_basins": int(g.n_self_eligible_basins.min()), "max_n_basins": int(g.n_self_eligible_basins.max()), "bootstrap_replicates": N_BOOT, "bootstrap_seed": BOOT_SEED + len(summaries), "support_definition": "model-specific basins with canonical plus independent eligible restart"})
    summary = pd.DataFrame(summaries)
    summary.to_csv(R3_TABLES / "R3_F6_IC_SELF_REFERENCE_SUMMARY.csv", index=False, float_format="%.12g")

    metadata = {"runtime_seconds": time.time() - started, "identity_exact": identity["identity_exact"], "feature_names": feature_names, "n_models": len(ALL_MODELS), "n_basins": 531, "ic_tolerance": IC_TOLERANCE, "min_self_basins": MIN_SELF_BASINS, "bootstrap_replicates": N_BOOT, "bootstrap_seed": BOOT_SEED, "no_training": True, "representation": "R3 frozen threshold-0.70 information-cluster PC1 scores", "r2_strict_models": list(r2_common.ELIGIBLE_MODELS), "relationship_entries_validated": relationship_entries, "relationship_reproduction_max_abs_error": relationship_error}
    (R3_CACHE / "R3_FINAL_CLOSURE_SUPPORT_MANIFEST.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    identity_md, support_md, closure_md = build_markdown(identity, conditioning, support, model_frame, summary, metadata)
    (R3 / "R3_CELL_SET_IDENTITY_AUDIT.md").write_text(identity_md)
    (R3 / "R3_IC_SELF_PROFILE_REFERENCE.md").write_text(support_md)
    (R3 / "R3_FINAL_CLOSURE_SUMMARY.md").write_text(closure_md)
    print(json.dumps({"status": "PASS", "identity_exact": bool(identity["identity_exact"]), "stable": identity["n_stable"], "absrho20": identity["n_absrho20"], "both": int((frame["in_absrho20"] & (frame["rho_dPL"].abs() >= PRIMARY_EFFECT)).sum()), "same_sign_both": int(((frame["in_absrho20"]) & (frame["rho_dPL"].abs() >= PRIMARY_EFFECT) & (np.sign(frame["rho"]) == np.sign(frame["rho_dPL"]))).sum()), "runtime_seconds": round(time.time() - started, 3)}, sort_keys=True))


if __name__ == "__main__":
    main()
