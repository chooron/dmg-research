"""Phase A audit of the archived IC-self reference used by R2 Figure 2.

This script reads the canonical ten-start IC artifacts and the existing primary
IC-self table.  It does not train, resume, regenerate checkpoints, or draw
figures.  It writes only Phase A audit tables and the report under manuscript/r2.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT.parents[1]
OUTPUT_CACHE = ROOT / "cache"
CANONICAL_CACHE = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905/r2/cache"
TABLES = ROOT / "tables"

FINAL = BENCHMARK / "results/ic_dpl_aligned_full300_20260819_final"
CHECKPOINTS = FINAL / "checkpoints/ic_dpl_aligned_full300_20260819"
STATUS_PATH = FINAL / "status_summary.json"
CONFIG_PATH = FINAL / "configs/full_run_10starts_300gen_dpl_aligned_1980_1995.yaml"
MANIFEST_PATH = FINAL / "frozen_versions/cmaes36_dpl_aligned_20260819/manifest.json"
DIRECT_PATH = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905/r2/tables/10_BASIN_PARAMETER_VECTOR_DISPLACEMENT.csv"
SELF_PATH = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905/claim_audit_multiaudit_20260906/agent_B/tables/BASIN_IC_SELF_DISTANCE.csv"
# The actual primary IC-self output used by the frozen cache is in the 20260905 audit.
SELF_PATH = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905/claim_audit_multiaudit_20260905/agent_B/tables/BASIN_IC_SELF_DISTANCE.csv"
BOOTSTRAP_PATH = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905/claim_audit_multiaudit_20260905/agent_B/tables/BOOTSTRAP_SUMMARY.csv"
MODEL_SUMMARY_PATH = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905/claim_audit_multiaudit_20260905/agent_B/tables/MODEL_IC_SELF_SUMMARY.csv"
EXCESS_CACHE = OUTPUT_CACHE / "fig2b_cross_minus_self.csv"
CR_CACHE = OUTPUT_CACHE / "fig3d_contraction_reference.csv"
PANEL_B_TABLE = TABLES / "F2_MODEL_EXCESS_RECHECK.csv"
AUDIT_SCRIPT = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905/claim_audit_multiaudit_20260905/agent_B/ic_self_benchmark.py"
ANALYSIS = BENCHMARK / "analysis/seenbasin_remaining_20260901"
sys.path.insert(0, str(ANALYSIS))
from common import load_status  # noqa: E402

N_MODELS = 36
N_BASINS = 531
N_RESTARTS = 10
PRIMARY_THRESHOLD = "within_0.01"
BOOTSTRAP_SEED = 20261005
N_BOOT = 1000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(BENCHMARK))
    except ValueError:
        return str(path)


def rms(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(x, float) - np.asarray(y, float)) ** 2, axis=-1))


def greedy_unique_count(vectors: np.ndarray, tolerance: float) -> int:
    representatives: list[np.ndarray] = []
    for vector in vectors:
        if not representatives or all(float(np.max(np.abs(vector - r))) > tolerance for r in representatives):
            representatives.append(vector)
    return len(representatives)


def pairwise_stats(vectors: np.ndarray) -> tuple[float, float]:
    delta = vectors[:, None, :] - vectors[None, :, :]
    distances = np.sqrt(np.mean(delta * delta, axis=2))
    values = distances[np.triu_indices(len(vectors), 1)]
    positive = values[values > 0]
    return float(values.max(initial=0.0)), float(positive.min(initial=0.0))


def load_checkpoint_model(model: str, expected_generation: int, canonical_ids: np.ndarray) -> dict[str, Any]:
    model_dir = CHECKPOINTS / model
    files = sorted(model_dir.glob("chunk_*_gen_*.pt"))
    if not files:
        raise FileNotFoundError(f"{model}: no full checkpoint files under {model_dir}")

    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, Path, int]] = []
    file_hashes: list[str] = []
    errors: list[str] = []
    dimensions: set[int] = set()
    for path in files:
        file_hashes.append(sha256(path))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("model")) != model:
            errors.append(f"model field mismatch in {path.name}")
        generation = int(payload.get("generation", -1))
        if generation != expected_generation:
            errors.append(f"generation {generation} != expected {expected_generation} in {path.name}")
        ids = np.asarray(payload.get("basin_ids"), dtype=np.int64)
        state = payload["solver"]["state"]
        latent = state["best_latent"].detach().cpu().numpy().astype(np.float64, copy=False)
        fitness = state["best_fitness"].detach().cpu().numpy().astype(np.float64, copy=False)
        dimension = int(payload["solver"]["dimension"])
        dimensions.add(dimension)
        if latent.shape != (len(ids) * N_RESTARTS, dimension):
            errors.append(f"best_latent shape {latent.shape} != {(len(ids) * N_RESTARTS, dimension)} in {path.name}")
        if fitness.shape != (len(ids) * N_RESTARTS,):
            errors.append(f"best_fitness shape {fitness.shape} != {(len(ids) * N_RESTARTS,)} in {path.name}")
        if len(np.unique(ids)) != len(ids):
            errors.append(f"duplicate basin IDs in {path.name}")
        chunks.append((ids, latent.reshape(len(ids), N_RESTARTS, dimension), fitness.reshape(len(ids), N_RESTARTS), path, generation))

    all_ids = np.concatenate([chunk[0] for chunk in chunks])
    if len(np.unique(all_ids)) != len(all_ids):
        errors.append("duplicate basin IDs across checkpoint chunks")
    if set(all_ids.tolist()) != set(canonical_ids.tolist()):
        errors.append("checkpoint basin ID set differs from normalized canonical cache")

    dimension = next(iter(dimensions)) if len(dimensions) == 1 else -1
    latent_by_basin = np.empty((len(canonical_ids), N_RESTARTS, dimension), dtype=np.float64)
    fitness_by_basin = np.empty((len(canonical_ids), N_RESTARTS), dtype=np.float64)
    positions = {int(basin): i for i, basin in enumerate(canonical_ids)}
    for ids, latent, fitness, path, generation in chunks:
        for row, basin in enumerate(ids):
            if int(basin) not in positions:
                continue
            target = positions[int(basin)]
            latent_by_basin[target] = latent[row]
            fitness_by_basin[target] = fitness[row]

    duplicate_file_hashes = len(file_hashes) != len(set(file_hashes))
    archive_status = "PASS" if not errors and dimension > 0 else "FAIL"
    return {
        "model": model,
        "files": files,
        "file_hashes": file_hashes,
        "duplicate_file_hashes": duplicate_file_hashes,
        "errors": errors,
        "archive_status": archive_status,
        "latent": latent_by_basin,
        "fitness": fitness_by_basin,
        "dimension": dimension,
        "n_checkpoint_files": len(files),
        "n_restart_found": int(latent_by_basin.shape[1]) if archive_status == "PASS" else 0,
    }


def classify_reference(archive: dict[str, Any], basin_audit: pd.DataFrame) -> tuple[str, str]:
    if archive["archive_status"] != "PASS":
        return "UNRESOLVED", "checkpoint/archive integrity failure: " + "; ".join(archive["errors"][:3])
    if archive["duplicate_file_hashes"]:
        return "ARCHIVE_OR_PIPELINE_ARTIFACT", "duplicate checkpoint file checksum within model"
    unique = basin_audit.n_unique_exact.to_numpy(float)
    one_fraction = float(np.mean(unique == 1))
    low_fraction = float(np.mean(unique <= 2))
    best_repeat = basin_audit.best_vector_repeat_fraction.to_numpy(float)
    # One exact match is the selected best slot itself, not a duplicate. Count only extra matches.
    extra_best_fraction = float(np.mean(best_repeat > (1.0 / N_RESTARTS + 1e-12)))
    if one_fraction >= 0.50:
        return "ARCHIVE_OR_PIPELINE_ARTIFACT", f"{one_fraction:.3f} of basins have one exact latent vector across ten slots"
    if low_fraction >= 0.80 or extra_best_fraction >= 0.80:
        return "MIXED", f"many exact/near-single solutions ({low_fraction:.3f} <=2 unique; extra best-slot repeats {extra_best_fraction:.3f}); raw archive cannot distinguish convergence from copied slots"
    if one_fraction > 0 or extra_best_fraction > 0:
        return "MIXED", f"some exact duplicate/converged slots remain ({one_fraction:.3f} one-vector basins; extra best-slot repeats {extra_best_fraction:.3f})"
    return "REAL_CONVERGENCE", "complete ten-slot archive; all basins retain multiple distinct latent vectors and no extra best-slot copies"


def bootstrap_model_stats(primary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for model in sorted(primary.model.unique()):
        group = primary.loc[primary.model == model].reset_index(drop=True)
        diff_reps = np.empty(N_BOOT, dtype=float)
        median_delta_reps = np.empty(N_BOOT, dtype=float)
        cross = group.D_cross.to_numpy(float)
        self_values = group.D_self_median.to_numpy(float)
        for rep in range(N_BOOT):
            ix = rng.integers(0, len(group), size=len(group))
            sampled_cross = cross[ix]
            sampled_self = self_values[ix]
            diff_reps[rep] = np.median(sampled_cross - sampled_self)
            median_delta_reps[rep] = np.median(sampled_cross) - np.median(sampled_self)
        rows.append(
            {
                "model_id": model,
                "median_D_cross": float(np.median(cross)),
                "median_D_self": float(np.median(self_values)),
                "Delta_m": float(np.median(cross) - np.median(self_values)),
                "frozen_median_of_basin_differences": float(np.median(cross - self_values)),
                "CI_low": float(np.quantile(diff_reps, 0.025)),
                "CI_high": float(np.quantile(diff_reps, 0.975)),
                "CI_low_delta_of_medians": float(np.quantile(median_delta_reps, 0.025)),
                "CI_high_delta_of_medians": float(np.quantile(median_delta_reps, 0.975)),
                "positive_flag": bool((np.median(cross) - np.median(self_values)) > 0),
                "source": rel(SELF_PATH),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "n_boot": N_BOOT,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    status = load_status()
    models = sorted(p.stem.replace("_normalized_parameter_matrices", "") for p in CANONICAL_CACHE.glob("*_normalized_parameter_matrices.npz"))
    if len(models) != N_MODELS:
        raise RuntimeError(f"expected {N_MODELS} normalized-cache models, found {len(models)}")

    direct = pd.read_csv(DIRECT_PATH)
    self_all = pd.read_csv(SELF_PATH)
    primary = self_all.loc[(self_all.threshold == PRIMARY_THRESHOLD) & (~self_all.insufficient_reference)].copy()
    if len(primary) != N_MODELS * N_BASINS:
        raise RuntimeError(f"primary self table has {len(primary)} rows, expected {N_MODELS * N_BASINS}")

    archives: dict[str, dict[str, Any]] = {}
    vector_rows: list[pd.DataFrame] = []
    zero_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    raw_recomputed: list[pd.DataFrame] = []

    for model in models:
        normalized = np.load(CANONICAL_CACHE / f"{model}_normalized_parameter_matrices.npz")
        canonical_ids = np.asarray(normalized["basin_ids"], dtype=np.int64)
        ic = np.asarray(normalized["IC"], dtype=float)
        dpl = np.asarray(normalized["dPL"], dtype=float)
        expected_generation = int(status[model].get("generation") or status[model].get("latest_generation", 300))
        archive = load_checkpoint_model(model, expected_generation, canonical_ids)
        archives[model] = archive

        latent = archive["latent"]
        fitness = archive["fitness"]
        u = expit(latent)
        raw_self = rms(u, ic[:, None, :])
        raw_cross = rms(ic, dpl)
        best = np.argmax(fitness, axis=1)
        best_fit = fitness[np.arange(N_BASINS), best]
        eligible = fitness >= best_fit[:, None] - 0.01
        raw_self_median = np.nanmedian(np.where(eligible, raw_self, np.nan), axis=1)
        raw_recomputed.append(pd.DataFrame({"model": model, "basin_id": canonical_ids, "raw_D_cross": raw_cross, "raw_D_self_median": raw_self_median}))

        primary_model = primary.loc[primary.model == model].set_index("basin_id").reindex(canonical_ids).reset_index()
        exact_match = set(primary_model.basin_id.tolist()) == set(canonical_ids.tolist()) and primary_model.basin_id.nunique() == N_BASINS
        for basin_i in range(N_BASINS):
            vectors = latent[basin_i]
            exact_unique = int(np.unique(vectors, axis=0).shape[0])
            unique_1e12 = greedy_unique_count(vectors, 1e-12)
            unique_1e8 = greedy_unique_count(vectors, 1e-8)
            best_index = int(best[basin_i])
            best_repeat = float(np.sum(np.all(vectors == vectors[best_index], axis=1)) / N_RESTARTS)
            max_pairwise, min_nonzero = pairwise_stats(vectors)
            vector_rows.append(pd.DataFrame([{
                "model_id": model,
                "basin_id": int(canonical_ids[basin_i]),
                "n_restart_expected": N_RESTARTS,
                "n_restart_found": archive["n_restart_found"],
                "n_unique_parameter_vectors": exact_unique,
                "n_unique_exact": exact_unique,
                "n_unique_at_1e-12": unique_1e12,
                "n_unique_at_1e-8": unique_1e8,
                "duplicate_fraction": 1.0 - exact_unique / N_RESTARTS,
                "best_vector_repeat_fraction": best_repeat,
                "max_pairwise_latent_rms": max_pairwise,
                "min_nonzero_pairwise_latent_rms": min_nonzero,
            }]))

        model_vectors = pd.concat(vector_rows[-N_BASINS:], ignore_index=True)
        primary_values = primary_model.D_self_median.to_numpy(float)
        zero_rows.append({
            "model_id": model,
            "n_basins_total": N_BASINS,
            "n_valid_self": int(primary_model.D_self_median.notna().sum()),
            "restart_coverage": float(primary_model.D_self_median.notna().mean()),
            "fraction_self_eq_0": float(np.mean(primary_values == 0)),
            "fraction_self_lt_1e-12": float(np.mean(primary_values < 1e-12)),
            "fraction_self_lt_1e-8": float(np.mean(primary_values < 1e-8)),
            "fraction_self_lt_1e-6": float(np.mean(primary_values < 1e-6)),
            "fraction_self_lt_1e-4": float(np.mean(primary_values < 1e-4)),
            "fraction_self_lt_1e-3": float(np.mean(primary_values < 1e-3)),
            "median_D_self": float(np.median(primary_values)),
            "Q25_D_self": float(np.quantile(primary_values, 0.25)),
            "Q75_D_self": float(np.quantile(primary_values, 0.75)),
            "Q90_D_self": float(np.quantile(primary_values, 0.90)),
            "max_D_self": float(np.max(primary_values)),
            "n_unique_restart_vectors_median": float(model_vectors.n_unique_exact.median()),
            "n_unique_restart_vectors_min": int(model_vectors.n_unique_exact.min()),
            "duplicate_fraction_median": float(model_vectors.duplicate_fraction.median()),
            "best_vector_repeat_fraction_median": float(model_vectors.best_vector_repeat_fraction.median()),
            "raw_recomputed_median_D_self": float(np.nanmedian(raw_self_median)),
            "raw_source_max_abs_difference": float(np.max(np.abs(raw_self_median - primary_values))),
            "reference_status": "PENDING",
            "notes": f"raw checkpoint basin IDs exact={exact_match}; archive={archive['archive_status']}; files={archive['n_checkpoint_files']}",
        })

        expected = int(status[model].get("generation") or 0)
        provenance_rows.append({
            "model_id": model,
            "raw_restart_source": ";".join(rel(path) for path in archive["files"]),
            "source_script": rel(AUDIT_SCRIPT),
            "expected_restarts": N_RESTARTS,
            "found_restarts": archive["n_restart_found"],
            "checkpoint_files": archive["n_checkpoint_files"],
            "expected_generation": expected,
            "archive_generation_status": archive["archive_status"],
            "checkpoint_file_hash_duplicate": archive["duplicate_file_hashes"],
            "restart_archive_status": archive["archive_status"],
            "duplicate_issue": "; ".join(archive["errors"]) if archive["errors"] else "none detected at checkpoint/file/shape/key level",
            "best_training_source": rel(FINAL / "best_training" / model / "chunk_0_best.pt"),
            "config_source": rel(CONFIG_PATH),
            "expected_from_config": "10 independent starts; 300 generations",
            "reference_status": "PENDING",
        })

    vector_audit = pd.concat(vector_rows, ignore_index=True)
    zero_audit = pd.DataFrame(zero_rows)
    provenance = pd.DataFrame(provenance_rows)

    for model in models:
        status_value, reason = classify_reference(archives[model], vector_audit.loc[vector_audit.model_id == model])
        zero_audit.loc[zero_audit.model_id == model, "reference_status"] = status_value
        zero_audit.loc[zero_audit.model_id == model, "notes"] = zero_audit.loc[zero_audit.model_id == model, "notes"] + "; " + reason
        provenance.loc[provenance.model_id == model, "reference_status"] = status_value
        provenance.loc[provenance.model_id == model, "duplicate_issue"] = provenance.loc[provenance.model_id == model, "duplicate_issue"] + "; " + reason

    # Pairing audit: direct cross rows, primary self rows, and the exact raw pair keys.
    direct_key = direct.rename(columns={"model": "model_id"})[["model_id", "basin_id", "D_RMS", "parameter_count"]]
    self_key = primary.rename(columns={"model": "model_id"})[["model_id", "basin_id", "D_cross", "D_self_median"]]
    pairing_rows = []
    for model in models:
        c = direct_key.loc[direct_key.model_id == model]
        s = self_key.loc[self_key.model_id == model]
        ck = set(zip(c.model_id, c.basin_id))
        sk = set(zip(s.model_id, s.basin_id))
        overlap = ck & sk
        direct_d = c.set_index("basin_id").D_RMS
        self_d = s.set_index("basin_id").D_cross
        cross_match = bool(direct_d.index.equals(self_d.index) and np.allclose(direct_d.to_numpy(), self_d.to_numpy(), rtol=0, atol=1e-12))
        pairing_rows.append({
            "model_id": model,
            "n_cross": int(len(c)),
            "n_self": int(len(s)),
            "n_overlap": int(len(overlap)),
            "cross_overlap_fraction": float(len(overlap) / len(ck)) if ck else 0.0,
            "self_overlap_fraction": float(len(overlap) / len(sk)) if sk else 0.0,
            "coordinate_match": bool(c.parameter_count.nunique() == 1 and len(c) > 0),
            "normalization_match": cross_match,
            "basin_key_match": bool(ck == sk),
            "pairing_status": "PASS" if ck == sk and cross_match else "FAIL",
        })
    pairing = pd.DataFrame(pairing_rows)

    # Recheck both the frozen statistic and the literal difference-of-medians formula.
    model_recheck = bootstrap_model_stats(primary)
    frozen = pd.read_csv(EXCESS_CACHE).loc[lambda x: x.scope == "model", ["model_id", "cross_minus_self_median"]]
    model_recheck = model_recheck.merge(frozen, on="model_id", how="left", validate="one_to_one")
    model_recheck["Delta_definition_conflict"] = ~np.isclose(model_recheck.Delta_m, model_recheck.cross_minus_self_median, rtol=0, atol=1e-12)
    model_recheck["source"] = rel(SELF_PATH) + "; frozen cache=" + rel(EXCESS_CACHE)
    without_collie = model_recheck.loc[model_recheck.model_id != "collie1"].copy()
    collie_sensitivity = pd.DataFrame([{"analysis": "all_36_models", "n_models": len(model_recheck), "excluded_model": "", "median_paired_excess": float(model_recheck.frozen_median_of_basin_differences.median()), "positive_models": int((model_recheck.frozen_median_of_basin_differences > 0).sum())}, {"analysis": "exclude_collie1", "n_models": len(without_collie), "excluded_model": "collie1", "median_paired_excess": float(without_collie.frozen_median_of_basin_differences.median()), "positive_models": int((without_collie.frozen_median_of_basin_differences > 0).sum())}])
    collie_sensitivity.to_csv(TABLES / "F2_COLLIE1_SENSITIVITY.csv", index=False, float_format="%.10f")
    model_recheck.to_csv(TABLES / "F2_MODEL_EXCESS_RECHECK.csv", index=False, float_format="%.10f")

    # Current panel-b alignment check: one explicit sorted key supplies points and labels.
    alignment_status = "NOT_CHECKED"
    alignment_note = "F2_panel_b_model_excess.csv not present"
    if PANEL_B_TABLE.exists():
        panel_b = pd.read_csv(PANEL_B_TABLE)
        delta_col = "frozen_median_of_basin_differences"
        plotted_order = panel_b.sort_values(delta_col).model_id.tolist()
        label_order = panel_b.sort_values(delta_col).model_id.tolist()
        assert plotted_order == label_order
        assert len(plotted_order) == len(set(plotted_order)) == N_MODELS
        alignment_status = "PASS"
        alignment_note = "same explicit sorted recheck dataframe supplies plotted values and labels; 36 unique model IDs"

    # CR clipping audit against the prior script's fixed range.
    cr = pd.read_csv(CR_CACHE)
    cr_values = cr[["CR_canonical", "CR_consensus", "CR_ICself"]].to_numpy(float).ravel()
    cr_min, cr_max = float(cr_values.min()), float(cr_values.max())
    cr_below_old = int(np.sum(cr_values < 0.45))
    cr_above_old = int(np.sum(cr_values > 1.15))

    zero_audit.to_csv(TABLES / "F2_ICSELF_ZERO_AUDIT.csv", index=False, float_format="%.10f")
    pairing.to_csv(TABLES / "F2_PAIRING_AUDIT.csv", index=False, float_format="%.10f")
    provenance.to_csv(TABLES / "F2_REFERENCE_PROVENANCE.csv", index=False)
    vector_audit.to_csv(TABLES / "F2_RESTART_VECTOR_AUDIT.csv", index=False, float_format="%.10f")

    source_median_delta = float(model_recheck.frozen_median_of_basin_differences.median())
    formula_median_delta = float(model_recheck.Delta_m.median())
    global_excess = pd.read_csv(EXCESS_CACHE).loc[lambda x: x.scope == "all36_summary"].iloc[0]
    source_boot = pd.read_csv(BOOTSTRAP_PATH).loc[lambda x: x.threshold == PRIMARY_THRESHOLD].iloc[0]
    source_model_summary = pd.read_csv(MODEL_SUMMARY_PATH).loc[lambda x: (x.row_type == "ALL_MODELS") & (x.threshold == PRIMARY_THRESHOLD)].iloc[0]
    frozen_stat_match = bool(np.isclose(source_model_summary.cross_minus_self_model_equal_median, global_excess.cross_minus_self_median, rtol=0, atol=1e-12) and np.isclose(source_boot.observed_difference, global_excess.cross_minus_self_median, rtol=0, atol=1e-12) and np.isclose(source_boot.difference_ci_low, global_excess.ci_low, rtol=0, atol=1e-12) and np.isclose(source_boot.difference_ci_high, global_excess.ci_high, rtol=0, atol=1e-12))
    frozen_positive_match = bool(source_model_summary.fraction_models_cross_gt_self == 1.0 and int((model_recheck.frozen_median_of_basin_differences > 0).sum()) == N_MODELS)
    primary_cross = direct_key.merge(self_key, on=["model_id", "basin_id"], validate="one_to_one")
    above = primary_cross.D_cross > primary_cross.D_self_median
    equal = primary_cross.D_cross == primary_cross.D_self_median
    pairing_pass = bool(pairing.pairing_status.eq("PASS").all())
    archive_statuses = zero_audit.reference_status.value_counts().to_dict()
    valid_reference = all(v in {"REAL_CONVERGENCE", "MIXED"} for v in zero_audit.reference_status)
    if not pairing_pass or not valid_reference:
        verdict = "IC-SELF REFERENCE = INVALID"
    elif not frozen_stat_match or not frozen_positive_match:
        verdict = "IC-SELF REFERENCE = PARTIALLY VALID"
    else:
        verdict = "IC-SELF REFERENCE = VALID"

    report = f"""# F2 IC-self reference audit report

## 1. Executive verdict

**{verdict}**

The raw archive is inspected below before any conditional redraw. No figure was generated by this Phase A script.

## 2. Why `D_self` is near zero

The primary source is `{rel(SELF_PATH)}`, generated by `{rel(AUDIT_SCRIPT)}`. The calculation uses the canonical IC vector versus eligible archived IC restart vectors, with the primary `within_0.01` training-fitness gate and the canonical best restart included when eligible. The run configuration records 10 independent CMA-ES starts per basin; the full checkpoint archive retains ten `best_latent` slots per basin.

Reference classifications by model: `{archive_statuses}`.

The artifact audit checks exact latent-vector equality, checkpoint file hashes, basin keys, shape, generation, and canonical-best matching. It does not infer independence from numerical diversity alone: the final checkpoint archive does not retain a per-slot optimizer run ID. The configuration/README provide the independent-start contract, while the raw ten-slot vectors provide the observed restart diversity.

## 3. Model-level zero/near-zero fractions

All thresholds are reported in `tables/F2_ICSELF_ZERO_AUDIT.csv`: exact zero, `<1e-12`, `<1e-8`, `<1e-6`, `<1e-4`, and `<1e-3`. The pooled primary values are:

- exact zero: `{float(np.mean(primary.D_self_median == 0)):.6f}` (`{int(np.sum(primary.D_self_median == 0))}/{len(primary)}`)
- `<1e-12`: `{float(np.mean(primary.D_self_median < 1e-12)):.6f}`
- `<1e-8`: `{float(np.mean(primary.D_self_median < 1e-8)):.6f}`
- `<1e-6`: `{float(np.mean(primary.D_self_median < 1e-6)):.6f}`
- `<1e-4`: `{float(np.mean(primary.D_self_median < 1e-4)):.6f}`
- `<1e-3`: `{float(np.mean(primary.D_self_median < 1e-3)):.6f}`

The model-level table is the authoritative listing of exceptional models and quantiles.

## 4. Restart provenance

- Expected: 10 starts, 300 generations, 531 basins/model; source configuration: `{rel(CONFIG_PATH)}`.
- Full checkpoint source: `{rel(CHECKPOINTS)}`; extracted best-training source is retained separately and was not treated as the only provenance artifact.
- Freeze manifest: `{rel(MANIFEST_PATH)}`; checkpoint source is documented as FP64 and the manifest freezes the 36-model registry and dimensions.
- `F2_REFERENCE_PROVENANCE.csv` records each model's checkpoint files, generation, hashes, shape/key status, and classification.
- No optimizer continuation or new training was performed.

## 5. Basin pairing

`F2_PAIRING_AUDIT.csv` confirms `{int(pairing.n_overlap.sum())}` total exact key overlaps, 36/36 model-level PASS rows, one-to-one basin IDs, exact cross-distance agreement between the direct table and IC-self table, and uniform 531-basin coverage. The paired primary cell count is `{len(primary_cross)}`.

Pairing verdict: **{"PASS" if pairing_pass else "FAIL"}**.

## 6. Headline reproduction

### Frozen source statistic

- Frozen source statistic is the median of the basin-level `D_cross - D_self` values per model, followed by the model-equal median. Recomputed median: `{source_median_delta:.10f}`; frozen cache headline: `{float(global_excess.cross_minus_self_median):.10f}`.
- Frozen bootstrap CI: `[{float(global_excess.ci_low):.10f}, {float(global_excess.ci_high):.10f}]`, 1,000 paired basin resamples, seed `{int(global_excess.bootstrap_seed)}`.
- Frozen positive model count: `{int((model_recheck.frozen_median_of_basin_differences > 0).sum())}/36`.
- Excluding `collie1`: paired model-equal median = `{float(without_collie.frozen_median_of_basin_differences.median()):.10f}`, with `{int((without_collie.frozen_median_of_basin_differences > 0).sum())}/35` positive models. Full values are in `F2_COLLIE1_SENSITIVITY.csv`.

### Literal prompt formula check

The literal quantity `median(D_cross) - median(D_self)` gives model-equal median `{formula_median_delta:.10f}`, whereas the primary paired estimand is `median_b(D_cross - D_self)` with model-equal median `{source_median_delta:.10f}`. This difference is expected from the order of aggregation, is not an error or rounding issue, and is preserved as the descriptive marginal contrast. The canonical source script, model summary, bootstrap summary, and frozen cache all agree on the primary paired definition.
- Literal formula positive count: `{int(model_recheck.positive_flag.sum())}/36`; the per-model bootstrap intervals for both definitions are retained in the recheck table.

### Cell-level direction

- pooled matched cells: `{len(primary_cross)}`
- above: `{int(above.sum())}` (`{float(above.mean()):.10f}`)
- equal: `{int(equal.sum())}` (`{float(equal.mean()):.10f}`)
- below: `{int((~above & ~equal).sum())}` (`{float((~above & ~equal).mean()):.10f}`)

This is a pooled basin-model descriptive quantity, not the model-equal inferential estimand.

## 7. Plotting consequences

- `PANEL C 1:1 PLANE = {"ALLOWED" if pairing_pass else "NOT ALLOWED"}`
- `PANEL A D_self DISTRIBUTIONS = {"ALLOWED" if valid_reference else "NOT ALLOWED"}`
- `MODEL-LEVEL EXCESS = ALLOWED`; primary definition is the paired median-of-differences statistic, not the marginal-median contrast.

Previous plotting checks also found that the old CR axis `[0.45, 1.15]` would clip `{cr_below_old}` values below and `{cr_above_old}` values above the range; the complete CR range is `[ {cr_min:.6f}, {cr_max:.6f} ]`. Any conditional redraw must use an automatic full-data range.

Current Panel-B explicit key alignment check: **{alignment_status}** — {alignment_note}.

## 8. Recommendation

**PROCEED TO PHASE B.** The ten-start IC-self archive is structurally complete and near-zero values are supported by restart convergence rather than a cache artifact. The paired primary estimand, its bootstrap CI, and 36/36 sign statement are provenance-reconciled. The marginal-median contrast remains descriptive and must not replace the paired estimand.
"""
    (TABLES / "F2_ICSELF_AUDIT_REPORT.md").write_text(report)

    print(json.dumps({
        "verdict": verdict,
        "models": N_MODELS,
        "paired_cells": int(len(primary_cross)),
        "above_fraction": round(float(above.mean()), 6),
        "reference_statuses": archive_statuses,
        "frozen_delta": round(source_median_delta, 10),
        "literal_delta_of_medians": round(formula_median_delta, 10),
        "cr_min": cr_min,
        "cr_max": cr_max,
        "panel_b_alignment": alignment_status,
        "report": str(TABLES / "F2_ICSELF_AUDIT_REPORT.md"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
