"""Generating-truth mapping ``theta* = g*(A)`` for the XAJ-CN structure.

Design (frozen constraints):

- input is the full 35-dimension CAMELS attribute vector exactly as the dPL
  student receives it (``robust_normalize`` median/IQR clip ±5, computed over
  the full 531-basin matrix), including ``frac_snow``;
- the mapping is deterministic, non-neural, and from a different family than
  the dPL MLP (regularized linear regression on PCA scores);
- the parameter manifold is anchored on the repository's existing 531-basin
  XAJ-CN IC-CMA-ES parameter field (best train-KGE restart per basin, the
  canonical R1 selection rule) but the field is *not* copied basin-by-basin:
  g* is an explicit, reproducible attribute-to-parameter rule;
- no random parameter residual is added: ``theta* = g*(A)`` exactly.

Procedure:

1. physical -> unit-normalized ``z`` in [0, 1] (linear map, XAJ-CN specs);
2. standardize ``z`` over basins and take the SVD; choose rank ``K`` by the
   smallest number of components reaching 95% cumulative explained variance
   (data-driven, not tuned to any snow gradient);
3. project basins onto the first ``K`` right singular vectors -> scores;
4. robust-normalized attributes -> scores via ridge regression; the ridge
   penalty ``alpha`` is chosen by 5-fold basin CV on a fixed grid (seed
   ``20260730``, the project's standard analysis seed);
5. ``g*(A) = mean_z + scores_hat @ V_K^T`` mapped back to physical bounds,
   clipped to the legal parameter bounds (clip counts are reported).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    COMMON_XAJ,
    DEFAULT_RESULTS_ROOT,
    IC_RAW_SUBDIRS,
    IC_RESULT_ROOTS,
    git_commit,
    sha256_file,
    write_json,
)

CN_PARAM_NAMES = ("cn_ctg", "cn_kf") + tuple(COMMON_XAJ)  # XAJ_CN spec order

EXPLAINED_VARIANCE_FRACTION = 0.95
RIDGE_ALPHA_GRID = np.logspace(-4.0, 4.0, 17)
CV_SEED = 20260730
CV_FOLDS = 5


@dataclass(frozen=True)
class GStarFit:
    z_mean: np.ndarray
    z_scale: np.ndarray
    V_k: np.ndarray          # [17, K] right singular vectors (z-space)
    ridge_coef: np.ndarray   # [35, K] attribute -> score coefficients
    intercept: np.ndarray    # [K]
    alpha: float
    k: int
    explained_variance: np.ndarray          # per-component
    cumulative_variance: np.ndarray
    cv_r2_per_component: np.ndarray
    cv_r2_total: float
    attrs_stats: dict[str, np.ndarray]      # robust_normalize stats (full 531)


def load_cn_ic_field(results_root: Path = DEFAULT_RESULTS_ROOT) -> dict[str, Any]:
    """Best train-KGE restart per basin from the 531 XAJ-CN IC-CMA-ES field."""
    raw_dir = results_root / IC_RESULT_ROOTS["XAJ_CN"] / "raw" / IC_RAW_SUBDIRS["XAJ_CN"]
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"CN-IC raw result directory not found: {raw_dir}")
    records: dict[str, list[tuple[float, int, dict[str, Any]]]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text())
        basin = str(data["basin_id"]).zfill(8)
        train_kge = float(data.get("train_metrics", {}).get("kge", np.nan))
        if data.get("status") == "complete" and np.isfinite(train_kge):
            records.setdefault(basin, []).append((train_kge, int(data["start"]), data))
    basins = sorted(records)
    if len(basins) != 531:
        raise ValueError(f"expected 531 CN-IC basins, found {len(basins)}")
    params: list[np.ndarray] = []
    restart_info: list[dict[str, Any]] = []
    names: tuple[str, ...] | None = None
    for basin in basins:
        candidates = records[basin]
        candidates.sort(key=lambda item: (-item[0], item[1]))  # best KGE, lowest start
        _kge, start, data = candidates[0]
        if names is None:
            names = tuple(data["parameter_names"])
        if tuple(data["parameter_names"]) != names:
            raise ValueError(f"CN-IC parameter-name mismatch at basin {basin}")
        params.append(np.asarray(data["parameters"], dtype=np.float64))
        restart_info.append({"basin_id": basin, "start": start, "train_kge": _kge,
                             "source_file": str(path)})
    return {
        "basin_ids": basins,
        "parameter_names": list(names),
        "parameters": np.stack(params),
        "restart_info": restart_info,
    }


def physical_to_z(physical: np.ndarray, names: tuple[str, ...], specs: dict[str, dict[str, Any]]) -> np.ndarray:
    lower = np.asarray([specs[n]["lower"] for n in names], dtype=np.float64)
    upper = np.asarray([specs[n]["upper"] for n in names], dtype=np.float64)
    return (np.asarray(physical, dtype=np.float64) - lower) / (upper - lower)


def z_to_physical(z: np.ndarray, names: tuple[str, ...], specs: dict[str, dict[str, Any]]) -> np.ndarray:
    lower = np.asarray([specs[n]["lower"] for n in names], dtype=np.float64)
    upper = np.asarray([specs[n]["upper"] for n in names], dtype=np.float64)
    return lower + np.asarray(z, dtype=np.float64) * (upper - lower)


def fit_g_star(attributes_raw: np.ndarray, z_cn: np.ndarray, *,
               n_components: int | None = None,
               variance_fraction: float = EXPLAINED_VARIANCE_FRACTION,
               alpha_grid: np.ndarray = RIDGE_ALPHA_GRID,
               cv_seed: int = CV_SEED, cv_folds: int = CV_FOLDS) -> GStarFit:
    """Fit the deterministic attribute->CN-parameter mapping on the 531 field."""
    from training.dpl.run_dpl_model import robust_normalize

    if attributes_raw.ndim != 2 or z_cn.ndim != 2:
        raise ValueError("attributes_raw [n,35] and z_cn [n,17] required")
    n, n_attr = attributes_raw.shape
    if z_cn.shape[0] != n:
        raise ValueError("attribute and parameter matrices must share the basin axis")
    if n_attr != 35:
        raise ValueError(f"expected the full 35-dimension attribute vector, got {n_attr}")

    attrs_norm, attrs_stats = robust_normalize(np.asarray(attributes_raw, dtype=np.float32))
    X = attrs_norm.astype(np.float64)
    X = np.column_stack([np.ones(n), X])  # affine

    z_mean = z_cn.mean(axis=0)
    z_scale = z_cn.std(axis=0)
    z_scale = np.where(z_scale < 1e-12, 1.0, z_scale)
    Zc = (z_cn - z_mean) / z_scale

    # SVD of the standardized parameter field (full 17-dim space).
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    explained = S ** 2 / np.maximum((S ** 2).sum(), 1e-300)
    cumulative = np.cumsum(explained)
    if n_components is None:
        k = int(np.searchsorted(cumulative, variance_fraction) + 1)
        k = min(k, Zc.shape[1])
    else:
        k = int(n_components)
        if not 1 <= k <= Zc.shape[1]:
            raise ValueError(f"n_components must be within [1, {Zc.shape[1]}]")
    V_k = Vt[:k].T  # [17, K]
    scores = Zc @ V_k  # [n, K]

    # 5-fold basin CV over the ridge penalty grid (fixed seed; basins are the
    # resampling unit, matching the project's basin-as-unit convention).
    rng = np.random.default_rng(cv_seed)
    fold_index = np.zeros(n, dtype=int)
    for f in range(cv_folds):
        fold_index[rng.permutation(n)[f::cv_folds]] = f
    best_alpha, best_mse = None, np.inf
    cv_mse: dict[float, float] = {}
    for alpha in alpha_grid:
        errors = np.zeros((n, k))
        for f in range(cv_folds):
            tr = fold_index != f
            te = fold_index == f
            Xtr, Xte = X[tr], X[te]
            Str = scores[tr]
            # closed-form ridge for the affine design
            gram = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
            gram[0, 0] -= alpha  # do not penalize the intercept
            coef = np.linalg.solve(gram, Xtr.T @ Str)
            errors[te] = Xte @ coef - scores[te]
        mse = float(np.mean(errors ** 2))
        cv_mse[float(alpha)] = mse
        if mse < best_mse:
            best_mse, best_alpha = mse, float(alpha)
    alpha = float(best_alpha)

    gram = X.T @ X + alpha * np.eye(X.shape[1])
    gram[0, 0] -= alpha
    coef = np.linalg.solve(gram, X.T @ scores)  # [36, K]
    ridge_coef, intercept = coef[1:], coef[0]

    # In-sample reconstruction diagnostics (mapping quality, not a target).
    scores_hat = X @ coef
    residual = scores - scores_hat
    ss_total = ((scores - scores.mean(axis=0)) ** 2).sum(axis=0)
    cv_r2_per_component = 1.0 - np.array([cv_mse[alpha] * n / ss_total[i]
                                          if ss_total[i] > 0 else np.nan
                                          for i in range(k)])
    total_var = float((Zc ** 2).sum())
    cv_r2_total = float(1.0 - (cv_mse[alpha] * n) / max(total_var, 1e-300))

    return GStarFit(
        z_mean=z_mean, z_scale=z_scale, V_k=V_k,
        ridge_coef=ridge_coef, intercept=intercept, alpha=alpha, k=int(k),
        explained_variance=explained, cumulative_variance=cumulative,
        cv_r2_per_component=cv_r2_per_component, cv_r2_total=cv_r2_total,
        attrs_stats={key: np.asarray(value) for key, value in attrs_stats.items()},
    )


def g_star_apply(fit: GStarFit, attributes_raw: np.ndarray) -> np.ndarray:
    """theta* (physical, [n, 17]) from raw attributes via the frozen fit."""
    from training.dpl.run_dpl_model import robust_normalize

    attrs_norm, _stats = robust_normalize(np.asarray(attributes_raw, dtype=np.float32))
    X = np.column_stack([np.ones(attrs_norm.shape[0]), attrs_norm.astype(np.float64)])
    scores_hat = X @ np.concatenate([fit.intercept[None, :], fit.ridge_coef], axis=0)
    z_star = fit.z_mean + (scores_hat @ fit.V_k.T) * fit.z_scale
    return z_star


def clip_to_bounds(physical: np.ndarray, names: tuple[str, ...],
                   specs: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([specs[n]["lower"] for n in names], dtype=np.float64)
    upper = np.asarray([specs[n]["upper"] for n in names], dtype=np.float64)
    clipped = np.clip(physical, lower, upper)
    mask = ~np.isclose(physical, clipped, atol=0.0, rtol=0.0)
    return clipped, mask


def parameter_diagnostics(physical: np.ndarray, names: tuple[str, ...],
                          specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lower = np.asarray([specs[n]["lower"] for n in names], dtype=np.float64)
    upper = np.asarray([specs[n]["upper"] for n in names], dtype=np.float64)
    z = (physical - lower) / (upper - lower)
    rows = {}
    for i, name in enumerate(names):
        rows[name] = {
            "min": float(physical[:, i].min()), "max": float(physical[:, i].max()),
            "mean": float(physical[:, i].mean()), "std": float(physical[:, i].std()),
            "median": float(np.median(physical[:, i])),
            "min_z": float(z[:, i].min()), "max_z": float(z[:, i].max()),
            "mean_z": float(z[:, i].mean()),
            "frac_at_lower": float((physical[:, i] <= lower[i] + 1e-9).mean()),
            "frac_at_upper": float((physical[:, i] >= upper[i] - 1e-9).mean()),
            "mean_relative_boundary_distance": float(
                np.minimum(z[:, i], 1.0 - z[:, i]).mean()
            ),
        }
    return rows


def build_and_save_truth(
    bundle, cn_field: dict[str, Any], output_dir: Path,
    project_root: Path, results_root: Path, data_root: Path,
    n_components: int | None = None,
) -> tuple[GStarFit, np.ndarray, np.ndarray]:
    """Fit g*, compute theta*, save parameters + diagnostics + manifest.

    Returns (fit, theta_star_physical [531,17], clip_mask [531,17]).
    """
    from models.parameter_specs import XAJ_CN_PARAM_SPECS

    specs = XAJ_CN_PARAM_SPECS
    names = tuple(specs)
    z_cn = physical_to_z(cn_field["parameters"], names, specs)
    fit = fit_g_star(bundle.raw_attributes, z_cn, n_components=n_components)
    z_star = g_star_apply(fit, bundle.raw_attributes)
    physical = z_to_physical(z_star, names, specs)
    physical_clipped, clip_mask = clip_to_bounds(physical, names, specs)
    n_clipped = int(clip_mask.sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "theta_star.npz",
        parameters=physical_clipped,
        parameters_unclipped=physical,
        z_normalized=z_star,
        clip_mask=clip_mask,
        parameter_names=np.asarray(names),
        basin_ids=np.asarray(bundle.basin_ids),
    )
    diag = {
        "parameter_diagnostics_g_star": parameter_diagnostics(physical_clipped, names, specs),
        "parameter_diagnostics_cn_ic_reference": parameter_diagnostics(
            cn_field["parameters"], names, specs
        ),
        "low_rank": {
            "n_components_k": fit.k,
            "variance_fraction_target": EXPLAINED_VARIANCE_FRACTION,
            "explained_variance_per_component": fit.explained_variance.tolist(),
            "cumulative_variance": fit.cumulative_variance.tolist(),
            "reconstruction_var_explained": float(fit.cumulative_variance[fit.k - 1]),
        },
        "attribute_mapping": {
            "family": "regularized_linear_ridge_on_pca_scores",
            "ridge_alpha": fit.alpha,
            "ridge_alpha_grid": alpha_grid_str(),
            "cv_folds": CV_FOLDS,
            "cv_seed": CV_SEED,
            "cv_r2_per_component": fit.cv_r2_per_component.tolist(),
            "cv_r2_total": fit.cv_r2_total,
            "attribute_input": "full 35-dimension CAMELS vector, robust median/IQR normalize clip +/-5 over all 531 basins",
            "frac_snow_included": True,
        },
        "boundary_handling": {
            "n_clipped_entries": n_clipped,
            "n_total_entries": int(clip_mask.size),
            "n_basins_with_any_clip": int(np.any(clip_mask, axis=1).sum()),
            "per_parameter_clip_counts": {
                name: int(clip_mask[:, i].sum()) for i, name in enumerate(names)
            },
        },
        "theta_star_equals_g_star": True,
        "random_residual": None,
    }
    write_json(output_dir / "gstar_diagnostics.json", diag)
    manifest = {
        "protocol": "r3_synthetic_truth_v1",
        "generating_structure": "XAJ_CN",
        "created_at": _utcnow(),
        "code": git_commit(project_root),
        "source_files": {
            name: sha256_file(Path(__file__).resolve().parent / name)
            for name in ("truth_generator.py", "recorded_forward.py", "common.py")
        },
        "inputs": {
            "basin_list_path": str(data_root / "531sub_id.txt"),
            "basin_list_fingerprint": sha256_file(data_root / "531sub_id.txt"),
            "dataset_path": str(data_root / "camels_dataset"),
            "dataset_fingerprint": sha256_file(data_root / "camels_dataset"),
            "gage_ids_fingerprint": sha256_file(data_root / "gage_id.npy"),
            "dates_fingerprint": sha256_file(data_root / "camels_dates.npy"),
            "n_basins": len(bundle.basin_ids),
            "attribute_definition": "camels_dataset attributes[:, :35]; order documented in ablation/ic_core/data_adapter.py ATTRIBUTE_NAMES",
            "frac_snow_index": 3,
        },
        "cn_ic_reference_field": {
            "source": str(results_root / IC_RESULT_ROOTS["XAJ_CN"]),
            "restart_selection": "best train-period KGE restart per basin (lowest start breaks ties); R1 canonical rule",
            "parameter_names": names,
        },
        "g_star": {
            "definition": (
                "z = (p - lower)/(upper - lower); standardize z; SVD; "
                "scores = Zc @ V_K; ridge(attributes -> scores); "
                "theta* = clip(mean + (scores_hat @ V_K^T) * scale); "
                "no random residual"
            ),
            "k": fit.k,
            "ridge_alpha": fit.alpha,
        },
        "parameter_bounds": {
            name: {"lower": specs[name]["lower"], "upper": specs[name]["upper"]}
            for name in names
        },
        "randomness": {"cv_split_seed": CV_SEED, "final_mapping_deterministic": True},
    }
    write_json(output_dir / "gstar_manifest.json", manifest)
    return fit, physical_clipped, clip_mask


def alpha_grid_str() -> list[float]:
    return [float(v) for v in RIDGE_ALPHA_GRID]


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
