"""Post-processing for dual-evidence benchmark outputs.

Reads calibration and parameter-learning results, then produces:
- cross_seed_stability_table       : per-basin/model variance of metrics across seeds
- cross_loss_stability_table       : per-basin/model variance across KGE vs KGE_LOG
- dominant_control_table           : which objective / evidence type drives recommendation
- dual_evidence_summary            : HH/HL/LH/LL classification per basin × model
- recommendation_readiness_table   : readiness score per model
- performance_equivalent_vs_recommendation_ready_table : comparison of top-N models

Usage
-----
from benchmark.postprocess import run_postprocess
run_postprocess(output_dir="outputs")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Thresholds for HH/HL/LH/LL classification
_KGE_HIGH_THRESHOLD = 0.5       # calibration KGE >= this → "high" calibration evidence
_PL_KGE_HIGH_THRESHOLD = 0.4    # param-learning KGE >= this → "high" PL evidence

# Cross-seed stability: CV threshold for "stable"
_CV_STABLE_THRESHOLD = 0.10

# Recommendation readiness thresholds
_READY_BASIN_FRACTION = 0.6     # fraction of basins that must be HH or LH
_READY_STABILITY_CV = 0.15      # max acceptable seed CV for a basin to be "stable"


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_calibration_results(calib_root: Path) -> pd.DataFrame:
    """Collect all calibration results.csv into one DataFrame."""
    frames = []
    for path in sorted(calib_root.rglob("results.csv")):
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
    if not frames:
        log.warning("No calibration results found under %s", calib_root)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_param_learning_results(pl_root: Path) -> pd.DataFrame:
    """Collect all parameter learning per_basin_results.parquet into one DataFrame."""
    frames = []
    for path in sorted(pl_root.rglob("per_basin_results.parquet")):
        try:
            df = pd.read_parquet(path)
            # Extract seed from path: .../seed42/per_basin_results.parquet
            seed_str = path.parent.name
            if seed_str.startswith("seed"):
                df["seed"] = int(seed_str[4:])
            frames.append(df)
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
    if not frames:
        log.warning("No param learning results found under %s", pl_root)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Table 1: Cross-seed stability (parameter learning)
# ---------------------------------------------------------------------------

def build_cross_seed_stability_table(pl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-basin × per-model metric variance across seeds.

    Returns a DataFrame with columns:
        basin_id, model_id, objective,
        test_KGE_mean, test_KGE_std, test_KGE_cv,
        test_KGE_LOG_mean, test_KGE_LOG_std, test_KGE_LOG_cv,
        test_NSE_mean, test_NSE_std, test_NSE_cv,
        n_seeds, is_stable
    """
    if pl_df.empty:
        return pd.DataFrame()

    group_keys = ["basin_id", "model_id", "objective"]
    existing_keys = [k for k in group_keys if k in pl_df.columns]
    if not existing_keys:
        return pd.DataFrame()

    metrics = [c for c in ["test_KGE", "test_KGE_LOG", "test_NSE"] if c in pl_df.columns]

    records = []
    for keys, grp in pl_df.groupby(existing_keys):
        rec: dict[str, Any] = dict(zip(existing_keys, keys if len(existing_keys) > 1 else [keys]))
        rec["n_seeds"] = int(grp["seed"].nunique()) if "seed" in grp.columns else len(grp)

        max_cv = 0.0
        for metric in metrics:
            vals = grp[metric].dropna().values
            if len(vals) < 2:
                rec[f"{metric}_mean"] = float(vals[0]) if len(vals) == 1 else float("nan")
                rec[f"{metric}_std"] = float("nan")
                rec[f"{metric}_cv"] = float("nan")
                continue
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            cv = std_v / abs(mean_v) if abs(mean_v) > 1e-9 else float("nan")
            rec[f"{metric}_mean"] = mean_v
            rec[f"{metric}_std"] = std_v
            rec[f"{metric}_cv"] = cv
            if np.isfinite(cv):
                max_cv = max(max_cv, cv)

        rec["is_stable"] = bool(max_cv <= _CV_STABLE_THRESHOLD)
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Table 2: Cross-loss stability (KGE vs KGE_LOG)
# ---------------------------------------------------------------------------

def build_cross_loss_stability_table(pl_df: pd.DataFrame) -> pd.DataFrame:
    """Compare metric consistency across KGE vs KGE_LOG objectives.

    Returns per-basin × model: mean metric difference, correlation between
    test_KGE ranks under KGE vs KGE_LOG objective.
    """
    if pl_df.empty or "objective" not in pl_df.columns:
        return pd.DataFrame()

    obj_vals = pl_df["objective"].unique()
    has_kge = any(str(o).upper() == "KGE" for o in obj_vals)
    has_kge_log = any(str(o).upper() == "KGE_LOG" for o in obj_vals)
    if not (has_kge and has_kge_log):
        log.warning("Both KGE and KGE_LOG objectives needed for cross-loss stability; found: %s", obj_vals)
        return pd.DataFrame()

    # Best seed per basin × model × objective (use test_KGE for ranking)
    def _best_seed_row(grp: pd.DataFrame) -> pd.Series:
        metric = "test_KGE" if "test_KGE" in grp.columns else grp.columns[0]
        idx = grp[metric].idxmax() if not grp[metric].isna().all() else grp.index[0]
        return grp.loc[idx]

    group_cols = [c for c in ["basin_id", "model_id", "objective"] if c in pl_df.columns]
    best = pl_df.groupby(group_cols).apply(_best_seed_row).reset_index(drop=True)

    kge_rows = best[best["objective"].str.upper() == "KGE"].copy()
    kge_log_rows = best[best["objective"].str.upper() == "KGE_LOG"].copy()

    merge_cols = [c for c in ["basin_id", "model_id"] if c in kge_rows.columns]
    if not merge_cols:
        return pd.DataFrame()

    merged = kge_rows.merge(
        kge_log_rows,
        on=merge_cols,
        suffixes=("_kge", "_kgelog"),
    )

    records = []
    for model_id, grp in merged.groupby("model_id") if "model_id" in merged.columns else [("all", merged)]:
        rec: dict[str, Any] = {"model_id": model_id}
        n = len(grp)
        rec["n_basins"] = n
        for base_metric in ["test_KGE", "test_NSE"]:
            col_a = f"{base_metric}_kge"
            col_b = f"{base_metric}_kgelog"
            if col_a in grp.columns and col_b in grp.columns:
                diff = (grp[col_a] - grp[col_b]).dropna()
                rec[f"{base_metric}_mean_diff"] = float(diff.mean()) if len(diff) else float("nan")
                rec[f"{base_metric}_abs_diff_mean"] = float(diff.abs().mean()) if len(diff) else float("nan")
                if n >= 5:
                    corr = float(grp[col_a].corr(grp[col_b]))
                    rec[f"{base_metric}_rank_corr"] = corr
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Table 3: Dominant control table
# ---------------------------------------------------------------------------

def build_dominant_control_table(
    calib_best: pd.DataFrame,
    pl_best: pd.DataFrame,
) -> pd.DataFrame:
    """Identify which evidence dimension (calibration vs PL) is dominant per basin × model.

    'Dominant' means the evidence type with higher absolute test_KGE.

    Returns columns: basin_id, model_id, calib_test_KGE, pl_test_KGE,
    dominant_evidence, kge_advantage
    """
    if calib_best.empty or pl_best.empty:
        return pd.DataFrame()

    merge_cols = [c for c in ["basin_id", "model_id"] if c in calib_best.columns and c in pl_best.columns]
    if not merge_cols:
        return pd.DataFrame()

    # Rename test_KGE to distinguish source
    calib_cols = merge_cols + [c for c in ["test_KGE", "test_NSE"] if c in calib_best.columns]
    pl_cols = merge_cols + [c for c in ["test_KGE", "test_NSE"] if c in pl_best.columns]

    merged = calib_best[calib_cols].merge(
        pl_best[pl_cols],
        on=merge_cols,
        suffixes=("_calib", "_pl"),
    )

    if "test_KGE_calib" not in merged.columns or "test_KGE_pl" not in merged.columns:
        return merged

    def _dominant(row: pd.Series) -> str:
        c = row.get("test_KGE_calib", float("nan"))
        p = row.get("test_KGE_pl", float("nan"))
        if np.isnan(c) and np.isnan(p):
            return "unknown"
        if np.isnan(c):
            return "param_learning"
        if np.isnan(p):
            return "calibration"
        return "calibration" if c >= p else "param_learning"

    merged["dominant_evidence"] = merged.apply(_dominant, axis=1)
    merged["kge_advantage"] = (merged["test_KGE_calib"] - merged["test_KGE_pl"]).abs()
    return merged


# ---------------------------------------------------------------------------
# Table 4: Dual-evidence summary (HH/HL/LH/LL)
# ---------------------------------------------------------------------------

def build_dual_evidence_summary(
    calib_best: pd.DataFrame,
    pl_best: pd.DataFrame,
    calib_kge_threshold: float = _KGE_HIGH_THRESHOLD,
    pl_kge_threshold: float = _PL_KGE_HIGH_THRESHOLD,
) -> pd.DataFrame:
    """Classify each basin × model into HH / HL / LH / LL.

    H = high performance, L = low performance.
    First letter = calibration evidence, second = parameter learning evidence.

    Columns: basin_id, model_id, calib_kge, pl_kge, calib_high, pl_high,
             dual_evidence_class (HH/HL/LH/LL)
    """
    if calib_best.empty or pl_best.empty:
        return pd.DataFrame()

    merge_cols = [c for c in ["basin_id", "model_id"] if c in calib_best.columns and c in pl_best.columns]
    if not merge_cols:
        return pd.DataFrame()

    c_cols = merge_cols + (["test_KGE"] if "test_KGE" in calib_best.columns else [])
    p_cols = merge_cols + (["test_KGE"] if "test_KGE" in pl_best.columns else [])

    merged = calib_best[c_cols].merge(pl_best[p_cols], on=merge_cols, suffixes=("_calib", "_pl"))

    if "test_KGE_calib" not in merged.columns or "test_KGE_pl" not in merged.columns:
        return merged

    merged["calib_high"] = merged["test_KGE_calib"] >= calib_kge_threshold
    merged["pl_high"] = merged["test_KGE_pl"] >= pl_kge_threshold

    def _class(row: pd.Series) -> str:
        c = row.get("calib_high", False)
        p = row.get("pl_high", False)
        return ("H" if c else "L") + ("H" if p else "L")

    merged["dual_evidence_class"] = merged.apply(_class, axis=1)
    merged = merged.rename(columns={
        "test_KGE_calib": "calib_kge",
        "test_KGE_pl": "pl_kge",
    })
    return merged


# ---------------------------------------------------------------------------
# Table 5: Recommendation readiness table
# ---------------------------------------------------------------------------

def build_recommendation_readiness_table(
    dual_summary: pd.DataFrame,
    stability_table: pd.DataFrame,
    ready_basin_fraction: float = _READY_BASIN_FRACTION,
    stability_cv_threshold: float = _READY_STABILITY_CV,
) -> pd.DataFrame:
    """Compute recommendation readiness per model.

    A model is 'ready' if:
    - >= ready_basin_fraction of basins classified HH or LH (calibration evidence high)
    - median cross-seed CV of test_KGE <= stability_cv_threshold

    Returns columns: model_id, n_basins, hh_fraction, lh_fraction,
    calibration_high_fraction, median_seed_cv_test_KGE, is_ready, readiness_score
    """
    if dual_summary.empty:
        return pd.DataFrame()

    records = []
    group_col = "model_id" if "model_id" in dual_summary.columns else None
    if group_col is None:
        return pd.DataFrame()

    for model_id, grp in dual_summary.groupby(group_col):
        n = len(grp)
        if n == 0:
            continue
        hh = (grp["dual_evidence_class"] == "HH").sum()
        hl = (grp["dual_evidence_class"] == "HL").sum()
        lh = (grp["dual_evidence_class"] == "LH").sum()
        ll = (grp["dual_evidence_class"] == "LL").sum()
        calib_high_frac = float((hh + hl) / n)

        # Stability from cross-seed table
        median_cv = float("nan")
        if not stability_table.empty and "model_id" in stability_table.columns:
            st_grp = stability_table[stability_table["model_id"] == model_id]
            cv_col = "test_KGE_cv" if "test_KGE_cv" in st_grp.columns else None
            if cv_col and not st_grp.empty:
                median_cv = float(st_grp[cv_col].dropna().median())

        is_ready = (
            calib_high_frac >= ready_basin_fraction
            and (np.isnan(median_cv) or median_cv <= stability_cv_threshold)
        )

        # Readiness score: weighted combination
        stability_score = max(0.0, 1.0 - (median_cv / stability_cv_threshold)) if np.isfinite(median_cv) else 0.5
        readiness_score = 0.7 * calib_high_frac + 0.3 * stability_score

        records.append({
            "model_id": model_id,
            "n_basins": n,
            "hh_count": int(hh),
            "hl_count": int(hl),
            "lh_count": int(lh),
            "ll_count": int(ll),
            "calibration_high_fraction": calib_high_frac,
            "pl_high_fraction": float((hh + lh) / n),
            "hh_fraction": float(hh / n),
            "median_seed_cv_test_KGE": median_cv,
            "is_ready": bool(is_ready),
            "readiness_score": float(readiness_score),
        })

    return pd.DataFrame(records).sort_values("readiness_score", ascending=False)


# ---------------------------------------------------------------------------
# Table 6: Performance-equivalent vs recommendation-ready comparison
# ---------------------------------------------------------------------------

def build_performance_equivalent_vs_recommendation_ready_table(
    readiness_table: pd.DataFrame,
    dual_summary: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Compare top-N models by raw performance vs recommendation readiness.

    'Performance-equivalent' top-N: models with highest mean calib_kge.
    'Recommendation-ready' top-N: models with highest readiness_score.

    Returns a wide table with both rankings and overlap statistics.
    """
    if readiness_table.empty or dual_summary.empty:
        return pd.DataFrame()

    # Compute mean calib KGE per model
    if "model_id" in dual_summary.columns and "calib_kge" in dual_summary.columns:
        perf_df = dual_summary.groupby("model_id")["calib_kge"].mean().reset_index()
        perf_df.columns = ["model_id", "mean_calib_kge"]
        perf_df = perf_df.sort_values("mean_calib_kge", ascending=False).reset_index(drop=True)
        perf_df["perf_rank"] = range(1, len(perf_df) + 1)
        top_perf = set(perf_df.head(top_n)["model_id"])
    else:
        perf_df = pd.DataFrame(columns=["model_id", "mean_calib_kge", "perf_rank"])
        top_perf = set()

    # Recommendation-ready top-N
    if "model_id" in readiness_table.columns and "readiness_score" in readiness_table.columns:
        ready_df = readiness_table.sort_values("readiness_score", ascending=False).reset_index(drop=True)
        ready_df = ready_df[["model_id", "readiness_score", "is_ready"]].copy()
        ready_df["ready_rank"] = range(1, len(ready_df) + 1)
        top_ready = set(ready_df.head(top_n)["model_id"])
    else:
        ready_df = pd.DataFrame(columns=["model_id", "readiness_score", "is_ready", "ready_rank"])
        top_ready = set()

    # Merge both rankings
    merged = perf_df.merge(ready_df, on="model_id", how="outer")
    merged["in_top_perf"] = merged["model_id"].isin(top_perf)
    merged["in_top_ready"] = merged["model_id"].isin(top_ready)
    merged["in_both"] = merged["in_top_perf"] & merged["in_top_ready"]
    merged = merged.sort_values("perf_rank", ascending=True)

    # Append overlap summary row
    overlap = len(top_perf & top_ready)
    log.info(
        "Top-%d overlap: perf_models=%s ready_models=%s overlap=%d",
        top_n, sorted(top_perf), sorted(top_ready), overlap,
    )
    return merged


# ---------------------------------------------------------------------------
# Best-row selectors
# ---------------------------------------------------------------------------

def _best_calib_per_basin_model(calib_df: pd.DataFrame) -> pd.DataFrame:
    """Select best calibration start per basin × model × objective."""
    if calib_df.empty:
        return pd.DataFrame()

    # If is_best_start column exists, use it
    if "is_best_start" in calib_df.columns:
        best = calib_df[calib_df["is_best_start"] == True].copy()
        if not best.empty:
            return best

    # Fallback: pick row with max test_KGE per group
    group_cols = [c for c in ["basin_id", "model_id", "objective"] if c in calib_df.columns]
    if not group_cols:
        return calib_df

    metric_col = "test_KGE" if "test_KGE" in calib_df.columns else None
    if metric_col is None:
        return calib_df.groupby(group_cols).first().reset_index()

    idx = calib_df.groupby(group_cols)[metric_col].idxmax()
    return calib_df.loc[idx.dropna()].reset_index(drop=True)


def _best_pl_per_basin_model(pl_df: pd.DataFrame, objective: str | None = None) -> pd.DataFrame:
    """Select best-seed row per basin × model (optionally filtered by objective)."""
    if pl_df.empty:
        return pd.DataFrame()

    if objective is not None and "objective" in pl_df.columns:
        pl_df = pl_df[pl_df["objective"].str.upper() == objective.upper()]

    group_cols = [c for c in ["basin_id", "model_id"] if c in pl_df.columns]
    if not group_cols:
        return pl_df

    metric_col = "test_KGE" if "test_KGE" in pl_df.columns else None
    if metric_col is None:
        return pl_df.groupby(group_cols).first().reset_index()

    idx = pl_df.groupby(group_cols)[metric_col].idxmax()
    return pl_df.loc[idx.dropna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_postprocess(
    output_dir: str | Path = "outputs",
    calib_kge_threshold: float = _KGE_HIGH_THRESHOLD,
    pl_kge_threshold: float = _PL_KGE_HIGH_THRESHOLD,
    top_n: int = 5,
) -> dict[str, Path]:
    """Run full post-processing pipeline.

    Parameters
    ----------
    output_dir : str | Path
        Root outputs directory (same as passed to calibration/param-learning runners).
    calib_kge_threshold : float
        KGE threshold for "high" calibration evidence in dual-evidence classification.
    pl_kge_threshold : float
        KGE threshold for "high" parameter learning evidence.
    top_n : int
        Number of top models for comparison table.

    Returns
    -------
    dict mapping table name → saved Parquet path
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(output_dir)
    calib_root = out / "independent_calibration"
    pl_root = out / "parameter_learning"
    post_dir = out / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    log.info("Loading calibration results from %s ...", calib_root)
    calib_df = _load_calibration_results(calib_root) if calib_root.exists() else pd.DataFrame()
    log.info("  Loaded %d calibration rows", len(calib_df))

    log.info("Loading parameter learning results from %s ...", pl_root)
    pl_df = _load_param_learning_results(pl_root) if pl_root.exists() else pd.DataFrame()
    log.info("  Loaded %d param learning rows", len(pl_df))

    # Select best starts / seeds
    calib_best = _best_calib_per_basin_model(calib_df)
    pl_best_kge = _best_pl_per_basin_model(pl_df, objective="KGE")
    pl_best_kgelog = _best_pl_per_basin_model(pl_df, objective="KGE_LOG")
    # Use whichever objective gives higher test_KGE for dual-evidence
    if not pl_best_kge.empty and not pl_best_kgelog.empty:
        merge_cols = [c for c in ["basin_id", "model_id"] if c in pl_best_kge.columns]
        pl_combined = pl_best_kge.merge(pl_best_kgelog, on=merge_cols, suffixes=("_kge", "_kgelog"), how="outer")
        # Pick higher test_KGE per basin
        if "test_KGE_kge" in pl_combined.columns and "test_KGE_kgelog" in pl_combined.columns:
            pl_combined["test_KGE"] = pl_combined[["test_KGE_kge", "test_KGE_kgelog"]].max(axis=1)
        pl_best = pl_combined
    else:
        pl_best = pl_best_kge if not pl_best_kge.empty else pl_best_kgelog

    # Table 1: Cross-seed stability
    log.info("Building cross_seed_stability_table ...")
    t1 = build_cross_seed_stability_table(pl_df)
    if not t1.empty:
        p = post_dir / "cross_seed_stability_table.parquet"
        t1.to_parquet(p, index=False)
        saved["cross_seed_stability_table"] = p
        log.info("  Saved %d rows → %s", len(t1), p)

    # Table 2: Cross-loss stability
    log.info("Building cross_loss_stability_table ...")
    t2 = build_cross_loss_stability_table(pl_df)
    if not t2.empty:
        p = post_dir / "cross_loss_stability_table.parquet"
        t2.to_parquet(p, index=False)
        saved["cross_loss_stability_table"] = p
        log.info("  Saved %d rows → %s", len(t2), p)

    # Table 3: Dominant control
    log.info("Building dominant_control_table ...")
    t3 = build_dominant_control_table(calib_best, pl_best)
    if not t3.empty:
        p = post_dir / "dominant_control_table.parquet"
        t3.to_parquet(p, index=False)
        saved["dominant_control_table"] = p
        log.info("  Saved %d rows → %s", len(t3), p)

    # Table 4: Dual-evidence summary
    log.info("Building dual_evidence_summary ...")
    t4 = build_dual_evidence_summary(
        calib_best, pl_best,
        calib_kge_threshold=calib_kge_threshold,
        pl_kge_threshold=pl_kge_threshold,
    )
    if not t4.empty:
        p = post_dir / "dual_evidence_summary.parquet"
        t4.to_parquet(p, index=False)
        saved["dual_evidence_summary"] = p
        log.info("  Saved %d rows → %s", len(t4), p)
        # Also write class distribution
        if "dual_evidence_class" in t4.columns:
            class_counts = t4.groupby(["model_id", "dual_evidence_class"]).size().unstack(fill_value=0) if "model_id" in t4.columns else t4["dual_evidence_class"].value_counts().to_frame("count")
            class_counts.to_csv(post_dir / "dual_evidence_class_distribution.csv")
    else:
        t4 = pd.DataFrame()

    # Table 5: Recommendation readiness
    log.info("Building recommendation_readiness_table ...")
    t5 = build_recommendation_readiness_table(
        t4, t1 if not t1.empty else pd.DataFrame()
    )
    if not t5.empty:
        p = post_dir / "recommendation_readiness_table.parquet"
        t5.to_parquet(p, index=False)
        saved["recommendation_readiness_table"] = p
        t5.to_csv(post_dir / "recommendation_readiness_table.csv", index=False)
        log.info("  Saved %d rows → %s", len(t5), p)
    else:
        t5 = pd.DataFrame()

    # Table 6: Performance-equivalent vs recommendation-ready
    log.info("Building performance_equivalent_vs_recommendation_ready_table ...")
    t6 = build_performance_equivalent_vs_recommendation_ready_table(t5, t4, top_n=top_n)
    if not t6.empty:
        p = post_dir / "performance_equivalent_vs_recommendation_ready_table.parquet"
        t6.to_parquet(p, index=False)
        saved["performance_equivalent_vs_recommendation_ready_table"] = p
        t6.to_csv(post_dir / "performance_equivalent_vs_recommendation_ready_table.csv", index=False)
        log.info("  Saved %d rows → %s", len(t6), p)

    log.info("Post-processing complete. %d tables saved to %s", len(saved), post_dir)
    return saved
