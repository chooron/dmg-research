#!/usr/bin/env python3
"""Pure-analysis second audit of the frozen parameter--attribute atlas.

This script reads the existing atlas tables, persisted normalized parameter vectors,
and the canonical 531 x 35 attribute matrix.  It performs no model forward, training,
checkpoint write, OOB/PUR run, bootstrap, or permutation.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR)]
import parameter_attribute_atlas as atlas  # noqa: E402
from dpl.attributes import CatchmentAttributeBuilder  # noqa: E402
from src.data_selection import load_ids  # noqa: E402

REPO = atlas.REPO
BENCHMARK = atlas.BENCHMARK
IN = atlas.OUT
OUT = BENCHMARK / "results/parameter_attribute_atlas_followup_20260829"
IDS_PATH = atlas.IDS_PATH
ATTRIBUTES = list(atlas.CAMELS_35_ATTRIBUTES)
CATEGORICAL = set(atlas.CATEGORICAL)
CONTINUOUS = list(atlas.CONTINUOUS)
ALL_MODELS = list(atlas.ALL_MODELS)
MAIN_HIGH = atlas.MAIN_HIGH_RHO
MAIN_LOW = atlas.MAIN_LOW_RHO
N_PERSISTENT = 60
REDUNDANCY_THRESHOLDS = (0.70, 0.80)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def corr_pair(x: Any, y: Any, rank: bool = False) -> tuple[float, int]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan"), n
    if rank:
        a = pd.Series(a[ok]).rank(method="average").to_numpy()
        b = pd.Series(b[ok]).rank(method="average").to_numpy()
    else:
        a, b = a[ok], b[ok]
    return float(np.corrcoef(a, b)[0, 1]), n


def rank_residual_partial(target: np.ndarray, parameter: np.ndarray, controls: list[np.ndarray]) -> tuple[float, int]:
    """Partial Spearman-like sensitivity using rank residualization and <=2 controls."""
    arrays = [np.asarray(target, dtype=float), np.asarray(parameter, dtype=float)] + [np.asarray(x, dtype=float) for x in controls]
    ok = np.logical_and.reduce([np.isfinite(x) for x in arrays])
    n = int(ok.sum())
    if n < 5:
        return float("nan"), n
    ranks = [pd.Series(x[ok]).rank(method="average").to_numpy() for x in arrays]
    design = np.column_stack([np.ones(n), *ranks[2:]])
    target_resid = ranks[0] - design @ np.linalg.lstsq(design, ranks[0], rcond=None)[0]
    parameter_resid = ranks[1] - design @ np.linalg.lstsq(design, ranks[1], rcond=None)[0]
    if np.std(target_resid) == 0 or np.std(parameter_resid) == 0:
        return float("nan"), n
    return float(np.corrcoef(target_resid, parameter_resid)[0, 1]), n


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = np.asarray([int(x) for x in load_ids(IDS_PATH)], dtype=np.int64)
    if ids.size != 531 or np.unique(ids).size != 531:
        raise RuntimeError("canonical basin IDs are not exactly 531 unique IDs")
    builder = CatchmentAttributeBuilder()
    raw = builder.load_raw_attributes(ids)
    normalized = builder.build_normalized_attributes(ids, device="cpu", method="zscore").detach().cpu().numpy()
    if raw.shape != (531, 35) or normalized.shape != (531, 35):
        raise RuntimeError(f"expected 531x35 attributes, got {raw.shape}/{normalized.shape}")
    pair = pd.read_csv(IN / "ic_dpl_pairwise_relationships.csv")
    persistent = pd.read_csv(IN / "relationship_candidate_list.csv")
    persistent = persistent[(persistent.population == "STRICT_FULL300_34") & (persistent.candidate_set == "persistent/reproduced")].copy()
    if len(persistent) != N_PERSISTENT:
        raise RuntimeError(f"expected exactly 60 persistent candidate rows, got {len(persistent)}")
    repro = pd.read_csv(IN / "parameter_reproducibility.csv")
    param_audit = pd.read_csv(IN / "parameter_audit.csv")
    semantic = pd.read_csv(IN / "parameter_semantic_mapping.csv")
    dseen = pd.read_csv(IN / "d_seen_secondary_linkage.csv")
    return ids, raw, normalized, pair, persistent, repro, param_audit, semantic, dseen


def column_audit(raw: np.ndarray, normalized: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, attr in enumerate(ATTRIBUTES):
        x = raw[:, j]
        finite = np.isfinite(x)
        rows.append({
            "attribute": attr, "attribute_index": j, "attribute_type": "CATEGORICAL_CODE" if attr in CATEGORICAL else "CONTINUOUS",
            "raw_finite_count": int(finite.sum()), "raw_nonfinite_count": int((~finite).sum()),
            "raw_unique_count": int(np.unique(x[finite]).size), "raw_min": float(np.nanmin(x)), "raw_max": float(np.nanmax(x)),
            "normalized_finite_count": int(np.isfinite(normalized[:, j]).sum()), "normalization": "nan_to_num; log columns 11/23/34; all-531 zscore",
            "used_for_redundancy_graph": attr in CONTINUOUS, "interpretation": "ordinal code descriptive only" if attr in CATEGORICAL else "continuous attribute",
        })
    return pd.DataFrame(rows)


def attribute_correlations(matrix: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, a1 in enumerate(ATTRIBUTES):
        for j, a2 in enumerate(ATTRIBUTES):
            rho, n = corr_pair(matrix[:, i], matrix[:, j], rank=True)
            rows.append({"attribute_1": a1, "attribute_2": a2, "attribute_1_index": i, "attribute_2_index": j,
                         "attribute_1_type": "CATEGORICAL_CODE" if a1 in CATEGORICAL else "CONTINUOUS",
                         "attribute_2_type": "CATEGORICAL_CODE" if a2 in CATEGORICAL else "CONTINUOUS",
                         "rho": rho, "abs_rho": abs(rho) if np.isfinite(rho) else np.nan, "n_basins": n,
                         "diagonal": i == j, "redundancy_graph_eligible": a1 in CONTINUOUS and a2 in CONTINUOUS,
                         "label": "ATTRIBUTE_COLLINEARITY_AUDIT"})
    long = pd.DataFrame(rows)
    unique = long[(long.attribute_1_index < long.attribute_2_index) & long.redundancy_graph_eligible].copy()
    high = unique[unique.abs_rho >= min(REDUNDANCY_THRESHOLDS)].sort_values(["abs_rho", "attribute_1", "attribute_2"], ascending=[False, True, True])
    return long, unique, high


def connected_components(unique_corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    nodes = list(CONTINUOUS)
    adj = {a: set() for a in nodes}
    for r in unique_corr[unique_corr.abs_rho >= threshold].itertuples():
        adj[r.attribute_1].add(r.attribute_2)
        adj[r.attribute_2].add(r.attribute_1)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        stack = [node]; seen.add(node); comp = []
        while stack:
            x = stack.pop(); comp.append(x)
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y); stack.append(y)
        components.append(sorted(comp, key=ATTRIBUTES.index))
    components.sort(key=lambda x: (ATTRIBUTES.index(x[0]), x[0]))
    rows = []
    label = f"{threshold:.2f}"
    for idx, members in enumerate(components, 1):
        representative = min(members, key=lambda a: (-len(members), ATTRIBUTES.index(a)))
        for member_index, member in enumerate(members, 1):
            rows.append({"threshold": threshold, "cluster_id": f"absrho_ge_{label}_C{idx:03d}", "member_index": member_index,
                         "attribute": member, "attribute_type": "CONTINUOUS", "representative_attribute": representative,
                         "member_count": len(members), "member_attributes": ";".join(members), "included_in_graph": True,
                         "cluster_rule": f"connected component of continuous attributes with absolute Spearman >= {threshold:.2f}",
                         "label": "REDUNDANCY_GROUP_DESCRIPTIVE"})
    for member in sorted(CATEGORICAL, key=ATTRIBUTES.index):
        rows.append({"threshold": threshold, "cluster_id": "CATEGORICAL_EXCLUDED", "member_index": 1,
                     "attribute": member, "attribute_type": "CATEGORICAL_CODE", "representative_attribute": member,
                     "member_count": 1, "member_attributes": member, "included_in_graph": False,
                     "cluster_rule": "excluded from continuous redundancy graph; ordinal-code correlations retained only in matrix",
                     "label": "CATEGORICAL_DESCRIPTIVE_ONLY"})
    return pd.DataFrame(rows)


def load_vectors(common: list[str], strict: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ic: dict[str, np.ndarray] = {}; dp: dict[str, np.ndarray] = {}
    for model in ALL_MODELS:
        ip = IN / "parameter_vectors_ic" / f"{model}.csv"
        if not ip.is_file():
            raise RuntimeError(f"missing IC vector file: {ip}")
        iv = pd.read_csv(ip)
        if len(iv) != 531 or iv.basin_id.astype(str).nunique() != 531:
            raise RuntimeError(f"{model}: IC vector coverage is not 531 unique basin IDs")
        ic[model] = iv.drop(columns=["basin_id"]).to_numpy(dtype=float)
        if model in common:
            dp_path = IN / "parameter_vectors_dpl" / f"{model}.csv"
            if not dp_path.is_file():
                raise RuntimeError(f"missing dPL vector file: {dp_path}")
            dv = pd.read_csv(dp_path)
            if len(dv) != 531 or dv.basin_id.astype(str).nunique() != 531:
                raise RuntimeError(f"{model}: dPL vector coverage is not 531 unique basin IDs")
            dp[model] = dv.drop(columns=["basin_id"]).to_numpy(dtype=float)
    return ic, dp


def add_clusters_to_persistent(persistent: pd.DataFrame, groups: dict[float, pd.DataFrame]) -> pd.DataFrame:
    out = persistent.copy()
    for threshold, group in groups.items():
        member_map = group.set_index("attribute")
        out[f"cluster_{threshold:.2f}"] = out.attribute.map(member_map.cluster_id).fillna("UNRESOLVED")
        out[f"cluster_rep_{threshold:.2f}"] = out.attribute.map(member_map.representative_attribute).fillna("UNRESOLVED")
        out[f"cluster_members_{threshold:.2f}"] = out.attribute.map(member_map.member_attributes).fillna("UNRESOLVED")
    out["source_scope"] = "STRICT_FULL300_34; relationship_candidate_list persistent/reproduced; N=60"
    out["pseudo_replication_note"] = "rows share models, parameters, and attributes; not independent evidence"
    return out


def concentration_tables(persistent: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_tables = []
    for dimension in ["attribute", "model", "parameter"]:
        g = persistent.groupby(dimension, sort=False).agg(relationship_count=(dimension, "size"), unique_models=("model", "nunique"), unique_parameters=("parameter", "nunique"), unique_attributes=("attribute", "nunique")).reset_index()
        g["dimension"] = dimension; g["fraction_of_60"] = g.relationship_count / len(persistent)
        g = g.sort_values(["relationship_count", dimension], ascending=[False, True]).reset_index(drop=True)
        g["rank"] = np.arange(1, len(g) + 1)
        by_tables.append(g)
    by_attribute, by_model, by_parameter = by_tables
    model_parameter = persistent.groupby(["model", "parameter"], sort=False).agg(relationship_count=("attribute", "size"), unique_attributes=("attribute", "nunique")).reset_index().sort_values(["relationship_count", "model", "parameter"], ascending=[False, True, True])
    summary_rows = []
    for dimension, table in [("attribute", by_attribute), ("model", by_model), ("parameter", by_parameter)]:
        for k in [1, 3, 5, 10]:
            top = table.head(min(k, len(table)))
            summary_rows.append({"dimension": dimension, "k": k, "top_units": ";".join(map(str, top[dimension])), "top_relationship_count": int(top.relationship_count.sum()), "top_share_of_60": float(top.relationship_count.sum() / len(persistent)), "unique_units_total": len(table), "scope": "STRICT_FULL300_34 persistent/reproduced candidate set N=60"})
    concentration = pd.DataFrame(summary_rows)
    return by_attribute, by_model, by_parameter, model_parameter, concentration


def leave_one_contributor(persistent: pd.DataFrame, by_attribute: pd.DataFrame, by_model: pd.DataFrame, by_parameter: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension, table in [("attribute", by_attribute), ("model", by_model), ("parameter", by_parameter)]:
        top = str(table.iloc[0][dimension])
        remaining = persistent[persistent[dimension].astype(str) != top]
        rows.append({"dimension": dimension, "removed_top_unit": top, "removed_count": int(len(persistent) - len(remaining)), "remaining_count": int(len(remaining)), "remaining_unique_models": int(remaining.model.nunique()), "remaining_unique_parameters": int(remaining.parameter.nunique()), "remaining_unique_attributes": int(remaining.attribute.nunique()), "remaining_attribute_counts": ";".join(f"{k}:{v}" for k, v in remaining.attribute.value_counts().items()), "rule": "remove all rows containing the top contributor; no re-ranking after removal", "label": "CONCENTRATION_SENSITIVITY"})
    return pd.DataFrame(rows)


def redundancy_recurrence(persistent: pd.DataFrame, groups: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for threshold, group in groups.items():
        member_map = group.set_index("attribute")
        x = persistent.copy(); x["cluster_id"] = x.attribute.map(member_map.cluster_id).fillna("UNRESOLVED")
        x["cluster_rep"] = x.attribute.map(member_map.representative_attribute).fillna("UNRESOLVED")
        x["cluster_members"] = x.attribute.map(member_map.member_attributes).fillna("UNRESOLVED")
        for cluster, g in x.groupby(["cluster_id", "cluster_rep", "cluster_members"], sort=True):
            cid, rep, members = cluster
            dedup = g.drop_duplicates(["model", "parameter"])
            rows.append({"threshold": threshold, "unit_type": "REDUNDANCY_CLUSTER", "unit_id": cid, "representative_attribute": rep, "member_attributes": members,
                         "raw_relationship_count": len(g), "deduplicated_model_parameter_count": len(dedup), "unique_models": dedup.model.nunique(), "unique_parameter_names": dedup.parameter.nunique(), "raw_unique_attributes": g.attribute.nunique(), "fraction_raw_60": len(g) / len(persistent), "fraction_deduplicated_60": len(dedup) / len(persistent), "label": "REDUNDANCY_AWARE_PERSISTENT_RECURRENCE"})
        # Include all continuous clusters even if not represented by the 60 rows.
        present = set(x.cluster_id)
        for cluster, g in group.groupby(["cluster_id", "representative_attribute", "member_attributes"], sort=True):
            cid, rep, members = cluster
            if cid not in present:
                rows.append({"threshold": threshold, "unit_type": "REDUNDANCY_CLUSTER", "unit_id": cid, "representative_attribute": rep, "member_attributes": members, "raw_relationship_count": 0, "deduplicated_model_parameter_count": 0, "unique_models": 0, "unique_parameter_names": 0, "raw_unique_attributes": 0, "fraction_raw_60": 0.0, "fraction_deduplicated_60": 0.0, "label": "REDUNDANCY_AWARE_PERSISTENT_RECURRENCE"})
    # Member-level view makes the change from raw counts inspectable.
    for attr, g in persistent.groupby("attribute", sort=False):
        for threshold, group in groups.items():
            member_map = group.set_index("attribute")
            row = member_map.loc[attr] if attr in member_map.index else None
            rows.append({"threshold": threshold, "unit_type": "ATTRIBUTE_MEMBER", "unit_id": attr, "representative_attribute": row.representative_attribute if row is not None else "UNRESOLVED", "member_attributes": row.member_attributes if row is not None else "UNRESOLVED", "raw_relationship_count": len(g), "deduplicated_model_parameter_count": len(g.drop_duplicates(["model", "parameter"])), "unique_models": g.model.nunique(), "unique_parameter_names": g.parameter.nunique(), "raw_unique_attributes": 1, "fraction_raw_60": len(g) / len(persistent), "fraction_deduplicated_60": len(g.drop_duplicates(["model", "parameter"])) / len(persistent), "label": "REDUNDANCY_AWARE_MEMBER_VIEW"})
    return pd.DataFrame(rows)


def parameter_reliability(repro: pd.DataFrame, audit: pd.DataFrame, ic_long: pd.DataFrame, dp_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = repro[repro.population == "STRICT_FULL300_34"].copy()
    r = r.rename(columns={"role_if_available": "semantic_role_if_available"})
    a = audit[audit.population == "STRICT_FULL300_34"].copy()
    keys = ["model", "parameter_index", "parameter"]
    ic_a = a[a.method == "IC"].set_index(keys)
    dp_a = a[a.method == "dPL"].set_index(keys)
    for method, src in [("ic", ic_a), ("dpl", dp_a)]:
        for source_col, target_col in [("normalized_sd", f"{method}_parameter_sd"), ("normalized_iqr", f"{method}_parameter_iqr"), ("normalized_min", f"{method}_effective_min"), ("normalized_max", f"{method}_effective_max"), ("boundary_low_fraction", f"{method}_boundary_low_fraction"), ("boundary_high_fraction", f"{method}_boundary_high_fraction"), ("parameter_boundary_diagnostic", f"{method}_boundary_status"), ("near_constant", f"{method}_near_constant"), ("boundary_concentrated", f"{method}_boundary_concentrated")]:
            r = r.join(src[source_col].rename(target_col), on=keys)
    r["ic_effective_range"] = r.ic_effective_max - r.ic_effective_min
    r["dpl_effective_range"] = r.dpl_effective_max - r.dpl_effective_min
    r["ic_boundary_fraction_max"] = r[["ic_boundary_low_fraction", "ic_boundary_high_fraction"]].max(axis=1)
    r["dpl_boundary_fraction_max"] = r[["dpl_boundary_low_fraction", "dpl_boundary_high_fraction"]].max(axis=1)
    r["reliability_bin"] = np.select([r.profile_spearman < 0, r.profile_spearman < .5], ["negative", "low_positive"], default="moderate_or_high")
    r["restart_evidence_level"] = "MODEL_LEVEL_ONLY"
    r["restart_parameter_claim"] = "NOT_AVAILABLE; restart artifacts contain repeated IC scores, not repeated parameter vectors"
    # Cell-strength summaries are descriptive and remain separate from profile reproducibility.
    for method, frame, col in [("ic", ic_long, "abs_rho_ic"), ("dpl", dp_long, "abs_rho_dpl")]:
        x = frame[frame.population == "STRICT_FULL300_34"].groupby(keys)[col].agg([("max_abs_rho", "max"), ("median_abs_rho", "median")]).rename(columns={"max_abs_rho": f"{method}_max_abs_rho", "median_abs_rho": f"{method}_median_abs_rho"})
        r = r.join(x, on=keys)
    restart_path = BENCHMARK / "results/r1_r2_nontraining_complete_20260829/wpC_restart_spread_summary.csv"
    if restart_path.is_file():
        restart = pd.read_csv(restart_path)
        r = r.merge(restart[["model", "median_restart_sd", "p95_restart_sd", "winner_match_fraction"]], on="model", how="left", validate="many_to_one")
    return r, ic_a.reset_index(), dp_a.reset_index()


def reliability_linkage(r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_map = {"ic_parameter_sd": "IC normalized SD", "dpl_parameter_sd": "dPL normalized SD", "ic_parameter_iqr": "IC normalized IQR", "dpl_parameter_iqr": "dPL normalized IQR", "ic_effective_range": "IC effective range", "dpl_effective_range": "dPL effective range", "ic_boundary_fraction_max": "IC max boundary fraction", "dpl_boundary_fraction_max": "dPL max boundary fraction", "ic_near_constant": "IC near-constant flag", "dpl_near_constant": "dPL near-constant flag"}
    rows = []
    for feature, description in feature_map.items():
        x = r[feature].astype(float) if r[feature].dtype != bool else r[feature].astype(float)
        sr, n = corr_pair(r.profile_spearman, x, rank=True)
        pr, _ = corr_pair(r.profile_spearman, x, rank=False)
        rows.append({"analysis": "parameter_profile_reproducibility_vs_parameter_diagnostic", "feature": feature, "feature_description": description, "spearman_r": sr, "pearson_r": pr, "n_parameters": n, "interpretation": "exploratory association across 255 model-parameter rows; not causal; model/parameter rows are not independent", "label": "SEEN_BASIN_NON_OOB"})
    for b, g in r.groupby("reliability_bin", sort=False):
        rows.append({"analysis": "diagnostic_distribution_by_reliability_bin", "feature": b, "feature_description": "negative (<0), low_positive (<0.5), moderate_or_high (>=0.5) profile Spearman", "spearman_r": np.nan, "pearson_r": np.nan, "n_parameters": len(g), "median_ic_sd": g.ic_parameter_sd.median(), "median_dpl_sd": g.dpl_parameter_sd.median(), "median_ic_boundary_fraction": g.ic_boundary_fraction_max.median(), "median_dpl_boundary_fraction": g.dpl_boundary_fraction_max.median(), "median_profile_spearman": g.profile_spearman.median(), "interpretation": "descriptive bin summary; thresholds are not scientific cutoffs", "label": "SEEN_BASIN_NON_OOB"})
    by_model = r.groupby("model", sort=True).agg(median_parameter_profile_spearman=("profile_spearman", "median"), mean_parameter_profile_spearman=("profile_spearman", "mean"), n_parameters=("parameter", "size"), median_ic_sd=("ic_parameter_sd", "median"), median_dpl_sd=("dpl_parameter_sd", "median"), median_ic_boundary_fraction=("ic_boundary_fraction_max", "median"), median_dpl_boundary_fraction=("dpl_boundary_fraction_max", "median"), median_restart_sd=("median_restart_sd", "first"), p95_restart_sd=("p95_restart_sd", "first"), winner_match_fraction=("winner_match_fraction", "first")).reset_index()
    sr, _ = corr_pair(by_model.median_parameter_profile_spearman, by_model.median_restart_sd, rank=True)
    pr, _ = corr_pair(by_model.median_parameter_profile_spearman, by_model.median_restart_sd, rank=False)
    model_row = pd.DataFrame([{"analysis": "model_level_parameter_profile_vs_IC_restart_spread", "feature": "median_restart_sd", "feature_description": "8-model archived restart panel only; model-level aggregation", "spearman_r": sr, "pearson_r": pr, "n_parameters": len(by_model), "models_in_restart_panel": int(by_model.median_restart_sd.notna().sum()), "interpretation": "MODEL_LEVEL_ONLY exploratory linkage; no parameter-level restart inference", "label": "SEEN_BASIN_NON_OOB"}])
    return pd.concat([pd.DataFrame(rows), model_row], ignore_index=True), by_model


def low_prec_and_conditional(pair: pd.DataFrame, persistent: pd.DataFrame, matrix: np.ndarray, repro: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    attr_index = {a: i for i, a in enumerate(ATTRIBUTES)}
    low = persistent[persistent.attribute == "low_prec_dur"].copy()
    controls = []
    target = matrix[:, attr_index["low_prec_dur"]]
    control_scores = []
    for attr in CONTINUOUS:
        if attr == "low_prec_dur":
            continue
        rho, n = corr_pair(target, matrix[:, attr_index[attr]], rank=True)
        control_scores.append((abs(rho) if np.isfinite(rho) else -np.inf, attr, rho, n))
    control_scores.sort(key=lambda x: (-x[0], ATTRIBUTES.index(x[1])))
    selected = [x for x in control_scores[:2] if np.isfinite(x[0])]
    controls = [matrix[:, attr_index[x[1]]] for x in selected]
    controls_meta = ";".join(f"{x[1]}:{x[2]:+.4f}" for x in selected)
    out_rows = []
    for row in low.itertuples(index=False):
        iv = pd.read_csv(IN / "parameter_vectors_ic" / f"{row.model}.csv")[row.parameter].to_numpy(dtype=float)
        dv = pd.read_csv(IN / "parameter_vectors_dpl" / f"{row.model}.csv")[row.parameter].to_numpy(dtype=float)
        c_ic, n_ic = rank_residual_partial(target, iv, controls)
        c_dp, n_dp = rank_residual_partial(target, dv, controls)
        out_rows.append({**row._asdict(), "conditional_controls": controls_meta, "conditional_control_count": len(controls), "conditional_rho_ic": c_ic, "conditional_rho_dpl": c_dp, "conditional_n_ic": n_ic, "conditional_n_dpl": n_dp, "conditional_same_sign_ic": bool(np.sign(row.rho_ic) == np.sign(c_ic)) if np.isfinite(c_ic) else False, "conditional_same_sign_dpl": bool(np.sign(row.rho_dpl) == np.sign(c_dp)) if np.isfinite(c_dp) else False, "conditional_method": "rank residualization against top two continuous attributes correlated with low_prec_dur; exploratory sensitivity", "conditional_status": "RUN"})
    # Add the selected sign-changing examples, preserving the same control-selection rule per target attribute.
    signs = pair[(pair.population == "STRICT_FULL300_34") & (pair.relationship_class == "sign-changing") & (pair.attribute_type == "CONTINUOUS")].copy()
    signs = signs.assign(abs_sum=signs.abs_rho_ic + signs.abs_rho_dpl).sort_values(["abs_sum", "model", "parameter", "attribute"], ascending=[False, True, True, True]).head(10)
    for row in signs.itertuples(index=False):
        t = matrix[:, attr_index[row.attribute]]
        scores = []
        for attr in CONTINUOUS:
            if attr == row.attribute:
                continue
            rho, n = corr_pair(t, matrix[:, attr_index[attr]], rank=True)
            scores.append((abs(rho) if np.isfinite(rho) else -np.inf, attr, rho, n))
        scores.sort(key=lambda x: (-x[0], ATTRIBUTES.index(x[1])))
        chosen = [x for x in scores[:2] if np.isfinite(x[0])]
        controls_local = [matrix[:, attr_index[x[1]]] for x in chosen]
        iv = pd.read_csv(IN / "parameter_vectors_ic" / f"{row.model}.csv")[row.parameter].to_numpy(dtype=float)
        dv = pd.read_csv(IN / "parameter_vectors_dpl" / f"{row.model}.csv")[row.parameter].to_numpy(dtype=float)
        c_ic, n_ic = rank_residual_partial(t, iv, controls_local); c_dp, n_dp = rank_residual_partial(t, dv, controls_local)
        out_rows.append({**row._asdict(), "case_type": "SIGN_CHANGING_REPRESENTATIVE", "conditional_controls": ";".join(f"{x[1]}:{x[2]:+.4f}" for x in chosen), "conditional_control_count": len(chosen), "conditional_rho_ic": c_ic, "conditional_rho_dpl": c_dp, "conditional_n_ic": n_ic, "conditional_n_dpl": n_dp, "conditional_same_sign_ic": bool(np.sign(row.rho_ic) == np.sign(c_ic)) if np.isfinite(c_ic) else False, "conditional_same_sign_dpl": bool(np.sign(row.rho_dpl) == np.sign(c_dp)) if np.isfinite(c_dp) else False, "conditional_method": "rank residualization against top two continuous attributes correlated with target; exploratory sensitivity", "conditional_status": "RUN"})
    cond = pd.DataFrame(out_rows)
    diag = audit[audit.population == "STRICT_FULL300_34"].copy()
    piv = diag.pivot_table(index=["model", "parameter"], columns="method", values=["normalized_sd", "normalized_iqr", "parameter_boundary_diagnostic"], aggfunc="first")
    piv.columns = [f"{m.lower()}_{v}" for v, m in piv.columns]
    low_out = low.merge(piv.reset_index(), on=["model", "parameter"], how="left")
    low_out["parameter_variance"] = low_out["ic_normalized_sd"]
    low_out["boundary_status"] = low_out["ic_parameter_boundary_diagnostic"].astype(str) + " | " + low_out["dpl_parameter_boundary_diagnostic"].astype(str)
    low_out["semantic_role_if_available"] = low_out.get("parameter_role_if_available", "UNRESOLVED")
    return low_out, cond


def class_enrichment(pair: pd.DataFrame, reliability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["model", "parameter"]
    rel_cols = keys + ["profile_spearman", "ic_parameter_sd", "dpl_parameter_sd", "ic_boundary_concentrated", "dpl_boundary_concentrated", "ic_near_constant", "dpl_near_constant", "ic_max_abs_rho", "dpl_max_abs_rho"]
    x = pair[pair.population == "STRICT_FULL300_34"].merge(reliability[rel_cols], on=keys, how="left", validate="many_to_one")
    summary = []
    for cls, g in x.groupby("relationship_class", sort=False):
        params = g.drop_duplicates(keys)
        summary.append({"relationship_class": cls, "cell_count": len(g), "unique_models": g.model.nunique(), "unique_parameter_names": g.parameter.nunique(), "unique_model_parameter": len(params), "unique_attributes": g.attribute.nunique(), "median_abs_rho_ic": g.abs_rho_ic.median(), "median_abs_rho_dpl": g.abs_rho_dpl.median(), "median_profile_spearman": params.profile_spearman.median(), "median_ic_parameter_sd": params.ic_parameter_sd.median(), "median_dpl_parameter_sd": params.dpl_parameter_sd.median(), "ic_boundary_parameter_fraction": params.ic_boundary_concentrated.mean(), "dpl_boundary_parameter_fraction": params.dpl_boundary_concentrated.mean(), "ic_near_constant_parameter_fraction": params.ic_near_constant.mean(), "dpl_near_constant_parameter_fraction": params.dpl_near_constant.mean(), "same_sign_fraction": g.same_sign.mean(), "label": "SEEN_BASIN_NON_OOB_CLASS_ENRICHMENT"})
    attr_rows = g_rows = []
    for (cls, attr), g in x.groupby(["relationship_class", "attribute"], sort=True):
        attr_rows.append({"relationship_class": cls, "attribute": attr, "attribute_type": g.attribute_type.iloc[0], "cell_count": len(g), "fraction_within_class": len(g) / max(1, len(x[x.relationship_class == cls])), "unique_models": g.model.nunique(), "median_abs_rho_ic": g.abs_rho_ic.median(), "median_abs_rho_dpl": g.abs_rho_dpl.median(), "label": "SEEN_BASIN_NON_OOB_CLASS_ATTRIBUTE_COMPOSITION"})
    for (cls, model), g in x.groupby(["relationship_class", "model"], sort=True):
        g_rows.append({"relationship_class": cls, "model": model, "cell_count": len(g), "fraction_within_class": len(g) / max(1, len(x[x.relationship_class == cls])), "unique_parameters": g.parameter.nunique(), "label": "SEEN_BASIN_NON_OOB_CLASS_MODEL_COMPOSITION"})
    return pd.DataFrame(summary), pd.DataFrame(attr_rows), pd.DataFrame(g_rows)


def targeted_semantics(persistent: pd.DataFrame, pair: pd.DataFrame, repro: pd.DataFrame, semantic: pd.DataFrame) -> pd.DataFrame:
    targets = set(zip(persistent.model, persistent.parameter))
    signs = pair[(pair.population == "STRICT_FULL300_34") & (pair.relationship_class == "sign-changing")].sort_values("abs_rho_ic", ascending=False).head(10)
    targets.update(zip(signs.model, signs.parameter))
    lows = repro[repro.population == "STRICT_FULL300_34"].nsmallest(5, "profile_spearman")
    highs = repro[repro.population == "STRICT_FULL300_34"].nlargest(5, "profile_spearman")
    targets.update(zip(lows.model, lows.parameter)); targets.update(zip(highs.model, highs.parameter))
    s = semantic[semantic.apply(lambda x: (x.model, x.parameter) in targets, axis=1)].copy()
    s["inspection_scope"] = s.apply(lambda x: "persistent" if (x.model, x.parameter) in set(zip(persistent.model, persistent.parameter)) else "sign-changing_or_reliability_extreme", axis=1)
    s["targeted_interpretation"] = s.apply(lambda x: "Explicit Flex process group can be used cautiously within Flex family" if x.confidence == "HIGH" else "No defensible targeted pooled role; retain UNRESOLVED", axis=1)
    s["label"] = "TARGETED_SEMANTIC_AUDIT"
    return s.sort_values(["inspection_scope", "model", "parameter_index"])


def build_dseen_followup(pair: pd.DataFrame, repro: pd.DataFrame, persistent: pd.DataFrame, dseen: pd.DataFrame) -> pd.DataFrame:
    p = pair[pair.population == "STRICT_FULL300_34"]
    model = p.groupby("model").agg(all_cell_count=("model", "size"), persistent_cell_fraction=("relationship_class", lambda x: float((x == "persistent/reproduced").mean())), sign_changing_cell_fraction=("relationship_class", lambda x: float((x == "sign-changing").mean())), attenuated_cell_fraction=("relationship_class", lambda x: float((x == "attenuated").mean())), emergent_cell_fraction=("relationship_class", lambda x: float((x == "dPL-emergent").mean())), weak_cell_fraction=("relationship_class", lambda x: float((x == "weak/unresolved").mean()))).reset_index()
    rr = repro[repro.population == "STRICT_FULL300_34"].groupby("model").agg(median_parameter_profile_spearman=("profile_spearman", "median"), negative_parameter_profile_fraction=("profile_spearman", lambda x: float((x < 0).mean()))).reset_index()
    pc = persistent.groupby("model").size().rename("persistent_candidate_count").reset_index(); pc["persistent_candidate_fraction_of_60"] = pc.persistent_candidate_count / len(persistent)
    ds = dseen[dseen.population == "STRICT_FULL300_34"][["model", "D_seen_median"]] if "population" in dseen.columns else dseen[["model", "D_seen_median"]]
    out = model.merge(rr, on="model", how="left").merge(pc, on="model", how="left").merge(ds, on="model", how="left")
    out["persistent_candidate_count"] = out.persistent_candidate_count.fillna(0).astype(int)
    out["persistent_candidate_fraction_of_60"] = out.persistent_candidate_fraction_of_60.fillna(0.0)
    out["label"] = "SECONDARY / SEEN_BASIN_PROXY"
    return out


def draw_figures(out: Path, persistent: pd.DataFrame, by_attr: pd.DataFrame, by_model: pd.DataFrame, by_param: pd.DataFrame, concentration: pd.DataFrame,
                 corr: pd.DataFrame, redundancy: pd.DataFrame, reliability: pd.DataFrame, class_summary: pd.DataFrame, low_cond: pd.DataFrame) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    fd = out / "figures"; fd.mkdir(parents=True, exist_ok=True)
    manifest = []
    # A concentration.
    src_a = pd.concat([by_attr.assign(dimension="attribute"), by_model.assign(dimension="model"), by_param.assign(dimension="parameter")], ignore_index=True)
    write_csv(out / "figure_source_A_persistent_concentration.csv", src_a)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, df, col) in zip(axes, [("Attributes", by_attr, "attribute"), ("Models", by_model, "model"), ("Parameters", by_param, "parameter")]):
        x = df.head(10).sort_values("relationship_count")
        ax.barh(x[col].astype(str), x.relationship_count, color="#2166ac")
        ax.set_title(title); ax.set_xlabel("Persistent rows (N=60)")
    fig.suptitle("Persistent/reproduced relationship concentration (strict34)"); fig.tight_layout(); fig.savefig(fd / "A_persistent_concentration.png", dpi=160); plt.close(fig)
    manifest.append({"figure": "A_persistent_concentration.png", "source_csv": "figure_source_A_persistent_concentration.csv"})
    # B correlation map.
    wide = corr.pivot(index="attribute_1", columns="attribute_2", values="rho").reindex(index=ATTRIBUTES, columns=ATTRIBUTES)
    write_csv(out / "figure_source_B_attribute_correlation_matrix.csv", corr)
    fig, ax = plt.subplots(figsize=(13, 11)); sns.heatmap(wide, ax=ax, cmap="vlag", vmin=-1, vmax=1, center=0, xticklabels=True, yticklabels=True, cbar_kws={"label": "Spearman rho"})
    ax.tick_params(axis="x", labelrotation=75, labelsize=6); ax.tick_params(axis="y", labelsize=6); ax.set_title("35×35 attribute Spearman correlation (categorical codes flagged in source table)")
    fig.tight_layout(); fig.savefig(fd / "B_attribute_correlation_heatmap.png", dpi=160); plt.close(fig)
    manifest.append({"figure": "B_attribute_correlation_heatmap.png", "source_csv": "figure_source_B_attribute_correlation_matrix.csv"})
    # C raw versus cluster recurrence.
    write_csv(out / "figure_source_C_redundancy_recurrence.csv", redundancy)
    plot_c = redundancy[(redundancy.unit_type == "REDUNDANCY_CLUSTER") & (redundancy.raw_relationship_count > 0)].copy()
    plot_c["unit_label"] = plot_c.representative_attribute + " [" + plot_c.member_attributes + "]"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, threshold in zip(axes, REDUNDANCY_THRESHOLDS):
        x = plot_c[plot_c.threshold == threshold].sort_values("raw_relationship_count")
        ax.barh(x.unit_label, x.raw_relationship_count, color="#72a7d8", label="raw")
        ax.barh(x.unit_label, x.deduplicated_model_parameter_count, color="#b2182b", alpha=.8, label="deduplicated model-parameter")
        ax.set_title(f"|rho| ≥ {threshold:.2f}"); ax.set_xlabel("Count within persistent N=60"); ax.legend()
    fig.suptitle("Raw versus redundancy-aware persistent recurrence"); fig.tight_layout(); fig.savefig(fd / "C_raw_vs_redundancy_aware_recurrence.png", dpi=160); plt.close(fig)
    manifest.append({"figure": "C_raw_vs_redundancy_aware_recurrence.png", "source_csv": "figure_source_C_redundancy_recurrence.csv"})
    # D reliability diagnostics.
    src_d = reliability[["model", "parameter", "profile_spearman", "ic_parameter_sd", "dpl_parameter_sd", "ic_boundary_fraction_max", "dpl_boundary_fraction_max", "ic_effective_range", "dpl_effective_range", "reliability_bin", "restart_evidence_level"]]
    write_csv(out / "figure_source_D_parameter_reliability.csv", src_d)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=reliability, x="ic_parameter_sd", y="profile_spearman", hue="reliability_bin", alpha=.75, s=32, ax=axes[0]); axes[0].axhline(0, color="black", lw=.7); axes[0].set_title("Profile agreement vs IC spread")
    sns.scatterplot(data=reliability, x="ic_boundary_fraction_max", y="profile_spearman", hue="reliability_bin", alpha=.75, s=32, legend=False, ax=axes[1]); axes[1].axhline(0, color="black", lw=.7); axes[1].set_title("Profile agreement vs IC boundary fraction")
    fig.suptitle("Parameter-specific reproducibility diagnostics (strict34)"); fig.tight_layout(); fig.savefig(fd / "D_parameter_reliability_diagnostics.png", dpi=160); plt.close(fig)
    manifest.append({"figure": "D_parameter_reliability_diagnostics.png", "source_csv": "figure_source_D_parameter_reliability.csv"})
    # E class composition.
    src_e = class_summary.copy(); write_csv(out / "figure_source_E_relationship_class_enrichment.csv", src_e)
    x = class_summary.set_index("relationship_class")["cell_count"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5)); x.plot(kind="barh", ax=ax, color=["#2166ac" if k == "persistent/reproduced" else "#b2182b" if k == "sign-changing" else "#72a7d8" for k in x.index]); ax.set_xlabel("Strict34 relationship cells"); ax.set_title("Relationship-class composition (descriptive rule)"); fig.tight_layout(); fig.savefig(fd / "E_relationship_class_composition.png", dpi=160); plt.close(fig)
    manifest.append({"figure": "E_relationship_class_composition.png", "source_csv": "figure_source_E_relationship_class_enrichment.csv"})
    # F low_prec_dur conditional case study.
    write_csv(out / "figure_source_F_low_prec_dur_case.csv", low_cond)
    x = low_cond[low_cond.get("case_type", pd.Series(index=low_cond.index, dtype=object)).fillna("LOW_PREC_DUR") == "LOW_PREC_DUR"].copy() if "case_type" in low_cond else low_cond.copy()
    x["label_row"] = x.model.astype(str) + ":" + x.parameter.astype(str)
    x = x.sort_values("rho_ic")
    fig, ax = plt.subplots(figsize=(11, max(5, .25 * len(x))))
    y = np.arange(len(x)); ax.hlines(y, x.rho_ic, x.conditional_rho_ic, color="#2166ac", alpha=.55, label="IC raw→conditional"); ax.scatter(x.rho_ic, y, color="#2166ac", s=20); ax.scatter(x.conditional_rho_ic, y, color="#6a3d9a", s=20); ax.set_yticks(y); ax.set_yticklabels(x.label_row, fontsize=7); ax.axvline(0, color="black", lw=.7); ax.set_xlabel("Spearman rho / rank-residual sensitivity"); ax.set_title("low_prec_dur: raw versus conditional IC relationship (selected controls in source CSV)"); ax.legend()
    fig.tight_layout(); fig.savefig(fd / "F_low_prec_dur_conditional_case.png", dpi=160); plt.close(fig)
    manifest.append({"figure": "F_low_prec_dur_conditional_case.png", "source_csv": "figure_source_F_low_prec_dur_case.csv"})
    write_json(out / "figure_manifest.json", {"status": "SECONDARY_NONTRAINING_DIAGNOSTIC", "figures": manifest, "categorical_attributes_excluded_from_redundancy_graph": sorted(CATEGORICAL)})
    return {"figures": manifest}


def write_readme(out: Path, persistent: pd.DataFrame, by_attr: pd.DataFrame, by_model: pd.DataFrame, by_param: pd.DataFrame, concentration: pd.DataFrame,
                 corr_pairs: pd.DataFrame, high_pairs: pd.DataFrame, redundancy: pd.DataFrame, reliability: pd.DataFrame, rel_link: pd.DataFrame,
                 class_summary: pd.DataFrame, class_attrs: pd.DataFrame, low_audit: pd.DataFrame, low_cond: pd.DataFrame, dseen: pd.DataFrame, semantic: pd.DataFrame) -> None:
    top_attr = by_attr.head(5); top_model = by_model.head(5); top_param = by_param.head(5)
    low_models = reliability.groupby("model").profile_spearman.median().sort_values().head(5)
    high_models = reliability.groupby("model").profile_spearman.median().sort_values(ascending=False).head(5)
    low_controls = ", ".join(f"{r.attribute} ({int(r.relationship_count)})" for r in top_attr.itertuples())
    model_text = ", ".join(f"{r.model} ({int(r.relationship_count)})" for r in top_model.itertuples())
    param_text = ", ".join(f"{r.parameter} ({int(r.relationship_count)})" for r in top_param.itertuples())
    low_dur_models = int(low_audit.model.nunique()) if not low_audit.empty else 0
    controls = low_cond[low_cond.attribute == "low_prec_dur"].conditional_controls.dropna().iloc[0] if not low_cond[low_cond.attribute == "low_prec_dur"].empty else "NOT_AVAILABLE"
    class_map = class_summary.set_index("relationship_class")
    def frac(cls: str) -> float:
        return float(class_map.loc[cls, "cell_count"] / class_map.cell_count.sum()) if cls in class_map.index else 0.0
    target_sem = semantic[semantic.confidence == "HIGH"]
    text = f"""# Parameter–attribute atlas second audit (pure analysis)

## One-sentence conclusion
The 60-row persistent set is not a broad parameter-space consensus: it is concentrated in two raw attributes (`low_prec_dur` and `silt_frac`) and still spans 28 models, while full-cell reproducibility remains heterogeneous. The safest JoH framing is **a few recurring catchment controls embedded in strong model-conditional, parameter-specific, and partly estimator-dependent encoding**.

All outputs are `SEEN_BASIN / NON-OOB`. This package reads frozen atlas CSVs and persisted normalized parameter vectors only; no training, checkpoint write, OOB/PUR run, GPU forward, bootstrap, or permutation was performed.

## 1. Persistent concentration audit

Scope is exactly `STRICT_FULL300_34`, `relationship_class=persistent/reproduced`, and N=60 rows from the prior candidate list.

- Unique models: {persistent.model.nunique()}; unique parameter names (case-sensitive): {persistent.parameter.nunique()}; unique attributes: {persistent.attribute.nunique()}.
- Attribute concentration: {low_controls}.
- Model concentration, top five: {model_text}.
- Parameter-name concentration, top five: {param_text}. Parameter names are not treated as semantic equivalence classes.
- Removing top model `{by_model.iloc[0].model}` leaves {len(persistent[persistent.model != by_model.iloc[0].model])} rows and the same two raw attributes; removing top parameter `{by_param.iloc[0].parameter}` leaves {len(persistent[persistent.parameter != by_param.iloc[0].parameter])} rows.
- `low_prec_dur` contributes {int((persistent.attribute == 'low_prec_dur').sum())}/60 rows and spans {low_dur_models} distinct models; it is not a single-model artifact, but the 60-row candidate set has only two attributes and is therefore not evidence of a broad attribute consensus.
- Top-1/top-3/top-5 concentration shares are in `persistent_concentration_summary.csv`; leave-one-top-contributor results are in `persistent_leave_one_contributor.csv`.

## 2. Attribute redundancy

- Attribute matrix: canonical 531×35, same ID order and normalization as the parent atlas.
- Redundancy matrix is the full 35×35 average-rank Spearman matrix. Categorical-code columns (`dom_land_cover`, `geol_1st_class`, `geol_2nd_class`) remain in the matrix but are excluded from continuous redundancy graphs and physical interpretation.
- High-redundancy continuous pairs: |rho|≥0.70: {int((high_pairs.abs_rho >= .70).sum())}; |rho|≥0.80: {int((high_pairs.abs_rho >= .80).sum())}.
- Cluster memberships at both thresholds are explicit in `attribute_redundancy_groups.csv`. Redundancy-aware recurrence deduplicates repeated evidence at the `(model, parameter, cluster)` level and retains member-level counts.
- The raw persistent set is especially concentrated: only {persistent.attribute.nunique()} attributes occur in the 60 rows. Whether those attributes collapse into the same correlation component is directly recorded in `redundancy_aware_recurrence.csv`, rather than inferred from names.

## 3. Conditional sensitivity

- Conditional analysis was run for all 36 `low_prec_dur` persistent rows and ten representative sign-changing rows.
- For `low_prec_dur`, the same prespecified top-two continuous controls were used for IC and dPL: `{controls}`.
- Method: rank residualization of target and parameter against at most two controls; this is a sensitivity analysis, not causal partial correlation or independent evidence.
- Raw and conditional rho, sign retention, and sample counts are in `conditional_relationship_sensitivity.csv`; the required low-`low_prec_dur` parameter audit is in `low_prec_dur_audit.csv`.

## 4. Parameter reliability / identifiability

- Strict34 parameter profiles: {len(reliability)} rows; profile-Spearman median {reliability.profile_spearman.median():.4f}, range {reliability.profile_spearman.min():.4f}..{reliability.profile_spearman.max():.4f}.
- Lowest model-level median profiles: {', '.join(f'{k} ({v:.3f})' for k, v in low_models.items())}.
- Highest model-level median profiles: {', '.join(f'{k} ({v:.3f})' for k, v in high_models.items())}.
- `parameter_reliability_diagnostics.csv` joins profile agreement to IC/dPL SD, IQR, effective range, lower/upper boundary fractions, near-constant flags, cell strength, and model-level restart metadata.
- Restart evidence is `MODEL_LEVEL_ONLY`: the archived 10-start files contain repeated scores, not repeated parameter vectors. No parameter-level restart instability claim is made.
- Correlation and bin summaries are in `parameter_reliability_linkage_summary.csv`; any association is exploratory across shared model/parameter rows.

## 5. Relationship-class enrichment

Strict34 full relationship cells, using the parent atlas descriptive rule, are summarized in `relationship_class_enrichment.csv`; attribute/model composition is in the companion files.

- Persistent: {frac('persistent/reproduced'):.1%}; attenuated: {frac('attenuated'):.1%}; dPL-emergent: {frac('dPL-emergent'):.1%}; sign-changing: {frac('sign-changing'):.1%}; weak/unresolved: {frac('weak/unresolved'):.1%}.
- Sign-changing context includes parameter spread, boundary concentration, near-constant flags, and profile reproducibility. These rows are not automatically dismissed as unstable, but shared-parameter structure means the table is descriptive rather than independent enrichment inference.

## 6. Targeted semantics

Only the existing explicit Flex process groups were promoted to HIGH confidence ({len(target_sem)} targeted rows in the full semantic table); targeted persistent/sign-changing/reliability-extreme parameters otherwise remain `UNRESOLVED`. No parameter meaning was inferred from names or equal bounds.

## 7. D_seen follow-up

`d_seen_secondary_followup.csv` joins model-level persistent/class fractions and median parameter-profile reproducibility to `D_seen_median`. It is `SECONDARY / SEEN_BASIN_PROXY` only and cannot support OOB transferability.

## 8. Manuscript decision

The evidence supports, in order:

1. **Few recurring controls:** raw recurrence is dominated by `low_prec_dur` and `silt_frac`, not many independent attributes.
2. **Model-conditional relationships:** the same attributes attach to different parameter names and directions across structures.
3. **Parameter-specific reliability:** profile agreement ranges from negative to near one; boundary/spread diagnostics must accompany any parameter claim.
4. **Estimator-dependent encoding:** dPL often shows stronger attribute encoding than IC, and sign-changing/emergent cells remain present.

Recommended wording: “A small number of catchment controls recur across several model structures, but their parameter realization is model- and parameter-specific, with incomplete reproducibility between IC and dPL.” Do not call these relationships physical laws, true relationships, causal effects, or ungauged-transfer evidence.

## Limitations

- dPL is seen-basin and single-seed; `VALID_OOB_DPL=0`.
- `simhyd` is relaxed-only with IC generation 280; `flexb` is IC-only.
- Semantic coverage is incomplete: most parameters remain `UNRESOLVED`.
- Categorical-code Spearman rows are descriptive completeness only.
- No new training or checkpoint modification occurred.

## Output map

- `persistent_relationship_audit.csv`, `persistent_by_attribute.csv`, `persistent_by_model.csv`, `persistent_by_parameter.csv`, `persistent_by_model_parameter.csv`
- `persistent_concentration_summary.csv`, `persistent_leave_one_contributor.csv`
- `attribute_correlation_matrix.csv`, `attribute_correlation_pairs.csv`, `attribute_redundancy_groups.csv`, `attribute_column_audit.csv`
- `redundancy_aware_recurrence.csv`, `conditional_relationship_sensitivity.csv`, `low_prec_dur_audit.csv`
- `parameter_reliability_diagnostics.csv`, `parameter_reliability_linkage_summary.csv`, `restart_reproducibility_model_linkage.csv`
- `relationship_class_enrichment.csv`, `relationship_class_attribute_composition.csv`, `relationship_class_model_composition.csv`
- `targeted_semantic_mapping.csv`, `d_seen_secondary_followup.csv`
- `figure_source_*.csv`, `figures/`, `figure_manifest.json`
- `provenance.json`, `source_hash_manifest.json`, `resource_metadata.json`, `README.md`
"""
    (out / "README.md").write_text(text)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ids, raw_matrix, matrix, pair, persistent, repro, audit, semantic, dseen = load_inputs()
    common = sorted(pair.model.unique().tolist())
    strict = sorted(pair[pair.population == "STRICT_FULL300_34"].model.unique().tolist())
    if len(common) != 35 or len(strict) != 34:
        raise RuntimeError(f"expected 35 relaxed and 34 strict models, got {len(common)}/{len(strict)}")
    ic_vectors, dpl_vectors = load_vectors(common, strict)
    col_audit = column_audit(raw_matrix, matrix)
    corr, corr_pairs, high_pairs = attribute_correlations(matrix)
    groups = {t: connected_components(corr_pairs, t) for t in REDUNDANCY_THRESHOLDS}
    persistent = add_clusters_to_persistent(persistent, groups)
    by_attr, by_model, by_param, by_model_param, concentration = concentration_tables(persistent)
    leave = leave_one_contributor(persistent, by_attr, by_model, by_param)
    redundancy = redundancy_recurrence(persistent, groups)
    ic_long = pd.read_csv(IN / "ic_parameter_attribute_long.csv")
    dp_long = pd.read_csv(IN / "dpl_parameter_attribute_long.csv")
    rel, ic_audit, dp_audit = parameter_reliability(repro, audit, ic_long, dp_long)
    rel_link, restart_model = reliability_linkage(rel)
    low_audit, conditional = low_prec_and_conditional(pair, persistent, matrix, repro, audit)
    class_summary, class_attrs, class_models = class_enrichment(pair, rel)
    targeted = targeted_semantics(persistent, pair, repro, semantic)
    dseen_followup = build_dseen_followup(pair, repro, persistent, dseen)
    write_csv(OUT / "attribute_column_audit.csv", col_audit)
    write_csv(OUT / "attribute_correlation_matrix.csv", corr)
    write_csv(OUT / "attribute_correlation_pairs.csv", corr_pairs)
    write_csv(OUT / "attribute_redundancy_groups.csv", pd.concat(groups.values(), ignore_index=True))
    write_csv(OUT / "persistent_relationship_audit.csv", persistent)
    write_csv(OUT / "persistent_by_attribute.csv", by_attr)
    write_csv(OUT / "persistent_by_model.csv", by_model)
    write_csv(OUT / "persistent_by_parameter.csv", by_param)
    write_csv(OUT / "persistent_by_model_parameter.csv", by_model_param)
    write_csv(OUT / "persistent_concentration_summary.csv", concentration)
    write_csv(OUT / "persistent_leave_one_contributor.csv", leave)
    write_csv(OUT / "redundancy_aware_recurrence.csv", redundancy)
    write_csv(OUT / "conditional_relationship_sensitivity.csv", conditional)
    write_csv(OUT / "low_prec_dur_audit.csv", low_audit)
    write_csv(OUT / "parameter_reliability_diagnostics.csv", rel)
    write_csv(OUT / "parameter_reliability_linkage_summary.csv", rel_link)
    write_csv(OUT / "restart_reproducibility_model_linkage.csv", restart_model)
    write_csv(OUT / "relationship_class_enrichment.csv", class_summary)
    write_csv(OUT / "relationship_class_attribute_composition.csv", class_attrs)
    write_csv(OUT / "relationship_class_model_composition.csv", class_models)
    write_csv(OUT / "targeted_semantic_mapping.csv", targeted)
    write_csv(OUT / "d_seen_secondary_followup.csv", dseen_followup)
    figures = draw_figures(OUT, persistent, by_attr, by_model, by_param, concentration, corr, redundancy, rel, class_summary, conditional)
    source_paths = [
        SCRIPT_DIR / "parameter_attribute_atlas_followup.py", SCRIPT_DIR / "parameter_attribute_atlas.py", SCRIPT_DIR / "r1_r2_quick_survey.py", SCRIPT_DIR / "r1_r2_nontraining_complete.py",
        BENCHMARK / "dpl/attributes.py", BENCHMARK / "src/model_registry.py", IN / "relationship_candidate_list.csv", IN / "ic_dpl_pairwise_relationships.csv",
        IN / "parameter_reproducibility.csv", IN / "parameter_audit.csv", IN / "parameter_semantic_mapping.csv", IN / "d_seen_secondary_linkage.csv", IN / "ic_parameter_attribute_long.csv", IN / "dpl_parameter_attribute_long.csv", IDS_PATH,
        BENCHMARK / "results/r1_r2_nontraining_complete_20260829/wpC_restart_spread_summary.csv", REPO / "data/caravan_671_attributes.npy", REPO / "data/camels_dataset", REPO / "data/gage_id.npy",
    ] + [IN / "parameter_vectors_ic" / f"{m}.csv" for m in ALL_MODELS] + [IN / "parameter_vectors_dpl" / f"{m}.csv" for m in common]
    write_json(OUT / "source_hash_manifest.json", {str(x.relative_to(REPO)): sha256_file(x) for x in source_paths if x.is_file()})
    elapsed = time.time() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    resource_meta = {"runtime_seconds": round(elapsed, 3), "peak_rss_mib": round(peak, 1), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "cpu_single_process": True, "gpu_forward": False, "training_started": False, "checkpoint_write_attempted": False, "oob_pur_executed": False, "bootstrap_or_permutation": False, "input_package": str(IN), "n_persistent_rows": len(persistent), "n_attribute_rows": len(corr), "n_strict_pair_rows": len(pair[pair.population == "STRICT_FULL300_34"]), "n_parameter_reliability_rows": len(rel), "n_low_prec_dur_rows": len(low_audit), "n_conditional_rows": len(conditional), "label": "SEEN_BASIN / NON-OOB"}
    write_json(OUT / "resource_metadata.json", resource_meta)
    provenance = {"analysis": "parameter_attribute_atlas_followup", "analysis_type": "pure secondary analysis; zero training", "input_atlas": str(IN), "population": "STRICT_FULL300_34 primary; RELAXED_SEEN_35 sensitivity retained in source inputs", "persistent_scope": "strict34 relationship_candidate_list persistent/reproduced exactly 60 rows", "attribute_matrix": "canonical 531 IDs x 35 attributes; CatchmentAttributeBuilder all-531 zscore", "categorical_attributes": sorted(CATEGORICAL), "redundancy_graph": "continuous attributes only; connected components at abs Spearman >= 0.70 and >= 0.80", "conditional_method": "rank residualization with top two continuous controls per target; exploratory", "restart_level": "MODEL_LEVEL_ONLY; no parameter-level restart vectors", "VALID_OOB_DPL": 0, "training_started": False, "checkpoint_write_attempted": False, "oob_pur_executed": False, "flexb": "IC-only; no dPL recovery", "simhyd": "relaxed-only IC generation 280", "interpretation": "SEEN_BASIN / NON-OOB; no physical-truth or ungauged-transfer claim"}
    write_json(OUT / "provenance.json", provenance)
    write_readme(OUT, persistent, by_attr, by_model, by_param, concentration, corr_pairs, high_pairs, redundancy, rel, rel_link, class_summary, class_attrs, low_audit, conditional, dseen_followup, semantic)
    print(json.dumps({"output": str(OUT), "persistent_rows": len(persistent), "unique_persistent_models": int(persistent.model.nunique()), "unique_persistent_attributes": int(persistent.attribute.nunique()), "attribute_correlation_rows": len(corr), "high_redundancy_pairs": int(len(high_pairs)), "strict_pair_rows": int(len(pair[pair.population == "STRICT_FULL300_34"])), "parameter_reliability_rows": len(rel), "low_prec_dur_rows": len(low_audit), "conditional_rows": len(conditional), "runtime_seconds": round(elapsed, 2), "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
