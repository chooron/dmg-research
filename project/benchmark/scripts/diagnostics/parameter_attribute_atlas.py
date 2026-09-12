#!/usr/bin/env python3
"""No-training IC/dPL parameter--attribute atlas.

This script consumes frozen IC and dPL artifacts only.  It reconstructs selected
normalized parameter vectors, computes the same basin-wise Spearman estimator for
IC and dPL, and writes a separate atlas package.  Current dPL relationships are
seen-basin descriptive evidence, never formal OOB/PUR evidence.
"""
from __future__ import annotations

import csv
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
import torch

REPO = Path(__file__).resolve().parents[4]
BENCHMARK = REPO / "project/benchmark"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"), str(SCRIPT_DIR)]

from dpl.attributes import CAMELS_35_ATTRIBUTES, CatchmentAttributeBuilder  # noqa: E402
from src.data_selection import load_ids  # noqa: E402
from src.model_registry import NPARAM_INFO_36, get_spec  # noqa: E402
import r1_r2_quick_survey as qs  # noqa: E402

BASE_OUT = BENCHMARK / "results/r1_r2_quick_survey_20260829"
IC_ROOT = BENCHMARK / "results/ic_dpl_aligned_full300_20260819_final"
DPL_ROOT = BENCHMARK / "results/dpl_full_retrain_20260813/auto100"
IDS_PATH = REPO / "data/531sub_id.txt"
OUT = BENCHMARK / "results/parameter_attribute_atlas_20260829"

ALL_MODELS = tuple(NPARAM_INFO_36)
CATEGORICAL = {"dom_land_cover", "geol_1st_class", "geol_2nd_class"}
CONTINUOUS = [a for a in CAMELS_35_ATTRIBUTES if a not in CATEGORICAL]
ATTRIBUTE_TYPE = {a: ("CATEGORICAL_CODE" if a in CATEGORICAL else "CONTINUOUS") for a in CAMELS_35_ATTRIBUTES}
REPRESENTATIVE = ["collie1", "gr4j", "hbv96", "hillslope", "modhydrolog", "mopex4", "topmodel", "xinanjiang"]

# These are prespecified descriptive rules, not scientific cutoffs.
MAIN_HIGH_RHO = 0.20
MAIN_LOW_RHO = 0.10
MAIN_TOP_RANK = 10
SENSITIVITY_HIGH = (0.15, 0.20, 0.30)
SENSITIVITY_LOW = (0.05, 0.10, 0.15)
BOUNDARY_EPS = 0.01
NEAR_CONSTANT_SD = 0.01


def write_csv(path: Path, frame: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame, list):
        frame = pd.DataFrame(frame)
    frame.to_csv(path, index=False, float_format="%.10f")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spearman(x: Any, y: Any) -> tuple[float, int]:
    """Return rank correlation and the exact pairwise finite count."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return float("nan"), n
    ar = pd.Series(a[ok]).rank(method="average").to_numpy()
    br = pd.Series(b[ok]).rank(method="average").to_numpy()
    return float(np.corrcoef(ar, br)[0, 1]), n


def pearson(x: Any, y: Any) -> tuple[float, int]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan"), n
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), n


def sign_label(value: float) -> str:
    if not np.isfinite(value):
        return "UNDEFINED"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def parameter_diagnostic(theta: np.ndarray) -> dict[str, Any]:
    values = np.asarray(theta, dtype=float)
    finite = np.isfinite(values)
    valid = values[finite]
    if valid.size == 0:
        return {"n_basins": 0, "invalid_count": int((~finite).sum()), "min": np.nan, "max": np.nan,
                "mean": np.nan, "sd": np.nan, "iqr": np.nan, "boundary_low_fraction": np.nan,
                "boundary_high_fraction": np.nan, "near_constant": True, "boundary_concentrated": True,
                "diagnostic": "INVALID_OR_EMPTY"}
    low = float(np.mean(valid <= BOUNDARY_EPS))
    high = float(np.mean(valid >= 1.0 - BOUNDARY_EPS))
    sd = float(np.std(valid))
    q25, q75 = np.quantile(valid, [0.25, 0.75])
    constant = np.unique(valid).size <= 1
    near_constant = constant or sd <= NEAR_CONSTANT_SD
    boundary = low >= 0.50 or high >= 0.50
    labels = []
    if int((~finite).sum()):
        labels.append("INVALID_VALUES")
    if constant:
        labels.append("CONSTANT")
    elif near_constant:
        labels.append("NEAR_CONSTANT_SD_LE_0.01")
    if boundary:
        labels.append("BOUNDARY_CONCENTRATED_GE_50PCT_AT_0.01")
    if not labels:
        labels.append("OK")
    return {"n_basins": int(valid.size), "invalid_count": int((~finite).sum()), "min": float(np.min(valid)),
            "max": float(np.max(valid)), "mean": float(np.mean(valid)), "sd": sd,
            "iqr": float(q75 - q25), "boundary_low_fraction": low, "boundary_high_fraction": high,
            "near_constant": bool(near_constant), "boundary_concentrated": bool(boundary),
            "diagnostic": "+".join(labels)}


def current_populations() -> tuple[list[str], list[str], list[str]]:
    dseen = pd.read_csv(BASE_OUT / "d_seen_by_basin.csv")
    common = sorted(dseen.model.unique().tolist())
    strict = sorted([m for m in common if qs.ic_status_generation(m) == 300])
    if len(common) != 35 or len(strict) != 34:
        raise RuntimeError(f"expected relaxed35/strict34, got {len(common)}/{len(strict)}")
    return common, strict, ["flexb"]


def load_vectors(model: str, ids: np.ndarray, attrs: torch.Tensor, device: torch.device, need_dpl: bool) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    latent, ic_meta = qs.ic_latent_and_metadata(model, ids)
    with torch.inference_mode():
        ic_theta = torch.sigmoid(latent.to(device=device, dtype=torch.float32)).detach().cpu().numpy()
    if ic_theta.shape != (len(ids), NPARAM_INFO_36[model]):
        raise RuntimeError(f"{model}: IC theta shape {ic_theta.shape} does not match {(len(ids), NPARAM_INFO_36[model])}")
    dpl_theta = None
    dpath, dmeta = qs.dpl_metadata(model)
    if need_dpl:
        if dpath is None:
            raise RuntimeError(f"{model}: dPL metadata missing in paired population")
        net = qs.model_network(model, dpath, attrs, device)
        with torch.inference_mode():
            dpl_theta = net(attrs).detach().cpu().numpy()
        if dpl_theta.shape != ic_theta.shape:
            raise RuntimeError(f"{model}: dPL theta shape {dpl_theta.shape} != IC {ic_theta.shape}")
        del net
        torch.cuda.empty_cache()
    parameter_meta = {
        "ic_source_used": ic_meta.get("source", ""),
        "ic_canonical_declared": ic_meta.get("canonical_checkpoint", ""),
        "ic_generation": ic_meta.get("generation", ""),
        "ic_selection_rule": "best_training fitness argmax over 10 CMA-ES starts per basin; training fitness only",
        "dpl_source_used": str(dpath) if dpath is not None else "",
        "dpl_checkpoint_epoch": dmeta.get("checkpoint_epoch", ""),
        "dpl_health_best_epoch": dmeta.get("health_best_epoch", ""),
        "dpl_health_stop_epoch": dmeta.get("health_stop_epoch", ""),
        "dpl_selection_rule": "latest saved epoch_*.pt, not heldout/test-selected; archived seed=42",
    }
    return ic_theta, dpl_theta, parameter_meta


def semantic_mapping(model: str) -> list[dict[str, Any]]:
    spec = get_spec(model, device="cpu")
    groups = getattr(spec, "parameter_groups", None) or {}
    reverse: dict[str, str] = {}
    for role, members in groups.items():
        for param in members:
            reverse[param] = role
    rows = []
    source_model = REPO / "dmotpy/models/core" / f"{model}.py"
    for index, param in enumerate(spec.parameter_names):
        if param in reverse:
            rows.append({
                "model": model, "parameter_index": index, "parameter": param,
                "role": reverse[param], "evidence_source": "project/benchmark/src/model_registry.py:FLEX_PROCESS_GROUPS",
                "confidence": "HIGH", "mapping_scope": "FLEX_FAMILY_ONLY",
                "notes": "Explicit repository process group; not a universal cross-model semantic equivalence.",
            })
        else:
            rows.append({
                "model": model, "parameter_index": index, "parameter": param,
                "role": "UNRESOLVED", "evidence_source": str(source_model.relative_to(REPO)) if source_model.is_file() else "model_registry.py",
                "confidence": "UNRESOLVED", "mapping_scope": "NONE",
                "notes": "No centralized parameter-role metadata sufficient for a defensible pooled role mapping; names/bounds alone were not used.",
            })
    return rows


def build_long_tables(model_arrays: dict[str, dict[str, Any]], ids: np.ndarray, attribute_matrix: np.ndarray,
                      common: list[str], strict: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ic_rows: list[dict[str, Any]] = []
    dpl_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    semantic = pd.DataFrame([row for model in ALL_MODELS for row in semantic_mapping(model)])
    role_lookup = {(r.model, r.parameter): (r.role, r.confidence) for r in semantic.itertuples()}
    for model in ALL_MODELS:
        item = model_arrays[model]
        spec = get_spec(model, device="cpu")
        bounds = [[float(a), float(b)] for a, b in spec.bounds.detach().cpu().numpy().tolist()]
        ic_theta = item["ic_theta"]
        dpl_theta = item["dpl_theta"]
        populations_ic = ["IC_ONLY_FLEXB"] if model == "flexb" else (["RELAXED_SEEN_35", "STRICT_FULL300_34"] if model in strict else (["RELAXED_SEEN_35"] if model in common else []))
        populations_dpl = ["RELAXED_SEEN_35", "STRICT_FULL300_34"] if model in strict else (["RELAXED_SEEN_35"] if model in common else [])
        for method, theta, populations in (("IC", ic_theta, populations_ic), ("dPL", dpl_theta, populations_dpl)):
            if theta is None:
                continue
            diagnostics = [parameter_diagnostic(theta[:, p]) for p in range(theta.shape[1])]
            for p, param in enumerate(spec.parameter_names):
                diag = diagnostics[p]
                role, role_conf = role_lookup[(model, param)]
                for pop in populations:
                    audit_rows.append({
                        "model": model, "population": pop, "method": method, "parameter_index": p,
                        "parameter": param, "lower_bound": bounds[p][0], "upper_bound": bounds[p][1],
                        "parameter_bounds": json.dumps(bounds[p]), "routed_kind": spec.routed_kind,
                        "n_basins": diag["n_basins"], "invalid_count": diag["invalid_count"],
                        "normalized_min": diag["min"], "normalized_max": diag["max"],
                        "normalized_mean": diag["mean"], "normalized_sd": diag["sd"], "normalized_iqr": diag["iqr"],
                        "boundary_low_fraction": diag["boundary_low_fraction"], "boundary_high_fraction": diag["boundary_high_fraction"],
                        "near_constant": diag["near_constant"], "boundary_concentrated": diag["boundary_concentrated"],
                        "parameter_boundary_diagnostic": diag["diagnostic"], "role_if_available": role,
                        "role_confidence": role_conf,
                        "parameter_transform": "sigmoid(best_latent)" if method == "IC" else "dPL network sigmoid output",
                        "physical_mapping": "linear" if method == "IC" else "auto (linear or monotone log by bound span)",
                        "checkpoint_selection": item["meta"]["ic_selection_rule"] if method == "IC" else item["meta"]["dpl_selection_rule"],
                        "label": "SEEN_BASIN_PROXY" if method == "dPL" else "IC_DESCRIPTIVE",
                    })
            for p, param in enumerate(spec.parameter_names):
                rhos: list[float] = []
                ns: list[int] = []
                for a, attr in enumerate(CAMELS_35_ATTRIBUTES):
                    rho, n = spearman(attribute_matrix[:, a], theta[:, p])
                    rhos.append(rho); ns.append(n)
                ranks = pd.Series(-np.abs(np.asarray(rhos, dtype=float))).rank(method="min", na_option="keep")
                diag = diagnostics[p]
                for a, attr in enumerate(CAMELS_35_ATTRIBUTES):
                    row = {
                        "model": model, "method": method, "parameter_index": p, "parameter": param,
                        "attribute": attr, "attribute_type": ATTRIBUTE_TYPE[attr],
                        "rho_ic" if method == "IC" else "rho_dpl": rhos[a],
                        "abs_rho_ic" if method == "IC" else "abs_rho_dpl": abs(rhos[a]) if np.isfinite(rhos[a]) else np.nan,
                        "rank_within_parameter": int(ranks.iloc[a]) if pd.notna(ranks.iloc[a]) else np.nan,
                        "parameter_bounds": json.dumps(bounds[p]), "lower_bound": bounds[p][0], "upper_bound": bounds[p][1],
                        "parameter_boundary_diagnostic": diag["diagnostic"], "n_basins": len(ids), "n_pairwise": ns[a],
                        "parameter_coordinate": "normalized sigmoid coordinate",
                        "attribute_coordinate": "all-531 normalized zscore matrix; rank-equivalent to finite raw monotone columns",
                        "label": "SEEN_BASIN_PROXY" if method == "dPL" else "IC_DESCRIPTIVE",
                    }
                    (ic_rows if method == "IC" else dpl_rows).append({**row, "population": "_POPULATION_PLACEHOLDER"})
    ic = pd.DataFrame(ic_rows)
    dpl = pd.DataFrame(dpl_rows)
    # Duplicate relationship values into each explicitly requested population; values use identical 531 IDs.
    def expand(frame: pd.DataFrame, method: str) -> pd.DataFrame:
        expanded: list[pd.DataFrame] = []
        for model in frame.model.unique():
            base = frame[frame.model == model].drop(columns=["population"])
            if method == "IC":
                pops = ["IC_ONLY_FLEXB"] if model == "flexb" else (["RELAXED_SEEN_35", "STRICT_FULL300_34"] if model in strict else (["RELAXED_SEEN_35"] if model in common else []))
            else:
                pops = ["RELAXED_SEEN_35", "STRICT_FULL300_34"] if model in strict else (["RELAXED_SEEN_35"] if model in common else [])
            for pop in pops:
                expanded.append(base.assign(population=pop))
        return pd.concat(expanded, ignore_index=True) if expanded else pd.DataFrame()
    return expand(ic, "IC"), expand(dpl, "dPL"), pd.DataFrame(audit_rows)


def merge_pairwise(ic: pd.DataFrame, dpl: pd.DataFrame, common: list[str], strict: list[str]) -> pd.DataFrame:
    left = ic[ic.population.isin(["RELAXED_SEEN_35", "STRICT_FULL300_34"])].copy()
    right = dpl[dpl.population.isin(["RELAXED_SEEN_35", "STRICT_FULL300_34"])].copy()
    keys = ["population", "model", "parameter_index", "parameter", "attribute", "attribute_type", "n_basins", "n_pairwise"]
    lcols = keys + ["rho_ic", "abs_rho_ic", "rank_within_parameter", "parameter_bounds", "lower_bound", "upper_bound", "parameter_boundary_diagnostic"]
    rcols = keys + ["rho_dpl", "abs_rho_dpl", "rank_within_parameter", "parameter_boundary_diagnostic"]
    pair = left[lcols].merge(right[rcols], on=keys, suffixes=("_ic", "_dpl"), validate="one_to_one")
    pair = pair.rename(columns={"rank_within_parameter_ic": "rank_ic", "rank_within_parameter_dpl": "rank_dpl",
                                "parameter_boundary_diagnostic_ic": "parameter_boundary_diagnostic_ic",
                                "parameter_boundary_diagnostic_dpl": "parameter_boundary_diagnostic_dpl"})
    pair["delta_abs_rho"] = pair.abs_rho_dpl - pair.abs_rho_ic
    pair["sign_ic"] = pair.rho_ic.map(sign_label)
    pair["sign_dpl"] = pair.rho_dpl.map(sign_label)
    pair["same_sign"] = (pair.rho_ic.notna() & pair.rho_dpl.notna() & (pair.sign_ic == pair.sign_dpl))
    pair["both_nontrivial"] = (pair.abs_rho_ic >= MAIN_LOW_RHO) & (pair.abs_rho_dpl >= MAIN_LOW_RHO)
    pair["same_sign_nontrivial"] = pair.both_nontrivial & (pair.sign_ic == pair.sign_dpl)
    pair["class_rule"] = ("persistent: same sign, abs rho >= 0.20 on both sides, and rank <= 10 on both; "
                           "attenuated: IC abs rho >= 0.20 and dPL abs rho < 0.10; "
                           "dPL-emergent: dPL abs rho >= 0.20 and IC abs rho < 0.10; "
                           "sign-changing: opposite signs with abs rho >= 0.10 on both; otherwise weak/unresolved")
    conditions = [
        pair.both_nontrivial & (pair.sign_ic != pair.sign_dpl),
        (pair.abs_rho_ic >= MAIN_HIGH_RHO) & (pair.abs_rho_dpl < MAIN_LOW_RHO),
        (pair.abs_rho_dpl >= MAIN_HIGH_RHO) & (pair.abs_rho_ic < MAIN_LOW_RHO),
        (pair.abs_rho_ic >= MAIN_HIGH_RHO) & (pair.abs_rho_dpl >= MAIN_HIGH_RHO) & pair.same_sign & (pair.rank_ic <= MAIN_TOP_RANK) & (pair.rank_dpl <= MAIN_TOP_RANK),
    ]
    pair["relationship_class"] = np.select(conditions, ["sign-changing", "attenuated", "dPL-emergent", "persistent/reproduced"], default="weak/unresolved")
    pair["label"] = "SEEN_BASIN_PROXY; descriptive cross-estimator comparison"
    return pair


def model_and_attribute_summaries(ic: pd.DataFrame, dpl: pd.DataFrame, pair: pd.DataFrame, semantic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    for method, frame, rho_col, abs_col in (("IC", ic, "rho_ic", "abs_rho_ic"), ("dPL", dpl, "rho_dpl", "abs_rho_dpl")):
        for (pop, model), g in frame.groupby(["population", "model"], sort=True):
            cont = g[g.attribute_type == "CONTINUOUS"]
            dominant = g.loc[g[abs_col].idxmax()] if g[abs_col].notna().any() else None
            per_param = g.groupby("parameter")[abs_col].max()
            model_rows.append({
                "population": pop, "method": method, "model": model, "parameter_count": g.parameter.nunique(),
                "attribute_count": g.attribute.nunique(), "continuous_attribute_count": cont.attribute.nunique(),
                "valid_cell_count": int(g[rho_col].notna().sum()), "continuous_valid_cell_count": int(cont[rho_col].notna().sum()),
                "median_abs_rho_all": float(g[abs_col].median()), "iqr_abs_rho_all": float(g[abs_col].quantile(.75) - g[abs_col].quantile(.25)),
                "median_abs_rho_continuous": float(cont[abs_col].median()), "p95_abs_rho_continuous": float(cont[abs_col].quantile(.95)),
                "high_abs_rho_ge_0.20_count_continuous": int((cont[abs_col] >= MAIN_HIGH_RHO).sum()),
                "parameters_with_high_control": int((per_param >= MAIN_HIGH_RHO).sum()),
                "dominant_attribute_all": dominant.attribute if dominant is not None else "UNRESOLVED",
                "dominant_attribute_abs_rho_all": float(dominant[abs_col]) if dominant is not None else np.nan,
                "label": "IC_DESCRIPTIVE" if method == "IC" else "SEEN_BASIN_PROXY",
            })
    model_summary = pd.DataFrame(model_rows)
    pair_rows: list[dict[str, Any]] = []
    for pop, g in pair.groupby(["population", "model"], sort=True):
        population, model = pop
        both = g[g.both_nontrivial]
        pair_rows.append({
            "population": population, "model": model, "parameter_count": g.parameter.nunique(), "attribute_count": g.attribute.nunique(),
            "pair_count": len(g), "valid_pair_count": int((g.rho_ic.notna() & g.rho_dpl.notna()).sum()),
            "flattened_profile_pearson": pearson(g.rho_ic, g.rho_dpl)[0],
            "flattened_profile_spearman": spearman(g.rho_ic, g.rho_dpl)[0],
            "sign_agreement_all": float((g.sign_ic == g.sign_dpl).mean()),
            "sign_agreement_nontrivial": float((both.sign_ic == both.sign_dpl).mean()) if len(both) else np.nan,
            "persistent_fraction": float((g.relationship_class == "persistent/reproduced").mean()),
            "attenuated_fraction": float((g.relationship_class == "attenuated").mean()),
            "dpl_emergent_fraction": float((g.relationship_class == "dPL-emergent").mean()),
            "sign_changing_fraction": float((g.relationship_class == "sign-changing").mean()),
            "weak_unresolved_fraction": float((g.relationship_class == "weak/unresolved").mean()),
            "label": "SEEN_BASIN_PROXY; same-estimator profile comparison",
        })
    model_pair_summary = pd.DataFrame(pair_rows)
    semantic_lookup = semantic[["model", "parameter", "role", "confidence"]].rename(columns={"role": "parameter_role", "confidence": "role_confidence"})
    pair = pair.merge(semantic_lookup, on=["model", "parameter"], how="left", validate="many_to_one")
    attr_rows: list[dict[str, Any]] = []
    for method, frame, rho_col, abs_col in (("IC", ic, "rho_ic", "abs_rho_ic"), ("dPL", dpl, "rho_dpl", "abs_rho_dpl")):
        for pop, g in frame.groupby("population", sort=True):
            for attr, ag in g.groupby("attribute", sort=False):
                cont = ATTRIBUTE_TYPE[attr] == "CONTINUOUS"
                ranks = ag[ag["rank_within_parameter"] <= 5]
                attr_rows.append({
                    "population": pop, "method": method, "attribute": attr, "attribute_type": ATTRIBUTE_TYPE[attr],
                    "n_model_parameter_pairs": int(len(ag)), "n_valid": int(ag[rho_col].notna().sum()),
                    "median_rho": float(ag[rho_col].median()), "median_abs_rho": float(ag[abs_col].median()),
                    "p75_abs_rho": float(ag[abs_col].quantile(.75)), "max_abs_rho": float(ag[abs_col].max()),
                    "high_abs_rho_ge_0.20_count": int((ag[abs_col] >= MAIN_HIGH_RHO).sum()),
                    "top5_parameter_control_count": int(len(ranks)), "top5_model_count": int(ranks.model.nunique()),
                    "positive_fraction": float((ag[rho_col] > 0).mean()), "negative_fraction": float((ag[rho_col] < 0).mean()),
                    "interpretation_scope": "CONTINUOUS_ATTRIBUTE_SUMMARY" if cont else "CATEGORICAL_CODE_DESCRIPTIVE_ONLY",
                    "label": "IC_DESCRIPTIVE" if method == "IC" else "SEEN_BASIN_PROXY",
                })
    return model_summary, model_pair_summary, pd.DataFrame(attr_rows), pair


def parameter_reproducibility(pair: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (pop, model, p), g in pair.groupby(["population", "model", "parameter"], sort=True):
        both = g[g.both_nontrivial]
        top_ic = set(g.nsmallest(5, "rank_ic").attribute)
        top_dp = set(g.nsmallest(5, "rank_dpl").attribute)
        rows.append({
            "population": pop, "model": model, "parameter_index": int(g.parameter_index.iloc[0]), "parameter": p,
            "n_attributes": int(len(g)), "profile_pearson": pearson(g.rho_ic, g.rho_dpl)[0],
            "profile_spearman": spearman(g.rho_ic, g.rho_dpl)[0],
            "sign_agreement_all": float((g.sign_ic == g.sign_dpl).mean()),
            "sign_agreement_nontrivial": float((both.sign_ic == both.sign_dpl).mean()) if len(both) else np.nan,
            "dominant_attribute_IC": g.loc[g.abs_rho_ic.idxmax(), "attribute"] if g.abs_rho_ic.notna().any() else "UNRESOLVED",
            "dominant_attribute_dPL": g.loc[g.abs_rho_dpl.idxmax(), "attribute"] if g.abs_rho_dpl.notna().any() else "UNRESOLVED",
            "dominant_control_agreement": bool(g.loc[g.abs_rho_ic.idxmax(), "attribute"] == g.loc[g.abs_rho_dpl.idxmax(), "attribute"]) if g.abs_rho_ic.notna().any() and g.abs_rho_dpl.notna().any() else False,
            "top5_overlap_count": len(top_ic & top_dp), "top5_overlap_jaccard": len(top_ic & top_dp) / len(top_ic | top_dp) if (top_ic | top_dp) else np.nan,
            "mean_abs_rho_ic": float(g.abs_rho_ic.mean()), "mean_abs_rho_dpl": float(g.abs_rho_dpl.mean()),
            "delta_abs_rho_mean": float(g.abs_rho_dpl.mean() - g.abs_rho_ic.mean()),
            "parameter_boundary_diagnostic_ic": g.parameter_boundary_diagnostic_ic.iloc[0],
            "parameter_boundary_diagnostic_dpl": g.parameter_boundary_diagnostic_dpl.iloc[0],
            "role_if_available": g.parameter_role.iloc[0], "role_confidence": g.role_confidence.iloc[0],
            "label": "SEEN_BASIN_PROXY; parameter-specific reproducibility",
        })
    return pd.DataFrame(rows)


def dseen_linkage(pair_model: pd.DataFrame, dseen_path: Path) -> pd.DataFrame:
    dseen = pd.read_csv(dseen_path)
    med = dseen.groupby("model").D_seen.median().rename("D_seen_median")
    out = pair_model.join(med, on="model")
    out["secondary_status"] = "SECONDARY / SEEN_BASIN_PROXY"
    return out


def class_sensitivity(pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for high in SENSITIVITY_HIGH:
        for low in SENSITIVITY_LOW:
            for pop, g in pair.groupby("population", sort=True):
                x = g.copy()
                cond = [
                    (x.abs_rho_ic >= low) & (x.abs_rho_dpl >= low) & (x.sign_ic != x.sign_dpl),
                    (x.abs_rho_ic >= high) & (x.abs_rho_dpl < low),
                    (x.abs_rho_dpl >= high) & (x.abs_rho_ic < low),
                    (x.abs_rho_ic >= high) & (x.abs_rho_dpl >= high) & (x.sign_ic == x.sign_dpl) & (x.rank_ic <= MAIN_TOP_RANK) & (x.rank_dpl <= MAIN_TOP_RANK),
                ]
                classes = np.select(cond, ["sign-changing", "attenuated", "dPL-emergent", "persistent/reproduced"], default="weak/unresolved")
                for cls in ["persistent/reproduced", "attenuated", "dPL-emergent", "sign-changing", "weak/unresolved"]:
                    rows.append({"population": pop, "high_rho_threshold": high, "low_rho_threshold": low, "relationship_class": cls,
                                 "count": int(np.sum(classes == cls)), "fraction": float(np.mean(classes == cls)),
                                 "rank_cutoff": MAIN_TOP_RANK, "status": "DESCRIPTIVE_SENSITIVITY_ONLY"})
    return pd.DataFrame(rows)


def candidates(pair: pd.DataFrame, semantic: pd.DataFrame) -> pd.DataFrame:
    strict = pair[pair.population == "STRICT_FULL300_34"].copy()
    recurring = strict[strict.relationship_class == "persistent/reproduced"].groupby("attribute").model.nunique().rename("persistent_attribute_model_count")
    strict = strict.join(recurring, on="attribute")
    strict["persistent_attribute_model_count"] = strict.persistent_attribute_model_count.fillna(0).astype(int)
    picks: list[pd.DataFrame] = []
    configs = {
        "persistent/reproduced": (60, ["persistent_attribute_model_count", "abs_rho_ic", "abs_rho_dpl"], [False, False, False]),
        "attenuated": (25, ["abs_rho_ic", "delta_abs_rho"], [False, True]),
        "dPL-emergent": (25, ["abs_rho_dpl", "delta_abs_rho"], [False, False]),
        "sign-changing": (25, ["abs_rho_ic", "abs_rho_dpl"], [False, False]),
    }
    for cls, (limit, cols, ascending) in configs.items():
        x = strict[strict.relationship_class == cls].sort_values(cols, ascending=ascending, kind="mergesort").head(limit).copy()
        x["candidate_set"] = cls
        x["candidate_rank"] = np.arange(1, len(x) + 1)
        picks.append(x)
    if not picks:
        return pd.DataFrame()
    out = pd.concat(picks, ignore_index=True)
    def why(cls: str) -> str:
        return {"persistent/reproduced": "Same direction and high rank/strength under both estimators; a direct OOB/PUR challenge target.",
                "attenuated": "IC attribute control weakens under dPL; tests estimator-constraint sensitivity and seed stability.",
                "dPL-emergent": "dPL encoding is strong without a corresponding IC profile; tests mapping emergence versus OOB reproducibility.",
                "sign-changing": "Opposite estimator directions at non-trivial magnitude; high-value semantic and OOB audit case."}[cls]
    out["why_it_is_informative"] = out.relationship_class.map(why)
    out["future_validation_target"] = out.relationship_class.map({
        "persistent/reproduced": "OOB;PUR;seed;threshold sensitivity",
        "attenuated": "OOB;seed;semantic audit",
        "dPL-emergent": "OOB;seed;mapping audit",
        "sign-changing": "OOB;PUR;semantic audit",
    })
    out["interpretation"] = "SEEN_BASIN_PROXY candidate only; not a truth label or physical law"
    out["parameter_role_if_available"] = out.parameter_role
    out["label"] = "FUTURE_VALIDATION_CANDIDATE / SEEN_BASIN_PROXY"
    fields = ["population", "model", "parameter_index", "parameter", "parameter_role_if_available", "role_confidence", "attribute", "attribute_type",
              "rho_ic", "rho_dpl", "abs_rho_ic", "abs_rho_dpl", "rank_ic", "rank_dpl", "delta_abs_rho", "relationship_class",
              "persistent_attribute_model_count", "candidate_set", "candidate_rank", "why_it_is_informative", "future_validation_target", "interpretation", "label"]
    return out[fields].sort_values(["candidate_set", "candidate_rank", "model", "parameter", "attribute"])


def population_manifest(ic: pd.DataFrame, dpl: pd.DataFrame, pair: pd.DataFrame, common: list[str], strict: list[str]) -> pd.DataFrame:
    rows = [
        {"population": "REGISTRY_36", "primary": False, "ic_models": 36, "dpl_models": 35, "paired_models": 34, "basins_per_model": 531, "status": "inventory only", "label": "REGISTRY"},
        {"population": "STRICT_FULL300_34", "primary": True, "ic_models": len(strict), "dpl_models": len(strict), "paired_models": len(strict), "basins_per_model": 531, "status": "IC generation 300; excludes simhyd gen280 and flexb", "label": "SEEN_BASIN_PROXY"},
        {"population": "RELAXED_SEEN_35", "primary": False, "ic_models": len(common), "dpl_models": len(common), "paired_models": len(common), "basins_per_model": 531, "status": "includes simhyd IC generation 280; excludes flexb dPL", "label": "SEEN_BASIN_PROXY"},
        {"population": "IC_ONLY_FLEXB", "primary": False, "ic_models": 1, "dpl_models": 0, "paired_models": 0, "basins_per_model": 531, "status": "IC descriptive atlas only; missing/incomplete dPL", "label": "IC_DESCRIPTIVE_ONLY"},
    ]
    out = pd.DataFrame(rows)
    out["ic_relationship_rows"] = out.population.map(ic.groupby("population").size()).fillna(0).astype(int)
    out["dpl_relationship_rows"] = out.population.map(dpl.groupby("population").size()).fillna(0).astype(int)
    out["pair_relationship_rows"] = out.population.map(pair.groupby("population").size()).fillna(0).astype(int)
    return out


def source_tables(out: Path, ic: pd.DataFrame, dpl: pd.DataFrame, pair: pd.DataFrame, repro: pd.DataFrame, attrs: pd.DataFrame, reps: list[str]) -> None:
    strict_ic = ic[(ic.population == "STRICT_FULL300_34") & ic.model.isin(reps)]
    strict_dpl = dpl[(dpl.population == "STRICT_FULL300_34") & dpl.model.isin(reps)]
    strict_pair = pair[(pair.population == "STRICT_FULL300_34") & pair.model.isin(reps)]
    strict_repro = repro[(repro.population == "STRICT_FULL300_34") & repro.model.isin(reps)]
    strict_attrs = attrs[(attrs.population == "STRICT_FULL300_34") & attrs.attribute_type == "CONTINUOUS"]
    write_csv(out / "figure_source_ic_atlas_representative.csv", strict_ic)
    write_csv(out / "figure_source_ic_dpl_atlas_representative.csv", strict_pair)
    write_csv(out / "figure_source_pairwise_scatter_strict34.csv", pair[pair.population == "STRICT_FULL300_34"])
    write_csv(out / "figure_source_parameter_reproducibility_strict34.csv", repro[repro.population == "STRICT_FULL300_34"])
    write_csv(out / "figure_source_attribute_crossmodel_strict34.csv", strict_attrs)


def draw_figures(out: Path, ic: pd.DataFrame, dpl: pd.DataFrame, pair: pd.DataFrame, repro: pd.DataFrame, attrs: pd.DataFrame, strict: list[str]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    fig_dir = out / "figures"
    all_dir = fig_dir / "ic_atlas_all"
    all_dir.mkdir(parents=True, exist_ok=True)
    reps = [m for m in REPRESENTATIVE if m in strict]
    source_tables(out, ic, dpl, pair, repro, attrs, reps)
    figures: list[dict[str, Any]] = []
    attr_order = CONTINUOUS

    def matrix(frame: pd.DataFrame, model: str, value: str) -> pd.DataFrame:
        x = frame[(frame.population == "STRICT_FULL300_34") & (frame.model == model) & frame.attribute.isin(attr_order)]
        return x.pivot(index="parameter", columns="attribute", values=value).reindex(columns=attr_order)

    # A: representative IC atlas.
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), squeeze=False)
    for ax, model in zip(axes.flat, reps):
        m = matrix(ic, model, "rho_ic")
        sns.heatmap(m, ax=ax, cmap="vlag", vmin=-1, vmax=1, center=0, cbar=False, xticklabels=True, yticklabels=True)
        ax.set_title(model); ax.tick_params(axis="x", labelrotation=75, labelsize=5); ax.tick_params(axis="y", labelsize=6)
        ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("IC parameter–attribute atlas (strict34; continuous attributes; normalized-coordinate Spearman)")
    fig.tight_layout(); fig.savefig(fig_dir / "A_ic_atlas_representative.png", dpi=160); plt.close(fig)
    figures.append({"figure": "A_ic_atlas_representative.png", "source_table": "figure_source_ic_atlas_representative.csv", "scope": "representative strict34 models"})

    # All-model IC atlas, including IC-only flexb and relaxed-only simhyd.
    for model in ALL_MODELS:
        pop = "IC_ONLY_FLEXB" if model == "flexb" else ("STRICT_FULL300_34" if model in strict else "RELAXED_SEEN_35")
        x = ic[(ic.population == pop) & (ic.model == model) & ic.attribute.isin(attr_order)]
        if x.empty:
            continue
        m = x.pivot(index="parameter", columns="attribute", values="rho_ic").reindex(columns=attr_order)
        fig, ax = plt.subplots(figsize=(13, max(2.5, 0.34 * len(m))))
        sns.heatmap(m, ax=ax, cmap="vlag", vmin=-1, vmax=1, center=0, xticklabels=True, yticklabels=True, cbar_kws={"label": "Spearman rho"})
        ax.set_title(f"IC atlas: {model} ({pop})"); ax.tick_params(axis="x", labelrotation=75, labelsize=6); ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout(); fig.savefig(all_dir / f"{model}.png", dpi=130); plt.close(fig)
    figures.append({"figure": "ic_atlas_all/*.png", "source_table": "ic_parameter_attribute_long.csv", "scope": "all available IC models; appendix atlas"})

    # B: paired IC/dPL representative atlas with shared color scale.
    paired_reps = [m for m in ["gr4j", "hbv96", "mopex4", "xinanjiang"] if m in strict]
    fig, axes = plt.subplots(len(paired_reps), 2, figsize=(16, 3.1 * len(paired_reps)), squeeze=False)
    for i, model in enumerate(paired_reps):
        for j, (method, frame, value) in enumerate([("IC", ic, "rho_ic"), ("dPL", dpl, "rho_dpl")]):
            m = matrix(frame, model, value)
            sns.heatmap(m, ax=axes[i, j], cmap="vlag", vmin=-1, vmax=1, center=0, cbar=(j == 1), cbar_kws={"label": "Spearman rho"} if j == 1 else None)
            axes[i, j].set_title(f"{model} — {method}"); axes[i, j].tick_params(axis="x", labelrotation=75, labelsize=5); axes[i, j].tick_params(axis="y", labelsize=6)
            axes[i, j].set_xlabel(""); axes[i, j].set_ylabel("")
    fig.suptitle("IC vs dPL parameter–attribute profiles (strict34; shared scale)")
    fig.tight_layout(); fig.savefig(fig_dir / "B_ic_vs_dpl_atlas_representative.png", dpi=160); plt.close(fig)
    figures.append({"figure": "B_ic_vs_dpl_atlas_representative.png", "source_table": "figure_source_ic_dpl_atlas_representative.csv", "scope": "four representative strict34 paired models"})

    # C: all strict pair cells.
    p = pair[pair.population == "STRICT_FULL300_34"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sns.scatterplot(data=p, x="rho_ic", y="rho_dpl", hue="relationship_class", style="attribute_type", alpha=.35, s=14, linewidth=0, ax=ax)
    ax.plot([-1, 1], [-1, 1], color="black", lw=.8, ls="--"); ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="rho_IC", ylabel="rho_dPL")
    ax.set_title("IC–dPL pairwise relationship reproducibility (strict34)")
    fig.tight_layout(); fig.savefig(fig_dir / "C_pairwise_ic_vs_dpl_scatter_strict34.png", dpi=160); plt.close(fig)
    figures.append({"figure": "C_pairwise_ic_vs_dpl_scatter_strict34.png", "source_table": "figure_source_pairwise_scatter_strict34.csv", "scope": "all strict34 model–parameter–attribute pairs"})

    # D: parameter-level profile agreement.
    r = repro[repro.population == "STRICT_FULL300_34"].copy(); r["model_parameter"] = r.model + " :: " + r.parameter
    r = r.sort_values(["model", "parameter_index"])
    fig, ax = plt.subplots(figsize=(12, max(6, .12 * len(r))))
    sns.scatterplot(data=r, x="profile_spearman", y="model_parameter", hue="dominant_control_agreement", palette={True: "#2166ac", False: "#b2182b"}, s=28, ax=ax)
    ax.axvline(0, color="black", lw=.7); ax.set_xlabel("Parameter-profile Spearman (35 attributes)"); ax.set_ylabel("")
    ax.set_title("Parameter-specific IC–dPL reproducibility (strict34)")
    fig.tight_layout(); fig.savefig(fig_dir / "D_parameter_profile_reproducibility_strict34.png", dpi=160); plt.close(fig)
    figures.append({"figure": "D_parameter_profile_reproducibility_strict34.png", "source_table": "figure_source_parameter_reproducibility_strict34.csv", "scope": "strict34 model × parameter"})

    # E: attribute-level cross-model summary; continuous only.
    a = attrs[(attrs.population == "STRICT_FULL300_34") & (attrs.attribute_type == "CONTINUOUS")].copy()
    piv = a.pivot(index="attribute", columns="method", values="top5_model_count").fillna(0)
    piv = piv.sort_values([c for c in ["IC", "dPL"] if c in piv], ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    piv.plot(kind="barh", ax=ax, color={"IC": "#2166ac", "dPL": "#b2182b"}, width=.8)
    ax.set_xlabel("Number of models where attribute is a top-5 parameter control"); ax.set_ylabel("")
    ax.set_title("Cross-model recurring attribute controls (strict34; continuous only)")
    fig.tight_layout(); fig.savefig(fig_dir / "E_crossmodel_attribute_summary_strict34.png", dpi=160); plt.close(fig)
    figures.append({"figure": "E_crossmodel_attribute_summary_strict34.png", "source_table": "figure_source_attribute_crossmodel_strict34.csv", "scope": "strict34 attribute summary"})
    write_json(out / "figure_manifest.json", {"status": "DIAGNOSTIC_ATLAS_ONLY", "figures": figures, "continuous_attributes_in_figures": attr_order, "categorical_rows_in_tables": sorted(CATEGORICAL)})
    return {"figures": figures, "representative_models": reps, "all_ic_atlas_count": len(list(all_dir.glob("*.png")))}


def write_readme(out: Path, common: list[str], strict: list[str], pop: pd.DataFrame, icm: pd.DataFrame, dplm: pd.DataFrame,
                 pairm: pd.DataFrame, repro: pd.DataFrame, attrs: pd.DataFrame, cand: pd.DataFrame, audit: pd.DataFrame) -> None:
    s_pair = pairm[pairm.population == "STRICT_FULL300_34"].iloc[0]
    s_repro = repro[repro.population == "STRICT_FULL300_34"]
    s_attr_ic = attrs[(attrs.population == "STRICT_FULL300_34") & (attrs.method == "IC") & (attrs.attribute_type == "CONTINUOUS")].sort_values(["top5_model_count", "median_abs_rho"], ascending=False).head(8)
    s_attr_dp = attrs[(attrs.population == "STRICT_FULL300_34") & (attrs.method == "dPL") & (attrs.attribute_type == "CONTINUOUS")].sort_values(["top5_model_count", "median_abs_rho"], ascending=False).head(8)
    strongest_ic = icm[(icm.population == "STRICT_FULL300_34") & (icm.method == "IC")].sort_values("median_abs_rho_continuous", ascending=False).head(5)
    strongest_dp = dplm[(dplm.population == "STRICT_FULL300_34") & (dplm.method == "dPL")].sort_values("median_abs_rho_continuous", ascending=False).head(5)
    cls = pairm[pairm.population == "STRICT_FULL300_34"].iloc[0]
    persistent = cand[cand.candidate_set == "persistent/reproduced"].head(5) if not cand.empty else pd.DataFrame()
    signchange = cand[cand.candidate_set == "sign-changing"].head(5) if not cand.empty else pd.DataFrame()
    top_attr_ic = ", ".join(f"{r.attribute} ({int(r.top5_model_count)} models)" for r in s_attr_ic.itertuples())
    top_attr_dp = ", ".join(f"{r.attribute} ({int(r.top5_model_count)} models)" for r in s_attr_dp.itertuples())
    strongest_ic_text = ", ".join(f"{r.model} (median |rho|={r.median_abs_rho_continuous:.3f})" for r in strongest_ic.itertuples())
    strongest_dp_text = ", ".join(f"{r.model} (median |rho|={r.median_abs_rho_continuous:.3f})" for r in strongest_dp.itertuples())
    persistent_text = "; ".join(f"{r.model}:{r.parameter}–{r.attribute} ({r.rho_ic:.2f}/{r.rho_dpl:.2f})" for r in persistent.itertuples())
    sign_text = "; ".join(f"{r.model}:{r.parameter}–{r.attribute} ({r.rho_ic:.2f}/{r.rho_dpl:.2f})" for r in signchange.itertuples())
    role_n = int(audit.loc[audit.role_confidence == "HIGH", ["model", "parameter"]].drop_duplicates().shape[0])
    text = f"""# Parameter–attribute atlas (no training)

## One-sentence conclusion
The frozen artifacts show a clear but model-conditional and parameter-specific attribute structure: strict34 IC–dPL profile agreement is non-trivial but incomplete, so Chapter 4 can proceed with an atlas and targeted hypotheses, not a pooled physical consensus or OOB/PUR claim.

This package is a separate output and does not overwrite the prior R1/R2 packages. Every dPL table is labelled `SEEN_BASIN_PROXY`; the current dPL was fit jointly on all 531 basins.

## Populations

- Primary: `STRICT_FULL300_34` ({len(strict)} models × 531 basins).
- Sensitivity: `RELAXED_SEEN_35` ({len(common)} models × 531 basins), including `simhyd` IC generation 280.
- `flexb`: IC-only descriptive atlas (`IC_ONLY_FLEXB`); no dPL reconstruction or imputation.
- Canonical attributes: 35 columns, 531 aligned IDs. The three categorical-code columns are retained in machine tables but are excluded from continuous cross-model interpretation and figures.

## Estimator and coordinate freeze

- Both sides use the same basin-wise Spearman implementation, average ranks, identical canonical 531-basin order/mask, and the same all-531 normalized attribute matrix.
- IC coordinate: `sigmoid(best_latent)` from the canonical `best_training` chunks; one best-training-fitness-selected start among 10 CMA-ES starts per basin; training fitness only.
- dPL coordinate: archived network sigmoid output from the latest saved `epoch_*.pt` checkpoint, seed 42; no held-out/test selection.
- IC physical mapping is `linear`; dPL hydrology mapping is archived `auto`. Both are coordinate-wise monotone (linear or positive log interpolation), so within-side Spearman ranks are preserved; normalized-coordinate agreement is not physical-parameter agreement.
- Parameter order and bounds come from `src/model_registry.py` → canonical `PARAM_INFO`; all 36 models are audited.
- Boundary diagnostics are descriptive: near-constant means normalized SD ≤ 0.01; boundary-concentrated means ≥50% of finite values ≤0.01 or ≥0.99.

## IC atlas findings (strict34)

- The most recurring continuous attributes among top-five parameter controls are: {top_attr_ic}.
- Models with the strongest overall continuous IC structure by median absolute cell rho are: {strongest_ic_text}.
- These are descriptive effect-size/rank patterns, not significance or causal claims. Parameter-specific boundary flags must be consulted before interpreting any individual cell.

## dPL atlas findings (strict34)

- The most recurring continuous attributes encoded by dPL are: {top_attr_dp}.
- Models with the strongest dPL profiles by median absolute cell rho are: {strongest_dp_text}.
- A strong dPL relationship alone does not establish physical truth: dPL is explicitly an attribute→parameter mapping.

## IC–dPL reproducibility (strict34)

- Flattened profile Spearman: {s_pair.flattened_profile_spearman:.4f}; Pearson: {s_pair.flattened_profile_pearson:.4f}.
- Sign agreement: all pairs {s_pair.sign_agreement_all:.1%}; non-trivial pairs {s_pair.sign_agreement_nontrivial:.1%}.
- Parameter-profile Spearman distribution: median {s_repro.profile_spearman.median():.4f}, IQR {(s_repro.profile_spearman.quantile(.75)-s_repro.profile_spearman.quantile(.25)):.4f}, range {s_repro.profile_spearman.min():.4f}..{s_repro.profile_spearman.max():.4f}.
- Dominant-control agreement: {s_repro.dominant_control_agreement.mean():.1%}; top-five overlap median {s_repro.top5_overlap_count.median():.1f} attributes.
- Main descriptive classes (high=0.20, low=0.10, top-rank≤10): persistent {cls.persistent_fraction:.1%}, attenuated {cls.attenuated_fraction:.1%}, dPL-emergent {cls.dpl_emergent_fraction:.1%}, sign-changing {cls.sign_changing_fraction:.1%}, weak/unresolved {cls.weak_unresolved_fraction:.1%}. Threshold sensitivity is in `relationship_class_sensitivity.csv`; none is a scientific cutoff.
- Representative persistent candidates: {persistent_text or 'none selected'}.
- Representative sign-changing candidates: {sign_text or 'none selected'}.

## Cross-model interpretation

There are recurring attribute-level controls, but parameter semantics are not sufficiently centralized for a safe pooled physical role atlas. Only {role_n} high-confidence role rows come from the explicit Flex process groups in `src/model_registry.py`; all other role mappings are `UNRESOLVED` rather than inferred from names or equal bounds. Therefore the defensible Chapter 4 statement is **model-conditional relationships with parameter-specific reliability**, not a universal attribute→process law.

## Future OOB/PUR targets

`relationship_candidate_list.csv` contains a finite, reproducible strict34 candidate set: persistent/reproduced positives plus attenuated, dPL-emergent, and sign-changing counterexamples. Targets are labelled only `OOB`, `PUR`, `seed`, `threshold sensitivity`, `semantic audit`, or `mapping audit`. No target was trained in this run.

## Secondary D_seen linkage

`d_seen_secondary_linkage.csv` is secondary seen-basin metadata only. It must not be interpreted as evidence that relationship reproducibility causes or predicts formal OOB performance.

## Limitations and non-claims

- `VALID_OOB_DPL=0`; no current dPL result is cross-fitted or ungauged-transfer evidence.
- dPL has one archived seed and uses the latest-saved-checkpoint rule; health best/terminal differences remain an audit limitation.
- `simhyd` is relaxed-only with IC generation 280; `flexb` has no complete dPL artifact.
- Categorical attributes are ordinal-code Spearman rows for descriptive completeness only.
- No p-value, causal, physical-truth, or parameter-equivalence claim is made. Current normalized-coordinate IC–dPL comparisons do not prove equal physical parameter values.
- No training, checkpoint write, production-equation change, OOB/PUB/PUR run, bootstrap, or multiprocessing was executed.

## Output map

- `population_manifest.csv`, `parameter_audit.csv`, `parameter_semantic_mapping.csv`
- `ic_parameter_attribute_long.csv`, `dpl_parameter_attribute_long.csv`
- `ic_model_summary.csv`, `dpl_model_summary.csv`, `model_relationship_summary.csv`
- `ic_dpl_pairwise_relationships.csv`, `parameter_reproducibility.csv`, `attribute_crossmodel_summary.csv`
- `relationship_class_sensitivity.csv`, `relationship_candidate_list.csv`, `d_seen_secondary_linkage.csv`
- `figure_source_*.csv`, `figures/`, `figure_manifest.json`
- `provenance.json`, `source_hash_manifest.json`, `resource_metadata.json`, `README.md`
"""
    (out / "README.md").write_text(text)


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved compact parameter-forward path")
    device = torch.device("cuda")
    ids = np.asarray([int(x) for x in load_ids(IDS_PATH)], dtype=np.int64)
    if len(ids) != 531 or len(np.unique(ids)) != 531:
        raise RuntimeError("canonical basin list is not exactly 531 unique IDs")
    common, strict, ic_only = current_populations()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = CatchmentAttributeBuilder().load_raw_attributes(ids)
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore").to(torch.float32)
    attribute_matrix = attrs.detach().cpu().numpy()
    if raw.shape != (531, 35) or attribute_matrix.shape != (531, 35):
        raise RuntimeError(f"expected raw/normalized attributes 531x35, got {raw.shape}/{attribute_matrix.shape}")
    gage_ids = np.asarray(np.load(REPO / "data/gage_id.npy"), dtype=np.int64)
    if set(ids.tolist()) - set(gage_ids.tolist()):
        raise RuntimeError("canonical IDs missing from gage_id.npy")
    alignment = {
        "canonical_id_count": 531, "canonical_id_unique": 531, "gage_id_count": int(len(gage_ids)),
        "gage_id_unique": int(len(np.unique(gage_ids))), "missing_canonical_ids": [], "raw_attribute_shape": list(raw.shape),
        "normalized_attribute_shape": list(attribute_matrix.shape), "attribute_order": CAMELS_35_ATTRIBUTES,
        "attribute_types": ATTRIBUTE_TYPE, "join_key": "basin_id", "mask": "same canonical 531 IDs for every model/method",
        "categorical_columns": sorted(CATEGORICAL), "label": "ALIGNMENT_AUDIT_PASS",
    }
    write_json(OUT / "basin_alignment_audit.json", alignment)
    model_arrays: dict[str, dict[str, Any]] = {}
    audit_meta: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        need_dpl = model in common
        ic_theta, dpl_theta, meta = load_vectors(model, ids, attrs, device, need_dpl)
        model_arrays[model] = {"ic_theta": ic_theta, "dpl_theta": dpl_theta, "meta": meta}
        spec = get_spec(model, device="cpu")
        audit_meta.append({"model": model, "parameter_count": spec.dimension, "parameter_names": json.dumps(list(spec.parameter_names)),
                           "routed_kind": spec.routed_kind, "ic_generation": meta["ic_generation"], "dpl_checkpoint_epoch": meta["dpl_checkpoint_epoch"],
                           "dpl_health_best_epoch": meta["dpl_health_best_epoch"], "dpl_health_stop_epoch": meta["dpl_health_stop_epoch"],
                           "ic_source_used": meta["ic_source_used"], "ic_canonical_declared": meta["ic_canonical_declared"],
                           "dpl_source_used": meta["dpl_source_used"], "dpl_seed": 42 if need_dpl else "", "dpl_status": "COMPLETE" if need_dpl else "SKIP_MISSING_DPL",
                           "population": "IC_ONLY_FLEXB" if model == "flexb" else ("STRICT_FULL300_34" if model in strict else "RELAXED_SEEN_35"),
                           "label": "SEEN_BASIN_PROXY" if need_dpl else "IC_DESCRIPTIVE_ONLY"})
        print(f"[{model}] vectors loaded: IC={ic_theta.shape} dPL={None if dpl_theta is None else dpl_theta.shape}", flush=True)
    semantic = pd.DataFrame([row for model in ALL_MODELS for row in semantic_mapping(model)])
    ic_long, dpl_long, param_audit = build_long_tables(model_arrays, ids, attribute_matrix, common, strict)
    pair = merge_pairwise(ic_long, dpl_long, common, strict)
    model_summary, pair_summary, attr_summary, pair = model_and_attribute_summaries(ic_long, dpl_long, pair, semantic)
    repro = parameter_reproducibility(pair)
    sensitivity = class_sensitivity(pair)
    cand = candidates(pair, semantic)
    pop = population_manifest(ic_long, dpl_long, pair, common, strict)
    # Method-specific and combined summaries.
    write_csv(OUT / "population_manifest.csv", pop)
    write_csv(OUT / "parameter_audit.csv", param_audit)
    write_csv(OUT / "parameter_semantic_mapping.csv", semantic)
    write_csv(OUT / "ic_parameter_attribute_long.csv", ic_long)
    write_csv(OUT / "dpl_parameter_attribute_long.csv", dpl_long)
    write_csv(OUT / "ic_model_summary.csv", model_summary[model_summary.method == "IC"])
    write_csv(OUT / "dpl_model_summary.csv", model_summary[model_summary.method == "dPL"])
    write_csv(OUT / "model_relationship_summary.csv", pair_summary)
    write_csv(OUT / "ic_dpl_pairwise_relationships.csv", pair)
    write_csv(OUT / "parameter_reproducibility.csv", repro)
    write_csv(OUT / "attribute_crossmodel_summary.csv", attr_summary)
    write_csv(OUT / "relationship_class_sensitivity.csv", sensitivity)
    write_csv(OUT / "relationship_candidate_list.csv", cand)
    dseen_link = dseen_linkage(pair_summary, BASE_OUT / "d_seen_by_basin.csv")
    write_csv(OUT / "d_seen_secondary_linkage.csv", dseen_link)
    write_csv(OUT / "parameter_vector_audit.csv", pd.DataFrame(audit_meta))
    # Persist normalized vectors for exact downstream reuse; no daily predictions are written.
    vector_manifest = []
    for model, item in model_arrays.items():
        spec = get_spec(model, device="cpu")
        for method, theta in (("IC", item["ic_theta"]), ("dPL", item["dpl_theta"])):
            if theta is None:
                continue
            subdir = OUT / ("parameter_vectors_ic" if method == "IC" else "parameter_vectors_dpl")
            subdir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(theta, columns=list(spec.parameter_names))
            frame.insert(0, "basin_id", [f"{int(x):08d}" for x in ids])
            frame.to_csv(subdir / f"{model}.csv", index=False, float_format="%.10f")
            vector_manifest.append({"model": model, "method": method, "path": str((subdir / f"{model}.csv").relative_to(OUT)), "n_basins": len(ids), "parameter_count": theta.shape[1], "coordinate": "normalized sigmoid coordinate", "label": "SEEN_BASIN_PROXY" if method == "dPL" else "IC_DESCRIPTIVE"})
    write_csv(OUT / "parameter_vector_manifest.csv", vector_manifest)
    write_json(OUT / "classification_rules.json", {"main_high_abs_rho": MAIN_HIGH_RHO, "main_low_abs_rho": MAIN_LOW_RHO, "main_top_rank": MAIN_TOP_RANK, "classes": "persistent/reproduced, attenuated, dPL-emergent, sign-changing, weak/unresolved", "status": "DESCRIPTIVE_ONLY; no scientific cutoff"})
    figures = draw_figures(OUT, ic_long, dpl_long, pair, repro, attr_summary, strict)
    source_paths = [
        SCRIPT_DIR / "parameter_attribute_atlas.py", SCRIPT_DIR / "r1_r2_quick_survey.py", SCRIPT_DIR / "r1_r2_nontraining_complete.py",
        BENCHMARK / "src/model_registry.py", BENCHMARK / "dpl/attributes.py", BENCHMARK / "dpl/nn_parameterizer.py",
        IC_ROOT / "status_summary.json", DPL_ROOT / "health.csv", DPL_ROOT / "status.csv", BASE_OUT / "basin_alignment_audit.json",
        BASE_OUT / "protocol_audit.json",
    ] + [REPO / "dmotpy/models/core" / f"{m}.py" for m in ALL_MODELS]
    hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in source_paths if path.is_file()}
    write_json(OUT / "source_hash_manifest.json", hashes)
    elapsed = time.time() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    resource_meta = {"runtime_seconds": round(elapsed, 3), "gpu": torch.cuda.get_device_name(0), "dtype": "FP32", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "max_concurrent_forward": 1, "multiprocessing": False, "training_started": False, "checkpoint_write_attempted": False, "daily_predictions_persisted": False, "peak_rss_mib": round(peak, 1), "n_models_registry": 36, "n_models_strict": len(strict), "n_models_relaxed": len(common), "n_rows_ic": len(ic_long), "n_rows_dpl": len(dpl_long), "n_rows_pairwise": len(pair), "n_parameter_reproducibility_rows": len(repro), "figure_count_primary": len(figures["figures"]), "label": "SEEN_BASIN_PROXY for all dPL outputs"}
    write_json(OUT / "resource_metadata.json", resource_meta)
    provenance = {"analysis": "parameter_attribute_atlas", "analysis_type": "non-training", "source_ic": str(IC_ROOT), "source_dpl": str(DPL_ROOT), "source_base_survey": str(BASE_OUT), "population_manifest": "population_manifest.csv", "attribute_estimator": "basin-wise Spearman with average ranks", "attribute_coordinate": "CatchmentAttributeBuilder all-531 zscore matrix; 35 columns", "parameter_coordinate": "normalized sigmoid coordinates", "ic_selection": "best_training training-fitness argmax among 10 starts", "dpl_selection": "latest saved epoch checkpoint; seed 42", "VALID_OOB_DPL": 0, "training_started": False, "training_allowed": False, "checkpoint_write_attempted": False, "oob_pur_executed": False, "flexb": "IC_ONLY_FLEXB; missing/incomplete dPL skipped", "simhyd": "RELAXED_SEEN_35 only; IC generation 280", "categorical_handling": "retained as ordinal-code descriptive rows; excluded from continuous pooled interpretation", "physical_truth_claim": False, "label": "SEEN_BASIN_PROXY"}
    write_json(OUT / "provenance.json", provenance)
    write_readme(OUT, common, strict, pop, model_summary, model_summary, pair_summary, repro, attr_summary, cand, param_audit)
    print(json.dumps({"output": str(OUT), "relaxed_models": len(common), "strict_models": len(strict), "ic_rows": len(ic_long), "dpl_rows": len(dpl_long), "pair_rows": len(pair), "parameter_rows": len(repro), "candidate_rows": len(cand), "runtime_seconds": round(elapsed, 2), "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
