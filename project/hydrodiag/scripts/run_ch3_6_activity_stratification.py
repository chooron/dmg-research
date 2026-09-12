#!/usr/bin/env python3
"""Add CAMELS-attribute activity stratification to the Chapter 3.6 package.

This script reads only the supplied CAMELS text attributes and existing
Chapter 3.6 basin-level/replay artifacts. It does not train, calibrate, or
forward any model. All joins are explicit basin_id joins; strata cut points
are fixed once on the finite canonical 531-basin attribute population.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
REPO = PROJECT.parent.parent
ROOT = PROJECT / "results" / "ch3_6_cross_process"
OUT = ROOT / "09_activity_stratification"
AUDIT = OUT / "00_attribute_audit"
STRATA = OUT / "01_strata"
ET_OUT = OUT / "02_et"
RESP_OUT = OUT / "03_response"
CROSS_OUT = OUT / "04_cross_process"
FIG_OUT = OUT / "figure_ready"
CLIM_PATH = PROJECT / "camels_clim.txt"
HYDRO_PATH = PROJECT / "camels_hydro.txt"
BASIN_LIST = REPO / "data" / "531sub_id.txt"
BUNDLE_PATH = REPO / "data" / "camels_dataset"
GAGE_PATH = REPO / "data" / "gage_id.npy"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260730
EVAL = slice(5478, 10957)
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MODEL_GROUPS = {
    "IC": {"N": "N", "D_E": "D_E", "G_E": "G_E", "D_R": "D_R", "G_R": "G_R"},
    "dPL": {
        "N": "XAJ_CONTROLLED_N_CN",
        "D_E": "XAJ_D_E_CN",
        "G_E": "XAJ_G_E_CN",
        "D_R": "XAJ_D_R_CN",
        "G_R": "XAJ_G_R_CN",
    },
}
CONTRAST_ROLES = {"ET": ("D_E", "G_E"), "Response": ("D_R", "G_R")}


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(value) else f"{value:.10g}"


def as_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def finite(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_median_ci(values: list[float] | np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = finite(values)
    if not values.size:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    medians = np.median(values[indices], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def distribution(values: list[float] | np.ndarray) -> dict[str, str]:
    values = finite(values)
    if not values.size:
        return {"valid_n": "0", "median": "", "q25": "", "q75": ""}
    return {
        "valid_n": str(values.size),
        "median": fmt(np.median(values)),
        "q25": fmt(np.percentile(values, 25)),
        "q75": fmt(np.percentile(values, 75)),
    }


def effect_summary(values: list[float] | np.ndarray, rng: np.random.Generator, absolute: bool = False) -> dict[str, str]:
    values = finite(values)
    if absolute:
        values = np.abs(values)
    result = distribution(values)
    low, high = bootstrap_median_ci(values, rng)
    result.update(
        {
            "ci95_low": fmt(low),
            "ci95_high": fmt(high),
            "positive_fraction": fmt(np.mean(values > 0)) if values.size else "",
        }
    )
    return result


def correlation_summary(x: list[float] | np.ndarray, y: list[float] | np.ndarray, rng: np.random.Generator) -> dict[str, str]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return {"finite_n": str(x.size), "rho": "", "ci95_low": "", "ci95_high": ""}
    rho = float(spearmanr(x, y).statistic)
    indices = np.arange(x.size)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(indices, size=x.size, replace=True)
        if np.unique(x[sample]).size < 2 or np.unique(y[sample]).size < 2:
            continue
        value = float(spearmanr(x[sample], y[sample]).statistic)
        if math.isfinite(value):
            boot.append(value)
    return {
        "finite_n": str(x.size),
        "rho": fmt(rho),
        "ci95_low": fmt(np.percentile(boot, 2.5)) if boot else "",
        "ci95_high": fmt(np.percentile(boot, 97.5)) if boot else "",
    }


def endpoint_summary(low: list[float] | np.ndarray, high: list[float] | np.ndarray, rng: np.random.Generator) -> dict[str, str]:
    low = finite(low)
    high = finite(high)
    if not low.size or not high.size:
        return {"low_valid_n": str(low.size), "high_valid_n": str(high.size), "low_median": "", "high_median": "", "difference_high_minus_low": "", "ci95_low": "", "ci95_high": ""}
    difference = float(np.median(high) - np.median(low))
    boot = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        low_sample = rng.choice(low, size=low.size, replace=True)
        high_sample = rng.choice(high, size=high.size, replace=True)
        boot[index] = np.median(high_sample) - np.median(low_sample)
    return {
        "low_valid_n": str(low.size),
        "high_valid_n": str(high.size),
        "low_median": fmt(np.median(low)),
        "high_median": fmt(np.median(high)),
        "difference_high_minus_low": fmt(difference),
        "ci95_low": fmt(np.percentile(boot, 2.5)),
        "ci95_high": fmt(np.percentile(boot, 97.5)),
    }


def contrast_parts(contrast: str) -> tuple[str, str]:
    variant, baseline = contrast.split("-", 1)
    return baseline, variant


def load_canonical_basins() -> list[str]:
    values = json.loads(BASIN_LIST.read_text(encoding="utf-8"))
    basin_ids = [str(value).zfill(8) for value in values]
    if len(basin_ids) != 531 or len(set(basin_ids)) != 531:
        raise RuntimeError("canonical basin list is not a unique 531-basin set")
    return basin_ids


def load_attributes() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    clim = read_csv(CLIM_PATH, delimiter=";")
    hydro = read_csv(HYDRO_PATH, delimiter=";")
    for source, name in ((clim, "camels_clim"), (hydro, "camels_hydro")):
        ids = [str(row["gauge_id"]).zfill(8) for row in source]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{name} contains duplicate gauge IDs")
    if "aridity" not in clim[0]:
        raise RuntimeError("supplied camels_clim.txt has no aridity field")
    if "baseflow_index" not in hydro[0]:
        raise RuntimeError("supplied camels_hydro.txt has no baseflow_index field")
    return clim, hydro


def canonical_attr_maps(basin_ids: list[str], clim: list[dict[str, str]], hydro: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    clim_map = {str(row["gauge_id"]).zfill(8): as_float(row["aridity"]) for row in clim}
    hydro_map = {str(row["gauge_id"]).zfill(8): as_float(row["baseflow_index"]) for row in hydro}
    return {
        "aridity": {basin: clim_map.get(basin, float("nan")) for basin in basin_ids},
        "baseflow_index": {basin: hydro_map.get(basin, float("nan")) for basin in basin_ids},
    }


def verify_bundle_aridity(basin_ids: list[str], aridity: dict[str, float]) -> dict[str, object]:
    with BUNDLE_PATH.open("rb") as handle:
        _, _, attributes = pickle.load(handle)
    gage_ids = np.asarray(np.load(GAGE_PATH, allow_pickle=False)).astype(str)
    bundle_map = {str(gage).zfill(8): float(attributes[index, 4]) for index, gage in enumerate(gage_ids)}
    differences = [abs(bundle_map[basin] - aridity[basin]) for basin in basin_ids if basin in bundle_map and math.isfinite(aridity[basin])]
    return {
        "bundle_attribute_name": "aridity",
        "bundle_attribute_index_zero_based": 4,
        "bundle_attribute_contract": "ATTRIBUTE_NAMES[4] in ablation/ic_core/data_adapter.py",
        "comparison_n": len(differences),
        "max_abs_difference": max(differences) if differences else None,
        "match": bool(differences) and max(differences) <= 1e-12,
    }


def write_attribute_audit(basin_ids: list[str], clim: list[dict[str, str]], hydro: list[dict[str, str]], attrs: dict[str, dict[str, float]], bundle_check: dict[str, object]) -> list[dict[str, str]]:
    AUDIT.mkdir(parents=True, exist_ok=True)
    source_specs = [
        {
            "source_kind": "CAMELS climate attribute text",
            "source_file": str(CLIM_PATH.relative_to(REPO)),
            "attribute": "aridity",
            "definition": "Canonical CAMELS climatic descriptor named aridity; supplied text contains the field but no free-text formula metadata",
            "units": "Not specified in supplied text",
            "direction": "Numeric ascending order is used as requested: A5 is highest aridity; no transformation or PET/P formula is asserted",
            "observed_provenance": "CAMELS climatic descriptor file",
            "rows": len(clim),
            "source_sha256": sha256(CLIM_PATH),
        },
        {
            "source_kind": "CAMELS hydro attribute text",
            "source_file": str(HYDRO_PATH.relative_to(REPO)),
            "attribute": "baseflow_index",
            "definition": "Canonical CAMELS hydro/catchment descriptor named baseflow_index; value is read without recalculation",
            "units": "Not specified in supplied text; numeric ratio/fraction values are retained unchanged",
            "direction": "Larger value is treated as higher baseflow contribution, consistent with the descriptor name and predeclared B1-to-B5 ordering",
            "observed_provenance": "CAMELS hydro descriptor file; not model-simulated BFI and not phase0_sampling.py output",
            "rows": len(hydro),
            "source_sha256": sha256(HYDRO_PATH),
        },
    ]
    audit_rows: list[dict[str, str]] = []
    for spec in source_specs:
        values = finite([attrs[spec["attribute"]][basin] for basin in basin_ids])
        source_rows = clim if spec["attribute"] == "aridity" else hydro
        source_ids = [str(row["gauge_id"]).zfill(8) for row in source_rows]
        source_set = set(source_ids)
        canonical_set = set(basin_ids)
        duplicate_ids = sorted({item for item in source_ids if source_ids.count(item) > 1})
        audit_rows.append(
            {
                **spec,
                "canonical_basin_n": str(len(basin_ids)),
                "source_unique_id_n": str(len(source_set)),
                "duplicate_id_n": str(len(source_ids) - len(source_set)),
                "join_n": str(len(source_set & canonical_set)),
                "missing_from_source_n": str(len(canonical_set - source_set)),
                "extra_source_n": str(len(source_set - canonical_set)),
                "finite_join_n": str(values.size),
                "min": fmt(np.min(values) if values.size else None),
                "q25": fmt(np.percentile(values, 25) if values.size else None),
                "median": fmt(np.median(values) if values.size else None),
                "q75": fmt(np.percentile(values, 75) if values.size else None),
                "max": fmt(np.max(values) if values.size else None),
                "missing_ids": ";".join(sorted(canonical_set - source_set)),
                "duplicate_ids": ";".join(duplicate_ids),
                "analysis_finite_n": str(values.size),
            }
        )
    audit_fields = [
        "source_kind", "source_file", "attribute", "definition", "units", "direction", "observed_provenance", "rows", "source_sha256",
        "canonical_basin_n", "source_unique_id_n", "duplicate_id_n", "join_n", "missing_from_source_n", "extra_source_n", "finite_join_n",
        "min", "q25", "median", "q75", "max", "missing_ids", "duplicate_ids", "analysis_finite_n",
    ]
    write_csv(AUDIT / "attribute_field_audit.csv", audit_rows, audit_fields)
    (AUDIT / "CAMELS_ATTRIBUTE_AUDIT.md").write_text(
        "# CAMELS attribute audit\n\n"
        "## Decision\n\n"
        "The supplied repository-local files `project/hydrodiag/camels_clim.txt` and `project/hydrodiag/camels_hydro.txt` are used directly. No Windows-mounted path, model output, or derived replacement attribute is used. Both files contain 671 unique gauges and join all 531 canonical basins by `basin_id`/`gauge_id`.\n\n"
        "## ET attribute\n\n"
        "- Exact field: `aridity` in `camels_clim.txt`.\n"
        "- Current CAMELS bundle contract: `ATTRIBUTE_NAMES[4] == aridity` (0-based index 4); the 531 values match the supplied climate file to the recorded tolerance.\n"
        "- The supplied text has no free-text formula or unit metadata. This package therefore does not silently assert a PET/P formula or apply a transform; it freezes the observed numeric field and uses ascending values for A1–A5, with A5 the highest aridity stratum as predeclared.\n"
        f"- Finite canonical N: {audit_rows[0]['finite_join_n']}; range and quartiles are recorded in `attribute_field_audit.csv`.\n\n"
        "## Response attribute\n\n"
        "- Exact field: `baseflow_index` in `camels_hydro.txt`.\n"
        "- This is the supplied CAMELS hydro/catchment descriptor and is read as an observed/static attribute. It is not the BFI calculated in `phase0_sampling.py`, not any model-simulated BFI, and not recalculated here.\n"
        "- The supplied text has no algorithm prose or unit row; values are retained unchanged and ascending values define B1–B5, with B5 the highest baseflow-index value.\n"
        f"- Finite canonical N: {audit_rows[1]['finite_join_n']}; range and quartiles are recorded in `attribute_field_audit.csv`.\n\n"
        "## Bundle consistency\n\n"
        f"`aridity` bundle comparison: N={bundle_check['comparison_n']}, max absolute difference={fmt(bundle_check['max_abs_difference'])}, match={bundle_check['match']}.\n",
        encoding="utf-8",
    )
    return audit_rows


def make_strata(values: dict[str, float], prefix: str) -> tuple[dict[str, int], dict[str, object]]:
    basin_ids = sorted(values)
    finite_values_array = finite([values[basin] for basin in basin_ids])
    cuts = np.quantile(finite_values_array, [0.2, 0.4, 0.6, 0.8])
    strata: dict[str, int] = {}
    for basin in basin_ids:
        value = values[basin]
        if not math.isfinite(value):
            strata[basin] = 0
        else:
            strata[basin] = int(np.searchsorted(cuts, value, side="right")) + 1
    detail: dict[str, object] = {
        "attribute": "aridity" if prefix == "A" else "baseflow_index",
        "stratum_prefix": prefix,
        "ordering": "ascending_numeric_value",
        "cut_method": "fixed empirical quintiles on all finite canonical 531-basin values",
        "cut_points_P20_P40_P60_P80": [float(x) for x in cuts],
        "boundary_ties_are_not_split": True,
        "strata": {},
    }
    for number in range(1, 6):
        members = [basin for basin in basin_ids if strata[basin] == number]
        member_values = finite([values[basin] for basin in members])
        detail["strata"][f"{prefix}{number}"] = {
            "n": len(members),
            "min": float(np.min(member_values)) if member_values.size else None,
            "max": float(np.max(member_values)) if member_values.size else None,
            "basin_ids": members,
        }
    return strata, detail


def write_strata(basin_ids: list[str], attrs: dict[str, dict[str, float]]) -> dict[str, int]:
    STRATA.mkdir(parents=True, exist_ok=True)
    a_strata, a_detail = make_strata(attrs["aridity"], "A")
    b_strata, b_detail = make_strata(attrs["baseflow_index"], "B")
    definitions = {
        "canonical_basin_n": len(basin_ids),
        "attribute_source": {
            "ET": str(CLIM_PATH.relative_to(REPO)),
            "Response": str(HYDRO_PATH.relative_to(REPO)),
        },
        "ET": a_detail,
        "Response": b_detail,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "basin"},
    }
    write_json(STRATA / "strata_definitions.json", definitions)
    rows = [
        {
            "basin_id": basin,
            "aridity": fmt(attrs["aridity"][basin]),
            "A_stratum": f"A{a_strata[basin]}" if a_strata[basin] else "",
            "baseflow_index": fmt(attrs["baseflow_index"][basin]),
            "B_stratum": f"B{b_strata[basin]}" if b_strata[basin] else "",
        }
        for basin in basin_ids
    ]
    write_csv(STRATA / "basin_strata.csv", rows, ["basin_id", "aridity", "A_stratum", "baseflow_index", "B_stratum"])
    return {**a_strata, **b_strata}


def model_maps(path: Path, value_key: str, regime: str) -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    models = set(row["model"] for row in rows if row["regime"] == regime)
    return {(regime, model): {row["basin_id"]: as_float(row[value_key]) for row in rows if row["regime"] == regime and row["model"] == model} for model in models}


def outlet_maps(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows = read_csv(path)
    output: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        output.setdefault((row["regime"], row["contrast"]), {})[row["basin_id"]] = as_float(row["delta_kge"])
    return output


def build_effect_records(basin_ids: list[str], attrs: dict[str, dict[str, float]], strata_values: dict[str, int], process: str) -> list[dict[str, str]]:
    if process == "ET":
        outlet = outlet_maps(ROOT / "03_et" / "et_outlet_paired.csv")
        partition = model_maps(ROOT / "03_et" / "et_partition_basin.csv", "et_over_p_test", "IC")
        partition.update(model_maps(ROOT / "03_et" / "et_partition_basin.csv", "et_over_p_test", "dPL"))
        attribute_name, stratum_name = "aridity", "A_stratum"
        contrasts = [(regime, f"{role_model}-{groups['N']}", role, groups) for regime, groups in MODEL_GROUPS.items() for role in CONTRAST_ROLES[process] for role_model in [groups[role]]]
        rows = []
        for regime, contrast, role, groups in contrasts:
            baseline, variant = groups["N"], groups[role]
            base_map = partition[(regime, baseline)]
            variant_map = partition[(regime, variant)]
            for basin in basin_ids:
                delta_kge = outlet.get((regime, contrast), {}).get(basin, float("nan"))
                base = base_map.get(basin, float("nan"))
                variant_value = variant_map.get(basin, float("nan"))
                delta_partition = variant_value - base if math.isfinite(base) and math.isfinite(variant_value) else float("nan")
                rows.append({
                    "regime": regime, "contrast": contrast, "basin_id": basin,
                    attribute_name: fmt(attrs[attribute_name][basin]), stratum_name: f"A{strata_values[basin]}" if strata_values[basin] else "",
                    "delta_et_over_p": fmt(delta_partition), "abs_delta_et_over_p": fmt(abs(delta_partition)),
                    "delta_kge": fmt(delta_kge), "abs_delta_kge": fmt(abs(delta_kge)),
                })
        return rows
    outlet = outlet_maps(ROOT / "04_response" / "response_outlet_paired.csv")
    bfi_rows = read_csv(ROOT / "04_response" / "response_bfi_basin.csv")
    bfi_maps: dict[tuple[str, str], dict[str, float]] = {}
    for row in bfi_rows:
        bfi_maps.setdefault((row["regime"], row["model"]), {})[row["basin_id"]] = as_float(row["bfi"])
    attribute_name, stratum_name = "baseflow_index", "B_stratum"
    rows = []
    for regime, groups in MODEL_GROUPS.items():
        for role in CONTRAST_ROLES[process]:
            contrast = f"{groups[role]}-{groups['N']}"
            base_map = bfi_maps[(regime, groups["N"])]
            variant_map = bfi_maps[(regime, groups[role])]
            for basin in basin_ids:
                delta_bfi = float("nan")
                base = base_map.get(basin, float("nan"))
                variant = variant_map.get(basin, float("nan"))
                if math.isfinite(base) and math.isfinite(variant):
                    delta_bfi = variant - base
                delta_kge = outlet.get((regime, contrast), {}).get(basin, float("nan"))
                rows.append({
                    "regime": regime, "contrast": contrast, "basin_id": basin,
                    attribute_name: fmt(attrs[attribute_name][basin]), stratum_name: f"B{strata_values[basin]}" if strata_values[basin] else "",
                    "delta_bfi_model": fmt(delta_bfi), "abs_delta_bfi_model": fmt(abs(delta_bfi)),
                    "delta_kge": fmt(delta_kge), "abs_delta_kge": fmt(abs(delta_kge)),
                })
    return rows


def write_join_audit(basin_ids: list[str], attrs: dict[str, dict[str, float]], et_rows: list[dict[str, str]], response_rows: list[dict[str, str]]) -> None:
    rows = []
    for field, source in (("aridity", CLIM_PATH), ("baseflow_index", HYDRO_PATH)):
        rows.append({"source_kind": "static_attribute", "source_file": str(source.relative_to(REPO)), "attribute": field, "regime": "", "contrast": "", "canonical_basin_n": len(basin_ids), "join_n": len(basin_ids), "duplicate_id_n": 0, "missing_attribute_n": sum(not math.isfinite(value) for value in attrs[field].values()), "analysis_finite_n": sum(math.isfinite(value) for value in attrs[field].values()), "missing_ids": ";".join(basin for basin in basin_ids if not math.isfinite(attrs[field][basin]))})
    for label, data, field, effect_fields in (("ET", et_rows, "aridity", ("delta_et_over_p", "delta_kge")), ("Response", response_rows, "baseflow_index", ("delta_bfi_model", "delta_kge"))):
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in data:
            grouped[(row["regime"], row["contrast"])].append(row)
        for (regime, contrast), group in sorted(grouped.items()):
            finite_effect = sum(all(as_float(row[key]) == as_float(row[key]) for key in effect_fields) and as_float(row[field]) == as_float(row[field]) for row in group)
            rows.append({"source_kind": f"canonical_{label.lower()}_outputs", "source_file": "results/ch3_6_cross_process/03_et or 04_response", "attribute": field, "regime": regime, "contrast": contrast, "canonical_basin_n": len(basin_ids), "join_n": len(group), "duplicate_id_n": len(group) - len({row['basin_id'] for row in group}), "missing_attribute_n": sum(not as_float(row[field]) == as_float(row[field]) for row in group), "analysis_finite_n": finite_effect, "missing_ids": ";".join(row["basin_id"] for row in group if not all(as_float(row[key]) == as_float(row[key]) for key in effect_fields) or not as_float(row[field]) == as_float(row[field]))})
    write_csv(AUDIT / "basin_join_audit.csv", rows, ["source_kind", "source_file", "attribute", "regime", "contrast", "canonical_basin_n", "join_n", "duplicate_id_n", "missing_attribute_n", "analysis_finite_n", "missing_ids"])


def strata_summary(rows: list[dict[str, str]], process: str, rng: np.random.Generator) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["regime"], row["contrast"], row["A_stratum"] if process == "ET" else row["B_stratum"])].append(row)
    attr = "aridity" if process == "ET" else "baseflow_index"
    abs_primary = "abs_delta_et_over_p" if process == "ET" else "abs_delta_bfi_model"
    signed_primary = "delta_et_over_p" if process == "ET" else "delta_bfi_model"
    abs_secondary = "abs_delta_kge"
    signed_secondary = "delta_kge"
    output = []
    fields = ["regime", "contrast", "stratum", "n_stratum", "attribute_min", "attribute_max"]
    for name in (abs_primary, signed_primary, abs_secondary, signed_secondary):
        fields.extend([f"{name}_valid_n", f"{name}_median", f"{name}_q25", f"{name}_q75", f"{name}_ci95_low", f"{name}_ci95_high", f"{name}_positive_fraction"])
    for (regime, contrast, stratum), group in sorted(groups.items()):
        attr_values = finite([as_float(row[attr]) for row in group])
        result = {"regime": regime, "contrast": contrast, "stratum": stratum, "n_stratum": str(len(group)), "attribute_min": fmt(np.min(attr_values) if attr_values.size else None), "attribute_max": fmt(np.max(attr_values) if attr_values.size else None)}
        for name in (abs_primary, signed_primary, abs_secondary, signed_secondary):
            result.update({f"{name}_{key}": value for key, value in effect_summary([as_float(row[name]) for row in group], rng, absolute=name.startswith("abs_")).items()})
        output.append(result)
    return output, fields


def endpoint_table(rows: list[dict[str, str]], process: str, rng: np.random.Generator) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["regime"], row["contrast"])].append(row)
    attr_stratum = "A_stratum" if process == "ET" else "B_stratum"
    primary = "abs_delta_et_over_p" if process == "ET" else "abs_delta_bfi_model"
    output = []
    for (regime, contrast), group in sorted(groups.items()):
        for metric in (primary, "abs_delta_kge"):
            low = [as_float(row[metric]) for row in group if row[attr_stratum] in ("A1", "B1")]
            high = [as_float(row[metric]) for row in group if row[attr_stratum] in ("A5", "B5")]
            result = endpoint_summary(low, high, rng)
            output.append({"regime": regime, "contrast": contrast, "attribute": "aridity" if process == "ET" else "baseflow_index", "low_stratum": "A1" if process == "ET" else "B1", "high_stratum": "A5" if process == "ET" else "B5", "effect_metric": metric, **result})
    return output


def stratum_median_rho(group: list[dict[str, str]], process: str, metric: str) -> str:
    stratum_key = "A_stratum" if process == "ET" else "B_stratum"
    medians = []
    levels = []
    for level in range(1, 6):
        values = finite([as_float(row[metric]) for row in group if row[stratum_key] == ("A" if process == "ET" else "B") + str(level)])
        if values.size:
            levels.append(level)
            medians.append(float(np.median(values)))
    if len(medians) < 3:
        return ""
    return fmt(spearmanr(levels, medians).statistic)


def classify_gradient(rho: float, ci_low: float, ci_high: float, endpoint: float, endpoint_low: float, endpoint_high: float, strata_rho: float, medians: list[float]) -> str:
    if len(medians) >= 3 and any((medians[i] - medians[i - 1]) * (medians[i + 1] - medians[i]) < 0 for i in range(1, len(medians) - 1)) and abs(rho) < 0.35:
        return "NON_MONOTONIC"
    increasing = rho > 0 and endpoint > 0
    decreasing = rho < 0 and endpoint < 0
    clear_increase = increasing and ci_low > 0 and endpoint_low > 0 and strata_rho > 0
    clear_decrease = decreasing and ci_high < 0 and endpoint_high < 0 and strata_rho < 0
    if clear_increase:
        return "CLEAR_INCREASING_ACTIVITY_DEPENDENCE"
    if clear_decrease:
        return "DECREASING_ACTIVITY_DEPENDENCE"
    if increasing or decreasing or (ci_low <= 0 <= ci_high and endpoint_low <= 0 <= endpoint_high):
        return "WEAK_ACTIVITY_DEPENDENCE" if increasing or decreasing else "NO_CLEAR_DEPENDENCE"
    return "NO_CLEAR_DEPENDENCE"


def gradient_table(rows: list[dict[str, str]], endpoints: list[dict[str, str]], process: str, rng: np.random.Generator) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["contrast"])].append(row)
    endpoint_map = {(row["regime"], row["contrast"], row["effect_metric"]): row for row in endpoints}
    attr = "aridity" if process == "ET" else "baseflow_index"
    metrics = [("abs_delta_et_over_p", "primary_absolute") if process == "ET" else ("abs_delta_bfi_model", "primary_absolute"), ("abs_delta_kge", "secondary_absolute"), ("delta_et_over_p", "signed_direction") if process == "ET" else ("delta_bfi_model", "signed_direction"), ("delta_kge", "signed_direction")]
    output = []
    for (regime, contrast), group in sorted(grouped.items()):
        x = [as_float(row[attr]) for row in group]
        for metric, role in metrics:
            y = [as_float(row[metric]) for row in group]
            corr = correlation_summary(x, y, rng)
            medians = [float(np.median(finite([as_float(row[metric]) for row in group if row["A_stratum" if process == "ET" else "B_stratum"] == ("A" if process == "ET" else "B") + str(level)]))) for level in range(1, 6) if finite([as_float(row[metric]) for row in group if row["A_stratum" if process == "ET" else "B_stratum"] == ("A" if process == "ET" else "B") + str(level)]).size]
            endpoint_metric = metric if metric in ("abs_delta_et_over_p", "abs_delta_bfi_model", "abs_delta_kge") else "abs_delta_et_over_p" if process == "ET" else "abs_delta_bfi_model"
            endpoint = endpoint_map.get((regime, contrast, endpoint_metric), {})
            classification = ""
            if role == "primary_absolute":
                classification = classify_gradient(as_float(corr["rho"]), as_float(corr["ci95_low"]), as_float(corr["ci95_high"]), as_float(endpoint.get("difference_high_minus_low")), as_float(endpoint.get("ci95_low")), as_float(endpoint.get("ci95_high")), as_float(stratum_median_rho(group, process, metric)), medians)
            output.append({"regime": regime, "contrast": contrast, "attribute": attr, "effect_metric": metric, "effect_role": role, "finite_n": corr["finite_n"], "rho": corr["rho"], "ci95_low": corr["ci95_low"], "ci95_high": corr["ci95_high"], "strata_median_rho": stratum_median_rho(group, process, metric), "primary_classification": classification, "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED})
    return output


def write_monthly(canonical_basin_ids: list[str], strata: dict[str, int]) -> None:
    rows = []
    models = MODEL_GROUPS
    for regime, groups in (("IC", {key: value for key, value in MODEL_GROUPS["IC"].items() if key in ("N", "D_E", "G_E")}), ("dPL", {key: value for key, value in MODEL_GROUPS["dPL"].items() if key in ("N", "D_E", "G_E")})):
        for role, model in groups.items():
            path = ROOT / "02_replay" / ("IC" if regime == "IC" else "dPL") / f"{model}_full_replay.npz"
            with np.load(path, allow_pickle=False) as archive:
                basin_ids = [str(value).zfill(8) for value in archive["basin_ids"]]
                dates = np.asarray(archive["dates"], dtype="datetime64[D]")
                evap = np.asarray(archive["evap"][:, EVAL], dtype=np.float64)
            if tuple(basin_ids) != tuple(canonical_basin_ids) or evap.shape != (531, 5479) or dates.shape != (12418,):
                raise RuntimeError(f"unexpected monthly source shape/order for {regime}/{model}")
            eval_dates = dates[EVAL]
            month_numbers = (eval_dates.astype("datetime64[M]").astype(np.int64) % 12) + 1
            monthly = np.vstack([np.nanmean(np.where(np.isfinite(evap[:, month_numbers == month]), evap[:, month_numbers == month], np.nan), axis=1) for month in range(1, 13)]).T
            for stratum in ("A1", "A3", "A5"):
                indices = np.asarray([strata[basin] == int(stratum[1:]) for basin in basin_ids])
                for month in range(1, 13):
                    values = finite(monthly[indices, month - 1])
                    dist = distribution(values)
                    rows.append({"regime": regime, "stratum": stratum, "model": model, "month": month, "month_name": MONTH_NAMES[month - 1], "metric": "monthly_mean_daily_et_mm_day", **dist, "source": "existing full-replay evap array; evaluation slice 5478:10957; no forward"})
    write_csv(ET_OUT / "et_monthly_A1_A3_A5.csv", rows, ["regime", "stratum", "model", "month", "month_name", "metric", "valid_n", "median", "q25", "q75", "source"])


def response_baseline_check(basin_ids: list[str], static_bfi: dict[str, float], rng: np.random.Generator) -> list[dict[str, str]]:
    rows = read_csv(ROOT / "04_response" / "response_bfi_basin.csv")
    model_maps_by_regime: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        model_maps_by_regime[(row["regime"], row["model"])][row["basin_id"]] = as_float(row["bfi"])
    output = []
    for regime, model in (("IC", "N"), ("dPL", "XAJ_CONTROLLED_N_CN")):
        model_bfi = model_maps_by_regime[(regime, model)]
        x = [static_bfi[basin] for basin in basin_ids]
        y = [model_bfi.get(basin, float("nan")) for basin in basin_ids]
        corr = correlation_summary(x, y, rng)
        output.append({"regime": regime, "model": model, "attribute": "CAMELS baseflow_index", "outcome": "baseline model BFI", "finite_n": corr["finite_n"], "rho_spearman": corr["rho"], "ci95_low": corr["ci95_low"], "ci95_high": corr["ci95_high"], "interpretation": "descriptive provenance check only; not a causal test"})
    write_csv(RESP_OUT / "response_attribute_baseline_bfi_check.csv", output, ["regime", "model", "attribute", "outcome", "finite_n", "rho_spearman", "ci95_low", "ci95_high", "interpretation"])
    return output


def snow_frozen_rows() -> tuple[list[dict[str, str]], dict[str, object]]:
    continuous = read_csv(PROJECT / "manuscript" / "results" / "R5" / "r5_figure9_continuous_summary.csv")
    endpoint = read_csv(PROJECT / "manuscript" / "results" / "R5" / "r5_figure9_endpoint_summary.csv")
    continuous_rows = [{"record_type": "continuous", "process": "Snow", "attribute": "frac_snow", "host": row["host"], "regime": row["regime"], "stratum": "", "metric": "frozen structural timing effect", "rho": row["spearman_rho"], "ci95_low": row["ci_low"], "ci95_high": row["ci_high"], "value": "", "source": "manuscript/results/R5/r5_figure9_continuous_summary.csv"} for row in continuous]
    endpoint_rows = [{"record_type": "endpoint", "process": "Snow", "attribute": "frac_snow", "host": row["host"], "regime": row["regime"], "stratum": row["endpoint"], "metric": "median_delta_abs_CT_Base_CN", "rho": "", "ci95_low": "", "ci95_high": "", "value": row["median_delta_abs_CT_Base_CN"], "source": "manuscript/results/R5/r5_figure9_endpoint_summary.csv"} for row in endpoint]
    rho_values = [as_float(row["spearman_rho"]) for row in continuous]
    high_values = [as_float(row["median_delta_abs_CT_Base_CN"]) for row in endpoint if row["endpoint"] == "S5"]
    frozen = {"continuous_rho_min": min(rho_values), "continuous_rho_max": max(rho_values), "high_endpoint_min": min(high_values), "high_endpoint_max": max(high_values), "classification": "FROZEN_REFERENCE_STRONG_GRADIENT"}
    return continuous_rows + endpoint_rows, frozen


def write_cross_matrix(et_gradient: list[dict[str, str]], response_gradient: list[dict[str, str]], snow: dict[str, object]) -> None:
    et_primary = [row for row in et_gradient if row["effect_role"] == "primary_absolute"]
    response_primary = [row for row in response_gradient if row["effect_role"] == "primary_absolute"]
    def concise(rows: list[dict[str, str]], label: str) -> str:
        return "; ".join(f"{row['regime']} {row['contrast']}: {row['primary_classification'] or 'unclassified'} (rho={row['rho']})" for row in rows) or f"{label} unavailable"
    rows = [
        {"Process": "Snow", "CAMELS activity attribute": "frac_snow", "Low→High strata": "existing frozen S1→S5", "Primary structural-effect metric": "existing frozen structural timing effect", "Continuous gradient": f"rho range {fmt(snow['continuous_rho_min'])} to {fmt(snow['continuous_rho_max'])}", "High-vs-low endpoint": f"S5 median range {fmt(snow['high_endpoint_min'])} to {fmt(snow['high_endpoint_max'])}; S1=0 in frozen table", "IC/dPL consistency": "read from frozen R5 rows", "Classification": snow["classification"]},
        {"Process": "ET", "CAMELS activity attribute": "aridity", "Low→High strata": "A1→A5", "Primary structural-effect metric": "|Δ(ET/P)|", "Continuous gradient": concise(et_primary, "ET"), "High-vs-low endpoint": "see et_aridity_endpoint.csv", "IC/dPL consistency": "reported per contrast/regime; dPL is seed42", "Classification": "see et_aridity_gradient.csv"},
        {"Process": "Response", "CAMELS activity attribute": "baseflow_index", "Low→High strata": "B1→B5", "Primary structural-effect metric": "|ΔBFI_model|", "Continuous gradient": concise(response_primary, "Response"), "High-vs-low endpoint": "see response_baseflow_endpoint.csv", "IC/dPL consistency": "reported per contrast/regime; dPL is seed42", "Classification": "see response_baseflow_gradient.csv"},
    ]
    write_csv(CROSS_OUT / "process_activity_evidence_matrix.csv", rows, list(rows[0]))


def figure_ready(et_summary: list[dict[str, str]], et_gradient: list[dict[str, str]], et_endpoint: list[dict[str, str]], response_summary: list[dict[str, str]], response_gradient: list[dict[str, str]], response_endpoint: list[dict[str, str]], snow_rows: list[dict[str, str]]) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    def panel(summary: list[dict[str, str]], gradient: list[dict[str, str]], endpoint: list[dict[str, str]], process: str) -> list[dict[str, str]]:
        primary = "abs_delta_et_over_p" if process == "ET" else "abs_delta_bfi_model"
        strata_key = "A_stratum" if process == "ET" else "B_stratum"
        gmap = {(row["regime"], row["contrast"]): row for row in gradient if row["effect_role"] == "primary_absolute"}
        output = []
        for row in summary:
            output.append({"record_type": "stratum", "process": process, "regime": row["regime"], "contrast": row["contrast"], "stratum": row["stratum"], "primary_metric": primary, "primary_median": row[f"{primary}_median"], "primary_ci95_low": row[f"{primary}_ci95_low"], "primary_ci95_high": row[f"{primary}_ci95_high"], "secondary_metric": "abs_delta_kge", "secondary_median": row["abs_delta_kge_median"], "secondary_ci95_low": row["abs_delta_kge_ci95_low"], "secondary_ci95_high": row["abs_delta_kge_ci95_high"], "rho_primary": gmap[(row["regime"], row["contrast"])] ["rho"], "rho_ci95_low": gmap[(row["regime"], row["contrast"])] ["ci95_low"], "rho_ci95_high": gmap[(row["regime"], row["contrast"])] ["ci95_high"], "classification": gmap[(row["regime"], row["contrast"])] ["primary_classification"]})
        for row in endpoint:
            output.append({"record_type": "endpoint", "process": process, "regime": row["regime"], "contrast": row["contrast"], "stratum": f"{row['low_stratum']}→{row['high_stratum']}", "primary_metric": row["effect_metric"], "primary_median": row["difference_high_minus_low"], "primary_ci95_low": row["ci95_low"], "primary_ci95_high": row["ci95_high"], "secondary_metric": "", "secondary_median": "", "secondary_ci95_low": "", "secondary_ci95_high": "", "rho_primary": "", "rho_ci95_low": "", "rho_ci95_high": "", "classification": ""})
        return output
    fields = ["record_type", "process", "regime", "contrast", "stratum", "primary_metric", "primary_median", "primary_ci95_low", "primary_ci95_high", "secondary_metric", "secondary_median", "secondary_ci95_low", "secondary_ci95_high", "rho_primary", "rho_ci95_low", "rho_ci95_high", "classification"]
    write_csv(FIG_OUT / "Fig3_X_process_activity_ET.csv", panel(et_summary, et_gradient, et_endpoint, "ET"), fields)
    write_csv(FIG_OUT / "Fig3_X_process_activity_Response.csv", panel(response_summary, response_gradient, response_endpoint, "Response"), fields)
    snow_fields = ["record_type", "process", "attribute", "host", "regime", "stratum", "metric", "rho", "ci95_low", "ci95_high", "value", "source"]
    write_csv(FIG_OUT / "Fig3_X_process_activity_Snow_frozen.csv", snow_rows, snow_fields)


def primary_rows(gradient: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in gradient if row["effect_role"] == "primary_absolute"]


def cross_pattern(et_gradient: list[dict[str, str]], response_gradient: list[dict[str, str]]) -> tuple[str, str]:
    et = primary_rows(et_gradient)
    response = primary_rows(response_gradient)
    def increasing(rows: list[dict[str, str]]) -> bool:
        return any(row["primary_classification"] == "CLEAR_INCREASING_ACTIVITY_DEPENDENCE" for row in rows)
    def directional_mismatch(rows: list[dict[str, str]]) -> bool:
        return any(as_float(row["rho"]) > 0 and as_float(row["ci95_low"]) > 0 for row in rows) and any(as_float(row["rho"]) < 0 and as_float(row["ci95_high"]) < 0 for row in rows)
    if directional_mismatch(et) or directional_mismatch(response):
        return "D", "IC/dPL gradient directions differ for at least one process; parameter-estimation constraints may modulate the association."
    if increasing(et) and increasing(response):
        return "A", "Snow has the strongest frozen gradient; ET and Response each show at least one clear increasing activity dependence, subject to their per-regime qualifications."
    if increasing(et) and not increasing(response):
        return "B", "Snow is strong and ET has a clear increasing gradient, while Response has no comparably stable clear gradient."
    return "C", "Snow is strong, but the available ET/Response primary gradients do not jointly provide stable increasing dependence."


def report_text(audit_rows: list[dict[str, str]], definitions: dict[str, object], et_summary: list[dict[str, str]], et_gradient: list[dict[str, str]], et_endpoint: list[dict[str, str]], response_summary: list[dict[str, str]], response_gradient: list[dict[str, str]], response_endpoint: list[dict[str, str]], baseline_check: list[dict[str, str]], snow: dict[str, object], pattern: tuple[str, str]) -> str:
    def primary_lines(rows: list[dict[str, str]], gradient: list[dict[str, str]], endpoint: list[dict[str, str]], process: str) -> str:
        primary_metric = "abs_delta_et_over_p" if process == "ET" else "abs_delta_bfi_model"
        lines = []
        for row in primary_rows(gradient):
            end = next(item for item in endpoint if item["regime"] == row["regime"] and item["contrast"] == row["contrast"] and item["effect_metric"] == primary_metric)
            lines.append(f"- {row['regime']} {row['contrast']}: rho={row['rho']} [{row['ci95_low']}, {row['ci95_high']}]; endpoint {end['difference_high_minus_low']} [{end['ci95_low']}, {end['ci95_high']}]; class={row['primary_classification'] or 'NO_CLEAR_DEPENDENCE'}.")
        return "\n".join(lines)
    def secondary_lines(gradient: list[dict[str, str]], endpoint: list[dict[str, str]], process: str) -> str:
        lines = []
        for row in gradient:
            if row["effect_metric"] != "abs_delta_kge":
                continue
            end = next(item for item in endpoint if item["regime"] == row["regime"] and item["contrast"] == row["contrast"] and item["effect_metric"] == "abs_delta_kge")
            lines.append(f"- {row['regime']} {row['contrast']}: |ΔKGE| rho={row['rho']} [{row['ci95_low']}, {row['ci95_high']}]; endpoint={end['difference_high_minus_low']} [{end['ci95_low']}, {end['ci95_high']}].")
        return "\n".join(lines)
    def summary_by_metric(summary: list[dict[str, str]], metric: str) -> str:
        lines = []
        for row in summary:
            if row["stratum"] in (("A1", "A5") if metric.startswith("abs_delta_et") else ("B1", "B5")):
                lines.append(f"{row['regime']} {row['contrast']} {row['stratum']}: median={row[metric + '_median']} CI=[{row[metric + '_ci95_low']},{row[metric + '_ci95_high']}]")
        return "; ".join(lines)
    claims = [
        ("ET structural sensitivity increases with aridity?", "Qualified / per-regime", "|Δ(ET/P)| strata, rho, endpoint", "Do not generalize if class is weak, non-monotonic, or regime-dependent."),
        ("Response structural sensitivity increases with baseflow dominance?", "Qualified / per-regime", "|ΔBFI_model| strata, rho, endpoint", "CAMELS BFI is a static descriptor; model BFI is the outcome."),
        ("Are IC/dPL gradients consistent?", "Conditional", "Separate IC and dPL rows", "dPL remains seed42 directional evidence."),
        ("Is Snow activity dependence generalizable?", "Not assumed", "Frozen Snow matrix plus ET/Response gradients", "No cross-process effect-size ranking is permitted."),
        ("Does process importance condition structural identifiability?", "Qualified context", "Attribute–effect associations", "Associations are descriptive, not causal."),
    ]
    claim_table = "| Claim | Strength | Evidence | Limitation |\n|---|---|---|---|\n" + "\n".join("| " + " | ".join(item) + " |" for item in claims)
    numbers = []
    for row in primary_rows(et_gradient):
        if len(numbers) < 4:
            numbers.append(f"ET {row['regime']} {row['contrast']} rho={row['rho']} (N={row['finite_n']})")
    for row in primary_rows(response_gradient):
        if len(numbers) < 8:
            numbers.append(f"Response {row['regime']} {row['contrast']} rho={row['rho']} (N={row['finite_n']})")
    for row in et_endpoint[:2] + response_endpoint[:2]:
        if len(numbers) < 12:
            numbers.append(f"{row['regime']} {row['contrast']} {row['effect_metric']} A/B endpoint={row['difference_high_minus_low']} CI=[{row['ci95_low']},{row['ci95_high']}]")
    return f"""# Chapter 3.6 Process-Activity Stratification Final Report

## 1. Executive verdict

**`ACTIVITY_ANALYSIS_ADDS_QUALIFIED_CONTEXT`**。本轮合法补充了 ET–aridity 与 Response–CAMELS-baseflow-index 的条件分层；结果按 contrast/regime 单独报告，不把弱、非单调或缺少梯度解释为失败。Snow 仅作为既有 frozen reference。支持的跨过程模式为 **Pattern {pattern[0]}**：{pattern[1]}

## 2. CAMELS attribute provenance

使用的唯一新增属性文件是 `project/hydrodiag/camels_clim.txt` 和 `project/hydrodiag/camels_hydro.txt`，均按 `gauge_id` 读取。`aridity` 与当前 bundle 的 `ATTRIBUTE_NAMES[4]` 一致；Response 使用 `camels_hydro.txt` 的 `baseflow_index`，不是 `phase0_sampling.py` 计算值，也不是模型 BFI。属性文件的 SHA256、定义边界、单位信息和分布统计见 `00_attribute_audit/attribute_field_audit.csv` 与 `CAMELS_ATTRIBUTE_AUDIT.md`。供应文本不含公式/单位元数据，报告不擅自补写 PET/P 算法。

## 3. Strata definitions and basin counts

A1–A5 和 B1–B5 均在全部 finite canonical 531-basin population 上一次计算 P20/P40/P60/P80，并固定到 IC/dPL。相同边界值不拆分；实际 N、min/max、basin IDs 和 cut points 见 `01_strata/strata_definitions.json` 与 `basin_strata.csv`。未插补缺失属性。

## 4. ET aridity-stratified outlet results

Secondary outlet outcome 为 `|ΔKGE|`，signed `ΔKGE` 同时保留。每个 A1–A5 的 median、Q25/Q75、95% basin-bootstrap CI、positive fraction 在 `02_et/et_aridity_strata_summary.csv` 中；outlet continuous gradient 与 A5−A1 endpoint 如下：

{secondary_lines(et_gradient, et_endpoint, 'ET')}

## 5. ET aridity-stratified ET/P results

Primary outcome 为 `|Δ(ET/P)|`，signed `Δ(ET/P)` 同时保留。分层 basin-level 原始表为 `et_aridity_strata_basin.csv`，summary 同时记录有效 N、分布统计、CI 与方向比例。示例端点结果：{summary_by_metric(et_summary, 'abs_delta_et_over_p')}。

## 6. ET continuous aridity gradient

Continuous association 使用 aridity 与同一 paired contrast outcome 的 Spearman rho；不引入新水文指标、不以 p-value hunting 为依据。所有 absolute/signed outcome、finite N、bootstrap CI、seed 和 classification 在 `et_aridity_gradient.csv`。

{primary_lines(et_summary, et_gradient, et_endpoint, 'ET')}

## 7. ET high-vs-low endpoint

A5−A1 的 primary `|Δ(ET/P)|` 与 secondary `|ΔKGE|` endpoint 及 95% CI 在 `et_aridity_endpoint.csv`。端点是在固定 A1/A5 后独立 basin bootstrap，不重新计算分位点。

## 8. ET activity-dependence classification

分类遵循预声明规则：clear increasing 需 primary rho、CI、A5−A1 endpoint 和 strata trend 同向；弱/无清晰/非单调/下降均保留原标签。不能因结果修改 bins。

## 9. Response baseflow-stratified outlet results

Secondary outlet outcome 为 `|ΔKGE|`，signed `ΔKGE` 同时保留。每个 B1–B5 的 summary 和实际有效 N 在 `03_response/response_bfi_attribute_strata_summary.csv`；outlet continuous gradient 与 B5−B1 endpoint 如下：

{secondary_lines(response_gradient, response_endpoint, 'Response')}

## 10. Response baseflow-stratified BFI results

Primary outcome 为 `|ΔBFI_model| = |BFI_variant−BFI_N|`。stratifier 是 CAMELS static `baseflow_index`，outcome 是模型结构对比，二者严格分列。原始 basin-level、分层 summary、signed difference 和 valid N 见 `response_bfi_attribute_strata_basin.csv` 与 `response_bfi_attribute_strata_summary.csv`。

## 11. Response continuous baseflow gradient

{primary_lines(response_summary, response_gradient, response_endpoint, 'Response')}

连续结果包含 `baseflow_index` vs `|ΔBFI_model|`、`|ΔKGE|` 及两种 signed outcome；所有 finite N、rho、95% bootstrap CI、draws=2000、seed=20260730 在 `response_baseflow_gradient.csv`。

## 12. Response high-vs-low endpoint

B5−B1 endpoint 结果见 `response_baseflow_endpoint.csv`。不计算跨过程 ratio，不把 endpoint 当作因果效应。

## 13. Response observed-attribute/model-BFI interpretation check

`response_attribute_baseline_bfi_check.csv` 仅描述 CAMELS observed/catchment descriptor 与 baseline-N simulated BFI 的 provenance association；不进入主结论、不构成独立实验。结果：{'; '.join(row['regime'] + ' rho=' + row['rho_spearman'] + ' N=' + row['finite_n'] for row in baseline_check)}。

## 14. Response activity-dependence classification

D_R/G_R、IC/dPL 分别分类；不能合并成统一结论。分类依据与 ET 相同，主分类以 `|ΔBFI_model|` 为准，`|ΔKGE|` 为 secondary。

## 15. IC vs dPL comparison

IC 与 dPL 使用相同 CAMELS cut points，但独立计算 outcomes。dPL 仍为 seed42 directional evidence；不会增加 seeds，也不把 regime 差异解释为某一 regime 优越。

## 16. Snow / ET / Response process-activity matrix

`04_cross_process/process_activity_evidence_matrix.csv` 保留 Snow 的 frozen `frac_snow` / S1–S5 参考，并并列 ET aridity 与 Response CAMELS baseflow-index 的分层和 continuous evidence。三个 primary metric 不做数值横向排名。

## 17. Which cross-process pattern (A/B/C/D) is supported?

**Pattern {pattern[0]}**。{pattern[1]} 该判定只比较梯度是否存在、方向及 IC/dPL 条件一致性，不比较不同过程的原始 effect size。

## 18. Recommended Figure 3-X

推荐新增 Figure 3-X：Panel (a) 使用 frozen Snow S1–S5/frac_snow；Panel (b) 使用 ET A1–A5 的 `|Δ(ET/P)|`；Panel (c) 使用 Response B1–B5 的 `|ΔBFI_model|`。每个 panel 使用自身 y-axis，可附本 panel 的 rho/CI。figure-ready 文件位于 `figure_ready/`。

## 19. Whether Fig3-13/3-14 should be simplified

不自动替换 Fig3-13/3-14。建议保留现有 ET/Response 基础诊断图，新增 Figure 3-X 作为条件依赖补充；只有后续人工排版确认 activity figure 承担主解释功能时，才考虑简化弱 panel。本轮不修改正文图编号。

## 20. 建议进入 3.6 正文的 8–12 个新数字

{chr(10).join('- ' + item for item in numbers)}

这些数字优先来自 high-vs-low endpoint、continuous rho 和实际 valid N；完整 strata 数字放附录/figure-ready CSV。

## 21. Claim hierarchy

{claim_table}

## 22. Final 3.6 narrative update

加入本轮属性分层后，3.6 的主线应从“不同过程必须产生同等强度的结构效应”调整为“结构敏感性具有过程背景条件依赖，但这种依赖并不跨过程普遍成立”。Snow 的 frozen 高活跃度梯度仍是最清晰参照；ET 的 aridity 分层和 Response 的 CAMELS baseflow-index 分层仅按各自 primary metric、contrast 和估计 regime 解释。高 aridity 或高基流贡献不能被写成结构误差的原因，也不能把 CAMELS descriptor 与模型输出混为同一变量。弱、非单调或 IC/dPL 不一致的梯度同样是过程特异性与参数约束的信息。

## Reproducibility boundary

本轮只读取两个 repository-local CAMELS text files、canonical 531 basin IDs、既有 ET/Response outputs、既有 full-replay `evap` arrays 和 frozen Snow R5 outputs；未训练、未率定、未 replay、未增加 signature、未扫描其他属性。所有新 CI 使用 basin bootstrap draws=2000, seed=20260730；monthly climatology 不进行逐月显著性检验。
"""


def main() -> None:
    basin_ids = load_canonical_basins()
    clim, hydro = load_attributes()
    attrs = canonical_attr_maps(basin_ids, clim, hydro)
    bundle_check = verify_bundle_aridity(basin_ids, attrs["aridity"])
    if not bundle_check["match"]:
        raise RuntimeError(f"supplied aridity does not match canonical bundle: {bundle_check}")
    audit_rows = write_attribute_audit(basin_ids, clim, hydro, attrs, bundle_check)
    STRATA.mkdir(parents=True, exist_ok=True)
    a_strata, _ = make_strata(attrs["aridity"], "A")
    b_strata, _ = make_strata(attrs["baseflow_index"], "B")
    write_strata(basin_ids, attrs)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    et_rows = build_effect_records(basin_ids, attrs, a_strata, "ET")
    response_rows = build_effect_records(basin_ids, attrs, b_strata, "Response")
    write_join_audit(basin_ids, attrs, et_rows, response_rows)
    et_fields = ["regime", "contrast", "basin_id", "aridity", "A_stratum", "delta_et_over_p", "abs_delta_et_over_p", "delta_kge", "abs_delta_kge"]
    response_fields = ["regime", "contrast", "basin_id", "baseflow_index", "B_stratum", "delta_bfi_model", "abs_delta_bfi_model", "delta_kge", "abs_delta_kge"]
    write_csv(ET_OUT / "et_aridity_strata_basin.csv", et_rows, et_fields)
    write_csv(RESP_OUT / "response_bfi_attribute_strata_basin.csv", response_rows, response_fields)
    et_summary, et_summary_fields = strata_summary(et_rows, "ET", rng)
    response_summary, response_summary_fields = strata_summary(response_rows, "Response", rng)
    write_csv(ET_OUT / "et_aridity_strata_summary.csv", et_summary, et_summary_fields)
    write_csv(RESP_OUT / "response_bfi_attribute_strata_summary.csv", response_summary, response_summary_fields)
    et_endpoint = endpoint_table(et_rows, "ET", rng)
    response_endpoint = endpoint_table(response_rows, "Response", rng)
    endpoint_fields = ["regime", "contrast", "attribute", "low_stratum", "high_stratum", "effect_metric", "low_valid_n", "high_valid_n", "low_median", "high_median", "difference_high_minus_low", "ci95_low", "ci95_high"]
    write_csv(ET_OUT / "et_aridity_endpoint.csv", et_endpoint, endpoint_fields)
    write_csv(RESP_OUT / "response_baseflow_endpoint.csv", response_endpoint, endpoint_fields)
    et_gradient = gradient_table(et_rows, et_endpoint, "ET", rng)
    response_gradient = gradient_table(response_rows, response_endpoint, "Response", rng)
    gradient_fields = ["regime", "contrast", "attribute", "effect_metric", "effect_role", "finite_n", "rho", "ci95_low", "ci95_high", "strata_median_rho", "primary_classification", "bootstrap_draws", "bootstrap_seed"]
    write_csv(ET_OUT / "et_aridity_gradient.csv", et_gradient, gradient_fields)
    write_csv(RESP_OUT / "response_baseflow_gradient.csv", response_gradient, gradient_fields)
    write_monthly(basin_ids, a_strata)
    baseline_check = response_baseline_check(basin_ids, attrs["baseflow_index"], rng)
    snow_rows, snow = snow_frozen_rows()
    write_cross_matrix(et_gradient, response_gradient, snow)
    figure_ready(et_summary, et_gradient, et_endpoint, response_summary, response_gradient, response_endpoint, snow_rows)
    pattern = cross_pattern(et_gradient, response_gradient)
    definitions = json.loads((STRATA / "strata_definitions.json").read_text(encoding="utf-8"))
    report = report_text(audit_rows, definitions, et_summary, et_gradient, et_endpoint, response_summary, response_gradient, response_endpoint, baseline_check, snow, pattern)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CH3_3_6_PROCESS_ACTIVITY_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    write_json(OUT / "analysis_metadata.json", {
        "status": "complete",
        "no_training": True,
        "no_calibration": True,
        "no_replay": True,
        "no_new_hydrological_signature": True,
        "canonical_basin_n": len(basin_ids),
        "attribute_sources": {"ET": str(CLIM_PATH.relative_to(REPO)), "Response": str(HYDRO_PATH.relative_to(REPO))},
        "source_sha256": {"camels_clim": sha256(CLIM_PATH), "camels_hydro": sha256(HYDRO_PATH)},
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pattern": pattern[0],
        "pattern_statement": pattern[1],
        "response_static_attribute_not_model_bfi": True,
    })
    print(json.dumps({"status": "complete", "output": str(OUT), "pattern": pattern[0], "et_rows": len(et_rows), "response_rows": len(response_rows)}, indent=2))


if __name__ == "__main__":
    main()
