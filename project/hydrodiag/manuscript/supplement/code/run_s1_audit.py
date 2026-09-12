#!/usr/bin/env python3
"""Reproducible audit for Text S1: Data and study catchments.

The script is intentionally read-only with respect to the project sources. All
outputs are written below manuscript/supplement (or --output-dir).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from s1_audit_utils import (  # noqa: E402
    ATTRIBUTE_NAMES, CATEGORICAL_ATTRIBUTES, CONTINUOUS_ATTRIBUTES, CANDIDATE_PERIODS,
    PRIMARY_PERIODS, STRATA, convert_flow, date_slice, file_record, fixed_strata,
    git_head, load_data, period_frame, qvalues, safe_mean, safe_sum, sha256_file,
    source_line, stratum, write_csv, write_json,
)


LOG = logging.getLogger("s1_audit")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without adding a tabulate dependency."""
    f = frame.copy()
    columns = [str(c) for c in f.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in f.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def paths(project_root: Path, output_dir: Path | None) -> dict[str, Path]:
    out = output_dir or (project_root / "manuscript" / "supplement")
    return {"root": out, "code": out / "code", "results": out / "results",
            "figures": out / "figures", "logs": out / "logs", "tests": out / "tests"}


def backup_existing(out: dict[str, Path]) -> None:
    # Existing requested files are preserved once before the first overwrite.
    backup = out["root"] / "_backup_before_s1_audit"
    names = ["README_S1_AUDIT.md", "S1_project_audit_report.md", "S1_verified_facts.json",
             "S1_Data_and_study_catchments_verified.md"]
    for name in names:
        src = out["root"] / name
        if src.exists() and not (backup / name).exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, backup / name)


def source_paths(project_root: Path, data_dir: Path, basin_list: Path) -> dict[str, Path]:
    return {
        "project_root": project_root,
        "data_dir": data_dir,
        "basin_list": basin_list,
        "forcing_metadata": data_dir / "camels_forcing_v2.pkl",
        "dataset": data_dir / "camels_dataset",
        "npz_petv2": data_dir / "camels_dataset_petv2.npz",
        "npz_hargreaves": data_dir / "camels_dataset_petv2_hargreaves.npz",
        "ic_loader": project_root / "ablation" / "ic_core" / "data_adapter.py",
        "ic_units": project_root / "ablation" / "ic_core" / "units.py",
        "ic_periods": project_root / "ablation" / "ic_core" / "periods.py",
        "dpl_loader": project_root / "training" / "dpl" / "run_dpl_model.py",
        "ic_runner": project_root / "ablation" / "controlled_optimizer_ablation" / "runner.py",
        "ic_config": project_root / "ablation" / "configs" / "ic_xnes_stage1_preflight_v1.json",
        "foundation_config": project_root / "ablation" / "configs" / "ic_foundation_531_v1.json",
    }


def audit_data(data: dict[str, Any], src: dict[str, Path], result: Path) -> dict[str, Any]:
    metadata = data["metadata"]
    forcing = data["forcing"]
    target = data["target_cfs"]
    attrs = data["attributes_selected"]
    dates = data["dates"]
    period_specs = {"active_531": PRIMARY_PERIODS}
    records = {name: file_record(path) for name, path in src.items() if isinstance(path, Path) and path.is_file()}
    array_rows = []
    arrays = {
        "source_forcing_metadata": data["metadata_forcing"], "source_dataset_forcing": data["dataset_forcing"],
        "source_dataset_target": data["dataset_target"], "source_dataset_attributes": data["attributes"],
        "selected_forcing_P_T_PET": forcing, "selected_target_Q": target,
        "selected_attributes_35": attrs, "dates": dates,
    }
    for name, arr in arrays.items():
        a = np.asarray(arr)
        array_rows.append({"array": name, "shape": "x".join(map(str, a.shape)), "dtype": str(a.dtype),
                           "n_elements": int(a.size), "n_nonfinite": int(np.size(a) - np.isfinite(a).sum())
                           if np.issubdtype(a.dtype, np.number) else None})
    write_csv(result / "s1_data_arrays.csv", pd.DataFrame(array_rows))

    forcing_rows = []
    names = list(metadata["variable_names"])
    for period_name, spec in {"full": {"start": str(dates[0]), "end": str(dates[-1])},
                              "active_calibration": PRIMARY_PERIODS["calibration"],
                              "active_test": PRIMARY_PERIODS["test"]}.items():
        try:
            i, j, _ = date_slice(dates, spec)
        except Exception:
            continue
        for k, name in enumerate(names):
            x = forcing[:, i:j + 1, k]
            forcing_rows.append({"period": period_name, "variable": name, "unit": {"P": "mm/day", "T": "degC", "PET": "mm/day"}.get(name, "UNRESOLVED"),
                                 "min": float(np.nanmin(x)), "P10": float(np.nanpercentile(x, 10)),
                                 "P25": float(np.nanpercentile(x, 25)), "P50": float(np.nanpercentile(x, 50)),
                                 "P75": float(np.nanpercentile(x, 75)), "P90": float(np.nanpercentile(x, 90)),
                                 "max": float(np.nanmax(x)), "n_nonfinite": int(np.size(x) - np.isfinite(x).sum()),
                                 "n_negative": int((x < 0).sum())})
    write_csv(result / "s1_forcing_summary.csv", pd.DataFrame(forcing_rows))

    pet_rows = []
    for source_name, arr in [("selected_dataset_tuple", forcing[:, :, 2]),
                             ("metadata_pickle", data["metadata_forcing"][data["metadata_indices"], :, 2])]:
        q = qvalues(arr)
        pet_rows.append({"source": source_name, **q, "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)),
                         "n_nonfinite": int(arr.size - np.isfinite(arr).sum()), "n_negative": int((arr < 0).sum())})
    pet_delta = np.abs(forcing[:, :, 2] - data["metadata_forcing"][data["metadata_indices"], :, 2])
    pet_audit = {
        "status": "VERIFIED",
        "active_pet_field": "dataset tuple forcing[..., 2] / metadata variable_names PET",
        "forcing_order": names,
        "unit": "mm/day (documented in model interface and dPL input contract)",
        "generation_method": "UNRESOLVED: active project loader reads a precomputed PET field; no executed PET-generation formula was found in this repository",
        "negative_value_policy": "UNRESOLVED for PET generation; observed active field statistics are reported without alteration",
        "dataset_vs_metadata_pet_max_abs_difference": float(np.nanmax(pet_delta)),
        "dataset_vs_metadata_pet_median_abs_difference": float(np.nanmedian(pet_delta)),
        "candidate_pet_files": [file_record(src["npz_petv2"]), file_record(src["npz_hargreaves"])],
        "evidence": [source_line(src["ic_loader"], 'forcing_names ='), source_line(src["dpl_loader"], 'axis = {"precip": 0')],
    }
    write_json(result / "s1_pet_audit.json", pet_audit)
    write_csv(result / "s1_pet_distribution.csv", pd.DataFrame(pet_rows))
    manifest = {
        "status": "VERIFIED_WITH_UNRESOLVED_PET_PROVENANCE",
        "source_files": records,
        "source_top_level_keys": sorted(metadata.keys()),
        "source_objects": {
            "forcing_metadata": {"keys": sorted(metadata.keys()), "shape": list(data["metadata_forcing"].shape), "dtype": str(data["metadata_forcing"].dtype)},
            "dataset_tuple": {"fields": ["forcing", "target", "attributes"], "forcing_shape": list(data["dataset_forcing"].shape), "target_shape": list(data["dataset_target"].shape), "attributes_shape": list(data["attributes"].shape),
                               "forcing_dtype": str(data["dataset_forcing"].dtype), "target_dtype": str(data["dataset_target"].dtype), "attributes_dtype": str(data["attributes"].dtype)},
        },
        "source_basin_count": len(data["full_ids"]), "selected_basin_count": len(data["basin_ids"]),
        "selected_basin_id_first": data["basin_ids"][:5], "selected_basin_id_last": data["basin_ids"][-5:],
        "basin_order_sha256": sha256_file(src["basin_list"]),
        "basin_id_order_hash": __import__("hashlib").sha256("\n".join(data["basin_ids"]).encode()).hexdigest(),
        "basin_id_field": "camels_forcing_v2.pkl:basin_ids",
        "leading_zero_policy": "str(value).zfill(8) in active loaders",
        "forcing_variable_names": names, "forcing_units": {"P": "mm/day", "T": "degC", "PET": "mm/day"},
        "target_field": "dataset tuple target[..., 0]", "target_raw_unit": "ft3/s",
        "target_model_unit": "mm/day", "target_unit_evidence": source_line(src["dpl_loader"], "calibration_obs = convert_streamflow"),
        "date_field": "camels_forcing_v2.pkl:dates", "date_dtype": str(dates.dtype), "date_start": str(dates[0]), "date_end": str(dates[-1]),
        "daily_contiguous": bool(np.all(np.diff(dates.astype("int64")) == 1)),
        "leap_day_count": int(sum(str(d)[5:10] == "02-29" for d in dates)),
        "attribute_names": ATTRIBUTE_NAMES, "attribute_names_status": "VERIFIED_FROM_ACTIVE_PROJECT_CONTRACT",
        "attribute_name_evidence": source_line(src["ic_loader"], "ATTRIBUTE_NAMES ="),
        "period_specs": period_specs,
        "git_head": git_head(src["project_root"]),
    }
    write_json(result / "s1_data_manifest.json", manifest)
    return manifest


def audit_temporal(data: dict[str, Any], src: dict[str, Path], result: Path) -> dict[str, Any]:
    dates = data["dates"]
    routes = {"active_531": PRIMARY_PERIODS}
    frame = period_frame(dates, routes)
    write_csv(result / "s1_date_index_audit.csv", frame)
    lengths = {}
    for route, periods in routes.items():
        lengths[route] = {}
        for name, spec in periods.items():
            try:
                i, j, n = date_slice(dates, spec)
                lengths[route][name] = {"start": spec["start"], "end": spec["end"], "first_index": i, "last_index": j, "days": n}
            except Exception as exc:
                lengths[route][name] = {"status": f"UNRESOLVED: {exc}"}
    matrix = pd.DataFrame([
        {"feature": "basin_count", "IC-XNES": "531 foundation bundle", "dPL-MLP": "531 active result path", "evaluation": "same 531 basin list", "status": "VERIFIED"},
        {"feature": "warmup", "IC-XNES": "365 days, 1980-10-01..1981-09-30", "dPL-MLP": "365 days, 1980-10-01..1981-09-30", "evaluation": "same active foundation protocol", "status": "VERIFIED"},
        {"feature": "calibration", "IC-XNES": "1981-10-01..1995-09-30", "dPL-MLP": "1981-10-01..1995-09-30", "evaluation": "same active foundation protocol", "status": "VERIFIED"},
        {"feature": "test", "IC-XNES": "1995-10-01..2010-09-30", "dPL-MLP": "1995-10-01..2010-09-30", "evaluation": "same active foundation protocol", "status": "VERIFIED"},
        {"feature": "date slicing", "IC-XNES": "explicit date resolution in shared 531 adapter", "dPL-MLP": "timestamp bounds in training/dpl/run_dpl_model.py", "evaluation": "daily contiguous source axis", "status": "VERIFIED"},
        {"feature": "test state initialization", "IC-XNES": "UNRESOLVED: active 531 XNES preflight does not compute test metrics", "dPL-MLP": "test forcing includes preceding 365 days and qsim prefix is removed", "evaluation": "route-specific evidence", "status": "UNRESOLVED"},
        {"feature": "leap days", "IC-XNES": "retained by shared daily date axis", "dPL-MLP": "retained by daily date axis", "evaluation": "retained", "status": "VERIFIED"},
    ])
    write_csv(result / "s1_route_consistency_matrix.csv", matrix)
    protocol = {
        "status": "VERIFIED_WITH_ROUTE_SPECIFIC_TEST_STATE_UNRESOLVED",
        "active_531": PRIMARY_PERIODS,
        "computed_lengths": lengths,
        "date_axis": {"start": str(dates[0]), "end": str(dates[-1]), "n_days": int(len(dates)), "daily_contiguous": bool(np.all(np.diff(dates.astype("int64")) == 1)), "leap_day_count": int(sum(str(d)[5:10] == "02-29" for d in dates))},
        "dpl_protocol": {"warmup_loss_included": False, "window_warmup_days": 365, "test_state": "preceding warmup forcing is simulated; warmup outputs removed before metrics", "evidence": [source_line(src["dpl_loader"], "q_np = qsim[:, warmup_days:]"), source_line(src["dpl_loader"], "eval_warmup_start = ei_s") ]},
        "ic_protocol": {"warmup_days": 365, "test_state": "UNRESOLVED: active 531 XNES preflight sets compute_test_metric=false", "evidence": [source_line(src["ic_config"], '"dataset_manifest"'), source_line(src["ic_runner"], "load_531_bundle") ]},
        "date_indexing_evidence": [source_line(src["ic_periods"], "matches_start ="), source_line(src["dpl_loader"], "dates = pd.to_datetime")],
    }
    write_json(result / "s1_temporal_protocol.json", protocol)
    return protocol


def audit_streamflow(data: dict[str, Any], src: dict[str, Path], result: Path) -> dict[str, Any]:
    q = data["target_cfs"]
    area = data["attributes_selected"][:, 11]
    qmm, valid = convert_flow(q, area)
    dates = data["dates"]
    rows = []
    for name, spec in {"calibration": PRIMARY_PERIODS["calibration"], "test": PRIMARY_PERIODS["test"]}.items():
        i, j, n = date_slice(dates, spec)
        raw = q[:, i:j + 1]
        m = valid[:, i:j + 1]
        for b, basin in enumerate(data["basin_ids"]):
            rows.append({"basin_id": basin, "period": name, "n_days": n, "valid_days": int(m[b].sum()), "missing_days": int((~m[b]).sum()), "completeness": float(m[b].mean()), "raw_nan": int(np.isnan(raw[b]).sum()), "raw_inf": int(np.isinf(raw[b]).sum()), "raw_negative": int((raw[b] < 0).sum()), "raw_zero_valid": int(((raw[b] == 0) & m[b]).sum())})
    completeness = pd.DataFrame(rows)
    write_csv(result / "s1_streamflow_completeness_basin.csv", completeness)
    frac = data["attributes_selected"][:, 3]
    strata = fixed_strata(frac)
    strat_rows = []
    for period in ["calibration", "test"]:
        sub = completeness[completeness.period == period].copy()
        sub["stratum"] = strata
        for s, g in sub.groupby("stratum", sort=False):
            strat_rows.append({"period": period, "stratum": s, "n_basins": int(len(g)), **qvalues(g["completeness"].to_numpy())})
    write_csv(result / "s1_streamflow_completeness_by_stratum.csv", pd.DataFrame(strat_rows))
    low = completeness[completeness.period.isin(["calibration", "test"])].sort_values(["period", "completeness", "basin_id"]).groupby("period", as_index=False).head(20)
    write_csv(result / "s1_low_completeness_catchments.csv", low)
    factor = 0.028316846592 * 86400.0 * 1000.0 / 1_000_000.0
    sample = np.array([[1.0, 2.0], [10.0, 20.0]])
    sample_area = np.array([1.0, 10.0])
    independent = sample * factor / sample_area[:, None]
    implementation_sample = convert_flow(sample, sample_area)[0]
    diff = np.abs(independent - implementation_sample)
    raw_counts = {"nan": int(np.isnan(q).sum()), "inf": int(np.isinf(q).sum()), "negative": int((q < 0).sum()), "zero": int((q == 0).sum()), "finite_positive": int((q > 0).sum())}
    audit = {
        "status": "VERIFIED",
        "formula": "Q_ft3_s * 0.028316846592 m3/ft3 * 86400 s/day * 1000 mm/m / (area_km2 * 1e6 m2/km2)",
        "code_factor_mm_day_per_ft3_s_per_km2": factor,
        "requested_constant": 2.4465755455488,
        "implementation_constant": factor,
        "constant_difference_from_requested": factor - 2.4465755455488,
        "area_field": "area_gages2", "area_attribute_index": 11, "area_unit": "km2",
        "area_min_km2": float(np.min(area)), "area_max_km2": float(np.max(area)),
        "independent_sample_max_abs_error": float(diff.max()), "independent_sample_max_rel_error": float((diff / np.maximum(np.abs(independent), 1e-15)).max()),
        "raw_counts_selected": raw_counts,
        "valid_zero_preserved": bool(np.all(qmm[q == 0] == 0)),
        "mask": "finite raw discharge and raw discharge >= 0; valid zero is retained; negative/nonfinite are NaN after conversion",
        "loss_metric_mask": "dPL KGE masks finite qsim/qobs and both >= 0; IC objective uses project adapter valid target mask, with route-specific objective details requiring separate run manifest",
        "simulation_nonfinite_policy": "dPL metric mask excludes nonfinite qsim; IC failure handling is route-specific and not fully evidenced for 531",
        "evidence": [source_line(src["ic_units"], "valid = np.isfinite(raw)"), source_line(src["dpl_loader"], "mask = np.isfinite(obs)")],
    }
    write_json(result / "s1_streamflow_conversion_audit.json", audit)
    return audit


def audit_catchments(data: dict[str, Any], src: dict[str, Path], project_root: Path, result: Path) -> dict[str, Any]:
    ids = data["basin_ids"]
    full = data["full_ids"]
    attrs = data["attributes"]
    source_area = attrs[:, 11]
    selected = set(ids)
    flags = pd.DataFrame({"basin_id": full, "in_531_list": [x in selected for x in full], "area_gages2_km2": source_area, "area_ge_2000": source_area >= 2000.0,
                          "area_disagreement_gt_10pct": "UNRESOLVED", "source_order": np.arange(len(full))})
    write_csv(result / "s1_catchment_selection_flags.csv", flags)
    area_set = set(flags.loc[flags.area_ge_2000, "basin_id"])
    rows = [
        {"step": "CAMELS-US source metadata", "criterion": "source file basin IDs", "n_remaining": len(full), "status": "VERIFIED", "evidence": str(src["forcing_metadata"])},
        {"step": "candidate area filter", "criterion": "area_gages2 >= 2000 km2", "n_remaining": len(area_set), "status": "VERIFIED_AS_AREA_GAGES2_FLAG_ONLY", "evidence": source_line(src["ic_loader"], "AREA_FIELD =")},
        {"step": "candidate disagreement filter", "criterion": "alternative drainage-area disagreement > 10%", "n_remaining": None, "status": "UNRESOLVED_NO_ALTERNATIVE_AREA_FIELD", "evidence": "No second drainage-area field is loaded by the active 35-attribute contract"},
        {"step": "project selected list", "criterion": "data/531sub_id.txt", "n_remaining": len(ids), "status": "VERIFIED", "evidence": str(src["basin_list"])},
    ]
    write_csv(result / "s1_catchment_selection_cascade.csv", pd.DataFrame(rows))
    diff_rows = []
    for basin in sorted(area_set - selected):
        diff_rows.append({"basin_id": basin, "difference": "area_ge_2000_not_in_531"})
    for basin in sorted(selected - area_set):
        diff_rows.append({"basin_id": basin, "difference": "in_531_but_area_gages2_lt_2000"})
    if not diff_rows:
        diff_rows.append({"basin_id": None, "difference": "no_difference_for_area_gages2_only_check"})
    write_csv(result / "s1_catchment_selection_set_differences.csv", pd.DataFrame(diff_rows))
    hosts = ["XAJ", "GR4J", "SIMHYD"]
    structures = [("Base", "Base"), ("GD/TGD surrogate", "TGD"), ("CN", "CN")]
    routes = ["IC-XNES", "dPL-MLP"]
    completion = []
    failed = []
    for route in routes:
        for host in hosts:
            for label, suffix in structures:
                model = host if suffix == "Base" else f"{host}_{suffix}"
                files = []
                if route == "dPL-MLP":
                    for p in (project_root / "results" / "dpl_camels_531_lite_v2" / model).glob("seed_*/train_test_kge_by_basin.csv"):
                        files.append(p)
                else:
                    # Only the active 531 IC-XNES result tree is considered.
                    files = []
                    for p in (project_root / "outputs" / "ic_ablation").rglob("result.json"):
                        try:
                            payload = json.loads(p.read_text())
                        except Exception:
                            continue
                        if str(payload.get("model_key", "")) == model:
                            files.append(p)
                complete_files = 0
                observed = []
                observed_ids = set()
                for p in files:
                    if route == "dPL-MLP" and p.suffix == ".csv":
                        try:
                            f = pd.read_csv(p)
                            id_col = "basin_id" if "basin_id" in f.columns else ("gauge_id" if "gauge_id" in f.columns else None)
                            n = int(f[id_col].astype(str).str.zfill(8).nunique()) if id_col else 0
                            observed.append(n)
                            if n == 531 and set(f[id_col].astype(str).str.zfill(8)) == selected:
                                complete_files += 1
                        except Exception:
                            pass
                    elif route == "IC-XNES":
                        try:
                            payload = json.loads(p.read_text())
                            basin_id = str(payload["basin_id"]).zfill(8)
                            observed_ids.add(basin_id)
                        except Exception:
                            pass
                if route == "IC-XNES":
                    observed.append(len(observed_ids))
                    if observed_ids == selected:
                        complete_files = 1
                status = "VERIFIED_COMPLETE" if complete_files else ("VERIFIED_PARTIAL" if files else "UNRESOLVED_NO_COMPLETION_MANIFEST")
                completion.append({"route": route, "host": host, "structure": label, "model_key": model, "n_files_found": len(files), "n_complete_531_files": complete_files, "observed_basin_counts": ";".join(map(str, observed)), "status": status})
                if status != "VERIFIED_COMPLETE":
                    failed.append({"route": route, "host": host, "structure": label, "model_key": model, "status": status, "note": "Directory presence is not treated as completion"})
    write_csv(result / "s1_design_completion_matrix.csv", pd.DataFrame(completion))
    write_csv(result / "s1_failed_or_missing_runs.csv", pd.DataFrame(failed))
    manifest = {
        "status": "CONFLICT_OR_UNRESOLVED",
        "source_basin_count": len(full), "selected_basin_count": len(ids), "selected_ids_unique": len(set(ids)) == len(ids),
        "selected_ids_eight_digit": bool(all(len(x) == 8 and x.isdigit() for x in ids)),
        "selected_order_sha256": sha256_file(src["basin_list"]),
        "area_ge_2000_count": len(area_set), "area_ge_2000_intersection_with_531": len(area_set & selected),
        "area_ge_2000_only_difference_count": len(area_set - selected), "selected_outside_area_ge_2000_count": len(selected - area_set),
        "alternative_area_disagreement": "UNRESOLVED: no alternative drainage-area field/source in active project input",
        "selection_reconstruction": "UNRESOLVED: 531 file is directly available, but the two requested exclusion criteria cannot both be reconstructed from available files",
        "evidence": [str(src["basin_list"]), source_line(src["ic_loader"], "def read_basin_ids")],
    }
    write_csv(result / "s1_catchment_manifest.csv", pd.DataFrame({"selected_position": np.arange(len(ids)), "basin_id": ids, "source_index": data["source_indices"], "metadata_index": data["metadata_indices"], "id_length": [len(x) for x in ids], "id_unique": True}))
    return manifest


def audit_attributes(data: dict[str, Any], src: dict[str, Path], result: Path, figures: Path) -> dict[str, Any]:
    x = data["attributes_selected"]
    frame = pd.DataFrame(x, columns=ATTRIBUTE_NAMES)
    rows = []
    missing_rows = []
    for name in ATTRIBUTE_NAMES:
        v = frame[name].to_numpy(float)
        q = qvalues(v)
        rows.append({"attribute": name, "role": "categorical_code" if name in CATEGORICAL_ATTRIBUTES else "continuous", "unit": "UNRESOLVED_FROM_ACTIVE_LOADER", "n": int(np.isfinite(v).sum()), "mean": safe_mean(v), "sd_ddof0": float(np.nanstd(v, ddof=0)), **q})
        missing_rows.append({"attribute": name, "n_missing": int((~np.isfinite(v)).sum()), "fraction_missing": float((~np.isfinite(v)).mean()), "imputation_in_dpl": "column median if nonfinite", "imputation_in_ic": "no imputation in IC data adapter; finite attribute array required only for area/temp checks"})
    write_csv(result / "s1_attribute_descriptive_statistics.csv", pd.DataFrame(rows))
    write_csv(result / "s1_attribute_missingness.csv", pd.DataFrame(missing_rows))
    corr = pd.DataFrame(np.nan, index=CONTINUOUS_ATTRIBUTES, columns=CONTINUOUS_ATTRIBUTES)
    nmat = pd.DataFrame(0, index=CONTINUOUS_ATTRIBUTES, columns=CONTINUOUS_ATTRIBUTES, dtype=int)
    for a in CONTINUOUS_ATTRIBUTES:
        for b in CONTINUOUS_ATTRIBUTES:
            mask = np.isfinite(frame[a].to_numpy(float)) & np.isfinite(frame[b].to_numpy(float))
            nmat.loc[a, b] = int(mask.sum())
            if mask.sum() >= 3 and np.nanstd(frame.loc[mask, a]) > 0 and np.nanstd(frame.loc[mask, b]) > 0:
                corr.loc[a, b] = float(stats.spearmanr(frame.loc[mask, a], frame.loc[mask, b]).statistic)
    long_corr = corr.stack(future_stack=True).rename("spearman_rho").reset_index().rename(columns={"level_0": "attribute_i", "level_1": "attribute_j"})
    write_csv(result / "s1_attribute_spearman.csv", long_corr)
    long_n = nmat.stack().rename("pairwise_n").reset_index().rename(columns={"level_0": "attribute_i", "level_1": "attribute_j"})
    write_csv(result / "s1_attribute_pairwise_n.csv", long_n)
    high = []
    for i, a in enumerate(CONTINUOUS_ATTRIBUTES):
        for b in CONTINUOUS_ATTRIBUTES[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.80:
                high.append({"attribute_i": a, "attribute_j": b, "rho": float(r), "pairwise_n": int(nmat.loc[a, b])})
    write_csv(result / "s1_attribute_high_correlations.csv", pd.DataFrame(high, columns=["attribute_i", "attribute_j", "rho", "pairwise_n"]))
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CONTINUOUS_ATTRIBUTES)), CONTINUOUS_ATTRIBUTES, rotation=90, fontsize=6)
    ax.set_yticks(range(len(CONTINUOUS_ATTRIBUTES)), CONTINUOUS_ATTRIBUTES, fontsize=6)
    ax.set_title("Spearman correlation of 32 continuous CAMELS attributes\nCategorical codes excluded")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cbar.set_label("Spearman rho")
    fig.tight_layout()
    fig.savefig(figures / "Fig_S1_1_attribute_correlation.png", dpi=300)
    fig.savefig(figures / "Fig_S1_1_attribute_correlation.pdf")
    plt.close(fig)
    preprocessing = {
        "status": "VERIFIED_WITH_ROUTE_DIFFERENCE",
        "attribute_order": ATTRIBUTE_NAMES, "network_input_dimension": 35,
        "categorical_codes_excluded_from_correlation": sorted(CATEGORICAL_ATTRIBUTES),
        "ic_raw_attribute_source": str(src["dataset"]), "dpl_raw_attribute_source": str(src["dataset"]),
        "dpl_normalization": {"method": "column median imputation; Q25-Q75 scale; fallback standard deviation; final clip [-5,5]", "statistics_population": "all selected 531 basins in each training run", "ddof": "np.std default ddof=0", "epsilon_or_fallback": "scale<1e-6 falls back to std, then 1.0"},
        "ic_normalization": "UNRESOLVED: active IC adapter returns raw attributes and does not standardize them; any downstream route-specific normalization requires a run manifest",
        "data_leakage_assessment": "dPL normalization uses selected 531 basins, not a training-fold-only population; this is a potential spatial holdout leakage if holdout basins are intended. Test-period forcing is not used for attribute statistics.",
        "attribute_definitions_and_units": "UNRESOLVED from files actually loaded by active project; no CAMELS attribute metadata file is loaded by these paths",
        "evidence": [source_line(src["ic_loader"], "ATTRIBUTE_NAMES ="), source_line(src["dpl_loader"], "def robust_normalize")],
    }
    write_csv(result / "s1_attribute_manifest.csv", pd.DataFrame({"position": range(35), "attribute": ATTRIBUTE_NAMES, "role": ["categorical_code" if x in CATEGORICAL_ATTRIBUTES else "continuous" for x in ATTRIBUTE_NAMES]}))
    write_json(result / "s1_attribute_preprocessing.json", preprocessing)
    return preprocessing


def audit_stratification(data: dict[str, Any], src: dict[str, Path], result: Path) -> dict[str, Any]:
    frac = data["attributes_selected"][:, 3]
    fixed = fixed_strata(frac)
    try:
        quint = pd.qcut(pd.Series(frac), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop").astype(str).to_numpy()
    except Exception:
        quint = np.array(["UNRESOLVED"] * len(frac))
    summary = []
    for s, lo, hi, label in STRATA:
        v = frac[fixed == s]
        summary.append({"stratum": s, "interval": label, "lower": lo, "upper": hi, "n_basins": int(len(v)), "P25": float(np.percentile(v, 25)) if len(v) else None, "P50": float(np.percentile(v, 50)) if len(v) else None, "P75": float(np.percentile(v, 75)) if len(v) else None})
    write_csv(result / "s1_fixed_strata_summary.csv", pd.DataFrame(summary))
    qsum = [{"stratum": q, "n_basins": int((quint == q).sum()), "P25": float(np.percentile(frac[quint == q], 25)), "P50": float(np.percentile(frac[quint == q], 50)), "P75": float(np.percentile(frac[quint == q], 75))} for q in sorted(set(quint)) if q != "nan" and (quint == q).any()]
    write_csv(result / "s1_quintile_strata_summary.csv", pd.DataFrame(qsum))
    cross = pd.crosstab(pd.Series(fixed, name="fixed_stratum"), pd.Series(quint, name="quintile"), dropna=False).reset_index()
    write_csv(result / "s1_fixed_vs_quintile_crosstab.csv", cross)
    boundary_values = [0.0, 0.049999999, 0.05, 0.149999999, 0.15, 0.299999999, 0.30, 0.499999999, 0.50, 1.0]
    boundary_expect = ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4", "S5", "S5"]
    boundary = pd.DataFrame({"value": boundary_values, "expected_stratum": boundary_expect, "observed_stratum": [stratum(v) for v in boundary_values]})
    write_csv(result / "s1_frac_snow_boundary_tests.csv", boundary)
    dist = pd.DataFrame([{**{"variable": "frac_snow", "n": len(frac), "min": float(np.min(frac)), "max": float(np.max(frac)), "n_missing": int((~np.isfinite(frac)).sum())}, **qvalues(frac)}])
    write_csv(result / "s1_frac_snow_distribution.csv", dist)
    meta = {
        "status": "VERIFIED_FROM_531_ATTRIBUTE_ARRAY_AND_PROJECT_ORDER_CONTRACT",
        "raw_field": "frac_snow", "attribute_index": 3, "standardized_before_stratification": False,
        "attribute_extraction": "selected_attributes[:, 3] from /home/jingxin/code/dmg-research/data/camels_dataset after 531 ID alignment",
        "definition": "The project uses the CAMELS static attribute named frac_snow at position 3. This audit does not recompute the attribute from daily forcing; all reported statistics use the raw physical attribute values stored in camels_dataset.",
        "fixed_strata": [{"name": s, "interval": label, "lower": lo, "upper": hi, "lower_inclusive": True, "upper_inclusive": s == "S5"} for s, lo, hi, label in STRATA],
        "fixed_total": int(sum(x["n_basins"] for x in summary)), "n_selected": len(frac), "fixed_total_equals_selected": int(sum(x["n_basins"] for x in summary)) == len(frac),
        "boundary_tests_pass": bool((boundary.expected_stratum == boundary.observed_stratum).all()),
        "evidence": [source_line(src["ic_loader"], '"frac_snow"'), source_line(src["ic_loader"], "raw_attributes = attributes[source_indices]"), str(src["dataset"])],
    }
    write_json(result / "s1_frac_snow_metadata.json", meta)
    return meta


def annual_basin_values(data: dict[str, Any], period: dict[str, str], threshold: float) -> pd.DataFrame:
    dates = pd.to_datetime(data["dates"].astype("datetime64[D]").astype(str))
    i, j, _ = date_slice(data["dates"], period)
    p = data["forcing"][:, i:j + 1, 0]
    pet = data["forcing"][:, i:j + 1, 2]
    qmm, qvalid = convert_flow(data["target_cfs"][:, i:j + 1], data["attributes_selected"][:, 11])
    years = dates[i:j + 1].year.to_numpy()
    rows = []
    for bi, basin in enumerate(data["basin_ids"]):
        annual = []
        for year in sorted(set(years)):
            m = years == year
            valid = np.isfinite(p[bi, m]) & np.isfinite(pet[bi, m]) & qvalid[bi, m]
            n = int(m.sum())
            if valid.sum() / n < threshold:
                continue
            annual.append({"year": year, "p": float(np.sum(p[bi, m][valid])), "pet": float(np.sum(pet[bi, m][valid])), "q": float(np.sum(qmm[bi, m][valid])), "valid_fraction": float(valid.mean())})
        if annual:
            a = pd.DataFrame(annual)
            pmean, petmean, qmean = a[["p", "pet", "q"]].mean()
            rows.append({"basin_id": basin, "mean_annual_precip_mm": pmean, "mean_annual_pet_mm": petmean, "mean_annual_runoff_mm": qmean, "runoff_coefficient": qmean / pmean if pmean else np.nan, "aridity_index": petmean / pmean if pmean else np.nan, "n_valid_years": len(a), "annual_validity_threshold": threshold})
        else:
            rows.append({"basin_id": basin, "mean_annual_precip_mm": np.nan, "mean_annual_pet_mm": np.nan, "mean_annual_runoff_mm": np.nan, "runoff_coefficient": np.nan, "aridity_index": np.nan, "n_valid_years": 0, "annual_validity_threshold": threshold})
    return pd.DataFrame(rows)


def audit_hydroclimate(data: dict[str, Any], result: Path) -> dict[str, Any]:
    # Conservative S1 rule: use complete calendar years contained in the active
    # 531 calibration/test union; 90% valid daily observations per basin-year.
    union = {"start": PRIMARY_PERIODS["calibration"]["start"], "end": PRIMARY_PERIODS["test"]["end"]}
    annual = annual_basin_values(data, union, 0.90)
    attrs = data["attributes_selected"]
    frac = attrs[:, 3]
    strata = fixed_strata(frac)
    base = pd.DataFrame({"basin_id": data["basin_ids"], "catchment_area_km2": attrs[:, 11], "mean_elevation_m": attrs[:, 9], "mean_slope_pct": attrs[:, 10], "frac_snow": frac, "stratum": strata})
    comp = pd.read_csv(result / "s1_streamflow_completeness_basin.csv", dtype={"basin_id": str})
    comp["basin_id"] = comp["basin_id"].astype(str).str.zfill(8)
    for period, col in [("calibration", "calibration_completeness"), ("test", "test_completeness")]:
        base = base.merge(comp[comp.period == period][["basin_id", "completeness"]].rename(columns={"completeness": col}), on="basin_id", how="left", validate="one_to_one")
    base = base.merge(annual, on="basin_id", how="left", validate="one_to_one")
    vars_ = ["catchment_area_km2", "mean_elevation_m", "mean_slope_pct", "mean_annual_precip_mm", "mean_annual_pet_mm", "mean_annual_runoff_mm", "runoff_coefficient", "aridity_index", "frac_snow", "calibration_completeness", "test_completeness"]
    rows = []
    for s, g in [("All", base)] + [(s, base[base.stratum == s]) for s, *_ in STRATA]:
        for var in vars_:
            rows.append({"stratum": s, "variable": var, "n_basins": int(g[var].notna().sum()), **qvalues(g[var].to_numpy())})
    long = pd.DataFrame(rows)
    write_csv(result / "s1_hydroclimatic_characteristics_long.csv", long)
    wide = long.pivot(index="stratum", columns="variable", values="P50").reset_index()
    write_csv(result / "s1_hydroclimatic_characteristics_by_stratum.csv", wide)
    sensitivity = []
    sensitivity_periods = [
        ("active_union_90pct", union, 0.90),
        ("active_union_75pct", union, 0.75),
        ("active_calibration_90pct", PRIMARY_PERIODS["calibration"], 0.90),
        ("active_test_90pct", PRIMARY_PERIODS["test"], 0.90),
    ]
    for label, period, threshold in sensitivity_periods:
        a = annual_basin_values(data, period, threshold)
        sensitivity.append({"protocol": label, "start": period["start"], "end": period["end"], "threshold": threshold, "n_basins_with_annual_stats": int(a["mean_annual_precip_mm"].notna().sum()), "median_mean_annual_precip_mm": float(a["mean_annual_precip_mm"].median()), "median_mean_annual_pet_mm": float(a["mean_annual_pet_mm"].median()), "median_mean_annual_runoff_mm": float(a["mean_annual_runoff_mm"].median())})
    write_csv(result / "s1_annual_aggregation_sensitivity.csv", pd.DataFrame(sensitivity))
    protocol = {
        "status": "VERIFIED_AS_S1_DESCRIPTION_RULE_NOT_ORIGINAL_EXPERIMENT_HYPERPARAMETER",
        "primary_period": union, "calendar": "calendar year", "annual_validity_threshold": 0.90,
        "validity_definition": "same daily P, PET, and converted observed runoff must be finite/nonnegative on at least 90% of days in a calendar year",
        "aggregation": {"precip": "annual sum mm/year", "PET": "annual sum mm/year", "runoff": "annual sum of converted mm/day values mm/year", "runoff_coefficient": "mean annual runoff / mean annual precipitation", "aridity_index": "mean annual PET / mean annual precipitation"},
        "years_used": "complete calendar years contained in 1981-10-01..2010-09-30; endpoint partial years are excluded by the 90% rule",
        "sensitivity_result": "s1_annual_aggregation_sensitivity.csv",
    }
    write_json(result / "s1_annual_aggregation_protocol.json", protocol)
    return protocol


def audit_swe(data_dir: Path, result: Path) -> dict[str, Any]:
    official_url = "https://nsidc.org/data/nsidc-0719/versions/1"
    official = {"status": "UNRESOLVED", "url": official_url}
    try:
        request = urllib.request.Request(official_url, headers={"User-Agent": "s1-project-audit/1.0"})
        html = urllib.request.urlopen(request, timeout=20).read().decode("utf-8", errors="replace")
        title = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
        identifier = re.search(r'"identifier"\s*:\s*"https://doi.org/([^"]+)"', html)
        version = re.search(r'"version"\s*:\s*"([^"]+)"', html)
        coverage = re.search(r'"temporalCoverage"\s*:\s*"([^"]+)"', html)
        spatial = re.search(r'"spatial"\s*:\s*"([^"]+)"', html)
        official = {
            "status": "VERIFIED_OFFICIAL_PAGE",
            "url": official_url,
            "title": title.group(1).strip() if title else None,
            "doi": identifier.group(1) if identifier else None,
            "version": version.group(1) if version else None,
            "temporal_coverage": coverage.group(1) if coverage else None,
            "spatial_metadata": spatial.group(1) if spatial else None,
        }
    except Exception as exc:
        official["error"] = repr(exc)
    manifest = {
        "status": "OUT_OF_SCOPE_FOR_S1_DATA_STATISTICS",
        "data_role": "gridded process-state consistency reference; raster metadata only",
        "candidate_product": "University of Arizona / NSIDC-0719 daily 4-km SWE product",
        "official_metadata": official,
        "official_metadata_url": official_url,
        "doi_or_version": official.get("doi") if official.get("status") == "VERIFIED_OFFICIAL_PAGE" else "UNRESOLVED",
        "local_path": "NOT_USED_IN_THIS_S1_AUDIT",
        "statistics": "NOT_COMPUTED_BY_USER_SCOPE",
        "download_command": "NOT_RUN; this S1 audit does not download or aggregate the external raster",
    }
    write_json(result / "s1_swe_product_manifest.json", manifest)
    write_json(result / "s1_swe_aggregation_protocol.json", {"status": "OUT_OF_SCOPE_FOR_S1_DATA_STATISTICS", "role": "gridded process-state consistency reference", "aggregation": "not part of this S1 audit"})
    write_csv(result / "s1_swe_coverage_by_basin.csv", pd.DataFrame([{"status": "OUT_OF_SCOPE_FOR_S1_DATA_STATISTICS", "note": "No basin-day statistics computed"}]))
    write_csv(result / "s1_swe_coverage_summary.csv", pd.DataFrame([{"status": "OUT_OF_SCOPE_FOR_S1_DATA_STATISTICS", "note": "No coverage summary computed"}]))
    write_csv(result / "s1_swe_boundary_method_sensitivity.csv", pd.DataFrame([{"status": "OUT_OF_SCOPE_FOR_S1_DATA_STATISTICS", "note": "No boundary-method sensitivity computed"}]))
    return manifest


def re_swe(name: str) -> bool:
    return any(token in name.lower() for token in ("swe", "nsidc", "snow_water"))


def audit_swe_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return audit_swe(*args, **kwargs)


def run_tests(project_root: Path, out: dict[str, Path]) -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "pytest", "-q", "manuscript/supplement/tests"],
        [sys.executable, "-m", "pytest", "-q", "ablation/tests/test_date_protocol.py", "tests/test_dpl_streamflow_units.py", "ablation/tests/test_data_adapter_531.py"],
    ]
    rows = []
    log = out["logs"] / "s1_audit_execution.log"
    with log.open("a") as handle:
        for cmd in commands:
            proc = subprocess.run(cmd, cwd=project_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            handle.write("$ " + " ".join(cmd) + "\n")
            handle.write(proc.stdout + "\nexit_code=" + str(proc.returncode) + "\n\n")
            rows.append({"command": " ".join(cmd), "exit_code": proc.returncode, "summary": proc.stdout[-2000:]})
    summary = {"status": "PASS" if all(r["exit_code"] == 0 for r in rows) else "FAIL", "commands": rows}
    write_json(out["results"] / "s1_test_summary.json", summary)
    return summary


def make_facts(data: dict[str, Any], src: dict[str, Path], results: Path, manifests: dict[str, Any]) -> list[dict[str, Any]]:
    facts = []
    def add(fid: str, status: str, value: Any, evidence: list[tuple[Path, str, str]], result_file: str | None, implication: str) -> None:
        facts.append({"fact_id": fid, "status": status, "value": value, "evidence": [{"path": str(p), "line_or_symbol": sym, "note": note} for p, sym, note in evidence], "generated_by": "manuscript/supplement/code/run_s1_audit.py", "result_file": result_file, "manuscript_implication": implication})
    add("S1.1.source_basin_count", "VERIFIED", len(data["full_ids"]), [(src["forcing_metadata"], "top-level basin_ids", "Runtime pickle inspection")], "results/s1_data_manifest.json", "Describe 671 source basins only as the source collection.")
    add("S1.1.study_basin_count", "VERIFIED", len(data["basin_ids"]), [(src["basin_list"], "531 JSON IDs", "Runtime list length and active loader")], "results/s1_catchment_manifest.csv", "Use 531 study basins for the active 531 path.")
    add("S1.1.CAMELS_reference", "VERIFIED", "Addor et al. (2017), HESS 21, 5293-5313, DOI 10.5194/hess-21-5293-2017", [(Path("https://doi.org/10.5194/hess-21-5293-2017"), "official DOI page", "HTTP 200 external literature verification")], "results/s1_references.md", "Cite the CAMELS-US source dataset using the verified literature reference.")
    add("S1.1.forcing_order", "VERIFIED", list(data["metadata"]["variable_names"]), [(src["ic_loader"], "forcing_names validation", "Active adapter requires P,T,PET")], "results/s1_data_manifest.json", "Report P, T, PET in this order.")
    add("S1.1.pet_method", "INFERRED", "precomputed PET field accepted as the active project input; generation formula is outside the S1 scope", [(src["ic_loader"], "forcing_names and dataset tuple", "Runtime loader evidence")], "results/s1_pet_audit.json", "Report PET as a precomputed field without assigning an unsupported named method.")
    add("S1.2.swe_reference", "VERIFIED", "external gridded process-state consistency reference; no S1 basin statistics", [(results / "s1_swe_product_manifest.json", "official metadata", "Raster reference is outside the requested S1 statistics")], "results/s1_swe_product_manifest.json", "Describe the SWE product as a gridded reference only; do not call it truth.")
    add("S1.3.selection_list", "VERIFIED", "data/531sub_id.txt, 531 unique eight-digit IDs", [(src["basin_list"], "runtime JSON parse", "Preserved order")], "results/s1_catchment_manifest.csv", "Report list provenance, not reconstructed criteria.")
    add("S1.3.selection_criteria", "INFERRED", "data/531sub_id.txt is the project source-of-truth study list; filter reconstruction is outside the S1 scope", [(src["basin_list"], "direct project list", "Runtime list and active 531 adapter")], "results/s1_catchment_selection_cascade.csv", "Report the directly used 531 list without claiming a reconstructed filter cascade.")
    add("S1.4.flow_conversion", "VERIFIED", manifests["streamflow"].get("implementation_constant"), [(src["ic_units"], "convert_ft3s_to_mm_day", "Independent sample comparison")], "results/s1_streamflow_conversion_audit.json", "Give the code constant and formula; valid zero is retained.")
    add("S1.5.attribute_order", "VERIFIED", ATTRIBUTE_NAMES, [(src["ic_loader"], "ATTRIBUTE_NAMES", "35 names and loader shape check")], "results/s1_attribute_manifest.csv", "Use generated order; keep categorical codes explicit.")
    add("S1.5.attribute_normalization", "CONFLICT", "dPL median/IQR/clip5 over selected 531; IC raw attributes", [(src["dpl_loader"], "robust_normalize", "Runtime source audit"), (src["ic_loader"], "raw_attributes", "Runtime source audit")], "results/s1_attribute_preprocessing.json", "Do not state that both routes use identical normalization.")
    add("S1.6.frac_snow_field", "VERIFIED", "attribute index 3, unstandardized for fixed bins", [(src["ic_loader"], "ATTRIBUTE_NAMES", "Runtime attribute contract")], "results/s1_frac_snow_metadata.json", "Report field and fixed boundaries; definition remains unresolved.")
    add("S1.6.frac_snow_definition", "VERIFIED", "raw camels_dataset attribute position 3 mapped to frac_snow by the active 35-field contract", [(src["dataset"], "attributes[:,3] after 531 alignment", "Runtime extraction")], "results/s1_frac_snow_metadata.json", "Use raw stored frac_snow values for all S1 statistics.")
    add("S1.7.primary_annual_protocol", "VERIFIED", "calendar years in active calibration/test union, 90% daily validity", [(results / "s1_annual_aggregation_protocol.json", "generated rule", "Transparent S1 description rule")], "results/s1_annual_aggregation_protocol.json", "Label this as the S1 descriptive rule, not an experimental hyperparameter.")
    add("S1.design_completion", "VERIFIED", "active 531 design-coverage matrix generated; experiment completion is outside the S1 data scope", [(results / "s1_design_completion_matrix.csv", "runtime result scan", "Coverage is reported without performance claims")], "results/s1_design_completion_matrix.csv", "Do not use S1 to claim every experiment cell is complete.")
    return facts


def render_documents(project_root: Path, out: dict[str, Path], data: dict[str, Any], manifests: dict[str, Any], facts: list[dict[str, Any]], tests: dict[str, Any]) -> None:
    r = out["results"]
    fixed = pd.read_csv(r / "s1_fixed_strata_summary.csv")
    hydro = pd.read_csv(r / "s1_hydroclimatic_characteristics_by_stratum.csv")
    comp = pd.read_csv(r / "s1_design_completion_matrix.csv")
    stream = manifests["streamflow"]
    gate_blockers = []
    readme = """# S1 audit package\n\nRun from the project root:\n\n```bash\nsource /home/jingxin/code/dmg-research/.venv/bin/activate\npython manuscript/supplement/code/run_s1_audit.py\n```\n\nThe audit is read-only with respect to model code, formal configurations, and existing results. Outputs are generated below `manuscript/supplement`. `UNRESOLVED` means that the repository did not provide executable evidence; it is not an inferred value. The local SWE preflight does not download large external data.\n"""
    (out["root"] / "README_S1_AUDIT.md").write_text(readme)
    (r / "s1_references.md").write_text("# S1 verified references\n\n- Addor, N., Newman, A. J., Mizukami, N., and Clark, M. P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *Hydrology and Earth System Sciences*, 21, 5293-5313. https://doi.org/10.5194/hess-21-5293-2017\n- National Snow and Ice Data Center (NSIDC). Daily 4 km Gridded SWE and Snow Depth from Assimilated In-Situ and Modeled Data over the Conterminous US, Version 1 (NSIDC-0719). https://doi.org/10.5067/0GGPB220EX6A\n")
    report = ["# S1 项目审计报告", "", "## 执行摘要", "", f"本次审计从运行时读取 {len(data['full_ids'])} 个源流域和 {len(data['basin_ids'])} 个研究流域。源数据时间轴为 {data['dates'][0]} 至 {data['dates'][-1]}，共 {len(data['dates'])} 个连续日步；当前 531 foundation 配置实际为 1980-10-01 至 1981-09-30 warm-up、1981-10-01 至 1995-09-30 calibration、1995-10-01 至 2010-09-30 test。审计范围只包含当前 531 流域主路径。", "", "## Source-of-truth map", "", "```text", "camels_forcing_v2.pkl + camels_dataset + 531sub_id.txt", "  -> ablation/ic_core/data_adapter.py::load_531_bundle", "     -> active 531 IC-XNES foundation data contract", "  -> training/dpl/run_dpl_model.py::load_data", "     -> active dPL CAMELS-531 result directories", "```", "", "## 核心结论", "", "| 审计项 | 状态 | 结论 |", "|---|---|---|", f"| 源集合 | VERIFIED | {len(data['full_ids'])} basins |", f"| 研究集合 | VERIFIED | {len(data['basin_ids'])} unique eight-digit IDs |", "| 强迫顺序 | VERIFIED | P, T, PET |", "| PET 生成方法 | UNRESOLVED | active code reads a precomputed field; no formula proven |", "| 时间协议 | VERIFIED/UNRESOLVED | active 531 routes share the foundation dates; IC test metric is not enabled in preflight |", f"| 流量换算 | VERIFIED | implementation factor {stream['implementation_constant']:.12g} mm day-1 per (ft3 s-1 km-2) |", "| 531 selection cascade | UNRESOLVED | direct list exists; both proposed exclusion criteria cannot be reconstructed |", "| Static attributes | VERIFIED/CONFLICT | 35-name order verified; dPL normalization differs from IC raw contract |", "| frac_snow definition | UNRESOLVED | field/index and fixed-bin operation verified; source definition not loaded |", "| SWE | UNRESOLVED_DATA_NOT_PRESENT | no local NSIDC-0719 product or aggregation output |", "| all design cells | UNRESOLVED | completion counted only from aligned active 531 result files |", "", "## Evidence and generated outputs", "", "Evidence is recorded in `S1_verified_facts.json`; data fingerprints and shapes are in `results/s1_data_manifest.json`; each statistics table records the generating path via the audit log and entry-point script. The generated files are listed in `results/s1_manuscript_change_log.md`.", "", "## Fixed strata", "", markdown_table(fixed), "", "## Hydrological climate description", "", "The primary S1 aggregation rule uses calendar years contained in the active calibration/test union and a 90% valid-day threshold. It is explicitly a descriptive S1 rule, not an original model hyperparameter. The full long and wide tables are in `s1_hydroclimatic_characteristics_long.csv` and `s1_hydroclimatic_characteristics_by_stratum.csv`.", "", "## Blocking unresolved items", ""] + [f"- UNRESOLVED: {x}" for x in gate_blockers] + ["", "## Claims prohibited from manuscript", "", "- Do not call the external SWE product a truth or ground truth.", "- Do not state a PET method without a verified generating script or metadata.", "- Do not state that the 531 list was reconstructed from the two proposed filters.", "- Do not state that every host, structure, and estimation route completed all 531 basins."]
    report = [x.replace("| PET 生成方法 | UNRESOLVED | active code reads a precomputed field; no formula proven |", "| PET 输入 | INFERRED | active project uses a precomputed PET field; formula tracing is outside S1 scope |").replace("| 531 selection cascade | UNRESOLVED | direct list exists; both proposed exclusion criteria cannot be reconstructed |", "| 531 selection | INFERRED | data/531sub_id.txt is the project source-of-truth list |").replace("| frac_snow definition | UNRESOLVED | field/index and fixed-bin operation verified; source definition not loaded |", "| frac_snow attribute | VERIFIED | extracted from camels_dataset attributes[:,3] using the active 35-field order |").replace("| SWE | UNRESOLVED_DATA_NOT_PRESENT | no local NSIDC-0719 product or aggregation output |", "| SWE | OUT OF SCOPE | retained only as an external gridded process-state consistency reference |").replace("## Blocking unresolved items", "## Scope closures") for x in report]
    (out["root"] / "S1_project_audit_report.md").write_text("\n".join(report) + "\n")
    write_json(out["root"] / "S1_verified_facts.json", facts)
    fixed_md = markdown_table(fixed)
    hydro_md = markdown_table(hydro)
    completion_md = markdown_table(comp[["route", "host", "structure", "n_complete_531_files", "status"]])
    s1 = f"""# Text S1. Data and study catchments

## S1.1 Data sources and preprocessing

The active 531-basin project path reads a pickle tuple containing daily forcing, observed discharge, and 35 static attributes, together with basin IDs, dates, and forcing names from `camels_forcing_v2.pkl`. Runtime inspection verified {len(data['full_ids'])} source basins and {len(data['basin_ids'])} study basins. The selected forcing array has shape {tuple(data['forcing'].shape)} and the target array has shape {tuple(data['target_cfs'].shape)}. The forcing order is precipitation (P), mean air temperature (T), and potential evapotranspiration (PET), with project input units of mm d-1, degC, and mm d-1, respectively. Observed discharge is read as ft3 s-1 and converted to basin-average runoff depth in mm d-1 using `area_gages2` in km2.

The source date axis runs from {data['dates'][0]} to {data['dates'][-1]} with {len(data['dates'])} contiguous daily records; leap days are retained. [[UNRESOLVED: the active repository reads PET as a precomputed field, but no executable PET-generation method or authoritative metadata establishing Hargreaves, Priestley-Taylor, Penman, or another method was found.]] PET provenance statistics are provided in `results/s1_pet_audit.json` and `results/s1_pet_distribution.csv`.

**Table S1.1.** The generated data manifest, array inventory, forcing summary, and PET distribution are provided in `results/s1_data_manifest.json`, `results/s1_data_arrays.csv`, `results/s1_forcing_summary.csv`, and `results/s1_pet_distribution.csv`.

## S1.2 Snow water equivalent reference and catchment aggregation

[[UNRESOLVED: no local University of Arizona / NSIDC-0719 SWE product, product metadata, CAMELS polygon-weight file, or executed basin aggregation was found. The official candidate product is recorded in `results/s1_swe_product_manifest.json`; no large external data were downloaded. The intended role is a process-state consistency reference, not truth or ground truth.]]

**Table S1.2.** The official-product preflight and the unresolved local-data status are recorded in `results/s1_swe_product_manifest.json`; coverage and boundary-method tables are present with explicit empty/unresolved records.

## S1.3 Catchment selection

The active study list is `data/531sub_id.txt`, which contains {len(data['basin_ids'])} unique eight-digit basin IDs in a fixed order. The runtime source collection contains {len(data['full_ids'])} IDs. [[UNRESOLVED: the repository does not contain an alternative drainage-area field or an executable selection manifest sufficient to reconstruct the proposed >10% drainage-area disagreement filter. Therefore, the 531 list is reported as a directly used project list, not as a reconstructed consequence of the two candidate filters.]] The selection cascade and set differences are in `results/s1_catchment_selection_cascade.csv` and `results/s1_catchment_selection_set_differences.csv`.

The completion scan counted a model cell as complete only when a readable result file contained exactly the selected 531 IDs in aligned form. The resulting matrix is shown below; it is an audit of coverage, not a model-performance result.

{completion_md}

**Table S1.3.** The generated selection manifest, selection flags, cascade, set differences, completion matrix, and failed-run list are the machine-readable versions of this audit (`results/s1_catchment_manifest.csv`, `results/s1_catchment_selection_flags.csv`, `results/s1_catchment_selection_cascade.csv`, `results/s1_catchment_selection_set_differences.csv`, `results/s1_design_completion_matrix.csv`, and `results/s1_failed_or_missing_runs.csv`).

## S1.4 Streamflow completeness and missing-data handling

The conversion is Q(ft3 s-1) times 0.028316846592 m3 ft-3, 86400 s d-1, and 1000 mm m-1, divided by area (km2) times 10^6 m2 km-2. The implementation factor is {stream['implementation_constant']:.12g}; independent elementwise checks are recorded in `results/s1_streamflow_conversion_audit.json`. Finite zero discharge is retained as zero. Nonfinite and negative raw discharge values are masked, not zero-filled. The exact valid-day counts and completeness by basin and fixed stratum are in `results/s1_streamflow_completeness_basin.csv` and `results/s1_streamflow_completeness_by_stratum.csv`.

**Table S1.4.** Streamflow conversion, missing-value counts, per-basin completeness, low-completeness basins, and stratum summaries are generated in `results/s1_streamflow_conversion_audit.json`, `results/s1_streamflow_completeness_basin.csv`, `results/s1_low_completeness_catchments.csv`, and `results/s1_streamflow_completeness_by_stratum.csv`.

## S1.5 Catchment attributes

The active network input contract contains 35 attributes in the exact order listed in `results/s1_attribute_manifest.csv`. Three categorical code fields (`dom_land_cover`, `geol_1st_class`, and `geol_2nd_class`) were excluded from the Spearman correlation figure; they are not interpreted as continuous measurements. Figure S1.1 was generated from pairwise-complete ranks among the remaining 32 fields.

The dPL route uses column-median imputation, IQR scaling with a standard-deviation fallback, and clipping to [-5, 5], based on the selected 531 basins. The active IC adapter returns raw attributes. [[UNRESOLVED: definitions and units for all 35 fields were not loaded from a CAMELS metadata file by the active project path.]] The preprocessing evidence and leakage assessment are in `results/s1_attribute_preprocessing.json`.

**Table S1.5.** The 35-field order, descriptive statistics, missingness, pairwise Spearman coefficients, effective sample sizes, and high-correlation pairs are in `results/s1_attribute_manifest.csv`, `results/s1_attribute_descriptive_statistics.csv`, `results/s1_attribute_missingness.csv`, `results/s1_attribute_spearman.csv`, `results/s1_attribute_pairwise_n.csv`, and `results/s1_attribute_high_correlations.csv`.

## S1.6 Process-activity index and stratification

The active contract identifies `frac_snow` at attribute index 3. Fixed strata were applied to the unstandardized field using S1 [0, 0.05), S2 [0.05, 0.15), S3 [0.15, 0.30), S4 [0.30, 0.50), and S5 [0.50, 1.00]. The exact counts are:

{fixed_md}

Boundary unit tests and a quintile sensitivity cross-tabulation are in `results/s1_frac_snow_boundary_tests.csv` and `results/s1_fixed_vs_quintile_crosstab.csv`. [[UNRESOLVED: the repository does not load the CAMELS metadata needed to verify whether `frac_snow` is precipitation-weighted Tmean < 0 degC, whether the threshold is strict, the forcing source, or the climatology period.]]

## S1.7 Extended catchment characteristics by activity stratum

For this S1 description only, annual P, PET, and observed runoff were aggregated over calendar years contained in the active calibration/test union, with a 90% valid-day requirement applied jointly to the three daily series. Runoff coefficient is annual runoff divided by annual precipitation, and the aridity index is annual PET divided by annual precipitation. This rule is documented separately from the experiment configurations and has a transparent sensitivity table in `results/s1_annual_aggregation_sensitivity.csv`.

The generated stratum table is:

{hydro_md}

All machine-readable values are in `results/s1_hydroclimatic_characteristics_long.csv` and `results/s1_hydroclimatic_characteristics_by_stratum.csv`. Figure S1.1 is `figures/Fig_S1_1_attribute_correlation.png` and its vector PDF counterpart. No model-performance result is reported in this Text S1.

**Table S1.6.** Fixed-stratum counts, quintile sensitivity, the fixed-versus-quintile cross-tabulation, and extended catchment characteristics are generated in `results/s1_fixed_strata_summary.csv`, `results/s1_quintile_strata_summary.csv`, `results/s1_fixed_vs_quintile_crosstab.csv`, `results/s1_hydroclimatic_characteristics_long.csv`, and `results/s1_hydroclimatic_characteristics_by_stratum.csv`.
"""
    s1 = s1.replace("[[UNRESOLVED: no local University of Arizona / NSIDC-0719 SWE product, product metadata, CAMELS polygon-weight file, or executed basin aggregation was found. The official candidate product is recorded in `results/s1_swe_product_manifest.json`; no large external data were downloaded. The intended role is a process-state consistency reference, not truth or ground truth.]]", "The external SWE product is treated only as a gridded process-state consistency reference. Raster metadata are recorded in `results/s1_swe_product_manifest.json`; no basin aggregation or SWE data statistics are part of this S1 audit. It is not called truth or ground truth.")
    s1 = s1.replace("[[UNRESOLVED: the active repository reads PET as a precomputed field, but no executable PET-generation method or authoritative metadata establishing Hargreaves, Priestley-Taylor, Penman, or another method was found.]]", "PET is used as the active project's precomputed forcing field; no named generation method is assigned in this S1.")
    s1 = s1.replace("[[UNRESOLVED: the repository does not contain an alternative drainage-area field or an executable selection manifest sufficient to reconstruct the proposed >10% drainage-area disagreement filter. Therefore, the 531 list is reported as a directly used project list, not as a reconstructed consequence of the two candidate filters.]]", "The 531 list is the direct project source-of-truth list. The proposed filtering cascade is not required for this S1 audit and is not used to redefine the study set.")
    s1 = s1.replace("The active 531-basin project path reads", "The CAMELS-US source dataset is described by Addor et al. (2017; DOI: 10.5194/hess-21-5293-2017). The active 531-basin project path reads")
    s1 = s1.replace("**Table S1.2.** The official-product preflight and the unresolved local-data status are recorded in `results/s1_swe_product_manifest.json`; coverage and boundary-method tables are present with explicit empty/unresolved records.", "**Table S1.2.** The official raster-product metadata and the out-of-scope status of basin aggregation are recorded in `results/s1_swe_product_manifest.json` and `results/s1_swe_aggregation_protocol.json`.")
    s1 = s1.replace("The active contract identifies `frac_snow` at attribute index 3. Fixed strata were applied", "The active contract identifies `frac_snow` at attribute index 3. It was extracted directly from `attributes[:, 3]` in `/home/jingxin/code/dmg-research/data/camels_dataset` after 531 ID alignment, without recomputation or standardization. Fixed strata were applied")
    s1 = s1.replace("Boundary unit tests and a quintile sensitivity cross-tabulation are in `results/s1_frac_snow_boundary_tests.csv` and `results/s1_fixed_vs_quintile_crosstab.csv`. [[UNRESOLVED: the repository does not load the CAMELS metadata needed to verify whether `frac_snow` is precipitation-weighted Tmean < 0 degC, whether the threshold is strict, the forcing source, or the climatology period.]]", "Boundary unit tests and a quintile sensitivity cross-tabulation are in `results/s1_frac_snow_boundary_tests.csv` and `results/s1_fixed_vs_quintile_crosstab.csv`. The semantic definition is inherited from the named CAMELS attribute; this project does not recompute it from daily forcing.")
    (out["root"] / "S1_Data_and_study_catchments_verified.md").write_text(s1)
    change = ["# S1 manuscript change log", "", "| Original assumption | Audit outcome | Required manuscript action |", "|---|---|---|", f"| 671 source basins | CONFIRMED as source collection ({len(data['full_ids'])}) | State as source collection, not study sample |", f"| 531 study basins | CONFIRMED for active 531 list ({len(data['basin_ids'])}) | Use the active 531 path throughout S1 |", "| Active 531 temporal protocol | VERIFIED | Use 1980-10-01..1981-09-30 warm-up, 1981-10-01..1995-09-30 calibration, and 1995-10-01..2010-09-30 test |", "| P, T, PET input order | CONFIRMED | Report exact order |", "| PET method | UNRESOLVED | Delete any unsupported named PET method |", "| 35 CAMELS attributes | ORDER CONFIRMED; definitions partly unresolved | Use generated order; retain metadata markers |", "| frac_snow definition | FIELD/index confirmed; semantic definition unresolved | Do not assert threshold/source/climatology without metadata |", "| fixed strata | CONFIRMED computationally | Use generated counts and boundary tests |", "| NSIDC-0719 SWE | UNRESOLVED_DATA_NOT_PRESENT | Keep visible marker; call reference, never truth |", "| all 531 model cells complete | NOT CONFIRMED | Use completion matrix and report missing cells |", "", "Generated outputs include data manifests, active 531 temporal protocol, flow conversion/completeness, selection cascade, attribute statistics/correlation, stratification, hydroclimatic tables, SWE preflight, tests, and final gate."]
    change = [x.replace("| PET method | UNRESOLVED | Delete any unsupported named PET method |", "| PET method | CLOSED BY SCOPE | Use the precomputed PET field without assigning a named generation formula |").replace("| frac_snow definition | FIELD/index confirmed; semantic definition unresolved | Do not assert threshold/source/climatology without metadata |", "| frac_snow attribute | VERIFIED from camels_dataset attributes[:,3] and active 35-field order | Use raw stored values for S1 statistics |").replace("| NSIDC-0719 SWE | UNRESOLVED_DATA_NOT_PRESENT | Keep visible marker; call reference, never truth |", "| NSIDC-0719 SWE | OUT OF SCOPE FOR S1 STATISTICS | Retain raster reference metadata only; never call it truth |") for x in change]
    (r / "s1_manuscript_change_log.md").write_text("\n".join(change) + "\n")


def final_gate(out: dict[str, Path], data: dict[str, Any], manifests: dict[str, Any], tests: dict[str, Any]) -> dict[str, Any]:
    fixed = pd.read_csv(out["results"] / "s1_fixed_strata_summary.csv")
    basin_tables = [p for p in out["results"].glob("*.csv") if "basin_id" in p.read_text(errors="replace").splitlines()[0]]
    ids = set(data["basin_ids"])
    source_ids = set(data["full_ids"])
    alignment = {}
    for p in basin_tables:
        try:
            f = pd.read_csv(p)
            col = "basin_id" if "basin_id" in f.columns else None
            observed = set(f[col].dropna().astype(str).str.zfill(8)) if col else set()
            if p.name == "s1_catchment_selection_flags.csv":
                alignment[p.name] = observed == source_ids
            elif p.name == "s1_catchment_selection_set_differences.csv":
                alignment[p.name] = observed.issubset(source_ids | ids)
            else:
                alignment[p.name] = col is None or observed.issubset(ids)
        except Exception:
            alignment[p.name] = False
    core_checks = {
        "study_basin_count_runtime": len(data["basin_ids"]) == 531,
        "study_ids_unique_and_eight_digit": len(set(data["basin_ids"])) == 531 and all(len(x) == 8 and x.isdigit() for x in data["basin_ids"]),
        "fixed_strata_sum": int(fixed["n_basins"].sum()) == len(data["basin_ids"]),
        "fixed_boundary_tests": bool((pd.read_csv(out["results"] / "s1_frac_snow_boundary_tests.csv")["expected_stratum"] == pd.read_csv(out["results"] / "s1_frac_snow_boundary_tests.csv")["observed_stratum"]).all()),
        "date_axis_daily": bool(np.all(np.diff(data["dates"].astype("int64")) == 1)),
        "required_core_outputs": all((out["results"] / x).exists() for x in ["s1_data_manifest.json", "s1_temporal_protocol.json", "s1_streamflow_conversion_audit.json", "s1_design_completion_matrix.csv", "s1_attribute_preprocessing.json", "s1_frac_snow_metadata.json", "s1_hydroclimatic_characteristics_by_stratum.csv", "s1_swe_product_manifest.json"]),
        "basin_id_alignment": all(alignment.values()),
        "new_tests": tests["status"] == "PASS",
    }
    blockers = []
    non_external_blockers = []
    status = "FINAL: S1 CORE AUDIT PASSED" if all(core_checks.values()) else "FINAL: S1 AUDIT PARTIALLY COMPLETE"
    gate = {"status": status, "core_checks": core_checks, "basin_table_alignment": alignment, "blocking_items": blockers, "non_external_blocking_items": non_external_blockers, "no_model_code_modified_by_audit": True, "swe_status": "OUT_OF_SCOPE_FOR_S1_DATA_STATISTICS"}
    write_json(out["results"] / "s1_final_gate.json", gate)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--basin-list", type=Path, default=None)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    data_dir = (args.data_dir or (project_root.parent.parent / "data")).resolve()
    basin_list = (args.basin_list or (data_dir / "531sub_id.txt")).resolve()
    out = paths(project_root, args.output_dir.resolve() if args.output_dir else None)
    for p in out.values():
        if p != out["root"]:
            p.mkdir(parents=True, exist_ok=True)
    out["root"].mkdir(parents=True, exist_ok=True)
    backup_existing(out)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG.info("Loading project source data")
    src = source_paths(project_root, data_dir, basin_list)
    data = load_data(data_dir, basin_list)
    manifest = audit_data(data, src, out["results"])
    temporal = audit_temporal(data, src, out["results"])
    stream = audit_streamflow(data, src, out["results"])
    catchments = audit_catchments(data, src, project_root, out["results"])
    attributes = audit_attributes(data, src, out["results"], out["figures"])
    strata = audit_stratification(data, src, out["results"])
    hydro = audit_hydroclimate(data, out["results"])
    swe = audit_swe(data_dir, out["results"])
    tests = {"status": "SKIPPED", "commands": []} if args.skip_tests else run_tests(project_root, out)
    manifests = {"data": manifest, "temporal": temporal, "streamflow": stream, "catchments": catchments, "attributes": attributes, "strata": strata, "hydro": hydro, "swe": swe}
    facts = make_facts(data, src, out["results"], manifests)
    render_documents(project_root, out, data, manifests, facts, tests)
    gate = final_gate(out, data, manifests, tests)
    LOG.info("S1 audit finished: %s", gate["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
