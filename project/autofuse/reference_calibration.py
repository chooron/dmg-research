"""Bounded, serial calibration using the pinned original Fortran FUSE driver.

This module is deliberately separate from dFUSE and never uses the S3 solver to
choose parameters.  It reproduces the paper's file-manager/SCE entry point and
stores only extracted best parameter vectors plus auditable provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import resource
import shutil
import subprocess
import tempfile
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import netcdf_file

from dfuse.spec import get_structure
from project.autofuse.fidelity import synthetic_forcing
from project.autofuse.reference_oracle import (
    _parameter_values,
    _write_elevation_bands,
    _write_forcing,
    _write_input_info,
    run_reference,
)

ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "vendor/upstream/fuse-mmcomparison-paper"
TEMPLATE_SETTINGS = PAPER_ROOT / "01_FUSEscripts/fuse_template/settings"
MANIFEST_PATH = ROOT / "project/autofuse/manifests/camels_544.json"
BUNDLE_PATH = ROOT / "data/camels_dataset"
GAGE_PATH = ROOT / "data/gage_id.npy"
BUNDLE_START = date(1980, 10, 1)
FORCING_START = date(1987, 1, 1)
SIMULATION_END = date(2009, 12, 31)
CALIBRATION_START = date(1989, 1, 1)
CALIBRATION_END = date(1998, 12, 31)
EVALUATION_START = date(1999, 1, 1)
EVALUATION_END = date(2009, 12, 31)
MOTHER_MODELS = (2, 108, 178, 210)
RSS_STOP_KB = 3_500_000
CALIBRATION_TIMEOUT_SECONDS = 7200
ATTRIBUTE_NAMES = (
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max",
    "lai_diff", "gvf_max", "gvf_diff", "dom_land_cover_frac", "dom_land_cover",
    "root_depth_50", "soil_depth_pelletier", "soil_depth_statsgo",
    "soil_porosity", "soil_conductivity", "max_water_content", "sand_frac",
    "silt_frac", "clay_frac", "geol_1st_class", "glim_1st_class_frac",
    "geol_2nd_class", "glim_2nd_class_frac", "carbonate_rocks_frac",
    "geol_porosity", "geol_permeability",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _period_slice(start: date, end: date) -> slice:
    return slice((start - BUNDLE_START).days, (end - BUNDLE_START).days + 1)


def _load_bundle(data_path: Path, gage_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with data_path.open("rb") as handle:
        forcing, target, attributes = pickle.load(handle)
    ids = np.load(gage_path).reshape(-1).astype(np.int64)
    forcing = np.asarray(forcing, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    attributes = np.asarray(attributes, dtype=np.float64)
    if forcing.shape[0] != ids.size or target.shape[:2] != forcing.shape[:2] or attributes.shape[0] != ids.size:
        raise ValueError("CAMELS bundle arrays are not aligned")
    if forcing.shape[1] != (date(2014, 9, 30) - BUNDLE_START).days + 1:
        raise ValueError("unexpected CAMELS bundle date axis")
    return forcing, target, attributes, ids


def _elevation_metadata(data_root: Path, gauge: str, fallback_elev: float) -> tuple[np.ndarray, np.ndarray, Path | None]:
    base = data_root / "camels_attributes_v2.0/elev_bands_forcing/daymet"
    matches = sorted(base.glob(f"*/{gauge}.list"))
    listing = matches[0] if matches else None
    if listing is None or not listing.is_file():
        return np.asarray([1.0]), np.asarray([fallback_elev]), None
    lines = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty elevation-band list: {listing}")
    entries = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 2:
            continue
        filename, area = fields[0], float(fields[1])
        match = re.search(r"_elev_band_(\d{3})_", filename)
        if match is None:
            raise ValueError(f"cannot recover elevation-band index from {filename}")
        entries.append((int(match.group(1)) * 100.0 + 50.0, area))
    if not entries or sum(area for _, area in entries) <= 0.0:
        raise ValueError(f"no usable elevation bands in {listing}")
    elevation = np.asarray([e for e, _ in entries], dtype=np.float64)
    area = np.asarray([a for _, a in entries], dtype=np.float64)
    return area / area.sum(), elevation, listing


def catchment_case(
    basin_id: str,
    data_root: Path,
    forcing: np.ndarray,
    target: np.ndarray,
    attributes: np.ndarray,
    ids: np.ndarray,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    hru = int(basin_id.split("_")[-1])
    lookup = {int(value): i for i, value in enumerate(ids)}
    if hru not in lookup:
        raise KeyError(f"{basin_id} is absent from CAMELS bundle")
    index = lookup[hru]
    window = _period_slice(FORCING_START, SIMULATION_END)
    raw_forcing = forcing[index, window, :]
    q_cfs = target[index, window, 0]
    if not np.isfinite(raw_forcing).all() or not np.isfinite(q_cfs).all() or (q_cfs < 0.0).any():
        raise ValueError(f"incomplete or negative CAMELS data for {basin_id}")
    area_km2 = float(attributes[index, 11])
    if not np.isfinite(area_km2) or area_km2 <= 0.0:
        raise ValueError(f"invalid CAMELS area for {basin_id}")
    q_mm_day = q_cfs * 0.0283168 * 86400.0 / (area_km2 * 1000.0)
    if not np.isfinite(q_mm_day).all() or float(q_mm_day.mean()) <= 0.0:
        raise ValueError(f"invalid converted streamflow for {basin_id}")
    manifest_row = next(row for row in manifest["basins"] if row["basin_id"] == basin_id)
    area_frac, mean_elev, elevation_listing = _elevation_metadata(data_root, f"{hru:08d}", float(manifest_row["elev_mean_m"]))
    dates = _date_range(FORCING_START, SIMULATION_END)
    return {
        "basin_id": basin_id,
        "hru_id": hru,
        "forcing": {
            "ppt": raw_forcing[:, 0],
            "temp": raw_forcing[:, 1],
            "pet": raw_forcing[:, 2],
            "q_obs": q_mm_day,
            "area_frac": area_frac,
            "mean_elev": mean_elev,
        },
        "dates": dates,
        "attributes": {name: float(attributes[index, i]) for i, name in enumerate(ATTRIBUTE_NAMES)},
        "source": {
            "bundle_row": int(index),
            "elevation_listing": str(elevation_listing) if elevation_listing else None,
            "elevation_listing_sha256": sha256_file(elevation_listing) if elevation_listing else None,
            "initial_protocol": "fracState0=0.25",
        },
    }


def select_catchments(
    data_root: Path,
    data_path: Path,
    gage_path: Path,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    forcing, target, attributes, ids = _load_bundle(data_path, gage_path)
    manifest = json.loads(manifest_path.read_text())
    eligible = []
    for row in manifest["basins"]:
        try:
            case = catchment_case(row["basin_id"], data_root, forcing, target, attributes, ids, manifest)
        except (KeyError, ValueError):
            continue
        eligible.append(case)
    if len(eligible) < 12:
        raise RuntimeError(f"only {len(eligible)} complete 544-member CAMELS cases are available")
    aridity = np.asarray([case["attributes"]["aridity"] for case in eligible])
    snow = np.asarray([case["attributes"]["frac_snow"] for case in eligible])
    aridity_cut = np.quantile(aridity, [1.0 / 3.0, 2.0 / 3.0])
    snow_cut = np.quantile(snow, [1.0 / 3.0, 2.0 / 3.0])
    selected = []
    candidate_counts = {}
    for aridity_group, lower, upper in (("wet", -np.inf, aridity_cut[0]), ("intermediate", aridity_cut[0], aridity_cut[1]), ("dry", aridity_cut[1], np.inf)):
        for snow_group, snow_lower, snow_upper in (("low_snow", -np.inf, snow_cut[0]), ("high_snow", snow_cut[1], np.inf)):
            candidates = [case for case in eligible if lower <= case["attributes"]["aridity"] < upper and snow_lower <= case["attributes"]["frac_snow"] < snow_upper]
            candidates.sort(key=lambda case: (case["attributes"]["aridity"], case["hru_id"]))
            candidate_counts[f"{aridity_group}:{snow_group}"] = len(candidates)
            if len(candidates) < 2:
                raise RuntimeError(f"stratum {aridity_group}:{snow_group} has only {len(candidates)} complete cases")
            for case in (candidates[0], candidates[-1]):
                selected.append({
                    "basin_id": case["basin_id"], "hru_id": case["hru_id"],
                    "aridity_group": aridity_group, "snow_group": snow_group,
                    "attributes": case["attributes"], "source": case["source"],
                })
    selected.sort(key=lambda row: row["hru_id"])
    return {
        "schema_version": "landscape-12-catchment-manifest-v1",
        "status": "frozen before solver comparison",
        "selection_rule": "complete 1987-01-01..2009-12-31 forcing/Qobs; aridity tertiles; low/high snow outer tertiles; select min/max aridity within each stratum; sort by hru_id",
        "source": {
            "bundle_path": str(data_path.resolve()), "bundle_sha256": sha256_file(data_path),
            "gage_id_path": str(gage_path.resolve()), "gage_id_sha256": sha256_file(gage_path),
            "catalogue_manifest": str(manifest_path.resolve()), "catalogue_manifest_sha256": sha256_file(manifest_path),
            "data_root": str(data_root.resolve()), "forcing_product": "CAMELS basin_mean_forcing/daymet-equivalent bundle",
        },
        "periods": {"forcing": [FORCING_START.isoformat(), SIMULATION_END.isoformat()], "calibration": [CALIBRATION_START.isoformat(), CALIBRATION_END.isoformat()], "evaluation": [EVALUATION_START.isoformat(), EVALUATION_END.isoformat()]},
        "attribute_names": list(ATTRIBUTE_NAMES),
        "thresholds": {"aridity_tertiles": aridity_cut.tolist(), "snow_tertiles": snow_cut.tolist()},
        "eligible_complete_count": len(eligible), "candidate_counts": candidate_counts,
        "catchments": selected,
    }


def _write_file_manager(path: Path, case: Path, domain_id: str, model_id: int, sim_start: date, sim_end: date, eval_start: date, eval_end: date, maxn: int, kstop: int = 3, pcento: float = 0.001) -> None:
    settings = str(case / "settings") + os.sep
    input_dir = str(case / "input") + os.sep
    output_dir = str(case / "output") + os.sep
    lines = [
        "FUSE_FILEMANAGER_V1.5",
        "! *** paths",
        f"'{settings}' ! SETNGS_PATH",
        f"'{input_dir}' ! INPUT_PATH",
        f"'{output_dir}' ! OUTPUT_PATH",
        "! *** suffixes for input files",
        "'_input.nc' ! suffix_forcing",
        "'_elev_bands.nc' ! suffix_elev_bands",
        "! *** settings files",
        "'input_info.txt' ! FORCING INFO",
        "'fuse_zConstraints_snow.txt' ! CONSTRAINTS",
        "'fuse_zNumerix.txt' ! MOD_NUMERIX",
        f"'fuse_zDecisions_{model_id}.txt' ! M_DECISIONS",
        "! *** output files",
        f"'{model_id}' ! FMODEL_ID",
        "'FALSE' ! Q_ONLY",
        "! *** dates",
        f"'{sim_start.isoformat()}' ! date_start_sim",
        f"'{sim_end.isoformat()}' ! date_end_sim",
        f"'{eval_start.isoformat()}' ! date_start_eval",
        f"'{eval_end.isoformat()}' ! date_end_eval",
        "'-9999' ! numtim_sub",
        "! *** evaluation metrics and transformation",
        "'KGECOMP' ! METRIC",
        "'1' ! TRANSFO",
        "! *** SCE parameters",
        f"'{maxn}' ! MAXN",
        f"'{kstop}' ! KSTOP",
        f"'{pcento:.12g}' ! PCENTO",
    ]
    path.write_text("\n".join(lines) + "\n")


def _copy_settings(case: Path, model_id: int) -> None:
    settings = case / "settings"
    settings.mkdir(parents=True, exist_ok=True)
    (case / "input").mkdir(parents=True, exist_ok=True)
    (case / "output").mkdir(parents=True, exist_ok=True)
    for name in ("fuse_zConstraints_snow.txt", "fuse_zNumerix.txt"):
        shutil.copy2(TEMPLATE_SETTINGS / name, settings / name)
    shutil.copy2(TEMPLATE_SETTINGS / "fuse_zDecisions" / f"fuse_zDecisions_{model_id}.txt", settings / f"fuse_zDecisions_{model_id}.txt")


def _run_mode(exe: Path, case: Path, domain_id: str, mode: str, env: Mapping[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        process = subprocess.run([str(exe), str(case / "fm_catch.txt"), domain_id, mode], cwd=case, env=dict(env), text=True, capture_output=True, timeout=CALIBRATION_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {"mode": mode, "returncode": -124, "timed_out": True, "elapsed_seconds": elapsed, "stdout_tail": stdout[-3000:], "stderr_tail": stderr[-3000:]}
    elapsed = time.perf_counter() - started
    return {"mode": mode, "returncode": process.returncode, "timed_out": False, "elapsed_seconds": elapsed, "stdout_tail": process.stdout[-3000:], "stderr_tail": process.stderr[-3000:]}


def _read_best(path: Path, model_id: int) -> tuple[dict[str, float], dict[str, Any]]:
    spec = get_structure(model_id)
    with netcdf_file(str(path), "r", mmap=False) as nc:
        values = {}
        for name in spec.parameter_names:
            if name not in nc.variables:
                raise RuntimeError(f"calibration parameter output missing {name}")
            value = float(np.asarray(nc.variables[name].data, dtype=np.float64).reshape(-1)[0])
            if not np.isfinite(value) or value <= -9000.0:
                raise RuntimeError(f"invalid best calibrated parameter {name}={value}")
            values[name] = value
        diagnostics = {}
        for name in ("metric_val", "kgecomp", "kge", "kgep", "raw_rmse", "log_rmse"):
            if name in nc.variables:
                diagnostics[name] = float(np.asarray(nc.variables[name].data, dtype=np.float64).reshape(-1)[0])
    return values, diagnostics


def calibrate_one(exe: Path, case_data: Mapping[str, Any], model_id: int, output_root: Path, maxn: int, kstop: int, pcento: float, source_label: str) -> dict[str, Any]:
    domain_id = f"{int(case_data['hru_id']):08d}"
    model_root = output_root / "reference_calibration_logs"
    model_root.mkdir(parents=True, exist_ok=True)
    dates = case_data["dates"]
    with tempfile.TemporaryDirectory(prefix=f"autofuse-cal-{domain_id}-{model_id}-") as temp:
        case = Path(temp)
        _copy_settings(case, model_id)
        _write_input_info(case / "settings" / "input_info.txt", 1.0)
        _write_forcing(case / "input" / f"{domain_id}_input.nc", case_data["forcing"], dates, 1.0)
        _write_elevation_bands(case / "input" / f"{domain_id}_elev_bands.nc", case_data["forcing"])
        _write_file_manager(case / "fm_catch.txt", case, domain_id, model_id, FORCING_START, SIMULATION_END, CALIBRATION_START, CALIBRATION_END, maxn, kstop, pcento)
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "FUSE_REFERENCE_LIB_DIR": str(Path(os.environ.get("FUSE_REFERENCE_LIB_DIR", "")).resolve()) if os.environ.get("FUSE_REFERENCE_LIB_DIR") else ""})
        library_dir = os.environ.get("FUSE_REFERENCE_LIB_DIR")
        if library_dir:
            env["LD_LIBRARY_PATH"] = os.pathsep.join([library_dir, env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
        calib = _run_mode(exe, case, domain_id, "calib_sce", env)
        (model_root / f"{domain_id}_{model_id}_calib_sce.log").write_text(calib["stdout_tail"] + "\nSTDERR\n" + calib["stderr_tail"])
        para_sce = case / "output" / f"{domain_id}_{model_id}_para_sce.nc"
        if calib["returncode"] != 0 or not para_sce.is_file():
            raise RuntimeError(f"Fortran calib_sce failed for {domain_id}/{model_id}: {calib['stderr_tail'][-1000:]}")
        run_best = _run_mode(exe, case, domain_id, "run_best", env)
        (model_root / f"{domain_id}_{model_id}_run_best.log").write_text(run_best["stdout_tail"] + "\nSTDERR\n" + run_best["stderr_tail"])
        para_best = case / "output" / f"{domain_id}_{model_id}_para_best.nc"
        if run_best["returncode"] != 0 or not para_best.is_file():
            raise RuntimeError(f"Fortran run_best failed for {domain_id}/{model_id}: {run_best['stderr_tail'][-1000:]}")
        params, diagnostics = _read_best(para_best, model_id)
        child_rss = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if child_rss > RSS_STOP_KB:
            raise RuntimeError(f"reference calibration RSS safety stop: {child_rss} KB")
        return {
            "basin_id": case_data["basin_id"], "hru_id": case_data["hru_id"], "model_id": model_id,
            "status": "passed", "parameter_vector": params, "diagnostics": diagnostics,
            "config": {"maxn": maxn, "kstop": kstop, "pcento": pcento, "metric": "KGECOMP", "transform": 1.0, "initial_fraction": 0.25, "seed": None, "seed_note": "pinned driver does not expose an external seed argument"},
            "source_label": source_label, "executable_sha256": sha256_file(exe),
            "forcing_protocol": {"forcing_start": FORCING_START.isoformat(), "simulation_end": SIMULATION_END.isoformat(), "calibration_start": CALIBRATION_START.isoformat(), "calibration_end": CALIBRATION_END.isoformat(), "forcing_kind": "CAMELS bundle with basin-mean Daymet-equivalent P/Tmean/Oudin-PET", "elevation_band_count": int(len(case_data["forcing"]["area_frac"]))},
            "runtime": {"calib_sce_seconds": calib["elapsed_seconds"], "run_best_seconds": run_best["elapsed_seconds"], "child_ru_maxrss_kb": child_rss},
            "parameter_output_format": "Fortran *_para_best.nc; active structure parameter_names extracted by model ID",
        }


def _load_cases(manifest: Mapping[str, Any], data_root: Path, data_path: Path, gage_path: Path) -> dict[str, Any]:
    forcing, target, attributes, ids = _load_bundle(data_path, gage_path)
    catalogue = json.loads(MANIFEST_PATH.read_text())
    return {row["basin_id"]: catchment_case(row["basin_id"], data_root, forcing, target, attributes, ids, catalogue) for row in manifest["catchments"]}


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    exe = Path(args.executable).resolve()
    if not exe.is_file():
        raise FileNotFoundError(exe)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.catchment_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = _load_cases(manifest, Path(args.data_root), Path(args.data_path), Path(args.gage_path))
    if args.structure_ids:
        model_ids = [int(value) for value in json.loads(Path(args.structure_ids).read_text())]
    else:
        model_ids = [int(row["ID"]) for row in json.loads((ROOT / "dfuse/specs/structures_78.json").read_text())["rows"]]
    exe_sha = sha256_file(exe)
    data_path = Path(args.data_path).resolve()
    gage_path = Path(args.gage_path).resolve()
    metadata = {
        "schema_version": "reference-calibration-12x78-v1",
        "configuration": {"maxn": args.maxn, "kstop": args.kstop, "pcento": args.pcento, "paper_budget_maxn": 10000, "metric": "KGECOMP", "transform": 1.0, "threads": 1},
        "executable": str(exe), "executable_sha256": exe_sha,
        "catchment_manifest": str(manifest_path), "catchment_manifest_sha256": sha256_file(manifest_path),
        "data_path": str(data_path), "data_sha256": sha256_file(data_path), "gage_path": str(gage_path), "gage_sha256": sha256_file(gage_path),
        "case_count_expected": len(cases) * len(model_ids),
    }
    prior_path = output_root / "reference_calibration_12x78.partial.json"
    prior = json.loads(prior_path.read_text()) if prior_path.is_file() and not args.restart else {}
    anomaly_row = None
    if args.early_stop_row:
        anomaly_payload = json.loads(Path(args.early_stop_row).read_text())
        anomaly_row = anomaly_payload.get("reference_row")
        if not isinstance(anomaly_row, dict) or anomaly_row.get("status") not in ("early_stopped", "passed", "retry"):
            raise ValueError("early-stop artifact does not contain an accepted reference row")
        if (int(anomaly_row["hru_id"]), int(anomaly_row["model_id"])) != (12010000, 206):
            raise ValueError("early-stop row is not USA_12010000/model206")
        if anomaly_row.get("executable_sha256") != exe_sha or anomaly_row.get("config", {}).get("maxn") != args.maxn or anomaly_row.get("config", {}).get("kstop") != args.kstop or anomaly_row.get("config", {}).get("pcento") != args.pcento:
            raise AssertionError("early-stop row does not match current executable/configuration")
    if prior:
        for field in ("executable_sha256", "catchment_manifest_sha256", "data_sha256", "gage_sha256", "case_count_expected"):
            if prior.get(field) != metadata[field]:
                raise AssertionError(f"partial calibration {field} does not match current inputs")
        if prior.get("configuration") != metadata["configuration"]:
            raise AssertionError("partial calibration configuration does not match current bounded budget")
    accepted_statuses = ("passed", "early_stopped", "retry")
    completed = {(int(row["hru_id"]), int(row["model_id"])): row for row in prior.get("case_results", []) if row.get("status") in accepted_statuses and row.get("executable_sha256") == exe_sha and row.get("config", {}).get("maxn") == args.maxn and row.get("config", {}).get("kstop") == args.kstop and row.get("config", {}).get("pcento") == args.pcento}
    if anomaly_row is not None:
        anomaly_key = (int(anomaly_row["hru_id"]), int(anomaly_row["model_id"]))
        if anomaly_key in completed and completed[anomaly_key].get("parameter_vector") != anomaly_row.get("parameter_vector"):
            raise AssertionError("existing model206 row differs from supplied anomaly reference row")
        completed[anomaly_key] = anomaly_row
    rows = list(completed.values())
    started = prior.get("started_unix", time.time())
    prior_failure = prior.get("failure")
    partial = {**metadata, "status": "partial", "started_unix": started, "case_count_completed": len(rows), "prior_failure": prior_failure, "case_results": sorted(rows, key=lambda row: (row["hru_id"], row["model_id"]))}
    prior_path.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
    for basin_id in sorted(cases, key=lambda key: cases[key]["hru_id"]):
        for model_id in model_ids:
            key = (cases[basin_id]["hru_id"], model_id)
            if key in completed:
                continue
            try:
                row = calibrate_one(exe, cases[basin_id], model_id, output_root, args.maxn, args.kstop, args.pcento, "bounded local original-Fortran SCE")
            except Exception as exc:
                failure = {"basin_id": basin_id, "hru_id": cases[basin_id]["hru_id"], "model_id": model_id, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]}
                partial = {**metadata, "status": "blocked on calibration case", "started_unix": started, "case_count_completed": len(rows), "prior_failure": prior_failure, "failure": failure, "case_results": sorted(rows, key=lambda value: (value["hru_id"], value["model_id"]))}
                prior_path.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
                raise
            completed[key] = row
            rows = list(completed.values())
            partial = {**metadata, "status": "partial", "started_unix": started, "case_count_completed": len(rows), "case_results": sorted(rows, key=lambda value: (value["hru_id"], value["model_id"]))}
            prior_path.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
    final = {**partial, "status": "complete" if len(rows) == metadata["case_count_expected"] else "partial", "finished_unix": time.time(), "parameter_source": "bounded local calibration by original Fortran FUSE calib_sce/run_best", "calibrated_case_count": len(rows), "calibrated_catchment_count": len({row["hru_id"] for row in rows}), "calibrated_structure_count": len({row["model_id"] for row in rows}), "case_status_counts": {status: sum(1 for row in rows if row["status"] == status) for status in sorted({row["status"] for row in rows})}, "anomaly_artifact": str(ROOT / "project/autofuse/docs/reference_calibration_anomaly_12010000_206.json"), "parameter_results_sha256": _canonical_hash(sorted(rows, key=lambda row: (row["hru_id"], row["model_id"]))) }
    (output_root / "reference_calibration_12x78.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    data_root, data_path, gage_path = Path(args.data_root), Path(args.data_path), Path(args.gage_path)
    forcing, target, attributes, ids = _load_bundle(data_path, gage_path)
    catalogue = json.loads(MANIFEST_PATH.read_text())
    # First complete basin in the authoritative 544 manifest, independent of solver results.
    case = None
    for row in catalogue["basins"]:
        try:
            case = catchment_case(row["basin_id"], data_root, forcing, target, attributes, ids, catalogue)
            break
        except (KeyError, ValueError):
            continue
    if case is None:
        raise RuntimeError("no complete CAMELS catchment available for calibration smoke")
    rows = []
    for model_id in MOTHER_MODELS:
        rows.append(calibrate_one(Path(args.executable).resolve(), case, model_id, Path(args.output_root).resolve(), args.maxn, args.kstop, args.pcento, "A2 one-catchment calibration smoke"))
    result = {"schema_version": "fortran-local-calibration-smoke-v1", "status": "passed", "catchment": case["basin_id"], "hru_id": case["hru_id"], "models": list(MOTHER_MODELS), "configuration": {"maxn": args.maxn, "kstop": args.kstop, "pcento": args.pcento, "metric": "KGECOMP", "transform": 1.0, "paper_budget_maxn": 10000, "threads": 1}, "executable": str(Path(args.executable).resolve()), "executable_sha256": sha256_file(Path(args.executable).resolve()), "results": rows, "parameter_vectors_are_for_smoke_only": True}
    path = Path(args.output_root) / "fortran_local_calibration_smoke.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "freeze-catchments", "calibrate"), required=True)
    parser.add_argument("--executable", default=os.environ.get("FUSE_REFERENCE_EXE", "/tmp/autofuse-reference-toolchain/repro-build/bin/fuse.exe"))
    parser.add_argument("--data-root", default=os.environ.get("CAMELS_ROOT", "/mnt/g/Dataset/CAMELS_US"))
    parser.add_argument("--data-path", default=str(BUNDLE_PATH))
    parser.add_argument("--gage-path", default=str(GAGE_PATH))
    parser.add_argument("--output-root", default=str(ROOT / "project/autofuse/docs"))
    parser.add_argument("--catchment-manifest", default=str(ROOT / "project/autofuse/docs/landscape_12catchment_manifest.json"))
    parser.add_argument("--structure-ids", default=None)
    parser.add_argument("--maxn", type=int, default=20)
    parser.add_argument("--kstop", type=int, default=3)
    parser.add_argument("--pcento", type=float, default=0.001)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--early-stop-row", default=None, help="merge one auditable model-206 reference row from anomaly diagnosis before resuming")
    args = parser.parse_args()
    if args.mode == "smoke":
        result = run_smoke(args)
    elif args.mode == "freeze-catchments":
        result = select_catchments(Path(args.data_root), Path(args.data_path), Path(args.gage_path))
        result["manifest_sha256"] = _canonical_hash({key: value for key, value in result.items() if key != "manifest_sha256"})
        Path(args.output_root, "landscape_12catchment_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        result = run_calibration(args)
    print(json.dumps({"status": result.get("status"), "catchment": result.get("catchment"), "completed": result.get("case_count_completed"), "expected": result.get("case_count_expected"), "manifest_sha256": result.get("manifest_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
