"""Bounded validation of the diagnostic storage-block explicit prototype.

This module is reached only after the frozen S4 subset fails.  It reuses the
same four catchments, twelve structures, fixed calibration vectors, forcing,
and completed S3/S4 summaries; only the block solver and fresh reference
forward are executed here so raw block-vs-reference fidelity can be measured.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import spearmanr

from dfuse import reset_compile_diagnostics, runtime_compile_diagnostics, simulate_storage_block
from dfuse.kernel import SEQUENTIAL_ORDERS
from project.autofuse.landscape_gate import _case_inputs, _canonical_hash, _date_range, _metrics, _flux_activation, _eval_slice, _cal_slice, sha256_file
from project.autofuse.reference_oracle import run_reference
from project.autofuse.s4_targeted_diagnostic import (
    DEFAULT_CALIBRATION,
    DEFAULT_CACHE,
    DEFAULT_INPUT_ROOT,
    DEFAULT_SELECTION_CATCHMENTS,
    DEFAULT_SELECTION_STRUCTURES,
    DEFAULT_S3,
    DEFAULT_EXE,
    REFERENCE_TIMEOUT_SECONDS,
    RSS_STOP_KB,
    _case_run,
    _fidelity,
    _load_frozen,
    _load_inputs,
    _load_s3,
    _set_resource_limits,
    _solver_summary,
    _summary,
    _sync,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
DEFAULT_S4 = DOCS / "s4_targeted_diagnostic.json"
DEFAULT_OUTPUT = DOCS / "storage_block_sequential_diagnostic.json"
DEFAULT_PARTIAL = DOCS / "storage_block_sequential_diagnostic.partial.json"
BLOCK_MODE = "storage_block"
BLOCK_ORDER_NAME = "S4"
BLOCK_ORDER = SEQUENTIAL_ORDERS[BLOCK_ORDER_NAME]
MOTHER_MODELS = (2, 108, 178, 210)
TOLERANCE = 1.0e-12
GPU_RESERVED_FRACTION_STOP = 0.85


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _dates() -> list[date]:
    from project.autofuse.reference_calibration import FORCING_START, SIMULATION_END
    return _date_range(FORCING_START, SIMULATION_END)


def _load_s4(path: Path, catchment_path: Path, structure_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or len(payload.get("rows", [])) != 48:
        raise RuntimeError("complete 48-case S4 targeted diagnostic is required")
    if payload["selection"]["catchments_sha256"] != sha256_file(catchment_path) or payload["selection"]["structures_sha256"] != sha256_file(structure_path):
        raise RuntimeError("S4 result selection does not match frozen storage-block subset")
    return {(int(row["hru_id"]), int(row["model_id"])): row for row in payload["rows"]}


def _block_run(basin: Mapping[str, Any], structure: Mapping[str, Any], input_path: Path, calibration: Mapping[tuple[int, int], Mapping[str, Any]], s4_rows: Mapping[tuple[int, int], Mapping[str, Any]], executable: Path, dates: list[date], device: torch.device) -> dict[str, Any]:
    hru_id = int(basin["hru_id"])
    model_id = int(structure["model_id"])
    values, _ = _case_inputs(input_path)
    wanted = calibration[(hru_id, model_id)]
    params = wanted["parameter_vector"]
    prior = s4_rows[(hru_id, model_id)]
    if prior["calibrated_parameter_hash"] != _canonical_hash(params):
        raise RuntimeError(f"calibration theta mismatch for {hru_id}/{model_id}")
    forcing = {name: values[name] for name in ("ppt", "pet", "temp", "q_obs", "area_frac", "mean_elev")}
    reference_started = time.perf_counter()
    reference = run_reference(executable, model_id, forcing, params=params, initial_fraction=0.25, dates=dates, dt_days=1.0, timeout_seconds=REFERENCE_TIMEOUT_SECONDS)
    reference_elapsed = time.perf_counter() - reference_started
    tensor_forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1), dtype=torch.float64, device=device)
    block_started = time.perf_counter()
    block = simulate_storage_block(model_id, tensor_forcing, params, initial_fraction=0.25, dates=dates, dt_days=1.0, n_substeps=1, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    torch.cuda.synchronize(device)
    block_elapsed = time.perf_counter() - block_started
    eval_slice = _eval_slice()
    reference_summary = _solver_summary(reference, None, model_id, params, values, eval_slice, reference_elapsed, "fortran")
    block_summary = _solver_summary(reference, block, model_id, params, values, eval_slice, block_elapsed, "block")
    block_delta = {
        "kgecomp_minus_fortran": float(block_summary["q_metrics"]["kgecomp"] - reference_summary["q_metrics"]["kgecomp"]),
        "kge_inv_q_minus_fortran": float(block_summary["q_metrics"]["kge_inv_q"] - reference_summary["q_metrics"]["kge_inv_q"]),
    }
    return {
        "basin_id": basin["basin_id"],
        "hru_id": hru_id,
        "model_id": model_id,
        "topology": prior["topology"],
        "calibrated_parameter_hash": _canonical_hash(params),
        "calibration_status": wanted["status"],
        "fortran": reference_summary,
        "s3": prior["s3"],
        "s4": prior["s4"],
        "block": block_summary,
        "fidelity_s3_vs_fortran": prior["fidelity_s3_vs_fortran"],
        "fidelity_s4_vs_fortran": prior["fidelity_s4_vs_fortran"],
        "fidelity_block_vs_fortran": _fidelity(reference, block, model_id, params, eval_slice),
        "delta_kgecomp": {
            "s3_minus_fortran": float(prior["delta_kgecomp"]["s3_minus_fortran"]),
            "s4_minus_fortran": float(prior["delta_kgecomp"]["s4_minus_fortran"]),
            "block_minus_fortran": block_delta["kgecomp_minus_fortran"],
        },
        "delta_kge_inv_q": {
            "s3_minus_fortran": float(prior["delta_kge_inv_q"]["s3_minus_fortran"]),
            "s4_minus_fortran": float(prior["delta_kge_inv_q"]["s4_minus_fortran"]),
            "block_minus_fortran": block_delta["kge_inv_q_minus_fortran"],
        },
        "s3_s4_reuse": {
            "source": str(DEFAULT_S4.resolve()),
            "s3_ranking_metrics_reused": True,
            "s4_ranking_metrics_reused": True,
            "s3_raw_arrays_rerun": False,
            "s4_raw_arrays_rerun": False,
        },
        "reference_process": {"stdout_tail": reference.metadata.get("stdout_tail", ""), "stderr_tail": reference.metadata.get("stderr_tail", "")},
        "resource_after_case": {
            "host_rss_kb": int(_rss_kb()),
            "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "fortran_child_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
            "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }


def _rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _generic_ranking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_basin: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_basin.setdefault(row["basin_id"], []).append(row)
    per = []
    for basin_id, basin_rows in sorted(by_basin.items()):
        def rank(solver: str) -> list[int]:
            return [r["model_id"] for r in sorted(basin_rows, key=lambda r: (-r[solver]["q_metrics"]["kgecomp"], r["model_id"]))]
        ranks = {solver: rank(solver) for solver in ("fortran", "s3", "s4", "block")}
        f = ranks["fortran"]
        f_scores = np.asarray([next(r for r in basin_rows if r["model_id"] == model)["fortran"]["q_metrics"]["kgecomp"] for model in f])
        result = {"basin_id": basin_id, "count": len(basin_rows), "fortran_rank": f, "best_model_fortran": f[0]}
        for solver in ("s3", "s4", "block"):
            scores = np.asarray([next(r for r in basin_rows if r["model_id"] == model)[solver]["q_metrics"]["kgecomp"] for model in f])
            solver_rank = ranks[solver]
            result[f"{solver}_rank"] = solver_rank
            result[f"spearman_{solver}"] = float(spearmanr(f_scores, scores).statistic)
            result[f"best_model_{solver}"] = solver_rank[0]
            result[f"best_agreement_{solver}"] = solver_rank[0] == f[0]
            for k in (3, 5):
                left, right = set(f[:k]), set(solver_rank[:k])
                result[f"top{k}_{solver}"] = {"intersection": len(left & right), "jaccard": float(len(left & right) / len(left | right))}
        per.append(result)
    summary: dict[str, Any] = {}
    for solver in ("s3", "s4", "block"):
        summary[f"spearman_{solver}"] = _summary([row[f"spearman_{solver}"] for row in per])
        summary[f"top3_intersection_{solver}"] = _summary([row[f"top3_{solver}"]["intersection"] for row in per])
        summary[f"top5_intersection_{solver}"] = _summary([row[f"top5_{solver}"]["intersection"] for row in per])
        summary[f"best_agreement_{solver}_count"] = int(sum(row[f"best_agreement_{solver}"] for row in per))
    summary["block_spearman_improved_vs_s4_count"] = int(sum(row["spearman_block"] > row["spearman_s4"] for row in per))
    summary["block_top3_improved_vs_s4_count"] = int(sum(row["top3_block"]["intersection"] > row["top3_s4"]["intersection"] for row in per))
    summary["block_top5_improved_vs_s4_count"] = int(sum(row["top5_block"]["intersection"] > row["top5_s4"]["intersection"] for row in per))
    return {"per_catchment": per, "summary": summary}


def _activation4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for solver in ("fortran", "s3", "s4", "block"):
        result[solver] = {}
        for process in ("ET", "qsurf", "qperc", "qintf", "qbase"):
            values = [float(row[solver]["activation"][process]) for row in rows]
            result[solver][process] = {**_summary(values), "near_zero_count": int(sum(value <= 1.0e-6 for value in values)), "count": len(values)}
    result["qperc_gap_to_fortran"] = {solver: _summary([abs(row[solver]["activation"]["qperc"] - row["fortran"]["activation"]["qperc"]) for row in rows]) for solver in ("s3", "s4", "block")}
    return result


def _topology4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for option in ("QPERC", "ARCH1", "ARCH2", "QSURF"):
        groups: dict[str, dict[str, list[float]]] = {}
        for row in rows:
            key = row["topology"][option]
            groups.setdefault(key, {solver: [] for solver in ("s3", "s4", "block")})
            for solver in ("s3", "s4", "block"):
                groups[key][solver].append(row["delta_kgecomp"][f"{solver}_minus_fortran" if solver != "block" else "block_minus_fortran"])
        result[option] = {}
        for key, values_by_solver in sorted(groups.items()):
            result[option][key] = {}
            for solver, values in values_by_solver.items():
                result[option][key][solver] = {**_summary(values), "iqr": [float(np.quantile(values, .25)), float(np.quantile(values, .75))], "negative_fraction": float(np.mean(np.asarray(values) < 0.0)), "positive_fraction": float(np.mean(np.asarray(values) > 0.0))}
            result[option][key]["absolute_median_reduction_block_vs_s4"] = 1.0 - abs(result[option][key]["block"]["median"]) / max(abs(result[option][key]["s4"]["median"]), 1.0e-15)
    return result


def _low4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {solver: _summary([row["delta_kge_inv_q"][f"{solver}_minus_fortran" if solver != "block" else "block_minus_fortran"] for row in rows]) for solver in ("s3", "s4", "block")}


def _fidelity4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = ("q_max_abs", "q_rmse", "state_max_abs", "state_rmse", "flux_max_abs", "water_balance_max_abs")
    for solver in ("s3", "s4", "block"):
        result[solver] = {key: _summary([row[f"fidelity_{solver}_vs_fortran"][key] for row in rows]) for key in keys}
        result[solver]["finite_count"] = int(sum(row[solver]["finite_capacity"]["finite"] for row in rows))
        result[solver]["capacity_clean_count"] = int(sum(row[solver]["finite_capacity"]["max_capacity_violation"] <= TOLERANCE for row in rows))
    return result


def _timed(fn: Any, device: torch.device, repeats: int = 2) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        fn()
        _sync(device)
        values.append(time.perf_counter() - started)
    return {"min_seconds": min(values), "median_seconds": statistics.median(values), "max_seconds": max(values)}


def _block_gradient(model_id: int, forcing: torch.Tensor, params: Mapping[str, float], dates: list[date], compiled: bool) -> torch.Tensor:
    tensors = {name: torch.tensor(value, dtype=torch.float64, device=forcing.device, requires_grad=True) for name, value in params.items()}
    result = simulate_storage_block(model_id, forcing, tensors, initial_fraction=0.25, dates=dates, dt_days=1.0, n_substeps=1, compile_step=compiled, compile_backend="inductor", compile_fullgraph=True)
    return result.q.sum()


def _benchmark(first_basin: Mapping[str, Any], input_path: Path, calibration: Mapping[tuple[int, int], Mapping[str, Any]], device: torch.device, dates: list[date]) -> dict[str, Any]:
    values, _ = _case_inputs(input_path)
    forcing = torch.as_tensor(np.stack((values["ppt"], values["pet"], values["temp"]), axis=1)[:64], dtype=torch.float64, device=device)
    rows = []
    for model_id in MOTHER_MODELS:
        params = calibration[(int(first_basin["hru_id"]), model_id)]["parameter_vector"]
        simulate_storage_block(model_id, forcing, params, initial_fraction=0.25, dates=dates[:64], dt_days=1.0, n_substeps=1, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
        eager_f = _timed(lambda: simulate_storage_block(model_id, forcing, params, initial_fraction=0.25, dates=dates[:64], dt_days=1.0, n_substeps=1, compile_step=False), device)
        compiled_f = _timed(lambda: simulate_storage_block(model_id, forcing, params, initial_fraction=0.25, dates=dates[:64], dt_days=1.0, n_substeps=1, compile_step=True, compile_backend="inductor", compile_fullgraph=True), device)
        eager_b = _timed(lambda: _block_gradient(model_id, forcing, params, dates[:64], False).backward(), device, 1)
        compiled_b = _timed(lambda: _block_gradient(model_id, forcing, params, dates[:64], True).backward(), device, 1)
        rows.append({"model_id": model_id, "steps": 64, "forward": {"eager": eager_f, "compiled": compiled_f, "compiled_speedup": eager_f["median_seconds"] / compiled_f["median_seconds"]}, "backward": {"eager": eager_b, "compiled": compiled_b, "compiled_speedup": eager_b["median_seconds"] / compiled_b["median_seconds"]}})
    return {"catchment": first_basin["basin_id"], "hru_id": int(first_basin["hru_id"]), "rows": rows}


def _decision(rows: list[dict[str, Any]], activation: Mapping[str, Any], topology: Mapping[str, Any], ranking: Mapping[str, Any]) -> dict[str, Any]:
    target_groups = []
    for option in ("QPERC", "ARCH1", "ARCH2", "QSURF"):
        for summary in topology[option].values():
            target_groups.append(summary)
    block_topology_improved = sum(summary["absolute_median_reduction_block_vs_s4"] >= 0.25 for summary in target_groups) >= max(1, math.ceil(0.6 * len(target_groups)))
    gap_s4 = activation["qperc_gap_to_fortran"]["s4"]["median"]
    gap_block = activation["qperc_gap_to_fortran"]["block"]["median"]
    block_qperc_recovered = gap_block <= 0.75 * gap_s4
    rank = ranking["summary"]
    block_ranking_improved = rank["block_spearman_improved_vs_s4_count"] >= 3 and (rank["block_top3_improved_vs_s4_count"] + rank["block_top5_improved_vs_s4_count"]) >= 2
    block_new_bias = any(summary["block"]["median"] < -0.01 and summary["block"]["negative_fraction"] >= 0.75 and summary["absolute_median_reduction_block_vs_s4"] < 0.0 for summary in target_groups)
    criteria = {"qperc_activation_gap_reduced_25pct_vs_s4": block_qperc_recovered, "topology_bias_reduced_25pct_in_60pct_groups_vs_s4": block_topology_improved, "ranking_improved_vs_s4": block_ranking_improved, "no_new_systematic_target_bias": not block_new_bias}
    successful = all(criteria.values())
    recommendation = "S4 insufficient — storage-block sequential explicit preferred" if successful else "explicit sequential family still unresolved"
    return {"criteria": criteria, "successful": successful, "recommendation": recommendation, "thresholds": {"qperc": "at least 25% median absolute activation-gap reduction versus S4", "topology": "at least 60% of all targeted topology groups reduce absolute median bias by at least 25% versus S4", "ranking": "block Spearman improves in at least 3/4 catchments and at least two Top-k improvements", "new_bias": "no targeted group with negative median <= -0.01, negative fraction >= 0.75, and worsening versus S4"}}


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    _set_resource_limits()
    if not torch.cuda.is_available():
        raise RuntimeError("storage-block diagnostic requires CUDA; refusing CPU fallback")
    device = torch.device("cuda")
    cache_dir = Path(args.cache_dir).resolve(); cache_dir.mkdir(parents=True, exist_ok=True); os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    catchments, structures = _load_frozen(Path(args.selection_catchments), Path(args.selection_structures))
    s4_rows = _load_s4(Path(args.s4_artifact), Path(args.selection_catchments), Path(args.selection_structures))
    s3_rows = _load_s3(Path(args.s3_artifact))
    calibration_payload = json.loads(Path(args.calibration).read_text())
    calibration = {(int(row["hru_id"]), int(row["model_id"])): row for row in calibration_payload["case_results"]}
    inputs = _load_inputs(Path(args.input_root)); dates = _dates(); reset_compile_diagnostics(); rows: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []; initial_rss = _rss_kb(); stopped = False
    for basin in catchments:
        for structure in structures:
            try:
                row = _block_run(basin, structure, inputs[int(basin["hru_id"])], calibration, s4_rows, Path(args.executable).resolve(), dates, device)
                if (int(row["hru_id"]), int(row["model_id"])) not in s3_rows:
                    raise RuntimeError("completed S3 row missing for selected block case")
                rows.append(row)
            except Exception as exc:
                failures.append({"basin_id": basin["basin_id"], "hru_id": int(basin["hru_id"]), "model_id": int(structure["model_id"]), "error": f"{type(exc).__name__}: {str(exc)[:2000]}"}); stopped = True
            _write_json(Path(args.partial), {"schema_version": "storage-block-sequential-diagnostic-v1", "status": "partial", "rows": rows, "failures": failures, "selection_catchments": str(Path(args.selection_catchments).resolve()), "selection_structures": str(Path(args.selection_structures).resolve()), "resource": {"initial_host_rss_kb": initial_rss, "host_rss_kb": _rss_kb(), "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device))}})
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
            total_gpu = int(torch.cuda.get_device_properties(device).total_memory)
            if _rss_kb() >= RSS_STOP_KB or int(torch.cuda.memory_reserved(device)) >= int(total_gpu * GPU_RESERVED_FRACTION_STOP):
                failures.append({"kind": "resource_safety_stop", "host_rss_kb": _rss_kb(), "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device))}); stopped = True
            if stopped: break
        if stopped: break
    if len(rows) != 48 or failures:
        payload = {"schema_version": "storage-block-sequential-diagnostic-v1", "status": "stopped", "rows": rows, "failures": failures, "scope": {"calibration_rerun": False, "s3_full_gate_rerun": False, "s4_full_gate_rerun": False, "formal_training_started": False, "sce_started": False, "dpl_started": False, "scientific_equations_modified": False}}
        _write_json(Path(args.output), payload); return payload
    activation = _activation4(rows); topology = _topology4(rows); ranking = _generic_ranking(rows); low = _low4(rows); fidelity = _fidelity4(rows); benchmark = _benchmark(catchments[0], inputs[int(catchments[0]["hru_id"])], calibration, device, dates); compile_diag = runtime_compile_diagnostics(); block_records = [record for record in compile_diag["records"].values() if record.get("graph_signature", {}).get("execution_mode") == BLOCK_MODE]
    compile_audit = {"execution_mode": BLOCK_MODE, "record_count": len(block_records), "compile_attempts": sum(int(r.get("compile_attempts", 0)) for r in block_records), "compile_successes": sum(int(r.get("compile_successes", 0)) for r in block_records), "fallbacks": sum(int(r.get("fallbacks", 0)) for r in block_records), "graph_breaks": sum(int(r.get("graph_breaks", 0)) for r in block_records), "recompilations": sum(int(r.get("recompilations", 0)) for r in block_records), "autograd_recompilations": sum(int(r.get("autograd_recompilations", 0)) for r in block_records), "records": block_records}
    decision = _decision(rows, activation, topology, ranking)
    payload = {"schema_version": "storage-block-sequential-diagnostic-v1", "status": "complete", "trigger": {"s4_artifact": str(Path(args.s4_artifact).resolve()), "s4_artifact_sha256": sha256_file(Path(args.s4_artifact)), "reason": "S4 did not recover qperc activation, topology bias, or ranking on the frozen 4x12 subset"}, "selection": {"catchments_artifact": str(Path(args.selection_catchments).resolve()), "catchments_sha256": sha256_file(Path(args.selection_catchments)), "structures_artifact": str(Path(args.selection_structures).resolve()), "structures_sha256": sha256_file(Path(args.selection_structures)), "catchments": catchments, "structures": structures}, "dependency_graph_audit": {"source": "dfuse/spec.py topology and dfuse/runtime.py _fixed_raw_process/_fixed_delta_with_theta dependency contract", "upper_block_outgoing": ["EVAP_1A", "EVAP_1B", "EVAP_1", "QPERC_12", "QINTF_1", "OFLOW_1", "QSURF"], "lower_block_outgoing": ["EVAP_2", "QBASE_2A", "QBASE_2B", "QBASE_2", "OFLOW_2A", "OFLOW_2B", "OFLOW_2"], "incoming_external_first": ["EFF_PPT", "snow_effective_precipitation"], "block_rules": ["recharge is committed first", "all outgoing upper/lower raw fluxes are evaluated from one post-recharge snapshot", "upper outgoing deltas are committed as one explicit update", "lower outgoing deltas are committed as one explicit update after the upper block", "percolation lower transfer is incoming while its source QPERC is scaled with the upper block", "no implicit solve, Newton, Jacobian, IFT, or dynamic order branch"]}, "protocol": {"same_forcing_qobs_evaluation": True, "same_fixed_theta": True, "initial_fraction": 0.25, "s3_s4_metrics_reused": True, "fortran_fresh_reference_for_block_fidelity": True, "s3_full_gate_rerun": False, "s4_full_gate_rerun": False, "calibration_rerun": False, "block_order": list(BLOCK_ORDER), "execution_mode": BLOCK_MODE, "compile_backend": "inductor", "compile_fullgraph": True}, "rows": rows, "activation": activation, "topology_delta_kgecomp": topology, "ranking": ranking, "low_flow_delta_kge_inv_q": low, "fidelity": fidelity, "runtime_benchmark": benchmark, "compile_audit": compile_audit, "compile_clean": compile_audit["record_count"] >= len(structures) and compile_audit["fallbacks"] == compile_audit["graph_breaks"] == compile_audit["recompilations"] == compile_audit["autograd_recompilations"] == 0, "resource_summary": {"host_rss_initial_kb": initial_rss, "host_rss_end_kb": _rss_kb(), "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "fortran_child_rss_peak_kb": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss), "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)), "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)), "gpu_peak_allocated_bytes": max((row["resource_after_case"]["gpu_peak_allocated_bytes"] for row in rows), default=0), "gpu_peak_reserved_bytes": max((row["resource_after_case"]["gpu_peak_reserved_bytes"] for row in rows), default=0), "cache_dir": str(cache_dir)}, "decision": {**decision, "compile_clean": compile_audit["record_count"] >= len(structures) and compile_audit["fallbacks"] == compile_audit["graph_breaks"] == compile_audit["recompilations"] == compile_audit["autograd_recompilations"] == 0}, "scope": {"calibration_rerun": False, "s3_full_gate_rerun": False, "s4_full_gate_rerun": False, "formal_training_started": False, "sce_started": False, "dpl_started": False, "scientific_equations_modified": False, "fortran_reference_modified": False}}
    _write_json(Path(args.output), payload); return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-catchments", default=str(DEFAULT_SELECTION_CATCHMENTS))
    parser.add_argument("--selection-structures", default=str(DEFAULT_SELECTION_STRUCTURES))
    parser.add_argument("--s4-artifact", default=str(DEFAULT_S4))
    parser.add_argument("--s3-artifact", default=str(DEFAULT_S3))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--executable", default=os.environ.get("FUSE_REFERENCE_EXE", DEFAULT_EXE))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--partial", default=str(DEFAULT_PARTIAL))
    args = parser.parse_args(); payload = run_diagnostic(args)
    print(json.dumps({"status": payload["status"], "rows": len(payload.get("rows", [])), "failures": len(payload.get("failures", [])), "recommendation": payload.get("decision", {}).get("recommendation")}, sort_keys=True))


if __name__ == "__main__":
    main()
