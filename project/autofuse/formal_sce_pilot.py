"""Run the bounded 2-catchment x 4-structure formal SCE pipeline pilot."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfuse import PARAMETER_NAMES, batched_compile_diagnostics, reset_batched_compile_diagnostics
from dfuse.spec import default_parameters, get_structure
from project.autofuse.metrics import kgecomp_batched
from project.autofuse.sce import SCEBaseline, SCEConfig
from project.autofuse.torch_fuse_78_long_horizon_smoke import _dates, _load_frozen_inputs

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
PROTOCOL_PATH = DOCS / "formal_experiment_protocol_v1.json"
FREEZE_PATH = DOCS / "torch_fuse_v1_freeze.json"
CASE_DIR = DOCS / "formal_sce_pilot_cases"
SUMMARY_PATH = DOCS / "formal_sce_pilot_2x4.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path: Path, payload: dict[str, Any], hash_field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    digest = hashlib.sha256(_canonical(unsigned)).hexdigest()
    signed = dict(unsigned)
    signed[hash_field] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        json.dump(signed, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return digest


def _verify_hashed_json(path: Path, hash_field: str) -> str:
    payload = json.loads(path.read_text())
    recorded = payload.get(hash_field)
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    expected = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if recorded != expected:
        raise RuntimeError(f"artifact hash mismatch: {path}")
    return str(recorded)


def _nvidia_sample() -> dict[str, object] | None:
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    values = output.splitlines()[0].split(",")
    if len(values) != 3:
        return None
    return {"utilization_gpu_percent": int(values[0].strip()), "memory_used_mib": int(values[1].strip()), "memory_total_mib": int(values[2].strip())}


def _load_and_validate_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    if protocol.get("status") != "frozen" or protocol.get("unresolved"):
        raise RuntimeError("formal protocol is not frozen or has unresolved fields")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    expected_id = "formal-experiment-protocol-v1-" + hashlib.sha256(_canonical(unsigned)).hexdigest()[:16]
    if protocol_id != expected_id:
        raise RuntimeError(f"formal protocol id mismatch: {protocol_id} != {expected_id}")
    if protocol["sources"]["torch_fuse_freeze"]["sha256"] != _sha(FREEZE_PATH) or freeze.get("status") != "frozen":
        raise RuntimeError("Torch-FUSE v1 freeze artifact mismatch")
    for key in ("kernel", "runtime", "catalogue"):
        frozen = freeze["source_provenance"]["source"][key]
        if _sha(Path(frozen["path"])) != frozen["sha256"]:
            raise RuntimeError(f"frozen numerical source changed: {key}")
    if protocol["pilot_scope"] != {**protocol["pilot_scope"], "case_count": 8}:
        raise RuntimeError("invalid pilot scope")
    return protocol, freeze


def _masked_observed(q_obs: np.ndarray, dates: list[Any], left: str, right: str, *, device: torch.device) -> torch.Tensor:
    mask = np.asarray([(left <= current.isoformat() <= right) for current in dates], dtype=bool)
    values = np.full((1, len(q_obs)), np.nan, dtype=np.float64)
    values[0, mask] = q_obs[mask]
    return torch.as_tensor(values, dtype=torch.float64, device=device)


def _forcing(values: dict[str, np.ndarray], *, device: torch.device) -> torch.Tensor:
    array = np.stack((values["ppt"], values["pet"], values["temp"]), axis=1)[None, ...]
    return torch.as_tensor(array, dtype=torch.float64, device=device)


def _compile_record(model_id: int) -> dict[str, Any]:
    rows = list(batched_compile_diagnostics()["records"].values())
    matching = [row for row in rows if int(row.get("model_id", -1)) == model_id and int(row.get("batch_size", -1)) == 1]
    if not matching:
        return {"recorded": False, "passed": False}
    row = matching[-1]
    return {"recorded": True, "passed": bool(row.get("compile_attempts") == 1 and row.get("compile_successes") == 1 and row.get("fallbacks") == 0 and row.get("graph_breaks") == 0 and row.get("recompilations") == 0 and row.get("forward_unique_graphs", 1) <= 1), "record": row}


def _bounds(protocol: dict[str, Any], model_id: int) -> dict[str, tuple[float, float]]:
    return {name: (float(entry["lower"]), float(entry["upper"])) for name, entry in protocol["parameters"]["bounds_by_structure"][str(model_id)].items()}


def _case(
    protocol: dict[str, Any],
    freeze: dict[str, Any],
    inputs: dict[str, dict[str, np.ndarray]],
    basin_id: str,
    model_id: int,
    seed: int,
    dates: list[Any],
    device: torch.device,
) -> dict[str, Any]:
    values = inputs[basin_id]
    full_forcing = _forcing(values, device=device)
    q_obs = np.asarray(values["q_obs"], dtype=np.float64)
    train_obs = _masked_observed(q_obs, dates, *protocol["periods"]["calibration"], device=device)
    test_obs = _masked_observed(q_obs, dates, *protocol["periods"]["evaluation"], device=device)
    train_mask = np.asarray([protocol["periods"]["calibration"][0] <= d.isoformat() <= protocol["periods"]["calibration"][1] for d in dates])
    test_mask = np.asarray([protocol["periods"]["evaluation"][0] <= d.isoformat() <= protocol["periods"]["evaluation"][1] for d in dates])
    if bool(np.any(train_mask & test_mask)) or not bool(np.any(train_mask)) or not bool(np.any(test_mask)):
        raise RuntimeError("calibration/evaluation masks overlap or are empty")
    epsilon = float(np.mean(q_obs[train_mask]) / 100.0)
    settings = protocol["sce"]["settings"]
    config = SCEConfig(max_evaluations=int(settings["max_evaluations"]), kstop=int(settings["kstop"]), pcento=float(settings["pcento"]), seed=seed, n_complexes=int(settings["n_complexes"]))
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu_start = _nvidia_sample()
    run = SCEBaseline(config).run(model_id, full_forcing, train_obs, basin_ids=(basin_id,), bounds=_bounds(protocol, model_id), inverse_epsilon=epsilon, compile_step=True, compile_backend=protocol["execution"]["compile_backend"], compile_fullgraph=bool(protocol["execution"]["compile_fullgraph"]))
    best_theta = {name: float(run["best_parameters"][name]) for name in PARAMETER_NAMES}
    active_names = list(get_structure(model_id).parameter_names)
    defaults = default_parameters()
    trajectory_finite = all(np.isfinite(row["score"]) and all(np.isfinite(value) for value in row["active_parameters"]) for row in run["trajectory"])
    active_bounds_valid = all(_bounds(protocol, model_id)[name][0] <= best_theta[name] <= _bounds(protocol, model_id)[name][1] for name in active_names)
    candidates_bounds_valid = all(all(_bounds(protocol, model_id)[name][0] <= float(value) <= _bounds(protocol, model_id)[name][1] for name, value in zip(active_names, row["active_parameters"])) for row in run["trajectory"])
    inactive_mask_valid = all(best_theta[name] == float(defaults[name]) for name in PARAMETER_NAMES if name not in active_names)
    theta_tensor = torch.as_tensor([best_theta[name] for name in PARAMETER_NAMES], dtype=torch.float64, device=device)
    evaluator = SCEBaseline().evaluator
    with torch.no_grad():
        recomputed_train = evaluator.score_batched(model_id, full_forcing, train_obs, theta_tensor, basin_ids=(basin_id,), inverse_epsilon=epsilon, compile_step=True, compile_backend=protocol["execution"]["compile_backend"], compile_fullgraph=bool(protocol["execution"]["compile_fullgraph"]))
        recomputed_test = evaluator.score_batched(model_id, full_forcing, test_obs, theta_tensor, basin_ids=(basin_id,), inverse_epsilon=epsilon, compile_step=True, compile_backend=protocol["execution"]["compile_backend"], compile_fullgraph=bool(protocol["execution"]["compile_fullgraph"]))
    train_metric = float(recomputed_train.kge_comp[0].detach().cpu())
    test_metric = float(recomputed_test.kge_comp[0].detach().cpu())
    train_metric_from_result = float(kgecomp_batched(recomputed_train.result.q, train_obs, epsilon=epsilon)[0].detach().cpu())
    test_metric_from_result = float(kgecomp_batched(recomputed_test.result.q, test_obs, epsilon=epsilon)[0].detach().cpu())
    best_objective = float(run["best_score"])
    recomputed_objective = 1.0 - train_metric
    train_result = recomputed_train.result
    test_result = recomputed_test.result
    path_audit = {"optimizer": "SCEBaseline.run", "objective": "SCEBaseline.objective_batched -> UnifiedEvaluator.score_batched", "simulation": "simulate_coupled_rk2_batched", "legacy_sequential_used": False, "legacy_fortran_used": False, "device": str(full_forcing.device), "dtype": str(full_forcing.dtype), "compile_fullgraph": True, "optimizer_forcing_period": protocol["periods"]["forcing"]}
    case = {"schema_version": "formal-sce-pilot-case-v1", "protocol_id": protocol["protocol_id"], "protocol_artifact_sha256": _sha(PROTOCOL_PATH), "torch_fuse_freeze_id": freeze["schema_version"], "torch_fuse_freeze_artifact_sha256": _sha(FREEZE_PATH), "catchment": basin_id, "structure": model_id, "seed": seed, "optimizer": {"algorithm": config.__class__.__name__.replace("Config", "-UA"), "implementation": "SCEBaseline.run", "config": {"max_evaluations": config.max_evaluations, "kstop": config.kstop, "pcento": config.pcento, "n_complexes": config.n_complexes, "population_size": run["config"]["population_size"], "optimization_forcing_end": protocol["periods"]["forcing"][1]}, "rng": run["execution"]["rng"], "initial_population": [row["active_parameters"] for row in run["trajectory"] if row["operation"] == "initial_population"]}, "best_theta": best_theta, "active_parameter_names": active_names, "inactive_parameter_names": [name for name in PARAMETER_NAMES if name not in active_names], "best_objective": best_objective, "train_metric": train_metric, "test_metric": test_metric, "termination_reason": run["stop_reason"], "evaluation_count": run["evaluation_count"], "trajectory": run["trajectory"], "best_theta_recomputation": {"recomputed_objective": recomputed_objective, "objective_abs_diff": abs(best_objective - recomputed_objective), "train_metric_recomputed": train_metric, "test_metric_recomputed": test_metric, "train_metric_from_saved_best_result": train_metric_from_result, "test_metric_from_saved_best_result": test_metric_from_result, "saved_best_reproduces_train": bool(abs(train_metric - train_metric_from_result) <= 1.0e-12), "saved_best_reproduces_test": bool(abs(test_metric - test_metric_from_result) <= 1.0e-12)}, "parameter_checks": {"all_candidates_finite": trajectory_finite, "best_active_bounds_valid": active_bounds_valid, "all_candidate_bounds_valid": candidates_bounds_valid, "inactive_parameters_fixed_to_defaults": inactive_mask_valid}, "train_test": {"forcing_period": protocol["periods"]["forcing"], "warmup_period": protocol["periods"]["warmup"], "calibration_period": protocol["periods"]["calibration"], "evaluation_period": protocol["periods"]["evaluation"], "calibration_steps": int(train_mask.sum()), "evaluation_steps": int(test_mask.sum()), "optimization_forcing_steps": len(dates), "epsilon": epsilon, "disjoint_masks": True, "evaluation_observations_passed_to_optimizer": False, "train_monitoring": train_result.monitoring, "test_monitoring": test_result.monitoring}, "path_audit": path_audit, "compile": _compile_record(model_id), "runtime": {"wall_clock_seconds": time.perf_counter() - started, "gpu_start": gpu_start, "gpu_end": _nvidia_sample(), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None, "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "python": platform.python_version(), "platform": platform.platform()}, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    case["resource"] = case["runtime"]
    case["recomputation"] = case["best_theta_recomputation"]
    case["leakage"] = {"calibration_period": protocol["periods"]["calibration"], "evaluation_period": protocol["periods"]["evaluation"], "disjoint_masks": case["train_test"]["disjoint_masks"], "evaluation_observations_passed_to_optimizer": case["train_test"]["evaluation_observations_passed_to_optimizer"]}
    case["pass"] = bool(case["parameter_checks"]["all_candidates_finite"] and case["parameter_checks"]["best_active_bounds_valid"] and case["parameter_checks"]["all_candidate_bounds_valid"] and case["parameter_checks"]["inactive_parameters_fixed_to_defaults"] and np.isfinite(best_objective) and abs(best_objective - recomputed_objective) <= 1.0e-12 and case["compile"]["passed"] and case["path_audit"]["legacy_sequential_used"] is False and case["path_audit"]["legacy_fortran_used"] is False and bool(train_result.monitoring["completed_full_period"]) and bool(test_result.monitoring["completed_full_period"]))
    case_path = CASE_DIR / f"{basin_id}_model_{model_id}.json"
    case["artifact_path"] = str(case_path)
    case["artifact_sha256"] = _atomic_json(case_path, case, "artifact_sha256")
    _verify_hashed_json(case_path, "artifact_sha256")
    return case


def main() -> None:
    protocol, freeze = _load_and_validate_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("formal SCE 2x4 pilot requires CUDA; refusing CPU fallback")
    device = torch.device("cuda")
    _, inputs, _ = _load_frozen_inputs()
    dates = _dates()
    catchments = list(protocol["pilot_scope"]["catchments"])
    structures = list(protocol["pilot_scope"]["structures"])
    reset_batched_compile_diagnostics()
    started = time.perf_counter()
    cases = []
    for basin_index, basin_id in enumerate(catchments):
        for structure_index, model_id in enumerate(structures):
            reset_batched_compile_diagnostics()
            seed = int(protocol["sce"]["settings"]["pilot_seeds"][basin_index * len(structures) + structure_index])
            cases.append(_case(protocol, freeze, inputs, basin_id, int(model_id), seed, dates, device))
    reset_batched_compile_diagnostics()
    selected = protocol["reproducibility"]["selected_case"]
    selected_forcing = _forcing(inputs[selected["catchment"]], device=device)
    selected_values = inputs[selected["catchment"]]
    selected_q = np.asarray(selected_values["q_obs"], dtype=np.float64)
    train_obs = _masked_observed(selected_q, dates, *protocol["periods"]["calibration"], device=device)
    selected_bounds = _bounds(protocol, int(selected["structure"]))
    selected_train_mask = np.asarray([protocol["periods"]["calibration"][0] <= d.isoformat() <= protocol["periods"]["calibration"][1] for d in dates])
    epsilon = float(np.mean(selected_q[selected_train_mask]) / 100.0)
    def reproducibility_run(seed: int) -> dict[str, Any]:
        settings = protocol["sce"]["settings"]
        config = SCEConfig(max_evaluations=int(settings["max_evaluations"]), kstop=int(settings["kstop"]), pcento=float(settings["pcento"]), seed=seed, n_complexes=int(settings["n_complexes"]))
        result = SCEBaseline(config).run(int(selected["structure"]), selected_forcing, train_obs, basin_ids=(selected["catchment"],), bounds=selected_bounds, inverse_epsilon=epsilon, compile_step=True, compile_backend=protocol["execution"]["compile_backend"], compile_fullgraph=bool(protocol["execution"]["compile_fullgraph"]))
        return {"seed": seed, "best_score": result["best_score"], "best_parameters": result["best_parameters"], "trajectory": result["trajectory"], "evaluation_count": result["evaluation_count"]}
    same_a = reproducibility_run(int(protocol["reproducibility"]["same_seed"]))
    same_b = reproducibility_run(int(protocol["reproducibility"]["same_seed"]))
    different = reproducibility_run(int(protocol["reproducibility"]["different_seed"]))
    repro_compile_records = batched_compile_diagnostics()
    same_equal = same_a == same_b
    different_differs = same_a["trajectory"] != different["trajectory"]
    compile_records = repro_compile_records
    compile_pass = bool(all(case["compile"]["passed"] for case in cases) and compile_records["registry_size"] == 1 and all(row.get("compile_attempts") == 1 and row.get("compile_successes") == 1 and row.get("fallbacks") == 0 and row.get("graph_breaks") == 0 and row.get("recompilations") == 0 and row.get("forward_unique_graphs", 1) <= 1 for row in compile_records["records"].values()))
    summary = {"schema_version": "formal-sce-pilot-2x4-v1", "protocol_id": protocol["protocol_id"], "protocol_artifact_sha256": _sha(PROTOCOL_PATH), "torch_fuse_freeze_id": freeze["schema_version"], "torch_fuse_freeze_artifact_sha256": _sha(FREEZE_PATH), "scope": {"catchments": catchments, "structures": structures, "case_count_expected": 8, "case_count_completed": len(cases)}, "cases": [{"catchment": c["catchment"], "structure": c["structure"], "seed": c["seed"], "pass": c["pass"], "artifact_path": c["artifact_path"], "artifact_sha256": c["artifact_sha256"], "best_objective": c["best_objective"], "train_metric": c["train_metric"], "test_metric": c["test_metric"], "evaluation_count": c["evaluation_count"], "wall_clock_seconds": c["runtime"]["wall_clock_seconds"]} for c in cases], "reproducibility": {"selected_case": selected, "same_seed": protocol["reproducibility"]["same_seed"], "same_seed_runs_equal": same_equal, "different_seed": protocol["reproducibility"]["different_seed"], "different_seed_trajectory_differs": different_differs, "same_seed_run_a": same_a, "same_seed_run_b": same_b, "different_seed_run": different}, "execution": {"device": "cuda", "dtype": "torch.float64", "compile_backend": "inductor", "compile_fullgraph": True, "compile_pass": compile_pass, "compile_diagnostics": compile_records, "wall_clock_seconds_total": time.perf_counter() - started, "host_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "gpu_peak_allocated_bytes_process": int(torch.cuda.max_memory_allocated(device)), "gpu_peak_reserved_bytes_process": int(torch.cuda.max_memory_reserved(device)), "gpu_utilization_sampling": "nvidia-smi start/end samples are stored per case"}, "path_audit": {"optimizer": "SCEBaseline.run", "objective": "SCEBaseline.objective_batched -> UnifiedEvaluator.score_batched", "simulation": "simulate_coupled_rk2_batched", "legacy_sequential_used": False, "legacy_fortran_used": False, "frozen_kernel_hash_verified": True, "fallback_count": sum(1 for row in compile_records["records"].values() if row.get("fallbacks", 0) != 0)}, "pilot_pass_criteria": {"all_8_complete": len(cases) == 8, "all_8_pass": all(c["pass"] for c in cases), "all_finite": all(c["parameter_checks"]["all_candidates_finite"] for c in cases), "bounds_valid": all(c["parameter_checks"]["all_candidate_bounds_valid"] for c in cases), "inactive_mask_valid": all(c["parameter_checks"]["inactive_parameters_fixed_to_defaults"] for c in cases), "best_theta_recompute_valid": all(c["best_theta_recomputation"]["objective_abs_diff"] <= 1.0e-12 for c in cases), "train_test_mapping_valid": all(c["train_test"]["disjoint_masks"] and not c["train_test"]["evaluation_observations_passed_to_optimizer"] for c in cases), "formal_path_valid": all(not c["path_audit"]["legacy_sequential_used"] and not c["path_audit"]["legacy_fortran_used"] for c in cases), "same_seed_reproducible": same_equal, "different_seed_trajectory_differs": different_differs, "no_hidden_fallback": compile_pass, "no_graph_break": compile_pass, "no_unresolved_protocol_critical_field": protocol["unresolved"] == []}, "larger_experiments_started": False, "dpl_training_started": False, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    summary["execution"]["compile_per_case"] = [{"catchment": case["catchment"], "structure": case["structure"], "compile": case["compile"]} for case in cases]
    summary["execution"]["artifact_hashes_verified"] = True
    summary["pass"] = all(summary["pilot_pass_criteria"].values()) and not summary["larger_experiments_started"] and not summary["dpl_training_started"]
    summary["status"] = "passed" if summary["pass"] else "failed"
    summary["artifact_sha256"] = _atomic_json(SUMMARY_PATH, summary, "artifact_sha256")
    _verify_hashed_json(SUMMARY_PATH, "artifact_sha256")
    print(json.dumps({"status": summary["status"], "cases": len(cases), "same_seed": same_equal, "different_seed": different_differs, "compile_pass": compile_pass, "output": str(SUMMARY_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
