"""Detailed profiling and execution accounting for formal Torch-SCE pilot."""
from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfuse import PARAMETER_NAMES, batched_compile_diagnostics, reset_batched_compile_diagnostics, simulate_coupled_rk2_batched
from dfuse.spec import default_parameters, get_structure
from project.autofuse.metrics import kgecomp_batched
from project.autofuse.sce import SCEBaseline, SCEConfig
from project.autofuse.torch_fuse_78_long_horizon_smoke import _dates, _load_frozen_inputs
from project.autofuse.formal_sce_pilot import _bounds, _forcing, _load_and_validate_protocol, _masked_observed, _sha

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
CACHE_DIR = DOCS.parent / ".cache/torch-fuse-perf-audit"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_raw_forward(
    model_id: int,
    forcing: torch.Tensor,
    theta_tensor: torch.Tensor,
    basin_ids: tuple[str, ...],
    *,
    monitor_chunk_size: int = 64,
) -> dict[str, Any]:
    _cuda_sync()
    t0 = time.perf_counter()
    res = simulate_coupled_rk2_batched(
        model_id,
        forcing,
        theta_tensor,
        basin_ids=basin_ids,
        compile_step=True,
        compile_backend="inductor",
        compile_fullgraph=True,
        monitor_chunk_size=monitor_chunk_size,
    )
    _cuda_sync()
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_seconds": elapsed,
        "n_steps": int(res.q.shape[1]),
        "batch_size": int(res.q.shape[0]),
        "steps_per_second": float(res.q.shape[1] / elapsed),
        "basin_days_per_second": float(res.q.shape[0] * res.q.shape[1] / elapsed),
        "max_water_balance_abs": float(res.max_abs_water_balance_error.detach().cpu()),
    }


def _profile_hierarchical_case(
    protocol: dict[str, Any],
    freeze: dict[str, Any],
    inputs: dict[str, dict[str, np.ndarray]],
    basin_id: str,
    model_id: int,
    seed: int,
    dates: list[Any],
    device: torch.device,
) -> dict[str, Any]:
    _cuda_sync()
    t_case_start = time.perf_counter()

    # 1. Setup phase
    t_setup_start = time.perf_counter()
    values = inputs[basin_id]
    full_forcing = _forcing(values, device=device)
    q_obs = np.asarray(values["q_obs"], dtype=np.float64)
    train_obs = _masked_observed(q_obs, dates, *protocol["periods"]["calibration"], device=device)
    test_obs = _masked_observed(q_obs, dates, *protocol["periods"]["evaluation"], device=device)
    train_mask = np.asarray([protocol["periods"]["calibration"][0] <= d.isoformat() <= protocol["periods"]["calibration"][1] for d in dates])
    test_mask = np.asarray([protocol["periods"]["evaluation"][0] <= d.isoformat() <= protocol["periods"]["evaluation"][1] for d in dates])
    epsilon = float(np.mean(q_obs[train_mask]) / 100.0)
    settings = protocol["sce"]["settings"]
    config = SCEConfig(max_evaluations=int(settings["max_evaluations"]), kstop=int(settings["kstop"]), pcento=float(settings["pcento"]), seed=seed, n_complexes=int(settings["n_complexes"]))
    case_bounds = _bounds(protocol, model_id)
    _cuda_sync()
    t_setup = time.perf_counter() - t_setup_start

    # 2. Optimizer phase with sub-instrumentation
    t_opt_start = time.perf_counter()
    evaluator = SCEBaseline().evaluator

    objective_timings = []
    def instrumented_objective(m_id, forc, obs, param_vec, **kwargs):
        _cuda_sync()
        t_obj_start = time.perf_counter()
        
        # Simulation
        t_sim_start = time.perf_counter()
        forward_kwargs = {k: v for k, v in kwargs.items() if k != "inverse_epsilon"}
        sim_res = evaluator.forward_batched(m_id, forc, param_vec, **forward_kwargs)
        _cuda_sync()
        t_sim = time.perf_counter() - t_sim_start

        # Metric
        t_metric_start = time.perf_counter()
        obs_dev = obs.to(device=sim_res.q.device, dtype=sim_res.q.dtype)
        kge = kgecomp_batched(sim_res.q, obs_dev, epsilon=epsilon)
        _cuda_sync()
        t_metric = time.perf_counter() - t_metric_start

        # Sync
        t_sync_start = time.perf_counter()
        score = 1.0 - kge.mean()
        score_val = float(score.detach().cpu())
        t_sync = time.perf_counter() - t_sync_start

        t_obj_total = time.perf_counter() - t_obj_start
        objective_timings.append({
            "simulation_seconds": t_sim,
            "metric_seconds": t_metric,
            "sync_seconds": t_sync,
            "total_seconds": t_obj_total,
            "steps": int(forc.shape[1]),
            "batch_size": int(forc.shape[0]),
        })
        return score

    # Run optimizer with instrumented evaluator
    sce_instance = SCEBaseline(config, evaluator=evaluator)
    sce_instance.objective_batched = instrumented_objective
    run = sce_instance.run(model_id, full_forcing, train_obs, basin_ids=(basin_id,), bounds=case_bounds, inverse_epsilon=epsilon, compile_step=True, compile_backend=protocol["execution"]["compile_backend"], compile_fullgraph=bool(protocol["execution"]["compile_fullgraph"]))
    _cuda_sync()
    t_opt = time.perf_counter() - t_opt_start

    # 3. Post-optimization recomputation phase
    t_post_start = time.perf_counter()
    best_theta = {name: float(run["best_parameters"][name]) for name in PARAMETER_NAMES}
    theta_tensor = torch.as_tensor([best_theta[name] for name in PARAMETER_NAMES], dtype=torch.float64, device=device)

    # Train recompute simulation
    _cuda_sync()
    t_train_sim_start = time.perf_counter()
    with torch.no_grad():
        recomputed_train = evaluator.score_batched(model_id, full_forcing, train_obs, theta_tensor, basin_ids=(basin_id,), inverse_epsilon=epsilon, compile_step=True, compile_backend=protocol["execution"]["compile_backend"], compile_fullgraph=bool(protocol["execution"]["compile_fullgraph"]))
    _cuda_sync()
    t_train_sim = time.perf_counter() - t_train_sim_start

    # Test recompute simulation
    _cuda_sync()
    t_test_sim_start = time.perf_counter()
    with torch.no_grad():
        recomputed_test = evaluator.score_batched(model_id, full_forcing, test_obs, theta_tensor, basin_ids=(basin_id,), inverse_epsilon=epsilon, compile_step=True, compile_backend=protocol["execution"]["compile_backend"], compile_fullgraph=bool(protocol["execution"]["compile_fullgraph"]))
    _cuda_sync()
    t_test_sim = time.perf_counter() - t_test_sim_start

    train_metric = float(recomputed_train.kge_comp[0].detach().cpu())
    test_metric = float(recomputed_test.kge_comp[0].detach().cpu())
    best_objective = float(run["best_score"])
    recomputed_objective = 1.0 - train_metric

    _cuda_sync()
    t_post = time.perf_counter() - t_post_start

    # 4. Artifact & verification phase
    t_artifact_start = time.perf_counter()
    # Dummy serialization test
    payload = {"catchment": basin_id, "structure": model_id, "best_objective": best_objective, "train_metric": train_metric, "test_metric": test_metric}
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = _sha(Path(protocol["sources"]["torch_fuse_freeze"]["path"]))
    t_artifact = time.perf_counter() - t_artifact_start

    _cuda_sync()
    t_case_total = time.perf_counter() - t_case_start

    return {
        "catchment": basin_id,
        "structure": model_id,
        "seed": seed,
        "wall_clock_seconds": {
            "case_total": t_case_total,
            "setup": t_setup,
            "optimizer_total": t_opt,
            "post_optimization_total": t_post,
            "post_train_sim": t_train_sim,
            "post_test_sim": t_test_sim,
            "artifact_and_checks": t_artifact,
        },
        "fractions_of_case_total": {
            "setup": t_setup / t_case_total,
            "optimizer_total": t_opt / t_case_total,
            "optimizer_simulations": sum(item["simulation_seconds"] for item in objective_timings) / t_case_total,
            "optimizer_metrics_and_sync": sum(item["metric_seconds"] + item["sync_seconds"] for item in objective_timings) / t_case_total,
            "post_optimization_total": t_post / t_case_total,
            "post_train_sim": t_train_sim / t_case_total,
            "post_test_sim": t_test_sim / t_case_total,
            "artifact_and_checks": t_artifact / t_case_total,
            "all_simulations_combined": (sum(item["simulation_seconds"] for item in objective_timings) + t_train_sim + t_test_sim) / t_case_total,
        },
        "objective_evaluations": objective_timings,
        "evaluation_counts": {
            "sce_evaluations": len(run["trajectory"]),
            "optimizer_simulations": len(objective_timings),
            "post_optimization_simulations": 2,
            "total_8401_day_simulations": len(objective_timings) + 2,
            "total_timesteps_simulated": (len(objective_timings) + 2) * int(full_forcing.shape[1]),
        },
        "best_score": best_objective,
        "recomputed_score": recomputed_objective,
        "diff": abs(best_objective - recomputed_objective),
    }


def main() -> None:
    print("Initializing profiling environment...")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR.resolve())
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    protocol, freeze = _load_and_validate_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    _, inputs, _ = _load_frozen_inputs()
    dates = _dates()

    basin_id = "USA_09447800"
    structures_to_test = [2, 8]

    # Run hierarchical cases
    case_profiles = []
    for model_id in structures_to_test:
        reset_batched_compile_diagnostics()
        print(f"Running hierarchical profile for Model {model_id} on {basin_id}...")
        profile = _profile_hierarchical_case(protocol, freeze, inputs, basin_id, model_id, 20260901, dates, device)
        case_profiles.append(profile)
        print(f"Model {model_id} finished in {profile['wall_clock_seconds']['case_total']:.2f}s (all sim: {profile['fractions_of_case_total']['all_simulations_combined']*100:.1f}%)")

    # Probe A: Objective-only repeated calls (warm-up / cache)
    # Probe A: Objective-only repeated calls (extracted directly from hierarchical profiles)
    print("Extracting Probe A: Objective-only repeated calls from profile...")
    probe_a_results = []
    for prof in case_profiles:
        probe_a_results.append({
            "model_id": prof["structure"],
            "calls": [
                {"call_index": 1, "simulation_seconds": prof["objective_evaluations"][0]["simulation_seconds"], "total_seconds": prof["objective_evaluations"][0]["total_seconds"]},
                {"call_index": 2, "simulation_seconds": prof["objective_evaluations"][1]["simulation_seconds"], "total_seconds": prof["objective_evaluations"][1]["total_seconds"]},
                {"call_index": 3, "description": "post_train_recompute", "simulation_seconds": prof["wall_clock_seconds"]["post_train_sim"], "total_seconds": prof["wall_clock_seconds"]["post_train_sim"]},
                {"call_index": 4, "description": "post_test_recompute", "simulation_seconds": prof["wall_clock_seconds"]["post_test_sim"], "total_seconds": prof["wall_clock_seconds"]["post_test_sim"]},
            ],
        })

    # Probe B: Raw forward vs full objective vs full case
    print("Extracting Probe B: Raw forward vs full objective vs full case from profile...")
    raw_forward_results = []
    for prof in case_profiles:
        raw_sim = prof["objective_evaluations"][1]["simulation_seconds"]
        full_obj = prof["objective_evaluations"][1]["total_seconds"]
        case_tot = prof["wall_clock_seconds"]["case_total"]
        raw_forward_results.append({
            "model_id": prof["structure"],
            "raw_simulation_seconds": raw_sim,
            "full_objective_seconds": full_obj,
            "full_case_seconds": case_tot,
            "sim_fraction_of_objective": raw_sim / full_obj,
            "sim_fraction_of_case": (raw_sim * 4) / case_tot,
            "basin_days_per_second": float(8401 / raw_sim),
        })

    # Probe C: Batch scaling B=1 vs B=2
    print("Running Probe C: Batch scaling B=1, B=2...")
    batch_scaling_results = []
    values = inputs[basin_id]
    forcing_single = _forcing(values, device=device)
    defaults = default_parameters()
    theta_single = torch.as_tensor([defaults[name] for name in PARAMETER_NAMES], dtype=torch.float64, device=device)
    b1_sec = raw_forward_results[0]["raw_simulation_seconds"]
    batch_scaling_results.append({
        "batch_size": 1,
        "elapsed_seconds": b1_sec,
        "basin_days_per_second": float(8401 / b1_sec),
        "speedup_vs_b1": 1.0,
    })
    # Test B=2
    forcing_2 = forcing_single.expand(2, -1, -1).contiguous()
    theta_2 = theta_single.unsqueeze(0).expand(2, -1).contiguous()
    _cuda_sync()
    t0 = time.perf_counter()
    res_2 = simulate_coupled_rk2_batched(2, forcing_2, theta_2, basin_ids=(f"{basin_id}_0", f"{basin_id}_1"), compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    _cuda_sync()
    t_2 = time.perf_counter() - t0
    batch_scaling_results.append({
        "batch_size": 2,
        "elapsed_seconds": t_2,
        "basin_days_per_second": float(2 * 8401 / t_2),
        "speedup_vs_b1": float((2 * 8401 / t_2) / (8401 / b1_sec)),
    })

    # Probe E: 8401-step Python loop detailed breakdown
    print("Running Probe E: 8401-step Python loop fine-grained timing...")
    spec_2 = get_structure(2)
    theta_single_map = [{name: theta_single[position] for position, name in enumerate(PARAMETER_NAMES)}]
    from dfuse.batched import _compiled_step, _initial_batch, _dates_for_batch, _monitor_code, _FAILURE_BITS, FLUX_NAMES, COUPLED_RK2_DIAGNOSTIC_NAMES, STATE_NAMES
    init_st, t_mean, t_max, fracs, caps = _initial_batch(spec_2, theta_single_map, fraction=0.25)
    bnd = torch.ones_like(caps, dtype=torch.bool)
    dys, lps = _dates_for_batch(8401, dates, device=device, dtype=torch.float64)
    pck = torch.cat((init_st, torch.zeros((1, 1 + 500), device=device, dtype=torch.float64)), dim=1)
    step_fn = _compiled_step(spec_2, batch_size=1, device=device, dtype=torch.float64, backend="inductor", fullgraph=True)
    
    # Warmup 8 steps
    for idx in range(8):
        s_f = torch.cat((forcing_single[:, idx, :], dys[idx].expand(1, 1), lps[idx].to(torch.float64).expand(1, 1), torch.full((1, 1), 1.0, dtype=torch.float64, device=device)), dim=1)
        pck, rtd, dg = step_fn(pck, s_f, theta_single.unsqueeze(0), t_mean, t_max, fracs)
    _cuda_sync()
    
    # Measure pure CUDA execution time of 8401 step_fn calls
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    _cuda_sync()
    t_loop_host_start = time.perf_counter()
    start_event.record()
    for idx in range(8401):
        s_f = torch.cat((forcing_single[:, idx, :], dys[idx].expand(1, 1), lps[idx].to(torch.float64).expand(1, 1), torch.full((1, 1), 1.0, dtype=torch.float64, device=device)), dim=1)
        pck, rtd, dg = step_fn(pck, s_f, theta_single.unsqueeze(0), t_mean, t_max, fracs)
    end_event.record()
    _cuda_sync()
    t_loop_host_only_step_fn = time.perf_counter() - t_loop_host_start
    pure_gpu_kernel_ms = start_event.elapsed_time(end_event)
    
    # Measure full loop with all list appends, monitoring, stacks
    _cuda_sync()
    t_full_sim_start = time.perf_counter()
    full_sim_res = simulate_coupled_rk2_batched(2, forcing_single, theta_single.unsqueeze(0), basin_ids=(basin_id,), compile_step=True, compile_backend="inductor", compile_fullgraph=True, monitor_chunk_size=64)
    _cuda_sync()
    t_full_sim = time.perf_counter() - t_full_sim_start
    
    loop_breakdown_result = {
        "timesteps": 8401,
        "full_simulate_coupled_rk2_batched_seconds": t_full_sim,
        "pure_gpu_kernel_execution_seconds": float(pure_gpu_kernel_ms / 1000.0),
        "pure_gpu_kernel_ms_per_step": float(pure_gpu_kernel_ms / 8401.0),
        "host_step_fn_loop_seconds": t_loop_host_only_step_fn,
        "host_step_fn_loop_ms_per_step": float(t_loop_host_only_step_fn * 1000.0 / 8401.0),
        "list_appends_monitoring_and_stack_overhead_seconds": float(t_full_sim - t_loop_host_only_step_fn),
        "gpu_kernel_time_fraction_of_simulation": float((pure_gpu_kernel_ms / 1000.0) / t_full_sim),
        "cpu_overhead_fraction_of_simulation": float(1.0 - (pure_gpu_kernel_ms / 1000.0) / t_full_sim),
    }

    # Probe D: Candidate batching feasibility
    print("Running Probe D: Candidate batching feasibility (C=4 on single basin)...")
    c_candidates = 4
    forcing_c = forcing_single.expand(c_candidates, -1, -1).contiguous()
    theta_candidates = torch.stack([theta_single * (1.0 + 0.01 * i) for i in range(c_candidates)], dim=0)
    basin_ids_c = tuple(f"{basin_id}_c{i}" for i in range(c_candidates))
    
    _cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        res_c = simulate_coupled_rk2_batched(2, forcing_c, theta_candidates, basin_ids=basin_ids_c, compile_step=True, compile_backend="inductor", compile_fullgraph=True)
    _cuda_sync()
    t_c = time.perf_counter() - t0

    candidate_batch_result = {
        "candidates": c_candidates,
        "elapsed_seconds": t_c,
        "evaluations_per_second": float(c_candidates / t_c),
        "speedup_vs_serial_c": float((c_candidates * raw_forward_results[0]["raw_simulation_seconds"]) / t_c),
    }

    # Compile diagnostics summary
    comp_diag = batched_compile_diagnostics()

    summary_output = {
        "schema_version": "formal-sce-performance-profile-v1",
        "protocol_id": protocol["protocol_id"],
        "case_profiles": case_profiles,
        "probe_a_objective_repeat": probe_a_results,
        "probe_b_raw_forward": raw_forward_results,
        "probe_c_batch_scaling": batch_scaling_results,
        "probe_d_candidate_batching": candidate_batch_result,
        "probe_e_loop_breakdown": loop_breakdown_result,
        "compile_diagnostics": comp_diag,
    }
    
    out_file = DOCS / "formal_sce_performance_profile_raw.json"
    out_file.write_text(json.dumps(summary_output, indent=2))
    print(f"Saved raw profiling results to {out_file}")


if __name__ == "__main__":
    main()
