"""Generate the three formal performance audit artifacts for formal Torch-SCE pilot."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
RAW_PROFILE_PATH = DOCS / "formal_sce_performance_profile_raw.json"
PILOT_2X4_PATH = DOCS / "formal_sce_pilot_2x4.json"
PROTOCOL_PATH = DOCS / "formal_experiment_protocol_v1.json"
FREEZE_PATH = DOCS / "torch_fuse_v1_freeze.json"


def main() -> None:
    raw = json.loads(RAW_PROFILE_PATH.read_text())
    pilot = json.loads(PILOT_2X4_PATH.read_text())
    protocol = json.loads(PROTOCOL_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())

    # 1. Performance Profile
    # Aggregate data across Model 2 and Model 8
    profile_artifact = {
        "schema_version": "formal-sce-performance-profile-v1",
        "status": "completed",
        "protocol_id": protocol["protocol_id"],
        "torch_fuse_freeze_id": freeze["schema_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "profiled_catchment": "USA_09447800",
            "profiled_structures": [2, 8],
            "pilot_reference_dataset": "formal_sce_pilot_2x4.json (8 cases, 2 evaluations/case, total 2113.6s)",
        },
        "hierarchical_breakdown": {
            "summary_fractions": {
                "hydrological_simulations_combined": 0.99986,
                "optimizer_simulations_fraction": 0.5038,
                "post_optimization_recomputation_fraction": 0.4960,
                "optimizer_bookkeeping_and_candidate_generation": 0.00007,
                "metric_computation_and_sync": 0.00007,
                "setup_and_data_loading": 0.00005,
                "artifact_serialization_and_disk_io": 0.000001,
            },
            "model_2_case_274s": {
                "total_wall_clock_seconds": raw["case_profiles"][0]["wall_clock_seconds"]["case_total"],
                "setup_seconds": raw["case_profiles"][0]["wall_clock_seconds"]["setup"],
                "optimizer_total_seconds": raw["case_profiles"][0]["wall_clock_seconds"]["optimizer_total"],
                "optimizer_simulations_seconds": sum(raw["case_profiles"][0]["objective_evaluations"][i]["simulation_seconds"] for i in range(2)),
                "optimizer_eval_1_sim_seconds": raw["case_profiles"][0]["objective_evaluations"][0]["simulation_seconds"],
                "optimizer_eval_2_sim_seconds": raw["case_profiles"][0]["objective_evaluations"][1]["simulation_seconds"],
                "post_optimization_total_seconds": raw["case_profiles"][0]["wall_clock_seconds"]["post_optimization_total"],
                "post_train_sim_seconds": raw["case_profiles"][0]["wall_clock_seconds"]["post_train_sim"],
                "post_test_sim_seconds": raw["case_profiles"][0]["wall_clock_seconds"]["post_test_sim"],
                "artifact_and_checks_seconds": raw["case_profiles"][0]["wall_clock_seconds"]["artifact_and_checks"],
            },
            "model_8_case_246s": {
                "total_wall_clock_seconds": raw["case_profiles"][1]["wall_clock_seconds"]["case_total"],
                "setup_seconds": raw["case_profiles"][1]["wall_clock_seconds"]["setup"],
                "optimizer_total_seconds": raw["case_profiles"][1]["wall_clock_seconds"]["optimizer_total"],
                "optimizer_simulations_seconds": sum(raw["case_profiles"][1]["objective_evaluations"][i]["simulation_seconds"] for i in range(2)),
                "optimizer_eval_1_sim_seconds": raw["case_profiles"][1]["objective_evaluations"][0]["simulation_seconds"],
                "optimizer_eval_2_sim_seconds": raw["case_profiles"][1]["objective_evaluations"][1]["simulation_seconds"],
                "post_optimization_total_seconds": raw["case_profiles"][1]["wall_clock_seconds"]["post_optimization_total"],
                "post_train_sim_seconds": raw["case_profiles"][1]["wall_clock_seconds"]["post_train_sim"],
                "post_test_sim_seconds": raw["case_profiles"][1]["wall_clock_seconds"]["post_test_sim"],
                "artifact_and_checks_seconds": raw["case_profiles"][1]["wall_clock_seconds"]["artifact_and_checks"],
            },
        },
        "probes": {
            "probe_a_objective_repeat": raw["probe_a_objective_repeat"],
            "probe_b_raw_forward_vs_objective": raw["probe_b_raw_forward"],
            "probe_c_batch_scaling": raw["probe_c_batch_scaling"],
            "probe_d_candidate_batching": raw["probe_d_candidate_batching"],
            "probe_e_loop_breakdown": raw["probe_e_loop_breakdown"],
        },
        "root_cause_ranking": [
            {
                "rank": 1,
                "cause": "Serial unbatched execution (B=1, C=1) on GPU",
                "classification": "confirmed",
                "measured_impact": "99.9% of case runtime is hydrological simulation running at effective batch size 1. Batching C=4 candidates in one forward yields 7.03x speedup (38.3s vs 269.3s for 4 evals); B=2 yields 3.06x speedup.",
                "evidence": "Probe C & D: B=1 takes 67.3s (124.8 days/s), while C=4 takes 38.3s (877.2 days/s total throughput, 7.03x speedup).",
            },
            {
                "rank": 2,
                "cause": "Post-optimization redundant forward simulations (train & test recomputation)",
                "classification": "confirmed",
                "measured_impact": "49.6% of total case runtime (125-132s out of 246-274s per case) is spent re-simulating the entire 8401-day horizon twice after optimization finishes (once for train KGE, once for test KGE).",
                "evidence": "Hierarchical profile: In Model 2, post-opt simulation takes 132.3s out of 274.6s total (48.2%). In Model 8, post-opt simulation takes 125.5s out of 245.9s (51.0%).",
            },
            {
                "rank": 3,
                "cause": "8401-timestep Python time loop and tensor orchestration overhead",
                "classification": "confirmed",
                "measured_impact": "The 8401-iteration loop in Python has ~6.55 ms/step invocation overhead + 6.58s list append/stack/monitor overhead (~10.7% of simulation time). When executed at B=1, GPU compute is underutilized (~5-13% GPU load).",
                "evidence": "Probe E: 8401-step simulation takes 61.6s (~7.3 ms/step). List appends and monitoring add 6.58s. GPU compute saturates only with higher batch size (B*C >= 4).",
            },
            {
                "rank": 4,
                "cause": "Cold compile overhead on initial structure lookup",
                "classification": "confirmed (minor in long runs)",
                "measured_impact": "~1.4s to 4.8s on first structure compile, negligible (0.005s) on warm cache hits. In a 2-eval pilot, it contributes 1-2% of the first case only.",
                "evidence": "Cold compile seconds: Model 2 = 4.79s, Model 8 = 1.42s; warm cache hits = 0.005s.",
            },
            {
                "rank": 5,
                "cause": "Metric computation, host-device synchronization, and artifact I/O",
                "classification": "not material",
                "measured_impact": "Less than 0.01% of case runtime (< 0.05s per case). KGE computation on GPU is < 0.001s, sync is < 0.001s, artifact JSON hashing/writing is < 0.001s.",
                "evidence": "Hierarchical profile: Metric + sync = 0.00014s/case; artifact write = 0.00025s/case.",
            },
        ],
    }

    # 2. Execution Accounting
    accounting_artifact = {
        "schema_version": "formal-sce-execution-accounting-v1",
        "status": "completed",
        "protocol_id": protocol["protocol_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accounting_rule": "Every hydrological simulation across all phases (setup, optimizer, recomputation, reproducibility) is explicitly accounted for.",
        "per_case_accounting": {
            "case_configuration": "1 catchment x 1 structure, max_evaluations=2",
            "forward_simulations_breakdown": [
                {
                    "phase": "optimizer_objective_eval_1",
                    "reason": "SCE initial population candidate 1 evaluation",
                    "forcing_steps": 8401,
                    "forcing_period": ["1987-01-01", "2009-12-31"],
                    "batch_shape": [1, 8401, 3],
                    "candidate_count": 1,
                    "basin_count": 1,
                    "effective_batch_size": 1,
                    "call_count": 1,
                    "wall_clock_seconds_typical": 67.3,
                },
                {
                    "phase": "optimizer_objective_eval_2",
                    "reason": "SCE initial population candidate 2 evaluation",
                    "forcing_steps": 8401,
                    "forcing_period": ["1987-01-01", "2009-12-31"],
                    "batch_shape": [1, 8401, 3],
                    "candidate_count": 1,
                    "basin_count": 1,
                    "effective_batch_size": 1,
                    "call_count": 1,
                    "wall_clock_seconds_typical": 56.2,
                },
                {
                    "phase": "post_optimization_train_metric",
                    "reason": "Recompute train KGE score and state trajectory at saved best theta",
                    "forcing_steps": 8401,
                    "forcing_period": ["1987-01-01", "2009-12-31"],
                    "batch_shape": [1, 8401, 3],
                    "candidate_count": 1,
                    "basin_count": 1,
                    "effective_batch_size": 1,
                    "call_count": 1,
                    "wall_clock_seconds_typical": 64.2,
                },
                {
                    "phase": "post_optimization_test_metric",
                    "reason": "Recompute test KGE score on evaluation period at saved best theta",
                    "forcing_steps": 8401,
                    "forcing_period": ["1987-01-01", "2009-12-31"],
                    "batch_shape": [1, 8401, 3],
                    "candidate_count": 1,
                    "basin_count": 1,
                    "effective_batch_size": 1,
                    "call_count": 1,
                    "wall_clock_seconds_typical": 64.7,
                },
            ],
            "total_simulations_per_case": 4,
            "total_timesteps_per_case": 33604,
            "simulation_to_eval_ratio": "4 full simulations for 2 SCE evaluations (2.0x multiplier)",
        },
        "pilot_2x4_total_accounting": {
            "calibration_cases": 8,
            "calibration_simulations": 8 * 4,
            "reproducibility_runs": 3,
            "reproducibility_evaluations_per_run": 2,
            "reproducibility_simulations": 3 * 2,
            "total_8401_day_simulations_in_pilot": 38,
            "total_timesteps_in_pilot": 38 * 8401,
            "total_wall_clock_seconds": 2113.6,
            "average_seconds_per_8401_day_simulation": float(2113.6 / 38),
        },
    }

    # 3. Batching Audit
    batching_artifact = {
        "schema_version": "formal-sce-batching-audit-v1",
        "status": "completed",
        "protocol_id": protocol["protocol_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_questions": {
            "is_basin_batching_utilized_in_current_sce": False,
            "basin_batch_size_in_pilot": 1,
            "is_candidate_batching_utilized_in_current_sce": False,
            "candidate_batch_size_in_pilot": 1,
            "effective_batch_size_in_pilot": 1,
            "why_b1_in_pilot": "The pilot treats each catchment calibration as an independent serial case to strictly validate per-case correctness and artifact reproducibility.",
            "why_c1_in_pilot": "The minimal SCEBaseline runner implements the canonical simplex update loop serially without batching the initial population or reflection/contraction candidates.",
        },
        "batching_scaling_evidence": {
            "basin_batching_probe": raw["probe_c_batch_scaling"],
            "candidate_batching_probe": raw["probe_d_candidate_batching"],
            "findings": "The compiled Torch-FUSE kernel supports any batch size N along the leading dimension [N, 8401, 3] and [N, 37]. Scaling from N=1 to N=2 gives 3.06x throughput; scaling to N=4 gives 7.03x throughput on this GPU.",
        },
        "combined_batching_feasibility_matrix": [
            {
                "batching_mode": "Option 1: Candidate batching within one catchment (C candidates x 1 basin)",
                "tensor_shapes": "forcing: [C, 8401, 3] (expanded), params: [C, 37]",
                "current_runtime_support": "Supported directly without kernel modification (simulate_coupled_rk2_batched handles arbitrary batch size N)",
                "algorithm_semantics_impact": "Zero change to SCE algorithm for initial population (evaluating all 2*P+1 initial population in 1 forward pass). For simplex reflection/contraction/random, candidates can be batched per complex.",
                "expected_speedup": "3x to 7x per calibration case",
                "risk_level": "low",
            },
            {
                "batching_mode": "Option 2: Multi-basin synchronized SCE batching (B basins x C candidates)",
                "tensor_shapes": "forcing: [B*C, 8401, 3], params: [B*C, 37]",
                "current_runtime_support": "Supported by numerical kernel; requires multi-case scheduler orchestration in Python",
                "algorithm_semantics_impact": "Requires careful lockstep synchronization across basins or masked stepping. If complex evolution diverges, active candidates per basin must be tracked.",
                "expected_speedup": "8x to 15x across multi-catchment campaigns",
                "risk_level": "medium",
            },
            {
                "batching_mode": "Option 3: Eliminate redundant post-optimization simulations",
                "tensor_shapes": "N/A",
                "current_runtime_support": "Immediately possible",
                "algorithm_semantics_impact": "Zero change to science or results. The optimizer objective evaluation for the best theta already produced the exact discharge trajectory on the full 8401-day forcing. Train KGE and test KGE can be computed directly by slicing that discharge tensor without re-running 2 extra 8401-day simulations.",
                "expected_speedup": "2.0x speedup per case (eliminates 50% of runtime per case)",
                "risk_level": "none",
            },
            {
                "batching_mode": "Option 4: GPU-resident forcing and metadata",
                "tensor_shapes": "forcing stays in VRAM",
                "current_runtime_support": "Already partially in VRAM; can be cached across cases",
                "algorithm_semantics_impact": "Zero",
                "expected_speedup": "Minor (< 0.1s/case)",
                "risk_level": "none",
            },
            {
                "batching_mode": "Option 5: Lightweight simulation mode for objective (omit unused flux/state stacks)",
                "tensor_shapes": "only q [N, 8401] accumulated, fluxes/states omitted during optimization",
                "current_runtime_support": "Requires lightweight return mode in batched simulator",
                "algorithm_semantics_impact": "Zero for objective (only Q is needed for KGE). Full states/fluxes retained only for final best-theta artifact.",
                "expected_speedup": "10% to 15% simulation speedup (avoids 38 list appends and 38 torch.stack calls per step)",
                "risk_level": "low",
            },
        ],
    }

    # Write the 3 files
    out_profile = DOCS / "formal_sce_performance_profile.json"
    out_accounting = DOCS / "formal_sce_execution_accounting.json"
    out_batching = DOCS / "formal_sce_batching_audit.json"

    out_profile.write_text(json.dumps(profile_artifact, indent=2, sort_keys=True) + "\n")
    out_accounting.write_text(json.dumps(accounting_artifact, indent=2, sort_keys=True) + "\n")
    out_batching.write_text(json.dumps(batching_artifact, indent=2, sort_keys=True) + "\n")

    print(f"Saved: {out_profile.name}, {out_accounting.name}, {out_batching.name}")


if __name__ == "__main__":
    main()
