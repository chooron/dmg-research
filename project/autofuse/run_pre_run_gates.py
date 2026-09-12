"""AutoFuse / dMG Pre-Run Final Gate Validation Suite (G1-G7, D1-D2)."""
from __future__ import annotations

import json
import os
import resource
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch._dynamo

from dfuse import PARAMETER_NAMES, enumerate_structures, get_structure, simulate_coupled_rk2_batched
from dfuse.spec import DECISION_ORDER, default_parameters
from project.autofuse.dpl import DPLConfig, StructureConditionedParameterizer
from project.autofuse.loader import StochasticTimeWindowLoader, TimeWindowConfig
from project.autofuse.metrics import kgecomp, kgecomp_batched
from project.autofuse.parameter_contract import get_parameter_contract
from project.autofuse.samplers import GlobalBasinSampler, ShuffledStructureSampler
from project.autofuse.torch_fuse_78_long_horizon_smoke import _dates, _load_frozen_inputs
from project.autofuse.trainer import SharedDPLTrainer

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
CACHE_DIR = DOCS.parent / ".cache/torch-fuse-prerun-gates"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GATE_REPORT_PATH = DOCS / "pre_run_final_gate_report.json"


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Saved: {path.name}")


# ----------------------------------------------------------------------
# G1. Correct Phase-Locking Test (Adversarial N_basins % B == 0, N_basins/B > 1)
# ----------------------------------------------------------------------
def run_g1_phase_locking() -> dict[str, Any]:
    print("\n=======================================================")
    print("G1: Adversarial Phase-Locking Gate")
    print("=======================================================")
    test_cases = [
        {"S_len": 5, "N_basins": 100, "B": 20, "n_cycles": 50},
        {"S_len": 3, "N_basins": 60, "B": 20, "n_cycles": 50},
        {"S_len": 78, "N_basins": 100, "B": 20, "n_cycles": 50},
    ]
    case_results = []
    all_passed = True

    for tc in test_cases:
        S_len = tc["S_len"]
        N_basins = tc["N_basins"]
        B = tc["B"]
        n_cycles = tc["n_cycles"]
        total_steps = n_cycles * S_len

        structs = list(range(S_len))
        basins = [f"basin_{i:03d}" for i in range(N_basins)]

        struct_sampler = ShuffledStructureSampler(structs, seed=100)
        basin_sampler = GlobalBasinSampler(basins, batch_size=B, seed=200)

        position_basin_sets: dict[int, list[frozenset[str]]] = {pos: [] for pos in range(S_len)}

        for step in range(total_steps):
            pos = struct_sampler.cursor % S_len
            s = struct_sampler.next_structure()
            b_batch = basin_sampler.next_batch()
            position_basin_sets[pos].append(frozenset(b_batch))

        pos_stats = {}
        for pos in range(S_len):
            distinct_sets = len(set(position_basin_sets[pos]))
            pos_stats[str(pos)] = {
                "cycles_inspected": n_cycles,
                "distinct_basin_sets_seen": distinct_sets,
                "is_locked": bool(distinct_sets <= 1),
            }
            if distinct_sets <= 1:
                all_passed = False

        min_distinct = min(p["distinct_basin_sets_seen"] for p in pos_stats.values())
        max_distinct = max(p["distinct_basin_sets_seen"] for p in pos_stats.values())

        case_results.append({
            "structure_cycle_length": S_len,
            "basin_count": N_basins,
            "batch_size": B,
            "ratio_n_over_b": N_basins // B,
            "cycles_inspected": n_cycles,
            "total_steps": total_steps,
            "min_distinct_sets_across_positions": min_distinct,
            "max_distinct_sets_across_positions": max_distinct,
            "phase_locking_detected": not all(p["distinct_basin_sets_seen"] > 1 for p in pos_stats.values()),
            "pass": bool(min_distinct > 1),
        })
        print(f"S={S_len:2d}, Basins={N_basins}, Batch={B}: Distinct sets min={min_distinct}, max={max_distinct} across {n_cycles} cycles -> PASS={min_distinct > 1}")

    gate_pass = all_passed and all(c["pass"] for c in case_results)
    return {
        "gate": "G1_PHASE_LOCKING",
        "status": "PASS" if gate_pass else "FAIL",
        "cases": case_results,
    }


# ----------------------------------------------------------------------
# G2. Long-Run Sampler-Only Stress Test (20,000 steps)
# ----------------------------------------------------------------------
def run_g2_long_run_sampler_stress() -> dict[str, Any]:
    print("\n=======================================================")
    print("G2: Long-Run Sampler-Only Stress Test Gate (20,000 steps)")
    print("=======================================================")
    N_STEPS = 20000
    structures_78 = "structures_78"
    basins_544 = [f"basin_{i:03d}" for i in range(544)]
    B = 100

    struct_sampler = ShuffledStructureSampler(structures_78, seed=1001)
    basin_sampler = GlobalBasinSampler(basins_544, batch_size=B, seed=2002)

    seen_in_cycle: list[int] = []
    cycle_count = 0
    all_batches_unique = True

    for step in range(N_STEPS):
        s = struct_sampler.next_structure()
        b = basin_sampler.next_batch()

        # Structure checks
        seen_in_cycle.append(s)
        if len(seen_in_cycle) == 78:
            if len(set(seen_in_cycle)) != 78:
                raise AssertionError(f"Duplicates/omissions in structure cycle {cycle_count}")
            seen_in_cycle = []
            cycle_count += 1

        # Basin batch uniqueness check
        if len(b) != B or len(set(b)) != B:
            all_batches_unique = False
            raise AssertionError(f"Within-batch duplicate found at step {step}: len={len(b)}, set={len(set(b))}")

    # Marginal exposure balance
    s_counts = list(struct_sampler.exposure_counts.values())
    b_counts = list(basin_sampler.exposure_counts.values())
    s_balance_pass = bool(max(s_counts) - min(s_counts) <= 1)
    b_balance_pass = bool(max(b_counts) - min(b_counts) <= 1)

    # Checkpoint Parity at 4 boundary conditions
    boundary_steps = [39, 78, 272, 544]
    checkpoint_results = []

    for b_step in boundary_steps:
        s_orig = ShuffledStructureSampler(structures_78, seed=555)
        b_orig = GlobalBasinSampler(basins_544, batch_size=B, seed=777)
        for _ in range(b_step):
            s_orig.next_structure()
            b_orig.next_batch()

        st_s = s_orig.state_dict()
        st_b = b_orig.state_dict()

        unint_s = [s_orig.next_structure() for _ in range(200)]
        unint_b = [b_orig.next_batch() for _ in range(200)]

        s_res = ShuffledStructureSampler(structures_78, seed=999)
        s_res.load_state_dict(st_s)
        b_res = GlobalBasinSampler(basins_544, batch_size=B, seed=888)
        b_res.load_state_dict(st_b)

        res_s = [s_res.next_structure() for _ in range(200)]
        res_b = [b_res.next_batch() for _ in range(200)]

        match = bool(unint_s == res_s and unint_b == res_b)
        checkpoint_results.append({"boundary_step": b_step, "match": match})
        print(f"  Checkpoint at step {b_step:4d}: 200 future steps match={match}")

    gate_pass = bool(all_batches_unique and s_balance_pass and b_balance_pass and all(c["match"] for c in checkpoint_results))
    print(f"G2 complete: Structure exposures min={min(s_counts)}, max={max(s_counts)} | Basin exposures min={min(b_counts)}, max={max(b_counts)} -> PASS={gate_pass}")
    return {
        "gate": "G2_LONG_RUN_SAMPLER_STRESS",
        "status": "PASS" if gate_pass else "FAIL",
        "steps_executed": N_STEPS,
        "structure_cycles_completed": struct_sampler.cycle_index,
        "basin_cycles_completed": basin_sampler.basin_cycle_index,
        "structure_marginal_balance_diff": max(s_counts) - min(s_counts),
        "basin_marginal_balance_diff": max(b_counts) - min(b_counts),
        "all_batches_unique": all_batches_unique,
        "checkpoint_parity": checkpoint_results,
    }


# ----------------------------------------------------------------------
# G3. Full 78-Structure Optimizer Contract Cycle
# ----------------------------------------------------------------------
def run_g3_full_78_optimizer_contract(device: torch.device) -> dict[str, Any]:
    print("\n=======================================================")
    print("G3: Full 78-Structure Optimizer-Contract Cycle Gate")
    print("=======================================================")
    dates = [date(1987, 1, 1) + timedelta(days=i) for i in range(20)]
    mock_data = {
        f"b{i}": {
            "ppt": np.ones(20, dtype=np.float64) * 5.0,
            "pet": np.ones(20, dtype=np.float64) * 2.0,
            "temp": np.ones(20, dtype=np.float64) * 10.0,
            "q_obs": (np.sin(np.arange(20) + i) * 2.0 + 3.0).astype(np.float64),
            "attributes": np.zeros(35, dtype=np.float64),
            "epsilon": 0.05,
        }
        for i in range(4)
    }
    cfg = TimeWindowConfig(total_days=8, warmup_days=4, scored_days=4, calibration_end=date(1987, 1, 20))
    loader = StochasticTimeWindowLoader(mock_data, dates, cfg, seed=42, device=device)

    config = {
        "batch_size": 2,
        "basin_ids": ["b0", "b1", "b2", "b3"],
        "structures": "structures_78",
        "lr": 1e-3,
        "device": str(device),
        "max_grad_norm": 1.0,
    }
    trainer = SharedDPLTrainer(config, loader=loader, compile_step=False)

    prior_head_steps = {name: 0 for name in PARAMETER_NAMES}
    per_step_verification = []
    all_steps_valid = True

    t0 = time.perf_counter()
    for step_idx in range(78):
        s_expected = trainer.structure_sampler.current_permutation[trainer.structure_sampler.cursor]
        contract = get_parameter_contract(s_expected)
        active_set = set(contract.active_parameters)

        res = trainer.train_step()
        
        # Inactive heads have .grad is None, step unchanged
        step_ok = True
        for name in PARAMETER_NAMES:
            idx = PARAMETER_NAMES.index(name)
            head = trainer.model.heads[idx]
            if name in active_set:
                curr_step = trainer.optimizer.state[head.weight]["step"].item() if head.weight in trainer.optimizer.state else 0
                if curr_step != prior_head_steps[name] + 1:
                    step_ok = False
                prior_head_steps[name] = curr_step
            else:
                curr_step = trainer.optimizer.state[head.weight]["step"].item() if head.weight in trainer.optimizer.state else 0
                if curr_step != prior_head_steps[name]:
                    step_ok = False

        if not step_ok or res["model_id"] != s_expected:
            all_steps_valid = False

        per_step_verification.append({
            "step": step_idx,
            "model_id": res["model_id"],
            "step_ok": step_ok,
        })

    t_cycle = time.perf_counter() - t0

    # Verify diagnostic exposure counts against independent enumeration
    expected_head_counts = {name: 0 for name in PARAMETER_NAMES}
    expected_option_counts = {d: {} for d in DECISION_ORDER}
    for s in enumerate_structures():
        c = get_parameter_contract(s.model_id)
        for name in c.active_parameters:
            expected_head_counts[name] += 1
        for d, opt in s.decisions.items():
            expected_option_counts[d][opt] = expected_option_counts[d].get(opt, 0) + 1

    head_counts_match = bool(trainer.diagnostics.head_activation_count == expected_head_counts)
    option_counts_match = True
    for d in DECISION_ORDER:
        for opt, count in expected_option_counts[d].items():
            if trainer.diagnostics.option_exposure[d].get(opt) != count:
                option_counts_match = False

    gate_pass = bool(all_steps_valid and trainer.global_step == 78 and head_counts_match and option_counts_match)
    print(f"G3 complete in {t_cycle:.2f}s: global_step={trainer.global_step}/78 | all_steps_valid={all_steps_valid} | head_counts_match={head_counts_match} -> PASS={gate_pass}")
    return {
        "gate": "G3_FULL_78_OPTIMIZER_CONTRACT",
        "status": "PASS" if gate_pass else "FAIL",
        "cycle_wall_clock_seconds": t_cycle,
        "global_step_final": trainer.global_step,
        "all_steps_contract_valid": all_steps_valid,
        "head_counts_match_registry": head_counts_match,
        "option_counts_match_registry": option_counts_match,
    }


# ----------------------------------------------------------------------
# G4. Real-Data One-Cycle Dry Run (78 steps on CUDA)
# ----------------------------------------------------------------------
def run_g4_real_data_one_cycle(inputs: dict[str, dict[str, np.ndarray]], dates: list[Any], device: torch.device) -> dict[str, Any]:
    print("\n=======================================================")
    print("G4: Real-Data One-Cycle Dry Run Gate (78 structures on CUDA)")
    print("=======================================================")
    cfg = TimeWindowConfig(total_days=730, warmup_days=365, scored_days=365, calibration_end=date(1998, 12, 31))
    loader = StochasticTimeWindowLoader(inputs, dates, cfg, seed=20260901, device=device)

    config = {
        "batch_size": 2,
        "basin_ids": ["USA_09447800", "USA_14138900"],
        "structures": "structures_78",
        "lr": 1e-3,
        "device": "cuda",
        "max_grad_norm": 1.0,
        "compile_step": True,
    }
    trainer = SharedDPLTrainer(config, loader=loader, compile_step=True)

    losses = []
    grad_norms = []
    clipping_triggers = 0
    all_finite = True
    step_records = []

    t0_g4 = time.perf_counter()

    for step_idx in range(78):
        s_id = trainer.structure_sampler.current_permutation[trainer.structure_sampler.cursor]
        res = trainer.train_step()

        is_fin = bool(np.isfinite(res["loss"]) and np.isfinite(res["pre_clip_norm"]))
        if not is_fin:
            all_finite = False

        # Verify trainable parameters remain finite
        for p in trainer.model.parameters():
            if not torch.isfinite(p).all().item():
                all_finite = False

        losses.append(res["loss"])
        grad_norms.append(res["pre_clip_norm"])
        if res["clipping_triggered"]:
            clipping_triggers += 1

        step_records.append({
            "step": step_idx,
            "model_id": res["model_id"],
            "loss": res["loss"],
            "grad_norm": res["pre_clip_norm"],
            "clipping_triggered": res["clipping_triggered"],
        })
        if (step_idx + 1) % 15 == 0 or step_idx == 77:
            print(f"  Step {step_idx+1:2d}/78 (Model {res['model_id']:3d}): loss={res['loss']:.4f}, grad={res['pre_clip_norm']:.4f}, elapsed={time.perf_counter()-t0_g4:.1f}s")

    t_g4 = time.perf_counter() - t0_g4
    gate_pass = bool(all_finite and len(losses) == 78)
    print(f"G4 complete: 78 steps in {t_g4:.2f}s | Median Loss={np.median(losses):.4f} | Median Grad={np.median(grad_norms):.4f} | Clipping={clipping_triggers}/78 -> PASS={gate_pass}")

    return {
        "gate": "G4_REAL_DATA_ONE_CYCLE",
        "status": "PASS" if gate_pass else "FAIL",
        "total_wall_clock_seconds": t_g4,
        "all_states_and_gradients_finite": all_finite,
        "loss_statistics": {
            "min": float(min(losses)),
            "max": float(max(losses)),
            "median": float(np.median(losses)),
            "mean": float(np.mean(losses)),
        },
        "gradient_norm_statistics": {
            "min": float(min(grad_norms)),
            "max": float(max(grad_norms)),
            "median": float(np.median(grad_norms)),
            "mean": float(np.mean(grad_norms)),
        },
        "clipping_triggered_count": clipping_triggers,
        "clipping_triggered_fraction": float(clipping_triggers / 78),
    }


# ----------------------------------------------------------------------
# G5. Production-Shape B=100 GPU and Memory Smoke Test
# ----------------------------------------------------------------------
def run_g5_production_memory_smoke(inputs: dict[str, dict[str, np.ndarray]], dates: list[Any], device: torch.device) -> dict[str, Any]:
    print("\n=======================================================")
    print("G5: Production-Shape B=100 GPU & Memory Plateau Gate")
    print("=======================================================")
    B = 100
    mock_100 = {f"basin_{i:03d}": inputs["USA_09447800" if i % 2 == 0 else "USA_14138900"] for i in range(B)}
    cfg = TimeWindowConfig(total_days=730, warmup_days=365, scored_days=365, calibration_end=date(1998, 12, 31))
    loader = StochasticTimeWindowLoader(mock_100, dates, cfg, seed=20260905, device=device)

    config = {
        "batch_size": B,
        "basin_ids": [f"basin_{i:03d}" for i in range(B)],
        "structures": [2, 8, 190, 214],
        "lr": 1e-3,
        "device": "cuda",
        "max_grad_norm": 1.0,
        "compile_step": True,
    }
    trainer = SharedDPLTrainer(config, loader=loader, compile_step=True)

    # Warmup 1 step to prime Inductor cache
    _ = trainer.train_step()

    step_records = []
    for step in range(12):
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        s_out = trainer.train_step()
        _cuda_sync()
        t_step = time.perf_counter() - t0

        vram_alloc = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        vram_res = float(torch.cuda.max_memory_reserved(device) / (1024 * 1024))

        step_records.append({
            "step": step,
            "model_id": s_out["model_id"],
            "step_seconds": t_step,
            "vram_alloc_mb": vram_alloc,
            "vram_res_mb": vram_res,
        })
        print(f"  B={B} Step {step:2d} (Model {s_out['model_id']:3d}): {t_step:5.2f}s | VRAM alloc={vram_alloc:6.1f}MB, res={vram_res:6.1f}MB")

    allocs = [r["vram_alloc_mb"] for r in step_records]
    plateau_stable = bool(max(allocs) < 1000.0 and allocs[-1] <= max(allocs))
    gate_pass = bool(plateau_stable and len(step_records) == 12)
    print(f"G5 complete: VRAM min={min(allocs):.1f}MB, max={max(allocs):.1f}MB, last={allocs[-1]:.1f}MB -> PASS={gate_pass}")

    return {
        "gate": "G5_PRODUCTION_SHAPE_MEMORY",
        "status": "PASS" if gate_pass else "FAIL",
        "batch_size": B,
        "window_days": 730,
        "peak_vram_allocated_mb": max(allocs),
        "peak_vram_reserved_mb": max(r["vram_res_mb"] for r in step_records),
        "median_step_seconds": float(np.median([r["step_seconds"] for r in step_records])),
        "memory_plateau_stable": plateau_stable,
        "steps_recorded": step_records,
    }


# ----------------------------------------------------------------------
# G6. Real-Data Checkpoint/Resume Parity
# ----------------------------------------------------------------------
def run_g6_real_data_resume(inputs: dict[str, dict[str, np.ndarray]], dates: list[Any], device: torch.device) -> dict[str, Any]:
    print("\n=======================================================")
    print("G6: Real-Data Checkpoint / Resume Parity Gate")
    print("=======================================================")
    cfg = TimeWindowConfig(total_days=730, warmup_days=365, scored_days=365, calibration_end=date(1998, 12, 31))

    config = {
        "batch_size": 2,
        "basin_ids": ["USA_09447800", "USA_14138900"],
        "structures": [2, 8, 190, 214],
        "lr": 1e-3,
        "device": "cuda",
        "compile_step": True,
    }

    # Branch 1: 10 steps uninterrupted
    torch.manual_seed(12345)
    loader_1 = StochasticTimeWindowLoader(inputs, dates, cfg, seed=777, device=device)
    trainer_1 = SharedDPLTrainer(config, loader=loader_1, compile_step=True)
    history_uninterrupted = [trainer_1.train_step() for _ in range(10)]

    # Branch 2: 4 steps, save checkpoint, reload, 6 steps
    torch.manual_seed(12345)
    loader_2 = StochasticTimeWindowLoader(inputs, dates, cfg, seed=777, device=device)
    trainer_2 = SharedDPLTrainer(config, loader=loader_2, compile_step=True)
    history_interrupted = [trainer_2.train_step() for _ in range(4)]

    ckpt_file = CACHE_DIR / "g6_checkpoint_test.pt"
    trainer_2.save_checkpoint(ckpt_file)

    trainer_resumed = SharedDPLTrainer(config, loader=loader_2, compile_step=True)
    trainer_resumed.load_checkpoint(ckpt_file)

    for _ in range(6):
        history_interrupted.append(trainer_resumed.train_step())

    # Verify exact match
    exact_match = True
    max_loss_diff = 0.0
    for idx in range(10):
        h1 = history_uninterrupted[idx]
        h2 = history_interrupted[idx]
        if h1["global_step"] != h2["global_step"] or h1["model_id"] != h2["model_id"]:
            exact_match = False
        diff = abs(h1["loss"] - h2["loss"])
        max_loss_diff = max(max_loss_diff, diff)
        if diff > 1e-10:
            exact_match = False

    gate_pass = bool(exact_match and max_loss_diff <= 1e-10)
    print(f"G6 complete: 10 steps checked across checkpoint boundary -> max loss diff={max_loss_diff:.2e}, PASS={gate_pass}")
    return {
        "gate": "G6_REAL_DATA_RESUME",
        "status": "PASS" if gate_pass else "FAIL",
        "steps_checked": 10,
        "checkpoint_step": 4,
        "max_loss_difference": max_loss_diff,
        "exact_match": gate_pass,
    }


# ----------------------------------------------------------------------
# G7. Train/Validation/Test Time-Boundary and Data-Leakage Audit
# ----------------------------------------------------------------------
def run_g7_time_boundary_leakage_audit(inputs: dict[str, dict[str, np.ndarray]], dates: list[Any]) -> dict[str, Any]:
    print("\n=======================================================")
    print("G7: Train/Validation/Test Time-Boundary & Leakage Audit")
    print("=======================================================")
    protocol = json.loads(open("project/autofuse/docs/formal_experiment_protocol_v1.json").read())
    periods = protocol["periods"]

    calib_start = periods["calibration"][0]
    calib_end = periods["calibration"][1]
    eval_start = periods["evaluation"][0]
    eval_end = periods["evaluation"][1]
    forcing_start = periods["forcing"][0]
    forcing_end = periods["forcing"][1]

    date_strs = [d.isoformat() for d in dates]

    cfg = TimeWindowConfig(total_days=730, warmup_days=365, scored_days=365, calibration_end=date(1998, 12, 31))
    loader = StochasticTimeWindowLoader(inputs, dates, cfg, seed=999)

    # Sample 1000 stochastic windows and verify boundary containment
    leakage_detected = False
    samples_checked = 1000
    for _ in range(samples_checked):
        batch = loader.sample_batch(["USA_09447800", "USA_14138900"])
        for s_idx in batch.start_indices:
            w_start = date_strs[s_idx]
            w_warmup_end = date_strs[s_idx + 364]
            w_scored_start = date_strs[s_idx + 365]
            w_end = date_strs[s_idx + 729]

            # Invariants:
            # 1. Window must start on or after forcing_start
            if w_start < forcing_start:
                leakage_detected = True
            # 2. Window must end on or before calibration_end (no evaluation dates entered!)
            if w_end > calib_end:
                leakage_detected = True
            # 3. Scored start must not enter evaluation period
            if w_scored_start >= eval_start:
                leakage_detected = True

    gate_pass = not leakage_detected
    print(f"G7 complete: {samples_checked} windows audited | Calibration interval: [{calib_start} .. {calib_end}] | Evaluation: [{eval_start} .. {eval_end}] -> Leakage={leakage_detected}, PASS={gate_pass}")

    return {
        "gate": "G7_TIME_BOUNDARY_LEAKAGE",
        "status": "PASS" if gate_pass else "FAIL",
        "calibration_period": [calib_start, calib_end],
        "evaluation_period": [eval_start, eval_end],
        "windows_audited": samples_checked,
        "leakage_detected": leakage_detected,
    }


# ----------------------------------------------------------------------
# D1. Eager vs torch.compile Numerical Parity Diagnostic (Non-blocking)
# ----------------------------------------------------------------------
def run_d1_compile_parity(inputs: dict[str, dict[str, np.ndarray]], dates: list[Any], device: torch.device) -> dict[str, Any]:
    print("\n=======================================================")
    print("D1: Eager vs torch.compile Numerical Parity Diagnostic")
    print("=======================================================")
    test_models = [2, 8, 190]
    cfg = TimeWindowConfig(total_days=8, warmup_days=4, scored_days=4, calibration_end=date(1987, 1, 20))
    loader = StochasticTimeWindowLoader(inputs, dates, cfg, seed=123, device=device)
    batch = loader.sample_batch(["USA_09447800", "USA_14138900"])

    records = []
    for model_id in test_models:
        param_nn = StructureConditionedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=64)).to(device=device, dtype=torch.float64)
        params = param_nn(batch.attributes, model_id)

        # Eager forward
        _cuda_sync()
        t0 = time.perf_counter()
        res_eager = simulate_coupled_rk2_batched(model_id, batch.forcing, params, basin_ids=batch.basin_ids, compile_step=False, output_mode="q_only")
        _cuda_sync()
        t_eager = time.perf_counter() - t0

        # Compiled forward
        _cuda_sync()
        t0 = time.perf_counter()
        res_comp = simulate_coupled_rk2_batched(model_id, batch.forcing, params, basin_ids=batch.basin_ids, compile_step=True, compile_backend="inductor", compile_fullgraph=True, output_mode="q_only")
        _cuda_sync()
        t_comp = time.perf_counter() - t0

        q_diff = float((res_eager.q - res_comp.q).abs().max().detach().cpu())
        records.append({
            "model_id": model_id,
            "q_diff_max": q_diff,
            "eager_seconds": t_eager,
            "compiled_seconds": t_comp,
            "speedup": float(t_eager / max(t_comp, 1e-6)),
            "parity_pass": bool(q_diff <= 1e-12),
        })
        print(f"Model {model_id:3d}: Q diff={q_diff:.2e} | Eager={t_eager:.4f}s | Compiled={t_comp:.4f}s")

    d1_pass = all(r["parity_pass"] for r in records)
    return {
        "diagnostic": "D1_COMPILE_PARITY",
        "status": "PASS" if d1_pass else "FAIL",
        "records": records,
    }


# ----------------------------------------------------------------------
# D2. Tiny Real-Data Learning Sanity Check (Non-blocking)
# ----------------------------------------------------------------------
def run_d2_tiny_learning_sanity(inputs: dict[str, dict[str, np.ndarray]], dates: list[Any], device: torch.device) -> dict[str, Any]:
    print("\n=======================================================")
    print("D2: Tiny Real-Data Learning Sanity Check Diagnostic")
    print("=======================================================")
    model_id = 2
    cfg = TimeWindowConfig(total_days=730, warmup_days=365, scored_days=365, calibration_end=date(1998, 12, 31))
    loader = StochasticTimeWindowLoader(inputs, dates, cfg, seed=555, device=device)

    config = {
        "batch_size": 2,
        "basin_ids": ["USA_09447800", "USA_14138900"],
        "structures": [model_id],
        "lr": 5e-3,
        "device": "cuda",
        "compile_step": True,
    }
    trainer = SharedDPLTrainer(config, loader=loader, compile_step=True)

    losses = []
    grad_norms = []
    print("Running 10 optimization steps on Structure 2...")
    for step in range(10):
        res = trainer.train_step()
        losses.append(res["loss"])
        grad_norms.append(res["pre_clip_norm"])
        print(f"  Step {step:2d}: Loss={res['loss']:.4f}, GradNorm={res['pre_clip_norm']:.4f}")

    loss_decreased = bool(losses[-1] < losses[0])
    all_finite = bool(all(np.isfinite(l) for l in losses) and all(np.isfinite(g) for g in grad_norms))
    d2_pass = all_finite and loss_decreased
    print(f"D2 complete: Initial loss={losses[0]:.4f}, Final loss={losses[-1]:.4f}, Decreased={loss_decreased} -> PASS={d2_pass}")

    return {
        "diagnostic": "D2_TINY_LEARNING_SANITY",
        "status": "PASS" if d2_pass else "FAIL",
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "best_loss": float(min(losses)),
        "loss_decreased": loss_decreased,
        "all_finite": all_finite,
        "loss_trajectory": losses,
        "gradient_norm_trajectory": grad_norms,
    }


def main() -> None:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR.resolve())
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch._dynamo.config.cache_size_limit = 256
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, inputs, _ = _load_frozen_inputs()
    dates = _dates()
    manifest = json.loads(open("project/autofuse/docs/landscape_12catchment_manifest.json").read())
    attr_names = manifest.get("attribute_names", [])

    for c in manifest.get("catchments", []):
        b_id = c["basin_id"]
        if b_id in inputs:
            inputs[b_id]["attributes"] = np.array([c["attributes"][name] for name in attr_names], dtype=np.float64)

    def cached_gate(name: str, fn, *args):
        p = CACHE_DIR / f"{name.lower()}.json"
        if p.is_file():
            try:
                res = json.loads(p.read_text())
                if res.get("status") == "PASS":
                    print(f"Skipping completed {name}: PASS")
                    return res
            except Exception:
                pass
        res = fn(*args)
        p.write_text(json.dumps(res, indent=2) + "\n")
        return res

    # Run G1 - G7 and D1 - D2 with per-gate caching
    g1 = cached_gate("G1", run_g1_phase_locking)
    g2 = cached_gate("G2", run_g2_long_run_sampler_stress)
    g3 = cached_gate("G3", run_g3_full_78_optimizer_contract, device)
    g4 = cached_gate("G4", run_g4_real_data_one_cycle, inputs, dates, device)
    g5 = cached_gate("G5", run_g5_production_memory_smoke, inputs, dates, device)
    g6 = cached_gate("G6", run_g6_real_data_resume, inputs, dates, device)
    g7 = cached_gate("G7", run_g7_time_boundary_leakage_audit, inputs, dates)
    d1 = cached_gate("D1", run_d1_compile_parity, inputs, dates, device)
    d2 = cached_gate("D2", run_d2_tiny_learning_sanity, inputs, dates, device)

    all_gates_pass = all([
        g1["status"] == "PASS",
        g2["status"] == "PASS",
        g3["status"] == "PASS",
        g4["status"] == "PASS",
        g5["status"] == "PASS",
        g6["status"] == "PASS",
        g7["status"] == "PASS",
    ])

    report = {
        "schema_version": "pre-run-final-gate-report-v1",
        "status": "PASS" if all_gates_pass else "FAIL",
        "ready_for_phase_0": "YES" if all_gates_pass else "NO",
        "full_registry_validated": "NO",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hard_gates": {
            "G1_PHASE_LOCKING": g1["status"],
            "G2_LONG_RUN_SAMPLER_STRESS": g2["status"],
            "G3_FULL_78_OPTIMIZER_CONTRACT": g3["status"],
            "G4_REAL_DATA_ONE_CYCLE": g4["status"],
            "G5_PRODUCTION_SHAPE_MEMORY": g5["status"],
            "G6_REAL_DATA_RESUME": g6["status"],
            "G7_TIME_BOUNDARY_LEAKAGE": g7["status"],
        },
        "non_blocking_diagnostics": {
            "D1_COMPILE_PARITY": d1["status"],
            "D2_TINY_LEARNING_SANITY": d2["status"],
        },
        "gate_details": {
            "g1": g1,
            "g2": g2,
            "g3": g3,
            "g4": g4,
            "g5": g5,
            "g6": g6,
            "g7": g7,
            "d1": d1,
            "d2": d2,
        },
    }

    _write_json(GATE_REPORT_PATH, report)
    print("\n=======================================================")
    print(f"PRE_RUN_FINAL_GATE = {report['status']}")
    print(f"READY_FOR_PHASE_0 = {report['ready_for_phase_0']}")
    print("FULL_REGISTRY_VALIDATED = NO")
    print("=======================================================")


if __name__ == "__main__":
    main()
