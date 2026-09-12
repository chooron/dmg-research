"""AutoFuse Shared Parameter dPL Gradient Probe Runner.

Pre-registered empirical gradient conflict probe on the 78 benchmark structures
of FUSE under pure shared parameterization (theta_c = g_phi(X_c)).

Scientific Protocol: project/autofuse/docs/shared_parameter_gradient_probe_protocol.json
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from dfuse import PARAMETER_NAMES, enumerate_structures, get_structure, simulate_coupled_rk2_batched
from project.autofuse.dpl import DPLConfig, PureSharedParameterizer
from project.autofuse.loader import StochasticTimeWindowLoader, TimeWindowConfig
from project.autofuse.metrics import kgecomp_batched
from project.autofuse.parameter_contract import get_parameter_contract
from project.autofuse.samplers import GlobalBasinSampler, ShuffledStructureSampler
from project.autofuse.torch_fuse_78_long_horizon_smoke import _dates

# CPU threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
PROTOCOL_PATH = DOCS / "shared_parameter_gradient_probe_protocol.json"
DIAG_BATCH_PATH = DOCS / "shared_parameter_gradient_probe_diagnostic_batch.json"
OWNERSHIP_PATH = DOCS / "fuse_parameter_ownership_78.json"
DEFAULT_RUN_DIR = ROOT / "project/autofuse/runs/shared_parameter_gradient_probe"
DEFAULT_CACHE_DIR = ROOT / "project/autofuse/.cache/torch-fuse-gradient-probe"


def _model_param_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_12catchment_inputs(device: torch.device, dtype: torch.dtype) -> tuple[dict[str, dict[str, Any]], list[date]]:
    manifest_path = DOCS / "landscape_12catchment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    attr_names = manifest.get("attribute_names", [])
    inputs_dir = DOCS / "landscape_inputs"

    dates = _dates()
    inputs = {}
    for c in manifest["catchments"]:
        b_id = c["basin_id"]
        hru_id = c["hru_id"]
        npz_path = inputs_dir / f"{hru_id:08d}.npz"
        data = np.load(npz_path)
        inputs[b_id] = {
            "ppt": data["ppt"].astype(np.float64),
            "pet": data["pet"].astype(np.float64),
            "temp": data["temp"].astype(np.float64),
            "q_obs": data["q_obs"].astype(np.float64),
            "attributes": np.array([c["attributes"][name] for name in attr_names], dtype=np.float64),
        }
    return inputs, dates


def pack_diagnostic_batch(inputs: dict[str, dict[str, Any]], dates: list[date], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    diag_doc = json.loads(DIAG_BATCH_PATH.read_text())
    windows = diag_doc["windows"]
    total_days = diag_doc["window_days"]
    warmup_days = diag_doc["warmup_days"]

    forcing_list = []
    q_scored_list = []
    attr_list = []
    basin_ids = []

    for w in windows:
        b_id = w["basin_id"]
        s_idx = w["start_idx"]
        d = inputs[b_id]

        ppt_win = d["ppt"][s_idx:s_idx + total_days]
        pet_win = d["pet"][s_idx:s_idx + total_days]
        temp_win = d["temp"][s_idx:s_idx + total_days]
        q_win = d["q_obs"][s_idx + warmup_days:s_idx + total_days]

        f_win = np.stack([ppt_win, pet_win, temp_win], axis=-1)
        forcing_list.append(f_win)
        q_scored_list.append(q_win)
        attr_list.append(d["attributes"])
        basin_ids.append(b_id)

    forcing_tensor = torch.tensor(np.stack(forcing_list, axis=0), device=device, dtype=dtype)
    q_scored_tensor = torch.tensor(np.stack(q_scored_list, axis=0), device=device, dtype=dtype)
    attr_tensor = torch.tensor(np.stack(attr_list, axis=0), device=device, dtype=dtype)

    return {
        "forcing": forcing_tensor,
        "q_scored": q_scored_tensor,
        "attributes": attr_tensor,
        "basin_ids": basin_ids,
        "warmup_days": warmup_days,
        "batch_size": len(windows),
    }


def run_diagnostic_sweep(
    model: PureSharedParameterizer,
    diag_batch: dict[str, Any],
    ckpt_name: str,
    run_dir: Path,
    structures: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Execute matched gradient diagnostic sweep over 78 structures without parameter updates."""
    print(f"\n>>> Running Matched Diagnostic Sweep at Checkpoint: {ckpt_name} <<<")
    raw_grad_dir = run_dir / "raw_gradients"
    raw_grad_dir.mkdir(parents=True, exist_ok=True)
    diag_log_file = run_dir / "diagnostic_sweeps.jsonl"

    pre_hash = _model_param_hash(model)
    model.eval()

    B = diag_batch["batch_size"]
    n_params = len(PARAMETER_NAMES)
    n_structs = len(structures)

    # Tensor to hold raw normalized pre-transform gradients: (78, 100, 37)
    G_u = torch.zeros(n_structs, B, n_params, device="cpu", dtype=torch.float64)
    # Tensor to hold physical space gradients: (78, 100, 37)
    G_theta = torch.zeros(n_structs, B, n_params, device="cpu", dtype=torch.float64)
    structure_records = []
    completed_indices = set()

    partial_file = raw_grad_dir / f"partial_{ckpt_name}.pt"
    if partial_file.is_file():
        try:
            p_data = torch.load(partial_file, map_location="cpu", weights_only=False)
            G_u = p_data["G_u"]
            G_theta = p_data["G_theta"]
            structure_records = p_data["structure_records"]
            completed_indices = set(p_data["completed_indices"])
            print(f"Resuming diagnostic sweep {ckpt_name} from {len(completed_indices)}/{n_structs} completed structures")
        except Exception:
            pass

    t0_sweep = time.perf_counter()

    lower = model.lower.to(device=device, dtype=dtype)
    upper = model.upper.to(device=device, dtype=dtype)
    defaults = model.defaults.to(device=device, dtype=dtype)

    for s_idx, struct_id in enumerate(structures):
        if s_idx in completed_indices:
            continue

        model.zero_grad()
        contract = get_parameter_contract(struct_id)
        active_indices_set = set(contract.active_indices)

        # Forward trunk & heads
        hidden = model.trunk(diag_batch["attributes"])
        u_list = []
        theta_list = []

        for j in range(n_params):
            if j in active_indices_set:
                u_j = model.heads[j](hidden).squeeze(-1)
                u_j.retain_grad()
                theta_j = lower[j] + torch.sigmoid(u_j) * (upper[j] - lower[j])
                theta_j.retain_grad()
                u_list.append(u_j)
                theta_list.append(theta_j)
            else:
                u_list.append(None)
                theta_list.append(defaults[j].expand(B))

        theta_active = torch.stack(theta_list, dim=1)

        # Simulate FUSE
        out = simulate_coupled_rk2_batched(struct_id, diag_batch["forcing"], theta_active, basin_ids=diag_batch["basin_ids"], compile_step=True)
        q_scored = out.q[:, diag_batch["warmup_days"]:]

        kge_scores = kgecomp_batched(q_scored, diag_batch["q_scored"])
        loss = 1.0 - kge_scores.mean()

        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Record gradients
        for j in range(n_params):
            if j in active_indices_set and u_list[j].grad is not None:
                G_u[s_idx, :, j] = u_list[j].grad.detach().cpu()
                if theta_list[j].grad is not None:
                    G_theta[s_idx, :, j] = theta_list[j].grad.detach().cpu()

        kge_np = kge_scores.detach().cpu().numpy()
        rec = {
            "checkpoint": ckpt_name,
            "structure_id": struct_id,
            "loss": float(loss.item()),
            "kge_mean": float(np.nanmean(kge_np)),
            "kge_median": float(np.nanmedian(kge_np)),
            "active_parameter_count": len(contract.active_parameters),
            "is_finite": bool(np.isfinite(loss.item())),
        }
        structure_records.append(rec)
        completed_indices.add(s_idx)

        # Save partial progress
        torch.save({
            "G_u": G_u,
            "G_theta": G_theta,
            "structure_records": structure_records,
            "completed_indices": list(completed_indices),
        }, partial_file)

        with open(diag_log_file, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if (s_idx + 1) % 15 == 0 or s_idx == n_structs - 1:
            print(f"  Sweep {ckpt_name} [{s_idx+1:2d}/{n_structs:2d}] Structure {struct_id:3d}: Loss={loss.item():.4f}, KGE={rec['kge_mean']:.3f}")

    post_hash = _model_param_hash(model)
    assert pre_hash == post_hash, f"CRITICAL: Model parameters mutated during diagnostic sweep {ckpt_name}!"

    # Clean up partial file on full completion
    if partial_file.is_file():
        partial_file.unlink(missing_ok=True)

    # Save raw tensors
    grad_file = raw_grad_dir / f"gradients_{ckpt_name}.pt"
    torch.save({
        "checkpoint": ckpt_name,
        "structures": structures,
        "G_u": G_u,
        "G_theta": G_theta,
        "parameter_names": PARAMETER_NAMES,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }, grad_file)

    grad_hash = hashlib.sha256(grad_file.read_bytes()).hexdigest()
    t_sweep = time.perf_counter() - t0_sweep
    print(f"Diagnostic sweep {ckpt_name} complete in {t_sweep:.2f}s. Saved {grad_file.name} (SHA-256: {grad_hash[:16]}...)")

    return {
        "checkpoint": ckpt_name,
        "wall_time_seconds": t_sweep,
        "raw_gradient_file": str(grad_file),
        "raw_gradient_sha256": grad_hash,
        "parameter_hash_invariant_verified": True,
        "structure_records": structure_records,
    }


def compute_conflict_analysis(
    run_dir: Path,
    structures: list[int],
    checkpoints: list[str] = ["D0", "D1", "D3", "D5"],
) -> dict[str, Any]:
    """Calculate pre-registered magnitude-weighted sign conflict index and determine A/B/C gate."""
    print("\n=== Computing Pre-Registered Gradient Conflict & Signal Support Analysis ===")
    raw_grad_dir = run_dir / "raw_gradients"
    ownership_doc = json.loads(OWNERSHIP_PATH.read_text())["parameters"]

    struct_to_idx = {s: i for i, s in enumerate(structures)}
    n_params = len(PARAMETER_NAMES)
    B = 100

    # 1. Determine empirical noise floor epsilon_grad from inactive structure entries
    inactive_grads = []
    loaded_tensors = {}

    for ckpt in checkpoints:
        p_file = raw_grad_dir / f"gradients_{ckpt}.pt"
        if not p_file.is_file():
            raise FileNotFoundError(f"Missing gradient tensor file: {p_file}")
        data = torch.load(p_file, weights_only=False)
        G_u = data["G_u"].numpy() # (78, 100, 37)
        loaded_tensors[ckpt] = G_u

        for j, name in enumerate(PARAMETER_NAMES):
            inactive_structs = ownership_doc[name]["inactive_structure_ids"]
            for s_id in inactive_structs:
                if s_id in struct_to_idx:
                    s_i = struct_to_idx[s_id]
                    inactive_grads.extend(np.abs(G_u[s_i, :, j]))

    max_inactive = float(np.max(inactive_grads)) if len(inactive_grads) > 0 else 0.0
    epsilon_grad = max(1e-12, 10.0 * max_inactive)
    print(f"Empirical Gradient Noise Floor: epsilon_grad = {epsilon_grad:.2e} (max observed inactive={max_inactive:.2e})")

    # 2. Per-parameter and per-checkpoint conflict metrics
    parameter_metrics = {}

    for j, name in enumerate(PARAMETER_NAMES):
        p_info = ownership_doc[name]
        active_structs = p_info["active_structure_ids"]
        n_active = len(active_structs)

        if n_active <= 1:
            parameter_metrics[name] = {
                "canonical_name": name,
                "canonical_index": j,
                "active_structures_count": n_active,
                "evaluable": False,
                "classification": "NOT_SHARED_OR_INACTIVE",
            }
            continue

        active_indices = [struct_to_idx[s_id] for s_id in active_structs if s_id in struct_to_idx]

        ckpt_stats = {}
        is_persistent = True

        for ckpt in checkpoints:
            G_u = loaded_tensors[ckpt] # (78, 100, 37)
            # Active slices for param j: (N_active, B)
            G_active = G_u[active_indices, :, j] # (N_active, 100)

            # Signal support: fraction of (s, b) with |g| > epsilon_grad
            sig_mask = np.abs(G_active) > epsilon_grad
            signal_support = float(np.mean(sig_mask))

            # Magnitude statistics
            abs_g = np.abs(G_active)
            med_abs_g = float(np.median(abs_g))
            max_abs_g = float(np.max(abs_g))

            # Sign conflict index C_{b,j} per sample b
            P_b = np.sum(np.maximum(G_active, 0.0), axis=0) # (B,)
            N_b = np.sum(np.maximum(-G_active, 0.0), axis=0) # (B,)
            denom = P_b + N_b
            C_b = np.where(denom > epsilon_grad, 2.0 * np.minimum(P_b, N_b) / np.maximum(denom, 1e-12), 0.0) # (B,)

            # Structure dominance: N_eff per sample b
            sum_g = np.sum(abs_g, axis=0) # (B,)
            sum_sq_g = np.sum(G_active ** 2, axis=0) # (B,)
            N_eff_b = np.where(sum_sq_g > 1e-12, (sum_g ** 2) / np.maximum(sum_sq_g, 1e-12), 1.0) # (B,)
            norm_N_eff_b = N_eff_b / max(n_active, 1)

            # Filter C_b on active samples where at least one structure has signal
            active_b_mask = denom > epsilon_grad
            if np.any(active_b_mask):
                C_active = C_b[active_b_mask]
                norm_N_eff_active = norm_N_eff_b[active_b_mask]
                med_C = float(np.median(C_active))
                iqr_C = float(np.percentile(C_active, 75) - np.percentile(C_active, 25))
                med_norm_N_eff = float(np.median(norm_N_eff_active))
            else:
                med_C = 0.0
                iqr_C = 0.0
                med_norm_N_eff = 0.0

            has_signal = signal_support >= 0.25
            strong_conflict_at_ckpt = bool(has_signal and med_C >= 0.30 and med_norm_N_eff >= 0.20)

            if ckpt in ["D1", "D3", "D5"]:
                if not strong_conflict_at_ckpt:
                    is_persistent = False

            ckpt_stats[ckpt] = {
                "signal_support": signal_support,
                "sufficient_signal": has_signal,
                "median_abs_grad": med_abs_g,
                "max_abs_grad": max_abs_g,
                "median_conflict_C": med_C,
                "iqr_conflict_C": iqr_C,
                "median_norm_N_eff": med_norm_N_eff,
                "strong_conflict": strong_conflict_at_ckpt,
            }

        # Check for insufficient signal across post-training checkpoints
        post_signals = [ckpt_stats[c]["sufficient_signal"] for c in ["D1", "D3", "D5"]]
        low_signal = not all(post_signals)

        if low_signal:
            status = "INSUFFICIENT_SIGNAL"
        elif is_persistent:
            status = "PERSISTENT_STRONG_CONFLICT"
        else:
            status = "CONSISTENT_OR_NEGLIGIBLE_CONFLICT"

        parameter_metrics[name] = {
            "canonical_name": name,
            "canonical_index": j,
            "active_structures_count": n_active,
            "evaluable": True,
            "status": status,
            "checkpoints": ckpt_stats,
        }

    # 3. Gate decision
    evaluable_params = [p for p in parameter_metrics.values() if p.get("evaluable", False)]
    n_evaluable = len(evaluable_params)
    insufficient_signal_count = len([p for p in evaluable_params if p["status"] == "INSUFFICIENT_SIGNAL"])
    persistent_conflict_count = len([p for p in evaluable_params if p["status"] == "PERSISTENT_STRONG_CONFLICT"])

    conflict_fraction = persistent_conflict_count / max(n_evaluable, 1)
    insufficient_fraction = insufficient_signal_count / max(n_evaluable, 1)

    if insufficient_fraction > 0.30:
        gate = "INCONCLUSIVE_LOW_SIGNAL"
        action = "Improve diagnostic process activation before modifying architecture; do not change to structure-conditioned."
    elif conflict_fraction <= 0.10:
        gate = "A"
        action = "Pure sharing supported for next stage: keep pure shared + active mask. Quantify sharing cost with independent calibration reference later."
    elif conflict_fraction <= 0.30:
        gate = "B"
        action = "Localized conflict: investigate flagged parameters for process-option interactions. Avoid global structure-conditioned parameterization."
    else:
        gate = "C"
        action = "Broad conflict: consider structure-conditioned parameterization."

    print(f"\nConflict Analysis Summary:")
    print(f"  Evaluable Canonical Parameters: {n_evaluable}")
    print(f"  Insufficient Signal:             {insufficient_signal_count} ({insufficient_fraction*100:.1f}%)")
    print(f"  Persistent Strong Conflict:      {persistent_conflict_count} ({conflict_fraction*100:.1f}%)")
    print(f"  -> Conflict Fraction:            {conflict_fraction:.4f}")
    print(f"  -> FINAL DECISION GATE:          GATE {gate}")
    print(f"  -> Recommended Next Action:      {action}")

    # 4. Generate CSV Table
    csv_file = DOCS / "shared_parameter_effective_signal.csv"
    with open(csv_file, "w") as f:
        f.write("parameter,active_structures,signal_support_D1,signal_support_D3,signal_support_D5,median_abs_grad_D5,median_conflict_C_D5,norm_N_eff_D5,status\n")
        for p in evaluable_params:
            name = p["canonical_name"]
            c1 = p["checkpoints"]["D1"]
            c3 = p["checkpoints"]["D3"]
            c5 = p["checkpoints"]["D5"]
            f.write(f"{name},{p['active_structures_count']},{c1['signal_support']:.3f},{c3['signal_support']:.3f},{c5['signal_support']:.3f},{c5['median_abs_grad']:.4e},{c5['median_conflict_C']:.3f},{c5['median_norm_N_eff']:.3f},{p['status']}\n")
    print(f"Saved effective signal table to {csv_file}")

    summary = {
        "schema_version": "shared-parameter-gradient-probe-summary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "noise_floor_epsilon_grad": epsilon_grad,
        "evaluable_parameter_count": n_evaluable,
        "insufficient_signal_count": insufficient_signal_count,
        "insufficient_signal_fraction": insufficient_fraction,
        "persistent_strong_conflict_count": persistent_conflict_count,
        "conflict_fraction": conflict_fraction,
        "final_gate": gate,
        "recommended_action": action,
        "parameters": parameter_metrics,
    }
    sum_json = DOCS / "shared_parameter_gradient_probe_summary.json"
    sum_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved summary JSON to {sum_json}")

    return summary


def run_probe(
    cycles: int = 5,
    seed: int = 20260901,
    lr: float = 1e-3,
    run_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir or DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    print(f"=======================================================")
    print(f"AutoFuse Shared Parameter dPL Gradient Probe")
    print(f"Device={device}, Dtype={dtype}, Seed={seed}, Cycles={cycles} (390 steps)")
    print(f"Run Dir: {run_dir}")
    print(f"=======================================================")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    inputs, dates = load_12catchment_inputs(device, dtype)
    diag_batch = pack_diagnostic_batch(inputs, dates, device, dtype)

    structures = [s.model_id for s in enumerate_structures()]
    n_structs = len(structures)

    # Initialize PureSharedParameterizer (FP64)
    model = PureSharedParameterizer(DPLConfig(attribute_dim=35, hidden_dim=64)).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Setup time window loader for training
    loader_cfg = TimeWindowConfig(total_days=730, warmup_days=365, scored_days=365, calibration_end=date(1998, 12, 31))
    train_loader = StochasticTimeWindowLoader(inputs, dates, loader_cfg, seed=seed, device=device)
    structure_sampler = ShuffledStructureSampler(structures, seed=seed)
    target_12_basins = list(inputs.keys())
    basin_population = [target_12_basins[i % len(target_12_basins)] for i in range(100)]

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_log_file = run_dir / "training_steps.jsonl"

    raw_grad_dir = run_dir / "raw_gradients"
    raw_grad_dir.mkdir(parents=True, exist_ok=True)

    # 1. D0 Diagnostic Sweep (before any training)
    if not (raw_grad_dir / "gradients_D0.pt").is_file():
        print("\n--- Executing Baseline D0 Diagnostic Sweep ---")
        torch.save(model.state_dict(), ckpt_dir / "model_D0.pt")
        run_diagnostic_sweep(model, diag_batch, "D0", run_dir, structures, device, dtype)
    else:
        print("Skipping completed D0 Diagnostic Sweep.")

    # 2. Training Loop: 5 Cycles
    # Step-level resume discovery
    completed_global_steps = set()
    if train_log_file.is_file():
        for line in train_log_file.read_text().split("\n"):
            if line.strip():
                try:
                    rec = json.loads(line)
                    completed_global_steps.add(int(rec["global_step"]))
                except Exception:
                    pass
    if completed_global_steps:
        print(f"Found {len(completed_global_steps)} previously completed training steps (latest step: {max(completed_global_steps)})")

    global_step = 0
    t0_train = time.perf_counter()
    for cycle in range(1, cycles + 1):
        cycle_ckpt = ckpt_dir / f"model_cycle_{cycle}.pt"
        opt_ckpt = ckpt_dir / f"optimizer_cycle_{cycle}.pt"
        diag_name = f"D{cycle}" if cycle in (1, 3, 5) else None
        diag_done = (raw_grad_dir / f"gradients_{diag_name}.pt").is_file() if diag_name else True

        if cycle_ckpt.is_file() and opt_ckpt.is_file() and diag_done:
            print(f"Skipping completed Cycle {cycle}/{cycles} and its diagnostics.")
            model.load_state_dict(torch.load(cycle_ckpt, map_location=device, weights_only=False))
            optimizer.load_state_dict(torch.load(opt_ckpt, map_location=device, weights_only=False))
            for _ in range(n_structs):
                structure_sampler.next_structure()
                train_loader.sample_batch(basin_population)
                global_step += 1
            continue

        print(f"\n=== Starting Structure Cycle {cycle}/{cycles} ===")
        for s_idx in range(n_structs):
            struct_id = structure_sampler.next_structure()
            global_step += 1
            if global_step in completed_global_steps:
                train_loader.sample_batch(basin_population)
                continue
            t0_step = time.perf_counter()

            # Sample B=100 batch from 12 basins
            batch = train_loader.sample_batch(basin_population)

            # Model forward
            model.train()
            optimizer.zero_grad()

            params_active = model(batch.attributes, struct_id)
            out = simulate_coupled_rk2_batched(struct_id, batch.forcing, params_active, basin_ids=batch.basin_ids, compile_step=True)
            if out.q.shape[1] < loader_cfg.total_days or out.monitoring.get("stopped_on_failure"):
                optimizer.zero_grad()
                raise RuntimeError(f"Simulation stopped early or failed on structure {struct_id} at step {global_step}")
            q_scored = out.q[:, loader_cfg.warmup_days:]

            kge_scores = kgecomp_batched(q_scored, batch.target_scored)
            loss = 1.0 - kge_scores.mean()

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                raise RuntimeError(f"Non-finite loss encountered at step {global_step} (Model {struct_id}): {loss.item()}")

            loss.backward()
            pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            if not np.isfinite(pre_clip_norm):
                print(f"  Warning: Non-finite pre-clip gradient norm at step {global_step} (Model {struct_id}): {pre_clip_norm}. Skipping optimizer step.")
                optimizer.zero_grad()
                grad_skipped = True
            else:
                optimizer.step()
                grad_skipped = False

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt_step = time.perf_counter() - t0_step

            step_rec = {
                "cycle": cycle,
                "cycle_step": s_idx + 1,
                "global_step": global_step,
                "structure_id": struct_id,
                "loss": float(loss.item()),
                "pre_clip_norm": pre_clip_norm if np.isfinite(pre_clip_norm) else -1.0,
                "grad_skipped": grad_skipped,
                "wall_time_seconds": dt_step,
            }
            with open(train_log_file, "a") as f:
                f.write(json.dumps(step_rec) + "\n")

            if (s_idx + 1) % 15 == 0 or s_idx == n_structs - 1:
                print(f"  Cycle {cycle} Step [{s_idx+1:2d}/{n_structs:2d}] (Global {global_step:3d}, Model {struct_id:3d}): Loss={loss.item():.4f}, Grad={pre_clip_norm:.4f}, Time={dt_step:.2f}s")

        # Save cycle checkpoint
        torch.save(model.state_dict(), cycle_ckpt)
        torch.save(optimizer.state_dict(), opt_ckpt)

        # Trigger diagnostic sweeps
        if cycle == 1 and not (raw_grad_dir / "gradients_D1.pt").is_file():
            run_diagnostic_sweep(model, diag_batch, "D1", run_dir, structures, device, dtype)
        elif cycle == 3 and not (raw_grad_dir / "gradients_D3.pt").is_file():
            run_diagnostic_sweep(model, diag_batch, "D3", run_dir, structures, device, dtype)
        elif cycle == 5 and not (raw_grad_dir / "gradients_D5.pt").is_file():
            run_diagnostic_sweep(model, diag_batch, "D5", run_dir, structures, device, dtype)
    t_train = time.perf_counter() - t0_train
    print(f"\nTraining finished: 5 cycles (390 steps) in {t_train:.2f}s ({t_train/60:.1f} minutes).")

    # 3. Post-run conflict analysis
    summary = compute_conflict_analysis(run_dir, structures, ["D0", "D1", "D3", "D5"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoFuse Shared Parameter dPL Gradient Probe")
    parser.add_argument("--cycles", type=int, default=5, help="Number of structure cycles (default: 5)")
    parser.add_argument("--seed", type=int, default=20260901, help="Random seed (default: 20260901)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR), help="Run output directory")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Inductor cache directory")

    args = parser.parse_args()
    run_probe(
        cycles=args.cycles,
        seed=args.seed,
        lr=args.lr,
        run_dir=Path(args.run_dir),
        cache_dir=Path(args.cache_dir),
    )


if __name__ == "__main__":
    main()
