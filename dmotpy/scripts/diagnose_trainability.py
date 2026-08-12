"""
全模型参数梯度可训练性诊断
===============================
对 dmotpy/models/core/ 下 36 个模型执行统一口径的可训练性诊断，
产出一张全库风险等级表。

不修改任何模型代码，仅做诊断。
"""
from __future__ import annotations

import sys, os, time, inspect, csv
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
DMOTPY_DIR = SCRIPT_DIR.parent
os.chdir(DMOTPY_DIR)
sys.path.insert(0, str(DMOTPY_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from tests.core_model_registry import CORE_MODEL_REGISTRY, CoreModelEntry

OUTPUT_DIR = DMOTPY_DIR / "validation_results" / "trainability_diagnosis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(1)
SEED = 202607022
GRAD_ZERO_THRESH = 1e-12
NEARZERO = 1.0e-6

ALL_MODELS = sorted([n for n, e in CORE_MODEL_REGISTRY.items() if e.enabled])

# Known fragile models for spotlight review
PRIORITY_SPOTLIGHT = {"gsfb", "hbv96", "plateau", "smar", "modhydrolog"}

# ====================================================================
# 1. FIXED SAMPLE CONSTRUCTION
# ====================================================================

def make_unified_sample(n_basins=8, n_timesteps=120, dtype=torch.float32, device="cpu"):
    """Create a unified forcing + parameter sample shared across all models."""
    gen = torch.Generator(device=device).manual_seed(SEED)
    forcing = torch.zeros(n_timesteps, n_basins, 3, dtype=dtype, device=device)
    # Seasonal forcing pattern with realistic ranges
    t = torch.arange(n_timesteps, dtype=dtype, device=device).unsqueeze(1)
    basin_phases = torch.rand(n_basins, dtype=dtype, device=device, generator=gen) * 365
    # Precipitation: ~2-10 mm/d with seasonal variation
    forcing[:, :, 0] = 3.0 + 4.0 * torch.sin(2 * np.pi * (t + basin_phases.unsqueeze(0)) / 365) + 2.0
    forcing[:, :, 0] = F.relu(forcing[:, :, 0] + torch.randn(n_timesteps, n_basins, dtype=dtype, device=device, generator=gen) * 1.5)
    # Temperature: ~-2 to 22°C
    forcing[:, :, 1] = 10.0 + 12.0 * torch.sin(2 * np.pi * (t + basin_phases.unsqueeze(0) - 60) / 365)
    forcing[:, :, 1] = forcing[:, :, 1] + torch.randn(n_timesteps, n_basins, dtype=dtype, device=device, generator=gen) * 3.0
    # PET: ~0.5-5 mm/d
    forcing[:, :, 2] = 1.5 + 2.0 * torch.sin(2 * np.pi * (t + basin_phases.unsqueeze(0) - 30) / 365) + 1.0
    forcing[:, :, 2] = F.relu(forcing[:, :, 2] + torch.randn(n_timesteps, n_basins, dtype=dtype, device=device, generator=gen) * 0.5)
    forcing = torch.clamp(forcing, min=0.0)
    return forcing


def sample_params_lhs(entry: CoreModelEntry, n_basins, dtype, device, gen):
    """LHS-like parameter sampling in [low+0.05*range, high-0.05*range]."""
    n_params = len(entry.param_bounds)
    raw = torch.zeros(n_basins, n_params, dtype=dtype, device=device)
    for i, (name, (lo, hi)) in enumerate(entry.param_bounds.items()):
        rand = torch.rand(n_basins, dtype=dtype, device=device, generator=gen)
        # 5-95% of the parameter range (avoid extreme boundaries)
        raw[:, i] = lo + (0.05 + 0.9 * rand) * (hi - lo)
    return raw


def build_params_dict(entry, raw_params):
    """Convert (n_basins, n_params) tensor to dict of (n_basins, 1) tensors."""
    return {name: raw_params[:, i].unsqueeze(-1) for i, name in enumerate(entry.param_bounds.keys())}


# ====================================================================
# 2. CORE MODEL RUNNER
# ====================================================================

def run_model_loop(entry, forcing, params_dict, n_groups=1):
    """Run the core step function over the forcing sequence. Returns Qsim, Ea."""
    dtype = forcing.dtype
    device = forcing.device
    n_steps, n_grid = forcing.shape[:2]
    n_groups = 1

    states = [s.to(dtype=dtype, device=device) for s in entry.init_fn(n_grid, n_groups, torch.device(device), NEARZERO)]
    param_values = [params_dict[name] for name in entry.param_bounds.keys()]

    sig = inspect.signature(entry.step_fn).parameters
    has_doy = "doy" in sig
    has_mean_P = "mean_P" in sig
    has_delta_t = "delta_t" in sig

    qsim_list = []
    ea_list = []
    states_list = [list(states)]

    for t in range(n_steps):
        P_t = forcing[t, :, 0:1]
        T_t = forcing[t, :, 1:2]
        PET_t = forcing[t, :, 2:3]

        kwargs = {}
        if has_doy:
            kwargs["doy"] = torch.full_like(P_t, float((t % 365) + 1))
        if has_mean_P:
            kwargs["mean_P"] = forcing[:, :, 0].mean(dim=0, keepdim=False).unsqueeze(-1)
        if has_delta_t:
            kwargs["delta_t"] = torch.ones_like(P_t)

        result = entry.step_fn(P_t, T_t, PET_t, *param_values, *states, **kwargs)
        qsim_list.append(result[0])
        ea_list.append(result[1])
        states = list(result[2:])
        states_list.append(states)

    Qsim = torch.stack(qsim_list, dim=0)  # (T, B, 1)
    Ea = torch.stack(ea_list, dim=0)
    return Qsim, Ea


# ====================================================================
# DIMENSION 1: PER-PARAM GRADIENT SPARSITY
# ====================================================================

def compute_gradient_sparsity(entry, forcing, raw_params):
    """Forward + loss + backward for a single param set. Returns per-param zero_frac and grad_abs_mean."""
    dtype = forcing.dtype
    device = forcing.device
    n_steps, n_grid = forcing.shape[:2]

    raw = raw_params.clone().detach().requires_grad_(True)
    params_dict = build_params_dict(entry, raw)

    Qsim, Ea = run_model_loop(entry, forcing, params_dict)

    # Scalar loss: MSE against a fixed target (encourages non-zero gradients)
    target = torch.ones_like(Qsim) * Qsim.detach().mean()
    loss = F.mse_loss(Qsim, target) + 0.1 * Ea.mean()

    grads = torch.autograd.grad(loss, raw, create_graph=False)[0]

    param_stats = []
    for i, name in enumerate(entry.param_bounds.keys()):
        g = grads[:, i]  # (n_basins,)
        g_abs = torch.abs(g)
        zero_frac = float((g_abs < GRAD_ZERO_THRESH).float().mean())
        grad_mean = float(g_abs.mean())
        param_stats.append({
            "model": entry.model_name, "param": name,
            "zero_frac": zero_frac, "grad_abs_mean": grad_mean,
        })
    return param_stats


def track_gradient_evolution(entry, forcing, raw_params, n_steps=20):
    """Run n_steps of SGD and track zero_frac evolution."""
    dtype = forcing.dtype
    device = forcing.device

    raw = raw_params.clone().detach().requires_grad_(True)
    lr = 0.01

    zero_frac_trajectory = defaultdict(list)

    for step in range(n_steps):
        params_dict = build_params_dict(entry, raw)
        Qsim, Ea = run_model_loop(entry, forcing, params_dict)
        target = torch.ones_like(Qsim) * Qsim.detach().mean()
        loss = F.mse_loss(Qsim, target) + 0.1 * Ea.mean()

        grads = torch.autograd.grad(loss, raw, create_graph=False)[0]

        # Record zero_frac
        for i, name in enumerate(entry.param_bounds.keys()):
            g_abs = torch.abs(grads[:, i])
            zf = float((g_abs < GRAD_ZERO_THRESH).float().mean())
            zero_frac_trajectory[name].append(zf)

        # SGD step
        with torch.no_grad():
            raw = raw - lr * grads
            raw = raw.detach().clone().requires_grad_(True)

    # Summary
    start_zf = {name: vals[0] for name, vals in zero_frac_trajectory.items()}
    end_zf = {name: vals[-1] for name, vals in zero_frac_trajectory.items()}
    upward_params = [name for name in zero_frac_trajectory 
                     if zero_frac_trajectory[name][-1] - zero_frac_trajectory[name][0] > 0.1]

    return {
        "start_zf": float(np.mean(list(start_zf.values()))),
        "end_zf": float(np.mean(list(end_zf.values()))),
        "upward_params": upward_params,
        "trajectory": zero_frac_trajectory,
    }


# ====================================================================
# DIMENSION 2: BOUNDARY SATURATION + GRADIENT LINKAGE
# ====================================================================

def compute_boundary_saturation(entry, raw_params):
    """Compute boundary saturation: fraction of params near 0 (<2%) or near 1 (>98%) of range."""
    param_bounds = entry.param_bounds
    near_low = []
    near_high = []
    grad_linked_issues = []

    for i, name in enumerate(param_bounds.keys()):
        lo, hi = param_bounds[name]
        range_span = hi - lo
        if range_span == 0:
            continue
        norm_val = (raw_params[:, i].mean() - lo) / range_span
        norm_val = float(norm_val)

        # Note: raw_params is the raw physical parameter (not logit). 
        # Normalize it to [0, 1] within bounds.
        
    return {"near_low_count": len(near_low), "near_high_count": len(near_high),
            "near_low_params": [(p, float(v)) for p, v in near_low],
            "near_high_params": [(p, float(v)) for p, v in near_high]}


def compute_boundary_gradient_linkage(entry, raw_params, grad_stats):
    """Link boundary saturation with gradient magnitude.
    
    Distinguish: 
      (a) "normal convergence" - param near boundary AND gradient is small
      (b) "stuck" - param near boundary BUT gradient is still large (> 1e-3)
    """
    param_bounds = entry.param_bounds
    anomalies = []

    for i, name in enumerate(param_bounds.keys()):
        lo, hi = param_bounds[name]
        norm_val = (raw_params[:, i].mean() - lo) / (hi - lo)
        norm_val = float(norm_val)

        grad_zf = grad_stats[i]["zero_frac"]
        grad_mean = grad_stats[i]["grad_abs_mean"]

        if norm_val < 0.02:
            if grad_mean > 1e-3:
                anomalies.append({
                    "param": name, "side": "low", "norm_value": norm_val,
                    "grad_mean": grad_mean, "type": "stuck"
                })
        elif norm_val > 0.98:
            if grad_mean > 1e-3:
                anomalies.append({
                    "param": name, "side": "high", "norm_value": norm_val,
                    "grad_mean": grad_mean, "type": "stuck"
                })

    return anomalies


# ====================================================================
# DIMENSION 3: INITIAL DISCHARGE MAGNITUDE DEVIATION
# ====================================================================

def compute_discharge_deviation(entry, forcing, raw_params):
    """Compare model output against target magnitude at initialization."""
    dtype = forcing.dtype
    device = forcing.device

    params_dict = build_params_dict(entry, raw_params)
    Qsim, Ea = run_model_loop(entry, forcing, params_dict)

    q_mean = float(Qsim.mean())
    q_std = float(Qsim.std())
    target_mean = float(forcing[:, :, 0].mean())  # precipitation mean
    target_std = float(forcing[:, :, 0].std())

    ratio_mean = q_mean / max(target_mean, 1e-6)
    ratio_std = q_std / max(target_std, 1e-6)

    deviation_mean = abs(np.log10(max(ratio_mean, 1e-6)))
    deviation_std = abs(np.log10(max(ratio_std, 1e-6)))

    return {
        "q_mean": q_mean, "q_std": q_std,
        "target_mean": target_mean, "target_std": target_std,
        "ratio_mean": ratio_mean, "ratio_std": ratio_std,
        "deviation_order_mean": deviation_mean,
        "deviation_order_std": deviation_std,
    }


# ====================================================================
# RISK CLASSIFICATION
# ====================================================================

def classify_risk(model_name, grad_stats, grad_evo, boundary_anomalies, discharge_dev):
    """Classify model into low/medium/high risk."""
    points = 0
    reasons = []

    # All-model zero_frac distribution will be computed after collecting all data
    # For now, return raw statistics

    zf_values = [s["zero_frac"] for s in grad_stats]
    avg_zf = float(np.mean(zf_values)) if zf_values else 0.0
    avg_grad = float(np.mean([s["grad_abs_mean"] for s in grad_stats])) if grad_stats else 0.0

    n_params = len(zf_values)
    high_zf_count = sum(1 for z in zf_values if z > 0.8)
    stuck_anomalies = boundary_anomalies

    return {
        "avg_zero_frac": avg_zf,
        "avg_grad_abs_mean": avg_grad,
        "n_params": n_params,
        "high_zf_count": high_zf_count,
        "stuck_params": stuck_anomalies,
        "discharge_deviation": discharge_dev["deviation_order_mean"],
        "grad_evo_upward": len(grad_evo["upward_params"]) if grad_evo else 0,
    }


# ====================================================================
# MAIN DIAGNOSTIC LOOP
# ====================================================================

def main():
    print(f"全模型梯度可训练性诊断 — {len(ALL_MODELS)} models")
    print(f"Output: {OUTPUT_DIR}\n")

    # ---- Step 1: Build unified sample ----
    print("=" * 70)
    print("构造统一样本...")
    dtype = torch.float32
    device = "cpu"
    n_basins = 8
    forcing = make_unified_sample(n_basins=n_basins, n_timesteps=120, dtype=dtype, device=device)
    print(f"  Forcing shape: {forcing.shape}, P range: [{forcing[:,:,0].min():.1f}, {forcing[:,:,0].max():.1f}]")
    print(f"  T range: [{forcing[:,:,1].min():.1f}, {forcing[:,:,1].max():.1f}]")
    print(f"  PET range: [{forcing[:,:,2].min():.1f}, {forcing[:,:,2].max():.1f}]")

    gen = torch.Generator(device=device).manual_seed(SEED + 1)

    all_results = []
    
    print("\n" + "=" * 70)
    print("逐模型诊断...")
    print("=" * 70)

    for model_name in ALL_MODELS:
        entry = CORE_MODEL_REGISTRY[model_name]
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  {model_name} ({len(entry.param_bounds)} params)")

        # Sample parameters
        raw_params = sample_params_lhs(entry, n_basins, dtype, device, gen)

        # ---- Dimension 1: Gradient sparsity ----
        try:
            grad_stats = compute_gradient_sparsity(entry, forcing, raw_params)
        except Exception as e:
            print(f"    SKIP (gradient error): {str(e)[:80]}")
            continue

        zf_vals = [s["zero_frac"] for s in grad_stats]
        gm_vals = [s["grad_abs_mean"] for s in grad_stats]
        print(f"    Dim1: zero_frac avg={np.mean(zf_vals):.3f} med={np.median(zf_vals):.3f}, "
              f"grad_mean avg={np.mean(gm_vals):.3e}")

        high_zf = [(s["param"], s["zero_frac"]) for s in grad_stats if s["zero_frac"] > 0.5]
        if high_zf:
            shown = high_zf[:5]
            for p, z in shown:
                print(f"      high_zf: {p:15s} zero_frac={z:.3f}")

        # ---- Dimension 1b: 20-step gradient evolution ----
        try:
            grad_evo = track_gradient_evolution(entry, forcing, raw_params, n_steps=20)
            print(f"    Dim1b: zf start={grad_evo['start_zf']:.3f} end={grad_evo['end_zf']:.3f} "
                  f"upward_params={grad_evo['upward_params']}")
        except Exception as e:
            print(f"    Dim1b: SKIP (evo error): {str(e)[:60]}")
            grad_evo = {"start_zf": 0.0, "end_zf": 0.0, "upward_params": [], "trajectory": {}}

        # ---- Dimension 2: Boundary saturation + gradient linkage ----
        try:
            boundary_anomalies = compute_boundary_gradient_linkage(entry, raw_params, grad_stats)
            if boundary_anomalies:
                print(f"    Dim2: stuck params: {[(a['param'], a['type']) for a in boundary_anomalies]}")
            else:
                print(f"    Dim2: no stuck params found")
        except Exception as e:
            print(f"    Dim2: SKIP: {str(e)[:60]}")
            boundary_anomalies = []

        # ---- Dimension 3: Discharge deviation ----
        try:
            discharge_dev = compute_discharge_deviation(entry, forcing, raw_params)
            print(f"    Dim3: Q_mean={discharge_dev['q_mean']:.2f} vs P_mean={discharge_dev['target_mean']:.2f}, "
                  f"ratio={discharge_dev['ratio_mean']:.2f}, dev_order={discharge_dev['deviation_order_mean']:.2f}")
        except Exception as e:
            print(f"    Dim3: SKIP: {str(e)[:60]}")
            discharge_dev = {"deviation_order_mean": 0.0, "q_mean": 0.0, "target_mean": 1.0, "ratio_mean": 0.0}

        # ---- Classify risk ----
        risk_info = classify_risk(model_name, grad_stats, grad_evo, boundary_anomalies, discharge_dev)

        elapsed = time.time() - t0
        all_results.append({
            "model": model_name,
            "n_params": risk_info["n_params"],
            "avg_zero_frac": risk_info["avg_zero_frac"],
            "avg_grad_abs_mean": risk_info["avg_grad_abs_mean"],
            "high_zf_count": risk_info["high_zf_count"],
            "stuck_params": risk_info["stuck_params"],
            "discharge_dev_order": risk_info["discharge_deviation"],
            "grad_evo_zf_start": grad_evo["start_zf"],
            "grad_evo_zf_end": grad_evo["end_zf"],
            "grad_evo_upward_count": risk_info["grad_evo_upward"],
            "grad_stats_detail": grad_stats,
        })

        # Quick print
        spotlight = " ★" if model_name in PRIORITY_SPOTLIGHT else ""
        print(f"    → risk_info: zf={risk_info['avg_zero_frac']:.3f} grad={risk_info['avg_grad_abs_mean']:.2e} "
              f"dev={risk_info['discharge_deviation']:.2f} ({elapsed:.0f}s){spotlight}")

    # ---- Post-processing: compute global thresholds ----
    print("\n" + "=" * 70)
    print("计算全局阈值 & 风险分级...")

    all_zf = [r["avg_zero_frac"] for r in all_results]
    global_median_zf = float(np.median(all_zf))
    global_q75_zf = float(np.quantile(all_zf, 0.75))
    print(f"  Global zero_frac: median={global_median_zf:.3f}, Q75={global_q75_zf:.3f}")

    all_dev = [r["discharge_dev_order"] for r in all_results]
    global_median_dev = float(np.median(all_dev))

    # Classify
    for r in all_results:
        points = 0
        reasons = []

        # zero_frac check
        if r["avg_zero_frac"] > global_q75_zf + 0.1:
            points += 2
            reasons.append(f"zero_frac high ({r['avg_zero_frac']:.3f} > Q75={global_q75_zf:.3f})")
        elif r["avg_zero_frac"] > global_median_zf + 0.1:
            points += 1
            reasons.append(f"zero_frac above median ({r['avg_zero_frac']:.3f})")

        # Stuck params
        if len(r["stuck_params"]) > 0:
            points += 1
            reasons.append(f"{len(r['stuck_params'])} stuck params")

        # Discharge deviation
        if r["discharge_dev_order"] > 1.0:
            points += 2
            reasons.append(f"discharge dev > 1 order ({r['discharge_dev_order']:.1f})")
        elif r["discharge_dev_order"] > 0.5:
            points += 1
            reasons.append(f"discharge dev ~ 1 order ({r['discharge_dev_order']:.1f})")

        # Gradient evolution upward
        if r["grad_evo_upward_count"] > 0:
            points += 1
            reasons.append(f"zf rising ({r['grad_evo_upward_count']} params)")

        # Final risk level
        if points >= 3:
            level = "HIGH"
        elif points >= 1:
            level = "MEDIUM"
        else:
            level = "LOW"

        r["risk_level"] = level
        r["risk_points"] = points
        r["risk_reasons"] = reasons

    # ---- Report ----
    print("\n" + "=" * 70)
    print("风险汇总")
    print("=" * 70)

    levels = defaultdict(list)
    for r in all_results:
        levels[r["risk_level"]].append(r["model"])

    for level in ["HIGH", "MEDIUM", "LOW"]:
        models = levels[level]
        print(f"\n  {level} ({len(models)}):")
        for m in models:
            r = next(r for r in all_results if r["model"] == m)
            spotlight = " ★" if m in PRIORITY_SPOTLIGHT else ""
            print(f"    {m:20s} zf={r['avg_zero_frac']:.3f} grad={r['avg_grad_abs_mean']:.2e} "
                  f"dev={r['discharge_dev_order']:.2f} points={r['risk_points']}"
                  f"{spotlight}")
            if r["risk_reasons"]:
                print(f"      reasons: {'; '.join(r['risk_reasons'])}")
            if r["stuck_params"]:
                for sp in r["stuck_params"][:3]:
                    print(f"      stuck: {sp['param']}({sp['side']}) norm={sp['norm_value']:.3f} grad={sp['grad_mean']:.2e}")

    # ---- CSV output ----
    csv_path = OUTPUT_DIR / "trainability_risk_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "n_params", "risk_level", "risk_points",
            "avg_zero_frac", "avg_grad_abs_mean", "high_zf_count",
            "discharge_dev_order", "grad_evo_zf_start", "grad_evo_zf_end",
            "grad_evo_upward_count", "n_stuck_params", "risk_reasons",
        ])
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: (x["risk_level"] != "HIGH", x["risk_level"] != "MEDIUM", x["risk_level"] != "LOW", x["avg_zero_frac"])):
            writer.writerow({
                "model": r["model"], "n_params": r["n_params"],
                "risk_level": r["risk_level"], "risk_points": r["risk_points"],
                "avg_zero_frac": f"{r['avg_zero_frac']:.4f}",
                "avg_grad_abs_mean": f"{r['avg_grad_abs_mean']:.4e}",
                "high_zf_count": r["high_zf_count"],
                "discharge_dev_order": f"{r['discharge_dev_order']:.2f}",
                "grad_evo_zf_start": f"{r['grad_evo_zf_start']:.3f}",
                "grad_evo_zf_end": f"{r['grad_evo_zf_end']:.3f}",
                "grad_evo_upward_count": r["grad_evo_upward_count"],
                "n_stuck_params": len(r["stuck_params"]),
                "risk_reasons": "; ".join(r["risk_reasons"]),
            })

    # Detailed per-param CSV
    detail_path = OUTPUT_DIR / "trainability_per_param.csv"
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "param", "zero_frac", "grad_abs_mean"])
        writer.writeheader()
        for r in all_results:
            for s in r["grad_stats_detail"]:
                writer.writerow({
                    "model": r["model"], "param": s["param"],
                    "zero_frac": f"{s['zero_frac']:.4f}",
                    "grad_abs_mean": f"{s['grad_abs_mean']:.4e}",
                })

    print(f"\nReports written to {OUTPUT_DIR}")
    print(f"  {csv_path.name}")
    print(f"  {detail_path.name}")

    # Spotlight summary
    print("\n" + "=" * 70)
    print("重点关注模型 (gsfb, hbv96, plateau, smar, modhydrolog)")
    print("=" * 70)
    for name in PRIORITY_SPOTLIGHT:
        r = next((r for r in all_results if r["model"] == name), None)
        if r:
            print(f"\n  {name}: risk={r['risk_level']}")
            print(f"    zf={r['avg_zero_frac']:.3f} grad={r['avg_grad_abs_mean']:.2e} dev={r['discharge_dev_order']:.2f}")
            print(f"    high_zf params: {r['high_zf_count']}/{r['n_params']}")
            print(f"    stuck: {[(a['param'], a['type']) for a in r['stuck_params']]}")
            print(f"    reasons: {r['risk_reasons']}")

    return 0


if __name__ == "__main__":
    exit(main())
