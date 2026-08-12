"""
Warmup-aware gradient retest for HIGH-risk models.
Tests: (a) core step with N-day warmup (b) full model with/without warmup
Key metric: zero_frac change after warmup + discharge deviation reduction
"""
import sys, os, inspect, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DMOTPY_DIR = SCRIPT_DIR.parent
os.chdir(DMOTPY_DIR)
sys.path.insert(0, str(DMOTPY_DIR))

import numpy as np, torch, torch.nn.functional as F
from tests.core_model_registry import CORE_MODEL_REGISTRY

SEED = 202607022; NEARZERO = 1.0e-6; GRAD_ZERO_THRESH = 1e-12
n_basins = 8; n_timesteps_overall = 400  # 365 warmup + 35 eval
WARMUP_DAYS = 365
dtype = torch.float32; device = "cpu"

# ---- Build forcing (400 days) ----
gen_f = torch.Generator(device=device).manual_seed(SEED)
forcing = torch.zeros(n_timesteps_overall, n_basins, 3, dtype=dtype, device=device)
t = torch.arange(n_timesteps_overall, dtype=dtype, device=device).unsqueeze(1)
phases = torch.rand(n_basins, dtype=dtype, device=device, generator=gen_f) * 365
forcing[:,:,0] = 3.0 + 4.0*torch.sin(2*np.pi*(t+phases.unsqueeze(0))/365) + 2.0
forcing[:,:,0] = F.relu(forcing[:,:,0] + torch.randn(n_timesteps_overall,n_basins,dtype=dtype,device=device,generator=gen_f)*1.5)
forcing[:,:,1] = 10.0 + 12.0*torch.sin(2*np.pi*(t+phases.unsqueeze(0)-60)/365)
forcing[:,:,1] = forcing[:,:,1] + torch.randn(n_timesteps_overall,n_basins,dtype=dtype,device=device,generator=gen_f)*3.0
forcing[:,:,2] = 1.5 + 2.0*torch.sin(2*np.pi*(t+phases.unsqueeze(0)-30)/365) + 1.0
forcing[:,:,2] = F.relu(forcing[:,:,2] + torch.randn(n_timesteps_overall,n_basins,dtype=dtype,device=device,generator=gen_f)*0.5)
forcing = torch.clamp(forcing, min=0.0)

gen_p = torch.Generator(device=device).manual_seed(SEED + 1)

HIGH_RISK = ['gsfb', 'hbv96', 'hillslope', 'penman', 'plateau', 'smar', 'tcm']

print("=" * 80)
print("WARMUP-AWARE GRADIENT DIAGNOSIS")
print(f"  {WARMUP_DAYS}-day warmup + 35-day eval")
print("=" * 80)

for model_name in HIGH_RISK:
    entry = CORE_MODEL_REGISTRY[model_name]
    t0 = time.time()
    print(f"\n{'─'*60}")
    print(f"  {model_name} ({len(entry.param_bounds)} params, {len(entry.state_names)} states)")
    
    pnames = list(entry.param_bounds.keys())
    n_params = len(pnames)
    
    raw = torch.zeros(n_basins, n_params, dtype=dtype, device=device)
    for i, (pn, (lo, hi)) in enumerate(entry.param_bounds.items()):
        rand = torch.rand(n_basins, dtype=dtype, device=device, generator=gen_p)
        raw[:, i] = lo + (0.15 + 0.7 * rand) * (hi - lo)
    
    params_map = {name: raw[:, i].unsqueeze(-1) for i, name in enumerate(pnames)}
    states_init = [s.to(dtype=dtype) for s in entry.init_fn(n_basins, 1, torch.device(device), NEARZERO)]
    n_states = len(states_init)
    
    sig = inspect.signature(entry.step_fn).parameters
    has_doy = "doy" in sig; has_mean_P = "mean_P" in sig; has_delta_t = "delta_t" in sig
    
    def step_fn(P, T, PET, states):
        kw = {}
        if has_doy: kw["doy"] = torch.full_like(P, float(np.random.randint(1,366)))
        if has_mean_P: kw["mean_P"] = forcing[:,:,0].mean(dim=0).unsqueeze(-1)
        if has_delta_t: kw["delta_t"] = torch.ones_like(P)
        args = [P, T, PET] + [params_map[n] for n in pnames]
        result = entry.step_fn(*args, *states, **kw)
        return result[0], result[1], list(result[2:2+n_states])
    
    # ---- NO WARMUP: run without spin-up, compute gradient ----
    curr = [s.clone() for s in states_init]
    Q_no_warmup = []
    for ti in range(n_timesteps_overall):
        Pi = forcing[ti,:,0:1]; Ti = forcing[ti,:,1:2]; PETi = forcing[ti,:,2:3]
        q, ea, curr = step_fn(Pi, Ti, PETi, curr)
        Q_no_warmup.append(float(q.mean()))
    
    q_nowu_mean = float(np.mean(Q_no_warmup[-35:]))  # last 35 days
    q_nowu_overall = float(np.mean(Q_no_warmup))
    
    # Gradient with no-warmup (last 35 steps with grad)
    raw_nw = raw.clone().detach().requires_grad_(True)
    params_nw = {name: raw_nw[:, i].unsqueeze(-1) for i, name in enumerate(pnames)}
    curr_nw = [s.clone() for s in states_init]
    
    # Spin up WITHOUT grad (warmup)
    with torch.no_grad():
        for ti in range(WARMUP_DAYS):
            Pi = forcing[ti,:,0:1]; Ti = forcing[ti,:,1:2]; PETi = forcing[ti,:,2:3]
            kw = {}
            if has_doy: kw["doy"] = torch.full_like(Pi, float(ti%365+1))
            if has_mean_P: kw["mean_P"] = forcing[:,:,0].mean(dim=0).unsqueeze(-1)
            if has_delta_t: kw["delta_t"] = torch.ones_like(Pi)
            args = [Pi, Ti, PETi] + [params_nw[n] for n in pnames]
            result = entry.step_fn(*args, *curr_nw, **kw)
            curr_nw = list(result[2:2+n_states])
    
    # Detach after warmup
    curr_nw = [c.detach().clone() for c in curr_nw]
    
    # ---- WITH WARMUP: gradient on last 35 steps ----
    Q_warmup = []  # no-grad warmup output
    total_with = torch.zeros(1, dtype=dtype, device=device)
    for ti in range(WARMUP_DAYS, n_timesteps_overall):
        Pi = forcing[ti,:,0:1]; Ti = forcing[ti,:,1:2]; PETi = forcing[ti,:,2:3]
        kw = {}
        if has_doy: kw["doy"] = torch.full_like(Pi, float(ti%365+1))
        if has_mean_P: kw["mean_P"] = forcing[:,:,0].mean(dim=0).unsqueeze(-1)
        if has_delta_t: kw["delta_t"] = torch.ones_like(Pi)
        args = [Pi, Ti, PETi] + [params_nw[n] for n in pnames]
        result = entry.step_fn(*args, *curr_nw, **kw)
        total_with = total_with + result[0].sum() + result[1].sum()
        curr_nw = list(result[2:2+n_states])
        Q_warmup.append(float(result[0].mean()))
    
    grads_with = torch.autograd.grad(total_with, raw_nw, create_graph=False)[0]
    
    q_wu_mean = float(np.mean(Q_warmup))
    
    # ---- NO WARMUP gradient (same steps, but no spin-up) ----
    raw_nw2 = raw.clone().detach().requires_grad_(True)
    params_nw2 = {name: raw_nw2[:, i].unsqueeze(-1) for i, name in enumerate(pnames)}
    curr_nw2 = [s.clone() for s in states_init]
    total_no = torch.zeros(1, dtype=dtype, device=device)
    for ti in range(WARMUP_DAYS, n_timesteps_overall):
        Pi = forcing[ti,:,0:1]; Ti = forcing[ti,:,1:2]; PETi = forcing[ti,:,2:3]
        kw = {}
        if has_doy: kw["doy"] = torch.full_like(Pi, float(ti%365+1))
        if has_mean_P: kw["mean_P"] = forcing[:,:,0].mean(dim=0).unsqueeze(-1)
        if has_delta_t: kw["delta_t"] = torch.ones_like(Pi)
        args = [Pi, Ti, PETi] + [params_nw2[n] for n in pnames]
        result = entry.step_fn(*args, *curr_nw2, **kw)
        total_no = total_no + result[0].sum() + result[1].sum()
        curr_nw2 = list(result[2:2+n_states])
    
    grads_no = torch.autograd.grad(total_no, raw_nw2, create_graph=False)[0]
    
    # ---- Compare ----
    p_mean = float(forcing[:,:,0].mean())
    
    print(f"  Q (last 35d):  no_wu={q_nowu_mean:.3f}  with_wu={q_wu_mean:.3f}  P_ref={p_mean:.1f}")
    print(f"  Q ratio to P:  no_wu={q_nowu_mean/p_mean:.2f}  with_wu={q_wu_mean/p_mean:.2f}")
    hdr = f"  {'Param':15s} {'no_wu zf':>9s} {'no_wu gm':>10s} {'with_wu zf':>9s} {'with_wu gm':>10s} {'DELTA':>8s}"
    sep = f"  {'─────':15s} {'────────':>9s} {'──────────':>10s} {'─────────':>9s} {'───────────':>10s} {'──────':>8s}"
    print(hdr)
    print(sep)
    
    improved = 0; degraded = 0
    for i, pn in enumerate(pnames):
        g_n = grads_no[:, i]; g_w = grads_with[:, i]
        zf_n = float((torch.abs(g_n) < GRAD_ZERO_THRESH).float().mean())
        zf_w = float((torch.abs(g_w) < GRAD_ZERO_THRESH).float().mean())
        gm_n = float(torch.abs(g_n).mean()); gm_w = float(torch.abs(g_w).mean())
        
        delta_zf = zf_n - zf_w
        if delta_zf > 0.1: improved += 1
        elif delta_zf < -0.1: degraded += 1
        
        marker = "↓" if delta_zf > 0.1 else ("↑" if delta_zf < -0.1 else "=")
        print(f"  {pn:15s} {zf_n:9.3f} {gm_n:10.2e} {zf_w:9.3f} {gm_w:10.2e} {delta_zf:8.3f} {marker}")
    
    avg_zf_n = float(np.mean([float((torch.abs(grads_no[:,i]) < GRAD_ZERO_THRESH).float().mean()) for i in range(len(pnames))]))
    avg_zf_w = float(np.mean([float((torch.abs(grads_with[:,i]) < GRAD_ZERO_THRESH).float().mean()) for i in range(len(pnames))]))
    
    elapsed = time.time() - t0
    print(f"  AVERAGE: no_wu zf={avg_zf_n:.3f} → with_wu zf={avg_zf_w:.3f}  "
          f"(Δ={avg_zf_n-avg_zf_w:+.3f})  improved={improved} degraded={degraded}  ({elapsed:.0f}s)")
    
    # Final assessment
    q_improvement = abs(np.log10(max(q_wu_mean/p_mean, 1e-6))) - abs(np.log10(max(q_nowu_mean/p_mean, 1e-6)))
    dev_before = round(abs(np.log10(max(q_nowu_mean/p_mean, 1e-6))), 2)
    dev_after = round(abs(np.log10(max(q_wu_mean/p_mean, 1e-6))), 2)
    
    if avg_zf_w < avg_zf_n * 0.8 or dev_after < dev_before * 0.5:
        verdict = "WARMUP HELPS SIGNIFICANTLY"
    elif avg_zf_w < avg_zf_n or dev_after < dev_before:
        verdict = "warmup helps slightly"
    else:
        verdict = "no improvement"
    
    print(f"  → {verdict}")
    print(f"  → Discharge deviation: {dev_before:.2f} orders → {dev_after:.2f} orders ({elapsed:.0f}s)")

print("\n" + "=" * 80)
print("COMPLETE")
