#!/usr/bin/env python3
"""Pre-training formula screening for the interception candidates (Phases 1/3/4).

Forward/Jacobian-only.  No training.  For every (formula, semantics) combo:

  formula:   original (A/C), normalized (B/D), E = bounded linear cosine,
             F = bounded logistic cosine
  semantics: S0 (production PET cap + shared budget), S1 (V1 independent loss
             with PET cap), S2 (independent loss without interception PET cap)

Part A — effective amplitude separation (no state loop; interception depends
only on P / PET_effective / the seasonal gate):

  * evaluation-period I/P (post all caps), and pre-cap I/P
  * mean interception, seasonal amplitude (std of I)
  * precipitation-cap and PET-cap binding fractions (rainy days)
  * variation of I/P across the internal shape sweep (median basin range),
    relative range and CV
  * median |d(I/P)/dshape| and |d(I/P)/dphase| (central finite differences)
  * distribution across basins: median / q25 / q75

Part B — structural identifiability (full model, autograd, official descale):

  * |cos(dQ/dw_int, dQ/dshape)| and |cos(dQ/dw_int, dQ/dphase)| (absolute)
  * median |dQ/dw_int|, |dQ/dshape|, |dQ/dphase|
  * local 2x2 Gram condition number of (dQ/dw_int, dQ/dshape)
  * zero-gradient fractions

Part C — deterministic kappa-range probe for candidate F (seasonal
expressiveness / gradient health), and calendar-mean checks for E/F.

Memory-bounded: 128 basins for Part A, 32 basins for Part B; sequential.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader,
)
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402
from project.flexmopex.models import mopex_core_candidates as cand  # noqa: E402
from project.flexmopex.models.learned_weight_mopex import LearnedWeightMopex  # noqa: E402
from project.flexmopex.models.learned_weight_mopex_v1 import (  # noqa: E402
    LearnedWeightMopexV1, LearnedWeightMopexDecoupled, LearnedWeightMopexV1Decoupled,
)
from project.flexmopex.models.learned_weight_mopex_candidates import (  # noqa: E402
    LearnedWeightMopexE, LearnedWeightMopexF,
)

# deterministic basin subsets (fixed, same for every candidate)
SWEEP_BASINS = list(range(128))          # Part A
JAC_BASINS = list(range(32))             # Part B
W0, W1 = 730, 1825                        # Part A evaluation window (3 full years)
JT0, JT1 = 365, 1095                      # Part B window (warmup 365 + scored 365)

SHAPE_GRID = {
    "original": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "normalized": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "linear": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "logistic": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
}
PHASE_GRID = [0.0, 91.25, 182.5, 273.75]
MID_SHAPE = {"original": 0.5, "normalized": 0.5, "linear": 0.5, "logistic": 2.5}
MID_PHASE = 182.5
FD_H_SHAPE = {"original": 0.05, "normalized": 0.05, "linear": 0.05, "logistic": 0.25}
FD_H_PHASE = 5.0

COMBOS = [
    ("original", "S0", "LearnedWeightMopex"),
    ("original", "S1", "LearnedWeightMopexV1"),
    ("normalized", "S0", "LearnedWeightMopexDecoupled"),
    ("normalized", "S1", "LearnedWeightMopexV1Decoupled"),
    ("linear", "S0", "LearnedWeightMopexE"),
    ("linear", "S1", "LearnedWeightMopexE"),
    ("linear", "S2", "LearnedWeightMopexE"),
    ("logistic", "S0", "LearnedWeightMopexF"),
    ("logistic", "S1", "LearnedWeightMopexF"),
    ("logistic", "S2", "LearnedWeightMopexF"),
]

PET_CAP = {"S0": True, "S1": True, "S2": False}
PET_INDEP = {"S0": False, "S1": True, "S2": True}


# ---------------------------------------------------------------------------
# config / handler helpers
# ---------------------------------------------------------------------------
def build_screen_config(phy_name: str, semantics: str, output_root: Path) -> dict:
    cli = parse_args(["--config", "conf/config_dmopex_intercept2x2_A.yaml",
                      "--gpu-id", "0", "--output-root", str(output_root), "--run-name", "screen"])
    cfg = load_config("conf/config_dmopex_intercept2x2_A.yaml")
    apply_runtime_overrides(cfg, cli, config_path="conf/config_dmopex_intercept2x2_A.yaml")
    cfg["mode"] = "train"
    cfg["model"]["phy"]["name"] = [phy_name]
    cfg["model"]["phy"]["model"] = [phy_name]
    cfg["delta_model"]["phy_model"]["model"] = [phy_name]
    cfg["model"]["phy"]["interception_semantics"] = semantics
    cfg["delta_model"]["phy_model"]["interception_semantics"] = semantics
    cfg["model"]["phy"]["disable_compile"] = True   # screening: eager steps
    cfg["delta_model"]["phy_model"]["disable_compile"] = True
    return cfg


def build_handler(cfg: dict) -> FlexMopexModelHandler:
    return FlexMopexModelHandler(cfg, verbose=False)


def eval_network(handler, attrs: torch.Tensor):
    """Eval-mode nn forward; returns params dict and softmax gate weights."""
    model = next(iter(handler.model_dict.values()))
    nn = model.nn_model
    nn.eval()
    with torch.no_grad():
        params = nn({"c_nn_norm": attrs})
        logits = params["weights"].view(attrs.shape[0], 4, 2).clamp(min=-10.0, max=10.0)
        w = F.softmax(logits, dim=-1)[..., 1]
    return params, w

# ---------------------------------------------------------------------------
# Part A — effective amplitude separation
# ---------------------------------------------------------------------------
def _series(season_mode, pet_cap, P, PET, doy, kappa, phi, w_int, w_phen, T, tmin, tmax):
    return cand.interception_series(
        P, PET, doy, season_mode, kappa, phi, w_int,
        pet_cap=pet_cap, w_phen=w_phen, T=T, tmin=tmin, tmax=tmax,
    )


def part_a(combo, P, PET, doy, T, w_phen, tmin, tmax) -> dict:
    """P/PET/doy/T/w_phen/tmin/tmax all (T, B, 1); kappa/phi are (1, B, S)/(1, B, 1)."""
    mode, sem, _ = combo
    pet_cap = PET_CAP[sem]
    n_basin = P.shape[1]
    grid = torch.tensor(SHAPE_GRID[mode])
    kappa_all = grid.view(1, 1, -1).expand(1, n_basin, -1).to(P.device)  # (1,B,S)

    def eval_ip(kappa, phi):
        return _series(mode, pet_cap, P, PET, doy, kappa, phi,
                       torch.ones_like(kappa), w_phen, T, tmin, tmax)   # w_int=1

    # shape sweep at mid phase
    phi_mid = torch.full((1, n_basin, 1), MID_PHASE, device=P.device)
    I = eval_ip(kappa_all, phi_mid)                       # (T,B,S)
    Psum = P.sum(0)                                        # (B,1)
    ip_shape = I.sum(0) / Psum                             # (B,S) post-cap I/P
    # pre-cap I/P (no caps): pet_cap=False keeps only the safety min(P*s, P)
    I_pre = _series(mode, False, P, PET, doy, kappa_all, phi_mid,
                    torch.ones_like(kappa_all), w_phen, T, tmin, tmax)
    ip_pre = I_pre.sum(0) / Psum

    # cap-binding fractions (rainy days, P > 0.1)
    rainy = (P[..., 0] > 0.1)                              # (T,B)
    trange = torch.clamp(tmax - tmin, min=0.1)
    gsi = torch.clamp((T - tmin) / trange, 0.0, 1.0)
    pet_eff = w_phen * (PET * gsi) + (1.0 - w_phen) * PET  # (T,B,1)
    if mode == "linear":
        s_all = cand.season_linear(doy, kappa_all, phi_mid)
    elif mode == "logistic":
        s_all = cand.season_logistic(doy, kappa_all, phi_mid)
    elif mode == "original":
        s_all = kappa_all * 0.5 * (torch.cos(cand._phase_rad(doy, phi_mid)) + 1.0)
    else:
        import project.flexmopex.models.mopex_core_v1 as v1
        nm = v1.decoupled_norm_mean(kappa_all, phi_mid)
        s_all = v1.decoupled_shape(doy, kappa_all, phi_mid, nm) * v1.C_REF
    fp_all = P * s_all                                       # (T,B,S)
    p_cap_bind = (fp_all > P) & rainy.unsqueeze(-1)
    pet_cap_bind = (fp_all > pet_eff) & (pet_eff <= P) & rainy.unsqueeze(-1)

    # per-basin variation across shape sweep (median over basins of range/CV)
    rng = (ip_shape.max(-1).values - ip_shape.min(-1).values)          # (B,)
    cv = ip_shape.std(-1) / (ip_shape.mean(-1) + 1e-9)
    rng_pre = (ip_pre.max(-1).values - ip_pre.min(-1).values)

    # phase sweep at mid shape
    kappa_mid = torch.full((1, n_basin, 1), MID_SHAPE[mode], device=P.device)
    ip_phase = []
    for ph in PHASE_GRID:
        I = eval_ip(kappa_mid, torch.full((1, n_basin, 1), ph, device=P.device))
        ip_phase.append(I.sum(0) / Psum)                               # (B,1)
    ip_phase = torch.cat(ip_phase, dim=-1)                             # (B,4)
    rng_phase = ip_phase.max(-1).values - ip_phase.min(-1).values

    # finite-difference sensitivities at mid
    h = FD_H_SHAPE[mode]
    I_lo = eval_ip(kappa_mid - h, phi_mid).sum(0) / Psum
    I_hi = eval_ip(kappa_mid + h, phi_mid).sum(0) / Psum
    dip_dshape = (I_hi - I_lo) / (2 * h)                               # (B,1)
    hp = FD_H_PHASE
    I_lo = eval_ip(kappa_mid, phi_mid - hp).sum(0) / Psum
    I_hi = eval_ip(kappa_mid, phi_mid + hp).sum(0) / Psum
    dip_dphase = (I_hi - I_lo) / (2 * hp)

    def q(x):
        x = x.detach().float().reshape(-1)
        return (float(x.median()), float(torch.quantile(x, 0.25)), float(torch.quantile(x, 0.75)))

    # mid-shape evaluation (explicit; consistent across formulas)
    kappa_mid2 = torch.full((1, n_basin, 1), MID_SHAPE[mode], device=P.device)
    I_mid = eval_ip(kappa_mid2, phi_mid)
    ip_mid = I_mid.sum(0) / Psum

    return {
        "ip_shape_range": q(rng),
        "ip_shape_range_precap": q(rng_pre),
        "ip_cv": q(cv),
        "ip_phase_range": q(rng_phase),
        "dip_dshape": q(dip_dshape.abs()),
        "dip_dphase": q(dip_dphase.abs()),
        "p_cap_bind_frac": float(p_cap_bind.float().mean()),
        "pet_cap_bind_frac": float(pet_cap_bind.float().mean()),
        "ip_mean_at_mid": q(ip_mid),
        "i_std_at_mid": q(I_mid.std(0)),
    }


# ---------------------------------------------------------------------------
# Part B — structural identifiability (autograd on the full model)
# ---------------------------------------------------------------------------
def part_b(combo, handler, sample, n_attr) -> dict:
    mode, sem, _ = combo
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    attrs = sample["c_nn_norm"]
    params = nn(sample)
    # inject canonical mid kappa/phi into the two interception slots
    kappa_mid = MID_SHAPE[mode]
    kappa_range = cand.KAPPA_MAX if mode == "logistic" else 1.0
    raw_k = torch.logit(torch.tensor(kappa_mid / kappa_range, dtype=params["params"].dtype,
                                    device=params["params"].device))
    raw_p = torch.logit(torch.tensor(MID_PHASE / 365.0, dtype=params["params"].dtype,
                                    device=params["params"].device))
    # clone preserves the autograd graph (no_grad would detach the slots)
    params_new = params["params"].clone()
    params_new[:, 8] = raw_k
    params_new[:, 9] = raw_p
    params["params"] = params_new

    logits = params["weights"].view(attrs.shape[0], 4, 2).clamp(min=-10.0, max=10.0)
    weights_on = F.softmax(logits, dim=-1)[..., 1]
    mopex_params = phy._descale_mopex_params(params["params"])
    routing = phy._descale_routing_params(params["gamma_uh"])
    P, T, PET, doy, n_steps, n_grid = phy._prepare_forcings(sample)
    Q_mopex = phy._run_weighted_loop(P, T, PET, doy, mopex_params, weights_on, n_steps, n_grid)
    q = phy._apply_routing(Q_mopex.mean(-1), routing)

    def grad_of(y, x):
        g = torch.autograd.grad(y, x, retain_graph=True, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(x)

    n = q.shape[0]
    gQ_w = grad_of(q[:n].sum(), weights_on)[:, 1]             # dQ/dw_int (B,)
    gQ_k = grad_of(q[:n].sum(), mopex_params["alpha"]).sum(-1)  # dQ/dshape (B,)
    gQ_p = grad_of(q[:n].sum(), mopex_params["is_time"]).sum(-1)  # dQ/dphase (B,)

    cos_wk = float(torch.nn.functional.cosine_similarity(gQ_w, gQ_k, dim=0).abs().mean())
    cos_wp = float(torch.nn.functional.cosine_similarity(gQ_w, gQ_p, dim=0).abs().mean())

    # 2x2 Gram condition number per basin
    gw, gk = gQ_w, gQ_k
    g11 = (gw * gw); g12 = (gw * gk); g22 = (gk * gk)
    tr = g11 + g22
    det = g11 * g22 - g12 * g12
    disc = torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0.0))
    lam1 = (tr + disc) / 2
    lam2 = (tr - disc) / 2
    cond = torch.where(lam2 > 1e-12, lam1 / lam2, torch.full_like(lam1, 1e6))
    cond_med = float(torch.median(cond))

    def q(x):
        x = x.detach().float().reshape(-1)
        return (float(x.median()), float(torch.quantile(x, 0.25)), float(torch.quantile(x, 0.75)))

    def zfrac(x):
        return float((x.abs() < 1e-12).float().mean())

    return {
        "abs_cos_dQ_dw_dshape": cos_wk,
        "abs_cos_dQ_dw_dphase": cos_wp,
        "dQ_dw_int": q(gQ_w.abs()),
        "dQ_dshape": q(gQ_k.abs()),
        "dQ_dphase": q(gQ_p.abs()),
        "zero_frac_dQ_dshape": zfrac(gQ_k),
        "zero_frac_dQ_dphase": zfrac(gQ_p),
        "gram_cond_median": cond_med,
    }


# ---------------------------------------------------------------------------
# Part C — deterministic shape/gradient probe (esp. kappa range for F)
# ---------------------------------------------------------------------------
def part_c() -> dict:
    doy = torch.linspace(1.0, 365.0, 365)
    out = {}
    for name, fn, kgrid in (
        ("E_linear", cand.season_linear, [0.0, 0.25, 0.5, 0.75, 1.0]),
        ("F_logistic", cand.season_logistic, [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
    ):
        rows = []
        for k in kgrid:
            kappa = torch.tensor(k)
            phi = torch.tensor(182.5)
            s = fn(doy, kappa, phi)
            h = 0.05 if name.startswith("E") else 0.25
            s_hi = fn(doy, kappa + h, phi)
            s_lo = fn(doy, kappa - h, phi)
            dk = ((s_hi - s_lo) / (2 * h)).abs().mean()
            hp = 2.0
            s_hi = fn(doy, kappa, phi + hp)
            s_lo = fn(doy, kappa, phi - hp)
            dp = ((s_hi - s_lo) / (2 * hp)).abs().mean()
            rows.append({
                "kappa": k, "s_mean": float(s.mean()), "s_min": float(s.min()),
                "s_max": float(s.max()), "mean_abs_ds_dkappa": float(dk),
                "mean_abs_ds_dphi": float(dp),
            })
        out[name] = rows
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="results/intercept_screen")
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    base_cfg = build_screen_config("LearnedWeightMopex", "S0", root)
    if str(base_cfg["device"]).startswith("cuda"):
        torch.cuda.set_device(base_cfg["device"])
    dl = _build_data_loader(base_cfg)
    td = dl.train_dataset
    dev = base_cfg["device"]

    # Part A data: 128 basins, 3-year window
    P = td["x_phy"][W0:W1, SWEEP_BASINS, 0].to(dev)
    T = td["x_phy"][W0:W1, SWEEP_BASINS, 1].to(dev)
    PET = td["x_phy"][W0:W1, SWEEP_BASINS, 2].to(dev)
    doy = td["doy"][W0:W1, SWEEP_BASINS, 0].to(dev)
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, SWEEP_BASINS, -n_attr:].to(dev)

    print(f"[screen] Part A window days {W0}-{W1}, basins {len(SWEEP_BASINS)}; "
          f"Part B basins {len(JAC_BASINS)}, days {JT0}-{JT1}")

    part_c_rows = part_c()
    (root / "partC_kappa_probe.csv").write_text(
        "family,kappa,s_mean,s_min,s_max,mean_abs_ds_dkappa,mean_abs_ds_dphi\n" + "\n".join(
            f"{fam},{r['kappa']},{r['s_mean']:.5f},{r['s_min']:.5f},{r['s_max']:.5f},"
            f"{r['mean_abs_ds_dkappa']:.5f},{r['mean_abs_ds_dphi']:.5f}"
            for fam, rows in part_c_rows.items() for r in rows))
    for fam, rows in part_c_rows.items():
        print(f"[screen] {fam}: " + "  ".join(
            f"k={r['kappa']:g}: mean={r['s_mean']:.3f} range=[{r['s_min']:.3f},{r['s_max']:.3f}] "
            f"|ds/dk|={r['mean_abs_ds_dkappa']:.3f}" for r in rows))

    results = {}
    jac_sample = {
        "x_phy": td["x_phy"][JT0:JT1, JAC_BASINS, :].to(dev),
        "doy": td["doy"][JT0:JT1, JAC_BASINS, :].to(dev),
        "c_nn_norm": td["xc_nn_norm"][0, JAC_BASINS, -n_attr:].to(dev),
    }

    for combo in COMBOS:
        mode, sem, cls = combo
        key = f"{mode}-{sem}"
        print(f"[screen] {key} ...", flush=True)

        # identical fresh network for every combo (same seed per build)
        torch.manual_seed(42)
        # Part A
        # per-basin phenology params from the same fresh network (basins 0..127)
        cfg_a = build_screen_config(cls, sem, root)
        handler_a = build_handler(cfg_a)
        params_a, w_a = eval_network(handler_a, attrs)
        phy_a = next(iter(handler_a.model_dict.values())).phy_model
        phys_a = phy_a._descale_mopex_params(params_a["params"])
        w_phen = w_a[:, 0].unsqueeze(0).unsqueeze(-1).expand(P.shape[0], -1, 1).to(dev)
        tmin = phys_a["tmin"][:, 0].view(1, -1, 1).expand(P.shape[0], -1, 1).to(dev)
        tmax = phys_a["tmax"][:, 0].view(1, -1, 1).expand(P.shape[0], -1, 1).to(dev)
        T3 = T.unsqueeze(-1)
        P3 = P.unsqueeze(-1)
        PET3 = PET.unsqueeze(-1)
        doy3 = doy.unsqueeze(-1)
        A = part_a(combo, P3, PET3, doy3, T3, w_phen, tmin, tmax)
        results[key] = {"part_a": A}

        # Part B
        handler_b = build_handler(build_screen_config(cls, sem, root))
        for m in handler_b.model_dict.values():
            m.eval()
        B = part_b(combo, handler_b, jac_sample, n_attr)
        results[key]["part_b"] = B
        print(f"  A: ip_range={A['ip_shape_range'][0]:.4f} precap={A['ip_shape_range_precap'][0]:.4f} "
              f"|dip/dk|={A['dip_dshape'][0]:.2e} pet_cap={A['pet_cap_bind_frac']:.3f} "
              f"p_cap={A['p_cap_bind_frac']:.3f}")
        print(f"  B: |cos(w,k)|={B['abs_cos_dQ_dw_dshape']:.3f} |cos(w,p)|={B['abs_cos_dQ_dw_dphase']:.3f} "
              f"|dQ/dw|={B['dQ_dw_int'][0]:.2e} |dQ/dk|={B['dQ_dshape'][0]:.2e} |dQ/dp|={B['dQ_dphase'][0]:.2e} "
              f"cond={B['gram_cond_median']:.2e}")
        del handler_a, handler_b, params_a
        torch.cuda.empty_cache()

    (root / "screen_summary.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[screen] summary -> {root / 'screen_summary.json'}")


if __name__ == "__main__":
    main()
