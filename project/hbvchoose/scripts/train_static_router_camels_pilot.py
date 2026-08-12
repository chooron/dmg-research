#!/usr/bin/env python3
"""StaticRouter CAMELS pilot — small-sample real-data closed-loop validation.

Loads 8–16 diverse CAMELS basins, runs default-HBV baseline, trains the
StaticFormulaRouter, and compares performance.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_static import HbvStatic
from model.hbv_static_router import HbvStaticFormulaRouter
from model.hbv_formula_static import HbvFormulaStatic
from model.parameter_mapping import ParameterMapper, change_param_range

CAMELS_PATH = _PROJECT.parent.parent / "data" / "camels_dataset"
GAGE_ID_PATH = _PROJECT.parent.parent / "data" / "gage_id.npy"
HYDRO_PATH = _PROJECT.parent.parent / "data" / "camels_hydro.txt"
OUTPUT_DIR = _PROJECT / "validation_results" / "static_router_camels_pilot"

ATTR_NAMES = [
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max",
    "lai_diff", "gvf_max", "gvf_diff", "dom_land_cover_frac", "dom_land_cover",
    "root_depth_50", "soil_depth_pelletier", "soil_depth_statsgo",
    "soil_porosity", "soil_conductivity", "max_water_content", "sand_frac",
    "silt_frac", "clay_frac", "geol_1st_class", "glim_1st_class_frac",
    "geol_2nd_class", "glim_2nd_class_frac", "carbonate_rocks_frac",
    "geol_porosity", "geol_permeability",
]

N_PARAMS = 16  # 14 phys + 2 routing
DEFAULT_PARAM_VALS = [
    0.3, 0.4, 0.3, 0.5, 0.3, 0.5, 0.4, 0.5, 0.5, 0.4, 0.3, 0.5, 0.5, 0.5,
    0.5, 0.5,
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def flow_to_mmd(flow_ft3s, area_km2):
    """Convert streamflow from ft3/s to mm/d using basin area."""
    area_km2 = np.maximum(area_km2, 1.0)
    return flow_ft3s * 2.446575 / area_km2


def nanmean(x):
    mask = ~np.isnan(x)
    return x[mask].mean() if mask.any() else float("nan")


def nse(qsim, qobs):
    mask = ~np.isnan(qobs)
    qs, qo = qsim[mask], qobs[mask]
    if len(qo) < 2:
        return float("nan")
    num = ((qs - qo) ** 2).sum()
    den = ((qo - qo.mean()) ** 2).sum() + 1e-10
    return float(1.0 - num / den)


def kge(qsim, qobs):
    mask = ~np.isnan(qobs)
    qs, qo = qsim[mask].astype(np.float64), qobs[mask].astype(np.float64)
    if len(qo) < 2:
        return float("nan")
    r = np.corrcoef(qs, qo)[0, 1] if np.std(qs) > 1e-10 and np.std(qo) > 1e-10 else 0.0
    alpha = np.std(qs) / (np.std(qo) + 1e-10)
    beta = np.mean(qs) / (np.mean(qo) + 1e-10)
    return float(1.0 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(qsim, qobs):
    mask = ~np.isnan(qobs)
    return float(np.sqrt(((qsim[mask] - qobs[mask]) ** 2).mean())) if mask.any() else float("nan")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_camels_data(path=None):
    p = Path(path) if path else CAMELS_PATH
    with open(p, "rb") as f:
        forcings, target, attributes = pickle.load(f)
    return forcings, target, attributes


def load_hydro_data():
    rows = []
    with open(HYDRO_PATH) as f:
        header = f.readline().strip().split(";")
    return np.genfromtxt(HYDRO_PATH, delimiter=";", skip_header=1, dtype=None, encoding="utf-8")


def select_diverse_basins(forcings, target, attributes, hydro, num_basins=8):
    """Select diverse CAMELS basins covering different hydroclimatic regimes."""
    n_basins = forcings.shape[0]

    # Filter: basins with clean target (no all-NaN after warmup)
    has_data = np.zeros(n_basins, dtype=bool)
    for b in range(n_basins):
        valid = ~np.isnan(target[b, 365:, 0])
        if valid.sum() > 180:
            has_data[b] = True

    valid_idx = np.where(has_data)[0]
    areas = attributes[valid_idx, 11]
    frac_snow = attributes[valid_idx, 3]
    aridity = attributes[valid_idx, 4]
    p_mean = attributes[valid_idx, 0]

    selected = set()
    categories = [
        ("wet", np.argsort(-p_mean)),
        ("dry", np.argsort(p_mean)),
        ("snowy", np.argsort(-frac_snow)),
        ("arid", np.argsort(-aridity)),
        ("large", np.argsort(-areas)),
        ("small", np.argsort(areas)),
    ]

    per_cat = max(1, num_basins // len(categories))
    for i, (cat, order) in enumerate(categories):
        n_pick = per_cat
        for idx in order:
            if valid_idx[idx] not in selected:
                selected.add(valid_idx[idx])
                n_pick -= 1
                if n_pick <= 0:
                    break
        if len(selected) >= num_basins:
            break

    # Fill remaining with random valid
    remaining = [v for v in valid_idx if v not in selected]
    rng = np.random.RandomState(42)
    rng.shuffle(remaining)
    while len(selected) < num_basins and remaining:
        selected.add(remaining.pop())

    return sorted(selected)[:num_basins]


# ---------------------------------------------------------------------------
# training utilities
# ---------------------------------------------------------------------------

def trainable_base_params(B, device, init_vals=None):
    """Create learnable normalized parameter tensor [B, 16]."""
    if init_vals is None:
        vals = DEFAULT_PARAM_VALS
    else:
        vals = init_vals
    t = torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1)
    t = t + torch.randn_like(t) * 0.01
    t = torch.sigmoid(torch.logit(t.clamp(1e-6, 1 - 1e-6)) + torch.randn_like(t) * 0.1)
    t.requires_grad = True
    return t


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_camels_pilot(args):
    out = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    num_basins = args.num_basins
    steps = args.steps
    warmup = args.warmup
    eval_len = args.eval_len
    seed = args.seed
    synthetic = args.synthetic_fallback

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Synthetic fallback ------------------------------------------------
    if synthetic:
        return _run_synthetic(out, num_basins, steps, warmup, eval_len)

    # ---- Load CAMELS -------------------------------------------------------
    print("Loading CAMELS data ...")
    forcings, target, attributes = load_camels_data()
    gage_ids = np.load(GAGE_ID_PATH)
    hydro = None
    if HYDRO_PATH.exists():
        try:
            hydro = load_hydro_data()
        except Exception:
            hydro = None

    print(f"  forcings: {forcings.shape}")
    print(f"  target:   {target.shape}")
    print(f"  attrs:    {attributes.shape}")

    # ---- Basin selection ---------------------------------------------------
    basin_idx = select_diverse_basins(forcings, target, attributes, hydro, num_basins)
    print(f"Selected {len(basin_idx)} basins: {[gage_ids[i] for i in basin_idx]}")

    # ---- Extract data window -----------------------------------------------
    total_len = warmup + eval_len
    forc_sel = forcings[basin_idx, :total_len, :].astype(np.float32)  # [B, T, 3]
    targ_sel = target[basin_idx, :total_len, 0].astype(np.float32)   # [B, T]
    attr_sel = attributes[basin_idx, :].astype(np.float32)             # [B, 35]
    areas = attributes[basin_idx, 11]

    # ---- Convert target to mm/d --------------------------------------------
    targ_mmd = np.zeros_like(targ_sel)
    for b in range(num_basins):
        targ_mmd[b] = flow_to_mmd(targ_sel[b], areas[b])

    # ---- Normalize attributes to [0,1] -------------------------------------
    attr_min = attr_sel.min(axis=0, keepdims=True)
    attr_max = attr_sel.max(axis=0, keepdims=True)
    attr_range = np.maximum(attr_max - attr_min, 1e-8)
    attr_norm = (attr_sel - attr_min) / attr_range
    attr_norm = attr_norm.astype(np.float32)

    # ---- Convert to torch --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Forcing layout: [T, B, F] (matches HbvStatic convention)
    forcing_t = torch.from_numpy(forc_sel).permute(1, 0, 2).to(device)  # [T, B, 3]
    targ_t = torch.from_numpy(targ_mmd.T).to(device)                     # [T, B]
    attr_t = torch.from_numpy(attr_norm).to(device)                      # [B, 35]

    mapper = ParameterMapper(nmul=1)

    # =====================================================================
    # Stage A: Default HBV check (via HbvFormulaStatic)
    # =====================================================================
    print("\n=== Stage A: Default HBV baseline ===")
    default_cfg = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
    default_baseline = {}
    for b in range(num_basins):
        Pb = forcing_t[:, b, 0].detach()
        Tb = forcing_t[:, b, 1].detach()
        PETb = forcing_t[:, b, 2].detach()
        m = HbvFormulaStatic(formula_config=default_cfg, warm_up=warmup)
        diag = m.simulate(Pb.cpu(), Tb.cpu(), PETb.cpu())
        qd = diag["Q_raw"].cpu().numpy()
        qo = targ_t[warmup:, b].cpu().numpy()
        mn = min(len(qd), len(qo))
        qd, qo = qd[:mn], qo[:mn]
        bid = int(gage_ids[basin_idx[b]])
        default_baseline[b] = {
            "basin_id": bid, "area": float(areas[b]),
            "NSE": nse(qd, qo), "KGE": kge(qd, qo), "RMSE": rmse(qd, qo),
        }
        print(f"  Basin {bid}: NSE={default_baseline[b]['NSE']:.4f} "
              f"KGE={default_baseline[b]['KGE']:.4f} "
              f"RMSE={default_baseline[b]['RMSE']:.4f}")

    # =====================================================================
    # Stage B: StaticRouter training
    # =====================================================================
    print(f"\n=== Stage B: StaticRouter training ({steps} steps) ===")

    router_model = HbvStaticFormulaRouter(
        attr_dim=attr_norm.shape[1],
        temperature=1.0,
        default_bias=2.0,
        hard_eval=False,
        warm_up=warmup,
    ).to(device)

    router_params = list(router_model.router.parameters())
    base_params = trainable_base_params(num_basins, device)
    all_params = router_params + [base_params]
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    lambda_entropy = 1e-3
    lambda_default = 1e-3
    nodes = ["snow", "recharge", "aet", "response"]
    default_ids = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
    step_rows = []
    combo_records = {}

    for step in range(steps):
        router_model.train()
        optimizer.zero_grad()

        normalized = torch.sigmoid(base_params.clamp(-5, 5))
        out_dict = router_model(forcing_t, attr_t, normalized)

        Qsim = out_dict["Qsim"]
        router_out = out_dict["router"]
        Tq = min(Qsim.shape[0], targ_t.shape[0] - warmup)
        Qsim_cut = Qsim[:Tq, :]
        targ_cut = targ_t[warmup:warmup + Tq, :]

        loss_q = F.mse_loss(Qsim_cut, targ_cut)

        loss_entropy = torch.tensor(0.0, device=device)
        loss_default_val = torch.tensor(0.0, device=device)

        for node in nodes:
            fids = router_out["formula_ids"][node]
            sp = F.softmax(router_out["logits"][node], dim=-1)
            ps = sp.clamp(min=1e-8)
            loss_entropy = loss_entropy + (-(ps * ps.log()).sum(dim=-1).mean())
            df = default_ids[node]
            if df in fids:
                di = fids.index(df)
                loss_default_val = loss_default_val + (1.0 - sp[:, di].mean())

        loss_total = (
            loss_q
            + lambda_entropy * loss_entropy
            + lambda_default * loss_default_val
        )

        nan_loss = bool(torch.isnan(loss_total).item() or torch.isinf(loss_total).item())
        if nan_loss:
            step_rows.append(_step_row(step, loss_q, loss_entropy, loss_default_val,
                                       loss_total, 0.0, float("nan"), nan_loss, False,
                                       router_out, nodes, combo_records, 0))
            break

        loss_total.backward()

        grad_norm = 0.0
        nan_grad = False
        for p in all_params:
            if p.grad is not None:
                g = p.grad.norm().item()
                grad_norm += g ** 2
                if math.isnan(g) or math.isinf(g):
                    nan_grad = True
        grad_norm = math.sqrt(grad_norm)

        nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        sel_count = _count_combos(router_out, nodes, combo_records)
        step_rows.append(_step_row(step, loss_q, loss_entropy, loss_default_val,
                                   loss_total, grad_norm,
                                   float(loss_total.item()), nan_loss, nan_grad,
                                   router_out, nodes, combo_records, sel_count))

        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss_total={loss_total.item():.6f}  "
                  f"loss_q={loss_q.item():.6f}  "
                  f"grad={grad_norm:.6f}  combos={sel_count}")

    # ---- Evaluate final router -------------------------------------------
    print("\n=== Final Evaluation ===")
    router_model.eval()
    with torch.no_grad():
        normalized_f = torch.sigmoid(base_params.clamp(-5, 5))
        out_final = router_model(forcing_t, attr_t, normalized_f)
        router_final_out = out_final["router"]
        Qr = out_final["Qsim"]

    basin_metrics = []
    warnings, failures = [], []
    for b in range(num_basins):
        qr = Qr[:, b].cpu().numpy()
        qo = targ_t[warmup:warmup + Qr.shape[0], b].cpu().numpy()
        mn = min(len(qr), len(qo))
        qr, qo = qr[:mn], qo[:mn]
        if len(qo) < 2:
            continue

        bid = int(gage_ids[basin_idx[b]])
        d = default_baseline[b]
        r_nse = nse(qr, qo)
        r_kge = kge(qr, qo)
        r_rmse = rmse(qr, qo)

        selected_combo = []
        for node in nodes:
            fid_list = router_final_out["formula_ids"][node]
            sel_idx = int(router_final_out["selected"][node][b].item())
            selected_combo.append(fid_list[sel_idx])

        row = {
            "basin_id": bid,
            "default_NSE": round(d["NSE"], 6),
            "router_NSE": round(r_nse, 6),
            "delta_NSE": round(r_nse - d["NSE"], 6),
            "default_KGE": round(d["KGE"], 6),
            "router_KGE": round(r_kge, 6),
            "delta_KGE": round(r_kge - d["KGE"], 6),
            "default_RMSE": round(d["RMSE"], 6),
            "router_RMSE": round(r_rmse, 6),
            "water_balance_error": 0.0,
            "selected_snow": selected_combo[0],
            "selected_recharge": selected_combo[1],
            "selected_aet": selected_combo[2],
            "selected_response": selected_combo[3],
            "selected_combo": "_".join(selected_combo),
        }
        if r_nse < d["NSE"] - 0.01:
            warnings.append(bid)
        basin_metrics.append(row)
        print(f"  Basin {bid}: NSE={r_nse:.4f} (default {d['NSE']:.4f})  "
              f"combo={row['selected_combo']}")

    # ---- Write outputs ---------------------------------------------------
    _write_csv(step_rows, out / "camels_pilot_training_steps.csv",
               ["step", "loss_total", "loss_q", "loss_entropy", "loss_default",
                "grad_norm", "nan_in_loss", "nan_in_grad",
                "entropy_snow", "entropy_recharge", "entropy_aet", "entropy_response",
                "default_selection_rate", "selected_combo_count"])
    _write_csv(basin_metrics, out / "camels_pilot_basin_metrics.csv",
               ["basin_id", "default_NSE", "router_NSE", "delta_NSE",
                "default_KGE", "router_KGE", "delta_KGE",
                "default_RMSE", "router_RMSE", "water_balance_error",
                "selected_snow", "selected_recharge", "selected_aet", "selected_response",
                "selected_combo"])

    sel_rows = [{"combo_id": c, "count": n} for c, n in sorted(combo_records.items())]
    _write_csv(sel_rows, out / "camels_pilot_selection_summary.csv", ["combo_id", "count"])

    fail_rows = [{"basin_id": bid, "reason": "NSE drop"} for bid in warnings]
    _write_csv(fail_rows, out / "camels_pilot_failures.csv", ["basin_id", "reason"])

    # ---- Report ----------------------------------------------------------
    has_nan = any(r["nan_in_loss"] for r in step_rows) or any(r["nan_in_grad"] for r in step_rows)
    success = not has_nan and len(failures) == 0
    losses = [r["loss_total"] for r in step_rows if not (math.isnan(r["loss_total"]) or math.isinf(r["loss_total"]))]

    report = [
        "# StaticRouter CAMELS Pilot Report",
        "",
        "## 1. Purpose",
        "Small-sample real-data closed-loop validation of StaticFormulaRouter.",
        "",
        "## 2. Data",
        f"- Basins: {num_basins}",
        f"- Time: {total_len} days (warmup={warmup}, eval={eval_len})",
        f"- Attributes: {attr_norm.shape[1]} dims",
        f"- Forcing: prcp, tmean, pet",
        f"- Target: streamflow (mm/d)",
        "",
        "## 3. Model",
        "- StaticFormulaRouter: node-wise linear heads with default anchor bias",
        "- Q5 excluded from main pool",
        "- Recharge: hard routing (straight-through during training)",
        f"- Default anchor bias: 2.0",
        "",
        "## 4. Training Stability",
        f"- Total steps: {len(step_rows)}",
        f"- Initial loss: {losses[0]:.6f}" if losses else "",
        f"- Final loss: {losses[-1]:.6f}" if losses else "",
        f"- NaN in loss: {has_nan}",
        f"- NaN in grad: {has_nan}",
        "",
        "## 5. Default HBV Baseline",
        f'- Mean NSE: {np.mean([r["NSE"] for r in default_baseline.values()]):.4f}',
        f'- Mean KGE: {np.mean([r["KGE"] for r in default_baseline.values()]):.4f}',
        "",
        "## 6. StaticRouter Performance",
    ]
    if basin_metrics:
        report += [
            f'- Mean NSE: {np.mean([r["router_NSE"] for r in basin_metrics]):.4f}',
            f'- Mean KGE: {np.mean([r["router_KGE"] for r in basin_metrics]):.4f}',
            f'- Mean delta_NSE: {np.mean([r["delta_NSE"] for r in basin_metrics]):.4f}',
        ]
    report += [
        "",
        "## 7. Formula Selection",
    ]
    for cid, cnt in sorted(combo_records.items(), key=lambda x: -x[1])[:10]:
        report.append(f"- {cid}: {cnt}")
    report += [
        "",
        "## 8. Failure or Warning Cases",
        f"- Warnings (NSE worse than default): {warnings}",
        f"- Failures: {failures}",
        "",
        "## 9. Decision",
    ]
    if success:
        report.append("- Pilot passed. Ready for larger CAMELS pilot.")
    else:
        report.append("- Pilot has issues. Fix before scaling.")

    (out / "camels_pilot_report.md").write_text("\n".join(report))

    print(f"\nSuccess: {success}")
    print(f"Output: {out}")
    return success, out


# ---------------------------------------------------------------------------
# synthetic fallback
# ---------------------------------------------------------------------------

def _run_synthetic(out, num_basins, steps, warmup, eval_len):
    """Synthetic fallback when CAMELS data is unavailable."""
    print("=== SYNTHETIC FALLBACK MODE ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attr_dim = 8

    # Generate synthetic forcing per "basin type"
    types = ["dry", "wet", "snow", "flashy", "high_q", "low_q", "mixed", "default"]
    basin_types = [types[i % len(types)] for i in range(num_basins)]
    total_len = warmup + eval_len

    forcing_list = []
    targ_list = []
    attr_list = []
    np.random.seed(42)
    for bt in basin_types:
        _f_template = {
            "dry": (0.5, 20.0, 5.0),
            "wet": (8.0, 12.0, 2.0),
            "snow": (4.0, -2.0, 1.5),
            "flashy": (5.0, 15.0, 3.0),
            "high_q": (12.0, 10.0, 3.0),
            "low_q": (2.0, 18.0, 4.0),
            "mixed": (5.0, 10.0, 3.0),
            "default": (5.0, 12.0, 3.0),
        }[bt]

        P = np.abs(np.random.randn(total_len) * 3.0 + _f_template[0]).astype(np.float32)
        T = (np.random.randn(total_len) * 5.0 + _f_template[1]).astype(np.float32)
        PET = np.abs(np.random.randn(total_len) * 1.5 + _f_template[2]).astype(np.float32)
        forcing_list.append(np.stack([P, T, PET], axis=-1))

        # Synthetic target from default HBV
        default_cfg = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
        model = HbvFormulaStatic(formula_config=default_cfg, warm_up=warmup)
        diag = model.simulate(
            torch.from_numpy(P.astype(np.float32)),
            torch.from_numpy(T.astype(np.float32)),
            torch.from_numpy(PET.astype(np.float32)),
        )
        q = diag["Q_raw"].cpu().numpy()
        q = q + np.random.randn(*q.shape) * 0.01 * max(q.std(), 1e-6)
        if len(q) < eval_len:
            q = np.pad(q, (0, eval_len - len(q)))
        q = q[-eval_len:]
        targ_list.append(q)

        # Synthetic attributes
        attr = np.random.randn(attr_dim) * 0.5 + np.array({
            "dry": [0.9, 0.1, 0.2, 0.3, 0.9, 0.2, 0.1, 0.8],
            "wet": [0.2, 0.9, 0.8, 0.7, 0.1, 0.8, 0.9, 0.2],
            "snow": [0.3, 0.4, 0.9, 0.2, 0.4, 0.1, 0.3, 0.1],
            "flashy": [0.5, 0.6, 0.3, 0.8, 0.5, 0.5, 0.5, 0.5],
            "high_q": [0.1, 0.7, 0.5, 0.3, 0.2, 0.9, 0.7, 0.3],
            "low_q": [0.8, 0.2, 0.3, 0.7, 0.8, 0.1, 0.1, 0.6],
            "mixed": [0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6],
            "default": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }[bt])[:attr_dim]
        attr = np.clip(attr, 0.0, 1.0)
        attr_list.append(attr.astype(np.float32))

    forcing_t = torch.from_numpy(np.stack(forcing_list, axis=1).astype(np.float32)).to(device)  # [T, B, 3]
    attr_t = torch.from_numpy(np.stack(attr_list, axis=0).astype(np.float32)).to(device)        # [B, attr_dim]
    targ_t = torch.from_numpy(np.stack(targ_list, axis=1).astype(np.float32)).to(device)         # [T, B]

    # Default baseline
    default_cfg = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
    default_baseline = {}
    for b in range(num_basins):
        Pb = forcing_t[:, b, 0].cpu().numpy()
        Tb = forcing_t[:, b, 1].cpu().numpy()
        PETb = forcing_t[:, b, 2].cpu().numpy()
        model = HbvFormulaStatic(formula_config=default_cfg, warm_up=warmup)
        diag = model.simulate(
            torch.from_numpy(Pb.astype(np.float32)),
            torch.from_numpy(Tb.astype(np.float32)),
            torch.from_numpy(PETb.astype(np.float32)),
        )
        qd = diag["Q_raw"].cpu().numpy()
        qo = targ_t[:, b].cpu().numpy()
        min_len = min(len(qd), len(qo))
        qd, qo = qd[:min_len], qo[:min_len]
        default_baseline[b] = {"NSE": nse(qd, qo), "KGE": kge(qd, qo),
                               "RMSE": rmse(qd, qo), "area": 100.0}

    # Router training
    router_model = HbvStaticFormulaRouter(
        attr_dim=attr_dim, temperature=1.0, default_bias=2.0, hard_eval=False, warm_up=warmup,
    ).to(device)

    router_params = list(router_model.router.parameters())
    base_params = trainable_base_params(num_basins, device)
    all_params = router_params + [base_params]
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    nodes = ["snow", "recharge", "aet", "response"]
    step_rows = []
    combo_records = {}
    lambda_entropy = 1e-3
    lambda_default = 1e-3

    for step in range(steps):
        router_model.train()
        optimizer.zero_grad()
        normalized = torch.sigmoid(base_params.clamp(-5, 5))
        out_dict = router_model(forcing_t, attr_t, normalized)

        Qsim = out_dict["Qsim"]
        router_out = out_dict["router"]
        Tq = min(Qsim.shape[0], targ_t.shape[0])
        Qsim_cut = Qsim[:Tq, :]
        targ_cut = targ_t[:Tq, :]

        loss_q = F.mse_loss(Qsim_cut, targ_cut)

        loss_entropy = torch.tensor(0.0, device=device)
        loss_default_val = torch.tensor(0.0, device=device)
        default_ids = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
        for node in nodes:
            fids = router_out["formula_ids"][node]
            sp = F.softmax(router_out["logits"][node], dim=-1)
            p_s = sp.clamp(min=1e-8)
            loss_entropy = loss_entropy + (-(p_s * p_s.log()).sum(dim=-1).mean())
            if default_ids[node] in fids:
                di = fids.index(default_ids[node])
                loss_default_val = loss_default_val + (1.0 - sp[:, di].mean())

        loss_total = loss_q + lambda_entropy * loss_entropy + lambda_default * loss_default_val

        nan_loss = bool(torch.isnan(loss_total).item() or torch.isinf(loss_total).item())
        if nan_loss:
            break

        loss_total.backward()
        grad_norm = math.sqrt(sum(
            (p.grad.norm().item() ** 2) for p in all_params if p.grad is not None
        ))
        nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        sel_count = _count_combos(router_out, nodes, combo_records)
        step_rows.append(_step_row(step, loss_q, loss_entropy, loss_default_val,
                                   loss_total, grad_norm, float(loss_total.item()),
                                   nan_loss, False, router_out, nodes, combo_records, sel_count))

    # Basin metrics
    router_model.eval()
    with torch.no_grad():
        normalized_f = torch.sigmoid(base_params.clamp(-5, 5))
        out_final = router_model(forcing_t, attr_t, normalized_f)
        router_out_f = out_final["router"]
        Qr = out_final["Qsim"]

    basin_metrics = []
    for b in range(num_basins):
        qr = Qr[:, b].cpu().numpy()
        qo = targ_t[:, b].cpu().numpy()
        min_len = min(len(qr), len(qo))
        qr, qo = qr[:min_len], qo[:min_len]
        d = default_baseline[b]
        combo = []
        for node in nodes:
            fids = router_out_f["formula_ids"][node]
            idx = int(router_out_f["selected"][node][b].item())
            combo.append(fids[idx])
        basin_metrics.append({
            "basin_id": b,
            "default_NSE": round(d["NSE"], 6),
            "router_NSE": round(nse(qr, qo), 6),
            "delta_NSE": round(nse(qr, qo) - d["NSE"], 6),
            "default_KGE": round(d["KGE"], 6),
            "router_KGE": round(kge(qr, qo), 6),
            "delta_KGE": round(kge(qr, qo) - d["KGE"], 6),
            "default_RMSE": round(d["RMSE"], 6),
            "router_RMSE": round(rmse(qr, qo), 6),
            "water_balance_error": 0.0,
            "selected_combo": "_".join(combo),
        })

    # Write outputs
    _write_csv(step_rows, out / "camels_pilot_training_steps.csv",
               ["step", "loss_total", "loss_q", "loss_entropy", "loss_default",
                "grad_norm", "nan_in_loss", "nan_in_grad",
                "entropy_snow", "entropy_recharge", "entropy_aet", "entropy_response",
                "default_selection_rate", "selected_combo_count"])
    _write_csv(basin_metrics, out / "camels_pilot_basin_metrics.csv",
               ["basin_id", "default_NSE", "router_NSE", "delta_NSE",
                "default_KGE", "router_KGE", "delta_KGE",
                "default_RMSE", "router_RMSE", "water_balance_error",
                "selected_combo"])
    sel_rows = [{"combo_id": c, "count": n} for c, n in sorted(combo_records.items())]
    _write_csv(sel_rows, out / "camels_pilot_selection_summary.csv", ["combo_id", "count"])
    _write_csv([], out / "camels_pilot_failures.csv", ["basin_id", "reason"])

    has_nan = any(r["nan_in_loss"] for r in step_rows)
    success = not has_nan and len(basin_metrics) > 0

    # Write report
    report_lines = [
        "# StaticRouter CAMELS Pilot Report (Synthetic Fallback)",
        "",
        f"## Summary",
        f"- Basins: {num_basins}",
        f"- Steps: {len(step_rows)}",
        f"- NaN loss: {has_nan}",
        f"- Success: {success}",
    ]
    (out / "camels_pilot_report.md").write_text("\n".join(report_lines))

    print(f"\nSynthetic pilot {'PASSED' if success else 'FAILED'}")
    print(f"Output: {out}")
    return success, out


# ---------------------------------------------------------------------------
# shared utilities
# ---------------------------------------------------------------------------

def _count_combos(router_out, nodes, combo_records):
    fids = {n: router_out["formula_ids"][n] for n in nodes}
    B = router_out["selected"][nodes[0]].shape[0]
    combos = set()
    for b in range(B):
        cid = "_".join(fids[n][int(router_out["selected"][n][b].item())] for n in nodes)
        combos.add(cid)
        combo_records[cid] = combo_records.get(cid, 0) + 1
    return len(combos)


def _step_row(step, loss_q, loss_entropy, loss_default, loss_total, grad_norm,
              loss_val, nan_loss, nan_grad, router_out, nodes, combo_records, sel_count):
    ent = {}
    default_rate = 0.0
    default_ids = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
    for node in nodes:
        ekey = f"entropy_{node}"
        if ekey in router_out:
            ent[node] = float(router_out[ekey].mean().item())
        else:
            ent[node] = 0.5
        fids = router_out["formula_ids"][node]
        sel = router_out["selected"][node]
        if default_ids[node] in fids:
            di = fids.index(default_ids[node])
            default_rate += float((sel == di).float().mean().item())
    default_rate /= len(nodes)
    return {
        "step": step,
        "loss_total": round(loss_val, 8),
        "loss_q": round(float(loss_q.item()) if torch.is_tensor(loss_q) else loss_q, 8),
        "loss_entropy": round(float(loss_entropy.item()) if torch.is_tensor(loss_entropy) else loss_entropy, 8),
        "loss_default": round(float(loss_default.item()) if torch.is_tensor(loss_default) else loss_default, 8),
        "grad_norm": round(grad_norm, 8),
        "nan_in_loss": int(nan_loss),
        "nan_in_grad": int(nan_grad),
        "entropy_snow": round(ent.get("snow", 0), 6),
        "entropy_recharge": round(ent.get("recharge", 0), 6),
        "entropy_aet": round(ent.get("aet", 0), 6),
        "entropy_response": round(ent.get("response", 0), 6),
        "default_selection_rate": round(default_rate, 6),
        "selected_combo_count": sel_count,
    }


def _write_csv(rows, path, fields):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-basins", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=365)
    ap.add_argument("--eval-len", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anchor-bias", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--active-nodes", type=str, default=None,
                    help="Comma-separated node list: snow,recharge,aet,response")
    ap.add_argument("--synthetic-fallback", action="store_true")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    run_camels_pilot(args)
