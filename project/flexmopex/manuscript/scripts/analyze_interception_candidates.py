#!/usr/bin/env python3
"""Training diagnostics for the interception candidate E-S0 run (Phase 7).

From the saved E-S0 checkpoints (epoch 0/1/2/5/10) and the canonical test
evaluation:

  * per-basin gates (w_phen, w_int, w_snow, w_sub): mean/median/IQR, active
    fractions; per-basin w_int saved
  * internal interception parameters kappa (= alpha slot) and phi (= is_time
    slot): median/IQR; seasonal-gate calendar mean sanity (should be ~0.5)
  * loss decomposition (total / predictive / AIC) on the fixed diagnostic batch
  * gradient / identifiability diagnostics on the fixed diagnostic batch:
    |dL/dw_int|, |dL/dkappa|, |dL/dphi|, zero-gradient fractions,
    |cos(dQ/dw_int, dQ/dkappa)|, |cos(dQ/dw_int, dQ/dphi)| (absolute),
    and the raw gate-logit gradient dL/d(logit_w_int) to separate process
    sensitivity from Gumbel/softmax saturation
  * NSE/KGE sanity from the canonical test split

Gradients use the official production descale and the eval-mode (softmax)
gate path, deterministic; identical machinery to the 2x2 analysis.
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
    apply_runtime_overrides, parse_args, _build_data_loader, _build_loss,
)
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402

GATE_NAMES = ["w_phen", "w_int", "w_snow", "w_sub"]
DIAG_BASINS = list(range(32))
DIAG_T0, DIAG_T1 = 365, 1095
EPOCHS = [0, 1, 2, 5, 10]


def qstats(x: torch.Tensor) -> dict:
    x = x.detach().float()
    q25, q75 = torch.quantile(x, 0.25), torch.quantile(x, 0.75)
    return {"mean": float(x.mean()), "median": float(x.median()),
            "iqr": float(q75 - q25), "q25": float(q25), "q75": float(q75)}


def build_handler(config: dict) -> FlexMopexModelHandler:
    cfg = dict(config)
    cfg["mode"] = "train"
    return FlexMopexModelHandler(cfg, verbose=False)


def epoch_gate_params(handler, config, train_dataset) -> dict:
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    n_attr = len(config["model"]["nn"]["attributes"])
    attrs = train_dataset["xc_nn_norm"][0, :, -n_attr:].to(config["device"])
    with torch.no_grad():
        p = nn({"c_nn_norm": attrs})
        w = F.softmax(p["weights"].view(attrs.shape[0], 4, 2).clamp(min=-10.0, max=10.0), dim=-1)[..., 1]
        phys = phy._descale_mopex_params(p["params"])
    out = {"gates": {g: qstats(w[:, i]) for i, g in enumerate(GATE_NAMES)},
           "w_int_per_basin": w[:, 1].detach().cpu().numpy()}
    for i, g in enumerate(GATE_NAMES):
        out["gates"][g]["frac>0.01"] = float((w[:, i] > 0.01).float().mean())
        out["gates"][g]["frac>0.1"] = float((w[:, i] > 0.1).float().mean())
    out["kappa"] = qstats(phys["alpha"][:, 0])     # slot 8 semantics = kappa
    out["phi"] = qstats(phys["is_time"][:, 0])     # slot 9 semantics = phi
    # calendar-mean sanity of the linear gate over a full year
    import project.flexmopex.models.mopex_core_candidates as cand
    grid = torch.linspace(1.0, 365.0, 365, device=config["device"])
    s = cand.season_linear(grid.view(-1, 1), phys["alpha"][:, 0], phys["is_time"][:, 0])
    out["gate_calendar_mean"] = float(s.mean())
    return out


def diagnostic_sample(train_dataset, device: str) -> dict:
    n_attr = train_dataset["xc_nn_norm"].shape[-1] - 3
    return {
        "x_phy": train_dataset["x_phy"][DIAG_T0:DIAG_T1, DIAG_BASINS, :].to(device),
        "doy": train_dataset["doy"][DIAG_T0:DIAG_T1, DIAG_BASINS, :].to(device),
        "c_nn_norm": train_dataset["xc_nn_norm"][0, DIAG_BASINS, -n_attr:].to(device),
        "target": train_dataset["target"][DIAG_T0 + 365:DIAG_T1, DIAG_BASINS, :].to(device),
        "batch_sample": np.asarray(DIAG_BASINS, dtype=np.int64),
    }


def diagnostic_forward(handler, sample) -> dict:
    """Eval-mode grad-enabled forward mirroring the phy forward exactly;
    returns outputs plus real-graph-node coordinates (full tensors only)."""
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    params = nn(sample)
    logits = params["weights"].view(sample["c_nn_norm"].shape[0], 4, 2)
    logits = torch.clamp(logits, min=-10.0, max=10.0)
    weights_on = F.softmax(logits, dim=-1)[..., 1]
    mopex_params = phy._descale_mopex_params(params["params"])
    routing = phy._descale_routing_params(params["gamma_uh"])
    P, T, PET, doy, n_steps, n_grid = phy._prepare_forcings(sample)
    Q_mopex = phy._run_weighted_loop(P, T, PET, doy, mopex_params, weights_on, n_steps, n_grid)
    Qrouted = phy._apply_routing(Q_mopex.mean(-1), routing)
    out = {"streamflow": Qrouted}
    for i, name in enumerate(phy.weight_names):
        out[name] = weights_on[:, i].view(1, -1, 1).expand(Q_mopex.shape[0], -1, -1)
    coords = {
        "weights_on": weights_on,               # (B, 4)
        "kappa": mopex_params["alpha"],         # (B, nmul) slot 8
        "phi": mopex_params["is_time"],         # (B, nmul) slot 9
        "weights_raw": params["weights"],       # (B, 8) raw gate logits
    }
    return out, coords


def gradient_diagnostics(handler, loss_func, sample) -> dict:
    out, coords = diagnostic_forward(handler, sample)
    q = out["streamflow"]
    target = sample["target"]
    n = min(q.shape[0], target.shape[0])
    weights = {g: out[g] for g in GATE_NAMES}
    loss = loss_func(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=weights)

    def grad_of(y, x):
        g = torch.autograd.grad(y, x, retain_graph=True, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(x)

    gL_w = grad_of(loss, coords["weights_on"])[:, 1]              # (B,)
    gL_k = grad_of(loss, coords["kappa"]).sum(-1)                 # (B,)
    gL_p = grad_of(loss, coords["phi"]).sum(-1)                   # (B,)
    gQ_w = grad_of(q[:n].sum(), coords["weights_on"])[:, 1]       # (B,)
    gQ_k = grad_of(q[:n].sum(), coords["kappa"]).sum(-1)          # (B,)
    gQ_p = grad_of(q[:n].sum(), coords["phi"]).sum(-1)            # (B,)
    # raw gate-logit gradient for w_int (diagnostic: process vs saturation)
    gL_logit = grad_of(loss, coords["weights_raw"]).view(coords["weights_raw"].shape[0], 4, 2)[:, 1, 1]

    def q(x):
        x = x.detach().float().reshape(-1)
        return (float(x.median()), float(torch.quantile(x, 0.25)), float(torch.quantile(x, 0.75)))

    def zfrac(x):
        return float((x.abs() < 1e-12).float().mean())

    del q, loss, out
    torch.cuda.empty_cache()
    return {
        "median_abs_dL_dw_int": float(gL_w.abs().median()),
        "median_abs_dL_dkappa": float(gL_k.abs().median()),
        "median_abs_dL_dphi": float(gL_p.abs().median()),
        "zero_frac_dL_dw_int": zfrac(gL_w),
        "zero_frac_dL_dkappa": zfrac(gL_k),
        "zero_frac_dL_dphi": zfrac(gL_p),
        "abs_cos_dQ_dw_dkappa": float(F.cosine_similarity(gQ_w, gQ_k, dim=0).abs().mean()),
        "abs_cos_dQ_dw_dphi": float(F.cosine_similarity(gQ_w, gQ_p, dim=0).abs().mean()),
        "median_abs_dQ_dw_int": float(gQ_w.abs().median()),
        "median_abs_dQ_dkappa": float(gQ_k.abs().median()),
        "median_abs_dQ_dphi": float(gQ_p.abs().median()),
        "median_abs_dL_dlogit_w_int": float(gL_logit.abs().median()),
    }


def loss_decomposition(handler, loss_func, sample) -> dict:
    out, _ = diagnostic_forward(handler, sample)
    q = out["streamflow"]
    target = sample["target"]
    n = min(q.shape[0], target.shape[0])
    weights = {g: out[g] for g in GATE_NAMES}
    total = float(loss_func(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=weights))
    loss_func.aic_alpha = 0.0
    predictive = float(loss_func(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=weights))
    loss_func.aic_alpha = 0.01
    return {"total": total, "predictive": predictive, "aic_term": total - predictive}


def test_metrics(arm_dir: Path) -> dict:
    m = arm_dir / "test1995-2010_Ep10" / "metrics_agg.json"
    if not m.exists():
        return {"nse_median": None, "kge_median": None}
    d = json.loads(m.read_text())
    med = d.get("median", d)
    return {"nse_median": med.get("nse", med.get("NSE")),
            "kge_median": med.get("kge", med.get("KGE"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0.yaml")
    ap.add_argument("--output-root", default="results/intercept_candidates")
    ap.add_argument("--run-name", default="E_S0")
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.output_root)
    arm_dir = root / args.run_name

    cli = parse_args(["--config", args.config, "--gpu-id", str(args.gpu_id),
                      "--output-root", str(args.output_root), "--run-name", args.run_name])
    config = load_config(args.config)
    apply_runtime_overrides(config, cli, config_path=args.config)
    config["mode"] = "train"
    if str(config["device"]).startswith("cuda"):
        torch.cuda.set_device(config["device"])

    dl = _build_data_loader(config)
    td = dl.train_dataset
    loss_func = _build_loss(config, td)
    sample = diagnostic_sample(td, config["device"])
    handler = build_handler(config)

    entry = {"run": args.run_name, "model": config["model"]["phy"]["name"],
             "semantics": config["model"]["phy"].get("interception_semantics"),
             "test": test_metrics(arm_dir)}
    for epoch in EPOCHS:
        try:
            handler.load_model(epoch)
        except FileNotFoundError as e:
            print(f"  [warn] epoch {epoch} checkpoint missing: {e}")
            continue
        for m in handler.model_dict.values():
            m.eval()
        ep = epoch_gate_params(handler, config, td)
        entry[f"ep{epoch}_gates"] = {g: ep["gates"][g] for g in GATE_NAMES}
        entry[f"ep{epoch}_kappa"] = ep["kappa"]
        entry[f"ep{epoch}_phi"] = ep["phi"]
        entry[f"ep{epoch}_gate_calendar_mean"] = ep["gate_calendar_mean"]
        np.save(arm_dir / f"w_int_ep{epoch}.npy", ep["w_int_per_basin"])
        entry[f"ep{epoch}_grad"] = gradient_diagnostics(handler, loss_func, sample)
        entry[f"ep{epoch}_loss"] = loss_decomposition(handler, loss_func, sample)
        print(f"[diag] epoch {epoch} done", flush=True)

    (root / "candidate_analysis.json").write_text(json.dumps(entry, indent=2))

    rows = []
    for ep in EPOCHS:
        g = entry.get(f"ep{ep}_gates", {})
        gr = entry.get(f"ep{ep}_grad", {})
        rows.append({
            "epoch": ep,
            "w_int_median": g.get("w_int", {}).get("median"),
            "w_int_iqr": g.get("w_int", {}).get("iqr"),
            "w_int_frac>0.01": g.get("w_int", {}).get("frac>0.01"),
            "w_int_frac>0.1": g.get("w_int", {}).get("frac>0.1"),
            "w_snow_median": g.get("w_snow", {}).get("median"),
            "w_sub_median": g.get("w_sub", {}).get("median"),
            "w_phen_median": g.get("w_phen", {}).get("median"),
            "kappa_median": entry.get(f"ep{ep}_kappa", {}).get("median"),
            "phi_median": entry.get(f"ep{ep}_phi", {}).get("median"),
            "|dL/dw_int|": gr.get("median_abs_dL_dw_int"),
            "|dL/dkappa|": gr.get("median_abs_dL_dkappa"),
            "|dL/dphi|": gr.get("median_abs_dL_dphi"),
            "|cos(dQ/dw,dQ/dkappa)|": gr.get("abs_cos_dQ_dw_dkappa"),
            "|cos(dQ/dw,dQ/dphi)|": gr.get("abs_cos_dQ_dw_dphi"),
            "|dL/dlogit_w_int|": gr.get("median_abs_dL_dlogit_w_int"),
            "loss_total": entry.get(f"ep{ep}_loss", {}).get("total"),
            "loss_aic": entry.get(f"ep{ep}_loss", {}).get("aic_term"),
        })
    with (arm_dir / "epoch_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[diag] summary -> {root / 'candidate_analysis.json'}")
    print(f"[diag] table   -> {arm_dir / 'epoch_table.csv'}")


if __name__ == "__main__":
    main()
