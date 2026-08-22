#!/usr/bin/env python3
"""Interception 2x2 diagnostics (Phases 6-10).

For each arm A/B/C/D, from saved checkpoints (epoch 0/5/10) and the canonical
test evaluation:

  * per-basin gates (w_phen, w_int, w_snow, w_sub): mean/median/IQR, active
    fractions, per-basin w_int saved
  * internal interception parameters alpha / is_time (+ decoupled annual-
    normalization sanity)
  * loss decomposition (total / predictive / AIC) on the fixed diagnostic batch
  * gradient / identifiability diagnostics on the fixed diagnostic batch:
    median |dL/dw_int|, |dL/dalpha|, |dL/dis_time|, zero-gradient fractions,
    cos(dQ/dw_int, dQ/dalpha)
  * NSE/KGE sanity from the canonical test split (metrics.json)
  * M4 ON/OFF external alignment (Spearman vs delta_NSE_int) if the reference
    file is available anywhere; otherwise reported unavailable
  * factorial contrasts + epoch-10 table + markdown report

Gradients use the official production descale (sigmoid -> hydrodl2
change_param_range) and the eval-mode (softmax) gate path, deterministic.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hydrodl2.core.calc import change_param_range  # noqa: E402

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import apply_runtime_overrides, parse_args  # noqa: E402
from project.flexmopex.run_model import _build_data_loader, _build_loss  # noqa: E402
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402

ARMS = ["A", "B", "C", "D"]
ARMS_DESC = {
    "A": "V0-original",
    "B": "V0-decoupled",
    "C": "V1-original",
    "D": "V1-decoupled",
}
GATE_NAMES = ["w_phen", "w_int", "w_snow", "w_sub"]
DIAG_BASINS = list(range(32))          # fixed deterministic basin subset
DIAG_T0, DIAG_T1 = 365, 1095           # 2-year window; warmup 365 -> 365 scored days
EPOCHS = [0, 5, 10]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def build_handler(config: dict) -> FlexMopexModelHandler:
    cfg = dict(config)
    cfg["mode"] = "train"  # do not auto-load any checkpoint
    return FlexMopexModelHandler(cfg, verbose=False)


def load_epoch_state(handler: FlexMopexModelHandler, epoch: int) -> None:
    handler.load_model(epoch)
    for model in handler.model_dict.values():
        model.eval()


def descale_mopex_params(phy, raw_params: torch.Tensor) -> dict[str, torch.Tensor]:
    """Official production descale (same code path as BaseMopex._descale_mopex_params)."""
    normalized = torch.sigmoid(raw_params).view(
        raw_params.shape[0], len(phy.mopex_param_names), phy.nmul
    )
    return {
        name: change_param_range(normalized[:, index, :], phy.param_bounds[name])
        for index, name in enumerate(phy.mopex_param_names)
    }


def qstats(x: torch.Tensor) -> dict:
    x = x.detach().float()
    q25, q75 = torch.quantile(x, 0.25), torch.quantile(x, 0.75)
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
    }


# ---------------------------------------------------------------------------
# per-epoch gate / parameter readout (eval mode, all 671 basins, real attributes)
# ---------------------------------------------------------------------------
def epoch_gate_params(handler, config, train_dataset) -> dict:
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    n_attr = len(config["model"]["nn"]["attributes"])
    attrs = train_dataset["xc_nn_norm"][0, :, -n_attr:].to(config["device"])
    with torch.no_grad():
        p = nn({"c_nn_norm": attrs})
        w = torch.softmax(p["weights"].view(attrs.shape[0], 4, 2).clamp(min=-10.0, max=10.0), dim=-1)[..., 1]
        phys = descale_mopex_params(phy, p["params"])
    out = {"gates": {g: qstats(w[:, i]) for i, g in enumerate(GATE_NAMES)},
           "w_int_per_basin": w[:, 1].detach().cpu().numpy()}
    out["gates"]["w_int"]["frac>0.01"] = float((w[:, 1] > 0.01).float().mean())
    out["gates"]["w_int"]["frac>0.1"] = float((w[:, 1] > 0.1).float().mean())
    for g in ("w_phen", "w_snow", "w_sub"):
        out["gates"][g]["frac>0.01"] = float((w[:, GATE_NAMES.index(g)] > 0.01).float().mean())
        out["gates"][g]["frac>0.1"] = float((w[:, GATE_NAMES.index(g)] > 0.1).float().mean())
    out["alpha"] = qstats(phys["alpha"][:, 0])
    out["is_time"] = qstats(phys["is_time"][:, 0])
    # decoupled annual-normalization sanity: mean over a full year of g_shape
    if "ecoupled" in type(phy).__name__:
        import project.flexmopex.models.mopex_core_v1 as v1
        grid = torch.linspace(1.0, 365.0, 365, device=config["device"])
        nm = v1.decoupled_norm_mean(phys["alpha"][:, 0], phys["is_time"][:, 0])
        shp = v1.decoupled_shape(grid.view(-1, 1), phys["alpha"][:, 0], phys["is_time"][:, 0], nm)
        out["decoupled_annual_mean"] = float(shp.mean())
    return out


# ---------------------------------------------------------------------------
# deterministic diagnostic batch
# ---------------------------------------------------------------------------
def diagnostic_sample(train_dataset, device: str) -> dict:
    b = torch.tensor(DIAG_BASINS)
    x = train_dataset["x_phy"][DIAG_T0:DIAG_T1, DIAG_BASINS, :].to(device)
    doy = train_dataset["doy"][DIAG_T0:DIAG_T1, DIAG_BASINS, :].to(device)
    n_attr = train_dataset["xc_nn_norm"].shape[-1] - 3
    attrs = train_dataset["xc_nn_norm"][0, DIAG_BASINS, -n_attr:].to(device)
    target = train_dataset["target"][DIAG_T0 + 365:DIAG_T1, DIAG_BASINS, :].to(device)
    return {
        "x_phy": x,
        "doy": doy,
        "c_nn_norm": attrs,
        "target": target,
        "batch_sample": np.asarray(DIAG_BASINS, dtype=np.int64),
    }


def diagnostic_forward(handler, sample) -> dict:
    """Eval-mode, grad-enabled forward that mirrors the phy forward exactly.

    The model's internal intermediates (weights_on, descale outputs) are
    recomputed here so that they are real autograd graph nodes: plain
    ``torch.autograd.grad`` cannot differentiate w.r.t. view/sibling tensors
    (e.g. ``out["w_int"][0, :, 0]``), only w.r.t. actual ancestors of the
    output.  This is the same code path the phy model executes (eval ->
    softmax gate path, official descale, weighted loop, routing).
    """
    import torch.nn.functional as F
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    params = nn(sample)
    logits = params["weights"].view(sample["c_nn_norm"].shape[0], len(phy.weight_names), 2)
    logits = torch.clamp(logits, min=-10.0, max=10.0)
    weights_on = F.softmax(logits, dim=-1)[..., 1]                      # (B, 4)
    mopex_params = phy._descale_mopex_params(params["params"])
    routing_params = phy._descale_routing_params(params["gamma_uh"])
    P, T, PET, doy, n_steps, n_grid = phy._prepare_forcings(sample)
    Q_mopex = phy._run_weighted_loop(P, T, PET, doy, mopex_params, weights_on, n_steps, n_grid)
    Qrouted = phy._apply_routing(Q_mopex.mean(-1), routing_params)
    out = {"streamflow": Qrouted}
    for i, name in enumerate(phy.weight_names):
        out[name] = weights_on[:, i].view(1, -1, 1).expand(Q_mopex.shape[0], -1, -1)
    coords = {
        # full tensors only: autograd.grad cannot target views/slices
        "weights_on": weights_on,       # (B, 4) real graph node
        "alpha": mopex_params["alpha"],     # (B, nmul) real graph node
        "is_time": mopex_params["is_time"],  # (B, nmul) real graph node
    }
    return out, coords


def gradient_diagnostics(handler, loss_func, sample) -> dict:
    out, coords = diagnostic_forward(handler, sample)
    q = out["streamflow"]
    target = sample["target"]
    n = min(q.shape[0], target.shape[0])
    weights = {g: out[g] for g in GATE_NAMES}
    loss = loss_func(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=weights)
    w_int = coords["weights_on"]       # (B, 4); per-basin coordinate = column 1
    alpha = coords["alpha"]            # (B, nmul)
    is_time = coords["is_time"]        # (B, nmul)

    # All targets below are real graph nodes -> autograd.grad is exact.
    # alpha/is_time: per-basin coordinate is the sum over the nmul replicates.
    def grad_of(y, x, retain=True):
        g = torch.autograd.grad(y, x, retain_graph=retain, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(x)

    gL_w = grad_of(loss, w_int)[:, 1]           # (B,)
    gL_a = grad_of(loss, alpha).sum(-1)         # (B,)
    gL_t = grad_of(loss, is_time).sum(-1)       # (B,)
    gQ_w = grad_of(q[:n].sum(), w_int)[:, 1]    # (B,)
    gQ_a = grad_of(q[:n].sum(), alpha).sum(-1)  # (B,)
    del q, loss, out
    torch.cuda.empty_cache()

    def frac_zero(g, eps=1e-12):
        return float((g.abs() < eps).float().mean())

    cos_wa = float(torch.nn.functional.cosine_similarity(gQ_w, gQ_a, dim=0))
    return {
        "median_abs_dL_dw_int": float(gL_w.abs().median()),
        "median_abs_dL_dalpha": float(gL_a.abs().median()),
        "median_abs_dL_dis_time": float(gL_t.abs().median()),
        "zero_frac_dL_dw_int": frac_zero(gL_w),
        "zero_frac_dL_dalpha": frac_zero(gL_a),
        "zero_frac_dL_dis_time": frac_zero(gL_t),
        "cos_dQ_dw_int_dQ_dalpha": cos_wa,
        "median_abs_dQ_dw_int": float(gQ_w.abs().median()),
        "median_abs_dQ_dalpha": float(gQ_a.abs().median()),
    }


# ---------------------------------------------------------------------------
# loss decomposition on the diagnostic batch
# ---------------------------------------------------------------------------
def loss_decomposition(handler, loss_func, sample) -> dict:
    out, _ = diagnostic_forward(handler, sample)
    q = out["streamflow"]
    target = sample["target"]
    n = min(q.shape[0], target.shape[0])
    weights = {g: out[g] for g in GATE_NAMES}
    total = float(loss_func(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=weights))
    # predictive-only: zero the AIC strength
    loss_func.aic_alpha = 0.0
    predictive = float(loss_func(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=weights))
    loss_func.aic_alpha = 0.01
    aic_term = total - predictive
    del q, out
    return {"total": total, "predictive": predictive, "aic_term": aic_term}


# ---------------------------------------------------------------------------
# canonical test metrics (sanity)
# ---------------------------------------------------------------------------
def test_metrics(arm_dir: Path) -> dict:
    candidates = [
        arm_dir / "test1995-2010_Ep10" / "metrics_agg.json",
        arm_dir / "test1995-2010_Ep50" / "metrics_agg.json",
        arm_dir / "sim" / "metrics_agg.json",
    ]
    m = next((p for p in candidates if p.exists()), None)
    if m is None:
        return {"nse_median": None, "kge_median": None}
    d = json.loads(m.read_text())
    med = d.get("median", d)
    return {
        "nse_median": med.get("NSE", med.get("nse")),
        "kge_median": med.get("KGE", med.get("kge")),
    }


# ---------------------------------------------------------------------------
# M4 ON/OFF external reference search
# ---------------------------------------------------------------------------
def find_m4_onoff_reference(repo_root: Path):
    import subprocess
    candidates = []
    for root in (repo_root / "project" / "benchmark" / "results",
                 repo_root / "results", PROJECT_DIR / "results"):
        if root.exists():
            candidates += [p for p in root.rglob("*.csv") if p.exists()]
            candidates += [p for p in root.rglob("*.json") if p.exists()]
    for p in candidates:
        try:
            head = p.read_text(errors="ignore")[:2000]
        except OSError:
            continue
        if "NSE_on" in head or "delta_NSE" in head or "NSE_off" in head:
            return p
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="results/intercept_2x2")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--config-dir", default="conf")
    args = ap.parse_args()

    root = Path(args.output_root)
    config_path = PROJECT_DIR / args.config_dir / "config_dmopex_intercept2x2_A.yaml"
    cli = parse_args(["--config", str(config_path), "--gpu-id", str(args.gpu_id),
                      "--output-root", str(args.output_root), "--run-name", "A"])
    config = load_config(str(config_path))
    apply_runtime_overrides(config, cli, config_path=str(config_path))
    config["mode"] = "train"
    if str(config["device"]).startswith("cuda"):
        torch.cuda.set_device(config["device"])

    data_loader = _build_data_loader(config)
    train_dataset = data_loader.train_dataset
    loss_func = _build_loss(config, train_dataset)
    sample = diagnostic_sample(train_dataset, config["device"])
    print(f"[diag] deterministic diagnostic batch: basins={DIAG_BASINS[:3]}... "
          f"days={DIAG_T0}-{DIAG_T1}, scored={DIAG_T1 - DIAG_T0 - 365}")

    summary = {}
    for arm in ARMS:
        arm_cfg_path = PROJECT_DIR / args.config_dir / f"config_dmopex_intercept2x2_{arm}.yaml"
        arm_cli = parse_args(["--config", str(arm_cfg_path), "--gpu-id", str(args.gpu_id),
                              "--output-root", str(args.output_root), "--run-name", arm])
        arm_cfg = load_config(str(arm_cfg_path))
        apply_runtime_overrides(arm_cfg, arm_cli, config_path=str(arm_cfg_path))
        arm_cfg["mode"] = "train"
        handler = build_handler(arm_cfg)
        arm_dir = root / arm
        entry = {"arm": arm, "desc": ARMS_DESC[arm]}
        entry["test"] = test_metrics(arm_dir)
        for epoch in EPOCHS:
            try:
                load_epoch_state(handler, epoch)
            except FileNotFoundError as e:
                print(f"  [warn] {arm} epoch {epoch} checkpoint missing: {e}")
                continue
            ep = epoch_gate_params(handler, arm_cfg, train_dataset)
            entry[f"ep{epoch}_gates"] = {g: ep["gates"][g] for g in GATE_NAMES}
            entry[f"ep{epoch}_alpha"] = ep["alpha"]
            entry[f"ep{epoch}_is_time"] = ep["is_time"]
            if "decoupled_annual_mean" in ep:
                entry[f"ep{epoch}_decoupled_annual_mean"] = ep["decoupled_annual_mean"]
            # save per-basin w_int (Phase 6/9)
            np.save(arm_dir / f"w_int_ep{epoch}.npy", ep["w_int_per_basin"])
            if epoch in (0, 5, 10):
                entry[f"ep{epoch}_grad"] = gradient_diagnostics(handler, loss_func, sample)
                entry[f"ep{epoch}_loss"] = loss_decomposition(handler, loss_func, sample)
        summary[arm] = entry
        print(f"[diag] arm {arm} done")

    out_json = root / "analysis_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # epoch-10 table + factorial contrasts
    rows = []
    for arm in ARMS:
        e = summary[arm]
        g10 = e.get("ep10_gates", {})
        gr0 = e.get("ep0_grad", {})
        rows.append({
            "arm": arm, "desc": ARMS_DESC[arm],
            "nse_median": e["test"]["nse_median"], "kge_median": e["test"]["kge_median"],
            "loss_total": e.get("ep10_loss", {}).get("total"),
            "loss_predictive": e.get("ep10_loss", {}).get("predictive"),
            "loss_aic": e.get("ep10_loss", {}).get("aic_term"),
            "w_int_median": g10.get("w_int", {}).get("median"),
            "w_int_iqr": g10.get("w_int", {}).get("iqr"),
            "w_int_frac>0.01": g10.get("w_int", {}).get("frac>0.01"),
            "w_int_frac>0.1": g10.get("w_int", {}).get("frac>0.1"),
            "w_snow_median": g10.get("w_snow", {}).get("median"),
            "w_sub_median": g10.get("w_sub", {}).get("median"),
            "w_phen_median": g10.get("w_phen", {}).get("median"),
            "alpha_median": e.get("ep10_alpha", {}).get("median"),
            "is_time_median": e.get("ep10_is_time", {}).get("median"),
            "|dL/dw_int|": gr0.get("median_abs_dL_dw_int"),
            "|dL/dalpha|": gr0.get("median_abs_dL_dalpha"),
            "|dL/dis_time|": gr0.get("median_abs_dL_dis_time"),
            "cos(dQ/dw_int,dQ/dalpha)": gr0.get("cos_dQ_dw_int_dQ_dalpha"),
        })
    with (root / "epoch10_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # external M4 ON/OFF alignment (Phase 8/9)
    m4_ref = find_m4_onoff_reference(PROJECT_DIR.parent.parent)
    align = {"reference_found": m4_ref is not None, "path": str(m4_ref) if m4_ref else None}
    if m4_ref is not None:
        import pandas as pd
        from scipy.stats import spearmanr
        df = pd.read_csv(m4_ref)
        for arm in ARMS:
            w10 = np.load(root / arm / f"w_int_ep10.npy")
            # try to align on basin id if present, else row order
            if "basin" in df.columns or "gage" in df.columns or "id" in df.columns:
                id_col = next(c for c in ("basin", "basin_id", "gage", "gage_id", "id") if c in df.columns)
                ids = df[id_col].astype(str).tolist()
                gage = np.load(PROJECT_DIR.parent.parent / "data" / "gage_id.npy")
                # 671 gage order matches xc_nn_norm basin order in the loader
                order = [str(int(x)) for x in gage]
                idx = [order.index(i) for i in ids if i in order]
                w = w10[idx] if len(idx) == len(ids) else w10[: len(ids)]
            else:
                w = w10[: len(df)]
            dn = df["delta_NSE_int"].to_numpy() if "delta_NSE_int" in df.columns else None
            dk = df["delta_KGE_int"].to_numpy() if "delta_KGE_int" in df.columns else None
            align[arm] = {}
            if dn is not None and len(dn) == len(w):
                rho, p = spearmanr(w, dn)
                align[arm]["spearman_w_int_delta_NSE"] = float(rho)
                align[arm]["spearman_p"] = float(p)
            if dk is not None and len(dk) == len(w):
                rho, p = spearmanr(w, dk)
                align[arm]["spearman_w_int_delta_KGE"] = float(rho)
    summary["_m4_onoff"] = align
    out_json.write_text(json.dumps(summary, indent=2))

    # markdown report
    md = ["# Interception 2x2 — epoch-10 table", ""]
    md.append("| arm | desc | NSE med | KGE med | loss tot | loss fit | loss AIC | "
              "w_int med | w_int IQR | w_int>0.01 | w_int>0.1 | w_snow med | w_sub med | "
              "alpha med | is_time med | |dL/dw_int| | |dL/dalpha| | |dL/dis_time| | cos(dQ/dw,dQ/da) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append("| " + " | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
                                    for v in r.values()) + " |")
    md.append("")
    md.append("## Trajectories (median, epoch 0 -> 5 -> 10)")
    for arm in ARMS:
        e = summary[arm]
        line = f"**{arm} ({ARMS_DESC[arm]})**: "
        for key, label in (("w_int", "w_int"), ("w_snow", "w_snow"), ("w_sub", "w_sub"), ("w_phen", "w_phen")):
            vals = [e.get(f"ep{ep}_gates", {}).get(key, {}).get("median") for ep in EPOCHS]
            line += f"{label}=[{', '.join('' if v is None else f'{v:.3f}' for v in vals)}] "
        vals = [e.get(f"ep{ep}_alpha", {}).get("median") for ep in EPOCHS]
        line += f"alpha=[{', '.join('' if v is None else f'{v:.3f}' for v in vals)}] "
        vals = [e.get(f"ep{ep}_is_time", {}).get("median") for ep in EPOCHS]
        line += f"is_time=[{', '.join('' if v is None else f'{v:.1f}' for v in vals)}] "
        vals = [e.get(f"ep{ep}_grad", {}).get("median_abs_dL_dw_int") for ep in EPOCHS]
        line += f"|dL/dw_int|=[{', '.join('' if v is None else f'{v:.2e}' for v in vals)}]"
        md.append(line)
    md.append("")
    md.append(f"## M4 ON/OFF alignment\n\nreference: {align}")
    (root / "analysis_report.md").write_text("\n".join(md))
    print(f"[diag] summary -> {out_json}")
    print(f"[diag] table    -> {root / 'epoch10_table.csv'}")
    print(f"[diag] report   -> {root / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
