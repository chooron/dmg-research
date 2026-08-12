"""CUDA edge-saturation probe for the 19 K1 boundary models."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
spec = importlib.util.spec_from_file_location("k_full_retrain", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(k)

OUT = ROOT / "results/dpl_round12_20260805"
MODELS = [
    "australia", "collie2", "collie3", "flexb", "flexi", "flexis", "hbv96",
    "modhydrolog", "mopex1", "mopex4", "mopex5", "newzealand1", "newzealand2",
    "penman", "plateau", "tcm", "topmodel", "vic", "wetland",
]
DEVICE = torch.device("cuda")


def cma_boundary_fraction(model: str) -> float | None:
    path = ROOT.parent.parent / "dmotpy/experiments/cmaes_36models/downloads/full300_20260729_160112_partial_20260730/checkpoints_latest" / model / "chunk_0_gen_300.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    solver = payload.get("solver", {}).get("state", payload)
    latent = solver.get("best_latent")
    if latent is None:
        return None
    theta = torch.sigmoid(latent)
    if theta.ndim == 3:
        fitness = solver.get("best_fitness")
        theta = theta[torch.arange(theta.shape[0]), fitness.argmax(1)] if fitness is not None else theta[:, 0]
    return float(((theta < .02) | (theta > .98)).float().mean())


def build_inputs(model: str):
    ids = [int(value) for value in k.load_ids("data/531sub_id.txt")]
    attrs = k.CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    tx, ty, vx, vy = k.NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(tx, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    val_x = torch.as_tensor(vx, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(vy, dtype=torch.float32, device=DEVICE)
    if model in k.CALENDAR_MODELS:
        train_x, _ = k.add_calendar_forcing(train_x, pd.date_range("1980-10-01", "1995-09-30", freq="D"), model_name=model)
        val_x, _ = k.add_calendar_forcing(val_x, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name=model)
    return ids, attrs, train_x, train_y, val_x, val_y


def latest_checkpoint(model: str) -> Path:
    files = sorted((ROOT / "results/dpl_full_retrain_20260804/auto100/checkpoints" / model).glob("epoch_*.pt"))
    if not files:
        raise FileNotFoundError(model)
    return files[-1]


def run_model(model: str) -> tuple[list[dict], dict]:
    started = time.time()
    ids, attrs, train_x, train_y, val_x, val_y = build_inputs(model)
    warm_mode = "truncate:90" if model == "penman" else "detach"
    hydro = k.build_model(model, DEVICE, warm_up=365, backend="compile", parameter_mapping="auto", warmup_grad_mode=warm_mode)
    net = k.CatchmentParameterizer(attrs.shape[1], k.NPARAM_INFO_36[model], hidden_dims=[256, 256], dropout=.05).to(DEVICE)
    payload = torch.load(latest_checkpoint(model), map_location="cpu", weights_only=False)
    net.load_state_dict(payload["network"])
    net.eval()
    with torch.no_grad():
        theta = net(attrs)
        q = hydro({"x_phy": val_x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        _, kge = k.NATIVE.compute_differentiable_kge(q, val_y, warmup_days=365)
        baseline = float(kge.median())

    names = list(k.PARAM_INFO[model])
    boundary = (theta < .02) | (theta > .98)
    boundary_params = [j for j in range(theta.shape[1]) if bool(boundary[:, j].any())]
    catalog, lengths = k.H1.make_catalog(train_y[365:])
    grad_samples = {j: [] for j in boundary_params}
    for _ in range(8):
        basins = torch.randperm(len(ids), device=DEVICE)[:k.BATCH]
        choices = (torch.rand(k.BATCH, device=DEVICE) * lengths[basins]).long()
        starts = catalog[basins, choices]
        x = k.H1.gather_window(train_x, starts, basins)
        y = k.H1.gather_window(train_y, starts, basins)
        net.zero_grad(set_to_none=True)
        sampled_theta = net(attrs[basins])
        sampled_theta.retain_grad()
        q_train = hydro({"x_phy": x}, (None, sampled_theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        loss, _ = k.NATIVE.compute_differentiable_kge(q_train, y[365:], warmup_days=0)
        loss.backward()
        for j in boundary_params:
            local = (sampled_theta[:, j] < .02) | (sampled_theta[:, j] > .98)
            if bool(local.any()):
                grad_samples[j].extend(sampled_theta.grad[:, j].detach().abs()[local].float().cpu().tolist())
        del q_train, loss, sampled_theta

    detail = []
    drops = []
    for j in boundary_params:
        forced = theta.detach().clone()
        mask = boundary[:, j]
        forced[mask, j] = torch.where(theta[mask, j] > .5, torch.ones_like(theta[mask, j]), torch.zeros_like(theta[mask, j]))
        with torch.no_grad():
            q_forced = hydro({"x_phy": val_x}, (None, forced.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            _, forced_kge = k.NATIVE.compute_differentiable_kge(q_forced, val_y, warmup_days=365)
            delta = float(forced_kge.median()) - baseline
        drops.append(delta)
        samples = np.asarray(grad_samples[j], dtype=float)
        detail.append({"model": model, "parameter": names[j], "boundary_basin_count": int(boundary[:, j].sum()),
                       "boundary_fraction": float(boundary[:, j].float().mean()),
                       "boundary_grad_median_abs_dloss_dtheta": float(np.median(samples)) if samples.size else np.nan,
                       "boundary_grad_sample_count": int(samples.size), "boundary_pin_kge_delta": delta})
    summary = {"model": model, "boundary_parameter_count": len(boundary_params),
               "boundary_entry_fraction": float(boundary.float().mean()),
               "boundary_grad_median_abs_dloss_dtheta": float(np.nanmedian([r["boundary_grad_median_abs_dloss_dtheta"] for r in detail])) if detail else np.nan,
               "boundary_pin_kge_delta_median": float(np.median(drops)) if drops else np.nan,
               "boundary_pin_kge_delta_min": float(np.min(drops)) if drops else np.nan,
               "baseline_validation_median_kge": baseline,
               "cma_boundary_entry_fraction": cma_boundary_fraction(model),
               "runtime_seconds": time.time() - started}
    del hydro, net, train_x, train_y, val_x, val_y, attrs
    torch.cuda.empty_cache()
    return detail, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    details, summaries = [], []
    for model in MODELS:
        print(f"[L2] {model}", flush=True)
        d, s = run_model(model)
        details.extend(d)
        summaries.append(s)
        pd.DataFrame(details).to_csv(OUT / "l2_edge_parameter_details.csv", index=False)
        pd.DataFrame(summaries).to_csv(OUT / "l2_edge_summary.csv", index=False)
        print(s, flush=True)


if __name__ == "__main__":
    main()
