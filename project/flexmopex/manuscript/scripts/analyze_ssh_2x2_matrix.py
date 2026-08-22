#!/usr/bin/env python3
"""Epoch-1..10 diagnostics and formula-specific grid oracle for SSH 2x2 runs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
for p in (REPO_ROOT, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import _build_data_loader, _attach_doy  # noqa: E402
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402

CONDITIONS = ("E1", "E2", "E3", "E4")
SEEDS = (42, 43, 44)
W_GRID = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
COSTS = np.array([2.0, 2.0, 2.0, 1.0], dtype=np.float64)
GATES = ("w_phen", "w_int", "w_snow", "w_sub")


def rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        idx = np.flatnonzero(x == value)
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr, yr = rank_average(x[mask]), rank_average(y[mask])
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "iqr": float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "frac_gt_001": float(np.mean(x > 0.01)),
        "frac_gt_01": float(np.mean(x > 0.1)),
        "frac_gt_05": float(np.mean(x > 0.5)),
    }


def make_config(config_path: Path, out_dir: Path, gpu_id: int) -> dict:
    cfg = load_config(str(config_path))
    device = f"cuda:{gpu_id}"
    cfg["mode"] = "test"
    cfg["device"] = device
    cfg["gpu_id"] = gpu_id
    cfg["test"]["test_epoch"] = 10
    cfg["model_dir"] = str(out_dir / "model")
    cfg["model_path"] = str(out_dir / "model")
    cfg["output_dir"] = str(out_dir)
    cfg["out_path"] = str(out_dir / "sim")
    cfg["sim_dir"] = str(out_dir / "sim")
    cfg["save_path"] = str(out_dir)
    cfg["trained_model"] = str(out_dir / "model")
    cfg.setdefault("model", {}).setdefault("phy", {})["disable_compile"] = False
    cfg["model"]["phy"]["require_torch_compile"] = True
    return cfg


def load_run(config_path: Path, out_dir: Path, gpu_id: int):
    cfg = make_config(config_path, out_dir, gpu_id)
    loader = _build_data_loader(cfg)
    _attach_doy(loader.train_dataset, cfg["train"])
    _attach_doy(loader.eval_dataset, cfg["test"])
    handler = FlexMopexModelHandler(cfg, verbose=False)
    handler.load_model(10)
    handler.eval()
    return cfg, loader, handler


def weights_for(handler, attrs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model = next(iter(handler.model_dict.values()))
    with torch.no_grad():
        nn_out = model.nn_model({"c_nn_norm": attrs})
        logits = nn_out["weights"].view(attrs.shape[0], 4, 2).clamp(-10, 10)
        weights = F.softmax(logits, dim=-1)[..., 1]
    return nn_out, weights


def checkpoint_gate_and_loss(cfg, loader, handler, epoch: int, device: str, out_dir: Path) -> tuple[dict, np.ndarray]:
    model = next(iter(handler.model_dict.values()))
    train = loader.train_dataset
    attrs = train["xc_nn_norm"][0, :, -35:].to(device)
    nn_out, weights = weights_for(handler, attrs)
    w_np = weights.cpu().numpy()
    row = {"epoch": epoch, "gates": {name: stats(w_np[:, i]) for i, name in enumerate(GATES)}}

    # Full training-window L_Q/Omega evaluation in basin chunks; this does not
    # alter the training protocol and uses the exact NseDynAicBatchLoss scaling.
    std = np.nanstd(train["target"][:, :, 0].cpu().numpy(), axis=0)
    fit_sum = 0.0
    fit_count = 0
    omega_sum = 0.0
    bsz = 64
    phy = model.phy_model
    for start in range(0, train["x_phy"].shape[1], bsz):
        idx = slice(start, min(start + bsz, train["x_phy"].shape[1]))
        sample = {
            "x_phy": train["x_phy"][:, idx].to(device),
            "doy": train["doy"][:, idx].to(device),
            "c_nn_norm": train["xc_nn_norm"][0, idx, -35:].to(device),
        }
        with torch.no_grad():
            out = model(sample)
        pred = out["streamflow"][:, :, 0].detach().cpu().numpy()
        obs = train["target"][phy.warm_up:, idx, 0].cpu().numpy()[: pred.shape[0]]
        valid = np.isfinite(obs)
        residual = (pred - np.nan_to_num(obs, nan=0.0)) ** 2
        denom = std[start : start + pred.shape[1]][None, :] + 0.1
        fit_sum += float(np.sum(np.where(valid, residual / (denom**2), 0.0)))
        fit_count += int(valid.sum())
        for i, name in enumerate(GATES):
            w = out[name].detach().cpu().numpy()
            omega_sum += float(np.mean(w) * pred.shape[1] * COSTS[i])
    lq = fit_sum / max(fit_count, 1)
    omega = omega_sum / train["x_phy"].shape[1]
    row["loss"] = {"L_Q": lq, "Omega": omega, "lambda": float(cfg["loss_function"]["aic_alpha"]), "lambda_Omega": float(cfg["loss_function"]["aic_alpha"] * omega), "L_Q_plus_lambda_Omega": lq + cfg["loss_function"]["aic_alpha"] * omega}
    out_dir.joinpath("diagnostics").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "diagnostics" / f"weights_epoch_{epoch:02d}.npy", w_np)
    return row, w_np


def oracle_for_checkpoint(cfg, loader, handler, device: str, out_dir: Path) -> dict:
    model = next(iter(handler.model_dict.values()))
    phy = model.phy_model
    train = loader.train_dataset
    ev = loader.eval_dataset
    attrs_all = ev["xc_nn_norm"][0, :, -35:].to(device)
    with torch.no_grad():
        nn_out, learned = weights_for(handler, attrs_all)
        params = phy._descale_mopex_params(nn_out["params"])
        routing = phy._descale_routing_params(nn_out["gamma_uh"])
    B = attrs_all.shape[0]
    std = np.nanstd(train["target"][:, :, 0].cpu().numpy(), axis=0)
    best_w = np.full(B, np.nan, dtype=float)
    grid_fit = np.full((B, len(W_GRID)), np.nan, dtype=float)
    grid_nse = np.full((B, len(W_GRID)), np.nan, dtype=float)
    learned_nse = np.full(B, np.nan, dtype=float)
    learned_kge = np.full(B, np.nan, dtype=float)
    bsz = 48
    for start in range(0, B, bsz):
        end = min(start + bsz, B)
        b = end - start
        attrs = attrs_all[start:end]
        with torch.no_grad():
            nn_b = {k: v[start:end] for k, v in nn_out.items()}
            base_w = learned[start:end]
            p_b = {k: v[start:end] for k, v in params.items()}
            r_b = {k: v[start:end] for k, v in routing.items()}
        sample = {"x_phy": ev["x_phy"][:, start:end].to(device), "doy": ev["doy"][:, start:end].to(device), "c_nn_norm": attrs}
        S = len(W_GRID)
        sample_rep = {"x_phy": sample["x_phy"].repeat(1, S, 1), "doy": sample["doy"].repeat(1, S, 1), "c_nn_norm": attrs.repeat(S, 1)}
        w_rep = base_w.repeat(S, 1)
        for s, wv in enumerate(W_GRID):
            w_rep[s * b : (s + 1) * b, 1] = float(wv)
        p_rep = {k: v.repeat(S, 1) for k, v in p_b.items()}
        r_rep = {k: v.repeat(S) for k, v in r_b.items()}
        with torch.no_grad():
            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(sample_rep)
            q = phy._run_weighted_loop(P, T, PET, doy, p_rep, w_rep, n_steps, b * S)
            q = phy._apply_routing(q.mean(-1), r_rep).cpu().numpy()[:, :, 0]
        q = q.reshape(q.shape[0], S, b).transpose(2, 0, 1)
        obs = ev["target"][phy.warm_up:, start:end, 0].cpu().numpy()[: q.shape[1]]
        valid = np.isfinite(obs)
        for j in range(b):
            v = valid[:, j]
            if v.sum() < 30:
                continue
            o = obs[v, j]
            for s in range(S):
                sim = q[j, v, s]
                grid_fit[start + j, s] = np.mean((sim - o) ** 2 / (std[start + j] + 0.1) ** 2)
                total = np.sum((o - o.mean()) ** 2)
                grid_nse[start + j, s] = 1.0 - np.sum((sim - o) ** 2) / (total + 1e-12)
        # Independent prediction sanity metrics at the learned epoch-10 weights.
        with torch.no_grad():
            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(sample)
            q_learn = phy._run_weighted_loop(P, T, PET, doy, p_b, base_w, n_steps, b)
            q_learn = phy._apply_routing(q_learn.mean(-1), r_b).cpu().numpy()[:, :, 0]
        obs_learn = ev["target"][phy.warm_up:, start:end, 0].cpu().numpy()[: q_learn.shape[0]]
        valid_learn = np.isfinite(obs_learn)
        for j in range(b):
            v = valid_learn[:, j]
            if v.sum() < 30:
                continue
            o = obs_learn[v, j]
            sim = q_learn[v, j]
            total = np.sum((o - o.mean()) ** 2)
            learned_nse[start + j] = 1.0 - np.sum((sim - o) ** 2) / (total + 1e-12)
            r = np.corrcoef(sim, o)[0, 1] if np.std(sim) > 0 and np.std(o) > 0 else 0.0
            alpha = np.std(sim) / (np.std(o) + 1e-12)
            beta = np.mean(sim) / (np.mean(o) + 1e-12)
            learned_kge[start + j] = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    best_idx = np.nanargmin(grid_fit, axis=1)
    best_w[:] = W_GRID[best_idx]
    learned_np = learned[:, 1].cpu().numpy()
    positive = best_w > 0
    zero = best_w == 0
    result = {
        "grid": W_GRID.tolist(),
        "objective": "training-normalized basin fit loss; no DeltaJ/AIC",
        "rho_w_int_wstar": spearman(learned_np, best_w),
        "oracle_positive_fraction": float(np.mean(positive)),
        "learned_active_fraction_gt_001": float(np.mean(learned_np > 0.01)),
        "learned_active_fraction_gt_01": float(np.mean(learned_np > 0.1)),
        "false_negative_fraction_oracle_positive_learned_near_off": float(np.mean(learned_np[positive] <= 0.01)) if positive.any() else float("nan"),
        "oracle_separation_mean_w_positive_minus_zero": float(np.mean(learned_np[positive]) - np.mean(learned_np[zero])) if positive.any() and zero.any() else float("nan"),
        "learned_w_int": stats(learned_np),
        "wstar": stats(best_w),
        "prediction": {
            "median_nse": float(np.nanmedian(learned_nse)),
            "mean_nse": float(np.nanmean(learned_nse)),
            "fraction_nse_gt_0": float(np.nanmean(learned_nse > 0)),
            "fraction_nse_gt_05": float(np.nanmean(learned_nse > 0.5)),
            "median_kge": float(np.nanmedian(learned_kge)),
            "mean_kge": float(np.nanmean(learned_kge)),
        }
    }
    np.save(out_dir / "diagnostics" / "oracle_wstar_epoch10.npy", best_w)
    np.save(out_dir / "diagnostics" / "oracle_fit_grid_epoch10.npy", grid_fit)
    np.save(out_dir / "diagnostics" / "oracle_nse_grid_epoch10.npy", grid_nse)
    (out_dir / "diagnostics" / "oracle_epoch10.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(PROJECT_DIR / "results/ssh_2x2"))
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for analysis; refusing CPU fallback")
    device = f"cuda:{args.gpu_id}"
    root = Path(args.root)
    summary = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            config_path = PROJECT_DIR / f"conf/ssh_2x2/config_{condition}_pure_x35_531_lambda0007.yaml"
            out_dir = root / condition / f"seed_{seed}"
            cfg, loader, handler = load_run(config_path, out_dir, args.gpu_id)
            rows = []
            for epoch in range(1, 11):
                handler.load_model(epoch)
                handler.eval()
                row, _ = checkpoint_gate_and_loss(cfg, loader, handler, epoch, device, out_dir)
                rows.append(row)
            cf_path = out_dir / "structural_diagnostics.jsonl"
            if cf_path.exists():
                cf_rows = {int(item["epoch"]): item for item in (json.loads(line) for line in cf_path.read_text().splitlines()) if line.strip()}
                for row in rows:
                    item = cf_rows.get(row["epoch"])
                    if item:
                        row["loss"]["CF_loss"] = float(item["bce_loss"])
                        row["loss"]["training_total_loss"] = float(item["total_loss"])
            handler.load_model(10)
            handler.eval()
            oracle = oracle_for_checkpoint(cfg, loader, handler, device, out_dir)
            payload = {"condition": condition, "seed": seed, "epochs": rows, "oracle_epoch10": oracle}
            (out_dir / "diagnostics" / "epoch_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")
            summary[f"{condition}/seed_{seed}"] = {"oracle": oracle, "w_int_epoch10": rows[-1]["gates"]["w_int"], "loss_epoch10": rows[-1]["loss"]}
            print(f"DONE {condition} seed={seed} rho={oracle['rho_w_int_wstar']:.4f} w_int_mean={rows[-1]['gates']['w_int']['mean']:.4f}", flush=True)
    (root / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
