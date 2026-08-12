#!/usr/bin/env python
"""HBV dPL KGE(Q) window-length ablation.

The parameter network maps static CAMELS attributes to normalized HBV
parameters. Training uses repeated windows with a fixed warmup prefix and a
KGE(Q) loss on the prediction suffix only. The default protocol is the first
ablation point: 365-day warmup + 365-day prediction, 100 epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from models import HBV
from models.parameter_specs import HBV_PARAM_SPECS
from optimization.pycma_calibrator_v3 import compute_kge_fp64
from training.data_contract import FORCING_NAMES, load_dates, load_gage_ids


class StaticParameterNet(nn.Module):
    """Compact static-attribute parameterizer for 559-basin DPL."""

    def __init__(self, n_attributes: int, parameter_specs: dict, hidden_size: int,
                 dropout: float, output_epsilon: float,
                 hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        self.parameter_names = list(parameter_specs)
        self.output_epsilon = float(output_epsilon)
        if hidden_sizes is None:
            hidden_sizes = [hidden_size, hidden_size]
        if not hidden_sizes or any(int(width) <= 0 for width in hidden_sizes):
            raise ValueError("network.hidden_sizes must contain positive widths")
        self.hidden_sizes = [int(width) for width in hidden_sizes]

        layers: list[nn.Module] = []
        input_size = n_attributes
        for layer_index, layer_size in enumerate(self.hidden_sizes):
            layers.extend([
                nn.Linear(input_size, layer_size),
                nn.LayerNorm(layer_size),
                nn.SiLU(),
            ])
            # Keep the original 2-layer baseline exactly: dropout follows the
            # first hidden layer, while the final hidden representation is not
            # dropped before the parameter head.
            if layer_index < len(self.hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            input_size = layer_size
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(self.hidden_sizes[-1], len(self.parameter_names))
        self._initialize(parameter_specs)

    def _initialize(self, specs: dict) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Start near physically meaningful defaults rather than at 0.5 in every
        # normalized dimension. A small head weight lets attributes perturb that
        # starting point immediately without destabilizing the first epochs.
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        normalized_defaults = []
        for name in self.parameter_names:
            spec = specs[name]
            value = (spec["default"] - spec["lower"]) / (spec["upper"] - spec["lower"])
            normalized_defaults.append(np.clip(value, self.output_epsilon, 1.0 - self.output_epsilon))
        defaults = torch.tensor(normalized_defaults, dtype=torch.float32)
        self.head.bias.data.copy_(torch.logit(defaults))

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.trunk(attributes))
        return torch.sigmoid(logits).clamp(self.output_epsilon, 1.0 - self.output_epsilon)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gate_time_index(config: dict) -> dict[str, tuple[int, int]]:
    dates = pd.to_datetime(load_dates(config["dates_path"]))
    periods = config["time_periods"]

    def bounds(name: str) -> tuple[int, int]:
        start = pd.Timestamp(periods[name]["start"])
        end = pd.Timestamp(periods[name]["end"])
        si = int((dates >= start).argmax())
        ei = len(dates) - 1 - int((dates <= end)[::-1].argmax())
        assert dates[si].date() == start.date()
        assert dates[ei].date() == end.date()
        return si, ei

    warmup = bounds("warmup")
    calibration = bounds("calibration")
    evaluation = bounds("evaluation")
    assert calibration[1] - calibration[0] + 1 == periods["calibration"]["days"]
    assert evaluation[1] - evaluation[0] + 1 == periods["evaluation"]["days"]
    assert warmup[1] == calibration[0] - 1
    assert evaluation[0] == calibration[1] + 1
    return {"warmup": warmup, "calibration": calibration, "evaluation": evaluation}


def load_data(config: dict, indices: dict[str, tuple[int, int]], max_basins: int | None):
    raw = np.load(config["data_npz"], allow_pickle=True)
    forcing = np.asarray(raw["forcing"], dtype=np.float32)
    target = np.asarray(raw["target"], dtype=np.float32)
    with open(config["data_basin_ids"]) as handle:
        basin_ids = [str(value).zfill(8) for value in json.load(handle)]
    if max_basins is not None:
        basin_ids = basin_ids[:max_basins]

    full_ids = load_gage_ids(config["gage_ids_path"])
    id_to_index = {basin_id: i for i, basin_id in enumerate(full_ids)}
    selected = np.array([id_to_index[basin_id] for basin_id in basin_ids], dtype=np.int64)

    with open(config["data_pkl_dataset"], "rb") as handle:
        _, _, all_attributes = pickle.load(handle)
    attributes = np.asarray(all_attributes, dtype=np.float32)[selected]

    axis = {"precip": FORCING_NAMES.index("P"),
            "temp": FORCING_NAMES.index("T"),
            "pet": FORCING_NAMES.index("PET")}
    wi_s, _ = indices["warmup"]
    ci_s, ci_e = indices["calibration"]
    ei_s, ei_e = indices["evaluation"]
    assert ci_s - wi_s == config["window"]["warmup_days"]

    # The PET-v2 NPZ is already ordered and restricted to the 559 selected
    # basins.  ``selected`` is only for aligning the 671-basin attribute array
    # above; indexing the NPZ a second time would mix the two basin spaces.
    assert forcing.shape[1] >= len(basin_ids)
    train_forcing = {
        key: forcing[wi_s:ci_e + 1, :len(basin_ids), axis[key]].transpose().copy()
        for key in ("precip", "pet", "temp")
    }
    calibration_obs = target[ci_s:ci_e + 1, :len(basin_ids), 0].transpose().copy()

    eval_warmup_start = ei_s - config["window"]["warmup_days"]
    evaluation_forcing = {
        key: forcing[eval_warmup_start:ei_e + 1, :len(basin_ids), axis[key]].transpose().copy()
        for key in ("precip", "pet", "temp")
    }
    evaluation_obs = target[ei_s:ei_e + 1, :len(basin_ids), 0].transpose().copy()
    return basin_ids, attributes, train_forcing, calibration_obs, evaluation_forcing, evaluation_obs


def robust_normalize(attributes: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(attributes, dtype=np.float32).copy()
    median = np.nanmedian(values, axis=0)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(median, np.where(missing)[1])
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    scale = q75 - q25
    fallback = values.std(axis=0)
    scale[scale < 1e-6] = fallback[scale < 1e-6]
    scale[scale < 1e-6] = 1.0
    normalized = np.clip((values - median) / scale, -5.0, 5.0)
    return normalized.astype(np.float32), {"median": median, "scale": scale}


def build_windows(calibration_days: int, warmup_days: int, prediction_days: int,
                  stride_days: int) -> list[tuple[int, int, int]]:
    windows = []
    forecast_start = 0
    while forecast_start + prediction_days <= calibration_days:
        windows.append((forecast_start, forecast_start + prediction_days, forecast_start + warmup_days))
        forecast_start += stride_days
    if not windows:
        raise ValueError("No complete prediction windows fit in the calibration period.")
    return windows


def kge_per_basin(qsim: torch.Tensor, qobs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable column-wise KGE, matching IC KGE(Q) aggregation."""
    mask = torch.isfinite(qsim) & torch.isfinite(qobs) & (qobs >= 0.0) & (qsim >= 0.0)
    mask_f = mask.to(qsim.dtype)
    count = mask_f.sum(dim=1).clamp_min(1.0)
    p = torch.where(mask, qsim, torch.zeros_like(qsim))
    o = torch.where(mask, qobs, torch.zeros_like(qobs))
    mean_p = p.sum(dim=1) / count
    mean_o = o.sum(dim=1) / count
    dp = (p - mean_p[:, None]) * mask_f
    do = (o - mean_o[:, None]) * mask_f
    eps_sq = eps * eps
    sim_ss = dp.square().sum(dim=1)
    obs_ss = do.square().sum(dim=1)
    std_p = torch.sqrt(sim_ss / count + eps_sq)
    std_o = torch.sqrt(obs_ss / count + eps_sq)
    covariance = (dp * do).sum(dim=1) / count
    r = covariance / (std_p * std_o)
    alpha = std_p / std_o
    beta = mean_p / (mean_o + eps)
    return 1.0 - torch.sqrt(
        (r - 1.0).square()
        + (alpha - 1.0).square()
        + (beta - 1.0).square()
        + eps_sq
    )


def physical_parameters(theta: torch.Tensor, names: list[str], lower: torch.Tensor,
                        parameter_range: torch.Tensor) -> dict[str, torch.Tensor]:
    physical = lower + theta * parameter_range
    return {name: physical[:, index] for index, name in enumerate(names)}


def evaluate_validation(net: nn.Module, attributes: torch.Tensor, forcing: dict[str, np.ndarray],
                        observations: np.ndarray, batch_size: int, device: torch.device,
                        names: list[str], lower: torch.Tensor, parameter_range: torch.Tensor,
                        warmup_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FP64 model rerun and FP64 KGE evaluation on the 1999–2009 period."""
    net.eval()
    model = HBV().to(device)
    n_basins = attributes.shape[0]
    parameters = np.full((n_basins, len(names)), np.nan, dtype=np.float64)
    kges = np.full(n_basins, np.nan, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_basins, batch_size):
            stop = min(start + batch_size, n_basins)
            index = slice(start, stop)
            theta = net(attributes[index].to(device)).to(torch.float64)
            params = physical_parameters(theta, names, lower, parameter_range)
            fc = {key: torch.from_numpy(value[index].copy()).to(device=device, dtype=torch.float64)
                  for key, value in forcing.items()}
            qsim, _ = model(forcings=fc, params=params)
            q_np = qsim[:, warmup_days:].cpu().numpy()
            theta_np = theta.cpu().numpy()
            parameters[index] = theta_np
            for local_index, basin_index in enumerate(range(start, stop)):
                kge, _ = compute_kge_fp64(q_np[local_index], observations[basin_index])
                kges[basin_index] = kge
    del model
    torch.cuda.empty_cache()
    return kges, parameters, lower.cpu().numpy() + parameters * parameter_range.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_DIR / "configs/dpl_hbv_kgeq_365d_v1.json")
    parser.add_argument("--epochs", type=int, help="Override config training.epochs.")
    parser.add_argument("--max-basins", type=int, help="Use only the first N basins for a smoke test.")
    parser.add_argument("--output-dir", help="Override config output_dir (useful for smoke tests).")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    output_dir = PROJECT_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    set_seed(config["training"]["seed"])
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    indices = gate_time_index(config)
    basin_ids, raw_attributes, train_forcing, calibration_obs, eval_forcing, eval_obs = load_data(
        config, indices, args.max_basins)
    attributes_np, attribute_stats = robust_normalize(raw_attributes)
    np.savez_compressed(output_dir / "attribute_normalization.npz", **attribute_stats)
    attributes = torch.from_numpy(attributes_np)
    n_basins = len(basin_ids)

    win_cfg = config["window"]
    calibration_days = calibration_obs.shape[1]
    windows = build_windows(calibration_days, win_cfg["warmup_days"],
                            win_cfg["prediction_days"], win_cfg["stride_days"])
    used_prediction_days = len(windows) * win_cfg["prediction_days"]
    tail_days = calibration_days - (windows[-1][1])
    assert train_forcing["precip"].shape[1] == calibration_days + win_cfg["warmup_days"]

    names = list(HBV_PARAM_SPECS)
    lower = torch.tensor([HBV_PARAM_SPECS[name]["lower"] for name in names], device=device, dtype=torch.float32)
    upper = torch.tensor([HBV_PARAM_SPECS[name]["upper"] for name in names], device=device, dtype=torch.float32)
    parameter_range = upper - lower
    net_cfg = config["network"]
    hidden_sizes = net_cfg.get("hidden_sizes")
    if hidden_sizes is None:
        hidden_sizes = [net_cfg["hidden_size"]] * int(net_cfg.get("depth", 2))
    hidden_sizes = [int(width) for width in hidden_sizes]
    net = StaticParameterNet(attributes.shape[1], HBV_PARAM_SPECS, hidden_sizes[0],
                             net_cfg["dropout"], net_cfg["output_epsilon"],
                             hidden_sizes=hidden_sizes).to(device)
    model = HBV().to(device)
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(net.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=train_cfg["min_lr"])

    print(f"DPL HBV/KGE(Q): basins={n_basins}, attrs={attributes.shape[1]}, params={len(names)}", flush=True)
    print(f"Windows={len(windows)} × warmup={win_cfg['warmup_days']} + prediction={win_cfg['prediction_days']}; "
          f"calibration tail excluded={tail_days} days", flush=True)
    network_shape = "→".join(["35", *map(str, hidden_sizes), str(len(names))])
    print(f"Network={network_shape}, "
          f"AdamW lr={train_cfg['lr']}, epochs={train_cfg['epochs']}", flush=True)

    history = []
    best_state = None
    best_validation = -np.inf
    basin_order = np.arange(n_basins)
    t0 = time.time()
    for epoch in range(1, train_cfg["epochs"] + 1):
        net.train()
        np.random.shuffle(basin_order)
        shuffled_windows = list(windows)
        random.shuffle(shuffled_windows)
        losses = []
        finite_batches = 0
        for forecast_start, forecast_stop, _ in shuffled_windows:
            forcing_start = forecast_start
            forcing_stop = forecast_stop + win_cfg["warmup_days"]
            obs_window = calibration_obs[:, forecast_start:forecast_stop]
            for batch_start in range(0, n_basins, train_cfg["batch_size"]):
                batch_index = basin_order[batch_start:batch_start + train_cfg["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                x = attributes[batch_index].to(device)
                theta = net(x)
                params = physical_parameters(theta, names, lower, parameter_range)
                fc = {
                    key: torch.from_numpy(values[batch_index, forcing_start:forcing_stop].copy()).to(device)
                    for key, values in train_forcing.items()
                }
                obs = torch.from_numpy(obs_window[batch_index].copy()).to(device)
                qsim, _ = model(forcings=fc, params=params)
                prediction = qsim[:, win_cfg["warmup_days"]:]
                kge = kge_per_basin(prediction, obs)
                valid = torch.isfinite(kge)
                if not valid.any():
                    continue
                loss = (1.0 - kge[valid]).mean()
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg["grad_clip_norm"])
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                finite_batches += 1
        scheduler.step()

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else np.nan,
               "finite_batches": finite_batches, "lr": optimizer.param_groups[0]["lr"],
               "elapsed_s": time.time() - t0}
        if epoch == 1 or epoch % train_cfg["validation_interval"] == 0 or epoch == train_cfg["epochs"]:
            val_kge, _, _ = evaluate_validation(
                net, attributes, eval_forcing, eval_obs, train_cfg["batch_size"], device,
                names, lower.to(torch.float64), parameter_range.to(torch.float64), win_cfg["warmup_days"])
            row["val_kge_mean"] = float(np.nanmean(val_kge))
            row["val_kge_median"] = float(np.nanmedian(val_kge))
            if row["val_kge_median"] > best_validation:
                best_validation = row["val_kge_median"]
                best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
                torch.save({"epoch": epoch, "state_dict": best_state, "val_kge_median": best_validation},
                           output_dir / "best_checkpoint.pt")
        history.append(row)
        print("epoch={epoch:03d} train_loss={loss:.5f} batches={batches} lr={lr:.2e}{val}".format(
            epoch=epoch, loss=row["train_loss"], batches=finite_batches, lr=row["lr"],
            val=(f" val_median={row['val_kge_median']:.4f}" if "val_kge_median" in row else "")), flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(output_dir / "epoch_history.csv", index=False)

    val_kge, parameters_norm, parameters_phys = evaluate_validation(
        net, attributes, eval_forcing, eval_obs, train_cfg["batch_size"], device,
        names, lower.to(torch.float64), parameter_range.to(torch.float64), win_cfg["warmup_days"])
    np.savez_compressed(output_dir / "best_parameters_normalized.npz", params=parameters_norm)
    np.savez_compressed(output_dir / "best_parameters_physical.npz", params=parameters_phys)
    with open(output_dir / "basin_final_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["basin_id", "basin_index", "val_kge"])
        writer.writeheader()
        for index, basin_id in enumerate(basin_ids):
            writer.writerow({"basin_id": basin_id, "basin_index": index, "val_kge": val_kge[index]})
    report = (
        "# HBV DPL KGE(Q) window ablation\n\n"
        f"Basins={n_basins}\n\n"
        f"Windows={len(windows)} × {win_cfg['warmup_days']} warmup + {win_cfg['prediction_days']} prediction\n\n"
        f"Excluded calibration tail={tail_days} days\n\n"
        f"Epochs={train_cfg['epochs']}\n\n"
        f"Validation KGE mean={np.nanmean(val_kge):.4f}, median={np.nanmedian(val_kge):.4f}\n"
    )
    (output_dir / "report.md").write_text(report)
    (output_dir / "COMPLETE").touch()
    print(report, flush=True)


if __name__ == "__main__":
    main()
