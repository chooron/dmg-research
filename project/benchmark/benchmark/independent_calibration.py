"""Independent basin-model-objective calibration runner.

Extends the base NSE/logNSE calibration to support KGE and KGE_LOG objectives,
and adds optional elite-pool mutation (inspired by EliteCalTrainer) to escape
local minima in multi-start gradient calibration.

All random starts are saved.  The best start (by train objective score) is
flagged but not the only row returned.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import BasinPeriod, CamelsStore
from .metrics import flow_diagnostics, kge_components, log_nse, nse
from .models import build_hydrology_model
from .objectives import objective_loss, kge_per_start

log = logging.getLogger(__name__)

# KGE_LOG epsilon: 1% of mean training flow per basin, stored in results
_KGE_LOG_EPS_FRAC = 0.01


def run_independent_calibration(
    config: dict[str, Any],
    basin_id: int | str,
    model_id: str,
    objective: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Run multi-start independent calibration for a single basin/model/objective.

    All random starts are retained in the output CSV.  The best start by
    training objective score is annotated with ``is_best_start = True``.

    Parameters
    ----------
    config : dict
        Loaded benchmark YAML config.
    basin_id : int | str
        CAMELS basin ID.
    model_id : str
        MARRMoT/dmotpy model name, e.g. ``"hbv96"``.
    objective : str
        One of ``"KGE"``, ``"KGE_LOG"``, ``"NSE"``, ``"LOG_NSE"``.
    output_dir : Path | None
        Override output directory (defaults to ``config["paths"]["output_dir"]``).

    Returns
    -------
    Path
        Path to the written ``results.csv``.
    """
    calibration = config["calibration"]
    device = str(calibration.get("device", "cpu"))
    seed = int(calibration.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    store = CamelsStore(config)
    periods = {
        name: store.period(basin_id, name, device=device)
        for name in ("train", "validation", "test")
    }

    model = build_hydrology_model(config, model_id, device=device).to(device)
    model.train()

    num_starts = int(calibration.get("num_random_starts", 64))
    epochs = int(calibration.get("epochs", 100))
    lr = float(calibration.get("learning_rate", 0.03))
    weight_decay = float(calibration.get("weight_decay", 0.0))
    grad_clip = float(calibration.get("grad_clip", 1.0))
    log_epsilon = float(calibration.get("log_epsilon", 1e-6))
    boundary_threshold = float(calibration.get("boundary_threshold", 0.02))

    # Elite mutation parameters
    elite_reset_interval = int(calibration.get("elite_reset_interval", 20))
    elite_reset_start = int(calibration.get("elite_reset_start", 30))
    elite_reset_end = int(calibration.get("elite_reset_end", 90))
    elite_threshold_ratio = float(calibration.get("elite_threshold_ratio", 0.25))
    elite_ratio = float(calibration.get("elite_ratio", 0.10))
    elite_noise_scale = float(calibration.get("elite_noise_scale", 0.05))
    use_elite_mutation = bool(calibration.get("use_elite_mutation", True))

    # Compute per-basin KGE_LOG epsilon (1% of mean train flow)
    train_obs = periods["train"].target.detach().cpu().numpy()
    train_obs_vals = train_obs[np.isfinite(train_obs)]
    kge_log_eps = float(np.nanmean(np.maximum(train_obs_vals, 0.0)) * _KGE_LOG_EPS_FRAC)
    kge_log_eps = max(kge_log_eps, log_epsilon)

    num_params = len(model.phy_param_names)
    # shape: [1, num_params, num_starts]
    initial = _lhs_uniform(num_starts, num_params, seed=seed)
    raw_logits = torch.nn.Parameter(
        _to_logit(initial).view(1, num_params, num_starts).to(device)
    )

    optimizer = torch.optim.Adam(
        [raw_logits],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 2), T_mult=1
    )

    start_time = time.perf_counter()
    final_loss = float("nan")
    success = True
    failure_reason = ""
    epoch_losses: list[float] = []

    try:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            normalized_params = torch.sigmoid(raw_logits)  # [1, num_params, num_starts]
            prediction = model(
                {"x_phy": periods["train"].x_phy},
                (None, normalized_params),
            )["streamflow"]  # [T, num_starts]

            loss = objective_loss(
                prediction,
                periods["train"].target,
                objective,
                log_epsilon=kge_log_eps if objective.upper() == "KGE_LOG" else log_epsilon,
            )

            if not torch.isfinite(loss):
                success = False
                failure_reason = f"non-finite loss at epoch {epoch}"
                log.warning(
                    "basin=%s model=%s obj=%s: non-finite loss at epoch %d, stopping.",
                    basin_id, model_id, objective, epoch,
                )
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw_logits], grad_clip)
            optimizer.step()
            scheduler.step()
            final_loss = float(loss.detach().cpu())
            epoch_losses.append(final_loss)

            # Elite mutation: reset poor starts to perturbed elite starts
            if (
                use_elite_mutation
                and num_starts > 1
                and epoch >= elite_reset_start
                and epoch <= elite_reset_end
                and (epoch - elite_reset_start) % elite_reset_interval == 0
            ):
                _apply_elite_mutation(
                    raw_logits=raw_logits,
                    optimizer=optimizer,
                    model=model,
                    periods=periods,
                    objective=objective,
                    log_epsilon=kge_log_eps if objective.upper() == "KGE_LOG" else log_epsilon,
                    num_starts=num_starts,
                    elite_ratio=elite_ratio,
                    threshold_ratio=elite_threshold_ratio,
                    noise_scale=elite_noise_scale,
                    epoch=epoch,
                    basin_id=basin_id,
                    model_id=model_id,
                )

    except Exception as exc:
        success = False
        failure_reason = str(exc)
        log.exception(
            "basin=%s model=%s obj=%s: calibration failed with exception.",
            basin_id, model_id, objective,
        )
    finally:
        runtime = time.perf_counter() - start_time

    with torch.no_grad():
        normalized_params = torch.sigmoid(raw_logits).detach()  # [1, num_params, num_starts]
        try:
            physical_params = model._descale_params(
                model.unpack_parameters((None, normalized_params))
            )
            physical_params_np = {
                key: value.detach().cpu().numpy()
                for key, value in physical_params.items()
            }
        except Exception as exc:
            log.warning("Failed to descale parameters: %s", exc)
            physical_params_np = {}

        predictions: dict[str, np.ndarray] = {}
        observations: dict[str, np.ndarray] = {}
        for split, period in periods.items():
            try:
                pred_t = model(
                    {"x_phy": period.x_phy},
                    (None, normalized_params),
                )["streamflow"].detach().cpu().numpy()
                predictions[split] = pred_t
                observations[split] = _aligned_observation(pred_t, period).detach().cpu().numpy()
            except Exception as exc:
                log.warning("Failed to run model on split %s: %s", split, exc)
                predictions[split] = np.full((1, num_starts), np.nan)
                observations[split] = np.full((1, num_starts), np.nan)

    # Identify best start by train objective score
    best_start_id = _best_start_index(
        predictions["train"], observations["train"], objective,
        kge_log_eps=kge_log_eps, log_epsilon=log_epsilon,
    )

    out_root = Path(output_dir or config["paths"]["output_dir"])
    task_dir = out_root / "independent_calibration" / str(basin_id) / model_id / _objective_key(objective)
    task_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(
        basin_id=basin_id,
        model_id=model_id,
        objective=objective,
        model=model,
        normalized_params=normalized_params.detach().cpu().numpy(),
        physical_params=physical_params_np,
        predictions=predictions,
        observations=observations,
        success=success,
        failure_reason=failure_reason,
        runtime=runtime,
        final_loss=final_loss,
        boundary_threshold=boundary_threshold,
        log_epsilon=log_epsilon,
        kge_log_eps=kge_log_eps,
        best_start_id=best_start_id,
        train_start=str(config["splits"]["train"]["start_time"]),
        train_end=str(config["splits"]["train"]["end_time"]),
        test_start=str(config["splits"]["test"]["start_time"]),
        test_end=str(config["splits"]["test"]["end_time"]),
    )

    result_path = task_dir / "results.csv"
    _write_rows(result_path, rows)

    if bool(calibration.get("save_simulations", True)):
        np.savez_compressed(
            task_dir / "simulations.npz",
            **{f"{split}_prediction": values for split, values in predictions.items()},
            **{f"{split}_observation": values for split, values in observations.items()},
        )

    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "basin_id": int(basin_id),
                "model_id": model_id,
                "objective": objective,
                "parameter_names": model.phy_param_names,
                "num_random_starts": num_starts,
                "epochs": epochs,
                "runtime": runtime,
                "optimization_success": success,
                "failure_reason": failure_reason,
                "final_loss": final_loss,
                "best_start_id": int(best_start_id) if best_start_id is not None else None,
                "kge_log_eps": kge_log_eps,
                "train_period": f"{config['splits']['train']['start_time']} / {config['splits']['train']['end_time']}",
                "test_period": f"{config['splits']['test']['start_time']} / {config['splits']['test']['end_time']}",
            },
            indent=2,
        )
    )
    log.info("Wrote independent calibration results to %s", result_path)
    return result_path


# ---------------------------------------------------------------------------
# Elite mutation helpers
# ---------------------------------------------------------------------------

def _apply_elite_mutation(
    *,
    raw_logits: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    model,
    periods: dict[str, Any],
    objective: str,
    log_epsilon: float,
    num_starts: int,
    elite_ratio: float,
    threshold_ratio: float,
    noise_scale: float,
    epoch: int,
    basin_id: Any,
    model_id: str,
) -> None:
    """Reset poor starts to perturbed versions of elite starts (logit domain).

    Mimics ``EliteCalTrainer._reset_poor_members`` but adapted for single-basin
    independent calibration where raw_logits has shape [1, num_params, num_starts].
    """
    n_elite = max(1, int(num_starts * elite_ratio))
    n_poor = max(1, int(num_starts * threshold_ratio))

    with torch.no_grad():
        # Compute per-start train scores
        normalized_params = torch.sigmoid(raw_logits)
        pred = model(
            {"x_phy": periods["train"].x_phy},
            (None, normalized_params),
        )["streamflow"].detach().cpu()

        obs = _aligned_observation(pred.numpy(), periods["train"])
        obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else obs

        scores = np.full(num_starts, np.nan)
        for m in range(num_starts):
            p_m = pred[:, m].numpy() if pred.ndim == 2 else pred[:, 0, m].numpy()
            o_m = obs_np[:, m] if obs_np.ndim == 2 else obs_np[:, 0]
            valid = np.isfinite(p_m) & np.isfinite(o_m)
            if valid.sum() < 2:
                continue
            p_v, o_v = p_m[valid], o_m[valid]
            if objective.upper() in ("KGE_LOG",):
                p_v = np.log(np.maximum(p_v, 0.0) + log_epsilon)
                o_v = np.log(np.maximum(o_v, 0.0) + log_epsilon)
            # KGE
            r = float(np.corrcoef(p_v, o_v)[0, 1])
            if np.isnan(r):
                continue
            beta = p_v.mean() / (o_v.mean() + 1e-10)
            gamma = p_v.std() / (o_v.std() + 1e-10)
            scores[m] = 1.0 - float(np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))

        filled = np.where(np.isnan(scores), -np.inf, scores)
        sorted_idx = np.argsort(filled)[::-1]
        elite_idx = sorted_idx[:n_elite]
        poor_idx = sorted_idx[-n_poor:]

        # Reset poor starts
        params_data = raw_logits.data  # [1, num_params, num_starts]
        n_reset = 0
        for poor_m in poor_idx:
            donor_m = int(np.random.choice(elite_idx))
            donor = params_data[0, :, donor_m].clone()
            noise = torch.randn_like(donor) * noise_scale
            params_data[0, :, poor_m] = donor + noise
            # Zero Adam momentum for reset starts
            if raw_logits in optimizer.state:
                state = optimizer.state[raw_logits]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key][0, :, poor_m] = 0.0
            n_reset += 1

    log.debug(
        "basin=%s model=%s obj=%s epoch=%d: elite mutation reset %d starts "
        "(elite KGE=%.4f, poor KGE=%.4f)",
        basin_id, model_id, objective, epoch, n_reset,
        float(np.nanmedian(scores[elite_idx])),
        float(np.nanmedian(scores[poor_idx])),
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _best_start_index(
    pred: np.ndarray,
    obs: np.ndarray,
    objective: str,
    kge_log_eps: float = 1e-3,
    log_epsilon: float = 1e-6,
) -> int | None:
    """Return index of the start with highest train-period score."""
    num_starts = pred.shape[1] if pred.ndim >= 2 else 1
    scores = np.full(num_starts, np.nan)

    for m in range(num_starts):
        p_m = pred[:, m] if pred.ndim == 2 else pred[:, 0, m]
        o_m = obs[:, m] if obs.ndim == 2 else obs[:, 0]
        valid = np.isfinite(p_m) & np.isfinite(o_m)
        if valid.sum() < 2:
            continue
        p_v, o_v = p_m[valid], o_m[valid]

        obj_key = objective.upper()
        if obj_key == "NSE":
            mean_o = o_v.mean()
            denom = ((o_v - mean_o) ** 2).sum()
            scores[m] = 1.0 - ((p_v - o_v) ** 2).sum() / (denom + 1e-10)
        elif obj_key == "LOG_NSE":
            eps = log_epsilon
            p_l = np.log(np.maximum(p_v, 0.0) + eps)
            o_l = np.log(np.maximum(o_v, 0.0) + eps)
            mean_ol = o_l.mean()
            denom = ((o_l - mean_ol) ** 2).sum()
            scores[m] = 1.0 - ((p_l - o_l) ** 2).sum() / (denom + 1e-10)
        elif obj_key in ("KGE", "KGE_LOG"):
            if obj_key == "KGE_LOG":
                eps = kge_log_eps
                p_v = np.log(np.maximum(p_v, 0.0) + eps)
                o_v = np.log(np.maximum(o_v, 0.0) + eps)
            r_val = float(np.corrcoef(p_v, o_v)[0, 1])
            if np.isnan(r_val):
                continue
            beta = p_v.mean() / (o_v.mean() + 1e-10)
            gamma = p_v.std() / (o_v.std() + 1e-10)
            scores[m] = 1.0 - float(np.sqrt((r_val - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))

    if np.all(np.isnan(scores)):
        return 0
    return int(np.nanargmax(scores))


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def _lhs_uniform(num_starts: int, num_params: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, num_starts + 1)
    samples = np.empty((num_starts, num_params), dtype=np.float32)
    for j in range(num_params):
        samples[:, j] = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(samples[:, j])
    samples = samples * 0.9 + 0.05
    return torch.from_numpy(samples)


def _to_logit(values: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(values, 1e-6, 1.0 - 1e-6)
    return torch.log(values / (1.0 - values))


def _aligned_observation(prediction: np.ndarray, period: BasinPeriod) -> torch.Tensor:
    observed = period.target[-prediction.shape[0]:, :, 0]
    if observed.shape[1] == 1 and prediction.shape[1] > 1:
        observed = observed.expand(-1, prediction.shape[1])
    return observed


# ---------------------------------------------------------------------------
# Row building and I/O
# ---------------------------------------------------------------------------

def _build_rows(
    *,
    basin_id: int | str,
    model_id: str,
    objective: str,
    model,
    normalized_params: np.ndarray,
    physical_params: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    success: bool,
    failure_reason: str,
    runtime: float,
    final_loss: float,
    boundary_threshold: float,
    log_epsilon: float,
    kge_log_eps: float,
    best_start_id: int | None,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> list[dict[str, Any]]:
    rows = []
    # normalized_params shape: [1, num_params, num_starts]
    # transpose to [num_starts, num_params]
    params_norm = normalized_params[0].T  # [num_starts, num_params]

    num_starts = params_norm.shape[0]

    for start_id in range(num_starts):
        param_norm = params_norm[start_id]  # [num_params]

        # Physical parameters for this start
        params_phys: dict[str, float] = {}
        for name in model.phy_param_names:
            arr = physical_params.get(name)
            if arr is not None:
                flat = np.asarray(arr).reshape(-1)
                params_phys[name] = float(flat[start_id]) if start_id < len(flat) else float("nan")
            else:
                params_phys[name] = float("nan")

        # Boundary flags per parameter
        boundary_flags = {
            name: bool(
                param_norm[i] <= boundary_threshold
                or param_norm[i] >= 1.0 - boundary_threshold
            )
            for i, name in enumerate(model.phy_param_names)
        }

        row: dict[str, Any] = {
            "basin_id": int(basin_id),
            "model_id": model_id,
            "objective": objective,
            "random_start_id": start_id,
            "is_best_start": bool(start_id == best_start_id),
            "train_period_start": train_start,
            "train_period_end": train_end,
            "test_period_start": test_start,
            "test_period_end": test_end,
            "kge_log_eps": kge_log_eps,
            "optimized_parameters": json.dumps(params_phys, sort_keys=True),
            "normalized_parameters": json.dumps(
                {name: float(param_norm[i]) for i, name in enumerate(model.phy_param_names)},
                sort_keys=True,
            ),
            "parameter_boundary_flags": json.dumps(boundary_flags, sort_keys=True),
            "any_boundary_flag": bool(any(boundary_flags.values())),
            "boundary_saturation_ratio": float(
                sum(boundary_flags.values()) / max(len(boundary_flags), 1)
            ),
            "optimization_success_flag": bool(success and np.all(np.isfinite(param_norm))),
            "success": bool(success),
            "failure_reason": failure_reason,
            "runtime": runtime,
            "final_loss": final_loss,
        }

        # Per-split metrics
        for split in ("train", "validation", "test"):
            if split not in predictions:
                for m_key in ("NSE", "logNSE", "KGE", "KGE_LOG"):
                    row[f"{split}_{m_key}"] = float("nan")
                continue
            pred = predictions[split]
            obs_arr = observations[split]

            # Handle shape: [T, num_starts] or [T, 1, num_starts]
            if pred.ndim == 3:
                pred = pred[:, 0, :]
            if obs_arr.ndim == 3:
                obs_arr = obs_arr[:, 0, :]

            p_s = pred[:, start_id] if pred.shape[1] > start_id else pred[:, 0]
            o_s = obs_arr[:, start_id] if obs_arr.shape[1] > start_id else obs_arr[:, 0]

            row[f"{split}_NSE"] = nse(p_s, o_s)
            row[f"{split}_logNSE"] = log_nse(p_s, o_s, eps=log_epsilon)
            row[f"{split}_KGE"] = _compute_kge_np(p_s, o_s)
            row[f"{split}_KGE_LOG"] = _compute_kge_np(p_s, o_s, log_transform=True, eps=kge_log_eps)

        # Detailed KGE components for test period only
        if "test" in predictions:
            pred_t = predictions["test"]
            obs_t = observations["test"]
            if pred_t.ndim == 3:
                pred_t = pred_t[:, 0, :]
            if obs_t.ndim == 3:
                obs_t = obs_t[:, 0, :]
            p_test = pred_t[:, start_id] if pred_t.shape[1] > start_id else pred_t[:, 0]
            o_test = obs_t[:, start_id] if obs_t.shape[1] > start_id else obs_t[:, 0]
            row.update(kge_components(p_test, o_test))
            row.update(flow_diagnostics(p_test, o_test))

        rows.append(row)

    return rows


def _compute_kge_np(
    pred: np.ndarray,
    obs: np.ndarray,
    log_transform: bool = False,
    eps: float = 1e-6,
) -> float:
    """Compute KGE between 1-D arrays, optionally on log-transformed values."""
    valid = np.isfinite(pred) & np.isfinite(obs)
    if valid.sum() < 2:
        return float("nan")
    p_v, o_v = pred[valid], obs[valid]
    if log_transform:
        p_v = np.log(np.maximum(p_v, 0.0) + eps)
        o_v = np.log(np.maximum(o_v, 0.0) + eps)
    corr = np.corrcoef(p_v, o_v)[0, 1]
    if np.isnan(corr):
        return float("nan")
    beta = p_v.mean() / (o_v.mean() + 1e-10)
    gamma = p_v.std() / (o_v.std() + 1e-10)
    return float(1.0 - np.sqrt((corr - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No result rows were produced.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _objective_key(objective: str) -> str:
    return objective.lower().replace("-", "_")
