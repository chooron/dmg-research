#!/usr/bin/env python3
"""Evaluate frozen full300 CMA-ES parameters on train and test periods.

Selection is deliberately *only* the best checkpointed training KGE among the
ten starts.  Test KGE is computed afterwards and is never fed back into CMA-ES
or used to choose a start.  The script handles completed models incrementally,
so it may safely be re-run while a long production controller is still active.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/cmaes_36models")]

from src.data_selection import load_ids
from src.model_registry import build_model
from src.objective import streaming_kge
from src.production_config import load_resolved_config, validate_full_run_config


MARRMOT_NAMES = {"hbv96": "hbv"}


def _bounds(dates: pd.DatetimeIndex, period: dict[str, str]) -> tuple[int, int]:
    return (
        int(dates.get_loc(pd.Timestamp(period["start_time"]))),
        int(dates.get_loc(pd.Timestamp(period["end_time"]))) + 1,
    )


def load_warmup_and_period(
    basin_ids: np.ndarray, config: dict, period_name: str, device: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load repeated warm forcing plus one scored split; targets exclude warm-up."""
    dcfg, wcfg = config["data"], config["warmup"]
    with (ROOT / dcfg["data_path"]).open("rb") as handle:
        forcings, target, attributes = pickle.load(handle)
    reference = load_ids(dcfg["reference_ids"])
    index = np.asarray([np.where(reference == int(b))[0][0] for b in basin_ids], dtype=np.int64)
    dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
    warm_left, warm_right = _bounds(dates, wcfg["source"])
    period_left, period_right = _bounds(dates, dcfg[period_name])
    source_days, repetitions = warm_right - warm_left, int(wcfg["repetitions"])
    if source_days != int(wcfg["source_days"]):
        raise ValueError(f"warm-up source has {source_days} days, expected {wcfg['source_days']}")

    def with_doy(values: np.ndarray, day_values: np.ndarray) -> np.ndarray:
        doy = day_values.dayofyear.to_numpy()
        return np.concatenate((values, np.broadcast_to(doy[None, :, None], (*values.shape[:2], 1))), axis=2)

    warm = with_doy(forcings[index, warm_left:warm_right, :3], dates[warm_left:warm_right])
    scored = with_doy(forcings[index, period_left:period_right, :3], dates[period_left:period_right])
    forcing = np.concatenate(([warm] * repetitions) + [scored], axis=1)
    y = target[index, period_left:period_right, 0].copy()
    area = attributes[index, 11].astype(np.float64)
    y *= (0.0283168 * 86400.0 * 1e3 / (area * 1e6))[:, None]
    return (
        torch.as_tensor(forcing.transpose(1, 0, 2), dtype=torch.float32, device=device),
        torch.as_tensor(y.T, dtype=torch.float32, device=device),
        source_days * repetitions,
    )


def final_checkpoints(model_dir: Path, generations: int) -> list[Path]:
    pieces = sorted(model_dir.glob(f"chunk_*_gen_{generations}.pt"), key=lambda p: int(p.name.split("_")[1]))
    if not pieces:
        raise FileNotFoundError(f"No generation-{generations} checkpoints in {model_dir}")
    return pieces


def frozen_parameters(model_dir: Path, generations: int, expected_starts: int) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Return basin ids, selected latent parameters, and checkpoint train KGE.

    Each basin's maximum over independent starts is selected here. No score
    from the test period is observed by this function.
    """
    basin_parts: list[np.ndarray] = []
    latent_parts: list[torch.Tensor] = []
    score_parts: list[np.ndarray] = []
    for path in final_checkpoints(model_dir, generations):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        basin_ids = np.asarray(payload["basin_ids"], dtype=np.int64)
        state = payload["solver"]["state"]
        best = state["best_fitness"].detach().cpu().numpy()
        latent = state["best_latent"].detach().cpu()
        if best.size != basin_ids.size * expected_starts:
            raise ValueError(f"{path}: {best.size} scores do not match {basin_ids.size} x {expected_starts} starts")
        dimension = latent.shape[-1]
        best = best.reshape(basin_ids.size, expected_starts)
        latent = latent.reshape(basin_ids.size, expected_starts, dimension)
        selected_start = best.argmax(axis=1)
        basin_parts.append(basin_ids)
        latent_parts.append(latent[torch.arange(basin_ids.size), torch.as_tensor(selected_start)])
        score_parts.append(best[np.arange(basin_ids.size), selected_start])
    basin_ids = np.concatenate(basin_parts)
    if np.unique(basin_ids).size != basin_ids.size:
        raise ValueError(f"{model_dir.name}: duplicate basin checkpoint pieces")
    return basin_ids, torch.cat(latent_parts, dim=0), np.concatenate(score_parts)


def evaluate_period(model_name: str, latent: torch.Tensor, basin_ids: np.ndarray, config: dict, period: str, device: str, backend: str) -> np.ndarray:
    x, y, warmup_days = load_warmup_and_period(basin_ids, config, period, device)
    model = build_model(model_name, device, warm_up=warmup_days, backend=backend)
    # HydrologyModel expects [basin, parameter, group]. One group is one
    # already-frozen best training start for each basin.
    raw = torch.sigmoid(latent.to(device=device, dtype=torch.float64)).unsqueeze(-1).float()
    with torch.inference_mode():
        q = model({"x_phy": x}, (None, raw))["streamflow"].reshape(-1, len(basin_ids), 1, 1)
        score, invalid = streaming_kge(q, y)
    if bool(invalid.any()):
        count = int(invalid.sum().item())
        raise RuntimeError(f"{model_name} {period}: {count} frozen solutions yielded invalid KGE")
    return score[:, 0, 0].detach().cpu().numpy()


def _marrmot_file(directory: Path, model: str) -> Path | None:
    name = MARRMOT_NAMES.get(model, model)
    matches = sorted(directory.glob(f"m_*_{name}_*_obj1_params.csv"))
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous MARRMoT file for {model}: {matches}")
    return matches[0] if matches else None


def marrmot_scores(directory: Path, model: str) -> pd.DataFrame:
    path = _marrmot_file(directory, model)
    if path is None:
        return pd.DataFrame(columns=["basin_id", "marrmot_train_kge", "marrmot_test_kge"])
    frame = pd.read_csv(path, usecols=["gauge_id", "cal", "eval"])
    return frame.rename(columns={"gauge_id": "basin_id", "cal": "marrmot_train_kge", "eval": "marrmot_test_kge"}).astype({"basin_id": np.int64})


def median(values: pd.Series) -> float:
    return float(values.median()) if values.notna().any() else float("nan")


def evaluate_model(model_dir: Path, config: dict, device: str, backend: str, marrmot_dir: Path) -> tuple[pd.DataFrame, dict]:
    model = model_dir.name
    starts, generations = int(config["optimization"]["starts"]), int(config["optimization"]["generations"])
    basin_ids, latent, checkpoint_train = frozen_parameters(model_dir, generations, starts)
    train = evaluate_period(model, latent, basin_ids, config, "train", device, backend)
    test = evaluate_period(model, latent, basin_ids, config, "test", device, backend)
    frame = pd.DataFrame({"model": model, "basin_id": basin_ids, "selected_checkpoint_train_kge": checkpoint_train,
                          "train_kge": train, "test_kge": test})
    frame = frame.merge(marrmot_scores(marrmot_dir, model), on="basin_id", how="left", validate="one_to_one")
    common = frame.dropna(subset=["marrmot_train_kge", "marrmot_test_kge"]).copy()
    common["train_delta_vs_marrmot"] = common["train_kge"] - common["marrmot_train_kge"]
    common["test_delta_vs_marrmot"] = common["test_kge"] - common["marrmot_test_kge"]
    tol = 1e-6
    row = {
        "model": model, "n_basins": len(frame), "n_common_marrmot": len(common), "starts": starts,
        "generation": generations, "selection": "best_of_10_checkpoint_train_kge_only",
        "train_kge_median": median(frame["train_kge"]), "test_kge_median": median(frame["test_kge"]),
        "marrmot_train_kge_median_common": median(common["marrmot_train_kge"]),
        "marrmot_test_kge_median_common": median(common["marrmot_test_kge"]),
        "train_kge_median_common": median(common["train_kge"]), "test_kge_median_common": median(common["test_kge"]),
        "paired_train_delta_median": median(common["train_delta_vs_marrmot"]),
        "paired_test_delta_median": median(common["test_delta_vs_marrmot"]),
        "train_win_fraction": float((common["train_delta_vs_marrmot"] >= 0).mean()) if len(common) else float("nan"),
        "test_win_fraction": float((common["test_delta_vs_marrmot"] >= 0).mean()) if len(common) else float("nan"),
        "checkpoint_vs_recomputed_train_max_abs": float(np.max(np.abs(train - checkpoint_train))),
        "checkpoint_vs_recomputed_train_median_abs": float(np.median(np.abs(train - checkpoint_train))),
        "recompute_matches_checkpoint": bool(np.max(np.abs(train - checkpoint_train)) <= tol),
    }
    return frame, row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--marrmot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["eager", "compile"], default="compile")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--overlay-model-dir", action="append", default=[], metavar="MODEL=DIR",
                        help="Replace one model's checkpoint directory, e.g. tcm=/.../checkpoints/run/tcm")
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    validate_full_run_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_by_name = {p.name: p for p in args.checkpoint_root.iterdir() if p.is_dir() and (p / "DONE").is_file()}
    for raw in args.overlay_model_dir:
        model, separator, directory = raw.partition("=")
        path = Path(directory)
        if not separator or not model or not directory or not (path / "DONE").is_file():
            raise ValueError(f"--overlay-model-dir requires MODEL=completed_checkpoint_directory, got {raw!r}")
        candidates_by_name[model] = path
    candidates = [candidates_by_name[name] for name in sorted(candidates_by_name)]
    if args.models:
        allowed = set(args.models); candidates = [p for p in candidates if p.name in allowed]
    if not candidates:
        raise RuntimeError("No completed model directories selected")
    all_rows: list[pd.DataFrame] = []
    summaries: list[dict] = []
    failures: list[dict] = []
    for index, model_dir in enumerate(candidates, start=1):
        started = time.perf_counter()
        try:
            frame, row = evaluate_model(model_dir, config, args.device, args.backend, args.marrmot_dir)
            row["elapsed_s"] = time.perf_counter() - started
            all_rows.append(frame); summaries.append(row)
            print(json.dumps({"status": "evaluated", "model": model_dir.name, "elapsed_s": row["elapsed_s"]}), flush=True)
        except Exception as exc:
            failures.append({"model": model_dir.name, "error_type": type(exc).__name__, "error": str(exc)})
            print(json.dumps({"status": "failed", **failures[-1]}), flush=True)
        finally:
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    by_basin = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    by_model = pd.DataFrame(summaries).sort_values("model") if summaries else pd.DataFrame()
    by_basin.to_csv(args.output_dir / "full300_kge_by_basin.csv", index=False)
    by_model.to_csv(args.output_dir / "full300_kge_model_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "full300_kge_evaluation_failures.csv", index=False)
    overall = {
        "selection_rule": "best_of_10_by_train_kge_only; test KGE evaluated after parameters are frozen",
        "models_evaluated": len(summaries), "models_failed": len(failures),
        "model_median_train_kge_median": median(by_model["train_kge_median"]) if len(by_model) else float("nan"),
        "model_median_test_kge_median": median(by_model["test_kge_median"]) if len(by_model) else float("nan"),
        "marrmot_model_median_train_kge_median": median(by_model["marrmot_train_kge_median_common"]) if len(by_model) else float("nan"),
        "marrmot_model_median_test_kge_median": median(by_model["marrmot_test_kge_median_common"]) if len(by_model) else float("nan"),
    }
    (args.output_dir / "full300_kge_overall.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(json.dumps({"status": "complete", **overall}), flush=True)


if __name__ == "__main__":
    main()
