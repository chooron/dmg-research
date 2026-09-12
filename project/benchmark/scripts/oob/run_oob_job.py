#!/usr/bin/env python3
"""Run one PRIMARY-8 basin-held-out OOB dPL job.

Formal execution is intentionally a two-phase process:

1. TRAIN: only the selected fold's training basins and training-period targets
   are loaded; all selection and early stopping uses train loss.
2. EVAL: after restoring the exact best.pt, held-out test targets are loaded
   under an explicit phase gate and scored post-hoc.

``--smoke`` uses the same code path with a small basin subset and two epochs;
it is marked NON-SCIENTIFIC_SMOKE_ONLY and must never be treated as a result.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from oob_common import (
    BENCHMARK_ROOT,
    CURRENT_PHASE,
    EXPECTED_BASINS,
    KGE_EPS,
    PRIMARY8,
    BasinDataRepository,
    FoldDataCache,
    NormalizationStats,
    Phase,
    TargetAccessPolicy,
    append_csv,
    build_informative_window_catalog,
    fold_ids,
    gather_window,
    git_sha,
    kge_numpy,
    load_ids,
    load_oob_config,
    read_fold_assignment,
    require_phase,
    resolve_repo_path,
    set_phase,
    sha256_file,
    validate_fold_contract,
    validate_oob_protocol,
    write_json,
)

# The benchmark package is placed on sys.path by oob_common.
from dpl.nn_parameterizer import CatchmentParameterizer
from dpl.optimizer_transaction import FiniteOptimizerTransaction
from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from src.checkpointing import atomic_torch_save
from src.model_registry import NPARAM_INFO_36, build_model, get_spec
from src.objective import streaming_kge


SCRIPT_PATH = Path(__file__).resolve()


def _squeeze_streamflow(value: torch.Tensor) -> torch.Tensor:
    result = value
    while result.ndim > 2 and result.shape[-1] == 1:
        result = result.squeeze(-1)
    if result.ndim != 2:
        raise ValueError(f"expected streamflow [time, basin], got {tuple(result.shape)}")
    return result


def _align_scored(prediction: torch.Tensor, observation: torch.Tensor, warmup_days: int) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = _squeeze_streamflow(prediction)
    if prediction.shape[0] == observation.shape[0]:
        return prediction, observation
    if prediction.shape[0] + warmup_days == observation.shape[0]:
        return prediction, observation[warmup_days:]
    if prediction.shape[0] == observation.shape[0] + warmup_days:
        return prediction[warmup_days:], observation
    raise ValueError(
        f"cannot align hydrology output ({prediction.shape[0]}) and target "
        f"({observation.shape[0]}) with warmup={warmup_days}"
    )


def _train_loss(prediction: torch.Tensor, observation: torch.Tensor, warmup_days: int) -> torch.Tensor:
    prediction, observation = _align_scored(prediction, observation, warmup_days)
    scores, invalid = streaming_kge(
        prediction.unsqueeze(-1).unsqueeze(-1), observation, eps=KGE_EPS
    )
    scores = scores.squeeze(-1).squeeze(-1)
    invalid = invalid.squeeze(-1).squeeze(-1)
    if bool(invalid.any()) or not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("training KGE was invalid; refusing to continue")
    return 1.0 - scores.mean()


def _initialize_midpoint(network: CatchmentParameterizer) -> None:
    if not hasattr(network, "net") or not isinstance(network.net[-1], nn.Linear):
        raise ValueError("the frozen OOB contract expects the legacy parameterizer output layer")
    with torch.no_grad():
        network.net[-1].weight.zero_()
        network.net[-1].bias.zero_()


def _backend(config: dict[str, Any], device: torch.device, forced: str | None) -> str:
    selected = forced or config["protocol"].get("backend", "auto")
    if selected == "auto":
        return "compile" if device.type == "cuda" and hasattr(torch, "compile") else "eager"
    if selected not in {"eager", "compile"}:
        raise ValueError(f"unsupported backend: {selected}")
    if selected == "compile" and not hasattr(torch, "compile"):
        raise RuntimeError("protocol requested torch.compile but it is unavailable")
    return selected


def _make_parameterizer(model_name: str, attributes: torch.Tensor, config: dict[str, Any], device: torch.device) -> CatchmentParameterizer:
    spec = get_spec(model_name, device=device)
    protocol = config["protocol"]
    network = CatchmentParameterizer(
        in_features=attributes.shape[1],
        out_features=NPARAM_INFO_36[model_name],
        hidden_dims=[int(value) for value in protocol["parameterizer_hidden_dims"]],
        dropout=float(protocol["parameterizer_dropout"]),
        architecture=protocol["parameterizer_architecture"],
        parameter_names=list(spec.parameter_names),
        parameter_groups=spec.parameter_groups,
        output_transform="sigmoid",
    ).to(device=device, dtype=torch.float64)
    _initialize_midpoint(network)
    return network


def _checkpoint_payload(
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    model_name: str,
    fold: int,
    epoch: int,
    train_ids: np.ndarray,
    heldout_ids: np.ndarray,
    best_train_loss: float,
    config: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    protocol = config["protocol"]
    return {
        "epoch": int(epoch),
        "network": {key: value.detach().cpu().clone() for key, value in network.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "metadata": {
            "experiment_id": config["experiment_id"],
            "model": model_name,
            "fold": int(fold),
            "best_epoch": int(epoch),
            "selection_metric_name": "train_loss",
            "selection_metric_value": float(best_train_loss),
            "seed": int(protocol["seed"]),
            "parameter_mapping": protocol["parameter_mapping"],
            "optimizer": protocol["optimizer"],
            "lr": float(protocol["lr"]),
            "weight_decay": float(protocol["weight_decay"]),
            "scheduler": protocol["scheduler"],
            "clip_norm": float(protocol["clip_norm"]),
            "kge_eps": float(protocol["kge_eps"]),
            "training_warmup_days": int(protocol["training_warmup_days"]),
            "scored_horizon_days": int(protocol["scored_horizon_days"]),
            "backend": backend,
            "train_basin_count": int(train_ids.size),
            "heldout_basin_count": int(heldout_ids.size),
            "heldout_target_access_during_training": 0,
            "test_informed_selection": False,
            "git_sha": git_sha(),
        },
    }


def _train_fold(
    *,
    model_name: str,
    fold: int,
    train_ids: np.ndarray,
    heldout_ids: np.ndarray,
    data: Any,
    attributes: np.ndarray,
    stats: NormalizationStats,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    backend_override: str | None,
) -> dict[str, Any]:
    protocol = config["protocol"]
    warmup = int(protocol["training_warmup_days"])
    scored = int(protocol["scored_horizon_days"])
    horizon = warmup + scored
    if data.x.shape[0] < horizon:
        raise ValueError("training period is shorter than warmup plus scored horizon")
    if data.x.shape[0] != data.y.shape[0]:
        raise ValueError("training forcing and target lengths differ")

    catalog = build_informative_window_catalog(data.y[warmup:].numpy().T, scored_days=scored)
    attrs = torch.as_tensor(stats.transform(attributes), dtype=torch.float64, device=device)
    x = data.x.to(device=device)
    x = _add_calendar_if_needed(x, data.dates, model_name)
    y = data.y.to(device=device)
    if attrs.shape[0] != train_ids.size or x.shape[1] != train_ids.size:
        raise ValueError("training attribute/data basin dimensions do not match fold IDs")

    network = _make_parameterizer(model_name, attrs, config, device)
    backend = _backend(config, device, backend_override)
    hydro = build_model(
        model_name,
        device,
        warm_up=warmup,
        backend=backend,
        parameter_mapping=protocol["parameter_mapping"],
        warmup_grad_mode=protocol["warmup_grad_mode"],
        dtype=torch.float64,
    )
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=float(protocol["lr"]), weight_decay=float(protocol["weight_decay"])
    )
    transaction = FiniteOptimizerTransaction(
        optimizer,
        network.parameters(),
        clip_norm=float(protocol["clip_norm"]),
        named_parameters=network.named_parameters(),
    )

    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    plateau_best = float("inf")
    epoch_rows: list[dict[str, Any]] = []
    clip_count = 0
    update_count = 0
    for epoch in range(1, epochs + 1):
        network.train()
        losses: list[float] = []
        pre_norms: list[float] = []
        post_norms: list[float] = []
        clipped: list[bool] = []
        epoch_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        epoch_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if epoch_start is not None and epoch_end is not None:
            epoch_start.record()

        for step in range(steps_per_epoch):
            selected = torch.randperm(train_ids.size, device=device)[: min(batch_size, train_ids.size)]
            starts = torch.as_tensor(
                [int(catalog[int(index)][torch.randint(len(catalog[int(index)]), (), device=device).item()]) for index in selected],
                dtype=torch.long,
                device=device,
            )
            x_batch = gather_window(x, starts, selected, horizon)
            y_batch = gather_window(y, starts, selected, horizon)
            optimizer.zero_grad(set_to_none=True)
            theta = network(attrs[selected])
            prediction = hydro({"x_phy": x_batch}, (None, theta.unsqueeze(-1)))["streamflow"]
            loss = _train_loss(prediction, y_batch, warmup_days=warmup)
            result = transaction.step(loss, epoch=epoch, batch_index=step, basin_ids=train_ids[selected.detach().cpu().numpy()])
            diagnostics = result.diagnostics
            losses.append(float(loss.detach().cpu()))
            pre_norms.append(float(diagnostics.get("pre_clip_grad_norm") or 0.0))
            post_norms.append(float(diagnostics.get("post_clip_grad_norm") or 0.0))
            clipped.append(bool(diagnostics.get("clipped", False)))
            update_count += 1

        if epoch_start is not None and epoch_end is not None:
            epoch_end.record()
            torch.cuda.synchronize(device)
            seconds = float(epoch_start.elapsed_time(epoch_end) / 1000.0)
        else:
            seconds = 0.0
        current_loss = float(np.mean(losses))
        clip_fraction = float(np.mean(clipped)) if clipped else 0.0
        clip_count += sum(clipped)
        epoch_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "epoch": epoch,
                "train_loss_1_minus_kge": current_loss,
                "selection_metric": "train_loss",
                "grad_norm_preclip_median": float(np.median(pre_norms)) if pre_norms else 0.0,
                "grad_norm_preclip_max": float(np.max(pre_norms)) if pre_norms else 0.0,
                "grad_norm_postclip_median": float(np.median(post_norms)) if post_norms else 0.0,
                "grad_clip_fraction": clip_fraction,
                "train_nonfinite": 0,
                "seconds": seconds,
                "updates": update_count,
            }
        )
        append_csv(run_dir / "epoch_metrics.csv", [epoch_rows[-1]])

        # Exact checkpoint selection and plateau significance are separate: a
        # strictly lower train loss must update best.pt even when the improvement
        # is smaller than plateau_eps.
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            payload = _checkpoint_payload(
                network,
                optimizer,
                model_name=model_name,
                fold=fold,
                epoch=epoch,
                train_ids=train_ids,
                heldout_ids=heldout_ids,
                best_train_loss=best_loss,
                config=config,
                backend=backend,
            )
            atomic_torch_save(payload, run_dir / "best.pt")

        if current_loss < plateau_best - float(protocol["plateau_eps"]):
            plateau_best = current_loss
            stale = 0
        else:
            stale += 1

        if epoch >= int(protocol["min_epochs"]) and stale >= int(protocol["patience"]):
            status = "PLATEAU_STOP"
            break
    else:
        status = "COMPLETED"

    best_path = run_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("training completed without exact best.pt")
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    network.load_state_dict(best_payload["network"])
    for parameter in network.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError("best.pt contains non-finite parameter")
    if int(best_payload["metadata"]["best_epoch"]) != best_epoch:
        raise RuntimeError("best.pt metadata epoch does not match in-memory selection")

    health = {
        "status": status,
        "model": model_name,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
        "stop_epoch": epoch,
        "min_epochs": int(protocol["min_epochs"]),
        "patience": int(protocol["patience"]),
        "selection_metric": "train_loss",
        "clip_fraction": float(clip_count / max(update_count, 1)),
        "optimizer_updates": update_count,
        "train_basin_count": int(train_ids.size),
        "heldout_basin_count": int(heldout_ids.size),
        "heldout_target_access_during_training": 0,
        "test_informed_selection": False,
        "finite_parameters": True,
        "backend": backend,
    }
    write_json(run_dir / "training_health.json", health)
    write_json(run_dir / "best_metadata.json", best_payload["metadata"])
    return {"network": network, "health": health, "backend": backend}


def _add_calendar_if_needed(x: torch.Tensor, dates: Sequence[Any], model_name: str) -> torch.Tensor:
    if model_name.lower() not in CALENDAR_MODELS:
        return x
    return add_calendar_forcing(x, dates, model_name=model_name)[0]


def _evaluate_heldout(
    *,
    model_name: str,
    fold: int,
    heldout_ids: np.ndarray,
    repository: BasinDataRepository,
    stats: NormalizationStats,
    cache: FoldDataCache,
    config: dict[str, Any],
    network: nn.Module,
    run_dir: Path,
    device: torch.device,
    backend_override: str | None,
    allow_subset: bool,
) -> dict[str, Any]:
    require_phase(Phase.EVAL, "held-out evaluation")
    protocol = config["protocol"]
    data = cache.load_heldout_evaluation(
        heldout_ids,
        TargetAccessPolicy(Phase.EVAL, heldout_ids),
        allow_subset=allow_subset,
    )
    attrs = torch.as_tensor(
        stats.transform(repository.load_canonical_attributes(heldout_ids)), dtype=torch.float64, device=device
    )
    x = data.x.to(device=device)
    y = data.y.to(device=device)
    x = _add_calendar_if_needed(x, data.dates, model_name)
    hydro = build_model(
        model_name,
        device,
        warm_up=int(protocol["evaluation_warmup_days"]),
        backend=_backend(config, device, backend_override),
        parameter_mapping=protocol["parameter_mapping"],
        warmup_grad_mode=protocol["warmup_grad_mode"],
        dtype=torch.float64,
    )
    network.eval()
    with torch.inference_mode():
        prediction = _squeeze_streamflow(hydro({"x_phy": x}, (None, network(attrs).unsqueeze(-1)))["streamflow"])
    warmup = int(data.warmup_days)
    prediction_np, observation_np = _align_scored(prediction, y, warmup)
    prediction_np = prediction_np.detach().cpu().numpy()
    observation_np = observation_np.detach().cpu().numpy()
    scores, invalid, pooled_stats = kge_numpy(prediction_np, observation_np, eps=float(protocol["kge_eps"]))
    if bool(invalid.any()) or not np.isfinite(scores).all():
        raise FloatingPointError("held-out evaluation produced invalid KGE")
    valid_points = np.isfinite(prediction_np) & np.isfinite(observation_np)
    append_csv(
        run_dir / "heldout_basin_kge.csv",
        [
            {"basin_id": int(basin_id), "kge": float(score), "valid_points": int(valid_points[:, index].sum())}
            for index, (basin_id, score) in enumerate(zip(heldout_ids, scores))
        ],
    )
    evaluation = {
        "model": model_name,
        "fold": fold,
        "phase": Phase.EVAL.value,
        "test_period": [config["data"]["test"]["start_time"], config["data"]["test"]["end_time"]],
        "evaluation_warmup_days": warmup,
        "scored_days": int(observation_np.shape[0]),
        "heldout_basin_count": int(heldout_ids.size),
        "valid_heldout_basin_count": int((~invalid).sum()),
        "kge_eps": float(protocol["kge_eps"]),
        "foldwise_mean_basin_kge": float(np.mean(scores)),
        "foldwise_median_basin_kge": float(np.median(scores)),
        "pooled_531_basin_kge": None,
        "pooled_statistics": pooled_stats,
        "heldout_target_access_during_training": 0,
        "test_informed_selection": False,
        "fold_contract": "outer held-out basins loaded only after exact best.pt restore",
    }
    # A single-fold pooled score is named explicitly; the final summarizer
    # combines sufficient statistics across all five held-out folds.
    evaluation["pooled_fold_kge"] = float(kge_numpy(prediction_np.reshape(-1, 1), observation_np.reshape(-1, 1), eps=float(protocol["kge_eps"]))[0][0])
    write_json(run_dir / "heldout_test_evaluation.json", evaluation)
    return evaluation


def _source_provenance(
    config: dict[str, Any],
    assignment_path: Path,
    repository: BasinDataRepository,
    cache: FoldDataCache,
 ) -> dict[str, Any]:
    paths = {
        "config": Path(config["_resolved_from"]),
        "assignment": assignment_path,
        "reference_ids": repository.reference_ids_path,
        "attributes": repository.attributes_path,
        "cache_metadata": cache.metadata_path,
        "train_forcing_cache": cache.cache_dir / "train_x.npy",
        "train_target_cache": cache.cache_dir / "train_y.npy",
        "runner": SCRIPT_PATH,
    }
    return {
        "git_sha": git_sha(),
        "files": {name: {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size} for name, path in paths.items()},
        "canonical_attribute_source": "caravan_671_attributes.npy",
        "normalization_scope": "train_basins_only",
        "target_access_policy": "TRAIN opens only fold train_y.npy; held-out heldout_y.npy opens only after Phase.EVAL",
        "monolithic_source_not_opened_by_training_worker": True,
    }


def run_job(
    *,
    model_name: str,
    fold: int,
    config: dict[str, Any],
    output_root: Path,
    assignment_path: Path,
    data_cache_root: Path,
    device: torch.device,
    smoke: bool = False,
    epochs_override: int | None = None,
    steps_override: int | None = None,
    batch_override: int | None = None,
    backend_override: str | None = None,
) -> dict[str, Any]:
    validate_oob_protocol(config)
    if model_name not in PRIMARY8:
        raise ValueError(f"model is not in frozen PRIMARY8: {model_name}")
    if not 0 <= fold < int(config["folds"]["n_splits"]):
        raise ValueError(f"fold must be in 0..{int(config['folds']['n_splits']) - 1}")
    if not assignment_path.exists():
        raise FileNotFoundError(f"frozen fold assignment missing: {assignment_path}")
    all_ids = load_ids(config["data"]["basin_ids"])
    rows = read_fold_assignment(assignment_path)
    validate_fold_contract(
        all_ids,
        rows,
        n_folds=int(config["folds"]["n_splits"]),
        expected_count=int(config["folds"].get("expected_basin_count", EXPECTED_BASINS)),
    )
    full_train_ids, full_heldout_ids = fold_ids(all_ids, rows, fold)
    train_ids, heldout_ids = full_train_ids, full_heldout_ids
    if smoke:
        subset_train = int(config.get("smoke", {}).get("train_basins", 4))
        subset_heldout = int(config.get("smoke", {}).get("heldout_basins", 2))
        train_ids = train_ids[:subset_train]
        heldout_ids = heldout_ids[:subset_heldout]
    elif any(value is not None for value in (epochs_override, steps_override, batch_override)):
        raise ValueError("epoch/step/batch overrides are allowed only for --smoke")

    run_dir = output_root / "runs" / model_name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".lock"
    done_path = run_dir / "DONE"
    if done_path.exists() and not smoke:
        return {"model": model_name, "fold": fold, "status": "ALREADY_COMPLETE"}
    done_path.unlink(missing_ok=True)
    (run_dir / "FAILED").unlink(missing_ok=True)
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise RuntimeError(f"job is already running: {run_dir}") from exc

    try:
        write_json(run_dir / "config.json", {**config, "smoke": smoke})
        (run_dir / "train_basin_ids.txt").write_text("\n".join(map(str, train_ids.tolist())) + "\n", encoding="utf-8")
        (run_dir / "heldout_basin_ids.txt").write_text("\n".join(map(str, heldout_ids.tolist())) + "\n", encoding="utf-8")
        (run_dir / "stdout.log").touch()
        (run_dir / "stderr.log").touch()

        seed = int(config["protocol"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        set_phase(Phase.TRAIN)
        repository = BasinDataRepository(config)
        cache = FoldDataCache(data_cache_root, fold, config)
        if not np.array_equal(cache.train_ids, full_train_ids) or not np.array_equal(cache.heldout_ids, full_heldout_ids):
            raise RuntimeError("fold data cache IDs do not match the frozen fold assignment")
        train_attributes = repository.load_canonical_attributes(train_ids)
        stats = NormalizationStats.fit(train_attributes, method="zscore", log_transform_skewed=True)
        stats.save(run_dir)
        data = cache.load_training_period(train_ids, TargetAccessPolicy(Phase.TRAIN, train_ids), allow_subset=smoke)
        write_json(run_dir / "source_provenance.json", _source_provenance(config, assignment_path, repository, cache))

        result = _train_fold(
            model_name=model_name,
            fold=fold,
            train_ids=train_ids,
            heldout_ids=heldout_ids,
            data=data,
            attributes=train_attributes,
            stats=stats,
            config=config,
            run_dir=run_dir,
            device=device,
            epochs=epochs_override or int(config["protocol"]["max_epochs"]),
            steps_per_epoch=steps_override or int(config["protocol"]["steps_per_epoch"]),
            batch_size=batch_override or int(config["protocol"]["batch_size"]),
            backend_override=backend_override,
        )
        del data, train_attributes
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        set_phase(Phase.EVAL)
        evaluation = _evaluate_heldout(
            model_name=model_name,
            fold=fold,
            heldout_ids=heldout_ids,
            repository=repository,
            stats=stats,
            cache=cache,
            config=config,
            network=result["network"],
            run_dir=run_dir,
            device=device,
            backend_override=backend_override,
            allow_subset=smoke,
        )
        result_payload = {
            "model": model_name,
            "fold": fold,
            "status": "SMOKE_PASS" if smoke else result["health"]["status"],
            "smoke_marker": "NON-SCIENTIFIC_SMOKE_ONLY" if smoke else None,
            "best_epoch": result["health"]["best_epoch"],
            "best_train_loss": result["health"]["best_train_loss"],
            "foldwise_mean_basin_kge": evaluation["foldwise_mean_basin_kge"],
            "foldwise_median_basin_kge": evaluation["foldwise_median_basin_kge"],
            "heldout_target_access_during_training": 0,
            "test_informed_selection": False,
            "normalization_scope": "train_basins_only",
        }
        write_json(run_dir / "result.json", result_payload)
        done_path.write_text("OK\n", encoding="utf-8")
        return result_payload
    except BaseException as exc:
        write_json(run_dir / "error.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        (run_dir / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
    finally:
        lock_path.unlink(missing_ok=True)
        set_phase(Phase.TRAIN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PRIMARY8)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--config", default="oob_primary8_5fold_20260902.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fold-assignment", default=None)
    parser.add_argument("--data-cache-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backend", choices=("eager", "compile"), default=None)
    args = parser.parse_args()
    config = load_oob_config(args.config)
    root_key = "smoke_root" if args.smoke else "root"
    output_root = resolve_repo_path(args.out or config["outputs"][root_key])
    assignment_path = resolve_repo_path(args.fold_assignment or config["folds"]["assignment_file"])
    data_cache_root = resolve_repo_path(args.data_cache_root or config["outputs"]["data_cache_root"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    result = run_job(
        model_name=args.model,
        fold=args.fold,
        config=config,
        output_root=output_root,
        assignment_path=assignment_path,
        data_cache_root=data_cache_root,
        device=device,
        smoke=args.smoke,
        epochs_override=args.epochs,
        steps_override=args.steps_per_epoch,
        batch_override=args.batch_size,
        backend_override=args.backend,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
