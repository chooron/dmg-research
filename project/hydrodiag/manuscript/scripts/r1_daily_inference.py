"""Read-only daily R1 inference using the repository's production components."""

from __future__ import annotations

import hashlib
import json
import re
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunSpec:
    paradigm: str
    model_key: str
    model_label: str
    run_id: str
    source_checkpoint: Path
    source_configuration: Path
    selection_reason: str
    parameter_files: tuple[Path, ...] = ()


MODEL_LABELS = {"XAJ": "XAJ-Base", "XAJ_TGD2": "XAJ-TGD", "XAJ_CN": "XAJ-CN", "HBV": "HBV"}
IC_ROOTS = {
    "XAJ": "xaj_base_cmaes_531_batched_paired_v2",
    "XAJ_TGD2": "xaj_tgd2_cmaes_531_batched_v1",
    "XAJ_CN": "xaj_cn_cmaes_531_batched_paired_v2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(data_root: Path):
    project_root = data_root.parent / "project" / "hydrodiag"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from ablation.ic_core.data_adapter import load_531_bundle

    config = {
        "project_root": str(project_root),
        "dataset_path": str(data_root / "camels_dataset"),
        "gage_ids_path": str(data_root / "gage_id.npy"),
        "dates_path": str(data_root / "camels_dates.npy"),
        "basin_list_path": str(data_root / "531sub_id.txt"),
        "periods": {
            "warmup": {"start": "1980-10-01", "end": "1981-09-30"},
            "train": {"start": "1981-10-01", "end": "1995-09-30"},
            "test": {"start": "1995-10-01", "end": "2010-09-30"},
        },
    }
    return load_531_bundle(config), config


def read_ic_parameters(results_root: Path, model_key: str, basin_ids: Iterable[str]) -> tuple[np.ndarray, list[Path], list[int]]:
    raw_dir = results_root / IC_ROOTS[model_key] / "raw" / {"XAJ": "xaj", "XAJ_TGD2": "xaj_tgd2", "XAJ_CN": "xaj_cn"}[model_key]
    records: dict[str, list[tuple[float, int, Path, dict[str, Any]]]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text())
        basin = str(data["basin_id"]).zfill(8)
        train_kge = float(data.get("train_metrics", {}).get("kge", np.nan))
        if np.isfinite(train_kge) and data.get("status") == "complete":
            records.setdefault(basin, []).append((train_kge, int(data["start"]), path, data))
    selected = []
    for basin in basin_ids:
        candidates = records.get(basin, [])
        if not candidates:
            raise ValueError(f"No valid IC restart for {model_key} basin {basin}")
        selected.append(sorted(candidates, key=lambda item: (-item[0], item[1]))[0])
    parameter_names = selected[0][3]["parameter_names"]
    parameters = np.asarray([item[3]["parameters"] for item in selected], dtype=np.float32)
    if any(item[3]["parameter_names"] != parameter_names for item in selected):
        raise ValueError(f"IC parameter-name mismatch for {model_key}")
    return parameters, [item[2] for item in selected], [item[1] for item in selected]


def _valid_checkpoint(path: Path, model_key: str) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != model_key or not checkpoint.get("lite_mode", False):
        raise ValueError(f"checkpoint metadata mismatch: {path}")
    if not isinstance(checkpoint.get("state_dict"), dict) or not checkpoint["state_dict"]:
        raise ValueError(f"checkpoint has no state_dict: {path}")
    return checkpoint


def select_dpl_checkpoints(results_root: Path, model_key: str, requested_epoch: int | None = None) -> tuple[list[tuple[str, Path, Path, str]], dict[str, Any]]:
    root = results_root / ("dpl_camels_531_lite_v3_tgd2_dpl_audited" if model_key == "XAJ_TGD2" else "dpl_camels_531_lite_v2") / model_key
    seed_dirs = sorted(root.glob("seed_*"), key=lambda p: int(p.name.split("_")[-1]))
    if len(seed_dirs) != 3:
        raise ValueError(f"expected three dPL seeds for {model_key}, found {len(seed_dirs)}")
    selected: list[tuple[str, Path, Path, str]] = []
    if model_key == "XAJ_TGD2":
        common_epochs = None
        for seed_dir in seed_dirs:
            epochs = set()
            for path in seed_dir.glob("checkpoint_epoch_*.pt"):
                try:
                    epochs.add(int(path.stem.rsplit("_", 1)[-1]))
                except ValueError:
                    continue
            common_epochs = epochs if common_epochs is None else common_epochs & epochs
        if not common_epochs:
            raise ValueError("no common completed periodic TGD2 checkpoint epoch")
        epoch = requested_epoch if requested_epoch is not None else max(common_epochs)
        if epoch not in common_epochs:
            raise ValueError(f"requested TGD2 epoch {epoch} is not common to all seeds")
        for seed_dir in seed_dirs:
            checkpoint = seed_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            config = seed_dir / "config.json"
            metadata = _valid_checkpoint(checkpoint, model_key)
            history = pd.read_csv(seed_dir / "epoch_history.csv")
            if epoch not in set(history["epoch"].astype(int)) or int(metadata.get("epoch", -1)) != epoch:
                raise ValueError(f"TGD2 checkpoint/history mismatch: {checkpoint}")
            selected.append((seed_dir.name.removeprefix("seed_"), checkpoint, config, f"latest common valid periodic checkpoint epoch {epoch}"))
        return selected, {"root": str(root), "rule": "maximum checkpoint epoch common to all three seeds and present in epoch_history", "epoch": epoch}
    for seed_dir in seed_dirs:
        checkpoint = seed_dir / "best_checkpoint.pt"
        config = seed_dir / "config.json"
        _valid_checkpoint(checkpoint, model_key)
        selected.append((seed_dir.name.removeprefix("seed_"), checkpoint, config, "best_checkpoint.pt is the existing authoritative seed artifact"))
    return selected, {"root": str(root), "rule": "existing best_checkpoint.pt per seed", "epoch": None}


def _period_arrays(bundle, period: str, basin_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    periods = bundle.periods
    selector = basin_indices if basin_indices is not None else slice(None)
    if period == "train":
        forcing = bundle.forcing[selector, periods.train_forcing_start_index:periods.train_forcing_end_index, :]
        obs = bundle.target_mm_day[selector, periods.train.start_index:periods.train.end_index + 1]
        dates = pd.to_datetime(bundle.dates[periods.train.start_index:periods.train.end_index + 1])
    elif period == "test":
        forcing = bundle.forcing[selector, periods.test_forcing_start_index:periods.test_forcing_end_index, :]
        obs = bundle.target_mm_day[selector, periods.test.start_index:periods.test.end_index + 1]
        dates = pd.to_datetime(bundle.dates[periods.test.start_index:periods.test.end_index + 1])
    else:
        raise ValueError(period)
    return forcing, obs, dates


def _predict_ic(bundle, model_key: str, parameters: np.ndarray, device: str, batch_size: int, basin_indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    from ablation.ic_core.model_adapter import ModelAdapter
    import torch

    adapter = ModelAdapter(model_key, device=device, dtype=torch.float32, variant="lite")
    output: dict[str, np.ndarray] = {}
    for period in ("train", "test"):
        forcing, _obs, _dates = _period_arrays(bundle, period, basin_indices)
        indices = basin_indices if basin_indices is not None else np.arange(len(bundle.basin_ids))
        predicted = np.full((len(indices), forcing.shape[1]), np.nan, dtype=np.float32)
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            force = torch.from_numpy(forcing[start:stop])
            params = torch.from_numpy(parameters[indices[start:stop]])
            q, _ = adapter.run_model(force, params, temp_mean_train=torch.from_numpy(bundle.temp_mean_train[indices[start:stop]]), temp_std_train=torch.from_numpy(bundle.temp_std_train[indices[start:stop]]))
            predicted[start:stop] = q.detach().cpu().numpy()
        warmup = bundle.periods.warmup.days
        output[period] = predicted[:, warmup:]
    return output


def _predict_dpl(bundle, model_key: str, config_path: Path, checkpoint_path: Path, data_root: Path, device: str, batch_size: int, basin_indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    import torch
    from training.dpl.run_dpl_model import LITE_MODEL_REGISTRY, StaticParameterNet, physical_parameters, robust_normalize

    config = json.loads(config_path.read_text())
    config["gage_ids_path"] = str(data_root / "gage_id.npy")
    config["dates_path"] = str(data_root / "camels_dates.npy")
    config["data_pkl_dataset"] = str(data_root / "camels_dataset")
    config["data_basin_ids"] = str(data_root / "531sub_id.txt")
    model_cls, specs = LITE_MODEL_REGISTRY[model_key]
    all_attrs_np, _ = robust_normalize(bundle.raw_attributes.astype(np.float32))
    indices = basin_indices if basin_indices is not None else np.arange(len(bundle.basin_ids))
    attrs_np = all_attrs_np[indices]
    net_cfg = config["network"]
    hidden_sizes = [int(v) for v in net_cfg.get("hidden_sizes", [net_cfg["hidden_size"]] * net_cfg.get("depth", 2))]
    net = StaticParameterNet(attrs_np.shape[1], specs, hidden_sizes, net_cfg["dropout"], net_cfg["output_epsilon"]).to(device).eval()
    checkpoint = _valid_checkpoint(checkpoint_path, model_key)
    net.load_state_dict(checkpoint["state_dict"])
    model = model_cls().to(device).eval()
    names = list(specs)
    lower = torch.tensor([specs[name]["lower"] for name in names], device=device, dtype=torch.float32)
    upper = torch.tensor([specs[name]["upper"] for name in names], device=device, dtype=torch.float32)
    parameter_range = upper - lower
    attributes = torch.from_numpy(attrs_np)
    output: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for period in ("train", "test"):
            forcing, _obs, _dates = _period_arrays(bundle, period, indices)
            predicted = np.full((len(indices), forcing.shape[1]), np.nan, dtype=np.float32)
            for start in range(0, len(indices), batch_size):
                stop = min(start + batch_size, len(indices))
                theta = net(attributes[start:stop].to(device))
                params = physical_parameters(theta, names, lower, parameter_range)
                fc = {
                    "precip": torch.from_numpy(forcing[start:stop, :, 0]).to(device=device, dtype=torch.float32),
                    "temp": torch.from_numpy(forcing[start:stop, :, 1]).to(device=device, dtype=torch.float32),
                    "pet": torch.from_numpy(forcing[start:stop, :, 2]).to(device=device, dtype=torch.float32),
                    "temp_mean_train": torch.from_numpy(bundle.temp_mean_train[indices[start:stop]]).to(device=device, dtype=torch.float32),
                    "temp_std_train": torch.from_numpy(bundle.temp_std_train[indices[start:stop]]).to(device=device, dtype=torch.float32),
                }
                q, _ = model(forcings=fc, params=params)
                predicted[start:stop] = q.detach().cpu().numpy()
            output[period] = predicted[:, bundle.periods.warmup.days:]
    return output


def _write_run_parquet(path: Path, spec: RunSpec, bundle, predictions: dict[str, np.ndarray], parameter_files: list[Path], basin_indices: np.ndarray | None = None, batch_basins: int = 16) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.unlink(missing_ok=True)
    writer = None
    row_count = 0
    periods = bundle.periods
    indices = basin_indices if basin_indices is not None else np.arange(len(bundle.basin_ids))
    basin_ids = np.asarray(bundle.basin_ids)[indices]
    try:
        for period in ("train", "test"):
            _forcing, obs, dates = _period_arrays(bundle, period, indices)
            qsim = predictions[period]
            for start in range(0, len(indices), batch_basins):
                stop = min(start + batch_basins, len(indices))
                count = stop - start
                frame = pd.DataFrame({
                    "basin_id": np.repeat(basin_ids[start:stop], len(dates)),
                    "paradigm": spec.paradigm,
                    "model": spec.model_label,
                    "seed_or_restart": spec.run_id,
                    "selected_run": True,
                    "period": period,
                    "date": np.tile(dates.to_numpy(dtype="datetime64[ns]"), count),
                    "q_obs": obs[start:stop].reshape(-1).astype(np.float64),
                    "q_sim": qsim[start:stop].reshape(-1).astype(np.float64),
                    "discharge_unit": "mm/day",
                    "valid_obs": np.isfinite(obs[start:stop]).reshape(-1) & (obs[start:stop].reshape(-1) >= 0),
                    "valid_sim": np.isfinite(qsim[start:stop]).reshape(-1) & (qsim[start:stop].reshape(-1) >= 0),
                    "source_checkpoint_or_parameter_file": str(spec.source_checkpoint) if not parameter_files else np.repeat([str(parameter_files[index]) for index in indices[start:stop]], len(dates)),
                    "source_configuration": str(spec.source_configuration),
                })
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                row_count += len(frame)
    finally:
        if writer is not None:
            writer.close()
    return {"file": str(path), "rows": row_count, "sha256": sha256(path), "basins": len(bundle.basin_ids), "periods": ["train", "test"], "date_start": str(periods.warmup.start), "date_end": str(periods.test.end), "status": "complete"}


def _online_metric_rows(spec: RunSpec, bundle, predictions: dict[str, np.ndarray], parameter_files: list[Path], basin_indices: np.ndarray | None = None) -> list[dict[str, Any]]:
    from r1_statistics import standard_kge

    rows: list[dict[str, Any]] = []
    indices = basin_indices if basin_indices is not None else np.arange(len(bundle.basin_ids))
    for period in ("train", "test"):
        _forcing, observations, dates = _period_arrays(bundle, period, indices)
        simulations = predictions[period]
        for local_index, basin_index in enumerate(indices):
            basin = bundle.basin_ids[basin_index]
            observed = observations[local_index].astype(float)
            simulated = simulations[local_index].astype(float)
            kge, valid_observation_count, valid_simulation_count, valid_days, valid_metric = standard_kge(simulated, observed)
            mask = np.isfinite(observed) & np.isfinite(simulated) & (observed >= 0) & (simulated >= 0)
            error = simulated[mask] - observed[mask]
            denominator = float(observed[mask].sum())
            nse_denominator = float(np.sum((observed[mask] - observed[mask].mean()) ** 2)) if valid_days else 0.0
            nse = float(1.0 - np.sum(error ** 2) / nse_denominator) if nse_denominator > 0 else np.nan
            pbias = float(100.0 * error.sum() / denominator) if denominator != 0 else np.nan
            rmse = float(np.sqrt(np.mean(error ** 2))) if valid_days else np.nan
            source_file = parameter_files[basin_index] if parameter_files else spec.source_checkpoint
            run_id = spec.run_id
            if spec.paradigm == "IC-CMA-ES":
                match = re.search(r"_start(\d+)\.json", str(source_file))
                run_id = f"restart_{int(match.group(1)):02d}" if match else run_id
            rows.append({
                "basin_id": str(basin).zfill(8), "paradigm": spec.paradigm, "model": spec.model_label,
                "period": period, "seed_or_restart": run_id, "selected_run": True, "kge": kge,
                "kge_prime": np.nan, "nse": nse, "pbias": pbias, "rmse": rmse,
                "valid_observation_count": valid_observation_count, "valid_simulation_count": valid_simulation_count,
                "valid_days": valid_days, "period_start": dates.min().date().isoformat(), "period_end": dates.max().date().isoformat(),
                "discharge_unit": "mm/day", "status": "valid" if valid_metric else "invalid_metric",
                "source_file": str(source_file), "source_checkpoint_or_parameter_file": str(source_file),
                "source_configuration": str(spec.source_configuration),
            })
    return rows


def _online_signature_year_rows(spec: RunSpec, bundle, predictions: dict[str, np.ndarray], basin_indices: np.ndarray | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indices = basin_indices if basin_indices is not None else np.arange(len(bundle.basin_ids))
    for period in ("train", "test"):
        _forcing, observations, dates = _period_arrays(bundle, period, indices)
        simulations = predictions[period]
        dates = pd.DatetimeIndex(dates)
        water_years = dates.year + (dates.month >= 10).astype(int)
        for local_index, basin_index in enumerate(indices):
            basin = bundle.basin_ids[basin_index]
            for water_year in sorted(set(water_years)):
                year_mask = water_years == water_year
                year_dates = dates[year_mask]
                observed = observations[local_index, year_mask].astype(float)
                simulated = simulations[local_index, year_mask].astype(float)
                row = {"basin_id": str(basin).zfill(8), "paradigm": spec.paradigm, "model": spec.model_label, "seed_or_restart": spec.run_id, "period": period, "water_year": int(water_year), "status": "incomplete_water_year"}
                expected_days = int((year_dates[-1] - year_dates[0]).days + 1) if len(year_dates) else 0
                complete = len(year_dates) == expected_days and len(set(year_dates)) == expected_days and bool((np.isfinite(observed) & np.isfinite(simulated) & (observed >= 0) & (simulated >= 0)).all())
                if complete:
                    observed_total, simulated_total = float(observed.sum()), float(simulated.sum())
                    ct_obs = int(np.argmax(np.cumsum(observed) >= 0.5 * observed_total) + 1) if observed_total > 0 else np.nan
                    ct_sim = int(np.argmax(np.cumsum(simulated) >= 0.5 * simulated_total) + 1) if simulated_total > 0 else np.nan
                    april_july = (year_dates.month >= 4) & (year_dates.month <= 7)
                    amjj_obs = float(observed[april_july].sum() / observed_total) if observed_total > 0 else np.nan
                    amjj_sim = float(simulated[april_july].sum() / simulated_total) if simulated_total > 0 else np.nan
                    row.update({"ct_obs": ct_obs, "ct_sim": ct_sim, "ct_error_signed": ct_sim - ct_obs, "ct_error_absolute": abs(ct_sim - ct_obs), "spo_obs": np.nan, "spo_sim": np.nan, "spo_error_signed": np.nan, "spo_error_absolute": np.nan, "amjj_obs": amjj_obs, "amjj_sim": amjj_sim, "amjj_error_signed": amjj_sim - amjj_obs, "amjj_error_absolute": abs(amjj_sim - amjj_obs), "status": "valid_ct_amjj_spo_unresolved", "spo_status": "unresolved_definition_search_window"})
                else:
                    row.update({column: np.nan for column in ("ct_obs", "ct_sim", "ct_error_signed", "ct_error_absolute", "spo_obs", "spo_sim", "spo_error_signed", "spo_error_absolute", "amjj_obs", "amjj_sim", "amjj_error_signed", "amjj_error_absolute", "spo_status")})
                rows.append(row)
    return rows


def run_daily_export(project_root: Path, results_root: Path, data_root: Path, output_root: Path, device: str = "cuda", batch_size: int = 16, model_keys: tuple[str, ...] | None = None, tgd2_epoch: int | None = None, paradigm: str = "all", partition_count: int = 1, partition_index: int = 0, partition_suffix: str = "") -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / ".r1_torchinductor_cache"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    bundle, bundle_config = load_bundle(data_root)
    if partition_count < 1 or not 0 <= partition_index < partition_count:
        raise ValueError(f"invalid basin partition {partition_index}/{partition_count}")
    basin_indices = np.array_split(np.arange(len(bundle.basin_ids)), partition_count)[partition_index]
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for R1 inference but no CUDA device is available; refusing CPU fallback")
    specs: list[RunSpec] = []
    selected_tgd2: dict[str, Any] | None = None
    selected_model_keys = model_keys or ("XAJ", "XAJ_TGD2", "XAJ_CN", "HBV")
    if paradigm in ("all", "ic"):
        for model_key in selected_model_keys:
            if model_key == "HBV":
                continue
            parameters, files, starts = read_ic_parameters(results_root, model_key, bundle.basin_ids)
            specs.append(RunSpec("IC-CMA-ES", model_key, MODEL_LABELS[model_key], f"selected_restart", files[0], results_root / IC_ROOTS[model_key] / "manifest.json", "maximum train-period stored KGE; tie=min restart", tuple(files)))
    dpl_runs: list[tuple[str, str, Path, Path, str]] = []
    if paradigm in ("all", "dpl"):
        for model_key in selected_model_keys:
            selected, selection_meta = select_dpl_checkpoints(results_root, model_key, requested_epoch=tgd2_epoch if model_key == "XAJ_TGD2" else None)
            if model_key == "XAJ_TGD2":
                selected_tgd2 = {**selection_meta, "seeds": [{"seed": seed, "checkpoint": str(checkpoint), "epoch": int(__import__("torch").load(checkpoint, map_location="cpu", weights_only=False).get("epoch", -1)), "timestamp": pd.Timestamp(checkpoint.stat().st_mtime, unit="s", tz="UTC").isoformat(), "config": str(config)} for seed, checkpoint, config, _reason in selected]}
            for seed, checkpoint, config, reason in selected:
                dpl_runs.append((model_key, seed, checkpoint, config, reason))
    inventory: list[dict[str, Any]] = []
    online_metric_rows: list[dict[str, Any]] = []
    online_signature_year_rows: list[dict[str, Any]] = []
    for spec in specs:
        parameters, parameter_files, _starts = read_ic_parameters(results_root, spec.model_key, bundle.basin_ids)
        predictions = _predict_ic(bundle, spec.model_key, parameters, device, batch_size, basin_indices)
        path = output_root / f"r1_daily_simulations_ic_{spec.model_key.lower()}{partition_suffix}.parquet"
        row = _write_run_parquet(path, spec, bundle, predictions, parameter_files, basin_indices)
        row.update({"model": spec.model_label, "paradigm": spec.paradigm, "seed_or_restart": spec.run_id, "source_selection": spec.selection_reason})
        inventory.append(row)
        online_metric_rows.extend(_online_metric_rows(spec, bundle, predictions, parameter_files, basin_indices))
        online_signature_year_rows.extend(_online_signature_year_rows(spec, bundle, predictions, basin_indices))
    for model_key, seed, checkpoint, config, reason in dpl_runs:
        spec = RunSpec("dPL-MLP", model_key, MODEL_LABELS[model_key], f"seed_{seed}", checkpoint, config, reason)
        predictions = _predict_dpl(bundle, model_key, config, checkpoint, data_root, device, batch_size, basin_indices)
        path = output_root / f"r1_daily_simulations_dpl_{model_key.lower()}_seed_{seed}{partition_suffix}.parquet"
        row = _write_run_parquet(path, spec, bundle, predictions, [], basin_indices)
        row.update({"model": spec.model_label, "paradigm": spec.paradigm, "seed_or_restart": spec.run_id, "source_selection": reason, "checkpoint_epoch": int(__import__("torch").load(checkpoint, map_location="cpu", weights_only=False).get("epoch", -1))})
        inventory.append(row)
        online_metric_rows.extend(_online_metric_rows(spec, bundle, predictions, [], basin_indices))
        online_signature_year_rows.extend(_online_signature_year_rows(spec, bundle, predictions, basin_indices))
    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv(output_root / "r1_daily_simulation_inventory.csv", index=False)
    online_metrics = pd.DataFrame(online_metric_rows)
    online_signature_years = pd.DataFrame(online_signature_year_rows)
    online_metrics.to_csv(output_root / "r1_online_performance.csv", index=False)
    online_signature_years.to_csv(output_root / "r1_online_signature_basin_year.csv", index=False)
    audit = {
        "periods": bundle.periods.as_dict(),
        "basin_count": len(bundle.basin_ids),
        "partition": {"count": partition_count, "index": partition_index, "basins_in_partition": int(len(basin_indices)), "first_basin": str(bundle.basin_ids[basin_indices[0]]), "last_basin": str(bundle.basin_ids[basin_indices[-1]])},
        "execution": "partitions are launched one at a time; each partition uses GPU batched inference",
        "device": device,
        "batch_size": int(batch_size),
        "basin_list": str(data_root / "531sub_id.txt"),
        "forcing_source": str(data_root / "camels_dataset"),
        "forcing_order": ["P", "T", "PET"],
        "observation_source": str(data_root / "camels_dataset"),
        "observation_raw_unit": "ft3/s",
        "observation_model_unit": "mm/day",
        "conversion": "repository ablation.ic_core.data_adapter.convert_ft3s_to_mm_day; area_gages2 index 11",
        "missing_policy": "nonfinite and negative discharge are invalid; zero is retained",
        "dpl_normalization": "training.dpl.run_dpl_model.robust_normalize: selected-basin median/IQR, finite-value median fill, clip [-5,5]",
        "dpl_mapping": "training.dpl.run_dpl_model.physical_parameters: sigmoid output; TGD2 residence times inverse-log mapped",
        "ic_model_path": "ablation.ic_core.model_adapter.ModelAdapter with variant=lite",
        "dpl_model_path": "training.dpl.run_dpl_model.LITE_MODEL_REGISTRY and StaticParameterNet",
        "selected_tgd2": selected_tgd2,
        "files": inventory,
        "status": "complete",
        "online_statistics_inputs": [str(output_root / "r1_online_performance.csv"), str(output_root / "r1_online_signature_basin_year.csv")],
    }
    (output_root / "r1_inference_audit.md").write_text("# R1 Inference Audit\n\n" + json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    shutil.rmtree(cache_dir, ignore_errors=True)
    from r1_statistics import signature_tables_from_years
    signature_basin_level, signature_effects = signature_tables_from_years(online_signature_years)
    audit["precomputed"] = {
        "metrics": online_metrics,
        "signature_years": online_signature_years,
        "signature_basin_level": signature_basin_level,
        "signature_effects": signature_effects,
    }
    return audit
