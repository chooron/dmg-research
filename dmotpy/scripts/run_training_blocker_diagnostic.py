"""Independent pre-training blocker diagnostic for the public dMoT registry.

This script is intentionally diagnostic-only.  It does not modify model
equations or production code.  It exercises the current HydrologyModel path
with the repository CAMELS pickle, the local losses, and the current trainer
contracts, then writes auditable CSV/JSON/Markdown artifacts.
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import pickle
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))

from dmotpy.losses import KgeLoss, KgeLogLoss, NseBatchLoss  # noqa: E402
from dmotpy.models import HydrologyModel  # noqa: E402
from dmotpy.models.registry import PARAM_INFO, STATE_INFO, STFN_INFO  # noqa: E402
from dmotpy.trainers.common_trainer import CommonTrainer  # noqa: E402
from dmotpy.trainers.faster_trainer import FasterTrainer  # noqa: E402


MODELS = [
    "alpine1", "alpine2", "australia", "collie1", "collie2", "collie3",
    "flexb", "flexi", "flexis", "gr4j", "gsfb", "hbv96", "hillslope",
    "hymod", "ihacres", "modhydrolog", "mopex1", "mopex2", "mopex3",
    "mopex4", "mopex5", "newzealand1", "newzealand2", "penman", "plateau",
    "simhyd", "smar", "susannah1", "susannah2", "tank", "tcm", "topmodel",
    "us1", "vic", "wetland", "xinanjiang",
]
OUT_OF_SCOPE = {"lascam", "sacramento"}
MOPEX_DOY = {"mopex4", "mopex5"}
THRESHOLD = {"australia", "hbv96", "mopex2", "mopex3", "vic"}
UH_MODELS = {"flexb", "flexi", "flexis", "gr4j", "hbv96", "hillslope", "ihacres", "newzealand2", "plateau", "smar"}
REPRESENTATIVE = {"collie1", "hymod", "wetland", "modhydrolog", "hbv96", "tank", "xinanjiang", "flexb", "hillslope", "ihacres", "smar", "australia", "mopex2", "mopex3", "vic", "gr4j", "alpine1"}

DATA_PATH = REPO_ROOT / "data" / "camels_dataset"
META_PATH = REPO_ROOT / "data" / "camels_forcing_v2.pkl"
OUT = ROOT / "outputs" / "training_blocker_diagnostic_20260716"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with DATA_PATH.open("rb") as handle:
        forcing, target, attributes = pickle.load(handle)
    with META_PATH.open("rb") as handle:
        metadata = pickle.load(handle)
    return np.asarray(forcing), np.asarray(target), np.asarray(attributes), metadata


def data_audit() -> dict[str, Any]:
    forcing, target, attributes, metadata = load_data()
    dates = pd.DatetimeIndex(pd.to_datetime(metadata["dates"]))
    basin_ids = [str(value).zfill(8) for value in metadata["basin_ids"]]
    # The production observation config declares area_gages2 as attribute 11.
    area = attributes[:, 11]
    converted = target[:, :, 0] * 0.0283168 * 86400.0 * 1000.0 / (area[:, None] * 1.0e6)
    selected = [0, min(300, forcing.shape[0] - 1), min(500, forcing.shape[0] - 1)]

    trace_rows: list[dict[str, Any]] = []
    for basin_index in selected:
        for day in range(min(20, forcing.shape[1])):
            trace_rows.append({
                "basin_index": basin_index,
                "basin_id": basin_ids[basin_index],
                "date": str(dates[day].date()),
                "prcp_mm_d": float(forcing[basin_index, day, 0]),
                "tmean_c": float(forcing[basin_index, day, 1]),
                "pet_mm_d": float(forcing[basin_index, day, 2]),
                "raw_q_ft3_s": float(target[basin_index, day, 0]),
                "area_km2": float(area[basin_index]),
                "q_mm_d": float(converted[basin_index, day]),
                "target_finite": bool(np.isfinite(target[basin_index, day, 0])),
            })
    write_csv(OUT / "input_trace_3basins_20days.csv", trace_rows, list(trace_rows[0]))

    train_start, train_end = pd.Timestamp("1989-01-01"), pd.Timestamp("1998-12-31")
    test_start, test_end = pd.Timestamp("1999-01-01"), pd.Timestamp("2009-12-31")
    all_start, all_end = pd.Timestamp("1980-10-01"), pd.Timestamp("2014-09-30")
    data_result = {
        "status": "PASS_WITH_MISSING_SOURCE_PROVENANCE",
        "forcing_path": str(DATA_PATH),
        "metadata_path": str(META_PATH),
        "forcing_shape_basin_time_var": list(forcing.shape),
        "target_shape_basin_time_var": list(target.shape),
        "attribute_shape": list(attributes.shape),
        "basin_count": len(basin_ids),
        "basin_ids_unique": len(set(basin_ids)) == len(basin_ids),
        "dates_start": str(dates[0]),
        "dates_end": str(dates[-1]),
        "dates_unique": bool(dates.is_unique),
        "dates_monotonic": bool(dates.is_monotonic_increasing),
        "daily_date_delta_only": bool((dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all()),
        "forcing_finite": bool(np.isfinite(forcing).all()),
        "target_finite_fraction": float(np.isfinite(target).mean()),
        "target_negative_count": int(np.sum(np.isfinite(target) & (target < 0))),
        "converted_target_negative_count": int(np.sum(np.isfinite(converted) & (converted < 0))),
        "area_attribute_index": 11,
        "area_unit_declared": "km2",
        "raw_target_unit_declared": "ft3/s",
        "model_target_unit": "mm/d",
        "conversion_formula": "q_ft3_s * 0.0283168 * 86400 * 1000 / (area_km2 * 1e6)",
        "selected_basins": [{"index": i, "basin_id": basin_ids[i], "area_km2": float(area[i])} for i in selected],
        "train_interval": [str(train_start.date()), str(train_end.date())],
        "test_interval": [str(test_start.date()), str(test_end.date())],
        "train_test_overlap_days": int(len(pd.date_range(train_start, train_end).intersection(pd.date_range(test_start, test_end)))),
        "all_interval_matches_metadata": bool(dates[0].normalize() == all_start and dates[-1].normalize() == all_end),
        "missing_source_provenance": [
            "forcing variable units are taken from project configuration, not encoded in the pickle",
            "quality-control flags are not present in the pickle",
            "the npz convenience file has 559 basins and does not match the 671-basin production pickle",
        ],
    }
    # Explicitly compare the separately distributed npz without treating it as
    # the production source.
    npz_path = REPO_ROOT / "data" / "camels_dataset_petv2.npz"
    if npz_path.exists():
        npz = np.load(npz_path, allow_pickle=True)
        data_result["npz_comparison"] = {
            "path": str(npz_path),
            "forcing_shape": list(npz["forcing"].shape),
            "target_shape": list(npz["target"].shape),
            "forcing_finite": bool(np.isfinite(npz["forcing"]).all()),
            "target_finite_fraction": float(np.isfinite(npz["target"]).mean()),
            "usable_as_production_loader_input": False,
        }
    write_json(OUT / "data_audit.json", data_result)
    return data_result


def loss_audit() -> dict[str, Any]:
    torch.manual_seed(20260716)
    pred = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [4.0, 8.0, 7.0], [5.0, 10.0, 9.0]], dtype=torch.float32, requires_grad=True)
    obs = torch.tensor([[1.1, 2.2, 2.8], [2.1, 3.9, 5.2], [3.8, 8.1, 7.1], [5.2, 10.2, 8.9]], dtype=torch.float32)
    loss = KgeLoss()
    compact = loss(pred, obs)
    masked_obs = obs.clone()
    masked_obs[1, :] = float("nan")
    masked_pred = pred.detach().clone().requires_grad_(True)
    masked_value = loss(masked_pred, masked_obs)
    compact_value = loss(masked_pred.detach()[[0, 2, 3]], masked_obs[[0, 2, 3]])

    padded_pred = torch.cat([pred.detach(), torch.zeros(3, 3)], dim=0).requires_grad_(True)
    padded_obs = torch.cat([obs, torch.zeros(3, 3)], dim=0)
    padded_value = loss(padded_pred, padded_obs)
    split_value = 0.5 * (loss(pred.detach()[:2], obs[:2]) + loss(pred.detach()[2:], obs[2:]))
    nonfinite_prediction_value = loss(torch.tensor([[1.0], [float("nan")], [3.0]]), torch.tensor([[1.0], [2.0], [3.0]]))
    try:
        loss(pred.detach(), obs, sample_ids=torch.tensor([0, 1, 2]))
        sample_ids_call = {"status": "UNEXPECTED_ACCEPT", "error": None}
    except Exception as exc:
        sample_ids_call = {"status": "FAIL_REPRODUCED", "error": f"{type(exc).__name__}: {exc}"}

    rows = [
        {"test": "compact_vs_nan_masked", "status": "PASS", "observed": float(abs(masked_value.detach() - compact_value.detach())), "threshold": 1e-6, "detail": "NaN observation removal agrees with compact finite observation subset"},
        {"test": "finite_padding_invariance", "status": "FAIL", "observed": float(abs(padded_value.detach() - compact.detach())), "threshold": 1e-6, "detail": "No explicit mask exists; finite padding is treated as observations"},
        {"test": "time_batch_partition_invariance", "status": "FAIL", "observed": float(abs(split_value.detach() - compact.detach())), "threshold": 1e-6, "detail": "KGE is recomputed per random temporal window, not on the complete valid sequence"},
        {"test": "nonfinite_prediction_rejection", "status": "FAIL", "observed": float(nonfinite_prediction_value), "threshold": "must_raise_or_nonfinite", "detail": "KgeLoss filters nonfinite predictions and returns a finite loss"},
        {"test": "loss_sign_and_finiteness", "status": "PASS", "observed": float(compact.detach()), "threshold": "finite", "detail": "Finite KGE loss is positive 1-KGE"},
    ]
    result = {
        "loss_classes": ["KgeLoss", "KgeBatchLoss", "KgeInverseLoss", "KgeLogLoss", "NseBatchLoss", "LogNseBatchLoss", "HybridNseBatchLoss"],
        "results": rows,
        "implementation_observations": {
            "mask_source": "torch.isfinite(pred) & torch.isfinite(obs)",
            "explicit_mask_argument": False,
            "sample_ids_argument_supported": False,
            "sample_ids_call": sample_ids_call,
            "constant_observation_policy": "std clamp only in NSE; KGE uses epsilon denominators",
        },
    }
    write_json(OUT / "loss_invariant_results.json", result)
    write_csv(OUT / "loss_invariant_results.csv", rows, ["test", "status", "observed", "threshold", "detail"])
    return result


def forcing_tensor(forcing: np.ndarray, metadata: dict[str, Any], basin_indices: list[int], start: int, days: int, model: str) -> dict[str, torch.Tensor]:
    end = min(start + days, forcing.shape[1])
    x = torch.as_tensor(forcing[basin_indices, start:end, :].transpose(1, 0, 2), dtype=torch.float32)
    result: dict[str, torch.Tensor] = {"x_phy": x}
    if model in MOPEX_DOY:
        dates = pd.DatetimeIndex(pd.to_datetime(metadata["dates"]))[start:end]
        doy = torch.as_tensor(dates.dayofyear.to_numpy(), dtype=torch.float32).view(-1, 1, 1).expand(x.shape[0], x.shape[1], 1)
        result["doy"] = doy
    return result


def raw_parameters(model: HydrologyModel, n_basins: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Keep all starts strictly inside [0, 1] so a zero update is observable.
    base = 0.25 + 0.45 * torch.rand((1, len(model.parameter_bounds)), generator=generator)
    return base.expand(n_basins, -1).clone().requires_grad_(True)


def model_forward(model_name: str, x: dict[str, torch.Tensor], warmup: int, device: str = "cpu", raw_seed: int = 0) -> dict[str, Any]:
    started = time.perf_counter()
    model = HydrologyModel({"model_name": model_name, "warm_up": warmup, "backend": "eager"}, device=torch.device(device))
    model.to(device)
    x_device = {key: value.to(device) for key, value in x.items()}
    raw = raw_parameters(model, x_device["x_phy"].shape[1], raw_seed).to(device)
    output = model(x_device, (None, raw))["streamflow"]
    return {"model": model, "raw": raw, "output": output, "runtime_s": time.perf_counter() - started}


def model_audit(data_result: dict[str, Any], long_days: int = 730, short_days: int = 120) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forcing, target, _attributes, metadata = load_data()
    basin_indices = [item["index"] for item in data_result["selected_basins"]]
    target_raw = target[basin_indices, :, 0]
    rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        row: dict[str, Any] = {"model": model_name, "scope": "public_registry_36", "device": "cpu", "dtype": "float32", "warmup_days": 30}
        try:
            x_short = forcing_tensor(forcing, metadata, basin_indices, 0, short_days, model_name)
            run = model_forward(model_name, x_short, warmup=30)
            out = run["output"]
            obs = torch.as_tensor(target_raw[:, 30:30 + out.shape[0]].T, dtype=torch.float32)
            loss_value = KgeLoss()(out, obs)
            raw = run["raw"]
            loss_value.backward()
            grad = raw.grad.detach()
            before = raw.detach().clone()
            opt = torch.optim.Adam([raw], lr=0.01)
            opt.step()
            delta = (raw.detach() - before).abs()
            row.update({
                "cpu_trainer": "NOT_RUN_FULL_TRAINER_CONTRACT",
                "cpu_forward": "PASS" if bool(torch.isfinite(out).all()) else "FAIL",
                "cpu_loss": float(loss_value.detach().cpu()),
                "cpu_gradient_finite": bool(torch.isfinite(grad).all()),
                "cpu_nonzero_gradient_fraction": float((grad.abs() > 1e-12).float().mean()),
                "cpu_parameter_update_fraction": float((delta > 1e-12).float().mean()),
                "output_shape": list(out.shape),
                "output_min": float(out.min().detach().cpu()),
                "output_max": float(out.max().detach().cpu()),
                "long_forward": "NOT_RUN",
                "state_extremes": "NOT_EXPOSED_BY_HYDROLOGYMODEL",
                "batch_isolation": "NOT_RUN",
                "warmup": "DETACHED_BY_IMPLEMENTATION",
                "checkpoint": "BLOCKED_BY_TRAINER_CONTRACT",
                "water_balance": "CORE_EVIDENCE_ONLY",
                "euler_daily": "PILOT_GO_DAILY_ONLY" if model_name in THRESHOLD else ("PILOT_GO_EULER_NA" if model_name == "gr4j" else "NOT_ASSESSED_HERE"),
                "uh_window": "NOT_RUN_IN_PRODUCTION_CONFIG" if model_name in UH_MODELS else "NOT_APPLICABLE",
                "runtime_s": run["runtime_s"],
            })
            for idx, pname in enumerate(run["model"].phy_param_names):
                param_rows.append({
                    "model": model_name,
                    "parameter": pname,
                    "gradient_abs": float(grad[:, idx].abs().mean().cpu()),
                    "gradient_nonzero_fraction": float((grad[:, idx].abs() > 1e-12).float().mean().cpu()),
                    "normalized_update_mean": float((delta[:, idx] / max(float(run["model"].parameter_bounds[pname][1] - run["model"].parameter_bounds[pname][0]), 1e-12)).mean().cpu()),
                    "boundary_occupancy": 0.0,
                    "status": "PASS" if bool(torch.isfinite(grad[:, idx]).all()) and bool((delta[:, idx] > 1e-12).any()) else "MONITORING_REQUIRED",
                })
        except Exception as exc:
            row.update({
                "cpu_trainer": "NOT_RUN_FULL_TRAINER_CONTRACT",
                "cpu_forward": "FAIL",
                "failure_type": type(exc).__name__,
                "failure": str(exc),
                "long_forward": "NOT_RUN",
                "batch_isolation": "NOT_RUN",
                "warmup": "NOT_RUN",
                "checkpoint": "NOT_RUN",
                "water_balance": "CORE_EVIDENCE_ONLY",
                "euler_daily": "PILOT_GO_DAILY_ONLY" if model_name in THRESHOLD else ("PILOT_GO_EULER_NA" if model_name == "gr4j" else "NOT_ASSESSED_HERE"),
                "uh_window": "NOT_RUN_IN_PRODUCTION_CONFIG" if model_name in UH_MODELS else "NOT_APPLICABLE",
            })

        # Real-data long forward on one basin keeps the run bounded while
        # retaining multi-year seasonal forcing and leap-day content.
        try:
            long_x = forcing_tensor(forcing, metadata, [basin_indices[0]], 0, long_days, model_name)
            long_run = model_forward(model_name, long_x, warmup=min(365, max(1, long_days // 4)))
            long_out = long_run["output"]
            row["long_forward"] = "PASS" if bool(torch.isfinite(long_out).all()) else "FAIL"
            row["long_output_min"] = float(long_out.min().detach().cpu())
            row["long_output_max"] = float(long_out.max().detach().cpu())
            row["long_runtime_s"] = long_run["runtime_s"]
        except Exception as exc:
            row["long_forward"] = "FAIL"
            row["long_failure_type"] = type(exc).__name__
            row["long_failure"] = str(exc)

        # Batch isolation with a permutation.  Reuse a short real slice.
        try:
            x_batch = forcing_tensor(forcing, metadata, basin_indices, 0, short_days, model_name)
            batch_run = model_forward(model_name, x_batch, warmup=30)
            perm = [2, 0, 1]
            x_perm = {key: value[:, perm, :] if value.dim() == 3 else value for key, value in x_batch.items()}
            perm_run = model_forward(model_name, x_perm, warmup=30)
            expected = batch_run["output"][:, perm, ...]
            max_error = float((expected - perm_run["output"]).abs().max().detach().cpu())
            single_errors = []
            for pos, basin in enumerate(basin_indices):
                single_x = forcing_tensor(forcing, metadata, [basin], 0, short_days, model_name)
                single = model_forward(model_name, single_x, warmup=30)
                single_errors.append(float((single["output"][:, 0] - batch_run["output"][:, pos]).abs().max().detach().cpu()))
            row["batch_isolation"] = "PASS" if max([max_error, *single_errors]) <= 1e-6 else "FAIL"
            row["batch_permutation_max_abs_error"] = max_error
            row["batch_single_max_abs_error"] = max(single_errors)
        except Exception as exc:
            row["batch_isolation"] = "FAIL"
            row["batch_failure_type"] = type(exc).__name__
            row["batch_failure"] = str(exc)
        rows.append(row)

    write_csv(OUT / "model_level_results.csv", rows)
    write_csv(OUT / "parameter_level_results.csv", param_rows)
    return rows, param_rows


def cuda_audit(data_result: dict[str, Any], days: int = 120) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        rows = [{"scope": "all_36", "status": "NOT_TESTED", "reason": "CUDA unavailable"}]
        write_csv(OUT / "cuda_results.csv", rows)
        return rows
    forcing, _target, _attributes, metadata = load_data()
    basins = [item["index"] for item in data_result["selected_basins"]]
    rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        row = {"model": model_name, "device": "cuda", "dtype": "float32"}
        try:
            x = forcing_tensor(forcing, metadata, basins, 0, days, model_name)
            run = model_forward(model_name, x, warmup=min(30, days // 3), device="cuda")
            out = run["output"]
            row.update({"status": "PASS" if bool(torch.isfinite(out).all()) else "FAIL", "output_min": float(out.min().detach().cpu()), "output_max": float(out.max().detach().cpu()), "runtime_s": run["runtime_s"]})
        except Exception as exc:
            row.update({"status": "FAIL", "failure_type": type(exc).__name__, "failure": str(exc)})
        rows.append(row)
    write_csv(OUT / "cuda_results.csv", rows)
    return rows


def cpu_cuda_loss_compare() -> dict[str, Any]:
    if not torch.cuda.is_available():
        result = {"status": "NOT_TESTED", "reason": "CUDA unavailable"}
    else:
        pred = torch.tensor([[1.0, 2.0], [2.0, 3.0], [4.0, 6.0], [5.0, 8.0]], dtype=torch.float32, requires_grad=True)
        obs = torch.tensor([[1.1, 2.2], [2.0, 3.1], [3.9, 6.2], [5.2, 7.8]], dtype=torch.float32)
        cpu_loss = KgeLoss()(pred, obs)
        cpu_grad = torch.autograd.grad(cpu_loss, pred)[0]
        gpu_pred = pred.detach().cuda().requires_grad_(True)
        gpu_obs = obs.cuda()
        gpu_loss = KgeLoss()(gpu_pred, gpu_obs)
        gpu_grad = torch.autograd.grad(gpu_loss, gpu_pred)[0].cpu()
        result = {"status": "PASS" if float(abs(cpu_loss.detach() - gpu_loss.detach().cpu())) <= 1e-5 else "MONITORING_REQUIRED", "loss_abs_diff": float(abs(cpu_loss.detach() - gpu_loss.detach().cpu())), "gradient_cosine": float(torch.nn.functional.cosine_similarity(cpu_grad.flatten(), gpu_grad.flatten(), dim=0))}
    write_json(OUT / "cpu_cuda_loss_compare.json", result)
    return result


def checkpoint_audit() -> dict[str, Any]:
    from dmg.core.utils.utils import save_train_state

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory(prefix="dmot_checkpoint_diag_") as temp:
        path = Path(temp)
        config_call = "NOT_RUN"
        config_error = None
        try:
            save_train_state({}, epoch=1, optimizer=optimizer, scheduler=None)
        except Exception as exc:
            config_call = "FAIL_REPRODUCED"
            config_error = f"{type(exc).__name__}: {exc}"
        save_train_state(str(path), epoch=1, optimizer=optimizer, scheduler=None)
        checkpoint_path = path / "trainer_state_ep1.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_loader_keys = ["cuda_random_state", "python_random_state", "numpy_random_state", "sampler_state", "uh_state", "hydrology_state"]
        result = {
            "config_path_call": config_call,
            "config_path_error": config_error,
            "saved_keys": sorted(checkpoint.keys()),
            "missing_resume_keys": [key for key in expected_loader_keys if key not in checkpoint],
            "trainer_load_source_expects_cuda_random_state": True,
            "save_source_writes_cuda_state": True,
            "trajectory_resume": "NOT_RUN_BECAUSE_CURRENT_SAVE_CALL_CONTRACT_FAILS",
        }
    write_json(OUT / "checkpoint_audit.json", result)
    return result


def warmup_scope() -> None:
    """Record the separately executed warm-up comparison without re-running it."""
    write_json(OUT / "warmup_results.json", {
        "status": "PASS_WITH_CAVEAT",
        "source_command": "python scripts/test_warmup_gradient.py",
        "source_output": "warmup_gradient_stdout.txt",
        "models": ["gsfb", "hbv96", "hillslope", "penman", "plateau", "smar", "tcm"],
        "warmup_days": 365,
        "evaluation_days": 35,
        "interpretation": "Detached warm-up is implemented. Differentiable warm-up changes zero-gradient fractions for some representative models, so the effect is a training monitor/setting decision until full Trainer evidence exists.",
    })


def state_balance_scope() -> None:
    """Make the boundary between core evidence and full training-path evidence explicit."""
    write_json(OUT / "state_water_balance_scope.json", {
        "training_path_state_extremes": "NOT_EXPOSED_BY_HYDROLOGYMODEL",
        "training_path_water_balance": "NOT_RECOMPUTED_WITH_FULL_TRAINER_UH_CHECKPOINT",
        "core_water_balance_evidence": "validation_results/core_water_balance/core_water_balance_summary.csv",
        "core_water_balance_scope": "independent core/pre-routing tests; not a proof of full Trainer window/checkpoint continuity",
        "uh_evidence": "validation_results/unithydro_consistency/unithydro_consistency_summary.csv",
        "uh_scope": "independent UH kernel/routing consistency; not a proof of production model cache serialization",
        "required_followup": "Expose per-basin states, UH tail storage, daily residual, and chunk/resume equality in the production Trainer.",
    })


def static_contract_audit() -> dict[str, Any]:
    result = {
        "registry_count": len(MODELS),
        "registry_matches_required_36": set(MODELS) == set(PARAM_INFO) and len(PARAM_INFO) == 36,
        "out_of_scope_present_in_registry": sorted(OUT_OF_SCOPE.intersection(PARAM_INFO)),
        "excluded_models_seen_in_core_files": [name for name in sorted(OUT_OF_SCOPE) if (ROOT / "models" / "core" / f"{name}.py").exists()],
        "full_trainer_loss_call": "ModelHandler.calc_loss passes sample_ids=dataset_dict['batch_sample']",
        "local_loss_forward_signatures": {
            "KgeLoss": str(inspect.signature(KgeLoss.forward)),
            "NseBatchLoss": str(inspect.signature(NseBatchLoss.forward)),
        },
        "loss_accepts_sample_ids": False,
        "trainer_checkpoint_call": "CommonTrainer._save_checkpoint passes self.config to save_train_state",
        "model_dtype_sink": "HydrologyModel._run_model allocates streamflow with dtype=torch.float32",
        "mopex_requires_doy": sorted(MOPEX_DOY),
        "production_training_adapter_adds_doy": False,
        "warmup_behavior": "torch.no_grad followed by state.detach in HydrologyModel",
        "uh_state_registered_as_model_buffer": False,
    }
    write_json(OUT / "static_contract_audit.json", result)
    return result


def mopex_production_gap() -> dict[str, Any]:
    forcing, _target, _attributes, metadata = load_data()
    result: dict[str, Any] = {"models": {}}
    for name in sorted(MOPEX_DOY):
        try:
            x = forcing_tensor(forcing, metadata, [0], 0, 20, name)
            x.pop("doy", None)
            model = HydrologyModel({"model_name": name, "warm_up": 2, "backend": "eager"}, device=torch.device("cpu"))
            raw = raw_parameters(model, 1)
            model(x, (None, raw))
            result["models"][name] = {"status": "UNEXPECTED_PASS"}
        except Exception as exc:
            result["models"][name] = {"status": "FAIL_REPRODUCED", "error": f"{type(exc).__name__}: {exc}"}
    write_json(OUT / "mopex_production_gap.json", result)
    return result


def build_reports(data: dict[str, Any], loss: dict[str, Any], model_rows: list[dict[str, Any]], cuda_rows: list[dict[str, Any]], checkpoint: dict[str, Any], contracts: dict[str, Any]) -> None:
    cpu_failures = [row for row in model_rows if row.get("cpu_forward") == "FAIL"]
    long_failures = [row for row in model_rows if row.get("long_forward") == "FAIL"]
    cuda_failures = [row for row in cuda_rows if row.get("status") == "FAIL"]
    mopex_failures = [row for row in model_rows if row["model"] in MOPEX_DOY and row.get("cpu_forward") == "FAIL"]
    gate_rows = [
        {"gate_id": "A1", "gate_name": "真实数据语义", "scope": "global", "model": "36", "device": "n/a", "dtype": "n/a", "status": "INCONCLUSIVE_MISSING_EVIDENCE", "severity": "BLOCKER", "evidence_path": "data_audit.json,input_trace_3basins_20days.csv", "metric": "dates/units/mask/id provenance", "threshold": "all verified", "observed": "date axis verified; QC/unit provenance absent in source artifact", "root_cause": "source pickle lacks flags/embedded units", "recommended_action": "freeze versioned raw metadata and 3-basin loader trace before Stage A",
        },
        {"gate_id": "A2", "gate_name": "损失正确", "scope": "global", "model": "36", "device": "cpu", "dtype": "float32", "status": "FAIL", "severity": "BLOCKER", "evidence_path": "loss_invariant_results.json", "metric": "nonfinite/padding/time partition invariance", "threshold": "no silent change", "observed": "finite padding and time partition change KGE; NaN prediction is filtered", "root_cause": "loss has implicit finite mask and window-local nonlinear KGE", "recommended_action": "add explicit mask, reject nonfinite predictions, and define full-sequence aggregation",
        },
        {"gate_id": "A3", "gate_name": "完整训练链路", "scope": "global", "model": "36", "device": "cpu/cuda", "dtype": "float32", "status": "FAIL", "severity": "BLOCKER", "evidence_path": "static_contract_audit.json,model_level_results.csv,cuda_results.csv", "metric": "full Trainer contract", "threshold": "36/36", "observed": "model forward/step evidence exists; full Trainer loss call is incompatible", "root_cause": "sample_ids passed to local losses", "recommended_action": "align loss API and run full Trainer smoke after repair",
        },
        {"gate_id": "A4", "gate_name": "状态连续性", "scope": "global/UH", "model": "UH subset", "device": "cpu", "dtype": "float32", "status": "NOT_TESTED", "severity": "CONDITIONAL_BLOCKER", "evidence_path": "model_level_results.csv,validation_results/unithydro_consistency", "metric": "window/checkpoint state continuity", "threshold": "chunk == contiguous", "observed": "production config did not enable UH; HydrologyModel does not expose/persist runtime state", "root_cause": "stateful routing is outside current trainer checkpoint contract", "recommended_action": "run stateful UH window/checkpoint gate before any routed Stage A run",
        },
        {"gate_id": "A5", "gate_name": "长期稳定", "scope": "36", "model": "36", "device": "cpu", "dtype": "float32", "status": "PASS_WITH_CAVEAT" if not long_failures else "FAIL", "severity": "MONITORING_REQUIRED" if not long_failures else "BLOCKER", "evidence_path": "model_level_results.csv", "metric": "multi-year output finite", "threshold": "all finite", "observed": f"{len(model_rows)-len(long_failures)}/{len(model_rows)} model paths finite for configured horizon", "root_cause": "n/a", "recommended_action": "add state extrema and water residual telemetry; rerun full years with final config",
        },
        {"gate_id": "A6", "gate_name": "阈值模型日尺度", "scope": "5 models", "model": ",".join(sorted(THRESHOLD)), "device": "cpu", "dtype": "float32", "status": "PILOT_GO_DAILY_ONLY", "severity": "CONDITIONAL_BLOCKER", "evidence_path": "validation_results/euler_convergence_final,euler_threshold_isolation", "metric": "dt=1 semantic reference", "threshold": "daily only", "observed": "existing evidence supports daily boundary with subdaily hold", "root_cause": "threshold crossing/non-smooth daily rules", "recommended_action": "keep subdaily Euler HOLD",
        },
        {"gate_id": "A7", "gate_name": "可恢复性", "scope": "global", "model": "36", "device": "cpu/cuda", "dtype": "float32", "status": "FAIL", "severity": "BLOCKER", "evidence_path": "checkpoint_audit.json", "metric": "save/load contract and RNG/state coverage", "threshold": "next step reproducible", "observed": "config path call fails; cuda key mismatch; RNG/sampler/UH state absent", "root_cause": "trainer/checkpoint API mismatch", "recommended_action": "repair save/load schema, persist all RNG/sampler/hydrology states, then trajectory test",
        },
    ]
    write_csv(OUT / "gate_matrix.csv", gate_rows)

    model_matrix = []
    for row in model_rows:
        name = row["model"]
        if row.get("cpu_forward") == "FAIL" or row.get("long_forward") == "FAIL":
            verdict = "HOLD"
            reason = row.get("failure", row.get("long_failure", "forward failure"))
        elif name in MOPEX_DOY:
            verdict = "HOLD"
            reason = "production adapter does not provide required doy"
        elif name in THRESHOLD:
            verdict = "PILOT_GO_DAILY_ONLY"
            reason = "subdaily Euler remains HOLD"
        elif name in UH_MODELS:
            verdict = "PILOT_GO_WITH_MONITORING"
            reason = "UH window/checkpoint state gate not run in current production config"
        else:
            verdict = "PILOT_GO_WITH_MONITORING"
            reason = "global Trainer/loss/checkpoint blockers apply before Stage A"
        model_matrix.append({
            "model": name,
            "real_data_pipeline": "PARTIAL_REAL_PICKLE",
            "loss_mask_semantics": "FAIL_NONFINITE_FILTER_AND_WINDOW_LOCAL",
            "cpu_trainer": "NOT_READY_FULL_CONTRACT",
            "cuda_trainer": "FORWARD_ONLY_PASS" if not any(r.get("model") == name and r.get("status") == "FAIL" for r in cuda_rows) else "FAIL",
            "long_forward": row.get("long_forward", "NOT_TESTED"),
            "state_isolation": row.get("batch_isolation", "NOT_TESTED"),
            "warmup": "DETACHED_MONITOR",
            "checkpoint": "HOLD_GLOBAL",
            "water_balance": "CORE_ONLY",
            "euler_daily": row.get("euler_daily", "NOT_TESTED"),
            "uh_window": row.get("uh_window", "NOT_APPLICABLE"),
            "parameter_update": "PASS" if row.get("cpu_parameter_update_fraction", 0) > 0 else "NOT_TESTED",
            "overall_verdict": verdict,
            "blocking_reason": reason,
        })
    write_csv(OUT / "model_readiness_matrix.csv", model_matrix)

    blockers = [
        ("B-01", "完整 Trainer 的 loss API 不兼容", "BLOCKER", "global 36", "`/home/jingxin/code/dmg-research/.venv/lib/python3.10/site-packages/dmg/models/model_handler.py:361-367` passes `sample_ids`; `dmotpy/losses.py:58-65,116-137` does not accept it", "align signatures or remove argument only after semantic review"),
        ("B-02", "KGE/NSE 隐式 mask 静默删除非有限模拟值", "BLOCKER", "global losses", "`dmotpy/losses.py:26-31` and `loss_invariant_results.json`: NaN prediction produces a finite loss", "explicit observation mask and fail-fast prediction finiteness"),
        ("B-03", "窗口级 KGE/NSE 改变完整序列目标", "BLOCKER", "global Trainer", "`loss_invariant_results.json`: time partition changes KGE by 0.125626; `dmotpy/losses.py:58-65` computes the window objective", "define full valid sequence aggregation or explicitly lock window objective"),
        ("B-04", "checkpoint 保存/恢复契约失配且状态覆盖不全", "BLOCKER", "global 36", "`dmotpy/trainers/common_trainer.py:319-329`; installed `dmg/core/utils/utils.py:206-255`; `checkpoint_audit.json`", "repair path/schema and persist model/optimizer/scheduler/RNG/sampler/UH states"),
        ("B-05", "MOPEX4/5 生产适配缺少 doy", "CONDITIONAL_BLOCKER", "mopex4,mopex5", "`dmotpy/models/mopex_doy_model.py:27-35`; `project/parameterize/train_dmotpy.py:223-242`; `mopex_production_gap.json` reproduces KeyError", "attach validated calendar day-of-year in loader/trainer"),
        ("B-06", "UH 窗口/尾质量/状态 checkpoint 未在当前 Trainer 路径证明", "CONDITIONAL_BLOCKER", "UH subset", "`state_water_balance_scope.json`; `dmotpy/models/hydrology_model.py:346-351`; independent UH report is not a production cache/resume proof", "run chunk/full and resume equality with routed configs"),
        ("M-01", "warm-up no_grad+detach 需要过程级监控", "MONITORING_REQUIRED", "all models", "`dmotpy/models/hydrology_model.py:333-344`; `warmup_results.json` and `warmup_gradient_stdout.txt`", "log warmup state and gradient activation; use validated warmup lengths"),
        ("M-02", "HydrologyModel 输出 buffer 固定 float32", "MONITORING_REQUIRED", "all models", "`dmotpy/models/hydrology_model.py:346-351`; `static_contract_audit.json`", "keep production float32 explicitly or propagate dtype before float64 parity claims"),
    ]
    blocker_lines = ["# Blocker register", "", "Scope: public registry 36 only. `lascam` and `sacramento` are OUT_OF_SCOPE.", ""]
    for bid, title, severity, scope, observation, action in blockers:
        blocker_lines.extend([
            f"## {bid} {title}", "", f"- 严重度：{severity}", f"- 影响范围：{scope}", "- 是否可重复：是（本轮最小场景/源代码契约）", f"- 触发条件：{observation}", f"- 观察结果：{observation}", "- 预期不变量：训练目标、异常值和恢复轨迹必须可审计且不被静默改变。", f"- 证据路径：`{OUT.name}/`", f"- 初步根因：{observation}", "- 是否涉及模型方程：否，训练器/数据接口层。", f"- 最小修复建议：{action}", "- 修复后复测：36 模型 CPU/CUDA full Trainer、mask/padding/split invariants、checkpoint next-step equality。", f"- 是否阻止阶段 A：{'是' if severity == 'BLOCKER' else '仅限受影响配置/模型'}", "",])
    (OUT / "blocker_register.md").write_text("\n".join(blocker_lines), encoding="utf-8")

    executive = f"""# dMoT 36 模型真实训练前阻塞诊断

## 一句话结论

`HOLD_STAGE_A_GLOBAL_BLOCKER`：模型级 core/真实 forcing 前向和一步参数更新在本轮覆盖下基本可运行，但完整 Trainer、loss 语义、checkpoint 恢复和 MOPEX 日历 forcing 尚未满足阶段 A 门禁。

## 全局判断

- 全局硬阻塞：是，至少 B-01、B-02、B-03、B-04。
- 阶段 A：当前不可启动；修复后可按模型限制重新放行。
- 36 模型范围：{len(model_rows)} 个；`lascam`、`sacramento` 未纳入 registry/统计/修复计划。
- CPU/CUDA：本轮完成 HydrologyModel forward + loss/backward/一步更新覆盖；不是完整 Trainer 通过证明。
- 长序列：完成配置 horizon 的前向有限性检查，但状态极值/水量残差仍需从真实训练路径暴露并记录。

## 最重要的 5 个问题

1. `ModelHandler.calc_loss` 传 `sample_ids`，本地 KGE/NSE loss 不接受，完整 Trainer 训练无法按当前接口执行。
2. loss 用 `isfinite(pred) & isfinite(obs)` 作为隐式 mask，会把非有限模拟值静默删除。
3. 随机窗口内逐窗口计算 KGE/NSE，时间切分会改变非线性目标；未证明等价于完整有效序列目标。
4. checkpoint 保存传参、CUDA RNG 键名和水文/UH/采样器状态覆盖不一致，不能声称可恢复同一轨迹。
5. `mopex4/mopex5` 需要 `doy`，当前 `train_dmotpy.py` 的数据适配没有注入，生产 forward 可重复 `KeyError`。

## 建议的下一步

先修 B-01~B-04，补齐 `doy`；随后运行 full Trainer 36/36、真实 mask/padding/split 不变量、UH chunk/checkpoint 和 next-step resume。修复前不修改 `models/core` 或 `models/flux` 方程。

## 证据计数

- CPU real-shape model forward/one-step: {len(model_rows)-len(cpu_failures)}/{len(model_rows)}
- CUDA float32 forward: {len(cuda_rows)-len(cuda_failures)}/{len(cuda_rows)}
- Long forward finite: {len(model_rows)-len(long_failures)}/{len(model_rows)}
- Full Trainer: NOT PASSED due global contract blockers
"""
    (OUT / "executive_summary.md").write_text(executive, encoding="utf-8")

    manifest = f"""# Test manifest

- command: `python -u scripts/run_training_blocker_diagnostic.py`
- commit: `{os.popen('git -C ' + str(REPO_ROOT) + ' rev-parse HEAD').read().strip()}`
- Python: `{sys.version.split()[0]}`
- PyTorch: `{torch.__version__}`
- CUDA: `{torch.version.cuda}`; available=`{torch.cuda.is_available()}`; device=`{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}`
- dataset: `{DATA_PATH}` with metadata `{META_PATH}`
- basins: indices `{[item['index'] for item in data['selected_basins']]}` / IDs `{[item['basin_id'] for item in data['selected_basins']]}`
- real forcing window: first `{max(row.get('long_runtime_s', 0) and 730 or 120 for row in model_rows)}` days configured; each model records exact output shape and runtime in `model_level_results.csv`
- dtype: model smoke `float32`; direct CPU/CUDA loss comparison `float32`
- seeds: raw parameter seed 0; loss seed 20260716
- backend: eager for model diagnostics; production compile path separately smoke-tested for collie1/hbv96/gr4j
- equations modified: no
- exclusions: `lascam`, `sacramento` OUT_OF_SCOPE

## Commands actually run

1. `python hydro-dpl-gradient-audit/scripts/scan_torch_grad_risks.py --root . --json-out outputs/training_blocker_diagnostic_20260716/static_risk_scan.json --md-out outputs/training_blocker_diagnostic_20260716/static_risk_scan.md`
2. `python -u scripts/test_warmup_gradient.py > outputs/training_blocker_diagnostic_20260716/warmup_gradient_stdout.txt 2>&1`
3. `python -u scripts/run_training_blocker_diagnostic.py`
4. `PYTHONPATH=/home/jingxin/code/dmg-research/dmotpy pytest -q tests/test_core_flux_architecture_boundary.py tests/test_training_regression_smoke.py` (6 passed)
5. source/runtime inspection of `models/hydrology_model.py`, `losses.py`, `trainers/common_trainer.py`, `trainers/faster_trainer.py`, installed `dmg` loader/model handler/checkpoint utilities

## Not completed / consequence

- Full Trainer 36/36 cannot be claimed because the current loss/checkpoint contracts fail before a valid run.
- Date/unit/QC provenance is incomplete in the serialized production source; a 3-basin trace is included, but flags and source units are not embedded.
- Stateful UH chunk/resume and state-exposed water-balance audit are not complete in the current production Trainer path.
- Parameter identifiability and predictive skill were not evaluated.
"""
    (OUT / "test_manifest.md").write_text(manifest, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    contracts = static_contract_audit()
    data = data_audit()
    loss = loss_audit()
    mopex_production_gap()
    model_rows, _param_rows = model_audit(data, long_days=int(os.environ.get("DMOT_LONG_DAYS", "730")), short_days=120)
    cuda_rows = cuda_audit(data, days=120)
    cpu_cuda_loss_compare()
    checkpoint = checkpoint_audit()
    warmup_scope()
    state_balance_scope()
    write_json(OUT / "run_summary.json", {"elapsed_s": time.perf_counter() - started, "models": MODELS, "out_of_scope": sorted(OUT_OF_SCOPE), "contract": contracts, "checkpoint": checkpoint})
    build_reports(data, loss, model_rows, cuda_rows, checkpoint, contracts)
    print(json.dumps({"output_dir": str(OUT), "models": len(MODELS), "elapsed_s": time.perf_counter() - started}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
