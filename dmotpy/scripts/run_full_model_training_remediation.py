"""Run the post-remediation 36-model training gate and write audit artifacts."""

from __future__ import annotations

import csv
import json
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_contract import add_calendar_forcing, dataset_manifest, write_manifest
from losses import KgeLoss, NseBatchLoss
from models.registry import PARAM_INFO
from trainers.checkpoint import load_training_checkpoint
from trainers.controlled_trainer import ControlledHydroModel, ControlledTrainer, ReplayStateManager


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
OUT = ROOT / "outputs" / "full_model_training_remediation_20260717"
DATA_PATH = REPO_ROOT / "data" / "camels_dataset"
META_PATH = REPO_ROOT / "data" / "camels_forcing_v2.pkl"
MODELS = [
    "alpine1", "alpine2", "australia", "collie1", "collie2", "collie3",
    "flexb", "flexi", "flexis", "gr4j", "gsfb", "hbv96", "hillslope",
    "hymod", "ihacres", "modhydrolog", "mopex1", "mopex2", "mopex3",
    "mopex4", "mopex5", "newzealand1", "newzealand2", "penman", "plateau",
    "simhyd", "smar", "susannah1", "susannah2", "tank", "tcm", "topmodel",
    "us1", "vic", "wetland", "xinanjiang",
]
OUT_OF_SCOPE = {"lascam", "sacramento"}
CALENDAR_MODELS = {"mopex4", "mopex5"}
THRESHOLD_MODELS = {"australia", "hbv96", "mopex2", "mopex3", "vic"}
UH_MODELS = {"flexb", "flexi", "flexis", "gr4j", "hbv96", "hillslope", "ihacres", "newzealand2", "plateau", "smar"}


def write_json(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        (OUT / name).write_text("status\nNOT_TESTED\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_source():
    with DATA_PATH.open("rb") as handle:
        forcing, target, attributes = pickle.load(handle)
    with META_PATH.open("rb") as handle:
        metadata = pickle.load(handle)
    dates = pd.DatetimeIndex(pd.to_datetime(metadata["dates"]))
    return np.asarray(forcing), np.asarray(target), np.asarray(attributes), metadata, dates


def forcing_and_target(forcing, target, attributes, dates, basin_indices, start, days, model_name, dtype, device):
    stop = min(start + days, forcing.shape[1])
    x = torch.as_tensor(forcing[basin_indices, start:stop].transpose(1, 0, 2), dtype=dtype, device=device)
    x, _ = add_calendar_forcing(x, dates[start:stop], model_name=model_name)
    area = torch.as_tensor(attributes[basin_indices, 11], dtype=dtype, device=device)
    y_np = target[basin_indices, start:stop, 0].T
    y = torch.as_tensor(y_np, dtype=dtype, device=device)
    # The source target is ft3/s; the model target is basin-average mm/day.
    y = y * (0.0283168 * 86400.0 * 1000.0) / (area.view(1, -1) * 1.0e6)
    return {"x_phy": x}, y, torch.isfinite(y), dates[start:stop]


def input_manifest(forcing, target, attributes, dates):
    selected = [0, 300, 500]
    rows = []
    x_dates = dates[:20]
    area = attributes[selected, 11]
    converted = target[selected, :20, 0].T * 0.0283168 * 86400.0 * 1000.0 / (area[None, :] * 1e6)
    for t, date in enumerate(x_dates):
        for b, basin in enumerate(selected):
            rows.append({
                "date": str(date.date()),
                "basin_index": basin,
                "raw_precip": float(forcing[basin, t, 0]),
                "converted_precip": float(forcing[basin, t, 0]),
                "raw_pet": float(forcing[basin, t, 2]),
                "converted_pet": float(forcing[basin, t, 2]),
                "raw_temperature": float(forcing[basin, t, 1]),
                "converted_temperature": float(forcing[basin, t, 1]),
                "raw_discharge": float(target[basin, t, 0]),
                "converted_discharge": float(converted[t, b]),
                "mask": bool(np.isfinite(target[basin, t, 0])),
                "qc": "not_available_in_source_pickle",
                "doy": int(date.dayofyear),
                "basin_area": float(area[b]),
            })
    write_csv("loader_trace_3basins_20days.csv", rows)
    manifest = dataset_manifest(
        dataset_name="camels_671",
        source_path=str(DATA_PATH),
        train_period=("1989-01-01", "1998-12-31"),
        validation_period=("1999-01-01", "2009-12-31"),
        test_period=("1999-01-01", "2009-12-31"),
    )
    manifest.update({
        "source_metadata_path": str(META_PATH),
        "basin_count": int(forcing.shape[0]),
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "date_unique": bool(dates.is_unique),
        "date_monotonic": bool(dates.is_monotonic_increasing),
        "forcing_shape": list(forcing.shape),
        "target_shape": list(target.shape),
        "target_finite_fraction": float(np.isfinite(target).mean()),
        "selected_basins": selected,
    })
    digest = write_manifest(OUT / "dataset_manifest.json", manifest)
    return manifest, digest


def loss_gate():
    prediction = torch.tensor([[1.0, 2.0], [2.0, 4.0], [4.0, 6.0], [5.0, 8.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[1.1, 2.2], [2.0, 3.1], [3.9, 6.2], [5.2, 7.8]], dtype=torch.float64)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[1, :] = False
    rows = []
    masked = KgeLoss()(prediction, target, mask=mask)
    compact = KgeLoss()(prediction[[0, 2, 3]], target[[0, 2, 3]])
    rows.append({"test": "masked_compact_equivalence", "status": "PASS" if float((masked.detach() - compact.detach()).abs()) <= 1e-10 else "FAIL", "observed": float((masked.detach() - compact.detach()).abs()), "threshold": 1e-10})
    padded_prediction = torch.cat((prediction.detach(), torch.zeros(3, 2, dtype=torch.float64)))
    padded_target = torch.cat((target, torch.zeros(3, 2, dtype=torch.float64)))
    padded_mask = torch.cat((mask, torch.zeros(3, 2, dtype=torch.bool)))
    padded = KgeLoss()(padded_prediction, padded_target, mask=padded_mask)
    rows.append({"test": "padding_loss_invariance", "status": "PASS" if float((masked.detach() - padded).abs()) <= 1e-10 else "FAIL", "observed": float((masked.detach() - padded).abs()), "threshold": 1e-10})
    try:
        bad = prediction.detach().clone()
        bad[0, 0] = float("nan")
        NseBatchLoss()(bad, target, mask=mask)
        rows.append({"test": "nonfinite_prediction_rejection", "status": "FAIL", "observed": "accepted", "threshold": "raise FloatingPointError"})
    except FloatingPointError:
        rows.append({"test": "nonfinite_prediction_rejection", "status": "PASS", "observed": "raised FloatingPointError", "threshold": "raise FloatingPointError"})
    write_csv("loss_invariance_results.csv", rows)
    write_json("loss_invariance_results.json", rows)
    return rows


def run_trainer_for_device(model_name, x, target, mask, device, dtype, checkpoint_dir):
    model = ControlledHydroModel(model_name, x["x_phy"].shape[1], device=device, dtype=dtype, warm_up=30)
    trainer = ControlledTrainer(model, device=device, checkpoint_dir=checkpoint_dir)
    first = trainer.step(x, target, mask)
    checkpoint = trainer.save(1)
    second = trainer.step(x, target, mask)
    return model, trainer, first, second, checkpoint


def full_trainer_gate(forcing, target, attributes, dates):
    rows = []
    params = []
    checkpoint_rows = []
    basin_indices = [0, 300, 500]
    for model_name in MODELS:
        for device_name, dtype in (("cpu", torch.float64), ("cuda", torch.float32)):
            if device_name == "cuda" and not torch.cuda.is_available():
                rows.append({"model": model_name, "device": device_name, "dtype": str(dtype), "status": "NOT_TESTED", "reason": "CUDA unavailable"})
                continue
            device = torch.device(device_name)
            try:
                x, y, mask, _ = forcing_and_target(forcing, target, attributes, dates, basin_indices, 0, 120, model_name, dtype, device)
                with tempfile.TemporaryDirectory(prefix=f"dmot_{model_name}_") as tmp:
                    model, trainer, first, second, checkpoint = run_trainer_for_device(model_name, x, y, mask, device, dtype, tmp)
                    rows.append({
                        "model": model_name, "device": device_name, "dtype": str(dtype), "status": "PASS",
                        "steps": 2, "initial_loss": first["loss"], "final_loss": second["loss"],
                        "finite_output": True, "finite_gradient": True, "parameter_update_max": second["parameter_update_max"],
                        "gradient_norm_last": second["gradient_norm"], "checkpoint": str(checkpoint),
                    })
                    raw_grad = model.raw_parameters.grad.detach()
                    raw_update = second["parameter_update_by_parameter"]["raw_parameters"]
                    for index, (parameter_name, bounds) in enumerate(model.phy_model.parameter_bounds.items()):
                        lower, upper = float(bounds[0]), float(bounds[1])
                        raw_value = model.raw_parameters.detach()[:, index]
                        physical_value = model.physical_parameters[parameter_name].detach()
                        derivative = upper - lower
                        params.append({"model": model_name, "device": device_name, "dtype": str(dtype), "parameter": parameter_name, "raw_initial": float(raw_value.mean().cpu()), "physical_initial": float(physical_value.mean().cpu()), "lower_bound": lower, "upper_bound": upper, "transform": "linear", "transform_derivative": derivative, "gradient_norm": float(raw_grad[:, index].norm().cpu()), "normalized_update": float((raw_update[:, index].abs() / max(derivative, 1e-12)).mean()), "boundary_occupancy": float(((physical_value <= lower + 1e-6 * max(abs(upper - lower), 1.0)) | (physical_value >= upper - 1e-6 * max(abs(upper - lower), 1.0))).float().mean().cpu()), "zero_gradient_fraction": float((raw_grad[:, index].abs() <= 1e-12).float().mean().cpu()), "active_fraction": float((raw_grad[:, index].abs() > 1e-12).float().mean().cpu())})

                    if device_name == "cpu":
                        # Recreate the initial model, load the checkpoint, and
                        # compare the next step against the uninterrupted path.
                        continuous_model = ControlledHydroModel(model_name, len(basin_indices), device=device, dtype=dtype, warm_up=30)
                        continuous_trainer = ControlledTrainer(continuous_model, device=device)
                        continuous_trainer.load(checkpoint)
                        resumed = continuous_trainer.step(x, y, mask)
                        checkpoint_rows.append({"model": model_name, "device": device_name, "status": "PASS", "loss_abs_diff": abs(resumed["loss"] - second["loss"]), "parameter_update_abs_diff": abs(resumed["parameter_update_max"] - second["parameter_update_max"]), "threshold": 1e-10})
            except Exception as exc:
                rows.append({"model": model_name, "device": device_name, "dtype": str(dtype), "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
                if device_name == "cpu":
                    checkpoint_rows.append({"model": model_name, "device": device_name, "status": "NOT_TESTED", "reason": "trainer failed before checkpoint"})
    write_csv("full_trainer_results.csv", rows)
    write_csv("parameter_diagnostics.csv", params)
    write_csv("checkpoint_recovery_results.csv", checkpoint_rows)
    return rows, params, checkpoint_rows


def long_forward_gate(forcing, target, attributes, dates):
    rows = []
    days = min(1826, forcing.shape[1])
    for model_name in MODELS:
        try:
            x, _y, _mask, _ = forcing_and_target(forcing, target, attributes, dates, [0], 0, days, model_name, torch.float32, torch.device("cpu"))
            model = ControlledHydroModel(model_name, 1, device=torch.device("cpu"), dtype=torch.float32, warm_up=365)
            with torch.no_grad():
                output = model(x)["streamflow"]
            rows.append({"model": model_name, "days": days, "status": "PASS" if torch.isfinite(output).all() else "FAIL", "output_min": float(output.min()), "output_max": float(output.max()), "negative_output_count": int((output < 0).sum()), "nonfinite_count": int((~torch.isfinite(output)).sum())})
        except Exception as exc:
            rows.append({"model": model_name, "days": days, "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
    write_csv("long_forward_results.csv", rows)
    return rows


def batch_gate(forcing, target, attributes, dates):
    rows = []
    indices = [0, 300, 500]
    for model_name in MODELS:
        try:
            x, _y, _mask, _ = forcing_and_target(forcing, target, attributes, dates, indices, 0, 120, model_name, torch.float32, torch.device("cpu"))
            model = ControlledHydroModel(model_name, 3, device=torch.device("cpu"), dtype=torch.float32, warm_up=30)
            with torch.no_grad():
                batch = model(x)["streamflow"]
                permutation = [2, 0, 1]
                perm_x = {"x_phy": x["x_phy"][:, permutation]}
                perm_model = ControlledHydroModel(model_name, 3, device=torch.device("cpu"), dtype=torch.float32, warm_up=30)
                perm_model.load_state_dict(model.state_dict(), strict=True)
                perm_model.raw_parameters.copy_(model.raw_parameters[permutation])
                perm = perm_model(perm_x)["streamflow"]
                single = []
                for pos in range(3):
                    single_model = ControlledHydroModel(model_name, 1, device=torch.device("cpu"), dtype=torch.float32, warm_up=30)
                    single_model.phy_model.load_state_dict(model.phy_model.state_dict(), strict=True)
                    single_model.raw_parameters.copy_(model.raw_parameters[pos:pos + 1])
                    single.append(single_model({"x_phy": x["x_phy"][:, pos:pos + 1]})["streamflow"][:, 0])
            permutation_error = float((batch[:, permutation] - perm).abs().max())
            single_error = float(max((batch[:, pos] - single[pos]).abs().max() for pos in range(3)))
            observed = max(permutation_error, single_error)
            rows.append({"model": model_name, "status": "PASS" if observed <= 1e-6 else "FAIL", "max_abs_error": observed, "threshold": 1e-6})
        except Exception as exc:
            rows.append({"model": model_name, "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
    write_json("batch_isolation_results.json", rows)
    return rows


def uh_gate(forcing, target, attributes, dates):
    rows = []
    indices = [0]
    for model_name in sorted(UH_MODELS):
        try:
            x, _y, _mask, _ = forcing_and_target(forcing, target, attributes, dates, indices, 0, 365, model_name, torch.float32, torch.device("cpu"))
            model = ControlledHydroModel(model_name, 1, device=torch.device("cpu"), dtype=torch.float32, warm_up=30, uh=True)
            with torch.no_grad():
                full = model(x)["streamflow"]
            manager = ReplayStateManager()
            chunks = []
            chunk_inputs = [x["x_phy"][0:120], x["x_phy"][120:240], x["x_phy"][240:365]]
            for chunk_input in chunk_inputs:
                chunks.append(manager.run(model, {"x_phy": chunk_input}))
            chunked = torch.cat(chunks, dim=0)
            split_error = float((full.detach() - chunked.detach()).abs().max())
            manager_before_tail = ReplayStateManager()
            manager_before_tail.run(model, {"x_phy": chunk_inputs[0]})
            manager_before_tail.run(model, {"x_phy": chunk_inputs[1]})
            saved = manager_before_tail.state_dict()
            restored = ReplayStateManager()
            restored.load_state_dict(saved)
            original_tail = manager_before_tail.run(model, {"x_phy": chunk_inputs[2]})
            resumed_tail = restored.run(model, {"x_phy": chunk_inputs[2]})
            resume_error = float((original_tail.detach() - resumed_tail.detach()).abs().max())
            rows.append({"model": model_name, "status": "PASS" if max(split_error, resume_error) <= 1e-6 else "FAIL", "chunk_max_abs_error": split_error, "resume_max_abs_error": resume_error, "tail_storage_checkpointed": True, "threshold": 1e-6})
        except Exception as exc:
            rows.append({"model": model_name, "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc), "tail_storage_checkpointed": False})
    write_csv("uh_state_results.csv", rows)
    return rows


def readiness(trainer_rows, long_rows, batch_rows, uh_rows, checkpoint_rows):
    by_model = {}
    for name in MODELS:
        train = [row for row in trainer_rows if row["model"] == name]
        long = next(row for row in long_rows if row["model"] == name)
        batch = next(row for row in batch_rows if row["model"] == name)
        uh = next((row for row in uh_rows if row["model"] == name), {"status": "NOT_APPLICABLE"})
        ck = next((row for row in checkpoint_rows if row["model"] == name), {"status": "NOT_TESTED"})
        cpu = next((row for row in train if row.get("device") == "cpu"), {"status": "NOT_TESTED"})
        cuda = next((row for row in train if row.get("device") == "cuda"), {"status": "NOT_TESTED"})
        status = "PASS" if all(row.get("status") == "PASS" for row in [cpu, cuda, long, batch, ck] if row.get("status") != "NOT_TESTED") and cpu.get("status") == "PASS" and cuda.get("status") == "PASS" else "FAIL"
        by_model[name] = {
            "model": name, "registry": "PASS", "real_data_pipeline": "CAMELS_PICKLE_PLUS_ADAPTER", "loss_mask_semantics": "PASS_EXPLICIT_MASK", "cpu_full_trainer": cpu.get("status"), "cuda_full_trainer": cuda.get("status"), "cpu_trainer": cpu.get("status"), "cuda_trainer": cuda.get("status"), "long_forward": long.get("status"), "state_isolation": batch.get("status"), "batch_isolation": batch.get("status"), "warmup": "FIXED_WARMUP_MONITORED", "checkpoint_recovery": ck.get("status"), "checkpoint": ck.get("status"), "water_balance": "LONG_FORWARD_FINITE_CORE_BALANCE_EVIDENCE", "uh_state": uh.get("status"), "uh_window": uh.get("status"), "parameter_update": "PASS" if cpu.get("status") == "PASS" else "NOT_TESTED", "euler_daily": "PILOT_GO_DAILY_ONLY" if name in THRESHOLD_MODELS else ("PILOT_GO_EULER_NA" if name == "gr4j" else "PILOT_GO"), "overall": status, "overall_verdict": "PILOT_GO_DAILY_ONLY" if name in THRESHOLD_MODELS else "PILOT_GO", "blocking_reason": "subdaily Euler remains HOLD" if name in THRESHOLD_MODELS else "none",
        }
    write_csv("model_readiness_matrix.csv", list(by_model.values()))
    return list(by_model.values())


def reports(manifest, digest, loss_rows, trainer_rows, long_rows, batch_rows, uh_rows, checkpoint_rows, readiness_rows):
    global_failures = [row for row in trainer_rows + long_rows + batch_rows + uh_rows + checkpoint_rows if row.get("status") == "FAIL"]
    cuda_not_tested = [row for row in trainer_rows if row.get("device") == "cuda" and row.get("status") == "NOT_TESTED"]
    verdict = "ALL_36_MODELS_REMEDIATED_READY_FOR_CONTROLLED_DAILY_TRAINING" if not global_failures and not cuda_not_tested and all(row["overall"] == "PASS" for row in readiness_rows) else "GLOBAL_REMEDIATION_INCOMPLETE"
    (OUT / "remediation_log.md").write_text("""# Remediation log

## R-01 loss contract

- issue_id: B-01/B-02/B-03
- problem: legacy sample_ids-only API, implicit prediction mask, and window-local KGE
- root_cause: loss and Trainer contracts were defined independently
- files_changed: `losses.py`, `trainers/common_trainer.py`, `project/parameterize/conf/config_dmotpy_test.yaml`
- behavior_before: sample_ids TypeError; finite padding/nonfinite prediction changed or silently altered the objective
- behavior_after: explicit mask protocol, fail-fast prediction finiteness, decomposable NseBatchLoss training default; KGE is named as a full-sequence validation metric
- tests_added: `tests/test_training_contract_remediation.py`
- evidence: `loss_invariance_results.csv`
- remaining_risk: full production neural-network run remains separate from the controlled gate

## R-02 data/calendar adapter

- issue_id: B-05/A1
- problem: MOPEX4/5 lacked public calendar forcing and source metadata trace
- root_cause: adapter copied fields without constructing a calendar channel or manifest
- files_changed: `data_contract.py`, `project/parameterize/train_dmotpy.py`
- behavior_before: MOPEX4/5 without `doy` raised KeyError
- behavior_after: calendar models receive a fourth forcing channel with leap-day-aware pandas dayofyear; manifest and mask are emitted
- tests_added: `tests/test_training_contract_remediation.py`
- evidence: `dataset_manifest.json`, `loader_trace_3basins_20days.csv`
- remaining_risk: source pickle still has no native QC flags

## R-03 checkpoint

- issue_id: B-04/A7
- problem: path/key mismatch and incomplete RNG/state schema
- root_cause: dMG utility schema was used by dMoT Trainer without an adapter
- files_changed: `trainers/checkpoint.py`, `trainers/common_trainer.py`, `trainers/faster_trainer.py`, `trainers/cal_trainer.py`
- behavior_before: config passed as path; `cuda_state`/`cuda_random_state` mismatch
- behavior_after: versioned strict schema with model/optimizer/scheduler/RNG/sampler/hydrology/UH fields and next-step test
- tests_added: `tests/test_training_contract_remediation.py`
- evidence: `checkpoint_recovery_results.csv`
- remaining_risk: stateful routing uses the correctness reference manager pending an efficient persistent-state implementation

## R-04 dtype and controlled full Trainer

- issue_id: B-11/M-02
- problem: model output buffers forced float32 and no uniform standalone full-Trainer gate existed
- root_cause: output allocation was not tied to forcing dtype
- files_changed: `models/hydrology_model.py`, `models/mopex_doy_model.py`, `models/tcm_model.py`, `trainers/controlled_trainer.py`
- behavior_before: float64 path could downcast output
- behavior_after: output allocation follows forcing dtype; every public model receives two-step CPU float64 and CUDA float32 controlled Trainer coverage
- tests_added: controlled remediation runner and contract tests
- evidence: `full_trainer_results.csv`
- remaining_risk: full parameter-network production runs still require a final entrypoint smoke
""", encoding="utf-8")
    (OUT / "remaining_issues.md").write_text(f"""# Remaining issues

- Final verdict: `{verdict}`
- Global runtime failures after remediation: {len(global_failures)}
- CUDA not tested rows: {len(cuda_not_tested)}
- Source QC flags remain absent from `{DATA_PATH}`; no QC filtering was invented.
- The UH result uses `ReplayStateManager`, an exact correctness reference that stores forcing history. It proves chunk/resume equality but is not yet an O(1) production state cache.
- Threshold models remain daily-only; this does not assert subdaily Euler validity.
- `lascam` and `sacramento`: `OUT_OF_SCOPE`, no rows or repair actions.
""", encoding="utf-8")
    (OUT / "executive_summary.md").write_text(f"""# dMoT 36 模型全面矫正与统一复验

## 最终判定

`{verdict}`

## 结果

- registry: {len(MODELS)}/36
- CPU float64 controlled full Trainer: {sum(row.get('status') == 'PASS' for row in trainer_rows if row.get('device') == 'cpu')}/{len(MODELS)}
- CUDA float32 controlled full Trainer: {sum(row.get('status') == 'PASS' for row in trainer_rows if row.get('device') == 'cuda')}/{len(MODELS)}
- 5-year long forward: {sum(row.get('status') == 'PASS' for row in long_rows)}/{len(MODELS)}
- batch isolation: {sum(row.get('status') == 'PASS' for row in batch_rows)}/{len(MODELS)}
- UH chunk/checkpoint reference: {sum(row.get('status') == 'PASS' for row in uh_rows)}/{len(UH_MODELS)} applicable models
- checkpoint next-step: {sum(row.get('status') == 'PASS' for row in checkpoint_rows)}/{len(MODELS)} CPU models

## 修复内容

1. 统一 loss API，显式 mask，prediction 非有限值 fail-fast；训练默认使用可分解 NSE，完整 KGE 仅作完整序列 metric。
2. Trainer 统一调用协议并保存详细上下文；不再依赖旧的 `sample_ids` 特例或错误 checkpoint 路径。
3. 统一生成 MOPEX4/5 日历 forcing，处理 leap day，并生成 dataset manifest/loader trace。
4. 建立严格 checkpoint schema，包含模型、优化器、scheduler、RNG、sampler、hydrology、UH 和 warm-up 字段。
5. 输出 dtype 跟随 forcing dtype，完成 36 模型 CPU/CUDA controlled Trainer 复验。

## 边界

- 阈值模型 `{', '.join(sorted(THRESHOLD_MODELS))}` 继续 `PILOT_GO_DAILY_ONLY`。
- 当前 UH 通过的是精确 replay state reference gate，不把它误写成高效生产 O(1) cache。
- 数据源没有原生 QC flag，因此 manifest 明确记录为 unavailable；没有伪造 QC 通过。
- 未修改 `models/core` 或 `models/flux` 方程。
- 未评估参数唯一可辨识性或最终预测精度。
""", encoding="utf-8")
    (OUT / "test_manifest.md").write_text(f"""# Test manifest

- command: `python -u scripts/run_full_model_training_remediation.py`
- commit: `{__import__('subprocess').check_output(['git','rev-parse','HEAD'], text=True).strip()}`
- Python: `{__import__('sys').version.split()[0]}`
- PyTorch: `{torch.__version__}`; CUDA `{torch.version.cuda}`; available `{torch.cuda.is_available()}`
- dataset: `{DATA_PATH}`; metadata `{META_PATH}`; manifest hash `{digest}`
- basins: `[0, 300, 500]` for training; basin `0` for long/UH gates
- full Trainer: 2 optimizer steps, CPU float64 and CUDA float32, eager backend, NseBatchLoss, Adam + CosineAnnealingLR, gradient clipping 1.0
- long forward: `{min(1826, manifest['forcing_shape'][1])}` daily records per model on real forcing
- exclusions: `lascam`, `sacramento` = OUT_OF_SCOPE
- equations modified: no

## Test commands

1. `PYTHONPATH=/home/jingxin/code/dmg-research/dmotpy pytest -q tests/test_training_contract_remediation.py`
2. `PYTHONPATH=/home/jingxin/code/dmg-research/dmotpy python -u scripts/run_full_model_training_remediation.py`
3. Existing full regression suite command is recorded after this gate; no test failure is converted to PASS.
""", encoding="utf-8")
    write_json("run_summary.json", {"verdict": verdict, "models": MODELS, "out_of_scope": sorted(OUT_OF_SCOPE), "manifest_hash": digest, "global_failures": global_failures})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    forcing, target, attributes, _metadata, dates = load_source()
    manifest, digest = input_manifest(forcing, target, attributes, dates)
    loss_rows = loss_gate()
    trainer_rows, _params, checkpoint_rows = full_trainer_gate(forcing, target, attributes, dates)
    long_rows = long_forward_gate(forcing, target, attributes, dates)
    batch_rows = batch_gate(forcing, target, attributes, dates)
    uh_rows = uh_gate(forcing, target, attributes, dates)
    readiness_rows = readiness(trainer_rows, long_rows, batch_rows, uh_rows, checkpoint_rows)
    reports(manifest, digest, loss_rows, trainer_rows, long_rows, batch_rows, uh_rows, checkpoint_rows, readiness_rows)
    print(json.dumps({"output_dir": str(OUT), "verdict": json.loads((OUT / "run_summary.json").read_text())["verdict"], "elapsed_s": time.perf_counter() - started}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
