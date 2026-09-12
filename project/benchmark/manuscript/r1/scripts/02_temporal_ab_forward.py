#!/usr/bin/env python3
"""Step 5: canonical IC/dPL forward evaluation on the frozen temporal A/B split.

No optimizer, backward pass, training loop, or checkpoint write is performed.
Daily predictions are held in memory for one model and only scalar KGE values
are persisted in the R1 cache/table.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from r1_config import (
    BENCHMARK, CACHE_DIR, CARAVAN_PATH, DATA_ROOT, DATASET_PATH, DATE_INDEX_PATH, DPL_ROOT,
    EVAL_WARMUP_DAYS, EXPECTED_DATA, GAGE_IDS_PATH, IC_CHECKPOINT_ROOT, IC_STATUS_PATH,
    CANONICAL_DPL_SEED, CANONICAL_DPL_SOURCE_SHA, MODEL_REGISTRY, SCRIPTS_DIR, TABLES_DIR,
    TEMPORAL_AB, TEST_END, TEST_START,
 )
from r1_utils import append_cache_manifest, canonical_basin_id, read_canonical_ids, sha256_file, utc_now

from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.model_registry import build_model, get_spec
from src.objective import streaming_kge

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

CACHE_VERSION = "r1_temporal_ab_v2"
FT3S_TO_MMD_NUMERATOR = 0.0283168 * 86400.0 * 1000.0


def compute_kge(q_sim: torch.Tensor, q_obs: torch.Tensor, eps: float = 0.1) -> np.ndarray:
    """Return one IC-compatible KGE value per basin."""
    if q_obs.ndim == 3 and q_obs.shape[-1] == 1:
        q_obs = q_obs.squeeze(-1)
    if q_obs.ndim != 2:
        raise ValueError(f"q_obs must be [time, basin], got {tuple(q_obs.shape)}")
    if q_sim.ndim == 2:
        prediction = q_sim.unsqueeze(-1).unsqueeze(-1)
    elif q_sim.ndim == 3 and q_sim.shape[-1] == 1:
        prediction = q_sim.unsqueeze(-1)
    elif q_sim.ndim == 4:
        prediction = q_sim
    else:
        raise ValueError(f"q_sim shape not supported: {tuple(q_sim.shape)}")
    kge, invalid = streaming_kge(prediction, q_obs, eps=eps)
    kge = kge.squeeze(-1).squeeze(-1)
    invalid = invalid.squeeze(-1).squeeze(-1)
    values = kge.detach().cpu().numpy().reshape(-1)
    bad = invalid.detach().cpu().numpy().reshape(-1) | ~np.isfinite(values)
    values[bad] = np.nan
    return values


def load_temporal_data(ids: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load the canonical CAMELS pickle and return validation forcing/targets."""
    ids_int = np.asarray([int(x) for x in ids], dtype=np.int64)
    with DATASET_PATH.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        forcing_raw, streamflow_raw = data["forcings"], data["streamflow"]
    elif isinstance(data, (tuple, list)):
        forcing_raw, streamflow_raw = data[0], data[1]
    else:
        raise TypeError(f"unsupported camels_dataset payload: {type(data)}")

    gage_ids = np.asarray(np.load(GAGE_IDS_PATH), dtype=np.int64).reshape(-1)
    index = {int(gid): i for i, gid in enumerate(gage_ids)}
    missing = [int(x) for x in ids_int if int(x) not in index]
    if missing:
        raise RuntimeError(f"canonical basins missing from gage_id.npy: {missing[:5]}")
    rows = [index[int(x)] for x in ids_int]
    forcing = np.asarray(forcing_raw[rows])
    streamflow = np.asarray(streamflow_raw[rows])
    if forcing.ndim != 3 or forcing.shape[-1] != 3:
        raise RuntimeError(f"unexpected forcing shape: {forcing.shape}")
    if streamflow.ndim != 3 or streamflow.shape[-1] != 1:
        raise RuntimeError(f"unexpected streamflow shape: {streamflow.shape}")
    forcing = np.transpose(forcing, (1, 0, 2)).astype(np.float32, copy=False)
    streamflow = np.transpose(streamflow[:, :, 0], (1, 0)).astype(np.float32, copy=False)

    # This is the same ft3/s -> mm/day conversion used by the canonical loader.
    raw_attributes = CatchmentAttributeBuilder(data_root=DATA_ROOT).load_raw_attributes(ids_int)
    area_km2 = np.asarray(raw_attributes[:, 11], dtype=np.float64)
    if not np.isfinite(area_km2).all() or (area_km2 <= 0).any():
        raise RuntimeError("invalid area_gages2 values in canonical Caravan attributes")
    streamflow_mmd = streamflow * (FT3S_TO_MMD_NUMERATOR / (area_km2 * 1.0e6))[None, :]

    source_dates = pd.DatetimeIndex(np.load(DATE_INDEX_PATH).astype("datetime64[ns]"))
    if len(source_dates) != forcing.shape[0] or source_dates[0] != pd.Timestamp("1980-10-01") or source_dates[-1] != pd.Timestamp("2014-09-30"):
        raise RuntimeError("camels_dates.npy is not the expected canonical 1980-10-01..2014-09-30 index")
    left = int(source_dates.get_loc(pd.Timestamp(TEST_START)))
    right = int(source_dates.get_loc(pd.Timestamp(TEST_END))) + 1
    forcing_left = left - EVAL_WARMUP_DAYS
    if forcing_left < 0:
        raise RuntimeError("test period lacks the required 365-day forcing warm-up")
    x = forcing[forcing_left:right]
    y = streamflow_mmd[left:right]
    expected_days = right - left
    if x.shape != (expected_days + EVAL_WARMUP_DAYS, len(ids), 3) or y.shape != (expected_days, len(ids)):
        raise RuntimeError(f"validation shape mismatch: forcing={x.shape}, target={y.shape}")
    return x, y, source_dates[forcing_left:right]

def temporal_slices(output_dates: pd.DatetimeIndex) -> list[tuple[str, int, int, str, str]]:
    """Resolve configured A/B dates to indices and require complete coverage."""
    dates = pd.DatetimeIndex(output_dates)
    if dates[0] != pd.Timestamp(TEST_START) or dates[-1] != pd.Timestamp(TEST_END):
        raise RuntimeError("post-warmup date index does not equal the canonical TEST period")
    windows = []
    cursor = 0
    for definition in TEMPORAL_AB:
        start = pd.Timestamp(definition["start_date"])
        end = pd.Timestamp(definition["end_date"])
        left = int(dates.get_loc(start))
        right = int(dates.get_loc(end)) + 1
        if left != cursor or right <= left:
            raise RuntimeError(f"temporal definition is not contiguous at partition {definition['partition']}")
        windows.append((definition["partition"], left, right, definition["start_date"], definition["end_date"]))
        cursor = right
    if cursor != len(dates):
        raise RuntimeError("temporal A/B definition does not cover the full TEST period")
    return windows


def load_ic_latent(model: str, canonical_ids: list[str]) -> tuple[torch.Tensor, list[Path], int]:
    """Load full CMA-ES checkpoints and select their stored best per basin/start."""
    status_summary = json.loads(IC_STATUS_PATH.read_text())
    status = status_summary.get(model, {})
    generation_values = []
    if status.get("generation") is not None:
        generation_values = [int(status["generation"])]
    generation_values.extend(int(x) for x in status.get("latest_generation_by_chunk", {}).values())
    if not generation_values and status.get("latest_checkpoint"):
        import re
        match = re.search(r"_gen_(\d+)\.pt$", str(status["latest_checkpoint"]))
        if match:
            generation_values = [int(match.group(1))]
    if not generation_values or len(set(generation_values)) != 1:
        raise RuntimeError(f"{model}: IC generation metadata is missing or ambiguous")
    expected_generation = generation_values[0]
    paths = sorted((IC_CHECKPOINT_ROOT / model).glob(f"chunk_*_gen_{expected_generation}.pt"))
    if not paths:
        raise RuntimeError(f"{model}: no full IC checkpoints for generation {expected_generation}")
    ids, latent_parts, fitness_parts = [], [], []
    generations = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        generation = int(payload["generation"])
        if generation != expected_generation:
            raise RuntimeError(f"{model}: IC payload generation {generation} != status {expected_generation}")
        basin_ids = [canonical_basin_id(x) for x in np.asarray(payload["basin_ids"]).reshape(-1)]
        state = payload["solver"]["state"]
        latent = state["best_latent"].detach().cpu()
        fitness = state["best_fitness"].detach().cpu().numpy().reshape(-1)
        if latent.ndim != 2 or latent.shape[0] != len(basin_ids) * 10 or fitness.size != len(basin_ids) * 10:
            raise RuntimeError(f"{model}: unexpected full IC checkpoint shape in {path}")
        ids.extend(basin_ids)
        latent_parts.append(latent)
        fitness_parts.append(fitness)
        generations.append(generation)
    if len(ids) != len(set(ids)) or set(ids) != set(canonical_ids):
        raise RuntimeError(f"{model}: IC basin IDs do not exactly match canonical 531 IDs")
    latent_all = torch.cat(latent_parts, dim=0).reshape(len(ids), 10, -1)
    fitness_all = np.concatenate(fitness_parts).reshape(len(ids), 10)
    order = np.asarray([ids.index(basin) for basin in canonical_ids], dtype=np.int64)
    latent_all = latent_all[order]
    fitness_all = fitness_all[order]
    selected = fitness_all.argmax(axis=1)
    chosen = latent_all[torch.arange(len(canonical_ids)), torch.as_tensor(selected)]
    return chosen, paths, expected_generation


def load_dpl_network(model: str, attrs: torch.Tensor, device: torch.device) -> tuple[CatchmentParameterizer, Path, dict[str, object]]:
    checkpoint = DPL_ROOT / "runs" / model / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"{model}: missing canonical dPL best.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    job = payload.get("job_config", {})
    if not isinstance(job, dict) or int(job.get("seed", -1)) != CANONICAL_DPL_SEED or str(payload.get("git_sha", "")) != CANONICAL_DPL_SOURCE_SHA:
        raise RuntimeError(f"{model}: dPL best.pt fails canonical seed/source provenance gate")
    if str(job.get("test_period")) != "['1995-10-01', '2010-09-30']" or int(job.get("test_warmup_days", -1)) != EVAL_WARMUP_DAYS:
        raise RuntimeError(f"{model}: dPL best.pt fails canonical test-period/warm-up provenance gate")
    spec = get_spec(model, device="cpu")
    network = CatchmentParameterizer(
        in_features=attrs.shape[1], out_features=spec.dimension, hidden_dims=[256, 256],
        dropout=0.05, parameter_names=list(spec.parameter_names), output_transform="sigmoid",
    ).to(device=device, dtype=torch.float64)
    network.load_state_dict(payload["network"])
    network.eval()
    return network, checkpoint, payload


def implementation_fingerprints() -> dict[str, str]:
    roots = [SCRIPTS_DIR, BENCHMARK / "src", BENCHMARK / "dpl", BENCHMARK.parent.parent / "dmotpy"]
    paths = [path for root in roots if root.is_dir() for path in root.rglob("*.py")]
    return {str(path): sha256_file(path) for path in sorted(set(paths))}


def data_source_fingerprints() -> dict[str, str]:
    data_paths = {
        "camels_dataset": DATASET_PATH, "gage_id.npy": GAGE_IDS_PATH,
        "caravan_671_attributes.npy": CARAVAN_PATH, "camels_dates.npy": DATE_INDEX_PATH,
    }
    return {name: sha256_file(path) for name, path in data_paths.items() if path.is_file()}


def current_source_fingerprints(
    model: str, data_hashes: dict[str, str], implementation_hashes: dict[str, str],
 ) -> tuple[dict[str, str], str, dict[str, str], dict[str, str]]:
    ic_paths = sorted((IC_CHECKPOINT_ROOT / model).glob("chunk_*_gen_*.pt"))
    if not ic_paths:
        return {}, "", data_hashes, implementation_hashes
    ic_hashes = {str(path): sha256_file(path) for path in ic_paths}
    dpl_path = DPL_ROOT / "runs" / model / "best.pt"
    dpl_hash = sha256_file(dpl_path) if dpl_path.is_file() else ""
    return ic_hashes, dpl_hash, data_hashes, implementation_hashes


def valid_cache(model: str, canonical_ids: list[str], data_hashes: dict[str, str], implementation_hashes: dict[str, str]) -> bool:
    npz_path = CACHE_DIR / f"R1_temporal_AB_{model}.npz"
    meta_path = CACHE_DIR / f"R1_temporal_AB_{model}.json"
    if not npz_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        arrays = np.load(npz_path)
        basin_values = arrays["basin_id"].astype(str)
        partition_values = arrays["partition"].astype(str)
        expected_ids = np.asarray(canonical_ids)
        ic_hashes, dpl_hash, _, _ = current_source_fingerprints(model, data_hashes, implementation_hashes)
        if not (meta.get("cache_version") == CACHE_VERSION and meta.get("model") == model):
            return False
        if meta.get("test_period") != f"{TEST_START}..{TEST_END}" or meta.get("temporal_definition") != [dict(x) for x in TEMPORAL_AB]:
            return False
        if meta.get("dpl_seed") != CANONICAL_DPL_SEED or meta.get("dpl_source_sha") != CANONICAL_DPL_SOURCE_SHA:
            return False
        if meta.get("ic_source_sha256") != ic_hashes or meta.get("dpl_best_pt_sha256") != dpl_hash or meta.get("data_source_sha256") != data_hashes:
            return False
        if meta.get("implementation_sha256") != implementation_hashes or meta.get("cache_sha256") != sha256_file(npz_path):
            return False
        if arrays["model"].size != 1 or str(arrays["model"][0]) != model:
            return False
        if basin_values.size != 2 * len(canonical_ids):
            return False
        start_values = arrays["start_date"].astype(str)
        end_values = arrays["end_date"].astype(str)
        if start_values.size != basin_values.size or end_values.size != basin_values.size:
            return False
        for partition in ("A", "B"):
            mask = partition_values == partition
            part_ids = basin_values[mask]
            definition = next(item for item in TEMPORAL_AB if item["partition"] == partition)
            if (part_ids.size != len(canonical_ids) or not np.array_equal(part_ids, expected_ids)
                    or not np.all(start_values[mask] == definition["start_date"])
                    or not np.all(end_values[mask] == definition["end_date"])):
                return False
        kge = np.column_stack([arrays["KGE_IC"], arrays["KGE_dPL"], arrays["delta_KGE"]]).astype(float)
        return np.isfinite(kge).all() and np.allclose(kge[:, 2], kge[:, 1] - kge[:, 0], rtol=0.0, atol=1e-12)
    except Exception:
        return False


def forward_model(model: str, ids: list[str], x_np: np.ndarray, y_np: np.ndarray, input_dates: pd.DatetimeIndex, data_source_hashes: dict[str, str], implementation_hashes: dict[str, str], device: torch.device) -> tuple[pd.DataFrame, dict[str, object]]:
    ic_latent, ic_paths, ic_generation = load_ic_latent(model, ids)
    attrs = CatchmentAttributeBuilder(data_root=DATA_ROOT).build_normalized_attributes(
        np.asarray([int(x) for x in ids], dtype=np.int64), device=str(device), method="zscore",
    ).to(dtype=torch.float64)
    network, dpl_checkpoint, dpl_payload = load_dpl_network(model, attrs, device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    output_dates = pd.DatetimeIndex(input_dates[EVAL_WARMUP_DAYS:])
    windows = temporal_slices(output_dates)
    if len(input_dates) != len(x_np) or len(output_dates) != len(y_np):
        raise RuntimeError(f"{model}: forcing/date/target lengths are inconsistent")
    if model in CALENDAR_MODELS:
        x, _ = add_calendar_forcing(x, input_dates, model_name=model)
    hydro_ic = build_model(model, device, warm_up=EVAL_WARMUP_DAYS, backend="eager", parameter_mapping="linear", dtype=torch.float64)
    hydro_dpl = build_model(model, device, warm_up=EVAL_WARMUP_DAYS, backend="eager", parameter_mapping="auto", dtype=torch.float64)
    with torch.inference_mode():
        ic_theta = torch.sigmoid(ic_latent.to(device=device, dtype=torch.float64))
        dpl_theta = network(attrs)
        q_ic = hydro_ic({"x_phy": x}, (None, ic_theta.unsqueeze(-1)))["streamflow"]
        q_dpl = hydro_dpl({"x_phy": x}, (None, dpl_theta.unsqueeze(-1)))["streamflow"]
        expected = len(output_dates)
        if q_ic.shape[0] != expected or q_dpl.shape[0] != expected:
            raise RuntimeError(f"{model}: model output length does not equal post-warmup TEST length")
        rows: list[dict[str, object]] = []
        for label, left, right, start_date, end_date in windows:
            ic_kge = compute_kge(q_ic[left:right], y[left:right])
            dpl_kge = compute_kge(q_dpl[left:right], y[left:right])
            if len(ic_kge) != len(ids) or len(dpl_kge) != len(ids):
                raise RuntimeError(f"{model}/{label}: KGE vector length mismatch")
            for basin, ic_value, dpl_value in zip(ids, ic_kge, dpl_kge):
                rows.append({
                    "model": model, "basin_id": basin, "partition": label,
                    "start_date": start_date, "end_date": end_date,
                    "KGE_IC": ic_value, "KGE_dPL": dpl_value,
                    "delta_KGE": dpl_value - ic_value,
                })
    del hydro_ic, hydro_dpl, network, q_ic, q_dpl, x, y, attrs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    meta = {
        "cache_version": CACHE_VERSION, "model": model, "created_utc": utc_now(),
        "script": "02_temporal_ab_forward.py", "test_period": f"{TEST_START}..{TEST_END}",
        "evaluation_warmup_days": EVAL_WARMUP_DAYS, "temporal_definition": [dict(x) for x in TEMPORAL_AB],
        "basin_count": len(ids), "ic_generation": ic_generation,
        "ic_source_paths": [str(x) for x in ic_paths],
        "ic_source_sha256": {str(path): sha256_file(path) for path in ic_paths},
        "dpl_best_pt": str(dpl_checkpoint), "dpl_best_pt_sha256": sha256_file(dpl_checkpoint),
        "dpl_payload_epoch": dpl_payload.get("epoch"), "dpl_seed": int(dpl_payload["job_config"]["seed"]),
        "dpl_source_sha": str(dpl_payload.get("git_sha", "")),
        "dpl_test_period": dpl_payload["job_config"]["test_period"],
        "dpl_test_warmup_days": int(dpl_payload["job_config"]["test_warmup_days"]),
        "data_root": str(DATA_ROOT),
        "data_source_sha256": data_source_hashes, "implementation_sha256": implementation_hashes,
        "backend": "eager", "dtype_forcing_target": "float32", "dtype_network_hydrology": "float64",
        "forward_only": True, "optimizer_constructed": False, "backward_called": False,
    }
    return frame, meta


def save_model_cache(frame: pd.DataFrame, meta: dict[str, object]) -> None:
    model = str(meta["model"])
    npz_path = CACHE_DIR / f"R1_temporal_AB_{model}.npz"
    meta_path = CACHE_DIR / f"R1_temporal_AB_{model}.json"
    # Store the long rows compactly; final CSV/parquet are rebuilt below.
    string_cols = {"basin_id", "partition", "start_date", "end_date"}
    arrays = {col: np.asarray(frame[col].to_numpy(), dtype="<U32" if col in string_cols else float)
              for col in ["basin_id", "partition", "start_date", "end_date", "KGE_IC", "KGE_dPL", "delta_KGE"]}
    temp_npz = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp_npz, model=np.asarray([model]), **arrays)
    temp_npz.replace(npz_path)
    meta["cache_sha256"] = sha256_file(npz_path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")


def frame_from_cache(model: str) -> pd.DataFrame:
    arrays = np.load(CACHE_DIR / f"R1_temporal_AB_{model}.npz")
    return pd.DataFrame({
        "model": model, "basin_id": arrays["basin_id"].astype(str), "partition": arrays["partition"].astype(str),
        "start_date": arrays["start_date"].astype(str), "end_date": arrays["end_date"].astype(str),
        "KGE_IC": arrays["KGE_IC"].astype(float), "KGE_dPL": arrays["KGE_dPL"].astype(float),
        "delta_KGE": arrays["delta_KGE"].astype(float),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, choices=MODEL_REGISTRY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    models = list(args.models) if args.models else list(MODEL_REGISTRY)
    ids = read_canonical_ids(DATA_ROOT / "531sub_id.txt")
    if len(ids) != 531 or len(set(ids)) != 531:
        raise RuntimeError("canonical basin list is not exactly 531 unique IDs")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("R1 P0-1 requires the available CUDA forward path")
    print(f"FORWARD_DEVICE={torch.cuda.get_device_name(0)}", flush=True)
    x_np, y_np, input_dates = load_temporal_data(ids)
    data_source_hashes = data_source_fingerprints()
    implementation_hashes = implementation_fingerprints()
    all_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for model in models:
        t0 = time.perf_counter()
        if not args.force and valid_cache(model, ids, data_source_hashes, implementation_hashes):
            frame = frame_from_cache(model)
            meta = json.loads((CACHE_DIR / f"R1_temporal_AB_{model}.json").read_text())
            print(f"[{model}] cache reused", flush=True)
        else:
            frame, meta = forward_model(model, ids, x_np, y_np, input_dates, data_source_hashes, implementation_hashes, device)
            if not np.isfinite(frame[["KGE_IC", "KGE_dPL", "delta_KGE"]].to_numpy(dtype=float)).all():
                raise RuntimeError(f"{model}: non-finite temporal KGE encountered")
            save_model_cache(frame, meta)
            print(f"[{model}] forward complete in {time.perf_counter() - t0:.1f}s", flush=True)
        if len(frame) != 2 * 531 or set(frame.partition) != {"A", "B"}:
            raise RuntimeError(f"{model}: temporal cache does not contain both A and B partitions")
        for partition in ("A", "B"):
            part_ids = frame.loc[frame.partition == partition, "basin_id"].astype(str).to_numpy()
            if part_ids.size != len(ids) or not np.array_equal(part_ids, np.asarray(ids)):
                raise RuntimeError(f"{model}/{partition}: temporal basin IDs do not exactly match canonical IDs")
        all_frames.append(frame)
        manifest_rows.append({
            "cache_file": f"R1_temporal_AB_{model}.npz", "kind": "npz", "model": model,
            "created_utc": meta.get("created_utc", utc_now()), "script": "02_temporal_ab_forward.py",
            "source_provenance": f"IC full CMA-ES checkpoint + dPL {DPL_ROOT / 'runs' / model / 'best.pt'}",
            "configuration": "365d eval warmup; post-warmup A/B; eager; forcing/target float32; network/hydrology float64",
            "status": "PRESENT",
            "sha256": sha256_file(CACHE_DIR / f"R1_temporal_AB_{model}.npz"),
        })
    frame = pd.concat(all_frames, ignore_index=True)
    frame = frame.sort_values(["model", "partition", "basin_id"], kind="stable").reset_index(drop=True)
    expected_rows = len(models) * 531 * 2
    if len(frame) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} temporal rows, got {len(frame)}")
    frame.to_csv(TABLES_DIR / "R1_temporal_AB_model_basin.csv", index=False, float_format="%.10f")
    try:
        frame.to_parquet(CACHE_DIR / "R1_temporal_AB_model_basin.parquet", index=False)
    except Exception as exc:
        raise RuntimeError(f"failed to write required parquet cache: {exc}") from exc
    manifest_rows.append({
        "cache_file": "R1_temporal_AB_model_basin.parquet", "kind": "parquet", "model": "ALL_REQUESTED",
        "created_utc": utc_now(), "script": "02_temporal_ab_forward.py",
        "source_provenance": "reassembled from per-model forward caches; canonical IDs",
        "configuration": f"models={','.join(models)}; rows={len(frame)}", "status": "PRESENT",
    })
    append_cache_manifest(manifest_rows)
    print(f"PASS: wrote {TABLES_DIR / 'R1_temporal_AB_model_basin.csv'} ({len(frame)} rows)")
    print(f"PASS: wrote {CACHE_DIR / 'R1_temporal_AB_model_basin.parquet'}")
    print(f"PASS: wrote {CACHE_DIR / 'cache_manifest.csv'}")


if __name__ == "__main__":
    main()
