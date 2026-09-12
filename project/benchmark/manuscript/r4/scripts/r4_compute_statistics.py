#!/usr/bin/env python3
"""Compute the frozen-case R4 OOB statistics (Steps 5--12).

The script is intentionally one-model-at-a-time.  It reads the frozen seen-basin
atlas and the exact OOB best checkpoints only after Stage A has written and hashed
the case manifest.  No training, optimizer, checkpoint, or formal OOB result is
modified.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# Import the already validated OOB reconstruction path; this script never calls run_job.
SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[5]
BENCHMARK = REPO / "project/benchmark"
OOB_SCRIPT_DIR = BENCHMARK / "scripts/oob"
sys.path.insert(0, str(OOB_SCRIPT_DIR))
sys.path.insert(0, str(BENCHMARK))

from dpl.nn_parameterizer import CatchmentParameterizer  # noqa: E402
from oob_common import NormalizationStats  # noqa: E402
from run_oob_job import _make_parameterizer  # noqa: E402
from src.model_registry import get_spec  # noqa: E402
from src.data_selection import load_ids  # noqa: E402


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

R4 = BENCHMARK / "manuscript/r4"
FORMAL = BENCHMARK / "results/oob_primary8_5fold_20260902"
SEEN = BENCHMARK / "results/ic_dpl_seenbasin_formal_20260901"
CLUSTER_FILE = BENCHMARK / "results/parameter_attribute_atlas_followup_20260829/attribute_redundancy_groups.csv"
PRIMARY8 = ("alpine2", "hbv96", "xinanjiang", "newzealand2", "ihacres", "us1", "mopex4", "hillslope")
N_FOLDS = 5
N_BASINS = 531
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260902
NONTRIVIAL = 0.10
BOUNDARY_EPS = 0.01
BOUNDARY_CONCENTRATION = 0.50
LOG_MAPPING_SPAN_THRESHOLD = 100.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def sign(value: float) -> int:
    if not finite(value) or value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def rank_average_2d(values: torch.Tensor) -> torch.Tensor:
    """Average ranks per column, retaining ties, on the selected device."""
    if values.ndim != 2:
        raise ValueError(f"expected [n, columns], got {tuple(values.shape)}")
    n, columns = values.shape
    ranks = torch.empty_like(values)
    positions = torch.arange(n, device=values.device, dtype=torch.float64)
    for column in range(columns):
        vector = values[:, column]
        order = torch.argsort(vector, stable=True)
        sorted_values = vector[order]
        starts = torch.ones(n, device=values.device, dtype=torch.bool)
        if n > 1:
            starts[1:] = sorted_values[1:] != sorted_values[:-1]
        group = torch.cumsum(starts.to(torch.long), dim=0) - 1
        counts = torch.zeros(n, device=values.device, dtype=torch.float64)
        sums = torch.zeros(n, device=values.device, dtype=torch.float64)
        counts.scatter_add_(0, group, torch.ones(n, device=values.device, dtype=torch.float64))
        sums.scatter_add_(0, group, positions + 1.0)
        average_sorted = sums[group] / counts[group]
        ranks[:, column].scatter_(0, order, average_sorted)
    return ranks


def rank_average_batch(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for [batch, n, columns], including bootstrap duplicate ties."""
    if values.ndim != 3:
        raise ValueError(f"expected [batch, n, columns], got {tuple(values.shape)}")
    batch, n, columns = values.shape
    order = torch.argsort(values, dim=1, stable=True)
    sorted_values = torch.gather(values, 1, order)
    starts = torch.ones((batch, n, columns), device=values.device, dtype=torch.bool)
    if n > 1:
        starts[:, 1:, :] = sorted_values[:, 1:, :] != sorted_values[:, :-1, :]
    group = torch.cumsum(starts.to(torch.long), dim=1) - 1
    ones = torch.ones((batch, n, columns), device=values.device, dtype=torch.float64)
    positions = torch.arange(n, device=values.device, dtype=torch.float64).view(1, n, 1) + 1.0
    counts = torch.zeros((batch, n, columns), device=values.device, dtype=torch.float64)
    sums = torch.zeros((batch, n, columns), device=values.device, dtype=torch.float64)
    counts.scatter_add_(1, group, ones)
    sums.scatter_add_(1, group, positions.expand(batch, -1, columns))
    average_sorted = sums.gather(1, group) / counts.gather(1, group)
    ranks = torch.zeros_like(values)
    ranks.scatter_(1, order, average_sorted)
    return ranks


def correlation_matrix(attributes: np.ndarray, parameters: np.ndarray, device: torch.device) -> np.ndarray:
    """Compute attribute-by-parameter Spearman rho on CUDA with average ranks."""
    attr = torch.as_tensor(attributes, dtype=torch.float64, device=device)
    param = torch.as_tensor(parameters, dtype=torch.float64, device=device)
    attr_rank = rank_average_2d(attr)
    param_rank = rank_average_2d(param)
    attr_centered = attr_rank - attr_rank.mean(dim=0, keepdim=True)
    param_centered = param_rank - param_rank.mean(dim=0, keepdim=True)
    numerator = attr_centered.transpose(0, 1) @ param_centered
    denominator = torch.sqrt(
        attr_centered.square().sum(dim=0)[:, None] * param_centered.square().sum(dim=0)[None, :]
    )
    result = torch.where(denominator > 0.0, numerator / denominator, torch.full_like(numerator, float("nan")))
    return result.detach().cpu().numpy()


def choose_bootstrap_batch(device: torch.device, n: int, columns: int) -> int:
    if device.type != "cuda":
        return 1
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Sorting/scatter uses several working copies; cap deliberately below VRAM saturation.
    estimate_per_sample = max(n * columns * 8 * 8, 1)
    return max(16, min(256, int((free_bytes * 0.25) // estimate_per_sample)))


def bootstrap_spearman(
    attributes: np.ndarray,
    parameters: np.ndarray,
    device: torch.device,
    *,
    model_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return percentile CI statistics for every attribute/parameter cell."""
    if device.type != "cuda":
        raise RuntimeError("GPU_UNAVAILABLE_STOP: bootstrap is not permitted on CPU")
    n, n_attributes = attributes.shape
    n_parameters = parameters.shape[1]
    batch_size = choose_bootstrap_batch(device, n, n_attributes + n_parameters)
    attr = torch.as_tensor(attributes, dtype=torch.float64, device=device)
    param = torch.as_tensor(parameters, dtype=torch.float64, device=device)
    started = time.perf_counter()
    last_error: str | None = None
    for attempt in range(3):
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(BOOTSTRAP_SEED)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            batches: list[torch.Tensor] = []
            for start in range(0, BOOTSTRAP_N, batch_size):
                current = min(batch_size, BOOTSTRAP_N - start)
                indices = torch.randint(0, n, (current, n), device=device, generator=generator)
                attr_sample = attr[indices]
                param_sample = param[indices]
                attr_rank = rank_average_batch(attr_sample)
                param_rank = rank_average_batch(param_sample)
                attr_rank -= attr_rank.mean(dim=1, keepdim=True)
                param_rank -= param_rank.mean(dim=1, keepdim=True)
                numerator = torch.einsum("bna,bnp->bap", attr_rank, param_rank)
                denominator = torch.sqrt(
                    attr_rank.square().sum(dim=1)[:, :, None] * param_rank.square().sum(dim=1)[:, None, :]
                )
                rho = torch.where(denominator > 0.0, numerator / denominator, torch.full_like(numerator, float("nan")))
                batches.append(rho.detach().cpu())
                del indices, attr_sample, param_sample, attr_rank, param_rank, numerator, denominator, rho
            values = torch.cat(batches, dim=0).numpy()
            lower, upper = np.nanpercentile(values, [2.5, 97.5], axis=0)
            metadata = {
                "device": torch.cuda.get_device_name(device),
                "batch_size": batch_size,
                "attempt": attempt + 1,
                "n_boot": BOOTSTRAP_N,
                "seed": BOOTSTRAP_SEED,
                "runtime_seconds": time.perf_counter() - started,
                "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            del batches, values, attr, param
            torch.cuda.empty_cache()
            return np.stack([lower, upper], axis=0), metadata
        except torch.cuda.OutOfMemoryError as exc:
            last_error = str(exc)
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            gc.collect()
    raise RuntimeError(f"GPU bootstrap failed after two batch-size retries: {last_error}")


def parameter_physical(u: np.ndarray, bounds: np.ndarray, mapping: str = "auto") -> np.ndarray:
    result = np.empty_like(u, dtype=np.float64)
    for index, (lower, upper) in enumerate(bounds):
        values = u[:, index]
        use_log = mapping in {"auto", "auto_log", "log_auto"} and lower > 0.0 and upper > lower and upper / lower >= LOG_MAPPING_SPAN_THRESHOLD
        if use_log:
            result[:, index] = np.exp(np.log(lower) + values * (np.log(upper) - np.log(lower)))
        else:
            result[:, index] = lower + values * (upper - lower)
    return result


def boundary_summary(values: np.ndarray) -> dict[str, Any]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {"n": 0, "low_fraction": np.nan, "high_fraction": np.nan, "sd": np.nan, "unique": 0, "tie_fraction": np.nan, "near_constant": True, "boundary_concentrated": True}
    low = float(np.mean(finite_values <= BOUNDARY_EPS))
    high = float(np.mean(finite_values >= 1.0 - BOUNDARY_EPS))
    unique = int(np.unique(finite_values).size)
    return {
        "n": int(finite_values.size),
        "low_fraction": low,
        "high_fraction": high,
        "sd": float(np.std(finite_values)),
        "unique": unique,
        "tie_fraction": float(1.0 - unique / finite_values.size),
        "near_constant": bool(unique <= 1 or np.std(finite_values) <= 0.01),
        "boundary_concentrated": bool(low >= BOUNDARY_CONCENTRATION or high >= BOUNDARY_CONCENTRATION),
    }


def point_boundary(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & ((values <= BOUNDARY_EPS) | (values >= 1.0 - BOUNDARY_EPS))


def load_cluster_map() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    rows = read_csv(CLUSTER_FILE)
    for row in rows.itertuples(index=False):
        if not math.isclose(float(row.threshold), 0.70, rel_tol=0.0, abs_tol=1.0e-12) or row.attribute_type != "CONTINUOUS":
            continue
        result[str(row.attribute)] = {
            "cluster_id": str(row.cluster_id),
            "representative": str(row.representative_attribute),
            "members": str(row.member_attributes),
        }
    return result


def load_inputs() -> dict[str, Any]:
    basin_ids = load_ids(REPO / "data/531sub_id.txt")
    if basin_ids.size != N_BASINS or np.unique(basin_ids).size != N_BASINS:
        raise RuntimeError("canonical basin list is not exactly 531 unique IDs")
    gage_ids = np.asarray(np.load(REPO / "data/gage_id.npy"), dtype=np.int64).reshape(-1)
    raw_attributes = np.asarray(np.load(REPO / "data/caravan_671_attributes.npy"), dtype=np.float64)
    positions = {int(basin_id): index for index, basin_id in enumerate(gage_ids)}
    missing = [int(basin_id) for basin_id in basin_ids if int(basin_id) not in positions]
    if missing or raw_attributes.shape[0] != gage_ids.size or raw_attributes.shape[1] != 35:
        raise RuntimeError(f"canonical attribute alignment failed; missing={missing[:5]} shape={raw_attributes.shape}")
    indices = np.asarray([positions[int(basin_id)] for basin_id in basin_ids], dtype=np.int64)
    atlas = read_csv(
        SEEN / "09_PARAMETER_ATTRIBUTE_ATLAS_LONG.csv",
        usecols=["model", "method", "parameter_index", "parameter", "attribute", "attribute_type", "rho", "n"],
    )
    atlas = atlas[(atlas.model.isin(PRIMARY8)) & (atlas.attribute_type == "CONTINUOUS")].copy()
    attributes = sorted(atlas.attribute.unique().tolist(), key=lambda value: (value,))
    # Preserve the canonical formal order used in the atlas rather than lexical order.
    order = []
    for value in atlas.attribute.tolist():
        if value not in order:
            order.append(value)
    attributes = order
    contract = read_json(SEEN / "ATTRIBUTE_CONTRACT.json")
    canonical_order = list(contract["attribute_order"])
    attributes = [name for name in canonical_order if name in set(attributes)]
    categorical = {"dom_land_cover", "geol_1st_class", "geol_2nd_class"}
    attributes = [name for name in attributes if name not in categorical]
    raw_attr_full = np.nan_to_num(raw_attributes[indices], nan=0.0, posinf=0.0, neginf=0.0)
    attr_positions = {name: index for index, name in enumerate(canonical_order)}
    raw_attr_matrix = raw_attr_full[:, [attr_positions[name] for name in attributes]]
    seen_parameters = read_csv(
        SEEN / "05_PARAMETER_ESTIMATES_LONG.csv",
        usecols=["model", "basin_id", "method", "parameter_index", "parameter", "physical_value", "normalized_u", "lower_bound", "upper_bound"],
    )
    seen_parameters = seen_parameters[seen_parameters.model.isin(PRIMARY8)].copy()
    seen_parameters["basin_id_int"] = seen_parameters.basin_id.astype(str).str.lstrip("0").replace("", "0").astype(np.int64)
    seen_kge = read_csv(SEEN / "02_BASIN_PAIRED_KGE_LONG.csv", usecols=["model", "basin_id", "KGE_dPL"])
    seen_kge = seen_kge[seen_kge.model.isin(PRIMARY8)].copy()
    seen_kge["basin_id_int"] = seen_kge.basin_id.astype(str).str.lstrip("0").replace("", "0").astype(np.int64)
    return {
        "basin_ids": basin_ids,
        "raw_attr_full": raw_attr_full,
        "raw_attr": raw_attr_matrix,
        "attributes": attributes,
        "atlas": atlas,
        "seen_parameters": seen_parameters,
        "seen_kge": seen_kge,
        "clusters": load_cluster_map(),
    }


def seen_model_arrays(inputs: dict[str, Any], model: str, parameter_names: tuple[str, ...], basin_ids: np.ndarray) -> dict[str, Any]:
    frame = inputs["seen_parameters"]
    frame = frame[frame.model == model]
    output: dict[str, Any] = {}
    for method, prefix in (("IC", "ic"), ("dPL", "seen")):
        method_frame = frame[frame.method == method]
        physical = np.full((basin_ids.size, len(parameter_names)), np.nan, dtype=np.float64)
        normalized = np.full_like(physical, np.nan)
        lower = np.full(len(parameter_names), np.nan, dtype=np.float64)
        upper = np.full(len(parameter_names), np.nan, dtype=np.float64)
        for p, name in enumerate(parameter_names):
            rows = method_frame[(method_frame.parameter_index == p) & (method_frame.parameter == name)].set_index("basin_id_int")
            if rows.empty:
                continue
            aligned = rows.reindex(basin_ids)
            physical[:, p] = aligned.physical_value.to_numpy(dtype=np.float64)
            normalized[:, p] = aligned.normalized_u.to_numpy(dtype=np.float64)
            lower[p] = float(aligned.lower_bound.dropna().iloc[0])
            upper[p] = float(aligned.upper_bound.dropna().iloc[0])
        output[f"{prefix}_physical"] = physical
        output[f"{prefix}_u"] = normalized
        output[f"{prefix}_lower"] = lower
        output[f"{prefix}_upper"] = upper
    return output


def load_oob_fold(model: str, fold: int, global_config: dict[str, Any], basin_ids: np.ndarray, raw_attr_by_basin: dict[int, np.ndarray], device: torch.device) -> dict[str, Any]:
    run = FORMAL / "runs" / model / f"fold_{fold}"
    heldout = np.loadtxt(run / "heldout_basin_ids.txt", dtype=np.int64).reshape(-1)
    metadata = read_json(run / "normalization_metadata.json")
    stats = NormalizationStats(
        method=str(metadata["method"]),
        log_transform_skewed=bool(metadata["log_transform_skewed"]),
        shifts=np.asarray(metadata["train_fitted_log_shifts"], dtype=np.float64),
        mean=np.load(run / "attribute_mean.npy"),
        std=np.load(run / "attribute_std.npy"),
    )
    raw = np.asarray([raw_attr_by_basin[int(basin_id)] for basin_id in heldout], dtype=np.float64)
    normalized_attributes = stats.transform(raw)
    spec = get_spec(model, device="cpu")
    network = _make_parameterizer(
        model,
        torch.empty((1, normalized_attributes.shape[1]), dtype=torch.float64, device=device),
        global_config,
        device,
    )
    payload = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    network.load_state_dict(payload["network"])
    network.eval()
    with torch.inference_mode():
        u = network(torch.as_tensor(normalized_attributes, dtype=torch.float64, device=device)).detach().cpu().numpy()
    bounds = spec.bounds.detach().cpu().numpy()
    physical = parameter_physical(u, bounds, mapping=str(global_config["protocol"]["parameter_mapping"]))
    if not np.isfinite(u).all() or not np.isfinite(physical).all():
        raise FloatingPointError(f"non-finite OOB parameters in {model}_fold{fold}")
    del network, payload, raw, normalized_attributes
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"heldout_ids": heldout, "u": u, "physical": physical, "fold": fold}


class ParquetAppender:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: pq.ParquetWriter | None = None

    def append(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def relationship_class(rho_ic: float, rho_seen: float, rank_ic: int, rank_seen: int) -> str:
    if not finite(rho_ic) or not finite(rho_seen):
        return "weak/unresolved"
    if abs(rho_ic) >= 0.10 and abs(rho_seen) >= 0.10 and sign(rho_ic) != sign(rho_seen):
        return "sign-changing"
    if abs(rho_ic) >= 0.20 and abs(rho_seen) < 0.10:
        return "attenuated"
    if abs(rho_seen) >= 0.20 and abs(rho_ic) < 0.10:
        return "dPL-emergent"
    if abs(rho_ic) >= 0.20 and abs(rho_seen) >= 0.20 and sign(rho_ic) == sign(rho_seen) and rank_ic <= 10 and rank_seen <= 10:
        return "persistent/reproduced"
    return "weak/unresolved"


def build_seen_relationship_map(inputs: dict[str, Any], model: str, parameter_names: tuple[str, ...], attributes: list[str]) -> dict[tuple[int, str], dict[str, Any]]:
    atlas = inputs["atlas"]
    frame = atlas[atlas.model == model]
    values: dict[tuple[int, str], dict[str, Any]] = defaultdict(dict)
    for row in frame.itertuples(index=False):
        if row.attribute not in attributes:
            continue
        key = (int(row.parameter_index), str(row.attribute))
        values[key][row.method] = float(row.rho)
    # Formal atlas rank is minimum rank of absolute rho within each model/parameter/method.
    for p, parameter in enumerate(parameter_names):
        for method, key_name in (("IC", "rank_ic"), ("dPL", "rank_seen")):
            rows = frame[(frame.parameter_index == p) & (frame.parameter == parameter) & (frame.method == method) & (frame.attribute.isin(attributes))]
            ordered = sorted([(str(row.attribute), abs(float(row.rho))) for row in rows.itertuples(index=False)], key=lambda item: (-item[1], item[0]))
            ranks: dict[str, int] = {}
            last: float | None = None
            current = 0
            for index, (attribute, absolute) in enumerate(ordered, start=1):
                if last is None or not math.isclose(absolute, last, rel_tol=0.0, abs_tol=1e-15):
                    current = index
                    last = absolute
                ranks[attribute] = current
            for attribute in attributes:
                item = values[(p, attribute)]
                item[key_name] = ranks.get(attribute, 9999)
                item["parameter"] = parameter
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for (p, attr), item in values.items():
        if "IC" not in item or "dPL" not in item:
            continue
        rho_ic = float(item["IC"])
        rho_seen = float(item["dPL"])
        result[(p, attr)] = {
            "parameter_index": p,
            "parameter": item["parameter"],
            "attribute": attr,
            "rho_ic": rho_ic,
            "rho_dpl_seen": rho_seen,
            "rank_ic": int(item.get("rank_ic", 9999)),
            "rank_seen": int(item.get("rank_seen", 9999)),
            "relationship_class": relationship_class(rho_ic, rho_seen, int(item.get("rank_ic", 9999)), int(item.get("rank_seen", 9999))),
        }
    return result


def point_flags(values: np.ndarray, parameter_names: tuple[str, ...], prefix: str) -> dict[str, Any]:
    summaries = {name: boundary_summary(values[:, index]) for index, name in enumerate(parameter_names)}
    return {f"{prefix}_{name}": summary for name, summary in summaries.items()}


def make_parameter_master(
    model: str,
    parameter_names: tuple[str, ...],
    basin_ids: np.ndarray,
    fold_for_basin: dict[int, int],
    seen: dict[str, Any],
    oob_u: np.ndarray,
    oob_physical: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    ic_flags = [boundary_summary(seen["ic_u"][:, p]) for p in range(len(parameter_names))]
    seen_flags = [boundary_summary(seen["seen_u"][:, p]) for p in range(len(parameter_names))]
    oob_flags = [boundary_summary(oob_u[:, p]) for p in range(len(parameter_names))]
    for basin_index, basin_id in enumerate(basin_ids):
        for p, parameter in enumerate(parameter_names):
            records.append({
                "model": model,
                "basin_id": int(basin_id),
                "fold": int(fold_for_basin[int(basin_id)]),
                "parameter_index": p,
                "parameter": parameter,
                "theta_ic_u": float(seen["ic_u"][basin_index, p]),
                "theta_seen_u": float(seen["seen_u"][basin_index, p]),
                "theta_oob_u": float(oob_u[basin_index, p]),
                "theta_ic_physical": float(seen["ic_physical"][basin_index, p]),
                "theta_seen_physical": float(seen["seen_physical"][basin_index, p]),
                "theta_oob_physical": float(oob_physical[basin_index, p]),
                "lower_bound": float(seen["ic_lower"][p]),
                "upper_bound": float(seen["ic_upper"][p]),
                "boundary_flag_ic": bool(point_boundary(seen["ic_u"][:, p])[basin_index]),
                "boundary_flag_seen": bool(point_boundary(seen["seen_u"][:, p])[basin_index]),
                "boundary_flag_oob": bool(point_boundary(oob_u[:, p])[basin_index]),
                "boundary_concentrated_ic": ic_flags[p]["boundary_concentrated"],
                "boundary_concentrated_seen": seen_flags[p]["boundary_concentrated"],
                "boundary_concentrated_oob": oob_flags[p]["boundary_concentrated"],
                "near_constant_ic": ic_flags[p]["near_constant"],
                "near_constant_seen": seen_flags[p]["near_constant"],
                "near_constant_oob": oob_flags[p]["near_constant"],
                "tie_fraction_ic": ic_flags[p]["tie_fraction"],
                "tie_fraction_seen": seen_flags[p]["tie_fraction"],
                "tie_fraction_oob": oob_flags[p]["tie_fraction"],
            })
    return pd.DataFrame.from_records(records)


def make_performance_master(model: str, basin_ids: np.ndarray, fold_for_basin: dict[int, int], seen_kge: pd.DataFrame, oob_kge: dict[int, float]) -> pd.DataFrame:
    seen = seen_kge[seen_kge.model == model].set_index("basin_id_int").reindex(basin_ids)
    records = []
    for basin_index, basin_id in enumerate(basin_ids):
        seen_value = float(seen.iloc[basin_index].KGE_dPL)
        oob_value = float(oob_kge[int(basin_id)])
        records.append({
            "model": model,
            "basin_id": int(basin_id),
            "fold": int(fold_for_basin[int(basin_id)]),
            "KGE_dPL_seen": seen_value,
            "KGE_dPL_OOB": oob_value,
            "delta_KGE": oob_value - seen_value,
        })
    return pd.DataFrame.from_records(records)


def population_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    total = len(frame)
    finite_frame = frame[np.isfinite(frame.rho_dpl_oob) & np.isfinite(frame.rho_dpl_seen) & np.isfinite(frame.rho_ic)]
    both_nonzero_seen = finite_frame[(finite_frame.rho_dpl_seen != 0.0) & (finite_frame.rho_dpl_oob != 0.0)]
    both_nonzero_ic = finite_frame[(finite_frame.rho_ic != 0.0) & (finite_frame.rho_dpl_oob != 0.0)]
    nontriv_seen = finite_frame[(finite_frame.rho_dpl_seen.abs() >= NONTRIVIAL) & (finite_frame.rho_dpl_oob.abs() >= NONTRIVIAL)]
    nontriv_ic = finite_frame[(finite_frame.rho_ic.abs() >= NONTRIVIAL) & (finite_frame.rho_dpl_oob.abs() >= NONTRIVIAL)]

    def corr(left: pd.Series, right: pd.Series) -> float:
        if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
            return float("nan")
        return float(left.rank(method="average").corr(right.rank(method="average")))

    def same(left: pd.Series, right: pd.Series) -> float:
        return float(np.mean([sign(a) == sign(b) for a, b in zip(left, right)])) if len(left) else float("nan")

    def fold_fraction(values: pd.Series, threshold: int) -> float:
        return float(np.mean(values >= threshold)) if len(values) else float("nan")

    fold_values = finite_frame["same_sign_folds_vs_seen"].to_numpy(dtype=float) if len(finite_frame) else np.asarray([])
    metrics = {
        "n_relationships_total": total,
        "n_oob_finite": int(len(finite_frame)),
        "n_both_nontrivial_seen_oob": int(len(nontriv_seen)),
        "n_both_nontrivial_ic_oob": int(len(nontriv_ic)),
        "sign_retention_vs_seen": same(both_nonzero_seen.rho_dpl_seen, both_nonzero_seen.rho_dpl_oob),
        "sign_retention_vs_ic": same(both_nonzero_ic.rho_ic, both_nonzero_ic.rho_dpl_oob),
        "nontrivial_sign_retention_vs_seen": same(nontriv_seen.rho_dpl_seen, nontriv_seen.rho_dpl_oob),
        "nontrivial_sign_retention_vs_ic": same(nontriv_ic.rho_ic, nontriv_ic.rho_dpl_oob),
        "seen_oob_correspondence_spearman": corr(finite_frame.rho_dpl_seen, finite_frame.rho_dpl_oob),
        "ic_oob_correspondence_spearman": corr(finite_frame.rho_ic, finite_frame.rho_dpl_oob),
        "median_abs_coefficient_distance_seen_oob": float(np.median(np.abs(finite_frame.rho_dpl_seen - finite_frame.rho_dpl_oob))) if len(finite_frame) else float("nan"),
        "median_abs_coefficient_distance_ic_oob": float(np.median(np.abs(finite_frame.rho_ic - finite_frame.rho_dpl_oob))) if len(finite_frame) else float("nan"),
        "median_strength_change_seen_oob": float(np.median(np.abs(finite_frame.rho_dpl_oob) - np.abs(finite_frame.rho_dpl_seen))) if len(finite_frame) else float("nan"),
        "median_strength_change_ic_oob": float(np.median(np.abs(finite_frame.rho_dpl_oob) - np.abs(finite_frame.rho_ic))) if len(finite_frame) else float("nan"),
        "fraction_5of5_same_sign_vs_seen": fold_fraction(finite_frame.same_sign_folds_vs_seen, 5),
        "fraction_ge4of5_same_sign_vs_seen": fold_fraction(finite_frame.same_sign_folds_vs_seen, 4),
        "fraction_le2of5_same_sign_vs_seen": float(np.mean(finite_frame.same_sign_folds_vs_seen <= 2)) if len(finite_frame) else float("nan"),
        "median_fold_spread": float(np.nanmedian(finite_frame.fold_max - finite_frame.fold_min)) if len(finite_frame) else float("nan"),
    }
    return metrics


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def main() -> None:
    started = time.perf_counter()
    R4.mkdir(parents=True, exist_ok=True)
    for directory in ("tables", "figure_data", "si", "logs"):
        (R4 / directory).mkdir(parents=True, exist_ok=True)
    frozen_path = R4 / "R4_RELATIONSHIP_CASES_FROZEN.csv"
    frozen_sha_path = R4 / "R4_RELATIONSHIP_CASES_FROZEN.sha256"
    frozen_sha = sha256_file(frozen_path)
    declared_sha = frozen_sha_path.read_text(encoding="utf-8").split()[0]
    if frozen_sha != declared_sha:
        raise RuntimeError(f"frozen case manifest SHA mismatch: {frozen_sha} != {declared_sha}")
    frozen = read_csv(frozen_path)
    if len(frozen) != 4:
        raise RuntimeError(f"expected exactly four frozen cases, got {len(frozen)}")
    if tuple(frozen.model) != tuple(frozen.model):
        raise RuntimeError("unreachable manifest check")

    if not torch.cuda.is_available():
        raise SystemExit("GPU_UNAVAILABLE_STOP")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    free_start, total_vram = torch.cuda.mem_get_info(device)
    gpu_log: dict[str, Any] = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "free_vram_start_bytes": int(free_start),
        "total_vram_bytes": int(total_vram),
        "cpu_threads": {"torch": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(), "OMP": os.environ["OMP_NUM_THREADS"], "MKL": os.environ["MKL_NUM_THREADS"], "OPENBLAS": os.environ["OPENBLAS_NUM_THREADS"]},
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    inputs = load_inputs()
    global_config = read_json(FORMAL / "runs" / PRIMARY8[0] / "fold_0" / "config.json")
    raw_attr_by_basin = {int(basin_id): inputs["raw_attr_full"][index] for index, basin_id in enumerate(inputs["basin_ids"])}
    fold_for_basin: dict[int, int] = {}
    assignment = read_csv(FORMAL / "OOB_FOLD_ASSIGNMENT.csv")
    for row in assignment.itertuples(index=False):
        fold_for_basin[int(row.basin_id)] = int(row.fold)
    if len(fold_for_basin) != N_BASINS:
        raise RuntimeError("fold assignment did not cover 531 basins")

    parameter_writer = ParquetAppender(R4 / "tables/R4_OOB_PARAMETER_MASTER.parquet")
    performance_writer = ParquetAppender(R4 / "tables/R4_OOB_PERFORMANCE_MASTER.parquet")
    qc_rows: list[dict[str, Any]] = []
    viability_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    per_model_summary: list[dict[str, Any]] = []
    try:
        for model_index, model in enumerate(PRIMARY8):
            model_started = time.perf_counter()
            spec = get_spec(model, device="cpu")
            parameter_names = tuple(spec.parameter_names)
            seen = seen_model_arrays(inputs, model, parameter_names, inputs["basin_ids"])
            if not np.isfinite(seen["ic_u"]).all() or not np.isfinite(seen["seen_u"]).all():
                raise FloatingPointError(f"non-finite seen parameters for {model}")
            fold_data: dict[int, dict[str, Any]] = {}
            for fold in range(N_FOLDS):
                fold_data[fold] = load_oob_fold(model, fold, global_config, inputs["basin_ids"], raw_attr_by_basin, device)
                for basin_id in fold_data[fold]["heldout_ids"]:
                    if int(basin_id) in fold_for_basin and fold_for_basin[int(basin_id)] != fold:
                        raise RuntimeError(f"fold assignment mismatch for {model} basin {basin_id}")
            oob_u = np.full((N_BASINS, len(parameter_names)), np.nan, dtype=np.float64)
            oob_physical = np.full_like(oob_u, np.nan)
            for fold, item in fold_data.items():
                positions = np.asarray([int(np.where(inputs["basin_ids"] == basin_id)[0][0]) for basin_id in item["heldout_ids"]], dtype=np.int64)
                oob_u[positions] = item["u"]
                oob_physical[positions] = item["physical"]
            if not np.isfinite(oob_u).all():
                raise FloatingPointError(f"OOB parameter reconstruction incomplete for {model}")
            parameter_master = make_parameter_master(model, parameter_names, inputs["basin_ids"], fold_for_basin, seen, oob_u, oob_physical)
            parameter_writer.append(parameter_master)
            duplicate_cells = int(parameter_master.duplicated(["model", "basin_id", "parameter"]).sum())
            qc_rows.append({
                "model": model,
                "expected_basins": N_BASINS,
                "actual_basins": int(parameter_master.basin_id.nunique()),
                "unique_basins": int(parameter_master.basin_id.nunique()),
                "expected_parameters": len(parameter_names),
                "actual_parameters": int(parameter_master.parameter.nunique()),
                "expected_cells": N_BASINS * len(parameter_names),
                "actual_cells": len(parameter_master),
                "missing_cells": N_BASINS * len(parameter_names) - len(parameter_master),
                "duplicate_cells": duplicate_cells,
                "nan_cells": int(parameter_master.isna().any(axis=1).sum()),
                "inf_cells": int(np.isinf(parameter_master.select_dtypes(include=[np.number]).to_numpy()).sum()),
            })
            seen_kge_model = inputs["seen_kge"][inputs["seen_kge"].model == model]
            oob_kge: dict[int, float] = {}
            fold_medians: list[float] = []
            for fold, item in fold_data.items():
                kge_frame = read_csv(FORMAL / "runs" / model / f"fold_{fold}" / "heldout_basin_kge.csv")
                fold_values = kge_frame.kge.to_numpy(dtype=np.float64)
                fold_medians.append(float(np.median(fold_values)))
                for row in kge_frame.itertuples(index=False):
                    oob_kge[int(row.basin_id)] = float(row.kge)
            performance_master = make_performance_master(model, inputs["basin_ids"], fold_for_basin, seen_kge_model, oob_kge)
            performance_writer.append(performance_master)
            viability_rows.append({
                "model": model,
                "seen_dpl_median_kge": float(np.median(performance_master.KGE_dPL_seen)),
                "seen_dpl_q25_kge": float(np.quantile(performance_master.KGE_dPL_seen, 0.25)),
                "seen_dpl_q75_kge": float(np.quantile(performance_master.KGE_dPL_seen, 0.75)),
                "oob_pooled_median_kge": float(np.median(performance_master.KGE_dPL_OOB)),
                "oob_pooled_mean_kge": float(np.mean(performance_master.KGE_dPL_OOB)),
                "oob_q25_kge": float(np.quantile(performance_master.KGE_dPL_OOB, 0.25)),
                "oob_q75_kge": float(np.quantile(performance_master.KGE_dPL_OOB, 0.75)),
                "delta_median_kge_oob_minus_seen": float(np.median(performance_master.delta_KGE)),
                "delta_mean_kge_oob_minus_seen": float(np.mean(performance_master.delta_KGE)),
                "five_fold_median_kge": float(np.median(fold_medians)),
                "five_fold_min_kge": float(np.min(fold_medians)),
                "five_fold_max_kge": float(np.max(fold_medians)),
                "five_fold_iqr_kge": float(np.quantile(fold_medians, 0.75) - np.quantile(fold_medians, 0.25)),
            })

            seen_relationships = build_seen_relationship_map(inputs, model, parameter_names, inputs["attributes"])
            rho_oob = correlation_matrix(inputs["raw_attr"], oob_u, device)
            rho_folds = {fold: correlation_matrix(inputs["raw_attr"][[int(np.where(inputs["basin_ids"] == basin_id)[0][0]) for basin_id in item["heldout_ids"]]], item["u"], device) for fold, item in fold_data.items()}
            parameter_flags = [(boundary_summary(seen["ic_u"][:, p]), boundary_summary(seen["seen_u"][:, p]), boundary_summary(oob_u[:, p])) for p in range(len(parameter_names))]
            for p, parameter in enumerate(parameter_names):
                for a, attribute in enumerate(inputs["attributes"]):
                    source = seen_relationships.get((p, attribute), {})
                    rho_ic = float(source.get("rho_ic", np.nan))
                    rho_seen = float(source.get("rho_dpl_seen", np.nan))
                    rho_oob_value = float(rho_oob[a, p])
                    fold_values = [float(rho_folds[fold][a, p]) for fold in range(N_FOLDS)]
                    same_seen = sum(1 for value in fold_values if finite(value) and finite(rho_seen) and sign(value) == sign(rho_seen) and sign(value) != 0)
                    same_ic = sum(1 for value in fold_values if finite(value) and finite(rho_ic) and sign(value) == sign(rho_ic) and sign(value) != 0)
                    nontriv_seen = sum(1 for value in fold_values if finite(value) and finite(rho_seen) and abs(value) >= NONTRIVIAL and abs(rho_seen) >= NONTRIVIAL and sign(value) == sign(rho_seen))
                    nontriv_ic = sum(1 for value in fold_values if finite(value) and finite(rho_ic) and abs(value) >= NONTRIVIAL and abs(rho_ic) >= NONTRIVIAL and sign(value) == sign(rho_ic))
                    cluster = inputs["clusters"].get(attribute, {"cluster_id": f"SINGLETON_{attribute}", "representative": attribute, "members": attribute})
                    ic_flag, seen_flag, oob_flag = parameter_flags[p]
                    relationship_rows.append({
                        "model": model,
                        "parameter_index": p,
                        "parameter": parameter,
                        "attribute": attribute,
                        "attribute_type": "CONTINUOUS",
                        "information_cluster": cluster["cluster_id"],
                        "cluster_representative": cluster["representative"],
                        "rho_ic": rho_ic,
                        "rho_dpl_seen": rho_seen,
                        "rho_dpl_oob": rho_oob_value,
                        "abs_rho_ic": abs(rho_ic) if finite(rho_ic) else np.nan,
                        "abs_rho_dpl_seen": abs(rho_seen) if finite(rho_seen) else np.nan,
                        "abs_rho_dpl_oob": abs(rho_oob_value) if finite(rho_oob_value) else np.nan,
                        "strength_change_seen_oob": (abs(rho_oob_value) - abs(rho_seen)) if finite(rho_oob_value) and finite(rho_seen) else np.nan,
                        "strength_change_ic_oob": (abs(rho_oob_value) - abs(rho_ic)) if finite(rho_oob_value) and finite(rho_ic) else np.nan,
                        "coefficient_distance_seen_oob": abs(rho_oob_value - rho_seen) if finite(rho_oob_value) and finite(rho_seen) else np.nan,
                        "coefficient_distance_ic_oob": abs(rho_oob_value - rho_ic) if finite(rho_oob_value) and finite(rho_ic) else np.nan,
                        "sign_oob_vs_seen": sign(rho_oob_value) == sign(rho_seen) if sign(rho_oob_value) != 0 and sign(rho_seen) != 0 else np.nan,
                        "sign_oob_vs_ic": sign(rho_oob_value) == sign(rho_ic) if sign(rho_oob_value) != 0 and sign(rho_ic) != 0 else np.nan,
                        "nontrivial_oob_vs_seen": bool(finite(rho_oob_value) and finite(rho_seen) and abs(rho_oob_value) >= NONTRIVIAL and abs(rho_seen) >= NONTRIVIAL),
                        "nontrivial_oob_vs_ic": bool(finite(rho_oob_value) and finite(rho_ic) and abs(rho_oob_value) >= NONTRIVIAL and abs(rho_ic) >= NONTRIVIAL),
                        "relationship_class_seen": source.get("relationship_class", "weak/unresolved"),
                        "rank_ic_seen": int(source.get("rank_ic", 9999)),
                        "rank_dpl_seen": int(source.get("rank_seen", 9999)),
                        "rho_fold1": fold_values[0], "rho_fold2": fold_values[1], "rho_fold3": fold_values[2], "rho_fold4": fold_values[3], "rho_fold5": fold_values[4],
                        "n_basins_fold1": int(fold_data[0]["heldout_ids"].size), "n_basins_fold2": int(fold_data[1]["heldout_ids"].size), "n_basins_fold3": int(fold_data[2]["heldout_ids"].size), "n_basins_fold4": int(fold_data[3]["heldout_ids"].size), "n_basins_fold5": int(fold_data[4]["heldout_ids"].size),
                        "fold_median": float(np.nanmedian(fold_values)),
                        "fold_iqr": float(np.nanpercentile(fold_values, 75) - np.nanpercentile(fold_values, 25)),
                        "fold_min": float(np.nanmin(fold_values)),
                        "fold_max": float(np.nanmax(fold_values)),
                        "same_sign_folds_vs_seen": same_seen,
                        "same_sign_folds_vs_ic": same_ic,
                        "nontrivial_same_sign_folds_vs_seen": nontriv_seen,
                        "nontrivial_same_sign_folds_vs_ic": nontriv_ic,
                        "boundary_concentrated_ic": ic_flag["boundary_concentrated"], "boundary_concentrated_seen": seen_flag["boundary_concentrated"], "boundary_concentrated_oob": oob_flag["boundary_concentrated"],
                        "near_constant_ic": ic_flag["near_constant"], "near_constant_seen": seen_flag["near_constant"], "near_constant_oob": oob_flag["near_constant"],
                        "tie_fraction_ic": ic_flag["tie_fraction"], "tie_fraction_seen": seen_flag["tie_fraction"], "tie_fraction_oob": oob_flag["tie_fraction"],
                    })
            boot_ci, boot_meta = bootstrap_spearman(inputs["raw_attr"], oob_u, device, model_index=model_index)
            for a, attribute in enumerate(inputs["attributes"]):
                for p, parameter in enumerate(parameter_names):
                    bootstrap_rows.append({
                        "model": model, "parameter_index": p, "parameter": parameter, "attribute": attribute,
                        "rho_dpl_oob": float(rho_oob[a, p]), "ci95_low": float(boot_ci[0, a, p]), "ci95_high": float(boot_ci[1, a, p]),
                        "n_boot": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_unit": "531 basin rows", **boot_meta,
                    })
            del seen, fold_data, oob_u, oob_physical, parameter_master, performance_master, rho_oob, rho_folds, seen_relationships, boot_ci
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            per_model_summary.append({"model": model, "runtime_seconds": time.perf_counter() - model_started})
    finally:
        parameter_writer.close()
        performance_writer.close()

    qc = pd.DataFrame(qc_rows)
    write_csv(R4 / "tables/R4_OOB_MASTER_QC.csv", qc)
    if (qc.missing_cells > 0).any() or (qc.duplicate_cells > 0).any() or (qc.nan_cells > 0).any() or (qc.inf_cells > 0).any():
        raise RuntimeError("R4 master table QC failed; scientific interpretation is stopped")
    viability = pd.DataFrame(viability_rows)
    write_csv(R4 / "tables/R4_OOB_KGE_VIABILITY.csv", viability)
    write_csv(R4 / "figure_data/R4_PANEL_A_KGE_VIABILITY.csv", viability)
    relationships = pd.DataFrame(relationship_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    write_csv(R4 / "tables/R4_POPULATION_RELATIONSHIP_PERSISTENCE.csv", relationships)
    write_csv(R4 / "tables/R4_FOLDWISE_RELATIONSHIP_PERSISTENCE.csv", relationships)
    write_csv(R4 / "tables/R4_OOB_RELATIONSHIP_BOOTSTRAP_CI.csv", bootstrap)
    panel_b = relationships[np.isfinite(relationships.rho_dpl_seen) & np.isfinite(relationships.rho_dpl_oob)].copy()
    panel_b["behavior_class"] = panel_b.relationship_class_seen
    write_csv(R4 / "figure_data/R4_PANEL_B_SEEN_VS_OOB.csv", panel_b)

    summary_rows: list[dict[str, Any]] = []
    overall = population_metrics(relationships)
    summary_rows.append({"summary_level": "overall", "model": "ALL", "parameter": "ALL", **overall})
    for model in PRIMARY8:
        metrics = population_metrics(relationships[relationships.model == model])
        summary_rows.append({"summary_level": "model", "model": model, "parameter": "ALL", **metrics})
    for parameter in sorted(relationships.parameter.unique()):
        metrics = population_metrics(relationships[relationships.parameter == parameter])
        summary_rows.append({"summary_level": "parameter", "model": "ALL", "parameter": parameter, **metrics})
    population_summary = pd.DataFrame(summary_rows)
    write_csv(R4 / "tables/R4_POPULATION_SUMMARY.csv", population_summary)

    # Information-cluster retention is a secondary layer over non-weak seen relationships.
    source = relationships[(relationships.relationship_class_seen != "weak/unresolved") & np.isfinite(relationships.rho_dpl_seen)].copy()
    cluster_records: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        members = relationships[(relationships.model == row.model) & (relationships.parameter == row.parameter) & (relationships.information_cluster == row.information_cluster)]
        same_direction = members[np.isfinite(members.rho_dpl_oob) & (members.rho_dpl_oob.abs() >= NONTRIVIAL) & (members.rho_dpl_oob.apply(sign) == sign(row.rho_dpl_seen))]
        best = same_direction.iloc[np.argmax(same_direction.rho_dpl_oob.abs().to_numpy())] if len(same_direction) else None
        raw_retained = bool(finite(row.rho_dpl_oob) and abs(row.rho_dpl_oob) >= NONTRIVIAL and sign(row.rho_dpl_oob) == sign(row.rho_dpl_seen))
        cluster_retained = best is not None
        fold_consistency = 0
        if best is not None:
            fold_consistency = sum(1 for fold in range(1, 6) if finite(best[f"rho_fold{fold}"]) and abs(best[f"rho_fold{fold}"]) >= NONTRIVIAL and sign(best[f"rho_fold{fold}"]) == sign(row.rho_dpl_seen))
        cluster_records.append({
            "summary_level": "relationship",
            "model": row.model, "parameter": row.parameter, "parameter_index": row.parameter_index,
            "raw_attribute": row.attribute, "information_cluster": row.information_cluster,
            "cluster_representative": row.cluster_representative, "seen_relationship_class": row.relationship_class_seen,
            "rho_seen": row.rho_dpl_seen, "rho_oob_raw": row.rho_dpl_oob,
            "cluster_oob_attribute": best.attribute if best is not None else "",
            "cluster_oob_rho": float(best.rho_dpl_oob) if best is not None else np.nan,
            "raw_retained": raw_retained, "cluster_retained": cluster_retained,
            "proxy_substitution_rescued": bool(cluster_retained and not raw_retained),
            "information_dimension_failure": bool(not cluster_retained),
            "cluster_same_sign_folds": fold_consistency,
        })
    cluster_detail = pd.DataFrame(cluster_records)
    cluster_summary_rows: list[dict[str, Any]] = []
    for cluster, frame in cluster_detail.groupby("information_cluster", sort=True):
        cluster_summary_rows.append({
            "summary_level": "cluster", "information_cluster": cluster,
            "cluster_representative": frame.cluster_representative.iloc[0],
            "n_seen_relationships": len(frame), "raw_retained_count": int(frame.raw_retained.sum()), "cluster_retained_count": int(frame.cluster_retained.sum()), "proxy_substitution_rescued_count": int(frame.proxy_substitution_rescued.sum()), "dimension_failure_count": int(frame.information_dimension_failure.sum()),
            "raw_retention_rate": float(frame.raw_retained.mean()), "cluster_retention_rate": float(frame.cluster_retained.mean()), "proxy_substitution_rescued_rate": float(frame.proxy_substitution_rescued.mean()), "cluster_fold_consistency_mean": float(frame.cluster_same_sign_folds.mean() / 5.0),
        })
    if len(cluster_detail):
        cluster_summary_rows.append({
            "summary_level": "overall", "information_cluster": "ALL", "cluster_representative": "ALL",
            "n_seen_relationships": len(cluster_detail), "raw_retained_count": int(cluster_detail.raw_retained.sum()), "cluster_retained_count": int(cluster_detail.cluster_retained.sum()), "proxy_substitution_rescued_count": int(cluster_detail.proxy_substitution_rescued.sum()), "dimension_failure_count": int(cluster_detail.information_dimension_failure.sum()),
            "raw_retention_rate": float(cluster_detail.raw_retained.mean()), "cluster_retention_rate": float(cluster_detail.cluster_retained.mean()), "proxy_substitution_rescued_rate": float(cluster_detail.proxy_substitution_rescued.mean()), "cluster_fold_consistency_mean": float(cluster_detail.cluster_same_sign_folds.mean() / 5.0),
        })
    cluster_table = pd.DataFrame(cluster_summary_rows)
    write_csv(R4 / "tables/R4_INFORMATION_CLUSTER_RETENTION.csv", cluster_table)
    panel_d = cluster_table[cluster_table.summary_level == "cluster"].copy()
    write_csv(R4 / "figure_data/R4_PANEL_D_CLUSTER_RETENTION.csv", panel_d)

    # Boundary/tie diagnostics and the explicit population sensitivity rerun.
    sensitivity_rows: list[dict[str, Any]] = []
    for row in relationships.itertuples(index=False):
        sensitivity_rows.append({
            "record_type": "parameter_diagnostic", "model": row.model, "parameter": row.parameter, "parameter_index": row.parameter_index,
            "boundary_concentrated_ic": row.boundary_concentrated_ic, "boundary_concentrated_seen": row.boundary_concentrated_seen, "boundary_concentrated_oob": row.boundary_concentrated_oob,
            "near_constant_ic": row.near_constant_ic, "near_constant_seen": row.near_constant_seen, "near_constant_oob": row.near_constant_oob,
            "tie_fraction_ic": row.tie_fraction_ic, "tie_fraction_seen": row.tie_fraction_seen, "tie_fraction_oob": row.tie_fraction_oob,
        })
    diagnostics = relationships[["model", "parameter", "parameter_index", "boundary_concentrated_ic", "boundary_concentrated_seen", "boundary_concentrated_oob", "near_constant_ic", "near_constant_seen", "near_constant_oob"]].drop_duplicates()
    relationships["saturation_flag"] = relationships.boundary_concentrated_ic | relationships.boundary_concentrated_seen | relationships.boundary_concentrated_oob | relationships.near_constant_ic | relationships.near_constant_seen | relationships.near_constant_oob
    flagged_parameter_cells = int((diagnostics[["boundary_concentrated_ic", "boundary_concentrated_seen", "boundary_concentrated_oob", "near_constant_ic", "near_constant_seen", "near_constant_oob"]].any(axis=1)).sum())
    for scope, frame in (("all_primary_cells", relationships), ("exclude_high_saturation_near_constant", relationships[~relationships.saturation_flag])):
        metrics = population_metrics(frame)
        sensitivity_rows.append({"record_type": "population_summary", "analysis_scope": scope, **metrics, "n_flagged_parameter_cells": flagged_parameter_cells})
    sensitivity = pd.DataFrame(sensitivity_rows)
    write_csv(R4 / "si/R4_BOUNDARY_TIE_SENSITIVITY.csv", sensitivity)
    all_metrics = population_metrics(relationships)
    filtered_metrics = population_metrics(relationships[~relationships.saturation_flag])
    (R4 / "si/R4_BOUNDARY_TIE_SUMMARY.md").write_text(
        "# R4 Boundary/Tie Sensitivity\n\n"
        "The primary relationship population retains all cells and only flags near-constant/boundary-concentrated parameter groups. The sensitivity row excludes flagged groups after the primary calculation.\n\n"
        f"- Primary finite OOB cells: {all_metrics['n_oob_finite']}\n"
        f"- Sensitivity finite OOB cells: {filtered_metrics['n_oob_finite']}\n"
        f"- Primary seen→OOB nontrivial sign retention: {all_metrics['nontrivial_sign_retention_vs_seen']:.4f}\n"
        f"- Sensitivity seen→OOB nontrivial sign retention: {filtered_metrics['nontrivial_sign_retention_vs_seen']:.4f}\n"
        f"- Primary seen→OOB correspondence: {all_metrics['seen_oob_correspondence_spearman']:.4f}\n"
        f"- Sensitivity seen→OOB correspondence: {filtered_metrics['seen_oob_correspondence_spearman']:.4f}\n",
        encoding="utf-8",
    )

    # Frozen cases are joined only after all population calculations are complete.
    case_records: list[dict[str, Any]] = []
    for row in frozen.itertuples(index=False):
        match = relationships[(relationships.model == row.model) & (relationships.parameter == row.parameter) & (relationships.attribute == row.attribute)]
        if len(match) != 1:
            raise RuntimeError(f"frozen case did not resolve uniquely: {row.model}/{row.parameter}/{row.attribute}")
        rel = match.iloc[0]
        ci = bootstrap[(bootstrap.model == row.model) & (bootstrap.parameter == row.parameter) & (bootstrap.attribute == row.attribute)].iloc[0]
        case_records.append({
            "model": row.model, "parameter": row.parameter, "attribute": row.attribute, "information_cluster": rel.information_cluster,
            "R3_behavior_class": row.selection_category, "relationship_class_seen": row.relationship_class,
            "rho_IC": rel.rho_ic, "rho_dPL_seen": rel.rho_dpl_seen, "rho_dPL_OOB": rel.rho_dpl_oob,
            "OOB_CI_low": ci.ci95_low, "OOB_CI_high": ci.ci95_high,
            "rho_fold1": rel.rho_fold1, "rho_fold2": rel.rho_fold2, "rho_fold3": rel.rho_fold3, "rho_fold4": rel.rho_fold4, "rho_fold5": rel.rho_fold5,
            "same_sign_folds_vs_seen": rel.same_sign_folds_vs_seen, "same_sign_folds_vs_IC": rel.same_sign_folds_vs_ic,
            "nontrivial_same_sign_folds_vs_seen": rel.nontrivial_same_sign_folds_vs_seen, "nontrivial_same_sign_folds_vs_IC": rel.nontrivial_same_sign_folds_vs_ic,
            "boundary_concentrated_ic": rel.boundary_concentrated_ic, "boundary_concentrated_seen": rel.boundary_concentrated_seen, "boundary_concentrated_oob": rel.boundary_concentrated_oob,
            "near_constant_ic": rel.near_constant_ic, "near_constant_seen": rel.near_constant_seen, "near_constant_oob": rel.near_constant_oob,
            "tie_fraction_ic": rel.tie_fraction_ic, "tie_fraction_seen": rel.tie_fraction_seen, "tie_fraction_oob": rel.tie_fraction_oob,
        })
    cases = pd.DataFrame(case_records)
    write_csv(R4 / "tables/R4_FROZEN_CASE_RESULTS.csv", cases)
    write_csv(R4 / "figure_data/R4_PANEL_C_FROZEN_CASES.csv", cases)

    # GPU/provenance log and concise report scaffolding.
    gpu_log["free_vram_end_bytes"] = int(torch.cuda.mem_get_info(device)[0])
    gpu_log["peak_gpu_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    gpu_log["peak_gpu_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    gpu_log["peak_rss_mb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    gpu_log["runtime_seconds"] = time.perf_counter() - started
    gpu_log["per_model_runtime_seconds"] = per_model_summary
    (R4 / "logs/R4_GPU_EXECUTION_LOG.json").write_text(json.dumps(gpu_log, indent=2) + "\n", encoding="utf-8")
    write_json = lambda path, obj: path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    write_json(R4 / "logs/R4_ANALYSIS_METADATA.json", {"formal_oob_root": str(FORMAL), "frozen_manifest": str(frozen_path), "frozen_manifest_sha256": frozen_sha, "models": list(PRIMARY8), "bootstrap_n": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED, "no_training": True, "no_oob_case_selection": True, "outputs": {"parameter_master": str(R4 / "tables/R4_OOB_PARAMETER_MASTER.parquet"), "performance_master": str(R4 / "tables/R4_OOB_PERFORMANCE_MASTER.parquet")}})
    overall = population_summary[(population_summary.summary_level == "overall")].iloc[0].to_dict()
    cluster_overall = cluster_table[cluster_table.summary_level == "overall"].iloc[0].to_dict() if len(cluster_table[cluster_table.summary_level == "overall"]) else {}
    if float(overall["nontrivial_sign_retention_vs_seen"]) >= 0.80 and float(overall["seen_oob_correspondence_spearman"]) >= 0.70 and float(overall["fraction_ge4of5_same_sign_vs_seen"]) >= 0.70:
        tier = "A. broadly retained"
    elif float(overall["nontrivial_sign_retention_vs_seen"]) < 0.50 and float(overall["seen_oob_correspondence_spearman"]) < 0.40:
        tier = "C. weakly retained / clear transfer boundary"
    else:
        tier = "B. partially retained / heterogeneous"
    def fmt(value: Any) -> str:
        return "NA" if not finite(value) else f"{float(value):.4f}"
    (R4 / "R4_STATISTICAL_ANALYSIS_REPORT.md").write_text(
        f"""# R4 Statistical Analysis Report

## Direct answers

1. **Predictive viability:** OOB held-out KGE is reported as a viability control in `tables/R4_OOB_KGE_VIABILITY.csv`; it is not used to rank or select models.
2. **Population persistence:** across the retained PRIMARY8 relationship cells, seen→OOB nontrivial sign retention is **{fmt(overall['nontrivial_sign_retention_vs_seen'])}**, seen→OOB coefficient correspondence is **{fmt(overall['seen_oob_correspondence_spearman'])}**, and median absolute coefficient distance is **{fmt(overall['median_abs_coefficient_distance_seen_oob'])}**.
3. **Fold consistency:** fraction of finite relationships with at least 4/5 same-sign folds versus seen is **{fmt(overall['fraction_ge4of5_same_sign_vs_seen'])}**; median fold spread is **{fmt(overall['median_fold_spread'])}**.
4. **Information dimensions:** raw-level retention is **{fmt(cluster_overall.get('raw_retention_rate'))}**, cluster-level retention is **{fmt(cluster_overall.get('cluster_retention_rate'))}**, and proxy-substitution rescue is **{fmt(cluster_overall.get('proxy_substitution_rescued_rate'))}** among non-weak seen relationships.

## Current R4 judgment

**{tier}**. This is a descriptive held-out-basin challenge result. It does not ask which model transfers best; it asks whether structure-dependent catchment–parameter relationships identified under IC and seen-basin dPL persist when dPL generates parameters for genuinely held-out basins.

## Scope and safeguards

- Formal OOB source: `{FORMAL}`; 40/40 jobs and 531/531 OOF coverage per model were audited before analysis.
- Frozen case manifest: `{frozen_path}`; SHA256 `{frozen_sha}`.
- OOB coefficients were read only after the manifest was frozen.
- Primary population cells retain boundary/tie-flagged relationships. Optional exclusion is reported only in `si/R4_BOUNDARY_TIE_SENSITIVITY.csv`.
- `KGE_IC` and `KGE_dPL_seen` are not recomputed here; IC was not rerun. No PUR, multi-seed, model ranking, or training was performed.
- Bootstrap: 5,000 basin resamples, seed {BOOTSTRAP_SEED}, percentile 95% CIs, CUDA batch processing.

## Outputs

See `tables/`, `figure_data/`, `si/`, and `logs/`. The primary figure consumes only the four panel-data CSVs and is generated by `scripts/r4_make_figure.py`.
""",
        encoding="utf-8",
    )
    (R4 / "R4_FIGURE_PLAN.md").write_text(
        """# R4 Figure Plan

One four-panel figure, generated only after panel-data CSVs exist:

- **A:** seen dPL KGE versus OOB dPL KGE as a predictive viability control; no ranking language.
- **B:** all PRIMARY8 continuous relationship cells, x=`rho_dPL_seen`, y=`rho_dPL_OOB`, with 1:1 and zero reference lines.
- **C:** the four frozen cases with IC, seen dPL, pooled OOB 95% CI, and five fold values.
- **D:** raw-proxy retention versus information-cluster retention; proxy substitution is shown as rescue, not failure.

No additional main-text panel is added. Raw tables and boundary/tie sensitivity remain in the SI outputs.
""",
        encoding="utf-8",
    )
    (R4 / "R4_HANDOFF.md").write_text(
        f"""# R4 Handoff

- Analysis status: Steps 5–12 complete; no training was launched.
- Frozen manifest SHA256: `{frozen_sha}`.
- Master QC: PASS for {len(qc_rows)}/8 models.
- Bootstrap device: `{gpu_log['gpu_name']}`; peak allocated VRAM `{gpu_log['peak_gpu_memory_allocated_bytes']}` bytes; peak RSS `{gpu_log['peak_rss_mb']:.1f}` MiB.
- Current three-tier judgment: **{tier}**.
- Run `python project/benchmark/manuscript/r4/scripts/r4_make_figure.py` to render the single planned figure from frozen figure-data CSVs.
- `KGE_IC`/`KGE_dPL_seen` comparison columns remain blank by design because IC is not rerun in R4.
""",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "tier": tier, "overall": overall, "cluster_overall": cluster_overall, "gpu": gpu_log, "outputs": str(R4)}, indent=2, default=str))


if __name__ == "__main__":
    main()
