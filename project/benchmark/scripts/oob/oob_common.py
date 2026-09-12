#!/usr/bin/env python3
"""Shared contracts for the local PRIMARY-8 basin-held-out OOB experiment.

The important boundary in this module is basin ownership, not just time.  The
outer held-out basins are never part of the training tensors, training KGE,
window catalog, plateau counter, or checkpoint selection.  Their observations
are loaded only after ``Phase.EVAL`` is entered.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(REPO_ROOT / "dmotpy")]

PRIMARY8 = (
    "alpine2",
    "hbv96",
    "xinanjiang",
    "newzealand2",
    "ihacres",
    "us1",
    "mopex4",
    "hillslope",
)
N_FOLDS = 5
FOLD_SEED = 20260902
EXPECTED_BASINS = 531
N_ATTRIBUTES = 35
SKEWED_ATTRIBUTE_INDICES = (11, 23, 34)
KGE_EPS = 0.1


class Phase(str, Enum):
    PREPARE = "prepare"
    TRAIN = "train"
    EVAL = "eval"


CURRENT_PHASE = Phase.TRAIN


class LeakageError(RuntimeError):
    """Raised when an outer held-out target crosses the training boundary."""


def set_phase(phase: Phase) -> None:
    global CURRENT_PHASE
    CURRENT_PHASE = Phase(phase)


def require_phase(expected: Phase, operation: str) -> None:
    if CURRENT_PHASE != expected:
        raise LeakageError(
            f"{operation} is only allowed during {expected.value} phase; "
            f"current phase is {CURRENT_PHASE.value}"
        )


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "configs":
        return BENCHMARK_ROOT / candidate
    return BENCHMARK_ROOT / "configs" / candidate


def load_oob_config(path: str | Path) -> dict[str, Any]:
    """Load YAML inheritance without changing the repository's config files."""

    config_path = resolve_config_path(path).resolve()

    def load_one(current: Path) -> dict[str, Any]:
        with current.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        parent = raw.get("extends")
        if parent:
            parent_path = (current.parent / parent).resolve()
            base = load_one(parent_path)
        else:
            base = {}
        return deep_merge(base, {key: value for key, value in raw.items() if key != "extends"})

    resolved = load_one(config_path)
    resolved["_resolved_from"] = str(config_path)
    return resolved


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_ids(path: str | Path) -> np.ndarray:
    resolved = resolve_repo_path(path)
    if resolved.suffix == ".npy":
        values = np.load(resolved, allow_pickle=False)
    else:
        text = resolved.read_text(encoding="utf-8").strip()
        try:
            values = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            values = [line.strip() for line in text.splitlines() if line.strip()]
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    if result.size == 0:
        raise ValueError(f"basin ID file is empty: {resolved}")
    return result


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class FoldContract:
    basin_count: int
    n_folds: int
    fold_sizes: tuple[int, ...]
    assignment_sha256: str | None = None


def make_fold_assignment(
    basin_ids: Sequence[int] | np.ndarray,
    *,
    n_folds: int = N_FOLDS,
    random_state: int = FOLD_SEED,
) -> list[dict[str, int]]:
    """Return sklearn-KFold-compatible shuffled fold rows.

    ``RandomState.shuffle`` plus contiguous fold-size allocation is the exact
    mechanics used by ``KFold(n_splits=5, shuffle=True, random_state=...)``.
    Keeping the implementation local makes the frozen assignment auditable and
    avoids depending on sklearn at experiment runtime.
    """

    ids = np.asarray(basin_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.unique(ids).size != ids.size:
        raise ValueError("fold input must contain non-empty unique basin IDs")
    if n_folds < 2 or n_folds > ids.size:
        raise ValueError("n_folds must be between 2 and the number of basins")

    order = np.arange(ids.size, dtype=np.int64)
    np.random.RandomState(random_state).shuffle(order)
    fold_sizes = np.full(n_folds, ids.size // n_folds, dtype=np.int64)
    fold_sizes[: ids.size % n_folds] += 1
    assignment = np.empty(ids.size, dtype=np.int64)
    cursor = 0
    for fold, size in enumerate(fold_sizes):
        selected = order[cursor : cursor + int(size)]
        assignment[selected] = fold
        cursor += int(size)
    return [
        {"basin_id": int(basin_id), "fold": int(fold)}
        for basin_id, fold in zip(ids.tolist(), assignment.tolist())
    ]


def validate_fold_contract(
    basin_ids: Sequence[int] | np.ndarray,
    rows: Iterable[Mapping[str, Any]],
    *,
    n_folds: int = N_FOLDS,
    expected_count: int | None = EXPECTED_BASINS,
) -> FoldContract:
    ids = np.asarray(basin_ids, dtype=np.int64).reshape(-1)
    table = [(int(row["basin_id"]), int(row["fold"])) for row in rows]
    if expected_count is not None and ids.size != expected_count:
        raise ValueError(f"expected {expected_count} basins, found {ids.size}")
    if ids.size == 0 or np.unique(ids).size != ids.size:
        raise ValueError("configured basin IDs must be unique and non-empty")
    if len(table) != ids.size:
        raise ValueError(f"assignment has {len(table)} rows for {ids.size} configured basins")
    assigned = [basin_id for basin_id, _ in table]
    if len(set(assigned)) != len(assigned) or set(assigned) != set(map(int, ids)):
        raise ValueError("fold assignment basin IDs must match the configured basin list exactly")
    expected_rows = make_fold_assignment(ids, n_folds=n_folds, random_state=FOLD_SEED)
    expected_assignment = {int(row["basin_id"]): int(row["fold"]) for row in expected_rows}
    actual_assignment = {basin_id: fold for basin_id, fold in table}
    if actual_assignment != expected_assignment:
        raise ValueError("fold assignment does not match frozen KFold random_state=20260902")
    if any(fold < 0 or fold >= n_folds for _, fold in table):
        raise ValueError("fold labels are outside the configured range")

    by_fold = {fold: {basin_id for basin_id, value in table if value == fold} for fold in range(n_folds)}
    if set().union(*by_fold.values()) != set(map(int, ids)):
        raise ValueError("held-out fold union does not cover all basins")
    for left in range(n_folds):
        for right in range(left + 1, n_folds):
            if by_fold[left].intersection(by_fold[right]):
                raise ValueError(f"held-out folds {left} and {right} overlap")
    sizes = tuple(len(by_fold[fold]) for fold in range(n_folds))
    expected_sizes = tuple([ids.size // n_folds + (1 if fold < ids.size % n_folds else 0) for fold in range(n_folds)])
    if sizes != expected_sizes:
        raise ValueError(f"fold sizes {sizes} do not match deterministic KFold sizes {expected_sizes}")
    for fold in range(n_folds):
        held = by_fold[fold]
        train = set(map(int, ids)).difference(held)
        if train.intersection(held):
            raise ValueError(f"train/held-out overlap in fold {fold}")
    return FoldContract(int(ids.size), n_folds, sizes)


def read_fold_assignment(path: str | Path) -> list[dict[str, int]]:
    resolved = Path(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        return [{"basin_id": int(row["basin_id"]), "fold": int(row["fold"])} for row in csv.DictReader(handle)]


def write_fold_assignment(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["basin_id", "fold"])
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(destination)


def fold_ids(
    basin_ids: Sequence[int] | np.ndarray,
    rows: Iterable[Mapping[str, Any]],
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(basin_ids, dtype=np.int64).reshape(-1)
    assignment = {int(row["basin_id"]): int(row["fold"]) for row in rows}
    held = np.asarray(sorted(basin_id for basin_id in ids if assignment[int(basin_id)] == fold), dtype=np.int64)
    train = np.asarray(sorted(basin_id for basin_id in ids if assignment[int(basin_id)] != fold), dtype=np.int64)
    if not held.size or not train.size or set(map(int, held)).intersection(map(int, train)):
        raise ValueError(f"invalid train/held-out split for fold {fold}")
    return train, held


@dataclass(frozen=True)
class NormalizationStats:
    """Train-fitted transform; no held-out values enter fit or shift selection."""

    method: str
    log_transform_skewed: bool
    shifts: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(
        cls,
        raw_train: np.ndarray,
        *,
        method: str = "zscore",
        log_transform_skewed: bool = True,
    ) -> "NormalizationStats":
        prepared, shifts = _prepare_attributes(raw_train, log_transform_skewed=log_transform_skewed)
        if method == "zscore":
            mean = np.mean(prepared, axis=0)
            std = np.std(prepared, axis=0) + 1.0e-6
        elif method == "minmax":
            mean = np.min(prepared, axis=0)
            std = np.max(prepared, axis=0) - mean + 1.0e-6
        elif method == "none":
            mean = np.zeros(prepared.shape[1], dtype=np.float64)
            std = np.ones(prepared.shape[1], dtype=np.float64)
        else:
            raise ValueError(f"unsupported attribute normalization method: {method}")
        return cls(method, bool(log_transform_skewed), shifts, mean, std)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        prepared, _ = _prepare_attributes(
            raw,
            log_transform_skewed=self.log_transform_skewed,
            shifts=self.shifts,
        )
        transformed = (prepared - self.mean) / self.std
        if not np.isfinite(transformed).all():
            raise FloatingPointError("non-finite normalized attributes")
        return transformed.astype(np.float64, copy=False)

    def save(self, run_dir: str | Path) -> None:
        destination = Path(run_dir)
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "attribute_mean.npy", self.mean)
        np.save(destination / "attribute_std.npy", self.std)
        (destination / "normalization_metadata.json").write_text(
            json.dumps(
                {
                    "method": self.method,
                    "log_transform_skewed": self.log_transform_skewed,
                    "skewed_attribute_indices": list(SKEWED_ATTRIBUTE_INDICES),
                    "train_fitted_log_shifts": self.shifts.tolist(),
                    "scope": "train_basins_only",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _prepare_attributes(
    raw: np.ndarray,
    *,
    log_transform_skewed: bool,
    shifts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_ATTRIBUTES:
        raise ValueError(f"expected raw attributes with shape [basin, {N_ATTRIBUTES}], got {values.shape}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    fitted_shifts = np.ones(N_ATTRIBUTES, dtype=np.float64) if shifts is None else np.asarray(shifts, dtype=np.float64).copy()
    if fitted_shifts.shape != (N_ATTRIBUTES,):
        raise ValueError("attribute shifts must have one value per attribute")
    if log_transform_skewed:
        if shifts is None:
            for index in SKEWED_ATTRIBUTE_INDICES:
                minimum = float(np.min(values[:, index]))
                fitted_shifts[index] = abs(minimum) + 1.0 if minimum < 0.0 else 1.0
        for index in SKEWED_ATTRIBUTE_INDICES:
            argument = values[:, index] + fitted_shifts[index] - 1.0
            if np.any(argument < 0.0):
                raise ValueError("held-out attribute is outside the train-fitted log transform domain")
            values[:, index] = np.log(argument + 1.0)
    return values, fitted_shifts


class TargetAccessPolicy:
    """Explicit target ownership policy passed to each data-loading operation."""

    def __init__(self, phase: Phase, allowed_basin_ids: Iterable[int]):
        self.phase = Phase(phase)
        self.allowed_basin_ids = frozenset(int(value) for value in allowed_basin_ids)

    def authorize(self, basin_ids: Iterable[int], operation: str) -> None:
        requested = {int(value) for value in basin_ids}
        forbidden = requested.difference(self.allowed_basin_ids)
        if forbidden:
            raise LeakageError(
                f"{operation} requested target observations for unauthorized basins: "
                f"{sorted(forbidden)[:5]}"
            )


@dataclass(frozen=True)
class PeriodData:
    x: torch.Tensor
    y: torch.Tensor
    basin_ids: np.ndarray
    dates: pd.DatetimeIndex
    warmup_days: int = 0


class BasinDataRepository:
    """Load only the basin/time slices authorized by the current phase."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        data = config["data"]
        self.data_path = resolve_repo_path(data["data_path"])
        self.reference_ids_path = resolve_repo_path(data["reference_ids"])
        self.attributes_path = resolve_repo_path(data.get("attributes_path", "data/caravan_671_attributes.npy"))
        if not self.data_path.is_file():
            raise FileNotFoundError(f"canonical data bundle not found: {self.data_path}")
        if not self.reference_ids_path.is_file():
            raise FileNotFoundError(f"canonical gage ID file not found: {self.reference_ids_path}")
        if not self.attributes_path.is_file():
            raise FileNotFoundError(
                "canonical Caravan attributes are required for OOB: "
                f"{self.attributes_path}"
            )
        self.reference_ids = load_ids(self.reference_ids_path)
        self._id_to_index = {int(value): index for index, value in enumerate(self.reference_ids.tolist())}

    def load_canonical_attributes(self, basin_ids: Sequence[int]) -> np.ndarray:
        attributes = np.load(self.attributes_path, allow_pickle=False)
        if attributes.ndim != 2 or attributes.shape[1] != N_ATTRIBUTES:
            raise ValueError(f"canonical Caravan attributes must have shape [basin, {N_ATTRIBUTES}], got {attributes.shape}")
        indices = self._indices(basin_ids)
        return np.asarray(attributes[indices], dtype=np.float64)

    def _indices(self, basin_ids: Sequence[int]) -> np.ndarray:
        ids = np.asarray(basin_ids, dtype=np.int64).reshape(-1)
        missing = [int(value) for value in ids if int(value) not in self._id_to_index]
        if missing:
            raise ValueError(f"basin IDs missing from reference file: {missing[:5]}")
        return np.asarray([self._id_to_index[int(value)] for value in ids], dtype=np.int64)

    def _load_bundle(self) -> tuple[np.ndarray, np.ndarray]:
        if CURRENT_PHASE == Phase.TRAIN:
            raise LeakageError(
                "monolithic canonical bundle cannot be loaded during TRAIN; "
                "prepare fold-partitioned OOB data before training"
            )
        with self.data_path.open("rb") as handle:
            bundle = pickle.load(handle)
        if isinstance(bundle, Mapping):
            forcings = bundle.get("forcings", bundle.get("forcing"))
            target = bundle.get("streamflow", bundle.get("target", bundle.get("discharge")))
        elif isinstance(bundle, (tuple, list)) and len(bundle) >= 2:
            forcings, target = bundle[0], bundle[1]
        else:
            raise ValueError("unsupported canonical data bundle; expected mapping or (forcings, target, ...)")
        if forcings is None or target is None:
            raise ValueError("canonical data bundle has no forcing/streamflow arrays")
        forcing_array = np.asarray(forcings)
        target_array = np.asarray(target)
        if forcing_array.ndim != 3 or forcing_array.shape[0] != self.reference_ids.size:
            raise ValueError(f"forcing array must be [gage, time, channel], got {forcing_array.shape}")
        if target_array.ndim == 3 and target_array.shape[-1] == 1:
            target_array = target_array[..., 0]
        if target_array.ndim != 2 or target_array.shape[:2] != forcing_array.shape[:2]:
            raise ValueError(f"target array must align as [gage, time], got {target_array.shape}")
        if forcing_array.shape[-1] < 3:
            raise ValueError("canonical forcing must contain prcp, tmean, and pet channels")
        return forcing_array, target_array

    def _dates(self) -> pd.DatetimeIndex:
        source = self.config["data"].get("source_start", "1980-10-01")
        end = self.config["data"].get("source_end", "2014-09-30")
        dates = pd.date_range(source, end, freq="D")
        return dates

    def _bounds(self, dates: pd.DatetimeIndex, period: Mapping[str, str]) -> tuple[int, int]:
        start = pd.Timestamp(period["start_time"])
        end = pd.Timestamp(period["end_time"])
        if start < dates[0] or end > dates[-1] or start > end:
            raise ValueError(f"period is outside canonical dates: {start.date()}..{end.date()}")
        return int(dates.get_loc(start)), int(dates.get_loc(end)) + 1

    def _convert_target(self, target_ft3s: np.ndarray, basin_ids: Sequence[int]) -> np.ndarray:
        area = self.load_canonical_attributes(basin_ids)[:, 11]
        if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
            raise ValueError("canonical area_gages2 values must be finite and positive")
        factor = 0.0283168 * 86400.0 * 1000.0 / (area * 1.0e6)
        return target_ft3s * factor[:, None]

    def load_training_period(self, basin_ids: Sequence[int], policy: TargetAccessPolicy) -> PeriodData:
        require_phase(Phase.TRAIN, "load_training_period")
        policy.authorize(basin_ids, "training loader")
        if policy.phase != Phase.TRAIN:
            raise LeakageError("training period requires a TRAIN target policy")
        dates = self._dates()
        left, right = self._bounds(dates, self.config["data"]["train"])
        forcings, target = self._load_bundle()
        indices = self._indices(basin_ids)
        selected_x = np.asarray(forcings[indices, left:right, :3], dtype=np.float64).transpose(1, 0, 2)
        selected_y = self._convert_target(np.asarray(target[indices, left:right], dtype=np.float64), basin_ids).T
        del forcings, target
        return PeriodData(
            torch.as_tensor(selected_x, dtype=torch.float64),
            torch.as_tensor(selected_y, dtype=torch.float64),
            np.asarray(basin_ids, dtype=np.int64),
            dates[left:right],
        )

    def load_heldout_evaluation(self, basin_ids: Sequence[int], policy: TargetAccessPolicy) -> PeriodData:
        require_phase(Phase.EVAL, "load_heldout_evaluation")
        policy.authorize(basin_ids, "held-out evaluation loader")
        if policy.phase != Phase.EVAL:
            raise LeakageError("held-out evaluation requires an EVAL target policy")
        dates = self._dates()
        test_left, test_right = self._bounds(dates, self.config["data"]["test"])
        warmup = int(self.config["protocol"]["evaluation_warmup_days"])
        forcing_left = test_left - warmup
        if forcing_left < 0:
            raise ValueError("canonical date range has insufficient evaluation warm-up forcing")
        forcings, target = self._load_bundle()
        indices = self._indices(basin_ids)
        selected_x = np.asarray(forcings[indices, forcing_left:test_right, :3], dtype=np.float64).transpose(1, 0, 2)
        selected_y = self._convert_target(np.asarray(target[indices, test_left:test_right], dtype=np.float64), basin_ids).T
        del forcings, target
        return PeriodData(
            torch.as_tensor(selected_x, dtype=torch.float64),
            torch.as_tensor(selected_y, dtype=torch.float64),
            np.asarray(basin_ids, dtype=np.int64),
            dates[forcing_left:test_right],
            warmup_days=warmup,
        )

class FoldDataCache:
    """Fold-partitioned data source used by TRAIN/EVAL workers.

    The preparation command may read the monolithic canonical bundle once, but
    a training worker reads only ``train_x.npy`` and ``train_y.npy``.  The
    held-out target file is not opened until the worker enters EVAL.
    """

    def __init__(self, cache_root: str | Path, fold: int, config: Mapping[str, Any]):
        self.cache_dir = Path(cache_root) / f"fold_{int(fold)}"
        self.fold = int(fold)
        self.config = config
        self.metadata_path = self.cache_dir / "cache_metadata.json"
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"fold data cache missing: {self.cache_dir}; run prepare_oob_data.py first"
            )
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.train_ids = np.asarray(self.metadata["train_basin_ids"], dtype=np.int64)
        self.heldout_ids = np.asarray(self.metadata["heldout_basin_ids"], dtype=np.int64)

    @staticmethod
    def _dates(config: Mapping[str, Any]) -> pd.DatetimeIndex:
        data = config["data"]
        return pd.date_range(data.get("source_start", "1980-10-01"), data.get("source_end", "2014-09-30"), freq="D")

    @staticmethod
    def _bounds(dates: pd.DatetimeIndex, period: Mapping[str, str]) -> tuple[int, int]:
        return int(dates.get_loc(pd.Timestamp(period["start_time"]))), int(dates.get_loc(pd.Timestamp(period["end_time"]))) + 1

    def _check_ids(self, basin_ids: Sequence[int], expected: np.ndarray, operation: str, *, allow_subset: bool = False) -> np.ndarray:
        requested = np.asarray(basin_ids, dtype=np.int64)
        positions = {int(value): index for index, value in enumerate(expected.tolist())}
        if allow_subset and all(int(value) in positions for value in requested):
            return np.asarray([positions[int(value)] for value in requested], dtype=np.int64)
        if not np.array_equal(requested, expected):
            raise LeakageError(f"{operation} IDs do not match the frozen fold cache")
        return np.arange(expected.size, dtype=np.int64)

    def load_training_period(self, basin_ids: Sequence[int], policy: TargetAccessPolicy, *, allow_subset: bool = False) -> PeriodData:
        require_phase(Phase.TRAIN, "load cached training period")
        policy.authorize(basin_ids, "cached training loader")
        indices = self._check_ids(basin_ids, self.train_ids, "training", allow_subset=allow_subset)
        x = np.load(self.cache_dir / "train_x.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(self.cache_dir / "train_y.npy", mmap_mode="r", allow_pickle=False)
        dates = self._dates(config=self.config)
        left, right = self._bounds(dates, self.config["data"]["train"])
        if x.shape[0] != right - left or y.shape != (right - left, len(self.train_ids)) or x.shape[1] != len(self.train_ids):
            raise ValueError("cached training arrays do not match the canonical train period")
        selected_x = np.array(x[:, indices, :], copy=True)
        selected_y = np.array(y[:, indices], copy=True)
        requested = np.asarray(basin_ids, dtype=np.int64)
        return PeriodData(torch.as_tensor(selected_x, dtype=torch.float64), torch.as_tensor(selected_y, dtype=torch.float64), requested, dates[left:right])
    def load_heldout_evaluation(self, basin_ids: Sequence[int], policy: TargetAccessPolicy, *, allow_subset: bool = False) -> PeriodData:
        require_phase(Phase.EVAL, "load cached held-out evaluation")
        policy.authorize(basin_ids, "cached held-out evaluator")
        indices = self._check_ids(basin_ids, self.heldout_ids, "held-out", allow_subset=allow_subset)
        x = np.load(self.cache_dir / "heldout_x.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(self.cache_dir / "heldout_y.npy", mmap_mode="r", allow_pickle=False)
        dates = self._dates(config=self.config)
        test_left, test_right = self._bounds(dates, self.config["data"]["test"])
        warmup = int(self.config["protocol"]["evaluation_warmup_days"])
        if x.shape[0] != test_right - test_left + warmup or y.shape != (test_right - test_left, len(self.heldout_ids)) or x.shape[1] != len(self.heldout_ids):
            raise ValueError("cached held-out arrays do not match the canonical test period")
        selected_x = np.array(x[:, indices, :], copy=True)
        selected_y = np.array(y[:, indices], copy=True)
        requested = np.asarray(basin_ids, dtype=np.int64)
        return PeriodData(torch.as_tensor(selected_x, dtype=torch.float64), torch.as_tensor(selected_y, dtype=torch.float64), requested, dates[test_left - warmup:test_right], warmup_days=warmup)



def validate_oob_protocol(config: Mapping[str, Any], *, models: Sequence[str] | None = None) -> None:
    folds = config["folds"]
    protocol = config["protocol"]
    data = config["data"]
    expected_models = tuple(models or config["models"])
    if expected_models != PRIMARY8:
        raise ValueError(f"PRIMARY 8 is frozen as {PRIMARY8}; got {expected_models}")
    if int(folds["n_splits"]) != 5 or not bool(folds["shuffle"]) or int(folds["random_state"]) != FOLD_SEED:
        raise ValueError("OOB fold protocol must be 5-fold shuffle=True random_state=20260902")
    expected = {
        "training_warmup_days": 730,
        "scored_horizon_days": 365,
        "evaluation_warmup_days": 365,
        "parameter_mapping": "auto",
        "optimizer": "AdamW",
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "scheduler": "None",
        "clip_norm": 1.0,
        "kge_eps": KGE_EPS,
        "seed": 42,
        "min_epochs": 50,
        "patience": 10,
        "plateau_eps": 1.0e-4,
        "max_epochs": 100,
        "selection_metric": "train_loss",
        "exact_best_checkpoint": True,
    }
    for key, value in expected.items():
        actual = protocol.get(key)
        if actual != value:
            raise ValueError(f"protocol.{key} must be {value!r}, got {actual!r}")
    if (data["train"]["start_time"], data["train"]["end_time"]) != ("1980-10-01", "1995-09-30"):
        raise ValueError("OOB training period is not canonical 1980-10-01..1995-09-30")
    if (data["test"]["start_time"], data["test"]["end_time"]) != ("1995-10-01", "2010-09-30"):
        raise ValueError("OOB test period is not canonical 1995-10-01..2010-09-30")


def build_informative_window_catalog(
    observations: np.ndarray,
    *,
    scored_days: int = 365,
    min_valid_points: int = 30,
    min_observation_std: float = 0.01,
) -> list[np.ndarray]:
    """Build train-only window starts from post-warm-up observations.

    The returned start indices are relative to the full ``warmup + scored``
    training sequence: the caller passes observations beginning at
    ``train_y[warmup_days:]`` but gathers forcing/target from index zero.
    """

    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"observations must be [basin, time], got {values.shape}")
    n_basins, n_days = values.shape
    if n_days < scored_days:
        raise ValueError("not enough train-only observations for one scored window")
    finite = np.isfinite(values) & (values >= 0.0)
    clean = np.where(finite, values, 0.0)
    count_cs = np.concatenate((np.zeros((n_basins, 1)), np.cumsum(finite, axis=1)), axis=1)
    sum_cs = np.concatenate((np.zeros((n_basins, 1)), np.cumsum(clean, axis=1)), axis=1)
    square_cs = np.concatenate((np.zeros((n_basins, 1)), np.cumsum(clean * clean, axis=1)), axis=1)
    count = count_cs[:, scored_days:] - count_cs[:, :-scored_days]
    total = sum_cs[:, scored_days:] - sum_cs[:, :-scored_days]
    square_total = square_cs[:, scored_days:] - square_cs[:, :-scored_days]
    variance = np.maximum(square_total / np.maximum(count, 1.0) - (total / np.maximum(count, 1.0)) ** 2, 0.0)
    eligible = (count >= min_valid_points) & (variance >= float(min_observation_std) ** 2)
    catalog: list[np.ndarray] = []
    for basin in range(n_basins):
        starts = np.flatnonzero(eligible[basin])
        if starts.size == 0:
            fallback_score = np.where(count[basin] >= min_valid_points, variance[basin], -1.0)
            starts = np.asarray([int(np.argmax(fallback_score))], dtype=np.int64)
        catalog.append(starts.astype(np.int64))
    return catalog


def gather_window(values: torch.Tensor, starts: torch.Tensor, basin_indices: torch.Tensor, horizon_days: int) -> torch.Tensor:
    if values.ndim < 2 or starts.ndim != 1 or basin_indices.ndim != 1:
        raise ValueError("window gather expects values [time, basin, ...] and one-dimensional indices")
    days = torch.arange(horizon_days, device=values.device)[:, None] + starts[None, :]
    if int(days.max()) >= values.shape[0]:
        raise IndexError("requested training window exceeds available train period")
    return values[days, basin_indices[None, :]]


def kge_numpy(prediction: np.ndarray, observation: np.ndarray, *, eps: float = KGE_EPS) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return per-column KGE and invalid mask using the project convention."""

    pred = np.asarray(prediction, dtype=np.float64)
    obs = np.asarray(observation, dtype=np.float64)
    if pred.shape != obs.shape or pred.ndim != 2:
        raise ValueError(f"prediction and observation must share [time, basin] shape, got {pred.shape}, {obs.shape}")
    finite_pred = np.isfinite(pred)
    finite = finite_pred & np.isfinite(obs)
    clean_pred = np.where(finite, pred, 0.0)
    clean_obs = np.where(finite, obs, 0.0)
    count = finite.sum(axis=0).astype(np.float64)
    safe_n = np.maximum(count, 1.0)
    sum_pred = clean_pred.sum(axis=0)
    sum_obs = clean_obs.sum(axis=0)
    sum_pred2 = np.square(clean_pred).sum(axis=0)
    sum_obs2 = np.square(clean_obs).sum(axis=0)
    sum_cross = (clean_pred * clean_obs).sum(axis=0)
    score = kge_from_statistics(
        {
            "count": count,
            "sum_pred": sum_pred,
            "sum_obs": sum_obs,
            "sum_pred2": sum_pred2,
            "sum_obs2": sum_obs2,
            "sum_cross": sum_cross,
        },
        eps=eps,
    )
    invalid = (count < 2.0) | ~np.isfinite(score) | ~finite_pred.all(axis=0)
    stats = {
        "count": float(count.sum()),
        "sum_pred": float(sum_pred.sum()),
        "sum_obs": float(sum_obs.sum()),
        "sum_pred2": float(sum_pred2.sum()),
        "sum_obs2": float(sum_obs2.sum()),
        "sum_cross": float(sum_cross.sum()),
    }
    return score, invalid, stats


def kge_from_statistics(stats: Mapping[str, Any], *, eps: float = KGE_EPS) -> np.ndarray | float:
    count = np.asarray(stats["count"], dtype=np.float64)
    sum_pred = np.asarray(stats["sum_pred"], dtype=np.float64)
    sum_obs = np.asarray(stats["sum_obs"], dtype=np.float64)
    sum_pred2 = np.asarray(stats["sum_pred2"], dtype=np.float64)
    sum_obs2 = np.asarray(stats["sum_obs2"], dtype=np.float64)
    sum_cross = np.asarray(stats["sum_cross"], dtype=np.float64)
    safe_n = np.maximum(count, 1.0)
    centered_pred = np.maximum(sum_pred2 - np.square(sum_pred) / safe_n, 0.0)
    centered_obs = np.maximum(sum_obs2 - np.square(sum_obs) / safe_n, 0.0)
    denom_n = np.maximum(count - 1.0, 1.0)
    std_pred = np.sqrt(np.maximum(centered_pred / denom_n, 1.0e-24))
    std_obs = np.sqrt(np.maximum(centered_obs / denom_n, 1.0e-24))
    covariance_scale = np.sqrt(np.maximum(centered_pred * centered_obs, 1.0e-24))
    correlation = (sum_cross - sum_pred * sum_obs / safe_n) / (covariance_scale + eps)
    beta = (sum_pred / safe_n) / (sum_obs / safe_n + eps)
    gamma = std_pred / (std_obs + eps)
    score = 1.0 - np.sqrt(np.maximum((correlation - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2, 1.0e-24))
    return float(score) if score.ndim == 0 else score


def combine_statistics(items: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    keys = ("count", "sum_pred", "sum_obs", "sum_pred2", "sum_obs2", "sum_cross")
    return {key: float(sum(float(item[key]) for item in items)) for key in keys}


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def append_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exists = destination.exists()
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
