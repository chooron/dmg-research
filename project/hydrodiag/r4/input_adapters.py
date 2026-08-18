"""R4 observation-trained input adapters (IC + dPL).

These adapters load *trained* parameters from run artifacts without any
re-calibration:

- IC (canonical batched format): ``raw/<model>/<basin>_startNN.json`` per
  basin × start; canonical R1 restart selection = highest stored train-period
  KGE, lowest start number breaks ties (same rule as
  ``manuscript/scripts/r1_daily_inference.py::read_ic_parameters``).
- IC (fused format): ``per_start.csv`` with per-basin × start rows
  (train_kge, theta_normalized, physical ``p_*`` columns) — the format
  produced by the remote fused CMA-ES runs
  (``ic_cmaes_recalibration_p20_p25_*_fused``).
- dPL: seed checkpoint (``best_checkpoint.pt`` per seed, or an explicit
  ``checkpoint_epoch_NNN.pt``) + ``config.json`` + ``attribute_normalization.npz``
  → MLP forward → sigmoid/bounds mapping to physical parameters.

All adapters raise :class:`R4ArtifactError` (a ``RuntimeError``) when the
expected artifacts are missing or inconsistent — they never silently fall
back to R3 synthetic-trained results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .common import DPL_SEEDS, zfill8

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class R4ArtifactError(RuntimeError):
    """Raised when a required training artifact is missing or inconsistent."""


def _describe_missing(path: Path) -> str:
    if not path.exists():
        return f"missing: {path}"
    return f"unexpected state at {path}"


# ---------------------------------------------------------------------------
# IC — canonical batched format (raw/<model>/<basin>_startNN.json)
# ---------------------------------------------------------------------------


@dataclass
class ICRecord:
    basin_id: str
    start: int
    train_kge: float
    test_kge: float
    parameters: np.ndarray
    parameter_names: tuple[str, ...]
    source: Path
    raw: dict[str, Any] = field(default_factory=dict)


def iter_ic_records(run_root: Path, raw_subdir: str) -> list[ICRecord]:
    """Read all per-basin × start JSON records from a canonical IC run."""
    raw_dir = run_root / "raw" / raw_subdir
    if not raw_dir.is_dir():
        raise R4ArtifactError(
            f"IC run {run_root.name}: expected raw record directory "
            f"{raw_dir} (canonical batched format raw/<model>/<basin>_startNN.json); "
            f"{_describe_missing(raw_dir)}"
        )
    records: list[ICRecord] = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise R4ArtifactError(f"IC record {path} is not valid JSON: {exc}") from exc
        basin = zfill8(data.get("basin_id"))
        status = data.get("status")
        if status != "complete":
            continue  # failed/incomplete starts are excluded, exactly as in R1

        train_kge = (data.get("train_metrics") or {}).get("kge")
        test_kge = (data.get("test_metrics") or {}).get("kge")
        if train_kge is None or not np.isfinite(train_kge):
            continue
        parameters = np.asarray(data["parameters"], dtype=np.float64)
        names = tuple(str(x) for x in data["parameter_names"])
        records.append(ICRecord(
            basin_id=basin,
            start=int(data["start"]),
            train_kge=float(train_kge),
            test_kge=float(test_kge if test_kge is not None else np.nan),
            parameters=parameters,
            parameter_names=names,
            source=path,
            raw=data,
        ))
    if not records:
        raise R4ArtifactError(
            f"IC run {run_root.name}: no complete records found under {raw_dir}"
        )
    return records


def select_best_restart(records: Iterable[ICRecord]) -> ICRecord:
    """R1 canonical rule: highest train KGE; lowest start number breaks ties."""
    return min(records, key=lambda r: (-r.train_kge, r.start))


def read_ic_canonical(
    run_root: Path,
    model_key: str,
    raw_subdir: str,
    basin_ids: Iterable[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Best-restart physical parameters [n, D] for a canonical-format IC run.

    Returns (parameters, meta) with meta containing selected starts, sources,
    and per-basin KGE.  Raises R4ArtifactError if any basin has no valid
    restart (fail loud, no silent fallback).
    """
    expected = set(zfill8(b) for b in basin_ids)
    records = iter_ic_records(run_root, raw_subdir)
    by_basin: dict[str, list[ICRecord]] = {}
    for record in records:
        by_basin.setdefault(record.basin_id, []).append(record)

    selected: list[ICRecord] = []
    missing: list[str] = []
    # Parameters are stacked in the *provided* basin_ids order (the canonical
    # R1 bundle order).  Sorting here would silently misalign parameters vs
    # the forcing/target axes (regression: 439/531 positions differ from the
    # bundle order for the R3 gate IC run).
    for basin in [zfill8(b) for b in basin_ids]:
        candidates = by_basin.get(basin)
        if not candidates:
            missing.append(basin)
            continue
        selected.append(select_best_restart(candidates))

    if missing:
        raise R4ArtifactError(
            f"IC run {run_root.name} ({model_key}): no complete restart for "
            f"{len(missing)}/{len(expected)} basins (first: {missing[:5]}); "
            f"records found: {len(records)}"
        )

    names = selected[0].parameter_names
    if any(r.parameter_names != names for r in selected):
        raise R4ArtifactError(
            f"IC run {run_root.name}: parameter-name mismatch across basins"
        )
    parameters = np.stack([r.parameters for r in selected]).astype(np.float64)
    meta = {
        "format": "canonical_batched_json",
        "model_key": model_key,
        "run_root": str(run_root),
        "parameter_names": list(names),
        "selected_starts": [r.start for r in selected],
        "selected_sources": [str(r.source) for r in selected],
        "train_kge": [r.train_kge for r in selected],
        "test_kge": [r.test_kge for r in selected],
        "n_basins": len(selected),
        "restart_rule": "best train-period KGE restart per basin; lowest start number breaks ties (R1 canonical)",
    }
    return parameters, meta


# ---------------------------------------------------------------------------
# IC — fused per_start.csv format (remote fused CMA-ES runs)
# ---------------------------------------------------------------------------


def _fused_param_columns(header: list[str], parameter_names: tuple[str, ...]) -> list[str]:
    """Map canonical parameter names onto p_* columns of a fused per_start.csv.

    Physical columns appear as ``p_<name>`` (e.g. ``p_xaj_k``, ``p_cn_ctg``,
    ``p_tgd_tau_warm``).  A legacy TGD run (``p_tgd_a``/``p_tgd_k_slow``) is
    rejected because it is not the canonical TGD2 structure.
    """
    columns: list[str] = []
    for name in parameter_names:
        candidate = f"p_{name}"
        if candidate in header:
            columns.append(candidate)
            continue
        # old-TGD aliases are structurally different models — refuse silently
        # mapping them onto TGD2 parameters.
        raise R4ArtifactError(
            f"fused IC csv: column '{candidate}' not found for parameter "
            f"'{name}' (header has {len(header)} columns); legacy TGD (tgd_a/"
            f"tgd_k_slow) is not the canonical TGD2 structure"
        )
    return columns


def read_ic_fused(
    per_start_csv: Path,
    model_key: str,
    parameter_names: tuple[str, ...],
    basin_ids: Iterable[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Best-restart physical parameters [n, D] from a fused per_start.csv.

    Raises R4ArtifactError on missing file/columns/basins.
    """
    import pandas as pd

    if not per_start_csv.is_file():
        raise R4ArtifactError(f"fused IC csv missing: {per_start_csv}")
    df = pd.read_csv(per_start_csv)
    df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    expected = set(zfill8(b) for b in basin_ids)
    present = set(df["basin_id"])
    if not expected <= present:
        raise R4ArtifactError(
            f"fused IC csv {per_start_csv.name}: missing {len(expected - present)} "
            f"basins (first: {sorted(expected - present)[:5]})"
        )
    col_map = _fused_param_columns(list(df.columns), tuple(parameter_names))

    # Stack in the *provided* basin_ids order (canonical R1 bundle order) —
    # never sorted order, which would misalign parameters vs forcing/target.
    selected_rows: list[Any] = []
    for basin in [zfill8(b) for b in basin_ids]:
        rows = df[df["basin_id"] == basin]
        # R1 canonical rule on the fused format: train_kge max, start_id min tiebreak
        rows = rows.sort_values(["train_kge", "start_id"], ascending=[False, True])
        selected_rows.append(rows.iloc[0])
    sel = pd.DataFrame(selected_rows)
    parameters = sel[col_map].to_numpy(dtype=np.float64)
    meta = {
        "format": "fused_per_start_csv",
        "model_key": model_key,
        "source": str(per_start_csv),
        "parameter_names": list(parameter_names),
        "selected_starts": [int(s) for s in sel["start_id"]],
        "train_kge": [float(x) for x in sel["train_kge"]],
        "test_kge": [float(x) for x in sel["test_kge"]],
        "n_basins": len(sel),
        "restart_rule": "best train-period KGE restart per basin; lowest start number breaks ties (R1 canonical)",
    }
    return parameters, meta


# ---------------------------------------------------------------------------
# dPL — seed checkpoints
# ---------------------------------------------------------------------------


def _valid_checkpoint(path: Path, model_key: str) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt checkpoint
        raise R4ArtifactError(f"dPL checkpoint unreadable {path}: {exc}") from exc
    if checkpoint.get("model_name") != model_key:
        raise R4ArtifactError(
            f"dPL checkpoint {path}: model_name={checkpoint.get('model_name')!r}, "
            f"expected {model_key!r}"
        )
    if not checkpoint.get("lite_mode", False):
        raise R4ArtifactError(f"dPL checkpoint {path}: lite_mode is not True")
    if not isinstance(checkpoint.get("state_dict"), dict) or not checkpoint["state_dict"]:
        raise R4ArtifactError(f"dPL checkpoint {path}: empty state_dict")
    return checkpoint


def read_dpl_seed(
    seed_dir: Path,
    model_key: str,
    data_root: Path,
    basin_ids: Iterable[str],
    *,
    checkpoint_epoch: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Physical parameters [n, D] from one dPL seed.

    ``checkpoint_epoch=None`` uses the R1 lite-v2 convention
    (``best_checkpoint.pt``); an explicit epoch uses
    ``checkpoint_epoch_NNN.pt`` (the TGD2 audited-run convention).

    Raises R4ArtifactError when config/checkpoint/normalization are missing
    or inconsistent (fail loud, no silent fallback to synthetic runs).
    """
    import torch

    from training.dpl.run_dpl_model import (
        LITE_MODEL_REGISTRY, StaticParameterNet, physical_parameters, robust_normalize,
    )

    config_path = seed_dir / "config.json"
    if not config_path.is_file():
        raise R4ArtifactError(f"dPL seed {seed_dir}: config.json missing")
    config = json.loads(config_path.read_text())
    if config.get("model_name") != model_key:
        raise R4ArtifactError(
            f"dPL seed {seed_dir}: config model_name={config.get('model_name')!r}, "
            f"expected {model_key!r}"
        )

    if checkpoint_epoch is None:
        checkpoint_path = seed_dir / "best_checkpoint.pt"
        epoch_label = "best_checkpoint.pt"
    else:
        checkpoint_path = seed_dir / f"checkpoint_epoch_{checkpoint_epoch:03d}.pt"
        epoch_label = f"epoch {checkpoint_epoch}"
    if not checkpoint_path.is_file():
        raise R4ArtifactError(f"dPL seed {seed_dir}: {checkpoint_path.name} missing")
    checkpoint = _valid_checkpoint(checkpoint_path, model_key)

    norm_path = seed_dir / "attribute_normalization.npz"
    if not norm_path.is_file():
        raise R4ArtifactError(f"dPL seed {seed_dir}: attribute_normalization.npz missing")
    normalization = np.load(norm_path)

    # ---- rebuild the parameter net exactly as training did ----------------
    model_cls, specs = LITE_MODEL_REGISTRY[model_key]
    names = list(specs)
    lower = np.asarray([specs[name]["lower"] for name in names], dtype=np.float64)
    upper = np.asarray([specs[name]["upper"] for name in names], dtype=np.float64)
    net_cfg = config["network"]
    hidden_sizes = [int(v) for v in net_cfg.get("hidden_sizes", [net_cfg["hidden_size"]] * net_cfg.get("depth", 2))]
    net = StaticParameterNet(35, specs, hidden_sizes, net_cfg["dropout"], net_cfg["output_epsilon"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    # ---- attributes: full canonical 531 set, robust-normalized ------------
    from ablation.ic_core.data_adapter import load_531_bundle

    from .common import bundle_config

    bundle = load_531_bundle(bundle_config(data_root))
    attrs_np, _stats = robust_normalize(bundle.raw_attributes.astype(np.float32))
    stored_median = np.asarray(normalization["median"], dtype=np.float32)
    stored_scale = np.asarray(normalization["scale"], dtype=np.float32)
    # The stored normalization is authoritative: recomputing it over the
    # canonical 531 set must reproduce the stored statistics exactly.
    _ = attrs_np
    if stored_median.shape != (35,) or stored_scale.shape != (35,):
        raise R4ArtifactError(f"dPL seed {seed_dir}: normalization stats shape mismatch")
    if not np.allclose(stored_median, _stats["median"], atol=1e-6):
        raise R4ArtifactError(
            f"dPL seed {seed_dir}: stored normalization median does not match "
            f"recomputed median (max diff "
            f"{np.abs(stored_median - _stats['median']).max():.3e})"
        )
    if not np.allclose(stored_scale, _stats["scale"], atol=1e-5):
        raise R4ArtifactError(
            f"dPL seed {seed_dir}: stored normalization scale does not match "
            f"recomputed scale (max diff "
            f"{np.abs(stored_scale - _stats['scale']).max():.3e})"
        )
    # ---- basin order -------------------------------------------------------
    expected = [zfill8(b) for b in basin_ids]
    bundle_ids = [zfill8(b) for b in bundle.basin_ids]
    positions = [bundle_ids.index(b) for b in expected]
    attrs = torch.from_numpy(attrs_np[positions].astype(np.float32))

    with torch.no_grad():
        theta = net(attrs)
        parameter_range = upper - lower
        physical = physical_parameters(
            theta, names,
            torch.from_numpy(lower).float(),
            torch.from_numpy(parameter_range).float(),
        )
    parameters = np.stack([physical[name].detach().cpu().numpy() for name in names], axis=1).astype(np.float64)

    meta = {
        "format": "dpl_checkpoint",
        "model_key": model_key,
        "seed_dir": str(seed_dir),
        "checkpoint": str(checkpoint_path),
        "epoch_label": epoch_label,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "parameter_names": names,
        "n_basins": len(expected),
        "attribute_normalization": str(norm_path),
        "network_config": net_cfg,
    }
    return parameters, meta


def read_dpl_multiseed(
    run_root: Path,
    model_key: str,
    data_root: Path,
    basin_ids: Iterable[str],
    seeds: Iterable[int] = DPL_SEEDS,
    *,
    checkpoint_epoch: Optional[int] = None,
) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
    """Read one parameter field per seed; raises if any seed is missing."""
    result: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    for seed in seeds:
        seed_dir = run_root / f"seed_{seed}"
        if not seed_dir.is_dir():
            raise R4ArtifactError(
                f"dPL run {run_root.name}: expected seed directory {seed_dir}"
            )
        result[seed] = read_dpl_seed(
            seed_dir, model_key, data_root, basin_ids, checkpoint_epoch=checkpoint_epoch
        )
    return result
