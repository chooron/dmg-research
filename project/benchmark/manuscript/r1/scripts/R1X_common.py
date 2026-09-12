"""Shared table loading and balanced-matrix checks for the R1X analyses."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from r1_config import DATA_ROOT, MODEL_REGISTRY, TABLES_DIR, TEMPORAL_AB
from r1_utils import canonical_basin_id, read_canonical_ids

FULL_TABLE = TABLES_DIR / "R1_model_basin_delta_kge.csv"
TEMPORAL_TABLE = TABLES_DIR / "R1_temporal_AB_model_basin.csv"


def canonical_ids() -> list[str]:
    ids = read_canonical_ids(DATA_ROOT / "531sub_id.txt")
    if len(ids) != 531 or len(set(ids)) != 531:
        raise RuntimeError("canonical basin list is not exactly 531 unique IDs")
    return ids


def _validate_delta(frame: pd.DataFrame) -> None:
    delta = frame["delta_KGE"].to_numpy(float)
    expected = frame["KGE_dPL"].to_numpy(float) - frame["KGE_IC"].to_numpy(float)
    if not np.isfinite(frame[["KGE_IC", "KGE_dPL", "delta_KGE"]].to_numpy(float)).all():
        raise RuntimeError("R1X input contains non-finite KGE values")
    if not np.allclose(delta, expected, rtol=0.0, atol=1e-8):
        raise RuntimeError("R1X input violates delta_KGE = KGE_dPL - KGE_IC")


def load_full_table() -> pd.DataFrame:
    frame = pd.read_csv(FULL_TABLE, dtype={"model": str, "basin_id": str})
    required = {"model", "basin_id", "KGE_IC", "KGE_dPL", "delta_KGE"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"full R1 table missing columns: {sorted(missing)}")
    frame["model"] = frame["model"].astype(str)
    frame["basin_id"] = frame["basin_id"].map(canonical_basin_id)
    ids = canonical_ids()
    if len(frame) != len(MODEL_REGISTRY) * len(ids):
        raise RuntimeError(f"expected 36x531 full cells, got {len(frame)}")
    if set(frame["model"]) != set(MODEL_REGISTRY) or frame["model"].nunique() != 36:
        raise RuntimeError("full R1 table model set is not exactly registry_36")
    if set(frame["basin_id"]) != set(ids) or frame["basin_id"].nunique() != 531:
        raise RuntimeError("full R1 table basin set is not exactly canonical_531")
    if frame.duplicated(["model", "basin_id"]).any():
        raise RuntimeError("full R1 table contains duplicate (model, basin_id) cells")
    actual = set(zip(frame["model"], frame["basin_id"]))
    expected = {(model, basin) for model in MODEL_REGISTRY for basin in ids}
    if actual != expected:
        raise RuntimeError("full R1 table is not a complete balanced model-basin matrix")
    _validate_delta(frame)
    frame["model"] = pd.Categorical(frame["model"], categories=list(MODEL_REGISTRY), ordered=True)
    frame["basin_id"] = pd.Categorical(frame["basin_id"], categories=ids, ordered=True)
    return frame.sort_values(["model", "basin_id"], kind="stable").reset_index(drop=True)


def load_temporal_table() -> pd.DataFrame:
    frame = pd.read_csv(TEMPORAL_TABLE, dtype={"model": str, "basin_id": str, "partition": str})
    required = {"model", "basin_id", "partition", "start_date", "end_date", "KGE_IC", "KGE_dPL", "delta_KGE"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"temporal R1 table missing columns: {sorted(missing)}")
    frame["model"] = frame["model"].astype(str)
    frame["basin_id"] = frame["basin_id"].map(canonical_basin_id)
    ids = canonical_ids()
    expected_rows = len(MODEL_REGISTRY) * len(ids) * 2
    if len(frame) != expected_rows:
        raise RuntimeError(f"expected 36x531x2 temporal cells, got {len(frame)}")
    if set(frame["partition"]) != {"A", "B"}:
        raise RuntimeError("temporal R1 table does not contain exactly A and B")
    if frame.duplicated(["model", "basin_id", "partition"]).any():
        raise RuntimeError("temporal R1 table contains duplicate (model, basin_id, partition) cells")
    for definition in TEMPORAL_AB:
        part = frame.loc[frame["partition"] == definition["partition"]]
        if len(part) != len(MODEL_REGISTRY) * len(ids) or set(part["model"]) != set(MODEL_REGISTRY) or set(part["basin_id"]) != set(ids):
            raise RuntimeError(f"temporal partition {definition['partition']} is not complete")
        if set(part["start_date"].astype(str)) != {definition["start_date"]} or set(part["end_date"].astype(str)) != {definition["end_date"]}:
            raise RuntimeError(f"temporal partition {definition['partition']} has unexpected dates")
    _validate_delta(frame)
    frame["model"] = pd.Categorical(frame["model"], categories=list(MODEL_REGISTRY), ordered=True)
    frame["basin_id"] = pd.Categorical(frame["basin_id"], categories=ids, ordered=True)
    frame["partition"] = pd.Categorical(frame["partition"], categories=["A", "B"], ordered=True)
    return frame.sort_values(["model", "partition", "basin_id"], kind="stable").reset_index(drop=True)


def quantile_dict(values: pd.Series | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {
        "Q10": float(np.quantile(x, 0.10)),
        "Q25": float(np.quantile(x, 0.25)),
        "Q50": float(np.quantile(x, 0.50)),
        "Q75": float(np.quantile(x, 0.75)),
        "Q90": float(np.quantile(x, 0.90)),
    }
