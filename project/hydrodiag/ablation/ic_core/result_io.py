from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schemas import RESULT_FIELDS


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n"
    )


def atomic_write_csv(
    path: str | Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fieldnames), extrasaction="raise"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, sort_keys=True, default=json_default) + "\n" for row in rows
    )
    atomic_write_text(path, text)


def validate_result_record(record: dict[str, Any]) -> None:
    missing = [field for field in RESULT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"result record missing fields: {missing}")
    if record["optimizer"] == "none_smoke" and record["population"] is None:
        raise ValueError("smoke result must make population semantics explicit")
