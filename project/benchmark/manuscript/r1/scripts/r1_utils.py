"""Small, side-effect-free helpers shared by the R1 scripts."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from r1_config import CACHE_DIR, MODEL_REGISTRY


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_basin_id(value: Any) -> str:
    """Return the eight-character CAMELS gage ID used for every R1 join."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def read_canonical_ids(path: Path) -> list[str]:
    text = path.read_text().strip()
    # The canonical file is a serialized Python list on one line, although
    # newline-delimited lists are also accepted for audit portability.
    if text.startswith("[") and text.endswith("]"):
        import ast
        values = ast.literal_eval(text)
    else:
        values = text.splitlines()
    return [canonical_basin_id(x) for x in values if str(x).strip()]


def registry_sort(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    order = {name: i for i, name in enumerate(MODEL_REGISTRY)}
    out = df.copy()
    out["_model_order"] = out[model_col].map(order).fillna(len(order))
    cols = [c for c in out.columns if c != "_model_order"]
    return out.sort_values(["_model_order", model_col], kind="stable").drop(columns="_model_order")[cols]


def finite_values(values: Any) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def summary(values: Any) -> dict[str, float | int]:
    x = finite_values(values)
    if x.size == 0:
        return {"N": 0, "median": np.nan, "mean": np.nan, "Q10": np.nan, "Q25": np.nan,
                "Q75": np.nan, "Q90": np.nan, "min": np.nan, "max": np.nan}
    return {
        "N": int(x.size), "median": float(np.quantile(x, 0.50)), "mean": float(np.mean(x)),
        "Q10": float(np.quantile(x, 0.10)), "Q25": float(np.quantile(x, 0.25)),
        "Q75": float(np.quantile(x, 0.75)), "Q90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)), "max": float(np.max(x)),
    }


def spearman(x: Any, y: Any) -> float:
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    ar = pd.Series(a[ok]).rank(method="average").to_numpy()
    br = pd.Series(b[ok]).rank(method="average").to_numpy()
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def pearson(x: Any, y: Any) -> float:
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def append_cache_manifest(rows: list[dict[str, Any]]) -> None:
    """Write a deterministic manifest for all currently produced R1 cache files."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = list(rows)
    for row in manifest_rows:
        path = CACHE_DIR / str(row.get("cache_file", ""))
        if path.is_file():
            row["sha256"] = sha256_file(path)
    for path in sorted(CACHE_DIR.iterdir()):
        if path.name == "cache_manifest.csv" or path.is_dir() or path.suffix == ".json":
            continue
        if not path.is_file():
            continue
        existing = next((r for r in manifest_rows if r.get("cache_file") == path.name), None)
        if existing is None:
            manifest_rows.append({
                "cache_file": path.name, "kind": path.suffix.lstrip(".") or "file",
                "model": "", "created_utc": utc_now(), "script": "unknown",
                "source_provenance": "R1 cache; see adjacent metadata",
                "configuration": "", "status": "PRESENT",
                "sha256": sha256_file(path),
            })
    fields = ["cache_file", "kind", "model", "created_utc", "script",
              "source_provenance", "configuration", "status", "sha256"]
    write_csv(CACHE_DIR / "cache_manifest.csv", sorted(manifest_rows, key=lambda r: r.get("cache_file", "")), fields)
