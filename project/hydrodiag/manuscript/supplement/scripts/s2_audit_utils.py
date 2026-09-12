#!/usr/bin/env python3
"""Shared utilities for the S2 formula audit.

This module is deliberately independent of the production model code except
for importing it during runtime probes.  It writes only to the supplement
directory and keeps source evidence as file:line references.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


def project_root_from_args(value: str | None = None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def supplement_dir(project_root: Path, output_dir: str | None = None) -> Path:
    return Path(output_dir).expanduser().resolve() if output_dir else project_root / "manuscript" / "supplement"


def ensure_dirs(out: Path) -> None:
    for name in ("scripts", "results", "reports"):
        (out / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_ref(path: Path, line: int | None = None, symbol: str | None = None) -> str:
    bits = [str(path)]
    if line is not None:
        bits.append(str(line))
    if symbol:
        bits.append(symbol)
    return ":".join(bits)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def git_status(project_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(project_root)
        p = subprocess.run(["git", "-C", str(project_root), "status", "--short", "--", str(rel)], capture_output=True, text=True, check=False)
        return p.stdout.strip() or "clean"
    except Exception as exc:  # pragma: no cover
        return f"unavailable:{exc}"


def line_for(path: Path, needle: str) -> int | None:
    try:
        for n, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if needle in text:
                return n
    except OSError:
        return None
    return None


def rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def json_safe(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value

