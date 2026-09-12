"""Reproducible run metadata helpers for future experiment jobs."""

from __future__ import annotations
import hashlib
import json

import platform
import subprocess
import time
from pathlib import Path

from .protocol import ExperimentProtocol


def git_sha(root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def manifest_record(root: str | Path = ".", manifest_path: str | Path | None = None) -> dict[str, object]:
    path = Path(manifest_path) if manifest_path else Path(root) / "project/autofuse/manifests/camels_544.json"
    if not path.is_file():
        return {"path": str(path), "sha256": None, "manifest_sha256": None, "status": "missing"}
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_sha256": payload.get("manifest_sha256"),
        "status": "available",
    }

def run_record(
    protocol: ExperimentProtocol,
    *,
    seed: int,
    root: str | Path = ".",
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    return {
        "seed": seed,
        "config": protocol.to_dict(),
        "manifest": manifest_record(root, manifest_path),
        "git_sha": git_sha(root),
        "runtime_unix": time.time(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "training_started": False,
    }
