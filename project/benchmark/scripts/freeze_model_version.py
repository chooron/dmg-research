#!/usr/bin/env python3
"""Write a content-addressed model/runtime manifest without touching source files.

Canonical version-freeze step of the Full300 CMA-ES pipeline.
Creates <benchmark>/frozen_versions/<name>/manifest.json containing:
  * SHA-256 of every model/runtime source file;
  * resolved production config;
  * 36-model registry dimensions and parameter bounds.

Local adaptation of the Full300 deployment script
(remote: experiments/cmaes_36models/scripts/freeze_model_version.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from src.model_registry import NPARAM_INFO_36, get_spec
from src.production_config import load_resolved_config, validate_full_run_config


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    validate_full_run_config(config)

    paths = sorted((BENCHMARK_ROOT / "dmotpy/models").rglob("*.py"))
    paths += sorted((BENCHMARK_ROOT / "src").glob("*.py"))
    paths += [BENCHMARK_ROOT / "scripts/run_36model_benchmark.py", Path(config["_resolved_from"])]
    # Deduplicate on the resolved path (a dmotpy symlink under BENCHMARK_ROOT
    # resolves outside it) but keep the original relative path as the manifest
    # key so validate_full300_config.py can re-open it through the same layout.
    seen: set[Path] = set()
    source_hashes: dict[str, str] = {}
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        source_hashes[str(path.relative_to(BENCHMARK_ROOT))] = digest(resolved)
    bounds = {name: get_spec(name).bounds.cpu().tolist() for name in NPARAM_INFO_36}
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BENCHMARK_ROOT, text=True).strip()
    except Exception:
        git_head = None
    manifest = {
        "name": args.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "note": "The repository worktree is not assumed clean; source_hashes are the authoritative freeze identifier.",
        "resolved_config": config,
        "model_registry": NPARAM_INFO_36,
        "parameter_bounds": bounds,
        "source_hashes_sha256": source_hashes,
        "aggregate_source_hash": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
    }
    out = BENCHMARK_ROOT / "frozen_versions" / args.name / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)


if __name__ == "__main__":
    main()
