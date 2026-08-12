#!/usr/bin/env python3
"""Write a content-addressed model/runtime manifest without touching source files."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/cmaes_36models"
sys.path[:0] = [str(ROOT), str(EXPERIMENT)]

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

    paths = sorted((ROOT / "dmotpy/models").rglob("*.py"))
    paths += sorted((EXPERIMENT / "src").glob("*.py"))
    paths += [EXPERIMENT / "scripts/run_36model_pilot.py", Path(config["_resolved_from"])]
    unique = sorted({path.resolve() for path in paths if "__pycache__" not in path.parts})
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in unique}
    bounds = {name: get_spec(name).bounds.cpu().tolist() for name in NPARAM_INFO_36}
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
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
    out = EXPERIMENT / "frozen_versions" / args.name / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)


if __name__ == "__main__":
    main()
