#!/usr/bin/env python3
"""Consolidate / verify the canonical final Full300 checkpoint set.

Two modes:
  * verify (default): assert a checkpoint root contains all 36 models, each with
    DONE + chunk_*_gen_300.pt whose combined basin coverage is exactly 531 and
    whose embedded generation == 300.  Fails loudly on pilot/gen-30 content.
  * --create: build symlinks from one or more run checkpoint dirs into the
    final set (canonical consolidation step, mirroring the 2026-07-30 manual
    `full300_final_36models` symlink layout).

Usage:
  python scripts/consolidate_final_checkpoints.py \
      --checkpoint-root checkpoints/full300_final_36models
  python scripts/consolidate_final_checkpoints.py \
      --checkpoint-root checkpoints/full300_final_36models \
      --create --source checkpoints/run_a checkpoints/run_b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from src.model_registry import NPARAM_INFO_36


def _chunk_files(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("chunk_*_gen_*.pt"), key=lambda p: p.name)


def inspect_model(model_dir: Path) -> dict:
    """Return provenance facts about one model checkpoint dir (no evaluation)."""
    chunks = _chunk_files(model_dir)
    if not chunks:
        raise RuntimeError(f"{model_dir}: no chunk_*_gen_*.pt files")
    generations = set()
    total_basins = 0
    seen = set()
    model_names = set()
    for ck in chunks:
        payload = torch.load(ck, map_location="cpu", weights_only=False)
        generations.add(int(payload.get("generation", -1)))
        model_names.add(str(payload.get("model", "?")))
        ids = payload["basin_ids"]
        ids_tuple = tuple(sorted(int(b) for b in ids))
        total_basins += len(ids)
        overlap = len(seen & set(ids_tuple))
        if overlap:
            raise RuntimeError(f"{ck}: overlapping basin ids with another chunk ({overlap})")
        seen |= set(ids_tuple)
    return {
        "model": model_dir.name,
        "generations": sorted(generations),
        "n_chunks": len(chunks),
        "n_basins": total_basins,
        "basin_unique": len(seen),
        "embedded_model_names": sorted(model_names),
    }


def verify_final_set(root: Path, expected_generation: int = 300, expected_basins: int = 531) -> dict:
    models = sorted(p.name for p in root.iterdir() if p.is_dir())
    missing = sorted(set(NPARAM_INFO_36) - set(models))
    extra = sorted(set(models) - set(NPARAM_INFO_36))
    if missing:
        raise RuntimeError(f"{root}: missing models {missing}")
    if extra:
        raise RuntimeError(f"{root}: unexpected model dirs {extra}")
    report = {}
    for model in sorted(NPARAM_INFO_36):
        mdir = root / model
        if not (mdir / "DONE").is_file():
            raise RuntimeError(f"{mdir}: missing DONE marker")
        info = inspect_model(mdir)
        if info["generations"] != [expected_generation]:
            raise RuntimeError(
                f"{mdir}: generations {info['generations']} != [{expected_generation}] — "
                f"refusing non-canonical (e.g. pilot/gen-30) checkpoint content"
            )
        if info["basin_unique"] != expected_basins:
            raise RuntimeError(
                f"{mdir}: basin coverage {info['basin_unique']} != {expected_basins} "
                f"(n_chunks={info['n_chunks']})"
            )
        report[model] = info
    print(json.dumps({"root": str(root), "models": len(report), "generation": expected_generation,
                      "basins": expected_basins, "passed": True}, indent=2))
    return report


def create_final_set(root: Path, sources: list[Path]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for src in sources:
        for model in sorted(p.name for p in src.iterdir() if p.is_dir()):
            dst = root / model
            if dst.exists():
                print(f"[skip] {model}: {dst} already exists")
                continue
            dst.symlink_to(src / model, target_is_directory=True)
            print(f"[link] {model} -> {src / model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--source", action="append", default=[], help="run checkpoint dir(s) for --create")
    parser.add_argument("--generation", type=int, default=300)
    args = parser.parse_args()
    root = Path(args.checkpoint_root)
    if not root.is_absolute():
        root = BENCHMARK_ROOT / root
    if args.create:
        create_final_set(root, [Path(s) for s in args.source])
    verify_final_set(root, expected_generation=args.generation)


if __name__ == "__main__":
    main()
