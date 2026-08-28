"""Canonical Full300 checkpoint provenance guard.

Hard guard against the pilot/gen-30 misuse class: canonical evaluation and
downstream aligned evaluation MUST read gen-300, best-of-10, 531-basin final
checkpoints.  Any checkpoint root that contains older generations (e.g. the
12:05 pilot run) is rejected loudly — no silent fallback, no auto-downgrade.
"""
from __future__ import annotations
from collections.abc import Iterable
import json
import re
from pathlib import Path

import torch

from src.model_registry import NPARAM_INFO_36


def _chunk_generation_files(model_dir: Path) -> dict[int, list[Path]]:
    by_gen: dict[int, list[Path]] = {}
    for ck in model_dir.glob("chunk_*_gen_*.pt"):
        m = re.fullmatch(r"chunk_(\d+)_gen_(\d+)\.pt", ck.name)
        if m:
            by_gen.setdefault(int(m.group(2)), []).append(ck)
    return by_gen


def inspect_checkpoint_generations(model_dir: Path) -> dict:
    """List what generations actually exist in a model checkpoint dir."""
    by_gen = _chunk_generation_files(model_dir)
    return {gen: sorted(p.name for p in files) for gen, files in sorted(by_gen.items())}


def validate_canonical_checkpoint(
    model_dir: Path,
    *,
    model_name: str | None = None,
    required_generation: int = 300,
    required_basins: int = 531,
    required_basin_ids: Iterable[int] | None = None,
    require_done: bool = True,
    require_manifest_dim: bool = True,
) -> dict:
    """Validate one model dir is a canonical final Full300 checkpoint.

    Raises RuntimeError with an actionable message on any violation:
      * missing DONE marker;
      * no chunk_*_gen_300.pt (and a loud hint if only pilot/gen-30 content exists);
      * embedded payload generation != required_generation;
      * basin coverage != required_basins (after merging all chunks, no overlap);
      * basin IDs do not match required_basin_ids, when supplied;
      * registry dimension mismatch with the frozen manifest (when provided).
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise RuntimeError(f"checkpoint dir not found: {model_dir}")
    if model_name is not None and model_dir.name != model_name:
        raise RuntimeError(f"checkpoint dir name {model_dir.name!r} != model_name {model_name!r}")
    if require_done and not (model_dir / "DONE").is_file():
        raise RuntimeError(f"{model_dir}: missing DONE marker (incomplete training)")

    by_gen = _chunk_generation_files(model_dir)
    if not by_gen:
        raise RuntimeError(f"{model_dir}: no chunk_*_gen_*.pt files found")
    present = sorted(by_gen)
    if required_generation not in present:
        hint = ""
        if present and max(present) < required_generation:
            hint = (f" only generation(s) {present} present — this looks like a pilot/"
                    f"intermediate checkpoint (pilot run used gen-30/best-of-5); "
                    f"canonical evaluation requires gen-{required_generation}.")
        raise RuntimeError(
            f"{model_dir}: generation {required_generation} checkpoint not found.{hint} "
            f"Refusing to evaluate non-canonical content."
        )

    targets = by_gen[required_generation]
    payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in targets]
    for p in payloads:
        gen = int(p.get("generation", -1))
        if gen != required_generation:
            raise RuntimeError(f"{model_dir}: embedded generation {gen} != {required_generation}")
        if model_name is not None:
            emb = str(p.get("model", ""))
            if emb and emb != model_name:
                raise RuntimeError(f"{model_dir}: embedded model {emb!r} != {model_name!r}")

    seen: set[int] = set()
    for p in payloads:
        ids = tuple(sorted(int(b) for b in p["basin_ids"]))
        overlap = len(seen & set(ids))
        if overlap:
            raise RuntimeError(f"{model_dir}: overlapping basin ids across chunks ({overlap})")
        seen |= set(ids)
    if len(seen) != required_basins:
        raise RuntimeError(
            f"{model_dir}: basin coverage {len(seen)} != {required_basins} "
            f"(n_chunks={len(targets)}) — truncated/partial checkpoint"
        )
    if required_basin_ids is not None:
        expected = {int(basin_id) for basin_id in required_basin_ids}
        if seen != expected:
            missing = sorted(expected - seen)
            unexpected = sorted(seen - expected)
            raise RuntimeError(
                f"{model_dir}: basin ID coverage mismatch; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

    if require_manifest_dim:
        dim = NPARAM_INFO_36.get(model_dir.name) if model_dir.name in NPARAM_INFO_36 else None
        if dim is not None:
            latent_dim = payloads[0]["solver"]["state"]["best_latent"].shape[-1]
            if latent_dim != dim:
                raise RuntimeError(
                    f"{model_dir}: latent dimension {latent_dim} != registry dimension {dim} "
                    f"— parameter schema mismatch"
                )

    return {
        "model": model_dir.name if model_name is None else model_name,
        "generation": required_generation,
        "n_chunks": len(targets),
        "n_basins": len(seen),
        "passed": True,
    }


def validate_manifest_consistency(manifest_path: Path, model_dir: Path) -> None:
    """Optional: compare the frozen manifest's registry dims with the checkpoint."""
    manifest = json.loads(Path(manifest_path).read_text())
    expected_dim = manifest["model_registry"].get(model_dir.name)
    payload = torch.load(sorted(model_dir.glob("chunk_*_gen_*.pt"))[0], map_location="cpu",
                         weights_only=False)
    actual_dim = payload["solver"]["state"]["best_latent"].shape[-1]
    if expected_dim is not None and actual_dim != expected_dim:
        raise RuntimeError(
            f"{model_dir.name}: latent dim {actual_dim} != manifest dim {expected_dim} "
            f"(manifest {manifest_path})"
        )
