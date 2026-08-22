"""Build 531-only LORO held-out groups from the frozen region definitions."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent


def load_manifest(path: Path) -> list[int]:
    value = ast.literal_eval(path.read_text())
    ids = [int(x) for x in value]
    if len(ids) != 531 or len(set(ids)) != 531:
        raise ValueError(f"Expected 531 unique basin IDs, got {len(ids)}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "531sub_id.txt"))
    parser.add_argument("--source-dir", default=str(REPO_ROOT / "data" / "basin_groups"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "manifests" / "loro_531"),
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_ids = load_manifest(manifest_path)
    manifest_set = set(manifest_ids)

    rows: list[dict[str, object]] = []
    held_out_sets: list[set[int]] = []
    for region in range(7):
        group_id = 11 + region
        source = source_dir / f"group_{group_id}.npy"
        if not source.exists():
            raise FileNotFoundError(source)
        source_ids = {int(x) for x in np.load(source, allow_pickle=True).tolist()}
        held_out = sorted(source_ids & manifest_set)
        if not held_out:
            raise ValueError(f"group_{group_id} has no 531 intersection")
        np.save(output_dir / f"group_{group_id}.npy", np.asarray(held_out, dtype=np.int64))
        held_out_sets.append(set(held_out))
        rows.append(
            {
                "region": region,
                "source_group_id": group_id,
                "source_count": len(source_ids),
                "held_out_count_531": len(held_out),
                "output": str(output_dir / f"group_{group_id}.npy"),
            }
        )

    overlap = set().union(*(held_out_sets[i] & held_out_sets[j] for i in range(7) for j in range(i + 1, 7)))
    coverage = set().union(*held_out_sets)
    summary = {
        "manifest": str(manifest_path),
        "manifest_basin_count": len(manifest_ids),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "regions": rows,
        "overlap_count": len(overlap),
        "covered_basin_count": len(coverage),
        "uncovered_basin_count": len(manifest_set - coverage),
        "uncovered_basin_ids": sorted(manifest_set - coverage),
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
