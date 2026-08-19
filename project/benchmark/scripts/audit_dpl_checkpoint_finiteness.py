#!/usr/bin/env python3
"""CPU-only audit of dPL checkpoint tensors for NaN/Inf corruption."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

EPOCH_RE = re.compile(r"epoch[_-](\d+)", re.IGNORECASE)


def epoch_number(path: Path) -> int:
    match = EPOCH_RE.search(path.name)
    return int(match.group(1)) if match else -1


def tensor_issues(value: Any, prefix: str = "") -> list[tuple[str, int]]:
    import torch

    issues: list[tuple[str, int]] = []
    if torch.is_tensor(value):
        bad = int((~torch.isfinite(value.detach().float())).sum().item())
        if bad:
            issues.append((prefix, bad))
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            issues.extend(tensor_issues(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            issues.extend(tensor_issues(child, f"{prefix}[{index}]"))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="CSV output path")
    args = parser.parse_args()

    import torch

    checkpoint_root = args.root / "checkpoints" / "dpl"
    output = args.output or args.root / "results" / "dpl" / "_summary" / "dpl_checkpoint_finiteness.csv"
    rows: list[dict[str, Any]] = []
    for model_dir in sorted(p for p in checkpoint_root.iterdir() if p.is_dir()):
        checkpoints = sorted(model_dir.rglob("epoch_*.pt"), key=epoch_number)
        first_bad = ""
        first_bad_keys = ""
        for path in checkpoints:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            issues = tensor_issues(checkpoint.get("parameterizer_state", checkpoint))
            if issues and not first_bad:
                first_bad = path.name
                first_bad_keys = ";".join(key for key, _ in issues[:8])
        rows.append({
            "model": model_dir.name,
            "checkpoint_count": len(checkpoints),
            "first_nonfinite_checkpoint": first_bad,
            "nonfinite_parameter_keys": first_bad_keys,
            "status": "nonfinite_parameters" if first_bad else "finite",
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)

    bad = [row for row in rows if row["status"] == "nonfinite_parameters"]
    print(f"output={output}")
    print(f"models={len(rows)} nonfinite_models={len(bad)}")
    for row in bad:
        print(f"NONFINITE {row['model']} first={row['first_nonfinite_checkpoint']} keys={row['nonfinite_parameter_keys']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
