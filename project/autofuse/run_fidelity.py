"""CLI for bounded reference-vs-dFUSE fidelity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fidelity import compare_mothers, explicit_scan, explicit_stress_scan, regress_all_78, solver_scan, summarize_explicit_scan, synthetic_forcing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, help="path to the pinned FUSE executable")
    parser.add_argument("--mode", choices=("mothers", "solver", "explicit", "all"), default="mothers")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--substeps", type=int, nargs="+", default=[1, 2, 4, 8, 12, 24, 48])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    forcing = synthetic_forcing(args.steps)
    result: dict[str, object] = {
        "executable": str(Path(args.executable).resolve()),
        "mode": args.mode,
        "steps": args.steps,
        "implicit_iterations": args.iterations,
    }
    if args.mode in ("mothers", "all"):
        result["mothers"] = compare_mothers(args.executable, forcing, implicit_iterations=args.iterations)
    if args.mode in ("solver", "all"):
        result["solver_scan"] = solver_scan(args.executable, forcing)
    if args.mode == "explicit":
        explicit_rows = explicit_scan(
            args.executable,
            forcing,
            n_substeps=args.substeps,
            implicit_iterations=16,
        )
        result["explicit_scan"] = explicit_rows
        result["explicit_summary"] = summarize_explicit_scan(explicit_rows)
        result["solver"] = "fixed_substep_explicit_euler"
        result["n_substeps"] = list(args.substeps)
        result["reference"] = {
            "source_ref": "v1.0_MMpaper",
            "source_commit": "e6e23a4fc4ff4019bcab55f14537ea43b9525967",
            "executable_sha256": explicit_rows[0]["reference_executable_sha256"] if explicit_rows else None,
        }
        result["mother_summary"] = summarize_explicit_scan(
            [row for row in explicit_rows if row["model_id"] in (2, 108, 178, 210)]
        )
        result["summary_78"] = result["explicit_summary"]
        result["stress_5000_mm_day"] = explicit_stress_scan(args.substeps)
        result["formal_training_started"] = False
    if args.mode == "all":
        result["structures_78"] = regress_all_78(args.executable, forcing, implicit_iterations=args.iterations)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
