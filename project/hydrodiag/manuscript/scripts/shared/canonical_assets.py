"""Canonical manuscript asset and provenance helpers.

This module is deliberately small: it records figure/table provenance and
prevents incomplete TGD2 R4 products from being written to final filenames.
It does not train models or recompute scientific estimands.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TGD2_TRAINING_REL = Path("results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2")


def tgd2_training_status(results_root: Path) -> dict[str, dict[str, Any]]:
    """Return completion evidence for the three canonical dPL TGD2 seeds."""
    base = (
        results_root
        if (results_root / "dpl_camels_531_lite_v3_tgd2_dpl_audited").exists()
        else results_root / "results"
    )
    out: dict[str, dict[str, Any]] = {}
    for seed in (42, 123, 2026):
        seed_dir = (
            base
            / "dpl_camels_531_lite_v3_tgd2_dpl_audited"
            / "XAJ_TGD2"
            / f"seed_{seed}"
        )
        history = sorted(seed_dir.glob("*history*")) if seed_dir.exists() else []
        complete = (seed_dir / "COMPLETE").exists() if seed_dir.exists() else False
        final_summary = (
            seed_dir / "basin_final_summary.csv"
        ).exists() if seed_dir.exists() else False
        normalized = (
            seed_dir / "best_parameters_normalized.npz"
        ).exists() if seed_dir.exists() else False
        physical = (
            seed_dir / "best_parameters_physical.npz"
        ).exists() if seed_dir.exists() else False
        report = (seed_dir / "report.md").exists() if seed_dir.exists() else False
        out[str(seed)] = {
            "seed_dir": str(seed_dir),
            "complete": complete,
            "final_summary": final_summary,
            "normalized_parameters": normalized,
            "physical_parameters": physical,
            "report": report,
            "history_files": [str(p) for p in history],
            "formal_complete": all(
                (complete, final_summary, normalized, physical, report)
            ),
        }
    return out


def assert_tgd2_final_allowed(results_root: Path) -> None:
    """Raise unless all canonical dPL TGD2 training provenance is complete."""
    status = tgd2_training_status(results_root)
    incomplete = [seed for seed, row in status.items() if not row["formal_complete"]]
    if incomplete:
        raise RuntimeError(
            "TGD2_PENDING: final R4 three-structure asset generation is blocked for "
            f"seed(s) {', '.join(incomplete)}. Use an explicit interim output path."
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_output(out_dir: Path, filename: str, *, interim: bool) -> Path:
    """Resolve a non-overwriting final/interim filename."""
    if interim:
        stem, suffix = Path(filename).stem, Path(filename).suffix
        filename = f"{stem}_interim{suffix}"
    return out_dir / filename
