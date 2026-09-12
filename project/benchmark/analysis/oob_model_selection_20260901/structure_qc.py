#!/usr/bin/env python3
"""Objective structural descriptors for OOB-set coverage QC only.

These descriptors are not selection criteria.  They come from the frozen model
registry/core definitions and do not run a hydrological simulation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BENCHMARK = REPO / "project/benchmark"
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]
from dmotpy.models.registry import STATE_INFO, STFN_INFO  # noqa: E402
from src.model_registry import get_spec  # noqa: E402
from selection_common import OUT, write_csv  # noqa: E402

SNOW_MARKERS = {"tt", "tti", "ttm", "cfr", "cfmax", "ddf", "tcrit", "whc"}


def descriptors(models: list[str], population: str) -> pd.DataFrame:
    rows = []
    for model in models:
        spec = get_spec(model, device="cpu")
        snow_parameters = [name for name in spec.parameter_names if name.lower() in SNOW_MARKERS]
        step = STFN_INFO[model]
        rows.append({
            "population": population,
            "model": model,
            "parameter_count": spec.dimension,
            "state_count": int(STATE_INFO[model]),
            "storage_state_count": int(STATE_INFO[model]),
            "snow_process": bool(snow_parameters),
            "snow_marker_parameters": ";".join(snow_parameters),
            "routing_form": spec.routed_kind,
            "runoff_generation_definition": f"{step.__module__}.{step.__name__}",
            "descriptor_source": "frozen dmotpy.models.core registry: PARAM_INFO/STATE_INFO/STFN_INFO and benchmark routed_kind",
        })
    return pd.DataFrame(rows)


def coverage_summary(descriptor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population, frame in descriptor.groupby("population", sort=True):
        rows.append({
            "population": population,
            "model_count": len(frame),
            "mopex_count": int(frame.model.str.startswith("mopex").sum()),
            "parameter_count_min": int(frame.parameter_count.min()),
            "parameter_count_max": int(frame.parameter_count.max()),
            "parameter_count_unique": ";".join(map(str, sorted(frame.parameter_count.unique()))),
            "state_count_min": int(frame.state_count.min()),
            "state_count_max": int(frame.state_count.max()),
            "state_count_unique": ";".join(map(str, sorted(frame.state_count.unique()))),
            "snow_process_count": int(frame.snow_process.sum()),
            "routing_forms": ";".join(f"{key}:{value}" for key, value in frame.routing_form.value_counts().sort_index().items()),
            "runoff_definitions_unique": int(frame.runoff_generation_definition.nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    revised = pd.read_csv(OUT / "OOB_REVISED_SELECTION_FEATURES.csv")
    original = pd.read_csv(OUT.parent / "oob_model_selection_20260901/OOB_MODEL_SELECTION_FEATURES.csv")
    revised_models = revised[revised.selected_revised_primary_8].sort_values("primary_role").model.tolist()
    original_models = original[original.selected_primary_8].sort_values("primary_role").model.tolist()
    descriptor = pd.concat([descriptors(original_models, "ORIGINAL_PRIMARY_8"), descriptors(revised_models, "REVISED_PRIMARY_8")], ignore_index=True)
    write_csv(OUT / "OOB_STRUCTURAL_MODEL_DESCRIPTORS.csv", descriptor)
    summary = coverage_summary(descriptor)
    write_csv(OUT / "OOB_STRUCTURAL_COVERAGE_QC.csv", summary)
    (OUT / "OOB_STRUCTURAL_COVERAGE_QC.md").write_text("# Structural coverage QC\n\nThese descriptors are auxiliary checks only and were not used as selection criteria. They are extracted from the frozen model registry/core definitions without simulation. See `OOB_STRUCTURAL_MODEL_DESCRIPTORS.csv` and `OOB_STRUCTURAL_COVERAGE_QC.csv`.\n")
    print("Structural QC complete", len(descriptor), len(summary))


if __name__ == "__main__":
    main()
