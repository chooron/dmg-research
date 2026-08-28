"""Authoritative specification and normalization for the 15 shared XAJ parameters."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from r2_config import BOUNDS_FILE, RESULTS_DIR

# The 15 canonical shared XAJ parameters in strict order
SHARED_15_PARAMETERS = (
    "xaj_k",
    "xaj_b",
    "xaj_im",
    "xaj_um",
    "xaj_lm",
    "xaj_dm",
    "xaj_c",
    "xaj_sm",
    "xaj_ex",
    "xaj_ki",
    "xaj_kg",
    "xaj_ci",
    "xaj_cg",
    "xaj_a",
    "xaj_theta",
)

PARAMETER_METADATA = {
    "xaj_k": {"symbol": "k", "display": "k", "lower": 0.5, "upper": 2.0, "unit": "-", "process": "soil", "description": "Ratio of potential ET to reference crop evaporation"},
    "xaj_b": {"symbol": "b", "display": "b", "lower": 0.1, "upper": 2.0, "unit": "-", "process": "soil", "description": "Exponent of tension water capacity curve"},
    "xaj_im": {"symbol": "im", "display": "im", "lower": 0.0, "upper": 0.3, "unit": "-", "process": "soil", "description": "Impervious area fraction"},
    "xaj_um": {"symbol": "um", "display": "um", "lower": 5.0, "upper": 50.0, "unit": "mm", "process": "soil", "description": "Upper layer tension water capacity"},
    "xaj_lm": {"symbol": "lm", "display": "lm", "lower": 20.0, "upper": 200.0, "unit": "mm", "process": "soil", "description": "Lower layer tension water capacity"},
    "xaj_dm": {"symbol": "dm", "display": "dm", "lower": 20.0, "upper": 200.0, "unit": "mm", "process": "soil", "description": "Deep layer tension water capacity"},
    "xaj_c": {"symbol": "c", "display": "c", "lower": 0.05, "upper": 0.3, "unit": "-", "process": "soil", "description": "Deep layer evaporation coefficient"},
    "xaj_sm": {"symbol": "sm", "display": "sm", "lower": 5.0, "upper": 100.0, "unit": "mm", "process": "routing", "description": "Areal mean free water capacity of surface layer"},
    "xaj_ex": {"symbol": "ex", "display": "ex", "lower": 0.1, "upper": 2.0, "unit": "-", "process": "routing", "description": "Exponent of free water capacity curve"},
    "xaj_ki": {"symbol": "ki", "display": "ki", "lower": 0.0, "upper": 0.7, "unit": "1/day", "process": "routing", "description": "Outflow coefficient for interflow"},
    "xaj_kg": {"symbol": "kg", "display": "kg", "lower": 0.0, "upper": 0.7, "unit": "1/day", "process": "routing", "description": "Outflow coefficient for groundwater"},
    "xaj_ci": {"symbol": "ci", "display": "ci", "lower": 0.1, "upper": 1.0, "unit": "-", "process": "routing", "description": "Recession constant for interflow reservoir"},
    "xaj_cg": {"symbol": "cg", "display": "cg", "lower": 0.9, "upper": 1.0, "unit": "-", "process": "routing", "description": "Recession constant for groundwater reservoir"},
    "xaj_a": {"symbol": "a", "display": "a (UH shape)", "lower": 0.0, "upper": 2.9, "unit": "-", "process": "routing", "description": "Gamma-UH shape parameter"},
    "xaj_theta": {"symbol": "theta", "display": "theta (UH scale)", "lower": 0.0, "upper": 6.5, "unit": "day", "process": "routing", "description": "Gamma-UH scale parameter"},
}

# Structure parameter layouts
STRUCTURE_PARAM_LAYOUTS = {
    "Base": {
        "total_params": 15,
        "param_names": list(SHARED_15_PARAMETERS),
        "shared_indices": {p: i for i, p in enumerate(SHARED_15_PARAMETERS)},
    },
    "CN": {
        "total_params": 17,
        "param_names": ["cn_ctg", "cn_kf"] + list(SHARED_15_PARAMETERS),
        "shared_indices": {p: i + 2 for i, p in enumerate(SHARED_15_PARAMETERS)},
    },
    "TGD": {
        "total_params": 17,
        "param_names": ["tgd_tau_warm", "tgd_delta_tau_cold"] + list(SHARED_15_PARAMETERS),
        "shared_indices": {p: i + 2 for i, p in enumerate(SHARED_15_PARAMETERS)},
    },
}


def get_lowers_and_uppers() -> Tuple[np.ndarray, np.ndarray]:
    """Return numpy float64 arrays of lower and upper bounds for the 15 shared parameters."""
    lowers = np.array([PARAMETER_METADATA[p]["lower"] for p in SHARED_15_PARAMETERS], dtype=np.float64)
    uppers = np.array([PARAMETER_METADATA[p]["upper"] for p in SHARED_15_PARAMETERS], dtype=np.float64)
    return lowers, uppers


def normalize_parameters(phys_array: np.ndarray) -> np.ndarray:
    """Normalize physical parameter array of shape (..., 15) to [0, 1] range."""
    lowers, uppers = get_lowers_and_uppers()
    return (phys_array - lowers) / (uppers - lowers)


def physical_from_normalized(norm_array: np.ndarray) -> np.ndarray:
    """Denormalize normalized array of shape (..., 15) to physical range."""
    lowers, uppers = get_lowers_and_uppers()
    return lowers + norm_array * (uppers - lowers)


def build_authoritative_specs_table(output_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Build and write the authoritative 15-parameter specification CSV."""
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, p in enumerate(SHARED_15_PARAMETERS):
        meta = PARAMETER_METADATA[p]
        rows.append({
            "shared_index": idx,
            "parameter_name": p,
            "symbol": meta["symbol"],
            "display_name": meta["display"],
            "base_index": STRUCTURE_PARAM_LAYOUTS["Base"]["shared_indices"][p],
            "cn_index": STRUCTURE_PARAM_LAYOUTS["CN"]["shared_indices"][p],
            "tgd_index": STRUCTURE_PARAM_LAYOUTS["TGD"]["shared_indices"][p],
            "lower_bound": meta["lower"],
            "upper_bound": meta["upper"],
            "range": meta["upper"] - meta["lower"],
            "unit": meta["unit"],
            "process": meta["process"],
            "description": meta["description"],
            "same_semantic_across_structures": True,
            "same_bounds_across_structures": True,
        })

    fields = [
        "shared_index", "parameter_name", "symbol", "display_name",
        "base_index", "cn_index", "tgd_index",
        "lower_bound", "upper_bound", "range", "unit", "process", "description",
        "same_semantic_across_structures", "same_bounds_across_structures",
    ]

    out_file = out_dir / "authoritative_15_parameter_specs.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return rows


if __name__ == "__main__":
    specs = build_authoritative_specs_table()
    print(f"Authoritative 15 shared parameter specs written: {len(specs)} parameters.")
