"""Step 3: compute per-basin per-parameter variance across random seeds.

Parameter values are first normalized into [0, 1] using HBV parameter bounds.
The script then computes:
- variance across seeds
- mean absolute difference across seeds

Outputs include wide csv files with one row per basin and one column per
parameter for each model/loss combination.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data
from project.parameterize.analysis.parameter_analysis import compute_seed_parameter_variance, variance_long_to_wide


def main() -> None:
    parser = build_parser("Compute parameter variance across seeds.")
    args = parser.parse_args()
    data = load_analysis_data(
        config_path=args.config,
        outputs_root=args.outputs_root,
        analysis_root=args.analysis_root,
        device=args.device,
        parameter_csv=args.parameter_csv,
        attribute_csv=args.attribute_csv,
    )
    if "params_long" not in data:
        raise ValueError("--parameter-csv is required for parameter variance analysis.")
    variance_long = compute_seed_parameter_variance(data["params_long"], data["parameter_bounds"])
    print(variance_long.head().to_string(index=False))
    print("wide_tables:", len(variance_long_to_wide(variance_long, "variance_unit")))


if __name__ == "__main__":
    main()
