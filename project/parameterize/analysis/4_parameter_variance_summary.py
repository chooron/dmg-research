"""Step 4: summarize seed-variance csv outputs for parameter stability.

The main summary statistics are:
- mean variance
- median variance
- p90 variance
- mean absolute seed difference
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data
from project.parameterize.analysis.parameter_analysis import (
    compute_seed_parameter_variance,
    summarize_seed_parameter_variance,
)


def main() -> None:
    parser = build_parser("Summarize parameter variance across seeds.")
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
    summary = summarize_seed_parameter_variance(variance_long)
    print(summary["seed_parameter_variance_by_model_loss"].to_string(index=False))


if __name__ == "__main__":
    main()
