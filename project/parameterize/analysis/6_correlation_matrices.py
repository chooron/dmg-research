"""Step 6: compute and export correlation matrices for each run.

For every model/loss/seed combination this script exports:
- csv matrix
- npz matrix

Supported methods default to: spearman, pearson, kendall.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data, parse_corr_methods
from project.parameterize.analysis.correlation_analysis import build_correlation_long, export_correlation_matrices


def main() -> None:
    parser = build_parser("Build correlation matrices for each run.")
    parser.add_argument(
        "--corr-methods",
        default="spearman,pearson,kendall",
        help="Comma-separated correlation methods.",
    )
    args = parser.parse_args()
    data = load_analysis_data(
        config_path=args.config,
        outputs_root=args.outputs_root,
        analysis_root=args.analysis_root,
        device=args.device,
        parameter_csv=args.parameter_csv,
        attribute_csv=args.attribute_csv,
    )
    if "params_long" not in data or "attributes" not in data:
        raise ValueError("--parameter-csv and --attribute-csv are required for correlation matrix export.")
    methods = parse_corr_methods(args.corr_methods)
    corr_tables = build_correlation_long(data["params_long"], data["attributes"], methods=methods)
    paths = export_correlation_matrices(corr_tables, data["stability_output_dirs"]["correlation_matrices"])
    for name, path in paths.items():
        print(name, path)


if __name__ == "__main__":
    main()
