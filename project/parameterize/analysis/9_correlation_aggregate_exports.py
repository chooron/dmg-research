"""Step 9: export seed-aggregated correlation summaries.

This script saves:
- mean/std/variance correlation tables
- top relationship tables
for every requested correlation method.
"""

from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data, parse_corr_methods
from project.parameterize.analysis.correlation_analysis import aggregate_correlation_exports, build_correlation_long


def main() -> None:
    parser = build_parser("Export aggregated correlation summaries.")
    parser.add_argument("--corr-methods", default="spearman,pearson,kendall")
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
        raise ValueError("--parameter-csv and --attribute-csv are required for aggregated correlation export.")
    methods = parse_corr_methods(args.corr_methods)
    corr_tables = build_correlation_long(data["params_long"], data["attributes"], methods=methods)
    combined = pd.concat([table.assign(method=method) for method, table in corr_tables.items()], ignore_index=True)
    outputs = aggregate_correlation_exports(combined)
    for name, frame in outputs.items():
        print(name, frame.head().to_string(index=False))


if __name__ == "__main__":
    main()
