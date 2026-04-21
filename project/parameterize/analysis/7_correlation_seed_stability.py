"""Step 7: quantify how strong parameter-attribute correlations vary across seeds.

Default strategy:
- for each parameter, select its top-k attributes by mean |correlation|
- compute variance/range/absolute-difference across seeds
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data, parse_corr_methods
from project.parameterize.analysis.correlation_analysis import build_correlation_long, compute_seed_correlation_stability


def main() -> None:
    parser = build_parser("Compute cross-seed correlation stability.")
    parser.add_argument("--corr-methods", default="spearman,pearson,kendall")
    parser.add_argument("--top-k", type=int, default=10)
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
        raise ValueError("--parameter-csv and --attribute-csv are required for seed correlation stability analysis.")
    methods = parse_corr_methods(args.corr_methods)
    corr_tables = build_correlation_long(data["params_long"], data["attributes"], methods=methods)
    combined = pd.concat([table.assign(method=method) for method, table in corr_tables.items()], ignore_index=True)
    outputs = compute_seed_correlation_stability(combined, top_k=args.top_k)
    print(outputs["correlation_seed_stability_summary"].to_string(index=False))


if __name__ == "__main__":
    import pandas as pd

    main()
