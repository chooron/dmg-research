"""Step 8: quantify how strong parameter-attribute correlations vary across losses.

The default output reports pooled variability over seed*loss combinations and
seed-first variability where correlations are averaged within each seed first.
"""

from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data, parse_corr_methods
from project.parameterize.analysis.correlation_analysis import build_correlation_long, compute_loss_correlation_stability


def main() -> None:
    parser = build_parser("Compute cross-loss correlation stability.")
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
        raise ValueError("--parameter-csv and --attribute-csv are required for loss correlation stability analysis.")
    methods = parse_corr_methods(args.corr_methods)
    corr_tables = build_correlation_long(data["params_long"], data["attributes"], methods=methods)
    combined = pd.concat([table.assign(method=method) for method, table in corr_tables.items()], ignore_index=True)
    outputs = compute_loss_correlation_stability(combined, top_k=args.top_k)
    print(outputs["correlation_loss_stability_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
