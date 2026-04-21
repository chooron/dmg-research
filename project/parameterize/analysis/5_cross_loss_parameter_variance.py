"""Step 5: compare parameter variance across different loss functions.

Two modes are supported in the internal implementation:
- pooled: use loss * seed combinations directly as samples
- seed-first: average within seed first, then compare across losses

The pooled statistic is the default main comparison target.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data
from project.parameterize.analysis.parameter_analysis import (
    compute_cross_loss_parameter_variance,
    summarize_cross_loss_parameter_variance,
)


def main() -> None:
    parser = build_parser("Compare parameter variance across loss functions.")
    parser.add_argument(
        "--mode",
        default="pooled",
        choices=("pooled", "seed-first"),
        help="Cross-loss comparison mode.",
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
    if "params_long" not in data:
        raise ValueError("--parameter-csv is required for cross-loss parameter variance analysis.")
    mode = "seed-first" if args.mode == "seed-first" else "pooled"
    cross_loss = compute_cross_loss_parameter_variance(data["params_long"], data["parameter_bounds"], mode=mode)
    summary = summarize_cross_loss_parameter_variance(cross_loss)
    print(summary["cross_loss_parameter_variance_by_model"].to_string(index=False))


if __name__ == "__main__":
    main()
