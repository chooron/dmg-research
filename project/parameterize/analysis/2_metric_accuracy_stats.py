"""Step 2: summarize model accuracy across seed/loss combinations.

This script computes seed-wise and model/loss-wise summaries for selected
metrics such as NSE and KGE.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data, parse_metric_names
from project.parameterize.analysis.pipeline import run_metric_accuracy


def main() -> None:
    parser = build_parser("Summarize NSE/KGE/bias-style metric accuracy.")
    parser.add_argument(
        "--metrics",
        default="nse,kge,bias_abs",
        help="Comma-separated metric names to summarize.",
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
    summary, paths = run_metric_accuracy(data, metric_names=parse_metric_names(args.metrics))
    print(summary["metrics_by_model_loss"].to_string(index=False))
    for name, path in paths.items():
        print(name, path)


if __name__ == "__main__":
    main()
