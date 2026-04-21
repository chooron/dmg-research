"""Run the full seed/loss stability analysis pipeline.

This is the main entrypoint for the numbered analysis scripts. It reuses the
shared implementation modules under ``project.parameterize.analysis`` and writes
all outputs to ``outputs/analysis/stability_stats`` by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, parse_corr_methods, parse_metric_names
from project.parameterize.analysis.pipeline import run_all


def main() -> None:
    parser = build_parser("Run the full stability-analysis pipeline.")
    parser.add_argument("--metrics", default="nse,kge,bias_abs")
    parser.add_argument("--corr-methods", default="spearman,pearson,kendall")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    outputs = run_all(
        config_path=args.config,
        outputs_root=args.outputs_root,
        analysis_root=args.analysis_root,
        device=args.device,
        metric_names=parse_metric_names(args.metrics),
        corr_methods=parse_corr_methods(args.corr_methods),
        top_k=args.top_k,
        parameter_csv=args.parameter_csv,
        attribute_csv=args.attribute_csv,
    )
    print("report:", outputs["report_path"])


if __name__ == "__main__":
    main()
