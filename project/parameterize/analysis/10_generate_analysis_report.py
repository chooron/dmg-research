"""Step 10: build one markdown report that summarizes all analysis outputs.

This script is intended to be run after the metric, parameter, and correlation
analysis steps have completed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser
from project.parameterize.analysis.pipeline import run_all


def main() -> None:
    parser = build_parser("Run the full analysis and generate one markdown report.")
    parser.add_argument("--metrics", default="nse,kge,bias_abs")
    parser.add_argument("--corr-methods", default="spearman,pearson,kendall")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    outputs = run_all(
        config_path=args.config,
        outputs_root=args.outputs_root,
        analysis_root=args.analysis_root,
        device=args.device,
        parameter_csv=args.parameter_csv,
        attribute_csv=args.attribute_csv,
    )
    print(outputs["report_path"])


if __name__ == "__main__":
    main()
