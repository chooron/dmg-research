"""Run the focused distributional-vs-deterministic cross-loss stability test."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data
from project.parameterize.analysis.focused_cross_loss_analysis import (
    load_existing_relationship_classes,
    run_focused_cross_loss_analysis,
)


def main() -> None:
    parser = build_parser("Run the focused distributional-vs-deterministic cross-loss stability test.")
    parser.add_argument(
        "--relationship-classes-csv",
        default="project/parameterize/outputs/analysis/stability_stats/correlation_summaries/relationship_classes.csv",
        help="Existing relationship_classes.csv produced by the main relationship analysis.",
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
        raise ValueError("--parameter-csv and --attribute-csv are required for the focused cross-loss test.")

    relationship_classes = load_existing_relationship_classes(args.relationship_classes_csv)
    _, paths = run_focused_cross_loss_analysis(
        params_long=data["params_long"],
        attributes=data["attributes"],
        relationship_classes=relationship_classes,
        output_dir=data["stability_output_dirs"]["correlation_summaries"],
        reports_dir=data["stability_output_dirs"]["reports"],
    )
    print(paths["report"])


if __name__ == "__main__":
    main()
