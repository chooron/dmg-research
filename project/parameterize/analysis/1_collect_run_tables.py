"""Step 1: collect base long tables for metrics, parameters, and attributes.

Outputs:
- run inventory csv
- metric run manifest csv
- parameter run manifest csv
- basin attribute csv
- metrics long csv
- params long csv
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data
from project.parameterize.analysis.pipeline import run_collect_run_tables


def main() -> None:
    parser = build_parser("Collect base metric/parameter/attribute tables.")
    args = parser.parse_args()
    data = load_analysis_data(
        config_path=args.config,
        outputs_root=args.outputs_root,
        analysis_root=args.analysis_root,
        device=args.device,
        parameter_csv=args.parameter_csv,
        attribute_csv=args.attribute_csv,
    )
    paths = run_collect_run_tables(data)
    for name, path in paths.items():
        print(name, path)


if __name__ == "__main__":
    main()
