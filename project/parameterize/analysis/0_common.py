"""Step 0: inspect resolved analysis inputs and output roots.

Usage:
    python -m project.parameterize.analysis.0_common --config ...

This helper does not compute statistics. It only resolves:
- outputs root
- analysis root
- discovered runs
- available models / losses / seeds
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project.parameterize.analysis.common import build_parser, load_analysis_data


def main() -> None:
    parser = build_parser("Inspect analysis roots and discovered runs.")
    args = parser.parse_args()
    data = load_analysis_data(
        config_path=args.config,
        outputs_root=args.outputs_root,
        analysis_root=args.analysis_root,
        device=args.device,
        parameter_csv=args.parameter_csv,
        attribute_csv=args.attribute_csv,
    )
    print("outputs_root:", data["outputs_root"])
    print("analysis_root:", data["stability_analysis_root"])
    print("models:", data["model_order"])
    print("losses:", data["loss_order"])
    print("seeds:", data["seed_order"])
    print("run_count:", len(data["runs"]))
    print("has_parameter_csv:", "params_long" in data)
    print("has_attribute_csv:", "attributes" in data)


if __name__ == "__main__":
    main()
