"""Public entrypoints for generating the WRR figure suite."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from project.parameterize.figures.common import write_manifest
from project.parameterize.figures.data_loading import build_data_dict
from project.parameterize.figures.reporting import render_markdown_report


FIGURE_DIR = Path(__file__).resolve().parent


def _figure_paths() -> list[Path]:
    paths = []
    for idx in range(1, 15):
        matches = sorted(FIGURE_DIR.glob(f"{idx}_*.py"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one figure script for index {idx}, found {len(matches)}.")
        paths.append(matches[0])
    return paths


def load_figure_module(path: Path) -> ModuleType:
    module_name = f"project.parameterize.figures._dynamic_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load figure module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_all_figures(
    data_dict: dict[str, Any],
    output_root: str | Path | None = None,
    save_formats: tuple[str, ...] = ("png", "pdf"),
    figure_numbers: list[int] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_root).resolve() if output_root else Path(data_dict["analysis_root"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(figure_numbers or range(1, 15))
    generated: dict[str, Any] = {}
    for path in _figure_paths():
        figure_idx = int(path.stem.split("_", maxsplit=1)[0])
        if figure_idx not in selected:
            continue
        module = load_figure_module(path)
        result = module.generate_figure(data_dict, output_dir, formats=save_formats)
        generated[f"{figure_idx:02d}"] = {
            "name": path.stem,
            "outputs": {suffix: str(file_path) for suffix, file_path in result.items()},
        }

    manifest = {
        "analysis_root": str(output_dir),
        "generated_figures": generated,
    }
    manifest_path = write_manifest(output_dir, manifest)
    generated["manifest"] = str(manifest_path)
    return generated


def build_data_and_generate(
    config_path: str,
    outputs_root: str | None = None,
    analysis_root: str | None = None,
    stochastic_samples: int = 100,
    device: str = "cpu",
    save_formats: tuple[str, ...] = ("png", "pdf"),
    figure_numbers: list[int] | None = None,
) -> dict[str, Any]:
    data_dict = build_data_dict(
        config_path=config_path,
        outputs_root=outputs_root,
        stochastic_samples=stochastic_samples,
        device=device,
    )
    output_dir = analysis_root if analysis_root is not None else data_dict["analysis_root"]
    generated = generate_all_figures(
        data_dict=data_dict,
        output_root=output_dir,
        save_formats=save_formats,
        figure_numbers=figure_numbers,
    )
    report_path = render_markdown_report(
        data_dict=data_dict,
        generated_figures=generated,
        output_dir=output_dir,
    )
    generated["analysis_report"] = str(report_path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the WRR paper figure set.")
    parser.add_argument(
        "--config",
        default="project/parameterize/conf/config_param_paper.yaml",
        help="Base paper-stack config used to infer output roots and attributes.",
    )
    parser.add_argument(
        "--outputs-root",
        default=None,
        help="Optional override for the run output root. Defaults to config-derived ./outputs.",
    )
    parser.add_argument(
        "--analysis-root",
        default=None,
        help="Optional override for the analysis figure output directory.",
    )
    parser.add_argument(
        "--figures",
        default=None,
        help="Comma-separated figure numbers to generate. Defaults to all 14.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated save formats. Defaults to png,pdf.",
    )
    parser.add_argument(
        "--stochastic-samples",
        type=int,
        default=100,
        help="Samples used for MC-dropout/distributional parameter extraction.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="Device used when reading checkpoints to reconstruct parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_numbers = (
        [int(item.strip()) for item in args.figures.split(",") if item.strip()]
        if args.figures
        else None
    )
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    build_data_and_generate(
        config_path=args.config,
        outputs_root=args.outputs_root,
        analysis_root=args.analysis_root,
        stochastic_samples=args.stochastic_samples,
        device=args.device,
        save_formats=formats,
        figure_numbers=figure_numbers,
    )


if __name__ == "__main__":
    main()
