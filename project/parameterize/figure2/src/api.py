"""Public API for the figure2 generation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from project.parameterize.figure2.src import builders
from project.parameterize.figure2.src.data_registry import FigureDataRegistry, MissingFigureDataError
from project.parameterize.figure2.src.figure_utils import QcCollector, set_hess_style, write_manifest


def build_registry(config_path: str, analysis_root: str | None = None) -> FigureDataRegistry:
    registry = FigureDataRegistry.from_paths(config_path=config_path, analysis_root=analysis_root)
    font_info = set_hess_style(registry.style, registry.palette)
    registry.style["font_info"] = font_info
    return registry


def _default_figure_dir(registry: FigureDataRegistry) -> Path:
    return registry.config_path.parents[1] / "figure2" / "figures" / "main_revised"


def _default_table_dir(registry: FigureDataRegistry) -> Path:
    return registry.config_path.parents[1] / "figure2" / "tables" / "main_revised"


def _default_report_path(registry: FigureDataRegistry) -> Path:
    return registry.config_path.parents[1] / "figure2" / "reports" / "qc_report_revised.md"


def generate_figure(
    figure_number: int,
    config_path: str,
    analysis_root: str | None = None,
    output_dir: str | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> dict[str, Any]:
    registry = build_registry(config_path=config_path, analysis_root=analysis_root)
    resolved_output_dir = Path(output_dir).resolve() if output_dir else _default_figure_dir(registry)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = resolved_output_dir / "_tables_single"
    report_path = resolved_output_dir / "qc_report_revised.md"
    qc = QcCollector(style=registry.style, palette=registry.palette, font_info=registry.style.get("font_info", {}))
    figure_id = f"{figure_number:02d}"
    spec = builders.FIGURE_SPECS[figure_id]
    manifest: dict[str, Any] = {"generated": {}, "missing_inputs": registry.missing_inputs, "tables": {}}
    try:
        outputs = spec["builder"](registry, resolved_output_dir, qc, formats)
        manifest["generated"][figure_id] = {suffix: str(path) for suffix, path in outputs.items()}
    except MissingFigureDataError as exc:
        qc.add_todo(figure_id, str(exc))
        manifest["generated"][figure_id] = {"skipped": True, "reason": str(exc)}
    manifest["tables"] = {
        name: {suffix: str(path) for suffix, path in paths.items()}
        for name, paths in builders.generate_table_outputs(registry, table_dir).items()
    }
    qc_path = qc.write_report(report_path)
    manifest["qc_report"] = str(qc_path)
    manifest_path = resolved_output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def generate_all_figures(
    config_path: str,
    analysis_root: str | None = None,
    output_dir: str | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
    figure_numbers: list[int] | None = None,
) -> dict[str, Any]:
    registry = build_registry(config_path=config_path, analysis_root=analysis_root)
    resolved_output_dir = Path(output_dir).resolve() if output_dir else _default_figure_dir(registry)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir:
        table_dir = resolved_output_dir.parent / "tables_main_revised"
        report_path = resolved_output_dir.parent / "qc_report_revised.md"
    else:
        table_dir = _default_table_dir(registry)
        report_path = _default_report_path(registry)
    table_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    qc = QcCollector(style=registry.style, palette=registry.palette, font_info=registry.style.get("font_info", {}))
    manifest: dict[str, Any] = {"generated": {}, "missing_inputs": registry.missing_inputs, "tables": {}}
    selected = {f"{number:02d}" for number in (figure_numbers or [int(key) for key in builders.FIGURE_SPECS])}
    manifest["tables"] = {
        name: {suffix: str(path) for suffix, path in paths.items()}
        for name, paths in builders.generate_table_outputs(registry, table_dir).items()
    }
    for figure_id, spec in builders.FIGURE_SPECS.items():
        if figure_id not in selected:
            continue
        try:
            outputs = spec["builder"](registry, resolved_output_dir, qc, formats)
            manifest["generated"][figure_id] = {
                "title": spec["title"],
                "outputs": {suffix: str(path) for suffix, path in outputs.items()},
            }
        except MissingFigureDataError as exc:
            qc.add_todo(figure_id, str(exc))
            manifest["generated"][figure_id] = {"skipped": True, "reason": str(exc)}
    qc_path = qc.write_report(report_path)
    manifest["qc_report"] = str(qc_path)
    manifest_path = resolved_output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the figure2 manuscript figure suite.")
    parser.add_argument(
        "--config",
        default="project/parameterize/conf/config_param_paper.yaml",
        help="Paper config used to resolve project-relative paths.",
    )
    parser.add_argument(
        "--analysis-root",
        default=None,
        help="Override analysis artifact root. Defaults to project/parameterize/outputs/analysis/stability_stats.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override figure output directory. Defaults to project/parameterize/figure2/figures/main_revised.",
    )
    parser.add_argument(
        "--figures",
        default=None,
        help="Comma-separated figure numbers to generate. Defaults to the full 11-figure suite.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated save formats. Defaults to png,pdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    figure_numbers = (
        [int(item.strip()) for item in args.figures.split(",") if item.strip()]
        if args.figures
        else None
    )
    generate_all_figures(
        config_path=args.config,
        analysis_root=args.analysis_root,
        output_dir=args.output_dir,
        formats=formats,
        figure_numbers=figure_numbers,
    )
