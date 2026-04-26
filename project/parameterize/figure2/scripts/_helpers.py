"""Small CLI helpers for figure2 entrypoints."""

from __future__ import annotations

from project.parameterize.figure2.src.api import parse_args, generate_all_figures, generate_figure


def run_single(figure_number: int) -> None:
    args = parse_args()
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    generate_figure(
        figure_number=figure_number,
        config_path=args.config,
        analysis_root=args.analysis_root,
        output_dir=args.output_dir,
        formats=formats,
    )


def run_all() -> None:
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
