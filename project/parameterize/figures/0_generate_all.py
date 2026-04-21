"""CLI wrapper for generating the full WRR figure suite."""

from project.parameterize.figures.api import build_data_and_generate, generate_all_figures, main

__all__ = ["build_data_and_generate", "generate_all_figures", "main"]


if __name__ == "__main__":
    main()

