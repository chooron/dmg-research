from __future__ import annotations

from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.unithydro_validation_utils import (
    OUTPUT_DIR,
    PLOTS_DIR,
    build_inspection_summary_markdown,
    build_report_markdown,
    failures_from_results,
    gather_all_results,
    write_failure_details,
    write_report_markdown,
    write_summary_csv,
)


def _write_failure_plots(failures: list[dict], output_dir: Path) -> None:
    if not failures:
        return

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, failure in enumerate(failures[:10]):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        metrics = [
            failure["max_abs_error"],
            failure["mean_abs_error"],
            failure["rmse"],
            failure["relative_l2_error"],
        ]
        labels = ["max_abs", "mean_abs", "rmse", "rel_l2"]
        ax.bar(labels, metrics, color="#3b6fb6")
        ax.set_title(
            f"{failure['dmot_function']} | {failure['test_case']} | "
            f"{failure['parameter_case']} | {failure['dtype']} | {failure['device']}"
        )
        ax.set_ylabel("Error")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"failure_{index:02d}.png", dpi=150)
        plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    include_cuda = torch.cuda.is_available()

    inspection_path = OUTPUT_DIR / "unithydro_inspection_summary.md"
    summary_csv_path = OUTPUT_DIR / "unithydro_consistency_summary.csv"
    report_path = OUTPUT_DIR / "unithydro_consistency_report.md"
    failure_details_path = OUTPUT_DIR / "failed_case_details.json"

    inspection_path.write_text(build_inspection_summary_markdown(), encoding="utf-8")

    results = gather_all_results(include_cuda=include_cuda)
    failures = failures_from_results(results)

    write_summary_csv(results, summary_csv_path)
    write_failure_details(failures, failure_details_path)
    write_report_markdown(build_report_markdown(results, include_cuda), report_path)
    _write_failure_plots(failures, PLOTS_DIR)

    print(f"Wrote inspection summary to {inspection_path}")
    print(f"Wrote CSV summary to {summary_csv_path}")
    print(f"Wrote markdown report to {report_path}")
    if failures:
        print(f"Wrote failure details to {failure_details_path}")
        print(f"Wrote diagnostic plots to {PLOTS_DIR}")
        return 1

    if failure_details_path.exists():
        failure_details_path.unlink()
    if PLOTS_DIR.exists():
        for plot_path in PLOTS_DIR.glob("*.png"):
            plot_path.unlink()

    print(f"Validated {len(results)} cases with no failures.")
    if include_cuda:
        print("CUDA checks were included.")
    else:
        print("CUDA checks were skipped because CUDA is unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
