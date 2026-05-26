from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[4]))

from project.parameterize.manuscript.analysis_pipeline import (  # noqa: E402
    data_inventory,
    distributional_spatial,
    environmental_gradients,
    integrated_summary,
    mean_attribute_relationships,
    model_consistency,
    representative_basin_groups,
    seed_loss_sensitivity,
    uncertainty_attribute_relationships,
    uncertainty_spatial,
)
from project.parameterize.manuscript.analysis_pipeline.common import ANALYSIS_ROOT, PipelineLog, discover_runs, ensure_block_dirs, save_csv  # noqa: E402


def main() -> None:
    dirs = ensure_block_dirs()
    log = PipelineLog(ANALYSIS_ROOT / "logs" / "full_analysis_pipeline_log.txt")
    log.add("Pipeline started.")
    context: dict[str, object] = {}

    run_inventory = discover_runs()
    context["run_inventory"] = run_inventory
    save_csv(run_inventory, ANALYSIS_ROOT / "00_data_inventory" / "data" / "run_inventory_raw_discovered.csv", log)
    log.add(f"Discovered {len(run_inventory)} 531-run directories.")

    modules = [
        data_inventory,
        model_consistency,
        seed_loss_sensitivity,
        distributional_spatial,
        mean_attribute_relationships,
        environmental_gradients,
        uncertainty_spatial,
        uncertainty_attribute_relationships,
        representative_basin_groups,
        integrated_summary,
    ]
    for module in modules:
        log.add(f"Running {module.__name__.split('.')[-1]}.")
        module.run(dirs, context, log)
        log.add(f"Finished {module.__name__.split('.')[-1]}.")

    log.add("Pipeline completed.")
    log.write()


if __name__ == "__main__":
    main()

