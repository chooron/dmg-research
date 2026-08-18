from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation.ic_core.config import environment_snapshot, load_resolved_config
from ablation.ic_core.data_adapter import (
    AREA_ATTRIBUTE_INDEX,
    ATTRIBUTE_NAMES,
    load_531_bundle,
    manifest_for_bundle,
    write_basin_index_csv,
    write_manifest,
)
from ablation.ic_core.result_io import atomic_write_json, atomic_write_text


def _write_structure(bundle, config, output_root: Path) -> None:
    raw_target = bundle.target_cfs
    structure = {
        "serialization": bundle.source_metadata["serialization"],
        "top_level_keys": None,
        "tuple_fields": ["forcing", "target", "attributes"],
        "basin_id_key": "gage_id.npy",
        "forcing_key": "tuple[0]",
        "target_key": "tuple[1]",
        "attributes_key": "tuple[2]",
        "date_time_key": "camels_dates.npy",
        "forcing_names": list(bundle.forcing_names),
        "target_raw_unit": "ft3/s",
        "target_unit_evidence": "training/dpl/run_dpl_model.py:118-124",
        "attributes_shape": list(bundle.raw_attributes.shape),
        "attribute_names": list(ATTRIBUTE_NAMES),
        "attribute_names_evidence": "benchmark/core/icb_config.py:COMMON_ATTRIBUTES and dPL CAMELS loader",
        "raw_area_field": "area_gages2",
        "raw_area_attribute_index": AREA_ATTRIBUTE_INDEX,
        "raw_area_unit": "km2",
        "raw_area_range_selected": [
            float(np.min(bundle.raw_area_km2)),
            float(np.max(bundle.raw_area_km2)),
        ],
        "missing_encoding": {
            "target_nan_count_selected": int(np.isnan(raw_target).sum()),
            "target_inf_count_selected": int(np.isinf(raw_target).sum()),
            "target_negative_count_selected": int((raw_target < 0).sum()),
            "forcing_nan_count_selected": int(np.isnan(bundle.forcing).sum()),
            "forcing_inf_count_selected": int(np.isinf(bundle.forcing).sum()),
            "forcing_negative_count_selected": int((bundle.forcing < 0).sum()),
        },
        "dtypes": {
            "dataset_source": bundle.source_metadata["dataset_dtypes"],
            "forcing_loaded_for_model": str(bundle.forcing.dtype),
            "target_cfs": str(bundle.target_cfs.dtype),
            "target_mm_day": str(bundle.target_mm_day.dtype),
            "attributes": str(bundle.raw_attributes.dtype),
            "dates": str(bundle.dates.dtype),
        },
        "n_source_basins": int(bundle.source_metadata["n_source_basins"]),
        "n_selected_basins": len(bundle.basin_ids),
        "n_timesteps": int(bundle.forcing.shape[1]),
        "date_start": str(bundle.dates[0]),
        "date_end": str(bundle.dates[-1]),
        "dates_daily_contiguous": bool(
            np.all(np.diff(bundle.dates.astype("datetime64[D]").astype("int64")) == 1)
        ),
        "leap_day_count": int(
            sum(str(value)[5:10] == "02-29" for value in bundle.dates)
        ),
        "selected_shapes": {
            "forcing": list(bundle.forcing.shape),
            "target_cfs": list(bundle.target_cfs.shape),
            "target_mm_day": list(bundle.target_mm_day.shape),
            "valid_target_mask": list(bundle.valid_target_mask.shape),
            "attributes": list(bundle.raw_attributes.shape),
        },
        "basin_ids_first": list(bundle.basin_ids[:5]),
        "basin_ids_last": list(bundle.basin_ids[-5:]),
        "source_indices_first": bundle.source_indices[:5].tolist(),
        "source_indices_last": bundle.source_indices[-5:].tolist(),
        "metadata_indices_first": bundle.metadata_indices[:5].tolist(),
        "metadata_indices_last": bundle.metadata_indices[-5:].tolist(),
        "basin_axis_source": bundle.source_metadata["basin_axis_source"],
        "date_axis_source": bundle.source_metadata["date_axis_source"],
    }
    atomic_write_json(output_root / "dataset_structure.json", structure)
    markdown = "\n".join(
        [
            "# CAMELS-531 dataset structure",
            "",
            "The source is a pickle tuple `(forcing, target, attributes)` plus",
            "`gage_id.npy`, `camels_dates.npy`, and the code-defined forcing order.",
            "No old 559 NPZ is",
            "loaded by this foundation adapter.",
            "",
            f"- source basins: {structure['n_source_basins']}",
            f"- selected basins: {structure['n_selected_basins']}",
            f"- timesteps: {structure['n_timesteps']}",
            f"- dates: {structure['date_start']} to {structure['date_end']}",
            f"- forcing: {structure['forcing_names']} with shape {structure['selected_shapes']['forcing']}",
            f"- target raw: ft3/s with shape {structure['selected_shapes']['target_cfs']}",
            f"- target IC: mm/day with shape {structure['selected_shapes']['target_mm_day']}",
            f"- attributes: shape {structure['selected_shapes']['attributes']}",
            f"- area: area_gages2, attribute index {AREA_ATTRIBUTE_INDEX}, km2",
            f"- selected target NaN count: {structure['missing_encoding']['target_nan_count_selected']}",
            f"- selected target negative count: {structure['missing_encoding']['target_negative_count_selected']}",
            f"- leap days: {structure['leap_day_count']}",
            "",
            "The tuple has no embedded field names; attribute names and the area",
            "column are taken from the existing project CAMELS contract and dPL",
            "loader evidence, recorded above rather than guessed from column values.",
            "",
        ]
    )
    atomic_write_text(output_root / "dataset_structure.md", markdown)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "ablation/configs/ic_foundation_531_v1.json",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_resolved_config(args.config, device_override=args.device)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "resolved_config.json", config)
    atomic_write_json(output_root / "environment.json", environment_snapshot(config))
    bundle = load_531_bundle(config)
    _write_structure(bundle, config, output_root)
    atomic_write_json(output_root / "period_resolution.json", bundle.periods.as_dict())
    memory_report = "\n".join(
        [
            "# Memory and device report",
            "",
            "- The source pickle is loaded once by the adapter.",
            f"- Selected CPU forcing bytes: {bundle.forcing.nbytes}",
            f"- Selected raw target bytes: {bundle.target_cfs.nbytes}",
            f"- Selected converted target bytes: {bundle.target_mm_day.nbytes}",
            f"- Selected attributes bytes: {bundle.raw_attributes.nbytes}",
            "- The complete selected bundle remains on CPU; runtime transfers only a",
            f"  basin batch of at most {config['batching']['basin_batch_size']} basins and its candidates.",
            f"- Configured device: {config['device']}",
            "- No permanent copied dataset or GPU-resident full dataset is created.",
            "",
        ]
    )
    atomic_write_text(output_root / "memory_and_device_report.md", memory_report)
    manifest = manifest_for_bundle(bundle, config)
    write_manifest(
        bundle,
        config,
        PROJECT_ROOT / "ablation/manifests/ic_531_dataset_manifest_v1.json",
    )
    atomic_write_json(output_root / "ic_531_dataset_manifest_resolved.json", manifest)
    write_basin_index_csv(
        bundle, PROJECT_ROOT / "ablation/manifests/ic_531_basin_index_v1.csv"
    )
    atomic_write_json(
        output_root / "smoke_data.json",
        {
            "status": "pass",
            "n_basins": len(bundle.basin_ids),
            "basin_order_hash": manifest["basin_order_hash"],
            "selected_shapes": manifest["selected_shapes"],
            "date_range": [manifest["date_start"], manifest["date_end"]],
            "no_559_fallback": True,
            "target_valid_count_train_min": int(
                bundle.valid_target_mask[
                    :,
                    bundle.periods.train.start_index : bundle.periods.train.end_index
                    + 1,
                ]
                .sum(axis=1)
                .min()
            ),
        },
    )


if __name__ == "__main__":
    main()
