from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project.parameterize.figures import generate_all_figures
from project.parameterize.figures.common import resolve_analysis_output_root
from project.parameterize.figures.data_loading import RunSpec, discover_runs


def _synthetic_data_dict(tmp_dir: str) -> dict:
    basin_ids = np.arange(1001, 1013)
    models = ["deterministic", "mc_dropout", "distributional"]
    losses = ["NseBatchLoss", "LogNseBatchLoss", "HybridNseBatchLoss"]
    seeds = [111, 222]
    parameters = ["parFC", "parBETA", "parK1", "parTT"]
    attributes = pd.DataFrame(
        {
            "basin_id": basin_ids,
            "aridity": np.linspace(0.2, 2.5, len(basin_ids)),
            "frac_snow": np.linspace(0.1, 0.8, len(basin_ids)),
            "slope_mean": np.linspace(0.5, 4.0, len(basin_ids)),
            "frac_forest": np.linspace(0.2, 0.9, len(basin_ids)),
            "clay_frac": np.linspace(0.1, 0.4, len(basin_ids)),
            "p_mean": np.linspace(400, 1600, len(basin_ids)),
            "pet_mean": np.linspace(300, 1100, len(basin_ids)),
            "area_gages2": np.linspace(50, 5000, len(basin_ids)),
        }
    )

    metrics_rows = []
    params_rows = []
    corr_rows = []
    model_shift = {"deterministic": 0.00, "mc_dropout": 0.03, "distributional": 0.06}
    loss_shift = {"NseBatchLoss": -0.01, "LogNseBatchLoss": 0.00, "HybridNseBatchLoss": 0.02}
    parameter_shift = {"parFC": 20.0, "parBETA": 0.3, "parK1": 0.02, "parTT": -0.1}

    for model in models:
        for loss in losses:
            for seed in seeds:
                seed_shift = 0.005 if seed == 222 else 0.0
                for basin_idx, basin_id in enumerate(basin_ids):
                    base_skill = 0.45 + model_shift[model] + loss_shift[loss] + seed_shift + basin_idx * 0.01
                    metrics_rows.append(
                        {
                            "basin_id": int(basin_id),
                            "model": model,
                            "loss": loss,
                            "seed": seed,
                            "nse": base_skill,
                            "kge": min(base_skill + 0.08, 0.99),
                            "bias": (-1) ** basin_idx * (0.03 + 0.004 * basin_idx),
                            "bias_abs": 0.03 + 0.004 * basin_idx,
                        }
                    )

                    aridity = attributes.loc[attributes["basin_id"] == basin_id, "aridity"].iloc[0]
                    frac_snow = attributes.loc[attributes["basin_id"] == basin_id, "frac_snow"].iloc[0]
                    slope = attributes.loc[attributes["basin_id"] == basin_id, "slope_mean"].iloc[0]
                    for parameter in parameters:
                        mean = parameter_shift[parameter] + model_shift[model] * 10.0 + seed_shift * 5.0
                        if parameter == "parFC":
                            mean += 40.0 * aridity
                        elif parameter == "parBETA":
                            mean += 1.5 * aridity - 0.2 * frac_snow
                        elif parameter == "parK1":
                            mean += 0.03 * slope - 0.01 * aridity
                        elif parameter == "parTT":
                            mean += -0.5 * frac_snow + 0.05 * aridity

                        std = 0.0 if model == "deterministic" else (0.05 if model == "mc_dropout" else 0.08) + 0.01 * aridity
                        params_rows.append(
                            {
                                "basin_id": int(basin_id),
                                "model": model,
                                "loss": loss,
                                "seed": seed,
                                "sample_count": 1 if model == "deterministic" else 100,
                                "parameter": parameter,
                                "parameter_label": parameter.replace("par", ""),
                                "mean": float(mean),
                                "std": float(std),
                            }
                        )

                for parameter in parameters:
                    for attribute in attributes.columns[1:]:
                        base_rho = {
                            "parFC": 0.35,
                            "parBETA": 0.28,
                            "parK1": -0.22,
                            "parTT": -0.18,
                        }[parameter]
                        attr_bonus = 0.03 if attribute in {"aridity", "frac_snow"} else -0.01
                        rho = base_rho + attr_bonus + model_shift[model] - loss_shift[loss]
                        corr_rows.append(
                            {
                                "model": model,
                                "loss": loss,
                                "seed": seed,
                                "parameter": parameter,
                                "parameter_label": parameter.replace("par", ""),
                                "attribute": attribute,
                                "spearman_rho": rho,
                                "spearman_p": 0.001,
                                "spearman_r2": rho**2,
                                "abs_rho": abs(rho),
                            }
                        )

    return {
        "analysis_root": Path(tmp_dir) / "analysis",
        "reference_loss": "HybridNseBatchLoss",
        "model_order": models,
        "loss_order": losses,
        "seed_order": seeds,
        "attributes": attributes,
        "attribute_names": list(attributes.columns[1:]),
        "key_attributes": list(attributes.columns[1:]),
        "focus_parameters": ["parFC", "parBETA", "parK1", "parTT"],
        "climate_parameters": ["parFC", "parBETA", "parK1"],
        "metrics_long": pd.DataFrame(metrics_rows),
        "params_long": pd.DataFrame(params_rows),
        "corr_long": pd.DataFrame(corr_rows),
        "param_names": parameters,
        "hbv_priors": {
            "FC": (50.0, 500.0),
            "BETA": (1.0, 6.0),
            "K1": (0.01, 0.3),
            "TT": (-2.0, 2.0),
        },
    }


class TestFiguresPipeline(unittest.TestCase):
    def test_resolve_analysis_output_root_follows_config_output_tree(self) -> None:
        root = resolve_analysis_output_root("/workspace/autoresearch/project/parameterize/conf/config_param_paper.yaml")
        self.assertEqual(root, Path("/workspace/autoresearch/project/parameterize/outputs/analysis/wrr_figures"))

    def test_discover_runs_accepts_new_variant_loss_seed_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_dir = Path(tmp_dir) / "distributional" / "HybridNseBatchLoss" / "seed_111"
            seed_dir.mkdir(parents=True)
            (seed_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "paper_variant": "distributional",
                        "loss_name": "HybridNseBatchLoss",
                        "seed": 111,
                    }
                ),
                encoding="utf-8",
            )
            runs = discover_runs(Path(tmp_dir))
            self.assertEqual(runs, [RunSpec(model="distributional", loss="HybridNseBatchLoss", seed=111, run_dir=seed_dir)])

    def test_discover_runs_accepts_variant_directory_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_dir = Path(tmp_dir) / "mc_dropout-531" / "HybridNseBatchLoss" / "seed_222"
            seed_dir.mkdir(parents=True)
            (seed_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "paper_variant": "mc_dropout",
                        "loss_name": "HybridNseBatchLoss",
                        "seed": 222,
                    }
                ),
                encoding="utf-8",
            )
            runs = discover_runs(Path(tmp_dir))
            self.assertEqual(runs, [RunSpec(model="mc_dropout", loss="HybridNseBatchLoss", seed=222, run_dir=seed_dir)])

    def test_generate_all_figures_writes_all_outputs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dict = _synthetic_data_dict(tmp_dir)
            generated = generate_all_figures(data_dict, output_root=data_dict["analysis_root"])
            analysis_root = Path(data_dict["analysis_root"])
            self.assertTrue((analysis_root / "manifest.json").exists())
            for idx in range(1, 15):
                stem = f"{idx:02d}"
                self.assertIn(stem, generated)
                outputs = generated[stem]["outputs"]
                self.assertTrue(Path(outputs["png"]).exists())
                self.assertTrue(Path(outputs["pdf"]).exists())


if __name__ == "__main__":
    unittest.main()
