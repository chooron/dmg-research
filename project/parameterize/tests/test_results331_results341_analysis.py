from __future__ import annotations

import unittest

import pandas as pd

from project.parameterize.analysis.results331_results341_analysis import (
    attribute_type,
    build_results331_outputs,
    evidence_label,
)


class TestResults331Results341Analysis(unittest.TestCase):
    def test_attribute_type_groups_expected_domains(self) -> None:
        self.assertEqual(attribute_type("aridity"), "climate")
        self.assertEqual(attribute_type("frac_snow"), "snow")
        self.assertEqual(attribute_type("slope_mean"), "topography")
        self.assertEqual(attribute_type("lai_diff"), "vegetation")
        self.assertEqual(attribute_type("soil_conductivity"), "soil")
        self.assertEqual(attribute_type("geol_permeability"), "geology")

    def test_evidence_label_uses_focused_override(self) -> None:
        row = pd.Series({"parameter": "parUZL", "attribute": "soil_conductivity", "relationship_class": "robust"})
        self.assertEqual(evidence_label(row, {("parUZL", "soil_conductivity"): "supportive but not decisive"}), "supportive")
        self.assertEqual(evidence_label(row, {}), "robust")

    def test_build_results331_outputs_classifies_shared_and_sensitive_controls(self) -> None:
        relationship_classes = pd.DataFrame(
            {
                "model": ["deterministic", "mc_dropout", "distributional"] * 2,
                "parameter": ["parFC"] * 3 + ["parK2"] * 3,
                "attribute": ["pet_mean", "pet_mean", "pet_mean", "gvf_diff", "high_prec_dur", "high_prec_dur"],
                "mean_corr": [0.4, 0.5, 0.45, 0.3, -0.2, -0.25],
                "core_rank": [1] * 6,
                "relationship_class": ["robust", "robust", "robust", "model-sensitive", "model-sensitive", "model-sensitive"],
            }
        )
        parameter_level_consistency = pd.DataFrame(
            {
                "parameter": ["parFC", "parK2"],
                "deterministic_attribute": ["pet_mean", "gvf_diff"],
                "mc_dropout_attribute": ["pet_mean", "high_prec_dur"],
                "distributional_attribute": ["pet_mean", "high_prec_dur"],
                "deterministic_sign": ["positive", "positive"],
                "mc_dropout_sign": ["positive", "negative"],
                "distributional_sign": ["positive", "negative"],
                "dominant_attribute_consistency_across_models": ["all_same", "two_of_three"],
                "direction_consistency_across_models": ["all_same", "sign_flip_present"],
                "comments": ["shared", "sensitive"],
            }
        )
        focused_pair_classes = pd.DataFrame(columns=["parameter", "attribute", "evidence_class"])

        outputs = build_results331_outputs(
            relationship_classes=relationship_classes,
            parameter_level_consistency=parameter_level_consistency,
            focused_pair_classes=focused_pair_classes,
        )
        classes = dict(zip(outputs.relationship_classes["parameter"], outputs.relationship_classes["relationship_class"]))
        self.assertEqual(classes["parFC"], "shared dominant controls")
        self.assertEqual(classes["parK2"], "model-sensitive controls")


if __name__ == "__main__":
    unittest.main()
