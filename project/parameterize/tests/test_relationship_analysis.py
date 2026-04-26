from __future__ import annotations

import unittest

import pandas as pd

from project.parameterize.analysis.relationship_analysis import (
    build_parameter_level_consistency,
    classify_relationships,
    compute_core_relationships,
    summarize_stability_significance,
    top_feature_overlap_table,
)


class TestRelationshipAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        corr_map = {
            ("distributional", "L1", 111, "p1", "attr_a"): 0.82,
            ("distributional", "L1", 222, "p1", "attr_a"): 0.80,
            ("distributional", "L2", 111, "p1", "attr_a"): 0.79,
            ("distributional", "L2", 222, "p1", "attr_a"): 0.78,
            ("deterministic", "L1", 111, "p1", "attr_a"): 0.75,
            ("deterministic", "L1", 222, "p1", "attr_a"): 0.74,
            ("deterministic", "L2", 111, "p1", "attr_a"): 0.72,
            ("deterministic", "L2", 222, "p1", "attr_a"): 0.71,
            ("mc_dropout", "L1", 111, "p1", "attr_a"): 0.70,
            ("mc_dropout", "L1", 222, "p1", "attr_a"): 0.69,
            ("mc_dropout", "L2", 111, "p1", "attr_a"): 0.68,
            ("mc_dropout", "L2", 222, "p1", "attr_a"): 0.67,
            ("distributional", "L1", 111, "p2", "attr_a"): 0.84,
            ("distributional", "L1", 222, "p2", "attr_a"): 0.83,
            ("distributional", "L2", 111, "p2", "attr_a"): 0.82,
            ("distributional", "L2", 222, "p2", "attr_a"): 0.81,
            ("deterministic", "L1", 111, "p2", "attr_b"): -0.80,
            ("deterministic", "L1", 222, "p2", "attr_b"): -0.79,
            ("deterministic", "L2", 111, "p2", "attr_b"): -0.78,
            ("deterministic", "L2", 222, "p2", "attr_b"): -0.77,
            ("mc_dropout", "L1", 111, "p2", "attr_b"): -0.76,
            ("mc_dropout", "L1", 222, "p2", "attr_b"): -0.75,
            ("mc_dropout", "L2", 111, "p2", "attr_b"): -0.74,
            ("mc_dropout", "L2", 222, "p2", "attr_b"): -0.73,
            ("distributional", "L1", 111, "p3", "attr_a"): 0.83,
            ("distributional", "L1", 222, "p3", "attr_a"): 0.81,
            ("distributional", "L2", 111, "p3", "attr_a"): 0.28,
            ("distributional", "L2", 222, "p3", "attr_a"): 0.26,
            ("deterministic", "L1", 111, "p3", "attr_a"): 0.77,
            ("deterministic", "L1", 222, "p3", "attr_a"): 0.76,
            ("deterministic", "L2", 111, "p3", "attr_a"): 0.74,
            ("deterministic", "L2", 222, "p3", "attr_a"): 0.73,
            ("mc_dropout", "L1", 111, "p3", "attr_a"): 0.75,
            ("mc_dropout", "L1", 222, "p3", "attr_a"): 0.74,
            ("mc_dropout", "L2", 111, "p3", "attr_a"): 0.72,
            ("mc_dropout", "L2", 222, "p3", "attr_a"): 0.71,
        }
        for model in ("distributional", "deterministic", "mc_dropout"):
            for loss in ("L1", "L2"):
                for seed in (111, 222):
                    for parameter in ("p1", "p2", "p3"):
                        for attribute in ("attr_a", "attr_b"):
                            corr = corr_map.get((model, loss, seed, parameter, attribute), 0.10)
                            rows.append(
                                {
                                    "method": "spearman",
                                    "model": model,
                                    "loss": loss,
                                    "seed": seed,
                                    "parameter": parameter,
                                    "attribute": attribute,
                                    "corr": corr,
                                    "abs_corr": abs(corr),
                                }
                            )
        self.corr_long = pd.DataFrame(rows)

    def test_compute_core_relationships_returns_top_pairs_and_stability_tables(self) -> None:
        core, pair_seed, pair_loss = compute_core_relationships(
            self.corr_long,
            top_k=1,
            pairs_per_parameter=1,
        )

        self.assertEqual(len(core), 9)
        self.assertEqual(
            core.loc[(core["model"] == "distributional") & (core["parameter"] == "p1"), "attribute"].iloc[0],
            "attr_a",
        )
        self.assertGreater(len(pair_seed), 0)
        self.assertGreater(len(pair_loss), 0)

    def test_classification_covers_robust_model_sensitive_and_loss_sensitive(self) -> None:
        core, _, _ = compute_core_relationships(self.corr_long, top_k=1, pairs_per_parameter=1)
        consistency = build_parameter_level_consistency(core)
        classified = classify_relationships(core, consistency)

        p1 = consistency.loc[consistency["parameter"] == "p1"].iloc[0]
        self.assertEqual(p1["dominant_attribute_consistency_across_models"], "all_same")

        p2 = consistency.loc[consistency["parameter"] == "p2"].iloc[0]
        self.assertEqual(p2["dominant_attribute_consistency_across_models"], "two_of_three")

        robust_row = classified.loc[
            (classified["model"] == "distributional") & (classified["parameter"] == "p1")
        ].iloc[0]
        self.assertEqual(robust_row["relationship_class"], "robust")

        loss_sensitive_row = classified.loc[
            (classified["model"] == "distributional") & (classified["parameter"] == "p3")
        ].iloc[0]
        self.assertEqual(loss_sensitive_row["relationship_class"], "loss-sensitive")

        model_sensitive_row = classified.loc[
            (classified["model"] == "distributional") & (classified["parameter"] == "p2")
        ].iloc[0]
        self.assertEqual(model_sensitive_row["relationship_class"], "model-sensitive")

    def test_stability_significance_summary_contains_distributional_comparisons(self) -> None:
        core, _, _ = compute_core_relationships(self.corr_long, top_k=1, pairs_per_parameter=1)
        consistency = build_parameter_level_consistency(core)
        classified = classify_relationships(core, consistency)
        summary = summarize_stability_significance(classified)

        self.assertEqual(
            set(summary["comparison"]),
            {"distributional_vs_deterministic", "distributional_vs_mc_dropout"},
        )
        self.assertIn("dominant_attribute_consistency", set(summary["metric"]))

    def test_top_feature_overlap_table_reports_shared_top_features(self) -> None:
        importance = pd.DataFrame(
            {
                "parameter": ["p1"] * 6,
                "model": ["deterministic"] * 2 + ["distributional"] * 2 + ["mc_dropout"] * 2,
                "attribute": ["attr_a", "attr_b", "attr_a", "attr_c", "attr_a", "attr_b"],
                "mean_rank": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            }
        )
        overlap = top_feature_overlap_table(importance, top_k=1)
        det_dist = overlap.loc[
            (overlap["model_a"] == "deterministic") & (overlap["model_b"] == "distributional")
        ].iloc[0]
        self.assertEqual(det_dist["shared_feature_count"], 1)
        self.assertEqual(det_dist["shared_features"], "attr_a")


if __name__ == "__main__":
    unittest.main()
