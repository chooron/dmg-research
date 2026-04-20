from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from project.parameterize.analyze_seed_stability import (
    basin_variance_table,
    canonical_parameter_name,
    infer_parameter_names,
    key_parameter_correlation_summary,
    method_consistency_overall,
    method_consistency_summary,
    mean_pairwise_topk_overlap,
    seedwise_spearman_table,
    sign_consistency_rate,
    summarize_pairwise_correlation_stability,
    validate_mc_dropout_sample_counts,
)


class TestSeedStabilityAnalysis(unittest.TestCase):
    def test_canonical_parameter_name_maps_smax_alias(self) -> None:
        available = ["parFC", "parBETA"]
        self.assertEqual(canonical_parameter_name("smax", available), "parFC")
        self.assertEqual(canonical_parameter_name("parBETA", available), "parBETA")

    def test_basin_variance_table_averages_same_basin_cross_seed_variance(self) -> None:
        frame = pd.DataFrame(
            {
                "basin_id": [1, 1, 1, 2, 2, 2],
                "variant": ["deterministic"] * 6,
                "seed": [111, 222, 333, 111, 222, 333],
                "sample_count": [1] * 6,
                "parFC_mean": [1.0, 3.0, 5.0, 2.0, 2.0, 4.0],
                "parFC_std": [0.0] * 6,
            }
        )

        basin_table, parameter_summary, variant_summary = basin_variance_table(frame, ["parFC"])

        self.assertEqual(len(basin_table), 2)
        basin_variances = basin_table["basin_variance"].to_numpy()
        np.testing.assert_allclose(np.sort(basin_variances), np.array([8.0 / 9.0, 8.0 / 3.0]))
        self.assertAlmostEqual(
            parameter_summary.loc[0, "mean_basin_variance"],
            ((8.0 / 3.0) + (8.0 / 9.0)) / 2.0,
        )
        self.assertAlmostEqual(
            variant_summary.loc[0, "mean_of_mean_basin_variance"],
            ((8.0 / 3.0) + (8.0 / 9.0)) / 2.0,
        )

    def test_seedwise_spearman_table_tracks_cross_seed_r2_variance(self) -> None:
        params = pd.DataFrame(
            {
                "basin_id": [1, 2, 3, 4] * 3,
                "variant": ["distributional"] * 12,
                "seed": [111] * 4 + [222] * 4 + [333] * 4,
                "sample_count": [100] * 12,
                "parFC_mean": [1, 2, 3, 4, 1, 3, 2, 4, 4, 1, 2, 3],
                "parFC_std": [0.1] * 12,
            }
        )
        attrs = pd.DataFrame({"basin_id": [1, 2, 3, 4], "attrA": [1, 2, 3, 4]})

        seed_table = seedwise_spearman_table(params, attrs, ["parFC"])
        pair_summary, variant_summary = summarize_pairwise_correlation_stability(seed_table)

        self.assertEqual(len(pair_summary), 1)
        self.assertAlmostEqual(pair_summary.loc[0, "mean_spearman_r2"], (1.0 + 0.64 + 0.04) / 3.0)
        self.assertAlmostEqual(pair_summary.loc[0, "variance_spearman_r2"], 0.1568, places=4)
        self.assertAlmostEqual(
            variant_summary.loc[0, "mean_variance_spearman_r2"],
            0.1568,
            places=4,
        )

    def test_infer_parameter_names_ignores_metadata(self) -> None:
        frame = pd.DataFrame(
            {
                "basin_id": [1],
                "variant": ["deterministic"],
                "seed": [111],
                "sample_count": [1],
                "parFC_mean": [10.0],
                "parFC_std": [0.0],
                "route_b_mean": [1.0],
            }
        )
        self.assertEqual(infer_parameter_names(frame), ["parFC", "route_b"])

    def test_sign_consistency_rate_reports_majority_sign_fraction(self) -> None:
        self.assertAlmostEqual(sign_consistency_rate(pd.Series([0.2, 0.1, 0.5])), 1.0)
        self.assertAlmostEqual(sign_consistency_rate(pd.Series([0.2, -0.1, 0.5])), 2.0 / 3.0)
        self.assertAlmostEqual(sign_consistency_rate(pd.Series([0.0, 0.0, 0.0])), 1.0)

    def test_mean_pairwise_topk_overlap_averages_pairwise_seed_overlap(self) -> None:
        rankings = {
            111: ["a", "b", "c"],
            222: ["a", "c", "d"],
            333: ["a", "b", "d"],
        }
        self.assertAlmostEqual(mean_pairwise_topk_overlap(rankings, top_k=2), 2.0 / 3.0)

    def test_key_parameter_summary_includes_sign_consistency_and_topk_overlap(self) -> None:
        seed_table = pd.DataFrame(
            {
                "variant": ["deterministic"] * 6,
                "seed": [111, 111, 222, 222, 333, 333],
                "attribute": ["attr1", "attr2"] * 3,
                "parameter": ["parFC"] * 6,
                "spearman_rho": [0.9, 0.1, 0.8, -0.2, 0.7, 0.3],
                "spearman_r2": [0.81, 0.01, 0.64, 0.04, 0.49, 0.09],
                "abs_spearman_rho": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            }
        )
        pair_summary, _ = summarize_pairwise_correlation_stability(seed_table)
        key_summary = key_parameter_correlation_summary(
            seed_table=seed_table,
            pair_table=pair_summary,
            key_parameters=["parFC"],
            top_k=1,
        )

        self.assertEqual(len(key_summary), 1)
        self.assertAlmostEqual(key_summary.loc[0, "mean_sign_consistency"], (1.0 + (2.0 / 3.0)) / 2.0)
        self.assertAlmostEqual(key_summary.loc[0, "topk_overlap_rate"], 1.0)

    def test_method_consistency_summary_compares_methods_on_attribute_rho_vectors(self) -> None:
        pair_table = pd.DataFrame(
            {
                "variant": [
                    "deterministic",
                    "deterministic",
                    "distributional",
                    "distributional",
                    "mc_dropout",
                    "mc_dropout",
                ],
                "attribute": ["attr1", "attr2"] * 3,
                "parameter": ["parFC"] * 6,
                "mean_spearman_rho": [0.8, -0.4, 0.7, -0.3, 0.2, 0.1],
                "mean_abs_spearman_rho": [0.8, 0.4, 0.7, 0.3, 0.2, 0.1],
            }
        )

        summary = method_consistency_summary(pair_table, key_parameters=["parFC"], top_k=1)
        self.assertEqual(len(summary), 3)

        det_dist = summary.loc[
            (summary["variant_a"] == "deterministic")
            & (summary["variant_b"] == "distributional")
        ].iloc[0]
        self.assertAlmostEqual(det_dist["sign_agreement_rate"], 1.0)
        self.assertAlmostEqual(det_dist["topk_overlap_rate"], 1.0)

        overall = method_consistency_overall(summary)
        self.assertEqual(len(overall), 3)
        self.assertIn("mean_spearman_corr_of_mean_rho", overall.columns)

    def test_validate_mc_dropout_sample_counts_enforces_minimum(self) -> None:
        frame = pd.DataFrame(
            {
                "basin_id": [1, 1, 1, 1],
                "variant": ["deterministic", "mc_dropout", "mc_dropout", "distributional"],
                "seed": [111, 111, 222, 111],
                "sample_count": [1, 100, 120, 100],
                "parFC_mean": [1.0, 2.0, 2.0, 3.0],
            }
        )

        summary = validate_mc_dropout_sample_counts(frame, min_samples=100)
        self.assertEqual(summary["mc_dropout"], 100)

        bad_frame = frame.copy()
        bad_frame.loc[bad_frame["variant"] == "mc_dropout", "sample_count"] = 50
        with self.assertRaisesRegex(ValueError, "mc_dropout sample_count must be at least 100"):
            validate_mc_dropout_sample_counts(bad_frame, min_samples=100)


if __name__ == "__main__":
    unittest.main()
