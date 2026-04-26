from __future__ import annotations

import unittest

import pandas as pd

from project.parameterize.analysis.focused_cross_loss_analysis import (
    build_focused_pair_model_comparison,
    build_focused_pair_significance,
    classify_focused_pairs,
    select_focused_pairs,
)


class TestFocusedCrossLossAnalysis(unittest.TestCase):
    def test_select_focused_pairs_keeps_mandatory_core_and_top_extensions(self) -> None:
        frame = pd.DataFrame(
            {
                "model": ["distributional"] * 6,
                "parameter": ["parUZL", "parBETA", "parFC", "parK1", "parCFR", "parTT"],
                "attribute": [
                    "soil_conductivity",
                    "slope_mean",
                    "pet_mean",
                    "lai_diff",
                    "elev_mean",
                    "high_prec_dur",
                ],
                "relationship_class": ["robust"] * 6,
                "seed_stable": [True] * 6,
                "loss_stable": [True] * 6,
                "mean_abs_corr": [0.53, 0.52, 0.48, 0.46, 0.41, 0.39],
                "core_rank": [1, 1, 1, 1, 2, 2],
            }
        )
        selected = select_focused_pairs(frame)
        self.assertEqual(
            set(selected.loc[selected["focus_group"] == "core_candidate", "parameter"]),
            {"parUZL", "parBETA", "parFC", "parK1", "parCFR", "parTT"},
        )
        self.assertEqual(len(selected.loc[selected["focus_group"] == "extended_robust"]), 0)

    def test_model_comparison_and_significance_use_core_pairs(self) -> None:
        summary = pd.DataFrame(
            {
                "focus_group": ["core_candidate"] * 8,
                "pair_label": ["a", "a", "b", "b", "c", "c", "d", "d"],
                "parameter": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
                "attribute": ["x", "x", "y", "y", "z", "z", "w", "w"],
                "model": ["distributional", "deterministic"] * 4,
                "method": ["spearman"] * 8,
                "mean_rho": [0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2],
                "abs_mean_rho": [0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2],
                "cross_loss_std": [0.04, 0.08, 0.03, 0.09, 0.05, 0.07, 0.02, 0.03],
                "cross_loss_range": [0.10, 0.18, 0.08, 0.19, 0.09, 0.13, 0.04, 0.06],
                "cross_loss_max_abs_dev": [0.06, 0.10, 0.05, 0.11, 0.05, 0.07, 0.02, 0.03],
                "sign_consistency_across_losses": [1.0] * 8,
                "topk_consistency": [1.0, 0.8, 1.0, 0.6, 0.9, 0.8, 0.7, 0.7],
                "dominant_consistency": [1.0, 0.8, 1.0, 0.6, 0.7, 0.7, 0.5, 0.5],
                "all_losses_majority_topk": [True] * 8,
                "all_losses_majority_dominant": [True, True, True, True, True, True, True, True],
            }
        )
        comparison = build_focused_pair_model_comparison(summary)
        self.assertEqual(len(comparison), 4)
        self.assertTrue((comparison["advantage_loss_std_det_minus_dist"] > 0).all())

        significance = build_focused_pair_significance(comparison)
        self.assertIn("loss_std", set(significance["metric"]))
        self.assertEqual(int(significance.loc[significance["metric"] == "loss_std", "pair_count"].iloc[0]), 4)

    def test_classify_focused_pairs_distinguishes_headline_and_supportive(self) -> None:
        comparison = pd.DataFrame(
            {
                "focus_group": ["core_candidate", "core_candidate", "extended_robust"],
                "pair_label": ["a", "b", "c"],
                "parameter": ["p1", "p2", "p3"],
                "attribute": ["x", "y", "z"],
                "advantage_loss_std_det_minus_dist": [0.03, 0.005, -0.002],
                "advantage_loss_range_det_minus_dist": [0.08, 0.02, -0.01],
                "advantage_topk_consistency_dist_minus_det": [0.2, 0.0, -0.1],
                "advantage_dominant_consistency_dist_minus_det": [0.2, 0.0, -0.1],
                "dist_dominant_consistency": [1.0, 0.7, 0.3],
                "det_dominant_consistency": [0.8, 0.7, 0.4],
                "dist_all_losses_majority_topk": [True, True, False],
                "det_all_losses_majority_topk": [True, True, False],
                "dist_all_losses_majority_dominant": [True, True, False],
                "det_all_losses_majority_dominant": [True, True, False],
            }
        )
        classes = classify_focused_pairs(comparison)
        labels = dict(zip(classes["pair_label"], classes["evidence_class"]))
        self.assertEqual(labels["a"], "headline evidence")
        self.assertEqual(labels["b"], "supportive but not decisive")
        self.assertEqual(labels["c"], "not supportive")


if __name__ == "__main__":
    unittest.main()
