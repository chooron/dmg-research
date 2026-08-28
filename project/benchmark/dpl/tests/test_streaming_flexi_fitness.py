from __future__ import annotations

import torch

from project.benchmark.src.model_registry import build_model
from project.benchmark.src.objective import streaming_kge
from project.benchmark.src.streaming_evaluator import compute_flexi_streaming_fitness


def test_streaming_kge_matches_full_prediction_reference() -> None:
    generator = torch.Generator().manual_seed(17)
    prediction = torch.randn(13, 2, 2, 3, generator=generator)
    observation = torch.randn(13, 2, generator=generator)
    prediction[2, 1, 0, 1] = float("nan")
    prediction[8, 0, 1, 2] = float("inf")
    observation[5, 1] = float("nan")

    full_score, full_invalid = streaming_kge(prediction, observation)

    from project.benchmark.src.objective import (
        finalize_streaming_kge_tensors,
        initialize_streaming_kge,
        update_streaming_kge_tensors,
    )

    state = initialize_streaming_kge((2, 6), prediction.device)
    for timestep in range(prediction.shape[0]):
        values = update_streaming_kge_tensors(
            state.count,
            state.sum_pred,
            state.sum_obs,
            state.sum_pred2,
            state.sum_obs2,
            state.sum_cross,
            state.invalid_prediction,
            prediction[timestep].reshape(2, 6),
            observation[timestep],
        )
        state = state.__class__(*values)

    score, invalid = finalize_streaming_kge_tensors(
        state.count,
        state.sum_pred,
        state.sum_obs,
        state.sum_pred2,
        state.sum_obs2,
        state.sum_cross,
        state.invalid_prediction,
    )
    torch.testing.assert_close(score, full_score.reshape(2, 6), rtol=1e-8, atol=1e-8)
    assert torch.equal(invalid, full_invalid.reshape(2, 6))


def test_flexi_streaming_fitness_matches_compiled_forward() -> None:
    torch.set_num_threads(1)
    basin_count, starts, population, dimension = 2, 1, 2, 10
    warmup_days, total_days = 3, 12
    generator = torch.Generator().manual_seed(23)
    forcing = torch.rand(total_days, basin_count, 3, generator=generator, dtype=torch.float64)
    observation = torch.rand(
        total_days - warmup_days, basin_count, generator=generator, dtype=torch.float64
    )
    latent = torch.randn(
        basin_count,
        starts,
        population,
        dimension,
        generator=generator,
        dtype=torch.float64,
    )
    model = build_model(
        "flexi", "cpu", warm_up=warmup_days, backend="compile", dtype=torch.float64
    )
    raw = torch.sigmoid(latent).permute(0, 3, 1, 2).reshape(
        basin_count, dimension, starts * population
    ).to(torch.float64)

    with torch.inference_mode():
        prediction = model({"x_phy": forcing}, (None, raw))["streamflow"]
        prediction = prediction.reshape(total_days - warmup_days, basin_count, starts, population)
    expected, expected_invalid = streaming_kge(prediction, observation)

    actual, actual_invalid = compute_flexi_streaming_fitness(
        model, forcing, observation, latent, warmup_days=warmup_days
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(actual_invalid, expected_invalid)


def test_streaming_kge_edge_cases_match_full_reference() -> None:
    from project.benchmark.src.objective import (
        finalize_streaming_kge_tensors,
        initialize_streaming_kge,
        update_streaming_kge_tensors,
    )

    tiny_variation = torch.tensor([1.0, 1.0 + 1e-12, 1.0 - 1e-12, 1.0], dtype=torch.float64)
    cases = {
        "all_observations_invalid": (
            torch.ones(4, 1, 1, 1),
            torch.full((4, 1), float("nan")),
        ),
        "one_valid_timestep": (
            torch.ones(4, 1, 1, 1),
            torch.tensor([[float("nan")], [2.0], [float("nan")], [float("nan")]]),
        ),
        "constant_observations": (
            torch.tensor([1.0, 2.0, 3.0, 4.0]).view(4, 1, 1, 1),
            torch.ones(4, 1),
        ),
        "constant_predictions": (
            torch.ones(4, 1, 1, 1),
            torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        ),
        "mixed_valid_invalid": (
            torch.tensor([1.0, float("nan"), float("inf"), 4.0]).view(4, 1, 1, 1),
            torch.tensor([[1.0], [2.0], [float("nan")], [4.0]]),
        ),
        "observation_mean_near_zero": (
            torch.tensor([0.02, -0.02, 0.03, -0.03]).view(4, 1, 1, 1),
            torch.tensor([[0.05], [-0.05], [0.05], [-0.05]]),
        ),
        "variance_near_floor": (
            tiny_variation.view(4, 1, 1, 1),
            tiny_variation.view(4, 1),
        ),
    }

    for name, (prediction, observation) in cases.items():
        expected, expected_invalid = streaming_kge(prediction, observation)
        state = initialize_streaming_kge((1, 1), prediction.device)
        for timestep in range(prediction.shape[0]):
            state = state.__class__(*update_streaming_kge_tensors(
                state.count,
                state.sum_pred,
                state.sum_obs,
                state.sum_pred2,
                state.sum_obs2,
                state.sum_cross,
                state.invalid_prediction,
                prediction[timestep].reshape(1, 1),
                observation[timestep],
            ))
        actual, actual_invalid = finalize_streaming_kge_tensors(
            state.count,
            state.sum_pred,
            state.sum_obs,
            state.sum_pred2,
            state.sum_obs2,
            state.sum_cross,
            state.invalid_prediction,
        )
        torch.testing.assert_close(actual, expected.reshape(1, 1), rtol=1e-12, atol=1e-12, msg=name)
        assert torch.equal(actual_invalid, expected_invalid.reshape(1, 1)), name
