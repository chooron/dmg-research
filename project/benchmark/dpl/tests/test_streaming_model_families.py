from __future__ import annotations

import torch

from project.benchmark.src.model_registry import NPARAM_INFO_36, build_model
from project.benchmark.src.objective import streaming_kge
from project.benchmark.src.streaming_evaluator import compute_streaming_fitness


def test_all_registered_models_streaming_matches_short_reference() -> None:
    """Exercise every registered execution family without a long training run."""
    torch.set_num_threads(1)
    for index, (model_name, dimension) in enumerate(NPARAM_INFO_36.items(), start=1):
        basin_count, starts, population = 1, 1, 1
        total_days, warmup_days = 8, 3
        generator = torch.Generator().manual_seed(10_000 + index)
        channels = 4 if model_name in {"mopex4", "mopex5"} else 3
        forcing = torch.rand(
            total_days, basin_count, channels, generator=generator, dtype=torch.float64
        )
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
            model_name, "cpu", warm_up=warmup_days, backend="compile", dtype=torch.float64
        )
        assert all(state.dtype == torch.float64 for state in model._init_states(1, 1))
        raw = torch.sigmoid(latent).permute(0, 3, 1, 2).reshape(
            basin_count, dimension, starts * population
        ).to(torch.float64)

        with torch.inference_mode():
            prediction = model({"x_phy": forcing}, (None, raw))["streamflow"]
            prediction = prediction.reshape(
                total_days - warmup_days, basin_count, starts, population
            )
        assert prediction.dtype == torch.float64
        expected, expected_invalid = streaming_kge(prediction, observation)
        actual, actual_invalid = compute_streaming_fitness(
            model, forcing, observation, latent, warmup_days=warmup_days
        )
        assert actual.dtype == torch.float64

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=model_name)
        assert torch.equal(actual_invalid, expected_invalid), model_name
