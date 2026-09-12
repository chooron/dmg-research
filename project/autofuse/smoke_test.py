"""Small local smoke test; never starts full SCE or dPL training."""

from __future__ import annotations

import json
import time

import torch

from dfuse import simulate

from .evaluator import UnifiedEvaluator
from .protocol import ExperimentProtocol
from .run_record import run_record
from .reference import reference_status


def run_smoke() -> dict[str, object]:
    started = time.perf_counter()
    protocol = ExperimentProtocol()
    forcing = torch.tensor(
        [[5.0, 2.0, 10.0], [0.0, 2.0, -2.0], [4.0, 2.0, 10.0], [3.0, 1.5, 8.0]],
        dtype=torch.float64,
    )
    forward = simulate(84, forcing, implicit_iterations=8)
    observed = forward.q.detach() * 0.95 + 0.1
    scored = UnifiedEvaluator().score(84, forcing, observed, implicit_iterations=8)
    record = run_record(protocol, seed=protocol.default_seed)
    record.update(
        {
            "model_id": 84,
            "n_timesteps": int(forcing.shape[0]),
            "max_abs_water_balance_error": float(forward.max_abs_water_balance_error.detach()),
            "max_abs_snow_balance_error": float(forward.snow_balance_residual.abs().amax().detach()),
            "kgecomp": float(scored.kge_comp.detach()),
            "elapsed_seconds": time.perf_counter() - started,
            "full_scale_started": False,
            "reference": reference_status(),
        }
    )
    return record


def main() -> None:
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
