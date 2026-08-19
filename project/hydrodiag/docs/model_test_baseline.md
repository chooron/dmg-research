# Hydrological model/test baseline

Baseline ID: `model-test-baseline-2026-07-20`

This is the frozen quality gate for the models in `models/`.  The registry is
in [`tests/model_registry.py`](../tests/model_registry.py); the runner is
[`manuscript/scripts/shared/run_model_test_suite.py`](../manuscript/scripts/shared/run_model_test_suite.py).

## Registered models

| Model | Parameters | Snow | Routing | Required checks |
|---|---:|---|---|---|
| HBV | 12 | built-in | no | forward, gradient, step compile |
| GR4J | 4 | no | GR4J UH | forward, gradient, x4/UH, boundaries, state carry, compile |
| XAJ | 15 | no | gamma | forward, gradient, state carry, boundaries, compile |
| SIMHYD | 10 | no | gamma | forward, gradient, water balance, boundaries, state carry, compile |
| CemaNeige | 2 | standalone | no | forward, gradient, compile |
| CemaNeigeHyst | 4 | standalone | no | forward, gradient, compile |
| PrecipitationDelay | 2 | temperature-agnostic control | no | forward, gradient, conservation, boundaries, compile |
| GR4J+CemaNeige | 6 | preprocessing | GR4J UH | forward, gradient, boundaries, compile |
| XAJ+CemaNeige | 17 | preprocessing | gamma | forward, gradient, boundaries, state carry, compile |
| SIMHYD+CemaNeige | 12 | preprocessing | gamma | forward, gradient, water balance, boundaries, state carry, compile |
| GR4J+PrecipitationDelay | 6 | preprocessing control | GR4J UH | forward, gradient, conservation, boundaries, compile |
| XAJ+PrecipitationDelay | 17 | preprocessing control | gamma | forward, gradient, boundaries, compile |
| SIMHYD+PrecipitationDelay | 12 | preprocessing control | gamma | forward, gradient, water balance, boundaries, state carry, compile |

## Commands

From `project/hydrodiag`:

```bash
python manuscript/scripts/shared/run_model_test_suite.py
python manuscript/scripts/shared/run_model_test_suite.py --full
```

The runner automatically includes CUDA cases when CUDA is available.  The
canonical model gate includes boundary values, finite outputs and states,
finite parameter gradients, fullgraph step compilation, unit-hydrograph
continuation, and water-balance checks.  The `--full` mode additionally runs
the independent paper-style CMA-ES tests.

## Recorded result

After adding the temperature-agnostic precipitation-delay control, the
canonical model-only runner completed with **98 passed**.  This includes the
three new composed models and their gradient, boundary, compilation, and
water-balance checks.  The separate `--full` mode remains an optimizer-level
diagnostic and was not used for this model-structure gate.

## Script review

`scripts/test_water_balance.py`, `scripts/test_gradient_stability.py`, and
`scripts/test_euler_stability.py` are legacy report-generating diagnostics.
They remain available for exploratory audits, but are not the release gate:
the first two predate SIMHYD/CemaNeige composition and omit some routing/snow
stores, while the Euler script still targets an older XAJ lag-state interface.
The canonical runner deliberately excludes all three and uses only `tests/`.
Use the canonical runner above for release decisions.
