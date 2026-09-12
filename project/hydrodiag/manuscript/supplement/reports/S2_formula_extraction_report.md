# S2 Formula Extraction Report

## Active full-model path

The active 531 foundation manifest is `ablation/configs/ic_foundation_531_v1.json:2-12`. IC uses `MODEL_CLASSES` in `ablation/ic_core/model_adapter.py:14-28`; `ModelAdapter` selects full classes unless `variant="lite"` is explicitly passed (`ablation/ic_core/model_adapter.py:31-45`). IC evaluation calls this adapter at `ablation/ic_core/runtime.py:104-110`. dPL registers the same full classes at `training/dpl/run_dpl_model.py:84-98` and selects the lite registry only under the explicit `--lite` option at `training/dpl/run_dpl_model.py:613-629`. Thus the default IC and dPL full forward path is shared. HBV is registered as a standalone class and is not a Base/TGD/CN wrapper.

The full active classes are `XAJ`, `GR4J`, `SIMHYD`, `HBV`; `XAJWithCemaNeige`, `GR4JWithCemaNeige`, `SIMHYDWithCemaNeige`; and `XAJWithTemperatureConditionedDelay`, `GR4JWithTemperatureConditionedDelay`, `SIMHYDWithTemperatureConditionedDelay`. The corresponding wrappers call the fused step in the order module first, host second: `models/composed.py:28-93` and `models/composed_temperature_delay.py:32-110`.

## Formula package

The complete ordered formulas are in `results/s2_formula_inventory.csv` and the writing-ready LaTeX version is `S2_full_equations_for_writing.md`. The state, flux, parameter, initialization, routing, coupling and piecewise-operation tables are the companion machine-readable files in `results/`.

## Scope decisions

This extraction contains Base, TGD, CN and the HBV snow-process reference only. PD and GD are excluded by scope. No mass-balance, gradient, training or performance validation was run.

## Implementation distinctions

CN is the active basic two-parameter CemaNeige path calling `_cemaneige_step`, not `_cemaneige_hyst_step`. It uses a 0--3 degC piecewise solid fraction, G/eTG states, a fixed `0.9 * estimated annual solid precipitation` threshold and storage-limited melt. TGD is a three-parameter generic delay with one storage state, frozen training-period temperature statistics and a smooth bounded temperature signal. It has no rain/snow partition, SWE state or melt equation. Both leave PET unchanged.

## Unresolved formula boundaries

The local repository does not uniquely identify a canonical SIMHYD literature variant, and the detailed discretization inside the imported `hydrodl2` UH helper is outside the local source tree. These are explicitly listed in `results/s2_formula_unresolved.csv`; they are not filled with textbook assumptions.
