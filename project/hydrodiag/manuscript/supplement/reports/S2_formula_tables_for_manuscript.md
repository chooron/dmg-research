# S2 Formula Tables for Manuscript

## State variables

See `results/s2_state_inventory.csv` and `results/s2_initialization_inventory.csv`. They list every host, CN and TGD state, units, defaults and source lines.

## Parameters and bounds

See `results/s2_parameter_inventory.csv`. It contains code name, symbol, description, lower/upper bound, default, unit, scope, transform and source line for every Base/TGD/CN/HBV parameter. `tgd_tau` is log-interpolated; other parameters use linear physical-bound interpolation after the dPL sigmoid output.

## Structural comparison

| Structure | Preprocessing state | Parameters | Temperature use | PET | Explicit snow |
|---|---|---:|---|---|---|
| Base | none | host only | validated; unused by XAJ/GR4J/SIMHYD host kernels | unchanged | no |
| TGD | `S` | host + 3 | standardized, clipped, tanh signal | unchanged | no |
| CN | `G,eTG` | host + 2 | solid fraction and thermal state | unchanged | yes |

## Host coupling

See `results/s2_coupling_inventory.csv`: every row follows raw `P,T,PET`, preprocessing or bypass, effective precipitation, host step, host routing and final discharge.

## Routing

See `results/s2_routing_inventory.csv`. XAJ and SIMHYD use finite gamma UH routing; GR4J uses finite differentiable UH1/UH2 plus a routing store; HBV has no convolutional UH.

## Implementation-specific operations

See `results/s2_piecewise_operations.csv`. It records all locally identified `where`, `min`, `max`, `clamp`, `tanh`, `expm1`, fractional-power floors, epsilon denominators, kernel normalization and finite-window tail handling with file:line evidence.
