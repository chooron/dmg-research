# S2 Formula Source-of-Truth Audit Report

## Executive summary

The active 531 foundation configuration selects the model keys recorded in `results/s2_active_call_graph.json`. IC-XNES uses `ablation/ic_core/model_adapter.py` and defaults to full classes; dPL uses the same `models` classes through `training/dpl/run_dpl_model.py` unless `--lite` is explicitly supplied. The active structural controls are Base, basic two-parameter CemaNeige (CN), precipitation delay (PD), and three-parameter temperature-conditioned generic delay (TGD).

The implementation facts below are code facts, not textbook substitutions.

## Active implementation map

| Route | Evidence | Fact |
|---|---|---|
| IC model selection | `ablation/ic_core/model_adapter.py:14-28` | `MODEL_CLASSES` maps all active keys to the classes in `models`. |
| IC execution | `ablation/ic_core/runtime.py:104-110` | forcing and physical parameters enter the shared model adapter. |
| dPL execution | `training/dpl/run_dpl_model.py:84-114,627-650` | dPL registry uses the same full/lite model classes; full is default. |
| 531 design | `ablation/configs/ic_foundation_531_v1.json:2-12` | current foundation configuration, not legacy 559. |

## Model formulas

The machine-readable equation inventory is `results/s2_equation_inventory.csv`. XAJ has three-layer evaporation, tension-water capacity runoff, free-water separation, linear interflow/groundwater reservoirs and a finite 15-ordinate gamma UH. GR4J has the production store, 0.9/0.1 split, differentiable S-curve UH1/UH2, exchange term and routing store. SIMHYD has interception, exponential infiltration, interflow, recharge, soil overflow transfer, groundwater recession and finite gamma UH. HBV is a standalone five-state explicit snow/soil/response model.

## CN and TGD

CN calls `_cemaneige_step` (`models/composed.py:46-53`) before the host step. Its basic implementation uses a 0/3 degC piecewise solid fraction, G and eTG states, an instantaneous SCA ratio, and storage-limited melt. TGD calls `_temperature_conditioned_delay_step` (`models/temperature_delay.py:26-51`) before the host step. It has a single delay storage, frozen training temperature mean/standard deviation, bounded smooth temperature signal, dynamic tau, and conservative release. TGD has no rain/snow partition, SWE state, or melt equation. Neither module changes PET.

## Coupling and routing

The coupling matrix is `results/s2_module_coupling_matrix.csv`. In every wrapper the order is raw `P,T,PET`, preprocessing, same-day `effective_precip`, host runoff, and host routing. Base bypasses preprocessing; PD is a separate temperature-agnostic control and must not be described as TGD. The initialization and routing matrix is `results/s2_initialization_and_routing.csv`; finite UH buffers are carried for continuation, while a finite output window does not include its future tail.

## Parameters and bounds

`results/s2_parameter_manifest.csv` contains every active key's code name, symbol, bounds, unit, scope and mapping. The physical parameter adapter is `ablation/ic_core/parameter_adapter.py:56-67`: all parameters use linear bounds except `tgd_tau`, which uses log interpolation. dPL outputs a sigmoid normalized value (`training/dpl/run_dpl_model.py:166-168`) and maps it to physical bounds. Parameters are basin-specific at model execution; network weights are shared but are not hydrological parameters.

## Smoothing and numerical details

The threshold inventory is `results/s2_threshold_and_smoothing_inventory.csv`. Important implementation-specific details are epsilon denominators, fractional-power base floors, store clamps, `torch.where` branches, TGD `tanh` temperature clipping, `expm1` release calculation, and finite UH normalization. These alter derivatives and sometimes the exact discrete map; they must be reported as implementation details.

## Runtime verification

`results/s2_one_step_results.csv` records deterministic CPU forward probes and `results/s2_mass_balance_results.csv` records short-sequence diagnostics. `results/s2_gradient_check_results.csv` compares autograd with central finite differences for representative interior parameters. A complete whole-system mass balance remains unresolved for XAJ, GR4J and HBV because current active auxiliaries do not expose all daily storage and ET terms and UH tails; this is reported rather than silently treated as a pass.

## Literature comparison

`results/s2_reference_comparison.csv` separates canonical references from implemented equations. The manuscript must lead with implemented formulas. The current CemaNeige wrapper is the basic variant, not the hysteresis class, and the SIMHYD name alone does not establish a unique canonical variant.

## Prohibited or unresolved manuscript claims

Do not state that TGD is an explicit snow model, that external SWE is truth, that CN and TGD have identical state dimension, that all 531 design cells have completed results, or that the current implementation is exactly canonical XAJ/GR4J/SIMHYD/HBV without qualification.

## Generated artifacts

Scripts are in `manuscript/supplement/scripts/`; results are in `manuscript/supplement/results/`; candidate equations and readiness are in this report directory.
