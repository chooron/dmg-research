# 531 IC Foundation Protocol

This protocol is independent of the legacy 559 IC runner. It is the shared
data, model, objective, configuration, and result contract for later IC
optimizer screening and a final 531-basin calibration.

## Data and periods

- Dataset: `/home/jingxin/code/dmg-research/data/camels_dataset`.
- Basin IDs: `/home/jingxin/code/dmg-research/data/531sub_id.txt`, exactly 531
  unique IDs in file order.
- Source basin IDs are read from `gage_id.npy`; the 531 list is resolved against
  that array and the resulting indices select the tuple rows.
- Dates are read from the standalone `camels_dates.npy` axis.
- Forcing names/order: `P`, `T`, `PET`.
- Raw target: `ft3/s`; IC target: `mm/day`.
- Area: `area_gages2`, attribute column 11, physical `km2`, as documented by
  the existing dPL CAMELS loader.

The dPL CAMELS-531 daily protocol is used exactly:

| split | dates | days |
|---|---|---:|
| warmup | 1980-10-01 through 1981-09-30 | 365 |
| train target | 1981-10-01 through 1995-09-30 | 5113 |
| test target | 1995-10-01 through 2010-09-30 | 5479 |

The inspected dPL CAMELS-531 configs do not use monthly streamflow windows:
their `window` contract is daily `warmup_days=365`, `prediction_days=365`,
`stride_days=365`. Monthly ERA5 files elsewhere in the repository are not part
of this IC forcing/target protocol.

Training model input contains warmup + train target. Test model input contains
the 365 days immediately before test plus test target. Temperature mean/std are
computed from train target only and reused for test.

## Unit conversion

Valid discharge is converted with:

`Q_mm_day = Q_ft3_s * 0.028316846592 * 86400 * 1000 / (area_km2 * 1e6)`.

Nonfinite or negative raw discharge remains missing (`NaN`); valid zero remains
zero. No blanket clipping is performed.

## Parameter and model protocol

`models/parameter_specs.py` remains the only parameter-boundary source. The
adapter preserves parameter order and uses linear mapping for the active
`temperature_conditioned_input_routing_a_ks_v1` TGD parameters. Models
are imported from `models/` and never copied into `ablation/`.

For low-resource startup validation, `ic_core.model_adapter` also exposes an
explicit `model_variant="lite"` mapping for all 13 current IC model keys.
These are the repository's native Lite classes (for example `XAJLite`,
`GR4JWithCemaNeigeLite`, and `HBVLite`), preserving the same equations while
using the compact streamflow-only path when diagnostics are not requested.
The validation runner uses one basin and a small candidate batch; it is not an
optimizer benchmark.

## Objective and runtime

Forward uses FP32. KGE(Q) reductions use FP64 and maximize direction. A runtime
call accepts normalized candidates with shape `[D]`, `[P,D]`, or `[B,P,D]` and
returns `[B,P]` fitness, validity, valid counts, and evaluation diagnostics.
The runtime owns no optimizer state and has no XNES-specific assumptions.

## Result and checkpoint contract

Every candidate evaluation is one basin/parameter vector completing a full
train-period forward and producing one fitness. Generation-level records use
`candidate_evaluations_generation` and cumulative counts separately. The
minimum checkpoint protocol atomically records resolved-config identity,
basin/model/start/seed, completed evaluation count, and a completed or failed
marker. Optimizer state is optional and reserved for optimizer adapters.

Future optimizers implement `ic_core.optimizer_protocol.OptimizerAdapter` with
`initialize`, `ask`, `tell`, `state_dict`, `load_state_dict`, and `reset`.
The foundation runtime remains the sole owner of data and fitness evaluation.

The GPU startup check is run with:

```bash
python -m ablation.runners.run_lite_gpu_validation --population 2
```

It uses the train period only, sets Torch CPU threads to one, transfers only
the selected basin/candidates, and writes its records to
`outputs/ic_ablation/foundation_v1/lite_gpu_validation/`.

All foundation and future ablation outputs belong under
`outputs/ic_ablation/`; the legacy `results/ic_xnes_full/` root is forbidden.
