# S3 Parameter Estimation: source of truth

## Executive summary

The active foundation-531 data contract is verified: 531 basins, 35 static attributes, explicit 1980-10-01 to 2010-09-30 dates, KGE(Q), and 365-day warm-up windows. The active IC design is EvoTorch XNES in normalized coordinates with clipping, deterministic LHS centers, three starts, 400 generations, population 48 for the XAJ screening configuration, and fixed-budget termination. A complete formal 531 IC-XNES result inventory was not found. Therefore restart dispersion, budget saturation, all-basin convergence, and the compensation-upper-bound claim are unresolved.

The active dPL code is a static attribute-to-parameter MLP with 35 inputs, three 256-unit SiLU/LayerNorm blocks, dropout 0.05 between hidden blocks, a linear output head, sigmoid/clamp outputs, AdamW, learning rate 1e-3, cosine annealing to 1e-4, batch size 128, 100 epochs, and gradient clipping 1.0. The active launcher protocol specifies seeds 42, 123, and 2026. Local 531 outputs are smoke results, not the full production run.

## S3.1 Independent route: optimizer and settings

The active XNES adapter calls local `evotorch.algorithms.XNES` through `ablation/optimizers/xnes.py`; candidate coordinates are normalized and clipped to [0,1] before `normalized_to_physical`. The stage-1 XAJ configuration records population 48, width 0.25, 400 generations, three starts, seed 0, and no test metric. It does not expose a user learning rate; XNES uses its implementation defaults. Initial centers are deterministic LHS points per basin/start, not normalized 0.5. See `s3_ic_optimizer_settings.csv` for rows whose model-specific production evidence is absent.

## S3.2 Hyperparameter calibration

`large_scale_screening/v1` contains real population/model scans, but its manifest uses the 1988 warm-up and 1989-1998 training protocol with 366 warm-up days. It is therefore a screening/calibration asset, not active foundation-531 calibration. No active 531 calibration set covering all hosts/structures was located.

## S3.3 Estimation adequacy

No formal 531 restart result or intermediate generation manifest was located. Screening traces are retained as legacy evidence only. It is not supported to state that independent estimation reached a global optimum, recovered the true parameter set, or established a compensation upper bound.

## S3.4 Constrained route: from attributes to parameters

The dPL runner consumes 35 attributes in the order defined by `ablation/ic_core/data_adapter.py`, robust median/IQR-normalizes them, imputes nonfinite attributes by column median, and clips normalized attributes to ±5. Each model has one parameter head with output dimension equal to its parameter specification. Physical mapping is linear for ordinary parameters and log-linear for `tgd_tau`; the head bias is initialized from physical defaults.

## S3.5 Constrained route: training protocol

The active 531 config specifies AdamW, 1e-3 initial learning rate, weight decay 1e-4, cosine annealing to 1e-4, batch 128, 100 epochs, validation every 10 epochs, and gradient clipping 1.0. The `balanced_valid_kge_windows` sampler samples basins uniformly and filters within-basin windows by at least 30 valid observations and observed standard deviation 0.05. The active launcher protocol specifies seeds 42, 123, and 2026, but no local three-seed production result inventory was found.

## S3.6 Shared objective, warm-up and evaluation path

Both routes target KGE(Q) and use the same broad validity rule, but the IC objective adapter and dPL differentiable KGE are separate implementations. They differ in epsilon stabilization and aggregation context. Code supports the same foundation dates and model classes, but a full parameter/state/forcing equivalence test was not found. Use a qualified statement, not “exactly the same objective/evaluation path.”

## S3.7 Computational environment and reproducibility

The screening freeze manifest records EvoTorch 0.6.1, PyTorch 2.9.1+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 3080 Ti Laptop GPU. A formal foundation-531 production freeze manifest, hardware record, and complete runtime log were not found. The local audit machine lacked importable NumPy/PyTorch in its selected environments; this is not evidence about the production machine.

## Conflicts and unresolved items

See `s3_conflicts.csv` and `s3_unresolved_items.csv`. In particular, 559 assets under `results/ic_xnes_full` and `outputs/dpl_unified_365d_v1` are legacy and must not be cited as foundation-531 settings.

## Evidence index

| Fact | Evidence |
|---|---|
| 531 basin list, 35 attributes, periods, KGE(Q), clipping | `ablation/configs/ic_foundation_531_v1.json`; `outputs/ic_ablation/foundation_v1/ic_531_dataset_manifest_resolved.json` |
| IC XNES adapter and maximize/ranking path | `ablation/optimizers/xnes.py:55-90`; `ablation/ic_core/runtime.py:79-117` |
| normalized-to-physical mapping and log `tgd_tau` | `ablation/ic_core/parameter_adapter.py:56-80` |
| XNES population/width/generations/starts/seed | `ablation/configs/ic_xnes_stage1_screening_v1.json`; `outputs/ic_ablation/stage1_screening/v1/xnes/dry_run_plan.json` |
| optimizer learning-rate/covariance defaults | `evotorch/algorithms/distributed/gaussian.py:1369-1405` |
| screening calibration traces | `outputs/ic_ablation/large_scale_screening/v1/**/result.json`; `outputs/ic_ablation/large_scale_screening/v1/**/trace.json` |
| dPL network construction | `training/dpl/run_dpl_model.py:125-168` |
| dPL attribute preprocessing and windows | `training/dpl/run_dpl_model.py:298-477` |
| dPL optimizer, checkpoint and training loop | `training/dpl/run_dpl_model.py:685-885`; `training/dpl/base_config_camels_531.json` |
| dPL three-seed launcher protocol | `training/dpl/run_camels_531_multiseed_autodl.sh:25-26,104-117` |
| IC objective | `ablation/ic_core/objective_adapter.py:11-40`; `experiments/ic_xnes/gpu_kge.py:58-98` |
| dPL objective | `training/dpl/run_dpl_model.py:480-520` |
| environment and screening runtime | `outputs/ic_ablation/stage1_screening/v1/xnes/environment.json`; `input_freeze_manifest.json` |

## Safe manuscript claims

Claim the verified code/configuration protocol, the XNES implementation and normalized mapping, the dPL architecture/training configuration, and the existence of screening evidence with its different date protocol. State explicitly that formal 531 IC adequacy, three-seed result equivalence, and complete shared-path equivalence remain to be documented.

## Prohibited claims

Do not claim global optimality, recovery of true parameters, a proven compensation upper bound, all-basin convergence, exact objective/evaluation equivalence, equivalent three-seed dPL results, or fully reproducible production runtime.
