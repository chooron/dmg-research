# Active IC/dPL training baseline

This document reflects the 2026-07-30 project cleanup. Historical source is
under [`../archive/`](../archive/README.md); every physical run product is
under [`../results/`](../results/README.md).

| Purpose | Active module | Configuration/output |
|---|---|---|
| IC foundation/adapters | [`ablation/ic_core/`](../ablation/ic_core/) and [`training/ic/`](../training/ic/) | [`ablation/configs/`](../ablation/configs/) |
| dPL model training | [`training/dpl/run_dpl_model.py`](../training/dpl/run_dpl_model.py) | [`training/dpl/base_config.json`](../training/dpl/base_config.json) |
| Active XAJ-TGD2 dPL | [`training/dpl/launch_xaj_tgd2_lite_v3.py`](../training/dpl/launch_xaj_tgd2_lite_v3.py) | `results/dpl_camels_531_lite_v3_tgd2_dpl_audited/` |
| dPL multi-model launcher | [`training/dpl/launch_models.py`](../training/dpl/launch_models.py) | New runs must use `results/<run_id>/` |
| HBV window selection | [`training/dpl/run_hbv_window_ablation.py`](../training/dpl/run_hbv_window_ablation.py) | [`configs/dpl_hbv_kgeq_365d_v1.json`](../configs/dpl_hbv_kgeq_365d_v1.json) |

The historical XNES production runner and its 559-basin configuration are
retained under `archive/project_cleanup_20260730/legacy_experiments/` and
`configs/ic_xnes_production_v1.json` respectively for reproducibility; they
are not the active 531-basin production entry point.

The dPL and IC registries include HBV, GR4J, XAJ, SIMHYD, the three
two-parameter CemaNeige compositions, and the three parameter-matched
temperature-agnostic precipitation-delay controls. The delay controls have the
same total parameter dimensionality as their corresponding CemaNeige models.
The active dPL registry additionally includes XAJ-TGD2. All model calls use
the current gamma-unit-hydrograph routing implementations.

Before a production run:

```bash
cd project/hydrodiag
python training/dpl/launch_models.py --dry-run
python scripts/run_model_test_suite.py
```

Production IC requires CUDA and writes resumable per-model checkpoints. Do not
write new runs into `archive/`, `outputs/`, or `experiment/results/`; use a
new directory under `results/`.

GPU smoke validation on 2026-07-20 completed one epoch for two basins for both
`SIMHYD` (validation median KGE 0.4589) and `SIMHYD_CN` (0.7299). Those
temporary outputs are physically archived under
`results/archive/legacy_archive_outputs_20260730/`.
