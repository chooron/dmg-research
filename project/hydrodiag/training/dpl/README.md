# Active unified dPL training module

This run fixes the HBV ablation winner for the other four models:

`35 → 256 → 256 → 256 → n_parameters`, SiLU, dropout `0.05`, AdamW,
learning rate `1e-3`, weight decay `1e-4`, batch size `128`, 100 epochs,
cosine annealing to `1e-4`, median/IQR attribute normalization clipped to ±5,
seed `42`, and sigmoid-normalized parameter outputs. Ordinary parameters use
linear physical inverse mapping; the positive TGD2 residence times retain the
Lite-v2 inverse-log residence-time mapping.

The active runner supports HBV, GR4J, XAJ, SIMHYD, each CemaNeige composition,
and each two-parameter temperature-agnostic precipitation-delay control
(`GR4J_PD`, `XAJ_PD`, and `SIMHYD_PD`).
The active XAJ-only TGD2 composition is registered as `XAJ_TGD2`.
The hydrological model and its parameter specification are the only
model-specific elements. Existing complete outputs are kept as baselines; new
SIMHYD outputs are written only when explicitly launched.
All jobs use 365-day warmup + 365-day prediction windows with a 365-day stride.

### 531 loss-stability sampling policy

The CAMELS-531 dPL configuration uses `balanced_valid_kge_windows`. Each batch
first samples basin IDs uniformly, so every basin retains equal sampling
probability. It then samples a time window uniformly from that basin's
calibration-window catalogue. A catalogue window must contain at least 30
finite, non-negative observations and have observed streamflow standard
deviation at least 0.05 mm/day. This removes zero/near-zero-variance windows
that make KGE's `alpha` and `beta` terms arbitrarily large after the required
epsilon floor, without deleting basins or changing the KGE function. A basin
with no eligible window is retained through a documented highest-variance
fallback window; the runner records the catalogue counts and fallback count in
`config.json` and the startup log.

## CAMELS-531 / flexmopex protocol

Use [`base_config_camels_531.json`](base_config_camels_531.json) locally or
[`base_config_camels_531_autodl.json`](base_config_camels_531_autodl.json) on
AutoDL. These configurations read the full flexmopex tensor
`camels_dataset` directly, use `gage_id.npy` for the source basin axis,
`camels_dates.npy` for the daily time axis, select `/data/531sub_id.txt` by
gauge ID before time slicing, and use the flexmopex periods. The forcing order
is fixed in code as `P,T,PET`:

- warmup: 1980-10-01 to 1981-09-30 (365 days);
- training loss: 1981-10-01 to 1995-09-30;
- test: 1995-10-01 to 2010-09-30, with its preceding 365 days as warmup.

For a non-performance smoke run that preserves the same architecture and date
protocol, use the smoke config plus two basins, one training window, and one
epoch:

```bash
python training/dpl/run_dpl_model.py \
  --config training/dpl/smoke_config_camels_531_autodl.json \
  --model GR4J_PD --max-basins 2 --max-windows 1 --epochs 1
```

Run one model directly:

```bash
python project/hydrodiag/training/dpl/run_dpl_model.py \
  --config project/hydrodiag/training/dpl/base_config.json \
  --model GR4J
```

For the streamflow-only deployment path, pass `--lite`; the checkpoint records
the selected model class and refuses a resume when the model/Lite mode or
parameter specification does not match.  The multi-model launcher enables the
same path with `LITE_MODELS=1`.

Run the four models concurrently:

```bash
python project/hydrodiag/training/dpl/launch_models.py \
  --jobs 4 --gpus 0
```

New launchers must write model outputs under
`project/hydrodiag/results/<run_id>/`. The historical
`outputs/dpl_unified_365d_v1/` path is a compatibility link into
`results/archive/legacy_outputs_20260730/`. Former configuration-ablation
launchers and generated files are preserved under `archive/`.
