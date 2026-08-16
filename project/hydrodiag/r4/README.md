# R4 — real-basin snow-state consistency pipeline

R4 compares the internal snow states of the R1/R2 observation-trained
Base / CN / TGD2 models against external CAMELS-US Snow-17/SAC-SMA SWE on
real basins.  This package implements the R4 pipeline *mechanics*; it never
re-trains and never treats R3 synthetic-q* checkpoints as scientific results.

## Stages / tags

- `DEV_ONLY` + `SYNTHETIC_TRAINED` — outputs produced from R3 runs
  (`r3_gate_*`, `r3_misspec_*`); pipeline smoke tests only.
- `OFFICIAL_OBSERVATION_TRAINED` — the only tag allowed for formal R4 outputs;
  requires provenance-verified R1/R2 observation-trained parameters
  (see `sync_manifest.json`).

## Module map

| Module | Purpose |
|---|---|
| `common.py` | canonical paths, R1/R2 periods, model keys, run identities |
| `state_export.py` | continuous full-axis recorded forward (port of R3 `recorded_forward`), per-day CN states, production-forward identity validation, psol/g_thresh recording |
| `input_adapters.py` | IC (canonical batched JSON + fused per_start.csv) and dPL (checkpoint + normalization + specs) parameter readers; R1 best-restart rule; fail-loud |
| `forward_export.py` | parameters -> continuous forward -> per-period daily CSV + manifest |
| `smoke_test.py` | `DEV_ONLY` pipeline smoke test (runs S1–S6 checks) |
| `snow_reference.py` | CAMELS-US Snow-17/SAC-SMA SWE reader (target basins), annual burden metrics, G-vs-SWE consistency stats |
| `sync_manifest.json/.md` | minimal remote sync list for observation-trained R1/R2 artifacts |

## Semantics pinned by R4

- Continuous forward over the full 12418-day axis from zero initial states
  (matches R3 posthoc convention and the R4 requirement "连续演算后截取目标
  period"); train = 1981-10-01..1995-09-30, test = 1995-10-01..2010-09-30.
- CN `psol_annual` / `g_thresh = 0.9 * psol_annual` uses the **window-based**
  R1/R2 historical semantics (the window passed to the forward).  The R3
  `canonical_cn_psol_annual` path is deliberately NOT used; per-basin
  psol/g_thresh for the export window and for the R1 per-period inference
  windows are recorded in every manifest.
- Base/TGD2 exports never construct pseudo-SWE columns.

## Commands

```bash
# DEV_ONLY smoke test (validates adapters, forward identity, alignment,
# fail-loud behavior, snow-reader parser).  Runs on CUDA when available.
python -m r4.smoke_test \
  --results-root /home/jingxin/code/dmg-research/project/hydrodiag/results \
  --data-root /home/jingxin/code/dmg-research/data \
  --device cuda

# After observation-trained artifacts are synced (sync_manifest.json):
# IC export (one structure at a time)
python - <<'EOF'
from r4.common import load_bundle, default_data_root, default_results_root, IC_CANONICAL_RUNS
from r4.input_adapters import read_ic_canonical
from r4.forward_export import export_run
from r4 import OFFICIAL_OBSERVATION_TRAINED
import torch
res, data = default_results_root(), default_data_root()
bundle = load_bundle(data)
bids = [str(b) for b in bundle.basin_ids]
for model, (run, raw) in IC_CANONICAL_RUNS.items():
    params, meta = read_ic_canonical(res / run, model, raw, bids)
    export_run(structure=model, parameters=params, parameter_meta=meta,
               basin_ids=bids, data_root=data, results_root=res,
               run_id=f"official_ic_{model}", tag=OFFICIAL_OBSERVATION_TRAINED,
               provenance={"source_run": run}, device="cuda")
EOF

# dPL export (per seed; median reduction is a downstream analysis step)
python - <<'EOF'
from r4.common import default_data_root, default_results_root, DPL_CANONICAL_RUNS, DPL_SEEDS
from r4.input_adapters import read_dpl_seed
from r4.forward_export import export_run
from r4 import OFFICIAL_OBSERVATION_TRAINED
import torch
res, data = default_results_root(), default_data_root()
from r4.common import load_bundle
bundle = load_bundle(data)
bids = [str(b) for b in bundle.basin_ids]
for model, run in DPL_CANONICAL_RUNS.items():
    for seed in DPL_SEEDS:
        seed_dir = res / run / model / f"seed_{seed}"
        params, meta = read_dpl_seed(seed_dir, model, data, bids)
        export_run(structure=model, parameters=params, parameter_meta=meta,
                   basin_ids=bids, data_root=data, results_root=res,
                   run_id=f"official_dpl_{model}_seed{seed}", tag=OFFICIAL_OBSERVATION_TRAINED,
                   provenance={"source_run": run, "seed": seed}, device="cuda")
EOF
```

## Caveats / current blockers

- TGD2 artifacts (IC `xaj_tgd2_cmaes_531_batched_v1`, dPL
  `dpl_camels_531_lite_v3_tgd2_dpl_audited`) are not present locally nor on
  the AutoDL node — the R4 TGD2 track stays blocked until they are recovered.
- The fused IC runs on the AutoDL node (5 starts × 200 generations) differ
  in protocol from the canonical paired_v2 runs (10 starts × 300 generations);
  they are valid observation-trained inputs for R4 but cannot byte-reproduce
  the R1 IC tables.
- CAMELS-US SWE mount (`G:\Dataset\CAMELS_US`) is a data dependency;
  `snow_reference.py` fails loudly with the probed locations when it is
  unavailable, and the rest of the pipeline does not block on it.