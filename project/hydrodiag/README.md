# Hydro Structure Diagnosis

Hydrological model structure correctness diagnosis toolkit.

## Project Goal

Diagnose the structural correctness of hydrological models by analyzing step-kernel behaviors,
gradient flows, and parameter interactions.

## Current Stage

Models, canonical integrity tests, and the active IC/dPL training entry points
are maintained together. Formal and historical result data is catalogued
under [`results/`](results/README.md); inactive code and runtime provenance is
preserved under [`archive/`](archive/README.md). File-placement rules for
future Codex work are defined in [`local-agent-setting.md`](local-agent-setting.md).

## Implemented Models

| Model | Step Kernel | Snow Module | Parameters |
|-------|-------------|-------------|------------|
| HBV | `_hbv_step` | Built-in | 12 |
| GR4J | `_gr4j_step` | None | 4 |
| XAJ | `_xaj_step` | None | 15 |
| SIMHYD | `_simhyd_step` | None | 10 |
| CemaNeige | `_cemaneige_step` | Standalone | 2 |
| CemaNeigeHyst | `_cemaneige_hyst_step` | Standalone | 4 |
| PrecipitationDelay | `_precip_delay_step` | Temperature-agnostic control | 2 |
| GR4J + CemaNeige | Both compiled separately | 2-parameter CemaNeige preprocessing | 6 |
| XAJ + CemaNeige | Both compiled separately | 2-parameter CemaNeige preprocessing | 17 |
| SIMHYD + CemaNeige | Both compiled separately | 2-parameter CemaNeige preprocessing | 12 |
| GR4J + PrecipitationDelay | Both compiled separately | 2-parameter precipitation-delay control | 6 |
| XAJ + PrecipitationDelay | Both compiled separately | 2-parameter precipitation-delay control | 17 |
| SIMHYD + PrecipitationDelay | Both compiled separately | 2-parameter precipitation-delay control | 12 |
| XAJ + TGD2 | TGD2 input router -> unchanged XAJ | `tgd_tau_warm`, `tgd_delta_tau_cold`; one generic memory state | 17 |

## Compilation Strategy

- `torch.compile(step_function, fullgraph=True)` on individual step kernels
- Eager Python time loop in `_step_loop`
- No compilation of `forward`, `_step_loop`, or full model
- No silent fallback — compile failures are hard errors

## Testing

```bash
python manuscript/scripts/shared/run_model_test_suite.py
python manuscript/scripts/shared/run_model_test_suite.py --full
```

The frozen model/test baseline and the responsibility of each test file are
documented in [`docs/model_test_baseline.md`](docs/model_test_baseline.md).

The active training modules are documented in
[`docs/training_baseline.md`](docs/training_baseline.md). dPL uses the
Lite-v2-compatible runner under `training/dpl/`; maintained IC adapters and
runners live under `ablation/` and `training/ic/`. Completed one-off runners
are retained in `archive/project_cleanup_20260730/`.

## Directory map

- Active scientific code: `models/`, `training/`, `ablation/`,
  `optimization/`.
- Tests and maintained tooling: `tests/`, `manuscript/scripts/`, `configs/`, `docs/`.
- Result source of truth: `results/`.
- Inactive code and exact remote snapshots: `archive/`.
- Manuscript and supplement tooling: `manuscript/`.

The root `outputs` and `experiment/results` names are compatibility symlinks;
new code must use `results/` directly.

## R1 Statistics

The reproducible R1 analysis is under `manuscript/scripts/`, with all task-owned
tables and audit files under `manuscript/results/R1/`. The current comparison is:

- IC-CMA-ES: XAJ-Base, XAJ-TGD, XAJ-CN.
- dPL-MLP: XAJ-Base, XAJ-TGD, XAJ-CN, and HBV as an absolute benchmark.
- Obsolete GD, historical TGD variants, and XNES outputs are excluded.

### Inputs and periods

The analysis uses the CAMELS-531 basin list in `data/531sub_id.txt` and the
`data/camels_dataset` pickle as the canonical numerical data source. This
pickle contains the daily forcing tensor, observed discharge, and 35 static
attributes. Basin IDs come from `data/gage_id.npy`, dates from the extracted
`data/camels_dates.npy` axis, and the forcing order is defined in code as
`P,T,PET`. The active loaders do not depend on `data/camels_forcing_v2.pkl`.

The fixed periods are warm-up `1980-10-01..1981-09-30`, train
`1981-10-01..1995-09-30`, and test `1995-10-01..2010-09-30`. Observed ft3/s is
converted to mm/day with the repository `area_gages2` conversion. Non-finite
and negative discharge is excluded; zero discharge remains valid. No discharge
is imputed.

Daily simulations are generated only from existing IC parameter JSON files and
dPL checkpoints. IC selects one restart per basin using the highest stored
train-period KGE, with the lowest restart number breaking ties. The selected
restart is reused for train and test. dPL keeps seeds 42, 123, and 2026
separate; basin-level effects are calculated within seed and then reduced by
the median across seeds. The current XAJ-TGD dPL artifact is
`checkpoint_epoch_100.pt` in
`results/dpl_camels_531_lite_v3_tgd2_dpl_audited/XAJ_TGD2`, selected as the
latest common valid epoch present in checkpoint metadata and `epoch_history.csv`.

### Metrics and signatures

The primary metric is the repository's standard KGE, not modified KGE-prime:

`KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)`

where `alpha = std(sim)/std(obs)` and `beta = mean(sim)/mean(obs)`. KGE, NSE,
PBIAS, and RMSE use the same finite, non-negative paired daily mask. Absolute
and paired summaries report basin count, median, quartiles, mean, standard
deviation, extrema, fraction positive where applicable, and 10,000 paired basin
bootstrap resamples using seed `20260730`. Wilcoxon and sign tests are
supplementary and receive Benjamini-Hochberg correction within their families.

Structural contrasts are CN-Base, TGD-Base, and CN-TGD for train and test;
generalization is train minus test; transfer is IC minus dPL. All contrasts use
matched basin sets. Snow relationships use the documented `frac_snow` field as
a continuous attribute. Water years begin on October 1. CT is the first day
when cumulative annual flow reaches 50 percent, and AMJJ is April-July flow
divided by water-year flow. Signature summaries require at least five complete
water years, with a three-year sensitivity retained. SPO is left unresolved
because the project does not define its start date, search window, and no-pulse
rule sufficiently to implement it without guessing.

### Reproduction

If the daily epoch-100 XAJ-TGD exports are absent, run the five CUDA partitions
sequentially. The loop waits for each partition to finish before starting the
next; it does not launch five processes concurrently:

```bash
for i in 0 1 2 3 4; do
  python manuscript/scripts/r1/build_r1_statistics.py \
    --mode daily-inference --models XAJ_TGD2 --paradigm dpl \
    --tgd2-epoch 100 --device cuda --batch-size 64 \
    --partition-count 5 --partition-index "$i" --partition-suffix "_part_$i" \
    --project-root /home/jingxin/code/dmg-research/project/hydrodiag \
    --results-root /home/jingxin/code/dmg-research/project/hydrodiag/results \
    --data-root /home/jingxin/code/dmg-research/data \
    --output-root /home/jingxin/code/dmg-research/project/hydrodiag/manuscript/results/R1/epoch100_partitions/part_$i || exit $?
done
```

Then rebuild the complete R1 statistical package from the existing daily
exports and online partition summaries:

```bash
python manuscript/scripts/r1/build_r1_statistics.py \
  --mode merge-partitions --partition-count 5 --tgd2-epoch 100 \
  --partition-root /home/jingxin/code/dmg-research/project/hydrodiag/manuscript/results/R1/epoch100_partitions \
  --project-root /home/jingxin/code/dmg-research/project/hydrodiag \
  --results-root /home/jingxin/code/dmg-research/project/hydrodiag/results \
  --data-root /home/jingxin/code/dmg-research/data \
  --output-root /home/jingxin/code/dmg-research/project/hydrodiag/manuscript/results/R1
```

The main outputs are `r1_basin_level_performance.csv`,
`r1_structural_effects_basin_level.csv`,
`r1_generalization_effects_basin_level.csv`,
`r1_snow_signatures_basin_level.csv`,
`r1_absolute_metrics_summary.csv`, `r1_paired_effects_summary.csv`,
`r1_bootstrap_intervals.csv`, `r1_statistical_tests.csv`, and
`r1_result_manifest.json`. The audit, exclusions, exact source paths, and
execution status are recorded in `manuscript/results/R1/`.

## Design Principles

1. Step-kernel compilation with forced fullgraph
2. Uniform forcing/params interface
3. Batch-basin parallel execution
4. Physical-scale parameter dictionaries
5. No legacy model library dependencies
