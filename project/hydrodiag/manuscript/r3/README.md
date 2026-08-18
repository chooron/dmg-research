# R3 — Controlled synthetic-truth experiment (XAJ-CN generating structure)

R3 runs a known-truth controlled synthetic experiment on the full CAMELS-531
basin set to answer four questions about snow-module deletion:

1. can Base/TGD2 recover the synthetic discharge after re-calibration;
2. do shared XAJ parameters deviate additionally from the generating truth
   when discharge is recovered;
3. do common internal states deviate from the generating truth;
4. how do these compensation traces differ between the IC-CMA-ES and dPL
   parameter-estimation regimes.

R3 is **not** an IC-vs-dPL benchmark, and it does **not** claim that
parameter deviation *causes* state deviation: parameter and state deviations
are two parallel diagnostics of structure deletion + recalibration.

## Frozen design constraints

- The formal experiment keeps the full 531 basins; any small basin subset is
  pilot/gate material only.
- IC and dPL share the same synthetic truth, basin set, forcing,
  train/validation/test split and warm-up convention.
- The dPL input is the full original basin-attribute vector (35 attributes,
  including `frac_snow` at index 3); `frac_snow` is kept as an environmental
  diagnostic axis, with no imposed monotone parameter relationship.
- Generating structure: XAJ-CN. Fitted structures: Base / TGD2 / CN.
- Calibration regimes: the repository's IC-CMA-ES and dPL pipelines, reused
  unchanged except for the calibration target (Q*).
- The main synthetic truth is noise-free.
- `Base-no-refit` is retained (shared XAJ params from `theta*`, snow module
  removed, no re-calibration). No `Base-rainonly-no-refit` variant exists.

## Module layout

| File | Purpose |
|---|---|
| `common.py` | canonical periods/roots, KGE/NSE/PBIAS, R1 CT/AMJJ water-year signatures, pilot subset selection |
| `truth_generator.py` | `theta* = g*(A)`: PCA (rank K, 95% explained variance) + ridge regression (5-fold basin CV, seed 20260730) from the full 35-attribute vector onto the CN-IC parameter manifold; diagnostics + manifest |
| `recorded_forward.py` | per-day state-recording forwards for CN/Base/TGD2 that replay the production compiled kernels exactly (validated bitwise against the production forward) |
| `generate_truth.py` | Phase 2: `theta*`, `Q*`, `X*`, CN snow diagnostics for all 531 basins + round-trip check |
| `run_base_no_refit.py` | Phase 3: Base with shared `theta*` params, no calibration; discharge + CT/AMJJ diagnostics vs `frac_snow` |
| `pilot.py` | Phase 4/5 gate orchestration (IC via `training/ic`, dPL via `training/dpl`) |
| `analyze_pilot.py` | gate outputs: discharge metrics vs Q*, shared-parameter deviations (`z_hat - z*`), state RMSE vs X*, run flags |

Production pipeline changes (minimal, additive):

- `training/ic/run_tgd2_batched_cmaes_531.py`: registered `XAJ` (15) and
  `XAJ_CN` (17) model keys; optional `--target-npz` to replace the
  calibration target with Q* (default behavior unchanged).
- `training/dpl/run_dpl_model.py`: attribute normalization statistics are
  always computed on the full configured basin list (canonical 531) before
  subsetting (`--max-basins` runs therefore share the full-run preprocessing
  semantics); optional `target_override_npz` config key to calibrate on Q*.

## g*(A) definition

1. `z = (p - lower) / (upper - lower)` for the 17 CN parameters (linear
   unit-normalized coordinates; the CN spec has no log-scaled parameters).
2. Standardize `z` over basins; SVD of the standardized field; rank `K` =
   smallest number of components reaching 95% cumulative explained variance
   (data-driven; K = 15 for the current CN-IC field).
3. Robust-normalize the full 35-attribute matrix (median/IQR, clipped ±5,
   exactly the dPL preprocessing) and regress the PCA scores on it with
   ridge regression; `alpha` chosen by 5-fold basin CV on a fixed grid
   (seed 20260730).
4. `g*(A) = mean_z + (scores_hat @ V_Kᵀ) * scale_z` mapped back to physical
   bounds and clipped (clip counts are reported in
   `gstar_diagnostics.json`).

`theta* = g*(A)` exactly; no random parameter residual is added. The mapping
is deterministic, non-neural, and from a different family than the dPL MLP.
The CN-IC field is only used to define the parameter manifold; it is never
copied basin-by-basin.

## Canonical run commands

```bash
# Phase 2 — truth generation (531 basins)
python manuscript/r3/generate_truth.py --device cuda

# Phase 3 — Base-no-refit
python manuscript/r3/run_base_no_refit.py --device cuda

# Phase 4/5 — pilot gate (engineering only; --dry-run to preview commands)
python manuscript/r3/pilot.py --stage cn-ic --dry-run
python manuscript/r3/pilot.py --stage all --device cuda

# gate outputs
python manuscript/r3/analyze_pilot.py --device cuda
```

## Result locations

All run products are written under the canonical results root
(`/home/jingxin/code/dmg-research/project/hydrodiag/results/`):

- `r3_synthetic_truth_v1/` — `theta_star.npz`, `q_star.npz` (key
  `target_mm_day`), `x_star.npz`, `snow_star.npz`, `final_states.npz`,
  `gstar_manifest.json`, `gstar_diagnostics.json`, `manifest.json`;
- `r3_base_no_refit_v1/` — per-basin metrics + summary;
- `r3_pilot_v1/` — pilot manifest + generated configs;
- `r3_pilot_v1_ic_<model>/` and `r3_pilot_v1_dpl_<model>_seed_<s>/` — fits;
- `r3_pilot_analysis_v1/` — machine-readable gate outputs.

R1/R2 observed results are never modified.

## Forward identity (Phase A fix, 2026-08)

The CN model computed its snow-cover threshold `g_thresh = 0.9 * psol_annual`
from the mean annual solid precipitation of the *input sequence*.  The truth
was generated on the full 12418-day record, while the IC objective
re-computed it per split and dPL per 730-day window — a systematic
preprocessing inconsistency that made correct-CN oracle identity impossible
(documented by `manuscript/r3/oracle_identity.py`; e.g. up to ~1.3 mm/day discharge
difference on snowy pilot basins from the g_thresh term alone).

Fix (backward compatible; default behavior unchanged when the key is
absent):

- the CN compositions (`models/composed.py`) and the standalone CemaNeige
  accept an optional `forcings["cn_psol_annual"]` (per-basin mean annual
  solid precipitation) and use it for `g_thresh` instead of re-estimating it
  from the call's own sequence;
- the IC runtime passes the canonical full-record value when the config flag
  `canonical_cn_psol_annual` is set; the IC runner sets that flag exactly
  when `--target-npz` is used (synthetic protocol);
- the dPL loader computes the canonical value from the full-record forcing
  when `target_override_npz` is set, and both the training loop and the
  evaluation path pass it through.

Oracle identity over all 531 basins (`results/r3_gate_v1/oracle_identity*`):

| path | canonical (fixed) median max-abs diff | split/window g_thresh (old) |
|---|---|---|
| full-axis recorded vs production | 0.0 | — |
| IC train split (theta* vs q_star) | 3.8e-6 | 0.216 |
| IC test split | 2.0e-4 (365-d warm-up effect) | 0.280 |
| dPL 730-d window | 2.0e-5 .. 3.2e-5 | 0.266 .. 0.278 |
| dPL evaluation path | 2.0e-4 | 0.280 |

The remaining canonical residual on split-start paths is the frozen 365-day
warm-up convention (the split runs start from default initial states 365
days before the scored period, the truth runs continuously from 1980-10-01);
it is bounded by ~2.3 mm/day on the snowiest basins and is not a g_thresh
redefinition.  `Base-no-refit`, IC and dPL all use the Lite model variants
(`XAJLite` / `XAJWithCemaNeigeLite` / `XAJWithTGD2Lite`), matching the
R1/R2 inference paths; CN Lite is numerically identical to CN full (same
kernels; only output storage differs).
