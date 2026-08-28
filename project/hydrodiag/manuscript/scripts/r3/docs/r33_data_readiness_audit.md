# R3 / Results 3.3 — Data Readiness Audit (controlled synthetic truth)

Read-only audit, 2026-08-21. Scope: freeze the data conditions for the
formal 3.3 statistics (structural-gap recovery, generic-control recovery,
truth-relative parameter/state recovery). No statistics were run, nothing
was modified. Companion documents: `docs/kge_audit.md`,
`R3_RECOVERY_AUDIT.md`, `R3_RECOVERY_INVENTORY.json`.

---

## A. Executive verdict

**`READY WITH DERIVATIONS`**

All four experimental branches (`Base_no-refit`, `Base_refit`, `TGD`, `CN_refit`)
exist at basin level for IC and dPL, with fitted parameters and daily
discharge (cached recorded-forward replay), and all per-basin ingredients of
`F_close` / `F_TGD` are traceable. No retraining or re-calibration is needed
for the frozen design. Remaining work is derivation / forward replay only.

Blocking / near-blocking items:

1. **Daily internal states of fitted structures are not persisted.**
   `r3_misspec_analysis_v1/` stores daily **discharge** (`posthoc_q_*.npy`)
   and state **metrics** (`state_metrics_basin.csv`, `state_excess.csv`),
   but not daily state series. Any daily-resolution truth-relative state
   analysis (Q3, seasonal trajectory at full resolution) requires the
   validated recorded-forward replay (`r3/recorded_forward.py`; forward-only,
   no training).
2. **`posthoc_summary.json` is not strict JSON** (contains `NaN` /
   `-Infinity` / `Infinity`). Python `json.load` accepts it; strict
   JSON readers / JS parsers fail. Same class of risk in
   `frac_snow_quartiles` bin edges.
3. **S1–S5 strata are not pre-joined into the R3 basin tables.**
   `posthoc_basin_table.csv` carries `frac_snow` (continuous) only;
   snow-stratified 3.3 summaries require joining
   `manuscript/results/R1/r1_snow_attributes.csv` (same 531 universe,
   same `basin_id`).
4. **IC budgets differ slightly across structures by design:** all IC runs
   use 10 starts × 300 generations, but population scales with dimension
   (`XAJ` 15D → pop 22 = 6,600 evals/basin; `XAJ_CN`/`XAJ_TGD2` 17D →
   pop 25 = 7,500). Documented rule (`population_for_dimension`); must be
   stated if IC between-structure variability is ever compared.
5. **dPL seed variability is only 3 seeds, best-epoch parameters only.**
   Per-epoch parameter vectors are not saved (`epoch_history.csv` tracks
   `val_kge` only); across-seed variability is limited to seeds 42/123/2026.
6. **Stale-risk directory:** `results/r3_gate_dpl_xaj_cn_localcheck_seed_42`
   is a temporary local cross-check (`_note`: "temporary, not the canonical
   gate dir", protocol `r3_gate_531_dpl_synthetic_target_v1`). Do not
   consume it as canonical.

---

## B. Canonical data inventory

Experiment logic (frozen): `CN_generating → Base_no-refit →
{Base_refit, TGD, CN_refit}`; regimes IC-CMA-ES (10 starts × 300 gens) and
dPL (3 seeds × 100 epochs); basin unit = 531; train 1981-10-01..1995-09-30,
test 1995-10-01..2010-09-30, warm-up 365 d (split paths);
truth axis 1980-10-01..2010-09-30 (12418 d, full-axis continuous).

| Experimental branch | Regime | Basin level | Restart/seed level | Daily outlet | Parameters | States/fluxes | Provenance | Status |
|---|---|---|---|---|---|---|---|---|
| `CN_generating` (truth v1) | — | 531 | — | `q_star.npz` [531,12418] | `theta_star.npz` [531,17] (+z, unclipped, clip_mask) | `x_star.npz` (7 states), `snow_star.npz` (6 vars), `final_states.npz` [531,9] | `r3_synthetic_truth_v1/`, manifest 2026-08-13, commit d59365e; roundtrip 0.0; `r3_recovery_manifest.json` all_pass | ✅ |
| `Base_no-refit` | IC=dPL (shared) | 531 | — (deterministic) | metrics only; daily q cached in `posthoc_q_Base_no_refit.npy` | `base_no_refit_parameters.npz` (15 shared θ* columns, verbatim) | not stored (replayable) | `r3_base_no_refit_v1/` 2026-08-15; summary.json (median test KGE 0.8980) | ✅ |
| `Base_refit` | IC | 531 | **10 restarts saved** (raw JSON ×10/start) | metrics in raw JSON; daily q cached | physical + normalized per start; best-restart rule (max train KGE, lowest start) | not stored (replayable) | `r3_misspec_ic_xaj_531_v1/` DONE 2026-08-16 | ✅ |
| `Base_refit` | dPL | 531 | 3 seeds (42/123/2026), best-epoch only | `basin_final_summary.val_kge` (test); daily q cached | `best_parameters_physical/normalized.npz` [531,15] per seed | not stored (replayable) | `r3_misspec_dpl_xaj_seed_{42,123,2026}/` COMPLETE | ✅ |
| `TGD` (=`TGD2`) | IC | 531 | **10 restarts saved** | raw JSON; daily q cached | per start; 17 params | not stored (replayable) | `r3_misspec_ic_xaj_tgd2_531_v1/` | ✅ |
| `TGD2` | dPL | 531 | 3 seeds | val_kge; daily q cached | [531,17] per seed | not stored (replayable) | `r3_misspec_dpl_xaj_tgd2_seed_{42,123,2026}/` | ✅ |
| `CN_refit` (correct-CN gate) | IC | 531 | **10 restarts saved** | raw JSON; daily q cached | per start; 17 params | gate state metrics basin CSV (CN baseline for delta_E) | `r3_gate_ic_xaj_cn_531_v1/` (rebuilt 2026-08-15) | ✅ |
| `CN_refit` | dPL | 531 | 3 seeds | val_kge; daily q cached | [531,17] per seed | same via `r3_gate_v1/gate_state_metrics_basin.csv` | `r3_gate_dpl_xaj_cn_seed_{42,123,2026}/` (rebuilt 2026-08-15/16) | ✅ |
| Posthoc package | — | 531 | per seed rows retained | `posthoc_q_<fit>.npy` ×14 fits [531,12418] f64 | `paired_parameters.csv` (e, e_cn, delta_e, delta_abs_e, tier) | `state_metrics_basin.csv`, `state_excess.csv` (delta_E) | `r3_misspec_analysis_v1/` 2026-08-16 | ✅ |
| Figure/table data | — | 531 | dPL seed-median aggregates | figure5/6 basin tables | figure5/6 summaries | `fig6_seasonal/{input,state}.npz` (monthly, WY, high-snow ≥q75) | `manuscript/results/R3/` 2026-08-17..19 | ✅ |

Missing entirely (status): nothing among the four branches × two regimes.
Engineering-only pilot (`r3_pilot_v1*`) was lost and not rebuilt — it is
not a scientific artifact. The pre-recovery `R3_RECOVERY_INVENTORY.json`
(2026-08-16) entries "LOST_REBUILD / NEVER_RUN" are superseded: all four
branches now exist locally (gates rebuilt 2026-08-15/16, misspec IC runs
2026-08-16, validated by `r3_recovery_manifest.json` all_pass for truth +
dPL Base/TGD2, and by `gate_analysis` outputs + `gate_report.json` for CN).

Timing / warm-up convention (documented, frozen):
- Truth: continuous full axis from default initial states; warm-up days
  retained in stored arrays (manifest).
- Split paths (IC objective, dPL windows, analysis replays): 365-d warm-up
  before each scored period from default states → small canonical residual
  vs. continuous truth (IC train 3.8e-6; IC test ~2e-4; dPL eval ~2e-4,
  bounded ~2.3 mm/d on snowiest basins; `results/r3_gate_v1/oracle_identity.json`,
  `oracle_dpl_audit.json`). Full-axis recorded-forward replays (posthoc)
  remove this by construction but re-derive KGE slightly differently
  (e.g. Base-IC test median 0.8988 raw JSON vs 0.8994 replayed).

## C. Estimand readiness matrix

KGE = repository standard KGE everywhere (see F). Per-basin inputs are all
in `r3_misspec_analysis_v1/posthoc_basin_table.csv`
(basin_id, paradigm, seed, period, kge_base_no_refit, kge_base, kge_tgd2,
kge_cn, G_base, F_close, G_tgd2, F_tgd2, frac_snow) plus `posthoc_theta_cost.csv`,
`posthoc_state_cost.csv`, `paired_parameters.csv`, `state_excess.csv`.

| Target quantity | Required inputs | Existing source | Can compute now? | Missing/ambiguity | Action needed |
|---|---|---|---|---|---|
| Raw Base recalibration recovery `G_base = KGE(Base_refit) − KGE(Base_no-refit)` | kge_base, kge_base_no_refit per basin/period/regime/seed | `posthoc_basin_table.csv` (column G_base, already bootstrapped in `posthoc_summary.json`) | ✅ yes | IC test median +0.0026, dPL +0.006..0.007 — frozen | — |
| Reference structural gap `KGE(CN_refit) − KGE(Base_no-refit)` | kge_cn, kge_base_no_refit | both columns per basin; not precomputed as a column | ✅ yes (derivation) | — | none |
| `F_close = G_base / (KGE_CN − KGE_Base_no-refit)` | numerator + denominator, per basin | computed in `posthoc_stats.py` (denom > 1e-6, no clipping); per-basin column F_close | ✅ yes | IC: valid 427 / excluded 104; dPL: 458..465 / 66..73; summary.json non-strict JSON | strict-JSON sanitize if non-Python consumers |
| Raw TGD recovery `G_tgd2 = KGE(TGD2) − KGE(Base_refit)` | kge_tgd2, kge_base | column G_tgd2 | ✅ yes | — | — |
| `F_TGD` (=F_tgd2) `= G_tgd2 / (KGE_CN − KGE_Base_refit)` | kge_tgd2, kge_base, kge_cn | column F_tgd2; denominator **≠ F_close denominator** (see note) | ✅ yes | IC: 466/65; dPL: 514..520/11..17 | report denominator difference explicitly |
| Correct-CN recoverability baseline | CN gate KGE + oracle | gate_report.json (IC test 0.9924, oracle gap 0.00645; dPL 0.9952..0.9953, eval ceiling 1.0); raw JSONs | ✅ yes | — | — |
| Shared-parameter truth error `e = (θ̂−θ*)/(upper−lower)`; `delta_e = e_M−e_CN`; `delta_abs_e` | θ̂ (fits), θ* (truth), bounds | `paired_parameters.csv` (all 15 shared params × basin × regime × seed, tier-labeled); θ* z/raw in theta_star.npz | ✅ yes | e is in normalized (fraction-of-range) units; physical-scale error must be re-derived from raw values | none |
| Shared-state truth error (RMSE/NRMSE/corr/bias, delta_E = E_M − E_CN) | daily states of fits vs x_star | metrics exist (`state_excess.csv`, `state_metrics_basin.csv`, `state_summary.json`) | ✅ yes (metrics) | **daily state series not persisted** for fits | forward replay to persist daily states |
| Seasonal trajectory error (Fig. 6 common input / wt) | monthly liquid input + wt per fit | `fig6_seasonal/` npz (input & state, WY-month means, test, high-snow ≥ q75 only) | ✅ (existing scope) | daily-level or full-basin seasonal profiles not persisted | forward replay (derivation) for extended scope |
| Recovery–internal-error association (raw + partial Spearman on frac_snow) | posthoc_basin_table + theta/state cost | `posthoc_validation_partial.csv`, `posthoc_validation_summary.json` (V1) | ✅ yes | IC partial ~0 (CIs include 0); dPL C_theta partial +0.08..+0.11; daily-state variants would need replay | — |

**`F_close` / `F_TGD` formula record (single canonical implementation,
`posthoc_stats.py`, `DENOM_TOL = 1e-6`):**

- `F_close = (KGE_Base_fitted − KGE_Base_no-refit) / (KGE_CN − KGE_Base_no-refit)`,
  denominator rule: `denom > 1e-6`, **no clipping** (values >1 = over-closure
  of the KGE gap; values <0 retained).
- `F_TGD = (KGE_TGD2 − KGE_Base_fitted) / (KGE_CN − KGE_Base_fitted)`,
  same rule. **Denominators differ**: F_close anchors on the no-refit
  knockout, F_TGD anchors on fitted Base. `F_explicit_residual =
  (KGE_CN − KGE_TGD2)/(KGE_CN − KGE_Base_fitted)` satisfies
  `= 1 − F_TGD` (max abs diff ≤ 4e-11 in V4) — an algebraic identity, not a
  nesting claim.
- Per-basin numerator, denominator and validity flag are recoverable from
  the four kge columns; no other F-ratio definition exists in current R3
  code (no conflicts found).

**Q2 checks feasible on existing data:** `F_TGD < 0`, `F_TGD > 1`,
`F_TGD < F_close` (note different denominators), TGD beyond CN or below
Base-refit — all per-basin identifiabilities exist from `posthoc_basin_table`
(F_TGD unclipped; frac_ge_1 not in summary but computable; F_tgd2 > 1 not
explicitly tabulated — derivable).

## D. Shared-parameter mapping (generating CN ↔ Base/TGD2/CN)

`COMMON_XAJ` (15 shared host parameters, R2 order, `r3/common.py`):
xaj_k, xaj_b, xaj_im, xaj_um, xaj_lm, xaj_dm, xaj_c, xaj_sm, xaj_ex,
xaj_ki, xaj_kg, xaj_ci, xaj_cg, xaj_a, xaj_theta.
Bounds (`models/parameter_specs.py::XAJ_PARAM_SPECS`; upper−lower is the
distance normalizer): k 0.5–2.0, b 0.1–2.0, im 0–0.3, um 5–50 mm, lm 20–200 mm,
dm 20–200 mm, c 0.05–0.3, sm 5–100 mm, ex 0.1–2.0, ki 0–0.7 d⁻¹, kg 0–0.7 d⁻¹,
ci 0.1–1.0, cg 0.9–1.0, a 0–2.9 (Gamma-UH shape), theta 0–6.5 (Gamma-UH shape).
No log-scaled parameters among the shared 15; all linearly unit-normalized
in IC/dPL optimization and in `e`.

| Structure | Vector | Shared host (15) | Structure-specific | Scale/transform |
|---|---|---|---|---|
| CN generating (truth) | 17 | exact xaj_* block (same order) | cn_ctg [0,1], cn_kf [0,10] (CemaNeige) | physical (f64); z_normalized + unclipped + clip_mask stored |
| Base (XAJLite) | 15 | all 15 | none | physical + normalized stored (IC raw, dPL npz); log scaling: none |
| TGD2 | 17 | all 15 (identical names/order) | tgd_tau_warm [1e-4,3] d, tgd_delta_tau_cold [0.1,180] d | τ params **log-scaled** in normalized space (`LOG_SCALED_PARAMETERS`); physical values stored |
| CN fitted | 17 | all 15 | cn_ctg, cn_kf | physical + normalized stored |

- Alignment: by parameter name via `COMMON_XAJ` (`misspec_analysis.py`
  maps each fit's own order; truth's parameter_names order = gate order).
  One-to-one, basin-level pairing exists in `paired_parameters.csv`
  (IC best-restart; dPL seed-matched to CN per seed 42/123/2026).
- Truth-relative error is currently defined in **normalized units**:
  `e = (θ̂ − θ*) / (upper − lower)`. Physical-scale error is derivable from
  stored physical θ̂/θ* (IC raw `parameters`, dPL `best_parameters_physical`,
  truth `parameters`).
- Restart/seed variability: IC **10 restarts per basin saved** (raw JSONs,
  parameters + theta_normalized + train/test metrics per start) —
  `ic_restart_parameter_dispersion.csv` (gate) exists; misspec raw JSONs
  allow the same. dPL: only 3 seeds × best-epoch vectors.
- Only one parameter-distance convention exists in current code (protocol
  `normalized_error` + `delta_abs_e`/`delta_e`); no competing definitions found.

## E. Comparable state/flux mapping

Truth states: `x_star.npz` (7 core states, full axis, continuous init);
`snow_star.npz` (6 CN snow diagnostics). Fitted-structure daily states are
reproducible via `recorded_forward.py` (validated bitwise vs production;
states recorded include: wu, wl, wd, s, fr, qi, qg, rs_instant, evap;
CN adds G/eTG/sca/rain/melt/effective_precip; TGD2 adds effective_precip,
tgd_storage, tgd_tau, tgd_retention). Units: wu/wl/wd/s all mm (tension /
free-water storages of the shared XAJ core); fr dimensionless 0–1; qi, qg
mm/d (interflow / groundwater release fluxes); rs_instant mm/d (surface
runoff generation pre-UH routing); eff. precip mm/d.

`PRIMARY_COMPARABLE` — same XAJ-core definitions in all three structures and
truth; direct truth-relative error valid:
- `wu`, `wl`, `wd` (upper/lower/deep tension storage, mm)
- `s` (surface free-water storage, mm)
- `fr` (free-water fraction, −)
- `qi`, `qg` (interflow/groundwater release fluxes, mm/d)
- derived `wt = wu + wl + wd` (protocol-registered; identical derivation in
  truth and fits) — protocol level: secondary variable, primary for Fig. 6.

`CONDITIONAL_COMPARABLE` — same semantic target, different generators;
needs aggregation/conversion and explicit caveats:
- `effective_precip` (liquid water entering the XAJ core): CN (CemaNeige)
  and TGD2 (tgd2_step) generate their own; Base uses raw precip (no
  transformation). Truth reference = CN `snow_star.effective_precip`.
  Pointwise Base-vs-truth error mixes precip definitions; seasonal/monthly
  trajectory comparison (as in Fig. 6) is the sanctioned use.
- `rs_instant` (surface-runoff generation): same XAJ core, but surface
  routing differs (CN `_route_xaj_surface_runoff` vs Base/TGD2
  `_route_xaj_surface_runoff_hydrodl2`); not present in truth `x_star`
  (must be replayed from θ*) — comparable only after replay + routing-
  kernel caveat.

`STRUCTURE_SPECIFIC` — not cross-structure state-error quantities
(protocol prohibitions):
- CN: `G` (SWE), `sca`, `eTG`, `melt`, `rain` — snow-process diagnostics.
- TGD2: `tgd_storage`, `tgd_tau`, `tgd_retention` — generic temperature-
  conditioned memory; **never paired against CN `G`** (protocol).
- Seasonal Fig. 6 uses `wt` + liquid input only; snow-state alignment is
  explicitly not claimed.

Alignment/quality: all daily series are on the same 12418-day axis
(1980-10-01..2014-09-30 in stored arrays; scored periods ≤ 2010-09-30),
basin-indexed identically; missing values: none in truth (q finite
non-negative, manifest); fits replayed from finite θ̂ (no missing).
Warm-up: truth continuous; fits split-replay warm-up convention documented
(bounded residuals); full-axis replays avoid it.

## F. Provenance and metric conflicts

1. **KGE vs NSE.** 3.3 canonical outlet metric = **standard KGE**
   (r/α/β with population moments; ≥30 valid days; obs_std ≥ 1e-10;
   zero discharge valid). All four KGE implementations audited in
   `docs/kge_audit.md` are the same formula; `posthoc_stats.py` KGE
   components reconstruct to max abs diff 0.0. NSE appears only as
   clearly-labeled extra columns (`base_no_refit_basin_metrics.csv`,
   `gate_discharge_metrics.csv`, `common.nse`) and in Table R1 (R1 scope).
   The "modified KGE′" wording survives only in legacy docs
   (HANDOFF / kge_audit / R3_RECOVERY_AUDIT) as a *documented discrepancy*;
   no live 3.3 script or canonical manuscript uses it. No 3.3 metric-mix risk.
2. **Run/version.** Truth = v1 (2026-08-13, commit d59365e, dirty);
   correct-CN gates were **rebuilt** 2026-08-15/16 after the 2026-08-16
   recovery inventory (original remote copies lost); misspec IC 2026-08-16;
   posthoc 2026-08-16; figure/table prep 2026-08-17..19. Current repo HEAD
   7027cee (dirty; unrelated uncommitted changes remain untouched).
   Manifests carry per-artifact commits (455c246, 7d1132b remote code,
   55b1ecb fig6 re-export).
3. **Seed/restart aggregation.** Not aggregated away: per-seed rows kept in
   `posthoc_basin_table.csv`; dPL seed-median aggregation happens only at
   figure/table presentation (`seed_med`). IC per-restart records all saved.
   Gate dPL val_kge medians 0.9953/0.9954/0.9954; misspec Base 0.9081/0.9080/0.9092;
   TGD2 0.9443/0.9447/0.9443 (per seed) — no accidental pooling.
4. **Old/stale summaries.** `r3_gate_dpl_xaj_cn_localcheck_seed_42` =
   explicit temporary cross-check (flagged, non-canonical).
   `R3_RECOVERY_INVENTORY.json` (2026-08-16) is stale-by-design (pre-rebuild).
   `results/archive/legacy_outputs_20260730` = old outputs, superseded.
5. **Missing source-level data.** Nothing among the four branches: all
   basin-level CSVs/NPZ/JSON present under `results/`; no summary-only
   quantities without basin-level source found for 3.3 (every manuscript
   `[[DATA:...]]` placeholder traces to a per-basin CSV or per-seed JSON).
6. **Naming (`TGD` vs internal `TGD2`).** Manuscript-facing name `TGD`;
   implementation `TGD2` everywhere in code/results (`XAJ_TGD2`,
   `tgd_*` params, `r3_misspec_*_tgd2_*`, `F_tgd2`, figure scripts).
   Mapping documented in R3 README; no rename performed.
7. **`r3/common.py::IC_RESULT_ROOTS`** point at *real-catchment* R1/R2
   result dirs (`xaj_base_cmaes_531_batched_paired_v2`, ...). Posthoc/misspec
   scripts override with explicit `r3_*` synthetic dirs, so no mixing — but
   the constant names are a foot-gun for new scripts.
8. **IC budget dimension-scaling** (pop 22 vs 25, §A.4) and **IC restart
   seeds are run-local** (e.g. 264637382 gate CN; 428360206 Base) with no
   cross-structure restart matching — by frozen protocol ("no restart
   matching across structures is invented").
9. **Non-strict JSON** in `posthoc_summary.json` (NaN/±Infinity) — §A.2.

## G. Minimal next-step data actions

**1. Derivation only (no model execution):**
- Join S1–S5 (`r1_snow_attributes.csv`) onto `posthoc_basin_table.csv` for
  snow-stratified F_close / F_TGD / G summaries and stratum-level partial
  associations (optional; continuous frac_snow already present).
- Compute explicit per-basin reference-gap column
  `KGE_CN − KGE_Base_no-refit` and F_TGD tail fractions (<0, >1, <F_close),
  TGD-vs-CN / TGD-vs-Base-refit per-basin comparisons from the existing
  four KGE columns.
- Physical-scale parameter truth error from paired raw θ̂/θ*.
- Sanitize `posthoc_summary.json` for strict-JSON downstream.

**2. Forward execution (recorded-forward replay only — deterministic,
validated vs production; no training):**
- Persist daily state series (wu, wl, wd, s, fr, qi, qg ± rs_instant, evap)
  for Base/TGD2/CN fits if the Q3 analysis requires daily state-error or
  full-basin seasonal trajectories beyond the stored monthly high-snow
  profiles; optionally replay θ* for truth-side rs_instant/evap.

**3. Retraining / recalibration:** **No retraining/recalibration required.**
All branches × regimes exist at basin level with parameters (10 IC restarts,
3 dPL seeds), and official KGEs are frozen in gate/misspec artifacts.
(Sampling additional dPL seeds or additional IC starts would be new
training — not required by the frozen design.)