# R3 / Results 3.3 — Canonical Estimand & Figure-Readiness Audit

Read-only audit, 2026-08-21 (follow-up to `docs/r33_data_readiness_audit.md`,
which concluded READY WITH DERIVATIONS). Scope: freeze the **outlet estimand
stack**, the **excess-error baselines**, and the **F5/F6 evidence
architecture**. No training, no recalibration, no modification of canonical
results. New derived basin-level artifacts (from canonical sources only):
`docs/estimand_audit/derive_estimand_audit.py`,
`docs/estimand_audit/audit_estimands_basin.csv`,
`docs/estimand_audit/audit_summary.json`.

All numbers below were re-derived basin-wise from
`results/r3_misspec_analysis_v1/posthoc_basin_table.csv`,
`posthoc_theta_cost.csv`, `posthoc_state_cost.csv`,
`paired_parameters.csv`, and
`manuscript/results/R1/r1_snow_attributes.csv` (S1–S5 join), with the
canonical conventions (standard KGE; basin unit; paired bootstrap 2000 reps
seed 20260730; dPL per-seed ratio → per-basin seed median; denom rule
`> 1e-6`, unclipped). The association audit reproduces the frozen
`posthoc_validation_partial.csv` values exactly (e.g. IC G_base vs C_state:
raw 0.6607 / partial +0.0054), validating the derivation.

---

## A. Executive verdict

1. **Reference gap frozen as `D_b = KGE(CN_refit,b) − KGE(Base_no-refit,b)`** — the only
   ratio denominator in current code. No implementation uses `CN-generating`
   scores or theoretical KGE=1 as a ratio denominator; the `1 − KGE_CN`
   quantity (F5 panel a, `prepare_figure5_data.py`) is a display-only
   "deficit" ECDF of **CN-refit**, correctly labeled.
2. **`F_close` is statistically stable at the median (IC 0.101, CI [0.080,
   0.113]; dPL seed-median 0.101, CI [0.089, 0.109]) but has a large
   per-basin spread** (26–32 % negative; |F|>5 in 5–12 basins). It is a
   legitimate **secondary normalized quantity**. The **primary** 3.3 outlet
   evidence for parameter-estimation recovery should be the raw paired
   `G_base` (median +0.0026 IC / +0.0073 dPL, CI excluding 0 for dPL).
3. **Current `F_TGD` (denominator `CN_refit − Base_refit`) does not answer
   the frozen Q2.** Q2 asks recovery from the *imposed structural knockout*;
   the current ratio answers "how much of the residual Base-refit→CN gap is
   closed *after* Base re-estimation" and uses a **different basin set**
   (IC 466 vs 427 valid).
4. **Candidate `F_TGD*` = `(KGE_TGD − KGE_Base_no-refit)/D` is the
   common-reference fraction consistent with the frozen Q2 and shares the
   exact `F_close` basin set and denominator.** Empirical (test): IC median
   0.546 [0.519, 0.568], dPL 0.523 [0.505, 0.536]; 87–93 % of valid ratios
   in [0,1] (vs 68–74 % for F_close); tail |F|>5 in 5 basins both regimes —
   distributionally comparable to or better than F_close, and much tighter
   than the raw F_close spread. **Recommendation: `REPORT BOTH WITH
   DIFFERENT ROLES`** — `F_TGD*` as the manuscript-facing generic-control
   fraction, current `F_TGD` retained as a secondary "incremental-over-Base"
   quantity (or moved to the supplement), never combined additively.
5. **Raw recovery should be the F5 primary evidence; fractions are
   normalized secondary summaries.** Raw `G_base` / `G_TGD_ko` and their
   fractions give the same S1→S5 direction; the ratio adds no information
   about ordering and loses 104 (IC) / 71 (dPL) basins to the D≤0 rule.
   KGE ceilings are not degenerate in the test period (CN ≥0.99 in 305/531
   IC, 407/531 dPL; only 26/40 basins ≥0.999), so raw differences are
   resolvable; train-period CN KGE is near-ceiling (314/531 ≥0.999) and raw
   differences there must be read cautiously.
6. **Parameter excess error must be defined over the full 15 shared params,
   not the frozen tier subsets.** Current `C_theta_primary` is a median over
   `{xaj_k}` **only** under IC (single parameter) and over 5 params under
   dPL; `C_theta_primary_secondary` covers {k,b} (IC). No 15-param aggregate
   is persisted (derivable from `paired_parameters.csv`; derived medians
   stable: |e|-median 0.167 IC / 0.065–0.072 dPL; excess vs CN
   median|e_M|−median|e_CN| +0.080 / +0.034–0.037).
7. **State excess error is correctly CN-refit-baselined** (`delta_E =
   E_M − E_CN`, CN gate state metrics as baseline; IC best-restart / dPL
   seed-matched pairing per frozen protocol). Primary metric NRMSE
   (denominator `std(truth)+1e-8`) is **not stable for `wd`** (truth test
   std min = 0, q10 ≈ 7.6e-6 across basins) and is marginal for `fr/qi/qg`
   (small but nonzero std) — consistent with the frozen protocol (wd
   secondary). Recommend `wt` as the F6 headline storage quantity and
   `wu/wl/wd` only as labeled components; `qi/qg` must be termed fluxes.
8. **Historical "raw association attenuates after controlling frac_snow"
   is confirmed from basin-level source and is snow-activity-organized** for
   Base recovery (G_base vs C_state: raw 0.66 IC / 0.82 dPL → partial
   +0.01 / −0.06..−0.02) and for TGD-vs-knockout recovery vs state cost
   (partials ≈ −0.06..−0.19). A **weak residual remains only** for the
   incremental TGD-vs-Base gain vs the state-cost *reduction* (partial
   +0.25 IC, +0.29..+0.41 dPL) and for dPL G_base vs C_theta (partial
   +0.08..+0.11). Verdict flags:
   `SNOW_ACTIVITY_COMMON_ORGANIZER_SUPPORTED` for the main Base claims;
   `RESIDUAL_ASSOCIATION_REMAINS` for the two incremental pairs.
9. **F5 can begin formal plotting as soon as the estimand stack is applied
   (see §H). F6 can also begin — `Option A (no daily-state replay)` — since
   every F6 panel input already exists** (posthoc CSVs, Fig6 seasonal
   monthly NPZs, process-conditioned errors). Daily-state replay is only
   needed for future daily-resolution analyses, not for the frozen figure.
10. **`No retraining/recalibration required`** — the only open action items
    are derivation (strata join, F_TGD* column, 15-param aggregates) and
    deterministic forward replay if daily-resolution internal error is ever
    added to the main text.

---

## B. Frozen object / reference definitions

| Object | Definition | Role in 3.3 |
|---|---|---|
| `CN-generating` (truth) | `r3_synthetic_truth_v1/` θ*, Q*, X*, snow diagnostics; XAJ-CN with θ*=g*(A), noise-free | **the only truth**; never a ratio reference |
| `Base-no-refit` | XAJLite forward with the 15 shared θ* params, snow representation removed, no calibration | **imposed structural deficit** (reference baseline for all ratios) |
| `Base-refit` | XAJLite, IC best-train-KGE restart / dPL seed-matched fit to Q* | parameter-estimation recovery |
| `TGD-refit` | XAJ_TGD2 (TGD2), same regime protocols | generic temperature-conditioned control recovery |
| `CN-refit` | correct-structure fit, same regime protocols (gate) | **recoverability / estimation baseline** (ratio denominator side) |
| truth reference | θ*, X*, Q* | distance origin for parameter/state errors |
| recoverability reference | KGE(CN_refit) per basin/regime/seed | normalizer for F-family fractions; not truth |

Roles verified in code: KGE columns used in `posthoc_stats.py`,
`posthoc_validation.py`, `prepare_figure5_data.py`, `prepare_figure6_data.py`,
`generate_table_r3_{main,si}.py` all read `kge_cn` from the fitted gate
runs; `gate_analysis.py` oracle quantities (θ* through IC/dPL paths) are
reported as **gaps/differences** (`oracle_gap = oracle − fitted`), never as
denominators. No `CN-generating`-denominated ratio exists.

## C. Outlet estimand audit (test period primary; train in CSV/JSON)

| Estimand | Formula | Scientific question | Denominator | Valid N (IC / dPL seed-med) | Stability | Recommended role |
|---|---|---|---|---|---|---|
| Imposed gap `D_b` | `KGE_CN − KGE_Base_no-refit` | size of correct-structure-recoverable knockout deficit | — | 531 / 531 | IC med 0.0867 (q25 0.0090, q75 0.356); dPL med 0.0911 (0.0181, 0.355) | primary context (panel) |
| Raw Base recovery `G_base` | `KGE_Base_refit − KGE_Base_no-refit` | how much parameter re-estimation closes the deficit | — | 531 / 531 | IC med +0.0026 [0.0008, 0.0080], 43.7 % ≤0; dPL med +0.0073 [0.0039, 0.0099], 34.5 % ≤0 | **PRIMARY evidence** (raw paired) |
| `F_close` | `G_base / D` | normalized parameter-estimation closure fraction | `D_b > 1e-6` | **427/104** / **468/63** | med 0.101 [0.080, 0.113] / 0.101 [0.089, 0.109]; lt0 31.6 % / 26.3 %; >1: 2 / 1; |F|>5: 5 / 12 | SECONDARY normalized summary |
| Raw TGD recovery from knockout `G_TGD_ko` | `KGE_TGD − KGE_Base_no-refit` | generic control recovery from the imposed knockout | — | 531 / 531 | IC med +0.0386 [0.0254, 0.0508], 24.1 % ≤0; dPL med +0.0360 [0.0268, 0.0446], 18.6 % ≤0 | **PRIMARY evidence** for the generic control |
| Current `F_TGD` (`F_tgd2`) | `(KGE_TGD − KGE_Base_refit)/(KGE_CN − KGE_Base_refit)` | incremental closure of the *residual Base-refit→CN* gap after re-estimation | `CN−Base_refit > 1e-6` | 466/65 / 528/3 | med 0.510 [0.487, 0.541] / 0.504 [0.473, 0.529]; lt0 9.2 % / 7.8 %; |F|>5: 4 / 1 | SECONDARY (incremental; different basin set) |
| Common-reference `F_TGD*` | `G_TGD_ko / D` | generic control recovery of the correct-structure-recoverable gap from the same knockout | identical to `F_close` (`D_b > 1e-6`) | **427/104** / **468/63** | med 0.546 [0.519, 0.568] / 0.523 [0.505, 0.536]; lt0 8.4 % / 7.1 %; >1: 17 / 4; |F|>5: 5 / 5; IQR (0.37–0.72) / (0.35–0.66) | **manuscript-facing generic-control fraction** (candidate) |
| Correct-CN recoverability | KGE(CN_refit); oracle gap | attainable ceiling per regime | — | 531 / 531 | IC 0.9924 test; dPL 0.9952–0.9953; oracle gap 0.00645 / 0.00406 | context panel (a) | primary context |

Train-period (for the decay framing only): F_close IC 0.185 / dPL 0.138 (seed-median; per-seed 0.126–0.143);
F_TGD* IC 0.651 / dPL 0.584; G_base IC +0.016 / dPL +0.0096–0.0105 (per-seed);
G_TGD_ko IC +0.054 / dPL +0.040 (seed-median).

### Algebraic relationships (identities only — no causal/additive reading)

Define `R = Base_no-refit`, `B = Base_refit`, `T = TGD`, `C = CN_refit`,
`D = S_C − S_R`, `d2 = S_C − S_B`.

- `G_base = S_B − S_R`;  `G_TGD_ko = S_T − S_R`;  `G_tgd2 = S_T − S_B`.
- `G_TGD_ko = G_tgd2 + G_base` — **algebraic identity** (telescoping), not
  additive decomposition of effects.
- `F_close = G_base/D`;  `F_TGD* = G_TGD_ko/D = F_close + G_tgd2/D` —
  identity; the second term uses the F_close denominator, not d2.
- `F_TGD(current) = G_tgd2/d2`;  `F_explicit_residual =
  (S_C − S_T)/d2 = 1 − F_TGD(current)` — identity (verified ≤4e-11 in V4).
- **Nothing of the form `F_close + F_TGD + residual = 1` holds** (different
  denominators); do not construct a 100 % decomposition.

## D. Ratio diagnostics

Denominator distribution `D_b` (test):
- IC: ≤0 **104**; (0,1e-6] **0**; (1e-6,1e-4] **0**; (1e-4,1e-3] **3**;
  >1e-3 **424**; median 0.0867.
- dPL (seed-median): ≤0 **71**; (0,1e-6] **0**; (1e-6,1e-4] **0**;
  (1e-4,1e-3] **6**; >1e-3 **454**; median 0.0911.
- Spearman(D, frac_snow) = 0.943 (IC) / 0.953 (dPL).
- dPL nuance: the D bins use the per-basin **seed-median** D (71 basins ≤ 0), while
  F_close/F_TGD* exclude only basins with **no valid per-seed ratio** (63):
  per-seed D can cross zero within a basin across seeds, so 8 basins have
  median D ≤ 0 yet ≥1 valid seed ratio. The valid-set statistics (468 basins)
  are what the manuscript-facing medians use; the D≤0 count (71) is the
  stricter per-basin statement.
- **The exclusion boundary is exactly D≤0**: zero basins in the
  (0,1e-6] bin, so the `DENOM_TOL=1e-6` choice is numerically immaterial
  (implementation convention with a zero-mass guard band). No sensitivity
  audit of the threshold value is required; a stability audit of the
  *estimand family* (raw vs ratio) is, and it is provided in §C/E.
- Excluded-set behaviour (no hidden positive recovery): G_base median
  −0.0099 (IC) / −0.0062 (dPL), G_TGD_ko −0.0132 / −0.0036; frac(G_base>0)
  0.067 / 0.028 on the excluded 104 / 71 basins (S1-dominated negative
  control), vs +0.0134 / +0.0118 on the valid set.

Current F_TGD denominator `d2` (test): ≤0 in 65 (IC) / 11 (dPL seed-med);
(1e-6,1e-4]: 0 / 0; medians 0.0812 / 0.0835 — mildly smaller than D and
with fewer exclusions; **the two F-families therefore sit on different basin
sets (427 vs 466 IC)**, which is the concrete reason their values are not
directly comparable and why F5 panels should share the F_close/F_TGD*
set when juxtaposed.

S1–S5 valid-rate organization (IC test / dPL seed-med test):
| Stratum (n) | S1 (165) | S2 (156) | S3 (121) | S4 (34) | S5 (55) |
|---|---|---|---|---|---|
| F-valid rate IC | 0.424 | 0.949 | 0.992 | 1.0 | 1.0 |
| F-valid rate dPL | 0.648 | 0.974 | 0.992 | 1.0 | 1.0 |
| F_close med IC | −0.096 | +0.002 | +0.136 | +0.163 | +0.208 |
| F_TGD* med IC | 0.508 | 0.498 | 0.549 | 0.509 | 0.831 |
| F_TGD* med dPL | 0.448 | 0.501 | 0.531 | 0.481 | 0.811 |
| G_base med IC | −0.005 | ≈0 | +0.045 | +0.075 | +0.178 |
| G_TGD_ko med IC | −0.003 | +0.029 | +0.182 | +0.284 | +0.834 |

Valid-rate decline is concentrated in S1 (frac_snow<0.05), where D is
negative or ≈0 by construction; S2–S5 are essentially fully valid. The
fractions move in the same snow direction as the raw quantities; F_TGD* is
flatter across S1–S4 than F_close (the knockout-normalized reference removes
the small-denominator inflation visible in S1), with the largest value in S5.

## E. Parameter truth-error audit

> **CORRECTED 2026-08-21 by `docs/r33_param_truth_error_correction.md`.**
> The aggregate `median_p |e_M,p − e_CN,p|` (`C_theta15`) measures
> **shared-parameter separation from CN-refit** (θ* cancels in the
> difference), NOT truth-relative excess error. Canonical F6 parameter
> quantities are now `E_param_M = median_p |e_M,p|` (distance to generating
> truth) and `E_param_excess_M = E_param_M − E_param_CN` (correct-CN-
> adjusted excess, sign-preserved). Values below were derived before the
> correction; §3–§4 of the correction note supersede the numbers here.

Frozen comparison: only the **15 shared host parameters** (`COMMON_XAJ`,
order: xaj_k, xaj_b, xaj_im, xaj_um, xaj_lm, xaj_dm, xaj_c, xaj_sm, xaj_ex,
xaj_ki, xaj_kg, xaj_ci, xaj_cg, xaj_a, xaj_theta) with
`e = (θ̂ − θ*)/(upper − lower)` (normalized-by-range, physical units
unavailable; no log-scaled shared param). All 15 are present per basin ×
regime × seed in `paired_parameters.csv` (columns `e`, `e_cn`, `delta_abs_e`,
`delta_e`, `tier`).

Answers:
1. **Current aggregate**: `C_theta_primary` (posthoc) = *median over the
   frozen primary-tier parameter set of |e_M − e_CN|* = `|delta_e|` of
   **xaj_k only under IC** (ic_primary = {xaj_k}), and of 5 params under
   dPL. `C_theta_primary_secondary` = median over {xaj_k, xaj_b} for IC.
   Not mean-abs, not RMSE, not Euclidean; and **not** over the full 15.
2. `delta_abs_e` is **parameter-wise** (per basin × parameter row in
   `paired_parameters.csv`); basin-level aggregation happens only inside
   C_theta.
3. Pairing: dPL is seed-matched (Base/TGD2/CN share seeds 42/123/2026);
   IC is best-train-KGE-restart per basin for all three structures — **no
   cross-structure restart matching exists**, which is exactly the frozen
   protocol rule ("no restart matching across structures is invented");
   CN and Base/TGD2 restart selections are therefore independent draws from
   restart variability.
4. Negatives: C_theta is absolute (never negative). **Signed excess exists
   only at the per-parameter level** (`delta_e` in paired_parameters.csv)
   and in the derived 15-param difference median|e_M|−median|e_CN|
   (20.7 % negative IC; 6.0–7.5 % dPL). The main-text excess claims should
   either keep the absolute form (C15) or explicitly define the signed
   variant; do not silently mix the two.
5. Normalized-by-range: **keep as primary** — it is the frozen protocol
   definition, comparable across parameters, and free of the log-scale
   issue; physical-unit per-parameter errors remain available
   (IC raw `parameters`, dPL `best_parameters_physical`, truth
   `parameters`) for **supplemental, parameter-specific** description only
   (cross-parameter physical aggregation is scale-mixing and is not
   recommended).

**Minimal canonical definition (recommended):**
- Per basin/regime/seed: `C_theta15(M) = median over the 15 shared params
  of |e_{M,p} − e_{CN,p}|` (excess, absolute) — derived from
  `paired_parameters.csv` (medians: IC 0.182, dPL 0.054–0.062);
- a clearly-labeled signed companion `Δbar_e = median_p|e_M| −
  median_p|e_CN|` only if sign is wanted (IC +0.080, dPL +0.034–0.037);
- per-parameter display stays tier-labeled (frozen tiers), but **no
  basin-level aggregate may silently collapse to the IC single-parameter
  tier**.
No new distance family is introduced; this re-aggregates the existing
per-parameter errors.

## F. State/flux truth-error audit

Frozen comparison: common variables only; `delta_E(b) = E_M(state) −
E_CN(state)` with the CN gate state metrics (`r3_gate_v1/
gate_state_metrics_basin.csv`) as the estimation baseline; IC best-restart /
dPL seed-matched; metrics RMSE, NRMSE, bias stored per variable/period in
`state_metrics_basin.csv`, paired in `state_excess.csv`; periods train/test
+ DJF/MAM/JJAS.

Answers:
1. `delta_E` is exactly `E_M − E_CN` (verified in `misspec_states.py`);
   no other baseline is used.
2. Current primary metric: **NRMSE** (`state_summary.json` headline;
   `posthoc_state_cost` selects metric=="nrmse"); RMSE/bias retained;
   Pearson corr exists in the metrics CSV but is **not** aggregated into
   delta_E.
3. NRMSE denominator `std(truth)+1e-8`: **unstable for `wd`** (truth test
   std min = 0, q10 ≈ 7.6e-6) and marginal for `fr` (min 0.02), `qi`
   (min 0.038), `qg` (min 0.011) — matches the protocol's decision to make
   `wd` secondary; treat `fr/qi/qg` NRMSE as diagnostics, not headline
   magnitude metrics.
4. **Primary magnitude quantity: `wt = wu+wl+wd`** (identical derivation
   in truth and fits; already the F6 state quantity) with **wu/wl/wd as
   labeled components**; this avoids the wd denominator instability by
   construction.
5. `qi/qg` are **fluxes** (interflow/groundwater release, mm/d) — term them
   fluxes; corr/bias (not NRMSE) are their safer comparators.
6. `effective_precip` is trajectory-only (seasonal water-delivery; Fig6
   input panel); it must not enter a joint Euclidean state distance with
   storage variables.
7. TGD2 `tgd_storage`/`tgd_tau`/`tgd_retention`: **DO_NOT_CROSS_COMPARE**
   with CN `G`/snow diagnostics (protocol prohibition).

Classification:
| Class | Variables |
|---|---|
| `PRIMARY_COMMON_STATE/FLUX` | `wt` (headline), `wu`, `wl`; fluxes `qi`, `qg` (NRMSE-cautious; corr/bias as primary for fluxes) |
| `SECONDARY_COMMON_STATE/FLUX` | `wd` (with explicit low-variance caveat), `s`, `fr` |
| `TRAJECTORY_ONLY` | `effective_precip` (CN/TGD2/Base raw precip), monthly WY seasonal profiles |
| `DO_NOT_CROSS_COMPARE` | TGD2 `tgd_*` vs CN `G/sca/melt/rain/eTG`; any snow-state alignment Base↔CN |

## G. Association audit (recovery ↔ internal excess error; test; basin-level)

Derived with rank-residual partial Spearman controlling continuous
frac_snow (identical method to `posthoc_validation.py`; values reproduce
the frozen V1 table exactly).

| Pair | IC raw → partial | dPL raw → partial (range of 3 seeds) |
|---|---|---|
| G_base ↔ C_theta[Base] | 0.356 → −0.078 | 0.77–0.78 → +0.08..+0.11 |
| G_base ↔ C_state[Base] | 0.661 → **+0.005** | 0.82 → −0.06..−0.02 |
| G_TGD_ko ↔ C_theta[TGD2] | 0.256 → −0.243 | 0.77–0.81 → +0.11..+0.17 |
| G_TGD_ko ↔ C_state[TGD2] | 0.614 → −0.062 | 0.79–0.80 → −0.19..−0.18 |
| G_tgd2 (vs Base) ↔ ΔC_theta (Base−TGD2) | 0.198 → +0.182 | 0.16–0.28 → +0.09..+0.17 |
| G_tgd2 (vs Base) ↔ ΔC_state (Base−TGD2) | 0.376 → **+0.248** | 0.61–0.68 → **+0.29..+0.41** |
| C_theta[Base] ↔ frac_snow | 0.496 | 0.87–0.88 |
| C_state[Base] ↔ frac_snow | 0.823 | 0.96–0.97 |

S1–S5 within-stratum raw Spearman (G_base ↔ C_state; per-seed): mostly small
(|ρ| ≲ 0.4), **sign-unstable across strata and seeds** (e.g. dPL S4 negative
−0.04..−0.12, S5 +0.16..+0.18); no consistent within-stratum association
signal. Small strata (S4 n=34, S5 n=55) make these descriptive only.

Verdicts:
- **G_base ↔ internal excess error: controlled by the shared snow-activity
  gradient (partials ≈ 0–0.11)** →
  `SNOW_ACTIVITY_COMMON_ORGANIZER_SUPPORTED` (consistent with the frozen V1
  determination; the dPL C_theta residual +0.08..+0.11 is weak and CI
  includes 0 for 2 of 3 seeds).
- **G_TGD_ko ↔ C_state[TGD2]: same verdict**
  (`SNOW_ACTIVITY_COMMON_ORGANIZER_SUPPORTED`; partials ≈ −0.06..−0.19).
- **Incremental TGD-beyond-Base gain ↔ state-cost reduction survives
  partial control** (+0.25 IC, +0.29..+0.41 dPL) →
  `RESIDUAL_ASSOCIATION_REMAINS` for this pair only. Reported as a
  direction, not a causal trade-off; it involves the *reduction* of excess
  state cost (ΔC_state), not a distortion–recovery exchange.
- No "compensation–distortion trade-off" claim is supported by any pair.

## H. F5 / F6 readiness

### F5 — "How much of the imposed outlet deficit is recovered?"
Evidence architecture available now:
- imposed gap `D` (per basin, per regime/seed, per period) ✓
- raw Base recovery `G_base` + CI ✓ (existing posthoc/Figure5 data)
- raw TGD recovery `G_TGD_ko` + CI ✓ (derivation; not yet in
  figure5_basin_table — must be added)
- correct-CN recoverability (KGE_CN + oracle gap) ✓ (panel a exists)
- `F_close` ✓ (panel c exists)
- recommended TGD fraction `F_TGD*` ✓ (derivation; panel b of F6 currently
  uses `F_tgd2` — switch source column when the estimand stack is frozen)
- snow-activity organization (frac_snow quartiles + S1–S5 join) ✓
- IC/dPL as secondary estimation-regime dimension ✓
- Train→test decay (existing panel d) ✓

**F5 READY AFTER DERIVATION** — minimal derivation actions: add
`G_TGD_ko` (and optionally `F_TGD*`) to `figure5_basin_table.csv`-
generating step; join S1–S5; re-check the panel (a) label "Correct-CN
baseline" explicitly reads as CN-refit recoverability.

### F6 — "Does outlet recovery imply recovery of the internal generating system?"
Evidence architecture available now (all existing):
- shared-parameter excess error (C_theta / 15-param aggregates) ✓
  (derive 15-param aggregate per §E before final numbers)
- common-state/flux excess error (delta_E, NRMSE/corr/bias) ✓
- seasonal liquid-water delivery (`fig6_seasonal_input.npz`) and storage
  (`fig6_seasonal_state.npz` = wt, WY-monthly, test, high-snow ≥q75) ✓
- recovery–internal-error conditioned audit (V1 raw/partial Spearman; §G re-derivation) ✓
- correct-CN baseline (gate state metrics; S5 R-state relief) ✓
- process-conditioned residual (posthoc_process_errors; V4) ✓

**F6 READY (Option A — no daily-state replay)** for the frozen figure
scope: every panel's input file exists (posthoc CSVs + fig6_seasonal NPZs).
Daily-resolution state analysis (e.g., state error on melt days, daily
trajectory panels) is NOT required by the current F6 duties; if adopted,
it would require a deterministic recorded-forward replay of the 7 common
states + wt for all 14 fits (forward-only, ~CPU-hours, no training).

### Minimal remaining actions before plotting
1. Freeze the estimand stack: primary = raw `G_base` / `G_TGD_ko`; secondary
   = `F_close` and `F_TGD*` (common denominator set); retain current
   `F_TGD` only as an explicitly-labeled incremental quantity (supplement).
2. Add `G_TGD_ko`, `F_TGD*` columns to the figure data prep (derivation).
3. Join S1–S5 into figure tables; report N_valid per stratum with D≤0
   exclusions visible.
4. Replace F6 panel-b source `F_tgd2` → `F_TGD*` (both may be shown with
   roles labeled).
5. Derive the 15-param `C_theta15` aggregate for F6 internal panels (no
   single-parameter IC collapse).
6. State terminology: `wt` headline; `qi/qg` as fluxes; no tgd_* vs G
   comparisons.

## I. Minimal next actions

1. **Derivation only (no model execution):**
   - Add `F_TGD*` and `G_TGD_ko` to the figure-data pipeline (formula and
     basin set identical to F_close; script provided in
     `docs/estimand_audit/derive_estimand_audit.py`);
   - 15-param `C_theta15` aggregate + labeled signed companion;
   - S1–S5 join for all figure/table outputs;
   - re-label any "correct-CN" display quantity as CN-refit recoverability.
2. **Targeted forward replay (only if daily-resolution internal evidence is
   approved for main text):** replay the 7 common states (+wt) for the 14
   fits and θ* truth side as needed; deterministic, validated kernels
   (`recorded_forward.py`), no training. Not required for the frozen F5/F6.
3. **Retraining / recalibration:** **No retraining/recalibration required.**

---

## Acceptance answers (mapped)

1. Reference gap = `CN_refit − Base_no-refit` ✓ (sole definition in code;
   no generating-denominated ratio).
2. `F_close` statistically stable at the median; due to 26–32 % negative
   mass and tail sensitivity → **secondary**; raw `G_base` primary.
3. Current `F_TGD` answers "incremental closure of the residual Base-refit
   → CN gap", not the frozen knockout-based Q2 → **not consistent**.
4. `F_TGD*` fits the frozen Q2 and shares the F_close basin set → preferred
   for main text; keep current `F_TGD` as labeled secondary
   (`REPORT BOTH WITH DIFFERENT ROLES`).
5. Raw recovery = F5 primary evidence (fractions secondary).
6. Parameter excess: per-basin median over the **15 shared params** of
   |e_M − e_CN| (plus optional labeled signed variant); per-parameter
   tiers for display; normalized-by-range primary; no new distance family.
7. State excess: `delta_E = E_M − E_CN` (CN-refit baseline); NRMSE headline
   for wu/wl/wt (+corr/bias for fluxes); `wd` secondary; `wt` F6 headline;
   no cross-comparison of tgd_* with snow diagnostics.
8. Historical attenuation confirmed; main Base associations are
   snow-activity-organized (`SNOW_ACTIVITY_COMMON_ORGANIZER_SUPPORTED`);
   weak residual only for incremental TGD-vs-Base gain
   (`RESIDUAL_ASSOCIATION_REMAINS`).
9. **F5 READY AFTER DERIVATION** (§H).
10. **F6 READY** — Option A (no daily-state replay needed for the frozen
    figure; replay only if daily-resolution internal evidence is added).
11. **No retraining/recalibration required.**