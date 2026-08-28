# R3 / Results 3.3 — F6 Parameter Truth-Error Definition (Correction Note)

2026-08-21. Minimal-scope correction of the Figure-6 **shared-parameter
error** definition. No training, calibration, forward replay, or
modification of canonical results. Companion docs:
`r33_data_readiness_audit.md`, `r33_estimand_figure_readiness_audit.md`.

Derived artifacts (new, this directory's `estimand_audit/` subfolder):
- `estimand_audit/derive_param_truth_error.py` (deterministic; validation
  asserts built in)
- `estimand_audit/param_truth_error_basin.csv` (per basin × regime × seed:
  E_param_base/tgd/cn, E_param_excess_base/tgd, C15_base/tgd, G_base,
  G_TGD_ko, frac_snow, snow_stratum)
- `estimand_audit/param_truth_error_summary.json` (all statistics below)

---

## 1. Why the correction

The aggregate previously recommended for F6,

```
C_theta15(M) = median_{p=1..15} |e_{M,p} - e_{CN,p}|,
e_{M,p} = (theta_hat_M,p - theta*_p)/(U_p - L_p)
```

is algebraically identical to

```
median_p |theta_hat_M,p - theta_hat_CN,p| / (U_p - L_p),
```

so the generating truth θ* cancels in the difference. It measures
**shared-parameter separation from CN-refit**, not a truth-relative excess
error. The F6 question "does outlet recovery accompany recovery of the
shared parameters toward the generating truth" requires quantities in which
θ* does not cancel.

## 2. Canonical definitions (frozen parameter scope: the 15 COMMON_XAJ only)

Per basin b × regime × seed (dPL) or frozen selected fit (IC):

```
E_param_M,b      = median_{p=1..15} |e_{M,b,p}|          (M in {Base, TGD, CN})
E_param_excess_M,b = E_param_M,b - E_param_CN,b          (CN = CN-refit;
                   negatives retained, no clipping)
C15_M,b          = median_{p=1..15} |e_{M,b,p} - e_{CN,b,p}|
                   (KEPT: shared-parameter separation from CN-refit;
                    NOT the truth-relative excess)
```

- Pairing unchanged (frozen protocol): IC = per-structure best train-KGE
  restart, no cross-structure restart matching introduced; dPL = seed 42 /
  123 / 2026 matched, quantities computed at seed level, then per-basin
  seed-median aggregation (no ratio-of-medians).
- Truth origin is always `CN-generating` θ* (`theta_star.npz`); `e_cn` is
  computed from CN-refit gate estimates (identical values for Base and
  TGD2 rows in `paired_parameters.csv` — asserted, max diff 0.0).
- Only 15 shared params enter all aggregates (asserted: the parameter set
  of `paired_parameters.csv` equals `COMMON_XAJ` exactly).
- `C_theta_primary` (posthoc tier-based median) keeps its existing role for
  backward compatibility; wherever it is displayed it must be labeled
  **CN-refit parameter separation**, not excess/truth error. Current
  mislabeled spots (3.3 scope only, relabel at figure/table production):
  - `plot_figure5.py` panel (e) title "Parameter excess error vs. snow
    activity" plots `C_theta_base` (F5 data-prep column);
  - `generate_table_r3_main.py` Table R3 row "Delta C_theta
    (R_theta_tgd2)" / "Parameter relief";
  - `posthoc_validation.py` V3 wording "internal-distortion reduction"
    (R_theta_tgd2 = C_theta[Base] − C_theta[TGD2]).
  F6 data-prep (`prepare_figure6_data.py`) reads `R_theta_tgd2` from the
  canonical `posthoc_validation_tgd2_reduction.csv`, not
  `posthoc_theta_cost.csv` directly; it needs no code change this round,
  but its parameter panel source should be re-pointed to the §5 reduction
  quantities when F6 is produced.

## 3. Validation (all performed, all pass)

1. Exactly the 15 COMMON_XAJ parameters enter the aggregates (assert).
2. Independent recomputation: for sampled basins 01022500/01031500/01047000
   × Base/TGD2 (IC + dPL s42) and CN (IC + dPL s42, as e_cn), `e` was
   recomputed from the raw fitted physical parameters (IC raw JSON best
   restart; dPL `best_parameters_physical.npz`) vs θ* with bounds from
   `gstar_manifest.json`; max |csv − recomputed| ≤ 1.9e-16 (assert < 1e-9).
   Example (IC): basin 01022500 E_param_base = 0.3082,
   E_param_cn = 0.0704, E_param_excess_base = +0.2378.
3. New ≠ old: E_param_excess_base == C15_base in only 2/531 IC basins
   (coincidental); Spearman(E_param_excess_base, C15_base) = 0.609 (IC) —
   related but distinct quantities.
4. dPL pairing by seed verified (e_cn identical across Base/TGD2 rows;
   per-seed rows preserved; seed-median aggregation only at summary level).
5. IC: no restart matching introduced (independent best-restart selections
   per structure, exactly as before).
6. Negatives retained: IC E_param_excess_base < 0 in 118/531 basins (min
   −0.2665); dPL 3.6% (Base) / 15.4% (TGD) negative; no clipping anywhere.
7. All summaries derived basin-wise; canonical result files untouched
   (write-only outputs in `estimand_audit/`).

## 4. Re-derived statistics (test period)

Population (IC n=531; dPL = per-basin seed median n=531; medians with
paired-basin bootstrap 95% CI, 2000 reps, seed 20260730):

| Quantity | IC median [CI] (IQR) | dPL seed-median [CI] (IQR) | dPL per seed |
|---|---|---|---|
| E_param_base | 0.180 [0.170, 0.196] (0.094–0.288) | 0.094 [0.071, 0.112] (0.032–0.235) | 0.083 / 0.096 / 0.093 |
| E_param_tgd | 0.163 [0.149, 0.179] (0.083–0.263) | 0.055 [0.049, 0.064] (0.020–0.144) | 0.050 / 0.057 / 0.058 |
| E_param_cn | 0.070 [0.065, 0.077] (0.047–0.115) | 0.026 [0.023, 0.029] (0.014–0.045) | 0.026 / 0.025 / 0.025 |
| E_param_excess_base | +0.097 [0.083, 0.113] (0.013–0.200); lt0 22.2% | +0.064 [0.048, 0.079] (0.014–0.187); lt0 3.6% | +0.057 / +0.066 / +0.064 |
| E_param_excess_tgd | +0.077 [0.062, 0.093] (0.003–0.180); lt0 23.7% | +0.024 [0.019, 0.033] (0.004–0.098); lt0 15.4% | +0.020 / +0.028 / +0.027 |
| C15_base (secondary, kept) | 0.194 [0.179, 0.208] | 0.061 [0.054, 0.074] | — |
| C15_tgd (secondary, kept) | 0.178 [0.166, 0.193] | 0.055 [0.046, 0.066] | — |

Snow-activity organization (medians per stratum; N per stratum 165/156/121/
34/55):

| Stratum | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| E_param_base IC | 0.073 | 0.157 | 0.256 | 0.294 | 0.406 |
| E_param_excess_base IC | +0.003 | +0.083 | +0.170 | +0.234 | +0.305 |
| E_param_excess_tgd IC | +0.006 | +0.056 | +0.151 | +0.203 | +0.265 |
| E_param_cn IC | 0.061 | 0.070 | 0.072 | 0.082 | 0.093 |
| E_param_base dPL | 0.023 | 0.068 | 0.224 | 0.277 | 0.341 |
| E_param_excess_base dPL | +0.009 | +0.044 | +0.177 | +0.236 | +0.252 |
| E_param_excess_tgd dPL | +0.003 | +0.017 | +0.091 | +0.187 | +0.202 |
| E_param_cn dPL | 0.014 | 0.023 | 0.045 | 0.038 | 0.087 |

Continuous frac_snow Spearman (descriptive): E_param_base 0.801 (IC) /
0.939 (dPL); E_param_excess_base 0.739 / 0.917; E_param_excess_tgd 0.621 /
0.826; E_param_cn 0.096 / 0.74 (computed, see JSON). Both E_param and the
excess rise systematically S1→S5; CN-refit's own truth error rises mildly
in the stratum medians (e.g. E_param_cn IC 0.061→0.093) but is an order
of magnitude smaller than M's distance to truth at high snow, so the
excess gradient is dominated by M's distance to truth.

## 5. Recovery–parameter association (raw outlet recovery, test)

Raw Spearman → partial Spearman controlling continuous frac_snow
(method identical to `posthoc_validation.py`):

| Pair | IC | dPL s42 | dPL s123 | dPL s2026 | dPL seed-median |
|---|---|---|---|---|---|
| G_base ↔ E_param_excess_base | 0.601 → **+0.023** | 0.780 → +0.042 | 0.806 → +0.116 | 0.803 → +0.144 | 0.812 → +0.138 |
| G_TGD_ko ↔ E_param_excess_tgd | 0.522 → **−0.067** | 0.633 → −0.156 | 0.677 → −0.118 | 0.684 → −0.129 | 0.685 → −0.128 |
| G_base ↔ C15_base (secondary separation) | 0.561 → −0.021 | 0.824 → +0.177 | 0.833 → +0.200 | 0.832 → +0.219 | 0.845 → +0.256 |
| G_TGD_ko ↔ C15_tgd (secondary separation) | 0.433 → −0.091 | — | — | — | 0.824 → +0.117 |

S1–S5 within-stratum raw Spearman (G_base ↔ E_param_excess_base): IC
−0.02/+0.02/+0.18/+0.28/−0.07; dPL seed-median +0.39/+0.23/+0.40/+0.20/
+0.15. G_TGD_ko ↔ E_param_excess_tgd within strata is sign-unstable
(e.g. S4 −0.35 IC, −0.39 dPL). Descriptive only (S4 n=34, S5 n=55).

**Verdict (based solely on the new truth-relative excess):**
`SNOW_ACTIVITY_COMMON_ORGANIZER_SUPPORTED` — the raw recovery↔excess
associations (0.52–0.81) attenuate strongly after controlling frac_snow
(IC partials ≈ 0; dPL G_base partial +0.04..+0.14, G_TGD_ko partial −0.13..
−0.16). A weak positive residual remains for the dPL **Base** pair
(partial ≈ +0.1, consistent with the earlier C_theta-based V1 finding) but
it is an order of magnitude below the raw association; no residual positive
association exists for the TGD pair. This matches, and does not overturn,
the historical direction; it is now expressed on quantities that are
genuinely truth-relative.

## 6. F6 parameter-side readiness

The F6 parameter panel must distinguish three quantities (do not merge):

1. **distance to generating truth** — E_param_M (Base 0.180/0.094,
   TGD 0.163/0.055, CN 0.070/0.026, IC/dPL medians);
2. **excess truth-relative error beyond correct-CN refit** —
   E_param_excess_M (Base +0.097/+0.064, TGD +0.077/+0.024);
3. **optional secondary: shared-parameter separation from CN-refit** —
   C15_M (old C_theta15 semantics, relabeled).

Reduction quantities for the F6 "parameter relief" panel (replacing the
C_theta-based R_theta_tgd2 when F6 is produced; derivation-only):
- R_E_param = E_param_base − E_param_tgd: IC 0.0170, dPL 0.0389
  (seed-median; per seed 0.033/0.040/0.035);
- R_E_excess = E_param_excess_base − E_param_excess_tgd: IC 0.0209,
  dPL 0.0398 (per seed 0.037/0.038/0.037).
(Old C_theta-based R_theta_tgd2 was IC 0.0047 / dPL 0.007–0.011 —
one-to-three smaller; label and interpretation change accordingly.)

**`F6 PARAMETER SIDE READY — no retraining, recalibration, or state replay
required.`** Remaining work is derivation at figure-production time only:
re-point the F6 parameter panel source to E_param / E_param_excess (and
relabel C_theta-based displays as CN-refit separation in F5 panel e / Table
R3 / SI), and extend `prepare_figure6_data.py` with the new fields (backward
compatible: old columns remain).