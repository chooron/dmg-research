# Canonical R1 Analysis Results Summary

- **Status:** COMPLETED
- **Canonical Promotion Gates:** PASS
- **Evaluation Dataset:** 531 basins × 3 structures × 2 regimes = 3186 rows (test period)
- **Resampling / Bootstrap:** 10,000 paired basin draws (Seed `20260730`)
- **Execution Time:** 3.57 s (Peak VRAM: 10.0 MB, Peak RAM: 1269.7 MB)

## 1. Canonical Gates Status

| Gate | Status | Description |
| :--- | :---: | :--- |
| Provenance Gate | **PASS** | Pinned digests and exact schemas verified for all staged sources. |
| Basin Alignment Gate | **PASS** | 531 paired basins for each paradigm; 0 silent drops, 0 duplicate keys. |
| CT Definition Gate | **PASS** | Delta_CT = CT_sim - CT_obs; basin CT = median valid years; absolute_CT = abs(signed). |
| Statistical Unit Gate | **PASS** | Inferential unit is basin (N=531). Seeds/restarts are aggregated prior to inference. |
| Reproducibility Gate | **PASS** | All outputs reproducible from verified staged tables without daily raw files. |

## 2. Primary Base-CN Snow-Activity Estimands

Positive values denote improvement in CN relative to Base.

### A. S5-S1 Endpoint Activity Contrast ($D_{\text{activity}} = \text{median}(S5) - \text{median}(S1)$)

| Paradigm | N(S1) | N(S5) | Median S1 (d) | Median S5 (d) | $D_{\text{activity}}$ (d) | 95% Bootstrap CI (d) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| IC-CMA-ES | 165 | 55 | 0.0 | 47.0 | **47.0** | [44.0, 51.0] |
| dPL-MLP | 165 | 55 | 0.0 | 46.0 | **46.0** | [40.0, 48.0] |

### B. Continuous Spearman Association with Snow Fraction (`frac_snow` vs `delta_absCT_Base_CN`)

| Paradigm | N | Spearman $\rho$ | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: |
| IC-CMA-ES | 531 | **0.546** | [0.458, 0.616] |
| dPL-MLP | 531 | **0.459** | [0.372, 0.550] |

## 3. Secondary TGD Structural Control

| Paradigm | Metric | Stratum | Median (d) | 95% Bootstrap CI (d) | Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| IC-CMA-ES | delta_absCT_TGD_CN | overall | 1.0 | [0.0, 1.0] | Secondary output-level control |
| dPL-MLP | delta_absCT_TGD_CN | overall | 0.0 | [0.0, 0.0] | Secondary output-level control |

## 4. KGE-Qualified Timing Inconsistency Prevalence Audit

Audits the prevalence of timing inconsistency ($|CT| \ge 15\text{ d}$) among basins with acceptable hydrograph fit ($KGE \ge 0.60$).

- **Conditional Prevalence:** $P(|CT| \ge 15\text{ d} \mid KGE \ge 0.60) = \frac{N(KGE \ge 0.60 \cap |CT| \ge 15\text{ d})}{N(KGE \ge 0.60)}$
- **Joint Prevalence:** $P(KGE \ge 0.60 \cap |CT| \ge 15\text{ d}) = \frac{N(KGE \ge 0.60 \cap |CT| \ge 15\text{ d})}{531}$

### A. Structure-Specific Denominator ($N_s(KGE_s \ge 0.60)$)

| Paradigm | Structure | Numerator | Conditional Denom | Conditional Prevalence | 95% Bootstrap CI | Joint Prevalence ($N=531$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| IC-CMA-ES | Base | 56 | 331 | **16.92%** | [13.58%, 21.52%] | 10.55% (56/531) |
| IC-CMA-ES | TGD | 46 | 397 | **11.59%** | [8.62%, 14.80%] | 8.66% (46/531) |
| IC-CMA-ES | CN | 25 | 427 | **5.85%** | [3.88%, 8.30%] | 4.71% (25/531) |
| dPL-MLP | Base | 46 | 344 | **13.37%** | [9.54%, 16.43%] | 8.66% (46/531) |
| dPL-MLP | TGD | 39 | 405 | **9.63%** | [7.12%, 12.28%] | 7.34% (39/531) |
| dPL-MLP | CN | 20 | 426 | **4.69%** | [2.83%, 6.70%] | 3.77% (20/531) |

### B. Common-Pass Denominator (Same-Basin $KGE \ge 0.60$ Across Base, TGD, CN)

| Paradigm | Structure | Numerator | Common Denom | Conditional Prevalence | 95% Bootstrap CI | Joint Prevalence ($N=531$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| IC-CMA-ES | Base | 55 | 321 | **17.13%** | [13.78%, 21.64%] | 10.36% (55/531) |
| IC-CMA-ES | TGD | 35 | 321 | **10.90%** | [7.97%, 14.48%] | 6.59% (35/531) |
| IC-CMA-ES | CN | 18 | 321 | **5.61%** | [3.44%, 8.15%] | 3.39% (18/531) |
| dPL-MLP | Base | 44 | 331 | **13.29%** | [9.59%, 16.27%] | 8.29% (44/531) |
| dPL-MLP | TGD | 27 | 331 | **8.16%** | [5.40%, 10.67%] | 5.08% (27/531) |
| dPL-MLP | CN | 14 | 331 | **4.23%** | [1.93%, 6.57%] | 2.64% (14/531) |

## 5. Robustness & Sensitivity Summary

- **dPL Seed Robustness (Seeds 42, 123, 2026):** PASS (all seeds show positive D_activity, positive Spearman rho, and monotonic S1-S5 increase).
- **IC Restart Stability:** Uses canonical `selected_restart` determined from training-period KGE.
- **Regional LORO Robustness:** Status `not_executed` (authoritative group_11..group_17 metadata unavailable in repository).

## 6. Artifact Manifest

- `canonical_basin_level.csv`: 3,186 canonical basin-level evaluation rows.
- `canonical_paired_contrasts.csv`: 1,062 paired basin contrast rows.
- `snow_stratified_summaries.csv`: S1-S5 and overall stratified distributions with 95% CIs.
- `spearman_associations.csv`: Continuous rank correlations and 95% CIs.
- `endpoint_activity_contrast.csv`: S5-S1 endpoint activity contrasts and 95% CIs.
- `secondary_tgd_control_summaries.csv`: Secondary TGD structural control summaries.
- `threshold_denominator_audit.csv`: Full grid threshold prevalence across denominator types.
- `threshold_prevalence_summary.csv`: Key cutoffs (KGE 0.40..0.80, CT 10, 15, 20 d) with bootstrap CIs.
- `seed_restart_robustness.csv`: Per-seed dPL evaluations and IC stability records.
- `canonical_gates_summary.json`: Formal validation records of all 5 gates.
- `machine_readable_summary.json`: Complete machine-readable summary.