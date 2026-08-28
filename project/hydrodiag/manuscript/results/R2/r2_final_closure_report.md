# Results 3.2 (R2) Final Completeness & Statistical Validity Closure Report

- **Final Status:** **`R2_FINAL_STATUS = READY`**
- **Closure Verdict:** Current R2 statistics are complete, internally consistent, and methodologically adequate for Figure 3/4 finalization, Results 3.2 drafting, and Discussion 4.2 evidence alignment. No additional R2 statistical analysis is required.
- **Domination Verdict:** **`ROBUST_MULTIVARIATE`** (Whole-space reorganization is distributed across the 15-parameter space and does not collapse when any single parameter is removed).

## 1. Data Integrity and Provenance Verdict

- **Lowest-level raw parameters:** Verified from 15,930 IC raw JSONs (531 basins × 3 structures × 10 starts) and 9 dPL parameter arrays (531 basins × 3 structures × 3 seeds).
- **15 Shared Parameters:** Verified identities, order, and physical bounds across Base, CN, and TGD; extra structure-specific parameters (cn_ctg, cn_kf, tgd_tau_warm, tgd_delta_tau_cold) strictly isolated.
- **Normalized Coordinates:** $z = (\theta - \text{lower})/(\text{upper} - \text{lower})$ verified across all 310,635 ledger rows.
- **Subset Consistency:** Full531 ($N=531$) and ExcludeS5 ($N=476$) exactly match frozen R1 manifest.

## 2. Canonical Prevalence Definition and 4-Basin Calculation Trace

- **Manuscript-Facing Prevalence Formula:** $\text{Prevalence} = P_b(\text{between\_all}_b > \text{within\_pooled}_b)$ where $b$ indexes individual basins.
- **Canonical Values:** IC Full531 = **63.09%** (335/531) [59.13%, 67.04%]; dPL Full531 = **83.80%** (445/531) [80.60%, 86.82%].
- **Legacy Explanation:** The draft ~97.36% (IC) / 100% (dPL) occurred because a draft script substituted the basin-specific `within_pooled` with a fixed scalar threshold `0.08` (`between_all > 0.08`), which is non-canonical. The canonical formulation strictly evaluates `between_all > within_pooled` per basin.

### Step-by-Step Calculation Trace for Sample Basins

| Paradigm | Stratum | Basin ID | $f_{\text{snow}}$ | within_Base | within_CN | within_pooled | between_all | Excess ($b_{\text{all}} - w_{\text{pool}}$) | Outcome ($b > w$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IC | **S1** | `01411300` | 0.0481 | 0.4599 | 0.4813 | **0.4706** | **0.4605** | **-0.0101** | **False** |
| dPL | **S1** | `01411300` | 0.0481 | 0.0579 | 0.0720 | **0.0649** | **0.2300** | **+0.1651** | **True** |
| IC | **S5** | `06221400` | 0.7148 | 0.4966 | 0.4074 | **0.4520** | **0.4927** | **+0.0407** | **True** |
| dPL | **S5** | `06221400` | 0.7148 | 0.0515 | 0.0991 | **0.0753** | **0.0726** | **-0.0027** | **False** |

## 3. Final S1–S5 Base–CN & Base–TGD Trajectory

| Paradigm | Stratum | n | Base–CN Excess [95% CI] | Base–CN Prev. | Base–TGD Excess [95% CI] | Base–TGD Prev. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| IC | **S1** | 165 | **-0.0018** [-0.0044, +0.0015] | 46.7% | **+0.0006** [-0.0021, +0.0040] | 51.5% |
| IC | **S2** | 156 | **+0.0022** [-0.0034, +0.0041] | 52.6% | **+0.0022** [-0.0001, +0.0055] | 57.1% |
| IC | **S3** | 121 | **+0.0123** [+0.0090, +0.0185] | 74.4% | **+0.0133** [+0.0080, +0.0163] | 71.1% |
| IC | **S4** | 34 | **+0.0430** [+0.0324, +0.0649] | 94.1% | **+0.0308** [+0.0106, +0.0460] | 76.5% |
| IC | **S5** | 55 | **+0.0829** [+0.0702, +0.1104] | 98.2% | **+0.0937** [+0.0743, +0.1153] | 94.5% |
| dPL | **S1** | 165 | **+0.0186** [+0.0074, +0.0258] | 70.3% | **+0.0195** [+0.0104, +0.0270] | 70.9% |
| dPL | **S2** | 156 | **+0.0557** [+0.0433, +0.0793] | 87.2% | **+0.0310** [+0.0234, +0.0406] | 75.6% |
| dPL | **S3** | 121 | **+0.1267** [+0.1006, +0.1702] | 92.6% | **+0.0791** [+0.0547, +0.1378] | 88.4% |
| dPL | **S4** | 34 | **+0.1322** [+0.0926, +0.2667] | 97.1% | **+0.1070** [+0.0681, +0.1537] | 91.2% |
| dPL | **S5** | 55 | **+0.1252** [+0.0826, +0.2189] | 87.3% | **+0.0851** [+0.0546, +0.1173] | 87.3% |

## 4. Basin-Paired CN–TGD Macro Contrast: $\Delta_{\text{excess}} = \text{excess}(\text{Base-CN}) - \text{excess}(\text{Base-TGD})$

| Paradigm | Stratum | n | Median $\Delta_{\text{excess}}$ [95% CI] | IQR | $P(\Delta_{\text{excess}} > 0)$ [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: |
| IC | **Full531** | 531 | **-0.0014** [-0.0034, +0.0011] | 0.0341 | **48.2%** [43.9%, 52.6%] |
| IC | **ExcludeS5** | 476 | **-0.0014** [-0.0033, +0.0011] | 0.0325 | **47.9%** [43.2%, 52.5%] |
| IC | **S1** | 165 | **-0.0046** [-0.0091, +0.0005] | 0.0320 | **44.2%** [36.4%, 51.5%] |
| IC | **S2** | 156 | **-0.0009** [-0.0031, +0.0030] | 0.0312 | **48.7%** [40.4%, 55.8%] |
| IC | **S3** | 121 | **-0.0013** [-0.0046, +0.0020] | 0.0270 | **46.3%** [36.8%, 55.0%] |
| IC | **S4** | 34 | **+0.0155** [+0.0012, +0.0205] | 0.0578 | **67.6%** [52.9%, 82.4%] |
| IC | **S5** | 55 | **+0.0018** [-0.0243, +0.0219] | 0.0961 | **50.9%** [38.2%, 63.6%] |
| dPL | **Full531** | 531 | **+0.0177** [+0.0123, +0.0236] | 0.0786 | **64.6%** [60.5%, 68.7%] |
| dPL | **ExcludeS5** | 476 | **+0.0160** [+0.0099, +0.0222] | 0.0759 | **63.9%** [60.0%, 67.9%] |
| dPL | **S1** | 165 | **+0.0047** [-0.0004, +0.0091] | 0.0488 | **56.4%** [49.1%, 64.2%] |
| dPL | **S2** | 156 | **+0.0287** [+0.0169, +0.0368] | 0.0731 | **69.2%** [62.2%, 76.3%] |
| dPL | **S3** | 121 | **+0.0239** [+0.0151, +0.0334] | 0.0956 | **65.3%** [57.0%, 73.6%] |
| dPL | **S4** | 34 | **+0.0363** [+0.0072, +0.1029] | 0.1353 | **70.6%** [55.9%, 85.3%] |
| dPL | **S5** | 55 | **+0.0317** [+0.0065, +0.0490] | 0.0808 | **70.9%** [58.2%, 83.6%] |

## 5. Whole-Space One-Parameter-Domination Robustness (14-D LOPO Sensitivity)

- **IC 14-D Slope Range across all 15 exclusions:** **$[+0.1470, +0.1651]$** (Baseline 15-D $= +0.1542$)
- **dPL 14-D Slope Range across all 15 exclusions:** **$[+0.1651, +0.2084]$** (Baseline 15-D $= +0.1974$)
- **Distance Contribution Shares (Option B):** Highest single-parameter mean share is 13.50% (`xaj_c`) in IC and 13.76% (`xaj_cg`) in dPL. No single parameter dominates the multivariate distance.
- **Domination Verdict:** **`ROBUST_MULTIVARIATE`** — The whole-space macro response is strictly distributed across the parameter space and does not collapse when any single parameter is removed.

## 6. Wording Verdicts & Discussion Evidence Mapping

### A. Wording Verdicts
- **IC-CMA-ES:** **`MONOTONIC / NEAR-MONOTONIC ORGANIZATION`** — *"parameter-space reorganization became progressively stronger with snow activity"*
- **dPL-MLP:** **`ORDERED BUT NONLINEAR`** — *"parameter-space reorganization was increasingly organized across the snow-activity gradient, steep across moderate snow regimes (S2-S4) and plateauing in high-snow basins (S5)"*

### B. Final Claim Wording Audit (6 Core Claims)
1. `Structural omission was associated with systematic reorganization of the calibrated shared parameter space.` -> **KEEP** (Supported by whole-space macro excess and prevalence across IC and dPL).
2. `IC: Parameter-space separation became progressively stronger with increasing snow activity.` -> **KEEP** (Supported by strictly monotonic S1->S5 excess progression).
3. `dPL: Parameter-space separation strengthened from low to moderate/high snow activity and plateaued at the highest snow activity.` -> **KEEP** (Supported by steep rise in S1-S4 and saturation in S5).
4. `TGD: The specified temperature-conditioned generic control reproduced part of the macro parameter-space response.` -> **KEEP** (Supported by TGD excess slopes +0.154 in IC and +0.156 in dPL).
5. `dPL TGD qualification: Additional Base–CN separation relative to TGD was already evident across intermediate snow-activity strata and persisted into higher-snow conditions.` -> **KEEP** (Supported by positive delta_excess in S2..S5 and ExcludeS5 Delta_beta = +0.086).
6. `Constraint regime: The same structural perturbation was expressed differently under basin-wise independent calibration and shared cross-basin parameter learning.` -> **KEEP** (Reflects observational constraint difference without ranking).

### C. Prohibited Phrases vs Recommended Replacements
- Avoid: `IC is unconstrained` -> Use: `basin-wise independent calibration`
- Avoid: `dPL regularization causes ...` -> Use: `shared cross-basin parameter mapping`
- Avoid: `CN-TGD proves snow-specific contribution` -> Use: `additional separation relative to the specified TGD control`
- Avoid: `um/ki/ci directly compensate snow storage/melt` -> Use: `recurring directional parameter signatures`
- Avoid: `R2 quantifies structural deficit recovery` -> (Reserved exclusively for R3 synthetic truth).

## 7. Closure Decision

**`R2_FINAL_STATUS = READY`**

All data, models, estimands, and boundaries for Section 3.2 are complete and formally frozen. No additional R2 statistical analysis is required. Proceed directly to Figure 3/4 finalization and Results 3.2 drafting.