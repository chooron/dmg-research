# Hostile Final Definition Alignment Review (H1–H4)

**Reviewer Role:** Adversarial senior peer reviewer evaluating the technical consistency, representation fidelity, and IC-self comparison validity of the R4 out-of-bag (OOB) challenge prior to final manuscript freeze.

---

## H1. Did R4 truly execute an estimand-matched replay of frozen R3?

### Verdict: **PASS WITH LIMITATION (PRESPECIFIED ALTERNATIVE REPRESENTATION CONFIRMED)**

### Reviewer Assessment:
- **The Issue:** Frozen R3 established its primary headline on a 20-dimensional information space (PC1 scores of 20 clusters derived from all 35 CAMELS attributes, including 3 categorical variables). In contrast, R4 OOB evaluation operates on 13 continuous information clusters and 32 continuous attributes.
- **Audit Findings:**
  1. OOB parameter correlation pipelines strictly process continuous variables without arbitrary integer encoding of discrete categoricals (`dom_land_cover`, `geol_1st_class`, `geol_2nd_class`).
  2. Applying the exact same $|\rho| \ge 0.70$ redundancy grouping to the 32 continuous attributes yields 13 orthogonal continuous clusters.
  3. Replaying profile correspondence across both representations yields virtually identical results ($R_{\text{paired}} = 0.7390$ on 13 clusters, $0.7620$ on 32 continuous attributes, compared to $0.7158$ on the 20-D seen benchmark).
- **Mandated Manuscript Phrasing:** The manuscript must describe the R4 replay as an evaluation across the *prespecified 13-cluster representative and 32-continuous attribute representations*, rather than claiming an identical 20-D PC1 score replay.

---

## H2. Can `R_paired,OOB` and `A_diag,OOB` be legitimately compared with the Full36 frozen 20-D numbers?

### Verdict: **PASS WITH LIMITATION**

### Reviewer Assessment:
- **The Issue:** If the feature count differs (13 or 32 vs. 20), is cross-paradigm profile correlation comparable?
- **Audit Findings:**
  1. Profile correspondence ($R_{\text{paired}}$) measures the Spearman correlation between two 13-D or 32-D vectors of parameter–catchment associations.
  2. On the same 8 tested models, seen-basin correspondence is $0.7129$ (13 clusters), $0.7447$ (32 attributes), and $0.7496$ (20-D clusters).
  3. Out-of-bag correspondence is $0.7390$ (13 clusters) and $0.7620$ (32 attributes).
  4. The diagonal advantage ($A_{\text{diag}}$) remains highly significant ($A_{\text{diag}} = 0.7184, p = 0.000999$).
- **Mandated Manuscript Phrasing:** The comparison is mathematically and scientifically sound, provided the manuscript explicitly presents the 13-cluster and 32-attribute spaces as continuous cluster-level representations.

---

## H3. Did R4 truly verify the frozen R2 "beyond IC-self" headline, or only raw parameter separation?

### Verdict: **PASS (CONFIRMED IN 5 / 5 ELIGIBLE TESTED MODELS)**

### Reviewer Assessment:
- **The Issue:** Frozen R2 Headline 1 claimed parameter displacement exceeded multi-start calibration dispersion ($D_{\text{cross}} - D_{\text{self}} > 0$). Did R4 verify this on OOB outputs, or merely show $D_{\text{RMS, OOB}} > 0$?
- **Audit Findings:**
  1. Under the strict frozen R2 contract ($\ge 90\%$ multi-start coverage within $\Delta\text{KGE} \le 0.01$), exactly **5 of the 8 tested models** are strictly eligible (`alpine2`, `hillslope`, `ihacres`, `us1`, `xinanjiang`).
  2. For all 5 eligible models, OOB parameter separation strictly exceeds the archived IC multi-start dispersion:
     - Median $D_{\text{cross, OOB}} = 0.3293$ vs. Median $D_{\text{self}} = 0.0326$
     - Net excess separation: **$D_{\text{cross, OOB}} - D_{\text{self}} = \mathbf{+0.2270}$**
     - Exceedance is positive in **5 / 5 (100%)** of eligible models.
  3. The remaining 3 models (`hbv96`, `mopex4`, `newzealand2`) were already classified as `INSUFFICIENT_REFERENCE` in frozen R2.
- **Mandated Manuscript Phrasing:** The manuscript can legitimately claim that OOB parameter separation exceeds calibration restart dispersion *in the eligible tested models*, while scoping the 3 insufficient models.

---

## H4. Does correcting these technical definitions alter the R1–R3 evidence chain or require a rerun?

### Verdict: **PASS (CORE EVIDENCE CHAIN FULLY UPHOLDED; NO RERUN REQUIRED)**

### Reviewer Assessment:
1. **R1 (Outlet Performance):** `PARTIALLY SUPPORTED` (regionalization gap of $\approx 0.057$ KGE points, with 67.1% basin-additive variance).
2. **R2 (Parameter Realization):** `PARTIALLY SUPPORTED` (core separation confirmed, excess displacement confirmed in 5/5 eligible models, performance bridge positive in 8/8 models, geometry sub-claims un-tested).
3. **R3 (Catchment Information):** `STRONGLY SUPPORTED` (profile correspondence $0.7390$ and same-coordinate specificity $A_{\text{diag}} = 0.7184, p = 0.000999$ confirmed on held-out basins).
4. **No Validity Blocker:** The 40 OOB runs are completely valid, free of leakage, and scientifically sound. No re-training or re-calibrations are warranted.

---

## Final Review Verdict

**`PASS WITH LIMITATION (EXPLICIT SCOPE BOUNDARIES ADOPTED — READY FOR FREEZE)`**
