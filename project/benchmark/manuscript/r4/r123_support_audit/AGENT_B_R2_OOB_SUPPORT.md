# Agent B — R2 Held-Out Replay and Support Audit

## 1. Executive Summary & Verdict

### Final Verdict: **`R2 PARTIALLY SUPPORTED (CORE PARAMETER SEPARATION & PERFORMANCE BRIDGE CONFIRMED; RANK REORGANIZATION & LOCALIZATION NOT TESTED BY R4)`**

- **Frozen R2 Core Claims:**
  1. *Parameter Separation:* Shared attribute-constrained parameter learning (dPL) forces conceptual parameter vectors into a distinct realization regime separated from independent calibration (IC) far beyond multi-start restart uncertainty ($D_{\text{RMS}} \approx 0.384$).
  2. *Rank Reorganization & Localization:* Parameter values across catchments undergo rank reorganization, with coordinate-specific localization.
  3. *Performance–Parameter Bridge:* The magnitude of parameter displacement correlates with the absolute performance difference between IC and dPL ($\rho_b(|\Delta\text{KGE}|, D_{\theta}) \approx +0.241$, positive in 34/36 models).
- **R4 OOB Replay Findings:**
  1. *Parameter Separation Replayed:* Under 5-fold held-out evaluation, normalized parameter displacement between IC and OOB-dPL remains large, positive, and structurally stable: model-equal median **$D_{\text{RMS, OOB}} = 0.342886$** (mean $0.332670$, range $[0.2098, 0.4517]$), closely matching seen-basin displacement ($D_{\text{RMS, seen}} = 0.313853$).
  2. *Seen-to-OOB Parameter Shift is Minimal:* The median distance between seen-dPL and OOB-dPL parameter vectors across basins is only **$0.100356$**, demonstrating that uncalibrated regional dPL occupies the same separated parameter regime as seen dPL.
  3. *Performance Bridge Confirmed:* Spearman correlation $\rho_b(|\Delta\text{KGE}_{\text{OOB}}|, D_{\theta, \text{OOB}})$ is **$+0.242456$** (model-equal median) and is strictly positive across **8/8 models** (range $[+0.0436, +0.4412]$), replicating the seen-basin bridge ($\rho = +0.210350$).
  4. *Un-Replayed Estimands Explicitly Scoped:* Cross-catchment rank reorganization ($R_{\text{rank}}$) and coordinate localization ($C_{\text{eff}}$) were not prespecified in the R4 protocol and are explicitly classified as `NOT TESTED BY R4`.

---

## 2. Prespecification Audit Table for R2 Estimands

| R2 Finding / Estimand | Was Prespecified in R4 Protocol? | OOB Data Sufficient? | Replay Status in R4 | Audit Verdict |
|---|---|---|---|---|
| **R2.1 Parameter Separation ($D_{\text{RMS}}$)** | **YES** (`oob_common.py`, `r4_compute_statistics.py`) | YES (531 basins × 8 models) | Replayed directly | **SUPPORTED** |
| **R2.2 Rank Reorganization ($R_{\text{rank}}$)** | **NO** | YES (requires un-prespecified code) | Not prespecified | **NOT TESTED BY R4** |
| **R2.3 Coordinate Localization ($C_{\text{eff}}$)** | **NO** | YES (requires un-prespecified code) | Not prespecified | **NOT TESTED BY R4** |
| **R2.4 Contraction Boundary** | **NO** | Partial | Canonical-IC dependent | **NOT TESTED BY R4** |
| **R2.5 Performance Bridge ($\rho(|\Delta\text{KGE}|, D_{\theta})$)** | **YES** (part of performance/parameter master) | YES (531 basins × 8 models) | Replayed directly | **SUPPORTED** |

---

## 3. Quantitative Replay Results for 8 Tested Models

### Table B1: Per-Model Parameter Displacement and Performance Bridge

| Model | Free Parameters ($P$) | $D_{\text{RMS, seen}}$ (Median) | $D_{\text{RMS, OOB}}$ (Median) | $D_{\text{RMS}}$ Shift ($\text{OOB}-\text{Seen}$) | $D_{\text{RMS, OOB}}$ IQR ($[Q25, Q75]$) | $\rho_b(\text{Seen})$ Bridge | $\rho_b(\text{OOB})$ Bridge |
|---|---:|---:|---:|---:|---|---:|---:|
| **alpine2** | 6 | 0.185413 | 0.209835 | +0.024422 | $[0.1347, 0.3298]$ | +0.204087 | +0.281021 |
| **hbv96** | 15 | 0.353147 | 0.361400 | +0.008253 | $[0.2897, 0.4355]$ | +0.216612 | +0.244343 |
| **hillslope** | 7 | 0.302113 | 0.329326 | +0.027213 | $[0.2034, 0.4384]$ | +0.384391 | +0.441178 |
| **ihacres** | 6 | 0.266324 | 0.343291 | +0.076967 | $[0.2137, 0.4207]$ | +0.163666 | +0.216666 |
| **mopex4** | 10 | 0.446162 | 0.451681 | +0.005519 | $[0.3794, 0.5130]$ | +0.113480 | +0.043640 |
| **newzealand2** | 8 | 0.386839 | 0.396316 | +0.009477 | $[0.3004, 0.4713]$ | +0.132122 | +0.060090 |
| **us1** | 5 | 0.194392 | 0.227033 | +0.032641 | $[0.1338, 0.3434]$ | +0.390985 | +0.300704 |
| **xinanjiang** | 12 | 0.325592 | 0.342482 | +0.016890 | $[0.2634, 0.4234]$ | +0.222574 | +0.240569 |

---

### Table B2: Ensemble Summary Comparison (8-Model Equal Weight)

| Metric | Seen-Basin Value (8 Models) | OOB Held-Out Value (8 Models) | Paired Difference | Full 36-Model Seen Reference |
|---|---|---|---|---|
| **$D_{\text{RMS}}$ Median** | 0.313853 | 0.342886 | +0.029033 | 0.384375 |
| **$D_{\text{RMS}}$ Mean** | 0.307498 | 0.332670 | +0.025172 | 0.375100 |
| **$D_{\text{RMS}}$ Range** | $[0.185413, 0.446162]$ | $[0.209835, 0.451681]$ | Shifted slightly higher | $[0.150600, 0.592700]$ |
| **$D(\text{Seen-dPL}, \text{OOB-dPL})$ Median** | N/A | 0.100356 | Minimal shift | N/A |
| **Performance Bridge $\rho_b$ Median** | +0.210350 | +0.242456 | +0.032106 | +0.241190 |
| **Positive Bridge Models** | 8 / 8 (100%) | 8 / 8 (100%) | 0 change | 34 / 36 (94.4%) |

---

## 4. Detailed Evaluation of R2 Sub-Claims

### R2.1 Parameter Realization Separation: **`SUPPORTED`**
- In all eight tested models, the parameter vectors produced by out-of-bag dPL are markedly separated from independent calibrations ($D_{\text{RMS, OOB}} \in [0.210, 0.452]$).
- This separation is not an artifact of seen-basin fitting: regionalized parameterizations transferred to held-out catchments retain the exact same separation regime (median shift between seen and OOB dPL parameters is only $0.100$).

### R2.2 Cross-Catchment Rank Reorganization: **`NOT TESTED BY R4`**
- The R4 held-out experiment was designed as a targeted challenge of parameter displacement and information organization. Cross-catchment parameter ranking benchmarks ($R_{\text{rank}}$) were not prespecified for OOB re-computation.

### R2.3 Coordinate Localization: **`NOT TESTED BY R4`**
- Coordinate localization metrics ($C_{\text{eff}}$, top-1, top-2 coordinate concentration) were not prespecified for R4 replay.

### R2.4 Performance–Parameter Bridge: **`SUPPORTED`**
- Catchments with larger performance differences between IC and dPL also exhibit larger parameter displacement under out-of-bag evaluation (median $\rho = +0.2425$, positive in 8/8 models).

---

## 5. R2 Support Matrix

```text
R2_SUPPORT_MATRIX:
- R2.1 parameter separation:        SUPPORTED
- R2.2 rank reorganization:         NOT TESTED BY R4
- R2.3 coordinate localization:     NOT TESTED BY R4
- R2.4 contraction boundary:        NOT TESTED BY R4
- R2.5 performance bridge:          SUPPORTED
-------------------------------------------------------------
- R2 OVERALL:                       PARTIALLY SUPPORTED (CORE SEPARATION & BRIDGE CONFIRMED; GEOMETRY SUB-CLAIMS NOT RETESTED)
```

---

## 6. Manuscript-Ready Statements for R2

- **Methods/Results Statement:**
  > *"Out-of-bag evaluation confirms the core parameter separation discovered in R2: bounds-normalized parameter displacement between IC and regional dPL remains substantial across all eight models (median $D_{\text{RMS, OOB}} = 0.343$, compared to $0.314$ on seen basins), while parameter vectors transferred to held-out basins remain tightly aligned with their seen-dPL counterparts (median parameter shift of $0.100$). Furthermore, the performance–parameter bridge is preserved out-of-bag, with catchment-level $|\Delta\text{KGE}_{\text{OOB}}|$ positively correlated with parameter displacement across all eight models (median Spearman $\rho = +0.242$)."*
- **Scope Boundary:**
  > *"R4 directly challenges and confirms the primary parameter-separation and performance-bridge findings of R2, but does not independently retest secondary parameter-geometry properties (such as rank reorganization and coordinate localization), which remain scoped to the 36-model seen-basin benchmark."*
