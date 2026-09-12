# R4 Methods and Results Manuscript Handoff

This document specifies the exact definitions, computation procedures, statistical tests, numerical results, and permissible scientific interpretations for the R4 out-of-bag (OOB) challenge of R1, R2, and R3.

---

## 1. R1 Replay: Outlet Performance Response

### Estimand 1.1: Out-of-Bag Outlet Performance Comparison
- **Definition:** Model-level median KGE difference between uncalibrated regional dPL evaluated on held-out basins and locally calibrated IC:
  $$\Delta\text{KGE}_{\text{OOB}}(m, b) = \text{KGE}_{\text{OOB-dPL}}(m, b) - \text{KGE}_{\text{IC}}(m, b)$$
- **Computation:** 5-fold cross-validation across 531 basins for each of the 8 prespecified models ($N=4,248$ model–basin instances). Target basins are strictly held out from training and normalization fitting.
- **Uncertainty / Statistics:** Model-equal median $\Delta\text{KGE}_{\text{OOB}} = -0.065384$ (IQR: $[-0.0804, -0.0570]$, range: $[-0.0879, -0.0321]$). Regionalization shift relative to seen dPL is $-0.057526$.
- **Results:**
  - IC Median KGE (8 models): **0.649873**
  - Seen-dPL Median KGE (8 models): **0.648533**
  - OOB-dPL Median KGE (8 models): **0.581336**
- **Allowed Interpretation:** Regional out-of-bag parameter learning incurs a modest, expected performance degradation ($\approx 0.057$ KGE points) when transferring to ungauged catchments without site calibration, but maintains robust baseline hydrological function ($K_{\text{OOB}} \ge 0.53$ across all 8 models).

### Estimand 1.2: Place-Specific Heterogeneity and Variance Decomposition
- **Definition:** Proportion of catchments where uncalibrated OOB-dPL outperforms locally calibrated IC, and two-way ANOVA decomposition of $\Delta\text{KGE}_{\text{OOB}}$ total sum of squares into model-additive, basin-additive, and remainder components.
- **Computation:**
  $$\text{SS}_{\text{total}} = \text{SS}_{\text{model}} + \text{SS}_{\text{basin}} + \text{SS}_{\text{remainder}}$$
- **Uncertainty / Statistics:** Pooled 4,248-instance $\Delta\text{KGE}_{\text{OOB}}$ IQR is $[-0.1745, +0.0035]$.
- **Results:**
  - OOB-dPL outperforms IC in **26.13%** of all catchment instances (ranging from 18.27% in `us1` to 38.98% in `newzealand2`).
  - Variance Shares: **Model Additive = 0.53%**, **Basin Additive = 67.14%**, **Remainder = 32.33%**.
- **Allowed Interpretation:** Confirms R1's fundamental insight: aggregate model-level performance parity or gaps obscure extreme place-specific heterogeneity, which is overwhelmingly governed by catchment attributes rather than model structure.

---

## 2. R2 Replay: Parameter Realization Response

### Estimand 2.1: Out-of-Bag Parameter Realization Displacement ($D_{\text{RMS}}$)
- **Definition:** Bounds-normalized root-mean-square parameter displacement between locally calibrated IC and regional out-of-bag dPL:
  $$D_{\text{RMS, OOB}}(m, b) = \sqrt{\frac{1}{P_m} \sum_{p=1}^{P_m} \left(\tilde{\theta}_{\text{IC}}(m, b, p) - \tilde{\theta}_{\text{OOB-dPL}}(m, b, p)\right)^2}$$
- **Computation:** Evaluated across all 531 basins for each model using physical-bound normalization $\tilde{\theta} \in [0, 1]$.
- **Uncertainty / Statistics:** Model-equal median $D_{\text{RMS, OOB}} = 0.342886$ (IQR: $[0.3146, 0.3923]$, range: $[0.2098, 0.4517]$).
- **Results:**
  - $D_{\text{RMS, seen}} = 0.313853$ (8 models) vs. $D_{\text{RMS, OOB}} = 0.342886$ (paired difference: $+0.029033$).
  - Median displacement between seen-dPL and OOB-dPL parameters is only **$0.100356$**.
- **Allowed Interpretation:** Regional dPL parameters in uncalibrated held-out catchments remain firmly in the distinct, separated realization regime discovered in R2. The separation is not a seen-basin calibration artifact.

### Estimand 2.2: Performance–Parameter Bridge
- **Definition:** Within-model Spearman rank correlation across 531 basins between absolute performance gap and parameter displacement:
  $$\rho_m^{\text{OOB}} = \text{Spearman}_b\left(|\Delta\text{KGE}_{\text{OOB}}(m, b)|, D_{\text{RMS, OOB}}(m, b)\right)$$
- **Computation:** Evaluated per model across all 531 basins.
- **Uncertainty / Statistics:** Statistically significant ($p < 0.05$) in 7/8 models; positive across all 8 models.
- **Results:** Model-equal median $\rho_b = \mathbf{+0.242456}$ (range: $[+0.0436, +0.4412]$), replicating seen-basin $\rho_b = +0.210350$.
- **Allowed Interpretation:** Confirms the R2 bridge mechanism: catchments where regional dPL deviates most in performance from IC are also those where parameter vectors undergo the greatest physical displacement.

---

## 3. R3 Replay: Catchment Information Organization

### Estimand 3.1: Same-Parameter Profile Correspondence ($R_{\text{paired}}$)
- **Definition:** Spearman rank correlation across 13 cluster dimensions (or 32 continuous attributes) between the IC and OOB-dPL parameter–attribute association vectors:
  $$R_{\text{paired, OOB}}(m, p) = \text{Spearman}_a\left(\rho_{\text{IC}}(m, p, \cdot), \rho_{\text{OOB-dPL}}(m, p, \cdot)\right)$$
- **Computation:** Computed for each parameter coordinate, aggregated to model median, then to 8-model equal-weight median.
- **Uncertainty / Statistics:** Model-equal median $R_{\text{paired, OOB}} = \mathbf{0.739011}$ (13 cluster dimensions) and **0.762005** (32 continuous attributes). Range: $[0.4890, 0.8242]$.
- **Results:** Directly matches seen-basin benchmark ($R_{\text{paired, seen}} = 0.712912$ / $0.744685$; Full36 seen = $0.715789$).
- **Allowed Interpretation:** Direct cross-paradigm challenge confirms that conceptual parameters under shared parameter learning preserve substantial catchment-information profiles in uncalibrated held-out catchments.

### Estimand 3.2: Same-Coordinate Specificity ($A_{\text{diag}}$) and Permutation Test
- **Definition:** Diagonal advantage of same-coordinate profile correspondence over within-model off-diagonal alternatives:
  $$A_{\text{diag, OOB}} = \text{median}(\text{diag}(C)) - \text{median}(\text{offdiag}(C))$$
- **Computation:** Full $P_m \times P_m$ within-model cross-profile matrix $C[p, q]$. Significance assessed via 1,000 parameter-label permutations.
- **Uncertainty / Statistics:** Exact within-model permutation **$p = 0.000999$** ($p < 0.001$).
- **Results:**
  - Diagonal Median: **0.739011**
  - Off-Diagonal Median: **-0.010989**
  - Diagonal Advantage $A_{\text{diag, OOB}}$: **0.718407** (13 clusters) / **0.729106** (32 attributes).
- **Allowed Interpretation:** Proves that information retention is strictly coordinate-specific and not a diffuse neural-mapping artifact.

### Estimand 3.3: Supporting Relationship Retention
- **Definition:** Correlation and sign retention between seen-dPL and OOB-dPL parameter–attribute relationship coefficients.
- **Results:** Overall Spearman $\rho = \mathbf{0.9616}$ (median $0.9574$), non-trivial sign retention = **99.93%**, information cluster retention = **96.44%**.
- **Allowed Interpretation:** Supporting check confirming that the learned neural mapping is numerically stable and continuous across data folds. (Must not be cited as independent proof of physical realism).

---

## 4. Scope Boundaries and Un-Tested Sub-Claims

1. **Explicitly Un-Tested R2 Sub-Claims:**
   - Cross-catchment rank reorganization ($R_{\text{rank}}$)
   - Coordinate localization ($C_{\text{eff}}$, top-1, top-2 shares)
   - Parameter contraction boundary
   *These remain valid seen-basin benchmark properties but were not prespecified for R4 OOB re-computation.*
2. **Explicitly Maintained R3 Boundaries:**
   - Functional-role extensions remain unsupported ($A_{\text{role}} = -0.1188, p = 0.8057$). Information organization is coordinate-specific only.
   - Profile correspondence does not establish physical parameter correctness, identifiability, or causality.
3. **Generalization Scope:**
   - Findings apply strictly to the prespecified eight-model panel and must not be extrapolated as an unbiased 36-model population census.
