# R4 Rejected Interpretations and Scientific Boundaries

This document catalogues scientific claims and over-generalizations that are **strictly rejected** by the R4 held-out-basin audit. These boundary conditions must be enforced across all manuscript text, figures, captions, and reviewer responses.

---

## 1. Rejected Claim: "R4 proves that R1–R3 findings generalize to all 36 models"
- **Why Rejected:** R4 executed 5-fold OOB on a prespecified eight-model subset ($N=8$), chosen for architectural contrast and performance viability. It did not run OOB for the remaining 28 models.
- **Enforced Boundary:** R4 provides a targeted architectural stress test demonstrating that key R1–R3 mechanisms persist in held-out basins across diverse conceptual structures. It is not an exhaustive 36-model population census.

---

## 2. Rejected Claim: "The eight tested models form a statistically representative sample of the benchmark"
- **Why Rejected:** The eight models were chosen via deterministic 4-quadrant stratification and 5D contrast maximization ($G \times R \times P \times D_{\theta} \times U$), accompanied by a viability gate ($K_{\text{joint}} \ge 0.561$). They do not constitute an independent, identically distributed (IID) probability sample.
- **Enforced Boundary:** Must use exact phrasing: *"a prespecified eight-model subset spanning contrasting conceptual model formulations"* or *"prespecified stress-test panel"*. The word *"representative"* is strictly forbidden for population inference.

---

## 3. Rejected Claim: "The high seen $\leftrightarrow$ OOB correlation ($\rho \approx 0.9616$) proves external generalization"
- **Why Rejected:** Seen-dPL and OOB-dPL share the same neural parameterizer architecture and 80% overlapping training basins across folds. High correlation between them demonstrates mathematical stability and parameterizer continuity, not independent physical validation.
- **Enforced Boundary:** The primary empirical challenge in R4 is strictly the direct comparison between **IC and OOB-dPL** ($R_{\text{paired}} = 0.739, A_{\text{diag}} = 0.718, p = 0.000999$). The seen $\leftrightarrow$ OOB relationship correlation ($0.9616$) is strictly an auxiliary continuity check.

---

## 4. Rejected Claim: "Held-out profile retention proves that dPL recovers true physical parameter meaning"
- **Why Rejected:** Profile correspondence ($R_{\text{paired}} \approx 0.74$) shows that conceptual parameters in uncalibrated basins correlate with catchment attributes in a manner consistent with local calibration. However, conceptual parameters are lumped effective values, and correspondence does not imply that parameters represent unmodeled in-situ physical quantities.
- **Enforced Boundary:** Replaying R3 out-of-bag demonstrates *catchment-information organization*, not physical truth or parameter identifiability.

---

## 5. Rejected Claim: "R4 proves that dPL parameter realizations are superior or more correct than IC"
- **Why Rejected:** IC is an unconstrained basin-by-basin calibration, whereas dPL is a shared attribute-constrained mapping. Neither estimator constitutes ground truth.
- **Enforced Boundary:** R4 evaluates realization separation ($D_{\text{RMS}}$) and information organization across estimation paradigms. It makes no normative claims regarding estimator superiority.

---

## 6. Rejected Claim: "R4 independently retested and generalized all R2 parameter-geometry findings"
- **Why Rejected:** Secondary R2 geometric estimands—specifically cross-catchment rank reorganization ($R_{\text{rank}}$), coordinate localization ($C_{\text{eff}}$), and the contraction boundary—were not prespecified in the R4 protocol and were not recomputed on OOB outputs.
- **Enforced Boundary:** R4 supports the core parameter-separation finding ($D_{\text{RMS, OOB}} = 0.343$) and the performance bridge ($\rho_b = +0.242$), but secondary geometric properties remain scoped to the 36-model seen-basin benchmark and are explicitly marked as *`NOT TESTED BY R4`*.

---

## 7. Rejected Claim: "Held-out relationship persistence implies causal catchment-process linkages"
- **Why Rejected:** Spearman correlations between static catchment attributes and conceptual parameters reflect observational associations across the CAMELS domain. They do not demonstrate causal hydro-climatic mechanisms.
- **Enforced Boundary:** Relationships must be described observationally as statistical association profiles, not causal drivers.

---

## 8. Rejected Claim: "OOB attribute relationships are entirely non-constructive"
- **Why Rejected:** By mathematical design, dPL formulates parameter vectors as a direct functional mapping of catchment attributes ($\theta = g(X)$).
- **Enforced Boundary:** The manuscript explicitly acknowledges the constructive nature of dPL parameterization while highlighting that *same-coordinate specificity against independently calibrated IC* ($A_{\text{diag}} = 0.718, p = 0.000999$) is an empirical cross-paradigm result that cannot be produced by the neural network alone.
