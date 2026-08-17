# Flex-MOPEX Research Worklog & Decision Chain (R1–R19)

## Overview
This document logs the core hypotheses, decisive diagnostic evidence, and key architectural pivots from the initial discovery of structural gate collapse to the final frozen counterfactual structural learning method (R19).

---

## Chronological Decision Chain

### Rounds 1–7: Problem Discovery & Baseline Formulation
- **Observation**: In standard differentiable physics-neural models, canopy interception gate $w_{\text{int}}$ rapidly collapses to 0 within 1–2 epochs across all 671 CAMELS basins, despite positive forest cover and observational necessity.
- **Formula screening (R1–R5)**: Evaluated continuous physical canopy formulations. **Candidate E-S0** (smooth continuous canopy reservoir with explicit evaporation & drainage kinetics) was selected for superior numerical stability and smooth gradients.
- **Oracle benchmark (R6–R7)**: Grid search over process weights established ground-truth $w^*$ and demonstrated that interception improves fit in 18.5% of basins (124/671), proving the collapse is an optimization failure, not a physical impossibility.

### Round 8: AIC Delay Mitigation
- **Hypothesis**: Immediate AIC penalty at epoch 0 penalizes parameter complexity before the neural network learns meaningful hydrologic parameters.
- **Intervention**: `gate_aic_delay_epochs=2` (masking AIC gradients to gates during early warm-up).
- **Result**: Slowed early collapse but $w_{\text{int}}$ still collapsed by epoch 10; established that AIC penalty accelerates, but does not cause, the collapse.

### Round 9: Gradient Separability Diagnostic
- **Finding**: Evaluated per-basin gradients on uncollapsed checkpoints. Local gradient $\nabla_{w_{\text{int}}} L$ is strongly positive in Oracle-positive basins, but negatively correlated with population aggregate ($\cos(\text{pos}, \text{full}) = -0.826$). The signal is lost during batch/population aggregation.

### Round 10: Sensitivity Reweighting
- **Intervention**: Reweighted gate activations by localized gradient sensitivity.
- **Result**: Achieved first positive interception detection (10.8% Oracle recall, 62.5% precision), but overall gate variance remained heavily damped.

### Rounds 11–12: Direction-Balancing Preflight Failures
- **Hypothesis**: Re-balancing gradient signs across positive and negative basins.
- **Finding**: Preflight failed; symmetric noise in 81.5% zero-signal basins generated artificial reverse-sign drift. **Rule established**: strict preflight gates prevent unpromising training runs.

### Round 13: Systematic Root-Cause Diagnosis
- **Hypotheses Tested**:
  - *H-A (Representation Interference)*: Excluded. Probe on backbone $h_{128}$ predicted sensitivity with ROC-AUC = 0.787.
  - *H-B (Parameter Compensation)*: Secondary. Soil parameters compensated by ~28% but positive benefit remained in 83.5% of basins.
  - *H-C (Common-Mode / Number Bias)*: **CONFIRMED PRIMARY CAUSE**. The 568:103 zero-to-positive basin imbalance causes population gradients to point almost entirely in the OFF direction ($\cos(\text{pos}, \text{zero}) = -0.929$).
  - *H-D (Initialization Bias)*: Excluded. Early initialization gradient was coherent ($\cos = +0.921$).

### Round 14: Counterfactual Structural Target Feasibility
- **Concept**: Generate online empirical counterfactual evidence $\Delta J_{b,p} = J_{\text{OFF}} - J_{\text{ON}}$ by evaluating model states with process forced ON vs OFF.
- **Validation**: $\Delta J$ achieved **100% precision and 0.0% false positive rate** against Oracle ground-truth. Candidate C soft-target $q = \sigma(\Delta J / T)$ with robust median scale $T$ achieved ROC-AUC = 0.984 with negligible runtime overhead (11.3s / epoch).
- **Decision**: Feasible; proceed to formal counterfactual training.

### Round 15: Counterfactual Structural Supervision
- **Implementation**: `CFTrainer` with `L_CF` (BCE loss against $q$) and gradient isolation (blocking fit/AIC gradients to gates, and $L_{\text{CF}}$ to backbone).
- **Result**: Interception collapse completely eliminated (mean $w_{\text{int}} = 0.284$). However, gate learned near-constant values across all basins (std = 0.032, Spearman $\rho = +0.108$).

### Rounds 16 & 16.5: Diagnostic Reconciliation of Constant Gates
- **Phase 1–5 Diagnostic (R16)**: Discovered that 73.6% of basins sit in an ambiguous middle ($0.2 \le q \le 0.6$), pulling BCE gradients strongly toward population prevalence ($q \approx 0.30$).
- **Reconciliation Audit (R16.5)**: Resolved metric discrepancies. Showed that Adadelta optimizer stalled on linear `weights_head` (achieving only 5.26% of potential BCE reduction).
- **Decision**: Introduce dedicated Adam optimizer for structure head (R17-A) and confidence-weighted BCE loss (R17-B).

### Rounds 17-A & 17-B: Optimizer & Confidence Weighting
- **R17-A (Dual Optimizer)**: Adam optimizer increased weight norm (std 1.75×), improving snow ($\rho = +0.7838$) and median NSE (0.6429), but $w_{\text{int}}$ remained flat.
- **R17-B (Confidence Weighting)**: Bounded confidence weighting $c = 2|q - 0.5|$ down-weighted ambiguous middle basins. Snow and phenology separated, but $w_{\text{int}}$ still showed $\Delta \approx 0$.
- **Key Insight**: *Shared-Backbone Representation Entanglement*. The shared $h_{128}$ representation, shaped by hydrologic runoff loss, lacks interception-aligned orthogonal sub-spaces. Linear heads on frozen $h_{128}$ cannot separate micro-canopy signals.

### Round 18: Hybrid Dedicated Structure Encoder
- **Architecture**: `LearnedStructureNetHybridEncoder` feeding $[x_{35}, \text{stop\_gradient}(h_{128})] \to 163 \to 128 \to 64 \to 8$ MLP directly into structural logits.
- **Result**: **Decisive breakthrough**:
  - $w_{\text{int}}$ achieved positive polarization for the first time: $\Delta = +0.0582$ ($\bar{w}_{\text{pos}} = 0.3248$ vs $\bar{w}_{\text{zero}} = 0.2666$, Spearman $\rho = +0.1264$).
  - Phenology surged: $\Delta = +0.3049$, Spearman $\rho = +0.5740$.
  - Snow maintained: $\Delta = +0.4338$, Spearman $\rho = +0.7638$.
  - Predictive accuracy: All-time project record median NSE = **0.6470** (peak 0.6511).

### Round 19: Unified Adadelta Simplification & Multi-Seed Replication
- **Hypothesis**: The deep 3-layer nonlinear structure encoder generates sufficient gradient signal to eliminate the need for a separate Adam optimizer, enabling a single unified Adadelta optimizer.
- **Replication (Seeds 42, 43, 44)**:
  - Unified Adadelta not only simplified the system but **substantially widened structural separation**:
    - $w_{\text{int}}$ mean $\Delta = \mathbf{+0.1345 \pm 0.0036}$ (2.3× wider than R18 dual-optimizer), mean Spearman $\rho = \mathbf{+0.3277 \pm 0.0165}$.
    - $w_{\text{snow}}$ mean $\Delta = \mathbf{+0.4721 \pm 0.0148}$, mean Spearman $\rho = \mathbf{+0.7965 \pm 0.0088}$.
    - $w_{\text{phen}}$ mean $\Delta = \mathbf{+0.2893 \pm 0.0082}$, mean Spearman $\rho = \mathbf{+0.5800 \pm 0.0233}$.
    - $w_{\text{sub}}$ mean $\Delta = \mathbf{+0.2204 \pm 0.0148}$, mean Spearman $\rho = \mathbf{+0.4167 \pm 0.0196}$.
  - Median NSE improved to **0.6502 ± 0.0012** across seeds (peak 0.6518 on Seed 42).
- **Final Decision**: `FREEZE_UNIFIED_ADADELTA`.
