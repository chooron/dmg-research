# MARRMoT Performance Comparison: Provenance and Statistical Analysis

**Audit Target**: Empirical comparison between DMOT Individual Calibration (IC Full300 CMA-ES) and archived MARRMoT reference calibrations  
**Provenance Files**:
- Model Summary: `/mnt/g/Dataset/dmotpy_dpl_ic_training_20260819_final/results/remote_comparison_20260818/ic/full300_kge_model_summary.csv`
- Basin Records: `/mnt/g/Dataset/dmotpy_dpl_ic_training_20260819_final/results/remote_comparison_20260818/ic/full300_kge_by_basin.csv`
- Extracted Comparison Matrix: `project/benchmark/manuscript/si/marrmot_performance_comparison.csv`

---

## 1. Study Alignment & Scope

| Dimension | Current DMOT IC Study | Archived MARRMoT Reference | Alignment Status |
| :--- | :--- | :--- | :---: |
| **Model Portfolio** | 36 conceptual models | 36 conceptual models | **Identical (36/36)** |
| **Target Basins** | 531 CAMELS-US catchments | 480–483 CAMELS-US catchments | **Overlapping ($N_{\text{pairs}}=17,380$)** |
| **Calibration Period** | `1980-10-01` to `1995-09-30` (15 years) | `1980-10-01` to `1995-09-30` (15 years) | **Identical** |
| **Evaluation Period** | `1995-10-01` to `2010-09-30` (15 years) | `1995-10-01` to `2010-09-30` (15 years) | **Identical** |
| **Forcing Inputs** | Daymet ($P, T, \text{PET}_{\text{Hargreaves}}$) | Daymet ($P, T, \text{PET}_{\text{Hargreaves}}$) | **Identical** |
| **Objective Function** | Kling-Gupta Efficiency ($\text{KGE}$) | Kling-Gupta Efficiency ($\text{KGE}$) | **Identical** |
| **Numerical Scheme** | PyTorch autograd discrete Euler | MATLAB continuous ODE solver | *Independent Calibrations* |

---

## 2. Statistical Findings

### 2.1 Model-Level Performance Concordance
Across the 36 conceptual hydrological models, median outlet performance in the reconstructed differentiable framework strongly tracks median performance in the reference MARRMoT suite:
- **Evaluation Period (1995–2010)**:
  - Spearman rank correlation across 36 model median KGEs: $\mathbf{\rho = 0.7079}$ ($p = 1.38 \times 10^{-6}$).
  - Model-level median KGE difference ($\text{KGE}_{\text{DMOT}} - \text{KGE}_{\text{MARRMoT}}$): Median $= \mathbf{+0.0324}$, $\text{IQR} = \mathbf{0.0575}$ ($\text{Q25} = +0.0094, \text{Q75} = +0.0669$).
- **Calibration Period (1980–1995)**:
  - Spearman rank correlation across 36 model median KGEs: $\mathbf{\rho = 0.7385}$ ($p = 2.73 \times 10^{-7}$).
  - Model-level median KGE difference: Median $= \mathbf{+0.0329}$, $\text{IQR} = \mathbf{0.0650}$ ($\text{Q25} = +0.0150, \text{Q75} = +0.0800$).

### 2.2 Catchment-Level Paired Distribution
At the individual catchment level across all overlapping model–basin instances ($N = 17,380$ evaluation pairs):
- **Evaluation Period**:
  - Median paired difference: $\Delta\text{KGE} = \mathbf{+0.0274}$ ($\text{IQR} = \mathbf{0.1104}$, $\text{Q25} = -0.0137, \text{Q75} = +0.0967$).
  - DMOT IC achieves higher or equal evaluation KGE in **69.57%** of model–basin instances (model-equal median win rate).
- **Calibration Period**:
  - Median paired difference: $\Delta\text{KGE} = \mathbf{+0.0269}$ ($\text{IQR} = \mathbf{0.0869}$, $\text{Q25} = -0.0041, \text{Q75} = +0.0827$).
  - DMOT IC achieves higher or equal calibration KGE in **77.74%** of model–basin instances (model-equal median win rate).

---

## 3. Scientific Interpretation & Permitted Wording

### Authorized Manuscript/SI Framing (Level B):
- *"When calibrated independently across identical 15-year historical parameter-estimation and evaluation periods, the 36 differentiable model reconstructions exhibit performance ranks and median efficiencies that strongly align with reference MARRMoT implementations across overlapping CAMELS catchments (model-level Spearman rank $\rho = 0.708$, $p = 1.38 \times 10^{-6}$; median paired evaluation $\Delta\text{KGE} = +0.027$)."*
- *"This performance-context comparison confirms that the differentiable formulations preserve the relative structural capabilities and hydrologic efficacy of the parent MARRMoT ensemble under standard operational conditions."*

### Prohibited Claims:
- Do **NOT** claim "exact parameter-level numerical identity" or "solver equivalence".
- Do **NOT** claim that parameters calibrated in MATLAB MARRMoT can be directly transferred without solver-induced drift.
